"""
Retinexia - Model Training with MobileNetV2 Transfer Learning
=============================================================
Two-phase training: (1) Feature extraction with frozen base,
(2) Fine-tuning with top layers unfrozen. Outputs: DR stage + CVD risk.
"""

import json
import math
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
)
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from preprocess import (
    load_data_generators,
    IMG_SIZE,
    DR_CLASSES,
    get_train_datagen,
    get_inference_datagen,
)

# CVD risk levels (derived from DR severity in literature)
CVD_CLASSES = ["Low", "Moderate", "High"]
NUM_DR = len(DR_CLASSES)
NUM_CVD = len(CVD_CLASSES)

# Map DR index -> CVD index: 0,1->Low(0), 2->Moderate(1), 3,4->High(2)
DR_TO_CVD = [0, 0, 1, 2, 2]


def dr_to_cvd_label(dr_index):
    """Convert DR class index to CVD risk class index."""
    return DR_TO_CVD[int(dr_index)]


def build_model(num_dr=NUM_DR, num_cvd=NUM_CVD, input_shape=(224, 224, 3)):
    """
    Build MobileNetV2-based dual-head model.
    - Base: MobileNetV2 pretrained on ImageNet (transfer learning).
    - Head 1: Diabetic Retinopathy stage (5 classes).
    - Head 2: Cardiovascular risk level (3 classes).
    """
    base = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
        pooling="avg",
    )
    base.trainable = False  # Phase 1: freeze

    x = base.output
    # Shared dense layer before branching
    x = layers.Dense(256, activation="relu", name="shared_fc")(x)
    x = layers.Dropout(0.4)(x)

    # DR head
    dr_out = layers.Dense(num_dr, activation="softmax", name="dr_stage")(x)
    # CVD head (from same representation)
    cvd_out = layers.Dense(num_cvd, activation="softmax", name="cvd_risk")(x)

    model = Model(inputs=base.input, outputs=[dr_out, cvd_out], name="Retinexia")
    return model, base


def get_class_weights_from_generator(gen):
    """Compute class weights for imbalanced DR classes."""
    classes = np.array(gen.classes)
    n_classes = len(gen.class_indices)
    counts = np.bincount(classes, minlength=n_classes)
    total = counts.sum()
    # weight inversely proportional to frequency
    weights = total / (n_classes * np.maximum(counts, 1))
    return dict(enumerate(weights))


def prepare_cvd_labels(train_gen, val_gen):
    """
    Derive CVD labels from DR labels for training the CVD head.
    Returns (train_cvd, val_cvd) as lists of class indices.
    """
    train_cvd = [dr_to_cvd_label(c) for c in train_gen.classes]
    val_cvd = [dr_to_cvd_label(c) for c in val_gen.classes]
    return train_cvd, val_cvd


class DualOutputSequence(keras.utils.Sequence):
    """
    Wraps an ImageDataGenerator that yields (x, y_dr) so that we yield
    (x, (y_dr, y_cvd)) for the dual-head model. CVD labels are derived from DR.
    """

    def __init__(self, base_gen, cvd_label_list):
        self.base_gen = base_gen
        self.cvd_list = cvd_label_list

    def __len__(self):
        return len(self.base_gen)

    def __getitem__(self, index):
        x, y_dr = self.base_gen[index]
        start = index * self.base_gen.batch_size
        end = min(start + len(x), len(self.cvd_list))
        batch_cvd = np.array(self.cvd_list[start:end], dtype=np.int32)
        y_cvd = keras.utils.to_categorical(batch_cvd, num_classes=NUM_CVD)
        return x, {"dr_stage": y_dr, "cvd_risk": y_cvd}


def train_phase1(model, train_gen, val_gen, class_weights, epochs=15, lr=1e-3):
    """Phase 1: Feature extraction with frozen base."""
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss={
            "dr_stage": "categorical_crossentropy",
            "cvd_risk": "categorical_crossentropy",
        },
        loss_weights={"dr_stage": 1.0, "cvd_risk": 0.5},
        metrics={
            "dr_stage": ["accuracy"],
            "cvd_risk": ["accuracy"],
        },
    )

    train_cvd = [dr_to_cvd_label(c) for c in train_gen.classes]
    val_cvd = [dr_to_cvd_label(c) for c in val_gen.classes]
    train_wrapper = DualOutputSequence(train_gen, train_cvd)
    val_wrapper = DualOutputSequence(val_gen, val_cvd)
    train_steps = len(train_wrapper)
    val_steps = len(val_wrapper)

    callbacks = [
        ReduceLROnPlateau(
            monitor="val_dr_stage_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_dr_stage_loss",
            mode="min",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
    ]
    out_dir = Path(__file__).resolve().parent / "models"
    out_dir.mkdir(exist_ok=True)
    callbacks.append(
        ModelCheckpoint(
            str(out_dir / "retinexia_phase1_best.keras"),
            monitor="val_dr_stage_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        )
    )

    history = model.fit(
        train_wrapper,
        steps_per_epoch=train_steps,
        epochs=epochs,
        validation_data=val_wrapper,
        validation_steps=val_steps,
        callbacks=callbacks,
    )
    return history


def train_phase2(model, base, train_gen, val_gen, class_weights, epochs=20, lr=1e-5):
    """Phase 2: Fine-tune top layers of base."""
    base.trainable = True
    # Freeze first ~100 layers, fine-tune the rest
    for layer in base.layers[:100]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss={
            "dr_stage": "categorical_crossentropy",
            "cvd_risk": "categorical_crossentropy",
        },
        loss_weights={"dr_stage": 1.0, "cvd_risk": 0.5},
        metrics={
            "dr_stage": ["accuracy"],
            "cvd_risk": ["accuracy"],
        },
    )

    train_cvd = [dr_to_cvd_label(c) for c in train_gen.classes]
    val_cvd = [dr_to_cvd_label(c) for c in val_gen.classes]
    train_wrapper = DualOutputSequence(train_gen, train_cvd)
    val_wrapper = DualOutputSequence(val_gen, val_cvd)
    train_steps = len(train_wrapper)
    val_steps = len(val_wrapper)

    callbacks = [
        ReduceLROnPlateau(
            monitor="val_dr_stage_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_dr_stage_loss",
            mode="min",
            patience=6,
            restore_best_weights=True,
            verbose=1,
        ),
    ]
    out_dir = Path(__file__).resolve().parent / "models"
    out_dir.mkdir(exist_ok=True)
    callbacks.append(
        ModelCheckpoint(
            str(out_dir / "retinexia_best.keras"),
            monitor="val_dr_stage_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        )
    )

    history = model.fit(
        train_wrapper,
        steps_per_epoch=train_steps,
        epochs=epochs,
        validation_data=val_wrapper,
        validation_steps=val_steps,
        callbacks=callbacks,
    )
    return history


def evaluate_model(model, val_gen, save_dir="models"):
    """Compute accuracy, precision, recall, F1, confusion matrix, ROC-AUC."""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    val_cvd = [dr_to_cvd_label(c) for c in val_gen.classes]
    val_wrapper = DualOutputSequence(val_gen, val_cvd)

    # Predict
    dr_preds = []
    cvd_preds = []
    dr_true = []
    cvd_true = []
    for i in range(len(val_wrapper)):
        x, ys = val_wrapper[i]
        y_dr = ys["dr_stage"] if isinstance(ys, dict) else ys[0]
        y_cvd = ys["cvd_risk"] if isinstance(ys, dict) else ys[1]
        out_dr, out_cvd = model.predict(x, verbose=0)
        dr_preds.append(np.argmax(out_dr, axis=1))
        cvd_preds.append(np.argmax(out_cvd, axis=1))
        dr_true.append(np.argmax(y_dr, axis=1))
        cvd_true.append(np.argmax(y_cvd, axis=1))

    dr_preds = np.concatenate(dr_preds)
    cvd_preds = np.concatenate(cvd_preds)
    dr_true = np.concatenate(dr_true)
    cvd_true = np.concatenate(cvd_true)

    # DR metrics
    acc_dr = accuracy_score(dr_true, dr_preds)
    prec_dr, rec_dr, f1_dr, _ = precision_recall_fscore_support(
        dr_true, dr_preds, average="weighted", zero_division=0
    )
    cm_dr = confusion_matrix(dr_true, dr_preds)

    # CVD metrics
    acc_cvd = accuracy_score(cvd_true, cvd_preds)
    prec_cvd, rec_cvd, f1_cvd, _ = precision_recall_fscore_support(
        cvd_true, cvd_preds, average="weighted", zero_division=0
    )
    cm_cvd = confusion_matrix(cvd_true, cvd_preds)

    # ROC-AUC (one-vs-rest for multi-class)
    try:
        dr_proba = model.predict(val_wrapper, verbose=0)[0]
        roc_auc_dr = roc_auc_score(
            keras.utils.to_categorical(dr_true, NUM_DR),
            dr_proba,
            multi_class="ovr",
            average="weighted",
        )
    except Exception:
        roc_auc_dr = 0.0
    try:
        cvd_proba = model.predict(val_wrapper, verbose=0)[1]
        roc_auc_cvd = roc_auc_score(
            keras.utils.to_categorical(cvd_true, NUM_CVD),
            cvd_proba,
            multi_class="ovr",
            average="weighted",
        )
    except Exception:
        roc_auc_cvd = 0.0

    # Plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(
        cm_dr,
        ax=axes[0],
        annot=True,
        fmt="d",
        xticklabels=DR_CLASSES,
        yticklabels=DR_CLASSES,
        cmap="Blues",
    )
    axes[0].set_title("DR Stage Confusion Matrix")
    axes[0].set_ylabel("True")
    axes[0].set_xlabel("Predicted")
    sns.heatmap(
        cm_cvd,
        ax=axes[1],
        annot=True,
        fmt="d",
        xticklabels=CVD_CLASSES,
        yticklabels=CVD_CLASSES,
        cmap="Blues",
    )
    axes[1].set_title("CVD Risk Confusion Matrix")
    axes[1].set_ylabel("True")
    axes[1].set_xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(save_dir / "confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ROC curves (one class for brevity - DR class 1 vs rest)
    try:
        dr_proba = model.predict(val_wrapper, verbose=0)[0]
        fpr, tpr, _ = roc_curve(dr_true == 1, dr_proba[:, 1])
        plt.figure()
        plt.plot(fpr, tpr, label=f"DR (AUC = {roc_auc_dr:.3f})")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve (DR)")
        plt.legend()
        plt.savefig(save_dir / "roc_curve.png", dpi=150, bbox_inches="tight")
        plt.close()
    except Exception:
        pass

    metrics = {
        "dr": {
            "accuracy": float(acc_dr),
            "precision": float(prec_dr),
            "recall": float(rec_dr),
            "f1_score": float(f1_dr),
            "roc_auc": float(roc_auc_dr),
        },
        "cvd": {
            "accuracy": float(acc_cvd),
            "precision": float(prec_cvd),
            "recall": float(rec_cvd),
            "f1_score": float(f1_cvd),
            "roc_auc": float(roc_auc_cvd),
        },
    }
    with open(save_dir / "evaluation_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("DR - Accuracy:", acc_dr, "Precision:", prec_dr, "Recall:", rec_dr, "F1:", f1_dr, "ROC-AUC:", roc_auc_dr)
    print("CVD - Accuracy:", acc_cvd, "Precision:", prec_cvd, "Recall:", rec_cvd, "F1:", f1_cvd, "ROC-AUC:", roc_auc_cvd)
    return metrics


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=None, help="Path to train directory (class subdirs 0..4). Default: dataset/train")
    parser.add_argument("--epochs_phase1", type=int, default=15)
    parser.add_argument("--epochs_phase2", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr_phase1", type=float, default=1e-3)
    parser.add_argument("--lr_phase2", type=float, default=1e-5)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    data_dir = Path(args.data_dir) if args.data_dir else project_root / "dataset" / "train"
    dataset_root = project_root / "dataset"

    # If train folder missing or empty, run prepare_data to use manually downloaded data
    train_0 = data_dir / "0"
    if not data_dir.exists() or not train_0.exists() or not any(train_0.iterdir()):
        print("Preparing dataset from folder (using manually downloaded data)...")
        try:
            from prepare_data import prepare_train_structure
            prepare_train_structure(str(dataset_root))
        except Exception as e:
            print("Prepare data failed:", e)
        data_dir = project_root / "dataset" / "train"
    if not data_dir.exists():
        print("Data directory not found. Put your dataset in the 'dataset' folder (with subfolders 0,1,2,3,4 or No_DR, Mild, etc.) then run again.")
        return
    train_gen, val_gen, class_indices = load_data_generators(
        str(data_dir),
        batch_size=args.batch_size,
        img_size=IMG_SIZE,
    )
    if train_gen.samples == 0:
        print("No images found in", data_dir, "- Check that dataset/train/0, 1, 2, 3, 4 contain images.")
        return
    class_weights = get_class_weights_from_generator(train_gen)
    print("Class indices:", class_indices)
    print("Class weights:", class_weights)

    model, base = build_model()
    model.summary()

    # Phase 1
    print("\n--- Phase 1: Feature extraction ---")
    train_phase1(
        model,
        train_gen,
        val_gen,
        class_weights,
        epochs=args.epochs_phase1,
        lr=args.lr_phase1,
    )

    # Phase 2
    print("\n--- Phase 2: Fine-tuning ---")
    train_phase2(
        model,
        base,
        train_gen,
        val_gen,
        class_weights,
        epochs=args.epochs_phase2,
        lr=args.lr_phase2,
    )

    # Save final model and class indices (use project root)
    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)
    model.save(str(models_dir / "retinexia_final.keras"))
    with open(models_dir / "class_indices.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "dr": class_indices,
                "dr_classes": DR_CLASSES,
                "cvd_classes": CVD_CLASSES,
            },
            f,
            indent=2,
        )

    # Evaluation
    print("\n--- Evaluation ---")
    evaluate_model(model, val_gen, save_dir=str(models_dir))


if __name__ == "__main__":
    main()
