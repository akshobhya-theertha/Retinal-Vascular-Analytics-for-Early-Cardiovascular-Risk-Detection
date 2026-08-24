"""
Retinexia - Image Preprocessing for Retinal Fundus Images
=========================================================
Handles resize to 224x224, normalization, and data augmentation.
Dataset is Gaussian-filtered for improved vessel clarity.
"""

import numpy as np
from pathlib import Path
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Target size for MobileNetV2 (ImageNet input size)
IMG_SIZE = (224, 224)
# Diabetic Retinopathy stages (0-4)
DR_CLASSES = ["Normal", "Mild", "Moderate", "Severe", "Proliferative"]


def get_train_datagen(rotation=15, zoom=0.2, h_flip=True):
    """
    Training data generator with augmentation.
    Rotation, zoom, and horizontal flip help generalize across
    different camera angles and patient positioning.
    """
    return ImageDataGenerator(
        rescale=1.0 / 255.0,  # Normalize pixel values to [0, 1]
        rotation_range=rotation,
        zoom_range=zoom,
        horizontal_flip=h_flip,
        fill_mode="nearest",
        validation_split=0.2,
    )


def get_inference_datagen():
    """No augmentation for validation/test or inference."""
    return ImageDataGenerator(rescale=1.0 / 255.0)


def load_data_generators(
    data_dir,
    batch_size=32,
    img_size=IMG_SIZE,
    seed=42,
):
    """
    Create train and validation generators from directory structure.
    Expects: data_dir/class_name/*.png (e.g. train/0/, train/1/, ...)
    """
    train_datagen = get_train_datagen()
    data_path = Path(data_dir)

    train_gen = train_datagen.flow_from_directory(
        data_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        subset="training",
        seed=seed,
        shuffle=True,
    )
    val_gen = train_datagen.flow_from_directory(
        data_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation",
        seed=seed,
        shuffle=False,
    )
    return train_gen, val_gen, train_gen.class_indices


def preprocess_single_image(img_array):
    """
    Preprocess a single image for prediction: resize to 224x224 and normalize [0,1].
    Matches training pipeline for correct detection.
    img_array: numpy array (H, W) or (H, W, 3), values 0-255.
    Returns (1, 224, 224, 3) float32.
    """
    from PIL import Image as PILImage
    import tensorflow as tf

    if isinstance(img_array, np.ndarray):
        if img_array.ndim == 2:
            img_array = np.stack([img_array] * 3, axis=-1)
        if img_array.dtype != np.uint8:
            img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        img = PILImage.fromarray(img_array).convert("RGB")
    else:
        img = img_array
    img = img.resize((IMG_SIZE[1], IMG_SIZE[0]), PILImage.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr
