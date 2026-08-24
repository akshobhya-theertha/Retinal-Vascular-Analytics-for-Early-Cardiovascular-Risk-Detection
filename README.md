# Retinexia: Retinal Vascular Analytics for Early Cardiovascular Risk Detection

AI-based medical system that analyzes retinal fundus images to detect **Diabetic Retinopathy (DR) stage** and **Cardiovascular Disease (CVD) risk**, with **Explainable AI (Grad-CAM)** and a **Flask web UI**.

## Project overview

- **DR stage**: Normal / Mild / Moderate / Severe / Proliferative (5 classes).
- **CVD risk**: Low / Moderate / High (derived from DR severity).
- **Model**: MobileNetV2 (ImageNet) + transfer learning, two-phase training.
- **XAI**: Grad-CAM heatmaps overlaid on the fundus image.

## Dataset (Kaggle API only)

- **Name**: [Diabetic Retinopathy 224×224 (Gaussian Filtered)](https://www.kaggle.com/datasets/sovitrath/diabetic-retinopathy-224x224-gaussian-filtered)
- Do **not** download manually. Use the Kaggle API and environment variable:

```bash
# Option 1: API token (kaggle.json or env)
set KAGGLE_USERNAME=your_username
set KAGGLE_KEY=your_key

# Option 2: or place kaggle.json in ~/.kaggle/ (Windows: C:\Users\<you>\.kaggle\)
```

Then:

```bash
pip install -r requirements.txt
python download_dataset.py
python prepare_data.py
```

Training expects images under `dataset/train/0`, `dataset/train/1`, … `dataset/train/4` (one folder per DR class). `prepare_data.py` tries to organize the unzipped Kaggle data into this layout.

## Model architecture

- **Base**: MobileNetV2, pretrained on ImageNet, input 224×224.
- **Heads**: (1) DR stage (5-way softmax), (2) CVD risk (3-way softmax).
- **Training**:  
  - Phase 1: freeze base, train heads (feature extraction).  
  - Phase 2: unfreeze top layers, lower learning rate (fine-tuning).  
- **Class weights** for DR to handle imbalance; CVD labels are derived from DR (0,1→Low, 2→Moderate, 3,4→High).

## Grad-CAM (Explainable AI)

- Uses the last convolutional layer of the model.
- Gradients of the predicted class score w.r.t. feature maps are computed, then combined and resized to the image size.
- The result is overlaid on the fundus image to show regions (e.g. vessels, lesions) that influenced the prediction.

## Run training

```bash
# Full training (dataset path default: dataset/train)
python train.py --data_dir dataset/train --epochs_phase1 15 --epochs_phase2 20 --batch_size 32
```

Outputs:

- `models/retinexia_best.keras` (best validation accuracy)
- `models/retinexia_final.keras`
- `models/class_indices.json`
- `models/confusion_matrices.png`, `models/roc_curve.png`, `models/evaluation_metrics.json`

## Run Flask app – one link

**Open in browser:** **http://localhost:5000/**

1. Start the server:
   - Double‑click **START_RETINEXIA.bat** (opens browser after 5 sec), or
   - In a terminal: `python app.py 5000`
2. If the browser did not open, go to: **http://localhost:5000/**

See **LINK.txt** in the project folder for the same URL.  
- **Predict**: upload image via the UI or POST to `/predict` with form field `file` or `image`.

The app returns DR stage, CVD risk, confidence scores, and a Grad-CAM heatmap (base64 image).

## Project structure

```
Retinexia/
├── dataset/           # Kaggle data → dataset/train/0..4
├── models/            # Saved model and metrics
├── static/
├── templates/         # index.html
├── app.py             # Flask backend
├── train.py           # Two-phase MobileNetV2 training
├── preprocess.py      # Resize, normalize, augmentation
├── gradcam.py         # Grad-CAM heatmaps
├── download_dataset.py
├── prepare_data.py
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.8+
- TensorFlow 2.10+, Flask, Pillow, scikit-learn, matplotlib, kaggle, etc. (see `requirements.txt`).

## Disclaimer

For research and educational use only. Not a substitute for clinical diagnosis; always rely on qualified healthcare providers.
