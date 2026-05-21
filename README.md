# 🌲 Forest Fire Detection System

A web-based forest fire detection app that lets you upload a forest image and get an instant fire/no-fire prediction — with confidence scores — using your choice of two ML models.

---

## Overview

This project compares two approaches to forest fire detection from images:

- **SVM (Support Vector Machine)** — a classical machine learning model using HOG (Histogram of Oriented Gradients) features extracted from grayscale images.
- **CNN (Convolutional Neural Network)** — a deep learning model that processes full-color images resized to 224×224 and outputs a fire probability score.

Both models are served through a lightweight Flask web app where users can upload an image, select a model, and receive a prediction along with a confidence bar and safety message.

---

## Project Structure

```
forest-fire-detection-updated/
│
├── app.py                  # Flask application (routes, inference logic)
├── requirements.txt        # Python dependencies
│
├── model/
│   ├── fire_svm_model.pkl          # Trained SVM model (joblib)
│   ├── saved_model.pb              # CNN SavedModel definition
│   ├── fingerprint.pb
│   └── variables/
│       ├── variables.data-00000-of-00001
│       └── variables.index
│
├── static/
│   └── style.css           # Frontend styles
│
└── templates/
    └── index.html          # Main UI template (Jinja2)
```

---

## How It Works

### SVM Pipeline
1. Image is resized to **128×128** and converted to grayscale.
2. HOG features are extracted (9 orientations, 16×16 pixels per cell, 2×2 cells per block, L2-Hys normalization).
3. Features are passed to the trained SVM; prediction and probability are returned.

### CNN Pipeline
1. Image is resized to **224×224** and normalized to `[0, 1]`.
2. The SavedModel is loaded via Keras `TFSMLayer` and called at the `serving_default` endpoint.
3. The raw output probability is compared against a tuned threshold of **0.182** (derived from a PR curve) to classify fire vs. no fire.

---

## Setup & Installation

### Prerequisites
- Python 3.8+
- pip

### Install dependencies

```bash
pip install -r requirements.txt
pip install tensorflow keras  # not in requirements.txt but required for CNN
```

> **Note:** `tensorflow` and `keras` are used in `app.py` but are not listed in `requirements.txt`. Add them if you're setting up a fresh environment.

### Run the app

```bash
python app.py
```

Then open your browser at `http://127.0.0.1:5000`.

---

## Usage

1. Open the web app in your browser.
2. Select a detection model — **SVM** or **CNN**.
3. Upload a forest image (JPG, PNG, etc.).
4. Click **Analyze Image**.
5. The app displays:
   - **Result**: Fire Detected / No Fire Detected
   - **Model used**
   - **Confidence** (shown as a percentage + progress bar)
   - **Safety message** with recommended action

---

## Dependencies

| Package | Purpose |
|---|---|
| `flask` | Web framework |
| `numpy` | Array operations |
| `opencv-python` | Image loading and resizing |
| `scikit-image` | HOG feature extraction |
| `scikit-learn` | SVM model |
| `joblib` | Model serialization |
| `tensorflow` / `keras` | CNN inference via TFSMLayer |

---

## Notes

- The SVM model file (`fire_svm_model.pkl`) is ~43MB and the CNN SavedModel is ~19MB — both are loaded once at app startup.
- The CNN threshold (0.182) was selected based on precision-recall analysis and can be tuned in `app.py` if you retrain the model.
- The app runs in debug mode by default (`debug=True`). Disable this for any production deployment.
