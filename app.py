from flask import Flask, render_template, request
import cv2
import numpy as np
import joblib

from skimage.feature import hog

import tensorflow as tf
from keras.layers import TFSMLayer


# ======================================================
# App initialization
# ======================================================
app = Flask(__name__)


# ======================================================
# Load models ONCE at startup
# ======================================================
svm_model = joblib.load("model/fire_svm_model.pkl")

# Keras 3 compatible CNN loading (SavedModel inference)
cnn_model = TFSMLayer(
    "model",                  # folder containing saved_model.pb
    call_endpoint="serving_default"
)


# ======================================================
# SVM feature extraction
# ======================================================
def extract_svm_features(img):
    img = cv2.resize(img, (128, 128))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )
    return features.reshape(1, -1)


# ======================================================
# CNN preprocessing
# ======================================================
def preprocess_cnn_image(img):
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img


# ======================================================
# Routes
# ======================================================
@app.route("/", methods=["GET", "POST"])
def index():

    # -------- Single model outputs --------
    result = None
    confidence = None
    safety_msg = None
    selected_model = None

    # -------- Comparison outputs --------
    comparison = False
    svm_result = svm_confidence = None
    cnn_result = cnn_confidence = None

    threshold = 0.182   # Optimized threshold from PR curve

    if request.method == "POST":

        file = request.files["image"]
        selected_model = request.form.get("model_type")

        # Decode image
        img = cv2.imdecode(
            np.frombuffer(file.read(), np.uint8),
            cv2.IMREAD_COLOR
        )

        # ==================================================
        # SVM ONLY
        # ==================================================
        if selected_model == "svm":

            features = extract_svm_features(img)
            pred = svm_model.predict(features)[0]
            prob = svm_model.predict_proba(features).max()

            result = "Fire Detected" if pred == 0 else "No Fire Detected"
            safety_msg = (
                "⚠️ Possible forest fire detected. Inform authorities immediately."
                if pred == 0 else
                "🌲 Forest appears safe. No fire signs detected."
            )
            confidence = round(prob * 100, 2)

        # ==================================================
        # CNN ONLY
        # ==================================================
        elif selected_model == "cnn":

            img_cnn = preprocess_cnn_image(img)
            outputs = cnn_model(img_cnn)

            # Extract tensor from dict (Keras 3)
            cnn_prob = float(list(outputs.values())[0].numpy())

            result = "Fire Detected" if cnn_prob > threshold else "No Fire Detected"
            safety_msg = (
                "⚠️ Possible forest fire detected. Inform authorities immediately."
                if cnn_prob > threshold else
                "🌲 Forest appears safe. No fire signs detected."
            )
            confidence = round(cnn_prob * 100, 2)

        # ==================================================
        # DYNAMIC COMPARISON (SVM + CNN)
        # ==================================================
        elif selected_model == "compare":

            comparison = True

            # ----- SVM -----
            svm_features = extract_svm_features(img)
            svm_pred = svm_model.predict(svm_features)[0]
            svm_prob = svm_model.predict_proba(svm_features).max()

            svm_result = "Fire Detected" if svm_pred == 0 else "No Fire Detected"
            svm_confidence = round(svm_prob * 100, 2)

            # ----- CNN -----
            img_cnn = preprocess_cnn_image(img)
            outputs = cnn_model(img_cnn)
            cnn_prob = float(list(outputs.values())[0].numpy())

            cnn_result = "Fire Detected" if cnn_prob > threshold else "No Fire Detected"
            cnn_confidence = round(cnn_prob * 100, 2)

    # ======================================================
    # Render template
    # ======================================================
    return render_template(
        "index.html",

        # Single model
        result=result,
        confidence=confidence,
        safety_msg=safety_msg,
        selected_model=selected_model,

        # Comparison
        comparison=comparison,
        svm_result=svm_result,
        svm_confidence=svm_confidence,
        cnn_result=cnn_result,
        cnn_confidence=cnn_confidence,
        threshold=threshold
    )


# ======================================================
# Run app
# ======================================================
if __name__ == "__main__":
    app.run(debug=True)
