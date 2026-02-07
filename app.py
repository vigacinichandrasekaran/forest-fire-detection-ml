from flask import Flask, render_template, request
import cv2
import numpy as np
import joblib
import tensorflow as tf
from skimage.feature import hog
import tensorflow as tf
from keras.layers import TFSMLayer


# -----------------------------
# App initialization
# -----------------------------
app = Flask(__name__)

# -----------------------------
# Load models (ONCE at startup)
# -----------------------------
svm_model = joblib.load("model/fire_svm_model.pkl")
cnn_model = TFSMLayer(
    "model",
    call_endpoint="serving_default"
)




# -----------------------------
# SVM feature extraction
# -----------------------------
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

# -----------------------------
# CNN preprocessing
# -----------------------------
def preprocess_cnn_image(img):
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# -----------------------------
# Routes
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    confidence = None
    safety_msg = None
    selected_model = None

    if request.method == "POST":
        file = request.files["image"]
        selected_model = request.form.get("model_type")

        img = cv2.imdecode(
            np.frombuffer(file.read(), np.uint8),
            cv2.IMREAD_COLOR
        )

        # ---------- SVM ----------
        if selected_model == "svm":
            features = extract_svm_features(img)
            pred = svm_model.predict(features)[0]
            prob = svm_model.predict_proba(features).max()

            if pred == 0:
                result = "Fire Detected"
                safety_msg = "⚠️ Possible forest fire detected. Inform authorities immediately."
            else:
                result = "No Fire Detected"
                safety_msg = "🌲 Forest appears safe. No fire signs detected."

            confidence = round(prob * 100, 2)

        # ---------- CNN ----------
        elif selected_model == "cnn":
            img_cnn = preprocess_cnn_image(img)
            
            outputs = cnn_model(img_cnn)
            prob = float(list(outputs.values())[0].numpy())



            threshold = 0.182  # from your PR curve
            if prob > threshold:
                result = "Fire Detected"
                safety_msg = "⚠️ Possible forest fire detected. Inform authorities immediately."
            else:
                result = "No Fire Detected"
                safety_msg = "🌲 Forest appears safe. No fire signs detected."

            confidence = round(prob * 100, 2)

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        safety_msg=safety_msg,
        selected_model=selected_model
    )

# -----------------------------
# Run app
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
