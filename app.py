from flask import Flask, render_template, request
import cv2
import numpy as np
import joblib
from skimage.feature import hog

app = Flask(__name__)
model = joblib.load("model/fire_svm_model.pkl")

def extract_features_from_image(img):
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
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    confidence = None
    safety_msg = None

    if request.method == "POST":
        file = request.files["image"]
        img = cv2.imdecode(
            np.frombuffer(file.read(), np.uint8),
            cv2.IMREAD_COLOR
        )

        features = extract_features_from_image(img)
        pred = model.predict(features)[0]
        prob = model.predict_proba(features).max()

        if pred == 0:
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
        safety_msg=safety_msg
    )

    

if __name__ == "__main__":
    app.run(debug=True)
