import base64
import io

import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image, ImageOps

from cnn import load_model

app = Flask(__name__)
model = None


def get_model():
    global model
    if model is None:
        model = load_model()
    return model


def preprocess(image: Image.Image) -> np.ndarray:
    """Turn an RGBA canvas drawing into a normalized 28x28 MNIST-style array."""
    background = Image.new("RGBA", image.size, (255, 255, 255, 255))
    grayscale = ImageOps.invert(Image.alpha_composite(background, image).convert("L"))

    bbox = grayscale.getbbox()
    if bbox is not None:
        grayscale = grayscale.crop(bbox)

    grayscale.thumbnail((20, 20), Image.LANCZOS)
    canvas = Image.new("L", (28, 28), 0)
    canvas.paste(grayscale, ((28 - grayscale.width) // 2, (28 - grayscale.height) // 2))

    return np.array(canvas, dtype="float32").reshape(1, 28, 28, 1) / 255.0


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True) or {}
    data_url = payload.get("image", "")
    if "," not in data_url:
        return jsonify({"error": "No image provided"}), 400

    image = Image.open(io.BytesIO(base64.b64decode(data_url.split(",", 1)[1]))).convert("RGBA")
    probabilities = get_model().predict(preprocess(image), verbose=0)[0]

    return jsonify({
        "digit": int(np.argmax(probabilities)),
        "confidence": float(np.max(probabilities)),
        "probabilities": [float(p) for p in probabilities],
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
