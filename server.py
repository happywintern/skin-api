from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import gdown

app = Flask(__name__)
CORS(app)  # allows your website to call this server

MODEL_PATH = "skin_model.keras"
if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    gdown.download(
        "https://drive.google.com/uc?id=1hkwHM6Ml1kfGWht1Jt1_KUa-yHfULiaj",
        MODEL_PATH,
        quiet=False
    )

model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded!")

# Your class labels — change these to match YOUR model's output
CLASS_LABELS = ["Papule", "Pustule", "Whitehead", "Blackhead", "Cyst"]
RECOMMENDATIONS = {
    "Papule": {
        "description": "Small, raised, red bumps caused by inflamed or infected hair follicles. No visible pus, but tender to touch.",
        "ingredients": [
            {"name": "Benzoyl Peroxide", "benefit": "Kills acne-causing bacteria deep in the follicle", "concentration": "2.5–5%"},
            {"name": "Niacinamide", "benefit": "Reduces redness and calms inflammation", "concentration": "5–10%"},
            {"name": "Azelaic Acid", "benefit": "Unclogs pores and reduces swelling", "concentration": "10–20%"},
            {"name": "Centella Asiatica", "benefit": "Soothes irritated skin and supports healing", "concentration": "As listed"},
        ]
    },
    "Pustule": {
        "description": "Inflamed, pus-filled bumps with a white or yellow center. A more progressed form of papule with visible infection.",
        "ingredients": [
            {"name": "Benzoyl Peroxide", "benefit": "Eliminates bacteria causing pus buildup", "concentration": "5–10%"},
            {"name": "Salicylic Acid", "benefit": "Exfoliates and drains blocked pores", "concentration": "0.5–2%"},
            {"name": "Tea Tree Oil", "benefit": "Natural antibacterial to reduce infection", "concentration": "5%"},
            {"name": "Zinc PCA", "benefit": "Controls oil and reduces bacterial activity", "concentration": "0.5–1%"},
        ]
    },
    "Whitehead": {
        "description": "Closed comedones where dead skin cells and sebum are trapped beneath the skin surface, forming a small white bump.",
        "ingredients": [
            {"name": "Retinol", "benefit": "Speeds up cell turnover to prevent pore blockage", "concentration": "0.1–0.3% (start low)"},
            {"name": "Salicylic Acid", "benefit": "Penetrates and dissolves the clog inside the pore", "concentration": "1–2%"},
            {"name": "Glycolic Acid", "benefit": "Surface exfoliant that loosens dead skin buildup", "concentration": "5–10%"},
            {"name": "Niacinamide", "benefit": "Minimizes pore appearance and regulates sebum", "concentration": "5%"},
        ]
    },
    "Blackhead": {
        "description": "Open comedones where the clogged pore is exposed to air, oxidizing the sebum and turning it dark or black.",
        "ingredients": [
            {"name": "Salicylic Acid", "benefit": "Oil-soluble — gets inside pores and dissolves buildup", "concentration": "1–2%"},
            {"name": "Niacinamide", "benefit": "Tightens pores and reduces excess oil production", "concentration": "5–10%"},
            {"name": "AHA (Glycolic/Lactic Acid)", "benefit": "Removes dead skin cells that contribute to clogs", "concentration": "5–10%"},
            {"name": "Retinol", "benefit": "Prevents future blackheads by regulating cell turnover", "concentration": "0.1–0.5%"},
        ]
    },
    "Cyst": {
        "description": "Deep, painful, fluid-filled lumps beneath the skin surface. The most severe form of acne — prone to scarring if untreated.",
        "ingredients": [
            {"name": "Azelaic Acid", "benefit": "Reduces deep inflammation and prevents scarring", "concentration": "15–20%"},
            {"name": "Niacinamide", "benefit": "Calms severe redness and strengthens skin barrier", "concentration": "10%"},
            {"name": "Centella Asiatica", "benefit": "Promotes wound healing and reduces post-acne marks", "concentration": "As listed"},
            {"name": "Retinol", "benefit": "Prevents new cysts by keeping pores clear", "concentration": "0.1–0.3% (use carefully)"},
        ]
    },
}
@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    img = Image.open(io.BytesIO(file.read())).convert("RGB")
    img = img.resize((150, 150))  # adjust size if your model uses different input

    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)[0]
    top_index = int(np.argmax(predictions))
    confidence = float(predictions[top_index]) * 100

    condition = CLASS_LABELS[top_index]
    rec = RECOMMENDATIONS[condition]

    return jsonify({
        "condition": condition,
        "confidence": round(confidence, 1),
        "description": rec["description"],
        "ingredients": rec["ingredients"]
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
