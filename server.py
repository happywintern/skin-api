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
# Ensure models directory exists
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    gdown.download(
        "https://drive.google.com/uc?id=1hkwHM6Ml1kfGWht1Jt1_KUa-yHfULiaj",
        MODEL_PATH,
        quiet=False
    )

model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded!")

# Class labels must match the training ordering used in the notebook
CLASS_LABELS = ["Blackheads", "Cyst", "Papules", "Pustules", "Whiteheads"]

# Indonesian labels for returned prediction text
ID_LABELS = {
    "Blackheads": "Komedo terbuka (Blackhead)",
    "Cyst": "Kista",
    "Papules": "Papula",
    "Pustules": "Pustula",
    "Whiteheads": "Komedo tertutup (Whitehead)"
}

RECOMMENDATIONS = {
    "Papules": {
        "description": "Benjolan kecil berwarna merah yang disebabkan peradangan atau infeksi folikel rambut. Tidak terlihat nanah, namun terasa nyeri jika disentuh.",
        "ingredients": [
            {"name": "Benzoyl Peroxide", "benefit": "Menghilangkan bakteri penyebab jerawat di dalam folikel", "concentration": "2.5–5%"},
            {"name": "Niacinamide", "benefit": "Mengurangi kemerahan dan menenangkan peradangan", "concentration": "5–10%"},
            {"name": "Azelaic Acid", "benefit": "Membersihkan pori-pori dan mengurangi pembengkakan", "concentration": "10–20%"},
            {"name": "Centella Asiatica", "benefit": "Menenangkan kulit yang iritasi dan mendukung penyembuhan", "concentration": "Sesuai label produk"},
        ]
    },
    "Pustules": {
        "description": "Benjolan meradang berisi nanah dengan inti putih atau kekuningan. Bentuk lanjut dari papula yang menunjukkan infeksi.",
        "ingredients": [
            {"name": "Benzoyl Peroxide", "benefit": "Mengeliminasi bakteri penyebab nanah", "concentration": "5–10%"},
            {"name": "Salicylic Acid", "benefit": "Mengelupas dan membantu mengosongkan pori yang tersumbat", "concentration": "0.5–2%"},
            {"name": "Tea Tree Oil", "benefit": "Antibakteri alami untuk mengurangi infeksi", "concentration": "Sekitar 5%"},
            {"name": "Zinc PCA", "benefit": "Mengontrol minyak dan mengurangi aktivitas bakteri", "concentration": "0.5–1%"},
        ]
    },
    "Whiteheads": {
        "description": "Komedo tertutup di mana sel kulit mati dan sebum terperangkap di bawah permukaan kulit, membentuk benjolan kecil berwarna putih.",
        "ingredients": [
            {"name": "Retinol", "benefit": "Mempercepat pergantian sel untuk mencegah penyumbatan pori", "concentration": "0.1–0.3% (mulai dari dosis rendah)"},
            {"name": "Salicylic Acid", "benefit": "Menembus dan melarutkan sumbatan di dalam pori", "concentration": "1–2%"},
            {"name": "Glycolic Acid", "benefit": "Eksfoliasi permukaan yang melonggarkan penumpukan sel kulit mati", "concentration": "5–10%"},
            {"name": "Niacinamide", "benefit": "Memperkecil tampilan pori dan mengatur produksi sebum", "concentration": "Sekitar 5%"},
        ]
    },
    "Blackheads": {
        "description": "Komedo terbuka di mana pori yang tersumbat terekspos ke udara sehingga sebum mengalami oksidasi dan berubah gelap/hitam.",
        "ingredients": [
            {"name": "Salicylic Acid", "benefit": "Larut dalam minyak — masuk ke pori dan melarutkan penumpukan", "concentration": "1–2%"},
            {"name": "Niacinamide", "benefit": "Merapikan pori dan mengurangi produksi minyak berlebih", "concentration": "5–10%"},
            {"name": "AHA (Glycolic/Lactic Acid)", "benefit": "Mengangkat sel kulit mati yang menyumbat pori", "concentration": "5–10%"},
            {"name": "Retinol", "benefit": "Mencegah blackhead baru dengan mengatur pergantian sel", "concentration": "0.1–0.5%"},
        ]
    },
    "Cyst": {
        "description": "Benjolan dalam yang nyeri berisi cairan di bawah permukaan kulit. Bentuk jerawat paling parah dan berisiko meninggalkan bekas jika tidak diobati.",
        "ingredients": [
            {"name": "Azelaic Acid", "benefit": "Mengurangi peradangan dalam dan mencegah jaringan parut", "concentration": "15–20%"},
            {"name": "Niacinamide", "benefit": "Menenangkan kemerahan yang parah dan memperkuat barrier kulit", "concentration": "Sekitar 10%"},
            {"name": "Centella Asiatica", "benefit": "Mendukung penyembuhan luka dan mengurangi bekas pasca-jerawat", "concentration": "Sesuai label produk"},
            {"name": "Retinol", "benefit": "Mencegah pembentukan kista baru dengan menjaga pori tetap bersih", "concentration": "0.1–0.3% (gunakan hati-hati)"},
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
    img = img.resize((150, 150))

    # Preprocess the image the same way as in training (EfficientNet)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array.astype("float32"))

    predictions = model.predict(img_array)[0]
    top_index = int(np.argmax(predictions))
    confidence = float(predictions[top_index]) * 100

    condition = CLASS_LABELS[top_index]
    rec = RECOMMENDATIONS.get(condition, {})
    # Return Indonesian-friendly condition label
    condition_id = ID_LABELS.get(condition, condition)

    return jsonify({
        "condition": condition_id,
        "confidence": round(confidence, 1),
        "description": rec.get("description", ""),
        "ingredients": rec.get("ingredients", [])
    })

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8001)
