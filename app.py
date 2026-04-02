# ============================================================
# app.py — Flask Dashboard for Food Freshness Monitoring
# AI & IoT Food Freshness & Safety System
# Authors: Kirthic Madavan, Raghul Siddarth, Lavanyah
# SRMIST Final Year Project — 2025
# ============================================================

import os
import numpy as np
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import io
import base64

app = Flask(__name__)

# ── Load trained CNN model ────────────────────────────────────
MODEL_PATH = os.path.join("..", "models", "food_freshness_cnn.h5")
model = None

CLASS_LABELS = {
    0: "🟢 Fresh",
    1: "🟡 Minor Defect",
    2: "🟠 Major Defect",
    3: "🔴 Spoiled"
}

def load_cnn_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH)
        print("✅ CNN model loaded successfully.")
    else:
        print("⚠️  Model file not found. Please train the model first.")

# ── Predict freshness from image ──────────────────────────────
def predict_freshness(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)[0]
    class_idx   = np.argmax(predictions)
    confidence  = float(predictions[class_idx]) * 100

    return CLASS_LABELS[class_idx], confidence

# ── Routes ────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded. Train the model first.'}), 500

    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file      = request.files['image']
    img_bytes = file.read()

    label, confidence = predict_freshness(img_bytes)

    return jsonify({
        'prediction': label,
        'confidence': f"{confidence:.1f}%"
    })

@app.route('/api/sensor-data')
def sensor_data():
    """
    Returns latest sensor readings.
    In production, connect this to your Adafruit IO API or database.
    """
    import random  # Replace with real sensor data in production
    return jsonify({
        'temperature': round(random.uniform(18, 35), 1),
        'humidity':    round(random.uniform(40, 90), 1),
        'gas_level':   random.randint(100, 500),
        'timestamp':   __import__('datetime').datetime.now().strftime('%H:%M:%S')
    })

# ── Main ──────────────────────────────────────────────────────
if __name__ == '__main__':
    load_cnn_model()
    print("🌐 Dashboard running at http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
