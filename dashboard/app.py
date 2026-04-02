# ================================================================
# MODULE 4 — Django/Flask Deployment Web Application
# AI & IoT Food Freshness & Safety System
# SRMIST Final Year Project 2025
# Authors: Kirthic Madavan, Raghul Siddarth, Lavanyah
# ================================================================
#
# Flask web app for real-time food freshness prediction.
# Upload an image → get instant freshness classification.
# Also shows live IoT sensor data from Adafruit IO.
# ================================================================

import os
import io
import base64
import numpy as np
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
from PIL import Image
import datetime

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# ── Class Configuration ───────────────────────────────────────
CLASS_CONFIG = {
    0: {"label": "Fresh",        "emoji": "🟢", "color": "#4CAF50", "bg": "#E8F5E9",
        "advice": "Product is in excellent condition. Safe for consumption."},
    1: {"label": "Minor Defect", "emoji": "🟡", "color": "#FFC107", "bg": "#FFFDE7",
        "advice": "Minor surface defects detected. Consume soon or inspect closely."},
    2: {"label": "Major Defect", "emoji": "🟠", "color": "#FF9800", "bg": "#FFF3E0",
        "advice": "Significant deterioration. Not recommended for direct consumption."},
    3: {"label": "Spoiled",      "emoji": "🔴", "color": "#F44336", "bg": "#FFEBEE",
        "advice": "Product is spoiled. Discard immediately. Do not consume."},
}

# ── Load Model ────────────────────────────────────────────────
MODEL = None
MODEL_PATH = os.path.join('..', 'saved_models', 'SHUFFLENET.h5')

def load_cnn_model():
    global MODEL
    if os.path.exists(MODEL_PATH):
        MODEL = load_model(MODEL_PATH)
        print(f"✅ Model loaded from {MODEL_PATH}")
    else:
        print(f"⚠️  Model not found at {MODEL_PATH}")
        print("   Please train the model first: python models/cnn_shufflenet.py")

# ── Image Prediction ──────────────────────────────────────────
def predict_image(img_bytes):
    """Predict freshness class and confidence from raw image bytes."""
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB').resize((224, 224))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)

    preds      = MODEL.predict(arr, verbose=0)[0]
    class_idx  = int(np.argmax(preds))
    confidence = float(preds[class_idx]) * 100

    all_probs = {
        CLASS_CONFIG[i]['label']: round(float(preds[i]) * 100, 1)
        for i in range(len(CLASS_CONFIG))
    }

    return class_idx, confidence, all_probs

# ── Image to Base64 ───────────────────────────────────────────
def image_to_base64(img_bytes):
    return base64.b64encode(img_bytes).decode('utf-8')

# ── Routes ────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if MODEL is None:
        return jsonify({'error': 'Model not loaded. Train the model first.'}), 500

    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file      = request.files['image']
    img_bytes = file.read()

    if len(img_bytes) == 0:
        return jsonify({'error': 'Empty file'}), 400

    try:
        class_idx, confidence, all_probs = predict_image(img_bytes)
        cfg = CLASS_CONFIG[class_idx]

        return jsonify({
            'success':    True,
            'class_idx':  class_idx,
            'label':      cfg['label'],
            'emoji':      cfg['emoji'],
            'color':      cfg['color'],
            'bg':         cfg['bg'],
            'confidence': round(confidence, 1),
            'advice':     cfg['advice'],
            'all_probs':  all_probs,
            'image_b64':  image_to_base64(img_bytes),
            'timestamp':  datetime.datetime.now().strftime('%d %b %Y, %H:%M:%S')
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/sensor')
def sensor_api():
    """
    Returns simulated sensor data.
    In production: connect to Adafruit IO REST API.
    Replace the random values with real API calls.
    """
    import random
    temp = round(random.uniform(15, 38), 1)
    hum  = round(random.uniform(35, 92), 1)
    gas  = random.randint(80, 520)

    # Determine status
    temp_status = "ALERT" if temp > 30 else "OK"
    hum_status  = "ALERT" if hum > 85  else "OK"
    gas_status  = "ALERT" if gas > 400 else "OK"

    return jsonify({
        'temperature':   temp,
        'humidity':      hum,
        'gas_level':     gas,
        'temp_status':   temp_status,
        'hum_status':    hum_status,
        'gas_status':    gas_status,
        'timestamp':     datetime.datetime.now().strftime('%H:%M:%S'),
        'any_alert':     any(s == "ALERT" for s in [temp_status, hum_status, gas_status])
    })

@app.route('/health')
def health():
    return jsonify({
        'status':       'running',
        'model_loaded': MODEL is not None,
        'timestamp':    datetime.datetime.now().isoformat()
    })

# ── Run App ───────────────────────────────────────────────────
if __name__ == '__main__':
    load_cnn_model()
    print("\n🌐 Starting Food Freshness Dashboard...")
    print("   Open: http://localhost:5000\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
