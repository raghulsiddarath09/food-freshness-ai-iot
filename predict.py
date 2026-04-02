# ================================================================
# predict.py — Predict freshness of a single food image
# AI & IoT Food Freshness & Safety System
# SRMIST Final Year Project 2025
# ================================================================
#
# Usage:
#   python predict.py --image path/to/your/image.jpg
#   python predict.py --image path/to/image.jpg --model shufflenet
# ================================================================

import argparse
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
from PIL import Image

# ── Class Labels ─────────────────────────────────────────────
CLASS_LABELS = {
    0: ("🟢 Fresh",        "#4CAF50"),
    1: ("🟡 Minor Defect", "#FFC107"),
    2: ("🟠 Major Defect", "#FF9800"),
    3: ("🔴 Spoiled",      "#F44336"),
}

MODEL_PATHS = {
    'manual':     'saved_models/MANUAL.h5',
    'squeezenet': 'saved_models/SQUEEZE.h5',
    'shufflenet': 'saved_models/SHUFFLENET.h5',
}

# ── Argument Parser ───────────────────────────────────────────
parser = argparse.ArgumentParser(description='Predict food freshness from an image')
parser.add_argument('--image', required=True, help='Path to the food image')
parser.add_argument('--model', default='shufflenet',
                    choices=['manual', 'squeezenet', 'shufflenet'],
                    help='Which model to use (default: shufflenet)')
args = parser.parse_args()

# ── Load Model ────────────────────────────────────────────────
model_path = MODEL_PATHS[args.model]
print(f"📂 Loading {args.model} model from {model_path}...")
model = load_model(model_path)
print("✅ Model loaded!")

# ── Preprocess Image ──────────────────────────────────────────
img = keras_image.load_img(args.image, target_size=(224, 224))
img_array = keras_image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# ── Predict ───────────────────────────────────────────────────
predictions  = model.predict(img_array)[0]
class_idx    = np.argmax(predictions)
confidence   = predictions[class_idx] * 100
label, color = CLASS_LABELS[class_idx]

# ── Display Results ───────────────────────────────────────────
print("\n" + "=" * 40)
print(f"  Prediction : {label}")
print(f"  Confidence : {confidence:.1f}%")
print(f"  Model Used : {args.model}")
print("=" * 40 + "\n")

# All class probabilities
print("📊 Class Probabilities:")
for idx, (lbl, _) in CLASS_LABELS.items():
    bar = "█" * int(predictions[idx] * 30)
    print(f"  {lbl:<20} {predictions[idx]*100:5.1f}%  {bar}")

# ── Visual Output ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Show image
original = Image.open(args.image)
axes[0].imshow(original)
axes[0].set_title('Input Image', fontsize=13)
axes[0].axis('off')

# Show bar chart of probabilities
labels_list = [CLASS_LABELS[i][0] for i in range(len(CLASS_LABELS))]
colors_list = [CLASS_LABELS[i][1] for i in range(len(CLASS_LABELS))]
bars = axes[1].barh(labels_list, predictions * 100, color=colors_list, alpha=0.85)
axes[1].set_xlabel('Confidence (%)', fontsize=12)
axes[1].set_title(f'Prediction: {label}\nConfidence: {confidence:.1f}%',
                   fontsize=13, fontweight='bold')
axes[1].set_xlim(0, 105)

for bar, val in zip(bars, predictions * 100):
    axes[1].text(val + 1, bar.get_y() + bar.get_height()/2,
                  f'{val:.1f}%', va='center', fontsize=11, fontweight='bold')

axes[1].grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('results/prediction_output.png', dpi=150)
plt.show()
print("📊 Prediction visual saved → results/prediction_output.png")
