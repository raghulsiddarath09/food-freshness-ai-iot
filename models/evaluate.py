# ================================================================
# evaluate.py — Evaluate & Compare All Models
# AI & IoT Food Freshness & Safety System
# SRMIST Final Year Project 2025
# ================================================================
#
# Run this AFTER training all 3 models.
# Generates: confusion matrix, classification report, comparison table
# ================================================================

import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, precision_score,
                             recall_score, f1_score)

# ── Config ────────────────────────────────────────────────────
TEST_DIR   = 'dataset/test'
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

MODELS = {
    'Manual CNN':  'saved_models/MANUAL.h5',
    'SqueezeNet':  'saved_models/SQUEEZE.h5',
    'ShuffleNet':  'saved_models/SHUFFLENET.h5',
}

os.makedirs('results', exist_ok=True)

# ── Load test data ────────────────────────────────────────────
test_gen = ImageDataGenerator(rescale=1./255)
test_data = test_gen.flow_from_directory(
    TEST_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

CLASS_NAMES = list(test_data.class_indices.keys())
y_true = test_data.classes
print(f"✅ Classes: {CLASS_NAMES}")
print(f"📊 Test samples: {len(y_true)}\n")

# ── Evaluate each model ───────────────────────────────────────
results = {}

for name, path in MODELS.items():
    if not os.path.exists(path):
        print(f"⚠️  Skipping {name} — model file not found at {path}")
        continue

    print(f"🔍 Evaluating {name}...")
    model = load_model(path)

    y_pred_prob = model.predict(test_data, verbose=0)
    y_pred      = np.argmax(y_pred_prob, axis=1)

    acc  = accuracy_score(y_true, y_pred) * 100
    prec = precision_score(y_true, y_pred, average='weighted') * 100
    rec  = recall_score(y_true, y_pred, average='weighted') * 100
    f1   = f1_score(y_true, y_pred, average='weighted') * 100

    results[name] = {
        'accuracy':  round(acc, 2),
        'precision': round(prec, 2),
        'recall':    round(rec, 2),
        'f1_score':  round(f1, 2),
        'y_pred':    y_pred
    }

    print(f"   Accuracy : {acc:.2f}%")
    print(f"   Precision: {prec:.2f}%")
    print(f"   Recall   : {rec:.2f}%")
    print(f"   F1-Score : {f1:.2f}%\n")

    # ── Confusion Matrix ──────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title(f'{name} — Confusion Matrix')
    plt.xlabel('Predicted'); plt.ylabel('Actual')
    plt.tight_layout()
    safe_name = name.lower().replace(' ', '_')
    plt.savefig(f'results/{safe_name}_confusion_matrix.png', dpi=150)
    plt.close()

    # ── Classification Report ─────────────────────────────────
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES)
    with open(f'results/{safe_name}_report.txt', 'w') as f:
        f.write(f"{name} — Classification Report\n")
        f.write("=" * 50 + "\n")
        f.write(report)
    print(f"   📄 Report saved → results/{safe_name}_report.txt")
    print(f"   📊 Confusion matrix saved → results/{safe_name}_confusion_matrix.png\n")

# ── Comparison Table ──────────────────────────────────────────
if results:
    print("\n" + "=" * 60)
    print("📈 MODEL COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Model':<15} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
    print("-" * 60)
    for name, r in results.items():
        print(f"{name:<15} {r['accuracy']:>9.2f}% {r['precision']:>9.2f}% "
              f"{r['recall']:>9.2f}% {r['f1_score']:>9.2f}%")
    print("=" * 60)

    # ── Bar Chart Comparison ──────────────────────────────────
    metrics  = ['accuracy', 'precision', 'recall', 'f1_score']
    labels   = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    x        = np.arange(len(metrics))
    width    = 0.25
    colors   = ['#2196F3', '#4CAF50', '#FF9800']

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (name, r) in enumerate(results.items()):
        values = [r[m] for m in metrics]
        bars = ax.bar(x + i * width, values, width, label=name, color=colors[i], alpha=0.85)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f'{v:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_xlabel('Metric', fontsize=12)
    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(70, 105)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/model_comparison.png', dpi=150)
    plt.show()
    print("\n📊 Comparison chart saved → results/model_comparison.png")
