# ================================================================
# prepare_dataset.py — Dataset Preparation & Analysis
# AI & IoT Food Freshness & Safety System
# SRMIST Final Year Project 2025
# ================================================================
#
# Run this BEFORE training.
# Checks dataset structure, shows sample images, prints stats.
# ================================================================

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# ── Config ────────────────────────────────────────────────────
DATASET_ROOT = 'dataset'
SPLITS       = ['train', 'test']

# ── Dataset Statistics ────────────────────────────────────────
def get_image_stats(folder_path):
    """Returns count and image dimension stats for a folder."""
    files = glob.glob(os.path.join(folder_path, '**', '*.*'), recursive=True)
    files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if not files:
        return {'count': 0}

    widths, heights = [], []
    for f in files:
        try:
            img = Image.open(f)
            w, h = img.size
            widths.append(w); heights.append(h)
        except:
            pass

    return {
        'count':      len(files),
        'min_width':  min(widths),
        'max_width':  max(widths),
        'min_height': min(heights),
        'max_height': max(heights),
        'avg_width':  int(np.mean(widths)),
        'avg_height': int(np.mean(heights)),
    }

# ── Show Sample Images ────────────────────────────────────────
def show_samples(folder_path, class_name, n=5):
    """Displays n sample images from a class folder."""
    files = glob.glob(os.path.join(folder_path, '*.*'))
    files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:n]

    if not files:
        print(f"   ⚠️  No images found in {folder_path}")
        return

    fig, axes = plt.subplots(1, len(files), figsize=(3 * len(files), 3))
    if len(files) == 1:
        axes = [axes]

    for ax, f in zip(axes, files):
        img = Image.open(f)
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(os.path.basename(f)[:12], fontsize=8)

    plt.suptitle(f'Samples — {class_name}', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()

# ── Main Analysis ─────────────────────────────────────────────
print("=" * 60)
print("  DATASET ANALYSIS — Food Freshness Project")
print("=" * 60)

total_images   = 0
class_counts   = {}

for split in SPLITS:
    split_path = os.path.join(DATASET_ROOT, split)
    if not os.path.exists(split_path):
        print(f"\n⚠️  Split folder not found: {split_path}")
        continue

    classes = sorted(os.listdir(split_path))
    print(f"\n📂 {split.upper()} SET ({len(classes)} classes):")
    print("-" * 50)

    split_total = 0
    for cls in classes:
        cls_path = os.path.join(split_path, cls)
        if not os.path.isdir(cls_path):
            continue

        stats = get_image_stats(cls_path)
        count = stats.get('count', 0)
        split_total += count
        total_images += count

        if cls not in class_counts:
            class_counts[cls] = 0
        class_counts[cls] += count

        print(f"  📁 {cls:<20} → {count:>5} images", end='')
        if count > 0:
            print(f"  | Dims: {stats['min_width']}–{stats['max_width']} x "
                  f"{stats['min_height']}–{stats['max_height']} px")
        else:
            print()

    print(f"  {'TOTAL':<20} → {split_total:>5} images")

print(f"\n{'='*60}")
print(f"  TOTAL IMAGES IN DATASET: {total_images}")
print(f"{'='*60}\n")

# ── Class Distribution Plot ───────────────────────────────────
if class_counts:
    os.makedirs('results', exist_ok=True)
    names  = list(class_counts.keys())
    counts = list(class_counts.values())
    colors = ['#4CAF50', '#FFC107', '#FF9800', '#F44336'][:len(names)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart
    bars = axes[0].bar(names, counts, color=colors, alpha=0.85, edgecolor='white', linewidth=1.5)
    for bar, count in zip(bars, counts):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                      str(count), ha='center', fontweight='bold')
    axes[0].set_title('Class Distribution', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Class'); axes[0].set_ylabel('Image Count')
    axes[0].grid(axis='y', alpha=0.3)

    # Pie chart
    axes[1].pie(counts, labels=names, colors=colors, autopct='%1.1f%%',
                 startangle=90, pctdistance=0.8,
                 wedgeprops={'edgecolor': 'white', 'linewidth': 2})
    axes[1].set_title('Class Balance (%)', fontsize=13, fontweight='bold')

    plt.suptitle('Dataset Analysis — Food Freshness System', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/dataset_distribution.png', dpi=150)
    plt.show()
    print("📊 Distribution chart saved → results/dataset_distribution.png")

# ── Show sample images for each class ────────────────────────
print("\n🖼️  Showing sample images from training set...")
train_path = os.path.join(DATASET_ROOT, 'train')
if os.path.exists(train_path):
    for cls in sorted(os.listdir(train_path)):
        cls_path = os.path.join(train_path, cls)
        if os.path.isdir(cls_path):
            show_samples(cls_path, cls, n=5)
