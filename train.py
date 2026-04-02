# ============================================================
# train.py — CNN Model Training Script
# AI & IoT Food Freshness & Safety System
# Authors: Kirthic Madavan, Raghul Siddarth, Lavanyah
# SRMIST Final Year Project — 2025
# ============================================================

import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten,
                                     Dense, Dropout, BatchNormalization)
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam

# ── Configuration ─────────────────────────────────────────────
DATASET_DIR   = "dataset/"
TRAIN_DIR     = os.path.join(DATASET_DIR, "train")
TEST_DIR      = os.path.join(DATASET_DIR, "test")
IMAGE_SIZE    = (224, 224)
BATCH_SIZE    = 32
EPOCHS        = 50
NUM_CLASSES   = 4        # fresh, minor_defect, major_defect, spoiled
MODEL_SAVE    = "models/food_freshness_cnn.h5"

# ── Data Augmentation & Generators ───────────────────────────
print("📦 Loading dataset...")

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rotation_range=10,
    brightness_range=[0.8, 1.2]
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

training_set = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_set = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

print(f"✅ Classes found: {training_set.class_indices}")

# ── CNN Model Architecture ────────────────────────────────────
print("\n🧠 Building CNN model...")

model = Sequential([
    # Block 1
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(pool_size=(2, 2)),

    # Block 2
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    # Block 3
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    # Block 4
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    # Block 5
    Conv2D(128, (3, 3), activation='relu'),

    # Fully Connected Layers
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ── Callbacks ─────────────────────────────────────────────────
os.makedirs("models", exist_ok=True)

callbacks = [
    ModelCheckpoint(
        MODEL_SAVE,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
]

# ── Training ──────────────────────────────────────────────────
print("\n🚀 Training started...\n")

history = model.fit(
    training_set,
    steps_per_epoch=training_set.samples // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=test_set,
    validation_steps=test_set.samples // BATCH_SIZE,
    callbacks=callbacks
)

print(f"\n✅ Training complete! Model saved to: {MODEL_SAVE}")

# ── Plot Results ──────────────────────────────────────────────
os.makedirs("results", exist_ok=True)

def plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy plot
    axes[0].plot(history.history['accuracy'],    label='Train Accuracy', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy',   linewidth=2)
    axes[0].set_title('Model Accuracy', fontsize=14)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss plot
    axes[1].plot(history.history['loss'],    label='Train Loss', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Val Loss',   linewidth=2)
    axes[1].set_title('Model Loss', fontsize=14)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/training_history.png', dpi=150)
    plt.show()
    print("📊 Training graph saved to results/training_history.png")

plot_history(history)

# ── Final Evaluation ──────────────────────────────────────────
loss, accuracy = model.evaluate(test_set, verbose=0)
print(f"\n📈 Final Test Accuracy : {accuracy * 100:.2f}%")
print(f"📉 Final Test Loss     : {loss:.4f}")
