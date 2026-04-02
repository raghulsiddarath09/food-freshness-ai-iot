# ================================================================
# MODULE 1 — Manual CNN Architecture (Baseline)
# AI & IoT Food Freshness & Safety System
# SRMIST Final Year Project 2025
# Authors: Kirthic Madavan, Raghul Siddarth, Lavanyah
# ================================================================
#
# This is the BASELINE model from Sprint I.
# Classifies food images into: Fresh | Slightly Spoiled | Spoiled
# ================================================================

import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense, Dropout)
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import RMSprop

# ── Paths ─────────────────────────────────────────────────────
TRAIN_DIR  = 'dataset/train'
TEST_DIR   = 'dataset/test'
MODEL_PATH = 'saved_models/MANUAL.h5'
os.makedirs('saved_models', exist_ok=True)
os.makedirs('results', exist_ok=True)

# ── Hyperparameters ───────────────────────────────────────────
IMAGE_SIZE  = (224, 224)
BATCH_SIZE  = 32
EPOCHS      = 100

# ── Data Generators ───────────────────────────────────────────
print("📦 Loading dataset...")

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
test_datagen = ImageDataGenerator(rescale=1./255)

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

NUM_CLASSES = len(training_set.class_indices)
print(f"✅ Found {NUM_CLASSES} classes: {training_set.class_indices}")

# ── Model Architecture ────────────────────────────────────────
print("\n🧠 Building Manual CNN model...")

model = Sequential([
    Conv2D(32, (3,3), input_shape=(224,224,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    optimizer=RMSprop(),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

# ── Callbacks ─────────────────────────────────────────────────
callbacks = [
    ModelCheckpoint(MODEL_PATH, monitor='accuracy', verbose=1, save_best_only=True)
]

# ── Training ──────────────────────────────────────────────────
print("\n🚀 Training Manual CNN...")
history = model.fit(
    training_set,
    steps_per_epoch=training_set.samples // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=test_set,
    validation_steps=test_set.samples // BATCH_SIZE,
    callbacks=callbacks
)
print(f"\n✅ Model saved → {MODEL_PATH}")

# ── Plot Accuracy & Loss ──────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history.history['accuracy'],     label='Train', linewidth=2)
axes[0].plot(history.history['val_accuracy'], label='Val',   linewidth=2)
axes[0].set_title('Manual CNN — Accuracy')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Accuracy')
axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(history.history['loss'],     label='Train', linewidth=2)
axes[1].plot(history.history['val_loss'], label='Val',   linewidth=2)
axes[1].set_title('Manual CNN — Loss')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Loss')
axes[1].legend(); axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('results/manual_cnn_history.png', dpi=150)
plt.show()
print("📊 Plot saved → results/manual_cnn_history.png")
