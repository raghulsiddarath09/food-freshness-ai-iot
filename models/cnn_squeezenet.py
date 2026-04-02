# ================================================================
# MODULE 2 — SqueezeNet Architecture
# AI & IoT Food Freshness & Safety System
# SRMIST Final Year Project 2025
# Authors: Kirthic Madavan, Raghul Siddarth, Lavanyah
# ================================================================
#
# SqueezeNet uses "Fire Modules" — very small model (~1.2MB)
# Ideal for deployment on IoT/embedded devices.
# ================================================================

import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras.backend as K
from keras.models import Model
from keras.layers import (Input, Conv2D, MaxPool2D, GlobalAvgPool2D,
                          Concatenate, Activation)
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

# ── Paths ─────────────────────────────────────────────────────
TRAIN_DIR  = 'dataset/train'
TEST_DIR   = 'dataset/test'
MODEL_PATH = 'saved_models/SQUEEZE.h5'
os.makedirs('saved_models', exist_ok=True)
os.makedirs('results', exist_ok=True)

# ── Hyperparameters ───────────────────────────────────────────
IMAGE_SIZE   = (224, 224)
INPUT_SHAPE  = (224, 224, 3)
BATCH_SIZE   = 32
EPOCHS       = 50

# ── Data Generators ───────────────────────────────────────────
print("📦 Loading dataset...")

train_gen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)
test_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)
test_data = test_gen.flow_from_directory(
    TEST_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

N_CLASSES = len(train_data.class_indices)
print(f"✅ Found {N_CLASSES} classes: {train_data.class_indices}")

# ── SqueezeNet Fire Module ─────────────────────────────────────
def fire_module(x, squeeze_filters, expand_filters):
    """
    Fire Module: squeeze layer + expand layer (1x1 + 3x3 parallel)
    """
    # Squeeze
    s = Conv2D(squeeze_filters, 1, activation='relu', padding='same')(x)
    # Expand
    e1 = Conv2D(expand_filters, 1, activation='relu', padding='same')(s)
    e3 = Conv2D(expand_filters, 3, activation='relu', padding='same')(s)
    return Concatenate()([e1, e3])

# ── SqueezeNet Model ──────────────────────────────────────────
def build_squeezenet(input_shape, n_classes):
    inputs = Input(input_shape)

    x = Conv2D(96, 7, strides=2, padding='same', activation='relu')(inputs)
    x = MaxPool2D(3, strides=2, padding='same')(x)

    # Fire modules
    x = fire_module(x, 16,  64)
    x = fire_module(x, 16,  64)
    x = fire_module(x, 32, 128)
    x = MaxPool2D(3, strides=2, padding='same')(x)

    x = fire_module(x, 32, 128)
    x = fire_module(x, 48, 192)
    x = fire_module(x, 48, 192)
    x = fire_module(x, 64, 256)
    x = MaxPool2D(3, strides=2, padding='same')(x)

    x = fire_module(x, 64, 256)

    # Output
    x = Conv2D(n_classes, 1)(x)
    x = GlobalAvgPool2D()(x)
    outputs = Activation('softmax')(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer='Adam',
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision()]
    )
    return model

# ── Build & Train ─────────────────────────────────────────────
print("\n🧠 Building SqueezeNet...")
K.clear_session()
model = build_squeezenet(INPUT_SHAPE, N_CLASSES)
model.summary()

callbacks = [
    ModelCheckpoint(MODEL_PATH, monitor='accuracy', verbose=1, save_best_only=True),
    EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1)
]

print("\n🚀 Training SqueezeNet...")
history = model.fit(
    train_data,
    steps_per_epoch=train_data.samples // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=test_data,
    validation_steps=test_data.samples // BATCH_SIZE,
    callbacks=callbacks
)
print(f"\n✅ Model saved → {MODEL_PATH}")

# ── Plot Results ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Accuracy with epoch annotations
axes[0].plot(history.history['accuracy'], linewidth=2, label='Train')
for i, v in enumerate(history.history['accuracy']):
    if i % 5 == 0:
        axes[0].annotate(f'{v*100:.1f}', xy=(i, v), fontsize=8)
axes[0].set_title('SqueezeNet — Accuracy')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Accuracy')
axes[0].legend(); axes[0].grid(alpha=0.3)

# Loss
axes[1].plot(history.history['loss'], linewidth=2, label='Train')
for i, v in enumerate(history.history['loss']):
    if i % 5 == 0:
        axes[1].annotate(f'{v*100:.1f}', xy=(i, v), fontsize=8)
axes[1].set_title('SqueezeNet — Loss')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Loss')
axes[1].legend(); axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('results/squeezenet_history.png', dpi=150)
plt.show()
print("📊 Plot saved → results/squeezenet_history.png")
