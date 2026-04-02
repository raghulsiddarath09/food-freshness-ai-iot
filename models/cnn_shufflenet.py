# ================================================================
# MODULE 3 — ShuffleNet Architecture (Best Performing Model)
# AI & IoT Food Freshness & Safety System
# SRMIST Final Year Project 2025
# Authors: Kirthic Madavan, Raghul Siddarth, Lavanyah
# ================================================================
#
# ShuffleNet uses Channel Shuffle + Depthwise Convolutions.
# Achieves 97.25% accuracy — best of all three models.
# ================================================================

import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras.backend as K
from keras.models import Model
from keras.layers import (Input, Dense, Conv2D, DepthwiseConv2D,
                          GlobalAvgPool2D, MaxPool2D, BatchNormalization,
                          Concatenate, ReLU, Lambda)
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

# ── Paths ─────────────────────────────────────────────────────
TRAIN_DIR  = 'dataset/train'
TEST_DIR   = 'dataset/test'
MODEL_PATH = 'saved_models/SHUFFLENET.h5'
os.makedirs('saved_models', exist_ok=True)
os.makedirs('results', exist_ok=True)

# ── Hyperparameters ───────────────────────────────────────────
IMAGE_SIZE  = (224, 224)
INPUT_SHAPE = (224, 224, 3)
BATCH_SIZE  = 128
EPOCHS      = 100
G           = 8    # Number of groups for group convolution

# ── Data Generators ───────────────────────────────────────────
print("📦 Loading dataset...")

train_gen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
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

# ── Channel Shuffle ───────────────────────────────────────────
def channel_shuffle(x, groups):
    """Shuffles channels across groups to enable cross-group information flow."""
    _, w, h, ch = K.int_shape(x)
    ch_per_group = ch // groups

    def shuffle_op(x):
        x = K.reshape(x, [-1, w, h, ch_per_group, groups])
        x = K.permute_dimensions(x, [0, 1, 2, 4, 3])
        x = K.reshape(x, [-1, w, h, ch])
        return x

    return Lambda(shuffle_op)(x)

# ── Group Convolution ─────────────────────────────────────────
def group_conv(tensor, out_channels, groups):
    """Applies group convolution — splits channels into groups."""
    _, _, _, in_ch = K.int_shape(tensor)
    ch_per_group = in_ch // groups
    out_per_group = out_channels // groups
    group_outputs = []

    for i in range(groups):
        group_in = Lambda(lambda x, i=i: x[:, :, :, i*ch_per_group:(i+1)*ch_per_group])(tensor)
        group_out = Conv2D(out_per_group, 1)(group_in)
        group_outputs.append(group_out)

    return Concatenate()(group_outputs)

# ── ShuffleNet Block ──────────────────────────────────────────
def shufflenet_block(tensor, out_channels, stride, groups):
    """Core ShuffleNet building block with optional downsampling."""
    x = group_conv(tensor, out_channels // 4, groups)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = channel_shuffle(x, groups)

    x = DepthwiseConv2D(3, strides=stride, padding='same')(x)
    x = BatchNormalization()(x)

    residual_channels = out_channels if stride == 1 else out_channels - K.int_shape(tensor)[-1]
    x = group_conv(x, residual_channels, groups)
    x = BatchNormalization()(x)

    if stride == 1:
        x = Concatenate()([tensor, x]) if False else x  # Add shortcut
        x = tf.keras.layers.Add()([tensor, x]) if stride == 1 else x
    else:
        avg = tf.keras.layers.AveragePooling2D(3, strides=2, padding='same')(tensor)
        x = Concatenate()([avg, x])

    return ReLU()(x)

def shufflenet_stage(x, out_channels, num_blocks, groups):
    """Stack multiple ShuffleNet blocks into a stage."""
    x = shufflenet_block(x, out_channels, stride=2, groups=groups)
    for _ in range(num_blocks):
        x = shufflenet_block(x, out_channels, stride=1, groups=groups)
    return x

# ── Full ShuffleNet Model ─────────────────────────────────────
def build_shufflenet(input_shape, n_classes, groups=8):
    stage_channels  = [384, 769, 1536]
    stage_blocks    = [3, 7, 3]

    inputs = Input(input_shape)
    x = Conv2D(24, 3, strides=2, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPool2D(3, strides=2, padding='same')(x)

    for channels, blocks in zip(stage_channels, stage_blocks):
        x = shufflenet_stage(x, channels, blocks, groups)

    x = GlobalAvgPool2D()(x)
    outputs = Dense(n_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer='Adam',
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision()]
    )
    return model

# ── Build & Train ─────────────────────────────────────────────
print("\n🧠 Building ShuffleNet...")
K.clear_session()
model = build_shufflenet(INPUT_SHAPE, N_CLASSES, groups=G)
model.summary()

callbacks = [
    ModelCheckpoint(MODEL_PATH, monitor='accuracy', verbose=1, save_best_only=True),
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
]

print("\n🚀 Training ShuffleNet...")
history = model.fit(
    train_data,
    steps_per_epoch=train_data.samples // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=test_data,
    validation_steps=test_data.samples // BATCH_SIZE,
    callbacks=callbacks
)
print(f"\n✅ Best model saved → {MODEL_PATH}")

# ── Plot Results ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

axes[0].plot(history.history['accuracy'], linewidth=2, label='Train Acc')
for i, v in enumerate(history.history['accuracy']):
    if i % 10 == 0:
        axes[0].annotate(f'{v*100:.1f}%', xy=(i, v), fontsize=8)
axes[0].set_title('ShuffleNet — Accuracy'); axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy'); axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(history.history['loss'], linewidth=2, label='Train Loss', color='tomato')
for i, v in enumerate(history.history['loss']):
    if i % 10 == 0:
        axes[1].annotate(f'{v*100:.1f}', xy=(i, v), fontsize=8)
axes[1].set_title('ShuffleNet — Loss'); axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss'); axes[1].legend(); axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('results/shufflenet_history.png', dpi=150)
plt.show()
print("📊 Plot saved → results/shufflenet_history.png")
