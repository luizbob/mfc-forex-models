"""
Train Transformer for Quality Entry Prediction
===============================================
Predicts: Is this a quality entry? (binary)
Quality = MFC returns to center without big adverse move first.

Input: MFC features across all timeframes (M5, M15, M30, H1, H4)
Output: Binary (1 = quality entry, 0 = not quality)
"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from datetime import datetime

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix

def log(msg=""):
    print(msg, flush=True)

log("=" * 70)
log("QUALITY ENTRY TRANSFORMER TRAINING")
log("=" * 70)
log(f"Started: {datetime.now()}")
log(f"TensorFlow version: {tf.__version__}")

# Check GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    log(f"GPU available: {gpus}")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    log("No GPU found, using CPU")

# Paths
DATA_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/lstm_model/data')
MODEL_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/lstm_model/models')
MODEL_DIR.mkdir(exist_ok=True, parents=True)

# Load data
log("\n1. Loading data...")
with open(DATA_DIR / 'quality_entry_data.pkl', 'rb') as f:
    data = pickle.load(f)

df = data['data']
config = data['config']

log(f"  Total samples: {len(df)}")
log(f"  Quality rate: {df['is_quality'].mean()*100:.1f}%")
log(f"  Config: {config}")

# ============================================================================
# PREPARE FEATURES
# ============================================================================
log("\n2. Preparing features...")

# Feature columns - all MFC values and velocities
feature_cols = []

# MFC values for all timeframes
for tf in ['m5', 'm15', 'm30', 'h1', 'h4']:
    feature_cols.extend([f'base_{tf}', f'quote_{tf}'])
    feature_cols.extend([f'base_vel_{tf}', f'quote_vel_{tf}'])

# Additional features
feature_cols.extend(['base_vel2_h1', 'base_acc_h1', 'divergence', 'vel_divergence'])

# Direction encoding (buy=1, sell=0)
df['direction_code'] = (df['direction'] == 'buy').astype(int)
feature_cols.append('direction_code')

log(f"  Feature columns: {len(feature_cols)}")

# Prepare X and y
X = df[feature_cols].values.astype(np.float32)
y = df['is_quality'].values.astype(np.float32)

# Handle any NaN/inf
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

log(f"  X shape: {X.shape}")
log(f"  y shape: {y.shape}")

# Train/val/test split (time-based: 70/15/15)
n = len(df)
train_idx = int(n * 0.70)
val_idx = int(n * 0.85)

X_train, X_val, X_test = X[:train_idx], X[train_idx:val_idx], X[val_idx:]
y_train, y_val, y_test = y[:train_idx], y[train_idx:val_idx], y[val_idx:]

log(f"\n  Train: {len(X_train)} samples")
log(f"  Val: {len(X_val)} samples")
log(f"  Test: {len(X_test)} samples")

log(f"\n  Train quality rate: {y_train.mean()*100:.1f}%")
log(f"  Val quality rate: {y_val.mean()*100:.1f}%")
log(f"  Test quality rate: {y_test.mean()*100:.1f}%")

# ============================================================================
# BUILD TRANSFORMER MODEL
# ============================================================================
log("\n3. Building Transformer model...")


def build_quality_model(input_dim, hidden_units=128, dropout=0.3):
    """
    Build model for quality entry classification.
    Using simpler MLP architecture for tabular data.
    """
    inputs = layers.Input(shape=(input_dim,), name='features')

    # Dense layers with residual-like connections
    x = layers.Dense(hidden_units, activation='gelu', name='dense1')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout)(x)

    x = layers.Dense(hidden_units, activation='gelu', name='dense2')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout)(x)

    x = layers.Dense(hidden_units // 2, activation='gelu', name='dense3')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout * 0.5)(x)

    x = layers.Dense(hidden_units // 4, activation='gelu', name='dense4')(x)
    x = layers.Dropout(dropout * 0.5)(x)

    outputs = layers.Dense(1, activation='sigmoid', name='quality')(x)

    model = Model(inputs, outputs, name='quality_classifier')
    return model


# Build model
model = build_quality_model(
    input_dim=len(feature_cols),
    hidden_units=128,
    dropout=0.3
)

model.summary()

# ============================================================================
# COMPILE AND TRAIN
# ============================================================================
log("\n4. Compiling and training...")

# Class weights (to handle imbalance)
n_quality = y_train.sum()
n_non_quality = len(y_train) - n_quality
class_weight = {
    0: len(y_train) / (2 * n_non_quality),
    1: len(y_train) / (2 * n_quality)
}
log(f"  Class weights: {class_weight}")

model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=0.001, weight_decay=0.01),
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.AUC(name='auc')]
)

callbacks = [
    EarlyStopping(
        monitor='val_auc',
        patience=15,
        restore_best_weights=True,
        mode='max'
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=7,
        min_lr=1e-6
    ),
    ModelCheckpoint(
        MODEL_DIR / 'quality_transformer_best.keras',
        monitor='val_auc',
        save_best_only=True,
        mode='max'
    )
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=512,
    callbacks=callbacks,
    class_weight=class_weight,
    verbose=1
)

# ============================================================================
# EVALUATE
# ============================================================================
log("\n" + "=" * 70)
log("5. EVALUATION")
log("=" * 70)

# Test set evaluation
test_results = model.evaluate(X_test, y_test, verbose=0)
log(f"\nTest Results:")
log(f"  Loss: {test_results[0]:.4f}")
log(f"  Accuracy: {test_results[1]*100:.2f}%")
log(f"  AUC: {test_results[2]:.4f}")

# Predictions
y_pred_prob = model.predict(X_test, verbose=0).flatten()
y_pred = (y_pred_prob >= 0.5).astype(int)

# Classification report
log("\nClassification Report:")
log(classification_report(y_test, y_pred, target_names=['Non-Quality', 'Quality']))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
log(f"\nConfusion Matrix:")
log(f"  TN={cm[0,0]}, FP={cm[0,1]}")
log(f"  FN={cm[1,0]}, TP={cm[1,1]}")

# Precision at different thresholds
log("\nPrecision at different confidence thresholds:")
for thresh in [0.5, 0.6, 0.7, 0.8, 0.9]:
    high_conf_mask = y_pred_prob >= thresh
    if high_conf_mask.sum() > 0:
        precision = y_test[high_conf_mask].mean()
        count = high_conf_mask.sum()
        log(f"  Threshold {thresh}: Precision={precision*100:.1f}%, Count={count}")

# ============================================================================
# SAVE
# ============================================================================
log("\n" + "=" * 70)
log("6. SAVING")
log("=" * 70)

# Save final model
model.save(MODEL_DIR / 'quality_transformer_final.keras')
log(f"Saved model to: {MODEL_DIR / 'quality_transformer_final.keras'}")

# Save training history
with open(MODEL_DIR / 'quality_transformer_history.pkl', 'wb') as f:
    pickle.dump(history.history, f)

# Save feature columns for inference
with open(MODEL_DIR / 'quality_transformer_features.pkl', 'wb') as f:
    pickle.dump(feature_cols, f)

log(f"\nCompleted: {datetime.now()}")
