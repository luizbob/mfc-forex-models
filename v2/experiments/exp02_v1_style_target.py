"""
Experiment 02: V1-Style Target (Per-Pair)
=========================================
Uses the same target logic as V1 model but applied per-pair:
- Binary classification: LONG vs SHORT
- Based on which direction moves more in the horizon window
- No neutral class - every sample is tradeable

This removes the "neutral" problem and focuses on direction prediction.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import pickle
from pathlib import Path
from datetime import datetime
import gc

import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def log(msg=""):
    print(msg, flush=True)

log("=" * 70)
log("EXPERIMENT 02: V1-Style Target (Per-Pair)")
log("=" * 70)
log(f"Started: {datetime.now()}")

# ============================================================================
# CONFIGURATION
# ============================================================================

V2_DIR = Path(__file__).parent.parent
DATA_DIR = V2_DIR / "data" / "pairs"
EXP_DIR = Path(__file__).parent
RESULTS_DIR = EXP_DIR / "results_exp02"
RESULTS_DIR.mkdir(exist_ok=True)

# Training parameters
BATCH_SIZE = 512
EPOCHS = 50
PATIENCE = 10

# Model architecture
LSTM_UNITS = 64
DENSE_UNITS = 32
DROPOUT = 0.3

# Test on subset of pairs first
TEST_PAIRS = ['EURUSD', 'GBPUSD', 'USDJPY', 'EURJPY', 'AUDUSD', 'GBPJPY']

# Load config
config_path = DATA_DIR / "config_pairs.pkl"
with open(config_path, 'rb') as f:
    config = pickle.load(f)

LOOKBACKS = config['lookbacks']

log(f"\nConfig loaded:")
log(f"   Lookbacks: {LOOKBACKS}")
log(f"   Test pairs: {TEST_PAIRS}")
log(f"   Target: Binary (LONG=1 vs SHORT=0)")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def calculate_v1_style_target(y_long_pips, y_short_pips):
    """
    Calculate binary target based on which direction moved more.

    Returns:
        y: 0=SHORT (short pips > long pips), 1=LONG (long pips >= short pips)
    """
    # LONG if upward movement >= downward movement
    y = (y_long_pips >= y_short_pips).astype(np.int8)
    return y

def create_lstm_model_binary():
    """Create LSTM model for binary classification."""

    # Base currency inputs
    input_base_m5 = layers.Input(shape=(LOOKBACKS['M5'], 1), name='base_m5')
    input_base_m15 = layers.Input(shape=(LOOKBACKS['M15'], 1), name='base_m15')
    input_base_m30 = layers.Input(shape=(LOOKBACKS['M30'], 1), name='base_m30')
    input_base_h1 = layers.Input(shape=(LOOKBACKS['H1'], 1), name='base_h1')
    input_base_h4 = layers.Input(shape=(LOOKBACKS['H4'], 1), name='base_h4')

    # Quote currency inputs
    input_quote_m5 = layers.Input(shape=(LOOKBACKS['M5'], 1), name='quote_m5')
    input_quote_m15 = layers.Input(shape=(LOOKBACKS['M15'], 1), name='quote_m15')
    input_quote_m30 = layers.Input(shape=(LOOKBACKS['M30'], 1), name='quote_m30')
    input_quote_h1 = layers.Input(shape=(LOOKBACKS['H1'], 1), name='quote_h1')
    input_quote_h4 = layers.Input(shape=(LOOKBACKS['H4'], 1), name='quote_h4')

    # Auxiliary input
    input_aux = layers.Input(shape=(6,), name='aux')

    # Shared LSTM layers
    lstm_m5 = layers.LSTM(LSTM_UNITS, return_sequences=False, name='lstm_m5')
    lstm_m15 = layers.LSTM(LSTM_UNITS, return_sequences=False, name='lstm_m15')
    lstm_m30 = layers.LSTM(LSTM_UNITS, return_sequences=False, name='lstm_m30')
    lstm_h1 = layers.LSTM(LSTM_UNITS, return_sequences=False, name='lstm_h1')
    lstm_h4 = layers.LSTM(LSTM_UNITS, return_sequences=False, name='lstm_h4')

    # Process both currencies
    base_m5_out = lstm_m5(input_base_m5)
    base_m15_out = lstm_m15(input_base_m15)
    base_m30_out = lstm_m30(input_base_m30)
    base_h1_out = lstm_h1(input_base_h1)
    base_h4_out = lstm_h4(input_base_h4)

    quote_m5_out = lstm_m5(input_quote_m5)
    quote_m15_out = lstm_m15(input_quote_m15)
    quote_m30_out = lstm_m30(input_quote_m30)
    quote_h1_out = lstm_h1(input_quote_h1)
    quote_h4_out = lstm_h4(input_quote_h4)

    # Concatenate
    combined = layers.Concatenate()([
        base_m5_out, base_m15_out, base_m30_out, base_h1_out, base_h4_out,
        quote_m5_out, quote_m15_out, quote_m30_out, quote_h1_out, quote_h4_out,
        input_aux
    ])

    # Dense layers
    x = layers.Dense(DENSE_UNITS * 2, activation='relu')(combined)
    x = layers.Dropout(DROPOUT)(x)
    x = layers.Dense(DENSE_UNITS, activation='relu')(x)
    x = layers.Dropout(DROPOUT)(x)

    # Binary output - single sigmoid
    output = layers.Dense(1, activation='sigmoid', name='output')(x)

    model = Model(
        inputs=[
            input_base_m5, input_base_m15, input_base_m30, input_base_h1, input_base_h4,
            input_quote_m5, input_quote_m15, input_quote_m30, input_quote_h1, input_quote_h4,
            input_aux
        ],
        outputs=output
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model

def load_pair_data(pair):
    """Load data for a pair."""
    path = DATA_DIR / f"v2_pair_{pair}.pkl"

    with open(path, 'rb') as f:
        data = pickle.load(f)

    X = {
        'base_m5': data['X_base_M5'][..., np.newaxis],
        'base_m15': data['X_base_M15'][..., np.newaxis],
        'base_m30': data['X_base_M30'][..., np.newaxis],
        'base_h1': data['X_base_H1'][..., np.newaxis],
        'base_h4': data['X_base_H4'][..., np.newaxis],
        'quote_m5': data['X_quote_M5'][..., np.newaxis],
        'quote_m15': data['X_quote_M15'][..., np.newaxis],
        'quote_m30': data['X_quote_M30'][..., np.newaxis],
        'quote_h1': data['X_quote_H1'][..., np.newaxis],
        'quote_h4': data['X_quote_H4'][..., np.newaxis],
        'aux': data['X_aux'],
    }

    return X, data['y_long_pips'], data['y_short_pips'], data['datetimes']

def time_based_split(X, y, test_ratio=0.1, val_ratio=0.15):
    """Split data by time."""
    n = len(y)
    test_idx = int(n * (1 - test_ratio))

    X_test = {k: v[test_idx:] for k, v in X.items()}
    y_test = y[test_idx:]

    X_trainval = {k: v[:test_idx] for k, v in X.items()}
    y_trainval = y[:test_idx]

    n_trainval = len(y_trainval)
    val_idx = int(n_trainval * (1 - val_ratio))

    X_train = {k: v[:val_idx] for k, v in X_trainval.items()}
    y_train = y_trainval[:val_idx]

    X_val = {k: v[val_idx:] for k, v in X_trainval.items()}
    y_val = y_trainval[val_idx:]

    return X_train, y_train, X_val, y_val, X_test, y_test

# ============================================================================
# RUN EXPERIMENTS
# ============================================================================

results = []

for pair in TEST_PAIRS:
    log(f"\n{'='*60}")
    log(f"Training {pair} (Binary V1-Style)...")
    log(f"{'='*60}")

    # Load data
    log(f"   Loading data...")
    X, y_long_pips, y_short_pips, datetimes = load_pair_data(pair)

    # Calculate binary target
    y = calculate_v1_style_target(y_long_pips, y_short_pips)

    n_samples = len(y)
    n_long = (y == 1).sum()
    n_short = (y == 0).sum()
    log(f"   Total samples: {n_samples:,}")
    log(f"   Classes: SHORT={n_short:,} ({100*n_short/n_samples:.1f}%), "
        f"LONG={n_long:,} ({100*n_long/n_samples:.1f}%)")

    # Split data
    X_train, y_train, X_val, y_val, X_test, y_test = time_based_split(X, y)
    log(f"   Train: {len(y_train):,}, Val: {len(y_val):,}, Test: {len(y_test):,}")

    # Class weights for imbalance
    n_train_long = (y_train == 1).sum()
    n_train_short = (y_train == 0).sum()
    class_weights = {
        0: len(y_train) / (2 * n_train_short) if n_train_short > 0 else 1.0,
        1: len(y_train) / (2 * n_train_long) if n_train_long > 0 else 1.0,
    }

    # Create and train model
    log(f"   Training...")
    model = create_lstm_model_binary()

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001, verbose=0),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=0
    )

    epochs_trained = len(history.history['loss'])
    best_val_acc = max(history.history['val_accuracy'])

    # Evaluate
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

    log(f"   Epochs: {epochs_trained}")
    log(f"   Best Val Acc: {best_val_acc:.4f}")
    log(f"   Test Acc: {test_acc:.4f}")

    # Per-class accuracy
    y_pred = (model.predict(X_test, verbose=0) > 0.5).astype(int).flatten()

    short_mask = y_test == 0
    long_mask = y_test == 1

    short_acc = (y_pred[short_mask] == 0).mean() if short_mask.sum() > 0 else 0
    long_acc = (y_pred[long_mask] == 1).mean() if long_mask.sum() > 0 else 0

    log(f"   SHORT accuracy: {short_acc:.4f} ({short_mask.sum():,} samples)")
    log(f"   LONG accuracy: {long_acc:.4f} ({long_mask.sum():,} samples)")

    # Save model
    model_path = RESULTS_DIR / f"lstm_v1style_{pair}.keras"
    model.save(model_path)
    log(f"   Saved: {model_path.name}")

    results.append({
        'pair': pair,
        'val_acc': best_val_acc,
        'test_acc': test_acc,
        'short_acc': short_acc,
        'long_acc': long_acc,
        'epochs': epochs_trained,
        'n_samples': n_samples,
    })

    # Cleanup
    del model, X, y
    keras.backend.clear_session()
    gc.collect()

# ============================================================================
# SUMMARY
# ============================================================================

log(f"\n{'='*70}")
log("EXPERIMENT SUMMARY: V1-Style Binary Target")
log(f"{'='*70}")

log(f"\n{'Pair':<10} {'Val Acc':>10} {'Test Acc':>10} {'SHORT Acc':>10} {'LONG Acc':>10}")
log("-" * 55)

for r in results:
    log(f"{r['pair']:<10} {r['val_acc']:>10.4f} {r['test_acc']:>10.4f} "
        f"{r['short_acc']:>10.4f} {r['long_acc']:>10.4f}")

# Averages
avg_val = np.mean([r['val_acc'] for r in results])
avg_test = np.mean([r['test_acc'] for r in results])
avg_short = np.mean([r['short_acc'] for r in results])
avg_long = np.mean([r['long_acc'] for r in results])

log("-" * 55)
log(f"{'AVERAGE':<10} {avg_val:>10.4f} {avg_test:>10.4f} {avg_short:>10.4f} {avg_long:>10.4f}")

# Save results
results_path = RESULTS_DIR / "exp02_results.pkl"
with open(results_path, 'wb') as f:
    pickle.dump(results, f)

log(f"\nResults saved to: {results_path}")
log(f"\nCompleted: {datetime.now()}")
log("=" * 70)
