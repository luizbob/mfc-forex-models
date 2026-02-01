"""
Experiment 01: Test Different Pip Thresholds
=============================================
Tests 5, 8, 10 pip thresholds instead of 15 pips.
Lower thresholds = more tradeable signals, potentially easier to predict.

This script regenerates targets from price data and trains models.
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
log("EXPERIMENT 01: Different Pip Thresholds")
log("=" * 70)
log(f"Started: {datetime.now()}")

# ============================================================================
# CONFIGURATION
# ============================================================================

V2_DIR = Path(__file__).parent.parent
DATA_DIR = V2_DIR / "data" / "pairs"
EXP_DIR = Path(__file__).parent
RESULTS_DIR = EXP_DIR / "results_exp01"
RESULTS_DIR.mkdir(exist_ok=True)

# Thresholds to test (in pips)
THRESHOLDS = [5, 8, 10]

# Training parameters
BATCH_SIZE = 512
EPOCHS = 50
PATIENCE = 10

# Model architecture
LSTM_UNITS = 64
DENSE_UNITS = 32
DROPOUT = 0.3

# Test on subset of pairs first
TEST_PAIRS = ['EURUSD', 'GBPUSD', 'USDJPY', 'EURJPY']

# Load config
config_path = DATA_DIR / "config_pairs.pkl"
with open(config_path, 'rb') as f:
    config = pickle.load(f)

LOOKBACKS = config['lookbacks']
HORIZON_BARS = config['horizon_bars']  # 96 bars = 8 hours

log(f"\nConfig loaded:")
log(f"   Lookbacks: {LOOKBACKS}")
log(f"   Horizon: {HORIZON_BARS} bars")
log(f"   Thresholds to test: {THRESHOLDS}")
log(f"   Test pairs: {TEST_PAIRS}")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def recalculate_direction(y_long_pips, y_short_pips, threshold):
    """
    Recalculate direction labels based on new threshold.

    The stored y_long_pips and y_short_pips contain the maximum pip movements
    in each direction during the horizon window. We need to determine which
    threshold was hit first.

    Note: This is an approximation since we don't have bar-by-bar data.
    For accurate results, we'd need to recalculate from raw price data.
    """
    n = len(y_long_pips)
    y_direction = np.ones(n, dtype=np.int8)  # Default to NEUTRAL

    for i in range(n):
        long_hit = y_long_pips[i] >= threshold
        short_hit = y_short_pips[i] >= threshold

        if long_hit and short_hit:
            # Both hit - use ratio to estimate which hit first
            # Higher ratio = more likely to hit first
            if y_long_pips[i] > y_short_pips[i]:
                y_direction[i] = 2  # LONG
            else:
                y_direction[i] = 0  # SHORT
        elif long_hit:
            y_direction[i] = 2  # LONG
        elif short_hit:
            y_direction[i] = 0  # SHORT
        # else: stays NEUTRAL

    return y_direction

def create_lstm_model():
    """Create LSTM model for pair prediction."""

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

    output = layers.Dense(3, activation='softmax', name='output')(x)

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
        loss='sparse_categorical_crossentropy',
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

all_results = []

for threshold in THRESHOLDS:
    log(f"\n{'='*70}")
    log(f"TESTING THRESHOLD: {threshold} pips")
    log(f"{'='*70}")

    threshold_results = []

    for pair in TEST_PAIRS:
        log(f"\n{'-'*50}")
        log(f"Training {pair} with {threshold} pip threshold...")
        log(f"{'-'*50}")

        # Load data
        log(f"   Loading data...")
        X, y_long_pips, y_short_pips, datetimes = load_pair_data(pair)

        # Recalculate targets with new threshold
        y = recalculate_direction(y_long_pips, y_short_pips, threshold)

        n_samples = len(y)
        class_counts = np.bincount(y, minlength=3)
        log(f"   Total samples: {n_samples:,}")
        log(f"   Classes: SHORT={class_counts[0]:,} ({100*class_counts[0]/n_samples:.1f}%), "
            f"NEUTRAL={class_counts[1]:,} ({100*class_counts[1]/n_samples:.1f}%), "
            f"LONG={class_counts[2]:,} ({100*class_counts[2]/n_samples:.1f}%)")

        # Split data
        X_train, y_train, X_val, y_val, X_test, y_test = time_based_split(X, y)
        log(f"   Train: {len(y_train):,}, Val: {len(y_val):,}, Test: {len(y_test):,}")

        # Class weights
        class_weights = {}
        total = len(y_train)
        for cls in range(3):
            count = np.sum(y_train == cls)
            class_weights[cls] = total / (3 * count) if count > 0 else 1.0

        # Create and train model
        log(f"   Training...")
        model = create_lstm_model()

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

        log(f"   Epochs: {epochs_trained}, Val Acc: {best_val_acc:.4f}, Test Acc: {test_acc:.4f}")

        # Per-class accuracy
        y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
        for cls, name in enumerate(['SHORT', 'NEUTRAL', 'LONG']):
            mask = y_test == cls
            if mask.sum() > 0:
                cls_acc = (y_pred[mask] == cls).mean()
                log(f"   {name}: {cls_acc:.4f} ({mask.sum():,} samples)")

        threshold_results.append({
            'pair': pair,
            'threshold': threshold,
            'val_acc': best_val_acc,
            'test_acc': test_acc,
            'epochs': epochs_trained,
            'class_dist': class_counts.tolist(),
        })

        # Cleanup
        del model, X, y
        keras.backend.clear_session()
        gc.collect()

    all_results.extend(threshold_results)

# ============================================================================
# SUMMARY
# ============================================================================

log(f"\n{'='*70}")
log("EXPERIMENT SUMMARY")
log(f"{'='*70}")

log(f"\n{'Threshold':<12} {'Pair':<10} {'Val Acc':>10} {'Test Acc':>10}")
log("-" * 50)

for r in all_results:
    log(f"{r['threshold']:<12} {r['pair']:<10} {r['val_acc']:>10.4f} {r['test_acc']:>10.4f}")

# Average by threshold
log(f"\n{'Threshold':<12} {'Avg Val Acc':>12} {'Avg Test Acc':>12}")
log("-" * 40)

for threshold in THRESHOLDS:
    t_results = [r for r in all_results if r['threshold'] == threshold]
    avg_val = np.mean([r['val_acc'] for r in t_results])
    avg_test = np.mean([r['test_acc'] for r in t_results])
    log(f"{threshold:<12} {avg_val:>12.4f} {avg_test:>12.4f}")

# Save results
results_path = RESULTS_DIR / "exp01_results.pkl"
with open(results_path, 'wb') as f:
    pickle.dump(all_results, f)

log(f"\nResults saved to: {results_path}")
log(f"\nCompleted: {datetime.now()}")
log("=" * 70)
