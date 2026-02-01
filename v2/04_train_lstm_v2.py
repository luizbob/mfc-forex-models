"""
V2: Train LSTM Models for Per-Pair Price Prediction
====================================================
Trains one LSTM model per pair to predict price direction:
- 0 = SHORT (price will drop ≥15 pips first)
- 1 = NEUTRAL (neither direction within 8h)
- 2 = LONG (price will rise ≥15 pips first)

Input: MFC sequences from both currencies in the pair
Output: 3-class probability distribution
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import pickle
from pathlib import Path
from datetime import datetime
import gc

# Suppress TF warnings
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

def log(msg=""):
    print(msg, flush=True)

log("=" * 70)
log("V2: TRAIN LSTM MODELS (Per-Pair)")
log("=" * 70)
log(f"Started: {datetime.now()}")

# ============================================================================
# CONFIGURATION
# ============================================================================

V2_DIR = Path(__file__).parent
DATA_DIR = V2_DIR / "data" / "pairs"
MODEL_DIR = V2_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

# Training parameters
BATCH_SIZE = 512
EPOCHS = 50
PATIENCE = 10
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.10  # Hold out last 10% by time for final evaluation

# Model architecture
LSTM_UNITS = 64
DENSE_UNITS = 32
DROPOUT = 0.3

# Pairs to train (can be subset for testing)
PAIRS_TO_TRAIN = None  # None = all pairs, or list like ['EURUSD', 'GBPUSD']

# ============================================================================
# LOAD CONFIG
# ============================================================================

config_path = DATA_DIR / "config_pairs.pkl"
with open(config_path, 'rb') as f:
    config = pickle.load(f)

LOOKBACKS = config['lookbacks']
PAIRS = config['pairs'] if PAIRS_TO_TRAIN is None else PAIRS_TO_TRAIN

log(f"\nConfig loaded:")
log(f"   Lookbacks: {LOOKBACKS}")
log(f"   Pairs to train: {len(PAIRS)}")

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

def create_lstm_model():
    """
    Create LSTM model for pair prediction.

    Inputs:
    - 5 MFC sequences for base currency (M5, M15, M30, H1, H4)
    - 5 MFC sequences for quote currency
    - Auxiliary features

    Output:
    - 3-class probability (SHORT, NEUTRAL, LONG)
    """

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

    # Process base currency sequences with shared LSTM
    lstm_m5 = layers.LSTM(LSTM_UNITS, return_sequences=False, name='lstm_m5')
    lstm_m15 = layers.LSTM(LSTM_UNITS, return_sequences=False, name='lstm_m15')
    lstm_m30 = layers.LSTM(LSTM_UNITS, return_sequences=False, name='lstm_m30')
    lstm_h1 = layers.LSTM(LSTM_UNITS, return_sequences=False, name='lstm_h1')
    lstm_h4 = layers.LSTM(LSTM_UNITS, return_sequences=False, name='lstm_h4')

    # Base currency LSTM outputs
    base_m5_out = lstm_m5(input_base_m5)
    base_m15_out = lstm_m15(input_base_m15)
    base_m30_out = lstm_m30(input_base_m30)
    base_h1_out = lstm_h1(input_base_h1)
    base_h4_out = lstm_h4(input_base_h4)

    # Quote currency LSTM outputs (same LSTM layers - shared weights)
    quote_m5_out = lstm_m5(input_quote_m5)
    quote_m15_out = lstm_m15(input_quote_m15)
    quote_m30_out = lstm_m30(input_quote_m30)
    quote_h1_out = lstm_h1(input_quote_h1)
    quote_h4_out = lstm_h4(input_quote_h4)

    # Concatenate all outputs
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

    # Output layer - 3 classes
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

# ============================================================================
# DATA PREPARATION
# ============================================================================

def load_pair_data(pair):
    """Load and prepare data for a pair."""
    path = DATA_DIR / f"v2_pair_{pair}.pkl"

    with open(path, 'rb') as f:
        data = pickle.load(f)

    # Prepare inputs (add channel dimension for LSTM)
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

    y = data['y_direction']  # 0=SHORT, 1=NEUTRAL, 2=LONG

    return X, y, data['datetimes']

def time_based_split(X, y, datetimes, test_ratio=0.1, val_ratio=0.15):
    """
    Split data by time to avoid lookahead bias.
    Last test_ratio for test, then val_ratio of remaining for validation.
    """
    n = len(y)
    test_idx = int(n * (1 - test_ratio))

    # Test set (last portion)
    X_test = {k: v[test_idx:] for k, v in X.items()}
    y_test = y[test_idx:]

    # Train+val (earlier portion)
    X_trainval = {k: v[:test_idx] for k, v in X.items()}
    y_trainval = y[:test_idx]

    # Split train/val
    n_trainval = len(y_trainval)
    val_idx = int(n_trainval * (1 - val_ratio))

    X_train = {k: v[:val_idx] for k, v in X_trainval.items()}
    y_train = y_trainval[:val_idx]

    X_val = {k: v[val_idx:] for k, v in X_trainval.items()}
    y_val = y_trainval[val_idx:]

    return X_train, y_train, X_val, y_val, X_test, y_test

# ============================================================================
# TRAINING
# ============================================================================

results = []

for pair in PAIRS:
    log(f"\n{'='*60}")
    log(f"Training {pair}...")
    log(f"{'='*60}")

    # Load data
    log(f"   Loading data...")
    X, y, datetimes = load_pair_data(pair)

    n_samples = len(y)
    class_counts = np.bincount(y, minlength=3)
    log(f"   Total samples: {n_samples:,}")
    log(f"   Class distribution: SHORT={class_counts[0]:,}, NEUTRAL={class_counts[1]:,}, LONG={class_counts[2]:,}")

    # Split data by time
    log(f"   Splitting data (by time)...")
    X_train, y_train, X_val, y_val, X_test, y_test = time_based_split(
        X, y, datetimes, test_ratio=TEST_SPLIT, val_ratio=VALIDATION_SPLIT
    )

    log(f"   Train: {len(y_train):,}, Val: {len(y_val):,}, Test: {len(y_test):,}")

    # Calculate class weights to handle imbalance
    class_weights = {}
    total = len(y_train)
    for cls in range(3):
        count = np.sum(y_train == cls)
        if count > 0:
            class_weights[cls] = total / (3 * count)
        else:
            class_weights[cls] = 1.0

    log(f"   Class weights: {class_weights}")

    # Create model
    log(f"   Creating model...")
    model = create_lstm_model()

    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=0
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.0001,
            verbose=0
        ),
    ]

    # Train
    log(f"   Training...")
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
    best_val_loss = min(history.history['val_loss'])
    best_val_acc = max(history.history['val_accuracy'])

    log(f"   Epochs trained: {epochs_trained}")
    log(f"   Best val loss: {best_val_loss:.4f}")
    log(f"   Best val accuracy: {best_val_acc:.4f}")

    # Evaluate on test set
    log(f"   Evaluating on test set...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    log(f"   Test loss: {test_loss:.4f}")
    log(f"   Test accuracy: {test_acc:.4f}")

    # Detailed test metrics
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Per-class accuracy
    for cls, name in enumerate(['SHORT', 'NEUTRAL', 'LONG']):
        mask = y_test == cls
        if mask.sum() > 0:
            cls_acc = (y_pred_classes[mask] == cls).mean()
            log(f"   {name} accuracy: {cls_acc:.4f} ({mask.sum():,} samples)")

    # Save model
    model_path = MODEL_DIR / f"lstm_v2_{pair}.keras"
    model.save(model_path)
    log(f"   Saved: {model_path.name}")

    # Store results
    results.append({
        'pair': pair,
        'samples': n_samples,
        'epochs': epochs_trained,
        'val_loss': best_val_loss,
        'val_acc': best_val_acc,
        'test_loss': test_loss,
        'test_acc': test_acc,
    })

    # Free memory
    del model, X, y, X_train, y_train, X_val, y_val, X_test, y_test
    keras.backend.clear_session()
    gc.collect()

# ============================================================================
# SUMMARY
# ============================================================================

log(f"\n{'='*70}")
log("TRAINING SUMMARY")
log(f"{'='*70}")

log(f"\n{'Pair':<10} {'Samples':>10} {'Epochs':>8} {'Val Acc':>10} {'Test Acc':>10}")
log("-" * 55)

for r in results:
    log(f"{r['pair']:<10} {r['samples']:>10,} {r['epochs']:>8} {r['val_acc']:>10.4f} {r['test_acc']:>10.4f}")

# Calculate averages
avg_val_acc = np.mean([r['val_acc'] for r in results])
avg_test_acc = np.mean([r['test_acc'] for r in results])
log("-" * 55)
log(f"{'AVERAGE':<10} {'':<10} {'':<8} {avg_val_acc:>10.4f} {avg_test_acc:>10.4f}")

# Save results
results_path = MODEL_DIR / "training_results.pkl"
with open(results_path, 'wb') as f:
    pickle.dump(results, f)

log(f"\nResults saved to: {results_path}")
log(f"\nCompleted: {datetime.now()}")
log("=" * 70)
