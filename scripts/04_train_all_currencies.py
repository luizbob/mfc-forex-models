"""
Script 04: Train LSTM Models for All 8 Currencies
==================================================
Trains separate models for: EUR, USD, GBP, JPY, CHF, CAD, AUD, NZD
"""
import warnings
warnings.filterwarnings('ignore')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pickle
from pathlib import Path
from datetime import datetime

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def log(msg=""):
    print(msg, flush=True)

log("=" * 70)
log("TRAIN ALL CURRENCY MODELS")
log("=" * 70)
log(f"Started: {datetime.now()}")

# Paths
DATA_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/lstm_model/data')
MODEL_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/lstm_model/models')
MODEL_DIR.mkdir(exist_ok=True, parents=True)

# Load config
with open(DATA_DIR / 'config.pkl', 'rb') as f:
    config = pickle.load(f)

LOOKBACK = config['lookback']
CURRENCIES = config['currencies']

log(f"\nCurrencies to train: {CURRENCIES}")

def build_model(lookback_dict, lstm_units=64, dense_units=32):
    """Build multi-timeframe LSTM model."""
    input_m5 = layers.Input(shape=(lookback_dict['M5'], 1), name='input_M5')
    input_m15 = layers.Input(shape=(lookback_dict['M15'], 1), name='input_M15')
    input_m30 = layers.Input(shape=(lookback_dict['M30'], 1), name='input_M30')
    input_h1 = layers.Input(shape=(lookback_dict['H1'], 1), name='input_H1')
    input_h4 = layers.Input(shape=(lookback_dict['H4'], 1), name='input_H4')
    input_aux = layers.Input(shape=(5,), name='input_aux')

    x_m5 = layers.LSTM(lstm_units, return_sequences=True)(input_m5)
    x_m5 = layers.LSTM(lstm_units // 2)(x_m5)

    x_m15 = layers.LSTM(lstm_units, return_sequences=True)(input_m15)
    x_m15 = layers.LSTM(lstm_units // 2)(x_m15)

    x_m30 = layers.LSTM(lstm_units, return_sequences=True)(input_m30)
    x_m30 = layers.LSTM(lstm_units // 2)(x_m30)

    x_h1 = layers.LSTM(lstm_units // 2)(input_h1)
    x_h4 = layers.LSTM(lstm_units // 2)(input_h4)

    concat = layers.Concatenate()([x_m5, x_m15, x_m30, x_h1, x_h4, input_aux])

    x = layers.Dense(dense_units * 2, activation='relu')(concat)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(dense_units, activation='relu')(x)
    x = layers.Dropout(0.2)(x)

    output_direction = layers.Dense(3, activation='softmax', name='direction')(x)
    output_completed = layers.Dense(1, activation='sigmoid', name='completed')(x)

    return Model(
        inputs=[input_m5, input_m15, input_m30, input_h1, input_h4, input_aux],
        outputs=[output_direction, output_completed]
    )


def load_currency_data(currency):
    """Load and prepare data for one currency."""
    with open(DATA_DIR / f'lstm_data_{currency}.pkl', 'rb') as f:
        data = pickle.load(f)

    X_M5 = data['X_M5'].reshape(-1, LOOKBACK['M5'], 1)
    X_M15 = data['X_M15'].reshape(-1, LOOKBACK['M15'], 1)
    X_M30 = data['X_M30'].reshape(-1, LOOKBACK['M30'], 1)
    X_H1 = data['X_H1'].reshape(-1, LOOKBACK['H1'], 1)
    X_H4 = data['X_H4'].reshape(-1, LOOKBACK['H4'], 1)
    X_aux = data['X_aux']

    y_direction = data['y_direction'].astype(int) + 1  # -1->0, 0->1, 1->2
    y_completed = data['y_completed']

    return {
        'X': [X_M5, X_M15, X_M30, X_H1, X_H4, X_aux],
        'y_direction': y_direction,
        'y_completed': y_completed,
    }


# Train each currency
results = {}

for ccy in CURRENCIES:
    log(f"\n{'=' * 70}")
    log(f"Training {ccy}")
    log(f"{'=' * 70}")

    # Check if already trained
    model_path = MODEL_DIR / f'lstm_{ccy}_final.keras'
    if model_path.exists():
        log(f"  Model already exists, skipping...")
        results[ccy] = {'status': 'skipped'}
        continue

    # Load data
    log(f"  Loading data...")
    data = load_currency_data(ccy)
    n_samples = len(data['y_direction'])

    # Train/val split
    split_idx = int(n_samples * 0.8)
    X_train = [x[:split_idx] for x in data['X']]
    X_val = [x[split_idx:] for x in data['X']]
    y_train_dir = data['y_direction'][:split_idx]
    y_val_dir = data['y_direction'][split_idx:]
    y_train_comp = data['y_completed'][:split_idx]
    y_val_comp = data['y_completed'][split_idx:]

    log(f"  Train: {split_idx}, Val: {n_samples - split_idx}")

    # Build model
    log(f"  Building model...")
    tf.keras.backend.clear_session()
    model = build_model(LOOKBACK)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss={
            'direction': 'sparse_categorical_crossentropy',
            'completed': 'binary_crossentropy'
        },
        loss_weights={'direction': 1.0, 'completed': 0.5},
        metrics={'direction': 'accuracy', 'completed': 'accuracy'}
    )

    callbacks = [
        EarlyStopping(
            monitor='val_direction_accuracy',
            patience=10,
            restore_best_weights=True,
            mode='max'
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        ),
    ]

    # Train
    log(f"  Training...")
    history = model.fit(
        X_train,
        [y_train_dir, y_train_comp],
        validation_data=(X_val, [y_val_dir, y_val_comp]),
        epochs=50,
        batch_size=256,
        callbacks=callbacks,
        verbose=0
    )

    # Evaluate
    val_results = model.evaluate(X_val, [y_val_dir, y_val_comp], verbose=0)
    dir_acc = val_results[3] * 100
    comp_acc = val_results[4] * 100

    log(f"  Direction Accuracy: {dir_acc:.1f}%")
    log(f"  Completed Accuracy: {comp_acc:.1f}%")

    # Save
    model.save(MODEL_DIR / f'lstm_{ccy}_final.keras')
    log(f"  Saved to {MODEL_DIR / f'lstm_{ccy}_final.keras'}")

    results[ccy] = {
        'status': 'trained',
        'direction_acc': dir_acc,
        'completed_acc': comp_acc,
    }

    # Cleanup
    del model, data, X_train, X_val
    tf.keras.backend.clear_session()

# Summary
log(f"\n{'=' * 70}")
log("SUMMARY")
log(f"{'=' * 70}")

for ccy, res in results.items():
    if res['status'] == 'trained':
        log(f"  {ccy}: {res['direction_acc']:.1f}% direction, {res['completed_acc']:.1f}% completed")
    else:
        log(f"  {ccy}: {res['status']}")

log(f"\nCompleted: {datetime.now()}")
