"""
Experiment 03: Add Technical Indicators
=======================================
Adds RSI, MACD, and Moving Averages to the model inputs.
Technical indicators might help the model identify momentum and trend.

Indicators added:
- RSI (14 period)
- MACD (12, 26, 9)
- SMA 20, SMA 50
- Price position relative to MAs
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import pandas as pd
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
log("EXPERIMENT 03: Technical Indicators")
log("=" * 70)
log(f"Started: {datetime.now()}")

# ============================================================================
# CONFIGURATION
# ============================================================================

V1_DATA_DIR = Path("/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/data")
V2_DIR = Path(__file__).parent.parent
DATA_DIR = V2_DIR / "data" / "pairs"
EXP_DIR = Path(__file__).parent
RESULTS_DIR = EXP_DIR / "results_exp03"
RESULTS_DIR.mkdir(exist_ok=True)

# Training parameters
BATCH_SIZE = 512
EPOCHS = 50
PATIENCE = 10

# Model architecture
LSTM_UNITS = 64
DENSE_UNITS = 32
DROPOUT = 0.3

# Technical indicator parameters
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
SMA_SHORT = 20
SMA_LONG = 50

# Test on subset of pairs first
TEST_PAIRS = ['EURUSD', 'GBPUSD', 'USDJPY', 'EURJPY']

# Load config
config_path = DATA_DIR / "config_pairs.pkl"
with open(config_path, 'rb') as f:
    config = pickle.load(f)

LOOKBACKS = config['lookbacks']

log(f"\nConfig loaded:")
log(f"   Lookbacks: {LOOKBACKS}")
log(f"   Test pairs: {TEST_PAIRS}")
log(f"   Technical indicators: RSI({RSI_PERIOD}), MACD({MACD_FAST},{MACD_SLOW},{MACD_SIGNAL}), "
    f"SMA({SMA_SHORT},{SMA_LONG})")

# ============================================================================
# TECHNICAL INDICATOR FUNCTIONS
# ============================================================================

def calculate_rsi(prices, period=14):
    """Calculate RSI indicator."""
    delta = np.diff(prices)
    gains = np.where(delta > 0, delta, 0)
    losses = np.where(delta < 0, -delta, 0)

    avg_gain = np.zeros(len(prices))
    avg_loss = np.zeros(len(prices))

    # First average
    avg_gain[period] = np.mean(gains[:period])
    avg_loss[period] = np.mean(losses[:period])

    # Smoothed averages
    for i in range(period + 1, len(prices)):
        avg_gain[i] = (avg_gain[i-1] * (period-1) + gains[i-1]) / period
        avg_loss[i] = (avg_loss[i-1] * (period-1) + losses[i-1]) / period

    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
    rsi = 100 - (100 / (1 + rs))
    rsi[:period] = 50  # Fill initial values

    return rsi

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD indicator."""
    # EMA calculation
    def ema(data, period):
        alpha = 2 / (period + 1)
        result = np.zeros(len(data))
        result[0] = data[0]
        for i in range(1, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result

    ema_fast = ema(prices, fast)
    ema_slow = ema(prices, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram

def calculate_sma(prices, period):
    """Calculate Simple Moving Average."""
    sma = np.zeros(len(prices))
    for i in range(period - 1, len(prices)):
        sma[i] = np.mean(prices[i-period+1:i+1])
    sma[:period-1] = sma[period-1]  # Fill initial values
    return sma

def load_price_data(pair):
    """Load M5 price data for a pair."""
    path = V1_DATA_DIR / f"{pair}_GMT+0_US-DST_M1.csv"

    if not path.exists():
        return None, None

    df = pd.read_csv(path, usecols=['Date', 'Time', 'Open', 'High', 'Low', 'Close'])
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df = df.set_index('datetime').sort_index()

    # Resample to M5
    df_m5 = df.resample('5min').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last'
    }).dropna()

    return df_m5, df_m5.index.values

def calculate_technical_indicators(df):
    """Calculate all technical indicators for a price dataframe."""
    close = df['Close'].values

    # RSI
    rsi = calculate_rsi(close, RSI_PERIOD)

    # MACD
    macd_line, macd_signal, macd_hist = calculate_macd(close, MACD_FAST, MACD_SLOW, MACD_SIGNAL)

    # SMAs
    sma_short = calculate_sma(close, SMA_SHORT)
    sma_long = calculate_sma(close, SMA_LONG)

    # Price position relative to MAs (normalized)
    price_vs_sma_short = (close - sma_short) / sma_short * 100
    price_vs_sma_long = (close - sma_long) / sma_long * 100

    # Normalize indicators
    rsi_norm = (rsi - 50) / 50  # Scale to -1 to 1
    macd_norm = macd_hist / (np.abs(macd_hist).max() + 1e-8)  # Normalize

    return {
        'rsi': rsi_norm,
        'macd_hist': macd_norm,
        'price_vs_sma_short': np.clip(price_vs_sma_short / 5, -1, 1),  # Clip to ±1
        'price_vs_sma_long': np.clip(price_vs_sma_long / 10, -1, 1),
    }

# ============================================================================
# MODEL WITH TECHNICAL INDICATORS
# ============================================================================

def create_lstm_model_with_ti():
    """Create LSTM model with technical indicator inputs."""

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

    # Auxiliary input (original 6 + 4 technical indicators = 10)
    input_aux = layers.Input(shape=(10,), name='aux')

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

    # Dense layers (slightly larger to handle additional inputs)
    x = layers.Dense(DENSE_UNITS * 2, activation='relu')(combined)
    x = layers.Dropout(DROPOUT)(x)
    x = layers.Dense(DENSE_UNITS, activation='relu')(x)
    x = layers.Dropout(DROPOUT)(x)

    # 3-class output
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

def load_pair_data_with_ti(pair):
    """Load data for a pair and add technical indicators."""
    # Load original data
    path = DATA_DIR / f"v2_pair_{pair}.pkl"

    with open(path, 'rb') as f:
        data = pickle.load(f)

    # Load price data
    log(f"   Loading price data for technical indicators...")
    df_price, price_times = load_price_data(pair)

    if df_price is None:
        log(f"   WARNING: No price data for {pair}")
        return None, None, None

    # Calculate technical indicators
    log(f"   Calculating technical indicators...")
    ti = calculate_technical_indicators(df_price)

    # Create time index mapping
    datetimes = data['datetimes']
    n_samples = len(datetimes)

    # Map datetimes to price index
    price_time_idx = {t: i for i, t in enumerate(price_times)}

    # Build extended aux features
    X_aux_extended = np.zeros((n_samples, 10), dtype=np.float32)
    valid_mask = np.ones(n_samples, dtype=bool)

    for i, dt in enumerate(datetimes):
        if dt in price_time_idx:
            p_idx = price_time_idx[dt]

            # Original 6 aux features
            X_aux_extended[i, :6] = data['X_aux'][i]

            # Add 4 technical indicators
            X_aux_extended[i, 6] = ti['rsi'][p_idx]
            X_aux_extended[i, 7] = ti['macd_hist'][p_idx]
            X_aux_extended[i, 8] = ti['price_vs_sma_short'][p_idx]
            X_aux_extended[i, 9] = ti['price_vs_sma_long'][p_idx]
        else:
            valid_mask[i] = False

    # Filter to valid samples
    log(f"   Valid samples with TI: {valid_mask.sum():,} / {n_samples:,}")

    X = {
        'base_m5': data['X_base_M5'][valid_mask][..., np.newaxis],
        'base_m15': data['X_base_M15'][valid_mask][..., np.newaxis],
        'base_m30': data['X_base_M30'][valid_mask][..., np.newaxis],
        'base_h1': data['X_base_H1'][valid_mask][..., np.newaxis],
        'base_h4': data['X_base_H4'][valid_mask][..., np.newaxis],
        'quote_m5': data['X_quote_M5'][valid_mask][..., np.newaxis],
        'quote_m15': data['X_quote_M15'][valid_mask][..., np.newaxis],
        'quote_m30': data['X_quote_M30'][valid_mask][..., np.newaxis],
        'quote_h1': data['X_quote_H1'][valid_mask][..., np.newaxis],
        'quote_h4': data['X_quote_H4'][valid_mask][..., np.newaxis],
        'aux': X_aux_extended[valid_mask],
    }

    y = data['y_direction'][valid_mask]

    return X, y, datetimes[valid_mask]

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
    log(f"Training {pair} with Technical Indicators...")
    log(f"{'='*60}")

    # Load data with technical indicators
    log(f"   Loading data...")
    X, y, datetimes = load_pair_data_with_ti(pair)

    if X is None:
        log(f"   SKIPPING: Could not load data")
        continue

    n_samples = len(y)
    class_counts = np.bincount(y, minlength=3)
    log(f"   Total samples: {n_samples:,}")
    log(f"   Classes: SHORT={class_counts[0]:,}, NEUTRAL={class_counts[1]:,}, LONG={class_counts[2]:,}")

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
    model = create_lstm_model_with_ti()

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
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    for cls, name in enumerate(['SHORT', 'NEUTRAL', 'LONG']):
        mask = y_test == cls
        if mask.sum() > 0:
            cls_acc = (y_pred[mask] == cls).mean()
            log(f"   {name}: {cls_acc:.4f} ({mask.sum():,} samples)")

    # Save model
    model_path = RESULTS_DIR / f"lstm_ti_{pair}.keras"
    model.save(model_path)
    log(f"   Saved: {model_path.name}")

    results.append({
        'pair': pair,
        'val_acc': best_val_acc,
        'test_acc': test_acc,
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
log("EXPERIMENT SUMMARY: Technical Indicators")
log(f"{'='*70}")

log(f"\n{'Pair':<10} {'Val Acc':>10} {'Test Acc':>10} {'Epochs':>8}")
log("-" * 45)

for r in results:
    log(f"{r['pair']:<10} {r['val_acc']:>10.4f} {r['test_acc']:>10.4f} {r['epochs']:>8}")

if results:
    avg_val = np.mean([r['val_acc'] for r in results])
    avg_test = np.mean([r['test_acc'] for r in results])
    log("-" * 45)
    log(f"{'AVERAGE':<10} {avg_val:>10.4f} {avg_test:>10.4f}")

# Save results
results_path = RESULTS_DIR / "exp03_results.pkl"
with open(results_path, 'wb') as f:
    pickle.dump(results, f)

log(f"\nResults saved to: {results_path}")
log(f"\nCompleted: {datetime.now()}")
log("=" * 70)
