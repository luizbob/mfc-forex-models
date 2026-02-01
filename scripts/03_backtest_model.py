"""
Script 03: Backtest LSTM Model
==============================
Tests if LSTM predictions translate to profitable trades.
"""
import warnings
warnings.filterwarnings('ignore')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU

# Load TensorFlow FIRST before other large allocations
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from datetime import datetime
import gc

def log(msg=""):
    print(msg, flush=True)

log("=" * 70)
log("LSTM MODEL BACKTEST")
log("=" * 70)
log(f"Started: {datetime.now()}")

# Paths
DATA_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/data')
LSTM_DATA_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/lstm_model/data')
MODEL_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/lstm_model/models')

# Load config
with open(LSTM_DATA_DIR / 'config.pkl', 'rb') as f:
    config = pickle.load(f)

LOOKBACK = config['lookback']

# ============================================================================
# 1. LOAD MODEL FIRST
# ============================================================================
log("\n1. Loading model...")
model = tf.keras.models.load_model(MODEL_DIR / 'lstm_EUR_final.keras')
log("  Model loaded")

# ============================================================================
# 2. LOAD DATA
# ============================================================================
log("\n2. Loading data...")

# Load EUR LSTM data
with open(LSTM_DATA_DIR / 'lstm_data_EUR.pkl', 'rb') as f:
    eur_data = pickle.load(f)

# Load USD LSTM data
with open(LSTM_DATA_DIR / 'lstm_data_USD.pkl', 'rb') as f:
    usd_data = pickle.load(f)

n_samples = len(eur_data['datetimes'])
split_idx = int(n_samples * 0.8)
val_datetimes = eur_data['datetimes'][split_idx:]

log(f"  Validation: {val_datetimes[0]} to {val_datetimes[-1]}")
log(f"  Samples: {len(val_datetimes)}")

# ============================================================================
# 3. GENERATE PREDICTIONS
# ============================================================================
log("\n3. Generating predictions...")

# Prepare validation data
X_val = [
    eur_data['X_M5'][split_idx:].reshape(-1, LOOKBACK['M5'], 1),
    eur_data['X_M15'][split_idx:].reshape(-1, LOOKBACK['M15'], 1),
    eur_data['X_M30'][split_idx:].reshape(-1, LOOKBACK['M30'], 1),
    eur_data['X_H1'][split_idx:].reshape(-1, LOOKBACK['H1'], 1),
    eur_data['X_H4'][split_idx:].reshape(-1, LOOKBACK['H4'], 1),
    eur_data['X_aux'][split_idx:],
]

# Predict
eur_pred = model.predict(X_val, verbose=0, batch_size=256)
eur_direction = np.argmax(eur_pred[0], axis=1)  # 0=DOWN, 1=NEUTRAL, 2=UP
eur_conf = np.max(eur_pred[0], axis=1)

log(f"  DOWN={np.sum(eur_direction==0)}, NEUTRAL={np.sum(eur_direction==1)}, UP={np.sum(eur_direction==2)}")

# USD current MFC (from aux features)
usd_mfc = usd_data['X_aux'][split_idx:, 2]

# Clear unused data
del model, X_val, eur_pred, eur_data
gc.collect()

# ============================================================================
# 4. LOAD PRICE DATA
# ============================================================================
log("\n4. Loading price data...")

PAIR = 'EURUSD'
SPREAD = 1.5
PIP_VAL = 0.0001

chunks = []
for chunk in pd.read_csv(DATA_DIR / f'{PAIR}_GMT+0_US-DST_M1.csv', chunksize=500000):
    chunk['datetime'] = pd.to_datetime(chunk['Date'] + ' ' + chunk['Time'], format='%Y.%m.%d %H:%M:%S')
    chunk = chunk.set_index('datetime')
    chunk = chunk[(chunk.index >= '2024-07-01') & (chunk.index <= '2024-12-31')]
    if len(chunk) > 0:
        m5_chunk = chunk[['Close']].resample('5min').last().dropna()
        chunks.append(m5_chunk)

price_df = pd.concat(chunks)
log(f"  Price bars: {len(price_df)}")

del chunks, usd_data
gc.collect()

# ============================================================================
# 5. GENERATE TRADES
# ============================================================================
log("\n5. Generating trades...")

trades = []

for i in range(len(val_datetimes)):
    dt = pd.Timestamp(val_datetimes[i])

    if dt not in price_df.index:
        continue

    eur_dir = eur_direction[i]
    conf = eur_conf[i]

    # Skip neutral or low confidence
    if eur_dir == 1 or conf < 0.6:
        continue

    usd_val = usd_mfc[i]

    # Trading logic
    signal = None
    if eur_dir == 2 and usd_val <= 0.3:  # EUR UP + USD weak = BUY
        signal = 'BUY'
    elif eur_dir == 0 and usd_val >= -0.3:  # EUR DOWN + USD not weak = SELL
        signal = 'SELL'

    if signal is None:
        continue

    price_idx = price_df.index.get_loc(dt)
    entry_price = price_df.iloc[price_idx]['Close']

    exit_idx = min(price_idx + 72, len(price_df) - 1)
    exit_price = price_df.iloc[exit_idx]['Close']

    if signal == 'BUY':
        pips = (exit_price - entry_price) / PIP_VAL
    else:
        pips = (entry_price - exit_price) / PIP_VAL

    trades.append({
        'datetime': dt,
        'signal': signal,
        'conf': conf,
        'pips': pips,
    })

# ============================================================================
# 6. RESULTS
# ============================================================================
log("\n" + "=" * 70)
log("RESULTS - EURUSD")
log("=" * 70)

if len(trades) == 0:
    log("\nNo trades generated!")
else:
    trades_df = pd.DataFrame(trades)
    trades_df['net_pips'] = trades_df['pips'] - SPREAD
    trades_df['win'] = (trades_df['pips'] > 0).astype(int)

    log(f"\nTotal trades: {len(trades_df)}")
    log(f"Win rate: {trades_df['win'].mean()*100:.1f}%")
    log(f"Avg pips: {trades_df['pips'].mean():.2f}")
    log(f"Avg net pips: {trades_df['net_pips'].mean():.2f}")
    log(f"Total net pips: {trades_df['net_pips'].sum():.0f}")

    log(f"\nBy Signal:")
    for sig in ['BUY', 'SELL']:
        sig_df = trades_df[trades_df['signal'] == sig]
        if len(sig_df) > 0:
            wr = sig_df['win'].mean() * 100
            avg_net = sig_df['net_pips'].mean()
            log(f"  {sig}: {len(sig_df)} trades, {wr:.1f}% WR, {avg_net:.2f} net avg")

    log(f"\nBy Confidence:")
    for conf_min in [0.7, 0.8, 0.9]:
        conf_df = trades_df[trades_df['conf'] >= conf_min]
        if len(conf_df) > 0:
            wr = conf_df['win'].mean() * 100
            avg_net = conf_df['net_pips'].mean()
            log(f"  conf>={conf_min}: {len(conf_df)} trades, {wr:.1f}% WR, {avg_net:.2f} net avg")

log(f"\nCompleted: {datetime.now()}")
