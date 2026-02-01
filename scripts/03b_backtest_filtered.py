"""
Script 03b: Backtest LSTM Model with V1.5-style Filters
========================================================
Combines LSTM predictions with MFC extreme conditions (like V1.5)
"""
import warnings
warnings.filterwarnings('ignore')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

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
log("LSTM BACKTEST WITH V1.5-STYLE FILTERS")
log("=" * 70)
log(f"Started: {datetime.now()}")

# Paths
DATA_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/data')
LSTM_DATA_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/lstm_model/data')
MODEL_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/lstm_model/models')

with open(LSTM_DATA_DIR / 'config.pkl', 'rb') as f:
    config = pickle.load(f)

LOOKBACK = config['lookback']

# ============================================================================
# 1. LOAD MODEL
# ============================================================================
log("\n1. Loading model...")
model = tf.keras.models.load_model(MODEL_DIR / 'lstm_EUR_final.keras')
log("  Model loaded")

# ============================================================================
# 2. LOAD DATA
# ============================================================================
log("\n2. Loading data...")

with open(LSTM_DATA_DIR / 'lstm_data_EUR.pkl', 'rb') as f:
    eur_data = pickle.load(f)

with open(LSTM_DATA_DIR / 'lstm_data_USD.pkl', 'rb') as f:
    usd_data = pickle.load(f)

n_samples = len(eur_data['datetimes'])
split_idx = int(n_samples * 0.8)
val_datetimes = eur_data['datetimes'][split_idx:]

log(f"  Validation: {val_datetimes[0]} to {val_datetimes[-1]}")

# ============================================================================
# 3. GENERATE PREDICTIONS
# ============================================================================
log("\n3. Generating predictions...")

X_val = [
    eur_data['X_M5'][split_idx:].reshape(-1, LOOKBACK['M5'], 1),
    eur_data['X_M15'][split_idx:].reshape(-1, LOOKBACK['M15'], 1),
    eur_data['X_M30'][split_idx:].reshape(-1, LOOKBACK['M30'], 1),
    eur_data['X_H1'][split_idx:].reshape(-1, LOOKBACK['H1'], 1),
    eur_data['X_H4'][split_idx:].reshape(-1, LOOKBACK['H4'], 1),
    eur_data['X_aux'][split_idx:],
]

eur_pred = model.predict(X_val, verbose=0, batch_size=256)
eur_direction = np.argmax(eur_pred[0], axis=1)  # 0=DOWN, 1=NEUTRAL, 2=UP
eur_conf = np.max(eur_pred[0], axis=1)

log(f"  DOWN={np.sum(eur_direction==0)}, NEUTRAL={np.sum(eur_direction==1)}, UP={np.sum(eur_direction==2)}")

# Get MFC values from aux features
# X_aux columns: vel_M5, vel_M30, current_M5, current_M30, current_H4
eur_mfc_m5 = eur_data['X_aux'][split_idx:, 2]
eur_mfc_m30 = eur_data['X_aux'][split_idx:, 3]
eur_mfc_h4 = eur_data['X_aux'][split_idx:, 4]
eur_vel_m5 = eur_data['X_aux'][split_idx:, 0]

usd_mfc_m5 = usd_data['X_aux'][split_idx:, 2]
usd_mfc_m30 = usd_data['X_aux'][split_idx:, 3]
usd_mfc_h4 = usd_data['X_aux'][split_idx:, 4]
usd_vel_m5 = usd_data['X_aux'][split_idx:, 0]

del model, X_val, eur_pred, eur_data, usd_data
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

del chunks
gc.collect()

# ============================================================================
# 5. GENERATE FILTERED TRADES
# ============================================================================
log("\n5. Generating trades with V1.5-style filters...")

# V1.5 thresholds
MFC_THRESHOLD = 0.5  # Base at extreme
QUOTE_THRESHOLD = 0.3  # Quote not blocking

trades = []

for i in range(len(val_datetimes)):
    dt = pd.Timestamp(val_datetimes[i])

    if dt not in price_df.index:
        continue

    eur_dir = eur_direction[i]
    conf = eur_conf[i]

    # Skip neutral
    if eur_dir == 1:
        continue

    # LSTM must be high confidence
    if conf < 0.7:
        continue

    # Get MFC values
    eur_m5 = eur_mfc_m5[i]
    eur_m30 = eur_mfc_m30[i]
    usd_m5 = usd_mfc_m5[i]
    usd_vel = usd_vel_m5[i]

    signal = None

    # BUY: LSTM predicts EUR UP + V1.5-style conditions
    if eur_dir == 2:  # EUR predicted UP
        # V1.5 BUY: EUR at low extreme, USD not too high, USD falling
        if eur_m5 <= -MFC_THRESHOLD and usd_m5 <= QUOTE_THRESHOLD and usd_vel < 0:
            signal = 'BUY'

    # SELL: LSTM predicts EUR DOWN + V1.5-style conditions
    elif eur_dir == 0:  # EUR predicted DOWN
        # V1.5 SELL: EUR at high extreme, USD not too low, USD rising
        if eur_m5 >= MFC_THRESHOLD and usd_m5 >= -QUOTE_THRESHOLD and usd_vel > 0:
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
        'eur_mfc': eur_m5,
        'usd_mfc': usd_m5,
        'pips': pips,
    })

# ============================================================================
# 6. RESULTS
# ============================================================================
log("\n" + "=" * 70)
log("RESULTS - EURUSD (WITH V1.5 FILTERS)")
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
    for conf_min in [0.75, 0.80, 0.85, 0.90]:
        conf_df = trades_df[trades_df['conf'] >= conf_min]
        if len(conf_df) > 0:
            wr = conf_df['win'].mean() * 100
            avg_net = conf_df['net_pips'].mean()
            log(f"  conf>={conf_min}: {len(conf_df)} trades, {wr:.1f}% WR, {avg_net:.2f} net avg")

    log(f"\n--- COMPARISON ---")
    log(f"  V1.5 (2 yrs, 28 pairs): 566 trades, 72.6% WR, +5.31 net avg")
    log(f"  LSTM+V1.5 (6 mo, EURUSD): {len(trades_df)} trades, {trades_df['win'].mean()*100:.1f}% WR, {trades_df['net_pips'].mean():.2f} net avg")

log(f"\nCompleted: {datetime.now()}")
