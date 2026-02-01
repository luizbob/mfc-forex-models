"""
Script 07: LSTM + V1.5 Filters Combined
========================================
LSTM predicts direction (when will currency cycle?)
V1.5 filters tell us WHEN to enter (extreme conditions)

Only trade when:
1. LSTM predicts BOTH currencies going opposite directions (divergence)
2. V1.5-style MFC extremes are met
3. High confidence from both models
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
log("LSTM + V1.5 COMBINED BACKTEST")
log("=" * 70)
log(f"Started: {datetime.now()}")

# Paths
DATA_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/data')
LSTM_DATA_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/lstm_model/data')
MODEL_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/lstm_model/models')

with open(LSTM_DATA_DIR / 'config.pkl', 'rb') as f:
    config = pickle.load(f)

LOOKBACK = config['lookback']
CURRENCIES = config['currencies']

# All 28 pairs
ALL_PAIRS = [
    ('EURUSD', 'EUR', 'USD'), ('GBPUSD', 'GBP', 'USD'), ('AUDUSD', 'AUD', 'USD'),
    ('NZDUSD', 'NZD', 'USD'), ('USDJPY', 'USD', 'JPY'), ('USDCHF', 'USD', 'CHF'),
    ('USDCAD', 'USD', 'CAD'), ('EURGBP', 'EUR', 'GBP'), ('EURJPY', 'EUR', 'JPY'),
    ('EURCHF', 'EUR', 'CHF'), ('EURCAD', 'EUR', 'CAD'), ('EURAUD', 'EUR', 'AUD'),
    ('EURNZD', 'EUR', 'NZD'), ('GBPJPY', 'GBP', 'JPY'), ('GBPCHF', 'GBP', 'CHF'),
    ('GBPCAD', 'GBP', 'CAD'), ('GBPAUD', 'GBP', 'AUD'), ('GBPNZD', 'GBP', 'NZD'),
    ('AUDJPY', 'AUD', 'JPY'), ('AUDCHF', 'AUD', 'CHF'), ('AUDCAD', 'AUD', 'CAD'),
    ('AUDNZD', 'AUD', 'NZD'), ('NZDJPY', 'NZD', 'JPY'), ('NZDCHF', 'NZD', 'CHF'),
    ('NZDCAD', 'NZD', 'CAD'), ('CADJPY', 'CAD', 'JPY'), ('CADCHF', 'CAD', 'CHF'),
    ('CHFJPY', 'CHF', 'JPY'),
]

SPREADS = {
    'EURUSD': 1.5, 'GBPUSD': 2.0, 'USDJPY': 1.5, 'USDCHF': 2.0, 'USDCAD': 2.0,
    'AUDUSD': 1.5, 'NZDUSD': 2.5, 'EURGBP': 2.0, 'EURJPY': 2.0, 'GBPJPY': 3.0,
    'EURCHF': 2.5, 'AUDJPY': 2.5, 'EURAUD': 3.0, 'EURCAD': 3.0, 'EURNZD': 4.0,
    'GBPAUD': 3.5, 'GBPCAD': 3.5, 'GBPCHF': 3.0, 'GBPNZD': 4.5, 'AUDCAD': 2.5,
    'AUDCHF': 2.5, 'AUDNZD': 3.0, 'CADJPY': 2.5, 'CHFJPY': 2.5, 'NZDCAD': 3.5,
    'NZDCHF': 3.5, 'NZDJPY': 3.0, 'CADCHF': 3.0,
}

# V1.5-style thresholds
MFC_EXTREME = 0.5   # Base currency must be at extreme
MFC_QUOTE_MAX = 0.3  # Quote currency not blocking
MIN_CONFIDENCE = 0.75  # Minimum LSTM confidence

def get_pip_value(pair):
    return 0.01 if 'JPY' in pair else 0.0001

# ============================================================================
# 1. LOAD LSTM DATA AND PREDICTIONS
# ============================================================================
log("\n1. Loading LSTM data and generating predictions...")

predictions = {}
mfc_data = {}

for ccy in CURRENCIES:
    log(f"  Processing {ccy}...")

    # Load data
    with open(LSTM_DATA_DIR / f'lstm_data_{ccy}.pkl', 'rb') as f:
        data = pickle.load(f)

    n_samples = len(data['datetimes'])
    split_idx = int(n_samples * 0.8)

    # Load model
    model = tf.keras.models.load_model(MODEL_DIR / f'lstm_{ccy}_final.keras')

    # Prepare validation data
    X_val = [
        data['X_M5'][split_idx:].reshape(-1, LOOKBACK['M5'], 1),
        data['X_M15'][split_idx:].reshape(-1, LOOKBACK['M15'], 1),
        data['X_M30'][split_idx:].reshape(-1, LOOKBACK['M30'], 1),
        data['X_H1'][split_idx:].reshape(-1, LOOKBACK['H1'], 1),
        data['X_H4'][split_idx:].reshape(-1, LOOKBACK['H4'], 1),
        data['X_aux'][split_idx:],
    ]

    # Predict
    pred = model.predict(X_val, verbose=0, batch_size=256)
    direction = np.argmax(pred[0], axis=1)  # 0=DOWN, 1=NEUTRAL, 2=UP
    confidence = np.max(pred[0], axis=1)

    predictions[ccy] = {
        'direction': direction,
        'confidence': confidence,
        'datetimes': data['datetimes'][split_idx:],
    }

    # Store MFC values (current_M5 is index 2 in X_aux)
    mfc_data[ccy] = {
        'mfc_m5': data['X_aux'][split_idx:, 2],
        'vel_m5': data['X_aux'][split_idx:, 0],
    }

    log(f"    DOWN={np.sum(direction==0)}, NEUTRAL={np.sum(direction==1)}, UP={np.sum(direction==2)}")

    del model, data, X_val, pred
    tf.keras.backend.clear_session()
    gc.collect()

# ============================================================================
# 2. LOAD PRICE DATA
# ============================================================================
log("\n2. Loading price data...")

price_data = {}
for pair, _, _ in ALL_PAIRS:
    try:
        chunks = []
        for chunk in pd.read_csv(DATA_DIR / f'{pair}_GMT+0_US-DST_M1.csv', chunksize=500000):
            chunk['datetime'] = pd.to_datetime(chunk['Date'] + ' ' + chunk['Time'], format='%Y.%m.%d %H:%M:%S')
            chunk = chunk.set_index('datetime')
            chunk = chunk[(chunk.index >= '2024-07-01') & (chunk.index <= '2024-12-31')]
            if len(chunk) > 0:
                m5_chunk = chunk[['Close']].resample('5min').last().dropna()
                chunks.append(m5_chunk)

        if chunks:
            price_data[pair] = pd.concat(chunks)
    except Exception as e:
        log(f"  {pair}: ERROR - {e}")

log(f"  Loaded {len(price_data)} pairs")

# ============================================================================
# 3. GENERATE SIGNALS WITH LSTM + V1.5 FILTERS
# ============================================================================
log("\n3. Generating signals (LSTM + V1.5 filters)...")

all_trades = []

for pair, base, quote in ALL_PAIRS:
    if pair not in price_data:
        continue

    pip_val = get_pip_value(pair)
    spread = SPREADS.get(pair, 2.0)
    price_df = price_data[pair]

    base_pred = predictions[base]
    quote_pred = predictions[quote]
    base_mfc = mfc_data[base]['mfc_m5']
    quote_mfc = mfc_data[quote]['mfc_m5']

    trades = []
    max_idx = min(len(base_pred['datetimes']), len(quote_pred['datetimes']))

    for i in range(max_idx):
        dt = pd.Timestamp(base_pred['datetimes'][i])

        if dt not in price_df.index:
            continue

        # LSTM predictions
        base_dir = base_pred['direction'][i]
        base_conf = base_pred['confidence'][i]
        quote_dir = quote_pred['direction'][i]
        quote_conf = quote_pred['confidence'][i]

        # Skip neutral or low confidence
        if base_dir == 1 or quote_dir == 1:
            continue
        if base_conf < MIN_CONFIDENCE or quote_conf < MIN_CONFIDENCE:
            continue

        # Current MFC values
        base_mfc_val = base_mfc[i]
        quote_mfc_val = quote_mfc[i]

        signal = None

        # BUY: Base UP + Quote DOWN + V1.5 conditions
        if base_dir == 2 and quote_dir == 0:
            # V1.5: Base at LOW extreme (will go up), Quote not too high
            if base_mfc_val <= -MFC_EXTREME and quote_mfc_val <= MFC_QUOTE_MAX:
                signal = 'BUY'

        # SELL: Base DOWN + Quote UP + V1.5 conditions
        elif base_dir == 0 and quote_dir == 2:
            # V1.5: Base at HIGH extreme (will go down), Quote not too low
            if base_mfc_val >= MFC_EXTREME and quote_mfc_val >= -MFC_QUOTE_MAX:
                signal = 'SELL'

        if signal is None:
            continue

        # Get prices
        try:
            price_idx = price_df.index.get_loc(dt)
            if isinstance(price_idx, slice):
                price_idx = price_idx.start
            entry_price = price_df.iloc[price_idx]['Close']

            exit_idx = min(price_idx + 72, len(price_df) - 1)
            exit_price = price_df.iloc[exit_idx]['Close']
        except:
            continue

        if signal == 'BUY':
            pips = (exit_price - entry_price) / pip_val
        else:
            pips = (entry_price - exit_price) / pip_val

        trades.append({
            'pair': pair,
            'datetime': dt,
            'signal': signal,
            'base_conf': base_conf,
            'quote_conf': quote_conf,
            'base_mfc': base_mfc_val,
            'quote_mfc': quote_mfc_val,
            'pips': pips,
            'spread': spread,
        })

    if trades:
        log(f"  {pair}: {len(trades)} trades")
        all_trades.extend(trades)

# ============================================================================
# 4. RESULTS
# ============================================================================
log("\n" + "=" * 70)
log("RESULTS - LSTM + V1.5 COMBINED")
log("=" * 70)

if len(all_trades) == 0:
    log("\nNo trades generated!")
else:
    trades_df = pd.DataFrame(all_trades)
    trades_df['net_pips'] = trades_df['pips'] - trades_df['spread']
    trades_df['win'] = (trades_df['pips'] > 0).astype(int)

    log(f"\nOVERALL:")
    log(f"  Total trades: {len(trades_df)}")
    log(f"  Win rate: {trades_df['win'].mean()*100:.1f}%")
    log(f"  Avg pips: {trades_df['pips'].mean():.2f}")
    log(f"  Avg net pips: {trades_df['net_pips'].mean():.2f}")
    log(f"  Total net pips: {trades_df['net_pips'].sum():.0f}")

    log(f"\nBY SIGNAL:")
    for sig in ['BUY', 'SELL']:
        sig_df = trades_df[trades_df['signal'] == sig]
        if len(sig_df) > 0:
            wr = sig_df['win'].mean() * 100
            avg_net = sig_df['net_pips'].mean()
            log(f"  {sig}: {len(sig_df)} trades, {wr:.1f}% WR, {avg_net:.2f} net avg")

    log(f"\nBY PAIR (top 10):")
    pair_stats = trades_df.groupby('pair').agg({
        'pips': 'count',
        'win': 'mean',
        'net_pips': 'mean'
    }).rename(columns={'pips': 'trades', 'win': 'wr'})
    pair_stats = pair_stats.sort_values('net_pips', ascending=False).head(10)
    for pair, row in pair_stats.iterrows():
        log(f"  {pair}: {int(row['trades'])} trades, {row['wr']*100:.1f}% WR, {row['net_pips']:.2f} net avg")

    log(f"\nBY CONFIDENCE:")
    for conf_min in [0.80, 0.85, 0.90, 0.95]:
        conf_df = trades_df[(trades_df['base_conf'] >= conf_min) & (trades_df['quote_conf'] >= conf_min)]
        if len(conf_df) > 0:
            wr = conf_df['win'].mean() * 100
            avg_net = conf_df['net_pips'].mean()
            log(f"  both>={conf_min}: {len(conf_df)} trades, {wr:.1f}% WR, {avg_net:.2f} net avg")

    log(f"\n" + "=" * 70)
    log(f"COMPARISON")
    log(f"=" * 70)
    log(f"  V1.5 Corrected: 509 trades, 71.3% WR, +4.30 net avg")
    log(f"  LSTM+V1.5:      {len(trades_df)} trades, {trades_df['win'].mean()*100:.1f}% WR, {trades_df['net_pips'].mean():.2f} net avg")

    # Save trades
    trades_df.to_csv(LSTM_DATA_DIR / 'lstm_v15_combined_trades.csv', index=False)
    log(f"\n  Trades saved to lstm_v15_combined_trades.csv")

log(f"\nCompleted: {datetime.now()}")
