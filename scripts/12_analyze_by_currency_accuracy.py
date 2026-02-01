"""
Script 12: Analyze Performance by Currency Model Accuracy
=========================================================
Some LSTM models are more accurate than others.
JPY=90%, USD=87.8%, AUD=87.4%, NZD=86.7%
CAD=82.2%, GBP=83.7%, EUR=82.0%

Hypothesis: Trades involving more accurate models should perform better.
"""
import warnings
warnings.filterwarnings('ignore')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from pathlib import Path
import gc

def log(msg=""):
    print(msg, flush=True)

log("=" * 70)
log("ANALYZE BY CURRENCY MODEL ACCURACY")
log("=" * 70)
log(f"Started: {datetime.now()}")

# Currency model accuracies (from training)
MODEL_ACCURACY = {
    'JPY': 90.0,
    'USD': 87.8,
    'AUD': 87.4,
    'NZD': 86.7,
    'CHF': 86.3,
    'GBP': 83.7,
    'CAD': 82.2,
    'EUR': 82.0,
}

DATA_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/data')
LSTM_DATA_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/lstm_model/data')
MODEL_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/lstm_model/models')

with open(LSTM_DATA_DIR / 'config.pkl', 'rb') as f:
    config = pickle.load(f)

LOOKBACK = config['lookback']
CURRENCIES = config['currencies']

START_DATE = '2024-07-01'
END_DATE = '2024-12-31'

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

# Calculate pair model accuracy (average of both currencies)
PAIR_ACCURACY = {}
for pair, base, quote in ALL_PAIRS:
    PAIR_ACCURACY[pair] = (MODEL_ACCURACY[base] + MODEL_ACCURACY[quote]) / 2

SPREADS = {
    'EURUSD': 1.5, 'GBPUSD': 2.0, 'USDJPY': 1.5, 'USDCHF': 2.0, 'USDCAD': 2.0,
    'AUDUSD': 1.5, 'NZDUSD': 2.5, 'EURGBP': 2.0, 'EURJPY': 2.0, 'GBPJPY': 3.0,
    'EURCHF': 2.5, 'AUDJPY': 2.5, 'EURAUD': 3.0, 'EURCAD': 3.0, 'EURNZD': 4.0,
    'GBPAUD': 3.5, 'GBPCAD': 3.5, 'GBPCHF': 3.0, 'GBPNZD': 4.5, 'AUDCAD': 2.5,
    'AUDCHF': 2.5, 'AUDNZD': 3.0, 'CADJPY': 2.5, 'CHFJPY': 2.5, 'NZDCAD': 3.5,
    'NZDCHF': 3.5, 'NZDJPY': 3.0, 'CADCHF': 3.0,
}

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def get_pip_value(pair):
    return 0.01 if 'JPY' in pair else 0.0001

# ============================================================================
# LOAD DATA
# ============================================================================
log("\n1. Loading data...")

lstm_predictions = {}
for ccy in CURRENCIES:
    with open(LSTM_DATA_DIR / f'lstm_data_{ccy}.pkl', 'rb') as f:
        data = pickle.load(f)

    n_samples = len(data['datetimes'])
    split_idx = int(n_samples * 0.8)

    model = tf.keras.models.load_model(MODEL_DIR / f'lstm_{ccy}_final.keras')

    X_val = [
        data['X_M5'][split_idx:].reshape(-1, LOOKBACK['M5'], 1),
        data['X_M15'][split_idx:].reshape(-1, LOOKBACK['M15'], 1),
        data['X_M30'][split_idx:].reshape(-1, LOOKBACK['M30'], 1),
        data['X_H1'][split_idx:].reshape(-1, LOOKBACK['H1'], 1),
        data['X_H4'][split_idx:].reshape(-1, LOOKBACK['H4'], 1),
        data['X_aux'][split_idx:],
    ]

    pred = model.predict(X_val, verbose=0, batch_size=256)
    datetimes = pd.to_datetime(data['datetimes'][split_idx:])
    lstm_predictions[ccy] = pd.DataFrame({
        'direction': np.argmax(pred[0], axis=1),
        'confidence': np.max(pred[0], axis=1)
    }, index=datetimes)

    del model, data, X_val, pred
    tf.keras.backend.clear_session()
    gc.collect()

log("  LSTM predictions loaded")

mfc_m5 = {}
for cur in CURRENCIES:
    df = pd.read_csv(DATA_DIR / f'mfc_currency_{cur}_M5.csv')
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y.%m.%d %H:%M')
    df = df.set_index('datetime')
    mfc_m5[cur] = df['MFC']

log("  MFC data loaded")

price_data = {}
for pair, base, quote in ALL_PAIRS:
    try:
        chunks = []
        for chunk in pd.read_csv(DATA_DIR / f'{pair}_GMT+0_US-DST_M1.csv', chunksize=500000):
            chunk['datetime'] = pd.to_datetime(chunk['Date'] + ' ' + chunk['Time'], format='%Y.%m.%d %H:%M:%S')
            chunk = chunk.set_index('datetime')
            chunk = chunk[(chunk.index >= START_DATE) & (chunk.index <= END_DATE)]
            if len(chunk) > 0:
                m5_chunk = chunk[['Open', 'High', 'Low', 'Close']].resample('5min').agg({
                    'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
                }).dropna()
                chunks.append(m5_chunk)

        if chunks:
            price_m5 = pd.concat(chunks)
            price_m5 = price_m5[~price_m5.index.duplicated(keep='first')]
            price_m5['rsi'] = calculate_rsi(price_m5['Close'], period=14)
            price_data[pair] = price_m5
    except:
        pass

log("  Price data loaded")

# ============================================================================
# RUN STRATEGY AND COLLECT TRADES WITH METADATA
# ============================================================================
log("\n2. Running LSTM divergence strategy...")

all_trades = []

for pair, base, quote in ALL_PAIRS:
    if pair not in price_data:
        continue

    pip_val = get_pip_value(pair)

    try:
        price_df = price_data[pair].copy()

        price_df['base_mfc'] = mfc_m5[base].shift(1).reindex(price_df.index, method='ffill')
        price_df['quote_mfc'] = mfc_m5[quote].shift(1).reindex(price_df.index, method='ffill')

        base_lstm = lstm_predictions[base].reindex(price_df.index, method='ffill')
        quote_lstm = lstm_predictions[quote].reindex(price_df.index, method='ffill')

        price_df['base_dir'] = base_lstm['direction']
        price_df['base_conf'] = base_lstm['confidence']
        price_df['quote_dir'] = quote_lstm['direction']
        price_df['quote_conf'] = quote_lstm['confidence']

        rsi = price_df['rsi']

        # LSTM divergence + MFC extreme + RSI extreme
        buy_signal = (
            (price_df['base_dir'] == 2) &
            (price_df['quote_dir'] == 0) &
            (price_df['base_conf'] >= 0.70) &
            (price_df['quote_conf'] >= 0.70) &
            (price_df['base_mfc'] <= -0.5) &
            (rsi < 20)
        )

        sell_signal = (
            (price_df['base_dir'] == 0) &
            (price_df['quote_dir'] == 2) &
            (price_df['base_conf'] >= 0.70) &
            (price_df['quote_conf'] >= 0.70) &
            (price_df['base_mfc'] >= 0.5) &
            (rsi > 80)
        )

        # Process BUY
        buy_indices = price_df.index[buy_signal].tolist()
        i = 0
        while i < len(buy_indices):
            signal_time = buy_indices[i]
            signal_idx = price_df.index.get_loc(signal_time)

            entry_price = price_df.loc[signal_time, 'Close']
            future_df = price_df.iloc[signal_idx+1:signal_idx+201]
            exit_mask = future_df['rsi'] >= 80

            if exit_mask.any():
                exit_time = future_df.index[exit_mask.argmax()]
                exit_price = price_df.loc[exit_time, 'Close']
                pips = (exit_price - entry_price) / pip_val

                all_trades.append({
                    'pair': pair,
                    'base': base,
                    'quote': quote,
                    'type': 'BUY',
                    'entry_time': signal_time,
                    'pips': pips,
                    'win': 1 if pips > 0 else 0,
                    'spread': SPREADS.get(pair, 2.0),
                    'base_acc': MODEL_ACCURACY[base],
                    'quote_acc': MODEL_ACCURACY[quote],
                    'pair_acc': PAIR_ACCURACY[pair],
                    'base_conf': price_df.loc[signal_time, 'base_conf'],
                    'quote_conf': price_df.loc[signal_time, 'quote_conf'],
                })

                while i < len(buy_indices) and buy_indices[i] <= exit_time:
                    i += 1
            else:
                i += 1

        # Process SELL
        sell_indices = price_df.index[sell_signal].tolist()
        i = 0
        while i < len(sell_indices):
            signal_time = sell_indices[i]
            signal_idx = price_df.index.get_loc(signal_time)

            entry_price = price_df.loc[signal_time, 'Close']
            future_df = price_df.iloc[signal_idx+1:signal_idx+201]
            exit_mask = future_df['rsi'] <= 20

            if exit_mask.any():
                exit_time = future_df.index[exit_mask.argmax()]
                exit_price = price_df.loc[exit_time, 'Close']
                pips = (entry_price - exit_price) / pip_val

                all_trades.append({
                    'pair': pair,
                    'base': base,
                    'quote': quote,
                    'type': 'SELL',
                    'entry_time': signal_time,
                    'pips': pips,
                    'win': 1 if pips > 0 else 0,
                    'spread': SPREADS.get(pair, 2.0),
                    'base_acc': MODEL_ACCURACY[base],
                    'quote_acc': MODEL_ACCURACY[quote],
                    'pair_acc': PAIR_ACCURACY[pair],
                    'base_conf': price_df.loc[signal_time, 'base_conf'],
                    'quote_conf': price_df.loc[signal_time, 'quote_conf'],
                })

                while i < len(sell_indices) and sell_indices[i] <= exit_time:
                    i += 1
            else:
                i += 1

    except:
        pass

trades_df = pd.DataFrame(all_trades)
trades_df['net_pips'] = trades_df['pips'] - trades_df['spread']

log(f"  Total trades: {len(trades_df)}")

# ============================================================================
# ANALYZE BY ACCURACY
# ============================================================================
log("\n" + "=" * 70)
log("3. ANALYSIS BY MODEL ACCURACY")
log("=" * 70)

# By pair accuracy
log("\n--- By Pair Average Accuracy ---")
log(f"{'Pair Acc':<12} {'Trades':>8} {'WR%':>8} {'Net Avg':>10} {'Total':>10}")
log("-" * 50)

for threshold in [85, 86, 87, 88]:
    filtered = trades_df[trades_df['pair_acc'] >= threshold]
    if len(filtered) > 0:
        wr = filtered['win'].mean() * 100
        net = filtered['net_pips'].mean()
        total = filtered['net_pips'].sum()
        log(f">= {threshold}%{'':<6} {len(filtered):>8} {wr:>7.1f}% {net:>10.2f} {total:>10.0f}")

# By minimum of base/quote accuracy
log("\n--- By Minimum Currency Accuracy ---")
log(f"{'Min Acc':<12} {'Trades':>8} {'WR%':>8} {'Net Avg':>10} {'Total':>10}")
log("-" * 50)

trades_df['min_acc'] = trades_df[['base_acc', 'quote_acc']].min(axis=1)

for threshold in [83, 85, 86, 87]:
    filtered = trades_df[trades_df['min_acc'] >= threshold]
    if len(filtered) > 0:
        wr = filtered['win'].mean() * 100
        net = filtered['net_pips'].mean()
        total = filtered['net_pips'].sum()
        log(f">= {threshold}%{'':<6} {len(filtered):>8} {wr:>7.1f}% {net:>10.2f} {total:>10.0f}")

# By individual currency involvement
log("\n--- By Currency Involvement ---")
log(f"{'Currency':<12} {'Acc%':>6} {'Trades':>8} {'WR%':>8} {'Net Avg':>10}")
log("-" * 50)

for ccy in sorted(CURRENCIES, key=lambda x: MODEL_ACCURACY[x], reverse=True):
    ccy_df = trades_df[(trades_df['base'] == ccy) | (trades_df['quote'] == ccy)]
    if len(ccy_df) > 0:
        wr = ccy_df['win'].mean() * 100
        net = ccy_df['net_pips'].mean()
        log(f"{ccy:<12} {MODEL_ACCURACY[ccy]:>5.1f}% {len(ccy_df):>8} {wr:>7.1f}% {net:>10.2f}")

# By specific pairs
log("\n--- By Individual Pair ---")
log(f"{'Pair':<12} {'Acc%':>6} {'Trades':>8} {'WR%':>8} {'Net Avg':>10}")
log("-" * 50)

pair_stats = trades_df.groupby('pair').agg({
    'pips': 'count',
    'win': 'mean',
    'net_pips': 'mean',
    'pair_acc': 'first'
}).rename(columns={'pips': 'trades', 'win': 'wr'})
pair_stats = pair_stats.sort_values('net_pips', ascending=False)

for pair, row in pair_stats.head(15).iterrows():
    log(f"{pair:<12} {row['pair_acc']:>5.1f}% {int(row['trades']):>8} {row['wr']*100:>7.1f}% {row['net_pips']:>10.2f}")

# JPY pairs specifically
log("\n--- JPY Pairs (Best Model at 90%) ---")
jpy_df = trades_df[(trades_df['base'] == 'JPY') | (trades_df['quote'] == 'JPY')]
if len(jpy_df) > 0:
    log(f"  Trades: {len(jpy_df)}")
    log(f"  Win Rate: {jpy_df['win'].mean()*100:.1f}%")
    log(f"  Net Avg: {jpy_df['net_pips'].mean():.2f}")
    log(f"  Total: {jpy_df['net_pips'].sum():.0f}")

# ============================================================================
# SUMMARY
# ============================================================================
log("\n" + "=" * 70)
log("4. SUMMARY")
log("=" * 70)

log("\nAll trades:")
log(f"  {len(trades_df)} trades, {trades_df['win'].mean()*100:.1f}% WR, {trades_df['net_pips'].mean():.2f} net avg")

if len(trades_df) > 0:
    best_acc = trades_df[trades_df['min_acc'] >= 87]
    if len(best_acc) > 0:
        log(f"\nBest accuracy pairs only (min acc >= 87%):")
        log(f"  {len(best_acc)} trades, {best_acc['win'].mean()*100:.1f}% WR, {best_acc['net_pips'].mean():.2f} net avg")

log(f"\nCompleted: {datetime.now()}")
