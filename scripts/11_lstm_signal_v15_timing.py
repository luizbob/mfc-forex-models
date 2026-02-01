"""
Script 11: LSTM Signal + V1.5 Timing
====================================
Instead of using V1.5 for signal and LSTM for filter,
use LSTM divergence for signal and V1.5-like conditions for timing.

Approach:
1. LSTM divergence gives us the DIRECTION (when both currencies predict opposite cycles)
2. Wait for V1.5-like extreme conditions for ENTRY TIMING
3. This may generate more trades than strict V1.5+LSTM while keeping quality

Trade Flow:
- LSTM predicts base UP + quote DOWN -> Mark pair as "BUY pending"
- Wait for MFC/RSI extremes to enter
- Exit when conditions reverse
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
log("LSTM SIGNAL + V1.5 TIMING")
log("=" * 70)
log(f"Started: {datetime.now()}")

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

log(f"  LSTM predictions loaded")

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

log(f"  Price data loaded")

# ============================================================================
# STRATEGY VARIATIONS
# ============================================================================

def run_strategy(entry_mode='basic', exit_mode='rsi', min_conf=0.70, mfc_threshold=0.5, rsi_low=20, rsi_high=80):
    """
    Run LSTM divergence strategy with different entry/exit modes.

    Entry modes:
    - 'basic': Enter when LSTM divergence + MFC extreme
    - 'rsi': Enter when LSTM divergence + MFC extreme + RSI extreme
    - 'relaxed': Enter when LSTM divergence + MFC > 0.3 (less extreme)

    Exit modes:
    - 'rsi': Exit when RSI crosses opposite extreme
    - 'mfc': Exit when base MFC crosses 0
    - 'time': Exit after 72 bars (6 hours)
    """
    all_trades = []

    for pair, base, quote in ALL_PAIRS:
        if pair not in price_data:
            continue

        pip_val = get_pip_value(pair)

        try:
            price_df = price_data[pair].copy()

            # MFC data (shifted by 1)
            price_df['base_mfc'] = mfc_m5[base].shift(1).reindex(price_df.index, method='ffill')
            price_df['quote_mfc'] = mfc_m5[quote].shift(1).reindex(price_df.index, method='ffill')

            # LSTM predictions
            base_lstm = lstm_predictions[base].reindex(price_df.index, method='ffill')
            quote_lstm = lstm_predictions[quote].reindex(price_df.index, method='ffill')

            price_df['base_dir'] = base_lstm['direction']
            price_df['base_conf'] = base_lstm['confidence']
            price_df['quote_dir'] = quote_lstm['direction']
            price_df['quote_conf'] = quote_lstm['confidence']

            rsi = price_df['rsi']

            # LSTM divergence conditions
            # BUY divergence: base UP (2) + quote DOWN (0)
            buy_divergence = (
                (price_df['base_dir'] == 2) &
                (price_df['quote_dir'] == 0) &
                (price_df['base_conf'] >= min_conf) &
                (price_df['quote_conf'] >= min_conf)
            )

            # SELL divergence: base DOWN (0) + quote UP (2)
            sell_divergence = (
                (price_df['base_dir'] == 0) &
                (price_df['quote_dir'] == 2) &
                (price_df['base_conf'] >= min_conf) &
                (price_df['quote_conf'] >= min_conf)
            )

            # Entry conditions based on mode
            if entry_mode == 'basic':
                # MFC extreme only
                buy_entry = buy_divergence & (price_df['base_mfc'] <= -mfc_threshold)
                sell_entry = sell_divergence & (price_df['base_mfc'] >= mfc_threshold)
            elif entry_mode == 'rsi':
                # MFC + RSI extreme
                buy_entry = buy_divergence & (price_df['base_mfc'] <= -mfc_threshold) & (rsi < rsi_low)
                sell_entry = sell_divergence & (price_df['base_mfc'] >= mfc_threshold) & (rsi > rsi_high)
            elif entry_mode == 'relaxed':
                # MFC > 0.3 (less extreme)
                buy_entry = buy_divergence & (price_df['base_mfc'] <= -0.3)
                sell_entry = sell_divergence & (price_df['base_mfc'] >= 0.3)
            else:
                continue

            # Process BUY signals
            buy_indices = price_df.index[buy_entry].tolist()
            i = 0
            while i < len(buy_indices):
                signal_time = buy_indices[i]
                signal_idx = price_df.index.get_loc(signal_time)

                entry_price = price_df.loc[signal_time, 'Close']
                future_df = price_df.iloc[signal_idx+1:signal_idx+201]

                # Exit conditions based on mode
                if exit_mode == 'rsi':
                    exit_mask = future_df['rsi'] >= rsi_high
                elif exit_mode == 'mfc':
                    exit_mask = future_df['base_mfc'] >= 0
                elif exit_mode == 'time':
                    exit_mask = pd.Series([False] * min(71, len(future_df)) + [True] * (len(future_df) - 71 if len(future_df) > 71 else 0), index=future_df.index)
                else:
                    exit_mask = future_df['rsi'] >= rsi_high

                if exit_mask.any():
                    exit_idx = exit_mask.argmax()
                    if exit_idx < len(future_df):
                        exit_time = future_df.index[exit_idx]
                        exit_price = price_df.loc[exit_time, 'Close']
                        pips = (exit_price - entry_price) / pip_val

                        all_trades.append({
                            'pair': pair, 'type': 'BUY', 'entry_time': signal_time,
                            'pips': pips, 'win': 1 if pips > 0 else 0,
                            'spread': SPREADS.get(pair, 2.0)
                        })

                        while i < len(buy_indices) and buy_indices[i] <= exit_time:
                            i += 1
                    else:
                        i += 1
                else:
                    i += 1

            # Process SELL signals
            sell_indices = price_df.index[sell_entry].tolist()
            i = 0
            while i < len(sell_indices):
                signal_time = sell_indices[i]
                signal_idx = price_df.index.get_loc(signal_time)

                entry_price = price_df.loc[signal_time, 'Close']
                future_df = price_df.iloc[signal_idx+1:signal_idx+201]

                if exit_mode == 'rsi':
                    exit_mask = future_df['rsi'] <= rsi_low
                elif exit_mode == 'mfc':
                    exit_mask = future_df['base_mfc'] <= 0
                elif exit_mode == 'time':
                    exit_mask = pd.Series([False] * min(71, len(future_df)) + [True] * (len(future_df) - 71 if len(future_df) > 71 else 0), index=future_df.index)
                else:
                    exit_mask = future_df['rsi'] <= rsi_low

                if exit_mask.any():
                    exit_idx = exit_mask.argmax()
                    if exit_idx < len(future_df):
                        exit_time = future_df.index[exit_idx]
                        exit_price = price_df.loc[exit_time, 'Close']
                        pips = (entry_price - exit_price) / pip_val

                        all_trades.append({
                            'pair': pair, 'type': 'SELL', 'entry_time': signal_time,
                            'pips': pips, 'win': 1 if pips > 0 else 0,
                            'spread': SPREADS.get(pair, 2.0)
                        })

                        while i < len(sell_indices) and sell_indices[i] <= exit_time:
                            i += 1
                    else:
                        i += 1
                else:
                    i += 1

        except Exception as e:
            pass

    return pd.DataFrame(all_trades)

# ============================================================================
# RUN STRATEGIES
# ============================================================================
log("\n" + "=" * 70)
log("2. TESTING STRATEGIES")
log("=" * 70)

results = []

strategies = [
    # (entry_mode, exit_mode, min_conf, mfc_threshold, rsi_low, rsi_high, label)
    ('basic', 'rsi', 0.70, 0.5, 20, 80, 'LSTM+MFC0.5 -> RSI exit'),
    ('basic', 'rsi', 0.70, 0.4, 20, 80, 'LSTM+MFC0.4 -> RSI exit'),
    ('basic', 'rsi', 0.70, 0.3, 20, 80, 'LSTM+MFC0.3 -> RSI exit'),
    ('rsi', 'rsi', 0.70, 0.5, 20, 80, 'LSTM+MFC0.5+RSI20 -> RSI exit'),
    ('rsi', 'rsi', 0.70, 0.5, 30, 70, 'LSTM+MFC0.5+RSI30 -> RSI exit'),
    ('rsi', 'rsi', 0.70, 0.4, 30, 70, 'LSTM+MFC0.4+RSI30 -> RSI exit'),
    ('basic', 'mfc', 0.70, 0.5, 20, 80, 'LSTM+MFC0.5 -> MFC exit'),
    ('basic', 'time', 0.70, 0.5, 20, 80, 'LSTM+MFC0.5 -> 6H exit'),
    ('relaxed', 'rsi', 0.70, 0.3, 20, 80, 'LSTM+MFC0.3 relaxed -> RSI'),
    ('basic', 'rsi', 0.80, 0.5, 20, 80, 'LSTM0.8+MFC0.5 -> RSI exit'),
]

for entry_mode, exit_mode, min_conf, mfc_threshold, rsi_low, rsi_high, label in strategies:
    log(f"\n--- {label} ---")
    trades = run_strategy(entry_mode, exit_mode, min_conf, mfc_threshold, rsi_low, rsi_high)

    if len(trades) > 0:
        trades['net_pips'] = trades['pips'] - trades['spread']
        wr = trades['win'].mean() * 100
        net_avg = trades['net_pips'].mean()
        total = trades['net_pips'].sum()
        results.append({
            'label': label,
            'trades': len(trades),
            'wr': wr,
            'net_avg': net_avg,
            'total': total
        })
        log(f"  {len(trades)} trades, {wr:.1f}% WR, {net_avg:.2f} net avg, {total:.0f} total")
    else:
        log(f"  No trades")

# ============================================================================
# SUMMARY
# ============================================================================
log("\n" + "=" * 70)
log("3. SUMMARY TABLE")
log("=" * 70)

log(f"\n{'Strategy':<35} {'Trades':>8} {'WR%':>8} {'Net Avg':>10} {'Total':>10}")
log("-" * 75)

# Sort by net_avg
results_sorted = sorted(results, key=lambda x: x['net_avg'], reverse=True)
for r in results_sorted:
    log(f"{r['label']:<35} {r['trades']:>8} {r['wr']:>7.1f}% {r['net_avg']:>10.2f} {r['total']:>10.0f}")

log("\n" + "=" * 70)
log("4. COMPARISON WITH BASELINES")
log("=" * 70)

log("\nBaselines:")
log("  V1.5 Corrected (Jul-Dec 2024): 115 trades, 73.9% WR, +4.96 net avg")
log("  V1.5 + LSTM Full Div (0.8):     18 trades, 77.8% WR, +6.56 net avg")

if results:
    best = results_sorted[0]
    log(f"\nBest LSTM-first strategy:")
    log(f"  {best['label']}: {best['trades']} trades, {best['wr']:.1f}% WR, {best['net_avg']:.2f} net avg")

log(f"\nCompleted: {datetime.now()}")
