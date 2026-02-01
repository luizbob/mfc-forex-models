"""
Script 09: V1.5 + LSTM Filter Variations
========================================
Tests different ways to use LSTM predictions with V1.5:
1. Full divergence (base UP + quote DOWN for BUY)
2. Base only (base UP for BUY, ignore quote)
3. Either agrees (base UP OR quote DOWN for BUY)
4. Not wrong (base not DOWN for BUY)

Goal: Find the best balance between trade quality and quantity.
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
log("V1.5 + LSTM FILTER VARIATIONS")
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

RSI_LOW = 20
RSI_HIGH = 80
RSI_MEDIAN_PERIOD = 9
MFC_THRESHOLD = 0.5
QUOTE_THRESHOLD = 0.3

VELOCITY_PAIRS = [
    'GBPCHF', 'NZDUSD', 'CADCHF', 'USDCAD', 'AUDCHF', 'EURCAD', 'GBPNZD',
    'USDCHF', 'GBPAUD', 'EURGBP', 'EURCHF', 'AUDUSD', 'EURJPY', 'GBPUSD'
]

SELL_BASE_VEL_M30_MAX = 0.10
BUY_QUOTE_H4_MAX = 0.10

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
# 1. LOAD ALL DATA
# ============================================================================
log("\n1. Loading data...")

# LSTM predictions
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

log(f"  LSTM predictions loaded for {len(CURRENCIES)} currencies")

# MFC data
mfc_m5 = {}
mfc_m30_shifted = {}
mfc_h4_shifted = {}

for cur in CURRENCIES:
    df = pd.read_csv(DATA_DIR / f'mfc_currency_{cur}_M5.csv')
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y.%m.%d %H:%M')
    df = df.set_index('datetime')
    mfc_m5[cur] = df['MFC']

    df = pd.read_csv(DATA_DIR / f'mfc_currency_{cur}_M30.csv')
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y.%m.%d %H:%M')
    df = df.set_index('datetime')
    mfc_m30_shifted[cur] = df['MFC'].shift(1)

    df = pd.read_csv(DATA_DIR / 'cleaned' / f'mfc_currency_{cur}_H4_clean.csv')
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y.%m.%d %H:%M')
    df = df.set_index('datetime')
    mfc_h4_shifted[cur] = df['MFC'].shift(1)

log("  MFC data loaded")

# Price data
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

log(f"  Price data loaded for {len(price_data)} pairs")

# ============================================================================
# 2. DEFINE LSTM FILTER MODES
# ============================================================================

def check_lstm_filter(trade_type, base_dir, base_conf, quote_dir, quote_conf, mode, min_conf):
    """
    Check if LSTM conditions pass for given filter mode.

    Modes:
    - 'full_divergence': BUY needs base UP + quote DOWN
    - 'base_only': BUY needs base UP (ignore quote)
    - 'either_agrees': BUY needs base UP OR quote DOWN
    - 'not_wrong': BUY needs base NOT DOWN, SELL needs base NOT UP
    - 'no_neutral': Same as not_wrong but also reject neutral predictions
    """
    if trade_type == 'BUY':
        if mode == 'full_divergence':
            return (base_dir == 2 and quote_dir == 0 and
                    base_conf >= min_conf and quote_conf >= min_conf)

        elif mode == 'base_only':
            return base_dir == 2 and base_conf >= min_conf

        elif mode == 'either_agrees':
            return ((base_dir == 2 and base_conf >= min_conf) or
                    (quote_dir == 0 and quote_conf >= min_conf))

        elif mode == 'not_wrong':
            # Base should not be predicted DOWN with high confidence
            return not (base_dir == 0 and base_conf >= min_conf)

        elif mode == 'no_neutral':
            # Base should not be DOWN or NEUTRAL
            return base_dir == 2 or base_conf < min_conf

    else:  # SELL
        if mode == 'full_divergence':
            return (base_dir == 0 and quote_dir == 2 and
                    base_conf >= min_conf and quote_conf >= min_conf)

        elif mode == 'base_only':
            return base_dir == 0 and base_conf >= min_conf

        elif mode == 'either_agrees':
            return ((base_dir == 0 and base_conf >= min_conf) or
                    (quote_dir == 2 and quote_conf >= min_conf))

        elif mode == 'not_wrong':
            return not (base_dir == 2 and base_conf >= min_conf)

        elif mode == 'no_neutral':
            return base_dir == 0 or base_conf < min_conf

    return True

# ============================================================================
# 3. RUN BACKTEST
# ============================================================================

def run_backtest(lstm_mode=None, min_conf=0.70):
    """Run V1.5 backtest with optional LSTM filter mode."""
    all_trades = []

    for pair, base, quote in ALL_PAIRS:
        if pair not in price_data:
            continue

        use_base_vel = pair in VELOCITY_PAIRS
        pip_val = get_pip_value(pair)

        try:
            price_df = price_data[pair].copy()

            price_df['base_mfc'] = mfc_m5[base].shift(1).reindex(price_df.index, method='ffill')
            price_df['quote_mfc'] = mfc_m5[quote].shift(1).reindex(price_df.index, method='ffill')

            m30_base = mfc_m30_shifted[base].reindex(price_df.index, method='ffill')
            m30_base_prev = mfc_m30_shifted[base].shift(1).reindex(price_df.index, method='ffill')
            base_vel_m30 = m30_base - m30_base_prev

            quote_h4 = mfc_h4_shifted[quote].reindex(price_df.index, method='ffill')

            base_vel_m5 = price_df['base_mfc'] - price_df['base_mfc'].shift(1)
            quote_vel_m5 = price_df['quote_mfc'] - price_df['quote_mfc'].shift(1)

            rsi_series = price_df['rsi']
            rsi_med = rsi_series.rolling(RSI_MEDIAN_PERIOD).median()

            if lstm_mode:
                base_lstm = lstm_predictions[base].reindex(price_df.index, method='ffill')
                quote_lstm = lstm_predictions[quote].reindex(price_df.index, method='ffill')

            # BUY signals
            buy_signal = (
                (rsi_series < RSI_LOW) & (rsi_med < RSI_LOW) &
                (price_df['base_mfc'] <= -MFC_THRESHOLD) &
                (price_df['quote_mfc'] <= QUOTE_THRESHOLD) &
                (quote_vel_m5 < 0) &
                (quote_h4 <= BUY_QUOTE_H4_MAX)
            )
            if use_base_vel:
                buy_signal = buy_signal & (base_vel_m5 > 0)

            # SELL signals
            sell_signal = (
                (rsi_series > RSI_HIGH) & (rsi_med > RSI_HIGH) &
                (price_df['base_mfc'] >= MFC_THRESHOLD) &
                (price_df['quote_mfc'] >= -QUOTE_THRESHOLD) &
                (quote_vel_m5 > 0) &
                (base_vel_m30 <= SELL_BASE_VEL_M30_MAX)
            )
            if use_base_vel:
                sell_signal = sell_signal & (base_vel_m5 < 0)

            # Process BUY
            buy_indices = price_df.index[buy_signal].tolist()
            i = 0
            while i < len(buy_indices):
                signal_time = buy_indices[i]
                signal_idx = price_df.index.get_loc(signal_time)

                if lstm_mode:
                    try:
                        base_dir = base_lstm.loc[signal_time, 'direction']
                        base_conf = base_lstm.loc[signal_time, 'confidence']
                        quote_dir = quote_lstm.loc[signal_time, 'direction']
                        quote_conf = quote_lstm.loc[signal_time, 'confidence']

                        if not check_lstm_filter('BUY', base_dir, base_conf,
                                                 quote_dir, quote_conf, lstm_mode, min_conf):
                            i += 1
                            continue
                    except:
                        i += 1
                        continue

                entry_price = price_df.loc[signal_time, 'Close']
                future_df = price_df.iloc[signal_idx+1:signal_idx+201]
                exit_mask = future_df['rsi'] >= RSI_HIGH

                if exit_mask.any():
                    exit_time = future_df.index[exit_mask.argmax()]
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

            # Process SELL
            sell_indices = price_df.index[sell_signal].tolist()
            i = 0
            while i < len(sell_indices):
                signal_time = sell_indices[i]
                signal_idx = price_df.index.get_loc(signal_time)

                if lstm_mode:
                    try:
                        base_dir = base_lstm.loc[signal_time, 'direction']
                        base_conf = base_lstm.loc[signal_time, 'confidence']
                        quote_dir = quote_lstm.loc[signal_time, 'direction']
                        quote_conf = quote_lstm.loc[signal_time, 'confidence']

                        if not check_lstm_filter('SELL', base_dir, base_conf,
                                                 quote_dir, quote_conf, lstm_mode, min_conf):
                            i += 1
                            continue
                    except:
                        i += 1
                        continue

                entry_price = price_df.loc[signal_time, 'Close']
                future_df = price_df.iloc[signal_idx+1:signal_idx+201]
                exit_mask = future_df['rsi'] <= RSI_LOW

                if exit_mask.any():
                    exit_time = future_df.index[exit_mask.argmax()]
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

        except Exception as e:
            pass

    return pd.DataFrame(all_trades)

# ============================================================================
# 4. RUN ALL VARIATIONS
# ============================================================================
log("\n" + "=" * 70)
log("2. RUNNING BACKTEST VARIATIONS")
log("=" * 70)

results = []

# Baseline
log("\n--- V1.5 BASELINE ---")
trades = run_backtest(lstm_mode=None)
if len(trades) > 0:
    trades['net_pips'] = trades['pips'] - trades['spread']
    results.append({
        'mode': 'V1.5 Only',
        'trades': len(trades),
        'wr': trades['win'].mean() * 100,
        'net_avg': trades['net_pips'].mean(),
        'total_net': trades['net_pips'].sum()
    })
    log(f"  {len(trades)} trades, {trades['win'].mean()*100:.1f}% WR, {trades['net_pips'].mean():.2f} net avg")

# LSTM variations
lstm_modes = ['full_divergence', 'base_only', 'either_agrees', 'not_wrong']
confidence_levels = [0.70, 0.80]

for mode in lstm_modes:
    for conf in confidence_levels:
        log(f"\n--- {mode.upper()} (conf >= {conf}) ---")
        trades = run_backtest(lstm_mode=mode, min_conf=conf)
        if len(trades) > 0:
            trades['net_pips'] = trades['pips'] - trades['spread']
            results.append({
                'mode': f"{mode} (>={conf})",
                'trades': len(trades),
                'wr': trades['win'].mean() * 100,
                'net_avg': trades['net_pips'].mean(),
                'total_net': trades['net_pips'].sum()
            })
            log(f"  {len(trades)} trades, {trades['win'].mean()*100:.1f}% WR, {trades['net_pips'].mean():.2f} net avg")
        else:
            log(f"  No trades")

# ============================================================================
# 5. SUMMARY
# ============================================================================
log("\n" + "=" * 70)
log("3. SUMMARY TABLE")
log("=" * 70)

log(f"\n{'Mode':<30} {'Trades':>8} {'WR%':>8} {'Net Avg':>10} {'Total':>10}")
log("-" * 70)

for r in results:
    log(f"{r['mode']:<30} {r['trades']:>8} {r['wr']:>7.1f}% {r['net_avg']:>10.2f} {r['total_net']:>10.0f}")

# Find best configurations
log("\n" + "=" * 70)
log("4. ANALYSIS")
log("=" * 70)

if len(results) > 1:
    baseline = results[0]
    log(f"\nBaseline: {baseline['trades']} trades, {baseline['wr']:.1f}% WR, {baseline['net_avg']:.2f} net avg")

    log("\nImprovements over baseline:")
    for r in results[1:]:
        if r['wr'] > baseline['wr'] or r['net_avg'] > baseline['net_avg']:
            wr_diff = r['wr'] - baseline['wr']
            net_diff = r['net_avg'] - baseline['net_avg']
            trade_pct = r['trades'] / baseline['trades'] * 100
            log(f"  {r['mode']}: WR {wr_diff:+.1f}%, Net {net_diff:+.2f}, Trades={trade_pct:.0f}% of baseline")

log(f"\nCompleted: {datetime.now()}")
