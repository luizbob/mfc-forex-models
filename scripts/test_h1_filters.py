"""
H1 Filter Testing Script
========================
Tests different H1 filter configurations to find what improves win rate.
Based on strict_backtest_v3.py but with filter testing.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from datetime import datetime
import gc

def log(msg=""):
    print(msg, flush=True)

log("=" * 80)
log("H1 FILTER TESTING")
log("=" * 80)
log(f"Started: {datetime.now()}")
log()

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/data')
CLEAN_DIR = DATA_DIR / 'cleaned'
LSTM_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/lstm_model')
MODEL_DIR = LSTM_DIR / 'models'
DATA_LSTM_DIR = LSTM_DIR / 'data'

# Strategy parameters
MIN_CONFIDENCE = 0.70
MFC_EXTREME = 0.5
RSI_PERIOD = 14
RSI_LOW = 20
RSI_HIGH = 80
MAX_LOOK_FORWARD = 201

MODEL_TRAIN_END = '2024-12-31'

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

CURRENCIES = ['EUR', 'USD', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD', 'NZD']

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
# LOAD CONFIG
# ============================================================================

with open(DATA_LSTM_DIR / 'config.pkl', 'rb') as f:
    config = pickle.load(f)

LOOKBACK = config['lookback']

# ============================================================================
# 1. LOAD ALL DATA
# ============================================================================
log("1. Loading MFC data...")

mfc_m5 = {}
mfc_m15 = {}
mfc_m30 = {}
mfc_h1 = {}
mfc_h4 = {}

for ccy in CURRENCIES:
    # M5
    df = pd.read_csv(DATA_DIR / f'mfc_currency_{ccy}_M5.csv')
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y.%m.%d %H:%M')
    df = df.set_index('datetime')
    df = df[df.index > MODEL_TRAIN_END]
    mfc_m5[ccy] = df['MFC']

    # M15
    df = pd.read_csv(DATA_DIR / f'mfc_currency_{ccy}_M15.csv')
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y.%m.%d %H:%M')
    df = df.set_index('datetime')
    df = df[df.index > MODEL_TRAIN_END]
    mfc_m15[ccy] = df['MFC']

    # M30
    df = pd.read_csv(DATA_DIR / f'mfc_currency_{ccy}_M30.csv')
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y.%m.%d %H:%M')
    df = df.set_index('datetime')
    df = df[df.index > MODEL_TRAIN_END]
    mfc_m30[ccy] = df['MFC']

    # H1 (cleaned)
    df = pd.read_csv(CLEAN_DIR / f'mfc_currency_{ccy}_H1_clean.csv')
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y.%m.%d %H:%M')
    df = df.set_index('datetime')
    df = df[df.index > MODEL_TRAIN_END]
    mfc_h1[ccy] = df['MFC']

    # H4 (cleaned)
    df = pd.read_csv(CLEAN_DIR / f'mfc_currency_{ccy}_H4_clean.csv')
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y.%m.%d %H:%M')
    df = df.set_index('datetime')
    df = df[df.index > MODEL_TRAIN_END]
    mfc_h4[ccy] = df['MFC']

log(f"   Loaded data for {len(CURRENCIES)} currencies")

# ============================================================================
# 2. PRE-GENERATE LSTM PREDICTIONS
# ============================================================================
log("\n2. Generating LSTM predictions (batch)...")

lstm_predictions = {}

for ccy in CURRENCIES:
    log(f"   Processing {ccy}...")

    model = tf.keras.models.load_model(MODEL_DIR / f'lstm_{ccy}_final.keras')

    m5_idx = mfc_m5[ccy].index

    m5_shifted = mfc_m5[ccy].shift(1)
    m15_shifted = mfc_m15[ccy].shift(1).reindex(m5_idx, method='ffill')
    m30_shifted = mfc_m30[ccy].shift(1).reindex(m5_idx, method='ffill')
    h1_shifted = mfc_h1[ccy].shift(1).reindex(m5_idx, method='ffill')
    h4_shifted = mfc_h4[ccy].shift(1).reindex(m5_idx, method='ffill')

    m5_data = m5_shifted.values
    m15_data = m15_shifted.values
    m30_data = m30_shifted.values
    h1_data = h1_shifted.values
    h4_data = h4_shifted.values

    max_lb = max(LOOKBACK.values())
    valid_start = max_lb + 1
    n_samples = len(m5_data) - valid_start - 1

    if n_samples <= 0:
        continue

    X_M5 = np.array([m5_data[i-LOOKBACK['M5']:i] for i in range(valid_start, valid_start+n_samples)])
    X_M15 = np.array([m15_data[i-LOOKBACK['M15']:i] for i in range(valid_start, valid_start+n_samples)])
    X_M30 = np.array([m30_data[i-LOOKBACK['M30']:i] for i in range(valid_start, valid_start+n_samples)])
    X_H1 = np.array([h1_data[i-LOOKBACK['H1']:i] for i in range(valid_start, valid_start+n_samples)])
    X_H4 = np.array([h4_data[i-LOOKBACK['H4']:i] for i in range(valid_start, valid_start+n_samples)])

    vel_m5 = np.diff(m5_data, prepend=m5_data[0])
    vel_m30 = np.diff(m30_data, prepend=m30_data[0])

    X_aux = np.column_stack([
        vel_m5[valid_start:valid_start+n_samples],
        vel_m30[valid_start:valid_start+n_samples],
        m5_data[valid_start:valid_start+n_samples],
        m30_data[valid_start:valid_start+n_samples],
        h4_data[valid_start:valid_start+n_samples],
    ])

    X_val = [
        X_M5.reshape(-1, LOOKBACK['M5'], 1),
        X_M15.reshape(-1, LOOKBACK['M15'], 1),
        X_M30.reshape(-1, LOOKBACK['M30'], 1),
        X_H1.reshape(-1, LOOKBACK['H1'], 1),
        X_H4.reshape(-1, LOOKBACK['H4'], 1),
        X_aux,
    ]

    pred = model.predict(X_val, verbose=0, batch_size=512)

    datetimes = mfc_m5[ccy].index[valid_start:valid_start+n_samples]

    lstm_predictions[ccy] = pd.DataFrame({
        'direction': np.argmax(pred[0], axis=1),
        'confidence': np.max(pred[0], axis=1)
    }, index=datetimes)

    del model, X_val, pred
    tf.keras.backend.clear_session()
    gc.collect()

log(f"   Done - {len(lstm_predictions)} currencies")

# ============================================================================
# 3. LOAD PRICE DATA
# ============================================================================
log("\n3. Loading price data...")

price_data = {}

for pair, base, quote in ALL_PAIRS:
    filepath = DATA_DIR / f'{pair}_GMT+0_US-DST_M1.csv'
    if not filepath.exists():
        continue

    try:
        df = pd.read_csv(filepath)
        df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y.%m.%d %H:%M:%S')
        df = df.set_index('datetime')
        df = df[df.index > MODEL_TRAIN_END]

        df_m5 = df.resample('5min').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
        }).dropna()

        df_m5['RSI'] = calculate_rsi(df_m5['Close'], RSI_PERIOD)
        df_m5['RSI'] = df_m5['RSI'].shift(1)

        price_data[pair] = df_m5

    except Exception as e:
        log(f"   {pair}: ERROR - {e}")

log(f"   Loaded {len(price_data)} pairs")

# ============================================================================
# 4. PREPARE SIGNALS (BASE - NO FILTERS)
# ============================================================================
log("\n4. Preparing base signals...")

# Pre-compute all necessary data for each pair
pair_signals = {}

for pair, base, quote in ALL_PAIRS:
    if pair not in price_data:
        continue
    if base not in lstm_predictions or quote not in lstm_predictions:
        continue

    df_price = price_data[pair].copy()
    pip_value = get_pip_value(pair)
    spread = SPREADS.get(pair, 2.0)

    # Add MFC (shifted)
    m5_idx = df_price.index
    df_price['base_mfc'] = mfc_m5[base].shift(1).reindex(m5_idx, method='ffill')
    df_price['quote_mfc'] = mfc_m5[quote].shift(1).reindex(m5_idx, method='ffill')

    # Add H1 MFC for filters (shifted)
    df_price['base_h1'] = mfc_h1[base].shift(1).reindex(m5_idx, method='ffill')
    df_price['quote_h1'] = mfc_h1[quote].shift(1).reindex(m5_idx, method='ffill')

    # Add H1 velocity (change over last 12 M5 bars = 1 hour)
    base_h1_series = mfc_h1[base].shift(1).reindex(m5_idx, method='ffill')
    quote_h1_series = mfc_h1[quote].shift(1).reindex(m5_idx, method='ffill')
    df_price['base_vel_h1'] = base_h1_series - base_h1_series.shift(12)
    df_price['quote_vel_h1'] = quote_h1_series - quote_h1_series.shift(12)

    # Add LSTM predictions
    base_lstm = lstm_predictions[base].reindex(m5_idx, method='ffill')
    quote_lstm = lstm_predictions[quote].reindex(m5_idx, method='ffill')

    df_price['base_dir'] = base_lstm['direction']
    df_price['base_conf'] = base_lstm['confidence']
    df_price['quote_dir'] = quote_lstm['direction']
    df_price['quote_conf'] = quote_lstm['confidence']

    df_price = df_price.dropna()

    if len(df_price) == 0:
        continue

    # Find base signals (before any H1 filters)
    buy_signal = (
        (df_price['base_dir'] == 2) &
        (df_price['quote_dir'] == 0) &
        (df_price['base_conf'] >= MIN_CONFIDENCE) &
        (df_price['quote_conf'] >= MIN_CONFIDENCE) &
        (df_price['base_mfc'] <= -MFC_EXTREME) &
        (df_price['RSI'] < RSI_LOW)
    )

    sell_signal = (
        (df_price['base_dir'] == 0) &
        (df_price['quote_dir'] == 2) &
        (df_price['base_conf'] >= MIN_CONFIDENCE) &
        (df_price['quote_conf'] >= MIN_CONFIDENCE) &
        (df_price['base_mfc'] >= MFC_EXTREME) &
        (df_price['RSI'] > RSI_HIGH)
    )

    pair_signals[pair] = {
        'df': df_price,
        'buy_signal': buy_signal,
        'sell_signal': sell_signal,
        'base': base,
        'quote': quote,
        'pip_value': pip_value,
        'spread': spread,
    }

log(f"   Prepared signals for {len(pair_signals)} pairs")

# ============================================================================
# 5. FILTER TESTING FUNCTION
# ============================================================================

def run_backtest_with_filter(pair_signals, filter_func=None, filter_name="No Filter"):
    """Run backtest with optional filter function.

    filter_func takes (df, signal_type) and returns boolean mask for signals to KEEP
    """
    all_trades = []

    for pair, data in pair_signals.items():
        df = data['df']
        pip_value = data['pip_value']
        spread = data['spread']

        buy_signal = data['buy_signal'].copy()
        sell_signal = data['sell_signal'].copy()

        # Apply filter if provided
        if filter_func is not None:
            buy_filter = filter_func(df, 'BUY')
            sell_filter = filter_func(df, 'SELL')
            buy_signal = buy_signal & buy_filter
            sell_signal = sell_signal & sell_filter

        # Process BUY signals
        buy_indices = df.index[buy_signal].tolist()
        i = 0
        while i < len(buy_indices):
            signal_time = buy_indices[i]
            signal_idx = df.index.get_loc(signal_time)

            if signal_idx + 1 >= len(df):
                i += 1
                continue

            entry_idx = signal_idx + 1
            entry_time = df.index[entry_idx]
            entry_price = df.iloc[entry_idx]['Open']

            future_df = df.iloc[entry_idx+1:entry_idx+1+MAX_LOOK_FORWARD]

            if len(future_df) == 0:
                i += 1
                continue

            exit_mask = future_df['RSI'] >= RSI_HIGH

            if exit_mask.any():
                exit_idx = exit_mask.argmax()
                exit_time = future_df.index[exit_idx]
                exit_price = future_df.iloc[exit_idx]['Close']
                bars_held = exit_idx + 1

                cost = spread * pip_value
                pips = (exit_price - entry_price - cost) / pip_value

                all_trades.append({
                    'pair': pair,
                    'signal': 'BUY',
                    'entry_time': entry_time,
                    'pips': pips,
                    'win': 1 if pips > 0 else 0,
                })

                while i < len(buy_indices) and buy_indices[i] <= exit_time:
                    i += 1
            else:
                i += 1

        # Process SELL signals
        sell_indices = df.index[sell_signal].tolist()
        i = 0
        while i < len(sell_indices):
            signal_time = sell_indices[i]
            signal_idx = df.index.get_loc(signal_time)

            if signal_idx + 1 >= len(df):
                i += 1
                continue

            entry_idx = signal_idx + 1
            entry_time = df.index[entry_idx]
            entry_price = df.iloc[entry_idx]['Open']

            future_df = df.iloc[entry_idx+1:entry_idx+1+MAX_LOOK_FORWARD]

            if len(future_df) == 0:
                i += 1
                continue

            exit_mask = future_df['RSI'] <= RSI_LOW

            if exit_mask.any():
                exit_idx = exit_mask.argmax()
                exit_time = future_df.index[exit_idx]
                exit_price = future_df.iloc[exit_idx]['Close']
                bars_held = exit_idx + 1

                cost = spread * pip_value
                pips = (entry_price - exit_price - cost) / pip_value

                all_trades.append({
                    'pair': pair,
                    'signal': 'SELL',
                    'entry_time': entry_time,
                    'pips': pips,
                    'win': 1 if pips > 0 else 0,
                })

                while i < len(sell_indices) and sell_indices[i] <= exit_time:
                    i += 1
            else:
                i += 1

    if len(all_trades) == 0:
        return {
            'filter': filter_name,
            'trades': 0,
            'wr': 0,
            'avg_pips': 0,
            'total_pips': 0,
        }

    df_trades = pd.DataFrame(all_trades)

    return {
        'filter': filter_name,
        'trades': len(df_trades),
        'wr': df_trades['win'].mean() * 100,
        'avg_pips': df_trades['pips'].mean(),
        'total_pips': df_trades['pips'].sum(),
    }

# ============================================================================
# 6. DEFINE FILTERS TO TEST
# ============================================================================
log("\n5. Testing filters...")

results = []

# Baseline - no filter
log("   Testing: No Filter (baseline)")
results.append(run_backtest_with_filter(pair_signals, None, "No Filter"))

# ============================================================================
# H1 VELOCITY FILTERS (only one side)
# ============================================================================

# Filter: Skip BUY if base_vel_h1 < -threshold (base moving down)
for thresh in [0.02, 0.03, 0.04, 0.05, 0.06]:
    def make_filter(t):
        def f(df, signal_type):
            if signal_type == 'BUY':
                return df['base_vel_h1'] >= -t  # Keep if not strongly down
            return pd.Series(True, index=df.index)
        return f

    name = f"BUY: base_vel_h1 >= -{thresh}"
    log(f"   Testing: {name}")
    results.append(run_backtest_with_filter(pair_signals, make_filter(thresh), name))

# Filter: Skip BUY if quote_vel_h1 > +threshold (quote moving up against us)
for thresh in [0.02, 0.03, 0.04, 0.05, 0.06]:
    def make_filter(t):
        def f(df, signal_type):
            if signal_type == 'BUY':
                return df['quote_vel_h1'] <= t  # Keep if quote not strongly up
            return pd.Series(True, index=df.index)
        return f

    name = f"BUY: quote_vel_h1 <= +{thresh}"
    log(f"   Testing: {name}")
    results.append(run_backtest_with_filter(pair_signals, make_filter(thresh), name))

# Filter: Skip SELL if base_vel_h1 > +threshold (base moving up)
for thresh in [0.02, 0.03, 0.04, 0.05, 0.06]:
    def make_filter(t):
        def f(df, signal_type):
            if signal_type == 'SELL':
                return df['base_vel_h1'] <= t  # Keep if base not strongly up
            return pd.Series(True, index=df.index)
        return f

    name = f"SELL: base_vel_h1 <= +{thresh}"
    log(f"   Testing: {name}")
    results.append(run_backtest_with_filter(pair_signals, make_filter(thresh), name))

# Filter: Skip SELL if quote_vel_h1 < -threshold (quote moving down against us)
for thresh in [0.02, 0.03, 0.04, 0.05, 0.06]:
    def make_filter(t):
        def f(df, signal_type):
            if signal_type == 'SELL':
                return df['quote_vel_h1'] >= -t  # Keep if quote not strongly down
            return pd.Series(True, index=df.index)
        return f

    name = f"SELL: quote_vel_h1 >= -{thresh}"
    log(f"   Testing: {name}")
    results.append(run_backtest_with_filter(pair_signals, make_filter(thresh), name))

# ============================================================================
# H1 POSITION FILTERS
# ============================================================================

# Filter: BUY only if quote H1 is low (weak quote = pair can go up)
for thresh in [0.0, 0.1, 0.2, 0.3]:
    def make_filter(t):
        def f(df, signal_type):
            if signal_type == 'BUY':
                return df['quote_h1'] <= t
            return pd.Series(True, index=df.index)
        return f

    name = f"BUY: quote_h1 <= {thresh}"
    log(f"   Testing: {name}")
    results.append(run_backtest_with_filter(pair_signals, make_filter(thresh), name))

# Filter: SELL only if base H1 is high (strong base can reverse)
for thresh in [0.0, 0.1, 0.2, 0.3]:
    def make_filter(t):
        def f(df, signal_type):
            if signal_type == 'SELL':
                return df['base_h1'] >= t
            return pd.Series(True, index=df.index)
        return f

    name = f"SELL: base_h1 >= {thresh}"
    log(f"   Testing: {name}")
    results.append(run_backtest_with_filter(pair_signals, make_filter(thresh), name))

# ============================================================================
# COMBINED FILTERS (both BUY and SELL)
# ============================================================================

# Best velocity threshold (0.04) for both sides
def filter_vel_both_0p04(df, signal_type):
    if signal_type == 'BUY':
        return df['base_vel_h1'] >= -0.04
    else:  # SELL
        return df['base_vel_h1'] <= 0.04

log(f"   Testing: Both: base_vel_h1 filter 0.04")
results.append(run_backtest_with_filter(pair_signals, filter_vel_both_0p04, "Both: base_vel_h1 +/-0.04"))

# Position filter both sides
def filter_pos_both(df, signal_type):
    if signal_type == 'BUY':
        return df['quote_h1'] <= 0.1
    else:  # SELL
        return df['base_h1'] >= 0.3

log(f"   Testing: Both: position filter")
results.append(run_backtest_with_filter(pair_signals, filter_pos_both, "Both: quote_h1<=0.1 / base_h1>=0.3"))

# ============================================================================
# 7. RESULTS
# ============================================================================
log("\n" + "=" * 80)
log("FILTER TEST RESULTS")
log("=" * 80)

df_results = pd.DataFrame(results)
df_results = df_results.sort_values('wr', ascending=False)

# Get baseline for comparison
baseline = df_results[df_results['filter'] == 'No Filter'].iloc[0]

log(f"\nBASELINE (No Filter):")
log(f"   Trades: {baseline['trades']}")
log(f"   Win Rate: {baseline['wr']:.1f}%")
log(f"   Avg Pips: {baseline['avg_pips']:.2f}")
log(f"   Total Pips: {baseline['total_pips']:.0f}")

log(f"\n{'Filter':<45} {'Trades':<8} {'WR%':<8} {'AvgPips':<10} {'TotalPips':<12} {'WR Diff'}")
log("-" * 100)

for _, row in df_results.iterrows():
    wr_diff = row['wr'] - baseline['wr']
    sign = '+' if wr_diff >= 0 else ''
    log(f"{row['filter']:<45} {row['trades']:<8.0f} {row['wr']:<8.1f} {row['avg_pips']:<10.2f} {row['total_pips']:<12.0f} {sign}{wr_diff:.1f}%")

# Top improvements
log(f"\n{'='*80}")
log("TOP 10 FILTERS BY WIN RATE IMPROVEMENT:")
log("=" * 80)

top_filters = df_results[df_results['filter'] != 'No Filter'].head(10)
for i, (_, row) in enumerate(top_filters.iterrows(), 1):
    wr_diff = row['wr'] - baseline['wr']
    trades_diff = row['trades'] - baseline['trades']
    log(f"\n{i}. {row['filter']}")
    log(f"   WR: {row['wr']:.1f}% ({'+' if wr_diff >= 0 else ''}{wr_diff:.1f}%)")
    log(f"   Trades: {row['trades']:.0f} ({trades_diff:+.0f})")
    log(f"   Total Pips: {row['total_pips']:.0f}")

log(f"\n{'='*80}")
log(f"Completed: {datetime.now()}")
log("=" * 80)
