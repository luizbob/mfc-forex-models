"""
Analyze MFC patterns at entry for TIMEOUT vs RSI exit trades.
Goal: Find MFC characteristics that predict timeout trades.
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
log("MFC PATTERN ANALYSIS: TIMEOUT vs RSI EXIT TRADES")
log("=" * 70)
log(f"Started: {datetime.now()}")

MODEL_ACCURACY = {
    'JPY': 90.0, 'USD': 87.8, 'AUD': 87.4, 'NZD': 86.7,
    'CHF': 86.3, 'GBP': 83.7, 'CAD': 82.2, 'EUR': 82.0,
}

DATA_END_DATES = {
    'EUR': '2025-12-21', 'NZD': '2025-12-21', 'USD': '2025-12-21',
    'GBP': '2025-12-21', 'JPY': '2025-12-21', 'CHF': '2025-12-21',
    'CAD': '2025-12-21', 'AUD': '2025-12-21',
}

DATA_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/data')
LSTM_DATA_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/lstm_model/data')
MODEL_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/lstm_model/models')

with open(LSTM_DATA_DIR / 'config.pkl', 'rb') as f:
    config = pickle.load(f)

LOOKBACK = config['lookback']
CURRENCIES = config['currencies']

START_DATE = '2025-01-01'

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

def get_pair_end_date(base, quote):
    return min(DATA_END_DATES[base], DATA_END_DATES[quote])

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
# LOAD MFC DATA FOR ALL TIMEFRAMES
# ============================================================================
log("\n1. Loading MFC data...")

mfc_m5, mfc_m15, mfc_m30, mfc_h1, mfc_h4, mfc_d1 = {}, {}, {}, {}, {}, {}

for cur in CURRENCIES:
    end_date = DATA_END_DATES[cur]

    # M5
    df = pd.read_csv(DATA_DIR / 'cleaned' / f'mfc_currency_{cur}_M5_clean.csv')
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y.%m.%d %H:%M')
    df = df.set_index('datetime')
    df = df[(df.index >= START_DATE) & (df.index <= end_date)]
    mfc_m5[cur] = df['MFC']

    # M15
    df = pd.read_csv(DATA_DIR / 'cleaned' / f'mfc_currency_{cur}_M15_clean.csv')
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y.%m.%d %H:%M')
    df = df.set_index('datetime')
    df = df[(df.index >= START_DATE) & (df.index <= end_date)]
    mfc_m15[cur] = df['MFC']

    # M30
    df = pd.read_csv(DATA_DIR / 'cleaned' / f'mfc_currency_{cur}_M30_clean.csv')
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y.%m.%d %H:%M')
    df = df.set_index('datetime')
    df = df[(df.index >= START_DATE) & (df.index <= end_date)]
    mfc_m30[cur] = df['MFC']

    # H1
    df = pd.read_csv(DATA_DIR / 'cleaned' / f'mfc_currency_{cur}_H1_clean.csv')
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y.%m.%d %H:%M')
    df = df.set_index('datetime')
    df = df[(df.index >= START_DATE) & (df.index <= end_date)]
    mfc_h1[cur] = df['MFC']

    # H4
    df = pd.read_csv(DATA_DIR / 'cleaned' / f'mfc_currency_{cur}_H4_clean.csv')
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y.%m.%d %H:%M')
    df = df.set_index('datetime')
    df = df[(df.index >= START_DATE) & (df.index <= end_date)]
    mfc_h4[cur] = df['MFC']

    # D1
    try:
        df = pd.read_csv(DATA_DIR / 'cleaned' / f'mfc_currency_{cur}_D1_clean.csv')
        df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y.%m.%d %H:%M')
        df = df.set_index('datetime')
        df = df[(df.index >= START_DATE) & (df.index <= end_date)]
        mfc_d1[cur] = df['MFC']
    except:
        mfc_d1[cur] = pd.Series(dtype=float)

log(f"  Loaded MFC for {len(CURRENCIES)} currencies")

# ============================================================================
# LOAD LSTM PREDICTIONS
# ============================================================================
log("\n2. Generating LSTM predictions...")

lstm_predictions = {}

for ccy in CURRENCIES:
    end_date = DATA_END_DATES[ccy]
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

    pred = model.predict(X_val, verbose=0, batch_size=256)
    datetimes = mfc_m5[ccy].index[valid_start:valid_start+n_samples]

    lstm_predictions[ccy] = pd.DataFrame({
        'direction': np.argmax(pred[0], axis=1),
        'confidence': np.max(pred[0], axis=1)
    }, index=datetimes)

    del model, X_val, pred
    tf.keras.backend.clear_session()
    gc.collect()

log(f"  Generated predictions for {len(lstm_predictions)} currencies")

# ============================================================================
# LOAD PRICE DATA
# ============================================================================
log("\n3. Loading price data...")

price_data = {}
for pair, base, quote in ALL_PAIRS:
    end_date = get_pair_end_date(base, quote)
    try:
        chunks = []
        for chunk in pd.read_csv(DATA_DIR / f'{pair}_GMT+0_US-DST_M1.csv', chunksize=500000):
            chunk['datetime'] = pd.to_datetime(chunk['Date'] + ' ' + chunk['Time'], format='%Y.%m.%d %H:%M:%S')
            chunk = chunk.set_index('datetime')
            chunk = chunk[(chunk.index >= START_DATE) & (chunk.index <= end_date)]
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
    except Exception as e:
        pass

log(f"  Loaded {len(price_data)} pairs")

# ============================================================================
# RUN STRATEGY AND CAPTURE MFC AT ENTRY
# ============================================================================
log("\n4. Running strategy with MFC capture...")

MIN_CONF = 0.70
MFC_EXTREME = 0.5
RSI_LOW = 20
RSI_HIGH = 80
H1_VEL_THRESHOLD = 0.04
MAX_HOLD_BARS = 200
QUOTE_EXTENDED = 0.7

all_trades = []

for pair, base, quote in ALL_PAIRS:
    if pair not in price_data:
        continue
    if base not in lstm_predictions or quote not in lstm_predictions:
        continue

    pip_val = get_pip_value(pair)

    try:
        price_df = price_data[pair].copy()

        # MFC for all timeframes (shifted by 1 for no lookahead)
        price_df['base_m5'] = mfc_m5[base].shift(1).reindex(price_df.index, method='ffill')
        price_df['quote_m5'] = mfc_m5[quote].shift(1).reindex(price_df.index, method='ffill')
        price_df['base_m15'] = mfc_m15[base].shift(1).reindex(price_df.index, method='ffill')
        price_df['quote_m15'] = mfc_m15[quote].shift(1).reindex(price_df.index, method='ffill')
        price_df['base_m30'] = mfc_m30[base].shift(1).reindex(price_df.index, method='ffill')
        price_df['quote_m30'] = mfc_m30[quote].shift(1).reindex(price_df.index, method='ffill')
        price_df['base_h1'] = mfc_h1[base].shift(1).reindex(price_df.index, method='ffill')
        price_df['quote_h1'] = mfc_h1[quote].shift(1).reindex(price_df.index, method='ffill')
        price_df['base_h4'] = mfc_h4[base].shift(1).reindex(price_df.index, method='ffill')
        price_df['quote_h4'] = mfc_h4[quote].shift(1).reindex(price_df.index, method='ffill')

        if len(mfc_d1[base]) > 0:
            price_df['base_d1'] = mfc_d1[base].shift(1).reindex(price_df.index, method='ffill')
        else:
            price_df['base_d1'] = np.nan
        if len(mfc_d1[quote]) > 0:
            price_df['quote_d1'] = mfc_d1[quote].shift(1).reindex(price_df.index, method='ffill')
        else:
            price_df['quote_d1'] = np.nan

        # MFC velocities (1 bar change)
        price_df['base_vel_m5'] = price_df['base_m5'].diff()
        price_df['quote_vel_m5'] = price_df['quote_m5'].diff()
        price_df['base_vel_h1'] = price_df['base_h1'].diff(periods=12)  # 1 hour velocity
        price_df['quote_vel_h1'] = price_df['quote_h1'].diff(periods=12)

        # MFC spread (difference between base and quote)
        price_df['mfc_spread_m5'] = price_df['base_m5'] - price_df['quote_m5']
        price_df['mfc_spread_h1'] = price_df['base_h1'] - price_df['quote_h1']
        price_df['mfc_spread_h4'] = price_df['base_h4'] - price_df['quote_h4']

        # LSTM predictions
        base_lstm = lstm_predictions[base].reindex(price_df.index, method='ffill')
        quote_lstm = lstm_predictions[quote].reindex(price_df.index, method='ffill')
        price_df['base_dir'] = base_lstm['direction']
        price_df['base_conf'] = base_lstm['confidence']
        price_df['quote_dir'] = quote_lstm['direction']
        price_df['quote_conf'] = quote_lstm['confidence']

        price_df = price_df.dropna(subset=['base_m5', 'quote_m5', 'base_h1', 'quote_h1'])

        rsi = price_df['rsi']
        is_friday_afternoon = (price_df.index.dayofweek == 4) & (price_df.index.hour >= 12)

        buy_extended_ok = (price_df['base_h1'] < QUOTE_EXTENDED) | (price_df['quote_h1'] >= QUOTE_EXTENDED)
        sell_extended_ok = (price_df['quote_h1'] < QUOTE_EXTENDED) | (price_df['base_h1'] >= QUOTE_EXTENDED)

        buy_signal = (
            (price_df['base_dir'] == 2) & (price_df['quote_dir'] == 0) &
            (price_df['base_conf'] >= MIN_CONF) & (price_df['quote_conf'] >= MIN_CONF) &
            (price_df['base_m5'] <= -MFC_EXTREME) & (price_df['rsi'] < RSI_LOW) &
            (price_df['base_vel_h1'] >= -H1_VEL_THRESHOLD) & (~is_friday_afternoon) & (buy_extended_ok)
        )

        sell_signal = (
            (price_df['base_dir'] == 0) & (price_df['quote_dir'] == 2) &
            (price_df['base_conf'] >= MIN_CONF) & (price_df['quote_conf'] >= MIN_CONF) &
            (price_df['base_m5'] >= MFC_EXTREME) & (price_df['rsi'] > RSI_HIGH) &
            (price_df['base_vel_h1'] <= H1_VEL_THRESHOLD) & (~is_friday_afternoon) & (sell_extended_ok)
        )

        # Process BUY signals
        buy_indices = price_df.index[buy_signal].tolist()
        i = 0
        while i < len(buy_indices):
            signal_time = buy_indices[i]
            signal_idx = price_df.index.get_loc(signal_time)
            entry_idx = signal_idx + 1
            if entry_idx >= len(price_df):
                i += 1
                continue

            entry_time = price_df.index[entry_idx]
            entry_row = price_df.iloc[signal_idx]  # MFC at signal time
            entry_price = price_df.iloc[entry_idx]['Open']
            future_df = price_df.iloc[entry_idx+1:entry_idx+1+MAX_HOLD_BARS]

            if len(future_df) == 0:
                i += 1
                continue

            exit_mask = future_df['rsi'] >= RSI_HIGH

            if exit_mask.any():
                exit_time = future_df.index[exit_mask.argmax()]
                exit_price = price_df.loc[exit_time, 'Close']
                pips = (exit_price - entry_price) / pip_val
                exit_reason = 'RSI'
            else:
                exit_time = future_df.index[-1]
                exit_price = future_df.iloc[-1]['Close']
                pips = (exit_price - entry_price) / pip_val
                exit_reason = 'TIMEOUT'

            trade = {
                'pair': pair, 'base': base, 'quote': quote, 'type': 'BUY',
                'entry_time': entry_time, 'exit_time': exit_time,
                'pips': pips, 'win': 1 if pips > 0 else 0,
                'spread': SPREADS.get(pair, 2.0), 'exit_reason': exit_reason,
                # MFC values at entry
                'base_m5': entry_row['base_m5'], 'quote_m5': entry_row['quote_m5'],
                'base_m15': entry_row['base_m15'], 'quote_m15': entry_row['quote_m15'],
                'base_m30': entry_row['base_m30'], 'quote_m30': entry_row['quote_m30'],
                'base_h1': entry_row['base_h1'], 'quote_h1': entry_row['quote_h1'],
                'base_h4': entry_row['base_h4'], 'quote_h4': entry_row['quote_h4'],
                'base_d1': entry_row.get('base_d1', np.nan), 'quote_d1': entry_row.get('quote_d1', np.nan),
                # Velocities
                'base_vel_m5': entry_row['base_vel_m5'], 'quote_vel_m5': entry_row['quote_vel_m5'],
                'base_vel_h1': entry_row['base_vel_h1'], 'quote_vel_h1': entry_row['quote_vel_h1'],
                # Spreads
                'mfc_spread_m5': entry_row['mfc_spread_m5'],
                'mfc_spread_h1': entry_row['mfc_spread_h1'],
                'mfc_spread_h4': entry_row['mfc_spread_h4'],
                # RSI
                'rsi': entry_row['rsi'],
            }
            all_trades.append(trade)

            while i < len(buy_indices) and buy_indices[i] <= exit_time:
                i += 1

        # Process SELL signals
        sell_indices = price_df.index[sell_signal].tolist()
        i = 0
        while i < len(sell_indices):
            signal_time = sell_indices[i]
            signal_idx = price_df.index.get_loc(signal_time)
            entry_idx = signal_idx + 1
            if entry_idx >= len(price_df):
                i += 1
                continue

            entry_time = price_df.index[entry_idx]
            entry_row = price_df.iloc[signal_idx]
            entry_price = price_df.iloc[entry_idx]['Open']
            future_df = price_df.iloc[entry_idx+1:entry_idx+1+MAX_HOLD_BARS]

            if len(future_df) == 0:
                i += 1
                continue

            exit_mask = future_df['rsi'] <= RSI_LOW

            if exit_mask.any():
                exit_time = future_df.index[exit_mask.argmax()]
                exit_price = price_df.loc[exit_time, 'Close']
                pips = (entry_price - exit_price) / pip_val
                exit_reason = 'RSI'
            else:
                exit_time = future_df.index[-1]
                exit_price = future_df.iloc[-1]['Close']
                pips = (entry_price - exit_price) / pip_val
                exit_reason = 'TIMEOUT'

            trade = {
                'pair': pair, 'base': base, 'quote': quote, 'type': 'SELL',
                'entry_time': entry_time, 'exit_time': exit_time,
                'pips': pips, 'win': 1 if pips > 0 else 0,
                'spread': SPREADS.get(pair, 2.0), 'exit_reason': exit_reason,
                'base_m5': entry_row['base_m5'], 'quote_m5': entry_row['quote_m5'],
                'base_m15': entry_row['base_m15'], 'quote_m15': entry_row['quote_m15'],
                'base_m30': entry_row['base_m30'], 'quote_m30': entry_row['quote_m30'],
                'base_h1': entry_row['base_h1'], 'quote_h1': entry_row['quote_h1'],
                'base_h4': entry_row['base_h4'], 'quote_h4': entry_row['quote_h4'],
                'base_d1': entry_row.get('base_d1', np.nan), 'quote_d1': entry_row.get('quote_d1', np.nan),
                'base_vel_m5': entry_row['base_vel_m5'], 'quote_vel_m5': entry_row['quote_vel_m5'],
                'base_vel_h1': entry_row['base_vel_h1'], 'quote_vel_h1': entry_row['quote_vel_h1'],
                'mfc_spread_m5': entry_row['mfc_spread_m5'],
                'mfc_spread_h1': entry_row['mfc_spread_h1'],
                'mfc_spread_h4': entry_row['mfc_spread_h4'],
                'rsi': entry_row['rsi'],
            }
            all_trades.append(trade)

            while i < len(sell_indices) and sell_indices[i] <= exit_time:
                i += 1

    except Exception as e:
        log(f"  {pair}: Error - {e}")

# ============================================================================
# ANALYZE MFC PATTERNS
# ============================================================================
log("\n" + "=" * 70)
log("5. MFC PATTERN ANALYSIS")
log("=" * 70)

trades_df = pd.DataFrame(all_trades)
trades_df['net_pips'] = trades_df['pips'] - trades_df['spread']

rsi_trades = trades_df[trades_df['exit_reason'] == 'RSI']
timeout_trades = trades_df[trades_df['exit_reason'] == 'TIMEOUT']

log(f"\nTotal trades: {len(trades_df)}")
log(f"RSI exits: {len(rsi_trades)} ({len(rsi_trades)/len(trades_df)*100:.1f}%)")
log(f"Timeout exits: {len(timeout_trades)} ({len(timeout_trades)/len(trades_df)*100:.1f}%)")

# ============================================================================
# COMPARE MFC VALUES AT ENTRY
# ============================================================================
log("\n" + "-" * 50)
log("MFC VALUES AT ENTRY (Mean)")
log("-" * 50)

mfc_cols = ['base_m5', 'quote_m5', 'base_m15', 'quote_m15', 'base_m30', 'quote_m30',
            'base_h1', 'quote_h1', 'base_h4', 'quote_h4']

log(f"\n{'Metric':<20} {'RSI Exit':>12} {'Timeout':>12} {'Diff':>12}")
log("-" * 56)

for col in mfc_cols:
    rsi_mean = rsi_trades[col].mean()
    timeout_mean = timeout_trades[col].mean()
    diff = timeout_mean - rsi_mean
    log(f"{col:<20} {rsi_mean:>12.4f} {timeout_mean:>12.4f} {diff:>+12.4f}")

# ============================================================================
# MFC VELOCITIES
# ============================================================================
log("\n" + "-" * 50)
log("MFC VELOCITIES AT ENTRY")
log("-" * 50)

vel_cols = ['base_vel_m5', 'quote_vel_m5', 'base_vel_h1', 'quote_vel_h1']

log(f"\n{'Metric':<20} {'RSI Exit':>12} {'Timeout':>12} {'Diff':>12}")
log("-" * 56)

for col in vel_cols:
    rsi_mean = rsi_trades[col].mean()
    timeout_mean = timeout_trades[col].mean()
    diff = timeout_mean - rsi_mean
    log(f"{col:<20} {rsi_mean:>12.4f} {timeout_mean:>12.4f} {diff:>+12.4f}")

# ============================================================================
# MFC SPREADS
# ============================================================================
log("\n" + "-" * 50)
log("MFC SPREAD (Base - Quote) AT ENTRY")
log("-" * 50)

spread_cols = ['mfc_spread_m5', 'mfc_spread_h1', 'mfc_spread_h4']

log(f"\n{'Metric':<20} {'RSI Exit':>12} {'Timeout':>12} {'Diff':>12}")
log("-" * 56)

for col in spread_cols:
    rsi_mean = rsi_trades[col].mean()
    timeout_mean = timeout_trades[col].mean()
    diff = timeout_mean - rsi_mean
    log(f"{col:<20} {rsi_mean:>12.4f} {timeout_mean:>12.4f} {diff:>+12.4f}")

# ============================================================================
# TIMEFRAME ALIGNMENT ANALYSIS
# ============================================================================
log("\n" + "-" * 50)
log("TIMEFRAME ALIGNMENT ANALYSIS")
log("-" * 50)

# For BUY trades: base should be negative across timeframes
# For SELL trades: base should be positive across timeframes
buy_trades = trades_df[trades_df['type'] == 'BUY']
sell_trades = trades_df[trades_df['type'] == 'SELL']

log("\n--- BUY Trades (Base should be negative) ---")
buy_rsi = buy_trades[buy_trades['exit_reason'] == 'RSI']
buy_timeout = buy_trades[buy_trades['exit_reason'] == 'TIMEOUT']

# Check alignment across timeframes
buy_rsi['aligned_m30'] = buy_rsi['base_m30'] < 0
buy_rsi['aligned_h1'] = buy_rsi['base_h1'] < 0
buy_rsi['aligned_h4'] = buy_rsi['base_h4'] < 0

buy_timeout['aligned_m30'] = buy_timeout['base_m30'] < 0
buy_timeout['aligned_h1'] = buy_timeout['base_h1'] < 0
buy_timeout['aligned_h4'] = buy_timeout['base_h4'] < 0

log(f"\nBase < 0 alignment:")
log(f"  M30:  RSI={buy_rsi['aligned_m30'].mean()*100:.1f}%  Timeout={buy_timeout['aligned_m30'].mean()*100:.1f}%")
log(f"  H1:   RSI={buy_rsi['aligned_h1'].mean()*100:.1f}%  Timeout={buy_timeout['aligned_h1'].mean()*100:.1f}%")
log(f"  H4:   RSI={buy_rsi['aligned_h4'].mean()*100:.1f}%  Timeout={buy_timeout['aligned_h4'].mean()*100:.1f}%")

log("\n--- SELL Trades (Base should be positive) ---")
sell_rsi = sell_trades[sell_trades['exit_reason'] == 'RSI']
sell_timeout = sell_trades[sell_trades['exit_reason'] == 'TIMEOUT']

sell_rsi['aligned_m30'] = sell_rsi['base_m30'] > 0
sell_rsi['aligned_h1'] = sell_rsi['base_h1'] > 0
sell_rsi['aligned_h4'] = sell_rsi['base_h4'] > 0

sell_timeout['aligned_m30'] = sell_timeout['base_m30'] > 0
sell_timeout['aligned_h1'] = sell_timeout['base_h1'] > 0
sell_timeout['aligned_h4'] = sell_timeout['base_h4'] > 0

log(f"\nBase > 0 alignment:")
log(f"  M30:  RSI={sell_rsi['aligned_m30'].mean()*100:.1f}%  Timeout={sell_timeout['aligned_m30'].mean()*100:.1f}%")
log(f"  H1:   RSI={sell_rsi['aligned_h1'].mean()*100:.1f}%  Timeout={sell_timeout['aligned_h1'].mean()*100:.1f}%")
log(f"  H4:   RSI={sell_rsi['aligned_h4'].mean()*100:.1f}%  Timeout={sell_timeout['aligned_h4'].mean()*100:.1f}%")

# ============================================================================
# MFC DISTANCE FROM ZERO
# ============================================================================
log("\n" + "-" * 50)
log("MFC DISTANCE FROM ZERO (Absolute Values)")
log("-" * 50)

log(f"\n{'Timeframe':<15} {'RSI base':>10} {'RSI quote':>10} {'TO base':>10} {'TO quote':>10}")
log("-" * 60)

for tf in ['m5', 'm15', 'm30', 'h1', 'h4']:
    base_col = f'base_{tf}'
    quote_col = f'quote_{tf}'

    rsi_base = rsi_trades[base_col].abs().mean()
    rsi_quote = rsi_trades[quote_col].abs().mean()
    to_base = timeout_trades[base_col].abs().mean()
    to_quote = timeout_trades[quote_col].abs().mean()

    log(f"{tf.upper():<15} {rsi_base:>10.4f} {rsi_quote:>10.4f} {to_base:>10.4f} {to_quote:>10.4f}")

# ============================================================================
# MFC REGIONS AT ENTRY
# ============================================================================
log("\n" + "-" * 50)
log("MFC REGIONS AT ENTRY (% of trades)")
log("-" * 50)

def categorize_mfc(value):
    if value >= 0.5:
        return 'High (>0.5)'
    elif value >= 0.2:
        return 'Box+ (0.2-0.5)'
    elif value >= -0.2:
        return 'Inside (-0.2 to 0.2)'
    elif value >= -0.5:
        return 'Box- (-0.5 to -0.2)'
    else:
        return 'Low (<-0.5)'

for tf in ['h1', 'h4']:
    base_col = f'base_{tf}'
    quote_col = f'quote_{tf}'

    rsi_trades[f'{base_col}_region'] = rsi_trades[base_col].apply(categorize_mfc)
    timeout_trades[f'{base_col}_region'] = timeout_trades[base_col].apply(categorize_mfc)

    log(f"\n{tf.upper()} Base Currency Region:")
    rsi_dist = rsi_trades[f'{base_col}_region'].value_counts(normalize=True) * 100
    timeout_dist = timeout_trades[f'{base_col}_region'].value_counts(normalize=True) * 100

    regions = ['High (>0.5)', 'Box+ (0.2-0.5)', 'Inside (-0.2 to 0.2)', 'Box- (-0.5 to -0.2)', 'Low (<-0.5)']
    for region in regions:
        rsi_pct = rsi_dist.get(region, 0)
        to_pct = timeout_dist.get(region, 0)
        log(f"  {region:<25} RSI: {rsi_pct:5.1f}%  Timeout: {to_pct:5.1f}%")

# ============================================================================
# KEY INSIGHTS
# ============================================================================
log("\n" + "=" * 70)
log("6. KEY INSIGHTS FOR FILTERING TIMEOUTS")
log("=" * 70)

# Calculate absolute MFC spread
trades_df['abs_mfc_spread_h1'] = trades_df['mfc_spread_h1'].abs()
trades_df['abs_mfc_spread_h4'] = trades_df['mfc_spread_h4'].abs()

# Binned analysis of MFC spread
log("\n--- MFC Spread (H1) Impact ---")
bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 2.0]
labels = ['0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0', '>1.0']
trades_df['spread_bin'] = pd.cut(trades_df['abs_mfc_spread_h1'], bins=bins, labels=labels)

log(f"\n{'Spread Range':<15} {'Trades':>8} {'Timeout%':>10} {'WR':>10} {'Net Avg':>10}")
log("-" * 55)

for bin_label in labels:
    bin_df = trades_df[trades_df['spread_bin'] == bin_label]
    if len(bin_df) > 10:
        to_pct = (bin_df['exit_reason'] == 'TIMEOUT').mean() * 100
        wr = bin_df['win'].mean() * 100
        net = bin_df['net_pips'].mean()
        log(f"{bin_label:<15} {len(bin_df):>8} {to_pct:>9.1f}% {wr:>9.1f}% {net:>+9.2f}")

# H4 base direction alignment
log("\n--- H4 Base MFC Direction Alignment ---")

# For BUY: base_h4 should be negative (same direction as M5 signal)
# For SELL: base_h4 should be positive
trades_df['h4_aligned'] = (
    ((trades_df['type'] == 'BUY') & (trades_df['base_h4'] < 0)) |
    ((trades_df['type'] == 'SELL') & (trades_df['base_h4'] > 0))
)

aligned = trades_df[trades_df['h4_aligned'] == True]
not_aligned = trades_df[trades_df['h4_aligned'] == False]

log(f"\nH4 Aligned (same dir as trade):")
log(f"  Trades: {len(aligned)}, Timeout: {(aligned['exit_reason']=='TIMEOUT').mean()*100:.1f}%, WR: {aligned['win'].mean()*100:.1f}%, Net: {aligned['net_pips'].mean():.2f}")
log(f"H4 NOT Aligned (opposite dir):")
log(f"  Trades: {len(not_aligned)}, Timeout: {(not_aligned['exit_reason']=='TIMEOUT').mean()*100:.1f}%, WR: {not_aligned['win'].mean()*100:.1f}%, Net: {not_aligned['net_pips'].mean():.2f}")

# Quote velocity
log("\n--- Quote Velocity Impact (H1) ---")
log("(Quote moving in expected direction = good for trade)")

# For BUY: quote should be falling (vel < 0)
# For SELL: quote should be rising (vel > 0)
trades_df['quote_vel_aligned'] = (
    ((trades_df['type'] == 'BUY') & (trades_df['quote_vel_h1'] < 0)) |
    ((trades_df['type'] == 'SELL') & (trades_df['quote_vel_h1'] > 0))
)

q_aligned = trades_df[trades_df['quote_vel_aligned'] == True]
q_not_aligned = trades_df[trades_df['quote_vel_aligned'] == False]

log(f"\nQuote Vel Aligned:")
log(f"  Trades: {len(q_aligned)}, Timeout: {(q_aligned['exit_reason']=='TIMEOUT').mean()*100:.1f}%, WR: {q_aligned['win'].mean()*100:.1f}%, Net: {q_aligned['net_pips'].mean():.2f}")
log(f"Quote Vel NOT Aligned:")
log(f"  Trades: {len(q_not_aligned)}, Timeout: {(q_not_aligned['exit_reason']=='TIMEOUT').mean()*100:.1f}%, WR: {q_not_aligned['win'].mean()*100:.1f}%, Net: {q_not_aligned['net_pips'].mean():.2f}")

# Save detailed trades
trades_df.to_csv(LSTM_DATA_DIR / 'trades_mfc_analysis.csv', index=False)
log(f"\nDetailed trades saved to trades_mfc_analysis.csv")

log(f"\nCompleted: {datetime.now()}")
