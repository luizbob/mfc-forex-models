"""
Script 14: 2025 Out-of-Sample Test
==================================
TRUE out-of-sample validation on 2025 data.

Data Limits:
- EUR/NZD pairs: M5/M15/M30 ends April 18, 2025
- Other pairs: M5/M15/M30 ends July 23, 2025
- H1/H4 cleaned: July 25, 2025 for all

Strategy: Use the LSTM models trained on 2023-mid2024 data
to predict on 2025 data they've never seen.
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
log("2025 OUT-OF-SAMPLE TEST")
log("=" * 70)
log(f"Started: {datetime.now()}")

MODEL_ACCURACY = {
    'JPY': 90.0, 'USD': 87.8, 'AUD': 87.4, 'NZD': 86.7,
    'CHF': 86.3, 'GBP': 83.7, 'CAD': 82.2, 'EUR': 82.0,
}

# Data limits per currency (M5 is the limiting factor)
DATA_END_DATES = {
    'EUR': '2025-04-18',
    'NZD': '2025-04-18',
    'USD': '2025-07-23',
    'GBP': '2025-07-23',
    'JPY': '2025-07-23',
    'CHF': '2025-07-23',
    'CAD': '2025-07-23',
    'AUD': '2025-07-23',
}

DATA_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/data')
LSTM_DATA_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/lstm_model/data')
MODEL_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/lstm_model/models')

with open(LSTM_DATA_DIR / 'config.pkl', 'rb') as f:
    config = pickle.load(f)

LOOKBACK = config['lookback']
CURRENCIES = config['currencies']

# 2025 test period
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
    """Get the earliest end date between base and quote currencies."""
    base_end = DATA_END_DATES[base]
    quote_end = DATA_END_DATES[quote]
    return min(base_end, quote_end)

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
# LOAD MFC DATA FOR 2025
# ============================================================================
log("\n1. Loading MFC data for 2025...")

mfc_m5 = {}
mfc_m15 = {}
mfc_m30 = {}
mfc_h1 = {}
mfc_h4 = {}

for cur in CURRENCIES:
    end_date = DATA_END_DATES[cur]

    # M5
    df = pd.read_csv(DATA_DIR / f'mfc_currency_{cur}_M5.csv')
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y.%m.%d %H:%M')
    df = df.set_index('datetime')
    df = df[(df.index >= START_DATE) & (df.index <= end_date)]
    mfc_m5[cur] = df['MFC']

    # M15
    df = pd.read_csv(DATA_DIR / f'mfc_currency_{cur}_M15.csv')
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y.%m.%d %H:%M')
    df = df.set_index('datetime')
    df = df[(df.index >= START_DATE) & (df.index <= end_date)]
    mfc_m15[cur] = df['MFC']

    # M30
    df = pd.read_csv(DATA_DIR / f'mfc_currency_{cur}_M30.csv')
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y.%m.%d %H:%M')
    df = df.set_index('datetime')
    df = df[(df.index >= START_DATE) & (df.index <= end_date)]
    mfc_m30[cur] = df['MFC']

    # H1 (cleaned)
    df = pd.read_csv(DATA_DIR / 'cleaned' / f'mfc_currency_{cur}_H1_clean.csv')
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y.%m.%d %H:%M')
    df = df.set_index('datetime')
    df = df[(df.index >= START_DATE) & (df.index <= end_date)]
    mfc_h1[cur] = df['MFC']

    # H4 (cleaned)
    df = pd.read_csv(DATA_DIR / 'cleaned' / f'mfc_currency_{cur}_H4_clean.csv')
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y.%m.%d %H:%M')
    df = df.set_index('datetime')
    df = df[(df.index >= START_DATE) & (df.index <= end_date)]
    mfc_h4[cur] = df['MFC']

    log(f"  {cur}: M5={len(mfc_m5[cur])}, ends {end_date}")

# ============================================================================
# LOAD LSTM MODELS AND GENERATE 2025 PREDICTIONS
# ============================================================================
log("\n2. Generating LSTM predictions for 2025...")

lstm_predictions = {}

for ccy in CURRENCIES:
    log(f"  Processing {ccy}...")

    end_date = DATA_END_DATES[ccy]

    # Load model
    model = tf.keras.models.load_model(MODEL_DIR / f'lstm_{ccy}_final.keras')

    # Prepare 2025 data for prediction
    # IMPORTANT: Apply shift(1) to match training data preparation!
    # Training used shifted data, so we must do the same here
    m5_idx = mfc_m5[ccy].index

    # Shift all data by 1 to avoid lookahead (same as training)
    m5_shifted = mfc_m5[ccy].shift(1)
    m15_shifted = mfc_m15[ccy].shift(1).reindex(m5_idx, method='ffill')
    m30_shifted = mfc_m30[ccy].shift(1).reindex(m5_idx, method='ffill')
    h1_shifted = mfc_h1[ccy].shift(1).reindex(m5_idx, method='ffill')
    h4_shifted = mfc_h4[ccy].shift(1).reindex(m5_idx, method='ffill')

    # Convert to arrays (drop first NaN from shift)
    m5_data = m5_shifted.values
    m15_data = m15_shifted.values
    m30_data = m30_shifted.values
    h1_data = h1_shifted.values
    h4_data = h4_shifted.values

    # Create sequences - match training's valid_start = max_lookback + 1
    max_lb = max(LOOKBACK.values())
    valid_start = max_lb + 1  # Match training
    n_samples = len(m5_data) - valid_start - 1

    if n_samples <= 0:
        log(f"    Not enough data for {ccy}")
        continue

    # Windows: [i-lookback:i] to match training
    X_M5 = np.array([m5_data[i-LOOKBACK['M5']:i] for i in range(valid_start, valid_start+n_samples)])
    X_M15 = np.array([m15_data[i-LOOKBACK['M15']:i] for i in range(valid_start, valid_start+n_samples)])
    X_M30 = np.array([m30_data[i-LOOKBACK['M30']:i] for i in range(valid_start, valid_start+n_samples)])
    X_H1 = np.array([h1_data[i-LOOKBACK['H1']:i] for i in range(valid_start, valid_start+n_samples)])
    X_H4 = np.array([h4_data[i-LOOKBACK['H4']:i] for i in range(valid_start, valid_start+n_samples)])

    # Aux features: vel_M5, vel_M30, current_M5, current_M30, current_H4
    # Velocities on shifted data
    vel_m5 = np.diff(m5_data, prepend=m5_data[0])
    vel_m30 = np.diff(m30_data, prepend=m30_data[0])

    X_aux = np.column_stack([
        vel_m5[valid_start:valid_start+n_samples],
        vel_m30[valid_start:valid_start+n_samples],
        m5_data[valid_start:valid_start+n_samples],
        m30_data[valid_start:valid_start+n_samples],
        h4_data[valid_start:valid_start+n_samples],
    ])

    # Predict
    X_val = [
        X_M5.reshape(-1, LOOKBACK['M5'], 1),
        X_M15.reshape(-1, LOOKBACK['M15'], 1),
        X_M30.reshape(-1, LOOKBACK['M30'], 1),
        X_H1.reshape(-1, LOOKBACK['H1'], 1),
        X_H4.reshape(-1, LOOKBACK['H4'], 1),
        X_aux,
    ]

    pred = model.predict(X_val, verbose=0, batch_size=256)

    # Get datetimes for predictions - match training alignment
    datetimes = mfc_m5[ccy].index[valid_start:valid_start+n_samples]

    lstm_predictions[ccy] = pd.DataFrame({
        'direction': np.argmax(pred[0], axis=1),
        'confidence': np.max(pred[0], axis=1)
    }, index=datetimes)

    log(f"    {len(lstm_predictions[ccy])} predictions generated")

    del model, X_val, pred
    tf.keras.backend.clear_session()
    gc.collect()

# ============================================================================
# LOAD PRICE DATA FOR 2025
# ============================================================================
log("\n3. Loading price data for 2025...")

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
            log(f"  {pair}: {len(price_m5)} bars (until {end_date})")
    except Exception as e:
        log(f"  {pair}: ERROR - {e}")

# ============================================================================
# RUN STRATEGY ON 2025 DATA
# ============================================================================
log("\n" + "=" * 70)
log("4. RUNNING STRATEGY ON 2025 DATA")
log("=" * 70)

MIN_CONF = 0.70
MFC_EXTREME = 0.5
RSI_LOW = 20
RSI_HIGH = 80

all_trades = []

for pair, base, quote in ALL_PAIRS:
    if pair not in price_data:
        continue
    if base not in lstm_predictions or quote not in lstm_predictions:
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

        # Drop rows with NaN and reset
        price_df = price_df.dropna()

        rsi = price_df['rsi']

        # LSTM divergence + MFC extreme + RSI extreme
        buy_signal = (
            (price_df['base_dir'] == 2) &
            (price_df['quote_dir'] == 0) &
            (price_df['base_conf'] >= MIN_CONF) &
            (price_df['quote_conf'] >= MIN_CONF) &
            (price_df['base_mfc'] <= -MFC_EXTREME) &
            (price_df['rsi'] < RSI_LOW)
        )

        sell_signal = (
            (price_df['base_dir'] == 0) &
            (price_df['quote_dir'] == 2) &
            (price_df['base_conf'] >= MIN_CONF) &
            (price_df['quote_conf'] >= MIN_CONF) &
            (price_df['base_mfc'] >= MFC_EXTREME) &
            (price_df['rsi'] > RSI_HIGH)
        )

        # Process BUY
        buy_indices = price_df.index[buy_signal].tolist()
        i = 0
        while i < len(buy_indices):
            signal_time = buy_indices[i]
            signal_idx = price_df.index.get_loc(signal_time)

            entry_price = price_df.loc[signal_time, 'Close']
            future_df = price_df.iloc[signal_idx+1:signal_idx+201]

            if len(future_df) == 0:
                i += 1
                continue

            exit_mask = future_df['rsi'] >= RSI_HIGH

            if exit_mask.any():
                exit_time = future_df.index[exit_mask.argmax()]
                exit_price = price_df.loc[exit_time, 'Close']
                pips = (exit_price - entry_price) / pip_val

                all_trades.append({
                    'pair': pair, 'type': 'BUY', 'entry_time': signal_time,
                    'exit_time': exit_time,
                    'pips': pips, 'win': 1 if pips > 0 else 0,
                    'spread': SPREADS.get(pair, 2.0),
                    'base_acc': MODEL_ACCURACY[base],
                    'quote_acc': MODEL_ACCURACY[quote],
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

            if len(future_df) == 0:
                i += 1
                continue

            exit_mask = future_df['rsi'] <= RSI_LOW

            if exit_mask.any():
                exit_time = future_df.index[exit_mask.argmax()]
                exit_price = price_df.loc[exit_time, 'Close']
                pips = (entry_price - exit_price) / pip_val

                all_trades.append({
                    'pair': pair, 'type': 'SELL', 'entry_time': signal_time,
                    'exit_time': exit_time,
                    'pips': pips, 'win': 1 if pips > 0 else 0,
                    'spread': SPREADS.get(pair, 2.0),
                    'base_acc': MODEL_ACCURACY[base],
                    'quote_acc': MODEL_ACCURACY[quote],
                })

                while i < len(sell_indices) and sell_indices[i] <= exit_time:
                    i += 1
            else:
                i += 1

    except Exception as e:
        log(f"  {pair}: Error - {e}")

# ============================================================================
# RESULTS
# ============================================================================
log("\n" + "=" * 70)
log("5. 2025 OUT-OF-SAMPLE RESULTS")
log("=" * 70)

if len(all_trades) == 0:
    log("\nNo trades generated!")
else:
    trades_df = pd.DataFrame(all_trades)
    trades_df['net_pips'] = trades_df['pips'] - trades_df['spread']
    trades_df['min_acc'] = trades_df[['base_acc', 'quote_acc']].min(axis=1)

    log(f"\n--- ALL PAIRS ---")
    log(f"  Trades: {len(trades_df)}")
    log(f"  Win Rate: {trades_df['win'].mean()*100:.1f}%")
    log(f"  Avg Pips: {trades_df['pips'].mean():.2f}")
    log(f"  Net Avg: {trades_df['net_pips'].mean():.2f}")
    log(f"  Total Net: {trades_df['net_pips'].sum():.0f}")

    # By accuracy filter
    log(f"\n--- BY MODEL ACCURACY ---")
    for min_acc in [85, 86, 87]:
        filtered = trades_df[trades_df['min_acc'] >= min_acc]
        if len(filtered) > 0:
            log(f"  Min acc >= {min_acc}%: {len(filtered)} trades, {filtered['win'].mean()*100:.1f}% WR, {filtered['net_pips'].mean():.2f} net avg")

    # By pair
    log(f"\n--- BY PAIR (top 10) ---")
    pair_stats = trades_df.groupby('pair').agg({
        'pips': 'count', 'win': 'mean', 'net_pips': 'mean'
    }).rename(columns={'pips': 'trades', 'win': 'wr'})
    pair_stats = pair_stats.sort_values('net_pips', ascending=False)
    for pair, row in pair_stats.head(10).iterrows():
        log(f"  {pair}: {int(row['trades'])} trades, {row['wr']*100:.1f}% WR, {row['net_pips']:.2f} net avg")

    # Hold time analysis
    trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
    trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])
    trades_df['hold_minutes'] = (trades_df['exit_time'] - trades_df['entry_time']).dt.total_seconds() / 60
    trades_df['hold_hours'] = trades_df['hold_minutes'] / 60

    log(f"\n--- HOLD TIME ---")
    log(f"  Average: {trades_df['hold_hours'].mean():.1f} hours ({trades_df['hold_minutes'].mean():.0f} min)")
    log(f"  Median:  {trades_df['hold_hours'].median():.1f} hours ({trades_df['hold_minutes'].median():.0f} min)")
    log(f"  Min:     {trades_df['hold_hours'].min():.1f} hours")
    log(f"  Max:     {trades_df['hold_hours'].max():.1f} hours")

    # Save trades
    trades_df.to_csv(LSTM_DATA_DIR / 'trades_2025_oos.csv', index=False)
    log(f"\n  Trades saved to trades_2025_oos.csv")

# ============================================================================
# COMPARISON
# ============================================================================
log("\n" + "=" * 70)
log("6. COMPARISON: 2024 VALIDATION vs 2025 OUT-OF-SAMPLE")
log("=" * 70)

if len(all_trades) > 0:
    log(f"| Period        | Trades | Win Rate | Net Avg | Total   |")
    log(f"|---------------|--------|----------|---------|---------|")
    log(f"| 2024 Val      | 2,711  | 66.0%    | +3.97   | +10,768 |")
    log(f"| 2025 OOS      | {len(trades_df):<6} | {trades_df['win'].mean()*100:.1f}%    | {trades_df['net_pips'].mean():+.2f}   | {trades_df['net_pips'].sum():+.0f} |")

log(f"\nCompleted: {datetime.now()}")
