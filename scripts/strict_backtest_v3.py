"""
Strict LSTM Backtest V3 - Optimized with Batch Predictions
==========================================================
Same strategy as lstm_trader_mt5.py but optimized:
- Pre-generates all LSTM predictions in batch
- Entry at NEXT bar's Open
- RSI-based exit (within 201 bars)
- H1 Velocity Filter: Skip BUY if base falling, skip SELL if base rising
- Realistic spread applied
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
log("STRICT BACKTEST V3 - WITH H1 VELOCITY FILTER")
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
OUTPUT_DIR = LSTM_DIR / 'backtest_results'

# Strategy parameters (from lstm_trader_mt5.py)
MIN_CONFIDENCE = 0.70
MFC_EXTREME = 0.5
RSI_PERIOD = 14
RSI_LOW = 20
RSI_HIGH = 80
MAX_BARS_HOLD = 96

# H1 Velocity Filter (best from filter testing: +5% WR improvement)
H1_VEL_THRESHOLD = 0.04

# Trading costs
SPREAD_PIPS = 1.5
SLIPPAGE_PIPS = 0.5

# Model training period - test must be AFTER this
MODEL_TRAIN_END = '2024-12-31'

# Test ALL pairs
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

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

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
# 1. LOAD MFC DATA (2025 out-of-sample period)
# ============================================================================
log("1. Loading MFC data for 2025...")

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

    log(f"   {ccy}: M5={len(mfc_m5[ccy]):,} bars")

# ============================================================================
# 2. PRE-GENERATE ALL LSTM PREDICTIONS (BATCH)
# ============================================================================
log("\n2. Generating LSTM predictions (batch)...")

lstm_predictions = {}

for ccy in CURRENCIES:
    log(f"   Processing {ccy}...")

    # Load model
    model = tf.keras.models.load_model(MODEL_DIR / f'lstm_{ccy}_final.keras')

    # Prepare data with shift(1) to avoid lookahead
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

    # Create sequences
    max_lb = max(LOOKBACK.values())
    valid_start = max_lb + 1
    n_samples = len(m5_data) - valid_start - 1

    if n_samples <= 0:
        log(f"      Not enough data")
        continue

    X_M5 = np.array([m5_data[i-LOOKBACK['M5']:i] for i in range(valid_start, valid_start+n_samples)])
    X_M15 = np.array([m15_data[i-LOOKBACK['M15']:i] for i in range(valid_start, valid_start+n_samples)])
    X_M30 = np.array([m30_data[i-LOOKBACK['M30']:i] for i in range(valid_start, valid_start+n_samples)])
    X_H1 = np.array([h1_data[i-LOOKBACK['H1']:i] for i in range(valid_start, valid_start+n_samples)])
    X_H4 = np.array([h4_data[i-LOOKBACK['H4']:i] for i in range(valid_start, valid_start+n_samples)])

    # Aux features
    vel_m5 = np.diff(m5_data, prepend=m5_data[0])
    vel_m30 = np.diff(m30_data, prepend=m30_data[0])

    X_aux = np.column_stack([
        vel_m5[valid_start:valid_start+n_samples],
        vel_m30[valid_start:valid_start+n_samples],
        m5_data[valid_start:valid_start+n_samples],
        m30_data[valid_start:valid_start+n_samples],
        h4_data[valid_start:valid_start+n_samples],
    ])

    # Batch predict
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

    log(f"      {len(lstm_predictions[ccy]):,} predictions")

    del model, X_val, pred
    tf.keras.backend.clear_session()
    gc.collect()

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

        # Resample to M5
        df_m5 = df.resample('5min').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
        }).dropna()

        # Calculate RSI (shifted by 1)
        df_m5['RSI'] = calculate_rsi(df_m5['Close'], RSI_PERIOD)
        df_m5['RSI'] = df_m5['RSI'].shift(1)

        price_data[pair] = df_m5
        log(f"   {pair}: {len(df_m5):,} bars")

    except Exception as e:
        log(f"   {pair}: ERROR - {e}")

# ============================================================================
# 4. RUN BACKTEST
# ============================================================================
log("\n4. Running backtest...")

all_trades = []

for pair, base, quote in ALL_PAIRS:
    if pair not in price_data:
        continue
    if base not in lstm_predictions or quote not in lstm_predictions:
        continue

    df_price = price_data[pair].copy()
    pip_value = get_pip_value(pair)
    spread = SPREADS.get(pair, 2.0)

    # Add MFC (shifted)
    df_price['base_mfc'] = mfc_m5[base].shift(1).reindex(df_price.index, method='ffill')
    df_price['quote_mfc'] = mfc_m5[quote].shift(1).reindex(df_price.index, method='ffill')

    # Add H1 velocity for filter (change over last 12 M5 bars = 1 hour)
    base_h1_series = mfc_h1[base].shift(1).reindex(df_price.index, method='ffill')
    df_price['base_vel_h1'] = base_h1_series - base_h1_series.shift(12)

    # Add LSTM predictions
    base_lstm = lstm_predictions[base].reindex(df_price.index, method='ffill')
    quote_lstm = lstm_predictions[quote].reindex(df_price.index, method='ffill')

    df_price['base_dir'] = base_lstm['direction']
    df_price['base_conf'] = base_lstm['confidence']
    df_price['quote_dir'] = quote_lstm['direction']
    df_price['quote_conf'] = quote_lstm['confidence']

    df_price = df_price.dropna()

    if len(df_price) == 0:
        continue

    # Find signals (with H1 velocity filter)
    buy_signal = (
        (df_price['base_dir'] == 2) &
        (df_price['quote_dir'] == 0) &
        (df_price['base_conf'] >= MIN_CONFIDENCE) &
        (df_price['quote_conf'] >= MIN_CONFIDENCE) &
        (df_price['base_mfc'] <= -MFC_EXTREME) &
        (df_price['RSI'] < RSI_LOW) &
        (df_price['base_vel_h1'] >= -H1_VEL_THRESHOLD)  # Skip if base falling on H1
    )

    sell_signal = (
        (df_price['base_dir'] == 0) &
        (df_price['quote_dir'] == 2) &
        (df_price['base_conf'] >= MIN_CONFIDENCE) &
        (df_price['quote_conf'] >= MIN_CONFIDENCE) &
        (df_price['base_mfc'] >= MFC_EXTREME) &
        (df_price['RSI'] > RSI_HIGH) &
        (df_price['base_vel_h1'] <= H1_VEL_THRESHOLD)  # Skip if base rising on H1
    )

    # Process trades - matching 14_test_2025 logic
    # Look for RSI exit within 201 bars, skip if not found
    pair_trades = []
    MAX_LOOK_FORWARD = 201

    # Process BUY signals
    buy_indices = df_price.index[buy_signal].tolist()
    i = 0
    while i < len(buy_indices):
        signal_time = buy_indices[i]
        signal_idx = df_price.index.get_loc(signal_time)

        # Entry at next bar's Open
        if signal_idx + 1 >= len(df_price):
            i += 1
            continue

        entry_idx = signal_idx + 1
        entry_time = df_price.index[entry_idx]
        entry_price = df_price.iloc[entry_idx]['Open']

        # Look forward for RSI exit
        future_df = df_price.iloc[entry_idx+1:entry_idx+1+MAX_LOOK_FORWARD]

        if len(future_df) == 0:
            i += 1
            continue

        exit_mask = future_df['RSI'] >= RSI_HIGH

        if exit_mask.any():
            exit_idx = exit_mask.argmax()
            exit_time = future_df.index[exit_idx]
            exit_price = future_df.iloc[exit_idx]['Close']
            bars_held = exit_idx + 1

            # Calculate P/L with costs
            cost = spread * pip_value
            pips = (exit_price - entry_price - cost) / pip_value

            pair_trades.append({
                'pair': pair,
                'signal': 'BUY',
                'entry_time': entry_time,
                'exit_time': exit_time,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'bars_held': bars_held,
                'exit_reason': f"RSI={future_df.iloc[exit_idx]['RSI']:.1f}",
                'pips': pips,
            })

            # Skip signals until after exit
            while i < len(buy_indices) and buy_indices[i] <= exit_time:
                i += 1
        else:
            i += 1

    # Process SELL signals
    sell_indices = df_price.index[sell_signal].tolist()
    i = 0
    while i < len(sell_indices):
        signal_time = sell_indices[i]
        signal_idx = df_price.index.get_loc(signal_time)

        # Entry at next bar's Open
        if signal_idx + 1 >= len(df_price):
            i += 1
            continue

        entry_idx = signal_idx + 1
        entry_time = df_price.index[entry_idx]
        entry_price = df_price.iloc[entry_idx]['Open']

        # Look forward for RSI exit
        future_df = df_price.iloc[entry_idx+1:entry_idx+1+MAX_LOOK_FORWARD]

        if len(future_df) == 0:
            i += 1
            continue

        exit_mask = future_df['RSI'] <= RSI_LOW

        if exit_mask.any():
            exit_idx = exit_mask.argmax()
            exit_time = future_df.index[exit_idx]
            exit_price = future_df.iloc[exit_idx]['Close']
            bars_held = exit_idx + 1

            # Calculate P/L with costs
            cost = spread * pip_value
            pips = (entry_price - exit_price - cost) / pip_value

            pair_trades.append({
                'pair': pair,
                'signal': 'SELL',
                'entry_time': entry_time,
                'exit_time': exit_time,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'bars_held': bars_held,
                'exit_reason': f"RSI={future_df.iloc[exit_idx]['RSI']:.1f}",
                'pips': pips,
            })

            # Skip signals until after exit
            while i < len(sell_indices) and sell_indices[i] <= exit_time:
                i += 1
        else:
            i += 1

    log(f"   {pair}: {len(pair_trades)} trades")
    all_trades.extend(pair_trades)

# ============================================================================
# 5. RESULTS
# ============================================================================
log("\n" + "=" * 80)
log("BACKTEST RESULTS - 2025 OUT-OF-SAMPLE (WITH H1 FILTER)")
log("=" * 80)
log(f"\nH1 Velocity Filter: threshold = {H1_VEL_THRESHOLD}")

if len(all_trades) == 0:
    log("\nNo trades generated!")
else:
    df_trades = pd.DataFrame(all_trades)
    df_trades['win'] = (df_trades['pips'] > 0).astype(int)

    log(f"\nOVERALL RESULTS:")
    log(f"   Total trades: {len(df_trades)}")
    log(f"   Win rate: {df_trades['win'].mean()*100:.1f}%")
    log(f"   Avg pips: {df_trades['pips'].mean():.2f}")
    log(f"   Total pips: {df_trades['pips'].sum():.0f}")

    # By signal
    log(f"\nBY SIGNAL:")
    for sig in ['BUY', 'SELL']:
        sig_df = df_trades[df_trades['signal'] == sig]
        if len(sig_df) > 0:
            wr = sig_df['win'].mean() * 100
            avg = sig_df['pips'].mean()
            log(f"   {sig}: {len(sig_df)} trades, {wr:.1f}% WR, {avg:.2f} avg pips")

    # By pair (top 10)
    log(f"\nBY PAIR (top 10 by net pips):")
    pair_stats = df_trades.groupby('pair').agg({
        'pips': ['count', 'mean', 'sum'],
        'win': 'mean'
    })
    pair_stats.columns = ['trades', 'avg_pips', 'total_pips', 'wr']
    pair_stats = pair_stats.sort_values('total_pips', ascending=False)
    for pair, row in pair_stats.head(10).iterrows():
        log(f"   {pair}: {int(row['trades'])} trades, {row['wr']*100:.1f}% WR, {row['avg_pips']:.2f} avg, {row['total_pips']:.0f} total")

    # By exit reason
    log(f"\nBY EXIT REASON:")
    for reason in df_trades['exit_reason'].unique():
        reason_df = df_trades[df_trades['exit_reason'] == reason]
        wr = reason_df['win'].mean() * 100
        avg = reason_df['pips'].mean()
        log(f"   {reason}: {len(reason_df)} trades, {wr:.1f}% WR, {avg:.2f} avg pips")

    # Save
    output_path = OUTPUT_DIR / f'strict_backtest_v3_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    df_trades.to_csv(output_path, index=False)
    log(f"\n   Results saved to: {output_path}")

log(f"\n{'='*80}")
log(f"Completed: {datetime.now()}")
log("=" * 80)
