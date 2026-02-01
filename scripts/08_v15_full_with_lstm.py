"""
Script 08: Full V1.5 + LSTM Divergence Filter
==============================================
Uses COMPLETE V1.5 logic with ALL filters, THEN adds LSTM divergence requirement.

The idea:
- V1.5 works because it's highly selective (509 trades, 71.3% WR)
- LSTM predicts direction well (~85% accuracy)
- Can LSTM improve V1.5 by filtering out bad trades?

Approach:
- Keep ALL V1.5 filters unchanged
- ADD requirement that LSTM predicts divergence (base up, quote down for BUY)
- Compare results
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
log("V1.5 FULL + LSTM DIVERGENCE FILTER")
log("=" * 70)
log(f"Started: {datetime.now()}")

DATA_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/data')
LSTM_DATA_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/lstm_model/data')
MODEL_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/lstm_model/models')

# Load LSTM config
with open(LSTM_DATA_DIR / 'config.pkl', 'rb') as f:
    config = pickle.load(f)

LOOKBACK = config['lookback']
CURRENCIES = config['currencies']

# V1.5 date range to match LSTM validation data
START_DATE = '2024-07-01'  # Match LSTM validation period
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

# V1.5 thresholds
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

# LSTM thresholds
MIN_LSTM_CONFIDENCE = 0.70  # Start with lower threshold

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
# 1. LOAD LSTM PREDICTIONS
# ============================================================================
log("\n1. Loading LSTM predictions...")

lstm_predictions = {}

for ccy in CURRENCIES:
    log(f"  Processing {ccy}...")

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

    # Build DataFrame indexed by datetime
    datetimes = pd.to_datetime(data['datetimes'][split_idx:])
    lstm_predictions[ccy] = pd.DataFrame({
        'direction': np.argmax(pred[0], axis=1),  # 0=DOWN, 1=NEUTRAL, 2=UP
        'confidence': np.max(pred[0], axis=1)
    }, index=datetimes)

    log(f"    {len(lstm_predictions[ccy])} predictions loaded")

    del model, data, X_val, pred
    tf.keras.backend.clear_session()
    gc.collect()

# ============================================================================
# 2. LOAD MFC DATA (with shifts)
# ============================================================================
log("\n2. Loading MFC data with proper shifts...")

mfc_m5 = {}
mfc_m30_shifted = {}
mfc_h4_shifted = {}

for cur in CURRENCIES:
    # M5
    df = pd.read_csv(DATA_DIR / f'mfc_currency_{cur}_M5.csv')
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y.%m.%d %H:%M')
    df = df.set_index('datetime')
    mfc_m5[cur] = df['MFC']

    # M30 (shifted)
    df = pd.read_csv(DATA_DIR / f'mfc_currency_{cur}_M30.csv')
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y.%m.%d %H:%M')
    df = df.set_index('datetime')
    mfc_m30_shifted[cur] = df['MFC'].shift(1)

    # H4 (shifted)
    df = pd.read_csv(DATA_DIR / 'cleaned' / f'mfc_currency_{cur}_H4_clean.csv')
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y.%m.%d %H:%M')
    df = df.set_index('datetime')
    mfc_h4_shifted[cur] = df['MFC'].shift(1)

log("  Loaded M5, M30 (shifted), H4 (shifted)")

# ============================================================================
# 3. LOAD PRICE DATA
# ============================================================================
log("\n3. Loading price data...")

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
    except Exception as e:
        log(f"  {pair}: ERROR - {e}")

log(f"  Loaded {len(price_data)} pairs")

# ============================================================================
# 4. RUN BACKTESTS
# ============================================================================

def run_backtest(use_lstm_filter=False, min_confidence=0.70):
    """Run V1.5 backtest with optional LSTM divergence filter."""
    all_trades = []
    rejected_by_lstm = 0

    for pair, base, quote in ALL_PAIRS:
        if pair not in price_data:
            continue

        use_base_vel = pair in VELOCITY_PAIRS
        pip_val = get_pip_value(pair)

        try:
            price_df = price_data[pair].copy()

            # M5 MFC - shifted by 1 M5 bar
            price_df['base_mfc'] = mfc_m5[base].shift(1).reindex(price_df.index, method='ffill')
            price_df['quote_mfc'] = mfc_m5[quote].shift(1).reindex(price_df.index, method='ffill')

            # M30 velocity (shifted)
            m30_base = mfc_m30_shifted[base].reindex(price_df.index, method='ffill')
            m30_base_prev = mfc_m30_shifted[base].shift(1).reindex(price_df.index, method='ffill')
            base_vel_m30 = m30_base - m30_base_prev

            # H4 (shifted)
            quote_h4 = mfc_h4_shifted[quote].reindex(price_df.index, method='ffill')

            # M5 velocities
            base_vel_m5 = price_df['base_mfc'] - price_df['base_mfc'].shift(1)
            quote_vel_m5 = price_df['quote_mfc'] - price_df['quote_mfc'].shift(1)

            rsi_series = price_df['rsi']
            rsi_med = rsi_series.rolling(RSI_MEDIAN_PERIOD).median()

            # LSTM predictions (reindex to price data)
            if use_lstm_filter:
                base_lstm = lstm_predictions[base].reindex(price_df.index, method='ffill')
                quote_lstm = lstm_predictions[quote].reindex(price_df.index, method='ffill')

            # BUY signals (V1.5 conditions)
            buy_signal = (
                (rsi_series < RSI_LOW) & (rsi_med < RSI_LOW) &
                (price_df['base_mfc'] <= -MFC_THRESHOLD) &
                (price_df['quote_mfc'] <= QUOTE_THRESHOLD) &
                (quote_vel_m5 < 0) &
                (quote_h4 <= BUY_QUOTE_H4_MAX)
            )
            if use_base_vel:
                buy_signal = buy_signal & (base_vel_m5 > 0)

            # SELL signals (V1.5 conditions)
            sell_signal = (
                (rsi_series > RSI_HIGH) & (rsi_med > RSI_HIGH) &
                (price_df['base_mfc'] >= MFC_THRESHOLD) &
                (price_df['quote_mfc'] >= -QUOTE_THRESHOLD) &
                (quote_vel_m5 > 0) &
                (base_vel_m30 <= SELL_BASE_VEL_M30_MAX)
            )
            if use_base_vel:
                sell_signal = sell_signal & (base_vel_m5 < 0)

            # Process BUY signals
            buy_indices = price_df.index[buy_signal].tolist()
            i = 0
            while i < len(buy_indices):
                signal_time = buy_indices[i]
                signal_idx = price_df.index.get_loc(signal_time)

                # Check LSTM filter if enabled
                if use_lstm_filter:
                    try:
                        base_dir = base_lstm.loc[signal_time, 'direction']
                        base_conf = base_lstm.loc[signal_time, 'confidence']
                        quote_dir = quote_lstm.loc[signal_time, 'direction']
                        quote_conf = quote_lstm.loc[signal_time, 'confidence']

                        # BUY: Base UP (2) + Quote DOWN (0)
                        if not (base_dir == 2 and quote_dir == 0 and
                                base_conf >= min_confidence and quote_conf >= min_confidence):
                            rejected_by_lstm += 1
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

            # Process SELL signals
            sell_indices = price_df.index[sell_signal].tolist()
            i = 0
            while i < len(sell_indices):
                signal_time = sell_indices[i]
                signal_idx = price_df.index.get_loc(signal_time)

                # Check LSTM filter if enabled
                if use_lstm_filter:
                    try:
                        base_dir = base_lstm.loc[signal_time, 'direction']
                        base_conf = base_lstm.loc[signal_time, 'confidence']
                        quote_dir = quote_lstm.loc[signal_time, 'direction']
                        quote_conf = quote_lstm.loc[signal_time, 'confidence']

                        # SELL: Base DOWN (0) + Quote UP (2)
                        if not (base_dir == 0 and quote_dir == 2 and
                                base_conf >= min_confidence and quote_conf >= min_confidence):
                            rejected_by_lstm += 1
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
            log(f"  {pair}: Error - {e}")

    return pd.DataFrame(all_trades), rejected_by_lstm


def print_results(trades_df, label):
    if len(trades_df) == 0:
        log(f"\n{label}: No trades")
        return

    trades_df['net_pips'] = trades_df['pips'] - trades_df['spread']
    wr = trades_df['win'].mean() * 100
    avg_pips = trades_df['pips'].mean()
    avg_net = trades_df['net_pips'].mean()
    total_net = trades_df['net_pips'].sum()

    log(f"\n{label}:")
    log(f"  Trades: {len(trades_df)}")
    log(f"  Win Rate: {wr:.1f}%")
    log(f"  Avg Pips: {avg_pips:.2f}")
    log(f"  Avg Net: {avg_net:.2f}")
    log(f"  Total Net: {total_net:.0f}")


# ============================================================================
# Run backtests
# ============================================================================
log("\n" + "=" * 70)
log("4. RUNNING BACKTESTS")
log("=" * 70)

# Baseline: V1.5 only (on same date range as LSTM validation)
log("\n--- V1.5 BASELINE (Jul-Dec 2024) ---")
v15_trades, _ = run_backtest(use_lstm_filter=False)
print_results(v15_trades, "V1.5 Only")

# With LSTM filter at different confidence levels
for conf in [0.60, 0.70, 0.80, 0.90]:
    log(f"\n--- V1.5 + LSTM (conf >= {conf}) ---")
    lstm_trades, rejected = run_backtest(use_lstm_filter=True, min_confidence=conf)
    print_results(lstm_trades, f"V1.5 + LSTM (>={conf})")
    log(f"  Rejected by LSTM: {rejected}")

# ============================================================================
# Summary
# ============================================================================
log("\n" + "=" * 70)
log("5. SUMMARY")
log("=" * 70)

log("\nKey Question: Does LSTM divergence filter improve V1.5?")
log("If LSTM correctly predicts direction, it should filter out bad trades.")
log("If filtered trades have HIGHER win rate than baseline, LSTM adds value.")

if len(v15_trades) > 0:
    v15_trades['net_pips'] = v15_trades['pips'] - v15_trades['spread']

log(f"\nBaseline (V1.5 only, Jul-Dec 2024):")
if len(v15_trades) > 0:
    log(f"  {len(v15_trades)} trades, {v15_trades['win'].mean()*100:.1f}% WR, {v15_trades['net_pips'].mean():.2f} net avg")

log(f"\nCompleted: {datetime.now()}")
