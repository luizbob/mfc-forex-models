"""
Script 10: Analyze Rejected Trades
==================================
Compare performance of trades that LSTM keeps vs rejects.
If LSTM is working correctly, rejected trades should have worse results.
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
log("ANALYZE LSTM REJECTED TRADES")
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
# RUN BACKTEST WITH CATEGORIZATION
# ============================================================================
log("\n2. Running backtest with trade categorization...")

MIN_CONF = 0.70

kept_trades = []
rejected_trades = []

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

            # Get LSTM data
            try:
                base_dir = base_lstm.loc[signal_time, 'direction']
                base_conf = base_lstm.loc[signal_time, 'confidence']
                quote_dir = quote_lstm.loc[signal_time, 'direction']
                quote_conf = quote_lstm.loc[signal_time, 'confidence']
            except:
                i += 1
                continue

            # Check divergence: BUY needs base UP (2) + quote DOWN (0)
            divergence_ok = (base_dir == 2 and quote_dir == 0 and
                            base_conf >= MIN_CONF and quote_conf >= MIN_CONF)

            entry_price = price_df.loc[signal_time, 'Close']
            future_df = price_df.iloc[signal_idx+1:signal_idx+201]
            exit_mask = future_df['rsi'] >= RSI_HIGH

            if exit_mask.any():
                exit_time = future_df.index[exit_mask.argmax()]
                exit_price = price_df.loc[exit_time, 'Close']
                pips = (exit_price - entry_price) / pip_val

                trade_data = {
                    'pair': pair, 'type': 'BUY', 'entry_time': signal_time,
                    'pips': pips, 'win': 1 if pips > 0 else 0,
                    'spread': SPREADS.get(pair, 2.0),
                    'base_dir': base_dir, 'base_conf': base_conf,
                    'quote_dir': quote_dir, 'quote_conf': quote_conf,
                    'rejection_reason': None
                }

                if divergence_ok:
                    kept_trades.append(trade_data)
                else:
                    # Categorize rejection reason
                    if base_dir != 2:
                        trade_data['rejection_reason'] = f"base_pred={['DOWN','NEUT','UP'][int(base_dir)]}"
                    elif quote_dir != 0:
                        trade_data['rejection_reason'] = f"quote_pred={['DOWN','NEUT','UP'][int(quote_dir)]}"
                    elif base_conf < MIN_CONF:
                        trade_data['rejection_reason'] = f"base_conf={base_conf:.2f}"
                    else:
                        trade_data['rejection_reason'] = f"quote_conf={quote_conf:.2f}"
                    rejected_trades.append(trade_data)

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

            try:
                base_dir = base_lstm.loc[signal_time, 'direction']
                base_conf = base_lstm.loc[signal_time, 'confidence']
                quote_dir = quote_lstm.loc[signal_time, 'direction']
                quote_conf = quote_lstm.loc[signal_time, 'confidence']
            except:
                i += 1
                continue

            # Check divergence: SELL needs base DOWN (0) + quote UP (2)
            divergence_ok = (base_dir == 0 and quote_dir == 2 and
                            base_conf >= MIN_CONF and quote_conf >= MIN_CONF)

            entry_price = price_df.loc[signal_time, 'Close']
            future_df = price_df.iloc[signal_idx+1:signal_idx+201]
            exit_mask = future_df['rsi'] <= RSI_LOW

            if exit_mask.any():
                exit_time = future_df.index[exit_mask.argmax()]
                exit_price = price_df.loc[exit_time, 'Close']
                pips = (entry_price - exit_price) / pip_val

                trade_data = {
                    'pair': pair, 'type': 'SELL', 'entry_time': signal_time,
                    'pips': pips, 'win': 1 if pips > 0 else 0,
                    'spread': SPREADS.get(pair, 2.0),
                    'base_dir': base_dir, 'base_conf': base_conf,
                    'quote_dir': quote_dir, 'quote_conf': quote_conf,
                    'rejection_reason': None
                }

                if divergence_ok:
                    kept_trades.append(trade_data)
                else:
                    if base_dir != 0:
                        trade_data['rejection_reason'] = f"base_pred={['DOWN','NEUT','UP'][int(base_dir)]}"
                    elif quote_dir != 2:
                        trade_data['rejection_reason'] = f"quote_pred={['DOWN','NEUT','UP'][int(quote_dir)]}"
                    elif base_conf < MIN_CONF:
                        trade_data['rejection_reason'] = f"base_conf={base_conf:.2f}"
                    else:
                        trade_data['rejection_reason'] = f"quote_conf={quote_conf:.2f}"
                    rejected_trades.append(trade_data)

                while i < len(sell_indices) and sell_indices[i] <= exit_time:
                    i += 1
            else:
                i += 1

    except Exception as e:
        pass

# ============================================================================
# RESULTS
# ============================================================================
log("\n" + "=" * 70)
log("3. RESULTS")
log("=" * 70)

kept_df = pd.DataFrame(kept_trades) if kept_trades else pd.DataFrame()
rejected_df = pd.DataFrame(rejected_trades) if rejected_trades else pd.DataFrame()

if len(kept_df) > 0:
    kept_df['net_pips'] = kept_df['pips'] - kept_df['spread']
if len(rejected_df) > 0:
    rejected_df['net_pips'] = rejected_df['pips'] - rejected_df['spread']

log(f"\n{'Category':<20} {'Trades':>8} {'WR%':>8} {'Net Avg':>10} {'Total':>10}")
log("-" * 60)

if len(kept_df) > 0:
    log(f"{'KEPT by LSTM':<20} {len(kept_df):>8} {kept_df['win'].mean()*100:>7.1f}% {kept_df['net_pips'].mean():>10.2f} {kept_df['net_pips'].sum():>10.0f}")
if len(rejected_df) > 0:
    log(f"{'REJECTED by LSTM':<20} {len(rejected_df):>8} {rejected_df['win'].mean()*100:>7.1f}% {rejected_df['net_pips'].mean():>10.2f} {rejected_df['net_pips'].sum():>10.0f}")

total = len(kept_df) + len(rejected_df)
if total > 0:
    all_trades = pd.concat([kept_df, rejected_df]) if len(rejected_df) > 0 else kept_df
    log(f"{'ALL V1.5 TRADES':<20} {len(all_trades):>8} {all_trades['win'].mean()*100:>7.1f}% {all_trades['net_pips'].mean():>10.2f} {all_trades['net_pips'].sum():>10.0f}")

# ============================================================================
# REJECTION ANALYSIS
# ============================================================================
log("\n" + "=" * 70)
log("4. REJECTION REASON ANALYSIS")
log("=" * 70)

if len(rejected_df) > 0:
    log("\nBreakdown of rejected trades by reason:")

    # Group by main rejection type
    rejected_df['reason_type'] = rejected_df['rejection_reason'].apply(
        lambda x: x.split('=')[0] if x else 'unknown'
    )

    for reason_type in rejected_df['reason_type'].unique():
        reason_df = rejected_df[rejected_df['reason_type'] == reason_type]
        if len(reason_df) > 0:
            wr = reason_df['win'].mean() * 100
            net_avg = reason_df['net_pips'].mean()
            log(f"  {reason_type}: {len(reason_df)} trades, {wr:.1f}% WR, {net_avg:.2f} net avg")

    # More detailed breakdown
    log("\nDetailed rejection reasons:")
    reason_stats = rejected_df.groupby('rejection_reason').agg({
        'pips': 'count',
        'win': 'mean',
        'net_pips': 'mean'
    }).rename(columns={'pips': 'trades', 'win': 'wr'})
    reason_stats = reason_stats.sort_values('trades', ascending=False)

    for reason, row in reason_stats.head(15).iterrows():
        log(f"  {reason}: {int(row['trades'])} trades, {row['wr']*100:.1f}% WR, {row['net_pips']:.2f} net avg")

# ============================================================================
# KEY INSIGHT
# ============================================================================
log("\n" + "=" * 70)
log("5. KEY INSIGHT")
log("=" * 70)

if len(kept_df) > 0 and len(rejected_df) > 0:
    kept_wr = kept_df['win'].mean() * 100
    rejected_wr = rejected_df['win'].mean() * 100
    kept_net = kept_df['net_pips'].mean()
    rejected_net = rejected_df['net_pips'].mean()

    log(f"\nKept trades:     {kept_wr:.1f}% WR, {kept_net:.2f} net avg")
    log(f"Rejected trades: {rejected_wr:.1f}% WR, {rejected_net:.2f} net avg")
    log(f"Difference:      {kept_wr - rejected_wr:+.1f}% WR, {kept_net - rejected_net:+.2f} net avg")

    if kept_wr > rejected_wr and kept_net > rejected_net:
        log("\n>>> LSTM IS WORKING: Kept trades are better than rejected trades!")
    elif kept_wr > rejected_wr:
        log("\n>>> LSTM improves win rate but not net pips")
    elif kept_net > rejected_net:
        log("\n>>> LSTM improves net pips but not win rate")
    else:
        log("\n>>> WARNING: LSTM may not be adding value")

log(f"\nCompleted: {datetime.now()}")
