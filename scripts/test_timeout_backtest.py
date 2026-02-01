"""
Test different timeout values with actual backtest.
"""
import warnings
warnings.filterwarnings('ignore')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import keras
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from pathlib import Path
import gc

def log(msg=""):
    print(msg, flush=True)

log("=" * 70)
log("TIMEOUT OPTIMIZATION BACKTEST")
log("=" * 70)

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

# Load all data once
log("\n1. Loading data...")

mfc_m5, mfc_m15, mfc_m30, mfc_h1, mfc_h4 = {}, {}, {}, {}, {}

for cur in CURRENCIES:
    end_date = DATA_END_DATES[cur]
    for tf, mfc_dict, folder in [('M5', mfc_m5, 'cleaned'), ('M15', mfc_m15, 'cleaned'),
                                   ('M30', mfc_m30, 'cleaned'), ('H1', mfc_h1, 'cleaned'),
                                   ('H4', mfc_h4, 'cleaned')]:
        df = pd.read_csv(DATA_DIR / folder / f'mfc_currency_{cur}_{tf}_clean.csv')
        df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y.%m.%d %H:%M')
        df = df.set_index('datetime')
        df = df[(df.index >= START_DATE) & (df.index <= end_date)]
        mfc_dict[cur] = df['MFC']

log("  MFC data loaded")

# LSTM predictions
lstm_predictions = {}
for ccy in CURRENCIES:
    model = keras.models.load_model(MODEL_DIR / f'lstm_{ccy}_final.keras')
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
    keras.backend.clear_session()
    gc.collect()

log("  LSTM predictions loaded")

# Price data
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
    except:
        pass

log("  Price data loaded")

# Strategy parameters
MIN_CONF = 0.70
MFC_EXTREME = 0.5
RSI_LOW = 20
RSI_HIGH = 80
H1_VEL_THRESHOLD = 0.04
QUOTE_EXTENDED = 0.7

def run_backtest(max_hold_bars):
    """Run backtest with specific timeout value."""
    all_trades = []

    for pair, base, quote in ALL_PAIRS:
        if pair not in price_data:
            continue
        if base not in lstm_predictions or quote not in lstm_predictions:
            continue

        pip_val = get_pip_value(pair)

        try:
            price_df = price_data[pair].copy()

            price_df['base_mfc'] = mfc_m5[base].shift(1).reindex(price_df.index, method='ffill')
            price_df['quote_mfc'] = mfc_m5[quote].shift(1).reindex(price_df.index, method='ffill')

            base_h1 = mfc_h1[base].shift(1).reindex(price_df.index, method='ffill')
            quote_h1_vel = mfc_h1[quote].shift(1).reindex(price_df.index, method='ffill')
            price_df['base_vel_h1'] = base_h1.diff(periods=12)
            price_df['quote_vel_h1'] = quote_h1_vel.diff(periods=12)

            base_lstm = lstm_predictions[base].reindex(price_df.index, method='ffill')
            quote_lstm = lstm_predictions[quote].reindex(price_df.index, method='ffill')
            price_df['base_dir'] = base_lstm['direction']
            price_df['base_conf'] = base_lstm['confidence']
            price_df['quote_dir'] = quote_lstm['direction']
            price_df['quote_conf'] = quote_lstm['confidence']

            price_df = price_df.dropna()

            is_friday_afternoon = (price_df.index.dayofweek == 4) & (price_df.index.hour >= 12)

            price_df['base_h1_mfc'] = base_h1.reindex(price_df.index, method='ffill')
            price_df['quote_h1_mfc'] = quote_h1_vel.reindex(price_df.index, method='ffill')

            buy_extended_ok = (price_df['base_h1_mfc'] < QUOTE_EXTENDED) | (price_df['quote_h1_mfc'] >= QUOTE_EXTENDED)
            sell_extended_ok = (price_df['quote_h1_mfc'] < QUOTE_EXTENDED) | (price_df['base_h1_mfc'] >= QUOTE_EXTENDED)

            buy_vel_ok = (price_df['base_vel_h1'] - price_df['quote_vel_h1']) > 0
            sell_vel_ok = (price_df['quote_vel_h1'] - price_df['base_vel_h1']) > 0

            buy_signal = (
                (price_df['base_dir'] == 2) & (price_df['quote_dir'] == 0) &
                (price_df['base_conf'] >= MIN_CONF) & (price_df['quote_conf'] >= MIN_CONF) &
                (price_df['base_mfc'] <= -MFC_EXTREME) & (price_df['rsi'] < RSI_LOW) &
                (price_df['base_vel_h1'] >= -H1_VEL_THRESHOLD) &
                (~is_friday_afternoon) & (buy_extended_ok) & (buy_vel_ok)
            )

            sell_signal = (
                (price_df['base_dir'] == 0) & (price_df['quote_dir'] == 2) &
                (price_df['base_conf'] >= MIN_CONF) & (price_df['quote_conf'] >= MIN_CONF) &
                (price_df['base_mfc'] >= MFC_EXTREME) & (price_df['rsi'] > RSI_HIGH) &
                (price_df['base_vel_h1'] <= H1_VEL_THRESHOLD) &
                (~is_friday_afternoon) & (sell_extended_ok) & (sell_vel_ok)
            )

            # Process BUY
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
                entry_price = price_df.iloc[entry_idx]['Open']
                future_df = price_df.iloc[entry_idx+1:entry_idx+1+max_hold_bars]

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

                all_trades.append({
                    'pair': pair, 'type': 'BUY', 'entry_time': entry_time,
                    'exit_time': exit_time, 'pips': pips, 'win': 1 if pips > 0 else 0,
                    'spread': SPREADS.get(pair, 2.0), 'exit_reason': exit_reason,
                })

                while i < len(buy_indices) and buy_indices[i] <= exit_time:
                    i += 1

            # Process SELL
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
                entry_price = price_df.iloc[entry_idx]['Open']
                future_df = price_df.iloc[entry_idx+1:entry_idx+1+max_hold_bars]

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

                all_trades.append({
                    'pair': pair, 'type': 'SELL', 'entry_time': entry_time,
                    'exit_time': exit_time, 'pips': pips, 'win': 1 if pips > 0 else 0,
                    'spread': SPREADS.get(pair, 2.0), 'exit_reason': exit_reason,
                })

                while i < len(sell_indices) and sell_indices[i] <= exit_time:
                    i += 1

        except Exception as e:
            pass

    return pd.DataFrame(all_trades)

# Test different timeout values
log("\n" + "=" * 70)
log("2. TESTING DIFFERENT TIMEOUT VALUES")
log("=" * 70)

results = []

for timeout_bars in [75, 100, 125, 150, 175, 200, 250, 300]:
    log(f"\n  Testing {timeout_bars} bars ({timeout_bars*5/60:.1f}h)...")
    trades_df = run_backtest(timeout_bars)

    if len(trades_df) > 0:
        trades_df['net_pips'] = trades_df['pips'] - trades_df['spread']

        rsi_trades = trades_df[trades_df['exit_reason'] == 'RSI']
        timeout_trades = trades_df[trades_df['exit_reason'] == 'TIMEOUT']

        results.append({
            'timeout_bars': timeout_bars,
            'timeout_hours': timeout_bars * 5 / 60,
            'trades': len(trades_df),
            'rsi_exits': len(rsi_trades),
            'timeouts': len(timeout_trades),
            'timeout_pct': len(timeout_trades) / len(trades_df) * 100,
            'wr': trades_df['win'].mean() * 100,
            'rsi_wr': rsi_trades['win'].mean() * 100 if len(rsi_trades) > 0 else 0,
            'net_avg': trades_df['net_pips'].mean(),
            'total': trades_df['net_pips'].sum(),
            'timeout_net': timeout_trades['net_pips'].mean() if len(timeout_trades) > 0 else 0,
        })

# Display results
log("\n" + "=" * 70)
log("3. RESULTS COMPARISON")
log("=" * 70)

log(f"\n{'Timeout':<10} {'Hours':<8} {'Trades':>8} {'TO%':>8} {'WR':>8} {'RSI WR':>8} {'Net Avg':>10} {'Total':>10} {'TO Net':>10}")
log("-" * 95)

for r in results:
    log(f"{r['timeout_bars']:<10} {r['timeout_hours']:<8.1f} {r['trades']:>8} {r['timeout_pct']:>7.1f}% {r['wr']:>7.1f}% {r['rsi_wr']:>7.1f}% {r['net_avg']:>+9.2f} {r['total']:>+9.0f} {r['timeout_net']:>+9.2f}")

# Find best
best = max(results, key=lambda x: x['total'])
log(f"\n*** BEST TOTAL: {best['timeout_bars']} bars ({best['timeout_hours']:.1f}h) with {best['total']:.0f} pips ***")

best_avg = max(results, key=lambda x: x['net_avg'])
log(f"*** BEST AVG: {best_avg['timeout_bars']} bars ({best_avg['timeout_hours']:.1f}h) with {best_avg['net_avg']:.2f} net avg ***")

log(f"\nAnalysis complete!")
