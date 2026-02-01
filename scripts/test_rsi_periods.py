"""
Test Stochastic 25 with H4 filter across multiple years
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

DATA_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/data')
LSTM_DATA_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/lstm_model/data')
MODEL_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/lstm_model/models')

with open(LSTM_DATA_DIR / 'config.pkl', 'rb') as f:
    config = pickle.load(f)

LOOKBACK = config['lookback']
CURRENCIES = config['currencies']

# Will be set per year test
START_DATE = None
END_DATE = None

YEAR_RANGES = {
    '2021': ('2021-01-01', '2021-12-31'),
    '2022': ('2022-01-01', '2022-12-31'),
    '2025': ('2025-01-01', '2025-12-21'),
}

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

def calculate_stochastic(high, low, close, period=14):
    """Calculate Stochastic %K"""
    lowest_low = low.rolling(window=period).min()
    highest_high = high.rolling(window=period).max()
    stoch_k = ((close - lowest_low) / (highest_high - lowest_low)) * 100
    return stoch_k

def get_pip_value(pair):
    return 0.01 if 'JPY' in pair else 0.0001

print("=" * 70)
print("STOCHASTIC 25 + H4 FILTER - MULTI-YEAR TEST")
print("=" * 70)
print(f"Started: {datetime.now()}")

# Load ALL MFC data (no date filter)
print("\nLoading MFC data (all dates)...")
mfc_m5_all = {}
mfc_h1_all = {}
mfc_h4_all = {}

for cur in CURRENCIES:
    df = pd.read_csv(DATA_DIR / 'cleaned' / f'mfc_currency_{cur}_M5_clean.csv')
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y.%m.%d %H:%M')
    df = df.set_index('datetime')
    mfc_m5_all[cur] = df['MFC']

    df = pd.read_csv(DATA_DIR / 'cleaned' / f'mfc_currency_{cur}_H1_clean.csv')
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y.%m.%d %H:%M')
    df = df.set_index('datetime')
    mfc_h1_all[cur] = df['MFC']

    df = pd.read_csv(DATA_DIR / 'cleaned' / f'mfc_currency_{cur}_H4_clean.csv')
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y.%m.%d %H:%M')
    df = df.set_index('datetime')
    mfc_h4_all[cur] = df['MFC']

# Load LSTM predictions (all dates)
print("Loading LSTM predictions (all dates)...")
lstm_predictions_all = {}

for ccy in CURRENCIES:
    m5_idx = mfc_m5_all[ccy].index
    m5_shifted = mfc_m5_all[ccy].shift(1)

    df_m15 = pd.read_csv(DATA_DIR / 'cleaned' / f'mfc_currency_{ccy}_M15_clean.csv')
    df_m15['datetime'] = pd.to_datetime(df_m15['Date'] + ' ' + df_m15['Time'], format='%Y.%m.%d %H:%M')
    df_m15 = df_m15.set_index('datetime')
    m15_shifted = df_m15['MFC'].shift(1).reindex(m5_idx, method='ffill')

    df_m30 = pd.read_csv(DATA_DIR / 'cleaned' / f'mfc_currency_{ccy}_M30_clean.csv')
    df_m30['datetime'] = pd.to_datetime(df_m30['Date'] + ' ' + df_m30['Time'], format='%Y.%m.%d %H:%M')
    df_m30 = df_m30.set_index('datetime')
    m30_shifted = df_m30['MFC'].shift(1).reindex(m5_idx, method='ffill')

    h1_shifted = mfc_h1_all[ccy].shift(1).reindex(m5_idx, method='ffill')
    h4_shifted = mfc_h4_all[ccy].shift(1).reindex(m5_idx, method='ffill')

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

    model = tf.keras.models.load_model(MODEL_DIR / f'lstm_{ccy}_final.keras')

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
    datetimes = mfc_m5_all[ccy].index[valid_start:valid_start+n_samples]

    lstm_predictions_all[ccy] = pd.DataFrame({
        'direction': np.argmax(pred[0], axis=1),
        'confidence': np.max(pred[0], axis=1)
    }, index=datetimes)

    del model, X_val, pred
    tf.keras.backend.clear_session()
    gc.collect()

# Load ALL price data (no date filter)
print("Loading price data (all dates)...")
price_data_all = {}

for pair, base, quote in ALL_PAIRS:
    try:
        chunks = []
        for chunk in pd.read_csv(DATA_DIR / f'{pair}_GMT+0_US-DST_M1.csv', chunksize=500000):
            chunk['datetime'] = pd.to_datetime(chunk['Date'] + ' ' + chunk['Time'], format='%Y.%m.%d %H:%M:%S')
            chunk = chunk.set_index('datetime')
            if len(chunk) > 0:
                m5_chunk = chunk[['Open', 'High', 'Low', 'Close']].resample('5min').agg({
                    'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
                }).dropna()
                chunks.append(m5_chunk)

        if chunks:
            price_m5 = pd.concat(chunks)
            price_m5 = price_m5[~price_m5.index.duplicated(keep='first')]
            price_data_all[pair] = price_m5
    except Exception as e:
        pass

print(f"Loaded {len(price_data_all)} pairs")

# Config
STOCH_PERIOD = 25
MIN_CONF = 0.70
MFC_EXTREME = 0.5
STOCH_LOW = 20
STOCH_HIGH = 80
H1_VEL_THRESHOLD = 0.04
MAX_HOLD_BARS = 250

def test_year(year_name, start_date, end_date):
    """Test Stochastic 25 + H4 filter for a specific year"""
    print(f"\n{'='*70}")
    print(f"TESTING {year_name}")
    print(f"{'='*70}")

    all_trades = []

    for pair, base, quote in ALL_PAIRS:
        if pair not in price_data_all:
            continue
        if base not in lstm_predictions_all or quote not in lstm_predictions_all:
            continue

        pip_val = get_pip_value(pair)

        try:
            # Filter price data for this year
            price_df = price_data_all[pair].copy()
            price_df = price_df[(price_df.index >= start_date) & (price_df.index <= end_date)]
            if len(price_df) < 100:
                continue

            price_df['stoch'] = calculate_stochastic(price_df['High'], price_df['Low'], price_df['Close'], period=STOCH_PERIOD)

            price_df['base_mfc'] = mfc_m5_all[base].shift(1).reindex(price_df.index, method='ffill')
            price_df['quote_mfc'] = mfc_m5_all[quote].shift(1).reindex(price_df.index, method='ffill')

            # H1 velocity
            base_h1 = mfc_h1_all[base].shift(1).reindex(price_df.index, method='ffill')
            quote_h1 = mfc_h1_all[quote].shift(1).reindex(price_df.index, method='ffill')
            price_df['base_vel_h1'] = base_h1.diff(periods=12)
            price_df['quote_vel_h1'] = quote_h1.diff(periods=12)

            # H4 velocity (72 M5 bars = 6 H4 bars = 24 hours)
            base_h4 = mfc_h4_all[base].shift(1).reindex(price_df.index, method='ffill')
            quote_h4 = mfc_h4_all[quote].shift(1).reindex(price_df.index, method='ffill')
            price_df['base_vel_h4'] = base_h4.diff(periods=72)
            price_df['quote_vel_h4'] = quote_h4.diff(periods=72)

            # LSTM predictions
            base_lstm = lstm_predictions_all[base].reindex(price_df.index, method='ffill')
            quote_lstm = lstm_predictions_all[quote].reindex(price_df.index, method='ffill')

            price_df['base_dir'] = base_lstm['direction']
            price_df['base_conf'] = base_lstm['confidence']
            price_df['quote_dir'] = quote_lstm['direction']
            price_df['quote_conf'] = quote_lstm['confidence']

            price_df = price_df.dropna()

            is_friday_afternoon = (price_df.index.dayofweek == 4) & (price_df.index.hour >= 6)

            buy_vel_ok = (price_df['base_vel_h1'] - price_df['quote_vel_h1']) > 0
            sell_vel_ok = (price_df['quote_vel_h1'] - price_df['base_vel_h1']) > 0

            # H4 filter: For BUY, quote H4 vel should be < 0 (weakening)
            # For SELL, base H4 vel should be < 0 (weakening)
            buy_h4_ok = price_df['quote_vel_h4'] < 0
            sell_h4_ok = price_df['base_vel_h4'] < 0

            buy_signal = (
                (price_df['base_dir'] == 2) &
                (price_df['quote_dir'] == 0) &
                (price_df['base_conf'] >= MIN_CONF) &
                (price_df['quote_conf'] >= MIN_CONF) &
                (price_df['base_mfc'] <= -MFC_EXTREME) &
                (price_df['stoch'] < STOCH_LOW) &
                (price_df['base_vel_h1'] >= -H1_VEL_THRESHOLD) &
                (~is_friday_afternoon) &
                (buy_vel_ok) &
                (buy_h4_ok)  # H4 filter
            )

            sell_signal = (
                (price_df['base_dir'] == 0) &
                (price_df['quote_dir'] == 2) &
                (price_df['base_conf'] >= MIN_CONF) &
                (price_df['quote_conf'] >= MIN_CONF) &
                (price_df['base_mfc'] >= MFC_EXTREME) &
                (price_df['stoch'] > STOCH_HIGH) &
                (price_df['base_vel_h1'] <= H1_VEL_THRESHOLD) &
                (~is_friday_afternoon) &
                (sell_vel_ok) &
                (sell_h4_ok)  # H4 filter
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
                future_df = price_df.iloc[entry_idx+1:entry_idx+1+MAX_HOLD_BARS]

                if len(future_df) == 0:
                    i += 1
                    continue

                exit_mask = future_df['stoch'] >= STOCH_HIGH

                if exit_mask.any():
                    exit_time = future_df.index[exit_mask.argmax()]
                    exit_price = price_df.loc[exit_time, 'Close']
                    pips = (exit_price - entry_price) / pip_val
                    exit_reason = 'STOCH'
                else:
                    exit_time = future_df.index[-1]
                    exit_price = future_df.iloc[-1]['Close']
                    pips = (exit_price - entry_price) / pip_val
                    exit_reason = 'TIMEOUT'

                all_trades.append({
                    'pair': pair, 'type': 'BUY', 'entry_time': entry_time,
                    'pips': pips, 'win': 1 if pips > 0 else 0,
                    'spread': SPREADS.get(pair, 2.0),
                    'exit_reason': exit_reason,
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
                future_df = price_df.iloc[entry_idx+1:entry_idx+1+MAX_HOLD_BARS]

                if len(future_df) == 0:
                    i += 1
                    continue

                exit_mask = future_df['stoch'] <= STOCH_LOW

                if exit_mask.any():
                    exit_time = future_df.index[exit_mask.argmax()]
                    exit_price = price_df.loc[exit_time, 'Close']
                    pips = (entry_price - exit_price) / pip_val
                    exit_reason = 'STOCH'
                else:
                    exit_time = future_df.index[-1]
                    exit_price = future_df.iloc[-1]['Close']
                    pips = (entry_price - exit_price) / pip_val
                    exit_reason = 'TIMEOUT'

                all_trades.append({
                    'pair': pair, 'type': 'SELL', 'entry_time': entry_time,
                    'pips': pips, 'win': 1 if pips > 0 else 0,
                    'spread': SPREADS.get(pair, 2.0),
                    'exit_reason': exit_reason,
                })

                while i < len(sell_indices) and sell_indices[i] <= exit_time:
                    i += 1

        except Exception as e:
            pass

    if all_trades:
        trades_df = pd.DataFrame(all_trades)
        trades_df['net_pips'] = trades_df['pips'] - trades_df['spread']

        stoch_exits = trades_df[trades_df['exit_reason'] == 'STOCH']
        timeouts = trades_df[trades_df['exit_reason'] == 'TIMEOUT']

        print(f"\n  Trades: {len(trades_df)}")
        print(f"  Win Rate: {trades_df['win'].mean()*100:.1f}%")
        print(f"  Avg Pips: {trades_df['net_pips'].mean():+.1f}")
        print(f"  Total: {trades_df['net_pips'].sum():+.0f}")
        print(f"  Stoch exits: {len(stoch_exits)}, Timeouts: {len(timeouts)}")

        return {
            'year': year_name,
            'trades': len(trades_df),
            'wr': trades_df['win'].mean() * 100,
            'avg': trades_df['net_pips'].mean(),
            'total': trades_df['net_pips'].sum(),
            'timeouts': len(timeouts),
        }
    return None

# Run tests for each year
results = []
for year_name, (start_date, end_date) in YEAR_RANGES.items():
    result = test_year(year_name, start_date, end_date)
    if result:
        results.append(result)

# Summary table
print("\n" + "=" * 70)
print("STOCHASTIC 25 + H4 FILTER - MULTI-YEAR SUMMARY")
print("=" * 70)

print(f"\n{'Year':<6} {'Trades':>10} {'WR':>10} {'Avg':>12} {'Total':>12} {'Timeouts':>10}")
print("-" * 65)

total_trades = 0
total_pips = 0
total_timeouts = 0

for r in results:
    print(f"{r['year']:<6} {r['trades']:>10} {r['wr']:>9.1f}% {r['avg']:>+11.1f} {r['total']:>+12.0f} {r['timeouts']:>10}")
    total_trades += r['trades']
    total_pips += r['total']
    total_timeouts += r['timeouts']

print("-" * 65)
avg_wr = sum(r['wr'] * r['trades'] for r in results) / total_trades if total_trades > 0 else 0
print(f"{'TOTAL':<6} {total_trades:>10} {avg_wr:>9.1f}% {total_pips/total_trades:>+11.1f} {total_pips:>+12.0f} {total_timeouts:>10}")

print(f"\nCompleted: {datetime.now()}")
