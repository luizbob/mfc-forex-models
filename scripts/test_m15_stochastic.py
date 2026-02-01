"""
Test M15 Stochastic for entry and exit signals
Compare with M5 Stochastic baseline
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

TEST_YEAR = ('2025-01-01', '2025-12-21')

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

def get_pip_value(pair):
    return 0.01 if 'JPY' in pair else 0.0001

def calculate_stochastic(high, low, close, period=25):
    lowest_low = low.rolling(window=period).min()
    highest_high = high.rolling(window=period).max()
    stoch_k = ((close - lowest_low) / (highest_high - lowest_low)) * 100
    return stoch_k

print("=" * 70)
print("M15 STOCHASTIC TEST")
print("=" * 70)
print(f"Started: {datetime.now()}")

# Load MFC data
print("\nLoading MFC data...")
mfc_m5_all = {}
mfc_h1_all = {}

for cur in CURRENCIES:
    df = pd.read_csv(DATA_DIR / 'cleaned' / f'mfc_currency_{cur}_M5_clean.csv')
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y.%m.%d %H:%M')
    df = df.set_index('datetime')
    mfc_m5_all[cur] = df['MFC']

    df = pd.read_csv(DATA_DIR / 'cleaned' / f'mfc_currency_{cur}_H1_clean.csv')
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y.%m.%d %H:%M')
    df = df.set_index('datetime')
    mfc_h1_all[cur] = df['MFC']

# Load LSTM predictions
print("Loading LSTM predictions...")
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

    df_h1 = pd.read_csv(DATA_DIR / 'cleaned' / f'mfc_currency_{ccy}_H1_clean.csv')
    df_h1['datetime'] = pd.to_datetime(df_h1['Date'] + ' ' + df_h1['Time'], format='%Y.%m.%d %H:%M')
    df_h1 = df_h1.set_index('datetime')
    h1_shifted = df_h1['MFC'].shift(1).reindex(m5_idx, method='ffill')

    df_h4 = pd.read_csv(DATA_DIR / 'cleaned' / f'mfc_currency_{ccy}_H4_clean.csv')
    df_h4['datetime'] = pd.to_datetime(df_h4['Date'] + ' ' + df_h4['Time'], format='%Y.%m.%d %H:%M')
    df_h4 = df_h4.set_index('datetime')
    h4_shifted = df_h4['MFC'].shift(1).reindex(m5_idx, method='ffill')

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

# Load price data - both M5 and M15
print("Loading price data...")
price_m5_all = {}
price_m15_all = {}

for pair, base, quote in ALL_PAIRS:
    try:
        chunks = []
        for chunk in pd.read_csv(DATA_DIR / f'{pair}_GMT+0_US-DST_M1.csv', chunksize=500000):
            chunk['datetime'] = pd.to_datetime(chunk['Date'] + ' ' + chunk['Time'], format='%Y.%m.%d %H:%M:%S')
            chunk = chunk.set_index('datetime')
            # Filter to test period
            chunk = chunk[(chunk.index >= TEST_YEAR[0]) & (chunk.index <= TEST_YEAR[1])]
            if len(chunk) > 0:
                # M5 data
                m5_chunk = chunk[['Open', 'High', 'Low', 'Close']].resample('5min').agg({
                    'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
                }).dropna()
                chunks.append(m5_chunk)

        if chunks:
            price_m5 = pd.concat(chunks)
            price_m5 = price_m5[~price_m5.index.duplicated(keep='first')]
            price_m5_all[pair] = price_m5

            # Create M15 from M5
            price_m15 = price_m5.resample('15min').agg({
                'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
            }).dropna()
            price_m15_all[pair] = price_m15

    except Exception as e:
        pass

print(f"Loaded {len(price_m5_all)} pairs")

# Strategy config
STOCH_PERIOD = 25
MIN_CONF = 0.70
MFC_EXTREME = 0.5
STOCH_LOW = 20
STOCH_HIGH = 80
H1_VEL_THRESHOLD = 0.04
MAX_HOLD_BARS_M5 = 250
MAX_HOLD_BARS_M15 = 84  # ~21 hours in M15 bars

def run_backtest(use_m15_stoch=False, stoch_period=25):
    """Run backtest with M5 or M15 stochastic"""
    all_trades = []

    for pair, base, quote in ALL_PAIRS:
        if pair not in price_m5_all:
            continue
        if base not in lstm_predictions_all or quote not in lstm_predictions_all:
            continue

        pip_val = get_pip_value(pair)

        try:
            # Always use M5 for entry timing and price
            price_m5 = price_m5_all[pair].copy()
            price_m15 = price_m15_all[pair].copy()

            if len(price_m5) < 100:
                continue

            # Calculate stochastic on chosen timeframe
            if use_m15_stoch:
                stoch_m15 = calculate_stochastic(price_m15['High'], price_m15['Low'], price_m15['Close'], period=stoch_period)
                # Forward fill M15 stoch to M5 timeframe
                price_m5['stoch'] = stoch_m15.reindex(price_m5.index, method='ffill')
                max_hold = MAX_HOLD_BARS_M5
            else:
                price_m5['stoch'] = calculate_stochastic(price_m5['High'], price_m5['Low'], price_m5['Close'], period=stoch_period)
                max_hold = MAX_HOLD_BARS_M5

            # MFC data
            price_m5['base_mfc'] = mfc_m5_all[base].shift(1).reindex(price_m5.index, method='ffill')
            price_m5['quote_mfc'] = mfc_m5_all[quote].shift(1).reindex(price_m5.index, method='ffill')

            # H1 velocity
            base_h1 = mfc_h1_all[base].shift(1).reindex(price_m5.index, method='ffill')
            quote_h1 = mfc_h1_all[quote].shift(1).reindex(price_m5.index, method='ffill')
            price_m5['base_vel_h1'] = base_h1.diff(periods=12)
            price_m5['quote_vel_h1'] = quote_h1.diff(periods=12)

            # LSTM predictions
            base_lstm = lstm_predictions_all[base].reindex(price_m5.index, method='ffill')
            quote_lstm = lstm_predictions_all[quote].reindex(price_m5.index, method='ffill')

            price_m5['base_dir'] = base_lstm['direction']
            price_m5['base_conf'] = base_lstm['confidence']
            price_m5['quote_dir'] = quote_lstm['direction']
            price_m5['quote_conf'] = quote_lstm['confidence']

            price_m5 = price_m5.dropna()

            is_friday_afternoon = (price_m5.index.dayofweek == 4) & (price_m5.index.hour >= 6)

            buy_vel_ok = (price_m5['base_vel_h1'] - price_m5['quote_vel_h1']) > 0
            sell_vel_ok = (price_m5['quote_vel_h1'] - price_m5['base_vel_h1']) > 0

            # Entry signals
            buy_signal = (
                (price_m5['base_dir'] == 2) &
                (price_m5['quote_dir'] == 0) &
                (price_m5['base_conf'] >= MIN_CONF) &
                (price_m5['quote_conf'] >= MIN_CONF) &
                (price_m5['base_mfc'] <= -MFC_EXTREME) &
                (price_m5['stoch'] < STOCH_LOW) &
                (price_m5['base_vel_h1'] >= -H1_VEL_THRESHOLD) &
                (~is_friday_afternoon) &
                (buy_vel_ok)
            )

            sell_signal = (
                (price_m5['base_dir'] == 0) &
                (price_m5['quote_dir'] == 2) &
                (price_m5['base_conf'] >= MIN_CONF) &
                (price_m5['quote_conf'] >= MIN_CONF) &
                (price_m5['base_mfc'] >= MFC_EXTREME) &
                (price_m5['stoch'] > STOCH_HIGH) &
                (price_m5['base_vel_h1'] <= H1_VEL_THRESHOLD) &
                (~is_friday_afternoon) &
                (sell_vel_ok)
            )

            # Process BUY trades
            buy_indices = price_m5.index[buy_signal].tolist()
            i = 0
            while i < len(buy_indices):
                signal_time = buy_indices[i]
                signal_idx = price_m5.index.get_loc(signal_time)

                entry_idx = signal_idx + 1
                if entry_idx >= len(price_m5):
                    i += 1
                    continue

                entry_time = price_m5.index[entry_idx]
                entry_price = price_m5.iloc[entry_idx]['Open']
                future_df = price_m5.iloc[entry_idx+1:entry_idx+1+max_hold]

                if len(future_df) == 0:
                    i += 1
                    continue

                # Exit when stoch crosses high threshold
                exit_mask = future_df['stoch'] >= STOCH_HIGH

                if exit_mask.any():
                    exit_time = future_df.index[exit_mask.argmax()]
                    exit_price = price_m5.loc[exit_time, 'Close']
                    pips = (exit_price - entry_price) / pip_val
                    exit_reason = 'STOCH'
                    bars_held = (price_m5.index.get_loc(exit_time) - entry_idx)
                else:
                    exit_time = future_df.index[-1]
                    exit_price = future_df.iloc[-1]['Close']
                    pips = (exit_price - entry_price) / pip_val
                    exit_reason = 'TIMEOUT'
                    bars_held = len(future_df)

                net_pips = pips - SPREADS.get(pair, 2.0)

                all_trades.append({
                    'pair': pair,
                    'type': 'BUY',
                    'entry_time': entry_time,
                    'pips': pips,
                    'net_pips': net_pips,
                    'win': 1 if net_pips > 0 else 0,
                    'exit_reason': exit_reason,
                    'bars_held': bars_held,
                })

                while i < len(buy_indices) and buy_indices[i] <= exit_time:
                    i += 1

            # Process SELL trades
            sell_indices = price_m5.index[sell_signal].tolist()
            i = 0
            while i < len(sell_indices):
                signal_time = sell_indices[i]
                signal_idx = price_m5.index.get_loc(signal_time)

                entry_idx = signal_idx + 1
                if entry_idx >= len(price_m5):
                    i += 1
                    continue

                entry_time = price_m5.index[entry_idx]
                entry_price = price_m5.iloc[entry_idx]['Open']
                future_df = price_m5.iloc[entry_idx+1:entry_idx+1+max_hold]

                if len(future_df) == 0:
                    i += 1
                    continue

                exit_mask = future_df['stoch'] <= STOCH_LOW

                if exit_mask.any():
                    exit_time = future_df.index[exit_mask.argmax()]
                    exit_price = price_m5.loc[exit_time, 'Close']
                    pips = (entry_price - exit_price) / pip_val
                    exit_reason = 'STOCH'
                    bars_held = (price_m5.index.get_loc(exit_time) - entry_idx)
                else:
                    exit_time = future_df.index[-1]
                    exit_price = future_df.iloc[-1]['Close']
                    pips = (entry_price - exit_price) / pip_val
                    exit_reason = 'TIMEOUT'
                    bars_held = len(future_df)

                net_pips = pips - SPREADS.get(pair, 2.0)

                all_trades.append({
                    'pair': pair,
                    'type': 'SELL',
                    'entry_time': entry_time,
                    'pips': pips,
                    'net_pips': net_pips,
                    'win': 1 if net_pips > 0 else 0,
                    'exit_reason': exit_reason,
                    'bars_held': bars_held,
                })

                while i < len(sell_indices) and sell_indices[i] <= exit_time:
                    i += 1

        except Exception as e:
            pass

    return pd.DataFrame(all_trades)

# Run tests
print("\n" + "=" * 70)
print("RUNNING BACKTESTS")
print("=" * 70)

results = {}

# M5 Stochastic (baseline)
print("\nTesting M5 Stochastic (baseline)...")
df_m5 = run_backtest(use_m15_stoch=False, stoch_period=25)
results['M5 Stoch 25'] = df_m5

# M15 Stochastic with different periods
for period in [15, 20, 25, 30]:
    print(f"Testing M15 Stochastic period {period}...")
    df = run_backtest(use_m15_stoch=True, stoch_period=period)
    results[f'M15 Stoch {period}'] = df

# Print results
print("\n" + "=" * 70)
print("RESULTS COMPARISON")
print("=" * 70)

print(f"\n{'Strategy':<20} {'Trades':>7} {'WR':>7} {'Avg':>8} {'Total':>10} {'Timeouts':>10}")
print("-" * 70)

for name, df in results.items():
    if len(df) > 0:
        trades = len(df)
        wr = df['win'].mean() * 100
        avg = df['net_pips'].mean()
        total = df['net_pips'].sum()
        timeouts = (df['exit_reason'] == 'TIMEOUT').sum()
        to_pct = timeouts / trades * 100
        print(f"{name:<20} {trades:>7} {wr:>6.1f}% {avg:>+7.1f} {total:>+10.0f} {timeouts:>6} ({to_pct:.1f}%)")

# Detailed M15 analysis
print("\n" + "=" * 70)
print("M15 STOCHASTIC 25 - DETAILED ANALYSIS")
print("=" * 70)

df_m15 = results.get('M15 Stoch 25')
if df_m15 is not None and len(df_m15) > 0:
    print(f"\nTrades: {len(df_m15)}")
    print(f"Win Rate: {df_m15['win'].mean()*100:.1f}%")
    print(f"Avg Pips: {df_m15['net_pips'].mean():+.1f}")
    print(f"Total Pips: {df_m15['net_pips'].sum():+.0f}")

    # By exit reason
    print("\nBy Exit Reason:")
    for reason in df_m15['exit_reason'].unique():
        subset = df_m15[df_m15['exit_reason'] == reason]
        print(f"  {reason}: {len(subset)} trades, {subset['win'].mean()*100:.1f}% WR, {subset['net_pips'].sum():+.0f} pips")

    # Average hold time
    avg_bars = df_m15['bars_held'].mean()
    print(f"\nAvg hold time: {avg_bars:.0f} M5 bars ({avg_bars*5/60:.1f} hours)")

    # Compare with M5
    df_m5 = results.get('M5 Stoch 25')
    if df_m5 is not None:
        print("\n--- Comparison with M5 Stoch 25 ---")
        print(f"M5:  {len(df_m5)} trades, {df_m5['net_pips'].sum():+.0f} pips, {df_m5['net_pips'].mean():+.1f} avg")
        print(f"M15: {len(df_m15)} trades, {df_m15['net_pips'].sum():+.0f} pips, {df_m15['net_pips'].mean():+.1f} avg")
        print(f"Difference: {df_m15['net_pips'].sum() - df_m5['net_pips'].sum():+.0f} pips")

print(f"\nCompleted: {datetime.now()}")
