"""
Test CCI and Bollinger Bands as alternative indicators
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

# Indicator calculations
def calculate_stochastic(high, low, close, period=25):
    lowest_low = low.rolling(window=period).min()
    highest_high = high.rolling(window=period).max()
    stoch_k = ((close - lowest_low) / (highest_high - lowest_low)) * 100
    return stoch_k

def calculate_cci(high, low, close, period=20):
    """CCI = (Typical Price - SMA) / (0.015 * Mean Deviation)"""
    tp = (high + low + close) / 3
    sma = tp.rolling(window=period).mean()
    mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    cci = (tp - sma) / (0.015 * mad)
    return cci

def calculate_bollinger(close, period=20, std_dev=2):
    """Returns lower_band, middle_band, upper_band, and %B"""
    middle = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    upper = middle + (std_dev * std)
    lower = middle - (std_dev * std)
    pct_b = (close - lower) / (upper - lower)
    return lower, middle, upper, pct_b

print("=" * 70)
print("CCI AND BOLLINGER BANDS TEST")
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

# Load price data
print("Loading price data...")
price_m5_all = {}

for pair, base, quote in ALL_PAIRS:
    try:
        chunks = []
        for chunk in pd.read_csv(DATA_DIR / f'{pair}_GMT+0_US-DST_M1.csv', chunksize=500000):
            chunk['datetime'] = pd.to_datetime(chunk['Date'] + ' ' + chunk['Time'], format='%Y.%m.%d %H:%M:%S')
            chunk = chunk.set_index('datetime')
            chunk = chunk[(chunk.index >= TEST_YEAR[0]) & (chunk.index <= TEST_YEAR[1])]
            if len(chunk) > 0:
                m5_chunk = chunk[['Open', 'High', 'Low', 'Close']].resample('5min').agg({
                    'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
                }).dropna()
                chunks.append(m5_chunk)

        if chunks:
            price_m5 = pd.concat(chunks)
            price_m5 = price_m5[~price_m5.index.duplicated(keep='first')]
            price_m5_all[pair] = price_m5

    except Exception as e:
        pass

print(f"Loaded {len(price_m5_all)} pairs")

# Strategy config
MIN_CONF = 0.70
MFC_EXTREME = 0.5
H1_VEL_THRESHOLD = 0.04
MAX_HOLD_BARS = 250

def run_backtest_stochastic(period=25, entry_low=20, entry_high=80):
    """Run backtest with Stochastic"""
    all_trades = []

    for pair, base, quote in ALL_PAIRS:
        if pair not in price_m5_all:
            continue
        if base not in lstm_predictions_all or quote not in lstm_predictions_all:
            continue

        pip_val = get_pip_value(pair)

        try:
            price_df = price_m5_all[pair].copy()
            if len(price_df) < 100:
                continue

            price_df['indicator'] = calculate_stochastic(price_df['High'], price_df['Low'], price_df['Close'], period)

            # MFC and LSTM data
            price_df['base_mfc'] = mfc_m5_all[base].shift(1).reindex(price_df.index, method='ffill')
            price_df['quote_mfc'] = mfc_m5_all[quote].shift(1).reindex(price_df.index, method='ffill')

            base_h1 = mfc_h1_all[base].shift(1).reindex(price_df.index, method='ffill')
            quote_h1 = mfc_h1_all[quote].shift(1).reindex(price_df.index, method='ffill')
            price_df['base_vel_h1'] = base_h1.diff(periods=12)
            price_df['quote_vel_h1'] = quote_h1.diff(periods=12)

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

            buy_signal = (
                (price_df['base_dir'] == 2) & (price_df['quote_dir'] == 0) &
                (price_df['base_conf'] >= MIN_CONF) & (price_df['quote_conf'] >= MIN_CONF) &
                (price_df['base_mfc'] <= -MFC_EXTREME) &
                (price_df['indicator'] < entry_low) &
                (price_df['base_vel_h1'] >= -H1_VEL_THRESHOLD) &
                (~is_friday_afternoon) & buy_vel_ok
            )

            sell_signal = (
                (price_df['base_dir'] == 0) & (price_df['quote_dir'] == 2) &
                (price_df['base_conf'] >= MIN_CONF) & (price_df['quote_conf'] >= MIN_CONF) &
                (price_df['base_mfc'] >= MFC_EXTREME) &
                (price_df['indicator'] > entry_high) &
                (price_df['base_vel_h1'] <= H1_VEL_THRESHOLD) &
                (~is_friday_afternoon) & sell_vel_ok
            )

            # Process trades
            for signal_type, signals, exit_thresh, is_buy in [
                ('BUY', buy_signal, entry_high, True),
                ('SELL', sell_signal, entry_low, False)
            ]:
                indices = price_df.index[signals].tolist()
                i = 0
                while i < len(indices):
                    signal_time = indices[i]
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

                    if is_buy:
                        exit_mask = future_df['indicator'] >= exit_thresh
                    else:
                        exit_mask = future_df['indicator'] <= exit_thresh

                    if exit_mask.any():
                        exit_time = future_df.index[exit_mask.argmax()]
                        exit_price = price_df.loc[exit_time, 'Close']
                        exit_reason = 'INDICATOR'
                        bars_held = price_df.index.get_loc(exit_time) - entry_idx
                    else:
                        exit_time = future_df.index[-1]
                        exit_price = future_df.iloc[-1]['Close']
                        exit_reason = 'TIMEOUT'
                        bars_held = len(future_df)

                    if is_buy:
                        pips = (exit_price - entry_price) / pip_val
                    else:
                        pips = (entry_price - exit_price) / pip_val

                    net_pips = pips - SPREADS.get(pair, 2.0)

                    all_trades.append({
                        'pair': pair, 'type': signal_type, 'entry_time': entry_time,
                        'pips': pips, 'net_pips': net_pips, 'win': 1 if net_pips > 0 else 0,
                        'exit_reason': exit_reason, 'bars_held': bars_held,
                    })

                    while i < len(indices) and indices[i] <= exit_time:
                        i += 1

        except Exception as e:
            pass

    return pd.DataFrame(all_trades)


def run_backtest_cci(period=20, entry_level=100):
    """Run backtest with CCI"""
    all_trades = []

    for pair, base, quote in ALL_PAIRS:
        if pair not in price_m5_all:
            continue
        if base not in lstm_predictions_all or quote not in lstm_predictions_all:
            continue

        pip_val = get_pip_value(pair)

        try:
            price_df = price_m5_all[pair].copy()
            if len(price_df) < 100:
                continue

            price_df['indicator'] = calculate_cci(price_df['High'], price_df['Low'], price_df['Close'], period)

            price_df['base_mfc'] = mfc_m5_all[base].shift(1).reindex(price_df.index, method='ffill')
            price_df['quote_mfc'] = mfc_m5_all[quote].shift(1).reindex(price_df.index, method='ffill')

            base_h1 = mfc_h1_all[base].shift(1).reindex(price_df.index, method='ffill')
            quote_h1 = mfc_h1_all[quote].shift(1).reindex(price_df.index, method='ffill')
            price_df['base_vel_h1'] = base_h1.diff(periods=12)
            price_df['quote_vel_h1'] = quote_h1.diff(periods=12)

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

            # CCI: Buy when < -entry_level, exit when >= +entry_level
            buy_signal = (
                (price_df['base_dir'] == 2) & (price_df['quote_dir'] == 0) &
                (price_df['base_conf'] >= MIN_CONF) & (price_df['quote_conf'] >= MIN_CONF) &
                (price_df['base_mfc'] <= -MFC_EXTREME) &
                (price_df['indicator'] < -entry_level) &
                (price_df['base_vel_h1'] >= -H1_VEL_THRESHOLD) &
                (~is_friday_afternoon) & buy_vel_ok
            )

            sell_signal = (
                (price_df['base_dir'] == 0) & (price_df['quote_dir'] == 2) &
                (price_df['base_conf'] >= MIN_CONF) & (price_df['quote_conf'] >= MIN_CONF) &
                (price_df['base_mfc'] >= MFC_EXTREME) &
                (price_df['indicator'] > entry_level) &
                (price_df['base_vel_h1'] <= H1_VEL_THRESHOLD) &
                (~is_friday_afternoon) & sell_vel_ok
            )

            for signal_type, signals, is_buy in [('BUY', buy_signal, True), ('SELL', sell_signal, False)]:
                indices = price_df.index[signals].tolist()
                i = 0
                while i < len(indices):
                    signal_time = indices[i]
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

                    if is_buy:
                        exit_mask = future_df['indicator'] >= entry_level
                    else:
                        exit_mask = future_df['indicator'] <= -entry_level

                    if exit_mask.any():
                        exit_time = future_df.index[exit_mask.argmax()]
                        exit_price = price_df.loc[exit_time, 'Close']
                        exit_reason = 'INDICATOR'
                        bars_held = price_df.index.get_loc(exit_time) - entry_idx
                    else:
                        exit_time = future_df.index[-1]
                        exit_price = future_df.iloc[-1]['Close']
                        exit_reason = 'TIMEOUT'
                        bars_held = len(future_df)

                    if is_buy:
                        pips = (exit_price - entry_price) / pip_val
                    else:
                        pips = (entry_price - exit_price) / pip_val

                    net_pips = pips - SPREADS.get(pair, 2.0)

                    all_trades.append({
                        'pair': pair, 'type': signal_type, 'entry_time': entry_time,
                        'pips': pips, 'net_pips': net_pips, 'win': 1 if net_pips > 0 else 0,
                        'exit_reason': exit_reason, 'bars_held': bars_held,
                    })

                    while i < len(indices) and indices[i] <= exit_time:
                        i += 1

        except Exception as e:
            pass

    return pd.DataFrame(all_trades)


def run_backtest_bollinger(period=20, std_dev=2, exit_at='middle'):
    """Run backtest with Bollinger Bands"""
    all_trades = []

    for pair, base, quote in ALL_PAIRS:
        if pair not in price_m5_all:
            continue
        if base not in lstm_predictions_all or quote not in lstm_predictions_all:
            continue

        pip_val = get_pip_value(pair)

        try:
            price_df = price_m5_all[pair].copy()
            if len(price_df) < 100:
                continue

            lower, middle, upper, pct_b = calculate_bollinger(price_df['Close'], period, std_dev)
            price_df['lower'] = lower
            price_df['middle'] = middle
            price_df['upper'] = upper
            price_df['pct_b'] = pct_b

            price_df['base_mfc'] = mfc_m5_all[base].shift(1).reindex(price_df.index, method='ffill')
            price_df['quote_mfc'] = mfc_m5_all[quote].shift(1).reindex(price_df.index, method='ffill')

            base_h1 = mfc_h1_all[base].shift(1).reindex(price_df.index, method='ffill')
            quote_h1 = mfc_h1_all[quote].shift(1).reindex(price_df.index, method='ffill')
            price_df['base_vel_h1'] = base_h1.diff(periods=12)
            price_df['quote_vel_h1'] = quote_h1.diff(periods=12)

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

            # Bollinger: Buy when price below lower band (pct_b < 0)
            buy_signal = (
                (price_df['base_dir'] == 2) & (price_df['quote_dir'] == 0) &
                (price_df['base_conf'] >= MIN_CONF) & (price_df['quote_conf'] >= MIN_CONF) &
                (price_df['base_mfc'] <= -MFC_EXTREME) &
                (price_df['pct_b'] < 0) &
                (price_df['base_vel_h1'] >= -H1_VEL_THRESHOLD) &
                (~is_friday_afternoon) & buy_vel_ok
            )

            sell_signal = (
                (price_df['base_dir'] == 0) & (price_df['quote_dir'] == 2) &
                (price_df['base_conf'] >= MIN_CONF) & (price_df['quote_conf'] >= MIN_CONF) &
                (price_df['base_mfc'] >= MFC_EXTREME) &
                (price_df['pct_b'] > 1) &
                (price_df['base_vel_h1'] <= H1_VEL_THRESHOLD) &
                (~is_friday_afternoon) & sell_vel_ok
            )

            for signal_type, signals, is_buy in [('BUY', buy_signal, True), ('SELL', sell_signal, False)]:
                indices = price_df.index[signals].tolist()
                i = 0
                while i < len(indices):
                    signal_time = indices[i]
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

                    if exit_at == 'middle':
                        if is_buy:
                            exit_mask = future_df['Close'] >= future_df['middle']
                        else:
                            exit_mask = future_df['Close'] <= future_df['middle']
                    else:  # opposite
                        if is_buy:
                            exit_mask = future_df['Close'] >= future_df['upper']
                        else:
                            exit_mask = future_df['Close'] <= future_df['lower']

                    if exit_mask.any():
                        exit_time = future_df.index[exit_mask.argmax()]
                        exit_price = price_df.loc[exit_time, 'Close']
                        exit_reason = 'INDICATOR'
                        bars_held = price_df.index.get_loc(exit_time) - entry_idx
                    else:
                        exit_time = future_df.index[-1]
                        exit_price = future_df.iloc[-1]['Close']
                        exit_reason = 'TIMEOUT'
                        bars_held = len(future_df)

                    if is_buy:
                        pips = (exit_price - entry_price) / pip_val
                    else:
                        pips = (entry_price - exit_price) / pip_val

                    net_pips = pips - SPREADS.get(pair, 2.0)

                    all_trades.append({
                        'pair': pair, 'type': signal_type, 'entry_time': entry_time,
                        'pips': pips, 'net_pips': net_pips, 'win': 1 if net_pips > 0 else 0,
                        'exit_reason': exit_reason, 'bars_held': bars_held,
                    })

                    while i < len(indices) and indices[i] <= exit_time:
                        i += 1

        except Exception as e:
            pass

    return pd.DataFrame(all_trades)


# Run tests
print("\n" + "=" * 70)
print("RUNNING BACKTESTS")
print("=" * 70)

results = {}

# Stochastic baseline
print("\nTesting Stochastic 25 (baseline)...")
results['Stoch 25'] = run_backtest_stochastic(period=25)

# CCI variations
for period in [14, 20, 25]:
    for level in [100, 150, 200]:
        name = f'CCI {period} L{level}'
        print(f"Testing {name}...")
        results[name] = run_backtest_cci(period=period, entry_level=level)

# Bollinger Bands variations
for period in [20, 25]:
    for std in [2, 2.5]:
        for exit_at in ['middle', 'opposite']:
            name = f'BB {period} {std}s {exit_at[:3]}'
            print(f"Testing {name}...")
            results[name] = run_backtest_bollinger(period=period, std_dev=std, exit_at=exit_at)

# Print results
print("\n" + "=" * 70)
print("RESULTS COMPARISON")
print("=" * 70)

print(f"\n{'Strategy':<25} {'Trades':>7} {'WR':>7} {'Avg':>8} {'Total':>10} {'Timeouts':>10}")
print("-" * 75)

# Sort by total pips
sorted_results = sorted(results.items(), key=lambda x: x[1]['net_pips'].sum() if len(x[1]) > 0 else -999999, reverse=True)

for name, df in sorted_results:
    if len(df) > 0:
        trades = len(df)
        wr = df['win'].mean() * 100
        avg = df['net_pips'].mean()
        total = df['net_pips'].sum()
        timeouts = (df['exit_reason'] == 'TIMEOUT').sum()
        to_pct = timeouts / trades * 100
        marker = " <-- BASELINE" if name == 'Stoch 25' else ""
        print(f"{name:<25} {trades:>7} {wr:>6.1f}% {avg:>+7.1f} {total:>+10.0f} {timeouts:>6} ({to_pct:.1f}%){marker}")

# Best vs Baseline
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

baseline = results['Stoch 25']
if len(baseline) > 0:
    baseline_total = baseline['net_pips'].sum()
    print(f"\nBaseline (Stoch 25): {len(baseline)} trades, {baseline_total:+.0f} pips")

    print("\nTop 5 strategies:")
    for i, (name, df) in enumerate(sorted_results[:5]):
        if len(df) > 0:
            diff = df['net_pips'].sum() - baseline_total
            print(f"  {i+1}. {name}: {len(df)} trades, {df['net_pips'].sum():+.0f} pips ({diff:+.0f} vs baseline)")

print(f"\nCompleted: {datetime.now()}")
