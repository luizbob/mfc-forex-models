"""
Train XGBoost model to filter trades for best returns
Uses features available at entry time to predict trade outcome
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
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

DATA_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/data')
LSTM_DATA_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/lstm_model/data')
MODEL_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/lstm_model/models')

with open(LSTM_DATA_DIR / 'config.pkl', 'rb') as f:
    config = pickle.load(f)

LOOKBACK = config['lookback']
CURRENCIES = config['currencies']

# Training data: 2021-2022 (out of sample for LSTM which trained on 2023-2024)
# Test data: 2025
TRAIN_YEARS = {
    '2021': ('2021-01-01', '2021-12-31'),
    '2022': ('2022-01-01', '2022-12-31'),
}
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

# Pair encoding for XGBoost
PAIR_ENCODING = {pair: i for i, (pair, _, _) in enumerate(ALL_PAIRS)}

def calculate_stochastic(high, low, close, period=25):
    lowest_low = low.rolling(window=period).min()
    highest_high = high.rolling(window=period).max()
    stoch_k = ((close - lowest_low) / (highest_high - lowest_low)) * 100
    return stoch_k

def get_pip_value(pair):
    return 0.01 if 'JPY' in pair else 0.0001

print("=" * 70)
print("XGBOOST TRADE FILTER - TRAINING")
print("=" * 70)
print(f"Started: {datetime.now()}")

# Load ALL MFC data
print("\nLoading MFC data...")
mfc_m5_all = {}
mfc_m15_all = {}
mfc_m30_all = {}
mfc_h1_all = {}
mfc_h4_all = {}

for cur in CURRENCIES:
    df = pd.read_csv(DATA_DIR / 'cleaned' / f'mfc_currency_{cur}_M5_clean.csv')
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y.%m.%d %H:%M')
    df = df.set_index('datetime')
    mfc_m5_all[cur] = df['MFC']

    df = pd.read_csv(DATA_DIR / 'cleaned' / f'mfc_currency_{cur}_M15_clean.csv')
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y.%m.%d %H:%M')
    df = df.set_index('datetime')
    mfc_m15_all[cur] = df['MFC']

    df = pd.read_csv(DATA_DIR / 'cleaned' / f'mfc_currency_{cur}_M30_clean.csv')
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y.%m.%d %H:%M')
    df = df.set_index('datetime')
    mfc_m30_all[cur] = df['MFC']

    df = pd.read_csv(DATA_DIR / 'cleaned' / f'mfc_currency_{cur}_H1_clean.csv')
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y.%m.%d %H:%M')
    df = df.set_index('datetime')
    mfc_h1_all[cur] = df['MFC']

    df = pd.read_csv(DATA_DIR / 'cleaned' / f'mfc_currency_{cur}_H4_clean.csv')
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y.%m.%d %H:%M')
    df = df.set_index('datetime')
    mfc_h4_all[cur] = df['MFC']

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

# Load price data
print("Loading price data...")
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
MAX_HOLD_BARS = 24  # 24 M5 bars = 2 hours (must match live trading)

def generate_trades_with_features(start_date, end_date):
    """Generate trades with all features for XGBoost"""
    all_trades = []

    for pair, base, quote in ALL_PAIRS:
        if pair not in price_data_all:
            continue
        if base not in lstm_predictions_all or quote not in lstm_predictions_all:
            continue

        pip_val = get_pip_value(pair)

        try:
            price_df = price_data_all[pair].copy()
            price_df = price_df[(price_df.index >= start_date) & (price_df.index <= end_date)]
            if len(price_df) < 100:
                continue

            price_df['stoch'] = calculate_stochastic(price_df['High'], price_df['Low'], price_df['Close'], period=STOCH_PERIOD)

            # M5 MFC
            price_df['base_mfc'] = mfc_m5_all[base].shift(1).reindex(price_df.index, method='ffill')
            price_df['quote_mfc'] = mfc_m5_all[quote].shift(1).reindex(price_df.index, method='ffill')

            # M15 MFC
            price_df['base_mfc_m15'] = mfc_m15_all[base].shift(1).reindex(price_df.index, method='ffill')
            price_df['quote_mfc_m15'] = mfc_m15_all[quote].shift(1).reindex(price_df.index, method='ffill')

            # M30 MFC
            price_df['base_mfc_m30'] = mfc_m30_all[base].shift(1).reindex(price_df.index, method='ffill')
            price_df['quote_mfc_m30'] = mfc_m30_all[quote].shift(1).reindex(price_df.index, method='ffill')

            # H1 MFC
            price_df['base_mfc_h1'] = mfc_h1_all[base].shift(1).reindex(price_df.index, method='ffill')
            price_df['quote_mfc_h1'] = mfc_h1_all[quote].shift(1).reindex(price_df.index, method='ffill')

            # H4 MFC
            price_df['base_mfc_h4'] = mfc_h4_all[base].shift(1).reindex(price_df.index, method='ffill')
            price_df['quote_mfc_h4'] = mfc_h4_all[quote].shift(1).reindex(price_df.index, method='ffill')

            # H1 velocity
            base_h1 = mfc_h1_all[base].shift(1).reindex(price_df.index, method='ffill')
            quote_h1 = mfc_h1_all[quote].shift(1).reindex(price_df.index, method='ffill')
            price_df['base_vel_h1'] = base_h1.diff(periods=12)
            price_df['quote_vel_h1'] = quote_h1.diff(periods=12)

            # H4 velocity (2 H4 bars = 8 hours = 96 M5 bars for consistent calculation)
            base_h4 = mfc_h4_all[base].shift(1).reindex(price_df.index, method='ffill')
            quote_h4 = mfc_h4_all[quote].shift(1).reindex(price_df.index, method='ffill')
            price_df['base_vel_h4'] = base_h4.diff(periods=96)
            price_df['quote_vel_h4'] = quote_h4.diff(periods=96)

            # M5 velocity
            price_df['base_vel_m5'] = mfc_m5_all[base].shift(1).reindex(price_df.index, method='ffill').diff(periods=12)
            price_df['quote_vel_m5'] = mfc_m5_all[quote].shift(1).reindex(price_df.index, method='ffill').diff(periods=12)

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

            # Original strategy signals (no H4 filter for more data)
            buy_signal = (
                (price_df['base_dir'] == 2) &
                (price_df['quote_dir'] == 0) &
                (price_df['base_conf'] >= MIN_CONF) &
                (price_df['quote_conf'] >= MIN_CONF) &
                (price_df['base_mfc'] <= -MFC_EXTREME) &
                (price_df['stoch'] < STOCH_LOW) &
                (price_df['base_vel_h1'] >= -H1_VEL_THRESHOLD) &
                (~is_friday_afternoon) &
                (buy_vel_ok)
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
                (sell_vel_ok)
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
                signal_row = price_df.loc[signal_time]
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
                    bars_held = (price_df.index.get_loc(exit_time) - entry_idx)
                else:
                    exit_time = future_df.index[-1]
                    exit_price = future_df.iloc[-1]['Close']
                    pips = (exit_price - entry_price) / pip_val
                    exit_reason = 'TIMEOUT'
                    bars_held = len(future_df)

                net_pips = pips - SPREADS.get(pair, 2.0)

                # Features available at signal time
                all_trades.append({
                    'pair': pair,
                    'pair_code': PAIR_ENCODING[pair],
                    'type': 'BUY',
                    'type_code': 1,
                    'entry_time': entry_time,
                    'hour': signal_time.hour,
                    'dayofweek': signal_time.dayofweek,
                    'stoch': signal_row['stoch'],
                    'base_mfc': signal_row['base_mfc'],
                    'quote_mfc': signal_row['quote_mfc'],
                    'mfc_diff': signal_row['base_mfc'] - signal_row['quote_mfc'],
                    'base_mfc_m15': signal_row['base_mfc_m15'],
                    'quote_mfc_m15': signal_row['quote_mfc_m15'],
                    'mfc_diff_m15': signal_row['base_mfc_m15'] - signal_row['quote_mfc_m15'],
                    'base_mfc_m30': signal_row['base_mfc_m30'],
                    'quote_mfc_m30': signal_row['quote_mfc_m30'],
                    'mfc_diff_m30': signal_row['base_mfc_m30'] - signal_row['quote_mfc_m30'],
                    'base_mfc_h1': signal_row['base_mfc_h1'],
                    'quote_mfc_h1': signal_row['quote_mfc_h1'],
                    'mfc_diff_h1': signal_row['base_mfc_h1'] - signal_row['quote_mfc_h1'],
                    'base_mfc_h4': signal_row['base_mfc_h4'],
                    'quote_mfc_h4': signal_row['quote_mfc_h4'],
                    'mfc_diff_h4': signal_row['base_mfc_h4'] - signal_row['quote_mfc_h4'],
                    'base_vel_h1': signal_row['base_vel_h1'],
                    'quote_vel_h1': signal_row['quote_vel_h1'],
                    'vel_h1_diff': signal_row['base_vel_h1'] - signal_row['quote_vel_h1'],
                    'base_vel_h4': signal_row['base_vel_h4'],
                    'quote_vel_h4': signal_row['quote_vel_h4'],
                    'vel_h4_diff': signal_row['base_vel_h4'] - signal_row['quote_vel_h4'],
                    'base_vel_m5': signal_row['base_vel_m5'],
                    'quote_vel_m5': signal_row['quote_vel_m5'],
                    'vel_m5_diff': signal_row['base_vel_m5'] - signal_row['quote_vel_m5'],
                    'base_conf': signal_row['base_conf'],
                    'quote_conf': signal_row['quote_conf'],
                    'conf_avg': (signal_row['base_conf'] + signal_row['quote_conf']) / 2,
                    'pips': pips,
                    'net_pips': net_pips,
                    'win': 1 if net_pips > 0 else 0,
                    'exit_reason': exit_reason,
                    'bars_held': bars_held,
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
                signal_row = price_df.loc[signal_time]
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
                    bars_held = (price_df.index.get_loc(exit_time) - entry_idx)
                else:
                    exit_time = future_df.index[-1]
                    exit_price = future_df.iloc[-1]['Close']
                    pips = (entry_price - exit_price) / pip_val
                    exit_reason = 'TIMEOUT'
                    bars_held = len(future_df)

                net_pips = pips - SPREADS.get(pair, 2.0)

                all_trades.append({
                    'pair': pair,
                    'pair_code': PAIR_ENCODING[pair],
                    'type': 'SELL',
                    'type_code': 0,
                    'entry_time': entry_time,
                    'hour': signal_time.hour,
                    'dayofweek': signal_time.dayofweek,
                    'stoch': signal_row['stoch'],
                    'base_mfc': signal_row['base_mfc'],
                    'quote_mfc': signal_row['quote_mfc'],
                    'mfc_diff': signal_row['base_mfc'] - signal_row['quote_mfc'],
                    'base_mfc_m15': signal_row['base_mfc_m15'],
                    'quote_mfc_m15': signal_row['quote_mfc_m15'],
                    'mfc_diff_m15': signal_row['base_mfc_m15'] - signal_row['quote_mfc_m15'],
                    'base_mfc_m30': signal_row['base_mfc_m30'],
                    'quote_mfc_m30': signal_row['quote_mfc_m30'],
                    'mfc_diff_m30': signal_row['base_mfc_m30'] - signal_row['quote_mfc_m30'],
                    'base_mfc_h1': signal_row['base_mfc_h1'],
                    'quote_mfc_h1': signal_row['quote_mfc_h1'],
                    'mfc_diff_h1': signal_row['base_mfc_h1'] - signal_row['quote_mfc_h1'],
                    'base_mfc_h4': signal_row['base_mfc_h4'],
                    'quote_mfc_h4': signal_row['quote_mfc_h4'],
                    'mfc_diff_h4': signal_row['base_mfc_h4'] - signal_row['quote_mfc_h4'],
                    'base_vel_h1': signal_row['base_vel_h1'],
                    'quote_vel_h1': signal_row['quote_vel_h1'],
                    'vel_h1_diff': signal_row['base_vel_h1'] - signal_row['quote_vel_h1'],
                    'base_vel_h4': signal_row['base_vel_h4'],
                    'quote_vel_h4': signal_row['quote_vel_h4'],
                    'vel_h4_diff': signal_row['base_vel_h4'] - signal_row['quote_vel_h4'],
                    'base_vel_m5': signal_row['base_vel_m5'],
                    'quote_vel_m5': signal_row['quote_vel_m5'],
                    'vel_m5_diff': signal_row['base_vel_m5'] - signal_row['quote_vel_m5'],
                    'base_conf': signal_row['base_conf'],
                    'quote_conf': signal_row['quote_conf'],
                    'conf_avg': (signal_row['base_conf'] + signal_row['quote_conf']) / 2,
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

# Generate training data (2021-2022)
print("\n" + "=" * 70)
print("GENERATING TRAINING DATA (2021-2022)")
print("=" * 70)

train_dfs = []
for year_name, (start_date, end_date) in TRAIN_YEARS.items():
    print(f"\nGenerating {year_name}...")
    df = generate_trades_with_features(start_date, end_date)
    print(f"  {len(df)} trades")
    train_dfs.append(df)

train_df = pd.concat(train_dfs, ignore_index=True)
print(f"\nTotal training trades: {len(train_df)}")
print(f"Win rate: {train_df['win'].mean()*100:.1f}%")
print(f"Avg pips: {train_df['net_pips'].mean():+.1f}")

# Generate test data (2025)
print("\n" + "=" * 70)
print("GENERATING TEST DATA (2025)")
print("=" * 70)

test_df = generate_trades_with_features(TEST_YEAR[0], TEST_YEAR[1])
print(f"Total test trades: {len(test_df)}")
print(f"Win rate: {test_df['win'].mean()*100:.1f}%")
print(f"Avg pips: {test_df['net_pips'].mean():+.1f}")

# Feature columns for XGBoost (now with MTF MFC features)
FEATURE_COLS = [
    'pair_code', 'type_code', 'hour', 'dayofweek',
    'stoch', 'base_mfc', 'quote_mfc', 'mfc_diff',
    # MTF MFC features (NEW)
    'base_mfc_m15', 'quote_mfc_m15', 'mfc_diff_m15',
    'base_mfc_m30', 'quote_mfc_m30', 'mfc_diff_m30',
    'base_mfc_h1', 'quote_mfc_h1', 'mfc_diff_h1',
    'base_mfc_h4', 'quote_mfc_h4', 'mfc_diff_h4',
    # Velocities
    'base_vel_h1', 'quote_vel_h1', 'vel_h1_diff',
    'base_vel_h4', 'quote_vel_h4', 'vel_h4_diff',
    'base_vel_m5', 'quote_vel_m5', 'vel_m5_diff',
    'base_conf', 'quote_conf', 'conf_avg',
]

X_train = train_df[FEATURE_COLS].values
y_train_class = train_df['win'].values
y_train_reg = train_df['net_pips'].values

X_test = test_df[FEATURE_COLS].values
y_test_class = test_df['win'].values
y_test_reg = test_df['net_pips'].values

# Train Classification model (predict win/loss)
print("\n" + "=" * 70)
print("TRAINING XGBOOST CLASSIFIER (Win/Loss)")
print("=" * 70)

clf = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
)

clf.fit(X_train, y_train_class)

# Train Regression model (predict pips)
print("\n" + "=" * 70)
print("TRAINING XGBOOST REGRESSOR (Predict Pips)")
print("=" * 70)

reg = XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
)

reg.fit(X_train, y_train_reg)

# Predictions on test set
test_df['pred_win'] = clf.predict(X_test)
test_df['pred_prob'] = clf.predict_proba(X_test)[:, 1]
test_df['pred_pips'] = reg.predict(X_test)

# Evaluate different filtering strategies
print("\n" + "=" * 70)
print("TEST SET RESULTS (2025)")
print("=" * 70)

print(f"\n--- BASELINE (No Filter) ---")
print(f"Trades: {len(test_df)}")
print(f"Win Rate: {test_df['win'].mean()*100:.1f}%")
print(f"Avg Pips: {test_df['net_pips'].mean():+.1f}")
print(f"Total Pips: {test_df['net_pips'].sum():+.0f}")

print(f"\n--- CLASSIFIER FILTER (pred_win == 1) ---")
filtered = test_df[test_df['pred_win'] == 1]
print(f"Trades: {len(filtered)} ({len(filtered)/len(test_df)*100:.0f}%)")
print(f"Win Rate: {filtered['win'].mean()*100:.1f}%")
print(f"Avg Pips: {filtered['net_pips'].mean():+.1f}")
print(f"Total Pips: {filtered['net_pips'].sum():+.0f}")

print(f"\n--- PROBABILITY FILTERS ---")
for prob_thresh in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]:
    filtered = test_df[test_df['pred_prob'] >= prob_thresh]
    if len(filtered) > 0:
        print(f"prob >= {prob_thresh}: {len(filtered):>5} trades, {filtered['win'].mean()*100:>5.1f}% WR, {filtered['net_pips'].mean():>+6.1f} avg, {filtered['net_pips'].sum():>+8.0f} total")

print(f"\n--- REGRESSION FILTERS (pred_pips > threshold) ---")
for pips_thresh in [0, 2, 4, 6, 8, 10]:
    filtered = test_df[test_df['pred_pips'] >= pips_thresh]
    if len(filtered) > 0:
        print(f"pred >= {pips_thresh}: {len(filtered):>5} trades, {filtered['win'].mean()*100:>5.1f}% WR, {filtered['net_pips'].mean():>+6.1f} avg, {filtered['net_pips'].sum():>+8.0f} total")

# Feature importance
print("\n" + "=" * 70)
print("FEATURE IMPORTANCE (Classifier)")
print("=" * 70)

importance = pd.DataFrame({
    'feature': FEATURE_COLS,
    'importance': clf.feature_importances_
}).sort_values('importance', ascending=False)

for _, row in importance.iterrows():
    print(f"{row['feature']:<20} {row['importance']:.4f}")

# Save models
print("\n" + "=" * 70)
print("SAVING MODELS")
print("=" * 70)

joblib.dump(clf, MODEL_DIR / 'xgb_trade_classifier.joblib')
joblib.dump(reg, MODEL_DIR / 'xgb_trade_regressor.joblib')
print(f"Saved classifier to: {MODEL_DIR / 'xgb_trade_classifier.joblib'}")
print(f"Saved regressor to: {MODEL_DIR / 'xgb_trade_regressor.joblib'}")

# Save test trades with predictions
test_df.to_csv(LSTM_DATA_DIR / 'trades_2025_with_predictions.csv', index=False)
print(f"Saved test trades to: {LSTM_DATA_DIR / 'trades_2025_with_predictions.csv'}")

print(f"\nCompleted: {datetime.now()}")
