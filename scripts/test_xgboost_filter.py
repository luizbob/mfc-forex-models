"""
Test XGBoost filter on 2021 and 2022
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
import joblib

DATA_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/data')
LSTM_DATA_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/lstm_model/data')
MODEL_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/lstm_model/models')

with open(LSTM_DATA_DIR / 'config.pkl', 'rb') as f:
    config = pickle.load(f)

LOOKBACK = config['lookback']
CURRENCIES = config['currencies']

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

PAIR_ENCODING = {pair: i for i, (pair, _, _) in enumerate(ALL_PAIRS)}

def calculate_stochastic(high, low, close, period=25):
    lowest_low = low.rolling(window=period).min()
    highest_high = high.rolling(window=period).max()
    stoch_k = ((close - lowest_low) / (highest_high - lowest_low)) * 100
    return stoch_k

def get_pip_value(pair):
    return 0.01 if 'JPY' in pair else 0.0001

print("=" * 70)
print("XGBOOST FILTER TEST - 2021, 2022, 2025")
print("=" * 70)
print(f"Started: {datetime.now()}")

# Load XGBoost model
print("\nLoading XGBoost classifier...")
clf = joblib.load(MODEL_DIR / 'xgb_trade_classifier.joblib')

# Load MFC data
print("Loading MFC data...")
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
    except:
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

FEATURE_COLS = [
    'pair_code', 'type_code', 'hour', 'dayofweek',
    'stoch', 'base_mfc', 'quote_mfc', 'mfc_diff',
    'base_vel_h1', 'quote_vel_h1', 'vel_h1_diff',
    'base_vel_h4', 'quote_vel_h4', 'vel_h4_diff',
    'base_vel_m5', 'quote_vel_m5', 'vel_m5_diff',
    'base_conf', 'quote_conf', 'conf_avg',
]

def test_year_with_xgb(year_name, start_date, end_date, prob_threshold=0.75,
                       use_mtf_filter=False, mtf_threshold=0.3, check_m30=False):
    """Test with XGBoost filter and optional MTF conflict filter"""
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

            price_df['base_mfc'] = mfc_m5_all[base].shift(1).reindex(price_df.index, method='ffill')
            price_df['quote_mfc'] = mfc_m5_all[quote].shift(1).reindex(price_df.index, method='ffill')

            # M15 and M30 MFC for MTF filter
            price_df['base_mfc_m15'] = mfc_m15_all[base].shift(1).reindex(price_df.index, method='ffill')
            price_df['quote_mfc_m15'] = mfc_m15_all[quote].shift(1).reindex(price_df.index, method='ffill')
            price_df['base_mfc_m30'] = mfc_m30_all[base].shift(1).reindex(price_df.index, method='ffill')
            price_df['quote_mfc_m30'] = mfc_m30_all[quote].shift(1).reindex(price_df.index, method='ffill')

            base_h1 = mfc_h1_all[base].shift(1).reindex(price_df.index, method='ffill')
            quote_h1 = mfc_h1_all[quote].shift(1).reindex(price_df.index, method='ffill')
            price_df['base_vel_h1'] = base_h1.diff(periods=12)
            price_df['quote_vel_h1'] = quote_h1.diff(periods=12)

            base_h4 = mfc_h4_all[base].shift(1).reindex(price_df.index, method='ffill')
            quote_h4 = mfc_h4_all[quote].shift(1).reindex(price_df.index, method='ffill')
            price_df['base_vel_h4'] = base_h4.diff(periods=72)
            price_df['quote_vel_h4'] = quote_h4.diff(periods=72)

            price_df['base_vel_m5'] = mfc_m5_all[base].shift(1).reindex(price_df.index, method='ffill').diff(periods=12)
            price_df['quote_vel_m5'] = mfc_m5_all[quote].shift(1).reindex(price_df.index, method='ffill').diff(periods=12)

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

            # MTF Conflict Filter
            # BUY conflict: base overbought (>threshold) OR quote oversold (<-threshold) on M15/M30
            # SELL conflict: base oversold (<-threshold) OR quote overbought (>threshold) on M15/M30
            if use_mtf_filter:
                buy_conflict_m15 = (price_df['base_mfc_m15'] > mtf_threshold) | (price_df['quote_mfc_m15'] < -mtf_threshold)
                sell_conflict_m15 = (price_df['base_mfc_m15'] < -mtf_threshold) | (price_df['quote_mfc_m15'] > mtf_threshold)

                if check_m30:
                    buy_conflict_m30 = (price_df['base_mfc_m30'] > mtf_threshold) | (price_df['quote_mfc_m30'] < -mtf_threshold)
                    sell_conflict_m30 = (price_df['base_mfc_m30'] < -mtf_threshold) | (price_df['quote_mfc_m30'] > mtf_threshold)
                    buy_no_conflict = ~(buy_conflict_m15 | buy_conflict_m30)
                    sell_no_conflict = ~(sell_conflict_m15 | sell_conflict_m30)
                else:
                    buy_no_conflict = ~buy_conflict_m15
                    sell_no_conflict = ~sell_conflict_m15
            else:
                buy_no_conflict = True
                sell_no_conflict = True

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
                (buy_no_conflict)
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
                (sell_no_conflict)
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

                # Create features for XGBoost
                features = {
                    'pair_code': PAIR_ENCODING[pair],
                    'type_code': 1,
                    'hour': signal_time.hour,
                    'dayofweek': signal_time.dayofweek,
                    'stoch': signal_row['stoch'],
                    'base_mfc': signal_row['base_mfc'],
                    'quote_mfc': signal_row['quote_mfc'],
                    'mfc_diff': signal_row['base_mfc'] - signal_row['quote_mfc'],
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
                }

                X = np.array([[features[col] for col in FEATURE_COLS]])
                prob = clf.predict_proba(X)[0, 1]

                exit_mask = future_df['stoch'] >= STOCH_HIGH

                if exit_mask.any():
                    exit_time = future_df.index[exit_mask.argmax()]
                    exit_price = price_df.loc[exit_time, 'Close']
                    pips = (exit_price - entry_price) / pip_val
                else:
                    exit_time = future_df.index[-1]
                    exit_price = future_df.iloc[-1]['Close']
                    pips = (exit_price - entry_price) / pip_val

                net_pips = pips - SPREADS.get(pair, 2.0)

                all_trades.append({
                    'pair': pair,
                    'type': 'BUY',
                    'entry_time': entry_time,
                    'net_pips': net_pips,
                    'win': 1 if net_pips > 0 else 0,
                    'xgb_prob': prob,
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

                features = {
                    'pair_code': PAIR_ENCODING[pair],
                    'type_code': 0,
                    'hour': signal_time.hour,
                    'dayofweek': signal_time.dayofweek,
                    'stoch': signal_row['stoch'],
                    'base_mfc': signal_row['base_mfc'],
                    'quote_mfc': signal_row['quote_mfc'],
                    'mfc_diff': signal_row['base_mfc'] - signal_row['quote_mfc'],
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
                }

                X = np.array([[features[col] for col in FEATURE_COLS]])
                prob = clf.predict_proba(X)[0, 1]

                exit_mask = future_df['stoch'] <= STOCH_LOW

                if exit_mask.any():
                    exit_time = future_df.index[exit_mask.argmax()]
                    exit_price = price_df.loc[exit_time, 'Close']
                    pips = (entry_price - exit_price) / pip_val
                else:
                    exit_time = future_df.index[-1]
                    exit_price = future_df.iloc[-1]['Close']
                    pips = (entry_price - exit_price) / pip_val

                net_pips = pips - SPREADS.get(pair, 2.0)

                all_trades.append({
                    'pair': pair,
                    'type': 'SELL',
                    'entry_time': entry_time,
                    'net_pips': net_pips,
                    'win': 1 if net_pips > 0 else 0,
                    'xgb_prob': prob,
                })

                while i < len(sell_indices) and sell_indices[i] <= exit_time:
                    i += 1

        except Exception as e:
            pass

    if all_trades:
        trades_df = pd.DataFrame(all_trades)
        return trades_df
    return None

# Test each year
print("\n" + "=" * 70)
print("TESTING WITH XGBOOST FILTER (prob >= 0.75)")
print("=" * 70)

results = []

for year_name, (start_date, end_date) in YEAR_RANGES.items():
    print(f"\n{'='*70}")
    print(f"TESTING {year_name}")
    print(f"{'='*70}")

    trades_df = test_year_with_xgb(year_name, start_date, end_date)

    if trades_df is not None:
        # Baseline (no filter)
        baseline_wr = trades_df['win'].mean() * 100
        baseline_avg = trades_df['net_pips'].mean()
        baseline_total = trades_df['net_pips'].sum()

        # XGB filtered (prob >= 0.75)
        filtered = trades_df[trades_df['xgb_prob'] >= 0.75]

        print(f"\n  BASELINE (no XGB filter):")
        print(f"    Trades: {len(trades_df)}")
        print(f"    Win Rate: {baseline_wr:.1f}%")
        print(f"    Avg Pips: {baseline_avg:+.1f}")
        print(f"    Total: {baseline_total:+.0f}")

        if len(filtered) > 0:
            filtered_wr = filtered['win'].mean() * 100
            filtered_avg = filtered['net_pips'].mean()
            filtered_total = filtered['net_pips'].sum()

            print(f"\n  XGB FILTERED (prob >= 0.75):")
            print(f"    Trades: {len(filtered)} ({len(filtered)/len(trades_df)*100:.0f}%)")
            print(f"    Win Rate: {filtered_wr:.1f}%")
            print(f"    Avg Pips: {filtered_avg:+.1f}")
            print(f"    Total: {filtered_total:+.0f}")

            results.append({
                'year': year_name,
                'baseline_trades': len(trades_df),
                'baseline_wr': baseline_wr,
                'baseline_avg': baseline_avg,
                'baseline_total': baseline_total,
                'filtered_trades': len(filtered),
                'filtered_wr': filtered_wr,
                'filtered_avg': filtered_avg,
                'filtered_total': filtered_total,
            })

# Summary
print("\n" + "=" * 70)
print("SUMMARY: XGBOOST FILTER (prob >= 0.75)")
print("=" * 70)

print(f"\n{'Year':<6} | {'--- BASELINE ---':^30} | {'--- XGB FILTERED ---':^30}")
print(f"{'':6} | {'Trades':>8} {'WR':>8} {'Avg':>8} {'Total':>8} | {'Trades':>8} {'WR':>8} {'Avg':>8} {'Total':>8}")
print("-" * 80)

total_base_trades = 0
total_base_pips = 0
total_filt_trades = 0
total_filt_pips = 0

for r in results:
    print(f"{r['year']:<6} | {r['baseline_trades']:>8} {r['baseline_wr']:>7.1f}% {r['baseline_avg']:>+7.1f} {r['baseline_total']:>+8.0f} | {r['filtered_trades']:>8} {r['filtered_wr']:>7.1f}% {r['filtered_avg']:>+7.1f} {r['filtered_total']:>+8.0f}")
    total_base_trades += r['baseline_trades']
    total_base_pips += r['baseline_total']
    total_filt_trades += r['filtered_trades']
    total_filt_pips += r['filtered_total']

print("-" * 80)
print(f"{'TOTAL':<6} | {total_base_trades:>8} {'':>8} {'':>8} {total_base_pips:>+8.0f} | {total_filt_trades:>8} {'':>8} {'':>8} {total_filt_pips:>+8.0f}")

print(f"\nCompleted: {datetime.now()}")
