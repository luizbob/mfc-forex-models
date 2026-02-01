"""
Script 15: Multi-Timeframe Conflict Filter Test
================================================
Test the idea that trades fail when M15/M30 conflict with M5 signal.

Based on live trading analysis:
- Losing trades had M15/M30/H1 CONFLICT with M5 signal
- Winning trades were more aligned across timeframes

Filter: Don't enter if M15 or M30 shows CONFLICT with trade direction.
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
log("TEST: MULTI-TIMEFRAME CONFLICT FILTER")
log("=" * 70)
log(f"Started: {datetime.now()}")

MODEL_ACCURACY = {
    'JPY': 90.0, 'USD': 87.8, 'AUD': 87.4, 'NZD': 86.7,
    'CHF': 86.3, 'GBP': 83.7, 'CAD': 82.2, 'EUR': 82.0,
}

DATA_END_DATES = {
    'EUR': '2025-12-21', 'NZD': '2025-12-21', 'USD': '2025-12-21',
    'GBP': '2025-12-21', 'JPY': '2025-12-21', 'CHF': '2025-12-21',
    'CAD': '2025-12-21', 'AUD': '2025-12-21',
}

EXCLUDE_PAIRS = []

DATA_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/data')
LSTM_DATA_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/lstm_model/data')
MODEL_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/lstm_model/models')

with open(LSTM_DATA_DIR / 'config.pkl', 'rb') as f:
    config = pickle.load(f)

LOOKBACK = config['lookback']
CURRENCIES = config['currencies']

START_DATE = '2025-01-01'
END_DATE = '2025-12-21'

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

# ============================================================================
# LOAD MFC DATA
# ============================================================================
log("\n1. Loading MFC data...")

mfc_m5 = {}
mfc_m15 = {}
mfc_m30 = {}
mfc_h1 = {}
mfc_h4 = {}

for cur in CURRENCIES:
    end_date = DATA_END_DATES[cur]

    for timeframe, mfc_dict in [('M5', mfc_m5), ('M15', mfc_m15), ('M30', mfc_m30), ('H1', mfc_h1), ('H4', mfc_h4)]:
        df = pd.read_csv(DATA_DIR / 'cleaned' / f'mfc_currency_{cur}_{timeframe}_clean.csv')
        df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y.%m.%d %H:%M')
        df = df.set_index('datetime')
        df = df[(df.index >= START_DATE) & (df.index <= end_date)]
        mfc_dict[cur] = df['MFC']

    log(f"  {cur}: M5={len(mfc_m5[cur])}")

# ============================================================================
# LOAD LSTM MODELS
# ============================================================================
log("\n2. Generating LSTM predictions...")

lstm_predictions = {}

for ccy in CURRENCIES:
    end_date = DATA_END_DATES[ccy]
    model = tf.keras.models.load_model(MODEL_DIR / f'lstm_{ccy}_final.keras')

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

    log(f"  {ccy}: {len(lstm_predictions[ccy])} predictions")

    del model, X_val, pred
    tf.keras.backend.clear_session()
    gc.collect()

# ============================================================================
# LOAD PRICE DATA
# ============================================================================
log("\n3. Loading price data...")

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
            price_m5['rsi'] = calculate_rsi(price_m5['Close'], period=21)
            price_data[pair] = price_m5
            log(f"  {pair}: {len(price_m5)} bars")
    except Exception as e:
        log(f"  {pair}: ERROR - {e}")

# ============================================================================
# RUN STRATEGY - WITH AND WITHOUT MTF FILTER
# ============================================================================
log("\n" + "=" * 70)
log("4. RUNNING STRATEGY COMPARISON")
log("=" * 70)

MIN_CONF = 0.70
MFC_EXTREME = 0.5
RSI_LOW = 20
RSI_HIGH = 80
H1_VEL_THRESHOLD = 0.04
MAX_HOLD_BARS = 24  # 2 hours (matching live trader)

# MTF Conflict filter thresholds
MTF_CONFLICT_THRESHOLD = 0.3  # If base/quote exceeds this in wrong direction = CONFLICT

def run_backtest(use_mtf_filter=False, filter_name="", conflict_threshold=0.3, check_m30=True):
    all_trades = []

    for pair, base, quote in ALL_PAIRS:
        if pair not in price_data:
            continue
        if base not in lstm_predictions or quote not in lstm_predictions:
            continue
        if pair in EXCLUDE_PAIRS:
            continue

        pip_val = get_pip_value(pair)

        try:
            price_df = price_data[pair].copy()

            # M5 MFC (shifted)
            price_df['base_mfc_m5'] = mfc_m5[base].shift(1).reindex(price_df.index, method='ffill')
            price_df['quote_mfc_m5'] = mfc_m5[quote].shift(1).reindex(price_df.index, method='ffill')

            # M15 MFC (shifted) - for conflict filter
            price_df['base_mfc_m15'] = mfc_m15[base].shift(1).reindex(price_df.index, method='ffill')
            price_df['quote_mfc_m15'] = mfc_m15[quote].shift(1).reindex(price_df.index, method='ffill')

            # M30 MFC (shifted) - for conflict filter
            price_df['base_mfc_m30'] = mfc_m30[base].shift(1).reindex(price_df.index, method='ffill')
            price_df['quote_mfc_m30'] = mfc_m30[quote].shift(1).reindex(price_df.index, method='ffill')

            # H1 MFC (shifted) - for conflict filter
            price_df['base_mfc_h1'] = mfc_h1[base].shift(1).reindex(price_df.index, method='ffill')
            price_df['quote_mfc_h1'] = mfc_h1[quote].shift(1).reindex(price_df.index, method='ffill')

            # H1 velocity
            base_h1 = mfc_h1[base].shift(1).reindex(price_df.index, method='ffill')
            quote_h1 = mfc_h1[quote].shift(1).reindex(price_df.index, method='ffill')
            price_df['base_vel_h1'] = base_h1.diff(periods=12)
            price_df['quote_vel_h1'] = quote_h1.diff(periods=12)

            # LSTM predictions
            base_lstm = lstm_predictions[base].reindex(price_df.index, method='ffill')
            quote_lstm = lstm_predictions[quote].reindex(price_df.index, method='ffill')

            price_df['base_dir'] = base_lstm['direction']
            price_df['base_conf'] = base_lstm['confidence']
            price_df['quote_dir'] = quote_lstm['direction']
            price_df['quote_conf'] = quote_lstm['confidence']

            price_df = price_df.dropna()

            # Friday filter
            is_friday_afternoon = (price_df.index.dayofweek == 4) & (price_df.index.hour >= 6)

            # Velocity filter
            buy_vel_ok = (price_df['base_vel_h1'] - price_df['quote_vel_h1']) > 0
            sell_vel_ok = (price_df['quote_vel_h1'] - price_df['base_vel_h1']) > 0

            # ================================================================
            # MTF CONFLICT FILTER
            # ================================================================
            # For BUY: CONFLICT if base is overbought (>0.3) or quote is oversold (<-0.3) on M15/M30/H1
            # For SELL: CONFLICT if base is oversold (<-0.3) or quote is overbought (>0.3) on M15/M30/H1

            if use_mtf_filter:
                # BUY conflicts: base overbought OR quote oversold on higher TFs
                buy_conflict_m15 = (price_df['base_mfc_m15'] > conflict_threshold) | (price_df['quote_mfc_m15'] < -conflict_threshold)
                buy_conflict_m30 = (price_df['base_mfc_m30'] > conflict_threshold) | (price_df['quote_mfc_m30'] < -conflict_threshold)

                # SELL conflicts: base oversold OR quote overbought on higher TFs
                sell_conflict_m15 = (price_df['base_mfc_m15'] < -conflict_threshold) | (price_df['quote_mfc_m15'] > conflict_threshold)
                sell_conflict_m30 = (price_df['base_mfc_m30'] < -conflict_threshold) | (price_df['quote_mfc_m30'] > conflict_threshold)

                if check_m30:
                    buy_no_conflict = ~(buy_conflict_m15 | buy_conflict_m30)
                    sell_no_conflict = ~(sell_conflict_m15 | sell_conflict_m30)
                else:
                    buy_no_conflict = ~buy_conflict_m15
                    sell_no_conflict = ~sell_conflict_m15
            else:
                buy_no_conflict = True
                sell_no_conflict = True

            # BUY signal
            buy_signal = (
                (price_df['base_dir'] == 2) &
                (price_df['quote_dir'] == 0) &
                (price_df['base_conf'] >= MIN_CONF) &
                (price_df['quote_conf'] >= MIN_CONF) &
                (price_df['base_mfc_m5'] <= -MFC_EXTREME) &
                (price_df['rsi'] < RSI_LOW) &
                (price_df['base_vel_h1'] >= -H1_VEL_THRESHOLD) &
                (~is_friday_afternoon) &
                (buy_vel_ok) &
                (buy_no_conflict)
            )

            # SELL signal
            sell_signal = (
                (price_df['base_dir'] == 0) &
                (price_df['quote_dir'] == 2) &
                (price_df['base_conf'] >= MIN_CONF) &
                (price_df['quote_conf'] >= MIN_CONF) &
                (price_df['base_mfc_m5'] >= MFC_EXTREME) &
                (price_df['rsi'] > RSI_HIGH) &
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
                future_df = price_df.iloc[entry_idx+1:entry_idx+1+MAX_HOLD_BARS]

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
                future_df = price_df.iloc[entry_idx+1:entry_idx+1+MAX_HOLD_BARS]

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

    return all_trades

# ============================================================================
# TEST MULTIPLE CONFIGURATIONS
# ============================================================================

def get_stats(trades):
    if len(trades) == 0:
        return {'trades': 0, 'wr': 0, 'net_avg': 0, 'total': 0}
    df = pd.DataFrame(trades)
    df['net_pips'] = df['pips'] - df['spread']
    return {
        'trades': len(df),
        'wr': df['win'].mean()*100,
        'net_avg': df['net_pips'].mean(),
        'total': df['net_pips'].sum(),
        'df': df
    }

# Run baseline first
log("\nRunning baseline (no filter)...")
trades_baseline = run_backtest(use_mtf_filter=False)
baseline = get_stats(trades_baseline)

log(f"  Baseline: {baseline['trades']} trades, {baseline['wr']:.1f}% WR, {baseline['total']:.0f} pips")

# Test multiple configurations
log("\n" + "=" * 70)
log("TESTING DIFFERENT FILTER CONFIGURATIONS")
log("=" * 70)

configs = [
    # (threshold, check_m30, name)
    (0.3, False, "M15 only, thresh=0.3"),
    (0.3, True, "M15+M30, thresh=0.3"),
    (0.4, False, "M15 only, thresh=0.4"),
    (0.4, True, "M15+M30, thresh=0.4"),
    (0.5, False, "M15 only, thresh=0.5"),
    (0.5, True, "M15+M30, thresh=0.5"),
    (0.6, False, "M15 only, thresh=0.6"),
    (0.6, True, "M15+M30, thresh=0.6"),
    (0.7, False, "M15 only, thresh=0.7"),
    (0.7, True, "M15+M30, thresh=0.7"),
]

results = []

for threshold, check_m30, name in configs:
    trades = run_backtest(use_mtf_filter=True, conflict_threshold=threshold, check_m30=check_m30)
    stats = get_stats(trades)

    # Calculate filtered trades stats
    if stats['trades'] > 0:
        no_filter_times = set(pd.DataFrame(trades_baseline)['entry_time'])
        with_filter_times = set(pd.DataFrame(trades)['entry_time'])
        filtered_times = no_filter_times - with_filter_times
        filtered_trades = [t for t in trades_baseline if t['entry_time'] in filtered_times]

        if filtered_trades:
            filt_df = pd.DataFrame(filtered_trades)
            filt_df['net_pips'] = filt_df['pips'] - filt_df['spread']
            filt_wr = filt_df['win'].mean()*100
            filt_pips = filt_df['net_pips'].sum()
        else:
            filt_wr = 0
            filt_pips = 0
    else:
        filt_wr = 0
        filt_pips = 0

    results.append({
        'name': name,
        'threshold': threshold,
        'check_m30': check_m30,
        'trades': stats['trades'],
        'wr': stats['wr'],
        'net_avg': stats['net_avg'],
        'total': stats['total'],
        'removed': baseline['trades'] - stats['trades'],
        'removed_wr': filt_wr,
        'removed_pips': filt_pips
    })

    good = "GOOD" if filt_wr < baseline['wr'] else "BAD"
    log(f"  {name}: {stats['trades']} trades, {stats['wr']:.1f}% WR, {stats['total']:.0f} pips | Removed: {baseline['trades']-stats['trades']} ({filt_wr:.1f}% WR) [{good}]")

# ============================================================================
# RESULTS SUMMARY
# ============================================================================
log("\n" + "=" * 70)
log("RESULTS SUMMARY")
log("=" * 70)

log(f"\nBaseline: {baseline['trades']} trades, {baseline['wr']:.1f}% WR, {baseline['total']:.0f} pips")

log(f"\n{'Config':<25} | {'Trades':>7} | {'WR':>6} | {'Pips':>8} | {'Removed':>8} | {'Rem WR':>7} | {'Quality'}")
log("-" * 90)

for r in results:
    quality = "GOOD" if r['removed_wr'] < baseline['wr'] - 5 else ("OK" if r['removed_wr'] < baseline['wr'] else "BAD")
    log(f"{r['name']:<25} | {r['trades']:>7} | {r['wr']:>5.1f}% | {r['total']:>+8.0f} | {r['removed']:>8} | {r['removed_wr']:>6.1f}% | {quality}")

# Find best configuration (removes losers while keeping most winners)
log("\n" + "=" * 70)
log("BEST CONFIGURATIONS")
log("=" * 70)

# Best = removes trades with lower WR than baseline while keeping decent trade count
good_configs = [r for r in results if r['removed_wr'] < baseline['wr'] and r['trades'] >= baseline['trades'] * 0.5]
if good_configs:
    best = max(good_configs, key=lambda x: x['wr'])
    log(f"\nBest for WR improvement:")
    log(f"  {best['name']}: {best['trades']} trades, {best['wr']:.1f}% WR (+{best['wr']-baseline['wr']:.1f}%)")
    log(f"  Removed {best['removed']} trades with {best['removed_wr']:.1f}% WR")

    best_pips = max(good_configs, key=lambda x: x['net_avg'])
    if best_pips != best:
        log(f"\nBest for avg pips:")
        log(f"  {best_pips['name']}: {best_pips['trades']} trades, {best_pips['net_avg']:.2f} avg pips")
else:
    log("\nNo configuration found that removes mostly losers while keeping 50%+ trades")
    log("The filter may need different criteria")

log(f"\nCompleted: {datetime.now()}")
