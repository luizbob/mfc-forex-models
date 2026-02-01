"""
Backtest 2-hour time stop rule
For trades that take longer than 2 hours, calculate what P/L would be if closed at 2h
"""
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

DATA_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/data')
LSTM_DATA_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/lstm_model/data')

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

print("=" * 70)
print("2-HOUR TIME STOP BACKTEST")
print("=" * 70)
print(f"Started: {datetime.now()}")

# Load trades with XGBoost predictions
print("\nLoading trades...")
trades_df = pd.read_csv(LSTM_DATA_DIR / 'trades_2025_with_predictions.csv')
trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])

# Filter by XGBoost (prob >= 0.75)
xgb_trades = trades_df[trades_df['pred_prob'] >= 0.75].copy()
print(f"Total XGB-filtered trades (prob >= 0.75): {len(xgb_trades)}")

# 2 hours = 24 bars (5min each)
TIME_STOP_BARS = 24

# Separate quick vs slow trades
quick_trades = xgb_trades[xgb_trades['bars_held'] <= TIME_STOP_BARS]
slow_trades = xgb_trades[xgb_trades['bars_held'] > TIME_STOP_BARS].copy()

print(f"\nQuick trades (≤2h): {len(quick_trades)}")
print(f"Slow trades (>2h): {len(slow_trades)}")

# Load price data for pairs that have slow trades
pairs_needed = slow_trades['pair'].unique()
print(f"\nLoading price data for {len(pairs_needed)} pairs...")

price_data = {}
for pair in pairs_needed:
    try:
        chunks = []
        for chunk in pd.read_csv(DATA_DIR / f'{pair}_GMT+0_US-DST_M1.csv', chunksize=500000):
            chunk['datetime'] = pd.to_datetime(chunk['Date'] + ' ' + chunk['Time'], format='%Y.%m.%d %H:%M:%S')
            chunk = chunk.set_index('datetime')
            if len(chunk) > 0:
                # Filter to 2025 only
                chunk = chunk[(chunk.index >= '2025-01-01') & (chunk.index <= '2025-12-31')]
                if len(chunk) > 0:
                    m5_chunk = chunk[['Open', 'High', 'Low', 'Close']].resample('5min').agg({
                        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
                    }).dropna()
                    chunks.append(m5_chunk)

        if chunks:
            price_m5 = pd.concat(chunks)
            price_m5 = price_m5[~price_m5.index.duplicated(keep='first')]
            price_data[pair] = price_m5
            print(f"  {pair}: {len(price_m5)} bars")
    except Exception as e:
        print(f"  {pair}: ERROR - {e}")

# Calculate P/L at 2-hour mark for slow trades
print("\n" + "=" * 70)
print("CALCULATING 2-HOUR EXIT P/L FOR SLOW TRADES")
print("=" * 70)

results = []
errors = 0

for idx, trade in slow_trades.iterrows():
    pair = trade['pair']
    entry_time = trade['entry_time']
    trade_type = trade['type']
    pip_val = get_pip_value(pair)
    spread = SPREADS.get(pair, 2.0)

    if pair not in price_data:
        errors += 1
        continue

    prices = price_data[pair]

    # Find entry price (Open at entry_time)
    if entry_time not in prices.index:
        # Find nearest bar
        time_diffs = abs(prices.index - entry_time)
        if time_diffs.min() > timedelta(minutes=10):
            errors += 1
            continue
        nearest_idx = time_diffs.argmin()
        entry_bar_time = prices.index[nearest_idx]
    else:
        entry_bar_time = entry_time

    entry_price = prices.loc[entry_bar_time, 'Open']

    # Find 2-hour exit time (24 bars later)
    try:
        entry_loc = prices.index.get_loc(entry_bar_time)
        exit_2h_loc = entry_loc + TIME_STOP_BARS

        if exit_2h_loc >= len(prices):
            errors += 1
            continue

        exit_2h_price = prices.iloc[exit_2h_loc]['Close']

        # Calculate pips at 2h
        if trade_type == 'BUY':
            pips_2h = (exit_2h_price - entry_price) / pip_val
        else:  # SELL
            pips_2h = (entry_price - exit_2h_price) / pip_val

        net_pips_2h = pips_2h - spread

        results.append({
            'pair': pair,
            'type': trade_type,
            'entry_time': entry_time,
            'bars_held_original': trade['bars_held'],
            'actual_net_pips': trade['net_pips'],
            'pips_at_2h': net_pips_2h,
            'improvement': net_pips_2h - trade['net_pips']
        })

    except Exception as e:
        errors += 1

print(f"\nProcessed: {len(results)} trades")
print(f"Errors: {errors}")

if len(results) > 0:
    results_df = pd.DataFrame(results)

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print(f"\n--- SLOW TRADES (>2h) ---")
    print(f"Trades analyzed: {len(results_df)}")
    print(f"Actual total pips: {results_df['actual_net_pips'].sum():+.1f}")
    print(f"2h exit total pips: {results_df['pips_at_2h'].sum():+.1f}")
    print(f"Improvement: {results_df['improvement'].sum():+.1f}")

    # Win rates
    actual_wins = (results_df['actual_net_pips'] > 0).sum()
    exit_2h_wins = (results_df['pips_at_2h'] > 0).sum()

    print(f"\nActual WR: {actual_wins/len(results_df)*100:.1f}%")
    print(f"2h Exit WR: {exit_2h_wins/len(results_df)*100:.1f}%")

    # Overall comparison
    print("\n" + "=" * 70)
    print("OVERALL COMPARISON")
    print("=" * 70)

    quick_pips = quick_trades['net_pips'].sum()
    slow_actual_pips = results_df['actual_net_pips'].sum()
    slow_2h_pips = results_df['pips_at_2h'].sum()

    print(f"\nQuick trades (≤2h): {len(quick_trades)} trades, {quick_pips:+.0f} pips")
    print(f"Slow trades actual: {len(results_df)} trades, {slow_actual_pips:+.0f} pips")
    print(f"Slow trades @ 2h:   {len(results_df)} trades, {slow_2h_pips:+.0f} pips")

    print(f"\n--- TOTAL ---")
    print(f"Current strategy: {quick_pips + slow_actual_pips:+.0f} pips")
    print(f"With 2h time stop: {quick_pips + slow_2h_pips:+.0f} pips")
    print(f"Improvement: {slow_2h_pips - slow_actual_pips:+.0f} pips")

    # By pair breakdown
    print("\n" + "=" * 70)
    print("BY PAIR BREAKDOWN")
    print("=" * 70)

    pair_summary = results_df.groupby('pair').agg({
        'actual_net_pips': ['count', 'sum'],
        'pips_at_2h': 'sum',
        'improvement': 'sum'
    }).round(1)
    pair_summary.columns = ['trades', 'actual_pips', 'pips_2h', 'improvement']
    pair_summary = pair_summary.sort_values('improvement', ascending=False)

    print(f"\n{'Pair':<10} {'Trades':>6} {'Actual':>10} {'2h Exit':>10} {'Improve':>10}")
    print("-" * 50)
    for pair, row in pair_summary.iterrows():
        print(f"{pair:<10} {row['trades']:>6.0f} {row['actual_pips']:>+10.0f} {row['pips_2h']:>+10.0f} {row['improvement']:>+10.0f}")

    # Save results
    results_df.to_csv(LSTM_DATA_DIR / 'backtest_2h_timestop_results.csv', index=False)
    print(f"\nSaved detailed results to: {LSTM_DATA_DIR / 'backtest_2h_timestop_results.csv'}")

print(f"\nCompleted: {datetime.now()}")
