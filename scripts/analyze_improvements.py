"""
Analyze potential improvements to the trading strategy
"""
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from pathlib import Path

LSTM_DATA_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/lstm_model/data')

# Load trades with XGBoost predictions
trades = pd.read_csv(LSTM_DATA_DIR / 'trades_2025_with_predictions.csv')
trades['entry_time'] = pd.to_datetime(trades['entry_time'])

# Filter by XGBoost (prob >= 0.75) - current strategy
xgb = trades[trades['pred_prob'] >= 0.75].copy()

print("=" * 70)
print("POTENTIAL IMPROVEMENTS ANALYSIS")
print("=" * 70)
print(f"\nCurrent XGB-filtered trades: {len(xgb)}, {xgb['net_pips'].sum():+.0f} pips")

# 1. PAIR ANALYSIS
print("\n" + "=" * 70)
print("1. PAIR PERFORMANCE")
print("=" * 70)

pair_stats = xgb.groupby('pair').agg({
    'net_pips': ['count', 'sum', 'mean'],
    'win': 'mean'
}).round(2)
pair_stats.columns = ['trades', 'total_pips', 'avg_pips', 'win_rate']
pair_stats = pair_stats.sort_values('total_pips')

print(f"\n{'Pair':<10} {'Trades':>6} {'Total':>8} {'Avg':>7} {'WR':>6}")
print("-" * 40)
for pair, row in pair_stats.iterrows():
    print(f"{pair:<10} {row['trades']:>6.0f} {row['total_pips']:>+8.0f} {row['avg_pips']:>+7.1f} {row['win_rate']*100:>5.1f}%")

# Worst pairs
worst_pairs = pair_stats[pair_stats['total_pips'] < 0].index.tolist()
print(f"\nWorst pairs (negative total): {worst_pairs}")

# Without worst pairs
xgb_no_worst = xgb[~xgb['pair'].isin(worst_pairs)]
print(f"Without worst pairs: {len(xgb_no_worst)} trades, {xgb_no_worst['net_pips'].sum():+.0f} pips")
print(f"Improvement: {xgb_no_worst['net_pips'].sum() - xgb['net_pips'].sum():+.0f} pips")

# 2. DAY OF WEEK
print("\n" + "=" * 70)
print("2. DAY OF WEEK ANALYSIS")
print("=" * 70)

day_names = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
day_stats = xgb.groupby('dayofweek').agg({
    'net_pips': ['count', 'sum', 'mean'],
    'win': 'mean'
}).round(2)
day_stats.columns = ['trades', 'total_pips', 'avg_pips', 'win_rate']

print(f"\n{'Day':<12} {'Trades':>6} {'Total':>8} {'Avg':>7} {'WR':>6}")
print("-" * 45)
for day, row in day_stats.iterrows():
    print(f"{day_names[day]:<12} {row['trades']:>6.0f} {row['total_pips']:>+8.0f} {row['avg_pips']:>+7.1f} {row['win_rate']*100:>5.1f}%")

# 3. HOUR ANALYSIS
print("\n" + "=" * 70)
print("3. HOUR ANALYSIS")
print("=" * 70)

hour_stats = xgb.groupby('hour').agg({
    'net_pips': ['count', 'sum', 'mean'],
    'win': 'mean'
}).round(2)
hour_stats.columns = ['trades', 'total_pips', 'avg_pips', 'win_rate']

print(f"\n{'Hour':>4} {'Trades':>6} {'Total':>8} {'Avg':>7} {'WR':>6}")
print("-" * 35)
for hour, row in hour_stats.iterrows():
    marker = " ***" if row['total_pips'] < 0 else ""
    print(f"{hour:>4} {row['trades']:>6.0f} {row['total_pips']:>+8.0f} {row['avg_pips']:>+7.1f} {row['win_rate']*100:>5.1f}%{marker}")

bad_hours = hour_stats[hour_stats['total_pips'] < 0].index.tolist()
print(f"\nBad hours: {bad_hours}")

# Without bad hours
xgb_good_hours = xgb[~xgb['hour'].isin(bad_hours)]
print(f"Without bad hours: {len(xgb_good_hours)} trades, {xgb_good_hours['net_pips'].sum():+.0f} pips")
print(f"Improvement: {xgb_good_hours['net_pips'].sum() - xgb['net_pips'].sum():+.0f} pips")

# 4. CONFIDENCE THRESHOLD
print("\n" + "=" * 70)
print("4. HIGHER CONFIDENCE THRESHOLD")
print("=" * 70)

for conf_thresh in [0.75, 0.80, 0.85, 0.90, 0.95]:
    filtered = xgb[xgb['conf_avg'] >= conf_thresh]
    if len(filtered) > 0:
        print(f"conf >= {conf_thresh}: {len(filtered):>5} trades, {filtered['win'].mean()*100:>5.1f}% WR, {filtered['net_pips'].mean():>+6.1f} avg, {filtered['net_pips'].sum():>+8.0f} total")

# 5. XGB PROBABILITY THRESHOLD
print("\n" + "=" * 70)
print("5. HIGHER XGB PROBABILITY")
print("=" * 70)

for prob in [0.75, 0.80, 0.85, 0.90, 0.95]:
    filtered = trades[trades['pred_prob'] >= prob]
    if len(filtered) > 0:
        print(f"prob >= {prob}: {len(filtered):>5} trades, {filtered['win'].mean()*100:>5.1f}% WR, {filtered['net_pips'].mean():>+6.1f} avg, {filtered['net_pips'].sum():>+8.0f} total")

# 6. STOCHASTIC VALUE AT ENTRY
print("\n" + "=" * 70)
print("6. STOCHASTIC EXTREMITY")
print("=" * 70)

# More extreme stochastic = better?
xgb['stoch_extreme'] = np.minimum(xgb['stoch'], 100 - xgb['stoch'])

for thresh in [20, 15, 10, 5]:
    filtered = xgb[xgb['stoch_extreme'] <= thresh]
    if len(filtered) > 0:
        print(f"stoch <= {thresh} or >= {100-thresh}: {len(filtered):>5} trades, {filtered['win'].mean()*100:>5.1f}% WR, {filtered['net_pips'].mean():>+6.1f} avg, {filtered['net_pips'].sum():>+8.0f} total")

# 7. MFC DIFFERENCE
print("\n" + "=" * 70)
print("7. MFC DIFFERENCE (base - quote)")
print("=" * 70)

xgb['mfc_abs_diff'] = abs(xgb['mfc_diff'])
for thresh in [0.5, 0.75, 1.0, 1.25, 1.5]:
    filtered = xgb[xgb['mfc_abs_diff'] >= thresh]
    if len(filtered) > 0:
        print(f"|mfc_diff| >= {thresh}: {len(filtered):>5} trades, {filtered['win'].mean()*100:>5.1f}% WR, {filtered['net_pips'].mean():>+6.1f} avg, {filtered['net_pips'].sum():>+8.0f} total")

# 8. VELOCITY DIFFERENCE
print("\n" + "=" * 70)
print("8. H1 VELOCITY DIFFERENCE")
print("=" * 70)

xgb['vel_abs_diff'] = abs(xgb['vel_h1_diff'])
for thresh in [0.02, 0.04, 0.06, 0.08, 0.10]:
    filtered = xgb[xgb['vel_abs_diff'] >= thresh]
    if len(filtered) > 0:
        print(f"|vel_h1_diff| >= {thresh}: {len(filtered):>5} trades, {filtered['win'].mean()*100:>5.1f}% WR, {filtered['net_pips'].mean():>+6.1f} avg, {filtered['net_pips'].sum():>+8.0f} total")

# 9. COMBINED IMPROVEMENTS
print("\n" + "=" * 70)
print("9. COMBINED IMPROVEMENTS")
print("=" * 70)

# Combine best filters
combined = xgb.copy()

# Remove worst pairs
combined = combined[~combined['pair'].isin(worst_pairs)]
print(f"After removing worst pairs: {len(combined)} trades, {combined['net_pips'].sum():+.0f} pips")

# Remove bad hours
combined = combined[~combined['hour'].isin(bad_hours)]
print(f"After removing bad hours: {len(combined)} trades, {combined['net_pips'].sum():+.0f} pips")

# Higher confidence
combined_conf = combined[combined['conf_avg'] >= 0.85]
print(f"+ conf >= 0.85: {len(combined_conf)} trades, {combined_conf['net_pips'].sum():+.0f} pips")

# Higher XGB prob
combined_prob = combined[combined['pred_prob'] >= 0.85]
print(f"+ prob >= 0.85: {len(combined_prob)} trades, {combined_prob['net_pips'].sum():+.0f} pips")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"\nCurrent: {len(xgb)} trades, {xgb['net_pips'].sum():+.0f} pips")
print(f"Best combined: {len(combined)} trades, {combined['net_pips'].sum():+.0f} pips")
print(f"Improvement: {combined['net_pips'].sum() - xgb['net_pips'].sum():+.0f} pips")
print(f"Trades reduced by: {len(xgb) - len(combined)} ({(len(xgb) - len(combined))/len(xgb)*100:.0f}%)")
