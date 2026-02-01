"""
Detailed Backtest - V2 M5 PnL Model
====================================
"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import pickle
import joblib
from pathlib import Path
from datetime import datetime

def log(msg=""):
    print(msg, flush=True)

log("=" * 70)
log("DETAILED BACKTEST - V2 M5 PNL MODEL")
log("=" * 70)

DATA_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/lstm_model/v2_both_currencies/data')
MODEL_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/lstm_model/v2_both_currencies/models')

# Load data
with open(DATA_DIR / 'quality_entry_data_m5_v2.pkl', 'rb') as f:
    data = pickle.load(f)

df = data['data'].copy()

# Load model
model = joblib.load(MODEL_DIR / 'quality_xgb_m5_v2_pnl.joblib')
with open(MODEL_DIR / 'quality_xgb_features_m5_v2_pnl.pkl', 'rb') as f:
    feature_cols = pickle.load(f)

# Sort by time, take test set (last 15%)
df = df.sort_values('datetime').reset_index(drop=True)
n = len(df)
test_df = df.iloc[int(n * 0.85):].copy()

log(f"Test period: {test_df['datetime'].min().date()} to {test_df['datetime'].max().date()}")
log(f"Total entries: {len(test_df):,}")

# Fix JPY pips
jpy_mask = test_df['pair'].str.contains('JPY')
test_df.loc[jpy_mask, 'exit_pnl_pips'] = test_df.loc[jpy_mask, 'exit_pnl_pips'] / 100
test_df.loc[jpy_mask, 'max_dd_pips'] = test_df.loc[jpy_mask, 'max_dd_pips'] / 100
test_df.loc[jpy_mask, 'max_profit_pips'] = test_df.loc[jpy_mask, 'max_profit_pips'] / 100

# Predict
test_df['direction_code'] = (test_df['direction'] == 'buy').astype(int)
test_df['trigger_code'] = (test_df['trigger'] == 'base').astype(int)

X_test = test_df[feature_cols].values.astype(np.float32)
X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
test_df['pred_prob'] = model.predict_proba(X_test)[:, 1]

# Overall results
log("\n" + "=" * 70)
log("OVERALL RESULTS")
log("=" * 70)

log(f"\n| Threshold | Trades | Win% | Avg PnL | Total PnL | PF   | Trades/Day |")
log(f"|-----------|--------|------|---------|-----------|------|------------|")

days = (test_df['datetime'].max() - test_df['datetime'].min()).days
for thresh in [0.5, 0.6, 0.7, 0.8, 0.9]:
    f = test_df[test_df['pred_prob'] >= thresh]
    if len(f) == 0:
        continue
    pnl = f['exit_pnl_pips']
    wr = (pnl > 0).mean() * 100
    avg = pnl.mean()
    total = pnl.sum()
    w = pnl[pnl > 0].sum()
    l = abs(pnl[pnl <= 0].sum())
    pf = w / l if l > 0 else float('inf')
    tpd = len(f) / days if days > 0 else 0
    log(f"| {thresh:^9} | {len(f):>6,} | {wr:>4.1f}% | {avg:>7.1f} | {total:>9.0f} | {pf:>4.2f} | {tpd:>10.1f} |")

# Detailed stats at 0.8
log("\n" + "=" * 70)
log("DETAILED STATS (Threshold 0.80)")
log("=" * 70)

filtered = test_df[test_df['pred_prob'] >= 0.80].copy()
pnl = filtered['exit_pnl_pips']
winners = filtered[pnl > 0]
losers = filtered[pnl <= 0]

log(f"\nTrades: {len(filtered):,} over {days} days ({len(filtered)/days:.1f}/day)")
log(f"Win rate: {(pnl > 0).mean()*100:.1f}%")

log(f"\nWinners: {len(winners):,}")
log(f"  Avg win: {winners['exit_pnl_pips'].mean():.1f} pips")
log(f"  Median win: {winners['exit_pnl_pips'].median():.1f} pips")
log(f"  Max win: {winners['exit_pnl_pips'].max():.1f} pips")

log(f"\nLosers: {len(losers):,}")
if len(losers) > 0:
    log(f"  Avg loss: {losers['exit_pnl_pips'].mean():.1f} pips")
    log(f"  Median loss: {losers['exit_pnl_pips'].median():.1f} pips")
    log(f"  Max loss: {losers['exit_pnl_pips'].min():.1f} pips")

log(f"\nRisk/Reward:")
log(f"  Avg PnL: {pnl.mean():.1f} pips")
log(f"  Median PnL: {pnl.median():.1f} pips")
log(f"  Total PnL: {pnl.sum():.0f} pips")
w = pnl[pnl > 0].sum()
l = abs(pnl[pnl <= 0].sum())
log(f"  Profit Factor: {w/l:.2f}" if l > 0 else "  Profit Factor: inf")

# Drawdown
log(f"\nDrawdown (max_dd_pips):")
log(f"  Median DD (all): {filtered['max_dd_pips'].median():.1f} pips")
log(f"  75th pct DD: {filtered['max_dd_pips'].quantile(0.75):.1f} pips")
log(f"  90th pct DD: {filtered['max_dd_pips'].quantile(0.90):.1f} pips")
log(f"  Median DD (winners): {winners['max_dd_pips'].median():.1f} pips")
log(f"  Median DD (losers): {losers['max_dd_pips'].median():.1f} pips" if len(losers) > 0 else "")

# Duration
bars = filtered['bars_to_exit']
log(f"\nTrade Duration:")
log(f"  Median: {bars.median():.0f} bars ({bars.median()*5/60:.1f} hours)")
log(f"  Mean: {bars.mean():.0f} bars ({bars.mean()*5/60:.1f} hours)")
log(f"  75th pct: {bars.quantile(0.75):.0f} bars ({bars.quantile(0.75)*5/60:.1f} hours)")
log(f"  90th pct: {bars.quantile(0.90):.0f} bars ({bars.quantile(0.90)*5/60:.1f} hours)")

# By trigger type
log("\n" + "=" * 70)
log("BY TRIGGER TYPE (Threshold 0.80)")
log("=" * 70)

log(f"\n| Trigger | Trades | Win% | Avg PnL | Total PnL | PF   |")
log(f"|---------|--------|------|---------|-----------|------|")
for trigger in ['base', 'quote']:
    s = filtered[filtered['trigger'] == trigger]
    if len(s) > 0:
        p = s['exit_pnl_pips']
        wr = (p > 0).mean() * 100
        w = p[p > 0].sum()
        l = abs(p[p <= 0].sum())
        pf = w / l if l > 0 else float('inf')
        log(f"| {trigger:^7} | {len(s):>6,} | {wr:>4.1f}% | {p.mean():>7.1f} | {p.sum():>9.0f} | {pf:>4.2f} |")

# By trigger currency
log("\n" + "=" * 70)
log("BY TRIGGER CURRENCY (Threshold 0.80)")
log("=" * 70)

log(f"\n| Currency | Trades | Win% | Avg PnL | Total PnL | PF   |")
log(f"|----------|--------|------|---------|-----------|------|")
for ccy in ['EUR', 'USD', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD', 'NZD']:
    s = filtered[filtered['trigger_ccy'] == ccy]
    if len(s) > 0:
        p = s['exit_pnl_pips']
        wr = (p > 0).mean() * 100
        w = p[p > 0].sum()
        l = abs(p[p <= 0].sum())
        pf = w / l if l > 0 else float('inf')
        log(f"| {ccy:^8} | {len(s):>6,} | {wr:>4.1f}% | {p.mean():>7.1f} | {p.sum():>9.0f} | {pf:>4.2f} |")

# By pair (top 10)
log("\n" + "=" * 70)
log("TOP 10 PAIRS BY TOTAL PNL (Threshold 0.80)")
log("=" * 70)

log(f"\n| Pair     | Trades | Win% | Avg PnL | Total PnL | PF   |")
log(f"|----------|--------|------|---------|-----------|------|")
pair_stats = []
for pair in filtered['pair'].unique():
    s = filtered[filtered['pair'] == pair]
    p = s['exit_pnl_pips']
    w = p[p > 0].sum()
    l = abs(p[p <= 0].sum())
    pair_stats.append({
        'pair': pair, 'trades': len(s), 'wr': (p > 0).mean() * 100,
        'avg': p.mean(), 'total': p.sum(), 'pf': w / l if l > 0 else float('inf')
    })
pair_stats = sorted(pair_stats, key=lambda x: x['total'], reverse=True)
for ps in pair_stats[:10]:
    log(f"| {ps['pair']:^8} | {ps['trades']:>6,} | {ps['wr']:>4.1f}% | {ps['avg']:>7.1f} | {ps['total']:>9.0f} | {ps['pf']:>4.2f} |")

# By direction
log("\n" + "=" * 70)
log("BY DIRECTION (Threshold 0.80)")
log("=" * 70)

log(f"\n| Direction | Trades | Win% | Avg PnL | Total PnL | PF   |")
log(f"|-----------|--------|------|---------|-----------|------|")
for direction in ['buy', 'sell']:
    s = filtered[filtered['direction'] == direction]
    if len(s) > 0:
        p = s['exit_pnl_pips']
        wr = (p > 0).mean() * 100
        w = p[p > 0].sum()
        l = abs(p[p <= 0].sum())
        pf = w / l if l > 0 else float('inf')
        log(f"| {direction:^9} | {len(s):>6,} | {wr:>4.1f}% | {p.mean():>7.1f} | {p.sum():>9.0f} | {pf:>4.2f} |")

log(f"\nCompleted: {datetime.now()}")
