"""
Backtest V2 PnL Model on Unseen Data
=====================================
Compares PnL-based model vs old quality-based model.
Uses REAL exit PnL (price when MFC crosses 0).
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
log("BACKTEST V2 PNL MODEL ON UNSEEN DATA (REALISTIC EXIT)")
log("=" * 70)

DATA_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/lstm_model/v2_both_currencies/data')
MODEL_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/lstm_model/v2_both_currencies/models')

# Load data
with open(DATA_DIR / 'quality_entry_data_m15_v2.pkl', 'rb') as f:
    data = pickle.load(f)

df = data['data'].copy()

# Load PnL-based model (primary)
pnl_model_path = MODEL_DIR / 'quality_xgb_m15_v2_pnl.joblib'
pnl_features_path = MODEL_DIR / 'quality_xgb_features_m15_v2_pnl.pkl'

model = joblib.load(pnl_model_path)
with open(pnl_features_path, 'rb') as f:
    feature_cols = pickle.load(f)

log(f"Loaded PnL model: {pnl_model_path.name}")

# Try loading old quality model for comparison
old_model = None
old_model_path = MODEL_DIR / 'quality_xgb_m15_v2.joblib'
if old_model_path.exists():
    old_model = joblib.load(old_model_path)
    log(f"Loaded old quality model for comparison: {old_model_path.name}")

# Sort by time
df = df.sort_values('datetime').reset_index(drop=True)

# Unseen data after validation period
unseen_start = '2023-12-26'
test_df = df[df['datetime'] >= unseen_start].copy()

log(f"\nUnseen data period: {test_df['datetime'].min().date()} to {test_df['datetime'].max().date()}")
log(f"Total entries: {len(test_df):,}")
log(f"Base quality rate: {test_df['is_quality'].mean()*100:.1f}%")

# Prepare features
test_df['direction_code'] = (test_df['direction'] == 'buy').astype(int)
test_df['trigger_code'] = (test_df['trigger'] == 'base').astype(int)

X_test = test_df[feature_cols].values.astype(np.float32)
X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

# Predict
test_df['pred_prob'] = model.predict_proba(X_test)[:, 1]

# Fix JPY pairs for pips
jpy_mask = test_df['pair'].str.contains('JPY')
test_df.loc[jpy_mask, 'exit_pnl_pips'] = test_df.loc[jpy_mask, 'exit_pnl_pips'] / 100
test_df.loc[jpy_mask, 'max_dd_pips'] = test_df.loc[jpy_mask, 'max_dd_pips'] / 100
test_df.loc[jpy_mask, 'max_profit_pips'] = test_df.loc[jpy_mask, 'max_profit_pips'] / 100

# Overall results
log("\n" + "=" * 70)
log("OVERALL RESULTS (Real Exit PnL)")
log("=" * 70)

log(f"\n| Threshold | Trades | Quality% | Win% | Avg PnL | Total PnL | PF |")
log(f"|-----------|--------|----------|------|---------|-----------|------|")

for thresh in [0.5, 0.6, 0.7, 0.8, 0.9]:
    filtered = test_df[test_df['pred_prob'] >= thresh].copy()
    if len(filtered) == 0:
        continue

    quality_rate = filtered['is_quality'].mean() * 100
    pnl = filtered['exit_pnl_pips']

    win_rate = (pnl > 0).mean() * 100
    avg_pnl = pnl.mean()
    total_pnl = pnl.sum()

    winners = pnl[pnl > 0].sum()
    losers = abs(pnl[pnl <= 0].sum())
    pf = winners / losers if losers > 0 else float('inf')

    log(f"| {thresh:^9} | {len(filtered):>6,} | {quality_rate:>7.1f}% | {win_rate:>4.1f}% | {avg_pnl:>7.1f} | {total_pnl:>9.0f} | {pf:>4.2f} |")

# By trigger type
log("\n" + "=" * 70)
log("BY TRIGGER TYPE (Threshold 0.80)")
log("=" * 70)

filtered = test_df[test_df['pred_prob'] >= 0.80].copy()

log(f"\n| Trigger | Trades | Quality% | Win% | Avg PnL | Total PnL | PF |")
log(f"|---------|--------|----------|------|---------|-----------|------|")

for trigger in ['base', 'quote']:
    subset = filtered[filtered['trigger'] == trigger]
    if len(subset) > 0:
        q_rate = subset['is_quality'].mean() * 100
        pnl = subset['exit_pnl_pips']
        w_rate = (pnl > 0).mean() * 100
        avg_pnl = pnl.mean()
        total_pnl = pnl.sum()
        winners = pnl[pnl > 0].sum()
        losers = abs(pnl[pnl <= 0].sum())
        pf = winners / losers if losers > 0 else float('inf')
        log(f"| {trigger:^7} | {len(subset):>6,} | {q_rate:>7.1f}% | {w_rate:>4.1f}% | {avg_pnl:>7.1f} | {total_pnl:>9.0f} | {pf:>4.2f} |")

# By trigger currency
log("\n" + "=" * 70)
log("BY TRIGGER CURRENCY (Threshold 0.80)")
log("=" * 70)

log(f"\n| Currency | Trades | Quality% | Win% | Avg PnL | Total PnL | PF |")
log(f"|----------|--------|----------|------|---------|-----------|------|")

for ccy in ['EUR', 'USD', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD', 'NZD']:
    subset = filtered[filtered['trigger_ccy'] == ccy]
    if len(subset) > 0:
        q_rate = subset['is_quality'].mean() * 100
        pnl = subset['exit_pnl_pips']
        w_rate = (pnl > 0).mean() * 100
        avg_pnl = pnl.mean()
        total_pnl = pnl.sum()
        winners = pnl[pnl > 0].sum()
        losers = abs(pnl[pnl <= 0].sum())
        pf = winners / losers if losers > 0 else float('inf')
        log(f"| {ccy:^8} | {len(subset):>6,} | {q_rate:>7.1f}% | {w_rate:>4.1f}% | {avg_pnl:>7.1f} | {total_pnl:>9.0f} | {pf:>4.2f} |")

# Trade duration
log("\n" + "=" * 70)
log("TRADE DURATION (Threshold 0.80)")
log("=" * 70)

bars = filtered['bars_to_exit']
log(f"\nMedian bars to exit: {bars.median():.0f} bars ({bars.median() * 15 / 60:.1f} hours)")
log(f"Mean bars to exit: {bars.mean():.0f} bars ({bars.mean() * 15 / 60:.1f} hours)")
log(f"75th percentile: {bars.quantile(0.75):.0f} bars ({bars.quantile(0.75) * 15 / 60:.1f} hours)")
log(f"90th percentile: {bars.quantile(0.90):.0f} bars ({bars.quantile(0.90) * 15 / 60:.1f} hours)")

# Winners vs Losers comparison
log("\n" + "=" * 70)
log("WINNERS vs LOSERS (Threshold 0.80)")
log("=" * 70)

pnl = filtered['exit_pnl_pips']
winners = filtered[pnl > 0]
losers = filtered[pnl <= 0]

log(f"\nWinners: {len(winners):,} trades ({len(winners)/len(filtered)*100:.1f}%)")
log(f"  Avg win: {winners['exit_pnl_pips'].mean():.1f} pips")
log(f"  Median win: {winners['exit_pnl_pips'].median():.1f} pips")

log(f"\nLosers: {len(losers):,} trades ({len(losers)/len(filtered)*100:.1f}%)")
if len(losers) > 0:
    log(f"  Avg loss: {losers['exit_pnl_pips'].mean():.1f} pips")
    log(f"  Median loss: {losers['exit_pnl_pips'].median():.1f} pips")

log(f"\nOverall: {len(filtered):,} trades")
log(f"  Avg PnL: {pnl.mean():.1f} pips")
log(f"  Total PnL: {pnl.sum():.0f} pips")

# V2 comparison
log("\n" + "=" * 70)
log("BASE vs QUOTE TRIGGERS")
log("=" * 70)

log(f"\nBase triggers at 0.80: {len(filtered[filtered['trigger'] == 'base']):,} trades")
log(f"Quote triggers at 0.80: {len(filtered[filtered['trigger'] == 'quote']):,} trades")
log(f"Total V2 at 0.80: {len(filtered):,} trades")

# Compare PnL model vs old quality model
if old_model is not None:
    log("\n" + "=" * 70)
    log("COMPARISON: PNL MODEL vs QUALITY MODEL")
    log("=" * 70)

    old_probs = old_model.predict_proba(X_test)[:, 1]
    test_df['old_pred_prob'] = old_probs

    log(f"\n| Model   | Thresh | Trades | Win% | Avg PnL | Total PnL | PF |")
    log(f"|---------|--------|--------|------|---------|-----------|------|")

    for thresh in [0.5, 0.6, 0.7, 0.8, 0.9]:
        # PnL model
        f_new = test_df[test_df['pred_prob'] >= thresh]
        if len(f_new) > 0:
            pnl_new = f_new['exit_pnl_pips']
            wr_new = (pnl_new > 0).mean() * 100
            avg_new = pnl_new.mean()
            tot_new = pnl_new.sum()
            w = pnl_new[pnl_new > 0].sum()
            l = abs(pnl_new[pnl_new <= 0].sum())
            pf_new = w / l if l > 0 else float('inf')
            log(f"| PnL     | {thresh:>5} | {len(f_new):>6,} | {wr_new:>4.1f}% | {avg_new:>7.1f} | {tot_new:>9.0f} | {pf_new:>4.2f} |")

        # Old quality model
        f_old = test_df[test_df['old_pred_prob'] >= thresh]
        if len(f_old) > 0:
            pnl_old = f_old['exit_pnl_pips']
            wr_old = (pnl_old > 0).mean() * 100
            avg_old = pnl_old.mean()
            tot_old = pnl_old.sum()
            w = pnl_old[pnl_old > 0].sum()
            l = abs(pnl_old[pnl_old <= 0].sum())
            pf_old = w / l if l > 0 else float('inf')
            log(f"| Quality | {thresh:>5} | {len(f_old):>6,} | {wr_old:>4.1f}% | {avg_old:>7.1f} | {tot_old:>9.0f} | {pf_old:>4.2f} |")
        log(f"|---------|--------|--------|------|---------|-----------|------|")
