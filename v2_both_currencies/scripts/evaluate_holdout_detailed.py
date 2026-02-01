"""
Evaluate ALL Saved Models on TRUE HOLDOUT
==========================================
Loads SAVED models (no retraining) and compares them on holdout data.

Models:
1. Original Tuned (quality_xgb_m5_v2_tuned.joblib)
2. V2 Tuned (quality_xgb_m5_v2_tuned_v2.joblib)
3. V3 Tuned (quality_xgb_m5_v2_tuned_v3.joblib)
4. Walkforward Rolling (quality_xgb_m5_v2_pnl_walkforward_rolling.joblib)

Holdout period: last 3 months (~Sep-Dec 2025)
"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import pickle
import joblib
from pathlib import Path

DATA_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/lstm_model/v2_both_currencies/data')
MODEL_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/lstm_model/v2_both_currencies/models')

# Typical spreads in pips per pair
SPREADS = {
    'EURUSD': 1.5, 'GBPUSD': 2.0, 'USDJPY': 1.5, 'USDCHF': 2.0, 'USDCAD': 2.0,
    'AUDUSD': 1.5, 'NZDUSD': 2.5, 'EURGBP': 2.0, 'EURJPY': 2.0, 'GBPJPY': 3.0,
    'EURCHF': 2.5, 'AUDJPY': 2.5, 'EURAUD': 3.0, 'EURCAD': 3.0, 'EURNZD': 4.0,
    'GBPAUD': 3.5, 'GBPCAD': 3.5, 'GBPCHF': 3.0, 'GBPNZD': 4.5, 'AUDCAD': 2.5,
    'AUDCHF': 2.5, 'AUDNZD': 3.0, 'CADJPY': 2.5, 'CHFJPY': 2.5, 'NZDCAD': 3.5,
    'NZDCHF': 3.5, 'NZDJPY': 3.0, 'CADCHF': 3.0,
}

print("=" * 90)
print("HOLDOUT EVALUATION: All Saved Models Comparison")
print("=" * 90)

# ============================================================================
# LOAD DATA
# ============================================================================

print("\nLoading data...")
with open(DATA_DIR / 'quality_entry_data_m5_v2.pkl', 'rb') as f:
    data = pickle.load(f)

df = data['data'].copy()
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.sort_values('datetime').reset_index(drop=True)

# Fix JPY pips
jpy_mask = df['pair'].str.contains('JPY')
df.loc[jpy_mask, 'exit_pnl_pips'] = df.loc[jpy_mask, 'exit_pnl_pips'] / 100
df.loc[jpy_mask, 'max_dd_pips'] = df.loc[jpy_mask, 'max_dd_pips'] / 100

# Create target
df['is_profitable'] = (df['exit_pnl_pips'] > 0).astype(int)

# Add time-based features
df['hour'] = df['datetime'].dt.hour
df['dayofweek'] = df['datetime'].dt.dayofweek
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 5)
df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 5)

# Add codes
df['direction_code'] = (df['direction'] == 'buy').astype(int)
df['trigger_code'] = (df['trigger'] == 'base').astype(int)

# Features (same as used in training)
feature_cols = [
    'base_m5', 'quote_m5', 'base_vel_m5', 'quote_vel_m5',
    'base_m15', 'quote_m15', 'base_vel_m15', 'quote_vel_m15',
    'base_m30', 'quote_m30', 'base_vel_m30', 'quote_vel_m30',
    'base_h1', 'quote_h1', 'base_vel_h1', 'quote_vel_h1',
    'base_h4', 'quote_h4', 'base_vel_h4', 'quote_vel_h4',
    'base_vel2_m5', 'base_acc_m5',
    'quote_vel2_m5', 'quote_acc_m5',
    'divergence', 'vel_divergence',
    'direction_code', 'trigger_code',
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
]

# Holdout period: last 3 months
max_date = df['datetime'].max()
final_val_end = max_date - pd.DateOffset(months=3)
holdout_mask = df['datetime'] >= final_val_end
holdout_df = df[holdout_mask].copy()

# Add spread cost per trade
holdout_df['spread'] = holdout_df['pair'].map(SPREADS).fillna(2.5)
holdout_df['pnl_after_spread'] = holdout_df['exit_pnl_pips'] - holdout_df['spread']

print(f"Date range: {df['datetime'].min().date()} to {df['datetime'].max().date()}")
print(f"Holdout period: {holdout_df['datetime'].min().date()} to {holdout_df['datetime'].max().date()} ({len(holdout_df):,} samples)")

# Prepare holdout features
X_holdout = holdout_df[feature_cols].values.astype(np.float32)
X_holdout = np.nan_to_num(X_holdout, nan=0.0, posinf=0.0, neginf=0.0)

# ============================================================================
# LOAD SAVED MODELS
# ============================================================================

print("\nLoading saved models...")

models = {}

# 1. Original Tuned
try:
    models['Original'] = {
        'model': joblib.load(MODEL_DIR / 'quality_xgb_m5_v2_tuned.joblib'),
        'desc': 'max_depth=4, LR=0.05'
    }
    print("  [OK] Original Tuned loaded")
except Exception as e:
    print(f"  [X] Original Tuned: {e}")

# 2. V2 Tuned
try:
    models['V2'] = {
        'model': joblib.load(MODEL_DIR / 'quality_xgb_m5_v2_tuned_v2.joblib'),
        'desc': 'max_depth=6, LR=0.1, regularization'
    }
    print("  [OK] V2 Tuned loaded")
except Exception as e:
    print(f"  [X] V2 Tuned: {e}")

# 3. V3 Tuned
try:
    models['V3'] = {
        'model': joblib.load(MODEL_DIR / 'quality_xgb_m5_v2_tuned_v3.joblib'),
        'desc': 'max_depth=8, LR=0.01, complex'
    }
    print("  [OK] V3 Tuned loaded")
except Exception as e:
    print(f"  [X] V3 Tuned: {e}")

# 4. Walkforward Rolling
try:
    models['WF_Roll'] = {
        'model': joblib.load(MODEL_DIR / 'quality_xgb_m5_v2_pnl_walkforward_rolling.joblib'),
        'desc': 'walkforward rolling training'
    }
    print("  [OK] Walkforward Rolling loaded")
except Exception as e:
    print(f"  [X] Walkforward Rolling: {e}")

# ============================================================================
# GENERATE PREDICTIONS
# ============================================================================

print("\nGenerating predictions on holdout...")

for name, m in models.items():
    try:
        holdout_df[f'pred_{name}'] = m['model'].predict_proba(X_holdout)[:, 1]
        print(f"  [OK] {name} predictions generated")
    except Exception as e:
        print(f"  [X] {name}: {e}")
        holdout_df[f'pred_{name}'] = 0.5  # Default

# ============================================================================
# EVALUATION FUNCTION
# ============================================================================

def evaluate_at_threshold_with_spread(df, pred_col, threshold):
    filtered = df[df[pred_col] >= threshold]
    if len(filtered) == 0:
        return {'trades': 0, 'wr': 0, 'wr_net': 0, 'pnl': 0, 'pnl_net': 0, 'pf': 0, 'pf_net': 0,
                'winners': 0, 'losers': 0, 'spread_cost': 0}

    pnl = filtered['exit_pnl_pips']
    pnl_net = filtered['pnl_after_spread']
    spread_cost = filtered['spread'].sum()

    trades = len(filtered)
    wr = (pnl > 0).mean() * 100
    wr_net = (pnl_net > 0).mean() * 100

    total_pnl = pnl.sum()
    total_pnl_net = pnl_net.sum()

    winners = pnl[pnl > 0].sum()
    losers = abs(pnl[pnl <= 0].sum())
    pf = winners / losers if losers > 0 else 100.0

    winners_net = pnl_net[pnl_net > 0].sum()
    losers_net = abs(pnl_net[pnl_net <= 0].sum())
    pf_net = winners_net / losers_net if losers_net > 0 else 100.0

    return {
        'trades': trades,
        'wr': wr,
        'wr_net': wr_net,
        'pnl': total_pnl,
        'pnl_net': total_pnl_net,
        'pf': pf,
        'pf_net': pf_net,
        'winners': winners,
        'losers': losers,
        'spread_cost': spread_cost,
    }

# ============================================================================
# DETAILED RESULTS FOR EACH MODEL
# ============================================================================

detailed_thresholds = [0.70, 0.71, 0.72, 0.73, 0.74, 0.75]

for name, m in models.items():
    pred_col = f'pred_{name}'
    print("\n" + "=" * 95)
    print(f"{name.upper()} - Holdout Performance ({m['desc']})")
    print("=" * 95)
    print(f"{'Thresh':<7} {'Trades':>8} {'WR%':>7} {'WR%Net':>7} {'PF':>8} {'PFNet':>8} {'PnL':>11} {'PnLNet':>11} {'Spread':>8}")
    print("-" * 95)

    for thresh in detailed_thresholds:
        r = evaluate_at_threshold_with_spread(holdout_df, pred_col, thresh)
        print(f"{thresh:<7} {r['trades']:>8,} {r['wr']:>6.1f}% {r['wr_net']:>6.1f}% {r['pf']:>8.2f} {r['pf_net']:>8.2f} {r['pnl']:>+11,.0f} {r['pnl_net']:>+11,.0f} {r['spread_cost']:>8,.0f}")

# ============================================================================
# SIDE-BY-SIDE COMPARISON @ 0.75 THRESHOLD
# ============================================================================

print("\n" + "=" * 95)
print("SIDE-BY-SIDE COMPARISON @ 0.75 THRESHOLD")
print("=" * 95)

results_75 = {}
for name in models.keys():
    results_75[name] = evaluate_at_threshold_with_spread(holdout_df, f'pred_{name}', 0.75)

# Header
header = f"{'Metric':<20}"
for name in models.keys():
    header += f" {name:>12}"
print(f"\n{header}")
print("-" * (20 + 13 * len(models)))

# Metrics
metrics = [
    ('Trades', 'trades', ','),
    ('Win Rate', 'wr', '.1f%'),
    ('Win Rate (Net)', 'wr_net', '.1f%'),
    ('Profit Factor', 'pf', '.2f'),
    ('PF (Net)', 'pf_net', '.2f'),
    ('PnL', 'pnl', '+,.0f'),
    ('PnL (Net)', 'pnl_net', '+,.0f'),
    ('Spread Cost', 'spread_cost', ',.0f'),
]

for label, key, fmt in metrics:
    row = f"{label:<20}"
    for name in models.keys():
        val = results_75[name][key]
        if fmt == ',':
            row += f" {val:>12,}"
        elif fmt == '.1f%':
            row += f" {val:>11.1f}%"
        elif fmt == '.2f':
            row += f" {val:>12.2f}"
        elif fmt == '+,.0f':
            row += f" {val:>+12,.0f}"
        elif fmt == ',.0f':
            row += f" {val:>12,.0f}"
    print(row)

# PnL per trade
row = f"{'PnL/Trade':<20}"
for name in models.keys():
    r = results_75[name]
    ppt = r['pnl'] / r['trades'] if r['trades'] > 0 else 0
    row += f" {ppt:>12.2f}"
print(row)

row = f"{'PnL/Trade (Net)':<20}"
for name in models.keys():
    r = results_75[name]
    ppt = r['pnl_net'] / r['trades'] if r['trades'] > 0 else 0
    row += f" {ppt:>12.2f}"
print(row)

# ============================================================================
# SIDE-BY-SIDE COMPARISON @ 0.70 THRESHOLD
# ============================================================================

print("\n" + "=" * 95)
print("SIDE-BY-SIDE COMPARISON @ 0.70 THRESHOLD")
print("=" * 95)

results_70 = {}
for name in models.keys():
    results_70[name] = evaluate_at_threshold_with_spread(holdout_df, f'pred_{name}', 0.70)

print(f"\n{header}")
print("-" * (20 + 13 * len(models)))

for label, key, fmt in metrics:
    row = f"{label:<20}"
    for name in models.keys():
        val = results_70[name][key]
        if fmt == ',':
            row += f" {val:>12,}"
        elif fmt == '.1f%':
            row += f" {val:>11.1f}%"
        elif fmt == '.2f':
            row += f" {val:>12.2f}"
        elif fmt == '+,.0f':
            row += f" {val:>+12,.0f}"
        elif fmt == ',.0f':
            row += f" {val:>12,.0f}"
    print(row)

# ============================================================================
# CONCLUSION
# ============================================================================

print("\n" + "=" * 95)
print("CONCLUSION @ 0.75 THRESHOLD")
print("=" * 95)

# Find best model by different metrics
best_pf = max(results_75.items(), key=lambda x: x[1]['pf'])
best_pf_net = max(results_75.items(), key=lambda x: x[1]['pf_net'])
best_wr = max(results_75.items(), key=lambda x: x[1]['wr'])
best_wr_net = max(results_75.items(), key=lambda x: x[1]['wr_net'])
best_pnl = max(results_75.items(), key=lambda x: x[1]['pnl'])
best_pnl_net = max(results_75.items(), key=lambda x: x[1]['pnl_net'])
best_trades = max(results_75.items(), key=lambda x: x[1]['trades'])

print(f"\n  Best Profit Factor:     {best_pf[0]} (PF={best_pf[1]['pf']:.2f})")
print(f"  Best PF (Net):          {best_pf_net[0]} (PF Net={best_pf_net[1]['pf_net']:.2f})")
print(f"  Best Win Rate:          {best_wr[0]} (WR={best_wr[1]['wr']:.1f}%)")
print(f"  Best Win Rate (Net):    {best_wr_net[0]} (WR Net={best_wr_net[1]['wr_net']:.1f}%)")
print(f"  Best Total PnL:         {best_pnl[0]} (PnL={best_pnl[1]['pnl']:+,.0f})")
print(f"  Best Total PnL (Net):   {best_pnl_net[0]} (PnL Net={best_pnl_net[1]['pnl_net']:+,.0f})")
print(f"  Most Trades:            {best_trades[0]} (Trades={best_trades[1]['trades']:,})")

# Ranking summary
print("\n" + "-" * 60)
print("RANKING BY NET PF @ 0.75:")
ranked = sorted(results_75.items(), key=lambda x: x[1]['pf_net'], reverse=True)
for i, (name, r) in enumerate(ranked, 1):
    print(f"  {i}. {name:<12} PF Net={r['pf_net']:.2f}, WR Net={r['wr_net']:.1f}%, Trades={r['trades']:,}, PnL Net={r['pnl_net']:+,.0f}")

print("\n" + "-" * 60)
print("RANKING BY NET PnL @ 0.75:")
ranked = sorted(results_75.items(), key=lambda x: x[1]['pnl_net'], reverse=True)
for i, (name, r) in enumerate(ranked, 1):
    print(f"  {i}. {name:<12} PnL Net={r['pnl_net']:+,.0f}, PF Net={r['pf_net']:.2f}, Trades={r['trades']:,}")

print()
