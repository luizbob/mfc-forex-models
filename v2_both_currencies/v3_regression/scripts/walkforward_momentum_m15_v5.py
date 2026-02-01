"""
Walk-Forward Validation - Momentum Model M15 V5 (with MACD)
============================================================
Compare V3 (no MACD) vs V5 (with MACD) to see if MACD helps.
"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from xgboost import XGBClassifier

def log(msg=""):
    print(msg, flush=True)

log("=" * 70)
log("WALK-FORWARD VALIDATION - V5 (with MACD)")
log("=" * 70)

DATA_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/lstm_model/v2_both_currencies/v3_regression/data')

log("\nLoading V5 data (with MACD)...")
with open(DATA_DIR / 'momentum_data_m15_v5.pkl', 'rb') as f:
    data = pickle.load(f)

df = data['data'].copy()
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.sort_values('datetime').reset_index(drop=True)
df['year'] = df['datetime'].dt.year

df = df[df['datetime'] <= '2025-12-21'].copy()

log(f"Total samples: {len(df):,}")

# Prepare features
df['hour'] = df['datetime'].dt.hour
df['dayofweek'] = df['datetime'].dt.dayofweek
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 5)
df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 5)
df['direction_code'] = (df['direction'] == 'buy').astype(int)
df['trigger_code'] = (df['trigger'] == 'base').astype(int)

# Feature columns WITHOUT MACD (for comparison)
feature_cols_no_macd = [
    'base_m15', 'base_vel_m15', 'base_mom_m15', 'base_acc_m15',
    'quote_m15', 'quote_vel_m15', 'quote_mom_m15', 'quote_acc_m15',
    'base_m30', 'base_vel_m30', 'base_mom_m30',
    'quote_m30', 'quote_vel_m30', 'quote_mom_m30',
    'base_h1', 'base_vel_h1', 'base_mom_h1',
    'quote_h1', 'quote_vel_h1', 'quote_mom_h1',
    'base_h4', 'base_vel_h4', 'base_mom_h4',
    'quote_h4', 'quote_vel_h4', 'quote_mom_h4',
    'divergence', 'vel_divergence',
    'base_in_box', 'quote_in_box',
    'base_dist_box', 'quote_dist_box',
    'h4_base_confirms_up', 'h4_base_confirms_down',
    'h4_quote_confirms_up', 'h4_quote_confirms_down',
    'entry_vel',
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
    'direction_code', 'trigger_code',
]

# Feature columns WITH MACD
feature_cols_with_macd = feature_cols_no_macd + [
    'macd_h1', 'macd_signal_h1', 'macd_hist_h1', 'macd_above_signal_h1', 'macd_above_zero_h1',
    'macd_h4', 'macd_signal_h4', 'macd_hist_h4', 'macd_above_signal_h4', 'macd_above_zero_h4',
    'macd_d1', 'macd_signal_d1', 'macd_hist_d1', 'macd_above_signal_d1', 'macd_above_zero_d1',
]

def evaluate(df_subset, threshold=0.45):
    filtered = df_subset[df_subset['pred_prob'] >= threshold]
    if len(filtered) < 10:
        return None
    net = filtered['net_pips']
    trades = len(filtered)
    wr = (net > 0).mean() * 100
    winners = net[net > 0].sum()
    losers = abs(net[net <= 0].sum())
    pf = winners / losers if losers > 0 else 999
    return {'trades': trades, 'wr': wr, 'pf': pf, 'net': net.sum(), 'avg': net.mean()}

test_years = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]

results_no_macd = []
results_with_macd = []

for test_year in test_years:
    train_mask = df['year'] < test_year
    test_mask = df['year'] == test_year

    if train_mask.sum() < 1000 or test_mask.sum() < 100:
        continue

    df_test = df[test_mask].copy()
    y_train = df[train_mask]['is_profitable'].values.astype(np.int32)

    log(f"\n{'='*70}")
    log(f"TEST YEAR: {test_year}")
    log(f"Train: {train_mask.sum():,} | Test: {test_mask.sum():,}")

    # Model WITHOUT MACD
    X_train_no = df[train_mask][feature_cols_no_macd].values.astype(np.float32)
    X_test_no = df[test_mask][feature_cols_no_macd].values.astype(np.float32)
    X_train_no = np.nan_to_num(X_train_no, nan=0.0, posinf=0.0, neginf=0.0)
    X_test_no = np.nan_to_num(X_test_no, nan=0.0, posinf=0.0, neginf=0.0)

    model_no = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        min_child_weight=10, subsample=0.9, colsample_bytree=0.8,
        gamma=0.2, random_state=42, n_jobs=-1, eval_metric='auc'
    )
    model_no.fit(X_train_no, y_train, verbose=False)
    df_test['pred_prob'] = model_no.predict_proba(X_test_no)[:, 1]
    r_no = evaluate(df_test, 0.45)
    if r_no:
        results_no_macd.append({'year': test_year, **r_no})

    # Model WITH MACD
    X_train_macd = df[train_mask][feature_cols_with_macd].values.astype(np.float32)
    X_test_macd = df[test_mask][feature_cols_with_macd].values.astype(np.float32)
    X_train_macd = np.nan_to_num(X_train_macd, nan=0.0, posinf=0.0, neginf=0.0)
    X_test_macd = np.nan_to_num(X_test_macd, nan=0.0, posinf=0.0, neginf=0.0)

    model_macd = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        min_child_weight=10, subsample=0.9, colsample_bytree=0.8,
        gamma=0.2, random_state=42, n_jobs=-1, eval_metric='auc'
    )
    model_macd.fit(X_train_macd, y_train, verbose=False)
    df_test['pred_prob'] = model_macd.predict_proba(X_test_macd)[:, 1]
    r_macd = evaluate(df_test, 0.45)
    if r_macd:
        results_with_macd.append({'year': test_year, **r_macd})

    log(f"\n  NO MACD:   {r_no['trades']:>5,} trades, {r_no['wr']:>5.1f}% WR, PF {r_no['pf']:.2f}, {r_no['avg']:>+6.2f} avg")
    log(f"  WITH MACD: {r_macd['trades']:>5,} trades, {r_macd['wr']:>5.1f}% WR, PF {r_macd['pf']:.2f}, {r_macd['avg']:>+6.2f} avg")

    diff_pips = r_macd['net'] - r_no['net']
    log(f"  DIFFERENCE: {diff_pips:>+,.0f} pips")

# Summary
log("\n" + "=" * 70)
log("SUMMARY COMPARISON")
log("=" * 70)

log("\nWithout MACD:")
total_no = sum(r['net'] for r in results_no_macd)
trades_no = sum(r['trades'] for r in results_no_macd)
log(f"  Total: {trades_no:,} trades, {total_no:+,.0f} pips, {total_no/trades_no:+.2f} avg")

log("\nWith MACD:")
total_macd = sum(r['net'] for r in results_with_macd)
trades_macd = sum(r['trades'] for r in results_with_macd)
log(f"  Total: {trades_macd:,} trades, {total_macd:+,.0f} pips, {total_macd/trades_macd:+.2f} avg")

log(f"\nDIFFERENCE: {total_macd - total_no:+,.0f} pips")

if total_macd > total_no:
    log("MACD HELPS!")
else:
    log("MACD does NOT help.")

# Feature importance for MACD model (last year)
log("\n" + "=" * 70)
log("TOP 20 FEATURES (WITH MACD, 2025 model)")
log("=" * 70)
importance = model_macd.feature_importances_
indices = np.argsort(importance)[::-1][:20]
for i, idx in enumerate(indices):
    log(f"  {i+1:2}. {feature_cols_with_macd[idx]:<30} {importance[idx]:.4f}")

log("\nDONE")
