"""
Walk-Forward Validation - Momentum Model M15 V4 (box crossing only)
====================================================================
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
log("WALK-FORWARD VALIDATION - MOMENTUM MODEL M15 V4 (box crossing)")
log("=" * 70)

DATA_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/lstm_model/v2_both_currencies/v3_regression/data')

log("\nLoading V4 data...")
with open(DATA_DIR / 'momentum_data_m15_v4.pkl', 'rb') as f:
    data = pickle.load(f)

df = data['data'].copy()
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.sort_values('datetime').reset_index(drop=True)
df['year'] = df['datetime'].dt.year

# Filter to only use data where we have price (up to 2025-12-21)
df = df[df['datetime'] <= '2025-12-21'].copy()

log(f"Total samples: {len(df):,}")
log(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")

# Prepare features
df['hour'] = df['datetime'].dt.hour
df['dayofweek'] = df['datetime'].dt.dayofweek
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 5)
df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 5)
df['direction_code'] = (df['direction'] == 'buy').astype(int)
df['trigger_code'] = (df['trigger'] == 'base').astype(int)

feature_cols = [
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

X = df[feature_cols].values.astype(np.float32)
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
y = df['is_profitable'].values.astype(np.int32)

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
    avg_hold = filtered['bars_held'].mean()
    return {'trades': trades, 'wr': wr, 'pf': pf, 'net': net.sum(), 'avg': net.mean(), 'hold': avg_hold}

test_years = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]

log("\n" + "=" * 70)
log("WALK-FORWARD RESULTS")
log("=" * 70)

results = []

for test_year in test_years:
    train_mask = df['year'] < test_year
    test_mask = df['year'] == test_year

    if train_mask.sum() < 1000 or test_mask.sum() < 100:
        continue

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    df_test = df[test_mask].copy()

    log(f"\n{'='*70}")
    log(f"TEST YEAR: {test_year}")
    log(f"Train: {len(X_train):,} samples | Test: {len(X_test):,} samples")

    model = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        min_child_weight=10, subsample=0.9, colsample_bytree=0.8,
        gamma=0.2, random_state=42, n_jobs=-1, eval_metric='auc'
    )
    model.fit(X_train, y_train, verbose=False)

    df_test['pred_prob'] = model.predict_proba(X_test)[:, 1]

    log(f"\n{'Thresh':<8} {'Trades':>8} {'Win %':>8} {'PF':>8} {'Net':>12} {'Avg':>10}")
    log("-" * 60)

    for thresh in [0.40, 0.45, 0.50, 0.55]:
        r = evaluate(df_test, thresh)
        if r:
            log(f"{thresh:<8} {r['trades']:>8,} {r['wr']:>7.1f}% {r['pf']:>8.2f} {r['net']:>+12,.0f} {r['avg']:>+10.2f}")
            if thresh == 0.45:
                results.append({
                    'year': test_year,
                    'trades': r['trades'],
                    'wr': r['wr'],
                    'pf': r['pf'],
                    'net': r['net'],
                    'avg': r['avg'],
                    'hold': r['hold']
                })

    # By bars held
    filtered = df_test[df_test['pred_prob'] >= 0.45]
    if len(filtered) > 50:
        log(f"\nBy Bars Held (threshold 0.45):")
        log(f"{'Bars':<10} {'Trades':>8} {'Win %':>8} {'Avg':>10}")
        log("-" * 40)
        for low, high, label in [(1, 3, '1-2'), (3, 6, '3-5'), (6, 12, '6-11'), (12, 50, '12+')]:
            subset = filtered[(filtered['bars_held'] >= low) & (filtered['bars_held'] < high)]
            if len(subset) > 10:
                wr = (subset['net_pips'] > 0).mean() * 100
                avg = subset['net_pips'].mean()
                log(f"{label:<10} {len(subset):>8,} {wr:>7.1f}% {avg:>+10.2f}")

# Summary
log("\n" + "=" * 70)
log("SUMMARY (threshold 0.45)")
log("=" * 70)

if results:
    log(f"\n{'Year':<8} {'Trades':>8} {'Win %':>8} {'PF':>8} {'Net Pips':>12} {'Avg':>10}")
    log("-" * 60)

    total_trades = 0
    total_net = 0
    all_wr = []
    all_pf = []

    for r in results:
        log(f"{r['year']:<8} {r['trades']:>8,} {r['wr']:>7.1f}% {r['pf']:>8.2f} {r['net']:>+12,.0f} {r['avg']:>+10.2f}")
        total_trades += r['trades']
        total_net += r['net']
        all_wr.append(r['wr'])
        all_pf.append(r['pf'])

    log("-" * 60)
    log(f"{'TOTAL':<8} {total_trades:>8,} {np.mean(all_wr):>7.1f}% {np.mean(all_pf):>8.2f} {total_net:>+12,.0f} {total_net/total_trades:>+10.2f}")

    log(f"\nConsistency Check:")
    log(f"  Years profitable: {sum(1 for r in results if r['net'] > 0)}/{len(results)}")
    log(f"  Win rate range: {min(all_wr):.1f}% - {max(all_wr):.1f}%")
    log(f"  PF range: {min(all_pf):.2f} - {max(all_pf):.2f}")

log("\nDONE")
