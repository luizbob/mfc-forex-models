"""
Walk-Forward Inversion Test
============================
Test if inverting low probability trades works across all years.
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

DATA_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/lstm_model/v2_both_currencies/v3_regression/data')

SPREADS = {
    'AUDUSD': 0.9, 'EURUSD': 0.8, 'GBPUSD': 1.0, 'NZDUSD': 1.8,
    'USDCAD': 1.5, 'USDCHF': 1.3, 'USDJPY': 1.0,
    'AUDCAD': 2.2, 'AUDCHF': 0.9, 'AUDJPY': 1.9, 'AUDNZD': 2.0,
    'CADCHF': 0.8, 'CADJPY': 3.8, 'CHFJPY': 2.4,
    'EURAUD': 3.4, 'EURCAD': 2.9, 'EURCHF': 2.5, 'EURGBP': 1.4,
    'EURJPY': 2.4, 'EURNZD': 5.4,
    'GBPAUD': 2.5, 'GBPCAD': 4.8, 'GBPCHF': 2.4, 'GBPJPY': 2.2, 'GBPNZD': 5.8,
    'NZDCAD': 2.1, 'NZDCHF': 1.5, 'NZDJPY': 4.3,
}

log("=" * 70)
log("WALK-FORWARD INVERSION TEST")
log("=" * 70)

log("\nLoading data...")
with open(DATA_DIR / 'momentum_data_m15_v3.pkl', 'rb') as f:
    data = pickle.load(f)

df = data['data'].copy()
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.sort_values('datetime').reset_index(drop=True)
df['year'] = df['datetime'].dt.year

# Filter to only use data where we have price (up to 2025-12-21)
df = df[df['datetime'] <= '2025-12-21'].copy()
log(f"Total samples (with price data): {len(df):,}")
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

# Add spread and inverted pips columns
df['spread'] = df['pair'].map(SPREADS).fillna(2.0)
df['inverted_pips'] = -df['adjusted_pips'] - df['spread']

def evaluate(subset, pips_col='net_pips'):
    if len(subset) < 10:
        return None
    net = subset[pips_col]
    trades = len(net)
    wr = (net > 0).mean() * 100
    winners = net[net > 0].sum()
    losers = abs(net[net <= 0].sum())
    pf = winners / losers if losers > 0 else 999
    return {'trades': trades, 'wr': wr, 'pf': pf, 'net': net.sum(), 'avg': net.mean()}

# Walk-forward test years
test_years = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]

log("\n" + "=" * 70)
log("RESULTS BY YEAR")
log("=" * 70)

results_high = []
results_inverted = []

for test_year in test_years:
    train_mask = df['year'] < test_year
    test_mask = df['year'] == test_year

    if train_mask.sum() < 1000 or test_mask.sum() < 100:
        continue

    X_train, y_train = X[train_mask], y[train_mask]
    X_test = X[test_mask]
    df_test = df[test_mask].copy()

    # Train model
    model = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        min_child_weight=10, subsample=0.9, colsample_bytree=0.8,
        gamma=0.2, random_state=42, n_jobs=-1, eval_metric='auc'
    )
    model.fit(X_train, y_train, verbose=False)

    # Predict
    df_test['pred_prob'] = model.predict_proba(X_test)[:, 1]

    log(f"\n{'='*70}")
    log(f"TEST YEAR: {test_year}")
    log(f"Train: {len(X_train):,} | Test: {len(df_test):,}")

    # High probability - trade as signaled
    log(f"\n  HIGH PROBABILITY (trade as signaled):")
    log(f"  {'Thresh':<8} {'Trades':>8} {'Win %':>8} {'PF':>8} {'Avg':>10}")
    log(f"  " + "-" * 50)
    for thresh in [0.45, 0.50, 0.55]:
        subset = df_test[df_test['pred_prob'] >= thresh]
        r = evaluate(subset, 'net_pips')
        if r:
            log(f"  {'>=' + str(thresh):<8} {r['trades']:>8,} {r['wr']:>7.1f}% {r['pf']:>8.2f} {r['avg']:>+10.2f}")
            if thresh == 0.45:
                results_high.append({'year': test_year, **r})

    # Low probability - trade INVERTED
    log(f"\n  LOW PROBABILITY (trade INVERTED):")
    log(f"  {'Thresh':<8} {'Trades':>8} {'Win %':>8} {'PF':>8} {'Avg':>10}")
    log(f"  " + "-" * 50)
    for thresh in [0.25, 0.20, 0.15]:
        subset = df_test[df_test['pred_prob'] <= thresh]
        r = evaluate(subset, 'inverted_pips')
        if r:
            log(f"  {'<=' + str(thresh):<8} {r['trades']:>8,} {r['wr']:>7.1f}% {r['pf']:>8.2f} {r['avg']:>+10.2f}")
            if thresh == 0.25:
                results_inverted.append({'year': test_year, **r})

# Summary
log("\n" + "=" * 70)
log("SUMMARY COMPARISON")
log("=" * 70)

log("\nHIGH PROBABILITY (>=0.45) - Trade as signaled:")
log(f"{'Year':<8} {'Trades':>8} {'Win %':>8} {'PF':>8} {'Avg':>10} {'Net':>12}")
log("-" * 55)
total_trades_h, total_net_h = 0, 0
for r in results_high:
    log(f"{r['year']:<8} {r['trades']:>8,} {r['wr']:>7.1f}% {r['pf']:>8.2f} {r['avg']:>+10.2f} {r['net']:>+12,.0f}")
    total_trades_h += r['trades']
    total_net_h += r['net']
log("-" * 55)
log(f"{'TOTAL':<8} {total_trades_h:>8,} {'-':>8} {'-':>8} {total_net_h/total_trades_h:>+10.2f} {total_net_h:>+12,.0f}")

log("\nLOW PROBABILITY (<=0.25) - Trade INVERTED:")
log(f"{'Year':<8} {'Trades':>8} {'Win %':>8} {'PF':>8} {'Avg':>10} {'Net':>12}")
log("-" * 55)
total_trades_i, total_net_i = 0, 0
for r in results_inverted:
    log(f"{r['year']:<8} {r['trades']:>8,} {r['wr']:>7.1f}% {r['pf']:>8.2f} {r['avg']:>+10.2f} {r['net']:>+12,.0f}")
    total_trades_i += r['trades']
    total_net_i += r['net']
log("-" * 55)
log(f"{'TOTAL':<8} {total_trades_i:>8,} {'-':>8} {'-':>8} {total_net_i/total_trades_i:>+10.2f} {total_net_i:>+12,.0f}")

log("\n" + "=" * 70)
log("CONCLUSION")
log("=" * 70)
log(f"\nHigh prob strategy: {total_trades_h:,} trades, {total_net_h:+,.0f} pips, {total_net_h/total_trades_h:+.2f} avg")
log(f"Inverted strategy:  {total_trades_i:,} trades, {total_net_i:+,.0f} pips, {total_net_i/total_trades_i:+.2f} avg")

if total_net_i > 0:
    log("\nInversion strategy IS profitable!")
else:
    log("\nInversion strategy is NOT profitable.")

log("\nDONE")
