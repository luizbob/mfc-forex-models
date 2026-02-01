"""
Train PnL Model V2 - Rolling Window Walk-Forward (V2 - 3 month validation)
==========================================================================
Rolling window approach: Train on last N years, not all history.
Reduced validation to 3 months to leave 3 months unseen for testing.

Saves to: quality_xgb_m5_v2_pnl_walkforward_rolling_v2.joblib
"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import pickle
import joblib
from pathlib import Path
from datetime import datetime
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

def log(msg=""):
    print(msg, flush=True)

# ============================================================================
# CONFIGURATION
# ============================================================================
TRAIN_YEARS = 4          # Train on last 4 years
VAL_MONTHS = 3           # Use last 3 months for validation (REDUCED from 6)
TEST_MONTHS = 12         # Test on next 12 months (1 year)

log("=" * 70)
log("ROLLING WINDOW WALK-FORWARD TRAINING (V2 - 3 month val)")
log("=" * 70)
log(f"Started: {datetime.now()}")
log(f"\nConfig:")
log(f"  Train window: {TRAIN_YEARS} years")
log(f"  Validation: last {VAL_MONTHS} months of train window")
log(f"  Test: next {TEST_MONTHS} months")

DATA_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/lstm_model/v2_both_currencies/data')
MODEL_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/lstm_model/v2_both_currencies/models')

# Load data
log("\nLoading data...")
with open(DATA_DIR / 'quality_entry_data_m5_v2.pkl', 'rb') as f:
    data = pickle.load(f)

df = data['data'].copy()
log(f"Total entries: {len(df):,}")

# Fix JPY pips
jpy_mask = df['pair'].str.contains('JPY')
df.loc[jpy_mask, 'exit_pnl_pips'] = df.loc[jpy_mask, 'exit_pnl_pips'] / 100
df.loc[jpy_mask, 'max_dd_pips'] = df.loc[jpy_mask, 'max_dd_pips'] / 100

# Create target
df['is_profitable'] = (df['exit_pnl_pips'] > 0).astype(int)

# Sort by datetime
df = df.sort_values('datetime').reset_index(drop=True)
df['datetime'] = pd.to_datetime(df['datetime'])
log(f"Date range: {df['datetime'].min().date()} to {df['datetime'].max().date()}")

# Add time-based features
log("\nAdding time-based features...")
df['hour'] = df['datetime'].dt.hour
df['dayofweek'] = df['datetime'].dt.dayofweek
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 5)
df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 5)

# Features
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

# Add codes
df['direction_code'] = (df['direction'] == 'buy').astype(int)
df['trigger_code'] = (df['trigger'] == 'base').astype(int)

# Model parameters
model_params = {
    'n_estimators': 500,
    'max_depth': 8,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.9,
    'min_child_weight': 5,
    'gamma': 0.1,
    'random_state': 42,
    'eval_metric': 'auc',
    'early_stopping_rounds': 20,
    'n_jobs': -1
}

# ============================================================================
# ROLLING WINDOW WALK-FORWARD
# ============================================================================

log("\n" + "=" * 70)
log("ROLLING WINDOW WALK-FORWARD VALIDATION")
log("=" * 70)

min_date = df['datetime'].min()
max_date = df['datetime'].max()

# Generate test periods
test_starts = pd.date_range(
    start=min_date + pd.DateOffset(years=TRAIN_YEARS),
    end=max_date - pd.DateOffset(months=TEST_MONTHS),
    freq='12MS'
)

log(f"\nRolling window periods:")
for test_start in test_starts:
    train_start = test_start - pd.DateOffset(years=TRAIN_YEARS)
    val_start = test_start - pd.DateOffset(months=VAL_MONTHS)
    test_end = test_start + pd.DateOffset(months=TEST_MONTHS)
    log(f"  Train: {train_start.date()} to {val_start.date()} | Val: {val_start.date()} to {test_start.date()} | Test: {test_start.date()} to {test_end.date()}")

walk_forward_results = []

log("\n" + "-" * 70)

for test_start in test_starts:
    train_start = test_start - pd.DateOffset(years=TRAIN_YEARS)
    val_start = test_start - pd.DateOffset(months=VAL_MONTHS)
    test_end = test_start + pd.DateOffset(months=TEST_MONTHS)

    train_mask = (df['datetime'] >= train_start) & (df['datetime'] < val_start)
    val_mask = (df['datetime'] >= val_start) & (df['datetime'] < test_start)
    test_mask = (df['datetime'] >= test_start) & (df['datetime'] < test_end)

    train_df = df[train_mask].copy()
    val_df = df[val_mask].copy()
    test_df = df[test_mask].copy()

    if len(test_df) == 0 or len(train_df) == 0:
        continue

    X_train = train_df[feature_cols].values.astype(np.float32)
    X_val = val_df[feature_cols].values.astype(np.float32)
    X_test = test_df[feature_cols].values.astype(np.float32)

    y_train = train_df['is_profitable'].values
    y_val = val_df['is_profitable'].values
    y_test = test_df['is_profitable'].values

    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    model = XGBClassifier(**model_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    test_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    test_df['pred'] = model.predict_proba(X_test)[:, 1]

    test_year = test_start.year
    results = {
        'test_year': test_year,
        'train_period': f"{train_start.year}-{val_start.year}",
        'auc': test_auc,
        'n_train': len(train_df),
        'n_val': len(val_df),
        'n_test': len(test_df)
    }

    for thresh in [0.70, 0.75, 0.80]:
        filtered = test_df[test_df['pred'] >= thresh]
        if len(filtered) > 0:
            pnl = filtered['exit_pnl_pips']
            wr = (pnl > 0).mean() * 100
            total_pnl = pnl.sum()
            w = pnl[pnl > 0].sum()
            l = abs(pnl[pnl <= 0].sum())
            pf = w / l if l > 0 else float('inf')
            results[f'n_{thresh}'] = len(filtered)
            results[f'wr_{thresh}'] = wr
            results[f'pnl_{thresh}'] = total_pnl
            results[f'pf_{thresh}'] = pf
        else:
            results[f'n_{thresh}'] = 0
            results[f'wr_{thresh}'] = 0
            results[f'pnl_{thresh}'] = 0
            results[f'pf_{thresh}'] = 0

    walk_forward_results.append(results)

    log(f"\nTest Year {test_year}:")
    log(f"  Train: {train_start.date()} to {val_start.date()} ({len(train_df):,})")
    log(f"  Val:   {val_start.date()} to {test_start.date()} ({len(val_df):,})")
    log(f"  Test:  {test_start.date()} to {test_end.date()} ({len(test_df):,})")
    log(f"  AUC:   {test_auc:.4f}")
    log(f"  @ 0.75: {results.get('n_0.75', 0):,} trades, {results.get('wr_0.75', 0):.1f}% WR, PF {results.get('pf_0.75', 0):.2f}")

# ============================================================================
# SUMMARY
# ============================================================================

log("\n" + "=" * 70)
log("ROLLING WINDOW SUMMARY")
log("=" * 70)

log(f"\n{'Year':<6} {'Train':<12} {'AUC':>6} {'Trades':>8} {'WinRate':>8} {'TotalPnL':>10} {'PF':>6}  (@ 0.75)")
log("-" * 65)

total_trades = 0
total_wins = 0
total_pnl = 0

for r in walk_forward_results:
    year = r['test_year']
    train = r['train_period']
    auc = r['auc']
    n = r.get('n_0.75', 0)
    wr = r.get('wr_0.75', 0)
    pnl = r.get('pnl_0.75', 0)
    pf = r.get('pf_0.75', 0)

    total_trades += n
    total_wins += int(n * wr / 100)
    total_pnl += pnl

    log(f"{year:<6} {train:<12} {auc:>6.4f} {n:>8,} {wr:>7.1f}% {pnl:>+10,.0f} {pf:>6.2f}")

log("-" * 65)
avg_wr = total_wins / total_trades * 100 if total_trades > 0 else 0
log(f"{'TOTAL':<6} {'':<12} {'':>6} {total_trades:>8,} {avg_wr:>7.1f}% {total_pnl:>+10,.0f}")

# ============================================================================
# TRAIN FINAL MODEL (with 3 months unseen for testing)
# ============================================================================

log("\n" + "=" * 70)
log(f"TRAINING FINAL MODEL (last {TRAIN_YEARS} years, {VAL_MONTHS} month val)")
log("=" * 70)

# Train on last TRAIN_YEARS minus last 6 months
# Val on months -6 to -3
# Test (unseen) on last 3 months
final_train_start = max_date - pd.DateOffset(years=TRAIN_YEARS)
final_val_start = max_date - pd.DateOffset(months=6)  # Val starts 6 months ago
final_val_end = max_date - pd.DateOffset(months=3)    # Val ends 3 months ago
# Last 3 months is UNSEEN

final_train_mask = (df['datetime'] >= final_train_start) & (df['datetime'] < final_val_start)
final_val_mask = (df['datetime'] >= final_val_start) & (df['datetime'] < final_val_end)
final_test_mask = df['datetime'] >= final_val_end  # Last 3 months UNSEEN

final_train_df = df[final_train_mask].copy()
final_val_df = df[final_val_mask].copy()
final_test_df = df[final_test_mask].copy()

log(f"\nFinal model data splits:")
log(f"  Train:  {final_train_start.date()} to {final_val_start.date()} ({len(final_train_df):,} samples)")
log(f"  Val:    {final_val_start.date()} to {final_val_end.date()} ({len(final_val_df):,} samples)")
log(f"  UNSEEN: {final_val_end.date()} to {max_date.date()} ({len(final_test_df):,} samples)")

X_train_final = final_train_df[feature_cols].values.astype(np.float32)
X_val_final = final_val_df[feature_cols].values.astype(np.float32)
X_test_final = final_test_df[feature_cols].values.astype(np.float32)

y_train_final = final_train_df['is_profitable'].values
y_val_final = final_val_df['is_profitable'].values
y_test_final = final_test_df['is_profitable'].values

X_train_final = np.nan_to_num(X_train_final, nan=0.0, posinf=0.0, neginf=0.0)
X_val_final = np.nan_to_num(X_val_final, nan=0.0, posinf=0.0, neginf=0.0)
X_test_final = np.nan_to_num(X_test_final, nan=0.0, posinf=0.0, neginf=0.0)

final_model = XGBClassifier(**model_params)
final_model.fit(
    X_train_final, y_train_final,
    eval_set=[(X_val_final, y_val_final)],
    verbose=False
)

# Evaluate on UNSEEN test data (last 3 months)
final_test_df['pred'] = final_model.predict_proba(X_test_final)[:, 1]
test_auc = roc_auc_score(y_test_final, final_test_df['pred'])

log(f"\n*** UNSEEN TEST RESULTS (last 3 months) ***")
log(f"  AUC: {test_auc:.4f}")

log(f"\n{'Thresh':<8} {'Trades':>10} {'WinRate':>10} {'AvgPnL':>10} {'TotalPnL':>12} {'PF':>8}")
log("-" * 70)

for thresh in [0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:
    filtered = final_test_df[final_test_df['pred'] >= thresh]
    if len(filtered) > 0:
        pnl = filtered['exit_pnl_pips']
        wr = (pnl > 0).mean() * 100
        w = pnl[pnl > 0].sum()
        l = abs(pnl[pnl <= 0].sum())
        pf = w / l if l > 0 else float('inf')
        log(f'{thresh:<8} {len(filtered):>10,} {wr:>9.1f}% {pnl.mean():>+9.1f} {pnl.sum():>+12,.0f} {pf:>8.2f}')

# Save model
model_path = MODEL_DIR / 'quality_xgb_m5_v2_pnl_walkforward_rolling_v2.joblib'
features_path = MODEL_DIR / 'quality_xgb_features_m5_v2_pnl_walkforward_rolling_v2.pkl'

joblib.dump(final_model, model_path)
with open(features_path, 'wb') as f:
    pickle.dump(feature_cols, f)

# Save results
results_path = MODEL_DIR / 'walkforward_rolling_results_m5_v2.pkl'
with open(results_path, 'wb') as f:
    pickle.dump(walk_forward_results, f)

log(f"\n" + "=" * 70)
log(f"Model saved to: {model_path}")
log(f"Features saved to: {features_path}")
log(f"Results saved to: {results_path}")
log(f"\nCompleted: {datetime.now()}")
