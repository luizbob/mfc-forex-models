"""
Train PnL Model V2 - Walk-Forward Validation
=============================================
Proper time-series validation: always train on past, test on future.
Shows model stability across different market regimes/years.
Saves to: quality_xgb_m5_v2_pnl_walkforward.joblib
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
from sklearn.model_selection import TimeSeriesSplit

def log(msg=""):
    print(msg, flush=True)

log("=" * 70)
log("WALK-FORWARD TRAINING - PNL MODEL V2")
log("=" * 70)
log(f"Started: {datetime.now()}")

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
df['year'] = df['datetime'].dt.year
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

# Model parameters (fixed - no CV search to avoid temporal leakage)
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
# WALK-FORWARD VALIDATION
# ============================================================================

log("\n" + "=" * 70)
log("WALK-FORWARD VALIDATION")
log("=" * 70)
log("\nStrategy: Train on all years up to Y, test on year Y+1")
log("This simulates how the model would perform if retrained annually.\n")

# Define walk-forward periods
# Train on expanding window, test on next year
walk_forward_results = []

years = sorted(df['year'].unique())
log(f"Years in data: {years}")

# We need at least 3 years of training data, test on subsequent years
min_train_years = 3
test_years = [y for y in years if y >= years[0] + min_train_years]

log(f"\nWalk-forward periods:")
for test_year in test_years:
    train_years = [y for y in years if y < test_year]
    log(f"  Train: {train_years[0]}-{train_years[-1]} → Test: {test_year}")

log("\n" + "-" * 70)

for test_year in test_years:
    # Training data: all years before test_year
    train_mask = df['year'] < test_year
    # Use last 20% of training as validation for early stopping
    train_df_full = df[train_mask].copy()
    val_split = int(len(train_df_full) * 0.85)
    train_df = train_df_full.iloc[:val_split]
    val_df = train_df_full.iloc[val_split:]

    # Test data: the test year
    test_df = df[df['year'] == test_year].copy()

    if len(test_df) == 0:
        continue

    # Prepare features
    X_train = train_df[feature_cols].values.astype(np.float32)
    X_val = val_df[feature_cols].values.astype(np.float32)
    X_test = test_df[feature_cols].values.astype(np.float32)

    y_train = train_df['is_profitable'].values
    y_val = val_df['is_profitable'].values
    y_test = test_df['is_profitable'].values

    # Handle NaN/Inf
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    # Train model
    model = XGBClassifier(**model_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    # Evaluate
    test_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    test_df['pred'] = model.predict_proba(X_test)[:, 1]

    # Calculate metrics at different thresholds
    results = {'year': test_year, 'auc': test_auc, 'n_total': len(test_df)}

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

    log(f"\nYear {test_year}:")
    log(f"  Train: {years[0]}-{test_year-1} ({len(train_df):,} samples)")
    log(f"  Test:  {test_year} ({len(test_df):,} samples)")
    log(f"  AUC:   {test_auc:.4f}")
    log(f"  @ 0.75: {results.get('n_0.75', 0):,} trades, {results.get('wr_0.75', 0):.1f}% WR, PF {results.get('pf_0.75', 0):.2f}")

# ============================================================================
# SUMMARY TABLE
# ============================================================================

log("\n" + "=" * 70)
log("WALK-FORWARD SUMMARY")
log("=" * 70)

log(f"\n{'Year':<6} {'AUC':>6} {'Trades':>8} {'WinRate':>8} {'TotalPnL':>10} {'PF':>6}  (@ 0.75 threshold)")
log("-" * 55)

total_trades = 0
total_wins = 0
total_pnl = 0

for r in walk_forward_results:
    year = r['year']
    auc = r['auc']
    n = r.get('n_0.75', 0)
    wr = r.get('wr_0.75', 0)
    pnl = r.get('pnl_0.75', 0)
    pf = r.get('pf_0.75', 0)

    total_trades += n
    total_wins += int(n * wr / 100)
    total_pnl += pnl

    log(f"{year:<6} {auc:>6.4f} {n:>8,} {wr:>7.1f}% {pnl:>+10,.0f} {pf:>6.2f}")

log("-" * 55)
avg_wr = total_wins / total_trades * 100 if total_trades > 0 else 0
log(f"{'TOTAL':<6} {'':>6} {total_trades:>8,} {avg_wr:>7.1f}% {total_pnl:>+10,.0f}")

# Also show 0.80 threshold summary
log(f"\n{'Year':<6} {'AUC':>6} {'Trades':>8} {'WinRate':>8} {'TotalPnL':>10} {'PF':>6}  (@ 0.80 threshold)")
log("-" * 55)

total_trades_80 = 0
total_wins_80 = 0
total_pnl_80 = 0

for r in walk_forward_results:
    year = r['year']
    auc = r['auc']
    n = r.get('n_0.8', 0)
    wr = r.get('wr_0.8', 0)
    pnl = r.get('pnl_0.8', 0)
    pf = r.get('pf_0.8', 0)

    total_trades_80 += n
    total_wins_80 += int(n * wr / 100)
    total_pnl_80 += pnl

    log(f"{year:<6} {auc:>6.4f} {n:>8,} {wr:>7.1f}% {pnl:>+10,.0f} {pf:>6.2f}")

log("-" * 55)
avg_wr_80 = total_wins_80 / total_trades_80 * 100 if total_trades_80 > 0 else 0
log(f"{'TOTAL':<6} {'':>6} {total_trades_80:>8,} {avg_wr_80:>7.1f}% {total_pnl_80:>+10,.0f}")

# ============================================================================
# TRAIN FINAL MODEL ON ALL DATA
# ============================================================================

log("\n" + "=" * 70)
log("TRAINING FINAL MODEL ON ALL DATA (2013-2024)")
log("=" * 70)

# Train on 2013-2024, keep 2025 as final holdout
final_train_mask = df['year'] <= 2024
final_train_df = df[final_train_mask].copy()
final_test_df = df[df['year'] == 2025].copy()

# Split training for early stopping
val_split = int(len(final_train_df) * 0.9)
train_split_df = final_train_df.iloc[:val_split]
val_split_df = final_train_df.iloc[val_split:]

X_train_final = train_split_df[feature_cols].values.astype(np.float32)
X_val_final = val_split_df[feature_cols].values.astype(np.float32)
X_test_final = final_test_df[feature_cols].values.astype(np.float32)

y_train_final = train_split_df['is_profitable'].values
y_val_final = val_split_df['is_profitable'].values
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

log(f"\nTrained on: 2013-2024 ({len(final_train_df):,} samples)")
log(f"Holdout test: 2025 ({len(final_test_df):,} samples)")

# Evaluate on 2025 holdout
final_test_df['pred'] = final_model.predict_proba(X_test_final)[:, 1]
test_auc = roc_auc_score(y_test_final, final_test_df['pred'])

log(f"\n2025 Holdout Results:")
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

# ============================================================================
# CONSISTENCY CHECK
# ============================================================================

log("\n" + "=" * 70)
log("CONSISTENCY CHECK")
log("=" * 70)

# Check if model is profitable every year
profitable_years_75 = sum(1 for r in walk_forward_results if r.get('pnl_0.75', 0) > 0)
profitable_years_80 = sum(1 for r in walk_forward_results if r.get('pnl_0.8', 0) > 0)
total_years = len(walk_forward_results)

log(f"\n@ 0.75 threshold:")
log(f"  Profitable years: {profitable_years_75}/{total_years}")
log(f"  Win rate range: {min(r.get('wr_0.75', 0) for r in walk_forward_results):.1f}% - {max(r.get('wr_0.75', 0) for r in walk_forward_results):.1f}%")

log(f"\n@ 0.80 threshold:")
log(f"  Profitable years: {profitable_years_80}/{total_years}")
log(f"  Win rate range: {min(r.get('wr_0.8', 0) for r in walk_forward_results):.1f}% - {max(r.get('wr_0.8', 0) for r in walk_forward_results):.1f}%")

if profitable_years_75 == total_years:
    log("\n✓ Model is profitable EVERY year at 0.75 threshold - strong consistency!")
elif profitable_years_75 >= total_years * 0.8:
    log(f"\n~ Model is profitable in {profitable_years_75}/{total_years} years - good consistency")
else:
    log(f"\n! Model has {total_years - profitable_years_75} losing years - may have regime issues")

# Save model
model_path = MODEL_DIR / 'quality_xgb_m5_v2_pnl_walkforward.joblib'
features_path = MODEL_DIR / 'quality_xgb_features_m5_v2_pnl_walkforward.pkl'

joblib.dump(final_model, model_path)
with open(features_path, 'wb') as f:
    pickle.dump(feature_cols, f)

# Save walk-forward results for analysis
results_path = MODEL_DIR / 'walkforward_results_m5.pkl'
with open(results_path, 'wb') as f:
    pickle.dump(walk_forward_results, f)

log(f"\n" + "=" * 70)
log(f"Model saved to: {model_path}")
log(f"Features saved to: {features_path}")
log(f"Walk-forward results saved to: {results_path}")
log(f"\nCompleted: {datetime.now()}")
