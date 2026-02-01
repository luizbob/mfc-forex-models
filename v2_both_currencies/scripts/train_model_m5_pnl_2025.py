"""
Train PnL Model V2 - 2025 Forward Test
=======================================
Train on data up to June 2025, validate on recent data.
Tests if model generalizes to most recent market conditions.
Saves to: quality_xgb_m5_v2_pnl_2025.joblib
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

log("=" * 70)
log("TRAINING PNL MODEL V2 - 2025 FORWARD TEST")
log("=" * 70)
log(f"Started: {datetime.now()}")
log("Train: 2013 - June 2025 | Val: Jul-Sep 2025 | Test: Oct-Dec 2025")

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

# Date-based split for 2025 forward test
train_end = pd.Timestamp('2025-06-30')
val_end = pd.Timestamp('2025-09-30')

train_df = df[df['datetime'] <= train_end].copy()
val_df = df[(df['datetime'] > train_end) & (df['datetime'] <= val_end)].copy()
test_df = df[df['datetime'] > val_end].copy()

log(f"\nDate-based split:")
log(f"  Train: {len(train_df):,} ({train_df['datetime'].min().date()} to {train_df['datetime'].max().date()})")
log(f"  Val:   {len(val_df):,} ({val_df['datetime'].min().date()} to {val_df['datetime'].max().date()})")
log(f"  Test:  {len(test_df):,} ({test_df['datetime'].min().date()} to {test_df['datetime'].max().date()})")

log(f"\nSplit percentages:")
total = len(df)
log(f"  Train: {len(train_df)/total*100:.1f}%")
log(f"  Val:   {len(val_df)/total*100:.1f}%")
log(f"  Test:  {len(test_df)/total*100:.1f}%")

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
for split_df in [train_df, val_df, test_df]:
    split_df['direction_code'] = (split_df['direction'] == 'buy').astype(int)
    split_df['trigger_code'] = (split_df['trigger'] == 'base').astype(int)

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

# Train model with same hyperparameters as original
log("\n" + "=" * 70)
log("TRAINING MODEL")
log("=" * 70)

model = XGBClassifier(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.9,
    min_child_weight=5,
    gamma=0.1,
    random_state=42,
    eval_metric='auc',
    early_stopping_rounds=20,
    n_jobs=-1
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

# Evaluate
train_auc = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
val_auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
test_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

log(f"\nAUC Scores:")
log(f"  Train: {train_auc:.4f}")
log(f"  Val:   {val_auc:.4f}")
log(f"  Test:  {test_auc:.4f}")

# Feature importance
log("\nTop 15 Feature Importances:")
importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

for _, row in importance.head(15).iterrows():
    log(f"  {row['feature']:20s}: {row['importance']:.4f}")

# Backtest on test set (Oct-Dec 2025 - truly unseen recent data)
log("\n" + "=" * 70)
log("BACKTEST ON TEST SET (Oct-Dec 2025)")
log("=" * 70)

test_df['pred'] = model.predict_proba(X_test)[:, 1]

log(f"\n{'Thresh':<8} {'Trades':>10} {'WinRate':>10} {'AvgPnL':>10} {'TotalPnL':>12} {'PF':>8}")
log("-" * 70)

for thresh in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:
    filtered = test_df[test_df['pred'] >= thresh]
    if len(filtered) > 0:
        pnl = filtered['exit_pnl_pips']
        wr = (pnl > 0).mean() * 100
        w = pnl[pnl > 0].sum()
        l = abs(pnl[pnl <= 0].sum())
        pf = w / l if l > 0 else float('inf')
        log(f'{thresh:<8} {len(filtered):>10,} {wr:>9.1f}% {pnl.mean():>+9.1f} {pnl.sum():>+12,.0f} {pf:>8.2f}')

# Compare with original model
log("\n" + "=" * 70)
log("COMPARISON WITH ORIGINAL MODEL (on Oct-Dec 2025 data)")
log("=" * 70)

orig_model_path = MODEL_DIR / 'quality_xgb_m5_v2_pnl.joblib'
orig_features_path = MODEL_DIR / 'quality_xgb_features_m5_v2_pnl.pkl'

if orig_model_path.exists():
    orig_model = joblib.load(orig_model_path)
    with open(orig_features_path, 'rb') as f:
        orig_features = pickle.load(f)

    # Get original features for test set
    X_test_orig = test_df[orig_features].values.astype(np.float32)
    X_test_orig = np.nan_to_num(X_test_orig, nan=0.0, posinf=0.0, neginf=0.0)

    test_df['orig_pred'] = orig_model.predict_proba(X_test_orig)[:, 1]

    log(f"\n{'Model':<12} {'Thresh':<8} {'Trades':>8} {'WinRate':>8} {'AvgPnL':>8} {'TotalPnL':>10} {'PF':>6}")
    log("-" * 70)

    for thresh in [0.70, 0.75, 0.80, 0.85]:
        # Original
        f_orig = test_df[test_df['orig_pred'] >= thresh]
        if len(f_orig) > 0:
            pnl = f_orig['exit_pnl_pips']
            wr = (pnl > 0).mean() * 100
            w = pnl[pnl > 0].sum()
            l = abs(pnl[pnl <= 0].sum())
            pf = w / l if l > 0 else float('inf')
            log(f"{'Original':<12} {thresh:<8} {len(f_orig):>8,} {wr:>7.1f}% {pnl.mean():>+7.1f} {pnl.sum():>+10,.0f} {pf:>6.2f}")

        # New 2025
        f_new = test_df[test_df['pred'] >= thresh]
        if len(f_new) > 0:
            pnl = f_new['exit_pnl_pips']
            wr = (pnl > 0).mean() * 100
            w = pnl[pnl > 0].sum()
            l = abs(pnl[pnl <= 0].sum())
            pf = w / l if l > 0 else float('inf')
            log(f"{'2025':<12} {thresh:<8} {len(f_new):>8,} {wr:>7.1f}% {pnl.mean():>+7.1f} {pnl.sum():>+10,.0f} {pf:>6.2f}")

        log("-" * 70)
else:
    log("Original model not found for comparison")

# Analyze by currency trigger
log("\n" + "=" * 70)
log("PERFORMANCE BY TRIGGER CURRENCY (0.75 threshold)")
log("=" * 70)

test_df['trigger_currency'] = test_df.apply(
    lambda x: x['pair'][:3] if x['trigger'] == 'base' else x['pair'][3:6],
    axis=1
)

filtered = test_df[test_df['pred'] >= 0.75]
log(f"\n{'Currency':<10} {'Trades':>8} {'WinRate':>8} {'AvgPnL':>8} {'TotalPnL':>10}")
log("-" * 50)

for curr in ['EUR', 'USD', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD', 'NZD']:
    curr_trades = filtered[filtered['trigger_currency'] == curr]
    if len(curr_trades) > 0:
        pnl = curr_trades['exit_pnl_pips']
        wr = (pnl > 0).mean() * 100
        log(f"{curr:<10} {len(curr_trades):>8,} {wr:>7.1f}% {pnl.mean():>+7.1f} {pnl.sum():>+10,.0f}")

# Save model
model_path = MODEL_DIR / 'quality_xgb_m5_v2_pnl_2025.joblib'
features_path = MODEL_DIR / 'quality_xgb_features_m5_v2_pnl_2025.pkl'

joblib.dump(model, model_path)
with open(features_path, 'wb') as f:
    pickle.dump(feature_cols, f)

log(f"\n" + "=" * 70)
log(f"Model saved to: {model_path}")
log(f"Features saved to: {features_path}")
log(f"\nCompleted: {datetime.now()}")
