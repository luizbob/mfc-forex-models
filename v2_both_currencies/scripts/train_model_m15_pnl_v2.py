"""
Train PnL Model M15 V2 - Fixed Version
======================================
Saves to NEW files: quality_xgb_m15_v2_pnl_fixed.joblib
Does NOT overwrite existing models.
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
log("TRAINING PNL MODEL M15 V2 - FIXED")
log("=" * 70)
log(f"Started: {datetime.now()}")
log("NOTE: Saves to NEW files, does not overwrite existing!")

DATA_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/lstm_model/v2_both_currencies/data')
MODEL_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/lstm_model/v2_both_currencies/models')

# Load data
log("\nLoading data...")
with open(DATA_DIR / 'quality_entry_data_m15_v2.pkl', 'rb') as f:
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
log(f"Date range: {df['datetime'].min().date()} to {df['datetime'].max().date()}")

# Add time-based features
log("\nAdding time-based features...")
df['hour'] = pd.to_datetime(df['datetime']).dt.hour
df['dayofweek'] = pd.to_datetime(df['datetime']).dt.dayofweek
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 5)
df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 5)

# Time-based split: 70/15/15
n = len(df)
train_idx = int(n * 0.70)
val_idx = int(n * 0.85)

train_df = df.iloc[:train_idx].copy()
val_df = df.iloc[train_idx:val_idx].copy()
test_df = df.iloc[val_idx:].copy()

log(f"\nTime-based split:")
log(f"  Train: {len(train_df):,} ({train_df['datetime'].min().date()} to {train_df['datetime'].max().date()})")
log(f"  Val:   {len(val_df):,} ({val_df['datetime'].min().date()} to {val_df['datetime'].max().date()})")
log(f"  Test:  {len(test_df):,} ({test_df['datetime'].min().date()} to {test_df['datetime'].max().date()})")

# Features
feature_cols = [
    'base_m5', 'quote_m5', 'base_vel_m5', 'quote_vel_m5',
    'base_m15', 'quote_m15', 'base_vel_m15', 'quote_vel_m15',
    'base_m30', 'quote_m30', 'base_vel_m30', 'quote_vel_m30',
    'base_h1', 'quote_h1', 'base_vel_h1', 'quote_vel_h1',
    'base_h4', 'quote_h4', 'base_vel_h4', 'quote_vel_h4',
    'base_vel2_m15', 'base_acc_m15',
    'quote_vel2_m15', 'quote_acc_m15',
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

# Train model with good hyperparameters
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

# Backtest on test set
log("\n" + "=" * 70)
log("BACKTEST ON TEST SET")
log("=" * 70)

test_df['pred'] = model.predict_proba(X_test)[:, 1]

log(f"\n{'Thresh':<8} {'Trades':>10} {'WinRate':>10} {'AvgPnL':>10} {'TotalPnL':>12} {'PF':>8}")
log("-" * 70)

for thresh in [0.65, 0.70, 0.75, 0.80, 0.85]:
    filtered = test_df[test_df['pred'] >= thresh]
    if len(filtered) > 0:
        pnl = filtered['exit_pnl_pips']
        wr = (pnl > 0).mean() * 100
        w = pnl[pnl > 0].sum()
        l = abs(pnl[pnl <= 0].sum())
        pf = w / l if l > 0 else float('inf')
        log(f'{thresh:<8} {len(filtered):>10,} {wr:>9.1f}% {pnl.mean():>+9.1f} {pnl.sum():>+12,.0f} {pf:>8.2f}')

# Save NEW model
model_path = MODEL_DIR / 'quality_xgb_m15_v2_pnl_fixed.joblib'
features_path = MODEL_DIR / 'quality_xgb_features_m15_v2_pnl_fixed.pkl'

joblib.dump(model, model_path)
with open(features_path, 'wb') as f:
    pickle.dump(feature_cols, f)

log(f"\nModel saved to: {model_path}")
log(f"Features saved to: {features_path}")
log(f"\nCompleted: {datetime.now()}")
