"""
Train PnL Model V2 - Optimized Version
=======================================
Experimental improvements - DO NOT overwrite original model.
Saves to separate files: quality_xgb_m5_v2_pnl_optimized.joblib
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
from sklearn.model_selection import RandomizedSearchCV

def log(msg=""):
    print(msg, flush=True)

log("=" * 70)
log("TRAINING PNL MODEL V2 - OPTIMIZED (EXPERIMENTAL)")
log("=" * 70)
log(f"Started: {datetime.now()}")
log("NOTE: This will NOT overwrite the current model!")

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

# Features - expanded version
feature_cols = [
    # Original features
    'base_m5', 'quote_m5', 'base_vel_m5', 'quote_vel_m5',
    'base_m15', 'quote_m15', 'base_vel_m15', 'quote_vel_m15',
    'base_m30', 'quote_m30', 'base_vel_m30', 'quote_vel_m30',
    'base_h1', 'quote_h1', 'base_vel_h1', 'quote_vel_h1',
    'base_h4', 'quote_h4', 'base_vel_h4', 'quote_vel_h4',
    'base_vel2_m5', 'base_acc_m5',
    'quote_vel2_m5', 'quote_acc_m5',
    'divergence', 'vel_divergence',
    'direction_code', 'trigger_code',
    # New time features
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

# ============================================================================
# HYPERPARAMETER SEARCH
# ============================================================================

log("\n" + "=" * 70)
log("HYPERPARAMETER SEARCH")
log("=" * 70)

# Define search space
param_dist = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [4, 5, 6, 7, 8],
    'learning_rate': [0.01, 0.05, 0.1, 0.15],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.1, 0.2],
}

base_model = XGBClassifier(
    random_state=42,
    eval_metric='auc',
    n_jobs=-1
)

log("Running randomized search (20 iterations)...")
search = RandomizedSearchCV(
    base_model,
    param_distributions=param_dist,
    n_iter=20,
    scoring='roc_auc',
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=1  # XGBoost already parallelizes
)

# Use a subsample for faster search
sample_size = min(500000, len(X_train))
sample_idx = np.random.choice(len(X_train), sample_size, replace=False)
X_train_sample = X_train[sample_idx]
y_train_sample = y_train[sample_idx]

search.fit(X_train_sample, y_train_sample)

log(f"\nBest parameters: {search.best_params_}")
log(f"Best CV AUC: {search.best_score_:.4f}")

# ============================================================================
# TRAIN FINAL MODEL WITH BEST PARAMS
# ============================================================================

log("\n" + "=" * 70)
log("TRAINING FINAL MODEL WITH BEST PARAMS")
log("=" * 70)

best_params = search.best_params_
best_params['random_state'] = 42
best_params['eval_metric'] = 'auc'
best_params['early_stopping_rounds'] = 20

model = XGBClassifier(**best_params)

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

# ============================================================================
# COMPARE WITH ORIGINAL MODEL
# ============================================================================

log("\n" + "=" * 70)
log("COMPARISON: ORIGINAL vs OPTIMIZED")
log("=" * 70)

# Load original model
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
    test_df['opt_pred'] = model.predict_proba(X_test)[:, 1]

    log(f"\n{'Model':<12} {'Thresh':<8} {'Trades':>8} {'WinRate':>8} {'AvgPnL':>8} {'TotalPnL':>10} {'PF':>6}")
    log("-" * 70)

    for thresh in [0.70, 0.80, 0.90]:
        # Original
        f_orig = test_df[test_df['orig_pred'] >= thresh]
        if len(f_orig) > 0:
            pnl = f_orig['exit_pnl_pips']
            wr = (pnl > 0).mean() * 100
            w = pnl[pnl > 0].sum()
            l = abs(pnl[pnl <= 0].sum())
            pf = w / l if l > 0 else float('inf')
            log(f"{'Original':<12} {thresh:<8} {len(f_orig):>8,} {wr:>7.1f}% {pnl.mean():>+7.1f} {pnl.sum():>+10,.0f} {pf:>6.2f}")

        # Optimized
        f_opt = test_df[test_df['opt_pred'] >= thresh]
        if len(f_opt) > 0:
            pnl = f_opt['exit_pnl_pips']
            wr = (pnl > 0).mean() * 100
            w = pnl[pnl > 0].sum()
            l = abs(pnl[pnl <= 0].sum())
            pf = w / l if l > 0 else float('inf')
            log(f"{'Optimized':<12} {thresh:<8} {len(f_opt):>8,} {wr:>7.1f}% {pnl.mean():>+7.1f} {pnl.sum():>+10,.0f} {pf:>6.2f}")

        log("-" * 70)

# Save OPTIMIZED model (separate files!)
opt_model_path = MODEL_DIR / 'quality_xgb_m5_v2_pnl_optimized.joblib'
opt_features_path = MODEL_DIR / 'quality_xgb_features_m5_v2_pnl_optimized.pkl'

joblib.dump(model, opt_model_path)
with open(opt_features_path, 'wb') as f:
    pickle.dump(feature_cols, f)

log(f"\nOptimized model saved to: {opt_model_path}")
log(f"Features saved to: {opt_features_path}")
log(f"\nNOTE: Original model is UNCHANGED at: {orig_model_path}")

log(f"\nCompleted: {datetime.now()}")
