"""
Train Momentum Model - M15
==========================
XGBoost classifier to predict which momentum crossings will be profitable.
"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import pickle
import joblib
from pathlib import Path
from datetime import datetime
from sklearn.metrics import roc_auc_score
import xgboost as xgb

def log(msg=""):
    print(msg, flush=True)

log("=" * 70)
log("TRAINING MOMENTUM MODEL (M15)")
log("=" * 70)
log(f"Started: {datetime.now()}")

DATA_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/lstm_model/v2_both_currencies/data')
MODEL_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/lstm_model/v2_both_currencies/models')

# Load data
log("\nLoading data...")
with open(DATA_DIR / 'momentum_entry_data_m15.pkl', 'rb') as f:
    data = pickle.load(f)

df = data['data'].copy()
log(f"Total entries: {len(df):,}")

# Sort by time
df = df.sort_values('datetime').reset_index(drop=True)

# Create target: profitable trade (positive PnL)
# Fix JPY pairs first
jpy_mask = df['pair'].str.contains('JPY')
df.loc[jpy_mask, 'exit_pnl_pips'] = df.loc[jpy_mask, 'exit_pnl_pips'] / 100

df['is_profitable'] = (df['exit_pnl_pips'] > 0).astype(int)

log(f"Profitable trades: {df['is_profitable'].sum():,} ({df['is_profitable'].mean()*100:.1f}%)")

# Feature columns - M15 base timeframe
feature_cols = [
    # MFC levels across timeframes
    'base_m5', 'quote_m5',
    'base_m15', 'quote_m15',
    'base_m30', 'quote_m30',
    'base_h1', 'quote_h1',
    'base_h4', 'quote_h4',
    # Velocities
    'base_vel_m5', 'quote_vel_m5',
    'base_vel_m15', 'quote_vel_m15',
    'base_vel_m30', 'quote_vel_m30',
    'base_vel_h1', 'quote_vel_h1',
    'base_vel_h4', 'quote_vel_h4',
    # Additional momentum features (M15)
    'base_vel3_m15', 'base_vel5_m15',
    'quote_vel3_m15', 'quote_vel5_m15',
    'base_acc_m15', 'quote_acc_m15',
    # Divergence
    'divergence', 'vel_divergence',
    # Velocity at crossing (key feature!)
    'velocity_at_cross',
]

# Add encoded features
df['direction_code'] = (df['direction'] == 'buy').astype(int)
df['trigger_code'] = (df['trigger'] == 'base').astype(int)
feature_cols.extend(['direction_code', 'trigger_code'])

log(f"Features: {len(feature_cols)}")

# Time-based split (70/15/15)
n = len(df)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

train_df = df.iloc[:train_end]
val_df = df.iloc[train_end:val_end]
test_df = df.iloc[val_end:]

log(f"\nTrain: {len(train_df):,} ({train_df['datetime'].min().date()} to {train_df['datetime'].max().date()})")
log(f"Val:   {len(val_df):,} ({val_df['datetime'].min().date()} to {val_df['datetime'].max().date()})")
log(f"Test:  {len(test_df):,} ({test_df['datetime'].min().date()} to {test_df['datetime'].max().date()})")

# Prepare data
X_train = train_df[feature_cols].values.astype(np.float32)
y_train = train_df['is_profitable'].values
X_val = val_df[feature_cols].values.astype(np.float32)
y_val = val_df['is_profitable'].values
X_test = test_df[feature_cols].values.astype(np.float32)
y_test = test_df['is_profitable'].values

# Handle NaN/Inf
X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

log(f"\nTrain positive rate: {y_train.mean()*100:.1f}%")
log(f"Val positive rate: {y_val.mean()*100:.1f}%")
log(f"Test positive rate: {y_test.mean()*100:.1f}%")

# Train model
log("\nTraining XGBoost...")
model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    eval_metric='auc',
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

# Evaluate
train_pred = model.predict_proba(X_train)[:, 1]
val_pred = model.predict_proba(X_val)[:, 1]
test_pred = model.predict_proba(X_test)[:, 1]

train_auc = roc_auc_score(y_train, train_pred)
val_auc = roc_auc_score(y_val, val_pred)
test_auc = roc_auc_score(y_test, test_pred)

log(f"\nAUC Scores:")
log(f"  Train: {train_auc:.4f}")
log(f"  Val:   {val_auc:.4f}")
log(f"  Test:  {test_auc:.4f}")

# Feature importance
log("\nTop 10 Features:")
importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

for i, row in importance.head(10).iterrows():
    log(f"  {row['feature']:20s}: {row['importance']:.4f}")

# Threshold analysis on test set
log("\n" + "=" * 70)
log("THRESHOLD ANALYSIS (Test Set)")
log("=" * 70)

test_df = test_df.copy()
test_df['pred_prob'] = test_pred

log(f"\n| Thresh | Trades | Success% | Win%  | Avg PnL | Total PnL | PF    |")
log(f"|--------|--------|----------|-------|---------|-----------|-------|")

for thresh in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
    filtered = test_df[test_df['pred_prob'] >= thresh]
    if len(filtered) < 10:
        continue

    success_rate = filtered['reached_target'].mean() * 100
    pnl = filtered['exit_pnl_pips']
    win_rate = (pnl > 0).mean() * 100
    avg_pnl = pnl.mean()
    total_pnl = pnl.sum()

    winners = pnl[pnl > 0].sum()
    losers = abs(pnl[pnl <= 0].sum())
    pf = winners / losers if losers > 0 else float('inf')

    # Trades per day
    days = (filtered['datetime'].max() - filtered['datetime'].min()).days
    tpd = len(filtered) / days if days > 0 else 0

    log(f"| {thresh:>5.2f} | {len(filtered):>6,} | {success_rate:>7.1f}% | {win_rate:>5.1f}% | {avg_pnl:>7.1f} | {total_pnl:>9.0f} | {pf:>5.2f} | {tpd:.1f}/day")

# By velocity quartile at best threshold
log("\n" + "=" * 70)
log("BY VELOCITY (at 0.60 threshold)")
log("=" * 70)

filtered = test_df[test_df['pred_prob'] >= 0.60].copy()
if len(filtered) > 100:
    filtered['vel_q'] = pd.qcut(filtered['velocity_at_cross'].abs(), 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])

    log(f"\n| Velocity | Trades | Win%  | Avg PnL | Total PnL |")
    log(f"|----------|--------|-------|---------|-----------|")
    for q in ['Q1', 'Q2', 'Q3', 'Q4']:
        subset = filtered[filtered['vel_q'] == q]
        if len(subset) > 0:
            pnl = subset['exit_pnl_pips']
            wr = (pnl > 0).mean() * 100
            log(f"| {q:8s} | {len(subset):>6,} | {wr:>5.1f}% | {pnl.mean():>7.1f} | {pnl.sum():>9.0f} |")

# By trigger currency
log("\n" + "=" * 70)
log("BY TRIGGER CURRENCY (at 0.60 threshold)")
log("=" * 70)

log(f"\n| Currency | Trades | Win%  | Avg PnL | Total PnL |")
log(f"|----------|--------|-------|---------|-----------|")
for ccy in ['EUR', 'USD', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD', 'NZD']:
    subset = filtered[filtered['trigger_ccy'] == ccy]
    if len(subset) > 0:
        pnl = subset['exit_pnl_pips']
        wr = (pnl > 0).mean() * 100
        log(f"| {ccy:8s} | {len(subset):>6,} | {wr:>5.1f}% | {pnl.mean():>7.1f} | {pnl.sum():>9.0f} |")

# Save model
model_path = MODEL_DIR / 'momentum_xgb_m15.joblib'
joblib.dump(model, model_path)
log(f"\nModel saved to: {model_path}")

# Save features
features_path = MODEL_DIR / 'momentum_xgb_features_m15.pkl'
with open(features_path, 'wb') as f:
    pickle.dump(feature_cols, f)
log(f"Features saved to: {features_path}")

log(f"\nCompleted: {datetime.now()}")
