"""
Train Quality Entry XGBoost Model for M15 - Proper Time-Sorted Data
====================================================================
Same as H1/M30 but for M15 base timeframe.
Uses proper time-based train/val/test split to avoid data leakage.
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
from sklearn.metrics import roc_auc_score, classification_report

def log(msg=""):
    print(msg, flush=True)

log("=" * 70)
log("TRAINING QUALITY ENTRY MODEL (M15) - PROPER TIME SPLIT")
log("=" * 70)
log(f"Started: {datetime.now()}")

DATA_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/lstm_model/data')
MODEL_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/lstm_model/models')

# Load data
log("\nLoading M15 quality entry data...")
with open(DATA_DIR / 'quality_entry_data_m15.pkl', 'rb') as f:
    data = pickle.load(f)

df = data['data'].copy()
log(f"Total entries: {len(df):,}")

# CRITICAL: Sort by datetime for proper time-based split
log("\nSorting by datetime for proper time-based split...")
df = df.sort_values('datetime').reset_index(drop=True)

log(f"Date range: {df['datetime'].min().date()} to {df['datetime'].max().date()}")

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

# Verify no overlap
log(f"\nVerifying no data leakage:")
log(f"  Train ends: {train_df['datetime'].max()}")
log(f"  Val starts: {val_df['datetime'].min()}")
log(f"  Val ends:   {val_df['datetime'].max()}")
log(f"  Test starts: {test_df['datetime'].min()}")

# Quality rates
log(f"\nQuality rates:")
log(f"  Train: {train_df['is_quality'].mean()*100:.1f}%")
log(f"  Val:   {val_df['is_quality'].mean()*100:.1f}%")
log(f"  Test:  {test_df['is_quality'].mean()*100:.1f}%")

# Features
feature_cols = [
    'base_m5', 'quote_m5', 'base_vel_m5', 'quote_vel_m5',
    'base_m15', 'quote_m15', 'base_vel_m15', 'quote_vel_m15',
    'base_m30', 'quote_m30', 'base_vel_m30', 'quote_vel_m30',
    'base_h1', 'quote_h1', 'base_vel_h1', 'quote_vel_h1',
    'base_h4', 'quote_h4', 'base_vel_h4', 'quote_vel_h4',
    'base_vel2_m15', 'base_acc_m15',
    'divergence', 'vel_divergence',
    'direction_code'
]

# Add direction code
for split_df in [train_df, val_df, test_df]:
    split_df['direction_code'] = (split_df['direction'] == 'buy').astype(int)

# Prepare features
X_train = train_df[feature_cols].values.astype(np.float32)
X_val = val_df[feature_cols].values.astype(np.float32)
X_test = test_df[feature_cols].values.astype(np.float32)

y_train = train_df['is_quality'].values
y_val = val_df['is_quality'].values
y_test = test_df['is_quality'].values

# Handle NaN/Inf
X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

# Train XGBoost
log("\nTraining XGBoost...")

model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='auc',
    early_stopping_rounds=20
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

# Threshold analysis on test set
log("\nTest Set Performance by Threshold:")
log(f"| Threshold | Precision | Recall | Count | Base Rate |")
log(f"|-----------|-----------|--------|-------|-----------|")

test_probs = model.predict_proba(X_test)[:, 1]
test_df['pred_prob'] = test_probs

base_rate = y_test.mean()
for thresh in [0.5, 0.6, 0.7, 0.8, 0.9]:
    mask = test_probs >= thresh
    if mask.sum() > 0:
        precision = y_test[mask].mean()
        recall = y_test[mask].sum() / y_test.sum()
        log(f"| {thresh:^9} | {precision:>8.1%} | {recall:>5.1%} | {mask.sum():>5,} | {base_rate:>8.1%} |")

# Feature importance
log("\nTop 10 Feature Importances:")
importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

for _, row in importance.head(10).iterrows():
    log(f"  {row['feature']:20s}: {row['importance']:.4f}")

# Save model
model_path = MODEL_DIR / 'quality_xgb_m15_proper.joblib'
features_path = MODEL_DIR / 'quality_xgb_features_m15_proper.pkl'

joblib.dump(model, model_path)
with open(features_path, 'wb') as f:
    pickle.dump(feature_cols, f)

log(f"\nModel saved to: {model_path}")
log(f"Features saved to: {features_path}")

# Backtest simulation
log("\n" + "=" * 70)
log("BACKTEST SIMULATION ON TEST SET")
log("=" * 70)

log(f"\n| Threshold | Trades | Quality% | Win% | Avg PnL | PF |")
log(f"|-----------|--------|----------|------|---------|------|")

for thresh in [0.5, 0.6, 0.7, 0.8, 0.9]:
    filtered = test_df[test_df['pred_prob'] >= thresh].copy()

    if len(filtered) == 0:
        continue

    quality_rate = filtered['is_quality'].mean() * 100

    # P&L simulation
    filtered['trade_pnl'] = np.where(
        filtered['is_quality'] == 1,
        filtered['max_profit_pips'] * 0.7,
        -filtered['max_dd_pips'] * 0.5
    )

    win_rate = (filtered['trade_pnl'] > 0).mean() * 100
    avg_pnl = filtered['trade_pnl'].mean()

    winners = filtered[filtered['trade_pnl'] > 0]['trade_pnl'].sum()
    losers = abs(filtered[filtered['trade_pnl'] <= 0]['trade_pnl'].sum())
    pf = winners / losers if losers > 0 else float('inf')

    log(f"| {thresh:^9} | {len(filtered):>6,} | {quality_rate:>7.1f}% | {win_rate:>4.1f}% | {avg_pnl:>7.1f} | {pf:>4.2f} |")

# By direction
log(f"\nBy Direction (Threshold 0.7):")
filtered = test_df[test_df['pred_prob'] >= 0.7].copy()
filtered['trade_pnl'] = np.where(
    filtered['is_quality'] == 1,
    filtered['max_profit_pips'] * 0.7,
    -filtered['max_dd_pips'] * 0.5
)

for direction in ['buy', 'sell']:
    dir_df = filtered[filtered['direction'] == direction]
    if len(dir_df) > 0:
        q_rate = dir_df['is_quality'].mean() * 100
        win_rate = (dir_df['trade_pnl'] > 0).mean() * 100
        log(f"  {direction.upper()}: {len(dir_df):,} trades, {q_rate:.1f}% quality, {win_rate:.1f}% win rate")

log(f"\nCompleted: {datetime.now()}")
