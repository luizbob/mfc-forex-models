"""
Train PnL-Based XGBoost Model - V2 Both Currencies (H1)
========================================================
Target: is_profitable (exit_pnl_pips > 0) instead of is_quality.
Proper time-sorted train/val/test split.
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
log("TRAINING PNL-BASED MODEL - V2 BOTH CURRENCIES (H1)")
log("=" * 70)
log(f"Started: {datetime.now()}")

DATA_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/lstm_model/v2_both_currencies/data')
MODEL_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/lstm_model/v2_both_currencies/models')

# Load data
log("\nLoading V2 quality entry data (H1)...")
with open(DATA_DIR / 'quality_entry_data_h1_v2.pkl', 'rb') as f:
    data = pickle.load(f)

df = data['data'].copy()
log(f"Total entries: {len(df):,}")

# Fix JPY pips before creating target
jpy_mask = df['pair'].str.contains('JPY')
df.loc[jpy_mask, 'exit_pnl_pips'] = df.loc[jpy_mask, 'exit_pnl_pips'] / 100

# Create PnL-based target
df['is_profitable'] = (df['exit_pnl_pips'] > 0).astype(int)

# Sort by datetime
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

# Target rates
log(f"\nProfitable rates (is_profitable):")
log(f"  Train: {train_df['is_profitable'].mean()*100:.1f}%")
log(f"  Val:   {val_df['is_profitable'].mean()*100:.1f}%")
log(f"  Test:  {test_df['is_profitable'].mean()*100:.1f}%")
log(f"\nFor comparison - Quality rates (is_quality):")
log(f"  Train: {train_df['is_quality'].mean()*100:.1f}%")
log(f"  Val:   {val_df['is_quality'].mean()*100:.1f}%")
log(f"  Test:  {test_df['is_quality'].mean()*100:.1f}%")

# Features - H1 version (vel2 and acc on H1)
feature_cols = [
    'base_m5', 'quote_m5', 'base_vel_m5', 'quote_vel_m5',
    'base_m15', 'quote_m15', 'base_vel_m15', 'quote_vel_m15',
    'base_m30', 'quote_m30', 'base_vel_m30', 'quote_vel_m30',
    'base_h1', 'quote_h1', 'base_vel_h1', 'quote_vel_h1',
    'base_h4', 'quote_h4', 'base_vel_h4', 'quote_vel_h4',
    'base_vel2_h1', 'base_acc_h1',
    'quote_vel2_h1', 'quote_acc_h1',
    'divergence', 'vel_divergence',
    'direction_code', 'trigger_code'
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

# Threshold analysis
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
model_path = MODEL_DIR / 'quality_xgb_h1_v2_pnl.joblib'
features_path = MODEL_DIR / 'quality_xgb_features_h1_v2_pnl.pkl'

joblib.dump(model, model_path)
with open(features_path, 'wb') as f:
    pickle.dump(feature_cols, f)

log(f"\nModel saved to: {model_path}")
log(f"Features saved to: {features_path}")

# Backtest by trigger type
log("\n" + "=" * 70)
log("BACKTEST BY TRIGGER TYPE")
log("=" * 70)

for trigger in ['base', 'quote']:
    subset = test_df[test_df['trigger'] == trigger].copy()
    log(f"\n{trigger.upper()} TRIGGERS ({len(subset):,} entries):")
    log(f"| Threshold | Trades | Profitable% | Win% | Avg PnL | Total PnL | PF |")
    log(f"|-----------|--------|-------------|------|---------|-----------|------|")

    for thresh in [0.7, 0.8, 0.9]:
        filtered = subset[subset['pred_prob'] >= thresh].copy()
        if len(filtered) == 0:
            continue

        profitable_rate = filtered['is_profitable'].mean() * 100
        pnl = filtered['exit_pnl_pips']

        win_rate = (pnl > 0).mean() * 100
        avg_pnl = pnl.mean()
        total_pnl = pnl.sum()
        winners = pnl[pnl > 0].sum()
        losers = abs(pnl[pnl <= 0].sum())
        pf = winners / losers if losers > 0 else float('inf')

        log(f"| {thresh:^9} | {len(filtered):>6,} | {profitable_rate:>10.1f}% | {win_rate:>4.1f}% | {avg_pnl:>7.1f} | {total_pnl:>9.0f} | {pf:>4.2f} |")

# By trigger currency
log("\n" + "=" * 70)
log("BACKTEST BY TRIGGER CURRENCY (Threshold 0.8)")
log("=" * 70)

filtered = test_df[test_df['pred_prob'] >= 0.8].copy()

log(f"\n| Currency | Trades | Profitable% | Win% | Total PnL |")
log(f"|----------|--------|-------------|------|-----------|")

for ccy in ['EUR', 'USD', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD', 'NZD']:
    ccy_subset = filtered[filtered['trigger_ccy'] == ccy]
    if len(ccy_subset) > 0:
        p_rate = ccy_subset['is_profitable'].mean() * 100
        ccy_pnl = ccy_subset['exit_pnl_pips']
        w_rate = (ccy_pnl > 0).mean() * 100
        total_pnl = ccy_pnl.sum()
        log(f"| {ccy:^8} | {len(ccy_subset):>6,} | {p_rate:>10.1f}% | {w_rate:>4.1f}% | {total_pnl:>9.0f} |")

log(f"\nCompleted: {datetime.now()}")
