"""
Train PnL Model - V2 Both Currencies (M5)
==========================================
Target: is_profitable = exit_pnl_pips > 0
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
log("TRAINING PNL MODEL (PROFITABLE) - V2 M5")
log("=" * 70)
log(f"Started: {datetime.now()}")

DATA_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/lstm_model/v2_both_currencies/data')
MODEL_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/lstm_model/v2_both_currencies/models')

# Load data
log("\nLoading V2 quality entry data (M5)...")
with open(DATA_DIR / 'quality_entry_data_m5_v2.pkl', 'rb') as f:
    data = pickle.load(f)

df = data['data'].copy()
log(f"Total entries: {len(df):,}")

# Fix JPY pips before creating target
jpy_mask = df['pair'].str.contains('JPY')
df.loc[jpy_mask, 'exit_pnl_pips'] = df.loc[jpy_mask, 'exit_pnl_pips'] / 100
df.loc[jpy_mask, 'max_dd_pips'] = df.loc[jpy_mask, 'max_dd_pips'] / 100

# Create target: is_profitable
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
log(f"\nTarget rates (is_profitable):")
log(f"  Train: {train_df['is_profitable'].mean()*100:.1f}%")
log(f"  Val:   {val_df['is_profitable'].mean()*100:.1f}%")
log(f"  Test:  {test_df['is_profitable'].mean()*100:.1f}%")

# Features - M5 version with all timeframes
feature_cols = [
    'base_m5', 'quote_m5', 'base_vel_m5', 'quote_vel_m5',
    'base_m15', 'quote_m15', 'base_vel_m15', 'quote_vel_m15',
    'base_m30', 'quote_m30', 'base_vel_m30', 'quote_vel_m30',
    'base_h1', 'quote_h1', 'base_vel_h1', 'quote_vel_h1',
    'base_h4', 'quote_h4', 'base_vel_h4', 'quote_vel_h4',
    'base_vel2_m5', 'base_acc_m5',
    'quote_vel2_m5', 'quote_acc_m5',
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

# Feature importance
log("\nTop 15 Feature Importances:")
importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

for _, row in importance.head(15).iterrows():
    log(f"  {row['feature']:20s}: {row['importance']:.4f}")

# Save model
model_path = MODEL_DIR / 'quality_xgb_m5_v2_pnl.joblib'
features_path = MODEL_DIR / 'quality_xgb_features_m5_v2_pnl.pkl'

joblib.dump(model, model_path)
with open(features_path, 'wb') as f:
    pickle.dump(feature_cols, f)

log(f"\nModel saved to: {model_path}")
log(f"Features saved to: {features_path}")

# Backtest with REAL PnL metrics
log("\n" + "=" * 70)
log("BACKTEST ON TEST SET")
log("=" * 70)

test_probs = model.predict_proba(X_test)[:, 1]
test_df['pred_prob'] = test_probs

log(f"\n{'Threshold':<10} {'Trades':>8} {'WinRate':>8} {'AvgPnL':>8} {'TotalPnL':>10} {'PF':>6} {'TPD':>6}")
log("-" * 60)

days = (test_df['datetime'].max() - test_df['datetime'].min()).days
for thresh in [0.50, 0.60, 0.70, 0.80, 0.90]:
    f = test_df[test_df['pred_prob'] >= thresh]
    if len(f) == 0:
        continue
    pnl = f['exit_pnl_pips']
    wr = (pnl > 0).mean() * 100
    avg = pnl.mean()
    total = pnl.sum()
    w = pnl[pnl > 0].sum()
    l = abs(pnl[pnl <= 0].sum())
    pf = w / l if l > 0 else float('inf')
    tpd = len(f) / days if days > 0 else 0
    log(f"{thresh:<10} {len(f):>8,} {wr:>7.1f}% {avg:>+7.1f} {total:>+10,.0f} {pf:>6.2f} {tpd:>5.1f}")

# By year
log("\n" + "=" * 70)
log("BY YEAR (Threshold 0.70)")
log("=" * 70)

filtered = test_df[test_df['pred_prob'] >= 0.70].copy()
filtered['year'] = pd.to_datetime(filtered['datetime']).dt.year

log(f"\n{'Year':<6} {'Trades':>8} {'WinRate':>8} {'AvgPnL':>8} {'TotalPnL':>10} {'PF':>6}")
log("-" * 50)

for year in sorted(filtered['year'].unique()):
    yf = filtered[filtered['year'] == year]
    pnl = yf['exit_pnl_pips']
    wr = (pnl > 0).mean() * 100
    w = pnl[pnl > 0].sum()
    l = abs(pnl[pnl <= 0].sum())
    pf = w / l if l > 0 else float('inf')
    log(f"{year:<6} {len(yf):>8,} {wr:>7.1f}% {pnl.mean():>+7.1f} {pnl.sum():>+10,.0f} {pf:>6.2f}")

# By pair (top 10)
log("\n" + "=" * 70)
log("TOP 10 PAIRS BY TOTAL PNL (Threshold 0.70)")
log("=" * 70)

pair_stats = filtered.groupby('pair').agg({
    'exit_pnl_pips': ['count', 'sum', 'mean']
}).round(1)
pair_stats.columns = ['trades', 'total_pnl', 'avg_pnl']
pair_stats['win_rate'] = filtered.groupby('pair').apply(lambda x: (x['exit_pnl_pips'] > 0).mean() * 100)
pair_stats = pair_stats.sort_values('total_pnl', ascending=False)

log(f"\n{'Pair':<10} {'Trades':>8} {'WinRate':>8} {'AvgPnL':>8} {'TotalPnL':>10}")
log("-" * 50)

for pair, row in pair_stats.head(10).iterrows():
    log(f"{pair:<10} {int(row['trades']):>8,} {row['win_rate']:>7.1f}% {row['avg_pnl']:>+7.1f} {row['total_pnl']:>+10,.0f}")

log(f"\nCompleted: {datetime.now()}")
