"""
Train Momentum Model - M15 Signal
=================================
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

def log(msg=""):
    print(msg, flush=True)

log("=" * 70)
log("TRAINING MOMENTUM MODEL - M15 SIGNAL")
log("=" * 70)

DATA_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/lstm_model/v2_both_currencies/v3_regression/data')
MODEL_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/lstm_model/v2_both_currencies/v3_regression/models')

# Load data
log("\nLoading data...")
with open(DATA_DIR / 'momentum_data_m15.pkl', 'rb') as f:
    data = pickle.load(f)

df = data['data'].copy()
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.sort_values('datetime').reset_index(drop=True)

log(f"Total samples: {len(df):,}")
log(f"Date range: {df['datetime'].min().date()} to {df['datetime'].max().date()}")

# Add time features
df['hour'] = df['datetime'].dt.hour
df['dayofweek'] = df['datetime'].dt.dayofweek
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 5)
df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 5)

df['direction_code'] = (df['direction'] == 'buy').astype(int)
df['trigger_code'] = (df['trigger'] == 'base').astype(int)

# Feature columns
feature_cols = [
    # M15 features
    'base_m15', 'base_vel_m15', 'base_mom_m15', 'base_acc_m15',
    'quote_m15', 'quote_vel_m15', 'quote_mom_m15', 'quote_acc_m15',
    # M30 features
    'base_m30', 'base_vel_m30', 'base_mom_m30',
    'quote_m30', 'quote_vel_m30', 'quote_mom_m30',
    # H1 features
    'base_h1', 'base_vel_h1', 'base_mom_h1',
    'quote_h1', 'quote_vel_h1', 'quote_mom_h1',
    # H4 features
    'base_h4', 'base_vel_h4', 'base_mom_h4',
    'quote_h4', 'quote_vel_h4', 'quote_mom_h4',
    # Derived
    'divergence', 'vel_divergence',
    'base_in_box', 'quote_in_box',
    'base_dist_box', 'quote_dist_box',
    'h4_base_confirms_up', 'h4_base_confirms_down',
    'h4_quote_confirms_up', 'h4_quote_confirms_down',
    'entry_vel',
    # Time
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
    # Categorical
    'direction_code', 'trigger_code',
]

log(f"Features: {len(feature_cols)}")

X = df[feature_cols].values.astype(np.float32)
y = df['is_profitable'].values.astype(np.int32)
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

# Time split
n = len(df)
train_size = int(n * 0.8)

X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]
df_test = df.iloc[train_size:].copy()

log(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

# Train
log("\nTraining...")
params = {
    'n_estimators': 300,
    'max_depth': 4,
    'learning_rate': 0.05,
    'min_child_weight': 10,
    'subsample': 0.9,
    'colsample_bytree': 0.8,
    'gamma': 0.2,
    'random_state': 42,
    'n_jobs': -1,
    'eval_metric': 'auc',
}

model = XGBClassifier(**params)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=100)

# Evaluate
log("\n" + "=" * 70)
log("EVALUATION")
log("=" * 70)

df_test['pred_prob'] = model.predict_proba(X_test)[:, 1]

def evaluate(df):
    if len(df) == 0:
        return None
    net = df['net_pips']
    trades = len(df)
    wr = (net > 0).mean() * 100
    winners = net[net > 0].sum()
    losers = abs(net[net <= 0].sum())
    pf = winners / losers if losers > 0 else 999
    return {'trades': trades, 'wr': wr, 'pf': pf, 'net': net.sum(), 'avg': net.mean()}

baseline = evaluate(df_test)
log(f"\nBaseline: {baseline['trades']:,} trades, {baseline['wr']:.1f}% WR, PF {baseline['pf']:.2f}, {baseline['avg']:+.2f} avg")

log(f"\n{'Threshold':<10} {'Trades':>10} {'Win %':>8} {'PF':>8} {'Net Pips':>12} {'Avg':>10}")
log("-" * 60)

for thresh in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
    filtered = df_test[df_test['pred_prob'] >= thresh]
    if len(filtered) < 50:
        continue
    r = evaluate(filtered)
    log(f"{thresh:<10} {r['trades']:>10,} {r['wr']:>7.1f}% {r['pf']:>8.2f} {r['net']:>+12,.0f} {r['avg']:>+10.2f}")

# Feature importance
log("\n" + "=" * 70)
log("TOP 15 FEATURES")
log("=" * 70)

importance = model.feature_importances_
indices = np.argsort(importance)[::-1][:15]
for i, idx in enumerate(indices):
    log(f"  {i+1:2}. {feature_cols[idx]:<30} {importance[idx]:.4f}")

# Save
model_path = MODEL_DIR / 'momentum_xgb_m15.joblib'
features_path = MODEL_DIR / 'momentum_xgb_features_m15.pkl'

joblib.dump(model, model_path)
with open(features_path, 'wb') as f:
    pickle.dump(feature_cols, f)

log(f"\nSaved to: {model_path}")
log("DONE")
