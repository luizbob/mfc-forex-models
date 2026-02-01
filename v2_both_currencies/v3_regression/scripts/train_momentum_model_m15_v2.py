"""
Train Momentum Model - M15 V2 (velocity exit)
==============================================
"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import pickle
import joblib
from pathlib import Path
from xgboost import XGBClassifier

def log(msg=""):
    print(msg, flush=True)

log("=" * 70)
log("TRAINING MOMENTUM MODEL - M15 V2 (velocity exit)")
log("=" * 70)

DATA_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/lstm_model/v2_both_currencies/v3_regression/data')
MODEL_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/lstm_model/v2_both_currencies/v3_regression/models')

# Load data
log("\nLoading data...")
with open(DATA_DIR / 'momentum_data_m15_v2.pkl', 'rb') as f:
    data = pickle.load(f)

df = data['data'].copy()
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.sort_values('datetime').reset_index(drop=True)

log(f"Total samples: {len(df):,}")
log(f"Profitable: {df['is_profitable'].mean()*100:.1f}%")
log(f"Avg bars held: {df['bars_held'].mean():.1f}")

# Time features
df['hour'] = df['datetime'].dt.hour
df['dayofweek'] = df['datetime'].dt.dayofweek
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 5)
df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 5)
df['direction_code'] = (df['direction'] == 'buy').astype(int)
df['trigger_code'] = (df['trigger'] == 'base').astype(int)

feature_cols = [
    'base_m15', 'base_vel_m15', 'base_mom_m15', 'base_acc_m15',
    'quote_m15', 'quote_vel_m15', 'quote_mom_m15', 'quote_acc_m15',
    'base_m30', 'base_vel_m30', 'base_mom_m30',
    'quote_m30', 'quote_vel_m30', 'quote_mom_m30',
    'base_h1', 'base_vel_h1', 'base_mom_h1',
    'quote_h1', 'quote_vel_h1', 'quote_mom_h1',
    'base_h4', 'base_vel_h4', 'base_mom_h4',
    'quote_h4', 'quote_vel_h4', 'quote_mom_h4',
    'divergence', 'vel_divergence',
    'base_in_box', 'quote_in_box',
    'base_dist_box', 'quote_dist_box',
    'h4_base_confirms_up', 'h4_base_confirms_down',
    'h4_quote_confirms_up', 'h4_quote_confirms_down',
    'entry_vel',
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
    'direction_code', 'trigger_code',
]

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
model = XGBClassifier(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    min_child_weight=10, subsample=0.9, colsample_bytree=0.8,
    gamma=0.2, random_state=42, n_jobs=-1, eval_metric='auc'
)
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
log(f"\nBaseline: {baseline['trades']:,} trades, {baseline['wr']:.1f}% WR, {baseline['avg']:+.2f} avg")

log(f"\n{'Threshold':<10} {'Trades':>10} {'Win %':>8} {'PF':>8} {'Net Pips':>12} {'Avg':>10} {'Hold':>6}")
log("-" * 70)

for thresh in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
    filtered = df_test[df_test['pred_prob'] >= thresh]
    if len(filtered) < 50:
        continue
    r = evaluate(filtered)
    avg_hold = filtered['bars_held'].mean()
    log(f"{thresh:<10} {r['trades']:>10,} {r['wr']:>7.1f}% {r['pf']:>8.2f} {r['net']:>+12,.0f} {r['avg']:>+10.2f} {avg_hold:>6.1f}")

# By bars held at best threshold
log("\n" + "=" * 70)
log("BY BARS HELD (threshold 0.45)")
log("=" * 70)

filtered = df_test[df_test['pred_prob'] >= 0.45]
log(f"\n{'Bars Held':<12} {'Trades':>8} {'Win %':>8} {'Avg Pips':>10}")
log("-" * 45)
for low, high, label in [(1, 3, '1-2'), (3, 6, '3-5'), (6, 12, '6-11'), (12, 25, '12+')]:
    subset = filtered[(filtered['bars_held'] >= low) & (filtered['bars_held'] < high)]
    if len(subset) > 50:
        r = evaluate(subset)
        log(f"{label:<12} {r['trades']:>8,} {r['wr']:>7.1f}% {r['avg']:>+10.2f}")

# Feature importance
log("\n" + "=" * 70)
log("TOP 15 FEATURES")
log("=" * 70)
importance = model.feature_importances_
indices = np.argsort(importance)[::-1][:15]
for i, idx in enumerate(indices):
    log(f"  {i+1:2}. {feature_cols[idx]:<30} {importance[idx]:.4f}")

# Save
joblib.dump(model, MODEL_DIR / 'momentum_xgb_m15_v2.joblib')
with open(MODEL_DIR / 'momentum_xgb_features_m15_v2.pkl', 'wb') as f:
    pickle.dump(feature_cols, f)

log("\nModel saved.")
log("DONE")
