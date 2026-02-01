"""
Train Momentum Continuation Model
=================================
XGBoost classifier to predict profitable momentum continuations.
Same approach as quality model.
"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import pickle
import joblib
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

def log(msg=""):
    print(msg, flush=True)

log("=" * 70)
log("TRAINING MOMENTUM CONTINUATION MODEL")
log("=" * 70)

DATA_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/lstm_model/v2_both_currencies/v3_regression/data')
MODEL_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/lstm_model/v2_both_currencies/v3_regression/models')
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Load data
log("\nLoading data...")
with open(DATA_DIR / 'momentum_data_m30.pkl', 'rb') as f:
    data = pickle.load(f)

df = data['data'].copy()
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.sort_values('datetime').reset_index(drop=True)

log(f"Total samples: {len(df):,}")
log(f"Date range: {df['datetime'].min().date()} to {df['datetime'].max().date()}")
log(f"Profitable rate: {df['is_profitable'].mean()*100:.1f}%")

# ============================================================================
# PREPARE FEATURES
# ============================================================================

log("\n" + "=" * 70)
log("PREPARING FEATURES")
log("=" * 70)

# Add time features
df['hour'] = df['datetime'].dt.hour
df['dayofweek'] = df['datetime'].dt.dayofweek
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 5)
df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 5)

# Encode categorical
df['direction_code'] = (df['direction'] == 'buy').astype(int)
df['trigger_code'] = (df['trigger'] == 'base').astype(int)

# Feature columns
feature_cols = [
    # M30 features
    'base_m30', 'base_vel_m30', 'base_mom_m30', 'base_acc_m30',
    'quote_m30', 'quote_vel_m30', 'quote_mom_m30', 'quote_acc_m30',
    # H1 features
    'base_h1', 'base_vel_h1', 'base_mom_h1',
    'quote_h1', 'quote_vel_h1', 'quote_mom_h1',
    # H4 features
    'base_h4', 'base_vel_h4', 'base_mom_h4',
    'quote_h4', 'quote_vel_h4', 'quote_mom_h4',
    # Derived features
    'divergence', 'vel_divergence',
    'base_in_box', 'quote_in_box',
    'base_dist_box', 'quote_dist_box',
    'entry_vel',
    # Time features
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
    # Categorical
    'direction_code', 'trigger_code',
]

log(f"Features: {len(feature_cols)}")

# Target
target_col = 'is_profitable'

# Prepare X and y
X = df[feature_cols].values.astype(np.float32)
y = df[target_col].values.astype(np.int32)

# Handle NaN/Inf
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

log(f"X shape: {X.shape}")
log(f"y distribution: {y.mean()*100:.1f}% positive")

# ============================================================================
# TIME SERIES SPLIT
# ============================================================================

log("\n" + "=" * 70)
log("TIME SERIES CROSS-VALIDATION")
log("=" * 70)

# Split: 80% train, 20% test (time-ordered)
n = len(df)
train_size = int(n * 0.8)
test_start = train_size

X_train = X[:train_size]
y_train = y[:train_size]
X_test = X[train_size:]
y_test = y[train_size:]
df_test = df.iloc[train_size:].copy()

log(f"Train: {len(X_train):,} samples ({df['datetime'].iloc[0].date()} to {df['datetime'].iloc[train_size-1].date()})")
log(f"Test: {len(X_test):,} samples ({df['datetime'].iloc[train_size].date()} to {df['datetime'].iloc[-1].date()})")

# ============================================================================
# TRAIN MODEL
# ============================================================================

log("\n" + "=" * 70)
log("TRAINING XGBOOST")
log("=" * 70)

# Use similar params to quality model
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
    'use_label_encoder': False,
    'eval_metric': 'auc',
}

log(f"Params: {params}")

model = XGBClassifier(**params)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=50)

# ============================================================================
# EVALUATE
# ============================================================================

log("\n" + "=" * 70)
log("EVALUATION ON TEST SET")
log("=" * 70)

# Predict probabilities
df_test['pred_prob'] = model.predict_proba(X_test)[:, 1]

# Spreads
SPREADS = {
    'AUDUSD': 0.9, 'EURUSD': 0.8, 'GBPUSD': 1.0, 'NZDUSD': 1.8,
    'USDCAD': 1.5, 'USDCHF': 1.3, 'USDJPY': 1.0,
    'AUDCAD': 2.2, 'AUDCHF': 0.9, 'AUDJPY': 1.9, 'AUDNZD': 2.0,
    'CADCHF': 0.8, 'CADJPY': 3.8, 'CHFJPY': 2.4,
    'EURAUD': 3.4, 'EURCAD': 2.9, 'EURCHF': 2.5, 'EURGBP': 1.4,
    'EURJPY': 2.4, 'EURNZD': 5.4,
    'GBPAUD': 2.5, 'GBPCAD': 4.8, 'GBPCHF': 2.4, 'GBPJPY': 2.2, 'GBPNZD': 5.8,
    'NZDCAD': 2.1, 'NZDCHF': 1.5, 'NZDJPY': 4.3,
}

def evaluate(df, label=""):
    if len(df) == 0:
        return None

    net = df['net_pips']
    trades = len(df)
    wr = (net > 0).mean() * 100
    winners = net[net > 0].sum()
    losers = abs(net[net <= 0].sum())
    pf = winners / losers if losers > 0 else 999
    net_pnl = net.sum()
    avg = net.mean()

    return {
        'trades': trades,
        'wr': wr,
        'pf': pf,
        'net_pnl': net_pnl,
        'avg': avg,
    }

# Baseline (no filter)
baseline = evaluate(df_test, "No filter")
log(f"\nBaseline (all signals):")
log(f"  Trades: {baseline['trades']:,}")
log(f"  Win rate: {baseline['wr']:.1f}%")
log(f"  PF: {baseline['pf']:.2f}")
log(f"  Net pips: {baseline['net_pnl']:+,.0f}")
log(f"  Avg pips: {baseline['avg']:+.2f}")

# By threshold
log(f"\n{'Threshold':<12} {'Trades':>10} {'Win %':>10} {'PF':>8} {'Net Pips':>12} {'Avg':>10}")
log("-" * 65)

for thresh in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
    filtered = df_test[df_test['pred_prob'] >= thresh]
    if len(filtered) < 50:
        continue
    r = evaluate(filtered)
    log(f"{thresh:<12} {r['trades']:>10,} {r['wr']:>9.1f}% {r['pf']:>8.2f} {r['net_pnl']:>+12,.0f} {r['avg']:>+10.2f}")

# ============================================================================
# FEATURE IMPORTANCE
# ============================================================================

log("\n" + "=" * 70)
log("FEATURE IMPORTANCE (Top 15)")
log("=" * 70)

importance = model.feature_importances_
indices = np.argsort(importance)[::-1][:15]

for i, idx in enumerate(indices):
    log(f"  {i+1:2}. {feature_cols[idx]:<25} {importance[idx]:.4f}")

# ============================================================================
# MONTHLY BREAKDOWN
# ============================================================================

log("\n" + "=" * 70)
log("MONTHLY BREAKDOWN (threshold 0.60)")
log("=" * 70)

df_test['month'] = df_test['datetime'].dt.to_period('M')
filtered = df_test[df_test['pred_prob'] >= 0.60]

log(f"\n{'Month':<10} {'Trades':>8} {'Win %':>8} {'PF':>8} {'Net Pips':>10}")
log("-" * 50)

for month in sorted(filtered['month'].unique()):
    m_df = filtered[filtered['month'] == month]
    r = evaluate(m_df)
    if r:
        log(f"{str(month):<10} {r['trades']:>8,} {r['wr']:>7.1f}% {r['pf']:>8.2f} {r['net_pnl']:>+10,.0f}")

# ============================================================================
# SAVE MODEL
# ============================================================================

log("\n" + "=" * 70)
log("SAVING MODEL")
log("=" * 70)

model_path = MODEL_DIR / 'momentum_xgb_m30.joblib'
features_path = MODEL_DIR / 'momentum_xgb_features_m30.pkl'

joblib.dump(model, model_path)
with open(features_path, 'wb') as f:
    pickle.dump(feature_cols, f)

log(f"Model saved to: {model_path}")
log(f"Features saved to: {features_path}")

log("\n" + "=" * 70)
log("DONE")
log("=" * 70)
