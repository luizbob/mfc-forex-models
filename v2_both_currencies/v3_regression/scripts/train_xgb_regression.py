"""
XGBoost Regression Model - Predict Net Pips (after spread)
==========================================================
Instead of classifying "profitable or not", predict actual expected pips.
Target: net_pips = exit_pnl_pips - spread

This allows threshold tuning based on expected profit, e.g.:
- Only trade when predicted net pips > 5
- Risk/reward optimization

Uses Exness Standard spreads.
"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import pickle
import joblib
from pathlib import Path
from datetime import datetime
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def log(msg=""):
    print(msg, flush=True)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Exness Standard spreads (pips)
SPREADS = {
    # Majors
    'AUDUSD': 0.9, 'EURUSD': 0.8, 'GBPUSD': 1.0, 'NZDUSD': 1.8,
    'USDCAD': 1.5, 'USDCHF': 1.3, 'USDJPY': 1.0,
    # Minors
    'AUDCAD': 2.2, 'AUDCHF': 0.9, 'AUDJPY': 1.9, 'AUDNZD': 2.0,
    'CADCHF': 0.8, 'CADJPY': 3.8, 'CHFJPY': 2.4,
    'EURAUD': 3.4, 'EURCAD': 2.9, 'EURCHF': 2.5, 'EURGBP': 1.4,
    'EURJPY': 2.4, 'EURNZD': 5.4,
    'GBPAUD': 2.5, 'GBPCAD': 4.8, 'GBPCHF': 2.4, 'GBPJPY': 2.2, 'GBPNZD': 5.8,
    'NZDCAD': 2.1, 'NZDCHF': 1.5, 'NZDJPY': 4.3,
}

# Time-series validation settings
TRAIN_YEARS = 4
VAL_MONTHS = 6
EMBARGO_HOURS = 24

# XGBoost parameters (start with tuned classifier params, adjust for regression)
XGB_PARAMS = {
    'n_estimators': 500,
    'max_depth': 4,
    'learning_rate': 0.05,
    'min_child_weight': 10,
    'subsample': 0.9,
    'colsample_bytree': 0.8,
    'gamma': 0.2,
    'random_state': 42,
    'tree_method': 'hist',
    'device': 'cuda',
    'objective': 'reg:squarederror',  # MSE loss
    'early_stopping_rounds': 20,
}

DATA_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/lstm_model/v2_both_currencies/data')
MODEL_DIR = Path(__file__).parent.parent / 'models'

log("=" * 70)
log("XGBOOST REGRESSION - PREDICT NET PIPS")
log("=" * 70)
log(f"Started: {datetime.now()}")

# ============================================================================
# LOAD AND PREPARE DATA
# ============================================================================

log("\nLoading data...")
with open(DATA_DIR / 'quality_entry_data_m5_v2.pkl', 'rb') as f:
    data = pickle.load(f)

df = data['data'].copy()
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.sort_values('datetime').reset_index(drop=True)

log(f"Total entries: {len(df):,}")
log(f"Date range: {df['datetime'].min().date()} to {df['datetime'].max().date()}")

# Fix JPY pips
jpy_mask = df['pair'].str.contains('JPY')
df.loc[jpy_mask, 'exit_pnl_pips'] = df.loc[jpy_mask, 'exit_pnl_pips'] / 100
df.loc[jpy_mask, 'max_dd_pips'] = df.loc[jpy_mask, 'max_dd_pips'] / 100

# Add spread and calculate net pips
df['spread'] = df['pair'].map(SPREADS)
df['net_pips'] = df['exit_pnl_pips'] - df['spread']

log(f"\nSpread statistics:")
log(f"  Mean spread: {df['spread'].mean():.2f} pips")
log(f"  Gross pips mean: {df['exit_pnl_pips'].mean():.2f}")
log(f"  Net pips mean: {df['net_pips'].mean():.2f}")

# Add time-based features
df['hour'] = df['datetime'].dt.hour
df['dayofweek'] = df['datetime'].dt.dayofweek
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 5)
df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 5)

# Add codes
df['direction_code'] = (df['direction'] == 'buy').astype(int)
df['trigger_code'] = (df['trigger'] == 'base').astype(int)

# Features (same as classifier)
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

# ============================================================================
# TIME-SERIES SPLIT
# ============================================================================

log("\n" + "=" * 70)
log("SETTING UP TIME-SERIES SPLITS")
log("=" * 70)

max_date = df['datetime'].max()
embargo = pd.Timedelta(hours=EMBARGO_HOURS)

# Training split (for final model)
final_train_start = max_date - pd.DateOffset(years=TRAIN_YEARS)
final_val_start = max_date - pd.DateOffset(months=VAL_MONTHS)
final_val_end = max_date - pd.DateOffset(months=3)
final_holdout_start = final_val_end

final_train_end = final_val_start - embargo
final_val_start_emb = final_val_start + embargo
final_val_end_emb = final_val_end - embargo

final_train_mask = (df['datetime'] >= final_train_start) & (df['datetime'] < final_train_end)
final_val_mask = (df['datetime'] >= final_val_start_emb) & (df['datetime'] < final_val_end_emb)
final_holdout_mask = df['datetime'] >= final_holdout_start

train_df = df[final_train_mask]
val_df = df[final_val_mask]
holdout_df = df[final_holdout_mask].copy()

log(f"Train:   {final_train_start.date()} to {final_train_end.date()} ({len(train_df):,} samples)")
log(f"Val:     {final_val_start_emb.date()} to {final_val_end_emb.date()} ({len(val_df):,} samples)")
log(f"Holdout: {final_holdout_start.date()} to {max_date.date()} ({len(holdout_df):,} samples)")

# Prepare arrays
X_train = train_df[feature_cols].values.astype(np.float32)
X_val = val_df[feature_cols].values.astype(np.float32)
X_holdout = holdout_df[feature_cols].values.astype(np.float32)

y_train = train_df['net_pips'].values.astype(np.float32)
y_val = val_df['net_pips'].values.astype(np.float32)

X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
X_holdout = np.nan_to_num(X_holdout, nan=0.0, posinf=0.0, neginf=0.0)

log(f"\nTarget (net_pips) statistics:")
log(f"  Train - mean: {y_train.mean():.2f}, std: {y_train.std():.2f}, min: {y_train.min():.2f}, max: {y_train.max():.2f}")
log(f"  Val   - mean: {y_val.mean():.2f}, std: {y_val.std():.2f}")

# ============================================================================
# TRAIN REGRESSION MODEL
# ============================================================================

log("\n" + "=" * 70)
log("TRAINING XGBOOST REGRESSOR")
log("=" * 70)

model = XGBRegressor(**XGB_PARAMS)
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

log(f"Best iteration: {model.best_iteration}")

# Validation metrics
y_val_pred = model.predict(X_val)
val_mae = mean_absolute_error(y_val, y_val_pred)
val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
val_r2 = r2_score(y_val, y_val_pred)

log(f"\nValidation Metrics:")
log(f"  MAE:  {val_mae:.3f} pips")
log(f"  RMSE: {val_rmse:.3f} pips")
log(f"  R²:   {val_r2:.4f}")

# ============================================================================
# EVALUATE ON HOLDOUT
# ============================================================================

log("\n" + "=" * 70)
log("HOLDOUT EVALUATION")
log("=" * 70)

holdout_df['pred_net_pips'] = model.predict(X_holdout)

# Evaluate at different prediction thresholds
def evaluate_threshold(df, pred_col, threshold):
    """Evaluate trades where predicted net pips >= threshold."""
    filtered = df[df[pred_col] >= threshold]
    if len(filtered) == 0:
        return {'trades': 0, 'wr': 0, 'pf': 0, 'net_pnl': 0, 'avg_pred': 0, 'avg_actual': 0}

    actual = filtered['net_pips']
    pred = filtered[pred_col]

    trades = len(filtered)
    wr = (actual > 0).mean() * 100
    winners = actual[actual > 0].sum()
    losers = abs(actual[actual <= 0].sum())
    pf = winners / losers if losers > 0 else 100
    net_pnl = actual.sum()
    avg_pred = pred.mean()
    avg_actual = actual.mean()

    return {
        'trades': trades,
        'wr': wr,
        'pf': pf,
        'net_pnl': net_pnl,
        'avg_pred': avg_pred,
        'avg_actual': avg_actual,
    }

log(f"\nHoldout period: {holdout_df['datetime'].min().date()} to {holdout_df['datetime'].max().date()}")
log(f"Total samples: {len(holdout_df):,}")

log(f"\n{'Threshold':>10} {'Trades':>8} {'WR%':>8} {'PF':>8} {'Net PnL':>12} {'Avg Pred':>10} {'Avg Actual':>12}")
log("-" * 80)

for thresh in [0, 2, 4, 6, 8, 10, 12, 15]:
    r = evaluate_threshold(holdout_df, 'pred_net_pips', thresh)
    if r['trades'] > 0:
        log(f"{thresh:>10} {r['trades']:>8,} {r['wr']:>7.1f}% {r['pf']:>8.2f} {r['net_pnl']:>+12,.0f} {r['avg_pred']:>10.1f} {r['avg_actual']:>+12.1f}")

# ============================================================================
# MONTHLY BREAKDOWN
# ============================================================================

log("\n" + "=" * 70)
log("MONTHLY BREAKDOWN (pred >= 6 pips threshold)")
log("=" * 70)

holdout_df['month'] = holdout_df['datetime'].dt.to_period('M')

log(f"\n{'Month':<10} {'Trades':>8} {'WR%':>8} {'PF':>8} {'Net PnL':>12} {'Avg Actual':>12}")
log("-" * 65)

for month in sorted(holdout_df['month'].unique()):
    month_df = holdout_df[holdout_df['month'] == month]
    r = evaluate_threshold(month_df, 'pred_net_pips', 6)
    if r['trades'] > 0:
        log(f"{str(month):<10} {r['trades']:>8,} {r['wr']:>7.1f}% {r['pf']:>8.2f} {r['net_pnl']:>+12,.0f} {r['avg_actual']:>+12.1f}")

# Total
r = evaluate_threshold(holdout_df, 'pred_net_pips', 6)
log("-" * 65)
log(f"{'TOTAL':<10} {r['trades']:>8,} {r['wr']:>7.1f}% {r['pf']:>8.2f} {r['net_pnl']:>+12,.0f} {r['avg_actual']:>+12.1f}")

# ============================================================================
# BY PAIR BREAKDOWN
# ============================================================================

log("\n" + "=" * 70)
log("BY PAIR (pred >= 6 pips threshold)")
log("=" * 70)

log(f"\n{'Pair':<12} {'Trades':>8} {'WR%':>8} {'PF':>8} {'Net PnL':>10} {'Avg Actual':>12}")
log("-" * 60)

pair_results = []
for pair in sorted(holdout_df['pair'].unique()):
    pair_df = holdout_df[holdout_df['pair'] == pair]
    r = evaluate_threshold(pair_df, 'pred_net_pips', 6)
    if r['trades'] > 0:
        pair_results.append((pair, r['trades'], r['wr'], r['pf'], r['net_pnl'], r['avg_actual']))

for pair, trades, wr, pf, net_pnl, avg_actual in sorted(pair_results, key=lambda x: x[4], reverse=True):
    log(f"{pair:<12} {trades:>8,} {wr:>7.1f}% {pf:>8.2f} {net_pnl:>+10,.0f} {avg_actual:>+12.1f}")

# ============================================================================
# COMPARE WITH CLASSIFIER
# ============================================================================

log("\n" + "=" * 70)
log("COMPARISON: REGRESSION vs CLASSIFICATION (@ 0.75 threshold)")
log("=" * 70)

# Load classifier for comparison
try:
    classifier = joblib.load(DATA_DIR.parent / 'models' / 'quality_xgb_m5_v2_tuned.joblib')
    holdout_df['pred_class'] = classifier.predict_proba(X_holdout)[:, 1]

    # Compare at similar trade counts
    # Find regression threshold that gives ~2300 trades (similar to classifier @ 0.75)
    class_r = evaluate_threshold(holdout_df, 'pred_class', 0.75)

    log(f"\n{'Model':<15} {'Thresh':>8} {'Trades':>8} {'WR%':>8} {'PF':>8} {'Net PnL':>12} {'Pips/Trade':>12}")
    log("-" * 80)

    log(f"{'Classifier':<15} {'0.75':>8} {class_r['trades']:>8,} {class_r['wr']:>7.1f}% {class_r['pf']:>8.2f} {class_r['net_pnl']:>+12,.0f} {class_r['avg_actual']:>+12.1f}")

    # Find best regression thresholds
    for thresh in [4, 6, 8, 10]:
        reg_r = evaluate_threshold(holdout_df, 'pred_net_pips', thresh)
        if reg_r['trades'] > 0:
            log(f"{'Regression':<15} {thresh:>8} {reg_r['trades']:>8,} {reg_r['wr']:>7.1f}% {reg_r['pf']:>8.2f} {reg_r['net_pnl']:>+12,.0f} {reg_r['avg_actual']:>+12.1f}")

except Exception as e:
    log(f"Could not load classifier for comparison: {e}")

# ============================================================================
# SAVE MODEL
# ============================================================================

log("\n" + "=" * 70)
log("SAVING MODEL")
log("=" * 70)

model_path = MODEL_DIR / 'quality_xgb_m5_regression.joblib'
features_path = MODEL_DIR / 'quality_xgb_features_m5_regression.pkl'

joblib.dump(model, model_path)
with open(features_path, 'wb') as f:
    pickle.dump(feature_cols, f)

# Save spreads dict
spreads_path = MODEL_DIR / 'spreads_exness_standard.pkl'
with open(spreads_path, 'wb') as f:
    pickle.dump(SPREADS, f)

log(f"Model saved to: {model_path}")
log(f"Features saved to: {features_path}")
log(f"Spreads saved to: {spreads_path}")

log(f"\n" + "=" * 70)
log(f"Completed: {datetime.now()}")
