"""
Hyperparameter Tuning V3 - More Complex Models
===============================================
Explores MORE COMPLEX models with LOWER learning rates:
- Higher max_depth (8, 10, 12)
- Lower learning_rate (0.01, 0.02, 0.03)
- Lower min_child_weight (3, 5)
- More n_estimators (1500) to compensate for low LR
- Lower gamma (0, 0.05)

Theory: Current best model (max_depth=4) might be too simple.
A deeper model with slower learning could capture more patterns.

DOES NOT modify existing models - saves to new file.
Saves to: quality_xgb_m5_v2_tuned_v3.joblib
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
from itertools import product

def log(msg=""):
    print(msg, flush=True)

# ============================================================================
# CONFIGURATION - V3 (MORE COMPLEX + LOWER LR)
# ============================================================================

# Time-series validation settings
TRAIN_YEARS = 4
VAL_MONTHS = 6
N_VALIDATION_FOLDS = 3
EMBARGO_HOURS = 24

# V3 HYPERPARAMETER GRID - More complex models
PARAM_GRID = {
    'max_depth': [8, 10, 12],  # DEEPER trees
    'learning_rate': [0.01, 0.02, 0.03],  # LOWER learning rate
    'min_child_weight': [3, 5],  # Lower than original (was 10)
    'gamma': [0, 0.05],  # Lower than original (was 0.2)
    'subsample': [0.8, 0.9],  # Row subsampling
    'colsample_bytree': [0.7, 0.8],  # Column subsampling
}

# Fixed parameters
FIXED_PARAMS = {
    'n_estimators': 1500,  # MORE trees for low learning rate
    'reg_alpha': 0,
    'reg_lambda': 1,
    'random_state': 42,
    'eval_metric': 'auc',
    'early_stopping_rounds': 50,  # More patience for slow learning
    'tree_method': 'hist',
    'device': 'cuda',
}

# Evaluation thresholds
EVAL_THRESHOLDS = [0.70, 0.75, 0.80]
THRESHOLD_WEIGHTS = [0.25, 0.50, 0.25]
MIN_TRADES_PER_THRESHOLD = 500

log("=" * 70)
log("HYPERPARAMETER TUNING V3 - MORE COMPLEX MODELS")
log("=" * 70)
log(f"Started: {datetime.now()}")
log(f"\nExploring: deeper trees + lower learning rate")
log(f"n_estimators: {FIXED_PARAMS['n_estimators']} (with early_stopping_rounds={FIXED_PARAMS['early_stopping_rounds']})")

DATA_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/lstm_model/v2_both_currencies/data')
MODEL_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/lstm_model/v2_both_currencies/models')

# ============================================================================
# LOAD AND PREPARE DATA
# ============================================================================

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

# Add time features
df['hour'] = df['datetime'].dt.hour
df['dayofweek'] = df['datetime'].dt.dayofweek
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 5)
df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 5)

# Direction and trigger codes
df['direction_code'] = (df['direction'] == 'buy').astype(int)
trigger_map = {'USD': 0, 'EUR': 1, 'GBP': 2, 'JPY': 3, 'AUD': 4, 'NZD': 5, 'CAD': 6, 'CHF': 7}
df['trigger_code'] = df['trigger_ccy'].map(trigger_map)

log(f"Date range: {df['datetime'].min().date()} to {df['datetime'].max().date()}")

# Positive class rate
pos_rate = df['is_profitable'].mean()
log(f"Positive class rate: {pos_rate:.1%} (1={df['is_profitable'].sum():,}, 0={(~df['is_profitable'].astype(bool)).sum():,})")

# Features
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
# TIME-SERIES CROSS-VALIDATION FOLDS
# ============================================================================

log("\n" + "=" * 70)
log("SETTING UP TIME-SERIES VALIDATION FOLDS")
log("=" * 70)

max_date = df['datetime'].max()
folds = []

for i in range(N_VALIDATION_FOLDS):
    # Work backwards from max_date
    test_end = max_date - pd.DateOffset(years=i)
    test_start = test_end - pd.DateOffset(years=1)

    val_end = test_start - pd.Timedelta(hours=EMBARGO_HOURS)
    val_start = val_end - pd.DateOffset(months=VAL_MONTHS)

    train_end = val_start - pd.Timedelta(hours=EMBARGO_HOURS)
    train_start = train_end - pd.DateOffset(years=TRAIN_YEARS)

    train_mask = (df['datetime'] >= train_start) & (df['datetime'] < train_end)
    val_mask = (df['datetime'] >= val_start) & (df['datetime'] < val_end)
    test_mask = (df['datetime'] >= test_start) & (df['datetime'] < test_end)

    train_idx = df[train_mask].index.tolist()
    val_idx = df[val_mask].index.tolist()
    test_idx = df[test_mask].index.tolist()

    if len(train_idx) > 0 and len(val_idx) > 0 and len(test_idx) > 0:
        test_year = test_end.year
        folds.append({
            'train_idx': train_idx,
            'val_idx': val_idx,
            'test_idx': test_idx,
            'test_year': test_year,
        })
        log(f"Fold {test_year}: Train {len(train_idx):,}, Val {len(val_idx):,}, Test {len(test_idx):,}")

# Prepare feature arrays
X = df[feature_cols].values.astype(np.float32)
y = df['is_profitable'].values
pnl = df['exit_pnl_pips'].values

# Handle NaN/Inf
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

# ============================================================================
# EVALUATION FUNCTION
# ============================================================================

def evaluate_model(model, X_test, y_test, pnl_test, thresholds=EVAL_THRESHOLDS):
    """Evaluate model at multiple thresholds."""
    probs = model.predict_proba(X_test)[:, 1]

    results = {}
    for thresh in thresholds:
        mask = probs >= thresh
        trades = mask.sum()

        if trades < MIN_TRADES_PER_THRESHOLD:
            results[thresh] = {'trades': trades, 'pf': 0, 'wr': 0, 'pnl': 0}
            continue

        trade_pnl = pnl_test[mask]
        winners = trade_pnl[trade_pnl > 0].sum()
        losers = abs(trade_pnl[trade_pnl < 0].sum())

        pf = winners / losers if losers > 0 else 10.0
        wr = (trade_pnl > 0).mean() * 100
        total_pnl = trade_pnl.sum()

        results[thresh] = {
            'trades': trades,
            'pf': pf,
            'wr': wr,
            'pnl': total_pnl,
            'winners': winners,
            'losers': losers,
        }

    return results

def calculate_weighted_pf(results, weights=THRESHOLD_WEIGHTS, thresholds=EVAL_THRESHOLDS):
    """Calculate weighted PF score across thresholds."""
    weighted_pf = 0
    total_weight = 0

    for thresh, weight in zip(thresholds, weights):
        if thresh in results and results[thresh]['trades'] >= MIN_TRADES_PER_THRESHOLD:
            weighted_pf += results[thresh]['pf'] * weight
            total_weight += weight

    return weighted_pf / total_weight if total_weight > 0 else 0

# ============================================================================
# GRID SEARCH V3
# ============================================================================

log("\n" + "=" * 70)
log("GRID SEARCH V3 (deeper trees + lower learning rate)")
log("=" * 70)

param_names = list(PARAM_GRID.keys())
param_values = list(PARAM_GRID.values())
all_combinations = list(product(*param_values))

log(f"\nTotal combinations to test: {len(all_combinations)}")
log(f"Parameters: {param_names}")

best_result = None
best_params = None
best_weighted_pf = 0
all_results = []

for combo_idx, combo in enumerate(all_combinations):
    params = dict(zip(param_names, combo))
    full_params = {**FIXED_PARAMS, **params}

    if (combo_idx + 1) % 10 == 1:
        log(f"\nTesting combination {combo_idx + 1}/{len(all_combinations)}...")

    # Evaluate across all folds
    fold_results = []

    for fold in folds:
        X_train = X[fold['train_idx']]
        y_train = y[fold['train_idx']]
        X_val = X[fold['val_idx']]
        y_val = y[fold['val_idx']]
        X_test = X[fold['test_idx']]
        y_test = y[fold['test_idx']]
        pnl_test = pnl[fold['test_idx']]

        try:
            model = XGBClassifier(**full_params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

            results = evaluate_model(model, X_test, y_test, pnl_test)
            fold_results.append(results)
        except Exception as e:
            log(f"  Error: {e}")
            fold_results.append({})

    # Pool results across folds
    pooled = {}
    for thresh in EVAL_THRESHOLDS:
        trades = sum(r.get(thresh, {}).get('trades', 0) for r in fold_results)
        winners = sum(r.get(thresh, {}).get('winners', 0) for r in fold_results)
        losers = sum(r.get(thresh, {}).get('losers', 0) for r in fold_results)
        pnl_total = sum(r.get(thresh, {}).get('pnl', 0) for r in fold_results)

        pf = winners / losers if losers > 0 else 0
        wr = 0
        for r in fold_results:
            if thresh in r and r[thresh]['trades'] > 0:
                wr += r[thresh]['wr'] * r[thresh]['trades']
        wr = wr / trades if trades > 0 else 0

        pooled[thresh] = {'trades': trades, 'pf': pf, 'wr': wr, 'pnl': pnl_total, 'winners': winners, 'losers': losers}

    weighted_pf = calculate_weighted_pf(pooled)

    result = {
        'params': params,
        'weighted_pf': weighted_pf,
        'pooled': pooled,
        'fold_results': fold_results,
    }
    all_results.append(result)

    # Check if best
    if weighted_pf > best_weighted_pf:
        best_weighted_pf = weighted_pf
        best_params = params
        best_result = result

        trades_str = "/".join([str(pooled[t]['trades']) for t in EVAL_THRESHOLDS])
        log(f"  NEW BEST: Weighted PF={weighted_pf:.2f}, WR={pooled[0.75]['wr']:.1f}%, Trades@.70/.75/.80={trades_str}")

# ============================================================================
# RESULTS SUMMARY
# ============================================================================

log("\n" + "=" * 70)
log("TOP 10 PARAMETER COMBINATIONS")
log("=" * 70)

sorted_results = sorted(all_results, key=lambda x: x['weighted_pf'], reverse=True)[:10]

log(f"\n{'Rank':<5} {'Wtd PF':>8} {'PF@.75':>8} {'WR%':>7} {'Trades':>10} | Parameters")
log("-" * 90)

for rank, res in enumerate(sorted_results, 1):
    pooled = res['pooled']
    params = res['params']
    params_str = ", ".join([f"{k}={v}" for k, v in params.items()])
    log(f"{rank:<5} {res['weighted_pf']:>8.2f} {pooled[0.75]['pf']:>8.2f} {pooled[0.75]['wr']:>6.1f}% {pooled[0.75]['trades']:>10,} | {params_str}")

# Best model details
log("\n" + "=" * 70)
log("BEST MODEL DETAILS (V3 Tuning)")
log("=" * 70)

log("\nBest Parameters:")
for k, v in best_params.items():
    log(f"  {k}: {v}")

log("\nPerformance by Threshold:")
log(f"{'Thresh':<8} {'Pooled PF':>10} {'Trades':>10} {'WR%':>8} {'PnL':>12}")
log("-" * 55)
for thresh in EVAL_THRESHOLDS:
    p = best_result['pooled'][thresh]
    log(f"{thresh:<8} {p['pf']:>10.2f} {p['trades']:>10,} {p['wr']:>7.1f}% {p['pnl']:>+12,.0f}")

log("\nPerformance by Fold (at threshold 0.75):")
log(f"{'Year':<8} {'Trades':>10} {'WR%':>8} {'PnL':>12} {'PF':>8}")
log("-" * 50)
for fold, fold_result in zip(folds, best_result['fold_results']):
    if 0.75 in fold_result:
        r = fold_result[0.75]
        log(f"{fold['test_year']:<8} {r['trades']:>10,} {r['wr']:>7.1f}% {r['pnl']:>+12,.0f} {r['pf']:>8.2f}")

# ============================================================================
# COMPARE WITH ORIGINAL TUNED MODEL
# ============================================================================

log("\n" + "=" * 70)
log("COMPARISON WITH ORIGINAL TUNED MODEL")
log("=" * 70)

# Original tuned results (from tune_output.log)
original_tuned = {
    'weighted_pf': 7.18,
    'pf_75': 5.97,
    'wr_75': 80.6,
    'trades_75': 31368,
    'pnl_75': 448491,
}

log(f"\n{'Model':<20} {'Wtd PF':>8} {'PF@.75':>8} {'WR%':>7} {'Trades':>10} {'PnL':>12}")
log("-" * 75)
log(f"{'Original Tuned':<20} {original_tuned['weighted_pf']:>8.2f} {original_tuned['pf_75']:>8.2f} {original_tuned['wr_75']:>6.1f}% {original_tuned['trades_75']:>10,} {original_tuned['pnl_75']:>+12,}")
log(f"{'V3 Tuned':<20} {best_weighted_pf:>8.2f} {best_result['pooled'][0.75]['pf']:>8.2f} {best_result['pooled'][0.75]['wr']:>6.1f}% {best_result['pooled'][0.75]['trades']:>10,} {best_result['pooled'][0.75]['pnl']:>+12,.0f}")

improvement = (best_weighted_pf - original_tuned['weighted_pf']) / original_tuned['weighted_pf'] * 100
pnl_diff = (best_result['pooled'][0.75]['pnl'] - original_tuned['pnl_75']) / original_tuned['pnl_75'] * 100
log(f"\nImprovement over original: Weighted PF {improvement:+.1f}%, PnL {pnl_diff:+.1f}%")

# ============================================================================
# SAVE V3 MODEL
# ============================================================================

log("\n" + "=" * 70)
log("TRAINING FINAL V3 TUNED MODEL")
log("=" * 70)

# Train on most recent data split
max_date = df['datetime'].max()
final_train_start = max_date - pd.DateOffset(years=TRAIN_YEARS)
final_val_start = max_date - pd.DateOffset(months=VAL_MONTHS)
final_val_end = max_date - pd.DateOffset(months=3)

final_train_end = final_val_start - pd.Timedelta(hours=EMBARGO_HOURS)
final_val_start_emb = final_val_start + pd.Timedelta(hours=EMBARGO_HOURS)

train_mask = (df['datetime'] >= final_train_start) & (df['datetime'] < final_train_end)
val_mask = (df['datetime'] >= final_val_start_emb) & (df['datetime'] < final_val_end)

X_train_final = X[train_mask.values]
y_train_final = y[train_mask.values]
X_val_final = X[val_mask.values]
y_val_final = y[val_mask.values]

final_model = XGBClassifier(**{**FIXED_PARAMS, **best_params})
final_model.fit(X_train_final, y_train_final, eval_set=[(X_val_final, y_val_final)], verbose=False)

log(f"\nFinal model trained on {len(X_train_final):,} samples")
log(f"Best iteration: {final_model.best_iteration}")

# Save model
model_path = MODEL_DIR / 'quality_xgb_m5_v2_tuned_v3.joblib'
joblib.dump(final_model, model_path)
log(f"\nV3 Tuned model saved to: {model_path}")

# Save features
features_path = MODEL_DIR / 'quality_xgb_features_m5_v2_tuned_v3.pkl'
with open(features_path, 'wb') as f:
    pickle.dump(feature_cols, f)
log(f"Features saved to: {features_path}")

# Save tuning results
results_path = MODEL_DIR / 'tuning_results_m5_v3.pkl'
with open(results_path, 'wb') as f:
    pickle.dump({
        'best_params': best_params,
        'best_result': best_result,
        'all_results': all_results,
    }, f)
log(f"Tuning results saved to: {results_path}")

log("\n" + "=" * 70)
log(f"Completed: {datetime.now()}")
log("=" * 70)
