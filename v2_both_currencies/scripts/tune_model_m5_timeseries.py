"""
Hyperparameter Tuning with Proper Time-Series Validation
=========================================================
Uses walk-forward validation to tune hyperparameters.
Optimizes for Profit Factor, not just AUC.

DOES NOT modify existing models - saves to new file.
Saves to: quality_xgb_m5_v2_tuned.joblib
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
# CONFIGURATION
# ============================================================================

# Time-series validation settings
TRAIN_YEARS = 4
VAL_MONTHS = 6
N_VALIDATION_FOLDS = 3  # Use last 3 years for validation (2022, 2023, 2024)
EMBARGO_HOURS = 24  # 1-day embargo at boundaries (max hold ~14h, so 24h is safe)

# Hyperparameter grid
PARAM_GRID = {
    'max_depth': [4, 6, 8, 10],
    'learning_rate': [0.05, 0.1, 0.15],
    'min_child_weight': [3, 5, 10],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'gamma': [0, 0.1, 0.2],
}

# Fixed parameters (GPU accelerated)
FIXED_PARAMS = {
    'n_estimators': 500,
    'random_state': 42,
    'eval_metric': 'auc',
    'early_stopping_rounds': 20,
    'tree_method': 'hist',
    'device': 'cuda',
}

# Thresholds for evaluation (avoid "lucky at exactly one threshold")
EVAL_THRESHOLDS = [0.70, 0.75, 0.80]
THRESHOLD_WEIGHTS = [0.25, 0.50, 0.25]  # Weight toward 0.75
MIN_TRADES_PER_THRESHOLD = 500  # Require min trades at EACH threshold to avoid sparse results

log("=" * 70)
log("HYPERPARAMETER TUNING WITH TIME-SERIES VALIDATION")
log("=" * 70)
log(f"Started: {datetime.now()}")

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
df['datetime'] = pd.to_datetime(df['datetime'])
log(f"Date range: {df['datetime'].min().date()} to {df['datetime'].max().date()}")

# Add time-based features
df['hour'] = df['datetime'].dt.hour
df['dayofweek'] = df['datetime'].dt.dayofweek
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 5)
df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 5)

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

# Add codes
df['direction_code'] = (df['direction'] == 'buy').astype(int)
df['trigger_code'] = (df['trigger'] == 'base').astype(int)

# ============================================================================
# DEFINE TIME-SERIES VALIDATION FOLDS
# ============================================================================

log("\n" + "=" * 70)
log("SETTING UP TIME-SERIES VALIDATION FOLDS")
log("=" * 70)

# We'll use 2022, 2023, 2024 as validation years
# Each fold trains on 4 years before and tests on that year
# EMBARGO: 1-day gap at boundaries to prevent leakage from trades that begin before
# the split and resolve after it (max hold ~14h, so 24h embargo is safe)
validation_years = [2022, 2023, 2024]
embargo = pd.Timedelta(hours=EMBARGO_HOURS)

folds = []
for test_year in validation_years:
    test_start = pd.Timestamp(f'{test_year}-01-01')
    test_end = pd.Timestamp(f'{test_year + 1}-01-01')
    train_start = test_start - pd.DateOffset(years=TRAIN_YEARS)
    val_start = test_start - pd.DateOffset(months=VAL_MONTHS)

    # Apply embargo: gap between train/val and val/test
    train_end_emb = val_start - embargo
    val_start_emb = val_start + embargo
    val_end_emb = test_start - embargo
    test_start_emb = test_start + embargo

    train_mask = (df['datetime'] >= train_start) & (df['datetime'] < train_end_emb)
    val_mask = (df['datetime'] >= val_start_emb) & (df['datetime'] < val_end_emb)
    test_mask = (df['datetime'] >= test_start_emb) & (df['datetime'] < test_end)

    folds.append({
        'year': test_year,
        'train_idx': df[train_mask].index.tolist(),
        'val_idx': df[val_mask].index.tolist(),
        'test_idx': df[test_mask].index.tolist(),
    })

    log(f"Fold {test_year}: Train {len(folds[-1]['train_idx']):,}, Val {len(folds[-1]['val_idx']):,}, Test {len(folds[-1]['test_idx']):,} (with {EMBARGO_HOURS}h embargo)")

# ============================================================================
# EVALUATION FUNCTION
# ============================================================================

def evaluate_params(params, folds, df, feature_cols, thresholds=EVAL_THRESHOLDS, weights=THRESHOLD_WEIGHTS):
    """
    Evaluate a parameter set across all folds and multiple thresholds.
    Returns weighted pooled PF score to avoid "lucky at exactly one threshold".
    """
    all_params = {**FIXED_PARAMS, **params}

    # Store predictions for all folds (train once, evaluate at multiple thresholds)
    fold_predictions = []

    for fold in folds:
        # Get data
        train_df = df.loc[fold['train_idx']]
        val_df = df.loc[fold['val_idx']]
        test_df = df.loc[fold['test_idx']].copy()

        X_train = train_df[feature_cols].values.astype(np.float32)
        X_val = val_df[feature_cols].values.astype(np.float32)
        X_test = test_df[feature_cols].values.astype(np.float32)

        y_train = train_df['is_profitable'].values
        y_val = val_df['is_profitable'].values

        # Handle NaN/Inf
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

        # Train model (once per fold)
        model = XGBClassifier(**all_params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        # Predict on test
        test_df['pred'] = model.predict_proba(X_test)[:, 1]
        fold_predictions.append({
            'year': fold['year'],
            'test_df': test_df,
        })

    # Evaluate at multiple thresholds
    threshold_results = {}
    for thresh in thresholds:
        total_winners_pips = 0.0
        total_losers_pips = 0.0
        total_winning_trades = 0
        total_trades_all = 0
        fold_results = []

        for fp in fold_predictions:
            test_df = fp['test_df']
            filtered = test_df[test_df['pred'] >= thresh]

            if len(filtered) == 0:
                fold_results.append({
                    'year': fp['year'],
                    'n_trades': 0,
                    'n_winning_trades': 0,
                    'win_rate': 0,
                    'pnl': 0,
                    'pf': 0,
                    'winners': 0,
                    'losers': 0,
                })
                continue

            pnl = filtered['exit_pnl_pips']
            n_trades = len(filtered)
            n_winning_trades = (pnl > 0).sum()
            win_rate = (pnl > 0).mean() * 100
            total_pnl = pnl.sum()

            winners = pnl[pnl > 0].sum()
            losers = abs(pnl[pnl <= 0].sum())
            pf = winners / losers if losers > 0 else 100.0

            total_winners_pips += winners
            total_losers_pips += losers
            total_winning_trades += n_winning_trades
            total_trades_all += n_trades

            fold_results.append({
                'year': fp['year'],
                'n_trades': n_trades,
                'n_winning_trades': n_winning_trades,
                'win_rate': win_rate,
                'pnl': total_pnl,
                'pf': pf,
                'winners': winners,
                'losers': losers,
            })

        pooled_pf = total_winners_pips / total_losers_pips if total_losers_pips > 0 else 0.0
        # Trade-weighted win rate (not simple mean of fold win rates)
        weighted_wr = 100.0 * (total_winning_trades / total_trades_all) if total_trades_all > 0 else 0.0
        total_pnl = sum(r['pnl'] for r in fold_results)

        threshold_results[thresh] = {
            'pooled_pf': pooled_pf,
            'weighted_wr': weighted_wr,
            'total_pnl': total_pnl,
            'total_trades': total_trades_all,
            'total_winning_trades': total_winning_trades,
            'total_winners_pips': total_winners_pips,
            'total_losers_pips': total_losers_pips,
            'fold_results': fold_results,
        }

    # Calculate weighted score across thresholds
    weighted_pf = sum(
        threshold_results[t]['pooled_pf'] * w
        for t, w in zip(thresholds, weights)
    )

    # Check if all thresholds meet minimum trades requirement
    meets_min_trades = all(
        threshold_results[t]['total_trades'] >= MIN_TRADES_PER_THRESHOLD
        for t in thresholds
    )

    # Use 0.75 threshold results as primary (for display), but score by weighted
    primary = threshold_results[0.75]

    return {
        'weighted_pf': weighted_pf,
        'pooled_pf': primary['pooled_pf'],  # Keep for display (at 0.75)
        'weighted_wr': primary['weighted_wr'],  # Trade-weighted win rate
        'total_pnl': primary['total_pnl'],
        'total_trades': primary['total_trades'],
        'total_winners_pips': primary['total_winners_pips'],
        'total_losers_pips': primary['total_losers_pips'],
        'fold_results': primary['fold_results'],
        'threshold_results': threshold_results,  # Full results at all thresholds
        'meets_min_trades': meets_min_trades,  # True if all thresholds have enough trades
    }

# ============================================================================
# GRID SEARCH WITH TIME-SERIES VALIDATION
# ============================================================================

log("\n" + "=" * 70)
log("GRID SEARCH (this will take a while...)")
log("=" * 70)

# Generate all parameter combinations
param_names = list(PARAM_GRID.keys())
param_values = list(PARAM_GRID.values())
all_combinations = list(product(*param_values))

log(f"\nTotal combinations to test: {len(all_combinations)}")
log(f"Parameters: {param_names}")
log()

results = []
best_score = 0
best_params = None
best_result = None

for i, combo in enumerate(all_combinations):
    params = dict(zip(param_names, combo))

    # Progress update every 10 combos
    if i % 10 == 0:
        log(f"Testing combination {i+1}/{len(all_combinations)}...")

    try:
        result = evaluate_params(params, folds, df, feature_cols, EVAL_THRESHOLDS, THRESHOLD_WEIGHTS)
        result['params'] = params
        results.append(result)

        # Track best (using weighted PF across thresholds, require min trades at ALL thresholds)
        if result['weighted_pf'] > best_score and result['meets_min_trades']:
            best_score = result['weighted_pf']
            best_params = params
            best_result = result
            trades_str = "/".join(str(result['threshold_results'][t]['total_trades']) for t in EVAL_THRESHOLDS)
            log(f"  NEW BEST: Weighted PF={best_score:.2f} (0.75: {result['pooled_pf']:.2f}), WR={result['weighted_wr']:.1f}%, Trades@.70/.75/.80={trades_str}")

    except Exception as e:
        log(f"  Error with {params}: {e}")
        continue

# ============================================================================
# RESULTS
# ============================================================================

log("\n" + "=" * 70)
log("TOP 10 PARAMETER COMBINATIONS (by Weighted PF across thresholds)")
log("=" * 70)

# Sort by weighted PF (across multiple thresholds)
results_sorted = sorted(results, key=lambda x: x['weighted_pf'], reverse=True)

log(f"\n{'Rank':<5} {'Wtd PF':>8} {'PF@.70':>8} {'PF@.75':>8} {'PF@.80':>8} {'WR%':>7} {'Trades@.70/.75/.80':>20} | Parameters")
log("-" * 115)

for rank, r in enumerate(results_sorted[:10], 1):
    params_str = ", ".join(f"{k}={v}" for k, v in r['params'].items())
    pf_70 = r['threshold_results'][0.70]['pooled_pf']
    pf_75 = r['threshold_results'][0.75]['pooled_pf']
    pf_80 = r['threshold_results'][0.80]['pooled_pf']
    trades_70 = r['threshold_results'][0.70]['total_trades']
    trades_75 = r['threshold_results'][0.75]['total_trades']
    trades_80 = r['threshold_results'][0.80]['total_trades']
    trades_str = f"{trades_70}/{trades_75}/{trades_80}"
    log(f"{rank:<5} {r['weighted_pf']:>8.2f} {pf_70:>8.2f} {pf_75:>8.2f} {pf_80:>8.2f} {r['weighted_wr']:>6.1f}% {trades_str:>20} | {params_str}")

# ============================================================================
# BEST MODEL DETAILS
# ============================================================================

log("\n" + "=" * 70)
log("BEST MODEL DETAILS")
log("=" * 70)

log(f"\nBest Parameters:")
for k, v in best_params.items():
    log(f"  {k}: {v}")

log(f"\nPerformance by Threshold:")
log(f"{'Thresh':<8} {'Pooled PF':>10} {'Trades':>10} {'WR%':>8} {'PnL':>12}")
log("-" * 55)
for thresh in EVAL_THRESHOLDS:
    tr = best_result['threshold_results'][thresh]
    log(f"{thresh:<8} {tr['pooled_pf']:>10.2f} {tr['total_trades']:>10,} {tr['weighted_wr']:>7.1f}% {tr['total_pnl']:>+12,.0f}")

log(f"\nWeighted PF Score: {best_result['weighted_pf']:.2f} (weights: {THRESHOLD_WEIGHTS})")

log(f"\nPerformance by Fold (at threshold 0.75):")
log(f"{'Year':<6} {'Trades':>10} {'WR%':>8} {'PnL':>12} {'PF':>8} {'Winners':>12} {'Losers':>12}")
log("-" * 80)
for fr in best_result['fold_results']:
    log(f"{fr['year']:<6} {fr['n_trades']:>10,} {fr['win_rate']:>7.1f}% {fr['pnl']:>+12,.0f} {fr['pf']:>8.2f} {fr['winners']:>+12,.0f} {fr['losers']:>12,.0f}")

log(f"\nPooled PF @0.75: {best_result['pooled_pf']:.2f} (sum winners / sum losers = {best_result['total_winners_pips']:,.0f} / {best_result['total_losers_pips']:,.0f})")
log(f"Trade-Weighted WR: {best_result['weighted_wr']:.1f}%")
log(f"Total PnL:  {best_result['total_pnl']:+,.0f}")

# ============================================================================
# COMPARE WITH CURRENT MODEL
# ============================================================================

log("\n" + "=" * 70)
log("COMPARISON WITH CURRENT MODEL")
log("=" * 70)

current_params = {
    'max_depth': 8,
    'learning_rate': 0.1,
    'min_child_weight': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.9,
    'gamma': 0.1,
}

current_result = evaluate_params(current_params, folds, df, feature_cols, EVAL_THRESHOLDS, THRESHOLD_WEIGHTS)

log(f"\n{'Model':<15} {'Wtd PF':>8} {'PF@.70':>8} {'PF@.75':>8} {'PF@.80':>8} {'WR%':>7} {'Trades':>10} {'PnL':>12}")
log("-" * 95)
curr_70 = current_result['threshold_results'][0.70]['pooled_pf']
curr_75 = current_result['threshold_results'][0.75]['pooled_pf']
curr_80 = current_result['threshold_results'][0.80]['pooled_pf']
best_70 = best_result['threshold_results'][0.70]['pooled_pf']
best_75 = best_result['threshold_results'][0.75]['pooled_pf']
best_80 = best_result['threshold_results'][0.80]['pooled_pf']
log(f"{'Current':<15} {current_result['weighted_pf']:>8.2f} {curr_70:>8.2f} {curr_75:>8.2f} {curr_80:>8.2f} {current_result['weighted_wr']:>6.1f}% {current_result['total_trades']:>10,} {current_result['total_pnl']:>+12,.0f}")
log(f"{'Tuned':<15} {best_result['weighted_pf']:>8.2f} {best_70:>8.2f} {best_75:>8.2f} {best_80:>8.2f} {best_result['weighted_wr']:>6.1f}% {best_result['total_trades']:>10,} {best_result['total_pnl']:>+12,.0f}")

improvement_pf = (best_result['weighted_pf'] - current_result['weighted_pf']) / current_result['weighted_pf'] * 100 if current_result['weighted_pf'] > 0 else 0
improvement_pnl = (best_result['total_pnl'] - current_result['total_pnl']) / abs(current_result['total_pnl']) * 100 if current_result['total_pnl'] != 0 else 0

log(f"\nImprovement: Weighted PF {improvement_pf:+.1f}%, PnL {improvement_pnl:+.1f}%")

# ============================================================================
# TRAIN FINAL TUNED MODEL (on all available data)
# ============================================================================

if best_result['weighted_pf'] > current_result['weighted_pf']:
    log("\n" + "=" * 70)
    log("TRAINING FINAL TUNED MODEL")
    log("=" * 70)

    # TRUE HOLDOUT: last 3 months completely isolated (model never sees this data)
    # Train on 4 years, use months 6-3 for validation (early stopping)
    max_date = df['datetime'].max()
    final_train_start = max_date - pd.DateOffset(years=TRAIN_YEARS)
    final_val_start = max_date - pd.DateOffset(months=6)
    final_val_end = max_date - pd.DateOffset(months=3)
    final_holdout_start = final_val_end

    # Apply 1-day embargo at boundaries (max hold ~14h, so 24h is safe)
    EMBARGO_HOURS = 24
    final_train_end = final_val_start - pd.Timedelta(hours=EMBARGO_HOURS)
    final_val_start_emb = final_val_start + pd.Timedelta(hours=EMBARGO_HOURS)
    final_val_end_emb = final_val_end - pd.Timedelta(hours=EMBARGO_HOURS)

    final_train_mask = (df['datetime'] >= final_train_start) & (df['datetime'] < final_train_end)
    final_val_mask = (df['datetime'] >= final_val_start_emb) & (df['datetime'] < final_val_end_emb)
    final_holdout_mask = df['datetime'] >= final_holdout_start

    final_train_df = df[final_train_mask]
    final_val_df = df[final_val_mask]
    final_holdout_df = df[final_holdout_mask]

    X_train_final = final_train_df[feature_cols].values.astype(np.float32)
    X_val_final = final_val_df[feature_cols].values.astype(np.float32)

    y_train_final = final_train_df['is_profitable'].values
    y_val_final = final_val_df['is_profitable'].values

    X_train_final = np.nan_to_num(X_train_final, nan=0.0, posinf=0.0, neginf=0.0)
    X_val_final = np.nan_to_num(X_val_final, nan=0.0, posinf=0.0, neginf=0.0)

    final_params = {**FIXED_PARAMS, **best_params}
    final_model = XGBClassifier(**final_params)
    final_model.fit(
        X_train_final, y_train_final,
        eval_set=[(X_val_final, y_val_final)],
        verbose=False
    )

    log(f"\nFinal model trained on:")
    log(f"  Train:   {final_train_start.date()} to {final_train_end.date()} ({len(final_train_df):,} samples)")
    log(f"  Val:     {final_val_start_emb.date()} to {final_val_end_emb.date()} ({len(final_val_df):,} samples)")
    log(f"  Holdout: {final_holdout_start.date()} to {max_date.date()} ({len(final_holdout_df):,} samples) - TRUE HOLDOUT, never seen by model")

    # Save tuned model (NEW FILE - does not overwrite existing)
    model_path = MODEL_DIR / 'quality_xgb_m5_v2_tuned.joblib'
    features_path = MODEL_DIR / 'quality_xgb_features_m5_v2_tuned.pkl'

    joblib.dump(final_model, model_path)
    with open(features_path, 'wb') as f:
        pickle.dump(feature_cols, f)

    # Save tuning results
    tuning_results = {
        'best_params': best_params,
        'best_result': best_result,
        'all_results': results,
        'current_params': current_params,
        'current_result': current_result,
    }
    results_path = MODEL_DIR / 'tuning_results_m5.pkl'
    with open(results_path, 'wb') as f:
        pickle.dump(tuning_results, f)

    log(f"\nTuned model saved to: {model_path}")
    log(f"Features saved to: {features_path}")
    log(f"Tuning results saved to: {results_path}")
else:
    log("\nTuned model is NOT better than current. No new model saved.")

log(f"\n" + "=" * 70)
log(f"Completed: {datetime.now()}")
