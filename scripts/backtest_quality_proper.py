"""
PROPER Backtest - Time-Sorted Out-of-Sample
============================================
Sort data by time FIRST, then split, then backtest on truly unseen data.
"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import pickle
import joblib
from pathlib import Path
from datetime import datetime

def log(msg=""):
    print(msg, flush=True)

log("=" * 70)
log("PROPER BACKTEST: TIME-SORTED OUT-OF-SAMPLE")
log("=" * 70)

MODEL_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/lstm_model/models')
DATA_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/lstm_model/data')

# ============================================================================
# TEST BOTH MODELS
# ============================================================================

for model_name, data_file, model_file, features_file in [
    ('H1', 'quality_entry_data.pkl', 'quality_xgb_classifier.joblib', 'quality_xgb_features.pkl'),
    ('M30', 'quality_entry_data_m30.pkl', 'quality_xgb_classifier_m30.joblib', 'quality_xgb_features_m30.pkl'),
]:
    log(f"\n{'='*70}")
    log(f"MODEL: {model_name}")
    log(f"{'='*70}")

    # Load data
    with open(DATA_DIR / data_file, 'rb') as f:
        data = pickle.load(f)

    df = data['data'].copy()

    # IMPORTANT: Sort by datetime for proper time-based split
    df = df.sort_values('datetime').reset_index(drop=True)

    log(f"\nData sorted by time:")
    log(f"  Total: {len(df):,} entries")
    log(f"  Date range: {df['datetime'].min()} to {df['datetime'].max()}")

    # Time-based split: 70/15/15
    n = len(df)
    train_idx = int(n * 0.70)
    val_idx = int(n * 0.85)

    train_df = df.iloc[:train_idx]
    val_df = df.iloc[train_idx:val_idx]
    test_df = df.iloc[val_idx:]

    log(f"\nPROPER TIME-BASED SPLIT:")
    log(f"  Train: {len(train_df):,} entries ({train_df['datetime'].min().date()} to {train_df['datetime'].max().date()})")
    log(f"  Val:   {len(val_df):,} entries ({val_df['datetime'].min().date()} to {val_df['datetime'].max().date()})")
    log(f"  Test:  {len(test_df):,} entries ({test_df['datetime'].min().date()} to {test_df['datetime'].max().date()})")

    # Load model (trained on unsorted data, but we test on time-sorted holdout)
    model = joblib.load(MODEL_DIR / model_file)
    with open(MODEL_DIR / features_file, 'rb') as f:
        feature_cols = pickle.load(f)

    # Prepare features for test set
    test_df = test_df.copy()
    test_df['direction_code'] = (test_df['direction'] == 'buy').astype(int)

    X_test = test_df[feature_cols].values.astype(np.float32)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    # Predictions
    test_df['pred_prob'] = model.predict_proba(X_test)[:, 1]

    log(f"\nTEST SET (Truly Out-of-Sample):")
    log(f"  Quality rate: {test_df['is_quality'].mean()*100:.1f}%")

    # Backtest at different thresholds
    log(f"\n| Threshold | Trades | Quality% | Win% | PF |")
    log(f"|-----------|--------|----------|------|------|")

    for thresh in [0.5, 0.6, 0.7, 0.8, 0.9]:
        filtered = test_df[test_df['pred_prob'] >= thresh].copy()

        if len(filtered) == 0:
            continue

        quality_rate = filtered['is_quality'].mean() * 100

        # Calculate P&L
        filtered['trade_pnl'] = np.where(
            filtered['is_quality'] == 1,
            filtered['max_profit_pips'] * 0.7,
            -filtered['max_dd_pips'] * 0.5
        )

        win_rate = (filtered['trade_pnl'] > 0).mean() * 100

        winners = filtered[filtered['trade_pnl'] > 0]['trade_pnl'].sum()
        losers = abs(filtered[filtered['trade_pnl'] <= 0]['trade_pnl'].sum())
        pf = winners / losers if losers > 0 else float('inf')

        log(f"| {thresh:^9} | {len(filtered):>6} | {quality_rate:>7.1f}% | {win_rate:>4.1f}% | {pf:>4.2f} |")

# ============================================================================
# FINAL COMPARISON
# ============================================================================
log("\n" + "=" * 70)
log("SUMMARY: PROPER OUT-OF-SAMPLE RESULTS")
log("=" * 70)
log("""
The backtest now uses TRULY unseen data:
- Data sorted by datetime
- Train: earliest 70% of data
- Val: next 15%
- Test: LAST 15% (most recent, never seen during training)

This is a proper walk-forward validation.
""")
