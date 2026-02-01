"""
Evaluate Current vs Tuned Model on TRUE HOLDOUT
================================================
Holdout period: last 3 months (~Sep-Dec 2025)
This data was NEVER seen by either model during training or validation.
"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import pickle
import joblib
from pathlib import Path
from xgboost import XGBClassifier

DATA_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/lstm_model/v2_both_currencies/data')
MODEL_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/lstm_model/v2_both_currencies/models')

print("=" * 70)
print("HOLDOUT EVALUATION: Current vs Tuned Model")
print("=" * 70)

# Load data
print("\nLoading data...")
with open(DATA_DIR / 'quality_entry_data_m5_v2.pkl', 'rb') as f:
    data = pickle.load(f)

df = data['data'].copy()
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.sort_values('datetime').reset_index(drop=True)

# Fix JPY pips
jpy_mask = df['pair'].str.contains('JPY')
df.loc[jpy_mask, 'exit_pnl_pips'] = df.loc[jpy_mask, 'exit_pnl_pips'] / 100
df.loc[jpy_mask, 'max_dd_pips'] = df.loc[jpy_mask, 'max_dd_pips'] / 100

# Create target
df['is_profitable'] = (df['exit_pnl_pips'] > 0).astype(int)

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

# Training period setup
max_date = df['datetime'].max()
TRAIN_YEARS = 4
EMBARGO_HOURS = 24

final_train_start = max_date - pd.DateOffset(years=TRAIN_YEARS)
final_val_start = max_date - pd.DateOffset(months=6)
final_val_end = max_date - pd.DateOffset(months=3)

final_train_end = final_val_start - pd.Timedelta(hours=EMBARGO_HOURS)
final_val_start_emb = final_val_start + pd.Timedelta(hours=EMBARGO_HOURS)
final_val_end_emb = final_val_end - pd.Timedelta(hours=EMBARGO_HOURS)

final_train_mask = (df['datetime'] >= final_train_start) & (df['datetime'] < final_train_end)
final_val_mask = (df['datetime'] >= final_val_start_emb) & (df['datetime'] < final_val_end_emb)

final_train_df = df[final_train_mask]
final_val_df = df[final_val_mask]

# Holdout period: last 3 months (after validation ends)
holdout_start = final_val_end
holdout_mask = df['datetime'] >= holdout_start
holdout_df = df[holdout_mask].copy()

print(f"Date range: {df['datetime'].min().date()} to {df['datetime'].max().date()}")
print(f"Training data: {final_train_start.date()} to {final_train_end.date()} ({len(final_train_df):,} samples)")
print(f"Validation data: {final_val_start_emb.date()} to {final_val_end_emb.date()} ({len(final_val_df):,} samples)")
print(f"Holdout period: {holdout_df['datetime'].min().date()} to {holdout_df['datetime'].max().date()} ({len(holdout_df):,} samples)")

# Prepare training data
X_train = final_train_df[feature_cols].values.astype(np.float32)
X_val = final_val_df[feature_cols].values.astype(np.float32)
y_train = final_train_df['is_profitable'].values
y_val = final_val_df['is_profitable'].values

X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)

# Prepare holdout data
X_holdout = holdout_df[feature_cols].values.astype(np.float32)
X_holdout = np.nan_to_num(X_holdout, nan=0.0, posinf=0.0, neginf=0.0)

# Fixed params (GPU)
FIXED_PARAMS = {
    'n_estimators': 500,
    'random_state': 42,
    'eval_metric': 'auc',
    'early_stopping_rounds': 20,
    'tree_method': 'hist',
    'device': 'cuda',
}

# Current model params (from tuning script)
current_params = {
    'max_depth': 8,
    'learning_rate': 0.1,
    'min_child_weight': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.9,
    'gamma': 0.1,
}

# Tuned model params (best from grid search)
tuned_params = {
    'max_depth': 4,
    'learning_rate': 0.05,
    'min_child_weight': 10,
    'subsample': 0.9,
    'colsample_bytree': 0.8,
    'gamma': 0.2,
}

print("\nTraining CURRENT model...")
current_model = XGBClassifier(**{**FIXED_PARAMS, **current_params})
current_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

print("Training TUNED model...")
tuned_model = XGBClassifier(**{**FIXED_PARAMS, **tuned_params})
tuned_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

# Get predictions
print("Generating predictions on holdout...")
holdout_df['pred_current'] = current_model.predict_proba(X_holdout)[:, 1]
holdout_df['pred_tuned'] = tuned_model.predict_proba(X_holdout)[:, 1]

# Evaluate function
def evaluate_at_threshold(df, pred_col, threshold):
    filtered = df[df[pred_col] >= threshold]
    if len(filtered) == 0:
        return {'trades': 0, 'wr': 0, 'pnl': 0, 'pf': 0, 'winners': 0, 'losers': 0}

    pnl = filtered['exit_pnl_pips']
    trades = len(filtered)
    wr = (pnl > 0).mean() * 100
    total_pnl = pnl.sum()
    winners = pnl[pnl > 0].sum()
    losers = abs(pnl[pnl <= 0].sum())
    pf = winners / losers if losers > 0 else 100.0

    return {
        'trades': trades,
        'wr': wr,
        'pnl': total_pnl,
        'pf': pf,
        'winners': winners,
        'losers': losers,
    }

# Evaluate at multiple thresholds
thresholds = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85]

print("\n" + "=" * 70)
print("CURRENT MODEL - Holdout Performance")
print("=" * 70)
print(f"{'Thresh':<8} {'Trades':>8} {'WR%':>8} {'PF':>8} {'PnL':>12} {'Winners':>12} {'Losers':>10}")
print("-" * 75)

for thresh in thresholds:
    r = evaluate_at_threshold(holdout_df, 'pred_current', thresh)
    print(f"{thresh:<8} {r['trades']:>8,} {r['wr']:>7.1f}% {r['pf']:>8.2f} {r['pnl']:>+12,.0f} {r['winners']:>+12,.0f} {r['losers']:>10,.0f}")

print("\n" + "=" * 70)
print("TUNED MODEL - Holdout Performance")
print("=" * 70)
print(f"{'Thresh':<8} {'Trades':>8} {'WR%':>8} {'PF':>8} {'PnL':>12} {'Winners':>12} {'Losers':>10}")
print("-" * 75)

for thresh in thresholds:
    r = evaluate_at_threshold(holdout_df, 'pred_tuned', thresh)
    print(f"{thresh:<8} {r['trades']:>8,} {r['wr']:>7.1f}% {r['pf']:>8.2f} {r['pnl']:>+12,.0f} {r['winners']:>+12,.0f} {r['losers']:>10,.0f}")

# Side by side comparison at key threshold
print("\n" + "=" * 70)
print("SIDE-BY-SIDE COMPARISON @ 0.75 THRESHOLD")
print("=" * 70)

curr = evaluate_at_threshold(holdout_df, 'pred_current', 0.75)
tuned = evaluate_at_threshold(holdout_df, 'pred_tuned', 0.75)

print(f"\n{'Metric':<15} {'Current':>12} {'Tuned':>12} {'Diff':>12}")
print("-" * 55)
print(f"{'Trades':<15} {curr['trades']:>12,} {tuned['trades']:>12,} {tuned['trades'] - curr['trades']:>+12,}")
print(f"{'Win Rate':<15} {curr['wr']:>11.1f}% {tuned['wr']:>11.1f}% {tuned['wr'] - curr['wr']:>+11.1f}%")
print(f"{'Profit Factor':<15} {curr['pf']:>12.2f} {tuned['pf']:>12.2f} {tuned['pf'] - curr['pf']:>+12.2f}")
print(f"{'PnL (pips)':<15} {curr['pnl']:>+12,.0f} {tuned['pnl']:>+12,.0f} {tuned['pnl'] - curr['pnl']:>+12,.0f}")

# Monthly breakdown
print("\n" + "=" * 70)
print("MONTHLY BREAKDOWN @ 0.75 THRESHOLD")
print("=" * 70)

holdout_df['month'] = holdout_df['datetime'].dt.to_period('M')

print(f"\n{'Month':<10} {'Model':<10} {'Trades':>8} {'WR%':>8} {'PF':>8} {'PnL':>10}")
print("-" * 60)

for month in sorted(holdout_df['month'].unique()):
    month_df = holdout_df[holdout_df['month'] == month]

    for model_name, pred_col in [('Current', 'pred_current'), ('Tuned', 'pred_tuned')]:
        r = evaluate_at_threshold(month_df, pred_col, 0.75)
        if r['trades'] > 0:
            print(f"{str(month):<10} {model_name:<10} {r['trades']:>8,} {r['wr']:>7.1f}% {r['pf']:>8.2f} {r['pnl']:>+10,.0f}")
    print()

# By pair breakdown
print("\n" + "=" * 70)
print("BY PAIR BREAKDOWN @ 0.75 THRESHOLD (Tuned Model)")
print("=" * 70)

print(f"\n{'Pair':<12} {'Trades':>8} {'WR%':>8} {'PF':>8} {'PnL':>10}")
print("-" * 50)

for pair in sorted(holdout_df['pair'].unique()):
    pair_df = holdout_df[holdout_df['pair'] == pair]
    r = evaluate_at_threshold(pair_df, 'pred_tuned', 0.75)
    if r['trades'] > 0:
        print(f"{pair:<12} {r['trades']:>8,} {r['wr']:>7.1f}% {r['pf']:>8.2f} {r['pnl']:>+10,.0f}")

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)

# Calculate which is better
curr_75 = evaluate_at_threshold(holdout_df, 'pred_current', 0.75)
tuned_75 = evaluate_at_threshold(holdout_df, 'pred_tuned', 0.75)

if tuned_75['pf'] > curr_75['pf'] and tuned_75['trades'] > 100:
    print("\n✓ Tuned model CONFIRMS better on true holdout data")
    print(f"  - Higher PF: {tuned_75['pf']:.2f} vs {curr_75['pf']:.2f}")
    print(f"  - Higher WR: {tuned_75['wr']:.1f}% vs {curr_75['wr']:.1f}%")
    if tuned_75['pnl'] < curr_75['pnl']:
        print(f"  - Lower PnL due to fewer trades ({tuned_75['trades']:,} vs {curr_75['trades']:,})")
        print(f"  - But PnL per trade: {tuned_75['pnl']/tuned_75['trades']:.1f} vs {curr_75['pnl']/curr_75['trades']:.1f} pips")
else:
    print("\n✗ Tuned model does NOT outperform on holdout")
    print(f"  Current: PF={curr_75['pf']:.2f}, WR={curr_75['wr']:.1f}%, Trades={curr_75['trades']:,}")
    print(f"  Tuned:   PF={tuned_75['pf']:.2f}, WR={tuned_75['wr']:.1f}%, Trades={tuned_75['trades']:,}")

print()
