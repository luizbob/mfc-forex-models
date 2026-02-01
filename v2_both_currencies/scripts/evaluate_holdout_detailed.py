"""
Evaluate Current vs Tuned Model on TRUE HOLDOUT - DETAILED VERSION
===================================================================
Includes intermediate thresholds (0.70 to 0.75 in 0.01 steps)
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

# Typical spreads in pips per pair
SPREADS = {
    'EURUSD': 1.5, 'GBPUSD': 2.0, 'USDJPY': 1.5, 'USDCHF': 2.0, 'USDCAD': 2.0,
    'AUDUSD': 1.5, 'NZDUSD': 2.5, 'EURGBP': 2.0, 'EURJPY': 2.0, 'GBPJPY': 3.0,
    'EURCHF': 2.5, 'AUDJPY': 2.5, 'EURAUD': 3.0, 'EURCAD': 3.0, 'EURNZD': 4.0,
    'GBPAUD': 3.5, 'GBPCAD': 3.5, 'GBPCHF': 3.0, 'GBPNZD': 4.5, 'AUDCAD': 2.5,
    'AUDCHF': 2.5, 'AUDNZD': 3.0, 'CADJPY': 2.5, 'CHFJPY': 2.5, 'NZDCAD': 3.5,
    'NZDCHF': 3.5, 'NZDJPY': 3.0, 'CADCHF': 3.0,
}

print("=" * 70)
print("HOLDOUT EVALUATION: Current vs Tuned Model (DETAILED)")
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

# Add spread cost per trade
holdout_df['spread'] = holdout_df['pair'].map(SPREADS).fillna(2.5)  # Default 2.5 pips
holdout_df['pnl_after_spread'] = holdout_df['exit_pnl_pips'] - holdout_df['spread']

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

# Evaluate function (without spread)
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

# Evaluate function WITH spread deduction
def evaluate_at_threshold_with_spread(df, pred_col, threshold):
    filtered = df[df[pred_col] >= threshold]
    if len(filtered) == 0:
        return {'trades': 0, 'wr': 0, 'wr_net': 0, 'pnl': 0, 'pnl_net': 0, 'pf': 0, 'pf_net': 0,
                'winners': 0, 'losers': 0, 'spread_cost': 0}

    pnl = filtered['exit_pnl_pips']
    pnl_net = filtered['pnl_after_spread']
    spread_cost = filtered['spread'].sum()

    trades = len(filtered)
    wr = (pnl > 0).mean() * 100
    wr_net = (pnl_net > 0).mean() * 100

    total_pnl = pnl.sum()
    total_pnl_net = pnl_net.sum()

    winners = pnl[pnl > 0].sum()
    losers = abs(pnl[pnl <= 0].sum())
    pf = winners / losers if losers > 0 else 100.0

    winners_net = pnl_net[pnl_net > 0].sum()
    losers_net = abs(pnl_net[pnl_net <= 0].sum())
    pf_net = winners_net / losers_net if losers_net > 0 else 100.0

    return {
        'trades': trades,
        'wr': wr,
        'wr_net': wr_net,
        'pnl': total_pnl,
        'pnl_net': total_pnl_net,
        'pf': pf,
        'pf_net': pf_net,
        'winners': winners,
        'losers': losers,
        'spread_cost': spread_cost,
    }

# Main thresholds
main_thresholds = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85]

# Detailed thresholds (0.70 to 0.75 in 0.01 steps)
detailed_thresholds = [0.70, 0.71, 0.72, 0.73, 0.74, 0.75]

print("\n" + "=" * 70)
print("CURRENT MODEL - Holdout Performance")
print("=" * 70)
print(f"{'Thresh':<8} {'Trades':>8} {'WR%':>8} {'PF':>8} {'PnL':>12} {'Winners':>12} {'Losers':>10}")
print("-" * 75)

for thresh in main_thresholds:
    r = evaluate_at_threshold(holdout_df, 'pred_current', thresh)
    print(f"{thresh:<8} {r['trades']:>8,} {r['wr']:>7.1f}% {r['pf']:>8.2f} {r['pnl']:>+12,.0f} {r['winners']:>+12,.0f} {r['losers']:>10,.0f}")

print("\n" + "=" * 70)
print("TUNED MODEL - Holdout Performance")
print("=" * 70)
print(f"{'Thresh':<8} {'Trades':>8} {'WR%':>8} {'PF':>8} {'PnL':>12} {'Winners':>12} {'Losers':>10}")
print("-" * 75)

for thresh in main_thresholds:
    r = evaluate_at_threshold(holdout_df, 'pred_tuned', thresh)
    print(f"{thresh:<8} {r['trades']:>8,} {r['wr']:>7.1f}% {r['pf']:>8.2f} {r['pnl']:>+12,.0f} {r['winners']:>+12,.0f} {r['losers']:>10,.0f}")

# ============================================================================
# DETAILED THRESHOLD ANALYSIS (0.70 - 0.75)
# ============================================================================
print("\n" + "=" * 70)
print("DETAILED THRESHOLD ANALYSIS: 0.70 to 0.75 (TUNED MODEL)")
print("=" * 70)
print(f"{'Thresh':<8} {'Trades':>8} {'WR%':>8} {'PF':>8} {'PnL':>12} {'Winners':>12} {'Losers':>10}")
print("-" * 75)

for thresh in detailed_thresholds:
    r = evaluate_at_threshold(holdout_df, 'pred_tuned', thresh)
    print(f"{thresh:<8} {r['trades']:>8,} {r['wr']:>7.1f}% {r['pf']:>8.2f} {r['pnl']:>+12,.0f} {r['winners']:>+12,.0f} {r['losers']:>10,.0f}")

print("\n" + "=" * 70)
print("DETAILED THRESHOLD ANALYSIS: 0.70 to 0.75 (CURRENT MODEL)")
print("=" * 70)
print(f"{'Thresh':<8} {'Trades':>8} {'WR%':>8} {'PF':>8} {'PnL':>12} {'Winners':>12} {'Losers':>10}")
print("-" * 75)

for thresh in detailed_thresholds:
    r = evaluate_at_threshold(holdout_df, 'pred_current', thresh)
    print(f"{thresh:<8} {r['trades']:>8,} {r['wr']:>7.1f}% {r['pf']:>8.2f} {r['pnl']:>+12,.0f} {r['winners']:>+12,.0f} {r['losers']:>10,.0f}")

# ============================================================================
# SPREAD-ADJUSTED ANALYSIS
# ============================================================================
print("\n" + "=" * 90)
print("SPREAD-ADJUSTED ANALYSIS: 0.70 to 0.75 (TUNED MODEL)")
print("=" * 90)
print(f"{'Thresh':<7} {'Trades':>7} {'WR%':>7} {'WR%Net':>7} {'PF':>7} {'PF Net':>7} {'PnL':>10} {'PnL Net':>10} {'Spread':>8}")
print("-" * 90)

for thresh in detailed_thresholds:
    r = evaluate_at_threshold_with_spread(holdout_df, 'pred_tuned', thresh)
    print(f"{thresh:<7} {r['trades']:>7,} {r['wr']:>6.1f}% {r['wr_net']:>6.1f}% {r['pf']:>7.2f} {r['pf_net']:>7.2f} {r['pnl']:>+10,.0f} {r['pnl_net']:>+10,.0f} {r['spread_cost']:>8,.0f}")

print("\n" + "=" * 90)
print("SPREAD-ADJUSTED ANALYSIS: 0.70 to 0.75 (CURRENT MODEL)")
print("=" * 90)
print(f"{'Thresh':<7} {'Trades':>7} {'WR%':>7} {'WR%Net':>7} {'PF':>7} {'PF Net':>7} {'PnL':>10} {'PnL Net':>10} {'Spread':>8}")
print("-" * 90)

for thresh in detailed_thresholds:
    r = evaluate_at_threshold_with_spread(holdout_df, 'pred_current', thresh)
    print(f"{thresh:<7} {r['trades']:>7,} {r['wr']:>6.1f}% {r['wr_net']:>6.1f}% {r['pf']:>7.2f} {r['pf_net']:>7.2f} {r['pnl']:>+10,.0f} {r['pnl_net']:>+10,.0f} {r['spread_cost']:>8,.0f}")

# Side by side comparison at key threshold
print("\n" + "=" * 70)
print("SIDE-BY-SIDE COMPARISON @ 0.75 THRESHOLD")
print("=" * 70)

curr = evaluate_at_threshold_with_spread(holdout_df, 'pred_current', 0.75)
tuned = evaluate_at_threshold_with_spread(holdout_df, 'pred_tuned', 0.75)

print(f"\n{'Metric':<20} {'Current':>12} {'Tuned':>12} {'Diff':>12}")
print("-" * 60)
print(f"{'Trades':<20} {curr['trades']:>12,} {tuned['trades']:>12,} {tuned['trades'] - curr['trades']:>+12,}")
print(f"{'Win Rate':<20} {curr['wr']:>11.1f}% {tuned['wr']:>11.1f}% {tuned['wr'] - curr['wr']:>+11.1f}%")
print(f"{'Win Rate (Net)':<20} {curr['wr_net']:>11.1f}% {tuned['wr_net']:>11.1f}% {tuned['wr_net'] - curr['wr_net']:>+11.1f}%")
print(f"{'Profit Factor':<20} {curr['pf']:>12.2f} {tuned['pf']:>12.2f} {tuned['pf'] - curr['pf']:>+12.2f}")
print(f"{'Profit Factor (Net)':<20} {curr['pf_net']:>12.2f} {tuned['pf_net']:>12.2f} {tuned['pf_net'] - curr['pf_net']:>+12.2f}")
print(f"{'PnL (pips)':<20} {curr['pnl']:>+12,.0f} {tuned['pnl']:>+12,.0f} {tuned['pnl'] - curr['pnl']:>+12,.0f}")
print(f"{'PnL Net (pips)':<20} {curr['pnl_net']:>+12,.0f} {tuned['pnl_net']:>+12,.0f} {tuned['pnl_net'] - curr['pnl_net']:>+12,.0f}")
print(f"{'Spread Cost':<20} {curr['spread_cost']:>12,.0f} {tuned['spread_cost']:>12,.0f} {tuned['spread_cost'] - curr['spread_cost']:>+12,.0f}")

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

# ============================================================================
# SWEET SPOT ANALYSIS
# ============================================================================
print("\n" + "=" * 70)
print("SWEET SPOT ANALYSIS (Tuned Model)")
print("=" * 70)

print("\nFinding optimal threshold based on different criteria...")
print()

# Collect metrics for all detailed thresholds
results = []
for thresh in detailed_thresholds:
    r = evaluate_at_threshold(holdout_df, 'pred_tuned', thresh)
    r['threshold'] = thresh
    r['pnl_per_trade'] = r['pnl'] / r['trades'] if r['trades'] > 0 else 0
    results.append(r)

# Best by different criteria
print("Best by Win Rate:")
best_wr = max(results, key=lambda x: x['wr'])
print(f"  Threshold {best_wr['threshold']}: WR={best_wr['wr']:.1f}%, PF={best_wr['pf']:.2f}, Trades={best_wr['trades']:,}")

print("\nBest by Profit Factor:")
best_pf = max(results, key=lambda x: x['pf'])
print(f"  Threshold {best_pf['threshold']}: PF={best_pf['pf']:.2f}, WR={best_wr['wr']:.1f}%, Trades={best_pf['trades']:,}")

print("\nBest by Total PnL:")
best_pnl = max(results, key=lambda x: x['pnl'])
print(f"  Threshold {best_pnl['threshold']}: PnL={best_pnl['pnl']:+,.0f}, WR={best_pnl['wr']:.1f}%, Trades={best_pnl['trades']:,}")

print("\nBest by PnL per Trade:")
best_ppt = max(results, key=lambda x: x['pnl_per_trade'])
print(f"  Threshold {best_ppt['threshold']}: PnL/Trade={best_ppt['pnl_per_trade']:.2f}, WR={best_ppt['wr']:.1f}%, Trades={best_ppt['trades']:,}")

# Balanced recommendation (min 1000 trades, good WR and PF)
print("\nBalanced Recommendation (min 1000 trades, good WR/PF):")
balanced = [r for r in results if r['trades'] >= 1000]
if balanced:
    # Score = WR * PF_normalized * sqrt(trades)
    for r in balanced:
        r['score'] = r['wr'] * min(r['pf'], 20) / 20 * np.sqrt(r['trades'] / 1000)
    best_balanced = max(balanced, key=lambda x: x['score'])
    print(f"  Threshold {best_balanced['threshold']}: WR={best_balanced['wr']:.1f}%, PF={best_balanced['pf']:.2f}, Trades={best_balanced['trades']:,}, PnL={best_balanced['pnl']:+,.0f}")
else:
    print("  No threshold has 1000+ trades")

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)

# Calculate which is better
curr_75 = evaluate_at_threshold(holdout_df, 'pred_current', 0.75)
tuned_75 = evaluate_at_threshold(holdout_df, 'pred_tuned', 0.75)

if tuned_75['pf'] > curr_75['pf'] and tuned_75['trades'] > 100:
    print("\n[OK] Tuned model CONFIRMS better on true holdout data")
    print(f"  - Higher PF: {tuned_75['pf']:.2f} vs {curr_75['pf']:.2f}")
    print(f"  - Higher WR: {tuned_75['wr']:.1f}% vs {curr_75['wr']:.1f}%")
    if tuned_75['pnl'] < curr_75['pnl']:
        print(f"  - Lower PnL due to fewer trades ({tuned_75['trades']:,} vs {curr_75['trades']:,})")
        print(f"  - But PnL per trade: {tuned_75['pnl']/tuned_75['trades']:.1f} vs {curr_75['pnl']/curr_75['trades']:.1f} pips")
else:
    print("\n[X] Tuned model does NOT outperform on holdout")
    print(f"  Current: PF={curr_75['pf']:.2f}, WR={curr_75['wr']:.1f}%, Trades={curr_75['trades']:,}")
    print(f"  Tuned:   PF={tuned_75['pf']:.2f}, WR={tuned_75['wr']:.1f}%, Trades={tuned_75['trades']:,}")

print()
