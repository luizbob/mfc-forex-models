"""
Backtest Both Models (Mean Reversion + Momentum)
=================================================
Run both M5 models and compare performance.
"""
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import pickle
import joblib
from pathlib import Path
from datetime import datetime

def log(msg=""):
    print(msg, flush=True)

log("=" * 70)
log("BACKTEST: MEAN REVERSION + MOMENTUM MODELS")
log("=" * 70)
log(f"Started: {datetime.now()}")

DATA_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/lstm_model/v2_both_currencies/data')
MODEL_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/lstm_model/v2_both_currencies/models')

# Load models
log("\nLoading models...")

# Mean Reversion
mr_model = joblib.load(MODEL_DIR / 'quality_xgb_m5_v2_pnl.joblib')
with open(MODEL_DIR / 'quality_xgb_features_m5_v2_pnl.pkl', 'rb') as f:
    mr_features = pickle.load(f)
log(f"Mean Reversion: {len(mr_features)} features")

# Momentum
mom_model = joblib.load(MODEL_DIR / 'momentum_xgb_m5.joblib')
with open(MODEL_DIR / 'momentum_xgb_features_m5.pkl', 'rb') as f:
    mom_features = pickle.load(f)
log(f"Momentum: {len(mom_features)} features")

# Load data
log("\nLoading data...")

# Mean Reversion data
with open(DATA_DIR / 'quality_entry_data_m5_v2.pkl', 'rb') as f:
    mr_data = pickle.load(f)
mr_df = mr_data['data'].copy()

# Fix JPY pips
jpy_mask = mr_df['pair'].str.contains('JPY')
mr_df.loc[jpy_mask, 'exit_pnl_pips'] = mr_df.loc[jpy_mask, 'exit_pnl_pips'] / 100

# Momentum data
with open(DATA_DIR / 'momentum_entry_data_m5.pkl', 'rb') as f:
    mom_data = pickle.load(f)
mom_df = mom_data['data'].copy()

# Fix JPY pips
jpy_mask = mom_df['pair'].str.contains('JPY')
mom_df.loc[jpy_mask, 'exit_pnl_pips'] = mom_df.loc[jpy_mask, 'exit_pnl_pips'] / 100

log(f"Mean Reversion entries: {len(mr_df):,}")
log(f"Momentum entries: {len(mom_df):,}")

# Sort by datetime
mr_df = mr_df.sort_values('datetime').reset_index(drop=True)
mom_df = mom_df.sort_values('datetime').reset_index(drop=True)

# Use test period only (last 15%)
mr_test_start = int(len(mr_df) * 0.85)
mom_test_start = int(len(mom_df) * 0.85)

mr_test = mr_df.iloc[mr_test_start:].copy()
mom_test = mom_df.iloc[mom_test_start:].copy()

log(f"\nTest period:")
log(f"  Mean Reversion: {len(mr_test):,} entries ({mr_test['datetime'].min().date()} to {mr_test['datetime'].max().date()})")
log(f"  Momentum: {len(mom_test):,} entries ({mom_test['datetime'].min().date()} to {mom_test['datetime'].max().date()})")

# Add codes
mr_test['direction_code'] = (mr_test['direction'] == 'buy').astype(int)
mr_test['trigger_code'] = (mr_test['trigger'] == 'base').astype(int)

mom_test['direction_code'] = (mom_test['direction'] == 'buy').astype(int)
mom_test['trigger_code'] = (mom_test['trigger'] == 'base').astype(int)

# Get predictions
log("\nGenerating predictions...")

X_mr = mr_test[mr_features].values.astype(np.float32)
X_mr = np.nan_to_num(X_mr, nan=0.0, posinf=0.0, neginf=0.0)
mr_test['pred_prob'] = mr_model.predict_proba(X_mr)[:, 1]

X_mom = mom_test[mom_features].values.astype(np.float32)
X_mom = np.nan_to_num(X_mom, nan=0.0, posinf=0.0, neginf=0.0)
mom_test['pred_prob'] = mom_model.predict_proba(X_mom)[:, 1]

# ============================================================================
# BACKTEST FUNCTION
# ============================================================================

def calc_stats(df, name):
    """Calculate trading stats for a dataframe with exit_pnl_pips."""
    if len(df) == 0:
        return None

    pnl = df['exit_pnl_pips']
    winners = pnl[pnl > 0].sum()
    losers = abs(pnl[pnl <= 0].sum())
    pf = winners / losers if losers > 0 else float('inf')

    # Trades per day
    days = (df['datetime'].max() - df['datetime'].min()).days
    tpd = len(df) / days if days > 0 else 0

    return {
        'name': name,
        'trades': len(df),
        'win_rate': (pnl > 0).mean() * 100,
        'avg_pnl': pnl.mean(),
        'total_pnl': pnl.sum(),
        'pf': pf,
        'tpd': tpd,
    }

# ============================================================================
# MEAN REVERSION RESULTS
# ============================================================================

log("\n" + "=" * 70)
log("MEAN REVERSION MODEL RESULTS")
log("=" * 70)

log(f"\n{'Threshold':<10} {'Trades':>8} {'WinRate':>8} {'AvgPnL':>8} {'TotalPnL':>10} {'PF':>6} {'TPD':>6}")
log("-" * 60)

for thresh in [0.50, 0.60, 0.70, 0.80, 0.90]:
    filtered = mr_test[mr_test['pred_prob'] >= thresh]
    stats = calc_stats(filtered, f"MR_{thresh}")
    if stats:
        log(f"{thresh:<10} {stats['trades']:>8,} {stats['win_rate']:>7.1f}% {stats['avg_pnl']:>+7.1f} {stats['total_pnl']:>+10,.0f} {stats['pf']:>6.2f} {stats['tpd']:>5.1f}")

# ============================================================================
# MOMENTUM MODEL RESULTS
# ============================================================================

log("\n" + "=" * 70)
log("MOMENTUM MODEL RESULTS")
log("=" * 70)

log(f"\n{'Threshold':<10} {'Trades':>8} {'WinRate':>8} {'AvgPnL':>8} {'TotalPnL':>10} {'PF':>6} {'TPD':>6}")
log("-" * 60)

for thresh in [0.50, 0.60, 0.70, 0.80, 0.90]:
    filtered = mom_test[mom_test['pred_prob'] >= thresh]
    stats = calc_stats(filtered, f"MOM_{thresh}")
    if stats:
        log(f"{thresh:<10} {stats['trades']:>8,} {stats['win_rate']:>7.1f}% {stats['avg_pnl']:>+7.1f} {stats['total_pnl']:>+10,.0f} {stats['pf']:>6.2f} {stats['tpd']:>5.1f}")

# ============================================================================
# COMBINED RESULTS
# ============================================================================

log("\n" + "=" * 70)
log("COMBINED (BOTH MODELS)")
log("=" * 70)

# Best thresholds for each
mr_best_thresh = 0.70
mom_best_thresh = 0.70

mr_filtered = mr_test[mr_test['pred_prob'] >= mr_best_thresh].copy()
mom_filtered = mom_test[mom_test['pred_prob'] >= mom_best_thresh].copy()

mr_filtered['model'] = 'MeanReversion'
mom_filtered['model'] = 'Momentum'

# Combine
combined = pd.concat([
    mr_filtered[['datetime', 'pair', 'direction', 'exit_pnl_pips', 'model', 'pred_prob']],
    mom_filtered[['datetime', 'pair', 'direction', 'exit_pnl_pips', 'model', 'pred_prob']]
], ignore_index=True)

combined = combined.sort_values('datetime').reset_index(drop=True)

log(f"\nCombined at threshold 0.70:")
log(f"  Mean Reversion trades: {len(mr_filtered):,}")
log(f"  Momentum trades: {len(mom_filtered):,}")
log(f"  Total trades: {len(combined):,}")

if len(combined) > 0:
    pnl = combined['exit_pnl_pips']
    winners = pnl[pnl > 0].sum()
    losers = abs(pnl[pnl <= 0].sum())
    pf = winners / losers if losers > 0 else 0

    days = (combined['datetime'].max() - combined['datetime'].min()).days
    tpd = len(combined) / days if days > 0 else 0

    log(f"\nCombined Performance:")
    log(f"  Win Rate: {(pnl > 0).mean()*100:.1f}%")
    log(f"  Avg PnL: {pnl.mean():+.1f} pips")
    log(f"  Total PnL: {pnl.sum():+,.0f} pips")
    log(f"  Profit Factor: {pf:.2f}")
    log(f"  Trades/Day: {tpd:.1f}")

# ============================================================================
# BY YEAR
# ============================================================================

log("\n" + "=" * 70)
log("BY YEAR (at 0.70 threshold)")
log("=" * 70)

combined['year'] = pd.to_datetime(combined['datetime']).dt.year

log(f"\n{'Year':<6} {'MR Trades':>10} {'MR PnL':>10} {'Mom Trades':>10} {'Mom PnL':>10} {'Combined':>10}")
log("-" * 70)

for year in sorted(combined['year'].unique()):
    year_data = combined[combined['year'] == year]
    mr_year = year_data[year_data['model'] == 'MeanReversion']
    mom_year = year_data[year_data['model'] == 'Momentum']

    mr_pnl = mr_year['exit_pnl_pips'].sum() if len(mr_year) > 0 else 0
    mom_pnl = mom_year['exit_pnl_pips'].sum() if len(mom_year) > 0 else 0

    log(f"{year:<6} {len(mr_year):>10,} {mr_pnl:>+10,.0f} {len(mom_year):>10,} {mom_pnl:>+10,.0f} {mr_pnl + mom_pnl:>+10,.0f}")

# ============================================================================
# BY PAIR (TOP 10)
# ============================================================================

log("\n" + "=" * 70)
log("TOP 10 PAIRS BY TOTAL PNL (Combined at 0.70)")
log("=" * 70)

pair_stats = combined.groupby('pair').agg({
    'exit_pnl_pips': ['count', 'sum', 'mean']
}).round(1)
pair_stats.columns = ['trades', 'total_pnl', 'avg_pnl']
pair_stats['win_rate'] = combined.groupby('pair').apply(lambda x: (x['exit_pnl_pips'] > 0).mean() * 100)
pair_stats = pair_stats.sort_values('total_pnl', ascending=False)

log(f"\n{'Pair':<10} {'Trades':>8} {'WinRate':>8} {'AvgPnL':>8} {'TotalPnL':>10}")
log("-" * 50)

for pair, row in pair_stats.head(10).iterrows():
    log(f"{pair:<10} {int(row['trades']):>8,} {row['win_rate']:>7.1f}% {row['avg_pnl']:>+7.1f} {row['total_pnl']:>+10,.0f}")

log("\n" + "-" * 50)
log("WORST 5 PAIRS:")
for pair, row in pair_stats.tail(5).iterrows():
    log(f"{pair:<10} {int(row['trades']):>8,} {row['win_rate']:>7.1f}% {row['avg_pnl']:>+7.1f} {row['total_pnl']:>+10,.0f}")

log(f"\nCompleted: {datetime.now()}")
