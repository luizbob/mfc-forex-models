"""
Test Daily MFC Filter on M5 Signals
====================================
Filter: Only trade when daily MFC confirms the direction.

Condition:
- Daily MFC is below box (< -0.2) OR inside box (-0.2 to +0.2)
- Daily MFC velocity > 0 (going up / currency strengthening)
- Only trade BUY direction for that currency (base buy or quote sell)

The inverse:
- Daily MFC is above box (> +0.2) OR inside box
- Daily MFC velocity < 0 (going down / currency weakening)
- Only trade SELL direction for that currency (base sell or quote buy)
"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from datetime import datetime

def log(msg=""):
    print(msg, flush=True)

# Exness Standard spreads
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

DATA_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/lstm_model/v2_both_currencies/data')

log("=" * 70)
log("DAILY MFC FILTER BACKTEST")
log("=" * 70)

# ============================================================================
# LOAD DATA
# ============================================================================

log("\nLoading data...")
with open(DATA_DIR / 'quality_entry_data_m5_v2.pkl', 'rb') as f:
    data = pickle.load(f)

df = data['data'].copy()
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.sort_values('datetime').reset_index(drop=True)

# Fix JPY pips
jpy_mask = df['pair'].str.contains('JPY')
df.loc[jpy_mask, 'exit_pnl_pips'] = df.loc[jpy_mask, 'exit_pnl_pips'] / 100

# Add spread and net pips
df['spread'] = df['pair'].map(SPREADS)
df['net_pips'] = df['exit_pnl_pips'] - df['spread']

log(f"Total entries: {len(df):,}")
log(f"Date range: {df['datetime'].min().date()} to {df['datetime'].max().date()}")

# ============================================================================
# CHECK IF WE HAVE DAILY DATA
# ============================================================================

# Check what columns we have
daily_cols = [c for c in df.columns if 'd1' in c.lower() or 'daily' in c.lower()]
log(f"\nDaily columns found: {daily_cols}")

# Check H4 as proxy if no daily
h4_cols = [c for c in df.columns if 'h4' in c.lower()]
log(f"H4 columns found: {h4_cols}")

# We have H4 data but might not have D1 in the entry data
# Let's check if we need to load daily MFC separately or use H4 as higher timeframe

# For now, let's use H4 as the higher timeframe filter (close enough to daily concept)
# We can adjust later if we load actual D1 data

log("\nUsing H4 as higher timeframe filter (proxy for daily concept)")

# ============================================================================
# DEFINE FILTER LOGIC
# ============================================================================

BOX_THRESHOLD = 0.2

def check_daily_filter(row):
    """
    Check if trade passes the daily (H4) filter.

    For trigger currency going UP (strengthening):
    - H4 MFC <= +0.2 (below or inside box)
    - H4 velocity > 0 (going up)
    - Trade must be BUY direction for that currency

    For trigger currency going DOWN (weakening):
    - H4 MFC >= -0.2 (above or inside box)
    - H4 velocity < 0 (going down)
    - Trade must be SELL direction for that currency
    """
    trigger = row['trigger']  # 'base' or 'quote'
    direction = row['direction']  # 'buy' or 'sell'

    if trigger == 'base':
        h4_mfc = row['base_h4']
        h4_vel = row['base_vel_h4']
        # Base currency in BUY trade = buying the base
        currency_being_bought = (direction == 'buy')
    else:
        h4_mfc = row['quote_h4']
        h4_vel = row['quote_vel_h4']
        # Quote currency in SELL trade = buying the quote (selling base)
        currency_being_bought = (direction == 'sell')

    # Filter 1: Currency strengthening (going up from below/inside box) -> only buy it
    if h4_mfc <= BOX_THRESHOLD and h4_vel > 0:
        return currency_being_bought

    # Filter 2: Currency weakening (going down from above/inside box) -> only sell it
    if h4_mfc >= -BOX_THRESHOLD and h4_vel < 0:
        return not currency_being_bought

    # Otherwise, no clear signal from H4
    return False

# Apply filter
log("\nApplying H4 direction filter...")
df['passes_h4_filter'] = df.apply(check_daily_filter, axis=1)

log(f"Trades passing H4 filter: {df['passes_h4_filter'].sum():,} / {len(df):,} ({100*df['passes_h4_filter'].mean():.1f}%)")

# ============================================================================
# EVALUATE RESULTS
# ============================================================================

def evaluate(df, label=""):
    if len(df) == 0:
        return {'trades': 0, 'wr': 0, 'pf': 0, 'net_pnl': 0, 'pips_per': 0}

    net = df['net_pips']
    trades = len(df)
    wr = (net > 0).mean() * 100
    winners = net[net > 0].sum()
    losers = abs(net[net <= 0].sum())
    pf = winners / losers if losers > 0 else 100
    net_pnl = net.sum()
    pips_per = net_pnl / trades

    return {
        'trades': trades,
        'wr': wr,
        'pf': pf,
        'net_pnl': net_pnl,
        'pips_per': pips_per,
    }

# Holdout period (last 3 months)
holdout_start = df['datetime'].max() - pd.DateOffset(months=3)
holdout_df = df[df['datetime'] >= holdout_start].copy()

log(f"\nHoldout period: {holdout_df['datetime'].min().date()} to {holdout_df['datetime'].max().date()}")
log(f"Holdout samples: {len(holdout_df):,}")

log("\n" + "=" * 70)
log("HOLDOUT RESULTS - ALL M5 EXTREME SIGNALS")
log("=" * 70)

# Without filter
no_filter = evaluate(holdout_df, "No filter")
# With H4 filter
with_filter = evaluate(holdout_df[holdout_df['passes_h4_filter']], "H4 filter")

log(f"\n{'Filter':<20} {'Trades':>10} {'WR%':>8} {'PF':>8} {'Net PnL':>12} {'Pips/Trade':>12}")
log("-" * 75)
log(f"{'No filter':<20} {no_filter['trades']:>10,} {no_filter['wr']:>7.1f}% {no_filter['pf']:>8.2f} {no_filter['net_pnl']:>+12,.0f} {no_filter['pips_per']:>+12.1f}")
log(f"{'With H4 filter':<20} {with_filter['trades']:>10,} {with_filter['wr']:>7.1f}% {with_filter['pf']:>8.2f} {with_filter['net_pnl']:>+12,.0f} {with_filter['pips_per']:>+12.1f}")

# ============================================================================
# MONTHLY BREAKDOWN
# ============================================================================

log("\n" + "=" * 70)
log("MONTHLY BREAKDOWN")
log("=" * 70)

holdout_df['month'] = holdout_df['datetime'].dt.to_period('M')

log(f"\n{'Month':<10} {'Filter':<15} {'Trades':>8} {'WR%':>8} {'PF':>8} {'Net PnL':>10}")
log("-" * 65)

for month in sorted(holdout_df['month'].unique()):
    month_df = holdout_df[holdout_df['month'] == month]

    r1 = evaluate(month_df)
    r2 = evaluate(month_df[month_df['passes_h4_filter']])

    log(f"{str(month):<10} {'No filter':<15} {r1['trades']:>8,} {r1['wr']:>7.1f}% {r1['pf']:>8.2f} {r1['net_pnl']:>+10,.0f}")
    log(f"{'':<10} {'H4 filter':<15} {r2['trades']:>8,} {r2['wr']:>7.1f}% {r2['pf']:>8.2f} {r2['net_pnl']:>+10,.0f}")
    log()

# ============================================================================
# COMBINED WITH QUALITY MODEL
# ============================================================================

log("\n" + "=" * 70)
log("COMBINED: QUALITY MODEL + H4 FILTER")
log("=" * 70)

# Load the tuned classifier
import joblib
MODEL_DIR = DATA_DIR.parent / 'models'

try:
    classifier = joblib.load(MODEL_DIR / 'quality_xgb_m5_v2_tuned.joblib')
    with open(MODEL_DIR / 'quality_xgb_features_m5_v2_tuned.pkl', 'rb') as f:
        feature_cols = pickle.load(f)

    # Add time features
    holdout_df['hour'] = holdout_df['datetime'].dt.hour
    holdout_df['dayofweek'] = holdout_df['datetime'].dt.dayofweek
    holdout_df['hour_sin'] = np.sin(2 * np.pi * holdout_df['hour'] / 24)
    holdout_df['hour_cos'] = np.cos(2 * np.pi * holdout_df['hour'] / 24)
    holdout_df['dow_sin'] = np.sin(2 * np.pi * holdout_df['dayofweek'] / 5)
    holdout_df['dow_cos'] = np.cos(2 * np.pi * holdout_df['dayofweek'] / 5)
    holdout_df['direction_code'] = (holdout_df['direction'] == 'buy').astype(int)
    holdout_df['trigger_code'] = (holdout_df['trigger'] == 'base').astype(int)

    X_holdout = holdout_df[feature_cols].values.astype(np.float32)
    X_holdout = np.nan_to_num(X_holdout, nan=0.0, posinf=0.0, neginf=0.0)

    holdout_df['quality'] = classifier.predict_proba(X_holdout)[:, 1]

    log(f"\nQuality model loaded. Comparing at threshold 0.75:")

    log(f"\n{'Combination':<30} {'Trades':>8} {'WR%':>8} {'PF':>8} {'Net PnL':>12} {'Pips/Trade':>12}")
    log("-" * 85)

    # Quality only
    q_only = holdout_df[holdout_df['quality'] >= 0.75]
    r = evaluate(q_only)
    log(f"{'Quality >= 0.75 only':<30} {r['trades']:>8,} {r['wr']:>7.1f}% {r['pf']:>8.2f} {r['net_pnl']:>+12,.0f} {r['pips_per']:>+12.1f}")

    # H4 filter only
    h4_only = holdout_df[holdout_df['passes_h4_filter']]
    r = evaluate(h4_only)
    log(f"{'H4 filter only':<30} {r['trades']:>8,} {r['wr']:>7.1f}% {r['pf']:>8.2f} {r['net_pnl']:>+12,.0f} {r['pips_per']:>+12.1f}")

    # Quality + H4 filter
    both = holdout_df[(holdout_df['quality'] >= 0.75) & (holdout_df['passes_h4_filter'])]
    r = evaluate(both)
    log(f"{'Quality >= 0.75 + H4 filter':<30} {r['trades']:>8,} {r['wr']:>7.1f}% {r['pf']:>8.2f} {r['net_pnl']:>+12,.0f} {r['pips_per']:>+12.1f}")

    # Try different quality thresholds with H4 filter
    log(f"\n{'Quality Thresh + H4':<30} {'Trades':>8} {'WR%':>8} {'PF':>8} {'Net PnL':>12} {'Pips/Trade':>12}")
    log("-" * 85)

    for q_thresh in [0.60, 0.65, 0.70, 0.75, 0.80]:
        combo = holdout_df[(holdout_df['quality'] >= q_thresh) & (holdout_df['passes_h4_filter'])]
        r = evaluate(combo)
        if r['trades'] > 0:
            log(f"{'Q >= ' + str(q_thresh) + ' + H4':<30} {r['trades']:>8,} {r['wr']:>7.1f}% {r['pf']:>8.2f} {r['net_pnl']:>+12,.0f} {r['pips_per']:>+12.1f}")

except Exception as e:
    log(f"Error loading quality model: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# ANALYSIS BY TRIGGER TYPE
# ============================================================================

log("\n" + "=" * 70)
log("BY TRIGGER TYPE (with H4 filter)")
log("=" * 70)

for trigger in ['base', 'quote']:
    filtered = holdout_df[(holdout_df['passes_h4_filter']) & (holdout_df['trigger'] == trigger)]
    r = evaluate(filtered)
    log(f"\n{trigger.upper()} trigger: {r['trades']:,} trades, {r['wr']:.1f}% WR, PF={r['pf']:.2f}, {r['net_pnl']:+,.0f} pips")

# ============================================================================
# STRICTER H4 FILTER VARIATIONS
# ============================================================================

log("\n" + "=" * 70)
log("FILTER VARIATIONS")
log("=" * 70)

def check_filter_strict(row, mfc_max=0.0, vel_min=0.01):
    """Stricter filter: MFC must be below 0 and velocity must be significant."""
    trigger = row['trigger']
    direction = row['direction']

    if trigger == 'base':
        h4_mfc = row['base_h4']
        h4_vel = row['base_vel_h4']
        currency_being_bought = (direction == 'buy')
    else:
        h4_mfc = row['quote_h4']
        h4_vel = row['quote_vel_h4']
        currency_being_bought = (direction == 'sell')

    # Strengthening from BELOW zero
    if h4_mfc <= mfc_max and h4_vel >= vel_min:
        return currency_being_bought

    # Weakening from ABOVE zero
    if h4_mfc >= -mfc_max and h4_vel <= -vel_min:
        return not currency_being_bought

    return False

log(f"\n{'Variation':<40} {'Trades':>8} {'WR%':>8} {'PF':>8} {'Net PnL':>12}")
log("-" * 80)

variations = [
    ("MFC <= 0.2, vel > 0 (original)", 0.2, 0.001),
    ("MFC <= 0.0, vel > 0", 0.0, 0.001),
    ("MFC <= -0.1, vel > 0", -0.1, 0.001),
    ("MFC <= 0.0, vel >= 0.01", 0.0, 0.01),
    ("MFC <= 0.0, vel >= 0.02", 0.0, 0.02),
    ("MFC <= -0.2, vel >= 0.01", -0.2, 0.01),
]

for name, mfc_max, vel_min in variations:
    holdout_df['passes_var'] = holdout_df.apply(lambda r: check_filter_strict(r, mfc_max, vel_min), axis=1)
    filtered = holdout_df[holdout_df['passes_var']]
    r = evaluate(filtered)
    if r['trades'] > 0:
        log(f"{name:<40} {r['trades']:>8,} {r['wr']:>7.1f}% {r['pf']:>8.2f} {r['net_pnl']:>+12,.0f}")

log("\n" + "=" * 70)
log("DONE")
log("=" * 70)
