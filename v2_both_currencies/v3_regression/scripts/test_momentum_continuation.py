"""
Test: Momentum Continuation
===========================
After seeing high velocity (shift 1), how much does price CONTINUE?

Logic:
- At bar T, we see MFC velocity from bar T-1 (shift 1)
- If velocity was high, momentum may continue
- Enter at bar T, ride the continuation
- Test with different holding periods and H4 confirmation
"""
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from pathlib import Path

def log(msg=""):
    print(msg, flush=True)

DATA_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/data')
CLEANED_DIR = DATA_DIR / 'cleaned'

CURRENCIES = ['EUR', 'USD', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD', 'NZD']
PAIRS = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD',
         'EURJPY', 'EURGBP', 'GBPJPY', 'AUDJPY', 'EURAUD']

PIP_SIZE = {p: 0.01 if 'JPY' in p else 0.0001 for p in PAIRS}

log("=" * 70)
log("MOMENTUM CONTINUATION TEST")
log("After high velocity, does price continue?")
log("=" * 70)

# ============================================================================
# LOAD DATA
# ============================================================================

def load_mfc(currency, timeframe):
    fp = CLEANED_DIR / f'mfc_currency_{currency}_{timeframe}_clean.csv'
    df = pd.read_csv(fp)
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df = df.set_index('datetime').sort_index()
    return df['MFC']

def load_price(pair, timeframe):
    fp = DATA_DIR / f'{pair}_GMT+0_US-DST_{timeframe}.csv'
    if not fp.exists():
        return None
    df = pd.read_csv(fp, header=None, names=['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df = df.set_index('datetime').sort_index()
    return df

log("\nLoading M30 data...")

# MFC with shift(1)
mfc_m30 = pd.DataFrame()
for ccy in CURRENCIES:
    mfc = load_mfc(ccy, 'M30')
    if mfc_m30.empty:
        mfc_m30 = pd.DataFrame(index=mfc.index)
    # SHIFT 1 - we see previous bar's MFC
    mfc_m30[ccy] = mfc.shift(1)
    mfc_m30[f'{ccy}_vel'] = mfc_m30[ccy].diff()

mfc_m30 = mfc_m30.dropna()

# Load H4 MFC (also shifted)
log("Loading H4 data...")
mfc_h4 = pd.DataFrame()
for ccy in CURRENCIES:
    mfc = load_mfc(ccy, 'H4')
    if mfc_h4.empty:
        mfc_h4 = pd.DataFrame(index=mfc.index)
    mfc_h4[f'{ccy}_h4'] = mfc.shift(1)
    mfc_h4[f'{ccy}_h4_vel'] = mfc_h4[f'{ccy}_h4'].diff()

# Forward fill H4 to M30 grid
for ccy in CURRENCIES:
    mfc_m30[f'{ccy}_h4'] = mfc_h4[f'{ccy}_h4'].reindex(mfc_m30.index, method='ffill')
    mfc_m30[f'{ccy}_h4_vel'] = mfc_h4[f'{ccy}_h4_vel'].reindex(mfc_m30.index, method='ffill')

mfc_m30 = mfc_m30.dropna()

# Price
price_data = {}
for pair in PAIRS:
    price_data[pair] = load_price(pair, 'M30')

log(f"MFC M30: {len(mfc_m30):,} bars")

# ============================================================================
# FIND HIGH VELOCITY SIGNALS AND MEASURE CONTINUATION
# ============================================================================

log("\n" + "=" * 70)
log("MEASURING CONTINUATION AFTER HIGH VELOCITY")
log("=" * 70)

def measure_continuation(mfc_df, price_data, vel_threshold=0.10, hold_bars=6):
    """
    Find high velocity moments and measure price continuation.
    """
    results = []

    for pair in PAIRS:
        if price_data[pair] is None:
            continue

        base = pair[:3]
        quote = pair[3:]
        pdf = price_data[pair]

        for i in range(1, len(mfc_df) - hold_bars):
            idx = mfc_df.index[i]

            # Check base currency velocity
            base_vel = mfc_df[f'{base}_vel'].iloc[i]
            quote_vel = mfc_df[f'{quote}_vel'].iloc[i]

            # High velocity on base or quote?
            signal = None
            signal_vel = 0

            if base_vel > vel_threshold:
                signal = 'base_strong'
                signal_vel = base_vel
                expected_dir = 1  # Pair should go UP
            elif base_vel < -vel_threshold:
                signal = 'base_weak'
                signal_vel = abs(base_vel)
                expected_dir = -1  # Pair should go DOWN
            elif quote_vel > vel_threshold:
                signal = 'quote_strong'
                signal_vel = quote_vel
                expected_dir = -1  # Pair should go DOWN
            elif quote_vel < -vel_threshold:
                signal = 'quote_weak'
                signal_vel = abs(quote_vel)
                expected_dir = 1  # Pair should go UP

            if signal is None:
                continue

            # Get price at entry and after hold_bars
            if idx not in pdf.index:
                continue

            entry_idx = pdf.index.get_loc(idx)
            if entry_idx + hold_bars >= len(pdf):
                continue

            entry_price = pdf.iloc[entry_idx]['Close']
            exit_price = pdf.iloc[entry_idx + hold_bars]['Close']

            pip_move = (exit_price - entry_price) / PIP_SIZE[pair]
            adjusted_pips = pip_move * expected_dir

            # H4 confirmation
            base_h4_vel = mfc_df[f'{base}_h4_vel'].iloc[i]
            quote_h4_vel = mfc_df[f'{quote}_h4_vel'].iloc[i]

            # Does H4 confirm M30 direction?
            if signal in ['base_strong', 'quote_weak']:
                h4_confirms = base_h4_vel > 0.02 or quote_h4_vel < -0.02
            else:
                h4_confirms = base_h4_vel < -0.02 or quote_h4_vel > 0.02

            # Both currencies agreeing?
            if signal in ['base_strong', 'quote_weak']:
                both_agree = base_vel > 0.03 and quote_vel < -0.03
            else:
                both_agree = base_vel < -0.03 and quote_vel > 0.03

            results.append({
                'datetime': idx,
                'pair': pair,
                'signal': signal,
                'signal_vel': signal_vel,
                'base_vel': base_vel,
                'quote_vel': quote_vel,
                'h4_confirms': h4_confirms,
                'both_agree': both_agree,
                'pip_move': pip_move,
                'adjusted_pips': adjusted_pips,
                'win': adjusted_pips > 0,
            })

    return pd.DataFrame(results)

# Test different velocity thresholds
log(f"\n{'Vel Thresh':<12} {'Signals':>10} {'Win %':>10} {'Avg Pips':>12} {'Total Pips':>12}")
log("-" * 60)

for vel_thresh in [0.05, 0.08, 0.10, 0.15, 0.20, 0.30]:
    df = measure_continuation(mfc_m30, price_data, vel_threshold=vel_thresh, hold_bars=6)
    if len(df) > 0:
        wr = df['win'].mean() * 100
        avg = df['adjusted_pips'].mean()
        total = df['adjusted_pips'].sum()
        log(f"{vel_thresh:<12} {len(df):>10,} {wr:>9.1f}% {avg:>+12.2f} {total:>+12,.0f}")

# ============================================================================
# BEST THRESHOLD - DETAILED ANALYSIS
# ============================================================================

log("\n" + "=" * 70)
log("DETAILED ANALYSIS AT VELOCITY >= 0.15")
log("=" * 70)

df = measure_continuation(mfc_m30, price_data, vel_threshold=0.15, hold_bars=6)

log(f"\nTotal signals: {len(df):,}")
log(f"Win rate: {df['win'].mean()*100:.1f}%")
log(f"Avg pips: {df['adjusted_pips'].mean():+.2f}")

# By H4 confirmation
log("\n" + "-" * 40)
log("BY H4 CONFIRMATION")
log("-" * 40)

h4_yes = df[df['h4_confirms']]
h4_no = df[~df['h4_confirms']]

log(f"\n{'H4 Confirms':<15} {'Signals':>10} {'Win %':>10} {'Avg Pips':>12}")
log("-" * 50)
log(f"{'YES':<15} {len(h4_yes):>10,} {h4_yes['win'].mean()*100:>9.1f}% {h4_yes['adjusted_pips'].mean():>+12.2f}")
log(f"{'NO':<15} {len(h4_no):>10,} {h4_no['win'].mean()*100:>9.1f}% {h4_no['adjusted_pips'].mean():>+12.2f}")

# By both currencies agreeing
log("\n" + "-" * 40)
log("BY BOTH CURRENCIES AGREEING")
log("-" * 40)

both_yes = df[df['both_agree']]
both_no = df[~df['both_agree']]

log(f"\n{'Both Agree':<15} {'Signals':>10} {'Win %':>10} {'Avg Pips':>12}")
log("-" * 50)
log(f"{'YES':<15} {len(both_yes):>10,} {both_yes['win'].mean()*100:>9.1f}% {both_yes['adjusted_pips'].mean():>+12.2f}")
log(f"{'NO':<15} {len(both_no):>10,} {both_no['win'].mean()*100:>9.1f}% {both_no['adjusted_pips'].mean():>+12.2f}")

# By velocity strength
log("\n" + "-" * 40)
log("BY VELOCITY STRENGTH")
log("-" * 40)

log(f"\n{'Velocity':<15} {'Signals':>10} {'Win %':>10} {'Avg Pips':>12}")
log("-" * 50)

for low, high, label in [(0.15, 0.20, '0.15-0.20'), (0.20, 0.30, '0.20-0.30'), (0.30, 0.50, '0.30-0.50'), (0.50, 10.0, '>0.50')]:
    subset = df[(df['signal_vel'] >= low) & (df['signal_vel'] < high)]
    if len(subset) >= 50:
        wr = subset['win'].mean() * 100
        avg = subset['adjusted_pips'].mean()
        log(f"{label:<15} {len(subset):>10,} {wr:>9.1f}% {avg:>+12.2f}")

# ============================================================================
# VARY HOLDING PERIOD
# ============================================================================

log("\n" + "=" * 70)
log("BY HOLDING PERIOD (velocity >= 0.15)")
log("=" * 70)

log(f"\n{'Hold Bars':<12} {'Signals':>10} {'Win %':>10} {'Avg Pips':>12}")
log("-" * 50)

for hold in [1, 2, 3, 4, 6, 8, 12, 24]:
    df = measure_continuation(mfc_m30, price_data, vel_threshold=0.15, hold_bars=hold)
    if len(df) > 0:
        wr = df['win'].mean() * 100
        avg = df['adjusted_pips'].mean()
        log(f"{hold:<12} {len(df):>10,} {wr:>9.1f}% {avg:>+12.2f}")

# ============================================================================
# BEST COMBO: HIGH VEL + H4 CONFIRMS + BOTH AGREE
# ============================================================================

log("\n" + "=" * 70)
log("BEST COMBINATION")
log("=" * 70)

df = measure_continuation(mfc_m30, price_data, vel_threshold=0.15, hold_bars=6)

best = df[(df['h4_confirms']) & (df['both_agree'])]
log(f"\nHigh velocity + H4 confirms + Both currencies agree:")
log(f"  Signals: {len(best):,}")
if len(best) > 0:
    log(f"  Win rate: {best['win'].mean()*100:.1f}%")
    log(f"  Avg pips: {best['adjusted_pips'].mean():+.2f}")
    log(f"  Total pips: {best['adjusted_pips'].sum():+,.0f}")

# Very high velocity only
very_high = df[df['signal_vel'] >= 0.30]
log(f"\nVery high velocity (>= 0.30) only:")
log(f"  Signals: {len(very_high):,}")
if len(very_high) > 0:
    log(f"  Win rate: {very_high['win'].mean()*100:.1f}%")
    log(f"  Avg pips: {very_high['adjusted_pips'].mean():+.2f}")

# ============================================================================
# HOLDOUT
# ============================================================================

log("\n" + "=" * 70)
log("HOLDOUT (Last 3 Months)")
log("=" * 70)

df = measure_continuation(mfc_m30, price_data, vel_threshold=0.15, hold_bars=6)
holdout_start = mfc_m30.index.max() - pd.DateOffset(months=3)
holdout = df[df['datetime'] >= holdout_start]

log(f"\nHoldout signals: {len(holdout):,}")
if len(holdout) > 0:
    log(f"Win rate: {holdout['win'].mean()*100:.1f}%")
    log(f"Avg pips: {holdout['adjusted_pips'].mean():+.2f}")
    log(f"Total pips: {holdout['adjusted_pips'].sum():+,.0f}")

log("\n" + "=" * 70)
log("CONCLUSION")
log("=" * 70)
log()
