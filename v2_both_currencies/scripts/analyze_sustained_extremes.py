"""
Analyze Sustained Extremes and Dip Patterns
============================================
1. What makes MFC stay at ±0.5 for extended periods?
2. Can we trade dips within the extreme zone (pullback continuation)?
"""
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

def log(msg=""):
    print(msg, flush=True)

log("=" * 70)
log("ANALYZING SUSTAINED EXTREMES & DIP PATTERNS")
log("=" * 70)
log(f"Started: {datetime.now()}")

DATA_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/data')
CLEANED_DIR = DATA_DIR / 'cleaned'

CURRENCIES = ['EUR', 'USD', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD', 'NZD']

# Load M5 MFC data for all currencies
log("\nLoading MFC data...")
mfc_data = {}
for ccy in CURRENCIES:
    df = pd.read_csv(CLEANED_DIR / f'mfc_currency_{ccy}_M5_clean.csv')
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df = df.set_index('datetime').sort_index()
    mfc_data[ccy] = df['MFC']
    log(f"  {ccy}: {len(df):,} bars")

# Also load H1 for context
mfc_h1 = {}
for ccy in CURRENCIES:
    df = pd.read_csv(CLEANED_DIR / f'mfc_currency_{ccy}_H1_clean.csv')
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df = df.set_index('datetime').sort_index()
    mfc_h1[ccy] = df['MFC']

log("\n" + "=" * 70)
log("PART 1: SUSTAINED EXTREMES ANALYSIS")
log("=" * 70)

# Find sustained extreme episodes
# Define: MFC stays above 0.4 (or below -0.4) for N consecutive bars

EXTREME_THRESHOLD = 0.4
MIN_DURATION = 20  # At least 20 M5 bars (100 minutes) to be "sustained"

all_episodes = []

for ccy in CURRENCIES:
    mfc = mfc_data[ccy]
    arr = mfc.values
    idx = mfc.index

    # Track episodes
    in_extreme = False
    extreme_start = None
    extreme_sign = 0

    for i in range(len(arr)):
        val = arr[i]

        if not in_extreme:
            # Check if entering extreme
            if val >= EXTREME_THRESHOLD:
                in_extreme = True
                extreme_start = i
                extreme_sign = 1
            elif val <= -EXTREME_THRESHOLD:
                in_extreme = True
                extreme_start = i
                extreme_sign = -1
        else:
            # Check if exiting extreme
            exit_condition = (extreme_sign == 1 and val < EXTREME_THRESHOLD) or \
                           (extreme_sign == -1 and val > -EXTREME_THRESHOLD)

            if exit_condition or i == len(arr) - 1:
                duration = i - extreme_start
                if duration >= MIN_DURATION:
                    # Get episode data
                    ep_mfc = arr[extreme_start:i+1]

                    # Calculate features at entry
                    if extreme_start > 12:
                        entry_vel = arr[extreme_start] - arr[extreme_start - 12]  # H1 velocity
                    else:
                        entry_vel = 0

                    # Get H1 context
                    h1_at_start = mfc_h1[ccy].asof(idx[extreme_start])

                    # How deep did it go?
                    if extreme_sign == 1:
                        max_extreme = ep_mfc.max()
                        min_pullback = ep_mfc.min()
                    else:
                        max_extreme = ep_mfc.min()
                        min_pullback = ep_mfc.max()

                    # What happened after?
                    if i + 50 < len(arr):
                        after_mfc = arr[i:i+50]
                        # Did it return to 0?
                        if extreme_sign == 1:
                            returned_to_zero = (after_mfc <= 0).any()
                            continued_extreme = (after_mfc >= 0.5).any()
                        else:
                            returned_to_zero = (after_mfc >= 0).any()
                            continued_extreme = (after_mfc <= -0.5).any()
                    else:
                        returned_to_zero = None
                        continued_extreme = None

                    all_episodes.append({
                        'currency': ccy,
                        'start_time': idx[extreme_start],
                        'end_time': idx[i],
                        'duration_bars': duration,
                        'duration_hours': duration * 5 / 60,
                        'sign': extreme_sign,
                        'entry_mfc': arr[extreme_start],
                        'entry_vel': entry_vel,
                        'h1_at_start': h1_at_start,
                        'max_extreme': max_extreme,
                        'min_pullback': min_pullback,
                        'returned_to_zero': returned_to_zero,
                        'continued_extreme': continued_extreme,
                    })

                in_extreme = False

episodes_df = pd.DataFrame(all_episodes)
log(f"\nFound {len(episodes_df):,} sustained extreme episodes (>= {MIN_DURATION} bars)")

# Analyze episodes
log(f"\nBy Duration:")
duration_bins = [20, 50, 100, 200, 500, 1000, 5000]
for i in range(len(duration_bins) - 1):
    low, high = duration_bins[i], duration_bins[i+1]
    subset = episodes_df[(episodes_df['duration_bars'] >= low) & (episodes_df['duration_bars'] < high)]
    if len(subset) > 0:
        avg_hours = subset['duration_hours'].mean()
        log(f"  {low}-{high} bars: {len(subset):,} episodes ({avg_hours:.1f}h avg)")

log(f"\nBy Currency:")
for ccy in CURRENCIES:
    subset = episodes_df[episodes_df['currency'] == ccy]
    if len(subset) > 0:
        avg_duration = subset['duration_hours'].mean()
        log(f"  {ccy}: {len(subset):,} episodes, avg {avg_duration:.1f}h")

log(f"\nWhat predicts LONG sustained extremes (>100 bars / 8+ hours)?")
long_episodes = episodes_df[episodes_df['duration_bars'] >= 100]
short_episodes = episodes_df[(episodes_df['duration_bars'] >= 20) & (episodes_df['duration_bars'] < 100)]

if len(long_episodes) > 0 and len(short_episodes) > 0:
    log(f"\n  Long episodes ({len(long_episodes):,}):")
    log(f"    Entry velocity (H1): {long_episodes['entry_vel'].mean():.4f}")
    log(f"    Entry MFC: {long_episodes['entry_mfc'].abs().mean():.3f}")
    log(f"    H1 MFC at start: {long_episodes['h1_at_start'].abs().mean():.3f}")
    log(f"    Max extreme reached: {long_episodes['max_extreme'].abs().mean():.3f}")

    log(f"\n  Short episodes ({len(short_episodes):,}):")
    log(f"    Entry velocity (H1): {short_episodes['entry_vel'].mean():.4f}")
    log(f"    Entry MFC: {short_episodes['entry_mfc'].abs().mean():.3f}")
    log(f"    H1 MFC at start: {short_episodes['h1_at_start'].abs().mean():.3f}")
    log(f"    Max extreme reached: {short_episodes['max_extreme'].abs().mean():.3f}")

# What happens after sustained extreme breaks?
log(f"\nAfter sustained extreme breaks:")
valid_after = episodes_df[episodes_df['returned_to_zero'].notna()]
if len(valid_after) > 0:
    returned = valid_after['returned_to_zero'].mean() * 100
    continued = valid_after['continued_extreme'].mean() * 100
    log(f"  Returned to 0: {returned:.1f}%")
    log(f"  Continued extreme: {continued:.1f}%")

log("\n" + "=" * 70)
log("PART 2: DIP PATTERNS WITHIN EXTREME ZONE")
log("=" * 70)

# Find dip patterns: MFC reaches ≥0.5, pulls back to 0.3-0.45, then continues
# This is a potential "buy the dip" setup

DIP_ENTRY_MIN = 0.3  # Pullback must reach at least this
DIP_ENTRY_MAX = 0.45  # But not go below this (still in box)
EXTREME_LEVEL = 0.5

all_dips = []

for ccy in CURRENCIES:
    mfc = mfc_data[ccy]
    arr = mfc.values
    idx = mfc.index

    # Find points where MFC was at extreme, then pulled back
    for i in range(50, len(arr) - 100):
        val = arr[i]

        # Check for positive dip (was high, pulled back but still positive)
        if DIP_ENTRY_MIN <= val <= DIP_ENTRY_MAX:
            # Was it at extreme recently?
            lookback = arr[i-50:i]
            if lookback.max() >= EXTREME_LEVEL:
                # This is a dip from positive extreme
                # Check what happens next
                future = arr[i+1:i+101]  # Next 100 bars

                # Did it return to extreme?
                reached_extreme_again = (future >= EXTREME_LEVEL).any()
                if reached_extreme_again:
                    bars_to_extreme = np.argmax(future >= EXTREME_LEVEL) + 1
                else:
                    bars_to_extreme = None

                # Did it break down (go to 0 or negative)?
                broke_down = (future <= 0).any()
                if broke_down:
                    bars_to_breakdown = np.argmax(future <= 0) + 1
                else:
                    bars_to_breakdown = None

                # Entry velocity
                entry_vel = arr[i] - arr[i-12] if i > 12 else 0

                # How deep was the pullback?
                pullback_depth = lookback.max() - val

                all_dips.append({
                    'currency': ccy,
                    'time': idx[i],
                    'dip_level': val,
                    'prior_extreme': lookback.max(),
                    'pullback_depth': pullback_depth,
                    'entry_vel': entry_vel,
                    'reached_extreme': reached_extreme_again,
                    'bars_to_extreme': bars_to_extreme,
                    'broke_down': broke_down,
                    'bars_to_breakdown': bars_to_breakdown,
                    'direction': 'long',
                })

        # Check for negative dip (was low, pulled back but still negative)
        elif -DIP_ENTRY_MAX <= val <= -DIP_ENTRY_MIN:
            lookback = arr[i-50:i]
            if lookback.min() <= -EXTREME_LEVEL:
                future = arr[i+1:i+101]

                reached_extreme_again = (future <= -EXTREME_LEVEL).any()
                if reached_extreme_again:
                    bars_to_extreme = np.argmax(future <= -EXTREME_LEVEL) + 1
                else:
                    bars_to_extreme = None

                broke_down = (future >= 0).any()
                if broke_down:
                    bars_to_breakdown = np.argmax(future >= 0) + 1
                else:
                    bars_to_breakdown = None

                entry_vel = arr[i] - arr[i-12] if i > 12 else 0
                pullback_depth = val - lookback.min()

                all_dips.append({
                    'currency': ccy,
                    'time': idx[i],
                    'dip_level': val,
                    'prior_extreme': lookback.min(),
                    'pullback_depth': pullback_depth,
                    'entry_vel': entry_vel,
                    'reached_extreme': reached_extreme_again,
                    'bars_to_extreme': bars_to_extreme,
                    'broke_down': broke_down,
                    'bars_to_breakdown': bars_to_breakdown,
                    'direction': 'short',
                })

dips_df = pd.DataFrame(all_dips)
log(f"\nFound {len(dips_df):,} dip patterns")

if len(dips_df) > 0:
    log(f"\nDip Pattern Success Rate:")
    success = dips_df['reached_extreme'].mean() * 100
    failure = dips_df['broke_down'].mean() * 100
    log(f"  Reached extreme again: {success:.1f}%")
    log(f"  Broke down to 0: {failure:.1f}%")

    # When both could happen, which happened first?
    both_possible = dips_df[(dips_df['bars_to_extreme'].notna()) & (dips_df['bars_to_breakdown'].notna())]
    if len(both_possible) > 0:
        extreme_first = (both_possible['bars_to_extreme'] < both_possible['bars_to_breakdown']).mean() * 100
        log(f"  When both possible, extreme first: {extreme_first:.1f}%")

    # Avg bars to target/stop
    success_dips = dips_df[dips_df['reached_extreme'] == True]
    if len(success_dips) > 0:
        avg_bars = success_dips['bars_to_extreme'].mean()
        log(f"  Avg bars to reach extreme: {avg_bars:.1f} ({avg_bars*5/60:.1f}h)")

    failed_dips = dips_df[dips_df['broke_down'] == True]
    if len(failed_dips) > 0:
        avg_bars = failed_dips['bars_to_breakdown'].mean()
        log(f"  Avg bars to breakdown: {avg_bars:.1f} ({avg_bars*5/60:.1f}h)")

    log(f"\nBy Currency:")
    for ccy in CURRENCIES:
        subset = dips_df[dips_df['currency'] == ccy]
        if len(subset) > 0:
            success = subset['reached_extreme'].mean() * 100
            log(f"  {ccy}: {len(subset):,} dips, {success:.1f}% reached extreme")

    log(f"\nBy Pullback Depth (how deep the dip was):")
    dips_df['depth_q'] = pd.qcut(dips_df['pullback_depth'].abs(), 4, labels=['Q1 (shallow)', 'Q2', 'Q3', 'Q4 (deep)'])
    for q in ['Q1 (shallow)', 'Q2', 'Q3', 'Q4 (deep)']:
        subset = dips_df[dips_df['depth_q'] == q]
        if len(subset) > 0:
            success = subset['reached_extreme'].mean() * 100
            failure = subset['broke_down'].mean() * 100
            log(f"  {q}: {len(subset):,} dips, {success:.1f}% success, {failure:.1f}% breakdown")

    log(f"\nBy Entry Velocity (momentum at dip):")
    # For long dips, positive velocity is good (bouncing up)
    # For short dips, negative velocity is good (bouncing down)
    dips_df['favorable_vel'] = np.where(
        dips_df['direction'] == 'long',
        dips_df['entry_vel'] > 0,
        dips_df['entry_vel'] < 0
    )

    favorable = dips_df[dips_df['favorable_vel'] == True]
    unfavorable = dips_df[dips_df['favorable_vel'] == False]

    if len(favorable) > 0:
        log(f"  Favorable velocity: {favorable['reached_extreme'].mean()*100:.1f}% success ({len(favorable):,} dips)")
    if len(unfavorable) > 0:
        log(f"  Unfavorable velocity: {unfavorable['reached_extreme'].mean()*100:.1f}% success ({len(unfavorable):,} dips)")

log("\n" + "=" * 70)
log("PART 3: DIP TRADING SIMULATION")
log("=" * 70)

# Simple dip trade: enter when MFC dips within extreme zone, exit at extreme or 0
# Target: MFC reaches ±0.5 again
# Stop: MFC crosses 0

log(f"\nSimulating dip trades...")

# Filter for better setups: favorable velocity
good_dips = dips_df[dips_df['favorable_vel'] == True].copy()

if len(good_dips) > 0:
    # Calculate win rate based on which happened first
    wins = 0
    losses = 0
    neither = 0

    for _, row in good_dips.iterrows():
        if row['reached_extreme'] and row['broke_down']:
            # Both happened, which first?
            if row['bars_to_extreme'] < row['bars_to_breakdown']:
                wins += 1
            else:
                losses += 1
        elif row['reached_extreme']:
            wins += 1
        elif row['broke_down']:
            losses += 1
        else:
            neither += 1

    total = wins + losses
    if total > 0:
        wr = wins / total * 100
        log(f"\nDip Trading Results (favorable velocity only):")
        log(f"  Total setups: {len(good_dips):,}")
        log(f"  Wins (hit target): {wins:,}")
        log(f"  Losses (stopped out): {losses:,}")
        log(f"  Neither (timeout): {neither:,}")
        log(f"  Win Rate: {wr:.1f}%")

        # Estimate R:R
        avg_win_bars = good_dips[good_dips['reached_extreme']]['bars_to_extreme'].mean()
        avg_loss_bars = good_dips[good_dips['broke_down']]['bars_to_breakdown'].mean()
        log(f"\n  Avg bars to win: {avg_win_bars:.1f}")
        log(f"  Avg bars to loss: {avg_loss_bars:.1f}")

# Also check: deep pullbacks only
log(f"\nDeep pullbacks only (Q3-Q4):")
deep_dips = dips_df[dips_df['depth_q'].isin(['Q3', 'Q4 (deep)'])]
if len(deep_dips) > 0:
    wins = 0
    losses = 0
    for _, row in deep_dips.iterrows():
        if row['reached_extreme'] and row['broke_down']:
            if row['bars_to_extreme'] < row['bars_to_breakdown']:
                wins += 1
            else:
                losses += 1
        elif row['reached_extreme']:
            wins += 1
        elif row['broke_down']:
            losses += 1

    total = wins + losses
    if total > 0:
        log(f"  Win Rate: {wins/total*100:.1f}% ({wins}/{total})")

log(f"\nCompleted: {datetime.now()}")
