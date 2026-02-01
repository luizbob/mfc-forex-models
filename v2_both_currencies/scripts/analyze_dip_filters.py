"""
Find Filters That Make Dip Trading Profitable
==============================================
The raw dip has 36% win rate. Can we find conditions that improve it?
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
log("FINDING PROFITABLE DIP FILTERS")
log("=" * 70)
log(f"Started: {datetime.now()}")

DATA_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/data')
CLEANED_DIR = DATA_DIR / 'cleaned'

CURRENCIES = ['EUR', 'USD', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD', 'NZD']

# Load all timeframes
log("\nLoading MFC data...")
mfc_m5 = {}
mfc_m15 = {}
mfc_m30 = {}
mfc_h1 = {}
mfc_h4 = {}

for ccy in CURRENCIES:
    for tf, storage in [('M5', mfc_m5), ('M15', mfc_m15), ('M30', mfc_m30), ('H1', mfc_h1), ('H4', mfc_h4)]:
        df = pd.read_csv(CLEANED_DIR / f'mfc_currency_{ccy}_{tf}_clean.csv')
        df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
        df = df.set_index('datetime').sort_index()
        storage[ccy] = df['MFC']

log("Data loaded.")

# Parameters
DIP_ENTRY_MIN = 0.3
DIP_ENTRY_MAX = 0.45
EXTREME_LEVEL = 0.5
LOOKBACK_BARS = 50
FORWARD_BARS = 100

log("\n" + "=" * 70)
log("COLLECTING DIP DATA WITH FEATURES")
log("=" * 70)

all_dips = []

for ccy in CURRENCIES:
    log(f"\nProcessing {ccy}...")

    m5 = mfc_m5[ccy]
    arr = m5.values
    idx = m5.index

    # Align higher TFs to M5
    m15_aligned = mfc_m15[ccy].reindex(idx, method='ffill')
    m30_aligned = mfc_m30[ccy].reindex(idx, method='ffill')
    h1_aligned = mfc_h1[ccy].reindex(idx, method='ffill')
    h4_aligned = mfc_h4[ccy].reindex(idx, method='ffill')

    count = 0
    for i in range(LOOKBACK_BARS, len(arr) - FORWARD_BARS):
        val = arr[i]

        # Check for positive dip
        if DIP_ENTRY_MIN <= val <= DIP_ENTRY_MAX:
            lookback = arr[i-LOOKBACK_BARS:i]
            if lookback.max() >= EXTREME_LEVEL:
                # This is a dip
                future = arr[i+1:i+1+FORWARD_BARS]

                # Target and stop
                reached_target = (future >= EXTREME_LEVEL).any()
                hit_stop = (future <= 0).any()

                if reached_target:
                    bars_to_target = np.argmax(future >= EXTREME_LEVEL) + 1
                else:
                    bars_to_target = FORWARD_BARS + 1

                if hit_stop:
                    bars_to_stop = np.argmax(future <= 0) + 1
                else:
                    bars_to_stop = FORWARD_BARS + 1

                # Determine outcome
                if bars_to_target < bars_to_stop:
                    outcome = 1  # Win
                elif bars_to_stop <= FORWARD_BARS:
                    outcome = 0  # Loss
                else:
                    outcome = -1  # Timeout

                # Features
                # 1. M5 velocity (short term momentum)
                vel_m5_3 = (arr[i] - arr[i-3]) / 3
                vel_m5_6 = (arr[i] - arr[i-6]) / 6
                vel_m5_12 = (arr[i] - arr[i-12]) / 12

                # 2. Higher TF alignment
                m15_val = m15_aligned.iloc[i]
                m30_val = m30_aligned.iloc[i]
                h1_val = h1_aligned.iloc[i]
                h4_val = h4_aligned.iloc[i]

                # 3. How long since the extreme was hit?
                bars_since_extreme = LOOKBACK_BARS - np.argmax(lookback >= EXTREME_LEVEL)

                # 4. How deep is this dip from the prior extreme?
                prior_extreme = lookback.max()
                dip_depth = prior_extreme - val

                # 5. Is M5 bouncing? (velocity turning positive)
                is_bouncing = vel_m5_3 > 0

                # 6. Higher TF still in extreme zone?
                h1_in_extreme = h1_val >= 0.4
                h4_in_extreme = h4_val >= 0.4

                all_dips.append({
                    'currency': ccy,
                    'time': idx[i],
                    'dip_level': val,
                    'outcome': outcome,
                    'bars_to_target': bars_to_target,
                    'bars_to_stop': bars_to_stop,
                    # Features
                    'vel_m5_3': vel_m5_3,
                    'vel_m5_6': vel_m5_6,
                    'vel_m5_12': vel_m5_12,
                    'm15_val': m15_val,
                    'm30_val': m30_val,
                    'h1_val': h1_val,
                    'h4_val': h4_val,
                    'bars_since_extreme': bars_since_extreme,
                    'dip_depth': dip_depth,
                    'is_bouncing': is_bouncing,
                    'h1_in_extreme': h1_in_extreme,
                    'h4_in_extreme': h4_in_extreme,
                    'direction': 'long',
                })
                count += 1

        # Check for negative dip
        elif -DIP_ENTRY_MAX <= val <= -DIP_ENTRY_MIN:
            lookback = arr[i-LOOKBACK_BARS:i]
            if lookback.min() <= -EXTREME_LEVEL:
                future = arr[i+1:i+1+FORWARD_BARS]

                reached_target = (future <= -EXTREME_LEVEL).any()
                hit_stop = (future >= 0).any()

                if reached_target:
                    bars_to_target = np.argmax(future <= -EXTREME_LEVEL) + 1
                else:
                    bars_to_target = FORWARD_BARS + 1

                if hit_stop:
                    bars_to_stop = np.argmax(future >= 0) + 1
                else:
                    bars_to_stop = FORWARD_BARS + 1

                if bars_to_target < bars_to_stop:
                    outcome = 1
                elif bars_to_stop <= FORWARD_BARS:
                    outcome = 0
                else:
                    outcome = -1

                vel_m5_3 = (arr[i] - arr[i-3]) / 3
                vel_m5_6 = (arr[i] - arr[i-6]) / 6
                vel_m5_12 = (arr[i] - arr[i-12]) / 12

                m15_val = m15_aligned.iloc[i]
                m30_val = m30_aligned.iloc[i]
                h1_val = h1_aligned.iloc[i]
                h4_val = h4_aligned.iloc[i]

                bars_since_extreme = LOOKBACK_BARS - np.argmax(lookback <= -EXTREME_LEVEL)
                prior_extreme = lookback.min()
                dip_depth = val - prior_extreme

                # For short, bouncing means velocity negative
                is_bouncing = vel_m5_3 < 0
                h1_in_extreme = h1_val <= -0.4
                h4_in_extreme = h4_val <= -0.4

                all_dips.append({
                    'currency': ccy,
                    'time': idx[i],
                    'dip_level': val,
                    'outcome': outcome,
                    'bars_to_target': bars_to_target,
                    'bars_to_stop': bars_to_stop,
                    'vel_m5_3': vel_m5_3,
                    'vel_m5_6': vel_m5_6,
                    'vel_m5_12': vel_m5_12,
                    'm15_val': m15_val,
                    'm30_val': m30_val,
                    'h1_val': h1_val,
                    'h4_val': h4_val,
                    'bars_since_extreme': bars_since_extreme,
                    'dip_depth': dip_depth,
                    'is_bouncing': is_bouncing,
                    'h1_in_extreme': h1_in_extreme,
                    'h4_in_extreme': h4_in_extreme,
                    'direction': 'short',
                })
                count += 1

    log(f"  {ccy}: {count:,} dips")

df = pd.DataFrame(all_dips)
log(f"\nTotal dips: {len(df):,}")

# Remove timeouts for analysis
df_valid = df[df['outcome'] >= 0]
log(f"Valid (win/loss): {len(df_valid):,}")

log("\n" + "=" * 70)
log("BASELINE")
log("=" * 70)

baseline_wr = df_valid['outcome'].mean() * 100
log(f"Baseline Win Rate: {baseline_wr:.1f}%")

log("\n" + "=" * 70)
log("FILTER ANALYSIS")
log("=" * 70)

def analyze_filter(name, mask):
    subset = df_valid[mask]
    if len(subset) < 100:
        return
    wr = subset['outcome'].mean() * 100
    count = len(subset)
    log(f"  {name}: {wr:.1f}% WR ({count:,} trades)")

# 1. Is bouncing (velocity favorable)
log("\n1. IS BOUNCING (velocity in trade direction):")
analyze_filter("Bouncing", df_valid['is_bouncing'] == True)
analyze_filter("Not bouncing", df_valid['is_bouncing'] == False)

# 2. H1 still in extreme
log("\n2. H1 STILL IN EXTREME ZONE (>=0.4):")
analyze_filter("H1 in extreme", df_valid['h1_in_extreme'] == True)
analyze_filter("H1 NOT in extreme", df_valid['h1_in_extreme'] == False)

# 3. H4 still in extreme
log("\n3. H4 STILL IN EXTREME ZONE (>=0.4):")
analyze_filter("H4 in extreme", df_valid['h4_in_extreme'] == True)
analyze_filter("H4 NOT in extreme", df_valid['h4_in_extreme'] == False)

# 4. Both H1 and H4 in extreme
log("\n4. BOTH H1 AND H4 IN EXTREME:")
both_in_extreme = (df_valid['h1_in_extreme'] == True) & (df_valid['h4_in_extreme'] == True)
analyze_filter("Both in extreme", both_in_extreme)
analyze_filter("Not both", ~both_in_extreme)

# 5. Bars since extreme
log("\n5. BARS SINCE HITTING EXTREME:")
df_valid['since_q'] = pd.qcut(df_valid['bars_since_extreme'], 4, labels=['Q1 (recent)', 'Q2', 'Q3', 'Q4 (old)'])
for q in ['Q1 (recent)', 'Q2', 'Q3', 'Q4 (old)']:
    analyze_filter(q, df_valid['since_q'] == q)

# 6. Dip depth
log("\n6. DIP DEPTH:")
df_valid['depth_q'] = pd.qcut(df_valid['dip_depth'].abs(), 4, labels=['Q1 (shallow)', 'Q2', 'Q3', 'Q4 (deep)'])
for q in ['Q1 (shallow)', 'Q2', 'Q3', 'Q4 (deep)']:
    analyze_filter(q, df_valid['depth_q'] == q)

# 7. M5 velocity
log("\n7. M5 VELOCITY (3-bar):")
df_valid['vel_q'] = pd.qcut(df_valid['vel_m5_3'], 4, labels=['Q1 (neg)', 'Q2', 'Q3', 'Q4 (pos)'])
for q in ['Q1 (neg)', 'Q2', 'Q3', 'Q4 (pos)']:
    analyze_filter(q, df_valid['vel_q'] == q)

# 8. Combined filters
log("\n" + "=" * 70)
log("COMBINED FILTERS")
log("=" * 70)

# Best combination: bouncing + H1 in extreme + H4 in extreme
log("\n1. Bouncing + H1 extreme + H4 extreme:")
mask = (df_valid['is_bouncing'] == True) & (df_valid['h1_in_extreme'] == True) & (df_valid['h4_in_extreme'] == True)
analyze_filter("Combined", mask)

# Bouncing + recent (bars since < median)
log("\n2. Bouncing + Recent extreme (Q1-Q2):")
mask = (df_valid['is_bouncing'] == True) & (df_valid['since_q'].isin(['Q1 (recent)', 'Q2']))
analyze_filter("Combined", mask)

# H1+H4 extreme + recent
log("\n3. H1+H4 extreme + Recent:")
mask = (df_valid['h1_in_extreme'] == True) & (df_valid['h4_in_extreme'] == True) & (df_valid['since_q'].isin(['Q1 (recent)', 'Q2']))
analyze_filter("Combined", mask)

# All together
log("\n4. Bouncing + H1+H4 extreme + Recent:")
mask = (df_valid['is_bouncing'] == True) & (df_valid['h1_in_extreme'] == True) & (df_valid['h4_in_extreme'] == True) & (df_valid['since_q'].isin(['Q1 (recent)', 'Q2']))
analyze_filter("Combined", mask)

# By currency for best filter
log("\n" + "=" * 70)
log("BEST FILTER BY CURRENCY")
log("=" * 70)

best_mask = (df_valid['is_bouncing'] == True) & (df_valid['h1_in_extreme'] == True) & (df_valid['h4_in_extreme'] == True)
best_subset = df_valid[best_mask]

log(f"\nFilter: Bouncing + H1 extreme + H4 extreme")
for ccy in CURRENCIES:
    subset = best_subset[best_subset['currency'] == ccy]
    if len(subset) > 0:
        wr = subset['outcome'].mean() * 100
        log(f"  {ccy}: {wr:.1f}% WR ({len(subset):,} trades)")

# Estimate trades per year
log("\n" + "=" * 70)
log("TRADE FREQUENCY")
log("=" * 70)

# Count by year
df['year'] = pd.to_datetime(df['time']).dt.year
best_all = df[(df['is_bouncing'] == True) & (df['h1_in_extreme'] == True) & (df['h4_in_extreme'] == True)]
trades_by_year = best_all.groupby('year').size()
log(f"\nTrades per year (best filter):")
for year, count in trades_by_year.items():
    log(f"  {year}: {count:,}")

avg_per_year = trades_by_year.mean()
log(f"\nAvg per year: {avg_per_year:.0f} trades ({avg_per_year/365:.1f}/day)")

log(f"\nCompleted: {datetime.now()}")
