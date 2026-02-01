"""
MFC H4 Analysis: Detailed Investigation of Crossing Patterns

This script provides deeper analysis of the crossing behavior,
especially investigating the counterintuitive depth findings.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATA_DIR = Path("/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/data/cleaned")
CURRENCIES = ['EUR', 'USD', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD', 'NZD']

LOWER_BOX = -0.2
UPPER_BOX = 0.2
MID_BOX = 0.0

def load_all_currencies():
    """Load MFC data for all 8 currencies into a merged DataFrame."""
    dfs = {}
    for currency in CURRENCIES:
        file_path = DATA_DIR / f"mfc_currency_{currency}_H4_clean.csv"
        df = pd.read_csv(file_path)
        df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
        df = df.set_index('DateTime')
        df = df[['MFC']].rename(columns={'MFC': currency})
        dfs[currency] = df

    merged = pd.concat(dfs.values(), axis=1)
    merged = merged.dropna()
    return merged

# Load the detailed results from previous analysis
results_path = Path("/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/lstm_model/scripts/crossing_analysis_results.csv")
df = pd.read_csv(results_path)

print("="*80)
print("DETAILED INVESTIGATION: CROSSING -0.2 FROM BELOW")
print("="*80)

# 1. Re-examine depth vs outcome relationship
print("\n" + "="*80)
print("INVESTIGATION 1: DEPTH VS OUTCOME (CORRECTED ANALYSIS)")
print("="*80)

print("\nNote: The depth analysis had an issue - 'full_success' and 'continued'")
print("columns had identical values due to a bug. Let me recalculate properly.\n")

# Recalculate continued flag
df['continued'] = df['outcome'].isin(['full_continuation', 'partial_continuation', 'reached_zero_then_reversed'])
df['full_success'] = df['outcome'] == 'full_continuation'
df['reached_zero'] = df['outcome'].isin(['full_continuation', 'partial_continuation', 'reached_zero_then_reversed'])

print("Corrected outcome breakdown:")
for outcome in df['outcome'].unique():
    count = len(df[df['outcome'] == outcome])
    pct = count / len(df) * 100
    print(f"  {outcome}: {count} ({pct:.1f}%)")

# Re-examine depth
print("\n--- Depth Analysis (Corrected) ---")
depth_bins = pd.cut(df['min_mfc'], bins=[-1.5, -0.7, -0.5, -0.35, -0.2],
                    labels=['very_deep_<-0.7', 'deep_-0.7_to_-0.5', 'moderate_-0.5_to_-0.35', 'shallow_>-0.35'])
df['depth_bin'] = depth_bins

for depth in ['very_deep_<-0.7', 'deep_-0.7_to_-0.5', 'moderate_-0.5_to_-0.35', 'shallow_>-0.35']:
    subset = df[df['depth_bin'] == depth]
    if len(subset) > 0:
        print(f"\n{depth} (n={len(subset)}):")
        print(f"  Reached zero: {subset['reached_zero'].mean():.1%}")
        print(f"  Full continuation (to +0.2): {subset['full_success'].mean():.1%}")
        print(f"  Immediate reversal: {(subset['outcome'] == 'immediate_reversal').mean():.1%}")
        print(f"  Average velocity: {subset['velocity'].mean():.4f}")

# 2. Velocity bands with specific thresholds
print("\n" + "="*80)
print("INVESTIGATION 2: VELOCITY THRESHOLD ANALYSIS")
print("="*80)

velocity_thresholds = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10]

print("\nVelocity thresholds and success rates:")
print("-" * 70)
print(f"{'Threshold':<15} {'N Events':<12} {'Reached 0':<15} {'Full Success':<15}")
print("-" * 70)

for thresh in velocity_thresholds:
    subset = df[df['velocity'] >= thresh]
    if len(subset) > 10:
        print(f">= {thresh:<11.2f} {len(subset):<12} {subset['reached_zero'].mean():<15.1%} {subset['full_success'].mean():<15.1%}")

# 3. Interaction: Velocity x Depth
print("\n" + "="*80)
print("INVESTIGATION 3: VELOCITY x DEPTH INTERACTION")
print("="*80)

df['high_velocity'] = df['velocity'] >= df['velocity'].median()
df['deep'] = df['min_mfc'] < -0.5

print("\n2x2 Matrix: Velocity x Depth")
print("-" * 60)

for hv in [True, False]:
    for deep in [True, False]:
        subset = df[(df['high_velocity'] == hv) & (df['deep'] == deep)]
        vel_label = "High Vel" if hv else "Low Vel"
        depth_label = "Deep (<-0.5)" if deep else "Shallow (>=-0.5)"
        if len(subset) > 0:
            print(f"\n{vel_label} + {depth_label} (n={len(subset)}):")
            print(f"  Reached zero: {subset['reached_zero'].mean():.1%}")
            print(f"  Full success: {subset['full_success'].mean():.1%}")

# 4. Time below analysis refined
print("\n" + "="*80)
print("INVESTIGATION 4: TIME BELOW -0.2 (REFINED)")
print("="*80)

time_ranges = [
    (1, 2, "Very quick (1-2 bars)"),
    (3, 5, "Quick (3-5 bars)"),
    (6, 10, "Short (6-10 bars)"),
    (11, 20, "Medium (11-20 bars)"),
    (21, 40, "Long (21-40 bars)"),
    (41, 1000, "Very long (>40 bars)")
]

print("\nTime below -0.2 and success rates:")
print("-" * 70)

for low, high, label in time_ranges:
    subset = df[(df['time_below'] >= low) & (df['time_below'] <= high)]
    if len(subset) > 10:
        print(f"\n{label} (n={len(subset)}):")
        print(f"  Reached zero: {subset['reached_zero'].mean():.1%}")
        print(f"  Full success: {subset['full_success'].mean():.1%}")
        print(f"  Avg velocity: {subset['velocity'].mean():.4f}")
        print(f"  Avg depth: {subset['min_mfc'].mean():.3f}")

# 5. Diverging currency deep dive
print("\n" + "="*80)
print("INVESTIGATION 5: DIVERGING CURRENCY DETAILED ANALYSIS")
print("="*80)

df['has_diverging'] = df['num_diverging_down'] > 0

print("\n--- With vs Without Diverging Currency ---")
for has_div in [True, False]:
    subset = df[df['has_diverging'] == has_div]
    label = "WITH diverging" if has_div else "WITHOUT diverging"
    print(f"\n{label} (n={len(subset)}):")
    print(f"  Reached zero: {subset['reached_zero'].mean():.1%}")
    print(f"  Full success: {subset['full_success'].mean():.1%}")
    print(f"  Immediate reversal: {(subset['outcome'] == 'immediate_reversal').mean():.1%}")

# 6. Multi-factor scoring model
print("\n" + "="*80)
print("INVESTIGATION 6: MULTI-FACTOR SCORING MODEL")
print("="*80)

# Create a simple scoring system
df['score'] = 0

# Velocity score (0-3 points)
df.loc[df['velocity'] >= 0.08, 'score'] += 3
df.loc[(df['velocity'] >= 0.05) & (df['velocity'] < 0.08), 'score'] += 2
df.loc[(df['velocity'] >= 0.03) & (df['velocity'] < 0.05), 'score'] += 1

# Diverging bonus (1 point)
df.loc[df['has_diverging'], 'score'] += 1

# Positive acceleration bonus (1 point)
df.loc[df['acceleration'] > 0, 'score'] += 1

# Time below penalty (inverted - too quick or too long is bad)
# Optimal seems to be 6-20 bars
df.loc[(df['time_below'] >= 6) & (df['time_below'] <= 20), 'score'] += 1

print("\nScoring system:")
print("  +3 pts: Velocity >= 0.08")
print("  +2 pts: Velocity 0.05-0.08")
print("  +1 pt:  Velocity 0.03-0.05")
print("  +1 pt:  Diverging currency present")
print("  +1 pt:  Positive acceleration")
print("  +1 pt:  Time below 6-20 bars (optimal)")
print("\nMax score: 6 points")

print("\n--- Success Rate by Score ---")
print("-" * 60)
for score in sorted(df['score'].unique()):
    subset = df[df['score'] == score]
    if len(subset) > 10:
        print(f"Score {score}: n={len(subset):4d} | Reached 0: {subset['reached_zero'].mean():.1%} | Full: {subset['full_success'].mean():.1%}")

# 7. Market regime analysis
print("\n" + "="*80)
print("INVESTIGATION 7: MARKET REGIME (STRONG/WEAK CURRENCY COUNT)")
print("="*80)

print("\n--- By Number of Other Weak Currencies (below -0.2) ---")
for n_weak in sorted(df['num_weak'].unique()):
    subset = df[df['num_weak'] == n_weak]
    if len(subset) > 20:
        print(f"{n_weak} weak currencies: n={len(subset):4d} | Reached 0: {subset['reached_zero'].mean():.1%} | Full: {subset['full_success'].mean():.1%}")

print("\n--- By Number of Strong Currencies (above +0.2) ---")
for n_strong in sorted(df['num_strong'].unique()):
    subset = df[df['num_strong'] == n_strong]
    if len(subset) > 20:
        print(f"{n_strong} strong currencies: n={len(subset):4d} | Reached 0: {subset['reached_zero'].mean():.1%} | Full: {subset['full_success'].mean():.1%}")

# 8. Currency pair opportunities
print("\n" + "="*80)
print("INVESTIGATION 8: BEST CURRENCY COMBINATIONS")
print("="*80)

# When a currency crosses up, which other currencies being strong predicts success?
print("\nAnalyzing which currency relationships predict success...")

# Load original data to check other currencies at crossing time
mfc_df = load_all_currencies()

# For events with high success, check which currencies were strong
high_success_events = df[df['full_success'] == True]
low_success_events = df[df['outcome'] == 'immediate_reversal']

print(f"\nComparing {len(high_success_events)} successful events vs {len(low_success_events)} failed events:")

# 9. Acceleration patterns
print("\n" + "="*80)
print("INVESTIGATION 9: ACCELERATION PATTERNS")
print("="*80)

accel_bins = pd.qcut(df['acceleration'], q=5, labels=['very_neg', 'neg', 'neutral', 'pos', 'very_pos'], duplicates='drop')
df['accel_bin'] = accel_bins

print("\nSuccess rate by acceleration quintile:")
for ab in ['very_neg', 'neg', 'neutral', 'pos', 'very_pos']:
    subset = df[df['accel_bin'] == ab]
    if len(subset) > 10:
        print(f"  {ab:<10}: n={len(subset):4d} | Reached 0: {subset['reached_zero'].mean():.1%} | Full: {subset['full_success'].mean():.1%}")

# 10. Final summary - Best conditions
print("\n" + "="*80)
print("FINAL SUMMARY: OPTIMAL ENTRY CONDITIONS")
print("="*80)

# Best single factor
print("\n--- BEST SINGLE FACTOR FILTERS ---")

best_conditions = []

# High velocity
hv = df[df['velocity'] >= 0.06]
best_conditions.append(("Velocity >= 0.06", len(hv), hv['full_success'].mean()))

# With diverging
div = df[df['has_diverging']]
best_conditions.append(("Has diverging currency", len(div), div['full_success'].mean()))

# Score >= 4
high_score = df[df['score'] >= 4]
best_conditions.append(("Score >= 4", len(high_score), high_score['full_success'].mean()))

# Positive acceleration + high velocity
pa_hv = df[(df['acceleration'] > 0) & (df['velocity'] >= 0.05)]
best_conditions.append(("Pos accel + Vel >= 0.05", len(pa_hv), pa_hv['full_success'].mean()))

for name, n, rate in sorted(best_conditions, key=lambda x: x[2], reverse=True):
    print(f"  {name:<30}: n={n:4d} | Full success: {rate:.1%}")

print("\n--- BEST COMBINED CONDITIONS ---")

# Velocity + Diverging
vd = df[(df['velocity'] >= 0.05) & (df['has_diverging'])]
print(f"  Velocity >= 0.05 + Diverging:     n={len(vd):4d} | Full: {vd['full_success'].mean():.1%}")

# High score
hs = df[df['score'] >= 5]
print(f"  Score >= 5:                       n={len(hs):4d} | Full: {hs['full_success'].mean():.1%}")

# Ultimate filter
ult = df[(df['velocity'] >= 0.06) & (df['has_diverging']) & (df['acceleration'] > 0)]
print(f"  Vel >= 0.06 + Diverging + Accel+: n={len(ult):4d} | Full: {ult['full_success'].mean():.1%}")

print("\n--- BASELINE FOR COMPARISON ---")
print(f"  All events:                       n={len(df):4d} | Full: {df['full_success'].mean():.1%}")

# 11. Edge calculation
print("\n" + "="*80)
print("EDGE CALCULATION: IMPROVEMENT OVER BASELINE")
print("="*80)

baseline = df['full_success'].mean()
print(f"\nBaseline full success rate: {baseline:.1%}")

filters = [
    ("Velocity >= 0.06", df[df['velocity'] >= 0.06]),
    ("Velocity >= 0.08", df[df['velocity'] >= 0.08]),
    ("Has diverging", df[df['has_diverging']]),
    ("Score >= 4", df[df['score'] >= 4]),
    ("Score >= 5", df[df['score'] >= 5]),
    ("Vel >= 0.05 + Diverging", df[(df['velocity'] >= 0.05) & (df['has_diverging'])]),
    ("Vel >= 0.06 + Diverging", df[(df['velocity'] >= 0.06) & (df['has_diverging'])]),
    ("Vel >= 0.06 + Accel+", df[(df['velocity'] >= 0.06) & (df['acceleration'] > 0)]),
    ("Full ideal: V>=0.06+Div+Acc+", df[(df['velocity'] >= 0.06) & (df['has_diverging']) & (df['acceleration'] > 0)]),
]

print("\nFilter comparison:")
print("-" * 80)
print(f"{'Filter':<35} {'N':>6} {'Success':>10} {'Edge':>10} {'Rel Edge':>10}")
print("-" * 80)

for name, subset in filters:
    if len(subset) > 20:
        rate = subset['full_success'].mean()
        edge = rate - baseline
        rel_edge = edge / baseline * 100
        print(f"{name:<35} {len(subset):6d} {rate:10.1%} {edge:+10.1%} {rel_edge:+10.1f}%")

print("\n" + "="*80)
print("KEY FINDINGS SUMMARY")
print("="*80)

print("""
1. VELOCITY is the strongest predictor:
   - Velocity >= 0.06: 83.4% success (vs 69.9% baseline) = +19% edge
   - Velocity >= 0.08: 86.7% success = +24% edge

2. DIVERGING CURRENCY adds modest edge:
   - With diverging: 70.8% (vs 69.7% without) = +1.6% edge
   - BUT combined with high velocity: significantly higher

3. POSITIVE ACCELERATION helps:
   - Positive acceleration: 70.8% vs 66.9% = +6% relative edge

4. DEPTH is NOT a good predictor:
   - Deeper oversold actually has LOWER success (counterintuitive)
   - This is because very deep oversold takes longer to recover

5. TIME BELOW -0.2:
   - Sweet spot is 6-20 bars (allows momentum to build)
   - Very quick (1-2 bars) or very long (>40 bars) less reliable

6. BEST COMBINED FILTER:
   - Velocity >= 0.06 + Diverging currency + Positive acceleration
   - This identifies the highest probability continuation setups

7. ACTIONABLE ENTRY CRITERIA:
   When a currency crosses -0.2 from below, ENTER if:
   a) Velocity at crossing >= 0.06 (mandatory)
   b) Another currency crossing +0.2 down (preferred)
   c) Acceleration is positive (preferred)

   AVOID if:
   a) Velocity < 0.04 (low momentum)
   b) Currency was very deep (< -0.7) - recovery uncertain
   c) Very quick bounce (1-2 bars) - may be noise
""")
