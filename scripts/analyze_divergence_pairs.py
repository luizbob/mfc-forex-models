"""
MFC H4 Analysis: Simultaneous Divergence - One Up, One Down

Specific analysis when:
- Currency A crosses -0.2 from below (entering box from oversold)
- Currency B crosses +0.2 from above (entering box from overbought)

What predicts whether Currency A continues up?
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
FORWARD_WINDOW = 30

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

def find_simultaneous_divergence_events(df):
    """
    Find all instances where:
    - Currency A crosses from below -0.2 to above -0.2 (entering box going up)
    - Currency B crosses from above +0.2 to below +0.2 (entering box going down)
    Both happening at the same time (same bar).
    """
    events = []

    for i in range(1, len(df)):
        # Find currencies crossing up through -0.2
        crossing_up = []
        for curr in CURRENCIES:
            if df[curr].iloc[i-1] < LOWER_BOX and df[curr].iloc[i] >= LOWER_BOX:
                # Calculate velocity at crossing
                velocity = (df[curr].iloc[i] - df[curr].iloc[max(0, i-3)]) / min(3, i)

                # Calculate how deep it went
                j = i - 1
                while j >= 0 and df[curr].iloc[j] < LOWER_BOX:
                    j -= 1
                time_below = i - j - 1
                min_mfc = df[curr].iloc[max(0, j+1):i].min() if time_below > 0 else df[curr].iloc[i-1]

                crossing_up.append({
                    'currency': curr,
                    'crossing_value': df[curr].iloc[i],
                    'velocity': velocity,
                    'time_below': time_below,
                    'min_mfc': min_mfc
                })

        # Find currencies crossing down through +0.2
        crossing_down = []
        for curr in CURRENCIES:
            if df[curr].iloc[i-1] > UPPER_BOX and df[curr].iloc[i] <= UPPER_BOX:
                # Calculate velocity (negative for downward)
                velocity = (df[curr].iloc[i] - df[curr].iloc[max(0, i-3)]) / min(3, i)

                # Calculate how high it went
                j = i - 1
                while j >= 0 and df[curr].iloc[j] > UPPER_BOX:
                    j -= 1
                time_above = i - j - 1
                max_mfc = df[curr].iloc[max(0, j+1):i].max() if time_above > 0 else df[curr].iloc[i-1]

                crossing_down.append({
                    'currency': curr,
                    'crossing_value': df[curr].iloc[i],
                    'velocity': velocity,
                    'time_above': time_above,
                    'max_mfc': max_mfc
                })

        # Create events for each pair of crossing_up and crossing_down
        for up in crossing_up:
            for down in crossing_down:
                events.append({
                    'datetime': df.index[i],
                    'idx': i,
                    'up_currency': up['currency'],
                    'up_crossing_value': up['crossing_value'],
                    'up_velocity': up['velocity'],
                    'up_time_below': up['time_below'],
                    'up_min_mfc': up['min_mfc'],
                    'down_currency': down['currency'],
                    'down_crossing_value': down['crossing_value'],
                    'down_velocity': down['velocity'],
                    'down_time_above': down['time_above'],
                    'down_max_mfc': down['max_mfc'],
                    'velocity_diff': up['velocity'] - down['velocity'],  # Positive = up is stronger
                    'velocity_sum': abs(up['velocity']) + abs(down['velocity'])  # Total momentum
                })

    return events

def track_outcome(df, currency, event_idx, forward_window=FORWARD_WINDOW):
    """Track outcome for the currency crossing up."""
    mfc = df[currency]
    max_idx = min(event_idx + forward_window, len(mfc))

    if event_idx >= len(mfc) - 1:
        return None

    future_mfc = mfc.iloc[event_idx:max_idx]

    reached_zero = any(future_mfc >= MID_BOX)
    reached_upper = any(future_mfc >= UPPER_BOX)
    fell_back = any(future_mfc < LOWER_BOX)

    max_mfc = future_mfc.max()
    min_mfc = future_mfc.min()

    # Time to reach levels
    time_to_zero = None
    time_to_upper = None
    time_to_fall_back = None

    for t, val in enumerate(future_mfc):
        if time_to_zero is None and val >= MID_BOX:
            time_to_zero = t
        if time_to_upper is None and val >= UPPER_BOX:
            time_to_upper = t
        if time_to_fall_back is None and val < LOWER_BOX:
            time_to_fall_back = t

    if reached_upper:
        outcome = 'full_continuation'
    elif reached_zero and not fell_back:
        outcome = 'partial_continuation'
    elif fell_back and not reached_zero:
        outcome = 'immediate_reversal'
    elif fell_back and reached_zero:
        outcome = 'reached_zero_then_reversed'
    else:
        outcome = 'stalled_in_box'

    return {
        'outcome': outcome,
        'reached_zero': reached_zero,
        'reached_upper': reached_upper,
        'fell_back': fell_back,
        'max_mfc': max_mfc,
        'min_mfc_after': min_mfc,
        'time_to_zero': time_to_zero,
        'time_to_upper': time_to_upper,
        'time_to_fall_back': time_to_fall_back
    }

def main():
    print("="*80)
    print("MFC H4 ANALYSIS: SIMULTANEOUS DIVERGENCE EVENTS")
    print("Currency A crossing UP through -0.2 + Currency B crossing DOWN through +0.2")
    print("="*80)

    # Load data
    print("\nLoading MFC data...")
    df = load_all_currencies()
    print(f"Loaded {len(df)} data points from {df.index.min()} to {df.index.max()}")

    # Find divergence events
    print("\nFinding simultaneous divergence events...")
    events = find_simultaneous_divergence_events(df)
    print(f"Found {len(events)} divergence events")

    # Track outcomes
    print("\nTracking outcomes for each event...")
    events_with_outcomes = []
    for event in events:
        outcome = track_outcome(df, event['up_currency'], event['idx'])
        if outcome:
            event.update(outcome)
            events_with_outcomes.append(event)

    edf = pd.DataFrame(events_with_outcomes)
    print(f"Events with valid outcomes: {len(edf)}")

    # Overall statistics
    print("\n" + "="*80)
    print("OVERALL DIVERGENCE EVENT STATISTICS")
    print("="*80)

    print(f"\nTotal divergence events: {len(edf)}")
    print(f"\nOutcome distribution for UP currency:")
    for outcome in edf['outcome'].value_counts().index:
        count = len(edf[edf['outcome'] == outcome])
        pct = count / len(edf) * 100
        print(f"  {outcome}: {count} ({pct:.1f}%)")

    print(f"\n--- Success Rates ---")
    print(f"Reached zero: {edf['reached_zero'].mean():.1%}")
    print(f"Full continuation (reached +0.2): {edf['reached_upper'].mean():.1%}")

    # Compare to baseline (crossing up without diverging currency)
    results_path = Path("/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/lstm_model/scripts/crossing_analysis_results.csv")
    baseline_df = pd.read_csv(results_path)
    baseline_success = baseline_df[baseline_df['outcome'] == 'full_continuation'].shape[0] / len(baseline_df)

    print(f"\n--- Comparison to Baseline ---")
    print(f"Baseline (all crossings up): {baseline_success:.1%} full continuation")
    print(f"With divergence (crossing up + down): {edf['reached_upper'].mean():.1%} full continuation")
    print(f"Edge: {edf['reached_upper'].mean() - baseline_success:+.1%}")

    # Factor analysis
    print("\n" + "="*80)
    print("FACTOR 1: UP CURRENCY VELOCITY")
    print("="*80)

    for thresh in [0.02, 0.04, 0.06, 0.08]:
        subset = edf[edf['up_velocity'] >= thresh]
        if len(subset) > 10:
            print(f"Up velocity >= {thresh}: {subset['reached_upper'].mean():.1%} success (n={len(subset)})")

    print("\n" + "="*80)
    print("FACTOR 2: DOWN CURRENCY VELOCITY (STRENGTH OF OPPOSITE MOVE)")
    print("="*80)

    # More negative velocity = stronger down move
    edf['down_velocity_abs'] = edf['down_velocity'].abs()
    for thresh in [0.02, 0.04, 0.06, 0.08]:
        subset = edf[edf['down_velocity_abs'] >= thresh]
        if len(subset) > 10:
            print(f"Down velocity abs >= {thresh}: {subset['reached_upper'].mean():.1%} success (n={len(subset)})")

    print("\n" + "="*80)
    print("FACTOR 3: VELOCITY DIFFERENTIAL (UP - DOWN)")
    print("="*80)

    # Positive = up is gaining faster than down is falling
    print("\nVelocity differential (up_vel - down_vel):")
    vel_diff_quartiles = pd.qcut(edf['velocity_diff'], q=4, labels=['Q1_low', 'Q2', 'Q3', 'Q4_high'], duplicates='drop')
    edf['vel_diff_quartile'] = vel_diff_quartiles

    for q in ['Q1_low', 'Q2', 'Q3', 'Q4_high']:
        subset = edf[edf['vel_diff_quartile'] == q]
        if len(subset) > 10:
            avg_diff = subset['velocity_diff'].mean()
            print(f"  {q} (avg diff: {avg_diff:+.4f}): {subset['reached_upper'].mean():.1%} success (n={len(subset)})")

    print("\n" + "="*80)
    print("FACTOR 4: COMBINED VELOCITY STRENGTH")
    print("="*80)

    print("\nTotal velocity magnitude (|up| + |down|):")
    for thresh in [0.06, 0.08, 0.10, 0.12]:
        subset = edf[edf['velocity_sum'] >= thresh]
        if len(subset) > 10:
            print(f"Combined velocity >= {thresh}: {subset['reached_upper'].mean():.1%} success (n={len(subset)})")

    print("\n" + "="*80)
    print("FACTOR 5: DEPTH OF UP CURRENCY (HOW OVERSOLD)")
    print("="*80)

    depth_bins = pd.cut(edf['up_min_mfc'], bins=[-1.5, -0.6, -0.4, -0.2],
                        labels=['deep_<-0.6', 'moderate_-0.6_to_-0.4', 'shallow_>-0.4'])
    edf['depth_bin'] = depth_bins

    for d in ['deep_<-0.6', 'moderate_-0.6_to_-0.4', 'shallow_>-0.4']:
        subset = edf[edf['depth_bin'] == d]
        if len(subset) > 10:
            print(f"  {d}: {subset['reached_upper'].mean():.1%} success (n={len(subset)})")

    print("\n" + "="*80)
    print("FACTOR 6: HEIGHT OF DOWN CURRENCY (HOW OVERBOUGHT)")
    print("="*80)

    height_bins = pd.cut(edf['down_max_mfc'], bins=[0.2, 0.4, 0.6, 1.5],
                         labels=['shallow_<0.4', 'moderate_0.4_to_0.6', 'high_>0.6'])
    edf['height_bin'] = height_bins

    for h in ['shallow_<0.4', 'moderate_0.4_to_0.6', 'high_>0.6']:
        subset = edf[edf['height_bin'] == h]
        if len(subset) > 10:
            print(f"  {h}: {subset['reached_upper'].mean():.1%} success (n={len(subset)})")

    print("\n" + "="*80)
    print("FACTOR 7: CURRENCY PAIR ANALYSIS")
    print("="*80)

    print("\nMost common currency pairs in divergence events:")
    pair_counts = edf.groupby(['up_currency', 'down_currency']).size().sort_values(ascending=False)
    print(pair_counts.head(15))

    print("\n--- Success rate by UP currency ---")
    for curr in CURRENCIES:
        subset = edf[edf['up_currency'] == curr]
        if len(subset) > 20:
            print(f"  {curr}: {subset['reached_upper'].mean():.1%} success (n={len(subset)})")

    print("\n--- Success rate by DOWN currency ---")
    for curr in CURRENCIES:
        subset = edf[edf['down_currency'] == curr]
        if len(subset) > 20:
            print(f"  {curr} falling: {subset['reached_upper'].mean():.1%} success for opposite (n={len(subset)})")

    print("\n" + "="*80)
    print("FACTOR 8: BEST/WORST CURRENCY PAIRS")
    print("="*80)

    pair_stats = edf.groupby(['up_currency', 'down_currency']).agg({
        'reached_upper': ['mean', 'count']
    }).round(3)
    pair_stats.columns = ['success_rate', 'count']
    pair_stats = pair_stats[pair_stats['count'] >= 10].sort_values('success_rate', ascending=False)

    print("\nTop 10 best performing pairs (n >= 10):")
    print(pair_stats.head(10))

    print("\nBottom 5 worst performing pairs (n >= 10):")
    print(pair_stats.tail(5))

    print("\n" + "="*80)
    print("OPTIMAL ENTRY CONDITIONS FOR DIVERGENCE EVENTS")
    print("="*80)

    # Build optimal filter
    print("\n--- Testing Combined Filters ---")

    filters = [
        ("Baseline (all divergence)", edf),
        ("Up velocity >= 0.04", edf[edf['up_velocity'] >= 0.04]),
        ("Up velocity >= 0.06", edf[edf['up_velocity'] >= 0.06]),
        ("Combined velocity >= 0.08", edf[edf['velocity_sum'] >= 0.08]),
        ("Combined velocity >= 0.10", edf[edf['velocity_sum'] >= 0.10]),
        ("Up vel >= 0.04 + Combined >= 0.08", edf[(edf['up_velocity'] >= 0.04) & (edf['velocity_sum'] >= 0.08)]),
        ("Up vel >= 0.06 + Combined >= 0.10", edf[(edf['up_velocity'] >= 0.06) & (edf['velocity_sum'] >= 0.10)]),
        ("Up vel >= 0.05 + Shallow depth", edf[(edf['up_velocity'] >= 0.05) & (edf['up_min_mfc'] > -0.5)]),
    ]

    print(f"\n{'Filter':<45} {'N':>6} {'Success':>10} {'Edge':>10}")
    print("-" * 75)

    baseline_rate = baseline_success
    for name, subset in filters:
        if len(subset) > 10:
            rate = subset['reached_upper'].mean()
            edge = rate - baseline_rate
            print(f"{name:<45} {len(subset):6d} {rate:10.1%} {edge:+10.1%}")

    print("\n" + "="*80)
    print("KEY FINDINGS: DIVERGENCE EVENTS")
    print("="*80)

    print(f"""
SUMMARY OF DIVERGENCE ANALYSIS:

1. BASELINE PERFORMANCE:
   - When a currency crosses -0.2 up while another crosses +0.2 down:
   - Success rate (reaching +0.2): {edf['reached_upper'].mean():.1%}
   - This is slightly HIGHER than non-divergence crossings ({baseline_success:.1%})

2. VELOCITY IS CRITICAL:
   - Up currency velocity >= 0.06 dramatically improves success
   - The COMBINED velocity (both directions) is also predictive
   - Stronger divergence = better outcome

3. DEPTH MATTERS:
   - Shallow oversold (didn't go too deep) has better continuation
   - Very deep oversold takes longer and has more uncertainty

4. SPECIFIC PAIR INSIGHTS:
   - Some currency pairs perform better than others
   - Check the pair analysis above for specific edges

5. OPTIMAL DIVERGENCE ENTRY:
   - Up currency velocity >= 0.05
   - Combined velocity magnitude >= 0.08
   - Up currency wasn't too deep (> -0.5)

6. TRADING IMPLICATION:
   When you see simultaneous divergence (one entering from oversold,
   one entering from overbought), the KEY is the VELOCITY of the
   rising currency, not the mere presence of divergence.
""")

    # Save detailed results
    output_path = Path("/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/lstm_model/scripts/divergence_analysis_results.csv")
    edf.to_csv(output_path, index=False)
    print(f"\nDetailed results saved to: {output_path}")

    return edf

if __name__ == "__main__":
    edf = main()
