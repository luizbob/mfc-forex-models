"""
MFC H4 Analysis: Crossing -0.2 from Below - Continuation Predictors

This script analyzes what factors predict whether a currency continues rising
after crossing -0.2 from below (entering the box from oversold territory).
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

# Box levels
LOWER_BOX = -0.2
UPPER_BOX = 0.2
MID_BOX = 0.0

# Lookback and forward windows (in H4 bars)
FORWARD_WINDOW = 30  # 30 H4 bars = 5 days to track outcome

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

    # Merge all currencies on DateTime
    merged = pd.concat(dfs.values(), axis=1)
    merged = merged.dropna()  # Only keep rows where all currencies have data
    return merged

def calculate_velocity(series, window=3):
    """Calculate velocity (rate of change) of MFC."""
    return series.diff(window) / window

def calculate_acceleration(velocity, window=2):
    """Calculate acceleration (rate of change of velocity)."""
    return velocity.diff(window) / window

def find_crossing_up_events(df, currency):
    """
    Find all instances where a currency crosses from below -0.2 to above -0.2.
    Returns list of crossing events with context.
    """
    mfc = df[currency]
    events = []

    # Track when we're below -0.2
    below_box = mfc < LOWER_BOX

    for i in range(1, len(mfc)):
        # Crossing condition: was below -0.2, now at or above -0.2
        if below_box.iloc[i-1] and mfc.iloc[i] >= LOWER_BOX:
            # Find how long it was below -0.2
            time_below = 0
            j = i - 1
            while j >= 0 and mfc.iloc[j] < LOWER_BOX:
                time_below += 1
                j -= 1

            # Find minimum MFC value during the period below -0.2
            start_below_idx = max(0, i - time_below)
            min_mfc = mfc.iloc[start_below_idx:i].min()

            # Calculate velocity at crossing point
            if i >= 3:
                velocity = (mfc.iloc[i] - mfc.iloc[i-3]) / 3
            else:
                velocity = mfc.iloc[i] - mfc.iloc[i-1]

            # Calculate acceleration
            if i >= 5:
                vel_now = (mfc.iloc[i] - mfc.iloc[i-2]) / 2
                vel_prev = (mfc.iloc[i-2] - mfc.iloc[i-4]) / 2
                acceleration = vel_now - vel_prev
            else:
                acceleration = 0

            events.append({
                'datetime': df.index[i],
                'idx': i,
                'currency': currency,
                'crossing_value': mfc.iloc[i],
                'time_below': time_below,
                'min_mfc': min_mfc,
                'velocity': velocity,
                'acceleration': acceleration
            })

    return events

def find_diverging_currencies(df, event_idx, crossing_currency):
    """
    At the time of crossing, find currencies that are:
    1. Crossing +0.2 from above (going down)
    2. Already strong (above +0.2)
    3. Already weak (below -0.2)
    """
    result = {
        'diverging_down': [],  # Crossing +0.2 from above
        'strong_currencies': [],  # Above +0.2
        'weak_currencies': [],  # Below -0.2
        'in_box': []  # Between -0.2 and +0.2
    }

    for currency in CURRENCIES:
        if currency == crossing_currency:
            continue

        mfc_now = df[currency].iloc[event_idx]
        mfc_prev = df[currency].iloc[event_idx - 1] if event_idx > 0 else mfc_now

        # Check for diverging crossing (going down through +0.2)
        if mfc_prev > UPPER_BOX and mfc_now <= UPPER_BOX:
            result['diverging_down'].append(currency)

        # Current position
        if mfc_now > UPPER_BOX:
            result['strong_currencies'].append(currency)
        elif mfc_now < LOWER_BOX:
            result['weak_currencies'].append(currency)
        else:
            result['in_box'].append(currency)

    return result

def track_outcome(df, currency, event_idx, forward_window=FORWARD_WINDOW):
    """
    Track what happens after crossing -0.2 from below.
    Returns outcome classification and details.
    """
    mfc = df[currency]
    max_idx = min(event_idx + forward_window, len(mfc))

    if event_idx >= len(mfc) - 1:
        return None

    future_mfc = mfc.iloc[event_idx:max_idx]

    # Track key levels reached
    reached_zero = any(future_mfc >= MID_BOX)
    reached_upper = any(future_mfc >= UPPER_BOX)
    fell_back = any(future_mfc < LOWER_BOX)

    # Find maximum reached
    max_mfc = future_mfc.max()
    min_mfc = future_mfc.min()

    # Time to reach each level
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

    # Classify outcome
    if reached_upper:
        outcome = 'full_continuation'  # Made it all the way through the box
    elif reached_zero and not fell_back:
        outcome = 'partial_continuation'  # Made it to middle but not through
    elif fell_back and not reached_zero:
        outcome = 'immediate_reversal'  # Fell back without reaching zero
    elif fell_back and reached_zero:
        outcome = 'reached_zero_then_reversed'  # Got to zero but then failed
    else:
        outcome = 'stalled_in_box'  # Stayed between -0.2 and 0

    return {
        'outcome': outcome,
        'reached_zero': reached_zero,
        'reached_upper': reached_upper,
        'fell_back': fell_back,
        'max_mfc': max_mfc,
        'min_mfc': min_mfc,
        'time_to_zero': time_to_zero,
        'time_to_upper': time_to_upper,
        'time_to_fall_back': time_to_fall_back
    }

def analyze_predictive_factors(events_with_outcomes):
    """Analyze which factors predict continuation vs reversal."""

    df = pd.DataFrame(events_with_outcomes)

    if len(df) == 0:
        return None

    # Create binary outcome for easier analysis
    df['continued'] = df['outcome'].isin(['full_continuation', 'partial_continuation'])
    df['full_success'] = df['outcome'] == 'full_continuation'

    results = {}

    # 1. Velocity analysis
    print("\n" + "="*80)
    print("FACTOR 1: VELOCITY AT CROSSING POINT")
    print("="*80)

    velocity_bins = pd.qcut(df['velocity'], q=4, labels=['slow', 'medium_slow', 'medium_fast', 'fast'], duplicates='drop')
    df['velocity_bin'] = velocity_bins

    velocity_stats = df.groupby('velocity_bin').agg({
        'continued': ['mean', 'count'],
        'full_success': 'mean'
    }).round(3)
    print("\nContinuation rate by velocity quartile:")
    print(velocity_stats)

    # Specific velocity thresholds
    print("\n--- Velocity Thresholds ---")
    for threshold in [0.02, 0.04, 0.06, 0.08]:
        high_vel = df[df['velocity'] >= threshold]
        low_vel = df[df['velocity'] < threshold]
        if len(high_vel) > 0 and len(low_vel) > 0:
            print(f"Velocity >= {threshold}: {high_vel['continued'].mean():.1%} continuation ({len(high_vel)} events)")
            print(f"Velocity <  {threshold}: {low_vel['continued'].mean():.1%} continuation ({len(low_vel)} events)")

    results['velocity'] = velocity_stats

    # 2. Depth analysis (how deep below -0.2 it went)
    print("\n" + "="*80)
    print("FACTOR 2: DEPTH BEFORE REVERSAL (MIN MFC VALUE)")
    print("="*80)

    depth_bins = pd.cut(df['min_mfc'], bins=[-1.5, -0.7, -0.5, -0.35, -0.2],
                        labels=['very_deep_<-0.7', 'deep_-0.7_to_-0.5', 'moderate_-0.5_to_-0.35', 'shallow_>-0.35'])
    df['depth_bin'] = depth_bins

    depth_stats = df.groupby('depth_bin').agg({
        'continued': ['mean', 'count'],
        'full_success': 'mean'
    }).round(3)
    print("\nContinuation rate by depth:")
    print(depth_stats)

    results['depth'] = depth_stats

    # 3. Time below -0.2 analysis
    print("\n" + "="*80)
    print("FACTOR 3: TIME SPENT BELOW -0.2 BEFORE CROSSING")
    print("="*80)

    time_bins = pd.cut(df['time_below'], bins=[0, 3, 8, 15, 1000],
                       labels=['quick_1-3bars', 'short_4-8bars', 'medium_9-15bars', 'long_>15bars'])
    df['time_bin'] = time_bins

    time_stats = df.groupby('time_bin').agg({
        'continued': ['mean', 'count'],
        'full_success': 'mean'
    }).round(3)
    print("\nContinuation rate by time below -0.2:")
    print(time_stats)

    results['time_below'] = time_stats

    # 4. Diverging currency analysis
    print("\n" + "="*80)
    print("FACTOR 4: PRESENCE OF DIVERGING CURRENCY (CROSSING +0.2 DOWN)")
    print("="*80)

    df['has_diverging'] = df['num_diverging_down'] > 0

    diverging_stats = df.groupby('has_diverging').agg({
        'continued': ['mean', 'count'],
        'full_success': 'mean'
    }).round(3)
    print("\nContinuation rate with/without diverging currency:")
    print(diverging_stats)

    # Specific diverging analysis
    if df['has_diverging'].any():
        with_div = df[df['has_diverging']]
        without_div = df[~df['has_diverging']]
        print(f"\nWith diverging currency:    {with_div['continued'].mean():.1%} continuation ({len(with_div)} events)")
        print(f"Without diverging currency: {without_div['continued'].mean():.1%} continuation ({len(without_div)} events)")
        print(f"\nWith diverging -> Full success:    {with_div['full_success'].mean():.1%}")
        print(f"Without diverging -> Full success: {without_div['full_success'].mean():.1%}")

    results['diverging'] = diverging_stats

    # 5. Market context (how many currencies strong/weak)
    print("\n" + "="*80)
    print("FACTOR 5: MARKET CONTEXT (OTHER CURRENCIES' POSITIONS)")
    print("="*80)

    print("\nContinuation rate by number of strong currencies (above +0.2):")
    strong_stats = df.groupby('num_strong').agg({
        'continued': ['mean', 'count'],
        'full_success': 'mean'
    }).round(3)
    print(strong_stats)

    print("\nContinuation rate by number of weak currencies (below -0.2):")
    weak_stats = df.groupby('num_weak').agg({
        'continued': ['mean', 'count'],
        'full_success': 'mean'
    }).round(3)
    print(weak_stats)

    results['market_context_strong'] = strong_stats
    results['market_context_weak'] = weak_stats

    # 6. Currency-specific analysis
    print("\n" + "="*80)
    print("FACTOR 6: CURRENCY-SPECIFIC BEHAVIOR")
    print("="*80)

    currency_stats = df.groupby('currency').agg({
        'continued': ['mean', 'count'],
        'full_success': 'mean',
        'velocity': 'mean',
        'min_mfc': 'mean'
    }).round(3)
    print("\nContinuation rate by currency:")
    print(currency_stats)

    results['by_currency'] = currency_stats

    # 7. Combined factors
    print("\n" + "="*80)
    print("FACTOR 7: COMBINED PREDICTIVE FACTORS")
    print("="*80)

    # High velocity + diverging currency
    df['high_velocity'] = df['velocity'] > df['velocity'].median()
    df['ideal_setup'] = df['high_velocity'] & df['has_diverging']

    print("\nIdeal setup (high velocity + diverging currency):")
    ideal_stats = df.groupby('ideal_setup').agg({
        'continued': ['mean', 'count'],
        'full_success': 'mean'
    }).round(3)
    print(ideal_stats)

    # Deep oversold + high velocity
    df['deep_oversold'] = df['min_mfc'] < -0.5
    df['deep_fast'] = df['deep_oversold'] & df['high_velocity']

    print("\nDeep oversold (< -0.5) + high velocity:")
    deep_fast_stats = df.groupby('deep_fast').agg({
        'continued': ['mean', 'count'],
        'full_success': 'mean'
    }).round(3)
    print(deep_fast_stats)

    results['combined'] = {
        'ideal_setup': ideal_stats,
        'deep_fast': deep_fast_stats
    }

    return df, results

def print_summary_statistics(df):
    """Print overall summary statistics."""
    print("\n" + "="*80)
    print("OVERALL SUMMARY STATISTICS")
    print("="*80)

    print(f"\nTotal crossing events analyzed: {len(df)}")
    print(f"\nOutcome distribution:")
    outcome_counts = df['outcome'].value_counts()
    for outcome, count in outcome_counts.items():
        pct = count / len(df) * 100
        print(f"  {outcome}: {count} ({pct:.1f}%)")

    print(f"\n--- Success Rates ---")
    print(f"Any continuation (reached 0 or +0.2): {df['continued'].mean():.1%}")
    print(f"Full continuation (reached +0.2):     {df['full_success'].mean():.1%}")

    # Time analysis
    print(f"\n--- Timing Statistics ---")
    reached_zero = df[df['time_to_zero'].notna()]
    reached_upper = df[df['time_to_upper'].notna()]

    if len(reached_zero) > 0:
        print(f"Average time to reach 0: {reached_zero['time_to_zero'].mean():.1f} bars ({reached_zero['time_to_zero'].mean() * 4:.0f} hours)")
    if len(reached_upper) > 0:
        print(f"Average time to reach +0.2: {reached_upper['time_to_upper'].mean():.1f} bars ({reached_upper['time_to_upper'].mean() * 4:.0f} hours)")

    fell_back = df[df['time_to_fall_back'].notna()]
    if len(fell_back) > 0:
        print(f"Average time before falling back below -0.2: {fell_back['time_to_fall_back'].mean():.1f} bars")

def find_best_predictors(df):
    """Identify the strongest predictive factors."""
    print("\n" + "="*80)
    print("BEST PREDICTIVE FACTORS RANKED")
    print("="*80)

    predictors = []

    # Velocity
    high_vel = df[df['velocity'] > df['velocity'].quantile(0.75)]
    low_vel = df[df['velocity'] <= df['velocity'].quantile(0.25)]
    vel_diff = high_vel['continued'].mean() - low_vel['continued'].mean()
    predictors.append(('High vs Low Velocity', vel_diff, len(high_vel), high_vel['continued'].mean()))

    # Depth
    deep = df[df['min_mfc'] < -0.5]
    shallow = df[df['min_mfc'] >= -0.35]
    if len(deep) > 0 and len(shallow) > 0:
        depth_diff = deep['continued'].mean() - shallow['continued'].mean()
        predictors.append(('Deep vs Shallow Oversold', depth_diff, len(deep), deep['continued'].mean()))

    # Diverging currency
    with_div = df[df['has_diverging']]
    without_div = df[~df['has_diverging']]
    if len(with_div) > 0:
        div_diff = with_div['continued'].mean() - without_div['continued'].mean()
        predictors.append(('With vs Without Diverging Currency', div_diff, len(with_div), with_div['continued'].mean()))

    # Time below
    quick = df[df['time_below'] <= 3]
    long_time = df[df['time_below'] > 15]
    if len(quick) > 0 and len(long_time) > 0:
        time_diff = quick['continued'].mean() - long_time['continued'].mean()
        predictors.append(('Quick vs Long Time Below -0.2', time_diff, len(quick), quick['continued'].mean()))

    # Sort by absolute difference
    predictors.sort(key=lambda x: abs(x[1]), reverse=True)

    print("\nRanked by impact on continuation probability:")
    print("-" * 70)
    for name, diff, n, rate in predictors:
        direction = "+" if diff > 0 else ""
        print(f"{name}:")
        print(f"  Impact: {direction}{diff:.1%} | Sample size: {n} | Success rate: {rate:.1%}")
        print()

    return predictors

def print_actionable_insights(df):
    """Print actionable trading insights."""
    print("\n" + "="*80)
    print("ACTIONABLE TRADING INSIGHTS")
    print("="*80)

    # Best conditions
    print("\n--- BEST CONDITIONS FOR CONTINUATION ---")

    # Condition 1: High velocity + diverging
    cond1 = df[(df['velocity'] > df['velocity'].median()) & (df['has_diverging'])]
    if len(cond1) > 0:
        print(f"\n1. High velocity + Diverging currency present:")
        print(f"   Continuation rate: {cond1['continued'].mean():.1%}")
        print(f"   Full success rate: {cond1['full_success'].mean():.1%}")
        print(f"   Sample size: {len(cond1)} events")

    # Condition 2: Deep oversold + high velocity
    cond2 = df[(df['min_mfc'] < -0.5) & (df['velocity'] > df['velocity'].median())]
    if len(cond2) > 0:
        print(f"\n2. Deep oversold (< -0.5) + High velocity:")
        print(f"   Continuation rate: {cond2['continued'].mean():.1%}")
        print(f"   Full success rate: {cond2['full_success'].mean():.1%}")
        print(f"   Sample size: {len(cond2)} events")

    # Condition 3: Quick bounce (short time below)
    cond3 = df[df['time_below'] <= 3]
    if len(cond3) > 0:
        print(f"\n3. Quick bounce (1-3 bars below -0.2):")
        print(f"   Continuation rate: {cond3['continued'].mean():.1%}")
        print(f"   Full success rate: {cond3['full_success'].mean():.1%}")
        print(f"   Sample size: {len(cond3)} events")

    # Condition 4: Ideal setup (all factors aligned)
    cond4 = df[(df['velocity'] > df['velocity'].quantile(0.7)) &
               (df['has_diverging']) &
               (df['min_mfc'] < -0.4)]
    if len(cond4) > 0:
        print(f"\n4. IDEAL SETUP (velocity top 30% + diverging + depth < -0.4):")
        print(f"   Continuation rate: {cond4['continued'].mean():.1%}")
        print(f"   Full success rate: {cond4['full_success'].mean():.1%}")
        print(f"   Sample size: {len(cond4)} events")

    # Worst conditions
    print("\n--- WORST CONDITIONS (AVOID) ---")

    # Low velocity + no diverging
    worst1 = df[(df['velocity'] < df['velocity'].median()) & (~df['has_diverging'])]
    if len(worst1) > 0:
        print(f"\n1. Low velocity + No diverging currency:")
        print(f"   Continuation rate: {worst1['continued'].mean():.1%}")
        print(f"   Full success rate: {worst1['full_success'].mean():.1%}")
        print(f"   Sample size: {len(worst1)} events")

    # Long time below + shallow depth
    worst2 = df[(df['time_below'] > 15) & (df['min_mfc'] > -0.35)]
    if len(worst2) > 0:
        print(f"\n2. Long time below (>15 bars) + Shallow depth (> -0.35):")
        print(f"   Continuation rate: {worst2['continued'].mean():.1%}")
        print(f"   Full success rate: {worst2['full_success'].mean():.1%}")
        print(f"   Sample size: {len(worst2)} events")

def main():
    print("="*80)
    print("MFC H4 ANALYSIS: CROSSING -0.2 FROM BELOW - CONTINUATION PREDICTORS")
    print("="*80)

    # Load data
    print("\nLoading MFC data for all 8 currencies...")
    df = load_all_currencies()
    print(f"Loaded {len(df)} data points from {df.index.min()} to {df.index.max()}")

    # Find all crossing events
    print("\nFinding all crossing events (from below -0.2 to above -0.2)...")
    all_events = []

    for currency in CURRENCIES:
        events = find_crossing_up_events(df, currency)
        print(f"  {currency}: {len(events)} crossing events found")
        all_events.extend(events)

    print(f"\nTotal crossing events found: {len(all_events)}")

    # Enrich events with diverging currency info and outcomes
    print("\nAnalyzing market context and outcomes for each event...")
    events_with_outcomes = []

    for event in all_events:
        idx = event['idx']
        currency = event['currency']

        # Get diverging currencies info
        div_info = find_diverging_currencies(df, idx, currency)
        event['num_diverging_down'] = len(div_info['diverging_down'])
        event['diverging_currencies'] = div_info['diverging_down']
        event['num_strong'] = len(div_info['strong_currencies'])
        event['num_weak'] = len(div_info['weak_currencies'])
        event['num_in_box'] = len(div_info['in_box'])

        # Track outcome
        outcome = track_outcome(df, currency, idx)
        if outcome:
            event.update(outcome)
            events_with_outcomes.append(event)

    print(f"Events with valid outcomes: {len(events_with_outcomes)}")

    # Analyze predictive factors
    analysis_df, results = analyze_predictive_factors(events_with_outcomes)

    # Print summary
    print_summary_statistics(analysis_df)

    # Find best predictors
    find_best_predictors(analysis_df)

    # Print actionable insights
    print_actionable_insights(analysis_df)

    # Additional: Specific diverging pair analysis
    print("\n" + "="*80)
    print("SPECIFIC ANALYSIS: SIMULTANEOUS DIVERGENCE EVENTS")
    print("="*80)

    div_events = analysis_df[analysis_df['num_diverging_down'] > 0]
    if len(div_events) > 0:
        print(f"\nTotal events with simultaneous diverging currency: {len(div_events)}")
        print(f"Continuation rate: {div_events['continued'].mean():.1%}")
        print(f"Full success rate: {div_events['full_success'].mean():.1%}")

        # Which pairs diverge most often?
        diverging_counts = defaultdict(int)
        for _, row in div_events.iterrows():
            for div_curr in row['diverging_currencies']:
                pair = f"{row['currency']}_up vs {div_curr}_down"
                diverging_counts[pair] += 1

        print("\nMost common diverging pairs:")
        sorted_pairs = sorted(diverging_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        for pair, count in sorted_pairs:
            print(f"  {pair}: {count} occurrences")

    # Acceleration analysis
    print("\n" + "="*80)
    print("ADDITIONAL FACTOR: ACCELERATION AT CROSSING")
    print("="*80)

    # Positive acceleration = momentum increasing
    analysis_df['positive_acceleration'] = analysis_df['acceleration'] > 0
    accel_stats = analysis_df.groupby('positive_acceleration').agg({
        'continued': ['mean', 'count'],
        'full_success': 'mean'
    }).round(3)
    print("\nContinuation rate by acceleration direction:")
    print(accel_stats)

    # Save detailed results
    output_path = Path("/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/lstm_model/scripts/crossing_analysis_results.csv")
    analysis_df.to_csv(output_path, index=False)
    print(f"\nDetailed results saved to: {output_path}")

    return analysis_df, results

if __name__ == "__main__":
    analysis_df, results = main()
