"""
MFC Divergence + Velocity Analysis on H4
=========================================
Analyzes whether price actually moves when MFC shows divergence signals:
- Currency A crosses -0.2 from below (with velocity >= 0.06)
- Currency B crosses +0.2 from above

Measures pip performance over 10, 20, 40 H4 bars.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from itertools import permutations
import warnings
warnings.filterwarnings('ignore')

# Paths
MFC_PATH = Path("/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/data/cleaned")
PRICE_PATH = Path("/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/data")

# Currency pairs and their pip values
CURRENCIES = ['EUR', 'USD', 'GBP', 'JPY', 'AUD', 'NZD', 'CAD', 'CHF']

# Pip multipliers (for JPY pairs it's different)
def get_pip_multiplier(pair):
    if 'JPY' in pair:
        return 100  # 0.01 = 1 pip
    return 10000  # 0.0001 = 1 pip

def load_mfc_data():
    """Load all H4 MFC data for all currencies"""
    mfc_data = {}
    for currency in CURRENCIES:
        filepath = MFC_PATH / f"mfc_currency_{currency}_H4_clean.csv"
        if filepath.exists():
            df = pd.read_csv(filepath)
            df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
            df = df.set_index('datetime')
            df = df[['MFC']].rename(columns={'MFC': currency})
            mfc_data[currency] = df
            print(f"Loaded {currency}: {len(df)} rows, range {df.index.min()} to {df.index.max()}")
    return mfc_data

def combine_mfc_data(mfc_data):
    """Combine all MFC data into a single DataFrame"""
    combined = None
    for currency, df in mfc_data.items():
        if combined is None:
            combined = df.copy()
        else:
            combined = combined.join(df, how='outer')
    combined = combined.sort_index()
    return combined

def load_and_resample_price_data(pair):
    """Load M1 price data and resample to H4"""
    filepath = PRICE_PATH / f"{pair}_GMT+0_US-DST_M1.csv"
    if not filepath.exists():
        return None

    # Read in chunks for large files
    df = pd.read_csv(filepath)
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df = df.set_index('datetime')

    # Resample to H4
    h4 = df.resample('4h').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last'
    }).dropna()

    return h4

def calculate_velocity(mfc_series, lookback=1):
    """Calculate MFC velocity (rate of change)"""
    return mfc_series.diff(lookback)

def find_divergence_events(mfc_combined, velocity_threshold=0.0):
    """
    Find events where:
    - Currency A crosses -0.2 from below (going up)
    - Currency B crosses +0.2 from above (going down)
    - Optionally with velocity filter
    """
    events = []

    # Calculate velocities for all currencies
    velocities = pd.DataFrame(index=mfc_combined.index)
    for col in mfc_combined.columns:
        velocities[col] = calculate_velocity(mfc_combined[col])

    # For each timestamp
    for i in range(1, len(mfc_combined)):
        curr_time = mfc_combined.index[i]
        prev_time = mfc_combined.index[i-1]

        # Find currencies crossing -0.2 from below (going up)
        up_crossers = []
        for curr in mfc_combined.columns:
            prev_val = mfc_combined[curr].iloc[i-1]
            curr_val = mfc_combined[curr].iloc[i]
            vel = velocities[curr].iloc[i]

            # Cross from below -0.2 to above -0.2
            if prev_val <= -0.2 and curr_val > -0.2 and vel >= velocity_threshold:
                up_crossers.append((curr, vel, curr_val))

        # Find currencies crossing +0.2 from above (going down)
        down_crossers = []
        for curr in mfc_combined.columns:
            prev_val = mfc_combined[curr].iloc[i-1]
            curr_val = mfc_combined[curr].iloc[i]
            vel = velocities[curr].iloc[i]

            # Cross from above +0.2 to below +0.2
            if prev_val >= 0.2 and curr_val < 0.2:
                down_crossers.append((curr, vel, curr_val))

        # Create events for each combination
        for up_curr, up_vel, up_mfc in up_crossers:
            for down_curr, down_vel, down_mfc in down_crossers:
                if up_curr != down_curr:
                    events.append({
                        'datetime': curr_time,
                        'idx': i,
                        'long_currency': up_curr,
                        'short_currency': down_curr,
                        'long_velocity': up_vel,
                        'short_velocity': down_vel,
                        'long_mfc': up_mfc,
                        'short_mfc': down_mfc
                    })

    return pd.DataFrame(events)

def get_pair_name(base, quote):
    """Get the correct forex pair name (some pairs are reversed)"""
    # Standard pair ordering
    order = ['EUR', 'GBP', 'AUD', 'NZD', 'USD', 'CAD', 'CHF', 'JPY']

    if base == quote:
        return None

    base_idx = order.index(base) if base in order else 99
    quote_idx = order.index(quote) if quote in order else 99

    if base_idx < quote_idx:
        return f"{base}{quote}", 1  # 1 means go long
    else:
        return f"{quote}{base}", -1  # -1 means go short (reversed pair)

def measure_price_movement(events_df, price_cache, mfc_combined, bars_forward=[10, 20, 40]):
    """Measure price movement after divergence events"""
    results = []

    for idx, event in events_df.iterrows():
        long_curr = event['long_currency']
        short_curr = event['short_currency']
        event_time = event['datetime']
        event_idx = event['idx']

        # Get pair name
        pair_info = get_pair_name(long_curr, short_curr)
        if pair_info is None:
            continue

        pair, direction = pair_info

        # Load price data if not in cache
        if pair not in price_cache:
            price_data = load_and_resample_price_data(pair)
            if price_data is not None:
                price_cache[pair] = price_data
            else:
                continue

        price_df = price_cache[pair]

        # Find price at event time (or closest after)
        # Account for MFC shift - MFC is typically shifted, so we use the timestamp directly
        try:
            # Find the closest price timestamp to the event
            price_times = price_df.index
            valid_times = price_times[price_times >= event_time]
            if len(valid_times) == 0:
                continue
            entry_time = valid_times[0]
            entry_idx = price_df.index.get_loc(entry_time)
            entry_price = price_df['Close'].iloc[entry_idx]
        except:
            continue

        pip_mult = get_pip_multiplier(pair)

        result = {
            'datetime': event_time,
            'pair': pair,
            'direction': 'LONG' if direction == 1 else 'SHORT',
            'long_currency': long_curr,
            'short_currency': short_curr,
            'long_velocity': event['long_velocity'],
            'entry_price': entry_price
        }

        # Measure at different forward bars
        for bars in bars_forward:
            try:
                future_idx = entry_idx + bars
                if future_idx < len(price_df):
                    future_price = price_df['Close'].iloc[future_idx]
                    # Calculate pips based on direction
                    raw_pips = (future_price - entry_price) * pip_mult * direction
                    result[f'pips_{bars}'] = raw_pips
                    result[f'max_high_{bars}'] = (price_df['High'].iloc[entry_idx+1:future_idx+1].max() - entry_price) * pip_mult * direction
                    result[f'max_low_{bars}'] = (entry_price - price_df['Low'].iloc[entry_idx+1:future_idx+1].min()) * pip_mult * direction
                else:
                    result[f'pips_{bars}'] = np.nan
                    result[f'max_high_{bars}'] = np.nan
                    result[f'max_low_{bars}'] = np.nan
            except:
                result[f'pips_{bars}'] = np.nan
                result[f'max_high_{bars}'] = np.nan
                result[f'max_low_{bars}'] = np.nan

        results.append(result)

    return pd.DataFrame(results)

def analyze_results(results_df, label=""):
    """Analyze and print statistics"""
    print(f"\n{'='*60}")
    print(f"RESULTS: {label}")
    print(f"{'='*60}")

    if len(results_df) == 0:
        print("No events found!")
        return

    print(f"\nTotal Events: {len(results_df)}")

    for bars in [10, 20, 40]:
        col = f'pips_{bars}'
        valid = results_df[col].dropna()
        if len(valid) == 0:
            continue

        wins = (valid > 0).sum()
        win_rate = wins / len(valid) * 100
        avg_pips = valid.mean()
        median_pips = valid.median()
        std_pips = valid.std()
        max_win = valid.max()
        max_loss = valid.min()

        print(f"\n--- {bars} H4 Bars Forward ---")
        print(f"  Win Rate: {win_rate:.1f}% ({wins}/{len(valid)})")
        print(f"  Avg Pips: {avg_pips:.1f}")
        print(f"  Median Pips: {median_pips:.1f}")
        print(f"  Std Dev: {std_pips:.1f}")
        print(f"  Best Trade: +{max_win:.1f} pips")
        print(f"  Worst Trade: {max_loss:.1f} pips")
        print(f"  Expectancy: {avg_pips:.2f} pips/trade")

        # Max favorable/adverse excursion
        max_fav_col = f'max_high_{bars}'
        max_adv_col = f'max_low_{bars}'
        if max_fav_col in results_df.columns:
            mfe = results_df[max_fav_col].dropna().mean()
            mae = results_df[max_adv_col].dropna().mean()
            print(f"  Avg Max Favorable: +{mfe:.1f} pips")
            print(f"  Avg Max Adverse: {mae:.1f} pips")

def analyze_by_pair(results_df, bars=20):
    """Analyze results by currency pair"""
    print(f"\n{'='*60}")
    print(f"PERFORMANCE BY PAIR ({bars} bar horizon)")
    print(f"{'='*60}")

    col = f'pips_{bars}'

    pair_stats = results_df.groupby('pair').agg({
        col: ['count', 'mean', 'median', 'std', lambda x: (x > 0).sum() / len(x) * 100]
    }).round(1)
    pair_stats.columns = ['Count', 'Avg Pips', 'Median', 'Std', 'Win%']
    pair_stats = pair_stats.sort_values('Avg Pips', ascending=False)

    print(pair_stats.to_string())

    return pair_stats

def analyze_by_long_currency(results_df, bars=20):
    """Analyze which long currency performs best"""
    print(f"\n{'='*60}")
    print(f"PERFORMANCE BY LONG CURRENCY ({bars} bar horizon)")
    print(f"{'='*60}")

    col = f'pips_{bars}'

    curr_stats = results_df.groupby('long_currency').agg({
        col: ['count', 'mean', 'median', 'std', lambda x: (x > 0).sum() / len(x) * 100]
    }).round(1)
    curr_stats.columns = ['Count', 'Avg Pips', 'Median', 'Std', 'Win%']
    curr_stats = curr_stats.sort_values('Avg Pips', ascending=False)

    print(curr_stats.to_string())

def main():
    print("="*60)
    print("MFC DIVERGENCE + VELOCITY ANALYSIS ON H4")
    print("="*60)

    # Load MFC data
    print("\nLoading MFC data...")
    mfc_data = load_mfc_data()
    mfc_combined = combine_mfc_data(mfc_data)
    print(f"\nCombined MFC data: {len(mfc_combined)} rows")

    # Price data cache
    price_cache = {}

    # Analysis WITHOUT velocity filter
    print("\n" + "="*60)
    print("FINDING DIVERGENCE EVENTS (NO VELOCITY FILTER)")
    print("="*60)

    events_no_filter = find_divergence_events(mfc_combined, velocity_threshold=0.0)
    print(f"Found {len(events_no_filter)} divergence events")

    if len(events_no_filter) > 0:
        print("\nMeasuring price movements...")
        results_no_filter = measure_price_movement(events_no_filter, price_cache, mfc_combined)
        print(f"Got price data for {len(results_no_filter)} events")

        analyze_results(results_no_filter, "WITHOUT Velocity Filter")
        analyze_by_pair(results_no_filter, 20)
        analyze_by_long_currency(results_no_filter, 20)

    # Analysis WITH velocity filter (>= 0.06)
    print("\n" + "="*60)
    print("FINDING DIVERGENCE EVENTS (VELOCITY >= 0.06)")
    print("="*60)

    events_with_filter = find_divergence_events(mfc_combined, velocity_threshold=0.06)
    print(f"Found {len(events_with_filter)} divergence events with velocity >= 0.06")

    if len(events_with_filter) > 0:
        print("\nMeasuring price movements...")
        results_with_filter = measure_price_movement(events_with_filter, price_cache, mfc_combined)
        print(f"Got price data for {len(results_with_filter)} events")

        analyze_results(results_with_filter, "WITH Velocity Filter >= 0.06")
        analyze_by_pair(results_with_filter, 20)
        analyze_by_long_currency(results_with_filter, 20)

    # Compare filters
    print("\n" + "="*60)
    print("FILTER COMPARISON SUMMARY")
    print("="*60)

    if len(events_no_filter) > 0 and len(events_with_filter) > 0:
        for bars in [10, 20, 40]:
            col = f'pips_{bars}'

            no_filt = results_no_filter[col].dropna()
            with_filt = results_with_filter[col].dropna()

            print(f"\n{bars} Bar Horizon:")
            print(f"  Without filter: {len(no_filt)} trades, {no_filt.mean():.1f} avg pips, {(no_filt>0).mean()*100:.1f}% win rate")
            if len(with_filt) > 0:
                print(f"  With velocity>=0.06: {len(with_filt)} trades, {with_filt.mean():.1f} avg pips, {(with_filt>0).mean()*100:.1f}% win rate")
            else:
                print(f"  With velocity>=0.06: No trades")

    # Save detailed results
    if len(results_no_filter) > 0:
        results_no_filter.to_csv('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/lstm_model/scripts/divergence_results_no_filter.csv', index=False)
    if len(events_with_filter) > 0 and len(results_with_filter) > 0:
        results_with_filter.to_csv('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/lstm_model/scripts/divergence_results_with_filter.csv', index=False)

    print("\n\nResults saved to CSV files for further analysis.")

if __name__ == "__main__":
    main()
