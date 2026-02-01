"""
MFC Divergence + Velocity Pattern: Multi-Timeframe Analysis
============================================================
Tests if the MFC divergence + velocity pattern works on lower timeframes.

Pattern Logic:
- Currency A crosses -0.2 from below with velocity >= threshold
- Currency B crosses +0.2 from above simultaneously
- Go LONG the pair where A is base currency (A/B)

Tests:
- H4: Baseline (for comparison)
- H1: Lower timeframe
- M5: Scalping timeframe (if available)

Velocity thresholds: 0.01, 0.02, 0.03, 0.04, 0.05, 0.06
Horizons adapted per timeframe.
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
OUTPUT_PATH = Path("/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/lstm_model/scripts")

# Currencies
CURRENCIES = ['EUR', 'USD', 'GBP', 'JPY', 'AUD', 'NZD', 'CAD', 'CHF']

# Standard pair ordering for Forex
PAIR_ORDER = ['EUR', 'GBP', 'AUD', 'NZD', 'USD', 'CAD', 'CHF', 'JPY']

def get_pip_multiplier(pair):
    """Get pip multiplier based on pair"""
    if 'JPY' in pair:
        return 100  # 0.01 = 1 pip
    return 10000  # 0.0001 = 1 pip

def get_pair_name(base, quote):
    """Get the correct forex pair name and direction"""
    if base == quote:
        return None, None

    base_idx = PAIR_ORDER.index(base) if base in PAIR_ORDER else 99
    quote_idx = PAIR_ORDER.index(quote) if quote in PAIR_ORDER else 99

    if base_idx < quote_idx:
        return f"{base}{quote}", 1  # Go LONG
    else:
        return f"{quote}{base}", -1  # Go SHORT (pair is reversed)

def load_mfc_data(timeframe):
    """Load all MFC data for a given timeframe"""
    mfc_data = {}
    for currency in CURRENCIES:
        filepath = MFC_PATH / f"mfc_currency_{currency}_{timeframe}_clean.csv"
        if filepath.exists():
            df = pd.read_csv(filepath)
            df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
            df = df.set_index('datetime')
            df = df[['MFC']].rename(columns={'MFC': currency})
            mfc_data[currency] = df
    return mfc_data

def combine_mfc_data(mfc_data):
    """Combine all MFC data into a single DataFrame"""
    combined = None
    for currency, df in mfc_data.items():
        if combined is None:
            combined = df.copy()
        else:
            combined = combined.join(df, how='outer')
    if combined is not None:
        combined = combined.sort_index().dropna()
    return combined

def load_and_resample_price_data(pair, target_tf):
    """Load M1 price data and resample to target timeframe"""
    filepath = PRICE_PATH / f"{pair}_GMT+0_US-DST_M1.csv"
    if not filepath.exists():
        return None

    # Load data - file has headers
    df = pd.read_csv(filepath)
    df.columns = ['date', 'time', 'open', 'high', 'low', 'close', 'volume']
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='%Y.%m.%d %H:%M:%S')
    df = df.set_index('datetime')

    # Resample based on target timeframe
    if target_tf == 'H4':
        resample_rule = '4h'
    elif target_tf == 'H1':
        resample_rule = '1h'
    elif target_tf == 'M5':
        resample_rule = '5min'
    else:
        return None

    resampled = df.resample(resample_rule).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).dropna()

    return resampled

def calculate_velocity(mfc_series, lookback=1):
    """Calculate MFC velocity (rate of change)"""
    return mfc_series.diff(lookback)

def find_divergence_events(mfc_combined, velocity_threshold=0.0):
    """
    Find divergence events:
    - Currency A crosses -0.2 from below (going up) with velocity >= threshold
    - Currency B crosses +0.2 from above (going down)
    """
    events = []

    # Calculate velocities
    velocities = pd.DataFrame(index=mfc_combined.index)
    for col in mfc_combined.columns:
        velocities[col] = calculate_velocity(mfc_combined[col])

    # Iterate through data
    for i in range(1, len(mfc_combined)):
        curr_time = mfc_combined.index[i]

        # Find currencies crossing -0.2 from below (bullish)
        up_crossers = []
        for curr in mfc_combined.columns:
            prev_val = mfc_combined[curr].iloc[i-1]
            curr_val = mfc_combined[curr].iloc[i]
            vel = velocities[curr].iloc[i]

            if prev_val <= -0.2 and curr_val > -0.2 and vel >= velocity_threshold:
                up_crossers.append((curr, vel, curr_val))

        # Find currencies crossing +0.2 from above (bearish)
        down_crossers = []
        for curr in mfc_combined.columns:
            prev_val = mfc_combined[curr].iloc[i-1]
            curr_val = mfc_combined[curr].iloc[i]
            vel = velocities[curr].iloc[i]

            if prev_val >= 0.2 and curr_val < 0.2:
                down_crossers.append((curr, vel, curr_val))

        # Create events
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

def measure_price_movement(events_df, price_cache, timeframe, bars_forward):
    """Measure price movement after divergence events"""
    results = []

    for idx, event in events_df.iterrows():
        long_curr = event['long_currency']
        short_curr = event['short_currency']
        event_time = event['datetime']

        # Get pair name
        pair, direction = get_pair_name(long_curr, short_curr)
        if pair is None:
            continue

        # Load price data if not cached
        if pair not in price_cache:
            price_data = load_and_resample_price_data(pair, timeframe)
            if price_data is not None:
                price_cache[pair] = price_data
            else:
                continue

        price_df = price_cache[pair]

        # Find entry price
        try:
            price_times = price_df.index
            valid_times = price_times[price_times >= event_time]
            if len(valid_times) == 0:
                continue
            entry_time = valid_times[0]
            entry_idx = price_df.index.get_loc(entry_time)
            entry_price = price_df['close'].iloc[entry_idx]
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

        # Measure at different horizons
        for bars in bars_forward:
            try:
                future_idx = entry_idx + bars
                if future_idx < len(price_df):
                    future_price = price_df['close'].iloc[future_idx]
                    raw_pips = (future_price - entry_price) * pip_mult * direction

                    # Max favorable/adverse excursion
                    max_high = price_df['high'].iloc[entry_idx+1:future_idx+1].max()
                    min_low = price_df['low'].iloc[entry_idx+1:future_idx+1].min()
                    mfe = (max_high - entry_price) * pip_mult * direction
                    mae = (entry_price - min_low) * pip_mult * direction

                    result[f'pips_{bars}'] = raw_pips
                    result[f'mfe_{bars}'] = mfe
                    result[f'mae_{bars}'] = mae
                else:
                    result[f'pips_{bars}'] = np.nan
                    result[f'mfe_{bars}'] = np.nan
                    result[f'mae_{bars}'] = np.nan
            except:
                result[f'pips_{bars}'] = np.nan
                result[f'mfe_{bars}'] = np.nan
                result[f'mae_{bars}'] = np.nan

        results.append(result)

    return pd.DataFrame(results)

def calculate_profit_factor(pips_series):
    """Calculate profit factor"""
    wins = pips_series[pips_series > 0].sum()
    losses = abs(pips_series[pips_series < 0].sum())
    if losses == 0:
        return float('inf') if wins > 0 else 0
    return wins / losses

def analyze_timeframe(timeframe, velocity_thresholds, bars_forward, max_events=None):
    """Analyze a single timeframe with multiple velocity thresholds"""
    print(f"\n{'='*70}")
    print(f"ANALYZING {timeframe} TIMEFRAME")
    print(f"{'='*70}")

    # Load MFC data
    print(f"\nLoading {timeframe} MFC data...")
    mfc_data = load_mfc_data(timeframe)
    if len(mfc_data) == 0:
        print(f"  ERROR: No {timeframe} MFC data found!")
        return None

    mfc_combined = combine_mfc_data(mfc_data)
    print(f"  Loaded {len(mfc_combined)} rows, {len(mfc_data)} currencies")
    print(f"  Date range: {mfc_combined.index.min()} to {mfc_combined.index.max()}")

    # Price cache
    price_cache = {}

    # Results for all velocity thresholds
    all_results = []

    for vel_thresh in velocity_thresholds:
        print(f"\n--- Velocity Threshold: {vel_thresh} ---")

        # Find events
        events = find_divergence_events(mfc_combined, velocity_threshold=vel_thresh)
        print(f"  Found {len(events)} divergence events")

        if len(events) == 0:
            continue

        # Limit events for faster processing if needed
        if max_events and len(events) > max_events:
            events = events.sample(n=max_events, random_state=42)
            print(f"  Sampled to {len(events)} events")

        # Measure price movement
        results = measure_price_movement(events, price_cache, timeframe, bars_forward)
        print(f"  Got price data for {len(results)} events")

        if len(results) == 0:
            continue

        # Calculate statistics for each horizon
        for bars in bars_forward:
            col = f'pips_{bars}'
            valid = results[col].dropna()

            if len(valid) < 10:
                continue

            wins = (valid > 0).sum()
            win_rate = wins / len(valid) * 100
            avg_pips = valid.mean()
            total_pips = valid.sum()
            pf = calculate_profit_factor(valid)

            # Calculate average win/loss
            avg_win = valid[valid > 0].mean() if (valid > 0).any() else 0
            avg_loss = abs(valid[valid < 0].mean()) if (valid < 0).any() else 0

            all_results.append({
                'timeframe': timeframe,
                'velocity_threshold': vel_thresh,
                'horizon_bars': bars,
                'n_trades': len(valid),
                'win_rate': win_rate,
                'avg_pips': avg_pips,
                'total_pips': total_pips,
                'profit_factor': pf,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'expectancy': avg_pips
            })

            print(f"    {bars} bars: n={len(valid)}, WR={win_rate:.1f}%, Avg={avg_pips:.1f} pips, PF={pf:.2f}")

    return pd.DataFrame(all_results) if all_results else None

def main():
    print("="*70)
    print("MFC DIVERGENCE + VELOCITY PATTERN: MULTI-TIMEFRAME ANALYSIS")
    print("="*70)
    print("\nPattern: Currency A crosses -0.2 from below (with velocity)")
    print("         Currency B crosses +0.2 from above")
    print("         Go LONG pair where A is base currency")

    # Configuration
    velocity_thresholds = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]

    # Timeframe-specific horizons (bars)
    # H4: 10, 20, 40 bars = 40h, 80h, 160h
    # H1: 20, 40, 80 bars = 20h, 40h, 80h (similar time periods)
    # M5: 48, 96, 192 bars = 4h, 8h, 16h

    timeframe_config = {
        'H4': {'bars': [10, 20, 40], 'max_events': 5000},
        'H1': {'bars': [20, 40, 80], 'max_events': 10000},
        'M5': {'bars': [48, 96, 192], 'max_events': 20000}
    }

    all_tf_results = []

    # Analyze each timeframe
    for tf, config in timeframe_config.items():
        results = analyze_timeframe(
            timeframe=tf,
            velocity_thresholds=velocity_thresholds,
            bars_forward=config['bars'],
            max_events=config['max_events']
        )

        if results is not None and len(results) > 0:
            all_tf_results.append(results)

    if not all_tf_results:
        print("\nNo results found for any timeframe!")
        return

    # Combine all results
    combined_results = pd.concat(all_tf_results, ignore_index=True)

    # =========================================================================
    # COMPARISON TABLE
    # =========================================================================
    print("\n" + "="*70)
    print("TIMEFRAME COMPARISON TABLE")
    print("="*70)

    # For comparison, map horizons to approximate time periods
    # H4 20 bars ~ 80h, H1 80 bars ~ 80h, M5 192 bars ~ 16h (different)
    # Let's use the middle horizon for each

    print("\n--- Best Velocity Threshold by Timeframe (Middle Horizon) ---")
    print(f"{'TF':<6} {'VelThr':>8} {'Bars':>6} {'Trades':>8} {'WinRate':>10} {'AvgPips':>10} {'PF':>8}")
    print("-"*66)

    for tf in ['H4', 'H1', 'M5']:
        tf_data = combined_results[combined_results['timeframe'] == tf]
        if len(tf_data) == 0:
            print(f"{tf:<6} {'No data':<50}")
            continue

        # Use middle horizon
        middle_horizon = tf_data['horizon_bars'].median()
        tf_mid = tf_data[tf_data['horizon_bars'] == middle_horizon]

        if len(tf_mid) == 0:
            # Fallback to any horizon
            tf_mid = tf_data

        # Find best by profit factor (with min trades)
        qualified = tf_mid[tf_mid['n_trades'] >= 30]
        if len(qualified) > 0:
            best = qualified.loc[qualified['profit_factor'].idxmax()]
            print(f"{best['timeframe']:<6} {best['velocity_threshold']:>8.2f} {int(best['horizon_bars']):>6} "
                  f"{int(best['n_trades']):>8} {best['win_rate']:>9.1f}% {best['avg_pips']:>+9.1f} "
                  f"{best['profit_factor']:>7.2f}")

    # =========================================================================
    # DETAILED COMPARISON BY VELOCITY THRESHOLD
    # =========================================================================
    print("\n" + "="*70)
    print("DETAILED COMPARISON BY VELOCITY THRESHOLD")
    print("="*70)

    for vel in velocity_thresholds:
        print(f"\n--- Velocity >= {vel} ---")
        vel_data = combined_results[combined_results['velocity_threshold'] == vel]

        print(f"{'TF':<6} {'Bars':>6} {'Trades':>8} {'WinRate':>10} {'AvgPips':>10} {'PF':>8} {'TotalPips':>12}")
        print("-"*70)

        for tf in ['H4', 'H1', 'M5']:
            tf_vel = vel_data[vel_data['timeframe'] == tf]
            if len(tf_vel) == 0:
                continue

            for _, row in tf_vel.iterrows():
                print(f"{row['timeframe']:<6} {int(row['horizon_bars']):>6} {int(row['n_trades']):>8} "
                      f"{row['win_rate']:>9.1f}% {row['avg_pips']:>+9.1f} {row['profit_factor']:>7.2f} "
                      f"{row['total_pips']:>+11.0f}")

    # =========================================================================
    # SUMMARY: BEST CONFIGURATION PER TIMEFRAME
    # =========================================================================
    print("\n" + "="*70)
    print("BEST CONFIGURATION PER TIMEFRAME")
    print("="*70)

    summary_rows = []

    for tf in ['H4', 'H1', 'M5']:
        tf_data = combined_results[combined_results['timeframe'] == tf]
        if len(tf_data) == 0:
            continue

        # Filter for minimum trades
        qualified = tf_data[tf_data['n_trades'] >= 50]
        if len(qualified) == 0:
            qualified = tf_data[tf_data['n_trades'] >= 20]

        if len(qualified) == 0:
            continue

        # Best by profit factor
        best_pf = qualified.loc[qualified['profit_factor'].idxmax()]

        # Best by win rate
        best_wr = qualified.loc[qualified['win_rate'].idxmax()]

        # Best by avg pips
        best_pips = qualified.loc[qualified['avg_pips'].idxmax()]

        print(f"\n{tf}:")
        print(f"  Best Profit Factor: vel={best_pf['velocity_threshold']:.2f}, bars={int(best_pf['horizon_bars'])}, "
              f"PF={best_pf['profit_factor']:.2f}, WR={best_pf['win_rate']:.1f}%, n={int(best_pf['n_trades'])}")
        print(f"  Best Win Rate:      vel={best_wr['velocity_threshold']:.2f}, bars={int(best_wr['horizon_bars'])}, "
              f"WR={best_wr['win_rate']:.1f}%, PF={best_wr['profit_factor']:.2f}, n={int(best_wr['n_trades'])}")
        print(f"  Best Avg Pips:      vel={best_pips['velocity_threshold']:.2f}, bars={int(best_pips['horizon_bars'])}, "
              f"Avg={best_pips['avg_pips']:+.1f}, WR={best_pips['win_rate']:.1f}%, n={int(best_pips['n_trades'])}")

        summary_rows.append({
            'timeframe': tf,
            'best_vel_for_pf': best_pf['velocity_threshold'],
            'best_pf': best_pf['profit_factor'],
            'best_wr': best_wr['win_rate'],
            'best_avg_pips': best_pips['avg_pips']
        })

    # =========================================================================
    # IS LOWER TIMEFRAME TRADEABLE?
    # =========================================================================
    print("\n" + "="*70)
    print("CONCLUSION: IS THE PATTERN SCALABLE TO LOWER TIMEFRAMES?")
    print("="*70)

    # Compare H4 vs H1 vs M5 at best settings
    for tf in ['H4', 'H1', 'M5']:
        tf_data = combined_results[combined_results['timeframe'] == tf]
        if len(tf_data) == 0:
            print(f"\n{tf}: No data available")
            continue

        qualified = tf_data[tf_data['n_trades'] >= 30]
        if len(qualified) == 0:
            print(f"\n{tf}: Insufficient trades (< 30)")
            continue

        best = qualified.loc[qualified['profit_factor'].idxmax()]

        # Determine if tradeable
        is_profitable = best['avg_pips'] > 0
        has_edge = best['profit_factor'] > 1.0
        decent_wr = best['win_rate'] >= 45

        status = "TRADEABLE" if (is_profitable and has_edge and decent_wr) else "MARGINAL" if is_profitable else "NOT RECOMMENDED"

        print(f"\n{tf}: {status}")
        print(f"  Best Config: velocity >= {best['velocity_threshold']:.2f}, {int(best['horizon_bars'])} bars")
        print(f"  Stats: WR={best['win_rate']:.1f}%, AvgPips={best['avg_pips']:+.1f}, PF={best['profit_factor']:.2f}, n={int(best['n_trades'])}")

    # Save results
    output_file = OUTPUT_PATH / 'divergence_multi_tf_results.csv'
    combined_results.to_csv(output_file, index=False)
    print(f"\n\nResults saved to: {output_file}")

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()
