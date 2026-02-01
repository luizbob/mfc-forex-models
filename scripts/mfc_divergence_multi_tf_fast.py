"""
MFC Divergence + Velocity Pattern: Multi-Timeframe Analysis (FAST VERSION)
============================================================================
Uses pre-existing resampled data or efficient sampling for faster processing.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
import sys
warnings.filterwarnings('ignore')

# Force unbuffered output
def log(msg=""):
    print(msg, flush=True)

# Paths
MFC_PATH = Path("/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/data/cleaned")
PRICE_PATH = Path("/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/data")
OUTPUT_PATH = Path("/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/lstm_model/scripts")

# Currencies
CURRENCIES = ['EUR', 'USD', 'GBP', 'JPY', 'AUD', 'NZD', 'CAD', 'CHF']

# Standard pair ordering
PAIR_ORDER = ['EUR', 'GBP', 'AUD', 'NZD', 'USD', 'CAD', 'CHF', 'JPY']

def get_pip_multiplier(pair):
    if 'JPY' in pair:
        return 100
    return 10000

def get_pair_name(base, quote):
    if base == quote:
        return None, None
    base_idx = PAIR_ORDER.index(base) if base in PAIR_ORDER else 99
    quote_idx = PAIR_ORDER.index(quote) if quote in PAIR_ORDER else 99
    if base_idx < quote_idx:
        return f"{base}{quote}", 1
    else:
        return f"{quote}{base}", -1

def load_mfc_data(timeframe):
    """Load MFC data for all currencies"""
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
    combined = None
    for currency, df in mfc_data.items():
        if combined is None:
            combined = df.copy()
        else:
            combined = combined.join(df, how='outer')
    if combined is not None:
        combined = combined.sort_index().dropna()
    return combined

def load_price_data_fast(pair, timeframe, mfc_dates):
    """Load price data efficiently by checking for pre-resampled files first"""

    # Try indicator_data files first (H1 and H4 available)
    indicator_path = PRICE_PATH / f"indicator_data_{pair}m_{timeframe}.csv"

    if indicator_path.exists():
        log(f"    Loading indicator data {timeframe} for {pair}...")
        df = pd.read_csv(indicator_path)
        # Handle different formats
        if 'DateTime' in df.columns:
            df['datetime'] = pd.to_datetime(df['DateTime'])
        elif 'Date' in df.columns and 'Time' in df.columns:
            df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
        else:
            df['datetime'] = pd.to_datetime(df.iloc[:, 0])
        df = df.set_index('datetime')
        df.columns = [c.lower() for c in df.columns]
        return df

    # Try standard resampled files
    if timeframe == 'H4':
        resampled_path = PRICE_PATH / f"{pair}_GMT+0_US-DST_H4.csv"
    elif timeframe == 'H1':
        resampled_path = PRICE_PATH / f"{pair}_GMT+0_US-DST_H1.csv"
    else:
        resampled_path = PRICE_PATH / f"{pair}_GMT+0_US-DST_{timeframe}.csv"

    if resampled_path.exists():
        log(f"    Loading pre-resampled {timeframe} for {pair}...")
        df = pd.read_csv(resampled_path)
        if 'Date' in df.columns and 'Time' in df.columns:
            df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
        elif 'date' in df.columns and 'time' in df.columns:
            df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
        else:
            df['datetime'] = pd.to_datetime(df.iloc[:, 0] + ' ' + df.iloc[:, 1])

        df = df.set_index('datetime')
        df.columns = [c.lower() for c in df.columns]
        return df

    # Fall back to M1 resampling for limited date range
    log(f"    Resampling M1 to {timeframe} for {pair} (limited range)...")
    m1_path = PRICE_PATH / f"{pair}_GMT+0_US-DST_M1.csv"
    if not m1_path.exists():
        return None

    # Only load rows we need - find date range from MFC
    min_date = mfc_dates.min()
    max_date = mfc_dates.max()

    # Read M1 data in chunks to find relevant dates
    chunks = []
    for chunk in pd.read_csv(m1_path, chunksize=500000):
        chunk.columns = ['date', 'time', 'open', 'high', 'low', 'close', 'volume']
        chunk['datetime'] = pd.to_datetime(chunk['date'] + ' ' + chunk['time'])
        chunk = chunk[(chunk['datetime'] >= min_date) & (chunk['datetime'] <= max_date)]
        if len(chunk) > 0:
            chunks.append(chunk)

    if not chunks:
        return None

    df = pd.concat(chunks, ignore_index=True)
    df = df.set_index('datetime')

    # Resample
    if timeframe == 'H4':
        resample_rule = '4h'
    elif timeframe == 'H1':
        resample_rule = '1h'
    elif timeframe == 'M5':
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

def find_divergence_events(mfc_combined, velocity_threshold=0.0):
    """Find divergence events efficiently using vectorized operations"""
    events = []

    # Calculate velocities
    velocities = mfc_combined.diff(1)

    # Previous and current values
    prev_mfc = mfc_combined.shift(1)
    curr_mfc = mfc_combined

    # Iterate (can't fully vectorize due to combinations)
    for i in range(1, len(mfc_combined)):
        curr_time = mfc_combined.index[i]

        # Find up crossers
        up_crossers = []
        for curr in mfc_combined.columns:
            prev_val = prev_mfc[curr].iloc[i]
            curr_val = curr_mfc[curr].iloc[i]
            vel = velocities[curr].iloc[i]

            if prev_val <= -0.2 and curr_val > -0.2 and vel >= velocity_threshold:
                up_crossers.append((curr, vel, curr_val))

        # Find down crossers
        down_crossers = []
        for curr in mfc_combined.columns:
            prev_val = prev_mfc[curr].iloc[i]
            curr_val = curr_mfc[curr].iloc[i]
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

def measure_price_movement(events_df, price_cache, timeframe, bars_forward, mfc_index):
    """Measure price movement for events"""
    results = []

    for idx, event in events_df.iterrows():
        long_curr = event['long_currency']
        short_curr = event['short_currency']
        event_time = event['datetime']

        pair, direction = get_pair_name(long_curr, short_curr)
        if pair is None:
            continue

        # Get price data
        if pair not in price_cache:
            price_data = load_price_data_fast(pair, timeframe, mfc_index)
            if price_data is not None:
                price_cache[pair] = price_data
            else:
                continue

        price_df = price_cache[pair]

        # Find entry
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

        for bars in bars_forward:
            try:
                future_idx = entry_idx + bars
                if future_idx < len(price_df):
                    future_price = price_df['close'].iloc[future_idx]
                    raw_pips = (future_price - entry_price) * pip_mult * direction
                    result[f'pips_{bars}'] = raw_pips
                else:
                    result[f'pips_{bars}'] = np.nan
            except:
                result[f'pips_{bars}'] = np.nan

        results.append(result)

    return pd.DataFrame(results)

def calculate_profit_factor(pips_series):
    wins = pips_series[pips_series > 0].sum()
    losses = abs(pips_series[pips_series < 0].sum())
    if losses == 0:
        return float('inf') if wins > 0 else 0
    return wins / losses

def analyze_timeframe(timeframe, velocity_thresholds, bars_forward, max_events=None):
    """Analyze a single timeframe"""
    log(f"\n{'='*70}")
    log(f"ANALYZING {timeframe} TIMEFRAME")
    log(f"{'='*70}")

    # Load MFC
    log(f"\nLoading {timeframe} MFC data...")
    mfc_data = load_mfc_data(timeframe)
    if len(mfc_data) == 0:
        log(f"  ERROR: No {timeframe} MFC data!")
        return None

    mfc_combined = combine_mfc_data(mfc_data)
    log(f"  Loaded {len(mfc_combined)} rows, {len(mfc_data)} currencies")
    log(f"  Range: {mfc_combined.index.min()} to {mfc_combined.index.max()}")

    price_cache = {}
    all_results = []

    for vel_thresh in velocity_thresholds:
        log(f"\n--- Velocity >= {vel_thresh} ---")

        events = find_divergence_events(mfc_combined, velocity_threshold=vel_thresh)
        log(f"  Found {len(events)} divergence events")

        if len(events) == 0:
            continue

        if max_events and len(events) > max_events:
            events = events.sample(n=max_events, random_state=42)
            log(f"  Sampled to {len(events)} events")

        results = measure_price_movement(events, price_cache, timeframe, bars_forward, mfc_combined.index)
        log(f"  Got price data for {len(results)} events")

        if len(results) == 0:
            continue

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
                'avg_loss': avg_loss
            })

            log(f"    {bars} bars: n={len(valid)}, WR={win_rate:.1f}%, Avg={avg_pips:.1f} pips, PF={pf:.2f}")

    return pd.DataFrame(all_results) if all_results else None

def main():
    log("="*70)
    log("MFC DIVERGENCE + VELOCITY: MULTI-TIMEFRAME ANALYSIS (FAST)")
    log("="*70)
    log("\nPattern: Currency A crosses -0.2 from below (with velocity)")
    log("         Currency B crosses +0.2 from above")
    log("         Go LONG pair where A is base currency")

    velocity_thresholds = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]

    # Horizons per timeframe (equivalent time periods where possible)
    # H4: 10, 20, 40 bars = 40h, 80h, 160h
    # H1: 40, 80, 160 bars = 40h, 80h, 160h (same as H4)
    # M5: 480, 960, 1920 bars = 40h, 80h, 160h (same time periods)

    timeframe_config = {
        'H4': {'bars': [10, 20, 40], 'max_events': 2000},
        'H1': {'bars': [40, 80, 160], 'max_events': 3000},
        'M5': {'bars': [480, 960, 1920], 'max_events': 5000}
    }

    all_tf_results = []

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
        log("\nNo results found!")
        return

    combined_results = pd.concat(all_tf_results, ignore_index=True)

    # =========================================================================
    # COMPARISON TABLE
    # =========================================================================
    log("\n" + "="*70)
    log("TIMEFRAME COMPARISON TABLE")
    log("="*70)

    # Add time equivalent column for comparison
    time_equiv_map = {
        ('H4', 10): '40h', ('H4', 20): '80h', ('H4', 40): '160h',
        ('H1', 40): '40h', ('H1', 80): '80h', ('H1', 160): '160h',
        ('M5', 480): '40h', ('M5', 960): '80h', ('M5', 1920): '160h'
    }

    combined_results['time_equiv'] = combined_results.apply(
        lambda r: time_equiv_map.get((r['timeframe'], int(r['horizon_bars'])), 'N/A'), axis=1
    )

    log("\n--- Comparison at 80h Horizon (~3.3 days) ---")
    log(f"{'TF':<6} {'VelThr':>8} {'Bars':>8} {'Trades':>8} {'WinRate':>10} {'AvgPips':>10} {'PF':>8}")
    log("-"*68)

    for tf in ['H4', 'H1', 'M5']:
        tf_data = combined_results[(combined_results['timeframe'] == tf) &
                                    (combined_results['time_equiv'] == '80h')]
        if len(tf_data) == 0:
            log(f"{tf:<6} {'No data':<60}")
            continue

        qualified = tf_data[tf_data['n_trades'] >= 20]
        if len(qualified) == 0:
            best = tf_data.iloc[0]
        else:
            best = qualified.loc[qualified['profit_factor'].idxmax()]

        log(f"{best['timeframe']:<6} {best['velocity_threshold']:>8.2f} {int(best['horizon_bars']):>8} "
              f"{int(best['n_trades']):>8} {best['win_rate']:>9.1f}% {best['avg_pips']:>+9.1f} "
              f"{best['profit_factor']:>7.2f}")

    # =========================================================================
    # DETAILED BY VELOCITY
    # =========================================================================
    log("\n" + "="*70)
    log("DETAILED RESULTS BY VELOCITY THRESHOLD")
    log("="*70)

    for vel in velocity_thresholds:
        log(f"\n--- Velocity >= {vel} ---")
        vel_data = combined_results[combined_results['velocity_threshold'] == vel]

        log(f"{'TF':<6} {'Bars':>8} {'~Time':>8} {'Trades':>8} {'WinRate':>10} {'AvgPips':>10} {'PF':>8} {'TotalPips':>12}")
        log("-"*82)

        for tf in ['H4', 'H1', 'M5']:
            tf_vel = vel_data[vel_data['timeframe'] == tf]
            for _, row in tf_vel.iterrows():
                log(f"{row['timeframe']:<6} {int(row['horizon_bars']):>8} {row['time_equiv']:>8} "
                      f"{int(row['n_trades']):>8} {row['win_rate']:>9.1f}% {row['avg_pips']:>+9.1f} "
                      f"{row['profit_factor']:>7.2f} {row['total_pips']:>+11.0f}")

    # =========================================================================
    # BEST CONFIG PER TIMEFRAME
    # =========================================================================
    log("\n" + "="*70)
    log("BEST CONFIGURATION PER TIMEFRAME")
    log("="*70)

    for tf in ['H4', 'H1', 'M5']:
        tf_data = combined_results[combined_results['timeframe'] == tf]
        if len(tf_data) == 0:
            continue

        qualified = tf_data[tf_data['n_trades'] >= 30]
        if len(qualified) == 0:
            qualified = tf_data[tf_data['n_trades'] >= 10]

        if len(qualified) == 0:
            continue

        best_pf = qualified.loc[qualified['profit_factor'].idxmax()]
        best_wr = qualified.loc[qualified['win_rate'].idxmax()]

        log(f"\n{tf}:")
        log(f"  Best Profit Factor: vel>={best_pf['velocity_threshold']:.2f}, {int(best_pf['horizon_bars'])} bars ({best_pf['time_equiv']})")
        log(f"    PF={best_pf['profit_factor']:.2f}, WR={best_pf['win_rate']:.1f}%, Avg={best_pf['avg_pips']:+.1f} pips, n={int(best_pf['n_trades'])}")
        log(f"  Best Win Rate: vel>={best_wr['velocity_threshold']:.2f}, {int(best_wr['horizon_bars'])} bars ({best_wr['time_equiv']})")
        log(f"    WR={best_wr['win_rate']:.1f}%, PF={best_wr['profit_factor']:.2f}, Avg={best_wr['avg_pips']:+.1f} pips, n={int(best_wr['n_trades'])}")

    # =========================================================================
    # CONCLUSION
    # =========================================================================
    log("\n" + "="*70)
    log("CONCLUSION: IS PATTERN TRADEABLE ON LOWER TIMEFRAMES?")
    log("="*70)

    for tf in ['H4', 'H1', 'M5']:
        tf_data = combined_results[combined_results['timeframe'] == tf]
        if len(tf_data) == 0:
            log(f"\n{tf}: No data available")
            continue

        qualified = tf_data[tf_data['n_trades'] >= 20]
        if len(qualified) == 0:
            log(f"\n{tf}: Insufficient trades (<20)")
            continue

        best = qualified.loc[qualified['profit_factor'].idxmax()]

        is_profitable = best['avg_pips'] > 0
        has_edge = best['profit_factor'] > 1.0
        decent_wr = best['win_rate'] >= 45

        if is_profitable and has_edge and decent_wr:
            status = "TRADEABLE"
        elif is_profitable and has_edge:
            status = "MARGINAL (low WR)"
        elif is_profitable:
            status = "WEAK EDGE"
        else:
            status = "NOT RECOMMENDED"

        log(f"\n{tf}: {status}")
        log(f"  Best: vel>={best['velocity_threshold']:.2f}, {int(best['horizon_bars'])} bars ({best['time_equiv']})")
        log(f"  WR={best['win_rate']:.1f}%, Avg={best['avg_pips']:+.1f} pips, PF={best['profit_factor']:.2f}, n={int(best['n_trades'])}")

    # Save
    output_file = OUTPUT_PATH / 'divergence_multi_tf_results.csv'
    combined_results.to_csv(output_file, index=False)
    log(f"\n\nResults saved to: {output_file}")

    log("\n" + "="*70)
    log("ANALYSIS COMPLETE")
    log("="*70)

if __name__ == "__main__":
    main()
