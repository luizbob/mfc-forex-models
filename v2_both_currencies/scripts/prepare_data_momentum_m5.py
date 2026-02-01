"""
Prepare Momentum Entry Dataset - M5
====================================
Trades WITH momentum after MFC crosses key levels.

Entry triggers:
- BUY when base MFC crosses UP through 0 (momentum up, target +0.5)
- SELL when base MFC crosses DOWN through 0 (momentum down, target -0.5)
- Same for quote currency (reversed pair direction)

Exit conditions:
- Target reached (MFC hits ±0.5)
- Stop: MFC reverses back through 0

Features include velocity, acceleration, time since extreme, etc.
"""
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime

def log(msg=""):
    print(msg, flush=True)

log("=" * 70)
log("PREPARING MOMENTUM ENTRY DATASET (M5)")
log("=" * 70)
log(f"Started: {datetime.now()}")

DATA_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/data')
CLEANED_DIR = DATA_DIR / 'cleaned'
OUTPUT_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/lstm_model/v2_both_currencies/data')

CURRENCIES = ['EUR', 'USD', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD', 'NZD']

# Strategy parameters
CROSS_LEVEL = 0.0          # Entry when crossing this level
TARGET_LEVEL = 0.5         # Target to reach
MAX_BARS_TO_TARGET = 200   # Max bars to wait for target
MIN_VELOCITY = 0.0         # Minimum velocity at crossing (0 = no filter)


def load_mfc_cleaned(currency, timeframe):
    """Load cleaned MFC data for a currency and timeframe."""
    fp = CLEANED_DIR / f'mfc_currency_{currency}_{timeframe}_clean.csv'
    if not fp.exists():
        return None
    df = pd.read_csv(fp)
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df = df.set_index('datetime').sort_index()
    return df['MFC']


def load_price_data(pair):
    """Load M1 price data for a pair."""
    fp = DATA_DIR / f'{pair}_GMT+0_US-DST_M1.csv'
    if not fp.exists():
        return None
    df = pd.read_csv(fp)
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df = df.set_index('datetime').sort_index()
    return df[['Open', 'Close']]


def find_bars_since_extreme(mfc_arr, idx, threshold=0.5):
    """Find how many bars since MFC was at opposite extreme."""
    # Look back to find last time it was at opposite extreme
    for i in range(idx - 1, max(0, idx - 500), -1):
        if abs(mfc_arr[i]) >= threshold:
            return idx - i
    return 500  # Default if not found


def create_dataset_for_pair(base_ccy, quote_ccy):
    """Create momentum entry dataset for a currency pair."""
    pair = f"{base_ccy}{quote_ccy}"
    log(f"\nProcessing {pair}...")

    # Load all timeframes
    base_mfc = {}
    quote_mfc = {}

    for tf in ['M5', 'M15', 'M30', 'H1', 'H4']:
        base_mfc[tf] = load_mfc_cleaned(base_ccy, tf)
        quote_mfc[tf] = load_mfc_cleaned(quote_ccy, tf)

    if base_mfc['M5'] is None or quote_mfc['M5'] is None:
        log(f"  Missing M5 data")
        return None

    # Load price
    price_df = load_price_data(pair)
    if price_df is None:
        log(f"  Missing price data")
        return None

    price_m5_open = price_df['Open'].resample('5min').first().dropna()
    price_m5_close = price_df['Close'].resample('5min').last().dropna()

    # Create base dataframe on M5 grid
    df = pd.DataFrame(index=base_mfc['M5'].index)
    df['price_open'] = price_m5_open
    df['price_close'] = price_m5_close

    # =========================================================================
    # SHIFT EVERYTHING FIRST (no leakage)
    # Each timeframe shifted by 1 bar of its own TF, velocity calculated in native TF
    # After this block, all data is "safe" to use directly
    # =========================================================================

    for tf in ['M5', 'M15', 'M30', 'H1', 'H4']:
        tf_lower = tf.lower()

        base_raw = base_mfc[tf]
        quote_raw = quote_mfc[tf]

        if base_raw is not None:
            # MFC level: shift by 1 bar of native TF
            base_level = base_raw.shift(1)
            # Velocity: diff in native TF, then shift by 1 bar
            base_vel = base_raw.diff().shift(1)

            if tf == 'M5':
                df[f'base_{tf_lower}'] = base_level
                df[f'base_vel_{tf_lower}'] = base_vel
            else:
                # Ffill to M5 grid for higher TFs
                df[f'base_{tf_lower}'] = base_level.reindex(df.index, method='ffill')
                df[f'base_vel_{tf_lower}'] = base_vel.reindex(df.index, method='ffill')

        if quote_raw is not None:
            quote_level = quote_raw.shift(1)
            quote_vel = quote_raw.diff().shift(1)

            if tf == 'M5':
                df[f'quote_{tf_lower}'] = quote_level
                df[f'quote_vel_{tf_lower}'] = quote_vel
            else:
                df[f'quote_{tf_lower}'] = quote_level.reindex(df.index, method='ffill')
                df[f'quote_vel_{tf_lower}'] = quote_vel.reindex(df.index, method='ffill')

    df = df.dropna()

    # =========================================================================
    # ADDITIONAL FEATURES (calculated on already-shifted data)
    # =========================================================================

    # Renamed for consistency with old code (all already shifted)
    df['base_m5_shifted'] = df['base_m5']
    df['quote_m5_shifted'] = df['quote_m5']
    for tf in ['m15', 'm30', 'h1', 'h4']:
        df[f'base_{tf}_shifted'] = df[f'base_{tf}']
        df[f'quote_{tf}_shifted'] = df[f'quote_{tf}']

    # Multi-bar velocities for M5
    df['base_vel3_m5'] = df['base_m5'].diff(3) / 3
    df['base_vel5_m5'] = df['base_m5'].diff(5) / 5
    df['quote_vel3_m5'] = df['quote_m5'].diff(3) / 3
    df['quote_vel5_m5'] = df['quote_m5'].diff(5) / 5

    # Acceleration
    df['base_acc_m5'] = df['base_vel_m5'].diff()
    df['quote_acc_m5'] = df['quote_vel_m5'].diff()

    # Divergence
    df['divergence'] = df['base_m5'] - df['quote_m5']
    df['vel_divergence'] = df['base_vel_m5'] - df['quote_vel_m5']

    df = df.dropna()
    log(f"  Data: {len(df)} M5 bars ({df.index[0].date()} to {df.index[-1].date()})")

    # Convert to numpy for speed
    # Note: base_m5 and quote_m5 are ALREADY shifted (for features/triggers)
    # We need RAW MFC for tracking exit (target reached, stopped out)
    base_shifted = df['base_m5'].values  # shifted - for triggers
    quote_shifted = df['quote_m5'].values  # shifted - for triggers
    base_vel = df['base_vel_m5'].values
    quote_vel = df['quote_vel_m5'].values
    price_open = df['price_open'].values
    price_close = df['price_close'].values

    # Load raw MFC for exit tracking
    base_raw_m5 = base_mfc['M5'].reindex(df.index)
    quote_raw_m5 = quote_mfc['M5'].reindex(df.index)
    base_m5_arr = base_raw_m5.values  # raw - for exit tracking
    quote_m5_arr = quote_raw_m5.values  # raw - for exit tracking

    all_entries = []

    # === BASE CURRENCY CROSSINGS ===
    # Cross UP through 0 -> BUY (expecting base to strengthen to +0.5)
    cross_up_base = (base_shifted <= CROSS_LEVEL) & (df['base_m5_shifted'].shift(-1) > CROSS_LEVEL)
    cross_up_idx = np.where(cross_up_base.values[:-1])[0]

    entries = process_crossings(
        df, cross_up_idx, base_m5_arr, price_open, price_close, base_vel,
        direction=1, trigger='base', trigger_ccy=base_ccy,
        base_ccy=base_ccy, quote_ccy=quote_ccy, pair=pair,
        target=TARGET_LEVEL
    )
    if entries is not None:
        all_entries.append(entries)

    # Cross DOWN through 0 -> SELL (expecting base to weaken to -0.5)
    cross_down_base = (base_shifted >= CROSS_LEVEL) & (df['base_m5_shifted'].shift(-1) < CROSS_LEVEL)
    cross_down_idx = np.where(cross_down_base.values[:-1])[0]

    entries = process_crossings(
        df, cross_down_idx, base_m5_arr, price_open, price_close, base_vel,
        direction=-1, trigger='base', trigger_ccy=base_ccy,
        base_ccy=base_ccy, quote_ccy=quote_ccy, pair=pair,
        target=-TARGET_LEVEL
    )
    if entries is not None:
        all_entries.append(entries)

    # === QUOTE CURRENCY CROSSINGS ===
    # Quote cross UP -> SELL pair (quote strengthening)
    cross_up_quote = (quote_shifted <= CROSS_LEVEL) & (df['quote_m5_shifted'].shift(-1) > CROSS_LEVEL)
    cross_up_idx = np.where(cross_up_quote.values[:-1])[0]

    entries = process_crossings(
        df, cross_up_idx, quote_m5_arr, price_open, price_close, quote_vel,
        direction=-1, trigger='quote', trigger_ccy=quote_ccy,
        base_ccy=base_ccy, quote_ccy=quote_ccy, pair=pair,
        target=TARGET_LEVEL
    )
    if entries is not None:
        all_entries.append(entries)

    # Quote cross DOWN -> BUY pair (quote weakening)
    cross_down_quote = (quote_shifted >= CROSS_LEVEL) & (df['quote_m5_shifted'].shift(-1) < CROSS_LEVEL)
    cross_down_idx = np.where(cross_down_quote.values[:-1])[0]

    entries = process_crossings(
        df, cross_down_idx, quote_m5_arr, price_open, price_close, quote_vel,
        direction=1, trigger='quote', trigger_ccy=quote_ccy,
        base_ccy=base_ccy, quote_ccy=quote_ccy, pair=pair,
        target=-TARGET_LEVEL
    )
    if entries is not None:
        all_entries.append(entries)

    if len(all_entries) == 0:
        return None

    result_df = pd.concat(all_entries, ignore_index=True)

    # Stats
    base_triggers = len(result_df[result_df['trigger'] == 'base'])
    quote_triggers = len(result_df[result_df['trigger'] == 'quote'])
    success_rate = result_df['reached_target'].mean() * 100
    avg_pnl = result_df['exit_pnl_pips'].mean()

    log(f"  Entries: {len(result_df)} (Base: {base_triggers}, Quote: {quote_triggers})")
    log(f"  Target success: {success_rate:.1f}%, Avg PnL: {avg_pnl:.1f} pips")

    return result_df


def process_crossings(df, indices, mfc_arr, price_open, price_close, vel_arr,
                      direction, trigger, trigger_ccy, base_ccy, quote_ccy, pair, target):
    """Process crossing entries."""
    n_total = len(df)
    max_bars = MAX_BARS_TO_TARGET

    # Filter valid indices
    valid_mask = (indices + 1 + max_bars < n_total) & (indices > 10)
    indices = indices[valid_mask]

    if len(indices) == 0:
        return None

    n = len(indices)

    # Arrays for results
    reached_target = np.zeros(n, dtype=np.int32)
    bars_to_target = np.full(n, max_bars, dtype=np.int32)
    stopped_out = np.zeros(n, dtype=np.int32)
    bars_to_stop = np.full(n, max_bars, dtype=np.int32)
    exit_pnl_pips = np.zeros(n, dtype=np.float64)
    max_favorable = np.zeros(n, dtype=np.float64)
    max_adverse = np.zeros(n, dtype=np.float64)
    velocity_at_cross = np.zeros(n, dtype=np.float64)

    for i, idx in enumerate(indices):
        # Entry at next bar
        entry_price = price_open[idx + 1]
        velocity_at_cross[i] = vel_arr[idx]

        future_mfc = mfc_arr[idx + 1:idx + 1 + max_bars]
        future_price = price_close[idx + 1:idx + 1 + max_bars]

        # Check if target reached
        if target > 0:
            target_mask = future_mfc >= target
            stop_mask = future_mfc <= -0.1  # Small buffer below 0
            max_favorable[i] = future_mfc.max()
            max_adverse[i] = future_mfc.min()
        else:
            target_mask = future_mfc <= target
            stop_mask = future_mfc >= 0.1  # Small buffer above 0
            max_favorable[i] = future_mfc.min()
            max_adverse[i] = future_mfc.max()

        target_idx = np.argmax(target_mask) if target_mask.any() else max_bars
        stop_idx = np.argmax(stop_mask) if stop_mask.any() else max_bars

        if target_mask.any() and target_idx < stop_idx:
            # Target reached first
            reached_target[i] = 1
            bars_to_target[i] = target_idx + 1
            exit_price = future_price[target_idx]
        elif stop_mask.any():
            # Stopped out
            stopped_out[i] = 1
            bars_to_stop[i] = stop_idx + 1
            exit_price = future_price[stop_idx]
        else:
            # Timeout
            exit_price = future_price[-1]

        # Calculate PnL
        if direction == 1:  # BUY
            exit_pnl_pips[i] = (exit_price - entry_price) * 10000
        else:  # SELL
            exit_pnl_pips[i] = (entry_price - exit_price) * 10000

    # Build results
    results = {
        'datetime': df.index[indices],
        'direction': ['buy' if direction == 1 else 'sell'] * n,
        'trigger': [trigger] * n,
        'trigger_ccy': [trigger_ccy] * n,
        'reached_target': reached_target,
        'bars_to_target': bars_to_target,
        'stopped_out': stopped_out,
        'bars_to_stop': bars_to_stop,
        'exit_pnl_pips': exit_pnl_pips,
        'max_favorable_mfc': max_favorable,
        'max_adverse_mfc': max_adverse,
        'velocity_at_cross': velocity_at_cross,
        'pair': [pair] * n,
        'base_ccy': [base_ccy] * n,
        'quote_ccy': [quote_ccy] * n,
    }

    # Add features
    for tf in ['m5', 'm15', 'm30', 'h1', 'h4']:
        results[f'base_{tf}'] = df[f'base_{tf}_shifted'].values[indices]
        results[f'quote_{tf}'] = df[f'quote_{tf}_shifted'].values[indices]
        results[f'base_vel_{tf}'] = df[f'base_vel_{tf}'].values[indices]
        results[f'quote_vel_{tf}'] = df[f'quote_vel_{tf}'].values[indices]

    results['base_vel3_m5'] = df['base_vel3_m5'].values[indices]
    results['base_vel5_m5'] = df['base_vel5_m5'].values[indices]
    results['quote_vel3_m5'] = df['quote_vel3_m5'].values[indices]
    results['quote_vel5_m5'] = df['quote_vel5_m5'].values[indices]
    results['base_acc_m5'] = df['base_acc_m5'].values[indices]
    results['quote_acc_m5'] = df['quote_acc_m5'].values[indices]
    results['divergence'] = df['divergence'].values[indices]
    results['vel_divergence'] = df['vel_divergence'].values[indices]

    return pd.DataFrame(results)


def main():
    pairs = [
        ('EUR', 'USD'), ('EUR', 'GBP'), ('EUR', 'JPY'), ('EUR', 'CHF'),
        ('EUR', 'CAD'), ('EUR', 'AUD'), ('EUR', 'NZD'),
        ('GBP', 'USD'), ('GBP', 'JPY'), ('GBP', 'CHF'), ('GBP', 'CAD'),
        ('GBP', 'AUD'), ('GBP', 'NZD'),
        ('USD', 'JPY'), ('USD', 'CHF'), ('USD', 'CAD'),
        ('AUD', 'USD'), ('AUD', 'JPY'), ('AUD', 'CHF'), ('AUD', 'CAD'), ('AUD', 'NZD'),
        ('NZD', 'USD'), ('NZD', 'JPY'), ('NZD', 'CHF'), ('NZD', 'CAD'),
        ('CAD', 'JPY'), ('CAD', 'CHF'), ('CHF', 'JPY'),
    ]

    all_entries = []
    for base_ccy, quote_ccy in pairs:
        result = create_dataset_for_pair(base_ccy, quote_ccy)
        if result is not None:
            all_entries.append(result)

    log("\n" + "=" * 70)
    log("COMBINING ALL DATA")
    log("=" * 70)

    final_df = pd.concat(all_entries, ignore_index=True)

    log(f"\nTotal entries: {len(final_df):,}")
    log(f"Reached target: {final_df['reached_target'].sum():,} ({final_df['reached_target'].mean()*100:.1f}%)")
    log(f"Stopped out: {final_df['stopped_out'].sum():,} ({final_df['stopped_out'].mean()*100:.1f}%)")

    # Profitable trades
    profitable = (final_df['exit_pnl_pips'] > 0).mean() * 100
    avg_pnl = final_df['exit_pnl_pips'].mean()
    total_pnl = final_df['exit_pnl_pips'].sum()
    log(f"Profitable: {profitable:.1f}%")
    log(f"Avg PnL: {avg_pnl:.2f} pips")
    log(f"Total PnL: {total_pnl:,.0f} pips")

    # By velocity quartile
    log(f"\nBy Velocity at Crossing:")
    final_df['vel_quartile'] = pd.qcut(final_df['velocity_at_cross'].abs(), 4, labels=['Q1 (slow)', 'Q2', 'Q3', 'Q4 (fast)'])
    for q in ['Q1 (slow)', 'Q2', 'Q3', 'Q4 (fast)']:
        subset = final_df[final_df['vel_quartile'] == q]
        log(f"  {q}: {len(subset):,} entries, {subset['reached_target'].mean()*100:.1f}% success, {subset['exit_pnl_pips'].mean():.1f} avg pips")

    # By trigger type
    log(f"\nBy Trigger:")
    for trigger in ['base', 'quote']:
        subset = final_df[final_df['trigger'] == trigger]
        log(f"  {trigger.upper()}: {len(subset):,} entries, {subset['reached_target'].mean()*100:.1f}% success")

    # By direction
    log(f"\nBy Direction:")
    for direction in ['buy', 'sell']:
        subset = final_df[final_df['direction'] == direction]
        log(f"  {direction.upper()}: {len(subset):,} entries, {subset['reached_target'].mean()*100:.1f}% success")

    # By trigger currency
    log(f"\nBy Trigger Currency:")
    for ccy in CURRENCIES:
        subset = final_df[final_df['trigger_ccy'] == ccy]
        if len(subset) > 0:
            log(f"  {ccy}: {len(subset):,} entries, {subset['reached_target'].mean()*100:.1f}% success, {subset['exit_pnl_pips'].mean():.1f} avg pips")

    # Save
    output_path = OUTPUT_DIR / 'momentum_entry_data_m5.pkl'
    output = {
        'data': final_df,
        'config': {
            'cross_level': CROSS_LEVEL,
            'target_level': TARGET_LEVEL,
            'max_bars_to_target': MAX_BARS_TO_TARGET,
            'base_timeframe': 'M5',
            'version': 'momentum_v1',
            'triggers': ['base', 'quote'],
        }
    }

    with open(output_path, 'wb') as f:
        pickle.dump(output, f)

    log(f"\nSaved to: {output_path}")
    log(f"Completed: {datetime.now()}")


if __name__ == '__main__':
    main()
