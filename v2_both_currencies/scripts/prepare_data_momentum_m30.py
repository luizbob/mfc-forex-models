"""
Prepare Momentum Entry Dataset - M30
=====================================
Trades WITH momentum after MFC crosses key levels.

Entry triggers when MFC crosses 0, target is ±0.5.
Includes all timeframes as features with proper shifting.
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
log("PREPARING MOMENTUM ENTRY DATASET (M30)")
log("=" * 70)
log(f"Started: {datetime.now()}")

DATA_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/data')
CLEANED_DIR = DATA_DIR / 'cleaned'
OUTPUT_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/lstm_model/v2_both_currencies/data')

CURRENCIES = ['EUR', 'USD', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD', 'NZD']

# Strategy parameters
CROSS_LEVEL = 0.0
TARGET_LEVEL = 0.5
MAX_BARS_TO_TARGET = 60  # M30 bars (~30 hours max)


def load_mfc_cleaned(currency, timeframe):
    """Load cleaned MFC data."""
    fp = CLEANED_DIR / f'mfc_currency_{currency}_{timeframe}_clean.csv'
    if not fp.exists():
        return None
    df = pd.read_csv(fp)
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df = df.set_index('datetime').sort_index()
    return df['MFC']


def load_price_data(pair):
    """Load M1 price data."""
    fp = DATA_DIR / f'{pair}_GMT+0_US-DST_M1.csv'
    if not fp.exists():
        return None
    df = pd.read_csv(fp)
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df = df.set_index('datetime').sort_index()
    return df[['Open', 'Close']]


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

    if base_mfc['M30'] is None or quote_mfc['M30'] is None:
        log(f"  Missing M30 data")
        return None

    # Load price
    price_df = load_price_data(pair)
    if price_df is None:
        log(f"  Missing price data")
        return None

    price_m30 = price_df['Close'].resample('30min').last().dropna()

    # Create base dataframe on M30 grid
    df = pd.DataFrame(index=base_mfc['M30'].index)
    df['base_m30'] = base_mfc['M30']
    df['quote_m30'] = quote_mfc['M30']
    df['price'] = price_m30

    # Add other timeframes with PROPER SHIFTING:
    # - Lower TFs (M5, M15): shift by 1 bar of THAT TF, then ffill to M30
    # - Higher TFs (H1, H4): shift by 1 bar of THAT TF, then ffill to M30

    # M5, M15 (lower TFs)
    for tf in ['M5', 'M15']:
        if base_mfc[tf] is not None:
            df[f'base_{tf.lower()}'] = base_mfc[tf].shift(1).reindex(df.index, method='ffill')
        if quote_mfc[tf] is not None:
            df[f'quote_{tf.lower()}'] = quote_mfc[tf].shift(1).reindex(df.index, method='ffill')

    # H1, H4 (higher TFs)
    for tf in ['H1', 'H4']:
        if base_mfc[tf] is not None:
            df[f'base_{tf.lower()}'] = base_mfc[tf].shift(1).reindex(df.index, method='ffill')
        if quote_mfc[tf] is not None:
            df[f'quote_{tf.lower()}'] = quote_mfc[tf].shift(1).reindex(df.index, method='ffill')

    df = df.dropna()

    # Shift for using previous bar's data
    # BASE TF (M30): shift by 1 M30 bar for trigger detection
    df['base_m30_shifted'] = df['base_m30'].shift(1)
    df['quote_m30_shifted'] = df['quote_m30'].shift(1)

    # OTHER TFs: already shifted by 1 bar of THEIR OWN TF before ffill
    # So we use them directly (renamed for consistency)
    for tf in ['m5', 'm15', 'h1', 'h4']:
        if f'base_{tf}' in df.columns:
            df[f'base_{tf}_shifted'] = df[f'base_{tf}']  # Already shifted
            df[f'quote_{tf}_shifted'] = df[f'quote_{tf}']  # Already shifted

    # Calculate velocities
    # M30 velocity: diff of shifted M30 (1 M30 bar change)
    df['base_vel_m30'] = df['base_m30_shifted'].diff()
    df['quote_vel_m30'] = df['quote_m30_shifted'].diff()

    # Other TF velocities: diff on their own shifted series
    for tf in ['m5', 'm15', 'h1', 'h4']:
        if f'base_{tf}_shifted' in df.columns:
            df[f'base_vel_{tf}'] = df[f'base_{tf}_shifted'].diff()
            df[f'quote_vel_{tf}'] = df[f'quote_{tf}_shifted'].diff()

    # Additional momentum features on M30
    df['base_vel3_m30'] = df['base_m30_shifted'].diff(3) / 3
    df['base_vel5_m30'] = df['base_m30_shifted'].diff(5) / 5
    df['quote_vel3_m30'] = df['quote_m30_shifted'].diff(3) / 3
    df['quote_vel5_m30'] = df['quote_m30_shifted'].diff(5) / 5
    df['base_acc_m30'] = df['base_vel_m30'].diff()
    df['quote_acc_m30'] = df['quote_vel_m30'].diff()

    # Divergence
    df['divergence'] = df['base_m30_shifted'] - df['quote_m30_shifted']
    df['vel_divergence'] = df['base_vel_m30'] - df['quote_vel_m30']

    df = df.dropna()
    log(f"  Data: {len(df)} M30 bars ({df.index[0].date()} to {df.index[-1].date()})")

    # Convert to numpy
    base_m30_arr = df['base_m30'].values
    quote_m30_arr = df['quote_m30'].values
    base_shifted = df['base_m30_shifted'].values
    quote_shifted = df['quote_m30_shifted'].values
    base_vel = df['base_vel_m30'].values
    quote_vel = df['quote_vel_m30'].values
    price_arr = df['price'].values

    all_entries = []

    # === BASE CURRENCY CROSSINGS ===
    cross_up_base = (base_shifted <= CROSS_LEVEL) & (df['base_m30_shifted'].shift(-1) > CROSS_LEVEL)
    cross_up_idx = np.where(cross_up_base.values[:-1])[0]

    entries = process_crossings(
        df, cross_up_idx, base_m30_arr, price_arr, base_vel,
        direction=1, trigger='base', trigger_ccy=base_ccy,
        base_ccy=base_ccy, quote_ccy=quote_ccy, pair=pair,
        target=TARGET_LEVEL
    )
    if entries is not None:
        all_entries.append(entries)

    cross_down_base = (base_shifted >= CROSS_LEVEL) & (df['base_m30_shifted'].shift(-1) < CROSS_LEVEL)
    cross_down_idx = np.where(cross_down_base.values[:-1])[0]

    entries = process_crossings(
        df, cross_down_idx, base_m30_arr, price_arr, base_vel,
        direction=-1, trigger='base', trigger_ccy=base_ccy,
        base_ccy=base_ccy, quote_ccy=quote_ccy, pair=pair,
        target=-TARGET_LEVEL
    )
    if entries is not None:
        all_entries.append(entries)

    # === QUOTE CURRENCY CROSSINGS ===
    cross_up_quote = (quote_shifted <= CROSS_LEVEL) & (df['quote_m30_shifted'].shift(-1) > CROSS_LEVEL)
    cross_up_idx = np.where(cross_up_quote.values[:-1])[0]

    entries = process_crossings(
        df, cross_up_idx, quote_m30_arr, price_arr, quote_vel,
        direction=-1, trigger='quote', trigger_ccy=quote_ccy,
        base_ccy=base_ccy, quote_ccy=quote_ccy, pair=pair,
        target=TARGET_LEVEL
    )
    if entries is not None:
        all_entries.append(entries)

    cross_down_quote = (quote_shifted >= CROSS_LEVEL) & (df['quote_m30_shifted'].shift(-1) < CROSS_LEVEL)
    cross_down_idx = np.where(cross_down_quote.values[:-1])[0]

    entries = process_crossings(
        df, cross_down_idx, quote_m30_arr, price_arr, quote_vel,
        direction=1, trigger='quote', trigger_ccy=quote_ccy,
        base_ccy=base_ccy, quote_ccy=quote_ccy, pair=pair,
        target=-TARGET_LEVEL
    )
    if entries is not None:
        all_entries.append(entries)

    if len(all_entries) == 0:
        return None

    result_df = pd.concat(all_entries, ignore_index=True)

    base_triggers = len(result_df[result_df['trigger'] == 'base'])
    quote_triggers = len(result_df[result_df['trigger'] == 'quote'])
    success_rate = result_df['reached_target'].mean() * 100

    log(f"  Entries: {len(result_df)} (Base: {base_triggers}, Quote: {quote_triggers})")
    log(f"  Target success: {success_rate:.1f}%")

    return result_df


def process_crossings(df, indices, mfc_arr, price_arr, vel_arr,
                      direction, trigger, trigger_ccy, base_ccy, quote_ccy, pair, target):
    """Process crossing entries."""
    n_total = len(df)
    max_bars = MAX_BARS_TO_TARGET

    valid_mask = (indices + 1 + max_bars < n_total) & (indices > 10)
    indices = indices[valid_mask]

    if len(indices) == 0:
        return None

    n = len(indices)

    reached_target = np.zeros(n, dtype=np.int32)
    bars_to_target = np.full(n, max_bars, dtype=np.int32)
    stopped_out = np.zeros(n, dtype=np.int32)
    bars_to_stop = np.full(n, max_bars, dtype=np.int32)
    exit_pnl_pips = np.zeros(n, dtype=np.float64)
    max_favorable = np.zeros(n, dtype=np.float64)
    max_adverse = np.zeros(n, dtype=np.float64)
    velocity_at_cross = np.zeros(n, dtype=np.float64)

    for i, idx in enumerate(indices):
        entry_price = price_arr[idx + 1]
        velocity_at_cross[i] = vel_arr[idx]

        future_mfc = mfc_arr[idx + 1:idx + 1 + max_bars]
        future_price = price_arr[idx + 1:idx + 1 + max_bars]

        if target > 0:
            target_mask = future_mfc >= target
            stop_mask = future_mfc <= -0.1
            max_favorable[i] = future_mfc.max()
            max_adverse[i] = future_mfc.min()
        else:
            target_mask = future_mfc <= target
            stop_mask = future_mfc >= 0.1
            max_favorable[i] = future_mfc.min()
            max_adverse[i] = future_mfc.max()

        target_idx = np.argmax(target_mask) if target_mask.any() else max_bars
        stop_idx = np.argmax(stop_mask) if stop_mask.any() else max_bars

        if target_mask.any() and target_idx < stop_idx:
            reached_target[i] = 1
            bars_to_target[i] = target_idx + 1
            exit_price = future_price[target_idx]
        elif stop_mask.any():
            stopped_out[i] = 1
            bars_to_stop[i] = stop_idx + 1
            exit_price = future_price[stop_idx]
        else:
            exit_price = future_price[-1]

        if direction == 1:
            exit_pnl_pips[i] = (exit_price - entry_price) * 10000
        else:
            exit_pnl_pips[i] = (entry_price - exit_price) * 10000

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
        if f'base_{tf}_shifted' in df.columns:
            results[f'base_{tf}'] = df[f'base_{tf}_shifted'].values[indices]
            results[f'quote_{tf}'] = df[f'quote_{tf}_shifted'].values[indices]
            results[f'base_vel_{tf}'] = df[f'base_vel_{tf}'].values[indices]
            results[f'quote_vel_{tf}'] = df[f'quote_vel_{tf}'].values[indices]

    results['base_vel3_m30'] = df['base_vel3_m30'].values[indices]
    results['base_vel5_m30'] = df['base_vel5_m30'].values[indices]
    results['quote_vel3_m30'] = df['quote_vel3_m30'].values[indices]
    results['quote_vel5_m30'] = df['quote_vel5_m30'].values[indices]
    results['base_acc_m30'] = df['base_acc_m30'].values[indices]
    results['quote_acc_m30'] = df['quote_acc_m30'].values[indices]
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

    profitable = (final_df['exit_pnl_pips'] > 0).mean() * 100
    avg_pnl = final_df['exit_pnl_pips'].mean()
    log(f"Profitable: {profitable:.1f}%")
    log(f"Avg PnL: {avg_pnl:.2f} pips")

    # By velocity
    log(f"\nBy Velocity at Crossing:")
    final_df['vel_quartile'] = pd.qcut(final_df['velocity_at_cross'].abs(), 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    for q in ['Q1', 'Q2', 'Q3', 'Q4']:
        subset = final_df[final_df['vel_quartile'] == q]
        log(f"  {q}: {len(subset):,} entries, {subset['reached_target'].mean()*100:.1f}% success, {subset['exit_pnl_pips'].mean():.1f} avg pips")

    # Save
    output_path = OUTPUT_DIR / 'momentum_entry_data_m30.pkl'
    output = {
        'data': final_df,
        'config': {
            'cross_level': CROSS_LEVEL,
            'target_level': TARGET_LEVEL,
            'max_bars_to_target': MAX_BARS_TO_TARGET,
            'base_timeframe': 'M30',
            'version': 'momentum_v1',
        }
    }

    with open(output_path, 'wb') as f:
        pickle.dump(output, f)

    log(f"\nSaved to: {output_path}")
    log(f"Completed: {datetime.now()}")


if __name__ == '__main__':
    main()
