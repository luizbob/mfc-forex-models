"""
Prepare Quality Entry Dataset - V2 Both Currencies (M15) - FAST VERSION
=======================================================================
Optimized with vectorized numpy operations instead of per-entry loops.
FIXED: Velocity calculated in native timeframe before ffill.

Triggers on BOTH base AND quote currency extremes.
Uses cleaned MFC data and M1 price data.
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
log("PREPARING QUALITY ENTRY DATASET - V2 BOTH CURRENCIES (M15) - FAST")
log("=" * 70)
log(f"Started: {datetime.now()}")

DATA_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/data')
CLEANED_DIR = DATA_DIR / 'cleaned'
OUTPUT_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/lstm_model/v2_both_currencies/data')

CURRENCIES = ['EUR', 'USD', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD', 'NZD']

# Quality entry parameters
MFC_EXTREME_THRESHOLD = 0.5
MAX_ADVERSE_MFC = 0.25
RETURN_TARGET = 0.0
MAX_BARS_TO_RETURN = 800  # M15 bars


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
    return df['Close']


def find_first_crossing(arr_2d, threshold, direction='ge'):
    """Find first index where condition is met for each row."""
    n_rows, n_cols = arr_2d.shape
    result = np.full(n_rows, n_cols, dtype=np.int32)

    if direction == 'ge':
        mask = arr_2d >= threshold
    else:
        mask = arr_2d <= threshold

    for i in range(n_rows):
        indices = np.where(mask[i])[0]
        if len(indices) > 0:
            result[i] = indices[0] + 1

    return result


def process_entries_batch(df, indices, mfc_arr, price_arr, shifted_arr,
                          direction, trigger_type, trigger_ccy,
                          base_ccy, quote_ccy, pair, max_bars):
    """Process a batch of entries efficiently using vectorized operations."""
    n_total = len(df)

    # Filter indices that have enough future data
    valid_mask = indices + 1 + max_bars < n_total
    indices = indices[valid_mask]

    if len(indices) == 0:
        return None

    n = len(indices)
    log(f"    Processing {n} {trigger_type} {'buy' if direction == 1 else 'sell'} entries...")

    # Get entry values
    entry_mfc_values = shifted_arr[indices]
    entry_prices = price_arr[indices + 1]

    # Build future MFC and price matrices
    future_mfc_matrix = np.zeros((n, max_bars), dtype=np.float64)
    future_price_matrix = np.zeros((n, max_bars), dtype=np.float64)

    for i, idx in enumerate(indices):
        future_mfc_matrix[i] = mfc_arr[idx + 1:idx + 1 + max_bars]
        future_price_matrix[i] = price_arr[idx + 1:idx + 1 + max_bars]

    # Determine if we expect MFC to go up or down
    expect_up = (trigger_type == 'base' and direction == 1) or \
                (trigger_type == 'quote' and direction == -1)

    # Calculate quality metrics - vectorized
    if expect_up:
        min_mfc = np.min(future_mfc_matrix, axis=1)
        adverse_move = entry_mfc_values - min_mfc
        returned = np.any(future_mfc_matrix >= RETURN_TARGET, axis=1).astype(np.int32)
        bars_to_exit = find_first_crossing(future_mfc_matrix, 0, 'ge')
    else:
        max_mfc = np.max(future_mfc_matrix, axis=1)
        adverse_move = max_mfc - entry_mfc_values
        returned = np.any(future_mfc_matrix <= RETURN_TARGET, axis=1).astype(np.int32)
        bars_to_exit = find_first_crossing(future_mfc_matrix, 0, 'le')

    is_quality = ((returned == 1) & (adverse_move < MAX_ADVERSE_MFC)).astype(np.int32)

    # Price metrics - vectorized
    if direction == 1:  # buy
        max_dd_pips = (entry_prices - np.min(future_price_matrix, axis=1)) * 10000
        max_profit_pips = (np.max(future_price_matrix, axis=1) - entry_prices) * 10000
    else:  # sell
        max_dd_pips = (np.max(future_price_matrix, axis=1) - entry_prices) * 10000
        max_profit_pips = (entry_prices - np.min(future_price_matrix, axis=1)) * 10000

    # Calculate exit PnL
    # FIXED: Exit at NEXT bar after MFC crosses 0 (realistic - we only know at bar close)
    exit_indices = np.clip(bars_to_exit, 0, max_bars - 1)
    exit_prices = future_price_matrix[np.arange(n), exit_indices]

    if direction == 1:  # buy
        exit_pnl_pips = (exit_prices - entry_prices) * 10000
    else:  # sell
        exit_pnl_pips = (entry_prices - exit_prices) * 10000

    # Build result dataframe
    results = {
        'datetime': df.index[indices],
        'direction': ['buy' if direction == 1 else 'sell'] * n,
        'trigger': [trigger_type] * n,
        'trigger_ccy': [trigger_ccy] * n,
        'is_quality': is_quality,
        'returned': returned,
        'adverse_move': adverse_move,
        'max_dd_pips': max_dd_pips,
        'max_profit_pips': max_profit_pips,
        'exit_pnl_pips': exit_pnl_pips,
        'bars_to_exit': bars_to_exit,
        'pair': [pair] * n,
        'base_ccy': [base_ccy] * n,
        'quote_ccy': [quote_ccy] * n,
    }

    # Add features from df (already shifted)
    for tf in ['m5', 'm15', 'm30', 'h1', 'h4']:
        results[f'base_{tf}'] = df[f'base_{tf}'].values[indices]
        results[f'quote_{tf}'] = df[f'quote_{tf}'].values[indices]
        results[f'base_vel_{tf}'] = df[f'base_vel_{tf}'].values[indices]
        results[f'quote_vel_{tf}'] = df[f'quote_vel_{tf}'].values[indices]

    results['base_vel2_m15'] = df['base_vel2_m15'].values[indices]
    results['base_acc_m15'] = df['base_acc_m15'].values[indices]
    results['quote_vel2_m15'] = df['quote_vel2_m15'].values[indices]
    results['quote_acc_m15'] = df['quote_acc_m15'].values[indices]
    results['divergence'] = df['divergence'].values[indices]
    results['vel_divergence'] = df['vel_divergence'].values[indices]

    return pd.DataFrame(results)


def create_dataset_for_pair(base_ccy, quote_ccy):
    """Create quality entry dataset for a currency pair using both currency extremes."""
    pair = f"{base_ccy}{quote_ccy}"
    log(f"\nProcessing {pair}...")

    # Load all timeframes (cleaned data)
    base_mfc = {}
    quote_mfc = {}

    for tf in ['M5', 'M15', 'M30', 'H1', 'H4']:
        base_mfc[tf] = load_mfc_cleaned(base_ccy, tf)
        quote_mfc[tf] = load_mfc_cleaned(quote_ccy, tf)

    if base_mfc['M15'] is None or quote_mfc['M15'] is None:
        log(f"  Missing M15 data")
        return None

    # Load price (M1)
    price = load_price_data(pair)
    if price is None:
        log(f"  Missing price data")
        return None

    price_m15 = price.resample('15min').last().dropna()

    # Create base dataframe on M15 grid
    df = pd.DataFrame(index=base_mfc['M15'].index)
    df['price'] = price_m15

    # Store raw M15 for exit tracking (unshifted)
    df['base_m15_raw'] = base_mfc['M15']
    df['quote_m15_raw'] = quote_mfc['M15']

    # =======================================================================
    # FIXED: Calculate level and velocity in NATIVE timeframe, then ffill
    # =======================================================================
    for tf in ['M5', 'M15', 'M30', 'H1', 'H4']:
        tf_lower = tf.lower()
        base_raw = base_mfc[tf]
        quote_raw = quote_mfc[tf]

        if base_raw is not None:
            # MFC level: shift by 1 bar of native TF (what we knew at bar start)
            base_level = base_raw.shift(1)
            # Velocity: diff in native TF, then shift by 1 bar
            # This gives us MFC[T-1] - MFC[T-2] at bar T
            base_vel = base_raw.diff().shift(1)

            if tf == 'M15':
                # Native timeframe - no ffill needed
                df[f'base_{tf_lower}'] = base_level
                df[f'base_vel_{tf_lower}'] = base_vel
            else:
                # Higher/lower TFs - ffill to M15 grid
                df[f'base_{tf_lower}'] = base_level.reindex(df.index, method='ffill')
                df[f'base_vel_{tf_lower}'] = base_vel.reindex(df.index, method='ffill')

        if quote_raw is not None:
            quote_level = quote_raw.shift(1)
            quote_vel = quote_raw.diff().shift(1)

            if tf == 'M15':
                df[f'quote_{tf_lower}'] = quote_level
                df[f'quote_vel_{tf_lower}'] = quote_vel
            else:
                df[f'quote_{tf_lower}'] = quote_level.reindex(df.index, method='ffill')
                df[f'quote_vel_{tf_lower}'] = quote_vel.reindex(df.index, method='ffill')

    # Additional M15 momentum features (on already-shifted data)
    df['base_vel2_m15'] = df['base_m15'].diff(2)
    df['base_acc_m15'] = df['base_vel_m15'].diff()
    df['quote_vel2_m15'] = df['quote_m15'].diff(2)
    df['quote_acc_m15'] = df['quote_vel_m15'].diff()

    # Divergence (on shifted M15)
    df['divergence'] = df['base_m15'] - df['quote_m15']
    df['vel_divergence'] = df['base_vel_m15'] - df['quote_vel_m15']

    df = df.dropna()
    log(f"  Data: {len(df)} M15 bars ({df.index[0].date()} to {df.index[-1].date()})")

    # Convert to numpy for speed
    base_mfc_arr = df['base_m15_raw'].values  # Raw for future tracking
    quote_mfc_arr = df['quote_m15_raw'].values
    price_arr = df['price'].values
    base_shifted_arr = df['base_m15'].values  # Shifted for entry detection
    quote_shifted_arr = df['quote_m15'].values

    all_entries = []

    # Process each trigger type
    for trigger_type, trigger_ccy in [('base', base_ccy), ('quote', quote_ccy)]:
        if trigger_type == 'base':
            shifted_arr = base_shifted_arr
            mfc_arr = base_mfc_arr
        else:
            shifted_arr = quote_shifted_arr
            mfc_arr = quote_mfc_arr

        # Find trigger masks
        buy_mask = shifted_arr <= -MFC_EXTREME_THRESHOLD
        sell_mask = shifted_arr >= MFC_EXTREME_THRESHOLD

        if trigger_type == 'base':
            buy_indices = np.where(buy_mask)[0]
            sell_indices = np.where(sell_mask)[0]
        else:
            sell_indices = np.where(buy_mask)[0]
            buy_indices = np.where(sell_mask)[0]

        # Process buys
        if len(buy_indices) > 0:
            entries = process_entries_batch(
                df, buy_indices, mfc_arr, price_arr, shifted_arr,
                direction=1, trigger_type=trigger_type,
                trigger_ccy=trigger_ccy, base_ccy=base_ccy, quote_ccy=quote_ccy,
                pair=pair, max_bars=MAX_BARS_TO_RETURN
            )
            if entries is not None:
                all_entries.append(entries)

        # Process sells
        if len(sell_indices) > 0:
            entries = process_entries_batch(
                df, sell_indices, mfc_arr, price_arr, shifted_arr,
                direction=-1, trigger_type=trigger_type,
                trigger_ccy=trigger_ccy, base_ccy=base_ccy, quote_ccy=quote_ccy,
                pair=pair, max_bars=MAX_BARS_TO_RETURN
            )
            if entries is not None:
                all_entries.append(entries)

    if len(all_entries) == 0:
        return None

    result_df = pd.concat(all_entries, ignore_index=True)

    # Count by trigger type
    base_triggers = len(result_df[result_df['trigger'] == 'base'])
    quote_triggers = len(result_df[result_df['trigger'] == 'quote'])
    quality_rate = result_df['is_quality'].mean() * 100

    log(f"  Entries: {len(result_df)} (Base triggers: {base_triggers}, Quote triggers: {quote_triggers})")
    log(f"  Quality rate: {quality_rate:.1f}%")

    return result_df


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
    log(f"Quality entries: {final_df['is_quality'].sum():,} ({final_df['is_quality'].mean()*100:.1f}%)")

    # By trigger type
    log(f"\nBy Trigger:")
    for trigger in ['base', 'quote']:
        subset = final_df[final_df['trigger'] == trigger]
        log(f"  {trigger.upper()}: {len(subset):,} entries, {subset['is_quality'].mean()*100:.1f}% quality")

    # By direction
    log(f"\nBy Direction:")
    for direction in ['buy', 'sell']:
        subset = final_df[final_df['direction'] == direction]
        log(f"  {direction.upper()}: {len(subset):,} entries, {subset['is_quality'].mean()*100:.1f}% quality")

    # By trigger currency
    log(f"\nBy Trigger Currency:")
    for ccy in CURRENCIES:
        subset = final_df[final_df['trigger_ccy'] == ccy]
        if len(subset) > 0:
            log(f"  {ccy}: {len(subset):,} entries, {subset['is_quality'].mean()*100:.1f}% quality")

    # Save
    output_path = OUTPUT_DIR / 'quality_entry_data_m15_v2.pkl'
    output = {
        'data': final_df,
        'config': {
            'mfc_extreme_threshold': MFC_EXTREME_THRESHOLD,
            'max_adverse_mfc': MAX_ADVERSE_MFC,
            'return_target': RETURN_TARGET,
            'max_bars_to_return': MAX_BARS_TO_RETURN,
            'base_timeframe': 'M15',
            'version': 'v2_both_currencies_fixed',
            'triggers': ['base', 'quote'],
        }
    }

    with open(output_path, 'wb') as f:
        pickle.dump(output, f)

    log(f"\nSaved to: {output_path}")
    log(f"Completed: {datetime.now()}")


if __name__ == '__main__':
    main()
