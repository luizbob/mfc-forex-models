"""
Prepare Quality Entry Dataset - V2 Both Currencies (M5) - FAST VERSION
======================================================================
Optimized with vectorized numpy operations instead of per-entry loops.

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
log("PREPARING QUALITY ENTRY DATASET - V2 BOTH CURRENCIES (M5) - FAST")
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
MAX_BARS_TO_RETURN = 2400  # M5 bars (~200 hours max)


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
    """Load M1 price data for a pair (Open and Close)."""
    fp = DATA_DIR / f'{pair}_GMT+0_US-DST_M1.csv'
    if not fp.exists():
        return None
    df = pd.read_csv(fp)
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df = df.set_index('datetime').sort_index()
    return df[['Open', 'Close']]


def compute_rolling_metrics(arr, window):
    """Compute rolling min and max using stride tricks for efficiency."""
    n = len(arr)
    if n < window:
        return None, None

    # Create strided view for rolling window
    from numpy.lib.stride_tricks import sliding_window_view
    windows = sliding_window_view(arr, window)

    rolling_min = np.min(windows, axis=1)
    rolling_max = np.max(windows, axis=1)

    return rolling_min, rolling_max


def find_first_crossing(arr_2d, threshold, direction='ge'):
    """Find first index where condition is met for each row.
    direction: 'ge' for >=, 'le' for <=
    Returns array of indices (MAX_BARS if not found).
    """
    n_rows, n_cols = arr_2d.shape
    result = np.full(n_rows, n_cols, dtype=np.int32)

    if direction == 'ge':
        mask = arr_2d >= threshold
    else:
        mask = arr_2d <= threshold

    for i in range(n_rows):
        indices = np.where(mask[i])[0]
        if len(indices) > 0:
            result[i] = indices[0] + 1  # +1 for bars_to_exit

    return result


def process_entries_batch(df, indices, mfc_arr, price_open_arr, price_close_arr,
                          trigger_arr, direction, trigger_type, trigger_ccy,
                          base_ccy, quote_ccy, pair, max_bars):
    """Process a batch of entries efficiently using vectorized operations.

    Args:
        mfc_arr: RAW (unshifted) MFC for tracking future returns
        trigger_arr: SHIFTED MFC used for trigger detection and entry values
    """
    n_total = len(df)

    # Filter indices that have enough future data
    valid_mask = indices + 1 + max_bars < n_total
    indices = indices[valid_mask]

    if len(indices) == 0:
        return None

    n = len(indices)
    log(f"    Processing {n} {trigger_type} {'buy' if direction == 1 else 'sell'} entries...")

    # Get entry values (from shifted/trigger array - what we knew at decision time)
    entry_mfc_values = trigger_arr[indices]
    entry_prices = price_open_arr[indices + 1]  # Entry at next bar's open

    # Build future MFC and price matrices
    future_mfc_matrix = np.zeros((n, max_bars), dtype=np.float64)
    future_price_matrix = np.zeros((n, max_bars), dtype=np.float64)

    for i, idx in enumerate(indices):
        future_mfc_matrix[i] = mfc_arr[idx + 1:idx + 1 + max_bars]
        future_price_matrix[i] = price_close_arr[idx + 1:idx + 1 + max_bars]

    # Determine if we expect MFC to go up or down
    # base+buy or quote+sell: expect MFC up (was low)
    # base+sell or quote+buy: expect MFC down (was high)
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
        # Track WHEN max DD and max profit occurred (bar index)
        bars_to_max_dd = np.argmin(future_price_matrix, axis=1) + 1  # +1 for 1-based
        bars_to_max_profit = np.argmax(future_price_matrix, axis=1) + 1
    else:  # sell
        max_dd_pips = (np.max(future_price_matrix, axis=1) - entry_prices) * 10000
        max_profit_pips = (entry_prices - np.min(future_price_matrix, axis=1)) * 10000
        # For sell: DD is when price goes UP (argmax), profit when DOWN (argmin)
        bars_to_max_dd = np.argmax(future_price_matrix, axis=1) + 1
        bars_to_max_profit = np.argmin(future_price_matrix, axis=1) + 1

    # Calculate exit PnL
    # FIXED: Exit at NEXT bar after MFC crosses 0 (realistic - we only know at bar close)
    # bars_to_exit is 1-based, so using it directly gives us the next bar
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
        'bars_to_max_dd': bars_to_max_dd,
        'bars_to_max_profit': bars_to_max_profit,
        'exit_pnl_pips': exit_pnl_pips,
        'bars_to_exit': bars_to_exit,
        'pair': [pair] * n,
        'base_ccy': [base_ccy] * n,
        'quote_ccy': [quote_ccy] * n,
    }

    # Add features from df (all columns are already shifted, safe to use directly)
    for tf in ['m5', 'm15', 'm30', 'h1', 'h4']:
        results[f'base_{tf}'] = df[f'base_{tf}'].values[indices]
        results[f'quote_{tf}'] = df[f'quote_{tf}'].values[indices]
        results[f'base_vel_{tf}'] = df[f'base_vel_{tf}'].values[indices]
        results[f'quote_vel_{tf}'] = df[f'quote_vel_{tf}'].values[indices]

    results['base_vel2_m5'] = df['base_vel2_m5'].values[indices]
    results['base_acc_m5'] = df['base_acc_m5'].values[indices]
    results['quote_vel2_m5'] = df['quote_vel2_m5'].values[indices]
    results['quote_acc_m5'] = df['quote_acc_m5'].values[indices]
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

    if base_mfc['M5'] is None or quote_mfc['M5'] is None:
        log(f"  Missing M5 data")
        return None

    # Load price (M1)
    price_df = load_price_data(pair)
    if price_df is None:
        log(f"  Missing price data")
        return None

    # Resample to M5: open = first, close = last
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

    # M5 acceleration and 2-bar velocity
    df['base_vel2_m5'] = df['base_m5'].diff(2)      # 2-bar change
    df['base_acc_m5'] = df['base_vel_m5'].diff()    # acceleration
    df['quote_vel2_m5'] = df['quote_m5'].diff(2)
    df['quote_acc_m5'] = df['quote_vel_m5'].diff()

    # Divergence between base and quote
    df['divergence'] = df['base_m5'] - df['quote_m5']
    df['vel_divergence'] = df['base_vel_m5'] - df['quote_vel_m5']

    df = df.dropna()
    log(f"  Data: {len(df)} M5 bars ({df.index[0].date()} to {df.index[-1].date()})")

    # Convert to numpy for speed
    # Note: base_m5 and quote_m5 are ALREADY shifted (safe to use for features/triggers)
    # We need the RAW (unshifted) MFC for tracking future returns
    base_m5_arr = df['base_m5'].values  # shifted - for triggers/features
    quote_m5_arr = df['quote_m5'].values  # shifted - for triggers/features
    price_open_arr = df['price_open'].values
    price_close_arr = df['price_close'].values

    # Load raw MFC for exit tracking (need to track actual MFC, not shifted)
    base_raw_m5 = base_mfc['M5'].reindex(df.index)
    quote_raw_m5 = quote_mfc['M5'].reindex(df.index)
    base_raw_arr = base_raw_m5.values
    quote_raw_arr = quote_raw_m5.values

    all_entries = []

    # Process each trigger type
    for trigger_type, trigger_ccy in [('base', base_ccy), ('quote', quote_ccy)]:
        if trigger_type == 'base':
            trigger_arr = base_m5_arr   # shifted - for detecting triggers
            mfc_arr = base_raw_arr      # raw - for tracking exit
        else:
            trigger_arr = quote_m5_arr
            mfc_arr = quote_raw_arr

        # Find trigger masks (using shifted data for trigger detection)
        buy_mask = trigger_arr <= -MFC_EXTREME_THRESHOLD
        sell_mask = trigger_arr >= MFC_EXTREME_THRESHOLD

        if trigger_type == 'base':
            # base buy = base low, base sell = base high
            buy_indices = np.where(buy_mask)[0]
            sell_indices = np.where(sell_mask)[0]
        else:
            # quote low -> sell pair, quote high -> buy pair
            sell_indices = np.where(buy_mask)[0]  # quote low -> sell
            buy_indices = np.where(sell_mask)[0]  # quote high -> buy

        # Process buys
        if len(buy_indices) > 0:
            entries = process_entries_batch(
                df, buy_indices, mfc_arr, price_open_arr, price_close_arr,
                trigger_arr, direction=1, trigger_type=trigger_type,
                trigger_ccy=trigger_ccy, base_ccy=base_ccy, quote_ccy=quote_ccy,
                pair=pair, max_bars=MAX_BARS_TO_RETURN
            )
            if entries is not None:
                all_entries.append(entries)

        # Process sells
        if len(sell_indices) > 0:
            entries = process_entries_batch(
                df, sell_indices, mfc_arr, price_open_arr, price_close_arr,
                trigger_arr, direction=-1, trigger_type=trigger_type,
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
    output_path = OUTPUT_DIR / 'quality_entry_data_m5_v2.pkl'
    output = {
        'data': final_df,
        'config': {
            'mfc_extreme_threshold': MFC_EXTREME_THRESHOLD,
            'max_adverse_mfc': MAX_ADVERSE_MFC,
            'return_target': RETURN_TARGET,
            'max_bars_to_return': MAX_BARS_TO_RETURN,
            'base_timeframe': 'M5',
            'version': 'v2_both_currencies',
            'triggers': ['base', 'quote'],
        }
    }

    with open(output_path, 'wb') as f:
        pickle.dump(output, f)

    log(f"\nSaved to: {output_path}")
    log(f"Completed: {datetime.now()}")


if __name__ == '__main__':
    main()
