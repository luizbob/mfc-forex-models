"""
V2: Create Per-Pair Training Data with Price-Based Targets
==========================================================
Creates training data for each pair using:
- MFC data from BOTH currencies in the pair
- Price movement of that SPECIFIC pair as target

Target considers BOTH directions:
- LONG opportunity: price moves UP ≥15 pips first
- SHORT opportunity: price moves DOWN ≥15 pips first
- NEUTRAL: neither threshold hit within 8h
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime
import gc

def log(msg=""):
    print(msg, flush=True)

log("=" * 70)
log("V2: CREATE PER-PAIR TRAINING DATA (Both Directions)")
log("=" * 70)
log(f"Started: {datetime.now()}")

# ============================================================================
# CONFIGURATION
# ============================================================================

V1_DATA_DIR = Path("/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/data")
CLEANED_DIR = V1_DATA_DIR / "cleaned"
V2_DIR = Path(__file__).parent
OUTPUT_DIR = V2_DIR / "data" / "pairs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# All 28 pairs from 8 currencies
PAIRS = [
    'EURUSD', 'EURGBP', 'EURJPY', 'EURCHF', 'EURCAD', 'EURAUD', 'EURNZD',
    'GBPUSD', 'GBPJPY', 'GBPCHF', 'GBPCAD', 'GBPAUD', 'GBPNZD',
    'USDJPY', 'USDCHF', 'USDCAD',
    'AUDUSD', 'AUDJPY', 'AUDCHF', 'AUDCAD', 'AUDNZD',
    'NZDUSD', 'NZDJPY', 'NZDCHF', 'NZDCAD',
    'CADJPY', 'CADCHF',
    'CHFJPY',
]

# Target parameters
HORIZON_BARS = 96  # 8 hours in M5 bars
TP_PIPS = 15  # Take profit threshold

# LSTM lookbacks
LOOKBACKS = {'M5': 48, 'M15': 32, 'M30': 24, 'H1': 24, 'H4': 18}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_currencies(pair):
    """Extract base and quote currencies from pair."""
    return pair[:3], pair[3:]

def get_pip_value(pair):
    """Get pip value for a pair."""
    return 0.01 if 'JPY' in pair else 0.0001

def load_mfc(currency, timeframe, apply_shift=True):
    """Load MFC data for a currency.

    apply_shift: If True, apply shift(1) so we only use CLOSED bar data.
                 This prevents lookahead bias - at decision time we can
                 only see data from closed bars, not the forming bar.
    """
    if timeframe in ['M5', 'M15', 'M30']:
        path = V1_DATA_DIR / f"mfc_currency_{currency}_{timeframe}.csv"
    else:
        path = CLEANED_DIR / f"mfc_currency_{currency}_{timeframe}_clean.csv"

    if not path.exists():
        return None, None

    df = pd.read_csv(path)
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df = df.set_index('datetime').sort_index()

    if apply_shift:
        # Shift MFC values forward by 1 - so at time T, we see the value from T-1
        # This matches V1 training which used shift(1)
        df['MFC'] = df['MFC'].shift(1)
        df = df.dropna()  # First row becomes NaN after shift

    return df['MFC'].values, df.index.values

def load_price(pair):
    """Load M1 price data and resample to M5."""
    path = V1_DATA_DIR / f"{pair}_GMT+0_US-DST_M1.csv"

    if not path.exists():
        return None, None

    df = pd.read_csv(path, usecols=['Date', 'Time', 'Close'])
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df = df.set_index('datetime').sort_index()
    df_m5 = df.resample('5min').last().dropna()
    return df_m5['Close'].values, df_m5.index.values

def create_tf_index_map(base_times, tf_times, lookback):
    """Pre-compute index mapping from base timeframe to lower timeframe."""
    index_map = np.full(len(base_times), -1, dtype=np.int32)
    tf_idx = 0

    for i, bt in enumerate(base_times):
        while tf_idx < len(tf_times) - 1 and tf_times[tf_idx + 1] <= bt:
            tf_idx += 1
        if tf_idx >= lookback - 1:
            index_map[i] = tf_idx

    return index_map

def calculate_direction_target(future_prices, start_price, pip_value, tp_pips):
    """
    Calculate which direction would win first.

    Returns:
        direction: 0=SHORT, 1=NEUTRAL, 2=LONG
        long_pips: max upward move in pips
        short_pips: max downward move in pips (positive number)
        first_tp_bar: bar number when first TP was hit (-1 if never)
    """
    pip_changes = (future_prices - start_price) / pip_value

    # Find first bar where each threshold is hit
    long_tp_bar = -1
    short_tp_bar = -1

    for bar_idx, pip_change in enumerate(pip_changes):
        if pip_change >= tp_pips and long_tp_bar == -1:
            long_tp_bar = bar_idx
        if pip_change <= -tp_pips and short_tp_bar == -1:
            short_tp_bar = bar_idx

        # Stop if both found
        if long_tp_bar >= 0 and short_tp_bar >= 0:
            break

    # Determine direction based on which hit first
    if long_tp_bar >= 0 and short_tp_bar >= 0:
        # Both hit - which was first?
        if long_tp_bar < short_tp_bar:
            direction = 2  # LONG
            first_tp_bar = long_tp_bar
        else:
            direction = 0  # SHORT
            first_tp_bar = short_tp_bar
    elif long_tp_bar >= 0:
        direction = 2  # LONG only
        first_tp_bar = long_tp_bar
    elif short_tp_bar >= 0:
        direction = 0  # SHORT only
        first_tp_bar = short_tp_bar
    else:
        direction = 1  # NEUTRAL
        first_tp_bar = -1

    # Max moves for analysis
    long_pips = pip_changes.max()
    short_pips = -pip_changes.min()  # Positive number

    return direction, long_pips, short_pips, first_tp_bar

# ============================================================================
# PROCESS EACH PAIR
# ============================================================================

log(f"\nProcessing {len(PAIRS)} pairs...")

summary = []

for pair in PAIRS:
    log(f"\n{'='*50}")
    log(f"Processing {pair}...")
    log(f"{'='*50}")

    base_ccy, quote_ccy = get_currencies(pair)
    pip_value = get_pip_value(pair)
    log(f"   Base: {base_ccy}, Quote: {quote_ccy}")

    # Load MFC data for both currencies
    log(f"   Loading MFC data...")

    mfc_data = {}
    times_data = {}

    for ccy in [base_ccy, quote_ccy]:
        mfc_data[ccy] = {}
        times_data[ccy] = {}
        for tf in LOOKBACKS.keys():
            mfc, times = load_mfc(ccy, tf)
            if mfc is None:
                log(f"   WARNING: Missing {ccy} {tf} MFC data")
                break
            mfc_data[ccy][tf] = mfc
            times_data[ccy][tf] = times

    # Check we have all data
    if len(mfc_data.get(base_ccy, {})) != 5 or len(mfc_data.get(quote_ccy, {})) != 5:
        log(f"   SKIPPING: Missing MFC data")
        continue

    # Load price data for this specific pair
    log(f"   Loading {pair} price data...")
    price_vals, price_times = load_price(pair)

    if price_vals is None:
        log(f"   SKIPPING: No price data for {pair}")
        continue

    log(f"   Price M5: {len(price_vals):,}")

    # Use base currency M5 as the primary timeline
    mfc_m5_base = mfc_data[base_ccy]['M5']
    times_m5 = times_data[base_ccy]['M5']

    log(f"   Base M5: {len(mfc_m5_base):,}")

    # Build index maps for all timeframes of both currencies
    log(f"   Building index maps...")

    idx_maps = {}
    for ccy in [base_ccy, quote_ccy]:
        idx_maps[ccy] = {}
        for tf in LOOKBACKS.keys():
            if tf == 'M5' and ccy == base_ccy:
                idx_maps[ccy][tf] = np.arange(len(times_m5), dtype=np.int32)
            else:
                idx_maps[ccy][tf] = create_tf_index_map(
                    times_m5, times_data[ccy][tf], LOOKBACKS[tf]
                )

    # Price index map
    idx_price = create_tf_index_map(times_m5, price_times, 1)

    # Find valid samples
    log(f"   Finding valid samples...")
    max_lookback = max(LOOKBACKS.values())
    valid_start = max_lookback
    valid_end = len(mfc_m5_base) - HORIZON_BARS

    # Build validity mask
    valid_mask = np.ones(valid_end - valid_start, dtype=bool)

    for ccy in [base_ccy, quote_ccy]:
        for tf in LOOKBACKS.keys():
            valid_mask &= (idx_maps[ccy][tf][valid_start:valid_end] >= 0)

    valid_mask &= (idx_price[valid_start:valid_end] >= 0)
    valid_mask &= (idx_price[valid_start:valid_end] + HORIZON_BARS < len(price_vals))

    valid_indices = np.arange(valid_start, valid_end)[valid_mask]
    n_samples = len(valid_indices)
    log(f"   Valid samples: {n_samples:,}")

    if n_samples == 0:
        log(f"   SKIPPING: No valid samples")
        continue

    # Pre-allocate arrays
    X_base = {tf: np.zeros((n_samples, LOOKBACKS[tf]), dtype=np.float32) for tf in LOOKBACKS}
    X_quote = {tf: np.zeros((n_samples, LOOKBACKS[tf]), dtype=np.float32) for tf in LOOKBACKS}
    X_aux = np.zeros((n_samples, 6), dtype=np.float32)

    y_direction = np.zeros(n_samples, dtype=np.int8)  # 0=SHORT, 1=NEUTRAL, 2=LONG
    y_long_pips = np.zeros(n_samples, dtype=np.float32)
    y_short_pips = np.zeros(n_samples, dtype=np.float32)
    y_first_tp_bar = np.zeros(n_samples, dtype=np.int16)

    # Process samples
    log(f"   Processing {n_samples:,} samples...")

    for out_idx, i in enumerate(valid_indices):
        if out_idx % 100000 == 0:
            log(f"      Progress: {out_idx:,}/{n_samples:,} ({100*out_idx/n_samples:.1f}%)")

        # Base currency sequences (MFC data already shifted at load time)
        for tf, lookback in LOOKBACKS.items():
            j = idx_maps[base_ccy][tf][i]
            X_base[tf][out_idx] = mfc_data[base_ccy][tf][j-lookback+1:j+1]

        # Quote currency sequences
        for tf, lookback in LOOKBACKS.items():
            j = idx_maps[quote_ccy][tf][i]
            X_quote[tf][out_idx] = mfc_data[quote_ccy][tf][j-lookback+1:j+1]

        # Price targets (for THIS pair, BOTH directions)
        p_idx = idx_price[i]
        start_price = price_vals[p_idx]
        future_prices = price_vals[p_idx:p_idx+HORIZON_BARS]

        direction, long_pips, short_pips, first_tp_bar = calculate_direction_target(
            future_prices, start_price, pip_value, TP_PIPS
        )

        y_direction[out_idx] = direction
        y_long_pips[out_idx] = long_pips
        y_short_pips[out_idx] = short_pips
        y_first_tp_bar[out_idx] = first_tp_bar

        # Aux features (MFC data already shifted at load time)
        j_base = idx_maps[base_ccy]['M5'][i]
        j_quote = idx_maps[quote_ccy]['M5'][i]
        mfc_base_curr = mfc_data[base_ccy]['M5'][j_base]
        mfc_quote_curr = mfc_data[quote_ccy]['M5'][j_quote]
        mfc_diff = mfc_base_curr - mfc_quote_curr

        X_aux[out_idx] = [
            mfc_base_curr,
            mfc_quote_curr,
            mfc_diff,
            abs(mfc_diff),
            1 if mfc_base_curr > 0.2 else 0,
            1 if mfc_quote_curr < -0.2 else 0,
        ]

    log(f"   Processing complete!")

    # Get datetimes
    datetimes = times_m5[valid_indices]

    # Stats
    dir_counts = np.bincount(y_direction, minlength=3)
    short_pct = 100 * dir_counts[0] / n_samples
    neutral_pct = 100 * dir_counts[1] / n_samples
    long_pct = 100 * dir_counts[2] / n_samples

    log(f"   Direction: SHORT={dir_counts[0]:,} ({short_pct:.1f}%), "
        f"NEUTRAL={dir_counts[1]:,} ({neutral_pct:.1f}%), "
        f"LONG={dir_counts[2]:,} ({long_pct:.1f}%)")

    summary.append({
        'pair': pair,
        'samples': n_samples,
        'short': dir_counts[0],
        'neutral': dir_counts[1],
        'long': dir_counts[2],
        'tradeable_pct': short_pct + long_pct,
    })

    # Save
    data = {
        'X_base_M5': X_base['M5'],
        'X_base_M15': X_base['M15'],
        'X_base_M30': X_base['M30'],
        'X_base_H1': X_base['H1'],
        'X_base_H4': X_base['H4'],
        'X_quote_M5': X_quote['M5'],
        'X_quote_M15': X_quote['M15'],
        'X_quote_M30': X_quote['M30'],
        'X_quote_H1': X_quote['H1'],
        'X_quote_H4': X_quote['H4'],
        'X_aux': X_aux,
        'y_direction': y_direction,  # 0=SHORT, 1=NEUTRAL, 2=LONG
        'y_long_pips': y_long_pips,
        'y_short_pips': y_short_pips,
        'y_first_tp_bar': y_first_tp_bar,
        'datetimes': datetimes,
        'base_currency': base_ccy,
        'quote_currency': quote_ccy,
    }

    output_path = OUTPUT_DIR / f"v2_pair_{pair}.pkl"
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    log(f"   Saved: {output_path.name}")

    # Free memory
    del X_base, X_quote, X_aux
    del y_direction, y_long_pips, y_short_pips, y_first_tp_bar
    del mfc_data, times_data, price_vals
    gc.collect()

# Save config
config = {
    'pairs': PAIRS,
    'lookbacks': LOOKBACKS,
    'horizon_bars': HORIZON_BARS,
    'tp_pips': TP_PIPS,
    'direction_labels': {0: 'SHORT', 1: 'NEUTRAL', 2: 'LONG'},
}
with open(OUTPUT_DIR / 'config_pairs.pkl', 'wb') as f:
    pickle.dump(config, f)

# Print summary
log(f"\n{'='*70}")
log("SUMMARY")
log(f"{'='*70}")
log(f"\n{'Pair':<10} {'Samples':>10} {'SHORT':>8} {'NEUTRAL':>8} {'LONG':>8} {'Trade%':>8}")
log("-" * 60)
for s in summary:
    log(f"{s['pair']:<10} {s['samples']:>10,} {s['short']:>8,} {s['neutral']:>8,} {s['long']:>8,} {s['tradeable_pct']:>7.1f}%")

log(f"\n{'='*70}")
log(f"Completed: {datetime.now()}")
log(f"{'='*70}")
