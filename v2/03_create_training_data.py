"""
V2: Create Training Data with Price-Based Targets
==================================================
Fast, memory-optimized version with pre-indexed lookups.
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
log("V2: CREATE TRAINING DATA (Optimized)")
log("=" * 70)
log(f"Started: {datetime.now()}")

# ============================================================================
# CONFIGURATION
# ============================================================================

V1_DATA_DIR = Path("/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/data")
CLEANED_DIR = V1_DATA_DIR / "cleaned"
V2_DIR = Path(__file__).parent
OUTPUT_DIR = V2_DIR / "data"
OUTPUT_DIR.mkdir(exist_ok=True)

CURRENCIES = ['EUR', 'USD', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD', 'NZD']

CURRENCY_PAIRS = {
    'EUR': ('EURUSD', 1),
    'USD': ('EURUSD', -1),
    'GBP': ('GBPUSD', 1),
    'JPY': ('USDJPY', -1),
    'CHF': ('USDCHF', -1),
    'CAD': ('USDCAD', -1),
    'AUD': ('AUDUSD', 1),
    'NZD': ('NZDUSD', 1),
}

HORIZON_BARS = 96
WIN_PIPS = 15
LOSS_PIPS = 15

LOOKBACKS = {'M5': 48, 'M15': 32, 'M30': 24, 'H1': 24, 'H4': 18}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_pip_value(pair):
    return 0.01 if 'JPY' in pair else 0.0001

def load_mfc(currency, timeframe):
    """Load MFC data."""
    if timeframe in ['M5', 'M15', 'M30']:
        path = V1_DATA_DIR / f"mfc_currency_{currency}_{timeframe}.csv"
    else:
        path = CLEANED_DIR / f"mfc_currency_{currency}_{timeframe}_clean.csv"

    df = pd.read_csv(path, parse_dates={'datetime': ['Date', 'Time']})
    df = df.set_index('datetime').sort_index()
    return df['MFC'].values, df.index.values

def load_price(pair):
    """Load M1 price data and resample to M5."""
    path = V1_DATA_DIR / f"{pair}_GMT+0_US-DST_M1.csv"
    df = pd.read_csv(path, usecols=['Date', 'Time', 'Close'],
                     parse_dates={'datetime': ['Date', 'Time']})
    df = df.set_index('datetime').sort_index()
    df_m5 = df.resample('5min').last().dropna()
    return df_m5['Close'].values, df_m5.index.values

def create_tf_index_map(base_times, tf_times, lookback):
    """
    Pre-compute index mapping from base timeframe to lower timeframe.
    Returns array where index_map[i] = end index in tf for base time i.
    Returns -1 if not enough history.
    """
    index_map = np.full(len(base_times), -1, dtype=np.int32)
    tf_idx = 0

    for i, bt in enumerate(base_times):
        # Find latest tf time <= base time
        while tf_idx < len(tf_times) - 1 and tf_times[tf_idx + 1] <= bt:
            tf_idx += 1

        # Check if we have enough history
        if tf_idx >= lookback - 1:
            index_map[i] = tf_idx

    return index_map

# ============================================================================
# PROCESS EACH CURRENCY
# ============================================================================

for ccy in CURRENCIES:
    log(f"\n{'='*50}")
    log(f"Processing {ccy}...")
    log(f"{'='*50}")

    pair, direction = CURRENCY_PAIRS[ccy]
    pip_value = get_pip_value(pair)

    # Load all MFC data
    log(f"   Loading MFC data...")
    mfc_m5, times_m5 = load_mfc(ccy, 'M5')
    mfc_m15, times_m15 = load_mfc(ccy, 'M15')
    mfc_m30, times_m30 = load_mfc(ccy, 'M30')
    mfc_h1, times_h1 = load_mfc(ccy, 'H1')
    mfc_h4, times_h4 = load_mfc(ccy, 'H4')

    log(f"   M5: {len(mfc_m5):,}, M15: {len(mfc_m15):,}, M30: {len(mfc_m30):,}, H1: {len(mfc_h1):,}, H4: {len(mfc_h4):,}")

    # Load price data
    log(f"   Loading {pair} price data...")
    price_vals, price_times = load_price(pair)
    log(f"   Price M5: {len(price_vals):,}")

    # Create index maps for each timeframe
    log(f"   Building index maps...")
    idx_m15 = create_tf_index_map(times_m5, times_m15, LOOKBACKS['M15'])
    idx_m30 = create_tf_index_map(times_m5, times_m30, LOOKBACKS['M30'])
    idx_h1 = create_tf_index_map(times_m5, times_h1, LOOKBACKS['H1'])
    idx_h4 = create_tf_index_map(times_m5, times_h4, LOOKBACKS['H4'])

    # Create price index map (M5 to price M5)
    idx_price = create_tf_index_map(times_m5, price_times, 1)

    # Valid samples: have all TF data and enough forward price data
    valid_start = LOOKBACKS['M5']
    valid_end = len(mfc_m5) - HORIZON_BARS

    # Pre-filter valid indices
    log(f"   Finding valid samples...")
    valid_mask = (
        (idx_m15[valid_start:valid_end] >= 0) &
        (idx_m30[valid_start:valid_end] >= 0) &
        (idx_h1[valid_start:valid_end] >= 0) &
        (idx_h4[valid_start:valid_end] >= 0) &
        (idx_price[valid_start:valid_end] >= 0) &
        (idx_price[valid_start:valid_end] + HORIZON_BARS < len(price_vals))
    )

    valid_indices = np.arange(valid_start, valid_end)[valid_mask]
    n_samples = len(valid_indices)
    log(f"   Valid samples: {n_samples:,}")

    if n_samples == 0:
        log(f"   WARNING: No valid samples for {ccy}")
        continue

    # Pre-allocate arrays
    X_M5 = np.zeros((n_samples, LOOKBACKS['M5']), dtype=np.float32)
    X_M15 = np.zeros((n_samples, LOOKBACKS['M15']), dtype=np.float32)
    X_M30 = np.zeros((n_samples, LOOKBACKS['M30']), dtype=np.float32)
    X_H1 = np.zeros((n_samples, LOOKBACKS['H1']), dtype=np.float32)
    X_H4 = np.zeros((n_samples, LOOKBACKS['H4']), dtype=np.float32)
    X_aux = np.zeros((n_samples, 5), dtype=np.float32)
    y_binary = np.zeros(n_samples, dtype=np.int8)
    y_multi = np.zeros(n_samples, dtype=np.int8)
    y_fav = np.zeros(n_samples, dtype=np.float32)
    y_adv = np.zeros(n_samples, dtype=np.float32)

    # Process samples
    log(f"   Processing {n_samples:,} samples...")

    for out_idx, i in enumerate(valid_indices):
        if out_idx % 100000 == 0:
            log(f"      Progress: {out_idx:,}/{n_samples:,} ({100*out_idx/n_samples:.1f}%)")

        # M5 sequence (direct slice)
        X_M5[out_idx] = mfc_m5[i-LOOKBACKS['M5']:i]

        # Other TF sequences (use pre-computed indices)
        j = idx_m15[i]
        X_M15[out_idx] = mfc_m15[j-LOOKBACKS['M15']+1:j+1]

        j = idx_m30[i]
        X_M30[out_idx] = mfc_m30[j-LOOKBACKS['M30']+1:j+1]

        j = idx_h1[i]
        X_H1[out_idx] = mfc_h1[j-LOOKBACKS['H1']+1:j+1]

        j = idx_h4[i]
        X_H4[out_idx] = mfc_h4[j-LOOKBACKS['H4']+1:j+1]

        # Price targets
        p_idx = idx_price[i]
        start_price = price_vals[p_idx]
        future_prices = price_vals[p_idx:p_idx+HORIZON_BARS]

        if direction == 1:
            max_fav = (future_prices.max() - start_price) / pip_value
            max_adv = (start_price - future_prices.min()) / pip_value
        else:
            max_fav = (start_price - future_prices.min()) / pip_value
            max_adv = (future_prices.max() - start_price) / pip_value

        y_fav[out_idx] = max_fav
        y_adv[out_idx] = max_adv
        y_binary[out_idx] = 1 if max_fav >= WIN_PIPS else 0

        if max_fav >= WIN_PIPS:
            y_multi[out_idx] = 2
        elif max_adv >= LOSS_PIPS:
            y_multi[out_idx] = 0
        else:
            y_multi[out_idx] = 1

        # Aux features
        mfc_curr = mfc_m5[i-1]
        mfc_prev = mfc_m5[i-2]
        X_aux[out_idx] = [
            mfc_curr,
            mfc_curr - mfc_prev,
            abs(mfc_curr),
            1 if -0.2 <= mfc_curr <= 0.2 else 0,
            direction
        ]

    log(f"   Processing complete!")

    # Get datetimes for valid samples
    datetimes = times_m5[valid_indices]

    # Stats
    win_pct = y_binary.mean() * 100
    multi_dist = np.bincount(y_multi, minlength=3)
    log(f"   Binary: {win_pct:.1f}% wins")
    log(f"   Multiclass: Loss={multi_dist[0]:,}, Flat={multi_dist[1]:,}, Win={multi_dist[2]:,}")

    # Save
    data = {
        'X_M5': X_M5,
        'X_M15': X_M15,
        'X_M30': X_M30,
        'X_H1': X_H1,
        'X_H4': X_H4,
        'X_aux': X_aux,
        'y_binary': y_binary,
        'y_multiclass': y_multi,
        'y_max_favorable': y_fav,
        'y_max_adverse': y_adv,
        'datetimes': datetimes,
    }

    output_path = OUTPUT_DIR / f"v2_data_{ccy}.pkl"
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    log(f"   Saved: {output_path.name}")

    # Free memory
    del X_M5, X_M15, X_M30, X_H1, X_H4, X_aux
    del y_binary, y_multi, y_fav, y_adv
    del mfc_m5, mfc_m15, mfc_m30, mfc_h1, mfc_h4
    del price_vals, data
    gc.collect()

# Save config
config = {
    'lookbacks': LOOKBACKS,
    'horizon_bars': HORIZON_BARS,
    'win_pips': WIN_PIPS,
    'loss_pips': LOSS_PIPS,
    'currency_pairs': CURRENCY_PAIRS,
}
with open(OUTPUT_DIR / 'config_v2.pkl', 'wb') as f:
    pickle.dump(config, f)

log(f"\n{'='*70}")
log(f"Completed: {datetime.now()}")
log(f"{'='*70}")
