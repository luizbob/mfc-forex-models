"""
V2 Model: Price-Based Target Data Preparation
==============================================
Prepares training data with price movement targets instead of MFC cycle.

Target: Will price move > X pips in direction within 8 hours?

Creates both:
- Binary target: 1 = profitable, 0 = not profitable
- Multiclass target: 0 = loss, 1 = flat, 2 = win
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime, timedelta

def log(msg=""):
    print(msg, flush=True)

log("=" * 70)
log("V2: PRICE-BASED TARGET DATA PREPARATION")
log("=" * 70)
log(f"Started: {datetime.now()}")

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
V2_DIR = Path(__file__).parent
OUTPUT_DIR = V2_DIR / "data"
OUTPUT_DIR.mkdir(exist_ok=True)

# Target parameters
HORIZON_HOURS = 8           # How far ahead to look for price movement
HORIZON_BARS_M5 = HORIZON_HOURS * 12  # 8 hours = 96 M5 bars

# Pip thresholds
WIN_THRESHOLD_PIPS = 15     # Minimum pips to count as "win"
LOSS_THRESHOLD_PIPS = -15   # Maximum pips to count as "loss"

# LSTM lookbacks (same as v1)
LOOKBACK = {'M5': 48, 'M15': 32, 'M30': 24, 'H1': 12, 'H4': 6}

# Currencies
CURRENCIES = ['EUR', 'USD', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD', 'NZD']

# Pairs for price data (need to map currency to pairs)
CURRENCY_PAIRS = {
    'EUR': 'EURUSD',
    'USD': 'EURUSD',  # Inverted
    'GBP': 'GBPUSD',
    'JPY': 'USDJPY',  # Inverted
    'CHF': 'USDCHF',  # Inverted
    'CAD': 'USDCAD',  # Inverted
    'AUD': 'AUDUSD',
    'NZD': 'NZDUSD',
}

# ============================================================================
# LOAD RAW DATA
# ============================================================================

log("\n1. Loading raw data...")

# Load MFC data (from v1)
mfc_data = {}
for ccy in CURRENCIES:
    mfc_path = DATA_DIR / f"mfc_{ccy}_all_timeframes.pkl"
    if mfc_path.exists():
        with open(mfc_path, 'rb') as f:
            mfc_data[ccy] = pickle.load(f)
        log(f"   {ccy}: Loaded MFC data")
    else:
        log(f"   {ccy}: MFC file not found at {mfc_path}")

# Load price data
price_data = {}
price_dir = DATA_DIR / "price"

if not price_dir.exists():
    log(f"\n   Price directory not found: {price_dir}")
    log("   We need to create price data from MT5 or CSV files.")
    log("\n   Looking for alternative price sources...")

    # Try to find price data in other locations
    alt_price_paths = [
        DATA_DIR / "prices",
        DATA_DIR.parent / "price_data",
        Path("/mnt/c/Users/luizh/Documents/mt4/tryea/data"),
    ]

    for alt_path in alt_price_paths:
        if alt_path.exists():
            log(f"   Found: {alt_path}")
            price_dir = alt_path
            break

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def calculate_pip_change(prices, start_idx, end_idx, pair):
    """Calculate pip change between two indices."""
    if end_idx >= len(prices):
        return None

    start_price = prices[start_idx]
    end_price = prices[end_idx]

    # Pip calculation depends on pair
    if 'JPY' in pair:
        pip_value = 0.01
    else:
        pip_value = 0.0001

    pip_change = (end_price - start_price) / pip_value
    return pip_change


def get_max_favorable_move(prices, start_idx, end_idx, direction, pair):
    """
    Get maximum favorable price move within window.
    direction: 1 for long (want price up), -1 for short (want price down)
    """
    if end_idx >= len(prices):
        return None

    start_price = prices[start_idx]
    window_prices = prices[start_idx:end_idx+1]

    if 'JPY' in pair:
        pip_value = 0.01
    else:
        pip_value = 0.0001

    if direction == 1:  # Long - want max price
        best_price = max(window_prices)
    else:  # Short - want min price
        best_price = min(window_prices)

    pip_move = (best_price - start_price) / pip_value * direction
    return pip_move


def create_target_binary(max_favorable_pips, threshold=WIN_THRESHOLD_PIPS):
    """Binary target: 1 if price moved favorably by threshold."""
    if max_favorable_pips is None:
        return None
    return 1 if max_favorable_pips >= threshold else 0


def create_target_multiclass(max_favorable_pips, final_pips,
                             win_threshold=WIN_THRESHOLD_PIPS,
                             loss_threshold=LOSS_THRESHOLD_PIPS):
    """
    Multiclass target:
    0 = Loss (final pips < loss_threshold)
    1 = Flat (between thresholds)
    2 = Win (max favorable >= win_threshold)
    """
    if max_favorable_pips is None or final_pips is None:
        return None

    if max_favorable_pips >= win_threshold:
        return 2  # Win - could have taken profit
    elif final_pips <= loss_threshold:
        return 0  # Loss
    else:
        return 1  # Flat


# ============================================================================
# CHECK WHAT DATA WE HAVE
# ============================================================================

log("\n2. Checking available data...")

# Check MFC data structure
if mfc_data:
    sample_ccy = list(mfc_data.keys())[0]
    sample_data = mfc_data[sample_ccy]
    log(f"\n   Sample MFC data structure for {sample_ccy}:")
    if isinstance(sample_data, dict):
        for key in list(sample_data.keys())[:5]:
            log(f"      {key}: {type(sample_data[key])}")
    elif isinstance(sample_data, pd.DataFrame):
        log(f"      DataFrame shape: {sample_data.shape}")
        log(f"      Columns: {list(sample_data.columns)[:10]}")

# List files in data directory
log(f"\n   Files in {DATA_DIR}:")
for f in sorted(DATA_DIR.glob("*")):
    if f.is_file():
        size = f.stat().st_size / 1024
        log(f"      {f.name}: {size:.1f} KB")
    else:
        log(f"      {f.name}/ (directory)")

# ============================================================================
# CREATE SAMPLE OUTPUT STRUCTURE
# ============================================================================

log("\n3. Output structure for V2 data:")
log("""
   For each currency, we'll create:

   X (inputs) - Same as V1:
   - X_M5:  (n_samples, 48, 1) - M5 MFC sequence
   - X_M15: (n_samples, 32, 1) - M15 MFC sequence
   - X_M30: (n_samples, 24, 1) - M30 MFC sequence
   - X_H1:  (n_samples, 12, 1) - H1 MFC sequence
   - X_H4:  (n_samples, 6, 1)  - H4 MFC sequence
   - X_aux: (n_samples, 5)     - Auxiliary features

   Y (targets) - NEW:
   - y_binary:     (n_samples,) - 0/1 profitable or not
   - y_multiclass: (n_samples,) - 0=loss, 1=flat, 2=win
   - y_pips:       (n_samples,) - Actual pip change (for analysis)

   Target is based on:
   - 8 hour forward window
   - Maximum favorable move within window
   - Threshold: 15 pips
""")

# ============================================================================
# NEXT STEPS
# ============================================================================

log("\n" + "=" * 70)
log("NEXT STEPS")
log("=" * 70)
log("""
To complete data preparation, we need:

1. Price data for each currency pair (M5 timeframe)
   - Either from MT5 export
   - Or from existing CSV files

2. Align MFC data with price data by timestamp

3. Calculate forward-looking price targets

Options to get price data:
a) Export from MT5 using Python script
b) Use existing price CSVs if available
c) Export from MT4 (add to existing exporter)

""")

# Check if we have the v1 LSTM data which might have timestamps
v1_data_files = list(DATA_DIR.glob("lstm_data_*.pkl"))
if v1_data_files:
    log(f"Found V1 LSTM data files: {[f.name for f in v1_data_files]}")

    # Load one to check structure
    with open(v1_data_files[0], 'rb') as f:
        v1_sample = pickle.load(f)

    log(f"\nV1 data structure:")
    for key in v1_sample.keys():
        val = v1_sample[key]
        if isinstance(val, np.ndarray):
            log(f"   {key}: shape={val.shape}, dtype={val.dtype}")
        else:
            log(f"   {key}: {type(val)}")

log(f"\nCompleted: {datetime.now()}")
