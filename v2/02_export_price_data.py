"""
V2: Export Price Data from MT5
==============================
Downloads historical price data for all pairs needed for price-based targets.

Run this on Windows where MT5 is installed.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime, timedelta

def log(msg=""):
    print(msg, flush=True)

log("=" * 70)
log("V2: EXPORT PRICE DATA FROM MT5")
log("=" * 70)

# ============================================================================
# CONFIGURATION
# ============================================================================

V2_DIR = Path(__file__).parent
DATA_DIR = V2_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# Pairs to export (all pairs we might need)
PAIRS = [
    'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'USDCAD',
    'AUDUSD', 'NZDUSD', 'EURGBP', 'EURJPY', 'GBPJPY',
    'AUDJPY', 'NZDJPY', 'CHFJPY', 'CADJPY',
    'AUDNZD', 'AUDCHF', 'NZDCHF', 'AUDCAD', 'GBPCHF',
]

SYMBOL_SUFFIX = "m"  # Adjust based on your broker

# Date range - USE ALL AVAILABLE DATA
START_DATE = datetime(2013, 1, 1)
END_DATE = datetime(2025, 12, 31)

# ============================================================================
# CONNECT TO MT5
# ============================================================================

log("\n1. Connecting to MT5...")

if not mt5.initialize():
    log(f"   ERROR: MT5 initialize failed: {mt5.last_error()}")
    log("   Make sure MT5 is running.")
    exit(1)

account_info = mt5.account_info()
log(f"   Connected to: {account_info.server}")
log(f"   Account: {account_info.login}")

# ============================================================================
# EXPORT PRICE DATA
# ============================================================================

log(f"\n2. Exporting price data from {START_DATE.date()} to {END_DATE.date()}...")

price_data = {}

for pair in PAIRS:
    symbol = pair + SYMBOL_SUFFIX

    # Enable symbol
    if not mt5.symbol_select(symbol, True):
        log(f"   {pair}: Symbol not found, skipping")
        continue

    # Download M5 data
    rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M5, START_DATE, END_DATE)

    if rates is None or len(rates) == 0:
        log(f"   {pair}: No data available")
        continue

    # Convert to DataFrame
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df = df.set_index('time')

    price_data[pair] = df

    log(f"   {pair}: {len(df):,} bars from {df.index[0]} to {df.index[-1]}")

# ============================================================================
# SAVE DATA
# ============================================================================

log(f"\n3. Saving price data...")

# Save as pickle (fast loading)
output_path = DATA_DIR / "price_data_m5.pkl"
with open(output_path, 'wb') as f:
    pickle.dump(price_data, f)
log(f"   Saved: {output_path}")

# Also save as CSV for inspection
csv_dir = DATA_DIR / "price_csv"
csv_dir.mkdir(exist_ok=True)

for pair, df in price_data.items():
    csv_path = csv_dir / f"{pair}_M5.csv"
    df.to_csv(csv_path)
log(f"   CSV files saved to: {csv_dir}")

# ============================================================================
# SUMMARY
# ============================================================================

log(f"\n" + "=" * 70)
log("SUMMARY")
log("=" * 70)

total_bars = sum(len(df) for df in price_data.values())
log(f"\nPairs exported: {len(price_data)}")
log(f"Total bars: {total_bars:,}")
log(f"Date range: {START_DATE.date()} to {END_DATE.date()}")

log(f"\nPairs available:")
for pair in sorted(price_data.keys()):
    df = price_data[pair]
    log(f"   {pair}: {len(df):,} bars")

# Shutdown MT5
mt5.shutdown()
log(f"\nCompleted: {datetime.now()}")
