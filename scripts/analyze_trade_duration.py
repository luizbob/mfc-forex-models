"""
Analyze Trade Duration
======================
Calculate how long it takes for MFC to return to center.
"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import pickle
from pathlib import Path

DATA_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/data')
LSTM_DATA_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/lstm_model/data')

CURRENCIES = ['EUR', 'USD', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD', 'NZD']
MFC_EXTREME = 0.5
MAX_BARS = 200

def load_mfc_data(currency):
    """Load H1 MFC data for a currency."""
    fp = DATA_DIR / f'mfc_currency_{currency}.csv'
    if not fp.exists():
        return None
    df = pd.read_csv(fp)
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df = df.set_index('datetime').sort_index()
    return df['MFC']

print("=" * 60)
print("TRADE DURATION ANALYSIS")
print("=" * 60)

# Analyze for each currency
all_durations = []

for ccy in CURRENCIES:
    print(f"\nAnalyzing {ccy}...")
    mfc = load_mfc_data(ccy)
    if mfc is None:
        continue

    # Apply shift
    mfc_shifted = mfc.shift(1)

    # Find entries
    buy_entries = mfc_shifted[mfc_shifted <= -MFC_EXTREME].index
    sell_entries = mfc_shifted[mfc_shifted >= MFC_EXTREME].index

    durations = []

    # BUY entries - time to return to 0
    for entry_time in buy_entries:
        try:
            pos = mfc.index.get_loc(entry_time)
            if pos + MAX_BARS >= len(mfc):
                continue

            future_mfc = mfc.iloc[pos+1:pos+MAX_BARS+1]

            # Find first bar where MFC >= 0
            returned = future_mfc >= 0
            if returned.any():
                first_return = returned.idxmax()
                bars_to_return = mfc.index.get_loc(first_return) - pos
                durations.append(bars_to_return)
        except:
            pass

    # SELL entries - time to return to 0
    for entry_time in sell_entries:
        try:
            pos = mfc.index.get_loc(entry_time)
            if pos + MAX_BARS >= len(mfc):
                continue

            future_mfc = mfc.iloc[pos+1:pos+MAX_BARS+1]

            # Find first bar where MFC <= 0
            returned = future_mfc <= 0
            if returned.any():
                first_return = returned.idxmax()
                bars_to_return = mfc.index.get_loc(first_return) - pos
                durations.append(bars_to_return)
        except:
            pass

    if durations:
        all_durations.extend(durations)
        print(f"  Entries: {len(durations)}")
        print(f"  Median: {np.median(durations):.0f} H1 bars ({np.median(durations):.0f} hours)")
        print(f"  Mean: {np.mean(durations):.1f} H1 bars")
        print(f"  Min: {np.min(durations)} bars, Max: {np.max(durations)} bars")

# Overall statistics
print("\n" + "=" * 60)
print("OVERALL STATISTICS")
print("=" * 60)

durations = np.array(all_durations)
print(f"\nTotal entries analyzed: {len(durations)}")
print(f"\nTime to MFC return to center (0):")
print(f"  Median: {np.median(durations):.0f} hours")
print(f"  Mean: {np.mean(durations):.1f} hours")
print(f"  Std: {np.std(durations):.1f} hours")

print(f"\nPercentiles:")
for p in [10, 25, 50, 75, 90, 95]:
    val = np.percentile(durations, p)
    print(f"  {p}th: {val:.0f} hours ({val/24:.1f} days)")

print(f"\nDistribution:")
bins = [0, 6, 12, 24, 48, 72, 100, 150, 200]
for i in range(len(bins)-1):
    count = ((durations >= bins[i]) & (durations < bins[i+1])).sum()
    pct = count / len(durations) * 100
    print(f"  {bins[i]:3d}-{bins[i+1]:3d}h: {count:6d} ({pct:5.1f}%)")

count_200 = (durations >= 200).sum()
print(f"  200h+:   {count_200:6d} ({count_200/len(durations)*100:5.1f}%)")
