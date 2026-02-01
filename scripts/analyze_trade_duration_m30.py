"""
Analyze M30 Trade Duration
==========================
Calculate how long it takes for MFC to return to center on M30.
"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import pickle
from pathlib import Path

DATA_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/data')

CURRENCIES = ['EUR', 'USD', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD', 'NZD']
MFC_EXTREME = 0.5
MAX_BARS = 400

def load_mfc_data(currency):
    """Load M30 MFC data for a currency."""
    fp = DATA_DIR / f'mfc_currency_{currency}_M30.csv'
    if not fp.exists():
        return None
    df = pd.read_csv(fp)
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df = df.set_index('datetime').sort_index()
    return df['MFC']

print("=" * 60)
print("M30 TRADE DURATION ANALYSIS")
print("=" * 60)

all_durations = []

for ccy in CURRENCIES:
    print(f"\nAnalyzing {ccy}...")
    mfc = load_mfc_data(ccy)
    if mfc is None:
        continue

    mfc_shifted = mfc.shift(1)

    buy_entries = mfc_shifted[mfc_shifted <= -MFC_EXTREME].index
    sell_entries = mfc_shifted[mfc_shifted >= MFC_EXTREME].index

    durations = []

    # BUY entries
    for entry_time in buy_entries:
        try:
            pos = mfc.index.get_loc(entry_time)
            if pos + MAX_BARS >= len(mfc):
                continue

            future_mfc = mfc.iloc[pos+1:pos+MAX_BARS+1]
            returned = future_mfc >= 0
            if returned.any():
                first_return = returned.idxmax()
                bars_to_return = mfc.index.get_loc(first_return) - pos
                durations.append(bars_to_return)
        except:
            pass

    # SELL entries
    for entry_time in sell_entries:
        try:
            pos = mfc.index.get_loc(entry_time)
            if pos + MAX_BARS >= len(mfc):
                continue

            future_mfc = mfc.iloc[pos+1:pos+MAX_BARS+1]
            returned = future_mfc <= 0
            if returned.any():
                first_return = returned.idxmax()
                bars_to_return = mfc.index.get_loc(first_return) - pos
                durations.append(bars_to_return)
        except:
            pass

    if durations:
        all_durations.extend(durations)
        hours = np.array(durations) * 0.5  # M30 bars to hours
        print(f"  Entries: {len(durations)}")
        print(f"  Median: {np.median(durations):.0f} M30 bars ({np.median(hours):.1f} hours)")
        print(f"  Mean: {np.mean(durations):.1f} M30 bars ({np.mean(hours):.1f} hours)")

# Overall statistics
print("\n" + "=" * 60)
print("OVERALL STATISTICS")
print("=" * 60)

durations = np.array(all_durations)
hours = durations * 0.5  # M30 bars to hours

print(f"\nTotal entries analyzed: {len(durations)}")
print(f"\nTime to MFC return to center (0):")
print(f"  Median: {np.median(durations):.0f} M30 bars = {np.median(hours):.1f} hours")
print(f"  Mean: {np.mean(durations):.1f} M30 bars = {np.mean(hours):.1f} hours")

print(f"\nPercentiles (in hours):")
for p in [10, 25, 50, 75, 90, 95]:
    val_bars = np.percentile(durations, p)
    val_hours = val_bars * 0.5
    print(f"  {p}th: {val_bars:.0f} bars = {val_hours:.1f} hours ({val_hours/24:.1f} days)")

print(f"\nDistribution (hours):")
# Bins in M30 bars, convert to hours for display
hour_bins = [0, 3, 6, 12, 24, 48, 72, 100]
bar_bins = [h * 2 for h in hour_bins]  # Convert hours to M30 bars

for i in range(len(bar_bins)-1):
    count = ((durations >= bar_bins[i]) & (durations < bar_bins[i+1])).sum()
    pct = count / len(durations) * 100
    print(f"  {hour_bins[i]:3d}-{hour_bins[i+1]:3d}h: {count:6d} ({pct:5.1f}%)")

count_100h = (durations >= 200).sum()  # 200 M30 bars = 100 hours
print(f"  100h+:   {count_100h:6d} ({count_100h/len(durations)*100:5.1f}%)")

print("\n" + "=" * 60)
print("COMPARISON: M30 vs H1")
print("=" * 60)
print(f"\n| Metric      | H1 Base | M30 Base |")
print(f"|-------------|---------|----------|")
print(f"| Median time | 23h     | {np.median(hours):.0f}h       |")
print(f"| Mean time   | 30h     | {np.mean(hours):.0f}h       |")
