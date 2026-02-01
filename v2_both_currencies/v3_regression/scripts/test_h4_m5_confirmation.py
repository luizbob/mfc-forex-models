"""
Test: H4 MFC Momentum + First M5 Candle Confirmation
=====================================================
Hypothesis: If H4 MFC momentum is up AND first M5 of H4 is up,
the rest of the H4 candle will continue up.
"""
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from pathlib import Path

def log(msg=""):
    print(msg, flush=True)

log("=" * 70)
log("H4 MFC + FIRST M5 CONFIRMATION TEST")
log("=" * 70)

DATA_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/data')
CLEANED_DIR = DATA_DIR / 'cleaned'

def load_mfc(currency, timeframe):
    fp = CLEANED_DIR / f'mfc_currency_{currency}_{timeframe}_clean.csv'
    if not fp.exists():
        return None
    df = pd.read_csv(fp)
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df = df.set_index('datetime').sort_index()
    return df['MFC']

def load_price(pair, timeframe):
    # Try direct file first
    fp = DATA_DIR / f'{pair}_GMT+0_US-DST_{timeframe}.csv'
    if fp.exists():
        with open(fp, 'r') as f:
            first_line = f.readline()
        if 'Date' in first_line or 'Open' in first_line:
            df = pd.read_csv(fp)
        else:
            df = pd.read_csv(fp, header=None,
                           names=['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
        df = df.set_index('datetime').sort_index()
        return df

    # Try M1 and resample
    fp = DATA_DIR / f'{pair}_GMT+0_US-DST_M1.csv'
    if fp.exists():
        with open(fp, 'r') as f:
            first_line = f.readline()
        if 'Date' in first_line or 'Open' in first_line:
            df = pd.read_csv(fp)
        else:
            df = pd.read_csv(fp, header=None,
                           names=['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
        df = df.set_index('datetime').sort_index()

        if timeframe == 'M5':
            return df.resample('5min').agg({
                'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
            }).dropna()
        elif timeframe == 'H4':
            return df.resample('4h').agg({
                'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
            }).dropna()

    return None

# Test on major pairs
pairs_to_test = [
    ('EUR', 'USD'), ('GBP', 'USD'), ('USD', 'JPY'), ('AUD', 'USD'),
    ('EUR', 'JPY'), ('GBP', 'JPY'), ('USD', 'CAD'), ('EUR', 'GBP'),
]

PIP_SIZE = {
    'EURUSD': 0.0001, 'GBPUSD': 0.0001, 'USDJPY': 0.01, 'AUDUSD': 0.0001,
    'EURJPY': 0.01, 'GBPJPY': 0.01, 'USDCAD': 0.0001, 'EURGBP': 0.0001,
}

SPREADS = {
    'EURUSD': 0.8, 'GBPUSD': 1.0, 'USDJPY': 1.0, 'AUDUSD': 0.9,
    'EURJPY': 2.4, 'GBPJPY': 2.2, 'USDCAD': 1.5, 'EURGBP': 1.4,
}

all_results = []

for base_ccy, quote_ccy in pairs_to_test:
    pair = f"{base_ccy}{quote_ccy}"
    log(f"\nProcessing {pair}...")

    # Load MFC
    base_mfc_h4 = load_mfc(base_ccy, 'H4')
    quote_mfc_h4 = load_mfc(quote_ccy, 'H4')

    if base_mfc_h4 is None or quote_mfc_h4 is None:
        log(f"  Missing MFC data")
        continue

    # Load price data
    price_m5 = load_price(pair, 'M5')
    price_h4 = load_price(pair, 'H4')

    if price_m5 is None or price_h4 is None:
        log(f"  Missing price data")
        continue

    # Calculate MFC features (shifted)
    base_vel_h4 = base_mfc_h4.diff().shift(1)
    quote_vel_h4 = quote_mfc_h4.diff().shift(1)
    base_mfc_shifted = base_mfc_h4.shift(1)
    quote_mfc_shifted = quote_mfc_h4.shift(1)

    pip_size = PIP_SIZE[pair]
    spread = SPREADS.get(pair, 1.5)

    # For each H4 candle, find the first M5 and check continuation
    results = []

    for h4_time in price_h4.index:
        # Get H4 candle data
        try:
            h4_open = price_h4.loc[h4_time, 'Open']
            h4_close = price_h4.loc[h4_time, 'Close']
            h4_high = price_h4.loc[h4_time, 'High']
            h4_low = price_h4.loc[h4_time, 'Low']
        except:
            continue

        # Get MFC momentum (from previous closed H4)
        try:
            base_vel = base_vel_h4.reindex([h4_time], method='ffill').iloc[0]
            quote_vel = quote_vel_h4.reindex([h4_time], method='ffill').iloc[0]
            base_mfc = base_mfc_shifted.reindex([h4_time], method='ffill').iloc[0]
            quote_mfc = quote_mfc_shifted.reindex([h4_time], method='ffill').iloc[0]
        except:
            continue

        if pd.isna(base_vel) or pd.isna(quote_vel):
            continue

        # Find first M5 candle of this H4
        # H4 candle spans h4_time to h4_time + 4 hours
        h4_end = h4_time + pd.Timedelta(hours=4)
        m5_in_h4 = price_m5[(price_m5.index >= h4_time) & (price_m5.index < h4_end)]

        if len(m5_in_h4) < 2:
            continue

        first_m5 = m5_in_h4.iloc[0]
        first_m5_time = m5_in_h4.index[0]
        first_m5_open = first_m5['Open']
        first_m5_close = first_m5['Close']
        first_m5_direction = 1 if first_m5_close > first_m5_open else -1

        # MFC divergence direction (base - quote)
        # If base going up relative to quote -> pair should go up
        mfc_direction = 1 if (base_vel - quote_vel) > 0 else -1

        # Check alignment
        aligned = (first_m5_direction == mfc_direction)

        # Calculate remaining H4 movement (from first M5 close to H4 close)
        remaining_pips = (h4_close - first_m5_close) / pip_size

        # If we traded in the direction of alignment
        if aligned:
            if mfc_direction == 1:  # Buy
                trade_pips = remaining_pips - spread
            else:  # Sell
                trade_pips = -remaining_pips - spread
        else:
            trade_pips = None  # No trade

        results.append({
            'datetime': h4_time,
            'pair': pair,
            'mfc_direction': mfc_direction,
            'm5_direction': first_m5_direction,
            'aligned': aligned,
            'remaining_pips': remaining_pips,
            'trade_pips': trade_pips,
            'base_mfc': base_mfc,
            'quote_mfc': quote_mfc,
            'base_vel': base_vel,
            'quote_vel': quote_vel,
        })

    df_results = pd.DataFrame(results)
    all_results.append(df_results)
    log(f"  {len(df_results):,} H4 candles analyzed")

# Combine all results
df = pd.concat(all_results, ignore_index=True)
log(f"\nTotal H4 candles: {len(df):,}")

# Analysis
log("\n" + "=" * 70)
log("ALIGNMENT CHECK: Does first M5 follow MFC direction?")
log("=" * 70)

aligned = df[df['aligned'] == True]
not_aligned = df[df['aligned'] == False]

log(f"\nAligned (MFC and M5 same direction): {len(aligned):,} ({len(aligned)/len(df)*100:.1f}%)")
log(f"Not aligned: {len(not_aligned):,} ({len(not_aligned)/len(df)*100:.1f}%)")

log("\n" + "=" * 70)
log("CONTINUATION CHECK: When aligned, does H4 continue?")
log("=" * 70)

# When aligned and MFC up, does remaining H4 go up?
aligned_up = aligned[aligned['mfc_direction'] == 1]
aligned_down = aligned[aligned['mfc_direction'] == -1]

log(f"\nAligned UP (MFC up, M5 up):")
log(f"  Count: {len(aligned_up):,}")
log(f"  Avg remaining pips: {aligned_up['remaining_pips'].mean():+.1f}")
log(f"  % that continued up: {(aligned_up['remaining_pips'] > 0).mean()*100:.1f}%")
log(f"  Trade result (after spread): {aligned_up['trade_pips'].mean():+.1f} avg")

log(f"\nAligned DOWN (MFC down, M5 down):")
log(f"  Count: {len(aligned_down):,}")
log(f"  Avg remaining pips: {aligned_down['remaining_pips'].mean():+.1f}")
log(f"  % that continued down: {(aligned_down['remaining_pips'] < 0).mean()*100:.1f}%")
log(f"  Trade result (after spread): {aligned_down['trade_pips'].mean():+.1f} avg")

log("\n" + "=" * 70)
log("BY MFC VELOCITY STRENGTH")
log("=" * 70)

log("\nWhen aligned, by MFC velocity strength:")
log(f"{'Velocity':<15} {'Count':>8} {'Continue %':>12} {'Avg Pips':>12}")
log("-" * 50)

for low, high, label in [(0.00, 0.02, '0.00-0.02'), (0.02, 0.05, '0.02-0.05'),
                          (0.05, 0.08, '0.05-0.08'), (0.08, 0.15, '0.08+')]:
    subset = aligned[abs(aligned['base_vel'] - aligned['quote_vel']) >= low]
    subset = subset[abs(subset['base_vel'] - subset['quote_vel']) < high]

    if len(subset) > 50:
        # Check if continued in the expected direction
        up_signals = subset[subset['mfc_direction'] == 1]
        down_signals = subset[subset['mfc_direction'] == -1]

        up_continued = (up_signals['remaining_pips'] > 0).mean() * 100 if len(up_signals) > 0 else 0
        down_continued = (down_signals['remaining_pips'] < 0).mean() * 100 if len(down_signals) > 0 else 0

        avg_continued = (up_continued + down_continued) / 2
        avg_pips = subset['trade_pips'].mean()

        log(f"{label:<15} {len(subset):>8,} {avg_continued:>11.1f}% {avg_pips:>+12.1f}")

log("\n" + "=" * 70)
log("BY MFC POSITION (in/out of box)")
log("=" * 70)

# Check if MFC position matters
aligned['base_in_box'] = (aligned['base_mfc'].abs() <= 0.2).astype(int)
aligned['base_extreme'] = (aligned['base_mfc'].abs() > 0.3).astype(int)

log("\nWhen aligned, by base MFC position:")

in_box = aligned[aligned['base_in_box'] == 1]
extreme = aligned[aligned['base_extreme'] == 1]

log(f"\n  Base MFC in box (-0.2 to 0.2):")
log(f"    Count: {len(in_box):,}")
log(f"    Trade result: {in_box['trade_pips'].mean():+.1f} avg")
log(f"    Win rate: {(in_box['trade_pips'] > 0).mean()*100:.1f}%")

log(f"\n  Base MFC extreme (>0.3 or <-0.3):")
log(f"    Count: {len(extreme):,}")
log(f"    Trade result: {extreme['trade_pips'].mean():+.1f} avg")
log(f"    Win rate: {(extreme['trade_pips'] > 0).mean()*100:.1f}%")

log("\n" + "=" * 70)
log("WHAT IF WE TRADE AGAINST ALIGNMENT?")
log("=" * 70)

# When NOT aligned, what happens?
log(f"\nNot aligned (MFC and M5 opposite):")
log(f"  Count: {len(not_aligned):,}")

# When M5 goes against MFC, does it continue or reverse?
not_aligned_m5_up = not_aligned[not_aligned['m5_direction'] == 1]
not_aligned_m5_down = not_aligned[not_aligned['m5_direction'] == -1]

log(f"\n  M5 up but MFC down:")
log(f"    Count: {len(not_aligned_m5_up):,}")
log(f"    H4 continued up: {(not_aligned_m5_up['remaining_pips'] > 0).mean()*100:.1f}%")
log(f"    Avg remaining: {not_aligned_m5_up['remaining_pips'].mean():+.1f} pips")

log(f"\n  M5 down but MFC up:")
log(f"    Count: {len(not_aligned_m5_down):,}")
log(f"    H4 continued down: {(not_aligned_m5_down['remaining_pips'] < 0).mean()*100:.1f}%")
log(f"    Avg remaining: {not_aligned_m5_down['remaining_pips'].mean():+.1f} pips")

log("\nDONE")
