"""
Analyze MFC Return Types: Real Reversal vs Decay
=================================================
When MFC returns to the box from an extreme:
1. FAST return = price actually reversed? (tradeable)
2. SLOW return = price went flat, MFC decayed? (not tradeable)

Compare:
- MFC velocity during return
- Price movement during return
- What happens AFTER reaching the box
"""
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from pathlib import Path

def log(msg=""):
    print(msg, flush=True)

DATA_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/data')
CLEANED_DIR = DATA_DIR / 'cleaned'

CURRENCIES = ['EUR', 'USD', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD', 'NZD']

# We need price data too - let's use a pair for each currency
PRICE_PAIRS = {
    'EUR': 'EURUSD',
    'USD': 'USDJPY',
    'GBP': 'GBPUSD',
    'JPY': 'USDJPY',
    'CHF': 'USDCHF',
    'CAD': 'USDCAD',
    'AUD': 'AUDUSD',
    'NZD': 'NZDUSD',
}

BOX_UPPER = 0.2
BOX_LOWER = -0.2
EXTREME_UPPER = 0.5
EXTREME_LOWER = -0.5

log("=" * 70)
log("MFC RETURN TYPE ANALYSIS: REVERSAL vs DECAY")
log("=" * 70)

# ============================================================================
# LOAD MFC DATA
# ============================================================================

def load_mfc(currency, timeframe):
    fp = CLEANED_DIR / f'mfc_currency_{currency}_{timeframe}_clean.csv'
    if not fp.exists():
        return None
    df = pd.read_csv(fp)
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df = df.set_index('datetime').sort_index()
    return df['MFC']

def load_price(pair):
    fp = DATA_DIR / f'{pair}_GMT+0_US-DST_M1.csv'
    if not fp.exists():
        return None
    df = pd.read_csv(fp)
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df = df.set_index('datetime').sort_index()
    # Resample to M30
    price_m30 = df['Close'].resample('30min').last().dropna()
    return price_m30

log("\nLoading data...")

mfc_data = {}
price_data = {}

for ccy in CURRENCIES:
    mfc_data[ccy] = load_mfc(ccy, 'M30')
    pair = PRICE_PAIRS[ccy]
    if pair not in price_data:
        price_data[pair] = load_price(pair)

# Build aligned dataframe
df = pd.DataFrame(index=mfc_data['EUR'].index)

for ccy in CURRENCIES:
    df[f'{ccy}_mfc'] = mfc_data[ccy]
    df[f'{ccy}_mfc_vel'] = mfc_data[ccy].diff()

    pair = PRICE_PAIRS[ccy]
    if price_data[pair] is not None:
        df[f'{ccy}_price'] = price_data[pair].reindex(df.index, method='ffill')
        df[f'{ccy}_price_pct'] = df[f'{ccy}_price'].pct_change() * 100

df = df.dropna()
log(f"Aligned: {len(df):,} M30 bars")

# ============================================================================
# FIND RETURN EVENTS
# ============================================================================

log("\n" + "=" * 70)
log("FINDING MFC RETURN EVENTS")
log("=" * 70)

def find_return_events(df, ccy, lookback=12, lookforward=12):
    """
    Find events where MFC returns from extreme to box.
    Classify by:
    - Return velocity (fast vs slow)
    - Price movement during return
    """
    mfc = df[f'{ccy}_mfc'].values
    mfc_vel = df[f'{ccy}_mfc_vel'].values
    price = df[f'{ccy}_price'].values

    events = []

    for i in range(lookback, len(df) - lookforward):
        # Current at/near box
        at_box = (mfc[i] >= BOX_LOWER - 0.05) and (mfc[i] <= BOX_UPPER + 0.05)

        if not at_box:
            continue

        # Was at extreme in lookback period
        past_mfc = mfc[i-lookback:i]
        was_above_extreme = (past_mfc > EXTREME_UPPER).any()
        was_below_extreme = (past_mfc < EXTREME_LOWER).any()

        if not (was_above_extreme or was_below_extreme):
            continue

        # Determine direction of return
        if was_above_extreme:
            direction = 'from_above'
            extreme_idx = np.where(past_mfc > EXTREME_UPPER)[0][-1]  # Last extreme
            extreme_val = past_mfc[extreme_idx]
        else:
            direction = 'from_below'
            extreme_idx = np.where(past_mfc < EXTREME_LOWER)[0][-1]
            extreme_val = past_mfc[extreme_idx]

        # Calculate return characteristics
        bars_to_return = lookback - extreme_idx
        mfc_change = mfc[i] - extreme_val

        # Return velocity (MFC change per bar)
        return_velocity = abs(mfc_change) / bars_to_return if bars_to_return > 0 else 0

        # Price movement during return
        price_at_extreme = price[i - lookback + extreme_idx]
        price_now = price[i]
        price_change_pct = ((price_now - price_at_extreme) / price_at_extreme) * 100

        # Classify return type
        # Fast return = velocity > 0.05 per bar
        # Slow return = velocity < 0.02 per bar
        if return_velocity > 0.05:
            return_type = 'fast'
        elif return_velocity < 0.02:
            return_type = 'slow'
        else:
            return_type = 'medium'

        # What happens AFTER reaching box?
        future_mfc = mfc[i+1:i+1+lookforward]
        future_price = price[i+1:i+1+lookforward]

        if direction == 'from_above':
            # Came from strong, now at box
            # Continuation = goes below box (keeps weakening)
            # Reversal = goes back above box (bounces)
            continued = (future_mfc < BOX_LOWER).any()
            reversed_back = (future_mfc > EXTREME_UPPER * 0.8).any()  # Goes back toward extreme
        else:
            # Came from weak, now at box
            # Continuation = goes above box (keeps strengthening)
            # Reversal = goes back below box (bounces)
            continued = (future_mfc > BOX_UPPER).any()
            reversed_back = (future_mfc < EXTREME_LOWER * 0.8).any()

        # Future price movement
        if len(future_price) > 0:
            future_price_change = ((future_price[-1] - price_now) / price_now) * 100
        else:
            future_price_change = 0

        events.append({
            'datetime': df.index[i],
            'ccy': ccy,
            'direction': direction,
            'extreme_val': extreme_val,
            'current_mfc': mfc[i],
            'bars_to_return': bars_to_return,
            'return_velocity': return_velocity,
            'return_type': return_type,
            'price_change_during': price_change_pct,
            'continued': continued,
            'reversed_back': reversed_back,
            'future_price_change': future_price_change,
        })

    return pd.DataFrame(events) if events else None

all_events = []
for ccy in CURRENCIES:
    events = find_return_events(df, ccy)
    if events is not None:
        log(f"{ccy}: {len(events):,} return events")
        all_events.append(events)

combined = pd.concat(all_events)
log(f"\nTotal return events: {len(combined):,}")

# ============================================================================
# ANALYSIS BY RETURN TYPE
# ============================================================================

log("\n" + "=" * 70)
log("RETURN TYPE ANALYSIS")
log("=" * 70)

log("\n--- FROM ABOVE (was strong, returned to box) ---")
from_above = combined[combined['direction'] == 'from_above']

log(f"\n{'Return Type':<12} {'Events':>8} {'Continued %':>12} {'Reversed %':>12} {'Avg Price Chg':>14}")
log("-" * 65)

for rtype in ['fast', 'medium', 'slow']:
    subset = from_above[from_above['return_type'] == rtype]
    if len(subset) > 50:
        cont = subset['continued'].mean() * 100
        rev = subset['reversed_back'].mean() * 100
        price_chg = subset['price_change_during'].mean()
        log(f"{rtype:<12} {len(subset):>8} {cont:>11.1f}% {rev:>11.1f}% {price_chg:>+13.2f}%")

log("\n--- FROM BELOW (was weak, returned to box) ---")
from_below = combined[combined['direction'] == 'from_below']

log(f"\n{'Return Type':<12} {'Events':>8} {'Continued %':>12} {'Reversed %':>12} {'Avg Price Chg':>14}")
log("-" * 65)

for rtype in ['fast', 'medium', 'slow']:
    subset = from_below[from_below['return_type'] == rtype]
    if len(subset) > 50:
        cont = subset['continued'].mean() * 100
        rev = subset['reversed_back'].mean() * 100
        price_chg = subset['price_change_during'].mean()
        log(f"{rtype:<12} {len(subset):>8} {cont:>11.1f}% {rev:>11.1f}% {price_chg:>+13.2f}%")

# ============================================================================
# VELOCITY BREAKDOWN
# ============================================================================

log("\n" + "=" * 70)
log("DETAILED VELOCITY ANALYSIS")
log("=" * 70)

log("\nFROM ABOVE - by velocity bucket:")
log(f"{'Velocity':<20} {'Events':>8} {'Cont %':>10} {'Rev %':>10} {'Price Chg':>12}")
log("-" * 65)

for low, high, label in [(0, 0.01, '<0.01 (very slow)'),
                          (0.01, 0.02, '0.01-0.02'),
                          (0.02, 0.03, '0.02-0.03'),
                          (0.03, 0.05, '0.03-0.05'),
                          (0.05, 0.08, '0.05-0.08'),
                          (0.08, 1.0, '>0.08 (very fast)')]:
    subset = from_above[(from_above['return_velocity'] >= low) & (from_above['return_velocity'] < high)]
    if len(subset) >= 30:
        cont = subset['continued'].mean() * 100
        rev = subset['reversed_back'].mean() * 100
        price_chg = subset['price_change_during'].mean()
        log(f"{label:<20} {len(subset):>8} {cont:>9.1f}% {rev:>9.1f}% {price_chg:>+11.2f}%")

log("\nFROM BELOW - by velocity bucket:")
log(f"{'Velocity':<20} {'Events':>8} {'Cont %':>10} {'Rev %':>10} {'Price Chg':>12}")
log("-" * 65)

for low, high, label in [(0, 0.01, '<0.01 (very slow)'),
                          (0.01, 0.02, '0.01-0.02'),
                          (0.02, 0.03, '0.02-0.03'),
                          (0.03, 0.05, '0.03-0.05'),
                          (0.05, 0.08, '0.05-0.08'),
                          (0.08, 1.0, '>0.08 (very fast)')]:
    subset = from_below[(from_below['return_velocity'] >= low) & (from_below['return_velocity'] < high)]
    if len(subset) >= 30:
        cont = subset['continued'].mean() * 100
        rev = subset['reversed_back'].mean() * 100
        price_chg = subset['price_change_during'].mean()
        log(f"{label:<20} {len(subset):>8} {cont:>9.1f}% {rev:>9.1f}% {price_chg:>+11.2f}%")

# ============================================================================
# PRICE MOVEMENT DURING RETURN
# ============================================================================

log("\n" + "=" * 70)
log("PRICE MOVEMENT DURING MFC RETURN")
log("=" * 70)

log("\nFROM ABOVE - Did price actually move down during MFC return?")
log(f"{'Price Change':<25} {'Events':>8} {'Cont %':>10} {'Rev %':>10}")
log("-" * 60)

for low, high, label in [(-10, -0.5, 'Price DOWN >0.5%'),
                          (-0.5, -0.1, 'Price DOWN 0.1-0.5%'),
                          (-0.1, 0.1, 'Price FLAT'),
                          (0.1, 0.5, 'Price UP 0.1-0.5%'),
                          (0.5, 10, 'Price UP >0.5%')]:
    subset = from_above[(from_above['price_change_during'] >= low) & (from_above['price_change_during'] < high)]
    if len(subset) >= 30:
        cont = subset['continued'].mean() * 100
        rev = subset['reversed_back'].mean() * 100
        log(f"{label:<25} {len(subset):>8} {cont:>9.1f}% {rev:>9.1f}%")

log("\nFROM BELOW - Did price actually move up during MFC return?")
log(f"{'Price Change':<25} {'Events':>8} {'Cont %':>10} {'Rev %':>10}")
log("-" * 60)

for low, high, label in [(-10, -0.5, 'Price DOWN >0.5%'),
                          (-0.5, -0.1, 'Price DOWN 0.1-0.5%'),
                          (-0.1, 0.1, 'Price FLAT'),
                          (0.1, 0.5, 'Price UP 0.1-0.5%'),
                          (0.5, 10, 'Price UP >0.5%')]:
    subset = from_below[(from_below['price_change_during'] >= low) & (from_below['price_change_during'] < high)]
    if len(subset) >= 30:
        cont = subset['continued'].mean() * 100
        rev = subset['reversed_back'].mean() * 100
        log(f"{label:<25} {len(subset):>8} {cont:>9.1f}% {rev:>9.1f}%")

# ============================================================================
# COMBINING VELOCITY + PRICE MOVEMENT
# ============================================================================

log("\n" + "=" * 70)
log("COMBINED: VELOCITY + PRICE MOVEMENT")
log("=" * 70)

log("\nFROM ABOVE:")
log(f"{'Condition':<40} {'Events':>8} {'Cont %':>10} {'Rev %':>10}")
log("-" * 75)

# Fast return + price moved down (real reversal?)
subset = from_above[(from_above['return_velocity'] >= 0.05) & (from_above['price_change_during'] < -0.1)]
if len(subset) >= 20:
    log(f"{'Fast return + Price DOWN (real reversal?)':<40} {len(subset):>8} {subset['continued'].mean()*100:>9.1f}% {subset['reversed_back'].mean()*100:>9.1f}%")

# Slow return + price flat (decay?)
subset = from_above[(from_above['return_velocity'] < 0.02) & (from_above['price_change_during'].abs() < 0.1)]
if len(subset) >= 20:
    log(f"{'Slow return + Price FLAT (decay?)':<40} {len(subset):>8} {subset['continued'].mean()*100:>9.1f}% {subset['reversed_back'].mean()*100:>9.1f}%")

# Fast return + price flat (MFC moved but price didn't?)
subset = from_above[(from_above['return_velocity'] >= 0.05) & (from_above['price_change_during'].abs() < 0.1)]
if len(subset) >= 20:
    log(f"{'Fast return + Price FLAT (divergence?)':<40} {len(subset):>8} {subset['continued'].mean()*100:>9.1f}% {subset['reversed_back'].mean()*100:>9.1f}%")

log("\nFROM BELOW:")
log(f"{'Condition':<40} {'Events':>8} {'Cont %':>10} {'Rev %':>10}")
log("-" * 75)

# Fast return + price moved up (real reversal?)
subset = from_below[(from_below['return_velocity'] >= 0.05) & (from_below['price_change_during'] > 0.1)]
if len(subset) >= 20:
    log(f"{'Fast return + Price UP (real reversal?)':<40} {len(subset):>8} {subset['continued'].mean()*100:>9.1f}% {subset['reversed_back'].mean()*100:>9.1f}%")

# Slow return + price flat (decay?)
subset = from_below[(from_below['return_velocity'] < 0.02) & (from_below['price_change_during'].abs() < 0.1)]
if len(subset) >= 20:
    log(f"{'Slow return + Price FLAT (decay?)':<40} {len(subset):>8} {subset['continued'].mean()*100:>9.1f}% {subset['reversed_back'].mean()*100:>9.1f}%")

# Fast return + price flat (divergence?)
subset = from_below[(from_below['return_velocity'] >= 0.05) & (from_below['price_change_during'].abs() < 0.1)]
if len(subset) >= 20:
    log(f"{'Fast return + Price FLAT (divergence?)':<40} {len(subset):>8} {subset['continued'].mean()*100:>9.1f}% {subset['reversed_back'].mean()*100:>9.1f}%")

# ============================================================================
# CONCLUSION
# ============================================================================

log("\n" + "=" * 70)
log("CONCLUSION")
log("=" * 70)

log("\nKey findings:")

# Compare fast vs slow
fast_above = from_above[from_above['return_velocity'] >= 0.05]
slow_above = from_above[from_above['return_velocity'] < 0.02]

if len(fast_above) > 0 and len(slow_above) > 0:
    log(f"\nFROM ABOVE:")
    log(f"  Fast return: {fast_above['continued'].mean()*100:.1f}% continued, {fast_above['reversed_back'].mean()*100:.1f}% reversed")
    log(f"  Slow return: {slow_above['continued'].mean()*100:.1f}% continued, {slow_above['reversed_back'].mean()*100:.1f}% reversed")

fast_below = from_below[from_below['return_velocity'] >= 0.05]
slow_below = from_below[from_below['return_velocity'] < 0.02]

if len(fast_below) > 0 and len(slow_below) > 0:
    log(f"\nFROM BELOW:")
    log(f"  Fast return: {fast_below['continued'].mean()*100:.1f}% continued, {fast_below['reversed_back'].mean()*100:.1f}% reversed")
    log(f"  Slow return: {slow_below['continued'].mean()*100:.1f}% continued, {slow_below['reversed_back'].mean()*100:.1f}% reversed")

log()
