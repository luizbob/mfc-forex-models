"""
Test: Box Pullback / Mean Reversion
===================================
Question 1: How often does a currency pull back in the OPPOSITE direction?
- Was ABOVE box, came DOWN to box → goes back UP? (mean reversion)
- Was BELOW box, came UP to box → goes back DOWN? (mean reversion)

Question 2: Does pair correlation matter?
- When USD pulls back at box, does EUR behavior affect it?
- Do correlated currencies move together?
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

BOX_UPPER = 0.2
BOX_LOWER = -0.2

log("=" * 70)
log("BOX PULLBACK / MEAN REVERSION TEST")
log("=" * 70)

# ============================================================================
# LOAD DATA
# ============================================================================

def load_mfc(currency, timeframe):
    fp = CLEANED_DIR / f'mfc_currency_{currency}_{timeframe}_clean.csv'
    if not fp.exists():
        return None
    df = pd.read_csv(fp)
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df = df.set_index('datetime').sort_index()
    return df['MFC']

log("\nLoading M30 MFC data...")

mfc_data = {}
for ccy in CURRENCIES:
    mfc_data[ccy] = load_mfc(ccy, 'M30')
    if mfc_data[ccy] is not None:
        log(f"  {ccy}: {len(mfc_data[ccy]):,} bars")

# Build aligned dataframe
df = pd.DataFrame(index=mfc_data['EUR'].index)
for ccy in CURRENCIES:
    if mfc_data[ccy] is not None:
        df[ccy] = mfc_data[ccy]

df = df.dropna()
log(f"\nAligned: {len(df):,} M30 bars ({df.index[0].date()} to {df.index[-1].date()})")

# ============================================================================
# FIND PULLBACK EVENTS
# ============================================================================

log("\n" + "=" * 70)
log("PULLBACK FROM ABOVE (was strong, came to box, goes back up?)")
log("=" * 70)

def find_pullback_from_above(df, ccy, lookback=6, lookforward=12):
    """
    Find events where:
    - Currency was ABOVE box in recent past (lookback bars ago)
    - Now AT the box (between -0.1 and +0.2)
    - Check if it goes back UP in next lookforward bars
    """
    values = df[ccy].values
    events = []

    for i in range(lookback, len(df) - lookforward):
        # Was above box recently
        was_above = (values[i-lookback:i] > BOX_UPPER).any()

        # Now at/near box
        now_at_box = (values[i] >= -0.1) and (values[i] <= BOX_UPPER + 0.1)

        # Came down (current lower than max of lookback)
        came_down = values[i] < values[i-lookback:i].max() - 0.1

        if was_above and now_at_box and came_down:
            future = values[i+1:i+1+lookforward]

            # Did it go back UP (above +0.3)?
            went_up = (future > 0.3).any()

            # Did it continue DOWN (below -0.2)?
            went_down = (future < BOX_LOWER).any()

            # Max/min future
            max_future = future.max()
            min_future = future.min()

            # Other currencies at this moment
            other_ccy_states = {}
            for other in CURRENCIES:
                if other != ccy:
                    other_ccy_states[f'{other}_val'] = df[other].iloc[i]

            events.append({
                'datetime': df.index[i],
                'ccy': ccy,
                'current': values[i],
                'recent_max': values[i-lookback:i].max(),
                'went_up': went_up,
                'went_down': went_down,
                'max_future': max_future,
                'min_future': min_future,
                **other_ccy_states,
            })

    return pd.DataFrame(events) if events else None

all_from_above = []
for ccy in CURRENCIES:
    events = find_pullback_from_above(df, ccy)
    if events is not None and len(events) > 0:
        up_rate = events['went_up'].mean() * 100
        down_rate = events['went_down'].mean() * 100
        log(f"{ccy}: {len(events):>5} events | Went UP: {up_rate:>5.1f}% | Went DOWN: {down_rate:>5.1f}%")
        all_from_above.append(events)

if all_from_above:
    combined = pd.concat(all_from_above)
    log(f"\nTOTAL: {len(combined):,} events")
    log(f"  Pulled back UP (mean reversion): {combined['went_up'].mean()*100:.1f}%")
    log(f"  Continued DOWN: {combined['went_down'].mean()*100:.1f}%")

log("\n" + "=" * 70)
log("PULLBACK FROM BELOW (was weak, came to box, goes back down?)")
log("=" * 70)

def find_pullback_from_below(df, ccy, lookback=6, lookforward=12):
    """
    Find events where:
    - Currency was BELOW box in recent past
    - Now AT the box
    - Check if it goes back DOWN
    """
    values = df[ccy].values
    events = []

    for i in range(lookback, len(df) - lookforward):
        was_below = (values[i-lookback:i] < BOX_LOWER).any()
        now_at_box = (values[i] >= BOX_LOWER - 0.1) and (values[i] <= 0.1)
        came_up = values[i] > values[i-lookback:i].min() + 0.1

        if was_below and now_at_box and came_up:
            future = values[i+1:i+1+lookforward]

            went_down = (future < -0.3).any()
            went_up = (future > BOX_UPPER).any()

            other_ccy_states = {}
            for other in CURRENCIES:
                if other != ccy:
                    other_ccy_states[f'{other}_val'] = df[other].iloc[i]

            events.append({
                'datetime': df.index[i],
                'ccy': ccy,
                'current': values[i],
                'recent_min': values[i-lookback:i].min(),
                'went_down': went_down,
                'went_up': went_up,
                **other_ccy_states,
            })

    return pd.DataFrame(events) if events else None

all_from_below = []
for ccy in CURRENCIES:
    events = find_pullback_from_below(df, ccy)
    if events is not None and len(events) > 0:
        down_rate = events['went_down'].mean() * 100
        up_rate = events['went_up'].mean() * 100
        log(f"{ccy}: {len(events):>5} events | Went DOWN: {down_rate:>5.1f}% | Went UP: {up_rate:>5.1f}%")
        all_from_below.append(events)

if all_from_below:
    combined = pd.concat(all_from_below)
    log(f"\nTOTAL: {len(combined):,} events")
    log(f"  Pulled back DOWN (mean reversion): {combined['went_down'].mean()*100:.1f}%")
    log(f"  Continued UP: {combined['went_up'].mean()*100:.1f}%")

# ============================================================================
# CORRELATION ANALYSIS
# ============================================================================

log("\n" + "=" * 70)
log("CORRELATION ANALYSIS")
log("Does other currency behavior affect pullback success?")
log("=" * 70)

# Define typical correlations (positive = move together, negative = inverse)
CORRELATIONS = {
    ('EUR', 'GBP'): 'positive',
    ('EUR', 'CHF'): 'positive',
    ('AUD', 'NZD'): 'positive',
    ('EUR', 'USD'): 'negative',
    ('GBP', 'USD'): 'negative',
    ('AUD', 'USD'): 'negative',
    ('USD', 'JPY'): 'mixed',
    ('USD', 'CHF'): 'negative',
}

if all_from_above:
    combined = pd.concat(all_from_above)

    log("\nPULLBACK FROM ABOVE - Effect of correlated currency:")
    log(f"{'Currency':<6} {'Corr Ccy':<8} {'Corr Type':<10} {'Condition':<25} {'Events':>8} {'Up %':>8}")
    log("-" * 75)

    for ccy in CURRENCIES:
        ccy_events = combined[combined['ccy'] == ccy]
        if len(ccy_events) < 50:
            continue

        for other in CURRENCIES:
            if other == ccy:
                continue

            corr_type = 'unknown'
            for pair, ct in CORRELATIONS.items():
                if (ccy in pair) and (other in pair):
                    corr_type = ct
                    break

            other_col = f'{other}_val'
            if other_col not in ccy_events.columns:
                continue

            # When correlated currency is ALSO above box (same direction)
            same_dir = ccy_events[ccy_events[other_col] > BOX_UPPER]
            if len(same_dir) >= 20:
                up_rate = same_dir['went_up'].mean() * 100
                log(f"{ccy:<6} {other:<8} {corr_type:<10} {'Other ALSO above box':<25} {len(same_dir):>8} {up_rate:>7.1f}%")

            # When correlated currency is OPPOSITE (below box)
            opp_dir = ccy_events[ccy_events[other_col] < BOX_LOWER]
            if len(opp_dir) >= 20:
                up_rate = opp_dir['went_up'].mean() * 100
                log(f"{ccy:<6} {other:<8} {corr_type:<10} {'Other BELOW box':<25} {len(opp_dir):>8} {up_rate:>7.1f}%")

# ============================================================================
# SUMMARY BY STRENGTH OF MOVE
# ============================================================================

log("\n" + "=" * 70)
log("PULLBACK SUCCESS BY STRENGTH OF PRIOR MOVE")
log("=" * 70)

if all_from_above:
    combined = pd.concat(all_from_above)
    combined['drop_size'] = combined['recent_max'] - combined['current']

    log("\nFrom ABOVE (testing if deeper drop = more likely to bounce):")
    log(f"{'Drop Size':<20} {'Events':>8} {'Went UP %':>12} {'Went DOWN %':>12}")
    log("-" * 55)

    for low, high, label in [(0.1, 0.2, '0.1-0.2 (small)'), (0.2, 0.3, '0.2-0.3'), (0.3, 0.4, '0.3-0.4'), (0.4, 1.0, '>0.4 (large)')]:
        subset = combined[(combined['drop_size'] >= low) & (combined['drop_size'] < high)]
        if len(subset) >= 30:
            log(f"{label:<20} {len(subset):>8} {subset['went_up'].mean()*100:>11.1f}% {subset['went_down'].mean()*100:>11.1f}%")

if all_from_below:
    combined = pd.concat(all_from_below)
    combined['rise_size'] = combined['current'] - combined['recent_min']

    log("\nFrom BELOW (testing if bigger rise = more likely to fall back):")
    log(f"{'Rise Size':<20} {'Events':>8} {'Went DOWN %':>12} {'Went UP %':>12}")
    log("-" * 55)

    for low, high, label in [(0.1, 0.2, '0.1-0.2 (small)'), (0.2, 0.3, '0.2-0.3'), (0.3, 0.4, '0.3-0.4'), (0.4, 1.0, '>0.4 (large)')]:
        subset = combined[(combined['rise_size'] >= low) & (combined['rise_size'] < high)]
        if len(subset) >= 30:
            log(f"{label:<20} {len(subset):>8} {subset['went_down'].mean()*100:>11.1f}% {subset['went_up'].mean()*100:>11.1f}%")

# ============================================================================
# CONCLUSION
# ============================================================================

log("\n" + "=" * 70)
log("CONCLUSION")
log("=" * 70)

if all_from_above:
    combined = pd.concat(all_from_above)
    up_rate = combined['went_up'].mean() * 100
    log(f"\nPullback from ABOVE to box:")
    log(f"  Mean reversion (went back UP): {up_rate:.1f}%")
    if up_rate > 55:
        log(f"  ✓ YES - Tends to pull back UP")
    elif up_rate < 45:
        log(f"  ✗ NO - Tends to continue DOWN")
    else:
        log(f"  ? INCONCLUSIVE")

if all_from_below:
    combined = pd.concat(all_from_below)
    down_rate = combined['went_down'].mean() * 100
    log(f"\nPullback from BELOW to box:")
    log(f"  Mean reversion (went back DOWN): {down_rate:.1f}%")
    if down_rate > 55:
        log(f"  ✓ YES - Tends to pull back DOWN")
    elif down_rate < 45:
        log(f"  ✗ NO - Tends to continue UP")
    else:
        log(f"  ? INCONCLUSIVE")

log()
