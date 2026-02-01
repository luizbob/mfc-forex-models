"""
Test: Box as Support/Resistance with HTF Confirmation
=====================================================
Scenario 1 (Box as Support):
- H4: Currency BELOW box (< -0.2) AND going UP (recovering)
- M30: Currency coming DOWN toward box (from above), slow velocity
- Question: Does M30 bounce UP off the box? (align with H4)

Scenario 2 (Box as Resistance):
- H4: Currency ABOVE box (> +0.2) AND going DOWN (weakening)
- M30: Currency coming UP toward box (from below), slow velocity
- Question: Does M30 bounce DOWN off the box? (align with H4)
"""
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

def log(msg=""):
    print(msg, flush=True)

DATA_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/data')
CLEANED_DIR = DATA_DIR / 'cleaned'

CURRENCIES = ['EUR', 'USD', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD', 'NZD']

BOX_UPPER = 0.2
BOX_LOWER = -0.2

log("=" * 70)
log("BOX AS SUPPORT/RESISTANCE WITH HTF CONFIRMATION")
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

log("\nLoading MFC data...")

mfc_data = {}
for ccy in CURRENCIES:
    mfc_data[ccy] = {}
    for tf in ['M30', 'H1', 'H4']:
        mfc_data[ccy][tf] = load_mfc(ccy, tf)

log("Data loaded.")

# ============================================================================
# BUILD ALIGNED DATAFRAME ON M30
# ============================================================================

log("Building aligned dataframe...")

ref_series = mfc_data['EUR']['M30']
df = pd.DataFrame(index=ref_series.index)

for ccy in CURRENCIES:
    # M30 values
    m30 = mfc_data[ccy]['M30']
    if m30 is not None:
        df[f'{ccy}_m30'] = m30
        df[f'{ccy}_m30_vel'] = m30.diff()

    # H4 - shift(1) and ffill to M30 grid (use closed bar)
    h4 = mfc_data[ccy]['H4']
    if h4 is not None:
        df[f'{ccy}_h4'] = h4.shift(1).reindex(df.index, method='ffill')
        df[f'{ccy}_h4_vel'] = df[f'{ccy}_h4'].diff()

    # H1 - shift(1) and ffill
    h1 = mfc_data[ccy]['H1']
    if h1 is not None:
        df[f'{ccy}_h1'] = h1.shift(1).reindex(df.index, method='ffill')
        df[f'{ccy}_h1_vel'] = df[f'{ccy}_h1'].diff()

df = df.dropna()
log(f"Aligned: {len(df):,} M30 bars ({df.index[0].date()} to {df.index[-1].date()})")

# ============================================================================
# SCENARIO 1: BOX AS SUPPORT
# H4 below box & going up, M30 coming down to box -> bounce up?
# ============================================================================

log("\n" + "=" * 70)
log("SCENARIO 1: BOX AS SUPPORT")
log("H4: Below box AND going UP | M30: Coming DOWN to box")
log("Question: Does M30 bounce UP off the box?")
log("=" * 70)

def find_support_events(df, ccy, lookforward=12):
    """
    H4: Below box (< -0.2) AND going UP (vel > 0)
    M30: Near/at box edge, coming from above, slow velocity
    """
    m30 = df[f'{ccy}_m30'].values
    m30_vel = df[f'{ccy}_m30_vel'].values
    h4 = df[f'{ccy}_h4'].values
    h4_vel = df[f'{ccy}_h4_vel'].values

    events = []

    for i in range(2, len(df) - lookforward):
        # H4 conditions: below box AND going up
        h4_below_box = h4[i] < BOX_LOWER
        h4_going_up = h4_vel[i] > 0

        # M30 conditions: near box (0 to +0.3), coming from above (was higher), slow velocity
        m30_near_box = (m30[i] >= BOX_LOWER) and (m30[i] <= 0.3)
        m30_coming_down = m30[i] < m30[i-1]  # Currently falling
        m30_slow = abs(m30_vel[i]) < 0.03  # Slow velocity

        if h4_below_box and h4_going_up and m30_near_box and m30_coming_down and m30_slow:
            future = m30[i+1:i+1+lookforward]

            # Did it bounce UP? (go above current + 0.1)
            bounced_up = (future > m30[i] + 0.1).any()

            # Did it continue DOWN? (go below current - 0.1)
            continued_down = (future < m30[i] - 0.1).any()

            # Net move after lookforward bars
            net_move = future[-1] - m30[i] if len(future) > 0 else 0

            events.append({
                'datetime': df.index[i],
                'ccy': ccy,
                'm30': m30[i],
                'm30_vel': m30_vel[i],
                'h4': h4[i],
                'h4_vel': h4_vel[i],
                'bounced_up': bounced_up,
                'continued_down': continued_down,
                'net_move': net_move,
            })

    return pd.DataFrame(events) if events else None

all_support_events = []
for ccy in CURRENCIES:
    events = find_support_events(df, ccy, lookforward=12)
    if events is not None and len(events) > 0:
        bounce_rate = events['bounced_up'].mean() * 100
        down_rate = events['continued_down'].mean() * 100
        avg_net = events['net_move'].mean()
        log(f"{ccy}: {len(events):>4} events | Bounced UP: {bounce_rate:>5.1f}% | Continued DOWN: {down_rate:>5.1f}% | Avg net: {avg_net:+.3f}")
        all_support_events.append(events)

if all_support_events:
    combined = pd.concat(all_support_events)
    log(f"\nTOTAL: {len(combined):,} events")
    log(f"  Bounced UP (box as support): {combined['bounced_up'].mean()*100:.1f}%")
    log(f"  Continued DOWN: {combined['continued_down'].mean()*100:.1f}%")
    log(f"  Average net move: {combined['net_move'].mean():+.3f}")

# ============================================================================
# SCENARIO 2: BOX AS RESISTANCE
# H4 above box & going down, M30 coming up to box -> bounce down?
# ============================================================================

log("\n" + "=" * 70)
log("SCENARIO 2: BOX AS RESISTANCE")
log("H4: Above box AND going DOWN | M30: Coming UP to box")
log("Question: Does M30 bounce DOWN off the box?")
log("=" * 70)

def find_resistance_events(df, ccy, lookforward=12):
    """
    H4: Above box (> +0.2) AND going DOWN (vel < 0)
    M30: Near/at box edge, coming from below, slow velocity
    """
    m30 = df[f'{ccy}_m30'].values
    m30_vel = df[f'{ccy}_m30_vel'].values
    h4 = df[f'{ccy}_h4'].values
    h4_vel = df[f'{ccy}_h4_vel'].values

    events = []

    for i in range(2, len(df) - lookforward):
        # H4 conditions: above box AND going down
        h4_above_box = h4[i] > BOX_UPPER
        h4_going_down = h4_vel[i] < 0

        # M30 conditions: near box (-0.3 to 0), coming from below (was lower), slow velocity
        m30_near_box = (m30[i] <= BOX_UPPER) and (m30[i] >= -0.3)
        m30_coming_up = m30[i] > m30[i-1]  # Currently rising
        m30_slow = abs(m30_vel[i]) < 0.03

        if h4_above_box and h4_going_down and m30_near_box and m30_coming_up and m30_slow:
            future = m30[i+1:i+1+lookforward]

            # Did it bounce DOWN?
            bounced_down = (future < m30[i] - 0.1).any()

            # Did it continue UP?
            continued_up = (future > m30[i] + 0.1).any()

            net_move = future[-1] - m30[i] if len(future) > 0 else 0

            events.append({
                'datetime': df.index[i],
                'ccy': ccy,
                'm30': m30[i],
                'm30_vel': m30_vel[i],
                'h4': h4[i],
                'h4_vel': h4_vel[i],
                'bounced_down': bounced_down,
                'continued_up': continued_up,
                'net_move': net_move,
            })

    return pd.DataFrame(events) if events else None

all_resistance_events = []
for ccy in CURRENCIES:
    events = find_resistance_events(df, ccy, lookforward=12)
    if events is not None and len(events) > 0:
        bounce_rate = events['bounced_down'].mean() * 100
        up_rate = events['continued_up'].mean() * 100
        avg_net = events['net_move'].mean()
        log(f"{ccy}: {len(events):>4} events | Bounced DOWN: {bounce_rate:>5.1f}% | Continued UP: {up_rate:>5.1f}% | Avg net: {avg_net:+.3f}")
        all_resistance_events.append(events)

if all_resistance_events:
    combined = pd.concat(all_resistance_events)
    log(f"\nTOTAL: {len(combined):,} events")
    log(f"  Bounced DOWN (box as resistance): {combined['bounced_down'].mean()*100:.1f}%")
    log(f"  Continued UP: {combined['continued_up'].mean()*100:.1f}%")
    log(f"  Average net move: {combined['net_move'].mean():+.3f}")

# ============================================================================
# VARY H4 STRENGTH
# ============================================================================

log("\n" + "=" * 70)
log("ANALYSIS BY H4 STRENGTH")
log("=" * 70)

if all_support_events:
    combined = pd.concat(all_support_events)
    log("\nSUPPORT (H4 below box, going up):")
    log(f"{'H4 Value Range':<20} {'Events':>8} {'Bounce UP %':>12} {'Avg Net':>10}")
    log("-" * 55)

    for low, high, label in [(-0.3, -0.2, '-0.3 to -0.2'), (-0.4, -0.3, '-0.4 to -0.3'), (-0.5, -0.4, '-0.5 to -0.4'), (-1.0, -0.5, '< -0.5')]:
        subset = combined[(combined['h4'] >= low) & (combined['h4'] < high)]
        if len(subset) >= 10:
            log(f"{label:<20} {len(subset):>8} {subset['bounced_up'].mean()*100:>11.1f}% {subset['net_move'].mean():>+10.3f}")

if all_resistance_events:
    combined = pd.concat(all_resistance_events)
    log("\nRESISTANCE (H4 above box, going down):")
    log(f"{'H4 Value Range':<20} {'Events':>8} {'Bounce DN %':>12} {'Avg Net':>10}")
    log("-" * 55)

    for low, high, label in [(0.2, 0.3, '+0.2 to +0.3'), (0.3, 0.4, '+0.3 to +0.4'), (0.4, 0.5, '+0.4 to +0.5'), (0.5, 1.0, '> +0.5')]:
        subset = combined[(combined['h4'] >= low) & (combined['h4'] < high)]
        if len(subset) >= 10:
            log(f"{label:<20} {len(subset):>8} {subset['bounced_down'].mean()*100:>11.1f}% {subset['net_move'].mean():>+10.3f}")

# ============================================================================
# CONCLUSION
# ============================================================================

log("\n" + "=" * 70)
log("CONCLUSION")
log("=" * 70)

if all_support_events:
    combined = pd.concat(all_support_events)
    bounce_rate = combined['bounced_up'].mean() * 100
    net = combined['net_move'].mean()
    log(f"\nBOX AS SUPPORT (when H4 below & going up):")
    if bounce_rate > 55:
        log(f"  ✓ YES - Box acts as support. Bounce rate: {bounce_rate:.1f}%, Avg move: {net:+.3f}")
    elif bounce_rate < 45:
        log(f"  ✗ NO - Box does NOT act as support. Bounce rate: {bounce_rate:.1f}%")
    else:
        log(f"  ? INCONCLUSIVE - Bounce rate: {bounce_rate:.1f}%")

if all_resistance_events:
    combined = pd.concat(all_resistance_events)
    bounce_rate = combined['bounced_down'].mean() * 100
    net = combined['net_move'].mean()
    log(f"\nBOX AS RESISTANCE (when H4 above & going down):")
    if bounce_rate > 55:
        log(f"  ✓ YES - Box acts as resistance. Bounce rate: {bounce_rate:.1f}%, Avg move: {net:+.3f}")
    elif bounce_rate < 45:
        log(f"  ✗ NO - Box does NOT act as resistance. Bounce rate: {bounce_rate:.1f}%")
    else:
        log(f"  ? INCONCLUSIVE - Bounce rate: {bounce_rate:.1f}%")

log()
