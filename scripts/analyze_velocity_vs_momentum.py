"""
Test: What is more predictive - Velocity or Momentum?

Velocity = MFC change (1st derivative)
Momentum = Velocity change (2nd derivative) - is movement accelerating or decelerating?

Example:
- High velocity + positive momentum = moving fast AND speeding up
- High velocity + negative momentum = moving fast BUT slowing down
- Low velocity + positive momentum = slow but picking up speed
- Low velocity + negative momentum = slow and getting slower (stalling)
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

DATA_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/data')

CURRENCIES = ['EUR', 'USD', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD', 'NZD']

PAIRS = [
    'EURUSD', 'EURGBP', 'EURJPY', 'EURCHF', 'EURCAD', 'EURAUD', 'EURNZD',
    'GBPUSD', 'GBPJPY', 'GBPCHF', 'GBPCAD', 'GBPAUD', 'GBPNZD',
    'AUDUSD', 'AUDJPY', 'AUDCHF', 'AUDCAD', 'AUDNZD',
    'NZDUSD', 'NZDJPY', 'NZDCHF', 'NZDCAD',
    'USDJPY', 'USDCHF', 'USDCAD',
    'CADJPY', 'CADCHF',
    'CHFJPY',
]

def get_pip_value(pair):
    return 0.01 if 'JPY' in pair else 0.0001

def get_base_quote(pair):
    return pair[:3], pair[3:]

print("=" * 70)
print("VELOCITY vs MOMENTUM ANALYSIS")
print("=" * 70)
print(f"Started: {datetime.now()}")

# Load H4 MFC data
print("\nLoading H4 MFC data...")
mfc_h4 = {}
for ccy in CURRENCIES:
    df = pd.read_csv(DATA_DIR / 'cleaned' / f'mfc_currency_{ccy}_H4_clean.csv')
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y.%m.%d %H:%M')
    df = df.set_index('datetime')
    mfc_h4[ccy] = df['MFC']

# Load price data (H4)
print("Loading H4 price data...")
price_h4 = {}
for pair in PAIRS:
    try:
        chunks = []
        for chunk in pd.read_csv(DATA_DIR / f'{pair}_GMT+0_US-DST_M1.csv', chunksize=500000):
            chunk['datetime'] = pd.to_datetime(chunk['Date'] + ' ' + chunk['Time'], format='%Y.%m.%d %H:%M:%S')
            chunk = chunk.set_index('datetime')
            h4_chunk = chunk[['Open', 'High', 'Low', 'Close']].resample('4h').agg({
                'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
            }).dropna()
            chunks.append(h4_chunk)
        if chunks:
            price_df = pd.concat(chunks)
            price_df = price_df[~price_df.index.duplicated(keep='first')]
            price_h4[pair] = price_df
    except:
        pass

print(f"Loaded {len(price_h4)} pairs")

# Parameters
VELOCITY_PERIOD = 3       # Bars to calculate velocity
MOMENTUM_PERIOD = 3       # Bars to calculate momentum (velocity change)
LOOKAHEAD = 6             # Bars to measure price movement

# Calculate velocity and momentum for all currencies
print("\nCalculating velocity and momentum...")
velocity_h4 = {}
momentum_h4 = {}
for ccy in CURRENCIES:
    # Velocity = MFC change over N bars (1st derivative)
    velocity_h4[ccy] = mfc_h4[ccy].diff(periods=VELOCITY_PERIOD)
    # Momentum = Velocity change over N bars (2nd derivative)
    momentum_h4[ccy] = velocity_h4[ccy].diff(periods=MOMENTUM_PERIOD)

# ============================================================================
# ANALYSIS FUNCTION
# ============================================================================

def analyze_condition(name, base_cond, quote_cond):
    """Analyze pair movement given conditions on base and quote currencies."""
    all_events = []

    for pair in PAIRS:
        if pair not in price_h4:
            continue

        base, quote = get_base_quote(pair)
        prices = price_h4[pair]

        common_idx = prices.index.intersection(mfc_h4[base].index).intersection(mfc_h4[quote].index)

        for t in common_idx:
            try:
                b_mfc = mfc_h4[base].loc[t]
                q_mfc = mfc_h4[quote].loc[t]
                b_vel = velocity_h4[base].loc[t]
                q_vel = velocity_h4[quote].loc[t]
                b_mom = momentum_h4[base].loc[t]
                q_mom = momentum_h4[quote].loc[t]

                if pd.isna(b_vel) or pd.isna(q_vel) or pd.isna(b_mom) or pd.isna(q_mom):
                    continue

                if not base_cond(b_mfc, b_vel, b_mom):
                    continue
                if not quote_cond(q_mfc, q_vel, q_mom):
                    continue

                idx = prices.index.get_loc(t)
                if idx + LOOKAHEAD >= len(prices):
                    continue

                entry = prices.iloc[idx]['Close']
                future = prices.iloc[idx+1:idx+1+LOOKAHEAD]
                final = future.iloc[-1]['Close']

                pip_val = get_pip_value(pair)
                net_move = (final - entry) / pip_val

                all_events.append({
                    'pair': pair,
                    'time': t,
                    'b_mfc': b_mfc, 'q_mfc': q_mfc,
                    'b_vel': b_vel, 'q_vel': q_vel,
                    'b_mom': b_mom, 'q_mom': q_mom,
                    'net_move': net_move,
                    'direction': 'UP' if net_move > 0 else 'DOWN',
                })
            except:
                continue

    if not all_events:
        return None, 0, 0, 0

    df = pd.DataFrame(all_events)
    up_pct = (df['direction'] == 'UP').mean() * 100
    avg = df['net_move'].mean()
    return df, len(df), up_pct, avg

# ============================================================================
# TEST 1: VELOCITY ONLY (ignore momentum)
# ============================================================================
print("\n" + "="*70)
print("TEST 1: VELOCITY ONLY")
print("="*70)

# Thresholds
VEL_THRESHOLD = 0.03  # Significant velocity

print(f"\nVelocity threshold: |velocity| > {VEL_THRESHOLD}")

# Base velocity UP, Quote velocity DOWN (pair should go UP)
_, n, up, avg = analyze_condition(
    "Base vel UP, Quote vel DOWN",
    lambda mfc, vel, mom: vel > VEL_THRESHOLD,
    lambda mfc, vel, mom: vel < -VEL_THRESHOLD
)
print(f"Base vel UP, Quote vel DOWN: {n:>6} events, {up:>5.1f}% UP, {avg:>+6.1f} pips")

# Base velocity DOWN, Quote velocity UP (pair should go DOWN)
_, n, up, avg = analyze_condition(
    "Base vel DOWN, Quote vel UP",
    lambda mfc, vel, mom: vel < -VEL_THRESHOLD,
    lambda mfc, vel, mom: vel > VEL_THRESHOLD
)
print(f"Base vel DOWN, Quote vel UP: {n:>6} events, {up:>5.1f}% UP, {avg:>+6.1f} pips")

# ============================================================================
# TEST 2: MOMENTUM ONLY (ignore velocity)
# ============================================================================
print("\n" + "="*70)
print("TEST 2: MOMENTUM ONLY (acceleration)")
print("="*70)

MOM_THRESHOLD = 0.02  # Significant momentum

print(f"\nMomentum threshold: |momentum| > {MOM_THRESHOLD}")

# Base momentum UP (accelerating up), Quote momentum DOWN
_, n, up, avg = analyze_condition(
    "Base mom UP, Quote mom DOWN",
    lambda mfc, vel, mom: mom > MOM_THRESHOLD,
    lambda mfc, vel, mom: mom < -MOM_THRESHOLD
)
print(f"Base mom UP, Quote mom DOWN: {n:>6} events, {up:>5.1f}% UP, {avg:>+6.1f} pips")

# Base momentum DOWN, Quote momentum UP
_, n, up, avg = analyze_condition(
    "Base mom DOWN, Quote mom UP",
    lambda mfc, vel, mom: mom < -MOM_THRESHOLD,
    lambda mfc, vel, mom: mom > MOM_THRESHOLD
)
print(f"Base mom DOWN, Quote mom UP: {n:>6} events, {up:>5.1f}% UP, {avg:>+6.1f} pips")

# ============================================================================
# TEST 3: VELOCITY + MOMENTUM COMBINED
# ============================================================================
print("\n" + "="*70)
print("TEST 3: VELOCITY + MOMENTUM COMBINED")
print("="*70)

print(f"\nVelocity > {VEL_THRESHOLD} AND Momentum > {MOM_THRESHOLD}")

# Base: vel UP + mom UP (strong move), Quote: vel DOWN + mom DOWN (strong move down)
_, n, up, avg = analyze_condition(
    "Base strong UP, Quote strong DOWN",
    lambda mfc, vel, mom: vel > VEL_THRESHOLD and mom > MOM_THRESHOLD,
    lambda mfc, vel, mom: vel < -VEL_THRESHOLD and mom < -MOM_THRESHOLD
)
print(f"Base vel+mom UP, Quote vel+mom DOWN: {n:>6} events, {up:>5.1f}% UP, {avg:>+6.1f} pips")

# Opposite
_, n, up, avg = analyze_condition(
    "Base strong DOWN, Quote strong UP",
    lambda mfc, vel, mom: vel < -VEL_THRESHOLD and mom < -MOM_THRESHOLD,
    lambda mfc, vel, mom: vel > VEL_THRESHOLD and mom > MOM_THRESHOLD
)
print(f"Base vel+mom DOWN, Quote vel+mom UP: {n:>6} events, {up:>5.1f}% UP, {avg:>+6.1f} pips")

# ============================================================================
# TEST 4: VELOCITY vs MOMENTUM - CONFLICTING SIGNALS
# ============================================================================
print("\n" + "="*70)
print("TEST 4: VELOCITY vs MOMENTUM - CONFLICTING")
print("="*70)

print("\nVelocity UP but Momentum DOWN (moving up but slowing down)...")

# Base: Moving UP but slowing (vel > 0, mom < 0)
_, n, up, avg = analyze_condition(
    "Base: moving UP but slowing",
    lambda mfc, vel, mom: vel > VEL_THRESHOLD and mom < -MOM_THRESHOLD,
    lambda mfc, vel, mom: True  # Any quote
)
print(f"Base moving UP but SLOWING: {n:>6} events, {up:>5.1f}% UP, {avg:>+6.1f} pips")

# Base: Moving DOWN but slowing (vel < 0, mom > 0)
_, n, up, avg = analyze_condition(
    "Base: moving DOWN but slowing",
    lambda mfc, vel, mom: vel < -VEL_THRESHOLD and mom > MOM_THRESHOLD,
    lambda mfc, vel, mom: True  # Any quote
)
print(f"Base moving DOWN but SLOWING: {n:>6} events, {up:>5.1f}% UP, {avg:>+6.1f} pips")

# ============================================================================
# TEST 5: ABOVE BOX + VELOCITY/MOMENTUM
# ============================================================================
print("\n" + "="*70)
print("TEST 5: POSITION + VELOCITY + MOMENTUM")
print("="*70)

print("\nAbove box (MFC > 0.2) with different velocity/momentum:")

# Above box + velocity DOWN (returning) + momentum DOWN (accelerating down)
_, n, up, avg = analyze_condition(
    "Above box, vel DOWN, mom DOWN",
    lambda mfc, vel, mom: mfc > 0.2 and vel < -VEL_THRESHOLD and mom < -MOM_THRESHOLD,
    lambda mfc, vel, mom: True
)
print(f"ABOVE box, vel DOWN, mom DOWN (strong return): {n:>5} events, {up:>5.1f}% UP, {avg:>+6.1f} pips")

# Above box + velocity DOWN but momentum UP (returning but slowing)
_, n, up, avg = analyze_condition(
    "Above box, vel DOWN, mom UP",
    lambda mfc, vel, mom: mfc > 0.2 and vel < -VEL_THRESHOLD and mom > MOM_THRESHOLD,
    lambda mfc, vel, mom: True
)
print(f"ABOVE box, vel DOWN, mom UP (return slowing):  {n:>5} events, {up:>5.1f}% UP, {avg:>+6.1f} pips")

# Above box + velocity UP (extending) + momentum UP
_, n, up, avg = analyze_condition(
    "Above box, vel UP, mom UP",
    lambda mfc, vel, mom: mfc > 0.2 and vel > VEL_THRESHOLD and mom > MOM_THRESHOLD,
    lambda mfc, vel, mom: True
)
print(f"ABOVE box, vel UP, mom UP (extending strong):  {n:>5} events, {up:>5.1f}% UP, {avg:>+6.1f} pips")

# Above box + velocity UP but momentum DOWN (extending but slowing)
_, n, up, avg = analyze_condition(
    "Above box, vel UP, mom DOWN",
    lambda mfc, vel, mom: mfc > 0.2 and vel > VEL_THRESHOLD and mom < -MOM_THRESHOLD,
    lambda mfc, vel, mom: True
)
print(f"ABOVE box, vel UP, mom DOWN (extend slowing):  {n:>5} events, {up:>5.1f}% UP, {avg:>+6.1f} pips")

print("\n" + "="*70)
print("Below box (MFC < -0.2) with different velocity/momentum:")
print("="*70)

# Below box + velocity UP (returning) + momentum UP (accelerating up)
_, n, up, avg = analyze_condition(
    "Below box, vel UP, mom UP",
    lambda mfc, vel, mom: mfc < -0.2 and vel > VEL_THRESHOLD and mom > MOM_THRESHOLD,
    lambda mfc, vel, mom: True
)
print(f"BELOW box, vel UP, mom UP (strong return):    {n:>5} events, {up:>5.1f}% UP, {avg:>+6.1f} pips")

# Below box + velocity UP but momentum DOWN (returning but slowing)
_, n, up, avg = analyze_condition(
    "Below box, vel UP, mom DOWN",
    lambda mfc, vel, mom: mfc < -0.2 and vel > VEL_THRESHOLD and mom < -MOM_THRESHOLD,
    lambda mfc, vel, mom: True
)
print(f"BELOW box, vel UP, mom DOWN (return slowing): {n:>5} events, {up:>5.1f}% UP, {avg:>+6.1f} pips")

print(f"\nCompleted: {datetime.now()}")
