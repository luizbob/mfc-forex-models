"""
Analyze what happens when BOTH currencies are sideways but in opposite box positions.

Example: USD above box (+0.2) + sideways, JPY below box (-0.2) + sideways
This creates a "divergence" - currencies are stuck but in opposite regions.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

DATA_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/data')

CURRENCIES = ['EUR', 'USD', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD', 'NZD']

# All pairs
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
print("SIDEWAYS DIVERGENCE ANALYSIS")
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
    print(f"  {ccy}: {len(df)} bars")

# Load price data (H4)
print("\nLoading H4 price data...")
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
SIDEWAYS_VELOCITY = 0.02  # Max absolute velocity to be considered "sideways"
LOOKBACK_VELOCITY = 3     # Bars to calculate velocity
LOOKAHEAD = 6             # Bars to measure price movement (6 H4 bars = 24 hours)

print(f"\nParameters:")
print(f"  Sideways threshold: velocity < {SIDEWAYS_VELOCITY}")
print(f"  Velocity lookback: {LOOKBACK_VELOCITY} H4 bars")
print(f"  Price lookahead: {LOOKAHEAD} H4 bars ({LOOKAHEAD * 4} hours)")

# Calculate velocities for all currencies
print("\nCalculating velocities...")
velocity_h4 = {}
for ccy in CURRENCIES:
    velocity_h4[ccy] = mfc_h4[ccy].diff(periods=LOOKBACK_VELOCITY)

# ============================================================================
# ANALYZE DIVERGENCE SCENARIOS
# ============================================================================

def analyze_divergence(scenario_name, base_condition, quote_condition):
    """
    Analyze pairs where base and quote currencies meet different conditions.

    base_condition: function(mfc, vel) -> bool for base currency
    quote_condition: function(mfc, vel) -> bool for quote currency
    """
    print(f"\n{'='*70}")
    print(f"SCENARIO: {scenario_name}")
    print(f"{'='*70}")

    all_events = []

    for pair in PAIRS:
        if pair not in price_h4:
            continue

        base, quote = get_base_quote(pair)
        prices = price_h4[pair]

        base_mfc = mfc_h4[base]
        quote_mfc = mfc_h4[quote]
        base_vel = velocity_h4[base]
        quote_vel = velocity_h4[quote]

        # Find common timestamps
        common_idx = prices.index.intersection(base_mfc.index).intersection(quote_mfc.index)

        for event_time in common_idx:
            # Check conditions
            try:
                b_mfc = base_mfc.loc[event_time]
                q_mfc = quote_mfc.loc[event_time]
                b_vel = base_vel.loc[event_time]
                q_vel = quote_vel.loc[event_time]

                if pd.isna(b_vel) or pd.isna(q_vel):
                    continue

                if not base_condition(b_mfc, b_vel):
                    continue
                if not quote_condition(q_mfc, q_vel):
                    continue

                # Get price movement
                idx = prices.index.get_loc(event_time)
                if idx + LOOKAHEAD >= len(prices):
                    continue

                entry_price = prices.iloc[idx]['Close']
                future_prices = prices.iloc[idx+1:idx+1+LOOKAHEAD]

                max_high = future_prices['High'].max()
                min_low = future_prices['Low'].min()
                final_close = future_prices.iloc[-1]['Close']

                pip_val = get_pip_value(pair)
                move_up = (max_high - entry_price) / pip_val
                move_down = (entry_price - min_low) / pip_val
                net_move = (final_close - entry_price) / pip_val

                all_events.append({
                    'pair': pair,
                    'base': base,
                    'quote': quote,
                    'time': event_time,
                    'base_mfc': b_mfc,
                    'quote_mfc': q_mfc,
                    'base_vel': b_vel,
                    'quote_vel': q_vel,
                    'mfc_diff': b_mfc - q_mfc,
                    'move_up_pips': move_up,
                    'move_down_pips': move_down,
                    'net_move_pips': net_move,
                    'direction': 'UP' if net_move > 0 else 'DOWN',
                })
            except:
                continue

    if not all_events:
        print("No events found")
        return None

    df = pd.DataFrame(all_events)

    print(f"\nTotal events: {len(df)}")
    print(f"Unique timestamps: {df['time'].nunique()}")
    print(f"Pairs with events: {df['pair'].nunique()}")

    # Overall stats
    print(f"\n--- Pair Price Movement Stats (next {LOOKAHEAD * 4} hours) ---")
    print(f"Avg net move: {df['net_move_pips'].mean():+.1f} pips")
    print(f"Median net move: {df['net_move_pips'].median():+.1f} pips")
    print(f"Avg move UP: {df['move_up_pips'].mean():.1f} pips")
    print(f"Avg move DOWN: {df['move_down_pips'].mean():.1f} pips")
    print(f"Pair direction: {(df['direction'] == 'UP').mean()*100:.1f}% UP, {(df['direction'] == 'DOWN').mean()*100:.1f}% DOWN")

    # By pair
    print(f"\n--- By Pair ---")
    for pair in sorted(df['pair'].unique()):
        pair_df = df[df['pair'] == pair]
        if len(pair_df) > 0:
            up_pct = (pair_df['direction'] == 'UP').mean() * 100
            avg_net = pair_df['net_move_pips'].mean()
            print(f"  {pair}: {len(pair_df):>5} events, {up_pct:>5.1f}% UP, {avg_net:>+6.1f} avg pips")

    return df

# ============================================================================
# RUN SCENARIOS
# ============================================================================

# Helper conditions
def is_sideways(vel):
    return abs(vel) < SIDEWAYS_VELOCITY

def is_above_box(mfc):
    return mfc > 0.2

def is_below_box(mfc):
    return mfc < -0.2

def is_inside_box(mfc):
    return (mfc > -0.2) & (mfc < 0.2)

# Scenario 1: Base ABOVE + sideways, Quote BELOW + sideways
print("\n" + "="*70)
print("Analyzing: BASE above box + sideways, QUOTE below box + sideways")
print("="*70)
df_base_up_quote_down = analyze_divergence(
    "BASE above box, QUOTE below box (BOTH sideways)",
    lambda mfc, vel: is_above_box(mfc) and is_sideways(vel),
    lambda mfc, vel: is_below_box(mfc) and is_sideways(vel)
)

# Scenario 2: Base BELOW + sideways, Quote ABOVE + sideways
print("\n" + "="*70)
print("Analyzing: BASE below box + sideways, QUOTE above box + sideways")
print("="*70)
df_base_down_quote_up = analyze_divergence(
    "BASE below box, QUOTE above box (BOTH sideways)",
    lambda mfc, vel: is_below_box(mfc) and is_sideways(vel),
    lambda mfc, vel: is_above_box(mfc) and is_sideways(vel)
)

# Scenario 3: BOTH above box + sideways (convergence)
print("\n" + "="*70)
print("Analyzing: BOTH above box + sideways")
print("="*70)
df_both_above = analyze_divergence(
    "BOTH above box (BOTH sideways)",
    lambda mfc, vel: is_above_box(mfc) and is_sideways(vel),
    lambda mfc, vel: is_above_box(mfc) and is_sideways(vel)
)

# Scenario 4: BOTH below box + sideways (convergence)
print("\n" + "="*70)
print("Analyzing: BOTH below box + sideways")
print("="*70)
df_both_below = analyze_divergence(
    "BOTH below box (BOTH sideways)",
    lambda mfc, vel: is_below_box(mfc) and is_sideways(vel),
    lambda mfc, vel: is_below_box(mfc) and is_sideways(vel)
)

# ============================================================================
# COMPARISON SUMMARY
# ============================================================================
print("\n" + "="*70)
print("COMPARISON SUMMARY")
print("="*70)

print(f"\n{'Scenario':<45} | {'Events':>8} | {'% UP':>7} | {'Avg Move':>10} | Interpretation")
print("-" * 100)

scenarios = [
    ("Base ABOVE + Quote BELOW (divergence)", df_base_up_quote_down),
    ("Base BELOW + Quote ABOVE (divergence)", df_base_down_quote_up),
    ("BOTH ABOVE box (convergence)", df_both_above),
    ("BOTH BELOW box (convergence)", df_both_below),
]

for name, df in scenarios:
    if df is not None:
        up_pct = (df['direction'] == 'UP').mean() * 100
        avg = df['net_move_pips'].mean()
        if up_pct > 52:
            interp = "Pair tends to GO UP"
        elif up_pct < 48:
            interp = "Pair tends to GO DOWN"
        else:
            interp = "Random (no edge)"
        print(f"{name:<45} | {len(df):>8} | {up_pct:>6.1f}% | {avg:>+9.1f} | {interp}")

print(f"\nCompleted: {datetime.now()}")
