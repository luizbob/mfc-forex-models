"""
Analyze what happens to price when currency MFC is:
1. Above box (>0.2) and moving sideways
2. Inside box (-0.2 to 0.2) and moving sideways

"Sideways" = low velocity (small MFC change over N bars)
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

DATA_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/data')

CURRENCIES = ['EUR', 'USD', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD', 'NZD']

# Pairs where currency is BASE
BASE_PAIRS = {
    'EUR': ['EURUSD', 'EURGBP', 'EURJPY', 'EURCHF', 'EURCAD', 'EURAUD', 'EURNZD'],
    'GBP': ['GBPUSD', 'GBPJPY', 'GBPCHF', 'GBPCAD', 'GBPAUD', 'GBPNZD'],
    'AUD': ['AUDUSD', 'AUDJPY', 'AUDCHF', 'AUDCAD', 'AUDNZD'],
    'NZD': ['NZDUSD', 'NZDJPY', 'NZDCHF', 'NZDCAD'],
    'USD': ['USDJPY', 'USDCHF', 'USDCAD'],
    'CAD': ['CADJPY', 'CADCHF'],
    'CHF': ['CHFJPY'],
    'JPY': [],
}

# Pairs where currency is QUOTE
QUOTE_PAIRS = {
    'USD': ['EURUSD', 'GBPUSD', 'AUDUSD', 'NZDUSD'],
    'GBP': ['EURGBP'],
    'JPY': ['USDJPY', 'EURJPY', 'GBPJPY', 'AUDJPY', 'NZDJPY', 'CADJPY', 'CHFJPY'],
    'CHF': ['USDCHF', 'EURCHF', 'GBPCHF', 'AUDCHF', 'NZDCHF', 'CADCHF'],
    'CAD': ['USDCAD', 'EURCAD', 'GBPCAD', 'AUDCAD', 'NZDCAD'],
    'AUD': ['EURAUD', 'GBPAUD'],
    'NZD': ['EURNZD', 'GBPNZD', 'AUDNZD'],
    'EUR': [],
}

def get_pip_value(pair):
    return 0.01 if 'JPY' in pair else 0.0001

print("=" * 70)
print("SIDEWAYS MFC ANALYSIS")
print("=" * 70)
print(f"Started: {datetime.now()}")

# Load H4 MFC data (better for this analysis)
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
all_pairs = set()
for pairs in BASE_PAIRS.values():
    all_pairs.update(pairs)

for pair in all_pairs:
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

# ============================================================================
# ANALYSIS
# ============================================================================

def analyze_scenario(scenario_name, mfc_condition):
    """
    Analyze price movement for a given MFC condition.

    mfc_condition: function(mfc_value) -> bool
    """
    print(f"\n{'='*70}")
    print(f"SCENARIO: {scenario_name}")
    print(f"{'='*70}")

    all_events = []

    for ccy in CURRENCIES:
        mfc = mfc_h4[ccy]

        # Calculate velocity
        velocity = mfc.diff(periods=LOOKBACK_VELOCITY)

        # Find sideways periods that match condition
        sideways_mask = velocity.abs() < SIDEWAYS_VELOCITY
        condition_mask = mfc.apply(mfc_condition)

        events = mfc.index[(sideways_mask & condition_mask)]

        for event_time in events:
            # Get price movement for all pairs involving this currency

            # As BASE currency (currency UP = pair UP)
            for pair in BASE_PAIRS.get(ccy, []):
                if pair not in price_h4:
                    continue

                prices = price_h4[pair]
                if event_time not in prices.index:
                    continue

                idx = prices.index.get_loc(event_time)
                if idx + LOOKAHEAD >= len(prices):
                    continue

                entry_price = prices.iloc[idx]['Close']
                future_prices = prices.iloc[idx+1:idx+1+LOOKAHEAD]

                # Max move up and down
                max_high = future_prices['High'].max()
                min_low = future_prices['Low'].min()
                final_close = future_prices.iloc[-1]['Close']

                pip_val = get_pip_value(pair)
                move_up = (max_high - entry_price) / pip_val
                move_down = (entry_price - min_low) / pip_val
                net_move = (final_close - entry_price) / pip_val

                all_events.append({
                    'currency': ccy,
                    'pair': pair,
                    'role': 'BASE',
                    'time': event_time,
                    'mfc': mfc.loc[event_time],
                    'velocity': velocity.loc[event_time],
                    'move_up_pips': move_up,
                    'move_down_pips': move_down,
                    'net_move_pips': net_move,
                    'direction': 'UP' if net_move > 0 else 'DOWN',
                })

            # As QUOTE currency (currency UP = pair DOWN)
            for pair in QUOTE_PAIRS.get(ccy, []):
                if pair not in price_h4:
                    continue

                prices = price_h4[pair]
                if event_time not in prices.index:
                    continue

                idx = prices.index.get_loc(event_time)
                if idx + LOOKAHEAD >= len(prices):
                    continue

                entry_price = prices.iloc[idx]['Close']
                future_prices = prices.iloc[idx+1:idx+1+LOOKAHEAD]

                max_high = future_prices['High'].max()
                min_low = future_prices['Low'].min()
                final_close = future_prices.iloc[-1]['Close']

                pip_val = get_pip_value(pair)
                # Inverted because currency is quote
                move_up = (entry_price - min_low) / pip_val  # Pair down = quote up
                move_down = (max_high - entry_price) / pip_val  # Pair up = quote down
                net_move = (entry_price - final_close) / pip_val  # Inverted

                all_events.append({
                    'currency': ccy,
                    'pair': pair,
                    'role': 'QUOTE',
                    'time': event_time,
                    'mfc': mfc.loc[event_time],
                    'velocity': velocity.loc[event_time],
                    'move_up_pips': move_up,
                    'move_down_pips': move_down,
                    'net_move_pips': net_move,
                    'direction': 'UP' if net_move > 0 else 'DOWN',
                })

    if not all_events:
        print("No events found")
        return None

    df = pd.DataFrame(all_events)

    print(f"\nTotal events: {len(df)}")
    print(f"Unique timestamps: {df['time'].nunique()}")

    # Overall stats
    print(f"\n--- Price Movement Stats (next {LOOKAHEAD * 4} hours) ---")
    print(f"Avg net move: {df['net_move_pips'].mean():+.1f} pips")
    print(f"Median net move: {df['net_move_pips'].median():+.1f} pips")
    print(f"Avg move UP: {df['move_up_pips'].mean():.1f} pips")
    print(f"Avg move DOWN: {df['move_down_pips'].mean():.1f} pips")
    print(f"Direction: {(df['direction'] == 'UP').mean()*100:.1f}% UP, {(df['direction'] == 'DOWN').mean()*100:.1f}% DOWN")

    # By currency
    print(f"\n--- By Currency ---")
    for ccy in CURRENCIES:
        ccy_df = df[df['currency'] == ccy]
        if len(ccy_df) > 0:
            up_pct = (ccy_df['direction'] == 'UP').mean() * 100
            avg_net = ccy_df['net_move_pips'].mean()
            print(f"  {ccy}: {len(ccy_df):>5} events, {up_pct:>5.1f}% UP, {avg_net:>+6.1f} avg pips")

    return df

# ============================================================================
# RUN SCENARIOS
# ============================================================================

# Scenario 1: Above box and sideways
print("\n" + "="*70)
print("Analyzing: ABOVE BOX (MFC > 0.2) + SIDEWAYS")
print("="*70)
df_above = analyze_scenario(
    "ABOVE BOX + SIDEWAYS",
    lambda mfc: mfc > 0.2
)

# Scenario 2: Inside box and sideways
print("\n" + "="*70)
print("Analyzing: INSIDE BOX (-0.2 < MFC < 0.2) + SIDEWAYS")
print("="*70)
df_inside = analyze_scenario(
    "INSIDE BOX + SIDEWAYS",
    lambda mfc: (mfc > -0.2) & (mfc < 0.2)
)

# Scenario 3: Below box and sideways (for comparison)
print("\n" + "="*70)
print("Analyzing: BELOW BOX (MFC < -0.2) + SIDEWAYS")
print("="*70)
df_below = analyze_scenario(
    "BELOW BOX + SIDEWAYS",
    lambda mfc: mfc < -0.2
)

# ============================================================================
# COMPARISON SUMMARY
# ============================================================================
print("\n" + "="*70)
print("COMPARISON SUMMARY")
print("="*70)

print(f"\n{'Scenario':<30} | {'Events':>8} | {'% UP':>7} | {'Avg Move':>10} | {'Interpretation'}")
print("-" * 90)

if df_above is not None:
    up_pct = (df_above['direction'] == 'UP').mean() * 100
    avg = df_above['net_move_pips'].mean()
    interp = "Currency tends to FALL" if up_pct < 50 else "Currency tends to RISE"
    print(f"{'Above Box + Sideways':<30} | {len(df_above):>8} | {up_pct:>6.1f}% | {avg:>+9.1f} | {interp}")

if df_inside is not None:
    up_pct = (df_inside['direction'] == 'UP').mean() * 100
    avg = df_inside['net_move_pips'].mean()
    interp = "Currency tends to FALL" if up_pct < 50 else "Currency tends to RISE"
    print(f"{'Inside Box + Sideways':<30} | {len(df_inside):>8} | {up_pct:>6.1f}% | {avg:>+9.1f} | {interp}")

if df_below is not None:
    up_pct = (df_below['direction'] == 'UP').mean() * 100
    avg = df_below['net_move_pips'].mean()
    interp = "Currency tends to FALL" if up_pct < 50 else "Currency tends to RISE"
    print(f"{'Below Box + Sideways':<30} | {len(df_below):>8} | {up_pct:>6.1f}% | {avg:>+9.1f} | {interp}")

print(f"\nCompleted: {datetime.now()}")
