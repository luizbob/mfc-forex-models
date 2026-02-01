"""
Test: Momentum Strategy vs Actual Price Movement
=================================================
For each currency momentum signal, check how ALL pairs containing
that currency moved. Does single currency momentum = profitable trade?
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

# All pairs we have
PAIRS = [
    'EURUSD', 'GBPUSD', 'AUDUSD', 'NZDUSD', 'USDCAD', 'USDCHF', 'USDJPY',
    'EURGBP', 'EURJPY', 'EURCHF', 'EURCAD', 'EURAUD', 'EURNZD',
    'GBPJPY', 'GBPCHF', 'GBPCAD', 'GBPAUD', 'GBPNZD',
    'AUDJPY', 'AUDCHF', 'AUDCAD', 'AUDNZD',
    'NZDJPY', 'NZDCHF', 'NZDCAD',
    'CADJPY', 'CADCHF',
    'CHFJPY'
]

# Pip values
PIP_SIZE = {p: 0.01 if 'JPY' in p else 0.0001 for p in PAIRS}

BOX_UPPER = 0.2
BOX_LOWER = -0.2
VELOCITY_THRESHOLD = 0.05

log("=" * 70)
log("MOMENTUM STRATEGY vs ACTUAL PRICE MOVEMENT")
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

log("\nLoading M30 MFC data...")

mfc_data = {}
for ccy in CURRENCIES:
    mfc_data[ccy] = load_mfc(ccy, 'M30')

# Build aligned MFC dataframe
mfc_df = pd.DataFrame(index=mfc_data['EUR'].index)
for ccy in CURRENCIES:
    if mfc_data[ccy] is not None:
        mfc_df[ccy] = mfc_data[ccy]
        mfc_df[f'{ccy}_vel'] = mfc_df[ccy].diff()

mfc_df = mfc_df.dropna()
log(f"MFC data: {len(mfc_df):,} bars")

# ============================================================================
# LOAD PRICE DATA
# ============================================================================

def load_price(pair, timeframe):
    # Price files are in main data folder with different naming
    fp = DATA_DIR / f'{pair}_GMT+0_US-DST_{timeframe}.csv'
    if not fp.exists():
        return None
    df = pd.read_csv(fp, header=None, names=['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df = df.set_index('datetime').sort_index()
    return df[['Open', 'High', 'Low', 'Close']]

log("\nLoading M30 price data...")

price_data = {}
for pair in PAIRS:
    price_data[pair] = load_price(pair, 'M30')
    if price_data[pair] is not None:
        log(f"  {pair}: {len(price_data[pair]):,} bars")

# ============================================================================
# FIND MOMENTUM SIGNALS AND MEASURE PRICE MOVEMENT
# ============================================================================

log("\n" + "=" * 70)
log("FINDING MOMENTUM SIGNALS AND MEASURING PRICE")
log("=" * 70)

def find_momentum_signals(mfc_df, ccy, vel_threshold=0.05):
    """Find momentum entry/exit points for a currency."""
    values = mfc_df[ccy].values
    velocities = mfc_df[f'{ccy}_vel'].values
    times = mfc_df.index

    signals = []
    i = 1

    while i < len(mfc_df) - 1:
        # Entry from ABOVE (MFC coming down) -> currency weakening
        prev_above = values[i-1] > BOX_UPPER
        now_in_or_below = values[i] <= BOX_UPPER
        vel_down = velocities[i] < -vel_threshold

        if prev_above and now_in_or_below and vel_down:
            entry_idx = i
            entry_vel = velocities[i]
            direction = 'weak'  # Currency is weakening

            j = i + 1
            while j < len(mfc_df):
                if velocities[j] > -vel_threshold or velocities[j] > 0:
                    break
                j += 1

            if j < len(mfc_df):
                signals.append({
                    'entry_time': times[entry_idx],
                    'exit_time': times[j],
                    'ccy': ccy,
                    'direction': direction,
                    'entry_vel': abs(entry_vel),
                    'entry_idx': entry_idx,
                    'exit_idx': j,
                })
                i = j + 1
                continue

        # Entry from BELOW (MFC coming up) -> currency strengthening
        prev_below = values[i-1] < BOX_LOWER
        now_in_or_above = values[i] >= BOX_LOWER
        vel_up = velocities[i] > vel_threshold

        if prev_below and now_in_or_above and vel_up:
            entry_idx = i
            entry_vel = velocities[i]
            direction = 'strong'  # Currency is strengthening

            j = i + 1
            while j < len(mfc_df):
                if velocities[j] < vel_threshold or velocities[j] < 0:
                    break
                j += 1

            if j < len(mfc_df):
                signals.append({
                    'entry_time': times[entry_idx],
                    'exit_time': times[j],
                    'ccy': ccy,
                    'direction': direction,
                    'entry_vel': abs(entry_vel),
                    'entry_idx': entry_idx,
                    'exit_idx': j,
                })
                i = j + 1
                continue

        i += 1

    return signals

def get_price_move(pair, entry_time, exit_time):
    """Get price movement in pips for a pair during signal period."""
    if price_data[pair] is None:
        return None

    price_df = price_data[pair]

    # Find closest timestamps
    try:
        if entry_time not in price_df.index:
            # Find nearest
            idx = price_df.index.get_indexer([entry_time], method='nearest')[0]
            entry_time = price_df.index[idx]
        if exit_time not in price_df.index:
            idx = price_df.index.get_indexer([exit_time], method='nearest')[0]
            exit_time = price_df.index[idx]

        entry_price = price_df.loc[entry_time, 'Close']
        exit_price = price_df.loc[exit_time, 'Close']

        pip_move = (exit_price - entry_price) / PIP_SIZE[pair]
        return pip_move
    except:
        return None

# Get all signals for all currencies
all_signals = []
for ccy in CURRENCIES:
    signals = find_momentum_signals(mfc_df, ccy, VELOCITY_THRESHOLD)
    all_signals.extend(signals)
    log(f"{ccy}: {len(signals)} momentum signals")

log(f"\nTotal signals: {len(all_signals)}")

# ============================================================================
# MEASURE PRICE MOVEMENT FOR ALL PAIRS
# ============================================================================

log("\n" + "=" * 70)
log("PRICE MOVEMENT BY CURRENCY SIGNAL")
log("=" * 70)

results = []

for signal in all_signals:
    ccy = signal['ccy']
    direction = signal['direction']
    entry_time = signal['entry_time']
    exit_time = signal['exit_time']
    entry_vel = signal['entry_vel']

    # Check all pairs containing this currency
    for pair in PAIRS:
        base = pair[:3]
        quote = pair[3:]

        if ccy not in [base, quote]:
            continue

        pip_move = get_price_move(pair, entry_time, exit_time)
        if pip_move is None:
            continue

        # Determine expected direction
        # If currency is strengthening (strong), we expect:
        #   - pair to go UP if currency is BASE (e.g., EUR strong -> EURUSD up)
        #   - pair to go DOWN if currency is QUOTE (e.g., USD strong -> EURUSD down)
        if ccy == base:
            expected_sign = 1 if direction == 'strong' else -1
        else:
            expected_sign = -1 if direction == 'strong' else 1

        # Adjusted pips (positive = trade went our way)
        adjusted_pips = pip_move * expected_sign

        results.append({
            'ccy': ccy,
            'pair': pair,
            'direction': direction,
            'entry_vel': entry_vel,
            'pip_move': pip_move,
            'adjusted_pips': adjusted_pips,
            'win': adjusted_pips > 0,
            'entry_time': entry_time,
        })

results_df = pd.DataFrame(results)
log(f"\nTotal pair-signals: {len(results_df):,}")

# ============================================================================
# OVERALL RESULTS
# ============================================================================

log("\n" + "=" * 70)
log("OVERALL RESULTS")
log("=" * 70)

win_rate = results_df['win'].mean() * 100
avg_pips = results_df['adjusted_pips'].mean()
total_pips = results_df['adjusted_pips'].sum()

log(f"\nAll signals trading all related pairs:")
log(f"  Trades: {len(results_df):,}")
log(f"  Win rate: {win_rate:.1f}%")
log(f"  Avg pips/trade: {avg_pips:+.2f}")
log(f"  Total pips: {total_pips:+,.0f}")

# ============================================================================
# BY VELOCITY
# ============================================================================

log("\n" + "=" * 70)
log("BY ENTRY VELOCITY")
log("=" * 70)

log(f"\n{'Velocity':<15} {'Trades':>10} {'Win %':>10} {'Avg Pips':>12} {'Total Pips':>12}")
log("-" * 65)

for low, high, label in [(0.05, 0.08, '0.05-0.08'), (0.08, 0.12, '0.08-0.12'), (0.12, 0.20, '0.12-0.20'), (0.20, 1.0, '>0.20')]:
    subset = results_df[(results_df['entry_vel'] >= low) & (results_df['entry_vel'] < high)]
    if len(subset) >= 100:
        wr = subset['win'].mean() * 100
        avg = subset['adjusted_pips'].mean()
        total = subset['adjusted_pips'].sum()
        log(f"{label:<15} {len(subset):>10,} {wr:>9.1f}% {avg:>+12.2f} {total:>+12,.0f}")

# ============================================================================
# BY PAIR
# ============================================================================

log("\n" + "=" * 70)
log("BY PAIR")
log("=" * 70)

log(f"\n{'Pair':<10} {'Trades':>8} {'Win %':>8} {'Avg Pips':>10} {'Total Pips':>12}")
log("-" * 55)

pair_stats = []
for pair in PAIRS:
    subset = results_df[results_df['pair'] == pair]
    if len(subset) >= 50:
        wr = subset['win'].mean() * 100
        avg = subset['adjusted_pips'].mean()
        total = subset['adjusted_pips'].sum()
        pair_stats.append((pair, len(subset), wr, avg, total))

# Sort by total pips
pair_stats.sort(key=lambda x: x[4], reverse=True)
for pair, trades, wr, avg, total in pair_stats:
    log(f"{pair:<10} {trades:>8,} {wr:>7.1f}% {avg:>+10.2f} {total:>+12,.0f}")

# ============================================================================
# BY CURRENCY SIGNAL
# ============================================================================

log("\n" + "=" * 70)
log("BY SIGNALING CURRENCY")
log("=" * 70)

log(f"\n{'Currency':<10} {'Trades':>8} {'Win %':>8} {'Avg Pips':>10} {'Total Pips':>12}")
log("-" * 55)

for ccy in CURRENCIES:
    subset = results_df[results_df['ccy'] == ccy]
    if len(subset) > 0:
        wr = subset['win'].mean() * 100
        avg = subset['adjusted_pips'].mean()
        total = subset['adjusted_pips'].sum()
        log(f"{ccy:<10} {subset.shape[0]:>8,} {wr:>7.1f}% {avg:>+10.2f} {total:>+12,.0f}")

# ============================================================================
# DOES OPPOSITE CURRENCY MATTER?
# ============================================================================

log("\n" + "=" * 70)
log("DOES OPPOSITE CURRENCY MOMENTUM MATTER?")
log("=" * 70)

# For each trade, check if the other currency in the pair also has momentum
def check_other_currency_state(row):
    """Check the state of the other currency at entry time."""
    pair = row['pair']
    signal_ccy = row['ccy']
    entry_time = row['entry_time']

    base = pair[:3]
    quote = pair[3:]
    other_ccy = quote if signal_ccy == base else base

    try:
        other_mfc = mfc_df.loc[entry_time, other_ccy]
        other_vel = mfc_df.loc[entry_time, f'{other_ccy}_vel']

        # Check if other currency has OPPOSITE momentum (ideal scenario)
        if row['direction'] == 'strong':
            # We're buying signal_ccy, ideal if other_ccy is weak
            other_opposite = other_vel < -0.03
            other_same = other_vel > 0.03
        else:
            # We're selling signal_ccy, ideal if other_ccy is strong
            other_opposite = other_vel > 0.03
            other_same = other_vel < -0.03

        other_neutral = not other_opposite and not other_same

        if other_opposite:
            return 'opposite'
        elif other_same:
            return 'same'
        else:
            return 'neutral'
    except:
        return 'unknown'

results_df['other_ccy_state'] = results_df.apply(check_other_currency_state, axis=1)

log(f"\n{'Other Currency':<20} {'Trades':>10} {'Win %':>10} {'Avg Pips':>12}")
log("-" * 55)

for state in ['opposite', 'neutral', 'same']:
    subset = results_df[results_df['other_ccy_state'] == state]
    if len(subset) >= 100:
        wr = subset['win'].mean() * 100
        avg = subset['adjusted_pips'].mean()
        log(f"{state.upper():<20} {len(subset):>10,} {wr:>9.1f}% {avg:>+12.2f}")

# ============================================================================
# BEST SCENARIO: BOTH CURRENCIES HAVE MOMENTUM
# ============================================================================

log("\n" + "=" * 70)
log("BEST SCENARIO: SIGNAL CCY STRONG + OTHER CCY OPPOSITE")
log("=" * 70)

best = results_df[results_df['other_ccy_state'] == 'opposite']
if len(best) > 0:
    log(f"\nOpposite momentum trades: {len(best):,}")
    log(f"Win rate: {best['win'].mean()*100:.1f}%")
    log(f"Avg pips: {best['adjusted_pips'].mean():+.2f}")
    log(f"Total pips: {best['adjusted_pips'].sum():+,.0f}")

    # By velocity for opposite momentum
    log(f"\n{'Velocity':<15} {'Trades':>10} {'Win %':>10} {'Avg Pips':>12}")
    log("-" * 50)
    for low, high, label in [(0.05, 0.08, '0.05-0.08'), (0.08, 0.12, '0.08-0.12'), (0.12, 0.20, '0.12-0.20'), (0.20, 1.0, '>0.20')]:
        subset = best[(best['entry_vel'] >= low) & (best['entry_vel'] < high)]
        if len(subset) >= 50:
            wr = subset['win'].mean() * 100
            avg = subset['adjusted_pips'].mean()
            log(f"{label:<15} {len(subset):>10,} {wr:>9.1f}% {avg:>+12.2f}")

# ============================================================================
# HOLDOUT
# ============================================================================

log("\n" + "=" * 70)
log("HOLDOUT (Last 3 Months)")
log("=" * 70)

holdout_start = mfc_df.index.max() - pd.DateOffset(months=3)
holdout = results_df[results_df['entry_time'] >= holdout_start]

if len(holdout) > 0:
    log(f"\nHoldout trades: {len(holdout):,}")
    log(f"Win rate: {holdout['win'].mean()*100:.1f}%")
    log(f"Avg pips: {holdout['adjusted_pips'].mean():+.2f}")
    log(f"Total pips: {holdout['adjusted_pips'].sum():+,.0f}")

    # By other currency state
    log(f"\n{'Other Currency':<20} {'Trades':>8} {'Win %':>8} {'Avg Pips':>10}")
    log("-" * 50)
    for state in ['opposite', 'neutral', 'same']:
        subset = holdout[holdout['other_ccy_state'] == state]
        if len(subset) >= 20:
            wr = subset['win'].mean() * 100
            avg = subset['adjusted_pips'].mean()
            log(f"{state.upper():<20} {len(subset):>8,} {wr:>7.1f}% {avg:>+10.2f}")

log("\n" + "=" * 70)
log("CONCLUSION")
log("=" * 70)
log()
