"""
Test: Momentum Strategy vs Price - Best Pair Selection
=======================================================
For each currency signal, find the BEST pair to trade.
Also show concrete examples of good and bad signals.
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

PAIRS = [
    'EURUSD', 'GBPUSD', 'AUDUSD', 'NZDUSD', 'USDCAD', 'USDCHF', 'USDJPY',
    'EURGBP', 'EURJPY', 'EURCHF', 'EURCAD', 'EURAUD', 'EURNZD',
    'GBPJPY', 'GBPCHF', 'GBPCAD', 'GBPAUD', 'GBPNZD',
    'AUDJPY', 'AUDCHF', 'AUDCAD', 'AUDNZD',
    'NZDJPY', 'NZDCHF', 'NZDCAD',
    'CADJPY', 'CADCHF',
    'CHFJPY'
]

PIP_SIZE = {p: 0.01 if 'JPY' in p else 0.0001 for p in PAIRS}

BOX_UPPER = 0.2
BOX_LOWER = -0.2
VELOCITY_THRESHOLD = 0.05

log("=" * 70)
log("MOMENTUM STRATEGY - BEST PAIR SELECTION")
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

def load_price(pair, timeframe):
    fp = DATA_DIR / f'{pair}_GMT+0_US-DST_{timeframe}.csv'
    if not fp.exists():
        return None
    df = pd.read_csv(fp, header=None, names=['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df = df.set_index('datetime').sort_index()
    return df[['Open', 'High', 'Low', 'Close']]

log("\nLoading data...")

# MFC
mfc_data = {}
for ccy in CURRENCIES:
    mfc_data[ccy] = load_mfc(ccy, 'M30')

mfc_df = pd.DataFrame(index=mfc_data['EUR'].index)
for ccy in CURRENCIES:
    if mfc_data[ccy] is not None:
        mfc_df[ccy] = mfc_data[ccy]
        mfc_df[f'{ccy}_vel'] = mfc_df[ccy].diff()
mfc_df = mfc_df.dropna()

# Price
price_data = {}
for pair in PAIRS:
    price_data[pair] = load_price(pair, 'M30')

log(f"MFC: {len(mfc_df):,} bars | Price data loaded for {len([p for p in PAIRS if price_data[p] is not None])} pairs")

# ============================================================================
# FIND SIGNALS WITH FULL CONTEXT
# ============================================================================

def find_momentum_signals_detailed(mfc_df, ccy, vel_threshold=0.05):
    """Find momentum signals with full MFC context for all currencies."""
    values = mfc_df[ccy].values
    velocities = mfc_df[f'{ccy}_vel'].values
    times = mfc_df.index

    signals = []
    i = 1

    while i < len(mfc_df) - 1:
        # Entry from ABOVE
        prev_above = values[i-1] > BOX_UPPER
        now_in_or_below = values[i] <= BOX_UPPER
        vel_down = velocities[i] < -vel_threshold

        if prev_above and now_in_or_below and vel_down:
            entry_idx = i
            direction = 'weak'

            j = i + 1
            while j < len(mfc_df):
                if velocities[j] > -vel_threshold or velocities[j] > 0:
                    break
                j += 1

            if j < len(mfc_df):
                # Get all currency states at entry
                all_mfc = {c: mfc_df[c].iloc[entry_idx] for c in CURRENCIES}
                all_vel = {c: mfc_df[f'{c}_vel'].iloc[entry_idx] for c in CURRENCIES}

                signals.append({
                    'entry_time': times[entry_idx],
                    'exit_time': times[j],
                    'ccy': ccy,
                    'direction': direction,
                    'entry_mfc': values[entry_idx],
                    'exit_mfc': values[j],
                    'entry_vel': abs(velocities[entry_idx]),
                    'all_mfc': all_mfc,
                    'all_vel': all_vel,
                    'bars_held': j - entry_idx,
                })
                i = j + 1
                continue

        # Entry from BELOW
        prev_below = values[i-1] < BOX_LOWER
        now_in_or_above = values[i] >= BOX_LOWER
        vel_up = velocities[i] > vel_threshold

        if prev_below and now_in_or_above and vel_up:
            entry_idx = i
            direction = 'strong'

            j = i + 1
            while j < len(mfc_df):
                if velocities[j] < vel_threshold or velocities[j] < 0:
                    break
                j += 1

            if j < len(mfc_df):
                all_mfc = {c: mfc_df[c].iloc[entry_idx] for c in CURRENCIES}
                all_vel = {c: mfc_df[f'{c}_vel'].iloc[entry_idx] for c in CURRENCIES}

                signals.append({
                    'entry_time': times[entry_idx],
                    'exit_time': times[j],
                    'ccy': ccy,
                    'direction': direction,
                    'entry_mfc': values[entry_idx],
                    'exit_mfc': values[j],
                    'entry_vel': abs(velocities[entry_idx]),
                    'all_mfc': all_mfc,
                    'all_vel': all_vel,
                    'bars_held': j - entry_idx,
                })
                i = j + 1
                continue

        i += 1

    return signals

def get_price_move(pair, entry_time, exit_time):
    """Get price movement in pips."""
    if price_data[pair] is None:
        return None, None, None

    pdf = price_data[pair]
    try:
        if entry_time not in pdf.index:
            idx = pdf.index.get_indexer([entry_time], method='nearest')[0]
            entry_time = pdf.index[idx]
        if exit_time not in pdf.index:
            idx = pdf.index.get_indexer([exit_time], method='nearest')[0]
            exit_time = pdf.index[idx]

        entry_price = pdf.loc[entry_time, 'Close']
        exit_price = pdf.loc[exit_time, 'Close']
        pip_move = (exit_price - entry_price) / PIP_SIZE[pair]
        return entry_price, exit_price, pip_move
    except:
        return None, None, None

# Get all signals
log("\nFinding momentum signals...")
all_signals = []
for ccy in CURRENCIES:
    signals = find_momentum_signals_detailed(mfc_df, ccy, VELOCITY_THRESHOLD)
    all_signals.extend(signals)
    log(f"  {ccy}: {len(signals)} signals")

log(f"\nTotal signals: {len(all_signals)}")

# ============================================================================
# FOR EACH SIGNAL, FIND BEST PAIR
# ============================================================================

log("\n" + "=" * 70)
log("ANALYZING EACH SIGNAL - BEST PAIR SELECTION")
log("=" * 70)

results = []
example_good = []
example_bad = []

for sig in all_signals:
    ccy = sig['ccy']
    direction = sig['direction']
    entry_time = sig['entry_time']
    exit_time = sig['exit_time']

    # Find all pairs with this currency
    pair_results = []

    for pair in PAIRS:
        base = pair[:3]
        quote = pair[3:]

        if ccy not in [base, quote]:
            continue

        entry_price, exit_price, pip_move = get_price_move(pair, entry_time, exit_time)
        if pip_move is None:
            continue

        # Expected direction
        if ccy == base:
            expected_sign = 1 if direction == 'strong' else -1
        else:
            expected_sign = -1 if direction == 'strong' else 1

        adjusted_pips = pip_move * expected_sign

        # Other currency info
        other_ccy = quote if ccy == base else base
        other_mfc = sig['all_mfc'].get(other_ccy, 0)
        other_vel = sig['all_vel'].get(other_ccy, 0)

        pair_results.append({
            'pair': pair,
            'other_ccy': other_ccy,
            'other_mfc': other_mfc,
            'other_vel': other_vel,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pip_move': pip_move,
            'adjusted_pips': adjusted_pips,
            'expected_dir': 'UP' if expected_sign > 0 else 'DOWN',
        })

    if not pair_results:
        continue

    # Find best pair (highest adjusted pips)
    best = max(pair_results, key=lambda x: x['adjusted_pips'])
    worst = min(pair_results, key=lambda x: x['adjusted_pips'])

    # Store result with best pair
    results.append({
        'entry_time': entry_time,
        'exit_time': exit_time,
        'ccy': ccy,
        'direction': direction,
        'entry_vel': sig['entry_vel'],
        'entry_mfc': sig['entry_mfc'],
        'bars_held': sig['bars_held'],
        'best_pair': best['pair'],
        'best_pips': best['adjusted_pips'],
        'best_other_ccy': best['other_ccy'],
        'best_other_mfc': best['other_mfc'],
        'best_other_vel': best['other_vel'],
        'worst_pair': worst['pair'],
        'worst_pips': worst['adjusted_pips'],
        'all_pairs': pair_results,
    })

    # Store examples
    if best['adjusted_pips'] > 10 and len(example_good) < 5:
        example_good.append({
            'signal': sig,
            'pair_results': pair_results,
            'best': best,
        })

    if best['adjusted_pips'] < -10 and len(example_bad) < 5:
        example_bad.append({
            'signal': sig,
            'pair_results': pair_results,
            'best': best,
            'worst': worst,
        })

results_df = pd.DataFrame(results)
log(f"\nSignals with valid price data: {len(results_df):,}")

# ============================================================================
# SHOW EXAMPLES
# ============================================================================

log("\n" + "=" * 70)
log("EXAMPLE: GOOD SIGNAL (profitable)")
log("=" * 70)

if example_good:
    ex = example_good[0]
    sig = ex['signal']

    log(f"\nEntry Time: {sig['entry_time']}")
    log(f"Exit Time:  {sig['exit_time']} ({sig['bars_held']} bars = {sig['bars_held']*30} min)")
    log(f"\nSignal Currency: {sig['ccy']}")
    log(f"Direction: {sig['direction'].upper()} (MFC {'coming UP from below' if sig['direction']=='strong' else 'coming DOWN from above'})")
    log(f"Entry MFC: {sig['entry_mfc']:.3f}")
    log(f"Entry Velocity: {sig['entry_vel']:.3f}")
    log(f"Exit MFC: {sig['exit_mfc']:.3f}")

    log(f"\nAll currencies at entry:")
    log(f"{'Currency':<6} {'MFC':>8} {'Velocity':>10}")
    log("-" * 28)
    for c in CURRENCIES:
        mfc = sig['all_mfc'][c]
        vel = sig['all_vel'][c]
        marker = " <-- SIGNAL" if c == sig['ccy'] else ""
        log(f"{c:<6} {mfc:>+8.3f} {vel:>+10.4f}{marker}")

    log(f"\nPrice results for all pairs with {sig['ccy']}:")
    log(f"{'Pair':<8} {'Other CCY':<6} {'Other MFC':>10} {'Expected':>8} {'Entry':>10} {'Exit':>10} {'Pips':>8}")
    log("-" * 75)
    for pr in sorted(ex['pair_results'], key=lambda x: -x['adjusted_pips']):
        log(f"{pr['pair']:<8} {pr['other_ccy']:<6} {pr['other_mfc']:>+10.3f} {pr['expected_dir']:>8} {pr['entry_price']:>10.5f} {pr['exit_price']:>10.5f} {pr['adjusted_pips']:>+8.1f}")

    log(f"\n--> Best pair: {ex['best']['pair']} with {ex['best']['adjusted_pips']:+.1f} pips")

log("\n" + "=" * 70)
log("EXAMPLE: BAD SIGNAL (losing)")
log("=" * 70)

if example_bad:
    ex = example_bad[0]
    sig = ex['signal']

    log(f"\nEntry Time: {sig['entry_time']}")
    log(f"Exit Time:  {sig['exit_time']} ({sig['bars_held']} bars = {sig['bars_held']*30} min)")
    log(f"\nSignal Currency: {sig['ccy']}")
    log(f"Direction: {sig['direction'].upper()} (MFC {'coming UP from below' if sig['direction']=='strong' else 'coming DOWN from above'})")
    log(f"Entry MFC: {sig['entry_mfc']:.3f}")
    log(f"Entry Velocity: {sig['entry_vel']:.3f}")
    log(f"Exit MFC: {sig['exit_mfc']:.3f}")

    log(f"\nAll currencies at entry:")
    log(f"{'Currency':<6} {'MFC':>8} {'Velocity':>10}")
    log("-" * 28)
    for c in CURRENCIES:
        mfc = sig['all_mfc'][c]
        vel = sig['all_vel'][c]
        marker = " <-- SIGNAL" if c == sig['ccy'] else ""
        log(f"{c:<6} {mfc:>+8.3f} {vel:>+10.4f}{marker}")

    log(f"\nPrice results for all pairs with {sig['ccy']}:")
    log(f"{'Pair':<8} {'Other CCY':<6} {'Other MFC':>10} {'Expected':>8} {'Entry':>10} {'Exit':>10} {'Pips':>8}")
    log("-" * 75)
    for pr in sorted(ex['pair_results'], key=lambda x: -x['adjusted_pips']):
        log(f"{pr['pair']:<8} {pr['other_ccy']:<6} {pr['other_mfc']:>+10.3f} {pr['expected_dir']:>8} {pr['entry_price']:>10.5f} {pr['exit_price']:>10.5f} {pr['adjusted_pips']:>+8.1f}")

    log(f"\n--> Even best pair lost: {ex['best']['pair']} with {ex['best']['adjusted_pips']:+.1f} pips")

# ============================================================================
# RESULTS - BEST PAIR PER SIGNAL
# ============================================================================

log("\n" + "=" * 70)
log("RESULTS: TRADING BEST PAIR PER SIGNAL")
log("=" * 70)

win_rate = (results_df['best_pips'] > 0).mean() * 100
avg_pips = results_df['best_pips'].mean()
total_pips = results_df['best_pips'].sum()

log(f"\nIf we could pick the best pair each time:")
log(f"  Signals: {len(results_df):,}")
log(f"  Win rate: {win_rate:.1f}%")
log(f"  Avg pips: {avg_pips:+.2f}")
log(f"  Total pips: {total_pips:+,.0f}")

# Worst pair
log(f"\nIf we picked the worst pair each time:")
log(f"  Win rate: {(results_df['worst_pips'] > 0).mean()*100:.1f}%")
log(f"  Avg pips: {results_df['worst_pips'].mean():+.2f}")

# ============================================================================
# WHAT MAKES A GOOD PAIR SELECTION?
# ============================================================================

log("\n" + "=" * 70)
log("WHAT MAKES A GOOD PAIR? (Other currency analysis)")
log("=" * 70)

# When other currency has opposite momentum
results_df['other_opposite'] = (
    ((results_df['direction'] == 'strong') & (results_df['best_other_vel'] < -0.03)) |
    ((results_df['direction'] == 'weak') & (results_df['best_other_vel'] > 0.03))
)

results_df['other_same'] = (
    ((results_df['direction'] == 'strong') & (results_df['best_other_vel'] > 0.03)) |
    ((results_df['direction'] == 'weak') & (results_df['best_other_vel'] < -0.03))
)

log(f"\n{'Other CCY State':<20} {'Signals':>10} {'Win %':>10} {'Avg Pips':>12}")
log("-" * 55)

opp = results_df[results_df['other_opposite']]
log(f"{'OPPOSITE momentum':<20} {len(opp):>10,} {(opp['best_pips']>0).mean()*100:>9.1f}% {opp['best_pips'].mean():>+12.2f}")

same = results_df[results_df['other_same']]
log(f"{'SAME momentum':<20} {len(same):>10,} {(same['best_pips']>0).mean()*100:>9.1f}% {same['best_pips'].mean():>+12.2f}")

neutral = results_df[~results_df['other_opposite'] & ~results_df['other_same']]
log(f"{'NEUTRAL':<20} {len(neutral):>10,} {(neutral['best_pips']>0).mean()*100:>9.1f}% {neutral['best_pips'].mean():>+12.2f}")

# ============================================================================
# BY VELOCITY
# ============================================================================

log("\n" + "=" * 70)
log("BY ENTRY VELOCITY (best pair)")
log("=" * 70)

log(f"\n{'Velocity':<15} {'Signals':>10} {'Win %':>10} {'Avg Pips':>12} {'Total':>12}")
log("-" * 65)

for low, high, label in [(0.05, 0.08, '0.05-0.08'), (0.08, 0.12, '0.08-0.12'), (0.12, 0.20, '0.12-0.20'), (0.20, 1.0, '>0.20')]:
    subset = results_df[(results_df['entry_vel'] >= low) & (results_df['entry_vel'] < high)]
    if len(subset) >= 100:
        wr = (subset['best_pips'] > 0).mean() * 100
        avg = subset['best_pips'].mean()
        total = subset['best_pips'].sum()
        log(f"{label:<15} {len(subset):>10,} {wr:>9.1f}% {avg:>+12.2f} {total:>+12,.0f}")

# ============================================================================
# REALISTIC: PICK PAIR WITH STRONGEST OPPOSITE MOMENTUM
# ============================================================================

log("\n" + "=" * 70)
log("REALISTIC: PICK PAIR WHERE OTHER CCY HAS STRONGEST OPPOSITE MOMENTUM")
log("=" * 70)

def pick_best_opposite_pair(row):
    """Pick pair where other currency has strongest opposite momentum."""
    direction = row['direction']
    all_pairs = row['all_pairs']

    best_score = -999
    best_pair = None
    best_pips = 0

    for pr in all_pairs:
        other_vel = pr['other_vel']

        # Score: how strongly opposite is the other currency?
        if direction == 'strong':
            # We want other currency to be weak (negative velocity)
            score = -other_vel
        else:
            # We want other currency to be strong (positive velocity)
            score = other_vel

        if score > best_score:
            best_score = score
            best_pair = pr['pair']
            best_pips = pr['adjusted_pips']

    return pd.Series({'picked_pair': best_pair, 'picked_pips': best_pips, 'opposite_score': best_score})

picked = results_df.apply(pick_best_opposite_pair, axis=1)
results_df['picked_pair'] = picked['picked_pair']
results_df['picked_pips'] = picked['picked_pips']
results_df['opposite_score'] = picked['opposite_score']

log(f"\nPicking pair with strongest opposite momentum:")
log(f"  Signals: {len(results_df):,}")
log(f"  Win rate: {(results_df['picked_pips'] > 0).mean()*100:.1f}%")
log(f"  Avg pips: {results_df['picked_pips'].mean():+.2f}")
log(f"  Total pips: {results_df['picked_pips'].sum():+,.0f}")

# Filter to only strong opposite scores
strong_opp = results_df[results_df['opposite_score'] > 0.03]
log(f"\nOnly when other currency has strong opposite (score > 0.03):")
log(f"  Signals: {len(strong_opp):,}")
log(f"  Win rate: {(strong_opp['picked_pips'] > 0).mean()*100:.1f}%")
log(f"  Avg pips: {strong_opp['picked_pips'].mean():+.2f}")
log(f"  Total pips: {strong_opp['picked_pips'].sum():+,.0f}")

# By velocity for strong opposite
log(f"\n{'Velocity':<15} {'Signals':>10} {'Win %':>10} {'Avg Pips':>12}")
log("-" * 50)
for low, high, label in [(0.05, 0.08, '0.05-0.08'), (0.08, 0.12, '0.08-0.12'), (0.12, 0.20, '0.12-0.20'), (0.20, 1.0, '>0.20')]:
    subset = strong_opp[(strong_opp['entry_vel'] >= low) & (strong_opp['entry_vel'] < high)]
    if len(subset) >= 50:
        wr = (subset['picked_pips'] > 0).mean() * 100
        avg = subset['picked_pips'].mean()
        log(f"{label:<15} {len(subset):>10,} {wr:>9.1f}% {avg:>+12.2f}")

log("\n" + "=" * 70)
log("CONCLUSION")
log("=" * 70)
log()
