"""
Test: Momentum Strategy with Proper MFC Shift
==============================================
Use shift(1) on MFC - at time T we see MFC from T-1.
This avoids look-ahead bias.
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
log("MOMENTUM STRATEGY WITH SHIFTED MFC")
log("Using shift(1) - at time T we see MFC from T-1")
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
    return df

log("\nLoading data...")

# Load MFC and apply shift(1)
mfc_data = {}
for ccy in CURRENCIES:
    mfc_data[ccy] = load_mfc(ccy, 'M30')

mfc_df = pd.DataFrame(index=mfc_data['EUR'].index)
for ccy in CURRENCIES:
    if mfc_data[ccy] is not None:
        # SHIFT(1) - use previous bar's MFC value
        mfc_df[ccy] = mfc_data[ccy].shift(1)
        mfc_df[f'{ccy}_vel'] = mfc_df[ccy].diff()

mfc_df = mfc_df.dropna()

# Load price
price_data = {}
for pair in PAIRS:
    price_data[pair] = load_price(pair, 'M30')

log(f"MFC (shifted): {len(mfc_df):,} bars")
log(f"Price data: {len([p for p in PAIRS if price_data[p] is not None])} pairs")

# ============================================================================
# FIND MOMENTUM SIGNALS
# ============================================================================

def find_signals(mfc_df, ccy, vel_threshold=0.05):
    """Find momentum entry/exit signals."""
    values = mfc_df[ccy].values
    velocities = mfc_df[f'{ccy}_vel'].values
    times = mfc_df.index

    signals = []
    i = 1

    while i < len(mfc_df) - 1:
        # Entry from ABOVE (MFC coming down = currency weakening)
        if values[i-1] > BOX_UPPER and values[i] <= BOX_UPPER and velocities[i] < -vel_threshold:
            entry_idx = i
            direction = 'weak'

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
                    'entry_vel': velocities[entry_idx],
                    'entry_mfc': values[entry_idx],
                    'exit_mfc': values[j],
                    'all_mfc': {c: mfc_df[c].iloc[entry_idx] for c in CURRENCIES},
                    'all_vel': {c: mfc_df[f'{c}_vel'].iloc[entry_idx] for c in CURRENCIES},
                })
                i = j + 1
                continue

        # Entry from BELOW (MFC coming up = currency strengthening)
        if values[i-1] < BOX_LOWER and values[i] >= BOX_LOWER and velocities[i] > vel_threshold:
            entry_idx = i
            direction = 'strong'

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
                    'entry_vel': velocities[entry_idx],
                    'entry_mfc': values[entry_idx],
                    'exit_mfc': values[j],
                    'all_mfc': {c: mfc_df[c].iloc[entry_idx] for c in CURRENCIES},
                    'all_vel': {c: mfc_df[f'{c}_vel'].iloc[entry_idx] for c in CURRENCIES},
                })
                i = j + 1
                continue

        i += 1

    return signals

log("\nFinding signals...")
all_signals = []
for ccy in CURRENCIES:
    signals = find_signals(mfc_df, ccy)
    all_signals.extend(signals)
    log(f"  {ccy}: {len(signals)} signals")

log(f"\nTotal: {len(all_signals)} signals")

# ============================================================================
# MEASURE PRICE MOVEMENT
# ============================================================================

def get_price_move(pair, entry_time, exit_time):
    if price_data[pair] is None:
        return None, None, None

    pdf = price_data[pair]
    try:
        # Get entry price
        if entry_time in pdf.index:
            entry_price = pdf.loc[entry_time, 'Close']
        else:
            idx = pdf.index.get_indexer([entry_time], method='nearest')[0]
            entry_price = pdf.iloc[idx]['Close']

        # Get exit price
        if exit_time in pdf.index:
            exit_price = pdf.loc[exit_time, 'Close']
        else:
            idx = pdf.index.get_indexer([exit_time], method='nearest')[0]
            exit_price = pdf.iloc[idx]['Close']

        pip_move = (exit_price - entry_price) / PIP_SIZE[pair]
        return entry_price, exit_price, pip_move
    except:
        return None, None, None

log("\n" + "=" * 70)
log("ANALYZING SIGNALS WITH PRICE")
log("=" * 70)

results = []

for sig in all_signals:
    ccy = sig['ccy']
    direction = sig['direction']

    for pair in PAIRS:
        base = pair[:3]
        quote = pair[3:]

        if ccy not in [base, quote]:
            continue

        entry_price, exit_price, pip_move = get_price_move(pair, sig['entry_time'], sig['exit_time'])
        if pip_move is None:
            continue

        # Expected direction
        if ccy == base:
            expected_sign = 1 if direction == 'strong' else -1
        else:
            expected_sign = -1 if direction == 'strong' else 1

        adjusted_pips = pip_move * expected_sign

        # Other currency
        other_ccy = quote if ccy == base else base
        other_vel = sig['all_vel'].get(other_ccy, 0)
        other_mfc = sig['all_mfc'].get(other_ccy, 0)

        # Check agreement
        if direction == 'strong':
            other_agrees = other_vel < -0.03  # Other should be weak
        else:
            other_agrees = other_vel > 0.03  # Other should be strong

        results.append({
            'entry_time': sig['entry_time'],
            'ccy': ccy,
            'direction': direction,
            'entry_vel': abs(sig['entry_vel']),
            'entry_mfc': sig['entry_mfc'],
            'pair': pair,
            'other_ccy': other_ccy,
            'other_vel': other_vel,
            'other_mfc': other_mfc,
            'other_agrees': other_agrees,
            'pip_move': pip_move,
            'adjusted_pips': adjusted_pips,
            'win': adjusted_pips > 0,
        })

results_df = pd.DataFrame(results)
log(f"\nTotal trades: {len(results_df):,}")

# ============================================================================
# OVERALL RESULTS
# ============================================================================

log("\n" + "=" * 70)
log("OVERALL RESULTS (with shifted MFC)")
log("=" * 70)

win_rate = results_df['win'].mean() * 100
avg_pips = results_df['adjusted_pips'].mean()
total_pips = results_df['adjusted_pips'].sum()

log(f"\nAll trades:")
log(f"  Win rate: {win_rate:.1f}%")
log(f"  Avg pips: {avg_pips:+.2f}")
log(f"  Total pips: {total_pips:+,.0f}")

# ============================================================================
# BY AGREEMENT
# ============================================================================

log("\n" + "=" * 70)
log("BY CURRENCY AGREEMENT")
log("=" * 70)

agree = results_df[results_df['other_agrees']]
disagree = results_df[~results_df['other_agrees']]

log(f"\n{'Scenario':<20} {'Trades':>10} {'Win %':>10} {'Avg Pips':>12}")
log("-" * 55)
log(f"{'AGREE':<20} {len(agree):>10,} {agree['win'].mean()*100:>9.1f}% {agree['adjusted_pips'].mean():>+12.2f}")
log(f"{'DISAGREE/NEUTRAL':<20} {len(disagree):>10,} {disagree['win'].mean()*100:>9.1f}% {disagree['adjusted_pips'].mean():>+12.2f}")

# ============================================================================
# BY VELOCITY
# ============================================================================

log("\n" + "=" * 70)
log("BY ENTRY VELOCITY")
log("=" * 70)

log(f"\n{'Velocity':<15} {'Trades':>10} {'Win %':>10} {'Avg Pips':>12}")
log("-" * 50)

for low, high, label in [(0.05, 0.08, '0.05-0.08'), (0.08, 0.12, '0.08-0.12'), (0.12, 0.20, '0.12-0.20'), (0.20, 1.0, '>0.20')]:
    subset = results_df[(results_df['entry_vel'] >= low) & (results_df['entry_vel'] < high)]
    if len(subset) >= 100:
        wr = subset['win'].mean() * 100
        avg = subset['adjusted_pips'].mean()
        log(f"{label:<15} {len(subset):>10,} {wr:>9.1f}% {avg:>+12.2f}")

# ============================================================================
# BY POSITION (where MFC is when signal triggers)
# ============================================================================

log("\n" + "=" * 70)
log("BY MFC POSITION AT ENTRY")
log("=" * 70)

log(f"\n{'MFC Position':<20} {'Trades':>10} {'Win %':>10} {'Avg Pips':>12}")
log("-" * 55)

# For WEAK signals (coming from above)
weak = results_df[results_df['direction'] == 'weak']
for low, high, label in [(0.15, 0.20, 'Just entered box'), (0.0, 0.15, 'Middle of box'), (-0.20, 0.0, 'Near bottom')]:
    subset = weak[(weak['entry_mfc'] >= low) & (weak['entry_mfc'] < high)]
    if len(subset) >= 100:
        wr = subset['win'].mean() * 100
        avg = subset['adjusted_pips'].mean()
        log(f"WEAK {label:<15} {len(subset):>10,} {wr:>9.1f}% {avg:>+12.2f}")

log()

# For STRONG signals (coming from below)
strong = results_df[results_df['direction'] == 'strong']
for low, high, label in [(-0.20, -0.15, 'Just entered box'), (-0.15, 0.0, 'Middle of box'), (0.0, 0.20, 'Near top')]:
    subset = strong[(strong['entry_mfc'] >= low) & (strong['entry_mfc'] < high)]
    if len(subset) >= 100:
        wr = subset['win'].mean() * 100
        avg = subset['adjusted_pips'].mean()
        log(f"STRONG {label:<13} {len(subset):>10,} {wr:>9.1f}% {avg:>+12.2f}")

# ============================================================================
# BEST COMBINATION
# ============================================================================

log("\n" + "=" * 70)
log("BEST COMBINATION: HIGH VELOCITY + AGREEMENT + POSITION")
log("=" * 70)

# High velocity + agreement
best = results_df[
    (results_df['entry_vel'] >= 0.12) &
    (results_df['other_agrees'])
]

log(f"\nHigh velocity (>=0.12) + Other agrees:")
log(f"  Trades: {len(best):,}")
if len(best) > 0:
    log(f"  Win rate: {best['win'].mean()*100:.1f}%")
    log(f"  Avg pips: {best['adjusted_pips'].mean():+.2f}")
    log(f"  Total pips: {best['adjusted_pips'].sum():+,.0f}")

# Very high velocity only
very_high = results_df[results_df['entry_vel'] >= 0.20]
log(f"\nVery high velocity (>=0.20) only:")
log(f"  Trades: {len(very_high):,}")
if len(very_high) > 0:
    log(f"  Win rate: {very_high['win'].mean()*100:.1f}%")
    log(f"  Avg pips: {very_high['adjusted_pips'].mean():+.2f}")

# ============================================================================
# HOLDOUT
# ============================================================================

log("\n" + "=" * 70)
log("HOLDOUT (Last 3 Months)")
log("=" * 70)

holdout_start = mfc_df.index.max() - pd.DateOffset(months=3)
holdout = results_df[results_df['entry_time'] >= holdout_start]

log(f"\nHoldout trades: {len(holdout):,}")
if len(holdout) > 0:
    log(f"Win rate: {holdout['win'].mean()*100:.1f}%")
    log(f"Avg pips: {holdout['adjusted_pips'].mean():+.2f}")
    log(f"Total pips: {holdout['adjusted_pips'].sum():+,.0f}")

    # By agreement in holdout
    h_agree = holdout[holdout['other_agrees']]
    h_disagree = holdout[~holdout['other_agrees']]
    log(f"\n  Agree: {len(h_agree):,} trades, {h_agree['win'].mean()*100:.1f}% WR, {h_agree['adjusted_pips'].mean():+.2f} avg")
    log(f"  Disagree: {len(h_disagree):,} trades, {h_disagree['win'].mean()*100:.1f}% WR, {h_disagree['adjusted_pips'].mean():+.2f} avg")

log("\n" + "=" * 70)
log("CONCLUSION")
log("=" * 70)
log()
