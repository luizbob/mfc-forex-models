"""
Test: Momentum Surfing Strategy
================================
Entry: MFC crosses into box from extreme with velocity > threshold
Direction: Trade WITH the momentum (not against it)
Exit: When velocity falls below threshold

- MFC coming DOWN from above box → SELL that currency
- MFC coming UP from below box → BUY that currency
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

VELOCITY_THRESHOLD = 0.05

log("=" * 70)
log("MOMENTUM SURFING STRATEGY TEST")
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
        df[f'{ccy}_vel'] = df[ccy].diff()

df = df.dropna()
log(f"\nAligned: {len(df):,} M30 bars ({df.index[0].date()} to {df.index[-1].date()})")

# ============================================================================
# FIND MOMENTUM SURF TRADES
# ============================================================================

log("\n" + "=" * 70)
log(f"MOMENTUM SURF TRADES (velocity threshold: {VELOCITY_THRESHOLD})")
log("=" * 70)

def find_momentum_trades(df, ccy, vel_threshold=0.05):
    """
    Find momentum surf trades:
    - Entry: MFC enters box from extreme with velocity > threshold
    - Exit: Velocity drops below threshold
    """
    values = df[ccy].values
    velocities = df[f'{ccy}_vel'].values
    times = df.index

    trades = []
    i = 1

    while i < len(df) - 1:
        # Check for entry from ABOVE (MFC coming down)
        prev_above = values[i-1] > BOX_UPPER
        now_in_or_below = values[i] <= BOX_UPPER
        vel_down = velocities[i] < -vel_threshold

        if prev_above and now_in_or_below and vel_down:
            # SELL signal - momentum going down
            entry_idx = i
            entry_val = values[i]
            entry_vel = velocities[i]
            direction = 'sell'

            # Hold until velocity weakens
            j = i + 1
            while j < len(df):
                # Exit when velocity drops below threshold (less negative)
                if velocities[j] > -vel_threshold:
                    break
                # Also exit if MFC reverses direction
                if velocities[j] > 0:
                    break
                j += 1

            if j < len(df):
                exit_idx = j
                exit_val = values[j]
                mfc_move = entry_val - exit_val  # Positive = good for sell
                bars_held = exit_idx - entry_idx

                trades.append({
                    'entry_time': times[entry_idx],
                    'exit_time': times[exit_idx],
                    'ccy': ccy,
                    'direction': direction,
                    'entry_mfc': entry_val,
                    'exit_mfc': exit_val,
                    'entry_vel': entry_vel,
                    'mfc_move': mfc_move,
                    'bars_held': bars_held,
                })

                i = j + 1
                continue

        # Check for entry from BELOW (MFC coming up)
        prev_below = values[i-1] < BOX_LOWER
        now_in_or_above = values[i] >= BOX_LOWER
        vel_up = velocities[i] > vel_threshold

        if prev_below and now_in_or_above and vel_up:
            # BUY signal - momentum going up
            entry_idx = i
            entry_val = values[i]
            entry_vel = velocities[i]
            direction = 'buy'

            # Hold until velocity weakens
            j = i + 1
            while j < len(df):
                # Exit when velocity drops below threshold
                if velocities[j] < vel_threshold:
                    break
                # Also exit if MFC reverses direction
                if velocities[j] < 0:
                    break
                j += 1

            if j < len(df):
                exit_idx = j
                exit_val = values[j]
                mfc_move = exit_val - entry_val  # Positive = good for buy
                bars_held = exit_idx - entry_idx

                trades.append({
                    'entry_time': times[entry_idx],
                    'exit_time': times[exit_idx],
                    'ccy': ccy,
                    'direction': direction,
                    'entry_mfc': entry_val,
                    'exit_mfc': exit_val,
                    'entry_vel': entry_vel,
                    'mfc_move': mfc_move,
                    'bars_held': bars_held,
                })

                i = j + 1
                continue

        i += 1

    return pd.DataFrame(trades) if trades else None

# Run for all currencies
all_trades = []
log(f"\n{'Currency':<8} {'Trades':>8} {'Avg MFC Move':>12} {'Win %':>8} {'Avg Bars':>10}")
log("-" * 50)

for ccy in CURRENCIES:
    trades = find_momentum_trades(df, ccy, VELOCITY_THRESHOLD)
    if trades is not None and len(trades) > 0:
        avg_move = trades['mfc_move'].mean()
        win_rate = (trades['mfc_move'] > 0).mean() * 100
        avg_bars = trades['bars_held'].mean()

        log(f"{ccy:<8} {len(trades):>8} {avg_move:>+12.4f} {win_rate:>7.1f}% {avg_bars:>10.1f}")
        all_trades.append(trades)

if all_trades:
    combined = pd.concat(all_trades)
    log("-" * 50)
    log(f"{'TOTAL':<8} {len(combined):>8} {combined['mfc_move'].mean():>+12.4f} {(combined['mfc_move'] > 0).mean()*100:>7.1f}% {combined['bars_held'].mean():>10.1f}")

# ============================================================================
# BY DIRECTION
# ============================================================================

log("\n" + "=" * 70)
log("BY DIRECTION")
log("=" * 70)

if all_trades:
    combined = pd.concat(all_trades)

    for direction in ['buy', 'sell']:
        subset = combined[combined['direction'] == direction]
        if len(subset) > 0:
            avg_move = subset['mfc_move'].mean()
            win_rate = (subset['mfc_move'] > 0).mean() * 100
            avg_bars = subset['bars_held'].mean()
            log(f"\n{direction.upper()}: {len(subset):,} trades")
            log(f"  Win rate: {win_rate:.1f}%")
            log(f"  Avg MFC move: {avg_move:+.4f}")
            log(f"  Avg bars held: {avg_bars:.1f}")

# ============================================================================
# BY ENTRY VELOCITY
# ============================================================================

log("\n" + "=" * 70)
log("BY ENTRY VELOCITY STRENGTH")
log("=" * 70)

if all_trades:
    combined = pd.concat(all_trades)
    combined['abs_vel'] = combined['entry_vel'].abs()

    log(f"\n{'Entry Velocity':<20} {'Trades':>8} {'Win %':>8} {'Avg Move':>12} {'Avg Bars':>10}")
    log("-" * 65)

    for low, high, label in [(0.05, 0.08, '0.05-0.08'), (0.08, 0.12, '0.08-0.12'), (0.12, 0.20, '0.12-0.20'), (0.20, 1.0, '>0.20')]:
        subset = combined[(combined['abs_vel'] >= low) & (combined['abs_vel'] < high)]
        if len(subset) >= 10:
            win_rate = (subset['mfc_move'] > 0).mean() * 100
            avg_move = subset['mfc_move'].mean()
            avg_bars = subset['bars_held'].mean()
            log(f"{label:<20} {len(subset):>8} {win_rate:>7.1f}% {avg_move:>+12.4f} {avg_bars:>10.1f}")

# ============================================================================
# VARY VELOCITY THRESHOLD
# ============================================================================

log("\n" + "=" * 70)
log("VARY VELOCITY THRESHOLD")
log("=" * 70)

log(f"\n{'Threshold':<12} {'Trades':>8} {'Win %':>8} {'Avg Move':>12} {'Total Move':>12}")
log("-" * 55)

for thresh in [0.03, 0.04, 0.05, 0.06, 0.08, 0.10]:
    all_t = []
    for ccy in CURRENCIES:
        trades = find_momentum_trades(df, ccy, thresh)
        if trades is not None and len(trades) > 0:
            all_t.append(trades)

    if all_t:
        combined = pd.concat(all_t)
        win_rate = (combined['mfc_move'] > 0).mean() * 100
        avg_move = combined['mfc_move'].mean()
        total_move = combined['mfc_move'].sum()
        log(f"{thresh:<12} {len(combined):>8} {win_rate:>7.1f}% {avg_move:>+12.4f} {total_move:>+12.2f}")

# ============================================================================
# HOLDOUT TEST (last 3 months)
# ============================================================================

log("\n" + "=" * 70)
log("HOLDOUT TEST (Last 3 months)")
log("=" * 70)

holdout_start = df.index.max() - pd.DateOffset(months=3)
df_holdout = df[df.index >= holdout_start]

log(f"\nHoldout period: {df_holdout.index[0].date()} to {df_holdout.index[-1].date()}")
log(f"Holdout bars: {len(df_holdout):,}")

holdout_trades = []
for ccy in CURRENCIES:
    # Create subset dataframe for holdout
    trades = find_momentum_trades(df_holdout, ccy, VELOCITY_THRESHOLD)
    if trades is not None and len(trades) > 0:
        holdout_trades.append(trades)

if holdout_trades:
    combined = pd.concat(holdout_trades)
    log(f"\nHoldout trades: {len(combined):,}")
    log(f"Win rate: {(combined['mfc_move'] > 0).mean()*100:.1f}%")
    log(f"Avg MFC move: {combined['mfc_move'].mean():+.4f}")
    log(f"Total MFC move: {combined['mfc_move'].sum():+.2f}")

    log(f"\n{'Currency':<8} {'Trades':>8} {'Win %':>8} {'Avg Move':>12}")
    log("-" * 40)
    for ccy in CURRENCIES:
        ccy_trades = combined[combined['ccy'] == ccy]
        if len(ccy_trades) > 0:
            win_rate = (ccy_trades['mfc_move'] > 0).mean() * 100
            avg_move = ccy_trades['mfc_move'].mean()
            log(f"{ccy:<8} {len(ccy_trades):>8} {win_rate:>7.1f}% {avg_move:>+12.4f}")

# ============================================================================
# CONCLUSION
# ============================================================================

log("\n" + "=" * 70)
log("CONCLUSION")
log("=" * 70)

if all_trades:
    combined = pd.concat(all_trades)
    win_rate = (combined['mfc_move'] > 0).mean() * 100
    avg_move = combined['mfc_move'].mean()

    log(f"\nMomentum Surfing Strategy:")
    log(f"  Total trades: {len(combined):,}")
    log(f"  Win rate: {win_rate:.1f}%")
    log(f"  Avg MFC move per trade: {avg_move:+.4f}")

    if win_rate > 55:
        log(f"\n  ✓ Strategy shows positive edge")
    elif win_rate < 45:
        log(f"\n  ✗ Strategy shows negative edge")
    else:
        log(f"\n  ? Inconclusive")

log()
