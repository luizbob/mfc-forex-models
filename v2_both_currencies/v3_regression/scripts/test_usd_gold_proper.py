"""
Test USD MFC Velocity vs Gold - PROPER Entry/Exit
==================================================
Entry: Next bar open after USD velocity signal
Exit: USD velocity drops OR fixed holding period
"""
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from pathlib import Path

def log(msg=""):
    print(msg, flush=True)

log("=" * 70)
log("USD MFC vs GOLD - PROPER TRADING TEST")
log("=" * 70)

DATA_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/data')
CLEANED_DIR = DATA_DIR / 'cleaned'

# Parameters
VELOCITY_THRESHOLD = 0.05
EXIT_THRESHOLD = 0.02  # Exit when USD velocity drops
MAX_HOLD_BARS = 6  # Max 6 H4 bars = 24 hours
GOLD_SPREAD = 30  # Gold spread in pips (0.1 per pip)

def load_mfc(currency, timeframe):
    fp = CLEANED_DIR / f'mfc_currency_{currency}_{timeframe}_clean.csv'
    if not fp.exists():
        return None
    df = pd.read_csv(fp)
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df = df.set_index('datetime').sort_index()
    return df['MFC']

# Load USD MFC H4
log("\nLoading data...")
usd_h4 = load_mfc('USD', 'H4')
usd_vel_raw = usd_h4.diff()  # Raw for exit tracking
usd_vel_shifted = usd_h4.diff().shift(1)  # Shifted for entry detection

# Load Gold H4
gold_fp = DATA_DIR / 'XAUUSDm_M5_202110271720_202501172155.csv'
gold_df = pd.read_csv(gold_fp, sep='\t')
gold_df.columns = ['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'TickVol', 'Vol', 'Spread']
gold_df['datetime'] = pd.to_datetime(gold_df['Date'].str.replace('.', '-', regex=False) + ' ' + gold_df['Time'])
gold_df = gold_df.set_index('datetime').sort_index()

gold_h4 = gold_df.resample('4h').agg({
    'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
}).dropna()

log(f"USD H4: {len(usd_h4):,} bars")
log(f"Gold H4: {len(gold_h4):,} bars")

# Align data
df = pd.DataFrame(index=gold_h4.index)
df['gold_open'] = gold_h4['Open']
df['gold_close'] = gold_h4['Close']
df['usd_vel_shifted'] = usd_vel_shifted.reindex(df.index, method='ffill')
df['usd_vel_raw'] = usd_vel_raw.reindex(df.index, method='ffill')
df = df.dropna()

log(f"Aligned data: {len(df):,} bars")
log(f"Date range: {df.index.min()} to {df.index.max()}")

# Find trades with proper entry/exit
all_trades = []

gold_open = df['gold_open'].values
gold_close = df['gold_close'].values
usd_vel_shifted = df['usd_vel_shifted'].values
usd_vel_raw = df['usd_vel_raw'].values
dates = df.index

n = len(df)
i = 0

while i < n - MAX_HOLD_BARS - 1:
    vel = usd_vel_shifted[i]

    signal = None

    # USD strengthening fast -> Sell Gold
    if vel > VELOCITY_THRESHOLD:
        signal = 'sell'
    # USD weakening fast -> Buy Gold
    elif vel < -VELOCITY_THRESHOLD:
        signal = 'buy'

    if signal is None:
        i += 1
        continue

    # Entry at next bar open
    entry_idx = i + 1
    entry_price = gold_open[entry_idx]

    # Find exit: USD velocity drops OR max bars
    exit_idx = entry_idx

    for j in range(entry_idx, min(entry_idx + MAX_HOLD_BARS, n)):
        current_usd_vel = usd_vel_raw[j]

        # Exit when USD velocity drops below threshold
        if signal == 'sell':
            # We sold Gold because USD was going up
            # Exit when USD stops going up
            if current_usd_vel < EXIT_THRESHOLD:
                exit_idx = j
                break
        else:
            # We bought Gold because USD was going down
            # Exit when USD stops going down
            if current_usd_vel > -EXIT_THRESHOLD:
                exit_idx = j
                break

        exit_idx = j

    # Calculate PnL
    exit_price = gold_close[exit_idx]
    bars_held = exit_idx - entry_idx + 1

    # Gold pip = 0.1
    raw_pips = (exit_price - entry_price) / 0.1

    if signal == 'sell':
        adjusted_pips = -raw_pips  # Sell = inverse
    else:
        adjusted_pips = raw_pips

    net_pips = adjusted_pips - GOLD_SPREAD

    all_trades.append({
        'datetime': dates[i],
        'signal': signal,
        'entry_price': entry_price,
        'exit_price': exit_price,
        'bars_held': bars_held,
        'raw_pips': raw_pips,
        'adjusted_pips': adjusted_pips,
        'net_pips': net_pips,
        'is_profitable': int(net_pips > 0),
        'usd_vel': vel,
    })

    # Skip to after exit
    i = exit_idx + 1

trades_df = pd.DataFrame(all_trades)
trades_df['datetime'] = pd.to_datetime(trades_df['datetime'])
trades_df['year'] = trades_df['datetime'].dt.year

log(f"\nTotal trades: {len(trades_df):,}")

# Results
log("\n" + "=" * 70)
log("OVERALL RESULTS")
log("=" * 70)

log(f"\nTotal trades: {len(trades_df):,}")
log(f"Win rate: {trades_df['is_profitable'].mean()*100:.1f}%")
log(f"Avg net pips: {trades_df['net_pips'].mean():+.1f}")
log(f"Total net pips: {trades_df['net_pips'].sum():+,.0f}")
log(f"Avg bars held: {trades_df['bars_held'].mean():.1f}")

# By signal type
log("\n" + "-" * 50)
sells = trades_df[trades_df['signal'] == 'sell']
buys = trades_df[trades_df['signal'] == 'buy']

log(f"\nSELL Gold (USD up):")
log(f"  Trades: {len(sells):,}")
log(f"  Win rate: {sells['is_profitable'].mean()*100:.1f}%")
log(f"  Avg pips: {sells['net_pips'].mean():+.1f}")
log(f"  Total: {sells['net_pips'].sum():+,.0f}")

log(f"\nBUY Gold (USD down):")
log(f"  Trades: {len(buys):,}")
log(f"  Win rate: {buys['is_profitable'].mean()*100:.1f}%")
log(f"  Avg pips: {buys['net_pips'].mean():+.1f}")
log(f"  Total: {buys['net_pips'].sum():+,.0f}")

# By bars held
log("\n" + "=" * 70)
log("BY BARS HELD")
log("=" * 70)

log(f"\n{'Bars':<10} {'Trades':>8} {'Win %':>10} {'Avg Pips':>12}")
log("-" * 45)
for low, high, label in [(1, 2, '1'), (2, 3, '2'), (3, 4, '3'), (4, 7, '4-6')]:
    subset = trades_df[(trades_df['bars_held'] >= low) & (trades_df['bars_held'] < high)]
    if len(subset) > 10:
        wr = subset['is_profitable'].mean() * 100
        avg = subset['net_pips'].mean()
        log(f"{label:<10} {len(subset):>8,} {wr:>9.1f}% {avg:>+12.1f}")

# By year
log("\n" + "=" * 70)
log("BY YEAR")
log("=" * 70)

log(f"\n{'Year':<8} {'Trades':>8} {'Win %':>10} {'Avg Pips':>12} {'Total':>12}")
log("-" * 55)
for year in sorted(trades_df['year'].unique()):
    subset = trades_df[trades_df['year'] == year]
    wr = subset['is_profitable'].mean() * 100
    avg = subset['net_pips'].mean()
    total = subset['net_pips'].sum()
    log(f"{year:<8} {len(subset):>8,} {wr:>9.1f}% {avg:>+12.1f} {total:>+12,.0f}")

# By USD velocity magnitude
log("\n" + "=" * 70)
log("BY USD VELOCITY MAGNITUDE")
log("=" * 70)

log(f"\n{'Velocity':<15} {'Trades':>8} {'Win %':>10} {'Avg Pips':>12}")
log("-" * 50)
for low, high, label in [(0.05, 0.08, '0.05-0.08'), (0.08, 0.12, '0.08-0.12'), (0.12, 0.20, '0.12-0.20'), (0.20, 1.0, '>0.20')]:
    subset = trades_df[(abs(trades_df['usd_vel']) >= low) & (abs(trades_df['usd_vel']) < high)]
    if len(subset) > 10:
        wr = subset['is_profitable'].mean() * 100
        avg = subset['net_pips'].mean()
        log(f"{label:<15} {len(subset):>8,} {wr:>9.1f}% {avg:>+12.1f}")

# Compare with random baseline
log("\n" + "=" * 70)
log("BASELINE COMPARISON")
log("=" * 70)

# All possible H4 bars as trades (random entry)
baseline_trades = len(df) - MAX_HOLD_BARS
log(f"\nIf we traded every H4 bar randomly:")
log(f"  Expected win rate: ~50%")
log(f"  Expected avg pips: -{GOLD_SPREAD} (just spread)")

log(f"\nOur strategy:")
log(f"  Win rate: {trades_df['is_profitable'].mean()*100:.1f}%")
log(f"  Avg pips: {trades_df['net_pips'].mean():+.1f}")
log(f"  Edge over random: {trades_df['net_pips'].mean() + GOLD_SPREAD:+.1f} pips per trade")

log("\nDONE")
