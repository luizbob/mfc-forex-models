"""
Test: Velocity + MFC Position
==============================
Does position matter? E.g., D1 velocity UP from BELOW the box.
Check all combinations of velocity direction + MFC position.
"""
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from pathlib import Path

def log(msg=""):
    print(msg, flush=True)

log("=" * 70)
log("VELOCITY + MFC POSITION ANALYSIS")
log("=" * 70)

DATA_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/data')
CLEANED_DIR = DATA_DIR / 'cleaned'

CURRENCIES = ['USD', 'EUR', 'GBP', 'JPY', 'AUD', 'CAD', 'CHF', 'NZD']
TIMEFRAMES = ['H1', 'H4', 'D1', 'W1', 'MN']

def load_mfc(currency, timeframe):
    fp = CLEANED_DIR / f'mfc_currency_{currency}_{timeframe}_clean.csv'
    if not fp.exists():
        return None
    df = pd.read_csv(fp)
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df = df.set_index('datetime').sort_index()
    df = df[(df.index >= '2023-01-01') & (df.index <= '2025-12-31')]
    return df['MFC']

def load_price(pair):
    fp = DATA_DIR / f'{pair}_GMT+0_US-DST_M30.csv'
    if fp.exists():
        with open(fp, 'r') as f:
            first_line = f.readline()
        if 'Date' in first_line or 'Open' in first_line:
            df = pd.read_csv(fp)
        else:
            df = pd.read_csv(fp, header=None,
                           names=['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
        df = df.set_index('datetime').sort_index()
        df = df[(df.index >= '2023-01-01') & (df.index <= '2025-12-31')]
        return df
    return None

# Load MFC data (both value and velocity)
log("\nLoading data...")
mfc_data = {ccy: {} for ccy in CURRENCIES}

for ccy in CURRENCIES:
    for tf in TIMEFRAMES:
        mfc = load_mfc(ccy, tf)
        if mfc is not None:
            mfc_data[ccy][tf] = {
                'mfc': mfc.shift(1),  # Shifted MFC value
                'vel': mfc.diff().shift(1),  # Shifted velocity
            }

# Pairs
PAIRS = {
    'EURUSD': ('EUR', 'USD'), 'GBPUSD': ('GBP', 'USD'), 'USDJPY': ('USD', 'JPY'),
    'AUDUSD': ('AUD', 'USD'), 'USDCAD': ('USD', 'CAD'), 'USDCHF': ('USD', 'CHF'),
    'NZDUSD': ('NZD', 'USD'), 'EURGBP': ('EUR', 'GBP'), 'EURJPY': ('EUR', 'JPY'),
    'GBPJPY': ('GBP', 'JPY'), 'AUDJPY': ('AUD', 'JPY'), 'EURAUD': ('EUR', 'AUD'),
    'GBPAUD': ('GBP', 'AUD'), 'EURCHF': ('EUR', 'CHF'), 'GBPCHF': ('GBP', 'CHF'),
    'AUDCAD': ('AUD', 'CAD'), 'EURCAD': ('EUR', 'CAD'), 'GBPCAD': ('GBP', 'CAD'),
    'CADJPY': ('CAD', 'JPY'), 'CHFJPY': ('CHF', 'JPY'), 'AUDCHF': ('AUD', 'CHF'),
    'AUDNZD': ('AUD', 'NZD'), 'EURNZD': ('EUR', 'NZD'), 'GBPNZD': ('GBP', 'NZD'),
    'NZDJPY': ('NZD', 'JPY'), 'NZDCAD': ('NZD', 'CAD'), 'NZDCHF': ('NZD', 'CHF'),
    'CADCHF': ('CAD', 'CHF'),
}

# Load prices
prices = {}
for pair in PAIRS.keys():
    price_df = load_price(pair)
    if price_df is not None:
        prices[pair] = price_df

log(f"Loaded {len(prices)} pairs")

# Get trading days
sample = mfc_data['USD']['D1']['mfc']
trading_days = sample.index.normalize().unique()
trading_days = trading_days[(trading_days >= '2023-01-01') & (trading_days <= '2025-12-31')]
log(f"Trading days: {len(trading_days)}")

# Build dataset
log("Building dataset...")
data = []

for day in trading_days:
    start_time = day
    end_time = day + pd.Timedelta(hours=12)

    for ccy in CURRENCIES:
        # Get MFC values and velocities for all timeframes
        states = {}

        for tf in TIMEFRAMES:
            if tf not in mfc_data[ccy]:
                continue

            mfc_series = mfc_data[ccy][tf]['mfc']
            vel_series = mfc_data[ccy][tf]['vel']

            try:
                idx = mfc_series.index[mfc_series.index <= day]
                if len(idx) == 0:
                    continue

                mfc_val = mfc_series.loc[idx[-1]]
                vel_val = vel_series.loc[idx[-1]]

                if not pd.isna(mfc_val) and not pd.isna(vel_val):
                    states[tf] = {'mfc': mfc_val, 'vel': vel_val}
            except:
                continue

        if len(states) < 5:
            continue

        # Calculate currency movement
        movements = []

        for pair, (base, quote) in PAIRS.items():
            if pair not in prices:
                continue
            if base != ccy and quote != ccy:
                continue

            price_df = prices[pair]

            try:
                start_window = price_df[(price_df.index >= start_time) &
                                       (price_df.index < start_time + pd.Timedelta(hours=1))]
                end_window = price_df[(price_df.index >= end_time - pd.Timedelta(minutes=30)) &
                                     (price_df.index <= end_time)]

                if len(start_window) == 0 or len(end_window) == 0:
                    continue

                open_price = start_window.iloc[0]['Open']
                close_price = end_window.iloc[-1]['Close']
                pct_change = (close_price - open_price) / open_price * 100

                if base == ccy:
                    movements.append(pct_change)
                else:
                    movements.append(-pct_change)
            except:
                continue

        if len(movements) == 0:
            continue

        avg_movement = np.mean(movements)

        # Build record with all TF states
        record = {
            'date': day,
            'currency': ccy,
            'ccy_movement': avg_movement,
        }

        for tf in TIMEFRAMES:
            if tf in states:
                mfc_val = states[tf]['mfc']
                vel_val = states[tf]['vel']

                # Velocity direction
                vel_dir = 1 if vel_val > 0.01 else (-1 if vel_val < -0.01 else 0)

                # Position: below box (<-0.2), in box (-0.2 to 0.2), above box (>0.2)
                if mfc_val < -0.2:
                    position = 'below'
                elif mfc_val > 0.2:
                    position = 'above'
                else:
                    position = 'inbox'

                record[f'{tf}_mfc'] = mfc_val
                record[f'{tf}_vel'] = vel_val
                record[f'{tf}_vel_dir'] = vel_dir
                record[f'{tf}_pos'] = position

        data.append(record)

df = pd.DataFrame(data)
log(f"Dataset size: {len(df):,}")

# Analysis for each timeframe
log("\n" + "=" * 70)
log("VELOCITY + POSITION: BY TIMEFRAME")
log("=" * 70)

for tf in TIMEFRAMES:
    log(f"\n{'='*60}")
    log(f"{tf} TIMEFRAME")
    log(f"{'='*60}")

    vel_col = f'{tf}_vel_dir'
    pos_col = f'{tf}_pos'

    if vel_col not in df.columns:
        continue

    # All combinations
    log(f"\n{'Position':<10} {'Vel Dir':<10} {'Trades':>8} {'Win %':>10} {'Avg %':>12}")
    log("-" * 55)

    combos = []

    for pos in ['below', 'inbox', 'above']:
        for vel_dir in [-1, 1]:
            vel_label = 'UP' if vel_dir == 1 else 'DOWN'

            subset = df[(df[pos_col] == pos) & (df[vel_col] == vel_dir)].copy()

            if len(subset) > 20:
                # Trade result: follow velocity direction
                subset['trade_result'] = vel_dir * subset['ccy_movement']
                wr = (subset['trade_result'] > 0).mean() * 100
                avg = subset['trade_result'].mean()

                combos.append((pos, vel_label, len(subset), wr, avg))
                log(f"{pos:<10} {vel_label:<10} {len(subset):>8,} {wr:>9.1f}% {avg:>+11.4f}%")

    # Best and worst
    if combos:
        combos.sort(key=lambda x: x[3], reverse=True)
        log(f"\n  Best: {combos[0][0]} + {combos[0][1]} = {combos[0][3]:.1f}% WR")
        log(f"  Worst: {combos[-1][0]} + {combos[-1][1]} = {combos[-1][3]:.1f}% WR")

# Cross-timeframe: D1 position + H1 velocity
log("\n" + "=" * 70)
log("CROSS-TIMEFRAME: D1 POSITION + H1 VELOCITY")
log("=" * 70)

log(f"\n{'D1 Pos':<10} {'H1 Vel':<10} {'Trades':>8} {'Win %':>10} {'Avg %':>12}")
log("-" * 55)

for d1_pos in ['below', 'inbox', 'above']:
    for h1_vel in [-1, 1]:
        vel_label = 'UP' if h1_vel == 1 else 'DOWN'

        subset = df[(df['D1_pos'] == d1_pos) & (df['H1_vel_dir'] == h1_vel)].copy()

        if len(subset) > 20:
            subset['trade_result'] = h1_vel * subset['ccy_movement']
            wr = (subset['trade_result'] > 0).mean() * 100
            avg = subset['trade_result'].mean()
            log(f"{d1_pos:<10} {vel_label:<10} {len(subset):>8,} {wr:>9.1f}% {avg:>+11.4f}%")

# The "return to mean" scenario: velocity toward box
log("\n" + "=" * 70)
log("RETURN TO MEAN: Velocity pointing TOWARD the box")
log("=" * 70)

for tf in TIMEFRAMES:
    vel_col = f'{tf}_vel_dir'
    pos_col = f'{tf}_pos'

    if vel_col not in df.columns:
        continue

    # Below box + velocity UP (returning to box)
    below_up = df[(df[pos_col] == 'below') & (df[vel_col] == 1)].copy()
    # Above box + velocity DOWN (returning to box)
    above_down = df[(df[pos_col] == 'above') & (df[vel_col] == -1)].copy()

    # Combine
    returning = pd.concat([below_up, above_down])

    if len(returning) > 20:
        returning['trade_result'] = returning[vel_col] * returning['ccy_movement']
        wr = (returning['trade_result'] > 0).mean() * 100
        avg = returning['trade_result'].mean()
        log(f"{tf}: {len(returning):,} trades, {wr:.1f}% WR, {avg:+.4f}% avg")

# The "extension" scenario: velocity away from box
log("\n" + "=" * 70)
log("EXTENSION: Velocity pointing AWAY from the box")
log("=" * 70)

for tf in TIMEFRAMES:
    vel_col = f'{tf}_vel_dir'
    pos_col = f'{tf}_pos'

    if vel_col not in df.columns:
        continue

    # Below box + velocity DOWN (extending further)
    below_down = df[(df[pos_col] == 'below') & (df[vel_col] == -1)].copy()
    # Above box + velocity UP (extending further)
    above_up = df[(df[pos_col] == 'above') & (df[vel_col] == 1)].copy()

    # Combine
    extending = pd.concat([below_down, above_up])

    if len(extending) > 20:
        extending['trade_result'] = extending[vel_col] * extending['ccy_movement']
        wr = (extending['trade_result'] > 0).mean() * 100
        avg = extending['trade_result'].mean()
        log(f"{tf}: {len(extending):,} trades, {wr:.1f}% WR, {avg:+.4f}% avg")

# Best combo: D1 returning + H1 aligned
log("\n" + "=" * 70)
log("BEST COMBO: D1 returning to box + H1 velocity aligned")
log("=" * 70)

# D1 below + vel UP + H1 vel UP
d1_below_up_h1_up = df[(df['D1_pos'] == 'below') & (df['D1_vel_dir'] == 1) & (df['H1_vel_dir'] == 1)].copy()
if len(d1_below_up_h1_up) > 10:
    d1_below_up_h1_up['trade_result'] = d1_below_up_h1_up['ccy_movement']  # Long
    wr = (d1_below_up_h1_up['trade_result'] > 0).mean() * 100
    avg = d1_below_up_h1_up['trade_result'].mean()
    log(f"\nD1 below box + D1 vel UP + H1 vel UP:")
    log(f"  Trades: {len(d1_below_up_h1_up):,}, WR: {wr:.1f}%, Avg: {avg:+.4f}%")

# D1 above + vel DOWN + H1 vel DOWN
d1_above_down_h1_down = df[(df['D1_pos'] == 'above') & (df['D1_vel_dir'] == -1) & (df['H1_vel_dir'] == -1)].copy()
if len(d1_above_down_h1_down) > 10:
    d1_above_down_h1_down['trade_result'] = -d1_above_down_h1_down['ccy_movement']  # Short
    wr = (d1_above_down_h1_down['trade_result'] > 0).mean() * 100
    avg = d1_above_down_h1_down['trade_result'].mean()
    log(f"\nD1 above box + D1 vel DOWN + H1 vel DOWN:")
    log(f"  Trades: {len(d1_above_down_h1_down):,}, WR: {wr:.1f}%, Avg: {avg:+.4f}%")

# Combined returning
combined_returning = pd.concat([d1_below_up_h1_up, d1_above_down_h1_down])
if len(combined_returning) > 20:
    combined_returning['trade_result'] = combined_returning['D1_vel_dir'] * combined_returning['ccy_movement']
    wr = (combined_returning['trade_result'] > 0).mean() * 100
    avg = combined_returning['trade_result'].mean()
    log(f"\nCOMBINED (D1 returning + H1 aligned):")
    log(f"  Trades: {len(combined_returning):,}, WR: {wr:.1f}%, Avg: {avg:+.4f}%")

# W1 and MN position effect
log("\n" + "=" * 70)
log("HIGHER TF POSITION: W1 and MN")
log("=" * 70)

for higher_tf in ['W1', 'MN']:
    log(f"\n--- {higher_tf} Position + D1 Velocity ---")
    pos_col = f'{higher_tf}_pos'

    log(f"{'Position':<10} {'D1 Vel':<10} {'Trades':>8} {'Win %':>10} {'Avg %':>12}")
    log("-" * 55)

    for pos in ['below', 'inbox', 'above']:
        for d1_vel in [-1, 1]:
            vel_label = 'UP' if d1_vel == 1 else 'DOWN'

            subset = df[(df[pos_col] == pos) & (df['D1_vel_dir'] == d1_vel)].copy()

            if len(subset) > 20:
                subset['trade_result'] = d1_vel * subset['ccy_movement']
                wr = (subset['trade_result'] > 0).mean() * 100
                avg = subset['trade_result'].mean()
                log(f"{pos:<10} {vel_label:<10} {len(subset):>8,} {wr:>9.1f}% {avg:>+11.4f}%")

# Ultimate combo: W1/MN position + D1 returning + H1 aligned
log("\n" + "=" * 70)
log("ULTIMATE: Higher TF position + D1 returning + H1 aligned")
log("=" * 70)

# W1 below + D1 below + both vel UP + H1 vel UP
for higher_tf in ['W1', 'MN']:
    pos_col = f'{higher_tf}_pos'

    # Long setup: higher TF below, D1 below, all vel UP
    long_setup = df[(df[pos_col] == 'below') &
                    (df['D1_pos'] == 'below') &
                    (df['D1_vel_dir'] == 1) &
                    (df['H1_vel_dir'] == 1)].copy()

    # Short setup: higher TF above, D1 above, all vel DOWN
    short_setup = df[(df[pos_col] == 'above') &
                     (df['D1_pos'] == 'above') &
                     (df['D1_vel_dir'] == -1) &
                     (df['H1_vel_dir'] == -1)].copy()

    if len(long_setup) > 5:
        long_setup['trade_result'] = long_setup['ccy_movement']
        wr = (long_setup['trade_result'] > 0).mean() * 100
        avg = long_setup['trade_result'].mean()
        log(f"\n{higher_tf} below + D1 below + vel UP + H1 UP:")
        log(f"  Trades: {len(long_setup):,}, WR: {wr:.1f}%, Avg: {avg:+.4f}%")

    if len(short_setup) > 5:
        short_setup['trade_result'] = -short_setup['ccy_movement']
        wr = (short_setup['trade_result'] > 0).mean() * 100
        avg = short_setup['trade_result'].mean()
        log(f"\n{higher_tf} above + D1 above + vel DOWN + H1 DOWN:")
        log(f"  Trades: {len(short_setup):,}, WR: {wr:.1f}%, Avg: {avg:+.4f}%")

# Summary by currency
log("\n" + "=" * 70)
log("BY CURRENCY: D1 returning to box + H1 aligned")
log("=" * 70)

returning_all = pd.concat([d1_below_up_h1_up, d1_above_down_h1_down])
returning_all['trade_result'] = returning_all['D1_vel_dir'] * returning_all['ccy_movement']

log(f"\n{'Currency':<10} {'Trades':>8} {'Win %':>10} {'Avg %':>12}")
log("-" * 45)

for ccy in CURRENCIES:
    subset = returning_all[returning_all['currency'] == ccy]
    if len(subset) > 5:
        wr = (subset['trade_result'] > 0).mean() * 100
        avg = subset['trade_result'].mean()
        log(f"{ccy:<10} {len(subset):>8,} {wr:>9.1f}% {avg:>+11.4f}%")

log("\nDONE")
