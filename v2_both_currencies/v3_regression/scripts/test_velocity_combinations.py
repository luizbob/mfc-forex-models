"""
Test: All Velocity Timeframe Combinations
==========================================
Test every combination of D1, H4, H1 velocity alignment.
At 00:00, check MFC velocity state, predict 12h currency movement.
"""
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from pathlib import Path
from itertools import combinations

def log(msg=""):
    print(msg, flush=True)

log("=" * 70)
log("ALL VELOCITY TIMEFRAME COMBINATIONS")
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

# Load MFC and calculate velocities
log("\nLoading MFC velocities...")
velocities = {ccy: {} for ccy in CURRENCIES}

for ccy in CURRENCIES:
    for tf in TIMEFRAMES:
        mfc = load_mfc(ccy, tf)
        if mfc is not None:
            velocities[ccy][tf] = mfc.diff().shift(1)  # Shifted velocity
    log(f"  {ccy}: {list(velocities[ccy].keys())}")

# Pairs structure
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

# Load price data
log("\nLoading price data...")
prices = {}
for pair in PAIRS.keys():
    price_df = load_price(pair)
    if price_df is not None:
        prices[pair] = price_df
log(f"Loaded {len(prices)} pairs")

# Get trading days
sample_vel = velocities['USD']['D1']
trading_days = sample_vel.index.normalize().unique()
trading_days = trading_days[(trading_days >= '2023-01-01') & (trading_days <= '2025-12-31')]
log(f"Trading days: {len(trading_days)}")

# Build dataset: for each day and currency, get all velocity states and price movement
log("\nBuilding dataset...")
data = []

for day in trading_days:
    start_time = day
    end_time = day + pd.Timedelta(hours=12)

    for ccy in CURRENCIES:
        vel_states = {}

        # Get velocity for each timeframe at 00:00
        for tf in TIMEFRAMES:
            if tf not in velocities[ccy]:
                continue

            vel_series = velocities[ccy][tf]

            try:
                # Find the velocity value at or before day start
                idx = vel_series.index[vel_series.index <= day]
                if len(idx) == 0:
                    continue
                vel_val = vel_series.loc[idx[-1]]

                if not pd.isna(vel_val):
                    vel_states[tf] = vel_val
            except:
                continue

        if len(vel_states) < 5:  # Need all 5 timeframes
            continue

        # Calculate currency movement from all pairs containing this currency
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

                # Adjust for currency position
                if base == ccy:
                    movements.append(pct_change)
                else:
                    movements.append(-pct_change)
            except:
                continue

        if len(movements) == 0:
            continue

        # Average movement across all pairs
        avg_movement = np.mean(movements)

        # Velocity directions
        h1_dir = 1 if vel_states['H1'] > 0.01 else (-1 if vel_states['H1'] < -0.01 else 0)
        h4_dir = 1 if vel_states['H4'] > 0.01 else (-1 if vel_states['H4'] < -0.01 else 0)
        d1_dir = 1 if vel_states['D1'] > 0.01 else (-1 if vel_states['D1'] < -0.01 else 0)
        w1_dir = 1 if vel_states['W1'] > 0.01 else (-1 if vel_states['W1'] < -0.01 else 0)
        mn_dir = 1 if vel_states['MN'] > 0.01 else (-1 if vel_states['MN'] < -0.01 else 0)

        data.append({
            'date': day,
            'currency': ccy,
            'vel_H1': vel_states['H1'],
            'vel_H4': vel_states['H4'],
            'vel_D1': vel_states['D1'],
            'vel_W1': vel_states['W1'],
            'vel_MN': vel_states['MN'],
            'dir_H1': h1_dir,
            'dir_H4': h4_dir,
            'dir_D1': d1_dir,
            'dir_W1': w1_dir,
            'dir_MN': mn_dir,
            'ccy_movement': avg_movement,
        })

df = pd.DataFrame(data)
log(f"Dataset size: {len(df):,}")

# Test all combinations
log("\n" + "=" * 70)
log("ALL TIMEFRAME COMBINATIONS")
log("=" * 70)

# Single timeframes
single_tfs = ['H1', 'H4', 'D1', 'W1', 'MN']

# All combinations (1 to 5 timeframes)
all_combos = []
for r in range(1, 6):
    all_combos.extend(combinations(single_tfs, r))

log(f"\n{'Combination':<20} {'Trades':>8} {'Win %':>10} {'Avg %':>12} {'Total %':>12}")
log("-" * 65)

results_summary = []

for combo in all_combos:
    combo_name = '+'.join(combo)

    # Filter rows where all timeframes in combo agree
    mask = pd.Series([True] * len(df))
    signal_dir = None

    for tf in combo:
        dir_col = f'dir_{tf}'
        if signal_dir is None:
            # First TF sets the direction
            mask = mask & (df[dir_col] != 0)
            signal_dir = df[dir_col]
        else:
            # Other TFs must match
            mask = mask & (df[dir_col] == signal_dir)

    subset = df[mask].copy()

    if len(subset) < 50:
        continue

    # Calculate trade result (follow the signal direction)
    first_tf = combo[0]
    subset['signal'] = subset[f'dir_{first_tf}']
    subset['trade_result'] = subset['signal'] * subset['ccy_movement']

    wr = (subset['trade_result'] > 0).mean() * 100
    avg = subset['trade_result'].mean()
    total = subset['trade_result'].sum()

    results_summary.append({
        'combo': combo_name,
        'count': len(subset),
        'win_rate': wr,
        'avg_pct': avg,
        'total_pct': total,
    })

    log(f"{combo_name:<20} {len(subset):>8,} {wr:>9.1f}% {avg:>+11.4f}% {total:>+11.2f}%")

# Sort by win rate
log("\n" + "=" * 70)
log("SORTED BY WIN RATE")
log("=" * 70)

results_df = pd.DataFrame(results_summary)
results_df = results_df.sort_values('win_rate', ascending=False)

log(f"\n{'Combination':<20} {'Trades':>8} {'Win %':>10} {'Avg %':>12} {'Total %':>12}")
log("-" * 65)

for _, row in results_df.iterrows():
    log(f"{row['combo']:<20} {row['count']:>8,} {row['win_rate']:>9.1f}% {row['avg_pct']:>+11.4f}% {row['total_pct']:>+11.2f}%")

# Best combo by currency
log("\n" + "=" * 70)
log("ALL 5 TFs ALIGNED (H1+H4+D1+W1+MN) BY CURRENCY")
log("=" * 70)

# All 5 aligned
mask = ((df['dir_H1'] == df['dir_H4']) & (df['dir_H4'] == df['dir_D1']) &
        (df['dir_D1'] == df['dir_W1']) & (df['dir_W1'] == df['dir_MN']) & (df['dir_H1'] != 0))
best_df = df[mask].copy()

if len(best_df) > 0:
    best_df['trade_result'] = best_df['dir_H1'] * best_df['ccy_movement']

    log(f"\nTotal: {len(best_df)} trades, {(best_df['trade_result'] > 0).mean()*100:.1f}% WR, {best_df['trade_result'].mean():+.4f}% avg")

    log(f"\n{'Currency':<10} {'Trades':>8} {'Win %':>10} {'Avg %':>12} {'Total %':>12}")
    log("-" * 55)

    for ccy in CURRENCIES:
        subset = best_df[best_df['currency'] == ccy]
        if len(subset) > 5:
            wr = (subset['trade_result'] > 0).mean() * 100
            avg = subset['trade_result'].mean()
            total = subset['trade_result'].sum()
            log(f"{ccy:<10} {len(subset):>8,} {wr:>9.1f}% {avg:>+11.4f}% {total:>+11.2f}%")
else:
    log("\nNo cases where all 5 timeframes align!")

# Test disagreement scenarios
log("\n" + "=" * 70)
log("DISAGREEMENT SCENARIOS: WHO WINS? (all pairwise comparisons)")
log("=" * 70)

# All pairwise comparisons
tf_pairs = [('H1', 'H4'), ('H1', 'D1'), ('H1', 'W1'), ('H1', 'MN'),
            ('H4', 'D1'), ('H4', 'W1'), ('H4', 'MN'),
            ('D1', 'W1'), ('D1', 'MN'),
            ('W1', 'MN')]

log(f"\n{'Matchup':<12} {'Cases':>8} {'TF1 Wins':>10} {'TF2 Wins':>10} {'Winner':>10}")
log("-" * 55)

for tf1, tf2 in tf_pairs:
    dir1_col = f'dir_{tf1}'
    dir2_col = f'dir_{tf2}'

    # TF1 up, TF2 down
    case1 = df[(df[dir1_col] == 1) & (df[dir2_col] == -1)]
    # TF1 down, TF2 up
    case2 = df[(df[dir1_col] == -1) & (df[dir2_col] == 1)]

    total_cases = len(case1) + len(case2)
    if total_cases < 20:
        continue

    # Count wins
    tf1_wins = 0
    tf2_wins = 0

    if len(case1) > 0:
        tf1_wins += (case1['ccy_movement'] > 0).sum()
        tf2_wins += (case1['ccy_movement'] < 0).sum()
    if len(case2) > 0:
        tf1_wins += (case2['ccy_movement'] < 0).sum()
        tf2_wins += (case2['ccy_movement'] > 0).sum()

    tf1_pct = tf1_wins / (tf1_wins + tf2_wins) * 100 if (tf1_wins + tf2_wins) > 0 else 0
    tf2_pct = 100 - tf1_pct

    winner = tf1 if tf1_pct > tf2_pct else tf2
    log(f"{tf1} vs {tf2:<6} {total_cases:>8,} {tf1_pct:>9.1f}% {tf2_pct:>9.1f}% {winner:>10}")

# Summary of all pairwise matchups
log("\n" + "=" * 70)
log("OVERALL HIERARCHY: TOTAL WINS IN ALL HEAD-TO-HEAD MATCHUPS")
log("=" * 70)

wins = {tf: 0 for tf in TIMEFRAMES}
total_matchups = 0

for tf1, tf2 in tf_pairs:
    dir1_col = f'dir_{tf1}'
    dir2_col = f'dir_{tf2}'

    for d1, d2 in [(1, -1), (-1, 1)]:
        subset = df[(df[dir1_col] == d1) & (df[dir2_col] == d2)]
        if len(subset) > 0:
            if d1 == 1:
                wins[tf1] += (subset['ccy_movement'] > 0).sum()
                wins[tf2] += (subset['ccy_movement'] < 0).sum()
            else:
                wins[tf1] += (subset['ccy_movement'] < 0).sum()
                wins[tf2] += (subset['ccy_movement'] > 0).sum()
            total_matchups += len(subset)

total_wins = sum(wins.values())
log(f"\nWin counts in all head-to-head disagreements:")

# Sort by wins
sorted_wins = sorted(wins.items(), key=lambda x: x[1], reverse=True)
for tf, w in sorted_wins:
    log(f"  {tf}: {w:,} wins ({w/total_wins*100:.1f}%)")

# Best higher TF combinations
log("\n" + "=" * 70)
log("HIGHER TF COMBINATIONS")
log("=" * 70)

# D1+W1
mask = (df['dir_D1'] == df['dir_W1']) & (df['dir_D1'] != 0)
d1w1 = df[mask].copy()
if len(d1w1) > 20:
    d1w1['trade_result'] = d1w1['dir_D1'] * d1w1['ccy_movement']
    wr = (d1w1['trade_result'] > 0).mean() * 100
    avg = d1w1['trade_result'].mean()
    log(f"\nD1+W1 aligned: {len(d1w1):,} trades, {wr:.1f}% WR, {avg:+.4f}% avg")

# D1+W1+MN
mask = (df['dir_D1'] == df['dir_W1']) & (df['dir_W1'] == df['dir_MN']) & (df['dir_D1'] != 0)
d1w1mn = df[mask].copy()
if len(d1w1mn) > 20:
    d1w1mn['trade_result'] = d1w1mn['dir_D1'] * d1w1mn['ccy_movement']
    wr = (d1w1mn['trade_result'] > 0).mean() * 100
    avg = d1w1mn['trade_result'].mean()
    log(f"D1+W1+MN aligned: {len(d1w1mn):,} trades, {wr:.1f}% WR, {avg:+.4f}% avg")

# W1+MN
mask = (df['dir_W1'] == df['dir_MN']) & (df['dir_W1'] != 0)
w1mn = df[mask].copy()
if len(w1mn) > 20:
    w1mn['trade_result'] = w1mn['dir_W1'] * w1mn['ccy_movement']
    wr = (w1mn['trade_result'] > 0).mean() * 100
    avg = w1mn['trade_result'].mean()
    log(f"W1+MN aligned: {len(w1mn):,} trades, {wr:.1f}% WR, {avg:+.4f}% avg")

# H1+D1+W1
mask = (df['dir_H1'] == df['dir_D1']) & (df['dir_D1'] == df['dir_W1']) & (df['dir_H1'] != 0)
h1d1w1 = df[mask].copy()
if len(h1d1w1) > 20:
    h1d1w1['trade_result'] = h1d1w1['dir_H1'] * h1d1w1['ccy_movement']
    wr = (h1d1w1['trade_result'] > 0).mean() * 100
    avg = h1d1w1['trade_result'].mean()
    log(f"H1+D1+W1 aligned: {len(h1d1w1):,} trades, {wr:.1f}% WR, {avg:+.4f}% avg")

log("\nDONE")
