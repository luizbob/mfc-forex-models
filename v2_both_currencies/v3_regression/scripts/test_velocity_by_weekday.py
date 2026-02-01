"""
Test: Velocity Combinations by Day of Week
===========================================
Check if certain days perform better than others.
"""
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from pathlib import Path

def log(msg=""):
    print(msg, flush=True)

log("=" * 70)
log("VELOCITY ANALYSIS BY DAY OF WEEK")
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

# Load MFC velocities
log("\nLoading data...")
velocities = {ccy: {} for ccy in CURRENCIES}

for ccy in CURRENCIES:
    for tf in TIMEFRAMES:
        mfc = load_mfc(ccy, tf)
        if mfc is not None:
            velocities[ccy][tf] = mfc.diff().shift(1)

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

# Build dataset
log("Building dataset...")
data = []

for day in trading_days:
    start_time = day
    end_time = day + pd.Timedelta(hours=12)
    weekday = day.dayofweek  # 0=Monday, 4=Friday
    weekday_name = day.strftime('%A')

    for ccy in CURRENCIES:
        vel_states = {}

        for tf in TIMEFRAMES:
            if tf not in velocities[ccy]:
                continue

            vel_series = velocities[ccy][tf]

            try:
                idx = vel_series.index[vel_series.index <= day]
                if len(idx) == 0:
                    continue
                vel_val = vel_series.loc[idx[-1]]

                if not pd.isna(vel_val):
                    vel_states[tf] = vel_val
            except:
                continue

        if len(vel_states) < 5:
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

        # Velocity directions
        h1_dir = 1 if vel_states['H1'] > 0.01 else (-1 if vel_states['H1'] < -0.01 else 0)
        h4_dir = 1 if vel_states['H4'] > 0.01 else (-1 if vel_states['H4'] < -0.01 else 0)
        d1_dir = 1 if vel_states['D1'] > 0.01 else (-1 if vel_states['D1'] < -0.01 else 0)
        w1_dir = 1 if vel_states['W1'] > 0.01 else (-1 if vel_states['W1'] < -0.01 else 0)
        mn_dir = 1 if vel_states['MN'] > 0.01 else (-1 if vel_states['MN'] < -0.01 else 0)

        # Check alignments
        h1_d1_aligned = (h1_dir == d1_dir) and h1_dir != 0
        h1_h4_d1_aligned = (h1_dir == h4_dir == d1_dir) and h1_dir != 0
        all_aligned = (h1_dir == h4_dir == d1_dir == w1_dir == mn_dir) and h1_dir != 0
        d1_w1_aligned = (d1_dir == w1_dir) and d1_dir != 0
        d1_w1_mn_aligned = (d1_dir == w1_dir == mn_dir) and d1_dir != 0

        data.append({
            'date': day,
            'weekday': weekday,
            'weekday_name': weekday_name,
            'currency': ccy,
            'dir_H1': h1_dir,
            'dir_H4': h4_dir,
            'dir_D1': d1_dir,
            'dir_W1': w1_dir,
            'dir_MN': mn_dir,
            'h1_d1_aligned': h1_d1_aligned,
            'h1_h4_d1_aligned': h1_h4_d1_aligned,
            'all_aligned': all_aligned,
            'd1_w1_aligned': d1_w1_aligned,
            'd1_w1_mn_aligned': d1_w1_mn_aligned,
            'ccy_movement': avg_movement,
        })

df = pd.DataFrame(data)
log(f"Dataset size: {len(df):,}")

# Analysis by day of week for different combinations
day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']

def analyze_combo(df, combo_col, combo_name, dir_col='dir_H1'):
    log("\n" + "=" * 70)
    log(f"{combo_name}: BY DAY OF WEEK")
    log("=" * 70)

    aligned = df[df[combo_col] == True].copy()
    aligned['trade_result'] = aligned[dir_col] * aligned['ccy_movement']

    log(f"\n{'Day':<12} {'Trades':>8} {'Win %':>10} {'Avg %':>12} {'Total %':>12}")
    log("-" * 55)

    for i, day_name in enumerate(day_names):
        subset = aligned[aligned['weekday'] == i]
        if len(subset) > 10:
            wr = (subset['trade_result'] > 0).mean() * 100
            avg = subset['trade_result'].mean()
            total = subset['trade_result'].sum()
            log(f"{day_name:<12} {len(subset):>8,} {wr:>9.1f}% {avg:>+11.4f}% {total:>+11.2f}%")

    # Overall
    log("-" * 55)
    wr = (aligned['trade_result'] > 0).mean() * 100
    avg = aligned['trade_result'].mean()
    total = aligned['trade_result'].sum()
    log(f"{'TOTAL':<12} {len(aligned):>8,} {wr:>9.1f}% {avg:>+11.4f}% {total:>+11.2f}%")

    return aligned

# Single timeframes
log("\n" + "=" * 70)
log("SINGLE TIMEFRAMES BY DAY OF WEEK")
log("=" * 70)

for tf in ['H1', 'H4', 'D1', 'W1', 'MN']:
    log(f"\n--- {tf} ---")
    tf_data = df[df[f'dir_{tf}'] != 0].copy()
    tf_data['trade_result'] = tf_data[f'dir_{tf}'] * tf_data['ccy_movement']

    log(f"{'Day':<12} {'Trades':>8} {'Win %':>10} {'Avg %':>12}")
    log("-" * 45)

    for i, day_name in enumerate(day_names):
        subset = tf_data[tf_data['weekday'] == i]
        if len(subset) > 20:
            wr = (subset['trade_result'] > 0).mean() * 100
            avg = subset['trade_result'].mean()
            log(f"{day_name:<12} {len(subset):>8,} {wr:>9.1f}% {avg:>+11.4f}%")

# Combinations
aligned_h1_d1 = analyze_combo(df, 'h1_d1_aligned', 'H1+D1 ALIGNED', 'dir_H1')
aligned_h1_h4_d1 = analyze_combo(df, 'h1_h4_d1_aligned', 'H1+H4+D1 ALIGNED', 'dir_H1')
aligned_d1_w1 = analyze_combo(df, 'd1_w1_aligned', 'D1+W1 ALIGNED', 'dir_D1')
aligned_d1_w1_mn = analyze_combo(df, 'd1_w1_mn_aligned', 'D1+W1+MN ALIGNED', 'dir_D1')
aligned_all = analyze_combo(df, 'all_aligned', 'ALL 5 TFs ALIGNED', 'dir_H1')

# Summary comparison
log("\n" + "=" * 70)
log("SUMMARY: ALL COMBINATIONS BY DAY")
log("=" * 70)

combos = [
    ('H1', df[df['dir_H1'] != 0].copy(), 'dir_H1'),
    ('H4', df[df['dir_H4'] != 0].copy(), 'dir_H4'),
    ('D1', df[df['dir_D1'] != 0].copy(), 'dir_D1'),
    ('W1', df[df['dir_W1'] != 0].copy(), 'dir_W1'),
    ('MN', df[df['dir_MN'] != 0].copy(), 'dir_MN'),
    ('H1+D1', df[df['h1_d1_aligned'] == True].copy(), 'dir_H1'),
    ('H1+H4+D1', df[df['h1_h4_d1_aligned'] == True].copy(), 'dir_H1'),
    ('D1+W1', df[df['d1_w1_aligned'] == True].copy(), 'dir_D1'),
    ('D1+W1+MN', df[df['d1_w1_mn_aligned'] == True].copy(), 'dir_D1'),
    ('ALL 5', df[df['all_aligned'] == True].copy(), 'dir_H1'),
]

log(f"\n{'Combo':<12} {'Mon':>10} {'Tue':>10} {'Wed':>10} {'Thu':>10} {'Fri':>10} {'Best Day':<12}")
log("-" * 80)

for combo_name, combo_df, dir_col in combos:
    if len(combo_df) < 50:
        continue

    combo_df['trade_result'] = combo_df[dir_col] * combo_df['ccy_movement']

    day_wrs = []
    for i in range(5):
        subset = combo_df[combo_df['weekday'] == i]
        if len(subset) > 10:
            wr = (subset['trade_result'] > 0).mean() * 100
            day_wrs.append(wr)
        else:
            day_wrs.append(0)

    best_day_idx = day_wrs.index(max(day_wrs))
    best_day = day_names[best_day_idx]

    log(f"{combo_name:<12} {day_wrs[0]:>9.1f}% {day_wrs[1]:>9.1f}% {day_wrs[2]:>9.1f}% {day_wrs[3]:>9.1f}% {day_wrs[4]:>9.1f}% {best_day:<12}")

# Best day for each combo with avg pips
log("\n" + "=" * 70)
log("BEST DAY FOR EACH COMBINATION (with avg %)")
log("=" * 70)

log(f"\n{'Combo':<12} {'Best Day':<12} {'Trades':>8} {'Win %':>10} {'Avg %':>12}")
log("-" * 55)

for combo_name, combo_df, dir_col in combos:
    if len(combo_df) < 50:
        continue

    combo_df['trade_result'] = combo_df[dir_col] * combo_df['ccy_movement']

    best_day = None
    best_wr = 0
    best_count = 0
    best_avg = 0

    for i, day_name in enumerate(day_names):
        subset = combo_df[combo_df['weekday'] == i]
        if len(subset) > 10:
            wr = (subset['trade_result'] > 0).mean() * 100
            if wr > best_wr:
                best_wr = wr
                best_day = day_name
                best_count = len(subset)
                best_avg = subset['trade_result'].mean()

    if best_day:
        log(f"{combo_name:<12} {best_day:<12} {best_count:>8,} {best_wr:>9.1f}% {best_avg:>+11.4f}%")

# Worst day for each combo
log("\n" + "=" * 70)
log("WORST DAY FOR EACH COMBINATION")
log("=" * 70)

log(f"\n{'Combo':<12} {'Worst Day':<12} {'Trades':>8} {'Win %':>10} {'Avg %':>12}")
log("-" * 55)

for combo_name, combo_df, dir_col in combos:
    if len(combo_df) < 50:
        continue

    combo_df['trade_result'] = combo_df[dir_col] * combo_df['ccy_movement']

    worst_day = None
    worst_wr = 100
    worst_count = 0
    worst_avg = 0

    for i, day_name in enumerate(day_names):
        subset = combo_df[combo_df['weekday'] == i]
        if len(subset) > 10:
            wr = (subset['trade_result'] > 0).mean() * 100
            if wr < worst_wr:
                worst_wr = wr
                worst_day = day_name
                worst_count = len(subset)
                worst_avg = subset['trade_result'].mean()

    if worst_day:
        log(f"{combo_name:<12} {worst_day:<12} {worst_count:>8,} {worst_wr:>9.1f}% {worst_avg:>+11.4f}%")

# Monday effect across all combos
log("\n" + "=" * 70)
log("MONDAY EFFECT: All combinations")
log("=" * 70)

log(f"\n{'Combo':<12} {'Mon WR':>10} {'Other WR':>10} {'Diff':>10}")
log("-" * 45)

for combo_name, combo_df, dir_col in combos:
    if len(combo_df) < 50:
        continue

    combo_df['trade_result'] = combo_df[dir_col] * combo_df['ccy_movement']

    monday = combo_df[combo_df['weekday'] == 0]
    other = combo_df[combo_df['weekday'] != 0]

    if len(monday) > 10 and len(other) > 10:
        mon_wr = (monday['trade_result'] > 0).mean() * 100
        other_wr = (other['trade_result'] > 0).mean() * 100
        diff = mon_wr - other_wr
        log(f"{combo_name:<12} {mon_wr:>9.1f}% {other_wr:>9.1f}% {diff:>+9.1f}%")

# Tuesday effect (worst day)
log("\n" + "=" * 70)
log("TUESDAY EFFECT: All combinations")
log("=" * 70)

log(f"\n{'Combo':<12} {'Tue WR':>10} {'Other WR':>10} {'Diff':>10}")
log("-" * 45)

for combo_name, combo_df, dir_col in combos:
    if len(combo_df) < 50:
        continue

    combo_df['trade_result'] = combo_df[dir_col] * combo_df['ccy_movement']

    tuesday = combo_df[combo_df['weekday'] == 1]
    other = combo_df[combo_df['weekday'] != 1]

    if len(tuesday) > 10 and len(other) > 10:
        tue_wr = (tuesday['trade_result'] > 0).mean() * 100
        other_wr = (other['trade_result'] > 0).mean() * 100
        diff = tue_wr - other_wr
        log(f"{combo_name:<12} {tue_wr:>9.1f}% {other_wr:>9.1f}% {diff:>+9.1f}%")

# Best: Skip Tuesday
log("\n" + "=" * 70)
log("SKIP TUESDAY: Performance without Tuesday")
log("=" * 70)

log(f"\n{'Combo':<12} {'All Days':>12} {'No Tuesday':>12} {'Improvement':>12}")
log("-" * 50)

for combo_name, combo_df, dir_col in combos:
    if len(combo_df) < 50:
        continue

    combo_df['trade_result'] = combo_df[dir_col] * combo_df['ccy_movement']

    all_days_wr = (combo_df['trade_result'] > 0).mean() * 100
    no_tue = combo_df[combo_df['weekday'] != 1]
    no_tue_wr = (no_tue['trade_result'] > 0).mean() * 100
    improvement = no_tue_wr - all_days_wr

    log(f"{combo_name:<12} {all_days_wr:>11.1f}% {no_tue_wr:>11.1f}% {improvement:>+11.1f}%")

log("\nDONE")
