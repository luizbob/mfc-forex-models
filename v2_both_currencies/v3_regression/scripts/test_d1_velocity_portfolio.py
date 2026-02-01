"""
Test: Multi-TF Velocity Portfolio Strategy (H1+H4+D1)
=====================================================
At 00:00, find the currency with aligned H1+H4+D1 velocity.
Trade that currency LONG against all others from 00:00 to 12:00.
Measure % price movement.
"""
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from pathlib import Path

def log(msg=""):
    print(msg, flush=True)

log("=" * 70)
log("MULTI-TF VELOCITY PORTFOLIO (H1+H4+D1)")
log("=" * 70)

DATA_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/data')
CLEANED_DIR = DATA_DIR / 'cleaned'

CURRENCIES = ['USD', 'EUR', 'GBP', 'JPY', 'AUD', 'CAD', 'CHF', 'NZD']

def load_mfc(currency, timeframe):
    fp = CLEANED_DIR / f'mfc_currency_{currency}_{timeframe}_clean.csv'
    if not fp.exists():
        return None
    df = pd.read_csv(fp)
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df = df.set_index('datetime').sort_index()
    # Filter to 2023-2025
    df = df[(df.index >= '2023-01-01') & (df.index <= '2025-12-31')]
    return df['MFC']

def load_price(pair):
    # Try M30 first (faster to process)
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

# Load H1, H4, D1 MFC velocities for all currencies
log("\nLoading MFC velocities (H1, H4, D1)...")
vel_h1 = {}
vel_h4 = {}
vel_d1 = {}

for ccy in CURRENCIES:
    mfc_h1 = load_mfc(ccy, 'H1')
    mfc_h4 = load_mfc(ccy, 'H4')
    mfc_d1 = load_mfc(ccy, 'D1')

    if mfc_h1 is not None:
        vel_h1[ccy] = mfc_h1.diff().shift(1)
    if mfc_h4 is not None:
        vel_h4[ccy] = mfc_h4.diff().shift(1)
    if mfc_d1 is not None:
        vel_d1[ccy] = mfc_d1.diff().shift(1)

    log(f"  {ccy}: H1={len(vel_h1.get(ccy, [])):,}, H4={len(vel_h4.get(ccy, [])):,}, D1={len(vel_d1.get(ccy, [])):,}")

# Available pairs and their structure
PAIRS = {
    'EURUSD': ('EUR', 'USD'),
    'GBPUSD': ('GBP', 'USD'),
    'USDJPY': ('USD', 'JPY'),
    'AUDUSD': ('AUD', 'USD'),
    'USDCAD': ('USD', 'CAD'),
    'USDCHF': ('USD', 'CHF'),
    'NZDUSD': ('NZD', 'USD'),
    'EURGBP': ('EUR', 'GBP'),
    'EURJPY': ('EUR', 'JPY'),
    'GBPJPY': ('GBP', 'JPY'),
    'AUDJPY': ('AUD', 'JPY'),
    'EURAUD': ('EUR', 'AUD'),
    'GBPAUD': ('GBP', 'AUD'),
    'EURCHF': ('EUR', 'CHF'),
    'GBPCHF': ('GBP', 'CHF'),
    'AUDCAD': ('AUD', 'CAD'),
    'EURCAD': ('EUR', 'CAD'),
    'GBPCAD': ('GBP', 'CAD'),
    'CADJPY': ('CAD', 'JPY'),
    'CHFJPY': ('CHF', 'JPY'),
    'AUDCHF': ('AUD', 'CHF'),
    'AUDNZD': ('AUD', 'NZD'),
    'EURNZD': ('EUR', 'NZD'),
    'GBPNZD': ('GBP', 'NZD'),
    'NZDJPY': ('NZD', 'JPY'),
    'NZDCAD': ('NZD', 'CAD'),
    'NZDCHF': ('NZD', 'CHF'),
    'CADCHF': ('CAD', 'CHF'),
}

# Load price data for available pairs
log("\nLoading price data...")
prices = {}
for pair in PAIRS.keys():
    price_df = load_price(pair)
    if price_df is not None:
        prices[pair] = price_df
        log(f"  {pair}: {len(price_df):,} bars")

log(f"\nLoaded {len(prices)} pairs")

# Get unique trading days
sample_vel = list(vel_d1.values())[0]
trading_days = sample_vel.index.normalize().unique()
trading_days = trading_days[(trading_days >= '2023-01-01') & (trading_days <= '2025-12-31')]
log(f"Trading days: {len(trading_days)}")

# For each day at 00:00
results = []
results_d1_only = []

for day in trading_days:
    start_time = day
    end_time = day + pd.Timedelta(hours=12)

    # Get velocities for all currencies at all timeframes
    ccy_scores = {}

    for ccy in CURRENCIES:
        try:
            # D1 velocity (at day start)
            v_d1 = vel_d1[ccy].loc[day] if ccy in vel_d1 else np.nan

            # H4 velocity (find the 00:00 H4 bar)
            h4_idx = vel_h4[ccy].index[vel_h4[ccy].index <= day]
            v_h4 = vel_h4[ccy].loc[h4_idx[-1]] if len(h4_idx) > 0 and ccy in vel_h4 else np.nan

            # H1 velocity (at 00:00)
            h1_idx = vel_h1[ccy].index[vel_h1[ccy].index <= day]
            v_h1 = vel_h1[ccy].loc[h1_idx[-1]] if len(h1_idx) > 0 and ccy in vel_h1 else np.nan

            if pd.isna(v_d1) or pd.isna(v_h4) or pd.isna(v_h1):
                continue

            # Check alignment: all three must agree on direction
            h1_dir = 1 if v_h1 > 0.01 else (-1 if v_h1 < -0.01 else 0)
            h4_dir = 1 if v_h4 > 0.01 else (-1 if v_h4 < -0.01 else 0)
            d1_dir = 1 if v_d1 > 0.01 else (-1 if v_d1 < -0.01 else 0)

            all_agree = (h1_dir == h4_dir == d1_dir) and h1_dir != 0

            # Combined velocity score (average of all three)
            avg_vel = (v_h1 + v_h4 + v_d1) / 3

            ccy_scores[ccy] = {
                'v_h1': v_h1,
                'v_h4': v_h4,
                'v_d1': v_d1,
                'avg_vel': avg_vel,
                'direction': h1_dir if all_agree else 0,
                'all_agree': all_agree,
            }
        except:
            continue

    if len(ccy_scores) < 4:
        continue

    # Find currencies with aligned velocities
    aligned_up = [(ccy, data) for ccy, data in ccy_scores.items() if data['all_agree'] and data['direction'] == 1]
    aligned_down = [(ccy, data) for ccy, data in ccy_scores.items() if data['all_agree'] and data['direction'] == -1]

    # Sort by velocity strength
    aligned_up.sort(key=lambda x: x[1]['avg_vel'], reverse=True)
    aligned_down.sort(key=lambda x: x[1]['avg_vel'])

    # Trade pairs where strongest aligned UP vs weakest aligned DOWN
    day_trades = []

    for pair, (base, quote) in PAIRS.items():
        if pair not in prices:
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
        except:
            continue

        trade_direction = None
        base_data = ccy_scores.get(base, {})
        quote_data = ccy_scores.get(quote, {})

        # Case 1: Base aligned UP, Quote aligned DOWN -> Strong BUY
        if base_data.get('all_agree') and base_data.get('direction') == 1 and \
           quote_data.get('all_agree') and quote_data.get('direction') == -1:
            trade_direction = 'buy'
            trade_pct = pct_change
            strength = 'both_aligned'

        # Case 2: Base aligned DOWN, Quote aligned UP -> Strong SELL
        elif base_data.get('all_agree') and base_data.get('direction') == -1 and \
             quote_data.get('all_agree') and quote_data.get('direction') == 1:
            trade_direction = 'sell'
            trade_pct = -pct_change
            strength = 'both_aligned'

        # Case 3: Only base aligned UP -> BUY
        elif base_data.get('all_agree') and base_data.get('direction') == 1:
            trade_direction = 'buy'
            trade_pct = pct_change
            strength = 'base_only'

        # Case 4: Only quote aligned UP -> SELL
        elif quote_data.get('all_agree') and quote_data.get('direction') == 1:
            trade_direction = 'sell'
            trade_pct = -pct_change
            strength = 'quote_only'

        # Case 5: Only base aligned DOWN -> SELL
        elif base_data.get('all_agree') and base_data.get('direction') == -1:
            trade_direction = 'sell'
            trade_pct = -pct_change
            strength = 'base_only'

        # Case 6: Only quote aligned DOWN -> BUY
        elif quote_data.get('all_agree') and quote_data.get('direction') == -1:
            trade_direction = 'buy'
            trade_pct = pct_change
            strength = 'quote_only'

        if trade_direction:
            avg_vel = max(abs(base_data.get('avg_vel', 0)), abs(quote_data.get('avg_vel', 0)))
            day_trades.append({
                'date': day,
                'pair': pair,
                'base': base,
                'quote': quote,
                'direction': trade_direction,
                'strength': strength,
                'avg_vel': avg_vel,
                'pct_change': pct_change,
                'trade_pct': trade_pct,
            })

    results.extend(day_trades)

df = pd.DataFrame(results)
log(f"\nTotal trades: {len(df):,}")
if len(df) > 0:
    log(f"Unique days: {df['date'].nunique()}")

# Overall results
log("\n" + "=" * 70)
log("OVERALL RESULTS (H1+H4+D1 aligned)")
log("=" * 70)

if len(df) == 0:
    log("\nNo trades found!")
else:
    log(f"\nTotal trades: {len(df):,}")
    log(f"Avg % return per trade: {df['trade_pct'].mean():+.4f}%")
    log(f"Win rate: {(df['trade_pct'] > 0).mean()*100:.1f}%")
    log(f"Total % return: {df['trade_pct'].sum():+.2f}%")

    # By alignment strength
    log("\n" + "=" * 70)
    log("BY ALIGNMENT STRENGTH")
    log("=" * 70)

    log(f"\n{'Strength':<15} {'Trades':>8} {'Win %':>10} {'Avg %':>12} {'Total %':>12}")
    log("-" * 60)

    for strength in ['both_aligned', 'base_only', 'quote_only']:
        subset = df[df['strength'] == strength]
        if len(subset) > 20:
            wr = (subset['trade_pct'] > 0).mean() * 100
            avg = subset['trade_pct'].mean()
            total = subset['trade_pct'].sum()
            log(f"{strength:<15} {len(subset):>8,} {wr:>9.1f}% {avg:>+11.4f}% {total:>+11.2f}%")

    # By velocity magnitude
    log("\n" + "=" * 70)
    log("BY AVERAGE VELOCITY MAGNITUDE")
    log("=" * 70)

    log(f"\n{'Velocity':<15} {'Trades':>8} {'Win %':>10} {'Avg %':>12} {'Total %':>12}")
    log("-" * 60)

    for low, high, label in [(0.00, 0.03, '0.00-0.03'), (0.03, 0.05, '0.03-0.05'),
                              (0.05, 0.08, '0.05-0.08'), (0.08, 0.12, '0.08-0.12'),
                              (0.12, 1.0, '0.12+')]:
        subset = df[(df['avg_vel'] >= low) & (df['avg_vel'] < high)]
        if len(subset) > 50:
            wr = (subset['trade_pct'] > 0).mean() * 100
            avg = subset['trade_pct'].mean()
            total = subset['trade_pct'].sum()
            log(f"{label:<15} {len(subset):>8,} {wr:>9.1f}% {avg:>+11.4f}% {total:>+11.2f}%")

    # By pair
    log("\n" + "=" * 70)
    log("BY PAIR")
    log("=" * 70)

    log(f"\n{'Pair':<10} {'Trades':>8} {'Win %':>10} {'Avg %':>12} {'Total %':>12}")
    log("-" * 55)

    pair_stats = []
    for pair in sorted(df['pair'].unique()):
        subset = df[df['pair'] == pair]
        if len(subset) > 20:
            wr = (subset['trade_pct'] > 0).mean() * 100
            avg = subset['trade_pct'].mean()
            total = subset['trade_pct'].sum()
            pair_stats.append((pair, len(subset), wr, avg, total))

    pair_stats.sort(key=lambda x: x[4], reverse=True)
    for pair, count, wr, avg, total in pair_stats:
        log(f"{pair:<10} {count:>8,} {wr:>9.1f}% {avg:>+11.4f}% {total:>+11.2f}%")

    # Best case: Both aligned
    log("\n" + "=" * 70)
    log("BEST CASE: BOTH CURRENCIES ALIGNED (strongest vs weakest)")
    log("=" * 70)

    both = df[df['strength'] == 'both_aligned']
    if len(both) > 0:
        log(f"\nTotal trades: {len(both):,}")
        log(f"Win rate: {(both['trade_pct'] > 0).mean()*100:.1f}%")
        log(f"Avg % return: {both['trade_pct'].mean():+.4f}%")
        log(f"Total % return: {both['trade_pct'].sum():+.2f}%")

        log(f"\n{'Pair':<10} {'Trades':>8} {'Win %':>10} {'Avg %':>12} {'Total %':>12}")
        log("-" * 55)

        pair_stats = []
        for pair in sorted(both['pair'].unique()):
            subset = both[both['pair'] == pair]
            if len(subset) > 5:
                wr = (subset['trade_pct'] > 0).mean() * 100
                avg = subset['trade_pct'].mean()
                total = subset['trade_pct'].sum()
                pair_stats.append((pair, len(subset), wr, avg, total))

        pair_stats.sort(key=lambda x: x[4], reverse=True)
        for pair, count, wr, avg, total in pair_stats:
            log(f"{pair:<10} {count:>8,} {wr:>9.1f}% {avg:>+11.4f}% {total:>+11.2f}%")

    # Filter: both_aligned + velocity > 0.05
    log("\n" + "=" * 70)
    log("FILTERED: Both aligned + velocity > 0.05")
    log("=" * 70)

    filtered = df[(df['strength'] == 'both_aligned') & (df['avg_vel'] > 0.05)]
    if len(filtered) > 0:
        log(f"\nTotal trades: {len(filtered):,}")
        log(f"Win rate: {(filtered['trade_pct'] > 0).mean()*100:.1f}%")
        log(f"Avg % return: {filtered['trade_pct'].mean():+.4f}%")
        log(f"Total % return: {filtered['trade_pct'].sum():+.2f}%")

log("\nDONE")
