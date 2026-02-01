"""
Test: Multi-Timeframe Velocity + Price Prediction (12 Hours)
=============================================================
Combine H1/H4/D1 MFC velocities with price to predict next 12 hours.
"""
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from pathlib import Path

def log(msg=""):
    print(msg, flush=True)

log("=" * 70)
log("MULTI-TIMEFRAME VELOCITY + 12H PRICE PREDICTION")
log("=" * 70)

DATA_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/data')
CLEANED_DIR = DATA_DIR / 'cleaned'

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

def load_price(pair, timeframe='H1'):
    # Try direct file first
    fp = DATA_DIR / f'{pair}_GMT+0_US-DST_{timeframe}.csv'
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
        # Filter to 2023-2025
        df = df[(df.index >= '2023-01-01') & (df.index <= '2025-12-31')]
        return df

    # Try M1 and resample
    fp = DATA_DIR / f'{pair}_GMT+0_US-DST_M1.csv'
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
        # Filter to 2023-2025
        df = df[(df.index >= '2023-01-01') & (df.index <= '2025-12-31')]

        if timeframe == 'H1':
            return df.resample('1h').agg({
                'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
            }).dropna()

    return None

# Test pairs
pairs_to_test = [
    ('EUR', 'USD'), ('GBP', 'USD'), ('USD', 'JPY'), ('AUD', 'USD'),
    ('EUR', 'JPY'), ('GBP', 'JPY'), ('USD', 'CAD'), ('EUR', 'GBP'),
]

PIP_SIZE = {
    'EURUSD': 0.0001, 'GBPUSD': 0.0001, 'USDJPY': 0.01, 'AUDUSD': 0.0001,
    'EURJPY': 0.01, 'GBPJPY': 0.01, 'USDCAD': 0.0001, 'EURGBP': 0.0001,
}

SPREADS = {
    'EURUSD': 0.8, 'GBPUSD': 1.0, 'USDJPY': 1.0, 'AUDUSD': 0.9,
    'EURJPY': 2.4, 'GBPJPY': 2.2, 'USDCAD': 1.5, 'EURGBP': 1.4,
}

all_results = []

for base_ccy, quote_ccy in pairs_to_test:
    pair = f"{base_ccy}{quote_ccy}"
    log(f"\nProcessing {pair}...")

    # Load MFC for both currencies at H1, H4, D1
    base_h1 = load_mfc(base_ccy, 'H1')
    base_h4 = load_mfc(base_ccy, 'H4')
    base_d1 = load_mfc(base_ccy, 'D1')
    quote_h1 = load_mfc(quote_ccy, 'H1')
    quote_h4 = load_mfc(quote_ccy, 'H4')
    quote_d1 = load_mfc(quote_ccy, 'D1')

    if any(x is None for x in [base_h1, base_h4, quote_h1, quote_h4]):
        log(f"  Missing MFC data")
        continue

    # Load price data at H1
    price_h1 = load_price(pair, 'H1')
    if price_h1 is None:
        log(f"  Missing price data")
        continue

    # Calculate velocities (shifted to avoid look-ahead)
    base_vel_h1 = base_h1.diff().shift(1)
    base_vel_h4 = base_h4.diff().shift(1)
    quote_vel_h1 = quote_h1.diff().shift(1)
    quote_vel_h4 = quote_h4.diff().shift(1)

    if base_d1 is not None and quote_d1 is not None:
        base_vel_d1 = base_d1.diff().shift(1)
        quote_vel_d1 = quote_d1.diff().shift(1)
    else:
        base_vel_d1 = None
        quote_vel_d1 = None

    pip_size = PIP_SIZE[pair]
    spread = SPREADS.get(pair, 1.5)

    # For each H1 bar, calculate 12-hour forward price movement
    results = []

    for h1_time in price_h1.index:
        try:
            entry_price = price_h1.loc[h1_time, 'Close']
        except:
            continue

        # Get 12-hour forward price (12 H1 bars ahead)
        forward_time = h1_time + pd.Timedelta(hours=12)

        # Find closest price within 12 hours
        future_prices = price_h1[(price_h1.index > h1_time) & (price_h1.index <= forward_time)]
        if len(future_prices) < 10:  # Need at least 10 of 12 bars
            continue

        exit_price = future_prices.iloc[-1]['Close']
        max_price = future_prices['High'].max()
        min_price = future_prices['Low'].min()

        # Calculate pip movement
        forward_pips = (exit_price - entry_price) / pip_size
        max_favorable_up = (max_price - entry_price) / pip_size
        max_favorable_down = (entry_price - min_price) / pip_size

        # Get velocities at signal time
        try:
            # H1 velocity (divergence: base - quote)
            b_vel_h1 = base_vel_h1.reindex([h1_time], method='ffill').iloc[0]
            q_vel_h1 = quote_vel_h1.reindex([h1_time], method='ffill').iloc[0]
            div_vel_h1 = b_vel_h1 - q_vel_h1

            # H4 velocity (need to align to H4 bar)
            h4_time = h1_time.floor('4h')
            b_vel_h4 = base_vel_h4.reindex([h4_time], method='ffill').iloc[0]
            q_vel_h4 = quote_vel_h4.reindex([h4_time], method='ffill').iloc[0]
            div_vel_h4 = b_vel_h4 - q_vel_h4

            # D1 velocity (if available)
            if base_vel_d1 is not None and quote_vel_d1 is not None:
                d1_time = h1_time.floor('D')
                b_vel_d1 = base_vel_d1.reindex([d1_time], method='ffill').iloc[0]
                q_vel_d1 = quote_vel_d1.reindex([d1_time], method='ffill').iloc[0]
                div_vel_d1 = b_vel_d1 - q_vel_d1
            else:
                div_vel_d1 = np.nan
        except:
            continue

        if pd.isna(div_vel_h1) or pd.isna(div_vel_h4):
            continue

        # Determine velocity directions
        h1_dir = 1 if div_vel_h1 > 0.01 else (-1 if div_vel_h1 < -0.01 else 0)
        h4_dir = 1 if div_vel_h4 > 0.01 else (-1 if div_vel_h4 < -0.01 else 0)
        d1_dir = 1 if div_vel_d1 > 0.01 else (-1 if div_vel_d1 < -0.01 else 0) if not pd.isna(div_vel_d1) else 0

        # Check alignment
        all_agree = (h1_dir == h4_dir == d1_dir) and h1_dir != 0
        h1_h4_agree = (h1_dir == h4_dir) and h1_dir != 0

        # Actual result
        price_went_up = forward_pips > 0

        results.append({
            'datetime': h1_time,
            'pair': pair,
            'div_vel_h1': div_vel_h1,
            'div_vel_h4': div_vel_h4,
            'div_vel_d1': div_vel_d1,
            'h1_dir': h1_dir,
            'h4_dir': h4_dir,
            'd1_dir': d1_dir,
            'all_agree': all_agree,
            'h1_h4_agree': h1_h4_agree,
            'forward_pips': forward_pips,
            'max_up': max_favorable_up,
            'max_down': max_favorable_down,
            'price_went_up': int(price_went_up),
        })

    df_results = pd.DataFrame(results)
    all_results.append(df_results)
    log(f"  {len(df_results):,} bars analyzed")

# Combine all results
df = pd.concat(all_results, ignore_index=True)
log(f"\nTotal bars: {len(df):,}")

# Analysis
log("\n" + "=" * 70)
log("WHEN ALL THREE AGREE (H1+H4+D1 velocity same direction)")
log("=" * 70)

all_agree = df[df['all_agree'] == True]
log(f"\nOccurrences: {len(all_agree):,} ({len(all_agree)/len(df)*100:.1f}%)")

up_signals = all_agree[all_agree['h1_dir'] == 1]
down_signals = all_agree[all_agree['h1_dir'] == -1]

log(f"\nALL UP (H1+H4+D1 velocity up):")
log(f"  Count: {len(up_signals):,}")
log(f"  Price went up (12h): {up_signals['price_went_up'].mean()*100:.1f}%")
log(f"  Avg forward pips: {up_signals['forward_pips'].mean():+.1f}")
log(f"  Avg max favorable (up): {up_signals['max_up'].mean():.1f}")
log(f"  Avg max adverse (down): {up_signals['max_down'].mean():.1f}")

log(f"\nALL DOWN (H1+H4+D1 velocity down):")
log(f"  Count: {len(down_signals):,}")
log(f"  Price went down (12h): {(1 - down_signals['price_went_up']).mean()*100:.1f}%")
log(f"  Avg forward pips: {down_signals['forward_pips'].mean():+.1f}")
log(f"  Avg max favorable (down): {down_signals['max_down'].mean():.1f}")
log(f"  Avg max adverse (up): {down_signals['max_up'].mean():.1f}")

# Trade simulation
log("\n" + "=" * 70)
log("SIMULATED TRADES (12h holding)")
log("=" * 70)

# When all agree UP -> Buy
buy_signals = all_agree[all_agree['h1_dir'] == 1].copy()
buy_signals['trade_pips'] = buy_signals['forward_pips'] - 1.0  # Avg spread

# When all agree DOWN -> Sell
sell_signals = all_agree[all_agree['h1_dir'] == -1].copy()
sell_signals['trade_pips'] = -sell_signals['forward_pips'] - 1.0  # Sell = inverse

all_trades = pd.concat([buy_signals, sell_signals])

log(f"\nTotal trades: {len(all_trades):,}")
log(f"Win rate: {(all_trades['trade_pips'] > 0).mean()*100:.1f}%")
log(f"Avg pips per trade: {all_trades['trade_pips'].mean():+.1f}")
log(f"Total pips: {all_trades['trade_pips'].sum():+,.0f}")

# By H1 velocity strength
log("\n" + "=" * 70)
log("BY H1 VELOCITY STRENGTH (when all agree)")
log("=" * 70)

log(f"\n{'H1 Velocity':<15} {'Trades':>8} {'Win %':>10} {'Avg Pips':>12} {'Total':>12}")
log("-" * 60)

for low, high, label in [(0.01, 0.03, '0.01-0.03'), (0.03, 0.05, '0.03-0.05'),
                          (0.05, 0.08, '0.05-0.08'), (0.08, 0.15, '0.08+')]:
    subset = all_agree[abs(all_agree['div_vel_h1']) >= low]
    subset = subset[abs(subset['div_vel_h1']) < high]

    if len(subset) > 50:
        # Trades
        buy_sub = subset[subset['h1_dir'] == 1].copy()
        buy_sub['trade_pips'] = buy_sub['forward_pips'] - 1.0
        sell_sub = subset[subset['h1_dir'] == -1].copy()
        sell_sub['trade_pips'] = -sell_sub['forward_pips'] - 1.0
        trades_sub = pd.concat([buy_sub, sell_sub])

        wr = (trades_sub['trade_pips'] > 0).mean() * 100
        avg = trades_sub['trade_pips'].mean()
        total = trades_sub['trade_pips'].sum()
        log(f"{label:<15} {len(trades_sub):>8,} {wr:>9.1f}% {avg:>+12.1f} {total:>+12,.0f}")

# Compare: H1 only vs H1+H4 vs H1+H4+D1
log("\n" + "=" * 70)
log("COMPARISON: ALIGNMENT LEVELS")
log("=" * 70)

# H1 only (velocity > 0.03)
h1_strong = df[abs(df['div_vel_h1']) > 0.03]
h1_buy = h1_strong[h1_strong['h1_dir'] == 1].copy()
h1_buy['trade_pips'] = h1_buy['forward_pips'] - 1.0
h1_sell = h1_strong[h1_strong['h1_dir'] == -1].copy()
h1_sell['trade_pips'] = -h1_sell['forward_pips'] - 1.0
h1_trades = pd.concat([h1_buy, h1_sell])

log(f"\nH1 velocity only (>0.03):")
log(f"  Trades: {len(h1_trades):,}")
log(f"  Win rate: {(h1_trades['trade_pips'] > 0).mean()*100:.1f}%")
log(f"  Avg pips: {h1_trades['trade_pips'].mean():+.1f}")
log(f"  Total: {h1_trades['trade_pips'].sum():+,.0f}")

# H1+H4 agree
h1_h4 = df[(df['h1_h4_agree'] == True) & (abs(df['div_vel_h1']) > 0.03)]
h1_h4_buy = h1_h4[h1_h4['h1_dir'] == 1].copy()
h1_h4_buy['trade_pips'] = h1_h4_buy['forward_pips'] - 1.0
h1_h4_sell = h1_h4[h1_h4['h1_dir'] == -1].copy()
h1_h4_sell['trade_pips'] = -h1_h4_sell['forward_pips'] - 1.0
h1_h4_trades = pd.concat([h1_h4_buy, h1_h4_sell])

log(f"\nH1+H4 agree (H1 vel>0.03):")
log(f"  Trades: {len(h1_h4_trades):,}")
log(f"  Win rate: {(h1_h4_trades['trade_pips'] > 0).mean()*100:.1f}%")
log(f"  Avg pips: {h1_h4_trades['trade_pips'].mean():+.1f}")
log(f"  Total: {h1_h4_trades['trade_pips'].sum():+,.0f}")

# H1+H4+D1 agree
all3 = df[(df['all_agree'] == True) & (abs(df['div_vel_h1']) > 0.03)]
all3_buy = all3[all3['h1_dir'] == 1].copy()
all3_buy['trade_pips'] = all3_buy['forward_pips'] - 1.0
all3_sell = all3[all3['h1_dir'] == -1].copy()
all3_sell['trade_pips'] = -all3_sell['forward_pips'] - 1.0
all3_trades = pd.concat([all3_buy, all3_sell])

log(f"\nH1+H4+D1 all agree (H1 vel>0.03):")
log(f"  Trades: {len(all3_trades):,}")
log(f"  Win rate: {(all3_trades['trade_pips'] > 0).mean()*100:.1f}%")
log(f"  Avg pips: {all3_trades['trade_pips'].mean():+.1f}")
log(f"  Total: {all3_trades['trade_pips'].sum():+,.0f}")

# By pair
log("\n" + "=" * 70)
log("BY PAIR (all agree, H1 vel>0.03)")
log("=" * 70)

log(f"\n{'Pair':<10} {'Trades':>8} {'Win %':>10} {'Avg Pips':>12} {'Total':>12}")
log("-" * 55)

for pair in df['pair'].unique():
    subset = all3[all3['pair'] == pair]
    if len(subset) > 20:
        buy_p = subset[subset['h1_dir'] == 1].copy()
        buy_p['trade_pips'] = buy_p['forward_pips'] - SPREADS.get(pair, 1.5)
        sell_p = subset[subset['h1_dir'] == -1].copy()
        sell_p['trade_pips'] = -sell_p['forward_pips'] - SPREADS.get(pair, 1.5)
        trades_p = pd.concat([buy_p, sell_p])

        wr = (trades_p['trade_pips'] > 0).mean() * 100
        avg = trades_p['trade_pips'].mean()
        total = trades_p['trade_pips'].sum()
        log(f"{pair:<10} {len(trades_p):>8,} {wr:>9.1f}% {avg:>+12.1f} {total:>+12,.0f}")

log("\nDONE")
