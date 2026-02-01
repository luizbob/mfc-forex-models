"""
Test USD MFC vs Gold - M30 with Moving Averages
================================================
Smaller timeframe + MA50/MA200 as trend filters
"""
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from pathlib import Path

def log(msg=""):
    print(msg, flush=True)

log("=" * 70)
log("USD MFC vs GOLD - M30 with MAs")
log("=" * 70)

DATA_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/data')
CLEANED_DIR = DATA_DIR / 'cleaned'

# Parameters
VELOCITY_THRESHOLD = 0.03  # Lower for M30
MAX_HOLD_BARS = 12  # 6 hours on M30
GOLD_SPREAD = 30  # Gold spread in pips

def load_mfc(currency, timeframe):
    fp = CLEANED_DIR / f'mfc_currency_{currency}_{timeframe}_clean.csv'
    if not fp.exists():
        return None
    df = pd.read_csv(fp)
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df = df.set_index('datetime').sort_index()
    return df['MFC']

# Load USD MFC
log("\nLoading data...")
usd_m30 = load_mfc('USD', 'M30')
usd_h1 = load_mfc('USD', 'H1')

if usd_m30 is None:
    log("No M30 MFC, using H1...")
    usd_m30 = usd_h1

# Load Gold M5 and resample to M30
gold_fp = DATA_DIR / 'XAUUSDm_M5_202110271720_202501172155.csv'
gold_df = pd.read_csv(gold_fp, sep='\t')
gold_df.columns = ['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'TickVol', 'Vol', 'Spread']
gold_df['datetime'] = pd.to_datetime(gold_df['Date'].str.replace('.', '-', regex=False) + ' ' + gold_df['Time'])
gold_df = gold_df.set_index('datetime').sort_index()

gold_m30 = gold_df.resample('30min').agg({
    'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
}).dropna()

log(f"USD MFC: {len(usd_m30):,} bars")
log(f"Gold M30: {len(gold_m30):,} bars")

# Calculate MAs on Gold
gold_m30['ma50'] = gold_m30['Close'].rolling(50).mean()
gold_m30['ma200'] = gold_m30['Close'].rolling(200).mean()
gold_m30['above_ma50'] = (gold_m30['Close'] > gold_m30['ma50']).astype(int)
gold_m30['above_ma200'] = (gold_m30['Close'] > gold_m30['ma200']).astype(int)
gold_m30['ma_trend'] = gold_m30['above_ma50'] + gold_m30['above_ma200']  # 0, 1, or 2

# USD velocity
usd_vel_raw = usd_m30.diff()
usd_vel_shifted = usd_m30.diff().shift(1)

# Align data
df = pd.DataFrame(index=gold_m30.index)
df['gold_open'] = gold_m30['Open']
df['gold_close'] = gold_m30['Close']
df['gold_ma50'] = gold_m30['ma50']
df['gold_ma200'] = gold_m30['ma200']
df['above_ma50'] = gold_m30['above_ma50']
df['above_ma200'] = gold_m30['above_ma200']
df['ma_trend'] = gold_m30['ma_trend']
df['usd_vel_shifted'] = usd_vel_shifted.reindex(df.index, method='ffill')
df['usd_vel_raw'] = usd_vel_raw.reindex(df.index, method='ffill')
df = df.dropna()

log(f"Aligned data: {len(df):,} bars")
log(f"Date range: {df.index.min()} to {df.index.max()}")

# Test different exit strategies
def run_backtest(df, exit_type='fixed', exit_bars=4, use_ma_filter=False):
    """Run backtest with different parameters"""

    gold_open = df['gold_open'].values
    gold_close = df['gold_close'].values
    usd_vel_shifted = df['usd_vel_shifted'].values
    usd_vel_raw = df['usd_vel_raw'].values
    above_ma50 = df['above_ma50'].values
    above_ma200 = df['above_ma200'].values
    dates = df.index

    all_trades = []
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

        # MA filter
        if use_ma_filter:
            if signal == 'buy' and above_ma200[i] == 0:
                # Don't buy if below MA200 (downtrend)
                i += 1
                continue
            if signal == 'sell' and above_ma200[i] == 1:
                # Don't sell if above MA200 (uptrend)
                i += 1
                continue

        # Entry at next bar open
        entry_idx = i + 1
        entry_price = gold_open[entry_idx]

        # Find exit based on exit_type
        exit_idx = entry_idx

        if exit_type == 'fixed':
            exit_idx = min(entry_idx + exit_bars, n - 1)

        elif exit_type == 'velocity':
            for j in range(entry_idx, min(entry_idx + MAX_HOLD_BARS, n)):
                current_usd_vel = usd_vel_raw[j]
                if signal == 'sell':
                    if current_usd_vel < 0.01:
                        exit_idx = j
                        break
                else:
                    if current_usd_vel > -0.01:
                        exit_idx = j
                        break
                exit_idx = j

        elif exit_type == 'ma_cross':
            # Exit when price crosses MA50
            for j in range(entry_idx, min(entry_idx + MAX_HOLD_BARS, n)):
                if signal == 'sell':
                    # Exit sell if price goes above MA50
                    if gold_close[j] > df['gold_ma50'].iloc[j]:
                        exit_idx = j
                        break
                else:
                    # Exit buy if price goes below MA50
                    if gold_close[j] < df['gold_ma50'].iloc[j]:
                        exit_idx = j
                        break
                exit_idx = j

        # Calculate PnL
        exit_price = gold_close[exit_idx]
        bars_held = exit_idx - entry_idx + 1

        raw_pips = (exit_price - entry_price) / 0.1

        if signal == 'sell':
            adjusted_pips = -raw_pips
        else:
            adjusted_pips = raw_pips

        net_pips = adjusted_pips - GOLD_SPREAD

        all_trades.append({
            'datetime': dates[i],
            'signal': signal,
            'bars_held': bars_held,
            'net_pips': net_pips,
            'is_profitable': int(net_pips > 0),
        })

        i = exit_idx + 1

    return pd.DataFrame(all_trades)


# Test 1: Fixed bar exits
log("\n" + "=" * 70)
log("TEST 1: FIXED BAR EXITS (no MA filter)")
log("=" * 70)

log(f"\n{'Bars':<8} {'Trades':>8} {'Win %':>10} {'Avg Pips':>12} {'Total':>12}")
log("-" * 55)

for bars in [1, 2, 4, 6, 8, 12]:
    trades = run_backtest(df, exit_type='fixed', exit_bars=bars, use_ma_filter=False)
    if len(trades) > 0:
        wr = trades['is_profitable'].mean() * 100
        avg = trades['net_pips'].mean()
        total = trades['net_pips'].sum()
        log(f"{bars:<8} {len(trades):>8,} {wr:>9.1f}% {avg:>+12.1f} {total:>+12,.0f}")


# Test 2: Fixed bar exits WITH MA filter
log("\n" + "=" * 70)
log("TEST 2: FIXED BAR EXITS (with MA200 filter)")
log("=" * 70)
log("Only buy if above MA200, only sell if below MA200")

log(f"\n{'Bars':<8} {'Trades':>8} {'Win %':>10} {'Avg Pips':>12} {'Total':>12}")
log("-" * 55)

for bars in [1, 2, 4, 6, 8, 12]:
    trades = run_backtest(df, exit_type='fixed', exit_bars=bars, use_ma_filter=True)
    if len(trades) > 0:
        wr = trades['is_profitable'].mean() * 100
        avg = trades['net_pips'].mean()
        total = trades['net_pips'].sum()
        log(f"{bars:<8} {len(trades):>8,} {wr:>9.1f}% {avg:>+12.1f} {total:>+12,.0f}")


# Test 3: Velocity exit
log("\n" + "=" * 70)
log("TEST 3: VELOCITY EXIT")
log("=" * 70)

trades = run_backtest(df, exit_type='velocity', use_ma_filter=False)
if len(trades) > 0:
    log(f"\nNo MA filter:")
    log(f"  Trades: {len(trades):,}")
    log(f"  Win rate: {trades['is_profitable'].mean()*100:.1f}%")
    log(f"  Avg pips: {trades['net_pips'].mean():+.1f}")
    log(f"  Total: {trades['net_pips'].sum():+,.0f}")
    log(f"  Avg bars held: {trades['bars_held'].mean():.1f}")

trades = run_backtest(df, exit_type='velocity', use_ma_filter=True)
if len(trades) > 0:
    log(f"\nWith MA200 filter:")
    log(f"  Trades: {len(trades):,}")
    log(f"  Win rate: {trades['is_profitable'].mean()*100:.1f}%")
    log(f"  Avg pips: {trades['net_pips'].mean():+.1f}")
    log(f"  Total: {trades['net_pips'].sum():+,.0f}")
    log(f"  Avg bars held: {trades['bars_held'].mean():.1f}")


# Test 4: Check if trend alignment helps
log("\n" + "=" * 70)
log("TEST 4: TREND ALIGNMENT CHECK")
log("=" * 70)

trades = run_backtest(df, exit_type='fixed', exit_bars=4, use_ma_filter=False)
trades['datetime'] = pd.to_datetime(trades['datetime'])
trades = trades.merge(
    df[['above_ma50', 'above_ma200', 'ma_trend']].reset_index(),
    left_on='datetime', right_on='datetime', how='left'
)

log("\nBUY signals by MA alignment:")
buys = trades[trades['signal'] == 'buy']
for ma_trend in [0, 1, 2]:
    subset = buys[buys['ma_trend'] == ma_trend]
    if len(subset) > 10:
        label = {0: 'Below both MAs', 1: 'Between MAs', 2: 'Above both MAs'}[ma_trend]
        wr = subset['is_profitable'].mean() * 100
        avg = subset['net_pips'].mean()
        log(f"  {label}: {len(subset):,} trades, {wr:.1f}% WR, {avg:+.1f} avg")

log("\nSELL signals by MA alignment:")
sells = trades[trades['signal'] == 'sell']
for ma_trend in [0, 1, 2]:
    subset = sells[sells['ma_trend'] == ma_trend]
    if len(subset) > 10:
        label = {0: 'Below both MAs', 1: 'Between MAs', 2: 'Above both MAs'}[ma_trend]
        wr = subset['is_profitable'].mean() * 100
        avg = subset['net_pips'].mean()
        log(f"  {label}: {len(subset):,} trades, {wr:.1f}% WR, {avg:+.1f} avg")

log("\nDONE")
