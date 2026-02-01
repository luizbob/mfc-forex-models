"""
Test USD MFC Velocity vs Gold (XAUUSD) Correlation
===================================================
Hypothesis: When USD velocity is high (USD strengthening fast), Gold drops.
"""
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from pathlib import Path

def log(msg=""):
    print(msg, flush=True)

log("=" * 70)
log("USD MFC VELOCITY vs GOLD PRICE CORRELATION")
log("=" * 70)

DATA_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/data')
CLEANED_DIR = DATA_DIR / 'cleaned'

# Load USD MFC for different timeframes
log("\nLoading USD MFC data...")

def load_mfc(currency, timeframe):
    fp = CLEANED_DIR / f'mfc_currency_{currency}_{timeframe}_clean.csv'
    if not fp.exists():
        return None
    df = pd.read_csv(fp)
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df = df.set_index('datetime').sort_index()
    return df['MFC']

usd_h1 = load_mfc('USD', 'H1')
usd_h4 = load_mfc('USD', 'H4')
usd_d1 = load_mfc('USD', 'D1')

log(f"USD H1: {len(usd_h1):,} bars")
log(f"USD H4: {len(usd_h4):,} bars")
log(f"USD D1: {len(usd_d1):,} bars" if usd_d1 is not None else "USD D1: Not found")

# Load Gold price
log("\nLoading Gold (XAUUSD) data...")
gold_fp = DATA_DIR / 'XAUUSDm_M5_202110271720_202501172155.csv'
gold_df = pd.read_csv(gold_fp, sep='\t')
gold_df.columns = ['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'TickVol', 'Vol', 'Spread']
gold_df['datetime'] = pd.to_datetime(gold_df['Date'].str.replace('.', '-', regex=False) + ' ' + gold_df['Time'])
gold_df = gold_df.set_index('datetime').sort_index()

log(f"Gold M5: {len(gold_df):,} bars")
log(f"Date range: {gold_df.index.min()} to {gold_df.index.max()}")

# Resample Gold to H1 and H4
gold_h1 = gold_df.resample('1h').agg({
    'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
}).dropna()

gold_h4 = gold_df.resample('4h').agg({
    'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
}).dropna()

log(f"Gold H1: {len(gold_h1):,} bars")
log(f"Gold H4: {len(gold_h4):,} bars")

# Calculate USD velocity
usd_vel_h1 = usd_h1.diff()
usd_vel_h4 = usd_h4.diff()

# Calculate Gold returns (in pips, 1 pip = 0.1 for Gold)
gold_h1['return_pips'] = (gold_h1['Close'] - gold_h1['Open']) / 0.1
gold_h4['return_pips'] = (gold_h4['Close'] - gold_h4['Open']) / 0.1

# Align data
log("\n" + "=" * 70)
log("H4 ANALYSIS")
log("=" * 70)

# Create aligned dataframe
df_h4 = pd.DataFrame(index=gold_h4.index)
df_h4['gold_return'] = gold_h4['return_pips']
df_h4['gold_close'] = gold_h4['Close']
df_h4['usd_vel'] = usd_vel_h4.shift(1).reindex(df_h4.index, method='ffill')  # Shifted to avoid look-ahead
df_h4['usd_mfc'] = usd_h4.shift(1).reindex(df_h4.index, method='ffill')
df_h4 = df_h4.dropna()

log(f"\nAligned H4 data: {len(df_h4):,} bars")

# Correlation
corr = df_h4['usd_vel'].corr(df_h4['gold_return'])
log(f"\nCorrelation (USD velocity vs Gold return): {corr:.4f}")

# When USD velocity is high positive (USD strengthening fast)
log("\n" + "-" * 50)
log("When USD velocity > 0.05 (USD strengthening fast):")
usd_up_fast = df_h4[df_h4['usd_vel'] > 0.05]
log(f"  Count: {len(usd_up_fast):,}")
log(f"  Avg Gold return: {usd_up_fast['gold_return'].mean():+.1f} pips")
log(f"  Gold down %: {(usd_up_fast['gold_return'] < 0).mean()*100:.1f}%")

log("\nWhen USD velocity < -0.05 (USD weakening fast):")
usd_down_fast = df_h4[df_h4['usd_vel'] < -0.05]
log(f"  Count: {len(usd_down_fast):,}")
log(f"  Avg Gold return: {usd_down_fast['gold_return'].mean():+.1f} pips")
log(f"  Gold up %: {(usd_down_fast['gold_return'] > 0).mean()*100:.1f}%")

log("\nWhen USD velocity neutral (-0.02 to 0.02):")
usd_neutral = df_h4[(df_h4['usd_vel'] >= -0.02) & (df_h4['usd_vel'] <= 0.02)]
log(f"  Count: {len(usd_neutral):,}")
log(f"  Avg Gold return: {usd_neutral['gold_return'].mean():+.1f} pips")

# Test trading strategy
log("\n" + "=" * 70)
log("SIMPLE STRATEGY TEST")
log("=" * 70)

log("\nStrategy: Sell Gold when USD velocity > 0.05, Buy Gold when USD velocity < -0.05")

# Sell Gold when USD up fast
sells = df_h4[df_h4['usd_vel'] > 0.05].copy()
sells['pnl'] = -sells['gold_return']  # Selling = inverse of return
log(f"\nSELL signals: {len(sells):,}")
log(f"  Win rate: {(sells['pnl'] > 0).mean()*100:.1f}%")
log(f"  Avg pips: {sells['pnl'].mean():+.1f}")
log(f"  Total pips: {sells['pnl'].sum():+,.0f}")

# Buy Gold when USD down fast
buys = df_h4[df_h4['usd_vel'] < -0.05].copy()
buys['pnl'] = buys['gold_return']  # Buying = same as return
log(f"\nBUY signals: {len(buys):,}")
log(f"  Win rate: {(buys['pnl'] > 0).mean()*100:.1f}%")
log(f"  Avg pips: {buys['pnl'].mean():+.1f}")
log(f"  Total pips: {buys['pnl'].sum():+,.0f}")

# Combined
combined_pnl = sells['pnl'].sum() + buys['pnl'].sum()
combined_trades = len(sells) + len(buys)
log(f"\nCOMBINED:")
log(f"  Total trades: {combined_trades:,}")
log(f"  Total pips: {combined_pnl:+,.0f}")
log(f"  Avg per trade: {combined_pnl/combined_trades:+.1f}")

# By velocity threshold
log("\n" + "=" * 70)
log("BY USD VELOCITY THRESHOLD")
log("=" * 70)

log(f"\n{'Threshold':<12} {'Sells':>8} {'Sell WR':>10} {'Buys':>8} {'Buy WR':>10} {'Total Pips':>12}")
log("-" * 65)

for thresh in [0.03, 0.05, 0.08, 0.10, 0.12]:
    sells = df_h4[df_h4['usd_vel'] > thresh]
    buys = df_h4[df_h4['usd_vel'] < -thresh]

    if len(sells) > 10 and len(buys) > 10:
        sell_pnl = -sells['gold_return']
        buy_pnl = buys['gold_return']

        sell_wr = (sell_pnl > 0).mean() * 100
        buy_wr = (buy_pnl > 0).mean() * 100
        total = sell_pnl.sum() + buy_pnl.sum()

        log(f"{thresh:<12} {len(sells):>8,} {sell_wr:>9.1f}% {len(buys):>8,} {buy_wr:>9.1f}% {total:>+12,.0f}")

# Check Daily MFC
if usd_d1 is not None:
    log("\n" + "=" * 70)
    log("DAILY MFC CHECK")
    log("=" * 70)

    usd_d1_vel = usd_d1.diff()

    df_d1 = pd.DataFrame(index=gold_h4.index)
    df_d1['gold_return'] = gold_h4['return_pips']
    df_d1['usd_d1'] = usd_d1.shift(1).reindex(df_d1.index, method='ffill')
    df_d1['usd_d1_vel'] = usd_d1_vel.shift(1).reindex(df_d1.index, method='ffill')
    df_d1 = df_d1.dropna()

    log(f"\nDaily MFC position check:")
    log(f"\nWhen D1 USD MFC > 0.2 (USD overbought):")
    overbought = df_d1[df_d1['usd_d1'] > 0.2]
    log(f"  Count: {len(overbought):,}")
    log(f"  Avg Gold return: {overbought['gold_return'].mean():+.1f} pips")

    log(f"\nWhen D1 USD MFC < -0.2 (USD oversold):")
    oversold = df_d1[df_d1['usd_d1'] < -0.2]
    log(f"  Count: {len(oversold):,}")
    log(f"  Avg Gold return: {oversold['gold_return'].mean():+.1f} pips")

log("\nDONE")
