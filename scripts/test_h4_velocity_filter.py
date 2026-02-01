"""
Test H4 velocity filter on LSTM trades
Check if filtering by H4 velocity removes bad trades
"""
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/data')
CLEANED_DIR = DATA_DIR / 'cleaned'
LSTM_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/lstm_model/data')

def load_mfc(currency, tf):
    try:
        df = pd.read_csv(CLEANED_DIR / f'mfc_currency_{currency}_{tf}_clean.csv')
        df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y.%m.%d %H:%M')
        df.set_index('datetime', inplace=True)
        return df['MFC']
    except:
        return None

# Load trades
trades = pd.read_csv(LSTM_DIR / 'trades_2025_oos.csv')
trades['entry_time'] = pd.to_datetime(trades['entry_time'])

print("=" * 80)
print("TEST: H4 Velocity Filter on LSTM Trades")
print("=" * 80)
print(f"\nTotal trades: {len(trades)}")
print(f"Win rate: {trades['win'].mean()*100:.1f}%")
print(f"Avg pips: {trades['net_pips'].mean():.1f}")
print(f"Total pips: {trades['net_pips'].sum():.0f}")

# Get currencies from pairs
def get_currencies(pair):
    currencies = ['EUR', 'USD', 'GBP', 'JPY', 'AUD', 'CAD', 'CHF', 'NZD']
    for base in currencies:
        for quote in currencies:
            if base != quote and pair == base + quote:
                return base, quote
    return None, None

# Load all MFC H4 data
mfc_h4 = {}
for ccy in ['EUR', 'USD', 'GBP', 'JPY', 'AUD', 'CAD', 'CHF', 'NZD']:
    mfc_h4[ccy] = load_mfc(ccy, 'H4')

# Calculate H4 velocity for each trade
h4_vel_base = []
h4_vel_quote = []

for idx, row in trades.iterrows():
    base, quote = get_currencies(row['pair'])
    entry = row['entry_time']

    # Get H4 velocity (6 bar change = 24 hours)
    if base and mfc_h4[base] is not None:
        mfc = mfc_h4[base]
        # Find closest H4 bar before entry
        mask = mfc.index <= entry
        if mask.sum() >= 7:
            recent = mfc[mask].iloc[-7:-1]  # Skip current bar, get 6 bars
            vel = recent.iloc[-1] - recent.iloc[0]  # 6 bar velocity
            h4_vel_base.append(vel)
        else:
            h4_vel_base.append(np.nan)
    else:
        h4_vel_base.append(np.nan)

    if quote and mfc_h4[quote] is not None:
        mfc = mfc_h4[quote]
        mask = mfc.index <= entry
        if mask.sum() >= 7:
            recent = mfc[mask].iloc[-7:-1]
            vel = recent.iloc[-1] - recent.iloc[0]
            h4_vel_quote.append(vel)
        else:
            h4_vel_quote.append(np.nan)
    else:
        h4_vel_quote.append(np.nan)

trades['h4_vel_base'] = h4_vel_base
trades['h4_vel_quote'] = h4_vel_quote
trades['h4_vel_diff'] = trades['h4_vel_base'] - trades['h4_vel_quote']

# Drop rows with missing H4 data
trades_clean = trades.dropna(subset=['h4_vel_base', 'h4_vel_quote'])
print(f"\nTrades with H4 data: {len(trades_clean)}")

# Separate BUY and SELL
buys = trades_clean[trades_clean['type'] == 'BUY']
sells = trades_clean[trades_clean['type'] == 'SELL']

print(f"\n{'='*80}")
print("CURRENT (NO H4 FILTER)")
print(f"{'='*80}")
print(f"BUY:  {len(buys)} trades, {buys['win'].mean()*100:.1f}% WR, {buys['net_pips'].mean():.1f} avg pips")
print(f"SELL: {len(sells)} trades, {sells['win'].mean()*100:.1f}% WR, {sells['net_pips'].mean():.1f} avg pips")

print(f"\n{'='*80}")
print("TEST H4 VELOCITY FILTERS")
print(f"{'='*80}")

# Test different H4 velocity filters
# For BUY: we want base going UP on H4 (vel > 0) or at least not falling fast
# For SELL: we want base going DOWN on H4 (vel < 0) or at least not rising fast

thresholds = [0, -0.05, -0.1, 0.05, 0.1]

print(f"\n--- BUY trades (filter: base H4 vel > threshold) ---")
print(f"{'Threshold':<12} {'Trades':>8} {'Removed':>8} {'WR':>8} {'Avg Pips':>10} {'Total':>10}")
print("-" * 60)

for thresh in thresholds:
    filtered = buys[buys['h4_vel_base'] > thresh]
    if len(filtered) > 0:
        wr = filtered['win'].mean() * 100
        avg = filtered['net_pips'].mean()
        total = filtered['net_pips'].sum()
        removed = len(buys) - len(filtered)
        print(f"{thresh:>12.2f} {len(filtered):>8} {removed:>8} {wr:>7.1f}% {avg:>+10.1f} {total:>+10.0f}")

print(f"\n--- BUY trades (filter: quote H4 vel < threshold) ---")
print(f"{'Threshold':<12} {'Trades':>8} {'Removed':>8} {'WR':>8} {'Avg Pips':>10} {'Total':>10}")
print("-" * 60)

for thresh in [0, 0.05, 0.1, -0.05]:
    filtered = buys[buys['h4_vel_quote'] < thresh]
    if len(filtered) > 0:
        wr = filtered['win'].mean() * 100
        avg = filtered['net_pips'].mean()
        total = filtered['net_pips'].sum()
        removed = len(buys) - len(filtered)
        print(f"{thresh:>12.2f} {len(filtered):>8} {removed:>8} {wr:>7.1f}% {avg:>+10.1f} {total:>+10.0f}")

print(f"\n--- BUY trades (combined: base vel > X AND quote vel < Y) ---")
print(f"{'Filter':<20} {'Trades':>8} {'Removed':>8} {'WR':>8} {'Avg Pips':>10} {'Total':>10}")
print("-" * 70)

for base_t, quote_t in [(0, 0), (-0.05, 0.05), (-0.1, 0.1), (0.05, -0.05)]:
    filtered = buys[(buys['h4_vel_base'] > base_t) & (buys['h4_vel_quote'] < quote_t)]
    if len(filtered) > 0:
        wr = filtered['win'].mean() * 100
        avg = filtered['net_pips'].mean()
        total = filtered['net_pips'].sum()
        removed = len(buys) - len(filtered)
        print(f"base>{base_t:.2f},quote<{quote_t:.2f}  {len(filtered):>8} {removed:>8} {wr:>7.1f}% {avg:>+10.1f} {total:>+10.0f}")

print(f"\n{'='*80}")
print("SELL TRADES")
print(f"{'='*80}")

print(f"\n--- SELL trades (filter: base H4 vel < threshold) ---")
print(f"{'Threshold':<12} {'Trades':>8} {'Removed':>8} {'WR':>8} {'Avg Pips':>10} {'Total':>10}")
print("-" * 60)

for thresh in [0, 0.05, 0.1, -0.05, -0.1]:
    filtered = sells[sells['h4_vel_base'] < thresh]
    if len(filtered) > 0:
        wr = filtered['win'].mean() * 100
        avg = filtered['net_pips'].mean()
        total = filtered['net_pips'].sum()
        removed = len(sells) - len(filtered)
        print(f"{thresh:>12.2f} {len(filtered):>8} {removed:>8} {wr:>7.1f}% {avg:>+10.1f} {total:>+10.0f}")

print(f"\n--- SELL trades (filter: quote H4 vel > threshold) ---")
print(f"{'Threshold':<12} {'Trades':>8} {'Removed':>8} {'WR':>8} {'Avg Pips':>10} {'Total':>10}")
print("-" * 60)

for thresh in [0, -0.05, -0.1, 0.05]:
    filtered = sells[sells['h4_vel_quote'] > thresh]
    if len(filtered) > 0:
        wr = filtered['win'].mean() * 100
        avg = filtered['net_pips'].mean()
        total = filtered['net_pips'].sum()
        removed = len(sells) - len(filtered)
        print(f"{thresh:>12.2f} {len(filtered):>8} {removed:>8} {wr:>7.1f}% {avg:>+10.1f} {total:>+10.0f}")

# Show some examples of trades that would be filtered
print(f"\n{'='*80}")
print("EXAMPLE: BUY trades with quote H4 vel > 0 (going against us)")
print(f"{'='*80}")

bad_buys = buys[buys['h4_vel_quote'] > 0.05].sort_values('net_pips')
if len(bad_buys) > 0:
    print(f"\nWorst BUY trades where quote H4 vel > 0.05:")
    print(bad_buys[['pair', 'entry_time', 'net_pips', 'win', 'h4_vel_base', 'h4_vel_quote']].head(10).to_string())

good_buys = buys[buys['h4_vel_quote'] < 0].sort_values('net_pips', ascending=False)
if len(good_buys) > 0:
    print(f"\nBest BUY trades where quote H4 vel < 0:")
    print(good_buys[['pair', 'entry_time', 'net_pips', 'win', 'h4_vel_base', 'h4_vel_quote']].head(10).to_string())
