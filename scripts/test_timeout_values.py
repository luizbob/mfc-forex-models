"""
Test different timeout values to find the optimal one.
"""
import pandas as pd
import numpy as np
from pathlib import Path

def log(msg=""):
    print(msg, flush=True)

LSTM_DATA_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/lstm_model/data')

# Load the trades with all details
trades_df = pd.read_csv(LSTM_DATA_DIR / 'trades_2025_oos.csv')

trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])
trades_df['hold_minutes'] = (trades_df['exit_time'] - trades_df['entry_time']).dt.total_seconds() / 60
trades_df['hold_bars'] = trades_df['hold_minutes'] / 5  # M5 bars

log("=" * 70)
log("TIMEOUT OPTIMIZATION ANALYSIS")
log("=" * 70)

log(f"\nCurrent trades: {len(trades_df)}")
log(f"RSI exits: {len(trades_df[trades_df['exit_reason'] == 'RSI'])}")
log(f"Timeouts: {len(trades_df[trades_df['exit_reason'] == 'TIMEOUT'])}")

# Analyze hold time distribution for RSI exits (winners)
rsi_trades = trades_df[trades_df['exit_reason'] == 'RSI']
rsi_winners = rsi_trades[rsi_trades['win'] == 1]
rsi_losers = rsi_trades[rsi_trades['win'] == 0]

log("\n--- RSI Exit Hold Time Distribution ---")
log(f"Winners: avg {rsi_winners['hold_bars'].mean():.0f} bars ({rsi_winners['hold_minutes'].mean()/60:.1f}h), median {rsi_winners['hold_bars'].median():.0f} bars")
log(f"Losers:  avg {rsi_losers['hold_bars'].mean():.0f} bars ({rsi_losers['hold_minutes'].mean()/60:.1f}h), median {rsi_losers['hold_bars'].median():.0f} bars")

# Percentiles
log("\n--- RSI Winners Hold Time Percentiles ---")
for p in [50, 75, 90, 95, 99]:
    bars = rsi_winners['hold_bars'].quantile(p/100)
    hours = bars * 5 / 60
    log(f"  {p}th percentile: {bars:.0f} bars ({hours:.1f} hours)")

log("\n" + "=" * 70)
log("SIMULATING DIFFERENT TIMEOUT VALUES")
log("=" * 70)

# For each timeout, calculate what would happen
# RSI exits that happen before timeout: keep as is
# RSI exits that happen after timeout: would become timeout (use timeout price)
# Timeouts: already timeout

log(f"\n{'Timeout':<12} {'Hours':<8} {'Trades':>8} {'TO%':>8} {'WR':>8} {'Net':>10} {'Total':>10}")
log("-" * 75)

# Current results (200 bars)
current_to = (trades_df['exit_reason'] == 'TIMEOUT').sum()
current_to_pct = current_to / len(trades_df) * 100
log(f"{'200 bars':<12} {'16.7h':<8} {len(trades_df):>8} {current_to_pct:>7.1f}% {trades_df['win'].mean()*100:>7.1f}% {trades_df['net_pips'].mean():>+9.2f} {trades_df['net_pips'].sum():>+9.0f}")

# Test different timeouts by looking at RSI exits
# We can simulate shorter timeouts by checking if RSI exit happened within the timeout window
# For trades that would timeout earlier, we'd need the actual price data which we don't have here

# Instead, let's analyze what % of RSI exits happen within each timeout window
log("\n--- RSI Exits by Hold Time ---")
log(f"{'Max Bars':<12} {'Hours':<8} {'RSI Exits':>10} {'% of RSI':>10} {'Avg Net':>10}")
log("-" * 55)

for max_bars in [50, 75, 100, 125, 150, 175, 200, 250, 300]:
    within = rsi_trades[rsi_trades['hold_bars'] <= max_bars]
    pct = len(within) / len(rsi_trades) * 100
    hours = max_bars * 5 / 60
    avg_net = within['net_pips'].mean() if len(within) > 0 else 0
    log(f"{max_bars:<12} {hours:.1f}h{'':<5} {len(within):>10} {pct:>9.1f}% {avg_net:>+9.2f}")

# Detailed analysis of timeout trades
log("\n" + "=" * 70)
log("TIMEOUT TRADE ANALYSIS")
log("=" * 70)

timeout_trades = trades_df[trades_df['exit_reason'] == 'TIMEOUT']

log(f"\nTimeout trades: {len(timeout_trades)}")
log(f"Timeout winners: {timeout_trades['win'].sum()} ({timeout_trades['win'].mean()*100:.1f}%)")
log(f"Timeout losers: {len(timeout_trades) - timeout_trades['win'].sum()}")
log(f"Timeout avg pips: {timeout_trades['pips'].mean():.2f}")
log(f"Timeout net avg: {timeout_trades['net_pips'].mean():.2f}")
log(f"Timeout total: {timeout_trades['net_pips'].sum():.0f}")

# What if we just skip timeout trades entirely?
rsi_only = trades_df[trades_df['exit_reason'] == 'RSI']
log(f"\n--- If we only count RSI exits ---")
log(f"Trades: {len(rsi_only)}")
log(f"WR: {rsi_only['win'].mean()*100:.1f}%")
log(f"Net avg: {rsi_only['net_pips'].mean():.2f}")
log(f"Total: {rsi_only['net_pips'].sum():.0f}")

# Timeout trade pips distribution
log("\n--- Timeout Pips Distribution ---")
log(f"Min: {timeout_trades['pips'].min():.1f}")
log(f"25th: {timeout_trades['pips'].quantile(0.25):.1f}")
log(f"Median: {timeout_trades['pips'].median():.1f}")
log(f"75th: {timeout_trades['pips'].quantile(0.75):.1f}")
log(f"Max: {timeout_trades['pips'].max():.1f}")

# How many timeout trades are big losers?
log("\n--- Timeout Trade Size Breakdown ---")
for threshold in [-50, -30, -20, -10, 0, 10, 20]:
    count = len(timeout_trades[timeout_trades['pips'] > threshold])
    pct = count / len(timeout_trades) * 100
    log(f"  pips > {threshold:>4}: {count:>4} trades ({pct:>5.1f}%)")

log("\n" + "=" * 70)
log("RECOMMENDATION")
log("=" * 70)

log("""
Analysis shows:
- 95% of RSI winners exit within ~165 bars (13.8 hours)
- 90% of RSI winners exit within ~115 bars (9.6 hours)

Current timeout (200 bars / 16.7h) captures almost all RSI exits.
The 204 timeout trades lose -26.19 pips on average.

Options:
1. Keep 200 bars - captures maximum RSI exits
2. Try 150 bars (12.5h) - might cut some losers earlier
3. Try 100 bars (8.3h) - more aggressive, but loses some winners
""")

log(f"\nAnalysis complete!")
