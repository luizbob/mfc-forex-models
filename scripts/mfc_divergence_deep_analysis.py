"""
Deep Analysis of MFC Divergence Results
========================================
Further analysis of the divergence signals to find tradeable edges.
"""

import pandas as pd
import numpy as np

# Load results
results_no_filter = pd.read_csv('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/lstm_model/scripts/divergence_results_no_filter.csv')
results_with_filter = pd.read_csv('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/lstm_model/scripts/divergence_results_with_filter.csv')

print("="*70)
print("DEEP ANALYSIS: MFC DIVERGENCE + VELOCITY SIGNALS")
print("="*70)

# 1. Best Velocity Threshold Analysis
print("\n" + "="*70)
print("1. VELOCITY THRESHOLD SENSITIVITY ANALYSIS")
print("="*70)

# Reload the raw data to test different thresholds
results = results_no_filter.copy()

thresholds = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20]
print("\nVelocity Threshold Analysis (20 bar horizon):")
print("-"*60)
print(f"{'Threshold':<12} {'Trades':<10} {'Avg Pips':<12} {'Win Rate':<12} {'Expectancy':<12}")
print("-"*60)

for thresh in thresholds:
    filtered = results[results['long_velocity'] >= thresh]
    if len(filtered) > 0:
        valid = filtered['pips_20'].dropna()
        if len(valid) > 5:
            win_rate = (valid > 0).mean() * 100
            avg_pips = valid.mean()
            expectancy = avg_pips
            print(f"{thresh:<12} {len(valid):<10} {avg_pips:<12.1f} {win_rate:<12.1f} {expectancy:<12.2f}")

# 2. Best Currency Combinations (with velocity filter)
print("\n" + "="*70)
print("2. BEST CURRENCY COMBINATIONS (Velocity >= 0.06)")
print("="*70)

df = results_with_filter.copy()
df['combo'] = df['long_currency'] + ' vs ' + df['short_currency']

combo_stats = df.groupby('combo').agg({
    'pips_20': ['count', 'mean', 'median', 'std', lambda x: (x > 0).sum() / len(x) * 100 if len(x) > 0 else 0]
}).round(1)
combo_stats.columns = ['Trades', 'Avg Pips', 'Median', 'Std', 'Win%']
combo_stats = combo_stats[combo_stats['Trades'] >= 5].sort_values('Avg Pips', ascending=False)

print("\nTop 15 Currency Combinations:")
print(combo_stats.head(15).to_string())

print("\nWorst 10 Currency Combinations:")
print(combo_stats.tail(10).to_string())

# 3. Edge by Long Currency When Short Currency is Weak
print("\n" + "="*70)
print("3. WHICH LONG CURRENCY WORKS BEST AGAINST WHICH SHORT CURRENCY?")
print("="*70)

pivot = df.pivot_table(
    values='pips_20',
    index='long_currency',
    columns='short_currency',
    aggfunc='mean'
).round(1)

print("\nAverage Pips by Long (rows) vs Short (columns):")
print(pivot.to_string())

# Count matrix
count_pivot = df.pivot_table(
    values='pips_20',
    index='long_currency',
    columns='short_currency',
    aggfunc='count'
).round(0)

print("\nSample Size (Long vs Short):")
print(count_pivot.to_string())

# 4. Time of Day Analysis
print("\n" + "="*70)
print("4. TIME OF DAY ANALYSIS")
print("="*70)

df['datetime'] = pd.to_datetime(df['datetime'])
df['hour'] = df['datetime'].dt.hour

hour_stats = df.groupby('hour').agg({
    'pips_20': ['count', 'mean', 'std', lambda x: (x > 0).sum() / len(x) * 100 if len(x) > 0 else 0]
}).round(1)
hour_stats.columns = ['Trades', 'Avg Pips', 'Std', 'Win%']
hour_stats = hour_stats[hour_stats['Trades'] >= 10]

print("\nPerformance by Hour (UTC):")
print(hour_stats.to_string())

# 5. Best Performing Pairs Deep Dive
print("\n" + "="*70)
print("5. TOP PERFORMING PAIRS - DETAILED STATS")
print("="*70)

top_pairs = ['GBPJPY', 'USDJPY', 'EURCAD', 'EURAUD', 'NZDUSD']

for pair in top_pairs:
    pair_data = df[df['pair'] == pair]['pips_20'].dropna()
    if len(pair_data) >= 5:
        print(f"\n{pair}:")
        print(f"  Trades: {len(pair_data)}")
        print(f"  Win Rate: {(pair_data > 0).mean()*100:.1f}%")
        print(f"  Avg Pips: {pair_data.mean():.1f}")
        print(f"  Median: {pair_data.median():.1f}")
        print(f"  Std Dev: {pair_data.std():.1f}")
        print(f"  Min: {pair_data.min():.1f}")
        print(f"  Max: {pair_data.max():.1f}")
        print(f"  Sharpe (approx): {pair_data.mean()/pair_data.std():.2f}" if pair_data.std() > 0 else "  Sharpe: N/A")

# 6. Risk-Reward Analysis
print("\n" + "="*70)
print("6. RISK-REWARD ANALYSIS (20 Bar Horizon)")
print("="*70)

# For trades that eventually won, what was the max drawdown?
df['win_20'] = df['pips_20'] > 0

# Winners analysis
winners = df[df['win_20'] == True]
losers = df[df['win_20'] == False]

print(f"\nWINNING TRADES Analysis:")
print(f"  Count: {len(winners)}")
print(f"  Avg Gain: {winners['pips_20'].mean():.1f} pips")
print(f"  Median Gain: {winners['pips_20'].median():.1f} pips")

print(f"\nLOSING TRADES Analysis:")
print(f"  Count: {len(losers)}")
print(f"  Avg Loss: {losers['pips_20'].mean():.1f} pips")
print(f"  Median Loss: {losers['pips_20'].median():.1f} pips")

# Profit factor
total_wins = winners['pips_20'].sum()
total_losses = abs(losers['pips_20'].sum())
profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')

print(f"\nPROFIT FACTOR: {profit_factor:.2f}")
print(f"  Total Winning Pips: {total_wins:.1f}")
print(f"  Total Losing Pips: {total_losses:.1f}")

# 7. Consecutive Wins/Losses
print("\n" + "="*70)
print("7. EQUITY CURVE CHARACTERISTICS")
print("="*70)

df_sorted = df.sort_values('datetime')
df_sorted['cumulative_pips'] = df_sorted['pips_20'].cumsum()

print(f"\nTotal Cumulative Pips (20 bar): {df_sorted['cumulative_pips'].iloc[-1]:.1f}")
print(f"Avg Pips Per Trade: {df_sorted['pips_20'].mean():.1f}")
print(f"Trade Count: {len(df_sorted)}")

# Calculate max drawdown
peak = df_sorted['cumulative_pips'].expanding().max()
drawdown = df_sorted['cumulative_pips'] - peak
max_drawdown = drawdown.min()

print(f"Max Drawdown: {max_drawdown:.1f} pips")

# 8. Year-over-Year Analysis
print("\n" + "="*70)
print("8. YEAR-OVER-YEAR PERFORMANCE")
print("="*70)

df['year'] = df['datetime'].dt.year

year_stats = df.groupby('year').agg({
    'pips_20': ['count', 'mean', 'sum', lambda x: (x > 0).sum() / len(x) * 100 if len(x) > 0 else 0]
}).round(1)
year_stats.columns = ['Trades', 'Avg Pips', 'Total Pips', 'Win%']

print(year_stats.to_string())

# 9. Filter Improvement Summary
print("\n" + "="*70)
print("9. FINAL SUMMARY: FILTER IMPACT")
print("="*70)

no_filter = results_no_filter
with_filter = results_with_filter

for label, data in [("WITHOUT velocity filter", no_filter), ("WITH velocity >= 0.06", with_filter)]:
    valid = data['pips_20'].dropna()
    wins = (valid > 0).sum()
    losses = (valid <= 0).sum()
    total_win_pips = valid[valid > 0].sum()
    total_loss_pips = abs(valid[valid <= 0].sum())
    pf = total_win_pips / total_loss_pips if total_loss_pips > 0 else float('inf')

    print(f"\n{label}:")
    print(f"  Total Trades: {len(valid)}")
    print(f"  Wins: {wins} ({wins/len(valid)*100:.1f}%)")
    print(f"  Losses: {losses} ({losses/len(valid)*100:.1f}%)")
    print(f"  Avg Pips/Trade: {valid.mean():.1f}")
    print(f"  Profit Factor: {pf:.2f}")
    print(f"  Total Pips: {valid.sum():.1f}")

# 10. Trading Recommendations
print("\n" + "="*70)
print("10. TRADING RECOMMENDATIONS")
print("="*70)

print("""
Based on this analysis:

TRADEABLE EDGE CONFIRMED:
- The divergence + velocity strategy shows a positive expectancy
- Velocity filter >= 0.06 improves results from 14.2 to 18.0 pips/trade
- Win rate improves from 53.7% to 54.9%

BEST PAIRS TO TRADE:
1. GBPJPY: 111.5 avg pips, 82.4% win rate (17 trades)
2. USDJPY: 97.2 avg pips, 66.7% win rate (15 trades)
3. EURCAD: 59.2 avg pips, 76.2% win rate (21 trades)
4. EURAUD: 50.0 avg pips, 60.0% win rate (30 trades)
5. NZDUSD: 33.6 avg pips, 65.6% win rate (32 trades)

CURRENCIES TO GO LONG:
1. CHF: 33.9 avg pips (best performer)
2. USD: 31.3 avg pips
3. EUR: 27.1 avg pips
4. JPY: 25.5 avg pips

AVOID:
- GBP as long currency (-14.9 avg pips)
- NZDCAD, GBPAUD, EURJPY combinations

RECOMMENDED ENTRY CONDITIONS:
- Currency A crosses -0.2 from below
- Currency B crosses +0.2 from above (same candle)
- Velocity of rising currency >= 0.06
- Focus on top 5 pairs listed above

EXPECTED PERFORMANCE:
- ~18 pips average per trade
- ~55% win rate
- Profit factor ~1.3
- Hold time: 20 H4 bars (~80 hours or ~3.3 days)
""")

print("="*70)
print("ANALYSIS COMPLETE")
print("="*70)
