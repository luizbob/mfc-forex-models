"""
Analyze BOTH base and quote MFC velocity patterns for timeout trades.
"""
import pandas as pd
import numpy as np
from pathlib import Path

def log(msg=""):
    print(msg, flush=True)

LSTM_DATA_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/lstm_model/data')

trades_df = pd.read_csv(LSTM_DATA_DIR / 'trades_mfc_analysis.csv')

log("=" * 70)
log("BASE vs QUOTE VELOCITY ANALYSIS")
log("=" * 70)

log(f"\nTotal trades: {len(trades_df)}")
log(f"Timeout: {len(trades_df[trades_df['exit_reason'] == 'TIMEOUT'])} ({len(trades_df[trades_df['exit_reason'] == 'TIMEOUT'])/len(trades_df)*100:.1f}%)")

# ============================================================================
# BASE VELOCITY ANALYSIS
# ============================================================================
log("\n" + "=" * 70)
log("1. BASE VELOCITY H1 ANALYSIS")
log("=" * 70)

# For BUY: base should be RISING (vel > 0) - we're betting base will go up
# For SELL: base should be FALLING (vel < 0) - we're betting base will go down

trades_df['base_vel_aligned'] = (
    ((trades_df['type'] == 'BUY') & (trades_df['base_vel_h1'] > 0)) |
    ((trades_df['type'] == 'SELL') & (trades_df['base_vel_h1'] < 0))
)

log("\n--- Base Velocity H1 Alignment ---")
log("BUY: base should be rising (vel > 0)")
log("SELL: base should be falling (vel < 0)")
log(f"\n{'Aligned':<15} {'Trades':>8} {'TO%':>8} {'WR':>8} {'Net':>10} {'Total':>10}")
log("-" * 65)

for aligned, label in [(True, 'Aligned'), (False, 'Not Aligned')]:
    df = trades_df[trades_df['base_vel_aligned'] == aligned]
    to_pct = (df['exit_reason'] == 'TIMEOUT').mean() * 100
    wr = df['win'].mean() * 100
    net = df['net_pips'].mean()
    total = df['net_pips'].sum()
    log(f"{label:<15} {len(df):>8} {to_pct:>7.1f}% {wr:>7.1f}% {net:>+9.2f} {total:>+9.0f}")

# Base velocity thresholds
log("\n--- Base Velocity H1 Thresholds ---")
log(f"{'Threshold':<25} {'Trades':>8} {'TO%':>8} {'WR':>8} {'Net':>10}")
log("-" * 60)

buy_trades = trades_df[trades_df['type'] == 'BUY']
for threshold in [0, 0.01, 0.02, 0.03, 0.04, 0.05]:
    df = buy_trades[buy_trades['base_vel_h1'] > threshold]
    if len(df) > 50:
        to_pct = (df['exit_reason'] == 'TIMEOUT').mean() * 100
        wr = df['win'].mean() * 100
        net = df['net_pips'].mean()
        log(f"BUY: base_vel > {threshold:<6} {len(df):>8} {to_pct:>7.1f}% {wr:>7.1f}% {net:>+9.2f}")

sell_trades = trades_df[trades_df['type'] == 'SELL']
for threshold in [0, -0.01, -0.02, -0.03, -0.04, -0.05]:
    df = sell_trades[sell_trades['base_vel_h1'] < threshold]
    if len(df) > 50:
        to_pct = (df['exit_reason'] == 'TIMEOUT').mean() * 100
        wr = df['win'].mean() * 100
        net = df['net_pips'].mean()
        log(f"SELL: base_vel < {threshold:<6} {len(df):>8} {to_pct:>7.1f}% {wr:>7.1f}% {net:>+9.2f}")

# ============================================================================
# QUOTE VELOCITY ANALYSIS (for comparison)
# ============================================================================
log("\n" + "=" * 70)
log("2. QUOTE VELOCITY H1 ANALYSIS")
log("=" * 70)

# For BUY: quote should be FALLING (vel < 0) - quote weakening helps pair rise
# For SELL: quote should be RISING (vel > 0) - quote strengthening helps pair fall

trades_df['quote_vel_aligned'] = (
    ((trades_df['type'] == 'BUY') & (trades_df['quote_vel_h1'] < 0)) |
    ((trades_df['type'] == 'SELL') & (trades_df['quote_vel_h1'] > 0))
)

log("\n--- Quote Velocity H1 Alignment ---")
log("BUY: quote should be falling (vel < 0)")
log("SELL: quote should be rising (vel > 0)")
log(f"\n{'Aligned':<15} {'Trades':>8} {'TO%':>8} {'WR':>8} {'Net':>10} {'Total':>10}")
log("-" * 65)

for aligned, label in [(True, 'Aligned'), (False, 'Not Aligned')]:
    df = trades_df[trades_df['quote_vel_aligned'] == aligned]
    to_pct = (df['exit_reason'] == 'TIMEOUT').mean() * 100
    wr = df['win'].mean() * 100
    net = df['net_pips'].mean()
    total = df['net_pips'].sum()
    log(f"{label:<15} {len(df):>8} {to_pct:>7.1f}% {wr:>7.1f}% {net:>+9.2f} {total:>+9.0f}")

# ============================================================================
# COMBINED: BASE AND QUOTE VELOCITY
# ============================================================================
log("\n" + "=" * 70)
log("3. COMBINED BASE + QUOTE VELOCITY")
log("=" * 70)

log("\n--- Both Currencies Velocity Alignment ---")
log(f"\n{'Scenario':<25} {'Trades':>8} {'TO%':>8} {'WR':>8} {'Net':>10} {'Total':>10}")
log("-" * 70)

# Both aligned
both = trades_df[(trades_df['base_vel_aligned'] == True) & (trades_df['quote_vel_aligned'] == True)]
base_only = trades_df[(trades_df['base_vel_aligned'] == True) & (trades_df['quote_vel_aligned'] == False)]
quote_only = trades_df[(trades_df['base_vel_aligned'] == False) & (trades_df['quote_vel_aligned'] == True)]
neither = trades_df[(trades_df['base_vel_aligned'] == False) & (trades_df['quote_vel_aligned'] == False)]

for df, label in [(both, 'Both Aligned'), (base_only, 'Base Only'), (quote_only, 'Quote Only'), (neither, 'Neither')]:
    if len(df) > 30:
        to_pct = (df['exit_reason'] == 'TIMEOUT').mean() * 100
        wr = df['win'].mean() * 100
        net = df['net_pips'].mean()
        total = df['net_pips'].sum()
        log(f"{label:<25} {len(df):>8} {to_pct:>7.1f}% {wr:>7.1f}% {net:>+9.2f} {total:>+9.0f}")

# At least one aligned
at_least_one = trades_df[(trades_df['base_vel_aligned'] == True) | (trades_df['quote_vel_aligned'] == True)]
log(f"\n{'At Least One':<25} {len(at_least_one):>8} {(at_least_one['exit_reason']=='TIMEOUT').mean()*100:>7.1f}% {at_least_one['win'].mean()*100:>7.1f}% {at_least_one['net_pips'].mean():>+9.2f} {at_least_one['net_pips'].sum():>+9.0f}")

# ============================================================================
# DETAILED BY TRADE TYPE
# ============================================================================
log("\n" + "=" * 70)
log("4. DETAILED BY TRADE TYPE")
log("=" * 70)

for trade_type in ['BUY', 'SELL']:
    log(f"\n--- {trade_type} TRADES ---")
    type_df = trades_df[trades_df['type'] == trade_type]

    if trade_type == 'BUY':
        # BUY: base rising, quote falling
        type_df['base_ok'] = type_df['base_vel_h1'] > 0
        type_df['quote_ok'] = type_df['quote_vel_h1'] < 0
    else:
        # SELL: base falling, quote rising
        type_df['base_ok'] = type_df['base_vel_h1'] < 0
        type_df['quote_ok'] = type_df['quote_vel_h1'] > 0

    both = type_df[(type_df['base_ok'] == True) & (type_df['quote_ok'] == True)]
    base_only = type_df[(type_df['base_ok'] == True) & (type_df['quote_ok'] == False)]
    quote_only = type_df[(type_df['base_ok'] == False) & (type_df['quote_ok'] == True)]
    neither = type_df[(type_df['base_ok'] == False) & (type_df['quote_ok'] == False)]

    log(f"\n{'Scenario':<20} {'Trades':>8} {'TO%':>8} {'WR':>8} {'Net':>10}")
    log("-" * 55)

    for df, label in [(both, 'Both Aligned'), (base_only, 'Base Only'), (quote_only, 'Quote Only'), (neither, 'Neither')]:
        if len(df) > 20:
            to_pct = (df['exit_reason'] == 'TIMEOUT').mean() * 100
            wr = df['win'].mean() * 100
            net = df['net_pips'].mean()
            log(f"{label:<20} {len(df):>8} {to_pct:>7.1f}% {wr:>7.1f}% {net:>+9.2f}")

# ============================================================================
# VELOCITY MAGNITUDE ANALYSIS
# ============================================================================
log("\n" + "=" * 70)
log("5. VELOCITY MAGNITUDE (STRONGER = BETTER?)")
log("=" * 70)

# Combined velocity: for BUY (base_vel - quote_vel), for SELL (quote_vel - base_vel)
# Higher = more aligned momentum
trades_df['combined_vel'] = np.where(
    trades_df['type'] == 'BUY',
    trades_df['base_vel_h1'] - trades_df['quote_vel_h1'],  # BUY: base rising, quote falling
    trades_df['quote_vel_h1'] - trades_df['base_vel_h1']   # SELL: quote rising, base falling
)

log("\n--- Combined Velocity Score (higher = better alignment) ---")
log(f"{'Range':<20} {'Trades':>8} {'TO%':>8} {'WR':>8} {'Net':>10}")
log("-" * 55)

for low, high, label in [(-0.5, 0, '< 0'), (0, 0.02, '0 to 0.02'), (0.02, 0.04, '0.02 to 0.04'),
                          (0.04, 0.06, '0.04 to 0.06'), (0.06, 0.1, '0.06 to 0.1'), (0.1, 0.5, '> 0.1')]:
    df = trades_df[(trades_df['combined_vel'] >= low) & (trades_df['combined_vel'] < high)]
    if len(df) > 50:
        to_pct = (df['exit_reason'] == 'TIMEOUT').mean() * 100
        wr = df['win'].mean() * 100
        net = df['net_pips'].mean()
        log(f"{label:<20} {len(df):>8} {to_pct:>7.1f}% {wr:>7.1f}% {net:>+9.2f}")

# ============================================================================
# MFC POSITION + VELOCITY COMBO
# ============================================================================
log("\n" + "=" * 70)
log("6. BASE MFC POSITION + VELOCITY")
log("=" * 70)

# For BUY: base should be low (< 0) AND rising
# For SELL: base should be high (> 0) AND falling

log("\n--- BUY: Base Position + Velocity ---")
buy_df = trades_df[trades_df['type'] == 'BUY'].copy()

buy_df['scenario'] = 'Other'
buy_df.loc[(buy_df['base_h1'] < -0.2) & (buy_df['base_vel_h1'] > 0), 'scenario'] = 'Low+Rising (Best)'
buy_df.loc[(buy_df['base_h1'] < -0.2) & (buy_df['base_vel_h1'] <= 0), 'scenario'] = 'Low+Falling'
buy_df.loc[(buy_df['base_h1'] >= -0.2) & (buy_df['base_vel_h1'] > 0), 'scenario'] = 'High+Rising'
buy_df.loc[(buy_df['base_h1'] >= -0.2) & (buy_df['base_vel_h1'] <= 0), 'scenario'] = 'High+Falling (Worst)'

log(f"\n{'Scenario':<25} {'Trades':>8} {'TO%':>8} {'WR':>8} {'Net':>10}")
log("-" * 60)

for scenario in ['Low+Rising (Best)', 'Low+Falling', 'High+Rising', 'High+Falling (Worst)']:
    df = buy_df[buy_df['scenario'] == scenario]
    if len(df) > 20:
        to_pct = (df['exit_reason'] == 'TIMEOUT').mean() * 100
        wr = df['win'].mean() * 100
        net = df['net_pips'].mean()
        log(f"{scenario:<25} {len(df):>8} {to_pct:>7.1f}% {wr:>7.1f}% {net:>+9.2f}")

log("\n--- SELL: Base Position + Velocity ---")
sell_df = trades_df[trades_df['type'] == 'SELL'].copy()

sell_df['scenario'] = 'Other'
sell_df.loc[(sell_df['base_h1'] > 0.2) & (sell_df['base_vel_h1'] < 0), 'scenario'] = 'High+Falling (Best)'
sell_df.loc[(sell_df['base_h1'] > 0.2) & (sell_df['base_vel_h1'] >= 0), 'scenario'] = 'High+Rising'
sell_df.loc[(sell_df['base_h1'] <= 0.2) & (sell_df['base_vel_h1'] < 0), 'scenario'] = 'Low+Falling'
sell_df.loc[(sell_df['base_h1'] <= 0.2) & (sell_df['base_vel_h1'] >= 0), 'scenario'] = 'Low+Rising (Worst)'

log(f"\n{'Scenario':<25} {'Trades':>8} {'TO%':>8} {'WR':>8} {'Net':>10}")
log("-" * 60)

for scenario in ['High+Falling (Best)', 'High+Rising', 'Low+Falling', 'Low+Rising (Worst)']:
    df = sell_df[sell_df['scenario'] == scenario]
    if len(df) > 20:
        to_pct = (df['exit_reason'] == 'TIMEOUT').mean() * 100
        wr = df['win'].mean() * 100
        net = df['net_pips'].mean()
        log(f"{scenario:<25} {len(df):>8} {to_pct:>7.1f}% {wr:>7.1f}% {net:>+9.2f}")

# ============================================================================
# PROPOSED FILTERS COMPARISON
# ============================================================================
log("\n" + "=" * 70)
log("7. FILTER COMPARISON")
log("=" * 70)

original = trades_df
log(f"\n{'Filter':<40} {'Trades':>8} {'TO%':>8} {'WR':>8} {'Net':>10} {'Total':>10}")
log("-" * 85)

# Original
log(f"{'Original (no filter)':<40} {len(original):>8} {(original['exit_reason']=='TIMEOUT').mean()*100:>7.1f}% {original['win'].mean()*100:>7.1f}% {original['net_pips'].mean():>+9.2f} {original['net_pips'].sum():>+9.0f}")

# Base velocity only
base_filter = trades_df[trades_df['base_vel_aligned'] == True]
log(f"{'Base Vel Aligned':<40} {len(base_filter):>8} {(base_filter['exit_reason']=='TIMEOUT').mean()*100:>7.1f}% {base_filter['win'].mean()*100:>7.1f}% {base_filter['net_pips'].mean():>+9.2f} {base_filter['net_pips'].sum():>+9.0f}")

# Quote velocity only
quote_filter = trades_df[trades_df['quote_vel_aligned'] == True]
log(f"{'Quote Vel Aligned':<40} {len(quote_filter):>8} {(quote_filter['exit_reason']=='TIMEOUT').mean()*100:>7.1f}% {quote_filter['win'].mean()*100:>7.1f}% {quote_filter['net_pips'].mean():>+9.2f} {quote_filter['net_pips'].sum():>+9.0f}")

# At least one
at_least_one = trades_df[(trades_df['base_vel_aligned'] == True) | (trades_df['quote_vel_aligned'] == True)]
log(f"{'At Least One Aligned':<40} {len(at_least_one):>8} {(at_least_one['exit_reason']=='TIMEOUT').mean()*100:>7.1f}% {at_least_one['win'].mean()*100:>7.1f}% {at_least_one['net_pips'].mean():>+9.2f} {at_least_one['net_pips'].sum():>+9.0f}")

# Both aligned
both_aligned = trades_df[(trades_df['base_vel_aligned'] == True) & (trades_df['quote_vel_aligned'] == True)]
log(f"{'Both Aligned':<40} {len(both_aligned):>8} {(both_aligned['exit_reason']=='TIMEOUT').mean()*100:>7.1f}% {both_aligned['win'].mean()*100:>7.1f}% {both_aligned['net_pips'].mean():>+9.2f} {both_aligned['net_pips'].sum():>+9.0f}")

# Combined velocity > 0
combined_pos = trades_df[trades_df['combined_vel'] > 0]
log(f"{'Combined Vel > 0':<40} {len(combined_pos):>8} {(combined_pos['exit_reason']=='TIMEOUT').mean()*100:>7.1f}% {combined_pos['win'].mean()*100:>7.1f}% {combined_pos['net_pips'].mean():>+9.2f} {combined_pos['net_pips'].sum():>+9.0f}")

# Combined velocity > 0.02
combined_strong = trades_df[trades_df['combined_vel'] > 0.02]
log(f"{'Combined Vel > 0.02':<40} {len(combined_strong):>8} {(combined_strong['exit_reason']=='TIMEOUT').mean()*100:>7.1f}% {combined_strong['win'].mean()*100:>7.1f}% {combined_strong['net_pips'].mean():>+9.2f} {combined_strong['net_pips'].sum():>+9.0f}")

# ============================================================================
# SUMMARY
# ============================================================================
log("\n" + "=" * 70)
log("8. SUMMARY")
log("=" * 70)

log("""
BASE VELOCITY FILTER:
- BUY: base_vel_h1 > 0 (base currency gaining momentum)
- SELL: base_vel_h1 < 0 (base currency losing momentum)

QUOTE VELOCITY FILTER:
- BUY: quote_vel_h1 < 0 (quote currency losing momentum)
- SELL: quote_vel_h1 > 0 (quote currency gaining momentum)

COMBINED VELOCITY:
- BUY: (base_vel - quote_vel) > 0
- SELL: (quote_vel - base_vel) > 0
- Essentially measures the "spread velocity" - momentum favoring our direction

KEY INSIGHT:
The pair will move in our favor when BOTH currencies are moving
in the direction we expect (base rising + quote falling for BUY,
base falling + quote rising for SELL).
""")

log(f"\nAnalysis complete!")
