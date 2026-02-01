"""
USD MFC vs Gold Price Correlation Analysis
==========================================
Test if USD MFC can predict gold price movements.

Theory: USD weak → Gold rises, USD strong → Gold falls
"""
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

def log(msg=""):
    print(msg, flush=True)

log("=" * 70)
log("USD MFC vs GOLD PRICE CORRELATION")
log("=" * 70)
log(f"Started: {datetime.now()}")

DATA_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/data')
CLEANED_DIR = DATA_DIR / 'cleaned'

# Load USD MFC
log("\nLoading USD MFC data...")
usd_m5 = pd.read_csv(CLEANED_DIR / 'mfc_currency_USD_M5_clean.csv')
usd_m5['datetime'] = pd.to_datetime(usd_m5['Date'] + ' ' + usd_m5['Time'])
usd_m5 = usd_m5.set_index('datetime').sort_index()
usd_mfc = usd_m5['MFC']

usd_h4 = pd.read_csv(CLEANED_DIR / 'mfc_currency_USD_H4_clean.csv')
usd_h4['datetime'] = pd.to_datetime(usd_h4['Date'] + ' ' + usd_h4['Time'])
usd_h4 = usd_h4.set_index('datetime').sort_index()
usd_h4_mfc = usd_h4['MFC']

log(f"USD MFC: {len(usd_mfc):,} M5 bars")

# Load Gold data
log("Loading Gold data...")
gold = pd.read_csv(DATA_DIR / 'XAUUSDm_M5_202110271720_202501172155.csv', sep='\t')
gold.columns = ['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'TickVol', 'Vol', 'Spread']
gold['datetime'] = pd.to_datetime(gold['Date'] + ' ' + gold['Time'], format='%Y.%m.%d %H:%M:%S')
gold = gold.set_index('datetime').sort_index()
gold_price = gold['Close']

log(f"Gold: {len(gold_price):,} M5 bars")
log(f"Gold range: ${gold_price.min():.2f} to ${gold_price.max():.2f}")

# Align data
log("\nAligning data...")
df = pd.DataFrame(index=gold_price.index)
df['gold'] = gold_price
df['usd_mfc'] = usd_mfc.reindex(df.index, method='ffill')
df['usd_h4'] = usd_h4_mfc.shift(1).reindex(df.index, method='ffill')  # Shifted H4

# Calculate changes
df['gold_change'] = df['gold'].diff()
df['gold_change_pct'] = df['gold'].pct_change() * 100
df['gold_change_12'] = df['gold'].diff(12)  # 1 hour change
df['gold_change_48'] = df['gold'].diff(48)  # 4 hour change

df['usd_mfc_prev'] = df['usd_mfc'].shift(1)
df['usd_vel'] = df['usd_mfc'].diff()
df['usd_vel_12'] = df['usd_mfc'].diff(12)

df = df.dropna()
log(f"Aligned data: {len(df):,} bars ({df.index[0].date()} to {df.index[-1].date()})")

# ============================================================================
# BASIC CORRELATION
# ============================================================================

log("\n" + "=" * 70)
log("BASIC CORRELATION ANALYSIS")
log("=" * 70)

# Correlation between USD MFC and Gold price change
corr_1bar = df['usd_vel'].corr(df['gold_change'])
corr_12bar = df['usd_vel_12'].corr(df['gold_change_12'])

log(f"\nCorrelation (USD MFC change vs Gold change):")
log(f"  1-bar (5min):  {corr_1bar:.4f}")
log(f"  12-bar (1h):   {corr_12bar:.4f}")

# If negative correlation, USD up → Gold down (expected)
if corr_12bar < 0:
    log("  → Negative = USD strength hurts gold (expected)")
else:
    log("  → Positive = unexpected relationship")

# ============================================================================
# USD MFC LEVEL vs GOLD DIRECTION
# ============================================================================

log("\n" + "=" * 70)
log("USD MFC LEVEL vs FUTURE GOLD MOVEMENT")
log("=" * 70)

# When USD MFC is at extreme, what happens to gold in next 1 hour?
log("\n--- Next 1 Hour Gold Change by USD MFC Level ---")

for (low, high, label) in [(-1, -0.5, "USD Very Weak"), (-0.5, -0.2, "USD Weak"),
                            (-0.2, 0.2, "USD Neutral"), (0.2, 0.5, "USD Strong"),
                            (0.5, 1, "USD Very Strong")]:
    mask = (df['usd_mfc_prev'] >= low) & (df['usd_mfc_prev'] < high)
    subset = df[mask]
    if len(subset) > 100:
        avg_gold_change = subset['gold_change_12'].mean()
        pct_up = (subset['gold_change_12'] > 0).mean() * 100
        log(f"{label:20s}: {len(subset):>6,} bars, Gold avg {avg_gold_change:>+6.2f}, up {pct_up:.1f}%")

# ============================================================================
# USD MFC EXTREME → GOLD REACTION
# ============================================================================

log("\n" + "=" * 70)
log("TRADING SIGNAL: USD MFC EXTREME → GOLD")
log("=" * 70)

# Find USD MFC extreme events and track gold
events = []

usd_arr = df['usd_mfc_prev'].values
gold_arr = df['gold'].values
n = len(df)

i = 50
while i < n - 48:
    usd_val = usd_arr[i]

    # USD at negative extreme (weak) → expect gold UP
    if usd_val <= -0.5:
        entry_gold = gold_arr[i]
        exit_gold_1h = gold_arr[i + 12]
        exit_gold_4h = gold_arr[i + 48]

        events.append({
            'datetime': df.index[i],
            'usd_mfc': usd_val,
            'signal': 'LONG_GOLD',
            'gold_change_1h': exit_gold_1h - entry_gold,
            'gold_change_4h': exit_gold_4h - entry_gold,
        })
        i += 12

    # USD at positive extreme (strong) → expect gold DOWN
    elif usd_val >= 0.5:
        entry_gold = gold_arr[i]
        exit_gold_1h = gold_arr[i + 12]
        exit_gold_4h = gold_arr[i + 48]

        events.append({
            'datetime': df.index[i],
            'usd_mfc': usd_val,
            'signal': 'SHORT_GOLD',
            'gold_change_1h': entry_gold - exit_gold_1h,
            'gold_change_4h': entry_gold - exit_gold_4h,
        })
        i += 12
    else:
        i += 1

events_df = pd.DataFrame(events)
log(f"\nTotal extreme USD events: {len(events_df):,}")

if len(events_df) > 0:
    long_gold = events_df[events_df['signal'] == 'LONG_GOLD']
    short_gold = events_df[events_df['signal'] == 'SHORT_GOLD']

    log(f"\n--- LONG GOLD (when USD MFC <= -0.5) ---")
    log(f"Events: {len(long_gold):,}")
    log(f"1h: Win {(long_gold['gold_change_1h'] > 0).mean()*100:.1f}%, Avg ${long_gold['gold_change_1h'].mean():.2f}")
    log(f"4h: Win {(long_gold['gold_change_4h'] > 0).mean()*100:.1f}%, Avg ${long_gold['gold_change_4h'].mean():.2f}")

    log(f"\n--- SHORT GOLD (when USD MFC >= 0.5) ---")
    log(f"Events: {len(short_gold):,}")
    log(f"1h: Win {(short_gold['gold_change_1h'] > 0).mean()*100:.1f}%, Avg ${short_gold['gold_change_1h'].mean():.2f}")
    log(f"4h: Win {(short_gold['gold_change_4h'] > 0).mean()*100:.1f}%, Avg ${short_gold['gold_change_4h'].mean():.2f}")

    # Combined
    log(f"\n--- COMBINED (trade gold based on USD MFC extreme) ---")
    all_1h = events_df['gold_change_1h']
    all_4h = events_df['gold_change_4h']
    log(f"1h: Win {(all_1h > 0).mean()*100:.1f}%, Avg ${all_1h.mean():.2f}, Total ${all_1h.sum():.0f}")
    log(f"4h: Win {(all_4h > 0).mean()*100:.1f}%, Avg ${all_4h.mean():.2f}, Total ${all_4h.sum():.0f}")

# ============================================================================
# ADD H4 USD MFC FILTER
# ============================================================================

log("\n" + "=" * 70)
log("WITH H4 USD MFC CONFIRMATION")
log("=" * 70)

# Find events with H4 confirmation
events2 = []

usd_h4_arr = df['usd_h4'].values

i = 50
while i < n - 48:
    usd_val = usd_arr[i]
    usd_h4_val = usd_h4_arr[i]

    # USD M5 weak + USD H4 weak → strong signal for gold long
    if usd_val <= -0.5 and usd_h4_val <= -0.3:
        entry_gold = gold_arr[i]
        exit_gold_1h = gold_arr[i + 12]
        exit_gold_4h = gold_arr[i + 48]

        events2.append({
            'signal': 'LONG_GOLD',
            'gold_change_1h': exit_gold_1h - entry_gold,
            'gold_change_4h': exit_gold_4h - entry_gold,
        })
        i += 12

    # USD M5 strong + USD H4 strong → strong signal for gold short
    elif usd_val >= 0.5 and usd_h4_val >= 0.3:
        entry_gold = gold_arr[i]
        exit_gold_1h = gold_arr[i + 12]
        exit_gold_4h = gold_arr[i + 48]

        events2.append({
            'signal': 'SHORT_GOLD',
            'gold_change_1h': entry_gold - exit_gold_1h,
            'gold_change_4h': entry_gold - exit_gold_4h,
        })
        i += 12
    else:
        i += 1

events2_df = pd.DataFrame(events2)
log(f"\nWith H4 confirmation: {len(events2_df):,} events")

if len(events2_df) > 0:
    all_1h = events2_df['gold_change_1h']
    all_4h = events2_df['gold_change_4h']
    log(f"1h: Win {(all_1h > 0).mean()*100:.1f}%, Avg ${all_1h.mean():.2f}, Total ${all_1h.sum():.0f}")
    log(f"4h: Win {(all_4h > 0).mean()*100:.1f}%, Avg ${all_4h.mean():.2f}, Total ${all_4h.sum():.0f}")

# ============================================================================
# MEAN REVERSION ON GOLD USING USD MFC
# ============================================================================

log("\n" + "=" * 70)
log("MEAN REVERSION: Wait for USD MFC to return to 0")
log("=" * 70)

# Entry: USD at extreme
# Exit: USD returns to 0
events3 = []

i = 50
while i < n - 200:
    usd_val = usd_arr[i]

    if usd_val <= -0.5:
        entry_gold = gold_arr[i]

        # Find when USD returns to 0
        for j in range(i+1, min(i+200, n)):
            if usd_arr[j] >= 0:
                exit_gold = gold_arr[j]
                events3.append({
                    'signal': 'LONG_GOLD',
                    'gold_pnl': exit_gold - entry_gold,
                    'bars': j - i,
                })
                break
        i = j + 1

    elif usd_val >= 0.5:
        entry_gold = gold_arr[i]

        for j in range(i+1, min(i+200, n)):
            if usd_arr[j] <= 0:
                exit_gold = gold_arr[j]
                events3.append({
                    'signal': 'SHORT_GOLD',
                    'gold_pnl': entry_gold - exit_gold,
                    'bars': j - i,
                })
                break
        i = j + 1
    else:
        i += 1

events3_df = pd.DataFrame(events3)
log(f"\nMean reversion trades: {len(events3_df):,}")

if len(events3_df) > 0:
    pnl = events3_df['gold_pnl']
    log(f"Win Rate: {(pnl > 0).mean()*100:.1f}%")
    log(f"Avg PnL: ${pnl.mean():.2f}")
    log(f"Total PnL: ${pnl.sum():.0f}")
    log(f"Avg hold: {events3_df['bars'].mean():.0f} bars ({events3_df['bars'].mean()*5/60:.1f}h)")

    # By signal
    for sig in ['LONG_GOLD', 'SHORT_GOLD']:
        subset = events3_df[events3_df['signal'] == sig]
        if len(subset) > 0:
            log(f"\n  {sig}: {len(subset)} trades, WR {(subset['gold_pnl'] > 0).mean()*100:.1f}%, Avg ${subset['gold_pnl'].mean():.2f}")

log(f"\nCompleted: {datetime.now()}")
