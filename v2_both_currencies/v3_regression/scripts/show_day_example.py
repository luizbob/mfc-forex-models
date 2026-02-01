"""
Show a day example with high velocity MFC movement
"""
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from pathlib import Path

def log(msg=""):
    print(msg, flush=True)

DATA_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/data')
CLEANED_DIR = DATA_DIR / 'cleaned'

CURRENCIES = ['EUR', 'USD', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD', 'NZD']

# Load MFC
def load_mfc(currency, timeframe):
    fp = CLEANED_DIR / f'mfc_currency_{currency}_{timeframe}_clean.csv'
    df = pd.read_csv(fp)
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df = df.set_index('datetime').sort_index()
    return df['MFC']

# Load price
def load_price(pair, timeframe):
    fp = DATA_DIR / f'{pair}_GMT+0_US-DST_{timeframe}.csv'
    df = pd.read_csv(fp, header=None, names=['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df = df.set_index('datetime').sort_index()
    return df

log("Loading data...")

# Build MFC dataframe
mfc_df = pd.DataFrame()
for ccy in CURRENCIES:
    mfc = load_mfc(ccy, 'M30')
    if mfc_df.empty:
        mfc_df = pd.DataFrame(index=mfc.index)
    mfc_df[ccy] = mfc
    mfc_df[f'{ccy}_vel'] = mfc.diff()

mfc_df = mfc_df.dropna()

# Load EURUSD price
eurusd = load_price('EURUSD', 'M30')

log(f"MFC data: {len(mfc_df):,} bars")

# ============================================================================
# FIND A HIGH VELOCITY DAY
# ============================================================================

log("\n" + "=" * 70)
log("FINDING HIGH VELOCITY DAYS")
log("=" * 70)

# Find days with very high EUR velocity
mfc_df['EUR_vel_abs'] = mfc_df['EUR_vel'].abs()
mfc_df['date'] = mfc_df.index.date

daily_max_vel = mfc_df.groupby('date')['EUR_vel_abs'].max()
high_vel_days = daily_max_vel[daily_max_vel > 0.15].sort_values(ascending=False)

log(f"\nTop 10 days with highest EUR velocity:")
for i, (date, vel) in enumerate(high_vel_days.head(10).items()):
    log(f"  {i+1}. {date}: max velocity = {vel:.3f}")

# Pick one day to show
example_date = high_vel_days.index[2]  # Pick 3rd highest
log(f"\n\nShowing example: {example_date}")

# ============================================================================
# SHOW THE DAY BAR BY BAR
# ============================================================================

log("\n" + "=" * 70)
log(f"DAY DETAIL: {example_date}")
log("=" * 70)

day_mfc = mfc_df[mfc_df['date'] == example_date].copy()
day_price = eurusd[eurusd.index.date == example_date].copy()

# Align
day_data = day_mfc.join(day_price[['Open', 'High', 'Low', 'Close']], how='inner')

log(f"\n{'Time':<8} {'EUR MFC':>8} {'EUR Vel':>8} {'USD MFC':>8} {'USD Vel':>8} | {'EURUSD':>10} {'Change':>8}")
log("-" * 75)

prev_close = None
for idx, row in day_data.iterrows():
    time_str = idx.strftime('%H:%M')
    eur_mfc = row['EUR']
    eur_vel = row['EUR_vel']
    usd_mfc = row['USD']
    usd_vel = row['USD_vel']
    close = row['Close']

    if prev_close:
        change = (close - prev_close) * 10000  # pips
        change_str = f"{change:+.1f}"
    else:
        change_str = "-"

    # Mark significant events
    marker = ""
    if abs(eur_vel) > 0.10:
        marker = " <-- HIGH VEL"
    elif abs(eur_vel) > 0.05:
        marker = " <-- signal"

    log(f"{time_str:<8} {eur_mfc:>+8.3f} {eur_vel:>+8.3f} {usd_mfc:>+8.3f} {usd_vel:>+8.3f} | {close:>10.5f} {change_str:>8}{marker}")

    prev_close = close

# ============================================================================
# SHOW ALL CURRENCIES FOR CONTEXT
# ============================================================================

log("\n" + "=" * 70)
log(f"ALL CURRENCIES AT PEAK VELOCITY MOMENT")
log("=" * 70)

# Find the peak velocity moment
peak_idx = day_mfc['EUR_vel_abs'].idxmax()
peak_row = mfc_df.loc[peak_idx]

log(f"\nTime: {peak_idx}")
log(f"\n{'Currency':<6} {'MFC':>10} {'Velocity':>10} {'Position':>15}")
log("-" * 45)

for ccy in CURRENCIES:
    mfc_val = peak_row[ccy]
    vel_val = peak_row[f'{ccy}_vel']

    if mfc_val > 0.2:
        pos = "ABOVE box"
    elif mfc_val < -0.2:
        pos = "BELOW box"
    else:
        pos = "IN box"

    marker = " <--" if ccy == 'EUR' else ""
    log(f"{ccy:<6} {mfc_val:>+10.3f} {vel_val:>+10.3f} {pos:>15}{marker}")

# Show what EURUSD did in the next few bars
log("\n" + "=" * 70)
log("WHAT HAPPENED NEXT (after peak velocity)")
log("=" * 70)

# Get next 6 bars after peak
future_start = peak_idx
future_bars = mfc_df.loc[future_start:].head(7)
future_price = eurusd.loc[future_start:].head(7)

log(f"\nEUR was {'weakening' if peak_row['EUR_vel'] < 0 else 'strengthening'} with velocity {peak_row['EUR_vel']:+.3f}")
log(f"USD was {'weakening' if peak_row['USD_vel'] < 0 else 'strengthening'} with velocity {peak_row['USD_vel']:+.3f}")

if peak_row['EUR_vel'] < 0 and peak_row['USD_vel'] > 0:
    expected = "DOWN (EUR weak, USD strong)"
elif peak_row['EUR_vel'] > 0 and peak_row['USD_vel'] < 0:
    expected = "UP (EUR strong, USD weak)"
elif peak_row['EUR_vel'] > peak_row['USD_vel']:
    expected = "UP (EUR stronger)"
else:
    expected = "DOWN (USD stronger)"

log(f"\nExpected EURUSD direction: {expected}")

log(f"\n{'Bars After':>10} {'EUR MFC':>10} {'EUR Vel':>10} {'EURUSD':>12} {'Cumul Pips':>12}")
log("-" * 60)

entry_price = future_price.iloc[0]['Close']
for i, (idx, row) in enumerate(future_bars.iterrows()):
    if idx in future_price.index:
        price = future_price.loc[idx, 'Close']
        cumul_pips = (price - entry_price) * 10000
    else:
        price = np.nan
        cumul_pips = np.nan

    log(f"{i:>10} {row['EUR']:>+10.3f} {row['EUR_vel']:>+10.3f} {price:>12.5f} {cumul_pips:>+12.1f}")

log("\n" + "=" * 70)
log("ANOTHER EXAMPLE - DIFFERENT DAY")
log("=" * 70)

# Pick another day
example_date2 = high_vel_days.index[5]
log(f"\nDay: {example_date2}")

day_mfc2 = mfc_df[mfc_df['date'] == example_date2].copy()
peak_idx2 = day_mfc2['EUR_vel_abs'].idxmax()
peak_row2 = mfc_df.loc[peak_idx2]

log(f"Peak velocity time: {peak_idx2}")
log(f"\n{'Currency':<6} {'MFC':>10} {'Velocity':>10}")
log("-" * 30)
for ccy in CURRENCIES:
    log(f"{ccy:<6} {peak_row2[ccy]:>+10.3f} {peak_row2[f'{ccy}_vel']:>+10.3f}")

# Future bars
future_bars2 = mfc_df.loc[peak_idx2:].head(7)
future_price2 = eurusd.loc[peak_idx2:].head(7)

log(f"\nEUR velocity: {peak_row2['EUR_vel']:+.3f} ({'weakening' if peak_row2['EUR_vel'] < 0 else 'strengthening'})")

log(f"\n{'Bars After':>10} {'EUR MFC':>10} {'EURUSD':>12} {'Cumul Pips':>12}")
log("-" * 50)

if len(future_price2) > 0:
    entry_price2 = future_price2.iloc[0]['Close']
    for i, (idx, row) in enumerate(future_bars2.iterrows()):
        if idx in future_price2.index:
            price = future_price2.loc[idx, 'Close']
            cumul_pips = (price - entry_price2) * 10000
            log(f"{i:>10} {row['EUR']:>+10.3f} {price:>12.5f} {cumul_pips:>+12.1f}")

log()
