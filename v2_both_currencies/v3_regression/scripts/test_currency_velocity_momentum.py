"""
Test: Single Currency Velocity + Momentum Prediction
=====================================================
For a currency (e.g., USD), if H1+H4 velocity AND momentum are all up,
does it predict that currency's price movement over next 12 hours?
"""
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from pathlib import Path

def log(msg=""):
    print(msg, flush=True)

log("=" * 70)
log("SINGLE CURRENCY: VELOCITY + MOMENTUM PREDICTION")
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
    df = df[(df.index >= '2023-01-01') & (df.index <= '2025-12-31')]
    return df['MFC']

def load_price(pair):
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

# Load MFC data for all currencies
log("\nLoading MFC data...")
mfc_data = {}
for ccy in CURRENCIES:
    mfc_data[ccy] = {
        'h1': load_mfc(ccy, 'H1'),
        'h4': load_mfc(ccy, 'H4'),
    }
    log(f"  {ccy}: H1={len(mfc_data[ccy]['h1']) if mfc_data[ccy]['h1'] is not None else 0}, "
        f"H4={len(mfc_data[ccy]['h4']) if mfc_data[ccy]['h4'] is not None else 0}")

# Calculate velocity (1-bar) and momentum (2-bar) for each
log("\nCalculating velocity and momentum...")
indicators = {}
for ccy in CURRENCIES:
    h1 = mfc_data[ccy]['h1']
    h4 = mfc_data[ccy]['h4']

    if h1 is None or h4 is None:
        continue

    indicators[ccy] = {
        'h1_vel': h1.diff().shift(1),      # 1-bar velocity, shifted
        'h1_mom': h1.diff(2).shift(1),     # 2-bar momentum, shifted
        'h4_vel': h4.diff().shift(1),      # 1-bar velocity, shifted
        'h4_mom': h4.diff(2).shift(1),     # 2-bar momentum, shifted
    }

# Pairs structure
PAIRS = {
    'EURUSD': ('EUR', 'USD'), 'GBPUSD': ('GBP', 'USD'), 'USDJPY': ('USD', 'JPY'),
    'AUDUSD': ('AUD', 'USD'), 'USDCAD': ('USD', 'CAD'), 'USDCHF': ('USD', 'CHF'),
    'NZDUSD': ('NZD', 'USD'), 'EURGBP': ('EUR', 'GBP'), 'EURJPY': ('EUR', 'JPY'),
    'GBPJPY': ('GBP', 'JPY'), 'AUDJPY': ('AUD', 'JPY'), 'EURAUD': ('EUR', 'AUD'),
    'GBPAUD': ('GBP', 'AUD'), 'EURCHF': ('EUR', 'CHF'), 'GBPCHF': ('GBP', 'CHF'),
    'AUDCAD': ('AUD', 'CAD'), 'EURCAD': ('EUR', 'CAD'), 'GBPCAD': ('GBP', 'CAD'),
    'CADJPY': ('CAD', 'JPY'), 'CHFJPY': ('CHF', 'JPY'), 'AUDCHF': ('AUD', 'CHF'),
    'AUDNZD': ('AUD', 'NZD'), 'EURNZD': ('EUR', 'NZD'), 'GBPNZD': ('GBP', 'NZD'),
    'NZDJPY': ('NZD', 'JPY'), 'NZDCAD': ('NZD', 'CAD'), 'NZDCHF': ('NZD', 'CHF'),
    'CADCHF': ('CAD', 'CHF'),
}

# Load price data
log("\nLoading price data...")
prices = {}
for pair in PAIRS.keys():
    price_df = load_price(pair)
    if price_df is not None:
        prices[pair] = price_df

log(f"Loaded {len(prices)} pairs")

# Get trading days
sample = list(indicators.values())[0]['h4_vel']
trading_days = sample.index.normalize().unique()
trading_days = trading_days[(trading_days >= '2023-01-01') & (trading_days <= '2025-12-31')]
log(f"Trading days: {len(trading_days)}")

# For each currency, check if velocity+momentum aligned predicts 12h movement
results = []

for day in trading_days:
    start_time = day
    end_time = day + pd.Timedelta(hours=12)

    for ccy in CURRENCIES:
        if ccy not in indicators:
            continue

        ind = indicators[ccy]

        try:
            # Get H4 indicators (at day start)
            h4_idx = ind['h4_vel'].index[ind['h4_vel'].index <= day]
            if len(h4_idx) == 0:
                continue
            h4_time = h4_idx[-1]

            h4_vel = ind['h4_vel'].loc[h4_time]
            h4_mom = ind['h4_mom'].loc[h4_time]

            # Get H1 indicators (at day start)
            h1_idx = ind['h1_vel'].index[ind['h1_vel'].index <= day]
            if len(h1_idx) == 0:
                continue
            h1_time = h1_idx[-1]

            h1_vel = ind['h1_vel'].loc[h1_time]
            h1_mom = ind['h1_mom'].loc[h1_time]

            if pd.isna(h4_vel) or pd.isna(h4_mom) or pd.isna(h1_vel) or pd.isna(h1_mom):
                continue

        except:
            continue

        # Determine alignment
        # Velocity direction (threshold 0.01)
        h1_vel_dir = 1 if h1_vel > 0.01 else (-1 if h1_vel < -0.01 else 0)
        h4_vel_dir = 1 if h4_vel > 0.01 else (-1 if h4_vel < -0.01 else 0)

        # Momentum direction (threshold 0.02 since it's 2-bar)
        h1_mom_dir = 1 if h1_mom > 0.02 else (-1 if h1_mom < -0.02 else 0)
        h4_mom_dir = 1 if h4_mom > 0.02 else (-1 if h4_mom < -0.02 else 0)

        # Check different alignment scenarios
        vel_aligned = (h1_vel_dir == h4_vel_dir) and h1_vel_dir != 0
        mom_aligned = (h1_mom_dir == h4_mom_dir) and h1_mom_dir != 0
        all_aligned = vel_aligned and mom_aligned and (h1_vel_dir == h1_mom_dir)

        signal_dir = h1_vel_dir if all_aligned else 0

        # Calculate price movement for all pairs containing this currency
        for pair, (base, quote) in PAIRS.items():
            if pair not in prices:
                continue

            if base != ccy and quote != ccy:
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

            # If currency is base, positive pct = currency went up
            # If currency is quote, negative pct = currency went up
            if base == ccy:
                ccy_movement = pct_change
            else:
                ccy_movement = -pct_change

            # Calculate trade result if we traded based on signal
            if all_aligned:
                if signal_dir == 1:  # Currency going up
                    trade_pct = ccy_movement
                else:  # Currency going down
                    trade_pct = -ccy_movement
            else:
                trade_pct = None

            results.append({
                'date': day,
                'currency': ccy,
                'pair': pair,
                'h1_vel': h1_vel,
                'h4_vel': h4_vel,
                'h1_mom': h1_mom,
                'h4_mom': h4_mom,
                'h1_vel_dir': h1_vel_dir,
                'h4_vel_dir': h4_vel_dir,
                'h1_mom_dir': h1_mom_dir,
                'h4_mom_dir': h4_mom_dir,
                'vel_aligned': vel_aligned,
                'mom_aligned': mom_aligned,
                'all_aligned': all_aligned,
                'signal_dir': signal_dir,
                'ccy_movement': ccy_movement,
                'trade_pct': trade_pct,
            })

df = pd.DataFrame(results)
log(f"\nTotal observations: {len(df):,}")

# Analysis
log("\n" + "=" * 70)
log("COMPARISON: VELOCITY ONLY vs MOMENTUM ONLY vs BOTH")
log("=" * 70)

# Velocity aligned only
vel_only = df[df['vel_aligned'] == True].copy()
vel_only['trade_result'] = vel_only.apply(
    lambda r: r['ccy_movement'] if r['h1_vel_dir'] == 1 else -r['ccy_movement'], axis=1)

log(f"\nVELOCITY ALIGNED (H1+H4 velocity same direction):")
log(f"  Observations: {len(vel_only):,}")
log(f"  Win rate: {(vel_only['trade_result'] > 0).mean()*100:.1f}%")
log(f"  Avg % return: {vel_only['trade_result'].mean():+.4f}%")
log(f"  Total %: {vel_only['trade_result'].sum():+.2f}%")

# Momentum aligned only
mom_only = df[df['mom_aligned'] == True].copy()
mom_only['trade_result'] = mom_only.apply(
    lambda r: r['ccy_movement'] if r['h1_mom_dir'] == 1 else -r['ccy_movement'], axis=1)

log(f"\nMOMENTUM ALIGNED (H1+H4 momentum same direction):")
log(f"  Observations: {len(mom_only):,}")
log(f"  Win rate: {(mom_only['trade_result'] > 0).mean()*100:.1f}%")
log(f"  Avg % return: {mom_only['trade_result'].mean():+.4f}%")
log(f"  Total %: {mom_only['trade_result'].sum():+.2f}%")

# All aligned (velocity + momentum)
all_aligned = df[df['all_aligned'] == True].copy()
all_aligned['trade_result'] = all_aligned['trade_pct']

log(f"\nALL ALIGNED (H1+H4 velocity + H1+H4 momentum all same direction):")
log(f"  Observations: {len(all_aligned):,}")
log(f"  Win rate: {(all_aligned['trade_result'] > 0).mean()*100:.1f}%")
log(f"  Avg % return: {all_aligned['trade_result'].mean():+.4f}%")
log(f"  Total %: {all_aligned['trade_result'].sum():+.2f}%")

# By currency
log("\n" + "=" * 70)
log("BY CURRENCY (all aligned)")
log("=" * 70)

log(f"\n{'Currency':<10} {'Trades':>8} {'Win %':>10} {'Avg %':>12} {'Total %':>12}")
log("-" * 55)

for ccy in CURRENCIES:
    subset = all_aligned[all_aligned['currency'] == ccy]
    if len(subset) > 20:
        wr = (subset['trade_result'] > 0).mean() * 100
        avg = subset['trade_result'].mean()
        total = subset['trade_result'].sum()
        log(f"{ccy:<10} {len(subset):>8,} {wr:>9.1f}% {avg:>+11.4f}% {total:>+11.2f}%")

# By velocity magnitude
log("\n" + "=" * 70)
log("BY H1 VELOCITY MAGNITUDE (all aligned)")
log("=" * 70)

log(f"\n{'H1 Velocity':<15} {'Trades':>8} {'Win %':>10} {'Avg %':>12} {'Total %':>12}")
log("-" * 60)

for low, high, label in [(0.01, 0.03, '0.01-0.03'), (0.03, 0.05, '0.03-0.05'),
                          (0.05, 0.08, '0.05-0.08'), (0.08, 0.15, '0.08+')]:
    subset = all_aligned[(abs(all_aligned['h1_vel']) >= low) & (abs(all_aligned['h1_vel']) < high)]
    if len(subset) > 50:
        wr = (subset['trade_result'] > 0).mean() * 100
        avg = subset['trade_result'].mean()
        total = subset['trade_result'].sum()
        log(f"{label:<15} {len(subset):>8,} {wr:>9.1f}% {avg:>+11.4f}% {total:>+11.2f}%")

# Compare velocity vs momentum importance
log("\n" + "=" * 70)
log("WHICH MATTERS MORE: VELOCITY OR MOMENTUM?")
log("=" * 70)

# Case 1: Velocity aligned, momentum NOT aligned
vel_yes_mom_no = df[(df['vel_aligned'] == True) & (df['mom_aligned'] == False)].copy()
vel_yes_mom_no['trade_result'] = vel_yes_mom_no.apply(
    lambda r: r['ccy_movement'] if r['h1_vel_dir'] == 1 else -r['ccy_movement'], axis=1)

log(f"\nVelocity YES, Momentum NO:")
log(f"  Observations: {len(vel_yes_mom_no):,}")
log(f"  Win rate: {(vel_yes_mom_no['trade_result'] > 0).mean()*100:.1f}%")
log(f"  Avg %: {vel_yes_mom_no['trade_result'].mean():+.4f}%")

# Case 2: Momentum aligned, velocity NOT aligned
mom_yes_vel_no = df[(df['mom_aligned'] == True) & (df['vel_aligned'] == False)].copy()
mom_yes_vel_no['trade_result'] = mom_yes_vel_no.apply(
    lambda r: r['ccy_movement'] if r['h1_mom_dir'] == 1 else -r['ccy_movement'], axis=1)

log(f"\nMomentum YES, Velocity NO:")
log(f"  Observations: {len(mom_yes_vel_no):,}")
log(f"  Win rate: {(mom_yes_vel_no['trade_result'] > 0).mean()*100:.1f}%")
log(f"  Avg %: {mom_yes_vel_no['trade_result'].mean():+.4f}%")

# Case 3: Both aligned
log(f"\nBOTH Velocity + Momentum aligned:")
log(f"  Observations: {len(all_aligned):,}")
log(f"  Win rate: {(all_aligned['trade_result'] > 0).mean()*100:.1f}%")
log(f"  Avg %: {all_aligned['trade_result'].mean():+.4f}%")

# Best filter: all aligned + high velocity
log("\n" + "=" * 70)
log("BEST FILTER: All aligned + H1 velocity > 0.05")
log("=" * 70)

best = all_aligned[abs(all_aligned['h1_vel']) > 0.05]
log(f"\nObservations: {len(best):,}")
log(f"Win rate: {(best['trade_result'] > 0).mean()*100:.1f}%")
log(f"Avg % return: {best['trade_result'].mean():+.4f}%")
log(f"Total %: {best['trade_result'].sum():+.2f}%")

log(f"\n{'Currency':<10} {'Trades':>8} {'Win %':>10} {'Avg %':>12}")
log("-" * 45)
for ccy in CURRENCIES:
    subset = best[best['currency'] == ccy]
    if len(subset) > 10:
        wr = (subset['trade_result'] > 0).mean() * 100
        avg = subset['trade_result'].mean()
        log(f"{ccy:<10} {len(subset):>8,} {wr:>9.1f}% {avg:>+11.4f}%")

log("\nDONE")
