"""
Backtest H4 Velocity Filter
============================
Test if H4 MFC velocity improves mean reversion and momentum strategies.
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
log("BACKTEST: H4 VELOCITY FILTER")
log("=" * 70)
log(f"Started: {datetime.now()}")

DATA_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/data')
CLEANED_DIR = DATA_DIR / 'cleaned'

CURRENCIES = ['EUR', 'USD', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD', 'NZD']

ALL_PAIRS = [
    ('EURUSD', 'EUR', 'USD'), ('GBPUSD', 'GBP', 'USD'), ('AUDUSD', 'AUD', 'USD'),
    ('NZDUSD', 'NZD', 'USD'), ('USDJPY', 'USD', 'JPY'), ('USDCHF', 'USD', 'CHF'),
    ('USDCAD', 'USD', 'CAD'), ('EURGBP', 'EUR', 'GBP'), ('EURJPY', 'EUR', 'JPY'),
    ('EURCHF', 'EUR', 'CHF'), ('EURCAD', 'EUR', 'CAD'), ('EURAUD', 'EUR', 'AUD'),
    ('EURNZD', 'EUR', 'NZD'), ('GBPJPY', 'GBP', 'JPY'), ('GBPCHF', 'GBP', 'CHF'),
    ('GBPCAD', 'GBP', 'CAD'), ('GBPAUD', 'GBP', 'AUD'), ('GBPNZD', 'GBP', 'NZD'),
    ('AUDJPY', 'AUD', 'JPY'), ('AUDCHF', 'AUD', 'CHF'), ('AUDCAD', 'AUD', 'CAD'),
    ('AUDNZD', 'AUD', 'NZD'), ('NZDJPY', 'NZD', 'JPY'), ('NZDCHF', 'NZD', 'CHF'),
    ('NZDCAD', 'NZD', 'CAD'), ('CADJPY', 'CAD', 'JPY'), ('CADCHF', 'CAD', 'CHF'),
    ('CHFJPY', 'CHF', 'JPY'),
]

SPREADS = {
    'EURUSD': 1.5, 'GBPUSD': 2.0, 'USDJPY': 1.5, 'USDCHF': 2.0, 'USDCAD': 2.0,
    'AUDUSD': 1.5, 'NZDUSD': 2.5, 'EURGBP': 2.0, 'EURJPY': 2.0, 'GBPJPY': 3.0,
    'EURCHF': 2.5, 'AUDJPY': 2.5, 'EURAUD': 3.0, 'EURCAD': 3.0, 'EURNZD': 4.0,
    'GBPAUD': 3.5, 'GBPCAD': 3.5, 'GBPCHF': 3.0, 'GBPNZD': 4.5, 'AUDCAD': 2.5,
    'AUDCHF': 2.5, 'AUDNZD': 3.0, 'CADJPY': 2.5, 'CHFJPY': 2.5, 'NZDCAD': 3.5,
    'NZDCHF': 3.5, 'NZDJPY': 3.0, 'CADCHF': 3.0,
}

def get_pip_value(pair):
    return 0.01 if 'JPY' in pair else 0.0001

# Load MFC data
log("\nLoading MFC data...")
mfc_m5 = {}
mfc_h4 = {}

for ccy in CURRENCIES:
    df = pd.read_csv(CLEANED_DIR / f'mfc_currency_{ccy}_M5_clean.csv')
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df = df.set_index('datetime').sort_index()
    mfc_m5[ccy] = df['MFC']

    df = pd.read_csv(CLEANED_DIR / f'mfc_currency_{ccy}_H4_clean.csv')
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df = df.set_index('datetime').sort_index()
    mfc_h4[ccy] = df['MFC']

# Load price data
log("Loading price data...")
price_data = {}
for pair, base, quote in ALL_PAIRS:
    try:
        chunks = []
        for chunk in pd.read_csv(DATA_DIR / f'{pair}_GMT+0_US-DST_M1.csv', chunksize=500000):
            chunk['datetime'] = pd.to_datetime(chunk['Date'] + ' ' + chunk['Time'])
            chunk = chunk.set_index('datetime')
            chunk = chunk[chunk.index >= '2020-01-01']
            if len(chunk) > 0:
                m5 = chunk['Close'].resample('5min').last().dropna()
                chunks.append(m5)
        if chunks:
            price = pd.concat(chunks)
            price = price[~price.index.duplicated(keep='first')]
            price_data[pair] = price
    except:
        pass

log(f"Loaded {len(price_data)} pairs")

# ============================================================================
# BACKTEST FUNCTION
# ============================================================================

def run_backtest(strategy='mean_reversion', h4_vel_filter=None):
    """
    strategy: 'mean_reversion' or 'momentum'
    h4_vel_filter: None, 'with_trend', 'against_trend'
    """
    all_trades = []

    for pair, base, quote in ALL_PAIRS:
        if pair not in price_data:
            continue

        price = price_data[pair]
        pip_val = get_pip_value(pair)

        # Align MFC to price index
        base_m5 = mfc_m5[base].reindex(price.index, method='ffill')
        quote_m5 = mfc_m5[quote].reindex(price.index, method='ffill')

        # H4 MFC - shift by 1 H4 bar before ffill
        base_h4 = mfc_h4[base].shift(1).reindex(price.index, method='ffill')
        quote_h4 = mfc_h4[quote].shift(1).reindex(price.index, method='ffill')

        # Build dataframe
        df = pd.DataFrame(index=price.index)
        df['price'] = price
        df['base_m5'] = base_m5.shift(1)  # Previous bar
        df['quote_m5'] = quote_m5.shift(1)
        df['base_h4'] = base_h4
        df['quote_h4'] = quote_h4

        # H4 velocity (change in H4 MFC)
        df['base_h4_vel'] = df['base_h4'].diff()
        df['quote_h4_vel'] = df['quote_h4'].diff()

        df = df.dropna()

        n = len(df)
        max_bars = 200

        i = 50
        while i < n - max_bars:
            row = df.iloc[i]
            base_val = row['base_m5']
            quote_val = row['quote_m5']
            base_h4_vel = row['base_h4_vel']
            quote_h4_vel = row['quote_h4_vel']

            trade = None

            if strategy == 'mean_reversion':
                # MEAN REVERSION: Entry at extreme, exit at 0

                # BUY when base at negative extreme (will recover)
                if base_val <= -0.5:
                    # H4 velocity filter
                    if h4_vel_filter == 'against_trend':
                        # For mean reversion, we want H4 moving AGAINST the extreme
                        # Base is at -0.5 (bearish), we want H4 vel positive (recovering)
                        if base_h4_vel <= 0:
                            i += 1
                            continue
                    elif h4_vel_filter == 'with_trend':
                        # H4 still moving down = trend continues, skip
                        if base_h4_vel >= 0:
                            i += 1
                            continue

                    trade = {'type': 'BUY', 'trigger': 'base', 'trigger_val': base_val}

                # SELL when base at positive extreme (will pull back)
                elif base_val >= 0.5:
                    if h4_vel_filter == 'against_trend':
                        # Base is at +0.5 (bullish), we want H4 vel negative (reversing)
                        if base_h4_vel >= 0:
                            i += 1
                            continue
                    elif h4_vel_filter == 'with_trend':
                        if base_h4_vel <= 0:
                            i += 1
                            continue

                    trade = {'type': 'SELL', 'trigger': 'base', 'trigger_val': base_val}

            elif strategy == 'momentum':
                # MOMENTUM: Entry when crossing 0, exit at extreme
                prev_base = df.iloc[i-1]['base_m5'] if i > 0 else 0

                # Cross UP through 0 -> BUY
                if prev_base <= 0 and base_val > 0:
                    if h4_vel_filter == 'with_trend':
                        # For momentum, we want H4 moving WITH the cross
                        if base_h4_vel <= 0:
                            i += 1
                            continue
                    elif h4_vel_filter == 'against_trend':
                        if base_h4_vel >= 0:
                            i += 1
                            continue

                    trade = {'type': 'BUY', 'trigger': 'base', 'trigger_val': base_val}

                # Cross DOWN through 0 -> SELL
                elif prev_base >= 0 and base_val < 0:
                    if h4_vel_filter == 'with_trend':
                        if base_h4_vel >= 0:
                            i += 1
                            continue
                    elif h4_vel_filter == 'against_trend':
                        if base_h4_vel <= 0:
                            i += 1
                            continue

                    trade = {'type': 'SELL', 'trigger': 'base', 'trigger_val': base_val}

            if trade is not None:
                entry_idx = i + 1
                entry_price = df.iloc[entry_idx]['price']

                # Find exit
                future_df = df.iloc[entry_idx:entry_idx+max_bars]
                future_mfc = future_df['base_m5'].values
                future_price = future_df['price'].values

                if strategy == 'mean_reversion':
                    # Exit when MFC crosses 0
                    if trade['type'] == 'BUY':
                        exit_mask = future_mfc >= 0
                    else:
                        exit_mask = future_mfc <= 0
                else:  # momentum
                    # Exit when MFC reaches extreme
                    if trade['type'] == 'BUY':
                        exit_mask = future_mfc >= 0.5
                    else:
                        exit_mask = future_mfc <= -0.5

                if exit_mask.any():
                    exit_idx = np.argmax(exit_mask)
                    exit_price = future_price[exit_idx]
                    bars_held = exit_idx + 1
                else:
                    exit_price = future_price[-1]
                    bars_held = len(future_price)

                # Calculate PnL
                if trade['type'] == 'BUY':
                    pips = (exit_price - entry_price) / pip_val
                else:
                    pips = (entry_price - exit_price) / pip_val

                net_pips = pips - SPREADS.get(pair, 2.0)

                all_trades.append({
                    'pair': pair,
                    'type': trade['type'],
                    'net_pips': net_pips,
                    'bars_held': bars_held,
                    'h4_vel': base_h4_vel,
                })

                i += bars_held
            else:
                i += 1

    return pd.DataFrame(all_trades)


# ============================================================================
# RUN TESTS
# ============================================================================

log("\n" + "=" * 70)
log("MEAN REVERSION STRATEGY")
log("=" * 70)

log("\n--- No Filter (Baseline) ---")
df_base = run_backtest('mean_reversion', None)
if len(df_base) > 0:
    wr = (df_base['net_pips'] > 0).mean() * 100
    avg = df_base['net_pips'].mean()
    total = df_base['net_pips'].sum()
    winners = df_base[df_base['net_pips'] > 0]['net_pips'].sum()
    losers = abs(df_base[df_base['net_pips'] <= 0]['net_pips'].sum())
    pf = winners / losers if losers > 0 else 0
    log(f"Trades: {len(df_base):,}")
    log(f"Win Rate: {wr:.1f}%")
    log(f"Avg Pips: {avg:+.1f}")
    log(f"Total Pips: {total:+,.0f}")
    log(f"Profit Factor: {pf:.2f}")

log("\n--- H4 Velocity AGAINST Trend (should help mean reversion) ---")
df_against = run_backtest('mean_reversion', 'against_trend')
if len(df_against) > 0:
    wr = (df_against['net_pips'] > 0).mean() * 100
    avg = df_against['net_pips'].mean()
    total = df_against['net_pips'].sum()
    winners = df_against[df_against['net_pips'] > 0]['net_pips'].sum()
    losers = abs(df_against[df_against['net_pips'] <= 0]['net_pips'].sum())
    pf = winners / losers if losers > 0 else 0
    log(f"Trades: {len(df_against):,}")
    log(f"Win Rate: {wr:.1f}%")
    log(f"Avg Pips: {avg:+.1f}")
    log(f"Total Pips: {total:+,.0f}")
    log(f"Profit Factor: {pf:.2f}")

log("\n--- H4 Velocity WITH Trend (should hurt mean reversion) ---")
df_with = run_backtest('mean_reversion', 'with_trend')
if len(df_with) > 0:
    wr = (df_with['net_pips'] > 0).mean() * 100
    avg = df_with['net_pips'].mean()
    total = df_with['net_pips'].sum()
    winners = df_with[df_with['net_pips'] > 0]['net_pips'].sum()
    losers = abs(df_with[df_with['net_pips'] <= 0]['net_pips'].sum())
    pf = winners / losers if losers > 0 else 0
    log(f"Trades: {len(df_with):,}")
    log(f"Win Rate: {wr:.1f}%")
    log(f"Avg Pips: {avg:+.1f}")
    log(f"Total Pips: {total:+,.0f}")
    log(f"Profit Factor: {pf:.2f}")

log("\n" + "=" * 70)
log("MOMENTUM STRATEGY")
log("=" * 70)

log("\n--- No Filter (Baseline) ---")
df_base = run_backtest('momentum', None)
if len(df_base) > 0:
    wr = (df_base['net_pips'] > 0).mean() * 100
    avg = df_base['net_pips'].mean()
    total = df_base['net_pips'].sum()
    winners = df_base[df_base['net_pips'] > 0]['net_pips'].sum()
    losers = abs(df_base[df_base['net_pips'] <= 0]['net_pips'].sum())
    pf = winners / losers if losers > 0 else 0
    log(f"Trades: {len(df_base):,}")
    log(f"Win Rate: {wr:.1f}%")
    log(f"Avg Pips: {avg:+.1f}")
    log(f"Total Pips: {total:+,.0f}")
    log(f"Profit Factor: {pf:.2f}")

log("\n--- H4 Velocity WITH Trend (should help momentum) ---")
df_with = run_backtest('momentum', 'with_trend')
if len(df_with) > 0:
    wr = (df_with['net_pips'] > 0).mean() * 100
    avg = df_with['net_pips'].mean()
    total = df_with['net_pips'].sum()
    winners = df_with[df_with['net_pips'] > 0]['net_pips'].sum()
    losers = abs(df_with[df_with['net_pips'] <= 0]['net_pips'].sum())
    pf = winners / losers if losers > 0 else 0
    log(f"Trades: {len(df_with):,}")
    log(f"Win Rate: {wr:.1f}%")
    log(f"Avg Pips: {avg:+.1f}")
    log(f"Total Pips: {total:+,.0f}")
    log(f"Profit Factor: {pf:.2f}")

log("\n--- H4 Velocity AGAINST Trend (should hurt momentum) ---")
df_against = run_backtest('momentum', 'against_trend')
if len(df_against) > 0:
    wr = (df_against['net_pips'] > 0).mean() * 100
    avg = df_against['net_pips'].mean()
    total = df_against['net_pips'].sum()
    winners = df_against[df_against['net_pips'] > 0]['net_pips'].sum()
    losers = abs(df_against[df_against['net_pips'] <= 0]['net_pips'].sum())
    pf = winners / losers if losers > 0 else 0
    log(f"Trades: {len(df_against):,}")
    log(f"Win Rate: {wr:.1f}%")
    log(f"Avg Pips: {avg:+.1f}")
    log(f"Total Pips: {total:+,.0f}")
    log(f"Profit Factor: {pf:.2f}")

log(f"\nCompleted: {datetime.now()}")
