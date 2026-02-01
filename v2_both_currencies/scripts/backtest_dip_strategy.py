"""
Backtest Dip Strategy with Price PnL
====================================
Entry: MFC dips within extreme zone (0.3-0.45), bouncing, H1+H4 in extreme
Exit: MFC reaches ±0.5 (target) or crosses 0 (stop)
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
log("BACKTEST: DIP STRATEGY")
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

# Parameters
DIP_ENTRY_MIN = 0.3
DIP_ENTRY_MAX = 0.45
EXTREME_LEVEL = 0.5
MAX_BARS = 200  # Max hold time
TEST_START = '2020-01-01'  # Test on recent data

# Load MFC data
log("\nLoading MFC data...")
mfc_m5 = {}
mfc_h1 = {}
mfc_h4 = {}

for ccy in CURRENCIES:
    df = pd.read_csv(CLEANED_DIR / f'mfc_currency_{ccy}_M5_clean.csv')
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df = df.set_index('datetime').sort_index()
    mfc_m5[ccy] = df['MFC']

    df = pd.read_csv(CLEANED_DIR / f'mfc_currency_{ccy}_H1_clean.csv')
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df = df.set_index('datetime').sort_index()
    mfc_h1[ccy] = df['MFC']

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
            chunk = chunk[chunk.index >= TEST_START]
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

# Run backtest
log("\n" + "=" * 70)
log("RUNNING BACKTEST")
log("=" * 70)

all_trades = []

for pair, base, quote in ALL_PAIRS:
    if pair not in price_data:
        continue

    price = price_data[pair]
    pip_val = get_pip_value(pair)

    # Align MFC to price index
    base_m5 = mfc_m5[base].reindex(price.index, method='ffill')
    quote_m5 = mfc_m5[quote].reindex(price.index, method='ffill')
    base_h1 = mfc_h1[base].reindex(price.index, method='ffill')
    quote_h1 = mfc_h1[quote].reindex(price.index, method='ffill')
    base_h4 = mfc_h4[base].reindex(price.index, method='ffill')
    quote_h4 = mfc_h4[quote].reindex(price.index, method='ffill')

    # Create features DataFrame
    df = pd.DataFrame(index=price.index)
    df['price'] = price
    df['base_m5'] = base_m5.shift(1)  # Use previous bar
    df['quote_m5'] = quote_m5.shift(1)
    df['base_h1'] = base_h1.shift(1)
    df['quote_h1'] = quote_h1.shift(1)
    df['base_h4'] = base_h4.shift(1)
    df['quote_h4'] = quote_h4.shift(1)

    # Velocities
    df['base_vel_3'] = df['base_m5'].diff(3) / 3
    df['quote_vel_3'] = df['quote_m5'].diff(3) / 3

    df = df.dropna()

    # Find dip entries - LONG (buy pair when base dips)
    # Base was at extreme, now pulled back but still positive, and bouncing
    for i in range(50, len(df) - MAX_BARS):
        row = df.iloc[i]
        base_val = row['base_m5']
        base_vel = row['base_vel_3']
        base_h1_val = row['base_h1']
        base_h4_val = row['base_h4']

        # Check for base currency positive dip (buy opportunity)
        if DIP_ENTRY_MIN <= base_val <= DIP_ENTRY_MAX:
            # Check lookback for prior extreme
            lookback = df['base_m5'].iloc[i-50:i].values
            if lookback.max() >= EXTREME_LEVEL:
                # Check filters: bouncing + H1 extreme + H4 extreme
                is_bouncing = base_vel > 0
                h1_in_extreme = base_h1_val >= 0.4
                h4_in_extreme = base_h4_val >= 0.4

                if is_bouncing and h1_in_extreme and h4_in_extreme:
                    # Entry signal - BUY
                    entry_idx = i + 1
                    entry_price = df.iloc[entry_idx]['price']
                    entry_time = df.index[entry_idx]

                    # Find exit
                    future_base = df['base_m5'].iloc[entry_idx:entry_idx+MAX_BARS].values
                    future_price = df['price'].iloc[entry_idx:entry_idx+MAX_BARS].values

                    # Target: base reaches 0.5
                    target_mask = future_base >= EXTREME_LEVEL
                    # Stop: base crosses 0
                    stop_mask = future_base <= 0

                    if target_mask.any():
                        bars_to_target = np.argmax(target_mask) + 1
                    else:
                        bars_to_target = MAX_BARS + 1

                    if stop_mask.any():
                        bars_to_stop = np.argmax(stop_mask) + 1
                    else:
                        bars_to_stop = MAX_BARS + 1

                    if bars_to_target < bars_to_stop and bars_to_target <= MAX_BARS:
                        exit_price = future_price[bars_to_target - 1]
                        exit_reason = 'TARGET'
                        bars_held = bars_to_target
                    elif bars_to_stop <= MAX_BARS:
                        exit_price = future_price[bars_to_stop - 1]
                        exit_reason = 'STOP'
                        bars_held = bars_to_stop
                    else:
                        exit_price = future_price[-1]
                        exit_reason = 'TIMEOUT'
                        bars_held = MAX_BARS

                    pips = (exit_price - entry_price) / pip_val
                    net_pips = pips - SPREADS.get(pair, 2.0)

                    all_trades.append({
                        'pair': pair,
                        'type': 'BUY',
                        'trigger_ccy': base,
                        'entry_time': entry_time,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'pips': pips,
                        'net_pips': net_pips,
                        'win': 1 if net_pips > 0 else 0,
                        'exit_reason': exit_reason,
                        'bars_held': bars_held,
                    })

        # Check for base currency negative dip (sell opportunity)
        elif -DIP_ENTRY_MAX <= base_val <= -DIP_ENTRY_MIN:
            lookback = df['base_m5'].iloc[i-50:i].values
            if lookback.min() <= -EXTREME_LEVEL:
                is_bouncing = base_vel < 0  # Bouncing down
                h1_in_extreme = base_h1_val <= -0.4
                h4_in_extreme = base_h4_val <= -0.4

                if is_bouncing and h1_in_extreme and h4_in_extreme:
                    entry_idx = i + 1
                    entry_price = df.iloc[entry_idx]['price']
                    entry_time = df.index[entry_idx]

                    future_base = df['base_m5'].iloc[entry_idx:entry_idx+MAX_BARS].values
                    future_price = df['price'].iloc[entry_idx:entry_idx+MAX_BARS].values

                    target_mask = future_base <= -EXTREME_LEVEL
                    stop_mask = future_base >= 0

                    if target_mask.any():
                        bars_to_target = np.argmax(target_mask) + 1
                    else:
                        bars_to_target = MAX_BARS + 1

                    if stop_mask.any():
                        bars_to_stop = np.argmax(stop_mask) + 1
                    else:
                        bars_to_stop = MAX_BARS + 1

                    if bars_to_target < bars_to_stop and bars_to_target <= MAX_BARS:
                        exit_price = future_price[bars_to_target - 1]
                        exit_reason = 'TARGET'
                        bars_held = bars_to_target
                    elif bars_to_stop <= MAX_BARS:
                        exit_price = future_price[bars_to_stop - 1]
                        exit_reason = 'STOP'
                        bars_held = bars_to_stop
                    else:
                        exit_price = future_price[-1]
                        exit_reason = 'TIMEOUT'
                        bars_held = MAX_BARS

                    pips = (entry_price - exit_price) / pip_val
                    net_pips = pips - SPREADS.get(pair, 2.0)

                    all_trades.append({
                        'pair': pair,
                        'type': 'SELL',
                        'trigger_ccy': base,
                        'entry_time': entry_time,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'pips': pips,
                        'net_pips': net_pips,
                        'win': 1 if net_pips > 0 else 0,
                        'exit_reason': exit_reason,
                        'bars_held': bars_held,
                    })

trades_df = pd.DataFrame(all_trades)
log(f"\nTotal trades: {len(trades_df):,}")

if len(trades_df) > 0:
    log("\n" + "=" * 70)
    log("RESULTS")
    log("=" * 70)

    wr = trades_df['win'].mean() * 100
    avg_pips = trades_df['net_pips'].mean()
    total_pips = trades_df['net_pips'].sum()

    winners = trades_df[trades_df['net_pips'] > 0]['net_pips'].sum()
    losers = abs(trades_df[trades_df['net_pips'] <= 0]['net_pips'].sum())
    pf = winners / losers if losers > 0 else float('inf')

    log(f"\nWin Rate: {wr:.1f}%")
    log(f"Avg Pips: {avg_pips:+.1f}")
    log(f"Total Pips: {total_pips:+.0f}")
    log(f"Profit Factor: {pf:.2f}")

    # By exit reason
    log(f"\nBy Exit Reason:")
    for reason in ['TARGET', 'STOP', 'TIMEOUT']:
        subset = trades_df[trades_df['exit_reason'] == reason]
        if len(subset) > 0:
            log(f"  {reason}: {len(subset)} trades, {subset['win'].mean()*100:.1f}% WR, {subset['net_pips'].sum():+.0f} pips")

    # By year
    log(f"\nBy Year:")
    trades_df['year'] = pd.to_datetime(trades_df['entry_time']).dt.year
    for year in sorted(trades_df['year'].unique()):
        subset = trades_df[trades_df['year'] == year]
        log(f"  {year}: {len(subset)} trades, {subset['win'].mean()*100:.1f}% WR, {subset['net_pips'].sum():+.0f} pips")

    # By trigger currency
    log(f"\nBy Trigger Currency:")
    for ccy in CURRENCIES:
        subset = trades_df[trades_df['trigger_ccy'] == ccy]
        if len(subset) > 0:
            log(f"  {ccy}: {len(subset)} trades, {subset['win'].mean()*100:.1f}% WR, {subset['net_pips'].sum():+.0f} pips")

    # Avg hold time
    avg_bars = trades_df['bars_held'].mean()
    log(f"\nAvg hold time: {avg_bars:.0f} bars ({avg_bars*5/60:.1f}h)")

    # Trades per day
    days = (trades_df['entry_time'].max() - trades_df['entry_time'].min()).days
    tpd = len(trades_df) / days if days > 0 else 0
    log(f"Trades per day: {tpd:.1f}")

log(f"\nCompleted: {datetime.now()}")
