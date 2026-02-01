"""
Backtest Dip Strategy V2 - With Quote Currency Filter
======================================================
Entry: Base dips within extreme zone + bouncing + H1+H4 extreme + Quote NOT strong
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
log("BACKTEST: DIP STRATEGY V2 (with quote filter)")
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
MAX_BARS = 200
TEST_START = '2020-01-01'

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

# Test different filter combinations
def run_test(quote_filter='none', quote_threshold=0.3):
    """
    quote_filter options:
    - 'none': No quote filter
    - 'not_extreme': Quote not in extreme (opposite direction)
    - 'weak': Quote below threshold (showing weakness)
    - 'divergence': Quote velocity opposite to base
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
        base_h1 = mfc_h1[base].reindex(price.index, method='ffill')
        quote_h1 = mfc_h1[quote].reindex(price.index, method='ffill')
        base_h4 = mfc_h4[base].reindex(price.index, method='ffill')
        quote_h4 = mfc_h4[quote].reindex(price.index, method='ffill')

        df = pd.DataFrame(index=price.index)
        df['price'] = price
        df['base_m5'] = base_m5.shift(1)
        df['quote_m5'] = quote_m5.shift(1)
        df['base_h1'] = base_h1.shift(1)
        df['quote_h1'] = quote_h1.shift(1)
        df['base_h4'] = base_h4.shift(1)
        df['quote_h4'] = quote_h4.shift(1)

        df['base_vel_3'] = df['base_m5'].diff(3) / 3
        df['quote_vel_3'] = df['quote_m5'].diff(3) / 3

        df = df.dropna()

        for i in range(50, len(df) - MAX_BARS):
            row = df.iloc[i]
            base_val = row['base_m5']
            base_vel = row['base_vel_3']
            base_h1_val = row['base_h1']
            base_h4_val = row['base_h4']
            quote_val = row['quote_m5']
            quote_vel = row['quote_vel_3']
            quote_h1_val = row['quote_h1']

            # === LONG (BUY) ===
            # Base dipping positive, bouncing
            if DIP_ENTRY_MIN <= base_val <= DIP_ENTRY_MAX:
                lookback = df['base_m5'].iloc[i-50:i].values
                if lookback.max() >= EXTREME_LEVEL:
                    is_bouncing = base_vel > 0
                    h1_in_extreme = base_h1_val >= 0.4
                    h4_in_extreme = base_h4_val >= 0.4

                    # Quote filter for BUY: quote should be weak/not strong
                    if quote_filter == 'none':
                        quote_ok = True
                    elif quote_filter == 'not_extreme':
                        quote_ok = quote_val < 0.4  # Quote not strongly positive
                    elif quote_filter == 'weak':
                        quote_ok = quote_val <= quote_threshold  # Quote is weak/negative
                    elif quote_filter == 'divergence':
                        quote_ok = quote_vel < 0  # Quote momentum down while base bouncing up

                    if is_bouncing and h1_in_extreme and h4_in_extreme and quote_ok:
                        entry_idx = i + 1
                        entry_price = df.iloc[entry_idx]['price']
                        entry_time = df.index[entry_idx]

                        future_base = df['base_m5'].iloc[entry_idx:entry_idx+MAX_BARS].values
                        future_price = df['price'].iloc[entry_idx:entry_idx+MAX_BARS].values

                        target_mask = future_base >= EXTREME_LEVEL
                        stop_mask = future_base <= 0

                        bars_to_target = np.argmax(target_mask) + 1 if target_mask.any() else MAX_BARS + 1
                        bars_to_stop = np.argmax(stop_mask) + 1 if stop_mask.any() else MAX_BARS + 1

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
                            'net_pips': net_pips,
                            'win': 1 if net_pips > 0 else 0,
                            'exit_reason': exit_reason,
                            'bars_held': bars_held,
                        })

            # === SHORT (SELL) ===
            # Base dipping negative, bouncing down
            elif -DIP_ENTRY_MAX <= base_val <= -DIP_ENTRY_MIN:
                lookback = df['base_m5'].iloc[i-50:i].values
                if lookback.min() <= -EXTREME_LEVEL:
                    is_bouncing = base_vel < 0
                    h1_in_extreme = base_h1_val <= -0.4
                    h4_in_extreme = base_h4_val <= -0.4

                    # Quote filter for SELL: quote should be strong
                    if quote_filter == 'none':
                        quote_ok = True
                    elif quote_filter == 'not_extreme':
                        quote_ok = quote_val > -0.4  # Quote not strongly negative
                    elif quote_filter == 'weak':
                        quote_ok = quote_val >= -quote_threshold  # Quote not weak
                    elif quote_filter == 'divergence':
                        quote_ok = quote_vel > 0  # Quote momentum up while base bouncing down

                    if is_bouncing and h1_in_extreme and h4_in_extreme and quote_ok:
                        entry_idx = i + 1
                        entry_price = df.iloc[entry_idx]['price']
                        entry_time = df.index[entry_idx]

                        future_base = df['base_m5'].iloc[entry_idx:entry_idx+MAX_BARS].values
                        future_price = df['price'].iloc[entry_idx:entry_idx+MAX_BARS].values

                        target_mask = future_base <= -EXTREME_LEVEL
                        stop_mask = future_base >= 0

                        bars_to_target = np.argmax(target_mask) + 1 if target_mask.any() else MAX_BARS + 1
                        bars_to_stop = np.argmax(stop_mask) + 1 if stop_mask.any() else MAX_BARS + 1

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
                            'net_pips': net_pips,
                            'win': 1 if net_pips > 0 else 0,
                            'exit_reason': exit_reason,
                            'bars_held': bars_held,
                        })

    return pd.DataFrame(all_trades)


# Test different filters
log("\n" + "=" * 70)
log("TESTING DIFFERENT QUOTE FILTERS")
log("=" * 70)

filters = [
    ('none', 0),
    ('not_extreme', 0),
    ('weak', 0.0),
    ('weak', -0.2),
    ('divergence', 0),
]

log(f"\n{'Filter':<25} {'Trades':>8} {'WR':>8} {'Avg':>8} {'Total':>10} {'PF':>8}")
log("-" * 75)

for filter_type, threshold in filters:
    df = run_test(quote_filter=filter_type, quote_threshold=threshold)
    if len(df) > 0:
        wr = df['win'].mean() * 100
        avg = df['net_pips'].mean()
        total = df['net_pips'].sum()
        winners = df[df['net_pips'] > 0]['net_pips'].sum()
        losers = abs(df[df['net_pips'] <= 0]['net_pips'].sum())
        pf = winners / losers if losers > 0 else 0

        name = f"{filter_type}" + (f" ({threshold})" if threshold else "")
        log(f"{name:<25} {len(df):>8,} {wr:>7.1f}% {avg:>+7.1f} {total:>+10.0f} {pf:>7.2f}")

# Best filter detailed analysis
log("\n" + "=" * 70)
log("DETAILED ANALYSIS: DIVERGENCE FILTER")
log("=" * 70)

df = run_test(quote_filter='divergence')
if len(df) > 0:
    log(f"\nTrades: {len(df):,}")
    log(f"Win Rate: {df['win'].mean()*100:.1f}%")
    log(f"Avg Pips: {df['net_pips'].mean():+.1f}")
    log(f"Total Pips: {df['net_pips'].sum():+.0f}")

    winners = df[df['net_pips'] > 0]['net_pips'].sum()
    losers = abs(df[df['net_pips'] <= 0]['net_pips'].sum())
    pf = winners / losers if losers > 0 else 0
    log(f"Profit Factor: {pf:.2f}")

    log(f"\nBy Year:")
    df['year'] = pd.to_datetime(df['entry_time']).dt.year
    for year in sorted(df['year'].unique()):
        subset = df[df['year'] == year]
        log(f"  {year}: {len(subset)} trades, {subset['win'].mean()*100:.1f}% WR, {subset['net_pips'].sum():+.0f} pips")

    log(f"\nBy Trigger Currency:")
    for ccy in CURRENCIES:
        subset = df[df['trigger_ccy'] == ccy]
        if len(subset) > 0:
            log(f"  {ccy}: {len(subset)} trades, {subset['win'].mean()*100:.1f}% WR, {subset['net_pips'].sum():+.0f} pips")

log(f"\nCompleted: {datetime.now()}")
