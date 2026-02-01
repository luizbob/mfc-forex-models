"""
Backtest Dip Strategy - Price-Based Targets
============================================
Entry: MFC dip within extreme zone + bouncing + H1+H4 extreme
Exit: Stochastic crosses (like LSTM model) or TP/SL in pips
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
log("BACKTEST: DIP STRATEGY (PRICE-BASED EXIT)")
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

def calculate_stochastic(high, low, close, period=25):
    lowest_low = low.rolling(window=period).min()
    highest_high = high.rolling(window=period).max()
    stoch_k = ((close - lowest_low) / (highest_high - lowest_low)) * 100
    return stoch_k

# Parameters
DIP_ENTRY_MIN = 0.3
DIP_ENTRY_MAX = 0.45
EXTREME_LEVEL = 0.5
MAX_BARS = 200
TEST_START = '2023-01-01'  # Recent data for faster test

# Exit parameters
STOCH_PERIOD = 25
STOCH_EXIT_HIGH = 80  # Exit BUY when stoch > 80
STOCH_EXIT_LOW = 20   # Exit SELL when stoch < 20

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

# Load price data with OHLC
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
                m5 = chunk[['Open', 'High', 'Low', 'Close']].resample('5min').agg({
                    'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
                }).dropna()
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

    # Calculate stochastic
    stoch = calculate_stochastic(price['High'], price['Low'], price['Close'], STOCH_PERIOD)

    # Align MFC to price index
    base_m5 = mfc_m5[base].reindex(price.index, method='ffill')
    quote_m5 = mfc_m5[quote].reindex(price.index, method='ffill')
    base_h1 = mfc_h1[base].reindex(price.index, method='ffill')
    quote_h1 = mfc_h1[quote].reindex(price.index, method='ffill')
    base_h4 = mfc_h4[base].reindex(price.index, method='ffill')
    quote_h4 = mfc_h4[quote].reindex(price.index, method='ffill')

    df = pd.DataFrame(index=price.index)
    df['open'] = price['Open']
    df['close'] = price['Close']
    df['stoch'] = stoch.shift(1)  # Use previous bar's stoch
    df['base_m5'] = base_m5.shift(1)
    df['quote_m5'] = quote_m5.shift(1)
    df['base_h1'] = base_h1.shift(1)
    df['quote_h1'] = quote_h1.shift(1)
    df['base_h4'] = base_h4.shift(1)
    df['quote_h4'] = quote_h4.shift(1)
    df['base_vel_3'] = df['base_m5'].diff(3) / 3
    df['quote_vel_3'] = df['quote_m5'].diff(3) / 3

    df = df.dropna()

    i = 50
    while i < len(df) - MAX_BARS:
        row = df.iloc[i]
        base_val = row['base_m5']
        base_vel = row['base_vel_3']
        base_h1_val = row['base_h1']
        base_h4_val = row['base_h4']
        quote_val = row['quote_m5']

        # === LONG (BUY) - Base dipping positive ===
        if DIP_ENTRY_MIN <= base_val <= DIP_ENTRY_MAX:
            lookback = df['base_m5'].iloc[i-50:i].values
            if lookback.max() >= EXTREME_LEVEL:
                is_bouncing = base_vel > 0
                h1_in_extreme = base_h1_val >= 0.4
                h4_in_extreme = base_h4_val >= 0.4
                # Quote divergence: quote not strengthening
                quote_diverge = quote_val < 0.3 or row['quote_vel_3'] <= 0

                if is_bouncing and h1_in_extreme and h4_in_extreme and quote_diverge:
                    entry_idx = i + 1
                    entry_price = df.iloc[entry_idx]['open']
                    entry_time = df.index[entry_idx]

                    # Find exit using stochastic
                    future_df = df.iloc[entry_idx+1:entry_idx+1+MAX_BARS]

                    exit_mask = future_df['stoch'] >= STOCH_EXIT_HIGH
                    if exit_mask.any():
                        exit_idx = exit_mask.argmax()
                        exit_price = future_df.iloc[exit_idx]['close']
                        exit_reason = 'STOCH'
                        bars_held = exit_idx + 1
                    else:
                        exit_price = future_df.iloc[-1]['close']
                        exit_reason = 'TIMEOUT'
                        bars_held = len(future_df)

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
                        'base_mfc': base_val,
                        'quote_mfc': quote_val,
                    })

                    # Skip bars while in trade
                    i += bars_held
                    continue

        # === SHORT (SELL) - Base dipping negative ===
        elif -DIP_ENTRY_MAX <= base_val <= -DIP_ENTRY_MIN:
            lookback = df['base_m5'].iloc[i-50:i].values
            if lookback.min() <= -EXTREME_LEVEL:
                is_bouncing = base_vel < 0
                h1_in_extreme = base_h1_val <= -0.4
                h4_in_extreme = base_h4_val <= -0.4
                # Quote divergence: quote not weakening
                quote_diverge = quote_val > -0.3 or row['quote_vel_3'] >= 0

                if is_bouncing and h1_in_extreme and h4_in_extreme and quote_diverge:
                    entry_idx = i + 1
                    entry_price = df.iloc[entry_idx]['open']
                    entry_time = df.index[entry_idx]

                    future_df = df.iloc[entry_idx+1:entry_idx+1+MAX_BARS]

                    exit_mask = future_df['stoch'] <= STOCH_EXIT_LOW
                    if exit_mask.any():
                        exit_idx = exit_mask.argmax()
                        exit_price = future_df.iloc[exit_idx]['close']
                        exit_reason = 'STOCH'
                        bars_held = exit_idx + 1
                    else:
                        exit_price = future_df.iloc[-1]['close']
                        exit_reason = 'TIMEOUT'
                        bars_held = len(future_df)

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
                        'base_mfc': base_val,
                        'quote_mfc': quote_val,
                    })

                    i += bars_held
                    continue

        i += 1

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

    log(f"\nBy Exit Reason:")
    for reason in trades_df['exit_reason'].unique():
        subset = trades_df[trades_df['exit_reason'] == reason]
        log(f"  {reason}: {len(subset)} trades, {subset['win'].mean()*100:.1f}% WR, {subset['net_pips'].sum():+.0f} pips")

    log(f"\nBy Year:")
    trades_df['year'] = pd.to_datetime(trades_df['entry_time']).dt.year
    for year in sorted(trades_df['year'].unique()):
        subset = trades_df[trades_df['year'] == year]
        log(f"  {year}: {len(subset)} trades, {subset['win'].mean()*100:.1f}% WR, {subset['net_pips'].sum():+.0f} pips")

    log(f"\nBy Trigger Currency:")
    for ccy in CURRENCIES:
        subset = trades_df[trades_df['trigger_ccy'] == ccy]
        if len(subset) > 0:
            log(f"  {ccy}: {len(subset)} trades, {subset['win'].mean()*100:.1f}% WR, {subset['net_pips'].sum():+.0f} pips")

    # Trades per day
    days = (trades_df['entry_time'].max() - trades_df['entry_time'].min()).days
    tpd = len(trades_df) / days if days > 0 else 0
    log(f"\nTrades per day: {tpd:.1f}")
    log(f"Avg hold time: {trades_df['bars_held'].mean():.0f} bars ({trades_df['bars_held'].mean()*5/60:.1f}h)")

log(f"\nCompleted: {datetime.now()}")
