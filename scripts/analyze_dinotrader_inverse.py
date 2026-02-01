"""
Analyze DinoTrader signals - INVERTED
If they lose at 35.9% WR, inverse should win at ~64% WR
"""
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

DINO_DIR = Path('/mnt/c/Users/luizh/AppData/Roaming/MetaQuotes/Terminal/Common/Files')
DATA_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/data')

ALL_PAIRS = [
    'EURUSD', 'GBPUSD', 'AUDUSD', 'NZDUSD', 'USDJPY', 'USDCHF', 'USDCAD',
    'EURGBP', 'EURJPY', 'EURCHF', 'EURCAD', 'EURAUD', 'EURNZD',
    'GBPJPY', 'GBPCHF', 'GBPCAD', 'GBPAUD', 'GBPNZD',
    'AUDJPY', 'AUDCHF', 'AUDCAD', 'AUDNZD',
    'NZDJPY', 'NZDCHF', 'NZDCAD',
    'CADJPY', 'CADCHF', 'CHFJPY',
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

print("=" * 70)
print("DINOTRADER INVERTED SIGNAL ANALYSIS")
print("=" * 70)
print(f"Started: {datetime.now()}")

# Load H1 signals
TIMEFRAME = 'H1'
print(f"\nLoading {TIMEFRAME} signals...")

all_signals = []
for pair in ALL_PAIRS:
    try:
        file_path = DINO_DIR / f'sinais_{pair}_TDS_DinoTrader_PERIOD_{TIMEFRAME}.csv'
        df = pd.read_csv(file_path, sep=';')
        df['pair'] = pair
        df['datetime'] = pd.to_datetime(df['time_utc_iso'])
        all_signals.append(df)
    except:
        pass

signals_df = pd.concat(all_signals, ignore_index=True)

# Load price data
print("Loading price data...")
price_data = {}

for pair in ALL_PAIRS:
    try:
        chunks = []
        for chunk in pd.read_csv(DATA_DIR / f'{pair}_GMT+0_US-DST_M1.csv', chunksize=500000):
            chunk['datetime'] = pd.to_datetime(chunk['Date'] + ' ' + chunk['Time'], format='%Y.%m.%d %H:%M:%S')
            chunk = chunk.set_index('datetime')
            if len(chunk) > 0:
                h1_chunk = chunk[['Open', 'High', 'Low', 'Close']].resample('1h').agg({
                    'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
                }).dropna()
                chunks.append(h1_chunk)

        if chunks:
            price_h1 = pd.concat(chunks)
            price_h1 = price_h1[~price_h1.index.duplicated(keep='first')]
            price_data[pair] = price_h1
    except:
        pass

print(f"Loaded {len(price_data)} pairs")

def backtest_inverted(pair, start_date='2021-01-01', end_date='2025-12-31'):
    """Backtest INVERTED DinoTrader signals"""
    if pair not in price_data:
        return []

    prices = price_data[pair]
    pair_signals = signals_df[signals_df['pair'] == pair].copy()
    pair_signals = pair_signals[(pair_signals['datetime'] >= start_date) &
                                 (pair_signals['datetime'] <= end_date)]
    pair_signals = pair_signals.sort_values('datetime')

    pip_val = get_pip_value(pair)
    spread = SPREADS.get(pair, 2.0)

    trades = []
    position = None
    entry_time = None
    entry_price = None

    for _, signal in pair_signals.iterrows():
        signal_time = signal['datetime']
        signal_type = signal['tipo_sinal']

        future_bars = prices[prices.index > signal_time]
        if len(future_bars) == 0:
            continue

        next_bar_time = future_bars.index[0]
        next_bar_open = future_bars.iloc[0]['Open']

        # INVERTED: ENTRY_BUY becomes SELL, ENTRY_SELL becomes BUY
        if signal_type == 'ENTRY_BUY' and position is None:
            position = 'SELL'  # INVERTED
            entry_time = next_bar_time
            entry_price = next_bar_open

        elif signal_type == 'ENTRY_SELL' and position is None:
            position = 'BUY'  # INVERTED
            entry_time = next_bar_time
            entry_price = next_bar_open

        # INVERTED exits too
        elif signal_type == 'EXIT_BUY' and position == 'SELL':  # was BUY, now SELL
            exit_price = next_bar_open
            pips = (entry_price - exit_price) / pip_val  # SELL calculation
            net_pips = pips - spread

            trades.append({
                'pair': pair,
                'type': 'SELL',
                'entry_time': entry_time,
                'exit_time': next_bar_time,
                'pips': pips,
                'net_pips': net_pips,
                'win': 1 if net_pips > 0 else 0,
                'bars_held': len(prices[(prices.index > entry_time) & (prices.index <= next_bar_time)]),
            })
            position = None

        elif signal_type == 'EXIT_SELL' and position == 'BUY':  # was SELL, now BUY
            exit_price = next_bar_open
            pips = (exit_price - entry_price) / pip_val  # BUY calculation
            net_pips = pips - spread

            trades.append({
                'pair': pair,
                'type': 'BUY',
                'entry_time': entry_time,
                'exit_time': next_bar_time,
                'pips': pips,
                'net_pips': net_pips,
                'win': 1 if net_pips > 0 else 0,
                'bars_held': len(prices[(prices.index > entry_time) & (prices.index <= next_bar_time)]),
            })
            position = None

    return trades

# Run inverted backtest
print("\n" + "=" * 70)
print("INVERTED BACKTEST (2021-2025)")
print("=" * 70)

all_trades = []
for pair in ALL_PAIRS:
    trades = backtest_inverted(pair, '2021-01-01', '2025-12-31')
    all_trades.extend(trades)
    if trades:
        df = pd.DataFrame(trades)
        print(f"{pair}: {len(trades)} trades, {df['win'].mean()*100:.1f}% WR, {df['net_pips'].sum():+.0f} pips")

trades_df = pd.DataFrame(all_trades)

if len(trades_df) > 0:
    print("\n" + "=" * 70)
    print("INVERTED RESULTS")
    print("=" * 70)

    print(f"\nTotal trades: {len(trades_df)}")
    print(f"Win rate: {trades_df['win'].mean()*100:.1f}%")
    print(f"Avg pips: {trades_df['net_pips'].mean():+.1f}")
    print(f"Total pips: {trades_df['net_pips'].sum():+.0f}")

    # By year
    print("\n--- By Year ---")
    trades_df['year'] = pd.to_datetime(trades_df['entry_time']).dt.year
    for year in sorted(trades_df['year'].unique()):
        year_df = trades_df[trades_df['year'] == year]
        print(f"{year}: {len(year_df)} trades, {year_df['win'].mean()*100:.1f}% WR, {year_df['net_pips'].sum():+.0f} pips")

    # By pair
    print("\n--- Best Pairs ---")
    pair_stats = trades_df.groupby('pair').agg({
        'net_pips': ['count', 'sum', 'mean'],
        'win': 'mean'
    }).round(2)
    pair_stats.columns = ['trades', 'total_pips', 'avg_pips', 'win_rate']
    pair_stats = pair_stats.sort_values('total_pips', ascending=False)

    for pair in pair_stats.head(10).index:
        row = pair_stats.loc[pair]
        print(f"  {pair}: {row['trades']:.0f} trades, {row['win_rate']*100:.1f}% WR, {row['total_pips']:+.0f} pips")

print(f"\nCompleted: {datetime.now()}")
