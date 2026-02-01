"""
Analyze DinoTrader + MFC filter
Only take DinoTrader signals when MFC confirms direction
"""
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

DINO_DIR = Path('/mnt/c/Users/luizh/AppData/Roaming/MetaQuotes/Terminal/Common/Files')
DATA_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/data')

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

CURRENCIES = ['EUR', 'USD', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD', 'NZD']

def get_pip_value(pair):
    return 0.01 if 'JPY' in pair else 0.0001

print("=" * 70)
print("DINOTRADER + MFC FILTER ANALYSIS")
print("=" * 70)
print(f"Started: {datetime.now()}")

# Load H1 signals
TIMEFRAME = 'H1'
print(f"\nLoading {TIMEFRAME} signals...")

all_signals = []
for pair, base, quote in ALL_PAIRS:
    try:
        file_path = DINO_DIR / f'sinais_{pair}_TDS_DinoTrader_PERIOD_{TIMEFRAME}.csv'
        df = pd.read_csv(file_path, sep=';')
        df['pair'] = pair
        df['base'] = base
        df['quote'] = quote
        df['datetime'] = pd.to_datetime(df['time_utc_iso'])
        all_signals.append(df)
    except:
        pass

signals_df = pd.concat(all_signals, ignore_index=True)
print(f"Total signals: {len(signals_df)}")

# Load MFC data (H1 for matching timeframe)
print("\nLoading MFC H1 data...")
mfc_h1 = {}
for ccy in CURRENCIES:
    try:
        df = pd.read_csv(DATA_DIR / 'cleaned' / f'mfc_currency_{ccy}_H1_clean.csv')
        df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y.%m.%d %H:%M')
        df = df.set_index('datetime')
        mfc_h1[ccy] = df['MFC']
    except Exception as e:
        print(f"  Error loading {ccy}: {e}")

print(f"Loaded MFC for {len(mfc_h1)} currencies")

# Load price data
print("\nLoading price data...")
price_data = {}

for pair, base, quote in ALL_PAIRS:
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

print(f"Loaded price data for {len(price_data)} pairs")

def backtest_dino_mfc(pair, base, quote, mfc_threshold=0.3, start_date='2021-01-01', end_date='2025-12-31'):
    """
    Backtest DinoTrader with MFC filter
    BUY: only if base MFC < -threshold (base oversold, expecting rise)
    SELL: only if base MFC > +threshold (base overbought, expecting fall)
    """
    if pair not in price_data:
        return []
    if base not in mfc_h1 or quote not in mfc_h1:
        return []

    prices = price_data[pair]
    pair_signals = signals_df[signals_df['pair'] == pair].copy()
    pair_signals = pair_signals[(pair_signals['datetime'] >= start_date) &
                                 (pair_signals['datetime'] <= end_date)]
    pair_signals = pair_signals.sort_values('datetime')

    base_mfc = mfc_h1[base]
    quote_mfc = mfc_h1[quote]

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

        # Get MFC at signal time (shifted by 1 for no lookahead)
        try:
            base_mfc_val = base_mfc.shift(1).loc[:signal_time].iloc[-1]
            quote_mfc_val = quote_mfc.shift(1).loc[:signal_time].iloc[-1]
        except:
            continue

        if pd.isna(base_mfc_val) or pd.isna(quote_mfc_val):
            continue

        # MFC filter conditions
        # BUY: base should be oversold (negative MFC) and/or quote overbought
        # SELL: base should be overbought (positive MFC) and/or quote oversold
        mfc_confirms_buy = (base_mfc_val <= -mfc_threshold)
        mfc_confirms_sell = (base_mfc_val >= mfc_threshold)

        if signal_type == 'ENTRY_BUY' and position is None:
            if mfc_confirms_buy:  # Only enter if MFC confirms
                position = 'BUY'
                entry_time = next_bar_time
                entry_price = next_bar_open

        elif signal_type == 'ENTRY_SELL' and position is None:
            if mfc_confirms_sell:  # Only enter if MFC confirms
                position = 'SELL'
                entry_time = next_bar_time
                entry_price = next_bar_open

        elif signal_type == 'EXIT_BUY' and position == 'BUY':
            exit_price = next_bar_open
            pips = (exit_price - entry_price) / pip_val
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

        elif signal_type == 'EXIT_SELL' and position == 'SELL':
            exit_price = next_bar_open
            pips = (entry_price - exit_price) / pip_val
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

    return trades

# Test different MFC thresholds
print("\n" + "=" * 70)
print("TESTING MFC THRESHOLDS")
print("=" * 70)

for mfc_thresh in [0.2, 0.3, 0.4, 0.5, 0.6]:
    all_trades = []
    for pair, base, quote in ALL_PAIRS:
        trades = backtest_dino_mfc(pair, base, quote, mfc_threshold=mfc_thresh)
        all_trades.extend(trades)

    if all_trades:
        df = pd.DataFrame(all_trades)
        print(f"MFC >= {mfc_thresh}: {len(df):>5} trades, {df['win'].mean()*100:>5.1f}% WR, {df['net_pips'].mean():>+6.1f} avg, {df['net_pips'].sum():>+8.0f} total")

# Detailed analysis with best threshold
print("\n" + "=" * 70)
print("DETAILED ANALYSIS (MFC >= 0.5)")
print("=" * 70)

MFC_THRESHOLD = 0.5
all_trades = []
for pair, base, quote in ALL_PAIRS:
    trades = backtest_dino_mfc(pair, base, quote, mfc_threshold=MFC_THRESHOLD)
    all_trades.extend(trades)
    if trades:
        df = pd.DataFrame(trades)
        print(f"{pair}: {len(trades):>4} trades, {df['win'].mean()*100:>5.1f}% WR, {df['net_pips'].sum():>+6.0f} pips")

trades_df = pd.DataFrame(all_trades)

if len(trades_df) > 0:
    print("\n" + "=" * 70)
    print("OVERALL RESULTS")
    print("=" * 70)

    print(f"\nTotal trades: {len(trades_df)}")
    print(f"Win rate: {trades_df['win'].mean()*100:.1f}%")
    print(f"Avg pips: {trades_df['net_pips'].mean():+.1f}")
    print(f"Total pips: {trades_df['net_pips'].sum():+.0f}")

    # Compare to baseline
    print("\n--- Comparison ---")
    print(f"DinoTrader alone: 42,137 trades, 35.9% WR, -170,400 pips")
    print(f"DinoTrader + MFC:  {len(trades_df)} trades, {trades_df['win'].mean()*100:.1f}% WR, {trades_df['net_pips'].sum():+.0f} pips")

    # By year
    print("\n--- By Year ---")
    trades_df['year'] = pd.to_datetime(trades_df['entry_time']).dt.year
    for year in sorted(trades_df['year'].unique()):
        year_df = trades_df[trades_df['year'] == year]
        print(f"{year}: {len(year_df)} trades, {year_df['win'].mean()*100:.1f}% WR, {year_df['net_pips'].sum():+.0f} pips")

    # Best pairs
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

    print("\n--- Worst Pairs ---")
    for pair in pair_stats.tail(5).index:
        row = pair_stats.loc[pair]
        print(f"  {pair}: {row['trades']:.0f} trades, {row['win_rate']*100:.1f}% WR, {row['total_pips']:+.0f} pips")

print(f"\nCompleted: {datetime.now()}")
