"""
Analyze DinoTrader H1 + MFC filters
Test 3 approaches:
1. H4 MFC extreme for confirmation
2. MFC velocity/direction
3. MFC divergence (base vs quote)
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
print("DINOTRADER + MFC FILTERS V2")
print("=" * 70)
print(f"Started: {datetime.now()}")

# Load H1 signals
print("\nLoading H1 signals...")
all_signals = []
for pair, base, quote in ALL_PAIRS:
    try:
        file_path = DINO_DIR / f'sinais_{pair}_TDS_DinoTrader_PERIOD_H1.csv'
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

# Load MFC data (H1 and H4)
print("\nLoading MFC data...")
mfc_h1 = {}
mfc_h4 = {}

for ccy in CURRENCIES:
    try:
        # H1
        df = pd.read_csv(DATA_DIR / 'cleaned' / f'mfc_currency_{ccy}_H1_clean.csv')
        df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y.%m.%d %H:%M')
        df = df.set_index('datetime')
        mfc_h1[ccy] = df['MFC']

        # H4
        df = pd.read_csv(DATA_DIR / 'cleaned' / f'mfc_currency_{ccy}_H4_clean.csv')
        df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y.%m.%d %H:%M')
        df = df.set_index('datetime')
        mfc_h4[ccy] = df['MFC']
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

# Prepare MFC velocity (H1)
print("\nCalculating MFC velocities...")
mfc_h1_vel = {}
for ccy in CURRENCIES:
    if ccy in mfc_h1:
        # Velocity over last 4 H1 bars (4 hours)
        mfc_h1_vel[ccy] = mfc_h1[ccy].diff(periods=4)

def backtest_filter(pair, base, quote, filter_type='h4_extreme', params=None,
                    start_date='2021-01-01', end_date='2025-12-31'):
    """
    Backtest DinoTrader with different MFC filters
    filter_type: 'h4_extreme', 'velocity', 'divergence'
    """
    if params is None:
        params = {}

    if pair not in price_data:
        return []
    if base not in mfc_h1 or quote not in mfc_h1:
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

        # Get MFC values
        try:
            if filter_type == 'h4_extreme':
                # Use H4 MFC for confirmation
                base_mfc = mfc_h4[base].shift(1).reindex(mfc_h1[base].index, method='ffill').loc[:signal_time].iloc[-1]
                quote_mfc = mfc_h4[quote].shift(1).reindex(mfc_h1[quote].index, method='ffill').loc[:signal_time].iloc[-1]
                threshold = params.get('threshold', 0.3)

                mfc_confirms_buy = (base_mfc <= -threshold)
                mfc_confirms_sell = (base_mfc >= threshold)

            elif filter_type == 'velocity':
                # Use MFC velocity direction
                base_vel = mfc_h1_vel[base].shift(1).loc[:signal_time].iloc[-1]
                quote_vel = mfc_h1_vel[quote].shift(1).loc[:signal_time].iloc[-1]
                vel_threshold = params.get('threshold', 0.02)

                # BUY: base velocity positive (MFC rising), quote velocity negative
                mfc_confirms_buy = (base_vel > vel_threshold)
                mfc_confirms_sell = (base_vel < -vel_threshold)

            elif filter_type == 'divergence':
                # Use MFC divergence (base vs quote)
                base_mfc = mfc_h1[base].shift(1).loc[:signal_time].iloc[-1]
                quote_mfc = mfc_h1[quote].shift(1).loc[:signal_time].iloc[-1]
                diff_threshold = params.get('threshold', 0.5)

                mfc_diff = base_mfc - quote_mfc
                # BUY: base weaker than quote (negative diff), expecting base to rise
                mfc_confirms_buy = (mfc_diff <= -diff_threshold)
                # SELL: base stronger than quote (positive diff), expecting base to fall
                mfc_confirms_sell = (mfc_diff >= diff_threshold)

            else:
                mfc_confirms_buy = True
                mfc_confirms_sell = True

        except:
            continue

        if pd.isna(mfc_confirms_buy) or pd.isna(mfc_confirms_sell):
            continue

        if signal_type == 'ENTRY_BUY' and position is None:
            if mfc_confirms_buy:
                position = 'BUY'
                entry_time = next_bar_time
                entry_price = next_bar_open

        elif signal_type == 'ENTRY_SELL' and position is None:
            if mfc_confirms_sell:
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

def run_test(filter_type, params, label):
    """Run backtest for all pairs"""
    all_trades = []
    for pair, base, quote in ALL_PAIRS:
        trades = backtest_filter(pair, base, quote, filter_type=filter_type, params=params)
        all_trades.extend(trades)

    if all_trades:
        df = pd.DataFrame(all_trades)
        print(f"{label}: {len(df):>5} trades, {df['win'].mean()*100:>5.1f}% WR, {df['net_pips'].mean():>+6.1f} avg, {df['net_pips'].sum():>+8.0f} total")
        return df
    return None

# ============================================================
# TEST 1: H4 MFC Extreme
# ============================================================
print("\n" + "=" * 70)
print("TEST 1: H4 MFC EXTREME")
print("=" * 70)

for thresh in [0.2, 0.3, 0.4, 0.5]:
    run_test('h4_extreme', {'threshold': thresh}, f"H4 MFC >= {thresh}")

# ============================================================
# TEST 2: MFC Velocity
# ============================================================
print("\n" + "=" * 70)
print("TEST 2: MFC VELOCITY (H1 over 4 bars)")
print("=" * 70)

for thresh in [0.01, 0.02, 0.03, 0.04, 0.05]:
    run_test('velocity', {'threshold': thresh}, f"Velocity >= {thresh}")

# ============================================================
# TEST 3: MFC Divergence
# ============================================================
print("\n" + "=" * 70)
print("TEST 3: MFC DIVERGENCE (base - quote)")
print("=" * 70)

for thresh in [0.3, 0.5, 0.7, 1.0]:
    run_test('divergence', {'threshold': thresh}, f"Divergence >= {thresh}")

# ============================================================
# Best performing filter detailed analysis
# ============================================================
print("\n" + "=" * 70)
print("DETAILED ANALYSIS - BEST FILTERS")
print("=" * 70)

# Test H4 MFC 0.3 in detail
print("\n--- H4 MFC >= 0.3 by Pair ---")
all_trades = []
for pair, base, quote in ALL_PAIRS:
    trades = backtest_filter(pair, base, quote, filter_type='h4_extreme', params={'threshold': 0.3})
    all_trades.extend(trades)
    if trades:
        df = pd.DataFrame(trades)
        print(f"{pair}: {len(trades):>4} trades, {df['win'].mean()*100:>5.1f}% WR, {df['net_pips'].sum():>+6.0f} pips")

if all_trades:
    df = pd.DataFrame(all_trades)
    print(f"\nTotal: {len(df)} trades, {df['win'].mean()*100:.1f}% WR, {df['net_pips'].sum():+.0f} pips")

    # Best pairs
    pair_stats = df.groupby('pair')['net_pips'].agg(['count', 'sum']).sort_values('sum', ascending=False)
    print("\nTop 5 pairs:")
    for pair in pair_stats.head(5).index:
        print(f"  {pair}: {pair_stats.loc[pair, 'sum']:+.0f} pips")

# Test Divergence 0.5 in detail
print("\n--- MFC Divergence >= 0.5 by Pair ---")
all_trades = []
for pair, base, quote in ALL_PAIRS:
    trades = backtest_filter(pair, base, quote, filter_type='divergence', params={'threshold': 0.5})
    all_trades.extend(trades)
    if trades:
        df = pd.DataFrame(trades)
        print(f"{pair}: {len(trades):>4} trades, {df['win'].mean()*100:>5.1f}% WR, {df['net_pips'].sum():>+6.0f} pips")

if all_trades:
    df = pd.DataFrame(all_trades)
    print(f"\nTotal: {len(df)} trades, {df['win'].mean()*100:.1f}% WR, {df['net_pips'].sum():+.0f} pips")

print(f"\nCompleted: {datetime.now()}")
