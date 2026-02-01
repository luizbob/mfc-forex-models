"""
Test RSI + Median + MFC Momentum Strategy

Strategy:
- SELL: RSI > 70 AND Median > 70, then RSI crosses below median + base MFC going DOWN
- BUY: RSI < 30 AND Median < 30, then RSI crosses above median + base MFC going UP

Both RSI and median must be in extreme zone, then crossover triggers entry.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

DATA_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/data')

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

def calculate_rb_rsi(close, period=14):
    """Royal Black RSI = 100 - Cutler's RSI (SMA-based, inverted)"""
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return 100 - rsi  # Inverted

print("=" * 70)
print("ROYAL BLACK RSI + MEDIAN + MFC STRATEGY")
print("=" * 70)
print(f"Started: {datetime.now()}")

# Load MFC data (H1 timeframe)
print("\nLoading H1 MFC data...")
mfc_h1 = {}
for ccy in CURRENCIES:
    df = pd.read_csv(DATA_DIR / 'cleaned' / f'mfc_currency_{ccy}_H1_clean.csv')
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y.%m.%d %H:%M')
    df = df.set_index('datetime')
    mfc_h1[ccy] = df['MFC']

# Load price data and resample to H1
print("Loading price data...")
price_h1 = {}
for pair, base, quote in ALL_PAIRS:
    try:
        chunks = []
        for chunk in pd.read_csv(DATA_DIR / f'{pair}_GMT+0_US-DST_M1.csv', chunksize=500000):
            chunk['datetime'] = pd.to_datetime(chunk['Date'] + ' ' + chunk['Time'], format='%Y.%m.%d %H:%M:%S')
            chunk = chunk.set_index('datetime')
            h1_chunk = chunk[['Open', 'High', 'Low', 'Close']].resample('1h').agg({
                'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
            }).dropna()
            chunks.append(h1_chunk)

        if chunks:
            price_df = pd.concat(chunks)
            price_df = price_df[~price_df.index.duplicated(keep='first')]
            price_h1[pair] = price_df
    except:
        pass

print(f"Loaded {len(price_h1)} pairs")

# Parameters - now using correct Royal Black RSI formula
RSI_PERIOD = 14  # Fixed - this is the correct formula
MEDIAN_PERIOD = 6
MFC_VEL_PERIODS = [1, 3, 6, 12]  # H1 bars for velocity
EXIT_BARS = [6, 12, 24, 48, 72]  # Hours to hold
RSI_HIGH = 70
RSI_LOW = 30

TEST_YEAR = ('2025-01-01', '2025-12-31')

print(f"\nTest period: {TEST_YEAR[0]} to {TEST_YEAR[1]}")
print(f"RSI: 100 - RSI_SMA({RSI_PERIOD}) [Royal Black formula]")
print(f"Median period: {MEDIAN_PERIOD}")
print(f"RSI thresholds: Overbought > {RSI_HIGH}, Oversold < {RSI_LOW}")

def run_backtest(mfc_vel_period, exit_bars):
    """Run backtest with given parameters"""
    all_trades = []

    for pair, base, quote in ALL_PAIRS:
        if pair not in price_h1:
            continue

        pip_val = get_pip_value(pair)

        try:
            price_df = price_h1[pair].copy()
            price_df = price_df[(price_df.index >= TEST_YEAR[0]) & (price_df.index <= TEST_YEAR[1])]

            if len(price_df) < 100:
                continue

            # Calculate Royal Black RSI and median
            price_df['rsi'] = calculate_rb_rsi(price_df['Close'], period=RSI_PERIOD)
            price_df['rsi_median'] = price_df['rsi'].rolling(window=MEDIAN_PERIOD).median()

            # Previous bar values for crossover detection
            price_df['rsi_prev'] = price_df['rsi'].shift(1)
            price_df['median_prev'] = price_df['rsi_median'].shift(1)

            # Crossover detection
            # RSI crosses below median: was above, now below
            price_df['cross_down'] = (price_df['rsi_prev'] > price_df['median_prev']) & (price_df['rsi'] < price_df['rsi_median'])
            # RSI crosses above median: was below, now above
            price_df['cross_up'] = (price_df['rsi_prev'] < price_df['median_prev']) & (price_df['rsi'] > price_df['rsi_median'])

            # MFC velocity
            base_mfc = mfc_h1[base].shift(1).reindex(price_df.index, method='ffill')
            quote_mfc = mfc_h1[quote].shift(1).reindex(price_df.index, method='ffill')

            price_df['base_mfc_vel'] = base_mfc.diff(periods=mfc_vel_period)
            price_df['quote_mfc_vel'] = quote_mfc.diff(periods=mfc_vel_period)

            price_df = price_df.dropna()

            # SELL signal:
            # - Previous bar: RSI > 70 AND Median > 70 (both overbought)
            # - Current bar: RSI crosses below median
            # - MFC: base going DOWN
            sell_signal = (
                (price_df['rsi_prev'] > RSI_HIGH) &
                (price_df['median_prev'] > RSI_HIGH) &
                price_df['cross_down'] &
                (price_df['base_mfc_vel'] < 0)
            )

            # BUY signal:
            # - Previous bar: RSI < 30 AND Median < 30 (both oversold)
            # - Current bar: RSI crosses above median
            # - MFC: base going UP
            buy_signal = (
                (price_df['rsi_prev'] < RSI_LOW) &
                (price_df['median_prev'] < RSI_LOW) &
                price_df['cross_up'] &
                (price_df['base_mfc_vel'] > 0)
            )

            # Process BUY trades
            for signal_time in price_df.index[buy_signal]:
                idx = price_df.index.get_loc(signal_time)
                if idx + exit_bars >= len(price_df):
                    continue

                entry_price = price_df.iloc[idx + 1]['Open']
                exit_price = price_df.iloc[idx + 1 + exit_bars]['Close']

                pips = (exit_price - entry_price) / pip_val - SPREADS.get(pair, 2.0)

                all_trades.append({
                    'pair': pair,
                    'type': 'BUY',
                    'time': signal_time,
                    'rsi': price_df.loc[signal_time, 'rsi'],
                    'pips': pips,
                    'win': 1 if pips > 0 else 0,
                })

            # Process SELL trades
            for signal_time in price_df.index[sell_signal]:
                idx = price_df.index.get_loc(signal_time)
                if idx + exit_bars >= len(price_df):
                    continue

                entry_price = price_df.iloc[idx + 1]['Open']
                exit_price = price_df.iloc[idx + 1 + exit_bars]['Close']

                pips = (entry_price - exit_price) / pip_val - SPREADS.get(pair, 2.0)

                all_trades.append({
                    'pair': pair,
                    'type': 'SELL',
                    'time': signal_time,
                    'rsi': price_df.loc[signal_time, 'rsi'],
                    'pips': pips,
                    'win': 1 if pips > 0 else 0,
                })

        except Exception as e:
            continue

    return pd.DataFrame(all_trades) if all_trades else None

# Run tests
print("\n" + "=" * 70)
print("RESULTS (RSI + Median both in extreme zone, then crossover + MFC)")
print("=" * 70)

results = []

for mfc_vel_period in MFC_VEL_PERIODS:
    for exit_bars in EXIT_BARS:
        df = run_backtest(mfc_vel_period, exit_bars)

        if df is not None and len(df) > 0:
            trades = len(df)
            buy_df = df[df['type'] == 'BUY']
            sell_df = df[df['type'] == 'SELL']
            win_rate = df['win'].mean() * 100
            avg_pips = df['pips'].mean()
            total_pips = df['pips'].sum()

            results.append({
                'mfc_vel': mfc_vel_period,
                'exit': exit_bars,
                'trades': trades,
                'buys': len(buy_df),
                'sells': len(sell_df),
                'wr': win_rate,
                'buy_wr': buy_df['win'].mean() * 100 if len(buy_df) > 0 else 0,
                'sell_wr': sell_df['win'].mean() * 100 if len(sell_df) > 0 else 0,
                'avg': avg_pips,
                'total': total_pips,
            })

if results:
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('total', ascending=False)

    print(f"\n{'MFC_V':>5} | {'Exit':>4} | {'Trades':>6} | {'Buys':>5} | {'Sells':>5} | {'WR%':>6} | {'BuyWR':>6} | {'SellWR':>6} | {'Avg':>6} | {'Total':>8}")
    print("-" * 85)
    for _, r in results_df.iterrows():
        print(f"{r['mfc_vel']:>5} | {r['exit']:>4} | {r['trades']:>6} | {r['buys']:>5} | {r['sells']:>5} | {r['wr']:>5.1f}% | {r['buy_wr']:>5.1f}% | {r['sell_wr']:>5.1f}% | {r['avg']:>+5.1f} | {r['total']:>+8.0f}")

    # Show best result details
    best = results_df.iloc[0]
    print(f"\n--- Best Configuration ---")
    print(f"MFC velocity: {int(best['mfc_vel'])} H1 bars")
    print(f"Exit after: {int(best['exit'])} hours")
    print(f"Trades: {int(best['trades'])} (Buys: {int(best['buys'])}, Sells: {int(best['sells'])})")
    print(f"Win Rate: {best['wr']:.1f}% (Buy: {best['buy_wr']:.1f}%, Sell: {best['sell_wr']:.1f}%)")
    print(f"Avg Pips: {best['avg']:+.1f}")
    print(f"Total Pips: {best['total']:+.0f}")
else:
    print("\nNo trades found with these parameters.")

print(f"\nCompleted: {datetime.now()}")
