"""
Test RSI + Median Strategy WITHOUT MFC filter

Strategy:
- SELL: RSI > 70 AND Median > 70, then RSI crosses below median
- BUY: RSI < 30 AND Median < 30, then RSI crosses above median
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

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
print("ROYAL BLACK RSI (100 - RSI_SMA_14) + MEDIAN(6) STRATEGY")
print("=" * 70)
print(f"Started: {datetime.now()}")

# Load price data
print("\nLoading H1 price data...")
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

# Parameters - now we know the exact formula!
RSI_PERIOD = 14  # Cutler's RSI period
MEDIAN_PERIOD = 6
EXIT_BARS = [6, 12, 24, 48, 72]
RSI_HIGH = 70
RSI_LOW = 30

TEST_YEAR = ('2025-01-01', '2025-12-31')

print(f"\nTest period: {TEST_YEAR[0]} to {TEST_YEAR[1]}")
print(f"RSI: 100 - RSI_SMA({RSI_PERIOD})")
print(f"Median period: {MEDIAN_PERIOD}")
print(f"Exit bars: {EXIT_BARS}")

def run_backtest(exit_bars):
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

            price_df['rsi'] = calculate_rb_rsi(price_df['Close'], period=RSI_PERIOD)
            price_df['rsi_median'] = price_df['rsi'].rolling(window=MEDIAN_PERIOD).median()

            price_df['rsi_prev'] = price_df['rsi'].shift(1)
            price_df['median_prev'] = price_df['rsi_median'].shift(1)

            # Crossover: RSI crosses below median
            price_df['cross_down'] = (price_df['rsi_prev'] > price_df['median_prev']) & (price_df['rsi'] < price_df['rsi_median'])
            # Crossover: RSI crosses above median
            price_df['cross_up'] = (price_df['rsi_prev'] < price_df['median_prev']) & (price_df['rsi'] > price_df['rsi_median'])

            price_df = price_df.dropna()

            # SELL: Both RSI and Median were > 70, then crossover down
            sell_signal = (
                (price_df['rsi_prev'] > RSI_HIGH) &
                (price_df['median_prev'] > RSI_HIGH) &
                price_df['cross_down']
            )

            # BUY: Both RSI and Median were < 30, then crossover up
            buy_signal = (
                (price_df['rsi_prev'] < RSI_LOW) &
                (price_df['median_prev'] < RSI_LOW) &
                price_df['cross_up']
            )

            for signal_time in price_df.index[buy_signal]:
                idx = price_df.index.get_loc(signal_time)
                if idx + exit_bars >= len(price_df):
                    continue

                entry_price = price_df.iloc[idx + 1]['Open']
                exit_price = price_df.iloc[idx + 1 + exit_bars]['Close']
                pips = (exit_price - entry_price) / pip_val - SPREADS.get(pair, 2.0)

                all_trades.append({
                    'pair': pair, 'type': 'BUY', 'time': signal_time,
                    'pips': pips, 'win': 1 if pips > 0 else 0,
                })

            for signal_time in price_df.index[sell_signal]:
                idx = price_df.index.get_loc(signal_time)
                if idx + exit_bars >= len(price_df):
                    continue

                entry_price = price_df.iloc[idx + 1]['Open']
                exit_price = price_df.iloc[idx + 1 + exit_bars]['Close']
                pips = (entry_price - exit_price) / pip_val - SPREADS.get(pair, 2.0)

                all_trades.append({
                    'pair': pair, 'type': 'SELL', 'time': signal_time,
                    'pips': pips, 'win': 1 if pips > 0 else 0,
                })

        except:
            continue

    return pd.DataFrame(all_trades) if all_trades else None

print("\n" + "=" * 70)
print("RESULTS (RSI + Median only, NO MFC)")
print("=" * 70)

results = []

for exit_bars in EXIT_BARS:
    df = run_backtest(exit_bars)

    if df is not None and len(df) > 0:
        buy_df = df[df['type'] == 'BUY']
        sell_df = df[df['type'] == 'SELL']

        results.append({
            'exit': exit_bars,
            'trades': len(df),
            'buys': len(buy_df),
            'sells': len(sell_df),
            'wr': df['win'].mean() * 100,
            'buy_wr': buy_df['win'].mean() * 100 if len(buy_df) > 0 else 0,
            'sell_wr': sell_df['win'].mean() * 100 if len(sell_df) > 0 else 0,
            'avg': df['pips'].mean(),
            'total': df['pips'].sum(),
        })

if results:
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('total', ascending=False)

    print(f"\n{'Exit':>4} | {'Trades':>6} | {'Buys':>5} | {'Sells':>5} | {'WR%':>6} | {'BuyWR':>6} | {'SellWR':>6} | {'Avg':>6} | {'Total':>8}")
    print("-" * 75)
    for _, r in results_df.iterrows():
        print(f"{r['exit']:>4} | {r['trades']:>6} | {r['buys']:>5} | {r['sells']:>5} | {r['wr']:>5.1f}% | {r['buy_wr']:>5.1f}% | {r['sell_wr']:>5.1f}% | {r['avg']:>+5.1f} | {r['total']:>+8.0f}")

    best = results_df.iloc[0]
    print(f"\n--- Best: Exit {int(best['exit'])}h ---")
    print(f"Trades: {int(best['trades'])}, WR: {best['wr']:.1f}%, Avg: {best['avg']:+.1f}, Total: {best['total']:+.0f}")

print(f"\nCompleted: {datetime.now()}")
