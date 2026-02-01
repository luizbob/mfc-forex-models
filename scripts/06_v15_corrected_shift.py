"""
Script 06: V1.5 Backtest with Corrected Shifts
==============================================
Fixes lookahead bias in H4 and M30 data.

The issue:
- M5 was shifted correctly: shift(1) after reindex
- H4 and M30 were NOT shifted - using current bar that hasn't closed yet

The fix:
- Shift H4 and M30 by 1 bar BEFORE reindexing to M5 timeline
- This ensures we only use COMPLETED bars
"""
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

def log(msg=""):
    print(msg, flush=True)

log("=" * 70)
log("V1.5 WITH CORRECTED SHIFTS")
log("=" * 70)
log(f"Started: {datetime.now()}")

DATA_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/data')
START_DATE = '2023-01-01'
END_DATE = '2024-12-31'

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

RSI_LOW = 20
RSI_HIGH = 80
RSI_MEDIAN_PERIOD = 9
MFC_THRESHOLD = 0.5
QUOTE_THRESHOLD = 0.3

VELOCITY_PAIRS = [
    'GBPCHF', 'NZDUSD', 'CADCHF', 'USDCAD', 'AUDCHF', 'EURCAD', 'GBPNZD',
    'USDCHF', 'GBPAUD', 'EURGBP', 'EURCHF', 'AUDUSD', 'EURJPY', 'GBPUSD'
]

# MTF filter thresholds
SELL_BASE_VEL_M30_MAX = 0.10
BUY_QUOTE_H4_MAX = 0.10

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def get_pip_value(pair):
    return 0.01 if 'JPY' in pair else 0.0001

# ============================================================================
# 1. LOAD MFC DATA WITH CORRECT SHIFTS
# ============================================================================
log("\n1. Loading MFC data...")

mfc_m5 = {}
mfc_m30 = {}
mfc_m30_shifted = {}  # Shifted by 1 M30 bar
mfc_h4 = {}
mfc_h4_shifted = {}   # Shifted by 1 H4 bar

for cur in ['EUR', 'USD', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD', 'NZD']:
    # M5
    df = pd.read_csv(DATA_DIR / f'mfc_currency_{cur}_M5.csv')
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y.%m.%d %H:%M')
    df = df.set_index('datetime')
    mfc_m5[cur] = df['MFC']

    # M30
    df = pd.read_csv(DATA_DIR / f'mfc_currency_{cur}_M30.csv')
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y.%m.%d %H:%M')
    df = df.set_index('datetime')
    mfc_m30[cur] = df['MFC']
    mfc_m30_shifted[cur] = df['MFC'].shift(1)  # SHIFT BY 1 M30 BAR

    # H4 (cleaned)
    df = pd.read_csv(DATA_DIR / 'cleaned' / f'mfc_currency_{cur}_H4_clean.csv')
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y.%m.%d %H:%M')
    df = df.set_index('datetime')
    mfc_h4[cur] = df['MFC']
    mfc_h4_shifted[cur] = df['MFC'].shift(1)  # SHIFT BY 1 H4 BAR

log("  Loaded M5, M30 (shifted), H4 (shifted)")

# ============================================================================
# 2. LOAD PRICE DATA
# ============================================================================
log("\n2. Loading price data...")

price_data = {}

for pair, base, quote in ALL_PAIRS:
    try:
        chunks = []
        for chunk in pd.read_csv(DATA_DIR / f'{pair}_GMT+0_US-DST_M1.csv', chunksize=500000):
            chunk['datetime'] = pd.to_datetime(chunk['Date'] + ' ' + chunk['Time'], format='%Y.%m.%d %H:%M:%S')
            chunk = chunk.set_index('datetime')
            chunk = chunk[(chunk.index >= START_DATE) & (chunk.index <= END_DATE)]
            if len(chunk) > 0:
                m5_chunk = chunk[['Open', 'High', 'Low', 'Close']].resample('5min').agg({
                    'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
                }).dropna()
                chunks.append(m5_chunk)

        if chunks:
            price_m5 = pd.concat(chunks)
            price_m5 = price_m5[~price_m5.index.duplicated(keep='first')]
            price_m5['rsi'] = calculate_rsi(price_m5['Close'], period=14)
            price_data[pair] = price_m5
            log(f"  {pair}: {len(price_m5)} bars")
    except Exception as e:
        log(f"  {pair}: ERROR - {e}")

log(f"  Loaded {len(price_data)} pairs")

# ============================================================================
# 3. RUN BACKTEST - ORIGINAL (with lookahead)
# ============================================================================
log("\n" + "=" * 70)
log("3. ORIGINAL V1.5 (with lookahead in H4/M30)")
log("=" * 70)

def run_backtest(use_shifted_mtf=False):
    """Run V1.5 backtest with optional corrected shifts."""
    all_trades = []

    for pair, base, quote in ALL_PAIRS:
        if pair not in price_data:
            continue

        use_base_vel = pair in VELOCITY_PAIRS
        pip_val = get_pip_value(pair)

        try:
            price_df = price_data[pair].copy()

            # M5 MFC - always shifted by 1 M5 bar (correct)
            price_df['base_mfc'] = mfc_m5[base].shift(1).reindex(price_df.index, method='ffill')
            price_df['quote_mfc'] = mfc_m5[quote].shift(1).reindex(price_df.index, method='ffill')

            # M30 for velocity - choose shifted or not
            if use_shifted_mtf:
                m30_base = mfc_m30_shifted[base].reindex(price_df.index, method='ffill')
                m30_base_prev = mfc_m30_shifted[base].shift(1).reindex(price_df.index, method='ffill')
            else:
                m30_base = mfc_m30[base].reindex(price_df.index, method='ffill')
                m30_base_prev = mfc_m30[base].shift(1).reindex(price_df.index, method='ffill')

            base_vel_m30 = m30_base - m30_base_prev

            # H4 for quote filter - choose shifted or not
            if use_shifted_mtf:
                quote_h4 = mfc_h4_shifted[quote].reindex(price_df.index, method='ffill')
            else:
                quote_h4 = mfc_h4[quote].reindex(price_df.index, method='ffill')

            # M5 velocities
            base_vel_m5 = price_df['base_mfc'] - price_df['base_mfc'].shift(1)
            quote_vel_m5 = price_df['quote_mfc'] - price_df['quote_mfc'].shift(1)

            rsi_series = price_df['rsi']
            rsi_med = rsi_series.rolling(RSI_MEDIAN_PERIOD).median()

            # BUY signals
            buy_signal = (
                (rsi_series < RSI_LOW) & (rsi_med < RSI_LOW) &
                (price_df['base_mfc'] <= -MFC_THRESHOLD) &
                (price_df['quote_mfc'] <= QUOTE_THRESHOLD) &
                (quote_vel_m5 < 0) &
                (quote_h4 <= BUY_QUOTE_H4_MAX)  # MTF filter integrated
            )
            if use_base_vel:
                buy_signal = buy_signal & (base_vel_m5 > 0)

            # SELL signals
            sell_signal = (
                (rsi_series > RSI_HIGH) & (rsi_med > RSI_HIGH) &
                (price_df['base_mfc'] >= MFC_THRESHOLD) &
                (price_df['quote_mfc'] >= -QUOTE_THRESHOLD) &
                (quote_vel_m5 > 0) &
                (base_vel_m30 <= SELL_BASE_VEL_M30_MAX)  # MTF filter integrated
            )
            if use_base_vel:
                sell_signal = sell_signal & (base_vel_m5 < 0)

            # Process BUY signals
            buy_indices = price_df.index[buy_signal].tolist()
            i = 0
            while i < len(buy_indices):
                signal_time = buy_indices[i]
                signal_idx = price_df.index.get_loc(signal_time)
                entry_price = price_df.loc[signal_time, 'Close']

                future_df = price_df.iloc[signal_idx+1:signal_idx+201]
                exit_mask = future_df['rsi'] >= RSI_HIGH

                if exit_mask.any():
                    exit_time = future_df.index[exit_mask.argmax()]
                    exit_price = price_df.loc[exit_time, 'Close']
                    pips = (exit_price - entry_price) / pip_val

                    all_trades.append({
                        'pair': pair, 'type': 'BUY', 'entry_time': signal_time,
                        'pips': pips, 'win': 1 if pips > 0 else 0,
                        'spread': SPREADS.get(pair, 2.0)
                    })

                    while i < len(buy_indices) and buy_indices[i] <= exit_time:
                        i += 1
                else:
                    i += 1

            # Process SELL signals
            sell_indices = price_df.index[sell_signal].tolist()
            i = 0
            while i < len(sell_indices):
                signal_time = sell_indices[i]
                signal_idx = price_df.index.get_loc(signal_time)
                entry_price = price_df.loc[signal_time, 'Close']

                future_df = price_df.iloc[signal_idx+1:signal_idx+201]
                exit_mask = future_df['rsi'] <= RSI_LOW

                if exit_mask.any():
                    exit_time = future_df.index[exit_mask.argmax()]
                    exit_price = price_df.loc[exit_time, 'Close']
                    pips = (entry_price - exit_price) / pip_val

                    all_trades.append({
                        'pair': pair, 'type': 'SELL', 'entry_time': signal_time,
                        'pips': pips, 'win': 1 if pips > 0 else 0,
                        'spread': SPREADS.get(pair, 2.0)
                    })

                    while i < len(sell_indices) and sell_indices[i] <= exit_time:
                        i += 1
                else:
                    i += 1

        except Exception as e:
            log(f"  {pair}: Error - {e}")

    return pd.DataFrame(all_trades)


# Run original (with lookahead)
trades_original = run_backtest(use_shifted_mtf=False)

if len(trades_original) > 0:
    trades_original['net_pips'] = trades_original['pips'] - trades_original['spread']
    wr = trades_original['win'].mean() * 100
    avg_pips = trades_original['pips'].mean()
    avg_net = trades_original['net_pips'].mean()
    total_net = trades_original['net_pips'].sum()

    log(f"\n  Trades: {len(trades_original)}")
    log(f"  Win Rate: {wr:.1f}%")
    log(f"  Avg Pips: {avg_pips:.2f}")
    log(f"  Avg Net: {avg_net:.2f}")
    log(f"  Total Net: {total_net:.0f}")

# ============================================================================
# 4. RUN BACKTEST - CORRECTED (no lookahead)
# ============================================================================
log("\n" + "=" * 70)
log("4. CORRECTED V1.5 (proper shifts on H4/M30)")
log("=" * 70)

trades_corrected = run_backtest(use_shifted_mtf=True)

if len(trades_corrected) > 0:
    trades_corrected['net_pips'] = trades_corrected['pips'] - trades_corrected['spread']
    wr = trades_corrected['win'].mean() * 100
    avg_pips = trades_corrected['pips'].mean()
    avg_net = trades_corrected['net_pips'].mean()
    total_net = trades_corrected['net_pips'].sum()

    log(f"\n  Trades: {len(trades_corrected)}")
    log(f"  Win Rate: {wr:.1f}%")
    log(f"  Avg Pips: {avg_pips:.2f}")
    log(f"  Avg Net: {avg_net:.2f}")
    log(f"  Total Net: {total_net:.0f}")

# ============================================================================
# 5. COMPARISON
# ============================================================================
log("\n" + "=" * 70)
log("5. COMPARISON")
log("=" * 70)

if len(trades_original) > 0 and len(trades_corrected) > 0:
    log(f"\n{'Metric':<20} {'Original':<15} {'Corrected':<15} {'Diff':<10}")
    log("-" * 60)

    orig_wr = trades_original['win'].mean() * 100
    corr_wr = trades_corrected['win'].mean() * 100
    log(f"{'Trades':<20} {len(trades_original):<15} {len(trades_corrected):<15} {len(trades_corrected) - len(trades_original):<10}")
    log(f"{'Win Rate %':<20} {orig_wr:<15.1f} {corr_wr:<15.1f} {corr_wr - orig_wr:<10.1f}")

    orig_net = trades_original['net_pips'].mean()
    corr_net = trades_corrected['net_pips'].mean()
    log(f"{'Avg Net Pips':<20} {orig_net:<15.2f} {corr_net:<15.2f} {corr_net - orig_net:<10.2f}")

    orig_total = trades_original['net_pips'].sum()
    corr_total = trades_corrected['net_pips'].sum()
    log(f"{'Total Net Pips':<20} {orig_total:<15.0f} {corr_total:<15.0f} {corr_total - orig_total:<10.0f}")

log(f"\nCompleted: {datetime.now()}")
