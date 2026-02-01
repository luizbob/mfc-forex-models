"""
Script 13: Final Optimized LSTM Strategy
========================================
Combines all best findings:
1. LSTM divergence (base opposite quote)
2. MFC extreme (0.5) for timing
3. RSI extreme (20/80) for entry
4. RSI-based exit
5. Filter by model accuracy (min 87%)
6. Focus on best performing pairs

This script produces the PRODUCTION-READY strategy parameters.
"""
import warnings
warnings.filterwarnings('ignore')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from pathlib import Path
import gc

def log(msg=""):
    print(msg, flush=True)

log("=" * 70)
log("FINAL OPTIMIZED LSTM STRATEGY")
log("=" * 70)
log(f"Started: {datetime.now()}")

MODEL_ACCURACY = {
    'JPY': 90.0, 'USD': 87.8, 'AUD': 87.4, 'NZD': 86.7,
    'CHF': 86.3, 'GBP': 83.7, 'CAD': 82.2, 'EUR': 82.0,
}

DATA_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/data')
LSTM_DATA_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/lstm_model/data')
MODEL_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/lstm_model/models')

with open(LSTM_DATA_DIR / 'config.pkl', 'rb') as f:
    config = pickle.load(f)

LOOKBACK = config['lookback']
CURRENCIES = config['currencies']

START_DATE = '2024-07-01'
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

# ============================================================================
# OPTIMIZED STRATEGY PARAMETERS
# ============================================================================
STRATEGY_PARAMS = {
    'min_lstm_conf': 0.70,      # Minimum LSTM confidence for both currencies
    'mfc_extreme': 0.5,          # MFC threshold for entry
    'rsi_low': 20,               # RSI oversold level
    'rsi_high': 80,              # RSI overbought level
    'min_model_accuracy': 85.0,  # Minimum model accuracy for pair
}

# Best performing pairs based on analysis
BEST_PAIRS = [
    'USDJPY', 'AUDJPY', 'EURJPY', 'GBPJPY', 'NZDJPY', 'CADJPY', 'CHFJPY',  # JPY pairs
    'AUDUSD', 'GBPUSD', 'USDCHF',  # Other high-accuracy pairs
    'AUDNZD', 'AUDCHF',  # More high-accuracy pairs
]

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
# LOAD DATA
# ============================================================================
log("\n1. Loading data...")

lstm_predictions = {}
for ccy in CURRENCIES:
    with open(LSTM_DATA_DIR / f'lstm_data_{ccy}.pkl', 'rb') as f:
        data = pickle.load(f)

    n_samples = len(data['datetimes'])
    split_idx = int(n_samples * 0.8)

    model = tf.keras.models.load_model(MODEL_DIR / f'lstm_{ccy}_final.keras')

    X_val = [
        data['X_M5'][split_idx:].reshape(-1, LOOKBACK['M5'], 1),
        data['X_M15'][split_idx:].reshape(-1, LOOKBACK['M15'], 1),
        data['X_M30'][split_idx:].reshape(-1, LOOKBACK['M30'], 1),
        data['X_H1'][split_idx:].reshape(-1, LOOKBACK['H1'], 1),
        data['X_H4'][split_idx:].reshape(-1, LOOKBACK['H4'], 1),
        data['X_aux'][split_idx:],
    ]

    pred = model.predict(X_val, verbose=0, batch_size=256)
    datetimes = pd.to_datetime(data['datetimes'][split_idx:])
    lstm_predictions[ccy] = pd.DataFrame({
        'direction': np.argmax(pred[0], axis=1),
        'confidence': np.max(pred[0], axis=1)
    }, index=datetimes)

    del model, data, X_val, pred
    tf.keras.backend.clear_session()
    gc.collect()

log("  LSTM predictions loaded")

mfc_m5 = {}
for cur in CURRENCIES:
    df = pd.read_csv(DATA_DIR / f'mfc_currency_{cur}_M5.csv')
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y.%m.%d %H:%M')
    df = df.set_index('datetime')
    mfc_m5[cur] = df['MFC']

log("  MFC data loaded")

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
    except:
        pass

log("  Price data loaded")

# ============================================================================
# RUN OPTIMIZED STRATEGY
# ============================================================================
log("\n" + "=" * 70)
log("2. RUNNING OPTIMIZED STRATEGY")
log("=" * 70)

def run_optimized_strategy(pairs_filter=None, min_accuracy=None):
    """Run the optimized LSTM strategy."""
    all_trades = []

    for pair, base, quote in ALL_PAIRS:
        if pair not in price_data:
            continue

        # Filter by allowed pairs
        if pairs_filter and pair not in pairs_filter:
            continue

        # Filter by model accuracy
        if min_accuracy:
            if min(MODEL_ACCURACY[base], MODEL_ACCURACY[quote]) < min_accuracy:
                continue

        pip_val = get_pip_value(pair)

        try:
            price_df = price_data[pair].copy()

            price_df['base_mfc'] = mfc_m5[base].shift(1).reindex(price_df.index, method='ffill')
            price_df['quote_mfc'] = mfc_m5[quote].shift(1).reindex(price_df.index, method='ffill')

            base_lstm = lstm_predictions[base].reindex(price_df.index, method='ffill')
            quote_lstm = lstm_predictions[quote].reindex(price_df.index, method='ffill')

            price_df['base_dir'] = base_lstm['direction']
            price_df['base_conf'] = base_lstm['confidence']
            price_df['quote_dir'] = quote_lstm['direction']
            price_df['quote_conf'] = quote_lstm['confidence']

            rsi = price_df['rsi']

            # LSTM divergence + MFC extreme + RSI extreme
            buy_signal = (
                (price_df['base_dir'] == 2) &  # Base UP
                (price_df['quote_dir'] == 0) &  # Quote DOWN
                (price_df['base_conf'] >= STRATEGY_PARAMS['min_lstm_conf']) &
                (price_df['quote_conf'] >= STRATEGY_PARAMS['min_lstm_conf']) &
                (price_df['base_mfc'] <= -STRATEGY_PARAMS['mfc_extreme']) &
                (rsi < STRATEGY_PARAMS['rsi_low'])
            )

            sell_signal = (
                (price_df['base_dir'] == 0) &  # Base DOWN
                (price_df['quote_dir'] == 2) &  # Quote UP
                (price_df['base_conf'] >= STRATEGY_PARAMS['min_lstm_conf']) &
                (price_df['quote_conf'] >= STRATEGY_PARAMS['min_lstm_conf']) &
                (price_df['base_mfc'] >= STRATEGY_PARAMS['mfc_extreme']) &
                (rsi > STRATEGY_PARAMS['rsi_high'])
            )

            # Process BUY
            buy_indices = price_df.index[buy_signal].tolist()
            i = 0
            while i < len(buy_indices):
                signal_time = buy_indices[i]
                signal_idx = price_df.index.get_loc(signal_time)

                entry_price = price_df.loc[signal_time, 'Close']
                future_df = price_df.iloc[signal_idx+1:signal_idx+201]
                exit_mask = future_df['rsi'] >= STRATEGY_PARAMS['rsi_high']

                if exit_mask.any():
                    exit_time = future_df.index[exit_mask.argmax()]
                    exit_price = price_df.loc[exit_time, 'Close']
                    pips = (exit_price - entry_price) / pip_val

                    all_trades.append({
                        'pair': pair, 'type': 'BUY', 'entry_time': signal_time,
                        'exit_time': exit_time,
                        'pips': pips, 'win': 1 if pips > 0 else 0,
                        'spread': SPREADS.get(pair, 2.0),
                        'base_conf': price_df.loc[signal_time, 'base_conf'],
                        'quote_conf': price_df.loc[signal_time, 'quote_conf'],
                    })

                    while i < len(buy_indices) and buy_indices[i] <= exit_time:
                        i += 1
                else:
                    i += 1

            # Process SELL
            sell_indices = price_df.index[sell_signal].tolist()
            i = 0
            while i < len(sell_indices):
                signal_time = sell_indices[i]
                signal_idx = price_df.index.get_loc(signal_time)

                entry_price = price_df.loc[signal_time, 'Close']
                future_df = price_df.iloc[signal_idx+1:signal_idx+201]
                exit_mask = future_df['rsi'] <= STRATEGY_PARAMS['rsi_low']

                if exit_mask.any():
                    exit_time = future_df.index[exit_mask.argmax()]
                    exit_price = price_df.loc[exit_time, 'Close']
                    pips = (entry_price - exit_price) / pip_val

                    all_trades.append({
                        'pair': pair, 'type': 'SELL', 'entry_time': signal_time,
                        'exit_time': exit_time,
                        'pips': pips, 'win': 1 if pips > 0 else 0,
                        'spread': SPREADS.get(pair, 2.0),
                        'base_conf': price_df.loc[signal_time, 'base_conf'],
                        'quote_conf': price_df.loc[signal_time, 'quote_conf'],
                    })

                    while i < len(sell_indices) and sell_indices[i] <= exit_time:
                        i += 1
                else:
                    i += 1

        except:
            pass

    return pd.DataFrame(all_trades)

def print_results(label, trades):
    if len(trades) == 0:
        log(f"  {label}: No trades")
        return

    trades['net_pips'] = trades['pips'] - trades['spread']
    wr = trades['win'].mean() * 100
    net_avg = trades['net_pips'].mean()
    total = trades['net_pips'].sum()
    log(f"  {label}:")
    log(f"    Trades: {len(trades)}")
    log(f"    Win Rate: {wr:.1f}%")
    log(f"    Net Avg: {net_avg:.2f} pips")
    log(f"    Total Net: {total:.0f} pips")

# Run different configurations
log("\n--- Configuration Comparison ---\n")

# All pairs
trades_all = run_optimized_strategy()
print_results("All 28 pairs", trades_all)

# High accuracy pairs only
trades_acc = run_optimized_strategy(min_accuracy=85)
print_results("High accuracy (min 85%)", trades_acc)

# Best pairs only
trades_best = run_optimized_strategy(pairs_filter=BEST_PAIRS)
print_results("Best pairs only", trades_best)

# JPY pairs only
jpy_pairs = [p[0] for p in ALL_PAIRS if 'JPY' in p[0]]
trades_jpy = run_optimized_strategy(pairs_filter=jpy_pairs)
print_results("JPY pairs only", trades_jpy)

# ============================================================================
# FINAL RECOMMENDATION
# ============================================================================
log("\n" + "=" * 70)
log("3. FINAL RECOMMENDATION")
log("=" * 70)

log("\n--- STRATEGY PARAMETERS FOR PRODUCTION ---")
log(f"""
ENTRY CONDITIONS:
  1. LSTM Divergence:
     - BUY: Base currency predicted UP, Quote currency predicted DOWN
     - SELL: Base currency predicted DOWN, Quote currency predicted UP
  2. Minimum LSTM confidence: {STRATEGY_PARAMS['min_lstm_conf']:.0%}
  3. MFC extreme threshold: {STRATEGY_PARAMS['mfc_extreme']}
  4. RSI extreme: < {STRATEGY_PARAMS['rsi_low']} for BUY, > {STRATEGY_PARAMS['rsi_high']} for SELL

EXIT CONDITIONS:
  - BUY: Exit when RSI >= {STRATEGY_PARAMS['rsi_high']}
  - SELL: Exit when RSI <= {STRATEGY_PARAMS['rsi_low']}
  - Max holding: 200 bars (~16.7 hours)

PAIR FILTER:
  - Minimum model accuracy: {STRATEGY_PARAMS['min_model_accuracy']}%
  - Or use best pairs: {', '.join(BEST_PAIRS[:7])}
""")

# Save best trades for analysis
if len(trades_acc) > 0:
    trades_acc.to_csv(LSTM_DATA_DIR / 'optimized_trades.csv', index=False)
    log(f"\nTrades saved to optimized_trades.csv")

log(f"\n" + "=" * 70)
log("COMPARISON WITH BASELINES")
log("=" * 70)

log(f"""
| Strategy                    | Trades | Win Rate | Net Avg | Total    |
|----------------------------|--------|----------|---------|----------|
| V1.5 Corrected             | 115    | 73.9%    | +4.96   | +571     |
| V1.5 + LSTM Full Div (0.8) | 18     | 77.8%    | +6.56   | +118     |
| LSTM Optimized (all pairs) | {len(trades_all):<6} | {trades_all['win'].mean()*100:.1f}%    | +{(trades_all['pips']-trades_all['spread']).mean():.2f}   | +{(trades_all['pips']-trades_all['spread']).sum():.0f}     |
| LSTM Optimized (acc>=85%)  | {len(trades_acc):<6} | {trades_acc['win'].mean()*100:.1f}%    | +{(trades_acc['pips']-trades_acc['spread']).mean():.2f}   | +{(trades_acc['pips']-trades_acc['spread']).sum():.0f}     |
""")

log(f"\nCompleted: {datetime.now()}")
