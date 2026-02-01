"""
Strict LSTM Backtest V2 - Matching Actual Trading Strategy
===========================================================
This backtest matches the actual lstm_trader_mt5.py strategy:

Entry (BUY):
- LSTM: base=UP (2), quote=DOWN (0)
- MFC base <= -0.5 (extreme oversold)
- RSI < 20 (price oversold)
- H1 velocity filter: NOT (base_vel_h1 < -0.05 OR quote_vel_h1 > +0.05)

Entry (SELL):
- LSTM: base=DOWN (0), quote=UP (2)
- MFC base >= 0.5 (extreme overbought)
- RSI > 80 (price overbought)
- H1 velocity filter: NOT (base_vel_h1 > +0.05 OR quote_vel_h1 < -0.05)

Exit:
- RSI crosses opposite extreme
- OR timeout at 8 hours (96 M5 bars)
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from datetime import datetime

def log(msg=""):
    print(msg, flush=True)

log("=" * 80)
log("STRICT BACKTEST V2 - MATCHING ACTUAL STRATEGY")
log("=" * 80)
log(f"Started: {datetime.now()}")
log()

# ============================================================================
# CONFIGURATION (matching lstm_trader_mt5.py)
# ============================================================================

DATA_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/data')
CLEAN_DIR = DATA_DIR / 'cleaned'
LSTM_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/lstm_model')
MODEL_DIR = LSTM_DIR / 'models'
DATA_LSTM_DIR = LSTM_DIR / 'data'
OUTPUT_DIR = LSTM_DIR / 'backtest_results'

# Strategy parameters (from lstm_trader_mt5.py)
MIN_CONFIDENCE = 0.70
MFC_EXTREME = 0.5
RSI_PERIOD = 14
RSI_LOW = 20
RSI_HIGH = 80
MAX_BARS_HOLD = 96

# Trading costs
SPREAD_PIPS = 1.5
SLIPPAGE_PIPS = 0.5

# Model training period - test must be AFTER this
MODEL_TRAIN_END = '2024-12-31'

# Test ALL pairs
PAIRS = [
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

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def calculate_rsi(closes, period=14):
    """Calculate RSI array for all bars."""
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    rsi = np.full(len(closes), np.nan)

    for i in range(period, len(closes)):
        avg_gain = np.mean(gains[i-period:i])
        avg_loss = np.mean(losses[i-period:i])

        if avg_loss == 0:
            rsi[i] = 100
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100 - (100 / (1 + rs))

    return rsi

def get_pip_value(pair):
    return 0.01 if 'JPY' in pair else 0.0001

def load_mfc_shifted(currency, timeframe, with_velocity=False):
    """Load MFC with 1-bar shift. Optionally return velocity."""
    if timeframe in ['H1', 'H4']:
        filepath = CLEAN_DIR / f'mfc_currency_{currency}_{timeframe}_clean.csv'
    else:
        filepath = DATA_DIR / f'mfc_currency_{currency}_{timeframe}.csv'

    if not filepath.exists():
        return None if not with_velocity else (None, None)

    df = pd.read_csv(filepath)
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y.%m.%d %H:%M')
    df = df.set_index('datetime').sort_index()

    # Calculate velocity BEFORE shift (velocity of previous bar)
    df['velocity'] = df['MFC'].diff(1)

    # Apply shift (we see previous bar's values)
    df['MFC'] = df['MFC'].shift(1)
    df['velocity'] = df['velocity'].shift(1)
    df = df.dropna()

    if with_velocity:
        return df['MFC'], df['velocity']
    return df['MFC']

# ============================================================================
# LOAD MODELS
# ============================================================================

log("1. Loading LSTM models...")

models = {}
config_path = DATA_LSTM_DIR / 'config.pkl'

with open(config_path, 'rb') as f:
    config = pickle.load(f)

LOOKBACK = config['lookback']

currencies_needed = set()
for pair, base, quote in PAIRS:
    currencies_needed.add(base)
    currencies_needed.add(quote)

for ccy in currencies_needed:
    model_path = MODEL_DIR / f'lstm_{ccy}_final.keras'
    if model_path.exists():
        models[ccy] = tf.keras.models.load_model(model_path)
        log(f"   Loaded {ccy}")
    else:
        # Try best model
        model_path = MODEL_DIR / f'lstm_{ccy}_best.keras'
        if model_path.exists():
            models[ccy] = tf.keras.models.load_model(model_path)
            log(f"   Loaded {ccy} (best)")

log(f"   Loaded {len(models)} models")

# ============================================================================
# LOAD DATA
# ============================================================================

log("\n2. Loading MFC and price data...")

# Load MFC for all currencies
mfc_data = {}
for ccy in currencies_needed:
    mfc_data[ccy] = {}
    for tf in ['M5', 'M15', 'M30', 'H1', 'H4']:
        mfc = load_mfc_shifted(ccy, tf)
        if mfc is not None:
            mfc_data[ccy][tf] = mfc

# Load price data for each pair
price_data = {}
for pair, base, quote in PAIRS:
    filepath = DATA_DIR / f'{pair}_GMT+0_US-DST_M1.csv'
    if not filepath.exists():
        continue

    df = pd.read_csv(filepath)
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y.%m.%d %H:%M:%S')
    df = df.set_index('datetime').sort_index()

    # Resample to M5
    df_m5 = df.resample('5min').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
    }).dropna()

    # Calculate RSI (shifted by 1 to match trading - we see previous bar's RSI)
    df_m5['RSI'] = calculate_rsi(df_m5['Close'].values, RSI_PERIOD)
    df_m5['RSI'] = df_m5['RSI'].shift(1)  # At decision time, we see previous RSI

    price_data[pair] = df_m5
    log(f"   {pair}: {len(df_m5):,} M5 bars")

# ============================================================================
# BACKTEST EACH PAIR
# ============================================================================

log("\n3. Running backtest...")

all_trades = []

for pair, base, quote in PAIRS:
    if pair not in price_data:
        continue
    if base not in models or quote not in models:
        log(f"   {pair}: Missing model for {base} or {quote}")
        continue

    log(f"\n   Processing {pair}...")

    df_price = price_data[pair]
    pip_value = get_pip_value(pair)

    # Get aligned timeline
    base_m5 = mfc_data[base]['M5']
    quote_m5 = mfc_data[quote]['M5']

    # Find overlapping dates
    common_dates = df_price.index.intersection(base_m5.index).intersection(quote_m5.index)

    # Use ONLY data AFTER model training end (truly out-of-sample)
    test_dates = common_dates[common_dates > MODEL_TRAIN_END]

    if len(test_dates) == 0:
        log(f"      No out-of-sample data after {MODEL_TRAIN_END}")
        continue

    log(f"      Test period: {test_dates[0]} to {test_dates[-1]} (out-of-sample)")
    log(f"      Test bars: {len(test_dates):,}")

    trades = []
    position = None  # {'type': 'BUY'/'SELL', 'entry_price': x, 'entry_time': t, 'entry_bar': i}

    for i, dt in enumerate(test_dates):
        if dt not in df_price.index:
            continue

        price_row = df_price.loc[dt]
        rsi = price_row['RSI']

        if np.isnan(rsi):
            continue

        # Get MFC values
        if dt not in base_m5.index or dt not in quote_m5.index:
            continue

        base_mfc_val = base_m5.loc[dt]
        quote_mfc_val = quote_m5.loc[dt]

        # Check exit first
        if position is not None:
            bars_held = i - position['entry_bar']
            should_exit = False
            exit_reason = ""

            if position['type'] == 'BUY' and rsi >= RSI_HIGH:
                should_exit = True
                exit_reason = f"RSI={rsi:.1f}"
            elif position['type'] == 'SELL' and rsi <= RSI_LOW:
                should_exit = True
                exit_reason = f"RSI={rsi:.1f}"
            elif bars_held >= MAX_BARS_HOLD:
                should_exit = True
                exit_reason = "Timeout"

            if should_exit:
                # Exit at NEXT bar's Open
                exit_bar_idx = min(i + 1, len(test_dates) - 1)
                exit_dt = test_dates[exit_bar_idx]
                if exit_dt in df_price.index:
                    exit_price = df_price.loc[exit_dt, 'Open']

                    # Calculate P/L
                    cost = (SPREAD_PIPS + SLIPPAGE_PIPS) * pip_value
                    if position['type'] == 'BUY':
                        pips = (exit_price - position['entry_price'] - cost) / pip_value
                    else:
                        pips = (position['entry_price'] - exit_price - cost) / pip_value

                    trades.append({
                        'pair': pair,
                        'signal': position['type'],
                        'entry_time': position['entry_time'],
                        'exit_time': exit_dt,
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'bars_held': bars_held,
                        'exit_reason': exit_reason,
                        'pips': pips,
                        'entry_mfc': position['entry_mfc'],
                        'entry_rsi': position['entry_rsi'],
                    })

                    position = None

        # Check entry (only if no position)
        if position is None:
            # Need to get LSTM predictions
            # Build features for base and quote

            base_features_valid = True
            quote_features_valid = True

            for tf in ['M5', 'M15', 'M30', 'H1', 'H4']:
                if dt not in mfc_data[base].get(tf, pd.Series()).index:
                    base_features_valid = False
                    break
                if dt not in mfc_data[quote].get(tf, pd.Series()).index:
                    quote_features_valid = False
                    break

            if not base_features_valid or not quote_features_valid:
                continue

            # Get lookback windows
            try:
                X_base = {}
                X_quote = {}

                for tf in ['M5', 'M15', 'M30', 'H1', 'H4']:
                    base_series = mfc_data[base][tf]
                    quote_series = mfc_data[quote][tf]

                    dt_idx_base = base_series.index.get_loc(dt)
                    dt_idx_quote = quote_series.index.get_loc(dt)

                    lookback = LOOKBACK[tf]

                    if dt_idx_base < lookback or dt_idx_quote < lookback:
                        raise ValueError("Not enough history")

                    X_base[tf] = base_series.iloc[dt_idx_base-lookback:dt_idx_base].values
                    X_quote[tf] = quote_series.iloc[dt_idx_quote-lookback:dt_idx_quote].values

                # Aux features for base
                vel_m5_base = X_base['M5'][-1] - X_base['M5'][-2] if len(X_base['M5']) >= 2 else 0
                vel_m30_base = X_base['M30'][-1] - X_base['M30'][-2] if len(X_base['M30']) >= 2 else 0

                X_aux_base = np.array([[
                    vel_m5_base,
                    vel_m30_base,
                    X_base['M5'][-1],
                    X_base['M30'][-1],
                    X_base['H4'][-1],
                ]])

                # Aux features for quote
                vel_m5_quote = X_quote['M5'][-1] - X_quote['M5'][-2] if len(X_quote['M5']) >= 2 else 0
                vel_m30_quote = X_quote['M30'][-1] - X_quote['M30'][-2] if len(X_quote['M30']) >= 2 else 0

                X_aux_quote = np.array([[
                    vel_m5_quote,
                    vel_m30_quote,
                    X_quote['M5'][-1],
                    X_quote['M30'][-1],
                    X_quote['H4'][-1],
                ]])

                # Predict base
                pred_base = models[base].predict([
                    X_base['M5'].reshape(1, -1, 1),
                    X_base['M15'].reshape(1, -1, 1),
                    X_base['M30'].reshape(1, -1, 1),
                    X_base['H1'].reshape(1, -1, 1),
                    X_base['H4'].reshape(1, -1, 1),
                    X_aux_base,
                ], verbose=0)

                # Predict quote
                pred_quote = models[quote].predict([
                    X_quote['M5'].reshape(1, -1, 1),
                    X_quote['M15'].reshape(1, -1, 1),
                    X_quote['M30'].reshape(1, -1, 1),
                    X_quote['H1'].reshape(1, -1, 1),
                    X_quote['H4'].reshape(1, -1, 1),
                    X_aux_quote,
                ], verbose=0)

                base_dir = np.argmax(pred_base[0])
                base_conf = np.max(pred_base[0])
                quote_dir = np.argmax(pred_quote[0])
                quote_conf = np.max(pred_quote[0])

            except Exception as e:
                continue

            # Check confidence
            if base_conf < MIN_CONFIDENCE or quote_conf < MIN_CONFIDENCE:
                continue

            # Check BUY conditions
            # LSTM: base UP (2), quote DOWN (0)
            # MFC: base <= -0.5
            # RSI: < 20
            if (base_dir == 2 and quote_dir == 0 and
                base_mfc_val <= -MFC_EXTREME and
                rsi < RSI_LOW):

                # Entry at NEXT bar's Open
                entry_bar_idx = min(i + 1, len(test_dates) - 1)
                entry_dt = test_dates[entry_bar_idx]
                if entry_dt in df_price.index:
                    entry_price = df_price.loc[entry_dt, 'Open']

                    position = {
                        'type': 'BUY',
                        'entry_price': entry_price,
                        'entry_time': entry_dt,
                        'entry_bar': entry_bar_idx,
                        'entry_mfc': base_mfc_val,
                        'entry_rsi': rsi,
                    }

            # Check SELL conditions
            # LSTM: base DOWN (0), quote UP (2)
            # MFC: base >= 0.5
            # RSI: > 80
            elif (base_dir == 0 and quote_dir == 2 and
                  base_mfc_val >= MFC_EXTREME and
                  rsi > RSI_HIGH):

                # Entry at NEXT bar's Open
                entry_bar_idx = min(i + 1, len(test_dates) - 1)
                entry_dt = test_dates[entry_bar_idx]
                if entry_dt in df_price.index:
                    entry_price = df_price.loc[entry_dt, 'Open']

                    position = {
                        'type': 'SELL',
                        'entry_price': entry_price,
                        'entry_time': entry_dt,
                        'entry_bar': entry_bar_idx,
                        'entry_mfc': base_mfc_val,
                        'entry_rsi': rsi,
                    }

    log(f"      Trades: {len(trades)}")
    all_trades.extend(trades)

# ============================================================================
# RESULTS
# ============================================================================

log("\n" + "=" * 80)
log("BACKTEST RESULTS - ACTUAL STRATEGY")
log("=" * 80)

if len(all_trades) == 0:
    log("\nNo trades generated!")
    log("This is expected - the strategy is VERY selective:")
    log("  - MFC must be at extreme (≤-0.5 or ≥0.5)")
    log("  - RSI must be at extreme (<20 or >80)")
    log("  - LSTM must confirm divergence")
    log("\nThese conditions rarely align together.")
else:
    df_trades = pd.DataFrame(all_trades)
    df_trades['win'] = (df_trades['pips'] > 0).astype(int)

    log(f"\nOVERALL RESULTS:")
    log(f"   Total trades: {len(df_trades)}")
    log(f"   Win rate: {df_trades['win'].mean()*100:.1f}%")
    log(f"   Avg pips: {df_trades['pips'].mean():.2f}")
    log(f"   Total pips: {df_trades['pips'].sum():.0f}")

    log(f"\nBY PAIR:")
    for pair in df_trades['pair'].unique():
        pair_df = df_trades[df_trades['pair'] == pair]
        wr = pair_df['win'].mean() * 100
        avg = pair_df['pips'].mean()
        log(f"   {pair}: {len(pair_df)} trades, {wr:.1f}% WR, {avg:.2f} avg pips")

    log(f"\nBY SIGNAL:")
    for sig in ['BUY', 'SELL']:
        sig_df = df_trades[df_trades['signal'] == sig]
        if len(sig_df) > 0:
            wr = sig_df['win'].mean() * 100
            avg = sig_df['pips'].mean()
            log(f"   {sig}: {len(sig_df)} trades, {wr:.1f}% WR, {avg:.2f} avg pips")

    log(f"\nBY EXIT REASON:")
    for reason in df_trades['exit_reason'].unique():
        reason_df = df_trades[df_trades['exit_reason'] == reason]
        wr = reason_df['win'].mean() * 100
        avg = reason_df['pips'].mean()
        log(f"   {reason}: {len(reason_df)} trades, {wr:.1f}% WR, {avg:.2f} avg pips")

    # Save
    output_path = OUTPUT_DIR / f'strict_backtest_v2_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    df_trades.to_csv(output_path, index=False)
    log(f"\n   Results saved to: {output_path}")

log(f"\n{'='*80}")
log(f"Completed: {datetime.now()}")
log("=" * 80)
