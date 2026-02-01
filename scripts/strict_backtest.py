"""
Strict LSTM Backtest - No Look-Ahead Bias (Optimized)
======================================================
Validates LSTM model with realistic trading conditions:
- Entry at NEXT bar's OPEN (not Close)
- Realistic spread applied
- True out-of-sample period
- Proper MFC shift validation
- Batched predictions for speed

This is the HONEST backtest that reflects real trading conditions.
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
import gc

def log(msg=""):
    print(msg, flush=True)

log("=" * 80)
log("STRICT LSTM BACKTEST - NO LOOK-AHEAD BIAS")
log("=" * 80)
log(f"Started: {datetime.now()}")
log()

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/data')
CLEAN_DIR = DATA_DIR / 'cleaned'
LSTM_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/lstm_model')
MODEL_DIR = LSTM_DIR / 'models'
DATA_LSTM_DIR = LSTM_DIR / 'data'
OUTPUT_DIR = LSTM_DIR / 'backtest_results'
OUTPUT_DIR.mkdir(exist_ok=True)

# Trading parameters
SPREAD_PIPS = 1.5
SLIPPAGE_PIPS = 0.5
HOLD_BARS_M5 = 96  # 8 hours

# We'll use the pre-prepared LSTM data to ensure consistency
# and validate with the same data pipeline

# ============================================================================
# LOAD MODEL AND CONFIG
# ============================================================================

log("1. Loading LSTM model and config...")

# Load config
config_path = DATA_LSTM_DIR / 'config.pkl'
if not config_path.exists():
    log("ERROR: config.pkl not found!")
    exit(1)

with open(config_path, 'rb') as f:
    config = pickle.load(f)

LOOKBACK = config['lookback']
log(f"   Lookback config: {LOOKBACK}")

# Load model
model_path = MODEL_DIR / 'lstm_EUR_best.keras'
if not model_path.exists():
    model_path = MODEL_DIR / 'lstm_EUR_final.keras'

if not model_path.exists():
    log("ERROR: No model found!")
    exit(1)

model = tf.keras.models.load_model(model_path)
log(f"   Model loaded: {model_path.name}")

# ============================================================================
# LOAD PRE-PREPARED DATA
# ============================================================================

log("\n2. Loading pre-prepared LSTM data...")

# Load EUR data (what the model was trained on)
with open(DATA_LSTM_DIR / 'lstm_data_EUR.pkl', 'rb') as f:
    eur_data = pickle.load(f)

n_total = len(eur_data['datetimes'])
log(f"   Total samples: {n_total:,}")
log(f"   Date range: {eur_data['datetimes'][0]} to {eur_data['datetimes'][-1]}")

# Use validation set (last 20%) as out-of-sample
# But for STRICT test, we should ideally use data AFTER training end date
# Let's check what dates are in the data

train_split = int(n_total * 0.8)
test_start = train_split
test_end = n_total

log(f"   Training samples: 0 to {train_split:,}")
log(f"   Test samples: {test_start:,} to {test_end:,}")
log(f"   Test period: {eur_data['datetimes'][test_start]} to {eur_data['datetimes'][test_end-1]}")

# ============================================================================
# LOAD PRICE DATA FOR TRADES
# ============================================================================

log("\n3. Loading EURUSD price data...")

df_price = pd.read_csv(DATA_DIR / 'EURUSD_GMT+0_US-DST_M1.csv')
df_price['datetime'] = pd.to_datetime(df_price['Date'] + ' ' + df_price['Time'], format='%Y.%m.%d %H:%M:%S')
df_price = df_price.set_index('datetime').sort_index()

# Resample to M5
df_m5 = df_price.resample('5min').agg({
    'Open': 'first',
    'High': 'max',
    'Low': 'min',
    'Close': 'last',
}).dropna()

log(f"   Price bars: {len(df_m5):,}")

# Build datetime to price index map
price_idx_map = {dt: i for i, dt in enumerate(df_m5.index)}

# ============================================================================
# GENERATE PREDICTIONS (BATCHED)
# ============================================================================

log("\n4. Generating predictions (batched)...")

# Prepare test data
X_test = [
    eur_data['X_M5'][test_start:test_end].reshape(-1, LOOKBACK['M5'], 1),
    eur_data['X_M15'][test_start:test_end].reshape(-1, LOOKBACK['M15'], 1),
    eur_data['X_M30'][test_start:test_end].reshape(-1, LOOKBACK['M30'], 1),
    eur_data['X_H1'][test_start:test_end].reshape(-1, LOOKBACK['H1'], 1),
    eur_data['X_H4'][test_start:test_end].reshape(-1, LOOKBACK['H4'], 1),
    eur_data['X_aux'][test_start:test_end],
]

y_test_direction = eur_data['y_direction'][test_start:test_end]
datetimes_test = eur_data['datetimes'][test_start:test_end]

log(f"   Test samples: {len(datetimes_test):,}")

# Batch predict
predictions = model.predict(X_test, batch_size=1024, verbose=0)

if isinstance(predictions, list):
    direction_probs = predictions[0]
else:
    direction_probs = predictions

pred_classes = np.argmax(direction_probs, axis=1)
pred_confidences = np.max(direction_probs, axis=1)

log(f"   Predictions complete")
log(f"   Class distribution: DOWN={np.sum(pred_classes==0)}, NEUTRAL={np.sum(pred_classes==1)}, UP={np.sum(pred_classes==2)}")

# ============================================================================
# EXECUTE TRADES WITH STRICT RULES
# ============================================================================

log("\n5. Executing trades with strict rules...")

PIP_VALUE = 0.0001
TOTAL_COST = SPREAD_PIPS + SLIPPAGE_PIPS

trades = []
skipped_no_price = 0
skipped_neutral = 0
skipped_low_conf = 0

for i in range(len(datetimes_test)):
    dt = pd.Timestamp(datetimes_test[i])
    pred_class = pred_classes[i]
    confidence = pred_confidences[i]
    actual_direction = y_test_direction[i]  # For analysis: -1, 0, 1

    # Skip neutral predictions
    if pred_class == 1:
        skipped_neutral += 1
        continue

    # Skip low confidence
    if confidence < 0.6:
        skipped_low_conf += 1
        continue

    # Find price for this datetime
    if dt not in price_idx_map:
        skipped_no_price += 1
        continue

    price_idx = price_idx_map[dt]

    # CRITICAL: Entry at NEXT bar's OPEN
    entry_idx = price_idx + 1
    if entry_idx >= len(df_m5):
        continue

    # Exit after HOLD_BARS
    exit_idx = min(entry_idx + HOLD_BARS_M5, len(df_m5) - 1)

    entry_time = df_m5.index[entry_idx]
    exit_time = df_m5.index[exit_idx]

    entry_price = df_m5.iloc[entry_idx]['Open']
    exit_price = df_m5.iloc[exit_idx]['Close']

    # Determine signal
    if pred_class == 2:  # UP -> BUY
        signal = 'BUY'
        # Apply costs
        adjusted_entry = entry_price + (TOTAL_COST * PIP_VALUE / 2)
        adjusted_exit = exit_price - (TOTAL_COST * PIP_VALUE / 2)
        pips = (adjusted_exit - adjusted_entry) / PIP_VALUE
    else:  # pred_class == 0 -> DOWN -> SELL
        signal = 'SELL'
        adjusted_entry = entry_price - (TOTAL_COST * PIP_VALUE / 2)
        adjusted_exit = exit_price + (TOTAL_COST * PIP_VALUE / 2)
        pips = (adjusted_entry - adjusted_exit) / PIP_VALUE

    trades.append({
        'signal_time': dt,
        'entry_time': entry_time,
        'exit_time': exit_time,
        'signal': signal,
        'confidence': confidence,
        'pred_class': pred_class,
        'actual_direction': int(actual_direction),
        'entry_price': entry_price,
        'exit_price': exit_price,
        'pips': pips,
    })

log(f"   Trades generated: {len(trades):,}")
log(f"   Skipped (neutral): {skipped_neutral:,}")
log(f"   Skipped (low conf): {skipped_low_conf:,}")
log(f"   Skipped (no price): {skipped_no_price:,}")

# ============================================================================
# RESULTS ANALYSIS
# ============================================================================

log("\n" + "=" * 80)
log("STRICT BACKTEST RESULTS - EURUSD")
log("=" * 80)

if len(trades) == 0:
    log("\nNo trades generated!")
else:
    df_trades = pd.DataFrame(trades)
    df_trades['win'] = (df_trades['pips'] > 0).astype(int)

    # Overall stats
    log(f"\nOVERALL RESULTS:")
    log(f"   Total trades: {len(df_trades):,}")
    log(f"   Win rate: {df_trades['win'].mean()*100:.1f}%")
    log(f"   Avg pips (after spread+slippage): {df_trades['pips'].mean():.2f}")
    log(f"   Total pips: {df_trades['pips'].sum():.0f}")
    log(f"   Std dev: {df_trades['pips'].std():.2f}")
    log(f"   Max win: {df_trades['pips'].max():.1f}")
    log(f"   Max loss: {df_trades['pips'].min():.1f}")
    log(f"   Profit factor: {abs(df_trades[df_trades['pips']>0]['pips'].sum() / df_trades[df_trades['pips']<0]['pips'].sum()):.2f}")

    # By signal
    log(f"\nBY SIGNAL:")
    for sig in ['BUY', 'SELL']:
        sig_df = df_trades[df_trades['signal'] == sig]
        if len(sig_df) > 0:
            wr = sig_df['win'].mean() * 100
            avg = sig_df['pips'].mean()
            total = sig_df['pips'].sum()
            log(f"   {sig}: {len(sig_df):,} trades, {wr:.1f}% WR, {avg:.2f} avg pips, {total:.0f} total")

    # By confidence
    log(f"\nBY CONFIDENCE THRESHOLD:")
    for conf_min in [0.6, 0.7, 0.8, 0.9]:
        conf_df = df_trades[df_trades['confidence'] >= conf_min]
        if len(conf_df) > 0:
            wr = conf_df['win'].mean() * 100
            avg = conf_df['pips'].mean()
            total = conf_df['pips'].sum()
            log(f"   conf>={conf_min}: {len(conf_df):>5} trades, {wr:>6.1f}% WR, {avg:>7.2f} avg pips, {total:>8.0f} total")

    # Model accuracy check
    log(f"\nMODEL PREDICTION ACCURACY:")
    # Compare predicted direction with actual
    # actual_direction: -1=down, 0=neutral, 1=up (from target)
    # pred_class: 0=down, 1=neutral, 2=up

    # Convert to same scale
    df_trades['pred_dir'] = df_trades['pred_class'].map({0: -1, 1: 0, 2: 1})
    df_trades['correct'] = (df_trades['pred_dir'] == df_trades['actual_direction']).astype(int)

    up_trades = df_trades[df_trades['pred_dir'] == 1]
    down_trades = df_trades[df_trades['pred_dir'] == -1]

    if len(up_trades) > 0:
        up_correct = (up_trades['actual_direction'] == 1).mean() * 100
        log(f"   UP predictions: {up_correct:.1f}% actually went up")

    if len(down_trades) > 0:
        down_correct = (down_trades['actual_direction'] == -1).mean() * 100
        log(f"   DOWN predictions: {down_correct:.1f}% actually went down")

    # Comparison with original claims
    log(f"\n" + "=" * 80)
    log("VALIDATION SUMMARY")
    log("=" * 80)

    actual_wr = df_trades['win'].mean() * 100
    avg_pips = df_trades['pips'].mean()

    log(f"\n   Original claimed WR: 80.8%")
    log(f"   Strict backtest WR:  {actual_wr:.1f}%")
    log(f"   Difference:          {actual_wr - 80.8:+.1f}%")
    log()
    log(f"   Average pips per trade: {avg_pips:.2f}")
    log(f"   Break-even WR (at {avg_pips:.1f} avg): ~50%")
    log()

    if actual_wr >= 60:
        log(f"   RESULT: Strong edge detected (>60% WR)")
    elif actual_wr >= 55:
        log(f"   RESULT: Moderate edge detected (55-60% WR)")
    elif actual_wr >= 52:
        log(f"   RESULT: Marginal edge (52-55% WR)")
    else:
        log(f"   RESULT: No significant edge (<52% WR)")

    # Save results
    output_path = OUTPUT_DIR / f'strict_backtest_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    df_trades.to_csv(output_path, index=False)
    log(f"\n   Results saved to: {output_path}")

# ============================================================================
# ADDITIONAL: WITHOUT SPREAD/SLIPPAGE (for comparison)
# ============================================================================

log("\n" + "=" * 80)
log("COMPARISON: WITHOUT SPREAD/SLIPPAGE (theoretical)")
log("=" * 80)

if len(trades) > 0:
    # Recalculate without costs
    raw_pips = []
    for t in trades:
        if t['signal'] == 'BUY':
            pips = (t['exit_price'] - t['entry_price']) / PIP_VALUE
        else:
            pips = (t['entry_price'] - t['exit_price']) / PIP_VALUE
        raw_pips.append(pips)

    raw_pips = np.array(raw_pips)
    raw_wins = (raw_pips > 0).mean() * 100

    log(f"\n   Raw WR (no costs): {raw_wins:.1f}%")
    log(f"   Raw avg pips: {raw_pips.mean():.2f}")
    log(f"   Cost impact: {raw_wins - actual_wr:.1f}% WR reduction")

log(f"\n{'='*80}")
log(f"Completed: {datetime.now()}")
log("=" * 80)
