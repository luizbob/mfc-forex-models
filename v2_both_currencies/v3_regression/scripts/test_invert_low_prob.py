"""
Test: Can we INVERT low probability trades?
============================================
If model says low probability for BUY, should we SELL instead?
"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import pickle
import joblib
from pathlib import Path

def log(msg=""):
    print(msg, flush=True)

DATA_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/lstm_model/v2_both_currencies/v3_regression/data')
MODEL_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/lstm_model/v2_both_currencies/v3_regression/models')

log("Loading data and model...")
with open(DATA_DIR / 'momentum_data_m15_v3.pkl', 'rb') as f:
    data = pickle.load(f)

model = joblib.load(MODEL_DIR / 'momentum_xgb_m15_v3.joblib')
with open(MODEL_DIR / 'momentum_xgb_features_m15_v3.pkl', 'rb') as f:
    feature_cols = pickle.load(f)

SPREADS = {
    'AUDUSD': 0.9, 'EURUSD': 0.8, 'GBPUSD': 1.0, 'NZDUSD': 1.8,
    'USDCAD': 1.5, 'USDCHF': 1.3, 'USDJPY': 1.0,
    'AUDCAD': 2.2, 'AUDCHF': 0.9, 'AUDJPY': 1.9, 'AUDNZD': 2.0,
    'CADCHF': 0.8, 'CADJPY': 3.8, 'CHFJPY': 2.4,
    'EURAUD': 3.4, 'EURCAD': 2.9, 'EURCHF': 2.5, 'EURGBP': 1.4,
    'EURJPY': 2.4, 'EURNZD': 5.4,
    'GBPAUD': 2.5, 'GBPCAD': 4.8, 'GBPCHF': 2.4, 'GBPJPY': 2.2, 'GBPNZD': 5.8,
    'NZDCAD': 2.1, 'NZDCHF': 1.5, 'NZDJPY': 4.3,
}

df = data['data'].copy()
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.sort_values('datetime').reset_index(drop=True)

# Prepare features
df['hour'] = df['datetime'].dt.hour
df['dayofweek'] = df['datetime'].dt.dayofweek
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 5)
df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 5)
df['direction_code'] = (df['direction'] == 'buy').astype(int)
df['trigger_code'] = (df['trigger'] == 'base').astype(int)

X = df[feature_cols].values.astype(np.float32)
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

# Use last 20% as test
n = len(df)
train_size = int(n * 0.8)
df_test = df.iloc[train_size:].copy()
X_test = X[train_size:]

df_test['pred_prob'] = model.predict_proba(X_test)[:, 1]

# Get spread for each trade
df_test['spread'] = df_test['pair'].map(SPREADS).fillna(2.0)

# Calculate inverted pips
# Original: adjusted_pips - spread = net_pips
# Inverted: -adjusted_pips - spread = inverted_net
# So: inverted_net = -(net_pips + spread) - spread = -net_pips - 2*spread
df_test['inverted_pips'] = -df_test['adjusted_pips'] - df_test['spread']

log(f"Test set: {len(df_test):,} trades")

log("\n" + "=" * 70)
log("STRATEGY COMPARISON")
log("=" * 70)

def evaluate(subset, pips_col='net_pips'):
    if len(subset) == 0:
        return None
    net = subset[pips_col]
    trades = len(net)
    wr = (net > 0).mean() * 100
    winners = net[net > 0].sum()
    losers = abs(net[net <= 0].sum())
    pf = winners / losers if losers > 0 else 999
    return {'trades': trades, 'wr': wr, 'pf': pf, 'net': net.sum(), 'avg': net.mean()}

log("\n1. ORIGINAL STRATEGY (high prob = trade as signaled)")
log("-" * 60)
for thresh in [0.45, 0.50, 0.55, 0.60]:
    subset = df_test[df_test['pred_prob'] >= thresh]
    r = evaluate(subset, 'net_pips')
    if r and r['trades'] > 50:
        log(f"   Prob >= {thresh}: {r['trades']:>6,} trades, {r['wr']:>5.1f}% WR, PF {r['pf']:.2f}, {r['avg']:>+6.2f} avg")

log("\n2. INVERTED STRATEGY (low prob = trade OPPOSITE direction)")
log("-" * 60)
for thresh in [0.30, 0.25, 0.20, 0.15]:
    subset = df_test[df_test['pred_prob'] <= thresh]
    r = evaluate(subset, 'inverted_pips')
    if r and r['trades'] > 50:
        log(f"   Prob <= {thresh}: {r['trades']:>6,} trades, {r['wr']:>5.1f}% WR, PF {r['pf']:.2f}, {r['avg']:>+6.2f} avg")

log("\n3. COMBINED: High prob normal + Low prob inverted")
log("-" * 60)

# High prob: trade as signaled
high_prob = df_test[df_test['pred_prob'] >= 0.45].copy()
high_prob['final_pips'] = high_prob['net_pips']

# Low prob: trade inverted
low_prob = df_test[df_test['pred_prob'] <= 0.25].copy()
low_prob['final_pips'] = low_prob['inverted_pips']

combined = pd.concat([high_prob, low_prob])
r = evaluate(combined, 'final_pips')
log(f"   High (>=0.45) + Inverted (<=0.25): {r['trades']:>6,} trades, {r['wr']:>5.1f}% WR, PF {r['pf']:.2f}, {r['avg']:>+6.2f} avg, {r['net']:>+,.0f} total")

log("\n" + "=" * 70)
log("DETAILED ANALYSIS - LOW PROBABILITY INVERTED")
log("=" * 70)

low_prob = df_test[df_test['pred_prob'] <= 0.25]
log(f"\nTrades with probability <= 0.25: {len(low_prob):,}")
log(f"\nOriginal direction:")
log(f"  Win rate: {(low_prob['net_pips'] > 0).mean()*100:.1f}%")
log(f"  Avg pips: {low_prob['net_pips'].mean():+.2f}")
log(f"\nInverted direction:")
log(f"  Win rate: {(low_prob['inverted_pips'] > 0).mean()*100:.1f}%")
log(f"  Avg pips: {low_prob['inverted_pips'].mean():+.2f}")

log("\n" + "=" * 70)
log("SAMPLE INVERTED TRADES (prob <= 0.25)")
log("=" * 70)

sample = low_prob.head(15)
log(f"\n{'Date':<20} {'Pair':<10} {'Orig':<6} {'Prob':>6} {'Orig Pips':>10} {'Inv Pips':>10}")
log("-" * 75)
for _, row in sample.iterrows():
    inv_dir = 'sell' if row['direction'] == 'buy' else 'buy'
    log(f"{str(row['datetime']):<20} {row['pair']:<10} {row['direction']:<6} {row['pred_prob']:>6.2f} {row['net_pips']:>+10.1f} {row['inverted_pips']:>+10.1f}")

log("\nDONE")
