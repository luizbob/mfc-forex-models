"""
Analyze Model Selection - Does high probability = long bars held?
=================================================================
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

# Use last 20% as test (same as training script)
n = len(df)
train_size = int(n * 0.8)
df_test = df.iloc[train_size:].copy()
X_test = X[train_size:]

log(f"Test set: {len(df_test):,} trades")

# Get model predictions
df_test['pred_prob'] = model.predict_proba(X_test)[:, 1]

log("\n" + "=" * 70)
log("PROOF: MODEL PROBABILITY vs BARS HELD")
log("=" * 70)

log("\nQuestion: When model gives HIGH probability, do trades hold LONGER?")
log("\n" + "-" * 70)

# Group by probability ranges
prob_ranges = [
    (0.00, 0.30, 'Low (0.00-0.30)'),
    (0.30, 0.40, 'Medium-Low (0.30-0.40)'),
    (0.40, 0.50, 'Medium (0.40-0.50)'),
    (0.50, 0.60, 'Medium-High (0.50-0.60)'),
    (0.60, 1.00, 'High (0.60+)'),
]

log(f"\n{'Probability Range':<25} {'Trades':>10} {'Avg Bars':>10} {'Win %':>10} {'Avg Pips':>12}")
log("-" * 70)

for low, high, label in prob_ranges:
    subset = df_test[(df_test['pred_prob'] >= low) & (df_test['pred_prob'] < high)]
    if len(subset) > 0:
        avg_bars = subset['bars_held'].mean()
        wr = (subset['net_pips'] > 0).mean() * 100
        avg_pips = subset['net_pips'].mean()
        log(f"{label:<25} {len(subset):>10,} {avg_bars:>10.2f} {wr:>9.1f}% {avg_pips:>+12.2f}")

log("\n" + "=" * 70)
log("DISTRIBUTION OF BARS HELD BY PROBABILITY")
log("=" * 70)

log("\nFor trades with probability >= 0.45:")
filtered = df_test[df_test['pred_prob'] >= 0.45]
log(f"  Total selected: {len(filtered):,}")
log(f"\n  Bars held distribution:")
for low, high, label in [(1, 3, '1-2 bars'), (3, 6, '3-5 bars'), (6, 12, '6-11 bars'), (12, 50, '12+ bars')]:
    subset = filtered[(filtered['bars_held'] >= low) & (filtered['bars_held'] < high)]
    pct = len(subset) / len(filtered) * 100
    log(f"    {label}: {len(subset):,} ({pct:.1f}%)")

log("\nFor trades with probability < 0.45 (rejected):")
rejected = df_test[df_test['pred_prob'] < 0.45]
log(f"  Total rejected: {len(rejected):,}")
log(f"\n  Bars held distribution:")
for low, high, label in [(1, 3, '1-2 bars'), (3, 6, '3-5 bars'), (6, 12, '6-11 bars'), (12, 50, '12+ bars')]:
    subset = rejected[(rejected['bars_held'] >= low) & (rejected['bars_held'] < high)]
    pct = len(subset) / len(rejected) * 100
    log(f"    {label}: {len(subset):,} ({pct:.1f}%)")

log("\n" + "=" * 70)
log("SAMPLE TRADES - HIGH PROBABILITY (should be 6+ bars)")
log("=" * 70)

# Show some high probability trades
high_prob = df_test[df_test['pred_prob'] >= 0.55].head(20)
log(f"\n{'Date':<20} {'Pair':<10} {'Dir':<6} {'Prob':>6} {'Bars':>6} {'Net Pips':>10} {'Result':<8}")
log("-" * 75)
for _, row in high_prob.iterrows():
    result = "WIN" if row['net_pips'] > 0 else "LOSS"
    log(f"{str(row['datetime']):<20} {row['pair']:<10} {row['direction']:<6} {row['pred_prob']:>6.2f} {row['bars_held']:>6} {row['net_pips']:>+10.1f} {result:<8}")

log("\n" + "=" * 70)
log("SAMPLE TRADES - LOW PROBABILITY (should be 1-2 bars)")
log("=" * 70)

# Show some low probability trades
low_prob = df_test[df_test['pred_prob'] < 0.30].head(20)
log(f"\n{'Date':<20} {'Pair':<10} {'Dir':<6} {'Prob':>6} {'Bars':>6} {'Net Pips':>10} {'Result':<8}")
log("-" * 75)
for _, row in low_prob.iterrows():
    result = "WIN" if row['net_pips'] > 0 else "LOSS"
    log(f"{str(row['datetime']):<20} {row['pair']:<10} {row['direction']:<6} {row['pred_prob']:>6.2f} {row['bars_held']:>6} {row['net_pips']:>+10.1f} {result:<8}")

log("\n" + "=" * 70)
log("CONCLUSION")
log("=" * 70)

# Calculate correlation
corr = df_test['pred_prob'].corr(df_test['bars_held'])
log(f"\nCorrelation between probability and bars_held: {corr:.3f}")

avg_bars_high = df_test[df_test['pred_prob'] >= 0.45]['bars_held'].mean()
avg_bars_low = df_test[df_test['pred_prob'] < 0.45]['bars_held'].mean()
log(f"Avg bars held (prob >= 0.45): {avg_bars_high:.2f}")
log(f"Avg bars held (prob < 0.45): {avg_bars_low:.2f}")

log("\nDONE")
