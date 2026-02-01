"""
Backtest Quality Entry XGBoost Classifier
==========================================
Simulates trading using the quality entry predictions.

Strategy:
- Enter when MFC is at extreme (already filtered in dataset)
- Use model to filter for "quality" entries only
- Exit when MFC returns to center (0)
- Track P&L, win rate, drawdown
"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import pickle
import joblib
from pathlib import Path
from datetime import datetime

def log(msg=""):
    print(msg, flush=True)

log("=" * 70)
log("BACKTEST: QUALITY ENTRY XGBoost CLASSIFIER")
log("=" * 70)
log(f"Started: {datetime.now()}")

# Paths
DATA_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/lstm_model/data')
MODEL_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/lstm_model/models')

# Load model and features
log("\n1. Loading model and data...")
model = joblib.load(MODEL_DIR / 'quality_xgb_classifier.joblib')

with open(MODEL_DIR / 'quality_xgb_features.pkl', 'rb') as f:
    feature_cols = pickle.load(f)

with open(DATA_DIR / 'quality_entry_data.pkl', 'rb') as f:
    data = pickle.load(f)

df = data['data'].copy()
config = data['config']

log(f"  Total entries: {len(df)}")
log(f"  Features: {len(feature_cols)}")

# Prepare features
df['direction_code'] = (df['direction'] == 'buy').astype(int)
X = df[feature_cols].values.astype(np.float32)
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

# Get predictions
df['pred_prob'] = model.predict_proba(X)[:, 1]
df['pred_quality'] = (df['pred_prob'] >= 0.5).astype(int)

# Use test set only (last 15%)
n = len(df)
test_start = int(n * 0.85)
test_df = df.iloc[test_start:].copy()

log(f"  Test set: {len(test_df)} entries")
log(f"  Test date range: {test_df['datetime'].min()} to {test_df['datetime'].max()}")

# ============================================================================
# BACKTEST FUNCTION
# ============================================================================

def run_backtest(entries_df, threshold=0.5, use_filter=True, name="Strategy"):
    """
    Run backtest on entries.

    Each entry has:
    - direction: buy or sell
    - max_dd_pips: maximum adverse excursion
    - max_profit_pips: maximum favorable excursion
    - is_quality: actual outcome (1 = returned to center without big DD)

    We simulate:
    - Enter at signal
    - Exit when MFC returns to center (we know it always does)
    - The question is: how much drawdown before the return?
    """
    if use_filter:
        trades = entries_df[entries_df['pred_prob'] >= threshold].copy()
    else:
        trades = entries_df.copy()

    if len(trades) == 0:
        return None

    results = {
        'name': name,
        'threshold': threshold,
        'total_trades': len(trades),
        'quality_trades': trades['is_quality'].sum(),
        'quality_rate': trades['is_quality'].mean() * 100,
    }

    # For quality entries: we get profit with minimal DD
    # For non-quality entries: we get hit by DD first

    # Simulate P&L assuming:
    # - Quality entry: we capture most of max_profit with small DD
    # - Non-quality entry: we might get stopped out or endure big DD

    # Simple model:
    # - Quality trades: profit = max_profit_pips * 0.7 (capture 70%)
    # - Non-quality trades: loss = -max_dd_pips * 0.5 (take half the DD as loss)

    quality_mask = trades['is_quality'] == 1

    quality_profit = trades.loc[quality_mask, 'max_profit_pips'].sum() * 0.7
    non_quality_loss = trades.loc[~quality_mask, 'max_dd_pips'].sum() * 0.5

    total_pnl = quality_profit - non_quality_loss

    results['quality_profit_pips'] = quality_profit
    results['non_quality_loss_pips'] = non_quality_loss
    results['total_pnl_pips'] = total_pnl
    results['avg_pnl_per_trade'] = total_pnl / len(trades) if len(trades) > 0 else 0

    # More realistic: track per-trade P&L
    trades = trades.copy()
    trades['trade_pnl'] = np.where(
        trades['is_quality'] == 1,
        trades['max_profit_pips'] * 0.7,
        -trades['max_dd_pips'] * 0.5
    )

    # Win rate (positive trades)
    results['win_rate'] = (trades['trade_pnl'] > 0).mean() * 100

    # Average winner/loser
    winners = trades[trades['trade_pnl'] > 0]['trade_pnl']
    losers = trades[trades['trade_pnl'] <= 0]['trade_pnl']

    results['avg_winner'] = winners.mean() if len(winners) > 0 else 0
    results['avg_loser'] = losers.mean() if len(losers) > 0 else 0
    results['profit_factor'] = abs(winners.sum() / losers.sum()) if losers.sum() != 0 else float('inf')

    # Max drawdown (cumulative)
    cumulative = trades['trade_pnl'].cumsum()
    running_max = cumulative.cummax()
    drawdown = running_max - cumulative
    results['max_drawdown_pips'] = drawdown.max()

    # Sharpe-like ratio (simplified)
    if trades['trade_pnl'].std() > 0:
        results['sharpe_ratio'] = trades['trade_pnl'].mean() / trades['trade_pnl'].std()
    else:
        results['sharpe_ratio'] = 0

    # By direction
    for direction in ['buy', 'sell']:
        dir_trades = trades[trades['direction'] == direction]
        results[f'{direction}_trades'] = len(dir_trades)
        results[f'{direction}_quality_rate'] = dir_trades['is_quality'].mean() * 100 if len(dir_trades) > 0 else 0
        results[f'{direction}_pnl'] = dir_trades['trade_pnl'].sum()

    return results

# ============================================================================
# RUN BACKTESTS
# ============================================================================
log("\n" + "=" * 70)
log("2. BACKTEST RESULTS")
log("=" * 70)

# Baseline: no filter (all entries)
log("\n--- BASELINE (No Filter) ---")
baseline = run_backtest(test_df, use_filter=False, name="Baseline")
if baseline:
    log(f"  Total trades: {baseline['total_trades']}")
    log(f"  Quality rate: {baseline['quality_rate']:.1f}%")
    log(f"  Win rate: {baseline['win_rate']:.1f}%")
    log(f"  Total P&L: {baseline['total_pnl_pips']:.0f} pips")
    log(f"  Avg P&L/trade: {baseline['avg_pnl_per_trade']:.1f} pips")
    log(f"  Max Drawdown: {baseline['max_drawdown_pips']:.0f} pips")
    log(f"  Profit Factor: {baseline['profit_factor']:.2f}")

# Test different thresholds
log("\n--- WITH MODEL FILTER ---")
thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]

results_list = []
for thresh in thresholds:
    result = run_backtest(test_df, threshold=thresh, use_filter=True, name=f"Threshold {thresh}")
    if result:
        results_list.append(result)
        log(f"\nThreshold {thresh}:")
        log(f"  Trades: {result['total_trades']}")
        log(f"  Quality rate: {result['quality_rate']:.1f}%")
        log(f"  Win rate: {result['win_rate']:.1f}%")
        log(f"  Total P&L: {result['total_pnl_pips']:.0f} pips")
        log(f"  Avg P&L/trade: {result['avg_pnl_per_trade']:.1f} pips")
        log(f"  Max DD: {result['max_drawdown_pips']:.0f} pips")
        log(f"  Profit Factor: {result['profit_factor']:.2f}")
        log(f"  Sharpe: {result['sharpe_ratio']:.3f}")

# ============================================================================
# COMPARISON TABLE
# ============================================================================
log("\n" + "=" * 70)
log("3. COMPARISON TABLE")
log("=" * 70)

log("\n| Threshold | Trades | Quality% | Win% | Total P&L | Avg P&L | Max DD | PF |")
log("|-----------|--------|----------|------|-----------|---------|--------|------|")

if baseline:
    log(f"| {'None':^9} | {baseline['total_trades']:>6} | {baseline['quality_rate']:>7.1f}% | {baseline['win_rate']:>4.1f}% | {baseline['total_pnl_pips']:>9.0f} | {baseline['avg_pnl_per_trade']:>7.1f} | {baseline['max_drawdown_pips']:>6.0f} | {baseline['profit_factor']:>4.2f} |")

for r in results_list:
    log(f"| {r['threshold']:^9} | {r['total_trades']:>6} | {r['quality_rate']:>7.1f}% | {r['win_rate']:>4.1f}% | {r['total_pnl_pips']:>9.0f} | {r['avg_pnl_per_trade']:>7.1f} | {r['max_drawdown_pips']:>6.0f} | {r['profit_factor']:>4.2f} |")

# ============================================================================
# BY PAIR ANALYSIS
# ============================================================================
log("\n" + "=" * 70)
log("4. PERFORMANCE BY PAIR (Threshold 0.7)")
log("=" * 70)

thresh = 0.7
filtered = test_df[test_df['pred_prob'] >= thresh].copy()
filtered['trade_pnl'] = np.where(
    filtered['is_quality'] == 1,
    filtered['max_profit_pips'] * 0.7,
    -filtered['max_dd_pips'] * 0.5
)

log("\n| Pair    | Trades | Quality% | Total PnL | Avg PnL |")
log("|---------|--------|----------|-----------|---------|")

pair_stats = []
for pair in sorted(filtered['pair'].unique()):
    pair_df = filtered[filtered['pair'] == pair]
    stats = {
        'pair': pair,
        'trades': len(pair_df),
        'quality_rate': pair_df['is_quality'].mean() * 100,
        'total_pnl': pair_df['trade_pnl'].sum(),
        'avg_pnl': pair_df['trade_pnl'].mean()
    }
    pair_stats.append(stats)
    log(f"| {pair:^7} | {stats['trades']:>6} | {stats['quality_rate']:>7.1f}% | {stats['total_pnl']:>9.0f} | {stats['avg_pnl']:>7.1f} |")

# Best and worst pairs
pair_stats_df = pd.DataFrame(pair_stats)
best_pairs = pair_stats_df.nlargest(5, 'avg_pnl')
worst_pairs = pair_stats_df.nsmallest(5, 'avg_pnl')

log("\nBest 5 pairs by Avg P&L:")
for _, row in best_pairs.iterrows():
    log(f"  {row['pair']}: {row['avg_pnl']:.1f} pips/trade")

log("\nWorst 5 pairs by Avg P&L:")
for _, row in worst_pairs.iterrows():
    log(f"  {row['pair']}: {row['avg_pnl']:.1f} pips/trade")

# ============================================================================
# MONTHLY PERFORMANCE
# ============================================================================
log("\n" + "=" * 70)
log("5. MONTHLY PERFORMANCE (Threshold 0.7)")
log("=" * 70)

filtered['month'] = pd.to_datetime(filtered['datetime']).dt.to_period('M')

log("\n| Month   | Trades | Quality% | Total PnL | Cum PnL |")
log("|---------|--------|----------|-----------|---------|")

cumulative_pnl = 0
for month in sorted(filtered['month'].unique()):
    month_df = filtered[filtered['month'] == month]
    month_pnl = month_df['trade_pnl'].sum()
    cumulative_pnl += month_pnl
    log(f"| {str(month):^7} | {len(month_df):>6} | {month_df['is_quality'].mean()*100:>7.1f}% | {month_pnl:>9.0f} | {cumulative_pnl:>7.0f} |")

# ============================================================================
# SUMMARY
# ============================================================================
log("\n" + "=" * 70)
log("6. SUMMARY")
log("=" * 70)

if baseline and len(results_list) > 0:
    best_result = max(results_list, key=lambda x: x['avg_pnl_per_trade'])

    log(f"\nBaseline (no filter):")
    log(f"  - {baseline['total_trades']} trades, {baseline['quality_rate']:.1f}% quality")
    log(f"  - Total P&L: {baseline['total_pnl_pips']:.0f} pips")
    log(f"  - Avg P&L: {baseline['avg_pnl_per_trade']:.1f} pips/trade")

    log(f"\nBest filtered (threshold {best_result['threshold']}):")
    log(f"  - {best_result['total_trades']} trades, {best_result['quality_rate']:.1f}% quality")
    log(f"  - Total P&L: {best_result['total_pnl_pips']:.0f} pips")
    log(f"  - Avg P&L: {best_result['avg_pnl_per_trade']:.1f} pips/trade")

    improvement = best_result['avg_pnl_per_trade'] - baseline['avg_pnl_per_trade']
    log(f"\nImprovement: {improvement:+.1f} pips/trade ({improvement/baseline['avg_pnl_per_trade']*100:+.1f}%)")

log(f"\nCompleted: {datetime.now()}")
