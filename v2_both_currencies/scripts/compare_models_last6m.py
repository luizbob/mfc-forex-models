"""
Compare Models on Last 6 Months of Data
========================================
Compares all M5 models on truly out-of-sample recent data.
"""

import pandas as pd
import numpy as np
import pickle
import joblib
from pathlib import Path
from sklearn.metrics import roc_auc_score

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
MODEL_DIR = Path(__file__).parent.parent / "models"

def load_data():
    """Load prepared M5 data."""
    data_path = DATA_DIR / "quality_entry_data_m5_v2.pkl"
    print(f"Loading data from {data_path}...")
    loaded = pd.read_pickle(data_path)

    # Handle dict format
    if isinstance(loaded, dict):
        df = loaded['data']
    else:
        df = loaded

    # Rename datetime to entry_time for consistency
    if 'datetime' in df.columns and 'entry_time' not in df.columns:
        df = df.rename(columns={'datetime': 'entry_time'})

    print(f"Total entries: {len(df):,}")
    print(f"Date range: {df['entry_time'].min().date()} to {df['entry_time'].max().date()}")
    return df

def add_derived_features(df):
    """Add derived features (time and codes)."""
    df = df.copy()

    # Time features
    df['hour'] = df['entry_time'].dt.hour
    df['dayofweek'] = df['entry_time'].dt.dayofweek
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 5)
    df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 5)

    # Direction and trigger codes
    df['direction_code'] = (df['direction'] == 'buy').astype(int)
    df['trigger_code'] = (df['trigger'] == 'base').astype(int)

    return df

def evaluate_model(model, feature_cols, df, model_name, thresholds=[0.70, 0.75, 0.80]):
    """Evaluate a model on the given data."""
    # Prepare features
    X = df[feature_cols].values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Get predictions
    y_pred_proba = model.predict_proba(X)[:, 1]

    # Target
    y_true = (df['exit_pnl_pips'] > 0).astype(int)

    # AUC
    auc = roc_auc_score(y_true, y_pred_proba)

    print(f"\n{'='*70}")
    print(f"MODEL: {model_name}")
    print(f"{'='*70}")
    print(f"AUC: {auc:.4f}")
    print(f"\nThresh    Trades    WinRate    AvgPnL    TotalPnL       PF")
    print("-" * 60)

    results = {'name': model_name, 'auc': auc, 'thresholds': {}}

    for thresh in thresholds:
        mask = y_pred_proba >= thresh
        n_trades = mask.sum()

        if n_trades > 0:
            trades_df = df[mask]
            wins = (trades_df['exit_pnl_pips'] > 0).sum()
            wr = wins / n_trades * 100
            total_pnl = trades_df['exit_pnl_pips'].sum()
            avg_pnl = total_pnl / n_trades

            winning_pnl = trades_df[trades_df['exit_pnl_pips'] > 0]['exit_pnl_pips'].sum()
            losing_pnl = abs(trades_df[trades_df['exit_pnl_pips'] < 0]['exit_pnl_pips'].sum())
            pf = winning_pnl / losing_pnl if losing_pnl > 0 else float('inf')

            print(f"{thresh:.2f}     {n_trades:>6,}     {wr:>5.1f}%    {avg_pnl:>+6.1f}    {total_pnl:>+10,.0f}    {pf:>5.2f}")

            results['thresholds'][thresh] = {
                'trades': n_trades,
                'win_rate': wr,
                'total_pnl': total_pnl,
                'pf': pf
            }
        else:
            print(f"{thresh:.2f}          0        N/A       N/A           N/A      N/A")

    return results

def main():
    print("="*70)
    print("MODEL COMPARISON - LAST 6 MONTHS")
    print("="*70)

    # Load data
    df = load_data()
    df = add_derived_features(df)

    # Filter to last 6 months
    latest_date = df['entry_time'].max()
    six_months_ago = latest_date - pd.Timedelta(days=180)
    df_test = df[df['entry_time'] >= six_months_ago].copy()

    print(f"\nTest period: {df_test['entry_time'].min().date()} to {df_test['entry_time'].max().date()}")
    print(f"Test samples: {len(df_test):,}")

    # Models to compare
    models_config = [
        {
            'name': 'Original (V2 PnL)',
            'model_file': 'quality_xgb_m5_v2_pnl.joblib',
            'features_file': 'quality_xgb_features_m5_v2_pnl.pkl'
        },
        {
            'name': 'Optimized (Time Features)',
            'model_file': 'quality_xgb_m5_v2_pnl_optimized.joblib',
            'features_file': 'quality_xgb_features_m5_v2_pnl_optimized.pkl'
        },
        {
            'name': 'Walk-Forward (Expanding)',
            'model_file': 'quality_xgb_m5_v2_pnl_walkforward.joblib',
            'features_file': 'quality_xgb_features_m5_v2_pnl_walkforward.pkl'
        },
        {
            'name': 'Walk-Forward (Rolling 4yr)',
            'model_file': 'quality_xgb_m5_v2_pnl_walkforward_rolling.joblib',
            'features_file': 'quality_xgb_features_m5_v2_pnl_walkforward_rolling.pkl'
        },
    ]

    all_results = []

    for config in models_config:
        model_path = MODEL_DIR / config['model_file']
        features_path = MODEL_DIR / config['features_file']

        if not model_path.exists() or not features_path.exists():
            print(f"\nSkipping {config['name']} - files not found")
            continue

        model = joblib.load(model_path)
        with open(features_path, 'rb') as f:
            feature_cols = pickle.load(f)

        results = evaluate_model(model, feature_cols, df_test, config['name'])
        all_results.append(results)

    # Summary comparison
    print("\n" + "="*70)
    print("SUMMARY COMPARISON @ 0.75 THRESHOLD")
    print("="*70)
    print(f"{'Model':<30} {'AUC':>8} {'Trades':>10} {'WinRate':>10} {'TotalPnL':>12} {'PF':>8}")
    print("-" * 78)

    for r in all_results:
        if 0.75 in r['thresholds']:
            t = r['thresholds'][0.75]
            print(f"{r['name']:<30} {r['auc']:>8.4f} {t['trades']:>10,} {t['win_rate']:>9.1f}% {t['total_pnl']:>+12,.0f} {t['pf']:>8.2f}")

    print("\n" + "="*70)
    print("SUMMARY COMPARISON @ 0.80 THRESHOLD")
    print("="*70)
    print(f"{'Model':<30} {'AUC':>8} {'Trades':>10} {'WinRate':>10} {'TotalPnL':>12} {'PF':>8}")
    print("-" * 78)

    for r in all_results:
        if 0.80 in r['thresholds']:
            t = r['thresholds'][0.80]
            print(f"{r['name']:<30} {r['auc']:>8.4f} {t['trades']:>10,} {t['win_rate']:>9.1f}% {t['total_pnl']:>+12,.0f} {t['pf']:>8.2f}")

if __name__ == "__main__":
    main()
