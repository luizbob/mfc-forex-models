"""
Train XGBoost Quality Entry - PROPER TIME-SORTED
================================================
Sort data by time FIRST, then split for proper out-of-sample validation.
"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import pickle
import joblib
from pathlib import Path
from datetime import datetime

from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score

def log(msg=""):
    print(msg, flush=True)

DATA_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/lstm_model/data')
MODEL_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/lstm_model/models')

def train_model(name, data_file, model_suffix=""):
    """Train model with proper time-sorted data."""

    log(f"\n{'='*70}")
    log(f"TRAINING {name} MODEL (Proper Time-Sorted)")
    log(f"{'='*70}")
    log(f"Started: {datetime.now()}")

    # Load data
    log("\n1. Loading data...")
    with open(DATA_DIR / data_file, 'rb') as f:
        data = pickle.load(f)

    df = data['data'].copy()
    config = data['config']

    # CRITICAL: Sort by datetime for proper time-based split
    log("\n2. Sorting by datetime...")
    df = df.sort_values('datetime').reset_index(drop=True)

    log(f"  Total samples: {len(df):,}")
    log(f"  Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    log(f"  Quality rate: {df['is_quality'].mean()*100:.1f}%")

    # Feature columns
    feature_cols = []
    for tf in ['m5', 'm15', 'm30', 'h1', 'h4']:
        feature_cols.extend([f'base_{tf}', f'quote_{tf}'])
        feature_cols.extend([f'base_vel_{tf}', f'quote_vel_{tf}'])

    # Additional features based on base timeframe
    if name == 'H1':
        feature_cols.extend(['base_vel2_h1', 'base_acc_h1', 'divergence', 'vel_divergence'])
    else:  # M30
        feature_cols.extend(['base_vel2_m30', 'base_acc_m30', 'divergence', 'vel_divergence'])

    df['direction_code'] = (df['direction'] == 'buy').astype(int)
    feature_cols.append('direction_code')

    log(f"  Features: {len(feature_cols)}")

    # Prepare X and y
    X = df[feature_cols].values.astype(np.float32)
    y = df['is_quality'].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Time-based split: 70/15/15
    n = len(df)
    train_idx = int(n * 0.70)
    val_idx = int(n * 0.85)

    X_train, X_val, X_test = X[:train_idx], X[train_idx:val_idx], X[val_idx:]
    y_train, y_val, y_test = y[:train_idx], y[train_idx:val_idx], y[val_idx:]

    train_dates = df.iloc[:train_idx]['datetime']
    val_dates = df.iloc[train_idx:val_idx]['datetime']
    test_dates = df.iloc[val_idx:]['datetime']

    log(f"\n3. PROPER TIME-BASED SPLIT:")
    log(f"  Train: {len(X_train):,} samples ({train_dates.min().date()} to {train_dates.max().date()})")
    log(f"  Val:   {len(X_val):,} samples ({val_dates.min().date()} to {val_dates.max().date()})")
    log(f"  Test:  {len(X_test):,} samples ({test_dates.min().date()} to {test_dates.max().date()})")

    log(f"\n  Train quality rate: {y_train.mean()*100:.1f}%")
    log(f"  Val quality rate: {y_val.mean()*100:.1f}%")
    log(f"  Test quality rate: {y_test.mean()*100:.1f}%")

    # Train XGBoost
    log("\n4. Training XGBoost...")

    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    log(f"  Scale pos weight: {scale_pos_weight:.3f}")

    model = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric='auc',
        early_stopping_rounds=30,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50
    )

    log(f"\n  Best iteration: {model.best_iteration}")

    # Evaluate on TEST set
    log("\n5. EVALUATION (Out-of-Sample Test Set)")
    log("=" * 50)

    y_pred_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_prob >= 0.5).astype(int)

    auc = roc_auc_score(y_test, y_pred_prob)
    log(f"\nTest AUC: {auc:.4f}")

    log("\nClassification Report:")
    log(classification_report(y_test, y_pred, target_names=['Non-Quality', 'Quality']))

    # Precision at thresholds
    log("\nPrecision at confidence thresholds:")
    for thresh in [0.5, 0.6, 0.7, 0.8, 0.9]:
        mask = y_pred_prob >= thresh
        if mask.sum() > 0:
            precision = y_test[mask].mean()
            count = mask.sum()
            log(f"  Threshold {thresh}: Precision={precision*100:.1f}%, Count={count:,}")

    # Feature importance
    log("\nTop 10 Features:")
    importance = model.feature_importances_
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': importance
    }).sort_values('importance', ascending=False)

    for _, row in importance_df.head(10).iterrows():
        log(f"  {row['feature']:20s}: {row['importance']:.4f}")

    # Save model
    log("\n6. Saving...")

    model_path = MODEL_DIR / f'quality_xgb{model_suffix}_proper.joblib'
    features_path = MODEL_DIR / f'quality_xgb_features{model_suffix}_proper.pkl'

    joblib.dump(model, model_path)
    with open(features_path, 'wb') as f:
        pickle.dump(feature_cols, f)

    log(f"  Model: {model_path}")
    log(f"  Features: {features_path}")

    log(f"\nCompleted: {datetime.now()}")

    return {
        'auc': auc,
        'model_path': model_path,
        'test_quality_rate': y_test.mean() * 100
    }


# ============================================================================
# TRAIN BOTH MODELS
# ============================================================================

log("=" * 70)
log("RETRAINING WITH PROPER TIME-SORTED DATA")
log("=" * 70)

# Train H1 model
h1_result = train_model('H1', 'quality_entry_data.pkl', model_suffix='')

# Train M30 model
m30_result = train_model('M30', 'quality_entry_data_m30.pkl', model_suffix='_m30')

# Summary
log("\n" + "=" * 70)
log("TRAINING COMPLETE")
log("=" * 70)

log("\n| Model | Test AUC | Test Quality Rate |")
log("|-------|----------|-------------------|")
log(f"| H1    | {h1_result['auc']:.4f}   | {h1_result['test_quality_rate']:.1f}%             |")
log(f"| M30   | {m30_result['auc']:.4f}   | {m30_result['test_quality_rate']:.1f}%             |")

log("\nModels saved with '_proper' suffix")
