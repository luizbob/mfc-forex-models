"""
Train XGBoost for Quality Entry Prediction (M30 Base)
=====================================================
Same as H1 version but for M30 timeframe data.
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
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

def log(msg=""):
    print(msg, flush=True)

log("=" * 70)
log("QUALITY ENTRY XGBoost TRAINING (M30 Base)")
log("=" * 70)
log(f"Started: {datetime.now()}")

# Paths
DATA_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/lstm_model/data')
MODEL_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/lstm_model/models')
MODEL_DIR.mkdir(exist_ok=True, parents=True)

# Load data
log("\n1. Loading M30 data...")
with open(DATA_DIR / 'quality_entry_data_m30.pkl', 'rb') as f:
    data = pickle.load(f)

df = data['data']
config = data['config']

log(f"  Total samples: {len(df)}")
log(f"  Quality rate: {df['is_quality'].mean()*100:.1f}%")
log(f"  Config: {config}")

# ============================================================================
# PREPARE FEATURES
# ============================================================================
log("\n2. Preparing features...")

# Feature columns - all MFC values and velocities
feature_cols = []

# MFC values for all timeframes
for tf in ['m5', 'm15', 'm30', 'h1', 'h4']:
    feature_cols.extend([f'base_{tf}', f'quote_{tf}'])
    feature_cols.extend([f'base_vel_{tf}', f'quote_vel_{tf}'])

# Additional features (M30 base timeframe)
feature_cols.extend(['base_vel2_m30', 'base_acc_m30', 'divergence', 'vel_divergence'])

# Direction encoding (buy=1, sell=0)
df['direction_code'] = (df['direction'] == 'buy').astype(int)
feature_cols.append('direction_code')

log(f"  Feature columns: {len(feature_cols)}")

# Prepare X and y
X = df[feature_cols].values.astype(np.float32)
y = df['is_quality'].values.astype(np.float32)

# Handle any NaN/inf
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

log(f"  X shape: {X.shape}")
log(f"  y shape: {y.shape}")

# Train/val/test split (time-based: 70/15/15)
n = len(df)
train_idx = int(n * 0.70)
val_idx = int(n * 0.85)

X_train, X_val, X_test = X[:train_idx], X[train_idx:val_idx], X[val_idx:]
y_train, y_val, y_test = y[:train_idx], y[train_idx:val_idx], y[val_idx:]

log(f"\n  Train: {len(X_train)} samples")
log(f"  Val: {len(X_val)} samples")
log(f"  Test: {len(X_test)} samples")

log(f"\n  Train quality rate: {y_train.mean()*100:.1f}%")
log(f"  Val quality rate: {y_val.mean()*100:.1f}%")
log(f"  Test quality rate: {y_test.mean()*100:.1f}%")

# ============================================================================
# TRAIN XGBOOST
# ============================================================================
log("\n3. Training XGBoost...")

# Calculate scale_pos_weight for class imbalance
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

# ============================================================================
# EVALUATE
# ============================================================================
log("\n" + "=" * 70)
log("4. EVALUATION")
log("=" * 70)

# Test set predictions
y_pred_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_pred_prob >= 0.5).astype(int)

# AUC
auc = roc_auc_score(y_test, y_pred_prob)
log(f"\nTest AUC: {auc:.4f}")

# Classification report
log("\nClassification Report:")
log(classification_report(y_test, y_pred, target_names=['Non-Quality', 'Quality']))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
log(f"\nConfusion Matrix:")
log(f"  TN={cm[0,0]}, FP={cm[0,1]}")
log(f"  FN={cm[1,0]}, TP={cm[1,1]}")

# Precision at different thresholds
log("\nPrecision at different confidence thresholds:")
for thresh in [0.5, 0.6, 0.7, 0.8, 0.9]:
    high_conf_mask = y_pred_prob >= thresh
    if high_conf_mask.sum() > 0:
        precision = y_test[high_conf_mask].mean()
        count = high_conf_mask.sum()
        log(f"  Threshold {thresh}: Precision={precision*100:.1f}%, Count={count}")

# ============================================================================
# FEATURE IMPORTANCE
# ============================================================================
log("\n" + "=" * 70)
log("5. FEATURE IMPORTANCE")
log("=" * 70)

importance = model.feature_importances_
importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': importance
}).sort_values('importance', ascending=False)

log("\nTop 15 most important features:")
for _, row in importance_df.head(15).iterrows():
    log(f"  {row['feature']:20s}: {row['importance']:.4f}")

# ============================================================================
# COMPARE WITH H1
# ============================================================================
log("\n" + "=" * 70)
log("6. COMPARISON: M30 vs H1")
log("=" * 70)

log("\n| Metric          | H1 Model | M30 Model |")
log("|-----------------|----------|-----------|")
log(f"| Total samples   | 323,286  | {len(df):>9,} |")
log(f"| Quality rate    | 70.2%    | {df['is_quality'].mean()*100:>8.1f}% |")
log(f"| Test AUC        | 0.656    | {auc:>9.3f} |")

# ============================================================================
# SAVE
# ============================================================================
log("\n" + "=" * 70)
log("7. SAVING")
log("=" * 70)

# Save model
model_path = MODEL_DIR / 'quality_xgb_classifier_m30.joblib'
joblib.dump(model, model_path)
log(f"Saved model to: {model_path}")

# Save feature columns for inference
with open(MODEL_DIR / 'quality_xgb_features_m30.pkl', 'wb') as f:
    pickle.dump(feature_cols, f)

log(f"\nCompleted: {datetime.now()}")
