"""
Experiment 04: XGBoost Baseline
===============================
Train XGBoost models for comparison with LSTM.
XGBoost doesn't use sequences - it uses flattened features.

Features:
- Last N MFC values from each timeframe (flattened)
- Auxiliary features
- Technical indicators if available

This provides a baseline to compare LSTM sequence learning vs
traditional ML on flattened features.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pickle
from pathlib import Path
from datetime import datetime
import gc

import warnings
warnings.filterwarnings('ignore')

# XGBoost
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("WARNING: XGBoost not installed. Run: pip install xgboost")

from sklearn.metrics import accuracy_score, classification_report

def log(msg=""):
    print(msg, flush=True)

log("=" * 70)
log("EXPERIMENT 04: XGBoost Baseline")
log("=" * 70)
log(f"Started: {datetime.now()}")

if not HAS_XGBOOST:
    log("ERROR: XGBoost not available. Exiting.")
    exit(1)

# ============================================================================
# CONFIGURATION
# ============================================================================

V2_DIR = Path(__file__).parent.parent
DATA_DIR = V2_DIR / "data" / "pairs"
EXP_DIR = Path(__file__).parent
RESULTS_DIR = EXP_DIR / "results_exp04"
RESULTS_DIR.mkdir(exist_ok=True)

# XGBoost parameters
XGB_PARAMS = {
    'objective': 'multi:softmax',
    'num_class': 3,
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 200,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'tree_method': 'hist',  # Fast histogram-based algorithm
    'device': 'cuda',  # Use GPU if available
}

# Feature extraction - use last N values from each timeframe
FEATURE_LOOKBACK = {
    'M5': 12,   # Last 12 M5 values (1 hour)
    'M15': 8,   # Last 8 M15 values (2 hours)
    'M30': 6,   # Last 6 M30 values (3 hours)
    'H1': 6,    # Last 6 H1 values (6 hours)
    'H4': 4,    # Last 4 H4 values (16 hours)
}

# Test pairs
TEST_PAIRS = ['EURUSD', 'GBPUSD', 'USDJPY', 'EURJPY', 'AUDUSD', 'GBPJPY']

# Load config
config_path = DATA_DIR / "config_pairs.pkl"
with open(config_path, 'rb') as f:
    config = pickle.load(f)

LOOKBACKS = config['lookbacks']

log(f"\nConfig loaded:")
log(f"   Test pairs: {TEST_PAIRS}")
log(f"   Feature lookback: {FEATURE_LOOKBACK}")

# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def extract_xgb_features(X):
    """
    Extract flattened features for XGBoost from sequence data.

    Takes the last N values from each timeframe and flattens them.
    """
    n_samples = len(X['aux'])

    features_list = []

    # Base currency features
    for tf in ['M5', 'M15', 'M30', 'H1', 'H4']:
        key = f'base_{tf.lower()}'
        seq = X[key]  # Shape: (n_samples, lookback, 1)
        lookback = FEATURE_LOOKBACK[tf]

        # Take last N values and flatten
        feat = seq[:, -lookback:, 0]  # Shape: (n_samples, lookback)
        features_list.append(feat)

    # Quote currency features
    for tf in ['M5', 'M15', 'M30', 'H1', 'H4']:
        key = f'quote_{tf.lower()}'
        seq = X[key]
        lookback = FEATURE_LOOKBACK[tf]

        feat = seq[:, -lookback:, 0]
        features_list.append(feat)

    # Auxiliary features
    features_list.append(X['aux'])

    # Concatenate all features
    X_flat = np.hstack(features_list)

    return X_flat

def load_pair_data(pair):
    """Load data for a pair."""
    path = DATA_DIR / f"v2_pair_{pair}.pkl"

    with open(path, 'rb') as f:
        data = pickle.load(f)

    X = {
        'base_m5': data['X_base_M5'][..., np.newaxis],
        'base_m15': data['X_base_M15'][..., np.newaxis],
        'base_m30': data['X_base_M30'][..., np.newaxis],
        'base_h1': data['X_base_H1'][..., np.newaxis],
        'base_h4': data['X_base_H4'][..., np.newaxis],
        'quote_m5': data['X_quote_M5'][..., np.newaxis],
        'quote_m15': data['X_quote_M15'][..., np.newaxis],
        'quote_m30': data['X_quote_M30'][..., np.newaxis],
        'quote_h1': data['X_quote_H1'][..., np.newaxis],
        'quote_h4': data['X_quote_H4'][..., np.newaxis],
        'aux': data['X_aux'],
    }

    y = data['y_direction']

    return X, y, data['datetimes']

def time_based_split(X, y, test_ratio=0.1, val_ratio=0.15):
    """Split data by time."""
    n = len(y)
    test_idx = int(n * (1 - test_ratio))

    X_test = X[test_idx:]
    y_test = y[test_idx:]

    X_trainval = X[:test_idx]
    y_trainval = y[:test_idx]

    n_trainval = len(y_trainval)
    val_idx = int(n_trainval * (1 - val_ratio))

    X_train = X_trainval[:val_idx]
    y_train = y_trainval[:val_idx]

    X_val = X_trainval[val_idx:]
    y_val = y_trainval[val_idx:]

    return X_train, y_train, X_val, y_val, X_test, y_test

# ============================================================================
# RUN EXPERIMENTS
# ============================================================================

results = []

for pair in TEST_PAIRS:
    log(f"\n{'='*60}")
    log(f"Training XGBoost for {pair}...")
    log(f"{'='*60}")

    # Load data
    log(f"   Loading data...")
    X_dict, y, datetimes = load_pair_data(pair)

    # Extract flattened features
    log(f"   Extracting features...")
    X = extract_xgb_features(X_dict)

    n_samples, n_features = X.shape
    class_counts = np.bincount(y, minlength=3)
    log(f"   Total samples: {n_samples:,}")
    log(f"   Features: {n_features}")
    log(f"   Classes: SHORT={class_counts[0]:,}, NEUTRAL={class_counts[1]:,}, LONG={class_counts[2]:,}")

    # Split data
    X_train, y_train, X_val, y_val, X_test, y_test = time_based_split(X, y)
    log(f"   Train: {len(y_train):,}, Val: {len(y_val):,}, Test: {len(y_test):,}")

    # Calculate sample weights for class imbalance
    class_weights = {}
    total = len(y_train)
    for cls in range(3):
        count = np.sum(y_train == cls)
        class_weights[cls] = total / (3 * count) if count > 0 else 1.0

    sample_weights = np.array([class_weights[y] for y in y_train])

    # Train XGBoost
    log(f"   Training XGBoost...")

    try:
        # Try GPU first
        model = xgb.XGBClassifier(**XGB_PARAMS)
        model.fit(
            X_train, y_train,
            sample_weight=sample_weights,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
    except Exception as e:
        log(f"   GPU failed ({e}), falling back to CPU...")
        params_cpu = XGB_PARAMS.copy()
        params_cpu['device'] = 'cpu'
        model = xgb.XGBClassifier(**params_cpu)
        model.fit(
            X_train, y_train,
            sample_weight=sample_weights,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

    # Evaluate on validation
    y_val_pred = model.predict(X_val)
    val_acc = accuracy_score(y_val, y_val_pred)

    # Evaluate on test
    y_test_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)

    log(f"   Val Accuracy: {val_acc:.4f}")
    log(f"   Test Accuracy: {test_acc:.4f}")

    # Per-class accuracy
    for cls, name in enumerate(['SHORT', 'NEUTRAL', 'LONG']):
        mask = y_test == cls
        if mask.sum() > 0:
            cls_acc = (y_test_pred[mask] == cls).mean()
            log(f"   {name}: {cls_acc:.4f} ({mask.sum():,} samples)")

    # Save model
    model_path = RESULTS_DIR / f"xgb_{pair}.json"
    model.save_model(str(model_path))
    log(f"   Saved: {model_path.name}")

    results.append({
        'pair': pair,
        'val_acc': val_acc,
        'test_acc': test_acc,
        'n_samples': n_samples,
        'n_features': n_features,
    })

    # Cleanup
    del model, X, y, X_train, y_train, X_val, y_val, X_test, y_test
    gc.collect()

# ============================================================================
# SUMMARY
# ============================================================================

log(f"\n{'='*70}")
log("EXPERIMENT SUMMARY: XGBoost Baseline")
log(f"{'='*70}")

log(f"\n{'Pair':<10} {'Val Acc':>10} {'Test Acc':>10}")
log("-" * 35)

for r in results:
    log(f"{r['pair']:<10} {r['val_acc']:>10.4f} {r['test_acc']:>10.4f}")

if results:
    avg_val = np.mean([r['val_acc'] for r in results])
    avg_test = np.mean([r['test_acc'] for r in results])
    log("-" * 35)
    log(f"{'AVERAGE':<10} {avg_val:>10.4f} {avg_test:>10.4f}")

# Save results
results_path = RESULTS_DIR / "exp04_results.pkl"
with open(results_path, 'wb') as f:
    pickle.dump(results, f)

log(f"\nResults saved to: {results_path}")
log(f"\nCompleted: {datetime.now()}")
log("=" * 70)
