"""
Script 02: LSTM Model Architecture
===================================
Multi-timeframe LSTM for MFC cycle prediction.

Architecture:
- Separate LSTM branch for each timeframe (M5, M15, M30, H1, H4)
- Branches concatenated
- Auxiliary features added (velocity, current values)
- Two outputs: direction (classification) and completion probability
"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pickle
from pathlib import Path
from datetime import datetime

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

def log(msg=""):
    print(msg, flush=True)

log("=" * 70)
log("LSTM MODEL TRAINING")
log("=" * 70)
log(f"Started: {datetime.now()}")
log(f"TensorFlow version: {tf.__version__}")

# Check GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    log(f"GPU available: {gpus}")
else:
    log("No GPU found, using CPU")

# Paths
DATA_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/lstm_model/data')
MODEL_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/lstm_model/models')
MODEL_DIR.mkdir(exist_ok=True, parents=True)

# Load config
with open(DATA_DIR / 'config.pkl', 'rb') as f:
    config = pickle.load(f)

LOOKBACK = config['lookback']
CURRENCIES = config['currencies']

log(f"\nConfig: {config}")

# ============================================================================
# 1. BUILD MODEL ARCHITECTURE
# ============================================================================
log("\n1. Building model architecture...")

def build_multi_tf_lstm(lookback_dict, lstm_units=64, dense_units=32):
    """
    Build multi-timeframe LSTM model.

    Inputs:
    - M5 sequence (lookback['M5'], 1)
    - M15 sequence (lookback['M15'], 1)
    - M30 sequence (lookback['M30'], 1)
    - H1 sequence (lookback['H1'], 1)
    - H4 sequence (lookback['H4'], 1)
    - Auxiliary features (5,): vel_M5, vel_M30, current_M5, current_M30, current_H4

    Outputs:
    - direction: 3-class (down=-1, neutral=0, up=1)
    - completed: binary (will cycle complete?)
    """
    # Input layers
    input_m5 = layers.Input(shape=(lookback_dict['M5'], 1), name='input_M5')
    input_m15 = layers.Input(shape=(lookback_dict['M15'], 1), name='input_M15')
    input_m30 = layers.Input(shape=(lookback_dict['M30'], 1), name='input_M30')
    input_h1 = layers.Input(shape=(lookback_dict['H1'], 1), name='input_H1')
    input_h4 = layers.Input(shape=(lookback_dict['H4'], 1), name='input_H4')
    input_aux = layers.Input(shape=(5,), name='input_aux')

    # LSTM branches - each timeframe gets its own LSTM
    # M5: Most granular, needs to capture short-term patterns
    x_m5 = layers.LSTM(lstm_units, return_sequences=True, name='lstm_m5_1')(input_m5)
    x_m5 = layers.LSTM(lstm_units // 2, name='lstm_m5_2')(x_m5)

    # M15: Medium-short term
    x_m15 = layers.LSTM(lstm_units, return_sequences=True, name='lstm_m15_1')(input_m15)
    x_m15 = layers.LSTM(lstm_units // 2, name='lstm_m15_2')(x_m15)

    # M30: Medium term (direction indicator)
    x_m30 = layers.LSTM(lstm_units, return_sequences=True, name='lstm_m30_1')(input_m30)
    x_m30 = layers.LSTM(lstm_units // 2, name='lstm_m30_2')(x_m30)

    # H1: Longer term context
    x_h1 = layers.LSTM(lstm_units // 2, name='lstm_h1')(input_h1)

    # H4: Macro context
    x_h4 = layers.LSTM(lstm_units // 2, name='lstm_h4')(input_h4)

    # Concatenate all branches
    concat = layers.Concatenate(name='concat')([x_m5, x_m15, x_m30, x_h1, x_h4, input_aux])

    # Shared dense layers
    x = layers.Dense(dense_units * 2, activation='relu', name='dense1')(concat)
    x = layers.Dropout(0.3, name='dropout1')(x)
    x = layers.Dense(dense_units, activation='relu', name='dense2')(x)
    x = layers.Dropout(0.2, name='dropout2')(x)

    # Output heads
    # Direction: 3 classes (down=-1, neutral=0, up=1) -> indices 0, 1, 2
    output_direction = layers.Dense(3, activation='softmax', name='direction')(x)

    # Completion probability: binary
    output_completed = layers.Dense(1, activation='sigmoid', name='completed')(x)

    model = Model(
        inputs=[input_m5, input_m15, input_m30, input_h1, input_h4, input_aux],
        outputs=[output_direction, output_completed],
        name='multi_tf_lstm'
    )

    return model


# Build model
model = build_multi_tf_lstm(LOOKBACK, lstm_units=64, dense_units=32)
model.summary()

# ============================================================================
# 2. LOAD AND PREPARE DATA
# ============================================================================
log("\n2. Loading training data...")

def load_currency_data(currency):
    """Load and prepare data for one currency."""
    with open(DATA_DIR / f'lstm_data_{currency}.pkl', 'rb') as f:
        data = pickle.load(f)

    # Reshape sequences for LSTM (add feature dimension)
    X_M5 = data['X_M5'].reshape(-1, LOOKBACK['M5'], 1)
    X_M15 = data['X_M15'].reshape(-1, LOOKBACK['M15'], 1)
    X_M30 = data['X_M30'].reshape(-1, LOOKBACK['M30'], 1)
    X_H1 = data['X_H1'].reshape(-1, LOOKBACK['H1'], 1)
    X_H4 = data['X_H4'].reshape(-1, LOOKBACK['H4'], 1)
    X_aux = data['X_aux']

    # Convert direction from (-1, 0, 1) to (0, 1, 2) for categorical
    y_direction = data['y_direction'].astype(int) + 1  # -1->0, 0->1, 1->2
    y_completed = data['y_completed']

    return {
        'X': [X_M5, X_M15, X_M30, X_H1, X_H4, X_aux],
        'y_direction': y_direction,
        'y_completed': y_completed,
        'datetimes': data['datetimes']
    }


# Train on all currencies combined (universal model) or per-currency
# Starting with per-currency for better specialization
TRAIN_CURRENCY = 'EUR'  # Start with EUR, can expand later

log(f"\nTraining on: {TRAIN_CURRENCY}")

data = load_currency_data(TRAIN_CURRENCY)
n_samples = len(data['y_direction'])
log(f"  Total samples: {n_samples}")

# Train/validation split (80/20, time-based)
split_idx = int(n_samples * 0.8)

X_train = [x[:split_idx] for x in data['X']]
X_val = [x[split_idx:] for x in data['X']]

y_train_dir = data['y_direction'][:split_idx]
y_val_dir = data['y_direction'][split_idx:]

y_train_comp = data['y_completed'][:split_idx]
y_val_comp = data['y_completed'][split_idx:]

log(f"  Train samples: {split_idx}")
log(f"  Validation samples: {n_samples - split_idx}")

# Class distribution
log(f"\n  Train direction dist: DOWN={np.sum(y_train_dir==0)}, NEUTRAL={np.sum(y_train_dir==1)}, UP={np.sum(y_train_dir==2)}")
log(f"  Train completion rate: {np.mean(y_train_comp)*100:.1f}%")

# ============================================================================
# 3. COMPILE AND TRAIN
# ============================================================================
log("\n3. Compiling model...")

# Compute class weights for direction (handle imbalance)
class_counts = np.bincount(y_train_dir.astype(int), minlength=3)
total = np.sum(class_counts)
class_weights = {i: total / (3 * count) if count > 0 else 1.0 for i, count in enumerate(class_counts)}
log(f"  Class weights: {class_weights}")

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss={
        'direction': 'sparse_categorical_crossentropy',
        'completed': 'binary_crossentropy'
    },
    loss_weights={
        'direction': 1.0,
        'completed': 0.5  # Secondary objective
    },
    metrics={
        'direction': 'accuracy',
        'completed': 'accuracy'
    }
)

# Callbacks
callbacks = [
    EarlyStopping(
        monitor='val_direction_accuracy',
        patience=10,
        restore_best_weights=True,
        mode='max'
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6
    ),
    ModelCheckpoint(
        MODEL_DIR / f'lstm_{TRAIN_CURRENCY}_best.keras',
        monitor='val_direction_accuracy',
        save_best_only=True,
        mode='max'
    )
]

log("\n4. Training model...")
log("=" * 70)

history = model.fit(
    X_train,
    [y_train_dir, y_train_comp],
    validation_data=(X_val, [y_val_dir, y_val_comp]),
    epochs=50,
    batch_size=256,
    callbacks=callbacks,
    verbose=1
)

# ============================================================================
# 4. EVALUATE
# ============================================================================
log("\n" + "=" * 70)
log("5. Evaluation")
log("=" * 70)

# Final evaluation
val_results = model.evaluate(X_val, [y_val_dir, y_val_comp], verbose=0)
log(f"\nValidation Results:")
log(f"  Total Loss: {val_results[0]:.4f}")
log(f"  Direction Loss: {val_results[1]:.4f}")
log(f"  Completed Loss: {val_results[2]:.4f}")
log(f"  Direction Accuracy: {val_results[3]*100:.2f}%")
log(f"  Completed Accuracy: {val_results[4]*100:.2f}%")

# Predictions analysis
y_pred = model.predict(X_val, verbose=0)
y_pred_dir = np.argmax(y_pred[0], axis=1)
y_pred_comp = (y_pred[1] > 0.5).astype(int).flatten()

# Confusion matrix for direction
from collections import Counter
log("\nDirection Predictions:")
log(f"  Predicted: {Counter(y_pred_dir)}")
log(f"  Actual: {Counter(y_val_dir.astype(int))}")

# Per-class accuracy
for cls, label in enumerate(['DOWN', 'NEUTRAL', 'UP']):
    mask = y_val_dir == cls
    if np.sum(mask) > 0:
        acc = np.mean(y_pred_dir[mask] == cls)
        log(f"  {label} accuracy: {acc*100:.1f}% (n={np.sum(mask)})")

# Save model
model.save(MODEL_DIR / f'lstm_{TRAIN_CURRENCY}_final.keras')
log(f"\nModel saved to: {MODEL_DIR / f'lstm_{TRAIN_CURRENCY}_final.keras'}")

# Save training history
with open(MODEL_DIR / f'history_{TRAIN_CURRENCY}.pkl', 'wb') as f:
    pickle.dump(history.history, f)

log(f"\nCompleted: {datetime.now()}")
