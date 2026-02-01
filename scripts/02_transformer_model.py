"""
Script 02b: Transformer Model Architecture
==========================================
Multi-timeframe Transformer for MFC cycle prediction.

Architecture:
- Positional encoding for each timeframe sequence
- Multi-head self-attention encoder for each timeframe
- Cross-timeframe attention to combine information
- Same inputs/outputs as LSTM for compatibility
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
log("TRANSFORMER MODEL TRAINING")
log("=" * 70)
log(f"Started: {datetime.now()}")
log(f"TensorFlow version: {tf.__version__}")

# Check GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    log(f"GPU available: {gpus}")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
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
# TRANSFORMER COMPONENTS
# ============================================================================

class PositionalEncoding(layers.Layer):
    """Sinusoidal positional encoding."""
    def __init__(self, max_len, d_model, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.d_model = d_model

        # Create positional encoding matrix
        position = np.arange(max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

        pe = np.zeros((max_len, d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)

        self.pe = tf.constant(pe, dtype=tf.float32)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + self.pe[:seq_len, :]

    def get_config(self):
        config = super().get_config()
        config.update({'max_len': self.max_len, 'd_model': self.d_model})
        return config


class TransformerEncoderBlock(layers.Layer):
    """Single Transformer encoder block."""
    def __init__(self, d_model, num_heads, ff_dim, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout

        self.mha = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout
        )
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation='gelu'),
            layers.Dropout(dropout),
            layers.Dense(d_model),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)

    def call(self, x, training=False):
        # Self-attention
        attn_output = self.mha(x, x, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        x = self.layernorm1(x + attn_output)

        # Feed-forward
        ffn_output = self.ffn(x, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(x + ffn_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'dropout': self.dropout_rate
        })
        return config


def build_timeframe_encoder(seq_len, d_model=64, num_heads=4, num_blocks=2, ff_dim=128, dropout=0.1, name='tf_encoder'):
    """Build a Transformer encoder for one timeframe."""
    inputs = layers.Input(shape=(seq_len, 1), name=f'input_{name}')

    # Project to d_model dimensions
    x = layers.Dense(d_model, name=f'{name}_proj')(inputs)

    # Add positional encoding
    x = PositionalEncoding(seq_len, d_model, name=f'{name}_pos_enc')(x)
    x = layers.Dropout(dropout, name=f'{name}_dropout_in')(x)

    # Transformer encoder blocks
    for i in range(num_blocks):
        x = TransformerEncoderBlock(
            d_model, num_heads, ff_dim, dropout,
            name=f'{name}_block_{i}'
        )(x)

    # Global average pooling to get fixed-size output
    x = layers.GlobalAveragePooling1D(name=f'{name}_pool')(x)

    return keras.Model(inputs, x, name=name)


def build_multi_tf_transformer(lookback_dict, d_model=64, num_heads=4, num_blocks=2, ff_dim=128, dense_units=64, dropout=0.2):
    """
    Build multi-timeframe Transformer model.

    Architecture:
    - Separate Transformer encoder for each timeframe
    - Encoders output fixed-size representations
    - All representations concatenated with auxiliary features
    - Dense layers for final prediction
    """
    # Input layers
    input_m5 = layers.Input(shape=(lookback_dict['M5'], 1), name='input_M5')
    input_m15 = layers.Input(shape=(lookback_dict['M15'], 1), name='input_M15')
    input_m30 = layers.Input(shape=(lookback_dict['M30'], 1), name='input_M30')
    input_h1 = layers.Input(shape=(lookback_dict['H1'], 1), name='input_H1')
    input_h4 = layers.Input(shape=(lookback_dict['H4'], 1), name='input_H4')
    input_aux = layers.Input(shape=(5,), name='input_aux')

    # Build separate encoders for each timeframe
    # M5: Short-term patterns (4 hours of data)
    enc_m5 = build_timeframe_encoder(lookback_dict['M5'], d_model, num_heads, num_blocks, ff_dim, dropout, 'enc_m5')
    x_m5 = enc_m5(input_m5)

    # M15: Short-medium term (8 hours)
    enc_m15 = build_timeframe_encoder(lookback_dict['M15'], d_model, num_heads, num_blocks, ff_dim, dropout, 'enc_m15')
    x_m15 = enc_m15(input_m15)

    # M30: Medium term (12 hours)
    enc_m30 = build_timeframe_encoder(lookback_dict['M30'], d_model, num_heads, num_blocks, ff_dim, dropout, 'enc_m30')
    x_m30 = enc_m30(input_m30)

    # H1: Longer term (24 hours)
    enc_h1 = build_timeframe_encoder(lookback_dict['H1'], d_model // 2, num_heads // 2, 1, ff_dim // 2, dropout, 'enc_h1')
    x_h1 = enc_h1(input_h1)

    # H4: Macro context (72 hours)
    enc_h4 = build_timeframe_encoder(lookback_dict['H4'], d_model // 2, num_heads // 2, 1, ff_dim // 2, dropout, 'enc_h4')
    x_h4 = enc_h4(input_h4)

    # Concatenate all timeframe representations + auxiliary features
    concat = layers.Concatenate(name='concat')([x_m5, x_m15, x_m30, x_h1, x_h4, input_aux])

    # Dense layers for final prediction
    x = layers.Dense(dense_units * 2, activation='gelu', name='dense1')(concat)
    x = layers.Dropout(dropout, name='dropout1')(x)
    x = layers.Dense(dense_units, activation='gelu', name='dense2')(x)
    x = layers.Dropout(dropout * 0.5, name='dropout2')(x)

    # Output heads
    output_direction = layers.Dense(3, activation='softmax', name='direction')(x)
    output_completed = layers.Dense(1, activation='sigmoid', name='completed')(x)

    model = Model(
        inputs=[input_m5, input_m15, input_m30, input_h1, input_h4, input_aux],
        outputs=[output_direction, output_completed],
        name='multi_tf_transformer'
    )

    return model


# ============================================================================
# BUILD MODEL
# ============================================================================
log("\n1. Building Transformer model...")

model = build_multi_tf_transformer(
    LOOKBACK,
    d_model=64,      # Embedding dimension
    num_heads=4,     # Attention heads
    num_blocks=2,    # Transformer blocks per timeframe
    ff_dim=128,      # Feed-forward dimension
    dense_units=64,  # Dense layer size
    dropout=0.2      # Dropout rate
)

model.summary()

# Count parameters
total_params = model.count_params()
log(f"\nTotal parameters: {total_params:,}")


# ============================================================================
# LOAD AND PREPARE DATA
# ============================================================================
log("\n2. Loading training data...")

def load_currency_data(currency):
    """Load and prepare data for one currency."""
    with open(DATA_DIR / f'lstm_data_{currency}.pkl', 'rb') as f:
        data = pickle.load(f)

    # Reshape sequences (already have feature dimension from LSTM format)
    X_M5 = data['X_M5'].reshape(-1, LOOKBACK['M5'], 1)
    X_M15 = data['X_M15'].reshape(-1, LOOKBACK['M15'], 1)
    X_M30 = data['X_M30'].reshape(-1, LOOKBACK['M30'], 1)
    X_H1 = data['X_H1'].reshape(-1, LOOKBACK['H1'], 1)
    X_H4 = data['X_H4'].reshape(-1, LOOKBACK['H4'], 1)
    X_aux = data['X_aux']

    # Convert direction from (-1, 0, 1) to (0, 1, 2) for categorical
    y_direction = data['y_direction'].astype(int) + 1
    y_completed = data['y_completed']

    return {
        'X': [X_M5, X_M15, X_M30, X_H1, X_H4, X_aux],
        'y_direction': y_direction,
        'y_completed': y_completed,
        'datetimes': data['datetimes']
    }


def train_currency_model(currency):
    """Train Transformer model for one currency."""
    log(f"\n{'='*70}")
    log(f"Training {currency}")
    log(f"{'='*70}")

    data = load_currency_data(currency)
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

    log(f"  Train: {split_idx}, Val: {n_samples - split_idx}")
    log(f"  Direction dist: DOWN={np.sum(y_train_dir==0)}, NEUTRAL={np.sum(y_train_dir==1)}, UP={np.sum(y_train_dir==2)}")

    # Build fresh model for this currency
    currency_model = build_multi_tf_transformer(
        LOOKBACK,
        d_model=64,
        num_heads=4,
        num_blocks=2,
        ff_dim=128,
        dense_units=64,
        dropout=0.2
    )

    # Compile
    currency_model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=0.001, weight_decay=0.01),
        loss={
            'direction': 'sparse_categorical_crossentropy',
            'completed': 'binary_crossentropy'
        },
        loss_weights={'direction': 1.0, 'completed': 0.5},
        metrics={'direction': 'accuracy', 'completed': 'accuracy'}
    )

    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_direction_accuracy',
            patience=15,
            restore_best_weights=True,
            mode='max'
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-6
        ),
        ModelCheckpoint(
            MODEL_DIR / f'transformer_{currency}_best.keras',
            monitor='val_direction_accuracy',
            save_best_only=True,
            mode='max'
        )
    ]

    # Train
    history = currency_model.fit(
        X_train,
        [y_train_dir, y_train_comp],
        validation_data=(X_val, [y_val_dir, y_val_comp]),
        epochs=100,
        batch_size=256,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate
    val_results = currency_model.evaluate(X_val, [y_val_dir, y_val_comp], verbose=0)
    dir_acc = val_results[3] * 100
    comp_acc = val_results[4] * 100

    log(f"\n  {currency} Results:")
    log(f"    Direction Accuracy: {dir_acc:.1f}%")
    log(f"    Completed Accuracy: {comp_acc:.1f}%")

    # Save final model
    currency_model.save(MODEL_DIR / f'transformer_{currency}_final.keras')

    return {
        'currency': currency,
        'direction_accuracy': dir_acc,
        'completed_accuracy': comp_acc,
        'history': history.history
    }


# ============================================================================
# TRAIN ALL CURRENCIES
# ============================================================================
log("\n3. Training all currencies...")

results = []
for currency in CURRENCIES:
    try:
        result = train_currency_model(currency)
        results.append(result)
    except Exception as e:
        log(f"ERROR training {currency}: {e}")
        import traceback
        traceback.print_exc()

# ============================================================================
# SUMMARY
# ============================================================================
log("\n" + "=" * 70)
log("TRANSFORMER TRAINING SUMMARY")
log("=" * 70)

log("\n| Currency | Direction Acc | Completed Acc |")
log("|----------|---------------|---------------|")

total_dir = 0
total_comp = 0
for r in results:
    log(f"| {r['currency']:^8} | {r['direction_accuracy']:>11.1f}% | {r['completed_accuracy']:>11.1f}% |")
    total_dir += r['direction_accuracy']
    total_comp += r['completed_accuracy']

if results:
    avg_dir = total_dir / len(results)
    avg_comp = total_comp / len(results)
    log("|----------|---------------|---------------|")
    log(f"| {'AVERAGE':^8} | {avg_dir:>11.1f}% | {avg_comp:>11.1f}% |")

# Compare with LSTM
log("\n" + "=" * 70)
log("COMPARISON: TRANSFORMER vs LSTM")
log("=" * 70)
log("\nLSTM Results (from previous training):")
log("| Currency | Direction Acc |")
log("|----------|---------------|")
lstm_results = {
    'EUR': 82.0, 'USD': 87.8, 'GBP': 83.7, 'JPY': 90.0,
    'CHF': 86.3, 'CAD': 82.2, 'AUD': 87.4, 'NZD': 86.7
}
lstm_avg = np.mean(list(lstm_results.values()))
for ccy, acc in lstm_results.items():
    log(f"| {ccy:^8} | {acc:>11.1f}% |")
log(f"| {'AVERAGE':^8} | {lstm_avg:>11.1f}% |")

if results:
    log(f"\nTransformer Average: {avg_dir:.1f}%")
    log(f"LSTM Average: {lstm_avg:.1f}%")
    diff = avg_dir - lstm_avg
    log(f"Difference: {diff:+.1f}%")

# Save results
with open(MODEL_DIR / 'transformer_results.pkl', 'wb') as f:
    pickle.dump(results, f)

log(f"\nCompleted: {datetime.now()}")
