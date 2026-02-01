"""
Script 01: LSTM Data Preparation
================================
Prepares MFC data for LSTM training:
- Loads all 8 currencies across 5 timeframes (M5, M15, M30, H1, H4)
- Creates training samples with lookback windows
- Labels with MFC cycle movement targets
- Applies shift(1) to avoid lookahead bias
"""
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import pickle

def log(msg=""):
    print(msg, flush=True)

log("=" * 70)
log("LSTM DATA PREPARATION")
log("=" * 70)
log(f"Started: {datetime.now()}")

# Paths
DATA_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/data')
CLEAN_DIR = DATA_DIR / 'cleaned'
OUTPUT_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/lstm_model/data')
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Config
CURRENCIES = ['EUR', 'USD', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD', 'NZD']
TIMEFRAMES = ['M5', 'M15', 'M30', 'H1', 'H4']

# Lookback bars per timeframe (how much history the LSTM sees)
LOOKBACK = {
    'M5': 48,   # 4 hours
    'M15': 32,  # 8 hours
    'M30': 24,  # 12 hours
    'H1': 24,   # 24 hours
    'H4': 18,   # 72 hours (3 days)
}

# Target config
TARGET_BARS_M5 = 72  # Predict movement over next 6 hours (72 M5 bars)
CYCLE_THRESHOLD = 0.2  # Box boundary

# Date range
START_DATE = '2023-01-01'
END_DATE = '2024-12-31'

# ============================================================================
# 1. LOAD MFC DATA
# ============================================================================
log("\n1. Loading MFC data...")

mfc_data = {}

for ccy in CURRENCIES:
    mfc_data[ccy] = {}

    for tf in TIMEFRAMES:
        try:
            # H1 and H4 use cleaned data
            if tf in ['H1', 'H4']:
                filepath = CLEAN_DIR / f'mfc_currency_{ccy}_{tf}_clean.csv'
            else:
                filepath = DATA_DIR / f'mfc_currency_{ccy}_{tf}.csv'

            df = pd.read_csv(filepath)
            df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y.%m.%d %H:%M')
            df = df.set_index('datetime')
            df = df[(df.index >= START_DATE) & (df.index <= END_DATE)]
            mfc_data[ccy][tf] = df['MFC']

        except Exception as e:
            log(f"  ERROR loading {ccy} {tf}: {e}")

log(f"  Loaded data for {len(CURRENCIES)} currencies, {len(TIMEFRAMES)} timeframes")

# ============================================================================
# 2. CREATE BASE TIMELINE (M5)
# ============================================================================
log("\n2. Creating base M5 timeline...")

# Use EUR M5 as base timeline (all currencies have same timestamps)
base_timeline = mfc_data['EUR']['M5'].index.copy()
log(f"  Timeline: {base_timeline[0]} to {base_timeline[-1]}")
log(f"  Total bars: {len(base_timeline)}")

# ============================================================================
# 3. ALIGN ALL TIMEFRAMES TO M5 TIMELINE
# ============================================================================
log("\n3. Aligning all timeframes to M5...")

aligned_mfc = {}

for ccy in CURRENCIES:
    aligned_mfc[ccy] = pd.DataFrame(index=base_timeline)

    for tf in TIMEFRAMES:
        # Reindex to M5 timeline with forward fill, then shift(1) for lookahead bias
        series = mfc_data[ccy][tf].shift(1).reindex(base_timeline, method='ffill')
        aligned_mfc[ccy][tf] = series

    # Also add velocity (M5 change)
    aligned_mfc[ccy]['vel_M5'] = aligned_mfc[ccy]['M5'] - aligned_mfc[ccy]['M5'].shift(1)
    aligned_mfc[ccy]['vel_M30'] = aligned_mfc[ccy]['M30'] - aligned_mfc[ccy]['M30'].shift(6)  # 6 M5 bars = 30 min

log(f"  Aligned all currencies to M5 timeline with shift(1)")

# ============================================================================
# 4. CREATE TRAINING SAMPLES
# ============================================================================
log("\n4. Creating training samples...")

def get_cycle_target(mfc_series, idx, future_bars, threshold=0.2):
    """
    Determine cycle target based on current MFC position.

    If MFC < -threshold (below box): target = will it rise to >= -threshold?
    If MFC > +threshold (above box): target = will it fall to <= +threshold?
    If in box: target = which direction will it break?

    Returns:
        direction: 1 (up), -1 (down), 0 (neutral/unclear)
        completed: 1 if cycle completed within future_bars, 0 otherwise
        max_move: maximum MFC movement in target direction
    """
    current_mfc = mfc_series.iloc[idx]

    if idx + future_bars >= len(mfc_series):
        return 0, 0, 0.0

    future_mfc = mfc_series.iloc[idx+1:idx+1+future_bars]

    if pd.isna(current_mfc) or len(future_mfc) == 0:
        return 0, 0, 0.0

    # Below box - expecting cycle UP
    if current_mfc < -threshold:
        target_direction = 1  # UP
        # Did it cross back into box?
        crossed = (future_mfc >= -threshold).any()
        completed = 1 if crossed else 0
        max_move = future_mfc.max() - current_mfc

    # Above box - expecting cycle DOWN
    elif current_mfc > threshold:
        target_direction = -1  # DOWN
        # Did it cross back into box?
        crossed = (future_mfc <= threshold).any()
        completed = 1 if crossed else 0
        max_move = current_mfc - future_mfc.min()

    # In box - check which direction it breaks
    else:
        up_break = (future_mfc > threshold).any()
        down_break = (future_mfc < -threshold).any()

        if up_break and not down_break:
            target_direction = 1
            completed = 1
            max_move = future_mfc.max() - current_mfc
        elif down_break and not up_break:
            target_direction = -1
            completed = 1
            max_move = current_mfc - future_mfc.min()
        else:
            # Both or neither - unclear
            target_direction = 0
            completed = 0
            max_move = 0.0

    return target_direction, completed, max_move


def create_samples_for_currency(ccy, aligned_df, mfc_m5_raw):
    """
    Create training samples for one currency.

    Each sample contains:
    - Lookback window of MFC values for each timeframe
    - Velocities
    - Target labels
    """
    samples = []
    max_lookback = max(LOOKBACK.values())

    # We need raw (non-shifted) M5 for target calculation
    mfc_m5_aligned = mfc_m5_raw.reindex(aligned_df.index, method='ffill')

    valid_start = max_lookback + 1
    valid_end = len(aligned_df) - TARGET_BARS_M5 - 1

    for i in range(valid_start, valid_end):
        sample = {'currency': ccy, 'datetime': aligned_df.index[i]}

        # Features: lookback windows for each timeframe
        for tf in TIMEFRAMES:
            lookback = LOOKBACK[tf]
            window = aligned_df[tf].iloc[i-lookback:i].values

            if len(window) == lookback and not np.any(np.isnan(window)):
                sample[f'{tf}_window'] = window
            else:
                sample[f'{tf}_window'] = None

        # Velocities at current time
        sample['vel_M5'] = aligned_df['vel_M5'].iloc[i]
        sample['vel_M30'] = aligned_df['vel_M30'].iloc[i]

        # Current MFC values (for reference)
        sample['current_M5'] = aligned_df['M5'].iloc[i]
        sample['current_M30'] = aligned_df['M30'].iloc[i]
        sample['current_H4'] = aligned_df['H4'].iloc[i]

        # Target: cycle direction and completion
        direction, completed, max_move = get_cycle_target(
            mfc_m5_aligned, i, TARGET_BARS_M5, CYCLE_THRESHOLD
        )
        sample['target_direction'] = direction
        sample['target_completed'] = completed
        sample['target_max_move'] = max_move

        # Only keep samples with valid windows
        if all(sample[f'{tf}_window'] is not None for tf in TIMEFRAMES):
            samples.append(sample)

    return samples


# Create samples for each currency
all_samples = {}
total_samples = 0

for ccy in CURRENCIES:
    log(f"  Processing {ccy}...")
    samples = create_samples_for_currency(
        ccy,
        aligned_mfc[ccy],
        mfc_data[ccy]['M5']
    )
    all_samples[ccy] = samples
    total_samples += len(samples)
    log(f"    {len(samples)} samples")

log(f"\n  Total samples: {total_samples}")

# ============================================================================
# 5. ANALYZE TARGETS
# ============================================================================
log("\n5. Analyzing target distribution...")

for ccy in CURRENCIES:
    samples = all_samples[ccy]
    if len(samples) == 0:
        continue

    directions = [s['target_direction'] for s in samples]
    completions = [s['target_completed'] for s in samples]

    up = sum(1 for d in directions if d == 1)
    down = sum(1 for d in directions if d == -1)
    neutral = sum(1 for d in directions if d == 0)
    completed = sum(completions)

    log(f"  {ccy}: UP={up} ({up/len(samples)*100:.1f}%), "
        f"DOWN={down} ({down/len(samples)*100:.1f}%), "
        f"NEUTRAL={neutral} ({neutral/len(samples)*100:.1f}%), "
        f"Completed={completed/len(samples)*100:.1f}%")

# ============================================================================
# 6. CONVERT TO NUMPY ARRAYS
# ============================================================================
log("\n6. Converting to numpy arrays...")

def samples_to_arrays(samples):
    """Convert list of sample dicts to numpy arrays for LSTM training."""
    n = len(samples)

    # Feature arrays
    X_M5 = np.zeros((n, LOOKBACK['M5']))
    X_M15 = np.zeros((n, LOOKBACK['M15']))
    X_M30 = np.zeros((n, LOOKBACK['M30']))
    X_H1 = np.zeros((n, LOOKBACK['H1']))
    X_H4 = np.zeros((n, LOOKBACK['H4']))
    X_aux = np.zeros((n, 5))  # vel_M5, vel_M30, current_M5, current_M30, current_H4

    # Target arrays
    y_direction = np.zeros(n)
    y_completed = np.zeros(n)
    y_max_move = np.zeros(n)

    # Metadata
    datetimes = []

    for i, s in enumerate(samples):
        X_M5[i] = s['M5_window']
        X_M15[i] = s['M15_window']
        X_M30[i] = s['M30_window']
        X_H1[i] = s['H1_window']
        X_H4[i] = s['H4_window']

        X_aux[i] = [
            s['vel_M5'] if not np.isnan(s['vel_M5']) else 0,
            s['vel_M30'] if not np.isnan(s['vel_M30']) else 0,
            s['current_M5'] if not np.isnan(s['current_M5']) else 0,
            s['current_M30'] if not np.isnan(s['current_M30']) else 0,
            s['current_H4'] if not np.isnan(s['current_H4']) else 0,
        ]

        y_direction[i] = s['target_direction']
        y_completed[i] = s['target_completed']
        y_max_move[i] = s['target_max_move']

        datetimes.append(s['datetime'])

    return {
        'X_M5': X_M5,
        'X_M15': X_M15,
        'X_M30': X_M30,
        'X_H1': X_H1,
        'X_H4': X_H4,
        'X_aux': X_aux,
        'y_direction': y_direction,
        'y_completed': y_completed,
        'y_max_move': y_max_move,
        'datetimes': np.array(datetimes),
    }


processed_data = {}

for ccy in CURRENCIES:
    if len(all_samples[ccy]) > 0:
        processed_data[ccy] = samples_to_arrays(all_samples[ccy])
        log(f"  {ccy}: X_M5 shape = {processed_data[ccy]['X_M5'].shape}")

# ============================================================================
# 7. SAVE DATA
# ============================================================================
log("\n7. Saving processed data...")

# Save per-currency files
for ccy in CURRENCIES:
    if ccy in processed_data:
        filepath = OUTPUT_DIR / f'lstm_data_{ccy}.pkl'
        with open(filepath, 'wb') as f:
            pickle.dump(processed_data[ccy], f)
        log(f"  Saved {filepath.name}")

# Save config for reference
config = {
    'lookback': LOOKBACK,
    'target_bars_m5': TARGET_BARS_M5,
    'cycle_threshold': CYCLE_THRESHOLD,
    'start_date': START_DATE,
    'end_date': END_DATE,
    'currencies': CURRENCIES,
    'timeframes': TIMEFRAMES,
}

with open(OUTPUT_DIR / 'config.pkl', 'wb') as f:
    pickle.dump(config, f)
log("  Saved config.pkl")

# ============================================================================
# 8. SUMMARY
# ============================================================================
log("\n" + "=" * 70)
log("SUMMARY")
log("=" * 70)

log(f"\nData prepared for LSTM training:")
log(f"  - {len(CURRENCIES)} currencies")
log(f"  - {len(TIMEFRAMES)} timeframes: {TIMEFRAMES}")
log(f"  - Lookback: {LOOKBACK}")
log(f"  - Target: {TARGET_BARS_M5} M5 bars ({TARGET_BARS_M5 * 5 / 60:.1f} hours)")
log(f"  - Cycle threshold: +/- {CYCLE_THRESHOLD}")
log(f"  - Total samples: {total_samples}")
log(f"\nOutput saved to: {OUTPUT_DIR}")

log(f"\nCompleted: {datetime.now()}")
