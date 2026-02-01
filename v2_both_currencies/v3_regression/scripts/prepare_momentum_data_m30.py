"""
Prepare Momentum Continuation Dataset - M30
============================================
Triggers on HIGH VELOCITY (momentum) instead of extreme MFC.
Uses same shift logic as quality model - all TFs shifted by 1 bar native.

Entry: When velocity > threshold (strong momentum detected)
Exit: Fixed bars or when velocity drops below threshold
Target: Was the continuation profitable?
"""
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime

def log(msg=""):
    print(msg, flush=True)

log("=" * 70)
log("PREPARING MOMENTUM CONTINUATION DATASET - M30")
log("=" * 70)
log(f"Started: {datetime.now()}")

DATA_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/data')
CLEANED_DIR = DATA_DIR / 'cleaned'
OUTPUT_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/lstm_model/v2_both_currencies/v3_regression/data')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CURRENCIES = ['EUR', 'USD', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD', 'NZD']

# Momentum parameters
VELOCITY_THRESHOLD = 0.10  # Minimum velocity to trigger
HOLD_BARS = 6  # Fixed hold period (6 M30 bars = 3 hours)

# Pip sizes
PIP_SIZE = {
    'EURUSD': 0.0001, 'GBPUSD': 0.0001, 'AUDUSD': 0.0001, 'NZDUSD': 0.0001,
    'USDCAD': 0.0001, 'USDCHF': 0.0001, 'USDJPY': 0.01,
    'EURGBP': 0.0001, 'EURJPY': 0.01, 'EURCHF': 0.0001, 'EURCAD': 0.0001,
    'EURAUD': 0.0001, 'EURNZD': 0.0001,
    'GBPJPY': 0.01, 'GBPCHF': 0.0001, 'GBPCAD': 0.0001, 'GBPAUD': 0.0001, 'GBPNZD': 0.0001,
    'AUDJPY': 0.01, 'AUDCHF': 0.0001, 'AUDCAD': 0.0001, 'AUDNZD': 0.0001,
    'NZDJPY': 0.01, 'NZDCHF': 0.0001, 'NZDCAD': 0.0001,
    'CADJPY': 0.01, 'CADCHF': 0.0001, 'CHFJPY': 0.01,
}

# Spreads (Exness Standard)
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


def load_mfc_cleaned(currency, timeframe):
    """Load cleaned MFC data."""
    fp = CLEANED_DIR / f'mfc_currency_{currency}_{timeframe}_clean.csv'
    if not fp.exists():
        return None
    df = pd.read_csv(fp)
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df = df.set_index('datetime').sort_index()
    return df['MFC']


def load_price_data(pair, timeframe):
    """Load price data."""
    fp = DATA_DIR / f'{pair}_GMT+0_US-DST_{timeframe}.csv'
    if not fp.exists():
        return None
    df = pd.read_csv(fp, header=None,
                     names=['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df = df.set_index('datetime').sort_index()
    return df


def create_dataset_for_pair(base_ccy, quote_ccy):
    """Create momentum dataset for a currency pair."""
    pair = f"{base_ccy}{quote_ccy}"

    if pair not in PIP_SIZE:
        return None

    log(f"\nProcessing {pair}...")

    # Load all timeframes
    base_mfc = {}
    quote_mfc = {}

    for tf in ['M30', 'H1', 'H4']:
        base_mfc[tf] = load_mfc_cleaned(base_ccy, tf)
        quote_mfc[tf] = load_mfc_cleaned(quote_ccy, tf)

    if base_mfc['M30'] is None or quote_mfc['M30'] is None:
        log(f"  Missing M30 data")
        return None

    # Load price
    price_df = load_price_data(pair, 'M30')
    if price_df is None:
        log(f"  Missing price data")
        return None

    # Create base dataframe on M30 grid
    df = pd.DataFrame(index=base_mfc['M30'].index)
    df['price_open'] = price_df['Open']
    df['price_close'] = price_df['Close']

    # =========================================================================
    # SHIFT ALL TIMEFRAMES (no leakage)
    # Each TF shifted by 1 bar of its native TF, velocity in native TF
    # =========================================================================

    for tf in ['M30', 'H1', 'H4']:
        tf_lower = tf.lower()

        base_raw = base_mfc[tf]
        quote_raw = quote_mfc[tf]

        if base_raw is not None:
            # MFC: shift by 1 native bar
            base_level = base_raw.shift(1)
            # Velocity: diff then shift
            base_vel = base_raw.diff().shift(1)
            # Momentum (2-bar velocity in native TF)
            base_mom = base_raw.diff(2).shift(1)

            if tf == 'M30':
                df[f'base_{tf_lower}'] = base_level
                df[f'base_vel_{tf_lower}'] = base_vel
                df[f'base_mom_{tf_lower}'] = base_mom
            else:
                df[f'base_{tf_lower}'] = base_level.reindex(df.index, method='ffill')
                df[f'base_vel_{tf_lower}'] = base_vel.reindex(df.index, method='ffill')
                df[f'base_mom_{tf_lower}'] = base_mom.reindex(df.index, method='ffill')

        if quote_raw is not None:
            quote_level = quote_raw.shift(1)
            quote_vel = quote_raw.diff().shift(1)
            quote_mom = quote_raw.diff(2).shift(1)

            if tf == 'M30':
                df[f'quote_{tf_lower}'] = quote_level
                df[f'quote_vel_{tf_lower}'] = quote_vel
                df[f'quote_mom_{tf_lower}'] = quote_mom
            else:
                df[f'quote_{tf_lower}'] = quote_level.reindex(df.index, method='ffill')
                df[f'quote_vel_{tf_lower}'] = quote_vel.reindex(df.index, method='ffill')
                df[f'quote_mom_{tf_lower}'] = quote_mom.reindex(df.index, method='ffill')

    df = df.dropna()

    # =========================================================================
    # ADDITIONAL FEATURES
    # =========================================================================

    # Acceleration
    df['base_acc_m30'] = df['base_vel_m30'].diff()
    df['quote_acc_m30'] = df['quote_vel_m30'].diff()

    # Divergence
    df['divergence'] = df['base_m30'] - df['quote_m30']
    df['vel_divergence'] = df['base_vel_m30'] - df['quote_vel_m30']

    # Position in box
    df['base_in_box'] = ((df['base_m30'] >= -0.2) & (df['base_m30'] <= 0.2)).astype(int)
    df['quote_in_box'] = ((df['quote_m30'] >= -0.2) & (df['quote_m30'] <= 0.2)).astype(int)

    # Distance from box
    df['base_dist_box'] = df['base_m30'].apply(lambda x: max(0, abs(x) - 0.2))
    df['quote_dist_box'] = df['quote_m30'].apply(lambda x: max(0, abs(x) - 0.2))

    df = df.dropna()
    log(f"  Data: {len(df)} M30 bars ({df.index[0].date()} to {df.index[-1].date()})")

    # =========================================================================
    # FIND MOMENTUM SIGNALS
    # =========================================================================

    all_entries = []
    pip_size = PIP_SIZE[pair]
    spread = SPREADS.get(pair, 2.0)

    # Convert to numpy for speed
    price_open = df['price_open'].values
    price_close = df['price_close'].values
    base_vel = df['base_vel_m30'].values
    quote_vel = df['quote_vel_m30'].values

    n = len(df)

    for trigger_type in ['base', 'quote']:
        trigger_ccy = base_ccy if trigger_type == 'base' else quote_ccy
        vel_arr = base_vel if trigger_type == 'base' else quote_vel

        for i in range(n - HOLD_BARS - 1):
            vel = vel_arr[i]

            # Check for high velocity
            if abs(vel) < VELOCITY_THRESHOLD:
                continue

            # Determine direction
            if trigger_type == 'base':
                if vel > VELOCITY_THRESHOLD:
                    direction = 'buy'  # Base strengthening -> buy pair
                    expected_sign = 1
                else:
                    direction = 'sell'  # Base weakening -> sell pair
                    expected_sign = -1
            else:  # quote trigger
                if vel > VELOCITY_THRESHOLD:
                    direction = 'sell'  # Quote strengthening -> sell pair
                    expected_sign = -1
                else:
                    direction = 'buy'  # Quote weakening -> buy pair
                    expected_sign = 1

            # Entry at next bar open
            entry_price = price_open[i + 1]

            # Exit at HOLD_BARS later close
            exit_price = price_close[i + 1 + HOLD_BARS]

            # Calculate pips
            raw_pips = (exit_price - entry_price) / pip_size
            adjusted_pips = raw_pips * expected_sign
            net_pips = adjusted_pips - spread

            # Get all features at signal time
            row = df.iloc[i]

            entry = {
                'datetime': df.index[i],
                'pair': pair,
                'base_ccy': base_ccy,
                'quote_ccy': quote_ccy,
                'trigger': trigger_type,
                'trigger_ccy': trigger_ccy,
                'direction': direction,
                'entry_vel': abs(vel),
                'raw_pips': raw_pips,
                'adjusted_pips': adjusted_pips,
                'net_pips': net_pips,
                'is_profitable': int(net_pips > 0),
                # Features
                'base_m30': row['base_m30'],
                'base_vel_m30': row['base_vel_m30'],
                'base_mom_m30': row['base_mom_m30'],
                'base_acc_m30': row['base_acc_m30'],
                'base_h1': row['base_h1'],
                'base_vel_h1': row['base_vel_h1'],
                'base_mom_h1': row['base_mom_h1'],
                'base_h4': row['base_h4'],
                'base_vel_h4': row['base_vel_h4'],
                'base_mom_h4': row['base_mom_h4'],
                'quote_m30': row['quote_m30'],
                'quote_vel_m30': row['quote_vel_m30'],
                'quote_mom_m30': row['quote_mom_m30'],
                'quote_acc_m30': row['quote_acc_m30'],
                'quote_h1': row['quote_h1'],
                'quote_vel_h1': row['quote_vel_h1'],
                'quote_mom_h1': row['quote_mom_h1'],
                'quote_h4': row['quote_h4'],
                'quote_vel_h4': row['quote_vel_h4'],
                'quote_mom_h4': row['quote_mom_h4'],
                'divergence': row['divergence'],
                'vel_divergence': row['vel_divergence'],
                'base_in_box': row['base_in_box'],
                'quote_in_box': row['quote_in_box'],
                'base_dist_box': row['base_dist_box'],
                'quote_dist_box': row['quote_dist_box'],
            }

            all_entries.append(entry)

    if not all_entries:
        return None

    result_df = pd.DataFrame(all_entries)

    profitable_rate = result_df['is_profitable'].mean() * 100
    avg_net = result_df['net_pips'].mean()

    log(f"  Entries: {len(result_df):,}")
    log(f"  Profitable: {profitable_rate:.1f}%, Avg net pips: {avg_net:+.2f}")

    return result_df


def main():
    pairs = [
        ('EUR', 'USD'), ('EUR', 'GBP'), ('EUR', 'JPY'), ('EUR', 'CHF'),
        ('EUR', 'CAD'), ('EUR', 'AUD'), ('EUR', 'NZD'),
        ('GBP', 'USD'), ('GBP', 'JPY'), ('GBP', 'CHF'), ('GBP', 'CAD'),
        ('GBP', 'AUD'), ('GBP', 'NZD'),
        ('USD', 'JPY'), ('USD', 'CHF'), ('USD', 'CAD'),
        ('AUD', 'USD'), ('AUD', 'JPY'), ('AUD', 'CHF'), ('AUD', 'CAD'), ('AUD', 'NZD'),
        ('NZD', 'USD'), ('NZD', 'JPY'), ('NZD', 'CHF'), ('NZD', 'CAD'),
        ('CAD', 'JPY'), ('CAD', 'CHF'), ('CHF', 'JPY'),
    ]

    all_entries = []
    for base_ccy, quote_ccy in pairs:
        result = create_dataset_for_pair(base_ccy, quote_ccy)
        if result is not None:
            all_entries.append(result)

    log("\n" + "=" * 70)
    log("COMBINING ALL DATA")
    log("=" * 70)

    final_df = pd.concat(all_entries, ignore_index=True)

    log(f"\nTotal entries: {len(final_df):,}")
    log(f"Profitable: {final_df['is_profitable'].sum():,} ({final_df['is_profitable'].mean()*100:.1f}%)")
    log(f"Avg net pips: {final_df['net_pips'].mean():+.2f}")

    # By trigger type
    log(f"\nBy Trigger:")
    for trigger in ['base', 'quote']:
        subset = final_df[final_df['trigger'] == trigger]
        log(f"  {trigger.upper()}: {len(subset):,}, {subset['is_profitable'].mean()*100:.1f}% profitable, {subset['net_pips'].mean():+.2f} avg")

    # By direction
    log(f"\nBy Direction:")
    for direction in ['buy', 'sell']:
        subset = final_df[final_df['direction'] == direction]
        log(f"  {direction.upper()}: {len(subset):,}, {subset['is_profitable'].mean()*100:.1f}% profitable")

    # By velocity strength
    log(f"\nBy Entry Velocity:")
    for low, high, label in [(0.10, 0.15, '0.10-0.15'), (0.15, 0.20, '0.15-0.20'),
                              (0.20, 0.30, '0.20-0.30'), (0.30, 1.0, '>0.30')]:
        subset = final_df[(final_df['entry_vel'] >= low) & (final_df['entry_vel'] < high)]
        if len(subset) > 0:
            log(f"  {label}: {len(subset):,}, {subset['is_profitable'].mean()*100:.1f}% profitable, {subset['net_pips'].mean():+.2f} avg")

    # Save
    output_path = OUTPUT_DIR / 'momentum_data_m30.pkl'
    output = {
        'data': final_df,
        'config': {
            'velocity_threshold': VELOCITY_THRESHOLD,
            'hold_bars': HOLD_BARS,
            'base_timeframe': 'M30',
            'version': 'v1_momentum',
        }
    }

    with open(output_path, 'wb') as f:
        pickle.dump(output, f)

    log(f"\nSaved to: {output_path}")
    log(f"Completed: {datetime.now()}")


if __name__ == '__main__':
    main()
