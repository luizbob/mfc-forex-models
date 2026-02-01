"""
Prepare Momentum Continuation Dataset - M15 Signal
===================================================
Signal on M15 velocity, features from M15/M30/H1/H4.
All timeframes properly shifted.
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
log("PREPARING MOMENTUM DATASET - M15 SIGNAL")
log("=" * 70)
log(f"Started: {datetime.now()}")

DATA_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/data')
CLEANED_DIR = DATA_DIR / 'cleaned'
OUTPUT_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/lstm_model/v2_both_currencies/v3_regression/data')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CURRENCIES = ['EUR', 'USD', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD', 'NZD']

VELOCITY_THRESHOLD = 0.08  # Lower threshold for M15 (faster signals)
HOLD_BARS = 8  # 8 M15 bars = 2 hours

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
    fp = CLEANED_DIR / f'mfc_currency_{currency}_{timeframe}_clean.csv'
    if not fp.exists():
        return None
    df = pd.read_csv(fp)
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df = df.set_index('datetime').sort_index()
    return df['MFC']


def load_price_data(pair, timeframe):
    fp = DATA_DIR / f'{pair}_GMT+0_US-DST_{timeframe}.csv'
    if not fp.exists():
        return None
    # Check if file has header
    with open(fp, 'r') as f:
        first_line = f.readline()
    if 'Date' in first_line or 'Open' in first_line:
        df = pd.read_csv(fp)
        # Handle different column name formats
        if 'Tick volume' in df.columns:
            df = df.rename(columns={'Tick volume': 'Volume'})
    else:
        df = pd.read_csv(fp, header=None,
                         names=['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df = df.set_index('datetime').sort_index()
    return df


def create_dataset_for_pair(base_ccy, quote_ccy):
    pair = f"{base_ccy}{quote_ccy}"

    if pair not in PIP_SIZE:
        return None

    log(f"\nProcessing {pair}...")

    # Load all timeframes
    base_mfc = {}
    quote_mfc = {}

    for tf in ['M15', 'M30', 'H1', 'H4']:
        base_mfc[tf] = load_mfc_cleaned(base_ccy, tf)
        quote_mfc[tf] = load_mfc_cleaned(quote_ccy, tf)

    if base_mfc['M15'] is None or quote_mfc['M15'] is None:
        log(f"  Missing M15 data")
        return None

    # Load M15 price
    price_df = load_price_data(pair, 'M15')
    if price_df is None:
        # Try to resample from M1
        price_m1 = load_price_data(pair, 'M1')
        if price_m1 is None:
            log(f"  Missing price data")
            return None
        price_df = price_m1.resample('15min').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
        }).dropna()

    # Create base dataframe on M15 grid
    df = pd.DataFrame(index=base_mfc['M15'].index)
    df['price_open'] = price_df['Open'].reindex(df.index, method='ffill')
    df['price_close'] = price_df['Close'].reindex(df.index, method='ffill')

    # =========================================================================
    # SHIFT ALL TIMEFRAMES
    # =========================================================================

    for tf in ['M15', 'M30', 'H1', 'H4']:
        tf_lower = tf.lower()

        base_raw = base_mfc[tf]
        quote_raw = quote_mfc[tf]

        if base_raw is not None:
            base_level = base_raw.shift(1)
            base_vel = base_raw.diff().shift(1)
            base_mom = base_raw.diff(2).shift(1)

            if tf == 'M15':
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

            if tf == 'M15':
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
    df['base_acc_m15'] = df['base_vel_m15'].diff()
    df['quote_acc_m15'] = df['quote_vel_m15'].diff()

    # Divergence
    df['divergence'] = df['base_m15'] - df['quote_m15']
    df['vel_divergence'] = df['base_vel_m15'] - df['quote_vel_m15']

    # Position features
    df['base_in_box'] = ((df['base_m15'] >= -0.2) & (df['base_m15'] <= 0.2)).astype(int)
    df['quote_in_box'] = ((df['quote_m15'] >= -0.2) & (df['quote_m15'] <= 0.2)).astype(int)
    df['base_dist_box'] = df['base_m15'].apply(lambda x: max(0, abs(x) - 0.2))
    df['quote_dist_box'] = df['quote_m15'].apply(lambda x: max(0, abs(x) - 0.2))

    # HTF confirmation features
    df['h4_base_confirms_up'] = ((df['base_h4'] < 0.2) & (df['base_vel_h4'] > 0)).astype(int)
    df['h4_base_confirms_down'] = ((df['base_h4'] > -0.2) & (df['base_vel_h4'] < 0)).astype(int)
    df['h4_quote_confirms_up'] = ((df['quote_h4'] < 0.2) & (df['quote_vel_h4'] > 0)).astype(int)
    df['h4_quote_confirms_down'] = ((df['quote_h4'] > -0.2) & (df['quote_vel_h4'] < 0)).astype(int)

    df = df.dropna()
    log(f"  Data: {len(df)} M15 bars ({df.index[0].date()} to {df.index[-1].date()})")

    # =========================================================================
    # FIND MOMENTUM SIGNALS ON M15
    # =========================================================================

    all_entries = []
    pip_size = PIP_SIZE[pair]
    spread = SPREADS.get(pair, 2.0)

    price_open = df['price_open'].values
    price_close = df['price_close'].values
    base_vel = df['base_vel_m15'].values
    quote_vel = df['quote_vel_m15'].values

    n = len(df)

    for trigger_type in ['base', 'quote']:
        trigger_ccy = base_ccy if trigger_type == 'base' else quote_ccy
        vel_arr = base_vel if trigger_type == 'base' else quote_vel

        for i in range(n - HOLD_BARS - 1):
            vel = vel_arr[i]

            if abs(vel) < VELOCITY_THRESHOLD:
                continue

            if trigger_type == 'base':
                if vel > VELOCITY_THRESHOLD:
                    direction = 'buy'
                    expected_sign = 1
                else:
                    direction = 'sell'
                    expected_sign = -1
            else:
                if vel > VELOCITY_THRESHOLD:
                    direction = 'sell'
                    expected_sign = -1
                else:
                    direction = 'buy'
                    expected_sign = 1

            entry_price = price_open[i + 1]
            exit_price = price_close[i + 1 + HOLD_BARS]

            raw_pips = (exit_price - entry_price) / pip_size
            adjusted_pips = raw_pips * expected_sign
            net_pips = adjusted_pips - spread

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
                # M15 features
                'base_m15': row['base_m15'],
                'base_vel_m15': row['base_vel_m15'],
                'base_mom_m15': row['base_mom_m15'],
                'base_acc_m15': row['base_acc_m15'],
                'quote_m15': row['quote_m15'],
                'quote_vel_m15': row['quote_vel_m15'],
                'quote_mom_m15': row['quote_mom_m15'],
                'quote_acc_m15': row['quote_acc_m15'],
                # M30 features
                'base_m30': row['base_m30'],
                'base_vel_m30': row['base_vel_m30'],
                'base_mom_m30': row['base_mom_m30'],
                'quote_m30': row['quote_m30'],
                'quote_vel_m30': row['quote_vel_m30'],
                'quote_mom_m30': row['quote_mom_m30'],
                # H1 features
                'base_h1': row['base_h1'],
                'base_vel_h1': row['base_vel_h1'],
                'base_mom_h1': row['base_mom_h1'],
                'quote_h1': row['quote_h1'],
                'quote_vel_h1': row['quote_vel_h1'],
                'quote_mom_h1': row['quote_mom_h1'],
                # H4 features
                'base_h4': row['base_h4'],
                'base_vel_h4': row['base_vel_h4'],
                'base_mom_h4': row['base_mom_h4'],
                'quote_h4': row['quote_h4'],
                'quote_vel_h4': row['quote_vel_h4'],
                'quote_mom_h4': row['quote_mom_h4'],
                # Derived
                'divergence': row['divergence'],
                'vel_divergence': row['vel_divergence'],
                'base_in_box': row['base_in_box'],
                'quote_in_box': row['quote_in_box'],
                'base_dist_box': row['base_dist_box'],
                'quote_dist_box': row['quote_dist_box'],
                'h4_base_confirms_up': row['h4_base_confirms_up'],
                'h4_base_confirms_down': row['h4_base_confirms_down'],
                'h4_quote_confirms_up': row['h4_quote_confirms_up'],
                'h4_quote_confirms_down': row['h4_quote_confirms_down'],
            }

            all_entries.append(entry)

    if not all_entries:
        return None

    result_df = pd.DataFrame(all_entries)
    log(f"  Entries: {len(result_df):,}, Profitable: {result_df['is_profitable'].mean()*100:.1f}%")

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

    log(f"\nBy Entry Velocity:")
    for low, high, label in [(0.08, 0.12, '0.08-0.12'), (0.12, 0.18, '0.12-0.18'),
                              (0.18, 0.25, '0.18-0.25'), (0.25, 1.0, '>0.25')]:
        subset = final_df[(final_df['entry_vel'] >= low) & (final_df['entry_vel'] < high)]
        if len(subset) > 0:
            log(f"  {label}: {len(subset):,}, {subset['is_profitable'].mean()*100:.1f}%, {subset['net_pips'].mean():+.2f}")

    # Save
    output_path = OUTPUT_DIR / 'momentum_data_m15.pkl'
    output = {
        'data': final_df,
        'config': {
            'velocity_threshold': VELOCITY_THRESHOLD,
            'hold_bars': HOLD_BARS,
            'base_timeframe': 'M15',
            'version': 'v1_momentum_m15',
        }
    }

    with open(output_path, 'wb') as f:
        pickle.dump(output, f)

    log(f"\nSaved to: {output_path}")
    log(f"Completed: {datetime.now()}")


if __name__ == '__main__':
    main()
