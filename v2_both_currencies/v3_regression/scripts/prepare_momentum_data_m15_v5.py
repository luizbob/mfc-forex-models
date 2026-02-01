"""
Prepare Momentum Dataset - M15 Signal - V5
===========================================
V5: Add MACD from H1, H4, D1 as trend confirmation features
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
log("PREPARING MOMENTUM DATASET - M15 V5 (with MACD)")
log("=" * 70)
log(f"Started: {datetime.now()}")

DATA_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/data')
CLEANED_DIR = DATA_DIR / 'cleaned'
OUTPUT_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/lstm_model/v2_both_currencies/v3_regression/data')

VELOCITY_THRESHOLD = 0.08
EXIT_THRESHOLD = 0.02
MAX_HOLD_BARS = 24

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


def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD, Signal, and Histogram"""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist


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
    with open(fp, 'r') as f:
        first_line = f.readline()
    if 'Date' in first_line or 'Open' in first_line:
        df = pd.read_csv(fp)
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

    # Load MFC data
    base_mfc = {}
    quote_mfc = {}
    for tf in ['M15', 'M30', 'H1', 'H4']:
        base_mfc[tf] = load_mfc_cleaned(base_ccy, tf)
        quote_mfc[tf] = load_mfc_cleaned(quote_ccy, tf)

    if base_mfc['M15'] is None or quote_mfc['M15'] is None:
        log(f"  Missing M15 MFC data")
        return None

    # Load price data for different timeframes
    price_m15 = load_price_data(pair, 'M15')
    if price_m15 is None:
        price_m1 = load_price_data(pair, 'M1')
        if price_m1 is None:
            log(f"  Missing M15 price data")
            return None
        price_m15 = price_m1.resample('15min').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
        }).dropna()

    # Load H1, H4, D1 price for MACD
    price_h1 = load_price_data(pair, 'H1')
    price_h4 = load_price_data(pair, 'H4')
    price_d1 = load_price_data(pair, 'D1')

    # If missing, resample from M15
    if price_h1 is None:
        price_h1 = price_m15.resample('1h').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
        }).dropna()
    if price_h4 is None:
        price_h4 = price_m15.resample('4h').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
        }).dropna()
    if price_d1 is None:
        price_d1 = price_m15.resample('1D').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
        }).dropna()

    # Calculate MACD for each timeframe
    macd_h1, signal_h1, hist_h1 = calculate_macd(price_h1['Close'])
    macd_h4, signal_h4, hist_h4 = calculate_macd(price_h4['Close'])
    macd_d1, signal_d1, hist_d1 = calculate_macd(price_d1['Close'])

    # Create base dataframe on M15 index
    df = pd.DataFrame(index=base_mfc['M15'].index)
    df['price_open'] = price_m15['Open'].reindex(df.index, method='ffill')
    df['price_close'] = price_m15['Close'].reindex(df.index, method='ffill')

    # Raw velocity for exit tracking
    df['base_vel_raw'] = base_mfc['M15'].diff()
    df['quote_vel_raw'] = quote_mfc['M15'].diff()

    # Shift MFC for features (no look-ahead)
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

    # Add MACD features (shifted by 1 to avoid look-ahead)
    # H1 MACD
    df['macd_h1'] = macd_h1.shift(1).reindex(df.index, method='ffill')
    df['macd_signal_h1'] = signal_h1.shift(1).reindex(df.index, method='ffill')
    df['macd_hist_h1'] = hist_h1.shift(1).reindex(df.index, method='ffill')
    df['macd_above_signal_h1'] = (df['macd_h1'] > df['macd_signal_h1']).astype(int)
    df['macd_above_zero_h1'] = (df['macd_h1'] > 0).astype(int)

    # H4 MACD
    df['macd_h4'] = macd_h4.shift(1).reindex(df.index, method='ffill')
    df['macd_signal_h4'] = signal_h4.shift(1).reindex(df.index, method='ffill')
    df['macd_hist_h4'] = hist_h4.shift(1).reindex(df.index, method='ffill')
    df['macd_above_signal_h4'] = (df['macd_h4'] > df['macd_signal_h4']).astype(int)
    df['macd_above_zero_h4'] = (df['macd_h4'] > 0).astype(int)

    # D1 MACD
    df['macd_d1'] = macd_d1.shift(1).reindex(df.index, method='ffill')
    df['macd_signal_d1'] = signal_d1.shift(1).reindex(df.index, method='ffill')
    df['macd_hist_d1'] = hist_d1.shift(1).reindex(df.index, method='ffill')
    df['macd_above_signal_d1'] = (df['macd_d1'] > df['macd_signal_d1']).astype(int)
    df['macd_above_zero_d1'] = (df['macd_d1'] > 0).astype(int)

    df = df.dropna()

    # Additional MFC features
    df['base_acc_m15'] = df['base_vel_m15'].diff()
    df['quote_acc_m15'] = df['quote_vel_m15'].diff()
    df['divergence'] = df['base_m15'] - df['quote_m15']
    df['vel_divergence'] = df['base_vel_m15'] - df['quote_vel_m15']
    df['base_in_box'] = ((df['base_m15'] >= -0.2) & (df['base_m15'] <= 0.2)).astype(int)
    df['quote_in_box'] = ((df['quote_m15'] >= -0.2) & (df['quote_m15'] <= 0.2)).astype(int)
    df['base_dist_box'] = df['base_m15'].apply(lambda x: max(0, abs(x) - 0.2))
    df['quote_dist_box'] = df['quote_m15'].apply(lambda x: max(0, abs(x) - 0.2))
    df['h4_base_confirms_up'] = ((df['base_h4'] < 0.2) & (df['base_vel_h4'] > 0)).astype(int)
    df['h4_base_confirms_down'] = ((df['base_h4'] > -0.2) & (df['base_vel_h4'] < 0)).astype(int)
    df['h4_quote_confirms_up'] = ((df['quote_h4'] < 0.2) & (df['quote_vel_h4'] > 0)).astype(int)
    df['h4_quote_confirms_down'] = ((df['quote_h4'] > -0.2) & (df['quote_vel_h4'] < 0)).astype(int)

    df = df.dropna()
    log(f"  Data: {len(df)} M15 bars")

    # Find signals
    all_entries = []
    pip_size = PIP_SIZE[pair]
    spread = SPREADS.get(pair, 2.0)

    price_open = df['price_open'].values
    price_close = df['price_close'].values
    base_vel_shifted = df['base_vel_m15'].values
    quote_vel_shifted = df['quote_vel_m15'].values
    base_vel_raw = df['base_vel_raw'].values
    quote_vel_raw = df['quote_vel_raw'].values

    n = len(df)
    i = 0

    while i < n - MAX_HOLD_BARS - 1:
        base_vel = base_vel_shifted[i]
        quote_vel = quote_vel_shifted[i]

        signal = None
        trigger_type = None
        expected_sign = 0

        if base_vel > VELOCITY_THRESHOLD:
            signal = 'buy'
            trigger_type = 'base'
            expected_sign = 1
        elif base_vel < -VELOCITY_THRESHOLD:
            signal = 'sell'
            trigger_type = 'base'
            expected_sign = -1
        elif quote_vel > VELOCITY_THRESHOLD:
            signal = 'sell'
            trigger_type = 'quote'
            expected_sign = -1
        elif quote_vel < -VELOCITY_THRESHOLD:
            signal = 'buy'
            trigger_type = 'quote'
            expected_sign = 1

        if signal is None:
            i += 1
            continue

        entry_idx = i + 1
        entry_price = price_open[entry_idx]

        exit_idx = entry_idx
        vel_to_track = base_vel_raw if trigger_type == 'base' else quote_vel_raw

        for j in range(entry_idx, min(entry_idx + MAX_HOLD_BARS, n)):
            current_vel = vel_to_track[j]
            if trigger_type == 'base':
                if signal == 'buy':
                    if current_vel < EXIT_THRESHOLD:
                        exit_idx = j
                        break
                else:
                    if current_vel > -EXIT_THRESHOLD:
                        exit_idx = j
                        break
            else:
                if signal == 'sell':
                    if current_vel < EXIT_THRESHOLD:
                        exit_idx = j
                        break
                else:
                    if current_vel > -EXIT_THRESHOLD:
                        exit_idx = j
                        break
            exit_idx = j

        exit_price = price_close[exit_idx]
        bars_held = exit_idx - entry_idx + 1

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
            'trigger_ccy': base_ccy if trigger_type == 'base' else quote_ccy,
            'direction': signal,
            'entry_vel': abs(base_vel if trigger_type == 'base' else quote_vel),
            'bars_held': bars_held,
            'raw_pips': raw_pips,
            'adjusted_pips': adjusted_pips,
            'net_pips': net_pips,
            'is_profitable': int(net_pips > 0),
            # MFC Features
            'base_m15': row['base_m15'],
            'base_vel_m15': row['base_vel_m15'],
            'base_mom_m15': row['base_mom_m15'],
            'base_acc_m15': row['base_acc_m15'],
            'quote_m15': row['quote_m15'],
            'quote_vel_m15': row['quote_vel_m15'],
            'quote_mom_m15': row['quote_mom_m15'],
            'quote_acc_m15': row['quote_acc_m15'],
            'base_m30': row['base_m30'],
            'base_vel_m30': row['base_vel_m30'],
            'base_mom_m30': row['base_mom_m30'],
            'quote_m30': row['quote_m30'],
            'quote_vel_m30': row['quote_vel_m30'],
            'quote_mom_m30': row['quote_mom_m30'],
            'base_h1': row['base_h1'],
            'base_vel_h1': row['base_vel_h1'],
            'base_mom_h1': row['base_mom_h1'],
            'quote_h1': row['quote_h1'],
            'quote_vel_h1': row['quote_vel_h1'],
            'quote_mom_h1': row['quote_mom_h1'],
            'base_h4': row['base_h4'],
            'base_vel_h4': row['base_vel_h4'],
            'base_mom_h4': row['base_mom_h4'],
            'quote_h4': row['quote_h4'],
            'quote_vel_h4': row['quote_vel_h4'],
            'quote_mom_h4': row['quote_mom_h4'],
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
            # MACD Features
            'macd_h1': row['macd_h1'],
            'macd_signal_h1': row['macd_signal_h1'],
            'macd_hist_h1': row['macd_hist_h1'],
            'macd_above_signal_h1': row['macd_above_signal_h1'],
            'macd_above_zero_h1': row['macd_above_zero_h1'],
            'macd_h4': row['macd_h4'],
            'macd_signal_h4': row['macd_signal_h4'],
            'macd_hist_h4': row['macd_hist_h4'],
            'macd_above_signal_h4': row['macd_above_signal_h4'],
            'macd_above_zero_h4': row['macd_above_zero_h4'],
            'macd_d1': row['macd_d1'],
            'macd_signal_d1': row['macd_signal_d1'],
            'macd_hist_d1': row['macd_hist_d1'],
            'macd_above_signal_d1': row['macd_above_signal_d1'],
            'macd_above_zero_d1': row['macd_above_zero_d1'],
        }

        all_entries.append(entry)
        i = exit_idx + 1

    if not all_entries:
        return None

    result_df = pd.DataFrame(all_entries)
    avg_bars = result_df['bars_held'].mean()
    log(f"  Entries: {len(result_df):,}, Profitable: {result_df['is_profitable'].mean()*100:.1f}%, Avg hold: {avg_bars:.1f} bars")

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
    log(f"Profitable: {final_df['is_profitable'].mean()*100:.1f}%")
    log(f"Avg net pips: {final_df['net_pips'].mean():+.2f}")
    log(f"Avg bars held: {final_df['bars_held'].mean():.1f}")

    log(f"\nBy Bars Held:")
    for low, high, label in [(1, 3, '1-2 bars'), (3, 6, '3-5 bars'), (6, 12, '6-11 bars'), (12, 25, '12+ bars')]:
        subset = final_df[(final_df['bars_held'] >= low) & (final_df['bars_held'] < high)]
        if len(subset) > 0:
            log(f"  {label}: {len(subset):,}, {subset['is_profitable'].mean()*100:.1f}%, {subset['net_pips'].mean():+.2f}")

    # Check MACD alignment with direction
    log(f"\nMACD Alignment Check:")
    buys = final_df[final_df['direction'] == 'buy']
    sells = final_df[final_df['direction'] == 'sell']

    buy_macd_aligned = buys[buys['macd_above_zero_h4'] == 1]
    sell_macd_aligned = sells[sells['macd_above_zero_h4'] == 0]

    log(f"  Buys with H4 MACD > 0: {len(buy_macd_aligned):,} ({len(buy_macd_aligned)/len(buys)*100:.1f}%)")
    log(f"    Profitable: {buy_macd_aligned['is_profitable'].mean()*100:.1f}%, Avg: {buy_macd_aligned['net_pips'].mean():+.2f}")
    log(f"  Buys with H4 MACD < 0: {len(buys) - len(buy_macd_aligned):,}")
    buy_macd_not_aligned = buys[buys['macd_above_zero_h4'] == 0]
    log(f"    Profitable: {buy_macd_not_aligned['is_profitable'].mean()*100:.1f}%, Avg: {buy_macd_not_aligned['net_pips'].mean():+.2f}")

    log(f"\n  Sells with H4 MACD < 0: {len(sell_macd_aligned):,} ({len(sell_macd_aligned)/len(sells)*100:.1f}%)")
    log(f"    Profitable: {sell_macd_aligned['is_profitable'].mean()*100:.1f}%, Avg: {sell_macd_aligned['net_pips'].mean():+.2f}")
    log(f"  Sells with H4 MACD > 0: {len(sells) - len(sell_macd_aligned):,}")
    sell_macd_not_aligned = sells[sells['macd_above_zero_h4'] == 1]
    log(f"    Profitable: {sell_macd_not_aligned['is_profitable'].mean()*100:.1f}%, Avg: {sell_macd_not_aligned['net_pips'].mean():+.2f}")

    # Save
    output_path = OUTPUT_DIR / 'momentum_data_m15_v5.pkl'
    output = {
        'data': final_df,
        'config': {
            'velocity_threshold': VELOCITY_THRESHOLD,
            'exit_threshold': EXIT_THRESHOLD,
            'max_hold_bars': MAX_HOLD_BARS,
            'exit_type': 'velocity_drop',
            'base_timeframe': 'M15',
            'macd_timeframes': ['H1', 'H4', 'D1'],
            'version': 'v5_momentum_m15_macd',
        }
    }

    with open(output_path, 'wb') as f:
        pickle.dump(output, f)

    log(f"\nSaved to: {output_path}")
    log(f"Completed: {datetime.now()}")


if __name__ == '__main__':
    main()
