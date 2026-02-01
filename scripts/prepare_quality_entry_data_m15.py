"""
Prepare Quality Entry Dataset for M15 Base Timeframe
=====================================================
Same concept but uses M15 as decision timeframe.
Even faster signals than M30.
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
log("PREPARING QUALITY ENTRY DATASET (M15 Base)")
log("=" * 70)
log(f"Started: {datetime.now()}")

DATA_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/data')
OUTPUT_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/lstm_model/data')

CURRENCIES = ['EUR', 'USD', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD', 'NZD']

# Quality entry parameters
MFC_EXTREME_THRESHOLD = 0.5
MAX_ADVERSE_MFC = 0.25
RETURN_TARGET = 0.0
MAX_BARS_TO_RETURN = 800  # M15 bars (200 hours, same time window)


def load_mfc_data(currency, timeframe):
    """Load MFC data for a currency and timeframe."""
    suffix = '' if timeframe == 'H1' else f'_{timeframe}'
    fp = DATA_DIR / f'mfc_currency_{currency}{suffix}.csv'
    if not fp.exists():
        return None
    df = pd.read_csv(fp)
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df = df.set_index('datetime').sort_index()
    return df['MFC']


def load_price_data(pair):
    """Load price data for a pair."""
    fp = DATA_DIR / f'{pair}_GMT+0_US-DST_M1.csv'
    if not fp.exists():
        return None
    df = pd.read_csv(fp)
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df = df.set_index('datetime').sort_index()
    return df['Close']


def create_dataset_for_pair(base_ccy, quote_ccy):
    """Create quality entry dataset for a currency pair using M15 base."""
    pair = f"{base_ccy}{quote_ccy}"
    log(f"\nProcessing {pair}...")

    # Load all timeframes
    base_mfc = {}
    quote_mfc = {}

    for tf in ['M5', 'M15', 'M30', 'H1', 'H4']:
        base_mfc[tf] = load_mfc_data(base_ccy, tf)
        quote_mfc[tf] = load_mfc_data(quote_ccy, tf)

    if base_mfc['M15'] is None or quote_mfc['M15'] is None:
        log(f"  Missing M15 data")
        return None

    # Load price
    price = load_price_data(pair)
    if price is None:
        log(f"  Missing price data")
        return None

    price_m15 = price.resample('15min').last().dropna()

    # Create base dataframe on M15
    df = pd.DataFrame(index=base_mfc['M15'].index)
    df['base_m15'] = base_mfc['M15']
    df['quote_m15'] = quote_mfc['M15']
    df['price'] = price_m15

    # Forward fill other timeframes to M15 index
    if base_mfc['M5'] is not None:
        df['base_m5'] = base_mfc['M5'].reindex(df.index, method='ffill')
    if quote_mfc['M5'] is not None:
        df['quote_m5'] = quote_mfc['M5'].reindex(df.index, method='ffill')

    for tf in ['M30', 'H1', 'H4']:
        if base_mfc[tf] is not None:
            df[f'base_{tf.lower()}'] = base_mfc[tf].reindex(df.index, method='ffill')
        if quote_mfc[tf] is not None:
            df[f'quote_{tf.lower()}'] = quote_mfc[tf].reindex(df.index, method='ffill')

    df = df.dropna()

    # Apply shift(1)
    for tf in ['m5', 'm15', 'm30', 'h1', 'h4']:
        df[f'base_{tf}_shifted'] = df[f'base_{tf}'].shift(1)
        df[f'quote_{tf}_shifted'] = df[f'quote_{tf}'].shift(1)

    # Calculate velocities
    for tf in ['m5', 'm15', 'm30', 'h1', 'h4']:
        df[f'base_vel_{tf}'] = df[f'base_{tf}_shifted'].diff()
        df[f'quote_vel_{tf}'] = df[f'quote_{tf}_shifted'].diff()

    # Additional M15 features
    df['base_vel2_m15'] = df['base_m15_shifted'].diff(2)
    df['base_acc_m15'] = df['base_vel_m15'].diff()

    # Divergence
    df['divergence'] = df['base_m15_shifted'] - df['quote_m15_shifted']
    df['vel_divergence'] = df['base_vel_m15'] - df['quote_vel_m15']

    df = df.dropna()
    log(f"  Data: {len(df)} M15 bars ({df.index[0].date()} to {df.index[-1].date()})")

    # Find and label entries
    entries = []

    # BUY entries
    buy_mask = df['base_m15_shifted'] <= -MFC_EXTREME_THRESHOLD
    for idx in df[buy_mask].index:
        entry = label_entry(df, idx, direction='buy')
        if entry is not None:
            entry['pair'] = pair
            entry['base_ccy'] = base_ccy
            entry['quote_ccy'] = quote_ccy
            entries.append(entry)

    # SELL entries
    sell_mask = df['base_m15_shifted'] >= MFC_EXTREME_THRESHOLD
    for idx in df[sell_mask].index:
        entry = label_entry(df, idx, direction='sell')
        if entry is not None:
            entry['pair'] = pair
            entry['base_ccy'] = base_ccy
            entry['quote_ccy'] = quote_ccy
            entries.append(entry)

    if len(entries) == 0:
        return None

    result_df = pd.DataFrame(entries)
    buy_count = len(result_df[result_df['direction'] == 'buy'])
    sell_count = len(result_df[result_df['direction'] == 'sell'])
    quality_rate = result_df['is_quality'].mean() * 100

    log(f"  Entries: {len(result_df)} (BUY: {buy_count}, SELL: {sell_count})")
    log(f"  Quality rate: {quality_rate:.1f}%")

    return result_df


def label_entry(df, idx, direction):
    """Label an entry as quality (1) or not (0)."""
    try:
        pos = df.index.get_loc(idx)
        if pos + MAX_BARS_TO_RETURN >= len(df):
            return None

        entry_mfc = df.iloc[pos]['base_m15_shifted']
        entry_price = df.iloc[pos]['price']

        future_mfc = df.iloc[pos+1:pos+MAX_BARS_TO_RETURN+1]['base_m15']
        future_price = df.iloc[pos+1:pos+MAX_BARS_TO_RETURN+1]['price']

        if direction == 'buy':
            min_mfc = future_mfc.min()
            adverse_move = entry_mfc - min_mfc
            returned = (future_mfc >= RETURN_TARGET).any()
            is_quality = returned and (adverse_move < MAX_ADVERSE_MFC)
            max_dd_pips = (entry_price - future_price.min()) * 10000
            max_profit_pips = (future_price.max() - entry_price) * 10000
        else:
            max_mfc = future_mfc.max()
            adverse_move = max_mfc - entry_mfc
            returned = (future_mfc <= RETURN_TARGET).any()
            is_quality = returned and (adverse_move < MAX_ADVERSE_MFC)
            max_dd_pips = (future_price.max() - entry_price) * 10000
            max_profit_pips = (entry_price - future_price.min()) * 10000

        row = df.iloc[pos]

        features = {
            'datetime': idx,
            'direction': direction,
            'is_quality': int(is_quality),
            'returned': int(returned),
            'adverse_move': adverse_move,
            'max_dd_pips': max_dd_pips,
            'max_profit_pips': max_profit_pips,
        }

        for tf in ['m5', 'm15', 'm30', 'h1', 'h4']:
            features[f'base_{tf}'] = row[f'base_{tf}_shifted']
            features[f'quote_{tf}'] = row[f'quote_{tf}_shifted']
            features[f'base_vel_{tf}'] = row[f'base_vel_{tf}']
            features[f'quote_vel_{tf}'] = row[f'quote_vel_{tf}']

        features['base_vel2_m15'] = row['base_vel2_m15']
        features['base_acc_m15'] = row['base_acc_m15']
        features['divergence'] = row['divergence']
        features['vel_divergence'] = row['vel_divergence']

        return features
    except:
        return None


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
    log(f"Quality entries: {final_df['is_quality'].sum():,} ({final_df['is_quality'].mean()*100:.1f}%)")

    for direction in ['buy', 'sell']:
        subset = final_df[final_df['direction'] == direction]
        log(f"\n{direction.upper()}: {len(subset):,} entries, {subset['is_quality'].mean()*100:.1f}% quality")

    # Save
    output_path = OUTPUT_DIR / 'quality_entry_data_m15.pkl'
    output = {
        'data': final_df,
        'config': {
            'mfc_extreme_threshold': MFC_EXTREME_THRESHOLD,
            'max_adverse_mfc': MAX_ADVERSE_MFC,
            'return_target': RETURN_TARGET,
            'max_bars_to_return': MAX_BARS_TO_RETURN,
            'base_timeframe': 'M15',
        }
    }

    with open(output_path, 'wb') as f:
        pickle.dump(output, f)

    log(f"\nSaved to: {output_path}")
    log(f"Completed: {datetime.now()}")


if __name__ == '__main__':
    main()
