"""
Prepare Quality Entry Dataset for M30 Base Timeframe
=====================================================
Same concept as H1 version but uses M30 as decision timeframe.
This gives 2x more data points and faster signals.

Quality Entry = Entry where MFC returns to center (0) WITHOUT going
much further against you first (minimal drawdown).
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
log("PREPARING QUALITY ENTRY DATASET (M30 Base)")
log("=" * 70)
log(f"Started: {datetime.now()}")

# Paths
DATA_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/data')
OUTPUT_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/lstm_model/data')
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

CURRENCIES = ['EUR', 'USD', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD', 'NZD']

# Quality entry parameters (same as H1 version)
MFC_EXTREME_THRESHOLD = 0.5  # Entry zone: MFC <= -0.5 (buy) or >= 0.5 (sell)
MAX_ADVERSE_MFC = 0.25       # Max allowed adverse MFC move before return
RETURN_TARGET = 0.0          # Target: MFC returns to 0 (center)
MAX_BARS_TO_RETURN = 400     # Max M30 bars to wait (200 hours, same time window as H1)


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
    """
    Create quality entry dataset for a currency pair.
    Uses M30 as base timeframe, includes ALL timeframes (M5, M15, M30, H1, H4).
    Applies shift to avoid lookahead.
    """
    pair = f"{base_ccy}{quote_ccy}"
    log(f"\nProcessing {pair}...")

    # Load all timeframes for base and quote
    base_mfc = {}
    quote_mfc = {}

    for tf in ['M5', 'M15', 'M30', 'H1', 'H4']:
        base_mfc[tf] = load_mfc_data(base_ccy, tf)
        quote_mfc[tf] = load_mfc_data(quote_ccy, tf)

    if base_mfc['M30'] is None or quote_mfc['M30'] is None:
        log(f"  Missing M30 data")
        return None

    # Load price
    price = load_price_data(pair)
    if price is None:
        log(f"  Missing price data")
        return None

    price_m30 = price.resample('30min').last().dropna()

    # Create base dataframe on M30 (decision timeframe)
    df = pd.DataFrame(index=base_mfc['M30'].index)
    df['base_m30'] = base_mfc['M30']
    df['quote_m30'] = quote_mfc['M30']
    df['price'] = price_m30

    # Forward fill M5 and M15 to M30 index
    if base_mfc['M5'] is not None:
        df['base_m5'] = base_mfc['M5'].reindex(df.index, method='ffill')
    if quote_mfc['M5'] is not None:
        df['quote_m5'] = quote_mfc['M5'].reindex(df.index, method='ffill')
    if base_mfc['M15'] is not None:
        df['base_m15'] = base_mfc['M15'].reindex(df.index, method='ffill')
    if quote_mfc['M15'] is not None:
        df['quote_m15'] = quote_mfc['M15'].reindex(df.index, method='ffill')

    # Forward fill H1 and H4 to M30 index
    if base_mfc['H1'] is not None:
        df['base_h1'] = base_mfc['H1'].reindex(df.index, method='ffill')
    if quote_mfc['H1'] is not None:
        df['quote_h1'] = quote_mfc['H1'].reindex(df.index, method='ffill')
    if base_mfc['H4'] is not None:
        df['base_h4'] = base_mfc['H4'].reindex(df.index, method='ffill')
    if quote_mfc['H4'] is not None:
        df['quote_h4'] = quote_mfc['H4'].reindex(df.index, method='ffill')

    df = df.dropna()

    # IMPORTANT: Apply shift(1) - we can only see data from previous bar
    for tf in ['m5', 'm15', 'm30', 'h1', 'h4']:
        df[f'base_{tf}_shifted'] = df[f'base_{tf}'].shift(1)
        df[f'quote_{tf}_shifted'] = df[f'quote_{tf}'].shift(1)

    # Calculate velocities on shifted data (all timeframes)
    for tf in ['m5', 'm15', 'm30', 'h1', 'h4']:
        df[f'base_vel_{tf}'] = df[f'base_{tf}_shifted'].diff()
        df[f'quote_vel_{tf}'] = df[f'quote_{tf}_shifted'].diff()

    # 2-bar velocity and acceleration for M30 (base timeframe)
    df['base_vel2_m30'] = df['base_m30_shifted'].diff(2)
    df['base_acc_m30'] = df['base_vel_m30'].diff()

    # Divergence (using shifted M30)
    df['divergence'] = df['base_m30_shifted'] - df['quote_m30_shifted']
    df['vel_divergence'] = df['base_vel_m30'] - df['quote_vel_m30']

    df = df.dropna()
    log(f"  Data: {len(df)} M30 bars ({df.index[0].date()} to {df.index[-1].date()})")

    # Find and label entries
    entries = []

    # BUY entries: base MFC at extreme low (using shifted = what we can see)
    buy_mask = df['base_m30_shifted'] <= -MFC_EXTREME_THRESHOLD

    for idx in df[buy_mask].index:
        entry = label_entry(df, idx, direction='buy')
        if entry is not None:
            entry['pair'] = pair
            entry['base_ccy'] = base_ccy
            entry['quote_ccy'] = quote_ccy
            entries.append(entry)

    # SELL entries: base MFC at extreme high
    sell_mask = df['base_m30_shifted'] >= MFC_EXTREME_THRESHOLD

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
    """
    Label an entry as quality (1) or not (0).

    Quality = MFC returns to 0 without going much further against first.
    """
    try:
        pos = df.index.get_loc(idx)

        if pos + MAX_BARS_TO_RETURN >= len(df):
            return None

        # Entry values (shifted = what we see at decision time)
        entry_mfc = df.iloc[pos]['base_m30_shifted']
        entry_price = df.iloc[pos]['price']

        # Future MFC (actual values, not shifted - this is what happens)
        future_mfc = df.iloc[pos+1:pos+MAX_BARS_TO_RETURN+1]['base_m30']
        future_price = df.iloc[pos+1:pos+MAX_BARS_TO_RETURN+1]['price']

        if direction == 'buy':
            # For buy: MFC should return to 0 without dropping much more
            min_mfc = future_mfc.min()
            adverse_move = entry_mfc - min_mfc  # Positive = went further down

            returned = (future_mfc >= RETURN_TARGET).any()
            is_quality = returned and (adverse_move < MAX_ADVERSE_MFC)

            # Price metrics
            max_dd_pips = (entry_price - future_price.min()) * 10000
            max_profit_pips = (future_price.max() - entry_price) * 10000

        else:  # sell
            # For sell: MFC should return to 0 without rising much more
            max_mfc = future_mfc.max()
            adverse_move = max_mfc - entry_mfc  # Positive = went further up

            returned = (future_mfc <= RETURN_TARGET).any()
            is_quality = returned and (adverse_move < MAX_ADVERSE_MFC)

            max_dd_pips = (future_price.max() - entry_price) * 10000
            max_profit_pips = (entry_price - future_price.min()) * 10000

        # Features at entry (all from shifted data = what we can see)
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

        # MFC values for all timeframes (shifted)
        for tf in ['m5', 'm15', 'm30', 'h1', 'h4']:
            features[f'base_{tf}'] = row[f'base_{tf}_shifted']
            features[f'quote_{tf}'] = row[f'quote_{tf}_shifted']
            features[f'base_vel_{tf}'] = row[f'base_vel_{tf}']
            features[f'quote_vel_{tf}'] = row[f'quote_vel_{tf}']

        # Additional M30 features (base timeframe)
        features['base_vel2_m30'] = row['base_vel2_m30']
        features['base_acc_m30'] = row['base_acc_m30']
        features['divergence'] = row['divergence']
        features['vel_divergence'] = row['vel_divergence']

        return features

    except Exception as e:
        return None


def main():
    # All 28 pairs
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

    # Combine all
    log("\n" + "=" * 70)
    log("COMBINING ALL DATA")
    log("=" * 70)

    final_df = pd.concat(all_entries, ignore_index=True)

    log(f"\nTotal entries: {len(final_df)}")
    log(f"Quality entries: {final_df['is_quality'].sum()} ({final_df['is_quality'].mean()*100:.1f}%)")
    log(f"Returned (regardless of DD): {final_df['returned'].mean()*100:.1f}%")

    # By direction
    for direction in ['buy', 'sell']:
        subset = final_df[final_df['direction'] == direction]
        log(f"\n{direction.upper()}:")
        log(f"  Count: {len(subset)}")
        log(f"  Quality: {subset['is_quality'].mean()*100:.1f}%")
        log(f"  Avg adverse MFC: {subset['adverse_move'].mean():.3f}")

    # Feature comparison: quality vs non-quality
    log("\n" + "=" * 70)
    log("QUALITY vs NON-QUALITY COMPARISON")
    log("=" * 70)

    quality = final_df[final_df['is_quality'] == 1]
    non_quality = final_df[final_df['is_quality'] == 0]

    for col in ['base_vel_m30', 'base_vel2_m30', 'base_acc_m30', 'divergence', 'vel_divergence']:
        q_mean = quality[col].mean()
        nq_mean = non_quality[col].mean()
        log(f"  {col:15s}: Quality={q_mean:+.4f}, Non-Q={nq_mean:+.4f}")

    # Save
    log("\n" + "=" * 70)
    log("SAVING")
    log("=" * 70)

    output_path = OUTPUT_DIR / 'quality_entry_data_m30.pkl'

    output = {
        'data': final_df,
        'config': {
            'mfc_extreme_threshold': MFC_EXTREME_THRESHOLD,
            'max_adverse_mfc': MAX_ADVERSE_MFC,
            'return_target': RETURN_TARGET,
            'max_bars_to_return': MAX_BARS_TO_RETURN,
            'base_timeframe': 'M30',
        }
    }

    with open(output_path, 'wb') as f:
        pickle.dump(output, f)

    log(f"Saved to: {output_path}")
    log(f"\nCompleted: {datetime.now()}")


if __name__ == '__main__':
    main()
