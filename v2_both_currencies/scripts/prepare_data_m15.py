"""
Prepare Quality Entry Dataset - V2 Both Currencies
===================================================
Triggers on BOTH base AND quote currency extremes.

Entry triggers:
- BUY when base_mfc <= -0.5 (base weak, will recover up)
- SELL when base_mfc >= +0.5 (base strong, will pull back)
- SELL when quote_mfc <= -0.5 (quote weak, will recover -> pair DOWN)
- BUY when quote_mfc >= +0.5 (quote strong, will pull back -> pair UP)

Uses cleaned MFC data and M1 price data.
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
log("PREPARING QUALITY ENTRY DATASET - V2 BOTH CURRENCIES (M15)")
log("=" * 70)
log(f"Started: {datetime.now()}")

DATA_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/data')
CLEANED_DIR = DATA_DIR / 'cleaned'
OUTPUT_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/lstm_model/v2_both_currencies/data')

CURRENCIES = ['EUR', 'USD', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD', 'NZD']

# Quality entry parameters
MFC_EXTREME_THRESHOLD = 0.5
MAX_ADVERSE_MFC = 0.25
RETURN_TARGET = 0.0
MAX_BARS_TO_RETURN = 800  # M15 bars


def load_mfc_cleaned(currency, timeframe):
    """Load cleaned MFC data for a currency and timeframe."""
    fp = CLEANED_DIR / f'mfc_currency_{currency}_{timeframe}_clean.csv'
    if not fp.exists():
        return None
    df = pd.read_csv(fp)
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df = df.set_index('datetime').sort_index()
    return df['MFC']


def load_price_data(pair):
    """Load M1 price data for a pair (Open and Close)."""
    fp = DATA_DIR / f'{pair}_GMT+0_US-DST_M1.csv'
    if not fp.exists():
        return None
    df = pd.read_csv(fp)
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df = df.set_index('datetime').sort_index()
    return df[['Open', 'Close']]


def create_dataset_for_pair(base_ccy, quote_ccy):
    """Create quality entry dataset for a currency pair using both currency extremes."""
    pair = f"{base_ccy}{quote_ccy}"
    log(f"\nProcessing {pair}...")

    # Load all timeframes (cleaned data)
    base_mfc = {}
    quote_mfc = {}

    for tf in ['M5', 'M15', 'M30', 'H1', 'H4']:
        base_mfc[tf] = load_mfc_cleaned(base_ccy, tf)
        quote_mfc[tf] = load_mfc_cleaned(quote_ccy, tf)

    if base_mfc['M15'] is None or quote_mfc['M15'] is None:
        log(f"  Missing M15 data")
        return None

    # Load price (M1)
    price_df = load_price_data(pair)
    if price_df is None:
        log(f"  Missing price data")
        return None

    # Resample to M15: open = first, close = last
    price_m15_open = price_df['Open'].resample('15min').first().dropna()
    price_m15_close = price_df['Close'].resample('15min').last().dropna()

    # Create base dataframe on M15
    df = pd.DataFrame(index=base_mfc['M15'].index)
    df['base_m15'] = base_mfc['M15']
    df['quote_m15'] = quote_mfc['M15']
    df['price_open'] = price_m15_open
    df['price_close'] = price_m15_close

    # Add other timeframes
    # IMPORTANT: Shift by 1 bar of THAT timeframe BEFORE ffill to use LAST CLOSED bar
    # This prevents data leakage for both higher TFs (H1, H4) and lower TFs (M5)
    for tf in ['M5', 'M30', 'H1', 'H4']:
        if base_mfc[tf] is not None:
            df[f'base_{tf.lower()}'] = base_mfc[tf].shift(1).reindex(df.index, method='ffill')
        if quote_mfc[tf] is not None:
            df[f'quote_{tf.lower()}'] = quote_mfc[tf].shift(1).reindex(df.index, method='ffill')

    df = df.dropna()

    # Apply shift(1) on M15 grid - use previous M15 bar's data for trigger/features
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
    df['quote_vel2_m15'] = df['quote_m15_shifted'].diff(2)
    df['quote_acc_m15'] = df['quote_vel_m15'].diff()

    # Divergence
    df['divergence'] = df['base_m15_shifted'] - df['quote_m15_shifted']
    df['vel_divergence'] = df['base_vel_m15'] - df['quote_vel_m15']

    df = df.dropna()
    log(f"  Data: {len(df)} M15 bars ({df.index[0].date()} to {df.index[-1].date()})")

    # Find and label entries
    entries = []

    # === BASE CURRENCY TRIGGERS ===
    # BUY when base at extreme LOW (base weak, will recover)
    buy_mask_base = df['base_m15_shifted'] <= -MFC_EXTREME_THRESHOLD
    for idx in df[buy_mask_base].index:
        entry = label_entry(df, idx, direction='buy', trigger='base', trigger_ccy=base_ccy)
        if entry is not None:
            entry['pair'] = pair
            entry['base_ccy'] = base_ccy
            entry['quote_ccy'] = quote_ccy
            entries.append(entry)

    # SELL when base at extreme HIGH (base strong, will pull back)
    sell_mask_base = df['base_m15_shifted'] >= MFC_EXTREME_THRESHOLD
    for idx in df[sell_mask_base].index:
        entry = label_entry(df, idx, direction='sell', trigger='base', trigger_ccy=base_ccy)
        if entry is not None:
            entry['pair'] = pair
            entry['base_ccy'] = base_ccy
            entry['quote_ccy'] = quote_ccy
            entries.append(entry)

    # === QUOTE CURRENCY TRIGGERS ===
    # SELL when quote at extreme LOW (quote weak, will recover -> pair DOWN)
    sell_mask_quote = df['quote_m15_shifted'] <= -MFC_EXTREME_THRESHOLD
    for idx in df[sell_mask_quote].index:
        entry = label_entry(df, idx, direction='sell', trigger='quote', trigger_ccy=quote_ccy)
        if entry is not None:
            entry['pair'] = pair
            entry['base_ccy'] = base_ccy
            entry['quote_ccy'] = quote_ccy
            entries.append(entry)

    # BUY when quote at extreme HIGH (quote strong, will pull back -> pair UP)
    buy_mask_quote = df['quote_m15_shifted'] >= MFC_EXTREME_THRESHOLD
    for idx in df[buy_mask_quote].index:
        entry = label_entry(df, idx, direction='buy', trigger='quote', trigger_ccy=quote_ccy)
        if entry is not None:
            entry['pair'] = pair
            entry['base_ccy'] = base_ccy
            entry['quote_ccy'] = quote_ccy
            entries.append(entry)

    if len(entries) == 0:
        return None

    result_df = pd.DataFrame(entries)

    # Count by trigger type
    base_triggers = len(result_df[result_df['trigger'] == 'base'])
    quote_triggers = len(result_df[result_df['trigger'] == 'quote'])
    quality_rate = result_df['is_quality'].mean() * 100

    log(f"  Entries: {len(result_df)} (Base triggers: {base_triggers}, Quote triggers: {quote_triggers})")
    log(f"  Quality rate: {quality_rate:.1f}%")

    return result_df


def label_entry(df, idx, direction, trigger, trigger_ccy):
    """Label an entry as quality (1) or not (0)."""
    try:
        pos = df.index.get_loc(idx)
        if pos + 1 + MAX_BARS_TO_RETURN >= len(df):
            return None

        # Entry at NEXT bar's open (realistic execution)
        entry_price = df.iloc[pos+1]['price_open']

        # For quality check, use the TRIGGER currency's MFC
        if trigger == 'base':
            entry_mfc = df.iloc[pos]['base_m15_shifted']
            future_mfc = df.iloc[pos+1:pos+MAX_BARS_TO_RETURN+1]['base_m15']
        else:  # quote trigger
            entry_mfc = df.iloc[pos]['quote_m15_shifted']
            future_mfc = df.iloc[pos+1:pos+MAX_BARS_TO_RETURN+1]['quote_m15']

        # Future prices for drawdown/profit calculation (use close prices)
        future_price = df.iloc[pos+1:pos+MAX_BARS_TO_RETURN+1]['price_close']

        # Quality is based on the TRIGGER currency returning to center
        if trigger == 'base':
            if direction == 'buy':  # base was low, expecting up
                min_mfc = future_mfc.min()
                adverse_move = entry_mfc - min_mfc
                returned = (future_mfc >= RETURN_TARGET).any()
            else:  # sell, base was high, expecting down
                max_mfc = future_mfc.max()
                adverse_move = max_mfc - entry_mfc
                returned = (future_mfc <= RETURN_TARGET).any()
        else:  # quote trigger
            if direction == 'sell':  # quote was low, expecting quote up (pair down)
                min_mfc = future_mfc.min()
                adverse_move = entry_mfc - min_mfc
                returned = (future_mfc >= RETURN_TARGET).any()
            else:  # buy, quote was high, expecting quote down (pair up)
                max_mfc = future_mfc.max()
                adverse_move = max_mfc - entry_mfc
                returned = (future_mfc <= RETURN_TARGET).any()

        is_quality = returned and (adverse_move < MAX_ADVERSE_MFC)

        # Price-based metrics (always from pair direction perspective)
        if direction == 'buy':
            max_dd_pips = (entry_price - future_price.min()) * 10000
            max_profit_pips = (future_price.max() - entry_price) * 10000
        else:
            max_dd_pips = (future_price.max() - entry_price) * 10000
            max_profit_pips = (entry_price - future_price.min()) * 10000

        # Find ACTUAL exit: price when trigger MFC crosses 0
        # This is the realistic trade PnL
        exit_pnl_pips = 0.0
        bars_to_exit = MAX_BARS_TO_RETURN  # default: timeout

        if trigger == 'base':
            if direction == 'buy':
                # Exit when base_mfc >= 0
                exit_mask = future_mfc >= 0
            else:
                # Exit when base_mfc <= 0
                exit_mask = future_mfc <= 0
        else:  # quote trigger
            if direction == 'sell':
                # Exit when quote_mfc >= 0
                exit_mask = future_mfc >= 0
            else:
                # Exit when quote_mfc <= 0
                exit_mask = future_mfc <= 0

        if exit_mask.any():
            # First bar where MFC crosses 0
            exit_idx = exit_mask.idxmax()
            exit_bar_pos = future_mfc.index.get_loc(exit_idx)
            bars_to_exit = exit_bar_pos + 1
            exit_price = future_price.iloc[exit_bar_pos]
        else:
            # Timeout: exit at last bar's price
            exit_price = future_price.iloc[-1]

        if direction == 'buy':
            exit_pnl_pips = (exit_price - entry_price) * 10000
        else:
            exit_pnl_pips = (entry_price - exit_price) * 10000

        row = df.iloc[pos]

        features = {
            'datetime': idx,
            'direction': direction,
            'trigger': trigger,
            'trigger_ccy': trigger_ccy,
            'is_quality': int(is_quality),
            'returned': int(returned),
            'adverse_move': adverse_move,
            'max_dd_pips': max_dd_pips,
            'max_profit_pips': max_profit_pips,
            'exit_pnl_pips': exit_pnl_pips,
            'bars_to_exit': bars_to_exit,
        }

        # All MFC features (shifted)
        for tf in ['m5', 'm15', 'm30', 'h1', 'h4']:
            features[f'base_{tf}'] = row[f'base_{tf}_shifted']
            features[f'quote_{tf}'] = row[f'quote_{tf}_shifted']
            features[f'base_vel_{tf}'] = row[f'base_vel_{tf}']
            features[f'quote_vel_{tf}'] = row[f'quote_vel_{tf}']

        features['base_vel2_m15'] = row['base_vel2_m15']
        features['base_acc_m15'] = row['base_acc_m15']
        features['quote_vel2_m15'] = row['quote_vel2_m15']
        features['quote_acc_m15'] = row['quote_acc_m15']
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

    # By trigger type
    log(f"\nBy Trigger:")
    for trigger in ['base', 'quote']:
        subset = final_df[final_df['trigger'] == trigger]
        log(f"  {trigger.upper()}: {len(subset):,} entries, {subset['is_quality'].mean()*100:.1f}% quality")

    # By direction
    log(f"\nBy Direction:")
    for direction in ['buy', 'sell']:
        subset = final_df[final_df['direction'] == direction]
        log(f"  {direction.upper()}: {len(subset):,} entries, {subset['is_quality'].mean()*100:.1f}% quality")

    # By trigger currency
    log(f"\nBy Trigger Currency:")
    for ccy in CURRENCIES:
        subset = final_df[final_df['trigger_ccy'] == ccy]
        if len(subset) > 0:
            log(f"  {ccy}: {len(subset):,} entries, {subset['is_quality'].mean()*100:.1f}% quality")

    # Save
    output_path = OUTPUT_DIR / 'quality_entry_data_m15_v2.pkl'
    output = {
        'data': final_df,
        'config': {
            'mfc_extreme_threshold': MFC_EXTREME_THRESHOLD,
            'max_adverse_mfc': MAX_ADVERSE_MFC,
            'return_target': RETURN_TARGET,
            'max_bars_to_return': MAX_BARS_TO_RETURN,
            'base_timeframe': 'M15',
            'version': 'v2_both_currencies',
            'triggers': ['base', 'quote'],
        }
    }

    with open(output_path, 'wb') as f:
        pickle.dump(output, f)

    log(f"\nSaved to: {output_path}")
    log(f"Completed: {datetime.now()}")


if __name__ == '__main__':
    main()
