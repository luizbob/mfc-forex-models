"""
Test Price ROC as Feature
=========================
Use shifted close to avoid look-ahead bias.
At bar i, we only know close[i-1] (previous bar completed).
"""
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from xgboost import XGBClassifier

def log(msg=""):
    print(msg, flush=True)

log("=" * 70)
log("TEST PRICE ROC AS FEATURE")
log("=" * 70)

DATA_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/data')
CLEANED_DIR = DATA_DIR / 'cleaned'

VELOCITY_THRESHOLD = 0.08
EXIT_THRESHOLD = 0.02
MAX_HOLD_BARS = 24

PIP_SIZE = {
    'EURUSD': 0.0001, 'GBPUSD': 0.0001, 'USDJPY': 0.01, 'AUDUSD': 0.0001,
    'EURJPY': 0.01, 'GBPJPY': 0.01, 'EURGBP': 0.0001, 'USDCAD': 0.0001,
}

SPREADS = {
    'EURUSD': 0.8, 'GBPUSD': 1.0, 'USDJPY': 1.0, 'AUDUSD': 0.9,
    'EURJPY': 2.4, 'GBPJPY': 2.2, 'EURGBP': 1.4, 'USDCAD': 1.5,
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


def calculate_roc(close, periods):
    """Rate of Change: (close - close[N]) / close[N] * 100"""
    return (close - close.shift(periods)) / close.shift(periods) * 100


def create_dataset_for_pair(base_ccy, quote_ccy):
    pair = f"{base_ccy}{quote_ccy}"
    if pair not in PIP_SIZE:
        return None

    base_mfc = load_mfc_cleaned(base_ccy, 'M15')
    quote_mfc = load_mfc_cleaned(quote_ccy, 'M15')
    if base_mfc is None or quote_mfc is None:
        return None

    # Load price data
    price_m15 = load_price_data(pair, 'M15')
    if price_m15 is None:
        price_m1 = load_price_data(pair, 'M1')
        if price_m1 is None:
            return None
        price_m15 = price_m1.resample('15min').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
        }).dropna()

    price_h1 = load_price_data(pair, 'H1')
    price_h4 = load_price_data(pair, 'H4')

    if price_h1 is None:
        price_h1 = price_m15.resample('1h').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
        }).dropna()
    if price_h4 is None:
        price_h4 = price_m15.resample('4h').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
        }).dropna()

    # Create dataframe
    df = pd.DataFrame(index=base_mfc.index)
    df['price_open'] = price_m15['Open'].reindex(df.index, method='ffill')
    df['price_close'] = price_m15['Close'].reindex(df.index, method='ffill')

    # MFC features (shifted)
    df['base_vel'] = base_mfc.diff().shift(1)
    df['quote_vel'] = quote_mfc.diff().shift(1)
    df['base_vel_raw'] = base_mfc.diff()
    df['quote_vel_raw'] = quote_mfc.diff()

    # Price ROC - SHIFTED by 1 to use only completed bars
    # ROC on H1: 4-bar ROC (4 hours of movement)
    roc_h1_4 = calculate_roc(price_h1['Close'], 4).shift(1)
    roc_h1_8 = calculate_roc(price_h1['Close'], 8).shift(1)

    # ROC on H4: 2-bar and 4-bar ROC
    roc_h4_2 = calculate_roc(price_h4['Close'], 2).shift(1)
    roc_h4_4 = calculate_roc(price_h4['Close'], 4).shift(1)

    df['roc_h1_4'] = roc_h1_4.reindex(df.index, method='ffill')
    df['roc_h1_8'] = roc_h1_8.reindex(df.index, method='ffill')
    df['roc_h4_2'] = roc_h4_2.reindex(df.index, method='ffill')
    df['roc_h4_4'] = roc_h4_4.reindex(df.index, method='ffill')

    # Direction alignment: ROC positive = price going up
    df['roc_h1_positive'] = (df['roc_h1_4'] > 0).astype(int)
    df['roc_h4_positive'] = (df['roc_h4_2'] > 0).astype(int)

    df = df.dropna()

    # Find signals
    pip_size = PIP_SIZE[pair]
    spread = SPREADS.get(pair, 2.0)

    all_entries = []
    n = len(df)
    i = 0

    while i < n - MAX_HOLD_BARS - 1:
        base_v = df['base_vel'].iloc[i]
        quote_v = df['quote_vel'].iloc[i]

        signal = None
        trigger_type = None
        expected_sign = 0

        if base_v > VELOCITY_THRESHOLD:
            signal = 'buy'
            trigger_type = 'base'
            expected_sign = 1
        elif base_v < -VELOCITY_THRESHOLD:
            signal = 'sell'
            trigger_type = 'base'
            expected_sign = -1
        elif quote_v > VELOCITY_THRESHOLD:
            signal = 'sell'
            trigger_type = 'quote'
            expected_sign = -1
        elif quote_v < -VELOCITY_THRESHOLD:
            signal = 'buy'
            trigger_type = 'quote'
            expected_sign = 1

        if signal is None:
            i += 1
            continue

        entry_idx = i + 1
        entry_price = df['price_open'].iloc[entry_idx]

        exit_idx = entry_idx
        vel_col = 'base_vel_raw' if trigger_type == 'base' else 'quote_vel_raw'

        for j in range(entry_idx, min(entry_idx + MAX_HOLD_BARS, n)):
            current_vel = df[vel_col].iloc[j]
            if trigger_type == 'base':
                if signal == 'buy' and current_vel < EXIT_THRESHOLD:
                    exit_idx = j
                    break
                elif signal == 'sell' and current_vel > -EXIT_THRESHOLD:
                    exit_idx = j
                    break
            else:
                if signal == 'sell' and current_vel < EXIT_THRESHOLD:
                    exit_idx = j
                    break
                elif signal == 'buy' and current_vel > -EXIT_THRESHOLD:
                    exit_idx = j
                    break
            exit_idx = j

        exit_price = df['price_close'].iloc[exit_idx]
        bars_held = exit_idx - entry_idx + 1

        raw_pips = (exit_price - entry_price) / pip_size
        adjusted_pips = raw_pips * expected_sign
        net_pips = adjusted_pips - spread

        row = df.iloc[i]

        all_entries.append({
            'datetime': df.index[i],
            'pair': pair,
            'direction': signal,
            'bars_held': bars_held,
            'net_pips': net_pips,
            'is_profitable': int(net_pips > 0),
            'base_vel': row['base_vel'],
            'quote_vel': row['quote_vel'],
            'roc_h1_4': row['roc_h1_4'],
            'roc_h1_8': row['roc_h1_8'],
            'roc_h4_2': row['roc_h4_2'],
            'roc_h4_4': row['roc_h4_4'],
            'roc_h1_positive': row['roc_h1_positive'],
            'roc_h4_positive': row['roc_h4_positive'],
        })

        i = exit_idx + 1

    if not all_entries:
        return None

    return pd.DataFrame(all_entries)


# Test on major pairs
pairs = [
    ('EUR', 'USD'), ('GBP', 'USD'), ('USD', 'JPY'), ('AUD', 'USD'),
    ('EUR', 'JPY'), ('GBP', 'JPY'), ('EUR', 'GBP'), ('USD', 'CAD'),
]

log("\nProcessing 8 major pairs...")
all_data = []
for base, quote in pairs:
    result = create_dataset_for_pair(base, quote)
    if result is not None:
        all_data.append(result)
        log(f"  {base}{quote}: {len(result):,} entries")

df = pd.concat(all_data, ignore_index=True)
df['datetime'] = pd.to_datetime(df['datetime'])
df['year'] = df['datetime'].dt.year

log(f"\nTotal entries: {len(df):,}")

# Check ROC alignment
log("\n" + "=" * 70)
log("ROC ALIGNMENT CHECK (raw signals)")
log("=" * 70)

buys = df[df['direction'] == 'buy']
sells = df[df['direction'] == 'sell']

log("\nBUYS:")
buy_aligned = buys[buys['roc_h4_positive'] == 1]
buy_not_aligned = buys[buys['roc_h4_positive'] == 0]
log(f"  H4 ROC positive (aligned): {len(buy_aligned):,}, {buy_aligned['is_profitable'].mean()*100:.1f}% WR, {buy_aligned['net_pips'].mean():+.2f} avg")
log(f"  H4 ROC negative (counter): {len(buy_not_aligned):,}, {buy_not_aligned['is_profitable'].mean()*100:.1f}% WR, {buy_not_aligned['net_pips'].mean():+.2f} avg")

log("\nSELLS:")
sell_aligned = sells[sells['roc_h4_positive'] == 0]
sell_not_aligned = sells[sells['roc_h4_positive'] == 1]
log(f"  H4 ROC negative (aligned): {len(sell_aligned):,}, {sell_aligned['is_profitable'].mean()*100:.1f}% WR, {sell_aligned['net_pips'].mean():+.2f} avg")
log(f"  H4 ROC positive (counter): {len(sell_not_aligned):,}, {sell_not_aligned['is_profitable'].mean()*100:.1f}% WR, {sell_not_aligned['net_pips'].mean():+.2f} avg")

# Walk-forward test
log("\n" + "=" * 70)
log("WALK-FORWARD: WITH vs WITHOUT ROC")
log("=" * 70)

feature_cols_no_roc = ['base_vel', 'quote_vel']
feature_cols_with_roc = ['base_vel', 'quote_vel', 'roc_h1_4', 'roc_h1_8', 'roc_h4_2', 'roc_h4_4']

df = df[df['datetime'] <= '2025-12-21'].copy()

results_no_roc = []
results_with_roc = []

for test_year in [2020, 2021, 2022, 2023, 2024, 2025]:
    train_mask = df['year'] < test_year
    test_mask = df['year'] == test_year

    if train_mask.sum() < 1000 or test_mask.sum() < 100:
        continue

    y_train = df[train_mask]['is_profitable'].values

    # Without ROC
    X_train_no = df[train_mask][feature_cols_no_roc].values.astype(np.float32)
    X_test_no = df[test_mask][feature_cols_no_roc].values.astype(np.float32)
    X_train_no = np.nan_to_num(X_train_no)
    X_test_no = np.nan_to_num(X_test_no)

    model_no = XGBClassifier(n_estimators=100, max_depth=3, random_state=42, n_jobs=-1)
    model_no.fit(X_train_no, y_train, verbose=False)

    df_test = df[test_mask].copy()
    df_test['pred'] = model_no.predict_proba(X_test_no)[:, 1]
    filtered_no = df_test[df_test['pred'] >= 0.45]

    # With ROC
    X_train_roc = df[train_mask][feature_cols_with_roc].values.astype(np.float32)
    X_test_roc = df[test_mask][feature_cols_with_roc].values.astype(np.float32)
    X_train_roc = np.nan_to_num(X_train_roc)
    X_test_roc = np.nan_to_num(X_test_roc)

    model_roc = XGBClassifier(n_estimators=100, max_depth=3, random_state=42, n_jobs=-1)
    model_roc.fit(X_train_roc, y_train, verbose=False)

    df_test['pred'] = model_roc.predict_proba(X_test_roc)[:, 1]
    filtered_roc = df_test[df_test['pred'] >= 0.45]

    if len(filtered_no) > 10 and len(filtered_roc) > 10:
        wr_no = (filtered_no['net_pips'] > 0).mean() * 100
        avg_no = filtered_no['net_pips'].mean()
        net_no = filtered_no['net_pips'].sum()

        wr_roc = (filtered_roc['net_pips'] > 0).mean() * 100
        avg_roc = filtered_roc['net_pips'].mean()
        net_roc = filtered_roc['net_pips'].sum()

        log(f"\n{test_year}:")
        log(f"  No ROC:   {len(filtered_no):>5} trades, {wr_no:.1f}% WR, {avg_no:+.2f} avg, {net_no:+,.0f} net")
        log(f"  With ROC: {len(filtered_roc):>5} trades, {wr_roc:.1f}% WR, {avg_roc:+.2f} avg, {net_roc:+,.0f} net")
        log(f"  Diff: {net_roc - net_no:+,.0f} pips")

        results_no_roc.append(net_no)
        results_with_roc.append(net_roc)

log("\n" + "=" * 70)
log("SUMMARY")
log("=" * 70)
log(f"Total without ROC: {sum(results_no_roc):+,.0f} pips")
log(f"Total with ROC:    {sum(results_with_roc):+,.0f} pips")
log(f"Difference:        {sum(results_with_roc) - sum(results_no_roc):+,.0f} pips")

log("\nDONE")
