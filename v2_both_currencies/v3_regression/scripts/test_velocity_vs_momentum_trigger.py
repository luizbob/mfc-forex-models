"""
Test: Velocity vs Momentum as Entry Trigger
============================================
Compare using 1-bar change (velocity) vs 2-bar change (momentum) for entry.
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
log("VELOCITY vs MOMENTUM TRIGGER TEST")
log("=" * 70)

DATA_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/data')
CLEANED_DIR = DATA_DIR / 'cleaned'

VELOCITY_THRESHOLD = 0.08
MOMENTUM_THRESHOLD = 0.12  # Higher because 2 bars of movement
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


def test_trigger(base_ccy, quote_ccy, use_momentum=False, threshold=0.08):
    """Test either velocity or momentum as trigger"""
    pair = f"{base_ccy}{quote_ccy}"
    if pair not in PIP_SIZE:
        return None

    base_mfc = load_mfc_cleaned(base_ccy, 'M15')
    quote_mfc = load_mfc_cleaned(quote_ccy, 'M15')

    if base_mfc is None or quote_mfc is None:
        return None

    price_df = load_price_data(pair, 'M15')
    if price_df is None:
        price_m1 = load_price_data(pair, 'M1')
        if price_m1 is None:
            return None
        price_df = price_m1.resample('15min').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
        }).dropna()

    df = pd.DataFrame(index=base_mfc.index)
    df['price_open'] = price_df['Open'].reindex(df.index, method='ffill')
    df['price_close'] = price_df['Close'].reindex(df.index, method='ffill')

    # Velocity (1-bar) and Momentum (2-bar)
    df['base_vel_raw'] = base_mfc.diff()
    df['quote_vel_raw'] = quote_mfc.diff()
    df['base_mom_raw'] = base_mfc.diff(2)
    df['quote_mom_raw'] = quote_mfc.diff(2)

    # Shifted versions for entry detection
    df['base_vel'] = base_mfc.diff().shift(1)
    df['quote_vel'] = quote_mfc.diff().shift(1)
    df['base_mom'] = base_mfc.diff(2).shift(1)
    df['quote_mom'] = quote_mfc.diff(2).shift(1)

    df = df.dropna()

    pip_size = PIP_SIZE[pair]
    spread = SPREADS.get(pair, 2.0)

    price_open = df['price_open'].values
    price_close = df['price_close'].values

    if use_momentum:
        base_trigger = df['base_mom'].values
        quote_trigger = df['quote_mom'].values
    else:
        base_trigger = df['base_vel'].values
        quote_trigger = df['quote_vel'].values

    base_vel_raw = df['base_vel_raw'].values
    quote_vel_raw = df['quote_vel_raw'].values

    n = len(df)
    all_entries = []
    i = 0

    while i < n - MAX_HOLD_BARS - 1:
        base_t = base_trigger[i]
        quote_t = quote_trigger[i]

        signal = None
        trigger_type = None
        expected_sign = 0

        if base_t > threshold:
            signal = 'buy'
            trigger_type = 'base'
            expected_sign = 1
        elif base_t < -threshold:
            signal = 'sell'
            trigger_type = 'base'
            expected_sign = -1
        elif quote_t > threshold:
            signal = 'sell'
            trigger_type = 'quote'
            expected_sign = -1
        elif quote_t < -threshold:
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

        all_entries.append({
            'datetime': df.index[i],
            'pair': pair,
            'direction': signal,
            'bars_held': bars_held,
            'net_pips': net_pips,
            'is_profitable': int(net_pips > 0),
        })

        i = exit_idx + 1

    if not all_entries:
        return None

    return pd.DataFrame(all_entries)


# Test pairs
pairs = [
    ('EUR', 'USD'), ('GBP', 'USD'), ('USD', 'JPY'), ('AUD', 'USD'),
    ('EUR', 'JPY'), ('GBP', 'JPY'), ('EUR', 'GBP'), ('USD', 'CAD'),
]

log("\nTesting on 8 major pairs...")

# Test velocity trigger
log("\n" + "=" * 70)
log("VELOCITY TRIGGER (1-bar change, threshold 0.08)")
log("=" * 70)

vel_results = []
for base, quote in pairs:
    result = test_trigger(base, quote, use_momentum=False, threshold=0.08)
    if result is not None:
        vel_results.append(result)

vel_df = pd.concat(vel_results, ignore_index=True)
log(f"\nTotal entries: {len(vel_df):,}")
log(f"Profitable: {vel_df['is_profitable'].mean()*100:.1f}%")
log(f"Avg pips: {vel_df['net_pips'].mean():+.2f}")
log(f"Avg bars held: {vel_df['bars_held'].mean():.1f}")

log(f"\nBy Bars Held:")
for low, high, label in [(1, 3, '1-2'), (3, 6, '3-5'), (6, 12, '6-11'), (12, 25, '12+')]:
    subset = vel_df[(vel_df['bars_held'] >= low) & (vel_df['bars_held'] < high)]
    if len(subset) > 100:
        wr = subset['is_profitable'].mean() * 100
        avg = subset['net_pips'].mean()
        log(f"  {label}: {len(subset):,}, {wr:.1f}% WR, {avg:+.2f} avg")

# Test momentum trigger with different thresholds
for mom_thresh in [0.10, 0.12, 0.14, 0.16]:
    log("\n" + "=" * 70)
    log(f"MOMENTUM TRIGGER (2-bar change, threshold {mom_thresh})")
    log("=" * 70)

    mom_results = []
    for base, quote in pairs:
        result = test_trigger(base, quote, use_momentum=True, threshold=mom_thresh)
        if result is not None:
            mom_results.append(result)

    mom_df = pd.concat(mom_results, ignore_index=True)
    log(f"\nTotal entries: {len(mom_df):,}")
    log(f"Profitable: {mom_df['is_profitable'].mean()*100:.1f}%")
    log(f"Avg pips: {mom_df['net_pips'].mean():+.2f}")
    log(f"Avg bars held: {mom_df['bars_held'].mean():.1f}")

    log(f"\nBy Bars Held:")
    for low, high, label in [(1, 3, '1-2'), (3, 6, '3-5'), (6, 12, '6-11'), (12, 25, '12+')]:
        subset = mom_df[(mom_df['bars_held'] >= low) & (mom_df['bars_held'] < high)]
        if len(subset) > 100:
            wr = subset['is_profitable'].mean() * 100
            avg = subset['net_pips'].mean()
            log(f"  {label}: {len(subset):,}, {wr:.1f}% WR, {avg:+.2f} avg")

log("\n" + "=" * 70)
log("DONE")
