"""
Daily Setup Analyzer
====================
Given a date, show the best trade setups based on our research.
"""
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from pathlib import Path
import sys

def log(msg=""):
    print(msg, flush=True)

DATA_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/data')
CLEANED_DIR = DATA_DIR / 'cleaned'

CURRENCIES = ['USD', 'EUR', 'GBP', 'JPY', 'AUD', 'CAD', 'CHF', 'NZD']
TIMEFRAMES = ['H1', 'H4', 'D1', 'W1', 'MN']

# Win rates from our research
SETUP_WIN_RATES = {
    # Single TF
    'D1_above_DOWN': 51.2,
    'D1_above_UP': 55.6,
    'D1_inbox_UP': 53.5,
    'D1_inbox_DOWN': 52.9,
    'D1_below_UP': 50.4,
    'D1_below_DOWN': 52.5,

    'H1_above_DOWN': 55.6,
    'H1_above_UP': 48.9,
    'H1_inbox_UP': 53.0,
    'H1_inbox_DOWN': 52.1,
    'H1_below_UP': 52.7,
    'H1_below_DOWN': 48.3,

    # Combined setups
    'D1_above_DOWN_H1_DOWN': 58.0,
    'D1_below_UP_H1_UP': 48.9,
    'H1_D1_aligned': 55.0,
    'H1_H4_D1_aligned': 55.0,

    # With higher TF position
    'W1_below_D1_DOWN': 56.7,
    'W1_below_D1_UP': 53.3,
    'MN_below_D1_DOWN': 56.6,
    'MN_below_D1_UP': 55.0,
    'MN_above_D1_above_DOWN_H1_DOWN': 59.1,
    'MN_below_D1_below_UP_H1_UP': 58.0,

    # Day of week adjustments
    'Monday_bonus': 5.0,
    'Tuesday_penalty': -7.0,
}

# Currency adjustments
CURRENCY_ADJ = {
    'JPY': +8.0,  # Best performer
    'EUR': +3.0,
    'USD': +2.0,
    'AUD': +1.0,
    'GBP': -2.0,
    'NZD': -3.0,
    'CAD': -5.0,  # Poor performer
    'CHF': -7.0,  # Worst
}

def load_mfc(currency, timeframe):
    fp = CLEANED_DIR / f'mfc_currency_{currency}_{timeframe}_clean.csv'
    if not fp.exists():
        return None
    df = pd.read_csv(fp)
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df = df.set_index('datetime').sort_index()
    return df['MFC']

def load_price(pair):
    fp = DATA_DIR / f'{pair}_GMT+0_US-DST_M30.csv'
    if fp.exists():
        with open(fp, 'r') as f:
            first_line = f.readline()
        if 'Date' in first_line or 'Open' in first_line:
            df = pd.read_csv(fp)
        else:
            df = pd.read_csv(fp, header=None,
                           names=['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
        df = df.set_index('datetime').sort_index()
        return df
    return None

def get_position(mfc_val):
    if mfc_val < -0.2:
        return 'below'
    elif mfc_val > 0.2:
        return 'above'
    else:
        return 'inbox'

def get_vel_dir(vel_val):
    if vel_val > 0.01:
        return 'UP'
    elif vel_val < -0.01:
        return 'DOWN'
    else:
        return 'FLAT'

def analyze_day(target_date):
    target_date = pd.Timestamp(target_date).normalize()
    weekday = target_date.dayofweek
    weekday_name = target_date.strftime('%A')

    log("=" * 70)
    log(f"DAILY SETUP ANALYZER: {target_date.strftime('%Y-%m-%d')} ({weekday_name})")
    log("=" * 70)

    # Day of week adjustment
    if weekday == 0:  # Monday
        day_adj = SETUP_WIN_RATES['Monday_bonus']
        log(f"\n✓ MONDAY BONUS: +{day_adj}% to all setups")
    elif weekday == 1:  # Tuesday
        day_adj = SETUP_WIN_RATES['Tuesday_penalty']
        log(f"\n✗ TUESDAY PENALTY: {day_adj}% to all setups - CONSIDER SKIPPING")
    else:
        day_adj = 0

    # Load all MFC data
    mfc_data = {}
    for ccy in CURRENCIES:
        mfc_data[ccy] = {}
        for tf in TIMEFRAMES:
            mfc = load_mfc(ccy, tf)
            if mfc is not None:
                # Get value at or before target date (shifted - yesterday's close)
                idx = mfc.index[mfc.index < target_date]
                if len(idx) > 0:
                    mfc_val = mfc.loc[idx[-1]]

                    # Velocity
                    vel = mfc.diff()
                    vel_val = vel.loc[idx[-1]] if idx[-1] in vel.index else np.nan

                    if not pd.isna(mfc_val) and not pd.isna(vel_val):
                        mfc_data[ccy][tf] = {
                            'mfc': mfc_val,
                            'vel': vel_val,
                            'pos': get_position(mfc_val),
                            'vel_dir': get_vel_dir(vel_val),
                        }

    # Display state for each currency
    log("\n" + "=" * 70)
    log("MFC STATE AT 00:00 (shifted - based on previous close)")
    log("=" * 70)

    log(f"\n{'CCY':<6} {'H1':^18} {'H4':^18} {'D1':^18} {'W1':^18} {'MN':^18}")
    log("-" * 96)

    for ccy in CURRENCIES:
        row = f"{ccy:<6}"
        for tf in TIMEFRAMES:
            if tf in mfc_data[ccy]:
                d = mfc_data[ccy][tf]
                cell = f"{d['pos'][:3]} {d['vel_dir']}"
                row += f" {cell:^18}"
            else:
                row += f" {'N/A':^18}"
        log(row)

    # Find best setups
    log("\n" + "=" * 70)
    log("TRADE SETUPS (sorted by estimated win rate)")
    log("=" * 70)

    setups = []

    for ccy in CURRENCIES:
        if len(mfc_data[ccy]) < 5:
            continue

        d1 = mfc_data[ccy].get('D1', {})
        h1 = mfc_data[ccy].get('H1', {})
        h4 = mfc_data[ccy].get('H4', {})
        w1 = mfc_data[ccy].get('W1', {})
        mn = mfc_data[ccy].get('MN', {})

        if not d1 or not h1:
            continue

        d1_pos = d1.get('pos')
        d1_vel = d1.get('vel_dir')
        h1_vel = h1.get('vel_dir')
        h4_vel = h4.get('vel_dir', 'FLAT')
        w1_pos = w1.get('pos')
        mn_pos = mn.get('pos')

        # Handle different scenarios
        base_wr = 50.0
        reasons = []
        direction = None

        # SCENARIO 1: D1 has velocity - use D1 as primary
        if d1_vel != 'FLAT':
            direction = 'LONG' if d1_vel == 'UP' else 'SHORT'
            d1_key = f"D1_{d1_pos}_{d1_vel}"
            if d1_key in SETUP_WIN_RATES:
                base_wr = SETUP_WIN_RATES[d1_key]
                reasons.append(f"D1 {d1_pos} {d1_vel}")

        # SCENARIO 2: D1 is FLAT - look at other signals
        else:
            # When D1 is FLAT, use H1 as primary but with lower confidence
            # Also check W1/MN position for context

            if h1_vel == 'FLAT':
                continue  # No signal at all

            direction = 'LONG' if h1_vel == 'UP' else 'SHORT'
            reasons.append("D1 FLAT - using H1 as primary")

            # H1 signal alone is weaker
            h1_key = f"H1_{h1.get('pos')}_{h1_vel}"
            if h1_key in SETUP_WIN_RATES:
                base_wr = SETUP_WIN_RATES[h1_key] - 2.0  # Penalty for no D1 confirmation
            else:
                base_wr = 51.0

            # But if W1 or MN position supports the direction, it's better
            if h1_vel == 'UP':
                if w1_pos == 'below':
                    base_wr += 3.0
                    reasons.append("W1 below supports UP")
                if mn_pos == 'below':
                    base_wr += 2.0
                    reasons.append("MN below supports UP")
                if d1_pos == 'below':
                    base_wr += 2.0
                    reasons.append("D1 position below (room to go up)")
            else:  # h1_vel == 'DOWN'
                if w1_pos == 'above':
                    base_wr += 3.0
                    reasons.append("W1 above supports DOWN")
                if mn_pos == 'above':
                    base_wr += 2.0
                    reasons.append("MN above supports DOWN")
                if d1_pos == 'above':
                    base_wr += 2.0
                    reasons.append("D1 position above (room to go down)")

            # H4 alignment helps even when D1 flat
            if h4_vel == h1_vel:
                base_wr += 2.0
                reasons.append("H4 confirms H1")

        if direction is None:
            continue

        # H1 alignment bonus (only when D1 has velocity)
        if d1_vel != 'FLAT' and h1_vel == d1_vel:
            if d1_pos == 'above' and d1_vel == 'DOWN':
                base_wr = SETUP_WIN_RATES['D1_above_DOWN_H1_DOWN']
                reasons.append("H1 aligned (overbought reversal)")
            elif d1_pos == 'below' and d1_vel == 'UP':
                base_wr = SETUP_WIN_RATES['D1_below_UP_H1_UP']
                reasons.append("H1 aligned (oversold reversal)")
            else:
                base_wr = max(base_wr, SETUP_WIN_RATES['H1_D1_aligned'])
                reasons.append("H1+D1 aligned")

        # H4 alignment (when D1 active)
        if d1_vel != 'FLAT' and h4_vel == d1_vel and h1_vel == d1_vel:
            base_wr = max(base_wr, SETUP_WIN_RATES['H1_H4_D1_aligned'])
            reasons.append("H4 also aligned")

        # W1/MN position bonus (when D1 active)
        if d1_vel != 'FLAT':
            if w1_pos == 'below' and d1_vel == 'DOWN':
                base_wr = max(base_wr, SETUP_WIN_RATES['W1_below_D1_DOWN'])
                reasons.append("W1 below + D1 DOWN")
            elif w1_pos == 'below' and d1_vel == 'UP':
                base_wr = max(base_wr, SETUP_WIN_RATES['W1_below_D1_UP'])
                reasons.append("W1 below supports UP")

            if mn_pos == 'below' and d1_pos == 'below' and d1_vel == 'UP' and h1_vel == 'UP':
                base_wr = max(base_wr, SETUP_WIN_RATES['MN_below_D1_below_UP_H1_UP'])
                reasons.append("MN+D1 below + all UP (ultimate long)")
            elif mn_pos == 'above' and d1_pos == 'above' and d1_vel == 'DOWN' and h1_vel == 'DOWN':
                base_wr = max(base_wr, SETUP_WIN_RATES['MN_above_D1_above_DOWN_H1_DOWN'])
                reasons.append("MN+D1 above + all DOWN (ultimate short)")

        # Currency adjustment
        ccy_adj = CURRENCY_ADJ.get(ccy, 0)

        # Day adjustment
        final_wr = base_wr + ccy_adj + day_adj

        # Quality score
        quality = 'A' if final_wr >= 58 else 'B' if final_wr >= 54 else 'C' if final_wr >= 50 else 'D'

        setups.append({
            'ccy': ccy,
            'direction': direction,
            'base_wr': base_wr,
            'ccy_adj': ccy_adj,
            'day_adj': day_adj,
            'final_wr': final_wr,
            'quality': quality,
            'reasons': reasons,
            'd1_pos': d1_pos,
            'd1_vel': d1_vel,
            'd1_flat': d1_vel == 'FLAT',
            'h1_vel': h1_vel,
            'h4_vel': h4_vel,
            'w1_pos': w1_pos,
            'mn_pos': mn_pos,
        })

    # Sort by final win rate
    setups.sort(key=lambda x: x['final_wr'], reverse=True)

    log(f"\n{'#':<3} {'CCY':<6} {'Dir':<7} {'Signal':<8} {'D1 Pos':<8} {'H1':<6} {'H4':<6} {'Base':>6} {'Final':>7} {'Q':<3}")
    log("-" * 75)

    for i, s in enumerate(setups[:15], 1):
        signal = "H1" if s['d1_flat'] else "D1"
        d1_pos_str = s['d1_pos'][:3]
        log(f"{i:<3} {s['ccy']:<6} {s['direction']:<7} {signal:<8} {d1_pos_str:<8} {s['h1_vel']:<6} {s['h4_vel']:<6} "
            f"{s['base_wr']:>5.1f}% {s['final_wr']:>6.1f}% {s['quality']:<3}")

    # Top recommendations
    log("\n" + "=" * 70)
    log("TOP RECOMMENDATIONS")
    log("=" * 70)

    grade_a = [s for s in setups if s['quality'] == 'A']
    grade_b = [s for s in setups if s['quality'] == 'B']

    if grade_a:
        log("\n🌟 GRADE A SETUPS (58%+ estimated WR):")
        for s in grade_a:
            log(f"\n  {s['ccy']} {s['direction']} - {s['final_wr']:.1f}% estimated")
            log(f"    Reasons: {', '.join(s['reasons'])}")
    else:
        log("\n  No Grade A setups today")

    if grade_b:
        log("\n⭐ GRADE B SETUPS (54-58% estimated WR):")
        for s in grade_b[:5]:
            log(f"\n  {s['ccy']} {s['direction']} - {s['final_wr']:.1f}% estimated")
            log(f"    Reasons: {', '.join(s['reasons'])}")

    # Currencies to avoid
    avoid = [s for s in setups if s['quality'] == 'D']
    if avoid:
        log("\n⚠️  AVOID THESE (below 50%):")
        for s in avoid:
            log(f"  {s['ccy']} - {s['final_wr']:.1f}%")

    # What actually happened (if date is in the past)
    log("\n" + "=" * 70)
    log("ACTUAL RESULTS (12h movement from 00:00)")
    log("=" * 70)

    PAIRS = {
        'EURUSD': ('EUR', 'USD'), 'GBPUSD': ('GBP', 'USD'), 'USDJPY': ('USD', 'JPY'),
        'AUDUSD': ('AUD', 'USD'), 'USDCAD': ('USD', 'CAD'), 'USDCHF': ('USD', 'CHF'),
    }

    log(f"\n{'CCY':<6} {'Expected':<10} {'Actual 12h':>12} {'Result':<10}")
    log("-" * 45)

    for s in setups[:10]:
        ccy = s['ccy']
        expected_dir = s['direction']

        # Calculate actual movement
        movements = []
        for pair, (base, quote) in PAIRS.items():
            if base != ccy and quote != ccy:
                continue

            price_df = load_price(pair)
            if price_df is None:
                continue

            try:
                start_time = target_date
                end_time = target_date + pd.Timedelta(hours=12)

                start_window = price_df[(price_df.index >= start_time) &
                                       (price_df.index < start_time + pd.Timedelta(hours=1))]
                end_window = price_df[(price_df.index >= end_time - pd.Timedelta(minutes=30)) &
                                     (price_df.index <= end_time)]

                if len(start_window) > 0 and len(end_window) > 0:
                    open_p = start_window.iloc[0]['Open']
                    close_p = end_window.iloc[-1]['Close']
                    pct = (close_p - open_p) / open_p * 100

                    if base == ccy:
                        movements.append(pct)
                    else:
                        movements.append(-pct)
            except:
                continue

        if movements:
            avg_move = np.mean(movements)
            actual_dir = 'UP' if avg_move > 0 else 'DOWN'

            if (expected_dir == 'LONG' and avg_move > 0) or (expected_dir == 'SHORT' and avg_move < 0):
                result = '✓ WIN'
            else:
                result = '✗ LOSS'

            log(f"{ccy:<6} {expected_dir:<10} {avg_move:>+11.4f}% {result:<10}")
        else:
            log(f"{ccy:<6} {expected_dir:<10} {'N/A':>12} {'':<10}")

    return setups

if __name__ == '__main__':
    if len(sys.argv) > 1:
        date = sys.argv[1]
    else:
        date = '2024-06-03'  # Default example

    analyze_day(date)
