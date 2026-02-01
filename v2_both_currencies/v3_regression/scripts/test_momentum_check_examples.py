"""
Check specific examples: when signals agree vs conflict
"""
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from pathlib import Path

def log(msg=""):
    print(msg, flush=True)

DATA_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/data')
CLEANED_DIR = DATA_DIR / 'cleaned'

CURRENCIES = ['EUR', 'USD', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD', 'NZD']
PAIRS = [
    'EURUSD', 'GBPUSD', 'AUDUSD', 'NZDUSD', 'USDCAD', 'USDCHF', 'USDJPY',
    'EURGBP', 'EURJPY', 'EURCHF', 'EURCAD', 'EURAUD', 'EURNZD',
    'GBPJPY', 'GBPCHF', 'GBPCAD', 'GBPAUD', 'GBPNZD',
    'AUDJPY', 'AUDCHF', 'AUDCAD', 'AUDNZD',
    'NZDJPY', 'NZDCHF', 'NZDCAD',
    'CADJPY', 'CADCHF',
    'CHFJPY'
]
PIP_SIZE = {p: 0.01 if 'JPY' in p else 0.0001 for p in PAIRS}
BOX_UPPER = 0.2
BOX_LOWER = -0.2
VELOCITY_THRESHOLD = 0.05

# Load data
def load_mfc(currency, timeframe):
    fp = CLEANED_DIR / f'mfc_currency_{currency}_{timeframe}_clean.csv'
    if not fp.exists():
        return None
    df = pd.read_csv(fp)
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df = df.set_index('datetime').sort_index()
    return df['MFC']

def load_price(pair, timeframe):
    fp = DATA_DIR / f'{pair}_GMT+0_US-DST_{timeframe}.csv'
    if not fp.exists():
        return None
    df = pd.read_csv(fp, header=None, names=['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df = df.set_index('datetime').sort_index()
    return df

log("Loading data...")

mfc_data = {}
for ccy in CURRENCIES:
    mfc_data[ccy] = load_mfc(ccy, 'M30')

mfc_df = pd.DataFrame(index=mfc_data['EUR'].index)
for ccy in CURRENCIES:
    if mfc_data[ccy] is not None:
        mfc_df[ccy] = mfc_data[ccy]
        mfc_df[f'{ccy}_vel'] = mfc_df[ccy].diff()
mfc_df = mfc_df.dropna()

price_data = {}
for pair in PAIRS:
    price_data[pair] = load_price(pair, 'M30')

log(f"Data loaded: {len(mfc_df):,} bars\n")

# ============================================================================
# FIND ALL MOMENTUM SIGNALS WITH PAIR ANALYSIS
# ============================================================================

def get_signal_with_pairs(mfc_df, price_data, ccy, vel_threshold=0.05):
    """Find signals and analyze all pairs."""
    values = mfc_df[ccy].values
    velocities = mfc_df[f'{ccy}_vel'].values
    times = mfc_df.index

    signals = []
    i = 1

    while i < len(mfc_df) - 1:
        entry_signal = None

        # Entry from ABOVE
        if values[i-1] > BOX_UPPER and values[i] <= BOX_UPPER and velocities[i] < -vel_threshold:
            entry_signal = ('weak', i, velocities[i])

        # Entry from BELOW
        elif values[i-1] < BOX_LOWER and values[i] >= BOX_LOWER and velocities[i] > vel_threshold:
            entry_signal = ('strong', i, velocities[i])

        if entry_signal:
            direction, entry_idx, entry_vel = entry_signal

            # Find exit
            j = i + 1
            while j < len(mfc_df):
                if direction == 'weak':
                    if velocities[j] > -vel_threshold or velocities[j] > 0:
                        break
                else:
                    if velocities[j] < vel_threshold or velocities[j] < 0:
                        break
                j += 1

            if j < len(mfc_df):
                entry_time = times[entry_idx]
                exit_time = times[j]

                # Get all currency velocities at entry
                all_vel = {c: mfc_df[f'{c}_vel'].iloc[entry_idx] for c in CURRENCIES}
                all_mfc = {c: mfc_df[c].iloc[entry_idx] for c in CURRENCIES}

                # Analyze each pair containing this currency
                pair_analysis = []
                for pair in PAIRS:
                    base = pair[:3]
                    quote = pair[3:]

                    if ccy not in [base, quote]:
                        continue

                    other_ccy = quote if ccy == base else base
                    other_vel = all_vel[other_ccy]

                    # Determine what BOTH currencies are saying
                    # For a pair, base strong or quote weak = pair UP
                    # base weak or quote strong = pair DOWN

                    if ccy == base:
                        signal_ccy_says = 'UP' if direction == 'strong' else 'DOWN'
                    else:  # ccy is quote
                        signal_ccy_says = 'DOWN' if direction == 'strong' else 'UP'

                    # What does other currency say?
                    other_direction = 'strong' if other_vel > 0.03 else ('weak' if other_vel < -0.03 else 'neutral')

                    if other_ccy == base:
                        other_ccy_says = 'UP' if other_direction == 'strong' else ('DOWN' if other_direction == 'weak' else 'neutral')
                    else:  # other is quote
                        other_ccy_says = 'DOWN' if other_direction == 'strong' else ('UP' if other_direction == 'weak' else 'neutral')

                    # Agreement?
                    if other_ccy_says == 'neutral':
                        agreement = 'neutral'
                    elif signal_ccy_says == other_ccy_says:
                        agreement = 'AGREE'
                    else:
                        agreement = 'CONFLICT'

                    # Get actual price move
                    if price_data[pair] is not None:
                        pdf = price_data[pair]
                        try:
                            if entry_time in pdf.index:
                                entry_price = pdf.loc[entry_time, 'Close']
                            else:
                                idx = pdf.index.get_indexer([entry_time], method='nearest')[0]
                                entry_price = pdf.iloc[idx]['Close']

                            if exit_time in pdf.index:
                                exit_price = pdf.loc[exit_time, 'Close']
                            else:
                                idx = pdf.index.get_indexer([exit_time], method='nearest')[0]
                                exit_price = pdf.iloc[idx]['Close']

                            pip_move = (exit_price - entry_price) / PIP_SIZE[pair]
                            actual_dir = 'UP' if pip_move > 0 else 'DOWN'

                            # Did signal currency's prediction work?
                            signal_correct = (signal_ccy_says == actual_dir)

                            pair_analysis.append({
                                'pair': pair,
                                'other_ccy': other_ccy,
                                'other_vel': other_vel,
                                'signal_says': signal_ccy_says,
                                'other_says': other_ccy_says,
                                'agreement': agreement,
                                'actual_dir': actual_dir,
                                'pip_move': pip_move,
                                'signal_correct': signal_correct,
                            })
                        except:
                            pass

                signals.append({
                    'entry_time': entry_time,
                    'exit_time': exit_time,
                    'ccy': ccy,
                    'direction': direction,
                    'entry_vel': entry_vel,
                    'all_vel': all_vel,
                    'all_mfc': all_mfc,
                    'pairs': pair_analysis,
                })

                i = j + 1
                continue

        i += 1

    return signals

# Collect all signals
log("=" * 70)
log("ANALYZING SIGNALS: AGREEMENT vs CONFLICT")
log("=" * 70)

all_agree_win = []
all_agree_lose = []
all_conflict_win = []
all_conflict_lose = []

for ccy in CURRENCIES:
    signals = get_signal_with_pairs(mfc_df, price_data, ccy)

    for sig in signals:
        for pr in sig['pairs']:
            if pr['agreement'] == 'AGREE':
                if pr['signal_correct']:
                    all_agree_win.append({'sig': sig, 'pair': pr})
                else:
                    all_agree_lose.append({'sig': sig, 'pair': pr})
            elif pr['agreement'] == 'CONFLICT':
                if pr['signal_correct']:
                    all_conflict_win.append({'sig': sig, 'pair': pr})
                else:
                    all_conflict_lose.append({'sig': sig, 'pair': pr})

log(f"\nAGREE + WIN:     {len(all_agree_win):,}")
log(f"AGREE + LOSE:    {len(all_agree_lose):,}")
log(f"CONFLICT + WIN:  {len(all_conflict_win):,}")
log(f"CONFLICT + LOSE: {len(all_conflict_lose):,}")

agree_wr = len(all_agree_win) / (len(all_agree_win) + len(all_agree_lose)) * 100 if (len(all_agree_win) + len(all_agree_lose)) > 0 else 0
conflict_wr = len(all_conflict_win) / (len(all_conflict_win) + len(all_conflict_lose)) * 100 if (len(all_conflict_win) + len(all_conflict_lose)) > 0 else 0

log(f"\nAGREE win rate:    {agree_wr:.1f}%")
log(f"CONFLICT win rate: {conflict_wr:.1f}%")

# ============================================================================
# EXAMPLES
# ============================================================================

def show_example(label, item):
    sig = item['sig']
    pr = item['pair']

    log(f"\n{'='*70}")
    log(f"EXAMPLE: {label}")
    log(f"{'='*70}")

    log(f"\nTime: {sig['entry_time']} to {sig['exit_time']}")
    log(f"Signal: {sig['ccy']} is {sig['direction'].upper()} (velocity: {sig['entry_vel']:+.3f})")

    log(f"\nAll currency velocities at entry:")
    log(f"{'CCY':<5} {'Velocity':>10} {'Direction':>12}")
    log("-" * 30)
    for c in CURRENCIES:
        vel = sig['all_vel'][c]
        d = 'STRONG' if vel > 0.03 else ('WEAK' if vel < -0.03 else 'neutral')
        marker = " <--" if c == sig['ccy'] else ""
        log(f"{c:<5} {vel:>+10.4f} {d:>12}{marker}")

    log(f"\nPair: {pr['pair']}")
    log(f"  Signal currency ({sig['ccy']}) says: {pr['signal_says']}")
    log(f"  Other currency ({pr['other_ccy']}, vel={pr['other_vel']:+.3f}) says: {pr['other_says']}")
    log(f"  Agreement: {pr['agreement']}")
    log(f"  Actual move: {pr['actual_dir']} ({pr['pip_move']:+.1f} pips)")
    log(f"  Signal correct: {'YES' if pr['signal_correct'] else 'NO'}")

# Show examples
if all_agree_win:
    show_example("AGREE + WIN (both currencies agree, price followed)", all_agree_win[100])

if all_agree_lose:
    show_example("AGREE + LOSE (both currencies agree, price went OPPOSITE)", all_agree_lose[100])

if all_conflict_win:
    show_example("CONFLICT + WIN (currencies disagree, signal currency won)", all_conflict_win[100])

if all_conflict_lose:
    show_example("CONFLICT + LOSE (currencies disagree, signal currency lost)", all_conflict_lose[100])

# ============================================================================
# AVERAGE PIPS
# ============================================================================

log("\n" + "=" * 70)
log("AVERAGE PIPS BY SCENARIO")
log("=" * 70)

def avg_pips(items):
    if not items:
        return 0
    return np.mean([abs(x['pair']['pip_move']) for x in items])

log(f"\n{'Scenario':<25} {'Count':>10} {'Avg Pips':>12}")
log("-" * 50)
log(f"{'AGREE + WIN':<25} {len(all_agree_win):>10,} {avg_pips(all_agree_win):>+12.1f}")
log(f"{'AGREE + LOSE':<25} {len(all_agree_lose):>10,} {avg_pips(all_agree_lose):>+12.1f}")
log(f"{'CONFLICT + WIN':<25} {len(all_conflict_win):>10,} {avg_pips(all_conflict_win):>+12.1f}")
log(f"{'CONFLICT + LOSE':<25} {len(all_conflict_lose):>10,} {avg_pips(all_conflict_lose):>+12.1f}")

log("\n" + "=" * 70)
log("CONCLUSION")
log("=" * 70)
log(f"\nWhen both currencies AGREE: {agree_wr:.1f}% win rate")
log(f"When currencies CONFLICT:   {conflict_wr:.1f}% win rate")

if agree_wr > conflict_wr:
    log("\n--> Agreement between currencies DOES help predict direction")
else:
    log("\n--> Agreement doesn't improve prediction (or makes it worse)")

log()
