"""
Analyze Trend Indicators vs MFC Behavior
==========================================
Question: Can traditional trend indicators predict if MFC will stay at extremes?
Can higher timeframes "spoil" what's going to happen?

We'll check:
1. When MFC reaches extreme (+/-0.5), does HTF trend predict if it stays or reverses?
2. Does MA slope, ADX, price structure on H1/H4 correlate with MFC duration at extremes?
3. Can H4 trend direction predict M5 MFC behavior?
"""
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

def log(msg=""):
    print(msg, flush=True)

log("=" * 70)
log("TREND INDICATORS vs MFC BEHAVIOR ANALYSIS")
log("=" * 70)
log(f"Started: {datetime.now()}")

DATA_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/data')
CLEANED_DIR = DATA_DIR / 'cleaned'

CURRENCIES = ['EUR', 'USD', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD', 'NZD']

# ============================================================================
# INDICATOR CALCULATIONS
# ============================================================================

def calc_sma(series, period):
    return series.rolling(window=period).mean()

def calc_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def calc_ma_slope(ma, period=5):
    """Slope of MA over last N bars (normalized)"""
    return (ma - ma.shift(period)) / period

def calc_adx(high, low, close, period=14):
    """Calculate ADX, +DI, -DI"""
    # True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()

    # Directional Movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

    plus_di = 100 * pd.Series(plus_dm, index=high.index).rolling(period).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=high.index).rolling(period).mean() / atr

    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=period).mean()

    return adx, plus_di, minus_di

def calc_rsi(close, period=14):
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calc_macd(close, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist

def detect_higher_highs_lows(high, low, lookback=20):
    """Detect trend structure: HH/HL (uptrend) or LH/LL (downtrend)"""
    # Rolling max/min of highs and lows
    prev_high = high.rolling(lookback).max().shift(1)
    prev_low = low.rolling(lookback).min().shift(1)

    # Current vs previous structure
    hh = high > prev_high  # Higher High
    ll = low < prev_low    # Lower Low

    # Trend score: positive = uptrend, negative = downtrend
    trend_score = hh.astype(int) - ll.astype(int)
    trend_score_smooth = trend_score.rolling(5).mean()

    return trend_score_smooth


# ============================================================================
# LOAD AND PREPARE DATA
# ============================================================================

log("\nLoading MFC data...")
mfc_m5 = {}
for ccy in CURRENCIES:
    df = pd.read_csv(CLEANED_DIR / f'mfc_currency_{ccy}_M5_clean.csv')
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df = df.set_index('datetime').sort_index()
    mfc_m5[ccy] = df['MFC']

log("Loading price data for indicators...")

# We need price data to calculate indicators
# Use a major pair for each currency to get its "price behavior"
# For simplicity, use vs USD (or vs EUR for USD)
price_pairs = {
    'EUR': 'EURUSD',
    'GBP': 'GBPUSD',
    'AUD': 'AUDUSD',
    'NZD': 'NZDUSD',
    'USD': 'EURUSD',  # Inverted
    'JPY': 'USDJPY',  # Inverted
    'CHF': 'USDCHF',  # Inverted
    'CAD': 'USDCAD',  # Inverted
}

def load_price_ohlc(pair, timeframe='H1'):
    """Load price data and resample to timeframe"""
    fp = DATA_DIR / f'{pair}_GMT+0_US-DST_M1.csv'
    if not fp.exists():
        return None

    chunks = []
    for chunk in pd.read_csv(fp, chunksize=500000):
        chunk['datetime'] = pd.to_datetime(chunk['Date'] + ' ' + chunk['Time'])
        chunk = chunk.set_index('datetime')
        chunk = chunk[chunk.index >= '2020-01-01']
        if len(chunk) > 0:
            if timeframe == 'H1':
                resampled = chunk.resample('1h').agg({
                    'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
                }).dropna()
            elif timeframe == 'H4':
                resampled = chunk.resample('4h').agg({
                    'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
                }).dropna()
            elif timeframe == 'D1':
                resampled = chunk.resample('1D').agg({
                    'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
                }).dropna()
            else:
                resampled = chunk.resample('5min').agg({
                    'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
                }).dropna()
            chunks.append(resampled)

    if chunks:
        result = pd.concat(chunks)
        result = result[~result.index.duplicated(keep='first')]
        return result
    return None

# Load H1 and H4 price for each currency's reference pair
log("Calculating indicators for each currency...")

currency_indicators = {}

for ccy in CURRENCIES:
    pair = price_pairs[ccy]
    invert = ccy in ['USD', 'JPY', 'CHF', 'CAD']  # These are quote currencies

    # Load H1 data
    h1 = load_price_ohlc(pair, 'H1')
    h4 = load_price_ohlc(pair, 'H4')

    if h1 is None or h4 is None:
        log(f"  {ccy}: Missing price data")
        continue

    # If currency is quote, we need to "invert" the interpretation
    # (price going up means quote weakening, so flip signals)

    # Calculate H1 indicators
    h1_close = h1['Close']
    h1_high = h1['High']
    h1_low = h1['Low']

    h1_ma50 = calc_sma(h1_close, 50)
    h1_ma200 = calc_sma(h1_close, 200)
    h1_ma_slope = calc_ma_slope(h1_ma50, 5)
    h1_adx, h1_plus_di, h1_minus_di = calc_adx(h1_high, h1_low, h1_close, 14)
    h1_rsi = calc_rsi(h1_close, 14)
    h1_macd, h1_macd_sig, h1_macd_hist = calc_macd(h1_close)
    h1_structure = detect_higher_highs_lows(h1_high, h1_low, 20)

    # Calculate H4 indicators
    h4_close = h4['Close']
    h4_high = h4['High']
    h4_low = h4['Low']

    h4_ma50 = calc_sma(h4_close, 50)
    h4_ma200 = calc_sma(h4_close, 200)
    h4_ma_slope = calc_ma_slope(h4_ma50, 5)
    h4_adx, h4_plus_di, h4_minus_di = calc_adx(h4_high, h4_low, h4_close, 14)
    h4_rsi = calc_rsi(h4_close, 14)
    h4_macd, h4_macd_sig, h4_macd_hist = calc_macd(h4_close)
    h4_structure = detect_higher_highs_lows(h4_high, h4_low, 20)

    # Create indicator dataframe
    h1_ind = pd.DataFrame(index=h1.index)
    h1_ind['ma_trend'] = (h1_close > h1_ma200).astype(int) * 2 - 1  # 1 = above, -1 = below
    h1_ind['ma_cross'] = (h1_ma50 > h1_ma200).astype(int) * 2 - 1
    h1_ind['ma_slope'] = h1_ma_slope
    h1_ind['adx'] = h1_adx
    h1_ind['di_trend'] = (h1_plus_di > h1_minus_di).astype(int) * 2 - 1
    h1_ind['rsi'] = h1_rsi
    h1_ind['rsi_trend'] = (h1_rsi > 50).astype(int) * 2 - 1
    h1_ind['macd_trend'] = (h1_macd > h1_macd_sig).astype(int) * 2 - 1
    h1_ind['macd_zero'] = (h1_macd > 0).astype(int) * 2 - 1
    h1_ind['structure'] = h1_structure

    h4_ind = pd.DataFrame(index=h4.index)
    h4_ind['ma_trend'] = (h4_close > h4_ma200).astype(int) * 2 - 1
    h4_ind['ma_cross'] = (h4_ma50 > h4_ma200).astype(int) * 2 - 1
    h4_ind['ma_slope'] = h4_ma_slope
    h4_ind['adx'] = h4_adx
    h4_ind['di_trend'] = (h4_plus_di > h4_minus_di).astype(int) * 2 - 1
    h4_ind['rsi'] = h4_rsi
    h4_ind['rsi_trend'] = (h4_rsi > 50).astype(int) * 2 - 1
    h4_ind['macd_trend'] = (h4_macd > h4_macd_sig).astype(int) * 2 - 1
    h4_ind['macd_zero'] = (h4_macd > 0).astype(int) * 2 - 1
    h4_ind['structure'] = h4_structure

    # If inverted currency, flip the signals
    if invert:
        for col in ['ma_trend', 'ma_cross', 'di_trend', 'rsi_trend', 'macd_trend', 'macd_zero', 'structure']:
            h1_ind[col] = -h1_ind[col]
            h4_ind[col] = -h4_ind[col]
        h1_ind['ma_slope'] = -h1_ind['ma_slope']
        h4_ind['ma_slope'] = -h4_ind['ma_slope']
        h1_ind['rsi'] = 100 - h1_ind['rsi']
        h4_ind['rsi'] = 100 - h4_ind['rsi']

    currency_indicators[ccy] = {'H1': h1_ind, 'H4': h4_ind}
    log(f"  {ccy}: H1 {len(h1_ind)} bars, H4 {len(h4_ind)} bars")


# ============================================================================
# ANALYZE: When MFC reaches extreme, do indicators predict duration?
# ============================================================================

log("\n" + "=" * 70)
log("ANALYSIS: HTF Indicators vs MFC Extreme Duration")
log("=" * 70)

all_episodes = []

for ccy in CURRENCIES:
    if ccy not in currency_indicators:
        continue

    mfc = mfc_m5[ccy]
    h1_ind = currency_indicators[ccy]['H1']
    h4_ind = currency_indicators[ccy]['H4']

    # Align H1/H4 indicators to M5 grid (forward fill)
    h1_aligned = h1_ind.reindex(mfc.index, method='ffill')
    h4_aligned = h4_ind.reindex(mfc.index, method='ffill')

    mfc_arr = mfc.values
    n = len(mfc)

    # Find episodes where MFC reaches +0.5 or -0.5
    i = 0
    while i < n - 50:
        val = mfc_arr[i]

        if abs(val) >= 0.5:
            direction = 1 if val > 0 else -1
            start_idx = i

            # Count how long it stays at extreme (within 0.3 of extreme)
            duration = 0
            max_mfc = val
            j = i
            while j < n and mfc_arr[j] * direction > 0.3:
                duration += 1
                if abs(mfc_arr[j]) > abs(max_mfc):
                    max_mfc = mfc_arr[j]
                j += 1

            if duration >= 5:  # At least 5 bars (25 min)
                entry_time = mfc.index[start_idx]

                # Get indicator values at entry
                try:
                    h1_row = h1_aligned.loc[entry_time]
                    h4_row = h4_aligned.loc[entry_time]

                    episode = {
                        'currency': ccy,
                        'datetime': entry_time,
                        'direction': direction,
                        'mfc_at_entry': val,
                        'max_mfc': max_mfc,
                        'duration_bars': duration,
                        'duration_hours': duration * 5 / 60,
                        # H1 indicators
                        'h1_ma_trend': h1_row['ma_trend'],
                        'h1_ma_cross': h1_row['ma_cross'],
                        'h1_ma_slope': h1_row['ma_slope'],
                        'h1_adx': h1_row['adx'],
                        'h1_di_trend': h1_row['di_trend'],
                        'h1_rsi': h1_row['rsi'],
                        'h1_rsi_trend': h1_row['rsi_trend'],
                        'h1_macd_trend': h1_row['macd_trend'],
                        'h1_macd_zero': h1_row['macd_zero'],
                        'h1_structure': h1_row['structure'],
                        # H4 indicators
                        'h4_ma_trend': h4_row['ma_trend'],
                        'h4_ma_cross': h4_row['ma_cross'],
                        'h4_ma_slope': h4_row['ma_slope'],
                        'h4_adx': h4_row['adx'],
                        'h4_di_trend': h4_row['di_trend'],
                        'h4_rsi': h4_row['rsi'],
                        'h4_rsi_trend': h4_row['rsi_trend'],
                        'h4_macd_trend': h4_row['macd_trend'],
                        'h4_macd_zero': h4_row['macd_zero'],
                        'h4_structure': h4_row['structure'],
                    }
                    all_episodes.append(episode)
                except:
                    pass

            i = j  # Skip to end of episode
        else:
            i += 1

episodes_df = pd.DataFrame(all_episodes)
log(f"\nTotal extreme episodes: {len(episodes_df):,}")

if len(episodes_df) > 0:
    # Classification: Long duration (>2 hours) vs Short duration (<30 min)
    episodes_df['is_sustained'] = (episodes_df['duration_hours'] >= 2).astype(int)
    episodes_df['is_brief'] = (episodes_df['duration_hours'] < 0.5).astype(int)

    log(f"Sustained (>2h): {episodes_df['is_sustained'].sum():,} ({episodes_df['is_sustained'].mean()*100:.1f}%)")
    log(f"Brief (<30m): {episodes_df['is_brief'].sum():,} ({episodes_df['is_brief'].mean()*100:.1f}%)")
    log(f"Avg duration: {episodes_df['duration_hours'].mean():.1f} hours")

    # Check if indicator alignment with MFC direction predicts duration
    # For positive MFC direction, bullish indicators should help
    # "Aligned" = indicator direction matches MFC direction

    episodes_df['h1_ma_aligned'] = (episodes_df['h1_ma_trend'] == episodes_df['direction']).astype(int)
    episodes_df['h1_cross_aligned'] = (episodes_df['h1_ma_cross'] == episodes_df['direction']).astype(int)
    episodes_df['h1_di_aligned'] = (episodes_df['h1_di_trend'] == episodes_df['direction']).astype(int)
    episodes_df['h1_rsi_aligned'] = (episodes_df['h1_rsi_trend'] == episodes_df['direction']).astype(int)
    episodes_df['h1_macd_aligned'] = (episodes_df['h1_macd_trend'] == episodes_df['direction']).astype(int)
    episodes_df['h1_structure_aligned'] = (episodes_df['h1_structure'] > 0) == (episodes_df['direction'] > 0)

    episodes_df['h4_ma_aligned'] = (episodes_df['h4_ma_trend'] == episodes_df['direction']).astype(int)
    episodes_df['h4_cross_aligned'] = (episodes_df['h4_ma_cross'] == episodes_df['direction']).astype(int)
    episodes_df['h4_di_aligned'] = (episodes_df['h4_di_trend'] == episodes_df['direction']).astype(int)
    episodes_df['h4_rsi_aligned'] = (episodes_df['h4_rsi_trend'] == episodes_df['direction']).astype(int)
    episodes_df['h4_macd_aligned'] = (episodes_df['h4_macd_trend'] == episodes_df['direction']).astype(int)
    episodes_df['h4_structure_aligned'] = (episodes_df['h4_structure'] > 0) == (episodes_df['direction'] > 0)

    # Count alignments
    episodes_df['h1_alignment_score'] = (
        episodes_df['h1_ma_aligned'] + episodes_df['h1_cross_aligned'] +
        episodes_df['h1_di_aligned'] + episodes_df['h1_rsi_aligned'] +
        episodes_df['h1_macd_aligned']
    )
    episodes_df['h4_alignment_score'] = (
        episodes_df['h4_ma_aligned'] + episodes_df['h4_cross_aligned'] +
        episodes_df['h4_di_aligned'] + episodes_df['h4_rsi_aligned'] +
        episodes_df['h4_macd_aligned']
    )
    episodes_df['total_alignment'] = episodes_df['h1_alignment_score'] + episodes_df['h4_alignment_score']

    # ============================================================================
    # RESULTS: Which indicators predict sustained extremes?
    # ============================================================================

    log("\n" + "=" * 70)
    log("INDICATOR ALIGNMENT vs DURATION")
    log("=" * 70)

    log("\n--- H1 Indicators (when aligned with MFC direction) ---")
    log(f"{'Indicator':<20} {'Aligned':<12} {'Misaligned':<12} {'Diff':<10}")
    log("-" * 55)

    for ind in ['h1_ma_aligned', 'h1_cross_aligned', 'h1_di_aligned', 'h1_rsi_aligned', 'h1_macd_aligned']:
        aligned = episodes_df[episodes_df[ind] == 1]['duration_hours'].mean()
        misaligned = episodes_df[episodes_df[ind] == 0]['duration_hours'].mean()
        diff = aligned - misaligned
        name = ind.replace('h1_', '').replace('_aligned', '').upper()
        log(f"{name:<20} {aligned:>8.1f}h    {misaligned:>8.1f}h    {diff:>+6.1f}h")

    log("\n--- H4 Indicators (when aligned with MFC direction) ---")
    log(f"{'Indicator':<20} {'Aligned':<12} {'Misaligned':<12} {'Diff':<10}")
    log("-" * 55)

    for ind in ['h4_ma_aligned', 'h4_cross_aligned', 'h4_di_aligned', 'h4_rsi_aligned', 'h4_macd_aligned']:
        aligned = episodes_df[episodes_df[ind] == 1]['duration_hours'].mean()
        misaligned = episodes_df[episodes_df[ind] == 0]['duration_hours'].mean()
        diff = aligned - misaligned
        name = ind.replace('h4_', '').replace('_aligned', '').upper()
        log(f"{name:<20} {aligned:>8.1f}h    {misaligned:>8.1f}h    {diff:>+6.1f}h")

    # ADX strength effect
    log("\n--- ADX Strength Effect ---")
    log(f"{'ADX Range':<20} {'Count':<10} {'Avg Duration':<15} {'Sustained%':<12}")
    log("-" * 60)

    for (low, high) in [(0, 20), (20, 30), (30, 40), (40, 100)]:
        mask = (episodes_df['h4_adx'] >= low) & (episodes_df['h4_adx'] < high)
        subset = episodes_df[mask]
        if len(subset) > 50:
            log(f"H4 ADX {low}-{high:<10} {len(subset):<10} {subset['duration_hours'].mean():<15.1f} {subset['is_sustained'].mean()*100:<12.1f}%")

    # Total alignment score
    log("\n--- Total Alignment Score (H1+H4 combined, 0-10) ---")
    log(f"{'Score':<15} {'Count':<10} {'Avg Duration':<15} {'Sustained%':<12}")
    log("-" * 55)

    for score in range(0, 11):
        subset = episodes_df[episodes_df['total_alignment'] == score]
        if len(subset) > 30:
            log(f"Score {score:<8} {len(subset):<10} {subset['duration_hours'].mean():<15.1f} {subset['is_sustained'].mean()*100:<12.1f}%")

    # Group by alignment level
    log("\n--- Summary by Alignment Level ---")
    low_align = episodes_df[episodes_df['total_alignment'] <= 3]
    mid_align = episodes_df[(episodes_df['total_alignment'] >= 4) & (episodes_df['total_alignment'] <= 6)]
    high_align = episodes_df[episodes_df['total_alignment'] >= 7]

    log(f"\nLow alignment (0-3):  {len(low_align):,} episodes, {low_align['duration_hours'].mean():.1f}h avg, {low_align['is_sustained'].mean()*100:.1f}% sustained")
    log(f"Mid alignment (4-6):  {len(mid_align):,} episodes, {mid_align['duration_hours'].mean():.1f}h avg, {mid_align['is_sustained'].mean()*100:.1f}% sustained")
    log(f"High alignment (7+):  {len(high_align):,} episodes, {high_align['duration_hours'].mean():.1f}h avg, {high_align['is_sustained'].mean()*100:.1f}% sustained")

    # ============================================================================
    # KEY FINDING: Can HTF "spoil" the outcome?
    # ============================================================================

    log("\n" + "=" * 70)
    log("KEY FINDING: Can H4 Trend 'Spoil' MFC Reversal?")
    log("=" * 70)

    # When MFC is at extreme but H4 trend is OPPOSITE, does it reverse faster?
    episodes_df['h4_opposing'] = (episodes_df['h4_ma_trend'] != episodes_df['direction']).astype(int)

    h4_aligned_eps = episodes_df[episodes_df['h4_opposing'] == 0]
    h4_opposed_eps = episodes_df[episodes_df['h4_opposing'] == 1]

    log(f"\nH4 MA aligned with MFC:")
    log(f"  Episodes: {len(h4_aligned_eps):,}")
    log(f"  Avg duration: {h4_aligned_eps['duration_hours'].mean():.1f} hours")
    log(f"  Sustained (>2h): {h4_aligned_eps['is_sustained'].mean()*100:.1f}%")
    log(f"  Brief (<30m): {h4_aligned_eps['is_brief'].mean()*100:.1f}%")

    log(f"\nH4 MA OPPOSING MFC (potential spoiler):")
    log(f"  Episodes: {len(h4_opposed_eps):,}")
    log(f"  Avg duration: {h4_opposed_eps['duration_hours'].mean():.1f} hours")
    log(f"  Sustained (>2h): {h4_opposed_eps['is_sustained'].mean()*100:.1f}%")
    log(f"  Brief (<30m): {h4_opposed_eps['is_brief'].mean()*100:.1f}%")

    # Best predictor combination
    log("\n" + "=" * 70)
    log("BEST PREDICTOR COMBINATIONS")
    log("=" * 70)

    # H4 MA + H4 ADX > 25 (trending)
    strong_trend_aligned = episodes_df[
        (episodes_df['h4_ma_aligned'] == 1) &
        (episodes_df['h4_adx'] >= 25)
    ]
    weak_or_opposed = episodes_df[
        (episodes_df['h4_ma_aligned'] == 0) |
        (episodes_df['h4_adx'] < 20)
    ]

    log(f"\nH4 Trend Aligned + ADX>25:")
    log(f"  Episodes: {len(strong_trend_aligned):,}")
    log(f"  Avg duration: {strong_trend_aligned['duration_hours'].mean():.1f} hours")
    log(f"  Sustained: {strong_trend_aligned['is_sustained'].mean()*100:.1f}%")

    log(f"\nH4 Weak/Opposed (ADX<20 or MA misaligned):")
    log(f"  Episodes: {len(weak_or_opposed):,}")
    log(f"  Avg duration: {weak_or_opposed['duration_hours'].mean():.1f} hours")
    log(f"  Sustained: {weak_or_opposed['is_sustained'].mean()*100:.1f}%")

    # Save results
    output_path = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/lstm_model/v2_both_currencies/data/trend_mfc_episodes.pkl')
    episodes_df.to_pickle(output_path)
    log(f"\nSaved episodes to: {output_path}")

log(f"\nCompleted: {datetime.now()}")
