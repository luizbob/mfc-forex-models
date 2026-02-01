"""
MFC vs Price Divergence Analysis
=================================
Key Question: When MFC falls from 0.5 to 0, does price follow?
Can H1/H4 trend indicators predict when price WON'T follow MFC?

Scenario:
- MFC at +0.5 (currency strong)
- MFC drops back to 0 (mean reversion)
- But PRICE stays high or keeps rising = strong trend, new "normal"

If H4 trend is bullish, maybe price won't drop even when MFC reverts.
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
log("MFC vs PRICE DIVERGENCE ANALYSIS")
log("=" * 70)
log(f"Started: {datetime.now()}")

DATA_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/data')
CLEANED_DIR = DATA_DIR / 'cleaned'

CURRENCIES = ['EUR', 'USD', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD', 'NZD']

ALL_PAIRS = [
    ('EURUSD', 'EUR', 'USD'), ('GBPUSD', 'GBP', 'USD'), ('AUDUSD', 'AUD', 'USD'),
    ('NZDUSD', 'NZD', 'USD'), ('USDJPY', 'USD', 'JPY'), ('USDCHF', 'USD', 'CHF'),
    ('USDCAD', 'USD', 'CAD'), ('EURGBP', 'EUR', 'GBP'), ('EURJPY', 'EUR', 'JPY'),
    ('GBPJPY', 'GBP', 'JPY'), ('AUDJPY', 'AUD', 'JPY'), ('EURAUD', 'EUR', 'AUD'),
]

# ============================================================================
# INDICATOR FUNCTIONS
# ============================================================================

def calc_sma(series, period):
    return series.rolling(window=period).mean()

def calc_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def calc_adx(high, low, close, period=14):
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()

    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

    plus_di = 100 * pd.Series(plus_dm, index=high.index).rolling(period).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=high.index).rolling(period).mean() / atr

    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=period).mean()

    return adx, plus_di, minus_di

# ============================================================================
# LOAD DATA
# ============================================================================

log("\nLoading MFC data...")
mfc_m5 = {}
for ccy in CURRENCIES:
    df = pd.read_csv(CLEANED_DIR / f'mfc_currency_{ccy}_M5_clean.csv')
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df = df.set_index('datetime').sort_index()
    mfc_m5[ccy] = df['MFC']

log("Loading price data...")

def load_price(pair):
    fp = DATA_DIR / f'{pair}_GMT+0_US-DST_M1.csv'
    if not fp.exists():
        return None, None, None

    chunks_m5 = []
    chunks_h1 = []
    chunks_h4 = []

    for chunk in pd.read_csv(fp, chunksize=500000):
        chunk['datetime'] = pd.to_datetime(chunk['Date'] + ' ' + chunk['Time'])
        chunk = chunk.set_index('datetime')
        chunk = chunk[chunk.index >= '2020-01-01']
        if len(chunk) > 0:
            # M5
            m5 = chunk[['Open', 'High', 'Low', 'Close']].resample('5min').agg({
                'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
            }).dropna()
            chunks_m5.append(m5)

            # H1
            h1 = chunk[['Open', 'High', 'Low', 'Close']].resample('1h').agg({
                'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
            }).dropna()
            chunks_h1.append(h1)

            # H4
            h4 = chunk[['Open', 'High', 'Low', 'Close']].resample('4h').agg({
                'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
            }).dropna()
            chunks_h4.append(h4)

    if chunks_m5:
        m5 = pd.concat(chunks_m5)
        h1 = pd.concat(chunks_h1)
        h4 = pd.concat(chunks_h4)
        m5 = m5[~m5.index.duplicated(keep='first')]
        h1 = h1[~h1.index.duplicated(keep='first')]
        h4 = h4[~h4.index.duplicated(keep='first')]
        return m5, h1, h4
    return None, None, None

# ============================================================================
# ANALYZE: When MFC reverts from extreme, does price follow?
# ============================================================================

log("\n" + "=" * 70)
log("FINDING MFC REVERSION EVENTS")
log("=" * 70)

all_events = []

for pair, base, quote in ALL_PAIRS:
    log(f"\nProcessing {pair}...")

    m5_price, h1_price, h4_price = load_price(pair)
    if m5_price is None:
        continue

    base_mfc = mfc_m5[base].reindex(m5_price.index, method='ffill')

    # Calculate H1 and H4 indicators
    h1_ma50 = calc_sma(h1_price['Close'], 50)
    h1_ma200 = calc_sma(h1_price['Close'], 200)
    h1_adx, h1_plus_di, h1_minus_di = calc_adx(h1_price['High'], h1_price['Low'], h1_price['Close'], 14)

    h4_ma50 = calc_sma(h4_price['Close'], 50)
    h4_ma200 = calc_sma(h4_price['Close'], 200)
    h4_adx, h4_plus_di, h4_minus_di = calc_adx(h4_price['High'], h4_price['Low'], h4_price['Close'], 14)

    # Create H1/H4 trend signals
    h1_trend = pd.DataFrame(index=h1_price.index)
    h1_trend['above_ma200'] = (h1_price['Close'] > h1_ma200).astype(int)
    h1_trend['ma50_above_200'] = (h1_ma50 > h1_ma200).astype(int)
    h1_trend['adx'] = h1_adx
    h1_trend['trend_up'] = (h1_plus_di > h1_minus_di).astype(int)

    h4_trend = pd.DataFrame(index=h4_price.index)
    h4_trend['above_ma200'] = (h4_price['Close'] > h4_ma200).astype(int)
    h4_trend['ma50_above_200'] = (h4_ma50 > h4_ma200).astype(int)
    h4_trend['adx'] = h4_adx
    h4_trend['trend_up'] = (h4_plus_di > h4_minus_di).astype(int)

    # IMPORTANT: Shift by 1 bar of THEIR OWN timeframe before ffill
    # This ensures we only use CLOSED bar data (no look-ahead bias)
    h1_trend = h1_trend.shift(1)
    h4_trend = h4_trend.shift(1)

    # Align to M5
    h1_aligned = h1_trend.reindex(m5_price.index, method='ffill')
    h4_aligned = h4_trend.reindex(m5_price.index, method='ffill')

    # Build analysis dataframe
    df = pd.DataFrame(index=m5_price.index)
    df['price'] = m5_price['Close']
    df['base_mfc'] = base_mfc
    df['h1_above_ma200'] = h1_aligned['above_ma200']
    df['h1_ma_cross'] = h1_aligned['ma50_above_200']
    df['h1_adx'] = h1_aligned['adx']
    df['h1_trend_up'] = h1_aligned['trend_up']
    df['h4_above_ma200'] = h4_aligned['above_ma200']
    df['h4_ma_cross'] = h4_aligned['ma50_above_200']
    df['h4_adx'] = h4_aligned['adx']
    df['h4_trend_up'] = h4_aligned['trend_up']

    df = df.dropna()

    mfc_arr = df['base_mfc'].values
    price_arr = df['price'].values
    n = len(df)

    # Find events: MFC at 0.5, then drops to 0
    i = 100
    while i < n - 100:
        mfc_val = mfc_arr[i]

        # Check for MFC at extreme (positive or negative)
        if abs(mfc_val) >= 0.45:
            direction = 1 if mfc_val > 0 else -1  # 1 = bullish extreme, -1 = bearish
            extreme_idx = i
            extreme_price = price_arr[i]

            # Look for MFC to cross back to 0 (within 100 bars)
            revert_idx = None
            for j in range(i + 1, min(i + 100, n)):
                # MFC crossed through 0
                if direction == 1 and mfc_arr[j] <= 0.05:
                    revert_idx = j
                    break
                elif direction == -1 and mfc_arr[j] >= -0.05:
                    revert_idx = j
                    break

            if revert_idx is not None:
                revert_price = price_arr[revert_idx]
                bars_to_revert = revert_idx - extreme_idx

                # Calculate price change during MFC reversion
                # For BUY direction (base strong): if price stayed up, that's "divergence"
                # For SELL direction (base weak): if price stayed down, that's "divergence"

                if direction == 1:  # Was bullish (MFC +0.5)
                    # Price should have dropped if following MFC, but if it stayed up...
                    price_change_pips = (revert_price - extreme_price) * 10000
                    price_followed = price_change_pips < -10  # Price dropped > 10 pips
                    price_diverged = price_change_pips > 10   # Price rose > 10 pips (didn't follow)
                else:  # Was bearish (MFC -0.5)
                    price_change_pips = (revert_price - extreme_price) * 10000
                    price_followed = price_change_pips > 10   # Price rose > 10 pips
                    price_diverged = price_change_pips < -10  # Price dropped (didn't follow)

                # Get H1/H4 indicators at the extreme point
                row = df.iloc[extreme_idx]

                event = {
                    'pair': pair,
                    'base': base,
                    'datetime': df.index[extreme_idx],
                    'direction': 'bullish' if direction == 1 else 'bearish',
                    'mfc_at_extreme': mfc_val,
                    'bars_to_revert': bars_to_revert,
                    'price_change_pips': price_change_pips,
                    'price_followed_mfc': price_followed,
                    'price_diverged': price_diverged,
                    # H1 indicators
                    'h1_above_ma200': row['h1_above_ma200'],
                    'h1_ma_cross': row['h1_ma_cross'],
                    'h1_adx': row['h1_adx'],
                    'h1_trend_up': row['h1_trend_up'],
                    # H4 indicators
                    'h4_above_ma200': row['h4_above_ma200'],
                    'h4_ma_cross': row['h4_ma_cross'],
                    'h4_adx': row['h4_adx'],
                    'h4_trend_up': row['h4_trend_up'],
                }
                all_events.append(event)

                i = revert_idx + 10  # Skip ahead
            else:
                i += 1
        else:
            i += 1

    log(f"  Found {len([e for e in all_events if e['pair'] == pair])} reversion events")

events_df = pd.DataFrame(all_events)
log(f"\n\nTotal MFC reversion events: {len(events_df):,}")

# ============================================================================
# ANALYSIS: When does price NOT follow MFC?
# ============================================================================

log("\n" + "=" * 70)
log("KEY QUESTION: When does price NOT follow MFC reversion?")
log("=" * 70)

# Split by direction
bullish = events_df[events_df['direction'] == 'bullish']
bearish = events_df[events_df['direction'] == 'bearish']

log(f"\nBullish extremes (MFC +0.5 → 0): {len(bullish):,}")
log(f"  Price followed (dropped): {bullish['price_followed_mfc'].sum():,} ({bullish['price_followed_mfc'].mean()*100:.1f}%)")
log(f"  Price diverged (stayed up): {bullish['price_diverged'].sum():,} ({bullish['price_diverged'].mean()*100:.1f}%)")

log(f"\nBearish extremes (MFC -0.5 → 0): {len(bearish):,}")
log(f"  Price followed (rose): {bearish['price_followed_mfc'].sum():,} ({bearish['price_followed_mfc'].mean()*100:.1f}%)")
log(f"  Price diverged (stayed down): {bearish['price_diverged'].sum():,} ({bearish['price_diverged'].mean()*100:.1f}%)")

# ============================================================================
# H4 TREND PREDICTION
# ============================================================================

log("\n" + "=" * 70)
log("H4 TREND AS PREDICTOR OF DIVERGENCE")
log("=" * 70)

# For bullish MFC (was +0.5): if H4 is bullish, price might stay up
log("\n--- Bullish MFC Events (MFC was +0.5, dropped to 0) ---")
log("Question: When H4 trend is UP, does price stay up instead of dropping?")

bull_h4_up = bullish[bullish['h4_trend_up'] == 1]
bull_h4_down = bullish[bullish['h4_trend_up'] == 0]

log(f"\nH4 Trend UP (should support price staying high):")
log(f"  Events: {len(bull_h4_up):,}")
log(f"  Price diverged (stayed up): {bull_h4_up['price_diverged'].mean()*100:.1f}%")
log(f"  Avg price change: {bull_h4_up['price_change_pips'].mean():+.1f} pips")

log(f"\nH4 Trend DOWN (price should follow MFC down):")
log(f"  Events: {len(bull_h4_down):,}")
log(f"  Price diverged (stayed up): {bull_h4_down['price_diverged'].mean()*100:.1f}%")
log(f"  Avg price change: {bull_h4_down['price_change_pips'].mean():+.1f} pips")

# For bearish MFC
log("\n--- Bearish MFC Events (MFC was -0.5, rose to 0) ---")
log("Question: When H4 trend is DOWN, does price stay down instead of rising?")

bear_h4_up = bearish[bearish['h4_trend_up'] == 1]
bear_h4_down = bearish[bearish['h4_trend_up'] == 0]

log(f"\nH4 Trend DOWN (should support price staying low):")
log(f"  Events: {len(bear_h4_down):,}")
log(f"  Price diverged (stayed down): {bear_h4_down['price_diverged'].mean()*100:.1f}%")
log(f"  Avg price change: {bear_h4_down['price_change_pips'].mean():+.1f} pips")

log(f"\nH4 Trend UP (price should follow MFC up):")
log(f"  Events: {len(bear_h4_up):,}")
log(f"  Price diverged (stayed down): {bear_h4_up['price_diverged'].mean()*100:.1f}%")
log(f"  Avg price change: {bear_h4_up['price_change_pips'].mean():+.1f} pips")

# ============================================================================
# H4 MA200 AS PREDICTOR
# ============================================================================

log("\n" + "=" * 70)
log("H4 PRICE vs MA200 AS PREDICTOR")
log("=" * 70)

log("\n--- Bullish MFC (was +0.5) ---")
bull_above_ma = bullish[bullish['h4_above_ma200'] == 1]
bull_below_ma = bullish[bullish['h4_above_ma200'] == 0]

log(f"\nH4 Price ABOVE MA200 (strong uptrend):")
log(f"  Events: {len(bull_above_ma):,}")
log(f"  Price stayed up (diverged): {bull_above_ma['price_diverged'].mean()*100:.1f}%")
log(f"  Avg price change: {bull_above_ma['price_change_pips'].mean():+.1f} pips")

log(f"\nH4 Price BELOW MA200 (weak/downtrend):")
log(f"  Events: {len(bull_below_ma):,}")
log(f"  Price stayed up (diverged): {bull_below_ma['price_diverged'].mean()*100:.1f}%")
log(f"  Avg price change: {bull_below_ma['price_change_pips'].mean():+.1f} pips")

log("\n--- Bearish MFC (was -0.5) ---")
bear_above_ma = bearish[bearish['h4_above_ma200'] == 1]
bear_below_ma = bearish[bearish['h4_above_ma200'] == 0]

log(f"\nH4 Price BELOW MA200 (strong downtrend):")
log(f"  Events: {len(bear_below_ma):,}")
log(f"  Price stayed down (diverged): {bear_below_ma['price_diverged'].mean()*100:.1f}%")
log(f"  Avg price change: {bear_below_ma['price_change_pips'].mean():+.1f} pips")

log(f"\nH4 Price ABOVE MA200 (weak/uptrend):")
log(f"  Events: {len(bear_above_ma):,}")
log(f"  Price stayed down (diverged): {bear_above_ma['price_diverged'].mean()*100:.1f}%")
log(f"  Avg price change: {bear_above_ma['price_change_pips'].mean():+.1f} pips")

# ============================================================================
# ADX STRENGTH
# ============================================================================

log("\n" + "=" * 70)
log("H4 ADX (TREND STRENGTH) AS PREDICTOR")
log("=" * 70)

log("\n--- Bullish MFC: Does strong H4 trend keep price up? ---")
for adx_thresh in [20, 25, 30]:
    strong = bullish[bullish['h4_adx'] >= adx_thresh]
    weak = bullish[bullish['h4_adx'] < adx_thresh]
    if len(strong) > 50 and len(weak) > 50:
        log(f"\nADX >= {adx_thresh} (trending): {len(strong):,} events, price diverged {strong['price_diverged'].mean()*100:.1f}%, avg change {strong['price_change_pips'].mean():+.1f}")
        log(f"ADX <  {adx_thresh} (ranging):  {len(weak):,} events, price diverged {weak['price_diverged'].mean()*100:.1f}%, avg change {weak['price_change_pips'].mean():+.1f}")

# ============================================================================
# COMBINED: H4 Trend Aligned + Strong ADX
# ============================================================================

log("\n" + "=" * 70)
log("BEST PREDICTOR: H4 Trend Direction + ADX Combined")
log("=" * 70)

# For bullish MFC: H4 trend up + ADX > 25 = price should stay up
bull_aligned_strong = bullish[(bullish['h4_trend_up'] == 1) & (bullish['h4_adx'] >= 25)]
bull_opposed_weak = bullish[(bullish['h4_trend_up'] == 0) | (bullish['h4_adx'] < 20)]

log(f"\nBullish MFC + H4 Uptrend + ADX>25 (strong trend support):")
log(f"  Events: {len(bull_aligned_strong):,}")
log(f"  Price STAYED UP: {bull_aligned_strong['price_diverged'].mean()*100:.1f}%")
log(f"  Avg change: {bull_aligned_strong['price_change_pips'].mean():+.1f} pips")

log(f"\nBullish MFC + H4 Downtrend or ADX<20 (weak/opposite trend):")
log(f"  Events: {len(bull_opposed_weak):,}")
log(f"  Price STAYED UP: {bull_opposed_weak['price_diverged'].mean()*100:.1f}%")
log(f"  Avg change: {bull_opposed_weak['price_change_pips'].mean():+.1f} pips")

# For bearish MFC: H4 trend down + ADX > 25 = price should stay down
bear_aligned_strong = bearish[(bearish['h4_trend_up'] == 0) & (bearish['h4_adx'] >= 25)]
bear_opposed_weak = bearish[(bearish['h4_trend_up'] == 1) | (bearish['h4_adx'] < 20)]

log(f"\nBearish MFC + H4 Downtrend + ADX>25 (strong trend support):")
log(f"  Events: {len(bear_aligned_strong):,}")
log(f"  Price STAYED DOWN: {bear_aligned_strong['price_diverged'].mean()*100:.1f}%")
log(f"  Avg change: {bear_aligned_strong['price_change_pips'].mean():+.1f} pips")

log(f"\nBearish MFC + H4 Uptrend or ADX<20 (weak/opposite trend):")
log(f"  Events: {len(bear_opposed_weak):,}")
log(f"  Price STAYED DOWN: {bear_opposed_weak['price_diverged'].mean()*100:.1f}%")
log(f"  Avg change: {bear_opposed_weak['price_change_pips'].mean():+.1f} pips")

# ============================================================================
# TRADING IMPLICATION
# ============================================================================

log("\n" + "=" * 70)
log("TRADING IMPLICATION")
log("=" * 70)
log("""
If H4 trend is ALIGNED with MFC extreme direction and ADX > 25:
  → Price is less likely to follow MFC back to 0
  → Mean reversion trades are RISKIER
  → Consider NOT trading mean reversion in this case

If H4 trend is OPPOSITE to MFC extreme direction or ADX < 20:
  → Price is more likely to follow MFC
  → Mean reversion trades are SAFER
  → This is when mean reversion works best
""")

log(f"\nCompleted: {datetime.now()}")
