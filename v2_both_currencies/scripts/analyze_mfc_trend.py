"""
MFC-Based Trend Detection
==========================
Find ways to detect trend using MFC itself (not price indicators).

Ideas:
1. H4 MFC position (> 0.3 = bullish trend)
2. MTF alignment (M5, H1, H4 same sign)
3. MFC "floor" - minimum MFC in last N bars (trend has higher floor)
4. Time at extreme - bars since MFC was below 0.3
5. H4 MFC velocity - sustained movement in one direction

Question: Which MFC pattern best predicts if M5 extreme will hold or reverse?
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
log("MFC-BASED TREND DETECTION ANALYSIS")
log("=" * 70)
log(f"Started: {datetime.now()}")

DATA_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/data')
CLEANED_DIR = DATA_DIR / 'cleaned'

CURRENCIES = ['EUR', 'USD', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD', 'NZD']

# Load MFC data
log("\nLoading MFC data...")
mfc_data = {}
for tf in ['M5', 'H1', 'H4']:
    mfc_data[tf] = {}
    for ccy in CURRENCIES:
        df = pd.read_csv(CLEANED_DIR / f'mfc_currency_{ccy}_{tf}_clean.csv')
        df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
        df = df.set_index('datetime').sort_index()
        mfc_data[tf][ccy] = df['MFC']

log("Building analysis dataset...")

all_events = []

for ccy in CURRENCIES:
    log(f"  Processing {ccy}...")

    m5 = mfc_data['M5'][ccy]
    h1 = mfc_data['H1'][ccy]
    h4 = mfc_data['H4'][ccy]

    # Shift H1/H4 by 1 bar of their own TF before ffill
    h1_shifted = h1.shift(1).reindex(m5.index, method='ffill')
    h4_shifted = h4.shift(1).reindex(m5.index, method='ffill')

    # Build dataframe
    df = pd.DataFrame(index=m5.index)
    df['m5'] = m5
    df['h1'] = h1_shifted
    df['h4'] = h4_shifted
    df = df.dropna()

    # Calculate MFC-based trend indicators
    # 1. H4 MFC position
    df['h4_bullish'] = (df['h4'] >= 0.3).astype(int)
    df['h4_bearish'] = (df['h4'] <= -0.3).astype(int)
    df['h4_neutral'] = ((df['h4'] > -0.3) & (df['h4'] < 0.3)).astype(int)

    # 2. H1 MFC position
    df['h1_bullish'] = (df['h1'] >= 0.3).astype(int)
    df['h1_bearish'] = (df['h1'] <= -0.3).astype(int)

    # 3. MTF alignment (H1 and H4 same direction)
    df['mtf_bullish'] = ((df['h1'] > 0) & (df['h4'] > 0)).astype(int)
    df['mtf_bearish'] = ((df['h1'] < 0) & (df['h4'] < 0)).astype(int)
    df['mtf_aligned'] = df['mtf_bullish'] | df['mtf_bearish']

    # 4. Strong MTF alignment (H1 >= 0.3 AND H4 >= 0.3, same sign)
    df['mtf_strong_bull'] = ((df['h1'] >= 0.3) & (df['h4'] >= 0.3)).astype(int)
    df['mtf_strong_bear'] = ((df['h1'] <= -0.3) & (df['h4'] <= -0.3)).astype(int)

    # 5. MFC floor (rolling min of last 50 M5 bars)
    df['m5_floor_50'] = df['m5'].rolling(50).min()
    df['m5_ceiling_50'] = df['m5'].rolling(50).max()

    # 6. H4 velocity (change over last bar)
    df['h4_vel'] = df['h4'].diff()

    # 7. Time since MFC was at opposite extreme
    # (implemented in the loop below)

    df = df.dropna()

    m5_arr = df['m5'].values
    h1_arr = df['h1'].values
    h4_arr = df['h4'].values
    n = len(df)

    # Find M5 extreme events and track what happens
    i = 100
    while i < n - 100:
        m5_val = m5_arr[i]

        # M5 at positive extreme
        if m5_val >= 0.45:
            # Track how long it stays above 0.3 (sustained) vs drops below 0
            duration_above_03 = 0
            crossed_zero = False
            max_drop = 0

            for j in range(i + 1, min(i + 200, n)):
                if m5_arr[j] >= 0.3:
                    duration_above_03 += 1
                if m5_arr[j] <= 0:
                    crossed_zero = True
                    break
                max_drop = min(max_drop, m5_arr[j] - m5_val)

            row = df.iloc[i]
            event = {
                'currency': ccy,
                'datetime': df.index[i],
                'direction': 'bullish',
                'm5_val': m5_val,
                'h1_val': row['h1'],
                'h4_val': row['h4'],
                'duration_above_03': duration_above_03,
                'crossed_zero': int(crossed_zero),
                'max_drop': max_drop,
                # Trend indicators at entry
                'h4_bullish': row['h4_bullish'],
                'h4_bearish': row['h4_bearish'],
                'h1_bullish': row['h1_bullish'],
                'mtf_bullish': row['mtf_bullish'],
                'mtf_strong_bull': row['mtf_strong_bull'],
                'm5_floor_50': row['m5_floor_50'],
                'h4_vel': row['h4_vel'],
            }
            all_events.append(event)
            i += 20  # Skip ahead

        # M5 at negative extreme
        elif m5_val <= -0.45:
            duration_below_minus03 = 0
            crossed_zero = False
            max_rise = 0

            for j in range(i + 1, min(i + 200, n)):
                if m5_arr[j] <= -0.3:
                    duration_below_minus03 += 1
                if m5_arr[j] >= 0:
                    crossed_zero = True
                    break
                max_rise = max(max_rise, m5_arr[j] - m5_val)

            row = df.iloc[i]
            event = {
                'currency': ccy,
                'datetime': df.index[i],
                'direction': 'bearish',
                'm5_val': m5_val,
                'h1_val': row['h1'],
                'h4_val': row['h4'],
                'duration_above_03': duration_below_minus03,  # renamed for consistency
                'crossed_zero': int(crossed_zero),
                'max_drop': max_rise,  # renamed for consistency
                # Trend indicators at entry
                'h4_bullish': row['h4_bearish'],  # For bearish, "bullish" means H4 aligned
                'h4_bearish': row['h4_bullish'],
                'h1_bullish': row['h1_bearish'],
                'mtf_bullish': row['mtf_bearish'],
                'mtf_strong_bull': row['mtf_strong_bear'],
                'm5_floor_50': -row['m5_ceiling_50'],  # Flip for bearish
                'h4_vel': -row['h4_vel'],  # Flip for bearish
            }
            all_events.append(event)
            i += 20
        else:
            i += 1

events_df = pd.DataFrame(all_events)
log(f"\nTotal M5 extreme events: {len(events_df):,}")

# ============================================================================
# ANALYSIS: Which MFC trend indicator predicts sustained extremes?
# ============================================================================

log("\n" + "=" * 70)
log("QUESTION: Which MFC pattern predicts sustained extreme vs quick reversal?")
log("=" * 70)

# Outcome metrics
events_df['sustained'] = (events_df['duration_above_03'] >= 24).astype(int)  # 2+ hours
events_df['quick_reversal'] = (events_df['crossed_zero'] == 1).astype(int)

log(f"\nOverall:")
log(f"  Sustained (stayed at extreme 2h+): {events_df['sustained'].mean()*100:.1f}%")
log(f"  Quick reversal (crossed 0): {events_df['quick_reversal'].mean()*100:.1f}%")
log(f"  Avg duration at extreme: {events_df['duration_above_03'].mean():.0f} bars ({events_df['duration_above_03'].mean()*5/60:.1f}h)")

# ============================================================================
# TEST 1: H4 MFC Position
# ============================================================================

log("\n" + "-" * 50)
log("TEST 1: H4 MFC Position")
log("-" * 50)

h4_aligned = events_df[events_df['h4_bullish'] == 1]
h4_opposed = events_df[events_df['h4_bearish'] == 1]
h4_neutral = events_df[(events_df['h4_bullish'] == 0) & (events_df['h4_bearish'] == 0)]

log(f"\nH4 ALIGNED with M5 extreme (H4 >= 0.3 same direction):")
log(f"  Events: {len(h4_aligned):,}")
log(f"  Sustained: {h4_aligned['sustained'].mean()*100:.1f}%")
log(f"  Quick reversal: {h4_aligned['quick_reversal'].mean()*100:.1f}%")
log(f"  Avg duration: {h4_aligned['duration_above_03'].mean():.0f} bars")

log(f"\nH4 OPPOSED to M5 extreme (H4 opposite direction):")
log(f"  Events: {len(h4_opposed):,}")
log(f"  Sustained: {h4_opposed['sustained'].mean()*100:.1f}%")
log(f"  Quick reversal: {h4_opposed['quick_reversal'].mean()*100:.1f}%")
log(f"  Avg duration: {h4_opposed['duration_above_03'].mean():.0f} bars")

log(f"\nH4 NEUTRAL (-0.3 to 0.3):")
log(f"  Events: {len(h4_neutral):,}")
log(f"  Sustained: {h4_neutral['sustained'].mean()*100:.1f}%")
log(f"  Quick reversal: {h4_neutral['quick_reversal'].mean()*100:.1f}%")
log(f"  Avg duration: {h4_neutral['duration_above_03'].mean():.0f} bars")

# ============================================================================
# TEST 2: MTF Alignment (H1 + H4)
# ============================================================================

log("\n" + "-" * 50)
log("TEST 2: MTF Alignment (H1 + H4 both aligned)")
log("-" * 50)

mtf_aligned = events_df[events_df['mtf_bullish'] == 1]
mtf_not_aligned = events_df[events_df['mtf_bullish'] == 0]

log(f"\nMTF ALIGNED (H1 and H4 both same direction as M5):")
log(f"  Events: {len(mtf_aligned):,}")
log(f"  Sustained: {mtf_aligned['sustained'].mean()*100:.1f}%")
log(f"  Quick reversal: {mtf_aligned['quick_reversal'].mean()*100:.1f}%")
log(f"  Avg duration: {mtf_aligned['duration_above_03'].mean():.0f} bars")

log(f"\nMTF NOT ALIGNED:")
log(f"  Events: {len(mtf_not_aligned):,}")
log(f"  Sustained: {mtf_not_aligned['sustained'].mean()*100:.1f}%")
log(f"  Quick reversal: {mtf_not_aligned['quick_reversal'].mean()*100:.1f}%")
log(f"  Avg duration: {mtf_not_aligned['duration_above_03'].mean():.0f} bars")

# ============================================================================
# TEST 3: Strong MTF Alignment (H1 >= 0.3 AND H4 >= 0.3)
# ============================================================================

log("\n" + "-" * 50)
log("TEST 3: STRONG MTF Alignment (H1 >= 0.3 AND H4 >= 0.3)")
log("-" * 50)

strong_aligned = events_df[events_df['mtf_strong_bull'] == 1]
not_strong = events_df[events_df['mtf_strong_bull'] == 0]

log(f"\nSTRONG MTF (H1 >= 0.3 AND H4 >= 0.3, same direction):")
log(f"  Events: {len(strong_aligned):,}")
log(f"  Sustained: {strong_aligned['sustained'].mean()*100:.1f}%")
log(f"  Quick reversal: {strong_aligned['quick_reversal'].mean()*100:.1f}%")
log(f"  Avg duration: {strong_aligned['duration_above_03'].mean():.0f} bars")

log(f"\nNOT STRONG MTF:")
log(f"  Events: {len(not_strong):,}")
log(f"  Sustained: {not_strong['sustained'].mean()*100:.1f}%")
log(f"  Quick reversal: {not_strong['quick_reversal'].mean()*100:.1f}%")
log(f"  Avg duration: {not_strong['duration_above_03'].mean():.0f} bars")

# ============================================================================
# TEST 4: M5 Floor (recent minimum)
# ============================================================================

log("\n" + "-" * 50)
log("TEST 4: M5 Floor (min MFC in last 50 bars)")
log("-" * 50)

# High floor = MFC hasn't dropped much recently = strong trend
high_floor = events_df[events_df['m5_floor_50'] >= 0.2]
mid_floor = events_df[(events_df['m5_floor_50'] >= 0) & (events_df['m5_floor_50'] < 0.2)]
low_floor = events_df[events_df['m5_floor_50'] < 0]

log(f"\nHIGH FLOOR (min >= 0.2, never dipped much):")
log(f"  Events: {len(high_floor):,}")
log(f"  Sustained: {high_floor['sustained'].mean()*100:.1f}%")
log(f"  Quick reversal: {high_floor['quick_reversal'].mean()*100:.1f}%")

log(f"\nMID FLOOR (min 0 to 0.2):")
log(f"  Events: {len(mid_floor):,}")
log(f"  Sustained: {mid_floor['sustained'].mean()*100:.1f}%")
log(f"  Quick reversal: {mid_floor['quick_reversal'].mean()*100:.1f}%")

log(f"\nLOW FLOOR (min < 0, crossed zero recently):")
log(f"  Events: {len(low_floor):,}")
log(f"  Sustained: {low_floor['sustained'].mean()*100:.1f}%")
log(f"  Quick reversal: {low_floor['quick_reversal'].mean()*100:.1f}%")

# ============================================================================
# TEST 5: H4 Velocity
# ============================================================================

log("\n" + "-" * 50)
log("TEST 5: H4 Velocity (momentum)")
log("-" * 50)

h4_vel_up = events_df[events_df['h4_vel'] > 0.02]
h4_vel_flat = events_df[(events_df['h4_vel'] >= -0.02) & (events_df['h4_vel'] <= 0.02)]
h4_vel_down = events_df[events_df['h4_vel'] < -0.02]

log(f"\nH4 Velocity UP (> 0.02, momentum with trend):")
log(f"  Events: {len(h4_vel_up):,}")
log(f"  Sustained: {h4_vel_up['sustained'].mean()*100:.1f}%")
log(f"  Quick reversal: {h4_vel_up['quick_reversal'].mean()*100:.1f}%")

log(f"\nH4 Velocity FLAT:")
log(f"  Events: {len(h4_vel_flat):,}")
log(f"  Sustained: {h4_vel_flat['sustained'].mean()*100:.1f}%")
log(f"  Quick reversal: {h4_vel_flat['quick_reversal'].mean()*100:.1f}%")

log(f"\nH4 Velocity DOWN (< -0.02, momentum against):")
log(f"  Events: {len(h4_vel_down):,}")
log(f"  Sustained: {h4_vel_down['sustained'].mean()*100:.1f}%")
log(f"  Quick reversal: {h4_vel_down['quick_reversal'].mean()*100:.1f}%")

# ============================================================================
# TEST 6: H4 MFC Level (continuous)
# ============================================================================

log("\n" + "-" * 50)
log("TEST 6: H4 MFC Level (continuous)")
log("-" * 50)

for h4_thresh in [0.0, 0.2, 0.3, 0.4, 0.5]:
    subset = events_df[events_df['h4_val'] >= h4_thresh]
    if len(subset) > 100:
        log(f"H4 >= {h4_thresh}: {len(subset):,} events, sustained {subset['sustained'].mean()*100:.1f}%, reversal {subset['quick_reversal'].mean()*100:.1f}%")

# ============================================================================
# BEST COMBINATION
# ============================================================================

log("\n" + "=" * 70)
log("BEST COMBINATIONS FOR PREDICTING TREND CONTINUATION")
log("=" * 70)

# Combo 1: H4 aligned + high floor
combo1 = events_df[(events_df['h4_bullish'] == 1) & (events_df['m5_floor_50'] >= 0.2)]
log(f"\nH4 aligned + High floor (min >= 0.2):")
log(f"  Events: {len(combo1):,}")
log(f"  Sustained: {combo1['sustained'].mean()*100:.1f}%")
log(f"  Quick reversal: {combo1['quick_reversal'].mean()*100:.1f}%")

# Combo 2: Strong MTF + H4 vel positive
combo2 = events_df[(events_df['mtf_strong_bull'] == 1) & (events_df['h4_vel'] > 0)]
log(f"\nStrong MTF + H4 velocity positive:")
log(f"  Events: {len(combo2):,}")
log(f"  Sustained: {combo2['sustained'].mean()*100:.1f}%")
log(f"  Quick reversal: {combo2['quick_reversal'].mean()*100:.1f}%")

# Combo 3: H4 >= 0.4 (very strong)
combo3 = events_df[events_df['h4_val'] >= 0.4]
log(f"\nH4 >= 0.4 (very strong higher TF):")
log(f"  Events: {len(combo3):,}")
log(f"  Sustained: {combo3['sustained'].mean()*100:.1f}%")
log(f"  Quick reversal: {combo3['quick_reversal'].mean()*100:.1f}%")

# Anti-combo: Best for mean reversion
log("\n" + "-" * 50)
log("BEST FOR MEAN REVERSION (quick reversal likely)")
log("-" * 50)

# H4 opposed or neutral + low floor
anti1 = events_df[(events_df['h4_bullish'] == 0) & (events_df['m5_floor_50'] < 0)]
log(f"\nH4 NOT aligned + Low floor (crossed 0 recently):")
log(f"  Events: {len(anti1):,}")
log(f"  Sustained: {anti1['sustained'].mean()*100:.1f}%")
log(f"  Quick reversal: {anti1['quick_reversal'].mean()*100:.1f}%")

# H4 opposed
anti2 = events_df[events_df['h4_bearish'] == 1]
log(f"\nH4 OPPOSED (opposite direction):")
log(f"  Events: {len(anti2):,}")
log(f"  Sustained: {anti2['sustained'].mean()*100:.1f}%")
log(f"  Quick reversal: {anti2['quick_reversal'].mean()*100:.1f}%")

log("\n" + "=" * 70)
log("SUMMARY: MFC-BASED TREND INDICATORS")
log("=" * 70)
log("""
Best predictors of SUSTAINED extreme (trend continuation):
  1. H4 MFC >= 0.4 - very strong signal
  2. Strong MTF alignment (H1 >= 0.3 AND H4 >= 0.3)
  3. High M5 floor (hasn't dipped below 0.2 recently)

Best predictors of QUICK REVERSAL (mean reversion opportunity):
  1. H4 MFC opposed (opposite direction)
  2. Low M5 floor (crossed 0 recently)
  3. H4 velocity against the extreme

Trading implications:
  - For MEAN REVERSION: Trade when H4 opposed or neutral
  - For MOMENTUM: Trade when H4 strongly aligned (>= 0.4)
""")

log(f"\nCompleted: {datetime.now()}")
