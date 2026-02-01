# Momentum Surfing Strategy

## Discovery Date: 2025-01-31

## Core Insight

MFC returning to box from extreme is NOT a reversal signal - it's a momentum signal.
Fast MFC return = strong momentum that will likely continue through the box.

### Key Data Points (from test_mfc_return_type.py)
- Very fast return (velocity >0.08): 70-71% continues through box
- Fast return (0.05-0.08): 48-50% continues through
- Slow return (0.02-0.03): Only 19-21% continues (more likely to stall)

## Strategy Rules

### Entry
- MFC crosses INTO the box from extreme (above +0.2 or below -0.2)
- Entry velocity > 0.05 (momentum is strong)
- Direction: Trade WITH the momentum
  - MFC coming DOWN from above → SELL that currency
  - MFC coming UP from below → BUY that currency

### Exit
- Velocity falls below 0.05 threshold (momentum weakening)
- OR MFC reverses direction

## Backtest Results (M30, 2013-2026)

### Overall Performance
| Metric | Value |
|--------|-------|
| Total Trades | 43,613 |
| Win Rate | 80.9% |
| Avg MFC Move | +0.1640 |
| Avg Hold Time | 2.4 bars (~1.2 hours) |

### By Entry Velocity
| Entry Velocity | Trades | Win Rate | Avg Move |
|----------------|--------|----------|----------|
| 0.05-0.08 | 15,447 | 76.7% | +0.0896 |
| 0.08-0.12 | 14,244 | 81.5% | +0.1433 |
| 0.12-0.20 | 10,418 | 84.5% | +0.2210 |
| >0.20 | 3,464 | 86.8% | +0.4035 |

### Holdout Test (Last 3 Months)
- Period: 2025-10-15 to 2026-01-15
- Trades: 845
- Win Rate: 80.2%
- Avg MFC Move: +0.1597

## Threshold Tradeoff
| Threshold | Trades | Win Rate | Total MFC Move |
|-----------|--------|----------|----------------|
| 0.03 | 53,025 | 78.2% | +9,510 |
| 0.05 | 43,613 | 80.9% | +7,155 |
| 0.08 | 28,223 | 83.8% | +4,353 |
| 0.10 | 20,002 | 85.1% | +3,051 |

Lower threshold = more trades, lower win rate, higher total
Higher threshold = fewer trades, higher win rate, lower total

## M15 Momentum Model (Velocity-Based Exit)

### Exit Logic
Instead of fixed time exit, we exit when the velocity drops below a threshold:
- Entry: velocity > 0.08 (strong momentum)
- Exit: velocity drops below threshold (momentum weakening)

### V2 vs V3 Comparison

| Metric | V2 (exit 0.04) | V3 (exit 0.02) |
|--------|----------------|----------------|
| Total entries | 1,407,448 | 1,254,555 |
| Avg hold | 2.5 bars | 3.2 bars |
| Profitable % (raw) | 29.0% | 29.6% |

**XGBoost Model Results (threshold 0.45):**

| Metric | V2 (exit 0.04) | V3 (exit 0.02) |
|--------|----------------|----------------|
| Trades | 6,334 | 7,166 |
| Win Rate | 48.3% | 48.2% |
| Profit Factor | 1.74 | 1.78 |
| Avg Pips | +3.11 | +3.50 |

**By Bars Held (threshold 0.45):**

| Bars Held | V2 WR | V2 Avg | V3 WR | V3 Avg |
|-----------|-------|--------|-------|--------|
| 1-2 | 23.6% | -3.55 | 27.7% | -3.00 |
| 3-5 | 57.9% | +1.91 | 50.7% | +0.91 |
| 6-11 | 84.0% | +21.04 | 77.5% | +15.87 |
| 12+ | 91.2% | +36.42 | 87.5% | +33.99 |

### Key Discovery
**Trade duration predicts success:**
- Momentum that sustains 6+ bars = 77-91% win rate, +15 to +36 pips avg
- Momentum that dies in 1-2 bars = 23-28% win rate, losing trades

### Top Predictive Features
1. direction_code (buy/sell)
2. hour_cos (time of day)
3. base_vel_h4 (H4 velocity confirms momentum)
4. quote_vel_h1 / quote_vel_h4 (higher TF confirmation)
5. divergence (MFC difference between currencies)

### Recommendation
**Use V3 (exit 0.02)** for production:
- Higher profit factor (1.78 vs 1.74)
- Higher avg pips (+3.50 vs +3.11)
- Slightly longer hold (3.2 vs 2.5 bars) allows momentum to develop

### Model Files
- `models/momentum_xgb_m15_v2.joblib` - V2 model
- `models/momentum_xgb_m15_v3.joblib` - V3 model (recommended)
- `models/momentum_xgb_features_m15_v2.pkl` - V2 features
- `models/momentum_xgb_features_m15_v3.pkl` - V3 features

## Related Files
- `scripts/prepare_momentum_data_m15_v2.py` - Data prep with exit 0.04
- `scripts/prepare_momentum_data_m15_v3.py` - Data prep with exit 0.02
- `scripts/train_momentum_model_m15_v2.py` - Train V2 model
- `scripts/train_momentum_model_m15_v3.py` - Train V3 model
- `scripts/test_momentum_surf.py` - Main backtest script
- `scripts/test_mfc_return_type.py` - Analysis that led to this discovery
