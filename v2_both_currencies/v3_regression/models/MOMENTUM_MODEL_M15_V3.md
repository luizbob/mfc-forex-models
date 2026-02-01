# Momentum Model M15 V3

## Entry Logic

**Trigger**: MFC returning FROM extreme INTO box with high velocity

```
Conditions:
1. MFC was in extreme zone (above +0.2 or below -0.2)
2. MFC velocity > 0.08 (fast move)
3. MFC now entering box zone (-0.2 to +0.2)

Direction:
- MFC coming DOWN from above → SELL that currency
- MFC coming UP from below → BUY that currency

For pairs:
- Base currency triggers: trade the pair direction
- Quote currency triggers: trade opposite direction
```

## Exit Logic

**Velocity-based exit**: Exit when momentum weakens

```
Exit when:
- Velocity drops below 0.02 threshold
- OR max 24 bars (6 hours)
```

## Filter

**XGBoost model**: Probability threshold ≥ 0.45

```
Model: momentum_xgb_m15_v3.joblib
Features: momentum_xgb_features_m15_v3.pkl

43 features including:
- MFC values (M15, M30, H1, H4)
- Velocities (all timeframes)
- Momentum, acceleration
- Divergence between currencies
- H4 confirmation flags
- Time of day (hour_sin, hour_cos)
- Direction and trigger type
```

## Walk-Forward Results (2018-2025)

| Year | Trades | Win % | PF | Avg Pips | Net Pips |
|------|--------|-------|-----|----------|----------|
| 2018 | 1,156 | 54.7% | 2.77 | +7.19 | +8,314 |
| 2019 | 1,441 | 51.5% | 1.97 | +4.28 | +6,172 |
| 2020 | 1,074 | 53.2% | 1.98 | +5.73 | +6,152 |
| 2021 | 1,062 | 53.8% | 2.53 | +5.74 | +6,094 |
| 2022 | 1,435 | 59.2% | 3.59 | +14.14 | +20,294 |
| 2023 | 1,659 | 58.7% | 3.96 | +11.50 | +19,073 |
| 2024 | 2,474 | 47.2% | 1.53 | +2.53 | +6,270 |
| 2025 | 4,248 | 47.4% | 1.53 | +2.43 | +10,320 |
| **TOTAL** | **14,549** | **~53%** | **~2.5** | **+5.68** | **+82,690** |

## Key Insights

1. **Win rate is modest (~53%)** but winners are bigger than losers
2. **Profit Factor 1.5-4.0** depending on market conditions
3. **Consistent**: Profitable all 8 years tested
4. **Trade duration predicts success**:
   - 1-2 bars held = ~25% WR (losers)
   - 6+ bars held = ~80% WR (big winners)

## Files

```
models/
  momentum_xgb_m15_v3.joblib      # Trained model
  momentum_xgb_features_m15_v3.pkl # Feature list
  MOMENTUM_MODEL_M15_V3.md         # This file

scripts/
  prepare_momentum_data_m15_v3.py  # Data preparation
  train_momentum_model_m15_v3.py   # Model training
  walkforward_momentum_m15.py      # Walk-forward validation

data/
  momentum_data_m15_v3.pkl         # Prepared dataset
```

## Spreads Used (pips)

```
EURUSD: 0.8, GBPUSD: 1.0, USDJPY: 1.0
AUDUSD: 0.9, USDCAD: 1.5, USDCHF: 1.3
EURJPY: 2.4, GBPJPY: 2.2, EURGBP: 1.4
... (see prepare script for full list)
```

## Notes

- All MFC values are shifted by 1 bar to avoid look-ahead bias
- Higher timeframes (H4, H1) reindexed to M15 with forward-fill
- Spread is subtracted from each trade result
- Model should be retrained periodically (2024-2025 showed some degradation)
