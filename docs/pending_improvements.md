# Pending Improvements for LSTM Trader

Based on analysis of 2025 data with XGBoost filter (prob >= 0.75).

## Current Strategy Performance
- Trades: 1,333
- Total Pips: +5,725
- Win Rate: 73%

## Recommended Improvements (Not Yet Implemented)

### 1. Hour Filter (+878 pips improvement)
Remove hours: 5, 6, 7, 20, 23

| Hour | Trades | Pips | WR |
|------|--------|------|-----|
| 5 | 32 | -178 | 66% |
| 6 | 37 | -244 | 59% |
| 7 | 44 | -293 | 50% |
| 20 | 14 | -42 | 50% |
| 23 | 27 | -120 | 56% |

### 2. Pair Filter (+138 pips improvement)
Remove pairs: GBPAUD, EURCHF

| Pair | Trades | Pips | WR |
|------|--------|------|-----|
| GBPAUD | 33 | -124 | 61% |
| EURCHF | 15 | -14 | 67% |

### 3. Sunday Filter
Remove Sunday trades: 25 trades, -7 pips, 44% WR

### Combined Impact
- After all filters: 1,135 trades, +6,697 pips
- Improvement: +972 pips (+17%)

## Other Potential Filters (Quality vs Quantity Tradeoffs)

### Higher MFC Difference
- |mfc_diff| >= 1.25: 551 trades, 77% WR, +3,495 pips

### Higher Velocity Difference
- |vel_h1_diff| >= 0.04: 991 trades, 74.3% WR, +5,571 pips

### Higher XGB Probability
- prob >= 0.90: 284 trades, 77.1% WR, +2,192 pips
- prob >= 0.95: 77 trades, 89.6% WR, +905 pips

## 2-Hour Time Stop (Minor)
- Improvement: +406 pips
- Not recommended as primary improvement

---
Last updated: 2026-01-20
