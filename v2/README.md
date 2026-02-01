# V2 Model: Price-Based Target

## Goal
Train a new model that predicts **price movement** instead of MFC cycle completion.

## Key Differences from V1

| Aspect | V1 | V2 |
|--------|-----|-----|
| Target | MFC cycle direction | Price move in 8h |
| Output | UP/NEUTRAL/DOWN | Binary or Multiclass |
| Logic | Will cycle complete? | Will trade profit? |
| Data Range | 2023-2024 only | 2013-2025 (12 years) |
| Samples | ~148k per currency | ~900k per currency |

## Target Types to Test

### Binary Target
- `1` = Price moved > 15 pips favorably within 8h
- `0` = Did not reach 15 pips

### Multiclass Target
- `0` = Loss (hit adverse threshold first)
- `1` = Flat (neither threshold hit within 8h)
- `2` = Win (hit favorable threshold)

## Models to Test

1. **LSTM** (same architecture as V1)
2. **XGBoost** (gradient boosting, for comparison)

## Scripts

| Script | Purpose | Run On |
|--------|---------|--------|
| `03_create_training_data.py` | Create X/Y with price targets | WSL/Windows |
| `04_train_lstm_v2.py` | Train LSTM model | WSL/Windows |
| `05_train_xgboost_v2.py` | Train XGBoost model | WSL/Windows |
| `06_compare_models.py` | Compare performance | WSL/Windows |

## Workflow

1. Run `03_create_training_data.py` to prepare data (uses existing M1 price data)
2. Train both LSTM and XGBoost
3. Compare results and pick best model
4. Backtest winning model on 2025 data

## Data Sources (Already Available)

See `DATA_STRUCTURE.md` for full paths.

### MFC Data
- M5, M15, M30: `v1-model/data/mfc_currency_{CUR}_{TF}.csv`
- H1, H4, D1: `v1-model/data/cleaned/mfc_currency_{CUR}_{TF}_clean.csv`

### Price Data
- M1 OHLC: `v1-model/data/{PAIR}_GMT+0_US-DST_M1.csv`

### Date Ranges
| Data | Start | End |
|------|-------|-----|
| MFC M5 | 2013-01-01 | 2025-04-18 |
| MFC H1 | 2008-01-07 | 2025-07-25 |
| Price M1 | 2010-02-01 | 2025-12-21 |

## Expected Improvement

V1 predicts MFC direction at 85% accuracy, but:
- MFC direction ≠ profitable trade
- Timing matters more than direction

V2 directly predicts "will this trade profit?" which is what we actually care about.

## V2 Advantages

1. **6x more training data** (12 years vs 2 years)
2. **Direct profit prediction** (not proxy metric)
3. **Both favorable and adverse moves** tracked (for risk assessment)
