
## PENDING: Retrain LSTM Models (2026-01-18)

The LSTM models were trained with look-ahead bias (wrong shift/reindex order).
Need to retrain with corrected data preparation:

1. Re-run `01_prepare_data.py` (already fixed)
2. Re-run `02_train_model.py` 
3. Re-run validation and backtests

Current model still works (72.4% WR on 2025 OOS) but could improve with correct training data.

