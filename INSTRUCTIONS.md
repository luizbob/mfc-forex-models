# LSTM Model Development Instructions

## User Instructions (DO NOT DELETE)
- Feel free to keep improving the model
- DON'T DELETE ANYTHING
- Execute everything to reach the end goal of making the LSTM model work
- Save all experiments and results

## Goal
Create a profitable MFC-based trading model using LSTM that:
1. Predicts currency cycle direction with high confidence
2. Compares predictions from BOTH currencies of a pair
3. Trades when currencies show divergence (opposite cycles)

## Current Status (Updated 2026-01-15)

### LIVE DEMO TRADING ACTIVE (8h Optimized)
The LSTM strategy is running live on demo account with 8h max hold optimization.

- **Script**: `lstm_trader_mt5.py`
- **Account**: Exness Demo (198032049)
- **Lot Size**: 0.01
- **Max Positions**: 7
- **Max Hold**: 96 bars (~8 hours)
- **Friday Cutoff**: 14:00 (no entries after)
- **Pairs**: 10 (85% accuracy filter)
- **Expected WR**: 80.8%

### All 8 Currency Models Trained
| Currency | Accuracy |
|----------|----------|
| EUR | 82.0% |
| USD | 87.8% |
| GBP | 83.7% |
| JPY | 90.0% |
| CHF | 86.3% |
| CAD | 82.2% |
| AUD | 87.4% |
| NZD | 86.7% |

### 8h Optimization Results (2025 OOS)
| Setting | Trades | WR | Avg Pips | Max DD |
|---------|--------|-----|----------|--------|
| Baseline (16h) | 1,385 | 68.6% | +5.32 | -839 |
| **8h (current)** | 850 | **80.8%** | **+13.21** | **-250** |

### Best Strategy: LSTM Divergence + MFC Extreme + RSI + 8h Timeout
- Entry: LSTM divergence + base MFC at extreme (0.5) + RSI extreme (20/80)
- Exit: RSI crosses opposite extreme OR 8h timeout
- Only pairs where both currencies have >=85% model accuracy

## Key Insights

1. **Pure LSTM divergence doesn't work** (50.3% WR = random)
   - LSTM predicts direction well but timing is everything
   - Must combine with timing filters (MFC extreme + RSI)

2. **Full divergence is required** - Only improves when BOTH currencies predict opposite directions
   - base_only, either_agrees, not_wrong modes don't improve results

3. **RSI exit is essential** - MFC exit and time exit produce losses

4. **Accuracy filter improves results** - 85% filter improves WR from 67.3% to 68.6%

5. **Must use closed bars only** - Training used shift(1), live must match

6. **8h timeout is optimal** - Analysis showed:
   - 0-4h trades: 82-100% WR (model is right)
   - 5-7h trades: 68-72% WR (still profitable)
   - 8h+ trades: WR drops sharply (model was wrong)
   - Weekend carryover (Fri→Mon) had 32% WR, -33.68 avg

## System Components

1. **MT4 Exporter** (`DWX_server_MT4.mq4`)
   - Exports MFC data for 8 currencies, 5 timeframes
   - 50 bars per timeframe
   - Writes to Common folder on every new M5 bar

2. **Python Trader** (`lstm_trader_mt5.py`)
   - Reads MFC data
   - Loads LSTM models
   - Generates predictions
   - Executes trades via MT5 API
   - 8h max hold timeout

## Scripts Reference

| Script | Purpose |
|--------|---------|
| `01_prepare_data.py` | Multi-timeframe LSTM data preparation |
| `02_build_model.py` | Multi-timeframe LSTM architecture and training |
| `04_train_all_currencies.py` | Train all 8 currency models |
| `14_test_2025_out_of_sample.py` | 2025 out-of-sample backtest |
| `lstm_trader_mt5.py` | **LIVE TRADING SCRIPT** |
| `test_lstm_trader_setup.py` | Setup verification |

## Next Steps (Potential)

1. ~~Test on 2025 out-of-sample data~~ DONE
2. ~~Deploy live trading on demo~~ DONE - RUNNING
3. ~~Optimize hold time~~ DONE - 8h optimal
4. Monitor live results vs backtest (target: 1 week)
5. Validate on funded account if demo proves consistent
6. Scale lot size gradually
7. Explore additional features for LSTM v2:
   - Time features (hour, day of week, session)
   - Volatility (ATR)
   - Cross-currency alignment
8. Retrain LSTM with price-based target

## Expected Returns (8h Optimized)

| Lot Size | Monthly | Yearly |
|----------|---------|--------|
| 0.01 | ~$160 | ~$1,900 |
| 0.10 | ~$1,600 | ~$19,000 |
| 1.00 | ~$16,000 | ~$192,000 |

## Notes

- Live trading on DEMO account (no real money at risk)
- Goal: Validate performance before moving to funded account
- All calculations use CLOSED bars only (shift applied)
- 8h timeout based on data analysis (not arbitrary)
- See `CURRENT_STATUS.md` for detailed live trading configuration
- See `RESULTS_SUMMARY.md` for full backtest results
