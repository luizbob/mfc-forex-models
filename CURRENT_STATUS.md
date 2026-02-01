# LSTM Model - Current Status (2026-01-15)

## Live Demo Trading Active

The LSTM strategy is currently running on a demo account (Exness-MT5Trial11).

### Configuration
- **Account**: 198032049 (Demo)
- **Lot Size**: 0.01
- **Max Positions**: 7
- **One Per Currency**: False (multiple trades per currency allowed)
- **Magic Number**: 150002

### Strategy Parameters
- **Min LSTM Confidence**: 0.70
- **MFC Extreme**: 0.5 (base MFC threshold)
- **RSI Period**: 14
- **RSI Entry**: < 20 (BUY) / > 80 (SELL)
- **RSI Exit**: >= 80 (BUY exit) / <= 20 (SELL exit)
- **Max Hold**: 96 bars (~8 hours)
- **Friday Cutoff**: No entries after 14:00 (prevents weekend carryover)

### Accuracy Filter (85%)
Only trading pairs where BOTH currencies have model accuracy >= 85%:

| Currency | Accuracy |
|----------|----------|
| JPY | 90.0% |
| USD | 87.8% |
| AUD | 87.4% |
| NZD | 86.7% |
| CHF | 86.3% |

**Qualified Pairs (10):**
- USDJPY, AUDJPY, NZDJPY, CHFJPY
- AUDUSD, NZDUSD, USDCHF
- AUDNZD, AUDCHF, NZDCHF

## 8h Optimization (2026-01-15)

### Discovery: Weekend Carryover Problem
Analysis of worst loss days revealed:
- Friday → Monday carryover trades had **32% WR**, **-33.68 avg pips**
- These 25 trades lost **-842 pips** total
- Root cause: Weekend gaps and 60+ hour holds

### Hold Time Analysis
Win rate by hold time showed clear pattern:
- **0-4h**: 82-100% WR (model is RIGHT)
- **5-7h**: 68-72% WR (still profitable)
- **8h+**: Falls off a cliff (model was WRONG)

### Optimization Applied
Changed MAX_BARS_HOLD from 200 (16.7h) to **96 (8h)**

| Setting | Trades | WR | Avg Pips | Total | Max DD |
|---------|--------|-----|----------|-------|--------|
| Baseline (16h) | 1,385 | 68.6% | +5.32 | +7,364 | -839 |
| **8h (new)** | 850 | **80.8%** | **+13.21** | **+11,225** | **-250** |

### Expected Performance (8h model)
| Metric | Value |
|--------|-------|
| Win Rate | 80.8% |
| Avg Pips | +13.21 |
| Max Drawdown | -250 pips |
| Profitable Days | 87% |
| Profit/DD Ratio | 44.93x |

### Expected Returns (8h model)
| Lot Size | Monthly Est. | Yearly Est. |
|----------|-------------|-------------|
| 0.01 | ~$160 | ~$1,900 |
| 0.05 | ~$800 | ~$9,500 |
| 0.10 | ~$1,600 | ~$19,000 |
| 0.50 | ~$8,000 | ~$95,000 |
| 1.00 | ~$16,000 | ~$192,000 |

## System Architecture

```
MT4 (Exporter)                    Python                         MT5
      |                              |                             |
      | DWX_MFC_Auto.txt            |                             |
      | (50 bars per TF)            |                             |
      |      |                      |                             |
      +----->|  lstm_trader_mt5.py  |<--------------------------->|
             |  - Reads MFC data    |     MetaTrader5 API         |
             |  - LSTM predictions  |     - Get prices/RSI        |
             |  - Trade signals     |     - Open/Close trades     |
             +----------------------+                             |
```

### Components Running
1. **MT4 Expert Advisor**: `DWX_server_MT4.mq4`
   - Exports MFC data for all 8 currencies
   - Timeframes: M5, M15, M30, H1, H4
   - Bars exported: 50 per timeframe
   - Exports on init + every new M5 bar
   - Writes to: `Common/Files/DWX/DWX_MFC_Auto.txt`

2. **Python Trader**: `lstm_trader_mt5.py`
   - Reads MFC from Common folder
   - Loads 5 LSTM models (JPY, USD, AUD, CHF, NZD)
   - Scans 10 pairs every M5 bar
   - Uses only CLOSED bars (shift applied)
   - Manages entries and exits via MT5 API

## Important Fixes Applied

### 1. Closed Bars Only (No Forming Bar)
Training used `shift(1)`, so live trading must skip the forming bar:
- LSTM sequences: `[-(lookback+1):-1]` instead of `[-lookback:]`
- Aux features: `[-2]` and `[-3]` instead of `[-1]` and `[-2]`
- MFC extreme check: `[-2]` instead of `[-1]`
- RSI: Already correct with `shift=1`

### 2. File Read Safety
- 3 retry attempts with 100ms delay
- Validates JSON completeness
- Waits 5 seconds and retries if all attempts fail
- No stale cached data fallback

### 3. Timeout from Position Time
- Calculates bars held from actual MT5 position open time
- Works correctly even after script restart

### 4. 8h Max Hold Optimization
- Prevents weekend carryover (auto-closed before weekend)
- Cuts losing trades faster
- Improves WR from 68.6% to 80.8%

### 5. Friday Entry Cutoff
- No new entries after Friday 14:00
- Ensures all trades close before market closes
- Prevents any weekend gap risk

## Files

### Scripts
- `lstm_trader_mt5.py` - Main live trading script
- `test_lstm_trader_setup.py` - Setup verification
- `14_test_2025_out_of_sample.py` - 2025 backtest

### MT4
- `DWX_server_MT4.mq4` - MFC data exporter (modified for auto-export)

### Models
- `models/lstm_EUR_final.keras`
- `models/lstm_USD_final.keras`
- `models/lstm_GBP_final.keras`
- `models/lstm_JPY_final.keras`
- `models/lstm_CHF_final.keras`
- `models/lstm_CAD_final.keras`
- `models/lstm_AUD_final.keras`
- `models/lstm_NZD_final.keras`

### Data
- `data/config.pkl` - Model configuration (lookbacks, etc.)
- `data/trades_2025_oos.csv` - Backtest trades

## Next Steps (Potential)

1. Monitor live demo results vs backtest (target: 1 week)
2. Compare live WR to expected 80.8%
3. Test on funded account when validated
4. Scale lot size gradually
5. Explore additional features for LSTM v2:
   - Time features (hour, day of week, session)
   - Volatility (ATR)
   - Cross-currency alignment
6. Retrain with price-based targets

## Notes

- Model is running on demo - no real money at risk
- Goal: Validate live performance matches backtest
- If successful: Move to funded account for real returns
- All data uses CLOSED bars only to match training
- 8h timeout optimized based on data analysis (not arbitrary)
