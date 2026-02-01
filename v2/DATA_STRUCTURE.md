# V2 Data Structure

## Data Paths

### MFC Data (Multi-Factor Currency Indicator)

**M5, M15, M30 timeframes:**
```
/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/data/mfc_currency_{CUR}_{TF}.csv
```
Example: `mfc_currency_EUR_M5.csv`

**H1, H4, D1 timeframes (cleaned):**
```
/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/data/cleaned/mfc_currency_{CUR}_{TF}_clean.csv
```
Example: `mfc_currency_EUR_H1_clean.csv`

Currencies: EUR, USD, GBP, JPY, CHF, CAD, AUD, NZD

### Price Data (OHLC)

**M1 timeframe:**
```
/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/data/{PAIR}_GMT+0_US-DST_M1.csv
```
Example: `EURUSD_GMT+0_US-DST_M1.csv`

## Date Ranges

| Data Type | Start Date | End Date | Notes |
|-----------|------------|----------|-------|
| MFC M5 | 2013-01-01 | 2025-04-18 | ~916k rows per currency |
| MFC M15 | 2013-01-01 | 2025-04-18 | ~305k rows per currency |
| MFC M30 | 2013-01-01 | 2025-04-18 | ~153k rows per currency |
| MFC H1 | 2008-01-07 | 2025-07-25 | ~152k rows per currency |
| MFC H4 | 2008-01-07 | 2025-07-25 | ~38k rows per currency |
| MFC D1 | 2008-01-07 | 2025-07-25 | ~4k rows per currency |
| Price M1 | 2010-02-01 | 2025-12-21 | ~8M rows per pair |

## CSV Formats

### MFC Files
```csv
Date,Time,MFC
2013.01.01,22:00,0.17706179285714288
```

### Price Files
```csv
Date,Time,Open,High,Low,Close,Tick volume
2010.02.01,00:00:00,1.38724,1.38762,1.38718,1.38758,7
```

## V1 vs V2 Data Usage

| Version | Training Data | Samples per Currency |
|---------|--------------|---------------------|
| V1 | 2023-2024 only | ~148k |
| V2 | 2013-2025 (full) | ~916k (6x more) |

## Currency to Pair Mapping

For price-based targets, map currency to representative pair:

| Currency | Pair | Direction |
|----------|------|-----------|
| EUR | EURUSD | Direct (EUR up = price up) |
| USD | EURUSD | Inverse (USD up = price down) |
| GBP | GBPUSD | Direct |
| JPY | USDJPY | Inverse |
| CHF | USDCHF | Inverse |
| CAD | USDCAD | Inverse |
| AUD | AUDUSD | Direct |
| NZD | NZDUSD | Direct |

## Overlap Period for Training

The overlapping period with all data sources:
- Start: 2013-01-01 (MFC M5 starts)
- End: 2025-04-18 (MFC M5 ends)

This gives us ~12 years of aligned data for V2 training.
