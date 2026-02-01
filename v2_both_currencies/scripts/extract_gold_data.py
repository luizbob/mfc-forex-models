"""
Extract Gold (XAUUSD) data from MetaTrader 5
=============================================
"""
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

def log(msg=""):
    print(msg, flush=True)

log("=" * 70)
log("EXTRACTING GOLD DATA FROM MT5")
log("=" * 70)

# Initialize MT5
if not mt5.initialize():
    log(f"MT5 initialization failed: {mt5.last_error()}")
    quit()

log(f"MT5 connected: {mt5.terminal_info().name}")

# Try different gold symbol names
gold_symbols = ['XAUUSD', 'XAUUSDm', 'GOLD', 'GOLDm', 'XAUUSD.', 'XAUUSD#']

gold_symbol = None
for sym in gold_symbols:
    info = mt5.symbol_info(sym)
    if info is not None:
        gold_symbol = sym
        log(f"Found gold symbol: {sym}")
        break

if gold_symbol is None:
    log("Gold symbol not found. Available symbols containing 'XAU' or 'GOLD':")
    symbols = mt5.symbols_get()
    for s in symbols:
        if 'XAU' in s.name.upper() or 'GOLD' in s.name.upper():
            log(f"  {s.name}")
    mt5.shutdown()
    quit()

# Enable symbol
if not mt5.symbol_select(gold_symbol, True):
    log(f"Failed to select {gold_symbol}")
    mt5.shutdown()
    quit()

# Get M5 data - last 2 years
log(f"\nExtracting M5 data for {gold_symbol}...")
end_date = datetime.now()
start_date = end_date - timedelta(days=730)  # ~2 years

rates = mt5.copy_rates_range(gold_symbol, mt5.TIMEFRAME_M5, start_date, end_date)

if rates is None or len(rates) == 0:
    log(f"No data received for {gold_symbol}")
    mt5.shutdown()
    quit()

log(f"Received {len(rates):,} M5 bars")

# Convert to DataFrame
df = pd.DataFrame(rates)
df['datetime'] = pd.to_datetime(df['time'], unit='s')
df = df.set_index('datetime')
df = df[['open', 'high', 'low', 'close', 'tick_volume']]
df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

log(f"Date range: {df.index[0]} to {df.index[-1]}")
log(f"Price range: ${df['Close'].min():.2f} to ${df['Close'].max():.2f}")

# Save to CSV
output_path = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/data/XAUUSD_M5.csv')
df.to_csv(output_path)
log(f"\nSaved to: {output_path}")

# Show sample
log(f"\nSample data:")
log(df.tail(10).to_string())

mt5.shutdown()
log(f"\nCompleted: {datetime.now()}")
