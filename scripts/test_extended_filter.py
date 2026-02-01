"""
Quick test: Compare velocity only vs velocity + extended filter
"""
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/data')
LSTM_DATA_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/lstm_model/data')

# Load the latest trades
trades_df = pd.read_csv(LSTM_DATA_DIR / 'trades_2025_oos.csv')

print("=" * 60)
print("EXTENDED FILTER ANALYSIS")
print("=" * 60)

print(f"\nCurrent results (Velocity + Extended + Friday filter):")
print(f"  Trades: {len(trades_df)}")
print(f"  WR: {trades_df['win'].mean()*100:.1f}%")
print(f"  Net Avg: {trades_df['net_pips'].mean():.2f}")
print(f"  Total: {trades_df['net_pips'].sum():.0f}")

# The extended filter removes trades where:
# BUY: base_h1 >= 0.7 AND quote_h1 < 0.7 (base extended but quote not)
# SELL: quote_h1 >= 0.7 AND base_h1 < 0.7 (quote extended but base not)

# To test impact, we need to run backtest without extended filter
# For now, let's check how many trades would have been filtered

print("\n" + "=" * 60)
print("To properly test, need to run backtest without extended filter")
print("=" * 60)
