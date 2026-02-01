"""
Backtest on Truly Unseen Data (Post-Training Period)
====================================================
Test existing models on data they've never seen:
- Models trained on data up to ~July 2025
- Test on July 2025 - Dec 2025 (5+ months unseen)
"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import pickle
import joblib
from pathlib import Path
from datetime import datetime

def log(msg=""):
    print(msg, flush=True)

log("=" * 70)
log("BACKTEST ON UNSEEN DATA (POST-TRAINING)")
log("=" * 70)

DATA_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/data')
CLEANED_DIR = DATA_DIR / 'cleaned'
MODEL_DIR = Path('/mnt/c/Users/luizh/Documents/mt4/tryea/v1-model/lstm_model/models')

CURRENCIES = ['EUR', 'USD', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD', 'NZD']

MFC_EXTREME_THRESHOLD = 0.5
MAX_ADVERSE_MFC = 0.25
RETURN_TARGET = 0.0


def load_mfc_cleaned(currency, timeframe):
    """Load cleaned MFC data."""
    fp = CLEANED_DIR / f'mfc_currency_{currency}_{timeframe}_clean.csv'
    if not fp.exists():
        return None
    df = pd.read_csv(fp)
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df = df.set_index('datetime').sort_index()
    return df['MFC']


def load_price_data(pair):
    """Load price data."""
    fp = DATA_DIR / f'{pair}_GMT+0_US-DST_M1.csv'
    if not fp.exists():
        return None
    df = pd.read_csv(fp)
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df = df.set_index('datetime').sort_index()
    return df['Close']


def backtest_model(name, model_file, features_file, base_tf, start_date, max_bars):
    """Backtest a model on unseen data."""
    
    log(f"\n{'='*70}")
    log(f"BACKTEST: {name} MODEL")
    log(f"Unseen data from: {start_date}")
    log(f"{'='*70}")
    
    # Load model
    model = joblib.load(MODEL_DIR / model_file)
    with open(MODEL_DIR / features_file, 'rb') as f:
        feature_cols = pickle.load(f)
    
    pairs = [
        ('EUR', 'USD'), ('EUR', 'GBP'), ('EUR', 'JPY'), ('EUR', 'CHF'),
        ('EUR', 'CAD'), ('EUR', 'AUD'), ('EUR', 'NZD'),
        ('GBP', 'USD'), ('GBP', 'JPY'), ('GBP', 'CHF'), ('GBP', 'CAD'),
        ('GBP', 'AUD'), ('GBP', 'NZD'),
        ('USD', 'JPY'), ('USD', 'CHF'), ('USD', 'CAD'),
        ('AUD', 'USD'), ('AUD', 'JPY'), ('AUD', 'CHF'), ('AUD', 'CAD'), ('AUD', 'NZD'),
        ('NZD', 'USD'), ('NZD', 'JPY'), ('NZD', 'CHF'), ('NZD', 'CAD'),
        ('CAD', 'JPY'), ('CAD', 'CHF'), ('CHF', 'JPY'),
    ]
    
    all_entries = []
    
    for base_ccy, quote_ccy in pairs:
        pair = f"{base_ccy}{quote_ccy}"
        
        # Load MFC data for all timeframes
        base_mfc = {}
        quote_mfc = {}
        
        for tf in ['M5', 'M15', 'M30', 'H1', 'H4']:
            base_mfc[tf] = load_mfc_cleaned(base_ccy, tf)
            quote_mfc[tf] = load_mfc_cleaned(quote_ccy, tf)
        
        if base_mfc[base_tf] is None:
            continue
            
        # Load price
        price = load_price_data(pair)
        if price is None:
            continue
        
        # Resample price to base timeframe
        if base_tf == 'M15':
            price_resampled = price.resample('15min').last().dropna()
        elif base_tf == 'M30':
            price_resampled = price.resample('30min').last().dropna()
        else:  # H1
            price_resampled = price.resample('1h').last().dropna()
        
        # Create dataframe
        df = pd.DataFrame(index=base_mfc[base_tf].index)
        df[f'base_{base_tf.lower()}'] = base_mfc[base_tf]
        df[f'quote_{base_tf.lower()}'] = quote_mfc[base_tf]
        df['price'] = price_resampled
        
        # Add other timeframes
        for tf in ['M5', 'M15', 'M30', 'H1', 'H4']:
            tf_lower = tf.lower()
            if base_mfc[tf] is not None:
                df[f'base_{tf_lower}'] = base_mfc[tf].reindex(df.index, method='ffill')
            if quote_mfc[tf] is not None:
                df[f'quote_{tf_lower}'] = quote_mfc[tf].reindex(df.index, method='ffill')
        
        df = df.dropna()
        
        # Filter to unseen period only
        df = df[df.index >= start_date]
        
        if len(df) < max_bars + 10:
            continue
        
        # Apply shift
        for tf in ['m5', 'm15', 'm30', 'h1', 'h4']:
            df[f'base_{tf}_shifted'] = df[f'base_{tf}'].shift(1)
            df[f'quote_{tf}_shifted'] = df[f'quote_{tf}'].shift(1)
        
        # Velocities
        for tf in ['m5', 'm15', 'm30', 'h1', 'h4']:
            df[f'base_vel_{tf}'] = df[f'base_{tf}_shifted'].diff()
            df[f'quote_vel_{tf}'] = df[f'quote_{tf}_shifted'].diff()
        
        # Additional features based on base timeframe
        base_tf_lower = base_tf.lower()
        df[f'base_vel2_{base_tf_lower}'] = df[f'base_{base_tf_lower}_shifted'].diff(2)
        df[f'base_acc_{base_tf_lower}'] = df[f'base_vel_{base_tf_lower}'].diff()
        df['divergence'] = df[f'base_{base_tf_lower}_shifted'] - df[f'quote_{base_tf_lower}_shifted']
        df['vel_divergence'] = df[f'base_vel_{base_tf_lower}'] - df[f'quote_vel_{base_tf_lower}']
        
        df = df.dropna()
        
        # Find entries and label them
        for direction in ['buy', 'sell']:
            if direction == 'buy':
                mask = df[f'base_{base_tf_lower}_shifted'] <= -MFC_EXTREME_THRESHOLD
            else:
                mask = df[f'base_{base_tf_lower}_shifted'] >= MFC_EXTREME_THRESHOLD
            
            for idx in df[mask].index:
                pos = df.index.get_loc(idx)
                if pos + max_bars >= len(df):
                    continue
                
                entry_mfc = df.iloc[pos][f'base_{base_tf_lower}_shifted']
                entry_price = df.iloc[pos]['price']
                
                future_mfc = df.iloc[pos+1:pos+max_bars+1][f'base_{base_tf_lower}']
                future_price = df.iloc[pos+1:pos+max_bars+1]['price']
                
                if direction == 'buy':
                    min_mfc = future_mfc.min()
                    adverse_move = entry_mfc - min_mfc
                    returned = (future_mfc >= RETURN_TARGET).any()
                    is_quality = returned and (adverse_move < MAX_ADVERSE_MFC)
                    max_dd_pips = (entry_price - future_price.min()) * 10000
                    max_profit_pips = (future_price.max() - entry_price) * 10000
                else:
                    max_mfc = future_mfc.max()
                    adverse_move = max_mfc - entry_mfc
                    returned = (future_mfc <= RETURN_TARGET).any()
                    is_quality = returned and (adverse_move < MAX_ADVERSE_MFC)
                    max_dd_pips = (future_price.max() - entry_price) * 10000
                    max_profit_pips = (entry_price - future_price.min()) * 10000
                
                # Fix JPY pairs
                if 'JPY' in pair:
                    max_dd_pips /= 100
                    max_profit_pips /= 100
                
                row = df.iloc[pos]
                
                features = {
                    'datetime': idx,
                    'pair': pair,
                    'direction': direction,
                    'is_quality': int(is_quality),
                    'max_dd_pips': max_dd_pips,
                    'max_profit_pips': max_profit_pips,
                }
                
                for tf in ['m5', 'm15', 'm30', 'h1', 'h4']:
                    features[f'base_{tf}'] = row[f'base_{tf}_shifted']
                    features[f'quote_{tf}'] = row[f'quote_{tf}_shifted']
                    features[f'base_vel_{tf}'] = row[f'base_vel_{tf}']
                    features[f'quote_vel_{tf}'] = row[f'quote_vel_{tf}']
                
                features[f'base_vel2_{base_tf_lower}'] = row[f'base_vel2_{base_tf_lower}']
                features[f'base_acc_{base_tf_lower}'] = row[f'base_acc_{base_tf_lower}']
                features['divergence'] = row['divergence']
                features['vel_divergence'] = row['vel_divergence']
                features['direction_code'] = 1 if direction == 'buy' else 0
                
                all_entries.append(features)
    
    if len(all_entries) == 0:
        log("No entries found in unseen period")
        return None
    
    result_df = pd.DataFrame(all_entries)
    
    log(f"\nUnseen data entries: {len(result_df):,}")
    log(f"Date range: {result_df['datetime'].min().date()} to {result_df['datetime'].max().date()}")
    log(f"Base quality rate: {result_df['is_quality'].mean()*100:.1f}%")
    
    # Predict
    X = result_df[feature_cols].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    result_df['pred_prob'] = model.predict_proba(X)[:, 1]
    
    # Results by threshold
    log(f"\n| Threshold | Trades | Quality% | Win% | Avg PnL | PF |")
    log(f"|-----------|--------|----------|------|---------|------|")
    
    results = []
    for thresh in [0.5, 0.6, 0.7, 0.8, 0.9]:
        filtered = result_df[result_df['pred_prob'] >= thresh].copy()
        if len(filtered) == 0:
            continue
        
        quality_rate = filtered['is_quality'].mean() * 100
        
        filtered['trade_pnl'] = np.where(
            filtered['is_quality'] == 1,
            filtered['max_profit_pips'] * 0.7,
            -filtered['max_dd_pips'] * 0.5
        )
        
        win_rate = (filtered['trade_pnl'] > 0).mean() * 100
        avg_pnl = filtered['trade_pnl'].mean()
        
        winners = filtered[filtered['trade_pnl'] > 0]['trade_pnl'].sum()
        losers = abs(filtered[filtered['trade_pnl'] <= 0]['trade_pnl'].sum())
        pf = winners / losers if losers > 0 else float('inf')
        
        log(f"| {thresh:^9} | {len(filtered):>6,} | {quality_rate:>7.1f}% | {win_rate:>4.1f}% | {avg_pnl:>7.1f} | {pf:>4.2f} |")
        
        results.append({
            'threshold': thresh,
            'trades': len(filtered),
            'quality': quality_rate,
            'win_rate': win_rate,
            'pf': pf
        })
    
    return results


# Run backtests
# H1 model - trained on data up to ~July 2025
h1_results = backtest_model(
    'H1', 
    'quality_xgb_proper.joblib',
    'quality_xgb_features_proper.pkl',
    'H1',
    '2025-07-15',  # Start after training data ends
    400  # H1 bars
)

# M30 model - trained on data up to ~April 2025  
m30_results = backtest_model(
    'M30',
    'quality_xgb_m30_proper.joblib', 
    'quality_xgb_features_m30_proper.pkl',
    'M30',
    '2025-04-20',
    800  # M30 bars
)

# M15 model - trained on data up to ~April 2025
m15_results = backtest_model(
    'M15',
    'quality_xgb_m15_proper.joblib',
    'quality_xgb_features_m15_proper.pkl', 
    'M15',
    '2025-04-20',
    800  # M15 bars
)

log("\n" + "=" * 70)
log("SUMMARY: PERFORMANCE ON TRULY UNSEEN DATA")
log("=" * 70)
log("\nThis tests models on data they were NEVER trained or validated on.")
log("If performance holds, the models have real predictive power.\n")
