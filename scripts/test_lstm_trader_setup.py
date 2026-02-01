"""
Test script to verify LSTM trader setup
=======================================
Run this before the main trader to check:
1. MT5 connection
2. MFC file reading
3. LSTM model loading
4. RSI calculation
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
from pathlib import Path

print("=" * 60)
print("LSTM TRADER SETUP TEST")
print("=" * 60)

# Test 1: MT5 Connection
print("\n1. Testing MT5 connection...")
try:
    import MetaTrader5 as mt5

    if mt5.initialize():
        account = mt5.account_info()
        print(f"   OK - Connected to {account.server}")
        print(f"   Account: {account.login}")
        print(f"   Balance: {account.balance}")

        # Test symbol access
        symbol = "EURUSDm"
        info = mt5.symbol_info(symbol)
        if info:
            print(f"   Symbol {symbol}: bid={info.bid:.5f}, ask={info.ask:.5f}")
        else:
            mt5.symbol_select(symbol, True)
            info = mt5.symbol_info(symbol)
            if info:
                print(f"   Symbol {symbol}: bid={info.bid:.5f}, ask={info.ask:.5f}")
            else:
                print(f"   WARNING: Symbol {symbol} not available")

        mt5.shutdown()
    else:
        print(f"   FAILED: {mt5.last_error()}")
except ImportError:
    print("   FAILED: MetaTrader5 package not installed")
    print("   Run: pip install MetaTrader5")
except Exception as e:
    print(f"   FAILED: {e}")

# Test 2: MFC File Reading
print("\n2. Testing MFC file reading...")
try:
    MFC_FILE = Path(os.environ.get('APPDATA', '')) / "MetaQuotes/Terminal/Common/Files/DWX_MFC_Auto.txt"

    if MFC_FILE.exists():
        import json
        with open(MFC_FILE, 'r') as f:
            data = json.load(f)

        print(f"   OK - File found: {MFC_FILE}")
        print(f"   Timeframes in file: {list(data.keys())}")

        # Check data structure
        first_key = list(data.keys())[0]
        currencies = list(data[first_key].keys())
        print(f"   Currencies: {currencies}")

        # Check bar counts
        for tf_key in data.keys():
            if 'M5' in tf_key:
                for ccy in ['USD', 'EUR', 'JPY']:
                    if ccy in data[tf_key]:
                        bars = len(data[tf_key][ccy])
                        print(f"   {tf_key} {ccy}: {bars} bars")
                break
    else:
        print(f"   FAILED: File not found at {MFC_FILE}")
        print("   Make sure MT4 exporter is running and wrote the file")
except Exception as e:
    print(f"   FAILED: {e}")

# Test 3: LSTM Models
print("\n3. Testing LSTM model loading...")
try:
    import tensorflow as tf

    MODEL_DIR = Path(__file__).parent.parent / "models"
    print(f"   Model directory: {MODEL_DIR}")

    test_currencies = ['USD', 'JPY', 'AUD']
    for ccy in test_currencies:
        model_path = MODEL_DIR / f'lstm_{ccy}_final.keras'
        if model_path.exists():
            model = tf.keras.models.load_model(model_path)
            print(f"   OK - {ccy} model loaded")
            del model
        else:
            print(f"   FAILED: {ccy} model not found at {model_path}")

    tf.keras.backend.clear_session()
except ImportError:
    print("   FAILED: TensorFlow not installed")
    print("   Run: pip install tensorflow")
except Exception as e:
    print(f"   FAILED: {e}")

# Test 4: RSI Calculation
print("\n4. Testing RSI calculation...")
try:
    import MetaTrader5 as mt5

    if mt5.initialize():
        symbol = "USDJPYm"
        mt5.symbol_select(symbol, True)

        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 20)

        if rates is not None and len(rates) >= 15:
            closes = [r['close'] for r in rates]

            # Calculate RSI
            period = 14
            shift = 1
            avg_gain = 0
            avg_loss = 0

            for i in range(shift, shift + period):
                idx = len(closes) - 1 - i
                prev_idx = idx - 1
                delta = closes[idx] - closes[prev_idx]
                if delta > 0:
                    avg_gain += delta
                else:
                    avg_loss += (-delta)

            avg_gain /= period
            avg_loss /= period

            if avg_loss == 0:
                rsi = 100
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))

            print(f"   OK - {symbol} RSI(14): {rsi:.2f}")
        else:
            print(f"   FAILED: Could not get price data for {symbol}")

        mt5.shutdown()
    else:
        print(f"   FAILED: MT5 not connected")
except Exception as e:
    print(f"   FAILED: {e}")

# Test 5: Full prediction test
print("\n5. Testing full prediction pipeline...")
try:
    import json
    import numpy as np
    import tensorflow as tf
    import pickle

    # Load config
    DATA_DIR = Path(__file__).parent.parent / "data"
    config_path = DATA_DIR / 'config.pkl'

    if config_path.exists():
        with open(config_path, 'rb') as f:
            config = pickle.load(f)
        lookback = config.get('lookback', {'M5': 48, 'M15': 32, 'M30': 24, 'H1': 12, 'H4': 6})
        print(f"   Config loaded: lookback={lookback}")
    else:
        lookback = {'M5': 48, 'M15': 32, 'M30': 24, 'H1': 12, 'H4': 6}
        print(f"   Using default lookback: {lookback}")

    # Load MFC data
    MFC_FILE = Path(os.environ.get('APPDATA', '')) / "MetaQuotes/Terminal/Common/Files/DWX_MFC_Auto.txt"

    with open(MFC_FILE, 'r') as f:
        mfc_raw = json.load(f)

    # Parse for USD
    ccy = 'USD'
    host = 'EURUSDm'
    mfc_data = {}

    for tf in ['M5', 'M15', 'M30', 'H1', 'H4']:
        key = f"{host}_{tf}"
        if key in mfc_raw and ccy in mfc_raw[key]:
            sorted_items = sorted(mfc_raw[key][ccy].items(), key=lambda x: x[0])
            mfc_data[tf] = [v for k, v in sorted_items]
            print(f"   {tf}: {len(mfc_data[tf])} bars available, need {lookback[tf]}")

    # Check we have enough data
    enough_data = True
    for tf in ['M5', 'M15', 'M30', 'H1', 'H4']:
        if len(mfc_data.get(tf, [])) < lookback[tf]:
            print(f"   WARNING: Not enough {tf} data ({len(mfc_data.get(tf, []))} < {lookback[tf]})")
            enough_data = False

    if enough_data:
        # Load model and predict
        MODEL_DIR = Path(__file__).parent.parent / "models"
        model = tf.keras.models.load_model(MODEL_DIR / f'lstm_{ccy}_final.keras')

        # Prepare inputs
        X_M5 = np.array(mfc_data['M5'][-lookback['M5']:]).reshape(1, lookback['M5'], 1)
        X_M15 = np.array(mfc_data['M15'][-lookback['M15']:]).reshape(1, lookback['M15'], 1)
        X_M30 = np.array(mfc_data['M30'][-lookback['M30']:]).reshape(1, lookback['M30'], 1)
        X_H1 = np.array(mfc_data['H1'][-lookback['H1']:]).reshape(1, lookback['H1'], 1)
        X_H4 = np.array(mfc_data['H4'][-lookback['H4']:]).reshape(1, lookback['H4'], 1)

        vel_m5 = mfc_data['M5'][-1] - mfc_data['M5'][-2]
        vel_m30 = mfc_data['M30'][-1] - mfc_data['M30'][-2]

        X_aux = np.array([[
            vel_m5,
            vel_m30,
            mfc_data['M5'][-1],
            mfc_data['M30'][-1],
            mfc_data['H4'][-1],
        ]])

        pred = model.predict([X_M5, X_M15, X_M30, X_H1, X_H4, X_aux], verbose=0)

        direction = np.argmax(pred[0])
        confidence = np.max(pred[0])

        dir_names = {0: 'DOWN', 1: 'NEUTRAL', 2: 'UP'}
        print(f"   OK - {ccy} prediction: {dir_names[direction]} (conf: {confidence:.2%})")

        tf.keras.backend.clear_session()
    else:
        print("   SKIPPED: Not enough MFC data - increase mfcBarsToExport in MT4")

except Exception as e:
    print(f"   FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)
print("\nIf all tests passed, you can run:")
print("  python lstm_trader_mt5.py")
