"""
Quality Entry Trader V2 - Both Currency Triggers (H1)
=====================================================
Triggers on BOTH base AND quote currency extremes.

Entry triggers:
- BUY when base_mfc <= -0.5 (base weak, will recover up)
- SELL when base_mfc >= +0.5 (base strong, will pull back)
- SELL when quote_mfc <= -0.5 (quote weak, will recover -> pair DOWN)
- BUY when quote_mfc >= +0.5 (quote strong, will pull back -> pair UP)
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import MetaTrader5 as mt5
import numpy as np
import json
import time
import pickle
import joblib
from datetime import datetime
from pathlib import Path
import logging

# ============================================================================
# CONFIGURATION
# ============================================================================

MT5_LOGIN = None
MT5_PASSWORD = None
MT5_SERVER = None
SYMBOL_SUFFIX = "m"

MFC_FILE_PATH = Path(os.environ.get('APPDATA', '')) / "MetaQuotes/Terminal/Common/Files/DWX/DWX_MFC_Auto.txt"
MODEL_DIR = Path(__file__).parent.parent / "models"

# Strategy Parameters
MFC_EXTREME_THRESHOLD = 0.5
QUALITY_THRESHOLD = 0.80
MAX_BARS_HOLD = 100
FRIDAY_CUTOFF_HOUR = 12

# Trading Parameters
LOT_SIZE = 0.01
MAGIC_NUMBER = 150022  # V2 H1 magic number
MAX_SPREAD_PIPS = 5
MAX_POSITIONS = 10

ALL_PAIRS = [
    ('EURUSD', 'EUR', 'USD'), ('GBPUSD', 'GBP', 'USD'), ('AUDUSD', 'AUD', 'USD'),
    ('NZDUSD', 'NZD', 'USD'), ('USDJPY', 'USD', 'JPY'), ('USDCHF', 'USD', 'CHF'),
    ('USDCAD', 'USD', 'CAD'), ('EURGBP', 'EUR', 'GBP'), ('EURJPY', 'EUR', 'JPY'),
    ('EURCHF', 'EUR', 'CHF'), ('EURCAD', 'EUR', 'CAD'), ('EURAUD', 'EUR', 'AUD'),
    ('EURNZD', 'EUR', 'NZD'), ('GBPJPY', 'GBP', 'JPY'), ('GBPCHF', 'GBP', 'CHF'),
    ('GBPCAD', 'GBP', 'CAD'), ('GBPAUD', 'GBP', 'AUD'), ('GBPNZD', 'GBP', 'NZD'),
    ('AUDJPY', 'AUD', 'JPY'), ('AUDCHF', 'AUD', 'CHF'), ('AUDCAD', 'AUD', 'CAD'),
    ('AUDNZD', 'AUD', 'NZD'), ('NZDJPY', 'NZD', 'JPY'), ('NZDCHF', 'NZD', 'CHF'),
    ('NZDCAD', 'NZD', 'CAD'), ('CADJPY', 'CAD', 'JPY'), ('CADCHF', 'CAD', 'CHF'),
    ('CHFJPY', 'CHF', 'JPY'),
]

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger(__name__)


# ============================================================================
# QUALITY MODEL MANAGER (V2)
# ============================================================================

class QualityModelManager:
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.model = None
        self.feature_cols = None

    def load(self) -> bool:
        model_path = self.model_dir / 'quality_xgb_h1_v2.joblib'
        features_path = self.model_dir / 'quality_xgb_features_h1_v2.pkl'

        if not model_path.exists() or not features_path.exists():
            log.error(f"V2 H1 model not found")
            return False

        try:
            self.model = joblib.load(model_path)
            with open(features_path, 'rb') as f:
                self.feature_cols = pickle.load(f)
            log.info(f"Loaded V2 H1 quality model: {len(self.feature_cols)} features")
            return True
        except Exception as e:
            log.error(f"Error loading model: {e}")
            return False

    def predict_quality(self, features: dict) -> float:
        if self.model is None:
            return 0.0
        try:
            X = np.array([[features.get(col, 0.0) for col in self.feature_cols]])
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            return float(self.model.predict_proba(X)[0, 1])
        except Exception as e:
            log.error(f"Prediction error: {e}")
            return 0.0


# ============================================================================
# MFC DATA READER
# ============================================================================

class MFCDataReader:
    def __init__(self, file_path: Path, host_symbol: str = "EURUSDm"):
        self.file_path = file_path
        self.host_symbol = host_symbol
        self.last_mtime = 0

    def wait_for_fresh_data(self, bar_time: datetime, timeout: float = 120) -> bool:
        start_time = time.time()
        bar_timestamp = bar_time.timestamp()

        while time.time() - start_time < timeout:
            if not self.file_path.exists():
                time.sleep(1)
                continue

            current_mtime = self.file_path.stat().st_mtime
            if current_mtime > bar_timestamp:
                self.last_mtime = current_mtime
                log.info(f"Fresh MFC data (mtime: {datetime.fromtimestamp(current_mtime).strftime('%H:%M:%S')} > bar: {bar_time.strftime('%H:%M:%S')})")
                return True
            time.sleep(1)

        log.warning(f"Timeout waiting for fresh MFC data")
        return False

    def read(self) -> dict:
        if not self.file_path.exists():
            return None

        try:
            with open(self.file_path, 'r') as f:
                raw_json = f.read()

            raw_json = raw_json.strip()
            if not raw_json.startswith('{') or not raw_json.endswith('}'):
                return None

            parsed = json.loads(raw_json)
            result = {}
            timeframes = ['M5', 'M15', 'M30', 'H1', 'H4']
            currencies = ['EUR', 'USD', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD', 'NZD']

            for ccy in currencies:
                result[ccy] = {}
                for tf in timeframes:
                    key = f"{self.host_symbol}_{tf}"
                    if key in parsed and ccy in parsed[key]:
                        tf_data = parsed[key][ccy]
                        sorted_items = sorted(tf_data.items(), key=lambda x: x[0])
                        result[ccy][tf] = [v for k, v in sorted_items]
                    else:
                        result[ccy][tf] = []

            return result
        except Exception as e:
            log.error(f"MFC read error: {e}")
            return None

    def get_mfc_features(self, base: str, quote: str, direction: str, trigger: str) -> dict:
        """Get features for V2 H1 model (includes trigger_code)."""
        data = self.read()
        if not data or base not in data or quote not in data:
            return None

        features = {}

        # MFC values and velocities (shifted = use [-2])
        for tf in ['m5', 'm15', 'm30', 'h1', 'h4']:
            tf_upper = tf.upper()
            base_values = data[base].get(tf_upper, [])
            quote_values = data[quote].get(tf_upper, [])

            if len(base_values) >= 3 and len(quote_values) >= 3:
                features[f'base_{tf}'] = base_values[-2]
                features[f'quote_{tf}'] = quote_values[-2]
                features[f'base_vel_{tf}'] = base_values[-2] - base_values[-3]
                features[f'quote_vel_{tf}'] = quote_values[-2] - quote_values[-3]
            else:
                features[f'base_{tf}'] = 0
                features[f'quote_{tf}'] = 0
                features[f'base_vel_{tf}'] = 0
                features[f'quote_vel_{tf}'] = 0

        # Momentum features (H1 - trading timeframe)
        base_h1 = data[base].get('H1', [])
        quote_h1 = data[quote].get('H1', [])

        if len(base_h1) >= 4:
            features['base_vel2_h1'] = base_h1[-2] - base_h1[-4]
            features['base_acc_h1'] = (base_h1[-2] - base_h1[-3]) - (base_h1[-3] - base_h1[-4])
        else:
            features['base_vel2_h1'] = 0
            features['base_acc_h1'] = 0

        if len(quote_h1) >= 4:
            features['quote_vel2_h1'] = quote_h1[-2] - quote_h1[-4]
            features['quote_acc_h1'] = (quote_h1[-2] - quote_h1[-3]) - (quote_h1[-3] - quote_h1[-4])
        else:
            features['quote_vel2_h1'] = 0
            features['quote_acc_h1'] = 0

        # Divergence
        features['divergence'] = features['base_h1'] - features['quote_h1']
        features['vel_divergence'] = features['base_vel_h1'] - features['quote_vel_h1']

        # Codes
        features['direction_code'] = 1 if direction == 'buy' else 0
        features['trigger_code'] = 1 if trigger == 'base' else 0

        return features


# ============================================================================
# MT5 TRADER
# ============================================================================

class MT5Trader:
    def __init__(self):
        self.connected = False

    def connect(self) -> bool:
        if not mt5.initialize():
            log.error(f"MT5 initialize() failed")
            return False

        if MT5_LOGIN and MT5_PASSWORD and MT5_SERVER:
            if not mt5.login(MT5_LOGIN, MT5_PASSWORD, MT5_SERVER):
                log.error(f"MT5 login failed")
                return False

        account_info = mt5.account_info()
        if account_info:
            log.info(f"Connected: {account_info.server}, Balance: {account_info.balance}")

        self.connected = True
        return True

    def disconnect(self):
        mt5.shutdown()
        self.connected = False

    def get_symbol(self, pair: str) -> str:
        return pair + SYMBOL_SUFFIX

    def symbol_info(self, symbol: str):
        info = mt5.symbol_info(symbol)
        if info is None:
            mt5.symbol_select(symbol, True)
            info = mt5.symbol_info(symbol)
        return info

    def get_spread_pips(self, symbol: str) -> float:
        info = self.symbol_info(symbol)
        if info is None:
            return 999
        spread_points = info.spread
        point = info.point
        # 1 pip = 0.01 for JPY, 0.0001 for others
        pip_size = 0.01 if 'JPY' in symbol else 0.0001
        spread_price = spread_points * point
        return spread_price / pip_size

    def get_positions(self) -> list:
        positions = mt5.positions_get()
        if positions is None:
            return []
        return [p for p in positions if p.magic == MAGIC_NUMBER]

    def has_position(self, symbol: str) -> bool:
        return any(p.symbol == symbol for p in self.get_positions())

    def get_position_type(self, symbol: str) -> int:
        for p in self.get_positions():
            if p.symbol == symbol:
                return p.type
        return -1

    def get_position_info(self, symbol: str) -> tuple:
        """Returns (position_type, trigger_type, trigger_ccy) from position comment.

        Comment format: V2_BASE_EUR_BUY or V2_QUOTE_JPY_SELL
        """
        for p in self.get_positions():
            if p.symbol == symbol:
                try:
                    # Parse comment: V2_BASE_EUR_BUY or V2_QUOTE_JPY_SELL
                    parts = p.comment.split('_')
                    if len(parts) >= 3 and parts[0] == 'V2':
                        trigger_type = parts[1].lower()  # 'base' or 'quote'
                        trigger_ccy = parts[2]           # 'EUR', 'JPY', etc.
                        return p.type, trigger_type, trigger_ccy
                except:
                    pass
                # Fallback: assume base trigger if can't parse
                return p.type, 'base', None
        return -1, None, None

    def get_hours_since_entry(self, symbol: str) -> float:
        for p in self.get_positions():
            if p.symbol == symbol:
                return (datetime.now() - datetime.fromtimestamp(p.time)).total_seconds() / 3600
        return 0

    def count_positions(self) -> int:
        return len(self.get_positions())

    def open_trade(self, symbol: str, direction: str, lots: float, comment: str = "") -> bool:
        info = self.symbol_info(symbol)
        if info is None:
            return False

        if direction == 'buy':
            price = info.ask
            order_type = mt5.ORDER_TYPE_BUY
        else:
            price = info.bid
            order_type = mt5.ORDER_TYPE_SELL

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lots,
            "type": order_type,
            "price": price,
            "deviation": 10,
            "magic": MAGIC_NUMBER,
            "comment": comment,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            log.error(f"{direction.upper()} failed: {result.retcode}")
            return False

        log.info(f"{direction.upper()} {symbol} @ {price:.5f}")
        return True

    def close_position(self, symbol: str, reason: str = "") -> bool:
        positions = mt5.positions_get(symbol=symbol)
        if positions is None:
            return False

        for pos in positions:
            if pos.magic != MAGIC_NUMBER:
                continue

            if pos.type == mt5.ORDER_TYPE_BUY:
                price = mt5.symbol_info_tick(symbol).bid
                close_type = mt5.ORDER_TYPE_SELL
            else:
                price = mt5.symbol_info_tick(symbol).ask
                close_type = mt5.ORDER_TYPE_BUY

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": pos.volume,
                "type": close_type,
                "position": pos.ticket,
                "price": price,
                "deviation": 10,
                "magic": MAGIC_NUMBER,
                "comment": f"Close: {reason}",
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                log.error(f"Close failed: {result.retcode}")
                return False

            log.info(f"CLOSED {symbol} - {reason} - P/L: {pos.profit:.2f}")
            return True

        return False


# ============================================================================
# MAIN STRATEGY
# ============================================================================

class QualityEntryStrategyV2:
    def __init__(self):
        self.trader = MT5Trader()
        self.mfc_reader = MFCDataReader(MFC_FILE_PATH)
        self.model_manager = QualityModelManager(MODEL_DIR)
        self.last_bar_time = None
        self.running = False

    def initialize(self) -> bool:
        log.info("=" * 60)
        log.info("QUALITY ENTRY TRADER V2 - BOTH CURRENCIES (H1)")
        log.info("=" * 60)

        if not self.trader.connect():
            return False

        if not self.model_manager.load():
            return False

        if not MFC_FILE_PATH.exists():
            log.error(f"MFC file not found: {MFC_FILE_PATH}")
            return False

        log.info(f"Quality threshold: {QUALITY_THRESHOLD}")
        log.info(f"MFC extreme: +/-{MFC_EXTREME_THRESHOLD}")
        log.info(f"Triggers: BASE + QUOTE currencies")
        log.info(f"Magic number: {MAGIC_NUMBER}")
        log.info("=" * 60)

        return True

    def check_entry(self, pair: str, base: str, quote: str) -> tuple:
        """Returns (direction, trigger_type, trigger_ccy) or (None, None, None)."""
        symbol = self.trader.get_symbol(pair)

        # Friday cutoff
        tick = mt5.symbol_info_tick(symbol)
        if tick:
            server_time = datetime.fromtimestamp(tick.time)
            if server_time.weekday() == 4 and server_time.hour >= FRIDAY_CUTOFF_HOUR:
                return None, None, None

        # Spread check
        if self.trader.get_spread_pips(symbol) > MAX_SPREAD_PIPS:
            return None, None, None

        # Position limits
        if self.trader.count_positions() >= MAX_POSITIONS:
            return None, None, None

        if self.trader.has_position(symbol):
            return None, None, None

        # Get MFC data
        data = self.mfc_reader.read()
        if not data or base not in data or quote not in data:
            return None, None, None

        base_h1 = data[base].get('H1', [])
        quote_h1 = data[quote].get('H1', [])

        if len(base_h1) < 2 or len(quote_h1) < 2:
            return None, None, None

        base_mfc = base_h1[-2]  # Shifted
        quote_mfc = quote_h1[-2]

        # === BASE CURRENCY TRIGGERS ===
        # BUY when base at extreme LOW
        if base_mfc <= -MFC_EXTREME_THRESHOLD:
            features = self.mfc_reader.get_mfc_features(base, quote, 'buy', 'base')
            if features:
                quality = self.model_manager.predict_quality(features)
                if quality >= QUALITY_THRESHOLD:
                    log.info(f"{pair} BASE BUY: {base}_mfc={base_mfc:.3f}, quality={quality:.1%}")
                    return 'buy', 'base', base

        # SELL when base at extreme HIGH
        if base_mfc >= MFC_EXTREME_THRESHOLD:
            features = self.mfc_reader.get_mfc_features(base, quote, 'sell', 'base')
            if features:
                quality = self.model_manager.predict_quality(features)
                if quality >= QUALITY_THRESHOLD:
                    log.info(f"{pair} BASE SELL: {base}_mfc={base_mfc:.3f}, quality={quality:.1%}")
                    return 'sell', 'base', base

        # === QUOTE CURRENCY TRIGGERS ===
        # SELL when quote at extreme LOW (quote will recover -> pair DOWN)
        if quote_mfc <= -MFC_EXTREME_THRESHOLD:
            features = self.mfc_reader.get_mfc_features(base, quote, 'sell', 'quote')
            if features:
                quality = self.model_manager.predict_quality(features)
                if quality >= QUALITY_THRESHOLD:
                    log.info(f"{pair} QUOTE SELL: {quote}_mfc={quote_mfc:.3f}, quality={quality:.1%}")
                    return 'sell', 'quote', quote

        # BUY when quote at extreme HIGH (quote will pull back -> pair UP)
        if quote_mfc >= MFC_EXTREME_THRESHOLD:
            features = self.mfc_reader.get_mfc_features(base, quote, 'buy', 'quote')
            if features:
                quality = self.model_manager.predict_quality(features)
                if quality >= QUALITY_THRESHOLD:
                    log.info(f"{pair} QUOTE BUY: {quote}_mfc={quote_mfc:.3f}, quality={quality:.1%}")
                    return 'buy', 'quote', quote

        return None, None, None

    def check_exit(self, pair: str, base: str, quote: str) -> bool:
        symbol = self.trader.get_symbol(pair)

        if not self.trader.has_position(symbol):
            return False

        # Get position info including trigger type from comment
        pos_type, trigger_type, trigger_ccy = self.trader.get_position_info(symbol)
        if pos_type == -1:
            return False

        # Check MFC return for the TRIGGER currency only
        data = self.mfc_reader.read()
        if data:
            # Determine which currency to check based on trigger
            if trigger_type == 'base':
                mfc_values = data[base].get('H1', [])
                check_ccy = base
            else:  # quote trigger
                mfc_values = data[quote].get('H1', [])
                check_ccy = quote

            if len(mfc_values) >= 2:
                trigger_mfc = mfc_values[-2]

                # Exit when trigger currency returns to center
                if trigger_type == 'base':
                    if pos_type == mt5.ORDER_TYPE_BUY and trigger_mfc >= 0:
                        self.trader.close_position(symbol, f"{check_ccy} MFC returned: {trigger_mfc:.2f}")
                        return True
                    elif pos_type == mt5.ORDER_TYPE_SELL and trigger_mfc <= 0:
                        self.trader.close_position(symbol, f"{check_ccy} MFC returned: {trigger_mfc:.2f}")
                        return True
                else:  # quote trigger
                    if pos_type == mt5.ORDER_TYPE_SELL and trigger_mfc >= 0:
                        self.trader.close_position(symbol, f"{check_ccy} MFC returned: {trigger_mfc:.2f}")
                        return True
                    elif pos_type == mt5.ORDER_TYPE_BUY and trigger_mfc <= 0:
                        self.trader.close_position(symbol, f"{check_ccy} MFC returned: {trigger_mfc:.2f}")
                        return True

        # Timeout
        hours_held = self.trader.get_hours_since_entry(symbol)
        max_hours = MAX_BARS_HOLD * 1.0  # 1 hour per H1 bar
        if hours_held >= max_hours:
            self.trader.close_position(symbol, f"Timeout ({hours_held:.1f}h)")
            return True

        return False

    def process_bar(self):
        log.info(f"[{self.last_bar_time.strftime('%Y-%m-%d %H:%M')}] Scanning {len(ALL_PAIRS)} pairs...")

        signals = 0
        for pair, base, quote in ALL_PAIRS:
            symbol = self.trader.get_symbol(pair)

            if self.trader.has_position(symbol):
                self.check_exit(pair, base, quote)
            else:
                direction, trigger, trigger_ccy = self.check_entry(pair, base, quote)
                if direction:
                    signals += 1
                    comment = f"V2_{trigger.upper()}_{trigger_ccy}_{direction.upper()}"
                    self.trader.open_trade(symbol, direction, LOT_SIZE, comment)

        if signals == 0:
            log.info("No signals this bar")

    def wait_for_new_bar(self) -> bool:
        ref_symbol = self.trader.get_symbol(ALL_PAIRS[0][0])
        rates = mt5.copy_rates_from_pos(ref_symbol, mt5.TIMEFRAME_H1, 0, 1)
        if rates is None:
            return False

        current_bar_time = datetime.fromtimestamp(rates[0]['time'])

        if self.last_bar_time is None:
            self.last_bar_time = current_bar_time
            return True

        if current_bar_time > self.last_bar_time:
            self.last_bar_time = current_bar_time
            return True

        return False

    def run(self):
        self.running = True
        log.info("Strategy started. Waiting for H1 bar signals...")

        while self.running:
            try:
                if self.wait_for_new_bar():
                    log.info(f"New H1 bar ({self.last_bar_time.strftime('%H:%M')}), waiting for fresh MFC...")
                    if self.mfc_reader.wait_for_fresh_data(self.last_bar_time, timeout=120):
                        self.process_bar()
                    else:
                        log.warning("MFC timeout - skipping bar")

                    positions = self.trader.get_positions()
                    if positions:
                        log.info(f"Open positions: {len(positions)}")
                        for p in positions:
                            hours = self.trader.get_hours_since_entry(p.symbol)
                            log.info(f"  {p.symbol}: {'BUY' if p.type == 0 else 'SELL'} P/L: {p.profit:.2f} | {hours:.1f}h")

                time.sleep(5)

            except KeyboardInterrupt:
                log.info("Interrupted")
                self.running = False
            except Exception as e:
                log.error(f"Error: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(30)

        log.info("Strategy stopped")

    def shutdown(self):
        self.running = False
        self.trader.disconnect()


def main():
    strategy = QualityEntryStrategyV2()
    try:
        if strategy.initialize():
            strategy.run()
    except Exception as e:
        log.error(f"Fatal error: {e}")
    finally:
        strategy.shutdown()


if __name__ == "__main__":
    main()
