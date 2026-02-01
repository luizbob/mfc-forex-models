"""
Quality Entry Trader for MT5 (M30 Base)
=======================================
Uses XGBoost to filter quality entries on M30 timeframe.

Strategy:
- Entry: MFC at extreme (≤ -0.5 or ≥ 0.5) AND model predicts quality
- Exit: MFC returns to center (0) OR timeout
- Filter: Only take trades where quality_prob >= threshold

Quality = Small drawdown (adverse MFC < 0.25) + MFC returns to center

M30 trades are faster than H1:
- Median duration: 8 hours (vs 23h for H1)
- 95% complete within 25 hours
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import json
import time
import pickle
import joblib
from datetime import datetime, timedelta
from pathlib import Path
import logging

# ============================================================================
# CONFIGURATION
# ============================================================================

# MT5 Settings
MT5_LOGIN = None
MT5_PASSWORD = None
MT5_SERVER = None
SYMBOL_SUFFIX = "m"

# File Paths
MFC_FILE_PATH = Path(os.environ.get('APPDATA', '')) / "MetaQuotes/Terminal/Common/Files/DWX/DWX_MFC_Auto.txt"
MODEL_DIR = Path(__file__).parent.parent / "models"

# Strategy Parameters - M30 Quality Entry
MFC_EXTREME_THRESHOLD = 0.5   # Entry zone
QUALITY_THRESHOLD = 0.90      # Minimum quality probability (0.9 = 89.5% quality, 87.7% win rate)
STOCH_PERIOD = 25             # Stochastic period (on M30)
STOCH_EXIT_HIGH = 80
STOCH_EXIT_LOW = 20
MAX_BARS_HOLD = 50            # M30 bars (25 hours - covers 95th percentile)
FRIDAY_CUTOFF_HOUR = 12       # No entries after Friday 12:00

# Trading Parameters
LOT_SIZE = 0.01
MAGIC_NUMBER = 150011         # Different from H1 trader (150010)
MAX_SPREAD_PIPS = 5
MAX_POSITIONS = 10

# All 28 pairs
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

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger(__name__)


# ============================================================================
# QUALITY MODEL MANAGER (M30)
# ============================================================================

class QualityModelManager:
    """Manages the M30 XGBoost quality entry classifier."""

    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.model = None
        self.feature_cols = None

    def load(self) -> bool:
        """Load the M30 quality classifier model (properly trained)."""
        model_path = self.model_dir / 'quality_xgb_m30_proper.joblib'
        features_path = self.model_dir / 'quality_xgb_features_m30_proper.pkl'

        if not model_path.exists():
            log.error(f"M30 model not found: {model_path}")
            return False

        if not features_path.exists():
            log.error(f"M30 features not found: {features_path}")
            return False

        try:
            self.model = joblib.load(model_path)
            with open(features_path, 'rb') as f:
                self.feature_cols = pickle.load(f)
            log.info(f"Loaded M30 quality model (proper): {len(self.feature_cols)} features")
            return True
        except Exception as e:
            log.error(f"Error loading model: {e}")
            return False

    def predict_quality(self, features: dict) -> float:
        """Predict quality probability for an entry."""
        if self.model is None:
            return 0.0

        try:
            X = np.array([[features.get(col, 0.0) for col in self.feature_cols]])
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            prob = self.model.predict_proba(X)[0, 1]
            return float(prob)
        except Exception as e:
            log.error(f"Prediction error: {e}")
            return 0.0


# ============================================================================
# MFC DATA READER
# ============================================================================

class MFCDataReader:
    """Reads MFC data from the auto-export file."""

    def __init__(self, file_path: Path, host_symbol: str = "EURUSDm"):
        self.file_path = file_path
        self.host_symbol = host_symbol
        self.last_mtime = 0  # Track file modification time

    def wait_for_fresh_data(self, bar_time: datetime, timeout: float = 120) -> bool:
        """Wait for MFC file to be updated AFTER bar close time."""
        start_time = time.time()
        bar_timestamp = bar_time.timestamp()

        while time.time() - start_time < timeout:
            if not self.file_path.exists():
                time.sleep(1)
                continue

            current_mtime = self.file_path.stat().st_mtime

            # File must be written AFTER the bar closed
            if current_mtime > bar_timestamp:
                self.last_mtime = current_mtime
                log.info(f"Fresh MFC data detected (mtime: {datetime.fromtimestamp(current_mtime).strftime('%H:%M:%S')} > bar: {bar_time.strftime('%H:%M:%S')})")
                return True

            time.sleep(1)

        log.warning(f"Timeout waiting for fresh MFC data ({timeout}s) - last mtime: {datetime.fromtimestamp(self.file_path.stat().st_mtime).strftime('%H:%M:%S')}")
        return False

    def read(self) -> dict:
        """Read MFC data from file."""
        if not self.file_path.exists():
            return None

        try:
            with open(self.file_path, 'r') as f:
                raw_json = f.read()

            raw_json = raw_json.strip()
            if not raw_json.startswith('{') or not raw_json.endswith('}'):
                return None

            parsed = json.loads(raw_json)

        except Exception as e:
            log.error(f"MFC read error: {e}")
            return None

        try:
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
            log.error(f"Error parsing MFC data: {e}")
            return None

    def get_mfc_features(self, base: str, quote: str, direction: str) -> dict:
        """
        Get all MFC features for M30 quality prediction.
        Uses M30 as base timeframe.
        """
        data = self.read()
        if not data:
            return None

        if base not in data or quote not in data:
            return None

        features = {}

        # Get MFC values for all timeframes (shifted - use [-2] for last closed bar)
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

        # Additional M30 features (base timeframe for this model)
        base_m30 = data[base].get('M30', [])

        if len(base_m30) >= 4:
            features['base_vel2_m30'] = base_m30[-2] - base_m30[-4]
            vel1 = base_m30[-2] - base_m30[-3]
            vel2 = base_m30[-3] - base_m30[-4]
            features['base_acc_m30'] = vel1 - vel2
        else:
            features['base_vel2_m30'] = 0
            features['base_acc_m30'] = 0

        # Divergence (M30 based)
        features['divergence'] = features['base_m30'] - features['quote_m30']
        features['vel_divergence'] = features['base_vel_m30'] - features['quote_vel_m30']

        # Direction code
        features['direction_code'] = 1 if direction == 'buy' else 0

        return features


# ============================================================================
# MT5 TRADER
# ============================================================================

class MT5Trader:
    """Handles all MT5 trading operations."""

    def __init__(self):
        self.connected = False

    def connect(self) -> bool:
        if not mt5.initialize():
            log.error(f"MT5 initialize() failed: {mt5.last_error()}")
            return False

        if MT5_LOGIN and MT5_PASSWORD and MT5_SERVER:
            if not mt5.login(MT5_LOGIN, MT5_PASSWORD, MT5_SERVER):
                log.error(f"MT5 login failed: {mt5.last_error()}")
                return False

        account_info = mt5.account_info()
        if account_info:
            log.info(f"Connected to MT5: {account_info.server}")
            log.info(f"Account: {account_info.login}, Balance: {account_info.balance}")

        self.connected = True
        return True

    def disconnect(self):
        mt5.shutdown()
        self.connected = False
        log.info("Disconnected from MT5")

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

        if 'JPY' in symbol:
            pip_value = point * 100
        else:
            pip_value = point * 10

        return (spread_points * point) / pip_value

    def calculate_stochastic(self, symbol: str, period: int = 25, shift: int = 1) -> float:
        """Calculate Stochastic on M30."""
        bars_needed = period + shift
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M30, 0, bars_needed)

        if rates is None or len(rates) < bars_needed:
            return -1

        highs = [r['high'] for r in rates[:-shift]] if shift > 0 else [r['high'] for r in rates]
        lows = [r['low'] for r in rates[:-shift]] if shift > 0 else [r['low'] for r in rates]
        closes = [r['close'] for r in rates[:-shift]] if shift > 0 else [r['close'] for r in rates]

        if len(closes) < period:
            return -1

        recent_highs = highs[-period:]
        recent_lows = lows[-period:]
        current_close = closes[-1]

        highest_high = max(recent_highs)
        lowest_low = min(recent_lows)

        if highest_high == lowest_low:
            return 50

        stoch_k = ((current_close - lowest_low) / (highest_high - lowest_low)) * 100
        return stoch_k

    def get_positions(self) -> list:
        positions = mt5.positions_get()
        if positions is None:
            return []
        return [p for p in positions if p.magic == MAGIC_NUMBER]

    def has_position(self, symbol: str) -> bool:
        for p in self.get_positions():
            if p.symbol == symbol:
                return True
        return False

    def get_position_type(self, symbol: str) -> int:
        for p in self.get_positions():
            if p.symbol == symbol:
                return p.type
        return -1

    def get_hours_since_entry(self, symbol: str) -> float:
        for p in self.get_positions():
            if p.symbol == symbol:
                open_time = datetime.fromtimestamp(p.time)
                now = datetime.now()
                hours_held = (now - open_time).total_seconds() / 3600
                return hours_held
        return 0

    def count_positions(self) -> int:
        return len(self.get_positions())

    def open_buy(self, symbol: str, lots: float, comment: str = "") -> bool:
        info = self.symbol_info(symbol)
        if info is None:
            return False

        price = info.ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lots,
            "type": mt5.ORDER_TYPE_BUY,
            "price": price,
            "deviation": 10,
            "magic": MAGIC_NUMBER,
            "comment": comment,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            log.error(f"BUY failed for {symbol}: {result.retcode} - {result.comment}")
            return False

        log.info(f"BUY {symbol} @ {price:.5f} - Ticket: {result.order}")
        return True

    def open_sell(self, symbol: str, lots: float, comment: str = "") -> bool:
        info = self.symbol_info(symbol)
        if info is None:
            return False

        price = info.bid

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lots,
            "type": mt5.ORDER_TYPE_SELL,
            "price": price,
            "deviation": 10,
            "magic": MAGIC_NUMBER,
            "comment": comment,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            log.error(f"SELL failed for {symbol}: {result.retcode} - {result.comment}")
            return False

        log.info(f"SELL {symbol} @ {price:.5f} - Ticket: {result.order}")
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

class QualityEntryStrategyM30:
    """M30-based quality entry strategy."""

    def __init__(self):
        self.trader = MT5Trader()
        self.mfc_reader = MFCDataReader(MFC_FILE_PATH)
        self.model_manager = QualityModelManager(MODEL_DIR)
        self.last_bar_time = None
        self.running = False

    def initialize(self) -> bool:
        log.info("=" * 60)
        log.info("QUALITY ENTRY TRADER (M30 Base)")
        log.info("=" * 60)

        if not self.trader.connect():
            return False

        if not self.model_manager.load():
            return False

        if not MFC_FILE_PATH.exists():
            log.error(f"MFC file not found: {MFC_FILE_PATH}")
            return False

        log.info(f"MFC file: {MFC_FILE_PATH}")
        log.info(f"Trading pairs: {len(ALL_PAIRS)} pairs")
        log.info(f"Quality threshold: {QUALITY_THRESHOLD}")
        log.info(f"MFC extreme: ±{MFC_EXTREME_THRESHOLD}")
        log.info(f"Max positions: {MAX_POSITIONS}")
        log.info(f"Max hold: {MAX_BARS_HOLD} M30 bars ({MAX_BARS_HOLD * 0.5:.0f} hours)")
        log.info(f"Lot size: {LOT_SIZE}")
        log.info(f"Magic number: {MAGIC_NUMBER}")
        log.info("=" * 60)

        return True

    def check_entry(self, pair: str, base: str, quote: str) -> str:
        symbol = self.trader.get_symbol(pair)

        # Friday cutoff
        tick = mt5.symbol_info_tick(symbol)
        if tick:
            server_time = datetime.fromtimestamp(tick.time)
            if server_time.weekday() == 4 and server_time.hour >= FRIDAY_CUTOFF_HOUR:
                return None

        # Check spread
        spread = self.trader.get_spread_pips(symbol)
        if spread > MAX_SPREAD_PIPS:
            return None

        # Position limits
        if self.trader.count_positions() >= MAX_POSITIONS:
            return None

        if self.trader.has_position(symbol):
            return None

        # Get MFC features (M30 based)
        buy_features = self.mfc_reader.get_mfc_features(base, quote, 'buy')
        sell_features = self.mfc_reader.get_mfc_features(base, quote, 'sell')

        if buy_features is None:
            return None

        base_mfc = buy_features['base_m30']  # Use M30 for entry decision

        # Check BUY: base MFC at extreme LOW
        if base_mfc <= -MFC_EXTREME_THRESHOLD:
            quality_prob = self.model_manager.predict_quality(buy_features)

            if quality_prob >= QUALITY_THRESHOLD:
                log.info(f"{pair} BUY SIGNAL: base_mfc={base_mfc:.3f}, "
                        f"vel={buy_features['base_vel_m30']:.4f}, "
                        f"quality={quality_prob:.1%}")
                return 'BUY'
            else:
                log.debug(f"{pair} BUY rejected: quality={quality_prob:.1%}")

        # Check SELL: base MFC at extreme HIGH
        if base_mfc >= MFC_EXTREME_THRESHOLD:
            quality_prob = self.model_manager.predict_quality(sell_features)

            if quality_prob >= QUALITY_THRESHOLD:
                log.info(f"{pair} SELL SIGNAL: base_mfc={base_mfc:.3f}, "
                        f"vel={sell_features['base_vel_m30']:.4f}, "
                        f"quality={quality_prob:.1%}")
                return 'SELL'
            else:
                log.debug(f"{pair} SELL rejected: quality={quality_prob:.1%}")

        return None

    def check_exit(self, pair: str, base: str) -> bool:
        symbol = self.trader.get_symbol(pair)

        if not self.trader.has_position(symbol):
            return False

        pos_type = self.trader.get_position_type(symbol)

        # Check MFC return to center (using M30)
        data = self.mfc_reader.read()
        if data and base in data:
            base_m30 = data[base].get('M30', [])
            if len(base_m30) >= 2:
                current_mfc = base_m30[-2]

                if pos_type == mt5.ORDER_TYPE_BUY and current_mfc >= 0:
                    self.trader.close_position(symbol, f"MFC returned: {current_mfc:.3f}")
                    return True

                if pos_type == mt5.ORDER_TYPE_SELL and current_mfc <= 0:
                    self.trader.close_position(symbol, f"MFC returned: {current_mfc:.3f}")
                    return True

        # Check Stochastic exit
        stoch = self.trader.calculate_stochastic(symbol, STOCH_PERIOD, shift=1)
        if stoch >= 0:
            if pos_type == mt5.ORDER_TYPE_BUY and stoch >= STOCH_EXIT_HIGH:
                self.trader.close_position(symbol, f"Stoch={stoch:.1f}")
                return True

            if pos_type == mt5.ORDER_TYPE_SELL and stoch <= STOCH_EXIT_LOW:
                self.trader.close_position(symbol, f"Stoch={stoch:.1f}")
                return True

        # Check timeout (in hours, convert from M30 bars)
        hours_held = self.trader.get_hours_since_entry(symbol)
        max_hours = MAX_BARS_HOLD * 0.5  # M30 bars to hours
        if hours_held >= max_hours:
            self.trader.close_position(symbol, f"Timeout ({hours_held:.1f}h)")
            return True

        return False

    def process_bar(self):
        log.info(f"[{self.last_bar_time.strftime('%Y-%m-%d %H:%M')}] Scanning {len(ALL_PAIRS)} pairs...")

        signals_found = 0
        for pair, base, quote in ALL_PAIRS:
            symbol = self.trader.get_symbol(pair)

            if self.trader.has_position(symbol):
                self.check_exit(pair, base)
            else:
                signal = self.check_entry(pair, base, quote)

                if signal == 'BUY':
                    signals_found += 1
                    self.trader.open_buy(symbol, LOT_SIZE, "M30_QUALITY_BUY")
                elif signal == 'SELL':
                    signals_found += 1
                    self.trader.open_sell(symbol, LOT_SIZE, "M30_QUALITY_SELL")

        if signals_found == 0:
            log.info(f"No signals this bar")

    def wait_for_new_bar(self) -> bool:
        """Wait for new M30 bar."""
        ref_symbol = self.trader.get_symbol(ALL_PAIRS[0][0])

        rates = mt5.copy_rates_from_pos(ref_symbol, mt5.TIMEFRAME_M30, 0, 1)
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
        log.info("Strategy started. Waiting for M30 bar signals...")

        while self.running:
            try:
                if self.wait_for_new_bar():
                    log.info(f"New M30 bar detected ({self.last_bar_time.strftime('%H:%M')}), waiting for fresh MFC data...")
                    if self.mfc_reader.wait_for_fresh_data(self.last_bar_time, timeout=120):
                        self.process_bar()
                    else:
                        log.warning("MFC data timeout - skipping this bar")

                    # Show status
                    positions = self.trader.get_positions()
                    if positions:
                        log.info(f"Open positions: {len(positions)}")
                        for p in positions:
                            hours_held = self.trader.get_hours_since_entry(p.symbol)
                            log.info(f"  {p.symbol}: {('BUY' if p.type == 0 else 'SELL')} "
                                   f"P/L: {p.profit:.2f} | {hours_held:.1f}h held")

                time.sleep(5)

            except KeyboardInterrupt:
                log.info("Interrupted by user")
                self.running = False
            except Exception as e:
                log.error(f"Error in main loop: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(30)

        log.info("Strategy stopped")

    def shutdown(self):
        self.running = False
        self.trader.disconnect()


# ============================================================================
# MAIN
# ============================================================================

def main():
    strategy = QualityEntryStrategyM30()

    try:
        if strategy.initialize():
            strategy.run()
    except Exception as e:
        log.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        strategy.shutdown()


if __name__ == "__main__":
    main()
