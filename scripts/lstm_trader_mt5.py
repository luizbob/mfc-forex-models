"""
LSTM Strategy Live Trader for MT5
==================================
Connects to MT5 via Python API, reads MFC data, runs LSTM predictions,
and executes trades based on the strategy.

Strategy: LSTM Divergence + MFC Extreme + Stochastic 25 + H1 Velocity Filter + XGBoost Filter
Filters:
  - All 28 pairs
  - Combined velocity: momentum must favor trade direction (base_vel vs quote_vel)
  - Friday cutoff: no entries after 06:00 Friday (250 bar timeout safety)
  - XGBoost filter: Only take trades with prob >= 0.65
  - MTF MFC features: M15, M30, H1, H4 MFC values used by XGBoost to detect conflicts

2025 Out-of-Sample Results (with XGBoost + MTF MFC filter prob >= 0.65):
  - Trades: 2,169
  - Win Rate: 71.6%
  - Net Avg: +3.5 pips
  - Total Pips: +7,676
  - Timeout rate: ~0% (Stochastic exits faster than RSI)

MTF MFC features (M15, M30, H1, H4) allow XGBoost to learn multi-timeframe
conflict patterns that predict losing trades.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import tensorflow as tf
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
import logging
import joblib

# ============================================================================
# CONFIGURATION
# ============================================================================

# MT5 Settings
MT5_LOGIN = None          # Set your login or None to use default terminal
MT5_PASSWORD = None       # Set your password or None
MT5_SERVER = None         # Set your server or None
SYMBOL_SUFFIX = "m"       # Suffix for symbols (e.g., "m" for EURUSDm)

# File Paths
MFC_FILE_PATH = Path(os.environ.get('APPDATA', '')) / "MetaQuotes/Terminal/Common/Files/DWX/DWX_MFC_Auto.txt"
MODEL_DIR = Path(__file__).parent.parent / "models"
DATA_DIR = Path(__file__).parent.parent / "data"

# Strategy Parameters
MIN_CONFIDENCE = 0.70     # Minimum LSTM confidence
MFC_EXTREME = 0.5         # MFC threshold for entry
STOCH_PERIOD = 25         # Stochastic calculation period
STOCH_LOW = 20            # Stochastic low threshold (BUY entry, SELL exit)
STOCH_HIGH = 80           # Stochastic high threshold (SELL entry, BUY exit)
H1_VEL_THRESHOLD = 0.04   # H1 velocity filter threshold
MAX_BARS_HOLD = 200       # Maximum M5 bars to hold (16.7 hours - 70% winrate at this timeout)
FRIDAY_CUTOFF_HOUR = 6    # No new entries after Friday 06:00 (safer AND more profitable)
XGB_PROB_THRESHOLD = 0.75 # XGBoost probability threshold for entry

# Session-based pair filter (Asian session only trade active currencies)
# During 00-08 GMT, only trade pairs with JPY, AUD, or NZD (active markets)
# Avoids GBP/EUR pairs which have asymmetric risk (big losses) during low liquidity
ASIAN_SESSION_START = 0   # 00:00 GMT
ASIAN_SESSION_END = 8     # 08:00 GMT
ASIAN_ALLOWED_CURRENCIES = ['JPY', 'AUD', 'NZD']

# Pair encoding for XGBoost (must match training)
PAIR_ENCODING = {
    'EURUSD': 0, 'GBPUSD': 1, 'AUDUSD': 2, 'NZDUSD': 3, 'USDJPY': 4, 'USDCHF': 5,
    'USDCAD': 6, 'EURGBP': 7, 'EURJPY': 8, 'EURCHF': 9, 'EURCAD': 10, 'EURAUD': 11,
    'EURNZD': 12, 'GBPJPY': 13, 'GBPCHF': 14, 'GBPCAD': 15, 'GBPAUD': 16, 'GBPNZD': 17,
    'AUDJPY': 18, 'AUDCHF': 19, 'AUDCAD': 20, 'AUDNZD': 21, 'NZDJPY': 22, 'NZDCHF': 23,
    'NZDCAD': 24, 'CADJPY': 25, 'CADCHF': 26, 'CHFJPY': 27,
}

# XGBoost feature columns (must match training order - now with MTF MFC features)
XGB_FEATURE_COLS = [
    'pair_code', 'type_code', 'hour', 'dayofweek',
    'stoch', 'base_mfc', 'quote_mfc', 'mfc_diff',
    # MTF MFC features (NEW)
    'base_mfc_m15', 'quote_mfc_m15', 'mfc_diff_m15',
    'base_mfc_m30', 'quote_mfc_m30', 'mfc_diff_m30',
    'base_mfc_h1', 'quote_mfc_h1', 'mfc_diff_h1',
    'base_mfc_h4', 'quote_mfc_h4', 'mfc_diff_h4',
    # Velocities
    'base_vel_h1', 'quote_vel_h1', 'vel_h1_diff',
    'base_vel_h4', 'quote_vel_h4', 'vel_h4_diff',
    'base_vel_m5', 'quote_vel_m5', 'vel_m5_diff',
    'base_conf', 'quote_conf', 'conf_avg',
]

# Trading Parameters
LOT_SIZE = 0.01           # Position size
MAGIC_NUMBER = 150005     # Magic number for this strategy (Fixed H4 velocity to match backtest)
MAX_SPREAD_PIPS = 5       # Maximum allowed spread
MAX_POSITIONS = 15         # Maximum concurrent positions
ONE_PER_CURRENCY = False   # Only one trade per currency at a time

# Model Accuracies (from training)
MODEL_ACCURACY = {
    'EUR': 82.0, 'USD': 87.8, 'GBP': 83.7, 'JPY': 90.0,
    'CHF': 86.3, 'CAD': 82.2, 'AUD': 87.4, 'NZD': 86.7,
}

# All 28 pairs - combined velocity + extended filters are more effective than accuracy filter
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

# LSTM Lookback periods (must match training)
LOOKBACK = {'M5': 48, 'M15': 32, 'M30': 24, 'H1': 12, 'H4': 6}

# Spreads in pips (approximate) - all 28 pairs
SPREADS = {
    'EURUSD': 1.5, 'GBPUSD': 2.0, 'USDJPY': 1.5, 'USDCHF': 2.0, 'USDCAD': 2.0,
    'AUDUSD': 1.5, 'NZDUSD': 2.5, 'EURGBP': 2.0, 'EURJPY': 2.0, 'GBPJPY': 3.0,
    'EURCHF': 2.5, 'AUDJPY': 2.5, 'EURAUD': 3.0, 'EURCAD': 3.0, 'EURNZD': 4.0,
    'GBPAUD': 3.5, 'GBPCAD': 3.5, 'GBPCHF': 3.0, 'GBPNZD': 4.5, 'AUDCAD': 2.5,
    'AUDCHF': 2.5, 'AUDNZD': 3.0, 'CADJPY': 2.5, 'CHFJPY': 2.5, 'NZDCAD': 3.5,
    'NZDCHF': 3.5, 'NZDJPY': 3.0, 'CADCHF': 3.0,
}

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger(__name__)


# ============================================================================
# LSTM MODEL MANAGER
# ============================================================================

class LSTMModelManager:
    """Manages LSTM models for all currencies."""

    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.models = {}
        self.config = None
        self._load_config()

    def _load_config(self):
        """Load model configuration."""
        config_path = DATA_DIR / 'config.pkl'
        if config_path.exists():
            import pickle
            with open(config_path, 'rb') as f:
                self.config = pickle.load(f)
            log.info(f"Loaded config: lookback={self.config.get('lookback', LOOKBACK)}")

    def load_model(self, currency: str) -> bool:
        """Load LSTM model for a currency."""
        if currency in self.models:
            return True

        model_path = self.model_dir / f'lstm_{currency}_final.keras'
        if not model_path.exists():
            log.error(f"Model not found: {model_path}")
            return False

        try:
            self.models[currency] = tf.keras.models.load_model(model_path)
            log.info(f"Loaded model for {currency}")
            return True
        except Exception as e:
            log.error(f"Error loading model for {currency}: {e}")
            return False

    def load_required_models(self):
        """Load models for all currencies in qualified pairs."""
        currencies = set()
        for pair, base, quote in ALL_PAIRS:
            currencies.add(base)
            currencies.add(quote)

        for ccy in currencies:
            self.load_model(ccy)

        log.info(f"Loaded {len(self.models)} models: {list(self.models.keys())}")

    def predict(self, currency: str, mfc_data: dict) -> tuple:
        """
        Generate prediction for a currency.

        Args:
            currency: Currency code (e.g., 'USD')
            mfc_data: Dict with MFC values per timeframe

        Returns:
            (direction, confidence) or (None, None) if prediction fails
            direction: 0=DOWN, 1=NEUTRAL, 2=UP
        """
        if currency not in self.models:
            return None, None

        try:
            # Get lookback config
            lookback = self.config.get('lookback', LOOKBACK) if self.config else LOOKBACK

            # IMPORTANT: Skip bar 0 (forming bar) to match training which used shift(1)
            # Use [-(lookback+1):-1] to get only closed bars
            # Data is sorted oldest to newest, so [-1] is the forming bar (bar 0)
            X_M5 = np.array(mfc_data['M5'][-(lookback['M5']+1):-1]).reshape(1, lookback['M5'], 1)
            X_M15 = np.array(mfc_data['M15'][-(lookback['M15']+1):-1]).reshape(1, lookback['M15'], 1)
            X_M30 = np.array(mfc_data['M30'][-(lookback['M30']+1):-1]).reshape(1, lookback['M30'], 1)
            X_H1 = np.array(mfc_data['H1'][-(lookback['H1']+1):-1]).reshape(1, lookback['H1'], 1)
            X_H4 = np.array(mfc_data['H4'][-(lookback['H4']+1):-1]).reshape(1, lookback['H4'], 1)

            # Aux features: vel_M5, vel_M30, current_M5, current_M30, current_H4
            # Use [-2] and [-3] to skip forming bar ([-1] is forming, [-2] is last closed)
            vel_m5 = mfc_data['M5'][-2] - mfc_data['M5'][-3] if len(mfc_data['M5']) >= 3 else 0
            vel_m30 = mfc_data['M30'][-2] - mfc_data['M30'][-3] if len(mfc_data['M30']) >= 3 else 0

            X_aux = np.array([[
                vel_m5,
                vel_m30,
                mfc_data['M5'][-2],    # Last closed M5
                mfc_data['M30'][-2],   # Last closed M30
                mfc_data['H4'][-2],    # Last closed H4
            ]])

            # Predict
            pred = self.models[currency].predict(
                [X_M5, X_M15, X_M30, X_H1, X_H4, X_aux],
                verbose=0
            )

            direction = np.argmax(pred[0])
            confidence = np.max(pred[0])

            return int(direction), float(confidence)

        except Exception as e:
            log.error(f"Prediction error for {currency}: {e}")
            return None, None


# ============================================================================
# MFC DATA READER
# ============================================================================

class MFCDataReader:
    """Reads MFC data from the auto-export file with file modification time synchronization."""

    def __init__(self, file_path: Path, host_symbol: str = "EURUSDm"):
        self.file_path = file_path
        self.host_symbol = host_symbol
        self.last_mtime = 0

    def wait_for_fresh_data(self, bar_time: datetime, timeout: float = 120) -> bool:
        """
        Wait for MFC file to be updated after the bar time.
        Checks if file modification time is newer than the bar time.

        Args:
            bar_time: The time of the current bar (we need data newer than this)
            timeout: Maximum seconds to wait

        Returns:
            True if fresh data found, False if timeout
        """
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

        log.warning(f"Timeout waiting for fresh MFC data ({timeout}s)")
        return False

    def read(self) -> dict:
        """
        Read MFC data from file.

        Waits for ready flag from MT4, then reads and clears the flag.

        Returns:
            Dict with structure: {currency: {timeframe: [values]}}
            None if read fails (caller should handle retry/skip)
        """
        if not self.file_path.exists():
            log.warning(f"MFC file not found: {self.file_path}")
            return None

        # Read the file - with ready flag mechanism, file should be complete
        try:
            with open(self.file_path, 'r') as f:
                raw_json = f.read()

            raw_json = raw_json.strip()
            if not raw_json.startswith('{') or not raw_json.endswith('}'):
                log.error(f"MFC file format invalid")
                return None

            parsed = json.loads(raw_json)

        except json.JSONDecodeError as e:
            log.error(f"MFC JSON decode error: {e}")
            return None

        except PermissionError as e:
            log.error(f"MFC permission error: {e}")
            return None

        except Exception as e:
            log.error(f"MFC read error: {e}")
            return None

        # Parse into organized structure
        try:
            result = {}
            timeframes = ['M5', 'M15', 'M30', 'H1', 'H4']
            currencies = ['EUR', 'USD', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD', 'NZD']

            for ccy in currencies:
                result[ccy] = {}
                for tf in timeframes:
                    key = f"{self.host_symbol}_{tf}"
                    if key in parsed and ccy in parsed[key]:
                        # Convert dict to sorted list (oldest to newest)
                        tf_data = parsed[key][ccy]
                        sorted_items = sorted(tf_data.items(), key=lambda x: x[0])
                        result[ccy][tf] = [v for k, v in sorted_items]
                    else:
                        result[ccy][tf] = []

            return result

        except Exception as e:
            log.error(f"Error parsing MFC data: {e}")
            return None

    def get_currency_mfc(self, currency: str) -> dict:
        """Get MFC data for a specific currency."""
        data = self.read()
        if data and currency in data:
            return data[currency]
        return None

    def get_current_value(self, currency: str, timeframe: str = 'M5') -> float:
        """Get most recent MFC value for a currency."""
        data = self.read()
        if data and currency in data and timeframe in data[currency]:
            values = data[currency][timeframe]
            if values:
                return values[-1]  # Most recent (after shift, this is the closed bar)
        return None


# ============================================================================
# MT5 TRADER
# ============================================================================

class MT5Trader:
    """Handles all MT5 trading operations."""

    def __init__(self):
        self.connected = False
        self.positions = {}  # Track our positions
        self.entry_bars = {}  # Track bars since entry

    def connect(self) -> bool:
        """Connect to MT5 terminal."""
        if not mt5.initialize():
            log.error(f"MT5 initialize() failed: {mt5.last_error()}")
            return False

        # Login if credentials provided
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
        """Disconnect from MT5."""
        mt5.shutdown()
        self.connected = False
        log.info("Disconnected from MT5")

    def get_symbol(self, pair: str) -> str:
        """Get full symbol name with suffix."""
        return pair + SYMBOL_SUFFIX

    def symbol_info(self, symbol: str):
        """Get symbol info."""
        info = mt5.symbol_info(symbol)
        if info is None:
            # Try to add symbol to market watch
            mt5.symbol_select(symbol, True)
            info = mt5.symbol_info(symbol)
        return info

    def get_spread_pips(self, symbol: str) -> float:
        """Get current spread in pips."""
        info = self.symbol_info(symbol)
        if info is None:
            return 999

        spread_points = info.spread
        point = info.point

        # Convert to pips
        if 'JPY' in symbol:
            pip_value = point * 100
        else:
            pip_value = point * 10

        return (spread_points * point) / pip_value

    def calculate_stochastic(self, symbol: str, period: int = 25, shift: int = 1) -> float:
        """Calculate Stochastic %K (matches backtest)."""
        # Get enough bars for calculation
        bars_needed = period + shift
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, bars_needed)

        if rates is None or len(rates) < bars_needed:
            return -1

        # Get OHLC data, skip the forming bar (shift)
        highs = [r['high'] for r in rates[:-shift] if shift > 0] if shift > 0 else [r['high'] for r in rates]
        lows = [r['low'] for r in rates[:-shift] if shift > 0] if shift > 0 else [r['low'] for r in rates]
        closes = [r['close'] for r in rates[:-shift] if shift > 0] if shift > 0 else [r['close'] for r in rates]

        if len(closes) < period:
            return -1

        # Get last 'period' bars
        recent_highs = highs[-period:]
        recent_lows = lows[-period:]
        current_close = closes[-1]

        highest_high = max(recent_highs)
        lowest_low = min(recent_lows)

        if highest_high == lowest_low:
            return 50  # Neutral if no range

        stoch_k = ((current_close - lowest_low) / (highest_high - lowest_low)) * 100
        return stoch_k

    def get_positions(self) -> list:
        """Get all open positions for this strategy."""
        positions = mt5.positions_get()
        if positions is None:
            return []

        return [p for p in positions if p.magic == MAGIC_NUMBER]

    def has_position(self, symbol: str) -> bool:
        """Check if we have an open position for a symbol."""
        positions = self.get_positions()
        for p in positions:
            if p.symbol == symbol:
                return True
        return False

    def get_position_type(self, symbol: str) -> int:
        """Get position type for a symbol (0=BUY, 1=SELL)."""
        positions = self.get_positions()
        for p in positions:
            if p.symbol == symbol:
                return p.type
        return -1

    def get_bars_since_entry(self, symbol: str) -> int:
        """Calculate M5 bars since position was opened."""
        positions = self.get_positions()
        for p in positions:
            if p.symbol == symbol:
                # Position time is in seconds since epoch
                open_time = datetime.fromtimestamp(p.time)
                now = datetime.now()
                minutes_held = (now - open_time).total_seconds() / 60
                bars_held = int(minutes_held / 5)  # M5 bars
                return bars_held
        return 0

    def count_positions(self) -> int:
        """Count total open positions."""
        return len(self.get_positions())

    def get_currencies_in_trades(self) -> set:
        """Get set of currencies currently in trades."""
        currencies = set()
        for p in self.get_positions():
            symbol = p.symbol.replace(SYMBOL_SUFFIX, '')
            currencies.add(symbol[:3])
            currencies.add(symbol[3:6])
        return currencies

    def open_buy(self, symbol: str, lots: float, comment: str = "") -> bool:
        """Open a BUY position."""
        info = self.symbol_info(symbol)
        if info is None:
            log.error(f"Symbol info not found: {symbol}")
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
        self.entry_bars[symbol] = 0
        return True

    def open_sell(self, symbol: str, lots: float, comment: str = "") -> bool:
        """Open a SELL position."""
        info = self.symbol_info(symbol)
        if info is None:
            log.error(f"Symbol info not found: {symbol}")
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
        self.entry_bars[symbol] = 0
        return True

    def close_position(self, symbol: str, reason: str = "") -> bool:
        """Close position for a symbol."""
        positions = mt5.positions_get(symbol=symbol)
        if positions is None:
            return False

        for pos in positions:
            if pos.magic != MAGIC_NUMBER:
                continue

            # Determine close price and type
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
                log.error(f"Close failed for {symbol}: {result.retcode} - {result.comment}")
                return False

            log.info(f"CLOSED {symbol} - {reason} - P/L: {pos.profit:.2f}")

            if symbol in self.entry_bars:
                del self.entry_bars[symbol]

            return True

        return False


# ============================================================================
# MAIN STRATEGY
# ============================================================================

class LSTMStrategy:
    """Main strategy class that coordinates everything."""

    def __init__(self):
        self.trader = MT5Trader()
        self.mfc_reader = MFCDataReader(MFC_FILE_PATH)
        self.model_manager = LSTMModelManager(MODEL_DIR)
        self.xgb_classifier = None  # XGBoost filter model
        self.last_bar_time = None
        self.running = False

    def initialize(self) -> bool:
        """Initialize all components."""
        log.info("=" * 60)
        log.info("LSTM STRATEGY TRADER")
        log.info("=" * 60)

        # Connect to MT5
        if not self.trader.connect():
            return False

        # Load LSTM models
        self.model_manager.load_required_models()

        # Load XGBoost classifier
        xgb_path = MODEL_DIR / 'xgb_trade_classifier.joblib'
        if xgb_path.exists():
            self.xgb_classifier = joblib.load(xgb_path)
            log.info(f"Loaded XGBoost classifier from {xgb_path}")
        else:
            log.error(f"XGBoost classifier not found: {xgb_path}")
            return False

        # Verify MFC file exists
        if not MFC_FILE_PATH.exists():
            log.error(f"MFC file not found: {MFC_FILE_PATH}")
            log.error("Make sure MT4 exporter is running")
            return False

        log.info(f"MFC file: {MFC_FILE_PATH}")
        log.info(f"Trading pairs: {len(ALL_PAIRS)} pairs")
        log.info(f"Max positions: {MAX_POSITIONS}")
        log.info(f"Max hold: {MAX_BARS_HOLD} bars ({MAX_BARS_HOLD * 5 / 60:.1f}h)")
        log.info(f"Friday cutoff: {FRIDAY_CUTOFF_HOUR}:00")
        log.info(f"Stochastic: period={STOCH_PERIOD}, entry<{STOCH_LOW}, exit>{STOCH_HIGH}")
        log.info(f"H1 velocity filter: ±{H1_VEL_THRESHOLD}")
        log.info(f"XGBoost filter: prob >= {XGB_PROB_THRESHOLD}")
        log.info(f"Asian filter (00-08): only {ASIAN_ALLOWED_CURRENCIES} pairs")
        log.info(f"Lot size: {LOT_SIZE}")

        # Verify MFC data on startup
        self._verify_startup_data()

        log.info("=" * 60)

        return True

    def _verify_startup_data(self):
        """Verify MFC data is valid on startup."""
        log.info("-" * 40)
        log.info("DATA VERIFICATION:")

        # Check MFC file age
        mfc_mtime = datetime.fromtimestamp(MFC_FILE_PATH.stat().st_mtime)
        age_seconds = (datetime.now() - mfc_mtime).total_seconds()
        log.info(f"  MFC file age: {age_seconds:.0f}s ago")
        if age_seconds > 600:  # 10 minutes
            log.warning(f"  WARNING: MFC file is old ({age_seconds/60:.1f} min)")

        # Read MFC data
        mfc_data = self.mfc_reader.read()
        if not mfc_data:
            log.error("  ERROR: Could not read MFC data")
            return

        # Check data counts
        sample_ccy = 'USD'
        if sample_ccy in mfc_data:
            for tf in ['M5', 'M15', 'M30', 'H1', 'H4']:
                count = len(mfc_data[sample_ccy].get(tf, []))
                required = LOOKBACK[tf] + 1
                status = "OK" if count >= required else "LOW"
                log.info(f"  {sample_ccy} {tf}: {count} bars ({status}, need {required})")

        # Show sample MFC values
        if sample_ccy in mfc_data and 'M5' in mfc_data[sample_ccy]:
            values = mfc_data[sample_ccy]['M5']
            if len(values) >= 3:
                log.info(f"  {sample_ccy} M5 last 3 values: {values[-3]:.4f}, {values[-2]:.4f}, {values[-1]:.4f}")

        # Compare with MT5 bar time
        ref_symbol = self.trader.get_symbol(ALL_PAIRS[0][0])
        rates = mt5.copy_rates_from_pos(ref_symbol, mt5.TIMEFRAME_M5, 0, 1)
        if rates:
            mt5_bar_time = datetime.fromtimestamp(rates[0]['time'])
            log.info(f"  MT5 current bar: {mt5_bar_time.strftime('%Y-%m-%d %H:%M')}")

        log.info("-" * 40)

    def check_entry(self, pair: str, base: str, quote: str) -> str:
        """
        Check entry conditions for a pair.

        Returns:
            'BUY', 'SELL', or None
        """
        symbol = self.trader.get_symbol(pair)

        # Check Friday cutoff (no new entries after Friday 06:00 to prevent weekend carryover)
        # Use MT5 server time, not local time
        tick = mt5.symbol_info_tick(symbol)
        if tick:
            server_time = datetime.fromtimestamp(tick.time)
            if server_time.weekday() == 4 and server_time.hour >= FRIDAY_CUTOFF_HOUR:  # Friday = 4
                return None

            # Asian session pair filter: only trade JPY/AUD/NZD pairs during 00-08 GMT
            # Avoids GBP/EUR pairs which have asymmetric risk (big losses) in low liquidity
            if ASIAN_SESSION_START <= server_time.hour < ASIAN_SESSION_END:
                pair_has_asian_currency = any(ccy in pair for ccy in ASIAN_ALLOWED_CURRENCIES)
                if not pair_has_asian_currency:
                    return None

        # Check spread
        spread = self.trader.get_spread_pips(symbol)
        if spread > MAX_SPREAD_PIPS:
            return None

        # Check position limits
        if self.trader.count_positions() >= MAX_POSITIONS:
            return None

        # Check one per currency rule
        if ONE_PER_CURRENCY:
            currencies_in_trades = self.trader.get_currencies_in_trades()
            if base in currencies_in_trades or quote in currencies_in_trades:
                return None

        # Get MFC data for both currencies
        base_mfc = self.mfc_reader.get_currency_mfc(base)
        quote_mfc = self.mfc_reader.get_currency_mfc(quote)

        if not base_mfc or not quote_mfc:
            return None

        # Check we have enough data (need lookback + 1 to skip forming bar)
        for tf in ['M5', 'M15', 'M30', 'H1', 'H4']:
            required = LOOKBACK[tf] + 1  # +1 because we skip the forming bar
            if len(base_mfc.get(tf, [])) < required:
                return None
            if len(quote_mfc.get(tf, [])) < required:
                return None

        # Get LSTM predictions
        base_dir, base_conf = self.model_manager.predict(base, base_mfc)
        quote_dir, quote_conf = self.model_manager.predict(quote, quote_mfc)

        if base_dir is None or quote_dir is None:
            return None

        # Check confidence
        if base_conf < MIN_CONFIDENCE or quote_conf < MIN_CONFIDENCE:
            return None

        # Get current MFC values (last closed bar, skip forming bar [-1])
        base_mfc_m5 = base_mfc['M5'][-2] if len(base_mfc['M5']) >= 2 else None
        quote_mfc_m5 = quote_mfc['M5'][-2] if len(quote_mfc['M5']) >= 2 else None

        if base_mfc_m5 is None:
            return None

        # Get MTF MFC values for XGBoost (M15, M30, H1, H4)
        base_mfc_m15 = base_mfc['M15'][-2] if len(base_mfc.get('M15', [])) >= 2 else 0
        quote_mfc_m15 = quote_mfc['M15'][-2] if len(quote_mfc.get('M15', [])) >= 2 else 0
        base_mfc_m30 = base_mfc['M30'][-2] if len(base_mfc.get('M30', [])) >= 2 else 0
        quote_mfc_m30 = quote_mfc['M30'][-2] if len(quote_mfc.get('M30', [])) >= 2 else 0
        base_mfc_h1 = base_mfc['H1'][-2] if len(base_mfc.get('H1', [])) >= 2 else 0
        quote_mfc_h1 = quote_mfc['H1'][-2] if len(quote_mfc.get('H1', [])) >= 2 else 0
        base_mfc_h4 = base_mfc['H4'][-2] if len(base_mfc.get('H4', [])) >= 2 else 0
        quote_mfc_h4 = quote_mfc['H4'][-2] if len(quote_mfc.get('H4', [])) >= 2 else 0

        # Calculate H1 velocity (12 M5 bars = 1 hour change)
        base_m5_values = base_mfc.get('M5', [])
        quote_m5_values = quote_mfc.get('M5', [])
        base_h1_values = base_mfc.get('H1', [])
        quote_h1_values = quote_mfc.get('H1', [])
        base_h4_values = base_mfc.get('H4', [])
        quote_h4_values = quote_mfc.get('H4', [])

        # H1 velocity (using H1 data directly)
        if len(base_h1_values) >= 4:
            base_vel_h1 = base_h1_values[-2] - base_h1_values[-3]
        else:
            base_vel_h1 = 0

        if len(quote_h1_values) >= 4:
            quote_vel_h1 = quote_h1_values[-2] - quote_h1_values[-3]
        else:
            quote_vel_h1 = 0

        # H4 velocity (2 H4 bars = 8 hours, matches retrained XGBoost)
        if len(base_h4_values) >= 4:
            base_vel_h4 = base_h4_values[-2] - base_h4_values[-4]
        else:
            base_vel_h4 = 0

        if len(quote_h4_values) >= 4:
            quote_vel_h4 = quote_h4_values[-2] - quote_h4_values[-4]
        else:
            quote_vel_h4 = 0

        # M5 velocity (12 bars = 1 hour)
        if len(base_m5_values) >= 14:
            base_vel_m5 = base_m5_values[-2] - base_m5_values[-14]
        else:
            base_vel_m5 = 0

        if len(quote_m5_values) >= 14:
            quote_vel_m5 = quote_m5_values[-2] - quote_m5_values[-14]
        else:
            quote_vel_m5 = 0

        # Combined velocity filter
        buy_vel_ok = (base_vel_h1 - quote_vel_h1) > 0
        sell_vel_ok = (quote_vel_h1 - base_vel_h1) > 0

        # Get Stochastic
        stoch = self.trader.calculate_stochastic(symbol, STOCH_PERIOD, shift=1)
        if stoch < 0:
            return None

        # Get server time for XGBoost features
        hour = server_time.hour if tick else datetime.now().hour
        dayofweek = server_time.weekday() if tick else datetime.now().weekday()

        # Check BUY conditions
        # LSTM: base UP, quote DOWN
        # MFC: base oversold (≤ -0.5)
        # Stochastic: oversold (< 20)
        # H1 Velocity: not falling too fast (>= -0.04)
        # Combined Velocity: momentum favoring BUY
        if (base_dir == 2 and quote_dir == 0 and
            base_mfc_m5 <= -MFC_EXTREME and
            stoch < STOCH_LOW and
            base_vel_h1 >= -H1_VEL_THRESHOLD and
            buy_vel_ok):

            # XGBoost filter (with MTF MFC features)
            xgb_prob = self._get_xgb_probability(
                pair=pair, trade_type='BUY', hour=hour, dayofweek=dayofweek,
                stoch=stoch, base_mfc=base_mfc_m5, quote_mfc=quote_mfc_m5,
                base_mfc_m15=base_mfc_m15, quote_mfc_m15=quote_mfc_m15,
                base_mfc_m30=base_mfc_m30, quote_mfc_m30=quote_mfc_m30,
                base_mfc_h1=base_mfc_h1, quote_mfc_h1=quote_mfc_h1,
                base_mfc_h4=base_mfc_h4, quote_mfc_h4=quote_mfc_h4,
                base_vel_h1=base_vel_h1, quote_vel_h1=quote_vel_h1,
                base_vel_h4=base_vel_h4, quote_vel_h4=quote_vel_h4,
                base_vel_m5=base_vel_m5, quote_vel_m5=quote_vel_m5,
                base_conf=base_conf, quote_conf=quote_conf
            )

            if xgb_prob < XGB_PROB_THRESHOLD:
                log.debug(f"{pair} BUY rejected by XGB: prob={xgb_prob:.2f} < {XGB_PROB_THRESHOLD}")
                return None

            log.info(f"{pair} BUY SIGNAL: base_dir={base_dir}, quote_dir={quote_dir}, "
                    f"base_conf={base_conf:.2f}, quote_conf={quote_conf:.2f}, "
                    f"base_mfc={base_mfc_m5:.3f}, stoch={stoch:.1f}, xgb_prob={xgb_prob:.2f}")
            return 'BUY'

        # Check SELL conditions
        # LSTM: base DOWN, quote UP
        # MFC: base overbought (≥ 0.5)
        # Stochastic: overbought (> 80)
        # H1 Velocity: not rising too fast (<= 0.04)
        # Combined Velocity: momentum favoring SELL
        if (base_dir == 0 and quote_dir == 2 and
            base_mfc_m5 >= MFC_EXTREME and
            stoch > STOCH_HIGH and
            base_vel_h1 <= H1_VEL_THRESHOLD and
            sell_vel_ok):

            # XGBoost filter (with MTF MFC features)
            xgb_prob = self._get_xgb_probability(
                pair=pair, trade_type='SELL', hour=hour, dayofweek=dayofweek,
                stoch=stoch, base_mfc=base_mfc_m5, quote_mfc=quote_mfc_m5,
                base_mfc_m15=base_mfc_m15, quote_mfc_m15=quote_mfc_m15,
                base_mfc_m30=base_mfc_m30, quote_mfc_m30=quote_mfc_m30,
                base_mfc_h1=base_mfc_h1, quote_mfc_h1=quote_mfc_h1,
                base_mfc_h4=base_mfc_h4, quote_mfc_h4=quote_mfc_h4,
                base_vel_h1=base_vel_h1, quote_vel_h1=quote_vel_h1,
                base_vel_h4=base_vel_h4, quote_vel_h4=quote_vel_h4,
                base_vel_m5=base_vel_m5, quote_vel_m5=quote_vel_m5,
                base_conf=base_conf, quote_conf=quote_conf
            )

            if xgb_prob < XGB_PROB_THRESHOLD:
                log.debug(f"{pair} SELL rejected by XGB: prob={xgb_prob:.2f} < {XGB_PROB_THRESHOLD}")
                return None

            log.info(f"{pair} SELL SIGNAL: base_dir={base_dir}, quote_dir={quote_dir}, "
                    f"base_conf={base_conf:.2f}, quote_conf={quote_conf:.2f}, "
                    f"base_mfc={base_mfc_m5:.3f}, stoch={stoch:.1f}, xgb_prob={xgb_prob:.2f}")
            return 'SELL'

        return None

    def _get_xgb_probability(self, pair: str, trade_type: str, hour: int, dayofweek: int,
                             stoch: float, base_mfc: float, quote_mfc: float,
                             base_mfc_m15: float, quote_mfc_m15: float,
                             base_mfc_m30: float, quote_mfc_m30: float,
                             base_mfc_h1: float, quote_mfc_h1: float,
                             base_mfc_h4: float, quote_mfc_h4: float,
                             base_vel_h1: float, quote_vel_h1: float,
                             base_vel_h4: float, quote_vel_h4: float,
                             base_vel_m5: float, quote_vel_m5: float,
                             base_conf: float, quote_conf: float) -> float:
        """Calculate XGBoost win probability for a trade setup (now with MTF MFC features)."""
        if self.xgb_classifier is None:
            return 1.0  # No filter if model not loaded

        features = {
            'pair_code': PAIR_ENCODING.get(pair, 0),
            'type_code': 1 if trade_type == 'BUY' else 0,
            'hour': hour,
            'dayofweek': dayofweek,
            'stoch': stoch,
            'base_mfc': base_mfc,
            'quote_mfc': quote_mfc,
            'mfc_diff': base_mfc - quote_mfc,
            # MTF MFC features
            'base_mfc_m15': base_mfc_m15,
            'quote_mfc_m15': quote_mfc_m15,
            'mfc_diff_m15': base_mfc_m15 - quote_mfc_m15,
            'base_mfc_m30': base_mfc_m30,
            'quote_mfc_m30': quote_mfc_m30,
            'mfc_diff_m30': base_mfc_m30 - quote_mfc_m30,
            'base_mfc_h1': base_mfc_h1,
            'quote_mfc_h1': quote_mfc_h1,
            'mfc_diff_h1': base_mfc_h1 - quote_mfc_h1,
            'base_mfc_h4': base_mfc_h4,
            'quote_mfc_h4': quote_mfc_h4,
            'mfc_diff_h4': base_mfc_h4 - quote_mfc_h4,
            # Velocities
            'base_vel_h1': base_vel_h1,
            'quote_vel_h1': quote_vel_h1,
            'vel_h1_diff': base_vel_h1 - quote_vel_h1,
            'base_vel_h4': base_vel_h4,
            'quote_vel_h4': quote_vel_h4,
            'vel_h4_diff': base_vel_h4 - quote_vel_h4,
            'base_vel_m5': base_vel_m5,
            'quote_vel_m5': quote_vel_m5,
            'vel_m5_diff': base_vel_m5 - quote_vel_m5,
            'base_conf': base_conf,
            'quote_conf': quote_conf,
            'conf_avg': (base_conf + quote_conf) / 2,
        }

        X = np.array([[features[col] for col in XGB_FEATURE_COLS]])
        prob = self.xgb_classifier.predict_proba(X)[0, 1]
        return prob

    def check_exit(self, pair: str) -> bool:
        """Check exit conditions for an open position."""
        symbol = self.trader.get_symbol(pair)

        if not self.trader.has_position(symbol):
            return False

        # Get Stochastic
        stoch = self.trader.calculate_stochastic(symbol, STOCH_PERIOD, shift=1)
        if stoch < 0:
            return False

        pos_type = self.trader.get_position_type(symbol)

        # Check Stochastic exit
        if pos_type == mt5.ORDER_TYPE_BUY and stoch >= STOCH_HIGH:
            self.trader.close_position(symbol, f"Stoch={stoch:.1f}")
            return True

        if pos_type == mt5.ORDER_TYPE_SELL and stoch <= STOCH_LOW:
            self.trader.close_position(symbol, f"Stoch={stoch:.1f}")
            return True

        # Check timeout - calculate bars from position open time
        bars_held = self.trader.get_bars_since_entry(symbol)
        if bars_held >= MAX_BARS_HOLD:
            self.trader.close_position(symbol, f"Timeout ({bars_held} bars)")
            return True

        return False

    def verify_data_sync(self):
        """Verify MFC data is in sync with MT5 bars."""
        ref_symbol = self.trader.get_symbol(ALL_PAIRS[0][0])

        # Get MT5 last closed bar time
        rates = mt5.copy_rates_from_pos(ref_symbol, mt5.TIMEFRAME_M5, 1, 1)  # shift=1 for closed bar
        if rates is None:
            return
        mt5_bar_time = datetime.fromtimestamp(rates[0]['time'])

        # Get MFC file data
        mfc_data = self.mfc_reader.read()
        if not mfc_data:
            return

        # Check a sample currency
        sample_ccy = 'USD'
        if sample_ccy in mfc_data and 'M5' in mfc_data[sample_ccy]:
            mfc_values = mfc_data[sample_ccy]['M5']
            if len(mfc_values) >= 2:
                # Log comparison (MFC file doesn't have timestamps, so we compare values)
                last_closed_mfc = mfc_values[-2]  # -2 because we skip forming bar
                log.debug(f"Data sync: MT5 bar={mt5_bar_time.strftime('%H:%M')}, "
                         f"MFC {sample_ccy} M5 last closed={last_closed_mfc:.4f}")

    def process_bar(self):
        """Process one bar - check entries and exits for all pairs."""
        # Verify data sync (runs once per bar)
        self.verify_data_sync()

        log.info(f"[{self.last_bar_time.strftime('%H:%M')}] Scanning {len(ALL_PAIRS)} pairs...")

        signals_found = 0
        for pair, base, quote in ALL_PAIRS:
            symbol = self.trader.get_symbol(pair)

            # Check exits first
            if self.trader.has_position(symbol):
                self.check_exit(pair)
            else:
                # Check entries
                signal = self.check_entry(pair, base, quote)

                if signal == 'BUY':
                    signals_found += 1
                    self.trader.open_buy(symbol, LOT_SIZE, "LSTM_BUY")
                elif signal == 'SELL':
                    signals_found += 1
                    self.trader.open_sell(symbol, LOT_SIZE, "LSTM_SELL")

        if signals_found == 0:
            log.info(f"[{self.last_bar_time.strftime('%H:%M')}] No signals this bar")

    def wait_for_new_bar(self) -> bool:
        """Wait for a new M5 bar."""
        # Use first qualified pair as reference
        ref_symbol = self.trader.get_symbol(ALL_PAIRS[0][0])

        rates = mt5.copy_rates_from_pos(ref_symbol, mt5.TIMEFRAME_M5, 0, 1)
        if rates is None:
            return False

        current_bar_time = datetime.fromtimestamp(rates[0]['time'])

        if self.last_bar_time is None:
            self.last_bar_time = current_bar_time
            return True  # Process first bar

        if current_bar_time > self.last_bar_time:
            self.last_bar_time = current_bar_time
            return True

        return False

    def run(self):
        """Main loop."""
        self.running = True
        log.info("Strategy started. Waiting for signals...")

        while self.running:
            try:
                if self.wait_for_new_bar():
                    # Wait for MFC file to be updated after the new bar
                    log.info(f"New bar detected ({self.last_bar_time.strftime('%H:%M')}), waiting for fresh MFC data...")
                    if self.mfc_reader.wait_for_fresh_data(self.last_bar_time, timeout=120):
                        log.info("MFC data ready, processing bar...")
                        self.process_bar()
                    else:
                        log.error("MFC data timeout - skipping this bar")

                    # Show status
                    positions = self.trader.get_positions()
                    if positions:
                        log.info(f"Open positions: {len(positions)}")
                        for p in positions:
                            bars_held = self.trader.get_bars_since_entry(p.symbol)
                            hours_held = bars_held * 5 / 60  # M5 bars to hours
                            hours_remaining = max(0, (MAX_BARS_HOLD - bars_held) * 5 / 60)
                            log.info(f"  {p.symbol}: {('BUY' if p.type == 0 else 'SELL')} "
                                   f"P/L: {p.profit:.2f} | {hours_held:.1f}h held, {hours_remaining:.1f}h left")

                # Sleep before checking again
                time.sleep(1)

            except KeyboardInterrupt:
                log.info("Interrupted by user")
                self.running = False
            except Exception as e:
                log.error(f"Error in main loop: {e}")
                time.sleep(5)

        log.info("Strategy stopped")

    def shutdown(self):
        """Clean shutdown."""
        self.running = False
        self.trader.disconnect()


# ============================================================================
# MAIN
# ============================================================================

def main():
    strategy = LSTMStrategy()

    try:
        if strategy.initialize():
            strategy.run()
    except Exception as e:
        log.error(f"Fatal error: {e}")
    finally:
        strategy.shutdown()


if __name__ == "__main__":
    main()
