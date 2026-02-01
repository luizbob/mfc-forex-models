"""
Counter-Trade Strategy - Captures Model's Drawdown
===================================================
When the main model enters, we enter OPPOSITE direction with fixed TP/SL.
The idea: 60% of quality trades have drawdown before profit.

Strategy:
- TP: 20 pips (our target)
- SL: 10 pips (our risk)
- If +10 pips reached: lock profit at +10 pips

Outcomes: Win 20, Win 10 (locked), or Lose 10

Magic Number: 150054
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
LOG_DIR = Path(__file__).parent.parent / "logs"

# Strategy Parameters
MFC_EXTREME_THRESHOLD = 0.5
QUALITY_THRESHOLD = 0.75
FRIDAY_CUTOFF_HOUR = 18

# Counter-Trade Parameters
TP_PIPS = 20
SL_PIPS = 10
LOCK_PROFIT_PIPS = 10  # When +10, lock profit at +10

# Trading Parameters
LOT_SIZE = 0.01
MAGIC_NUMBER = 150054  # Counter-trade magic number
MAX_SPREAD_PIPS = 3  # Tighter spread for scalping
MAX_POSITIONS = 50

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
# TRADE LOGGER (CSV)
# ============================================================================

class TradeLogger:
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.trades_file = self.log_dir / "trades_counter.csv"
        self._init_file()

    def _init_file(self):
        if not self.trades_file.exists():
            with open(self.trades_file, 'w') as f:
                f.write("timestamp,action,symbol,direction,price,spread_pips,quality,tp,sl,ticket,pnl,reason,pips\n")

    def log_entry(self, symbol: str, direction: str, price: float, spread_pips: float,
                  quality: float, tp: float, sl: float, ticket: int = 0):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(self.trades_file, 'a') as f:
            f.write(f"{timestamp},OPEN,{symbol},{direction},{price:.5f},{spread_pips:.2f},{quality:.4f},{tp:.5f},{sl:.5f},{ticket},,,\n")

    def log_exit(self, symbol: str, direction: str, price: float, ticket: int,
                 pnl: float, reason: str, pips: float):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(self.trades_file, 'a') as f:
            f.write(f"{timestamp},CLOSE,{symbol},{direction},{price:.5f},,,,{ticket},{pnl:.2f},{reason},{pips:.1f}\n")


# ============================================================================
# POSITION TRACKER (for trailing stop logic)
# ============================================================================

class PositionTracker:
    """Tracks entry prices and profit lock status for each position."""

    def __init__(self):
        self.positions = {}  # symbol -> {entry_price, direction, locked, ticket}

    def add(self, symbol: str, entry_price: float, direction: str, ticket: int):
        self.positions[symbol] = {
            'entry_price': entry_price,
            'direction': direction,
            'locked': False,
            'ticket': ticket
        }

    def remove(self, symbol: str):
        if symbol in self.positions:
            del self.positions[symbol]

    def get(self, symbol: str):
        return self.positions.get(symbol)

    def set_locked(self, symbol: str):
        if symbol in self.positions:
            self.positions[symbol]['locked'] = True

    def is_locked(self, symbol: str) -> bool:
        pos = self.positions.get(symbol)
        return pos['locked'] if pos else False


# ============================================================================
# QUALITY MODEL MANAGER
# ============================================================================

class QualityModelManager:
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.model = None
        self.feature_cols = None

    def load(self) -> bool:
        model_path = self.model_dir / 'quality_xgb_m5_v2_pnl_walkforward_rolling.joblib'
        features_path = self.model_dir / 'quality_xgb_features_m5_v2_pnl_walkforward_rolling.pkl'

        if not model_path.exists() or not features_path.exists():
            log.error(f"Model not found")
            return False

        try:
            self.model = joblib.load(model_path)
            with open(features_path, 'rb') as f:
                self.feature_cols = pickle.load(f)
            log.info(f"Loaded model: {len(self.feature_cols)} features")
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
            return None

    def wait_for_fresh_data(self, bar_time: datetime, timeout: float = 120) -> bool:
        start_time = time.time()
        bar_timestamp = bar_time.timestamp()

        while time.time() - start_time < timeout:
            if not self.file_path.exists():
                time.sleep(1)
                continue

            current_mtime = self.file_path.stat().st_mtime
            if current_mtime > bar_timestamp:
                return True
            time.sleep(1)

        return False

    def get_mfc_features(self, base: str, quote: str, direction: str, trigger: str) -> dict:
        data = self.read()
        if not data or base not in data or quote not in data:
            return None

        features = {}

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

        base_m5 = data[base].get('M5', [])
        quote_m5 = data[quote].get('M5', [])

        if len(base_m5) >= 4:
            features['base_vel2_m5'] = base_m5[-2] - base_m5[-4]
            features['base_acc_m5'] = (base_m5[-2] - base_m5[-3]) - (base_m5[-3] - base_m5[-4])
        else:
            features['base_vel2_m5'] = 0
            features['base_acc_m5'] = 0

        if len(quote_m5) >= 4:
            features['quote_vel2_m5'] = quote_m5[-2] - quote_m5[-4]
            features['quote_acc_m5'] = (quote_m5[-2] - quote_m5[-3]) - (quote_m5[-3] - quote_m5[-4])
        else:
            features['quote_vel2_m5'] = 0
            features['quote_acc_m5'] = 0

        features['divergence'] = features['base_m5'] - features['quote_m5']
        features['vel_divergence'] = features['base_vel_m5'] - features['quote_vel_m5']
        features['direction_code'] = 1 if direction == 'buy' else 0
        features['trigger_code'] = 1 if trigger == 'base' else 0

        now = datetime.now()
        hour = now.hour
        dow = now.weekday()
        features['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        features['hour_cos'] = np.cos(2 * np.pi * hour / 24)
        features['dow_sin'] = np.sin(2 * np.pi * dow / 5)
        features['dow_cos'] = np.cos(2 * np.pi * dow / 5)

        return features


# ============================================================================
# MT5 TRADER
# ============================================================================

class MT5Trader:
    def __init__(self, trade_logger: TradeLogger = None):
        self.connected = False
        self.trade_logger = trade_logger
        self.position_tracker = PositionTracker()

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

    def get_pip_value(self, symbol: str) -> float:
        """Returns pip size: 0.01 for JPY, 0.0001 for others."""
        return 0.01 if 'JPY' in symbol else 0.0001

    def get_spread_pips(self, symbol: str) -> float:
        info = self.symbol_info(symbol)
        if info is None:
            return 999
        spread_points = info.spread
        point = info.point
        pip_size = self.get_pip_value(symbol)
        spread_price = spread_points * point
        return spread_price / pip_size

    def get_positions(self) -> list:
        positions = mt5.positions_get()
        if positions is None:
            return []
        return [p for p in positions if p.magic == MAGIC_NUMBER]

    def has_position(self, symbol: str) -> bool:
        return any(p.symbol == symbol for p in self.get_positions())

    def count_positions(self) -> int:
        return len(self.get_positions())

    def calculate_tp_sl_prices(self, symbol: str, direction: str, entry_price: float) -> tuple:
        """Calculate TP and SL prices based on pips."""
        pip_value = self.get_pip_value(symbol)

        if direction == 'buy':
            tp_price = entry_price + (TP_PIPS * pip_value)
            sl_price = entry_price - (SL_PIPS * pip_value)
        else:  # sell
            tp_price = entry_price - (TP_PIPS * pip_value)
            sl_price = entry_price + (SL_PIPS * pip_value)

        return tp_price, sl_price

    def calculate_lock_price(self, symbol: str, direction: str, entry_price: float) -> float:
        """Calculate locked profit price (+10 pips from entry)."""
        pip_value = self.get_pip_value(symbol)

        if direction == 'buy':
            return entry_price + (LOCK_PROFIT_PIPS * pip_value)
        else:  # sell
            return entry_price - (LOCK_PROFIT_PIPS * pip_value)

    def open_trade(self, symbol: str, direction: str, lots: float,
                   quality_score: float = 0.0) -> bool:
        info = self.symbol_info(symbol)
        if info is None:
            return False

        spread_pips = self.get_spread_pips(symbol)

        if direction == 'buy':
            entry_price = info.ask
            order_type = mt5.ORDER_TYPE_BUY
        else:
            entry_price = info.bid
            order_type = mt5.ORDER_TYPE_SELL

        tp_price, sl_price = self.calculate_tp_sl_prices(symbol, direction, entry_price)

        # Round prices to proper digits
        digits = info.digits
        tp_price = round(tp_price, digits)
        sl_price = round(sl_price, digits)

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lots,
            "type": order_type,
            "price": entry_price,
            "sl": sl_price,
            "tp": tp_price,
            "deviation": 10,
            "magic": MAGIC_NUMBER,
            "comment": f"COUNTER_{direction.upper()}",
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        for attempt in range(3):
            result = mt5.order_send(request)
            if result is None:
                error = mt5.last_error()
                log.error(f"COUNTER {direction.upper()} {symbol} failed: {error}, attempt {attempt+1}/3")
                time.sleep(2)
                continue
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                log.error(f"COUNTER {direction.upper()} {symbol} failed: retcode={result.retcode}, attempt {attempt+1}/3")
                time.sleep(2)
                continue

            log.info(f"COUNTER {direction.upper()} {symbol} | entry={entry_price:.5f} | TP={tp_price:.5f} | SL={sl_price:.5f} | spread={spread_pips:.1f}pip")

            # Track position
            self.position_tracker.add(symbol, entry_price, direction, result.order)

            if self.trade_logger:
                self.trade_logger.log_entry(
                    symbol=symbol, direction=direction, price=entry_price,
                    spread_pips=spread_pips, quality=quality_score,
                    tp=tp_price, sl=sl_price, ticket=result.order
                )
            return True

        return False

    def modify_sl(self, symbol: str, new_sl: float) -> bool:
        """Modify SL for existing position (to lock profit)."""
        positions = mt5.positions_get(symbol=symbol)
        if positions is None:
            return False

        for pos in positions:
            if pos.magic != MAGIC_NUMBER:
                continue

            info = self.symbol_info(symbol)
            digits = info.digits if info else 5
            new_sl = round(new_sl, digits)

            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "symbol": symbol,
                "position": pos.ticket,
                "sl": new_sl,
                "tp": pos.tp,
            }

            result = mt5.order_send(request)
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                log.info(f"LOCKED {symbol} | New SL={new_sl:.5f} (+{LOCK_PROFIT_PIPS} pips)")
                self.position_tracker.set_locked(symbol)
                return True
            else:
                error = mt5.last_error() if result is None else result.retcode
                log.error(f"Failed to modify SL for {symbol}: {error}")

        return False

    def check_and_lock_profit(self, symbol: str):
        """Check if position reached +10 pips, lock profit if so."""
        if self.position_tracker.is_locked(symbol):
            return  # Already locked

        pos_info = self.position_tracker.get(symbol)
        if not pos_info:
            return

        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            return

        entry_price = pos_info['entry_price']
        direction = pos_info['direction']
        pip_value = self.get_pip_value(symbol)

        # Calculate current profit in pips
        if direction == 'buy':
            current_price = tick.bid  # Exit price for buy
            profit_pips = (current_price - entry_price) / pip_value
        else:  # sell
            current_price = tick.ask  # Exit price for sell
            profit_pips = (entry_price - current_price) / pip_value

        # If reached +10 pips, lock profit by moving SL to +10
        if profit_pips >= LOCK_PROFIT_PIPS:
            lock_price = self.calculate_lock_price(symbol, direction, entry_price)
            self.modify_sl(symbol, lock_price)

    def sync_positions(self):
        """Sync position tracker with MT5 (handle closed positions)."""
        mt5_positions = {p.symbol for p in self.get_positions()}
        tracked = list(self.position_tracker.positions.keys())

        for symbol in tracked:
            if symbol not in mt5_positions:
                pos_info = self.position_tracker.get(symbol)
                if pos_info and self.trade_logger:
                    # Position was closed (by TP/SL)
                    # Try to get exit info from history
                    self.log_closed_position(symbol, pos_info)
                self.position_tracker.remove(symbol)

    def log_closed_position(self, symbol: str, pos_info: dict):
        """Log a position that was closed by TP/SL."""
        # Get recent deals to find the close
        from_time = datetime.now().timestamp() - 3600  # Last hour
        deals = mt5.history_deals_get(datetime.fromtimestamp(from_time), datetime.now(), group=f"*{symbol}*")

        if deals:
            for deal in reversed(deals):
                if deal.magic == MAGIC_NUMBER and deal.entry == mt5.DEAL_ENTRY_OUT:
                    entry_price = pos_info['entry_price']
                    exit_price = deal.price
                    direction = pos_info['direction']
                    pip_value = self.get_pip_value(symbol)

                    if direction == 'buy':
                        pips = (exit_price - entry_price) / pip_value
                    else:
                        pips = (entry_price - exit_price) / pip_value

                    reason = "TP hit" if pips > 0 else "SL hit"
                    if pos_info['locked'] and pips > 0:
                        reason = "Locked profit"

                    self.trade_logger.log_exit(
                        symbol=symbol, direction=direction, price=exit_price,
                        ticket=deal.order, pnl=deal.profit, reason=reason, pips=pips
                    )
                    return


# ============================================================================
# MAIN STRATEGY
# ============================================================================

class CounterTradeStrategy:
    def __init__(self):
        self.trade_logger = TradeLogger(LOG_DIR)
        self.trader = MT5Trader(trade_logger=self.trade_logger)
        self.mfc_reader = MFCDataReader(MFC_FILE_PATH)
        self.model_manager = QualityModelManager(MODEL_DIR)
        self.last_bar_time = None
        self.running = False

    def initialize(self) -> bool:
        log.info("=" * 60)
        log.info("COUNTER-TRADE STRATEGY (Capture Model Drawdown)")
        log.info("=" * 60)

        if not self.trader.connect():
            return False

        if not self.model_manager.load():
            return False

        if not MFC_FILE_PATH.exists():
            log.error(f"MFC file not found: {MFC_FILE_PATH}")
            return False

        log.info(f"TP: {TP_PIPS} pips | SL: {SL_PIPS} pips | Lock at: +{LOCK_PROFIT_PIPS} pips")
        log.info(f"Quality threshold: {QUALITY_THRESHOLD}")
        log.info(f"Max spread: {MAX_SPREAD_PIPS} pips")
        log.info(f"Magic number: {MAGIC_NUMBER}")
        log.info(f"Trade log: {self.trade_logger.trades_file}")
        log.info("=" * 60)

        return True

    def check_model_signal(self, pair: str, base: str, quote: str) -> tuple:
        """Check if model would signal. Returns (model_direction, quality) or (None, None)."""
        symbol = self.trader.get_symbol(pair)

        # Friday cutoff
        tick = mt5.symbol_info_tick(symbol)
        if tick:
            server_time = datetime.fromtimestamp(tick.time)
            if server_time.weekday() == 4 and server_time.hour >= FRIDAY_CUTOFF_HOUR:
                return None, None

        # Spread check (tighter for counter-trade)
        if self.trader.get_spread_pips(symbol) > MAX_SPREAD_PIPS:
            return None, None

        # Position limits
        if self.trader.count_positions() >= MAX_POSITIONS:
            return None, None

        if self.trader.has_position(symbol):
            return None, None

        # Get MFC data
        data = self.mfc_reader.read()
        if not data or base not in data or quote not in data:
            return None, None

        base_m5 = data[base].get('M5', [])
        quote_m5 = data[quote].get('M5', [])

        if len(base_m5) < 2 or len(quote_m5) < 2:
            return None, None

        base_mfc = base_m5[-2]
        quote_mfc = quote_m5[-2]

        # Check BASE triggers
        if base_mfc <= -MFC_EXTREME_THRESHOLD:
            features = self.mfc_reader.get_mfc_features(base, quote, 'buy', 'base')
            if features:
                quality = self.model_manager.predict_quality(features)
                if quality >= QUALITY_THRESHOLD:
                    return 'buy', quality  # Model would BUY

        if base_mfc >= MFC_EXTREME_THRESHOLD:
            features = self.mfc_reader.get_mfc_features(base, quote, 'sell', 'base')
            if features:
                quality = self.model_manager.predict_quality(features)
                if quality >= QUALITY_THRESHOLD:
                    return 'sell', quality  # Model would SELL

        # Check QUOTE triggers
        if quote_mfc <= -MFC_EXTREME_THRESHOLD:
            features = self.mfc_reader.get_mfc_features(base, quote, 'sell', 'quote')
            if features:
                quality = self.model_manager.predict_quality(features)
                if quality >= QUALITY_THRESHOLD:
                    return 'sell', quality  # Model would SELL

        if quote_mfc >= MFC_EXTREME_THRESHOLD:
            features = self.mfc_reader.get_mfc_features(base, quote, 'buy', 'quote')
            if features:
                quality = self.model_manager.predict_quality(features)
                if quality >= QUALITY_THRESHOLD:
                    return 'buy', quality  # Model would BUY

        return None, None

    def process_bar(self):
        log.info(f"[{self.last_bar_time.strftime('%Y-%m-%d %H:%M')}] Scanning...")

        # First, sync positions (detect TP/SL hits)
        self.trader.sync_positions()

        # Check for profit locking on existing positions
        for symbol in list(self.trader.position_tracker.positions.keys()):
            self.trader.check_and_lock_profit(symbol)

        # Look for new signals
        signals = 0
        for pair, base, quote in ALL_PAIRS:
            symbol = self.trader.get_symbol(pair)

            if not self.trader.has_position(symbol):
                model_direction, quality = self.check_model_signal(pair, base, quote)
                if model_direction:
                    # COUNTER: enter opposite direction
                    counter_direction = 'sell' if model_direction == 'buy' else 'buy'

                    log.info(f"{pair} Model={model_direction.upper()} q={quality:.1%} -> COUNTER {counter_direction.upper()}")

                    if self.trader.open_trade(symbol, counter_direction, LOT_SIZE, quality):
                        signals += 1

        if signals == 0:
            log.info("No signals")

        # Show positions
        positions = self.trader.get_positions()
        if positions:
            log.info(f"Open: {len(positions)} positions")
            for p in positions:
                locked = "LOCKED" if self.trader.position_tracker.is_locked(p.symbol) else ""
                log.info(f"  {p.symbol}: {'BUY' if p.type == 0 else 'SELL'} P/L={p.profit:.2f} {locked}")

    def wait_for_new_bar(self) -> bool:
        ref_symbol = self.trader.get_symbol(ALL_PAIRS[0][0])
        rates = mt5.copy_rates_from_pos(ref_symbol, mt5.TIMEFRAME_M5, 0, 1)
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
        log.info("Strategy started. Waiting for M5 bar signals...")

        while self.running:
            try:
                if self.wait_for_new_bar():
                    log.info(f"New M5 bar, waiting for MFC...")
                    if self.mfc_reader.wait_for_fresh_data(self.last_bar_time, timeout=120):
                        self.process_bar()

                # Check profit lock more frequently (every 5 sec)
                for symbol in list(self.trader.position_tracker.positions.keys()):
                    self.trader.check_and_lock_profit(symbol)

                # Sync positions (detect TP/SL hits)
                self.trader.sync_positions()

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
    strategy = CounterTradeStrategy()
    try:
        if strategy.initialize():
            strategy.run()
    except Exception as e:
        log.error(f"Fatal error: {e}")
    finally:
        strategy.shutdown()


if __name__ == "__main__":
    main()
