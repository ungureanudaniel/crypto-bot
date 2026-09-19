import sys
import os

from matplotlib import units
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging
from modules.logger_config import setup_logging, log_trade
from datetime import datetime, time
from typing import Dict, List, Optional, Tuple
from modules.data_feed import data_feed
from modules.strategy_tools import generate_trade_signal
from modules.regime_switcher import predict_regime, train_model
from modules.portfolio import (
    add_trade, load_portfolio, save_portfolio,
    get_cash, update_cash, get_positions, save_positions,
    get_futures_positions, open_futures_position, close_futures_position,
    get_portfolio_summary
)
from config_loader import get_binance_client, get_futures_client, get_pair_config, config
import json
import threading
import time as _time   # `time` is shadowed by datetime.time above
from modules.order_manager import SpotOrderManager
from modules import market_gate, trend_hold

PENDING_ORDERS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                   'pending_orders.json')
BREAKER_STATE_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                  'breaker_state.json')
MANUAL_STATE_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 'manual_state.json')

# ============================================================================
# SETUP LOGGING
# ============================================================================
try:
    from services.notifier import notifier
    setup_logging(verbose=True, notifier=notifier)
    print("✅ Logging initialized with Telegram support")
except ImportError:
    setup_logging(verbose=True)  # Fallback without Telegram
    print("⚠️ Notifier not available - Telegram logging disabled")

# Get logger for this module
logger = logging.getLogger(__name__)

# Try to import notifier
try:
    from services.notifier import notifier
    has_notifier = True
except ImportError:
    has_notifier = False
    logger.warning("⚠️ Notifier service not available")

# -------------------------------------------------------------------
# CONFIG LOADING
# -------------------------------------------------------------------
CONFIG = config.config

# -------------------------------------------------------------------
# PORTFOLIO POSITION HELPERS
# -------------------------------------------------------------------
def save_positions_to_file(positions: Dict):
    """Save open positions to portfolio.json"""
    try:
        print(f"🔍 DEBUG: save_positions_to_file called with {len(positions)} positions")  # Add this
        save_positions(positions)
        logger.debug(f"💾 Saved {len(positions)} positions to portfolio.json")
    except Exception as e:
        logger.error(f"❌ Failed to save positions: {e}")
        print(f"❌ DEBUG: Error: {e}")  # Add this

def load_positions_from_file() -> Dict:
    """Load open positions from portfolio.json"""
    try:
        print("🔍 DEBUG: load_positions_from_file called")
        positions = get_positions()
        logger.info(f"Loaded {len(positions)} positions from portfolio.json")
        return positions
    except Exception as e:
        logger.error(f"❌ Failed to load positions: {e}")
        print(f"❌ DEBUG: Error: {e}")
        return {}

# -------------------------------------------------------------------
# EXCHANGE DATA FETCHING
# -------------------------------------------------------------------
def get_usdt_balance(client) -> float:
    """Get USDT balance directly from exchange"""
    if not client:
        return 0
    try:
        account = client.get_account()
        for balance in account['balances']:
            if balance['asset'] == 'USDT':
                return float(balance['free'])
        return 0
    except Exception as e:
        logger.error(f"Error fetching USDT balance: {e}")
        return 0

def get_usdc_balance(client) -> float:
    """Get USDC balance directly from exchange"""
    if not client:
        return 0
    try:
        account = client.get_account()
        for balance in account['balances']:
            if balance['asset'] == 'USDC':
                return float(balance['free'])
        return 0
    except Exception as e:
        logger.error(f"Error fetching USDC balance: {e}")
        return 0

def get_asset_balance(client, asset: str) -> float:
    """Get balance for specific asset from exchange"""
    if not client:
        return 0
    try:
        account = client.get_account()
        for balance in account['balances']:
            if balance['asset'] == asset:
                return float(balance['free'])
        return 0
    except Exception as e:
        logger.error(f"Error fetching {asset} balance: {e}")
        return 0

def get_all_balances(client) -> Dict[str, float]:
    """Get all non-zero balances from exchange"""
    if not client:
        return {}
    try:
        account = client.get_account()
        balances = {}
        for balance in account['balances']:
            free = float(balance['free'])
            if free > 0:
                balances[balance['asset']] = free
        return balances
    except Exception as e:
        logger.error(f"Error fetching all balances: {e}")
        return {}

def get_current_price(client, symbol: str) -> Optional[float]:
    """Get current price for a symbol from exchange"""
    if not client:
        return None
    try:
        binance_symbol = symbol.replace('/', '')
        ticker = client.get_symbol_ticker(symbol=binance_symbol)
        return float(ticker['price'])
    except Exception as e:
        logger.debug(f"Error getting price for {symbol}: {e}")
        return None

# -------------------------------------------------------------------
# TRADING ENGINE
# -------------------------------------------------------------------
class TradingEngine:
    """Universal trading engine - exchange is source of truth"""
    
    def __init__(self):
        self.config = CONFIG
        self.data_feed = data_feed
        self.trading_mode = self.config.get('trading_mode', 'paper').lower()
        self.symbols = self.config.get('coins', ['BTC/USDC', 'ETH/USDC'])
        self.timeframe = self.config.get('trading_timeframe', '15m')
        self.max_positions = self.config.get('max_positions', 3)
        self.risk_per_trade = float(self.config.get('risk_per_trade', 0.02))
        self.circuit_breaker_triggered = False
        self.last_trade_time_per_pair = {}
        self.circuit_breaker_time = None
        self.last_trade_time = 0

        # Load open positions from file on startup
        try:
            self.open_positions = get_positions()
            logger.info(f"📂 Loaded {len(self.open_positions)} positions from portfolio.json")
            
            # Log first position to verify data
            if self.open_positions:
                first_symbol = list(self.open_positions.keys())[0]
                first_pos = self.open_positions[first_symbol]
                logger.info(f"   Sample: {first_symbol} - {first_pos.get('side')} @ ${first_pos.get('entry_price')}")
        except Exception as e:
            logger.warning(f"⚠️ Could not load positions: {e}")
            self.open_positions = {}
        
        # Initialize real trading client if needed
        self.binance_client = get_binance_client()
        if self.trading_mode in ('live', 'testnet') and self.binance_client:
            self.symbols = self._tradable_symbols(self.symbols)

        # Initialize futures client for short execution
        if self.trading_mode == 'paper' or not self.shorts_enabled:
            self.futures_client = None
            logger.info("📄 Futures client skipped (paper mode or shorts disabled)")
        else:
            try:
                self.futures_client = get_futures_client()
            except Exception as e:
                logger.warning(f"⚠️ Futures client not available: {e}")
                self.futures_client = None


        # Load open futures positions
        try:
            self.open_futures_positions = get_futures_positions()
            logger.info(f"📂 Loaded {len(self.open_futures_positions)} futures positions")
        except Exception as e:
            logger.warning(f"⚠️ Could not load futures positions: {e}")
            self.open_futures_positions = {}

        # Futures engine for short execution
        if self.trading_mode == 'paper' or not self.shorts_enabled:
            self.futures_engine = None
        else:
            try:
                from modules.futures_engine import futures_engine as _fe
                self.futures_engine = _fe
            except Exception as e:
                logger.warning(f"⚠️ FuturesEngine not available: {e}")
                self.futures_engine = None


        # Track pending signals to avoid duplicates
        self.last_signals = {}
        
        # Initial balance for return calculation (from first sync)
        self.initial_total_value = None

        # Load per-pair trade timestamps
        try:
            portfolio = load_portfolio()
            self.last_trade_time_per_pair = portfolio.get('last_trade_time_per_pair', {})
        except:
            self.last_trade_time_per_pair = {}
        
        # ---- Circuit breaker state (peak equity is persisted across restarts) ----
        self._equity_peak = None
        self._equity_cache = (0.0, None)
        self.paused = False            # /pause: no new automatic entries; stops and exits keep working
        self._load_breaker_state()
        self._manual_baseline = None   # symbol -> qty of untracked holdings we deliberately leave alone
        self._manual_checked_at = 0.0
        self._load_manual_state()

        # ---- Limit-order execution (spot longs, live/testnet only) ----
        # Set "use_limit_orders": false in config.json to fall back to plain market orders.
        self._lock = threading.RLock()
        self.pending_entries = {}   # symbol -> entry order state (limit buy being repriced)
        self.pending_exits = {}     # symbol -> {'st': exit order state, 'reason': str}
        self.order_manager = None
        if (self.trading_mode in ('live', 'testnet') and self.binance_client
                and self.config.get('use_limit_orders', True)):
            self.order_manager = SpotOrderManager(self.binance_client, self.config)
            self.stop_market_fallback_pct = float(self.config.get('stop_market_fallback_pct', 0.006))
            self.stop_trigger_timeout = float(self.config.get('stop_trigger_timeout_seconds', 120))
            self.oco_ratchet_min_pct = float(self.config.get('oco_ratchet_min_pct', 0.002))
            self._load_pending()
            logger.info("🧾 Limit-order execution enabled (entries, exits, OCO stops with market fallback)")

        logger.info(f"Trading Engine initialized for {self.trading_mode.upper()} mode")
        logger.info(f"Monitoring {len(self.symbols)} symbols on {self.timeframe}")

    def get_pair_risk(self, symbol: str) -> float:
        """Get risk per trade for a specific pair, falling back to global."""
        try:
            pair_cfg = get_pair_config(symbol)
            return float(pair_cfg.get('risk_per_trade', self.risk_per_trade))
        except Exception:
            return self.risk_per_trade
    
    def run_iteration(self):
        """Main loop called by scheduler"""
        if not self.check_drawdown():
            return

        # 1. Manage Exits first (Always clear the desk before taking new orders)
        self.check_stop_losses()

        # 2. Scan for new signals
        for symbol in self.symbols:
            self.process_pair(symbol)

    def process_pair(self, symbol: str):
        """Evaluates a single pair for entry with Regime-based filtering"""
        # Load fresh data
        df = self.data_feed.get_ohlcv(symbol, self.timeframe)
        if df is None or df.empty:
            return

        # 1. Get Regime String (e.g., "True Trend UPTREND (85% confidence)")
        # Point 3: Regime-Based Execution
        regime_str = predict_regime(df)
        logger.info(f"🔍 {symbol} Regime: {regime_str}")
        
        # --- POINT 3: REGIME-BASED EXECUTION RULES ---
        
        # Rule A: Block all new entries during 'Expansion' (Volatile Chop)
        if "Expansion" in regime_str:
            logger.info(f"🚫 Skipping {symbol}: Volatile Chop detected.")
            return

        # Rule B: Adjust risk based on Trend Direction
        base_pair_risk = self.get_pair_risk(symbol)
        current_risk = base_pair_risk  # start from per-pair risk
        if "DOWNTREND" in regime_str:
            current_risk *= 0.75  # Reduce risk by 25% in downtrends
            logger.debug(f"📉 Downtrend risk reduction: {current_risk:.2%}")
        
        # Rule C:Optional - Filter by Confidence
        #  parse the confidence to be extra strict:
        # if "confidence" in regime_str and int(regime_str.split('(')[1][:2]) < 70:
        #     logger.info(f"⚠️ Low confidence regime ({regime_str}). Skipping.")
        #     return
        # ---------------------------------------------

        # 2. Generate Signal (Pass the ADJUSTED risk)
        # Note: We pass the full regime_str so the strategy can log it
        signal_data = generate_trade_signal(
            df, 
            self.get_cash_balance("USDC"), 
            current_risk, 
            symbol, 
            self, 
            regime=regime_str 
        )
        if signal_data is None:
            return
        signal = signal_data.get('side')  # 'long' or 'short'
        if signal not in ['long', 'short']:

            # 3. Position Sizing (Using the adjusted risk)
            units = signal_data.get('units', 0)
            if units <= 0:
                return
            
            entry_price = signal_data.get('entry_price', df['close'].iloc[-1])
            if signal == 'long':
                self.open_position(
                    symbol=symbol,
                    side='long',
                    entry_price=entry_price,
                    units=units,
                    stop_loss=signal_data['stop_loss'],
                    take_profit=signal_data['take_profit'],
                    signal_type=signal_data.get('signal_type', 'unknown'),
                    atr=signal_data.get('atr', 0.0)
                )
            elif signal == 'short':
                if self.futures_engine:
                    self.futures_engine.open_short(
                        symbol=symbol,
                        amount=units,
                        entry_price=entry_price,
                        stop_loss=signal_data['stop_loss'],
                        take_profit=signal_data['take_profit'],
                        signal_type=signal_data.get('signal_type', 'unknown'),
                        atr=signal_data.get('atr', 0.0)
                    )
            
    def calculate_position_size(self, symbol, price, stop_loss, risk_override=None):
        """Risk-based sizing using current regime risk"""
        cash = self.get_cash_balance("USDC")
        
        # Use the overridden risk if provided, otherwise default to config
        risk_pct = risk_override if risk_override is not None else self.risk_per_trade
        risk_amt = cash * risk_pct
        
        if stop_loss and stop_loss != price:
            risk_per_unit = abs(price - stop_loss)
            units = risk_amt / risk_per_unit
        else:
            # Fallback to 10% of cash if no SL provided
            units = (cash * 0.1) / price
            
        return units

    # ------------------------------------------------------------------
    # CIRCUIT BREAKER
    #
    # Drawdown = (peak equity - current equity) / peak equity, where equity is read from the
    # EXCHANGE in live/testnet (stable coins + the coins we trade, at market price) and from
    # portfolio.json only in paper mode. The peak is persisted, so restarts don't reset it.
    #
    # Trips when drawdown > max_drawdown (default 5%). Resets when either
    #   - drawdown recovers below max_drawdown * circuit_breaker_reset_ratio (default 0.8), or
    #   - circuit_breaker_cooldown_hours (default 48) have passed; that timed reset REBASES the
    #     peak to the current equity, otherwise the same loss would trip it again immediately.
    # It only blocks NEW entries. Stops, OCOs and exits never depend on it.
    # ------------------------------------------------------------------
    STABLE_COINS = {'USDC', 'USDT', 'FDUSD', 'BUSD', 'TUSD', 'USDP', 'DAI'}

    def _exchange_equity(self) -> float:
        tracked = self.STABLE_COINS | {s.split('/')[0] for s in self.symbols if '/' in s}
        prices = {t['symbol']: float(t['price']) for t in self.binance_client.get_all_tickers()}
        total = 0.0
        for b in self.binance_client.get_account()['balances']:
            asset = b['asset']
            if asset not in tracked:
                continue
            amount = float(b['free']) + float(b['locked'])
            if amount <= 0:
                continue
            if asset in self.STABLE_COINS:
                total += amount
                continue
            for quote in ('USDC', 'USDT'):
                price = prices.get(asset + quote)
                if price:
                    total += amount * price
                    break
        return total

    def _current_equity(self) -> Optional[float]:
        """Account equity, cached for 30s. None if it can't be measured right now."""
        now = _time.time()
        cached_at, cached = self._equity_cache
        if cached is not None and now - cached_at < 30:
            return cached
        try:
            if self.trading_mode in ('live', 'testnet') and self.binance_client:
                value = self._exchange_equity()
            else:
                value = float(get_portfolio_summary()['total_value'])
        except Exception as e:
            logger.warning(f"⚠️ Could not measure equity for the circuit breaker: {e}")
            return None
        self._equity_cache = (now, value)
        return value

    def _save_breaker_state(self):
        try:
            with open(BREAKER_STATE_FILE, 'w') as f:
                json.dump({'peak': self._equity_peak,
                           'triggered': self.circuit_breaker_triggered,
                           'triggered_at': self.circuit_breaker_time,
                           'paused': getattr(self, 'paused', False)}, f)
        except Exception as e:
            logger.error(f"❌ Could not save circuit breaker state: {e}")

    def _load_breaker_state(self):
        try:
            if os.path.exists(BREAKER_STATE_FILE):
                with open(BREAKER_STATE_FILE) as f:
                    s = json.load(f)
                self._equity_peak = s.get('peak')
                self.circuit_breaker_triggered = bool(s.get('triggered', False))
                self.circuit_breaker_time = s.get('triggered_at')
                self.paused = bool(s.get('paused', False))
                if self.paused:
                    logger.warning("⏸️ Trading was PAUSED at last shutdown - still paused (/resume to continue)")
                if self.circuit_breaker_triggered:
                    logger.warning("🚨 Circuit breaker was ACTIVE at last shutdown - restored")
        except Exception as e:
            logger.warning(f"⚠️ Could not load circuit breaker state: {e}")

    def reset_circuit_breaker(self, rebase: bool = True):
        """Manual reset (Telegram /resetcircuitbreaker). Rebases the peak so it doesn't re-trip."""
        self.circuit_breaker_triggered = False
        self.circuit_breaker_time = None
        if rebase:
            equity = self._current_equity()
            if equity:
                self._equity_peak = equity
        self._save_breaker_state()
        logger.info("✅ Circuit breaker manually reset")

    def check_drawdown(self) -> bool:
        """Returns True if new entries are allowed, False while the circuit breaker is active."""
        equity = self._current_equity()
        if equity is None:
            return not self.circuit_breaker_triggered     # can't measure: keep the current state
        if equity <= 0 and not self._equity_peak:
            return not self.circuit_breaker_triggered     # empty/unfunded account: nothing to protect

        now = _time.time()
        max_drawdown = float(self.config.get('max_drawdown', 0.05))
        reset_ratio = float(self.config.get('circuit_breaker_reset_ratio', 0.8))
        cooldown = float(self.config.get('circuit_breaker_cooldown_hours', 48)) * 3600

        if self._equity_peak is None or equity > self._equity_peak:
            self._equity_peak = equity
            self._save_breaker_state()
        drawdown = (self._equity_peak - equity) / self._equity_peak if self._equity_peak else 0.0

        if not self.circuit_breaker_triggered:
            if drawdown > max_drawdown:
                self.circuit_breaker_triggered = True
                self.circuit_breaker_time = now
                self._save_breaker_state()
                logger.warning(f"🚨 Circuit breaker triggered – drawdown {drawdown:.1%} > {max_drawdown:.1%} "
                               f"(peak ${self._equity_peak:,.2f}, now ${equity:,.2f}). New entries paused; "
                               f"stops and exits stay active.")
                self._notify(f"🚨 <b>Circuit breaker</b>: drawdown {drawdown:.1%} "
                             f"(peak ${self._equity_peak:,.2f} → ${equity:,.2f}). New entries paused.")
                return False
            return True

        # ---- breaker is active: decide whether to reset ----
        since = now - (self.circuit_breaker_time or now)
        if drawdown < max_drawdown * reset_ratio:
            reason = f"drawdown recovered to {drawdown:.1%}"
        elif since >= cooldown:
            self._equity_peak = equity                    # rebase, or it re-trips on the same loss
            reason = f"{since / 3600:.0f}h cooldown elapsed (peak rebased to ${equity:,.2f})"
        else:
            return False

        self.circuit_breaker_triggered = False
        self.circuit_breaker_time = None
        self._save_breaker_state()
        logger.info(f"✅ Circuit breaker reset – {reason}")
        self._notify(f"✅ <b>Circuit breaker reset</b> – {reason}")
        return True

    def get_cash_balance(self, quote_currency: str = "USDC") -> float:
        """Get balance for specific quote currency"""
        
        # PAPER MODE
        if self.trading_mode == 'paper':
            try:
                cash_dict = get_cash()
                return cash_dict.get(quote_currency, 0)
            except Exception as e:
                logger.error(f"Error getting paper {quote_currency} balance: {e}")
                return 100 if quote_currency == "USDT" else 0
        
        # LIVE/TESTNET MODE
        if not self.binance_client:
            logger.error("No binance client available")
            return 0
        
        try:
            account = self.binance_client.get_account()
            for balance in account['balances']:
                if balance['asset'] == quote_currency:
                    return float(balance['free'])
            return 0
        except Exception as e:
            logger.error(f"Error fetching {quote_currency} balance: {e}")
            return 0

    def get_current_prices(self) -> Dict[str, float]:
        """Get current prices for all symbols"""
        prices = {}
        
        for symbol in self.symbols:
            if self.binance_client:
                price = get_current_price(self.binance_client, symbol)
            else:
                price = self.data_feed.get_price(symbol)
            
            if price and price > 0:
                prices[symbol] = price
        
        return prices
    
    def check_stop_losses(self) -> bool:
        """Fixed: Ensures state is saved after every evaluation"""
        with self._lock:
            positions_closed = False
            if not self.open_positions:
                return False

            current_prices = self.get_current_prices()
            from modules.exit_manager import evaluate_exit

            for symbol, position in list(self.open_positions.items()):
                if symbol in self.pending_exits:
                    continue                      # already being sold
                price = current_prices.get(symbol)
                if not price: continue

                # Get latest OHLCV for indicators used in exit_manager (like trailing ATR)
                df = self.data_feed.get_ohlcv(symbol, self.timeframe, limit=self._exit_history_limit(position))

                should_exit, reason = evaluate_exit(symbol, position, price, df)

                if should_exit:
                    if self.order_manager and position.get('side') == 'long':
                        self._handle_exit_signal(symbol, position, price, reason)
                    else:
                        self.close_position(symbol, price, reason)
                    positions_closed = True

                # Persist any changes (like a raised trailing SL) immediately, so a crash
                # can't lose them. manage_orders() ratchets the exchange OCO to this stop.
                if symbol in self.open_positions:
                    self.open_positions[symbol] = position
                    save_positions_to_file(self.open_positions)

            return positions_closed
    
    def close_position(self, symbol: str, exit_price: float, reason: str) -> bool:
        """Close an existing position and execute sell on exchange"""
        if symbol not in self.open_positions:
            logger.warning(f"⚠️ No position found for {symbol}")
            return False

        position      = self.open_positions[symbol]
        amount        = position['amount']
        entry_price   = position['entry_price']
        quote_currency = position.get('quote_currency', 'USDC')

        if position['side'] == 'long':
            pnl     = (exit_price - entry_price) * amount
            pnl_pct = (exit_price / entry_price - 1) * 100
        else:
            pnl     = (entry_price - exit_price) * amount
            pnl_pct = (1 - exit_price / entry_price) * 100

        if self.order_manager and position['side'] == 'long':
            # Non-blocking: cancels the OCO, sells with limit orders (market fallback).
            # The trade is recorded by _finalize_close() once the sell has filled.
            return self._begin_exit(symbol, reason)

        if self.trading_mode in ['live', 'testnet'] and self.binance_client:
            try:
                binance_symbol = symbol.replace('/', '')
                if position['side'] == 'long':
                    order = self.binance_client.order_market_sell(
                        symbol=binance_symbol,
                        quantity=round(amount, 6)
                    )
                    logger.info(f"📤 Live SELL executed: {order['orderId']}")
                # shorts close via futures_engine, not here
            except Exception as e:
                logger.error(f"❌ Failed to execute live sell: {e}")
                return False
        else:
            logger.warning(f"⚠️ Cannot close position — no exchange client or unsupported mode: {self.trading_mode}")
            return False

        return self._finalize_close(symbol, exit_price, reason)

    def _finalize_close(self, symbol: str, exit_price: float, reason: str,
                        qty: Optional[float] = None) -> bool:
        """Record a completed close (position already sold on the exchange)."""
        if symbol not in self.open_positions:
            return False
        position       = self.open_positions[symbol]
        amount         = qty if qty else position['amount']
        entry_price    = position['entry_price']
        quote_currency = position.get('quote_currency', 'USDC')

        if position['side'] == 'long':
            pnl     = (exit_price - entry_price) * amount
            pnl_pct = (exit_price / entry_price - 1) * 100
        else:
            pnl     = (entry_price - exit_price) * amount
            pnl_pct = (1 - exit_price / entry_price) * 100

        del self.open_positions[symbol]
        save_positions_to_file(self.open_positions)

        add_trade({
            'symbol':       symbol,
            'action':       'close',
            'side':         position['side'],
            'amount':       amount,
            'entry_price':  entry_price,
            'exit_price':   exit_price,
            'pnl':          pnl,
            'pnl_pct':      pnl_pct,
            'reason':       reason,
            'mode':         self.trading_mode,
            'signal_type':  position.get('signal_type', 'unknown'),
            'quote_currency': quote_currency,
        })

        log_trade('close',
                symbol=symbol,
                side=position['side'],
                entry=entry_price,
                exit=exit_price,
                pnl=pnl,
                pnl_pct=pnl_pct,
                reason=reason)

        if has_notifier:
            try:
                notifier.send_message_sync(
                    f"{'✅' if pnl > 0 else '❌'} <b>TRADE CLOSED</b> ({reason}) [{self.trading_mode.upper()}]\n"
                    f"📊 <b>{symbol}</b>\n"
                    f"💰 Side: {position['side'].upper()}\n"
                    f"💵 Exit: <code>${exit_price:.4f}</code>\n"
                    f"📈 PnL: <code>${pnl:+.4f}</code> ({pnl_pct:+.2f}%)"
                )
            except Exception:
                pass

        logger.info(f"✅ Closed {symbol}: PnL ${pnl:.2f} ({pnl_pct:+.1f}%) | Reason: {reason}")
        return True

    def open_position(self, symbol: str, side: str, entry_price: float,
                 units: float, stop_loss: float, take_profit: float,
                 signal_type: str = '', skip_exchange: bool = False, **kwargs) -> bool:
        """Open a new position with stop loss and take profit"""
        logger.info(f"🔍 OPEN POSITION ATTEMPT:")
        logger.info(f"   Symbol: {symbol}")
        logger.info(f"   Side: {side}")
        logger.info(f"   Entry: ${entry_price:.2f}")
        logger.info(f"   Units: {units:.6f}")
        logger.info(f"   Value: ${units * entry_price:.2f}")
        logger.info(f"   skip_exchange: {skip_exchange}")

        if units * entry_price < 5:
            logger.warning(f"⚠️ Order value ${units * entry_price:.2f} below minimum $5")
            return False

        if units <= 0:
            logger.warning(f"⚠️ Invalid units: {units}")
            return False

        # SAFETY: live trend_hold entries are opt-in until they have run on testnet. (Protection
        # is a stop-only order, ratcheted like the OCO stop; the BTC gate is in scan_and_trade.)
        if (str(signal_type).startswith('trend_hold') and self.trading_mode in ('live', 'testnet')
                and not self.config.get('trend_hold_live_enabled', False)):
            logger.error(f"🛑 {symbol}: live trend_hold entry blocked - trend_hold_live_enabled is false "
                         f"(validate on testnet first, then set it to true)")
            return False

        if entry_price <= 0:
            logger.warning(f"⚠️ Invalid entry price: {entry_price}")
            return False

        if symbol in self.open_positions or symbol in self.pending_entries:
            logger.info(f"⏭️ Already in position (or entry order pending) for {symbol}")
            return False

        if len(self.open_positions) + len(self.pending_entries) >= self.max_positions:
            logger.info(f"⏭️ At max positions ({self.max_positions})")
            return False

        quote_currency = symbol.split('/')[1]
        execution_success = False

        if self.trading_mode in ['live', 'testnet']:

            if skip_exchange:
                logger.info(f"📋 Registering existing exchange position for {symbol}")
                execution_success = True

            elif self.binance_client:
                try:
                    if side == 'long':
                        usdc_balance = get_asset_balance(self.binance_client, quote_currency)
                        cost = units * entry_price
                        logger.info(f"   USDC balance: ${usdc_balance:.2f}")
                        logger.info(f"   Required: ${cost:.2f}")

                        if usdc_balance < cost:
                            logger.error(f"❌ Insufficient USDC: have ${usdc_balance:.2f}, need ${cost:.2f}")
                            return False

                        is_valid, adjusted_units, error = self.validate_and_adjust_order(symbol, units)
                        if not is_valid:
                            logger.error(f"❌ Order validation failed: {error}")
                            return False
                        if adjusted_units != units:
                            logger.info(f"🔄 Quantity adjusted: {units} → {adjusted_units}")
                            units = adjusted_units

                        if self.order_manager:
                            # Limit entry: the position is registered (and its OCO stop
                            # placed) by manage_orders() once the order fills.
                            return self._submit_limit_entry(symbol, entry_price, units, stop_loss,
                                                            take_profit, signal_type,
                                                            kwargs.get('atr', 0.0))

                        order = self.binance_client.order_market_buy(
                            symbol=symbol.replace('/', ''),
                            quantity=round(units, 6)
                        )
                        logger.info(f"✅ Live BUY order executed: {order['orderId']}")
                        execution_success = True

                    elif side == 'short':
                        logger.error("❌ Use futures_engine for shorts")
                        return False

                except Exception as e:
                    logger.error(f"❌ Failed to execute live order: {e}")
                    return False
            else:
                logger.warning("⚠️ No Binance client available")
                return False

        else:
            logger.warning(f"⚠️ Unsupported trading mode: {self.trading_mode}")
            return False

        if execution_success:
            self.open_positions[symbol] = self._build_position(
                symbol, side, units, entry_price, stop_loss, take_profit,
                signal_type, kwargs.get('atr', 0.0))

            log_trade('open', symbol=symbol, side=side, entry=entry_price,
                    units=units, stop_loss=stop_loss, take_profit=take_profit)

            save_positions_to_file(self.open_positions)

            add_trade({
                'symbol':         symbol,
                'action':         'open',
                'side':           side,
                'amount':         units,
                'price':          entry_price,
                'stop_loss':      stop_loss,
                'take_profit':    take_profit,
                'mode':           self.trading_mode,
                'signal_type':    signal_type if signal_type else 'unknown',
                'quote_currency': quote_currency,
            })

            if has_notifier:
                try:
                    notifier.send_message_sync(
                        f"{'📈' if side == 'long' else '📉'} <b>TRADE OPENED</b> [{self.trading_mode.upper()}]\n"
                        f"📊 <b>{symbol}</b>\n"
                        f"💰 Side: {side.upper()}\n"
                        f"💵 Entry: <code>${entry_price:.4f}</code>\n"
                        f"📦 Units: <code>{units:.6f}</code>\n"
                        f"🛑 Stop: <code>${stop_loss:.4f}</code>\n"
                        f"🎯 Target: <code>${take_profit:.4f}</code>"
                    )
                except Exception:
                    pass

            logger.info(f"✅ Opened {side.upper()} position: {units:.6f} {symbol} at ${entry_price:.2f}")
            return True

        return False

    # ------------------------------------------------------------------
    # POSITION BUILDER (shared by market and limit-order paths)
    # ------------------------------------------------------------------
    def _build_position(self, symbol: str, side: str, amount: float, entry_price: float,
                        stop_loss: float, take_profit: float, signal_type: str,
                        atr: float = 0.0) -> Dict:
        pair_cfg = get_pair_config(symbol)
        return {
            'side':                    side,
            'amount':                  amount,
            'entry_price':             entry_price,
            'current_price':           entry_price,
            'stop_loss':               stop_loss,
            'take_profit':             take_profit,
            'quote_currency':          symbol.split('/')[1],
            'value':                   amount * entry_price,
            'pnl':                     0.0,
            'pnl_pct':                 0.0,
            'signal_type':             signal_type if signal_type else 'unknown',
            'atr':                     atr,
            'trailing_stop_active':    False,
            'entry_time':              datetime.now().isoformat(),
            'mode':                    self.trading_mode,
            'candles_held':            0,
            'last_candle_time':        None,
            'trailing_min_pct':        pair_cfg.get('trailing_min_pct', 0.04),
            'trailing_max_pct':        pair_cfg.get('trailing_max_pct', 0.08),
            'trailing_activation_pct': 0.15,
        }

    # ------------------------------------------------------------------
    # LIMIT-ORDER EXECUTION (spot longs, live / testnet)
    #
    # entry : limit buy, repriced toward the ask, cancelled if unfilled
    # stop  : exchange-side OCO (take-profit limit + stop-limit). The stop is ratcheted by
    #         cancel + re-place when the trailing logic raises it. If price runs through
    #         the stop-limit (or sits below the stop too long) the OCO is cancelled and the
    #         position is sold at MARKET.
    # exits : limit sell ask -> mid -> bid, then MARKET fallback (indicator exits, manual)
    #
    # manage_orders() is called every minute by the scheduler and advances all of it.
    # ------------------------------------------------------------------
    URGENT_EXITS = ('stop_loss', 'trailing_stop', 'emergency_sell', 'stop_loss_market')

    def _save_pending(self):
        try:
            with open(PENDING_ORDERS_FILE, 'w') as f:
                json.dump({'entries': self.pending_entries, 'exits': self.pending_exits}, f)
        except Exception as e:
            logger.error(f"❌ Could not save pending orders: {e}")

    def _load_pending(self):
        try:
            if os.path.exists(PENDING_ORDERS_FILE):
                with open(PENDING_ORDERS_FILE) as f:
                    data = json.load(f)
                self.pending_entries = data.get('entries', {})
                self.pending_exits = data.get('exits', {})
                logger.info(f"📂 Restored {len(self.pending_entries)} pending entries, "
                            f"{len(self.pending_exits)} pending exits")
        except Exception as e:
            logger.warning(f"⚠️ Could not load pending orders: {e}")

    # ------------------------------------------------------------------
    # MANUAL TRADES: adopt coins you bought yourself, protect + trail them, notice when you sell.
    #
    # Holdings that already exist when the bot first starts are LISTED but left alone (they may be
    # long-term coins you never want stopped out) - use /adopt SYMBOL to protect one. Anything that
    # APPEARS afterwards (a manual buy) is adopted automatically. Only coins in "coins" are watched.
    # ------------------------------------------------------------------
    def _save_manual_state(self):
        try:
            with open(MANUAL_STATE_FILE, 'w') as f:
                json.dump({'baseline': getattr(self, '_manual_baseline', None) or {}}, f)
        except Exception as e:
            logger.error(f"❌ Could not save manual state: {e}")

    def _load_manual_state(self):
        try:
            if os.path.exists(MANUAL_STATE_FILE):
                with open(MANUAL_STATE_FILE) as f:
                    self._manual_baseline = dict(json.load(f).get('baseline', {}))
        except Exception as e:
            logger.warning(f"⚠️ Could not load manual state: {e}")

    def _balances_total(self) -> Dict[str, float]:
        """asset -> free + locked (coins locked in our own stop orders still count as held)."""
        acct = self.binance_client.get_account()
        out = {}
        for b in acct['balances']:
            total = float(b['free']) + float(b['locked'])
            if total > 0:
                out[b['asset']] = total
        return out

    def _manual_stop(self, symbol: str, price: float):
        """(stop, atr) for an adopted/manual position. Measured from the CURRENT price, not the entry,
        so an already-losing position is not sold on the spot; 2 x daily ATR, clamped to 3-12%."""
        lo = float(self.config.get('manual_stop_min_pct', 0.03))
        hi = float(self.config.get('manual_stop_max_pct', 0.12))
        dist, atr = float(self.config.get('manual_stop_pct', 0.06)), 0.0
        try:
            df = self.data_feed.get_ohlcv(symbol=symbol, interval=self.timeframe,
                                          limit=self._exit_history_limit({'signal_type': 'manual'}))
            atr = trend_hold.daily_atr(df, self.config)
            if atr > 0:
                dist = min(hi, max(lo, trend_hold.params(self.config)['stop_atr_mult'] * atr / price))
        except Exception as e:
            logger.debug(f"manual stop: no ATR for {symbol}, using {dist:.0%}: {e}")
        return price * (1 - dist), atr

    def _estimate_entry(self, symbol: str, qty: float) -> Optional[float]:
        """Volume-weighted price of the most recent buys that add up to `qty` (from the trade history)."""
        try:
            trades = self.binance_client.get_my_trades(symbol=symbol.replace('/', ''), limit=200)
        except Exception as e:
            logger.debug(f"entry estimate for {symbol}: {e}")
            return None
        need, got, cost = qty, 0.0, 0.0
        for t in reversed(trades):
            if not t.get('isBuyer'):
                continue
            take = min(float(t['qty']), need - got)
            cost += take * float(t['price'])
            got += take
            if got >= need * 0.999:
                break
        return cost / got if got >= need * 0.5 and got > 0 else None

    def _adopt(self, symbol: str, qty: float, source: str = 'auto') -> bool:
        """Take `qty` coins under the bot's care: record the position and place the exchange-side stop."""
        om = self.order_manager
        price = om.last_price(symbol)
        entry = self._estimate_entry(symbol, qty) or price
        existing = self.open_positions.get(symbol)

        if existing:                                       # more coins bought on top of a managed position
            old = existing['amount']
            total = old + qty
            existing['entry_price'] = (existing['entry_price'] * old + entry * qty) / total
            existing['amount'] = total
            existing['value'] = total * price
            prot = existing.get('oco')
            if prot:
                res = om.cancel_protection(symbol, prot)
                if res == 'done':
                    return True                            # it just filled - handled by the normal path
                existing['oco'] = None
            save_positions_to_file(self.open_positions)
            self._place_protection(symbol)
            self._notify(f"➕ Added {qty:.6g} to your managed <b>{symbol}</b> (now {total:.6g}). "
                         f"Stop re-placed at ${existing['stop_loss']:.4f}.")
            return True

        stop, atr = self._manual_stop(symbol, price)
        pos = self._build_position(symbol, 'long', qty, entry, stop, 0.0, 'manual_adopted', atr)
        pos['oco'] = None
        pos['initial_stop'] = stop
        self.open_positions[symbol] = pos
        save_positions_to_file(self.open_positions)
        add_trade({'symbol': symbol, 'action': 'open', 'side': 'long', 'amount': qty, 'price': entry,
                   'stop_loss': stop, 'take_profit': 0.0, 'mode': self.trading_mode,
                   'signal_type': 'manual_adopted', 'quote_currency': symbol.split('/')[1]})
        pnl_pct = (price / entry - 1) * 100
        logger.info(f"🛡️ Adopted {symbol}: {qty} @ {entry:.6f} (now {price:.6f}), stop {stop:.6f} [{source}]")
        self._notify(f"🛡️ <b>Adopted your {symbol}</b> ({source})\n"
                     f"{qty:.6g} coins, entry ~${entry:,.4f} ({pnl_pct:+.1f}% now)\n"
                     f"Stop ${stop:,.4f} ({(stop / price - 1) * 100:+.1f}%), trailing up as it rises. "
                     f"/setstop {symbol.split('/')[0]} PRICE to change it.")
        self._place_protection(symbol)
        return True

    def _reconcile_position(self, symbol: str, pos: Dict, held: float):
        """The exchange holds less than we track: the user sold (some of) it outside the bot."""
        om = self.order_manager
        amount = pos['amount']
        if held + 1e-12 >= amount * 0.98:
            return
        prot = pos.get('oco')
        if prot:                                           # our own stop may just have filled: let that path run
            st = om.protection_status(symbol, prot)
            if st['status'] == 'ALL_DONE' and st['filled']:
                return
        price = om.last_price(symbol)
        if held < om.min_qty(symbol) or held * price < 5:
            if prot:
                om.cancel_protection(symbol, prot)
            logger.info(f"📤 {symbol} was sold outside the bot - closing the record at ~{price}")
            self._finalize_close(symbol, price, 'manual_sell', qty=amount)
            return
        logger.info(f"✂️ {symbol}: you sold part of it ({amount:.6g} -> {held:.6g})")
        if prot:
            if om.cancel_protection(symbol, prot) == 'done':
                return
        pos['amount'] = held
        pos['oco'] = None
        save_positions_to_file(self.open_positions)
        self._place_protection(symbol)
        self._notify(f"✂️ You sold part of <b>{symbol}</b>. Tracking {held:.6g} coins now, stop re-placed.")

    def sync_manual_positions(self, force: bool = False):
        """Every ~2 min: reconcile tracked positions with the exchange and adopt new manual buys."""
        if not (self.order_manager and self.config.get('adopt_manual_positions', True)):
            return
        now = _time.time()
        if not force and now - getattr(self, '_manual_checked_at', 0.0) < 120:
            return
        self._manual_checked_at = now
        try:
            balances = self._balances_total()
        except Exception as e:
            logger.warning(f"⚠️ Could not read balances for manual-trade sync: {e}")
            return

        om = self.order_manager
        ignore = {a.upper() for a in self.config.get('adopt_ignore', [])}
        min_usd = float(self.config.get('adopt_min_usd', 15))
        first_run = getattr(self, '_manual_baseline', None) is None
        if first_run:
            self._manual_baseline = {}
        found_at_start = []

        for symbol in self.symbols:
            base = symbol.split('/')[0]
            if symbol in self.pending_entries or symbol in self.pending_exits:
                continue
            try:
                held = balances.get(base, 0.0)
                pos = self.open_positions.get(symbol)
                if pos and pos.get('side') == 'long':
                    self._reconcile_position(symbol, pos, held)
                    pos = self.open_positions.get(symbol)
                tracked = pos['amount'] if pos else 0.0
                untracked = max(0.0, held - tracked)
                baseline = self._manual_baseline.get(symbol, 0.0)

                if base.upper() in ignore:
                    self._manual_baseline[symbol] = untracked
                    continue
                if first_run:
                    self._manual_baseline[symbol] = untracked
                    if untracked > 0 and untracked * om.last_price(symbol) >= min_usd:
                        found_at_start.append((symbol, untracked, untracked * om.last_price(symbol)))
                    continue
                if untracked < baseline:                   # user sold coins we were ignoring
                    self._manual_baseline[symbol] = baseline = untracked
                extra = untracked - baseline
                if extra > 0 and extra * om.last_price(symbol) >= min_usd:
                    self._adopt(symbol, extra, source='new manual buy')
            except Exception as e:
                logger.error(f"❌ Manual-trade sync failed for {symbol}: {e}")

        self._save_manual_state()
        if found_at_start:
            lines = ", ".join(f"{s.split('/')[0]} {q:.6g} (${v:,.0f})" for s, q, v in found_at_start)
            self._notify(f"👋 I found coins already in the account that I do NOT manage: {lines}.\n"
                         f"I will leave them alone. Send /adopt SYMBOL to protect one, or buy more and I will "
                         f"protect the new coins automatically.")

    def holdings_report(self) -> list:
        """[(symbol, held, tracked, ignored, value)] for coins with a meaningful balance."""
        balances = self._balances_total()
        out = []
        for symbol in self.symbols:
            base = symbol.split('/')[0]
            held = balances.get(base, 0.0)
            if held <= 0:
                continue
            price = self.order_manager.last_price(symbol) if self.order_manager else 0.0
            tracked = self.open_positions[symbol]['amount'] if symbol in self.open_positions else 0.0
            ignored = (getattr(self, '_manual_baseline', None) or {}).get(symbol, 0.0)
            if held * price >= 5 or tracked > 0:
                out.append((symbol, held, tracked, ignored, held * price))
        return out

    def adopt_now(self, symbol: str) -> str:
        """User asked to protect the coins we are currently ignoring (or any untracked balance)."""
        held = self._balances_total().get(symbol.split('/')[0], 0.0)
        tracked = self.open_positions[symbol]['amount'] if symbol in self.open_positions else 0.0
        qty = held - tracked
        price = self.order_manager.last_price(symbol)
        if qty * price < float(self.config.get('adopt_min_usd', 15)):
            return f"Nothing to adopt: no untracked {symbol.split('/')[0]} above ${self.config.get('adopt_min_usd', 15)}."
        base = getattr(self, '_manual_baseline', None)
        if base is not None:
            base[symbol] = max(0.0, base.get(symbol, 0.0) - qty)
            self._save_manual_state()
        self._adopt(symbol, qty, source='/adopt')
        return f"Adopted {qty:.6g} {symbol.split('/')[0]}."

    def ignore_holding(self, symbol: str) -> str:
        held = self._balances_total().get(symbol.split('/')[0], 0.0)
        tracked = self.open_positions[symbol]['amount'] if symbol in self.open_positions else 0.0
        if self._manual_baseline is None:
            self._manual_baseline = {}
        self._manual_baseline[symbol] = max(0.0, held - tracked)
        self._save_manual_state()
        return f"OK - I will not touch the untracked {symbol.split('/')[0]} in the account."

    def release_position(self, symbol: str) -> str:
        """Stop managing a position: cancel its stop and hand the coins back for manual trading."""
        with self._lock:
            pos = self.open_positions.get(symbol)
            if not pos:
                return f"No managed position in {symbol}."
            if symbol in self.pending_exits:
                return f"{symbol} is already being sold by the bot."
            if pos.get('oco') and self.order_manager:
                if self.order_manager.cancel_protection(symbol, pos['oco']) == 'error':
                    return f"Could not cancel the stop on {symbol}; try again."
            amount = pos['amount']
            del self.open_positions[symbol]
            save_positions_to_file(self.open_positions)
            if self._manual_baseline is None:
                self._manual_baseline = {}
            self._manual_baseline[symbol] = self._manual_baseline.get(symbol, 0.0) + amount
            self._save_manual_state()
        return f"Released {symbol}: stop cancelled, the coins are yours to sell by hand. I will not manage them."

    def set_stop(self, symbol: str, stop: float) -> str:
        """Move a managed position's stop (up OR down) and replace the exchange order."""
        with self._lock:
            pos = self.open_positions.get(symbol)
            if not pos:
                return f"No managed position in {symbol}."
            price = self.order_manager.last_price(symbol) if self.order_manager else pos.get('current_price', 0)
            if not (0 < stop < price):
                return f"A stop must be below the current price (${price:,.4f})."
            pos['stop_loss'] = stop
            pos['initial_stop'] = stop
            prot = pos.get('oco')
            if prot and self.order_manager:
                res = self.order_manager.cancel_protection(symbol, prot)
                if res == 'done':
                    return f"{symbol}'s old stop had just filled - nothing to move."
                pos['oco'] = None
            pos.pop('protect_after', None)
            save_positions_to_file(self.open_positions)
        ok = self._place_protection(symbol) if self.order_manager else True
        return (f"Stop for {symbol} set to ${stop:,.4f} ({(stop / price - 1) * 100:+.1f}%)." if ok
                else f"Stop saved at ${stop:,.4f} but the exchange order failed - retrying every minute.")

    def manual_buy_preview(self, symbol: str, usd: float, stop: Optional[float] = None) -> Dict:
        om = self.order_manager
        bid, ask = om.book(symbol)
        auto_stop, atr = self._manual_stop(symbol, ask)
        stop = stop if stop else auto_stop
        if not 0 < stop < ask:
            return {'error': f"The stop must be below the price (${ask:,.4f})."}
        return {'price': ask, 'qty': usd / ask, 'stop': stop, 'stop_pct': (stop / ask - 1) * 100, 'atr': atr,
                'usd': usd}

    def manual_buy(self, symbol: str, usd: float, stop: Optional[float] = None) -> str:
        """Limit-buy (repriced toward the ask) with an automatic stop the moment it fills."""
        pv = self.manual_buy_preview(symbol, usd, stop)
        if 'error' in pv:
            return pv['error']
        if symbol in self.open_positions or symbol in self.pending_entries:
            return f"Already have {symbol} (or an order for it). Use /close first, or buy it on the exchange and I will add to it."
        ok = self._submit_limit_entry(symbol, pv['price'], pv['qty'], pv['stop'], 0.0, 'manual', pv['atr'])
        return (f"Buy order placed for ~${usd:,.0f} of {symbol} with a stop at ${pv['stop']:,.4f}. "
                f"You will get a message when it fills and the stop is on." if ok
                else "The buy order could not be placed - check /orders and the logs.")

    def set_paused(self, value: bool):
        """Pause/resume NEW automatic entries. Open positions, stops and exits are unaffected."""
        self.paused = bool(value)
        self._save_breaker_state()
        logger.info(f"{'⏸️ Trading PAUSED' if self.paused else '▶️ Trading RESUMED'}")

    def cancel_pending_entries(self) -> int:
        """Cancel entry orders that are still working (partial fills are kept and registered)."""
        with self._lock:
            n = 0
            for st in self.pending_entries.values():
                st['cancel_requested'] = True
                n += 1
            if n:
                self._save_pending()
        if n and self.order_manager:
            self.manage_orders()                 # resolve them now instead of waiting for the next minute
        return n

    def gate_status(self) -> Dict:
        """BTC gate details for display: allowed flag plus BTC price and its EMA."""
        days = int(self.config.get('btc_gate_days', 50 if self.strategy_mode == 'trend_hold' else 0))
        out = {'enabled': days > 0, 'days': days, 'allowed': self._longs_allowed(),
               'symbol': None, 'btc': None, 'ema': None}
        if days <= 0:
            return out
        out['symbol'] = next((s for s in ('BTC/USDC', 'BTC/USDT') if s in self.symbols), 'BTC/USDC')
        try:
            df = market_gate.closed_only(
                self.data_feed.get_ohlcv(symbol=out['symbol'], interval=self.timeframe, limit=1000), self.timeframe)
            span = int(days * market_gate.BARS_PER_DAY.get(self.timeframe, 6))
            if len(df) >= span + 1:
                close = df['close']
                out['btc'] = float(close.iloc[-1])
                out['ema'] = float(close.ewm(span=span, adjust=False).mean().iloc[-1])
        except Exception as e:
            logger.debug(f"gate_status: {e}")
        return out

    def _tradable_symbols(self, symbols):
        """Keep only symbols that are TRADING on the connected exchange; log the rest."""
        try:
            info = self.binance_client.get_exchange_info()
            live = {s['symbol'] for s in info['symbols'] if s.get('status') == 'TRADING'}
        except Exception as e:
            logger.warning(f"⚠️ Could not verify symbols against the exchange: {e}")
            return list(symbols)
        keep = [s for s in symbols if s.replace('/', '') in live]
        dropped = [s for s in symbols if s not in keep]
        if dropped:
            logger.warning(f"⚠️ Not tradable on this exchange, ignoring: {', '.join(dropped)}")
        return keep

    @property
    def strategy_mode(self) -> str:
        return self.config.get('strategy_mode') or 'legacy'

    def _history_limit(self) -> int:
        """Candles to fetch when scanning: trend_hold needs ~330 (55-day channel on 4h)."""
        if self.strategy_mode == 'trend_hold':
            return min(1000, max(200, trend_hold.history_needed(self.config) + 50))
        return 200

    def _exit_history_limit(self, position: Dict) -> int:
        if trend_hold.is_trend_hold(position):
            return min(1000, max(50, trend_hold.history_needed(self.config) + 5))
        return 50

    def _longs_allowed(self) -> bool:
        """BTC gate: new longs only while BTC's close is above its N-day EMA (default 50 for
        trend_hold, off otherwise; btc_gate_days=0 disables). Fails CLOSED when BTC data is
        unavailable. Cached for 60s - it only changes when a candle closes."""
        days = int(self.config.get('btc_gate_days', 50 if self.strategy_mode == 'trend_hold' else 0))
        if days <= 0:
            return True
        now = _time.time()
        cached_at, cached = getattr(self, '_gate_cache', (0.0, None))
        if cached is not None and now - cached_at < 60:
            return cached
        symbol = next((s for s in ('BTC/USDC', 'BTC/USDT') if s in self.symbols), 'BTC/USDC')
        try:
            df = self.data_feed.get_ohlcv(symbol=symbol, interval=self.timeframe, limit=1000)
            up = market_gate.btc_uptrend(market_gate.closed_only(df, self.timeframe), days, self.timeframe)
        except Exception as e:
            logger.warning(f"⚠️ BTC gate: could not read {symbol}: {e}")
            up = None
        if up is None:
            logger.warning("⚠️ BTC gate: not enough BTC history - blocking new longs (fail closed)")
            up = False
        elif cached is not None and up != cached:
            logger.info(f"🚦 BTC gate {'OPEN' if up else 'CLOSED'}: BTC is "
                        f"{'above' if up else 'below'} its {days}-day EMA")
        self._gate_cache = (now, up)
        return up

    @property
    def shorts_enabled(self) -> bool:
        """Long-only unless "enable_shorts": true in config.json (default: off)."""
        return bool(self.config.get('enable_shorts', False))

    @property
    def active_slots(self) -> int:
        """Open spot + futures positions + entry orders still working."""
        return len(self.open_positions) + len(self.open_futures_positions) + len(self.pending_entries)

    def _notify(self, text: str):
        if has_notifier:
            try:
                notifier.send_message_sync(text)
            except Exception:
                pass

    def _submit_limit_entry(self, symbol: str, signal_price: float, units: float,
                            stop_loss: float, take_profit: float,
                            signal_type: str, atr: float) -> bool:
        om = self.order_manager
        with self._lock:
            try:
                qty = om.round_qty_down(symbol, units)
                if qty <= 0 or qty * signal_price < om.min_notional(symbol):
                    logger.warning(f"⚠️ {symbol}: order too small after rounding ({qty})")
                    return False
                st = om.start_entry(symbol, qty, signal_price)
            except Exception as e:
                logger.error(f"❌ Failed to place entry limit order for {symbol}: {e}")
                return False
            if st is None:
                logger.warning(f"⚠️ {symbol}: entry order could not be placed")
                return False
            st.update(stop_loss=float(stop_loss), take_profit=float(take_profit),
                      signal_type=signal_type or 'unknown', atr=float(atr or 0.0))
            self.pending_entries[symbol] = st
            self._save_pending()

        logger.info(f"📥 Entry limit order working for {symbol}: {qty} @ {st['price']}")
        self._notify(f"📥 <b>ENTRY ORDER</b> [{self.trading_mode.upper()}]\n"
                     f"📊 <b>{symbol}</b>  {qty} @ <code>${st['price']}</code>\n"
                     f"🛑 Stop: <code>${stop_loss:.4f}</code>  🎯 Target: <code>${take_profit:.4f}</code>")
        return True

    def manage_orders(self):
        """Advance entries, exits and exchange-side stops. Safe to call every minute."""
        if not self.order_manager:
            return
        self.order_manager.maybe_sync_clock()      # keeps signed requests inside Binance's time window
        with self._lock:
            for step in (self._manage_pending_entries, self._manage_pending_exits,
                         self._manage_oco_protection, self.sync_manual_positions):
                try:
                    step()
                except Exception as e:
                    logger.error(f"❌ manage_orders/{step.__name__}: {e}")

    def _manage_pending_entries(self):
        om = self.order_manager
        for symbol, st in list(self.pending_entries.items()):
            try:
                res = om.advance_entry(st)
            except Exception as e:
                logger.error(f"❌ Entry order error for {symbol}: {e}")
                continue
            self._save_pending()                       # step / order id may have changed
            if res['status'] == 'open':
                continue
            del self.pending_entries[symbol]
            self._save_pending()
            if res['status'] == 'filled':
                self._register_filled_entry(symbol, st, res)
            else:
                logger.info(f"⌛ {symbol}: entry not filled - cancelled, no position opened")
                self._notify(f"⌛ Entry not filled for <b>{symbol}</b> - order cancelled")

    def _register_filled_entry(self, symbol: str, st: dict, res: dict):
        fill, qty, sp = res['avg_price'], res['qty'], st['signal_price']
        # keep the signal's percentage distances, anchored to the real fill price
        stop = fill * (1 - abs(sp - st['stop_loss']) / sp)
        target = fill * (1 + abs(st['take_profit'] - sp) / sp) if st.get('take_profit') else 0.0   # none for trend_hold

        self.open_positions[symbol] = self._build_position(
            symbol, 'long', qty, fill, stop, target, st.get('signal_type'), st.get('atr', 0.0))
        self.open_positions[symbol]['oco'] = None
        save_positions_to_file(self.open_positions)

        log_trade('open', symbol=symbol, side='long', entry=fill, units=qty,
                  stop_loss=stop, take_profit=target)
        add_trade({
            'symbol': symbol, 'action': 'open', 'side': 'long', 'amount': qty,
            'price': fill, 'stop_loss': stop, 'take_profit': target,
            'mode': self.trading_mode, 'signal_type': st.get('signal_type') or 'unknown',
            'quote_currency': symbol.split('/')[1],
        })
        logger.info(f"✅ Entry filled {symbol}: {qty} @ {fill:.6f}")
        self._notify(f"📈 <b>TRADE OPENED</b> [{self.trading_mode.upper()}]\n"
                     f"📊 <b>{symbol}</b>\n💵 Entry: <code>${fill:.4f}</code>\n"
                     f"📦 Units: <code>{qty:.6f}</code>\n"
                     f"🛑 Stop: <code>${stop:.4f}</code>\n🎯 Target: <code>${target:.4f}</code>")
        self._place_protection(symbol)

    def _place_protection(self, symbol: str) -> bool:
        """Place the exchange-side OCO (take-profit limit + stop-limit) for a position."""
        om = self.order_manager
        pos = self.open_positions.get(symbol)
        if not pos:
            return False
        try:
            price = om.last_price(symbol)
            qty = om.sellable_qty(symbol, pos['amount'])
            if qty <= 0:
                res = {'ok': False, 'error': 'too_small', 'detail': 'no sellable balance'}
            else:
                res = om.place_protection(symbol, qty, pos['stop_loss'], pos.get('take_profit') or 0.0, price)
        except Exception as e:
            res = {'ok': False, 'error': 'api', 'detail': f"{type(e).__name__}(code={getattr(e, 'code', None)}): {e}"}

        if res['ok']:
            pos['oco'] = {k: v for k, v in res.items() if k != 'ok'}      # kind + order id(s) + stop/limit/target/qty
            save_positions_to_file(self.open_positions)
            logger.info(f"🛡️ {symbol} OCO placed: stop {res['stop']} (limit {res['limit']}), "
                        f"target {res['target']}")
            return True

        pos['oco'] = None
        save_positions_to_file(self.open_positions)
        err = res['error']
        if err == 'stop_breached':
            logger.warning(f"⚠️ {symbol}: price already at/below stop - exiting now")
            self._begin_exit(symbol, 'stop_loss')
        elif err == 'target_reached':
            logger.info(f"🎯 {symbol}: price already at/above target - exiting now")
            self._begin_exit(symbol, 'take_profit')
        else:
            logger.error(f"❌ {symbol} has NO exchange stop ({err}: {res.get('detail', '')}). "
                         f"Software stop + market fallback remain active; retrying every minute.")
            if not pos.get('unprotected_notified'):
                pos['unprotected_notified'] = True
                self._notify(f"⚠️ <b>{symbol}</b> has no exchange-side stop ({err}). "
                             f"Bot-side stop is active, retrying.")
        return False

    def _replace_oco(self, symbol: str):
        """Ratchet: cancel the OCO and re-place it at the position's raised stop."""
        om = self.order_manager
        pos = self.open_positions[symbol]
        old = pos['oco']
        result = om.cancel_protection(symbol, old)
        if result != 'cancelled':
            return                                     # 'done' -> settled next cycle; 'error' -> retry
        pos['oco'] = None
        logger.info(f"🔼 {symbol}: ratcheting stop {old['stop']} -> {pos['stop_loss']:.6f}")
        self._place_protection(symbol)

    def _market_stop(self, symbol: str, why: str):
        """Safety net: cancel the OCO and sell at MARKET."""
        om = self.order_manager
        pos = self.open_positions[symbol]
        oco = pos['oco']
        result = om.cancel_protection(symbol, oco)
        if result == 'done':
            return                                     # it just filled - handled next cycle
        if result == 'error':
            logger.critical(f"🚨 {symbol}: stop breached ({why}) but OCO cancel failed - retrying")
            self._notify(f"🚨 <b>{symbol}</b>: stop breached but could not cancel OCO, retrying")
            return
        pos['oco'] = None
        logger.warning(f"⚠️ {symbol}: stop-limit not filled ({why}) - MARKET SELL")
        sold = om.market_sell(symbol, pos['amount'])
        if sold and sold['qty'] > 0:
            self._finalize_close(symbol, sold['avg_price'], 'stop_loss_market', qty=sold['qty'])
        else:
            logger.error(f"❌ {symbol}: market fallback sold nothing - will retry")

    def _manage_oco_protection(self):
        om = self.order_manager
        now = _time.time()
        for symbol, pos in list(self.open_positions.items()):
            if pos.get('side') != 'long' or symbol in self.pending_exits:
                continue
            if pos.get('mode') != self.trading_mode:
                continue
            oco = pos.get('oco')
            if not oco:
                if pos.get('protect_after', 0) > now:
                    continue                       # user cancelled the stop on purpose: give them time
                pos.pop('protect_after', None)
                self._place_protection(symbol)
                continue
            try:
                status = om.protection_status(symbol, oco)
                if status['status'] == 'ALL_DONE':
                    if status['filled']:
                        if status['filled'] == 'take_profit':
                            reason = 'take_profit'
                        else:
                            reason = 'trailing_stop' if pos.get('trailing_stop_active') else 'stop_loss'
                        self._finalize_close(symbol, status['avg_price'], reason, qty=status['qty'])
                    else:
                        logger.warning(f"⚠️ {symbol}: protective order ended without a fill (cancelled outside the bot?)")
                        pos['oco'] = None
                        wait = float(self.config.get('manual_override_seconds', 300))
                        pos['protect_after'] = now + wait
                        self._notify(f"👀 The stop on <b>{symbol}</b> was cancelled outside the bot. I will put it back in "
                                     f"{wait / 60:.0f} min unless the coins are gone. Sell them now, or /release {symbol.split('/')[0]} "
                                     f"to take it off my hands.")
                        save_positions_to_file(self.open_positions)
                    continue

                price = om.last_price(symbol)
                stop = oco['stop']

                # ---- MARKET FALLBACK: stop-limit didn't (or can't) fill ----
                if price <= stop:
                    oco.setdefault('breach_since', now)
                else:
                    oco.pop('breach_since', None)
                through = price <= stop * (1 - self.stop_market_fallback_pct)
                stuck = 'breach_since' in oco and now - oco['breach_since'] >= self.stop_trigger_timeout
                if through or stuck:
                    self._market_stop(symbol, 'price through stop-limit' if through
                                      else f'below stop for {now - oco["breach_since"]:.0f}s')
                    continue

                # ---- RATCHET the exchange stop up to the bot's trailing stop ----
                new_stop = pos['stop_loss']
                if new_stop > stop * (1 + self.oco_ratchet_min_pct) and price > new_stop * 1.001:
                    self._replace_oco(symbol)
            except Exception as e:
                logger.error(f"❌ OCO management error for {symbol}: {e}")
        if self.open_positions:
            save_positions_to_file(self.open_positions)

    def _manage_pending_exits(self):
        om = self.order_manager
        for symbol, ex in list(self.pending_exits.items()):
            try:
                res = om.advance_exit(ex['st'])
            except Exception as e:
                logger.error(f"❌ Exit order error for {symbol}: {e}")
                continue
            self._save_pending()
            if res['status'] == 'open':
                continue
            del self.pending_exits[symbol]
            self._save_pending()
            if symbol not in self.open_positions:
                continue
            if res['qty'] > 0:
                self._finalize_close(symbol, res['avg_price'], ex['reason'], qty=res['qty'])
            else:
                logger.warning(f"⚠️ {symbol}: nothing left to sell (dust) - closing record")
                self._finalize_close(symbol, om.last_price(symbol), ex['reason'] + '_dust')

    def _begin_exit(self, symbol: str, reason: str) -> bool:
        """Start a non-blocking exit: cancel the OCO, then limit sell (market fallback)."""
        om = self.order_manager
        with self._lock:
            if symbol in self.pending_exits:
                return True
            pos = self.open_positions.get(symbol)
            if not pos:
                return False
            oco = pos.get('oco')
            if oco:
                result = om.cancel_protection(symbol, oco)
                if result == 'done':
                    return True                        # already filled - settled next cycle
                if result == 'error':
                    return False
                pos['oco'] = None
            try:
                qty = om.sellable_qty(symbol, pos['amount'])
                if qty <= 0:
                    logger.error(f"❌ {symbol}: no sellable balance for exit ({reason})")
                    return False
                st = om.start_exit(symbol, qty, urgent=reason in self.URGENT_EXITS)
            except Exception as e:
                logger.error(f"❌ Could not start exit for {symbol}: {e}")
                return False
            self.pending_exits[symbol] = {'st': st, 'reason': reason}
            self._save_pending()
        logger.info(f"📤 Exit started for {symbol} ({reason})")
        return True

    def _handle_exit_signal(self, symbol: str, position: Dict, price: float, reason: str):
        """An exit rule fired for a live spot long (called from check_stop_losses)."""
        oco = position.get('oco')
        if oco and reason == 'take_profit':
            return                                     # the exchange limit sell handles it
        if oco and reason in ('stop_loss', 'trailing_stop'):
            # The exchange stop + market fallback own this, unless the bot's stop was raised
            # above the OCO's and hasn't been ratcheted yet - then price is already through it.
            if position['stop_loss'] <= oco['stop'] * (1 + 1e-4):
                return
        self.close_position(symbol, price, reason)

    def validate_and_adjust_order(self, symbol: str, quantity: float) -> Tuple[bool, float, str]:
        """
        Validate and adjust order quantity to meet LOT_SIZE requirements
        Returns: (is_valid, adjusted_quantity, error_message)
        """
        try:
            binance_symbol = symbol.replace('/', '')
            
            # Check if binance_client is available
            if not self.binance_client:
                return True, quantity, "No Binance client available"
            
            # Get symbol info
            symbol_info = self.binance_client.get_symbol_info(binance_symbol)
            
            # Check if symbol_info is None
            if symbol_info is None:
                return False, quantity, f"Could not retrieve symbol info for {binance_symbol}"
            
            # Find LOT_SIZE filter
            lot_size_filter = None
            for f in symbol_info['filters']:
                if f['filterType'] == 'LOT_SIZE':
                    lot_size_filter = f
                    break
            
            if not lot_size_filter:
                return True, quantity, "No LOT_SIZE filter found"
            
            # Get filter values
            min_qty = float(lot_size_filter['minQty'])
            max_qty = float(lot_size_filter['maxQty'])
            step_size = float(lot_size_filter['stepSize'])
            
            logger.info(f"📏 LOT_SIZE for {symbol}: min={min_qty}, max={max_qty}, step={step_size}")
            logger.info(f"   Original quantity: {quantity}")
            
            # Check minimum
            if quantity < min_qty:
                return False, quantity, f"Quantity {quantity} below minimum {min_qty}"
            
            # Check maximum
            if quantity > max_qty:
                return False, quantity, f"Quantity {quantity} above maximum {max_qty}"
            
            # Round to step size
            # quantity must be: step_size * N where N is integer
            steps = quantity / step_size
            rounded_steps = round(steps)  # Round to nearest integer
            adjusted_qty = rounded_steps * step_size
            
            # Ensure we don't go below minimum after rounding
            if adjusted_qty < min_qty:
                adjusted_qty = min_qty
            
            # Ensure we don't exceed maximum
            if adjusted_qty > max_qty:
                adjusted_qty = max_qty
            
            logger.info(f"   Adjusted quantity: {adjusted_qty}")
            
            return True, adjusted_qty, ""
            
        except Exception as e:
            logger.error(f"❌ Error validating order: {e}")
            return False, quantity, str(e)

    def place_manual_limit_order(self, symbol: str, side: str, quantity: float, price: float, 
                            stop_loss: Optional[float] = None, take_profit: Optional[float] = None) -> Tuple[bool, str]:
        """
        Place a MANUAL limit order (for user commands only)
        Returns: (success, message)
        """
        try:
            logger.info(f"🔍 MANUAL LIMIT ORDER:")
            logger.info(f"   Symbol: {symbol}")
            logger.info(f"   Side: {side}")
            logger.info(f"   Quantity: {quantity:.6f}")
            logger.info(f"   Limit Price: ${price:.2f}")
            logger.info(f"   Value: ${quantity * price:.2f}")
            
            # Set default stop/target if not provided
            if stop_loss is None:
                stop_loss = price * 0.95
            if take_profit is None:
                take_profit = price * 1.05
            
            # Validation checks
            if quantity <= 0:
                return False, "Invalid quantity"
            if price <= 0:
                return False, "Invalid price"
            
            # PAPER MODE - Simulate
            if self.trading_mode == 'paper':
                logger.info(f"📄 PAPER MANUAL LIMIT: {side.upper()} {quantity:.6f} {symbol} at ${price:.2f}")
                
                internal_side = 'long' if side.lower() in ['buy', 'long'] else 'short'

                # In paper mode, execute immediately
                success = self.open_position(
                    symbol=symbol,
                    side=internal_side,
                    entry_price=price,
                    units=quantity,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    signal_type='manual',
                    atr=0.0
                )
                return success, "Paper limit order simulated" if success else "Failed to place paper order"
            
            # LIVE/TESTNET MODE
            elif self.trading_mode in ['live', 'testnet'] and self.binance_client:
                try:
                    binance_symbol = symbol.replace('/', '')
                    
                    # VALIDATE AND ADJUST QUANTITY
                    is_valid, adjusted_qty, error = self.validate_and_adjust_order(symbol, quantity)
                    if not is_valid:
                        return False, f"Order validation failed: {error}"
                    
                    if adjusted_qty != quantity:
                        logger.info(f"🔄 Quantity adjusted from {quantity} to {adjusted_qty}")
                        quantity = adjusted_qty
                    
                    # Place limit order
                    if side == 'buy':
                        order = self.binance_client.order_limit_buy(
                            symbol=binance_symbol,
                            quantity=round(quantity, 6),
                            price=str(round(price, 2))
                        )
                    else:  # sell
                        order = self.binance_client.order_limit_sell(
                            symbol=binance_symbol,
                            quantity=round(quantity, 6),
                            price=str(round(price, 2))
                        )
                    
                    logger.info(f"✅ Manual limit order placed: {order['orderId']}")
                    
                    # Store in pending orders for tracking
                    if not hasattr(self, 'pending_orders'):
                        self.pending_orders = {}
                    
                    self.pending_orders[order['orderId']] = {
                        'symbol': symbol,
                        'side': side,
                        'quantity': quantity,
                        'price': price,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'time': datetime.now().isoformat()
                    }
                    
                    return True, f"Order placed: {order['orderId']}"
                    
                except Exception as e:
                    logger.error(f"❌ Manual limit order failed: {e}")
                    return False, str(e)
            
            else:
                return False, "No valid trading mode or binance client"
                
        except Exception as e:
            logger.error(f"❌ Error in manual limit order: {e}")
            return False, str(e)

    def scan_and_trade(self) -> List[Dict]:
        """
        Scan all symbols for trading signals
        Returns list of signals found
        """
        import time
        current_time = time.time()
        min_time_between_trades = 3600  # 1 hour — matches trading timeframe
        
        # Initialize per-pair last trade times if not exists
        if not hasattr(self, 'last_trade_time_per_pair'):
            self.last_trade_time_per_pair = {}
            # Try to load from portfolio
            try:
                portfolio = load_portfolio()
                self.last_trade_time_per_pair = portfolio.get('last_trade_time_per_pair', {})
            except:
                pass
        
        # Check circuit breaker status
        if not self.check_drawdown():
            logger.info("⛔ Circuit breaker active – no new trades")
            return []

        if getattr(self, 'paused', False):
            logger.info("⏸️ Trading paused - no new entries")
            return []

        if not self.config.get('auto_entries', True):
            logger.info("🖐️ auto_entries is off - the bot only manages your manual trades")
            return []
        auto_cap = int(self.config.get('auto_max_positions', 2))       # 0 = no separate cap
        if auto_cap > 0:
            n_auto = (sum(1 for p in self.open_positions.values() if p.get('signal_type') not in trend_hold.MANUAL_TYPES)
                      + sum(1 for st in self.pending_entries.values() if st.get('signal_type') not in trend_hold.MANUAL_TYPES))
            if n_auto >= auto_cap:
                logger.info(f"🎯 {n_auto}/{auto_cap} automatic positions open - not looking for new automatic entries")
                return []
        
        # Check if we can take new positions (spot + futures combined)
        current_positions = self.active_slots
        if current_positions >= self.max_positions:
            logger.info(f"⏭️ At max positions ({current_positions}/{self.max_positions})")
            return []
        
        slow = self.strategy_mode == 'trend_hold'
        if slow and not self.shorts_enabled and not self._longs_allowed():
            logger.info("📉 BTC gate closed (BTC below its EMA) - no new longs this scan")
            return []

        slots_available = self.max_positions - current_positions
        logger.info(f"🔍 Scanning {len(self.symbols)} symbols for signals ({slots_available} slots available)...")
        
        # Get current cash balance
        # Use the quote currencies actually traded (e.g. USDT), not a hardcoded USDC
        quotes = {s.split('/')[1] for s in self.symbols if '/' in s}
        cash_balance = max((self.get_cash_balance(q) for q in quotes), default=0)
        if cash_balance <= 10:
            logger.warning(f"⚠️ Low cash balance: ${cash_balance:.2f}")
            return []
        
        signals_found = []
        
        for symbol in self.symbols:
            try:
                # Skip if already in spot or futures position for this symbol
                if (symbol in self.open_positions or symbol in self.open_futures_positions
                        or symbol in self.pending_entries):
                    continue
                
                # ===== PER-PAIR COOLDOWN CHECK =====
                last_trade_time = self.last_trade_time_per_pair.get(symbol, 0)
                if last_trade_time > 0:
                    time_since_last = current_time - last_trade_time
                    if time_since_last < min_time_between_trades:
                        logger.debug(f"⏱️ {symbol}: Too soon since last trade ({time_since_last:.0f}s < {min_time_between_trades}s). Skipping...")
                        continue
                
                # Fetch data
                df = self.data_feed.get_ohlcv(
                    symbol=symbol,
                    interval=self.timeframe,
                    limit=self._history_limit()
                )
                if slow and not df.empty:
                    df = market_gate.closed_only(df, self.timeframe)   # act on closed candles only
                
                if df.empty or len(df) < 50:
                    logger.debug(f"⏭️ {symbol}: Insufficient data ({len(df)} candles)")
                    continue
                
                # ===== REGIME DETECTION (trend_hold does not use regimes) =====
                if slow:
                    regime = "trend_hold"
                else:
                    try:
                        regime = predict_regime(df)
                        logger.debug(f"📊 {symbol} regime: {regime}")

                        # Skip trading in certain regimes if desired
                        if "Volatile" in regime and self.trading_mode != 'paper':
                            logger.debug(f"⏭️ Skipping {symbol} due to volatile regime")
                            continue

                    except Exception as e:
                        logger.debug(f"Could not detect regime for {symbol}: {e}")
                        regime = "unknown"
                
                # Generate signal — regime passed so strategy selection is regime-aware
                try:
                    signal = generate_trade_signal(
                        df,
                        cash_balance,
                        self.risk_per_trade,
                        symbol=symbol,
                        trading_engine=self,
                        regime=regime,
                    )
                except Exception as e:
                    logger.error(f"❌ Error generating signal for {symbol}: {e}")
                    continue
                
                if signal and signal.get('side') == 'short' and not self.shorts_enabled:
                    logger.debug(f"⏭️ {symbol}: short signal ignored (enable_shorts is false)")
                    continue

                if signal and signal.get('side') == 'long' and not self._longs_allowed():
                    logger.debug(f"⏭️ {symbol}: long signal ignored (BTC gate closed)")
                    continue

                if signal:
                    # Check for duplicate signals
                    if signal.get('entry_price', 0) <= 0:
                        logger.error(f"❌ Invalid signal for {symbol}: missing valid 'entry_price' (got {signal.get('entry_price')})")
                        continue
                    
                    if signal.get('units', 0) <= 0:
                        logger.error(f"❌ Invalid signal for {symbol}: missing valid 'units' (got {signal.get('units')})")
                        continue
                    
                    signal_key = f"{symbol}_{signal.get('signal_type', 'unknown')}"
                    if slow:
                        # one attempt per closed candle. The plain key never resets, which would
                        # block a coin's second breakout forever.
                        signal_key += f"_{df['timestamp'].iloc[-1]}"
    
                    if signal_key != self.last_signals.get(symbol):
                        signals_found.append({
                            'symbol': symbol,
                            'signal': signal,
                            'regime': regime
                        })
                        self.last_signals[symbol] = signal_key

                        logger.info(f"✅ {symbol}: {signal.get('signal_type', 'SIGNAL').upper()} {signal['side']} signal at ${signal['entry_price']:.2f} (Regime: {regime})")
                    else:
                        logger.info(f"⏭️ {symbol}: Duplicate signal skipped")
                        
            except Exception as e:
                logger.error(f"❌ Error scanning {symbol}: {e}")
        
        # If found signals, update the per-pair last trade time when executed
        # The update happens in execute_signal after successful execution
        
        logger.info(f"📊 Scan complete: Found {len(signals_found)} signals")
        return signals_found
    
    def execute_signal(self, signal_data: Dict) -> bool:
        """
        Execute a trading signal
        """
        if not self.check_drawdown():
            logger.info("⛔ Circuit breaker active – signal not executed")
            return False

        try:
            if 'symbol' not in signal_data:
                logger.error("❌ No 'symbol' in signal_data")
                return False
            
            if 'signal' not in signal_data:
                logger.error("❌ No 'signal' in signal_data")
                return False
            
            symbol = signal_data['symbol']
            signal = signal_data['signal']
            
            # VALIDATE required fields
            required_fields = ['side', 'entry_price', 'units', 'stop_loss', 'take_profit']
            for field in required_fields:
                if field not in signal:
                    logger.error(f"❌ Signal missing required field '{field}': {signal}")
                    return False
                if field == 'entry_price' and signal[field] <= 0:
                    logger.error(f"❌ Signal has invalid entry price: {signal[field]}")
                    return False
                if field == 'units' and signal[field] <= 0:
                    logger.error(f"❌ Signal has invalid units: {signal[field]}")
                    return False
            
            logger.info(f"   Symbol: {symbol}")
            logger.info(f"   Signal type: {signal.get('signal_type', 'unknown')}")
            logger.info(f"   Side: {signal.get('side', 'unknown')}")
            logger.info(f"   Entry: ${signal.get('entry_price', 0):.2f}")
            logger.info(f"   Units: {signal.get('units', 0):.6f}")
            logger.info(f"   Stop: ${signal.get('stop_loss', 0):.2f}")
            logger.info(f"   Target: ${signal.get('take_profit', 0):.2f}")
            
            # Check if we already have a position (spot or futures)
            if symbol in self.open_positions or symbol in self.pending_entries:
                logger.warning(f"⚠️ Already in spot position (or entry pending) for {symbol}")
                return False
            if symbol in self.open_futures_positions:
                logger.warning(f"⚠️ Already in futures position for {symbol}")
                return False

            # Check max positions (combined)
            total_positions = self.active_slots
            if total_positions >= self.max_positions:
                logger.warning(f"⚠️ At max positions ({self.max_positions})")
                return False

            # ===== ROUTE BY MARKET =====
            market = signal.get('market', 'spot')

            if (signal['side'] == 'short' or market == 'futures') and not self.shorts_enabled:
                logger.info(f"⏭️ {symbol}: short/futures signal not executed (enable_shorts is false)")
                return False

            # --- SHORT → Futures engine ---
            if signal['side'] == 'short' or market == 'futures':
                if not self.futures_engine:
                    logger.error("❌ FuturesEngine not available — cannot execute short")
                    return False

                # Check we have enough cash margin
                quote_currency = symbol.split('/')[1]
                leverage = signal.get('leverage', 1)
                margin_needed = (signal['units'] * signal['entry_price']) / leverage
                cash = self.get_cash_balance(quote_currency)
                logger.info(f"   Margin needed: ${margin_needed:.2f}, Cash: ${cash:.2f}")
                if cash < margin_needed:
                    logger.warning(f"⚠️ Insufficient margin: need ${margin_needed:.2f}, have ${cash:.2f}")
                    return False

                result = self.futures_engine.open_short(
                    symbol=symbol,
                    amount=signal['units'],
                    entry_price=signal['entry_price'],
                    stop_loss=signal['stop_loss'],
                    take_profit=signal['take_profit'],
                    signal_type=signal.get('signal_type', 'unknown'),
                    atr=signal.get('atr', 0.0)
                )
                if result:
                    # Sync in-memory futures positions
                    self.open_futures_positions = get_futures_positions()
                    logger.info(f"✅ Futures SHORT opened: {symbol}")
                    # Notify
                    if has_notifier:
                        try:
                            notifier.send_message_sync(
                                f"📉 <b>SHORT OPENED</b> [{self.trading_mode.upper()}]\n"
                                f"📊 <b>{symbol}</b>\n"
                                f"💵 Entry: <code>${signal['entry_price']:.4f}</code>\n"
                                f"📦 Units: <code>{signal['units']:.6f}</code>\n"
                                f"🛑 Stop: <code>${signal['stop_loss']:.4f}</code>\n"
                                f"🎯 Target: <code>${signal['take_profit']:.4f}</code>"
                            )
                        except Exception:
                            pass
                return result

            # --- LONG → Spot execution (existing path) ---
            elif signal['side'] == 'long':
                quote_currency = symbol.split('/')[1]
                cash = self.get_cash_balance(quote_currency)
                cost = signal['units'] * signal['entry_price']
                logger.info(f"   {quote_currency} balance: ${cash:.2f}, Cost: ${cost:.2f}")

                if cash < cost:
                    logger.warning(f"⚠️ Insufficient funds: Need ${cost:.2f}, have ${cash:.2f}")
                    return False

                if cost < 10:
                    logger.warning(f"⚠️ Order value ${cost:.2f} below minimum $10")
                    return False

            # Execute the spot long position
            result = self.open_position(
                symbol=symbol,
                side=signal['side'],
                entry_price=signal['entry_price'],
                units=signal['units'],
                stop_loss=signal['stop_loss'],
                take_profit=signal['take_profit'],
                signal_type=signal.get('signal_type', 'unknown'),
                atr=signal.get('atr', 0.0)
            )

            if result:
                # Update last trade time for this pair
                import time
                self.last_trade_time_per_pair[symbol] = time.time()
                
                # Persist to portfolio
                try:
                    portfolio = load_portfolio()
                    portfolio['last_trade_time_per_pair'] = self.last_trade_time_per_pair
                    save_portfolio(portfolio)
                except Exception as e:
                    logger.warning(f"Could not persist per-pair trade time: {e}")
                
                logger.info(f"✅ Successfully opened spot position for {symbol}")
            
            return result
        except Exception as e:
            logger.error(f"❌ Error executing signal: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

# Create singleton instance
trading_engine = TradingEngine()