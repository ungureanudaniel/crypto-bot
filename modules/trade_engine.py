import sys
import os
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from modules.data_feed import data_feed
from modules.strategy_tools import generate_trade_signal
from modules.regime_switcher import predict_regime, train_model
from modules.portfolio import (
    add_trade, load_portfolio, save_portfolio,
    get_cash, update_cash, get_positions, save_positions,
    get_futures_positions, open_futures_position, close_futures_position
)
from config_loader import get_binance_client, get_futures_client, config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
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
client = get_binance_client()

# -------------------------------------------------------------------
# PORTFOLIO POSITION HELPERS
# -------------------------------------------------------------------
def save_positions_to_file(positions: Dict):
    """Save open positions to portfolio.json"""
    try:
        save_positions(positions)
        logger.debug(f"💾 Saved {len(positions)} positions to portfolio.json")
    except Exception as e:
        logger.error(f"❌ Failed to save positions: {e}")

def load_positions_from_file() -> Dict:
    """Load open positions from portfolio.json"""
    try:
        positions = get_positions()
        logger.info(f"📂 Loaded {len(positions)} positions from portfolio.json")
        return positions
    except Exception as e:
        logger.error(f"❌ Failed to load positions: {e}")
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
        self.risk_per_trade = self.config.get('risk_per_trade', 0.02)
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

        # Initialize futures client for short execution
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

        # Restore last_trade_time from portfolio to survive restarts
        try:
            portfolio = load_portfolio()
            saved_time = portfolio.get('last_trade_time', 0)
            self.last_trade_time = saved_time
            if saved_time:
                import time as _time
                age = _time.time() - saved_time
                logger.info(f"⏱️ Last trade was {age/60:.0f} min ago (restored from portfolio)")
        except Exception:
            self.last_trade_time = 0
        
        logger.info(f"🚀 Trading Engine initialized for {self.trading_mode.upper()} mode")
        logger.info(f"📊 Monitoring {len(self.symbols)} symbols on {self.timeframe}")
        logger.info(f"📈 Max positions: {self.max_positions}, Risk per trade: {self.risk_per_trade:.1%}")
        logger.info(f"📂 Loaded {len(self.open_positions)} existing positions")
    
    def get_cash_balance(self, quote_currency: str = "USDT") -> float:
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
        """
        Check exits for all open positions each minute.
        Layer 1 (signal reversal) + Layer 2 (trailing stop) via exit_manager.
        Falls back to plain SL/TP if exit_manager is unavailable.
        Returns True if any positions were closed.
        """
        positions_closed = False

        # --- Spot positions ---
        if self.open_positions:
            try:
                from modules.exit_manager import evaluate_exit
            except ImportError:
                evaluate_exit = None

            current_prices = self.get_current_prices()

            for symbol, position in list(self.open_positions.items()):
                current_price = current_prices.get(symbol)
                if not current_price:
                    logger.warning(f"⚠️ Could not get price for {symbol}, skipping")
                    continue

                should_exit = False
                reason      = ''

                if evaluate_exit:
                    try:
                        df = self.data_feed.get_ohlcv(
                            symbol=symbol,
                            interval=self.timeframe,
                            limit=100
                        )
                    except Exception:
                        df = None

                    should_exit, reason = evaluate_exit(
                        symbol, position, current_price, df
                    )

                    # Persist updated trailing stop back to file
                    if not should_exit and position.get('trailing_stop_active'):
                        self.open_positions[symbol] = position
                        save_positions_to_file(self.open_positions)

                else:
                    # Plain SL/TP fallback
                    side = position['side']
                    sl   = position.get('stop_loss', 0)
                    tp   = position.get('take_profit', 0)
                    if side == 'long':
                        if sl and current_price <= sl:
                            should_exit, reason = True, 'stop_loss'
                        elif tp and current_price >= tp:
                            should_exit, reason = True, 'take_profit'
                    else:
                        if sl and current_price >= sl:
                            should_exit, reason = True, 'stop_loss'
                        elif tp and current_price <= tp:
                            should_exit, reason = True, 'take_profit'

                if should_exit:
                    logger.info(f"🚪 Exiting {symbol} @ ${current_price:.4f} | Reason: {reason}")
                    self.close_position(symbol, current_price, reason)
                    positions_closed = True

        # --- Futures positions ---
        if self.futures_engine:
            closed = self.futures_engine.check_stops()
            if closed:
                self.open_futures_positions = get_futures_positions()
                positions_closed = True

        return positions_closed
    
    def close_position(self, symbol: str, exit_price: float, reason: str) -> bool:
        """
        Close an existing position
        """
        if symbol not in self.open_positions:
            logger.warning(f"⚠️ No position found for {symbol}")
            return False
        
        position = self.open_positions[symbol]
        amount = position['amount']
        entry_price = position['entry_price']
        quote_currency = position.get('quote_currency', 'USDT')
        
        # Calculate PnL
        if position['side'] == 'long':
            pnl = (exit_price - entry_price) * amount
            pnl_pct = (exit_price / entry_price - 1) * 100
        else:  # short
            pnl = (entry_price - exit_price) * amount
            pnl_pct = (1 - exit_price / entry_price) * 100
        
        # PAPER MODE - update cash
        if self.trading_mode == 'paper':
            logger.info(f"📄 PAPER CLOSE: {position['side'].upper()} {amount:.6f} {symbol} at ${exit_price:.2f}")
            
            # Add cash back (original cost + profit)
            cash_return = amount * exit_price
            update_cash(quote_currency, cash_return, "add")
        
        # Execute REAL trade if in live/testnet mode
        elif self.trading_mode in ['live', 'testnet'] and self.binance_client:
            try:
                binance_symbol = symbol.replace('/', '')
                
                if position['side'] == 'long':
                    order = self.binance_client.order_market_sell(
                        symbol=binance_symbol,
                        quantity=round(amount, 6)
                    )
                    logger.info(f"📤 Live SELL order executed: {order['orderId']}")
            except Exception as e:
                logger.error(f"❌ Failed to execute live sell order: {e}")
                return False
        
        # Remove from open positions
        del self.open_positions[symbol]
        
        # Save updated positions to file
        save_positions_to_file(self.open_positions)
        
        # Record trade in history
        add_trade({
            'symbol': symbol,
            'action': 'close',
            'side': position['side'],
            'amount': amount,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'reason': reason,
            'mode': self.trading_mode,
            'quote_currency': quote_currency
        })
        
        # Send notification
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
        
        logger.info(f"✅ Closed {symbol}: PnL ${pnl:.2f} ({pnl_pct:+.1f}%) ({self.trading_mode})")
        return True

    def open_position(self, symbol: str, side: str, entry_price: float,
                     units: float, stop_loss: float, take_profit: float,
                     signal_type: str = '', **kwargs) -> bool:
        """Open a new position with stop loss and take profit"""
        # DEBUG
        logger.info(f"🔍 OPEN POSITION ATTEMPT:")
        logger.info(f"   Symbol: {symbol}")
        logger.info(f"   Side: {side}")
        logger.info(f"   Entry: ${entry_price:.2f}")
        logger.info(f"   Units: {units:.6f}")
        logger.info(f"   Value: ${units * entry_price:.2f}")
        
        # Check minimum order value
        min_order_value = 10
        if units * entry_price < min_order_value:
            logger.warning(f"⚠️ Order value ${units * entry_price:.2f} below minimum ${min_order_value}")
            return False
        
        # Validation checks
        if units <= 0:
            logger.warning(f"⚠️ Invalid units: {units}")
            return False
        
        if entry_price <= 0:
            logger.warning(f"⚠️ Invalid entry price: {entry_price}")
            return False
        
        # Check if already in position
        if symbol in self.open_positions:
            logger.info(f"⏭️ Already in position for {symbol}")
            return False
        
        # Check max positions
        if len(self.open_positions) >= self.max_positions:
            logger.info(f"⏭️ At max positions ({self.max_positions})")
            return False
        
        base_currency = symbol.split('/')[0]
        quote_currency = symbol.split('/')[1]
        
        # ===== EXECUTION BASED ON MODE =====
        execution_success = False
        
        # PAPER MODE
        if self.trading_mode == 'paper':
            logger.info(f"📄 PAPER TRADE: {side.upper()} {units:.6f} {symbol} at ${entry_price:.2f}")

            # Check cash balance
            cash_balance = self.get_cash_balance(quote_currency)
            cost = units * entry_price
            
            if cash_balance < cost:
                logger.error(f"❌ Insufficient {quote_currency}: have ${cash_balance:.2f}, need ${cost:.2f}")
                return False
            
            # Deduct cash
            update_cash(quote_currency, cost, "subtract")
            execution_success = True
        
        # LIVE/TESTNET MODE — longs only (shorts route via futures_engine)
        elif self.trading_mode in ['live', 'testnet'] and self.binance_client:
            try:
                binance_symbol = symbol.replace('/', '')

                if side == 'long':
                    usdt_balance = get_usdt_balance(self.binance_client)
                    cost = units * entry_price

                    logger.info(f"   USDT balance: ${usdt_balance:.2f}")
                    logger.info(f"   Required: ${cost:.2f}")

                    if usdt_balance < cost:
                        logger.error(f"❌ Insufficient USDT: have ${usdt_balance:.2f}, need ${cost:.2f}")
                        return False

                    is_valid, adjusted_units, error = self.validate_and_adjust_order(symbol, units)
                    if not is_valid:
                        logger.error(f"❌ Order validation failed: {error}")
                        return False
                    if adjusted_units != units:
                        logger.info(f"🔄 Quantity adjusted: {units} → {adjusted_units}")
                        units = adjusted_units

                    order = self.binance_client.order_market_buy(
                        symbol=binance_symbol,
                        quantity=round(units, 6)
                    )
                    logger.info(f"✅ Live BUY order executed: {order['orderId']}")
                    execution_success = True

                elif side == 'short':
                    # Shorts must go through futures_engine — should not reach here
                    logger.error("❌ open_position called with side='short' — use futures_engine instead")
                    return False

            except Exception as e:
                logger.error(f"❌ Failed to execute live order: {e}")
                return False
        
        else:
            logger.warning("⚠️ No valid trading mode or binance client")
            return False
        
        # ===== ONLY IF EXECUTION SUCCEEDED =====
        if execution_success:
            # Add to open positions — store signal_type and atr for exit_manager
            self.open_positions[symbol] = {
                'side': side,
                'amount': units,
                'entry_price': entry_price,
                'current_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'quote_currency': quote_currency,
                'value': units * entry_price,
                'pnl': 0.0,
                'pnl_pct': 0.0,
                'signal_type': signal_type if signal_type else 'unknown',
                'atr': kwargs.get('atr', 0.0),
                'trailing_stop_active': False,
                'entry_time': datetime.now().isoformat(),
                'mode': self.trading_mode
            }
            
            # Save updated positions to file
            save_positions_to_file(self.open_positions)
            
            # Record trade in history
            add_trade({
                'symbol': symbol,
                'action': 'open',
                'side': side,
                'amount': units,
                'price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'mode': self.trading_mode,
                'quote_currency': quote_currency
            })
            
            # Send notification
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
                
                # In paper mode, execute immediately (simulate fill)
                success = self.open_position(
                    symbol=symbol,
                    side=side,
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
        # ===== MINIMUM TIME BETWEEN TRADES =====
        import time
        current_time = time.time()
        min_time_between_trades = 3600  # 1 hour — matches trading timeframe
        
        # Check if enough time has passed since last trade
        if hasattr(self, 'last_trade_time') and self.last_trade_time:
            time_since_last = current_time - self.last_trade_time
            if time_since_last < min_time_between_trades:
                logger.info(f"⏱️ Too soon since last trade ({time_since_last:.0f}s < {min_time_between_trades}s). Waiting...")
                return []
        
        # Check if we can take new positions (spot + futures combined)
        current_positions = len(self.open_positions) + len(self.open_futures_positions)
        if current_positions >= self.max_positions:
            logger.info(f"⏭️ At max positions ({current_positions}/{self.max_positions})")
            return []
        
        slots_available = self.max_positions - current_positions
        logger.info(f"🔍 Scanning {len(self.symbols)} symbols for signals ({slots_available} slots available)...")
        
        # Get current cash balance
        cash_balance = self.get_cash_balance()
        if cash_balance <= 10:
            logger.warning(f"⚠️ Low cash balance: ${cash_balance:.2f}")
            return []
        
        signals_found = []
        
        for symbol in self.symbols:
            try:
                # Skip if already in spot or futures position for this symbol
                if symbol in self.open_positions or symbol in self.open_futures_positions:
                    continue
                
                # Fetch data
                df = self.data_feed.get_ohlcv(
                    symbol=symbol,
                    interval=self.timeframe,
                    limit=200
                )
                
                if df.empty or len(df) < 50:
                    logger.debug(f"⏭️ {symbol}: Insufficient data ({len(df)} candles)")
                    continue
                
                # ===== REGIME DETECTION =====
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
                
                if signal:
                    # Check for duplicate signals
                    signal_key = f"{symbol}_{signal.get('signal_type', 'unknown')}"
                    
                    if signal_key != self.last_signals.get(symbol):
                        signals_found.append({
                            'symbol': symbol,
                            'signal': signal,
                            'regime': regime
                        })
                        self.last_signals[symbol] = signal_key
                        
                        logger.info(f"✅ {symbol}: {signal.get('signal_type', 'SIGNAL').upper()} {signal['side']} signal at ${signal['entry']:.2f} (Regime: {regime})")
                    else:
                        logger.info(f"⏭️ {symbol}: Duplicate signal skipped")
                        
            except Exception as e:
                logger.error(f"❌ Error scanning {symbol}: {e}")
        
        # If we found signals, update the last trade time and persist it
        if signals_found:
            self.last_trade_time = current_time
            try:
                portfolio = load_portfolio()
                portfolio['last_trade_time'] = current_time
                save_portfolio(portfolio)
            except Exception as e:
                logger.warning(f"Could not persist last_trade_time: {e}")
            logger.info(f"⏱️ Updated last trade time to {time.strftime('%H:%M:%S', time.localtime(current_time))}")
        
        logger.info(f"📊 Scan complete: Found {len(signals_found)} signals")
        return signals_found
    
    def execute_signal(self, signal_data: Dict) -> bool:
        """
        Execute a trading signal
        """
        try:
            # DEBUG: Log the signal data
            logger.info(f"🔍 EXECUTE_SIGNAL called")
            logger.info(f"   Signal data type: {type(signal_data)}")
            logger.info(f"   Signal data keys: {signal_data.keys()}")
            
            if 'symbol' not in signal_data:
                logger.error("❌ No 'symbol' in signal_data")
                return False
            
            if 'signal' not in signal_data:
                logger.error("❌ No 'signal' in signal_data")
                return False
            
            symbol = signal_data['symbol']
            signal = signal_data['signal']
            
            logger.info(f"   Symbol: {symbol}")
            logger.info(f"   Signal type: {signal.get('signal_type', 'unknown')}")
            logger.info(f"   Side: {signal.get('side', 'unknown')}")
            logger.info(f"   Entry: ${signal.get('entry', 0):.2f}")
            logger.info(f"   Units: {signal.get('units', 0):.6f}")
            logger.info(f"   Stop: ${signal.get('stop_loss', 0):.2f}")
            logger.info(f"   Target: ${signal.get('take_profit', 0):.2f}")
            
            # Check if we already have a position (spot or futures)
            if symbol in self.open_positions:
                logger.warning(f"⚠️ Already in spot position for {symbol}")
                return False
            if symbol in self.open_futures_positions:
                logger.warning(f"⚠️ Already in futures position for {symbol}")
                return False

            # Check max positions (combined)
            total_positions = len(self.open_positions) + len(self.open_futures_positions)
            if total_positions >= self.max_positions:
                logger.warning(f"⚠️ At max positions ({self.max_positions})")
                return False

            # ===== ROUTE BY MARKET =====
            market = signal.get('market', 'spot')

            # --- SHORT → Futures engine ---
            if signal['side'] == 'short' or market == 'futures':
                if not self.futures_engine:
                    logger.error("❌ FuturesEngine not available — cannot execute short")
                    return False

                # Check we have enough cash margin
                quote_currency = symbol.split('/')[1]
                leverage = signal.get('leverage', 1)
                margin_needed = (signal['units'] * signal['entry']) / leverage
                cash = self.get_cash_balance(quote_currency)
                logger.info(f"   Margin needed: ${margin_needed:.2f}, Cash: ${cash:.2f}")
                if cash < margin_needed:
                    logger.warning(f"⚠️ Insufficient margin: need ${margin_needed:.2f}, have ${cash:.2f}")
                    return False

                result = self.futures_engine.open_short(
                    symbol=symbol,
                    amount=signal['units'],
                    entry_price=signal['entry'],
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
                                f"💵 Entry: <code>${signal['entry']:.4f}</code>\n"
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
                cost = signal['units'] * signal['entry']
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
                entry_price=signal['entry'],
                units=signal['units'],
                stop_loss=signal['stop_loss'],
                take_profit=signal['take_profit'],
                signal_type=signal.get('signal_type', 'unknown'),
                atr=signal.get('atr', 0.0)
            )

            if result:
                logger.info(f"✅ Successfully opened spot position for {symbol}")
            else:
                logger.warning(f"❌ Failed to open spot position for {symbol}")

            return result
            
        except Exception as e:
            logger.error(f"❌ Error executing signal: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

# Create singleton instance
trading_engine = TradingEngine()