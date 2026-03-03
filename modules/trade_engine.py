import sys
import os
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from data_feed import data_feed
from strategy_tools import generate_trade_signal
from config_loader import config
from modules.regime_switcher import predict_regime, train_model
from portfolio import add_trade, get_performance_summary, set_initial_balance, update_paper_balance, load_portfolio, save_portfolio
from config_loader import get_binance_client 

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
        portfolio = load_portfolio()
        portfolio["positions"] = positions
        save_portfolio(portfolio)
        logger.debug(f"💾 Saved {len(positions)} positions to portfolio.json")
    except Exception as e:
        logger.error(f"❌ Failed to save positions: {e}")

def load_positions_from_file() -> Dict:
    """Load open positions from portfolio.json"""
    try:
        portfolio = load_portfolio()
        positions = portfolio.get("positions", {})
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

def get_total_portfolio_value(client, symbols: List[str]) -> Dict:
    """Calculate total portfolio value in USDT, only pricing relevant assets."""
    if not client:
        return {'total_usdt': 0, 'cash_usdt': 0, 'holdings': {}}
    
    # Build set of base currencies we care about
    base_currencies = set()
    for sym in symbols:
        base = sym.split('/')[0]
        base_currencies.add(base)
    base_currencies.add('USDT')
    # base_currencies.add('USDC')

    try:
        account = client.get_account()
        total_usdt = 0
        cash_usdt = 0
        holdings = {}

        for balance in account['balances']:
            asset = balance['asset']
            free = float(balance['free'])
            if free <= 0:
                continue

            # If asset is USDT, it's cash
            if asset == 'USDT':
                cash_usdt = free
                total_usdt += free
                holdings[asset] = free
                continue

            # If asset is not in our base set, skip pricing (value 0)
            if asset not in base_currencies:
                # Optionally log at debug level
                logger.debug(f"Skipping {asset} (not in trading pairs)")
                holdings[asset] = {
                    'amount': free,
                    'value_usdt': 0,
                    'note': 'not priced (not in trading pairs)'
                }
                continue

            # Try to get price in USDT
            try:
                symbol = f"{asset}USDT"
                ticker = client.get_symbol_ticker(symbol=symbol)
                price = float(ticker['price'])
                value = free * price
                total_usdt += value
                holdings[asset] = {
                    'amount': free,
                    'price_usdt': price,
                    'value_usdt': value
                }
            except Exception as e:
                logger.debug(f"Could not price {asset}: {e}")
                holdings[asset] = {
                    'amount': free,
                    'value_usdt': 0,
                    'note': 'price fetch failed'
                }

        logger.info(f"💰 Total portfolio value: ${total_usdt:,.2f}")
        return {
            'total_usdt': total_usdt,
            'cash_usdt': cash_usdt,
            'holdings': holdings
        }

    except Exception as e:
        logger.error(f"Error calculating portfolio value: {e}")
        return {'total_usdt': 0, 'cash_usdt': 0, 'holdings': {}}


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
        
        # Load open positions from file on startup
        try:
            from portfolio import load_portfolio
            portfolio = load_portfolio()
            self.open_positions = portfolio.get("positions", {})
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

        # Track pending signals to avoid duplicates
        self.last_signals = {}
        
        # Initial balance for return calculation (from first sync)
        self.initial_total_value = None
        
        logger.info(f"🚀 Trading Engine initialized for {self.trading_mode.upper()} mode")
        logger.info(f"📊 Monitoring {len(self.symbols)} symbols on {self.timeframe}")
        logger.info(f"📈 Max positions: {self.max_positions}, Risk per trade: {self.risk_per_trade:.1%}")
        logger.info(f"📂 Loaded {len(self.open_positions)} existing positions")
        
        # Get initial portfolio value
        if self.binance_client:
            portfolio = get_total_portfolio_value(self.binance_client, self.symbols)
            self.initial_total_value = portfolio['total_usdt']
            # Save initial balance to history
            set_initial_balance(self.initial_total_value)
            logger.info(f"💰 Initial portfolio value: ${self.initial_total_value:,.2f}")
    
    def get_portfolio_summary(self) -> Dict:
        """
        Get comprehensive portfolio summary - from portfolio in paper mode, from exchange in live/testnet
        """
        logger.info("💰 Getting portfolio summary...")
        
        # PAPER MODE: Get from portfolio.json
        if self.trading_mode == 'paper':
            try:
                from portfolio import load_portfolio, get_performance_summary
                portfolio = load_portfolio()
                perf = get_performance_summary()
                
                cash = portfolio.get('cash_balance', 0)
                initial = portfolio.get('initial_balance', cash)
                holdings = portfolio.get('holdings', {})
                
                # Calculate total value (cash + holdings at current prices)
                total_value = cash
                holdings_value = 0
                
                # Get current prices for holdings
                current_prices = self.get_current_prices()
                
                for asset, amount in holdings.items():
                    if asset != 'USDT' and amount > 0:
                        symbol = f"{asset}/USDT"
                        price = current_prices.get(symbol, 0)
                        if price > 0:
                            asset_value = amount * price
                            holdings_value += asset_value
                
                total_value += holdings_value
                total_return = total_value - initial
                total_return_pct = (total_return / initial * 100) if initial > 0 else 0
                
                result = {
                    'trading_mode': 'paper',
                    'cash_balance': cash,
                    'holdings': holdings,
                    'holdings_value': holdings_value,
                    'portfolio_value': total_value,
                    'initial_balance': initial,
                    'total_return': total_return,
                    'total_return_pct': total_return_pct,
                    'active_positions': len(self.open_positions),
                    'total_trades': perf.get('total_trades', 0),
                    'winning_trades': perf.get('winning_trades', 0),
                    'win_rate': perf.get('win_rate', 0),
                    'total_pnl': perf.get('total_pnl', 0),
                    'last_sync': datetime.now().isoformat()
                }
                
                logger.info(f"✅ Paper portfolio: ${result['portfolio_value']:,.2f}, Cash: ${cash:.2f}")
                return result
                
            except Exception as e:
                logger.error(f"❌ Error in paper portfolio summary: {e}")
                import traceback
                logger.error(traceback.format_exc())
                return {
                    'trading_mode': 'paper',
                    'cash_balance': self.config.get('starting_balance', 1000),
                    'holdings': {},
                    'holdings_value': 0,
                    'portfolio_value': self.config.get('starting_balance', 1000),
                    'initial_balance': self.config.get('starting_balance', 1000),
                    'total_return': 0,
                    'total_return_pct': 0,
                    'active_positions': 0,
                    'total_trades': 0,
                    'winning_trades': 0,
                    'win_rate': 0,
                    'total_pnl': 0,
                    'last_sync': datetime.now().isoformat()
                }
        
        # LIVE/TESTNET MODE: Get from exchange
        try:
            # Get current portfolio value from exchange
            portfolio = get_total_portfolio_value(self.binance_client, self.symbols)
            logger.info(f"   Portfolio value: ${portfolio.get('total_usdt', 0):,.2f}")
            
            # Calculate returns
            if self.initial_total_value is None:
                self.initial_total_value = portfolio.get('total_usdt', 0)
                logger.info(f"   Initial value set to: ${self.initial_total_value:,.2f}")
            
            total_return = portfolio.get('total_usdt', 0) - self.initial_total_value
            total_return_pct = (total_return / self.initial_total_value * 100) if self.initial_total_value > 0 else 0
            
            # Get performance metrics from history
            try:
                from portfolio import get_performance_summary
                perf = get_performance_summary()
                logger.info(f"   Performance metrics loaded")
            except Exception as e:
                logger.warning(f"   Could not load performance metrics: {e}")
                perf = {'total_trades': 0, 'winning_trades': 0, 'win_rate': 0, 'total_pnl': 0}
            
            result = {
                'trading_mode': self.trading_mode,
                'cash_balance': portfolio.get('cash_usdt', 0),
                'holdings': portfolio.get('holdings', {}),
                'portfolio_value': portfolio.get('total_usdt', 0),
                'initial_balance': self.initial_total_value,
                'total_return': total_return,
                'total_return_pct': total_return_pct,
                'active_positions': len(self.open_positions),
                'total_trades': perf.get('total_trades', 0),
                'winning_trades': perf.get('winning_trades', 0),
                'win_rate': perf.get('win_rate', 0),
                'total_pnl': perf.get('total_pnl', 0),
                'last_sync': datetime.now().isoformat()
            }
            
            logger.info(f"✅ Portfolio summary: ${result['portfolio_value']:,.2f}, {result['active_positions']} positions")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error in get_portfolio_summary: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                'trading_mode': self.trading_mode,
                'cash_balance': 0,
                'holdings': {},
                'portfolio_value': 0,
                'initial_balance': self.initial_total_value or 0,
                'total_return': 0,
                'total_return_pct': 0,
                'active_positions': len(self.open_positions),
                'total_trades': 0,
                'winning_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'last_sync': datetime.now().isoformat(),
                'error': str(e)
            }
    
    def get_cash_balance(self) -> float:
        """Get USDT balance - from portfolio in paper mode, from exchange in live/testnet"""
        
        # PAPER MODE: Get from portfolio.json
        if self.trading_mode == 'paper':
            try:
                from portfolio import load_portfolio
                portfolio = load_portfolio()
                cash = portfolio.get('cash_balance', 0)
                logger.debug(f"Paper mode cash balance: ${cash:.2f}")
                return cash
            except Exception as e:
                logger.error(f"Error getting paper cash balance: {e}")
                # Fallback to starting balance from config
                return self.config.get('starting_balance', 1000)
        
        # LIVE/TESTNET MODE: Get from exchange
        if not self.binance_client:
            logger.error("No binance client available")
            return 0
        
        try:
            account = self.binance_client.get_account()
            for balance in account['balances']:
                if balance['asset'] == 'USDT':
                    return float(balance['free'])
            logger.debug("No USDT balance found")
            return 0
        except Exception as e:
            logger.error(f"Error fetching USDT balance: {e}")
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
        Check stop losses and take profits for all open positions
        Returns True if any positions were closed
        """
        if not self.open_positions:
            return False
        
        current_prices = self.get_current_prices()
        positions_closed = False
        
        for symbol, position in list(self.open_positions.items()):
            current_price = current_prices.get(symbol)
            
            if not current_price:
                logger.warning(f"⚠️ Could not get price for {symbol}, skipping stop check")
                continue
            
            # Check stop loss and take profit
            if position['side'] == 'long':
                if current_price <= position['stop_loss']:
                    logger.info(f"🛑 Stop loss triggered for {symbol} at ${current_price:.2f}")
                    self.close_position(symbol, current_price, "stop_loss")
                    positions_closed = True
                elif current_price >= position['take_profit']:
                    logger.info(f"🎯 Take profit triggered for {symbol} at ${current_price:.2f}")
                    self.close_position(symbol, current_price, "take_profit")
                    positions_closed = True
                    
            elif position['side'] == 'short':
                if current_price >= position['stop_loss']:
                    logger.info(f"🛑 Stop loss triggered for {symbol} at ${current_price:.2f}")
                    self.close_position(symbol, current_price, "stop_loss")
                    positions_closed = True
                elif current_price <= position['take_profit']:
                    logger.info(f"🎯 Take profit triggered for {symbol} at ${current_price:.2f}")
                    self.close_position(symbol, current_price, "take_profit")
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
        base_currency = symbol.split('/')[0]
        
        # Calculate PnL
        if position['side'] == 'long':
            pnl = (exit_price - entry_price) * amount
            pnl_pct = (exit_price / entry_price - 1) * 100
        else:  # short
            pnl = (entry_price - exit_price) * amount
            pnl_pct = (1 - exit_price / entry_price) * 100
        
        # PAPER MODE - update portfolio balance
        if self.trading_mode == 'paper':
            logger.info(f"📄 PAPER CLOSE: {position['side'].upper()} {amount:.6f} {symbol} at ${exit_price:.2f}")
            
            # Update paper portfolio balance
            from portfolio import update_paper_balance
            action = "sell" if position['side'] == 'long' else "buy"  # Reverse action
            update_paper_balance(base_currency, amount, exit_price, action)
        
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
            'mode': self.trading_mode
        })
        
        # Send notification
        if has_notifier:
            try:
                asyncio.create_task(notifier.send_trade_notification({
                    'symbol': symbol,
                    'side': 'SELL' if position['side'] == 'long' else 'COVER',
                    'price': exit_price,
                    'amount': amount,
                    'pnl': pnl,
                    'reason': reason,
                    'mode': self.trading_mode
                }))
            except:
                pass
        
        logger.info(f"✅ Closed {symbol}: PnL ${pnl:.2f} ({pnl_pct:+.1f}%) ({self.trading_mode})")
        return True

    def open_position(self, symbol: str, side: str, entry_price: float, 
                 units: float, stop_loss: float, take_profit: float) -> bool:
        """Open a new position with stop loss and take profit"""
        # DEBUG: Log everything
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
        
        # ===== EXECUTION BASED ON MODE =====
        execution_success = False
        
        # PAPER MODE
        if self.trading_mode == 'paper':
            logger.info(f"📄 PAPER TRADE: {side.upper()} {units:.6f} {symbol} at ${entry_price:.2f}")

            # Update paper portfolio balance
            from portfolio import update_paper_balance
            action = "buy" if side == 'long' else "sell"
            result = update_paper_balance(base_currency, units, entry_price, action)

            if result is None and action == "buy":
                logger.error("❌ Paper buy failed - insufficient funds")
                return False

            execution_success = True
        
        # LIVE/TESTNET MODE
        elif self.trading_mode in ['live', 'testnet'] and self.binance_client:
            try:
                binance_symbol = symbol.replace('/', '')
                
                if side == 'long':
                    # Check if we have enough USDT
                    usdt_balance = get_usdt_balance(self.binance_client)
                    cost = units * entry_price
                    
                    logger.info(f"   USDT balance: ${usdt_balance:.2f}")
                    logger.info(f"   Required: ${cost:.2f}")
                    
                    if usdt_balance < cost:
                        logger.error(f"❌ Insufficient USDT balance: have ${usdt_balance:.2f}, need ${cost:.2f}")
                        return False
                    
                    # VALIDATE AND ADJUST QUANTITY
                    is_valid, adjusted_units, error = self.validate_and_adjust_order(symbol, units)
                    
                    if not is_valid:
                        logger.error(f"❌ Order validation failed: {error}")
                        return False
                    
                    if adjusted_units != units:
                        logger.info(f"🔄 Quantity adjusted from {units} to {adjusted_units}")
                        units = adjusted_units
                    
                    # Place market buy order
                    logger.info(f"📤 Placing market BUY order for {units} {binance_symbol}...")
                    order = self.binance_client.order_market_buy(
                        symbol=binance_symbol,
                        quantity=round(units, 6)
                    )
                    logger.info(f"✅ Live BUY order executed: {order['orderId']}")
                    execution_success = True
                    
                elif side == 'short':
                    # Check if we have the asset to sell
                    base_balance = get_asset_balance(self.binance_client, base_currency)
                    
                    logger.info(f"   {base_currency} balance: {base_balance:.6f}")
                    logger.info(f"   Required: {units:.6f}")
                    
                    if base_balance < units:
                        logger.error(f"❌ Insufficient {base_currency} balance: have {base_balance:.6f}, need {units:.6f}")
                        return False
                    
                    # VALIDATE AND ADJUST QUANTITY
                    is_valid, adjusted_units, error = self.validate_and_adjust_order(symbol, units)
                    
                    if not is_valid:
                        logger.error(f"❌ Order validation failed: {error}")
                        return False
                    
                    if adjusted_units != units:
                        logger.info(f"🔄 Quantity adjusted from {units:.6f} to {adjusted_units:.6f}")
                        units = adjusted_units
                    
                    # Place market sell order
                    logger.info(f"📤 Placing market SELL order for {units} {binance_symbol}...")
                    order = self.binance_client.order_market_sell(
                        symbol=binance_symbol,
                        quantity=round(units, 6)
                    )
                    logger.info(f"✅ Live SELL order executed for SHORT: {order['orderId']}")
                    execution_success = True
                    
            except Exception as e:
                logger.error(f"❌ Failed to execute live order: {e}")
                return False
        
        else:
            logger.warning("⚠️ No valid trading mode or binance client")
            return False
        
        # ===== ONLY IF EXECUTION SUCCEEDED =====
        if execution_success:
            # Add to open positions
            self.open_positions[symbol] = {
                'side': side,
                'amount': units,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
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
                'mode': self.trading_mode
            })
            
            # Send notification
            if has_notifier:
                try:
                    asyncio.create_task(notifier.send_trade_notification({
                        'symbol': symbol,
                        'side': 'SHORT' if side == 'short' else 'LONG',
                        'price': entry_price,
                        'amount': units,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'mode': self.trading_mode
                    }))
                except:
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
                            stop_loss: float, take_profit: float) -> Tuple[bool, str]:
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
                    stop_loss=stop_loss if stop_loss else price * 0.95,
                    take_profit=take_profit if take_profit else price * 1.05
                )
                return success, "Paper limit order simulated"
            
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
        # Check if we can take new positions
        current_positions = len(self.open_positions)
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
                # Skip if already in position
                if symbol in self.open_positions:
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
                    # For example, skip if market is too volatile
                    if "Volatile" in regime and self.trading_mode != 'paper':
                        logger.debug(f"⏭️ Skipping {symbol} due to volatile regime")
                        continue
                        
                except Exception as e:
                    logger.debug(f"Could not detect regime for {symbol}: {e}")
                    regime = "unknown"
                
                # Get available balance for shorts (base currency)
                available_balance = None
                base_currency = symbol.split('/')[0]
                
                # PAPER MODE: Get from portfolio
                if self.trading_mode == 'paper':
                    try:
                        from portfolio import load_portfolio
                        portfolio = load_portfolio()
                        holdings = portfolio.get('holdings', {})
                        available_balance = holdings.get(base_currency, 0)
                    except Exception as e:
                        logger.debug(f"Could not get paper {base_currency} balance: {e}")
                
                # LIVE/TESTNET MODE: Get from exchange
                elif self.trading_mode in ['live', 'testnet'] and self.binance_client:
                    try:
                        account = self.binance_client.get_account()
                        for balance in account['balances']:
                            if balance['asset'] == base_currency:
                                available_balance = float(balance['free'])
                                break
                    except Exception as e:
                        logger.debug(f"Could not get {base_currency} balance: {e}")
                
                # Generate signal with symbol and balance info
                try:
                    signal = generate_trade_signal(
                        df, 
                        cash_balance, 
                        self.risk_per_trade,
                        symbol=symbol,
                        trading_engine=self,
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
                            'regime': regime  # Add regime to signal data
                        })
                        self.last_signals[symbol] = signal_key
                        
                        # Log with regime info
                        logger.info(f"✅ {symbol}: {signal.get('signal_type', 'SIGNAL').upper()} {signal['side']} signal at ${signal['entry']:.2f} (Regime: {regime})")
                    else:
                        logger.info(f"⏭️ {symbol}: Duplicate signal skipped")
                        
            except Exception as e:
                logger.error(f"❌ Error scanning {symbol}: {e}")
        
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
            
            # Check if we already have a position
            if symbol in self.open_positions:
                logger.warning(f"⚠️ Already in position for {symbol}")
                return False
            
            # Check max positions
            if len(self.open_positions) >= self.max_positions:
                logger.warning(f"⚠️ At max positions ({self.max_positions})")
                return False
            
            # Different checks based on side
            if signal['side'] == 'long':
                # For LONG positions, check USDT balance
                cash = self.get_cash_balance()
                cost = signal['units'] * signal['entry']
                logger.info(f"   Cash: ${cash:.2f}, Cost: ${cost:.2f}")
                
                if cash < cost:
                    logger.warning(f"⚠️ Insufficient funds: Need ${cost:.2f}, have ${cash:.2f}")
                    return False
                
                # Check minimum order value
                min_order = 10
                if cost < min_order:
                    logger.warning(f"⚠️ Order value ${cost:.2f} below minimum ${min_order}")
                    return False
                    
            elif signal['side'] == 'short':
                # For SHORT positions, check base currency balance
                base_currency = symbol.split('/')[0]
                
                # PAPER MODE: Check portfolio
                if self.trading_mode == 'paper':
                    try:
                        from portfolio import load_portfolio
                        portfolio = load_portfolio()
                        holdings = portfolio.get('holdings', {})
                        asset_balance = holdings.get(base_currency, 0)
                        logger.info(f"   Paper {base_currency} balance: {asset_balance:.6f}, Required: {signal['units']:.6f}")
                        
                        if asset_balance < signal['units']:
                            logger.warning(f"⚠️ Insufficient {base_currency}: have {asset_balance:.6f}, need {signal['units']:.6f}")
                            return False
                    except Exception as e:
                        logger.error(f"❌ Error checking paper {base_currency} balance: {e}")
                        return False
                
                # LIVE/TESTNET MODE: Check exchange
                elif self.trading_mode in ['live', 'testnet'] and self.binance_client:
                    try:
                        account = self.binance_client.get_account()
                        asset_balance = 0
                        for balance in account['balances']:
                            if balance['asset'] == base_currency:
                                asset_balance = float(balance['free'])
                                break
                        
                        logger.info(f"   {base_currency} balance: {asset_balance:.6f}, Required: {signal['units']:.6f}")
                        
                        if asset_balance < signal['units']:
                            logger.warning(f"⚠️ Insufficient {base_currency}: have {asset_balance:.6f}, need {signal['units']:.6f}")
                            return False
                            
                    except Exception as e:
                        logger.error(f"❌ Error checking {base_currency} balance: {e}")
                        return False
                else:
                    logger.warning(f"⚠️ Cannot check short balance - no balance source available")
                    return False
            
            # Execute the position opening
            result = self.open_position(
                symbol=symbol,
                side=signal['side'],
                entry_price=signal['entry'],
                units=signal['units'],
                stop_loss=signal['stop_loss'],
                take_profit=signal['take_profit']
            )
            
            if result:
                logger.info(f"✅ Successfully opened position for {symbol}")
            else:
                logger.warning(f"❌ Failed to open position for {symbol}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error executing signal: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

trading_engine = TradingEngine()
