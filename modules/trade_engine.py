# modules/trade_engine.py - FIXED VERSION
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from data_feed import data_feed
from strategy_tools import generate_trade_signal
from config_loader import config
from portfolio import add_trade, get_performance_summary, set_initial_balance
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
    
    # Build set of base currencies we care about (from symbols parameter)
    # symbols contains e.g. ["SOL/USDT", "BTC/USDT", ...]
    base_currencies = set()
    for sym in symbols:
        base = sym.split('/')[0]
        base_currencies.add(base)
    # Always include the quote currency (USDT) and maybe USDC if used
    base_currencies.add('USDT')
    # If you also hold USDC and want it valued, add 'USDC'
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
        self.trading_mode = self.config.get('trading_mode', 'testnet').lower()
        self.symbols = self.config.get('coins', ['BTC/USDC', 'ETH/USDC'])
        self.timeframe = self.config.get('trading_timeframe', '15m')
        self.max_positions = self.config.get('max_positions', 3)
        self.risk_per_trade = self.config.get('risk_per_trade', 0.02)
        
        # Track open positions in memory only (not file)
        self.open_positions = {}  # symbol -> position details
        
        # Initialize real trading client if needed
        self.binance_client = get_binance_client()

        # Track pending signals to avoid duplicates
        self.last_signals = {}
        
        # Initial balance for return calculation (from first sync)
        self.initial_total_value = None
        
        logger.info(f"🚀 Trading Engine initialized for {self.trading_mode.upper()} mode")
        logger.info(f"📊 Monitoring {len(self.symbols)} symbols on {self.timeframe}")
        logger.info(f"📈 Max positions: {self.max_positions}, Risk per trade: {self.risk_per_trade:.1%}")
        
        # Get initial portfolio value
        if self.binance_client:
            portfolio = self.get_total_portfolio_value()
            self.initial_total_value = portfolio['total_usdt']
            # Save initial balance to history
            set_initial_balance(self.initial_total_value)
            logger.info(f"💰 Initial portfolio value: ${self.initial_total_value:,.2f}")
    
    def get_portfolio_summary(self) -> Dict:
        """
        Get comprehensive portfolio summary directly from exchange
        """
        portfolio = self.get_total_portfolio_value()
        
        # Calculate returns
        if self.initial_total_value is None:
            self.initial_total_value = portfolio['total_usdt']
        
        total_return = portfolio['total_usdt'] - self.initial_total_value
        total_return_pct = (total_return / self.initial_total_value * 100) if self.initial_total_value > 0 else 0
        
        perf = get_performance_summary()
        
        return {
            'trading_mode': self.trading_mode,
            'cash_balance': portfolio['cash_usdt'],
            'holdings': portfolio['holdings'],
            'portfolio_value': portfolio['total_usdt'],
            'initial_balance': self.initial_total_value,
            'total_return': total_return,
            'total_return_pct': total_return_pct,
            'active_positions': len(self.open_positions),
            'total_trades': perf['total_trades'],
            'winning_trades': perf['winning_trades'],
            'win_rate': perf['win_rate'],
            'total_pnl': perf['total_pnl'],
            'last_sync': datetime.now().isoformat()
        }
    
    def get_cash_balance(self) -> float:
        """Get USDC/USDT balance directly from exchange"""
        if self.trading_mode in ['live', 'testnet'] and self.binance_client:
            return get_usdt_balance(self.binance_client)
        else:
            logger.info("Not able to fetch real cash balance!")
            return 0.0

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
        
        # Calculate PnL
        if position['side'] == 'long':
            pnl = (exit_price - entry_price) * amount
            pnl_pct = (exit_price / entry_price - 1) * 100
        else:  # short
            pnl = (entry_price - exit_price) * amount
            pnl_pct = (1 - exit_price / entry_price) * 100
        
        # Execute REAL trade if in live/testnet mode
        if self.trading_mode in ['live', 'testnet'] and self.binance_client:
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
        
        # Check minimum order value (Binance often requires $10 minimum)
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
        
        # For LIVE/TESTNET mode
        if self.trading_mode in ['live', 'testnet'] and self.binance_client:
            try:
                binance_symbol = symbol.replace('/', '')
                
                if side == 'long':
                    # Check if we have enough USDT
                    usdt_balance = get_usdt_balance(self.binance_client)  # You need this function
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
                        # Recalculate cost
                        cost = units * entry_price
                    
                    # Place market buy order
                    logger.info(f"📤 Placing market BUY order for {adjusted_units} {binance_symbol}...")
                    order = self.binance_client.order_market_buy(
                        symbol=binance_symbol,
                        quantity=adjusted_units
                    )
                    logger.info(f"✅ Live BUY order executed: {order['orderId']}")
                    
                elif side == 'short':
                    # Check if we have the asset to sell
                    asset_balance = get_asset_balance(self.binance_client, base_currency)
                    
                    logger.info(f"   {base_currency} balance: {asset_balance:.6f}")
                    logger.info(f"   Required: {units:.6f}")
                    
                    if asset_balance < units:
                        logger.error(f"❌ Insufficient {base_currency} balance: have {asset_balance:.6f}, need {units:.6f}")
                        return False
                    
                    # VALIDATE AND ADJUST QUANTITY for the asset we're selling
                    is_valid, adjusted_units, error = self.validate_and_adjust_order(symbol, units)
                    
                    if not is_valid:
                        logger.error(f"❌ Order validation failed: {error}")
                        return False
                    
                    if adjusted_units != units:
                        logger.info(f"🔄 Quantity adjusted from {units} to {adjusted_units}")
                        units = adjusted_units
                    
                    # Place market sell order
                    logger.info(f"📤 Placing market SELL order for {units} {binance_symbol}...")
                    order = self.binance_client.order_market_sell(
                        symbol=binance_symbol,
                        quantity=round(units, 6)  # Round to 6 decimal places
                    )
                    logger.info(f"✅ Live SELL order executed for SHORT: {order['orderId']}")
                
            except Exception as e:
                logger.error(f"❌ Failed to execute live order: {e}")
                return False
        
            # Add to open positions (in-memory tracking)
            self.open_positions[symbol] = {
                'side': side,
                'amount': units,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'entry_time': datetime.now().isoformat(),
                'mode': self.trading_mode
            }
            
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

        # connection issues
        else:
            logger.warning("⚠️ Possible binance connection issues - no live or testnet orders executed")
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
        
        # Get current cash balance from exchange
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
                
                # Generate signal
                try:
                    signal = generate_trade_signal(df, cash_balance, self.risk_per_trade)
                except Exception as e:
                    logger.error(f"❌ Error generating signal for {symbol}: {e}")
                    continue
                
                if signal:
                    # Check for duplicate signals
                    signal_key = f"{symbol}_{signal.get('signal_type', 'unknown')}"
                    
                    if signal_key != self.last_signals.get(symbol):
                        signals_found.append({
                            'symbol': symbol,
                            'signal': signal
                        })
                        self.last_signals[symbol] = signal_key
                        logger.info(f"✅ {symbol}: {signal.get('signal_type', 'SIGNAL').upper()} {signal['side']} signal at ${signal['entry']:.2f}")
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
            
            # Check cash balance before executing
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
            
            # Check if we already have a position
            if symbol in self.open_positions:
                logger.warning(f"⚠️ Already in position for {symbol}")
                return False
            
            # Check max positions
            if len(self.open_positions) >= self.max_positions:
                logger.warning(f"⚠️ At max positions ({self.max_positions})")
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
    
    def get_total_portfolio_value(self):
        """Get total portfolio value in USDT from exchange"""
        logger.info("Calculating total portfolio value...")
        total_usdt = 0
        cash_usdt = 0
        holdings = {}
        
        if not self.binance_client:
            logger.warning("No binance client available")
            return {'total_usdt': 0, 'cash_usdt': 0, 'holdings': {}}
        
        try:
            account = self.binance_client.get_account()
            logger.info(f"Account has {len(account['balances'])} balances")
            
            for balance in account['balances']:
                asset = balance['asset']
                free = float(balance['free'])
                if free <= 0:
                    continue
                
                logger.debug(f"Processing {asset} balance: {free}")
                
                if asset == 'USDT':
                    cash_usdt = free
                    total_usdt += free
                    holdings[asset] = free
                else:
                    try:
                        # Skip known test assets that don't have a USDT pair
                        if asset in ['这是测试币', '456', 'BTC', 'ETH', 'BNB']:
                            logger.debug(f"Skipping {asset} – no price needed")
                            holdings[asset] = {'amount': free, 'price': 0, 'value': 0}
                            continue
                        
                        symbol = f"{asset}USDT"
                        logger.debug(f"Fetching price for {symbol}...")
                        ticker = self.binance_client.get_symbol_ticker(symbol=symbol)
                        price = float(ticker['price'])
                        value = free * price
                        total_usdt += value
                        holdings[asset] = {'amount': free, 'price': price, 'value': value}
                        logger.debug(f"{asset} price: {price}, value: {value}")
                    except Exception as e:
                        logger.debug(f"Could not get price for {asset}: {e}")
                        holdings[asset] = {'amount': free, 'price': 0, 'value': 0}
            
            logger.info(f"Total portfolio value: ${total_usdt:.2f}")
            return {'total_usdt': total_usdt, 'cash_usdt': cash_usdt, 'holdings': holdings}
        except Exception as e:
            logger.error(f"Error calculating portfolio value: {e}")
            return {'total_usdt': 0, 'cash_usdt': 0, 'holdings': {}}
    
    def check_portfolio_health(self) -> bool:
        """
        Check if portfolio is healthy and can continue trading
        """
        try:
            summary = self.get_portfolio_summary()
            
            # Check drawdown
            max_drawdown = self.config.get('max_drawdown', 0.05)
            current_drawdown = -summary['total_return_pct'] / 100 if summary['total_return_pct'] < 0 else 0
            
            if current_drawdown > max_drawdown:
                logger.warning(f"⚠️ Drawdown {current_drawdown:.1%} > {max_drawdown:.1%}")
                return False
            
            # Check minimum cash
            min_trade_size = 10
            if summary['cash_balance'] < min_trade_size:
                logger.warning(f"⚠️ Low cash: ${summary['cash_balance']:.2f}")
                return False
            
            logger.info(f"✅ Portfolio healthy: ${summary['portfolio_value']:,.2f} (Return: {summary['total_return_pct']:+.1f}%)")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error checking health: {e}")
            return False

# Create singleton instance
trading_engine = TradingEngine()

if __name__ == "__main__":
    # Test the engine
    print(f"\n🧪 Testing Trading Engine")
    print("=" * 50)
    print(f"Mode: {trading_engine.trading_mode}")
    print(f"Symbols: {len(trading_engine.symbols)}")
    print(f"Timeframe: {trading_engine.timeframe}")
    print(f"Max positions: {trading_engine.max_positions}")
    print(f"Risk per trade: {trading_engine.risk_per_trade:.1%}")
    
    # Get portfolio summary directly from exchange
    print("\n💰 Portfolio Summary (from exchange):")
    summary = trading_engine.get_portfolio_summary()
    print(f"   Value: ${summary['portfolio_value']:,.2f}")
    print(f"   Cash: ${summary['cash_balance']:,.2f}")
    print(f"   Holdings: {len(summary.get('holdings', {}))}")
    print(f"   Positions: {summary['active_positions']}")
    print(f"   Return: {summary['total_return_pct']:+.1f}%")
    print(f"   Win Rate: {summary['win_rate']:.1f}%")
    
    # Test scan
    print("\n🔍 Testing scan...")
    signals = trading_engine.scan_and_trade()
    
    if signals:
        print(f"✅ Found {len(signals)} signals:")
        for s in signals:
            sig = s['signal']
            print(f"   {s['symbol']}: {sig.get('signal_type', 'SIGNAL')} {sig['side']} @ ${sig['entry']:.2f}")
    else:
        print("❌ No signals found")