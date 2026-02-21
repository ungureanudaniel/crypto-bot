# modules/trade_engine.py - REFACTORED FOR NEW STRATEGY
import asyncio
import logging
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from data_feed import data_feed
from strategy_tools import generate_trade_signal
from portfolio import load_portfolio, save_portfolio, update_position, get_summary
from config_loader import config

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
logger.info(f"✅ Config loaded: {CONFIG.get('trading_mode', 'paper')}")

# -------------------------------------------------------------------
# LIVE TRADING SUPPORT
# -------------------------------------------------------------------
def get_real_binance_client():
    """Get real Binance client for live/testnet trading"""
    try:
        from binance.client import Client
        
        trading_mode = CONFIG.get('trading_mode', 'paper').lower()
        api_key = CONFIG.get('binance_api_key', '')
        api_secret = CONFIG.get('binance_api_secret', '')
        
        if not api_key or not api_secret:
            logger.warning("⚠️ No API keys for real trading")
            return None
        
        if trading_mode == 'testnet':
            client = Client(api_key=api_key, api_secret=api_secret, testnet=True)
            logger.info("✅ Connected to Binance Testnet")
        else:
            client = Client(api_key=api_key, api_secret=api_secret)
            logger.info("✅ Connected to Binance Live")
        
        logger.info(f"🌐 API URL: {client.API_URL}")
        return client
    except ImportError:
        logger.error("❌ python-binance not installed for live trading")
        return None
    except Exception as e:
        logger.error(f"❌ Failed to connect to Binance: {e}")
        return None

# -------------------------------------------------------------------
# TRADING ENGINE
# -------------------------------------------------------------------
class TradingEngine:
    """Universal trading engine for paper/live/testnet trading"""
    
    def __init__(self):
        self.config = CONFIG
        self.data_feed = data_feed
        self.trading_mode = self.config.get('trading_mode', 'paper').lower()
        self.symbols = self.config.get('coins', ['BTC/USDC', 'ETH/USDC'])
        self.timeframe = self.config.get('trading_timeframe', '15m')
        self.max_positions = self.config.get('max_positions', 3)
        self.risk_per_trade = self.config.get('risk_per_trade', 0.02)
        
        # Initialize real trading client if needed
        self.binance_client = None
        if self.trading_mode in ['live', 'testnet']:
            self.binance_client = get_real_binance_client()
            if not self.binance_client:
                logger.warning("⚠️ Could not initialize Binance client, falling back to paper")
                self.trading_mode = 'paper'
        
        # Track pending signals to avoid duplicates
        self.last_signals = {}
        
        logger.info(f"🚀 Trading Engine initialized for {self.trading_mode.upper()} mode")
        logger.info(f"📊 Monitoring {len(self.symbols)} symbols on {self.timeframe}")
        logger.info(f"📈 Max positions: {self.max_positions}, Risk per trade: {self.risk_per_trade:.1%}")
    
    def get_current_prices(self) -> Dict[str, float]:
        """Get current prices for all symbols"""
        prices = {}
        for symbol in self.symbols:
            try:
                price = self.data_feed.get_price(symbol)
                if price and price > 0:
                    prices[symbol] = price
            except Exception as e:
                logger.debug(f"Error getting price for {symbol}: {e}")
        return prices
    
    def check_stop_losses(self) -> bool:
        """
        Check stop losses and take profits for all open positions
        Returns True if any positions were closed
        """
        portfolio = load_portfolio()
        positions = portfolio.get('positions', {})
        
        if not positions:
            return False
        
        current_prices = self.get_current_prices()
        positions_closed = False
        
        for symbol, position in list(positions.items()):
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
        portfolio = load_portfolio()
        positions = portfolio.get('positions', {})
        
        if symbol not in positions:
            logger.warning(f"⚠️ No position found for {symbol}")
            return False
        
        position = positions[symbol]
        amount = position['amount']
        base_currency = symbol.split('/')[0]
        
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
                # Still update paper tracking even if live order fails
        
        # Update portfolio (paper tracking)
        pnl = update_position(base_currency, "sell", amount, exit_price)
        
        if pnl is not None:
            # Record trade in history
            trade_history = portfolio.get('trade_history', [])
            trade_history.append({
                'symbol': symbol,
                'action': 'close',
                'side': position['side'],
                'amount': amount,
                'entry_price': position['entry_price'],
                'exit_price': exit_price,
                'pnl': pnl,
                'pnl_pct': ((exit_price / position['entry_price']) - 1) * 100 if position['side'] == 'long' else (1 - (exit_price / position['entry_price'])) * 100,
                'timestamp': datetime.now().isoformat(),
                'reason': reason,
                'mode': self.trading_mode
            })
            
            # Remove from active positions
            del positions[symbol]
            portfolio['positions'] = positions
            portfolio['trade_history'] = trade_history
            save_portfolio(portfolio)
            
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
            
            logger.info(f"✅ Closed {symbol}: PnL ${pnl:.2f} ({self.trading_mode})")
            return True
        
        return False

    def open_position(self, symbol: str, side: str, entry_price: float, 
                 units: float, stop_loss: float, take_profit: float, auto_stop: bool = True) -> bool:
        """Open a new position with stop loss and take profit"""
        portfolio = load_portfolio()
        
        # Validation checks
        if units <= 0:
            logger.warning(f"⚠️ Invalid units: {units}")
            return False
        
        if entry_price <= 0:
            logger.warning(f"⚠️ Invalid entry price: {entry_price}")
            return False
        
        # Check if already in position
        if symbol in portfolio.get('positions', {}):
            logger.info(f"⏭️ Already in position for {symbol}")
            return False
        
        # Check max positions
        if len(portfolio.get('positions', {})) >= self.max_positions:
            logger.info(f"⏭️ At max positions ({self.max_positions})")
            return False
        
        base_currency = symbol.split('/')[0]
        
        # For SHORT positions in live/testnet mode
        if self.trading_mode in ['live', 'testnet'] and self.binance_client:
            try:
                binance_symbol = symbol.replace('/', '')
                
                if side == 'long':
                    # BUY order for long positions
                    order = self.binance_client.order_market_buy(
                        symbol=binance_symbol,
                        quantity=round(units, 6)
                    )
                    logger.info(f"📤 Live BUY order executed: {order['orderId']}")
                    
                elif side == 'short':
                    # For SHORT positions on spot exchange, we need to:
                    # 1. Check if we have the base currency to sell
                    # 2. Place a market SELL order
                    
                    # First, check if we have the asset to sell
                    account = self.binance_client.get_account()
                    asset_balance = next(
                        (b for b in account['balances'] if b['asset'] == base_currency),
                        {'free': '0'}
                    )
                    
                    if float(asset_balance['free']) < units:
                        logger.error(f"❌ Insufficient {base_currency} balance for short: have {asset_balance['free']}, need {units}")
                        return False
                    
                    # Place market sell order
                    order = self.binance_client.order_market_sell(
                        symbol=binance_symbol,
                        quantity=round(units, 6)
                    )
                    logger.info(f"📤 Live SELL order executed for SHORT: {order['orderId']}")
                
            except Exception as e:
                logger.error(f"❌ Failed to execute live order: {e}")
                return False
        
        # For PAPER mode, just update the portfolio
        if side == 'long':
            # For long positions, we spend quote currency (USDC)
            cost = units * entry_price
            if portfolio.get('cash_balance', 0) < cost:
                logger.warning(f"Insufficient funds: Need ${cost:.2f}")
                return False
            
            # Update cash balance (spend USDC)
            update_position(base_currency, "buy", units, entry_price)
            
        elif side == 'short':
            # For short positions, we need to HAVE the base currency to sell
            # In paper mode, we'll simulate having it
            holdings = portfolio.get('holdings', {})
            current_holdings = holdings.get(base_currency, 0)
            
            if current_holdings < units:
                logger.warning(f"Insufficient {base_currency} for short: have {current_holdings}, need {units}")
                return False
            
            # Update holdings (sell the asset)
            update_position(base_currency, "sell", units, entry_price)
        
        # Add to positions (tracking)
        positions = portfolio.get('positions', {})
        positions[symbol] = {
            'side': side,
            'amount': units,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'entry_time': datetime.now().isoformat(),
            'mode': self.trading_mode
        }
        
        # Record trade in history
        trade_history = portfolio.get('trade_history', [])
        trade_history.append({
            'symbol': symbol,
            'action': 'open',
            'side': side,
            'amount': units,
            'price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'timestamp': datetime.now().isoformat(),
            'mode': self.trading_mode
        })
        
        portfolio['positions'] = positions
        portfolio['trade_history'] = trade_history
        save_portfolio(portfolio)
        
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
    
    def scan_and_trade(self) -> List[Dict]:
        """
        Scan all symbols for trading signals
        Returns list of signals found
        """
        signals_found = []
        portfolio = load_portfolio()
        
        # Check if we can take new positions
        current_positions = len(portfolio.get('positions', {}))
        if current_positions >= self.max_positions:
            logger.info(f"⏭️ At max positions ({current_positions}/{self.max_positions})")
            return signals_found
        
        slots_available = self.max_positions - current_positions
        logger.info(f"🔍 Scanning {len(self.symbols)} symbols for signals ({slots_available} slots available)...")
        
        for symbol in self.symbols:
            try:
                # Skip if already in position
                if symbol in portfolio.get('positions', {}):
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
                
                # Get current equity for position sizing
                equity = portfolio.get('cash_balance', 0)
                if equity <= 10:  # Need at least $10
                    logger.warning(f"⚠️ Low equity: ${equity:.2f}")
                    continue
                
                # Generate signal
                try:
                    signal = generate_trade_signal(df, equity, self.risk_per_trade)
                except Exception as e:
                    logger.error(f"❌ Error generating signal for {symbol}: {e}")
                    continue
                
                if signal:
                    # Check for duplicate signals (avoid same signal twice)
                    signal_key = f"{symbol}_{signal['signal_type']}"
                    
                    # Simple duplicate prevention (don't take same signal type twice in a row)
                    if signal_key != self.last_signals.get(symbol):
                        signals_found.append({
                            'symbol': symbol,
                            'signal': signal
                        })
                        self.last_signals[symbol] = signal_key
                        logger.info(f"✅ {symbol}: {signal['signal_type'].upper()} {signal['side']} signal at ${signal['entry']:.2f}")
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
            symbol = signal_data['symbol']
            signal = signal_data['signal']
            
            return self.open_position(
                symbol=symbol,
                side=signal['side'],
                entry_price=signal['entry'],
                units=signal['units'],
                stop_loss=signal['stop_loss'],
                take_profit=signal['take_profit']
            )
            
        except Exception as e:
            logger.error(f"❌ Error executing signal: {e}")
            return False
    
    def get_portfolio_summary(self) -> Dict:
        """
        Get comprehensive portfolio summary
        """
        current_prices = self.get_current_prices()
        portfolio = load_portfolio()
        
        cash = portfolio.get('cash_balance', 0)
        positions = portfolio.get('positions', {})
        trade_history = portfolio.get('trade_history', [])
        
        # Calculate total value
        total_value = cash
        positions_value = 0
        positions_pnl = 0
        
        for symbol, position in positions.items():
            current_price = current_prices.get(symbol, position.get('entry_price', 0))
            position_value = position['amount'] * current_price
            position_cost = position['amount'] * position['entry_price']
            
            positions_value += position_value
            
            if position['side'] == 'long':
                pnl = position_value - position_cost
            else:
                pnl = position_cost - position_value
            
            positions_pnl += pnl
        
        total_value += positions_value
        
        # Calculate performance metrics
        initial_balance = portfolio.get('initial_balance', total_value)
        total_return = total_value - initial_balance
        total_return_pct = (total_return / initial_balance * 100) if initial_balance > 0 else 0
        
        # Win rate from trade history
        closed_trades = [t for t in trade_history if t['action'] == 'close']
        winning_trades = [t for t in closed_trades if t.get('pnl', 0) > 0]
        
        win_rate = (len(winning_trades) / len(closed_trades) * 100) if closed_trades else 0
        
        return {
            'trading_mode': self.trading_mode,
            'cash_balance': cash,
            'positions_value': positions_value,
            'portfolio_value': total_value,
            'initial_balance': initial_balance,
            'total_return': total_return,
            'total_return_pct': total_return_pct,
            'active_positions': len(positions),
            'total_trades': len(closed_trades),
            'winning_trades': len(winning_trades),
            'win_rate': win_rate,
            'positions_pnl': positions_pnl
        }
    
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
    
    # Test scan
    print("\n🔍 Testing scan...")
    signals = trading_engine.scan_and_trade()
    
    if signals:
        print(f"✅ Found {len(signals)} signals:")
        for s in signals:
            sig = s['signal']
            print(f"   {s['symbol']}: {sig['signal_type']} {sig['side']} @ ${sig['entry']:.2f}")
    else:
        print("❌ No signals found")
    
    # Test portfolio summary
    print("\n💰 Portfolio Summary:")
    summary = trading_engine.get_portfolio_summary()
    print(f"   Value: ${summary['portfolio_value']:,.2f}")
    print(f"   Cash: ${summary['cash_balance']:,.2f}")
    print(f"   Positions: {summary['active_positions']}")
    print(f"   Return: {summary['total_return_pct']:+.1f}%")