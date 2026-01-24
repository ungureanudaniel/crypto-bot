# modules/trading_engine.py - UNIVERSAL VERSION
import sys
import os
import logging
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from modules.data_feed import *
from modules.strategy_tools import generate_trade_signal
from modules.portfolio import load_portfolio, save_portfolio, update_position, get_summary
try:
    from services.notifier import notifier
    has_notifier = True
except ImportError:
    has_notifier = False
    logger = logging.getLogger(__name__)
    logger.warning("⚠️ Notifier service not available")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# CONFIG LOADING
# -------------------------------------------------------------------
try:
    # Add parent directory to path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config_loader import config
    CONFIG = config.config
    logger.info(f"✅ Config loaded: {CONFIG.get('trading_mode', 'paper')}")
except ImportError:
    logger.warning("⚠️ Could not import config_loader, using defaults")
    CONFIG = {'trading_mode': 'paper', 'testnet': False, 'rate_limit_delay': 0.5}
logging.info("🔧 Configuration loaded for data feed. Trading mode: %s", CONFIG.get('trading_mode', 'paper'))

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
        
        client = Client(api_key=api_key, api_secret=api_secret)
        
        if trading_mode == 'testnet':
            client.API_URL = 'https://testnet.binance.vision'
            logger.info("✅ Connected to Binance Testnet")
        else:
            logger.info("✅ Connected to Binance Live")
        
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
        
        logger.info(f"🚀 Trading Engine initialized for {self.trading_mode.upper()} mode")
    
    def get_current_prices(self) -> Dict[str, float]:
        """Get current prices for all symbols"""
        prices = {}
        for symbol in self.symbols:
            try:
                price = self.data_feed.get_price(symbol)
                if price:
                    prices[symbol] = price
            except Exception as e:
                logger.error(f"Error getting price for {symbol}: {e}")
        return prices
    
    def check_stop_losses(self) -> bool:
        """Check stop losses for all positions"""
        portfolio = load_portfolio()
        positions = portfolio.get('positions', {})
        current_prices = self.get_current_prices()
        
        if not positions:
            return False
        
        positions_closed = False
        
        for symbol, position in list(positions.items()):
            current_price = current_prices.get(symbol)
            if not current_price:
                continue
            
            if position['side'] == 'long' and current_price <= position['stop_loss']:
                logger.info(f"🛑 Stop loss triggered for {symbol}")
                self.close_position(symbol, current_price, "stop_loss")
                positions_closed = True
            elif position['side'] == 'long' and current_price >= position['take_profit']:
                logger.info(f"🎯 Take profit triggered for {symbol}")
                self.close_position(symbol, current_price, "take_profit")
                positions_closed = True
        
        return positions_closed
    
    def close_position(self, symbol: str, exit_price: float, reason: str) -> bool:
        """Close a position"""
        portfolio = load_portfolio()
        positions = portfolio.get('positions', {})
        
        if symbol not in positions:
            return False
        
        position = positions[symbol]
        amount = position['amount']
        base_currency = symbol.split('/')[0]
        
        # Execute REAL trade if in live/testnet mode
        if self.trading_mode in ['live', 'testnet'] and self.binance_client:
            try:
                # Convert to Binance format
                binance_symbol = symbol.replace('/', '')
                
                # Place market sell order
                if position['side'] == 'long':
                    order = self.binance_client.order_market_sell(
                        symbol=binance_symbol,
                        quantity=round(amount, 6)
                    )
                    logger.info(f"📤 Live SELL order executed: {order}")
            except Exception as e:
                logger.error(f"❌ Failed to execute live sell order: {e}")
        
        # Update portfolio (paper tracking)
        pnl = update_position(base_currency, "sell", amount, exit_price)
        
        if pnl is not None:
            trade_history = portfolio.get('trade_history', [])
            trade_history.append({
                'symbol': symbol,
                'action': 'close',
                'side': position['side'],
                'amount': amount,
                'entry_price': position['entry_price'],
                'exit_price': exit_price,
                'pnl': pnl,
                'timestamp': datetime.now().isoformat(),
                'reason': reason,
                'mode': self.trading_mode
            })
            
            del positions[symbol]
            portfolio['positions'] = positions
            portfolio['trade_history'] = trade_history
            save_portfolio(portfolio)
            
            # Send notification
            if has_notifier:
                notifier.send_trade_notification({
                    'symbol': symbol,
                    'side': 'SELL',
                    'price': exit_price,
                    'amount': amount,
                    'pnl': pnl,
                    'mode': self.trading_mode
                })
            
            logger.info(f"✅ Closed {symbol}: PnL ${pnl:.2f} ({self.trading_mode})")
            return True
        
        return False
    
    def open_position(self, symbol: str, side: str, entry_price: float, 
                     units: float, stop_loss: float, take_profit: float, auto_stop: bool = True) -> bool:
        """Open a new position with auto stop loss"""
        portfolio = load_portfolio()

        # If no stop loss provided and auto_stop is True, calculate one
        if auto_stop and stop_loss is None:
            # Default 5% stop loss
            if side == 'long':
                stop_loss = entry_price * 0.95  # 5% stop
                take_profit = entry_price * 1.10  # 10% target (2:1)
            else:  # short
                stop_loss = entry_price * 1.05  # 5% stop
                take_profit = entry_price * 0.90  # 10% target (2:1)
        
        # Check if already in position
        if symbol in portfolio.get('positions', {}):
            logger.info(f"Already in position for {symbol}")
            return False
        
        # Check cash balance
        cost = units * entry_price
        if portfolio.get('cash_balance', 0) < cost:
            logger.warning(f"Insufficient funds: Need ${cost:.2f}")
            return False
        
        base_currency = symbol.split('/')[0]
        
        # Execute REAL trade if in live/testnet mode
        if self.trading_mode in ['live', 'testnet'] and self.binance_client and side == 'long':
            try:
                # Convert to Binance format
                binance_symbol = symbol.replace('/', '')
                
                # Place market buy order
                order = self.binance_client.order_market_buy(
                    symbol=binance_symbol,
                    quantity=round(units, 6)
                )
                logger.info(f"📤 Live BUY order executed: {order}")
            except Exception as e:
                logger.error(f"❌ Failed to execute live buy order: {e}")
                return False
        
        # Update portfolio (paper tracking)
        update_position(base_currency, "buy", units, entry_price)
        
        # Add to positions
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
        
        # Record trade
        trade_history = portfolio.get('trade_history', [])
        trade_history.append({
            'symbol': symbol,
            'action': 'open',
            'side': side,
            'amount': units,
            'price': entry_price,
            'timestamp': datetime.now().isoformat(),
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'mode': self.trading_mode
        })
        
        portfolio['positions'] = positions
        portfolio['trade_history'] = trade_history
        save_portfolio(portfolio)
        
        # Send notification
        if has_notifier:
            notifier.send_trade_notification({
                'symbol': symbol,
                'side': 'BUY',
                'price': entry_price,
                'amount': units,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'mode': self.trading_mode
            })
        
        logger.info(f"✅ Opened {side} position: {symbol} ({self.trading_mode})")
        return True
    
    def place_limit_order(self, symbol: str, side: str, amount: float, price: float) -> Tuple[bool, str]:
        """Place a limit order"""
        try:
            portfolio = load_portfolio()
            
            if side not in ['buy', 'sell']:
                return False, "Invalid side"
            
            if amount <= 0 or price <= 0:
                return False, "Invalid amount or price"
            
            # Execute REAL limit order if in live/testnet mode
            if self.trading_mode in ['live', 'testnet'] and self.binance_client:
                try:
                    binance_symbol = symbol.replace('/', '')
                    
                    if side == 'buy':
                        order = self.binance_client.order_limit_buy(
                            symbol=binance_symbol,
                            quantity=round(amount, 6),
                            price=str(price)
                        )
                    else:  # sell
                        order = self.binance_client.order_limit_sell(
                            symbol=binance_symbol,
                            quantity=round(amount, 6),
                            price=str(price)
                        )
                    
                    logger.info(f"📤 Live LIMIT {side.upper()} order placed: {order}")
                    
                    # Also track in portfolio
                    pending_orders = portfolio.get('pending_orders', [])
                    pending_orders.append({
                        'id': order['orderId'],
                        'symbol': symbol,
                        'side': side,
                        'amount': amount,
                        'price': price,
                        'timestamp': datetime.now().isoformat(),
                        'status': 'pending',
                        'type': 'limit',
                        'mode': self.trading_mode,
                        'binance_order_id': order['orderId']
                    })
                    
                    portfolio['pending_orders'] = pending_orders
                    save_portfolio(portfolio)
                    
                    return True, f"Live limit order placed: {order['orderId']}"
                    
                except Exception as e:
                    return False, f"Live order failed: {e}"
            
            # Paper trading limit order
            else:
                order_id = f"paper_limit_{symbol}_{side}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                pending_orders = portfolio.get('pending_orders', [])
                pending_orders.append({
                    'id': order_id,
                    'symbol': symbol,
                    'side': side,
                    'amount': amount,
                    'price': price,
                    'timestamp': datetime.now().isoformat(),
                    'status': 'pending',
                    'type': 'limit',
                    'mode': 'paper'
                })
                
                portfolio['pending_orders'] = pending_orders
                save_portfolio(portfolio)
                
                return True, f"Paper limit order placed: {order_id}"
                
        except Exception as e:
            logger.error(f"Error placing limit order: {e}")
            return False, str(e)
    
    def scan_and_trade(self) -> List[Dict]:
        """Scan for signals"""
        signals_found = []
        portfolio = load_portfolio()
        
        if len(portfolio.get('positions', {})) >= self.max_positions:
            logger.info("At max positions")
            return signals_found
        
        logger.info(f"🔍 Scanning {len(self.symbols)} symbols...")
        
        for symbol in self.symbols:
            try:
                if symbol in portfolio.get('positions', {}):
                    continue
                
                df = self.data_feed.get_ohlcv(
                    symbol=symbol,
                    interval=self.timeframe,
                    limit=200
                )
                
                if df.empty or len(df) < 50:
                    continue
                
                equity = portfolio.get('cash_balance', 0)
                if equity <= 0:
                    continue
                
                try:
                    signal = generate_trade_signal(df, equity, self.risk_per_trade)
                except:
                    # Basic signal if strategy_tools fails
                    signal = {
                        'side': 'long',
                        'entry': df['close'].iloc[-1],
                        'units': (equity * self.risk_per_trade) / df['close'].iloc[-1],
                        'stop_loss': df['close'].iloc[-1] * 0.95,
                        'take_profit': df['close'].iloc[-1] * 1.10
                    }
                
                if signal:
                    signals_found.append({
                        'symbol': symbol,
                        'signal': signal,
                        'data': df
                    })
                    
            except Exception as e:
                logger.error(f"Error scanning {symbol}: {e}")
        
        return signals_found
    
    def execute_signal(self, signal_data: Dict) -> bool:
        """Execute a trading signal"""
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
            logger.error(f"Error executing signal: {e}")
            return False
    
    def get_portfolio_summary(self) -> Dict:
        """Get portfolio summary"""
        current_prices = self.get_current_prices()
        portfolio = load_portfolio()
        
        cash = portfolio.get('cash_balance', 0)
        holdings = portfolio.get('holdings', {})
        positions = portfolio.get('positions', {})
        
        total_value = cash
        
        for asset, amount in holdings.items():
            if asset not in ['USDT', 'USDC']:
                symbol = f"{asset}/USDT"
                price = current_prices.get(symbol, 0)
                total_value += amount * price
        
        for symbol, position in positions.items():
            current_price = current_prices.get(symbol, position.get('entry_price', 0))
            total_value += position['amount'] * current_price
        
        initial_balance = portfolio.get('initial_balance', total_value)
        total_return_pct = ((total_value - initial_balance) / initial_balance * 100) if initial_balance > 0 else 0
        
        perf_metrics = portfolio.get('performance_metrics', {})
        total_trades = perf_metrics.get('total_trades', 0)
        winning_trades = perf_metrics.get('winning_trades', 0)
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        return {
            'trading_mode': self.trading_mode,
            'cash_balance': cash,
            'portfolio_value': total_value,
            'initial_balance': initial_balance,
            'total_return_pct': total_return_pct,
            'active_positions': len(positions),
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'win_rate': win_rate,
            'total_pnl': perf_metrics.get('total_pnl', 0),
            'max_drawdown': perf_metrics.get('max_drawdown', 0)
        }
    
    def check_portfolio_health(self) -> bool:
        """Check portfolio health"""
        try:
            summary = self.get_portfolio_summary()
            
            max_drawdown = self.config.get('max_drawdown', 0.05)
            current_drawdown = -summary['total_return_pct'] / 100 if summary['total_return_pct'] < 0 else 0
            
            if current_drawdown > max_drawdown:
                logger.warning(f"⚠️ Drawdown {current_drawdown:.1%} > {max_drawdown:.1%}")
                return False
            
            min_trade_size = 10
            if summary['cash_balance'] < min_trade_size:
                logger.warning(f"⚠️ Low cash: ${summary['cash_balance']:.2f}")
                return False
            
            logger.info(f"✅ Portfolio healthy: ${summary['portfolio_value']:,.2f}")
            return True
            
        except Exception as e:
            logger.error(f"Error checking health: {e}")
            return False

# Create singleton instance
trading_engine = TradingEngine()

# For backward compatibility
trading_engine = trading_engine

if __name__ == "__main__":
    # Test the engine
    print(f"🧪 Testing Trading Engine - Mode: {trading_engine.trading_mode}")
    summary = trading_engine.get_portfolio_summary()
    print(f"Portfolio: ${summary['portfolio_value']:,.2f}")
    print(f"Return: {summary['total_return_pct']:+.1f}%")
    print(f"Active positions: {summary['active_positions']}")