# modules/papertrade_engine.py - COMPLETE VERSION
import json
import logging
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional

from modules.data_feed import data_feed
from modules.strategy_tools import generate_trade_signal
from modules.portfolio import load_portfolio, save_portfolio, update_position, get_summary
from services.notifier import notifier

logger = logging.getLogger(__name__)

# Load config
with open('config.json', 'r') as f:
    config = json.load(f)

class PaperTradeEngine:
    """Complete paper trading engine"""
    
    def __init__(self):
        self.data_feed = data_feed
        self.symbols = config.get('coins', ['BTC/USDT', 'ETH/USDT'])
        self.timeframe = config.get('trading_timeframe', '15m')
        self.max_positions = config.get('max_positions', 3)
        self.risk_per_trade = config.get('risk_per_trade', 0.02)
        
        logger.info("📊 Paper Trade Engine initialized")
    
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
        """Check stop losses for all positions - returns True if any positions were closed"""
        portfolio = load_portfolio()
        positions = portfolio.get('positions', {})
        current_prices = self.get_current_prices()
        
        if not positions:
            return False
        
        positions_closed = False
        
        for symbol, position in list(positions.items()):
            current_price = current_prices.get(symbol)
            if not current_price:
                logger.warning(f"No current price for {symbol}, skipping stop loss check")
                continue
            
            # Check stop loss (for long positions)
            if position['side'] == 'long' and current_price <= position['stop_loss']:
                logger.info(f"🛑 Stop loss triggered for {symbol} at ${current_price:.2f}")
                self.close_position(symbol, current_price, "stop_loss")
                positions_closed = True
            
            # Check take profit (for long positions)  
            elif position['side'] == 'long' and current_price >= position['take_profit']:
                logger.info(f"🎯 Take profit triggered for {symbol} at ${current_price:.2f}")
                self.close_position(symbol, current_price, "take_profit")
                positions_closed = True
        
        return positions_closed
    
    def close_position(self, symbol: str, exit_price: float, reason: str) -> bool:
        """Close a position with notification"""
        portfolio = load_portfolio()
        positions = portfolio.get('positions', {})
        
        if symbol not in positions:
            logger.warning(f"No position found for {symbol}")
            return False
        
        position = positions[symbol]
        amount = position['amount']
        base_currency = symbol.split('/')[0]
        
        # Update portfolio
        pnl = update_position(base_currency, "sell", amount, exit_price)
        
        if pnl is not None:
            # Record trade
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
                'reason': reason
            })
            
            # Remove position
            del positions[symbol]
            portfolio['positions'] = positions
            portfolio['trade_history'] = trade_history
            save_portfolio(portfolio)
            
            # Send notification
            notifier.send_trade_notification({
                'symbol': symbol,
                'side': 'SELL',
                'price': exit_price,
                'amount': amount,
                'pnl': pnl
            })
            
            logger.info(f"✅ Closed {symbol}: PnL ${pnl:.2f}")
            return True
        
        return False
    
    def open_position(self, symbol: str, side: str, entry_price: float, 
                     units: float, stop_loss: float, take_profit: float) -> bool:
        """Open a new position with notification"""
        portfolio = load_portfolio()
        
        # Check if already in position
        if symbol in portfolio.get('positions', {}):
            logger.info(f"Already in position for {symbol}")
            return False
        
        # Check cash balance
        cost = units * entry_price
        if portfolio.get('cash_balance', 0) < cost:
            logger.warning(f"Insufficient funds for {symbol}: Need ${cost:.2f}, have ${portfolio.get('cash_balance', 0):.2f}")
            return False
        
        base_currency = symbol.split('/')[0]
        
        # Update portfolio (buy action)
        update_position(base_currency, "buy", units, entry_price)
        
        # Add to positions
        positions = portfolio.get('positions', {})
        positions[symbol] = {
            'side': side,
            'amount': units,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'entry_time': datetime.now().isoformat()
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
            'take_profit': take_profit
        })
        
        portfolio['positions'] = positions
        portfolio['trade_history'] = trade_history
        save_portfolio(portfolio)
        
        # Send notification
        notifier.send_trade_notification({
            'symbol': symbol,
            'side': 'BUY',
            'price': entry_price,
            'amount': units,
            'stop_loss': stop_loss,
            'take_profit': take_profit
        })
        
        logger.info(f"✅ Opened {side} position: {symbol} {units:.6f} @ ${entry_price:.2f}")
        logger.info(f"   Stop Loss: ${stop_loss:.2f}, Take Profit: ${take_profit:.2f}")
        return True
    
    def scan_and_trade(self) -> List[Dict]:
        """Scan for signals and return potential trades"""
        signals_found = []
        portfolio = load_portfolio()
        
        # Check if at max positions
        current_positions = len(portfolio.get('positions', {}))
        if current_positions >= self.max_positions:
            logger.info(f"📊 At max positions ({self.max_positions}) - skipping scan")
            return signals_found
        
        logger.info(f"🔍 Scanning {len(self.symbols)} symbols for trading signals...")
        
        for symbol in self.symbols:
            try:
                # Skip if already in position
                if symbol in portfolio.get('positions', {}):
                    continue
                
                # Get data
                df = self.data_feed.get_ohlcv(
                    symbol=symbol,
                    interval=self.timeframe,
                    limit=200
                )
                
                if df.empty or len(df) < 50:
                    logger.warning(f"Insufficient data for {symbol}")
                    continue
                
                # Generate signal using strategy tools
                equity = portfolio.get('cash_balance', 0)
                if equity <= 0:
                    logger.warning(f"No cash available for trading")
                    continue
                
                signal = generate_trade_signal(df, equity, self.risk_per_trade)
                
                if signal:
                    logger.info(f"📈 Signal found for {symbol}: {signal['side'].upper()}")
                    
                    signals_found.append({
                        'symbol': symbol,
                        'signal': signal,
                        'data': df
                    })
                    
            except Exception as e:
                logger.error(f"Error scanning {symbol}: {e}")
                continue
        
        logger.info(f"Found {len(signals_found)} potential trading signals")
        return signals_found
    
    def execute_signal(self, signal_data: Dict) -> bool:
        """Execute a trading signal"""
        try:
            symbol = signal_data['symbol']
            signal = signal_data['signal']
            
            logger.info(f"🔄 Executing {signal['side'].upper()} signal for {symbol}")
            
            success = self.open_position(
                symbol=symbol,
                side=signal['side'],
                entry_price=signal['entry'],
                units=signal['units'],
                stop_loss=signal['stop_loss'],
                take_profit=signal['take_profit']
            )
            
            if success:
                logger.info(f"✅ Successfully executed signal for {symbol}")
            else:
                logger.warning(f"⚠️ Failed to execute signal for {symbol}")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Error executing signal: {e}")
            return False

    def place_limit_order(self, symbol: str, side: str, amount: float, price: float) -> tuple[bool, str]:
        """Place a limit order"""
        try:
            from modules.portfolio import load_portfolio, save_portfolio
            
            portfolio = load_portfolio()
            
            # Validate inputs
            if side not in ['buy', 'sell']:
                return False, "Side must be 'buy' or 'sell'"
            
            if amount <= 0 or price <= 0:
                return False, "Amount and price must be positive"
            
            # Check balance for buy orders
            if side == 'buy':
                total_cost = amount * price
                if portfolio.get('cash_balance', 0) < total_cost:
                    return False, f"Insufficient funds. Need ${total_cost:.2f}, have ${portfolio.get('cash_balance', 0):.2f}"
            
            # Check holdings for sell orders
            if side == 'sell':
                coin = symbol.split('/')[0]
                current_holdings = portfolio.get('holdings', {}).get(coin, 0)
                if current_holdings < amount:
                    return False, f"Insufficient {coin}. Need {amount}, have {current_holdings}"
            
            # Create order
            order_id = f"limit_{symbol}_{side}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            pending_orders = portfolio.get('pending_orders', [])
            pending_orders.append({
                'id': order_id,
                'symbol': symbol,
                'side': side,
                'amount': amount,
                'price': price,
                'timestamp': datetime.now().isoformat(),
                'status': 'pending',
                'type': 'limit'
            })
            
            portfolio['pending_orders'] = pending_orders
            save_portfolio(portfolio)
            
            return True, f"Limit order placed: {symbol} {side} {amount} @ ${price:.2f}"
            
        except Exception as e:
            logger.error(f"Error placing limit order: {e}")
            return False, str(e)

    def execute_trade(self, symbol: str, regime: str, price: float) -> bool:
        """Execute a manual trade based on regime prediction"""
        portfolio = load_portfolio()
        
        # Skip if already in position
        if symbol in portfolio.get('positions', {}):
            logger.warning(f"Already in position for {symbol}")
            return False
        
        # Determine trade direction from regime
        regime_lower = regime.lower()
        
        if any(word in regime_lower for word in ['breakout', 'trending up', 'bullish']):
            side = 'long'
        elif any(word in regime_lower for word in ['trending down', 'bearish']):
            side = 'short'
        else:  # Range-bound or uncertain
            logger.info(f"Skipping {symbol} - regime: {regime}")
            return False
        
        # Calculate position size (simplified)
        equity = portfolio.get('cash_balance', 0)
        position_size_pct = 0.02  # 2% of equity for manual trades
        risk_amount = equity * position_size_pct
        
        # Ensure minimum trade size
        min_trade = 10  # $10 minimum
        risk_amount = max(risk_amount, min_trade)
        
        units = risk_amount / price
        
        # Calculate stop loss and take profit
        if side == 'long':
            stop_loss = price * 0.95  # 5% stop loss
            take_profit = price * 1.10  # 10% take profit
        else:  # short
            stop_loss = price * 1.05  # 5% stop loss
            take_profit = price * 0.90  # 10% take profit
        
        # Open position
        return self.open_position(
            symbol=symbol,
            side=side,
            entry_price=price,
            units=units,
            stop_loss=stop_loss,
            take_profit=take_profit
        )

    def get_portfolio_summary(self) -> Dict:
        """Get portfolio summary"""
        current_prices = self.get_current_prices()
        portfolio = load_portfolio()
        
        cash = portfolio.get('cash_balance', 0)
        holdings = portfolio.get('holdings', {})
        positions = portfolio.get('positions', {})
        
        # Calculate total value
        total_value = cash
        
        # Add holdings value
        for asset, amount in holdings.items():
            if asset != 'USDT' and asset != 'USDC':
                symbol = f"{asset}/USDT"
                price = current_prices.get(symbol, 0)
                total_value += amount * price
        
        # Add positions value
        for symbol, position in positions.items():
            current_price = current_prices.get(symbol, position.get('entry_price', 0))
            total_value += position['amount'] * current_price
        
        initial_balance = portfolio.get('initial_balance', total_value)
        total_return_pct = ((total_value - initial_balance) / initial_balance * 100) if initial_balance > 0 else 0
        
        # Performance metrics
        perf_metrics = portfolio.get('performance_metrics', {})
        total_trades = perf_metrics.get('total_trades', 0)
        winning_trades = perf_metrics.get('winning_trades', 0)
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        return {
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
        """Check portfolio health - returns True if healthy"""
        try:
            summary = self.get_portfolio_summary()
            
            # Check drawdown
            max_drawdown = config.get('max_drawdown', 0.05)
            current_drawdown = -summary['total_return_pct'] / 100 if summary['total_return_pct'] < 0 else 0
            
            if current_drawdown > max_drawdown:
                logger.warning(f"⚠️ Portfolio drawdown {current_drawdown:.1%} exceeds limit {max_drawdown:.1%}")
                return False
            
            # Check if enough cash for minimum trade
            min_trade_size = 10  # $10 minimum
            if summary['cash_balance'] < min_trade_size:
                logger.warning(f"⚠️ Insufficient cash for minimum trade (${min_trade_size})")
                return False
            
            logger.info(f"✅ Portfolio healthy: ${summary['portfolio_value']:,.2f} ({summary['total_return_pct']:+.1f}%)")
            return True
            
        except Exception as e:
            logger.error(f"Error checking portfolio health: {e}")
            return False

# Create singleton instance
paper_engine = PaperTradeEngine()