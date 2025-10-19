import json
import logging
import ccxt
import pandas as pd
from datetime import datetime, timedelta
from strategy_tools import generate_trade_signal

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

TRADING_FEE = 0.001  # 0.1%
MAX_POSITION_SIZE = 0.1  # Max 10% of portfolio per trade
MAX_DRAWDOWN = 0.15  # Max 15% portfolio drawdown
DAILY_LOSS_LIMIT = 0.05  # Max 5% daily loss

# -------------------------------------------------------------------
# Portfolio handling
# -------------------------------------------------------------------
def load_portfolio():
    """Load portfolio from JSON file"""
    try:
        with open("portfolio.json", "r") as f:
            portfolio = json.load(f)
        logging.info("Portfolio loaded successfully")
        return portfolio
    except FileNotFoundError:
        logging.warning("Portfolio file not found, creating default portfolio")
        return create_default_portfolio()
    except Exception as e:
        logging.error(f"Error loading portfolio: {e}")
        return create_default_portfolio()

def save_portfolio(portfolio):
    """Save portfolio to JSON file"""
    try:
        with open("portfolio.json", "w") as f:
            json.dump(portfolio, f, indent=2)
        logging.info("Portfolio saved successfully")
    except Exception as e:
        logging.error(f"Error saving portfolio: {e}")

def create_default_portfolio():
    """Create a default portfolio structure"""
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    portfolio = {
        "cash_balance": config.get('starting_balance', 1000),
        "holdings": {},
        "positions": {},
        "trade_history": [],
        "pending_orders": [],
        "initial_balance": config.get('starting_balance', 1000),
        "performance_metrics": {
            "total_trades": 0,
            "winning_trades": 0,
            "total_pnl": 0.0,
            "max_drawdown": 0.0
        }
    }
    save_portfolio(portfolio)
    return portfolio

# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------
def load_config():
    """Load configuration from JSON file"""
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
        return config
    except Exception as e:
        logging.error(f"Error loading config: {e}")
        return {}

# -------------------------------------------------------------------
# Trading Mode Detection
# -------------------------------------------------------------------
def get_trading_mode():
    """Get current trading mode from config"""
    try:
        config = load_config()
        return 'live' if config.get('live_trading', False) else 'paper'
    except:
        return 'paper'  # Always default to paper for safety

def is_paper_trading():
    """Check if we're in paper trading mode"""
    return get_trading_mode() == 'paper'

def get_exchange():
    """Get Binance exchange connection"""
    try:
        config = load_config()
        
        api_key = config.get('binance_api_key', '')
        api_secret = config.get('binance_api_secret', '')
        
        if not api_key or not api_secret:
            logging.warning("Binance API keys not configured")
            return None
        
        exchange = ccxt.binance({
            'apiKey': api_key,
            'secret': api_secret,
            'sandbox': True,  # Use testnet for safety
            'options': {
                'defaultType': 'spot',
            },
        })
        return exchange
    except Exception as e:
        logging.error(f"Error creating exchange connection: {e}")
        return None

# -------------------------------------------------------------------
# RISK MANAGEMENT FUNCTIONS
# -------------------------------------------------------------------
def calculate_position_size(symbol, price, portfolio, risk_pct=0.02, stop_loss_pct=0.05):
    """
    Calculate position size based on risk management rules
    """
    cash_balance = portfolio['cash_balance']
    
    # Rule 1: Maximum position size (10% of portfolio)
    max_by_portfolio = cash_balance * MAX_POSITION_SIZE
    
    # Rule 2: Risk-based position sizing
    risk_amount = cash_balance * risk_pct
    stop_loss_distance = price * stop_loss_pct
    max_by_risk = risk_amount / stop_loss_distance if stop_loss_distance > 0 else 0
    
    # Rule 3: Minimum position value ($10 to avoid dust)
    min_position = 10 / price
    
    # Use the most conservative value
    position_size = min(max_by_portfolio, max_by_risk)
    units = max(position_size / price, min_position)
    
    # Final check: don't exceed available cash
    max_affordable = cash_balance * 0.95 / price  # Leave 5% buffer
    units = min(units, max_affordable)
    
    logging.info(f"Position sizing for {symbol}: {units:.6f} units (${units * price:.2f})")
    return units

def check_portfolio_health():
    """
    Check overall portfolio health and enforce limits
    """
    try:
        portfolio = load_portfolio()
        initial_balance = portfolio.get('initial_balance', portfolio['cash_balance'])
        
        # Calculate current portfolio value
        current_value = portfolio['cash_balance']
        for symbol, position in portfolio.get('positions', {}).items():
            current_value += position['amount'] * position['entry_price']  # Simplified
        
        # Check drawdown limit
        drawdown = (initial_balance - current_value) / initial_balance
        if drawdown > MAX_DRAWDOWN:
            logging.warning(f"🛑 Portfolio drawdown {drawdown:.1%} exceeds limit {MAX_DRAWDOWN:.1%}")
            return False
        
        return True
        
    except Exception as e:
        logging.error(f"Error in portfolio health check: {e}")
        return False

def check_stop_losses(current_prices):
    """
    Check and execute stop losses for all open positions
    """
    try:
        portfolio = load_portfolio()
        positions = portfolio.get('positions', {})
        positions_closed = False
        
        for symbol, position in list(positions.items()):
            current_price = current_prices.get(symbol)
            if current_price is None:
                continue
                
            stop_loss = position['stop_loss']
            take_profit = position['take_profit']
            
            # Check stop loss (for long positions)
            if position['side'] == 'long' and current_price <= stop_loss:
                logging.info(f"🛑 Stop loss triggered for {symbol} at ${current_price:.2f}")
                close_position(symbol, current_price)
                positions_closed = True
                
            # Check take profit (for long positions)  
            elif position['side'] == 'long' and current_price >= take_profit:
                logging.info(f"🎯 Take profit triggered for {symbol} at ${current_price:.2f}")
                close_position(symbol, current_price)
                positions_closed = True
        
        return positions_closed
        
    except Exception as e:
        logging.error(f"Error checking stop losses: {e}")
        return False

def close_position(symbol, current_price):
    """
    Close an open position and update portfolio
    """
    try:
        portfolio = load_portfolio()
        positions = portfolio.get('positions', {})
        
        if symbol not in positions:
            logging.warning(f"No position found for {symbol}")
            return False
            
        position = positions[symbol]
        entry_price = position['entry_price']
        amount = position['amount']
        
        # Calculate PnL
        if position['side'] == 'long':
            pnl = (current_price - entry_price) * amount
        else:
            pnl = (entry_price - current_price) * amount
        
        # Add cash back
        portfolio['cash_balance'] += amount * current_price
        
        # Record trade history
        trade = {
            'action': 'close_position',
            'coin': symbol,
            'amount': amount,
            'entry_price': entry_price,
            'exit_price': current_price,
            'pnl': pnl,
            'timestamp': datetime.now().isoformat(),
            'side': position['side'],
            'reason': 'stop_loss' if current_price <= position['stop_loss'] else 'take_profit'
        }
        portfolio.setdefault('trade_history', []).append(trade)
        
        # Update performance metrics
        if pnl > 0:
            portfolio['performance_metrics']['winning_trades'] += 1
        portfolio['performance_metrics']['total_trades'] += 1
        portfolio['performance_metrics']['total_pnl'] += pnl
        
        # Remove position
        del portfolio['positions'][symbol]
        save_portfolio(portfolio)
        
        logging.info(f"Closed position for {symbol}: PnL ${pnl:.2f}")
        return True
        
    except Exception as e:
        logging.error(f"Error closing position for {symbol}: {e}")
        return False

def calculate_dynamic_stop_loss(symbol, entry_price, side='long', atr_multiplier=2):
    """
    Calculate dynamic stop loss and take profit
    """
    try:
        if side == 'long':
            stop_loss = entry_price * 0.95  # 5% stop loss
            take_profit = entry_price * 1.10  # 10% take profit
        else:
            stop_loss = entry_price * 1.05
            take_profit = entry_price * 0.90
        
        return stop_loss, take_profit
    except Exception as e:
        logging.error(f"Error calculating stop loss for {symbol}: {e}")
        # Fallback values
        if side == 'long':
            return entry_price * 0.95, entry_price * 1.10
        else:
            return entry_price * 1.05, entry_price * 0.90

# -------------------------------------------------------------------
# ORDER EXECUTION FUNCTIONS
# -------------------------------------------------------------------
def execute_limit_order(symbol, side, amount, price):
    """Execute a manual limit order with risk checks"""
    try:
        portfolio = load_portfolio()
        
        # Risk check before executing
        if not check_portfolio_health():
            return False, "Portfolio health check failed - order blocked"
        
        if side == 'buy':
            total_cost = amount * price
            
            # Additional risk check: position size
            if total_cost > portfolio['cash_balance'] * MAX_POSITION_SIZE:
                return False, f"Position size too large. Max: {MAX_POSITION_SIZE*100}% of portfolio"
            
            if portfolio['cash_balance'] >= total_cost:
                # Deduct cash
                portfolio['cash_balance'] -= total_cost
                
                # Add to holdings
                base_currency = symbol.split('/')[0]
                portfolio['holdings'][base_currency] = portfolio['holdings'].get(base_currency, 0) + amount
                
                # Add to trade history
                trade = {
                    'action': 'buy_limit',
                    'coin': symbol,
                    'amount': amount,
                    'price': price,
                    'total': total_cost,
                    'timestamp': datetime.now().isoformat(),
                    'type': 'manual_limit'
                }
                portfolio.setdefault('trade_history', []).append(trade)
                
                save_portfolio(portfolio)
                logging.info(f"Manual limit BUY executed: {amount} {symbol} @ ${price}")
                return True, "Limit buy order executed successfully"
            else:
                return False, f"Insufficient funds. Need ${total_cost:.2f}, have ${portfolio['cash_balance']:.2f}"
        
        elif side == 'sell':
            base_currency = symbol.split('/')[0]
            current_holdings = portfolio['holdings'].get(base_currency, 0)
            
            if current_holdings >= amount:
                total_value = amount * price
                
                # Add cash and remove holdings
                portfolio['cash_balance'] += total_value
                portfolio['holdings'][base_currency] -= amount
                
                # Clean up zero holdings
                if portfolio['holdings'][base_currency] == 0:
                    del portfolio['holdings'][base_currency]
                
                # Add to trade history
                trade = {
                    'action': 'sell_limit',
                    'coin': symbol,
                    'amount': amount,
                    'price': price,
                    'total': total_value,
                    'timestamp': datetime.now().isoformat(),
                    'type': 'manual_limit'
                }
                portfolio.setdefault('trade_history', []).append(trade)
                
                save_portfolio(portfolio)
                logging.info(f"Manual limit SELL executed: {amount} {symbol} @ ${price}")
                return True, "Limit sell order executed successfully"
            else:
                return False, f"Insufficient {base_currency}. Need {amount}, have {current_holdings}"
        else:
            return False, "Invalid side. Must be 'buy' or 'sell'"
    
    except Exception as e:
        logging.error(f"Error executing limit order: {e}")
        return False, str(e)

def check_pending_orders(current_prices):
    """Check if any pending limit orders should be executed based on current prices"""
    try:
        portfolio = load_portfolio()
        pending_orders = portfolio.get('pending_orders', [])
        
        if not pending_orders:
            return
        
        executed_orders = []
        
        for order in pending_orders[:]:  # Iterate over copy
            symbol = order['symbol']
            current_price = current_prices.get(symbol)
            
            if current_price is None:
                continue
                
            if order['side'] == 'buy' and current_price <= order['price']:
                # Risk check before execution
                if check_portfolio_health():
                    success, message = execute_limit_order(
                        order['symbol'], 'buy', order['amount'], order['price']
                    )
                    if success:
                        executed_orders.append(order)
                        logging.info(f"Pending buy limit order executed for {symbol} @ {order['price']}")
            
            elif order['side'] == 'sell' and current_price >= order['price']:
                success, message = execute_limit_order(
                    order['symbol'], 'sell', order['amount'], order['price']
                )
                if success:
                    executed_orders.append(order)
                    logging.info(f"Pending sell limit order executed for {symbol} @ {order['price']}")
        
        # Remove executed orders
        if executed_orders:
            portfolio['pending_orders'] = [
                order for order in pending_orders 
                if order not in executed_orders
            ]
            save_portfolio(portfolio)
            
    except Exception as e:
        logging.error(f"Error checking pending orders: {e}")

# -------------------------------------------------------------------
# MAIN TRADE EXECUTION
# -------------------------------------------------------------------
def execute_trade(symbol, regime, price):
    """
    Execute trade with comprehensive risk management
    """
    try:
        portfolio = load_portfolio()
        
        # Skip automatic trading if manual orders exist for this symbol
        pending_orders = portfolio.get('pending_orders', [])
        symbol_pending_orders = [order for order in pending_orders if order['symbol'] == symbol]
        
        if symbol_pending_orders:
            logging.info(f"[{symbol}] Skipping automatic trade - manual orders pending")
            return
        
        # Portfolio health check
        if not check_portfolio_health():
            logging.warning(f"[{symbol}] Portfolio health check failed - skipping trade")
            return
        
        positions = portfolio.get('positions', {})
        cash_balance = portfolio['cash_balance']
        config = load_config()

        risk_pct = config.get('risk_pct', 0.02)  # 2% risk per trade

        # Check if position already exists
        if symbol in positions:
            logging.info(f"[{symbol}] Position already open. Skipping new trade.")
            return

        # Trading logic based on regime
        if regime == 2:  # Breakout regime - BUY
            side = "long"
            # Calculate position size with risk management
            units = calculate_position_size(symbol, price, portfolio, risk_pct)
            
        elif regime == 1:  # Trending regime - could go either way
            # For now, skip trending or add your logic
            logging.info(f"[{symbol}] Trending regime - no clear signal")
            return
        else:  # Rangebound regime - no trade
            logging.info(f"[{symbol}] Rangebound regime - no trade")
            return

        # Check if enough cash
        required_cash = price * units
        if required_cash > cash_balance:
            logging.warning(f"[{symbol}] Not enough cash for trade. Required: ${required_cash:.2f}, Available: ${cash_balance:.2f}")
            return

        # Calculate dynamic stop loss and take profit
        stop_loss, take_profit = calculate_dynamic_stop_loss(symbol, price, side)

        # Save the position with risk management data
        positions[symbol] = {
            "side": side,
            "amount": units,
            "entry_price": price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "entry_time": datetime.now().isoformat(),
            "risk_pct": risk_pct
        }

        # Deduct cash
        portfolio['cash_balance'] -= required_cash
        portfolio['positions'] = positions
        
        # Track initial balance if not set
        if 'initial_balance' not in portfolio:
            portfolio['initial_balance'] = portfolio['cash_balance'] + required_cash
        
        save_portfolio(portfolio)

        # Log the trade
        logging.info(f"🚀 AUTOMATIC {side.upper()} {symbol} - Amount: {units:.6f} @ ${price:.2f} - Value: ${required_cash:.2f}")
        
    except Exception as e:
        logging.error(f"Error executing trade for {symbol}: {e}")

# Telegram notification function (will be imported from notifier in actual use)
def send_telegram_message(message):
    """Send telegram notification using notifier.py"""
    try:
        from notifier import send_telegram_message as send_telegram
        return send_telegram(message)
    except ImportError as e:
        logging.error(f"Failed to import notifier: {e}")
        logging.info(f"Telegram notification: {message}")
        return False
    except Exception as e:
        logging.error(f"Failed to send telegram message: {e}")
        return False