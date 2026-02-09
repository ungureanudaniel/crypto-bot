# services/scheduler.py - WITH PATH FIXES
import logging
import sys
import os
from datetime import datetime

# -------------------------------------------------------------------
# CRITICAL: SETUP PATHS FIRST
# -------------------------------------------------------------------
# Get the absolute path to the project root
current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_file_dir)  # Go up one level from services/

# Add project root to Python path
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'modules'))

# Now we can import modules
try:
    from config_loader import config
    CONFIG = config.config
except ImportError:
    CONFIG = {'trading_mode': 'paper', 'coins': ['BTC/USDC', 'ETH/USDC']}

# -------------------------------------------------------------------
# LOGGING
# -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# HELPER FUNCTIONS WITH PROPER IMPORTS
# -------------------------------------------------------------------
def _import_modules():
    """Import required modules with proper error handling"""
    try:
        from modules.data_feed import fetch_ohlcv
        from modules.regime_switcher import predict_regime
        from modules.trade_engine import trading_engine
        from modules.portfolio import load_portfolio, save_portfolio
        return fetch_ohlcv, predict_regime, trading_engine, load_portfolio, save_portfolio
    except ImportError as e:
        logger.error(f"Failed to import modules: {e}")
        logger.error(f"Python path: {sys.path}")
        logger.error(f"Project root: {project_root}")
        return None, None, None, None, None

# -------------------------------------------------------------------
# REAL SCHEDULER JOBS
# -------------------------------------------------------------------
def check_stop_losses_and_take_profits():
    """Check and execute stop losses"""
    logger.info("🛡️ Checking stop losses...")
    
    # Import inside function to ensure path is set
    fetch_ohlcv, predict_regime, trading_engine, load_portfolio, save_portfolio = _import_modules()
    
    if not fetch_ohlcv or not trading_engine or not load_portfolio:
        logger.error("❌ Cannot import required modules")
        return
    
    try:
        portfolio = load_portfolio()
        positions = portfolio.get('positions', {})
        
        if not positions:
            logger.info("📭 No active positions")
            return
        
        positions_closed = 0
        
        for symbol, position in positions.items():
            try:
                # Get current price
                df = fetch_ohlcv(symbol, "1m", limit=1)
                if df.empty:
                    continue
                
                current_price = df.iloc[-1]['close']
                entry_price = position.get('entry_price', 0)
                
                if entry_price == 0:
                    continue
                
                # Check stop loss (5% loss)
                stop_loss_price = entry_price * 0.95
                if current_price <= stop_loss_price:
                    logger.info(f"🛑 STOP LOSS: {symbol} at ${current_price:.2f}")
                    success = trading_engine.close_position(symbol, current_price, "stop_loss")
                    if success:
                        positions_closed += 1
                
                # Check take profit (10% gain)
                take_profit_price = entry_price * 1.10
                if current_price >= take_profit_price:
                    logger.info(f"🎯 TAKE PROFIT: {symbol} at ${current_price:.2f}")
                    success = trading_engine.close_position(symbol, current_price, "take_profit")
                    if success:
                        positions_closed += 1
                        
            except Exception as e:
                logger.warning(f"Error checking {symbol}: {e}")
        
        if positions_closed > 0:
            logger.info(f"✅ Closed {positions_closed} positions")
            
    except Exception as e:
        logger.error(f"❌ Error in stop loss check: {e}")

def scan_for_trading_signals():
    """Scan for trading opportunities"""
    logger.info("🔍 Scanning for signals...")
    
    fetch_ohlcv, predict_regime, trading_engine, load_portfolio, save_portfolio = _import_modules()
    
    if not fetch_ohlcv or not predict_regime or not load_portfolio:
        logger.error("❌ Cannot import required modules")
        return
    
    try:
        signals_found = 0
        
        for symbol in CONFIG.get('coins', ['BTC/USDC']):
            try:
                # Get data
                df = fetch_ohlcv(symbol, "15m", limit=50)
                if df.empty:
                    continue
                
                # Predict regime
                regime = predict_regime(df)
                
                # Simple signal logic
                if regime == "Bullish":
                    logger.info(f"📈 Bullish: {symbol}")
                    
                    # Get current price
                    current_price = df.iloc[-1]['close']
                    
                    # Check portfolio
                    portfolio = load_portfolio()
                    cash = portfolio.get('cash_balance', 0)
                    positions = len(portfolio.get('positions', {}))
                    
                    # Simple entry logic
                    if cash > 100 and positions < 3:
                        position_size = cash * 0.02
                        units = position_size / current_price
                        
                        logger.info(f"   Signal to BUY {units:.6f} at ${current_price:.2f}")
                        signals_found += 1
                        
            except Exception as e:
                logger.warning(f"Error analyzing {symbol}: {e}")
        
        if signals_found > 0:
            logger.info(f"🎯 Found {signals_found} signals")
        else:
            logger.info("📭 No signals found")
            
    except Exception as e:
        logger.error(f"❌ Error scanning signals: {e}")

def update_portfolio_summary():
    """Update portfolio summary"""
    logger.info("📊 Updating portfolio...")
    
    fetch_ohlcv, predict_regime, trading_engine, load_portfolio, save_portfolio = _import_modules()
    
    if not fetch_ohlcv or not load_portfolio:
        logger.error("❌ Cannot import required modules")
        return
    
    try:
        portfolio = load_portfolio()
        cash = portfolio.get('cash_balance', 0)
        positions = portfolio.get('positions', {})
        
        # Calculate total value
        total_value = cash
        
        for symbol, position in positions.items():
            try:
                df = fetch_ohlcv(symbol, "1m", limit=1)
                if not df.empty:
                    current_price = df.iloc[-1]['close']
                    position_value = position.get('amount', 0) * current_price
                    total_value += position_value
            except:
                continue
        
        # Log summary
        logger.info(f"💰 Portfolio: ${cash:,.2f} cash, {len(positions)} positions")
        logger.info(f"💵 Total Value: ${total_value:,.2f}")
        
        # Send notification every 6 hours
        current_hour = datetime.now().hour
        if current_hour % 6 == 0:
            try:
                from services.notifier import notifier
                notifier.send_message(
                    f"📊 Portfolio Update\n"
                    f"Cash: ${cash:,.2f}\n"
                    f"Total: ${total_value:,.2f}"
                )
            except:
                pass
                
    except Exception as e:
        logger.error(f"❌ Error updating portfolio: {e}")

def check_manual_stops():
    """Check stop losses for manual trades"""
    logger.info("🛡️ Checking manual stop losses...")
    
    try:
        from modules.portfolio import load_portfolio, save_portfolio
        from modules.data_feed import fetch_ohlcv
        from modules.trade_engine import trading_engine
        
        portfolio = load_portfolio()
        positions = portfolio.get('positions', {})
        
        for symbol, position in positions.items():
            # Check if position has stop loss
            if 'stop_loss' not in position:
                continue
                
            # Get current price
            df = fetch_ohlcv(symbol, "1m", limit=1)
            if df.empty:
                continue
                
            current_price = df.iloc[-1]['close']
            stop_loss = position['stop_loss']
            side = position.get('side', 'long')
            
            # Check stop loss
            if (side == 'long' and current_price <= stop_loss) or \
               (side == 'short' and current_price >= stop_loss):
                
                logger.info(f"🛑 Manual stop loss triggered: {symbol}")
                
                # Close position
                success = trading_engine.close_position(symbol, current_price, "stop_loss")
                
                if success:
                    # Send notification
                    try:
                        from services.notifier import notifier
                        notifier.send_message(
                            f"🛑 Stop Loss Executed\n"
                            f"Symbol: {symbol}\n"
                            f"Price: ${current_price:.2f}\n"
                            f"Stop: ${stop_loss:.2f}"
                        )
                    except:
                        pass
                        
    except Exception as e:
        logger.error(f"❌ Check manual stops error: {e}")

def check_pending_orders():
    """Check pending orders"""
    logger.info("📋 Checking pending orders...")
    
    fetch_ohlcv, predict_regime, trading_engine, load_portfolio, save_portfolio = _import_modules()
    
    if not load_portfolio or not fetch_ohlcv or not trading_engine or not save_portfolio:
        logger.error("❌ Cannot import required modules")
        return
    
    try:
        portfolio = load_portfolio()
        pending_orders = portfolio.get('pending_orders', [])
        
        if not pending_orders:
            return
        
        orders_executed = 0
        
        for order in pending_orders[:]:  # Copy for iteration
            symbol = order.get('symbol')
            side = order.get('side', 'buy')
            limit_price = order.get('price', 0)
            
            if not symbol or limit_price == 0:
                continue
            
            try:
                df = fetch_ohlcv(symbol, "1m", limit=1)
                if df.empty:
                    continue
                
                current_price = df.iloc[-1]['close']
                
                # Check limit order conditions
                should_execute = False
                
                if side == 'buy' and current_price <= limit_price:
                    should_execute = True
                elif side == 'sell' and current_price >= limit_price:
                    should_execute = True
                
                if should_execute:
                    logger.info(f"✅ Limit order triggered: {symbol} at ${current_price:.2f}")
                    
                    if side == 'buy':
                        success = trading_engine.open_position(
                            symbol=symbol,
                            side='long',
                            entry_price=current_price,
                            units=order.get('amount', 0),
                            stop_loss=order.get('stop_loss'),
                            take_profit=order.get('take_profit')
                        )
                    else:
                        success = trading_engine.close_position(symbol, current_price, "limit_order")
                    
                    if success:
                        pending_orders.remove(order)
                        orders_executed += 1
                        
            except Exception as e:
                logger.warning(f"Error processing order {symbol}: {e}")
        
        if orders_executed > 0:
            portfolio['pending_orders'] = pending_orders
            save_portfolio(portfolio)
            logger.info(f"📝 Executed {orders_executed} orders")
            
    except Exception as e:
        logger.error(f"❌ Error checking orders: {e}")

# -------------------------------------------------------------------
# GET JOBS FOR TELEGRAM BOT
# -------------------------------------------------------------------
def get_all_jobs():
    """Return all scheduler jobs"""
    return {
        'check_stops': check_stop_losses_and_take_profits,
        'scan_signals': scan_for_trading_signals,
        'update_portfolio': update_portfolio_summary,
        'check_orders': check_pending_orders,

    }

def start_schedulers(bot_data=None):
    """Setup scheduler jobs"""
    logger.info("⏰ Scheduler jobs defined")
    return get_all_jobs()

if __name__ == "__main__":
    print(f"🔧 Project root: {project_root}")
    print(f"🔧 Python path: {sys.path}")
    
    # Test imports
    try:
        from modules.data_feed import fetch_ohlcv
        print("✅ Successfully imported modules.data_feed")
    except ImportError as e:
        print(f"❌ Failed to import: {e}")
    
    # Test jobs
    print("\n🔄 Testing scheduler jobs...")
    jobs = get_all_jobs()
    
    for job_name, job_func in jobs.items():
        print(f"\nTesting {job_name}...")
        job_func()