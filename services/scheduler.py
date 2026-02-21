# services/scheduler.py - FIXED VERSION
import logging
import sys
import os
from datetime import datetime

# -------------------------------------------------------------------
# CRITICAL: SETUP PATHS FIRST
# -------------------------------------------------------------------
current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_file_dir)
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
        from modules.trade_engine import trading_engine
        from modules.portfolio import load_portfolio, save_portfolio
        from modules.data_feed import data_feed
        return trading_engine, load_portfolio, save_portfolio, data_feed
    except ImportError as e:
        logger.error(f"Failed to import modules: {e}")
        logger.error(f"Python path: {sys.path}")
        return None, None, None, None

# -------------------------------------------------------------------
# REAL SCHEDULER JOBS - USING TRADE ENGINE
# -------------------------------------------------------------------

def check_stop_losses_and_take_profits():
    """
    JOB 1: Check stop losses - runs every minute
    Uses trade_engine's built-in stop loss checking
    """
    logger.debug("🛡️ Checking stop losses...")
    
    trading_engine, load_portfolio, save_portfolio, data_feed = _import_modules()
    
    if not trading_engine:
        logger.error("❌ Cannot import trading_engine")
        return
    
    try:
        # Use the trade_engine's own stop loss checker
        positions_closed = trading_engine.check_stop_losses()
        
        if positions_closed:
            logger.info(f"✅ Closed {positions_closed} positions via stop/take profit")
            
    except Exception as e:
        logger.error(f"❌ Error in stop loss check: {e}")

def scan_for_trading_signals():
    """
    JOB 2: Scan for signals - runs every 5 minutes
    Uses trade_engine's scan_and_trade to generate signals with new strategy
    """
    logger.info("🔍 Scanning for trading signals...")
    
    trading_engine, load_portfolio, save_portfolio, data_feed = _import_modules()
    
    if not trading_engine:
        logger.error("❌ Cannot import trading_engine")
        return
    
    try:
        # Check if we can take new positions
        portfolio = load_portfolio()
        current_positions = len(portfolio.get('positions', {}))
        
        if current_positions >= trading_engine.max_positions:
            logger.info(f"⏭️ At max positions ({current_positions}/{trading_engine.max_positions})")
            return
        
        # USE THE TRADE ENGINE'S SCAN METHOD
        signals = trading_engine.scan_and_trade()
        
        if signals:
            logger.info(f"🎯 Found {len(signals)} signals")
            
            # Auto-execute if configured to do so
            if CONFIG.get('auto_execute_signals', False):
                executed = 0
                for signal in signals:
                    success = trading_engine.execute_signal(signal)
                    if success:
                        executed += 1
                
                if executed > 0:
                    logger.info(f"✅ Executed {executed} signals")
                    
                    # Send notification
                    try:
                        from services.notifier import notifier
                        notifier.send_message(f"🤖 Auto-executed {executed} trades")
                    except:
                        pass
        else:
            logger.debug("📭 No signals found")
            
    except Exception as e:
        logger.error(f"❌ Error scanning signals: {e}")

def update_portfolio_summary():
    """
    JOB 3: Update portfolio summary - runs every hour
    Logs current portfolio status
    """
    logger.info("📊 Updating portfolio summary...")
    
    trading_engine, load_portfolio, save_portfolio, data_feed = _import_modules()
    
    if not trading_engine or not load_portfolio:
        logger.error("❌ Cannot import required modules")
        return
    
    try:
        # Get portfolio summary from trade_engine
        summary = trading_engine.get_portfolio_summary()
        
        # Log summary
        logger.info(f"💰 Portfolio: ${summary['portfolio_value']:,.2f}")
        logger.info(f"   Cash: ${summary['cash_balance']:,.2f}")
        logger.info(f"   Positions: {summary['active_positions']}")
        logger.info(f"   Return: {summary['total_return_pct']:+.1f}%")
        logger.info(f"   Win Rate: {summary['win_rate']:.1f}%")
        
        # Send daily notification at 20:00
        current_hour = datetime.now().hour
        if current_hour == 20:
            try:
                from services.notifier import notifier
                notifier.send_message(
                    f"📊 Daily Portfolio Update\n"
                    f"Value: ${summary['portfolio_value']:,.2f}\n"
                    f"Return: {summary['total_return_pct']:+.1f}%\n"
                    f"Win Rate: {summary['win_rate']:.1f}%\n"
                    f"Active: {summary['active_positions']}"
                )
            except:
                pass
                
    except Exception as e:
        logger.error(f"❌ Error updating portfolio: {e}")

def check_pending_orders():
    """
    JOB 4: Check pending limit orders - runs every minute
    """
    logger.debug("📋 Checking pending orders...")
    
    trading_engine, load_portfolio, save_portfolio, data_feed = _import_modules()
    
    if not load_portfolio or not data_feed or not save_portfolio:
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
            amount = order.get('amount', 0)
            stop_loss = order.get('stop_loss')
            take_profit = order.get('take_profit')
            
            if not symbol or limit_price == 0 or amount == 0:
                continue
            
            try:
                # Get current price
                current_price = data_feed.get_price(symbol)
                if not current_price:
                    continue
                
                # Check limit order conditions
                should_execute = False
                
                if side == 'buy' and current_price <= limit_price:
                    should_execute = True
                    logger.info(f"✅ Buy limit triggered: {symbol} at ${current_price:.2f} (limit: ${limit_price:.2f})")
                elif side == 'sell' and current_price >= limit_price:
                    should_execute = True
                    logger.info(f"✅ Sell limit triggered: {symbol} at ${current_price:.2f} (limit: ${limit_price:.2f})")
                
                if should_execute and trading_engine:
                    if side == 'buy':
                        success = trading_engine.open_position(
                            symbol=symbol,
                            side='long',
                            entry_price=current_price,
                            units=amount,
                            stop_loss=stop_loss,
                            take_profit=take_profit
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
            logger.info(f"📝 Executed {orders_executed} limit orders")
            
    except Exception as e:
        logger.error(f"❌ Error checking orders: {e}")

def health_check():
    """
    JOB 5: Health check - runs every 6 hours
    Checks if bot is healthy and sends alert if not
    """
    logger.info("🏥 Running health check...")
    
    trading_engine, load_portfolio, save_portfolio, data_feed = _import_modules()
    
    if not trading_engine:
        logger.error("❌ Health check failed - cannot import trading_engine")
        return
    
    try:
        # Check portfolio health
        is_healthy = trading_engine.check_portfolio_health()
        
        if not is_healthy:
            logger.warning("⚠️ Portfolio health check FAILED")
            
            # Send alert
            try:
                from services.notifier import notifier
                summary = trading_engine.get_portfolio_summary()
                notifier.send_message(
                    f"⚠️ *Health Alert*\n"
                    f"Portfolio may be at risk\n"
                    f"Drawdown: {summary['total_return_pct']:+.1f}%"
                )
            except:
                pass
        else:
            logger.info("✅ Health check passed")
            
    except Exception as e:
        logger.error(f"❌ Error in health check: {e}")

# -------------------------------------------------------------------
# GET JOBS FOR TELEGRAM BOT
# -------------------------------------------------------------------
def get_all_jobs():
    """Return all scheduler jobs"""
    return {
        'check_stops': check_stop_losses_and_take_profits,      # Every minute
        'scan_signals': scan_for_trading_signals,               # Every 5 minutes
        'check_orders': check_pending_orders,                   # Every minute
        'update_portfolio': update_portfolio_summary,           # Every hour
        'health_check': health_check,                           # Every 6 hours
    }

def start_schedulers(bot_data=None):
    """Setup scheduler jobs"""
    logger.info("⏰ Scheduler jobs initialized")
    logger.info("   - Stop loss check: every 60 seconds")
    logger.info("   - Signal scan: every 5 minutes")
    logger.info("   - Pending orders: every 60 seconds")
    logger.info("   - Portfolio update: every hour")
    logger.info("   - Health check: every 6 hours")
    return get_all_jobs()

if __name__ == "__main__":
    print(f"🔧 Project root: {project_root}")
    
    # Test imports
    try:
        from modules.trade_engine import trading_engine
        print("✅ Successfully imported trade_engine")
    except ImportError as e:
        print(f"❌ Failed to import: {e}")
    
    # Test jobs
    print("\n🔄 Testing scheduler jobs...")
    jobs = get_all_jobs()
    
    for job_name, job_func in jobs.items():
        print(f"\n📋 Testing {job_name}...")
        job_func()