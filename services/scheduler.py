# services/scheduler.py - UPDATED FOR EXCHANGE-ONLY PORTFOLIO
import logging
import sys
import os
from datetime import datetime

# -------------------------------------------------------------------
# SETUP PATHS
# -------------------------------------------------------------------
current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_file_dir)
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'modules'))

# import modules
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
        from modules.data_feed import data_feed
        return trading_engine, data_feed
    except ImportError as e:
        logger.error(f"Failed to import modules: {e}")
        logger.error(f"Python path: {sys.path}")
        return None, None

# -------------------------------------------------------------------
# SCHEDULER JOBS - USING EXCHANGE DATA ONLY
# -------------------------------------------------------------------

def check_stop_losses_and_take_profits():
    """
    JOB 1: Check stop losses - runs every minute
    Uses trade_engine's built-in stop loss checking
    """
    logger.debug("🛡️ Checking stop losses...")
    
    trading_engine, data_feed = _import_modules()
    
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
    Uses trade_engine's scan_and_trade to generate signals
    """
    logger.info("🔍 Scanning for trading signals...")
    
    trading_engine, data_feed = _import_modules()
    
    if not trading_engine:
        logger.error("❌ Cannot import trading_engine")
        return
    
    try:
        # Check if we can take new positions directly from engine
        current_positions = len(trading_engine.open_positions)
        
        if current_positions >= trading_engine.max_positions:
            logger.info(f"⏭️ At max positions ({current_positions}/{trading_engine.max_positions})")
            return
        
        # USING THE TRADE ENGINE'S SCAN METHOD
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
                        # Use sync wrapper or create task
                        if hasattr(notifier, 'send_message_sync'):
                            notifier.send_message_sync(f"🤖 Auto-executed {executed} trades")
                        else:
                            logger.info(f"🤖 Auto-executed {executed} trades")
                    except:
                        pass
        else:
            logger.debug("📭 No signals found")
            
    except Exception as e:
        logger.error(f"❌ Error scanning signals: {e}")

def update_portfolio_summary():
    """
    JOB 3: Update portfolio summary - runs every hour
    Logs current portfolio status directly from exchange
    """
    logger.info("📊 Updating portfolio summary...")
    
    trading_engine, data_feed = _import_modules()
    
    if not trading_engine:
        logger.error("❌ Cannot import trading_engine")
        return
    
    try:
        # Get portfolio summary directly from trade_engine (which gets from exchange)
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
                message = (
                    f"📊 Daily Portfolio Update\n"
                    f"Value: ${summary['portfolio_value']:,.2f}\n"
                    f"Cash: ${summary['cash_balance']:,.2f}\n"
                    f"Return: {summary['total_return_pct']:+.1f}%\n"
                    f"Win Rate: {summary['win_rate']:.1f}%\n"
                    f"Active: {summary['active_positions']}"
                )
                
                # Try async first, fall back to sync
                if hasattr(notifier, 'send_message_sync'):
                    notifier.send_message_sync(message)
                else:
                    logger.info(f"📊 Daily summary:\n{message}")
                    
            except Exception as e:
                logger.error(f"❌ Failed to send daily notification: {e}")
                
    except Exception as e:
        logger.error(f"❌ Error updating portfolio: {e}")

def check_pending_orders():
    """
    JOB 4: Check pending limit orders - runs every minute
    Note: This now relies on exchange's open orders, not portfolio.json
    """
    logger.debug("📋 Checking pending orders...")
    
    trading_engine, data_feed = _import_modules()
    
    if not trading_engine or not trading_engine.binance_client:
        logger.debug("No binance client available - skipping order check")
        return
    
    try:
        # Get open orders directly from exchange
        open_orders = []
        
        for symbol in CONFIG.get('coins', ['BTC/USDC', 'ETH/USDC']):
            try:
                binance_symbol = symbol.replace('/', '')
                orders = trading_engine.binance_client.get_open_orders(symbol=binance_symbol)
                open_orders.extend(orders)
            except Exception as e:
                logger.debug(f"Error fetching orders for {symbol}: {e}")
        
        if open_orders:
            logger.info(f"📋 Found {len(open_orders)} open orders on exchange")
            
            # Check if any orders should be executed (they already will be by exchange)
            # This is just for logging
            for order in open_orders:
                logger.debug(f"   Order {order['orderId']}: {order['symbol']} {order['side']} "
                           f"{order['origQty']} @ {order['price']}")
        
    except Exception as e:
        logger.error(f"❌ Error checking orders: {e}")

def health_check():
    """
    JOB 5: Health check - runs every 6 hours
    Checks if bot is healthy and sends alert if not
    """
    logger.info("🏥 Running health check...")
    
    trading_engine, data_feed = _import_modules()
    
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
                message = (
                    f"⚠️ *Health Alert*\n"
                    f"Portfolio may be at risk\n"
                    f"Value: ${summary['portfolio_value']:,.2f}\n"
                    f"Return: {summary['total_return_pct']:+.1f}%\n"
                    f"Drawdown: {summary['total_return_pct'] if summary['total_return_pct'] < 0 else 0:.1f}%"
                )
                
                if hasattr(notifier, 'send_message_sync'):
                    notifier.send_message_sync(message)
                else:
                    logger.warning(message)
                    
            except Exception as e:
                logger.error(f"❌ Failed to send health alert: {e}")
        else:
            logger.info("✅ Health check passed")
            
    except Exception as e:
        logger.error(f"❌ Error in health check: {e}")

def log_daily_performance():
    """
    JOB 6: Log daily performance at market close - runs once per day
    """
    logger.info("📈 Logging daily performance...")
    
    trading_engine, data_feed = _import_modules()
    
    if not trading_engine:
        return
    
    try:
        summary = trading_engine.get_portfolio_summary()
        
        # Log to file for historical tracking
        log_entry = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'time': datetime.now().strftime('%H:%M:%S'),
            'portfolio_value': summary['portfolio_value'],
            'cash_balance': summary['cash_balance'],
            'return_pct': summary['total_return_pct'],
            'active_positions': summary['active_positions'],
            'win_rate': summary['win_rate']
        }
        
        # Append to daily log file
        log_file = os.path.join(project_root, 'daily_performance.log')
        try:
            with open(log_file, 'a') as f:
                f.write(f"{log_entry['date']} {log_entry['time']} | "
                       f"Value: ${log_entry['portfolio_value']:,.2f} | "
                       f"Return: {log_entry['return_pct']:+.1f}% | "
                       f"Win Rate: {log_entry['win_rate']:.1f}%\n")
        except Exception as e:
            logger.error(f"Failed to write daily log: {e}")
        
        logger.info(f"📊 Daily snapshot saved")
        
    except Exception as e:
        logger.error(f"❌ Error logging daily performance: {e}")

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
        'daily_log': log_daily_performance,                     # Once per day
    }

def start_schedulers(bot_data=None):
    """Setup scheduler jobs"""
    logger.info("⏰ Scheduler jobs initialized")
    logger.info("   - Stop loss check: every 60 seconds")
    logger.info("   - Signal scan: every 5 minutes")
    logger.info("   - Pending orders: every 60 seconds")
    logger.info("   - Portfolio update: every hour")
    logger.info("   - Health check: every 6 hours")
    logger.info("   - Daily log: once per day")
    return get_all_jobs()

if __name__ == "__main__":
    print(f"🔧 Project root: {project_root}")
    print(f"🔧 Trading mode: {CONFIG.get('trading_mode', 'unknown')}")
    
    # Test imports
    try:
        from modules.trade_engine import trading_engine
        print(f"✅ Successfully imported trade_engine")
        print(f"   Mode: {trading_engine.trading_mode}")
        print(f"   Client available: {'Yes' if trading_engine.binance_client else 'No'}")
    except ImportError as e:
        print(f"❌ Failed to import: {e}")
    
    # Test jobs
    print("\n🔄 Testing scheduler jobs...")
    jobs = get_all_jobs()
    
    for job_name, job_func in jobs.items():
        print(f"\n📋 Testing {job_name}...")
        try:
            job_func()
        except Exception as e:
            print(f"   ❌ Error: {e}")