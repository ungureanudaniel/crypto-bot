import logging
import sys
import os
import time
import threading
from datetime import datetime
from typing import Dict, Any

# -------------------------------------------------------------------
# SETUP PATHS
# -------------------------------------------------------------------
current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_file_dir)
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'modules'))

# Import modules
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
# GLOBAL VARIABLES FOR THREAD CONTROL
# -------------------------------------------------------------------
_scheduler_thread = None
_stop_event = threading.Event()

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
# SCHEDULER JOBS - UNCHANGED
# -------------------------------------------------------------------
def check_stop_losses_and_take_profits():
    """
    JOB 1: Check stop losses - runs every minute
    """
    trading_engine, data_feed = _import_modules()
    
    if not trading_engine:
        return
    
    try:
        positions_closed = trading_engine.check_stop_losses()
        if positions_closed:
            logger.info(f"✅ Closed {positions_closed} positions via stop/take profit")
    except Exception as e:
        logger.error(f"❌ Error in stop loss check: {e}")

def scan_for_trading_signals():
    """
    JOB 2: Scan for signals - runs every 5 minutes
    """
    trading_engine, data_feed = _import_modules()
    
    if not trading_engine:
        return
    
    try:
        current_positions = len(trading_engine.open_positions)
        
        if current_positions >= trading_engine.max_positions:
            logger.info(f"⏭️ At max positions ({current_positions}/{trading_engine.max_positions})")
            return
        
        signals = trading_engine.scan_and_trade()
        
        if signals:
            logger.info(f"🎯 Found {len(signals)} signals")
            
            if CONFIG.get('auto_execute_signals', False):
                executed = 0
                for signal in signals:
                    if trading_engine.execute_signal(signal):
                        executed += 1
                
                if executed > 0:
                    logger.info(f"✅ Auto-executed {executed} signals")
        else:
            logger.debug("📭 No signals found")
    except Exception as e:
        logger.error(f"❌ Error scanning signals: {e}")

def update_portfolio_summary():
    """
    JOB 3: Update portfolio summary - runs every hour
    """
    trading_engine, data_feed = _import_modules()
    
    if not trading_engine:
        return
    
    try:
        summary = trading_engine.get_portfolio_summary()
        
        logger.info(f"💰 Portfolio: ${summary['portfolio_value']:,.2f}")
        logger.info(f"   Cash: ${summary['cash_balance']:,.2f}")
        logger.info(f"   Positions: {summary['active_positions']}")
        logger.info(f"   Return: {summary['total_return_pct']:+.1f}%")
        logger.info(f"   Win Rate: {summary['win_rate']:.1f}%")
        
        # Send daily notification at 20:00
        current_hour = datetime.now().hour
        if current_hour == 20 and datetime.now().minute < 5:
            try:
                from services.notifier import notifier
                if hasattr(notifier, 'send_message_sync'):
                    notifier.send_message_sync(
                        f"📊 Daily Portfolio Update\n"
                        f"Value: ${summary['portfolio_value']:,.2f}\n"
                        f"Return: {summary['total_return_pct']:+.1f}%\n"
                        f"Win Rate: {summary['win_rate']:.1f}%\n"
                        f"Active: {summary['active_positions']}"
                    )
            except Exception as e:
                logger.error(f"❌ Failed to send daily notification: {e}")
    except Exception as e:
        logger.error(f"❌ Error updating portfolio: {e}")

def check_pending_orders():
    """
    JOB 4: Check pending limit orders - runs every minute
    """
    trading_engine, data_feed = _import_modules()
    
    if not trading_engine or not trading_engine.binance_client:
        return
    
    try:
        open_orders = []
        for symbol in CONFIG.get('coins', ['BTC/USDC', 'ETH/USDC'])[:5]:  # Limit to 5 symbols to avoid rate limits
            try:
                binance_symbol = symbol.replace('/', '')
                orders = trading_engine.binance_client.get_open_orders(symbol=binance_symbol)
                open_orders.extend(orders)
                time.sleep(0.1)  # Small delay to avoid rate limits
            except Exception as e:
                logger.debug(f"Error fetching orders for {symbol}: {e}")
        
        if open_orders:
            logger.info(f"📋 Found {len(open_orders)} open orders on exchange")
    except Exception as e:
        logger.error(f"❌ Error checking orders: {e}")

def health_check():
    """
    JOB 5: Health check - runs every 6 hours
    """
    trading_engine, data_feed = _import_modules()
    
    if not trading_engine:
        return
    
    try:
        is_healthy = trading_engine.check_portfolio_health()
        
        if not is_healthy:
            logger.warning("⚠️ Portfolio health check FAILED")
            try:
                from services.notifier import notifier
                summary = trading_engine.get_portfolio_summary()
                if hasattr(notifier, 'send_message_sync'):
                    notifier.send_message_sync(
                        f"⚠️ *Health Alert*\n"
                        f"Portfolio may be at risk\n"
                        f"Return: {summary['total_return_pct']:+.1f}%"
                    )
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
    trading_engine, data_feed = _import_modules()
    
    if not trading_engine:
        return
    
    try:
        summary = trading_engine.get_portfolio_summary()
        
        log_entry = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'time': datetime.now().strftime('%H:%M:%S'),
            'portfolio_value': summary['portfolio_value'],
            'cash_balance': summary['cash_balance'],
            'return_pct': summary['total_return_pct'],
            'active_positions': summary['active_positions'],
            'win_rate': summary['win_rate']
        }
        
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
# SCHEDULER THREAD MAIN LOOP
# -------------------------------------------------------------------
def _scheduler_loop():
    """Main scheduler loop running in separate thread"""
    logger.info("🚀 Scheduler thread started")
    
    last_minute = -1
    last_5min = -1
    last_hour = -1
    last_6hour = -1
    last_day = -1
    
    while not _stop_event.is_set():
        try:
            now = datetime.now()
            current_minute = now.minute
            current_hour = now.hour
            current_day = now.day
            
            # Run every minute
            if current_minute != last_minute:
                check_stop_losses_and_take_profits()
                check_pending_orders()
                last_minute = current_minute
            
            # Run every 5 minutes
            if current_minute % 5 == 0 and current_minute != last_5min:
                scan_for_trading_signals()
                last_5min = current_minute
            
            # Run every hour
            if current_minute == 0 and current_hour != last_hour:
                update_portfolio_summary()
                last_hour = current_hour
            
            # Run every 6 hours
            if current_hour % 6 == 0 and current_minute == 0 and current_hour != last_6hour:
                health_check()
                last_6hour = current_hour
            
            # Run once per day (at 23:59)
            if current_hour == 23 and current_minute == 59 and current_day != last_day:
                log_daily_performance()
                last_day = current_day
            
            # Sleep to avoid CPU spinning
            time.sleep(1)
            
        except Exception as e:
            logger.error(f"❌ Scheduler loop error: {e}")
            time.sleep(5)
    
    logger.info("🛑 Scheduler thread stopped")

# -------------------------------------------------------------------
# PUBLIC FUNCTIONS TO CONTROL THE SCHEDULER THREAD
# -------------------------------------------------------------------
def start_scheduler():
    """Start the scheduler in a separate thread"""
    global _scheduler_thread
    
    if _scheduler_thread and _scheduler_thread.is_alive():
        logger.warning("⚠️ Scheduler already running")
        return
    
    _stop_event.clear()
    _scheduler_thread = threading.Thread(target=_scheduler_loop, daemon=True)
    _scheduler_thread.start()
    logger.info("✅ Scheduler thread started")

def stop_scheduler():
    """Stop the scheduler thread"""
    logger.info("🛑 Stopping scheduler...")
    _stop_event.set()
    
    if _scheduler_thread:
        _scheduler_thread.join(timeout=10)
        logger.info("✅ Scheduler thread stopped")

def get_all_jobs():
    """Return all scheduler jobs (for compatibility)"""
    return {
        'check_stops': check_stop_losses_and_take_profits,
        'scan_signals': scan_for_trading_signals,
        'check_orders': check_pending_orders,
        'update_portfolio': update_portfolio_summary,
        'health_check': health_check,
        'daily_log': log_daily_performance,
    }

def start_schedulers(bot_data=None):
    """Legacy function for compatibility"""
    start_scheduler()
    return get_all_jobs()

# -------------------------------------------------------------------
# AUTO-START WHEN IMPORTED
# -------------------------------------------------------------------
# start_scheduler()

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
    
    # Test thread
    print("\n🧪 Testing scheduler thread...")
    start_scheduler()
    
    try:
        print("   Scheduler running for 10 seconds...")
        time.sleep(10)
    finally:
        stop_scheduler()