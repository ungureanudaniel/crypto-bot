import logging
import sys
import os
import time
import threading
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Setup custom logging FIRST
from modules.logger_config import setup_logging

# Initialize logging (no Telegram for scheduler, just file/console)
setup_logging(verbose=False)

# Get logger for this module
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# SETUP PATHS
# -------------------------------------------------------------------
current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_file_dir)

# Import modules
try:
    from config_loader import config
    CONFIG = config.config
except ImportError:
    CONFIG = {'trading_mode': 'paper', 'coins': ['BTC/USDC', 'ETH/USDC']}

# -------------------------------------------------------------------
# GLOBAL TRADING ENGINE INSTANCE (SINGLETON)
# -------------------------------------------------------------------
try:
    from modules.trade_engine import trading_engine
    from modules.futures_engine import futures_engine
    from modules.portfolio import get_portfolio_summary
    logger.info("Trading engine loaded once at scheduler startup")
except ImportError as e:
    logger.error(f"❌ Failed to import trading engine: {e}")
    trading_engine = None
    futures_engine = None

# -------------------------------------------------------------------
# GLOBAL VARIABLES FOR THREAD CONTROL
# -------------------------------------------------------------------
_scheduler_thread = None
_stop_event = threading.Event()

# -------------------------------------------------------------------
# SCHEDULER JOBS (using the global trading_engine instance)
# -------------------------------------------------------------------
def check_stop_losses_and_take_profits():
    """
    JOB 1: Check stop losses - runs every minute.
    """
    global trading_engine
    if not trading_engine:
        return

    # --- Spot stop losses ---
    try:
        positions_closed = trading_engine.check_stop_losses()
        if positions_closed:
            logger.info(f"Spot: closed positions via stop/take profit")
    except Exception as e:
        logger.error(f"Error in spot stop loss check: {e}")

    # --- Futures stop losses ---
    try:
        if futures_engine:
            futures_closed = futures_engine.check_stops()
            if futures_closed:
                logger.info(f"Futures: closed {len(futures_closed)} position(s): {futures_closed}")
    except Exception as e:
        logger.error(f"Error in futures stop loss check: {e}")

def scan_for_trading_signals():
    """
    JOB 2: Scan for signals - runs every 5 minutes
    """
    global trading_engine
    if not trading_engine:
        return
    
    try:
        current_positions = len(trading_engine.open_positions) + len(trading_engine.open_futures_positions)
        
        if current_positions >= trading_engine.max_positions:
            logger.info(f"At max positions ({current_positions}/{trading_engine.max_positions})")
            return
        
        signals = trading_engine.scan_and_trade()
        
        if signals:
            logger.info(f"Found {len(signals)} signals")
            
            if CONFIG.get('auto_execute_signals', False):
                executed = 0
                failed = 0
                
                for signal in signals:
                    if trading_engine.execute_signal(signal):
                        executed += 1
                        logger.info(f"Executed {signal['symbol']}")
                    else:
                        failed += 1
                        logger.warning(f"Failed to execute {signal['symbol']}")
                    
                    current_positions = len(trading_engine.open_positions) + len(trading_engine.open_futures_positions)
                    if current_positions >= trading_engine.max_positions:
                        logger.info(f"⏭Max positions reached ({trading_engine.max_positions}), stopping execution")
                        break
                
                if executed > 0:
                    logger.info(f"Auto-executed {executed} signals, {failed} failed")
        else:
            logger.debug("No signals found")
            
    except Exception as e:
        logger.error(f"Error scanning signals: {e}")

def update_portfolio_summary():
    """
    JOB 3: Update portfolio summary - runs every hour
    """
    try:
        summary = get_portfolio_summary()
        
        total_value = summary.get('total_value', 0)
        total_cash = summary.get('total_cash', 0)
        total_return = summary.get('total_return_pct', 0)
        positions_count = summary.get('positions_count', 0)
        win_rate = summary.get('win_rate', 0)
        
        logger.info(f"Portfolio: ${total_value:,.2f}")
        logger.info(f"   Cash: ${total_cash:,.2f}")
        logger.info(f"   Positions: {positions_count}")
        logger.info(f"   Return: {total_return:+.1f}%")
        logger.info(f"   Win Rate: {win_rate:.1f}%")
        
        # Send daily notification at 20:00
        current_hour = datetime.now().hour
        if current_hour == 20:
            try:
                from services.notifier import notifier
                if hasattr(notifier, 'send_message_sync') and notifier.token:
                    pnl_emoji = "[WIN]" if total_return >= 0 else "[LOSS]"
                    notifier.send_message_sync(
                        f"{pnl_emoji} <b>Daily Portfolio Update</b>\n\n"
                        f"Value: <code>${total_value:,.2f}</code>\n"
                        f"Return: <code>{total_return:+.1f}%</code>\n"
                        f"Cash: <code>${total_cash:,.2f}</code>\n"
                        f"Win Rate: <code>{win_rate:.1f}%</code>\n"
                        f"Active: <code>{positions_count}</code>"
                    )
            except Exception as e:
                logger.error(f"❌ Failed to send daily notification: {e}")
    except Exception as e:
        logger.error(f"❌ Error updating portfolio: {e}")

def check_pending_orders():
    """
    JOB 4: Check pending limit orders - runs every minute
    """
    global trading_engine
    if not trading_engine or not trading_engine.binance_client:
        return
    
    try:
        open_orders = []
        coins = CONFIG.get('coins', ['BTC/USDC', 'ETH/USDC'])
        
        for symbol in coins[:5]:
            try:
                binance_symbol = symbol.replace('/', '')
                orders = trading_engine.binance_client.get_open_orders(symbol=binance_symbol)
                open_orders.extend(orders)
                time.sleep(0.1)
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
    global trading_engine
    if not trading_engine:
        return
    
    try:
        trading_engine.check_drawdown()
    except Exception as e:
        logger.error(f"Error in health check: {e}")
        
    try:
        summary = get_portfolio_summary()
        return_pct = summary.get('total_return_pct', 0)
        is_healthy = return_pct > -10
        
        if not is_healthy:
            logger.warning(f"Portfolio health check FAILED: {return_pct:+.1f}%")
            try:
                from services.notifier import notifier
                if hasattr(notifier, 'send_message_sync') and notifier.token:
                    notifier.send_message_sync(
                        f"<b>Health Alert</b>\n"
                        f"Portfolio may be at risk\n"
                        f"Return: <code>{return_pct:+.1f}%</code>"
                    )
            except Exception as e:
                logger.error(f"Failed to send health alert: {e}")
        else:
            logger.debug("Health check passed")
    except Exception as e:
        logger.error(f"Error in health check: {e}")

def log_daily_performance():
    """
    JOB 6: Log daily performance at market close - runs once per day
    """
    try:
        summary = get_portfolio_summary()
        
        log_entry = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'time': datetime.now().strftime('%H:%M:%S'),
            'portfolio_value': summary.get('total_value', 0),
            'cash_balance': summary.get('total_cash', 0),
            'return_pct': summary.get('total_return_pct', 0),
            'active_positions': summary.get('positions_count', 0),
            'win_rate': summary.get('win_rate', 0)
        }
        
        log_file = os.path.join(project_root, 'daily_performance.log')
        with open(log_file, 'a') as f:
            f.write(f"{log_entry['date']} {log_entry['time']} | "
                   f"Value: ${log_entry['portfolio_value']:,.2f} | "
                   f"Return: {log_entry['return_pct']:+.1f}% | "
                   f"Win Rate: {log_entry['win_rate']:.1f}%\n")
        
        logger.info(f"Daily snapshot saved")
    except Exception as e:
        logger.error(f"Error logging daily performance: {e}")

# -------------------------------------------------------------------
# SCHEDULER THREAD MAIN LOOP
# -------------------------------------------------------------------
def _scheduler_loop():
    """Main scheduler loop running in separate thread"""
    logger.info("Scheduler thread started")
    
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
            
            if current_minute != last_minute:
                check_stop_losses_and_take_profits()
                check_pending_orders()
                last_minute = current_minute
            
            if current_minute % 5 == 0 and current_minute != last_5min:
                scan_for_trading_signals()
                last_5min = current_minute
            
            if current_minute == 0 and current_hour != last_hour:
                update_portfolio_summary()
                last_hour = current_hour
            
            if current_hour % 6 == 0 and current_minute == 0 and current_hour != last_6hour:
                health_check()
                last_6hour = current_hour
            
            if current_hour == 23 and current_minute == 59 and current_day != last_day:
                log_daily_performance()
                last_day = current_day
            
            time.sleep(1)
            
        except Exception as e:
            logger.error(f"❌ Scheduler loop error: {e}")
            time.sleep(5)
    
    logger.info("🛑 Scheduler thread stopped")

# -------------------------------------------------------------------
# PUBLIC FUNCTIONS
# -------------------------------------------------------------------
def start_scheduler():
    """Start the scheduler in a separate thread"""
    global _scheduler_thread
    
    if _scheduler_thread and _scheduler_thread.is_alive():
        logger.warning("Scheduler already running")
        return
    
    _stop_event.clear()
    _scheduler_thread = threading.Thread(target=_scheduler_loop, daemon=True)
    _scheduler_thread.start()
    logger.info("Scheduler thread started")

def stop_scheduler():
    """Stop the scheduler thread"""
    logger.info("Stopping scheduler...")
    _stop_event.set()
    
    if _scheduler_thread:
        _scheduler_thread.join(timeout=10)
        logger.info("Scheduler thread stopped")

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
if __name__ == "__main__":
    print(f"🔧 Project root: {project_root}")
    print(f"🔧 Trading mode: {CONFIG.get('trading_mode', 'unknown')}")
    
    if trading_engine:
        print(f"Trading engine loaded")
        print(f"   Mode: {trading_engine.trading_mode}")
    
    print("\nTesting scheduler thread...")
    start_scheduler()
    
    try:
        print("   Scheduler running for 10 seconds...")
        time.sleep(10)
    finally:
        stop_scheduler()