# services/scheduler.py - Updated for papertrade_engine
import schedule
import time
import threading
import json
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime, timedelta

from modules.regime_switcher import train_model, predict_regime
from modules.data_feed import fetch_ohlcv
from modules.trade_engine import paper_engine
from modules.portfolio import load_portfolio, save_portfolio

# -------------------------------------------------------------------
# CONFIG & LOGGING
# -------------------------------------------------------------------
def load_config():
    """Load and return config"""
    with open("config.json", "r") as f:
        return json.load(f)

CONFIG = load_config()

logging.basicConfig(
    handlers=[RotatingFileHandler("trading_bot.log", maxBytes=5_000_000, backupCount=3)],
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# -------------------------------------------------------------------
# JOBS
# -------------------------------------------------------------------
def get_current_prices():
    """Fetch current prices for all configured coins"""
    current_prices = {}
    interval = "1m"  # Use 1-minute data for latest prices
    
    for coin in CONFIG['coins']:
        try:
            df = fetch_ohlcv(coin, interval)
            if not df.empty:
                current_prices[coin] = df.iloc[-1]['close']
                logging.debug(f"Current price for {coin}: ${current_prices[coin]:.2f}")
            else:
                logging.warning(f"Could not fetch current price for {coin}")
        except Exception as e:
            logging.error(f"Error fetching price for {coin}: {e}")
    
    return current_prices

def check_pending_orders(current_prices):
    """Check and execute pending limit orders"""
    from modules.portfolio import load_portfolio, save_portfolio
    from modules.trade_engine import paper_engine
    from datetime import datetime
    
    portfolio = load_portfolio()
    pending_orders = portfolio.get('pending_orders', [])
    
    if not pending_orders:
        logging.debug("No pending orders to check")
        return False
    
    logging.info(f"🔍 Checking {len(pending_orders)} pending orders...")
    
    executed_orders = []
    
    for order in list(pending_orders):
        symbol = order['symbol']
        
        # Get current price for this symbol
        current_price = current_prices.get(symbol)
        
        if not current_price:
            logging.warning(f"No current price available for {symbol}")
            continue
        
        should_execute = False
        trigger_reason = ""
        
        # Check if limit order conditions are met
        if order.get('type') == 'limit':
            if order['side'] == 'buy' and current_price <= order['price']:
                should_execute = True
                trigger_reason = f"Price ${current_price:.2f} <= limit ${order['price']:.2f}"
            elif order['side'] == 'sell' and current_price >= order['price']:
                should_execute = True
                trigger_reason = f"Price ${current_price:.2f} >= limit ${order['price']:.2f}"
        
        if should_execute:
            logging.info(f"🚀 Executing {order['side']} order for {symbol}: {trigger_reason}")
            
            try:
                if order['side'] == 'buy':
                    # Check if we have enough cash
                    cost = order['amount'] * current_price
                    if portfolio.get('cash_balance', 0) >= cost:
                        # Execute buy order using paper_engine
                        success = paper_engine.open_position(
                            symbol=symbol,
                            side='long',
                            entry_price=current_price,
                            units=order['amount'],
                            stop_loss=current_price * 0.95,  # 5% stop loss
                            take_profit=current_price * 1.10  # 10% take profit
                        )
                        
                        if success:
                            # Mark order as executed
                            order['status'] = 'executed'
                            order['executed_at'] = datetime.now().isoformat()
                            order['executed_price'] = current_price
                            order['trigger_reason'] = trigger_reason
                            executed_orders.append(order)
                            logging.info(f"✅ Buy order executed for {symbol}")
                        else:
                            logging.warning(f"Failed to execute buy order for {symbol}")
                    else:
                        logging.warning(f"Insufficient funds for {symbol}: ${portfolio.get('cash_balance', 0):.2f} available, need ${cost:.2f}")
                
                elif order['side'] == 'sell':
                    # Check if we have enough holdings
                    coin = symbol.split('/')[0]
                    current_holdings = portfolio.get('holdings', {}).get(coin, 0)
                    
                    if current_holdings >= order['amount']:
                        # Execute sell - this is simplified, you might need to adjust
                        from modules.portfolio import update_position
                        
                        # Sell from holdings
                        pnl = update_position(coin, "sell", order['amount'], current_price)
                        
                        if pnl is not None:
                            # Mark order as executed
                            order['status'] = 'executed'
                            order['executed_at'] = datetime.now().isoformat()
                            order['executed_price'] = current_price
                            order['trigger_reason'] = trigger_reason
                            executed_orders.append(order)
                            
                            # Record trade
                            trade_history = portfolio.get('trade_history', [])
                            trade_history.append({
                                'symbol': symbol,
                                'action': 'sell_limit',
                                'side': 'sell',
                                'amount': order['amount'],
                                'price': current_price,
                                'timestamp': datetime.now().isoformat(),
                                'pnl': pnl
                            })
                            portfolio['trade_history'] = trade_history
                            
                            logging.info(f"✅ Sell order executed for {symbol}")
                    else:
                        logging.warning(f"Insufficient holdings for {symbol}: have {current_holdings}, need {order['amount']}")
            
            except Exception as e:
                logging.error(f"Error executing order for {symbol}: {e}")
    
    # Update portfolio to remove executed orders
    if executed_orders:
        # Keep only orders that weren't executed
        remaining_orders = [
            order for order in pending_orders 
            if order.get('status') != 'executed'
        ]
        
        portfolio['pending_orders'] = remaining_orders
        save_portfolio(portfolio)
        
        logging.info(f"🎯 Executed {len(executed_orders)} orders, {len(remaining_orders)} remaining")
        
        # Send notifications
        try:
            from services.notifier import notifier
            for order in executed_orders:
                notifier.send_message(
                    f"✅ Limit Order Executed!\n"
                    f"Symbol: {order['symbol']}\n"
                    f"Side: {order['side'].upper()}\n"
                    f"Amount: {order['amount']}\n"
                    f"Limit Price: ${order['price']:.2f}\n"
                    f"Executed at: ${order['executed_price']:.2f}\n"
                    f"Reason: {order.get('trigger_reason', 'Price triggered')}"
                )
        except Exception as e:
            logging.warning(f"Could not send notifications: {e}")
        
        return True
    
    return False

def limit_order_check_job(bot_data):
    """Check and execute pending limit orders"""
    if not bot_data.get("run_bot", True):
        return
        
    logging.info("Checking pending limit orders...")
    try:
        current_prices = get_current_prices()
        check_pending_orders(current_prices)
        logging.info("Limit order check completed")
    except Exception as e:
        logging.error(f"Error in limit order check job: {e}")

def intraday_trading_job(bot_data):
    if not bot_data.get("run_bot", True):
        return
        
    logging.info("🚀 Running 15m trading job...")
    
    try:
        # Check stop losses
        paper_engine.check_stop_losses()
        
        # Check portfolio health
        if not paper_engine.check_portfolio_health():
            logging.warning("Portfolio health check failed, skipping trades")
            return
        
        # Scan for signals
        signals = paper_engine.scan_and_trade()
        
        if signals:
            logging.info(f"Found {len(signals)} trading signals")
            
            # Execute signals
            for signal_data in signals[:2]:  # Max 2 at a time
                paper_engine.execute_signal(signal_data=signal_data)
        else:
            logging.info("No trading signals found")
            
    except Exception as e:
        logging.error(f"Error in trading job: {e}")

def data_refresh_job(bot_data):
    """Refresh data feed"""
    if not bot_data.get("run_bot", True):
        return
        
    interval = bot_data.get("trading_interval", "1h")
    logging.info("Refreshing data feed...")
    for coin in CONFIG['coins']:
        try:
            fetch_ohlcv(coin, interval)
        except Exception as e:
            logging.error(f"Error refreshing data for {coin}: {e}")

def portfolio_health_check(bot_data):
    """Regular portfolio health check and cleanup"""
    if not bot_data.get("run_bot", True):
        return
        
    logging.info("Running portfolio health check...")
    try:
        portfolio = load_portfolio()
        
        # Log current portfolio status
        cash = portfolio.get('cash_balance', 0)
        holdings = portfolio.get('holdings', {})
        positions = portfolio.get('positions', {})
        pending_orders = portfolio.get('pending_orders', [])
        
        logging.info(f"Portfolio Health - Cash: ${cash:.2f}, "
                    f"Holdings: {len(holdings)}, "
                    f"Positions: {len(positions)}, "
                    f"Pending Orders: {len(pending_orders)}")
        
        # Clean up expired pending orders (older than 7 days)
        if pending_orders:
            week_ago = datetime.now() - timedelta(days=7)
            
            initial_count = len(pending_orders)
            portfolio['pending_orders'] = [
                order for order in pending_orders 
                if datetime.fromisoformat(order['timestamp']) > week_ago
            ]
            
            if len(portfolio['pending_orders']) < initial_count:
                save_portfolio(portfolio)
                logging.info(f"Cleaned up {initial_count - len(portfolio['pending_orders'])} expired pending orders")
                
    except Exception as e:
        logging.error(f"Error in portfolio health check: {e}")

def risk_management_job(bot_data):
    """Check stop losses and portfolio health"""
    if not bot_data.get("run_bot", True):
        return
        
    logging.info("Running risk management job...")
    try:
        # Check stop losses using papertrade_engine
        positions_closed = paper_engine.check_stop_losses()
        
        if positions_closed:
            logging.info("Stop losses/take profits executed in risk management job")
            
        # Check portfolio value and drawdown
        current_prices = get_current_prices()
        portfolio = load_portfolio()
        
        # Calculate portfolio value
        total_value = portfolio.get('cash_balance', 0)
        for symbol, position in portfolio.get('positions', {}).items():
            current_price = current_prices.get(symbol, position['entry_price'])
            total_value += position['amount'] * current_price
        
        initial_balance = portfolio.get('initial_balance', total_value)
        drawdown = (initial_balance - total_value) / initial_balance
        max_drawdown = CONFIG.get('max_drawdown', 0.05)
        
        if drawdown > max_drawdown:
            logging.warning(f"⚠️ Portfolio drawdown {drawdown:.1%} exceeds limit {max_drawdown:.1%}")
            
    except Exception as e:
        logging.error(f"Error in risk management job: {e}")

def send_daily_report(bot_data):
    """Send daily portfolio report"""
    if not bot_data.get("run_bot", True):
        return
        
    try:
        from services.notifier import notifier
        
        portfolio = load_portfolio()
        current_prices = get_current_prices()
        
        # Calculate portfolio value
        cash = portfolio.get('cash_balance', 0)
        total_value = cash
        
        positions_summary = []
        for symbol, position in portfolio.get('positions', {}).items():
            current_price = current_prices.get(symbol, position['entry_price'])
            position_value = position['amount'] * current_price
            total_value += position_value
            pnl_pct = (current_price / position['entry_price'] - 1) * 100
            positions_summary.append(f"{symbol}: ${position_value:.2f} ({pnl_pct:+.1f}%)")
        
        initial_balance = portfolio.get('initial_balance', total_value)
        total_return = ((total_value - initial_balance) / initial_balance) * 100
        
        message = (
            f"📊 Daily Portfolio Report\n\n"
            f"💰 Total Value: ${total_value:,.2f}\n"
            f"📈 Total Return: {total_return:+.1f}%\n"
            f"💵 Cash: ${cash:,.2f}\n"
            f"📦 Positions: {len(portfolio.get('positions', {}))}\n"
        )
        
        if positions_summary:
            message += f"\nActive Positions:\n" + "\n".join(positions_summary)
        
        notifier.send_message(message)
        logging.info("Daily report sent")
        
    except Exception as e:
        logging.error(f"Error sending daily report: {e}")

# -------------------------------------------------------------------
# SCHEDULER
# -------------------------------------------------------------------
def start_schedulers(bot_data):
    """Start scheduler in a background thread"""
    logging.info("🟢 Starting scheduler in background thread...")
    
    # Setup all jobs
    schedule.every(15).minutes.do(intraday_trading_job, bot_data=bot_data)
    schedule.every(5).minutes.do(limit_order_check_job, bot_data=bot_data)
    schedule.every().sunday.at("02:00").do(train_model)
    schedule.every(5).minutes.do(risk_management_job, bot_data=bot_data)
    schedule.every().day.at("06:00").do(portfolio_health_check, bot_data=bot_data)
    schedule.every().day.at("20:00").do(send_daily_report, bot_data=bot_data)
    schedule.every().hour.do(data_refresh_job, bot_data=bot_data)
    
    # Create and start scheduler thread
    scheduler_thread = threading.Thread(
        target=_scheduler_loop,
        args=(bot_data,),
        daemon=True,  # Daemon thread will exit when main program exits
        name="SchedulerThread"
    )
    scheduler_thread.start()
    
    logging.info("✅ Scheduler started in background thread")
    return scheduler_thread

def _scheduler_loop(bot_data):
    """Main scheduler loop to run in background thread"""
    logging.info("📡 Scheduler loop running in background")
    
    while True:
        try:
            if bot_data.get("run_bot", True):
                schedule.run_pending()
            time.sleep(1)
        except KeyboardInterrupt:
            logging.info("Scheduler loop interrupted")
            break
        except Exception as e:
            logging.error(f"🔥 Scheduler loop error: {e}")
            time.sleep(5)

def stop_schedulers():
    """Stop all scheduled jobs"""
    logging.info("🛑 Stopping all scheduled jobs")
    schedule.clear()

def check_scheduler_status():
    """Check if scheduler is running"""
    jobs = schedule.get_jobs()
    return {
        'running': len(jobs) > 0,
        'jobs_count': len(jobs),
        'jobs': [str(job) for job in jobs]
    }

if __name__ == "__main__":
    # For direct testing
    bot_data = {
        "run_bot": True,
        "trading_interval": "15m"
    }
    
    print("🚀 Starting scheduler in test mode...")
    start_schedulers(bot_data)
    
    # Run once immediately
    
    # Keep the main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Scheduler stopped")