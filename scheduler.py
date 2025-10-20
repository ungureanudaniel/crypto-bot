from email.mime import application
import schedule
import time
import threading
import json
import logging
from logging.handlers import RotatingFileHandler
from regime_switcher import train_model, predict_regime
from data_feed import fetch_ohlcv
from trade_engine import execute_trade, check_pending_orders

# -------------------------------------------------------------------
# CONFIG & LOGGING
# -------------------------------------------------------------------
with open("config.json") as f:
    CONFIG = json.load(f)

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

def limit_order_check_job(bot_data):
    """Check and execute pending limit orders"""
    logging.info("Checking pending limit orders...")
    try:
        current_prices = get_current_prices()
        check_pending_orders(current_prices)
        logging.info("Limit order check completed")
    except Exception as e:
        logging.error(f"Error in limit order check job: {e}")

# Weekly trading job
def weekly_trading_job(bot_data):
    if not bot_data.get("run_bot", True):
        logging.info("Bot is stopped. Skipping weekly trading job.")
        return
        
    interval = bot_data.get("trading_interval", "1h")  # fallback to 1h
    logging.info("Running weekly trading job...")

    for coin in CONFIG['coins']:
        try:
            df = fetch_ohlcv(coin, interval)
            if df.empty:
                logging.warning(f"No data fetched for {coin}, skipping trade execution.")
                continue

            regime = predict_regime(df.iloc[-1])
            price = df.iloc[-1]['close']
            execute_trade(coin, regime, price)
        except Exception as e:
            logging.error(f"Error in weekly trading job for {coin}: {e}")

def intraday_trading_job(bot_data):
    """15min trading job - should run every 15 minutes"""
    logging.info("🚀 Running 15m trading job...")
    
    for coin in CONFIG['coins']:
        try:
            df = fetch_ohlcv(coin, "15m")
            if df.empty:
                continue
                
            regime = predict_regime(df)  # Returns string like "Trending 📈"
            price = df.iloc[-1]['close']
            
            # This should now work for trending regimes too!
            execute_trade(coin, regime, price)
            
        except Exception as e:
            logging.error(f"Error in 15m trading job for {coin}: {e}")

def data_refresh_job(bot_data):
    interval = bot_data.get("trading_interval", "1h")  # fallback to 1h
    logging.info("Refreshing data feed...")
    for coin in CONFIG['coins']:
        try:
            fetch_ohlcv(coin, interval)
        except Exception as e:
            logging.error(f"Error refreshing data for {coin}: {e}")

def portfolio_health_check(bot_data):
    """Regular portfolio health check and cleanup"""
    logging.info("Running portfolio health check...")
    try:
        from trade_engine import load_portfolio
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
            from datetime import datetime, timedelta
            week_ago = datetime.now() - timedelta(days=7)
            
            initial_count = len(pending_orders)
            portfolio['pending_orders'] = [
                order for order in pending_orders 
                if datetime.fromisoformat(order['timestamp']) > week_ago
            ]
            
            if len(portfolio['pending_orders']) < initial_count:
                from trade_engine import save_portfolio
                save_portfolio(portfolio)
                logging.info(f"Cleaned up {initial_count - len(portfolio['pending_orders'])} expired pending orders")
                
    except Exception as e:
        logging.error(f"Error in portfolio health check: {e}")

def risk_management_job(bot_data):
    """Check stop losses and portfolio health"""
    logging.info("Running risk management job...")
    try:
        from trade_engine import check_stop_losses, check_portfolio_health
        from data_feed import get_current_prices  # You'll need to implement this
        
        # Get current prices for all coins with open positions
        current_prices = get_current_prices()
        
        # Check and execute stop losses
        positions_closed = check_stop_losses(current_prices)
        
        # Check overall portfolio health
        portfolio_healthy = check_portfolio_health()
        
        if positions_closed:
            logging.info("Stop losses executed in risk management job")
            
    except Exception as e:
        logging.error(f"Error in risk management job: {e}")

# -------------------------------------------------------------------
# SCHEDULER
# -------------------------------------------------------------------
def start_schedulers(bot_data):
    logging.info("Starting scheduler for 15m trading...")
    
    # Set default interval if not set
    if "trading_interval" not in bot_data:
        bot_data["trading_interval"] = "15m"
    
    train_model()

    # Trading jobs - run every 15 minutes
    schedule.every(15).minutes.do(intraday_trading_job, bot_data=bot_data)
    
    # Limit order checks - more frequent (every 5 minutes)
    schedule.every(5).minutes.do(limit_order_check_job, bot_data=bot_data)
    
    # Model retraining - weekly
    schedule.every().sunday.at("02:00").do(train_model)

    # Risk management - every 5 minutes
    schedule.every(5).minutes.do(risk_management_job, bot_data=bot_data)

    # Portfolio health - daily
    schedule.every().day.at("06:00").do(portfolio_health_check, bot_data=bot_data)
    
    # Data refresh - hourly
    # ?
    
    def run_scheduler():
        while True:
            try:
                schedule.run_pending()
                time.sleep(1)
            except Exception as e:
                logging.error(f"Scheduler error: {e}")
                time.sleep(60)  # Wait a minute before retrying

    threading.Thread(target=run_scheduler, daemon=True).start()
    logging.info("Scheduler started and running in background.")

def manual_limit_order_check():
    """Manual trigger for limit order checking (for testing)"""
    logging.info("Manual limit order check triggered")
    current_prices = get_current_prices()
    check_pending_orders(current_prices)

if __name__ == "__main__":
    # For direct testing
    bot_data = {
        "run_bot": True,
        "trading_interval": "15m",
        "portfolio": {
            "cash_balance": 1000,
            "holdings": {},
            "positions": {},
            "pending_orders": []
        }
    }
    start_schedulers(bot_data)
    
    # Keep the main thread alive
    while True:
        time.sleep(60)