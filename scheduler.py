from email.mime import application
import schedule
import time
import threading
import json
import logging
from logging.handlers import RotatingFileHandler
from regime_switcher import train_model, predict_regime
from data_feed import fetch_ohlcv
from trade_engine import execute_trade

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
# Weekly trading job
def weekly_trading_job(bot_data):
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

# Hourly trading job
def intraday_trading_job(bot_data):
    interval = bot_data.get("trading_interval", "1h")
    logging.info(f"Running intraday trading job with interval {interval}...")

    for coin in CONFIG['coins']:
        try:
            df = fetch_ohlcv(coin, interval)
            if df.empty:
                logging.warning(f"No data for {coin}, skipping.")
                continue

            regime = predict_regime(df.iloc[-1])
            price = df.iloc[-1]['close']
            execute_trade(coin, regime, price)
        except Exception as e:
            logging.error(f"Error in intraday trading job for {coin}: {e}")

def data_refresh_job(bot_data):
    interval = bot_data.get("trading_interval", "1h")  # fallback to 1h
    logging.info("Refreshing data feed...")
    for coin in CONFIG['coins']:
        try:
            fetch_ohlcv(coin, interval)
        except Exception as e:
            logging.error(f"Error refreshing data for {coin}: {e}")

# -------------------------------------------------------------------
# SCHEDULER
# -------------------------------------------------------------------
def start_schedulers(bot_data):
    logging.info("Starting scheduler...")
    train_model()

    schedule.every().sunday.at("00:00").do(train_model)
    # schedule.every().monday.at("00:05").do(weekly_trading_job, bot_data=bot_data)
    schedule.every().hour.at(":05").do(intraday_trading_job, bot_data=bot_data) 
    schedule.every(6).day.at("00:00").do(data_refresh_job, bot_data=bot_data)

    def run_scheduler():
        while True:
            schedule.run_pending()
            time.sleep(1)

    threading.Thread(target=run_scheduler, daemon=True).start()
    logging.info("Scheduler started and running in background.")
