import schedule
import time
import threading
import json
import logging
from regime_switcher import train_model, predict_regime
from data_feed import fetch_ohlcv
from trade_engine import execute_trade

def weekly_trading_job():
    logging.info("Running weekly trading job...")
    for coin in json.load(open("config.json"))['coins']:
        df = fetch_ohlcv(coin, "1W")
        regime = predict_regime(df.iloc[-1])
        price = df.iloc[-1]['close']
        execute_trade(coin, regime, price)

def data_refresh_job():
    logging.info("Refreshing data feed...")
    for coin in json.load(open("config.json"))['coins']:
        fetch_ohlcv(coin, "1W")

def start_schedulers():
    logging.info("Starting scheduler...")
    train_model()  # Train model once at the start
    schedule.every().sunday.at("00:00").do(train_model)
    schedule.every().monday.at("00:05").do(weekly_trading_job)
    schedule.every(6).day.at("00:00").do(data_refresh_job)

    def run_scheduler():
        while True:
            schedule.run_pending()
            time.sleep(1)

    threading.Thread(target=run_scheduler, daemon=True).start()
