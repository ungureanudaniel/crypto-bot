import logging
import os
from telegram_bot import start_telegram_bot
from scheduler import start_schedulers

logging.info("✅ Logging works! This should appear immediately.")

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    filename="logs/trading_bot.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/trading_bot.log"),
        logging.StreamHandler()
    ]
)

if __name__ == "__main__":
    logging.info("Kraken AI AutoTrader starting...")
    start_telegram_bot()
    # Start Telegram bot and get the application instance
    application = start_telegram_bot()
    start_schedulers(application.bot_data)
