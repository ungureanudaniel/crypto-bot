import logging
from telegram_bot import start_telegram_bot
from scheduler import start_schedulers

logging.basicConfig(
    filename="logs/bot.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

if __name__ == "__main__":
    logging.info("Kraken AI AutoTrader starting...")
    start_telegram_bot()
    # Start Telegram bot and get the application instance
    application = start_telegram_bot()
    start_schedulers(application.bot_data)
