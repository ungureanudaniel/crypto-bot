import logging
import os
from telegram_bot import start_telegram_bot

# Configure logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/trading_bot.log"),
        logging.StreamHandler()
    ]
)

if __name__ == "__main__":
    logging.info("🚀 Starting Kraken AI AutoTrader...")
    
    # This starts everything - Telegram bot will handle schedulers internally
    start_telegram_bot()