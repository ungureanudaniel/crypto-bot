import json
import logging
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

logging.basicConfig(level=logging.INFO)

with open("config.json") as f:
    config = json.load(f)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Kraken AI AutoTrader is running.")

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    status_msg = "Running" if context.bot_data.get("run_bot", False) else "Stopped"
    await update.message.reply_text(f"Bot status: {status_msg}")

# This is a dummy balance fetcher — replace with your Kraken API call or state variable
def get_portfolio_balance():
    # Example balance - replace with your real balance fetching logic
    return {
        "USD": 800.50,
        "BTC": 0.000004,
        "ETH": 1.1,
        "JUP": 400.0
    }

async def balance(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    balance = get_portfolio_balance()
    msg_lines = ["💰 Current Portfolio Balance:"]
    for coin, amount in balance.items():
        msg_lines.append(f"{coin}: {amount}")
    msg = "\n".join(msg_lines)
    await update.message.reply_text(msg)

def start_telegram_bot():
    application = ApplicationBuilder().token(config['telegram_token']).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("status", status))

    logging.info("Starting Telegram bot...")
    application.run_polling()
    logging.info("Telegram bot started.")