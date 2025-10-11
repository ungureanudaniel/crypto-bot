import json
import logging
import os
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
from regime_switcher import train_model, predict_regime
from data_feed import fetch_ohlcv
from trade_engine import execute_trade
from scheduler import start_schedulers
import schedule

# -------------------------------------------------------------------
# CONFIG & LOGGING
# -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

with open("config.json") as f:
    CONFIG = json.load(f)

PORTFOLIO_FILE = "portfolio.json"

def load_portfolio():
    if os.path.exists(PORTFOLIO_FILE):
        with open(PORTFOLIO_FILE, "r") as f:
            return json.load(f)
    else:
        logging.warning("portfolio.json not found. Returning empty portfolio.")
        return {"cash_balance": 0, "holdings": {}, "positions": {}}

# -------------------------------------------------------------------
# COMMAND HANDLERS
# -------------------------------------------------------------------

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    context.bot_data["run_bot"] = True
    context.bot_data["portfolio"] = load_portfolio()
    await update.message.reply_text("🤖 Binance AI AutoTrader is now *running*!")
    start_schedulers(context.bot_data)
    logging.info("Scheduler started via Telegram /start command.")

async def stop(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    context.bot_data["run_bot"] = False
    await update.message.reply_text("🛑 Bot stopped. No new trades will be executed.")
    logging.info("Trading bot manually stopped via Telegram.")

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    status_msg = "✅ Running" if context.bot_data.get("run_bot", False) else "⛔ Stopped"
    await update.message.reply_text(f"Bot status: {status_msg}")

async def balance(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # Always refresh portfolio from file
    context.bot_data["portfolio"] = load_portfolio()
    portfolio = context.bot_data["portfolio"]

    msg_lines = [f"💰 *Portfolio Balance:*", f"Cash: ${portfolio.get('cash_balance', 0):,.2f}"]
    holdings = portfolio.get("holdings", {})
    if not holdings:
        msg_lines.append("No current holdings.")
    else:
        for coin, amount in holdings.items():
            msg_lines.append(f"{coin}: {amount}")

    await update.message.reply_text("\n".join(msg_lines))

async def train(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("📚 Retraining ML model...")
    train_model()
    await update.message.reply_text("✅ Model retrained successfully!")

async def regime(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text("Usage: `/regime BTC/USDT`", parse_mode='Markdown')
        return

    symbol = context.args[0].upper()
    df = fetch_ohlcv(symbol, context.bot_data.get("trading_interval", "1h"))
    if df.empty:
        await update.message.reply_text(f"❌ No data found for {symbol}.")
        return

    current_regime = predict_regime(df.iloc[-1])
    await update.message.reply_text(f"📊 Current regime for *{symbol}*: `{current_regime}`", parse_mode='Markdown')

async def trade(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text("Usage: `/trade BTC/USDT`", parse_mode='Markdown')
        return

    symbol = context.args[0].upper()
    df = fetch_ohlcv(symbol, context.bot_data.get("trading_interval", "1h"))
    if df.empty:
        await update.message.reply_text(f"❌ Could not fetch market data for {symbol}.")
        return

    regime = predict_regime(df.iloc[-1])
    price = df.iloc[-1]['close']
    execute_trade(symbol, regime, price)

    # Refresh portfolio after trade
    context.bot_data["portfolio"] = load_portfolio()

    await update.message.reply_text(f"🚀 Executed trade for {symbol} based on regime: `{regime}`", parse_mode='Markdown')

async def latest_trades(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show recent trades from portfolio"""
    from trade_engine import load_portfolio

    portfolio = load_portfolio()
    trade_history = portfolio.get("trade_history", [])

    if not trade_history:
        await update.message.reply_text("📭 No trades have been executed yet.")
        return

    # Show last 10 trades
    msg_lines = ["📊 *Latest Trades:*"]
    for trade in trade_history[-10:]:
        line = (
            f"{trade['action'].upper()} {trade['coin']} "
            f"{trade['amount']:.6f} @ ${trade['price']:.2f} "
            f"PnL: ${trade.get('pnl', 0):.2f}"
        )
        msg_lines.append(line)

    await update.message.reply_text("\n".join(msg_lines), parse_mode="Markdown")

async def coin_price(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show current price of a coin in USDC"""
    from data_feed import fetch_ohlcv

    if not context.args:
        await update.message.reply_text("Usage: `/price BTC/USDC`", parse_mode="Markdown")
        return

    symbol = context.args[0].upper()
    df = fetch_ohlcv(symbol, "1m")  # fetch latest 1-minute candle
    if df.empty:
        await update.message.reply_text(f"❌ Could not fetch market data for {symbol}.")
        return

    current_price = df.iloc[-1]["close"]
    await update.message.reply_text(f"💵 Current price of {symbol}: ${current_price:.2f}")

async def set_interval(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if len(context.args) != 1:
        await update.message.reply_text("Usage: /set_interval <interval>\nExamples: 1m, 1h, 1d, 1w")
        return

    interval = context.args[0]
    valid_intervals = ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"]
    if interval not in valid_intervals:
        await update.message.reply_text(f"Invalid interval. Valid intervals: {', '.join(valid_intervals)}")
        return

    context.bot_data["trading_interval"] = interval
    await update.message.reply_text(f"Trading interval set to {interval}")

async def get_interval(update: Update, context: ContextTypes.DEFAULT_TYPE):
    interval = context.bot_data.get("trading_interval", "1h")
    await update.message.reply_text(f"Current trading interval: {interval}")

async def scheduler_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    jobs = schedule.get_jobs()
    if not jobs:
        await update.message.reply_text("⚙️ No scheduled jobs are currently active.")
        return

    msg = ["📆 *Scheduler Jobs:*"]
    for job in jobs:
        msg.append(f"- {job.job_func.__name__}: next run at {job.next_run}")
    await update.message.reply_text("\n".join(msg), parse_mode='Markdown')

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    commands = [
        "🤖 /start - Start the trading bot and scheduler",
        "🛑 /stop - Stop all new trades",
        "📊 /status - Check if the bot is running",
        "💰 /balance - Show current portfolio holdings",
        "📈 /latest_trades - Show recent trades executed"
        "💵 /price <symbol> - Get current price of a symbol (e.g., BTC/USDC)",
        "📚 /train - Manually retrain the ML model",
        "📅 /set_interval <interval> - Set trading interval (e.g., 1m, 5m, 1h)",
        "📊 /get_interval - Get current trading interval",
        "📈 /regime <symbol> - Check current regime for a symbol",
        "🚀 /trade <symbol> - Manually execute a trade decision",
        "🕐 /scheduler_status - List scheduled jobs and their next runs",
        "❓ /help - Show this help message"
    ]
    await update.message.reply_text("\n".join(commands))

# -------------------------------------------------------------------
# START TELEGRAM BOT
# -------------------------------------------------------------------
def start_telegram_bot():
    application = ApplicationBuilder().token(CONFIG['telegram_token']).build()
    application.bot_data["run_bot"] = False
    application.bot_data["trading_interval"] = "1h"
    application.bot_data["portfolio"] = load_portfolio()

    # Register commands
    handlers = [
        ("start", start),
        ("stop", stop),
        ("status", status),
        ("balance", balance),
        ("latest_trades", latest_trades),
        ("price", coin_price),
        ("train", train),
        ("set_interval", set_interval),
        ("get_interval", get_interval),
        ("regime", regime),
        ("trade", trade),
        ("scheduler_status", scheduler_status),
        ("help", help_command),
    ]
    for cmd, func in handlers:
        application.add_handler(CommandHandler(cmd, func))

    logging.info("🚀 Starting Telegram bot...")
    application.run_polling()
    logging.info("✅ Telegram bot started and polling for commands.")

    return application
