# bot.py - SIMPLIFIED VERSION

import json
import sys
import logging
import os
import asyncio
from modules.trade_engine import trading_engine
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

# -------------------------------------------------------------------
# SETUP LOGGING FIRST
# -------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# SIMPLE CONFIG LOADING - DO THIS ONCE
# -------------------------------------------------------------------
def load_global_config():
    """Load config ONCE and make it globally available"""
    try:
        # Add parent directory to path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        sys.path.insert(0, parent_dir)
        
        from config_loader import config as app_config
        logger.info(f"✅ Config loaded: trading_mode={app_config.config.get('trading_mode', 'paper')}")
        return app_config.config
    except ImportError:
        logger.warning("⚠️ Could not import config_loader, using defaults")
        return {
            'trading_mode': 'paper',
            'testnet': False,
            'rate_limit_delay': 0.5,
            'telegram_token': '',
            'telegram_chat_id': '',
            'coins': ['BTC/USDC', 'ETH/USDC'],
            'starting_balance': 1000
        }

# LOAD CONFIG ONCE - GLOBAL VARIABLE
CONFIG = load_global_config()
logger.info(f"📋 Trading mode: {CONFIG.get('trading_mode')}")

# -------------------------------------------------------------------
# SIMPLIFIED PORTFOLIO LOADING
# -------------------------------------------------------------------
PORTFOLIO_FILE = "portfolio.json"

def load_portfolio():
    """Simple portfolio loading"""
    if os.path.exists(PORTFOLIO_FILE):
        try:
            with open(PORTFOLIO_FILE, "r") as f:
                return json.load(f)
        except:
            pass
    return {"cash_balance": 10000, "holdings": {}, "positions": {}}

# -------------------------------------------------------------------
# COMMAND HANDLERS - USE GLOBAL CONFIG
# -------------------------------------------------------------------

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Start the bot"""
    context.bot_data["run_bot"] = True
    context.bot_data["portfolio"] = load_portfolio()
    
    # Get mode from global CONFIG
    mode = CONFIG.get('trading_mode', 'paper')
    mode_display = "🚀 LIVE" if mode == 'live' else "🧪 TESTNET" if mode == 'testnet' else "📝 PAPER"
    
    await update.message.reply_text(
        f"🤖 *Binance AI AutoTrader Started!*\n\n"
        f"*Mode:* {mode_display}\n"
        f"*Initial Balance:* ${CONFIG.get('starting_balance', 1000):,.2f}\n"
        f"*Coins Monitored:* {len(CONFIG.get('coins', []))}\n\n"
        f"Use /help to see all commands",
        parse_mode='Markdown'
    )

async def trading_mode_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show current trading mode - SIMPLIFIED"""
    mode = CONFIG.get('trading_mode', 'paper')
    testnet = CONFIG.get('testnet', False)
    
    if mode == 'live':
        message = "🚀 *LIVE TRADING MODE*\nReal orders on Binance"
    elif mode == 'testnet':
        message = "🧪 *TESTNET MODE*\nTest orders on Binance Testnet"
    else:
        message = "📝 *PAPER TRADING MODE*\nSimulated trades only"
    
    await update.message.reply_text(message, parse_mode='Markdown')

async def config_info(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show config info - USE GLOBAL CONFIG"""
    info_lines = [
        "⚙️ *Bot Configuration*",
        f"Mode: {CONFIG.get('trading_mode', 'paper')}",
        f"Testnet: {CONFIG.get('testnet', False)}",
        f"Starting Balance: ${CONFIG.get('starting_balance', 1000):,.2f}",
        f"Coins: {', '.join(CONFIG.get('coins', ['BTC/USDC']))}",
        f"Max Positions: {CONFIG.get('max_positions', 3)}",
        f"Risk per Trade: {CONFIG.get('risk_per_trade', 0.02)*100:.1f}%",
    ]
    
    await update.message.reply_text("\n".join(info_lines), parse_mode='Markdown')

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show bot status"""
    is_running = context.bot_data.get("run_bot", False)
    mode = CONFIG.get('trading_mode', 'paper')
    
    await update.message.reply_text(
        f"✅ *Bot Status:* {'Running' if is_running else 'Stopped'}\n"
        f"*Mode:* {mode.upper()}\n"
        f"*Balance:* ${load_portfolio().get('cash_balance', 0):,.2f}",
        parse_mode='Markdown'
    )

async def balance(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show balance"""
    portfolio = load_portfolio()
    cash = portfolio.get('cash_balance', 0)
    
    await update.message.reply_text(
        f"💰 *Cash Balance:* ${cash:,.2f}\n"
        f"📦 *Holdings:* {len(portfolio.get('holdings', {}))} coins\n"
        f"🛡️ *Positions:* {len(portfolio.get('positions', {}))}",
        parse_mode='Markdown'
    )

# -------------------------------------------------------------------
# FIXED TRADING COMMANDS
# -------------------------------------------------------------------

async def regime(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Predict market regime - SIMPLIFIED"""
    if not context.args:
        await update.message.reply_text("Usage: `/regime BTC/USDC`", parse_mode='Markdown')
        return

    symbol = context.args[0].upper()
    
    try:
        # Import inside function to avoid circular imports
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        from modules.data_feed import fetch_ohlcv
        from modules.regime_switcher import predict_regime
        
        df = fetch_ohlcv(symbol, "1h", limit=100)
        if df.empty:
            await update.message.reply_text(f"❌ No data for {symbol}")
            return
            
        regime = predict_regime(df)
        await update.message.reply_text(f"📊 *{symbol}:* {regime}", parse_mode='Markdown')
        
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {str(e)}")

async def trade(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Execute manual trade - SIMPLIFIED"""
    if not context.args:
        await update.message.reply_text("Usage: `/trade BTC/USDC`", parse_mode='Markdown')
        return

    symbol = context.args[0].upper()
    
    try:
        # Import inside function
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        from modules.trade_engine import trading_engine
        
        success = trading_engine.place_limit_order(symbol, side, amount, price)
        
        if success:
            await update.message.reply_text(f"✅ Trade executed for {symbol}")
        else:
            await update.message.reply_text(f"❌ Failed to execute trade")
            
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {str(e)}")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Help command"""
    help_text = """
🤖 *Binance AI AutoTrader Commands*

*Basic Commands:*
/start - Start the bot
/stop - Stop the bot  
/status - Check bot status
/help - Show this help

*Portfolio:*
/balance - Show cash balance
/portfolio - Portfolio overview
/portfolio_value - Detailed valuation

*Trading:*
/regime <symbol> - Market regime
/trade <symbol> - Execute trade
/scan - Scan opportunities

*Configuration:*
/mode - Show trading mode
/config - Show config
/api_status - Check API status

*Example:*
/regime BTC/USDC
/portfolio
    """
    
    await update.message.reply_text(help_text, parse_mode='Markdown')

# -------------------------------------------------------------------
# REGISTER HANDLERS
# -------------------------------------------------------------------
def register_handlers(application):
    """Register all command handlers"""
    handlers = [
        ("start", start),
        ("stop", lambda update, context: update.message.reply_text("Bot stopped")),
        ("status", status),
        ("balance", balance),
        ("regime", regime),
        ("trade", trade),
        ("mode", trading_mode_cmd),
        ("config", config_info),
        ("help", help_command),
    ]
    
    for cmd, func in handlers:
        application.add_handler(CommandHandler(cmd, func))

# -------------------------------------------------------------------
# MAIN FUNCTION
# -------------------------------------------------------------------
def main():
    """Main function to start the bot"""
    # Check telegram token
    telegram_token = CONFIG.get('telegram_token')
    if not telegram_token:
        logger.error("❌ Telegram token not configured in config.json or .env")
        return
    
    logger.info(f"🤖 Starting Telegram bot with token: {telegram_token[:10]}...")
    
    try:
        # Create application
        application = ApplicationBuilder().token(telegram_token).build()
        
        # Initialize bot_data
        application.bot_data["run_bot"] = True
        application.bot_data["portfolio"] = load_portfolio()
        
        # Register handlers
        register_handlers(application)
        
        logger.info("✅ Bot setup complete")
        logger.info("📱 Listening for commands...")
        
        # Run bot
        application.run_polling()
        
    except Exception as e:
        logger.error(f"❌ Bot error: {e}")

if __name__ == "__main__":
    main()