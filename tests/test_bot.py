# test_bot.py
import asyncio
from asyncio.log import logger
import sys
import os
import logging
# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# -------------------------------------------------------------------
# CONFIG LOADING
# -------------------------------------------------------------------
try:
    # Add parent directory to path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config_loader import config
    CONFIG = config.config
    logger.info(f"✅ Config loaded: {CONFIG.get('trading_mode', 'paper')}")
    print(CONFIG)
except ImportError:
    logger.warning("⚠️ Could not import config_loader, using defaults")
    CONFIG = {'trading_mode': 'paper', 'testnet': False, 'rate_limit_delay': 0.5}
logging.info("🔧 Configuration loaded for data feed. Trading mode: %s", CONFIG.get('trading_mode', 'paper'))
# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Fix for Windows
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Get token from environment
token = CONFIG.get('telegram_token') or os.environ.get('TELEGRAM_TOKEN')
if not token:
    print("❌ Set TELEGRAM_TOKEN environment variable")
    sys.exit(1)

print(f"Token: {token[:10]}...")
print("Testing event loop...")

try:
    loop = asyncio.get_event_loop()
    print(f"✅ Event loop: {loop}")
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    print("✅ Created new event loop")

# Test simple bot
print("\n🤖 Testing simple bot...")
from telegram.ext import ApplicationBuilder, CommandHandler

async def test_handler(update, context):
    await update.message.reply_text("Test")

async def main():
    app = ApplicationBuilder().token(token).build()
    app.add_handler(CommandHandler("test", test_handler))
    
    print("✅ Bot created - starting...")
    
    # Run with context manager
    async with app:
        await app.start()
        await app.updater.start_polling()
        print("🤖 Bot running...")
        
        # Wait forever
        await asyncio.Event().wait()

# Run it
try:
    asyncio.run(main())
except KeyboardInterrupt:
    print("\n🛑 Stopped")