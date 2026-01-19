# services/notifier.py - Telegram notifications
import sys
import os
import asyncio
import logging
from telegram import Bot

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
except ImportError:
    logger.warning("⚠️ Could not import config_loader, using defaults")
    CONFIG = {'trading_mode': 'paper', 'testnet': False, 'rate_limit_delay': 0.5}
print(CONFIG)
logging.info("🔧 Configuration loaded for data feed. Trading mode: %s", CONFIG.get('trading_mode', 'paper'))
# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

class Notifier:
    """Handles Telegram notifications"""
    
    def __init__(self):
        self.bot = Bot(token=CONFIG['telegram_token'])
        self.chat_id = CONFIG['telegram_chat_id']
        logger.info("📱 Notifier initialized")
    
    def send_message(self, message: str) -> bool:
        """Send telegram message synchronously"""
        try:
            async def send_async():
                await self.bot.send_message(chat_id=self.chat_id, text=message)
            
            # Run event loop
            asyncio.run(send_async())
            
            logger.info(f"Message sent: {message[:50]}...")
            return True
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            return False
    
    def send_trade_notification(self, trade_data: dict):
        """Send formatted trade notification"""
        symbol = trade_data.get('symbol', 'Unknown')
        side = trade_data.get('side', '').upper()
        price = trade_data.get('price', 0)
        amount = trade_data.get('amount', 0)
        pnl = trade_data.get('pnl', 0)
        
        if side == 'BUY' or side == 'LONG':
            message = (
                f"📈 *TRADE OPENED*\n"
                f"Symbol: {symbol}\n"
                f"Side: {side}\n"
                f"Price: ${price:.2f}\n"
                f"Amount: {amount:.6f}\n"
                f"Value: ${price * amount:.2f}"
            )
        else:
            message = (
                f"📉 *TRADE CLOSED*\n"
                f"Symbol: {symbol}\n"
                f"Side: {side}\n"
                f"Exit Price: ${price:.2f}\n"
                f"PnL: ${pnl:.2f}"
            )
        
        return self.send_message(message)

# Create singleton instance
notifier = Notifier()