# services/notifier.py - Telegram notifications
import json
import asyncio
import logging
from telegram import Bot

logger = logging.getLogger(__name__)

# Load config
with open("config.json") as f:
    config = json.load(f)

class Notifier:
    """Handles Telegram notifications"""
    
    def __init__(self):
        self.bot = Bot(token=config['telegram_token'])
        self.chat_id = config['telegram_chat_id']
        logger.info("📱 Notifier initialized")
    
    def send_message(self, message: str) -> bool:
        """Send telegram message synchronously"""
        try:
            async def send_async():
                await self.bot.send_message(chat_id=self.chat_id, text=message)
            
            # Create and run event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(send_async())
            loop.close()
            
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