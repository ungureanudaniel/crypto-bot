import sys
import os
import asyncio
import logging
from telegram import Bot
from telegram.error import TelegramError

# Setup logging (ONCE)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# CONFIG LOADING
# -------------------------------------------------------------------
try:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config_loader import config
    CONFIG = config.config
    logger.info(f"✅ Notifier config loaded: {CONFIG.get('trading_mode', 'paper')}")
except ImportError:
    logger.warning("⚠️ Could not import config_loader, using defaults")
    CONFIG = {'trading_mode': 'paper', 'telegram_token': '', 'telegram_chat_id': ''}

class Notifier:
    """Handles Telegram notifications"""
    
    def __init__(self):
        self.token = CONFIG.get('telegram_token', '')
        self.chat_id = CONFIG.get('telegram_chat_id', '')
        self.bot = None
        
        if self.token and self.chat_id:
            try:
                self.bot = Bot(token=self.token)
                logger.info(f"📱 Notifier initialized for chat {self.chat_id}")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Telegram bot: {e}")
                self.bot = None
        else:
            logger.warning("⚠️ Telegram credentials missing - notifications disabled")
    
    async def send_message(self, message: str) -> bool:
        """Send telegram message asynchronously"""
        if not self.bot or not self.chat_id:
            logger.debug("Telegram not configured - message not sent")
            return False
        
        try:
            # Truncate very long messages (Telegram limit: 4096 chars)
            if len(message) > 4000:
                message = message[:4000] + "... (truncated)"
            
            await self.bot.send_message(
                chat_id=int(self.chat_id),
                text=message,
                parse_mode='html'
            )
            
            logger.info(f"📤 Message sent: {message[:50]}...")
            return True
            
        except TelegramError as e:
            logger.error(f"❌ Telegram error: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Failed to send message: {e}")
            return False
    
    def send_message_sync(self, message: str) -> bool:
        """Synchronous wrapper for send_message"""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # No event loop in this thread, create one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        try:
            return loop.run_until_complete(self.send_message(message))
        except Exception as e:
            logger.error(f"❌ Sync send failed: {e}")
            return False
    
    async def send_trade_notification(self, trade_data: dict) -> bool:
        """Send formatted trade notification"""
        symbol = trade_data.get('symbol', 'Unknown')
        side = trade_data.get('side', '').upper()
        price = trade_data.get('price', 0)
        amount = trade_data.get('amount', 0)
        pnl = trade_data.get('pnl', 0)
        reason = trade_data.get('reason', '')
        mode = trade_data.get('mode', 'paper').upper()
        
        # Determine message type
        if side in ['BUY', 'LONG'] or 'BUY' in str(trade_data.get('side', '')).upper():
            # Opening trade
            stop_loss = trade_data.get('stop_loss', 0)
            take_profit = trade_data.get('take_profit', 0)
            
            message = (
                f"🟢 *TRADE OPENED* [{mode}]\n"
                f"━━━━━━━━━━━━━━━━\n"
                f"📊 *{symbol}*\n"
                f"💰 Side: LONG\n"
                f"💵 Entry: `${price:.2f}`\n"
                f"📦 Amount: `{amount:.6f}`\n"
                f"💎 Value: `${price * amount:.2f}`\n"
            )
            
            if stop_loss:
                stop_pct = ((stop_loss / price) - 1) * 100
                message += f"🛑 Stop: `${stop_loss:.2f}` ({stop_pct:+.1f}%)\n"
            
            if take_profit:
                tp_pct = ((take_profit / price) - 1) * 100
                message += f"🎯 Target: `${take_profit:.2f}` ({tp_pct:+.1f}%)\n"
            
        elif side in ['SELL', 'SHORT'] and pnl == 0:
            # Opening short trade
            stop_loss = trade_data.get('stop_loss', 0)
            take_profit = trade_data.get('take_profit', 0)
            
            message = (
                f"🔴 *SHORT OPENED* [{mode}]\n"
                f"━━━━━━━━━━━━━━━━\n"
                f"📊 *{symbol}*\n"
                f"💰 Side: SHORT\n"
                f"💵 Entry: `${price:.2f}`\n"
                f"📦 Amount: `{amount:.6f}`\n"
                f"💎 Value: `${price * amount:.2f}`\n"
            )
            
            if stop_loss:
                stop_pct = (1 - (stop_loss / price)) * 100
                message += f"🛑 Stop: `${stop_loss:.2f}` ({stop_pct:+.1f}%)\n"
            
            if take_profit:
                tp_pct = (1 - (take_profit / price)) * 100
                message += f"🎯 Target: `${take_profit:.2f}` ({tp_pct:+.1f}%)\n"
            
        else:
            # Closing trade
            entry_price = trade_data.get('entry_price', price)
            pnl_pct = ((price / entry_price) - 1) * 100 if 'LONG' in side else (1 - (price / entry_price)) * 100
            
            # Choose emoji based on PnL
            if pnl > 0:
                emoji = "✅"
                pnl_sign = "+"
            elif pnl < 0:
                emoji = "❌"
                pnl_sign = ""
            else:
                emoji = "⚪"
                pnl_sign = ""
            
            reason_text = f" ({reason})" if reason else ""
            
            message = (
                f"{emoji} *TRADE CLOSED*{reason_text} [{mode}]\n"
                f"━━━━━━━━━━━━━━━━\n"
                f"📊 *{symbol}*\n"
                f"💰 Side: {side}\n"
                f"💵 Exit: `${price:.2f}`\n"
                f"📈 PnL: `${pnl_sign}{pnl:.2f}` ({pnl_sign}{pnl_pct:.1f}%)\n"
            )
        
        # Add timestamp
        from datetime import datetime
        message += f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        return await self.send_message(message)
    
    async def send_alert(self, message: str, level: str = "INFO") -> bool:
        """Send an alert message with level indicator"""
        emoji_map = {
            "INFO": "ℹ️",
            "WARNING": "⚠️",
            "ERROR": "🚨",
            "SUCCESS": "✅"
        }
        emoji = emoji_map.get(level.upper(), "📢")
        
        formatted = f"{emoji} *{level.upper()}*\n{message}"
        return await self.send_message(formatted)
    
    async def send_daily_report(self, summary: dict) -> bool:
        """Send daily trading report"""
        message = (
            f"📊 *Daily Trading Report*\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"💰 Portfolio: `${summary.get('portfolio_value', 0):,.2f}`\n"
            f"💵 Cash: `${summary.get('cash_balance', 0):,.2f}`\n"
            f"📈 Return: `{summary.get('total_return_pct', 0):+.1f}%`\n"
            f"🎯 Win Rate: `{summary.get('win_rate', 0):.1f}%`\n"
            f"📊 Active: `{summary.get('active_positions', 0)}`\n"
            f"📋 Total Trades: `{summary.get('total_trades', 0)}`"
        )
        return await self.send_message(message)

# Create singleton instance
notifier = Notifier()

# For backward compatibility
async def send_message(message: str):
    return await notifier.send_message(message)

async def send_trade_notification(trade_data: dict):
    return await notifier.send_trade_notification(trade_data)

if __name__ == "__main__":
    # Test the notifier
    import asyncio
    
    async def test():
        print("🧪 Testing notifier...")
        
        # Test trade open
        await notifier.send_trade_notification({
            'symbol': 'BTC/USDC',
            'side': 'LONG',
            'price': 45000,
            'amount': 0.001,
            'stop_loss': 42750,
            'take_profit': 49500,
            'mode': 'testnet'
        })
        
        await asyncio.sleep(2)
        
        # Test trade close (profit)
        await notifier.send_trade_notification({
            'symbol': 'BTC/USDC',
            'side': 'SELL',
            'price': 48000,
            'amount': 0.001,
            'pnl': 30,
            'entry_price': 45000,
            'reason': 'take_profit',
            'mode': 'testnet'
        })
        
        await asyncio.sleep(2)
        
        # Test alert
        await notifier.send_alert("Bot started successfully", "SUCCESS")
    
    asyncio.run(test())