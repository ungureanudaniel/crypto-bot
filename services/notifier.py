import sys
import os
import asyncio
import logging
from telegram import Bot
from telegram.error import TelegramError

# -------------------------------------------------------------------
# SETUP PATHS AND LOGGING
# -------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Setup custom logging first
from modules.logger_config import setup_logging

# Initialize logging (no Telegram for notifier itself to avoid recursion)
setup_logging(verbose=False)

# Get logger for this module
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# CONFIG LOADING
# -------------------------------------------------------------------
try:
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
    
    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters for Telegram"""
        import html
        return html.escape(text)
    
    async def send_message(self, message: str, parse_mode: str = 'HTML') -> bool:
        """Send telegram message asynchronously"""
        if not self.bot or not self.chat_id:
            logger.debug("Telegram not configured - message not sent")
            return False
        
        try:
            # Truncate very long messages (Telegram limit: 4096 chars)
            if len(message) > 4000:
                message = message[:4000] + "... (truncated)"
            
            # Ensure chat_id is int
            chat_id = int(self.chat_id) if not isinstance(self.chat_id, int) else self.chat_id

            # Only use HTML parse mode if message contains tags
            if '<' in message and '>' in message and parse_mode == 'HTML':
                await self.bot.send_message(
                    chat_id=chat_id,
                    text=message,
                    parse_mode='HTML'
                )
            else:
                await self.bot.send_message(
                    chat_id=chat_id,
                    text=message
                )
            
            logger.debug(f"📤 Message sent: {message[:50]}...")
            return True
            
        except TelegramError as e:
            logger.error(f"❌ Telegram error: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Failed to send message: {e}")
            return False
    
    def send_message_sync(self, message: str, parse_mode: str = 'HTML') -> bool:
        """Synchronous wrapper for send_message"""
        try:
            # Check if we're already in an async context
            try:
                loop = asyncio.get_running_loop()
                # We're in async context - need to run in thread
                import threading
                import concurrent.futures
                
                result = [False]
                def _run():
                    try:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            result[0] = loop.run_until_complete(self.send_message(message, parse_mode))
                        finally:
                            loop.close()
                    except Exception as e:
                        logger.error(f"Thread send error: {e}")
                
                t = threading.Thread(target=_run, daemon=True)
                t.start()
                t.join(timeout=15)
                return result[0]
                
            except RuntimeError:
                # No running loop - safe to use asyncio.run
                return asyncio.run(self.send_message(message, parse_mode))
                
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
        
        # Handle different trade types
        if pnl == 0:  # Opening trade
            stop_loss = trade_data.get('stop_loss', 0)
            take_profit = trade_data.get('take_profit', 0)
            
            if side in ['LONG', 'BUY']:
                message = (
                    f"🟢 <b>TRADE OPENED</b> [{mode}]\n"
                    f"━━━━━━━━━━━━━━━━\n"
                    f"📊 <b>{symbol}</b>\n"
                    f"💰 Side: LONG\n"
                    f"💵 Entry: <code>${price:.2f}</code>\n"
                    f"📦 Amount: <code>{amount:.6f}</code>\n"
                    f"💎 Value: <code>${price * amount:.2f}</code>\n"
                )
                if stop_loss:
                    stop_pct = ((stop_loss / price) - 1) * 100
                    message += f"🛑 Stop: <code>${stop_loss:.2f}</code> ({stop_pct:+.1f}%)\n"
                if take_profit:
                    tp_pct = ((take_profit / price) - 1) * 100
                    message += f"🎯 Target: <code>${take_profit:.2f}</code> ({tp_pct:+.1f}%)\n"
            
            elif side in ['SHORT', 'SELL']:
                message = (
                    f"🔴 <b>SHORT OPENED</b> [{mode}]\n"
                    f"━━━━━━━━━━━━━━━━\n"
                    f"📊 <b>{symbol}</b>\n"
                    f"💰 Side: SHORT\n"
                    f"💵 Entry: <code>${price:.2f}</code>\n"
                    f"📦 Amount: <code>{amount:.6f}</code>\n"
                    f"💎 Value: <code>${price * amount:.2f}</code>\n"
                )
                if stop_loss:
                    stop_pct = (1 - (stop_loss / price)) * 100
                    message += f"🛑 Stop: <code>${stop_loss:.2f}</code> ({stop_pct:+.1f}%)\n"
                if take_profit:
                    tp_pct = (1 - (take_profit / price)) * 100
                    message += f"🎯 Target: <code>${take_profit:.2f}</code> ({tp_pct:+.1f}%)\n"
            else:
                return False
        
        else:  # Closing trade
            entry_price = trade_data.get('entry_price', price)
            original_side = trade_data.get('original_side', side)
            is_long = original_side in ['LONG', 'BUY']
            pnl_pct = ((price / entry_price) - 1) * 100 if is_long else (1 - (price / entry_price)) * 100
            
            emoji = "✅" if pnl > 0 else "❌" if pnl < 0 else "⚪"
            pnl_sign = "+" if pnl > 0 else ""
            reason_text = f" ({reason})" if reason else ""
            
            message = (
                f"{emoji} <b>TRADE CLOSED</b>{reason_text} [{mode}]\n"
                f"━━━━━━━━━━━━━━━━\n"
                f"📊 <b>{symbol}</b>\n"
                f"💰 Side: {original_side}\n"
                f"💵 Exit: <code>${price:.2f}</code>\n"
                f"📈 PnL: <code>${pnl_sign}{pnl:.2f}</code> ({pnl_sign}{pnl_pct:.1f}%)\n"
            )
        
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
        
        # Escape any HTML in message to prevent parsing errors
        safe_message = self._escape_html(message)
        formatted = f"{emoji} <b>{level.upper()}</b>\n{safe_message}"
        return await self.send_message(formatted)
    
    async def send_daily_report(self, summary: dict) -> bool:
        """Send daily trading report"""
        message = (
            f"📊 <b>Daily Trading Report</b>\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"💰 Portfolio: <code>${summary.get('portfolio_value', 0):,.2f}</code>\n"
            f"💵 Cash: <code>${summary.get('cash_balance', 0):,.2f}</code>\n"
            f"📈 Return: <code>{summary.get('total_return_pct', 0):+.1f}%</code>\n"
            f"🎯 Win Rate: <code>{summary.get('win_rate', 0):.1f}%</code>\n"
            f"📊 Active: <code>{summary.get('active_positions', 0)}</code>\n"
            f"📋 Total Trades: <code>{summary.get('total_trades', 0)}</code>"
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
        print(f"   Token: {'***' if notifier.token else 'Missing'}")
        print(f"   Chat ID: {notifier.chat_id}")
        
        if not notifier.token or not notifier.chat_id:
            print("❌ Telegram not configured - skipping tests")
            return
        
        # Test trade open
        print("\n📤 Testing trade open notification...")
        await notifier.send_trade_notification({
            'symbol': 'BTC/USDT',
            'side': 'LONG',
            'price': 50000,
            'amount': 0.001,
            'stop_loss': 47500,
            'take_profit': 52500,
            'mode': 'paper'
        })
        
        await asyncio.sleep(2)
        
        # Test trade close (profit)
        print("\n📤 Testing trade close notification...")
        await notifier.send_trade_notification({
            'symbol': 'BTC/USDT',
            'side': 'CLOSE',
            'price': 52000,
            'amount': 0.001,
            'pnl': 20,
            'entry_price': 50000,
            'original_side': 'LONG',
            'reason': 'take_profit',
            'mode': 'paper'
        })
        
        await asyncio.sleep(2)
        
        # Test alert
        print("\n📤 Testing alert notification...")
        await notifier.send_alert("Bot started successfully", "SUCCESS")
        
        print("\n✅ All tests completed!")
    
    asyncio.run(test())