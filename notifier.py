import json
import asyncio
from telegram import Bot

# Load config with Telegram token
with open("config.json") as f:
    config = json.load(f)

bot = Bot(token=config['telegram_token'])

def send_telegram_message(message: str):
    """Send telegram message synchronously"""
    try:
        # Run the async function in an event loop
        async def send_async():
            await bot.send_message(chat_id=config['telegram_chat_id'], text=message)
        
        # Create and run event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(send_async())
        loop.close()
        
        print(f"Message sent: {message}")
        return True
    except Exception as e:
        print(f"Failed to send message: {e}")
        return False