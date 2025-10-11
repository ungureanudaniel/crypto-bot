import json
from telegram import Bot

# Load config with Telegram token
with open("config.json") as f:
    config = json.load(f)

bot = Bot(token=config['telegram_token'])

def send_telegram_message(message: str):
    try:
        bot.send_message(chat_id=config['telegram_chat_id'], text=message)
        print(f"Message sent: {message}")
    except Exception as e:
        print(f"Failed to send message: {e}")