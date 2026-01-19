import logging
import threading
import time
import pandas as pd
import numpy as np
from ta.trend import MACD, ADXIndicator
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import ccxt

from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

# === CONFIG ===
TELEGRAM_TOKEN = 'YOUR_TELEGRAM_BOT_TOKEN'
CHAT_ID = 'YOUR_TELEGRAM_CHAT_ID'  # your Telegram user ID or group ID
COINS = ['BTC/USDT', 'ETH/USDT']
EXCHANGE = ccxt.binance()
TIMEFRAME = '1w'
LOOKBACK = 120

# Global control flag
run_bot = False

# === FEATURE ENGINEERING, LABELING, MODEL TRAINING (same as before) ===
# ... (use the same add_features, label_regime, train_model functions here) ...

def add_features(df):
    df['rsi'] = RSIIndicator(df['close'], window=14).rsi()
    macd = MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['adx'] = ADXIndicator(df['high'], df['low'], df['close']).adx()
    bb = BollingerBands(df['close'])
    df['bb_width'] = bb.bollinger_wband()
    df['volume_change'] = df['volume'].pct_change()
    df['returns'] = df['close'].pct_change()
    df = df.dropna()
    return df

def label_regime(df):
    conditions = [
        (df['adx'] < 20) & (df['bb_width'] < 0.05),
        (df['adx'] >= 20),
        (df['volume_change'] > 0.5)
    ]
    choices = [0, 1, 2]
    df['regime'] = np.select(conditions, choices, default=0)
    return df

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    return model

def execute_strategy(coin, regime, bot):
    msg = ""
    if regime == 0:
        msg = f"[{coin}] Weekly Rangebound: RSI + BB mean reversion strategy."
    elif regime == 1:
        msg = f"[{coin}] Weekly Trending: SuperTrend + ADX trend following strategy."
    elif regime == 2:
        msg = f"[{coin}] Weekly Breakout: VWAP + Volume momentum strategy."
    print(msg)
    # Send message to Telegram
    if bot:
        bot.send_message(chat_id=CHAT_ID, text=msg)

async def start_bot(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global run_bot
    if run_bot:
        await update.message.reply_text("Bot is already running!")
        return
    run_bot = True
    await update.message.reply_text("Starting weekly trading bot...")
    threading.Thread(target=run_trading_bot, args=(context.bot,), daemon=True).start()

async def stop_bot(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global run_bot
    run_bot = False
    await update.message.reply_text("Stopping weekly trading bot...")

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global run_bot
    status_msg = "Running" if run_bot else "Stopped"
    await update.message.reply_text(f"Weekly trading bot status: {status_msg}")

def run_trading_bot(bot):
    global run_bot
    # Train model once
    all_features = []
    all_labels = []
    for coin in COINS:
        ohlcv = EXCHANGE.fetch_ohlcv(coin, timeframe=TIMEFRAME, limit=LOOKBACK)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df = add_features(df)
        df = label_regime(df)
        features = df[['rsi', 'macd', 'macd_signal', 'adx', 'bb_width', 'volume_change', 'returns']]
        labels = df['regime']
        all_features.append(features)
        all_labels.append(labels)
    X = pd.concat(all_features)
    y = pd.concat(all_labels)
    model = train_model(X, y)
    bot.send_message(chat_id=CHAT_ID, text="Model trained. Starting live regime detection.")

    while run_bot:
        for coin in COINS:
            ohlcv = EXCHANGE.fetch_ohlcv(coin, timeframe=TIMEFRAME, limit=LOOKBACK)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df = add_features(df)
            latest = df.iloc[-1]
            X_live = latest[['rsi', 'macd', 'macd_signal', 'adx', 'bb_width', 'volume_change', 'returns']].values.reshape(1, -1)
            regime = model.predict(X_live)[0]
            execute_strategy(coin, regime, bot)
        bot.send_message(chat_id=CHAT_ID, text="Waiting for next weekly candle...")
        time.sleep(60 * 60 * 24 * 7)  # sleep 1 week

def main():
    application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    application.add_handler(CommandHandler("startbot", start_bot))
    application.add_handler(CommandHandler("stopbot", stop_bot))
    application.add_handler(CommandHandler("status", status))

    print("Telegram bot started. Use /startbot to run weekly trading bot.")
    application.run_polling()

if __name__ == '__main__':
    main()
