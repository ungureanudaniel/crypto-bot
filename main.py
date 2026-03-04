# main.py - SIMPLE LAUNCHER WITH EVENT LOOP FIX
import sys
import os
import asyncio

# -------------------------------------------------------------------
# SETUP PATHS
# -------------------------------------------------------------------
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'modules'))

# -------------------------------------------------------------------
# FIX EVENT LOOP FOR WINDOWS
# -------------------------------------------------------------------
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

print("\n" + "=" * 60)
print("🚀 BINANCE AI TRADING BOT")
print("=" * 60)
print("\nStarting... Press Ctrl+C to stop\n")

# Check trading mode
try:
    from config_loader import config
    trading_mode = config.config.get('trading_mode', 'paper')
    print(f"📊 Trading Mode: {trading_mode.upper()}")
except Exception as e:
    print(f"⚠️ Could not get trading mode: {e}")

# Train model (only in paper/testnet mode to avoid live disruptions)
try:
    from modules.regime_switcher import train_model
    print("🔄 Training regime detection model...")
    train_model()
    print("✅ Model trained")
except Exception as e:
    print(f"⚠️ Could not train model: {e}")

# Start bot
print("\n🤖 Starting Telegram bot...")
print("📱 Use /start in Telegram")
print("🛑 Press Ctrl+C to stop\n")

try:
    from services.telegram_bot import run_telegram_bot
    run_telegram_bot()
except KeyboardInterrupt:
    print("\n🛑 Bot stopped by user")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n✅ System stopped")