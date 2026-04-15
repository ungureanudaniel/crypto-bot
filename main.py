import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import asyncio
import logging
import traceback
from modules.logger_config import init_logging

# 1. SETUP PATHS - Anchor everything to the script location
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 2. EVENT LOOP FIX (Windows specific)
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

def main():
    # Initialize Notifier & Logging
    try:
        from services.notifier import notifier
        logger = init_logging(verbose=True, notifier=notifier)
    except ImportError:
        logger = init_logging(verbose=True)
        logger.warning("⚠️ Notifier module not found, continuing without Telegram.")

    logger.info("🚀 Bot initialization sequence started...")

    # Load Config
    try:
        from config_loader import config
        trading_mode = config.config.get('trading_mode', 'paper').lower()
        logger.info(f"📊 Mode: {trading_mode.upper()}")
    except Exception as e:
        logger.error(f"❌ Failed to load config: {e}")
        return

    # 3. PRE-FLIGHT CHECKS (Connectivity)
    try:
        from modules.data_feed import get_current_price
        test_price = get_current_price("BTC/USDT")
        if test_price:
            logger.info(f"✅ API Connectivity verified. BTC: ${test_price}")
    except Exception as e:
        logger.error(f"❌ Connection failed: {e}")
        if trading_mode != 'paper':
            return # Don't start live/testnet if we can't get data

    # 4. REGIME MODEL TRAINING
    if trading_mode in ('paper', 'testnet'):
        try:
            from modules.regime_switcher import train_model
            logger.info("🔄 Training regime detection model...")
            train_model()
        except Exception as e:
            logger.warning(f"⚠️ Regime training failed (non-critical): {e}")

    # 5. START SYSTEM
    try:
        from services.telegram_bot import run_telegram_bot
        logger.info("🤖 Starting Telegram Interface & Scheduler...")
        
        # This usually blocks until the bot is stopped
        run_telegram_bot()

    except KeyboardInterrupt:
        logger.info("🛑 Stop signal received (Ctrl+C).")
    except Exception as e:
        logger.error(f"❌ Critical runtime error: {e}")
        traceback.print_exc()
    finally:
        # 6. GRACEFUL SHUTDOWN
        logger.info("🧹 Cleaning up resources...")
        try:
            from services.scheduler import stop_scheduler
            stop_scheduler()
            logger.info("✅ Scheduler stopped.")
        except Exception:
            pass
        logger.info("👋 System exit.")

if __name__ == "__main__":
    main()