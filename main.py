# main.py - FIXED VERSION
import logging
import time
import signal
import sys
import os
import threading
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.regime_switcher import train_model
from services.scheduler import start_schedulers
from services.telegram_bot import start_telegram_bot
from services.notifier import notifier

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class TradingSystem:
    def __init__(self):
        self.running = False
        self.telegram_thread = None
        self.scheduler_thread = None
        logger.info("🚀 Trading System Initializing...")

    def start(self):
        """Start system"""
        logger.info("=" * 60)
        logger.info("🚀 STARTING TRADING SYSTEM")
        logger.info("=" * 60)

        self.running = True

        try:
            # 1️⃣ Startup notification
            logger.info("📢 Sending startup notification...")
            notifier.send_message("🚀 Trading System Starting...")

            # 2️⃣ Train model
            logger.info("🔄 Training regime detection model...")
            train_model()

            # 3️⃣ Start scheduler in background thread
            logger.info("⏰ Starting scheduler...")
            bot_data = {
                "run_bot": True,
                "trading_interval": "15m",
            }

            self.scheduler_thread = threading.Thread(
                target=start_schedulers,
                args=(bot_data,),
                daemon=True,
                name="SchedulerThread"
            )
            self.scheduler_thread.start()
            logger.info("✅ Scheduler started")

            # 4️⃣ Start Telegram bot in its own thread
            logger.info("🤖 Starting Telegram bot...")
            logger.info("💡 Use /start command in Telegram to begin")
            
            self.telegram_thread = threading.Thread(
                target=start_telegram_bot,
                daemon=True,
                name="TelegramBotThread"
            )
            self.telegram_thread.start()
            logger.info("✅ Telegram bot started")

            # Wait for threads (with timeout)
            self.wait_for_threads()

        except KeyboardInterrupt:
            logger.info("\n🛑 Received Ctrl+C")
            self.stop()
        except Exception as e:
            logger.error(f"❌ System error: {e}")
            self.stop()

    def wait_for_threads(self):
        """Wait for threads with proper handling"""
        try:
            # Keep main thread alive, checking threads periodically
            while self.running:
                # Check if Telegram thread is alive
                if self.telegram_thread and not self.telegram_thread.is_alive():
                    logger.warning("⚠️ Telegram thread died, restarting...")
                    self.telegram_thread = threading.Thread(
                        target=start_telegram_bot,
                        daemon=True,
                        name="TelegramBotThreadRestart"
                    )
                    self.telegram_thread.start()
                
                # Check if scheduler thread is alive
                if self.scheduler_thread and not self.scheduler_thread.is_alive():
                    logger.error("❌ Scheduler thread died!")
                    self.running = False
                    break
                
                time.sleep(5)
                
        except KeyboardInterrupt:
            logger.info("\n🛑 Shutdown requested")

    def stop(self):
        """Stop the system gracefully"""
        logger.info("🛑 Stopping trading system...")
        self.running = False
        
        try:
            notifier.send_message("🛑 Trading System Stopped")
        except:
            pass
        
        # Give threads time to cleanup
        time.sleep(2)
        
        logger.info("✅ Trading system stopped")
        sys.exit(0)


def signal_handler(sig, frame):
    """Handle Ctrl+C"""
    print("\n🛑 Received shutdown signal")
    sys.exit(0)


if __name__ == "__main__":
    start_telegram_bot() # Ensure bot starts in main thread
    # Handle Ctrl+C gracefully 
    signal.signal(signal.SIGINT, signal_handler)
    
    print("\n" + "=" * 60)
    print("🚀 BINANCE AI TRADING SYSTEM")
    print("=" * 60)
    print("Starting in 3 seconds... Press Ctrl+C to stop")
    print("\nInitializing components...")

    time.sleep(3)

    # Start the system
    system = TradingSystem()
    system.start()