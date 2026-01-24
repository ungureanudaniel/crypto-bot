# services/scheduler_minimal.py
import schedule
import time
import threading
import logging

logger = logging.getLogger(__name__)

def simple_job():
    """A simple job that just logs"""
    logger.info("✅ Simple job running")

def start_simple_scheduler(bot_data=None):
    """Start a minimal scheduler"""
    logger.info("🟢 Starting SIMPLE scheduler...")
    
    # Clear and add one simple job
    schedule.clear()
    schedule.every(1).minutes.do(simple_job)
    
    def run_scheduler():
        logger.info("📡 Simple scheduler loop started")
        try:
            while True:
                schedule.run_pending()
                time.sleep(1)
        except Exception as e:
            logger.error(f"🔥 Simple scheduler error: {e}")
            raise
    
    thread = threading.Thread(target=run_scheduler, daemon=True, name="SimpleScheduler")
    thread.start()
    logger.info("✅ Simple scheduler started")
    return thread

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    start_simple_scheduler()
    # Keep main thread alive
    while True:
        time.sleep(10)