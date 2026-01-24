# services/scheduler_debug.py
import schedule
import time
import threading
import logging
import traceback

logger = logging.getLogger(__name__)

def debug_job():
    """Simple job to test if scheduler works"""
    logger.info("✅ DEBUG: Scheduler job running at %s", time.strftime("%H:%M:%S"))

def start_debug_scheduler():
    """Start a debug scheduler that should never die"""
    logger.info("🟢 Starting DEBUG scheduler...")
    
    # Clear any existing jobs
    schedule.clear()
    
    # Add simple jobs
    schedule.every(10).seconds.do(debug_job)
    schedule.every(1).minutes.do(lambda: logger.info("✅ 1-minute job running"))
    
    def super_safe_scheduler_loop():
        """Ultra-safe scheduler loop with maximum error handling"""
        logger.info("📡 DEBUG Scheduler loop started")
        
        error_count = 0
        max_errors = 10
        
        while error_count < max_errors:
            try:
                # Run pending jobs
                schedule.run_pending()
                
                # Reset error count on success
                if error_count > 0:
                    logger.info("✅ Scheduler recovered from errors")
                    error_count = 0
                    
            except KeyboardInterrupt:
                logger.info("🛑 Scheduler interrupted")
                break
            except Exception as e:
                error_count += 1
                logger.error(f"🔥 Scheduler error #{error_count}: {e}")
                logger.error(traceback.format_exc())
                
                if error_count >= max_errors:
                    logger.critical("🚨 Too many scheduler errors, stopping")
                    break
                    
                # Wait before retrying
                time.sleep(min(30, error_count * 5))  # Exponential backoff
            
            # Normal sleep
            time.sleep(1)
        
        logger.info("📡 DEBUG Scheduler loop ended")
    
    # Start thread
    thread = threading.Thread(
        target=super_safe_scheduler_loop,
        daemon=True,
        name="DebugScheduler"
    )
    thread.start()
    
    logger.info("✅ DEBUG Scheduler started")
    return thread

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    start_debug_scheduler()
    # Keep main thread alive
    while True:
        time.sleep(10)  