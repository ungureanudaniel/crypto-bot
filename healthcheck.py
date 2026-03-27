"""
Simple health check for Docker
Returns success if the bot is running and responsive
"""

import sys
import os
import argparse
import logging
import traceback

# Add paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'modules'))

def quick_check():
    """Quick health check - just verify the bot process is responsive"""
    try:
        # Try to import and access the trading engine
        from modules.trade_engine import trading_engine
        
        # Simple check - see if we can get a property
        _ = trading_engine.trading_mode
        
        print("✅ Quick health check passed")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Quick check failed: {e}")
        return False

def full_check():
    """Full health check - check bot functionality"""
    try:
        # Import trading engine
        from modules.trade_engine import trading_engine
        
        # Check if trading engine is initialized
        if trading_engine is None:
            print("❌ Trading engine not initialized")
            return False
        
        # Check if we can get cash balance (lightweight check)
        try:
            cash = trading_engine.get_cash_balance()
            print(f"✅ Cash balance check passed: ${cash:.2f}")
        except Exception as e:
            print(f"⚠️ Cash balance check warning: {e}")
            # Don't fail on this - it might be paper mode
        
        # Check if we have price data
        try:
            prices = trading_engine.get_current_prices()
            if prices:
                print(f"✅ Price data available: {len(prices)} symbols")
            else:
                print("⚠️ No price data available")
        except Exception as e:
            print(f"⚠️ Price data warning: {e}")
        
        # Check if scheduler is running (optional)
        try:
            from services.scheduler import _scheduler_thread
            if _scheduler_thread and _scheduler_thread.is_alive():
                print("✅ Scheduler thread is running")
            else:
                print("⚠️ Scheduler thread not running")
        except:
            pass
        
        print("✅ Full health check passed")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"❌ Full health check failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', 
                       help='Quick health check (no API calls)')
    args = parser.parse_args()
    
    if args.quick:
        success = quick_check()
    else:
        success = full_check()
    
    # Exit with appropriate code (0 = success, 1 = failure)
    sys.exit(0 if success else 1)