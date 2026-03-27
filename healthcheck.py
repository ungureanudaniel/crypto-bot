import sys
import os
import json
import time
import logging

# Add paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'modules'))

def check_bot_health():
    """Check if the bot is functioning properly"""
    try:
        # Check if trade_engine is importable and has open positions
        from modules.trade_engine import trading_engine
        
        # Basic check: Can we get current prices?
        prices = trading_engine.get_current_prices()
        if not prices:
            print("⚠️ No price data available")
            return 1  # Unhealthy
            
        # Check if we have at least some prices
        if len(prices) < 2:
            print(f"⚠️ Only {len(prices)} price feeds available")
            return 1
            
        # Check if the main process is still responsive
        # (This is a simple check - your bot might have additional health indicators)
        print(f"✅ Bot healthy - {len(prices)} price feeds, {len(trading_engine.open_positions)} active positions")
        return 0  # Healthy
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return 1
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return 1

if __name__ == "__main__":
    # Simple health check that doesn't require --quick argument
    # Just check basic bot functionality
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        # Quick check - just verify the process is running
        try:
            from modules.trade_engine import trading_engine
            # Check if the engine is responsive
            _ = trading_engine.get_cash_balance()
            print("✅ Quick health check passed")
            sys.exit(0)
        except:
            sys.exit(1)
    else:
        # Full health check
        sys.exit(check_bot_health())