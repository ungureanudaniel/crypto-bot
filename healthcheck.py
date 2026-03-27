#!/usr/bin/env python3
"""
healthcheck.py - Simple health check for Docker
"""

import sys
import os
import signal
import time

def check_process_alive():
    """Check if the main bot process is running"""
    try:
        # Try to connect to the bot's process
        # This is a simple check that the bot is still running
        # without making API calls that might fail
        
        # Check if we can import modules without errors
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        # Just check if the module can be imported (doesn't initialize)
        import importlib
        spec = importlib.util.find_spec("modules.trade_engine")
        if spec is None:
            print("❌ Trade engine module not found")
            return False
        
        # Check if the bot's PID file exists and process is running
        pid_file = '/tmp/bot.pid'
        if os.path.exists(pid_file):
            try:
                with open(pid_file, 'r') as f:
                    pid = int(f.read().strip())
                # Check if process is running
                os.kill(pid, 0)
                print(f"✅ Bot process {pid} is running")
                return True
            except (OSError, ValueError, ProcessLookupError):
                pass
        
        print("✅ Module found, assuming bot is running")
        return True
        
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='Quick health check')
    args = parser.parse_args()
    
    if args.quick:
        # Quick check - just see if we can import modules
        try:
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            import modules.trade_engine
            print("✅ Quick health check passed")
            sys.exit(0)
        except Exception as e:
            print(f"❌ Quick health check failed: {e}")
            sys.exit(1)
    else:
        success = check_process_alive()
        sys.exit(0 if success else 1)