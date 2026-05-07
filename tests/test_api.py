import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config_loader import config, get_binance_client

print(f"Mode: {config.get('trading_mode')}")
print(f"Auth method: {config.get('auth_method')}")

try:
    client = get_binance_client()
    if client:
        # Try a simple API call
        server_time = client.get_server_time()
        print(f"✅ Connected! Server time: {server_time}")
        
        # Get account info (should work with RSA)
        account = client.get_account()
        print(f"✅ Account access successful")
    else:
        print("📄 Paper mode")
except Exception as e:
    print(f"❌ Error: {e}")