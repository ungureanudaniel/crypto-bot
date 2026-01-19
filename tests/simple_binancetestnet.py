# test_api_keys.py
from binance.client import Client
import json

# Load config
with open('config.json', 'r') as f:
    config = json.load(f)

api_key = config.get('binance_testnet_api_key', '')
api_secret = config.get('binance_testnet_api_secret', '')

print(f"API Key: {api_key[:10]}...")
print(f"Secret: {api_secret[:10]}...")

# Test connection
try:
    client = Client(api_key, api_secret, testnet=True)
    
    # Test public endpoint first
    print("\n✅ Public connection OK")
    prices = client.get_all_tickers()
    print(f"   BTC price: ${next(p['price'] for p in prices if p['symbol'] == 'BTCUSDT')}")
    
    # Test private endpoint
    print("\n🔐 Testing private endpoint...")
    account = client.get_account()
    print(f"✅ Private connection OK!")
    print(f"   Account can trade: {account.get('canTrade', False)}")
    
    # Show balances
    print("\n💰 Balances:")
    for balance in account['balances']:
        free = float(balance['free'])
        locked = float(balance['locked'])
        if free > 0 or locked > 0:
            print(f"   {balance['asset']}: Free={free}, Locked={locked}")
            
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    print("\n🔑 Get NEW keys from: https://testnet.binance.vision/")