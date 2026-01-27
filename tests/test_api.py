import os
from binance.client import Client
import requests
import logging
import time
import sys
from datetime import datetime, timedelta
from typing import Optional, Dict
# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
BASE_URL = "https://api1.binance.com/"
# -------------------------------------------------------------------
# CONFIG LOADING
# -------------------------------------------------------------------
try:
    # Add parent directory to path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config_loader import config
    CONFIG = config.config
    logger.info(f"✅ Config loaded: {CONFIG.get('trading_mode', 'paper')}")
except ImportError:
    logger.warning("⚠️ Could not import config_loader, using defaults")
    CONFIG = {'trading_mode': 'paper', 'testnet': False, 'rate_limit_delay': 0.5}
logging.info("🔧 Configuration loaded for data feed. Trading mode: %s", CONFIG.get('trading_mode', 'paper'))
# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def get_server_time():
    endpoint = "/api/v3/time"
    response = requests.get(BASE_URL + endpoint)
    return response.json()

def get_price(symbol:str):
    url = f"{BASE_URL}/api/v3/ticker/price"
    params = {"symbol": symbol}
    response = requests.get(url, params=params)
    return response.json()

def main():
    try:
        api_key = CONFIG.get('binance_api_key')
        api_secret = CONFIG.get('binance_api_secret')
        symbol = []
        if not api_key or not api_secret:
            print("❌ API keys missing")
            return

        client = Client(api_key=api_key, api_secret=api_secret, testnet=True)
    
        time_data = get_server_time()
        print(f"Server Time: {time_data['serverTime']}")
        info = client.get_exchange_info()
        symbols = [s['symbol'] for s in info['symbols']]
        for s in symbols:
            if s.endswith('USDC') or s.endswith('USDT'):
                symbol.append(s)
                continue
        
        print(f"Symbols are {symbol}")
        
        print("✅ API test completed successfully!")
    except Exception as e:
        print(f"❌ API test failed: {e}")

if __name__ == "__main__":
    main()

