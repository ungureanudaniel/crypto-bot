import os
from binance.client import Client
from binance.exceptions import BinanceAPIException
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

def check_balance(client):
    """Check testnet account balance"""
    print("\n" + "=" * 50)
    print("💰 TESTNET ACCOUNT BALANCE")
    print("=" * 50)
    
    try:
        # Get account info
        account = client.get_account()
        
        # Check trading permissions
        can_trade = account.get('canTrade', False)
        print(f"✅ Trading enabled: {can_trade}")
        
        if not can_trade:
            print("   ⚠️  Your API keys need 'Enable Trading' permission!")
        
        # Show all balances
        print("\n📊 Asset Balances:")
        print("-" * 40)
        
        has_balance = False
        for balance in account['balances']:
            free = float(balance['free'])
            locked = float(balance['locked'])
            if free > 0 or locked > 0:
                has_balance = True
                asset = balance['asset']
                # Highlight important assets
                if asset in ['USDC', 'USDT', 'SOL', 'BTC', 'ETH']:
                    print(f"   🔹 {asset}: Free={free:.8f}, Locked={locked:.8f}")
                else:
                    print(f"   {asset}: Free={free:.8f}, Locked={locked:.8f}")
        
        if not has_balance:
            print("   ❌ NO BALANCE FOUND!")
            print("\n🔧 You need to request test funds:")
            print("   1. Go to: https://testnet.binance.vision")
            print("   2. Log in with your testnet credentials")
            print("   3. Click 'Faucet' or 'Get Test Funds'")
            print("   4. Request USDC, SOL, BTC, ETH, etc.")
        
        # Check specific assets for trading
        print("\n🎯 Trading-Ready Check:")
        print("-" * 40)
        
        # Check USDC
        usdc = next((b for b in account['balances'] if b['asset'] == 'USDC'), None)
        if usdc and float(usdc['free']) > 0:
            print(f"   ✅ USDC: {float(usdc['free']):.2f} (can buy)")
        else:
            print("   ❌ Need USDC to buy crypto")
        
        # Check SOL
        sol = next((b for b in account['balances'] if b['asset'] == 'SOL'), None)
        if sol and float(sol['free']) > 0:
            print(f"   ✅ SOL: {float(sol['free']):.4f} (can sell)")
        else:
            print("   ❌ Need SOL to sell")
        
        # Check BTC
        btc = next((b for b in account['balances'] if b['asset'] == 'BTC'), None)
        if btc and float(btc['free']) > 0:
            print(f"   ✅ BTC: {float(btc['free']):.8f}")
        
        # Check ETH
        eth = next((b for b in account['balances'] if b['asset'] == 'ETH'), None)
        if eth and float(eth['free']) > 0:
            print(f"   ✅ ETH: {float(eth['free']):.6f}")
        
        return account
        
    except BinanceAPIException as e:
        print(f"❌ Binance API Error: {e}")
        if e.code == -2015:
            print("   This usually means invalid API key or wrong permissions")
            print("   Make sure your API key has 'Enable Trading' permission")
        return None
    except Exception as e:
        print(f"❌ Error checking balance: {e}")
        return None

def main():
    try:
        api_key = CONFIG.get('binance_api_key')
        api_secret = CONFIG.get('binance_api_secret')
        symbol_list = []
        
        if not api_key or not api_secret:
            print("❌ API keys missing")
            return

        # Initialize client with testnet
        client = Client(api_key=api_key, api_secret=api_secret, testnet=True)
        
        # Test basic connection
        time_data = get_server_time()
        print(f"🕒 Server Time: {time_data['serverTime']}")
        
        # Get exchange info
        print("\n📊 Fetching exchange information...")
        info = client.get_exchange_info()
        symbols = [s['symbol'] for s in info['symbols']]
        
        # Filter for USDC and USDT pairs
        for s in symbols:
            if s.endswith('USDC') or s.endswith('USDT'):
                symbol_list.append(s)
        
        print(f"\n📈 Total trading pairs found: {len(symbol_list)}")
        print(f"   First 20 pairs: {symbol_list[:20]}")
        
        # Check SOL pairs specifically
        sol_pairs = [s for s in symbol_list if s.startswith('SOL')]
        print(f"\n🔍 SOL pairs available: {sol_pairs}")
        
        # CHECK BALANCE - INSERTED HERE
        check_balance(client)
        
        # Get current price for SOL/USDC
        try:
            ticker = client.get_symbol_ticker(symbol='SOLUSDC')
            print(f"\n💰 Current SOL/USDC price: ${float(ticker['price']):.2f}")
        except Exception as e:
            print(f"\n❌ Could not get SOL/USDC price: {e}")
        
        print("\n✅ API test completed successfully!")
        
    except Exception as e:
        print(f"❌ API test failed: {e}")

if __name__ == "__main__":
    main()