import os
from binance.client import Client
from binance.exceptions import BinanceAPIException
import requests
import logging
import sys
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_URL = "https://api1.binance.com/"

# -------------------------------------------------------------------
# CONFIG LOADING – NO DEFAULTS, RAISE ERRORS IMMEDIATELY
# -------------------------------------------------------------------
# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config_loader import config

# This will raise KeyError if any required key is missing
CONFIG = config.config
trading_mode = CONFIG['trading_mode'].lower()

if trading_mode not in ['live', 'testnet']:
    raise ValueError(f"❌ Invalid trading_mode '{trading_mode}'. Must be 'live' or 'testnet'.")

# Get API keys – these must exist (config_loader already validated them)
api_key = CONFIG['binance_api_key']
api_secret = CONFIG['binance_api_secret']

logger.info(f"✅ Config loaded: {trading_mode}")

# -------------------------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------------------------
def get_server_time():
    """Get Binance server time (used for timestamp sync)"""
    endpoint = "/api/v3/time"
    response = requests.get(BASE_URL + endpoint)
    return response.json()

def check_balance(client):
    """Check account balance (works for both live and testnet)"""
    print("\n" + "=" * 50)
    print("💰 ACCOUNT BALANCE")
    print("=" * 50)
    
    try:
        account = client.get_account()
        can_trade = account.get('canTrade', False)
        print(f"✅ Trading enabled: {can_trade}")
        
        if not can_trade:
            print("   ⚠️  Your API keys need 'Enable Trading' permission!")
        
        print("\n📊 Asset Balances:")
        print("-" * 40)
        
        has_balance = False
        for balance in account['balances']:
            free = float(balance['free'])
            locked = float(balance['locked'])
            if free > 0 or locked > 0:
                has_balance = True
                asset = balance['asset']
                if asset in ['USDC', 'USDT', 'SOL', 'BTC', 'ETH']:
                    print(f"   🔹 {asset}: Free={free:.8f}, Locked={locked:.8f}")
                else:
                    print(f"   {asset}: Free={free:.8f}, Locked={locked:.8f}")
        
        if not has_balance:
            print("   ❌ NO BALANCE FOUND!")
            if trading_mode == 'testnet':
                print("\n🔧 You need to request test funds:")
                print("   1. Go to: https://testnet.binance.vision")
                print("   2. Log in with your testnet credentials")
                print("   3. Click 'Faucet' or 'Get Test Funds'")
                print("   4. Request USDC, SOL, BTC, ETH, etc.")
        
        # Check specific assets
        print("\n🎯 Trading-Ready Check:")
        print("-" * 40)
        
        usdc = next((b for b in account['balances'] if b['asset'] == 'USDC'), None)
        if usdc and float(usdc['free']) > 0:
            print(f"   ✅ USDC: {float(usdc['free']):.2f} (can buy)")
        else:
            print("   ❌ Need USDC to buy crypto")
        
        sol = next((b for b in account['balances'] if b['asset'] == 'SOL'), None)
        if sol and float(sol['free']) > 0:
            print(f"   ✅ SOL: {float(sol['free']):.4f} (can sell)")
        else:
            print("   ❌ Need SOL to sell")
        
        btc = next((b for b in account['balances'] if b['asset'] == 'BTC'), None)
        if btc and float(btc['free']) > 0:
            print(f"   ✅ BTC: {float(btc['free']):.8f}")
        
        eth = next((b for b in account['balances'] if b['asset'] == 'ETH'), None)
        if eth and float(eth['free']) > 0:
            print(f"   ✅ ETH: {float(eth['free']):.6f}")
        
        return account
        
    except BinanceAPIException as e:
        print(f"❌ Binance API Error: {e}")
        if e.code == -2015:
            print("   This usually means invalid API key or wrong permissions")
            print("   Make sure your API key has 'Enable Trading' permission")
        raise
    except Exception as e:
        print(f"❌ Error checking balance: {e}")
        raise

def main():
    try:
        # Initialize client based on trading mode
        use_testnet = (trading_mode == 'testnet')
        client = Client(api_key, api_secret, testnet=use_testnet)
        
        # Test basic connection
        time_data = get_server_time()
        print(f"🕒 Server Time: {time_data['serverTime']}")
        
        # Get exchange info
        print("\n📊 Fetching exchange information...")
        info = client.get_exchange_info()
        symbols = [s['symbol'] for s in info['symbols']]
        
        # Filter for USDC and USDT pairs
        symbol_list = [s for s in symbols if s.endswith('USDC') or s.endswith('USDT')]
        print(f"\n📈 Total trading pairs found: {len(symbol_list)}")
        print(f"   First 20 pairs: {symbol_list[:20]}")
        
        # Check SOL pairs
        sol_pairs = [s for s in symbol_list if s.startswith('SOL')]
        print(f"\n🔍 SOL pairs available: {sol_pairs}")
        
        # Check balance
        check_balance(client)
        
        # Get current price for SOL/USDC (if pair exists)
        if 'SOLUSDC' in symbols:
            ticker = client.get_symbol_ticker(symbol='SOLUSDC')
            print(f"\n💰 Current SOL/USDC price: ${float(ticker['price']):.2f}")
        else:
            print("\n⚠️ SOL/USDC pair not available on this endpoint")
        
        print("\n✅ API test completed successfully!")
        
    except BinanceAPIException as e:
        print(f"❌ Binance API Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ API test failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()