# check_testnet_balance.py
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

# Get account info

print("💰 Testnet Account Balance:")
print("-" * 40)

# Show all assets with non-zero balance
for balance in account['balances']:
    free = float(balance['free'])
    locked = float(balance['locked'])
    if free > 0 or locked > 0:
        print(f"{balance['asset']}: Free={free:.8f}, Locked={locked:.8f}")

# Check if you have any USDC specifically
usdc_balance = next((b for b in account['balances'] if b['asset'] == 'USDC'), None)
if usdc_balance and float(usdc_balance['free']) > 0:
    print(f"\n✅ You have USDC: {float(usdc_balance['free']):.2f}")
else:
    print(f"\n❌ You have NO USDC balance!")
    
# Check if you have SOL
sol_balance = next((b for b in account['balances'] if b['asset'] == 'SOL'), None)
if sol_balance and float(sol_balance['free']) > 0:
    print(f"✅ You have SOL: {float(sol_balance['free']):.4f}")
else:
    print(f"❌ You have NO SOL balance!")