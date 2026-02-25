import os
import json
import logging
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

class Config:
    """Load configuration from both .env and config.json"""
    
    def __init__(self):
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load config from both sources"""
        
        # 1. Load from config.json (trading settings)
        if not os.path.exists("config.json"):
            raise FileNotFoundError("❌ CRITICAL: config.json not found! Bot cannot start without configuration.")
        
        with open("config.json", "r") as f:
            file_config = json.load(f)
        
        # 2. Load from environment variables (secrets)
        trading_mode = os.getenv('TRADING_MODE')
        if not trading_mode:
            raise ValueError("❌ CRITICAL: TRADING_MODE not set in .env file!")
        
        trading_mode = trading_mode.lower()
        
        env_config = {
            'trading_mode': trading_mode,
            'telegram_token': os.getenv('TELEGRAM_TOKEN', ''),
            'telegram_chat_id': os.getenv('TELEGRAM_CHAT_ID', ''),
        }
        
        # 3. Set API keys based on mode - FAIL if missing
        if trading_mode == 'live':
            api_key = os.getenv('BINANCE_LIVE_API_KEY')
            api_secret = os.getenv('BINANCE_LIVE_API_SECRET')
            if not api_key or not api_secret:
                raise ValueError("❌ CRITICAL: Live trading requires BINANCE_LIVE_API_KEY and BINANCE_LIVE_API_SECRET in .env")
            env_config['binance_api_key'] = api_key
            env_config['binance_api_secret'] = api_secret
            
        elif trading_mode == 'testnet':
            api_key = os.getenv('BINANCE_TESTNET_API_KEY')
            api_secret = os.getenv('BINANCE_TESTNET_PRIVATE_KEY')
            if not api_key or not api_secret:
                raise ValueError("❌ CRITICAL: Testnet trading requires BINANCE_TESTNET_API_KEY and BINANCE_TESTNET_PRIVATE_KEY in .env")
            env_config['binance_api_key'] = api_key
            env_config['binance_api_secret'] = api_secret
        else:
            raise ValueError(f"❌ CRITICAL: Invalid TRADING_MODE '{trading_mode}'. Must be 'live' or 'testnet'")
        
        # 4. Merge configs
        merged_config = {**file_config, **env_config}
        
        # 5. Set derived values
        merged_config['live_trading'] = trading_mode == 'live'
        merged_config['testnet'] = trading_mode == 'testnet'
        
        # 6. Set API URLs
        if merged_config['testnet']:
            merged_config['binance_api_url'] = 'https://testnet.binance.vision/api'
        else:
            merged_config['binance_api_url'] = 'https://api.binance.com'
        
        return merged_config
    
    def get(self, key: str, default=None):
        """Get config value - NO DEFAULTS!"""
        if key not in self.config:
            raise KeyError(f"❌ CRITICAL: Required config key '{key}' not found!")
        return self.config[key]
    
    def __getitem__(self, key):
        """Get config value using dict syntax - NO DEFAULTS!"""
        if key not in self.config:
            raise KeyError(f"❌ CRITICAL: Required config key '{key}' not found!")
        return self.config[key]
    
    def __contains__(self, key):
        """Check if key exists"""
        return key in self.config

# Global config instance
config = Config()

# -------------------------------------------------------------------
# Shared Binance client (singleton)
# -------------------------------------------------------------------
_binance_client = None

def get_binance_client():
    """Return a singleton Binance client instance."""
    global _binance_client
    if _binance_client is not None:
        return _binance_client

    from binance.client import Client
    from binance.exceptions import BinanceAPIException

    trading_mode = config.config['trading_mode'].lower()
    api_key = config.config['binance_api_key']
    api_secret = config.config['binance_api_secret']

    logger.info(f"Initializing shared Binance client for {trading_mode} mode")

    try:
        if trading_mode == 'testnet':
            client = Client(api_key, api_secret, testnet=True)
        else:
            client = Client(api_key, api_secret)

        # Test connection – fail fast
        client.ping()
        logger.info("✅ Shared client ping successful")
        client.get_account()
        logger.info("✅ Shared client authentication successful")
        logger.info(f"🌐 API URL: {client.API_URL}")

        _binance_client = client
        return client

    except (ImportError, BinanceAPIException, Exception) as e:
        logger.error(f"❌ Failed to initialize shared Binance client: {e}")
        raise

if __name__ == "__main__":
    # Test loading
    CONFIG = config.config
    print("\n📋 Configuration Summary:")
    print(f"Trading mode: {CONFIG.get('trading_mode')}")
    print(f"Testnet flag: {CONFIG.get('testnet')}")
    print(f"binance_api_url: {CONFIG.get('binance_api_url')}")
    # Optionally test client
    # client = get_binance_client()
    # print("✅ Client created")
