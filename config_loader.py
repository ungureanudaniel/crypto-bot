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
            # Futures keys — can reuse spot keys or use dedicated ones
            env_config['binance_futures_api_key'] = os.getenv('BINANCE_FUTURES_API_KEY', api_key)
            env_config['binance_futures_api_secret'] = os.getenv('BINANCE_FUTURES_API_SECRET', api_secret)

        elif trading_mode == 'paper':
            env_config['binance_api_key'] = ''
            env_config['binance_api_secret'] = ''
            env_config['binance_futures_api_key'] = ''
            env_config['binance_futures_api_secret'] = ''

        elif trading_mode == 'testnet':
            api_key = os.getenv('BINANCE_TESTNET_API_KEY')
            api_secret = os.getenv('BINANCE_TESTNET_PRIVATE_KEY')
            if not api_key or not api_secret:
                raise ValueError("❌ CRITICAL: Testnet trading requires BINANCE_TESTNET_API_KEY and BINANCE_TESTNET_PRIVATE_KEY in .env")
            env_config['binance_api_key'] = api_key
            env_config['binance_api_secret'] = api_secret
            # Futures testnet uses the same credentials by default
            env_config['binance_futures_api_key'] = os.getenv('BINANCE_FUTURES_TESTNET_API_KEY', api_key)
            env_config['binance_futures_api_secret'] = os.getenv('BINANCE_FUTURES_TESTNET_API_SECRET', api_secret)
        else:
            raise ValueError(f"❌ CRITICAL: Invalid TRADING_MODE '{trading_mode}'. Must be 'live', 'testnet', or 'paper'")
        
        # 4. Merge configs
        merged_config = {**file_config, **env_config}
        
        # 5. Set derived values
        merged_config['live_trading'] = trading_mode == 'live'
        merged_config['paper_trading'] = trading_mode == 'paper'

        # 6. Set API URLs
        if merged_config['paper_trading']:
            merged_config['binance_api_url'] = 'https://testnet.binance.vision/api'
            merged_config['binance_futures_url'] = 'https://testnet.binancefuture.com'
        else:
            merged_config['binance_api_url'] = 'https://api.binance.com'
            merged_config['binance_futures_url'] = 'https://fapi.binance.com'

        # 7. Futures trading flag — enabled unless explicitly disabled in config.json
        merged_config['enable_futures'] = merged_config.get('enable_futures', True)
        
        return merged_config
    
    def get(self, key: str, default=None):
        """Get config value with optional default (raises only if no default provided)"""
        if key not in self.config:
            if default is None:
                raise KeyError(f"❌ CRITICAL: Required config key '{key}' not found!")
            return default
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

def get_pair_config(symbol: str) -> dict:
    """
    Return per-pair configuration for a symbol.
    If not defined, return an empty dict (global defaults will be used).
    """
    per_pair = config.config.get('per_pair', {})
    return per_pair.get(symbol, {})
# -------------------------------------------------------------------
# Shared Binance client (singleton)
# -------------------------------------------------------------------
_binance_client = None

def get_binance_client():
    """Return a singleton Binance client instance."""
    global _binance_client
    if _binance_client is not None:
        return _binance_client

    trading_mode = config.config['trading_mode'].lower()
    
    # PAPER MODE: Return None (no client needed)
    if trading_mode == 'paper':
        logger.info("📄 Paper mode - no Binance client needed")
        _binance_client = None
        return None

    # For live/testnet, require API keys
    api_key = config.config.get('binance_api_key', '')
    api_secret = config.config.get('binance_api_secret', '')
    
    if not api_key or not api_secret:
        raise ValueError(f"❌ CRITICAL: API keys required for {trading_mode} mode")

    from binance.client import Client
    from binance.exceptions import BinanceAPIException

    logger.info(f"Initializing shared Binance client for {trading_mode} mode")

    try:
        if trading_mode == 'testnet':
            client = Client(api_key, api_secret, testnet=True)
        else:  # live mode
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

# -------------------------------------------------------------------
# Shared Binance Futures client (singleton)
# -------------------------------------------------------------------
_futures_client = None

def get_futures_client():
    """
    Return a singleton Binance USDT-M Futures client.
    Returns None in paper mode — futures positions are simulated in portfolio.py.
    Tries binance-futures-connector (UMFutures) first, falls back to python-binance.
    """
    global _futures_client
    if _futures_client is not None:
        return _futures_client

    trading_mode = config.config['trading_mode'].lower()

    if trading_mode == 'paper':
        logger.info("📄 Paper mode — futures simulated, no real futures client needed")
        return None

    api_key = config.config.get('binance_futures_api_key', '')
    api_secret = config.config.get('binance_futures_api_secret', '')

    if not api_key or not api_secret:
        raise ValueError(
            f"❌ CRITICAL: Futures API keys required for {trading_mode} mode. "
            f"Set BINANCE_FUTURES_API_KEY / BINANCE_FUTURES_API_SECRET in .env"
        )

    try:
        # Preferred: binance-futures-connector pip package
        from binance.um_futures import UMFutures
        base_url = 'https://testnet.binancefuture.com' if trading_mode == 'testnet' else None
        client = UMFutures(key=api_key, secret=api_secret, base_url=base_url)
        client.ping()
        logger.info("✅ Futures client (UMFutures) connected")
        _futures_client = client
        return client

    except ImportError:
        # Fallback: python-binance has futures methods too
        logger.warning("⚠️ binance-futures-connector not found — using python-binance futures endpoints")
        from binance.client import Client
        from binance.exceptions import BinanceAPIException as _BinanceAPIException
        client = Client(api_key, api_secret, testnet=(trading_mode == 'testnet'))
        client.ping()
        logger.info("✅ Futures client (python-binance fallback) connected")
        _futures_client = client
        return client

    except Exception as e:
        logger.error(f"❌ Failed to initialize Futures client: {e}")
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
