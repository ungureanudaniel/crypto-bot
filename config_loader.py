# config_loader.py - FIXED VERSION
import os
import json
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Load configuration from both .env and config.json"""
    
    def __init__(self):
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load config from both sources"""
        
        # 1. Load from config.json (trading settings)
        file_config = {}
        if os.path.exists("config.json"):
            with open("config.json", "r") as f:
                file_config = json.load(f)
        
        # 2. Load from environment variables (secrets)
        trading_mode = os.getenv('TRADING_MODE', 'paper').lower()
        
        env_config = {
            # Trading mode
            'trading_mode': trading_mode,
            
            # Telegram
            'telegram_token': os.getenv('TELEGRAM_TOKEN', ''),
            'telegram_chat_id': os.getenv('TELEGRAM_CHAT_ID', ''),
            
            # Binance API keys
            'binance_api_key': '',  # Will be set based on mode
            'binance_api_secret': '',  # Will be set based on mode
            'binance_testnet_api_key': os.getenv('BINANCE_TESTNET_API_KEY', ''),
            'binance_testnet_private_key': os.getenv('BINANCE_TESTNET_PRIVATE_KEY', ''),
        }
        
        # 3. Set the main API keys based on trading mode
        if trading_mode == 'live':
            env_config['binance_api_key'] = os.getenv('BINANCE_LIVE_API_KEY', '')
            env_config['binance_api_secret'] = os.getenv('BINANCE_LIVE_API_SECRET', '')
        elif trading_mode == 'testnet':
            # For testnet, use testnet keys as the main API keys
            env_config['binance_api_key'] = os.getenv('BINANCE_TESTNET_API_KEY', '')
            env_config['binance_api_secret'] = os.getenv('BINANCE_TESTNET_PRIVATE_KEY', '')
        # For paper mode, both remain empty
        
        # 4. Merge: env_config overrides file_config for overlapping keys
        merged_config = {**file_config, **env_config}
        
        # 5. Set derived values
        trading_mode = merged_config['trading_mode'].lower()
        merged_config['live_trading'] = trading_mode == 'live'
        merged_config['testnet'] = trading_mode == 'testnet'
        
        # 6. Set API URLs based on mode
        if merged_config['testnet']:
            merged_config['binance_api_url'] = merged_config.get('binance_testnet_api_url', 
                                                               'https://testnet.binance.vision')
        else:
            merged_config['binance_api_url'] = merged_config.get('binance_api_url', 
                                                               'https://api.binance.com')
        
        return merged_config
    
    def get(self, key: str, default=None):
        """Get config value"""
        return self.config.get(key, default)
    
    def __getitem__(self, key):
        """Get config value using dict syntax"""
        return self.config[key]
    
    def __contains__(self, key):
        """Check if key exists"""
        return key in self.config

# Global instance
config = Config()

if __name__ == "__main__":
    # Test loading
    CONFIG = config.config
    print("\n📋 Configuration Summary:")
    print(f"Trading mode: {CONFIG.get('trading_mode')}")
    print(f"Testnet flag: {CONFIG.get('testnet')}")
    print(f"binance_api_url: {CONFIG.get('binance_api_url')}")