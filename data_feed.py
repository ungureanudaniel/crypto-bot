import ccxt
import pandas as pd
import json
import logging

with open('config.json', 'r') as f:
    config = json.load(f)

logging.basicConfig(level=logging.INFO)

# Connect to Binance instead of Kraken
EXCHANGE = ccxt.binance({
    'apiKey': config.get('apiKey', ''),  # optional if public data only
    'secret': config.get('apiSecret', ''),  # optional if public data only
    'enableRateLimit': True,  # built-in rate limiter
    'options': {
        'defaultType': 'spot',  # or 'future', 'margin'
    },
})
LOOPBACK = 200

def fetch_ohlcv(coin, timeframe='1w'):
    logging.info(f"Fetching OHLCV data for {coin} from Binance...")
    try:
        # Ensure your coin symbols are in Binance format (e.g., 'BTC/USDC')
        ohlcv = EXCHANGE.fetch_ohlcv(coin, timeframe=timeframe, limit=LOOPBACK)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        logging.info(f"Successfully fetched {len(df)} OHLCV records for {coin} from Binance.")
        return df
    except Exception as e:
        logging.error(f"Error fetching OHLCV data for {coin} from Binance: {e}")
        return pd.DataFrame()

# Additional helper functions for Binance
def get_binance_symbols():
    """Fetch available trading pairs from Binance"""
    try:
        markets = EXCHANGE.load_markets()
        return list(markets.keys())
    except Exception as e:
        logging.error(f"Error fetching Binance symbols: {e}")
        return []

def format_symbol_for_binance(base_currency, quote_currency='USDC'):
    """Format symbol in Binance style (e.g., BTC/USDC)"""
    return f"{base_currency.upper()}/{quote_currency.upper()}"

# Example usage and common Binance timeframes
if __name__ == "__main__":
    # Test the connection
    common_timeframes = {
        '1m': '1 minute',
        '5m': '5 minutes', 
        '15m': '15 minutes',
        '1h': '1 hour',
        '4h': '4 hours',
        '1d': '1 day',
        '1w': '1 week'
    }
    
    # Test with a common pair
    test_symbol = 'BTC/USDC'
    df = fetch_ohlcv(test_symbol, '1h')
    if not df.empty:
        print(f"Latest data for {test_symbol}:")
        print(df.tail())