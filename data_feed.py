import ccxt
import pandas as pd
import json
import logging

with open('config.json', 'r') as f:
    config = json.load(f)

logging.basicConfig(level=logging.INFO)

# Connect to Binance
EXCHANGE = ccxt.binance({
    'apiKey': config.get('BINANCE_API_KEY', ''),  # Use consistent naming
    'secret': config.get('BINANCE_SECRET_KEY', ''),
    'rateLimit': 1200,
    'options': {
        'defaultType': 'spot',
    },
})
LOOPBACK = 1000

def fetch_ohlcv(coin, timeframe='15m'):  # Changed default to 15min for stability
    logging.info(f"Fetching OHLCV data for {coin} from Binance (timeframe: {timeframe})...")
    try:
        # Binance symbol format validation
        if '/' not in coin:
            logging.error(f"Invalid symbol format: {coin}. Use format: BTC/USDC")
            return pd.DataFrame()
            
        ohlcv = EXCHANGE.fetch_ohlcv(coin, timeframe=timeframe, limit=LOOPBACK)
        
        if not ohlcv:
            logging.warning(f"No data returned for {coin}")
            return pd.DataFrame()
            
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        logging.info(f"✅ Successfully fetched {len(df)} {timeframe} OHLCV records for {coin}")
        return df
        
    except ccxt.BadSymbol as e:
        logging.error(f"Invalid symbol {coin}: {e}")
    except ccxt.NetworkError as e:
        logging.error(f"Network error fetching {coin}: {e}")
    except ccxt.ExchangeError as e:
        logging.error(f"Exchange error fetching {coin}: {e}")
    except Exception as e:
        logging.error(f"Unexpected error fetching {coin}: {e}")
        
    return pd.DataFrame()

# Additional helper functions for Binance
def get_binance_symbols():
    """Fetch available trading pairs from Binance"""
    try:
        markets = EXCHANGE.load_markets()
        symbols = [symbol for symbol in markets.keys() if 'USDC' in symbol]  # Filter major pairs
        logging.info(f"Found {len(symbols)} trading pairs")
        return symbols
    except Exception as e:
        logging.error(f"Error fetching Binance symbols: {e}")
        return []

def format_symbol_for_binance(base_currency, quote_currency='USDC'):  # Changed default to USDC
    """Format symbol in Binance style (e.g., BTC/USDC)"""
    return f"{base_currency.upper()}/{quote_currency.upper()}"

def test_connection():
    """Test Binance connection"""
    try:
        EXCHANGE.fetch_status()
        logging.info("✅ Successfully connected to Binance")
        return True
    except Exception as e:
        logging.error(f"❌ Failed to connect to Binance: {e}")
        return False

def get_current_prices():
    """Get current prices for all coins in portfolio and config"""
    from trade_engine import load_portfolio
    from config import load_config
    
    portfolio = load_portfolio()
    config = load_config()
    
    current_prices = {}
    
    # Get prices for coins with open positions
    for symbol in portfolio.get('positions', {}).keys():
        try:
            df = fetch_ohlcv(symbol, '1m', limit=1)
            if not df.empty:
                current_prices[symbol] = df.iloc[-1]['close']
        except Exception as e:
            logging.error(f"Error fetching price for {symbol}: {e}")
    
    # Get prices for configured coins
    for symbol in config.get('coins', []):
        if symbol not in current_prices:
            try:
                df = fetch_ohlcv(symbol, '1m', limit=1)
                if not df.empty:
                    current_prices[symbol] = df.iloc[-1]['close']
            except Exception as e:
                logging.error(f"Error fetching price for {symbol}: {e}")
    
    return current_prices

# Example usage and common Binance timeframes
if __name__ == "__main__":
    # Test the connection first
    if test_connection():
        # Test with common pairs
        test_symbols = ['BTC/USDC', 'ETH/USDC', 'TRX/USDC']
        
        for symbol in test_symbols:
            print(f"\nTesting {symbol}:")
            df = fetch_ohlcv(symbol, '15m')
            if not df.empty:
                print(f"Latest 3 records for {symbol}:")
                print(df[['timestamp', 'close']].tail(3))
            else:
                print(f"❌ Failed to fetch data for {symbol}")