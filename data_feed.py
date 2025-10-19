import ccxt
import pandas as pd
import json
import logging
import time
from datetime import datetime, timedelta

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
# Binance timeframe mapping
TIMEFRAME_MAP = {
    '1m': '1m',
    '5m': '5m', 
    '15m': '15m',
    '1h': '1h',
    '4h': '4h',
    '1d': '1d',
    '1w': '1w'
}

def fetch_binance_ohlcv(symbol, timeframe='1h', limit=1000, since=None):
    """
    Reliable OHLCV data from Binance - FETCHES MAXIMUM DATA
    """
    try:
        # Format symbol for Binance (BTC/USDC -> BTCUSDC)
        if '/' in symbol:
            binance_symbol = symbol.replace('/', '')
        else:
            binance_symbol = symbol

        logging.info(f"Fetching {timeframe} data for {binance_symbol} from Binance...")
        
        # Fetch OHLCV data
        ohlcv = EXCHANGE.fetch_ohlcv(binance_symbol, timeframe, since=since, limit=limit)
        
        if ohlcv:
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            logging.info(f"✅ Binance: {len(df)} {timeframe} records for {binance_symbol}")
            logging.info(f"📅 From {df['timestamp'].min().strftime('%Y-%m-%d %H:%M')} to {df['timestamp'].max().strftime('%Y-%m-%d %H:%M')}")
            
            return df
        else:
            logging.warning(f"No data returned from Binance for {binance_symbol}")
            
    except ccxt.BaseError as e:
        logging.error(f"Binance API error for {symbol}: {e}")
    except Exception as e:
        logging.error(f"Unexpected error for {symbol}: {e}")
    
    return pd.DataFrame()

def fetch_historical_data(symbol, timeframe='1h', days=365):
    """
    Fetch extensive historical data for regime detection
    """
    try:
        # Calculate start date
        since = EXCHANGE.parse8601((datetime.now() - timedelta(days=days)).isoformat())
        
        all_data = []
        fetch_count = 0
        max_fetches = 10  # Prevent infinite loops
        
        while fetch_count < max_fetches:
            data = fetch_binance_ohlcv(symbol, timeframe, since=since, limit=1000)
            if data.empty:
                break
                
            all_data.append(data)
            
            # Update since to get next batch (move forward in time)
            last_timestamp = data['timestamp'].iloc[-1]
            since = EXCHANGE.parse8601(last_timestamp.isoformat())
            
            # Small delay to respect rate limits
            time.sleep(0.1)
            fetch_count += 1
            
            # Stop if we have sufficient data or reach present
            if len(data) < 1000:
                break
        
        if all_data:
            combined_df = pd.concat(all_data).drop_duplicates().sort_values('timestamp').reset_index(drop=True)
            logging.info(f"📊 Total historical data fetched: {len(combined_df)} records")
            return combined_df
        else:
            logging.warning("No historical data fetched")
            return pd.DataFrame()
            
    except Exception as e:
        logging.error(f"Error fetching historical data for {symbol}: {e}")
        return pd.DataFrame()

def get_current_price(symbol):
    """
    Get current price for a symbol
    """
    try:
        if '/' in symbol:
            binance_symbol = symbol.replace('/', '')
        else:
            binance_symbol = symbol
            
        ticker = EXCHANGE.fetch_ticker(binance_symbol)
        return ticker['last']
    except Exception as e:
        logging.error(f"Error getting current price for {symbol}: {e}")
        return None

# MAIN FUNCTION FOR REGIME DETECTION - FETCHES MAX DATA
def fetch_ohlcv(symbol, timeframe='1h', days=90):
    """
    MAIN FUNCTION USED BY REGIME DETECTOR
    Fetches maximum available historical data
    """
    # For regime detection, we need LOTS of historical data
    if timeframe in ['1h', '4h', '1d']:
        days_to_fetch = 365  # Get 1 year of data for better regime detection
    else:
        days_to_fetch = 90   # 90 days for shorter timeframes
    
    logging.info(f"Fetching {days_to_fetch} days of {timeframe} data for regime detection...")
    
    df = fetch_historical_data(symbol, timeframe, days=days_to_fetch)
    
    if df.empty:
        # Fallback: try with less data
        logging.warning("Historical fetch failed, trying basic fetch...")
        df = fetch_binance_ohlcv(symbol, timeframe, limit=1000)
    
    return df
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

    
    portfolio = load_portfolio()
    
    current_prices = {}
    
    # Get prices for coins with open positions
    for symbol in portfolio.get('positions', {}).keys():
        try:
            df = fetch_binance_ohlcv(symbol, '1m', limit=1)
            if not df.empty:
                current_prices[symbol] = df.iloc[-1]['close']
        except Exception as e:
            logging.error(f"Error fetching price for {symbol}: {e}")
    
    # Get prices for configured coins
    for symbol in config.get('coins', []):
        if symbol not in current_prices:
            try:
                df = fetch_binance_ohlcv(symbol, '1m', limit=1)
                if not df.empty:
                    current_prices[symbol] = df.iloc[-1]['close']
            except Exception as e:
                logging.error(f"Error fetching price for {symbol}: {e}")
    
    return current_prices
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
                print(df[['timestamp', 'close']].tail(60))
            else:
                print(f"❌ Failed to fetch data for {symbol}")