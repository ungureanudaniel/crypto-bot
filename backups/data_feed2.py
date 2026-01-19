# data_feed.py
import requests
import pandas as pd
import json
import logging
import time
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load config
try:
    with open('config.json', 'r') as f:
        config = json.load(f)
    logger.info("✅ Config loaded")
except Exception as e:
    logger.error(f"❌ Failed to load config.json: {e}")
    config = {}

# Binance public endpoints (NO API KEYS NEEDED)
BINANCE_API_URL = config.get("binance_api_url", "https://api1.binance.com")
BINANCE_TESTNET_URL = config.get("binance_testnet_api_url", "https://testnet.binance.vision")

# Use testnet if configured
USE_TESTNET = config.get('testnet', False)
BASE_URL = BINANCE_TESTNET_URL if USE_TESTNET else BINANCE_API_URL

def make_request(endpoint, params=None):
    """Make HTTP request to Binance API"""
    try:
        url = f"{BASE_URL}{endpoint}"
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Request failed: {e}")
        return None

def get_current_price(symbol):
    """Get current price for a symbol - NO API KEY """
    try:
        # Format symbol (BTC/USDC -> BTCUSDC)
        if '/' in symbol:
            binance_symbol = symbol.replace('/', '')
        else:
            binance_symbol = symbol
        
        endpoint = "/v3/ticker/price"
        params = {"symbol": binance_symbol}
        
        data = make_request(endpoint, params)
        if data and 'price' in data:
            price = float(data['price'])
            logger.debug(f"Current {symbol}: ${price:.2f}")
            return price
        else:
            logger.warning(f"No price data for {symbol}")
            return None
            
    except Exception as e:
        logger.error(f"Error getting price for {symbol}: {e}")
        return None

def get_klines(symbol, interval='1h', limit=500):
    """Get OHLCV data - NO API KEY """
    try:
        # Format symbol
        if '/' in symbol:
            binance_symbol = symbol.replace('/', '')
        else:
            binance_symbol = symbol
        
        endpoint = "/v3/klines"
        params = {
            "symbol": binance_symbol,
            "interval": interval,
            "limit": limit
        }
        
        data = make_request(endpoint, params)
        if data:
            # Convert to DataFrame
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            # Convert types
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            logger.info(f"✅ Got {len(df)} {interval} candles for {symbol}")
            return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        else:
            return pd.DataFrame()
            
    except Exception as e:
        logger.error(f"Error getting klines for {symbol}: {e}")
        return pd.DataFrame()

def fetch_binance_ohlcv(symbol, timeframe='1h', limit=1000, since=None):
    """Fetch OHLCV data with date range support"""
    try:
        if since:
            # Calculate limit based on timeframe and days
            days = (datetime.now() - since).days
            if timeframe == '1m':
                limit = min(days * 1440, limit)  # 1440 minutes per day
            elif timeframe == '5m':
                limit = min(days * 288, limit)   # 288 5-min candles per day
            elif timeframe == '15m':
                limit = min(days * 96, limit)    # 96 15-min candles per day
            elif timeframe == '1h':
                limit = min(days * 24, limit)    # 24 hours per day
            elif timeframe == '4h':
                limit = min(days * 6, limit)     # 6 4-hour candles per day
            elif timeframe == '1d':
                limit = min(days, limit)
        
        # Binance API max limit is 1000
        limit = min(limit, 1000)
        
        df = get_klines(symbol, timeframe, limit)
        
        if not df.empty and since:
            # Filter by start date
            df = df[df['timestamp'] >= since]
        
        return df
        
    except Exception as e:
        logger.error(f"Error in fetch_binance_ohlcv: {e}")
        return pd.DataFrame()

def fetch_historical_data(symbol, timeframe='1h', days=30):
    """Fetch historical data by making multiple requests"""
    try:
        all_data = []
        remaining_days = days
        
        while remaining_days > 0:
            # Calculate batch size (max 1000 candles)
            if timeframe == '1m':
                batch_days = min(remaining_days, 1)  # ~1440 candles
            elif timeframe == '5m':
                batch_days = min(remaining_days, 4)  # ~1152 candles
            elif timeframe == '15m':
                batch_days = min(remaining_days, 11) # ~1056 candles
            elif timeframe == '1h':
                batch_days = min(remaining_days, 42) # ~1008 candles
            elif timeframe == '4h':
                batch_days = min(remaining_days, 167) # ~1002 candles
            elif timeframe == '1d':
                batch_days = min(remaining_days, 1000)
            
            start_date = datetime.now() - timedelta(days=remaining_days)
            
            df = fetch_binance_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                limit=1000,
                since=start_date
            )
            
            if df.empty:
                break
            
            all_data.append(df)
            remaining_days -= batch_days
            
            # Rate limiting
            time.sleep(0.1)
        
        if all_data:
            combined_df = pd.concat(all_data).drop_duplicates().sort_values('timestamp').reset_index(drop=True)
            logger.info(f"📊 Total historical data: {len(combined_df)} records")
            return combined_df
            
    except Exception as e:
        logger.error(f"Error fetching historical data: {e}")
    
    return pd.DataFrame()

def fetch_ohlcv(symbol, timeframe='1h', days=90):
    """Main function for regime detection"""
    # For regime detection
    if timeframe in ['1h', '4h', '1d']:
        days_to_fetch = min(days, 365)  # Max 1 year
    else:
        days_to_fetch = min(days, 90)   # Max 90 days for shorter timeframes
    
    logger.info(f"Fetching {days_to_fetch} days of {timeframe} data...")
    return fetch_historical_data(symbol, timeframe, days_to_fetch)

def get_current_prices(symbols=None):
    """Get current prices for multiple symbols"""
    try:
        if symbols is None:
            symbols = config.get('coins', ['BTC/USDT', 'ETH/USDT'])
        
        prices = {}
        for symbol in symbols:
            price = get_current_price(symbol)
            if price:
                prices[symbol] = price
        
        return prices
        
    except Exception as e:
        logger.error(f"Error getting current prices: {e}")
        return {}

def test_connection():
    """Test connection to Binance"""
    try:
        # Simple ping to Binance API
        endpoint = "/v3/ping"
        data = make_request(endpoint)
        
        if data == {}:  # Binance returns empty object on success
            network = "Testnet" if USE_TESTNET else "Mainnet"
            logger.info(f"✅ Connected to Binance {network}")
            return True
        else:
            logger.error("❌ Unexpected response from Binance")
            return False
            
    except Exception as e:
        logger.error(f"❌ Failed to connect to Binance: {e}")
        return False

def get_exchange_info():
    """Get exchange info (available pairs)"""
    try:
        endpoint = "/v3/exchangeInfo"
        data = make_request(endpoint)
        
        if data and 'symbols' in data:
            symbols = [s['symbol'] for s in data['symbols'] if s['status'] == 'TRADING']
            logger.info(f"✅ Exchange info: {len(symbols)} trading pairs")
            return symbols
        return []
    except Exception as e:
        logger.error(f"Error getting exchange info: {e}")
        return []

def get_order_book(symbol, limit=10):
    """Get order book for a symbol"""
    try:
        if '/' in symbol:
            binance_symbol = symbol.replace('/', '')
        else:
            binance_symbol = symbol
        
        endpoint = "/v3/depth"
        params = {"symbol": binance_symbol, "limit": limit}
        
        data = make_request(endpoint, params)
        return data
    except Exception as e:
        logger.error(f"Error getting order book: {e}")
        return None

def get_24hr_stats(symbol):
    """Get 24-hour statistics"""
    try:
        if '/' in symbol:
            binance_symbol = symbol.replace('/', '')
        else:
            binance_symbol = symbol
        
        endpoint = "/v3/ticker/24hr"
        params = {"symbol": binance_symbol}
        
        data = make_request(endpoint, params)
        return data
    except Exception as e:
        logger.error(f"Error getting 24hr stats: {e}")
        return None

# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("📊 Binance Data Feed (Public API - No Keys)")
    print("=" * 60)
    
    if test_connection():
        network = "Testnet" if USE_TESTNET else "Mainnet"
        print(f"\n🌐 Connected to Binance {network}")
        
        # Test getting prices
        symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
        print(f"\n💰 Current Prices:")
        for symbol in symbols:
            price = get_current_price(symbol)
            if price:
                print(f"  {symbol}: ${price:,.2f}")
        
        # Test getting historical data
        print(f"\n📈 Testing historical data...")
        df = fetch_ohlcv('BTC/USDT', '15m', days=7)
        if not df.empty:
            print(f"  Got {len(df)} candles")
            print(f"  Latest: ${df['close'].iloc[-1]:,.2f}")
            print(f"  Date range: {df['timestamp'].min().strftime('%Y-%m-%d')} to {df['timestamp'].max().strftime('%Y-%m-%d')}")
        
        # Show exchange info
        print(f"\n📋 Exchange Info:")
        symbols = get_exchange_info()
        usdt_pairs = [s for s in symbols if 'USDT' in s]
        print(f"  Total pairs: {len(symbols)}")
        print(f"  USDT pairs: {len(usdt_pairs)}")
        print(f"  Sample: {usdt_pairs[:5]}")
        
        print(f"\n✅ Data feed is working! Ready for paper trading.")
    else:
        print(f"\n❌ Failed to connect. Check your internet connection.")