# data_feed.py - Simplified Binance data feed for regime detection
import pandas as pd
import os
import sys
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Optional, Dict
# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# ------------------------------------------------------------
# Binance client initialization
# ------------------------------------------------------------

def init_binance_client():
    """Initialize Binance client using config from config_loader"""
    try:
        from binance.client import Client
        trading_mode = CONFIG.get('trading_mode', 'paper').lower()
        logger.info(f"Initializing for {trading_mode} mode")

        # Get API keys (config_loader handles which ones to use)
        api_key = CONFIG.get('binance_api_key', '')
        api_secret = CONFIG.get('binance_api_secret', '')

        # For paper trading or missing keys, create mock client
        if trading_mode == 'paper' or not api_key or not api_secret:
            logger.info("📄 Using PAPER Trading mode - Mock Binance client")

        # ✅ FIXED: Use testnet=True parameter for testnet
        if trading_mode == 'testnet':
            logger.info("🔐 Using Binance Testnet mode")
            client = Client(api_key=api_key, api_secret=api_secret, testnet=True)
        else:  # live trading
            logger.info("🔐 Using Binance Live mode")
            client = Client(api_key=api_key, api_secret=api_secret)
        
        logger.info(f"✅ Binance client initialized for {trading_mode} mode")
        logger.info(f"🌐 API URL: {client.API_URL}")
        
        return client
        
    except ImportError:
        logger.error("❌ binance package not installed. Run: pip install python-binance")
        raise
    except Exception as e:
        logger.error(f"❌ Failed to initialize Binance client: {e}")
        # Return paper trading client as fallback
        logger.info("📄 Falling back to paper trading due to error")
        CONFIG['trading_mode'] = 'paper'
        return init_binance_client()

# Global client instance
client = init_binance_client()

# ------------------------------------------------------------
# Data fetching functions for regime detection
# ------------------------------------------------------------

def format_symbol(symbol: str) -> str:
    """Format trading symbol for Binance (BTC/USDT -> BTCUSDT)"""
    # Remove any spaces and convert format
    symbol = symbol.strip().upper()
    
    # Handle different formats
    if '/' in symbol:
        # Format: BTC/USDT -> BTCUSDT
        parts = symbol.split('/')
        if len(parts) == 2:
            return f"{parts[0]}{parts[1]}"
    
    # Already in correct format or unknown
    return symbol.replace('/', '')

def fetch_ohlcv(
    symbol: str,
    interval: str = '1h',
    limit: int = 1000,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    retry_count: int = 3
) -> pd.DataFrame:
    """
    Fetch OHLCV data from Binance for regime detection with retry logic.
    
    Args:
        symbol: Trading pair (e.g., 'BTC/USDT')
        interval: Timeframe ('1m', '5m', '15m', '1h', '4h', '1d')
        limit: Maximum number of candles (Binance max: 1000)
        start_time: Start datetime for data
        end_time: End datetime for data
        retry_count: Number of retry attempts on failure
    
    Returns:
        DataFrame with columns: ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    """
    if client is None:
        logger.error("❌ Binance client is not initialized. Cannot fetch OHLCV data.")
        return pd.DataFrame()
    
    for attempt in range(retry_count):
        try:
            binance_symbol = format_symbol(symbol)
            
            # Rate limiting
            if attempt > 0:
                delay = CONFIG['rate_limit_delay'] * (2 ** attempt)  # Exponential backoff
                logger.debug(f"Retry attempt {attempt + 1}, waiting {delay:.2f}s...")
                time.sleep(delay)
            
            # Convert datetime to milliseconds for Binance API
            params = {
                'symbol': binance_symbol,
                'interval': interval,
                'limit': min(limit, 1000)  # Binance max
            }
            
            if start_time:
                params['startTime'] = int(start_time.timestamp() * 1000)
            if end_time:
                params['endTime'] = int(end_time.timestamp() * 1000)
            
            # Fetch klines from Binance
            klines = client.get_klines(**params)
            
            if not klines:
                logger.warning(f"No data returned for {symbol}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            # Convert types
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            logger.debug(f"✅ Fetched {len(df)} {interval} candles for {symbol}")
            
            # Return only needed columns
            return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
            
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed for {symbol}: {str(e)[:100]}")
            if attempt == retry_count - 1:  # Last attempt
                logger.error(f"❌ Failed to fetch OHLCV for {symbol} after {retry_count} attempts: {e}")
    
    return pd.DataFrame()

def get_current_price(symbol: str, retry_count: int = 2) -> Optional[float]:
    """Get current price for a symbol with retry logic"""
    if client is None:
        logger.error("❌ Binance client is not initialized. Cannot fetch current price.")
        return None
    
    for attempt in range(retry_count):
        try:
            binance_symbol = format_symbol(symbol)
            
            # Rate limiting for retries
            if attempt > 0:
                time.sleep(0.5)
            
            ticker = client.get_symbol_ticker(symbol=binance_symbol)
            price = float(ticker['price'])
            logger.debug(f"Current {symbol}: ${price:.2f}")
            return price
            
        except Exception as e:
            logger.warning(f"Price fetch attempt {attempt + 1} failed: {str(e)[:100]}")
            if attempt == retry_count - 1:
                logger.error(f"❌ Failed to get price for {symbol}: {e}")
    
    return None

def fetch_historical_data(
    symbol: str,
    interval: str = '1h',
    days: int = 30,
    max_requests: int = 10
) -> pd.DataFrame:
    """
    Fetch historical data by making multiple requests if needed.
    Optimized for regime detection timeframes.
    
    Args:
        symbol: Trading pair
        interval: Timeframe for regime detection ('1h', '4h', '1d')
        days: Number of days of data to fetch
        max_requests: Maximum number of API requests to make
    
    Returns:
        DataFrame with historical OHLCV data
    """
    try:
        # Limit days based on interval to avoid too many requests
        interval_limits = {
            '1m': 7,    # 7 days max for 1m
            '5m': 30,   # 30 days max for 5m
            '15m': 90,  # 90 days max for 15m
            '1h': 365,  # 1 year max for 1h
            '4h': 730,  # 2 years max for 4h
            '1d': 1825, # 5 years max for 1d
        }
        
        max_allowed_days = interval_limits.get(interval, 90)
        days = min(days, max_allowed_days)
        
        # Calculate candles needed
        candles_per_day = {
            '1m': 1440,
            '5m': 288,
            '15m': 96,
            '1h': 24,
            '4h': 6,
            '1d': 1,
        }.get(interval, 24)
        
        total_candles_needed = days * candles_per_day
        
        # Binance API max is 1000 candles per request
        if total_candles_needed <= 1000:
            # Single request is enough
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days)
            
            df = fetch_ohlcv(
                symbol=symbol,
                interval=interval,
                limit=total_candles_needed,
                start_time=start_time
            )
            
            logger.info(f"📊 Fetched {len(df)} candles for {symbol} ({days} days of {interval})")
            return df
            
        else:
            # Need multiple requests
            logger.info(f"Fetching {days} days of {interval} data ({total_candles_needed} candles)...")
            
            all_data = []
            candles_per_request = 1000  # Binance maximum
            requests_made = 0
            
            # Calculate time per request
            days_per_request = candles_per_request / candles_per_day
            
            # Fetch backwards from now
            current_end = datetime.now()
            
            while days > 0 and requests_made < max_requests:
                current_days = int(min(days_per_request, days))
                current_start = current_end - timedelta(days=current_days)
                
                df = fetch_ohlcv(
                    symbol=symbol,
                    interval=interval,
                    limit=candles_per_request,
                    start_time=current_start,
                    end_time=current_end
                )
                
                if not df.empty:
                    all_data.append(df)
                    candles_fetched = len(df)
                    logger.debug(f"  Batch {requests_made + 1}: {candles_fetched} candles")
                    
                    # Update for next batch
                    current_end = current_start
                    days -= current_days
                    requests_made += 1
                    
                    # Rate limiting between requests
                    time.sleep(CONFIG['rate_limit_delay'])
                else:
                    logger.warning(f"No data in batch {requests_made + 1}, stopping")
                    break
            
            if all_data:
                # Combine and clean data
                combined_df = pd.concat(all_data, ignore_index=True)
                combined_df = combined_df.drop_duplicates(subset=['timestamp'])
                combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
                
                logger.info(f"📊 Total historical data: {len(combined_df)} candles from {requests_made} requests")
                return combined_df
            else:
                logger.warning(f"No data fetched for {symbol}")
                return pd.DataFrame()
                
    except Exception as e:
        logger.error(f"❌ Error fetching historical data for {symbol}: {e}")
        return pd.DataFrame()

def get_regime_data(
    symbol: str,
    timeframe: str = '1h',
    days: int = 90
) -> pd.DataFrame:
    """
    Main function for regime detection.
    Fetches and prepares data for regime analysis.
    
    Args:
        symbol: Trading pair
        timeframe: Regime timeframe ('1h', '4h', '1d')
        days: Days of data for regime detection
    
    Returns:
        DataFrame ready for regime analysis
    """
    logger.info(f"🔄 Fetching regime data for {symbol} ({timeframe}, {days} days)...")
    
    df = fetch_historical_data(symbol, timeframe, days)
    
    if not df.empty and len(df) > 10:
        # Ensure proper sorting
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Add returns for regime detection
        df['returns'] = df['close'].pct_change()
        
        # Add simple moving averages for regime features
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        
        logger.info(f"✅ Regime data ready: {len(df)} candles")
        logger.debug(f"  Latest: ${df['close'].iloc[-1]:.2f}, Return: {df['returns'].iloc[-1]:.4%}")
    else:
        logger.warning(f"⚠️ Insufficient data for {symbol}: {len(df) if not df.empty else 0} candles")
    
    return df

def test_connection() -> bool:
    """Test connection to Binance API"""
    if client is None:
        logger.info("📄 Using paper trading mode - no API connection needed")
        return True
    
    try:
        # Simple ping test
        client.ping()
        logger.info("✅ Connected to Binance API")
        return True
    except Exception as e:
        logger.warning(f"⚠️ Could not ping Binance: {e}")
        
        # Try a simple data fetch as alternative test
        try:
            price = get_current_price('BTCUSDT')
            if price:
                logger.info(f"✅ Connection test passed via price fetch: ${price:.2f}")
                return True
        except Exception as e2:
            logger.error(f"❌ Failed to connect to Binance: {e2}")
        
        return False

# ------------------------------------------------------------
# Data feed class for trading engine
# ------------------------------------------------------------

class BinanceDataFeed:
    """Simple data feed for regime detection and trading engine"""
    def __init__(self):
        self.client = client
        self.rate_limit_delay = CONFIG['rate_limit_delay']
        logger.info("📊 Binance Data Feed initialized")
        
        # Test connection on init
        if not test_connection():
            logger.warning("⚠️ Data feed initialized with connection issues")
    
    def get_price(self, symbol: str) -> Optional[float]:
        """Get current price"""
        return get_current_price(symbol)
    
    def get_ohlcv(self, symbol: str, interval: str, limit: int = 1000) -> pd.DataFrame:
        """Get OHLCV data"""
        return fetch_ohlcv(symbol, interval, limit)
    
    def get_regime_data(self, symbol: str, timeframe: str = '1h', days: int = 90) -> pd.DataFrame:
        """Get data for regime detection"""
        return get_regime_data(symbol, timeframe, days)
    
    def get_historical_data(self, symbol: str, interval: str, days: int) -> pd.DataFrame:
        """Get historical data"""
        return fetch_historical_data(symbol, interval, days)
    
    def get_multiple_prices(self, symbols: list) -> Dict[str, float]:
        """Get current prices for multiple symbols with rate limiting"""
        prices = {}
        for i, symbol in enumerate(symbols):
            price = self.get_price(symbol)
            if price is not None:
                prices[symbol] = price
            
            # Rate limiting between requests
            if i < len(symbols) - 1:
                time.sleep(self.rate_limit_delay)
        
        logger.debug(f"Fetched prices for {len(prices)}/{len(symbols)} symbols")
        return prices

# Initialize global data feed instance
data_feed = BinanceDataFeed()

# ------------------------------------------------------------
# Example usage
# ------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("📊 Binance Data Feed with Environment Variables")
    print("=" * 60)
    
    # Show config status
    print(f"\n🔧 Configuration:")
    print(f"   Config: {CONFIG}")
    print(f"   Trading Mode: {CONFIG.get('trading_mode')}")
    print(f"   Using Testnet: {CONFIG.get('testnet')}")
    print(f"   API Key Configured: {'Yes' if CONFIG.get('binance_api_key') else 'No'}")
    print(f"   Rate Limit Delay: {CONFIG.get('rate_limit_delay')}s")
    
    if test_connection():
        # Test current price
        symbol = 'BTC/USDC'
        price = get_current_price(symbol)
        if price:
            print(f"\n💰 Current {symbol}: ${price:,.2f}")
        
        # Test regime data
        print(f"\n📈 Fetching regime data for {symbol}...")
        df = get_regime_data(symbol, timeframe='4h', days=30)
        
        if not df.empty:
            print(f"✅ Data fetched: {len(df)} candles")
            print(f"   Date range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
            print(f"   Latest price: ${df['close'].iloc[-1]:,.2f}")
            print(f"   Latest return: {df['returns'].iloc[-1]:.4%}")
        
        # Test data feed class
        print(f"\n🧪 Testing data feed class...")
        prices = data_feed.get_multiple_prices(['BTC/USDC', 'ETH/USDC', 'SOL/USDC'])
        for sym, pr in prices.items():
            print(f"   {sym}: ${pr:,.2f}")
        
        print("\n✅ Data feed is working correctly with environment variables!")
    else:
        print("\n❌ Failed to connect. Check your internet connection.")
        print("   If using API keys, verify they are correct in .env file")
        print("   If not using keys, you may be rate limited. Try again later.")