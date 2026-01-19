# data_feed.py - Simplified Binance data feed for regime detection
import pandas as pd
import logging
import time
from datetime import datetime, timedelta
from typing import Optional, Dict
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Load config
try:
    with open('config.json', 'r') as f:
        config = json.load(f)
    logger.info("✅ Config loaded")
except Exception as e:
    logger.error(f"❌ Failed to load config.json: {e}")
    config = {}

# Initialize Binance client
def init_binance_client():
    """Initialize and return Binance client instance"""
    try:
        from binance.client import Client
        
        # Get API credentials from config
        api_key = config.get('binance_api_key')
        api_secret = config.get('binance_api_secret')
        
        if not api_key or not api_secret:
            logger.warning("⚠️ Binance API credentials not found in config.json")
            logger.warning("⚠️ Using public endpoints only (rate limited)")
        
        # Initialize client (public endpoints work without API keys)
        client = Client(api_key=api_key, api_secret=api_secret)
        logger.info("✅ Binance client initialized")
        return client
        
    except ImportError:
        logger.error("❌ binance package not installed. Install with: pip install python-binance")
        raise
    except Exception as e:
        logger.error(f"❌ Failed to initialize Binance client: {e}")
        raise

# Global client instance
client = init_binance_client()

# ------------------------------------------------------------
# Data fetching functions for regime detection
# ------------------------------------------------------------

def format_symbol(symbol: str) -> str:
    """Format trading symbol for Binance (BTC/USDT -> BTCUSDT)"""
    if '/' in symbol:
        return symbol.replace('/', '')
    return symbol

def fetch_ohlcv(
    symbol: str,
    interval: str = '1h',
    limit: int = 1000,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None
) -> pd.DataFrame:
    """
    Fetch OHLCV data from Binance for regime detection.
    
    Args:
        symbol: Trading pair (e.g., 'BTC/USDT')
        interval: Timeframe ('1m', '5m', '15m', '1h', '4h', '1d')
        limit: Maximum number of candles (Binance max: 1000)
        start_time: Start datetime for data
        end_time: End datetime for data
    
    Returns:
        DataFrame with columns: ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    """
    try:
        binance_symbol = format_symbol(symbol)
        
        # Convert datetime to milliseconds for Binance API
        start_str = None
        end_str = None
        
        if start_time:
            start_str = int(start_time.timestamp() * 1000)
        if end_time:
            end_str = int(end_time.timestamp() * 1000)
        
        # Fetch klines from Binance
        klines = client.get_klines(
            symbol=binance_symbol,
            interval=interval,
            limit=limit,
            startTime=start_str,
            endTime=end_str
        )
        
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
        
        logger.info(f"✅ Fetched {len(df)} {interval} candles for {symbol}")
        
        # Return only needed columns
        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
        
    except Exception as e:
        logger.error(f"❌ Error fetching OHLCV for {symbol}: {e}")
        return pd.DataFrame()

def get_current_price(symbol: str) -> Optional[float]:
    """Get current price for a symbol"""
    try:
        binance_symbol = format_symbol(symbol)
        ticker = client.get_symbol_ticker(symbol=binance_symbol)
        price = float(ticker['price'])
        logger.debug(f"Current {symbol}: ${price:.2f}")
        return price
    except Exception as e:
        logger.error(f"❌ Error getting current price for {symbol}: {e}")
        return None

def fetch_historical_data(
    symbol: str,
    interval: str = '1h',
    days: int = 30
) -> pd.DataFrame:
    """
    Fetch historical data by making multiple requests if needed.
    Optimized for regime detection timeframes.
    
    Args:
        symbol: Trading pair
        interval: Timeframe for regime detection ('1h', '4h', '1d')
        days: Number of days of data to fetch
    
    Returns:
        DataFrame with historical OHLCV data
    """
    try:
        # Limit days based on interval
        if interval in ['1h', '4h', '1d']:
            max_days = 365  # 1 year max for regime detection
        else:
            max_days = 90  # 90 days for shorter timeframes
        
        days = min(days, max_days)
        
        # Calculate candles needed
        if interval == '1m':
            candles_per_day = 1440
        elif interval == '5m':
            candles_per_day = 288
        elif interval == '15m':
            candles_per_day = 96
        elif interval == '1h':
            candles_per_day = 24
        elif interval == '4h':
            candles_per_day = 6
        elif interval == '1d':
            candles_per_day = 1
        else:
            candles_per_day = 24  # Default to hourly
        
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
            candles_fetched = 0
            batch_size = 1000  # Binance maximum
            
            # Calculate batch intervals
            batch_days = batch_size / candles_per_day
            remaining_days = days
            
            while remaining_days > 0:
                current_batch_days = min(batch_days, remaining_days)
                
                end_date = datetime.now() - timedelta(days=(days - remaining_days))
                start_date = end_date - timedelta(days=current_batch_days)
                
                df = fetch_ohlcv(
                    symbol=symbol,
                    interval=interval,
                    limit=batch_size,
                    start_time=start_date,
                    end_time=end_date
                )
                
                if df.empty:
                    break
                
                all_data.append(df)
                candles_fetched += len(df)
                remaining_days -= current_batch_days
                
                # Rate limiting
                time.sleep(0.1)
            
            if all_data:
                combined_df = pd.concat(all_data, ignore_index=True)
                combined_df = combined_df.drop_duplicates(subset=['timestamp'])
                combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
                
                logger.info(f"📊 Total historical data: {len(combined_df)} candles")
                return combined_df
            else:
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
    
    if not df.empty:
        # Ensure proper sorting
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Add returns for regime detection
        df['returns'] = df['close'].pct_change()
        
        logger.info(f"✅ Regime data ready: {len(df)} candles")
    else:
        logger.warning(f"⚠️ No data returned for {symbol}")
    
    return df

def test_connection() -> bool:
    """Test connection to Binance API"""
    try:
        # Simple ping test
        client.ping()
        logger.info("✅ Connected to Binance API")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to connect to Binance: {e}")
        return False

# ------------------------------------------------------------
# Data feed class for trading engine
# ------------------------------------------------------------

class BinanceDataFeed:
    """Simple data feed for regime detection and trading engine"""
    
    def __init__(self):
        self.client = client
        logger.info("📊 Binance Data Feed initialized")
    
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
        """Get current prices for multiple symbols"""
        prices = {}
        for symbol in symbols:
            price = self.get_price(symbol)
            if price is not None:
                prices[symbol] = price
        return prices

# Initialize global data feed instance
data_feed = BinanceDataFeed()

# ------------------------------------------------------------
# Example usage
# ------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("📊 Binance Data Feed for Regime Detection")
    print("=" * 60)
    
    if test_connection():
        # Test current price
        symbol = 'BTC/USDT'
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
        
        # Test multiple timeframes
        timeframes = ['1h', '4h', '1d']
        for tf in timeframes:
            df_tf = fetch_ohlcv(symbol, tf, limit=100)
            if not df_tf.empty:
                print(f"\n📊 {tf} timeframe: {len(df_tf)} candles")
                print(f"   Latest: ${df_tf['close'].iloc[-1]:,.2f}")
        
        print("\n✅ Data feed is working correctly!")
    else:
        print("\n❌ Failed to connect. Check your internet connection and API credentials.")