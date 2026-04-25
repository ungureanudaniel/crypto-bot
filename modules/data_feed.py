import sys
import os
# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import logging
import time
from config_loader import config
from datetime import datetime, timedelta
from typing import Optional, Dict
from config_loader import get_binance_client 

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# CONFIG LOADING
# -------------------------------------------------------------------
CONFIG = config.config
if CONFIG.get('trading_mode', 'paper') in ['live', 'testnet']:
    client = get_binance_client()
else:
    client = None
    logger.info("Paper mode - using public Binance API for price data")

# ------------------------------------------------------------
# Data fetching functions
# ------------------------------------------------------------
def format_symbol(symbol: str) -> str:
    """Format trading symbol for Binance (BTC/USDT -> BTCUSDT)"""
    symbol = symbol.strip().upper()
    
    if '/' in symbol:
        parts = symbol.split('/')
        if len(parts) == 2:
            return f"{parts[0]}{parts[1]}"
    
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
    Fetch OHLCV data from Binance public API (no auth needed).
    """
    import requests
    
    for attempt in range(retry_count):
        try:
            binance_symbol = format_symbol(symbol)
            
            # Rate limiting for retries
            if attempt > 0:
                delay = 0.5 * (2 ** attempt)
                logger.debug(f"Retry attempt {attempt + 1}, waiting {delay:.2f}s...")
                time.sleep(delay)
            
            # Build URL for public endpoint
            url = "https://api.binance.com/api/v3/klines"
            params = {
                'symbol': binance_symbol,
                'interval': interval,
                'limit': min(limit, 1000)
            }
            
            if start_time:
                params['startTime'] = int(start_time.timestamp() * 1000)
            if end_time:
                params['endTime'] = int(end_time.timestamp() * 1000)
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code != 200:
                logger.warning(f"API returned {response.status_code}: {response.text}")
                if attempt == retry_count - 1:
                    return pd.DataFrame()
                continue
            
            klines = response.json()
            
            if not klines:
                logger.warning(f"No data returned for {symbol}")
                return pd.DataFrame()
            
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            logger.debug(f"Fetched {len(df)} {interval} candles for {symbol}")
            
            return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
            
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed for {symbol}: {str(e)[:100]}")
            if attempt == retry_count - 1:
                logger.error(f"❌ Failed to fetch OHLCV for {symbol} after {retry_count} attempts: {e}")
    
    return pd.DataFrame()

def generate_mock_ohlcv(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    """Generate mock OHLCV data for paper trading"""
    import numpy as np
    
    # Create timestamps
    end_time = datetime.now()
    if interval.endswith('m'):
        minutes = int(interval[:-1])
        delta = timedelta(minutes=minutes)
    elif interval.endswith('h'):
        hours = int(interval[:-1])
        delta = timedelta(hours=hours)
    elif interval.endswith('d'):
        days = int(interval[:-1])
        delta = timedelta(days=days)
    else:
        delta = timedelta(hours=1)
    
    timestamps = [end_time - i * delta for i in range(limit)]
    timestamps.reverse()
    
    # Generate mock price data based on symbol
    base_price = {
        'BTC/USDT': 65000,
        'ETH/USDT': 3500,
        'SOL/USDT': 150,
        'BNB/USDT': 600,
        'XRP/USDT': 0.5,
    }.get(symbol, 100)
    
    # Create random walk
    returns = np.random.randn(limit) * 0.02  # 2% volatility
    prices = base_price * np.exp(np.cumsum(returns))
    
    # Create OHLC data
    df = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices * (1 + np.random.randn(limit) * 0.001),
        'high': prices * (1 + np.abs(np.random.randn(limit) * 0.01)),
        'low': prices * (1 - np.abs(np.random.randn(limit) * 0.01)),
        'close': prices,
        'volume': np.random.randint(1000, 10000, limit)
    })
    
    logger.info(f"Generated {limit} mock candles for {symbol}")
    return df

def get_current_price(symbol: str, retry_count: int = 2) -> Optional[float]:
    """Get current price for a symbol using public API (no auth needed)"""
    import requests
    
    for attempt in range(retry_count):
        try:
            binance_symbol = format_symbol(symbol)
            
            # Use public endpoint - no API keys needed!
            url = f"https://api.binance.com/api/v3/ticker/price?symbol={binance_symbol}"
            
            if attempt > 0:
                time.sleep(0.5)
            
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                price = float(response.json()['price'])
                logger.debug(f"Current {symbol}: ${price:.2f}")
                return price
            else:
                logger.warning(f"API returned {response.status_code}: {response.text}")
                
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
    """
    try:
        interval_limits = {
            '1m': 7, '5m': 30, '15m': 90,
            '1h': 365, '4h': 730, '1d': 1825,
        }
        
        max_allowed_days = interval_limits.get(interval, 90)
        days = min(days, max_allowed_days)
        
        candles_per_day = {
            '1m': 1440, '5m': 288, '15m': 96,
            '1h': 24, '4h': 6, '1d': 1,
        }.get(interval, 24)
        
        total_candles_needed = days * candles_per_day
        
        if total_candles_needed <= 1000:
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days)
            
            df = fetch_ohlcv(
                symbol=symbol,
                interval=interval,
                limit=total_candles_needed,
                start_time=start_time
            )
            
            logger.info(f"Fetched {len(df)} candles for {symbol} ({days} days of {interval})")
            return df
            
        else:
            logger.info(f"Fetching {days} days of {interval} data ({total_candles_needed} candles)...")
            
            all_data = []
            candles_per_request = 1000
            requests_made = 0
            days_per_request = candles_per_request / candles_per_day
            
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
                    
                    current_end = current_start
                    days -= current_days
                    requests_made += 1
                    
                    time.sleep(CONFIG['rate_limit_delay'])
                else:
                    logger.warning(f"No data in batch {requests_made + 1}, stopping")
                    break
            
            if all_data:
                combined_df = pd.concat(all_data, ignore_index=True)
                combined_df = combined_df.drop_duplicates(subset=['timestamp'])
                combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
                
                logger.info(f"Total historical data: {len(combined_df)} candles from {requests_made} requests")
                return combined_df
            else:
                logger.warning(f"No data fetched for {symbol}")
                return pd.DataFrame()
                
    except Exception as e:
        logger.error(f"❌ Error fetching historical data for {symbol}: {e}")
        raise  # Re-raise - don't hide errors

def get_regime_data(
    symbol: str,
    timeframe: str = '1h',
    days: int = 90
) -> pd.DataFrame:
    """
    Main function for regime detection.
    Fetches and prepares data for regime analysis.
    """
    logger.info(f"Fetching regime data for {symbol} ({timeframe}, {days} days)...")
    
    df = fetch_historical_data(symbol, timeframe, days)
    
    if df.empty or len(df) <= 10:
        raise ValueError(f"❌ Insufficient data for {symbol}: {len(df)} candles")
    
    # Ensure proper sorting
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Add returns for regime detection
    df['returns'] = df['close'].pct_change()
    
    # Add simple moving averages for regime features
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()
    
    logger.info(f"✅ Regime data ready: {len(df)} candles")
    logger.debug(f"  Latest: ${df['close'].iloc[-1]:.2f}, Return: {df['returns'].iloc[-1]:.4%}")
    
    return df

# ------------------------------------------------------------
# Data feed class
# ------------------------------------------------------------
class BinanceDataFeed:
    """Simple data feed for regime detection and trading engine"""
    def __init__(self):
        self.client = client
        self.rate_limit_delay = CONFIG['rate_limit_delay']
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
        """Get current prices for multiple symbols with rate limiting"""
        prices = {}
        for i, symbol in enumerate(symbols):
            price = self.get_price(symbol)
            if price is not None:
                prices[symbol] = price
            
            if i < len(symbols) - 1:
                time.sleep(self.rate_limit_delay)
        
        logger.debug(f"Fetched prices for {len(prices)}/{len(symbols)} symbols")
        return prices

# Initialize global data feed instance
data_feed = BinanceDataFeed()

# ------------------------------------------------------------
# Test / Example usage
# ------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("📊 Binance Data Feed Test")
    print("=" * 60)
    
    print(f"\n🔧 Trading Mode: {CONFIG['trading_mode']}")
    print(f"🔧 Testnet: {CONFIG.get('testnet', False)}")
    
    # Test current price
    symbol = 'BTC/USDT'
    price = get_current_price(symbol)
    if price:
        print(f"\n💰 Current {symbol}: ${price:,.2f}")
    
    # Test regime data
    print(f"\n📈 Fetching regime data for {symbol}...")
    try:
        df = get_regime_data(symbol, timeframe='4h', days=7)
        print(f"✅ Data fetched: {len(df)} candles")
        print(f"   Latest price: ${df['close'].iloc[-1]:,.2f}")
    except Exception as e:
        print(f"❌ Failed: {e}")
    
    print("\n✅ Data feed test complete!")