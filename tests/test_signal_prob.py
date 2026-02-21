# test_signal_frequency.py
import sys
import os
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.data_feed import data_feed
from modules.strategy_tools import generate_trade_signal
from modules.portfolio import load_portfolio

def test_signal_frequency(symbol, days=30):
    """Test how often signals would have triggered over past X days"""
    
    logger.info(f"📊 Testing signal frequency for {symbol} over {days} days")
    
    # Fetch historical data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # Get data in chunks to simulate real-time scanning
    df = data_feed.get_historical_data(symbol, interval="15m", days=days)
    
    if df.empty or len(df) < 100:
        logger.error(f"Insufficient data: {len(df)} rows")
        return
    
    logger.info(f"Loaded {len(df)} candles from {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
    
    # Simulate scanning every bar
    signals = []
    portfolio = load_portfolio()
    equity = portfolio.get('cash_balance', 1000)
    
    # Scan every 4th bar to simulate 1 hour intervals (15min * 4 = 1 hour)
    scan_interval = 4
    
    for i in range(100, len(df), scan_interval):  # Start after enough data
        window_df = df.iloc[:i].copy()
        
        try:
            signal = generate_trade_signal(window_df, equity, 0.02)
            
            if signal:
                signals.append({
                    'timestamp': window_df['timestamp'].iloc[-1],
                    'price': window_df['close'].iloc[-1],
                    'signal': signal
                })
                logger.info(f"✅ Signal at {window_df['timestamp'].iloc[-1]}: {signal['side']} @ ${signal['entry']:.2f}")
        except Exception as e:
            logger.error(f"Error at bar {i}: {e}")
    
    # Results
    total_scans = (len(df) - 100) // scan_interval
    logger.info("=" * 60)
    logger.info(f"📊 RESULTS for {symbol}:")
    logger.info(f"   Period: {days} days")
    logger.info(f"   Total scans: {total_scans}")
    logger.info(f"   Signals found: {len(signals)}")
    logger.info(f"   Signal frequency: {len(signals)/total_scans*100:.2f}%")
    
    if signals:
        logger.info(f"\n   Signal times:")
        for s in signals[:10]:  # Show first 10
            logger.info(f"     - {s['timestamp']}: {s['signal']['side']} @ ${s['price']:.2f}")
    
    return signals

if __name__ == "__main__":
    # Test with your main coins
    symbols = ["BTC/USDC", "ETH/USDC", "SOL/USDC"]
    
    for symbol in symbols:
        test_signal_frequency(symbol, days=30)
        print("\n" + "="*60 + "\n")