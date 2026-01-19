# debug_trading_logic.py
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.regime_switcher import predict_regime
from data_feed import fetch_ohlcv
from trade_engine import execute_trade

def test_trading_logic():
    print("Testing trading logic across all regimes...")
    
    test_cases = [
        ('BTC/USDC', '15m'),
        ('ETH/USDC', '15m'),
        ('SOL/USDC', '15m')
    ]
    
    for symbol, timeframe in test_cases:
        print(f"\n🔍 Testing {symbol} ({timeframe}):")
        
        try:
            # Get current data
            df = fetch_ohlcv(symbol, timeframe)
            if df.empty:
                print(f"   ❌ No data for {symbol}")
                continue
            
            # Predict regime
            regime_prediction = predict_regime(df)
            print(f"   📊 Regime prediction: {regime_prediction}")
            
            # Get current price
            current_price = df.iloc[-1]['close']
            print(f"   💰 Current price: ${current_price:.2f}")
            
            # Test what execute_trade would do
            print(f"   🤖 What bot would do:")
            
            if "Range-Bound" in regime_prediction:
                print("      ❌ SKIP TRADE (range-bound)")
            elif "Trending" in regime_prediction:
                print("      ❌ SKIP TRADE (trending) - THIS SHOULD TRADE!")
            elif "Breakout" in regime_prediction:
                print("      ✅ EXECUTE TRADE (breakout)")
            else:
                print(f"      ❓ Unknown regime: {regime_prediction}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")

# quick_test_eth.py
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from trade_engine import execute_trade
from modules.regime_switcher import predict_regime
from data_feed import fetch_ohlcv

def test_eth_trade():
    print("Testing ETH trending trade...")
    
    symbol = "ETH/USDC"
    df = fetch_ohlcv(symbol, "15m")
    
    if not df.empty:
        regime = predict_regime(df)
        price = df.iloc[-1]['close']
        
        print(f"Symbol: {symbol}")
        print(f"Regime: {regime}")
        print(f"Price: ${price:.2f}")
        print("Executing trade (should work now)...")
        
        # This should now execute a trade!
        execute_trade(symbol, regime, price)
        print("✅ Trade execution completed!")
    else:
        print("❌ No data")

if __name__ == "__main__":
    test_eth_trade()
    # test_trading_logic()