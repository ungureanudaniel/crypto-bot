# debug_regimes.py
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.regime_switcher import predict_regime
from data_feed import fetch_ohlcv

def check_all_regimes():
    print("Checking current market regimes...")
    
    coins = ['BTC/USDC', 'ETH/USDC', 'SOL/USDC', 'BNB/USDC', 'XRP/USDC']
    
    for coin in coins:
        try:
            df = fetch_ohlcv(coin, '15m')
            if not df.empty:
                regime = predict_regime(df)
                print(f"🔍 {coin}: {regime}")
                
                # Show recent price action
                recent = df.tail(3)
                for i, row in recent.iterrows():
                    change_pct = (row['close'] - row['open']) / row['open'] * 100
                    print(f"   {row['timestamp'].strftime('%H:%M')}: ${row['close']:.2f} ({change_pct:+.1f}%)")
            else:
                print(f"❌ {coin}: No data")
                
        except Exception as e:
            print(f"❌ {coin}: Error - {e}")
    
    print("\n📊 Regime Legend:")
    print("   0 = Range-Bound 📊 (NO TRADES)")
    print("   1 = Trending 📈 (NO TRADES in current config)")  
    print("   2 = Breakout 🚀 (TRADES EXECUTED)")