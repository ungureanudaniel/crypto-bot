#!/usr/bin/env python3
# tests/test_testnetbalance.py
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'modules'))

# Now import
from modules.trade_engine import trading_engine
from modules.portfolio import load_portfolio
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    print("=" * 60)
    print("💰 TESTNET BALANCE CHECK")
    print("=" * 60)
    
    print(f"\n🔧 Trading Mode: {trading_engine.trading_mode}")
    print(f"🔧 Binance Client: {'✅ Available' if trading_engine.binance_client else '❌ Not Available'}")
    
    if trading_engine.binance_client:
        # Force balance sync
        print("\n🔄 Syncing balance from testnet...")
        trading_engine.sync_balance_from_testnet()
        
        # Show portfolio
        portfolio = load_portfolio()
        print("\n📊 Portfolio after sync:")
        print(f"   Cash: ${portfolio.get('cash_balance', 0):,.2f}")
        print(f"   Holdings: {len(portfolio.get('holdings', {}))} assets")
        
        # Show detailed balances
        print("\n💎 Detailed Balances:")
        print("-" * 40)
        for asset, data in portfolio.get('holdings', {}).items():
            if asset == 'USDC':
                print(f"   🔹 {asset}: {data.get('free', 0):.2f} (Free)")
            else:
                total = data.get('total', 0)
                if total > 0:
                    print(f"   • {asset}: {total:.6f}")
        
        # Get summary
        summary = trading_engine.get_portfolio_summary()
        print("\n📈 Portfolio Summary:")
        print(f"   Total Value: ${summary['portfolio_value']:,.2f}")
        print(f"   Cash: ${summary['cash_balance']:,.2f}")
        print(f"   Positions: {summary['active_positions']}")
        print(f"   Return: {summary['total_return_pct']:+.1f}%")
        print(f"   Last Sync: {summary.get('last_sync', 'Never')}")
    else:
        print("\n❌ No Binance client available. Check your API keys and trading mode.")

if __name__ == "__main__":
    main()