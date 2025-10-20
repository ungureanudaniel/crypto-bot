# debug_portfolio.py
import json

def check_portfolio_structure():
    """Check what's actually in portfolio.json"""
    try:
        with open("portfolio.json", "r") as f:
            portfolio = json.load(f)
        
        print("🔍 Portfolio Structure Analysis:")
        print(f"💰 Cash: ${portfolio.get('cash_balance', 0):.2f}")
        
        holdings = portfolio.get('holdings', {})
        print(f"📦 Holdings count: {len(holdings)}")
        for coin, amount in holdings.items():
            print(f"   • {coin}: {amount}")
        
        positions = portfolio.get('positions', {})
        print(f"🛡️ Positions count: {len(positions)}")
        for symbol, position in positions.items():
            print(f"   • {symbol}: {position['amount']} units")
            print(f"     Entry: ${position['entry_price']:.2f}")
            print(f"     SL: ${position['stop_loss']:.2f}")
            print(f"     TP: ${position['take_profit']:.2f}")
            
        if not holdings and not positions:
            print("❌ No holdings OR positions found!")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_portfolio_structure()