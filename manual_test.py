import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def manual_test():
    """Quick manual test of the actual functions"""
    print("Manual test of trade engine...")
    
    try:
        from trade_engine import (
            load_config, load_portfolio, calculate_position_size,
            calculate_dynamic_stop_loss, get_trading_mode
        )
        
        print("1. Loading config...")
        config = load_config()
        print(f"   - Starting balance: ${config.get('starting_balance', 0)}")
        print(f"   - Coins: {len(config.get('coins', []))}")
        
        print("2. Loading portfolio...")
        portfolio = load_portfolio()
        print(f"   - Cash balance: ${portfolio['cash_balance']}")
        print(f"   - Holdings: {len(portfolio.get('holdings', {}))}")
        
        print("3. Testing position sizing...")
        size = calculate_position_size("BTC/USDC", 50000.0, portfolio)
        print(f"   - Position size for BTC @ $50,000: {size:.6f} units")
        
        print("4. Testing stop loss...")
        sl, tp = calculate_dynamic_stop_loss("BTC/USDC", 50000.0, "long")
        print(f"   - Stop loss: ${sl:.2f}")
        print(f"   - Take profit: ${tp:.2f}")
        
        print("5. Checking trading mode...")
        mode = get_trading_mode()
        print(f"   - Current mode: {mode}")
        
        print("\n✅ MANUAL TEST COMPLETED SUCCESSFULLY!")
        return True
        
    except Exception as e:
        print(f"❌ Manual test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = manual_test()