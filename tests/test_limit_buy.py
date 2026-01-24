# test_limit_order.py
import sys
import os
import logging

# Get the absolute path to the project root
current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_file_dir)  # Go up one level from services/

# Add project root to Python path
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'modules'))

# Now we can import modules
try:
    from config_loader import config
    CONFIG = config.config
except ImportError:
    CONFIG = {'trading_mode': 'paper', 'coins': ['BTC/USDC', 'ETH/USDC']}

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("🔍 Testing limit order functionality...")
print("=" * 60)

# Test 1: Import modules
print("\n1️⃣ Testing imports...")
try:
    from modules.trade_engine import trading_engine
    print("✅ trading_engine imported")
    
    # Check if place_limit_order exists
    if hasattr(trading_engine, 'place_limit_order'):
        print("✅ place_limit_order method exists")
    else:
        print("❌ place_limit_order method NOT found!")
        
except Exception as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# Test 2: Check trading mode
print("\n2️⃣ Checking trading mode...")
try:
    mode = getattr(trading_engine, 'trading_mode', 'paper')
    print(f"✅ Trading mode: {mode}")
except Exception as e:
    print(f"❌ Error getting mode: {e}")

# Test 3: Test the place_limit_order function directly
print("\n3️⃣ Testing place_limit_order function...")
try:
    # Test with paper trading
    symbol = "SOL/USDC"
    side = "buy"
    amount = 2.0
    price = 127.25
    
    print(f"Testing with:")
    print(f"  Symbol: {symbol}")
    print(f"  Side: {side}")
    print(f"  Amount: {amount}")
    print(f"  Price: ${price}")
    
    success, message = trading_engine.place_limit_order(
        symbol=symbol,
        side=side,
        amount=amount,
        price=price
    )
    
    print(f"\n✅ Function executed!")
    print(f"  Success: {success}")
    print(f"  Message: {message}")
    
    if success:
        print("🎉 Limit order placed successfully!")
    else:
        print("❌ Failed to place limit order")
        print(f"  Error: {message}")
        
except Exception as e:
    print(f"❌ Function error: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Check portfolio
print("\n4️⃣ Checking portfolio...")
try:
    from modules.portfolio import load_portfolio, save_portfolio
    
    portfolio = load_portfolio()
    cash = portfolio.get('cash_balance', 0)
    pending_orders = portfolio.get('pending_orders', [])
    
    print(f"💰 Cash balance: ${cash:,.2f}")
    print(f"📋 Pending orders: {len(pending_orders)}")
    
    if pending_orders:
        print("\nPending orders details:")
        for i, order in enumerate(pending_orders, 1):
            print(f"  {i}. {order.get('symbol')} {order.get('side')} "
                  f"{order.get('amount')} @ ${order.get('price', 0):.2f}")
            print(f"     ID: {order.get('id', 'N/A')}")
            
except Exception as e:
    print(f"❌ Portfolio error: {e}")

# Test 5: Try to place another order with invalid data
print("\n5️⃣ Testing error cases...")
try:
    # Test with invalid amount
    print("Testing invalid amount (0)...")
    success, message = trading_engine.place_limit_order(
        symbol="BTC/USDC",
        side="buy",
        amount=0,
        price=50000
    )
    print(f"  Result: Success={success}, Message={message}")
    
    # Test with invalid price
    print("\nTesting invalid price (0)...")
    success, message = trading_engine.place_limit_order(
        symbol="BTC/USDC",
        side="buy",
        amount=0.001,
        price=0
    )
    print(f"  Result: Success={success}, Message={message}")
    
    # Test with invalid side
    print("\nTesting invalid side...")
    success, message = trading_engine.place_limit_order(
        symbol="BTC/USDC",
        side="invalid",
        amount=0.001,
        price=50000
    )
    print(f"  Result: Success={success}, Message={message}")
    
except Exception as e:
    print(f"❌ Error test failed: {e}")

print("\n" + "=" * 60)
print("✅ Test complete!")

# Cleanup if you want to remove test orders
response = input("\nClean up test orders? (y/n): ")
if response.lower() == 'y':
    try:
        from modules.portfolio import load_portfolio, save_portfolio
        portfolio = load_portfolio()
        before = len(portfolio.get('pending_orders', []))
        portfolio['pending_orders'] = []
        save_portfolio(portfolio)
        after = len(portfolio.get('pending_orders', []))
        print(f"🧹 Cleared {before - after} orders")
    except Exception as e:
        print(f"❌ Cleanup error: {e}")