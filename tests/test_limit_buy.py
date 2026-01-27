# test_direct_order.py
import os
import sys
from binance.client import Client
from dotenv import load_dotenv
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
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
def test_direct_order():
    """Test placing a limit order directly on Binance Testnet"""
    
    print("=" * 60)
    print("🧪 DIRECT BINANCE TESTNET ORDER TEST")
    print("=" * 60)
    
    # Get API keys
    api_key = CONFIG.get('binance_api_key', '')
    api_secret = CONFIG.get('binance_api_secret', '')
    
    if not api_key or not api_secret:
        print("❌ Missing API keys in config file")
        return
    
    # Initialize client for testnet
    client = Client(api_key, api_secret)
    client.API_URL = 'https://testnet.binance.vision'
    
    print(f"✅ Connected to Binance Testnet")
    print(f"🔗 API URL: {client.API_URL}")
    
    # Test with BOTH USDT and USDC
    test_symbols = [
        ('BTCUSDC', 'USDC'),
        ('BTCUSDT', 'USDT'),
        ('ETHUSDC', 'USDC'),
        ('ETHUSDT', 'USDT'),
        ('SOLUSDC', 'USDC'),
        ('SOLUSDT', 'USDT')
    ]
    
    for symbol, quote in test_symbols:
        print(f"\n📊 Testing {symbol}:")
        print("-" * 40)
        
        try:
            # 1. Check if symbol exists
            symbol_info = client.get_symbol_info(symbol)
            print(symbol_info)
            if symbol_info:
                status = symbol_info.get('status', 'UNKNOWN')
                base_asset = symbol_info['baseAsset']
                quote_asset = symbol_info['quoteAsset']
                
                print(f"✅ Symbol exists")
                print(f"   Status: {status}")
                print(f"   Base: {base_asset}, Quote: {quote_asset}")
                
                # Check account balance for quote asset
                account = client.get_account()
                quote_balance = next(
                    (b for b in account['balances'] if b['asset'] == quote_asset),
                    None
                )
                
                if quote_balance:
                    free = float(quote_balance['free'])
                    locked = float(quote_balance['locked'])
                    print(f"   {quote_asset} Balance: Free={free:.2f}, Locked={locked:.2f}")
                
                # Get current price
                ticker = client.get_symbol_ticker(symbol=symbol)
                current_price = float(ticker['price'])
                print(f"   Current Price: ${current_price:.2f}")
                
                # Check filters
                min_qty = None
                step_size = None
                min_notional = None
                
                for f in symbol_info['filters']:
                    if f['filterType'] == 'LOT_SIZE':
                        min_qty = float(f.get('minQty', 0))
                        step_size = float(f['stepSize'])
                        print(f"   Min Qty: {min_qty}")
                        print(f"   Step Size: {step_size}")
                    elif f['filterType'] == 'MIN_NOTIONAL':
                        min_notional = float(f.get('minNotional', 0))
                        print(f"   Min Notional: ${min_notional:.2f}")
                
                # Try to place a SMALL test order
                if status == 'TRADING':
                    print(f"\n   🚀 Attempting test limit order...")
                    
                    # Calculate order parameters
                    if min_qty:
                        test_amount = max(min_qty * 2, 0.0001)  # At least 2x min or 0.0001
                    else:
                        test_amount = 0.0001
                    
                    # Round to step size
                    if step_size:
                        steps = test_amount / step_size
                        test_amount = int(steps) * step_size
                    
                    limit_price = current_price * 0.95  # 5% below market (won't fill)
                    
                    # Check min notional
                    order_value = test_amount * limit_price
                    if min_notional and order_value < min_notional:
                        test_amount = min_notional / limit_price
                        # Round up
                        if step_size:
                            steps = test_amount / step_size
                            test_amount = (int(steps) + 1) * step_size
                    
                    print(f"   Test Order:")
                    print(f"     Amount: {test_amount} {base_asset}")
                    print(f"     Limit Price: ${limit_price:.2f}")
                    print(f"     Value: ${test_amount * limit_price:.2f}")
                    
                    try:
                        # Place limit buy order
                        order = client.order_limit_buy(
                            symbol=symbol,
                            quantity=test_amount,
                            price=str(limit_price)
                        )
                        
                        print(f"   ✅ ORDER SUCCESSFUL!")
                        print(f"     Order ID: {order['orderId']}")
                        print(f"     Status: {order['status']}")
                        
                        # Cancel the order immediately
                        print(f"   🗑️  Cancelling order...")
                        cancel_result = client.cancel_order(
                            symbol=symbol,
                            orderId=order['orderId']
                        )
                        print(f"   ✅ Order cancelled")
                        
                    except Exception as order_error:
                        print(f"   ❌ ORDER FAILED: {order_error}")
                        
                else:
                    print(f"   ⚠️  Symbol not in TRADING status")
                    
            else:
                print(f"❌ Symbol not found on exchange")
                
        except Exception as e:
            print(f"❌ Error testing {symbol}: {e}")
    
    print("\n" + "=" * 60)
    print("📋 SUMMARY:")
    print("Check if any symbols succeeded or all failed.")
    print("If all failed, check:")
    print("1. API keys are correct")
    print("2. Testnet account has balance")
    print("3. Visit: https://testnet.binance.vision")

if __name__ == "__main__":
    test_direct_order()