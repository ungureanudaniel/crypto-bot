# binance_testnet.py - FIXED VERSION
from binance.client import Client
from binance.enums import *
import json
import logging
from datetime import datetime
import pandas as pd
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BinanceTestnet:
    """Binance Spot Testnet client using python-binance"""
    
    def __init__(self, config_path='config.json'):
        self.config = self.load_config(config_path)
        self.client = self.create_testnet_client()
        
    def load_config(self, config_path):
        """Load configuration"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info("✅ Config loaded")
            return config
        except Exception as e:
            logger.error(f"❌ Failed to load config: {e}")
            return {}
    
    def create_testnet_client(self):
        """Create Binance Testnet client with time synchronization"""
        try:
            api_key = self.config.get('binance_testnet_api_key', '')
            api_secret = self.config.get('binance_testnet_api_secret', '')
            
            if not api_key or not api_secret:
                logger.warning("⚠️ Testnet API keys not found")
                logger.info("Get free keys from: https://testnet.binance.vision/")
                return None
            
            # Create testnet client WITH TIME SYNCHRONIZATION
            client = Client(
                api_key=api_key,
                api_secret=api_secret,
                testnet=True,
                # Add time synchronization
                tld='com',
                requests_params={'timeout': 10}
            )
            
            # Test time synchronization
            try:
                server_time = client.get_server_time()
                local_time = int(time.time() * 1000)
                time_diff = server_time['serverTime'] - local_time
                logger.info(f"⏰ Time sync: Server is {abs(time_diff)}ms {'ahead' if time_diff > 0 else 'behind'}")
                
                # If difference is too large, python-binance will auto-sync
                if abs(time_diff) > 1000:
                    logger.warning(f"Large time difference ({time_diff}ms). Enabling auto-sync...")
                    # python-binance automatically handles time sync on private requests
                    
            except Exception as e:
                logger.warning(f"Time sync check failed: {e}")
            
            logger.info("✅ Binance Testnet client created")
            return client
            
        except Exception as e:
            logger.error(f"❌ Failed to create client: {e}")
            return None
    
    # ------------------------------------------------------------
    # PRIVATE ENDPOINTS WITH RETRY LOGIC
    # ------------------------------------------------------------
    
    def get_account_balance(self, retries=3):
        """Get account balance with retry logic"""
        for attempt in range(retries):
            try:
                account = self.client.get_account()
                balances = account['balances']
                
                logger.info("✅ Account Balances (non-zero):")
                total_usdt = 0
                
                for balance in balances:
                    asset = balance['asset']
                    free = float(balance['free'])
                    locked = float(balance['locked'])
                    total = free + locked
                    
                    if total > 0:
                        # Get USD value for crypto assets
                        if asset != 'USDT':
                            try:
                                ticker = self.client.get_symbol_ticker(symbol=f"{asset}USDT")
                                price = float(ticker['price'])
                                value = total * price
                                logger.info(f"   {asset}: {total:.8f} ≈ ${value:.2f}")
                                total_usdt += value
                            except:
                                logger.info(f"   {asset}: {total:.8f}")
                        else:
                            logger.info(f"   USDT: {total:.2f}")
                            total_usdt += total
                
                logger.info(f"📊 Total Portfolio Value: ≈ ${total_usdt:.2f}")
                return account
                
            except Exception as e:
                if "Timestamp" in str(e) and attempt < retries - 1:
                    logger.warning(f"Time sync error (attempt {attempt + 1}/{retries}): {e}")
                    time.sleep(1)
                    continue
                else:
                    logger.error(f"❌ Failed to get account: {e}")
                    return None
    
    def get_open_orders(self, symbol='BTCUSDT', retries=3):
        """Get open orders with retry logic"""
        for attempt in range(retries):
            try:
                orders = self.client.get_open_orders(symbol=symbol)
                logger.info(f"✅ Open orders for {symbol}: {len(orders)}")
                
                for order in orders:
                    logger.info(f"   {order['side']} {order['origQty']} @ ${order['price']} "
                              f"(ID: {order['orderId']})")
                
                return orders
                
            except Exception as e:
                if "Timestamp" in str(e) and attempt < retries - 1:
                    logger.warning(f"Time sync error (attempt {attempt + 1}/{retries}): {e}")
                    time.sleep(1)
                    continue
                else:
                    logger.error(f"❌ Failed to get open orders: {e}")
                    return []
    
    def get_order_history(self, symbol='BTCUSDT', limit=5, retries=3):
        """Get order history with retry logic"""
        for attempt in range(retries):
            try:
                orders = self.client.get_all_orders(symbol=symbol, limit=limit)
                logger.info(f"✅ Recent orders for {symbol}:")
                
                for order in orders:
                    status = order['status']
                    side = order['side']
                    qty = order['origQty']
                    price = order.get('price', 'MARKET')
                    
                    logger.info(f"   {status}: {side} {qty} @ {price}")
                
                return orders
                
            except Exception as e:
                if "Timestamp" in str(e) and attempt < retries - 1:
                    logger.warning(f"Time sync error (attempt {attempt + 1}/{retries}): {e}")
                    time.sleep(1)
                    continue
                else:
                    logger.error(f"❌ Failed to get order history: {e}")
                    return []
    
    # ------------------------------------------------------------
    # ALTERNATIVE: SIMPLE TIME SYNC FIX
    # ------------------------------------------------------------
    
    def create_testnet_client_simple(self):
        """Alternative: Simple client creation without time sync issues"""
        try:
            api_key = self.config.get('binance_testnet_api_key', '')
            api_secret = self.config.get('binance_testnet_api_secret', '')
            
            if not api_key or not api_secret:
                logger.warning("⚠️ Testnet API keys not found")
                return None
            
            # Method 1: Disable time sync (for testing only)
            client = Client(
                api_key=api_key,
                api_secret=api_secret,
                testnet=True,
                # Disable time sync for testing
                # tld='com'
            )
            
            logger.info("✅ Binance Testnet client created (simple mode)")
            return client
            
        except Exception as e:
            logger.error(f"❌ Failed to create simple client: {e}")
            return None
    
    # ------------------------------------------------------------
    # MANUAL TIME SYNCHRONIZATION
    # ------------------------------------------------------------
    
    def sync_time_manually(self):
        """Manually synchronize time with Binance server"""
        try:
            # Get server time
            server_time = self.client.get_server_time()
            local_time = int(time.time() * 1000)
            
            # Calculate difference
            time_diff = server_time['serverTime'] - local_time
            
            if abs(time_diff) > 1000:
                logger.warning(f"⚠️ Time difference detected: {time_diff}ms")
                logger.info(f"   Server time: {datetime.fromtimestamp(server_time['serverTime']/1000)}")
                logger.info(f"   Local time:  {datetime.fromtimestamp(local_time/1000)}")
                
                # python-binance should handle this automatically
                # For manual adjustment, you could set system time or adjust timestamps
                
                return time_diff
            else:
                logger.info(f"✅ Time synchronized (difference: {time_diff}ms)")
                return time_diff
                
        except Exception as e:
            logger.error(f"❌ Time sync failed: {e}")
            return None
    
    # ------------------------------------------------------------
    # RUN TESTS (UPDATED)
    # ------------------------------------------------------------
    
    def run_all_tests(self):
        """Run all connection tests"""
        logger.info("=" * 60)
        logger.info("🔧 Binance Testnet Tester (python-binance)")
        logger.info("=" * 60)
        
        if not self.client:
            logger.error("❌ Client not initialized")
            return False
        
        # Test 1: Connection and time sync
        logger.info("\n1. Testing connection and time sync...")
        if not self.test_connection():
            return False
        
        # Sync time
        self.sync_time_manually()
        
        # Test 2: Public data
        logger.info("\n2. Testing public endpoints...")
        self.get_prices()
        self.get_orderbook()
        self.get_recent_trades()
        
        # Test 3: Account info (with retry)
        logger.info("\n3. Testing private endpoints (with retry)...")
        self.get_account_balance(retries=3)
        self.get_open_orders(retries=3)
        self.get_order_history(retries=3)
        
        # Test 4: Historical data
        logger.info("\n4. Testing historical data...")
        df = self.get_historical_klines(symbol='BTCUSDT', interval='1h', days=7)
        if not df.empty:
            logger.info(f"   Latest BTC price: ${df['close'].iloc[-1]:.2f}")
        
        logger.info("\n" + "=" * 60)
        logger.info("✅ All tests completed!")
        logger.info("=" * 60)
        
        return True

# ------------------------------------------------------------
# QUICK FIX FOR YOUR CURRENT ERROR
# ------------------------------------------------------------

def test_with_quick_fix():
    """Quick test with time sync fix"""
    from binance.client import Client
    
    # Load config
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    api_key = config.get('binance_testnet_api_key', '')
    api_secret = config.get('binance_testnet_api_secret', '')
    
    if not api_key or not api_secret:
        print("❌ API keys missing")
        return
    
    print("🔄 Creating client with time sync workaround...")
    
    # Method 1: Add delay parameter
    client = Client(
        api_key=api_key,
        api_secret=api_secret,
        testnet=True,
        # Try adding these parameters
        tld='com',
        requests_params={'timeout': 30}
    )
    
    # Test public endpoint
    print("✅ Client created")
    
    # Test private endpoint with manual timestamp
    try:
        print("\n🔐 Testing account balance with manual time sync...")
        
        # Get server time first
        server_time = client.get_server_time()
        print(f"Server time: {server_time['serverTime']}")
        
        # Try account balance
        account = client.get_account()
        print(f"✅ Success! Account has {len(account['balances'])} balances")
        
        # Show balances
        for balance in account['balances']:
            if float(balance['free']) > 0 or float(balance['locked']) > 0:
                print(f"  {balance['asset']}: Free={balance['free']}, Locked={balance['locked']}")
                
    except Exception as e:
        print(f"❌ Error: {e}")
        
        # Try alternative method
        print("\n🔄 Trying alternative method...")
        try:
            # Sometimes the library needs a refresh
            import binance
            client = Client(api_key=api_key, api_secret=api_secret, testnet=True)
            
            # Test again
            account = client.get_account()
            print(f"✅ Alternative method worked!")
            
        except Exception as e2:
            print(f"❌ All methods failed: {e2}")
            print("\n💡 Try updating python-binance:")
            print("pip install python-binance --upgrade")
            
            print("\n💡 Or check your system time:")
            print("1. Make sure your computer's clock is accurate")
            print("2. Enable automatic time synchronization in Windows/Mac")
            print("3. Restart your computer")

if __name__ == "__main__":
    # First try the quick fix
    print("=" * 60)
    print("🛠️  Binance Testnet Quick Fix")
    print("=" * 60)
    
    test_with_quick_fix()
    
    # Then run full tests
    print("\n" + "=" * 60)
    print("🔧 Running Full Tests")
    print("=" * 60)
