import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Determine which API to use based on environment
TRADING_MODE = os.getenv('TRADING_MODE', 'paper').lower()

if TRADING_MODE == 'testnet':
    BASE_URL = "https://testnet.binance.vision/api/v3/"
    print("🧪 Testing BINANCE TESTNET API")
else:
    BASE_URL = "https://api1.binance.com/api/v3/"
    print("🧪 Testing BINANCE MAINNET API")

def get_server_time():
    """Get server time"""
    endpoint = "time"
    try:
        response = requests.get(BASE_URL + endpoint, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to get server time: {e}")
        return None

def get_price(symbol: str):
    """Get current price for a symbol"""
    endpoint = "ticker/price"
    params = {"symbol": symbol}
    try:
        response = requests.get(BASE_URL + endpoint, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to get price for {symbol}: {e}")
        return None

def get_exchange_info():
    """Get exchange info to see available symbols"""
    endpoint = "exchangeInfo"
    try:
        response = requests.get(BASE_URL + endpoint, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to get exchange info: {e}")
        return None

def test_available_symbols():
    """Test some common symbols"""
    symbols_to_test = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "ADAUSDT"]
    
    print("\n📊 Testing available symbols:")
    for symbol in symbols_to_test:
        price_data = get_price(symbol)
        if price_data and 'price' in price_data:
            print(f"✅ {symbol}: ${float(price_data['price']):.4f}")
        else:
            print(f"❌ {symbol}: Not available")

def test_rate_limits():
    """Test rate limiting by making multiple requests"""
    print("\n⏱️ Testing rate limits (3 rapid requests):")
    for i in range(3):
        time_data = get_server_time()
        if time_data:
            print(f"  Request {i+1}: OK")
        else:
            print(f"  Request {i+1}: Failed (might be rate limited)")

def test_connection_with_env():
    """Test if environment variables are loaded"""
    print("\n🔧 Environment check:")
    
    # Check trading mode
    print(f"  Trading Mode: {TRADING_MODE}")
    print(f"  Base URL: {BASE_URL}")
    
    # Check if API keys are loaded (optional)
    if os.getenv('BINANCE_TESTNET_API_KEY'):
        print("  ✅ Testnet API key found in .env")
    elif os.getenv('BINANCE_API_KEY'):
        print("  ✅ Mainnet API key found in .env")
    else:
        print("  ℹ️ No API keys in .env (using public endpoints)")
    
    if os.getenv('TELEGRAM_TOKEN'):
        print("  ✅ Telegram token found in .env")
    else:
        print("  ⚠️ Telegram token not found in .env")

def main():
    print("=" * 60)
    print("🤖 BINANCE API TEST WITH .env CONFIGURATION")
    print("=" * 60)
    
    # Test environment setup
    test_connection_with_env()
    
    # Test server connection
    print("\n🕐 Testing server connection...")
    time_data = get_server_time()
    
    if time_data:
        print(f"✅ Connected! Server time: {time_data['serverTime']}")
        
        # Test specific symbol (adjust based on available pairs)
        if TRADING_MODE == 'testnet':
            # Testnet might have different symbols
            test_symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
        else:
            test_symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "TRXUSDT"]
        
        for symbol in test_symbols:
            print(f"\n💵 Testing {symbol}...")
            price_data = get_price(symbol)
            
            if price_data and 'price' in price_data:
                print(f"✅ {symbol}: ${float(price_data['price']):.4f}")
            else:
                print(f"❌ {symbol}: Not available or error")
                
                # Try USDC pair if USDT fails
                if symbol.endswith("USDT"):
                    usdc_symbol = symbol.replace("USDT", "USDC")
                    print(f"  Trying {usdc_symbol} instead...")
                    price_data = get_price(usdc_symbol)
                    if price_data and 'price' in price_data:
                        print(f"✅ {usdc_symbol}: ${float(price_data['price']):.4f}")
        
        # Test more symbols
        test_available_symbols()
        
        # Test rate limits
        test_rate_limits()
        
        print("\n" + "=" * 60)
        print("🎉 API TEST COMPLETED SUCCESSFULLY!")
        
    else:
        print("❌ Failed to connect to Binance API")
        print("\n💡 Troubleshooting:")
        print("1. Check your internet connection")
        print("2. If using testnet, ensure BASE_URL is correct")
        print("3. Try without VPN/proxy")
        print("4. Check if Binance is blocked in your region")

if __name__ == "__main__":
    main()