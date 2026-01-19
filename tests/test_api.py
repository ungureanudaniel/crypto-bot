import requests

BASE_URL = "https://api1.binance.com/"

def get_server_time():
    endpoint = "/api/v3/time"
    response = requests.get(BASE_URL + endpoint)
    return response.json()

def get_price(symbol:str):
    url = f"{BASE_URL}/api/v3/ticker/price"
    params = {"symbol": symbol}
    response = requests.get(url, params=params)
    return response.json()

def main():
    try:
        time_data = get_server_time()
        print(f"Server Time: {time_data['serverTime']}")
        
        symbol = "TRXUSDC"
        price_data = get_price(symbol)
        print(f"Current price of {symbol}: {price_data['price']}")
        
        print("✅ API test completed successfully!")
    except Exception as e:
        print(f"❌ API test failed: {e}")

if __name__ == "__main__":
    main()

