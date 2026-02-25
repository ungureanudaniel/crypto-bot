from binance.client import Client
import os

client = Client(os.getenv('BINANCE_TESTNET_API_KEY'), os.getenv('BINANCE_TESTNET_PRIVATE_KEY'), testnet=True)
account = client.get_account()

for balance in account['balances']:
    asset = balance['asset']
    free = float(balance['free'])
    if free > 0 and asset != 'USDT':
        print(f"Checking {asset}...")
        try:
            ticker = client.get_symbol_ticker(symbol=f"{asset}USDT")
            print(f"  {asset} price: {ticker['price']}")
        except Exception as e:
            print(f"  ERROR: {e}")
