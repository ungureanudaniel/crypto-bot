#!/usr/bin/env python3
from binance.client import Client
import pandas as pd

print("🔍 Testing Binance Testnet Data Availability...")

# Connect to testnet
client = Client("", "")
client.API_URL = 'https://testnet.binance.vision/api'

# Test different timeframes
symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT"]
timeframes = ["5m", "15m", "1h", "4h", "1d"]

print("\n📊 Checking available data:")

for symbol in symbols:
    print(f"\n{symbol}:")
    for tf in timeframes:
        try:
            klines = client.get_klines(symbol=symbol, interval=tf, limit=1000)
            print(f"  {tf}: {len(klines)} candles")
        except Exception as e:
            print(f"  {tf}: Error - {str(e)[:50]}")