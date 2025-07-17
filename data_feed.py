import ccxt
import pandas as pd
import json
import logging

with open('config.json', 'r') as f:
    config = json.load(f)

logging.basicConfig(level=logging.INFO)

EXCHANGE = ccxt.kraken()
LOOPBACK = 200

def fetch_ohlcv(coin, timeframe='1w'):
    logging.info(f"Fetching OHLCV data for {coin}...")
    try:
        ohlcv = EXCHANGE.fetch_ohlcv(coin, timeframe=timeframe, limit=LOOPBACK)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        logging.info(f"Successfully fetched OHLCV data for {coin}.")
        return df
    except Exception as e:
        logging.error(f"Error fetching OHLCV data for {coin}: {e}")
        return pd.DataFrame()
