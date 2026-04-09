
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.data_feed import data_feed
import pandas as pd

df = data_feed.get_ohlcv("BTC/USDT", interval="1h", limit=10)
print(f"Index type: {type(df.index)}")
print(f"First few indices: {df.index[:3]}")
print(f"Has datetime index: {isinstance(df.index, pd.DatetimeIndex)}")