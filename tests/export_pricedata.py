# export_data.py
import os
import sys
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.data_feed import fetch_historical_data
from datetime import datetime, timedelta
days = 1086  # ~3 years of 1-hour data
interval = "4h"
# Fetch 2 months of 1-hour ETH/USDT data
df = fetch_historical_data(
    symbol="ETH/USDT",
    interval=interval,
    days=days
)

# Save to CSV
df.to_csv(f"eth_usdt_{interval}_{days}d.csv", index=False)
print(f"✅ Saved {len(df)} candles to eth_usdt_{interval}_{days}d.csv")
print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")