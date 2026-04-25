# export_data.py
import os
import sys
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.data_feed import fetch_historical_data
from datetime import datetime, timedelta
days = 365  # ~1 years of 1d-hour data
interval = "1d"
# Fetch 2 months of 1-hour BNB/USDT data
df = fetch_historical_data(
    symbol="BNB/USDT",
    interval=interval,
    days=days
)

# Save to CSV
df.to_csv(f"bnb_usdt_{interval}_{days}d.csv", index=False)
print(f"✅ Saved {len(df)} candles to bnb_usdt_{interval}_{days}d.csv")
print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")