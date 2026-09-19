"""
market_gate.py
==============
Live helpers for the trend-following strategy:

  closed_only()   drop the candle that is still forming. The backtest only ever acts on candles
                  that have CLOSED, so live must too (a forming candle can "break out" on partial
                  data and then fall back).
  btc_uptrend()   True while BTC's close is above its N-day EMA. New longs are only allowed then.
                  Mirrors Backtester._build_btc_gate (ewm with adjust=False, no look-ahead).
"""
from typing import Optional

import pandas as pd

BARS_PER_DAY = {'1m': 1440, '5m': 288, '15m': 96, '30m': 48, '1h': 24, '2h': 12,
                '4h': 6, '6h': 4, '12h': 2, '1d': 1}


def timeframe_seconds(timeframe: str) -> int:
    return int(86400 / BARS_PER_DAY.get(timeframe, 6))


def closed_only(df: pd.DataFrame, timeframe: str, now: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """Return df without its last row if that candle has not closed yet.
    `timestamp` is the candle OPEN time (naive UTC), as produced by data_feed.fetch_ohlcv."""
    if df is None or df.empty or 'timestamp' not in df.columns:
        return df
    now = now if now is not None else pd.Timestamp.now(tz='UTC').tz_localize(None)
    last_open = pd.Timestamp(df['timestamp'].iloc[-1])
    if last_open + pd.Timedelta(seconds=timeframe_seconds(timeframe)) > now:
        return df.iloc[:-1]
    return df


def btc_uptrend(df: pd.DataFrame, days: int, timeframe: str) -> Optional[bool]:
    """True/False, or None when there isn't enough history to decide (callers must fail closed)."""
    span = int(days * BARS_PER_DAY.get(timeframe, 6))
    if df is None or len(df) < span + 1:
        return None
    close = df['close']
    ema = close.ewm(span=span, adjust=False).mean()
    return bool(close.iloc[-1] > ema.iloc[-1])
