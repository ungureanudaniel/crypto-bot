
import logging
from time import time
import pandas as pd
import numpy as np
from typing import Dict
from ta.volatility import BollingerBands
from ta.trend import MACD, ADXIndicator
from ta.momentum import RSIIndicator

from modules.data_feed import fetch_ohlcv

logger = logging.getLogger(__name__)

ml_strategies = {}
sentiment_agent = None

async def get_ml_prediction(symbol: str, df: pd.DataFrame) -> Dict:
    """Get ML prediction for a symbol"""
    global ml_strategies
    try:
        from modules.ml_integration import MLStrategy
    except ImportError:
        logger.warning("⚠️ ml_integration module not available")
        return {}

    if symbol not in ml_strategies:
        ml_strategies[symbol] = MLStrategy(symbol)
        await ml_strategies[symbol].ensure_trained()
    
    ml_strategy = ml_strategies[symbol]
    prediction = await ml_strategy.get_prediction(df)
    
    return prediction

def calculate_donchian(df, length=20):
    """Returns Donchian channel: high, low, mid"""
    high = df['high'].rolling(length, min_periods=1).max()
    low = df['low'].rolling(length, min_periods=1).min()
    mid = (high + low) / 2
    return high, low, mid

def calculate_atr(df, length=14):
    """Returns ATR series"""
    try:
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift(1))
        low_close = np.abs(df['low'] - df['close'].shift(1))
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(length, min_periods=1).mean()
        return atr
    except Exception as e:
        logger.error(f"Error calculating ATR: {e}")
        return pd.Series([0] * len(df), index=df.index)
    
def rsi_signal(df, oversold=25, overbought=75):
    """
    RSI mean reversion signal
    Returns: 'long', 'short', or None
    """
    try:
        # Calculate RSI
        rsi_indicator = RSIIndicator(df['close'], window=14)
        rsi = rsi_indicator.rsi()
        
        current_rsi = rsi.iloc[-1]
        
        if pd.isna(current_rsi):
            return None
        
        # LONG: Oversold
        if current_rsi < oversold:
            return 'long'
        # SHORT: Overbought
        elif current_rsi > overbought:
            return 'short'
        else:
            return None
            
    except Exception as e:
        logger.error(f"Error in rsi_signal: {e}")
        return None

def macd_bar_exhaustion_signal(df, fast=12, slow=26, signal=9, 
                                min_bars=4,                      # IMPROVEMENT: increased from 3 to 4
                                min_shrink_bars=3,               # IMPROVEMENT: increased from 2 to 3
                                use_rsi_confirm=True, 
                                rsi_oversold=30,                 # IMPROVEMENT: tighter from 35 to 30
                                rsi_overbought=70,               # IMPROVEMENT: tighter from 65 to 70
                                use_volume_shrink=True):
    """
    MACD Bar Exhaustion Signal (Replaces standard MACD crossover)
    
    Entry Logic (Long):
      1. MACD histogram has 4+ consecutive red (negative) bars
      2. Last 3 red bars are shrinking in magnitude
      3. Price stabilizes (low volatility)
      4. One more small drop (final flush)
      5. Enter when histogram continues shrinking
    
    Entry Logic (Short):
      1. MACD histogram has 4+ consecutive green (positive) bars
      2. Last 3 green bars are shrinking in magnitude
      3. Price stabilizes (low volatility)
      4. One more small push up (final pump)
      5. Enter when histogram continues shrinking
    
    Returns: 'long', 'short', or None
    """
    try:
        if len(df) < 50:
            return None
        
        # Calculate MACD
        ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line

        # Identify positive/negative bars
        hist_pos = histogram > 0
        hist_neg = histogram < 0
        
        # Count consecutive bars
        pos_streak = hist_pos.groupby((hist_pos != hist_pos.shift()).cumsum()).cumsum()
        neg_streak = hist_neg.groupby((hist_neg != hist_neg.shift()).cumsum()).cumsum()
        pos_streak = pos_streak.where(hist_pos, 0).astype(int)
        neg_streak = neg_streak.where(hist_neg, 0).astype(int)
        
        # Bar magnitude (absolute value)
        hist_abs = histogram.abs()
        
        # Check if bars are shrinking (current smaller than previous)
        # Need at least min_shrink_bars consecutive shrinking bars
        bars_shrinking = True
        for i in range(1, min_shrink_bars + 1):
            if not (hist_abs.iloc[-i] < hist_abs.iloc[-i-1]):
                bars_shrinking = False
                break
        
        # Calculate ATR for stability check
        atr_series = calculate_atr(df)
        current_atr = atr_series.iloc[-1] if not atr_series.empty else 0
        atr_ma = atr_series.rolling(20).mean().iloc[-1] if not atr_series.empty else 0
        stable = current_atr < atr_ma
        
        # RSI confirmation (optional)
        rsi_confirm_ok = True
        if use_rsi_confirm:
            rsi = RSIIndicator(df['close'], window=14).rsi()
            current_rsi = rsi.iloc[-1]
        
        # Volume shrinkage: current volume < previous volume (for the final bar)
        volume_shrink_ok = True
        if use_volume_shrink:
            volume_shrink_ok = df['volume'].iloc[-1] < df['volume'].iloc[-2]
        
        # Price action: check for one more small drop/pump
        price_drop = df['close'].iloc[-1] < df['close'].iloc[-2]
        price_rise = df['close'].iloc[-1] > df['close'].iloc[-2]

        # IMPROVEMENT: ADX filter – skip if ADX > 30 (strong trend)
        adx = ADXIndicator(df['high'], df['low'], df['close'], window=14).adx()
        adx_val = adx.iloc[-1]
        if adx_val > 30:
            return None
        
        # ===== LONG SIGNAL =====
        if (neg_streak.iloc[-1] >= min_bars and 
            bars_shrinking and 
            stable and 
            price_drop and volume_shrink_ok):
            
            # RSI confirmation (optional)
            if use_rsi_confirm and current_rsi > rsi_oversold:
                rsi_confirm_ok = False
            
            if rsi_confirm_ok:
                logger.debug(f"MACD Bar Exhaustion LONG: {neg_streak.iloc[-1]} red bars, shrinking, stable, final drop, volume shrinking")
                return 'long'
        
        # ===== SHORT SIGNAL =====
        if (pos_streak.iloc[-1] >= min_bars and 
            bars_shrinking and 
            stable and 
            price_rise and volume_shrink_ok):
            
            # RSI confirmation (optional)
            if use_rsi_confirm and current_rsi < rsi_overbought:
                rsi_confirm_ok = False
            
            if rsi_confirm_ok:
                logger.debug(f"MACD Bar Exhaustion SHORT: {pos_streak.iloc[-1]} green bars, shrinking, stable, final pump, volume shrinking")
                return 'short'
        
        return None
        
    except Exception as e:
        logger.error(f"Error in macd_bar_exhaustion_signal: {e}")
        return None


def bollinger_band_signal(df, deviation=2):
    """
    Bollinger Band mean reversion signal — tightened to reduce false positives.

    Requires ALL of:
      1. Close outside the band (not just touching it)
      2. Previous candle also outside (confirms sustained extreme, not a spike)
      3. RSI confirmation (oversold < 35 for long, overbought > 65 for short)

    Returns: 'long', 'short', or None
    """
    try:
        if len(df) < 22:
            return None

        bb    = BollingerBands(df['close'], window=20, window_dev=deviation)
        upper = bb.bollinger_hband()
        lower = bb.bollinger_lband()

        current_close = df['close'].iloc[-1]
        prev_close    = df['close'].iloc[-2]
        current_upper = upper.iloc[-1]
        current_lower = lower.iloc[-1]
        prev_upper    = upper.iloc[-2]
        prev_lower    = lower.iloc[-2]

        # RSI confirmation
        rsi = RSIIndicator(df['close'], window=14).rsi()
        current_rsi = rsi.iloc[-1]
        if pd.isna(current_rsi):
            return None

        # LONG: close below lower band, prev candle also below, RSI oversold
        if (current_close < current_lower and
                prev_close < prev_lower and
                current_rsi < 35):
            return 'long'

        # SHORT: close above upper band, prev candle also above, RSI overbought
        elif (current_close > current_upper and
              prev_close > prev_upper and
              current_rsi > 65):
            return 'short'

        return None

    except Exception as e:
        logger.error(f"Error in bollinger_band_signal: {e}")
        return None

def breakout_signal(df, lookback=20, volume_confirmation=True):
    """
    Donchian breakout signal - MORE SENSITIVE
    Returns: 'long', 'short', or None
    """
    try:
        # Calculate Donchian channel
        highest_high = df['high'].rolling(lookback).max()
        lowest_low = df['low'].rolling(lookback).min()
        
        # Volume SMA
        volume_sma = df['volume'].rolling(20).mean()
        
        # Get current and previous values
        current_close = df['close'].iloc[-1]
        prev_close = df['close'].iloc[-2]
        prev_high = highest_high.iloc[-2]
        prev_low = lowest_low.iloc[-2]
        current_volume = df['volume'].iloc[-1]
        
        # LONG SIGNAL: Price breaks above previous period's high
        long_condition = current_close > prev_high
        
        # Add volume confirmation if requested
        if volume_confirmation:
            long_condition = long_condition and (current_volume > volume_sma.iloc[-2])
        
        # SHORT SIGNAL: Price breaks below previous period's low
        short_condition = current_close < prev_low
        
        if volume_confirmation:
            short_condition = short_condition and (current_volume > volume_sma.iloc[-2])
        
        if long_condition:
            return 'long'
        elif short_condition:
            return 'short'
        else:
            return None
            
    except Exception as e:
        logger.error(f"Error in breakout_signal: {e}")
        return None

def volume_breakout_signal(df):
    """
    Volume breakout signal
    Returns: 'long', 'short', or None
    """
    try:
        volume_sma = df['volume'].rolling(20).mean()
        price_change = df['close'].pct_change()
        
        current_volume = df['volume'].iloc[-1]
        current_volume_sma = volume_sma.iloc[-1]
        current_price_change = price_change.iloc[-1]
        
        # High volume + price up
        if current_volume > current_volume_sma * 1.5 and current_price_change > 0.01:
            return 'long'
        # High volume + price down
        elif current_volume > current_volume_sma * 1.5 and current_price_change < -0.01:
            return 'short'
        else:
            return None
            
    except Exception as e:
        logger.error(f"Error in volume_breakout_signal: {e}")
        return None

# -------------------------------------------------------------------
# Higher timeframe trend helper (cached)
# -------------------------------------------------------------------
_daily_trend_cache = {}

def get_daily_trend(symbol: str) -> str:
    """Return 'up', 'down', or 'side' based on daily EMA and Price location."""
    now = time.time()
    # cache for 1 hour
    if symbol in _daily_trend_cache:
        cached_time, cached_trend = _daily_trend_cache[symbol]
        if now - cached_time < 3600:
            return cached_trend
    try:
        # Increased limit to 200 to ensure stable EMA calculations
        df_daily = fetch_ohlcv(symbol, interval='1d', limit=200)
        if df_daily.empty or len(df_daily) < 50:
            return "side"

        current_price = df_daily['close'].iloc[-1]
        ema20 = df_daily['close'].ewm(span=20, adjust=False).mean()
        ema50 = df_daily['close'].ewm(span=50, adjust=False).mean()
        
        last_ema20 = ema20.iloc[-1]
        last_ema50 = ema50.iloc[-1]

        # --- RECOVERY LOGIC ---
        # 1. Standard Crossover (Long term strength)
        if last_ema20 > last_ema50:
            trend = "up"
        # 2. Price Lead (Early recovery)
        # If price is > EMA20 and EMA20 is sloping UP, 
        # we allow 'up' signals even if the 50 is still overhead.
        elif current_price > last_ema20 and last_ema20 > ema20.iloc[-2]:
            trend = "up"
        else:
            trend = "down"

        _daily_trend_cache[symbol] = (now, trend)
        return trend
        
    except Exception as e:
        logger.debug(f"Daily trend error for {symbol}: {e}")
        return "side"

_4H_trend_cache = {}
def get_4H_trend(symbol: str) -> str:
    """Return 'up', 'down', or 'side' based on 4-hour EMA cross."""
    now = time.time()
    # cache for 1 hour
    if symbol in _4H_trend_cache:
        cached_time, cached_trend = _4H_trend_cache[symbol]
        if now - cached_time < 3600:
            return cached_trend
    try:
        df_4h = fetch_ohlcv(symbol, interval='4h', limit=100)
        if df_4h.empty or len(df_4h) < 50:
            return "side"
        ema20 = df_4h['close'].ewm(span=20).mean()
        ema50 = df_4h['close'].ewm(span=50).mean()
        trend = "up" if ema20.iloc[-1] > ema50.iloc[-1] else "down"
        _4H_trend_cache[symbol] = (now, trend)
        return trend
    except Exception as e:
        logger.debug(f"4H trend error for {symbol}: {e}")
        return "side"