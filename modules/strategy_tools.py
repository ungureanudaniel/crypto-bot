import sys
import os
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
import logging
import time
from typing import Dict, Optional
from ta.volatility import BollingerBands
from ta.trend import MACD, ADXIndicator
from ta.momentum import RSIIndicator
from modules.data_feed import fetch_ohlcv
from config_loader import config as _cfg

logger = logging.getLogger(__name__)
ml_strategies = {}
sentiment_agent = None
use_volume_shrinkage = _cfg.config.get('use_volume_shrinkage', False)
# IMPROVEMENT: allow enabling daily trend filter via config
use_daily_trend_filter = _cfg.config.get('use_daily_trend_filter', True)

# -------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------
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

def get_available_balance(symbol, trading_engine):
    """
    Get available balance for position sizing.
    For shorts (futures): returns available USDT cash margin.
    For longs (spot): returns available USDT cash.
    """
    if not trading_engine:
        return 0

    # PAPER MODE: use portfolio cash
    if trading_engine.trading_mode == 'paper':
        try:
            from modules.portfolio import get_cash
            cash = get_cash()
            return cash.get('USDT', 0) + cash.get('USDC', 0)
        except Exception as e:
            logger.debug(f"Could not get paper cash balance: {e}")
            return 0

    # LIVE/TESTNET MODE: Check real exchange balance
    if not trading_engine.binance_client:
        return 0

    try:
        account = trading_engine.binance_client.get_account()
        for balance in account['balances']:
            if balance['asset'] == 'USDT':
                return float(balance['free'])
    except Exception as e:
        logger.debug(f"Could not get live balance: {e}")

    return 0

# -------------------------------------------------------------------
# Signal Generation Functions
# -------------------------------------------------------------------

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
                logger.debug(f"📈 MACD Bar Exhaustion LONG: {neg_streak.iloc[-1]} red bars, shrinking, stable, final drop, volume shrinking")
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
                logger.debug(f"📉 MACD Bar Exhaustion SHORT: {pos_streak.iloc[-1]} green bars, shrinking, stable, final pump, volume shrinking")
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
# Position Sizing (Only calculates, but execution in trade_engine.py)
# -------------------------------------------------------------------
def calculate_position_units(entry_price, equity, risk_per_trade=0.02, atr=None,
                              stop_atr_multiplier: float = 2.0, trading_fee: float = 0.0005,
                              side: str = 'long',
                              stop_loss_min_pct: Optional[float] = None,
                              stop_loss_max_pct: Optional[float] = None,
                              default_stop_loss_pct: Optional[float] = None,
                              min_rr: Optional[float] = None):
    try:
        from config_loader import config as cfg

        # Load config with fallbacks
        min_pct = stop_loss_min_pct if stop_loss_min_pct is not None else cfg.config.get('stop_loss_min_pct', 0.015)
        max_pct = stop_loss_max_pct if stop_loss_max_pct is not None else cfg.config.get('stop_loss_max_pct', 0.08)
        def_pct = default_stop_loss_pct if default_stop_loss_pct is not None else cfg.config.get('default_stop_loss_pct', 0.03)
        rr = min_rr if min_rr is not None else cfg.config.get('min_rr', 99.0)

        # Calculate stop loss distance from ATR
        if atr and atr > 0:
            stop_distance = atr * stop_atr_multiplier
            stop_loss_pct = stop_distance / entry_price
        else:
            stop_loss_pct = def_pct

        # Apply bounds
        stop_loss_pct = max(stop_loss_pct, min_pct)
        stop_loss_pct = min(stop_loss_pct, max_pct)

        # Fee cost for round trip
        round_trip_fee = trading_fee * 2

        # TP: risk:reward × stop
        take_profit_pct = max(stop_loss_pct * rr, round_trip_fee * 5)
        take_profit_pct = max(take_profit_pct, 0.02)

        # Direction‑aware SL/TP
        if side == 'short':
            stop_loss_price   = entry_price * (1 + stop_loss_pct)
            take_profit_price = entry_price * (1 - take_profit_pct)
        else:
            stop_loss_price   = entry_price * (1 - stop_loss_pct)
            take_profit_price = entry_price * (1 + take_profit_pct)

        # Fee-adjusted net profit check — skip if fees eat >20% of gain
        risk_amount   = equity * risk_per_trade
        risk_per_unit = entry_price * stop_loss_pct
        if risk_per_unit <= 0:
            logger.warning("⚠️ Risk per unit is zero or negative")
            return 0, None, None
        units_estimate  = min(risk_amount / risk_per_unit, (equity * 0.15) / entry_price)
        position_value  = units_estimate * entry_price
        actual_fee_cost = position_value * round_trip_fee
        expected_gain   = position_value * stop_loss_pct * rr
        if actual_fee_cost > expected_gain * 0.20:
            logger.debug(f"⏭️ Fee ratio too high: fees ${actual_fee_cost:.4f} vs gain ${expected_gain:.4f}")
            return 0, None, None

        # Position sizing
        units = risk_amount / risk_per_unit

        # Cap at 15% of equity per position
        max_units = (equity * 0.15) / entry_price
        units = min(units, max_units)

        # Minimum trade size ($10)
        min_units = 10 / entry_price
        if units < min_units:
            logger.debug(f"Position too small: {units:.6f}, required: {min_units:.6f}")
            return 0, None, None

        logger.info(
            f"Position calc [{side.upper()}]: Entry=${entry_price:.4f}, "
            f"SL=${stop_loss_price:.4f} ({stop_loss_pct:.2%}), "
            f"TP=${take_profit_price:.4f} ({take_profit_pct:.2%}), "
            f"R:R={take_profit_pct/stop_loss_pct:.1f}x, "
            f"Units={units:.6f}, Risk=${risk_amount:.2f}"
        )

        return units, stop_loss_price, take_profit_price

    except Exception as e:
        logger.error(f"Error in calculate_position_units: {e}")
        return 0, None, None

# -------------------------------------------------------------------
# Higher timeframe trend helper (cached)
# -------------------------------------------------------------------
_daily_trend_cache = {}
def get_daily_trend(symbol: str) -> str:
    """Return 'up', 'down', or 'side' based on daily EMA cross."""
    now = time.time()
    # cache for 1 hour
    if symbol in _daily_trend_cache:
        cached_time, cached_trend = _daily_trend_cache[symbol]
        if now - cached_time < 3600:
            return cached_trend
    try:
        df_daily = fetch_ohlcv(symbol, interval='1d', limit=100)
        if df_daily.empty or len(df_daily) < 50:
            return "side"
        ema20 = df_daily['close'].ewm(span=20).mean()
        ema50 = df_daily['close'].ewm(span=50).mean()
        trend = "up" if ema20.iloc[-1] > ema50.iloc[-1] else "down"
        _daily_trend_cache[symbol] = (now, trend)
        return trend
    except Exception as e:
        logger.debug(f"Daily trend error for {symbol}: {e}")
        return "side"

# ===========================================================================
# Main signal Generator
# ===========================================================================
def generate_trade_signal(df, equity, risk_per_trade=0.02, symbol=None, trading_engine=None, regime=None):
    """
    Main function that combines multiple strategies.
    Enhanced with trend detection to filter counter‑trend signals.
    
    REPLACED: Standard MACD crossover with MACD Bar Exhaustion strategy
    IMPROVEMENTS: stricter parameters, ADX filter, daily trend filter
    """
    try:
        if df.empty or len(df) < 50:
            logger.debug("Insufficient data")
            return None

        # Load fee from config
        try:
            from config_loader import config as _cfg
            trading_fee = float(_cfg.config.get('trading_fee', 0.0005))
        except Exception:
            trading_fee = 0.0005

        # --- Trend detection ---
        def detect_trend(df):
            try:
                ema20  = df['close'].ewm(span=20).mean()
                ema50  = df['close'].ewm(span=50).mean()
                direction = ("up" if ema20.iloc[-1] > ema50.iloc[-1]
                             else "down" if ema20.iloc[-1] < ema50.iloc[-1]
                             else "side")
                adx      = ADXIndicator(df['high'], df['low'], df['close'], window=14).adx()
                strength = float(adx.iloc[-1]) / 100.0
                ema200   = df['close'].ewm(span=min(200, len(df)-1)).mean()
                long_dir = "up" if df['close'].iloc[-1] > ema200.iloc[-1] else "down"
                confidence = 0.7 if direction == long_dir else 0.5
                if strength > 0.3:
                    confidence += 0.2
                return direction, min(strength, 1.0), min(confidence, 0.95)
            except Exception as e:
                logger.debug(f"Trend detection error: {e}")
                return "side", 0.0, 0.5

        trend_dir, trend_strength, trend_conf = detect_trend(df)
        
        # Adjust risk based on trend strength
        if trend_strength > 0.4:
            risk_multiplier = 1.2
        elif trend_strength > 0.25:
            risk_multiplier = 1.0
        else:
            risk_multiplier = 0.6
        
        pair_config = {}
        if symbol:
            try:
                from config_loader import get_pair_config
                pair_config = get_pair_config(symbol)
            except Exception as e:
                logger.debug(f"Could not load pair config for {symbol}: {e}")

        # Override risk if per‑pair value exists
        per_pair_risk = pair_config.get('risk_per_trade')
        if per_pair_risk is not None:
            # Apply same trend multiplier to the per‑pair risk
            adjusted_risk = per_pair_risk * risk_multiplier
            logger.info(f"📌 Using per‑pair risk for {symbol}: {per_pair_risk:.2%} (global {risk_per_trade:.2%})")
            logger.debug(f"Risk multiplier: {risk_multiplier:.2f}, Adjusted risk: {adjusted_risk:.2%}")
        else:
            adjusted_risk = risk_per_trade * risk_multiplier
            logger.debug(f"Risk multiplier: {risk_multiplier:.2f}, Adjusted risk: {adjusted_risk:.2%}")
        logger.debug(f"Trend: {trend_dir} | strength={trend_strength:.2f}")

        # ATR for position sizing
        atr_series = calculate_atr(df)
        current_atr = atr_series.iloc[-1] if not atr_series.empty else 0
        current_price = df['close'].iloc[-1]

        # Parse regime
        regime_type = "unknown"
        trend_direction = "unknown"
        if regime:
            if "UPTREND" in regime:
                trend_direction = "up"
                regime = regime.replace(" UPTREND", "")
            elif "DOWNTREND" in regime:
                trend_direction = "down"
                regime = regime.replace(" DOWNTREND", "")
            elif "SIDEWAYS" in regime:
                trend_direction = "side"
                regime = regime.replace(" SIDEWAYS", "")

            if "Range" in regime:
                regime_type = "range"
            elif "Compression" in regime:
                regime_type = "compression"
            elif "Expansion" in regime:
                regime_type = "expansion"
            elif "Breakout" in regime:
                regime_type = "breakout"
            elif "Trend" in regime or "Trending" in regime:
                regime_type = "trend"

        # --- Strategy selection with trend filtering ---
        signal = None
        signal_type = None
        multiplier = 2.0
        
        # ===== TRENDING MARKET =====
        if regime_type == "trend":
            logger.debug(f"📈 Trending market - following {trend_direction} trend")
            
            # Single call with all improvements: min_bars=4, RSI confirm, volume shrink
            signal = macd_bar_exhaustion_signal(df, 
                                                min_bars=4, 
                                                min_shrink_bars=3,
                                                use_rsi_confirm=True, 
                                                rsi_oversold=30, 
                                                rsi_overbought=70,
                                                use_volume_shrink=use_volume_shrinkage)
            
            if trend_direction == "up":
                if signal == 'long':
                    signal_type = "macd_bar_exhaustion_trend_up"
                    multiplier = 2.5
                else:
                    # fallback to breakout if no MACD signal
                    signal = breakout_signal(df)
                    if signal == 'long':
                        signal_type = "breakout_trend_up"
                        multiplier = 2.5
            elif trend_direction == "down":
                if signal == 'short':
                    signal_type = "macd_bar_exhaustion_trend_down"
                    multiplier = 2.5
                else:
                    signal = breakout_signal(df)
                    if signal == 'short':
                        signal_type = "breakout_trend_down"
                        multiplier = 2.5
            else:
                signal = rsi_signal(df)
                if signal:
                    signal_type = f"rsi_trend_side_{signal}"
                    multiplier = 1.5
                if not signal:
                    signal = bollinger_band_signal(df)
                    if signal:
                        signal_type = f"bollinger_trend_side_{signal}"
                        multiplier = 1.5

        # ===== RANGING MARKET =====
        elif regime_type in ["range", "compression"]:
            logger.debug(f"📊 Ranging market – favoring mean reversion")
            
            # Try MACD bar exhaustion first (works well in ranges)
            signal = macd_bar_exhaustion_signal(df, 
                                                min_bars=4, 
                                                min_shrink_bars=3,
                                                use_rsi_confirm=False, 
                                                use_volume_shrink=True)
            if signal:
                signal_type = f"macd_bar_exhaustion_range_{signal}"
                multiplier = 2.0
                logger.debug(f"✅ Range MACD Bar Exhaustion {signal}")
            else:
                # Fallback to RSI and Bollinger
                signal = rsi_signal(df)
                if signal:
                    signal_dir = 'up' if signal == 'long' else 'down'
                    if trend_strength < 0.25 or signal_dir == trend_dir or trend_dir == 'side':
                        signal_type = f"rsi_range_{signal}"
                        multiplier = 1.5
                        logger.debug(f"✅ Range RSI {signal}")
                    else:
                        signal = None
                if not signal:
                    signal = bollinger_band_signal(df)
                    if signal:
                        signal_dir = 'up' if signal == 'long' else 'down'
                        if trend_strength < 0.25 or signal_dir == trend_dir or trend_dir == 'side':
                            signal_type = f"bollinger_range_{signal}"
                            multiplier = 1.5
                            logger.debug(f"✅ Range Bollinger {signal}")
                        else:
                            signal = None

        # ===== BREAKOUT MARKET =====
        elif regime_type == "breakout":
            logger.debug(f"🚀 Breakout market")
            
            # Try MACD bar exhaustion first (can signal pullback entries)
            signal = macd_bar_exhaustion_signal(df, 
                                                min_bars=4, 
                                                min_shrink_bars=3,
                                                use_rsi_confirm=False, 
                                                use_volume_shrink=True)
            if signal:
                signal_dir = 'up' if signal == 'long' else 'down'
                if signal_dir == trend_dir or trend_strength < 0.2:
                    signal_type = f"macd_bar_exhaustion_breakout_{signal}"
                    multiplier = 2.5
                    logger.debug(f"✅ Breakout MACD Bar Exhaustion {signal}")
                else:
                    signal = None
            if not signal:
                signal = breakout_signal(df)
                if signal:
                    signal_dir = 'up' if signal == 'long' else 'down'
                    if signal_dir == trend_dir or trend_strength < 0.2:
                        signal_type = f"breakout_{signal}"
                        multiplier = 3.0
                        logger.debug(f"✅ Breakout {signal}")
                    else:
                        signal = None
            if not signal:
                signal = volume_breakout_signal(df)
                if signal:
                    signal_dir = 'up' if signal == 'long' else 'down'
                    if signal_dir == trend_dir or trend_strength < 0.2:
                        signal_type = f"volume_breakout_{signal}"
                        multiplier = 2.5
                        logger.debug(f"✅ Volume breakout {signal}")
                    else:
                        signal = None

        # ===== EXPANSION MARKET =====
        elif regime_type == "expansion":
            logger.debug(f"🌪️ High volatility – cautious")
            adjusted_risk = risk_per_trade * 0.5
            
            # MACD bar exhaustion may be less reliable in high volatility
            signal = macd_bar_exhaustion_signal(df, 
                                                min_bars=4, 
                                                min_shrink_bars=3,
                                                use_rsi_confirm=True, 
                                                use_volume_shrink=True)
            if signal:
                signal_dir = 'up' if signal == 'long' else 'down'
                if signal_dir == trend_dir:
                    signal_type = f"macd_bar_exhaustion_expansion_{signal}"
                    multiplier = 2.0
                    logger.debug(f"✅ Expansion MACD Bar Exhaustion {signal}")
                else:
                    signal = None
            if not signal:
                signal = breakout_signal(df, volume_confirmation=True)
                if signal:
                    signal_dir = 'up' if signal == 'long' else 'down'
                    if signal_dir == trend_dir:
                        signal_type = f"breakout_expansion_{signal}"
                        multiplier = 3.0
                        logger.debug(f"✅ Breakout in high volatility with trend")
                    else:
                        signal = None
            if not signal:
                signal = volume_breakout_signal(df)
                if signal and abs(df['close'].pct_change().iloc[-1]) > 0.02:
                    signal_dir = 'up' if signal == 'long' else 'down'
                    if signal_dir == trend_dir:
                        signal_type = f"volume_expansion_{signal}"
                        multiplier = 2.5
                        logger.debug(f"✅ Volume breakout in high volatility")
                    else:
                        signal = None

        # ===== UNKNOWN REGIME =====
        else:
            logger.debug(f"❓ Unknown regime – trying all strategies")
            # Prioritize MACD bar exhaustion
            for fn, stype, mult in [
                (lambda d: macd_bar_exhaustion_signal(d, min_bars=4, min_shrink_bars=3, use_volume_shrink=True), "macd_bar_exhaustion_unknown", 2.0),
                (breakout_signal, "breakout_unknown", 2.0),
                (rsi_signal, "rsi_unknown", 1.5),
                (bollinger_band_signal, "bollinger_unknown", 1.5),
                (volume_breakout_signal, "volume_unknown", 2.0),
            ]:
                sig = fn(df)
                if sig:
                    signal_dir = 'up' if sig == 'long' else 'down'
                    if trend_strength > 0.25 and signal_dir != trend_dir and trend_dir != 'side':
                        logger.debug(f"⏭️ Skipping {stype} - counter-trend")
                        continue
                    signal = sig
                    signal_type = stype
                    multiplier = mult
                    break

        # ===== FINAL TREND VALIDATION =====
        if signal:
            signal_dir = 'up' if signal == 'long' else 'down'
            
            # Strong trend: NEVER trade against it
            if trend_strength > 0.25 and signal_dir != trend_dir and trend_dir != 'side':
                logger.warning(f"❌ REJECTED: {signal} signal against {trend_dir} trend (ADX={trend_strength*100:.0f})")
                return None
            
            # --- Daily Trend Filter ---
            if use_daily_trend_filter and symbol:
                daily_trend = get_daily_trend(symbol)
                if daily_trend != "side" and signal_dir != daily_trend:
                    logger.warning(f"❌ REJECTED: {signal} signal against DAILY {daily_trend} trend")
                    return None

            # --- Position Sizing and Output ---
            # Use real-time balance if available
            available_balance = get_available_balance(symbol, trading_engine) or equity
            
            units, sl_price, tp_price = calculate_position_units(
                entry_price=current_price,
                equity=available_balance,
                risk_per_trade=adjusted_risk,
                atr=current_atr,
                stop_atr_multiplier=multiplier, # Uses the multiplier set in the regime logic
                side=signal
            )

            if units > 0:
                return {
                    'symbol': symbol,
                    'signal': signal,
                    'signal_type': signal_type,
                    'units': units,
                    'entry': current_price,
                    'stop_loss': sl_price,
                    'take_profit': tp_price,
                    'risk_pct': adjusted_risk,
                    'regime': regime_type
                }
                logger.debug(f"📊 Signal: {signal} | confidence: {confidence}% | trend: {trend_dir} ({trend_strength*100:.0f} ADX)")
                return result

        logger.debug("No signals detected")
        return None

    except Exception as e:
        logger.error(f"Error in generate_trade_signal: {e}")
        return None

def get_confidence_from_regime(regime_type, signal_type):
    """Calculate confidence based on regime and signal type"""
    # Base confidence
    confidence = 70
    
    # Boost confidence when strategy matches regime
    if regime_type == "range" and "rsi" in signal_type:
        confidence += 20
    elif regime_type == "range" and "bollinger" in signal_type:
        confidence += 15
    elif regime_type == "trend" and ("ema" in signal_type or "macd" in signal_type):
        confidence += 20
    elif regime_type == "breakout" and "breakout" in signal_type:
        confidence += 25
    elif regime_type == "expansion" and "breakout" in signal_type:
        confidence += 10  # Still good but risky
    
    # Cap at 95
    return min(confidence, 95)
    
if __name__ == "__main__":
    import sys
    import os
    from datetime import datetime, timedelta
    import time
    
    print("=" * 60)
    print("🧪 STRATEGY TOOLS COMPREHENSIVE TEST")
    print("=" * 60)
    
    # Test configuration
    test_symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    test_timeframes = ["1h", "4h"]
    test_equity = 10000
    
    # Create a mock trading engine for balance tests
    class MockTradingEngine:
        def __init__(self):
            self.trading_mode = "paper"
            self.binance_client = None
    
    mock_engine = MockTradingEngine()
    
    print(f"\n📊 Test Parameters:")
    print(f"   Symbols: {test_symbols}")
    print(f"   Timeframes: {test_timeframes}")
    print(f"   Equity: ${test_equity}")
    print(f"   Mode: {mock_engine.trading_mode}")
    
    # Test 1: Individual strategy functions
    print("\n" + "=" * 60)
    print("📈 TEST 1: Individual Strategy Functions")
    print("=" * 60)
    
    for symbol in test_symbols:
        print(f"\n🔍 Testing {symbol}...")
        
        for interval in test_timeframes:
            print(f"   ⏱️  Timeframe: {interval}")
            
            # Fetch data
            from modules.data_feed import fetch_ohlcv
            df = fetch_ohlcv(symbol, interval=interval, limit=200)
            
            if df.empty or len(df) < 50:
                print(f"      ❌ Insufficient data ({len(df)} candles)")
                continue
            
            print(f"      ✅ Data fetched: {len(df)} candles")
            print(f"      Latest price: ${df['close'].iloc[-1]:.2f}")
            
            # Test each strategy
            strategies = [
                ("Breakout", lambda: breakout_signal(df)),
                ("RSI", lambda: rsi_signal(df)),
                ("MACD Bar Exhaustion", lambda: macd_bar_exhaustion_signal(df)),
                ("Bollinger", lambda: bollinger_band_signal(df)),
                ("Volume Breakout", lambda: volume_breakout_signal(df))
            ]
            
            for name, func in strategies:
                try:
                    signal = func()
                    if signal:
                        print(f"      ✅ {name}: {signal.upper()}")
                    else:
                        print(f"      ⚪ {name}: No signal")
                except Exception as e:
                    print(f"      ❌ {name} error: {e}")
            
            # Small delay to avoid rate limiting
            time.sleep(0.5)
    
    # Test 2: Position sizing
    print("\n" + "=" * 60)
    print("💰 TEST 2: Position Sizing Calculator")
    print("=" * 60)
    
    test_prices = [50000, 3000, 100, 10, 1]
    test_atr_values = [1000, 100, 10, 1, 0.1]
    
    for i, price in enumerate(test_prices):
        atr = test_atr_values[i] if i < len(test_atr_values) else price * 0.02
        
        print(f"\n📊 Entry Price: ${price:.2f}, ATR: ${atr:.2f}")
        
        units, sl, tp = calculate_position_units(
            entry_price=price,
            equity=test_equity,
            risk_per_trade=0.02,
            atr=atr,
            stop_atr_multiplier=2
        )
        
        if units > 0 and sl is not None and tp is not None:
            print(f"   ✅ Units: {units:.6f}")
            print(f"   ✅ Stop Loss: ${sl:.2f} ({(1-sl/price)*100:.1f}% loss)")
            print(f"   ✅ Take Profit: ${tp:.2f} ({(tp/price-1)*100:.1f}% gain)")
            print(f"   ✅ Position Value: ${units * price:.2f}")
            print(f"   ✅ Risk Amount: ${test_equity * 0.02:.2f}")
        else:
            print(f"   ⚪ Position too small (min ${10/price:.6f} units)")
    
    # Test 3: Full signal generation with balance awareness
    print("\n" + "=" * 60)
    print("🎯 TEST 3: Full Signal Generation with Balance Awareness")
    print("=" * 60)
    
    # We'll use the get_available_balance function to test different scenarios
    # by monkey-patching it temporarily
    
    original_get_available_balance = get_available_balance
    
    for symbol in test_symbols:
        print(f"\n🔍 Testing {symbol}...")
        
        # Fetch data
        from modules.data_feed import fetch_ohlcv
        df = fetch_ohlcv(symbol, interval="1h", limit=200)
        
        if df.empty:
            continue
        
        # Test with different balance scenarios by mocking the get_available_balance function
        balance_scenarios = [
            ("No balance (returns 0)", 0),
            ("Has balance (100 units)", 100),
            ("Small balance (10 units)", 10),
        ]
        
        for scenario_name, mock_balance in balance_scenarios:
            print(f"\n   📊 Scenario: {scenario_name}")
            # Mock the get_available_balance function for this test
            def mock_get_balance(sym, engine, balance=mock_balance):
                return balance
            
            # Temporarily replace the function
            import modules.strategy_tools
            modules.strategy_tools.get_available_balance = mock_get_balance
            
            try:
                signal = generate_trade_signal(
                    df=df,
                    equity=test_equity,
                    risk_per_trade=0.02,
                    symbol=symbol,
                    trading_engine=mock_engine
                )
                
                if signal:
                    print(f"      ✅ Signal generated!")
                    print(f"         Type: {signal.get('signal_type', 'unknown')}")
                    print(f"         Side: {signal.get('side', 'unknown')}")
                    print(f"         Entry: ${signal.get('entry', 0):.2f}")
                    print(f"         Units: {signal.get('units', 0):.6f}")
                    print(f"         Stop: ${signal.get('stop_loss', 0):.2f}")
                    print(f"         Target: ${signal.get('take_profit', 0):.2f}")
                else:
                    print(f"      ⚪ No signal generated")
            finally:
                # Restore original function
                modules.strategy_tools.get_available_balance = original_get_available_balance
    
    # Test 4: Multiple timeframes
    print("\n" + "=" * 60)
    print("⏱️  TEST 4: Multiple Timeframe Analysis")
    print("=" * 60)
    
    symbol = "BTC/USDT"
    print(f"\n🔍 Analyzing {symbol} across timeframes...")
    
    signals_by_tf = {}
    
    for interval in test_timeframes:
        from modules.data_feed import fetch_ohlcv
        df = fetch_ohlcv(symbol, interval=interval, limit=200)
        
        if df.empty:
            continue
        
        signal = generate_trade_signal(
            df=df,
            equity=test_equity,
            risk_per_trade=0.02,
            symbol=symbol,
            trading_engine=mock_engine
        )
        
        if signal:
            signals_by_tf[interval] = signal
            print(f"\n   ⏱️  {interval}: {signal['side'].upper()} {signal['signal_type']} at ${signal['entry']:.2f}")
        else:
            print(f"\n   ⏱️  {interval}: No signal")
    
    # Check for alignment across timeframes
    if len(signals_by_tf) > 1:
        sides = [s['side'] for s in signals_by_tf.values()]
        if all(side == sides[0] for side in sides):
            print(f"\n   ✅ All timeframes agree on {sides[0].upper()}!")
        else:
            print(f"\n   ⚠️  Timeframes show mixed signals: {sides}")
    
    # Test 5: Performance metrics
    print("\n" + "=" * 60)
    print("📊 TEST 5: Signal Generation Speed")
    print("=" * 60)
    
    symbol = "BTC/USDT"
    from modules.data_feed import fetch_ohlcv
    df = fetch_ohlcv(symbol, interval="1h", limit=200)
    
    if not df.empty:
        iterations = 10  # Reduced for speed
        start_time = time.time()
        
        for i in range(iterations):
            signal = generate_trade_signal(
                df=df,
                equity=test_equity,
                risk_per_trade=0.02,
                symbol=symbol,
                trading_engine=mock_engine
            )
        
        elapsed = time.time() - start_time
        print(f"\n   ⏱️  Generated {iterations} signals in {elapsed:.2f} seconds")
        print(f"   ⚡ Average: {elapsed/iterations*1000:.2f} ms per signal")
    
    # Test 6: Edge cases
    print("\n" + "=" * 60)
    print("⚠️  TEST 6: Edge Cases")
    print("=" * 60)
    
    from modules.data_feed import fetch_ohlcv
    df = fetch_ohlcv("BTC/USDT", interval="1h", limit=200)
    
    edge_cases = [
        ("Empty DataFrame", pd.DataFrame()),
        ("Insufficient data", pd.DataFrame({'close': [1,2,3]})),
        ("Zero equity", 0),
        ("Negative equity", -1000),
        ("Zero risk", 0),
    ]
    
    for case_name, case_data in edge_cases:
        print(f"\n   📋 Testing: {case_name}")
        
        if case_name == "Empty DataFrame":
            result = generate_trade_signal(case_data, test_equity, trading_engine=mock_engine)
        elif case_name == "Insufficient data":
            result = generate_trade_signal(case_data, test_equity, trading_engine=mock_engine)
        elif case_name == "Zero equity":
            result = generate_trade_signal(df, 0, trading_engine=mock_engine)
        elif case_name == "Negative equity":
            result = generate_trade_signal(df, -1000, trading_engine=mock_engine)
        elif case_name == "Zero risk":
            result = generate_trade_signal(df, test_equity, 0, trading_engine=mock_engine)
        
        if result is None:
            print(f"      ✅ Handled correctly (returned None)")
        else:
            print(f"      ⚠️  Returned signal: {result}")
    
    # Restore original function
    modules.strategy_tools.get_available_balance = original_get_available_balance
    
    print("\n" + "=" * 60)
    print("✅ STRATEGY TOOLS TEST COMPLETE")
    print("=" * 60)