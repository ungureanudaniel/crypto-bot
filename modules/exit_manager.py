"""
exit_manager.py
===============
Indicator-based and hybrid Chandelier exit logic.

Used by both trade_engine (spot longs) and futures_engine (shorts).

Three layers:
  Layer 1 — Signal reversal: re-checks the entry indicator each cycle.
  Layer 2 — MACD Bar Exhaustion: exit when opposite bar exhaustion pattern appears.
  Layer 3 — Hybrid Chandelier trailing stop: dynamic stop that trails price.
"""

import logging
import pandas as pd
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# ATR helper
# -------------------------------------------------------------------
def _calculate_atr(df: pd.DataFrame, length: int = 14) -> float:
    """Return the most recent ATR value from a DataFrame."""
    try:
        import numpy as np
        high_low   = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift(1)).abs()
        low_close  = (df['low']  - df['close'].shift(1)).abs()
        tr  = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(length, min_periods=1).mean()
        val = atr.iloc[-1]
        return float(val) if not pd.isna(val) else 0.0
    except Exception:
        return 0.0


# -------------------------------------------------------------------
# MACD Bar Exhaustion Exit Check
# -------------------------------------------------------------------
def check_macd_bar_exhaustion_exit(df: pd.DataFrame, position: dict) -> Tuple[bool, str]:
    """
    Exit when the opposite MACD bar exhaustion pattern appears.
    
    For LONG positions: exit when SHORT bar exhaustion is detected
    For SHORT positions: exit when LONG bar exhaustion is detected
    
    Requires:
      1. Position has been held for at least 4 candles (minimum hold)
      2. MACD bar exhaustion signal in opposite direction
    """
    try:
        from modules.strategy_tools import macd_bar_exhaustion_signal
        
        side = position.get('side', 'long')
        candles_held = position.get('candles_held', 0)
        
        # Minimum hold period before considering MACD exit (4 candles = 4 hours)
        if candles_held < 4:
            return False, ''
        
        # Get MACD bar exhaustion signal
        signal = macd_bar_exhaustion_signal(df, use_rsi_confirm=False)
        
        if not signal:
            return False, ''
        
        # For long positions, exit on short signal
        if side == 'long' and signal == 'short':
            logger.info(f"🔴 MACD Bar Exhaustion EXIT LONG: opposite signal detected after {candles_held} candles")
            return True, 'macd_bar_exhaustion'
        
        # For short positions, exit on long signal
        if side == 'short' and signal == 'long':
            logger.info(f"🟢 MACD Bar Exhaustion EXIT SHORT: opposite signal detected after {candles_held} candles")
            return True, 'macd_bar_exhaustion'
        
        return False, ''
        
    except Exception as e:
        logger.debug(f"MACD bar exhaustion exit check failed: {e}")
        return False, ''


# -------------------------------------------------------------------
# Layer 1 — Signal reversal detection (original)
# -------------------------------------------------------------------
def check_signal_reversal(df: pd.DataFrame, position: dict) -> bool:
    """
    MODIFIED: Exit immediately when indicators flip, 
    without waiting for price to move against us.
    """
    signal_type = position.get('signal_type', '')
    side        = position.get('side', 'long')
    opposite    = 'short' if side == 'long' else 'long'

    try:
        from modules.strategy_tools import (
            rsi_signal, bollinger_band_signal, breakout_signal, 
            volume_breakout_signal, macd_bar_exhaustion_signal
        )

        if len(df) < 50: return False

        # --- REMOVED: The requirement for price to move against us ---
        # I want to exit on the 'FLIP', even if the price is still neutral.

        reversal_signal = None
        
        # Check the specific indicator that got us in
        if 'macd_bar_exhaustion' in signal_type:
            # For MACD, we check for a standard trend reversal
            reversal_signal = macd_bar_exhaustion_signal(df)
        elif 'rsi' in signal_type:
            reversal_signal = rsi_signal(df)
        elif 'bollinger' in signal_type:
            reversal_signal = bollinger_band_signal(df)
        elif 'breakout' in signal_type:
            reversal_signal = breakout_signal(df)
        
        # EXIT LOGIC: If the indicator generates the OPPOSITE signal, FLIP.
        if reversal_signal == opposite:
            logger.info(f"🔄 TREND FLIP: Indicator {signal_type} reversed to {opposite}. Exiting.")
            return True

        return False

    except Exception as e:
        logger.debug(f"Signal reversal check failed: {e}")
        return False

# -------------------------------------------------------------------
# Layer 2 — Chandelier Exit (Trailing Stop)
# -------------------------------------------------------------------
def update_chandelier_stop(
    position: dict,
    current_price: float,
    atr: float,
) -> Tuple[Optional[float], str]:
    """
    Hybrid trailing stop:
    - Uses peak/trough anchoring from Chandelier concept
    - Uses percentage-based distance (not ATR multiplier) for consistency
    - Adapts distance to asset volatility via ATR% with per‑pair min/max
    - Falls back to global trailing_stop_min_pct / trailing_stop_max_pct

    Long:  stop = peak_since_entry × (1 - trail_pct)
    Short: stop = trough_since_entry × (1 + trail_pct)

    Breakeven floor at 1.5% profit.
    Stop only moves in profitable direction (ratchet).
    """
    from config_loader import config

    if current_price <= 0:
        return None, ''

    entry_price  = position.get('entry_price', current_price)
    current_stop = position.get('stop_loss', 0.0)
    side         = position.get('side', 'long')

    # Adaptive trail: ATR% with per‑pair bounds
    atr_pct = (atr / entry_price) if (atr > 0 and entry_price > 0) else 0.02

    # Use per‑pair trailing bounds if available, else global
    trail_min = position.get('trailing_min_pct')
    trail_max = position.get('trailing_max_pct')
    if trail_min is None or trail_max is None:
        trail_min = config.config.get('trailing_stop_min_pct', 0.02)
        trail_max = config.config.get('trailing_stop_max_pct', 0.04)

    trail_pct = min(max(atr_pct, trail_min), trail_max)

    breakeven_threshold = 0.015  # 1.5% profit → breakeven floor activates

    if side == 'long':
        peak = max(position.get('peak_price', entry_price), current_price)
        position['peak_price'] = peak

        profit_pct      = (current_price - entry_price) / entry_price
        chandelier_stop = peak * (1 - trail_pct)

        if profit_pct >= breakeven_threshold:
            chandelier_stop = max(chandelier_stop, entry_price)

        if chandelier_stop > current_stop:
            is_be  = (abs(chandelier_stop - entry_price) < 1e-6)
            reason = 'breakeven' if is_be else 'trailing'
            logger.info(
                f"📈 Trail LONG: peak=${peak:.4f} stop ${current_stop:.4f}"
                f" → ${chandelier_stop:.4f} (trail {trail_pct:.2%})"
            )
            position['trailing_stop_active'] = True
            return chandelier_stop, reason

    else:  # short
        trough = min(position.get('trough_price', entry_price), current_price)
        position['trough_price'] = trough

        profit_pct      = (entry_price - current_price) / entry_price
        chandelier_stop = trough * (1 + trail_pct)

        if profit_pct >= breakeven_threshold:
            chandelier_stop = min(chandelier_stop, entry_price)

        if chandelier_stop < current_stop:
            is_be  = (abs(chandelier_stop - entry_price) < 1e-6)
            reason = 'breakeven' if is_be else 'trailing'
            logger.info(
                f"📉 Trail SHORT: trough=${trough:.4f} stop ${current_stop:.4f}"
                f" → ${chandelier_stop:.4f} (trail {trail_pct:.2%})"
            )
            position['trailing_stop_active'] = True
            return chandelier_stop, reason

    return None, ''
# -------------------------------------------------------------------
def calculate_fibonacci_target(position: dict, current_price: float, entry_price: float, side: str) -> Optional[float]:
    """
    Calculate 1.272 Fibonacci extension target from entry to peak/trough.
    
    For long: target = entry + (peak - entry) * 1.272
    For short: target = entry - (entry - trough) * 1.272
    """
    if side == 'long':
        peak = position.get('peak_price', entry_price)
        if peak <= entry_price:
            return None
        move = peak - entry_price
        target = entry_price + move * 1.272
        return target
    else:  # short
        trough = position.get('trough_price', entry_price)
        if trough >= entry_price:
            return None
        move = entry_price - trough
        target = entry_price - move * 1.272
        return target

# -------------------------------------------------------------------
# Combined exit check — call once per position per cycle
# -------------------------------------------------------------------
def evaluate_exit(
    symbol: str,
    position: dict,
    current_price: float,
    df: Optional[pd.DataFrame] = None,
) -> Tuple[bool, str]:
    """
    Run all exit layers for a position.

    Returns:
      (True, reason)  — exit now
      (False, '')     — hold; position dict may be mutated

    Order:
      0. Track candles held
      1. Recalculate ATR
      2. Update Chandelier stop
      3. Check if stop or TP was hit
      4. Check MACD bar exhaustion exit (NEW)
      5. Check signal reversal (after 6 candles)
    """
    if current_price <= 0:
        return False, ''

    # Always recalculate ATR from current window for accurate Chandelier anchoring
    if df is not None and not df.empty:
        atr = _calculate_atr(df)
    else:
        atr = position.get('atr', 0.0)

    # --- Candle counter: increment if a new candle has formed ---
    if df is not None and not df.empty:
        if 'timestamp' in df.columns:
            current_candle_time = df['timestamp'].iloc[-1]
        else:
            current_candle_time = len(df) - 1
        
        last_candle = position.get('last_candle_time')
        if last_candle is None or current_candle_time > last_candle:
            position['candles_held'] = position.get('candles_held', 0) + 1
            position['last_candle_time'] = current_candle_time

    # Attach symbol to position for logging
    position.setdefault('symbol', symbol)

    # --- Layer 3: Chandelier stop update ---
    if atr > 0:
        new_stop, _ = update_chandelier_stop(position, current_price, atr)
        if new_stop is not None:
            position['stop_loss'] = new_stop

    # --- Check if stop or TP was hit ---
    side = position.get('side', 'long')
    stop = position.get('stop_loss', 0.0)
    tp   = position.get('take_profit', 0.0)

    if side == 'long':
        if stop and current_price <= stop:
            reason = 'trailing_stop' if position.get('trailing_stop_active') else 'stop_loss'
            return True, reason
        if tp and current_price >= tp:
            return True, 'take_profit'
    else:  # short
        if stop and current_price >= stop:
            reason = 'trailing_stop' if position.get('trailing_stop_active') else 'stop_loss'
            return True, reason
        if tp and current_price <= tp:
            return True, 'take_profit'

    # --- Layer 2 — MACD Bar Exhaustion exit (after 4 candles minimum) ---
    if df is not None and not df.empty:
        should_exit, reason = check_macd_bar_exhaustion_exit(df, position)
        if should_exit:
            return True, reason

    # --- Layer 1: signal reversal (min 3 candles) ---
    if df is not None and not df.empty:
        candles_held = position.get('candles_held', 0)
        if candles_held >= 3: # Wait at least 3 candles before checking for signal reversal
            if check_signal_reversal(df, position):
                return True, 'signal_reversal'

    return False, ''