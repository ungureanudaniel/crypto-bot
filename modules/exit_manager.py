"""
exit_manager.py
===============
Indicator-based and hybrid Chandelier exit logic.

Used by both trade_engine (spot longs) and futures_engine (shorts).

Two layers:
  Layer 1 — Signal reversal: re-checks the entry indicator each cycle.
             Only fires after minimum hold + price momentum confirmation.

  Layer 2 — Hybrid Chandelier trailing stop:
             Anchors to PEAK (long) or TROUGH (short) since entry.
             Trail distance = ATR% of entry price, capped between 2% and 4%.
               BTC ATR ~1%  → 2% trail (floor)
               ETH ATR ~2.5% → 2.5% trail
               SOL ATR ~3.5% → 3.5% trail (near ceiling)
             Breakeven protection at 1.5% profit.
             Stop only moves in profitable direction (ratchet).
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
# Layer 1 — Signal reversal detection
# -------------------------------------------------------------------
def check_signal_reversal(df: pd.DataFrame, position: dict) -> bool:
    """
    Re-run the indicator that generated the entry signal.
    Returns True only if ALL of:
      1. Indicator fires the opposite direction
      2. Price is actually moving against the trade (momentum confirmation)
      3. For mean-reversion signals (bollinger/rsi): previous candle also reversed
    """
    signal_type = position.get('signal_type', '')
    side        = position.get('side', 'long')
    opposite    = 'short' if side == 'long' else 'long'
    entry_price = position.get('entry_price', 0.0)

    try:
        from modules.strategy_tools import (
            ema_crossover_signal, macd_signal, rsi_signal,
            bollinger_band_signal, breakout_signal, volume_breakout_signal,
        )

        if len(df) < 50:
            return False

        current_price = float(df['close'].iloc[-1])
        prev_price    = float(df['close'].iloc[-2])

        # Price momentum must confirm
        if side == 'long' and current_price >= prev_price:
            return False
        if side == 'short' and current_price <= prev_price:
            return False

        # Require at least 0.3% adverse move from entry
        if entry_price > 0:
            move_against = ((entry_price - current_price) / entry_price
                            if side == 'long'
                            else (current_price - entry_price) / entry_price)
            if move_against < 0.003:
                return False
        else:
            move_against = 0.0

        reversal_signal = None

        if 'ema' in signal_type:
            reversal_signal = ema_crossover_signal(df)
        elif 'macd' in signal_type:
            reversal_signal = macd_signal(df)
        elif 'rsi' in signal_type:
            reversal_signal = rsi_signal(df)
        elif 'bollinger' in signal_type:
            reversal_signal = bollinger_band_signal(df)
            if reversal_signal == opposite:
                prev_window = df.iloc[:-1]
                if len(prev_window) >= 50:
                    if bollinger_band_signal(prev_window) != opposite:
                        return False
        elif 'breakout' in signal_type:
            reversal_signal = breakout_signal(df)
        elif 'volume' in signal_type:
            reversal_signal = volume_breakout_signal(df)
        else:
            return False

        if reversal_signal == opposite:
            logger.info(
                f"🔄 Signal reversal confirmed: was {side}, now {reversal_signal} "
                f"(indicator: {signal_type}, move: {move_against:.2%})"
            )
            return True

        return False

    except Exception as e:
        logger.debug(f"Signal reversal check failed: {e}")
        return False


# -------------------------------------------------------------------
# Layer 2 — Chandelier Exit
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
# Combined exit check — call once per position per cycle
# -------------------------------------------------------------------
def evaluate_exit(
    symbol: str,
    position: dict,
    current_price: float,
    df: Optional[pd.DataFrame] = None,
) -> Tuple[bool, str]:
    """
    Run both exit layers for a position.

    Returns:
      (True, reason)  — exit now
      (False, '')     — hold; position dict may be mutated (stop / peak / trough updated)

    Order:
      0. Track candles held (for signal reversal)
      1. Recalculate ATR from fresh df
      2. Update Chandelier stop
      3. Check if stop or TP was hit
      4. Check signal reversal (only after 6 candle minimum hold)
    """
    if current_price <= 0:
        return False, ''

    # Always recalculate ATR from current window for accurate Chandelier anchoring
    # Falls back to stored entry ATR if no df provided
    if df is not None and not df.empty:
        atr = _calculate_atr(df)
    else:
        atr = position.get('atr', 0.0)

    # --- Candle counter: increment if a new candle has formed ---
    if df is not None and not df.empty:
        # Use the timestamp column if available, otherwise use index
        if 'timestamp' in df.columns:
            current_candle_time = df['timestamp'].iloc[-1]
        else:
            # Fallback to index (less reliable)
            current_candle_time = len(df) - 1
        
        last_candle = position.get('last_candle_time')
        if last_candle is None or current_candle_time > last_candle:
            position['candles_held'] = position.get('candles_held', 0) + 1
            position['last_candle_time'] = current_candle_time


    # Attach symbol to position for logging
    position.setdefault('symbol', symbol)

    # --- Layer 2: Chandelier stop update ---
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

    # --- Layer 1: signal reversal (min 6 candles) ---
    if df is not None and not df.empty:
        candles_held = position.get('candles_held', 0)
        if candles_held >= 6:
            if check_signal_reversal(df, position):
                return True, 'signal_reversal'

    return False, ''