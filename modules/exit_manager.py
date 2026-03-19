"""
exit_manager.py
===============
Indicator-based and trailing stop exit logic.

Used by both trade_engine (spot longs) and futures_engine (shorts).

Two layers:
  Layer 1 — Signal reversal: re-checks the entry indicator each cycle.
             If it now fires the opposite direction, exit immediately.
  Layer 2 — Trailing stop: activates once profit >= 1x ATR.
             Moves stop to breakeven at 1x ATR profit.
             Trails 1x ATR behind price once profit >= 2x ATR.
"""

import logging
import pandas as pd
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# ATR helper (standalone, no ta dependency)
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
    Returns True if the indicator now fires the OPPOSITE direction
    (i.e. the trade thesis has broken down and we should exit).

    position dict must contain 'signal_type' and 'side'.
    """
    signal_type = position.get('signal_type', '')
    side        = position.get('side', 'long')
    opposite    = 'short' if side == 'long' else 'long'

    try:
        # Import signal functions lazily to avoid circular imports
        from modules.strategy_tools import (
            ema_crossover_signal,
            macd_signal,
            rsi_signal,
            bollinger_band_signal,
            breakout_signal,
            volume_breakout_signal,
        )

        if len(df) < 50:
            return False

        reversal_signal = None

        if 'ema' in signal_type:
            reversal_signal = ema_crossover_signal(df)
        elif 'macd' in signal_type:
            reversal_signal = macd_signal(df)
        elif 'rsi' in signal_type:
            reversal_signal = rsi_signal(df)
        elif 'bollinger' in signal_type:
            reversal_signal = bollinger_band_signal(df)
        elif 'breakout' in signal_type:
            reversal_signal = breakout_signal(df)
        elif 'volume' in signal_type:
            reversal_signal = volume_breakout_signal(df)
        else:
            # Unknown signal type — don't force exit
            return False

        if reversal_signal == opposite:
            logger.info(
                f"🔄 Signal reversal detected: was {side}, now {reversal_signal} "
                f"(indicator: {signal_type})"
            )
            return True

        return False

    except Exception as e:
        logger.debug(f"Signal reversal check failed: {e}")
        return False


# -------------------------------------------------------------------
# Layer 2 — Trailing stop management
# -------------------------------------------------------------------
def update_trailing_stop(
    position: dict,
    current_price: float,
    atr: float,
) -> Tuple[Optional[float], str]:
    """
    Update the trailing stop for an open position.

    Rules:
      - Profit >= 1x ATR → move stop to breakeven (entry price)
      - Profit >= 2x ATR → trail stop 1x ATR behind current price

    Returns:
      (new_stop_loss, reason_string) if stop should be updated,
      (None, '')                     if no update needed.
    """
    if atr <= 0:
        return None, ''

    entry_price  = position.get('entry_price', current_price)
    current_stop = position.get('stop_loss', 0.0)
    side         = position.get('side', 'long')

    if side == 'long':
        profit         = current_price - entry_price
        breakeven      = entry_price
        min_trail_dist = max(atr, current_price * 0.01)
        trail_stop     = current_price - min_trail_dist

        # 3x ATR profit → start trailing 1x ATR behind price
        # 2x ATR profit → move stop to breakeven only
        # Rationale: on 1h candles, 1x ATR is just normal candle noise.
        # Waiting for 2x ATR confirms the move is real before protecting it.
        if profit >= 3 * atr:
            if trail_stop > current_stop:
                logger.info(
                    f"📈 Trailing stop updated: ${current_stop:.6f} → ${trail_stop:.6f} "
                    f"(price ${current_price:.6f}, dist ${min_trail_dist:.6f})"
                )
                return trail_stop, 'trailing'

        elif profit >= 2 * atr:
            if breakeven > current_stop:
                logger.info(
                    f"⚖️  Stop moved to breakeven: ${current_stop:.6f} → ${breakeven:.6f}"
                )
                return breakeven, 'breakeven'

    else:  # short
        profit         = entry_price - current_price
        breakeven      = entry_price
        min_trail_dist = max(atr, current_price * 0.01)
        trail_stop     = current_price + min_trail_dist

        if profit >= 3 * atr:
            if trail_stop < current_stop:
                logger.info(
                    f"📉 Trailing stop updated: ${current_stop:.6f} → ${trail_stop:.6f} "
                    f"(price ${current_price:.6f}, dist ${min_trail_dist:.6f})"
                )
                return trail_stop, 'trailing'

        elif profit >= 2 * atr:
            if breakeven < current_stop:
                logger.info(
                    f"⚖️  Short stop moved to breakeven: ${current_stop:.6f} → ${breakeven:.6f}"
                )
                return breakeven, 'breakeven'

    return None, ''


# -------------------------------------------------------------------
# Combined exit check — call this once per position per cycle
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
      (True, reason)  — exit the position now
      (False, '')     — hold, but position['stop_loss'] may have been updated in-place

    'position' dict is mutated in-place if the trailing stop moves.
    """
    if current_price <= 0:
        return False, ''

    # Use ATR stored at entry time first — only recalculate if not available
    # This ensures trailing stop distances are consistent with entry sizing
    stored_atr = position.get('atr', 0.0)
    if stored_atr > 0:
        atr = stored_atr
    elif df is not None and not df.empty:
        atr = _calculate_atr(df)
    else:
        atr = 0.0

    # --- Layer 2: update trailing stop first (mutates position in-place) ---
    if atr > 0:
        new_stop, stop_reason = update_trailing_stop(position, current_price, atr)
        if new_stop is not None:
            position['stop_loss'] = new_stop
            position['trailing_stop_active'] = True

    # --- Check if updated stop was hit ---
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

    # --- Layer 1: signal reversal (only if we have fresh data) ---
    if df is not None and not df.empty:
        if check_signal_reversal(df, position):
            return True, 'signal_reversal'

    return False, ''
