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
    Returns True only if ALL of:
      1. Indicator fires the opposite direction
      2. Price is actually moving against the trade (momentum confirmation)
      3. For mean-reversion signals (bollinger/rsi): previous candle also reversed
         — prevents single-candle noise from closing a valid position

    position dict must contain 'signal_type', 'side' and 'entry_price'.
    """
    signal_type  = position.get('signal_type', '')
    side         = position.get('side', 'long')
    opposite     = 'short' if side == 'long' else 'long'
    entry_price  = position.get('entry_price', 0.0)

    try:
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

        current_price = float(df['close'].iloc[-1])
        prev_price    = float(df['close'].iloc[-2])

        # Price momentum must confirm the reversal direction
        # For long exits: price must be falling (current < previous)
        # For short exits: price must be rising (current > previous)
        if side == 'long' and current_price >= prev_price:
            return False  # Price still rising — don't exit long
        if side == 'short' and current_price <= prev_price:
            return False  # Price still falling — don't exit short

        # Also require price to have moved at least 0.3% against position
        # Filters out flat/sideways candles that briefly touch the wrong side
        if entry_price > 0:
            move_against = ((entry_price - current_price) / entry_price
                           if side == 'long'
                           else (current_price - entry_price) / entry_price)
            if move_against < 0.003:
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
            # For mean-reversion signals, require previous candle also reversed
            # (two consecutive confirmations to avoid single-candle whipsaws)
            if reversal_signal == opposite:
                prev_window = df.iloc[:-1]
                if len(prev_window) >= 50:
                    prev_reversal = bollinger_band_signal(prev_window)
                    if prev_reversal != opposite:
                        return False  # Only one candle — not confirmed
        elif 'breakout' in signal_type:
            reversal_signal = breakout_signal(df)
        elif 'volume' in signal_type:
            reversal_signal = volume_breakout_signal(df)
        else:
            return False

        if reversal_signal == opposite:
            logger.info(
                f"🔄 Signal reversal confirmed: was {side}, now {reversal_signal} "
                f"(indicator: {signal_type}, price move: {move_against:.2%})"
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
    Percentage-based trailing stop — scales correctly regardless of account size or asset price.

    Rules:
      - Profit >= 1.5% → move stop to breakeven
      - Profit >= 3.0% → trail stop 1.5% behind current price (ratchets up, never back)

    ATR is still used to set the minimum trail distance so the stop isn't
    tighter than the asset's natural noise on 1h candles.
    """
    entry_price  = position.get('entry_price', current_price)
    current_stop = position.get('stop_loss', 0.0)
    side         = position.get('side', 'long')

    # Percentage thresholds
    breakeven_threshold = 0.015   # 1.5% profit → move to breakeven
    trail_threshold     = 0.030   # 3.0% profit → start trailing
    trail_distance_pct  = 0.020   # trail 2.0% behind price (was 1.5%)

    # Minimum trail distance: larger of 1.5% or 1x ATR as % of price
    # Prevents stop being tighter than normal 1h noise
    atr_pct = (atr / entry_price) if entry_price > 0 else 0
    effective_trail_pct = max(trail_distance_pct, atr_pct)

    if side == 'long':
        profit_pct = (current_price - entry_price) / entry_price
        breakeven  = entry_price
        trail_stop = current_price * (1 - effective_trail_pct)

        if profit_pct >= trail_threshold:
            if trail_stop > current_stop:
                logger.info(
                    f"📈 Trailing stop updated: ${current_stop:.6f} → ${trail_stop:.6f} "
                    f"(profit {profit_pct:.2%}, trail dist {effective_trail_pct:.2%})"
                )
                return trail_stop, 'trailing'

        elif profit_pct >= breakeven_threshold:
            if breakeven > current_stop:
                logger.info(
                    f"⚖️  Stop moved to breakeven: ${current_stop:.6f} → ${breakeven:.6f} "
                    f"(profit {profit_pct:.2%})"
                )
                return breakeven, 'breakeven'

    else:  # short
        profit_pct = (entry_price - current_price) / entry_price
        breakeven  = entry_price
        trail_stop = current_price * (1 + effective_trail_pct)

        if profit_pct >= trail_threshold:
            if trail_stop < current_stop:
                logger.info(
                    f"📉 Trailing stop updated: ${current_stop:.6f} → ${trail_stop:.6f} "
                    f"(profit {profit_pct:.2%}, trail dist {effective_trail_pct:.2%})"
                )
                return trail_stop, 'trailing'

        elif profit_pct >= breakeven_threshold:
            if breakeven < current_stop:
                logger.info(
                    f"⚖️  Short stop moved to breakeven: ${current_stop:.6f} → ${breakeven:.6f} "
                    f"(profit {profit_pct:.2%})"
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

    # --- Layer 1: signal reversal (only after minimum hold period) ---
    # Minimum 6 candles (6h on 1h tf) before reversal can trigger
    # Gives the trade time to develop past entry noise
    if df is not None and not df.empty:
        candles_held = position.get('candles_held', 0)
        if candles_held >= 6:
            if check_signal_reversal(df, position):
                return True, 'signal_reversal'

    return False, ''
