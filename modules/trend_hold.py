"""
trend_hold.py
=============
Slow, long-only trend following: few trades, held for as long as the trend lasts.
Classic Turtle-style rules with fixed (not tuned) defaults:

  ENTRY : close breaks above the highest high of the previous `entry_days` (default 20 days)
  STOP  : initial stop = entry - `stop_atr_mult` (2.0) x ATR(`atr_days`=20) of DAILY candles
  EXIT  : the stop is ratcheted UP to the lowest low of the previous `exit_days` (default 10 days)
          and never lowered; the position is closed when price breaks it.
          No take-profit and no indicator-flip exits - winners are allowed to run.

Because the exit rule *is* a rising stop price, it maps 1:1 onto an exchange-side stop order
that the bot ratchets, exactly like the existing trailing stop.

Enable with  "strategy_mode": "trend_hold"  in config.json. Optional overrides:
  "trend_hold": {"entry_days": 20, "exit_days": 10, "atr_days": 20, "stop_atr_mult": 2.0,
                 "max_position_pct": 0.15}
Day counts are converted to candles using "trading_timeframe".
"""
import pandas as pd
from typing import Dict, Optional, Tuple

DEFAULTS = {'entry_days': 20, 'exit_days': 10, 'atr_days': 20,
            'stop_atr_mult': 2.0, 'max_position_pct': 0.15}
BARS_PER_DAY = {'1m': 1440, '5m': 288, '15m': 96, '30m': 48, '1h': 24, '2h': 12,
                '4h': 6, '6h': 4, '12h': 2, '1d': 1}
SIGNAL_TYPE = 'trend_hold_breakout'
MIN_ORDER_USD = 10.0


def params(cfg: dict) -> Dict:
    p = dict(DEFAULTS)
    p.update(cfg.get('trend_hold') or {})
    bpd = BARS_PER_DAY.get(cfg.get('trading_timeframe', '4h'), 6)
    p['entry_bars'] = max(2, int(p['entry_days'] * bpd))
    p['exit_bars'] = max(2, int(p['exit_days'] * bpd))
    p['atr_bars'] = max(2, int(p['atr_days'] * bpd))
    p['bars_per_day'] = bpd
    return p


def is_trend_hold(position: dict) -> bool:
    return str(position.get('signal_type', '')).startswith('trend_hold')


def history_needed(cfg: dict) -> int:
    """Candles a caller must supply to entry_signal / exit_level."""
    p = params(cfg)
    return max(p['entry_bars'], p['exit_bars'], p['atr_bars'] + p['bars_per_day']) + 2


def exit_level(df: pd.DataFrame, cfg: dict) -> Optional[float]:
    """Lowest low of the previous `exit_bars` candles (the current one excluded)."""
    p = params(cfg)
    if df is None or len(df) < p['exit_bars'] + 2:
        return None
    return float(df['low'].iloc[-(p['exit_bars'] + 1):-1].min())


def daily_atr(df: pd.DataFrame, cfg: dict) -> float:
    """Average true range of DAILY candles over `atr_days` (Turtle 'N').
    Intraday candles are grouped into days of `bars_per_day` candles counted back from the newest
    one, so it uses only data up to and including the current candle. (Averaging the raw
    intraday ranges instead would give a stop about 2.4x too tight on 4h data.)"""
    p = params(cfg)
    bpd, days = p['bars_per_day'], p['atr_days']
    need = (days + 1) * bpd
    tail = df.iloc[-need:]
    if len(tail) < need:
        return 0.0
    g = (pd.Series(range(len(tail))[::-1], index=tail.index) // bpd)     # 0 = newest day
    daily = pd.DataFrame({'high': tail['high'].groupby(g).max(),
                          'low': tail['low'].groupby(g).min(),
                          'close': tail['close'].groupby(g).last()}).sort_index(ascending=False)
    prev_close = daily['close'].shift(-1)                                 # previous (older) day's close
    tr = pd.concat([daily['high'] - daily['low'],
                    (daily['high'] - prev_close).abs(),
                    (daily['low'] - prev_close).abs()], axis=1).max(axis=1)
    return float(tr.iloc[:days].mean())


def entry_signal(df: pd.DataFrame, equity: float, risk_per_trade: float,
                 symbol: Optional[str], cfg: dict) -> Optional[Dict]:
    p = params(cfg)
    if df is None or len(df) < history_needed(cfg):
        return None

    close = float(df['close'].iloc[-1])
    prev_high = float(df['high'].iloc[-(p['entry_bars'] + 1):-1].max())
    if close <= prev_high:
        return None                                        # no breakout

    atr = daily_atr(df, cfg)
    if not atr > 0:
        return None

    stop = max(close - p['stop_atr_mult'] * atr, exit_level(df, cfg) or 0.0)
    risk_per_unit = close - stop
    if stop <= 0 or risk_per_unit <= 0:
        return None

    units = min((equity * risk_per_trade) / risk_per_unit,      # risk-based size
                (equity * p['max_position_pct']) / close)       # ...capped per position
    if units * close < MIN_ORDER_USD:
        return None

    return {
        'symbol': symbol,
        'side': 'long',
        'signal_type': SIGNAL_TYPE,
        'units': units,
        'entry_price': close,
        'stop_loss': stop,
        'take_profit': 0.0,                # none: the rising stop is the exit
        'risk_pct': risk_per_trade,
        'regime': 'trend_hold',
        'atr': atr,
        'exit_strategy': 'trend_hold',
    }


def evaluate_exit(position: dict, price: float, df: Optional[pd.DataFrame],
                  cfg: dict) -> Tuple[bool, str]:
    """Ratchet the stop up to the exit channel and report whether price has broken it."""
    initial = position.setdefault('initial_stop', position.get('stop_loss', 0.0))
    level = exit_level(df, cfg) if df is not None else None
    stop = position.get('stop_loss', 0.0) or 0.0
    if level is not None and level > stop:
        stop = level
        position['stop_loss'] = stop
    if stop > initial:
        position['trailing_stop_active'] = True
    if stop and price <= stop:
        return True, 'trailing_stop' if position.get('trailing_stop_active') else 'stop_loss'
    return False, ''
