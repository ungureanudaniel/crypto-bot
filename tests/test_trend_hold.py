"""
Offline tests for the slow trend-following strategy (modules/trend_hold.py), its exit rule,
its backtest integration, and the long-only switch.

Run:  python tests/test_trend_hold.py     (or: pytest tests/test_trend_hold.py)
"""
import contextlib
import os
import sys
import tempfile
import types

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules import trend_hold as th
import backtest as bt

CFG = {'trading_timeframe': '4h'}          # entry 120 bars, exit 60, atr 120 (20/10/20 days)


def candles(closes, spread=0.004):
    closes = np.asarray(closes, dtype=float)
    opens = np.r_[closes[0], closes[:-1]]
    ts = pd.date_range('2025-01-01', periods=len(closes), freq='4h')
    return pd.DataFrame({'timestamp': ts, 'open': opens,
                         'high': np.maximum(opens, closes) * (1 + spread),
                         'low': np.minimum(opens, closes) * (1 - spread),
                         'close': closes, 'volume': 1000.0})


def flat_then_break(n_flat=200, jump=1.05):
    closes = np.full(n_flat, 100.0) + np.sin(np.arange(n_flat)) * 0.3
    return candles(np.r_[closes, closes[-1] * jump])


# ---------------------------------------------------------------- entry
def test_breakout_above_20_day_high_gives_a_long_with_no_take_profit():
    sig = th.entry_signal(flat_then_break(), 10_000.0, 0.03, 'AAA/USDC', CFG)
    assert sig and sig['side'] == 'long' and sig['signal_type'].startswith('trend_hold')
    assert sig['take_profit'] == 0.0
    assert 0 < sig['stop_loss'] < sig['entry_price']


def test_no_signal_without_a_breakout():
    df = flat_then_break(jump=1.0)                          # last close inside the range
    assert th.entry_signal(df, 10_000.0, 0.03, 'AAA/USDC', CFG) is None


def test_no_signal_with_too_little_history():
    df = flat_then_break().iloc[-60:]
    assert th.entry_signal(df, 10_000.0, 0.03, 'AAA/USDC', CFG) is None


def test_position_is_capped_at_15_percent_of_equity():
    sig = th.entry_signal(flat_then_break(), 10_000.0, 0.50, 'AAA/USDC', CFG)   # huge risk budget
    assert sig['units'] * sig['entry_price'] <= 10_000.0 * 0.15 + 1e-6


def test_tiny_account_is_rejected_below_minimum_order():
    assert th.entry_signal(flat_then_break(), 20.0, 0.03, 'AAA/USDC', CFG) is None


def test_stop_uses_daily_atr_not_the_much_smaller_4h_range():
    rng = np.random.default_rng(3)
    closes = 100 * np.exp(np.cumsum(rng.normal(0, 0.012, 400)))
    df = candles(closes, spread=0.004)
    daily = th.daily_atr(df, CFG)
    intraday = float((df['high'] - df['low']).iloc[-120:].mean())          # what the old code used
    assert daily > 1.8 * intraday, (daily, intraday)                       # ~sqrt(6) larger


def test_daily_atr_is_a_pure_function_of_data_up_to_the_current_candle():
    rng = np.random.default_rng(5)
    df = candles(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 400))))
    a = th.daily_atr(df.iloc[:300], CFG)
    df2 = df.copy()
    df2.iloc[300:, df2.columns.get_loc('high')] *= 3.0                     # wild "future" candles
    assert th.daily_atr(df2.iloc[:300], CFG) == a


def test_slow_variant_needs_more_history_and_uses_configured_days():
    slow = {'trading_timeframe': '4h', 'trend_hold': {'entry_days': 55, 'exit_days': 20}}
    p = th.params(slow)
    assert p['entry_bars'] == 330 and p['exit_bars'] == 120
    assert th.history_needed(slow) >= 332
    assert th.entry_signal(flat_then_break(n_flat=200), 10_000.0, 0.03, 'AAA/USDC', slow) is None  # too short


# ---------------------------------------------------------------- exit
def _pos(stop=95.0):
    return {'signal_type': 'trend_hold_breakout', 'stop_loss': stop, 'side': 'long'}


def test_exit_stop_only_ratchets_up_to_the_10_day_low():
    closes = np.linspace(100, 130, 100)                     # steady rise
    df = candles(closes)
    pos = _pos(95.0)
    exit_, _ = th.evaluate_exit(pos, 130.0, df, CFG)
    assert not exit_ and pos['stop_loss'] > 95.0            # ratcheted to the 10-day low
    raised = pos['stop_loss']
    # market dips: the 10-day low can't drag the stop back down
    df2 = candles(np.r_[closes, np.full(5, 118.0)])
    th.evaluate_exit(pos, 118.0, df2, CFG)
    assert pos['stop_loss'] >= raised


def test_exit_when_price_breaks_the_channel_and_reason_reflects_trailing():
    df = candles(np.linspace(100, 130, 100))
    pos = _pos(95.0)
    th.evaluate_exit(pos, 130.0, df, CFG)
    exit_, reason = th.evaluate_exit(pos, pos['stop_loss'] - 0.01, df, CFG)
    assert exit_ and reason == 'trailing_stop'


def test_initial_stop_hit_is_reported_as_stop_loss():
    # the 10-day low (a dip to 90 inside the window) sits BELOW the initial stop, so the stop never rises
    df = candles(np.r_[np.full(50, 100.0), np.full(10, 90.0), np.full(40, 100.0)])
    pos = _pos(95.0)
    exit_, reason = th.evaluate_exit(pos, 94.0, df, CFG)
    assert exit_ and reason == 'stop_loss' and pos['stop_loss'] == 95.0


def test_exit_manager_routes_trend_hold_positions_to_this_rule():
    from modules.exit_manager import evaluate_exit
    df = candles(np.linspace(100, 130, 100))
    pos = _pos(95.0)
    pos.update(entry_price=100.0, atr=1.0, take_profit=0.0, candles_held=99)
    # legacy indicator exits (MACD / signal flip) must NOT fire; only the channel stop matters
    exit_, _ = evaluate_exit('AAA/USDC', pos, 129.0, df)
    assert exit_ is False


# ---------------------------------------------------------------- backtest integration
@contextlib.contextmanager
def patched_backtest(fetch):
    saved = (bt.fetch_historical_data, bt.predict_regime)
    bt.fetch_historical_data = fetch
    bt.predict_regime = lambda w: "unused"
    bt.apply_strategy('trend_hold')
    try:
        yield
    finally:
        bt.fetch_historical_data, bt.predict_regime = saved
        bt.apply_strategy('legacy')
        os.environ.pop('BACKTEST_STRATEGY', None)


def steady_uptrend(symbol, interval='4h', days=365):
    rng = np.random.default_rng(7)
    n = 1200
    closes = 100 * np.exp(np.cumsum(rng.normal(0.0015, 0.004, n)))   # strong drift, low noise
    return candles(closes, spread=0.002)


def test_backtest_holds_a_trend_far_longer_than_the_legacy_timeout():
    with tempfile.TemporaryDirectory() as tmp, patched_backtest(steady_uptrend):
        b = bt.Backtester(coins=['AAA/USDC'], days=200, capital=5000.0, workers=1,
                          data_dir=tmp, results_dir=os.path.join(tmp, 'r'),
                          log_experiment=False, slippage=0.0)
        trades = b.run()
        assert trades, "the trend should have produced at least one breakout entry"
        assert all(t['signal_type'].startswith('trend_hold') for t in trades)
        assert all(t['exit_reason'] != 'take_profit' for t in trades)
        assert all(t['exit_reason'] != 'timeout' for t in trades)
        assert max(t['candles_held'] for t in trades) > 336          # beyond the legacy 56-day timeout
        assert sum(t['net_pnl'] for t in trades) > 0


def test_backtest_trades_rarely_on_choppy_data():
    def choppy(symbol, interval='4h', days=365):
        rng = np.random.default_rng(11)
        return candles(100 + np.cumsum(rng.normal(0, 0.6, 1200)) * 0.3)
    with tempfile.TemporaryDirectory() as tmp, patched_backtest(choppy):
        b = bt.Backtester(coins=['AAA/USDC'], days=200, capital=5000.0, workers=1,
                          data_dir=tmp, results_dir=os.path.join(tmp, 'r'), log_experiment=False)
        assert len(b.run()) <= 20                                     # breakouts are rare by design


# ---------------------------------------------------------------- long-only switch
def test_engine_refuses_shorts_unless_enabled():
    from test_limit_orders import make_engine
    eng, ex, clock = make_engine()
    eng.check_drawdown = lambda: True
    calls = []
    eng.futures_engine = types.SimpleNamespace(open_short=lambda **k: calls.append(k) or True)
    sig = {'symbol': 'ETH/USDC', 'signal': {'side': 'short', 'entry_price': 100.0, 'units': 1.0,
                                             'stop_loss': 105.0, 'take_profit': 90.0}}
    eng.config = {}                                                  # default: shorts off
    assert eng.shorts_enabled is False
    assert eng.execute_signal(sig) is False and calls == []
    eng.config = {'enable_shorts': True}
    assert eng.shorts_enabled is True


def test_live_trend_hold_entries_are_blocked_until_stop_only_protection_exists():
    from test_limit_orders import make_engine
    eng, ex, clock = make_engine()                                   # trading_mode = testnet
    eng.config = {}
    ok = eng.open_position('ETH/USDC', 'long', 100.0, 1.0, 95.0, 0.0, signal_type='trend_hold_breakout')
    assert ok is False and not ex.orders and 'ETH/USDC' not in eng.pending_entries
    # ...while a normal signal is unaffected
    ok = eng.open_position('ETH/USDC', 'long', 100.05, 1.0, 97.0, 109.0, signal_type='breakout_trend_up')
    assert ok is True and 'ETH/USDC' in eng.pending_entries


if __name__ == '__main__':
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith('test_') and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as e:
            failed += 1
            import traceback
            print(f"FAIL  {name}: {e!r}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
