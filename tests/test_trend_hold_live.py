"""
Offline tests for the LIVE side of the trend_hold strategy:
  - stop-only exchange protection (no take-profit): place / ratchet / fill / market fallback
  - live BTC gate + closed-candle handling (and parity with the backtest gate)
  - scan_and_trade for trend_hold: history size, no regime model, per-candle duplicate key

Run:  python tests/test_trend_hold_live.py     (or: pytest tests/test_trend_hold_live.py)
"""
import os
import sys
import tempfile
import types

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test_limit_orders import make_engine, SYMBOL
from test_trend_hold import candles, flat_then_break
import modules.trade_engine as te
from modules import market_gate, trend_hold as th
import backtest as bt


# ---------------------------------------------------------------- helpers
def open_trend_position(eng, ex, clock):
    """trend_hold entry: signal 100.05, stop 97.0, NO take-profit; the limit fills at step 2."""
    eng.config = {'trend_hold_live_enabled': True}
    assert eng.open_position(SYMBOL, 'long', 100.05, 1.0, 97.0, 0.0, signal_type='trend_hold_breakout')
    for _ in range(3):
        clock.advance(61)
        eng.manage_orders()
    eng.manage_orders()
    return eng.open_positions[SYMBOL]


def live_orders(ex, kind=None):
    return [o for o in ex.orders.values()
            if o['status'] == 'NEW' and o['side'] == 'SELL' and (kind is None or o['type'] == kind)]


# ---------------------------------------------------------------- stop-only protection
def test_entry_fill_places_a_stop_only_order_and_does_not_sell_at_once():
    eng, ex, clock = make_engine()
    pos = open_trend_position(eng, ex, clock)
    assert SYMBOL in eng.open_positions, "position must stay open (zero target must not mean 'target reached')"
    assert pos['take_profit'] == 0.0
    assert pos['oco']['kind'] == 'stop'
    stops = live_orders(ex, 'STOP_LOSS_LIMIT')
    assert len(stops) == 1 and not live_orders(ex, 'LIMIT_MAKER') and not ex.lists
    assert stops[0]['price'] < stops[0]['stop'] < ex.mid
    assert abs(stops[0]['qty'] - 0.999) < 1e-9                       # net of the fee taken in ETH


def test_stop_ratchets_up_by_replacing_the_stop_order():
    eng, ex, clock = make_engine()
    pos = open_trend_position(eng, ex, clock)
    old = live_orders(ex, 'STOP_LOSS_LIMIT')[0]
    ex.set_market(108.0, 108.1)
    pos['stop_loss'] = 104.0                                          # channel low rose
    eng.manage_orders()
    new = live_orders(ex, 'STOP_LOSS_LIMIT')
    assert len(new) == 1 and new[0]['orderId'] != old['orderId']
    assert ex.orders[old['orderId']]['status'] == 'CANCELED'
    assert abs(new[0]['stop'] - 104.0) < 0.01 and abs(pos['oco']['stop'] - 104.0) < 0.01


def test_stop_order_fill_closes_the_position():
    eng, ex, clock = make_engine()
    pos = open_trend_position(eng, ex, clock)
    stop, limit = pos['oco']['stop'], pos['oco']['limit']
    ex.set_market(limit + 0.05, limit + 0.15)                         # dips through the stop, bid >= limit
    eng.manage_orders()
    assert SYMBOL not in eng.open_positions
    assert eng.trades[-1]['reason'] == 'stop_loss'
    assert abs(eng.trades[-1]['exit_price'] - limit) < 1e-6


def test_market_fallback_when_price_gaps_through_the_stop_order():
    eng, ex, clock = make_engine()
    pos = open_trend_position(eng, ex, clock)
    order = live_orders(ex, 'STOP_LOSS_LIMIT')[0]
    ex.set_market(pos['oco']['stop'] * 0.90)                          # crash: bid far below the limit price
    eng.manage_orders()
    assert SYMBOL not in eng.open_positions
    assert eng.trades[-1]['reason'] == 'stop_loss_market'
    assert ex.orders[order['orderId']]['status'] == 'CANCELED'        # stop order cancelled first
    assert ex.bal['ETH'] < 1e-6                                       # everything sold


def test_market_fallback_when_stop_order_is_stuck_unfilled():
    eng, ex, clock = make_engine()
    pos = open_trend_position(eng, ex, clock)
    stop, limit = pos['oco']['stop'], pos['oco']['limit']
    ex.set_market(limit - 0.02, stop - 0.05)                          # triggered but bid < limit: can't fill
    eng.manage_orders()
    assert SYMBOL in eng.open_positions                               # inside the 120s grace period
    clock.advance(130)
    eng.manage_orders()
    assert SYMBOL not in eng.open_positions
    assert eng.trades[-1]['reason'] == 'stop_loss_market'


def test_exchange_owns_the_stop_unless_the_bot_stop_is_ahead_of_it():
    eng, ex, clock = make_engine()
    pos = open_trend_position(eng, ex, clock)
    eng._handle_exit_signal(SYMBOL, pos, pos['oco']['stop'] - 0.5, 'stop_loss')
    assert SYMBOL not in eng.pending_exits                            # left to the exchange order
    pos['stop_loss'] = 104.0                                          # ratchet not yet applied
    eng._handle_exit_signal(SYMBOL, pos, 103.0, 'trailing_stop')
    assert SYMBOL in eng.pending_exits and eng.pending_exits[SYMBOL]['st']['urgent'] is True


def test_manual_close_cancels_the_stop_order_first():
    eng, ex, clock = make_engine()
    open_trend_position(eng, ex, clock)
    stop_order = live_orders(ex, 'STOP_LOSS_LIMIT')[0]
    assert eng.close_position(SYMBOL, 100.0, 'emergency_sell') is True
    assert ex.orders[stop_order['orderId']]['status'] == 'CANCELED'
    assert SYMBOL in eng.pending_exits


def test_protection_saved_before_kinds_existed_is_treated_as_an_oco():
    """Positions persisted by the earlier version have no 'kind' key. They must still cancel and
    report status through the OCO path."""
    eng, ex, clock = make_engine()
    om = eng.order_manager
    ex.bal['ETH'] = 1.0
    res = om.place_oco(SYMBOL, 1.0, 97.0, 109.0, ex.mid)
    assert res['ok'] and res['kind'] == 'oco'
    legacy = {k: v for k, v in res.items() if k not in ('ok', 'kind')}          # as saved by the old version
    assert 'kind' not in legacy
    assert om.protection_status(SYMBOL, legacy)['status'] == 'EXECUTING'
    assert om.cancel_protection(SYMBOL, legacy) == 'cancelled'
    assert ex.lists[legacy['order_list_id']]['status'] == 'ALL_DONE'


def test_stop_order_helpers_report_cancel_and_done_correctly():
    eng, ex, clock = make_engine()
    om = eng.order_manager
    ex.bal['ETH'] = 2.0
    a = om.place_stop(SYMBOL, 1.0, 97.0, ex.mid)
    b = om.place_stop(SYMBOL, 1.0, 96.0, ex.mid)
    assert a['ok'] and b['ok'] and a['kind'] == 'stop' and a['target'] == 0.0
    assert om.protection_status(SYMBOL, a)['status'] == 'EXECUTING'
    assert om.cancel_protection(SYMBOL, a) == 'cancelled'
    assert om.protection_status(SYMBOL, a) == {'status': 'ALL_DONE', 'filled': None, 'avg_price': 0.0, 'qty': 0.0}
    ex.set_market(95.8, 95.9)                                                    # b (stop 96, limit ~95.7) triggers and fills
    st = om.protection_status(SYMBOL, b)
    assert st['status'] == 'ALL_DONE' and st['filled'] == 'stop' and st['qty'] == 1.0
    assert om.cancel_protection(SYMBOL, b) == 'done'                             # already filled
    # a stop at/above the market cannot be placed
    assert om.place_stop(SYMBOL, 1.0, 120.0, 100.0) == {'ok': False, 'error': 'stop_breached'}


# ---------------------------------------------------------------- gate + candles
def test_closed_only_drops_only_the_forming_candle():
    df = candles(np.linspace(100, 110, 20))
    now = df['timestamp'].iloc[-1] + pd.Timedelta(hours=1)            # last candle opened 1h ago (4h frame)
    assert len(market_gate.closed_only(df, '4h', now=now)) == 19
    now = df['timestamp'].iloc[-1] + pd.Timedelta(hours=4)            # exactly closed
    assert len(market_gate.closed_only(df, '4h', now=now)) == 20


def test_btc_uptrend_true_false_and_none():
    up = candles(np.linspace(100, 200, 400))
    down = candles(np.linspace(200, 100, 400))
    assert market_gate.btc_uptrend(up, 50, '4h') is True
    assert market_gate.btc_uptrend(down, 50, '4h') is False
    assert market_gate.btc_uptrend(up.iloc[:200], 50, '4h') is None   # < 300 candles: cannot decide


def test_live_gate_agrees_with_the_backtest_gate():
    rng = np.random.default_rng(21)
    closes = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.012, 1600)))
    df = candles(closes)
    with tempfile.TemporaryDirectory() as tmp:
        b = bt.Backtester(coins=['BTC/USDC'], btc_gate_days=50, data_dir=tmp, log_experiment=False)
        gate = b._build_btc_gate({'BTC/USDC': df}, list(df['timestamp']))
    agree = total = 0
    for i in range(1100, 1600, 7):                                    # live sees the last 1000 candles
        live = market_gate.btc_uptrend(df.iloc[i - 999:i + 1], 50, '4h')
        agree += (live == gate[df['timestamp'].iloc[i]])
        total += 1
    assert agree / total >= 0.98, f"live vs backtest gate agree on only {agree}/{total} bars"


class FakeFeed:
    def __init__(self, frames):
        self.frames, self.calls, self.fail = frames, [], set()

    def get_ohlcv(self, symbol, interval, limit=1000):
        self.calls.append((symbol, limit))
        if symbol in self.fail:
            raise RuntimeError("feed down")
        return self.frames[symbol].tail(limit).reset_index(drop=True)


def gate_engine(btc_closes):
    eng, ex, clock = make_engine()
    eng.symbols = ['ETH/USDC', 'BTC/USDC']
    eng.timeframe = '4h'
    eng.config = {'strategy_mode': 'trend_hold', 'trading_timeframe': '4h'}
    eng.data_feed = FakeFeed({'BTC/USDC': candles(btc_closes), 'ETH/USDC': flat_then_break(n_flat=420)})
    return eng, clock


def test_engine_gate_open_closed_and_fail_closed():
    eng, clock = gate_engine(np.linspace(100, 200, 500))
    assert eng._longs_allowed() is True

    eng, clock = gate_engine(np.linspace(200, 100, 500))
    assert eng._longs_allowed() is False

    eng, clock = gate_engine(np.linspace(100, 200, 500))
    eng.data_feed.fail.add('BTC/USDC')
    assert eng._longs_allowed() is False                              # data problem -> no new longs

    eng, clock = gate_engine(np.linspace(100, 200, 100))              # not enough history
    assert eng._longs_allowed() is False


def test_engine_gate_is_cached_for_a_minute_and_can_be_disabled():
    eng, clock = gate_engine(np.linspace(100, 200, 500))
    assert eng._longs_allowed() is True
    eng.data_feed.frames['BTC/USDC'] = candles(np.linspace(200, 100, 500))
    assert eng._longs_allowed() is True                               # cached
    clock.advance(61)
    assert eng._longs_allowed() is False                              # refreshed
    eng.config['btc_gate_days'] = 0
    assert eng._longs_allowed() is True                               # disabled


def test_history_limits_cover_what_trend_hold_needs():
    eng, ex, clock = make_engine()
    eng.config = {}
    assert eng._history_limit() == 200 and eng._exit_history_limit({'signal_type': 'rsi_range_long'}) == 50
    eng.config = {'strategy_mode': 'trend_hold', 'trading_timeframe': '4h',
                  'trend_hold': {'entry_days': 55, 'exit_days': 20}}
    assert eng._history_limit() >= th.history_needed(eng.config) and eng._history_limit() <= 1000
    assert eng._exit_history_limit({'signal_type': 'trend_hold_breakout'}) >= th.history_needed(eng.config)


# ---------------------------------------------------------------- scan_and_trade
class scan_env:
    """Configure a bare engine for scan_and_trade in trend_hold mode (and undo the global config)."""
    def __init__(self, eth_df, btc_closes=None):
        self.eth_df = eth_df
        self.btc_closes = np.linspace(100, 200, 500) if btc_closes is None else btc_closes

    def __enter__(self):
        self.saved = dict(te.config.config)
        te.config.config.update({'strategy_mode': 'trend_hold', 'trading_timeframe': '4h',
                                 'trend_hold': {'entry_days': 20, 'exit_days': 10}})
        eng, ex, clock = make_engine()
        eng.config = te.config.config
        eng.symbols = ['ETH/USDC', 'BTC/USDC']
        eng.timeframe = '4h'
        eng.risk_per_trade = 0.03
        eng.check_drawdown = lambda: True
        eng.get_cash_balance = lambda q='USDC': 10_000.0
        eng.last_trade_time_per_pair, eng.last_signals = {}, {}
        eng.data_feed = FakeFeed({'ETH/USDC': self.eth_df, 'BTC/USDC': candles(self.btc_closes)})
        self.saved_regime = te.predict_regime
        te.predict_regime = lambda df: (_ for _ in ()).throw(AssertionError("regime model must not be used"))
        self.eng = eng
        return eng

    def __exit__(self, *exc):
        te.predict_regime = self.saved_regime
        te.config.config.clear()
        te.config.config.update(self.saved)


def test_scan_finds_a_breakout_without_using_the_regime_model():
    with scan_env(flat_then_break(n_flat=420)) as eng:
        found = eng.scan_and_trade()
        eth = [s for s in found if s['symbol'] == 'ETH/USDC']
        assert len(eth) == 1 and eth[0]['signal']['signal_type'] == 'trend_hold_breakout'
        assert eth[0]['regime'] == 'trend_hold'
        assert max(limit for _, limit in eng.data_feed.calls if _ == 'ETH/USDC') >= th.history_needed(te.config.config)


def test_scan_blocks_all_longs_when_the_btc_gate_is_closed():
    with scan_env(flat_then_break(n_flat=420), btc_closes=np.linspace(200, 100, 500)) as eng:
        assert eng.scan_and_trade() == []
        assert not any(sym == 'ETH/USDC' for sym, _ in eng.data_feed.calls)   # did not even fetch


def test_scan_acts_only_on_closed_candles():
    df = flat_then_break(n_flat=420)
    now = pd.Timestamp.now(tz='UTC').tz_localize(None)
    forming = df.copy()
    forming['timestamp'] = pd.date_range(end=now - pd.Timedelta(hours=1), periods=len(df), freq='4h')
    with scan_env(forming) as eng:
        assert [s for s in eng.scan_and_trade() if s['symbol'] == 'ETH/USDC'] == []     # breakout candle still forming
    closed = df.copy()
    closed['timestamp'] = pd.date_range(end=now - pd.Timedelta(hours=5), periods=len(df), freq='4h')
    with scan_env(closed) as eng:
        assert len([s for s in eng.scan_and_trade() if s['symbol'] == 'ETH/USDC']) == 1


def test_duplicate_filter_resets_on_each_new_candle():
    """The legacy key (symbol + signal type) never resets, so a coin's SECOND breakout was blocked forever."""
    df1 = flat_then_break(n_flat=420, jump=1.05)
    with scan_env(df1) as eng:
        assert len([s for s in eng.scan_and_trade() if s['symbol'] == 'ETH/USDC']) == 1
        assert [s for s in eng.scan_and_trade() if s['symbol'] == 'ETH/USDC'] == []    # same candle: duplicate
        closes = np.r_[df1['close'].values, df1['close'].iloc[-1] * 1.03]               # next candle breaks out again
        eng.data_feed.frames['ETH/USDC'] = candles(closes)
        assert len([s for s in eng.scan_and_trade() if s['symbol'] == 'ETH/USDC']) == 1


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
