"""
Offline tests for the backtest experiment tooling (snapshot, signal cache, filters, periods,
benchmark, experiment log) and for the linear-sizing assumption the signal cache relies on.

Run:  python tests/test_backtest_tools.py     (or: pytest tests/test_backtest_tools.py)
Uses synthetic prices and workers=1 (in-process) so nothing touches the network.
"""
import os
import sys
import tempfile
import zlib

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import backtest as bt

COINS = ['AAA/USDT', 'BBB/USDT', 'CCC/USDT']
CALLS = {'fetch': 0, 'signal': 0, 'regime': 0}
_real_signal = bt.generate_trade_signal


def fake_fetch(symbol, interval='4h', days=365):
    CALLS['fetch'] += 1
    rng = np.random.default_rng(zlib.crc32(symbol.encode()))
    n = 1200
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.012, n)))
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) * (1 + rng.random(n) * 0.008)
    low = np.minimum(open_, close) * (1 - rng.random(n) * 0.008)
    ts = pd.date_range('2025-01-01', periods=n, freq='4h')
    return pd.DataFrame({'timestamp': ts, 'open': open_, 'high': high, 'low': low,
                         'close': close, 'volume': rng.random(n) * 1000 + 500})


def counting_signal(*a, **k):
    CALLS['signal'] += 1
    return _real_signal(*a, **k)


def fake_regime(window):
    return "Range UPTREND" if len(window) % 2 else "Trend DOWNTREND"


def counting_regime(window):
    CALLS['regime'] += 1
    return fake_regime(window)


bt.fetch_historical_data = fake_fetch
bt.predict_regime = counting_regime
bt.generate_trade_signal = counting_signal


def make(tmp, **kw):
    params = dict(coins=COINS, days=100, capital=5000.0, workers=1, data_dir=tmp,
                  results_dir=os.path.join(tmp, 'results'))
    params.update(kw)
    return bt.Backtester(**params)


def reset_calls():
    CALLS['fetch'] = CALLS['signal'] = CALLS['regime'] = 0


def test_snapshot_is_downloaded_once_then_reused():
    with tempfile.TemporaryDirectory() as tmp:
        reset_calls()
        make(tmp).run()
        assert CALLS['fetch'] == len(COINS)
        assert all(os.path.exists(os.path.join(tmp, f"{c.replace('/', '_')}_4h.csv")) for c in COINS)
        reset_calls()
        make(tmp).run()
        assert CALLS['fetch'] == 0                          # second run reads the snapshot
        reset_calls()
        make(tmp, refresh=True).run()
        assert CALLS['fetch'] == len(COINS)                 # --refresh re-downloads


def test_caches_reused_and_invalidated_independently():
    with tempfile.TemporaryDirectory() as tmp:
        reset_calls()
        t1 = make(tmp, log_experiment=False)
        t1.run()
        assert CALLS['signal'] > 0 and CALLS['regime'] > 0

        reset_calls()
        t2 = make(tmp, log_experiment=False)
        t2.run()
        assert CALLS['signal'] == 0 and CALLS['regime'] == 0   # everything from cache
        assert len(t1.trades) == len(t2.trades)                 # ...and identical results

        # Editing STRATEGY code: signals recompute, the expensive regimes do not
        real_sig_fp = bt.signal_fingerprint
        try:
            bt.signal_fingerprint = lambda cfg: 'changed-strategy'
            reset_calls()
            make(tmp, log_experiment=False).run()
            assert CALLS['signal'] > 0 and CALLS['regime'] == 0
        finally:
            bt.signal_fingerprint = real_sig_fp

        # Retraining the REGIME MODEL: both recompute (signals depend on regimes)
        real_reg_fp = bt.regime_fingerprint
        try:
            bt.regime_fingerprint = lambda cfg: 'retrained-model'
            reset_calls()
            make(tmp, log_experiment=False).run()
            assert CALLS['regime'] > 0 and CALLS['signal'] > 0
        finally:
            bt.regime_fingerprint = real_reg_fp


def test_side_filter_only_takes_that_side():
    with tempfile.TemporaryDirectory() as tmp:
        for side in ('long', 'short'):
            b = make(tmp, side=side, log_experiment=False)
            trades = b.run()
            assert trades and all(t['side'] == side for t in trades), side


def test_signal_include_and_exclude_filters():
    with tempfile.TemporaryDirectory() as tmp:
        inc = make(tmp, side='both', include=['macd'], log_experiment=False).run()
        assert inc and all('macd' in t['signal_type'] for t in inc)
        exc = make(tmp, side='both', exclude=['macd'], log_experiment=False).run()
        assert all('macd' not in t['signal_type'] for t in exc)


def test_dev_and_holdout_periods_are_disjoint_and_cover_the_window():
    with tempfile.TemporaryDirectory() as tmp:
        full = make(tmp, log_experiment=False)
        full.run()
        dev = make(tmp, period='dev', dev_frac=0.6, log_experiment=False)
        dev.run()
        hold = make(tmp, period='holdout', dev_frac=0.6, log_experiment=False)
        hold.run()
        assert dev.window[0] == full.window[0]
        assert hold.window[1] == full.window[1]
        assert dev.window[1] < hold.window[0]               # no overlap: holdout stays untouched
        assert all(t['entry_time'] <= dev.window[1] for t in dev.trades)
        assert all(t['entry_time'] >= hold.window[0] for t in hold.trades)


def test_btc_gate_blocks_longs_when_btc_is_below_its_ema():
    coins = ['BTC/USDT', 'AAA/USDT', 'BBB/USDT']
    days = 5                                               # 30 bars on 4h: toggles often
    with tempfile.TemporaryDirectory() as tmp:
        open_ = make(tmp, coins=coins, side='long', log_experiment=False)
        open_.run()
        gated = make(tmp, coins=coins, side='long', btc_gate_days=days, log_experiment=False)
        gated.run()
        assert gated.trades and len(gated.trades) < len(open_.trades)   # it blocked something

        btc = fake_fetch('BTC/USDT').set_index('timestamp')['close']
        span = days * 6
        up = btc > btc.ewm(span=span, adjust=False).mean()
        up.iloc[:span] = False
        for t in gated.trades:
            df = fake_fetch(t['symbol'])
            signal_ts = df['timestamp'].iloc[t['entry_idx'] - 1]      # the bar the signal fired on
            assert up.loc[signal_ts], f"long opened while BTC was below its EMA at {signal_ts}"


def test_gate_shorts_only_shorts_while_btc_is_below_its_ema():
    coins = ['BTC/USDT', 'AAA/USDT', 'BBB/USDT']
    days = 5
    with tempfile.TemporaryDirectory() as tmp:
        b = make(tmp, coins=coins, side='both', btc_gate_days=days, gate_shorts=True,
                 log_experiment=False)
        b.run()
        btc = fake_fetch('BTC/USDT').set_index('timestamp')['close']
        span = days * 6
        up = btc > btc.ewm(span=span, adjust=False).mean()
        up.iloc[:span] = False
        sides = set()
        for t in b.trades:
            signal_ts = fake_fetch(t['symbol'])['timestamp'].iloc[t['entry_idx'] - 1]
            sides.add(t['side'])
            assert up.loc[signal_ts] == (t['side'] == 'long'), \
                f"{t['side']} opened with BTC {'above' if up.loc[signal_ts] else 'below'} its EMA"
        assert sides == {'long', 'short'}, f"expected both sides to trade, got {sides}"


def test_btc_gate_needs_btc_in_the_coin_list():
    with tempfile.TemporaryDirectory() as tmp:
        try:
            make(tmp, btc_gate_days=5, log_experiment=False).run()
        except SystemExit:
            return
        raise AssertionError("expected SystemExit when BTC is missing")


def test_benchmark_buy_and_hold_is_computed():
    with tempfile.TemporaryDirectory() as tmp:
        b = make(tmp, log_experiment=False)
        b.run()
        assert b.benchmark and b.benchmark['coins'] == len(COINS)
        assert isinstance(b.benchmark['equal_weight_ret'], float)
        assert b.benchmark['equal_weight_mdd'] <= 0


def test_experiment_log_and_tagged_trades_are_written():
    with tempfile.TemporaryDirectory() as tmp:
        for tag in ('one', 'two'):
            b = make(tmp, tag=tag)
            b.run()
            b._summary(b.trades)
        log = pd.read_csv(os.path.join(tmp, 'results', 'experiments.csv'))
        assert list(log['tag']) == ['one', 'two']
        assert {'net_pnl', 'pf', 'oos_pnl', 'code'} <= set(log.columns)
        assert os.path.exists(os.path.join(tmp, 'results', 'one_trades.csv'))


def test_precomputed_sizing_matches_direct_sizing_when_rescaled():
    """The cache stores units computed at a nominal equity and rescales them. That is only valid
    if position size is exactly linear in equity - verify against direct calls."""
    df = fake_fetch('AAA/USDT')
    checked = 0
    for idx in range(bt.WARMUP, len(df) - 1, 3):
        window = df.iloc[max(0, idx - bt.SIGNAL_WINDOW + 1): idx + 1].copy()
        regime = fake_regime(window)
        big = _real_signal(df=window, equity=bt.PRECOMPUTE_EQUITY, risk_per_trade=0.03,
                           symbol='AAA/USDT', regime=regime)
        direct = _real_signal(df=window, equity=50_000.0, risk_per_trade=0.03,
                              symbol='AAA/USDT', regime=regime)
        if big is None or direct is None:
            assert not (big and not direct and big['units'] * 50_000 / bt.PRECOMPUTE_EQUITY * big['entry_price'] >= 10), \
                "signal vanished at real equity although the rescaled size is above the minimum"
            continue
        scaled = big['units'] * 50_000.0 / bt.PRECOMPUTE_EQUITY
        assert abs(scaled - direct['units']) / direct['units'] < 1e-9
        assert big['side'] == direct['side'] and abs(big['stop_loss'] - direct['stop_loss']) < 1e-9
        checked += 1
    assert checked >= 3, f"only {checked} comparable signals found"


def test_pnl_scales_with_capital():
    """Bigger account, same trades: PnL should scale ~linearly (no size-related rejections)."""
    with tempfile.TemporaryDirectory() as tmp:
        small = make(tmp, capital=5000.0, log_experiment=False)
        small.run()
        big = make(tmp, capital=50000.0, log_experiment=False)
        big.run()
        assert len(small.trades) == len(big.trades)
        s, g = sum(t['net_pnl'] for t in small.trades), sum(t['net_pnl'] for t in big.trades)
        assert abs(g / s - 10.0) < 0.05 if abs(s) > 1e-6 else abs(g) < 1e-3


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
