"""
The live futures path has no exchange-side stop loss, so live futures orders must be blocked
unless futures_live_enabled is explicitly true.

Run:  python tests/test_futures_guard.py     (or: pytest tests/test_futures_guard.py)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import modules.futures_engine as fe


class ExplodingClient:
    """Any exchange call is a test failure."""
    def __getattr__(self, name):
        raise AssertionError(f"exchange was called ({name}) although live futures are disabled")


def test_live_futures_blocked_by_default():
    fe.get_futures_client = lambda: ExplodingClient()
    fe.CONFIG.pop('futures_live_enabled', None)
    assert fe._live_open('BTC/USDC', 'short', 0.01, 70000.0, 60000.0) is False


def test_live_futures_blocked_when_flag_false():
    fe.get_futures_client = lambda: ExplodingClient()
    fe.CONFIG['futures_live_enabled'] = False
    assert fe._live_open('BTC/USDC', 'short', 0.01, 70000.0, 60000.0) is False
    assert fe.futures_engine.open_short('BTC/USDC', 0.01, 65000.0, 70000.0, 60000.0) in (False, None) \
        or fe.futures_engine.trading_mode == 'paper'          # paper mode is unaffected


def test_opt_in_reaches_the_exchange_client():
    calls = []

    class Recorder:
        def change_leverage(self, **kw):
            calls.append(('leverage', kw))

        def new_order(self, **kw):
            calls.append(('order', kw))
            raise RuntimeError("stop here - we only wanted to see that it got this far")

    fe.get_futures_client = lambda: Recorder()
    fe.CONFIG['futures_live_enabled'] = True
    try:
        assert fe._live_open('BTC/USDC', 'short', 0.01, 70000.0, 60000.0) is False   # our RuntimeError
        assert any(k == 'order' for k, _ in calls)
    finally:
        fe.CONFIG['futures_live_enabled'] = False


if __name__ == '__main__':
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith('test_') and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as e:
            failed += 1
            print(f"FAIL  {name}: {e!r}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
