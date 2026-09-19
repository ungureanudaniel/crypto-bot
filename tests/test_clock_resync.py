"""
Offline tests for the Binance clock handling in modules/order_manager.py (-1021 timestamp errors).

Run:  python tests/test_clock_resync.py     (or: pytest tests/test_clock_resync.py)
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test_limit_orders import make_engine, SYMBOL, Err


def manager():
    eng, ex, clock = make_engine()
    ex.timestamp_offset = 0
    ex.server_ahead_ms = 2500
    ex.get_server_time = lambda: {'serverTime': int(time.time() * 1000) + ex.server_ahead_ms}
    return eng.order_manager, ex, clock


def test_timestamp_error_resyncs_the_clock_and_retries_once():
    om, ex, clock = manager()
    ex.bal['ETH'] = 1.0
    real, calls = ex.create_order, []

    def flaky(**kw):
        calls.append(kw)
        if len(calls) == 1:
            raise Err(-1021, "Timestamp for this request was 1000ms ahead of the server's time.")
        return real(**kw)

    ex.create_order = flaky
    res = om.place_stop(SYMBOL, 1.0, 97.0, ex.mid)
    assert res['ok'] and len(calls) == 2                       # failed once, retried once, succeeded
    assert 2000 < ex.timestamp_offset < 3000                   # offset measured from Binance's clock


def test_other_errors_are_not_retried_and_the_detail_is_complete():
    om, ex, clock = manager()
    ex.bal['ETH'] = 1.0
    calls = []

    def rejected(**kw):
        calls.append(kw)
        raise Err(-2010, 'Stop price would trigger immediately.')

    ex.create_order = rejected
    res = om.place_stop(SYMBOL, 1.0, 97.0, ex.mid)
    assert res['ok'] is False and res['error'] == 'api' and len(calls) == 1
    assert 'Err' in res['detail'] and '-2010' in res['detail'] and 'trigger immediately' in res['detail']


def test_a_second_timestamp_error_is_reported_not_looped():
    om, ex, clock = manager()
    ex.bal['ETH'] = 1.0
    calls = []

    def always(**kw):
        calls.append(kw)
        raise Err(-1021, 'Timestamp for this request is outside of the recvWindow.')

    ex.create_order = always
    res = om.place_stop(SYMBOL, 1.0, 97.0, ex.mid)
    assert res['ok'] is False and len(calls) == 2 and '-1021' in res['detail']


def test_periodic_sync_runs_on_schedule_and_backs_off_after_a_failure():
    om, ex, clock = manager()
    om.maybe_sync_clock()
    assert ex.timestamp_offset != 0                            # first call syncs
    ex.timestamp_offset = 0
    clock.advance(600)
    om.maybe_sync_clock()
    assert ex.timestamp_offset == 0                            # too soon
    clock.advance(1300)
    om.maybe_sync_clock()
    assert ex.timestamp_offset != 0                            # 30 minutes passed

    ex.get_server_time = lambda: (_ for _ in ()).throw(RuntimeError("unreachable"))
    ex.timestamp_offset = 0
    clock.advance(2000)
    om.maybe_sync_clock()                                      # fails quietly
    attempts = []
    ex.get_server_time = lambda: attempts.append(1) or {'serverTime': int(time.time() * 1000)}
    clock.advance(60)
    om.maybe_sync_clock()
    assert attempts == []                                      # backed off: not retried after 1 minute
    clock.advance(300)
    om.maybe_sync_clock()
    assert attempts == [1]                                     # retried after ~5 minutes


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
