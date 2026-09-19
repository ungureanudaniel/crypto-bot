"""
Offline tests for TradingEngine.check_drawdown() (peak-based circuit breaker).

Run:  python tests/test_circuit_breaker.py     (or: pytest tests/test_circuit_breaker.py)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test_limit_orders import make_engine

HOUR = 3600


def breaker_engine(equity=1000.0):
    eng, ex, clock = make_engine()
    eng.circuit_breaker_triggered = False
    eng.circuit_breaker_time = None
    eng._equity_peak = None
    eng._equity_cache = (0.0, None)
    eng._save_breaker_state = lambda: None
    eng.eq = {'v': equity}
    eng._current_equity = lambda: eng.eq['v']
    eng.config = {'max_drawdown': 0.05}
    return eng, clock


def test_trip_sets_time_and_repeat_calls_do_not_crash():
    """Regression: circuit_breaker_time was never set on trip -> TypeError on the next call."""
    eng, clock = breaker_engine(1000)
    assert eng.check_drawdown() is True
    eng.eq['v'] = 940                                      # -6% from peak
    assert eng.check_drawdown() is False
    assert eng.circuit_breaker_time is not None
    for _ in range(5):                                     # used to raise TypeError here
        clock.advance(300)
        assert eng.check_drawdown() is False


def test_no_trip_within_limit():
    eng, clock = breaker_engine(1000)
    eng.check_drawdown()
    eng.eq['v'] = 960                                      # -4%
    assert eng.check_drawdown() is True


def test_drawdown_is_measured_from_peak_not_start():
    eng, clock = breaker_engine(1000)
    eng.check_drawdown()
    eng.eq['v'] = 1200                                     # new peak
    assert eng.check_drawdown() is True
    eng.eq['v'] = 1130                                     # -5.8% from the 1200 peak, still +13% overall
    assert eng.check_drawdown() is False


def test_resets_on_recovery_with_hysteresis():
    eng, clock = breaker_engine(1000)
    eng.check_drawdown()
    eng.eq['v'] = 940
    assert eng.check_drawdown() is False
    eng.eq['v'] = 965                                      # -3.5%: below 5% but not below 4% (0.8 x 5%)? it is < 4%
    assert eng.check_drawdown() is True                    # recovered under 4% -> reset
    eng.eq['v'] = 940
    assert eng.check_drawdown() is False                   # trips again
    eng.eq['v'] = 955                                      # -4.5%: better, but still above 4% -> stays paused
    assert eng.check_drawdown() is False


def test_timed_reset_rebases_so_it_does_not_retrip():
    eng, clock = breaker_engine(1000)
    eng.check_drawdown()
    eng.eq['v'] = 900                                      # -10%, and it stays there
    assert eng.check_drawdown() is False
    clock.advance(47 * HOUR)
    assert eng.check_drawdown() is False                   # cooldown not over
    clock.advance(2 * HOUR)
    assert eng.check_drawdown() is True                    # 49h: reset
    assert eng._equity_peak == 900                         # rebased to the current equity
    assert eng.check_drawdown() is True                    # same loss must NOT re-trip
    eng.eq['v'] = 850                                      # a NEW 5.6% drop from the new base does
    assert eng.check_drawdown() is False


def test_unmeasurable_equity_keeps_current_state():
    eng, clock = breaker_engine(1000)
    eng.check_drawdown()
    eng.eq['v'] = None
    assert eng.check_drawdown() is True                    # API hiccup must not trip it
    eng.eq['v'] = 900
    eng.check_drawdown()
    eng.eq['v'] = None
    assert eng.check_drawdown() is False                   # ...nor silently release it


def test_unfunded_account_does_not_trip():
    """The old code read a stale portfolio.json (cash 0, initial 9) -> -100% -> tripped instantly."""
    eng, clock = breaker_engine(0.0)
    assert eng.check_drawdown() is True
    assert eng.circuit_breaker_triggered is False


def test_manual_reset_rebases_peak():
    eng, clock = breaker_engine(1000)
    eng.check_drawdown()
    eng.eq['v'] = 900
    assert eng.check_drawdown() is False
    eng.reset_circuit_breaker(rebase=True)
    assert eng.check_drawdown() is True                    # not re-tripped by the old peak
    assert eng._equity_peak == 900


def test_custom_config_values_are_used():
    eng, clock = breaker_engine(1000)
    eng.config = {'max_drawdown': 0.10, 'circuit_breaker_cooldown_hours': 1,
                  'circuit_breaker_reset_ratio': 0.5}
    eng.check_drawdown()
    eng.eq['v'] = 930                                      # -7%: under a 10% limit
    assert eng.check_drawdown() is True
    eng.eq['v'] = 880                                      # -12%
    assert eng.check_drawdown() is False
    clock.advance(1.1 * HOUR)
    assert eng.check_drawdown() is True                    # 1h cooldown from config


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
