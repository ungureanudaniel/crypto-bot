"""
Offline tests for the testnet-readiness changes:
  - the engine only trades symbols that exist on the connected exchange
  - the scheduler's summary / health jobs use the exchange account, not portfolio.json
  - config_loader accepts an HMAC testnet key (as well as the Ed25519/RSA private key)

Run:  python tests/test_testnet_prep.py     (or: pytest tests/test_testnet_prep.py)
"""
import os
import sys
import tempfile
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test_limit_orders import make_engine
import config_loader


# ---------------------------------------------------------------- symbol filtering
class InfoClient:
    def __init__(self, symbols=None, fail=False):
        self.symbols, self.fail = symbols or [], fail

    def get_exchange_info(self):
        if self.fail:
            raise RuntimeError("exchange unreachable")
        return {'symbols': self.symbols}


def test_engine_keeps_only_symbols_that_are_trading_on_the_exchange():
    eng, ex, clock = make_engine()
    eng.binance_client = InfoClient([{'symbol': 'BTCUSDC', 'status': 'TRADING'},
                                     {'symbol': 'ETHUSDC', 'status': 'BREAK'}])
    assert eng._tradable_symbols(['BTC/USDC', 'ETH/USDC', 'XMR/USDT']) == ['BTC/USDC']


def test_engine_keeps_all_symbols_if_the_exchange_cannot_be_asked():
    eng, ex, clock = make_engine()
    eng.binance_client = InfoClient(fail=True)
    assert eng._tradable_symbols(['BTC/USDC', 'XMR/USDT']) == ['BTC/USDC', 'XMR/USDT']


# ---------------------------------------------------------------- scheduler summary / health
def _stub_engine(mode='testnet', equity=950.0, peak=1000.0):
    return types.SimpleNamespace(
        trading_mode=mode, _current_equity=lambda: equity, _equity_peak=peak,
        symbols=['BTC/USDC', 'ETH/USDC'], get_cash_balance=lambda q: 400.0,
        open_positions={'BTC/USDC': {}}, open_futures_positions={}, pending_entries={'ETH/USDC': {}},
        check_drawdown=lambda: True, shorts_enabled=False)


class swapped:
    """Temporarily replace attributes on a module."""
    def __init__(self, module, **attrs):
        self.module, self.attrs, self.saved = module, attrs, {}

    def __enter__(self):
        for k, v in self.attrs.items():
            self.saved[k] = getattr(self.module, k)
            setattr(self.module, k, v)

    def __exit__(self, *exc):
        for k, v in self.saved.items():
            setattr(self.module, k, v)


def test_account_view_reads_the_exchange_not_portfolio_json():
    import services.scheduler as sched
    with swapped(sched, trading_engine=_stub_engine()):
        v = sched._live_account_view()
        assert v['total_value'] == 950.0 and v['total_cash'] == 400.0
        assert abs(v['drawdown_pct'] - 5.0) < 1e-9 and v['positions_count'] == 2
    with swapped(sched, trading_engine=_stub_engine(mode='paper')):
        assert sched._live_account_view() is None                      # paper keeps the old behaviour
    with swapped(sched, trading_engine=_stub_engine(equity=None)):
        assert sched._live_account_view() is None                      # equity unavailable -> fall back


def test_health_check_ignores_stale_portfolio_numbers_on_testnet():
    """portfolio.json says -100% (cash 0) in live/testnet; that must not raise a health alert."""
    import services.scheduler as sched
    import services.notifier as notifier_mod
    stale = lambda *a, **k: {'total_return_pct': -100.0, 'total_value': 0.0}
    warned = []
    saved_token, saved_warn = notifier_mod.notifier.token, sched.logger.warning
    notifier_mod.notifier.token = ''                                    # never send a real Telegram message
    sched.logger.warning = lambda *a, **k: warned.append(a)
    try:
        with swapped(sched, get_portfolio_summary=stale, trading_engine=_stub_engine(equity=1000.0, peak=1000.0)):
            sched.health_check()
        assert warned == [], f"false alert on a healthy testnet account: {warned}"

        warned.clear()
        with swapped(sched, get_portfolio_summary=stale, trading_engine=_stub_engine(equity=850.0, peak=1000.0)):
            sched.health_check()                                        # a REAL 15% drawdown does alert
        assert warned, "a real drawdown must still raise the alert"

        warned.clear()
        with swapped(sched, get_portfolio_summary=stale, trading_engine=_stub_engine(mode='paper')):
            sched.health_check()                                        # paper: unchanged behaviour
        assert warned
    finally:
        notifier_mod.notifier.token, sched.logger.warning = saved_token, saved_warn


# ---------------------------------------------------------------- config_loader
def load_config(**env):
    saved = {k: os.environ.get(k) for k in env}
    try:
        for k, v in env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        return config_loader.Config().config
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_testnet_hmac_key_is_accepted():
    cfg = load_config(TRADING_MODE='testnet', BINANCE_TESTNET_API_KEY='K', BINANCE_TESTNET_SECRET_KEY='S',
                      BINANCE_TESTNET_PRIVATE_KEY=None)
    assert cfg['auth_method'] == 'hmac' and cfg['binance_api_key'] == 'K' and cfg['binance_api_secret'] == 'S'


def test_testnet_private_key_file_still_works():
    with tempfile.NamedTemporaryFile('w', suffix='.pem', delete=False) as f:
        f.write('PEM-CONTENT')
    try:
        cfg = load_config(TRADING_MODE='testnet', BINANCE_TESTNET_API_KEY='K', BINANCE_TESTNET_SECRET_KEY=None,
                          BINANCE_TESTNET_PRIVATE_KEY=f.name)
        assert cfg['auth_method'] == 'rsa' and cfg['rsa_private_key'] == 'PEM-CONTENT'
    finally:
        os.unlink(f.name)


def test_testnet_without_any_secret_is_rejected():
    try:
        load_config(TRADING_MODE='testnet', BINANCE_TESTNET_API_KEY='K', BINANCE_TESTNET_SECRET_KEY=None,
                    BINANCE_TESTNET_PRIVATE_KEY=None)
    except ValueError as e:
        assert 'SECRET_KEY' in str(e) and 'PRIVATE_KEY' in str(e)
        return
    raise AssertionError("expected ValueError")


def test_hmac_testnet_client_is_created_with_key_and_secret():
    import binance.client as bc

    class FakeClient:
        made = []

        def __init__(self, *a, **k):
            self.a, self.k, self.API_URL = a, k, 'fake'
            FakeClient.made.append(self)

        def ping(self):
            pass

        def get_account(self):
            return {}

    saved_client, saved_cfg, saved_singleton = bc.Client, config_loader.config.config, config_loader._binance_client
    try:
        bc.Client = FakeClient
        config_loader._binance_client = None
        config_loader.config.config = {'trading_mode': 'testnet', 'auth_method': 'hmac',
                                       'binance_api_key': 'K', 'binance_api_secret': 'S'}
        client = config_loader.get_binance_client()
        assert client is FakeClient.made[0]
        assert client.a == ('K', 'S') and client.k.get('testnet') is True
    finally:
        bc.Client, config_loader.config.config, config_loader._binance_client = saved_client, saved_cfg, saved_singleton


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
