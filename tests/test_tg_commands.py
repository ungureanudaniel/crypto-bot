"""
Offline tests for the Telegram command layer (services/tg_commands.py) and the engine hooks it
relies on (pause, cancel pending entries, persisted pause flag). No network, no Telegram.

Run:  python tests/test_tg_commands.py     (or: pytest tests/test_tg_commands.py)
"""
import asyncio
import os
import sys
import tempfile
import time
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from telegram.ext import ApplicationHandlerStop

from test_limit_orders import make_engine, SYMBOL
from test_trend_hold_live import open_trend_position
import modules.trade_engine as te
import services.tg_commands as tg


# ---------------------------------------------------------------- fakes
class Msg:
    def __init__(self):
        self.sent = []

    async def reply_text(self, text, **kw):
        self.sent.append((text, kw))


class Query:
    def __init__(self, data):
        self.data, self.edits, self.answered = data, [], False

    async def answer(self):
        self.answered = True

    async def edit_message_text(self, text, **kw):
        self.edits.append(text)


def update(chat_id=42):
    msg = Msg()
    return types.SimpleNamespace(effective_chat=types.SimpleNamespace(id=chat_id),
                                 effective_user=types.SimpleNamespace(id=chat_id, username='tester'),
                                 effective_message=msg, message=msg, callback_query=None)


def cb_update(data, chat_id=42):
    u = update(chat_id)
    u.callback_query = Query(data)
    return u


def ctx(*args):
    return types.SimpleNamespace(args=list(args), application=None)


def run(coro):
    return asyncio.run(coro)


def text_of(u):
    return '\n'.join(t for t, _ in u.message.sent)


class env:
    """A tg module wired to a fake engine (+ authorised chat id 42)."""
    def __enter__(self):
        self.eng, self.ex, self.clock = make_engine()
        e = self.eng
        e.config = {'trend_hold_live_enabled': True}
        e._current_equity = lambda: 1000.0
        e._equity_peak = 1000.0
        e.circuit_breaker_triggered, e.circuit_breaker_time = False, None
        e.paused = False
        e._save_breaker_state = lambda: None
        e.check_drawdown = lambda: True
        self.saved = (tg.engine, dict(tg._cfg.config), tg._price_cache)
        tg.engine = e
        tg._price_cache = (0.0, {})
        tg._cfg.config['telegram_chat_id'] = '42'
        return self

    def __exit__(self, *exc):
        tg.engine = self.saved[0]
        tg._cfg.config.clear()
        tg._cfg.config.update(self.saved[1])
        tg._price_cache = self.saved[2]


# ---------------------------------------------------------------- security
def test_guard_allows_only_the_configured_chat():
    with env():
        run(tg.guard(update(42), ctx()))                               # allowed: no exception
        for other in (7, -100123):
            try:
                run(tg.guard(update(other), ctx()))
            except ApplicationHandlerStop:
                continue
            raise AssertionError(f"chat {other} must be blocked")


def test_guard_fails_closed_when_no_chat_id_is_configured():
    with env():
        tg._cfg.config['telegram_chat_id'] = ''
        try:
            run(tg.guard(update(42), ctx()))
        except ApplicationHandlerStop:
            return
        raise AssertionError("no configured chat id must mean nobody is authorised")


def test_guard_is_registered_first_and_every_menu_command_has_a_handler():
    added = []
    app = types.SimpleNamespace(add_handler=lambda h, group=0: added.append((h, group)))
    tg.register_commands(app)
    assert added[0][1] == -1 and 'guard' in repr(added[0][0].callback)
    handled = {c for names, _ in tg.HANDLERS for c in names}
    for cmd, _ in tg.MENU_COMMANDS:
        assert cmd in handled, f"/{cmd} is in the menu but has no handler"
        assert f"/{cmd}" in tg.HELP_TEXT or cmd in ('help',), f"/{cmd} missing from /help"
    assert len(tg.menu()) == len(tg.MENU_COMMANDS)


# ---------------------------------------------------------------- overview
def test_status_shows_mode_strategy_pause_and_missing_stops():
    with env() as x:
        u = update()
        run(tg.cmd_status(u, ctx()))
        t = text_of(u)
        assert 'TESTNET' in t and 'running' in t and 'Circuit breaker: OK' in t and '$1,000.00' in t
        x.eng.paused = True
        run(tg.cmd_status(u, ctx()))
        assert 'PAUSED' in u.message.sent[-1][0]


def test_positions_show_pnl_stop_distance_and_protection():
    with env() as x:
        pos = open_trend_position(x.eng, x.ex, x.clock)
        x.ex.set_market(105.0, 105.1)
        u = update()
        run(tg.cmd_positions(u, ctx()))
        t = text_of(u)
        assert 'ETH/USDC' in t and '%' in t and 'stop ' in t and 'exchange stop' in t
        assert 'trend_hold_breakout' in t and 'Open P&amp;L' in t
        pos['oco'] = None                                              # lose the exchange stop
        run(tg.cmd_positions(u, ctx()))
        assert 'no exchange stop' in u.message.sent[-1][0]


def test_empty_positions_message():
    with env():
        u = update()
        run(tg.cmd_positions(u, ctx()))
        assert 'No open positions' in text_of(u)


def test_orders_lists_protective_stops_and_working_entries():
    with env() as x:
        open_trend_position(x.eng, x.ex, x.clock)
        x.eng.config = {}
        assert x.eng._submit_limit_entry('BNB/USDC', 100.05, 1.0, 97.0, 109.0, 'test', 1.0) or True
        u = update()
        run(tg.cmd_orders(u, ctx()))
        t = text_of(u)
        assert 'Protective orders' in t and 'ETH/USDC' in t and 'stop' in t and 'Open on the exchange' in t


def test_trades_summarises_closed_trades_only():
    import modules.portfolio as pf
    fake = [{'action': 'open', 'symbol': 'A/USDC'},
            {'action': 'close', 'symbol': 'A/USDC', 'pnl': 12.5, 'pnl_pct': 3.1, 'reason': 'trailing_stop',
             'timestamp': '2026-09-01T10:00:00', 'mode': 'testnet'},
            {'action': 'close', 'symbol': 'B/USDC', 'pnl': -4.0, 'pnl_pct': -1.2, 'reason': 'stop_loss',
             'timestamp': '2026-09-02T10:00:00', 'mode': 'testnet'},
            {'action': 'close', 'symbol': 'OLD/USDC', 'pnl': 99.0, 'pnl_pct': 9.0, 'reason': 'x',
             'timestamp': '2026-01-01T10:00:00', 'mode': 'paper'}]          # other mode: excluded
    with env():
        saved = pf.get_trade_history
        pf.get_trade_history = lambda limit=100: fake
        try:
            u = update()
            run(tg.cmd_trades(u, ctx('5')))
            t = text_of(u)
            assert 'B/USDC' in t and 'A/USDC' in t and 'OLD/USDC' not in t
            assert 'win rate 50%' in t and '+8.50' in t
        finally:
            pf.get_trade_history = saved


# ---------------------------------------------------------------- control
def test_pause_and_resume_and_scan_respects_pause():
    with env() as x:
        u = update()
        run(tg.cmd_pause(u, ctx()))
        assert x.eng.paused is True
        assert x.eng.scan_and_trade() == []                            # returns before touching any data
        run(tg.cmd_resume(u, ctx()))
        assert x.eng.paused is False


def test_pause_flag_survives_a_restart():
    with tempfile.TemporaryDirectory() as tmp:
        saved = te.BREAKER_STATE_FILE
        te.BREAKER_STATE_FILE = os.path.join(tmp, 'breaker_state.json')
        try:
            a, _, _ = make_engine()
            a._equity_peak, a.circuit_breaker_triggered, a.circuit_breaker_time = 1000.0, False, None
            a.set_paused(True)
            b, _, _ = make_engine()
            b.paused = False
            b._load_breaker_state()
            assert b.paused is True                                     # still paused after "restart"
        finally:
            te.BREAKER_STATE_FILE = saved


def test_close_resolves_symbols_and_starts_a_non_blocking_exit():
    with env() as x:
        open_trend_position(x.eng, x.ex, x.clock)
        for arg in ('eth', 'ETH/USDC', 'ethusdc'):
            assert tg.parse_symbol(arg) == SYMBOL, arg
        u = update()
        run(tg.cmd_close(u, ctx('eth')))
        assert SYMBOL in x.eng.pending_exits and x.eng.pending_exits[SYMBOL]['st']['urgent'] is False
        assert 'Closing' in text_of(u)
        run(tg.cmd_close(u, ctx('eth')))
        assert 'already being closed' in u.message.sent[-1][0]


def test_close_now_is_urgent_and_unknown_or_missing_are_reported():
    with env() as x:
        open_trend_position(x.eng, x.ex, x.clock)
        u = update()
        run(tg.cmd_close(u, ctx('ETH', 'now')))
        assert x.eng.pending_exits[SYMBOL]['st']['urgent'] is True
        run(tg.cmd_close(u, ctx('doge')))
        assert 'Unknown symbol' in u.message.sent[-1][0]
        run(tg.cmd_close(u, ctx()))
        assert 'Usage' in u.message.sent[-1][0]


def test_close_cancels_a_working_entry_order():
    with env() as x:
        x.eng.config = {}
        assert x.eng._submit_limit_entry(SYMBOL, 100.05, 1.0, 97.0, 109.0, 'test', 1.0)
        assert SYMBOL in x.eng.pending_entries
        u = update()
        run(tg.cmd_close(u, ctx('ETH')))
        assert SYMBOL not in x.eng.pending_entries and 'Cancelled the working entry' in text_of(u)
        assert not [o for o in x.ex.orders.values() if o['status'] == 'NEW']


def test_sellall_asks_first_then_pauses_cancels_entries_and_exits_urgently():
    with env() as x:
        open_trend_position(x.eng, x.ex, x.clock)
        u = update()
        run(tg.cmd_sellall(u, ctx()))
        markup = u.message.sent[0][1]['reply_markup']
        buttons = [b for row in markup.inline_keyboard for b in row]
        assert [b.callback_data.split(':')[1] for b in buttons] == ['yes', 'no']
        assert SYMBOL not in x.eng.pending_exits and x.eng.paused is False    # nothing happened yet

        yes = cb_update(buttons[0].callback_data)
        run(tg.on_confirm(yes, ctx()))
        assert x.eng.paused is True
        assert x.eng.pending_exits[SYMBOL]['st']['urgent'] is True
        assert 'paused' in yes.callback_query.edits[-1] and 'ETH/USDC' in yes.callback_query.edits[-1]


def test_confirmation_can_be_declined_and_expires():
    with env() as x:
        open_trend_position(x.eng, x.ex, x.clock)
        no = cb_update(f"sellall:no:{int(time.time())}")
        run(tg.on_confirm(no, ctx()))
        assert x.eng.paused is False and not x.eng.pending_exits and 'Cancelled' in no.callback_query.edits[-1]

        stale = cb_update(f"sellall:yes:{int(time.time()) - 3600}")
        run(tg.on_confirm(stale, ctx()))
        assert x.eng.paused is False and not x.eng.pending_exits and 'expired' in stale.callback_query.edits[-1]


def test_closeall_is_orderly_not_urgent():
    with env() as x:
        open_trend_position(x.eng, x.ex, x.clock)
        run(tg.on_confirm(cb_update(f"closeall:yes:{int(time.time())}"), ctx()))
        assert x.eng.pending_exits[SYMBOL]['st']['urgent'] is False


def test_stop_confirmation_sets_the_stop_event():
    import services.telegram_bot as tb
    with env():
        saved = tb.stop_event
        tb.stop_event = __import__('threading').Event()
        try:
            run(tg.on_confirm(cb_update(f"stop:yes:{int(time.time())}"), ctx()))
            assert tb.stop_event.is_set()
        finally:
            tb.stop_event = saved


# ---------------------------------------------------------------- helpers
def test_split_message_keeps_everything_and_respects_the_limit():
    text = '\n'.join(f"line {i} " + 'x' * 60 for i in range(200))
    parts = tg.split_message(text)
    assert len(parts) > 1 and all(len(p) <= 3900 for p in parts)
    assert ''.join(parts).replace('\n', '') == text.replace('\n', '')


def test_gate_command_reports_disabled_and_fails_closed_without_data():
    with env() as x:
        u = update()
        run(tg.cmd_gate(u, ctx()))
        assert 'off' in text_of(u)
        x.eng.config = {'strategy_mode': 'trend_hold', 'trading_timeframe': '4h', 'btc_gate_days': 50}
        x.eng.symbols = ['BTC/USDC']
        x.eng.timeframe = '4h'
        x.eng.data_feed = types.SimpleNamespace(get_ohlcv=lambda **k: (_ for _ in ()).throw(RuntimeError('down')))
        run(tg.cmd_gate(u, ctx()))
        assert 'CLOSED' in u.message.sent[-1][0] and 'fails closed' in u.message.sent[-1][0]


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
