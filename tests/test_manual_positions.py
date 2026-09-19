"""
Offline tests for MANUAL trading support: the bot adopts coins the user buys by hand, puts an
exchange-side stop under them, follows partial/full manual sells, and keeps automatic entries rare.

Run:  python tests/test_manual_positions.py     (or: pytest tests/test_manual_positions.py)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test_limit_orders import make_engine, SYMBOL
from test_trend_hold import candles
from test_trend_hold_live import FakeFeed, live_orders, scan_env
from test_tg_commands import update, ctx, run, text_of, cb_update
from test_trend_hold import flat_then_break
import modules.trade_engine as te
import services.tg_commands as tg
from modules import trend_hold as th


def manual_engine(baseline_now=True):
    """Engine on the fake exchange with a flat market (=> the default 3% minimum stop distance)."""
    eng, ex, clock = make_engine()
    eng.config = {'trading_timeframe': '4h'}
    eng.timeframe, eng.symbols = '4h', [SYMBOL]
    eng._manual_baseline, eng._manual_checked_at = None, 0.0
    eng._save_manual_state = lambda: None
    eng.data_feed = FakeFeed({SYMBOL: candles(np.full(400, 100.0))})
    eng.notes = []
    eng._notify = eng.notes.append
    if baseline_now:
        eng.sync_manual_positions(force=True)          # first run: records what is already in the account
    return eng, ex, clock


def stop_orders(ex):
    return live_orders(ex, 'STOP_LOSS_LIMIT')


# ---------------------------------------------------------------- adoption
def test_coins_already_in_the_account_are_listed_but_left_alone():
    eng, ex, clock = manual_engine(baseline_now=False)
    ex.bal['ETH'] = 2.0                                               # ~$200 of long-term coins
    eng.sync_manual_positions(force=True)
    assert not eng.open_positions and not ex.orders                   # nothing adopted, no order placed
    assert eng._manual_baseline == {SYMBOL: 2.0}
    assert any('do NOT manage' in n and 'ETH' in n and '/adopt' in n for n in eng.notes)
    eng.sync_manual_positions(force=True)                             # and it stays that way
    assert not eng.open_positions


def test_a_new_manual_buy_is_adopted_with_entry_estimate_and_exchange_stop():
    eng, ex, clock = manual_engine()
    ex.user_buy(1.0, 99.0)                                            # user buys on the exchange
    eng.sync_manual_positions(force=True)
    pos = eng.open_positions[SYMBOL]
    assert pos['signal_type'] == 'manual_adopted' and th.is_trend_hold(pos)
    assert abs(pos['amount'] - 0.999) < 1e-9                          # what is really held (fee taken in ETH)
    assert abs(pos['entry_price'] - 99.0) < 1e-6                      # worked out from the trade history
    assert pos['take_profit'] == 0.0 and pos['oco']['kind'] == 'stop'
    (order,) = stop_orders(ex)
    assert order['stop'] < ex.mid and abs(order['stop'] - 100.05 * 0.97) < 0.02      # 3% under the CURRENT price
    assert any('Adopted' in n for n in eng.notes)


def test_dust_ignored_coins_and_disabled_adoption_are_not_adopted():
    eng, ex, clock = manual_engine()
    ex.user_buy(0.05, 100.0)                                          # ~$5, below the $15 minimum
    eng.sync_manual_positions(force=True)
    assert not eng.open_positions

    eng, ex, clock = manual_engine()
    eng.config['adopt_ignore'] = ['ETH']
    ex.user_buy(1.0, 100.0)
    eng.sync_manual_positions(force=True)
    assert not eng.open_positions

    eng, ex, clock = manual_engine()
    eng.config['adopt_manual_positions'] = False
    ex.user_buy(1.0, 100.0)
    eng.sync_manual_positions(force=True)
    assert not eng.open_positions and not stop_orders(ex)             # (the user's own buy is an order too)


def test_an_already_losing_manual_position_is_not_sold_on_adoption():
    """Bought at 130, now 100: the stop is set from the current price, so nothing is sold instantly."""
    eng, ex, clock = manual_engine()
    ex.user_buy(1.0, 130.0)
    eng.sync_manual_positions(force=True)
    pos = eng.open_positions[SYMBOL]
    assert pos['entry_price'] > 129 and pos['stop_loss'] < ex.mid
    assert not eng.pending_exits and stop_orders(ex)                  # protected, not liquidated


def test_more_coins_on_top_of_a_managed_position_are_merged_and_the_stop_replaced():
    eng, ex, clock = manual_engine()
    ex.user_buy(1.0, 99.0)
    eng.sync_manual_positions(force=True)
    ex.user_buy(1.0, 101.0)
    eng.sync_manual_positions(force=True)
    pos = eng.open_positions[SYMBOL]
    assert abs(pos['amount'] - 1.998) < 1e-9 and abs(pos['entry_price'] - 100.0) < 0.01
    (order,) = stop_orders(ex)                                        # exactly one stop, covering everything
    assert abs(order['qty'] - 1.998) < 1e-9


# ---------------------------------------------------------------- the user sells by hand
def test_selling_everything_by_hand_closes_the_record():
    eng, ex, clock = manual_engine()
    ex.user_buy(1.0, 99.0)
    eng.sync_manual_positions(force=True)
    (order,) = stop_orders(ex)
    ex.cancel_order('ETHUSDC', order['orderId'])                      # user cancels the bot's stop ...
    ex.user_sell(0.999)                                               # ... and sells
    eng.sync_manual_positions(force=True)
    assert SYMBOL not in eng.open_positions
    assert eng.trades[-1]['reason'] == 'manual_sell'
    assert not stop_orders(ex)


def test_selling_part_by_hand_reduces_the_position_and_replaces_the_stop():
    eng, ex, clock = manual_engine()
    ex.user_buy(1.0, 99.0)
    eng.sync_manual_positions(force=True)
    (order,) = stop_orders(ex)
    ex.cancel_order('ETHUSDC', order['orderId'])
    ex.user_sell(0.5)
    eng.sync_manual_positions(force=True)
    pos = eng.open_positions[SYMBOL]
    assert abs(pos['amount'] - 0.499) < 1e-9
    (new,) = stop_orders(ex)
    assert abs(new['qty'] - 0.499) < 1e-9


def test_a_stop_cancelled_by_the_user_is_not_put_back_immediately():
    eng, ex, clock = manual_engine()
    ex.user_buy(1.0, 99.0)
    eng.sync_manual_positions(force=True)
    (order,) = stop_orders(ex)
    ex.cancel_order('ETHUSDC', order['orderId'])                      # user wants to sell manually
    eng.manage_orders()
    assert not stop_orders(ex) and any('cancelled outside the bot' in n for n in eng.notes)
    clock.advance(120)
    eng.manage_orders()
    assert not stop_orders(ex)                                        # still inside the 5 minute window
    clock.advance(200)
    eng.manage_orders()
    assert len(stop_orders(ex)) == 1                                  # window over: protected again


# ---------------------------------------------------------------- same trailing rule, no take-profit
def test_manual_positions_are_trailed_up_and_have_no_take_profit():
    from modules.exit_manager import evaluate_exit
    pos = {'signal_type': 'manual_adopted', 'side': 'long', 'stop_loss': 95.0, 'entry_price': 100.0,
           'take_profit': 0.0, 'atr': 1.0, 'candles_held': 50}
    df = candles(np.linspace(100, 160, 100))
    exit_, _ = evaluate_exit(SYMBOL, pos, 159.0, df)                  # +59%: a take-profit would have sold long ago
    assert exit_ is False and pos['stop_loss'] > 95.0                 # the stop only moved UP


# ---------------------------------------------------------------- automatic entries stay rare
def test_automatic_entries_respect_the_cap_and_the_off_switch():
    with scan_env(flat_then_break(n_flat=420)) as eng:
        te.config.config['auto_max_positions'] = 1
        eng.open_positions['BNB/USDC'] = {'signal_type': 'manual', 'side': 'long', 'amount': 1.0}
        assert len([s for s in eng.scan_and_trade() if s['symbol'] == 'ETH/USDC']) == 1   # manual ones don't count

        eng.last_signals.clear()
        eng.open_positions['ADA/USDC'] = {'signal_type': 'trend_hold_breakout', 'side': 'long', 'amount': 1.0}
        assert eng.scan_and_trade() == []                                                # 1 automatic = at the cap

        del eng.open_positions['ADA/USDC']
        te.config.config['auto_entries'] = False
        assert eng.scan_and_trade() == []


# ---------------------------------------------------------------- Telegram
class with_engine:
    def __init__(self, eng):
        self.eng = eng

    def __enter__(self):
        self.saved = (tg.engine, dict(tg._cfg.config), tg._price_cache)
        tg.engine = self.eng
        tg._price_cache = (0.0, {})
        tg._cfg.config['telegram_chat_id'] = '42'
        e = self.eng
        e._current_equity, e._equity_peak = (lambda: 1000.0), 1000.0
        e.circuit_breaker_triggered, e.circuit_breaker_time, e.paused = False, None, False
        e._save_breaker_state = lambda: None
        return e

    def __exit__(self, *exc):
        tg.engine = self.saved[0]
        tg._cfg.config.clear()
        tg._cfg.config.update(self.saved[1])
        tg._price_cache = self.saved[2]


def test_holdings_adopt_and_ignore_commands():
    eng, ex, clock = manual_engine(baseline_now=False)
    ex.bal['ETH'] = 2.0
    eng.sync_manual_positions(force=True)                             # legacy holdings, left alone
    with with_engine(eng):
        u = update()
        run(tg.cmd_holdings(u, ctx()))
        assert 'not managed' in text_of(u) and '/adopt ETH' in text_of(u)
        run(tg.cmd_adopt(u, ctx('eth')))
        assert SYMBOL in eng.open_positions and 'Adopted' in u.message.sent[-1][0]
        run(tg.cmd_holdings(u, ctx()))
        assert 'managed' in u.message.sent[-1][0] and 'not managed' not in u.message.sent[-1][0]


def test_ignore_command_keeps_the_bot_away_from_new_untracked_coins_only_up_to_now():
    eng, ex, clock = manual_engine()
    ex.bal['ETH'] = 3.0                                               # appears without a trade record
    with with_engine(eng):
        u = update()
        run(tg.cmd_ignore(u, ctx('ETH')))
        eng.sync_manual_positions(force=True)
        assert not eng.open_positions                                 # ignored what was there
        ex.user_buy(1.0, 99.0)
        eng.sync_manual_positions(force=True)
        assert SYMBOL in eng.open_positions                           # but a NEW buy is still adopted
        assert abs(eng.open_positions[SYMBOL]['amount'] - 0.999) < 1e-9


def test_release_hands_the_coins_back_and_the_bot_stays_away():
    eng, ex, clock = manual_engine()
    ex.user_buy(1.0, 99.0)
    eng.sync_manual_positions(force=True)
    with with_engine(eng):
        u = update()
        run(tg.cmd_release(u, ctx('ETH')))
        assert SYMBOL not in eng.open_positions and not stop_orders(ex)
        eng.sync_manual_positions(force=True)
        assert SYMBOL not in eng.open_positions and not ex.orders.get(999)      # not re-adopted
        assert 'Released' in u.message.sent[-1][0]


def test_setstop_moves_the_stop_up_and_down_on_the_exchange():
    eng, ex, clock = manual_engine()
    ex.user_buy(1.0, 99.0)
    eng.sync_manual_positions(force=True)
    with with_engine(eng):
        u = update()
        run(tg.cmd_setstop(u, ctx('ETH', '99')))
        (o,) = stop_orders(ex)
        assert abs(o['stop'] - 99.0) < 0.01
        run(tg.cmd_setstop(u, ctx('ETH', '90')))                      # LOWER it: allowed for manual control
        (o,) = stop_orders(ex)
        assert abs(o['stop'] - 90.0) < 0.01
        run(tg.cmd_setstop(u, ctx('ETH', '5%')))
        (o,) = stop_orders(ex)
        assert abs(o['stop'] - ex.mid * 0.95) < 0.02
        run(tg.cmd_setstop(u, ctx('ETH', '250')))                     # above the price: refused
        assert 'below the current price' in u.message.sent[-1][0]


def test_buy_command_asks_first_then_places_a_limit_order_and_protects_it_on_fill():
    eng, ex, clock = manual_engine()
    with with_engine(eng):
        u = update()
        run(tg.cmd_buy(u, ctx('eth', '150', '4%')))
        assert not eng.pending_entries                                # nothing yet: waiting for confirmation
        markup = u.message.sent[-1][1]['reply_markup']
        yes = [b for row in markup.inline_keyboard for b in row][0]
        assert 'Buy ~$150' in u.message.sent[-1][0] and '-4.0%' in u.message.sent[-1][0]

        cb = cb_update(yes.callback_data)
        run(tg.on_confirm(cb, ctx()))
        assert SYMBOL in eng.pending_entries and 'Buy order placed' in cb.callback_query.edits[-1]
        for _ in range(3):                                            # the entry limit walks up to the ask and fills
            clock.advance(61)
            eng.manage_orders()
        eng.manage_orders()
        pos = eng.open_positions[SYMBOL]
        assert pos['signal_type'] == 'manual' and pos['take_profit'] == 0.0
        (order,) = stop_orders(ex)                                    # protected the moment it filled
        assert abs(order['stop'] - 100.1 * 0.96) < 0.3                # the 4% stop the user asked for


def test_buy_command_validates_input_and_can_be_declined():
    eng, ex, clock = manual_engine()
    with with_engine(eng):
        u = update()
        run(tg.cmd_buy(u, ctx('eth')))
        assert 'Usage' in u.message.sent[-1][0]
        run(tg.cmd_buy(u, ctx('eth', 'lots')))
        assert 'Usage' in u.message.sent[-1][0]
        run(tg.cmd_buy(u, ctx('doge', '100')))
        assert 'Unknown symbol' in u.message.sent[-1][0]
        run(tg.cmd_buy(u, ctx('eth', '100', '250')))                  # stop above the price
        assert 'below the price' in u.message.sent[-1][0]

        run(tg.cmd_buy(u, ctx('eth', '100')))
        no = [b for row in u.message.sent[-1][1]['reply_markup'].inline_keyboard for b in row][1]
        run(tg.on_confirm(cb_update(no.callback_data), ctx()))
        assert not eng.pending_entries


def test_status_and_help_mention_the_manual_features():
    eng, ex, clock = manual_engine()
    ex.user_buy(1.0, 99.0)
    eng.sync_manual_positions(force=True)
    eng.config['auto_max_positions'] = 2
    eng.max_positions = 10
    with with_engine(eng):
        u = update()
        run(tg.cmd_status(u, ctx()))
        assert 'Manual 1' in text_of(u) and 'automatic 0/2' in text_of(u)
    for cmd in ('buy', 'holdings', 'adopt', 'setstop', 'release', 'ignore'):
        assert f"/{cmd}" in tg.HELP_TEXT


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
