"""
Offline tests for the limit-order lifecycle (modules/order_manager.py + TradingEngine).
Uses an in-memory fake Binance, so nothing touches the network.

Run:  python tests/test_limit_orders.py     (or: pytest tests/test_limit_orders.py)
"""
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.order_manager import SpotOrderManager
import modules.trade_engine as te

SYMBOL = 'ETH/USDC'
COMMISSION = 0.001


class Err(Exception):
    def __init__(self, code, msg=''):
        super().__init__(msg or str(code))
        self.code = code


class Clock:
    def __init__(self):
        self.t = 1_000_000.0

    def __call__(self):
        return self.t

    def advance(self, s):
        self.t += s


class FakeExchange:
    """Just enough of Binance spot: limit/market orders, OCO lists, fills on tick()."""

    def __init__(self):
        self.bid, self.ask = 100.0, 100.1
        self.orders, self.lists = {}, {}
        self.freeze_limit_sells = False       # simulate limit sells that just don't fill
        self._id = 0
        self.bal = {'ETH': 0.0, 'USDC': 10_000.0}

    # --- market data -----------------------------------------------------
    @property
    def mid(self):
        return (self.bid + self.ask) / 2

    def set_market(self, bid, ask=None):
        self.bid, self.ask = bid, ask if ask is not None else round(bid + 0.1, 2)
        self.tick()

    def get_symbol_info(self, symbol):
        return {'baseAsset': 'ETH', 'quoteAsset': 'USDC', 'filters': [
            {'filterType': 'PRICE_FILTER', 'tickSize': '0.01'},
            {'filterType': 'LOT_SIZE', 'stepSize': '0.001', 'minQty': '0.001', 'maxQty': '100000'},
            {'filterType': 'NOTIONAL', 'minNotional': '5'}]}

    def get_orderbook_ticker(self, symbol):
        return {'bidPrice': str(self.bid), 'askPrice': str(self.ask)}

    def get_symbol_ticker(self, symbol):
        return {'price': str(self.mid)}

    timestamp_offset = 0

    def get_server_time(self):
        import time as _t
        return {'serverTime': int(_t.time() * 1000)}

    def get_open_orders(self, **kw):
        return [{'symbol': o['symbol'], 'side': o['side'], 'type': o['type'], 'origQty': str(o['qty']),
                 'price': str(o['price']), 'stopPrice': str(o.get('stop', ''))}
                for o in self.orders.values() if o['status'] == 'NEW']

    def get_account(self):
        locked, seen_lists = 0.0, set()
        for o in self.orders.values():
            if o['status'] == 'NEW' and o['side'] == 'SELL':
                lid = o.get('list')
                if lid is not None:                          # an OCO locks its quantity once, not per leg
                    if lid in seen_lists:
                        continue
                    seen_lists.add(lid)
                locked += o['qty']
        return {'balances': [{'asset': a, 'free': str(round(v, 8)),
                              'locked': str(round(locked if a == 'ETH' else 0.0, 8))}
                             for a, v in self.bal.items()]}

    # --- things the USER does on the exchange, outside the bot -------------
    def user_buy(self, qty, price):
        o = self._new(symbol='ETHUSDC', side='BUY', type='MARKET', qty=float(qty), price=float(price))
        self._fill(o, price)
        return o['orderId']

    def user_sell(self, qty):
        self.bal['ETH'] -= float(qty)

    def get_asset_balance(self, asset):
        return {'free': str(round(self.bal[asset], 8))}

    # --- orders ----------------------------------------------------------
    def _new(self, **kw):
        self._id += 1
        o = dict(orderId=self._id, status='NEW', executedQty='0', cummulativeQuoteQty='0', **kw)
        self.orders[self._id] = o
        return o

    def _fill(self, o, price):
        o['status'] = 'FILLED'
        o['executedQty'] = str(o['qty'])
        o['cummulativeQuoteQty'] = str(round(o['qty'] * price, 8))
        if o['side'] == 'BUY':
            self.bal['ETH'] += o['qty'] * (1 - COMMISSION)     # fee taken in the base coin

    def _lock(self, qty):
        if self.bal['ETH'] + 1e-12 < qty:
            raise Err(-2010, 'insufficient balance')
        self.bal['ETH'] -= qty

    def order_limit_buy(self, symbol, quantity, price, timeInForce=None):
        o = self._new(symbol=symbol, side='BUY', type='LIMIT', qty=float(quantity), price=float(price))
        self.tick()
        return {'orderId': o['orderId']}

    def order_limit_sell(self, symbol, quantity, price, timeInForce=None):
        self._lock(float(quantity))
        o = self._new(symbol=symbol, side='SELL', type='LIMIT', qty=float(quantity), price=float(price))
        self.tick()
        return {'orderId': o['orderId']}

    def order_market_sell(self, symbol, quantity):
        q = float(quantity)
        self._lock(q)
        o = self._new(symbol=symbol, side='SELL', type='MARKET', qty=q, price=self.bid)
        self._fill(o, self.bid)
        return {'orderId': o['orderId'], 'executedQty': o['executedQty'],
                'cummulativeQuoteQty': o['cummulativeQuoteQty']}

    def cancel_order(self, symbol, orderId):
        o = self.orders[orderId]
        if o['status'] != 'NEW':
            raise Err(-2011, 'Unknown order sent.')
        o['status'] = 'CANCELED'
        if o['side'] == 'SELL':
            self.bal['ETH'] += o['qty']

    def get_order(self, symbol, orderId):
        return dict(self.orders[orderId])

    def get_my_trades(self, symbol, orderId=None, limit=None):
        if orderId is None:
            return [{'isBuyer': True, 'qty': str(o['qty']),
                     'price': str(float(o['cummulativeQuoteQty']) / float(o['executedQty']))}
                    for o in self.orders.values() if o['side'] == 'BUY' and o['status'] == 'FILLED']
        o = self.orders[orderId]
        if o['side'] == 'BUY' and o['status'] == 'FILLED':
            return [{'commission': str(o['qty'] * COMMISSION), 'commissionAsset': 'ETH'}]
        return []

    # --- OCO -------------------------------------------------------------
    def create_oco_order(self, **p):
        assert p['aboveType'] == 'LIMIT_MAKER' and p['belowType'] == 'STOP_LOSS_LIMIT'
        tp, stop, limit, qty = float(p['abovePrice']), float(p['belowStopPrice']), float(p['belowPrice']), float(p['quantity'])
        if not (stop < self.mid < tp) or limit >= stop:
            raise Err(-2010, 'OCO price restriction')
        self._lock(qty)
        a = self._new(symbol=p['symbol'], side='SELL', type='LIMIT_MAKER', qty=qty, price=tp)
        b = self._new(symbol=p['symbol'], side='SELL', type='STOP_LOSS_LIMIT', qty=qty, price=limit,
                      stop=stop, triggered=False)
        self._id += 1
        self.lists[self._id] = {'status': 'EXECUTING', 'legs': [a['orderId'], b['orderId']]}
        for leg in (a, b):
            leg['list'] = self._id
        return {'orderListId': self._id}

    def create_order(self, symbol, side, type, timeInForce=None, quantity=None, price=None, stopPrice=None):
        """Standalone STOP_LOSS_LIMIT sell (no OCO list), as used for trend_hold protection."""
        assert type == 'STOP_LOSS_LIMIT' and side == 'SELL', (type, side)
        qty, limit, stop = float(quantity), float(price), float(stopPrice)
        if not (stop < self.mid) or limit >= stop:
            raise Err(-2010, 'stop price restriction')
        self._lock(qty)
        o = self._new(symbol=symbol, side='SELL', type='STOP_LOSS_LIMIT', qty=qty, price=limit,
                      stop=stop, triggered=False)
        self.tick()
        return {'orderId': o['orderId']}

    def v3_delete_order_list(self, symbol, orderListId):
        ol = self.lists[orderListId]
        if ol['status'] != 'EXECUTING':
            raise Err(-2011, 'Unknown order list')
        restored = False
        for oid in ol['legs']:
            o = self.orders[oid]
            if o['status'] == 'NEW':
                o['status'] = 'CANCELED'
                if not restored:                    # an OCO locks its quantity once, not per leg
                    self.bal['ETH'] += o['qty']
                    restored = True
        ol['status'] = 'ALL_DONE'

    def v3_get_order_list(self, orderListId):
        ol = self.lists[orderListId]
        return {'orderListId': orderListId, 'listOrderStatus': ol['status'],
                'orders': [{'orderId': i} for i in ol['legs']]}

    # --- matching ---------------------------------------------------------
    def tick(self):
        for o in list(self.orders.values()):
            if o['status'] != 'NEW':
                continue
            t = o['type']
            if t == 'LIMIT' and o['side'] == 'BUY' and self.ask <= o['price']:
                self._fill(o, o['price'])
            elif t == 'LIMIT' and o['side'] == 'SELL' and self.bid >= o['price'] \
                    and not self.freeze_limit_sells:
                self._fill(o, o['price'])
            elif t == 'LIMIT_MAKER' and self.bid >= o['price']:
                self._finish_leg(o, o['price'])
            elif t == 'STOP_LOSS_LIMIT':
                if not o['triggered'] and self.mid <= o['stop']:
                    o['triggered'] = True
                if o['triggered'] and self.bid >= o['price']:
                    if 'list' in o:
                        self._finish_leg(o, o['price'])
                    else:
                        self._fill(o, o['price'])

    def _finish_leg(self, o, price):
        self._fill(o, price)
        ol = self.lists[o['list']]
        ol['status'] = 'ALL_DONE'
        for oid in ol['legs']:
            if oid != o['orderId'] and self.orders[oid]['status'] == 'NEW':
                self.orders[oid]['status'] = 'CANCELED'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_engine(cfg=None):
    ex, clock = FakeExchange(), Clock()
    om = SpotOrderManager(ex, cfg or {}, now=clock)
    eng = te.TradingEngine.__new__(te.TradingEngine)
    eng.config = cfg or {}
    eng.trading_mode = 'testnet'
    eng.symbols = [SYMBOL]
    eng.max_positions = 10
    eng.open_positions, eng.open_futures_positions = {}, {}
    eng.pending_entries, eng.pending_exits = {}, {}
    eng.order_manager = om
    eng.binance_client = ex
    eng._lock = te.threading.RLock()
    eng.stop_market_fallback_pct = 0.006
    eng.stop_trigger_timeout = 120.0
    eng.oco_ratchet_min_pct = 0.002
    eng._save_pending = lambda: None
    eng._save_manual_state = lambda: None          # never touch the real manual_state.json
    eng._manual_baseline, eng._manual_checked_at = None, 0.0
    # keep the tests off the real portfolio / logs / telegram
    eng.trades = []
    te.save_positions_to_file = lambda p: None
    te.add_trade = eng.trades.append
    te.log_trade = lambda *a, **k: None
    te.has_notifier = False
    te.get_pair_config = lambda s: {}
    te._time = types.SimpleNamespace(time=clock)
    return eng, ex, clock


def open_filled_position(eng, ex, clock):
    """Signal at 100.05, stop 97, target 109; market fills the limit at step 2 (the ask)."""
    assert eng._submit_limit_entry(SYMBOL, 100.05, 1.0, 97.0, 109.0, 'test', 1.5)
    for _ in range(3):
        clock.advance(61)
        eng.manage_orders()
    eng.manage_orders()
    return eng.open_positions[SYMBOL]


def active_oco(ex):
    live = [(i, l) for i, l in ex.lists.items() if l['status'] == 'EXECUTING']
    assert len(live) == 1, f"expected exactly one live OCO, found {len(live)}"
    i, l = live[0]
    return i, {ex.orders[o]['type']: ex.orders[o] for o in l['legs']}


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------
def test_entry_reprices_then_fills_and_uses_net_quantity():
    eng, ex, clock = make_engine()
    pos = open_filled_position(eng, ex, clock)
    assert SYMBOL not in eng.pending_entries
    # fee was taken in ETH, so we hold 0.999, not 1.0 - the position must record what we hold
    assert abs(pos['amount'] - 0.999) < 1e-9, pos['amount']
    assert abs(pos['entry_price'] - 100.1) < 1e-9        # filled at the ask on the last step
    assert pos['oco'] is not None


def test_entry_cancelled_when_never_filled():
    eng, ex, clock = make_engine()
    ex.set_market(100.0, 100.1)
    assert eng._submit_limit_entry(SYMBOL, 100.05, 1.0, 97.0, 109.0, 'test', 1.5)
    ex.set_market(103.0, 103.1)                           # runs away: cap (signal +0.3%) keeps us out
    for _ in range(4):
        clock.advance(61)
        eng.manage_orders()
    assert SYMBOL not in eng.pending_entries and SYMBOL not in eng.open_positions
    assert all(o['status'] != 'NEW' for o in ex.orders.values())


def test_entry_never_chases_above_cap():
    eng, ex, clock = make_engine()
    ex.set_market(105.0, 105.1)
    assert eng._submit_limit_entry(SYMBOL, 100.05, 1.0, 97.0, 109.0, 'test', 1.5)
    assert max(o['price'] for o in ex.orders.values()) <= 100.05 * 1.003 + 0.01


# ---------------------------------------------------------------------------
# OCO protection
# ---------------------------------------------------------------------------
def test_oco_placed_with_stop_limit_below_trigger():
    eng, ex, clock = make_engine()
    pos = open_filled_position(eng, ex, clock)
    _, legs = active_oco(ex)
    stop_leg, tp_leg = legs['STOP_LOSS_LIMIT'], legs['LIMIT_MAKER']
    assert stop_leg['price'] < stop_leg['stop'] < ex.mid < tp_leg['price']
    assert abs(stop_leg['qty'] - 0.999) < 1e-9
    assert abs(stop_leg['stop'] - pos['oco']['stop']) < 1e-9


def test_ratchet_replaces_oco_at_higher_stop():
    eng, ex, clock = make_engine()
    pos = open_filled_position(eng, ex, clock)
    old_id, _ = active_oco(ex)
    ex.set_market(108.0, 108.1)
    pos['stop_loss'] = 104.0                              # trailing logic raised it
    eng.manage_orders()
    new_id, legs = active_oco(ex)
    assert new_id != old_id and ex.lists[old_id]['status'] == 'ALL_DONE'
    assert abs(legs['STOP_LOSS_LIMIT']['stop'] - 104.0) < 0.01
    assert abs(pos['oco']['stop'] - 104.0) < 0.01


def test_tiny_stop_change_does_not_churn_orders():
    eng, ex, clock = make_engine()
    pos = open_filled_position(eng, ex, clock)
    old_id, _ = active_oco(ex)
    pos['stop_loss'] = pos['oco']['stop'] * 1.0005        # below oco_ratchet_min_pct
    eng.manage_orders()
    assert active_oco(ex)[0] == old_id


def test_take_profit_fill_closes_position():
    eng, ex, clock = make_engine()
    pos = open_filled_position(eng, ex, clock)
    target = pos['oco']['target']
    ex.set_market(target + 0.5)
    eng.manage_orders()
    assert SYMBOL not in eng.open_positions
    assert eng.trades[-1]['reason'] == 'take_profit'
    assert abs(eng.trades[-1]['exit_price'] - target) < 1e-6


def test_stop_limit_fill_closes_position():
    eng, ex, clock = make_engine()
    pos = open_filled_position(eng, ex, clock)
    stop, limit = pos['oco']['stop'], pos['oco']['limit']
    ex.set_market(limit + 0.05, stop - 0.05)              # dips through stop but bid stays >= limit
    ex.set_market(limit + 0.05, limit + 0.15)
    eng.manage_orders()
    assert SYMBOL not in eng.open_positions
    assert eng.trades[-1]['reason'] == 'stop_loss'


# ---------------------------------------------------------------------------
# MARKET FALLBACK on the stop (the safety requirement)
# ---------------------------------------------------------------------------
def test_market_fallback_when_price_gaps_through_stop_limit():
    eng, ex, clock = make_engine()
    pos = open_filled_position(eng, ex, clock)
    ex.set_market(pos['oco']['stop'] * 0.90)              # crash: bid far below the limit price
    live_before, _ = active_oco(ex)
    eng.manage_orders()
    assert SYMBOL not in eng.open_positions
    assert eng.trades[-1]['reason'] == 'stop_loss_market'
    assert ex.lists[live_before]['status'] == 'ALL_DONE'  # OCO was cancelled first
    assert ex.bal['ETH'] < 1e-6                           # everything sold


def test_market_fallback_when_stop_limit_stuck_unfilled():
    eng, ex, clock = make_engine()
    pos = open_filled_position(eng, ex, clock)
    stop, limit = pos['oco']['stop'], pos['oco']['limit']
    # just under the stop and 0.2% below - not "through" - but bid < limit so it can't fill
    ex.set_market(limit - 0.02, stop - 0.05)
    eng.manage_orders()
    assert SYMBOL in eng.open_positions                    # still waiting: inside the grace period
    clock.advance(130)
    eng.manage_orders()
    assert SYMBOL not in eng.open_positions
    assert eng.trades[-1]['reason'] == 'stop_loss_market'


def test_stale_ratchet_does_not_wait_for_exchange_stop():
    """Bot's stop was raised above the OCO's, price is already below the new stop."""
    eng, ex, clock = make_engine()
    pos = open_filled_position(eng, ex, clock)
    pos['stop_loss'] = 104.0
    eng._handle_exit_signal(SYMBOL, pos, 103.0, 'trailing_stop')
    assert SYMBOL in eng.pending_exits                     # exit started (urgent)
    assert eng.pending_exits[SYMBOL]['st']['urgent'] is True


def test_exchange_owned_stop_is_not_double_handled():
    eng, ex, clock = make_engine()
    pos = open_filled_position(eng, ex, clock)
    eng._handle_exit_signal(SYMBOL, pos, pos['oco']['stop'] - 0.5, 'stop_loss')
    assert SYMBOL not in eng.pending_exits                 # left to the OCO + fallback


def test_unprotected_position_below_stop_exits_urgently():
    """OCO missing (e.g. placement failed) and price already through the stop:
    an OCO can't be placed, so the bot must exit right away (urgent -> market fallback)."""
    eng, ex, clock = make_engine()
    pos = open_filled_position(eng, ex, clock)
    eng.order_manager.cancel_oco(SYMBOL, pos['oco']['order_list_id'])
    pos['oco'] = None
    ex.freeze_limit_sells = True                           # the limit sell won't fill
    ex.set_market(90.0)                                    # far below the 97.04 stop
    eng.manage_orders()
    assert SYMBOL in eng.pending_exits and eng.pending_exits[SYMBOL]['st']['urgent'] is True
    assert SYMBOL in eng.open_positions
    ex.set_market(85.0)
    clock.advance(46)                                      # urgent timeout is 45s
    eng.manage_orders()
    assert SYMBOL not in eng.open_positions
    assert abs(eng.trades[-1]['exit_price'] - 85.0) < 1e-6  # market sold at the bid
    assert not [o for o in ex.orders.values() if o['status'] == 'NEW']


def test_oco_placement_failure_is_retried_next_cycle():
    eng, ex, clock = make_engine()
    pos = open_filled_position(eng, ex, clock)
    eng.order_manager.cancel_oco(SYMBOL, pos['oco']['order_list_id'])
    pos['oco'] = None
    real = ex.create_oco_order
    ex.create_oco_order = lambda **p: (_ for _ in ()).throw(Err(-1021, 'timestamp'))
    eng.manage_orders()
    assert pos['oco'] is None and SYMBOL in eng.open_positions   # still open, still trying
    ex.create_oco_order = real
    eng.manage_orders()
    assert pos['oco'] is not None                                # protected again


# ---------------------------------------------------------------------------
# Other exits: limit sell, then market fallback
# ---------------------------------------------------------------------------
def test_signal_exit_walks_ask_mid_bid_and_fills():
    eng, ex, clock = make_engine()
    open_filled_position(eng, ex, clock)
    ex.set_market(105.0, 105.2)
    assert eng.close_position(SYMBOL, 105.1, 'signal_reversal') is True
    assert ex.lists and all(l['status'] == 'ALL_DONE' for l in ex.lists.values())   # OCO cancelled
    sell = [o for o in ex.orders.values() if o['side'] == 'SELL' and o['status'] == 'NEW'][0]
    assert abs(sell['price'] - 105.2) < 1e-9               # step 0: at the ask
    clock.advance(61); eng.manage_orders()
    sell = [o for o in ex.orders.values() if o['side'] == 'SELL' and o['status'] == 'NEW'][0]
    assert abs(sell['price'] - 105.1) < 1e-9               # step 1: mid
    clock.advance(61); eng.manage_orders()
    eng.manage_orders()
    assert SYMBOL not in eng.open_positions
    assert eng.trades[-1]['reason'] == 'signal_reversal'


def test_signal_exit_falls_back_to_market_after_timeout():
    eng, ex, clock = make_engine()
    open_filled_position(eng, ex, clock)
    ex.freeze_limit_sells = True                           # limit sells never fill
    eng.close_position(SYMBOL, 100.0, 'signal_reversal')
    ex.set_market(95.0)
    clock.advance(121)
    eng.manage_orders()
    assert SYMBOL in eng.open_positions                    # still inside the 180s window
    clock.advance(61)                                      # 182s total
    eng.manage_orders()
    assert SYMBOL not in eng.open_positions
    assert [o for o in ex.orders.values() if o['type'] == 'MARKET' and o['side'] == 'SELL']
    assert abs(eng.trades[-1]['exit_price'] - 95.0) < 1e-6  # market sold at the bid
    assert not [o for o in ex.orders.values() if o['status'] == 'NEW']


def test_urgent_exit_uses_short_market_fallback():
    eng, ex, clock = make_engine()
    open_filled_position(eng, ex, clock)
    ex.freeze_limit_sells = True
    eng.close_position(SYMBOL, 100.0, 'emergency_sell')
    assert SYMBOL in eng.pending_exits and SYMBOL in eng.open_positions
    ex.set_market(90.0)
    clock.advance(30)
    eng.manage_orders()
    assert SYMBOL in eng.open_positions                    # not yet: only 30s of 45s
    clock.advance(16)
    eng.manage_orders()
    assert SYMBOL not in eng.open_positions
    assert abs(eng.trades[-1]['exit_price'] - 90.0) < 1e-6  # market sold at the bid


def test_slots_count_pending_entries():
    eng, ex, clock = make_engine()
    eng.max_positions = 1
    assert eng._submit_limit_entry(SYMBOL, 100.05, 1.0, 97.0, 109.0, 'test', 1.5)
    assert eng.active_slots == 1
    assert eng.open_position(SYMBOL, 'long', 100.05, 1.0, 97.0, 109.0) is False


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
