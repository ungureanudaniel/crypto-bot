"""
order_manager.py
================
Spot limit-order execution for Binance (live / testnet).

Everything here is NON-BLOCKING: `start_*` places the first order and returns a small
JSON-serialisable state dict; the caller (TradingEngine.manage_orders, run every
minute by the scheduler) calls `advance_*` until the state reports it is finished.

Flows
-----
Entry : limit buy, repriced toward the ask a few times, then cancelled if unfilled.
Exit  : limit sell at the ask -> mid -> bid, then MARKET fallback after a timeout.
Stop  : an exchange-side OCO (LIMIT_MAKER take-profit + STOP_LOSS_LIMIT stop).
        The engine ratchets the stop by cancelling and re-placing the OCO, and
        falls back to a market sell when price runs through the stop-limit.

Only the new-style OCO endpoint (aboveType / belowType) is used; python-binance 1.0.34
routes create_oco_order() to POST /api/v3/orderList/oco. Cancel / status use the raw
v3_delete_order_list / v3_get_order_list endpoints.
"""
import logging
import time
from decimal import Decimal, ROUND_FLOOR, ROUND_CEILING
from typing import Dict, Optional

logger = logging.getLogger(__name__)

_TERMINAL = ('CANCELED', 'EXPIRED', 'REJECTED', 'EXPIRED_IN_MATCH')


def _d(x) -> Decimal:
    return Decimal(str(x))


def _fmt(d: Decimal) -> str:
    """Plain decimal string (no scientific notation) for the API."""
    return format(d.normalize(), 'f')


class SpotOrderManager:
    def __init__(self, client, cfg: dict, now=time.time):
        self.client = client
        self._now = now
        self._info_cache: Dict[str, dict] = {}
        g = cfg.get
        # entries
        self.entry_step_seconds = float(g('entry_step_seconds', 60))
        self.entry_fractions = list(g('entry_reprice_fractions', [0.0, 0.5, 1.0]))
        self.entry_max_chase_pct = float(g('entry_max_chase_pct', 0.003))
        # exits
        self.exit_step_seconds = float(g('exit_step_seconds', 60))
        self.exit_fractions = list(g('exit_reprice_fractions', [1.0, 0.5, 0.0]))
        self.exit_market_fallback_seconds = float(g('exit_market_fallback_seconds', 180))
        self.stop_exit_timeout_seconds = float(g('stop_exit_timeout_seconds', 45))
        # OCO
        self.stop_limit_buffer_pct = float(g('stop_limit_buffer_pct', 0.003))

    # ------------------------------------------------------------------
    # Market info / rounding
    # ------------------------------------------------------------------
    @staticmethod
    def sym(symbol: str) -> str:
        return symbol.replace('/', '')

    def info(self, symbol: str) -> dict:
        key = self.sym(symbol)
        if key not in self._info_cache:
            raw = self.client.get_symbol_info(key)
            if not raw:
                raise ValueError(f"No symbol info for {key}")
            f = {x['filterType']: x for x in raw['filters']}
            notional = f.get('NOTIONAL') or f.get('MIN_NOTIONAL') or {}
            self._info_cache[key] = {
                'base': raw['baseAsset'],
                'quote': raw['quoteAsset'],
                'tick': _d(f['PRICE_FILTER']['tickSize']),
                'step': _d(f['LOT_SIZE']['stepSize']),
                'min_qty': _d(f['LOT_SIZE']['minQty']),
                'min_notional': _d(notional.get('minNotional', 0)),
            }
        return self._info_cache[key]

    def _floor_qty(self, symbol: str, qty) -> Decimal:
        step = self.info(symbol)['step']
        return (_d(qty) / step).to_integral_value(rounding=ROUND_FLOOR) * step

    def round_qty_down(self, symbol: str, qty) -> float:
        return float(self._floor_qty(symbol, qty))

    def _round_price(self, symbol: str, price, up: bool = False) -> Decimal:
        tick = self.info(symbol)['tick']
        mode = ROUND_CEILING if up else ROUND_FLOOR
        return (_d(price) / tick).to_integral_value(rounding=mode) * tick

    def min_notional(self, symbol: str) -> float:
        return float(self.info(symbol)['min_notional'])

    def min_qty(self, symbol: str) -> float:
        return float(self.info(symbol)['min_qty'])

    def book(self, symbol: str):
        t = self.client.get_orderbook_ticker(symbol=self.sym(symbol))
        return float(t['bidPrice']), float(t['askPrice'])

    def last_price(self, symbol: str) -> float:
        return float(self.client.get_symbol_ticker(symbol=self.sym(symbol))['price'])

    def free_balance(self, asset: str) -> float:
        bal = self.client.get_asset_balance(asset=asset)
        return float(bal['free']) if bal else 0.0

    def sellable_qty(self, symbol: str, want: float) -> float:
        """min(wanted, free balance) floored to the lot step; 0 if below minQty."""
        free = self.free_balance(self.info(symbol)['base'])
        q = self._floor_qty(symbol, min(_d(want), _d(free)))
        return float(q) if q >= self.info(symbol)['min_qty'] else 0.0

    def _cancel_order(self, symbol: str, order_id):
        try:
            self.client.cancel_order(symbol=self.sym(symbol), orderId=order_id)
        except Exception as e:
            # -2011 = unknown order (already filled / cancelled) -> fine, we re-read it after
            if getattr(e, 'code', None) != -2011:
                raise

    # ------------------------------------------------------------------
    # ENTRY: limit buy, reprice, cancel
    # ------------------------------------------------------------------
    def start_entry(self, symbol: str, qty: float, signal_price: float) -> Optional[dict]:
        now = self._now()
        st = {
            'kind': 'entry', 'symbol': symbol, 'qty': float(qty),
            'signal_price': float(signal_price), 'step': 0,
            'order_id': None, 'order_ids': [], 'price': None,
            'placed_at': now, 'created_at': now,
            'filled_qty': 0.0, 'filled_quote': 0.0,
        }
        if not self._place_entry_order(st):
            return None
        return st

    def _entry_price(self, st: dict) -> Decimal:
        bid, ask = self.book(st['symbol'])
        frac = self.entry_fractions[min(st['step'], len(self.entry_fractions) - 1)]
        raw = bid + frac * (ask - bid)
        cap = st['signal_price'] * (1 + self.entry_max_chase_pct)
        return self._round_price(st['symbol'], min(raw, cap))     # buys round DOWN

    def _place_entry_order(self, st: dict) -> bool:
        symbol = st['symbol']
        remaining = self._floor_qty(symbol, _d(st['qty']) - _d(st['filled_qty']))
        price = self._entry_price(st)
        if price <= 0 or remaining < self.info(symbol)['min_qty'] \
                or remaining * price < self.info(symbol)['min_notional']:
            return False
        resp = self.client.order_limit_buy(symbol=self.sym(symbol),
                                           quantity=_fmt(remaining), price=_fmt(price))
        st['order_id'] = resp['orderId']
        st['order_ids'].append(resp['orderId'])
        st['price'] = float(price)
        st['placed_at'] = self._now()
        logger.info(f"📥 {symbol} entry limit BUY {remaining} @ {price} (step {st['step']})")
        return True

    @staticmethod
    def _absorb(st: dict, od: dict, prefix: str):
        st[f'{prefix}_qty'] += float(od['executedQty'])
        st[f'{prefix}_quote'] += float(od['cummulativeQuoteQty'])

    def _entry_result(self, st: dict) -> dict:
        if st['filled_qty'] <= 0:
            return {'status': 'cancelled'}
        symbol = st['symbol']
        base = self.info(symbol)['base']
        commission = 0.0
        try:
            for oid in st['order_ids']:
                for t in self.client.get_my_trades(symbol=self.sym(symbol), orderId=oid):
                    if t.get('commissionAsset') == base:
                        commission += float(t['commission'])
        except Exception as e:
            logger.warning(f"Could not read commissions for {symbol}: {e}")
        return {
            'status': 'filled',
            'qty': st['filled_qty'] - commission,       # what we actually hold
            'gross_qty': st['filled_qty'],
            'avg_price': st['filled_quote'] / st['filled_qty'],
        }

    def advance_entry(self, st: dict) -> dict:
        """Returns {'status': 'open'} or a final {'status': 'filled'|'cancelled', ...}."""
        symbol = st['symbol']
        sym = self.sym(symbol)

        if st['order_id'] is None:
            return self._entry_result(st)

        od = self.client.get_order(symbol=sym, orderId=st['order_id'])
        if od['status'] == 'FILLED' or od['status'] in _TERMINAL:
            self._absorb(st, od, 'filled')
            st['order_id'] = None
            return self._entry_result(st)

        if not st.get('cancel_requested') and self._now() - st['placed_at'] < self.entry_step_seconds:
            return {'status': 'open'}

        # time to reprice or give up: cancel, then re-read what actually filled
        self._cancel_order(symbol, st['order_id'])
        od = self.client.get_order(symbol=sym, orderId=st['order_id'])
        self._absorb(st, od, 'filled')
        st['order_id'] = None
        if od['status'] == 'FILLED':
            return self._entry_result(st)

        if st.get('cancel_requested') or st['step'] >= len(self.entry_fractions) - 1:
            logger.info(f"⌛ {symbol} entry not filled after {st['step'] + 1} steps - cancelled")
            return self._entry_result(st)

        st['step'] += 1
        try:
            placed = self._place_entry_order(st)
        except Exception as e:
            logger.error(f"❌ {symbol} entry reprice failed: {e}")
            placed = False
        return {'status': 'open'} if placed else self._entry_result(st)

    # ------------------------------------------------------------------
    # EXIT: limit sell ask -> mid -> bid, then MARKET fallback
    # ------------------------------------------------------------------
    def start_exit(self, symbol: str, qty: float, urgent: bool = False) -> dict:
        now = self._now()
        st = {
            'kind': 'exit', 'symbol': symbol, 'urgent': bool(urgent), 'qty': float(qty),
            'step': -1, 'order_id': None, 'created_at': now, 'placed_at': now,
            'sold_qty': 0.0, 'sold_quote': 0.0,
        }
        self._exit_reprice(st, 0)
        return st

    def _exit_fractions(self, st: dict):
        return [0.0] if st['urgent'] else self.exit_fractions

    def _exit_reprice(self, st: dict, step: int):
        symbol = st['symbol']
        remaining = _d(st['qty']) - _d(st['sold_qty'])
        qty = self.sellable_qty(symbol, float(remaining))
        if qty <= 0:
            return
        bid, ask = self.book(symbol)
        fr = self._exit_fractions(st)
        raw = bid + fr[min(step, len(fr) - 1)] * (ask - bid)
        price = self._round_price(symbol, raw, up=True)            # sells round UP
        resp = self.client.order_limit_sell(symbol=self.sym(symbol),
                                            quantity=_fmt(_d(qty)), price=_fmt(price))
        st['order_id'] = resp['orderId']
        st['step'] = step
        st['placed_at'] = self._now()
        logger.info(f"📤 {symbol} exit limit SELL {qty} @ {price} (step {step}, urgent={st['urgent']})")

    def _exit_cancel_and_absorb(self, st: dict):
        if st['order_id'] is None:
            return
        self._cancel_order(st['symbol'], st['order_id'])
        od = self.client.get_order(symbol=self.sym(st['symbol']), orderId=st['order_id'])
        self._absorb(st, od, 'sold')
        st['order_id'] = None

    def _exit_done(self, st: dict) -> dict:
        avg = st['sold_quote'] / st['sold_qty'] if st['sold_qty'] > 0 else 0.0
        return {'status': 'filled', 'avg_price': avg, 'qty': st['sold_qty']}

    def advance_exit(self, st: dict) -> dict:
        symbol = st['symbol']
        if st['order_id'] is not None:
            od = self.client.get_order(symbol=self.sym(symbol), orderId=st['order_id'])
            if od['status'] == 'FILLED' or od['status'] in _TERMINAL:
                self._absorb(st, od, 'sold')
                st['order_id'] = None

        remaining = self._floor_qty(symbol, _d(st['qty']) - _d(st['sold_qty']))
        if remaining < self.info(symbol)['min_qty']:
            return self._exit_done(st)

        elapsed = self._now() - st['created_at']
        fallback = self.stop_exit_timeout_seconds if st['urgent'] else self.exit_market_fallback_seconds

        if elapsed >= fallback:
            self._exit_cancel_and_absorb(st)
            qty = self.sellable_qty(symbol, float(_d(st['qty']) - _d(st['sold_qty'])))
            if qty > 0:
                logger.warning(f"⚠️ {symbol} exit not filled in {elapsed:.0f}s - MARKET fallback")
                resp = self.client.order_market_sell(symbol=self.sym(symbol), quantity=_fmt(_d(qty)))
                self._absorb(st, resp, 'sold')
            return self._exit_done(st)

        fr = self._exit_fractions(st)
        step = min(int(elapsed // self.exit_step_seconds), len(fr) - 1)
        if st['order_id'] is None or step != st['step']:
            self._exit_cancel_and_absorb(st)
            self._exit_reprice(st, step)
        return {'status': 'open'}

    def market_sell(self, symbol: str, qty: float) -> Optional[dict]:
        """Immediate market sell of min(qty, free balance). Returns {'avg_price','qty'} or None."""
        q = self.sellable_qty(symbol, qty)
        if q <= 0:
            return None
        resp = self.client.order_market_sell(symbol=self.sym(symbol), quantity=_fmt(_d(q)))
        eq = float(resp['executedQty'])
        return {'avg_price': float(resp['cummulativeQuoteQty']) / eq if eq else 0.0, 'qty': eq}

    # ------------------------------------------------------------------
    # OCO protection (exchange-side take-profit + stop-limit)
    # ------------------------------------------------------------------
    def place_oco(self, symbol: str, qty: float, stop: float, target: float,
                  last_price: float) -> dict:
        """Returns {'ok': True, 'order_list_id', 'stop', 'limit', 'target', 'qty'}
        or {'ok': False, 'error': 'stop_breached'|'target_reached'|'too_small'|'api', ...}."""
        info = self.info(symbol)
        q = self._floor_qty(symbol, qty)
        trigger = self._round_price(symbol, stop)
        limit = self._round_price(symbol, stop * (1 - self.stop_limit_buffer_pct))
        if limit >= trigger:
            limit = trigger - info['tick']
        tp = self._round_price(symbol, target, up=True)

        if last_price <= float(trigger):
            return {'ok': False, 'error': 'stop_breached'}
        if last_price >= float(tp):
            return {'ok': False, 'error': 'target_reached'}
        if q < info['min_qty'] or q * limit < info['min_notional']:
            return {'ok': False, 'error': 'too_small'}

        try:
            resp = self.client.create_oco_order(
                symbol=self.sym(symbol), side='SELL', quantity=_fmt(q),
                aboveType='LIMIT_MAKER', abovePrice=_fmt(tp),
                belowType='STOP_LOSS_LIMIT', belowStopPrice=_fmt(trigger),
                belowPrice=_fmt(limit), belowTimeInForce='GTC',
            )
        except Exception as e:
            return {'ok': False, 'error': 'api', 'detail': str(e)}

        return {'ok': True, 'kind': 'oco', 'order_list_id': resp['orderListId'], 'stop': float(trigger),
                'limit': float(limit), 'target': float(tp), 'qty': float(q),
                'placed_at': self._now()}

    def cancel_oco(self, symbol: str, order_list_id) -> str:
        """'cancelled' | 'done' (it already completed) | 'error'."""
        try:
            self.client.v3_delete_order_list(symbol=self.sym(symbol), orderListId=order_list_id)
            return 'cancelled'
        except Exception as e:
            try:
                if self.oco_status(symbol, order_list_id)['status'] == 'ALL_DONE':
                    return 'done'
            except Exception:
                pass
            logger.error(f"❌ Could not cancel OCO {order_list_id} on {symbol}: {e}")
            return 'error'

    def oco_status(self, symbol: str, order_list_id) -> dict:
        """{'status': 'EXECUTING'|'ALL_DONE'|..., 'filled': None|'take_profit'|'stop',
            'avg_price', 'qty'}"""
        ol = self.client.v3_get_order_list(orderListId=order_list_id)
        out = {'status': ol.get('listOrderStatus'), 'filled': None, 'avg_price': 0.0, 'qty': 0.0}
        if out['status'] != 'ALL_DONE':
            return out
        best = None
        for o in ol.get('orders', []):
            od = self.client.get_order(symbol=self.sym(symbol), orderId=o['orderId'])
            ex = float(od['executedQty'])
            if ex > 0 and (best is None or ex > best[0]):
                best = (ex, float(od['cummulativeQuoteQty']), od['type'])
        if best:
            out['qty'] = best[0]
            out['avg_price'] = best[1] / best[0]
            out['filled'] = 'take_profit' if best[2] in ('LIMIT_MAKER', 'LIMIT', 'TAKE_PROFIT_LIMIT') else 'stop'
        return out

    # ------------------------------------------------------------------
    # STOP-ONLY protection (no take-profit) - for trend_hold positions, whose exit IS a rising
    # stop. A single STOP_LOSS_LIMIT sell; the engine ratchets it by cancel + re-place.
    # ------------------------------------------------------------------
    def place_stop(self, symbol: str, qty: float, stop: float, last_price: float) -> dict:
        info = self.info(symbol)
        q = self._floor_qty(symbol, qty)
        trigger = self._round_price(symbol, stop)
        limit = self._round_price(symbol, stop * (1 - self.stop_limit_buffer_pct))
        if limit >= trigger:
            limit = trigger - info['tick']
        if last_price <= float(trigger):
            return {'ok': False, 'error': 'stop_breached'}
        if q < info['min_qty'] or q * limit < info['min_notional']:
            return {'ok': False, 'error': 'too_small'}
        try:
            resp = self.client.create_order(
                symbol=self.sym(symbol), side='SELL', type='STOP_LOSS_LIMIT', timeInForce='GTC',
                quantity=_fmt(q), price=_fmt(limit), stopPrice=_fmt(trigger))
        except Exception as e:
            return {'ok': False, 'error': 'api', 'detail': str(e)}
        return {'ok': True, 'kind': 'stop', 'order_id': resp['orderId'], 'stop': float(trigger),
                'limit': float(limit), 'target': 0.0, 'qty': float(q), 'placed_at': self._now()}

    def cancel_stop(self, symbol: str, order_id) -> str:
        """'cancelled' | 'done' (it already filled) | 'error'."""
        try:
            self.client.cancel_order(symbol=self.sym(symbol), orderId=order_id)
            return 'cancelled'
        except Exception as e:
            try:
                od = self.client.get_order(symbol=self.sym(symbol), orderId=order_id)
                if od['status'] == 'FILLED' or float(od['executedQty']) > 0:
                    return 'done'
                if od['status'] in _TERMINAL:
                    return 'cancelled'
            except Exception:
                pass
            logger.error(f"❌ Could not cancel stop order {order_id} on {symbol}: {e}")
            return 'error'

    def stop_status(self, symbol: str, order_id) -> dict:
        """Same shape as oco_status(): status EXECUTING | ALL_DONE, filled None | 'stop'."""
        od = self.client.get_order(symbol=self.sym(symbol), orderId=order_id)
        out = {'status': 'EXECUTING', 'filled': None, 'avg_price': 0.0, 'qty': 0.0}
        ex = float(od['executedQty'])
        if od['status'] == 'FILLED' or od['status'] in _TERMINAL:
            out['status'] = 'ALL_DONE'
            if ex > 0:
                out.update(filled='stop', qty=ex, avg_price=float(od['cummulativeQuoteQty']) / ex)
        return out

    # ------------------------------------------------------------------
    # Generic protection dispatch: kind 'oco' (take-profit + stop) or 'stop' (stop only)
    # ------------------------------------------------------------------
    def place_protection(self, symbol: str, qty: float, stop: float, target: float,
                         last_price: float) -> dict:
        if target:
            return self.place_oco(symbol, qty, stop, target, last_price)
        return self.place_stop(symbol, qty, stop, last_price)

    def cancel_protection(self, symbol: str, prot: dict) -> str:
        if prot.get('kind') == 'stop':
            return self.cancel_stop(symbol, prot['order_id'])
        return self.cancel_oco(symbol, prot['order_list_id'])

    def protection_status(self, symbol: str, prot: dict) -> dict:
        if prot.get('kind') == 'stop':
            return self.stop_status(symbol, prot['order_id'])
        return self.oco_status(symbol, prot['order_list_id'])
