"""
tg_commands.py - the Telegram command layer.

Registered BEFORE the older handlers in telegram_bot.py, so for any command defined here this
version answers and the old one is never reached; every other old command keeps working.

  Security   only your chat (TELEGRAM_CHAT_ID) can use the bot; anyone else is dropped silently.
             /sellall, /closeall and /stop ask for confirmation with buttons.
  Overview   /status  /positions  /orders  /trades  /gate  /breaker
  Control    /pause  /resume  /close SYMBOL [now]  /closeall  /sellall
  Help       /help (grouped) and the command menu Telegram shows next to the message box.

Everything that talks to the exchange runs in a worker thread so the bot stays responsive.
"""
import asyncio
import html
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional

from telegram import (BotCommand, InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup, Update)
from telegram.constants import ParseMode
from telegram.ext import (ApplicationHandlerStop, CallbackQueryHandler, CommandHandler, ContextTypes,
                          TypeHandler)

from config_loader import config as _cfg
from modules import trend_hold
from modules.trade_engine import trading_engine as engine

logger = logging.getLogger(__name__)

CONFIRM_TTL_SECONDS = 60
URGENT = 'emergency_sell'
NORMAL = 'manual_close'


# =====================================================================
# small helpers
# =====================================================================
def esc(x) -> str:
    return html.escape(str(x))


def price_fmt(p: Optional[float]) -> str:
    if p is None:
        return '?'
    return f"{p:,.2f}" if p >= 100 else (f"{p:,.4f}" if p >= 1 else f"{p:.6f}")


def age(seconds: float) -> str:
    s = max(0, int(seconds))
    if s < 90:
        return f"{s}s"
    m = s // 60
    if m < 120:
        return f"{m}m"
    h = m / 60
    return f"{h:.1f}h" if h < 48 else f"{h / 24:.1f}d"


def split_message(text: str, limit: int = 3800) -> List[str]:
    """Telegram rejects messages over 4096 characters; split on line boundaries."""
    if len(text) <= limit:
        return [text]
    parts, cur = [], ''
    for line in text.split('\n'):
        if len(cur) + len(line) + 1 > limit and cur:
            parts.append(cur)
            cur = ''
        cur += line + '\n'
    if cur.strip():
        parts.append(cur)
    return parts


async def reply(update: Update, text: str, **kw):
    kw.setdefault('parse_mode', ParseMode.HTML)
    msg = update.effective_message
    chunks = split_message(text)
    for i, chunk in enumerate(chunks):
        await msg.reply_text(chunk, **(kw if i == len(chunks) - 1 else {'parse_mode': kw['parse_mode']}))


_price_cache = (0.0, {})


def prices() -> Dict[str, float]:
    """Current prices for all symbols, cached for 10 seconds (each call is one request per symbol)."""
    global _price_cache
    now = time.time()
    if now - _price_cache[0] < 10 and _price_cache[1]:
        return _price_cache[1]
    _price_cache = (now, engine.get_current_prices())
    return _price_cache[1]


def parse_symbol(text: str) -> Optional[str]:
    """'btc', 'BTC/USDC', 'btcusdc' -> the configured symbol, preferring ones we hold."""
    t = text.strip().upper().replace('-', '/')
    universe = list(dict.fromkeys(list(engine.open_positions) + list(engine.pending_entries) + list(engine.symbols)))
    if t in universe:
        return t
    flat = t.replace('/', '')
    for s in universe:
        if s.replace('/', '') == flat or s.split('/')[0] == flat:
            return s
    return None


def protection_label(pos: dict) -> str:
    p = pos.get('oco')
    if not p:
        return "⚠️ no exchange stop"
    kind = 'stop' if p.get('kind') == 'stop' else 'stop + target'
    return f"🛡️ exchange {kind} @ {price_fmt(p.get('stop'))}"


# =====================================================================
# security
# =====================================================================
def authorized(update: Update) -> bool:
    allowed = str(_cfg.config.get('telegram_chat_id') or '').strip()
    chat = update.effective_chat
    return bool(allowed) and chat is not None and str(chat.id) == allowed


async def guard(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Runs before every handler. Anyone but the configured chat is ignored (no reply, so the bot
    does not even confirm it exists)."""
    if authorized(update):
        return
    user = update.effective_user
    logger.warning(f"🚫 Blocked unauthorized Telegram access: chat={getattr(update.effective_chat, 'id', None)} "
                   f"user={getattr(user, 'id', None)} @{getattr(user, 'username', None)}")
    raise ApplicationHandlerStop


# =====================================================================
# builders (blocking - call through asyncio.to_thread)
# =====================================================================
def build_status() -> str:
    e = engine
    state = "⏸️ PAUSED" if getattr(e, 'paused', False) else "▶️ running"
    lines = [f"🤖 <b>Bot status</b> [{esc(e.trading_mode.upper())}] {state}"]

    if e.strategy_mode == 'trend_hold':
        p = trend_hold.params(e.config)
        lines.append(f"📐 Strategy: <b>trend_hold</b> ({p['entry_days']}d breakout / {p['exit_days']}d exit)")
    else:
        lines.append(f"📐 Strategy: {esc(e.strategy_mode)}")

    g = e.gate_status()
    if g['enabled']:
        detail = ''
        if g['btc'] and g['ema']:
            detail = f" - BTC {price_fmt(g['btc'])} vs {g['days']}d EMA {price_fmt(g['ema'])} ({(g['btc'] / g['ema'] - 1) * 100:+.1f}%)"
        lines.append(f"🚦 BTC gate: {'🟢 OPEN' if g['allowed'] else '🔴 CLOSED (no new longs)'}{detail}")
    else:
        lines.append("🚦 BTC gate: off")

    equity = e._current_equity()
    peak = getattr(e, '_equity_peak', None)
    limit = float(e.config.get('max_drawdown', 0.05)) * 100
    if e.circuit_breaker_triggered:
        since = age(time.time() - e.circuit_breaker_time) if e.circuit_breaker_time else '?'
        lines.append(f"🚨 Circuit breaker: <b>TRIPPED</b> {since} ago - new entries paused")
    else:
        dd = ((peak - equity) / peak * 100) if (peak and equity is not None) else 0.0
        lines.append(f"🛡️ Circuit breaker: OK (drawdown {dd:.1f}% of {limit:.0f}% limit)")

    quotes = sorted({s.split('/')[1] for s in e.symbols if '/' in s})
    cash = max((e.get_cash_balance(q) for q in quotes), default=0.0)
    lines.append(f"💰 Equity: <b>{'$' + format(equity, ',.2f') if equity is not None else 'unavailable'}</b>"
                 f" | free cash ${cash:,.2f}")

    extra = ''
    if e.pending_entries:
        extra += f" | {len(e.pending_entries)} entry order(s) working"
    if e.pending_exits:
        extra += f" | {len(e.pending_exits)} exit(s) in progress"
    lines.append(f"📊 Positions: <b>{len(e.open_positions)}/{e.max_positions}</b>{extra}")

    if e.order_manager:
        bare = [s for s, p in e.open_positions.items()
                if p.get('side') == 'long' and not p.get('oco') and s not in e.pending_exits]
        if bare:
            lines.append(f"⚠️ No exchange stop on: {esc(', '.join(bare))}")
    return '\n'.join(lines)


def build_positions() -> str:
    e = engine
    if not e.open_positions and not e.pending_entries:
        return "📭 No open positions."
    px = prices()
    total = 0.0
    lines = [f"📊 <b>Open positions</b> ({len(e.open_positions)})"]
    for sym, p in sorted(e.open_positions.items()):
        price = px.get(sym) or p.get('current_price') or p['entry_price']
        entry, amt, side = p['entry_price'], p['amount'], p.get('side', 'long')
        pnl = (price - entry) * amt if side == 'long' else (entry - price) * amt
        pct = (price / entry - 1) * 100 if side == 'long' else (1 - price / entry) * 100
        total += pnl
        stop = p.get('stop_loss') or 0
        stop_txt = f"stop {price_fmt(stop)} ({(stop / price - 1) * 100:+.1f}%)" if stop else "no stop"
        held = '?'
        if p.get('entry_time'):
            try:
                held = age((datetime.now() - datetime.fromisoformat(p['entry_time'])).total_seconds())
            except ValueError:
                pass
        exiting = ' 🔚 exiting' if sym in e.pending_exits else ''
        lines.append(f"{'🟢' if pnl >= 0 else '🔴'} <b>{esc(sym)}</b> {pct:+.1f}% (${pnl:+,.2f}){exiting}\n"
                     f"   {price_fmt(entry)} → {price_fmt(price)} | {stop_txt} | held {held}\n"
                     f"   {protection_label(p)} | <i>{esc(p.get('signal_type', '?'))}</i>")
    if e.open_positions:
        lines.append(f"\nOpen P&amp;L: <b>${total:+,.2f}</b>")
    for sym, st in e.pending_entries.items():
        lines.append(f"⏳ <b>{esc(sym)}</b> entry order working: {st['qty']} @ {price_fmt(st.get('price'))}")
    return '\n'.join(lines)


def build_orders() -> str:
    e = engine
    lines = ["🧾 <b>Orders</b>"]
    now = time.time()
    if e.pending_entries:
        lines.append("\n<b>Entries working</b> (limit buy, repriced then cancelled)")
        for sym, st in e.pending_entries.items():
            lines.append(f"• {esc(sym)} {st['qty']} @ {price_fmt(st.get('price'))} "
                         f"(step {st.get('step', 0) + 1}, {age(now - st.get('created_at', now))})")
    if e.pending_exits:
        lines.append("\n<b>Exits in progress</b> (limit sell, market fallback)")
        for sym, ex in e.pending_exits.items():
            st = ex['st']
            lines.append(f"• {esc(sym)} {esc(ex['reason'])}{' (urgent)' if st.get('urgent') else ''} "
                         f"{age(now - st.get('created_at', now))}")
    prot = [(s, p) for s, p in e.open_positions.items() if p.get('oco')]
    if prot:
        lines.append("\n<b>Protective orders</b> (exchange-side)")
        for sym, p in prot:
            o = p['oco']
            tgt = f", target {price_fmt(o['target'])}" if o.get('target') else ''
            lines.append(f"• {esc(sym)} stop {price_fmt(o.get('stop'))} (limit {price_fmt(o.get('limit'))}){tgt}")
    if e.binance_client:
        try:
            oo = e.binance_client.get_open_orders()
            lines.append(f"\n<b>Open on the exchange</b>: {len(oo)}")
            for o in oo[:15]:
                lines.append(f"• {esc(o['symbol'])} {esc(o['side'])} {esc(o['type'])} {o['origQty']} @ {esc(o.get('price') or o.get('stopPrice'))}")
            if len(oo) > 15:
                lines.append(f"… and {len(oo) - 15} more")
        except Exception as ex:
            lines.append(f"\n⚠️ Could not read exchange orders: {esc(str(ex)[:80])}")
    if len(lines) == 1:
        lines.append("Nothing pending.")
    return '\n'.join(lines)


def build_trades(n: int = 10) -> str:
    from modules.portfolio import get_trade_history
    closed = [t for t in get_trade_history(1000)
              if t.get('action') == 'close' and t.get('mode', engine.trading_mode) == engine.trading_mode]
    if not closed:
        return "📭 No closed trades yet."
    last = closed[-n:][::-1]
    wins = sum(1 for t in closed if t.get('pnl', 0) > 0)
    total = sum(t.get('pnl', 0) for t in closed)
    lines = [f"📜 <b>Last {len(last)} closed trades</b>"]
    for t in last:
        when = str(t.get('timestamp', ''))[:16].replace('T', ' ')
        lines.append(f"{'🟢' if t.get('pnl', 0) >= 0 else '🔴'} {esc(t.get('symbol'))} "
                     f"{t.get('pnl_pct', 0):+.1f}% (${t.get('pnl', 0):+,.2f}) <i>{esc(t.get('reason', ''))}</i> {esc(when)}")
    lines.append(f"\nAll {len(closed)} trades: win rate {wins / len(closed) * 100:.0f}%, net <b>${total:+,.2f}</b>")
    return '\n'.join(lines)


def build_gate() -> str:
    g = engine.gate_status()
    if not g['enabled']:
        return "🚦 The BTC gate is <b>off</b> (set btc_gate_days in config.json)."
    lines = [f"🚦 <b>BTC gate</b>: {'🟢 OPEN - new longs allowed' if g['allowed'] else '🔴 CLOSED - no new longs'}"]
    if g['btc'] and g['ema']:
        lines.append(f"{esc(g['symbol'])} closed at {price_fmt(g['btc'])} vs its {g['days']}-day EMA {price_fmt(g['ema'])} "
                     f"({(g['btc'] / g['ema'] - 1) * 100:+.1f}%)")
    else:
        lines.append("BTC data unavailable or too short - the gate fails closed.")
    lines.append("Existing positions and their stops are not affected by the gate.")
    return '\n'.join(lines)


def build_breaker() -> str:
    e = engine
    equity, peak = e._current_equity(), getattr(e, '_equity_peak', None)
    limit = float(e.config.get('max_drawdown', 0.05)) * 100
    dd = ((peak - equity) / peak * 100) if (peak and equity is not None) else 0.0
    lines = ["🛡️ <b>Circuit breaker</b>"]
    if e.circuit_breaker_triggered:
        lines.append(f"🚨 <b>TRIPPED</b> {age(time.time() - e.circuit_breaker_time) if e.circuit_breaker_time else '?'} ago. "
                     f"New entries are paused; stops and exits still work.")
        lines.append(f"Resets by itself when drawdown < {limit * float(e.config.get('circuit_breaker_reset_ratio', 0.8)):.1f}% "
                     f"or after {e.config.get('circuit_breaker_cooldown_hours', 48)}h.")
        lines.append("Manual reset: /resetcircuitbreaker")
    else:
        lines.append(f"✅ OK - drawdown {dd:.1f}% from the peak (trips above {limit:.0f}%)")
    if peak:
        lines.append(f"Peak equity ${peak:,.2f}" + (f" | now ${equity:,.2f}" if equity is not None else ''))
    return '\n'.join(lines)


def do_close_all(urgent: bool) -> str:
    """Pause automatic entries, cancel working entry orders, then exit every position."""
    e = engine
    e.set_paused(True)
    cancelled = e.cancel_pending_entries()
    px = e.get_current_prices()
    reason = URGENT if urgent else NORMAL
    started, failed = [], []
    for sym, pos in list(e.open_positions.items()):
        if sym in e.pending_exits:
            continue
        try:
            price = px.get(sym) or pos.get('current_price') or pos['entry_price']
            (started if e.close_position(sym, price, reason) else failed).append(sym)
        except Exception as ex:
            failed.append(f"{sym} ({str(ex)[:30]})")
    out = ["⏸️ Trading is now <b>paused</b> (use /resume when you want new entries again)."]
    if cancelled:
        out.append(f"🧹 Cancelled {cancelled} working entry order(s).")
    if started:
        out.append(f"📤 Exiting: {esc(', '.join(started))} "
                   f"({'limit at the bid, market after 45s' if urgent else 'limit orders, market fallback after 3 min'})")
    if failed:
        out.append(f"❌ Could not start: {esc(', '.join(failed))}")
    if not started and not failed:
        out.append("📭 No open positions.")
    return '\n'.join(out)


# =====================================================================
# command handlers
# =====================================================================
KEYBOARD = ReplyKeyboardMarkup([["/status", "/positions"], ["/orders", "/gate"], ["/pause", "/resume"]],
                               resize_keyboard=True)


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = await asyncio.to_thread(build_status)
    await reply(update, text + "\n\nSend /help for all commands.", reply_markup=KEYBOARD)


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await reply(update, await asyncio.to_thread(build_status))


async def cmd_positions(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await reply(update, await asyncio.to_thread(build_positions))


async def cmd_orders(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await reply(update, await asyncio.to_thread(build_orders))


async def cmd_trades(update: Update, context: ContextTypes.DEFAULT_TYPE):
    n = 10
    if context.args:
        try:
            n = max(1, min(30, int(context.args[0])))
        except ValueError:
            pass
    await reply(update, await asyncio.to_thread(build_trades, n))


async def cmd_gate(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await reply(update, await asyncio.to_thread(build_gate))


async def cmd_breaker(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await reply(update, await asyncio.to_thread(build_breaker))


async def cmd_pause(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await asyncio.to_thread(engine.set_paused, True)
    await reply(update, "⏸️ <b>Paused.</b> No new automatic entries. Open positions, their stops and exits keep working.\n"
                        "/resume to continue.")


async def cmd_resume(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await asyncio.to_thread(engine.set_paused, False)
    note = ''
    if engine.circuit_breaker_triggered:
        note = "\n🚨 The circuit breaker is still tripped, so entries stay blocked until it resets (/breaker)."
    await reply(update, "▶️ <b>Resumed.</b> New entries are allowed again." + note)


async def cmd_close(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await reply(update, "Usage: <code>/close SYMBOL</code> (limit orders, market fallback after 3 min)\n"
                            "or <code>/close SYMBOL now</code> (limit at the bid, market after 45s)")
        return
    symbol = parse_symbol(context.args[0])
    urgent = len(context.args) > 1 and context.args[1].lower() in ('now', 'fast', 'urgent')

    def work() -> str:
        if symbol is None:
            return f"❓ Unknown symbol <code>{esc(context.args[0])}</code>."
        if symbol in engine.pending_entries and symbol not in engine.open_positions:
            engine.pending_entries[symbol]['cancel_requested'] = True
            engine.manage_orders()
            return f"🧹 Cancelled the working entry order for <b>{esc(symbol)}</b>."
        pos = engine.open_positions.get(symbol)
        if not pos:
            return f"📭 No open position in <b>{esc(symbol)}</b>."
        if symbol in engine.pending_exits:
            return f"🔚 <b>{esc(symbol)}</b> is already being closed."
        price = engine.get_current_prices().get(symbol) or pos.get('current_price') or pos['entry_price']
        ok = engine.close_position(symbol, price, URGENT if urgent else NORMAL)
        if not ok:
            return f"❌ Could not start closing <b>{esc(symbol)}</b> - check /orders and the logs."
        return (f"📤 Closing <b>{esc(symbol)}</b> "
                f"({'limit at the bid, market after 45s' if urgent else 'limit orders, market fallback after 3 min'}). "
                f"You will get a message when it is done.")

    await reply(update, await asyncio.to_thread(work))


def _confirm_keyboard(action: str, label: str) -> InlineKeyboardMarkup:
    stamp = int(time.time())
    return InlineKeyboardMarkup([[InlineKeyboardButton(label, callback_data=f"{action}:yes:{stamp}"),
                                  InlineKeyboardButton("❌ Cancel", callback_data=f"{action}:no:{stamp}")]])


async def cmd_sellall(update: Update, context: ContextTypes.DEFAULT_TYPE):
    n = len(engine.open_positions)
    await reply(update, f"🚨 <b>Sell ALL {n} position(s) now?</b>\nThis pauses trading, cancels working entry orders and "
                        f"exits everything (limit at the bid, market after 45s).",
                reply_markup=_confirm_keyboard('sellall', '🚨 Yes, sell everything'))


async def cmd_closeall(update: Update, context: ContextTypes.DEFAULT_TYPE):
    n = len(engine.open_positions)
    await reply(update, f"Close <b>all {n}</b> position(s) in an orderly way?\nPauses trading, cancels working entry orders, "
                        f"sells with limit orders (market fallback after 3 min).",
                reply_markup=_confirm_keyboard('closeall', '✅ Yes, close all'))


async def cmd_stop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await reply(update, "🛑 <b>Stop the bot process?</b>\nExchange-side stop orders stay on the exchange, but nothing will "
                        "manage the positions (no ratcheting, no market fallback) until it is started again on the server. "
                        "If you only want to stop new trades, use /pause instead.",
                reply_markup=_confirm_keyboard('stop', '🛑 Yes, stop the bot'))


async def on_confirm(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    try:
        action, answer, stamp = q.data.split(':')
        fresh = time.time() - int(stamp) <= CONFIRM_TTL_SECONDS
    except ValueError:
        return
    if answer == 'no':
        await q.edit_message_text("Cancelled. Nothing was done.")
        return
    if not fresh:
        await q.edit_message_text(f"⌛ That confirmation expired (over {CONFIRM_TTL_SECONDS}s). Send the command again.")
        return
    if action == 'stop':
        await q.edit_message_text("🛑 Stopping the bot process now. (Exchange-side stops stay in place.)")
        import services.telegram_bot as tb
        if tb.stop_event:
            tb.stop_event.set()              # the main loop exits, then stops the scheduler and shuts down cleanly
        return
    await q.edit_message_text("⏳ Working...")
    text = await asyncio.to_thread(do_close_all, action == 'sellall')
    await q.edit_message_text(text, parse_mode=ParseMode.HTML)


# =====================================================================
# help + menu + registration
# =====================================================================
# (command, description shown in Telegram's menu)
MENU_COMMANDS = [
    ('status', 'Dashboard: mode, gate, breaker, equity'),
    ('positions', 'Open positions with stops and P&L'),
    ('orders', 'Working orders and protective stops'),
    ('trades', 'Recent closed trades'),
    ('gate', 'BTC trend gate'),
    ('breaker', 'Circuit breaker'),
    ('pause', 'Pause new entries'),
    ('resume', 'Resume new entries'),
    ('close', 'Close one position: /close SYMBOL [now]'),
    ('closeall', 'Close everything (orderly)'),
    ('sellall', 'EMERGENCY: sell everything'),
    ('help', 'All commands'),
]

HELP_TEXT = """<b>🤖 Commands</b>

<b>Overview</b>
/status - mode, strategy, BTC gate, breaker, equity
/positions - each position: P&amp;L, stop distance, exchange stop
/orders - working entries/exits, protective stops, exchange orders
/trades [N] - last closed trades and win rate
/gate - BTC trend gate details
/breaker - circuit breaker details

<b>Control</b>
/pause - no new automatic entries (positions and stops keep working)
/resume - allow new entries again
/close SYMBOL [now] - close one position (<code>/close BTC</code>)
/closeall - close everything with limit orders
/sellall - EMERGENCY: pause and sell everything fast
/resetcircuitbreaker - reset the breaker manually
/stop - stop the bot process (asks first)

<b>Manual trading</b> (older commands, unchanged)
/scan  /execute SYMBOL  /executeall
/limitbuy SYMBOL AMOUNT PRICE [STOP] [TARGET]
/limitsell SYMBOL AMOUNT PRICE
/setstop SYMBOL STOP [TARGET]
/cancelorder ID  /cancelsymbol SYMBOL  /cancelall
/price SYMBOL  /balance  /summary  /syncpositions

Only this chat can use the bot. Dangerous commands ask for confirmation."""


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await reply(update, HELP_TEXT, reply_markup=KEYBOARD)


def menu() -> List[BotCommand]:
    return [BotCommand(c, d[:250]) for c, d in MENU_COMMANDS]


HANDLERS = [
    (['start'], cmd_start), (['help'], cmd_help),
    (['status'], cmd_status), (['positions'], cmd_positions),
    (['orders', 'openorders', 'pendingorders'], cmd_orders),          # old names now show the full picture
    (['trades'], cmd_trades), (['gate'], cmd_gate), (['breaker'], cmd_breaker),
    (['pause'], cmd_pause), (['resume'], cmd_resume),
    (['close'], cmd_close), (['closeall'], cmd_closeall),
    (['sellall'], cmd_sellall), (['stop'], cmd_stop),
]


def register_commands(application):
    """Call BEFORE adding the older handlers: the first matching handler in a group wins."""
    application.add_handler(TypeHandler(Update, guard), group=-1)     # runs first, for every update
    for names, fn in HANDLERS:
        application.add_handler(CommandHandler(names, fn))
    application.add_handler(CallbackQueryHandler(on_confirm, pattern=r'^(sellall|closeall|stop):(yes|no):\d+$'))
