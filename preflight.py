"""
preflight.py - "is this bot ready to run on testnet?" (read-only; never prints secrets)

    python preflight.py                     # uses TRADING_MODE from .env
    TRADING_MODE=testnet python preflight.py    # bash: check testnet without editing .env
    $env:TRADING_MODE="testnet"; python preflight.py   # PowerShell

It logs in, reads balances, checks that your coins exist on the exchange, compares testnet
prices with the real market, checks your config for the trend_hold strategy, and looks for
leftover state. Exit code 1 if anything FAILs. It places no orders.
"""
import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RESULTS = []          # (status, name, detail)
MIN_QUOTE_BALANCE = 200.0
MAX_PRICE_GAP_PCT = 0.30      # entries never chase more than this above the signal price


def record(status, name, detail=''):
    RESULTS.append((status, name, detail))
    icon = {'PASS': '  OK ', 'WARN': ' WARN', 'FAIL': ' FAIL', 'INFO': ' info'}[status]
    print(f"[{icon}] {name}" + (f" - {detail}" if detail else ''))


def http_json(url, timeout=20):
    return json.load(urllib.request.urlopen(url, timeout=timeout))


def main() -> int:
    print("=" * 64)
    print("PREFLIGHT")
    print("=" * 64)

    # ---- 1. configuration loads ------------------------------------------------
    try:
        from config_loader import config, get_binance_client
        cfg = config.config
    except Exception as e:
        record('FAIL', 'config loads', f"{type(e).__name__}: {e}")
        return _finish()
    mode = cfg['trading_mode']
    record('PASS' if mode in ('testnet', 'live') else 'WARN', f"trading mode = {mode}",
           {'testnet': 'orders go to the Binance TESTNET (fake money)',
            'live': 'REAL MONEY',
            'paper': "paper mode cannot place orders in this bot - set TRADING_MODE=testnet"}.get(mode, ''))
    if mode == 'live':
        record('WARN', 'this is LIVE mode', 'preflight is read-only, but everything after this is real money')
    if mode != 'paper':
        record('INFO', f"auth method = {cfg.get('auth_method')}")

    # ---- 2. strategy configuration ---------------------------------------------
    strat = cfg.get('strategy_mode') or 'legacy'
    record('INFO', f"strategy_mode = {strat}")
    if strat == 'trend_hold':
        th = cfg.get('trend_hold') or {}
        record('PASS', 'trend_hold settings', f"entry {th.get('entry_days', 20)}d / exit {th.get('exit_days', 10)}d, "
                                            f"BTC gate {cfg.get('btc_gate_days', 50)}d, max_positions {cfg.get('max_positions')}")
        if not cfg.get('trend_hold_live_enabled', False):
            record('FAIL', 'trend_hold_live_enabled', 'is false, so the bot will NOT open any trend_hold entries. '
                                                       'Set "trend_hold_live_enabled": true in config.json for testnet.')
    else:
        record('WARN', 'strategy_mode is not trend_hold',
               'the bot will run the legacy strategy; set "strategy_mode": "trend_hold" to use the tested design')
    if cfg.get('enable_shorts', False):
        record('WARN', 'enable_shorts is true', 'shorts need the futures path, which has no exchange-side stops yet')
    else:
        record('PASS', 'shorts disabled')
    if cfg.get('futures_live_enabled', False):
        record('WARN', 'futures_live_enabled is true', 'live futures orders are unprotected; leave this false')
    if not cfg.get('use_limit_orders', True):
        record('WARN', 'use_limit_orders is false', 'entries/exits will be market orders, stops software-only')
    coins = cfg.get('coins', [])
    record('INFO', f"{len(coins)} coins configured")

    # ---- 3. exchange login + balances ------------------------------------------
    client = None
    if mode in ('testnet', 'live'):
        try:
            client = get_binance_client()
            acct = client.get_account()
            record('PASS', 'login', f"canTrade={acct.get('canTrade')}")
            if not acct.get('canTrade'):
                record('FAIL', 'account cannot trade', 'check the API key permissions')
            free = {b['asset']: float(b['free']) for b in acct['balances'] if float(b['free']) > 0}
            quotes = sorted({c.split('/')[1] for c in coins if '/' in c})
            for q in quotes:
                bal = free.get(q, 0.0)
                st = 'PASS' if bal >= MIN_QUOTE_BALANCE else 'WARN'
                record(st, f"{q} balance", f"{bal:,.2f}" + ('' if st == 'PASS' else
                       f" (< {MIN_QUOTE_BALANCE:.0f}: positions are capped at 15% of it and orders under $10 are rejected)"))
            try:
                skew = abs(client.get_server_time()['serverTime'] / 1000 - time.time())
                record('PASS' if skew < 1 else 'WARN', 'clock vs Binance', f"{skew:.2f}s difference"
                       + ('' if skew < 1 else ' - sync your system clock or requests get rejected (-1021)'))
            except Exception as e:
                record('WARN', 'clock check', str(e)[:80])
            open_orders = client.get_open_orders()
            record('PASS' if not open_orders else 'WARN', 'open orders on the account', str(len(open_orders)))
        except Exception as e:
            code = getattr(e, 'code', None)
            hint = ''
            if code == -2015:
                hint = (" -> the exchange does not recognise this API key. On testnet this usually means the key "
                        "was wiped in a testnet reset: generate a new one at https://testnet.binance.vision and "
                        "update BINANCE_TESTNET_API_KEY (+ BINANCE_TESTNET_SECRET_KEY for an HMAC key).")
            elif code == -1022:
                hint = " -> signature rejected: the private key/secret does not match this API key."
            record('FAIL', 'login', f"{type(e).__name__} {str(e)[:120]}{hint}")

    # ---- 4. symbols + price sanity ---------------------------------------------
    base = 'https://testnet.binance.vision' if mode == 'testnet' else 'https://api.binance.com'
    try:
        info = http_json(f"{base}/api/v3/exchangeInfo")
        trading = {s['symbol'] for s in info['symbols'] if s['status'] == 'TRADING'}
        missing = [c for c in coins if c.replace('/', '') not in trading]
        if missing:
            record('WARN', 'coins not tradable on this exchange', f"{', '.join(missing)} (the bot will ignore them)")
        record('PASS', f"{len(coins) - len(missing)}/{len(coins)} coins tradable")
    except Exception as e:
        record('WARN', 'symbol check', str(e)[:100])
        trading = set()
    if mode == 'testnet':
        worst, worst_c = 0.0, ''
        for c in coins:
            s = c.replace('/', '')
            if s not in trading:
                continue
            try:
                t = float(http_json(f"{base}/api/v3/ticker/price?symbol={s}")['price'])
                m = float(http_json(f"https://api.binance.com/api/v3/ticker/price?symbol={s}")['price'])
                gap = (t / m - 1) * 100
                if abs(gap) > abs(worst):
                    worst, worst_c = gap, c
            except Exception:
                continue
        if worst_c:
            up = worst > MAX_PRICE_GAP_PCT
            record('WARN' if up else 'PASS', 'testnet vs real prices',
                   f"largest gap {worst:+.2f}% ({worst_c})" +
                   (f"; testnet ABOVE real by more than {MAX_PRICE_GAP_PCT}% means buy limits may not fill" if up else
                    " (signals use real candles, orders use the testnet book)"))

    # ---- 5. market data + BTC gate ---------------------------------------------
    try:
        from modules import market_gate, trend_hold
        from modules.data_feed import fetch_ohlcv
        tf = cfg.get('trading_timeframe', '4h')
        need = trend_hold.history_needed(cfg) if strat == 'trend_hold' else 200
        btc = next((c for c in ('BTC/USDC', 'BTC/USDT') if c in coins), 'BTC/USDC')
        df = fetch_ohlcv(btc, tf, 1000)
        record('PASS' if len(df) >= need else 'FAIL', f"candle history ({btc} {tf})", f"{len(df)} fetched, need {need}")
        if strat == 'trend_hold':
            up = market_gate.btc_uptrend(market_gate.closed_only(df, tf), int(cfg.get('btc_gate_days', 50)), tf)
            record('INFO', 'BTC gate right now',
                   {True: 'OPEN (BTC above its EMA: new longs allowed)', False: 'CLOSED (BTC below its EMA: no new longs)',
                    None: 'not enough history (fails closed)'}[up])
    except Exception as e:
        record('FAIL', 'market data', f"{type(e).__name__}: {str(e)[:100]}")

    # ---- 6. leftover state -----------------------------------------------------
    root = os.path.dirname(os.path.abspath(__file__))
    try:
        pf = json.load(open(os.path.join(root, 'portfolio.json')))
        n = len(pf.get('positions', {})) + len(pf.get('futures_positions', {}))
        record('PASS' if n == 0 else 'WARN', 'positions recorded in portfolio.json',
               str(n) + ('' if n == 0 else ' - make sure they match what the exchange holds'))
    except Exception as e:
        record('WARN', 'portfolio.json', str(e)[:80])
    for name, what in (('pending_orders.json', 'unfinished entry/exit orders'),
                       ('breaker_state.json', 'circuit-breaker peak equity / state')):
        p = os.path.join(root, name)
        if os.path.exists(p):
            try:
                data = json.load(open(p))
                extra = ''
                if name == 'breaker_state.json' and data.get('triggered'):
                    extra = ' - breaker is ACTIVE'
                record('WARN' if extra or (name == 'pending_orders.json' and (data.get('entries') or data.get('exits'))) else 'INFO',
                       f"{name} exists", what + extra)
            except Exception:
                record('WARN', f"{name} unreadable")
        else:
            record('INFO', f"{name} not present", 'fresh start')
    record('PASS' if cfg.get('telegram_token') and cfg.get('telegram_chat_id') else 'INFO', 'Telegram alerts',
           'configured' if cfg.get('telegram_token') else 'not configured (the bot runs without them)')

    return _finish()


def _finish() -> int:
    fails = sum(1 for s, _, _ in RESULTS if s == 'FAIL')
    warns = sum(1 for s, _, _ in RESULTS if s == 'WARN')
    print("=" * 64)
    print(f"{fails} FAIL, {warns} WARN" + ("  -> fix the FAILs before starting the bot" if fails else "  -> ready to start"))
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
