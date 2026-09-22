"""
Portfolio-level backtest with a fast, repeatable experiment workflow.

Workflow
--------
1. Price data is SNAPSHOTTED to backtest_data/ the first time (refresh with --refresh), so every
   experiment sees identical candles. Without this, results move every day as new candles arrive.
2. Signals (regime + generate_trade_signal) are PRECOMPUTED per coin, in parallel, and cached
   keyed by a fingerprint of the strategy code, regime model and signal-related config. Changing
   exits, sizing, side filters, dates, etc. re-runs in seconds; editing strategy code recomputes.
3. Every run is appended to backtest_results/experiments.csv (use --tag), so you can see how
   many variants you have tried. The more you try, the less any single good result means.

Examples
--------
  python backtest.py --days 365 --capital 1000 --tag baseline
  python backtest.py --side long --period dev --tag long_dev          # tune on the first 60%
  python backtest.py --side long --period holdout --tag long_holdout  # judge on the untouched 40%
  python backtest.py --signals trend breakout --exclude range         # substring filters
"""
import sys
import os
import io
import logging

# One thread per process: the regime model (XGBoost, n_jobs=-1) and numpy would otherwise each
# start a thread pool per worker (80+ threads on a 32-core box) and spin against each other,
# burning 100% CPU with almost no progress. Must be set before numpy / xgboost are imported;
# spawned workers inherit it. Override by setting the variables yourself.
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

# 1. ENCODING FIX
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    os.environ["PYTHONIOENCODING"] = "utf-8"

# 2. SAFE LOGGING
class UTF8Handler(logging.StreamHandler):
    def emit(self, record):
        try:
            msg = self.format(record)
            self.stream.write(msg + self.terminator)
            self.flush()
        except UnicodeEncodeError:
            msg = self.format(record).encode('ascii', 'ignore').decode('ascii')
            self.stream.write(msg + self.terminator)
            self.flush()

root_logger = logging.getLogger()
for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)
handler = UTF8Handler(sys.stdout)
root_logger.addHandler(handler)
root_logger.setLevel(logging.WARNING)

# 3. IMPORTS
import argparse
import csv
import hashlib
import json
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(PROJECT_ROOT))

from modules.data_feed import fetch_historical_data
from modules.regime_switcher import predict_regime
from modules.exit_manager import evaluate_exit
from modules.strategy_tools import generate_trade_signal
from config_loader import config as _cfg, get_pair_config
import modules.regime_switcher as _rs
from modules import trend_hold as _trend_hold


def apply_model_dir(path: str):
    """Use the regime model files from another folder (e.g. backups/regime_models_...).
    Exported through an env var so spawned worker processes pick it up too."""
    path = os.path.abspath(path)
    os.environ['BACKTEST_MODEL_DIR'] = path
    _rs.MODEL_PATH = os.path.join(path, 'regime_model.pkl')
    _rs.SCALER_PATH = os.path.join(path, 'regime_scaler.pkl')
    _rs.MAPPING_PATH = os.path.join(path, 'regime_class_mapping.pkl')
    _rs.model = None                      # force a reload from the new location


if os.environ.get('BACKTEST_MODEL_DIR'):
    apply_model_dir(os.environ['BACKTEST_MODEL_DIR'])


def apply_trend_hold_params(entry_days=None, exit_days=None, stage2_gain=None, stage2_days=None):
    """Override trend_hold day counts (exported by env var so worker processes match)."""
    over = {k: v for k, v in (('entry_days', entry_days), ('exit_days', exit_days),
                              ('stage2_gain', stage2_gain), ('stage2_exit_days', stage2_days)) if v}
    if over:
        os.environ['BACKTEST_TREND_HOLD'] = json.dumps(over)
        _cfg.config['trend_hold'] = {**(_cfg.config.get('trend_hold') or {}), **over}


if os.environ.get('BACKTEST_TREND_HOLD'):
    _cfg.config['trend_hold'] = {**(_cfg.config.get('trend_hold') or {}),
                                 **json.loads(os.environ['BACKTEST_TREND_HOLD'])}


def apply_strategy(mode: str):
    """'legacy' (regime + indicator signals) or 'trend_hold'. Exported via an env var so the
    spawned worker processes use the same strategy."""
    os.environ['BACKTEST_STRATEGY'] = mode
    if mode == 'trend_hold':
        _cfg.config['strategy_mode'] = 'trend_hold'
    else:
        _cfg.config.pop('strategy_mode', None)


if os.environ.get('BACKTEST_STRATEGY'):
    apply_strategy(os.environ['BACKTEST_STRATEGY'])

# Mirrors the live engine: 200 candles for signals, 50 for exit evaluation
SIGNAL_WINDOW = 200
EXIT_WINDOW = 50
WARMUP = 100
# regime_switcher.train_model() trains on the most recent 500 candles per coin,
# so entries inside that span are in-sample for the regime model.
REGIME_TRAIN_CANDLES = 500

# Signals are precomputed at a huge nominal equity so nothing is rejected for being "too small";
# position size is exactly linear in equity, so it is rescaled to the real free cash at fill time.
PRECOMPUTE_EQUITY = 1_000_000.0
MIN_ORDER_USD = 10.0

DATA_DIR = os.path.join(PROJECT_ROOT, 'backtest_data')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'backtest_results')

# Config keys that change what generate_trade_signal returns (part of the cache fingerprint)
SIGNAL_CONFIG_KEYS = ['stop_loss_min_pct', 'stop_loss_max_pct', 'default_stop_loss_pct', 'min_rr',
                      'trading_fee', 'use_volume_shrinkage', 'use_daily_trend_filter', 'per_pair',
                      'risk_per_trade', 'stop_atr_multiplier', 'trading_timeframe',
                      'strategy_mode', 'trend_hold']
REGIME_CODE_FILES = ['modules/regime_switcher.py']
SIGNAL_CODE_FILES = ['modules/strategy_tools.py', 'modules/indicators.py', 'modules/trend_hold.py']
MODEL_FILES = ['regime_model.pkl', 'regime_scaler.pkl', 'regime_class_mapping.pkl']


def _safe(symbol: str) -> str:
    return symbol.replace('/', '_')


def _hash_files(h, rels, base=None):
    for rel in rels:
        try:
            with open(os.path.join(base or PROJECT_ROOT, rel), 'rb') as f:
                h.update(f.read())
        except OSError:
            h.update(f'missing:{rel}'.encode())


def regime_fingerprint(cfg: dict) -> str:
    """Changes only when the regime code or model (or the timeframe) changes.
    Regime prediction is ~88% of the precompute time, so it is cached separately."""
    h = hashlib.sha1()
    _hash_files(h, REGIME_CODE_FILES)
    _hash_files(h, MODEL_FILES, base=os.environ.get('BACKTEST_MODEL_DIR'))
    h.update(f"{cfg.get('trading_timeframe')}|{SIGNAL_WINDOW}|{WARMUP}".encode())
    return h.hexdigest()[:12]


def signal_fingerprint(cfg: dict) -> str:
    """Changes whenever strategy code, signal-related config or the regimes change."""
    h = hashlib.sha1()
    _hash_files(h, SIGNAL_CODE_FILES)
    h.update(json.dumps({k: cfg.get(k) for k in SIGNAL_CONFIG_KEYS}, sort_keys=True, default=str).encode())
    h.update(f'{PRECOMPUTE_EQUITY}|{regime_fingerprint(cfg)}'.encode())
    return h.hexdigest()[:12]


def code_fingerprint(cfg: dict) -> str:
    return signal_fingerprint(cfg)


def regime_coin(args) -> Tuple[str, Dict[int, str]]:
    """Regime string for every bar of one coin (runs in a worker process)."""
    symbol, df, use_regime = args
    out: Dict[int, str] = {}
    for idx in range(WARMUP, len(df) - 1):
        window = df.iloc[max(0, idx - SIGNAL_WINDOW + 1): idx + 1].copy()
        try:
            out[idx] = predict_regime(window) if use_regime else "Trend UPTREND"
        except Exception:
            out[idx] = "unknown"
        if idx == WARMUP:                       # model is lazily loaded by the first prediction
            _single_thread_model()
    return symbol, out


def _single_thread_model():
    """The saved XGBoost model has n_jobs=-1; a single-row predict gains nothing from threads."""
    try:
        import modules.regime_switcher as rs
        if getattr(rs, 'model', None) is not None and hasattr(rs.model, 'set_params'):
            rs.model.set_params(n_jobs=1)
    except Exception:
        pass


def signal_coin(args) -> Tuple[str, Dict[int, dict]]:
    """Signal for every bar of one coin, given its regimes (runs in a worker process).
    Returns {bar_index: signal_dict}; bars with no signal (or a volatile regime) are omitted."""
    symbol, df, regimes, base_risk = args
    out: Dict[int, dict] = {}
    slow = _cfg.config.get('strategy_mode') == 'trend_hold'
    n_window = max(SIGNAL_WINDOW, _trend_hold.history_needed(_cfg.config)) if slow else SIGNAL_WINDOW
    for idx, regime_str in regimes.items():
        # Live scan_and_trade skips volatile ("Expansion (Volatile Chop)") regimes;
        # trend_hold does not use regimes at all.
        if not slow and ("Volatile" in regime_str or "Expansion" in regime_str):
            continue
        window = df.iloc[max(0, idx - n_window + 1): idx + 1].copy()
        sig = generate_trade_signal(df=window, equity=PRECOMPUTE_EQUITY, risk_per_trade=base_risk,
                                    symbol=symbol, regime=regime_str)
        if (sig and sig.get('side') in ('long', 'short')
                and sig.get('units', 0) > 0 and sig.get('entry_price', 0) > 0):
            sig['regime_label'] = regime_str.split('(')[0].strip()
            out[idx] = sig
    return symbol, out


class Backtester:
    """
    All coins share ONE timeline, one equity pool, the max_positions limit and the drawdown
    circuit breaker, like the live bot.

    Fill model:
      - signal on bar i close -> entry at bar i+1 open (+ slippage), SL/TP re-anchored
      - stops / take-profits are checked intrabar with the bar's high/low
        (stop wins if both are touched in the same bar; gaps fill at the open)
      - indicator exits (MACD exhaustion, signal flip, trailing update) act on the close
    """

    def __init__(self, coins: List[str], days: int = 365, capital: float = 5000.0,
                 use_regime: bool = True, slippage: float = 0.0005,
                 honor_exit_strategy: bool = False, verbose: bool = False,
                 side: Optional[str] = None, include: Optional[List[str]] = None,
                 exclude: Optional[List[str]] = None, period: str = 'all',
                 dev_frac: float = 0.6, start: Optional[str] = None, end: Optional[str] = None,
                 workers: Optional[int] = None, refresh: bool = False, fetch_days: int = 730,
                 tag: Optional[str] = None, use_cache: bool = True, log_experiment: bool = True,
                 data_dir: Optional[str] = None, results_dir: Optional[str] = None,
                 btc_gate_days: int = 0, gate_shorts: bool = False, fixed_tp: float = 0.0,
                 trail_min: Optional[float] = None, trail_max: Optional[float] = None):
        self.cfg = _cfg.config
        self.btc_gate_days = btc_gate_days
        self.gate_shorts = gate_shorts
        self.fixed_tp = fixed_tp
        self.coins = coins
        self.days = days
        self.use_regime = use_regime
        self.verbose = verbose
        self.initial_equity = float(capital)
        self.slippage = slippage
        self.honor_exit_strategy = honor_exit_strategy
        self.side = side or ('both' if self.cfg.get('enable_shorts', False) else 'long')
        self.include = [s.lower() for s in (include or [])]
        self.exclude = [s.lower() for s in (exclude or [])]
        self.period = period
        self.dev_frac = dev_frac
        self.start, self.end = start, end
        self.workers = workers
        self.refresh = refresh
        self.fetch_days = fetch_days
        self.tag = tag
        self.use_cache = use_cache
        self.log_experiment = log_experiment
        self.data_dir = data_dir or DATA_DIR
        self.results_dir = results_dir or RESULTS_DIR

        self.timeframe        = self.cfg.get('trading_timeframe', '4h')
        self.base_risk        = float(self.cfg.get('risk_per_trade', 0.03))
        self.spot_fee         = float(self.cfg.get('spot_fee', self.cfg.get('trading_fee', 0.001)))
        self.futures_fee      = float(self.cfg.get('futures_fee', self.cfg.get('trading_fee', 0.0005)))
        self.trail_min        = self.cfg.get('trailing_stop_min_pct', 0.04)
        self.trail_max        = self.cfg.get('trailing_stop_max_pct', 0.08)
        self.trail_activation = self.cfg.get('trailing_activation_pct', 0.15)
        self.max_candles      = self.cfg.get('max_trade_candles', 336)
        # explicit trailing bounds override the config AND the per-coin values, for every coin
        self.trail_override   = (trail_min, trail_max) if trail_min is not None and trail_max is not None else None
        self.max_positions    = int(self.cfg.get('max_positions', 3))
        self.max_drawdown     = float(self.cfg.get('max_drawdown', 0.05))
        self.breaker_reset_ratio    = float(self.cfg.get('circuit_breaker_reset_ratio', 0.8))
        self.breaker_cooldown_hours = float(self.cfg.get('circuit_breaker_cooldown_hours', 48))
        self.default_coins    = self.cfg.get('coins', ['BNB/USDT'])

        self.trades: List[dict] = []
        self.equity_curve: List[tuple] = []   # (timestamp, mark-to-market equity)
        self.breaker_trips: List = []
        self.breaker_bars = 0
        self.window: Tuple = (None, None)     # simulated (start, end) timestamps
        self.benchmark: Optional[dict] = None
        self._data_lengths: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Data snapshot
    # ------------------------------------------------------------------
    def _snapshot_path(self, coin: str) -> str:
        return os.path.join(self.data_dir, f'{_safe(coin)}_{self.timeframe}.csv')

    def _load_data(self) -> Dict[str, pd.DataFrame]:
        os.makedirs(self.data_dir, exist_ok=True)
        data: Dict[str, pd.DataFrame] = {}
        for coin in self.coins:
            path = self._snapshot_path(coin)
            df = None
            if os.path.exists(path) and not self.refresh:
                df = pd.read_csv(path, parse_dates=['timestamp'])
                print(f"  {coin}: snapshot {len(df)} candles "
                      f"({df['timestamp'].iloc[0]:%Y-%m-%d} -> {df['timestamp'].iloc[-1]:%Y-%m-%d})", flush=True)
            else:
                print(f"  {coin}: downloading {self.fetch_days} days...", flush=True)
                df = fetch_historical_data(coin, interval=self.timeframe, days=self.fetch_days)
                if df is not None and not df.empty:
                    df = df.sort_values('timestamp').reset_index(drop=True)
                    df.to_csv(path, index=False)
            if df is None or df.empty or len(df) <= WARMUP + 1:
                print(f"  No/insufficient data for {coin}, skipping", flush=True)
                continue
            data[coin] = df.sort_values('timestamp').reset_index(drop=True)
        return data

    # ------------------------------------------------------------------
    # Signal precompute + cache
    # ------------------------------------------------------------------
    def _cache_path(self, kind: str, coin: str, fp: str) -> str:
        return os.path.join(self.data_dir, kind, self.timeframe, f'{_safe(coin)}_{fp}.pkl')

    @staticmethod
    def _data_fingerprint(df: pd.DataFrame) -> str:
        return f"{df['timestamp'].iloc[0]}|{df['timestamp'].iloc[-1]}|{len(df)}"

    def _cached_compute(self, kind: str, label: str, fp: str, data: Dict[str, pd.DataFrame],
                        job_for, worker_fn) -> Dict[str, dict]:
        """Per-coin disk cache + parallel compute. job_for(coin) -> args tuple for worker_fn."""
        results: Dict[str, dict] = {}
        todo = []
        for coin, df in data.items():
            path = self._cache_path(kind, coin, fp)
            if self.use_cache and os.path.exists(path):
                try:
                    with open(path, 'rb') as f:
                        cached = pickle.load(f)
                    if cached.get('data_fp') == self._data_fingerprint(df):
                        results[coin] = cached['result']
                        continue
                except Exception:
                    pass
            todo.append(coin)

        print(f"{label}: {len(results)} coins from cache, {len(todo)} to compute (fingerprint {fp})", flush=True)
        if not todo:
            return results

        jobs = [job_for(c) for c in todo]
        # workers are single-threaded now, so use roughly one per core (leave a couple free)
        workers = (self.workers if self.workers is not None
                   else max(1, min((os.cpu_count() or 2) - 2, len(jobs), 16)))
        started = time.time()

        def _store(coin, result):
            results[coin] = result
            if self.use_cache:
                path = self._cache_path(kind, coin, fp)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, 'wb') as f:
                    pickle.dump({'data_fp': self._data_fingerprint(data[coin]), 'result': result}, f)
            print(f"  {coin}: {len(result)} entries ({time.time() - started:.0f}s elapsed)", flush=True)

        if workers > 1 and len(jobs) > 1:
            print(f"  computing with {workers} worker processes...", flush=True)
            with ProcessPoolExecutor(max_workers=workers) as pool:
                for coin, result in pool.map(worker_fn, jobs):
                    _store(coin, result)
        else:
            for job in jobs:
                coin, result = worker_fn(job)
                _store(coin, result)
        return results

    def _get_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[int, dict]]:
        if self.cfg.get('strategy_mode') == 'trend_hold':
            print("Strategy: trend_hold (no regime model needed)", flush=True)
            regimes = {c: {i: 'trend_hold' for i in range(WARMUP, len(df) - 1)} for c, df in data.items()}
            return self._cached_compute(
                'signals', 'Signals', signal_fingerprint(self.cfg), data,
                lambda c: (c, data[c], regimes[c], self.base_risk), signal_coin)
        # 1) regimes: slow (73 ms/bar), only invalidated by regime code / model / timeframe
        regimes = self._cached_compute(
            'regimes', 'Regimes', regime_fingerprint(self.cfg), data,
            lambda c: (c, data[c], self.use_regime), regime_coin)
        # 2) signals: fast (10 ms/bar), invalidated by strategy code / signal config
        return self._cached_compute(
            'signals', 'Signals', signal_fingerprint(self.cfg), data,
            lambda c: (c, data[c], regimes[c], self.base_risk), signal_coin)

    # ------------------------------------------------------------------
    def _accept(self, sig: dict) -> bool:
        if self.side != 'both' and sig['side'] != self.side:
            return False
        st = str(sig.get('signal_type', '')).lower()
        if self.include and not any(k in st for k in self.include):
            return False
        if self.exclude and any(k in st for k in self.exclude):
            return False
        return True

    BARS_PER_DAY = {'1m': 1440, '5m': 288, '15m': 96, '30m': 48, '1h': 24, '2h': 12,
                    '4h': 6, '6h': 4, '12h': 2, '1d': 1}

    def _build_btc_gate(self, data: Dict[str, pd.DataFrame], all_ts: List) -> Dict:
        """{timestamp: True} when BTC's close is above its N-day EMA. Uses only data up to and
        including each bar (ewm with adjust=False), so there is no look-ahead. Until N days of
        history exist the gate is closed (no longs)."""
        btc = next((c for c in ('BTC/USDC', 'BTC/USDT') if c in data), None)
        if btc is None:
            raise SystemExit("--btc-gate needs BTC/USDC or BTC/USDT in the coin list")
        span = int(self.btc_gate_days * self.BARS_PER_DAY.get(self.timeframe, 6))
        df = data[btc]
        close = df.set_index('timestamp')['close']
        up = (close > close.ewm(span=span, adjust=False).mean())
        up.iloc[:span] = False
        return up.reindex(all_ts, method='ffill').fillna(False).to_dict()

    def _select_timeline(self, all_ts: List) -> List:
        end_ts = pd.Timestamp(self.end) if self.end else max(all_ts)
        start_ts = pd.Timestamp(self.start) if self.start else end_ts - pd.Timedelta(days=self.days)
        timeline = [t for t in all_ts if start_ts <= t <= end_ts]
        if self.period != 'all' and timeline:
            cut = int(len(timeline) * self.dev_frac)
            timeline = timeline[:cut] if self.period == 'dev' else timeline[cut:]
        return timeline

    def _print_header(self):
        print("Backtest config:")
        print(f"   Timeframe:     {self.timeframe}")
        print(f"   Capital:       ${self.initial_equity:,.2f}")
        print(f"   Fees:          spot {self.spot_fee:.3%} / futures {self.futures_fee:.3%} per side")
        print(f"   Slippage:      {self.slippage:.3%} per fill")
        print(f"   Max positions: {self.max_positions}   Circuit breaker: {self.max_drawdown:.0%} from peak, "
              f"reset < {self.max_drawdown * self.breaker_reset_ratio:.1%} or after {self.breaker_cooldown_hours:.0f}h")
        print(f"   Exit mode:     {'signal exit_strategy honored' if self.honor_exit_strategy else 'live behaviour (trailing for all)'}")
        filt = []
        if self.side != 'both':
            filt.append(f"side={self.side}")
        if self.include:
            filt.append(f"only {self.include}")
        if self.exclude:
            filt.append(f"excluding {self.exclude}")
        print(f"   Strategy:      {self.cfg.get('strategy_mode') or 'legacy'}")
        print(f"   Filters:       {', '.join(filt) if filt else 'none'}")
        if self.btc_gate_days:
            print(f"   BTC gate:      longs only while BTC close > its {self.btc_gate_days}-day EMA"
                  + (", shorts only while below it" if self.gate_shorts else ""))

    # ------------------------------------------------------------------
    def run(self) -> List[dict]:
        self._print_header()
        print(f"\nLoading data: {len(self.coins)} coins...", flush=True)
        data = self._load_data()
        if not data:
            return []
        signals = self._get_signals(data)

        ts_index: Dict[str, Dict[pd.Timestamp, int]] = {
            c: {ts: i for i, ts in enumerate(df['timestamp'])} for c, df in data.items()}
        self._data_lengths = {s: len(d) for s, d in data.items()}
        all_ts = sorted(set().union(*[set(m.keys()) for m in ts_index.values()]))
        timeline = self._select_timeline(all_ts)
        if not timeline:
            print("No bars in the selected window.")
            return []
        self.window = (timeline[0], timeline[-1])
        self._gate_up = self._build_btc_gate(data, all_ts) if self.btc_gate_days else None
        label = {'all': 'full window', 'dev': f'DEV first {self.dev_frac:.0%}',
                 'holdout': f'HOLDOUT last {1 - self.dev_frac:.0%}'}[self.period]
        print(f"Window: {timeline[0]:%Y-%m-%d} -> {timeline[-1]:%Y-%m-%d} ({label}, {len(timeline)} bars)")

        symbols = [c for c in self.coins if c in data]
        equity = self.initial_equity      # realized equity
        open_pos: Dict[str, dict] = {}
        pending: Dict[str, dict] = {}     # signals waiting for next bar open
        last_close: Dict[str, float] = {}
        last_idx: Dict[str, int] = {}
        breaker = False
        breaker_since = None
        peak = self.initial_equity
        self.breaker_trips, self.breaker_bars = [], 0
        started = time.time()

        for n, t in enumerate(timeline):
            if n % 500 == 0:
                print(f"  bar {n}/{len(timeline)} ({t}) | trades: {len(self.trades)} "
                      f"| open: {len(open_pos)} | {time.time() - started:.0f}s", flush=True)

            # ---- 1. fill pending entries at this bar's open --------------
            for sym in list(pending.keys()):
                idx = ts_index[sym].get(t)
                if idx is None or idx <= pending[sym]['signal_idx']:
                    continue
                p = pending.pop(sym)
                pos = self._open_position(sym, p, data[sym].iloc[idx], idx)
                notional = pos['units'] * pos['entry_price']
                reserved = sum(q['units'] * q['entry_price'] for q in open_pos.values())
                if notional > equity - reserved:
                    if self.verbose:
                        print(f"  [SKIP] {sym}: not enough free cash at fill")
                    continue
                open_pos[sym] = pos
                if self.verbose:
                    print(f"  [ENTRY] {t} {sym} {pos['side'].upper()} @ {pos['entry_price']:.4f} ({pos['regime_label']})")

            # ---- 2. manage exits on this bar -----------------------------
            for sym in list(open_pos.keys()):
                idx = ts_index[sym].get(t)
                if idx is None:
                    continue
                pos = open_pos[sym]
                bar = data[sym].iloc[idx]
                exit_info = self._check_exit(sym, pos, data[sym], idx, bar)
                if exit_info:
                    trade = self._close_position(pos, *exit_info, exit_time=t, idx=idx)
                    equity += trade['net_pnl']
                    self.trades.append(trade)
                    del open_pos[sym]
                    if self.verbose:
                        print(f"  [EXIT]  {t} {sym} @ {trade['exit_price']:.4f} "
                              f"PnL ${trade['net_pnl']:+.2f} ({trade['exit_reason']})")

            # ---- 3. mark to market + circuit breaker ---------------------
            for sym in symbols:
                idx = ts_index[sym].get(t)
                if idx is not None:
                    last_close[sym] = float(data[sym].iloc[idx]['close'])
                    last_idx[sym] = idx
            unrealized = sum(self._unrealized(p, last_close.get(s, p['entry_price']))
                             for s, p in open_pos.items())
            mtm = equity + unrealized
            self.equity_curve.append((t, mtm))

            # Same rules as TradingEngine.check_drawdown(): drawdown from PEAK equity;
            # resets on recovery (< max * reset_ratio) or after the cooldown, which rebases the peak.
            peak = max(peak, mtm)
            drawdown = (peak - mtm) / peak if peak > 0 else 0.0
            if not breaker:
                if drawdown > self.max_drawdown:
                    breaker, breaker_since = True, t
                    self.breaker_trips.append(t)
            else:
                if drawdown < self.max_drawdown * self.breaker_reset_ratio:
                    breaker = False
                elif (t - breaker_since) >= pd.Timedelta(hours=self.breaker_cooldown_hours):
                    breaker = False
                    peak = mtm

            # ---- 4. take precomputed signals (entry on next bar open) ----
            if breaker:
                self.breaker_bars += 1
                continue
            for sym in symbols:
                if len(open_pos) + len(pending) >= self.max_positions:
                    break
                if sym in open_pos or sym in pending:
                    continue
                idx = ts_index[sym].get(t)
                if idx is None or idx + 1 >= len(data[sym]):
                    continue
                sig = signals.get(sym, {}).get(idx)
                if sig is None or not self._accept(sig):
                    continue
                if self._gate_up is not None:
                    btc_up = self._gate_up.get(t, False)
                    if sig['side'] == 'long' and not btc_up:
                        continue                  # BTC downtrend: no new longs
                    if sig['side'] == 'short' and self.gate_shorts and btc_up:
                        continue                  # BTC uptrend: no new shorts

                reserved = (sum(q['units'] * q['entry_price'] for q in open_pos.values())
                            + sum(q['units'] * q['entry_price'] for q in pending.values()))
                free_cash = equity - reserved
                if free_cash <= 10:
                    break
                # size scales linearly with equity: rescale from the nominal precompute equity
                units = sig['units'] * free_cash / PRECOMPUTE_EQUITY
                if units * sig['entry_price'] < MIN_ORDER_USD:
                    continue
                sig = dict(sig)
                sig['units'] = units
                sig['signal_idx'] = idx
                pending[sym] = sig

        # ---- close anything still open at the last simulated bar ---------
        for sym, pos in list(open_pos.items()):
            idx = last_idx.get(sym, pos['entry_idx'])
            bar = data[sym].iloc[idx]
            trade = self._close_position(pos, float(bar['close']), 'end_of_data',
                                         exit_time=bar['timestamp'], idx=idx)
            equity += trade['net_pnl']
            self.trades.append(trade)

        self.benchmark = self._buy_and_hold(data, timeline[0], timeline[-1])
        return self.trades

    # ------------------------------------------------------------------
    def _open_position(self, symbol: str, sig: dict, bar: pd.Series, idx: int) -> dict:
        side = sig['side']
        raw_open = float(bar['open'])
        # adverse slippage on entry
        entry = raw_open * (1 + self.slippage) if side == 'long' else raw_open * (1 - self.slippage)

        # Re-anchor SL/TP to the actual fill, keeping the signal's percentage distances
        sig_price = float(sig['entry_price'])
        sl_pct = abs(sig_price - sig['stop_loss']) / sig_price
        tp_pct = abs(sig['take_profit'] - sig_price) / sig_price if sig.get('take_profit') else 0.0
        if side == 'long':
            stop, target = entry * (1 - sl_pct), entry * (1 + tp_pct)
        else:
            stop, target = entry * (1 + sl_pct), entry * (1 - tp_pct)
        if not sig.get('take_profit'):
            target = 0.0                                   # no take-profit
        if self.fixed_tp:                                  # hardcoded take-profit: +X% from the fill
            target = entry * (1 + self.fixed_tp) if side == 'long' else entry * (1 - self.fixed_tp)

        pair_cfg = get_pair_config(symbol)
        pos = {
            'symbol':        symbol,
            'side':          side,
            'entry_price':   entry,
            'entry_time':    bar['timestamp'],
            'entry_idx':     idx,
            'stop_loss':     stop,
            'take_profit':   target,
            'units':         float(sig['units']),
            'atr':           sig.get('atr', 0.0),
            'signal_type':   sig.get('signal_type', 'backtest_trade'),
            'regime_label':  sig.get('regime_label', 'unknown'),
            'in_sample':     idx >= self._data_lengths[symbol] - REGIME_TRAIN_CANDLES,
            'candles_held':  0,
            'last_candle_time': None,
            'trailing_min_pct': (self.trail_override[0] if self.trail_override
                                 else pair_cfg.get('trailing_min_pct', self.trail_min)),
            'trailing_max_pct': (self.trail_override[1] if self.trail_override
                                 else pair_cfg.get('trailing_max_pct', self.trail_max)),
            'trailing_activation_pct': self.trail_activation,
        }
        # NOTE: the live open_position() stores neither 'regime' nor 'exit_strategy',
        # so live positions always use the trailing/chandelier path. Mirror that unless asked.
        if self.honor_exit_strategy and sig.get('exit_strategy'):
            pos['exit_strategy'] = sig['exit_strategy']
        return pos

    def _check_exit(self, symbol: str, pos: dict, df: pd.DataFrame, idx: int, bar: pd.Series):
        """Return (exit_price, reason) or None."""
        high, low = float(bar['high']), float(bar['low'])
        open_ = float(bar['open'])
        stop, tp = pos['stop_loss'], pos['take_profit']

        # Intrabar stop / take-profit using the stop as of the previous bar
        if pos['side'] == 'long':
            if stop and low <= stop:
                fill = min(stop, open_)                       # gap through the stop fills at the open
                return fill, ('trailing_stop' if pos.get('trailing_stop_active') else 'stop_loss')
            if tp and high >= tp:
                return max(tp, open_), 'take_profit'
        else:
            if stop and high >= stop:
                fill = max(stop, open_)
                return fill, ('trailing_stop' if pos.get('trailing_stop_active') else 'stop_loss')
            if tp and low <= tp:
                return min(tp, open_), 'take_profit'

        # Close-based logic: trailing update, indicator exits
        slow = str(pos.get('signal_type', '')).startswith('trend_hold')
        n = max(EXIT_WINDOW, _trend_hold.history_needed(self.cfg)) if slow else EXIT_WINDOW
        window = df.iloc[max(0, idx - n + 1): idx + 1].copy()
        should_exit, reason = evaluate_exit(symbol, pos, float(bar['close']), window)
        if should_exit:
            return float(bar['close']), reason

        if not slow and pos['candles_held'] > self.max_candles:   # trend_hold holds until stopped
            return float(bar['close']), 'timeout'
        return None

    def _close_position(self, pos: dict, raw_exit: float, reason: str, exit_time, idx: int) -> dict:
        side = pos['side']
        # adverse slippage on exit
        exit_price = raw_exit * (1 - self.slippage) if side == 'long' else raw_exit * (1 + self.slippage)
        units = pos['units']
        entry = pos['entry_price']

        gross = (exit_price - entry) * units if side == 'long' else (entry - exit_price) * units
        fee_rate = self.spot_fee if side == 'long' else self.futures_fee
        fees = (entry + exit_price) * units * fee_rate
        net = gross - fees

        return {
            **pos,
            'exit_time':   exit_time,
            'exit_price':  exit_price,
            'exit_reason': reason,
            'net_pnl':     net,
            'fees':        fees,
            'pnl_pct':     net / (entry * units) * 100 if units > 0 else 0,
            'candles_held': pos['candles_held'],
        }

    @staticmethod
    def _unrealized(pos: dict, price: float) -> float:
        if pos['side'] == 'long':
            return (price - pos['entry_price']) * pos['units']
        return (pos['entry_price'] - price) * pos['units']

    # ------------------------------------------------------------------
    # Benchmark: what buying and holding the same coins would have done
    # ------------------------------------------------------------------
    def _buy_and_hold(self, data: Dict[str, pd.DataFrame], t0, t1) -> Optional[dict]:
        series = {}
        for sym, df in data.items():
            w = df[(df['timestamp'] >= t0) & (df['timestamp'] <= t1)]
            if len(w) >= 2:
                s = w.set_index('timestamp')['close']
                series[sym] = s / s.iloc[0]
        if not series:
            return None
        panel = pd.concat(series, axis=1).sort_index().ffill().bfill()
        ew = panel.mean(axis=1)                    # equal-weight, bought at the start, never rebalanced
        out = {'equal_weight_ret': (ew.iloc[-1] - 1) * 100,
               'equal_weight_mdd': ((ew - ew.cummax()) / ew.cummax()).min() * 100,
               'coins': len(series)}
        for btc in ('BTC/USDC', 'BTC/USDT'):
            if btc in series:
                b = series[btc]
                out['btc_ret'] = (b.iloc[-1] - 1) * 100
                out['btc_mdd'] = ((b - b.cummax()) / b.cummax()).min() * 100
                break
        return out

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    @staticmethod
    def _group_table(df: pd.DataFrame, key: str) -> pd.DataFrame:
        g = df.groupby(key)['net_pnl']
        return pd.DataFrame({
            'Trades':    g.count(),
            'Total PnL': g.sum().round(2),
            'Avg PnL':   g.mean().round(2),
            'Win %':     g.apply(lambda s: (s > 0).mean() * 100).round(1),
        })

    @staticmethod
    def _stats(df: pd.DataFrame) -> str:
        wins = df[df['net_pnl'] > 0]['net_pnl'].sum()
        losses = abs(df[df['net_pnl'] <= 0]['net_pnl'].sum())
        pf = wins / losses if losses else float('inf')
        return (f"{len(df)} trades | win {(df['net_pnl'] > 0).mean() * 100:.1f}% | "
                f"PF {pf:.2f} | PnL ${df['net_pnl'].sum():,.2f}")

    def metrics(self, trades: List[dict]) -> dict:
        df = pd.DataFrame(trades)
        total_pnl = df['net_pnl'].sum()
        gross_profit = df[df['net_pnl'] > 0]['net_pnl'].sum()
        gross_loss = abs(df[df['net_pnl'] <= 0]['net_pnl'].sum())
        eq = pd.Series([e for _, e in self.equity_curve])
        oos = df[~df['in_sample']]['net_pnl']
        return {
            'trades': len(df), 'net_pnl': total_pnl, 'roi_pct': total_pnl / self.initial_equity * 100,
            'fees': df['fees'].sum(), 'win_rate': (df['net_pnl'] > 0).mean() * 100,
            'profit_factor': gross_profit / gross_loss if gross_loss else float('inf'),
            'max_dd_pct': ((eq - eq.cummax()) / eq.cummax() * 100).min() if len(eq) else 0.0,
            'oos_pnl': oos.sum(), 'oos_trades': len(oos),
        }

    def _summary(self, trades: List[dict]):
        if not trades:
            print("\nNo trades executed. Check signal logic, filters or data range.")
            return

        df = pd.DataFrame(trades)
        df['net_pnl'] = pd.to_numeric(df['net_pnl'])
        m = self.metrics(trades)
        final_bal = self.initial_equity + m['net_pnl']

        print(f"\n{'=' * 55}")
        print("FINANCIAL PERFORMANCE SUMMARY")
        print(f"  window {self.window[0]:%Y-%m-%d} -> {self.window[1]:%Y-%m-%d}  ({self.period})")
        print(f"{'=' * 55}")
        rows = [
            ["Starting Capital", f"${self.initial_equity:,.2f}"],
            ["Final Balance",    f"${final_bal:,.2f}"],
            ["Total Net PnL",    f"${m['net_pnl']:,.2f} ({m['roi_pct']:+.2f}%)"],
            ["Gross PnL (pre-fee)", f"${m['net_pnl'] + m['fees']:,.2f}"],
            ["Total Fees",       f"${m['fees']:,.2f}"],
            ["Total Trades",     m['trades']],
            ["Win Rate",         f"{m['win_rate']:.1f}%"],
            ["Profit Factor",    f"{m['profit_factor']:.2f}"],
            ["Avg. PnL / Trade", f"${df['net_pnl'].mean():.2f}"],
            ["Hold time (days)", f"avg {df['candles_held'].mean() / self.BARS_PER_DAY.get(self.timeframe, 6):.1f} | "
                                 f"median {df['candles_held'].median() / self.BARS_PER_DAY.get(self.timeframe, 6):.1f} | "
                                 f"max {df['candles_held'].max() / self.BARS_PER_DAY.get(self.timeframe, 6):.0f}"],
            ["Max Drawdown (MTM)", f"{m['max_dd_pct']:.2f}%"],
            ["Breaker trips",    f"{len(self.breaker_trips)} ({self.breaker_bars} bars with entries paused)"],
        ]
        for label, value in rows:
            print(f"{label:<20} : {value}")

        if self.benchmark:
            b = self.benchmark
            print(f"\n--- Benchmark: buy & hold over the same window ---")
            print(f"Equal-weight {b['coins']} coins : {b['equal_weight_ret']:+.1f}%  (max DD {b['equal_weight_mdd']:.1f}%)")
            if 'btc_ret' in b:
                print(f"BTC alone            : {b['btc_ret']:+.1f}%  (max DD {b['btc_mdd']:.1f}%)")
            print(f"This strategy        : {m['roi_pct']:+.1f}%  (max DD {m['max_dd_pct']:.1f}%)")

        print("\n--- Regime model in-sample check ---")
        print("(the regime model is trained on the last "
              f"{REGIME_TRAIN_CANDLES} candles per coin; trust the out-of-sample line)")
        for flag, label in ((False, 'Out-of-sample'), (True, 'In-sample (optimistic)')):
            sub = df[df['in_sample'] == flag]
            print(f"{label:<24}: {self._stats(sub) if len(sub) else 'no trades'}")

        print("\n--- By month of entry (is the result stable?) ---")
        df['month'] = pd.to_datetime(df['entry_time']).dt.strftime('%Y-%m')
        print(self._group_table(df, 'month'))

        print("\n--- Side ---")
        print(self._group_table(df, 'side'))

        print("\n--- Performance by Market Regime ---")
        print(self._group_table(df, 'regime_label'))

        print("\n--- Exit Breakdown ---")
        for reason, count in df['exit_reason'].value_counts().items():
            print(f"{reason:<20} : {count} trades")

        print("\n--- Signal Type Breakdown ---")
        print(self._group_table(df, 'signal_type'))

        print(f"{'=' * 55}\n")
        self._record(df, m)

    def _record(self, df: pd.DataFrame, m: dict):
        """Append this run to backtest_results/experiments.csv (and save trades if tagged)."""
        if not self.log_experiment:
            return
        os.makedirs(self.results_dir, exist_ok=True)
        path = os.path.join(self.results_dir, 'experiments.csv')
        row = {
            'when': datetime.now().strftime('%Y-%m-%d %H:%M'), 'tag': self.tag or '',
            'timeframe': self.timeframe, 'period': self.period,
            'window': f"{self.window[0]:%Y-%m-%d}..{self.window[1]:%Y-%m-%d}",
            'strategy': self.cfg.get('strategy_mode') or 'legacy', 'fixed_tp': self.fixed_tp, 'btc_gate_days': self.btc_gate_days, 'gate_shorts': self.gate_shorts, 'side': self.side, 'include': ' '.join(self.include), 'exclude': ' '.join(self.exclude),
            'trades': m['trades'], 'net_pnl': round(m['net_pnl'], 2), 'roi_pct': round(m['roi_pct'], 2),
            'pf': round(m['profit_factor'], 3), 'win_pct': round(m['win_rate'], 1),
            'max_dd_pct': round(m['max_dd_pct'], 2), 'fees': round(m['fees'], 2),
            'oos_trades': m['oos_trades'], 'oos_pnl': round(m['oos_pnl'], 2),
            'bh_equal_weight_pct': round(self.benchmark['equal_weight_ret'], 1) if self.benchmark else '',
            'code': code_fingerprint(self.cfg),
        }
        new = not os.path.exists(path)
        with open(path, 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if new:
                w.writeheader()
            w.writerow(row)
        n = sum(1 for _ in open(path)) - 1
        print(f"Logged to {path}  ({n} experiments recorded so far - every variant you try "
              f"lowers how much any single good result proves)")
        if self.tag:
            df.drop(columns=['month'], errors='ignore').to_csv(
                os.path.join(self.results_dir, f'{self.tag}_trades.csv'), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--coins', nargs='+', default=None)
    parser.add_argument('--days', type=int, default=365, help='simulated window length (ending at the newest candle)')
    parser.add_argument('--start', default=None, help='window start YYYY-MM-DD (overrides --days)')
    parser.add_argument('--end', default=None, help='window end YYYY-MM-DD')
    parser.add_argument('--capital', type=float, default=5000.0, help='starting capital (portfolio.json is NOT used)')
    parser.add_argument('--slippage', type=float, default=0.0005, help='adverse slippage per fill, e.g. 0.0005 = 0.05%%')
    parser.add_argument('--fixed-exits', action='store_true',
                        help="honor the signal's exit_strategy='fixed' (live bot currently ignores it)")
    parser.add_argument('--side', choices=['both', 'long', 'short'], default=None,
                        help="only take this side (default: long unless enable_shorts is true in config.json)")
    parser.add_argument('--signals', nargs='+', default=None, metavar='TEXT',
                        help='only signal types containing any of these (e.g. trend breakout)')
    parser.add_argument('--exclude', nargs='+', default=None, metavar='TEXT',
                        help='skip signal types containing any of these (e.g. range bottom_fib)')
    parser.add_argument('--period', choices=['all', 'dev', 'holdout'], default='all',
                        help='dev = first --dev-frac of the window (tune here), holdout = the rest (judge here)')
    parser.add_argument('--dev-frac', type=float, default=0.6)
    parser.add_argument('--btc-gate', type=int, default=0, metavar='DAYS',
                        help='only take LONGS while BTC close > its DAYS-day EMA (0 = off)')
    parser.add_argument('--gate-shorts', action='store_true',
                        help='with --btc-gate: also only take SHORTS while BTC is below its EMA')
    parser.add_argument('--strategy', choices=['legacy', 'trend_hold'], default=None,
                        help="entry/exit logic: 'legacy' (regime + indicators) or 'trend_hold' "
                             "(default: strategy_mode from config.json)")
    parser.add_argument('--th-entry-days', type=int, default=None,
                        help='trend_hold: breakout lookback in days (default 20; classic slow variant: 55)')
    parser.add_argument('--th-exit-days', type=int, default=None,
                        help='trend_hold: exit-channel lookback in days (default 10; classic slow variant: 20)')
    parser.add_argument('--fixed-tp', type=float, default=0.0, metavar='FRACTION',
                        help='hardcoded take-profit from the fill, e.g. 0.02 = +2% (stops unchanged)')
    parser.add_argument('--trail-min', type=float, default=None, metavar='FRACTION',
                        help='legacy trailing stop: minimum distance below the peak, for ALL coins (e.g. 0.10)')
    parser.add_argument('--trail-max', type=float, default=None, metavar='FRACTION',
                        help='legacy trailing stop: maximum distance below the peak, for ALL coins (e.g. 0.15)')
    parser.add_argument('--th-stage2-gain', type=float, default=None, metavar='FRACTION',
                        help='trend_hold two-stage exit: once the trade has gained this much (e.g. 0.30), widen the exit')
    parser.add_argument('--th-stage2-days', type=int, default=None,
                        help='trend_hold two-stage exit: channel length in days after the gain is reached (default 40)')
    parser.add_argument('--refresh', action='store_true', help='re-download the price snapshot')
    parser.add_argument('--fetch-days', type=int, default=730, help='history to download for the snapshot')
    parser.add_argument('--workers', type=int, default=None, help='processes for signal precompute (1 = in-process)')
    parser.add_argument('--no-cache', action='store_true', help='ignore the cached signals')
    parser.add_argument('--model-dir', default=None,
                        help='use regime model files from this folder (e.g. backups/regime_models_...)')
    parser.add_argument('--tag', default=None, help='name for this experiment (also saves its trades to CSV)')
    parser.add_argument('--no-log', action='store_true', help="don't record this run in experiments.csv")
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    if args.strategy:
        apply_strategy(args.strategy)
    apply_trend_hold_params(args.th_entry_days, args.th_exit_days, args.th_stage2_gain, args.th_stage2_days)
    if args.model_dir:
        apply_model_dir(args.model_dir)
        print(f"Using regime model from {os.environ['BACKTEST_MODEL_DIR']}")

    bt = Backtester(coins=args.coins or [], days=args.days, capital=args.capital,
                    slippage=args.slippage, honor_exit_strategy=args.fixed_exits,
                    verbose=args.verbose, side=args.side, include=args.signals,
                    exclude=args.exclude, period=args.period, dev_frac=args.dev_frac,
                    start=args.start, end=args.end, workers=args.workers, refresh=args.refresh,
                    fetch_days=args.fetch_days, tag=args.tag, use_cache=not args.no_cache,
                    log_experiment=not args.no_log, btc_gate_days=args.btc_gate,
                    gate_shorts=args.gate_shorts, fixed_tp=args.fixed_tp,
                    trail_min=args.trail_min, trail_max=args.trail_max)
    bt.coins = args.coins if args.coins else bt.default_coins

    all_trades = bt.run()
    bt._summary(all_trades)
