"""
backtest.py
===========
Backtests the bot's actual strategy stack (regime detection + signal generation
+ exit manager) against historical OHLCV data from Binance.

Uses the SAME code as production:
  - regime_switcher.predict_regime()
  - regime_switcher.detect_trend()
  - regime_switcher.confirm_trend_with_higher_tf()
  - strategy_tools.generate_trade_signal()
  - exit_manager.evaluate_exit()

Run from project root:
    python backtest.py                        # all coins, 1 year
    python backtest.py --coins BTC/USDT       # single coin
    python backtest.py --days 90              # 90 days
    python backtest.py --no-regime            # skip regime filter (faster)
    python --no-trend                         # skip trend filter
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import time
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import pandas as pd
import numpy as np

from config_loader import get_pair_config

# -------------------------------------------------------------------
# Setup paths
# -------------------------------------------------------------------

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------
try:
    from config_loader import config
    CONFIG = config.config
    print("\n📌 Per‑pair overrides loaded:")
    for sym in CONFIG.get('coins', []):
        cfg = get_pair_config(sym)
        if cfg:
            risk = cfg.get('risk_per_trade')
            trail_min = cfg.get('trailing_min_pct')
            trail_max = cfg.get('trailing_max_pct')
            print(f"   {sym}: risk={risk}, trail={trail_min:.0%}‑{trail_max:.0%}" if trail_min and trail_max else f"   {sym}: risk={risk}")
        else:
            print(f"   {sym}: (none)")
except Exception:
    CONFIG = {
        'coins': ['BTC/USDT', 'ETH/USDT', 'SOL/USDT'],
        'risk_per_trade': 0.03,
        'trading_fee': 0.0005,
        'max_positions': 5,
        'trading_timeframe': config.get('trading_timeframe', '1h'),
    }

TIMEFRAME      = CONFIG.get('trading_timeframe', '1h')
RISK_PER_TRADE = float(CONFIG.get('risk_per_trade', 0.03))
FEE            = float(CONFIG.get('trading_fee', 0.0005))
MAX_POSITIONS  = int(CONFIG.get('max_positions', 5))
INITIAL_EQUITY = int(CONFIG.get('cash', 5000))

# -------------------------------------------------------------------
# Import strategy stack (same as production)
# -------------------------------------------------------------------
from modules.strategy_tools import generate_trade_signal, calculate_position_units
from modules.exit_manager import evaluate_exit, _calculate_atr

try:
    from modules.regime_switcher import predict_regime, train_model, model as regime_model
    from modules.regime_switcher import detect_trend, confirm_trend_with_higher_tf
    HAS_REGIME = True
    HAS_TREND = True
except ImportError:
    HAS_REGIME = False
    HAS_TREND = False
    print("⚠️  Regime switcher not available — running without regime filter")

try:
    from modules.data_feed import fetch_historical_data
    HAS_DATA = True
except ImportError:
    print("❌ data_feed not available")
    sys.exit(1)

# -------------------------------------------------------------------
# Backtest engine
# -------------------------------------------------------------------
class Backtester:
    def __init__(self, coins: List[str], days: int = 365,
                 use_regime: bool = True, use_trend: bool = True,
                 verbose: bool = False):
        self.coins       = coins
        self.days        = days
        self.use_regime  = use_regime and HAS_REGIME
        self.use_trend   = use_trend and HAS_TREND
        self.verbose     = verbose
        self.equity      = INITIAL_EQUITY
        self.trades      = []
        self.equity_curve = [INITIAL_EQUITY]

    # ------------------------------------------------------------------
    def _log(self, msg):
        if self.verbose:
            print(msg)

    # ------------------------------------------------------------------
    def _fetch(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetch full OHLCV once — all simulation uses this, zero further API calls."""
        try:
            df = fetch_historical_data(symbol, interval=TIMEFRAME, days=self.days)
            if df is None or df.empty or len(df) < 100:
                return None
            df = df.sort_values('timestamp').reset_index(drop=True)
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            return df
        except Exception as e:
            self._log(f"  ⚠️  Fetch failed for {symbol}: {e}")
            return None

    # ------------------------------------------------------------------
    def _simulate_position(self, symbol: str, signal: dict,
                       df: pd.DataFrame, entry_idx: int) -> dict:
        """Walk forward using pre-loaded df only — zero API calls."""
        side        = signal['side']
        entry_price = signal['entry']
        units       = signal['units']
        atr         = signal.get('atr', 0.0)
        max_candles = 168  # 7 days on 1h

        position = {
            'side':                 side,
            'entry_price':          entry_price,
            'stop_loss':            signal['stop_loss'],
            'take_profit':          signal['take_profit'],
            'signal_type':          signal.get('signal_type', 'unknown'),
            'atr':                  atr,
            'trailing_stop_active': False,
            'trailing_min_pct':     signal.get('trailing_min_pct'),
            'trailing_max_pct':     signal.get('trailing_max_pct'),
        }

        exit_price   = None
        exit_reason  = 'max_hold'
        candles_held = 0
        end_idx      = min(entry_idx + max_candles + 1, len(df))

        for i in range(entry_idx + 1, end_idx):
            current_price = float(df.iloc[i]['close'])
            candles_held  = i - entry_idx

            position['candles_held'] = candles_held

            w_start = max(0, i - 49)
            window  = df.iloc[w_start: i + 1]

            should_exit, reason = evaluate_exit(
                symbol, position, current_price, window
            )

            if should_exit:
                exit_price  = current_price
                exit_reason = reason
                break

        if exit_price is None:
            last_idx     = min(entry_idx + max_candles, len(df) - 1)
            exit_price   = float(df.iloc[last_idx]['close'])
            candles_held = last_idx - entry_idx

        gross_pnl = ((exit_price - entry_price) if side == 'long'
                    else (entry_price - exit_price)) * units
        entry_fee = entry_price * units * FEE
        exit_fee  = exit_price  * units * FEE
        net_pnl   = gross_pnl - entry_fee - exit_fee
        pnl_pct   = net_pnl / max(entry_price * units, 1e-9) * 100

        return {
            'symbol':       symbol,
            'side':         side,
            'entry_price':  entry_price,
            'exit_price':   exit_price,
            'units':        units,
            'gross_pnl':    round(gross_pnl, 6),
            'net_pnl':      round(net_pnl, 6),
            'pnl_pct':      round(pnl_pct, 4),
            'fees':         round(entry_fee + exit_fee, 6),
            'candles_held': candles_held,
            'hours_held':   candles_held if TIMEFRAME == '4h' else round(candles_held / 4, 1),
            'exit_reason':  exit_reason,
            'signal_type':  signal.get('signal_type', 'unknown'),
            'regime':       signal.get('regime', 'unknown'),
            'trend_strength': signal.get('trend_strength', 0),
        }
    # ------------------------------------------------------------------
    def run_coin(self, symbol: str) -> List[dict]:
        """Run backtest for one coin with trend filtering."""
        self._log(f"\n  📊 {symbol}")
        df = self._fetch(symbol)
        if df is None:
            self._log(f"  ⏭️  No data")
            return []

        # ===== PER‑PAIR CONFIGURATION =====
        pair_cfg = get_pair_config(symbol)
        pair_risk = pair_cfg.get('risk_per_trade')
        risk_used = pair_risk if pair_risk is not None else RISK_PER_TRADE
        trailing_min = pair_cfg.get('trailing_min_pct')
        trailing_max = pair_cfg.get('trailing_max_pct')
        self._log(f"  Risk: {risk_used:.1%} (global: {RISK_PER_TRADE:.1%})")
        if trailing_min and trailing_max:
            self._log(f"  Trailing: {trailing_min:.1%}–{trailing_max:.1%}")

        coin_trades   = []
        warmup        = 100
        i             = warmup

        SCAN_STEP     = 2
        MIN_TRADE_GAP = 24

        max_dd_pct    = float(CONFIG.get('max_drawdown', 0.06))
        peak_equity   = self.equity

        # Cache higher timeframe data if needed (optional)
        df_4h = None
        if self.use_trend:
            try:
                df_4h = fetch_historical_data(symbol, interval='4h', days=self.days)
            except:
                pass

        while i < len(df) - 2:
            # Drawdown check
            if self.equity > peak_equity:
                peak_equity = self.equity
            current_dd = (self.equity - peak_equity) / peak_equity
            if current_dd < -max_dd_pct:
                self._log(f"  🛑 Drawdown circuit breaker hit: {current_dd:.1%} — stopping {symbol}")
                break

            w_start = max(0, i - 99)
            window  = df.iloc[w_start: i + 1].copy()

            if len(window) < 50:
                i += SCAN_STEP
                continue

            # ===== REGIME DETECTION =====
            regime = 'unknown'
            if self.use_regime:
                try:
                    regime = predict_regime(window)
                except Exception:
                    regime = 'unknown'

            # ===== TREND FILTERING =====
            skip_trade = False
            trend_strength = 0
            trend_direction = "side"

            if self.use_trend:
                try:
                    direction, strength, conf = detect_trend(window)
                    trend_strength = strength
                    trend_direction = direction

                    # Skip if trend is too weak (ADX < 20)
                    if strength < 0.2:
                        self._log(f"  ⏭️  {symbol}: trend too weak (ADX={strength*100:.0f})")
                        skip_trade = True

                    # Optional: skip if higher timeframe doesn't align
                    # if not skip_trade and df_4h is not None and not df_4h.empty:
                    #     if not confirm_trend_with_higher_tf(symbol, window):
                    #         self._log(f"  ⏭️  {symbol}: 4h trend doesn't align with 1h")
                    #         skip_trade = True

                except Exception as e:
                    self._log(f"  ⚠️  Trend detection error: {e}")

            if skip_trade:
                i += SCAN_STEP
                continue

            # ===== SIGNAL GENERATION =====
            try:
                signal = generate_trade_signal(
                    df=window,
                    equity=self.equity,
                    risk_per_trade=risk_used,          # per‑pair risk
                    symbol=symbol,
                    trading_engine=None,
                    regime=regime,
                )
            except Exception:
                i += SCAN_STEP
                continue

            if not signal:
                i += SCAN_STEP
                continue

            # Add trailing bounds to signal (will be passed to position)
            signal['trailing_min_pct'] = trailing_min
            signal['trailing_max_pct'] = trailing_max

            cost = signal['units'] * signal['entry']
            if cost > self.equity * 0.95 or cost < 1.0:
                i += SCAN_STEP
                continue

            # ===== SIMULATE POSITION =====
            result = self._simulate_position(symbol, signal, df, i)
            self.equity += result['net_pnl']
            self.equity_curve.append(self.equity)
            coin_trades.append(result)

            # Jump past trade + minimum gap
            i += max(result['candles_held'], 1) + MIN_TRADE_GAP

            self._log(
                f"  {'✅' if result['net_pnl'] > 0 else '❌'} "
                f"{result['side'].upper()} @ ${result['entry_price']:.4f} → "
                f"${result['exit_price']:.4f} | "
                f"PnL: ${result['net_pnl']:+.4f} | "
                f"{result['exit_reason']} | {result['candles_held']}c"
                f"{f' | trend={trend_direction}/{trend_strength:.2f}' if self.use_trend else ''}"
            )

        return coin_trades

    # ------------------------------------------------------------------
    def run(self) -> dict:
        """Run full backtest across all coins."""
        print(f"\n{'='*60}")
        print(f"  🔁 Backtest — {len(self.coins)} coins | {self.days} days | {TIMEFRAME}")
        print(f"  Initial equity: ${INITIAL_EQUITY:.2f}")
        print(f"  Risk/trade: {RISK_PER_TRADE:.1%} | Fee: {FEE:.3%} | Max pos: {MAX_POSITIONS}")
        print(f"  Trailing stop: {CONFIG.get('trailing_stop_min_pct', 0.02):.1%} - "
            f"{CONFIG.get('trailing_stop_max_pct', 0.04):.1%}")
        print(f"  Regime filter: {'on' if self.use_regime else 'off'}")
        print(f"  Trend filter:  {'on' if self.use_trend else 'off'}")
        print(f"{'='*60}\n")

        # --- Per‑pair settings ---
        print("📊 Per‑pair configurations (global defaults unless overridden):\n")
        print(f"{'Symbol':<12} {'Risk':>8} {'Trail min':>10} {'Trail max':>10}")
        print("-" * 45)
        
        for symbol in self.coins:
            try:
                from config_loader import get_pair_config
                pair_cfg = get_pair_config(symbol)
                risk = pair_cfg.get('risk_per_trade')
                trail_min = pair_cfg.get('trailing_min_pct')
                trail_max = pair_cfg.get('trailing_max_pct')
                
                risk_str = f"{risk:.1%}" if risk is not None else "global"
                min_str = f"{trail_min:.1%}" if trail_min is not None else "global"
                max_str = f"{trail_max:.1%}" if trail_max is not None else "global"
            except Exception:
                risk_str = min_str = max_str = "global"
            
            print(f"{symbol:<12} {risk_str:>8} {min_str:>10} {max_str:>10}")
        
        print(f"\n{'='*60}\n")


        # Train regime model if needed
        if self.use_regime and HAS_REGIME:
            try:
                import modules.regime_switcher as rs
                if rs.model is None:
                    print("🔄 Training regime model...")
                    train_model()
                    print("✅ Model ready\n")
            except Exception as e:
                print(f"⚠️  Could not train regime model: {e}\n")
                self.use_regime = False

        all_trades = []
        for i, symbol in enumerate(self.coins):
            print(f"[{i+1}/{len(self.coins)}] {symbol}...", end=' ', flush=True)
            t0 = time.time()
            trades = self.run_coin(symbol)
            elapsed = time.time() - t0
            print(f"{len(trades)} trades | {elapsed:.1f}s")
            all_trades.extend(trades)
            self.trades = all_trades

        return self._summary(all_trades)

    # ------------------------------------------------------------------
    def _summary(self, trades: List[dict]) -> dict:
        if not trades:
            return {'error': 'No trades generated'}

        df = pd.DataFrame(trades)

        total_trades  = len(df)
        winners       = df[df['net_pnl'] > 0]
        losers        = df[df['net_pnl'] <= 0]
        win_rate      = len(winners) / total_trades * 100
        total_pnl     = df['net_pnl'].sum()
        total_fees    = df['fees'].sum()
        total_return  = (self.equity - INITIAL_EQUITY) / INITIAL_EQUITY * 100
        avg_win       = winners['net_pnl'].mean() if len(winners) else 0
        avg_loss      = losers['net_pnl'].mean() if len(losers) else 0
        profit_factor = (winners['net_pnl'].sum() / abs(losers['net_pnl'].sum())
                         if len(losers) and losers['net_pnl'].sum() != 0 else float('inf'))
        avg_hold_h    = df['hours_held'].mean()
        max_hold_h    = df['hours_held'].max()

        # Max drawdown
        curve  = np.array(self.equity_curve)
        peak   = np.maximum.accumulate(curve)
        dd     = (curve - peak) / peak * 100
        max_dd = dd.min()

        # Best and worst trades
        best  = df.loc[df['net_pnl'].idxmax()]
        worst = df.loc[df['net_pnl'].idxmin()]

        # Per-coin breakdown
        by_coin = (df.groupby('symbol')
                     .agg(trades=('net_pnl', 'count'),
                          net_pnl=('net_pnl', 'sum'),
                          win_rate=('net_pnl', lambda x: (x > 0).mean() * 100))
                     .sort_values('net_pnl', ascending=False))

        exit_reasons = df['exit_reason'].value_counts().to_dict()
        signal_types = df['signal_type'].value_counts().head(5).to_dict()

        summary = {
            'total_trades':   total_trades,
            'win_rate':       round(win_rate, 1),
            'total_pnl':      round(total_pnl, 4),
            'total_fees':     round(total_fees, 4),
            'total_return':   round(total_return, 2),
            'final_equity':   round(self.equity, 2),
            'profit_factor':  round(profit_factor, 2),
            'max_drawdown':   round(max_dd, 2),
            'avg_win':        round(avg_win, 4),
            'avg_loss':       round(avg_loss, 4),
            'avg_hold_hours': round(avg_hold_h, 1),
            'max_hold_hours': int(max_hold_h),
            'exit_reasons':   exit_reasons,
            'signal_types':   signal_types,
            'by_coin':        by_coin,
            'best_trade':     best,
            'worst_trade':    worst,
        }

        self._print_summary(summary)
        return summary

    # ------------------------------------------------------------------
    def _print_summary(self, s: dict):
        pnl_sign  = '+' if s['total_pnl'] >= 0 else ''
        ret_sign  = '+' if s['total_return'] >= 0 else ''
        dd_color  = '🔴' if s['max_drawdown'] < -10 else '🟡' if s['max_drawdown'] < -5 else '🟢'
        ret_color = '🟢' if s['total_return'] > 0 else '🔴'

        print(f"\n{'='*60}")
        print(f"  📊 BACKTEST RESULTS — {self.days} days")
        print(f"{'='*60}")
        print(f"\n  💰 Performance")
        print(f"     Initial equity:   ${INITIAL_EQUITY:.2f}")
        print(f"     Trailing stop:     {CONFIG.get('trailing_stop_min_pct', 0.02):.1%} - "
              f"{CONFIG.get('trailing_stop_max_pct', 0.04):.1%}")
        print(f"     Risk/trade:       {RISK_PER_TRADE:.1%}")
        print(f"     Final equity:     ${s['final_equity']:.2f}")
        print(f"  {ret_color}  Total return:     {ret_sign}{s['total_return']:.2f}%")
        print(f"     Total PnL:        ${pnl_sign}{s['total_pnl']:.4f}")
        print(f"     Total fees paid:  ${s['total_fees']:.4f}")

        print(f"\n  🎯 Trade Stats")
        print(f"     Total trades:     {s['total_trades']}")
        print(f"     Win rate:         {s['win_rate']:.1f}%")
        print(f"     Profit factor:    {s['profit_factor']:.2f}x")
        print(f"     Avg winner:       ${s['avg_win']:+.4f}")
        print(f"     Avg loser:        ${s['avg_loss']:+.4f}")
        print(f"     Avg hold time:    {s['avg_hold_hours']:.1f}h")
        print(f"     Max hold time:    {s['max_hold_hours']}h")

        print(f"\n  {dd_color} Risk")
        print(f"     Max drawdown:     {s['max_drawdown']:.2f}%")

        print(f"\n  ⚙️ Per‑pair settings (global unless overridden):")
        print(f"  {'Symbol':<12} {'Risk':>8} {'Trail min':>10} {'Trail max':>10}")
        print(f"  {'-'*45}")
        for symbol in self.coins:
            try:
                from config_loader import get_pair_config
                pair_cfg = get_pair_config(symbol)
                risk = pair_cfg.get('risk_per_trade')
                trail_min = pair_cfg.get('trailing_min_pct')
                trail_max = pair_cfg.get('trailing_max_pct')
                risk_str = f"{risk:.1%}" if risk is not None else "global"
                min_str = f"{trail_min:.1%}" if trail_min is not None else "global"
                max_str = f"{trail_max:.1%}" if trail_max is not None else "global"
            except Exception:
                risk_str = min_str = max_str = "global"
            print(f"  {symbol:<12} {risk_str:>8} {min_str:>10} {max_str:>10}")
        print()

        print(f"\n  🚪 Exit reasons")
        for reason, count in sorted(s['exit_reasons'].items(),
                                    key=lambda x: x[1], reverse=True):
            pct = count / s['total_trades'] * 100
            print(f"     {reason:<20} {count:>4} ({pct:.0f}%)")

        print(f"\n  ⚡ Top signal types")
        for sig, count in s['signal_types'].items():
            print(f"     {sig:<30} {count:>4}")

        best  = s['best_trade']
        worst = s['worst_trade']
        print(f"\n  🏆 Best trade:  {best['symbol']} {best['side']} "
              f"${best['net_pnl']:+.4f} ({best['pnl_pct']:+.2f}%) "
              f"via {best['exit_reason']}")
        print(f"  💀 Worst trade: {worst['symbol']} {worst['side']} "
              f"${worst['net_pnl']:+.4f} ({worst['pnl_pct']:+.2f}%) "
              f"via {worst['exit_reason']}")

        print(f"\n  📈 Per-coin breakdown (top 10 by PnL)")
        print(f"  {'Symbol':<14} {'Trades':>6} {'PnL':>10} {'Win%':>8}")
        print(f"  {'-'*42}")
        for symbol, row in s['by_coin'].head(10).iterrows():
            icon = '🟢' if row['net_pnl'] > 0 else '🔴'
            print(f"  {icon} {symbol:<12} {int(row['trades']):>6} "
                  f"  ${row['net_pnl']:>+8.4f}   {row['win_rate']:>5.1f}%")

        print(f"\n{'='*60}\n")


# -------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Backtest trading bot strategies')
    parser.add_argument('--coins',     nargs='+', default=None,
                        help='Coins to test e.g. BTC/USDT ETH/USDT')
    parser.add_argument('--days',      type=int,  default=365,
                        help='Number of days to backtest (default: 365)')
    parser.add_argument('--no-regime', action='store_true',
                        help='Disable regime filter')
    parser.add_argument('--no-trend',  action='store_true',
                        help='Disable trend filter')
    parser.add_argument('--verbose',   action='store_true',
                        help='Print every trade as it happens')
    args = parser.parse_args()

    coins = args.coins or CONFIG.get('coins', ['BTC/USDT', 'ETH/USDT', 'SOL/USDT'])

    bt = Backtester(
        coins=coins,
        days=args.days,
        use_regime=not args.no_regime,
        use_trend=not args.no_trend,
        verbose=args.verbose,
    )
    bt.run()