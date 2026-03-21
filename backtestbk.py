"""
backtest.py
===========
Backtests the bot's actual strategy stack (regime detection + signal generation
+ exit manager) against 1 year of historical 1h OHLCV data from Binance.

Uses the SAME code as production:
  - regime_switcher.predict_regime()
  - strategy_tools.generate_trade_signal()
  - exit_manager.evaluate_exit() (trailing stop + signal reversal)

Run from project root:
    python backtest.py                        # all coins, 1 year
    python backtest.py --coins BTC/USDT       # single coin
    python backtest.py --days 90              # 90 days
    python backtest.py --no-regime            # skip regime filter (faster)
"""

import sys
import os
import argparse
import time
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import pandas as pd
import numpy as np

# -------------------------------------------------------------------
# Setup paths
# -------------------------------------------------------------------
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'modules'))

logging.basicConfig(level=logging.WARNING)  # Suppress noise during backtest
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------
try:
    from config_loader import config
    CONFIG = config.config
except Exception:
    CONFIG = {
        'coins': ['BTC/USDT', 'ETH/USDT', 'SOL/USDT'],
        'risk_per_trade': 0.02,
        'trading_fee': 0.0005,
        'max_positions': 5,
        'trading_timeframe': '1h',
    }

TIMEFRAME      = CONFIG.get('trading_timeframe', '1h')
RISK_PER_TRADE = float(CONFIG.get('risk_per_trade', 0.02))
FEE            = float(CONFIG.get('trading_fee', 0.0005))
MAX_POSITIONS  = int(CONFIG.get('max_positions', 5))
INITIAL_EQUITY = 100.0

# -------------------------------------------------------------------
# Import strategy stack (same as production)
# -------------------------------------------------------------------
try:
    from modules.strategy_tools import generate_trade_signal, calculate_position_units
    from modules.exit_manager import evaluate_exit, _calculate_atr
    HAS_STRATEGY = True
except ImportError as e:
    print(f"❌ Could not import strategy modules: {e}")
    sys.exit(1)

try:
    from modules.regime_switcher import predict_regime, train_model, model as regime_model
    HAS_REGIME = True
except ImportError:
    HAS_REGIME = False
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
                 use_regime: bool = True, verbose: bool = False):
        self.coins       = coins
        self.days        = days
        self.use_regime  = use_regime and HAS_REGIME
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
        """Fetch 1h OHLCV for the full backtest window."""
        try:
            df = fetch_historical_data(symbol, interval=TIMEFRAME, days=self.days)
            if df is None or df.empty or len(df) < 100:
                return None
            df = df.sort_values('timestamp').reset_index(drop=True)
            return df
        except Exception as e:
            self._log(f"  ⚠️  Fetch failed for {symbol}: {e}")
            return None

    # ------------------------------------------------------------------
    def _simulate_position(self, symbol: str, signal: dict,
                           df: pd.DataFrame, entry_idx: int) -> dict:
        """
        Walk forward candle by candle from entry_idx until exit.
        Uses evaluate_exit() (trailing stop + signal reversal) on each candle.
        """
        side        = signal['side']
        entry_price = signal['entry']
        stop_loss   = signal['stop_loss']
        take_profit = signal['take_profit']
        units       = signal['units']
        atr         = signal.get('atr', 0.0)

        position = {
            'side':                side,
            'entry_price':         entry_price,
            'stop_loss':           stop_loss,
            'take_profit':         take_profit,
            'signal_type':         signal.get('signal_type', 'unknown'),
            'atr':                 atr,
            'trailing_stop_active': False,
        }

        exit_price  = None
        exit_reason = 'end_of_data'
        candles_held = 0

        for i in range(entry_idx + 1, len(df)):
            candle       = df.iloc[i]
            current_price = float(candle['close'])
            candles_held += 1

            # Pass a window of recent candles to evaluate_exit for ATR + signal checks
            window = df.iloc[max(0, i - 99): i + 1]

            should_exit, reason = evaluate_exit(
                symbol, position, current_price, window
            )

            if should_exit:
                exit_price  = current_price
                exit_reason = reason
                break

            # Safety: max hold = 7 days (168 candles on 1h)
            if candles_held >= 168:
                exit_price  = current_price
                exit_reason = 'max_hold'
                break

        if exit_price is None:
            exit_price  = float(df.iloc[-1]['close'])
            exit_reason = 'end_of_data'

        # PnL calculation
        if side == 'long':
            gross_pnl = (exit_price - entry_price) * units
        else:
            gross_pnl = (entry_price - exit_price) * units

        entry_fee = entry_price * units * FEE
        exit_fee  = exit_price  * units * FEE
        net_pnl   = gross_pnl - entry_fee - exit_fee
        pnl_pct   = (net_pnl / (entry_price * units)) * 100

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
            'hours_held':   candles_held,
            'exit_reason':  exit_reason,
            'signal_type':  signal.get('signal_type', 'unknown'),
            'regime':       signal.get('regime', 'unknown'),
        }

    # ------------------------------------------------------------------
    def run_coin(self, symbol: str) -> List[dict]:
        """Run backtest for a single coin. Returns list of trade results."""
        self._log(f"\n  📊 {symbol}")
        df = self._fetch(symbol)
        if df is None:
            self._log(f"  ⏭️  No data")
            return []

        coin_trades = []
        warmup      = 100   # candles needed before first signal
        step        = 1     # check every candle (walk-forward)
        in_position = False
        last_entry_idx = -999

        for i in range(warmup, len(df) - 1):
            if in_position:
                continue

            # Enforce cooldown: at least 1 candle gap between trades
            if i - last_entry_idx < 2:
                continue

            window = df.iloc[i - 99: i + 1].copy()

            # Regime detection (optional)
            regime = 'unknown'
            if self.use_regime:
                try:
                    regime = predict_regime(window)
                except Exception:
                    regime = 'unknown'

            # Signal generation
            try:
                signal = generate_trade_signal(
                    df=window,
                    equity=self.equity,
                    risk_per_trade=RISK_PER_TRADE,
                    symbol=symbol,
                    trading_engine=None,
                    regime=regime,
                )
            except Exception as e:
                self._log(f"  ⚠️  Signal error at idx {i}: {e}")
                continue

            if not signal:
                continue

            # Skip if equity too low
            cost = signal['units'] * signal['entry']
            if cost > self.equity * 0.95:
                continue
            if cost < 1.0:
                continue

            # Simulate the position
            in_position = True
            result = self._simulate_position(symbol, signal, df, i)
            last_entry_idx = i + result['candles_held']
            in_position = False

            self.equity += result['net_pnl']
            self.equity_curve.append(self.equity)
            coin_trades.append(result)

            self._log(
                f"  {'✅' if result['net_pnl'] > 0 else '❌'} "
                f"{result['side'].upper()} @ ${result['entry_price']:.4f} → "
                f"${result['exit_price']:.4f} | "
                f"PnL: ${result['net_pnl']:+.4f} | "
                f"{result['exit_reason']} | {result['candles_held']}h"
            )

        return coin_trades

    # ------------------------------------------------------------------
    def run(self) -> dict:
        """Run full backtest across all coins. Returns summary stats."""
        print(f"\n{'='*60}")
        print(f"  🔁 Backtest — {len(self.coins)} coins | {self.days} days | {TIMEFRAME}")
        print(f"  Initial equity: ${INITIAL_EQUITY:.2f}")
        print(f"  Risk/trade: {RISK_PER_TRADE:.1%} | Fee: {FEE:.3%} | Max pos: {MAX_POSITIONS}")
        print(f"  Regime filter: {'on' if self.use_regime else 'off'}")
        print(f"{'='*60}\n")

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

        # Max drawdown from equity curve
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

        # Exit reason breakdown
        exit_reasons = df['exit_reason'].value_counts().to_dict()

        # Signal type breakdown
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
                        help='Disable regime filter (faster but less accurate)')
    parser.add_argument('--verbose',   action='store_true',
                        help='Print every trade as it happens')
    args = parser.parse_args()

    coins = args.coins or CONFIG.get('coins', ['BTC/USDT', 'ETH/USDT', 'SOL/USDT'])

    bt = Backtester(
        coins=coins,
        days=args.days,
        use_regime=not args.no_regime,
        verbose=args.verbose,
    )
    bt.run()
