import sys
import os
import io
import logging

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
import json
from typing import List, Optional
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.data_feed import fetch_historical_data
from modules.regime_switcher import predict_regime
from modules.exit_manager import evaluate_exit
from modules.strategy_tools import generate_trade_signal

# ✅ FIXED: use config_loader same as live engine
from config_loader import config as _cfg, get_pair_config

INITIAL_EQUITY = 5000.0
try:
    with open('portfolio.json', 'r') as f:
        portfolio_data = json.load(f)
        INITIAL_EQUITY = float(portfolio_data.get('cash', {}).get('USDT', 5000.0))
except Exception:
    portfolio_data = {}


class Backtester:
    def __init__(self, coins: List[str], days: int = 365, use_regime: bool = True, verbose: bool = False):
        # FIXED: use config_loader directly, same as all other modules
        self.cfg = _cfg.config
        self.coins = coins
        self.days = days
        self.use_regime = use_regime
        self.verbose = verbose
        self.equity = INITIAL_EQUITY
        self.trades = []
        self.equity_curve = [INITIAL_EQUITY]

        # Read all settings from live config once
        self.timeframe           = self.cfg.get('trading_timeframe', '4h')
        self.base_risk           = self.cfg.get('risk_per_trade', 0.03)
        self.trading_fee         = self.cfg.get('trading_fee', 0.0005)
        self.trail_min           = self.cfg.get('trailing_stop_min_pct', 0.04)
        self.trail_max           = self.cfg.get('trailing_stop_max_pct', 0.08)  
        self.trail_activation    = self.cfg.get('trailing_activation_pct', 0.15)
        self.max_candles         = self.cfg.get('max_trade_candles', 336)
        self.default_coins       = self.cfg.get('coins', ['BNB/USDT'])           

        print(f"📋 Backtest config loaded:")
        print(f"   Timeframe:   {self.timeframe}")
        print(f"   Base risk:   {self.base_risk:.1%}")
        print(f"   Fee:         {self.trading_fee:.4%}")
        print(f"   Trail min:   {self.trail_min:.1%}")
        print(f"   Trail max:   {self.trail_max:.1%}")

    def run(self):
        print(f"\nStarting Backtest: {self.coins} ({self.days} days)")
        for coin in self.coins:
            print(f"Analyzing {coin}...")
            self.trades.extend(self.run_coin(coin))
        return self.trades

    def run_coin(self, symbol: str) -> List[dict]:
        df = fetch_historical_data(symbol, interval=self.timeframe, days=self.days)
        if df is None or df.empty:
            print(f"  No data for {symbol}")
            return []

        # ✅ FIXED: use get_pair_config same as live engine
        pair_cfg  = get_pair_config(symbol)
        pair_risk = pair_cfg.get('risk_per_trade', self.base_risk)

        coin_trades = []
        i = 100

        while i < len(df) - 1:
            window     = df.iloc[max(0, i - 150): i + 1].copy()
            regime_str = predict_regime(window) if self.use_regime else "Trend UPTREND"

            # ✅ Mirror live engine filtering
            if "Expansion" in regime_str:
                i += 1
                continue

            # ✅ FIXED: correct risk adjustment (was reading same key twice)
            current_risk = pair_risk
            if "DOWNTREND" in regime_str:
                current_risk *= 0.75

            signal_data = generate_trade_signal(
                df=window,
                equity=self.equity,
                risk_per_trade=current_risk,
                symbol=symbol,
                regime=regime_str,
            )

            if signal_data and signal_data.get('side') in ('long', 'short'):
                entry_price = df.iloc[i + 1]['open']
                side        = signal_data['side']
                regime_label = regime_str.split('(')[0].strip()

                print(f"  [ENTRY] {symbol} {side.upper()} | Price: {entry_price:.4f} | Regime: {regime_label}")

                result = self._simulate_position(symbol, signal_data, df, i)
                if result:
                    pnl_sign = "+" if result['net_pnl'] >= 0 else ""
                    print(f"  [EXIT]  {symbol} | Price: {result['exit_price']:.4f} | "
                          f"PnL: {pnl_sign}${result['net_pnl']:.2f} ({result['pnl_pct']:.2f}%) | "
                          f"Reason: {result['exit_reason']}")

                    result['regime'] = regime_label
                    self.equity += result['net_pnl']
                    self.equity_curve.append(self.equity)
                    coin_trades.append(result)
                    i += result['candles_held'] + 1
                    continue

            i += 1

        return coin_trades

    def _simulate_position(self, symbol: str, signal: dict, df: pd.DataFrame, entry_idx: int) -> Optional[dict]:
        if entry_idx + 1 >= len(df):
            return None

        # ✅ FIXED: use get_pair_config for trailing params
        pair_cfg = get_pair_config(symbol)

        entry_price = float(df.iloc[entry_idx + 1]['open'])
        position = {
            'symbol':        symbol,
            'side':          signal['side'],
            'entry_price':   entry_price,
            'stop_loss':     signal.get('stop_loss'),
            'take_profit':   signal.get('take_profit'),
            'units':         signal.get('units', 0),
            'atr':           signal.get('atr'),
            'regime':        signal.get('regime', 'Neutral'),
            'signal_type':   signal.get('signal_type', 'backtest_trade'),
            'candles_held':  0,
            'last_candle_time': None,  # needed for candle counter in evaluate_exit
            'exit_strategy': signal.get('exit_strategy', 'trailing'),
            'trailing_min_pct': pair_cfg.get('trailing_min_pct', self.trail_min),
            'trailing_max_pct': pair_cfg.get('trailing_max_pct', self.trail_max),
            'trailing_activation_pct': self.trail_activation,  
        }

        exit_price  = None
        exit_reason = 'timeout'

        for j in range(entry_idx + 1, len(df)):
            current_bar = df.iloc[j]

            # ✅ FIXED: let evaluate_exit track candles_held via last_candle_time
            # instead of setting it directly — matches live engine behaviour
            window_at_time = df.iloc[max(0, j - 50): j + 1].copy()

            should_exit, reason = evaluate_exit(
                symbol, position, float(current_bar['close']), window_at_time
            )

            if should_exit:
                exit_reason = reason
                if reason in ('stop_loss', 'trailing_stop'):
                    exit_price = float(position['stop_loss'])   # ✅ use the actual stop price
                elif reason == 'take_profit':
                    exit_price = float(position['take_profit'])
                else:
                    exit_price = float(current_bar['close'])
                break

            if position['candles_held'] > self.max_candles:
                exit_price  = float(current_bar['close'])
                exit_reason = 'timeout'
                break

        if exit_price is None:
            return None

        # ✅ FIXED: fee uses trading_fee same as live engine
        if position['side'] == 'long':
            gross_pnl = (exit_price - entry_price) * position['units']
        else:
            gross_pnl = (entry_price - exit_price) * position['units']

        costs   = (entry_price + exit_price) * position['units'] * self.trading_fee
        net_pnl = gross_pnl - costs

        return {
            **position,
            'exit_price':  exit_price,
            'exit_reason': exit_reason,
            'net_pnl':     net_pnl,
            'fees':        costs,
            'pnl_pct':     (net_pnl / (entry_price * position['units']) * 100) if position['units'] > 0 else 0,
        }

    def _summary(self, trades: List[dict]):
        if not trades:
            print("\nNo trades executed. Check signal logic or data range.")
            return

        df = pd.DataFrame(trades)
        df['net_pnl'] = pd.to_numeric(df['net_pnl'])

        total_pnl    = df['net_pnl'].sum()
        final_bal    = INITIAL_EQUITY + total_pnl
        roi_pct      = (total_pnl / INITIAL_EQUITY) * 100
        win_trades   = df[df['net_pnl'] > 0]
        loss_trades  = df[df['net_pnl'] <= 0]
        win_rate     = (len(win_trades) / len(df)) * 100
        gross_profit = win_trades['net_pnl'].sum()
        gross_loss   = abs(loss_trades['net_pnl'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')

        # Max drawdown
        equity_curve = pd.Series(self.equity_curve)
        rolling_max  = equity_curve.cummax()
        drawdown     = (equity_curve - rolling_max) / rolling_max * 100
        max_drawdown = drawdown.min()

        print(f"\n{'='*55}")
        print(f"FINANCIAL PERFORMANCE SUMMARY")
        print(f"{'='*55}")

        results = [
            ["Starting Capital",  f"${INITIAL_EQUITY:,.2f}"],
            ["Final Balance",     f"${final_bal:,.2f}"],
            ["Total Net PnL",     f"${total_pnl:,.2f} ({roi_pct:+.2f}%)"],
            ["Total Trades",      len(df)],
            ["Win Rate",          f"{win_rate:.1f}%"],
            ["Profit Factor",     f"{profit_factor:.2f}"],
            ["Avg. PnL / Trade",  f"${df['net_pnl'].mean():.2f}"],
            ["Max Drawdown",      f"{max_drawdown:.2f}%"],  # ✅ added
        ]
        for label, value in results:
            print(f"{label:<20} : {value}")

        if 'regime' in df.columns:
            print(f"\n--- Performance by Market Regime ---")
            regime_stats = df.groupby('regime')['net_pnl'].agg(['count', 'sum', 'mean']).round(2)
            regime_stats.columns = ['Trades', 'Total PnL', 'Avg PnL']
            regime_win_rate = df.groupby('regime').apply(
                lambda x: (len(x[x['net_pnl'] > 0]) / len(x) * 100) if len(x) > 0 else 0
            ).round(1)
            regime_stats['Win %'] = regime_win_rate
            print(regime_stats)

        print(f"\n--- Exit Breakdown ---")
        for reason, count in df['exit_reason'].value_counts().items():
            print(f"{reason:<20} : {count} trades")

        print(f"\n--- Signal Type Breakdown ---")  # ✅ added — useful for debugging
        if 'signal_type' in df.columns:
            sig_stats = df.groupby('signal_type')['net_pnl'].agg(['count', 'sum', 'mean']).round(2)
            sig_stats.columns = ['Trades', 'Total PnL', 'Avg PnL']
            win_by_sig = df.groupby('signal_type').apply(
                lambda x: (len(x[x['net_pnl'] > 0]) / len(x) * 100) if len(x) > 0 else 0
            ).round(1)
            sig_stats['Win %'] = win_by_sig
            print(sig_stats)

        print(f"{'='*55}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--coins',   nargs='+', default=None)
    parser.add_argument('--days',    type=int,  default=365)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    bt = Backtester(days=args.days, verbose=args.verbose,
                    coins=args.coins or []) 
    # Resolve coins — CLI args override config
    target_coins = args.coins if args.coins else bt.default_coins
    bt.coins = target_coins

    all_trades = bt.run()
    bt._summary(all_trades)
