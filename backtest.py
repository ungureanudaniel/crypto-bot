import sys
import os
import io
import logging

# 1. THE NUCLEAR ENCODING FIX (MUST BE FIRST)
# This forces the entire Python process to prefer UTF-8 for everything.
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    os.environ["PYTHONIOENCODING"] = "utf-8"

# 2. HIJACK LOGGING BEFORE IMPORTS
# This prevents sub-modules from crashing when they try to print emojis
class UTF8Handler(logging.StreamHandler):
    def emit(self, record):
        try:
            msg = self.format(record)
            stream = self.stream
            stream.write(msg + self.terminator)
            self.flush()
        except UnicodeEncodeError:
            # If it still fails, strip emojis and try again
            msg = self.format(record).encode('ascii', 'ignore').decode('ascii')
            self.stream.write(msg + self.terminator)
            self.flush()

# Force the root logger to use our safe handler
root_logger = logging.getLogger()
for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)
handler = UTF8Handler(sys.stdout)
root_logger.addHandler(handler)
root_logger.setLevel(logging.WARNING)

# 3. NOW START IMPORTS
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

try:
    with open('portfolio.json', 'r') as f:
        portfolio_data = json.load(f)
        INITIAL_EQUITY = float(portfolio_data.get('cash', {}).get('USDT', 5000.0))
except Exception as e:
    INITIAL_EQUITY = 5000.0

class Backtester:
    def __init__(self, coins: List[str], days: int = 365, use_regime: bool = True, verbose: bool = False, config_path="config.json"):
        self.config = self._load_config(config_path)
        self.coins = coins
        self.days = days
        self.use_regime = use_regime
        self.verbose = verbose
        self.equity = INITIAL_EQUITY
        self.trades = []
        self.equity_curve = [INITIAL_EQUITY]

    def _load_config(self, path):
        """Safely loads the JSON config file"""
        if not os.path.exists(path):
            print(f"⚠️ Warning: {path} not found. Using default settings.")
            return {}
        with open(path, 'r') as f:
            return json.load(f)

    def get_pair_risk(self, symbol: str):
        """Hierarchical risk lookup: Pair-specific -> Global -> 1% Default"""
        per_pair_config = self.config.get('per_pair', {})
        pair_settings = per_pair_config.get(symbol, {})
        return pair_settings.get('risk_per_trade', self.config.get('risk_per_trade', 0.01))

    def run(self):
        print(f"🚀 Starting Backtest: {self.coins} ({self.days} days)")
        for coin in self.coins:
            print(f"🔍 Analyzing {coin}...")
            coin_trades = self.run_coin(coin)
            self.trades.extend(coin_trades)
        return self.trades

    def run_coin(self, symbol: str) -> List[dict]:
        df = fetch_historical_data(symbol, interval=portfolio_data.get('trading_timeframe', '4h'), days=self.days)
        if df is None or df.empty: return []
        coin_trades = []
        i = 100 

        # Inside run_coin
        base_risk = self.config.get('risk_per_trade', self.config.get('risk_per_trade', 0.03))
        while i < len(df) - 1:
            window = df.iloc[max(0, i - 150): i + 1].copy()
            regime_str = predict_regime(window) if self.use_regime else "Trend UPTREND"
            
            # --- POINT 3: MIRROR LIVE ENGINE FILTERING ---
            if "Expansion" in regime_str:
                i += 1
                continue 
            
            # Adjust risk for Downtrends
            current_risk = base_risk
            if "DOWNTREND" in regime_str:
                current_risk *= 0.75
            # ---------------------------------------------

            signal_data = generate_trade_signal(
                df=window, 
                equity=self.equity,
                risk_per_trade=current_risk, # Corrected parameter name
                symbol=symbol, 
                regime=regime_str,
            )

            if signal_data and signal_data.get('side', 'none') != 'none':
                # Entry happens on the OPEN of the NEXT candle
                entry_price = df.iloc[i+1]['open']
                
                print(f"▶️ [ENTRY] {symbol} {signal_data['side'].upper()} | Price: {entry_price:.2f} | Regime: {regime_str.split('(')[0].strip()}")

                result = self._simulate_position(symbol, signal_data, df, i)
                if result:
                    pnl_color = "🟢" if result['net_pnl'] > 0 else "🔴"
                    print(f"⏹️ [EXIT]  {symbol} | Price: {result['exit_price']:.2f} | PnL: {pnl_color} ${result['net_pnl']:.2f} ({result['pnl_pct']:.2f}%) | Reason: {result['exit_reason']}")
                    
                    result['regime'] = regime_str.split('(')[0].strip()
                    self.equity += result['net_pnl']
                    self.equity_curve.append(self.equity)
                    coin_trades.append(result)
                    
                    i += result['candles_held'] + 1 
                    continue
            
            i += 1
        return coin_trades

    def _simulate_position(self, symbol: str, signal: dict, df: pd.DataFrame, entry_idx: int) -> Optional[dict]:
        if entry_idx + 1 >= len(df): return None
        
        # 1. Fetch pair-specific config for trailing params
        pair_config = self.config.get('per_pair', {}).get(symbol, {})
        
        entry_price = float(df.iloc[entry_idx + 1]['open'])
        position = {
            'symbol': symbol,
            'side': signal['side'],
            'entry_price': entry_price,
            'stop_loss': signal.get('stop_loss'),
            'take_profit': signal.get('take_profit'),
            'units': signal.get('units', 0),
            'atr': signal.get('atr'),
            'regime': signal.get('regime', 'Neutral'),
            'signal_type': signal.get('signal_type', 'backtest_trade'),
            'candles_held': 0,
            # Pass config trailing params into the position object for evaluate_exit
            'trailing_min_pct': pair_config.get('trailing_min_pct', self.config.get('trailing_stop_min_pct')),
            'trailing_max_pct': pair_config.get('trailing_max_pct', self.config.get('trailing_stop_max_pct'))
        }
        
        exit_price = None
        exit_reason = 'timeout'
        
        # Simulation Loop
        for j in range(entry_idx + 1, len(df)):
            current_bar = df.iloc[j]
            position['candles_held'] = j - entry_idx
            
            # Create a rolling window for indicators used in exit logic (e.g., MACD exhaustion)
            window_at_time = df.iloc[max(0, j-50): j+1]
            
            # 2. EVALUATE EXIT
            # We pass the full bar to check for intra-candle Stop Loss/Take Profit
            should_exit, reason = evaluate_exit(symbol, position, current_bar['close'], window_at_time)
            
            if should_exit:
                exit_reason = reason
                # 3. REALISM CHECK: Did we hit SL/TP intra-candle?
                if reason == 'stop_loss':
                    exit_price = position['stop_loss']
                elif reason == 'take_profit':
                    exit_price = position['take_profit']
                else:
                    exit_price = current_bar['close']
                break
            
            # Time-based exit (Hard cap at 2 weeks/336 4h candles)
            if position['candles_held'] > 336:
                exit_price = current_bar['close']
                exit_reason = 'timeout'
                break

        if exit_price is not None:
            # Calculate PnL based on Side
            if position['side'] == 'long':
                gross_pnl = (exit_price - entry_price) * position['units']
            else: # Short
                gross_pnl = (entry_price - exit_price) * position['units']
            
            # 4. DYNAMIC FEE LOOKUP
            # Use spot_fee or futures_fee based on config, fallback to 0.001
            fee_rate = self.config.get('futures_fee' if self.config.get('enable_futures') else 'spot_fee', 0.001)
            costs = (entry_price + exit_price) * position['units'] * fee_rate
            net_pnl = gross_pnl - costs
            
            return {
                **position,
                'exit_price': exit_price,
                'exit_reason': exit_reason,
                'net_pnl': net_pnl,
                'fees': costs,
                'pnl_pct': (net_pnl / (entry_price * position['units']) * 100) if position['units'] > 0 else 0
            }
        return None

    def _summary(self, trades: List[dict]):
        if not trades:
            print("\n⚠️  No trades were executed. Check your signal logic or data range.")
            return

        df = pd.DataFrame(trades)
        # Ensure numeric types for calculations
        df['net_pnl'] = pd.to_numeric(df['net_pnl'])
        
        # Financial Metrics
        total_pnl = df['net_pnl'].sum()
        starting_balance = INITIAL_EQUITY
        final_balance = starting_balance + total_pnl
        total_roi_pct = (total_pnl / starting_balance) * 100
        
        # Performance Metrics
        win_trades = df[df['net_pnl'] > 0]
        loss_trades = df[df['net_pnl'] <= 0]
        win_rate = (len(win_trades) / len(df)) * 100
        
        

        # Profit Factor: Gross Profits / Abs(Gross Losses)
        gross_profit = win_trades['net_pnl'].sum()
        gross_loss = abs(loss_trades['net_pnl'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')

        print(f"\n{'='*55}")
        print(f"💰 FINANCIAL PERFORMANCE SUMMARY 💰")
        print(f"{'='*55}")
        
        # Financial Table
        results_data = [
            ["Starting Capital", f"${starting_balance:,.2f}"],
            ["Final Balance",    f"${final_balance:,.2f}"],
            ["Total Net PnL",    f"${total_pnl:,.2f} ({total_roi_pct:+.2f}%)"],
            ["Total Trades",     len(df)],
            ["Win Rate",         f"{win_rate:.1f}%"],
            ["Profit Factor",    f"{profit_factor:.2f}"],
            ["Avg. PnL / Trade", f"${df['net_pnl'].mean():.2f}"]
        ]
        
        for label, value in results_data:
            print(f"{label:<20} : {value}")

        # Regime Performance Breakdown
        if 'regime' in df.columns:
            print(f"\n--- Performance by Market Regime ---")
            # Group by regime to see where the money is going
            regime_stats = df.groupby('regime')['net_pnl'].agg(['count', 'sum', 'mean']).round(2)
            regime_stats.columns = ['Trades', 'Total PnL', 'Avg PnL']
            
            # Calculate Win Rate per regime
            regime_win_rate = df.groupby('regime').apply(
                lambda x: (len(x[x['net_pnl'] > 0]) / len(x) * 100) if len(x) > 0 else 0
            ).round(1)
            regime_stats['Win %'] = regime_win_rate
            
            print(regime_stats)

        # Exit Reason Breakdown (Essential for debugging)
        print(f"\n--- Exit Breakdown ---")
        exit_counts = df['exit_reason'].value_counts()
        for reason, count in exit_counts.items():
            print(f"{reason:<20} : {count} trades")

        print(f"{'='*55}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--coins', nargs='+', default=None)
    parser.add_argument('--days', type=int, default=365)
    parser.add_argument('--verbose', action='store_true') # ADDED THIS
    args = parser.parse_args()

    target_coins = args.coins if args.coins else ['BNB/USDT']
    bt = Backtester(coins=target_coins, days=args.days, verbose=args.verbose)
    all_trades = bt.run()
    bt._summary(all_trades)