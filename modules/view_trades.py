"""
Display and analyze trade logs
"""

import os
import pandas as pd
from datetime import datetime

LOG_FILE = "logs/trades.log"

def load_trades():
    """Load trades from log file into a DataFrame"""
    if not os.path.exists(LOG_FILE):
        print(f"❌ Trade log not found: {LOG_FILE}")
        return None
    
    trades = []
    with open(LOG_FILE, 'r') as f:
        for line in f:
            if '|' not in line:
                continue
            parts = line.strip().split('|')
            if len(parts) < 3:
                continue
            
            timestamp = parts[0]
            action = parts[1]
            
            if action == 'OPEN':
                # OPEN|symbol|side|entry|units|stop_loss|take_profit
                trades.append({
                    'timestamp': timestamp,
                    'action': 'open',
                    'symbol': parts[2],
                    'side': parts[3],
                    'entry_price': float(parts[4]),
                    'units': float(parts[5]),
                    'stop_loss': float(parts[6]),
                    'take_profit': float(parts[7])
                })
            elif action == 'CLOSE':
                # CLOSE|symbol|side|entry|exit|pnl|pnl_pct|reason
                trades.append({
                    'timestamp': timestamp,
                    'action': 'close',
                    'symbol': parts[2],
                    'side': parts[3],
                    'entry_price': float(parts[4]),
                    'exit_price': float(parts[5]),
                    'pnl': float(parts[6]),
                    'pnl_pct': float(parts[7]),
                    'reason': parts[8]
                })
    
    return pd.DataFrame(trades)


def show_summary(df):
    """Show trade summary"""
    if df is None or df.empty:
        return
    
    opens = df[df['action'] == 'open']
    closes = df[df['action'] == 'close']
    
    print(f"\n{'='*60}")
    print(f"📊 TRADE SUMMARY")
    print(f"{'='*60}")
    print(f"Total trades: {len(closes)}")
    print(f"Winning trades: {len(closes[closes['pnl'] > 0])}")
    print(f"Losing trades: {len(closes[closes['pnl'] <= 0])}")
    print(f"Win rate: {len(closes[closes['pnl'] > 0]) / len(closes) * 100:.1f}%" if len(closes) > 0 else "N/A")
    print(f"Total PnL: ${closes['pnl'].sum():.4f}")
    print(f"Best trade: ${closes['pnl'].max():.4f} ({closes.loc[closes['pnl'].idxmax(), 'symbol']})")
    print(f"Worst trade: ${closes['pnl'].min():.4f} ({closes.loc[closes['pnl'].idxmin(), 'symbol']})")
    
    print(f"\n{'='*60}")
    print("📈 RECENT TRADES (last 10)")
    print(f"{'='*60}")
    
    for _, trade in closes.tail(10).iterrows():
        icon = '✅' if trade['pnl'] > 0 else '❌'
        print(f"{icon} {trade['timestamp']} | {trade['symbol']} {trade['side'].upper()} | "
              f"Entry: ${trade['entry_price']:.4f} → Exit: ${trade['exit_price']:.4f} | "
              f"PnL: ${trade['pnl']:+.4f} ({trade['pnl_pct']:+.2f}%) | {trade['reason']}")


def show_best_trades(df):
    """Show best performing trades"""
    closes = df[df['action'] == 'close']
    if closes.empty:
        return
    
    print(f"\n{'='*60}")
    print("🏆 TOP 10 BEST TRADES")
    print(f"{'='*60}")
    
    best = closes.nlargest(10, 'pnl_pct')
    for _, trade in best.iterrows():
        print(f"{trade['timestamp']} | {trade['symbol']} {trade['side'].upper()} | "
              f"+{trade['pnl_pct']:.2f}% | ${trade['pnl']:+.4f} | {trade['reason']}")


if __name__ == "__main__":
    df = load_trades()
    if df is not None:
        show_summary(df)
        show_best_trades(df)