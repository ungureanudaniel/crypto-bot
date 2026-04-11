import pandas as pd
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Load the data - use your CSV file
df = pd.read_csv('eth_usdt_4h_1086d.csv')

# Fix timestamp parsing - use the correct format for your data
# Your timestamps are like "10-04-25 17:00" (day-month-year hour:minute)
df['timestamp'] = pd.to_datetime(df['timestamp'], format='%d-%m-%y %H:%M')
df = df.sort_values('timestamp').reset_index(drop=True)

print(f"Data loaded: {len(df)} candles from {df['timestamp'].min()} to {df['timestamp'].max()}")
print(f"Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}\n")

# Calculate MACD
def calculate_macd(df, fast=12, slow=26, signal=9):
    ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

df['macd_line'], df['signal_line'], df['histogram'] = calculate_macd(df)

# Find consecutive same-direction bars
df['hist_pos'] = df['histogram'] > 0
df['hist_neg'] = df['histogram'] < 0

# Count consecutive positive and negative bars
df['pos_streak'] = df['hist_pos'].groupby((df['hist_pos'] != df['hist_pos'].shift()).cumsum()).cumsum()
df['neg_streak'] = df['hist_neg'].groupby((df['hist_neg'] != df['hist_neg'].shift()).cumsum()).cumsum()
df['pos_streak'] = df['pos_streak'].where(df['hist_pos'], 0).astype(int)
df['neg_streak'] = df['neg_streak'].where(df['hist_neg'], 0).astype(int)

# Bar magnitude (absolute value)
df['hist_abs'] = df['histogram'].abs()

# Shrinking condition: current bar smaller than previous bar in same direction
df['hist_shrinking'] = False
# For positive bars (green, getting smaller)
pos_mask = df['hist_pos'] & df['hist_pos'].shift(1)
df.loc[pos_mask, 'hist_shrinking'] = df.loc[pos_mask, 'hist_abs'] < df.loc[pos_mask, 'hist_abs'].shift(1)
# For negative bars (red, getting smaller)
neg_mask = df['hist_neg'] & df['hist_neg'].shift(1)
df.loc[neg_mask, 'hist_shrinking'] = df.loc[neg_mask, 'hist_abs'] < df.loc[neg_mask, 'hist_abs'].shift(1)

# Price stabilization: ATR < 20-period average ATR
def calculate_atr(df, period=14):
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift(1)).abs()
    low_close = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    return atr

df['atr'] = calculate_atr(df)
df['atr_ma'] = df['atr'].rolling(20).mean()
df['stable'] = df['atr'] < df['atr_ma']  # Low volatility = stable

# Entry signals
df['signal'] = 'none'

# LONG signal: 2+ red bars, shrinking, stable, then enter on continuation
for i in range(5, len(df)):
    # Check if we have 2+ red bars ending at i-1
    if df.loc[i-1, 'neg_streak'] >= 2:
        # Check if the last two bars are shrinking
        if df.loc[i-1, 'hist_shrinking'] and df.loc[i-2, 'hist_shrinking']:
            # Check if stable at i-1
            if df.loc[i-1, 'stable']:
                # Check for one more small drop (price lower than previous bar)
                if df.loc[i, 'close'] < df.loc[i-1, 'close']:
                    # Check if histogram continues shrinking at i
                    if df.loc[i, 'hist_shrinking']:
                        df.loc[i, 'signal'] = 'long'

# SHORT signal: 2+ green bars, shrinking, stable, then enter on continuation
for i in range(5, len(df)):
    if df.loc[i-1, 'pos_streak'] >= 2:
        if df.loc[i-1, 'hist_shrinking'] and df.loc[i-2, 'hist_shrinking']:
            if df.loc[i-1, 'stable']:
                # Check for one more small push up
                if df.loc[i, 'close'] > df.loc[i-1, 'close']:
                    if df.loc[i, 'hist_shrinking']:
                        df.loc[i, 'signal'] = 'short'

# Count signals
long_signals = (df['signal'] == 'long').sum()
short_signals = (df['signal'] == 'short').sum()
print(f"Signals found: {long_signals} long, {short_signals} short\n")

# Backtest simulation
initial_capital = 5000
capital = initial_capital
position = None
entry_price = 0
position_size = 0
entry_time = None
entry_idx = 0
trades = []

for i in range(len(df)):
    current_price = float(df.loc[i, 'close'])
    
    # Exit logic: stop loss at 2%, take profit at 4%
    if position == 'long':
        pnl_pct = (current_price - entry_price) / entry_price
        if pnl_pct <= -0.02 or pnl_pct >= 0.04:
            pnl = position_size * (current_price - entry_price)
            capital += pnl
            trades.append({
                'entry_time': entry_time,
                'exit_time': df.loc[i, 'timestamp'],
                'side': position,
                'entry_price': entry_price,
                'exit_price': current_price,
                'pnl': pnl,
                'pnl_pct': pnl_pct * 100,
                'bars_held': i - entry_idx
            })
            position = None
    
    elif position == 'short':
        pnl_pct = (entry_price - current_price) / entry_price
        if pnl_pct <= -0.02 or pnl_pct >= 0.04:
            pnl = position_size * (entry_price - current_price)
            capital += pnl
            trades.append({
                'entry_time': entry_time,
                'exit_time': df.loc[i, 'timestamp'],
                'side': position,
                'entry_price': entry_price,
                'exit_price': current_price,
                'pnl': pnl,
                'pnl_pct': pnl_pct * 100,
                'bars_held': i - entry_idx
            })
            position = None
    
    # Entry logic
    if position is None and df.loc[i, 'signal'] != 'none':
        position = df.loc[i, 'signal']
        entry_price = current_price
        entry_time = df.loc[i, 'timestamp']
        entry_idx = i
        # Position size: risk 2% of capital
        risk_amount = capital * 0.02
        stop_distance = entry_price * 0.02
        position_size = risk_amount / stop_distance
        position_size = min(position_size, capital * 0.5 / entry_price)  # Cap at 50% of capital

# Final capital
final_capital = capital
total_return = (final_capital - initial_capital) / initial_capital * 100

# Trade statistics
df_trades = pd.DataFrame(trades)

print("=" * 60)
print("📊 MACD BAR EXHAUSTION STRATEGY BACKTEST")
print("=" * 60)
print(f"Period: {df['timestamp'].min()} to {df['timestamp'].max()}")
print(f"Initial Capital: ${initial_capital:,.2f}")
print(f"Final Capital:   ${final_capital:,.2f}")
print(f"Total Return:    {total_return:+.2f}%")
print()

if len(df_trades) > 0:
    winning_trades = df_trades[df_trades['pnl'] > 0]
    losing_trades = df_trades[df_trades['pnl'] <= 0]
    
    print("📈 Trade Statistics:")
    print(f"Total Trades:     {len(df_trades)}")
    print(f"Winning Trades:   {len(winning_trades)}")
    print(f"Losing Trades:    {len(losing_trades)}")
    print(f"Win Rate:         {len(winning_trades)/len(df_trades)*100:.1f}%")
    print(f"Avg Win:          ${winning_trades['pnl'].mean():+.2f}")
    print(f"Avg Loss:         ${losing_trades['pnl'].mean():+.2f}")
    if len(losing_trades) > 0:
        print(f"Profit Factor:    {abs(winning_trades['pnl'].sum() / losing_trades['pnl'].sum()):.2f}")
    else:
        print(f"Profit Factor:    ∞")
    print(f"Avg Bars Held:    {df_trades['bars_held'].mean():.1f}")
    print()
    
    print("📊 Signal Distribution:")
    print(df_trades['side'].value_counts())
    print()
    
    print("🏆 TOP 5 WINNING TRADES:")
    top_winners = df_trades.nlargest(5, 'pnl')[['entry_time', 'side', 'entry_price', 'exit_price', 'pnl_pct', 'bars_held']]
    for _, row in top_winners.iterrows():
        print(f"  {row['entry_time'].strftime('%Y-%m-%d %H:%M')} | {row['side'].upper()} | "
              f"Entry: ${row['entry_price']:.2f} → Exit: ${row['exit_price']:.2f} | "
              f"PnL: {row['pnl_pct']:+.2f}% | Held: {row['bars_held']:.0f}h")
    print()
    
    print("💀 TOP 5 LOSING TRADES:")
    top_losers = df_trades.nsmallest(5, 'pnl')[['entry_time', 'side', 'entry_price', 'exit_price', 'pnl_pct', 'bars_held']]
    for _, row in top_losers.iterrows():
        print(f"  {row['entry_time'].strftime('%Y-%m-%d %H:%M')} | {row['side'].upper()} | "
              f"Entry: ${row['entry_price']:.2f} → Exit: ${row['exit_price']:.2f} | "
              f"PnL: {row['pnl_pct']:+.2f}% | Held: {row['bars_held']:.0f}h")
    print()
    
    # Calculate equity curve and drawdown
    equity_curve = [initial_capital]
    running_capital = initial_capital
    for trade in trades:
        running_capital += trade['pnl']
        equity_curve.append(running_capital)
    
    equity_array = np.array(equity_curve)
    peak = np.maximum.accumulate(equity_array)
    drawdown = (equity_array - peak) / peak * 100
    max_drawdown = drawdown.min()
    
    print("📉 Risk Metrics:")
    print(f"Max Drawdown:     {max_drawdown:.2f}%")
    
else:
    print("❌ No trades generated by the strategy")
    print("\n📊 Signal Condition Breakdown (last 1000 candles):")
    recent = df.tail(1000)
    print(f"Bars with 2+ consecutive red: {(recent['neg_streak'] >= 2).sum()}")
    print(f"Bars with 2+ consecutive green: {(recent['pos_streak'] >= 2).sum()}")
    print(f"Bars with shrinking histogram: {recent['hist_shrinking'].sum()}")
    print(f"Bars with stable condition: {recent['stable'].sum()}")
    
    # Show examples of where conditions were met
    print("\n📋 Example of bars meeting multiple conditions:")
    mask = (recent['neg_streak'] >= 2) & recent['hist_shrinking'] & recent['stable']
    examples = recent[mask].head(5)
    if len(examples) > 0:
        for idx, row in examples.iterrows():
            print(f"  {row['timestamp']} | neg_streak={row['neg_streak']} | hist_shrinking={row['hist_shrinking']} | stable={row['stable']}")
    else:
        print("  No bars met all three conditions simultaneously")