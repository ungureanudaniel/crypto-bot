# debug_trading_logic.py
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from telegram import Update
from telegram.ext import ContextTypes

from modules.regime_switcher import predict_regime
from modules.data_feed import fetch_ohlcv
from modules.trade_engine import paper_engine

async def debug_order_details(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Detailed debug of a specific order"""
    if not context.args:
        await update.message.reply_text("Usage: `/debug_order_details SYMBOL`")
        return
    
    symbol = context.args[0].upper()
    
    from modules.portfolio import load_portfolio
    from modules.trade_engine import paper_engine
    
    portfolio = load_portfolio()
    pending_orders = portfolio.get('pending_orders', [])
    current_prices = paper_engine.get_current_prices()
    
    # Find orders for this symbol
    symbol_orders = [o for o in pending_orders if o['symbol'] == symbol]
    
    if not symbol_orders:
        await update.message.reply_text(f"📭 No pending orders for {symbol}")
        return
    
    current_price = current_prices.get(symbol, 0)
    
    msg_lines = [
        f"🔍 *Detailed Order Debug for {symbol}:*",
        f"Current Price: `${current_price:.4f}` (from paper_engine)",
        f"Found {len(symbol_orders)} order(s)",
        ""
    ]
    
    for i, order in enumerate(symbol_orders, 1):
        # Calculate trigger conditions
        buy_triggered = order['side'] == 'buy' and current_price <= order['price']
        sell_triggered = order['side'] == 'sell' and current_price >= order['price']
        
        msg_lines.extend([
            f"*Order {i}:*",
            f"  Side: {order['side'].upper()}",
            f"  Amount: {order['amount']}",
            f"  Limit Price: ${order['price']:.4f}",
            f"  Current Price: ${current_price:.4f}",
            f"  Difference: ${abs(current_price - order['price']):.4f}",
            f"  Condition: Current {'≤' if order['side'] == 'buy' else '≥'} Limit",
            f"  Status: {'✅ TRIGGERED!' if buy_triggered or sell_triggered else '⏳ Waiting'}",
            f"  Price Check: ${current_price:.4f} {'≤' if buy_triggered else '>'} ${order['price']:.4f} = {buy_triggered or sell_triggered}",
            f"  Timestamp: {order.get('timestamp', 'N/A')}",
            f"  Order Type: {order.get('type', 'limit')}",
            ""
        ])
    
    await update.message.reply_text("\n".join(msg_lines), parse_mode='Markdown')

async def debug_scheduler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Debug scheduler status"""
    from services.scheduler import check_pending_orders
    
    await update.message.reply_text("🔧 Checking scheduler status...")
    
    try:
        # Get portfolio info
        from modules.portfolio import load_portfolio
        from modules.trade_engine import paper_engine
        
        portfolio = load_portfolio()
        pending_orders = portfolio.get('pending_orders', [])
        current_prices = paper_engine.get_current_prices()
        
        # Manually run the check
        executed = check_pending_orders(current_prices)
        
        msg_lines = [
            "⚙️ *Scheduler Debug:*",
            f"Pending Orders: {len(pending_orders)}",
            f"Current Prices Available: {len(current_prices)} symbols",
            f"Manual Check Result: {'Executed orders' if executed else 'No orders executed'}",
            ""
        ]
        
        # Check scheduler jobs
        import schedule
        jobs = schedule.get_jobs()
        msg_lines.append(f"Scheduled Jobs: {len(jobs)}")
        
        for job in jobs:
            msg_lines.append(f"  • {job.job_func.__name__}: next run {job.next_run}")
        
        await update.message.reply_text("\n".join(msg_lines), parse_mode='Markdown')
        
    except Exception as e:
        await update.message.reply_text(f"❌ Debug error: {str(e)}")

async def force_order_check(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Force check and execute a specific order"""
    if not context.args:
        await update.message.reply_text("Usage: `/force_order_check SYMBOL`")
        return
    
    symbol = context.args[0].upper()
    
    from modules.portfolio import load_portfolio, save_portfolio
    from modules.trade_engine import paper_engine
    from datetime import datetime
    
    portfolio = load_portfolio()
    pending_orders = portfolio.get('pending_orders', [])
    current_prices = paper_engine.get_current_prices()
    
    # Find orders for this symbol
    symbol_orders = [o for o in pending_orders if o['symbol'] == symbol]
    
    if not symbol_orders:
        await update.message.reply_text(f"📭 No pending orders for {symbol}")
        return
    
    current_price = current_prices.get(symbol, 0)
    
    if not current_price:
        await update.message.reply_text(f"❌ No current price for {symbol}")
        return
    
    executed_count = 0
    
    for order in symbol_orders:
        # Check if should execute
        should_execute = False
        if order['side'] == 'buy' and current_price <= order['price']:
            should_execute = True
        elif order['side'] == 'sell' and current_price >= order['price']:
            should_execute = True
        
        if should_execute:
            try:
                # Execute buy order
                if order['side'] == 'buy':
                    success = paper_engine.open_position(
                        symbol=symbol,
                        side='long',
                        entry_price=current_price,
                        units=order['amount'],
                        stop_loss=current_price * 0.95,
                        take_profit=current_price * 1.10
                    )
                    
                    if success:
                        order['status'] = 'executed'
                        order['executed_at'] = datetime.now().isoformat()
                        order['executed_price'] = current_price
                        executed_count += 1
                        await update.message.reply_text(
                            f"✅ Force executed buy order: {symbol} {order['amount']} @ ${current_price:.4f}"
                        )
                
                # Execute sell order
                elif order['side'] == 'sell':
                    # Your sell logic here
                    pass
                    
            except Exception as e:
                await update.message.reply_text(f"❌ Error executing order: {str(e)}")
    
    # Update portfolio
    if executed_count > 0:
        remaining_orders = [
            order for order in pending_orders 
            if order.get('status') != 'executed'
        ]
        portfolio['pending_orders'] = remaining_orders
        save_portfolio(portfolio)
        
        await update.message.reply_text(
            f"✅ Force executed {executed_count} order(s) for {symbol}"
        )
    else:
        await update.message.reply_text(
            f"⚠️ No orders executed for {symbol} (price: ${current_price:.4f})"
        )

def test_trading_logic():
    """Comprehensive trading logic test across all symbols and regimes"""
    print("=" * 60)
    print("🤖 COMPREHENSIVE TRADING LOGIC TEST")
    print("=" * 60)
    
    # Load config for symbols
    import json
    try:
        with open("config.json", "r") as f:
            config = json.load(f)
        symbols = config.get('coins', ['BTC/USDC', 'ETH/USDC', 'SOL/USDC'])
        trading_mode = config.get('live_trading', False)
    except:
        symbols = ['BTC/USDC', 'ETH/USDC', 'SOL/USDC']
        trading_mode = False
    
    timeframes = ['15m', '1h', '4h']
    
    total_tests = 0
    valid_signals = 0
    actionable_signals = []
    
    print(f"Mode: {'🚀 LIVE TRADING' if trading_mode else '📝 PAPER TRADING'}")
    print(f"Symbols to test: {len(symbols)}")
    print("-" * 60)
    
    for symbol in symbols:
        print(f"\n📊 {symbol}")
        print("-" * 40)
        
        for timeframe in timeframes:
            total_tests += 1
            
            try:
                # Get OHLCV data
                df = fetch_ohlcv(symbol, timeframe, limit=200)
                
                if df.empty or len(df) < 50:
                    print(f"   [{timeframe}] ❌ Insufficient data ({len(df)} rows)")
                    continue
                
                # Predict regime
                regime = predict_regime(df)
                
                # Get price metrics
                current_price = df.iloc[-1]['close']
                prev_price = df.iloc[-2]['close'] if len(df) > 1 else current_price
                price_change = ((current_price - prev_price) / prev_price) * 100
                
                # Get volume metrics
                current_volume = df.iloc[-1]['volume']
                avg_volume = df['volume'].mean()
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
                
                # Get volatility (ATR approximation)
                high_low_range = (df['high'] - df['low']).mean()
                atr_pct = (high_low_range / current_price) * 100 if current_price > 0 else 0
                
                # Get moving averages
                if len(df) >= 20:
                    sma_20 = df['close'].rolling(window=20).mean().iloc[-1]
                    sma_50 = df['close'].rolling(window=50).mean().iloc[-1] if len(df) >= 50 else None
                    
                    above_sma_20 = current_price > sma_20
                    if sma_50:
                        above_sma_50 = current_price > sma_50
                        trend_aligned = above_sma_20 == above_sma_50
                    else:
                        trend_aligned = True
                else:
                    sma_20 = None
                    trend_aligned = True
                
                # Determine trade action based on regime
                trade_action = "HOLD"
                trade_confidence = 0
                trade_side = None
                
                if "Breakout" in regime:
                    # Extract confidence from regime string (e.g., "Breakout (85%)")
                    try:
                        if '(' in regime and '%' in regime:
                            confidence_str = regime.split('(')[1].split('%')[0]
                            trade_confidence = int(confidence_str)
                        else:
                            trade_confidence = 75
                    except:
                        trade_confidence = 75
                    
                    if trade_confidence > 70:
                        trade_action = "BUY"
                        trade_side = "long"
                        if "Bullish" in regime or price_change > 0:
                            trade_action = "BUY"
                            trade_side = "long"
                        elif "Bearish" in regime or price_change < 0:
                            trade_action = "SELL" 
                            trade_side = "short"
                
                elif "Trending" in regime:
                    try:
                        if '(' in regime and '%' in regime:
                            confidence_str = regime.split('(')[1].split('%')[0]
                            trade_confidence = int(confidence_str)
                        else:
                            trade_confidence = 65
                    except:
                        trade_confidence = 65
                    
                    if trade_confidence > 60:
                        if "Up" in regime or price_change > 0:
                            trade_action = "BUY"
                            trade_side = "long"
                        elif "Down" in regime or price_change < 0:
                            trade_action = "SELL"
                            trade_side = "short"
                
                elif "Range-Bound" in regime:
                    try:
                        if '(' in regime and '%' in regime:
                            confidence_str = regime.split('(')[1].split('%')[0]
                            trade_confidence = int(confidence_str)
                        else:
                            trade_confidence = 80
                    except:
                        trade_confidence = 80
                    
                    if trade_confidence > 75:
                        trade_action = "RANGE_TRADE"
                        # Range trading logic would go here
                
                # Check if this is an actionable signal
                is_actionable = trade_action in ["BUY", "SELL"] and trade_confidence > 70
                
                if is_actionable:
                    valid_signals += 1
                    actionable_signals.append({
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'regime': regime,
                        'action': trade_action,
                        'confidence': trade_confidence,
                        'price': current_price,
                        'price_change': price_change
                    })
                
                # Print results
                print(f"   [{timeframe}] {regime}")
                print(f"      Price: ${current_price:.2f} ({price_change:+.2f}%)")
                print(f"      Volume: {volume_ratio:.1f}x avg")
                print(f"      Volatility: {atr_pct:.1f}%")
                if sma_20:
                    print(f"      SMA20: ${sma_20:.2f} ({'above' if above_sma_20 else 'below'})")
                
                if trade_action != "HOLD":
                    print(f"      ⚡ Action: {trade_action} ({trade_confidence}% confidence)")
                    if trade_side:
                        # Calculate position size
                        from modules.portfolio import load_portfolio
                        portfolio = load_portfolio()
                        equity = portfolio.get('cash_balance', 1000)
                        position_size = equity * 0.02  # 2% risk
                        units = position_size / current_price
                        
                        print(f"      📈 Position: {units:.6f} units (${position_size:.2f})")
                        if trade_side == "long":
                            sl = current_price * 0.95
                            tp = current_price * 1.10
                        else:
                            sl = current_price * 1.05
                            tp = current_price * 0.90
                        print(f"      🛡️  SL/TP: ${sl:.2f} / ${tp:.2f}")
                else:
                    print(f"      💤 No trade (confidence: {trade_confidence}%)")
                    
                print()
                
            except Exception as e:
                print(f"   [{timeframe}] ❌ Error: {str(e)[:50]}")
                continue
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY")
    print("=" * 60)
    print(f"Total tests: {total_tests}")
    print(f"Valid signals: {valid_signals}")
    print(f"Signal rate: {(valid_signals/total_tests*100):.1f}%")
    
    if actionable_signals:
        print(f"\n🎯 ACTIONABLE SIGNALS ({len(actionable_signals)}):")
        for signal in actionable_signals:
            print(f"  • {signal['symbol']} [{signal['timeframe']}]: {signal['action']} @ ${signal['price']:.2f}")
            print(f"    Regime: {signal['regime']} ({signal['confidence']}% confidence)")
        
        # Show the strongest signal
        strongest = max(actionable_signals, key=lambda x: x['confidence'])
        print(f"\n💪 STRONGEST SIGNAL:")
        print(f"  {strongest['symbol']} [{strongest['timeframe']}]: {strongest['action']}")
        print(f"  Confidence: {strongest['confidence']}%, Price: ${strongest['price']:.2f}")
        
        # Ask if user wants to execute
        print(f"\n🚀 Execute strongest signal?")
        print(f"  Command: /trade {strongest['symbol']}")
    else:
        print("\n😴 No actionable signals found")
        print("💡 Markets might be quiet or in consolidation")
    
    print("=" * 60)
    
    # Return actionable signals for further processing
    return actionable_signals


def quick_regime_check():
    """Quick regime check for all coins"""
    print("⚡ Quick Regime Check")
    print("-" * 40)
    
    import json
    try:
        with open("config.json", "r") as f:
            config = json.load(f)
        symbols = config.get('coins', ['BTC/USDC', 'ETH/USDC', 'SOL/USDC'])[:5]
    except:
        symbols = ['BTC/USDC', 'ETH/USDC', 'SOL/USDC']
    
    results = []
    
    for symbol in symbols:
        try:
            df = fetch_ohlcv(symbol, '15m', limit=50)
            if not df.empty and len(df) > 20:
                regime = predict_regime(df)
                price = df.iloc[-1]['close']
                prev_price = df.iloc[-2]['close']
                change = ((price - prev_price) / prev_price) * 100
                
                # Color code based on regime
                if "Breakout" in regime:
                    icon = "🚀"
                elif "Trending Up" in regime:
                    icon = "📈"
                elif "Trending Down" in regime:
                    icon = "📉"
                elif "Range-Bound" in regime:
                    icon = "📊"
                else:
                    icon = "❓"
                
                print(f"{icon} {symbol:12} ${price:8.2f} ({change:+.2f}%) - {regime}")
                results.append({
                    'symbol': symbol,
                    'regime': regime,
                    'price': price,
                    'change': change
                })
        except Exception as e:
            print(f"❌ {symbol:12} Error: {str(e)[:30]}")
    
    return results


# Add these to your main.py or debug commands
if __name__ == "__main__":
    print("\nRunning trading logic tests...")
    
    # Run quick check first
    quick_regime_check()
    
    print("\n" + "=" * 60)
    
    # Run comprehensive test
    actionable = test_trading_logic()
    
    # If you want to test execution
    if actionable:
        print("\nTesting trade execution for strongest signal...")
        strongest = max(actionable, key=lambda x: x['confidence'])
        
        from modules.trade_engine import paper_engine
        from modules.portfolio import load_portfolio
        
        portfolio = load_portfolio()
        cash = portfolio.get('cash_balance', 0)
        
        print(f"\n💰 Available cash: ${cash:.2f}")
        print(f"🎯 Would execute: {strongest['action']} {strongest['symbol']}")
        print(f"📊 Price: ${strongest['price']:.2f}")
        
        if cash > 10:  # Minimum trade size
            position_size = cash * 0.02
            units = position_size / strongest['price']
            print(f"📈 Position size: ${position_size:.2f} = {units:.6f} units")
            
            # Calculate SL/TP
            if strongest['action'] == "BUY":
                sl = strongest['price'] * 0.95
                tp = strongest['price'] * 1.10
            else:
                sl = strongest['price'] * 1.05
                tp = strongest['price'] * 0.90
            
            print(f"🛡️  Stop Loss: ${sl:.2f}")
            print(f"🎯 Take Profit: ${tp:.2f}")
            
            # Ask if we should execute
            print("\n💡 To execute manually:")
            print(f"   /trade {strongest['symbol']}")

# quick_test_eth.py
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_eth_trade():
    from modules.regime_switcher import predict_regime
    print("Testing ETH trending trade...")
    
    symbol = "ETH/USDC"
    df = fetch_ohlcv(symbol, "15m")
    
    if not df.empty:
        regime = predict_regime(df)
        price = df.iloc[-1]['close']
        
        print(f"Symbol: {symbol}")
        print(f"Regime: {regime}")
        print(f"Price: ${price:.2f}")
        print("Executing trade (should work now)...")
        
        # This should now execute a trade!
        paper_engine.execute_trade(symbol, regime, price)
        print("✅ Trade execution completed!")
    else:
        print("❌ No data")

if __name__ == "__main__":
    test_eth_trade()
    # test_trading_logic()