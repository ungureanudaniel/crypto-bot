# debug_trading_logic.py
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from telegram import Update
from telegram.ext import ContextTypes

from modules.regime_switcher import predict_regime
from modules.data_feed import fetch_ohlcv
from modules.papertrade_engine import paper_engine

async def debug_order_details(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Detailed debug of a specific order"""
    if not context.args:
        await update.message.reply_text("Usage: `/debug_order_details SYMBOL`")
        return
    
    symbol = context.args[0].upper()
    
    from modules.portfolio import load_portfolio
    from modules.papertrade_engine import paper_engine
    
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
        from modules.papertrade_engine import paper_engine
        
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
    from modules.papertrade_engine import paper_engine
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
    print("Testing trading logic across all regimes...")
    
    test_cases = [
        ('BTC/USDC', '15m'),
        ('ETH/USDC', '15m'),
        ('SOL/USDC', '15m')
    ]
    
    for symbol, timeframe in test_cases:
        print(f"\n🔍 Testing {symbol} ({timeframe}):")
        
        try:
            # Get current data
            df = fetch_ohlcv(symbol, timeframe)
            if df.empty:
                print(f"   ❌ No data for {symbol}")
                continue
            
            # Predict regime
            regime_prediction = predict_regime(df)
            print(f"   📊 Regime prediction: {regime_prediction}")
            
            # Get current price
            current_price = df.iloc[-1]['close']
            print(f"   💰 Current price: ${current_price:.2f}")
            
            # Test what execute_trade would do
            print(f"   🤖 What bot would do:")
            
            if "Range-Bound" in regime_prediction:
                print("      ❌ SKIP TRADE (range-bound)")
            elif "Trending" in regime_prediction:
                print("      ❌ SKIP TRADE (trending) - THIS SHOULD TRADE!")
            elif "Breakout" in regime_prediction:
                print("      ✅ EXECUTE TRADE (breakout)")
            else:
                print(f"      ❓ Unknown regime: {regime_prediction}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")

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