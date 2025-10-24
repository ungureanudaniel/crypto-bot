# debug_portfolio.py
import json
from telegram import Update
from telegram.ext import ContextTypes

def check_portfolio_structure():
    """Check what's actually in portfolio.json"""
    try:
        with open("portfolio.json", "r") as f:
            portfolio = json.load(f)
        
        print("🔍 Portfolio Structure Analysis:")
        print(f"💰 Cash: ${portfolio.get('cash_balance', 0):.2f}")
        
        holdings = portfolio.get('holdings', {})
        print(f"📦 Holdings count: {len(holdings)}")
        for coin, amount in holdings.items():
            print(f"   • {coin}: {amount}")
        
        positions = portfolio.get('positions', {})
        print(f"🛡️ Positions count: {len(positions)}")
        for symbol, position in positions.items():
            print(f"   • {symbol}: {position['amount']} units")
            print(f"     Entry: ${position['entry_price']:.2f}")
            print(f"     SL: ${position['stop_loss']:.2f}")
            print(f"     TP: ${position['take_profit']:.2f}")
            
        if not holdings and not positions:
            print("❌ No holdings OR positions found!")
            
    except Exception as e:
        print(f"Error: {e}")

async def simple_debug(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Simple debug command that should definitely work"""
    from trade_engine import load_portfolio
    
    try:
        portfolio = load_portfolio()
        positions = portfolio.get('positions', {})
        
        if not positions:
            await update.message.reply_text("❌ No positions found in portfolio")
            return
        
        response = ["🛡️ ACTIVE POSITIONS:"]
        for symbol, position in positions.items():
            response.append(
                f"• {symbol}: {position['amount']:.6f} units\n"
                f"  Entry: ${position['entry_price']:.2f}\n"
                f"  Stop Loss: ${position['stop_loss']:.2f}\n"
                f"  Take Profit: ${position['take_profit']:.2f}"
            )
        
        await update.message.reply_text("\n".join(response))
        
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {str(e)}")

async def debug_stop_losses(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Check why stop losses aren't working"""
    from trade_engine import load_portfolio
    from data_feed import get_current_prices
    
    try:
        portfolio = load_portfolio()
        positions = portfolio.get('positions', {})
        current_prices = get_current_prices()
        
        if not positions:
            await update.message.reply_text("❌ No positions to check")
            return
        
        response = ["🔍 STOP LOSS ANALYSIS:"]
        
        for symbol, position in positions.items():
            current_price = current_prices.get(symbol, position['entry_price'])
            stop_loss = position['stop_loss']
            take_profit = position['take_profit']
            
            # Calculate distances
            distance_to_sl_pct = (current_price - stop_loss) / current_price * 100
            distance_to_tp_pct = (take_profit - current_price) / current_price * 100
            current_pnl_pct = (current_price - position['entry_price']) / position['entry_price'] * 100
            
            sl_triggered = current_price <= stop_loss
            tp_triggered = current_price >= take_profit
            
            response.append(
                f"🎯 {symbol}:\n"
                f"  Current: ${current_price:.2f} ({current_pnl_pct:+.1f}%)\n"
                f"  Stop Loss: ${stop_loss:.2f} ({distance_to_sl_pct:+.1f}% away)\n"
                f"  Take Profit: ${take_profit:.2f} ({distance_to_tp_pct:+.1f}% away)\n"
                f"  SL Triggered: {'✅ YES' if sl_triggered else '❌ NO'}\n"
                f"  TP Triggered: {'✅ YES' if tp_triggered else '❌ NO'}"
            )
        
        await update.message.reply_text("\n".join(response))
        
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {str(e)}")
        
if __name__ == "__main__":
    check_portfolio_structure()