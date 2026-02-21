# services/telegram_bot.py - UPDATED FOR NEW STRATEGY
import json
import sys
import logging
import signal
import os
import asyncio
from datetime import datetime
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

stop_event = None

# -------------------------------------------------------------------
# SETUP PATHS FIRST
# -------------------------------------------------------------------
current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_file_dir)
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'modules'))

# -------------------------------------------------------------------
# LOGGING
# -------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# FIX EVENT LOOP FOR WINDOWS
# -------------------------------------------------------------------
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

def signal_handler(signum, frame):
    """Handle system signals"""
    logger.info(f"📴 Received signal {signum}")
    if stop_event:
        stop_event.set()

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
def get_config():
    try:
        from config_loader import config as app_config
        return app_config.config
    except:
        return {
            'trading_mode': 'paper',
            'telegram_token': os.environ.get('TELEGRAM_TOKEN', ''),
            'telegram_chat_id': os.environ.get('TELEGRAM_CHAT_ID', '')
        }

CONFIG = get_config()

# -------------------------------------------------------------------
# PORTFOLIO
# -------------------------------------------------------------------
PORTFOLIO_FILE = os.path.join(project_root, "portfolio.json")

def load_portfolio():
    if os.path.exists(PORTFOLIO_FILE):
        try:
            with open(PORTFOLIO_FILE, "r") as f:
                return json.load(f)
        except:
            pass
    return {"cash_balance": 10000, "positions": {}, "trade_history": []}

# -------------------------------------------------------------------
# SCHEDULER JOBS - USING UPDATED SCHEDULER
# -------------------------------------------------------------------
async def trading_job_callback(context: ContextTypes.DEFAULT_TYPE):
    """Run trading jobs - stop loss check and signal scan"""
    logger.info("⏰ Running scheduled trading jobs...")
    try:
        # Import scheduler functions
        from services.scheduler import (
            check_stop_losses_and_take_profits,
            scan_for_trading_signals,
            check_pending_orders
        )
        
        # Run jobs
        check_stop_losses_and_take_profits()
        check_pending_orders()
        
        # Only scan for signals if we have capacity
        from modules.portfolio import load_portfolio
        from modules.trade_engine import trading_engine
        
        portfolio = load_portfolio()
        current_positions = len(portfolio.get('positions', {}))
        
        if current_positions < trading_engine.max_positions:
            scan_for_trading_signals()
        
        logger.info("✅ Scheduled jobs completed")
    except Exception as e:
        logger.error(f"❌ Trading job error: {e}")

async def portfolio_job_callback(context: ContextTypes.DEFAULT_TYPE):
    """Portfolio update job"""
    logger.info("💰 Running portfolio update...")
    try:
        from services.scheduler import update_portfolio_summary
        update_portfolio_summary()
    except Exception as e:
        logger.error(f"❌ Portfolio job error: {e}")

async def health_job_callback(context: ContextTypes.DEFAULT_TYPE):
    """Health check job"""
    logger.info("🏥 Running health check...")
    try:
        from services.scheduler import health_check
        health_check()
    except Exception as e):
        logger.error(f"❌ Health check error: {e}")

# -------------------------------------------------------------------
# TELEGRAM COMMANDS
# -------------------------------------------------------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Start command"""
    try:
        from modules.trade_engine import trading_engine
        summary = trading_engine.get_portfolio_summary()
        
        await update.message.reply_text(
            f"🤖 *Trading Bot Started!*\n\n"
            f"📊 Mode: {summary['trading_mode'].upper()}\n"
            f"💰 Portfolio: ${summary['portfolio_value']:,.2f}\n"
            f"💵 Cash: ${summary['cash_balance']:,.2f}\n"
            f"📈 Return: {summary['total_return_pct']:+.1f}%\n"
            f"🎯 Active: {summary['active_positions']}/{trading_engine.max_positions}\n\n"
            f"Use /help for commands",
            parse_mode='Markdown'
        )
    except Exception as e:
        logger.error(f"Start error: {e}")
        await update.message.reply_text("❌ Error starting bot")

async def balance(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show balance"""
    try:
        from modules.trade_engine import trading_engine
        summary = trading_engine.get_portfolio_summary()
        
        await update.message.reply_text(
            f"💰 *Portfolio Balance*\n\n"
            f"Total Value: `${summary['portfolio_value']:,.2f}`\n"
            f"Cash: `${summary['cash_balance']:,.2f}`\n"
            f"Positions Value: `${summary['positions_value']:,.2f}`\n"
            f"Return: `{summary['total_return_pct']:+.1f}%`",
            parse_mode='Markdown'
        )
    except Exception as e:
        logger.error(f"Balance error: {e}")
        await update.message.reply_text("❌ Error getting balance")

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show detailed status"""
    try:
        from modules.trade_engine import trading_engine
        from modules.portfolio import load_portfolio
        
        summary = trading_engine.get_portfolio_summary()
        portfolio = load_portfolio()
        positions = portfolio.get('positions', {})
        trade_history = portfolio.get('trade_history', [])
        
        # Calculate win rate from closed trades
        closed_trades = [t for t in trade_history if t.get('action') == 'close']
        winning_trades = [t for t in closed_trades if t.get('pnl', 0) > 0]
        win_rate = (len(winning_trades) / len(closed_trades) * 100) if closed_trades else 0
        
        # Build message
        message_lines = [
            f"🤖 *Trading Bot Status*\n",
            f"━━━━━━━━━━━━━━━━",
            f"📊 Mode: `{summary['trading_mode'].upper()}`",
            f"💰 Portfolio: `${summary['portfolio_value']:,.2f}`",
            f"💵 Cash: `${summary['cash_balance']:,.2f}`",
            f"📈 Return: `{summary['total_return_pct']:+.1f}%`",
            f"🎯 Win Rate: `{win_rate:.1f}%`",
            f"📊 Active: `{summary['active_positions']}/{trading_engine.max_positions}`",
            f"📋 Total Trades: `{len(closed_trades)}`",
        ]
        
        if positions:
            message_lines.append(f"\n📊 *Active Positions:*")
            message_lines.append(f"━━━━━━━━━━━━━━━━")
            
            # Get current prices
            current_prices = trading_engine.get_current_prices()
            
            for symbol, position in positions.items():
                current_price = current_prices.get(symbol, position.get('entry_price', 0))
                entry_price = position.get('entry_price', 0)
                
                if position['side'] == 'long':
                    pnl_pct = ((current_price / entry_price) - 1) * 100
                    pnl_emoji = "🟢" if pnl_pct > 0 else "🔴" if pnl_pct < 0 else "⚪"
                else:  # short
                    pnl_pct = (1 - (current_price / entry_price)) * 100
                    pnl_emoji = "🟢" if pnl_pct > 0 else "🔴" if pnl_pct < 0 else "⚪"
                
                message_lines.append(
                    f"\n{pnl_emoji} *{symbol}*"
                    f"\n   Entry: `${entry_price:.2f}`"
                    f"\n   Current: `${current_price:.2f}`"
                    f"\n   P&L: `{pnl_pct:+.1f}%`"
                    f"\n   Stop: `${position.get('stop_loss', 0):.2f}`"
                    f"\n   Target: `${position.get('take_profit', 0):.2f}`"
                )
        
        await update.message.reply_text("\n".join(message_lines), parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"❌ Status error: {e}")
        await update.message.reply_text(f"❌ Error: {str(e)[:100]}", parse_mode='Markdown')

async def scan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Manually trigger scan for signals"""
    await update.message.reply_text("🔍 Scanning for trading signals...", parse_mode='Markdown')
    
    try:
        from modules.trade_engine import trading_engine
        
        signals_found = trading_engine.scan_and_trade()
        
        if not signals_found:
            await update.message.reply_text(
                "📭 No trading signals found.",
                parse_mode='Markdown'
            )
            return
        
        # Group signals by type
        breakout_signals = [s for s in signals_found if 'breakout' in s['signal'].get('signal_type', '')]
        trend_signals = [s for s in signals_found if 'trend' in s['signal'].get('signal_type', '')]
        momentum_signals = [s for s in signals_found if s not in breakout_signals + trend_signals]
        
        message_lines = [f"📊 *Found {len(signals_found)} Signal(s):*\n"]
        
        if breakout_signals:
            message_lines.append(f"*🚀 Breakout Signals:*")
            for s in breakout_signals[:3]:
                sig = s['signal']
                message_lines.append(
                    f"  • {s['symbol']}: {sig['side'].upper()} @ `${sig['entry']:.2f}`"
                )
        
        if trend_signals:
            message_lines.append(f"\n*📈 Trend Signals:*")
            for s in trend_signals[:3]:
                sig = s['signal']
                message_lines.append(
                    f"  • {s['symbol']}: {sig['side'].upper()} @ `${sig['entry']:.2f}`"
                )
        
        if momentum_signals:
            message_lines.append(f"\n*⚡ Momentum Signals:*")
            for s in momentum_signals[:3]:
                sig = s['signal']
                message_lines.append(
                    f"  • {s['symbol']}: {sig['side'].upper()} @ `${sig['entry']:.2f}`"
                )
        
        # Add note if more signals
        if len(signals_found) > 9:
            message_lines.append(f"\n*... and {len(signals_found) - 9} more*")
        
        message_lines.append(f"\nUse `/execute_all` to execute all signals")
        
        # Store in user data
        if context.user_data is None:
            context.user_data = {}
        context.user_data['pending_signals'] = signals_found
        
        await update.message.reply_text("\n".join(message_lines), parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"❌ Scan error: {e}")
        await update.message.reply_text(f"❌ Error: {str(e)[:100]}", parse_mode='Markdown')

async def execute_all(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Execute all pending signals"""
    if not context.user_data or 'pending_signals' not in context.user_data:
        await update.message.reply_text(
            "❌ No pending signals. Run `/scan` first.",
            parse_mode='Markdown'
        )
        return
    
    signals = context.user_data['pending_signals']
    
    await update.message.reply_text(
        f"⚡ Executing {len(signals)} signal(s)...",
        parse_mode='Markdown'
    )
    
    try:
        from modules.trade_engine import trading_engine
        
        executed = []
        failed = []
        
        for signal_data in signals:
            success = trading_engine.execute_signal(signal_data)
            if success:
                executed.append(signal_data['symbol'])
            else:
                failed.append(signal_data['symbol'])
        
        # Clear pending signals
        context.user_data['pending_signals'] = []
        
        # Build response
        response = ["📊 *Execution Results:*\n"]
        
        if executed:
            response.append(f"✅ *Executed ({len(executed)}):*")
            for sym in executed[:5]:
                response.append(f"  • {sym}")
            if len(executed) > 5:
                response.append(f"  ... and {len(executed) - 5} more")
        
        if failed:
            response.append(f"\n❌ *Failed ({len(failed)}):*")
            for sym in failed[:5]:
                response.append(f"  • {sym}")
        
        await update.message.reply_text("\n".join(response), parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"❌ Execute all error: {e}")
        await update.message.reply_text(f"❌ Error: {str(e)[:100]}", parse_mode='Markdown')

async def execute(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Execute specific signal"""
    if not context.args:
        await update.message.reply_text(
            "Usage: `/execute SYMBOL`\nExample: `/execute BTC/USDC`",
            parse_mode='Markdown'
        )
        return
    
    symbol = context.args[0].upper()
    
    if not context.user_data or 'pending_signals' not in context.user_data:
        await update.message.reply_text(
            "❌ No pending signals. Run `/scan` first.",
            parse_mode='Markdown'
        )
        return
    
    # Find signal
    signal_to_execute = None
    for signal_data in context.user_data['pending_signals']:
        if signal_data['symbol'].upper() == symbol:
            signal_to_execute = signal_data
            break
    
    if not signal_to_execute:
        await update.message.reply_text(
            f"❌ No pending signal for {symbol}",
            parse_mode='Markdown'
        )
        return
    
    await update.message.reply_text(
        f"⚡ Executing signal for {symbol}...",
        parse_mode='Markdown'
    )
    
    try:
        from modules.trade_engine import trading_engine
        
        success = trading_engine.execute_signal(signal_to_execute)
        
        if success:
            # Remove from pending
            context.user_data['pending_signals'] = [
                s for s in context.user_data['pending_signals']
                if s['symbol'].upper() != symbol
            ]
            
            await update.message.reply_text(
                f"✅ *Signal Executed!*\n\n"
                f"Position opened for {symbol}",
                parse_mode='Markdown'
            )
        else:
            await update.message.reply_text(
                f"❌ Failed to execute signal for {symbol}",
                parse_mode='Markdown'
            )
            
    except Exception as e:
        logger.error(f"❌ Execute error: {e}")
        await update.message.reply_text(f"❌ Error: {str(e)[:100]}", parse_mode='Markdown')

async def positions(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show current positions"""
    try:
        from modules.portfolio import load_portfolio
        from modules.trade_engine import trading_engine
        
        portfolio = load_portfolio()
        positions = portfolio.get('positions', {})
        
        if not positions:
            await update.message.reply_text("📭 No open positions", parse_mode='Markdown')
            return
        
        current_prices = trading_engine.get_current_prices()
        
        message_lines = [f"📊 *Open Positions ({len(positions)}):*\n"]
        
        total_pnl = 0
        
        for symbol, position in positions.items():
            current_price = current_prices.get(symbol, position.get('entry_price', 0))
            entry_price = position.get('entry_price', 0)
            amount = position.get('amount', 0)
            
            if position['side'] == 'long':
                pnl = (current_price - entry_price) * amount
                pnl_pct = ((current_price / entry_price) - 1) * 100
            else:
                pnl = (entry_price - current_price) * amount
                pnl_pct = (1 - (current_price / entry_price)) * 100
            
            total_pnl += pnl
            
            emoji = "🟢" if pnl > 0 else "🔴" if pnl < 0 else "⚪"
            
            message_lines.append(
                f"{emoji} *{symbol}* ({position['side'].upper()})"
                f"\n   Entry: `${entry_price:.2f}`"
                f"\n   Current: `${current_price:.2f}`"
                f"\n   P&L: `${pnl:+.2f}` ({pnl_pct:+.1f}%)"
                f"\n   Stop: `${position.get('stop_loss', 0):.2f}`"
                f"\n   Target: `${position.get('take_profit', 0):.2f}`"
                f"\n"
            )
        
        message_lines.append(f"📈 *Total Unrealized P&L: ${total_pnl:+.2f}*")
        
        await update.message.reply_text("\n".join(message_lines), parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"❌ Positions error: {e}")
        await update.message.reply_text(f"❌ Error: {str(e)[:100]}", parse_mode='Markdown')

async def limit_buy(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Place limit buy order"""
    if not context.args or len(context.args) < 3:
        await update.message.reply_text(
            "Usage: `/limit_buy SYMBOL AMOUNT PRICE`\n"
            "Example: `/limit_buy SOL/USDC 2 122`",
            parse_mode='Markdown'
        )
        return
    
    symbol = context.args[0].upper()
    
    try:
        amount = float(context.args[1])
        price = float(context.args[2])
    except ValueError:
        await update.message.reply_text("❌ Amount and price must be numbers", parse_mode='Markdown')
        return
    
    # Get optional stop loss
    stop_loss = None
    if len(context.args) >= 4:
        try:
            stop_loss = float(context.args[3])
        except:
            pass
    
    await update.message.reply_text(
        f"📝 Placing limit BUY order...",
        parse_mode='Markdown'
    )
    
    try:
        from modules.trade_engine import trading_engine
        
        success, message = trading_engine.place_limit_order(
            symbol=symbol,
            side='buy',
            amount=amount,
            price=price
        )
        
        if success:
            response = (
                f"✅ *Limit BUY Order Placed!*\n\n"
                f"Symbol: {symbol}\n"
                f"Amount: {amount:.6f}\n"
                f"Price: `${price:.2f}`\n"
                f"Total: `${amount * price:.2f}`\n"
            )
            
            if stop_loss:
                response += f"Stop Loss: `${stop_loss:.2f}`\n"
            
            await update.message.reply_text(response, parse_mode='Markdown')
        else:
            await update.message.reply_text(
                f"❌ Failed: {message}",
                parse_mode='Markdown'
            )
            
    except Exception as e:
        logger.error(f"❌ Limit buy error: {e}")
        await update.message.reply_text(f"❌ Error: {str(e)[:150]}", parse_mode='Markdown')

async def limit_sell(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Place limit sell order"""
    if not context.args or len(context.args) < 3:
        await update.message.reply_text(
            "Usage: `/limit_sell SYMBOL AMOUNT PRICE`\n"
            "Example: `/limit_sell SOL/USDC 2 125`",
            parse_mode='Markdown'
        )
        return
    
    symbol = context.args[0].upper()
    
    try:
        amount = float(context.args[1])
        price = float(context.args[2])
    except ValueError:
        await update.message.reply_text("❌ Amount and price must be numbers", parse_mode='Markdown')
        return
    
    await update.message.reply_text(
        f"📝 Placing limit SELL order...",
        parse_mode='Markdown'
    )
    
    try:
        from modules.trade_engine import trading_engine
        
        success, message = trading_engine.place_limit_order(
            symbol=symbol,
            side='sell',
            amount=amount,
            price=price
        )
        
        if success:
            response = (
                f"✅ *Limit SELL Order Placed!*\n\n"
                f"Symbol: {symbol}\n"
                f"Amount: {amount:.6f}\n"
                f"Price: `${price:.2f}`\n"
                f"Total: `${amount * price:.2f}`\n"
            )
            
            await update.message.reply_text(response, parse_mode='Markdown')
        else:
            await update.message.reply_text(
                f"❌ Failed: {message}",
                parse_mode='Markdown'
            )
            
    except Exception as e:
        logger.error(f"❌ Limit sell error: {e}")
        await update.message.reply_text(f"❌ Error: {str(e)[:150]}", parse_mode='Markdown')

async def pending_orders(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show pending orders"""
    try:
        from modules.portfolio import load_portfolio
        
        portfolio = load_portfolio()
        orders = portfolio.get('pending_orders', [])
        
        if not orders:
            await update.message.reply_text("📭 No pending orders", parse_mode='Markdown')
            return
        
        message_lines = [f"📋 *Pending Orders ({len(orders)}):*\n"]
        
        for order in orders:
            emoji = "🟢" if order.get('side') == 'buy' else "🔴"
            side_text = "BUY" if order.get('side') == 'buy' else "SELL"
            
            message_lines.append(
                f"{emoji} *{order.get('symbol')} {side_text}*"
                f"\n   Amount: {order.get('amount', 0):.6f}"
                f"\n   Price: `${order.get('price', 0):.2f}`"
                f"\n   Total: `${order.get('amount', 0) * order.get('price', 0):.2f}`"
                f"\n"
            )
        
        await update.message.reply_text("\n".join(message_lines), parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"❌ Pending orders error: {e}")
        await update.message.reply_text(f"❌ Error: {str(e)[:100]}", parse_mode='Markdown')

async def cancel_all_orders(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Cancel all pending orders"""
    await update.message.reply_text("🗑️ Cancelling all pending orders...", parse_mode='Markdown')
    
    try:
        from modules.portfolio import load_portfolio, save_portfolio
        
        portfolio = load_portfolio()
        orders = portfolio.get('pending_orders', [])
        
        if not orders:
            await update.message.reply_text("📭 No orders to cancel", parse_mode='Markdown')
            return
        
        portfolio['pending_orders'] = []
        save_portfolio(portfolio)
        
        await update.message.reply_text(
            f"✅ Cancelled {len(orders)} order(s)",
            parse_mode='Markdown'
        )
        
    except Exception as e:
        logger.error(f"❌ Cancel error: {e}")
        await update.message.reply_text(f"❌ Error: {str(e)[:100]}", parse_mode='Markdown')

async def set_stop_loss(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Set stop loss for position"""
    if not context.args or len(context.args) < 2:
        await update.message.reply_text(
            "Usage: `/set_stop SYMBOL STOP_PRICE [TAKE_PROFIT]`\n"
            "Example: `/set_stop BTC/USDC 48000 52000`",
            parse_mode='Markdown'
        )
        return
    
    symbol = context.args[0].upper()
    
    try:
        stop_price = float(context.args[1])
        take_profit = float(context.args[2]) if len(context.args) > 2 else None
    except ValueError:
        await update.message.reply_text("❌ Invalid price", parse_mode='Markdown')
        return
    
    try:
        from modules.portfolio import load_portfolio, save_portfolio
        
        portfolio = load_portfolio()
        positions = portfolio.get('positions', {})
        
        if symbol not in positions:
            await update.message.reply_text(f"❌ No position for {symbol}", parse_mode='Markdown')
            return
        
        positions[symbol]['stop_loss'] = stop_price
        if take_profit:
            positions[symbol]['take_profit'] = take_profit
        
        save_portfolio(portfolio)
        
        response = f"✅ *Stop Loss Set for {symbol}*\n\n"
        response += f"Stop: `${stop_price:.2f}`\n"
        if take_profit:
            response += f"Target: `${take_profit:.2f}`\n"
        
        await update.message.reply_text(response, parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"❌ Set stop error: {e}")
        await update.message.reply_text(f"❌ Error: {str(e)[:80]}", parse_mode='Markdown')

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Help command"""
    help_text = """
🤖 *Trading Bot Commands*

*Basic Commands*
/start - Start bot
/status - Full status
/balance - Show balance
/positions - Show open positions
/help - This help

*Trading Signals*
/scan - Scan for signals
/execute_all - Execute all signals
/execute SYMBOL - Execute specific signal

*Manual Orders*
/limit_buy SYMBOL AMOUNT PRICE [STOP] - Limit buy
/limit_sell SYMBOL AMOUNT PRICE - Limit sell
/pending_orders - Show pending orders
/cancel_all - Cancel all orders

*Risk Management*
/set_stop SYMBOL STOP [TARGET] - Set stop loss
/stop - Stop bot

*Examples*
`/scan`
`/limit_buy BTC/USDC 0.001 50000 47500`
`/status`
    """
    
    await update.message.reply_text(help_text, parse_mode='Markdown')

async def stop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Stop the bot"""
    await update.message.reply_text("🛑 Stopping bot...", parse_mode='Markdown')
    raise SystemExit(0)

# -------------------------------------------------------------------
# MAIN FUNCTION
# -------------------------------------------------------------------
def run_telegram_bot():
    """Run Telegram bot"""
    global stop_event
    
    token = CONFIG.get('telegram_token')
    if not token:
        logger.error("❌ No Telegram token in config")
        return
    
    logger.info("🤖 Starting Telegram bot...")
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create application
    application = ApplicationBuilder().token(token).build()
    
    # Setup job queue
    try:
        job_queue = application.job_queue
        if job_queue:
            # Schedule jobs
            job_queue.run_repeating(trading_job_callback, interval=300, first=10)   # 5 min
            job_queue.run_repeating(portfolio_job_callback, interval=3600, first=30) # 1 hour
            job_queue.run_repeating(health_job_callback, interval=21600, first=60)  # 6 hours
            logger.info("✅ Scheduler jobs scheduled")
    except Exception as e:
        logger.warning(f"⚠️ Job queue setup failed: {e}")
    
    # Add command handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("balance", balance))
    application.add_handler(CommandHandler("status", status))
    application.add_handler(CommandHandler("positions", positions))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("scan", scan))
    application.add_handler(CommandHandler("execute_all", execute_all))
    application.add_handler(CommandHandler("execute", execute))
    application.add_handler(CommandHandler("limit_buy", limit_buy))
    application.add_handler(CommandHandler("limit_sell", limit_sell))
    application.add_handler(CommandHandler("pending_orders", pending_orders))
    application.add_handler(CommandHandler("cancel_all", cancel_all_orders))
    application.add_handler(CommandHandler("set_stop", set_stop_loss))
    application.add_handler(CommandHandler("stop", stop))
    
    logger.info("✅ Bot ready - starting polling...")
    
    try:
        application.run_polling(drop_pending_updates=True)
    except (KeyboardInterrupt, SystemExit):
        logger.info("🛑 Bot stopped")
    except Exception as e:
        logger.error(f"❌ Polling error: {e}")
    finally:
        logger.info("👋 Goodbye!")

if __name__ == "__main__":
    run_telegram_bot()