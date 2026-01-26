# services/telegram_bot.py - CLEAN VERSION
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
    return {"cash_balance": 10000}

# -------------------------------------------------------------------
# SCHEDULER JOBS - FIXED (async callbacks for run_repeating)
# -------------------------------------------------------------------
async def trading_job_callback(context: ContextTypes.DEFAULT_TYPE):
    """Simple trading job - async"""
    logger.info("⏰ Running trading job...")
    try:
        from modules.trade_engine import trading_engine
        trading_engine.check_stop_losses()
        logger.info("✅ Stop losses checked")
    except Exception as e:
        logger.error(f"❌ Trading job error: {e}")

async def portfolio_job_callback(context: ContextTypes.DEFAULT_TYPE):
    """Simple portfolio job - async"""
    logger.info("💰 Running portfolio job...")
    try:
        portfolio = load_portfolio()
        cash = portfolio.get('cash_balance', 0)
        logger.info(f"💰 Cash balance: ${cash:,.2f}")
    except Exception as e:
        logger.error(f"❌ Portfolio job error: {e}")

# -------------------------------------------------------------------
# TELEGRAM COMMANDS
# -------------------------------------------------------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    portfolio = load_portfolio()
    await update.message.reply_text(
        f"🤖 *Trading Bot Started!*\n\n"
        f"💰 Balance: ${portfolio.get('cash_balance', 0):,.2f}\n"
        f"⏰ Scheduler: Running",
        parse_mode='Markdown'
    )

async def balance(update: Update, context: ContextTypes.DEFAULT_TYPE):
    portfolio = load_portfolio()
    await update.message.reply_text(
        f"💰 Cash Balance: ${portfolio.get('cash_balance', 0):,.2f}",
        parse_mode='Markdown'
    )

def check_manual_stops(context: ContextTypes.DEFAULT_TYPE):
    """Check stop losses for manual trades - MUST BE REGULAR FUNCTION (not async)"""
    logger.info("🛡️ Checking manual stop losses...")
    
    try:
        from modules.portfolio import load_portfolio, save_portfolio
        from modules.data_feed import fetch_ohlcv
        from modules.trade_engine import trading_engine
        
        portfolio = load_portfolio()
        positions = portfolio.get('positions', {})
        
        for symbol, position in positions.items():
            # Check if position has stop loss
            if 'stop_loss' not in position:
                continue
                
            # Get current price
            df = fetch_ohlcv(symbol, "1m", limit=1)
            if df.empty:
                continue
                
            current_price = df.iloc[-1]['close']
            stop_loss = position['stop_loss']
            side = position.get('side', 'long')
            
            # Check stop loss
            if (side == 'long' and current_price <= stop_loss) or \
               (side == 'short' and current_price >= stop_loss):
                
                logger.info(f"🛑 Manual stop loss triggered: {symbol}")
                
                # Close position
                success = trading_engine.close_position(symbol, current_price, "stop_loss")
                
                if success:
                    # Send notification
                    try:
                        from services.notifier import notifier
                        notifier.send_message(
                            f"🛑 Stop Loss Executed\n"
                            f"Symbol: {symbol}\n"
                            f"Price: ${current_price:.2f}\n"
                            f"Stop: ${stop_loss:.2f}"
                        )
                    except:
                        pass
                        
    except Exception as e:
        logger.error(f"❌ Check manual stops error: {e}")

# -------------------------------------------------------------------
# LIMIT ORDER COMMANDS
# -------------------------------------------------------------------
async def limit_buy(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Place a limit buy order - FIXED VERSION"""
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
    
    await update.message.reply_text(
        f"📝 Placing limit BUY order...\n"
        f"Symbol: {symbol}\n"
        f"Amount: {amount}\n"
        f"Limit Price: ${price:.2f}",
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
            # SHORT response to avoid Telegram limits
            response = (
                f"✅ *Limit BUY Order Placed!*\n\n"
                f"Symbol: {symbol}\n"
                f"Amount: {amount}\n"
                f"Price: ${price:.2f}\n"
                f"Total: ${amount * price:.2f}\n\n"
            )
            
            # Truncate order ID if it's too long
            if len(message) > 50:
                response += f"Order ID: `{message[:30]}...`"
            else:
                response += f"Order ID: `{message}`"
            
            await update.message.reply_text(response, parse_mode='Markdown')
        else:
            # Truncate error message
            error_msg = str(message)[:200]
            await update.message.reply_text(
                f"❌ Failed to place order:\n{error_msg}",
                parse_mode='Markdown'
            )
            
    except Exception as e:
        logger.error(f"❌ Limit buy error: {e}")
        # Truncate error for Telegram
        error_msg = str(e)[:150]
        await update.message.reply_text(f"❌ Error: {error_msg}", parse_mode='Markdown')

async def limit_sell(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Place a limit sell order - FIXED VERSION"""
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
        f"📝 Placing limit SELL order...\n"
        f"Symbol: {symbol}\n"
        f"Amount: {amount}\n"
        f"Limit Price: ${price:.2f}",
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
            # SHORT response to avoid Telegram limits
            response = (
                f"✅ *Limit SELL Order Placed!*\n\n"
                f"Symbol: {symbol}\n"
                f"Amount: {amount}\n"
                f"Price: ${price:.2f}\n"
                f"Total: ${amount * price:.2f}\n\n"
            )
            
            # Truncate order ID if it's too long
            if len(message) > 50:
                response += f"Order ID: `{message[:30]}...`"
            else:
                response += f"Order ID: `{message}`"
            
            await update.message.reply_text(response, parse_mode='Markdown')
        else:
            # Truncate error message
            error_msg = str(message)[:200]
            await update.message.reply_text(
                f"❌ Failed to place order:\n{error_msg}",
                parse_mode='Markdown'
            )
            
    except Exception as e:
        logger.error(f"❌ Limit sell error: {e}")
        # Truncate error for Telegram
        error_msg = str(e)[:150]
        await update.message.reply_text(f"❌ Error: {error_msg}", parse_mode='Markdown')

async def scan_and_trade(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Manually trigger scan_and_trade function"""
    await update.message.reply_text("🔍 Running scan_and_trade...", parse_mode='Markdown')
    
    try:
        from modules.trade_engine import trading_engine
        
        # Run the scan
        signals_found = trading_engine.scan_and_trade()
        
        if not signals_found:
            await update.message.reply_text(
                "📭 No trading signals found or max positions reached.",
                parse_mode='Markdown'
            )
            return
        
        # Display signals found
        message_lines = [f"📊 *Found {len(signals_found)} Signal(s):*\n"]
        
        for signal_data in signals_found:
            symbol = signal_data['symbol']
            signal = signal_data['signal']
            
            message_lines.append(
                f"🔹 *{symbol}*\n"
                f"   Side: {signal['side'].upper()}\n"
                f"   Entry: ${signal['entry']:.2f}\n"
                f"   Stop Loss: ${signal['stop_loss']:.2f}\n"
                f"   Take Profit: ${signal['take_profit']:.2f}\n"
                f"   Units: {signal['units']:.6f}\n"
            )
        
        await update.message.reply_text("\n".join(message_lines), parse_mode='Markdown')
        
        # Ask if user wants to execute signals
        if signals_found:
            if context.user_data is None:
                context.user_data = {}
            context.user_data['pending_signals'] = signals_found
            await update.message.reply_text(
                "✅ Signals found! Use `/execute_all` to execute all signals or `/execute SYMBOL` to execute specific one.",
                parse_mode='Markdown'
            )
        
    except Exception as e:
        logger.error(f"❌ scan_and_trade error: {e}")
        await update.message.reply_text(f"❌ Error: {str(e)[:100]}", parse_mode='Markdown')

async def execute_all(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Execute all pending signals"""
    if not context.user_data or 'pending_signals' not in context.user_data or not context.user_data['pending_signals']:
        await update.message.reply_text("❌ No pending signals. Run `/scan_and_trade` first.", parse_mode='Markdown')
        return
    
    signals_found = context.user_data['pending_signals']
    await update.message.reply_text(f"⚡ Executing {len(signals_found)} signal(s)...", parse_mode='Markdown')
    
    try:
        from modules.trade_engine import trading_engine
        
        executed = []
        failed = []
        
        for signal_data in signals_found:
            symbol = signal_data['symbol']
            
            success = trading_engine.execute_signal(signal_data)
            if success:
                executed.append(symbol)
            else:
                failed.append(symbol)
        
        # Build response message
        message_lines = ["📊 *Execution Results:*\n"]
        
        if executed:
            message_lines.append(f"✅ *Executed:* {', '.join(executed)}")
        
        if failed:
            message_lines.append(f"❌ *Failed:* {', '.join(failed)}")
        
        # Clear pending signals
        context.user_data['pending_signals'] = []
        
        await update.message.reply_text("\n".join(message_lines), parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"❌ Execute all error: {e}")
        await update.message.reply_text(f"❌ Error: {str(e)[:100]}", parse_mode='Markdown')

async def execute_signal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Execute specific signal by symbol"""
    if not context.args:
        await update.message.reply_text(
            "Usage: `/execute SYMBOL`\nExample: `/execute BTC/USDC`",
            parse_mode='Markdown'
        )
        return
    
    symbol = context.args[0].upper()
    
    # Check if we have pending signals
    if not context.user_data or 'pending_signals' not in context.user_data:
        await update.message.reply_text(
            "❌ No pending signals. Run `/scan_and_trade` first.",
            parse_mode='Markdown'
        )
        return
    
    # Find the signal for this symbol
    signal_to_execute = None
    signals_found = context.user_data['pending_signals']
    
    for signal_data in signals_found:
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
            # Remove from pending signals
            if context.user_data:
                context.user_data['pending_signals'] = [
                    s for s in signals_found if s['symbol'].upper() != symbol
                ]
            
            await update.message.reply_text(
                f"✅ *Signal executed for {symbol}!*\n\n"
                f"Position opened successfully.",
                parse_mode='Markdown'
            )
        else:
            await update.message.reply_text(
                f"❌ Failed to execute signal for {symbol}",
                parse_mode='Markdown'
            )
            
    except Exception as e:
        logger.error(f"❌ Execute signal error: {e}")
        await update.message.reply_text(f"❌ Error: {str(e)[:100]}", parse_mode='Markdown')

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show bot status and current positions"""
    try:
        from modules.trade_engine import trading_engine
        
        # Get portfolio summary
        summary = trading_engine.get_portfolio_summary()
        
        # Load portfolio for positions
        from modules.portfolio import load_portfolio
        portfolio = load_portfolio()
        positions = portfolio.get('positions', {})
        
        # Build message
        message_lines = [
            f"🤖 *Trading Bot Status*\n",
            f"Mode: {summary['trading_mode'].upper()}",
            f"Portfolio Value: ${summary['portfolio_value']:,.2f}",
            f"Cash: ${summary['cash_balance']:,.2f}",
            f"Total Return: {summary['total_return_pct']:+.1f}%",
            f"Win Rate: {summary['win_rate']:.1f}%",
            f"Active Positions: {summary['active_positions']}/{trading_engine.max_positions}",
        ]
        
        if positions:
            message_lines.append(f"\n📊 *Active Positions:*")
            
            for symbol, position in positions.items():
                current_prices = trading_engine.get_current_prices()
                current_price = current_prices.get(symbol, position['entry_price'])
                
                pnl_pct = ((current_price - position['entry_price']) / position['entry_price'] * 100) if position['side'] == 'long' else 0
                
                message_lines.append(
                    f"\n🔹 *{symbol}* ({position['side'].upper()})"
                    f"\n   Entry: ${position['entry_price']:.2f}"
                    f"\n   Current: ${current_price:.2f}"
                    f"\n   P&L: {pnl_pct:+.1f}%"
                    f"\n   Stop: ${position.get('stop_loss', 'N/A')}"
                    f"\n   Target: ${position.get('take_profit', 'N/A')}"
                )
        
        await update.message.reply_text("\n".join(message_lines), parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"❌ Status error: {e}")
        await update.message.reply_text(f"❌ Error: {str(e)[:100]}", parse_mode='Markdown')

async def pending_orders(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show all pending orders"""
    try:
        from modules.portfolio import load_portfolio
        
        portfolio = load_portfolio()
        pending_orders = portfolio.get('pending_orders', [])
        
        if not pending_orders:
            await update.message.reply_text("📭 No pending orders", parse_mode='Markdown')
            return
        
        message_lines = [f"📋 *Pending Orders ({len(pending_orders)}):*\n"]
        
        for order in pending_orders:
            symbol = order.get('symbol', 'Unknown')
            side = order.get('side', 'buy')
            amount = order.get('amount', 0)
            price = order.get('price', 0)
            order_id = order.get('id', 'N/A')
            timestamp = order.get('timestamp', '')
            
            # Format timestamp
            if timestamp:
                try:
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    time_str = dt.strftime("%H:%M")
                except:
                    time_str = timestamp[:16]
            else:
                time_str = "N/A"
            
            # Get emoji
            side_emoji = "🟢" if side == 'buy' else "🔴"
            side_text = "BUY" if side == 'buy' else "SELL"
            
            message_lines.append(
                f"{side_emoji} *{symbol} {side_text}*\n"
                f"   Amount: {amount:.6f}\n"
                f"   Price: ${price:.2f}\n"
                f"   Total: ${amount * price:.2f}\n"
                f"   Time: {time_str}\n"
                f"   ID: `{order_id[:20]}...`"
            )
        
        full_message = "\n\n".join(message_lines)
        await update.message.reply_text(full_message, parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"❌ Pending orders error: {e}")
        await update.message.reply_text(f"❌ Error: {str(e)[:100]}", parse_mode='Markdown')

async def cancel_all_orders(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Cancel all pending orders"""
    await update.message.reply_text("🗑️ Cancelling ALL pending orders...", parse_mode='Markdown')
    
    try:
        from modules.portfolio import load_portfolio, save_portfolio
        
        portfolio = load_portfolio()
        pending_orders = portfolio.get('pending_orders', [])
        
        if not pending_orders:
            await update.message.reply_text("📭 No orders to cancel", parse_mode='Markdown')
            return
        
        order_count = len(pending_orders)
        
        # Clear all pending orders
        portfolio['pending_orders'] = []
        save_portfolio(portfolio)
        
        await update.message.reply_text(
            f"✅ *All Orders Cancelled!*\n"
            f"Cancelled {order_count} pending orders",
            parse_mode='Markdown'
        )
        
    except Exception as e:
        logger.error(f"❌ Cancel all orders error: {e}")
        await update.message.reply_text(f"❌ Error: {str(e)[:100]}", parse_mode='Markdown')

async def set_stop_loss(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Add stop loss to existing position"""
    if not context.args or len(context.args) < 3:
        await update.message.reply_text(
            "Usage: `/set_stop_loss SYMBOL STOP_PRICE [TAKE_PROFIT]`\n"
            "Example: `/set_stop_loss BTC/USDC 48000 52000`\n"
            "Example: `/set_stop_loss BTC/USDC 48000` (no take profit)",
            parse_mode='Markdown'
        )
        return
    
    symbol = context.args[0].upper()
    
    try:
        stop_price = float(context.args[1])
        take_profit = float(context.args[2]) if len(context.args) > 2 else None
    except ValueError:
        await update.message.reply_text("❌ Invalid prices", parse_mode='Markdown')
        return
    
    await update.message.reply_text(
        f"🛡️ Setting stop loss for {symbol}...",
        parse_mode='Markdown'
    )
    
    try:
        from modules.portfolio import load_portfolio, save_portfolio
        from modules.data_feed import fetch_ohlcv
        
        portfolio = load_portfolio()
        positions = portfolio.get('positions', {})
        
        if symbol not in positions:
            await update.message.reply_text(f"❌ No position for {symbol}", parse_mode='Markdown')
            return
        
        # Update position with stop loss
        position = positions[symbol]
        position['stop_loss'] = stop_price
        
        if take_profit:
            position['take_profit'] = take_profit
        
        # Save portfolio
        save_portfolio(portfolio)
        
        # Get current price
        df = fetch_ohlcv(symbol, "1m", limit=1)
        current_price = df.iloc[-1]['close'] if not df.empty else 0
        
        response = f"✅ *Stop Loss Set!*\n\n*{symbol}*\n"
        response += f"Current: ${current_price:.2f}\n"
        response += f"Stop Loss: ${stop_price:.2f}\n"
        
        if take_profit:
            response += f"Take Profit: ${take_profit:.2f}\n"
        
        # Calculate distance
        if current_price > 0:
            distance_pct = abs(current_price - stop_price) / current_price * 100
            response += f"Distance: {distance_pct:.1f}%"
        
        await update.message.reply_text(response, parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"❌ Set stop loss error: {e}")
        await update.message.reply_text(f"❌ *Error:* {str(e)[:80]}", parse_mode='Markdown')

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Help command with HTML formatting"""
    help_text = """
    <b>🤖 Trading Bot Commands</b>

    <b>Basic Commands</b>
    /start - Start bot
    /status - Check status  
    /balance - Show balance
    /stop - Stop bot
    /help - This help

    <b>Market Scanning</b>
    /scan - Scan for trading signals
    /scan_symbol SYMBOL - Detailed analysis

    <b>Limit Orders</b>
    /limit_buy SYMBOL AMOUNT PRICE - Limit buy
    /limit_sell SYMBOL AMOUNT PRICE - Limit sell
    /pending_orders - Show pending orders
    /cancel_all_orders - Cancel all

    <b>Portfolio & Trading</b>
    /trade SYMBOL BUY/SELL - Manual trade
    /positions - Show positions

    <b>Examples</b>
    /limit_buy BTC/USDC 0.001 50000
    /balance
    """
    
    await update.message.reply_text(help_text, parse_mode='HTML')

async def shutdown(application):
    """Gracefully shutdown the bot"""
    logger.info("🛑 Shutting down bot...")
    if application.updater.running:
        await application.updater.stop()
    await application.stop()
    await application.shutdown()
    logger.info("✅ Bot shutdown complete")

async def stop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Stop the bot gracefully"""
    await update.message.reply_text("🛑 Stopping bot...", parse_mode='Markdown')
    
    # Send a goodbye message
    await update.message.reply_text(
        "Bot is shutting down. Use /start to restart later.",
        parse_mode='Markdown'
    )
    
    # Stop the application
    context.application.stop_running = True
    
    # This will trigger the shutdown
    raise SystemExit(0)

# -------------------------------------------------------------------
# SINGLE MAIN FUNCTION - KEEP ONLY THIS ONE
# -------------------------------------------------------------------
def run_telegram_bot():
    """Run Telegram bot - SINGLE MAIN FUNCTION"""
    global stop_event
    token = CONFIG.get('telegram_token')
    if not token:
        logger.error("❌ No Telegram token")
        return
    
    logger.info("🤖 Starting bot...")
    
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
            job_queue.run_repeating(portfolio_job_callback, interval=1800, first=15) # 30 min
            # check_manual_stops set to non async
            # job_queue.run_repeating(check_manual_stops, interval=60, first=10)  # 1 min
            logger.info("✅ Scheduler jobs scheduled")
    except Exception as e:
        logger.warning(f"⚠️ Job queue setup failed: {e}")
        logger.info("ℹ️ Bot will run without scheduler")
    
    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("balance", balance))
    application.add_handler(CommandHandler("status", status))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("limit_buy", limit_buy))
    application.add_handler(CommandHandler("limit_sell", limit_sell))
    application.add_handler(CommandHandler("pending_orders", pending_orders))
    application.add_handler(CommandHandler("cancel_all_orders", cancel_all_orders))
    application.add_handler(CommandHandler("set_stop_loss", set_stop_loss))
    application.add_handler(CommandHandler("stop", stop))
    application.add_handler(CommandHandler("scan_and_trade", scan_and_trade))
    application.add_handler(CommandHandler("execute_all", execute_all))
    application.add_handler(CommandHandler("execute", execute_signal))
    
    logger.info("✅ Bot ready - starting polling...")
    
    try:
        # Run polling
        application.run_polling(
            drop_pending_updates=True,
            poll_interval=1.0
        )
    except (KeyboardInterrupt):
        logger.info("🛑 Received keyboard interrupt...")
    except (SystemExit):
        logger.info("🛑 Received system exit...")
    finally:
        logger.info("🛑 Stopping bot...")
        asyncio.run(shutdown(application))

if __name__ == "__main__":
    run_telegram_bot()