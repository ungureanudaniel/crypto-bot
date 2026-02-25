# services/telegram_bot.py - UPDATED FOR EXCHANGE-ONLY PORTFOLIO
import json
import sys
import logging
import signal
import os
import asyncio
from datetime import datetime
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
from services.scheduler import start_scheduler
import concurrent.futures

executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

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
# TRADE HISTORY (optional - for record keeping)
# -------------------------------------------------------------------
HISTORY_FILE = os.path.join(project_root, "trade_history.json")

def load_trade_history():
    """Load trade history from file (optional)"""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                return json.load(f)
        except:
            pass
    return []

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
        from modules.trade_engine import trading_engine
        
        current_positions = len(trading_engine.open_positions)
        
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
    except Exception as e:
        logger.error(f"❌ Health check error: {e}")

# -------------------------------------------------------------------
# TELEGRAM COMMANDS
# -------------------------------------------------------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Start command"""
    try:
        from modules.trade_engine import trading_engine
        
        # Get fresh summary from exchange
        summary = trading_engine.get_portfolio_summary()
        
        await update.message.reply_text(
            f"🤖 *Trading Bot Started!*\n\n"
            f"📊 Mode: `{summary['trading_mode'].upper()}`\n"
            f"💰 Portfolio: `${summary['portfolio_value']:,.2f}`\n"
            f"💵 Cash: `${summary['cash_balance']:,.2f}`\n"
            f"📈 Return: `{summary['total_return_pct']:+.1f}%`\n"
            f"🎯 Active: `{summary['active_positions']}/{trading_engine.max_positions}`\n\n"
            f"Use /help for commands",
            parse_mode='Markdown'
        )
    except Exception as e:
        logger.error(f"Start error: {e}")
        await update.message.reply_text("❌ Error starting bot")

async def balance(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show balance with debug info"""
    await update.message.reply_text("💰 Fetching balance...", parse_mode='Markdown')
    
    try:
        from modules.trade_engine import trading_engine
        
        # Debug: Check if trading_engine exists
        logger.info(f"🔍 Balance debug - trading_engine exists: {trading_engine is not None}")
        
        if not trading_engine:
            await update.message.reply_text("❌ Trading engine not initialized")
            return
        
        # Debug: Check binance_client
        logger.info(f"🔍 Balance debug - binance_client exists: {trading_engine.binance_client is not None}")
        
        if not trading_engine.binance_client:
            await update.message.reply_text("❌ Not connected to exchange")
            return
        
        # Try to get portfolio summary
        try:
            summary = trading_engine.get_portfolio_summary()
            logger.info(f"🔍 Balance debug - summary received: {summary is not None}")
        except Exception as e:
            logger.error(f"❌ Error in get_portfolio_summary: {e}")
            await update.message.reply_text(f"❌ Error getting summary: {str(e)[:100]}")
            return
        
        if not summary:
            await update.message.reply_text("❌ No portfolio data available")
            return
        
        # Format the response
        response = (
            f"💰 *Portfolio Balance*\n\n"
            f"Total Value: `${summary.get('portfolio_value', 0):,.2f}`\n"
            f"Cash: `${summary.get('cash_balance', 0):,.2f}`\n"
            f"Return: `{summary.get('total_return_pct', 0):+.1f}%`\n"
        )
        
        # Add holdings if any
        holdings = summary.get('holdings', {})
        if holdings and len(holdings) > 1:  # More than just USDT
            response += "\n📊 *Holdings:*\n"
            for asset, data in holdings.items():
                if asset != 'USDT':
                    if isinstance(data, dict):
                        amount = data.get('amount', 0)
                        value = data.get('value_usdt', 0)
                        if amount > 0:
                            response += f"   • {asset}: {amount:.4f} (${value:,.2f})\n"
        
        await update.message.reply_text(response, parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"❌ Balance error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        await update.message.reply_text(f"❌ Error: {str(e)[:100]}", parse_mode='Markdown')

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show detailed status"""
    try:
        from modules.trade_engine import trading_engine
        
        # Get fresh data from exchange
        summary = trading_engine.get_portfolio_summary()
        
        # Get trade history for win rate
        trade_history = load_trade_history()
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
        
        # Add open positions from engine
        if trading_engine.open_positions:
            message_lines.append(f"\n📊 *Active Positions:*")
            message_lines.append(f"━━━━━━━━━━━━━━━━")
            
            # Get current prices
            current_prices = trading_engine.get_current_prices()
            
            for symbol, position in trading_engine.open_positions.items():
                current_price = current_prices.get(symbol, position.get('entry_price', 0))
                entry_price = position.get('entry_price', 0)
                amount = position.get('amount', 0)
                
                if position['side'] == 'long':
                    pnl = (current_price - entry_price) * amount
                    pnl_pct = ((current_price / entry_price) - 1) * 100
                    pnl_emoji = "🟢" if pnl > 0 else "🔴" if pnl < 0 else "⚪"
                else:  # short
                    pnl = (entry_price - current_price) * amount
                    pnl_pct = (1 - (current_price / entry_price)) * 100
                    pnl_emoji = "🟢" if pnl > 0 else "🔴" if pnl < 0 else "⚪"
                
                message_lines.append(
                    f"\n{pnl_emoji} *{symbol}* ({position['side'].upper()})"
                    f"\n   Entry: `${entry_price:.2f}`"
                    f"\n   Current: `${current_price:.2f}`"
                    f"\n   P&L: `${pnl:+.2f}` ({pnl_pct:+.1f}%)"
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
        
        message_lines.append(f"\nUse `/executeall` to execute all signals")
        
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
    
    # DEBUG: Log what we're trying to execute
    logger.info("=" * 60)
    logger.info(f"🔍 EXECUTE_ALL: Attempting to execute {len(signals)} signals")
    
    for i, signal_data in enumerate(signals):
        logger.info(f"   Signal {i+1}: {signal_data['symbol']}")
        logger.info(f"      Signal dict keys: {signal_data.keys()}")
        if 'signal' in signal_data:
            logger.info(f"      Signal type: {signal_data['signal'].get('signal_type', 'unknown')}")
            logger.info(f"      Side: {signal_data['signal'].get('side', 'unknown')}")
            logger.info(f"      Entry: ${signal_data['signal'].get('entry', 0):.2f}")
            logger.info(f"      Units: {signal_data['signal'].get('units', 0):.6f}")
    
    await update.message.reply_text(
        f"⚡ Executing {len(signals)} signal(s)...",
        parse_mode='Markdown'
    )
    
    try:
        from modules.trade_engine import trading_engine
        
        executed = []
        failed = []
        fail_reasons = {}
        
        for signal_data in signals:
            symbol = signal_data['symbol']
            logger.info(f"🔍 Processing {symbol}...")
            
            try:
                # Check if we can execute this signal
                if 'signal' not in signal_data:
                    logger.error(f"❌ {symbol}: No 'signal' key in signal_data")
                    failed.append(symbol)
                    fail_reasons[symbol] = "Invalid signal format"
                    continue
                
                signal = signal_data['signal']
                
                # Validate required fields
                required_fields = ['side', 'entry', 'units', 'stop_loss', 'take_profit']
                missing_fields = [f for f in required_fields if f not in signal]
                if missing_fields:
                    logger.error(f"❌ {symbol}: Missing fields: {missing_fields}")
                    failed.append(symbol)
                    fail_reasons[symbol] = f"Missing fields: {missing_fields}"
                    continue
                
                # Check cash balance before executing
                cash = trading_engine.get_cash_balance()
                cost = signal['units'] * signal['entry']
                
                logger.info(f"   Cash: ${cash:.2f}, Cost: ${cost:.2f}")
                
                if cash < cost:
                    logger.warning(f"⚠️ {symbol}: Insufficient funds (need ${cost:.2f}, have ${cash:.2f})")
                    failed.append(symbol)
                    fail_reasons[symbol] = f"Insufficient funds (need ${cost:.2f})"
                    continue
                
                # Check minimum order value (Binance often requires $10)
                min_order = 10
                if cost < min_order:
                    logger.warning(f"⚠️ {symbol}: Order value ${cost:.2f} below minimum ${min_order}")
                    failed.append(symbol)
                    fail_reasons[symbol] = f"Order too small (min ${min_order})"
                    continue
                
                # Execute the signal
                logger.info(f"   Executing {signal['side']} signal for {symbol}")
                success = trading_engine.execute_signal(signal_data)
                
                if success:
                    executed.append(symbol)
                    logger.info(f"✅ Successfully executed {symbol}")
                else:
                    failed.append(symbol)
                    fail_reasons[symbol] = "Execution returned False"
                    logger.warning(f"❌ Failed to execute {symbol}")
                    
            except Exception as e:
                failed.append(symbol)
                fail_reasons[symbol] = str(e)[:50]
                logger.error(f"❌ Error executing {symbol}: {e}")
        
        # Clear pending signals
        context.user_data['pending_signals'] = []
        
        # Build detailed response
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
                reason = fail_reasons.get(sym, "Unknown")
                response.append(f"  • {sym}: `{reason}`")
        
        # Log summary
        logger.info("=" * 60)
        logger.info(f"📊 Execution Summary:")
        logger.info(f"   Executed: {len(executed)}")
        logger.info(f"   Failed: {len(failed)}")
        for sym, reason in fail_reasons.items():
            logger.info(f"      {sym}: {reason}")
        
        await update.message.reply_text("\n".join(response), parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"❌ Execute all error: {e}")
        import traceback
        logger.error(traceback.format_exc())
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
        from modules.trade_engine import trading_engine
        
        if not trading_engine.open_positions:
            await update.message.reply_text("📭 No open positions", parse_mode='Markdown')
            return
        
        current_prices = trading_engine.get_current_prices()
        
        message_lines = [f"📊 *Open Positions ({len(trading_engine.open_positions)}):*\n"]
        
        total_pnl = 0
        
        for symbol, position in trading_engine.open_positions.items():
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
            "Usage: `/limitbuy SYMBOL AMOUNT PRICE`\n"
            "Example: `/limitbuy SOL/USDC 2 122`",
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
    take_profit = None
    if len(context.args) >= 4:
        try:
            stop_loss = float(context.args[3])
        except:
            pass
    if len(context.args) >= 5:
        try:
            take_profit = float(context.args[4])
        except:
            pass
    
    await update.message.reply_text(
        f"📝 Placing limit BUY order...",
        parse_mode='Markdown'
    )
    
    try:
        from modules.trade_engine import trading_engine
        
        success, message = trading_engine.limit_buy(
            symbol=symbol,
            side='buy',
            amount=amount,
            price=price,
            stop_loss=stop_loss,
            take_profit=take_profit
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
            if take_profit:
                response += f"Take Profit: `${take_profit:.2f}`\n"
            
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
            "Usage: `/limitsell SYMBOL AMOUNT PRICE`\n"
            "Example: `/limitsell SOL/USDC 2 125`",
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
    """Show pending orders from exchange"""
    try:
        from modules.trade_engine import trading_engine
        
        if not trading_engine.binance_client:
            await update.message.reply_text("📭 No exchange connection", parse_mode='Markdown')
            return
        
        # Get open orders from exchange
        open_orders = []
        for symbol in CONFIG.get('coins', ['BTC/USDC', 'ETH/USDC']):
            try:
                binance_symbol = symbol.replace('/', '')
                orders = trading_engine.binance_client.get_open_orders(symbol=binance_symbol)
                for order in orders:
                    open_orders.append({
                        'symbol': symbol,
                        'order_id': order['orderId'],
                        'side': order['side'].lower(),
                        'amount': float(order['origQty']),
                        'price': float(order['price']),
                        'status': order['status']
                    })
            except Exception as e:
                logger.debug(f"Error fetching orders for {symbol}: {e}")
        
        if not open_orders:
            await update.message.reply_text("📭 No pending orders on exchange", parse_mode='Markdown')
            return
        
        message_lines = [f"📋 *Pending Orders ({len(open_orders)}):*\n"]
        
        for order in open_orders:
            emoji = "🟢" if order['side'] == 'buy' else "🔴"
            side_text = "BUY" if order['side'] == 'buy' else "SELL"
            
            message_lines.append(
                f"{emoji} *{order['symbol']} {side_text}*\n"
                f"   Amount: {order['amount']:.6f}\n"
                f"   Price: `${order['price']:.2f}`\n"
                f"   Total: `${order['amount'] * order['price']:.2f}`\n"
                f"   ID: `{order['order_id']}`\n"
            )
        
        await update.message.reply_text("\n".join(message_lines), parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"❌ Pending orders error: {e}")
        await update.message.reply_text(f"❌ Error: {str(e)[:100]}", parse_mode='Markdown')

async def cancel_all_orders(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Cancel all pending orders on exchange"""
    await update.message.reply_text("🗑️ Cancelling all pending orders on exchange...", parse_mode='Markdown')
    
    try:
        from modules.trade_engine import trading_engine
        
        if not trading_engine.binance_client:
            await update.message.reply_text("❌ No exchange connection", parse_mode='Markdown')
            return
        
        cancelled = 0
        failed = 0
        
        for symbol in CONFIG.get('coins', ['BTC/USDC', 'ETH/USDC']):
            try:
                binance_symbol = symbol.replace('/', '')
                result = trading_engine.binance_client.cancel_open_orders(symbol=binance_symbol)
                if result:
                    cancelled += len(result)
            except Exception as e:
                logger.debug(f"Error cancelling orders for {symbol}: {e}")
                failed += 1
        
        await update.message.reply_text(
            f"✅ Cancelled {cancelled} order(s) on exchange",
            parse_mode='Markdown'
        )
        
    except Exception as e:
        logger.error(f"❌ Cancel error: {e}")
        await update.message.reply_text(f"❌ Error: {str(e)[:100]}", parse_mode='Markdown')

async def set_stop_loss(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Set stop loss for position"""
    if not context.args or len(context.args) < 2:
        await update.message.reply_text(
            "Usage: `/setstop SYMBOL STOP_PRICE [TAKE_PROFIT]`\n"
            "Example: `/setstop BTC/USDC 48000 52000`",
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
        from modules.trade_engine import trading_engine
        
        # Check if position exists in engine
        if symbol not in trading_engine.open_positions:
            await update.message.reply_text(f"❌ No open position for {symbol}", parse_mode='Markdown')
            return
        
        # Update position in memory
        trading_engine.open_positions[symbol]['stop_loss'] = stop_price
        if take_profit:
            trading_engine.open_positions[symbol]['take_profit'] = take_profit
        
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
    <b>🤖 Trading Bot Commands</b>

    <b>Basic Commands</b>
    /start - Start bot
    /status - Full status
    /balance - Show balance
    /positions - Show open positions
    /help - This help

    <b>Trading Signals</b>
    /scan - Scan for signals
    /executeall - Execute all signals
    /execute SYMBOL - Execute specific signal

    <b>Manual Orders</b>
    /limitbuy SYMBOL AMOUNT PRICE [STOP] [TARGET] - Limit buy
    /limitsell SYMBOL AMOUNT PRICE - Limit sell
    /pendingorders - Show pending orders on exchange
    /cancelall - Cancel all orders on exchange

    <b>Risk Management</b>
    /setstop SYMBOL STOP [TARGET] - Set stop loss for position
    /stop - Stop bot

    <b>Examples</b>
    <code>/scan</code>
    <code>/limitbuy BTC/USDC 0.001 50000 47500 52500</code>
    <code>/status</code>
    """
    
    await update.message.reply_text(help_text, parse_mode='HTML')

async def stop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Stop the bot"""
    await update.message.reply_text("🛑 Stopping bot...", parse_mode='Markdown')
    raise SystemExit(0)

# -------------------------------------------------------------------
# MAIN FUNCTION
# -------------------------------------------------------------------
def run_telegram_bot():
    """Run Telegram bot - FIXED VERSION"""
    global stop_event
    
    token = CONFIG.get('telegram_token')
    if not token:
        logger.error("❌ No Telegram token in config")
        return
    
    logger.info("🤖 Starting Telegram bot...")
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create application WITHOUT job queue
    application = ApplicationBuilder().token(token).build()
    
    # IMPORTANT: DO NOT use application.job_queue!
    # Start our thread-based scheduler instead
    try:
        from services.scheduler import start_scheduler
        start_scheduler()  # This runs in a separate thread
        logger.info("✅ Thread-based scheduler started")
    except Exception as e:
        logger.error(f"❌ Failed to start scheduler: {e}")
    
    # Add command handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("balance", balance))
    application.add_handler(CommandHandler("status", status))
    application.add_handler(CommandHandler("positions", positions))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("scan", scan))
    application.add_handler(CommandHandler("executeall", execute_all))
    application.add_handler(CommandHandler("execute", execute))
    application.add_handler(CommandHandler("limitbuy", limit_buy))
    application.add_handler(CommandHandler("limitsell", limit_sell))
    application.add_handler(CommandHandler("pendingorders", pending_orders))
    application.add_handler(CommandHandler("cancelall", cancel_all_orders))
    application.add_handler(CommandHandler("setstop", set_stop_loss))
    application.add_handler(CommandHandler("stop", stop))
    
    logger.info(f"✅ Registered {len(application.handlers[0])} command handlers")
    logger.info("✅ Bot ready - starting polling...")
    
    try:
        # Run polling - this will now be responsive because scheduler is in another thread
        application.run_polling(
            drop_pending_updates=True,
            poll_interval=1.0,
            timeout=30
        )
    except (KeyboardInterrupt, SystemExit):
        logger.info("🛑 Bot stopped by user")
    except Exception as e:
        logger.error(f"❌ Polling error: {e}")
    finally:
        logger.info("👋 Goodbye!")
        try:
            from services.scheduler import stop_scheduler
            stop_scheduler()
        except:
            pass

if __name__ == "__main__":
    run_telegram_bot()