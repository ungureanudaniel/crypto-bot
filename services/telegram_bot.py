import json
import sys
import os
import threading

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Setup custom logging FIRST (before any other logging)
from modules.logger_config import setup_logging, log_trade

# Initialize logging with Telegram support
try:
    from services.notifier import notifier
    setup_logging(verbose=True, notifier=notifier)
    print("✅ Telegram bot logging initialized with Telegram support")
except ImportError:
    setup_logging(verbose=True)
    print("⚠️ Notifier not available - Telegram logging disabled")

import logging
import signal
import asyncio
import time
from datetime import datetime
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
from modules.trade_engine import trading_engine, save_positions_to_file
import concurrent.futures

stop_event = None

# Get logger for this module
logger = logging.getLogger(__name__)


_price_cache = {}
_price_cache_time = 0
CACHE_DURATION = 10  # seconds

def get_cached_prices():
    """Return cached current prices if fresh, otherwise fetch new ones."""
    global _price_cache, _price_cache_time
    now = time.time()
    if now - _price_cache_time > CACHE_DURATION:
        _price_cache = trading_engine.get_current_prices()
        _price_cache_time = now
    return _price_cache


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
            'trading_mode': os.environ.get('TRADING_MODE', 'paper'),
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
    if not update.message:
        logger.warning("⚠️ Start command triggered without message object")
        return
    try:
        from modules.portfolio import get_portfolio_summary
        summary = get_portfolio_summary(current_prices=get_cached_prices())
        
        await update.message.reply_text(
            f"🤖 *Trading Bot Started!*\n\n"
            f"📊 Mode: `{CONFIG.get('trading_mode', 'unknown').upper()}`\n"
            f"💰 Portfolio: `${summary.get('total_value', 0):,.2f}`\n"  # ← FIXED
            f"💵 Cash: `${summary.get('cash', {}).get('total', 0):,.2f}`\n"  # ← FIXED
            f"📈 Return: `{summary.get('total_return_pct', 0):+.1f}%`\n"
            f"🎯 Active: `{summary.get('positions_count', 0)}/{trading_engine.max_positions}`\n\n"
            f"Use /help for commands",
            parse_mode='Markdown'
        )
    except Exception as e:
        logger.error(f"Start error: {e}")
        await update.message.reply_text("❌ Error starting bot")

async def reset_circuit_breaker(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message:
        logger.warning("⚠️ Balance command triggered without message object")
        return
    await update.message.reply_text("💰 Fetching balance...", parse_mode='Markdown')

    trading_engine.circuit_breaker_triggered = False
    await update.message.reply_text("✅ Circuit breaker manually reset. Trading resumed.", parse_mode='Markdown')

async def balance(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show balance with debug info"""
    if not update.message:
        logger.warning("⚠️ Balance command triggered without message object")
        return
    await update.message.reply_text("💰 Fetching balance...", parse_mode='Markdown')
    
    try:
        logger.info("  ... importing trading_engine for balance command")
        
        # Debug: Check if trading_engine exists
        logger.info("🔍 BALANCE DEBUG - Starting balance check")
        logger.info(f"   trading_engine exists: {trading_engine is not None}")
        logger.info(f"   trading mode: {trading_engine.trading_mode}")
        
        if not trading_engine:
            await update.message.reply_text("❌ Trading engine not initialized")
            return
        
        # Check binance_client
        logger.info(f"   binance_client exists: {trading_engine.binance_client is not None}")
        
        if trading_engine.trading_mode == 'paper':
            try:
                from modules.portfolio import get_portfolio_summary
                summary = get_portfolio_summary(current_prices=get_cached_prices())
                perf = summary

                total_value = summary.get('total_value', 0)
                cash_total = summary.get('total_cash', 0)
                total_return_pct = summary.get('total_return_pct', 0)
                positions = summary.get('positions', {})

                response = (
                    f"💰 *Portfolio Balance*\n\n"
                    f"Total Value: `${total_value:,.2f}`\n"
                    f"Cash: `${cash_total:,.2f}`\n"
                    f"Return: `{total_return_pct:+.1f}%`\n"
                    f"Win Rate: `{summary.get('win_rate', 0):.1f}%`\n"
                    f"Trades: `{summary.get('total_trades', 0)}`\n"
                )

                if positions:
                    response += "\n📊 *Positions:*\n"
                    for symbol, pos in positions.items():
                        pnl = pos.get('pnl', 0)
                        pnl_emoji = "🟢" if pnl > 0 else "🔴" if pnl < 0 else "⚪"
                        response += (
                            f"   {pnl_emoji} {symbol}: {pos.get('amount', 0):.4f} "
                            f"@ ${pos.get('entry_price', 0):.2f} → "
                            f"${pos.get('current_price', 0):.2f} "
                            f"(PnL: ${pnl:+.2f})\n"
                        )

                await update.message.reply_text(response, parse_mode='Markdown')
                return

            except Exception as e:
                logger.error(f"❌ Paper balance error: {e}")
                await update.message.reply_text(f"❌ Error: {str(e)[:100]}", parse_mode='Markdown')
                return

        if not trading_engine.binance_client:
            await update.message.reply_text("❌ Not connected to exchange")
            return
        
        # LIVE MODE - use get_portfolio_summary from portfolio.py
        try:
            from modules.portfolio import get_portfolio_summary
            summary = get_portfolio_summary(current_prices=get_cached_prices())
            logger.info(f"   get_portfolio_summary() returned: {summary is not None}")
        except Exception as e:
            logger.error(f"   ❌ get_portfolio_summary() failed: {e}")
            summary = None
        
        if not summary:
            await update.message.reply_text("❌ Could not get portfolio summary")
            return
        
        # Format the response
        response = (
            f"💰 *Portfolio Balance*\n\n"
            f"Total Value: `${summary.get('total_value', 0):,.2f}`\n"
            f"Cash: `${summary.get('total_cash', 0):,.2f}`\n"
            f"Return: `{summary.get('total_return_pct', 0):+.1f}%`\n"
            f"Win Rate: `{summary.get('win_rate', 0):.1f}%`\n"
        )
        
        # Add positions if any
        positions = summary.get('positions', {})
        if positions:
            response += "\n📊 *Positions:*\n"
            for symbol, pos in positions.items():
                pnl = pos.get('pnl', 0)
                pnl_emoji = "🟢" if pnl > 0 else "🔴" if pnl < 0 else "⚪"
                response += f"   {pnl_emoji} {symbol}: {pos.get('amount', 0):.4f} @ ${pos.get('entry_price', 0):.2f} → ${pos.get('current_price', 0):.2f} (PnL: ${pnl:+.2f})\n"
        
        await update.message.reply_text(response, parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"❌ Balance error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        await update.message.reply_text(f"❌ Error: {str(e)[:100]}", parse_mode='Markdown')

async def summary(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Quick portfolio summary - works in both paper and live mode"""
    if not update.message:
        logger.warning("⚠️ Summary command triggered without message object")
        return
    try:
        from modules.portfolio import get_portfolio_summary
        summary_data = get_portfolio_summary(current_prices=get_cached_prices())

        total_value = summary_data.get('total_value', 0)
        total_cash = summary_data.get('total_cash', 0)
        total_return_pct = summary_data.get('total_return_pct', 0)
        pnl_emoji = "🟢" if total_return_pct >= 0 else "🔴"

        message = (
            f"{pnl_emoji} *Portfolio*: `${total_value:,.0f}` "
            f"({total_return_pct:+.1f}%) | "
            f"💵 Cash: `${total_cash:,.0f}` | "
            f"📊 {summary_data.get('positions_count', 0)} positions | "
            f"🎯 {summary_data.get('win_rate', 0):.0f}% win rate"
        )

        await update.message.reply_text(message, parse_mode='Markdown')

    except Exception as e:
        logger.error(f"❌ Summary error: {e}")
        await update.message.reply_text("❌ Error getting summary")

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show detailed status"""
    if not update.message:
        logger.warning("⚠️ Summary command triggered without message object")
        return
    try:
        from modules.portfolio import get_portfolio_summary
        
        # Get fresh data
        summary = get_portfolio_summary(current_prices=get_cached_prices())
        
        # Get trade history for win rate
        from modules.portfolio import load_portfolio
        portfolio = load_portfolio()
        trade_history = portfolio.get('trade_history', [])
        closed_trades = [t for t in trade_history if t.get('action') == 'close']
        winning_trades = [t for t in closed_trades if t.get('pnl', 0) > 0]
        win_rate = (len(winning_trades) / len(closed_trades) * 100) if closed_trades else 0
        
        # Build message
        message_lines = [
            f"🤖 *Trading Bot Status*\n",
            f"━━━━━━━━━━━━━━━━",
            f"📊 Mode: `{summary.get('trading_mode', 'PAPER')}`",
            f"💰 Portfolio: `${summary.get('total_value', 0):,.2f}`",
            f"💵 Cash: `${summary.get('total_cash', 0):,.2f}`",  # ← This shows your $47
            f"📈 Return: `{summary.get('total_return', 0):+.1f}%`",
            f"🎯 Win Rate: `{win_rate:.1f}%`",
            f"📊 Active: `{summary.get('positions_count', 0)}/{trading_engine.max_positions}`",
            f"📋 Total Trades: `{len(closed_trades)}`",
        ]
        
        # Add positions
        positions = summary.get('positions', {})
        if positions:
            message_lines.append(f"\n📊 *Active Positions:*")
            message_lines.append(f"━━━━━━━━━━━━━━━━")
            
            for symbol, position in positions.items():
                current_price = position.get('current_price', position.get('entry_price', 0))
                entry_price = position.get('entry_price', 0)
                amount = position.get('amount', 0)
                pnl = position.get('pnl', 0)
                pnl_pct = position.get('pnl_pct', 0)

                pnl_emoji = "🟢" if pnl > 0 else "🔴" if pnl < 0 else "⚪"

                # Smart price formatting — enough decimals for any coin price
                def fmt(p):
                    if p == 0: return "0"
                    if p >= 100:  return f"{p:.2f}"
                    if p >= 1:    return f"{p:.3f}"
                    if p >= 0.01: return f"{p:.4f}"
                    return f"{p:.6f}"

                trailing = " 🔄" if position.get('trailing_stop_active') else ""
                message_lines.append(
                    f"\n{pnl_emoji} *{symbol}* ({position.get('side', 'unknown').upper()})"
                    f"\n   Entry: `${fmt(entry_price)}`"
                    f"\n   Current: `${fmt(current_price)}`"
                    f"\n   P&L: `${pnl:+.4f}` ({pnl_pct:+.2f}%)"
                    f"\n   Stop: `${fmt(position.get('stop_loss', 0))}`{trailing}"
                    f"\n   Target: `${fmt(position.get('take_profit', 0))}`"
                )
        
        await update.message.reply_text("\n".join(message_lines), parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"❌ Status error: {e}")
        await update.message.reply_text(f"❌ Error: {str(e)[:100]}", parse_mode='Markdown')

async def scan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Manually trigger scan for signals"""
    if not update.message:
        logger.warning("⚠️ Start command triggered without message object")
        return
    await update.message.reply_text("🔍 Scanning for trading signals...", parse_mode='Markdown')
    
    try:
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
    if not update.message:
        logger.warning("⚠️ Start command triggered without message object")
        return

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

async def emergency_sell_all(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """EMERGENCY: Sell all positions at market price"""
    if not update.message:
        logger.warning("⚠️ Start command triggered without message object")
        return
    await update.message.reply_text("🚨 *EMERGENCY SELL ALL* 🚨\n\nSelling all positions...", parse_mode='Markdown')
    
    try:
        if not trading_engine.open_positions:
            await update.message.reply_text("📭 No open positions to sell", parse_mode='Markdown')
            return
        
        sold = []
        failed = []
        
        for symbol, position in list(trading_engine.open_positions.items()):
            try:
                # Get current price
                current_price = get_cached_prices().get(symbol)
                if not current_price:
                    failed.append(f"{symbol} (no price)")
                    continue
                
                # Close position
                success = trading_engine.close_position(symbol, current_price, "emergency_sell")
                if success:
                    sold.append(symbol)
                else:
                    failed.append(symbol)
                    
            except Exception as e:
                failed.append(f"{symbol} ({str(e)[:20]})")
        
        # Send summary
        message = "📊 *Emergency Sell Complete*\n\n"
        if sold:
            message += f"✅ Sold: {', '.join(sold)}\n"
        if failed:
            message += f"❌ Failed: {', '.join(failed)}"
        
        await update.message.reply_text(message, parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"Emergency sell error: {e}")
        await update.message.reply_text(f"❌ Error: {str(e)[:100]}", parse_mode='Markdown')

# -------------------------------------------------------------------
async def execute(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Execute specific signal"""
    if not update.message:
        logger.warning("⚠️ Start command triggered without message object")
        return
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


async def holdings(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show current holdings from portfolio"""
    if not update.message:
        logger.warning("⚠️ Start command triggered without message object")
        return
    try:
        from modules.portfolio import load_portfolio
        portfolio = load_portfolio()
        holdings = portfolio.get('holdings', {})
        
        if not holdings or all(asset == 'USDT' for asset in holdings):
            await update.message.reply_text("📭 No holdings", parse_mode='Markdown')
            return
        
        # Get current prices
        current_prices = get_cached_prices()
        
        message_lines = [f"📊 *Current Holdings:*\n"]
        
        total_value = 0
        cash = portfolio.get('cash_balance', 0)
        
        for asset, amount in holdings.items():
            if asset == 'USDT':
                continue
            symbol = f"{asset}/USDT"
            price = current_prices.get(symbol, 0)
            value = amount * price
            total_value += value
            
            message_lines.append(
                f"• *{asset}*: {amount:.4f} @ ${price:.2f} = ${value:.2f}\n"
            )
        
        message_lines.append(f"\n💰 *Holdings Value: ${total_value:.2f}*")
        message_lines.append(f"💵 *Cash: ${cash:.2f}*")
        message_lines.append(f"📊 *Total Portfolio: ${total_value + cash:.2f}*")
        
        await update.message.reply_text("\n".join(message_lines), parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"❌ Holdings error: {e}")
        await update.message.reply_text(f"❌ Error: {str(e)[:100]}", parse_mode='Markdown')

async def positions(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show active positions (trades with stop/target)"""
    if not update.message:
        logger.warning("⚠️ Start command triggered without message object")
        return
    try:        
        if not trading_engine.open_positions:
            await update.message.reply_text("📭 No active positions", parse_mode='Markdown')
            return
        
        current_prices = get_cached_prices()
        
        message_lines = [f"📊 *Active Positions ({len(trading_engine.open_positions)}):*\n"]
        
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
    """Place manual limit buy order"""
    if not update.message:
        logger.warning("⚠️ Start command triggered without message object")
        return
    if not context.args or len(context.args) < 3:
        await update.message.reply_text(
            "Usage: `/limitbuy SYMBOL AMOUNT PRICE`\n"
            "Example: `/limitbuy BTC/USDT 0.001 50000`\n"
            "         `/limitbuy SOL/USDT 2 122`",
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
    
    # Get optional stop loss and take profit
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
        f"📝 Placing manual limit BUY order...",
        parse_mode='Markdown'
    )
    
    try:
        success, message = trading_engine.place_manual_limit_order(
            symbol=symbol,
            side='buy',
            quantity=amount,
            price=price,
            stop_loss=stop_loss if stop_loss is not None else price * 0.98,
            take_profit=take_profit if take_profit is not None else price * 1.05
        )
        
        if success:
            response = (
                f"✅ *Manual Limit BUY Order Placed!*\n\n"
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
    """Place manual limit sell order"""
    if not update.message:
        logger.warning("⚠️ Start command triggered without message object")
        return
    if not context.args or len(context.args) < 3:
        await update.message.reply_text(
            "Usage: `/limitsell SYMBOL AMOUNT PRICE`\n"
            "Example: `/limitsell BTC/USDT 0.001 55000`\n"
            "         `/limitsell SOL/USDT 2 130`",
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
    
    # Get optional stop loss and take profit
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
        f"📝 Placing manual limit SELL order...",
        parse_mode='Markdown'
    )
    
    try:
        success, message = trading_engine.place_manual_limit_order(
            symbol=symbol,
            side='sell',
            quantity=amount,
            price=price,
            stop_loss=stop_loss if stop_loss is not None else price * 1.02,
            take_profit=take_profit if take_profit is not None else price * 0.95
        )
        
        if success:
            response = (
                f"✅ *Manual Limit SELL Order Placed!*\n\n"
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
        logger.error(f"❌ Limit sell error: {e}")
        await update.message.reply_text(f"❌ Error: {str(e)[:150]}", parse_mode='Markdown')

async def current_price(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Get current price for symbol"""
    if not update.message:
        logger.warning("⚠️ Start command triggered without message object")
        return
    if not context.args:
        await update.message.reply_text(
            "Usage: `/price SYMBOL`\nExample: `/price BTC/USDC`",
            parse_mode='Markdown'
        )
        return
    
    symbol = context.args[0].upper()
    
    try:
        price = get_cached_prices().get(symbol)
        
        if price:
            await update.message.reply_text(
                f"💰 Current price of {symbol} is `${price:.2f}`",
                parse_mode='Markdown'
            )
        else:
            await update.message.reply_text(
                f"❌ Could not fetch price for {symbol}",
                parse_mode='Markdown'
            )
            
    except Exception as e:
        logger.error(f"❌ Price error: {e}")
        await update.message.reply_text(f"❌ Error: {str(e)[:100]}", parse_mode='Markdown')

async def pending_orders(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show pending orders from exchange"""
    if not update.message:
        logger.warning("⚠️ Start command triggered without message object")
        return
    try:
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
    if not update.message:
        logger.warning("⚠️ Start command triggered without message object")
        return
    await update.message.reply_text("🗑️ Cancelling all pending orders on exchange...", parse_mode='Markdown')
    
    try:
        if not trading_engine.binance_client:
            await update.message.reply_text("❌ No exchange connection", parse_mode='Markdown')
            return
        
        cancelled = 0
        failed = 0
        failed_symbols = []
        
        for symbol in CONFIG.get('coins', ['BTC/USDC', 'ETH/USDC']):
            try:
                binance_symbol = symbol.replace('/', '')
                
                # First, get open orders for this symbol
                open_orders = trading_engine.binance_client.get_open_orders(symbol=binance_symbol)
                
                if not open_orders:
                    logger.debug(f"No open orders for {symbol}")
                    continue
                
                # Cancel each order individually
                for order in open_orders:
                    try:
                        result = trading_engine.binance_client.cancel_order(
                            symbol=binance_symbol,
                            orderId=order['orderId']
                        )
                        if result:
                            cancelled += 1
                            logger.info(f"✅ Cancelled order {order['orderId']} for {symbol}")
                    except Exception as e:
                        logger.debug(f"Failed to cancel order {order['orderId']}: {e}")
                        failed += 1
                
                # Small delay to avoid rate limits
                time.sleep(0.1)
                
            except Exception as e:
                logger.debug(f"Error processing orders for {symbol}: {e}")
                failed += 1
                failed_symbols.append(symbol)
        
        # Prepare response
        if cancelled > 0:
            response = f"✅ Cancelled {cancelled} order(s) on exchange"
            if failed > 0:
                response += f"\n⚠️ Failed to cancel {failed} order(s)"
            if failed_symbols:
                response += f"\n   Symbols with issues: {', '.join(failed_symbols[:3])}"
        else:
            response = "📭 No orders to cancel"
        
        await update.message.reply_text(response, parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"❌ Cancel error: {e}")
        await update.message.reply_text(f"❌ Error: {str(e)[:100]}", parse_mode='Markdown')

async def set_stop_loss(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Set stop loss for position"""
    if not update.message:
        logger.warning("⚠️ Start command triggered without message object")
        return
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
        # Check if position exists in engine
        if symbol not in trading_engine.open_positions:
            await update.message.reply_text(f"❌ No open position for {symbol}", parse_mode='Markdown')
            return
        
        # Update position in memory
        trading_engine.open_positions[symbol]['stop_loss'] = stop_price
        if take_profit:
            trading_engine.open_positions[symbol]['take_profit'] = take_profit
        save_positions_to_file(trading_engine.open_positions)
        
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
    if not update.message:
        logger.warning("⚠️ Start command triggered without message object")
        return
    help_text = """
    <b>🤖 Trading Bot Commands</b>

    <b>Basic Commands</b>
    /start - Start bot
    /status - Full status
    /balance - Show balance
    /summary - Quick portfolio summary
    /positions - Show open positions
    /holdings - Show current holdings
    /price SYMBOL - Show current price of a symbol
    /help - This help

    <b>Trading Signals</b>
    /scan - Scan for signals
    /executeall - Execute all signals
    /execute SYMBOL - Execute specific signal

    <b>Manual Orders</b>
    /limitbuy SYMBOL AMOUNT PRICE [STOP] [TARGET] - Limit buy
    /limitsell SYMBOL AMOUNT PRICE - Limit sell
    /pendingorders - Show pending orders on exchange
    /sellall - EMERGENCY: Sell all positions at market price
    /cancelall - Cancel all orders on exchange

    <b>Risk Management</b>
    /setstop SYMBOL STOP [TARGET] - Set stop loss for position
    /stop - Stop bot
    /resetcircuitbreaker - Reset circuit breaker

    <b>Examples</b>
    <code>/scan</code>
    <code>/limitbuy BTC/USDC 0.001 50000 47500 52500</code>
    <code>/status</code>
    """
    
    await update.message.reply_text(help_text, parse_mode='HTML')

async def stop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Stop the bot gracefully"""
    if not update.message:
        logger.warning("⚠️ Start command triggered without message object")
        return
    await update.message.reply_text("🛑 Stopping bot...", parse_mode='Markdown')
    logger.info("🛑 Stop command received - shutting down gracefully")
    
    # Set stop event
    global stop_event
    if stop_event:
        stop_event.set()
    
    # Stop scheduler
    try:
        from services.scheduler import stop_scheduler
        stop_scheduler()
    except Exception as e:
        logger.error(f"Error stopping scheduler: {e}")
    
    # Stop the application properly
    try:
        # Only stop if running
        if context.application.running:
            await context.application.stop()
            logger.info("✅ Application stopped")
        else:
            logger.info("Application already stopped")
    except Exception as e:
        logger.error(f"Error stopping application: {e}")
    
    # Send final message
    await update.message.reply_text("✅ Bot stopped. Use /start to restart.", parse_mode='Markdown') 

# -------------------------------------------------------------------
# MAIN FUNCTION
# -------------------------------------------------------------------
async def run_telegram_bot_async():
    """Async version of the Telegram bot runner"""
    global stop_event
    stop_event = threading.Event()  # Create the event

    
    token = CONFIG.get('telegram_token')
    if not token:
        logger.error("❌ No Telegram token in config")
        return
    
    # Initialize notifier with credentials
    try:
        from services.notifier import init_notifier
        chat_id = CONFIG.get('telegram_chat_id')
        if chat_id:
            init_notifier(token, chat_id)
            logger.info("✅ Telegram notifier initialized")
    except Exception as e:
        logger.warning(f"⚠️ Could not initialize notifier: {e}")
    
    logger.info("🤖 Starting Telegram bot...")
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create application WITHOUT job queue
    application = ApplicationBuilder().token(token).build()
    
    # Start thread-based scheduler 
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
    application.add_handler(CommandHandler("price", current_price))
    application.add_handler(CommandHandler("holdings", holdings))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("scan", scan))
    application.add_handler(CommandHandler("executeall", execute_all))
    application.add_handler(CommandHandler("execute", execute))
    application.add_handler(CommandHandler("limitbuy", limit_buy))
    application.add_handler(CommandHandler("limitsell", limit_sell))
    application.add_handler(CommandHandler("pendingorders", pending_orders))
    application.add_handler(CommandHandler("sellall", emergency_sell_all))
    application.add_handler(CommandHandler("cancelall", cancel_all_orders))
    application.add_handler(CommandHandler("summary", summary))
    application.add_handler(CommandHandler("setstop", set_stop_loss))
    application.add_handler(CommandHandler("stop", stop))
    application.add_handler(CommandHandler("resetcircuitbreaker", reset_circuit_breaker))
    
    logger.info(f"✅ Registered {len(application.handlers[0])} command handlers")
    logger.info("✅ Bot ready - starting polling...")
    
    try:
        # Run polling - this is async, so we await it
        await application.initialize()
        await application.start()
        
        # Start polling
        await application.updater.start_polling(
            drop_pending_updates=True,
            poll_interval=1.0,
            timeout=40
        )
        
        # Keep running until stop event
        while not stop_event or not stop_event.is_set():
            await asyncio.sleep(1)
            
    except (KeyboardInterrupt, SystemExit):
        logger.info("🛑 Bot stopped by user")
    except Exception as e:
        logger.error(f"❌ Polling error: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        # Clean shutdown
        try:
            # Only stop if running
            if application.running:
                await application.stop()
                logger.info("✅ Application stopped")
            await application.shutdown()
        except Exception as e:
            logger.debug(f"Shutdown error (normal during stop): {e}")
        
        logger.info("👋 Goodbye!")
        try:
            from services.scheduler import stop_scheduler
            stop_scheduler()
        except:
            pass



def run_telegram_bot():
    """Synchronous wrapper to run the async bot"""
    # Fix asyncio event loop for Windows
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    # Create a new event loop for this thread
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        loop.run_until_complete(run_telegram_bot_async())
    except KeyboardInterrupt:
        logger.info("🛑 Bot stopped by user")
    except Exception as e:
        logger.error(f"❌ Bot error: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        loop.close()

if __name__ == "__main__":
    run_telegram_bot()
