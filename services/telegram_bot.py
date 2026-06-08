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
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
from modules.trade_engine import trading_engine, save_positions_to_file
import concurrent.futures

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
HISTORY_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "history.json")

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
    logger.info(f"Received /start from {update.effective_user.id}")

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
    """Show REAL balance from exchange (not paper portfolio)"""
    logger.info(f"Received /balance from {update.effective_user.id}")

    if not update.message:
        logger.warning("⚠️ Balance command triggered without message object")
        return
    
    await update.message.reply_text("💰 Fetching live exchange balance...", parse_mode='Markdown')
    
    try:
        from config_loader import get_binance_client
        from modules.portfolio import get_portfolio_summary
        
        trading_mode = trading_engine.trading_mode if trading_engine else 'paper'
        
        # PAPER MODE - use portfolio.json
        if trading_mode == 'paper':
            summary = get_portfolio_summary(current_prices=get_cached_prices())
            response = (
                f"💰 *Paper Portfolio Balance*\n\n"
                f"Total Value: `${summary.get('total_value', 0):,.2f}`\n"
                f"Cash: `${summary.get('total_cash', 0):,.2f}`\n"
                f"Return: `{summary.get('total_return_pct', 0):+.1f}%`"
            )
        else:
            client = get_binance_client()
            if not client:
                await update.message.reply_text("❌ Not connected to exchange")
                return

            account = client.get_account()

            # Just show everything with a non-zero balance — no pairing needed
            usdc = 0.0
            usdt = 0.0
            assets = []

            for b in account['balances']:
                free   = float(b['free'])
                locked = float(b['locked'])
                total  = free + locked
                if total < 0.0001:
                    continue

                if b['asset'] == 'USDC':
                    usdc = total
                elif b['asset'] == 'USDT':
                    usdt = total
                else:
                    assets.append({
                        'asset':  b['asset'],
                        'free':   free,
                        'locked': locked,
                        'total':  total,
                    })

            assets.sort(key=lambda x: x['total'], reverse=True)

            from modules.portfolio import load_portfolio
            portfolio  = load_portfolio()
            initial    = portfolio.get('initial_balance', usdc + usdt)
            return_pct = ((usdc + usdt - initial) / initial * 100) if initial > 0 else 0.0

            response = (
                f"💰 *Exchange Balance* [{trading_mode.upper()}]\n\n"
                f"💵 USDC: `{usdc:.4f}`\n"
            )
            if usdt > 0.0001:
                response += f"💵 USDT: `{usdt:.4f}`\n"

            if assets:
                response += f"\n🪙 *All Assets:*\n"
                for a in assets:
                    line = f"   • {a['asset']}: `{a['total']:.6f}`"
                    if a['locked'] > 0:
                        line += f" (locked: `{a['locked']:.6f}`)"
                    response += line + "\n"

            response += f"\n📊 Stablecoin Return: `{return_pct:+.1f}%` vs initial `${initial:,.2f}`"

            await update.message.reply_text(response, parse_mode='Markdown')

        
    except Exception as e:
        logger.error(f"❌ Balance error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        await update.message.reply_text(f"❌ Error: {str(e)[:100]}", parse_mode='Markdown')


async def summary(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Quick portfolio summary with advanced risk metrics"""
    logger.info(f"Received /summary from {update.effective_user.id}")

    if not update.message:
        return

    try:
        trading_mode = trading_engine.trading_mode if trading_engine else 'paper'
        
        # PAPER MODE - use portfolio.json
        if trading_mode == 'paper':
            from modules.portfolio import get_portfolio_summary, get_detailed_stats
            
            summary_data = get_portfolio_summary(current_prices=get_cached_prices())
            stats = get_detailed_stats()

            total_val = summary_data.get('total_value', 0)
            ret_pct = summary_data.get('total_return_pct', 0)
            pnl_emoji = "🍏" if ret_pct >= 0 else "🍎"

            message = (
                f"{pnl_emoji} *PORTFOLIO SUMMARY* [PAPER]\n"
                f"━━━━━━━━━━━━━━━\n"
                f"💰 *Balance:* `${total_val:,.2f}` (`{ret_pct:+.2f}%`)\n"
                f"💵 *Cash:* `${summary_data.get('total_cash', 0):,.2f}`\n"
                f"📦 *Positions:* `{summary_data.get('positions_count', 0)}` active\n\n"
                
                f"📈 *PERFORMANCE*\n"
                f"🏆 *Win Rate:* `{stats.get('win_rate', 0)}%`\n"
                f"⚖️ *Profit Factor:* `{stats.get('profit_factor', 0)}`\n"
                f"💎 *Sharpe Ratio:* `{stats.get('sharpe_ratio', 0)}`\n"
                f"📉 *Max Drawdown:* `{stats.get('max_drawdown', 0)}%`\n"
                f"📊 *Avg Trade:* `${stats.get('avg_trade_pnl', 0)}`"
            )

            await update.message.reply_text(message, parse_mode='Markdown')
            return
        
        # LIVE/TESTNET MODE - fetch from exchange
        from config_loader import get_binance_client
        
        client = get_binance_client()
        if not client:
            await update.message.reply_text("❌ Not connected to exchange")
            return
        
        # Get account balances
        account = client.get_account()
        
        # Calculate total value
        total_value = 0
        usdc_balance = 0
        asset_count = 0
        
        # Get current prices
        tickers = {}
        for balance in account['balances']:
            asset = balance['asset']
            free = float(balance['free'])
            if free > 0.01 and asset != 'USDC':
                try:
                    symbol = f"{asset}USDC"
                    ticker = client.get_symbol_ticker(symbol=symbol)
                    price = float(ticker['price'])
                    tickers[asset] = price
                    value = free * price
                    total_value += value
                    asset_count += 1
                except:
                    pass
        
        # Get USDC balance
        for balance in account['balances']:
            if balance['asset'] == 'USDC':
                usdc_balance = float(balance['free'])
                total_value += usdc_balance
                break
        
        # Calculate return (need initial balance - store in config or use current as baseline)
        from modules.portfolio import get_portfolio_summary
        summary = get_portfolio_summary()
        initial_balance = float(summary.get('initial_balance', 14.23))
        ret_pct = ((total_value - initial_balance) / initial_balance) * 100
        pnl_emoji = "🍏" if ret_pct >= 0 else "🍎"
        
        # Get open positions from your bot
        positions_count = len(trading_engine.open_positions) if trading_engine else 0
        
        message = (
            f"{pnl_emoji} *PORTFOLIO SUMMARY* [{trading_mode.upper()}]\n"
            f"━━━━━━━━━━━━━━━\n"
            f"💰 *Balance:* `${total_value:,.2f}` (`{ret_pct:+.2f}%`)\n"
            f"💵 *USDC:* `${usdc_balance:,.2f}`\n"
            f"📦 *Positions:* `{positions_count}` active\n"
            f"🪙 *Assets:* `{asset_count}` tokens\n\n"
            
            f"📈 *ABOUT*\n"
            f"🤖 Mode: `{trading_mode.upper()}`\n"
            f"📊 Tracking: `{len(trading_engine.symbols) if trading_engine else 0}` pairs"
        )
        
        # Add top holdings if any non-USDT assets
        non_usdc_assets = []
        for balance in account['balances']:
            asset = balance['asset']
            free = float(balance['free'])
            if asset != 'USDC' and free > 0.1:
                price = tickers.get(asset, 0)
                value = free * price
                if value > 1.0:
                    non_usdc_assets.append((asset, value))
        
        if non_usdc_assets:
            non_usdc_assets.sort(key=lambda x: x[1], reverse=True)
            message += f"\n\n📊 *Top Holdings:*\n"
            for asset, value in non_usdc_assets[:3]:
                message += f"   • {asset}: `${value:.2f}`\n"
        
        await update.message.reply_text(message, parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"❌ Summary error: {e}", exc_info=True)
        await update.message.reply_text("❌ Error generating portfolio summary.")

async def sync_positions(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Fetch real open positions directly from exchange and sync to bot"""
    if not update.message:
        return
    await update.message.reply_text("🔍 Fetching real positions from exchange...", parse_mode='Markdown')

    try:
        from config_loader import get_binance_client
        from modules.portfolio import load_portfolio, save_portfolio
        from datetime import datetime

        client = get_binance_client()
        if not client:
            await update.message.reply_text("❌ Not connected to exchange", parse_mode='Markdown')
            return

        # Get all non-zero balances
        account = client.get_account()
        skip_assets = {'USDC', 'USDT', 'BNB', 'BAT', 'BCX', 'ICX', 'CHZ',
                       'ALGO', 'COMP', 'AXS', '1INCH', 'SKY', '2Z'}  # dust/untracked

        holdings = {}
        for b in account['balances']:
            asset = b['asset']
            free = float(b['free'])
            locked = float(b['locked'])
            total = free + locked
            if total > 0.001 and asset not in skip_assets:
                holdings[asset] = total

        if not holdings:
            await update.message.reply_text("📭 No open positions found on exchange", parse_mode='Markdown')
            return

        # Get current prices and build position display
        lines = ["📊 *Open Positions on Exchange*\n", "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"]
        portfolio = load_portfolio()
        tracked = portfolio.get('positions', {})
        total_value = 0.0

        for asset, amount in holdings.items():
            symbol_usdc = f"{asset}USDC"
            symbol_usdt = f"{asset}USDT"

            # Try USDC pair first, then USDT
            current_price = None
            quote = None
            for sym, q in [(symbol_usdc, 'USDC'), (symbol_usdt, 'USDT')]:
                try:
                    ticker = client.get_symbol_ticker(symbol=sym)
                    current_price = float(ticker['price'])
                    quote = q
                    break
                except Exception:
                    continue

            if not current_price:
                lines.append(f"⚪ *{asset}*: `{amount:.6f}` (no price available)\n")
                continue

            value = amount * current_price
            total_value += value

            # Check if bot is tracking this position
            bot_symbol = f"{asset}/{quote}"
            is_tracked = bot_symbol in tracked
            tracked_icon = "🤖" if is_tracked else "⚠️"

            # Get bot's stop/target if tracked
            stop_info = ""
            if is_tracked:
                pos = tracked[bot_symbol]
                entry = pos.get('entry_price', 0)
                stop = pos.get('stop_loss', 0)
                tp = pos.get('take_profit', 0)
                pnl = (current_price - entry) * amount
                pnl_pct = ((current_price / entry) - 1) * 100 if entry > 0 else 0
                pnl_icon = "🟢" if pnl >= 0 else "🔴"
                stop_info = (
                    f"\n   Entry: `${entry:.4f}`"
                    f"\n   {pnl_icon} PnL: `${pnl:+.2f}` ({pnl_pct:+.1f}%)"
                    f"\n   🛑 Stop: `${stop:.4f}`"
                    f"\n   🎯 Target: `${tp:.4f}`"
                )
            else:
                stop_info = "\n   ⚠️ *Not tracked by bot — no stop loss active!*"

            lines.append(
                f"{tracked_icon} *{asset}* ({bot_symbol})"
                f"\n   Amount: `{amount:.6f}`"
                f"\n   Price: `${current_price:.4f}`"
                f"\n   Value: `${value:.2f}`"
                f"{stop_info}\n"
            )

        lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        lines.append(f"📈 *Total Position Value: ${total_value:.2f}*")
        lines.append(f"\n🤖 = tracked by bot | ⚠️ = untracked (no stop loss!)")

        await update.message.reply_text("\n".join(lines), parse_mode='Markdown')

    except Exception as e:
        logger.error(f"Sync positions error: {e}", exc_info=True)
        await update.message.reply_text(f"❌ Error: {str(e)[:100]}", parse_mode='Markdown')

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show bot status with real exchange data"""
    logger.info(f"Received /status from {update.effective_user.id}")
    if not update.message:
        return
    
    try:
        trading_mode = trading_engine.trading_mode if trading_engine else 'paper'
        
        # PAPER MODE - use portfolio.json
        if trading_mode == 'paper':
            from modules.portfolio import get_portfolio_summary
            summary = get_portfolio_summary(current_prices=get_cached_prices())
            
            total_trades = summary.get('total_trades', 0)
            win_rate = summary.get('win_rate', 0)

            message_lines = [
                f"🤖 *Trading Bot Status* [PAPER]\n",
                f"━━━━━━━━━━━━━━━━",
                f"📊 Mode: `{summary.get('trading_mode', 'PAPER')}`",
                f"💰 Portfolio: `${summary.get('total_value', 0):,.2f}`",
                f"💵 Cash: `${summary.get('total_cash', 0):,.2f}`",
                f"📈 Return: `{summary.get('total_return_pct', 0):+.1f}%`",
                f"🎯 Win Rate: `{win_rate:.1f}%`",
                f"📊 Active: `{summary.get('positions_count', 0)}/{trading_engine.max_positions}`",
                f"📋 Total Trades: `{total_trades}`",
            ]

            # Show active positions from paper portfolio
            positions = summary.get('positions', {})
            if positions:
                message_lines.append(f"\n📊 *Active Positions:*")
                message_lines.append(f"━━━━━━━━━━━━━━━━")
                
                for symbol, position in positions.items():
                    current_price = position.get('current_price', position.get('entry_price', 0))
                    entry_price = position.get('entry_price', 0)
                    pnl = position.get('pnl', 0)
                    pnl_pct = position.get('pnl_pct', 0)
                    pnl_emoji = "🟢" if pnl > 0 else "🔴" if pnl < 0 else "⚪"

                    def fmt(p):
                        if p == 0: return "0"
                        if p >= 100: return f"{p:.2f}"
                        if p >= 1: return f"{p:.3f}"
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
            return
        
        # LIVE/TESTNET MODE - fetch from exchange
        from config_loader import get_binance_client
        from modules.portfolio import load_portfolio

        client = get_binance_client()
        if not client:
            await update.message.reply_text("❌ Not connected to exchange")
            return

        account = client.get_account()

        # ✅ Get all balances directly — no pairing needed for the count
        usdc_balance = 0.0
        usdt_balance = 0.0
        other_assets = {}

        for b in account['balances']:
            free   = float(b['free'])
            locked = float(b['locked'])
            total  = free + locked
            if total < 0.0001:
                continue
            if b['asset'] == 'USDC':
                usdc_balance = total
            elif b['asset'] == 'USDT':
                usdt_balance = total
            else:
                other_assets[b['asset']] = total

        # Get prices for non-stable assets — try USDC then USDT
        other_value = 0.0
        for asset, amount in other_assets.items():
            for quote in ('USDC', 'USDT'):
                try:
                    ticker = client.get_symbol_ticker(symbol=f"{asset}{quote}")
                    other_value += amount * float(ticker['price'])
                    break
                except Exception:
                    continue

        total_value = usdc_balance + usdt_balance + other_value

        # ✅ Use real initial_balance from portfolio.json
        portfolio     = load_portfolio()
        initial       = portfolio.get('initial_balance', total_value)
        return_pct    = ((total_value - initial) / initial * 100) if initial > 0 else 0.0
        return_icon   = "📈" if return_pct >= 0 else "📉"

        # Performance stats
        from modules.portfolio import get_performance_summary
        perf         = get_performance_summary()
        win_rate     = perf.get('win_rate', 0)
        total_trades = perf.get('total_trades', 0)

        active_pos   = len(trading_engine.open_positions) if trading_engine else 0
        futures_pos  = len(getattr(trading_engine, 'open_futures_positions', {}))

        message_lines = [
            f"🤖 *Trading Bot Status* [{trading_mode.upper()}]\n",
            f"━━━━━━━━━━━━━━━━",
            f"📊 Mode: `{trading_mode.upper()}`",
            f"💵 USDC: `${usdc_balance:,.2f}`",
            f"💰 Total Value: `${total_value:,.2f}`",
            f"{return_icon} Return: `{return_pct:+.1f}%` vs initial `${initial:,.2f}`",
            f"🎯 Win Rate: `{win_rate:.1f}%`",
            f"📊 Active: `{active_pos + futures_pos}/{trading_engine.max_positions}`",
            f"📋 Total Trades: `{total_trades}`",
        ]

        # Active spot positions
        if trading_engine and trading_engine.open_positions:
            message_lines.append(f"\n📊 *Active Spot Positions:*")
            message_lines.append(f"━━━━━━━━━━━━━━━━")

            for symbol, position in trading_engine.open_positions.items():
                current_price = trading_engine.get_current_prices().get(symbol, position.get('entry_price', 0))
                entry_price   = position.get('entry_price', 0)
                amount        = position.get('amount', 0)
                side          = position.get('side', 'long')

                pnl     = (current_price - entry_price) * amount if side == 'long' else (entry_price - current_price) * amount
                pnl_pct = ((current_price / entry_price) - 1) * 100 if entry_price > 0 else 0
                if side == 'short': pnl_pct = -pnl_pct

                pnl_emoji = "🟢" if pnl > 0 else "🔴" if pnl < 0 else "⚪"
                trailing  = " 🔄" if position.get('trailing_stop_active') else ""

                def fmt(p):
                    if p == 0:    return "0"
                    if p >= 100:  return f"{p:.2f}"
                    if p >= 1:    return f"{p:.3f}"
                    if p >= 0.01: return f"{p:.4f}"
                    return f"{p:.6f}"

                message_lines.append(
                    f"\n{pnl_emoji} *{symbol}* ({side.upper()})"
                    f"\n   Entry: `${fmt(entry_price)}` → `${fmt(current_price)}`"
                    f"\n   Amount: `{amount:.6f}`"
                    f"\n   P&L: `${pnl:+.4f}` ({pnl_pct:+.2f}%)"
                    f"\n   Stop: `${fmt(position.get('stop_loss', 0))}`{trailing}"
                    f"\n   Target: `${fmt(position.get('take_profit', 0))}`"
                )

        # Active futures positions
        if getattr(trading_engine, 'open_futures_positions', {}):
            message_lines.append(f"\n📉 *Active Futures Positions:*")
            message_lines.append(f"━━━━━━━━━━━━━━━━")

            for symbol, position in trading_engine.open_futures_positions.items():
                current_price = trading_engine.get_current_prices().get(symbol, position.get('entry_price', 0))
                entry_price   = position.get('entry_price', 0)
                amount        = position.get('amount', 0)
                side          = position.get('side', 'short')

                pnl     = (entry_price - current_price) * amount if side == 'short' else (current_price - entry_price) * amount
                pnl_pct = (pnl / position.get('margin_used', 1)) * 100

                pnl_emoji = "🟢" if pnl > 0 else "🔴" if pnl < 0 else "⚪"

                message_lines.append(
                    f"\n{pnl_emoji} *{symbol}* ({side.upper()})"
                    f"\n   Entry: `${fmt(entry_price)}` → `${fmt(current_price)}`"
                    f"\n   P&L: `${pnl:+.4f}` ({pnl_pct:+.2f}%)"
                    f"\n   Stop: `${fmt(position.get('stop_loss', 0))}`"
                    f"\n   Target: `${fmt(position.get('take_profit', 0))}`"
                )

        await update.message.reply_text("\n".join(message_lines), parse_mode='Markdown')

    except Exception as e:
        logger.error(f"❌ Status error: {e}")
        import traceback
        logger.error(traceback.format_exc())
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
                    f"  • {s['symbol']}: {sig['side'].upper()} @ `${sig['entry_price']:.2f}`"
                )
        
        if trend_signals:
            message_lines.append(f"\n*📈 Trend Signals:*")
            for s in trend_signals[:3]:
                sig = s['signal']
                message_lines.append(
                    f"  • {s['symbol']}: {sig['side'].upper()} @ `${sig['entry_price']:.2f}`"
                )
        
        if momentum_signals:
            message_lines.append(f"\n*⚡ Momentum Signals:*")
            for s in momentum_signals[:3]:
                sig = s['signal']
                message_lines.append(
                    f"  • {s['symbol']}: {sig['side'].upper()} @ `${sig['entry_price']:.2f}`"
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
            logger.info(f"      Entry: ${signal_data['signal'].get('entry_price', 0):.2f}")
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
                required_fields = ['side', 'entry_price', 'units', 'stop_loss', 'take_profit']
                missing_fields = [f for f in required_fields if f not in signal]
                if missing_fields:
                    logger.error(f"❌ {symbol}: Missing fields: {missing_fields}")
                    failed.append(symbol)
                    fail_reasons[symbol] = f"Missing fields: {missing_fields}"
                    continue
                
                # Check cash balance before executing
                cash = trading_engine.get_cash_balance()
                cost = signal['units'] * signal['entry_price']
                
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


# async def holdings(update: Update, context: ContextTypes.DEFAULT_TYPE):
#     """Show current holdings from portfolio"""
#     if not update.message:
#         logger.warning("⚠️ Start command triggered without message object")
#         return
#     try:
#         from modules.portfolio import load_portfolio
#         portfolio = load_portfolio()
#         holdings = portfolio.get('holdings', {})
        
#         if not holdings or all(asset == 'USDT' for asset in holdings):
#             await update.message.reply_text("📭 No holdings", parse_mode='Markdown')
#             return
        
#         # Get current prices
#         current_prices = get_cached_prices()
        
#         message_lines = [f"📊 *Current Holdings:*\n"]
        
#         total_value = 0
#         cash = portfolio.get('cash_balance', 0)
        
#         for asset, amount in holdings.items():
#             if asset == 'USDT':
#                 continue
#             symbol = f"{asset}/USDT"
#             price = current_prices.get(symbol, 0)
#             value = amount * price
#             total_value += value
            
#             message_lines.append(
#                 f"• *{asset}*: {amount:.4f} @ ${price:.2f} = ${value:.2f}\n"
#             )
        
#         message_lines.append(f"\n💰 *Holdings Value: ${total_value:.2f}*")
#         message_lines.append(f"💵 *Cash: ${cash:.2f}*")
#         message_lines.append(f"📊 *Total Portfolio: ${total_value + cash:.2f}*")
        
#         await update.message.reply_text("\n".join(message_lines), parse_mode='Markdown')
        
#     except Exception as e:
#         logger.error(f"❌ Holdings error: {e}")
#         await update.message.reply_text(f"❌ Error: {str(e)[:100]}", parse_mode='Markdown')

async def positions(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show active positions — combines bot-tracked and real exchange positions"""
    if not update.message:
        return

    try:
        from config_loader import get_binance_client

        trading_mode = trading_engine.trading_mode if trading_engine else 'paper'
        client = get_binance_client() if trading_mode != 'paper' else None

        # ── 1. BOT-TRACKED POSITIONS (from portfolio.json) ──────────
        spot_positions    = trading_engine.open_positions if trading_engine else {}
        futures_positions = trading_engine.open_futures_positions if hasattr(trading_engine, 'open_futures_positions') else {}

        # ── 2. REAL EXCHANGE POSITIONS (from Binance) ────────────────
        exchange_holdings = {}
        if client:
            try:
                # Assets we don't want to show as positions
                skip_assets = {
                    'USDC', 'USDT', 'BNB',
                    # dust from your account
                    'BAT', 'BCX', 'ICX', 'CHZ', 'ALGO',
                    'COMP', 'AXS', '1INCH', 'SKY', '2Z'
                }
                account = client.get_account()
                for b in account['balances']:
                    asset = b['asset']
                    total = float(b['free']) + float(b['locked'])
                    if total > 0.001 and asset not in skip_assets:
                        exchange_holdings[asset] = total
            except Exception as e:
                logger.warning(f"Could not fetch exchange balances: {e}")

        # ── 3. CURRENT PRICES ────────────────────────────────────────
        current_prices = get_cached_prices()

        def get_price(symbol: str) -> float:
            """Try cache first, then fetch live."""
            if symbol in current_prices and current_prices[symbol] > 0:
                return current_prices[symbol]
            if client:
                for quote in ('USDC', 'USDT'):
                    try:
                        binance_sym = symbol.replace('/', '').replace(f'/{quote}', '') + quote
                        ticker = client.get_symbol_ticker(symbol=binance_sym)
                        return float(ticker['price'])
                    except Exception:
                        continue
            return 0.0

        # ── 4. BUILD OUTPUT ──────────────────────────────────────────
        message_lines = [f"📊 *Active Positions* [{trading_mode.upper()}]\n",
                         "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"]

        total_pnl     = 0.0
        position_count = 0

        def fmt(p: float) -> str:
            if p == 0:    return "0"
            if p >= 1000: return f"{p:.2f}"
            if p >= 100:  return f"{p:.3f}"
            if p >= 1:    return f"{p:.4f}"
            return f"{p:.6f}"

        # ── Bot-tracked SPOT positions ────────────────────────────────
        if spot_positions:
            message_lines.append("🤖 *Bot-Tracked Spot Positions:*\n")
            for symbol, pos in spot_positions.items():
                price   = get_price(symbol) or pos.get('current_price', pos.get('entry_price', 0))
                entry   = pos.get('entry_price', 0)
                amount  = pos.get('amount', 0)
                side    = pos.get('side', 'long')
                stop    = pos.get('stop_loss', 0)
                target  = pos.get('take_profit', 0)
                trailing = " 🔄" if pos.get('trailing_stop_active') else ""

                pnl     = (price - entry) * amount if side == 'long' else (entry - price) * amount
                pnl_pct = ((price / entry) - 1) * 100 if entry > 0 else 0
                if side == 'short': pnl_pct = -pnl_pct

                total_pnl     += pnl
                position_count += 1
                emoji = "🟢" if pnl > 0 else "🔴" if pnl < 0 else "⚪"

                # Remove this asset from exchange_holdings so we don't double-count
                asset = symbol.split('/')[0]
                exchange_holdings.pop(asset, None)

                message_lines.append(
                    f"{emoji} *{symbol}* ({side.upper()})"
                    f"\n   Entry: `${fmt(entry)}` → `${fmt(price)}`"
                    f"\n   Amount: `{amount:.6f}`"
                    f"\n   P&L: `${pnl:+.2f}` ({pnl_pct:+.1f}%)"
                    f"\n   🛑 Stop: `${fmt(stop)}`{trailing}"
                    f"\n   🎯 Target: `${fmt(target)}`\n"
                )

        # ── Bot-tracked FUTURES positions ─────────────────────────────
        if futures_positions:
            message_lines.append("\n📉 *Bot-Tracked Futures Positions:*\n")
            for symbol, pos in futures_positions.items():
                price    = get_price(symbol) or pos.get('entry_price', 0)
                entry    = pos.get('entry_price', 0)
                amount   = pos.get('amount', 0)
                side     = pos.get('side', 'short')
                stop     = pos.get('stop_loss', 0)
                target   = pos.get('take_profit', 0)
                leverage = pos.get('leverage', 1)

                pnl     = (entry - price) * amount if side == 'short' else (price - entry) * amount
                pnl_pct = (pnl / (pos.get('margin_used', 1))) * 100

                total_pnl     += pnl
                position_count += 1
                emoji = "🟢" if pnl > 0 else "🔴" if pnl < 0 else "⚪"

                asset = symbol.split('/')[0]
                exchange_holdings.pop(asset, None)

                message_lines.append(
                    f"{emoji} *{symbol}* (SHORT x{leverage})"
                    f"\n   Entry: `${fmt(entry)}` → `${fmt(price)}`"
                    f"\n   Amount: `{amount:.6f}`"
                    f"\n   P&L: `${pnl:+.2f}` ({pnl_pct:+.1f}%)"
                    f"\n   🛑 Stop: `${fmt(stop)}`"
                    f"\n   🎯 Target: `${fmt(target)}`\n"
                )

        # ── Untracked exchange positions (manual buys) ────────────────
        if exchange_holdings:
            message_lines.append("\n⚠️ *Untracked Exchange Positions (no stop loss!):*\n")
            for asset, amount in exchange_holdings.items():
                # Try USDC then USDT price
                price = 0.0
                quote_used = 'USDC'
                for quote in ('USDC', 'USDT'):
                    if client:
                        try:
                            ticker = client.get_symbol_ticker(symbol=f"{asset}{quote}")
                            price  = float(ticker['price'])
                            quote_used = quote
                            break
                        except Exception:
                            continue

                value = amount * price
                if value < 0.50:  # skip real dust
                    continue

                position_count += 1
                message_lines.append(
                    f"⚠️ *{asset}/{quote_used}*"
                    f"\n   Amount: `{amount:.6f}`"
                    f"\n   Price: `${fmt(price)}`"
                    f"\n   Value: `${value:.2f}`"
                    f"\n   ❌ No stop loss — use `/setstop` or `/syncpositions`\n"
                )

        # ── Summary ───────────────────────────────────────────────────
        if position_count == 0:
            await update.message.reply_text("📭 No active positions", parse_mode='Markdown')
            return

        message_lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        if total_pnl != 0:
            message_lines.append(f"📈 *Total Unrealized P&L: ${total_pnl:+.2f}*")
        message_lines.append(f"📊 *Total Positions: {position_count}*")
        message_lines.append(f"\n🤖 bot-managed  ⚠️ untracked")

        await update.message.reply_text("\n".join(message_lines), parse_mode='Markdown')

    except Exception as e:
        logger.error(f"Positions error: {e}", exc_info=True)
        await update.message.reply_text(f"❌ Error: {str(e)[:100]}", parse_mode='Markdown')

async def limit_buy(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Place manual limit buy order and register it for stop loss management"""
    if not update.message:
        return
    if not context.args or len(context.args) < 3:
        await update.message.reply_text(
            "Usage: `/limitbuy SYMBOL AMOUNT PRICE [STOP] [TARGET]`\n"
            "Example: `/limitbuy SOL/USDC 0.2 140 126 200`\n"
            "Stop and target are optional — defaults to 3% stop, 9% target",
            parse_mode='Markdown'
        )
        return

    symbol = context.args[0].upper()

    try:
        amount = float(context.args[1])
        price  = float(context.args[2])
    except ValueError:
        await update.message.reply_text("❌ Amount and price must be numbers", parse_mode='Markdown')
        return

    stop_loss   = float(context.args[3]) if len(context.args) >= 4 else round(price * 0.97, 4)
    take_profit = float(context.args[4]) if len(context.args) >= 5 else round(price * 1.09, 4)

    # Basic validation
    order_value = amount * price
    if order_value < 5:
        await update.message.reply_text(
            f"❌ Order value `${order_value:.2f}` is below Binance minimum ($5).\n"
            f"Increase amount or price.",
            parse_mode='Markdown'
        )
        return

    if stop_loss >= price:
        await update.message.reply_text("❌ Stop loss must be below buy price", parse_mode='Markdown')
        return

    if take_profit <= price:
        await update.message.reply_text("❌ Take profit must be above buy price", parse_mode='Markdown')
        return

    await update.message.reply_text(
        f"📝 Placing limit BUY for {symbol}...\n"
        f"Amount: `{amount}` @ `${price}`\n"
        f"Stop: `${stop_loss}` | Target: `${take_profit}`",
        parse_mode='Markdown'
    )

    try:
        trading_mode = trading_engine.trading_mode

        # ── LIVE/TESTNET: place real order on exchange ────────────────
        if trading_mode in ('live', 'testnet') and trading_engine.binance_client:
            binance_symbol = symbol.replace('/', '')

            # Validate and adjust quantity for LOT_SIZE + MIN_NOTIONAL
            is_valid, adjusted_amount, error = trading_engine.validate_and_adjust_order(symbol, amount)
            if not is_valid:
                await update.message.reply_text(f"❌ Order validation failed: {error}", parse_mode='Markdown')
                return

            if adjusted_amount != amount:
                await update.message.reply_text(
                    f"ℹ️ Quantity adjusted: `{amount}` → `{adjusted_amount}` (LOT_SIZE rules)",
                    parse_mode='Markdown'
                )
                amount = adjusted_amount

            try:
                order = trading_engine.binance_client.order_limit_buy(
                    symbol=binance_symbol,
                    quantity=round(amount, 6),
                    price=str(price)
                )
                order_id = order['orderId']
                logger.info(f"✅ Limit buy placed: {order_id} for {symbol}")
            except Exception as e:
                await update.message.reply_text(f"❌ Exchange error: {str(e)[:150]}", parse_mode='Markdown')
                return

        # ── PAPER: simulate immediately ───────────────────────────────
        elif trading_mode == 'paper':
            order_id = f"paper_{int(__import__('time').time())}"
            quote    = symbol.split('/')[1]
            cash     = trading_engine.get_cash_balance(quote)
            cost     = amount * price
            if cash < cost:
                await update.message.reply_text(
                    f"❌ Insufficient {quote}: have `${cash:.2f}`, need `${cost:.2f}`",
                    parse_mode='Markdown'
                )
                return
        else:
            await update.message.reply_text("❌ No valid trading mode or exchange connection", parse_mode='Markdown')
            return

        # ── REGISTER IN BOT so stop loss watcher picks it up ─────────
        # This is the critical step that was missing before
        success = trading_engine.open_position(
            symbol=symbol,
            side='long',
            entry_price=price,
            units=amount,
            stop_loss=stop_loss,
            take_profit=take_profit,
            signal_type='manual_limit_buy',
            atr=0.0
        )

        if success:
            sl_pct  = ((price - stop_loss) / price) * 100
            tp_pct  = ((take_profit - price) / price) * 100
            rr      = tp_pct / sl_pct if sl_pct > 0 else 0

            await update.message.reply_text(
                f"✅ *Limit BUY Registered!*\n\n"
                f"📊 Symbol: `{symbol}`\n"
                f"💵 Price: `${price}`\n"
                f"📦 Amount: `{amount}`\n"
                f"💰 Value: `${amount * price:.2f}`\n\n"
                f"🛑 Stop Loss: `${stop_loss}` (-{sl_pct:.1f}%)\n"
                f"🎯 Take Profit: `${take_profit}` (+{tp_pct:.1f}%)\n"
                f"⚖️ R:R = `{rr:.1f}x`\n\n"
                f"🤖 Bot is now watching this position.",
                parse_mode='Markdown'
            )
        else:
            await update.message.reply_text(
                f"⚠️ Order placed on exchange but *failed to register* in bot.\n"
                f"Use `/syncpositions` to manually register it with a stop loss.",
                parse_mode='Markdown'
            )

    except Exception as e:
        logger.error(f"Limit buy error: {e}", exc_info=True)
        await update.message.reply_text(f"❌ Error: {str(e)[:150]}", parse_mode='Markdown')

async def limit_sell(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Place manual limit sell order"""
    if not update.message:
        logger.warning("⚠️ Start command triggered without message object")
        return
    if not context.args or len(context.args) < 3:
        await update.message.reply_text(
            "Usage: `/limitsell SYMBOL AMOUNT PRICE`\n"
            "Example: `/limitsell BTC/USDC 0.001 55000`\n"
            "         `/limitsell SOL/USDC 2 130`",
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
    /sync_positions - Sync positions with exchange
    /positions - Show open positions
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
    application.add_handler(CommandHandler("sync_positions", sync_positions))
    application.add_handler(CommandHandler("price", current_price))
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
