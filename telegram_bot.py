import json
import logging
import os
import pandas as pd
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
import debug_portfolio
from regime_switcher import train_model, predict_regime
from data_feed import fetch_ohlcv
from trade_engine import execute_trade
from scheduler import start_schedulers
import schedule

# -------------------------------------------------------------------
# CONFIG & LOGGING
# -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

with open("config.json") as f:
    CONFIG = json.load(f)

PORTFOLIO_FILE = "portfolio.json"

def load_portfolio():
    if os.path.exists(PORTFOLIO_FILE):
        with open(PORTFOLIO_FILE, "r") as f:
            return json.load(f)
    else:
        logging.warning("portfolio.json not found. Returning empty portfolio.")
        return {"cash_balance": 0, "holdings": {}, "positions": {}}

# -------------------------------------------------------------------
# COMMAND HANDLERS
# -------------------------------------------------------------------

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    context.bot_data["run_bot"] = True
    context.bot_data["portfolio"] = load_portfolio()
    await update.message.reply_text("🤖 Binance AI AutoTrader is now *running*!")
    start_schedulers(context.bot_data)
    logging.info("Scheduler started via Telegram /start command.")

async def stop(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    context.bot_data["run_bot"] = False
    await update.message.reply_text("🛑 Bot stopped. No new trades will be executed.")
    logging.info("Trading bot manually stopped via Telegram.")

async def trading_mode(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Switch between paper trading and live trading using your config"""
    if not context.args:
        # Show current mode
        with open("config.json") as f:
            config = json.load(f)
        
        current_mode = 'live' if config.get('live_trading', False) else 'paper'
        mode_display = "📝 PAPER TRADING" if current_mode == 'paper' else "🚀 LIVE TRADING"
        
        # Check if API keys are configured for live trading
        api_configured = bool(config.get('binance_api_key') and config.get('binance_api_secret'))
        api_status = "✅ Configured" if api_configured else "❌ Not Configured"
        
        portfolio = load_portfolio()
        
        await update.message.reply_text(
            f"🤖 *Trading Mode*: {mode_display}\n"
            f"💰 Starting Balance: `${config.get('starting_balance', 1000):,.2f}`\n"
            f"📈 Position Size: {config.get('position_size_pct', 10)}%\n"
            f"🔑 Binance API: {api_status}\n"
            f"💼 Current Balance: `${portfolio['cash_balance']:,.2f}`\n\n"
            "Usage:\n"
            "`/mode paper` - Enable paper trading (simulation)\n"
            "`/mode live` - Enable live trading (real orders)\n"
            "`/mode status` - Check current mode & API status",
            parse_mode='Markdown'
        )
        return
    
    command = context.args[0].lower()
    
    if command == 'paper':
        # Update config
        with open("config.json", "r") as f:
            config = json.load(f)
        
        config['live_trading'] = False
        
        with open("config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        await update.message.reply_text(
            "📝 *PAPER TRADING MODE ENABLED*\n\n"
            "All trades will be simulated in portfolio.json\n"
            "No real orders will be placed on Binance\n"
            "Perfect for testing strategies safely!",
            parse_mode='Markdown'
        )
        
    elif command == 'live':
        # Check if API keys are configured
        with open("config.json", "r") as f:
            config = json.load(f)
        
        if not config.get('binance_api_key') or not config.get('binance_api_secret'):
            await update.message.reply_text(
                "❌ *Cannot Enable Live Trading*\n\n"
                "Binance API keys are not configured in config.json\n\n"
                "Add your keys:\n"
                "```json\n"
                "{\n"
                '  "binance_api_key": "your_api_key_here",\n'
                '  "binance_api_secret": "your_secret_key_here"\n'
                "}\n"
                "```\n"
                "Get keys from: https://www.binance.com/en/my/settings/api-management",
                parse_mode='Markdown'
            )
            return
        
        # Confirm live trading
        if len(context.args) < 2 or context.args[1].lower() != 'confirm':
            await update.message.reply_text(
                "⚠️ *Enable Live Trading*\n\n"
                "This will execute REAL trades with REAL money on Binance!\n"
                "You could lose real money!\n\n"
                "To confirm, use: `/mode live confirm`",
                parse_mode='Markdown'
            )
            return
        
        config['live_trading'] = True
        
        with open("config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        await update.message.reply_text(
            "🚀 *LIVE TRADING MODE ENABLED*\n\n"
            "⚠️ REAL ORDERS WILL BE PLACED ON BINANCE!\n"
            "⚠️ REAL MONEY IS AT RISK!\n\n"
            "All trades will execute with real funds\n"
            "Use `/mode paper` to switch back to safe paper trading",
            parse_mode='Markdown'
        )
        
    elif command == 'status':
        with open("config.json") as f:
            config = json.load(f)
        
        current_mode = 'live' if config.get('live_trading', False) else 'paper'
        mode_display = "📝 PAPER TRADING" if current_mode == 'paper' else "🚀 LIVE TRADING"
        api_configured = bool(config.get('binance_api_key') and config.get('binance_api_secret'))
        api_status = "✅ Configured" if api_configured else "❌ Not Configured"
        
        portfolio = load_portfolio()
        
        await update.message.reply_text(
            f"*Trading Mode*: {mode_display}\n"
            f"*Binance API*: {api_status}\n"
            f"*Portfolio Balance*: `${portfolio['cash_balance']:,.2f}`\n"
            f"*Holdings*: {len(portfolio.get('holdings', {}))} coins\n"
            f"*Open Positions*: {len(portfolio.get('positions', {}))}\n"
            f"*Coins Monitored*: {len(config.get('coins', []))}",
            parse_mode='Markdown'
        )
        
    else:
        await update.message.reply_text("❌ Unknown command. Use `/mode paper` or `/mode live`")

async def api_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Check Binance API connection status"""
    from trade_engine import get_exchange
    
    await update.message.reply_text("🔗 Testing Binance API connection...")
    
    try:
        exchange = get_exchange()
        if exchange:
            # Test connection
            markets = exchange.load_markets()
            balance = exchange.fetch_balance()
            
            total_balance = balance.get('total', {})
            usdc_balance = total_balance.get('USDC', 0)
            
            await update.message.reply_text(
                "✅ *Binance API Connection Successful!*\n"
                f"💰 USDC Balance: `${usdc_balance:.2f}`\n"
                f"📈 Markets loaded: {len(markets)}\n"
                f"🔐 Testnet: {exchange.sandbox}",
                parse_mode='Markdown'
            )
        else:
            await update.message.reply_text("❌ Failed to create exchange connection - check API keys")
            
    except Exception as e:
        await update.message.reply_text(
            f"❌ *Binance API Connection Failed*\n\n"
            f"Error: {str(e)}\n\n"
            "Please check your API keys in config.json",
            parse_mode='Markdown'
        )

async def config_info(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show current configuration"""
    with open("config.json") as f:
        config = json.load(f)
    
    portfolio = load_portfolio()
    
    info_lines = [
        "⚙️ *Bot Configuration*",
        f"💰 Starting Balance: `${config.get('starting_balance', 1000):,.2f}`",
        f"📈 Position Size: {config.get('position_size_pct', 10)}%",
        f"⏰ Timeframe: {config.get('timeframe', '1h')}",
        f"📊 Coins Monitored: {len(config.get('coins', []))}",
        "",
        "💼 *Current Portfolio*",
        f"💰 Balance: `${portfolio['cash_balance']:,.2f}`",
        f"📈 Holdings: {len(portfolio.get('holdings', {}))} coins",
        f"📋 Open Positions: {len(portfolio.get('positions', {}))}",
        f"📜 Trade History: {len(portfolio.get('trade_history', []))} trades"
    ]
    
    await update.message.reply_text("\n".join(info_lines), parse_mode='Markdown')

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    status_msg = "✅ Running" if context.bot_data.get("run_bot", False) else "⛔ Stopped"
    await update.message.reply_text(f"Bot status: {status_msg}")

async def balance(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show cash balance and unprotected holdings only"""
    from trade_engine import load_portfolio
    
    portfolio = load_portfolio()
    cash = portfolio.get('cash_balance', 0)
    holdings = portfolio.get('holdings', {})
    positions = portfolio.get('positions', {})
    
    msg_lines = [
        "💼 *Quick Balance*",
        f"💰 Cash: `${cash:,.2f}`",
        f"📦 Unprotected Holdings: {len(holdings)}",
        f"🛡️ Protected Positions: {len(positions)}"
    ]
    
    if holdings:
        msg_lines.append("\n📦 *Holdings (No Stop Loss):*")
        for coin, amount in holdings.items():
            msg_lines.append(f"• {coin}: {amount}")
    else:
        msg_lines.append("\n📦 No unprotected holdings")
    
    msg_lines.append(f"\n💡 Use `/portfolio_value` for total portfolio value")
    msg_lines.append(f"💡 Use `/debug_portfolio` for detailed breakdown")
    
    await update.message.reply_text("\n".join(msg_lines), parse_mode='Markdown')

async def train(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("📚 Retraining ML model...")
    train_model()
    await update.message.reply_text("✅ Model retrained successfully!")

async def regime(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text("Usage: `/regime BTC/USDC`", parse_mode='Markdown')
        return

    symbol = context.args[0].upper()
    df = fetch_ohlcv(symbol, context.bot_data.get("trading_interval", "1h"))
    if df.empty:
        await update.message.reply_text(f"❌ No data found for {symbol}.")
        return

    current_regime = predict_regime(df.iloc[-1])
    await update.message.reply_text(f"📊 Current regime for *{symbol}*: `{current_regime}`", parse_mode='Markdown')

async def trade(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text("Usage: `/trade BTC/USDC`", parse_mode='Markdown')
        return

    symbol = context.args[0].upper()
    df = fetch_ohlcv(symbol, context.bot_data.get("trading_interval", "1h"))
    if df.empty:
        await update.message.reply_text(f"❌ Could not fetch market data for {symbol}.")
        return

    regime = predict_regime(df.iloc[-1])
    price = df.iloc[-1]['close']
    execute_trade(symbol, regime, price)

    # Refresh portfolio after trade
    context.bot_data["portfolio"] = load_portfolio()

    await update.message.reply_text(f"🚀 Executed trade for {symbol} based on regime: `{regime}`", parse_mode='Markdown')

async def latest_trades(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show recent trades from portfolio"""
    from trade_engine import load_portfolio

    portfolio = load_portfolio()
    trade_history = portfolio.get("trade_history", [])

    if not trade_history:
        await update.message.reply_text("📭 No trades have been executed yet.")
        return

    # Show last 10 trades
    msg_lines = ["📊 *Latest Trades:*"]
    for trade in trade_history[-10:]:
        line = (
            f"{trade['action'].upper()} {trade['coin']} "
            f"{trade['amount']:.6f} @ ${trade['price']:.2f} "
            f"PnL: ${trade.get('pnl', 0):.2f}"
        )
        msg_lines.append(line)

    await update.message.reply_text("\n".join(msg_lines), parse_mode="Markdown")

async def coin_price(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show current price of a coin in USDC"""
    from data_feed import fetch_ohlcv

    if not context.args:
        await update.message.reply_text("Usage: `/price BTC/USDC`", parse_mode="Markdown")
        return

    symbol = context.args[0].upper()
    df = fetch_ohlcv(symbol, "1m")  # fetch latest 1-minute candle
    if df.empty:
        await update.message.reply_text(f"❌ Could not fetch market data for {symbol}.")
        return

    current_price = df.iloc[-1]["close"]
    await update.message.reply_text(f"💵 Current price of {symbol}: ${current_price:.2f}")

async def set_interval(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if len(context.args) != 1:
        await update.message.reply_text("Usage: /set_interval <interval>\nExamples: 1m, 1h, 1d, 1w")
        return

    interval = context.args[0]
    valid_intervals = ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"]
    if interval not in valid_intervals:
        await update.message.reply_text(f"Invalid interval. Valid intervals: {', '.join(valid_intervals)}")
        return

    context.bot_data["trading_interval"] = interval
    await update.message.reply_text(f"Trading interval set to {interval}")

async def get_interval(update: Update, context: ContextTypes.DEFAULT_TYPE):
    interval = context.bot_data.get("trading_interval", "1h")
    await update.message.reply_text(f"Current trading interval: {interval}")

async def scheduler_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    jobs = schedule.get_jobs()
    if not jobs:
        await update.message.reply_text("⚙️ No scheduled jobs are currently active.")
        return

    msg = ["📆 *Scheduler Jobs:*"]
    for job in jobs:
        msg.append(f"- {job.job_func.__name__}: next run at {job.next_run}")
    await update.message.reply_text("\n".join(msg), parse_mode='Markdown')

async def portfolio_value(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show total portfolio value including cash, holdings, and positions"""
    from trade_engine import load_portfolio
    from data_feed import get_current_prices
    
    portfolio = load_portfolio()
    current_prices = get_current_prices()
    
    cash = portfolio.get('cash_balance', 0)
    holdings = portfolio.get('holdings', {})
    positions = portfolio.get('positions', {})
    
    # Calculate total value
    total_value = cash
    holdings_value = 0
    positions_value = 0
    
    # Calculate holdings value
    holdings_breakdown = []
    for coin, amount in holdings.items():
        symbol = f"{coin}/USDC"
        price = current_prices.get(symbol, 0)
        value = amount * price
        holdings_value += value
        holdings_breakdown.append(f"  • {coin}: {amount:.6f} = ${value:.2f}")
    
    # Calculate positions value
    positions_breakdown = []
    total_invested = 0
    total_pnl = 0
    
    for symbol, position in positions.items():
        current_price = current_prices.get(symbol, position['entry_price'])
        invested = position['amount'] * position['entry_price']
        current_value = position['amount'] * current_price
        pnl = current_value - invested
        pnl_pct = (current_price / position['entry_price'] - 1) * 100
        
        positions_value += current_value
        total_invested += invested
        total_pnl += pnl
        
        positions_breakdown.append(
            f"  • {symbol}: {position['amount']:.6f} = ${current_value:.2f} "
            f"(PnL: ${pnl:+.2f}, {pnl_pct:+.1f}%)"
        )
    
    total_value = cash + holdings_value + positions_value
    initial_balance = portfolio.get('initial_balance', total_value)
    total_return = total_value - initial_balance
    total_return_pct = (total_value / initial_balance - 1) * 100
    
    # Build the message
    msg_lines = [
        "💰 *PORTFOLIO SUMMARY*",
        f"",
        f"💵 *Cash Balance:* `${cash:,.2f}`",
        f"📦 *Holdings Value:* `${holdings_value:,.2f}`",
        f"🛡️ *Positions Value:* `${positions_value:,.2f}`",
        f"",
        f"📊 *TOTAL PORTFOLIO VALUE:* `${total_value:,.2f}`",
        f"",
        f"📈 *Performance:*",
        f"Initial Balance: `${initial_balance:,.2f}`",
        f"Total Return: `${total_return:+.2f}` ({total_return_pct:+.1f}%)",
    ]
    
    # Add holdings breakdown if any
    if holdings_breakdown:
        msg_lines.extend([
            f"",
            f"📦 *Holdings Breakdown:*"
        ] + holdings_breakdown)
    
    # Add positions breakdown if any
    if positions_breakdown:
        msg_lines.extend([
            f"",
            f"🛡️ *Positions Breakdown:*"
        ] + positions_breakdown)
    
    # Add PnL summary for positions
    if positions_value > 0:
        total_pnl_pct = (positions_value / total_invested - 1) * 100 if total_invested > 0 else 0
        msg_lines.extend([
            f"",
            f"🎯 *Positions Performance:*",
            f"Total Invested: `${total_invested:.2f}`",
            f"Total PnL: `${total_pnl:+.2f}` ({total_pnl_pct:+.1f}%)"
        ])
    
    # Add quick stats
    msg_lines.extend([
        f"",
        f"📋 *Quick Stats:*",
        f"• Cash: {cash/total_value*100:.1f}% of portfolio",
        f"• Holdings: {holdings_value/total_value*100:.1f}% of portfolio",
        f"• Positions: {positions_value/total_value*100:.1f}% of portfolio",
        f"• Total Assets: {len(holdings) + len(positions)}"
    ])
    
    await update.message.reply_text("\n".join(msg_lines), parse_mode='Markdown')

async def portfolio(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show portfolio overview with total value"""
    from trade_engine import load_portfolio
    from data_feed import get_current_prices
    
    portfolio = load_portfolio()
    current_prices = get_current_prices()
    
    cash = portfolio.get('cash_balance', 0)
    holdings = portfolio.get('holdings', {})
    positions = portfolio.get('positions', {})
    
    # Calculate total value quickly
    total_value = cash
    
    for coin, amount in holdings.items():
        symbol = f"{coin}/USDC"
        price = current_prices.get(symbol, 0)
        total_value += amount * price
    
    for symbol, position in positions.items():
        current_price = current_prices.get(symbol, position['entry_price'])
        total_value += position['amount'] * current_price
    
    # Build concise message
    msg_lines = [
        "💼 *Portfolio Overview*",
        f"",
        f"💰 *Cash:* `${cash:,.2f}`",
        f"📦 *Holdings:* {len(holdings)} coins",
        f"🛡️ *Positions:* {len(positions)} coins",
        f"",
        f"📊 *Total Value:* `${total_value:,.2f}`"
    ]
    
    # Show active positions if any
    if positions:
        msg_lines.append(f"\n🎯 *Active Positions:*")
        for symbol, position in positions.items():
            current_price = current_prices.get(symbol, position['entry_price'])
            pnl_pct = (current_price / position['entry_price'] - 1) * 100
            msg_lines.append(f"• {symbol}: {position['amount']:.6f} ({pnl_pct:+.1f}%)")
    
    await update.message.reply_text("\n".join(msg_lines), parse_mode='Markdown')

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Enhanced help command with debugging capabilities"""
    
    help_categories = {
        "🤖 BOT CONTROL": [
            "/start - Start trading bot",
            "/stop - Stop trading bot", 
            "/status - Check bot status",
            "/restart - Restart bot services (if implemented)"
        ],
        
        "💼 PORTFOLIO & BALANCE": [
            "/balance - Show cash & unprotected holdings",
            "/portfolio - Portfolio overview with total value", 
            "/portfolio_value - Detailed portfolio valuation",
            "/latest_trades - Recent trade history", 
            "/performance - Trading performance stats",
            "/price <symbol> - Current price (e.g., /price BTC/USDC)"
        ],
        
        "📊 TRADING & ANALYSIS": [
            "/regime <symbol> - Market regime analysis",
            "/trade <symbol> - Execute immediate trade",
            "/scan - Scan all coins for opportunities",
            "/volatility - Market volatility overview"
        ],
        
        "📋 ORDER MANAGEMENT": [
            "/limit_order <symbol> <side> <amount> <price> - Place limit order",
            "/pending_orders - View pending limit orders",
            "/cancel_order <symbol|all> - Cancel orders",
            "/check_orders_now - Manually check pending orders"
        ],
        
        "🛡️ RISK & PROTECTION": [
            "/protect <symbol> <sl%> <tp%> - Add stop loss/take profit",
            "/risk - Current risk exposure",
            "/emergency_stop - Immediately close all positions"
        ],
        
        "⚙️ SETTINGS & CONFIG": [
            "/mode <paper|live|status> - Trading mode",
            "/set_interval <timeframe> - Set trading interval",
            "/get_interval - Show current interval",
            "/config - Current configuration",
            "/coins - View monitored coins"
        ],
        
        "🔧 SYSTEM & MAINTENANCE": [
            "/train - Retrain ML model",
            "/scheduler_status - Job scheduler status",
            "/api_status - Test exchange API connection",
            "/logs - View recent bot logs",
            "/health - System health check"
        ],
        
        "🐛 DEBUGGING & MONITORING": [
            "/debug_portfolio - Detailed portfolio analysis",
            "/debug_positions - Show all protected positions",
            "/debug_holdings - Show all unprotected holdings",
            "/debug_regime <symbol> - Detailed regime analysis",
            "/debug_trade <symbol> - Test trade without executing",
            "/debug_scheduler - Detailed scheduler info",
            "/debug_model - ML model status and performance",
            "/debug_balance - Cash flow analysis"
        ],
        
        "💡 QUICK EXAMPLES": [
            "• /regime BTC/USDC",
            "• /limit_order ETH/USDC buy 0.1 2500.00",
            "• /portfolio (to see both holdings AND positions)",
            "• /debug_portfolio (for detailed analysis)",
            "• /debug_regime ETH/USDC",
            "• /scan"
        ],
        
        "🔐 SAFETY REMINDERS": [
            "• Always test with /mode paper first",
            "• Use /mode live confirm for real trading",
            "• Monitor with /risk and /portfolio regularly",
            "• Use /emergency_stop if needed",
            "• Check /debug_positions to see protected trades"
        ]
    }
    
    # Build the help message
    help_text = "🤖 *Binance AI AutoTrader - Command Reference*\n\n"
    
    for category, commands in help_categories.items():
        help_text += f"*{category}*\n"
        for command in commands:
            help_text += f"• {command}\n"
        help_text += "\n"
    
    help_text += "---\n"
    help_text += "🔍 *Having Issues?*\n"
    help_text += "Use the /debug_* commands to investigate problems\n"
    
    await update.message.reply_text(help_text, parse_mode='Markdown')

async def limit_order(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Place a manual limit order: /limit_order BTC/USDC buy 0.001 35000.50"""
    if not context.args or len(context.args) < 4:
        await update.message.reply_text(
            "📋 *Usage:* `/limit_order SYMBOL SIDE AMOUNT PRICE`\n"
            "*Examples:*\n"
            "`/limit_order BTC/USDC buy 0.001 35000.50`\n"
            "`/limit_order ETH/USDC sell 0.1 2500.00`\n"
            "*Sides:* `buy` or `sell`",
            parse_mode='Markdown'
        )
        return

    try:
        symbol = context.args[0].upper()
        side = context.args[1].lower()
        amount = float(context.args[2])
        price = float(context.args[3])
        
        # Validate inputs
        if side not in ['buy', 'sell']:
            await update.message.reply_text("❌ Side must be 'buy' or 'sell'")
            return
        
        if amount <= 0 or price <= 0:
            await update.message.reply_text("❌ Amount and price must be positive")
            return

        # Load current portfolio
        portfolio = load_portfolio()
        context.bot_data["portfolio"] = portfolio
        
        # Check balance for buy orders
        if side == 'buy':
            total_cost = amount * price
            if portfolio.get('cash_balance', 0) < total_cost:
                await update.message.reply_text(
                    f"❌ Insufficient funds. Need ${total_cost:.2f}, "
                    f"have ${portfolio.get('cash_balance', 0):.2f}"
                )
                return

        # Check holdings for sell orders
        if side == 'sell':
            coin = symbol.split('/')[0]  # Extract base currency
            current_holdings = portfolio.get('holdings', {}).get(coin, 0)
            if current_holdings < amount:
                await update.message.reply_text(
                    f"❌ Insufficient {coin}. Need {amount}, have {current_holdings}"
                )
                return

        # Execute the limit order
        from trade_engine import execute_limit_order
        success, message = execute_limit_order(symbol, side, amount, price)
        
        if success:
            # Refresh portfolio
            context.bot_data["portfolio"] = load_portfolio()
            
            # Add to pending orders
            if 'pending_orders' not in context.bot_data:
                context.bot_data['pending_orders'] = []
            
            order_id = f"{symbol}_{side}_{amount}_{price}"
            context.bot_data['pending_orders'].append({
                'id': order_id,
                'symbol': symbol,
                'side': side,
                'amount': amount,
                'price': price,
                'timestamp': pd.Timestamp.now().isoformat()
            })
            
            await update.message.reply_text(
                f"✅ *Limit Order Placed!*\n"
                f"Symbol: `{symbol}`\n"
                f"Side: `{side.upper()}`\n"
                f"Amount: `{amount}`\n"
                f"Price: `${price:.2f}`\n"
                f"Total: `${amount * price:.2f}`",
                parse_mode='Markdown'
            )
            # Ask if user wants stop loss protection
            await update.message.reply_text(
                f"✅ Limit order placed!\n"
                f"Would you like to add stop loss protection?\n"
                f"Use: /protect {symbol} {stop_loss_pct}% {take_profit_pct}%\n"
                f"Example: /protect BTC/USDC 5 10"
            )
        else:
            await update.message.reply_text(f"❌ Failed to place order: {message}")

    except ValueError as e:
        await update.message.reply_text("❌ Invalid amount or price format")

async def protect_position(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Add stop loss protection to existing position"""
    if not context.args or len(context.args) < 3:
        await update.message.reply_text(
            "Usage: /protect SYMBOL STOP_LOSS% TAKE_PROFIT%\n"
            "Example: /protect BTC/USDC 5 10"
        )
        return
    
    symbol = context.args[0].upper()
    stop_loss_pct = float(context.args[1]) / 100
    take_profit_pct = float(context.args[2]) / 100
    
    from trade_engine import add_stop_loss_to_manual_buy, load_portfolio
    
    portfolio = load_portfolio()
    
    # Check if we have this in holdings
    base_currency = symbol.split('/')[0]
    if base_currency in portfolio.get('holdings', {}):
        amount = portfolio['holdings'][base_currency]
        current_price = get_current_price(symbol)  # You'll need to implement this
        
        success = add_stop_loss_to_manual_buy(symbol, current_price, amount, stop_loss_pct, take_profit_pct)
        if success:
            await update.message.reply_text(
                f"✅ Protection added to {symbol}:\n"
                f"Amount: {amount}\n"
                f"Stop Loss: {stop_loss_pct*100}%\n"
                f"Take Profit: {take_profit_pct*100}%"
            )
        else:
            await update.message.reply_text("❌ Failed to add protection")
    else:
        await update.message.reply_text(f"❌ No {symbol} found in holdings")        

async def pending_orders(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show all pending limit orders"""
    from trade_engine import load_portfolio
    
    portfolio = load_portfolio()
    pending_orders = portfolio.get('pending_orders', [])
    
    if not pending_orders:
        await update.message.reply_text("📭 No pending orders")
        return
    
    msg_lines = ["📋 *Pending Limit Orders:*"]
    for i, order in enumerate(pending_orders, 1):
        from datetime import datetime
        order_time = datetime.fromisoformat(order['timestamp']).strftime("%m/%d %H:%M")
        msg_lines.append(
            f"{i}. `{order['symbol']}` {order['side'].upper()} "
            f"{order['amount']} @ ${order['price']:.2f} "
            f"({order_time})"
        )
    
    await update.message.reply_text("\n".join(msg_lines), parse_mode='Markdown')

async def cancel_order(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Cancel a pending limit order: /cancel_order BTC/USDC or /cancel_order all"""
    from trade_engine import load_portfolio, save_portfolio
    
    if not context.args:
        await update.message.reply_text(
            "Usage: `/cancel_order SYMBOL` or `/cancel_order all`",
            parse_mode='Markdown'
        )
        return
    
    symbol = context.args[0].upper()
    portfolio = load_portfolio()
    pending_orders = portfolio.get('pending_orders', [])
    
    if symbol == 'ALL':
        canceled_count = len(pending_orders)
        portfolio['pending_orders'] = []
        save_portfolio(portfolio)
        await update.message.reply_text(f"✅ Canceled all {canceled_count} pending orders")
        return
    
    # Cancel orders for specific symbol
    initial_count = len(pending_orders)
    portfolio['pending_orders'] = [
        order for order in pending_orders 
        if order['symbol'] != symbol
    ]
    canceled_count = initial_count - len(portfolio['pending_orders'])
    save_portfolio(portfolio)
    
    if canceled_count > 0:
        await update.message.reply_text(f"✅ Canceled {canceled_count} orders for {symbol}")
    else:
        await update.message.reply_text(f"❌ No pending orders found for {symbol}")

async def check_orders_now(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Manually trigger limit order check"""
    from scheduler import manual_limit_order_check
    await update.message.reply_text("🔍 Checking pending orders...")
    manual_limit_order_check()
    await update.message.reply_text("✅ Order check completed")

# -------------------------------------------------------------------
# START TELEGRAM BOT
# -------------------------------------------------------------------
def start_telegram_bot():
    application = ApplicationBuilder().token(CONFIG['telegram_token']).build()
    application.bot_data["run_bot"] = False
    application.bot_data["trading_interval"] = "1h"
    application.bot_data["portfolio"] = load_portfolio()

    # Register commands
    handlers = [
        ("start", start),
        ("stop", stop),
        ("status", status),
        ("balance", balance),
        ("latest_trades", latest_trades),
        ("price", coin_price),
        ("train", train),
        ("set_interval", set_interval),
        ("get_interval", get_interval),
        ("regime", regime),
        ("trade", trade),
        ("scheduler_status", scheduler_status),
        ("portfolio", portfolio),  # New portfolio overview
        ("portfolio_value", portfolio_value),  # New detailed portfolio value
        ("debug_portfolio", debug_portfolio),
        ("limit_order", limit_order),
        ("pending_orders", pending_orders),
        ("cancel_order", cancel_order),
        ("check_orders_now", check_orders_now),
        ("mode", trading_mode),
        ("api_status", api_status),
        ("config", config_info),
        ("help", help_command),
    ]
    for cmd, func in handlers:
        application.add_handler(CommandHandler(cmd, func))

    logging.info("🚀 Starting Telegram bot...")
    application.run_polling()
    logging.info("✅ Telegram bot started and polling for commands.")

    return application
