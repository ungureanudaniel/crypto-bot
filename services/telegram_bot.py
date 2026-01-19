import json
import logging
import os
import pandas as pd
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
from debug.debug_portfolio import check_portfolio_structure, debug_stop_losses, simple_debug
from modules.regime_switcher import train_model, predict_regime
from modules.data_feed import fetch_ohlcv
from modules.papertrade_engine import *
from services.scheduler import start_schedulers
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
    
    # Store scheduler thread in bot_data
    if 'scheduler_thread' not in context.bot_data:
        from services.scheduler import start_schedulers
        scheduler_thread = start_schedulers(context.bot_data)
        context.bot_data['scheduler_thread'] = scheduler_thread
        logging.info("✅ Scheduler started in background thread")
    
    await update.message.reply_text("🤖 Binance AI AutoTrader is now *running*!")
    logging.info("Bot started via Telegram /start command.")

async def stop(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    context.bot_data["run_bot"] = False
    await update.message.reply_text("🛑 Bot stopped. No new trades will be executed.")
    logging.info("Trading bot manually stopped via Telegram.")

async def trading_mode(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Switch between paper trading and live trading"""
    if not context.args:
        # Show current mode
        
        current_mode = 'live' if CONFIG.get('live_trading', False) else 'paper'
        mode_display = "📝 PAPER TRADING" if current_mode == 'paper' else "🚀 LIVE TRADING"
        
        # Check API keys
        has_api_key = bool(CONFIG.get('binance_api_key'))
        has_api_secret = bool(CONFIG.get('binance_api_secret'))
        api_status = "✅ Configured" if (has_api_key and has_api_secret) else "❌ Not Configured"
        
        # Get portfolio info
        from modules.portfolio import load_portfolio
        portfolio = load_portfolio()
        
        await update.message.reply_text(
            f"🤖 *Trading Mode*\n\n"
            f"{mode_display}\n\n"
            f"💰 Starting Balance: `${CONFIG.get('starting_balance', 1000):,.2f}`\n"
            f"📈 Position Size: {CONFIG.get('position_size_pct', 10)}%\n"
            f"🔑 Binance API: {api_status}\n"
            f"💼 Current Cash: `${portfolio['cash_balance']:,.2f}`\n\n"
            "Usage:\n"
            "`/mode paper` - Enable paper trading (simulation)\n"
            "`/mode live` - Enable live trading (requires API keys)\n"
            "`/mode status` - Check current mode & API status",
            parse_mode='Markdown'
        )
        return
    
    command = context.args[0].lower()
    
    if command == 'paper':
        import json
        # Update CONFIG        
        CONFIG['live_trading'] = False
        
        with open("CONFIG.json", "w") as f:
            json.dump(CONFIG, f, indent=2)
        
        await update.message.reply_text(
            "📝 *PAPER TRADING MODE ENABLED*\n\n"
            "All trades will be simulated in portfolio.json\n"
            "No real orders will be placed on Binance\n"
            "Perfect for testing strategies safely!",
            parse_mode='Markdown'
        )
        
    elif command == 'live':
        # Check if API keys are CONFIGured

        if not CONFIG.get('binance_api_key') or not CONFIG.get('binance_api_secret'):
            await update.message.reply_text(
                "❌ *Cannot Enable Live Trading*\n\n"
                "Binance API keys are not CONFIGured in CONFIG.json\n\n"
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
        
        CONFIG['live_trading'] = True
        import json
        with open("CONFIG.json", "w") as f:
            json.dump(CONFIG, f, indent=2)
        
        await update.message.reply_text(
            "🚀 *LIVE TRADING MODE ENABLED*\n\n"
            "⚠️ REAL ORDERS WILL BE PLACED ON BINANCE!\n"
            "⚠️ REAL MONEY IS AT RISK!\n\n"
            "All trades will execute with real funds\n"
            "Use `/mode paper` to switch back to safe paper trading",
            parse_mode='Markdown'
        )
        
    elif command == 'status':
        import json
        with open("CONFIG.json") as f:
            CONFIG = json.load(f)
        
        current_mode = 'live' if CONFIG.get('live_trading', False) else 'paper'
        mode_display = "📝 PAPER TRADING" if current_mode == 'paper' else "🚀 LIVE TRADING"
        
        # Test data feed
        from modules.data_feed import data_feed
        test_symbol = CONFIG.get('coins', ['BTC/USDT'])[0]
        current_price = data_feed.get_price(test_symbol)
        
        connection_status = "✅ Connected" if current_price else "❌ Disconnected"
        
        await update.message.reply_text(
            f"*Trading Mode*: {mode_display}\n"
            f"*Data Feed*: {connection_status}\n"
            f"*Test Symbol*: {test_symbol} ${current_price:.2f if current_price else 'N/A'}\n"
            f"*Coins Monitored*: {len(CONFIG.get('coins', []))}",
            parse_mode='Markdown'
        )
        
    else:
        await update.message.reply_text("❌ Unknown command. Use `/mode paper` or `/mode live`")

async def api_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Check system and Binance API connection status"""
    await update.message.reply_text("🔗 Checking system and Binance API status...")
    
    try:
        # Import what we need
        import json
        from modules.data_feed import data_feed
        from modules.papertrade_engine import paper_engine
        from modules.portfolio import load_portfolio
        
        # Load CONFIG to check API keys
        with open("CONFIG.json", "r") as f:
            CONFIG = json.load(f)
        
        # Check if API keys are CONFIGured
        has_api_key = bool(CONFIG.get('binance_api_key'))
        has_api_secret = bool(CONFIG.get('binance_api_secret'))
        
        # Test data feed connection
        await update.message.reply_text("📡 Testing Binance data feed...")
        
        # Try to get a price to test connection
        test_symbol = CONFIG.get('coins', ['BTC/USDT'])[0]
        current_price = data_feed.get_price(test_symbol)
        
        # Get portfolio info
        portfolio = load_portfolio()
        cash_balance = portfolio.get('cash_balance', 0)
        
        # Build status message
        status_lines = [
            "📊 *SYSTEM STATUS REPORT*",
            "",
            "🔑 *Binance API Configuration:*",
            f"   API Key: {'✅ Configured' if has_api_key else '❌ Missing'}",
            f"   API Secret: {'✅ Configured' if has_api_secret else '❌ Missing'}",
            f"   Trading Mode: {'🚀 LIVE' if CONFIG.get('live_trading') else '📝 PAPER'}",
            "",
            "📡 *Data Feed Status:*",
            f"   Connection: {'✅ Connected' if current_price else '❌ Failed'}",
            f"   Test Symbol: {test_symbol}",
            f"   Current Price: ${current_price:.2f}" if current_price else "   Price: N/A",
            "",
            "💰 *Portfolio Status:*",
            f"   Cash Balance: ${cash_balance:,.2f}",
            f"   Starting Balance: ${portfolio.get('initial_balance', cash_balance):,.2f}",
            f"   Positions: {len(portfolio.get('positions', {}))}",
            "",
            "⚙️ *System Configuration:*",
            f"   Coins Monitored: {len(CONFIG.get('coins', []))}",
            f"   Trading Timeframe: {CONFIG.get('trading_timeframe', '15m')}",
            f"   Max Positions: {CONFIG.get('max_positions', 3)}",
            f"   Risk per Trade: {CONFIG.get('risk_per_trade', 0.02)*100:.1f}%",
        ]
        
        # Add warnings if needed
        if not has_api_key or not has_api_secret:
            status_lines.extend([
                "",
                "⚠️ *Warnings:*",
                "   • Binance API keys not fully CONFIGured",
                "   • Data feed uses public endpoints (rate limited)",
                "   • Live trading will not work",
            ])
        
        if CONFIG.get('live_trading') and (not has_api_key or not has_api_secret):
            status_lines.extend([
                "",
                "❌ *Critical Issue:*",
                "   • Live trading enabled but API keys missing!",
                "   • Switch to paper trading with `/mode paper`",
            ])
        
        await update.message.reply_text("\n".join(status_lines), parse_mode='Markdown')
        
    except Exception as e:
        error_message = (
            f"❌ *Error Checking System Status*\n\n"
            f"Error: {str(e)[:100]}\n\n"
            f"Check your CONFIGuration and network connection."
        )
        await update.message.reply_text(error_message, parse_mode='Markdown')
        logging.error(f"Error in api_status: {e}")

async def CONFIG_info(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show current CONFIGuration"""
    import json
    with open("CONFIG.json") as f:
        CONFIG = json.load(f)
    
    from modules.portfolio import load_portfolio
    portfolio = load_portfolio()
    
    from modules.data_feed import data_feed
    test_symbol = CONFIG.get('coins', ['BTC/USDT'])[0]
    current_price = data_feed.get_price(test_symbol)
    
    info_lines = [
        "⚙️ *Bot Configuration*",
        f"💰 Starting Balance: `${CONFIG.get('starting_balance', 1000):,.2f}`",
        f"📈 Risk per Trade: {CONFIG.get('risk_per_trade', 0.02)*100:.1f}%",
        f"⏰ Timeframe: {CONFIG.get('trading_timeframe', '15m')}",
        f"📊 Coins Monitored: {len(CONFIG.get('coins', []))}",
        f"📈 Max Positions: {CONFIG.get('max_positions', 3)}",
        f"🔐 Trading Mode: {'🚀 LIVE' if CONFIG.get('live_trading') else '📝 PAPER'}",
        "",
        "💼 *Current Portfolio*",
        f"💰 Cash: `${portfolio['cash_balance']:,.2f}`",
        f"📈 Holdings: {len(portfolio.get('holdings', {}))} coins",
        f"📋 Positions: {len(portfolio.get('positions', {}))}",
        f"📜 Trade History: {len(portfolio.get('trade_history', []))} trades",
        "",
        "📡 *Data Feed Status*",
        f"🔗 Connection: {'✅ Connected' if current_price else '❌ Disconnected'}",
        f"💵 {test_symbol}: ${current_price:.2f}" if current_price else f"💵 {test_symbol}: N/A",
    ]
    
    await update.message.reply_text("\n".join(info_lines), parse_mode='Markdown')

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    status_msg = "✅ Running" if context.bot_data.get("run_bot", False) else "⛔ Stopped"
    await update.message.reply_text(f"Bot status: {status_msg}")

async def balance(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show cash balance and unprotected holdings only"""
    from modules.portfolio import load_portfolio
    
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
    """Retrain the ML model for regime prediction"""
    await update.message.reply_text("📚 Retraining ML model...")
    train_model()
    await update.message.reply_text("✅ Model retrained successfully!")

async def regime(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Predict market regime for a given coin"""
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
    """Execute manual trade for a given coin"""
    if not context.args:
        await update.message.reply_text("Usage: `/trade BTC/USDC`", parse_mode='Markdown')
        return

    symbol = context.args[0].upper()
    
    # Get price and regime
    from modules.papertrade_engine import paper_engine
    from modules.data_feed import fetch_ohlcv
    from modules.regime_switcher import predict_regime
    
    df = fetch_ohlcv(symbol, context.bot_data.get("trading_interval", "15m"))
    if df.empty:
        await update.message.reply_text(f"❌ Could not fetch market data for {symbol}.")
        return

    regime = predict_regime(df.iloc[-1])
    price = df.iloc[-1]['close']

    # Execute manual trade
    success = paper_engine.execute_trade(symbol, regime, price)
    
    if success:
        await update.message.reply_text(f"✅ Manual trade executed for {symbol} (Regime: {regime})")
    else:
        await update.message.reply_text(f"❌ Failed to execute trade for {symbol}")

async def close_position_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Close a protected position manually"""
    if not context.args:
        await update.message.reply_text("Usage: `/close_position SYMBOL`", parse_mode='Markdown')
        return
    
    symbol = context.args[0].upper()
    
    try:
        # Import what we need
        from modules.portfolio import load_portfolio
        from modules.papertrade_engine import paper_engine
        
        portfolio = load_portfolio()
        positions = portfolio.get('positions', {})
        
        if symbol not in positions:
            await update.message.reply_text(f"❌ No protected position found for {symbol}")
            return
        
        # Get current price using paper_engine
        current_prices = paper_engine.get_current_prices()
        current_price = float(current_prices.get(symbol) or positions[symbol]['entry_price'])
        
        # Close the position using paper_engine
        success = paper_engine.close_position(symbol, current_price, "manual_close")
        
        if success:
            await update.message.reply_text(
                f"✅ *Position Closed!*\n"
                f"Symbol: `{symbol}`\n"
                f"Price: `${current_price:.2f}`\n"
                f"Amount: `{positions[symbol]['amount']:.6f}`",
                parse_mode='Markdown'
            )
        else:
            await update.message.reply_text(f"❌ Failed to close position for {symbol}")
            
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {str(e)}")

async def latest_trades(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show recent trades from portfolio"""
    from modules.portfolio import load_portfolio

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
    from modules.data_feed import fetch_ohlcv

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
    from modules.portfolio import load_portfolio
    from modules.papertrade_engine import paper_engine
    
    portfolio = load_portfolio()
    current_prices = paper_engine.get_current_prices()

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

async def scan_opportunities(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Scan all coins for trading opportunities"""
    from modules.regime_switcher import predict_regime
    from modules.papertrade_engine import paper_engine
    
    await update.message.reply_text("🔍 Scanning all coins for trading opportunities...")
    
    # Load CONFIG
    with open('CONFIG.json', 'r') as f:
        CONFIG = json.load(f)
    coins = CONFIG.get('coins', [])
    
    if not coins:
        await update.message.reply_text("❌ No coins CONFIGured in CONFIG.json")
        return
    
    opportunities = {
        "breakout": [],
        "trending": [], 
        "rangebound": []
    }
    
    scanned = 0
    errors = 0
    
    # Scan each coin
    for coin in coins:
        try:
            df = fetch_ohlcv(coin, '15m')
            if not df.empty and len(df) > 50:  # Ensure enough data
                regime = predict_regime(df)
                current_price = df.iloc[-1]['close']
                price_change_1h = (df.iloc[-1]['close'] - df.iloc[-4]['close']) / df.iloc[-4]['close'] * 100
                
                # Extract confidence
                confidence = 70  # default
                if '%' in regime:
                    try:
                        confidence = int(regime.split('(')[-1].split('%')[0])
                    except:
                        pass
                
                opportunity = {
                    'symbol': coin,
                    'regime': regime,
                    'price': current_price,
                    'change_1h': price_change_1h,
                    'confidence': confidence
                }
                
                if "Breakout" in regime and confidence > 70:
                    opportunities["breakout"].append(opportunity)
                elif "Trending" in regime and confidence > 70:
                    opportunities["trending"].append(opportunity)
                elif "Range-Bound" in regime and confidence > 80:
                    opportunities["rangebound"].append(opportunity)
                
                scanned += 1
                
        except Exception as e:
            errors += 1
            continue
    
    # Build results message
    msg_lines = [
        f"📊 *Market Scan Complete*",
        f"Scanned: {scanned} coins | Errors: {errors}",
        f"",
    ]
    
    # Breakout opportunities
    if opportunities["breakout"]:
        msg_lines.append("🚀 *BREAKOUT OPPORTUNITIES* (High Momentum)")
        # Sort by confidence
        opportunities["breakout"].sort(key=lambda x: x['confidence'], reverse=True)
        for opp in opportunities["breakout"][:5]:  # Top 5
            msg_lines.append(
                f"• {opp['symbol']}: ${opp['price']:.2f} "
                f"({opp['change_1h']:+.1f}% 1h) - {opp['regime']}"
            )
        msg_lines.append("")
    
    # Trending opportunities
    if opportunities["trending"]:
        msg_lines.append("📈 *TRENDING OPPORTUNITIES* (Good Direction)")
        opportunities["trending"].sort(key=lambda x: x['confidence'], reverse=True)
        for opp in opportunities["trending"][:5]:
            msg_lines.append(
                f"• {opp['symbol']}: ${opp['price']:.2f} "
                f"({opp['change_1h']:+.1f}% 1h) - {opp['regime']}"
            )
        msg_lines.append("")
    
    # Range-bound opportunities  
    if opportunities["rangebound"]:
        msg_lines.append("📊 *RANGE-BOUND OPPORTUNITIES* (Conservative)")
        opportunities["rangebound"].sort(key=lambda x: x['confidence'], reverse=True)
        for opp in opportunities["rangebound"][:3]:
            msg_lines.append(
                f"• {opp['symbol']}: ${opp['price']:.2f} "
                f"({opp['change_1h']:+.1f}% 1h) - {opp['regime']}"
            )
        msg_lines.append("")
    
    if not any(opportunities.values()):
        msg_lines.extend([
            "😴 *No Strong Opportunities Found*",
            "",
            "💡 *Suggestions:*",
            "• Markets might be quiet right now",
            "• Try again in 15-30 minutes",
            "• Check individual coins with `/regime SYMBOL`"
        ])
    else:
        msg_lines.extend([
            "💡 *Trading Suggestions:*",
            "• Breakout 🚀: Aggressive positions (2% risk)",
            "• Trending 📈: Moderate positions (2% risk)", 
            "• Range-Bound 📊: Conservative positions (0.6% risk)",
            "",
            "Use `/trade SYMBOL` to execute immediately"
        ])
    
    await update.message.reply_text("\n".join(msg_lines), parse_mode='Markdown')

async def portfolio(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show portfolio overview with total value"""
    from modules.papertrade_engine import load_portfolio, paper_engine
    
    portfolio = load_portfolio()
    current_prices = paper_engine.get_current_prices()
    
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
    """Help command with no formatting - guaranteed to work"""
    
    help_text = """
        Binance AI AutoTrader - Command Reference

        BOT CONTROL
        /start - Start trading bot
        /stop - Stop trading bot
        /status - Check bot status

        PORTFOLIO & BALANCE  
        /balance - Show cash & holdings
        /portfolio - Portfolio overview
        /portfolio_value - Detailed valuation
        /latest_trades - Trade history
        /price <symbol> - Current price

        TRADING & ANALYSIS
        /regime <symbol> - Market regime
        /trade <symbol> - Execute trade  
        /scan - Scan for opportunities

        ORDER MANAGEMENT
        /limit_order <symbol> <side> <amount> <price>
        /pending_orders - View pending orders
        /close_position <symbol> - Close position
        /cancel_order <symbol|all> - Cancel orders

        RISK & PROTECTION
        /protect <symbol> <sl%> <tp%> - Add stop loss
        /risk - Risk exposure

        SETTINGS & CONFIG
        /mode <paper|live|status> - Trading mode
        /set_interval <timeframe> - Set interval
        /CONFIG - Configuration
        /api_status - Test API

        DEBUGGING & MAINTENANCE
        /train - Retrain ML model
        /scheduler_status - Scheduler info  
        /debug_portfolio - Portfolio analysis
        /debug_positions - Protected positions
        /debug_regime <symbol> - Regime analysis
        /check_portfolio_health - Health check

        QUICK EXAMPLES
        /regime BTC/USDC
        /portfolio_value
        /debug_portfolio
            """
    
    await update.message.reply_text(help_text)

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

        # Load current portfolio - FIXED IMPORT
        from modules.portfolio import load_portfolio
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

        # This will:
        # 1. Add the limit order to pending orders
        # 2. Let scheduler check and execute it
        
        from modules.portfolio import save_portfolio
        import datetime
        
        # Get existing pending orders
        pending_orders = portfolio.get('pending_orders', [])
        
        # Create order ID
        from datetime import datetime
        order_id = f"limit_{symbol}_{side}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Add to pending orders
        pending_orders.append({
            'id': order_id,
            'symbol': symbol,
            'side': side,
            'amount': amount,
            'price': price,
            'timestamp': datetime.now().isoformat(),
            'status': 'pending',
            'type': 'limit'
        })
        
        # Update portfolio
        portfolio['pending_orders'] = pending_orders
        save_portfolio(portfolio)
        
        # Update bot_data
        context.bot_data["portfolio"] = portfolio
        
        await update.message.reply_text(
            f"✅ *Limit Order Placed!*\n"
            f"Symbol: `{symbol}`\n"
            f"Side: `{side.upper()}`\n"
            f"Amount: `{amount}`\n"
            f"Price: `${price:.2f}`\n"
            f"Total: `${amount * price:.2f}`\n"
            f"Order ID: `{order_id}`",
            parse_mode='Markdown'
        )
        
        # Ask if user wants stop loss protection
        await update.message.reply_text(
            "💡 *Note:* This is a limit order and will execute when price reaches your specified level.\n"
            "Use `/pending_orders` to view all pending orders.\n"
            "Use `/cancel_order {order_id}` to cancel this order.\n\n"
            "Would you like to add stop loss protection when order executes?\n"
            f"Usage: `/protect {symbol} <stop_loss_pct> <take_profit_pct>`\n"
            "Example: `/protect BTC/USDC 5 10`"
        )

    except ValueError as e:
        await update.message.reply_text(f"❌ Invalid amount or price format: {str(e)}")
    except Exception as e:
        await update.message.reply_text(f"❌ Error placing limit order: {str(e)}")


async def check_portfolio_health_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Check portfolio health and risk exposure"""
    from modules.papertrade_engine import paper_engine

    health_report = paper_engine.check_portfolio_health()

    # Ensure we received a dict; some implementations may return False/None on error
    if not isinstance(health_report, dict):
        logging.error("check_portfolio_health returned unexpected value: %s", repr(health_report))
        await update.message.reply_text("❌ Failed to compute portfolio health. See logs for details.")
        return

    # Safely extract values with defaults to avoid KeyError / type issues
    total_value = health_report.get('total_value', 0)
    cash_balance = health_report.get('cash_balance', 0)
    holdings_value = health_report.get('holdings_value', 0)
    positions_value = health_report.get('positions_value', 0)
    max_drawdown = health_report.get('max_drawdown', 0)
    risk_exposure = health_report.get('risk_exposure', 0)
    issues = health_report.get('issues', [])

    msg_lines = [
        "🩺 *Portfolio Health Check*",
        f"",
        f"• Total Value: `${total_value:,.2f}`",
        f"• Cash Balance: `${cash_balance:,.2f}`",
        f"• Holdings Value: `${holdings_value:,.2f}`",
        f"• Positions Value: `${positions_value:,.2f}`",
        f"• Max Drawdown: {max_drawdown:.1f}%",
        f"• Risk Exposure: {risk_exposure:.1f}%",
        f"",
    ]
    
    if issues:
        msg_lines.append("⚠️ *Issues Found:*")
        # Normalize issues into a list if it's not already a list to avoid iterating strings/invalid types
        if isinstance(issues, list):
            for issue in issues:
                msg_lines.append(f"• {issue}")
        else:
            # If issues is a single string or another truthy value, append it as one item
            msg_lines.append(f"• {issues}")
    else:
        msg_lines.append("✅ No issues detected. Portfolio is healthy!")
    
    await update.message.reply_text("\n".join(msg_lines), parse_mode='Markdown')

async def protect_position(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Add stop loss protection to existing position or holdings"""
    if not context.args or len(context.args) < 3:
        await update.message.reply_text(
            "Usage: `/protect SYMBOL STOP_LOSS% TAKE_PROFIT%`\n"
            "*Example:* `/protect BTC/USDC 5 10`\n\n"
            "This adds protection to:\n"
            "1. Existing positions (moves to protected positions)\n"
            "2. Holdings (creates new protected position)\n\n"
            "Stop Loss %: Percentage below current price\n"
            "Take Profit %: Percentage above current price",
            parse_mode='Markdown'
        )
        return
    
    try:
        symbol = context.args[0].upper()
        stop_loss_pct = float(context.args[1]) / 100
        take_profit_pct = float(context.args[2]) / 100
        
        # Validate percentages
        if stop_loss_pct <= 0 or take_profit_pct <= 0:
            await update.message.reply_text("❌ Stop loss and take profit percentages must be positive")
            return
        
        # Use correct imports
        from modules.portfolio import load_portfolio, save_portfolio
        from modules.papertrade_engine import paper_engine
        
        portfolio = load_portfolio()
        current_prices = paper_engine.get_current_prices()
        
        if symbol not in current_prices:
            await update.message.reply_text(f"❌ Could not get current price for {symbol}")
            return
        
        current_price = current_prices[symbol]
        
        # Calculate stop loss and take profit prices
        stop_loss_price = current_price * (1 - stop_loss_pct)
        take_profit_price = current_price * (1 + take_profit_pct)
        
        base_currency = symbol.split('/')[0]
        success = False
        
        # Check if we have this in holdings (unprotected)
        if base_currency in portfolio.get('holdings', {}):
            amount = portfolio['holdings'][base_currency]
            
            # Remove from holdings
            del portfolio['holdings'][base_currency]
            
            # Add to positions with protection
            positions = portfolio.get('positions', {})
            positions[symbol] = {
                'side': 'long',
                'amount': amount,
                'entry_price': current_price,
                'stop_loss': stop_loss_price,
                'take_profit': take_profit_price,
                'entry_time': datetime.now().isoformat(),
                'protected_at': datetime.now().isoformat(),
                'protection_source': 'holdings'
            }
            
            portfolio['positions'] = positions
            save_portfolio(portfolio)
            success = True
            
            await update.message.reply_text(
                f"✅ *Holdings Now Protected!*\n\n"
                f"*Symbol:* `{symbol}`\n"
                f"*Amount:* `{amount:.6f}`\n"
                f"*Current Price:* `${current_price:.2f}`\n"
                f"*Stop Loss:* `${stop_loss_price:.2f}` ({stop_loss_pct*100:.1f}%)\n"
                f"*Take Profit:* `${take_profit_price:.2f}` ({take_profit_pct*100:.1f}%)",
                parse_mode='Markdown'
            )
            
        # Check if we have this in positions (already protected)
        elif symbol in portfolio.get('positions', {}):
            # Update existing position protection
            positions = portfolio.get('positions', {})
            position = positions[symbol]
            
            position['stop_loss'] = stop_loss_price
            position['take_profit'] = take_profit_price
            position['protection_updated'] = datetime.now().isoformat()
            
            portfolio['positions'] = positions
            save_portfolio(portfolio)
            success = True
            
            await update.message.reply_text(
                f"✅ *Position Protection Updated!*\n\n"
                f"*Symbol:* `{symbol}`\n"
                f"*Amount:* `{position['amount']:.6f}`\n"
                f"*Entry Price:* `${position['entry_price']:.2f}`\n"
                f"*New Stop Loss:* `${stop_loss_price:.2f}` ({stop_loss_pct*100:.1f}%)\n"
                f"*New Take Profit:* `${take_profit_price:.2f}` ({take_profit_pct*100:.1f}%)",
                parse_mode='Markdown'
            )
            
        else:
            await update.message.reply_text(
                f"❌ *No {symbol} Found*\n\n"
                f"To use protection, you need to have:\n"
                f"• Unprotected holdings in `/balance`\n"
                f"• OR an existing protected position\n\n"
                f"*Check:*\n"
                f"`/balance` - Shows unprotected holdings\n"
                f"`/portfolio_value` - Shows protected positions",
                parse_mode='Markdown'
            )
            return
        
        if success:
            # Send notification
            try:
                from services.notifier import notifier
                notifier.send_message(
                    f"🛡️ Protection added to {symbol}\n"
                    f"Stop Loss: {stop_loss_pct*100:.1f}% (${stop_loss_price:.2f})\n"
                    f"Take Profit: {take_profit_pct*100:.1f}% (${take_profit_price:.2f})"
                )
            except Exception as e:
                logging.warning(f"Could not send notification: {e}")
                
    except ValueError as e:
        await update.message.reply_text("❌ Invalid percentage format. Use numbers like '5' for 5%")
    except Exception as e:
        logging.error(f"Error in protect_position: {e}")
        await update.message.reply_text(f"❌ Error adding protection: {str(e)}")

async def pending_orders(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show all pending limit orders"""
    from modules.portfolio import load_portfolio
    
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
    from modules.portfolio import load_portfolio, save_portfolio
    
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

async def quick_scan(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Quick scan for top opportunities only"""
    from modules.regime_switcher import predict_regime
    from modules.data_feed import fetch_ohlcv
    
    await update.message.reply_text("⚡ Quick scanning for top opportunities...")
    
    with open('CONFIG.json', 'r') as f:
        CONFIG = json.load(f)
    # Scan only top 10 coins for speed
    coins = CONFIG.get('coins', [])[:10]
    
    top_opportunities = []
    
    for coin in coins:
        try:
            df = fetch_ohlcv(coin, '15m')
            if not df.empty:
                regime = predict_regime(df)
                current_price = df.iloc[-1]['close']
                
                # Only consider high confidence opportunities
                if "Breakout" in regime or "Trending" in regime:
                    # Extract confidence
                    confidence = 70
                    if '%' in regime:
                        try:
                            confidence = int(regime.split('(')[-1].split('%')[0])
                        except:
                            pass
                    
                    if confidence > 75:
                        top_opportunities.append({
                            'symbol': coin,
                            'regime': regime,
                            'price': current_price,
                            'confidence': confidence
                        })
                        
        except Exception as e:
            continue
    
    # Sort by confidence
    top_opportunities.sort(key=lambda x: x['confidence'], reverse=True)
    
    if top_opportunities:
        msg_lines = ["🎯 *Top Trading Opportunities*"]
        for opp in top_opportunities[:5]:  # Top 5 only
            msg_lines.append(
                f"• {opp['symbol']}: ${opp['price']:.2f} - {opp['regime']}"
            )
        
        msg_lines.extend([
            "",
            "💡 Use `/trade SYMBOL` to execute",
            "Or `/scan` for full market analysis"
        ])
    else:
        msg_lines = [
            "😴 *No Strong Opportunities Found*",
            "Markets are quiet right now.",
            "Try full scan with `/scan` for more details"
        ]
    
    await update.message.reply_text("\n".join(msg_lines), parse_mode='Markdown')

async def check_orders_now(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Manually trigger limit order check"""
    from services.scheduler import limit_order_check_job
    await update.message.reply_text("🔍 Checking pending orders...")
    limit_order_check_job(context.bot_data)
    await update.message.reply_text("✅ Order check completed")

# -------------------------------------------------------------------
# START TELEGRAM BOT
# -------------------------------------------------------------------
def start_telegram_bot():
    """Start Telegram bot - RUNS IN MAIN THREAD, BLOCKS FOREVER"""
    logger.info("🤖 Starting Telegram bot...")
    
    # Check token
    if 'telegram_token' not in CONFIG or not CONFIG['telegram_token']:
        logger.error("❌ Telegram token not found in CONFIG.json")
        return False
    
    try:
        # Create application
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
            ("close_position", close_position_cmd),
            ("scan", scan_opportunities),
            ("quick_scan", quick_scan),
            ("scheduler_status", scheduler_status),
            ("portfolio", portfolio),
            ("portfolio_value", portfolio_value),
            ("check_portfolio_health", check_portfolio_health_cmd),
            ("limit_order", limit_order),
            ("pending_orders", pending_orders),
            ("cancel_order", cancel_order),
            ("check_orders_now", check_orders_now),
            ("mode", trading_mode),
            ("api_status", api_status),
            ("CONFIG", CONFIG_info),
            ("help", help_command),
        ]
        
        for cmd, func in handlers:
            application.add_handler(CommandHandler(cmd, func))

        logger.info("✅ Telegram bot setup complete")
        logger.info("📱 Bot is now running and listening for commands...")
        
        # Run the bot - THIS BLOCKS FOREVER
        application.run_polling()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Telegram bot error: {e}")
        return False
