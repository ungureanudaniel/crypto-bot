import json
import logging
from strategy_tools import generate_trade_signal
from notifier import send_telegram_message

TRADING_FEE = 0.001  # 0.1%

# -------------------------------------------------------------------
# Portfolio handling
# -------------------------------------------------------------------
def load_portfolio():
    with open("portfolio.json", "r") as f:
        return json.load(f)

def save_portfolio(portfolio):
    with open("portfolio.json", "w") as f:
        json.dump(portfolio, f, indent=2)

# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------
def load_config():
    with open('config.json', 'r') as f:
        return json.load(f)

# -------------------------------------------------------------------
# Execute a trade using breakout/ATR strategy
# -------------------------------------------------------------------
def execute_trade(symbol, df, portfolio_cash=None, risk_pct=None):
    """
    symbol: trading pair (e.g., BTC)
    df: OHLCV dataframe
    portfolio_cash: optional override of cash equity
    risk_pct: optional risk percentage override
    """
    portfolio = load_portfolio()
    positions = portfolio.get('positions', {})
    cash_balance = portfolio_cash if portfolio_cash is not None else portfolio['cash_balance']
    config = load_config()

    risk_pct = risk_pct if risk_pct is not None else config.get('risk_pct', 0.01)

    # Generate trade signal from strategy_tools
    signal = generate_trade_signal(df, cash_balance, risk_pct=risk_pct)
    if not signal:
        logging.info(f"[{symbol}] No trade signal generated.")
        return

    side = signal['side']
    entry_price = signal['entry']
    stop_loss = signal['stop_loss']
    take_profit = signal['take_profit']
    units = signal['units'] * (1 - TRADING_FEE)

    # Check if position already exists
    if symbol in positions:
        logging.info(f"[{symbol}] Position already open. Skipping new trade.")
        return

    # Check if enough cash
    required_cash = entry_price * units
    if required_cash > cash_balance:
        logging.warning(f"[{symbol}] Not enough cash for trade. Required: ${required_cash:.2f}, Available: ${cash_balance:.2f}")
        return

    # Save the position
    positions[symbol] = {
        "side": side,
        "amount": units,
        "entry_price": entry_price,
        "stop_loss": stop_loss,
        "take_profit": take_profit
    }

    # Deduct cash
    portfolio['cash_balance'] -= required_cash
    portfolio['positions'] = positions
    save_portfolio(portfolio)

    # Send Telegram notification
    msg = (
        f"🚀 {side.upper()} {symbol}\n"
        f"Amount: {units:.6f} @ ${entry_price:.2f}\n"
        f"Stop-Loss: ${stop_loss:.2f} | Take-Profit: ${take_profit:.2f}\n"
        f"Cash remaining: ${portfolio['cash_balance']:.2f}"
    )
    logging.info(msg)
    send_telegram_message(msg)

# -------------------------------------------------------------------
# Close position manually or on regime exit
# -------------------------------------------------------------------
def close_position(symbol, price):
    portfolio = load_portfolio()
    positions = portfolio.get('positions', {})

    if symbol not in positions:
        logging.info(f"[{symbol}] No position to close.")
        return

    position = positions.pop(symbol)
    side = position['side']
    amount = position['amount']
    entry_price = position['entry_price']

    proceeds = amount * price * (1 - TRADING_FEE)
    portfolio['cash_balance'] += proceeds
    portfolio['positions'] = positions
    save_portfolio(portfolio)

    pnl = (price - entry_price) * amount if side == "long" else (entry_price - price) * amount

    msg = (
        f"💰 CLOSE {side.upper()} {symbol}\n"
        f"Amount: {amount:.6f} @ ${price:.2f}\n"
        f"P/L: ${pnl:.2f}\n"
        f"Cash balance: ${portfolio['cash_balance']:.2f}"
    )
    logging.info(msg)
    send_telegram_message(msg)
