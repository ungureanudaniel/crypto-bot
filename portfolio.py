import json
import os

PORTFOLIO_FILE = "portfolio.json"

def load_portfolio():
    if os.path.exists(PORTFOLIO_FILE):
        with open(PORTFOLIO_FILE, "r") as f:
            return json.load(f)
    return {"cash_balance": 1000, "positions": {}}

def save_portfolio(portfolio):
    with open(PORTFOLIO_FILE, "w") as f:
        json.dump(portfolio, f, indent=2)

def update_position(coin, action, amount, price):
    """
    Updates portfolio on buy/sell, returns PnL (for sells only)
    """
    portfolio = load_portfolio()
    pnl = None

    if action == "buy":
        if coin in portfolio['positions']:
            old_pos = portfolio['positions'][coin]
            old_amount = old_pos['amount']
            old_price = old_pos['entry_price']
            new_amount = old_amount + amount
            new_entry_price = (old_price*old_amount + price*amount)/new_amount
            portfolio['positions'][coin]['amount'] = new_amount
            portfolio['positions'][coin]['entry_price'] = new_entry_price
            portfolio['positions'][coin]['current_price'] = price
        else:
            portfolio['positions'][coin] = {
                'amount': amount,
                'entry_price': price,
                'current_price': price
            }
        portfolio['cash_balance'] -= amount * price

    elif action == "sell":
        if coin in portfolio['positions'] and portfolio['positions'][coin]['amount'] >= amount:
            pos = portfolio['positions'][coin]
            pnl = (price - pos['entry_price']) * amount
            pos['amount'] -= amount
            portfolio['cash_balance'] += amount * price
            pos['current_price'] = price
            if pos['amount'] == 0:
                portfolio['positions'].pop(coin)

    save_portfolio(portfolio)
    return pnl

def get_summary(prices: dict) -> str:
    portfolio = load_portfolio()
    cash = portfolio.get('cash_balance', 0)
    total_value = cash
    holdings_value = {}

    for coin, pos in portfolio.get('positions', {}).items():
        current_price = prices.get(coin, pos['current_price'])
        value = pos['amount'] * current_price
        total_value += value
        holdings_value[coin] = {
            'amount': pos['amount'],
            'value': value,
            'entry_price': pos['entry_price'],
            'current_price': current_price,
            'pnl': (current_price - pos['entry_price']) * pos['amount']
        }

    summary = [f"Portfolio Summary:",
               f"Cash Balance: ${cash:,.2f}",
               f"Total Value: ${total_value:,.2f}",
               f"Holdings:"]
    for coin, details in holdings_value.items():
        summary.append(f"{coin}: {details['amount']} @ ${details['current_price']:.2f} "
                       f"(Value: ${details['value']:.2f}, PnL: ${details['pnl']:.2f})")
    return "\n".join(summary)
