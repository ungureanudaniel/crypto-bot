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

# portfolio.py
def get_summary(prices=None):
    """Get portfolio summary with current values"""
    portfolio = load_portfolio()
    
    # Use provided prices or fetch them
    if prices is None:
        try:
            from modules.trade_engine import trading_engine
            prices = trading_engine.get_current_prices()
        except ImportError:
            prices = {}
    
    cash = portfolio.get('cash_balance', 0)
    holdings = portfolio.get('holdings', {})
    positions = portfolio.get('positions', {})
    
    # Calculate total value
    total_value = cash
    
    # Add holdings value
    holdings_value = 0
    for coin, amount in holdings.items():
        symbol = f"{coin}/USDC"  # Adjust based on your quote currency
        price = prices.get(symbol, 0)
        value = amount * price
        holdings_value += value
    
    # Add positions value
    positions_value = 0
    positions_pnl = 0
    total_invested = 0
    
    for symbol, pos in positions.items():
        current_price = prices.get(symbol, pos.get('entry_price', 0))
        entry_price = pos.get('entry_price', 0)
        amount = pos.get('amount', 0)
        
        if entry_price > 0 and amount > 0:
            invested = amount * entry_price
            current_value = amount * current_price
            pnl = current_value - invested
            pnl_pct = (current_price / entry_price - 1) * 100
            
            positions_value += current_value
            total_invested += invested
            positions_pnl += pnl
    
    total_value = cash + holdings_value + positions_value
    
    # Performance metrics
    initial_balance = portfolio.get('initial_balance', total_value)
    total_return = total_value - initial_balance
    total_return_pct = (total_return / initial_balance * 100) if initial_balance > 0 else 0
    
    return {
        'cash_balance': cash,
        'holdings_value': holdings_value,
        'positions_value': positions_value,
        'total_value': total_value,
        'total_invested': total_invested,
        'positions_pnl': positions_pnl,
        'positions_pnl_pct': (positions_pnl / total_invested * 100) if total_invested > 0 else 0,
        'total_return': total_return,
        'total_return_pct': total_return_pct,
        'active_positions': len(positions),
        'holdings_count': len(holdings)
    }

if __name__ == "__main__":
    # Example usage
    update_position("BTC", "buy", 0.1, 30000)
    update_position("ETH", "buy", 1, 2000)
    update_position("BTC", "sell", 0.05, 35000)
    prices = {"BTC": 36000, "ETH": 2200}
    print(get_summary(prices))