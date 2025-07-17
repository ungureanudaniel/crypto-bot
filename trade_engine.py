import json
import logging

def load_portfolio():
    with open("portfolio.json", "r") as f:
        return json.load(f)

def save_portfolio(portfolio):
    with open("portfolio.json", "w") as f:
        json.dump(portfolio, f, indent=2)

def execute_trade(coin, regime, price):
    portfolio = load_portfolio()
    cash = portfolio['cash_balance']
    position_size = (cash * 0.10)  # 10% per trade
    amount = position_size / price

    if regime == 1:  # Trending
        logging.info(f"Trending regime detected. Buying {coin}")
        portfolio['positions'][coin] = {"amount": amount, "entry_price": price}
        portfolio['cash_balance'] -= position_size
    elif regime == 2:  # Breakout
        logging.info(f"Breakout regime detected. Buying {coin}")
        portfolio['positions'][coin] = {"amount": amount, "entry_price": price}
        portfolio['cash_balance'] -= position_size
    else:
        if coin in portfolio['positions']:
            logging.info(f"Rangebound regime detected. Closing {coin} position.")
            position = portfolio['positions'].pop(coin)
            portfolio['cash_balance'] += position['amount'] * price

    save_portfolio(portfolio)
