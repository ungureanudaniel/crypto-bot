import json
import os
import sys
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging
from datetime import datetime
from typing import Dict, Optional
from config_loader import config, get_binance_client

logger = logging.getLogger(__name__)

PORTFOLIO_FILE = "portfolio.json"

# -------------------------------------------------------------------
# CONFIG LOADING
# -------------------------------------------------------------------
CONFIG = config.config
TRADING_MODE = CONFIG.get('trading_mode', 'paper').lower()

# -------------------------------------------------------------------
# PAPER MODE: JSON file operations
# -------------------------------------------------------------------
def load_portfolio() -> Dict:
    """Load portfolio from JSON file (paper mode)"""
    if os.path.exists(PORTFOLIO_FILE):
        try:
            with open(PORTFOLIO_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading portfolio: {e}")
    
    # Default portfolio with starting balance from config
    starting_balance = CONFIG.get('starting_balance', 100.0)
    return {
        "initial_balance": starting_balance,
        "cash_balance": starting_balance,
        "holdings": {},
        "positions": {},
        "trade_history": [],
        "performance_metrics": {
            "total_trades": 0,
            "winning_trades": 0,
            "total_pnl": 0.0,
            "win_rate": 0.0
        },
        "last_updated": datetime.now().isoformat()
    }

def save_portfolio(portfolio):
    """Save portfolio data"""
    portfolio["last_updated"] = datetime.now().isoformat()
    with open(PORTFOLIO_FILE, "w") as f:
        json.dump(portfolio, f, indent=2)

def add_trade(trade):
    """Add a trade to history (works in both modes)"""
    portfolio = load_portfolio()
    
    # Add timestamp if not present
    if "timestamp" not in trade:
        trade["timestamp"] = datetime.now().isoformat()
    
    # Add to history
    portfolio["trade_history"].append(trade)

    # Keep only last 1000 trades to prevent file bloat    
    if len(portfolio["trade_history"]) > 1000:
        portfolio["trade_history"] = portfolio["trade_history"][-1000:]
    
    # Update metrics if it's a closed trade
    if trade.get("action") == "close" and "pnl" in trade:
        metrics = portfolio["performance_metrics"]
        metrics["total_trades"] = metrics.get("total_trades", 0) + 1
        metrics["total_pnl"] = metrics.get("total_pnl", 0) + trade["pnl"]
        
        if trade["pnl"] > 0:
            metrics["winning_trades"] = metrics.get("winning_trades", 0) + 1
        
        # Update win rate
        if metrics["total_trades"] > 0:
            metrics["win_rate"] = (metrics["winning_trades"] / metrics["total_trades"]) * 100
    
    # Set initial balance if not set (for paper mode)
    if TRADING_MODE == 'paper' and portfolio["initial_balance"] is None and trade.get("action") == "open":
        pass
    
    save_portfolio(portfolio)
    return portfolio

def set_initial_balance(balance):
    """Set the initial balance (called on first sync)"""
    portfolio = load_portfolio()
    if portfolio["initial_balance"] is None:
        portfolio["initial_balance"] = balance
        save_portfolio(portfolio)
    return portfolio["initial_balance"]

def get_performance_summary():
    """Get performance metrics"""
    portfolio = load_portfolio()
    return {
        "initial_balance": portfolio.get("initial_balance", 0),
        "total_trades": portfolio["performance_metrics"].get("total_trades", 0),
        "winning_trades": portfolio["performance_metrics"].get("winning_trades", 0),
        "total_pnl": portfolio["performance_metrics"].get("total_pnl", 0),
        "win_rate": portfolio["performance_metrics"].get("win_rate", 0)
    }

# -------------------------------------------------------------------
# PAPER MODE: Balance management
# -------------------------------------------------------------------
def reset_paper_portfolio(initial_balance: Optional[float] = None):
    """Reset paper portfolio with starting balance"""
    if TRADING_MODE != 'paper':
        logger.warning("reset_paper_portfolio only works in paper mode")
        return
    
    if initial_balance is None:
        initial_balance = CONFIG.get('starting_balance', 100.0)
    
    portfolio = {
        "initial_balance": initial_balance,
        "cash_balance": initial_balance,
        "holdings": {},
        "trade_history": [],
        "performance_metrics": {
            "total_trades": 0,
            "winning_trades": 0,
            "total_pnl": 0.0,
            "win_rate": 0.0
        },
        "last_updated": datetime.now().isoformat()
    }
    save_portfolio(portfolio)
    logger.info(f"✅ Paper portfolio reset to ${initial_balance:.2f}")
    return portfolio

def update_paper_balance(asset: str, amount: float, price: float, action: str) -> Optional[float]:
    """Update paper portfolio balance (buy/sell)"""
    if TRADING_MODE != 'paper':
        logger.warning("update_paper_balance only works in paper mode")
        return None
    
    portfolio = load_portfolio()
    
    # Ensure paper mode fields exist
    if "cash_balance" not in portfolio:
        portfolio["cash_balance"] = portfolio.get("initial_balance", 10000)
    if "holdings" not in portfolio:
        portfolio["holdings"] = {}
    
    pnl = None

    if action == "buy":
        # Deduct cost from cash
        cost = amount * price
        if portfolio["cash_balance"] < cost:
            logger.warning(f"Insufficient funds: have ${portfolio['cash_balance']:.2f}, need ${cost:.2f}")
            return None
        
        portfolio["cash_balance"] -= cost
        
        # Update holdings
        if asset in portfolio["holdings"]:
            portfolio["holdings"][asset] += amount
        else:
            portfolio["holdings"][asset] = amount
            
        logger.info(f"✅ Paper buy: {amount:.6f} {asset} at ${price:.2f}")
        
    elif action == "sell":
        # Check if we have enough amount
        if asset not in portfolio["holdings"] or portfolio["holdings"][asset] < amount:
            logger.warning(f"Insufficient {asset} to sell")
            return None
        
        # Revenue from sale
        revenue = amount * price
        portfolio["cash_balance"] += revenue
        
        pnl = revenue  # This is revenue, not profit
        
        # Update holdings
        portfolio["holdings"][asset] -= amount
        if portfolio["holdings"][asset] <= 0:
            del portfolio["holdings"][asset]
            
        logger.info(f"✅ Paper sell: {amount:.6f} {asset} at ${price:.2f}, revenue: ${revenue:.2f}")
    
    # Save
    save_portfolio(portfolio)
    return pnl

def get_paper_summary():
    """Get paper portfolio summary with balances"""
    if TRADING_MODE != 'paper':
        return get_performance_summary()
    
    portfolio = load_portfolio()
    cash = portfolio.get("cash_balance", 0)
    initial = portfolio.get("initial_balance", cash)
    
    return {
        "mode": "paper",
        "cash_balance": cash,
        "holdings": portfolio.get("holdings", {}),
        "total_value": cash,  # Add holdings value with current prices in trade_engine
        "initial_balance": initial,
        "total_return": cash - initial,
        "total_return_pct": ((cash - initial) / initial * 100) if initial > 0 else 0,
        "performance_metrics": portfolio.get("performance_metrics", {}),
        "last_updated": portfolio.get("last_updated")
    }

# -------------------------------------------------------------------
# UNIFIED INTERFACE
# -------------------------------------------------------------------
def get_portfolio_summary():
    """Get portfolio summary based on trading mode"""
    if TRADING_MODE == 'paper':
        return get_paper_summary()
    else:
        # Live mode - return performance metrics
        perf = get_performance_summary()
        return {
            "mode": "live",
            "cash_balance": 0,  # Will be fetched from exchange by trade_engine
            "holdings": {},
            "total_value": 0,
            "initial_balance": perf["initial_balance"],
            "total_return": perf["total_pnl"],
            "total_return_pct": (perf["total_pnl"] / perf["initial_balance"] * 100) if perf["initial_balance"] > 0 else 0,
            "performance_metrics": perf,
            "last_updated": datetime.now().isoformat()
        }

# -------------------------------------------------------------------
# Test
# -------------------------------------------------------------------
if __name__ == "__main__":
    print(f"📊 Portfolio module - Mode: {TRADING_MODE.upper()}")
    
    if TRADING_MODE == 'paper':
        reset_paper_portfolio(100)
        
        # Test buy/sell
        update_paper_balance("BTC", 0.1, 50000, "buy")
        update_paper_balance("ETH", 1.5, 3000, "buy")
        update_paper_balance("BTC", 0.05, 51000, "sell")
        
        # Add to trade history
        add_trade({
            "symbol": "BTC/USDT",
            "action": "open",
            "side": "long",
            "amount": 0.1,
            "price": 50000
        })
        add_trade({
            "symbol": "BTC/USDT",
            "action": "close",
            "side": "long",
            "amount": 0.05,
            "price": 51000,
            "pnl": 50
        })
        
        summary = get_portfolio_summary()
        print(f"\n💰 Paper Portfolio:")
        print(f"Cash: ${summary['cash_balance']:.2f}")
        print(f"Holdings: {summary['holdings']}")
        print(f"Return: {summary['total_return_pct']:.1f}%")
        print(f"Win Rate: {summary['performance_metrics']['win_rate']:.1f}%")
    else:
        # Test live mode (just history)
        add_trade({
            "symbol": "BTC/USDT",
            "action": "close",
            "side": "long",
            "amount": 0.05,
            "price": 51000,
            "pnl": 50
        })
        perf = get_performance_summary()
        print(f"\n💰 Live Mode Performance:")
        print(f"Total Trades: {perf['total_trades']}")
        print(f"Win Rate: {perf['win_rate']:.1f}%")
        print(f"Total PnL: ${perf['total_pnl']:.2f}")