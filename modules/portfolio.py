# portfolio.py - ONLY for trade history, NOT balances
import json
import os
from datetime import datetime

PORTFOLIO_FILE = "portfolio.json"

def load_portfolio():
    """Load portfolio data (history only)"""
    if os.path.exists(PORTFOLIO_FILE):
        try:
            with open(PORTFOLIO_FILE, "r") as f:
                return json.load(f)
        except:
            pass
    
    # Default structure
    return {
        "initial_balance": None,  # Will be set on first trade/sync
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
    """Add a trade to history"""
    portfolio = load_portfolio()
    
    # Add timestamp if not present
    if "timestamp" not in trade:
        trade["timestamp"] = datetime.now().isoformat()
    
    # Add to history
    portfolio["trade_history"].append(trade)
    
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
    
    # Set initial balance if not set
    if portfolio["initial_balance"] is None and trade.get("action") == "open":
        # This is approximate - better to set explicitly
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