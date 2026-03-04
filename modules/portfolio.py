import json
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging
from datetime import datetime
from typing import Dict, Optional, List, Mapping

logger = logging.getLogger(__name__)

PORTFOLIO_FILE = "portfolio.json"

# -------------------------------------------------------------------
# CONFIG LOADING
# -------------------------------------------------------------------
from config_loader import config
CONFIG = config.config
TRADING_MODE = CONFIG.get('trading_mode', 'paper').lower()

# -------------------------------------------------------------------
# CORE FILE OPERATIONS
# -------------------------------------------------------------------
def load_portfolio() -> Dict:
    """Load portfolio data from file"""
    if os.path.exists(PORTFOLIO_FILE):
        try:
            with open(PORTFOLIO_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading portfolio: {e}")
    
    # Default portfolio with $100 USDT
    return {
        "positions": {},  # All open positions with full details
        "cash": {
            "USDT": 100.00,
            "USDC": 0.00
        },
        "trade_history": [],
        "performance_metrics": {
            "total_trades": 0,
            "winning_trades": 0,
            "total_pnl": 0.0,
            "win_rate": 0.0
        },
        "last_updated": datetime.now().isoformat()
    }

def save_portfolio(portfolio: Dict) -> None:
    """Save portfolio data to file"""
    portfolio["last_updated"] = datetime.now().isoformat()
    with open(PORTFOLIO_FILE, "w") as f:
        json.dump(portfolio, f, indent=2)

# -------------------------------------------------------------------
# POSITION MANAGEMENT (just data access, no logic)
# -------------------------------------------------------------------
def get_positions() -> Dict:
    """Get all open positions"""
    return load_portfolio().get("positions", {})

def save_positions(positions: Dict) -> None:
    """Save positions to file"""
    portfolio = load_portfolio()
    portfolio["positions"] = positions
    save_portfolio(portfolio)

def get_cash() -> Dict[str, float]:
    """Get cash balances"""
    return load_portfolio().get("cash", {"USDT": 0, "USDC": 0})

def update_cash(quote_currency: str, amount: float, operation: str) -> bool:
    """
    Update cash balance
    operation: 'add' or 'subtract'
    """
    portfolio = load_portfolio()
    
    if quote_currency not in portfolio["cash"]:
        portfolio["cash"][quote_currency] = 0
    
    if operation == "add":
        portfolio["cash"][quote_currency] += amount
    elif operation == "subtract":
        if portfolio["cash"][quote_currency] < amount:
            return False
        portfolio["cash"][quote_currency] -= amount
    
    save_portfolio(portfolio)
    return True

# -------------------------------------------------------------------
# TRADE HISTORY
# -------------------------------------------------------------------
def add_trade(trade: Dict) -> None:
    """Add a trade to history"""
    portfolio = load_portfolio()
    
    # Add timestamp
    trade["timestamp"] = datetime.now().isoformat()
    
    # Add to history
    portfolio["trade_history"].append(trade)
    
    # Keep only last 1000 trades
    if len(portfolio["trade_history"]) > 1000:
        portfolio["trade_history"] = portfolio["trade_history"][-1000:]
    
    # Update performance metrics for closed trades
    if trade.get("action") == "close" and "pnl" in trade:
        metrics = portfolio["performance_metrics"]
        metrics["total_trades"] = metrics.get("total_trades", 0) + 1
        metrics["total_pnl"] = metrics.get("total_pnl", 0) + trade["pnl"]
        
        if trade["pnl"] > 0:
            metrics["winning_trades"] = metrics.get("winning_trades", 0) + 1
        
        if metrics["total_trades"] > 0:
            metrics["win_rate"] = (metrics["winning_trades"] / metrics["total_trades"]) * 100
    
    save_portfolio(portfolio)

def get_trade_history(limit: int = 100) -> List[Dict]:
    """Get recent trade history"""
    portfolio = load_portfolio()
    return portfolio.get("trade_history", [])[-limit:]

# -------------------------------------------------------------------
# PERFORMANCE METRICS
# -------------------------------------------------------------------
def get_performance_summary() -> Dict:
    """Get performance metrics"""
    portfolio = load_portfolio()
    return portfolio.get("performance_metrics", {
        "total_trades": 0,
        "winning_trades": 0,
        "total_pnl": 0.0,
        "win_rate": 0.0
    })

# -------------------------------------------------------------------
# PORTFOLIO SUMMARY (READ-ONLY, NO LOGIC)
# -------------------------------------------------------------------
def get_portfolio_summary(current_prices: Optional[Mapping[str, float]] = None) -> Dict:
    """
    Get portfolio summary with current values
    current_prices: optional dict of {symbol: price} to update position values
    """
    portfolio = load_portfolio()
    perf = get_performance_summary()
    
    positions = portfolio.get("positions", {})
    cash = portfolio.get("cash", {"USDT": 0, "USDC": 0})
    
    # Calculate cash total
    total_cash = cash.get("USDT", 0) + cash.get("USDC", 0)
    
    # Calculate positions value
    positions_value = 0
    positions_pnl = 0
    enhanced_positions = {}
    
    for symbol, pos in positions.items():
        # Make a copy to avoid modifying stored data
        pos_copy = pos.copy()
        
        # Update current price if provided
        if current_prices and symbol in current_prices:
            pos_copy["current_price"] = current_prices[symbol]
        else:
            pos_copy["current_price"] = pos.get("current_price", pos["entry_price"])
        
        # Calculate current value and PnL
        current_price = pos_copy["current_price"]
        entry_price = pos["entry_price"]
        amount = pos["amount"]
        
        pos_copy["value"] = amount * current_price
        
        if pos["side"] == "long":
            pos_copy["pnl"] = (current_price - entry_price) * amount
            pos_copy["pnl_pct"] = (current_price / entry_price - 1) * 100
        else:  # short
            pos_copy["pnl"] = (entry_price - current_price) * amount
            pos_copy["pnl_pct"] = (1 - current_price / entry_price) * 100
        
        positions_value += pos_copy["value"]
        positions_pnl += pos_copy["pnl"]
        enhanced_positions[symbol] = pos_copy
    
    total_value = total_cash + positions_value
    
    return {
        "total_value": total_value,
        "cash": cash,
        "total_cash": total_cash,
        "positions": enhanced_positions,
        "positions_count": len(positions),
        "positions_value": positions_value,
        "positions_pnl": positions_pnl,
        "total_trades": perf.get("total_trades", 0),
        "winning_trades": perf.get("winning_trades", 0),
        "win_rate": perf.get("win_rate", 0),
        "total_pnl": perf.get("total_pnl", 0),
        "last_updated": portfolio.get("last_updated")
    }

# -------------------------------------------------------------------
# RESET (for testing)
# -------------------------------------------------------------------
def reset_portfolio(initial_balance: float = 100.0) -> None:
    """Reset portfolio to initial state (for testing)"""
    portfolio = {
        "positions": {},
        "cash": {
            "USDT": initial_balance,
            "USDC": 0.00
        },
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
    logger.info(f"✅ Portfolio reset to ${initial_balance:.2f}")

# -------------------------------------------------------------------
# TEST
# -------------------------------------------------------------------
if __name__ == "__main__":
    print("📊 Testing portfolio.py")
    reset_portfolio(100)
    
    # Test adding a position (normally done by trade_engine)
    portfolio = load_portfolio()
    portfolio["positions"]["BTC/USDT"] = {
        "side": "long",
        "amount": 0.001,
        "entry_price": 68765,
        "current_price": 68765,
        "stop_loss": 65326,
        "take_profit": 72203,
        "quote_currency": "USDT"
    }
    save_portfolio(portfolio)
    
    # Test summary with current prices
    current_prices = {"BTC/USDT": 69000}
    summary = get_portfolio_summary(current_prices)
    
    print(f"\n💰 Portfolio Summary:")
    print(f"Total Value: ${summary['total_value']:.2f}")
    print(f"Cash: ${summary['total_cash']:.2f}")
    for sym, pos in summary['positions'].items():
        print(f"   {sym}: {pos['side']} {pos['amount']} @ ${pos['entry_price']} -> ${pos['current_price']} (PnL: ${pos['pnl']:.2f})")
