import json
import os
import sys
import threading
import fcntl
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging
from datetime import datetime
from typing import Dict, Optional, List, Mapping

logger = logging.getLogger(__name__)

# Anchor portfolio file to the module's directory, not the working directory
PORTFOLIO_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "portfolio.json")


# Lock to prevent race conditions from concurrent read-modify-write operations
_portfolio_lock = threading.RLock()

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
    """Load portfolio data from file with retry on busy"""
    with _portfolio_lock:
        if os.path.exists(PORTFOLIO_FILE):
            for attempt in range(3):
                try:
                    with open(PORTFOLIO_FILE, "r") as f:
                        data = json.load(f)
                    
                    # Migrate old portfolios
                    if 'futures_positions' not in data:
                        data['futures_positions'] = {}
                    if 'initial_balance' not in data:
                        data['initial_balance'] = data['cash'].get('USDT', 100.0)
                    
                    return data
                    
                except OSError as e:
                    if e.errno == 16:  # Device or resource busy
                        logger.warning(f"⚠️ File busy, retry {attempt + 1}/3...")
                        time.sleep(0.1)
                        continue
                    raise
                except json.JSONDecodeError as e:
                    logger.error(f"❌ JSON corruption detected: {e}")
                    # Create backup of corrupted file
                    corrupted_file = PORTFOLIO_FILE + ".corrupted"
                    os.rename(PORTFOLIO_FILE, corrupted_file)
                    logger.warning(f"📁 Corrupted file saved as: {corrupted_file}")
                    break
                except Exception as e:
                    logger.error(f"Error loading portfolio: {e}")
                    break
        
        # Return default portfolio
        return {
            "positions": {},
            "futures_positions": {},
            "cash": {
                "USDT": 5000.00,
                "USDC": 0.00
            },
            "initial_balance": 5000.00,
            "trade_history": [],
            "performance_metrics": {
                "total_trades": 0,
                "winning_trades": 0,
                "total_pnl": 0.0,
                "win_rate": 0.0
            },
            "last_updated": datetime.now().isoformat()
        }

def _get_default_portfolio() -> Dict:
    """Get default portfolio - used when file is missing or corrupted"""
    # Try to preserve existing cash if possible
    default_cash = 5000.00  # Changed from 100 to 5000
    
    # Check if there's a backup with actual cash
    backup_file = PORTFOLIO_FILE + ".backup"
    if os.path.exists(backup_file):
        try:
            with open(backup_file, "r") as f:
                backup = json.load(f)
                if backup.get('cash', {}).get('USDT', 0) > default_cash:
                    default_cash = backup['cash']['USDT']
                    logger.info(f"💰 Restored cash from backup: ${default_cash:.2f}")
        except:
            pass
    
    return {
        "positions": {},
        "futures_positions": {},
        "cash": {
            "USDT": default_cash,
            "USDC": 0.00
        },
        "initial_balance": default_cash,
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
    """Save portfolio data to file with proper file locking"""
    with _portfolio_lock:
        # Validate JSON before saving
        try:
            json.dumps(portfolio)
        except Exception as e:
            logger.error(f"❌ Cannot save portfolio - invalid data: {e}")
            return
        
        portfolio["last_updated"] = datetime.now().isoformat()
        
        # Write to temp file first
        temp_file = PORTFOLIO_FILE + ".tmp"
        try:
            with open(temp_file, "w") as f:
                json.dump(portfolio, f, indent=2)
                f.flush()
                os.fsync(f.fileno())  # Force write to disk
        except Exception as e:
            logger.error(f"❌ Failed to write temp file: {e}")
            return
        
        # Try to rename with retry on busy
        for attempt in range(5):  # Retry up to 5 times
            try:
                os.replace(temp_file, PORTFOLIO_FILE)
                logger.debug("✅ Portfolio saved successfully")
                return
            except OSError as e:
                if e.errno == 16:  # Device or resource busy
                    logger.warning(f"⚠️ File busy, retry {attempt + 1}/5...")
                    time.sleep(0.1)  # Wait 100ms
                    continue
                raise
        
        logger.error(f"❌ Failed to save portfolio after 5 attempts")

# -------------------------------------------------------------------
# POSITION MANAGEMENT (just data access, no logic)
# -------------------------------------------------------------------
def get_positions() -> Dict:
    """Get all open positions"""
    return load_portfolio().get("positions", {})

def save_positions(positions: Dict) -> None:
    """Save spot positions to file"""
    portfolio = load_portfolio()
    portfolio["positions"] = positions
    save_portfolio(portfolio)

# -------------------------------------------------------------------
# FUTURES POSITION MANAGEMENT
# -------------------------------------------------------------------
def get_futures_positions() -> Dict:
    """Get all open futures positions (shorts and futures longs)"""
    return load_portfolio().get("futures_positions", {})

def save_futures_positions(futures_positions: Dict) -> None:
    """Save futures positions to file"""
    portfolio = load_portfolio()
    portfolio["futures_positions"] = futures_positions
    save_portfolio(portfolio)

def open_futures_position(symbol: str, side: str, amount: float,
                          entry_price: float, stop_loss: float,
                          take_profit: float, quote_currency: str = "USDT",
                          leverage: int = 1, atr: float = 0.0,
                          trailing_min_pct: Optional[float] = None,
                          trailing_max_pct: Optional[float] = None) -> bool:
    """
    Record opening a futures position (paper or live).
    side: 'short' or 'long'
    """
    with _portfolio_lock:
        portfolio = load_portfolio()
        margin_used = (amount * entry_price) / leverage

        if portfolio["cash"].get(quote_currency, 0) < margin_used:
            logger.warning(f"⚠️ Insufficient margin for futures {side} on {symbol}")
            return False

        portfolio["cash"][quote_currency] = portfolio["cash"].get(quote_currency, 0) - margin_used
        portfolio["futures_positions"][symbol] = {
            "side": side,
            "amount": amount,
            "entry_price": entry_price,
            "current_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "quote_currency": quote_currency,
            "leverage": leverage,
            "margin_used": margin_used,
            "atr": atr,
            'trailing_min_pct': trailing_min_pct,
            'trailing_max_pct': trailing_max_pct,
            "signal_type": "unknown",
            "trailing_stop_active": False,
            "opened_at": datetime.now().isoformat(),
            "market": "futures"
        }
        save_portfolio(portfolio)
        logger.info(f"📉 Futures {side.upper()} opened: {symbol} @ ${entry_price:.2f} "
                    f"| Amount: {amount} | Margin: ${margin_used:.2f}")
        return True

def close_futures_position(symbol: str, exit_price: float, reason: str = "") -> Optional[Dict]:
    """
    Close a futures position and return trade result dict, or None if not found.
    Handles both short and long futures positions correctly.
    """
    with _portfolio_lock:
        portfolio = load_portfolio()
        pos = portfolio["futures_positions"].get(symbol)
        if not pos:
            logger.warning(f"⚠️ No futures position found for {symbol}")
            return None

        entry_price = pos["entry_price"]
        amount = pos["amount"]
        side = pos["side"]
        leverage = pos.get("leverage", 1)
        margin_used = pos.get("margin_used", (amount * entry_price) / leverage)
        quote_currency = pos.get("quote_currency", "USDT")

        # PnL calculation
        if side == "short":
            pnl = (entry_price - exit_price) * amount
        else:  # futures long
            pnl = (exit_price - entry_price) * amount

        pnl_pct = (pnl / margin_used) * 100 if margin_used > 0 else 0

        # Return margin + PnL to cash
        returned = margin_used + pnl
        portfolio["cash"][quote_currency] = portfolio["cash"].get(quote_currency, 0) + max(returned, 0)

        # Remove position
        del portfolio["futures_positions"][symbol]

        # Record in trade history
        trade_record = {
            "action": "close",
            "market": "futures",
            "symbol": symbol,
            "side": side,
            "amount": amount,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl": round(pnl, 6),
            "pnl_pct": round(pnl_pct, 4),
            "margin_used": margin_used,
            "leverage": leverage,
            "reason": reason,
            "timestamp": datetime.now().isoformat()
        }
        portfolio["trade_history"].append(trade_record)
        if len(portfolio["trade_history"]) > 1000:
            portfolio["trade_history"] = portfolio["trade_history"][-1000:]

        # Update metrics
        metrics = portfolio["performance_metrics"]
        metrics["total_trades"] = metrics.get("total_trades", 0) + 1
        metrics["total_pnl"] = metrics.get("total_pnl", 0) + pnl
        if pnl > 0:
            metrics["winning_trades"] = metrics.get("winning_trades", 0) + 1
        if metrics["total_trades"] > 0:
            metrics["win_rate"] = (metrics["winning_trades"] / metrics["total_trades"]) * 100

        save_portfolio(portfolio)
        emoji = "✅" if pnl > 0 else "❌"
        logger.info(f"{emoji} Futures {side.upper()} closed: {symbol} | "
                    f"Entry: ${entry_price:.2f} → Exit: ${exit_price:.2f} | "
                    f"PnL: ${pnl:+.2f} ({pnl_pct:+.2f}%)")
        return trade_record

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
    futures_positions = portfolio.get("futures_positions", {})
    cash = portfolio.get("cash", {"USDT": 0, "USDC": 0})

    # Calculate cash total
    total_cash = cash.get("USDT", 0) + cash.get("USDC", 0)

    # --- Spot positions ---
    positions_value = 0
    positions_pnl = 0
    enhanced_positions = {}

    for symbol, pos in positions.items():
        pos_copy = pos.copy()
        if current_prices and symbol in current_prices:
            pos_copy["current_price"] = current_prices[symbol]
        else:
            pos_copy["current_price"] = pos.get("current_price", pos["entry_price"])

        current_price = pos_copy["current_price"]
        entry_price = pos["entry_price"]
        amount = pos["amount"]

        pos_copy["value"] = amount * current_price
        if pos["side"] == "long":
            pos_copy["pnl"] = (current_price - entry_price) * amount
            pos_copy["pnl_pct"] = (current_price / entry_price - 1) * 100
        else:
            pos_copy["pnl"] = (entry_price - current_price) * amount
            pos_copy["pnl_pct"] = (1 - current_price / entry_price) * 100

        pos_copy["market"] = "spot"
        positions_value += pos_copy["value"]
        positions_pnl += pos_copy["pnl"]
        enhanced_positions[symbol] = pos_copy

    # --- Futures positions ---
    futures_value = 0
    futures_pnl = 0
    enhanced_futures = {}

    for symbol, pos in futures_positions.items():
        pos_copy = pos.copy()
        # Use futures-prefixed symbol for price lookup if needed
        lookup_symbol = symbol
        if current_prices and lookup_symbol in current_prices:
            pos_copy["current_price"] = current_prices[lookup_symbol]
        else:
            pos_copy["current_price"] = pos.get("current_price", pos["entry_price"])

        current_price = pos_copy["current_price"]
        entry_price = pos["entry_price"]
        amount = pos["amount"]
        leverage = pos.get("leverage", 1)
        margin_used = pos.get("margin_used", (amount * entry_price) / leverage)

        if pos["side"] == "short":
            pos_copy["pnl"] = (entry_price - current_price) * amount
        else:
            pos_copy["pnl"] = (current_price - entry_price) * amount

        pos_copy["pnl_pct"] = (pos_copy["pnl"] / margin_used * 100) if margin_used > 0 else 0
        pos_copy["value"] = margin_used  # Margin locked, not notional
        pos_copy["market"] = "futures"

        futures_value += margin_used
        futures_pnl += pos_copy["pnl"]
        enhanced_futures[symbol] = pos_copy

    total_value = total_cash + positions_value + futures_value
    total_pnl_open = positions_pnl + futures_pnl

    initial_balance = portfolio.get("initial_balance", 100.0)
    total_return = total_value - initial_balance
    total_return_pct = (total_return / initial_balance * 100) if initial_balance > 0 else 0.0

    return {
        'trading_mode': TRADING_MODE,
        'total_value': total_value,
        'cash': cash,
        'total_cash': total_cash,
        'positions': enhanced_positions,
        'futures_positions': enhanced_futures,
        'positions_count': len(positions) + len(futures_positions),
        'spot_count': len(positions),
        'futures_count': len(futures_positions),
        "positions_pnl": positions_pnl,
        "futures_pnl": futures_pnl,
        "open_pnl": total_pnl_open,
        "total_trades": perf.get("total_trades", 0),
        "winning_trades": perf.get("winning_trades", 0),
        "win_rate": perf.get("win_rate", 0),
        "total_pnl": perf.get("total_pnl", 0),
        "initial_balance": initial_balance,
        "total_return": total_return,
        "total_return_pct": total_return_pct,
        "last_updated": portfolio.get("last_updated")
    }

# -------------------------------------------------------------------
# RESET (for testing)
# -------------------------------------------------------------------
def reset_portfolio(initial_balance: float = 100.0) -> None:
    """Reset portfolio to initial state (for testing)"""
    portfolio = {
        "positions": {},
        "futures_positions": {},
        "cash": {
            "USDT": initial_balance,
            "USDC": 0.00
        },
        "initial_balance": initial_balance,
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
