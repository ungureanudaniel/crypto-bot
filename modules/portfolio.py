import json
import os
import sys
import threading
import fcntl
import time
import pandas as pd
import numpy as np
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging
from datetime import datetime
from typing import Dict, Optional, List, Mapping, Any

logger = logging.getLogger(__name__)

# Anchor portfolio file to the module's directory, not the working directory
PORTFOLIO_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "portfolio.json")
HISTORY_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "history.json")

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

def _make_serializable(obj: Any) -> Any:
    """Convert non‑serializable objects (Timestamp, ndarray, etc.) to JSON‑compatible types."""
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(item) for item in obj]
    return obj

def save_portfolio(portfolio: Dict) -> None:
    """Atomic save: Writes to .tmp then renames to prevent corruption."""
    with _portfolio_lock:
        clean = _make_serializable(portfolio)
        clean["last_updated"] = datetime.now().isoformat()
        
        tmp_file = PORTFOLIO_FILE + ".tmp"
        try:
            with open(tmp_file, "w") as f:
                json.dump(clean, f, indent=2)
                f.flush()
                os.fsync(f.fileno()) # Force write to physical disk
            
            # Atomic swap
            os.replace(tmp_file, PORTFOLIO_FILE)
            logger.debug(f"✅ Portfolio saved successfully.")
        except Exception as e:
            logger.error(f"❌ Critical failure saving portfolio: {e}")
            if os.path.exists(tmp_file):
                os.remove(tmp_file)

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

def open_futures_position(symbol, side, amount, entry_price, stop_loss, take_profit, 
                          leverage=1, quote_currency="USDT", **kwargs) -> bool:
    with _portfolio_lock:
        portfolio = load_portfolio()
        notional_value = amount * entry_price
        margin_used = notional_value / leverage
        
        # FIX: Include the entry fee (assuming 0.05% taker fee)
        fee = notional_value * 0.0005 
        total_required = margin_used + fee

        if portfolio["cash"].get(quote_currency, 0) < total_required:
            logger.warning(f"⚠️ Insufficient funds (Margin+Fee) for {symbol}")
            return False

        portfolio["cash"][quote_currency] -= total_required
        
        # Calculate Liquidation Price (Simple Isolated approximation)
        # For Long: Entry * (1 - 1/Lev); For Short: Entry * (1 + 1/Lev)
        lev_factor = (1 / leverage) * 0.8 # 80% margin maintenance buffer
        if side == 'long':
            liq_price = entry_price * (1 - lev_factor)
        else:
            liq_price = entry_price * (1 + lev_factor)

        portfolio["futures_positions"][symbol] = {
            "side": side, "amount": amount, "entry_price": entry_price,
            "margin_used": margin_used, "leverage": leverage,
            "liq_price": liq_price, "fee_paid": fee,
            "opened_at": datetime.now().isoformat()
        }
        save_portfolio(portfolio)
        return True

def close_futures_position(symbol, exit_price, reason=""):
    with _portfolio_lock:
        portfolio = load_portfolio()
        pos = portfolio["futures_positions"].pop(symbol, None)
        if not pos: return None

        # Calculate Gross PnL
        if pos['side'] == "short":
            pnl = (pos['entry_price'] - exit_price) * pos['amount']
        else:
            pnl = (exit_price - pos['entry_price']) * pos['amount']

        # Subtract Exit Fee
        exit_fee = (pos['amount'] * exit_price) * 0.0005
        net_pnl = pnl - exit_fee - pos.get('fee_paid', 0)
        
        # Final amount returned to wallet
        # If net_pnl is -100, you return 0. If net_pnl is -110 (liquidation), you return 0.
        returned_to_wallet = pos['margin_used'] + pnl - exit_fee
        portfolio["cash"]["USDT"] += max(0, returned_to_wallet)

        # Update History & Metrics...
        save_portfolio(portfolio)
        return {"pnl": net_pnl, "pnl_pct": (net_pnl/pos['margin_used'])*100}

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
    """Add a trade to history.json and update portfolio metrics."""
    with _portfolio_lock:
        # 1. Prepare the trade object
        trade_entry = _make_serializable(trade)
        if "timestamp" not in trade_entry:
            trade_entry["timestamp"] = datetime.now().isoformat()

        # 2. Update history.json
        history = []
        if os.path.exists(HISTORY_FILE):
            try:
                with open(HISTORY_FILE, "r") as f:
                    history = json.load(f)
            except Exception as e:
                logger.error(f"❌ Error loading history.json: {e}")

        history.append(trade_entry)

        # Atomic save for history.json
        tmp_hist = HISTORY_FILE + ".tmp"
        try:
            with open(tmp_hist, "w") as f:
                json.dump(history, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_hist, HISTORY_FILE)
        except Exception as e:
            logger.error(f"❌ Failed to save history.json: {e}")

        # 3. Update Performance Metrics in portfolio.json
        if trade.get("action") == "close" and "pnl" in trade:
            portfolio = load_portfolio()
            metrics = portfolio["performance_metrics"]
            
            metrics["total_trades"] = metrics.get("total_trades", 0) + 1
            metrics["total_pnl"] = metrics.get("total_pnl", 0) + trade["pnl"]
            
            if trade["pnl"] > 0:
                metrics["winning_trades"] = metrics.get("winning_trades", 0) + 1
            
            if metrics["total_trades"] > 0:
                metrics["win_rate"] = (metrics["winning_trades"] / metrics["total_trades"]) * 100
                
            save_portfolio(portfolio)

def get_trade_history(limit: int = 100) -> List[Dict]:
    """Get recent trade history from history.json"""
    if not os.path.exists(HISTORY_FILE):
        return []
        
    try:
        with open(HISTORY_FILE, "r") as f:
            history = json.load(f)
        return history[-limit:]
    except Exception as e:
        logger.error(f"Error reading trade history: {e}")
        return []

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

def get_detailed_stats() -> Dict:
    """Calculate advanced trading metrics from history.json"""
    history = get_trade_history(limit=1000)
    if not history:
        return {"error": "No trade history found"}

    # Extract PnL from closed trades
    pnls = [t['pnl'] for t in history if t.get('action') == 'close' and 'pnl' in t]
    if not pnls:
        return {"error": "No closed trades with PnL data"}

    # 1. Profit Factor
    gross_profits = sum(p for p in pnls if p > 0)
    gross_losses = abs(sum(p for p in pnls if p < 0))
    profit_factor = gross_profits / gross_losses if gross_losses > 0 else float('inf')

    # 2. Sharpe Ratio (Simplified for Trade-by-Trade)
    # Risk-free rate assumed 0% for crypto simplicity
    avg_pnl = np.mean(pnls)
    std_pnl = np.std(pnls)
    sharpe = (avg_pnl / std_pnl) * np.sqrt(len(pnls)) if std_pnl > 0 else 0

    # 3. Drawdown Analysis
    cumulative_pnl = np.cumsum(pnls)
    peak = np.maximum.accumulate(cumulative_pnl)
    # Handle case where pnl is always negative/zero
    drawdowns = (peak - cumulative_pnl)
    max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0

    return {
        "total_trades": len(pnls),
        "profit_factor": round(profit_factor, 2),
        "sharpe_ratio": round(sharpe, 2),
        "max_drawdown_amount": round(max_drawdown, 2),
        "avg_trade_pnl": round(avg_pnl, 2),
        "win_rate": round((len([p for p in pnls if p > 0]) / len(pnls)) * 100, 2)
    }

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
