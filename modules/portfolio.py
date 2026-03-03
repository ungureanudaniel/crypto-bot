import json
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging
from datetime import datetime
from typing import Dict, Optional, List
from config_loader import config, get_binance_client

logger = logging.getLogger(__name__)

PORTFOLIO_FILE = "portfolio.json"

# -------------------------------------------------------------------
# CONFIG LOADING
# -------------------------------------------------------------------
CONFIG = config.config
TRADING_MODE = CONFIG.get('trading_mode', 'paper').lower()

# Global variables for live mode
_binance_client = None
_initial_total_value = None
_open_positions = {}  # This should come from trade_engine, but for standalone testing

# -------------------------------------------------------------------
# PAPER MODE: JSON file operations
# -------------------------------------------------------------------
def load_portfolio():
    """Load portfolio data"""
    if os.path.exists(PORTFOLIO_FILE):
        try:
            with open(PORTFOLIO_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading portfolio: {e}")
    
    # Default with $100 USDT, $0 USDC
    return {
        "holdings": {
            "USDT": 100,
            "USDC": 0
        },
        "positions": {},
        "trade_history": [],
        "performance_metrics": {
            "total_trades": 0,
            "winning_trades": 0,
            "total_pnl": 0.0,
            "win_rate": 0.0
        },
        "initial_balance": 100,
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
    
    save_portfolio(portfolio)
    return portfolio

def set_initial_balance(balance):
    """Set the initial balance (called on first sync)"""
    portfolio = load_portfolio()
    if portfolio.get("initial_balance") is None:
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
        "holdings": {
            "USDT": initial_balance,
            "USDC": 0
        },
        "positions": {},
        "trade_history": [],
        "performance_metrics": {
            "total_trades": 0,
            "winning_trades": 0,
            "total_pnl": 0.0,
            "win_rate": 0.0
        },
        "initial_balance": initial_balance,
        "last_updated": datetime.now().isoformat()
    }
    save_portfolio(portfolio)
    logger.info(f"✅ Paper portfolio reset to ${initial_balance:.2f}")
    return portfolio

def update_paper_balance(asset: str, amount: float, price: float, action: str, quote_currency: str = "USDT") -> Optional[float]:
    """Update paper portfolio holdings with specified quote currency"""
    if TRADING_MODE != 'paper':
        logger.warning("update_paper_balance only works in paper mode")
        return None
    
    portfolio = load_portfolio()
    
    # Ensure holdings exist
    if "holdings" not in portfolio:
        portfolio["holdings"] = {}
    
    # Ensure quote currency exists in holdings
    if quote_currency not in portfolio["holdings"]:
        portfolio["holdings"][quote_currency] = 0
    
    pnl = None

    if action == "buy":
        # Deduct quote currency
        cost = amount * price
        if portfolio["holdings"][quote_currency] < cost:
            logger.warning(f"Insufficient {quote_currency}: have ${portfolio['holdings'][quote_currency]:.2f}, need ${cost:.2f}")
            return None
        
        portfolio["holdings"][quote_currency] -= cost
        
        # Add bought asset
        if asset in portfolio["holdings"]:
            portfolio["holdings"][asset] += amount
        else:
            portfolio["holdings"][asset] = amount
            
        logger.info(f"✅ Paper buy: {amount:.6f} {asset} at ${price:.2f} using {quote_currency}")
        
    elif action == "sell":
        # Check if we have enough of the asset
        if asset not in portfolio["holdings"] or portfolio["holdings"][asset] < amount:
            logger.warning(f"Insufficient {asset} to sell")
            return None
        
        # Add quote currency from sale
        revenue = amount * price
        portfolio["holdings"][quote_currency] += revenue
        pnl = revenue  # Simplified PnL
        
        # Reduce asset holdings
        portfolio["holdings"][asset] -= amount
        if portfolio["holdings"][asset] <= 0:
            del portfolio["holdings"][asset]
            
        logger.info(f"✅ Paper sell: {amount:.6f} {asset} at ${price:.2f}, revenue: ${revenue:.2f} {quote_currency}")
    
    # Save
    save_portfolio(portfolio)
    return pnl

def get_current_prices() -> Dict[str, float]:
    """Get current prices for all symbols - mock for standalone testing"""
    # This is a mock function for testing
    # In real usage, this would come from data_feed
    return {
        "BTC/USDT": 65000.0,
        "ETH/USDT": 3500.0,
        "SOL/USDT": 150.0,
    }

# -------------------------------------------------------------------
# LIVE MODE: Exchange data fetching
# -------------------------------------------------------------------
def get_total_portfolio_value(client, symbols: List[str]) -> Dict:
    """Calculate total portfolio value in USDT, only pricing relevant assets."""
    if not client:
        return {'total_usdt': 0, 'cash_usdt': 0, 'holdings': {}}
    
    # Build set of base currencies we care about
    base_currencies = set()
    for sym in symbols:
        base = sym.split('/')[0]
        base_currencies.add(base)
    base_currencies.add('USDT')
    base_currencies.add('USDC')

    try:
        account = client.get_account()
        total_usdt = 0
        cash_usdt = 0
        holdings = {}

        for balance in account['balances']:
            asset = balance['asset']
            free = float(balance['free'])
            if free <= 0:
                continue

            # If asset is USDT, it's cash
            if asset == 'USDT':
                cash_usdt = free
                total_usdt += free
                holdings[asset] = free
                continue

            # If asset is USDC, approximate value (1:1 with USDT)
            if asset == 'USDC':
                total_usdt += free
                holdings[asset] = free
                continue

            # If asset is not in our base set, skip pricing (value 0)
            if asset not in base_currencies:
                logger.debug(f"Skipping {asset} (not in trading pairs)")
                holdings[asset] = {
                    'amount': free,
                    'value_usdt': 0,
                    'note': 'not priced (not in trading pairs)'
                }
                continue

            # Try to get price in USDT
            try:
                symbol = f"{asset}USDT"
                ticker = client.get_symbol_ticker(symbol=symbol)
                price = float(ticker['price'])
                value = free * price
                total_usdt += value
                holdings[asset] = {
                    'amount': free,
                    'price_usdt': price,
                    'value_usdt': value
                }
            except Exception as e:
                logger.debug(f"Could not price {asset}: {e}")
                holdings[asset] = {
                    'amount': free,
                    'value_usdt': 0,
                    'note': 'price fetch failed'
                }

        logger.info(f"💰 Total portfolio value: ${total_usdt:,.2f}")
        return {
            'total_usdt': total_usdt,
            'cash_usdt': cash_usdt,
            'holdings': holdings
        }

    except Exception as e:
        logger.error(f"Error calculating portfolio value: {e}")
        return {'total_usdt': 0, 'cash_usdt': 0, 'holdings': {}}

def get_portfolio_health(open_positions: Optional[Dict] = None) -> bool:
    """
    Simple portfolio health check
    Returns True if healthy, False if not
    """
    try:
        summary = get_portfolio_summary(open_positions=open_positions)
        
        # Define what "healthy" means
        # For example: return > -10% is healthy
        if summary.get('total_return_pct', 0) < -10:
            return False
        
        # Cash above $5 is healthy
        cash = summary.get('cash', {}).get('total', 0)
        if cash < 5:
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"Error in portfolio health check: {e}")
        return False
# -------------------------------------------------------------------
# UNIFIED PORTFOLIO SUMMARY
# -------------------------------------------------------------------
def get_portfolio_summary(open_positions: Optional[Dict] = None, binance_client=None, symbols: Optional[List[str]] = None) -> Dict:
    """
    Get comprehensive portfolio summary - total value of all holdings + positions
    Can be called from trade_engine with open_positions parameter
    """
    global _initial_total_value
    
    logger.info("💰 Getting portfolio summary...")
    
    # Use provided open_positions or fall back to empty dict
    if open_positions is None:
        open_positions = {}
    
    # PAPER MODE
    if TRADING_MODE == 'paper':
        try:
            portfolio = load_portfolio()
            perf = get_performance_summary()
            
            # Get all holdings from portfolio.json
            holdings = portfolio.get('holdings', {})
            
            # Start with cash (USDT + USDC)
            cash_usdt = holdings.get('USDT', 0)
            cash_usdc = holdings.get('USDC', 0)
            total_cash = cash_usdt + cash_usdc
            
            # Get current prices for all holdings
            current_prices = get_current_prices()
            
            # Calculate total value of all coins held
            holdings_value = 0
            detailed_holdings = {}
            
            for asset, amount in holdings.items():
                if asset in ['USDT', 'USDC']:
                    # Cash is already counted
                    detailed_holdings[asset] = {
                        'amount': amount,
                        'value_usd': amount
                    }
                else:
                    # It's a crypto holding - need current price
                    symbol = f"{asset}/USDT"
                    price = current_prices.get(symbol, 0)
                    
                    if price > 0:
                        value = amount * price
                        holdings_value += value
                        detailed_holdings[asset] = {
                            'amount': amount,
                            'price_usd': price,
                            'value_usd': value
                        }
                    else:
                        # Try USDC pair if USDT fails
                        symbol = f"{asset}/USDC"
                        price = current_prices.get(symbol, 0)
                        if price > 0:
                            value = amount * price
                            holdings_value += value
                            detailed_holdings[asset] = {
                                'amount': amount,
                                'price_usd': price,
                                'value_usd': value
                            }
                        else:
                            logger.debug(f"Could not get price for {asset}")
                            detailed_holdings[asset] = {
                                'amount': amount,
                                'price_usd': 0,
                                'value_usd': 0
                            }
            
            # Calculate value of open positions
            positions_value = 0
            positions_pnl = 0
            
            for symbol, position in open_positions.items():
                # Get current price
                current_price = current_prices.get(symbol, position.get('entry_price', 0))
                entry_price = position.get('entry_price', 0)
                amount = position.get('amount', 0)
                
                if entry_price > 0 and amount > 0:
                    current_value = amount * current_price
                    entry_value = amount * entry_price
                    positions_value += current_value
                    positions_pnl += (current_value - entry_value)
            
            # TOTAL PORTFOLIO VALUE = cash + all holdings
            total_value = total_cash + holdings_value
            
            # Get initial balance
            initial_balance = portfolio.get('initial_balance', 100)
            
            total_return = total_value - initial_balance
            total_return_pct = (total_return / initial_balance * 100) if initial_balance > 0 else 0
            
            result = {
                'trading_mode': 'paper',
                'total_value': total_value,
                'cash': {
                    'USDT': cash_usdt,
                    'USDC': cash_usdc,
                    'total': total_cash
                },
                'holdings': detailed_holdings,
                'holdings_value': holdings_value,
                'positions_count': len(open_positions),
                'positions_value': positions_value,
                'positions_pnl': positions_pnl,
                'initial_balance': initial_balance,
                'total_return': total_return,
                'total_return_pct': total_return_pct,
                'total_trades': perf.get('total_trades', 0),
                'winning_trades': perf.get('winning_trades', 0),
                'win_rate': perf.get('win_rate', 0),
                'total_pnl': perf.get('total_pnl', 0),
                'last_sync': datetime.now().isoformat()
            }
            
            logger.info(f"✅ Paper portfolio total: ${total_value:,.2f} (Cash: ${total_cash:.2f}, Holdings: ${holdings_value:.2f})")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error in paper portfolio summary: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                'trading_mode': 'paper',
                'total_value': 100,
                'cash': {'USDT': 100, 'USDC': 0, 'total': 100},
                'holdings': {},
                'holdings_value': 0,
                'positions_count': 0,
                'positions_value': 0,
                'positions_pnl': 0,
                'initial_balance': 100,
                'total_return': 0,
                'total_return_pct': 0,
                'total_trades': 0,
                'winning_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'last_sync': datetime.now().isoformat()
            }
    
    # LIVE/TESTNET MODE
    else:
        if not binance_client or not symbols:
            logger.error("Binance client and symbols required for live mode")
            return {
                'trading_mode': TRADING_MODE,
                'total_value': 0,
                'error': 'Missing client or symbols'
            }
        
        try:
            # Get current portfolio value from exchange
            portfolio = get_total_portfolio_value(binance_client, symbols)
            logger.info(f"   Portfolio value: ${portfolio.get('total_usdt', 0):,.2f}")
            
            # Calculate returns
            if _initial_total_value is None:
                _initial_total_value = portfolio.get('total_usdt', 0)
                logger.info(f"   Initial value set to: ${_initial_total_value:,.2f}")
            
            total_return = portfolio.get('total_usdt', 0) - _initial_total_value
            total_return_pct = (total_return / _initial_total_value * 100) if _initial_total_value > 0 else 0
            
            # Get performance metrics from history
            perf = get_performance_summary()
            
            result = {
                'trading_mode': TRADING_MODE,
                'total_value': portfolio.get('total_usdt', 0),
                'cash_balance': portfolio.get('cash_usdt', 0),
                'holdings': portfolio.get('holdings', {}),
                'initial_balance': _initial_total_value,
                'total_return': total_return,
                'total_return_pct': total_return_pct,
                'active_positions': len(open_positions),
                'total_trades': perf.get('total_trades', 0),
                'winning_trades': perf.get('winning_trades', 0),
                'win_rate': perf.get('win_rate', 0),
                'total_pnl': perf.get('total_pnl', 0),
                'last_sync': datetime.now().isoformat()
            }
            
            logger.info(f"✅ Portfolio summary: ${result['total_value']:,.2f}, {result['active_positions']} positions")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error in get_portfolio_summary: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                'trading_mode': TRADING_MODE,
                'total_value': 0,
                'cash_balance': 0,
                'holdings': {},
                'initial_balance': _initial_total_value or 0,
                'total_return': 0,
                'total_return_pct': 0,
                'active_positions': len(open_positions),
                'total_trades': 0,
                'winning_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'last_sync': datetime.now().isoformat(),
                'error': str(e)
            }

# -------------------------------------------------------------------
# Test
# -------------------------------------------------------------------
if __name__ == "__main__":
    print(f"📊 Portfolio module - Mode: {TRADING_MODE.upper()}")
    
    if TRADING_MODE == 'paper':
        reset_paper_portfolio(100)
        
        # Test buy/sell
        update_paper_balance("BTC", 0.001, 68765, "buy")
        update_paper_balance("ETH", 0.1, 3000, "buy")
        update_paper_balance("BTC", 0.001, 69000, "sell")
        
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
        
        summary = get_portfolio_summary(open_positions={})
        print(f"\n💰 Paper Portfolio:")
        print(f"Total Value: ${summary['total_value']:.2f}")
        print(f"Cash: ${summary['cash']['total']:.2f}")
        print(f"Holdings: {summary['holdings']}")
        print(f"Return: {summary['total_return_pct']:.1f}%")
        print(f"Win Rate: {summary['win_rate']:.1f}%")
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