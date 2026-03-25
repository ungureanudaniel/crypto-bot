"""
futures_engine.py
=================
Handles execution of futures positions (shorts and futures longs).

- Paper mode:  fully simulated — positions tracked in portfolio.json
- Live/testnet: routes to Binance USDT-M Futures API via get_futures_client()

All short signals from strategy_tools come here instead of trade_engine.
"""
import logging
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Optional, Dict

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
try:
    from config_loader import config, get_futures_client
    CONFIG = config.config
except ImportError:
    CONFIG = {'trading_mode': 'paper', 'futures_leverage': 1, 'futures_fee': 0.0005}
    get_futures_client = lambda: None

TRADING_MODE = CONFIG.get('trading_mode', 'paper')
DEFAULT_LEVERAGE = int(CONFIG.get('futures_leverage', 1))
FUTURES_FEE = float(CONFIG.get('futures_fee', 0.0005))


# -------------------------------------------------------------------
# Symbol helpers
# -------------------------------------------------------------------
def _to_futures_symbol(symbol: str) -> str:
    """Convert BTC/USDT -> BTCUSDT for Binance Futures API"""
    return symbol.replace('/', '')


# -------------------------------------------------------------------
# PAPER MODE — Simulation
# -------------------------------------------------------------------
def _paper_open(symbol: str, side: str, amount: float, entry_price: float,
                stop_loss: float, take_profit: float,
                leverage: int = DEFAULT_LEVERAGE, atr: float = 0.0,
                trailing_min: Optional[float] = None,
                trailing_max: Optional[float] = None) -> bool:
    """Simulate opening a futures position in paper mode."""
    from modules.portfolio import open_futures_position
    fee = amount * entry_price * FUTURES_FEE
    net_entry = entry_price * (1 + FUTURES_FEE) if side == 'long' else entry_price * (1 - FUTURES_FEE)

    success = open_futures_position(
        symbol=symbol,
        side=side,
        amount=amount,
        entry_price=net_entry,
        stop_loss=stop_loss,
        take_profit=take_profit,
        leverage=leverage,
        atr=atr,
        trailing_min_pct=trailing_min,
        trailing_max_pct=trailing_max
    )
    if success:
        logger.info(f"📄 [PAPER] Futures {side.upper()} opened: {symbol} "
                    f"@ ${net_entry:.4f} x{leverage} | Fee: ${fee:.4f}")
    return success


def _paper_close(symbol: str, exit_price: float, reason: str = "") -> Optional[Dict]:
    """Simulate closing a futures position in paper mode."""
    from modules.portfolio import close_futures_position
    result = close_futures_position(symbol, exit_price, reason)
    if result:
        logger.info(f"📄 [PAPER] Futures closed: {symbol} @ ${exit_price:.4f} "
                    f"| PnL: ${result['pnl']:+.4f} ({result['pnl_pct']:+.2f}%)")
    return result


def _paper_check_stops() -> list:
    """
    Check all open paper futures positions using exit_manager
    (trailing stop + signal reversal) plus plain SL/TP fallback.
    Returns list of symbols that were closed.
    """
    from modules.portfolio import get_futures_positions, save_futures_positions
    from modules.data_feed import get_current_price

    try:
        from modules.exit_manager import evaluate_exit
    except ImportError:
        evaluate_exit = None

    closed = []
    positions = get_futures_positions()

    for symbol, pos in list(positions.items()):
        try:
            current_price = get_current_price(symbol)
            if current_price is None:
                continue

            should_exit = False
            reason      = ''

            if evaluate_exit:
                try:
                    from modules.data_feed import fetch_ohlcv
                    df = fetch_ohlcv(symbol, interval='1h', limit=100)
                except Exception:
                    df = None

                should_exit, reason = evaluate_exit(symbol, pos, current_price, df)

                # Persist trailing stop update back to portfolio
                if not should_exit and pos.get('trailing_stop_active'):
                    positions[symbol] = pos
                    save_futures_positions(positions)

            else:
                side = pos['side']
                sl   = pos.get('stop_loss')
                tp   = pos.get('take_profit')
                if side == 'short':
                    if sl and current_price >= sl:
                        should_exit, reason = True, 'stop_loss'
                    elif tp and current_price <= tp:
                        should_exit, reason = True, 'take_profit'
                else:
                    if sl and current_price <= sl:
                        should_exit, reason = True, 'stop_loss'
                    elif tp and current_price >= tp:
                        should_exit, reason = True, 'take_profit'

            if should_exit:
                logger.info(f"🚪 Futures exit: {symbol} @ ${current_price:.4f} | {reason}")
                _paper_close(symbol, current_price, reason=reason)
                closed.append(symbol)

        except Exception as e:
            logger.error(f"Error checking stops for futures {symbol}: {e}")

    return closed


# -------------------------------------------------------------------
# LIVE / TESTNET MODE — Real Binance Futures
# -------------------------------------------------------------------
def _live_set_leverage(client, symbol: str, leverage: int) -> bool:
    """Set leverage for a futures symbol."""
    fsymbol = _to_futures_symbol(symbol)
    try:
        # UMFutures client
        if hasattr(client, 'change_leverage'):
            client.change_leverage(symbol=fsymbol, leverage=leverage)
        else:
            # python-binance fallback
            client.futures_change_leverage(symbol=fsymbol, leverage=leverage)
        logger.info(f"⚙️ Leverage set to {leverage}x for {fsymbol}")
        return True
    except Exception as e:
        logger.warning(f"Could not set leverage for {fsymbol}: {e}")
        return False


def _live_open(symbol: str, side: str, amount: float,
               stop_loss: float, take_profit: float,
               leverage: int = DEFAULT_LEVERAGE,
               trailing_min_pct: Optional[float] = None,
               trailing_max_pct: Optional[float] = None) -> bool:
    """Open a real futures position on Binance."""
    client = get_futures_client()
    if not client:
        logger.error("❌ No futures client available")
        return False

    fsymbol = _to_futures_symbol(symbol)
    binance_side = 'BUY' if side == 'long' else 'SELL'

    try:
        _live_set_leverage(client, symbol, leverage)

        # Place market order
        if hasattr(client, 'new_order'):
            # UMFutures
            order = client.new_order(
                symbol=fsymbol,
                side=binance_side,
                type='MARKET',
                quantity=round(amount, 6)
            )
        else:
            # python-binance fallback
            order = client.futures_create_order(
                symbol=fsymbol,
                side=binance_side,
                type='MARKET',
                quantity=round(amount, 6)
            )

        logger.info(f"✅ [LIVE] Futures {side.upper()} order placed: {fsymbol} "
                    f"| Amount: {amount} | OrderID: {order.get('orderId')}")

        # Record in portfolio
        fill_price = float(order.get('avgPrice') or order.get('price', 0))
        from modules.portfolio import open_futures_position
        open_futures_position(
            symbol=symbol,
            side=side,
            amount=amount,
            entry_price=fill_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            leverage=leverage,
            trailing_min_pct=trailing_min_pct,
            trailing_max_pct=trailing_max_pct
        )
        return True

    except Exception as e:
        logger.error(f"❌ Failed to open live futures {side} for {symbol}: {e}")
        return False


def _live_close(symbol: str, side: str, amount: float, reason: str = "") -> Optional[Dict]:
    """Close a real futures position on Binance."""
    client = get_futures_client()
    if not client:
        return None

    fsymbol = _to_futures_symbol(symbol)
    # To close: if we're short we BUY back, if we're long we SELL
    close_side = 'BUY' if side == 'short' else 'SELL'

    try:
        if hasattr(client, 'new_order'):
            order = client.new_order(
                symbol=fsymbol,
                side=close_side,
                type='MARKET',
                quantity=round(amount, 6),
                reduceOnly=True
            )
        else:
            order = client.futures_create_order(
                symbol=fsymbol,
                side=close_side,
                type='MARKET',
                quantity=round(amount, 6),
                reduceOnly=True
            )

        fill_price = float(order.get('avgPrice') or order.get('price', 0))
        logger.info(f"✅ [LIVE] Futures closed: {fsymbol} @ ${fill_price:.4f}")

        from modules.portfolio import close_futures_position
        return close_futures_position(symbol, fill_price, reason)

    except Exception as e:
        logger.error(f"❌ Failed to close live futures {side} for {symbol}: {e}")
        return None


def _live_check_stops() -> list:
    """
    For live mode, Binance handles SL/TP server-side if set as orders.
    This method syncs our local portfolio with any positions closed on exchange.
    """
    client = get_futures_client()
    if not client:
        return []

    from modules.portfolio import get_futures_positions, close_futures_position

    closed = []
    positions = get_futures_positions()

    for symbol in list(positions.keys()):
        fsymbol = _to_futures_symbol(symbol)
        try:
            if hasattr(client, 'get_position_risk'):
                pos_info = client.get_position_risk(symbol=fsymbol)
            else:
                pos_info = client.futures_position_information(symbol=fsymbol)

            # If exchange shows 0 position, it was closed server-side
            for p in pos_info:
                if float(p.get('positionAmt', 1)) == 0:
                    mark_price = float(p.get('markPrice', 0))
                    close_futures_position(symbol, mark_price, reason='exchange_closed')
                    closed.append(symbol)
                    break

        except Exception as e:
            logger.debug(f"Could not sync futures position for {symbol}: {e}")

    return closed


# -------------------------------------------------------------------
# PUBLIC API — used by trade_engine and scheduler
# -------------------------------------------------------------------
class FuturesEngine:
    """
    Public interface for futures trading.
    Automatically routes to paper simulation or live Binance based on trading mode.
    """

    def __init__(self):
        self.trading_mode = TRADING_MODE
        self.leverage = DEFAULT_LEVERAGE
        logger.info(f"🔮 FuturesEngine initialized | Mode: {self.trading_mode} | "
                    f"Leverage: {self.leverage}x")

    def open_short(self, symbol: str, amount: float, entry_price: float,
               stop_loss: float, take_profit: float,
               signal_type: str = '', atr: float = 0.0) -> bool:
        """Open a short position (futures sell)."""
        logger.info(f"📉 Opening SHORT: {symbol} | Amount: {amount:.6f} "
                    f"@ ${entry_price:.2f} | SL: ${stop_loss:.2f} | TP: ${take_profit:.2f}")

        # Fetch per‑pair trailing bounds
        trailing_min = trailing_max = None
        try:
            from config_loader import get_pair_config
            pair_cfg = get_pair_config(symbol)
            trailing_min = pair_cfg.get('trailing_min_pct')
            trailing_max = pair_cfg.get('trailing_max_pct')
        except Exception:
            pass

        if self.trading_mode == 'paper':
            ok = _paper_open(symbol, 'short', amount, entry_price,
                            stop_loss, take_profit, self.leverage, atr,
                            trailing_min, trailing_max)
            if ok and signal_type:
                from modules.portfolio import get_futures_positions, save_futures_positions
                pos = get_futures_positions()
                if symbol in pos:
                    pos[symbol]['signal_type'] = signal_type
                    pos[symbol]['trailing_stop_active'] = False
                    pos[symbol]['atr'] = atr
                    save_futures_positions(pos)
            return ok
        else:
            return _live_open(symbol, 'short', amount, stop_loss, take_profit,
                            self.leverage, trailing_min, trailing_max)

    def open_long(self, symbol: str, amount: float, entry_price: float,
              stop_loss: float, take_profit: float) -> bool:
        """Open a futures long position (alternative to spot long)."""
        logger.info(f"📈 Opening FUTURES LONG: {symbol} | Amount: {amount:.6f} "
                    f"@ ${entry_price:.2f}")

        # Fetch per‑pair trailing bounds (optional)
        trailing_min = trailing_max = None
        try:
            from config_loader import get_pair_config
            pair_cfg = get_pair_config(symbol)
            trailing_min = pair_cfg.get('trailing_min_pct')
            trailing_max = pair_cfg.get('trailing_max_pct')
        except Exception:
            pass

        if self.trading_mode == 'paper':
            return _paper_open(symbol, 'long', amount, entry_price,
                            stop_loss, take_profit, self.leverage, atr=0.0,
                            trailing_min=trailing_min, trailing_max=trailing_max)
        else:
            return _live_open(symbol, 'long', amount, stop_loss, take_profit,
                            self.leverage, trailing_min, trailing_max)

    def close_position(self, symbol: str, exit_price: Optional[float] = None,
                       reason: str = "") -> Optional[Dict]:
        """Close any open futures position for a symbol."""
        from modules.portfolio import get_futures_positions
        from modules.data_feed import get_current_price

        pos = get_futures_positions().get(symbol)
        if not pos:
            logger.warning(f"No futures position to close for {symbol}")
            return None

        if exit_price is None:
            exit_price = get_current_price(symbol) or pos['entry_price']

        if self.trading_mode == 'paper':
            return _paper_close(symbol, exit_price, reason)
        return _live_close(symbol, pos['side'], pos['amount'], reason)

    def check_stops(self) -> list:
        """
        Check and execute stop-loss / take-profit for all open futures positions.
        Called every minute by the scheduler.
        Returns list of closed symbols.
        """
        if self.trading_mode == 'paper':
            return _paper_check_stops()
        return _live_check_stops()

    def get_open_positions(self) -> Dict:
        """Return all open futures positions from portfolio."""
        from modules.portfolio import get_futures_positions
        return get_futures_positions()

    def has_position(self, symbol: str) -> bool:
        """Check if a futures position is open for a symbol."""
        return symbol in self.get_open_positions()


# Global singleton
futures_engine = FuturesEngine()


# -------------------------------------------------------------------
# Quick test
# -------------------------------------------------------------------
if __name__ == "__main__":
    print("🧪 FuturesEngine test")
    print(f"Mode: {futures_engine.trading_mode}")
    print(f"Leverage: {futures_engine.leverage}x")

    # Simulate a paper short
    ok = futures_engine.open_short(
        symbol="BTC/USDT",
        amount=0.001,
        entry_price=65000,
        stop_loss=66300,
        take_profit=62000
    )
    print(f"Short opened: {ok}")

    positions = futures_engine.get_open_positions()
    print(f"Open futures positions: {list(positions.keys())}")

    # Close it
    result = futures_engine.close_position("BTC/USDT", exit_price=63000, reason="test")
    if result:
        print(f"Closed with PnL: ${result['pnl']:+.2f}")
