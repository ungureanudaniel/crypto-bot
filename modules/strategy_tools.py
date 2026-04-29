import sys
import os
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
import logging
import time
from typing import Dict, Optional
from ta.volatility import BollingerBands
from ta.trend import MACD, ADXIndicator
from ta.momentum import RSIIndicator
from modules.data_feed import fetch_ohlcv
from modules.indicators import rsi_signal, bollinger_band_signal, macd_bar_exhaustion_signal, breakout_signal, volume_breakout_signal, calculate_atr
from config_loader import config as _cfg

logger = logging.getLogger(__name__)
# IMPROVEMENT: allow enabling daily trend filter via config
_TRADING_FEE = float(_cfg.config.get('trading_fee', 0.0005))
_SL_MIN_PCT = float(_cfg.config.get('stop_loss_min_pct', 0.015))
_SL_MAX_PCT = float(_cfg.config.get('stop_loss_max_pct', 0.08))
_DEFAULT_SL_PCT = float(_cfg.config.get('default_stop_loss_pct', 0.03))
_MIN_RR = float(_cfg.config.get('min_rr', 2.0))

use_volume_shrinkage   = _cfg.config.get('use_volume_shrinkage', False)
use_daily_trend_filter = _cfg.config.get('use_daily_trend_filter', True)

_trend_cache: dict = {}
# -------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------


def get_available_balance(symbol, trading_engine):
    """
    Get available balance for position sizing.
    For shorts (futures): returns available USDT cash margin.
    For longs (spot): returns available USDT cash.
    """
    if not trading_engine:
        return 0

    # PAPER MODE: use portfolio cash
    if trading_engine.trading_mode == 'paper':
        try:
            from modules.portfolio import get_cash
            cash = get_cash()
            return cash.get('USDT', 0) + cash.get('USDC', 0)
        except Exception as e:
            logger.debug(f"Could not get paper cash balance: {e}")
            return 0

    # LIVE/TESTNET MODE: Check real exchange balance
    if not trading_engine.binance_client:
        return 0

    try:
        account = trading_engine.binance_client.get_account()
        for balance in account['balances']:
            if balance['asset'] == 'USDT':
                return float(balance['free'])
    except Exception as e:
        logger.debug(f"Could not get live balance: {e}")

    return 0

# -------------------------------------------------------------------
# Position Sizing (Only calculates, but execution in trade_engine.py)
# -------------------------------------------------------------------
def calculate_position_units(entry_price, equity, risk_per_trade=0.02, atr=None,
                              stop_atr_multiplier: float = 2.0, trading_fee: float = _TRADING_FEE,
                              side: str = 'long',
                              stop_loss_min_pct: Optional[float] = None,
                              stop_loss_max_pct: Optional[float] = None,
                              default_stop_loss_pct: Optional[float] = None,
                              min_rr: Optional[float] = None):
    try:
        # Load config with fallbacks
        min_pct = stop_loss_min_pct if stop_loss_min_pct is not None else _SL_MIN_PCT
        max_pct = stop_loss_max_pct if stop_loss_max_pct is not None else _SL_MAX_PCT
        def_pct = default_stop_loss_pct if default_stop_loss_pct is not None else _DEFAULT_SL_PCT
        rr = min_rr if min_rr is not None else _MIN_RR

        # Calculate stop loss distance from ATR
        if atr and atr > 0:
            stop_distance = atr * stop_atr_multiplier
            stop_loss_pct = stop_distance / entry_price
        else:
            stop_loss_pct = def_pct

        # Apply bounds
        stop_loss_pct = max(stop_loss_pct, min_pct)
        stop_loss_pct = min(stop_loss_pct, max_pct)

        # Fee cost for round trip
        round_trip_fee = trading_fee * 2

        # TP: risk:reward × stop
        take_profit_pct = max(stop_loss_pct * rr, round_trip_fee * 5)
        take_profit_pct = max(take_profit_pct, 0.02)

        # Direction‑aware SL/TP
        if side == 'short':
            stop_loss_price   = entry_price * (1 + stop_loss_pct)
            take_profit_price = entry_price * (1 - take_profit_pct)
        else:
            stop_loss_price   = entry_price * (1 - stop_loss_pct)
            take_profit_price = entry_price * (1 + take_profit_pct)

        # Fee-adjusted net profit check — skip if fees eat >20% of gain
        risk_amount   = equity * risk_per_trade
        risk_per_unit = entry_price * stop_loss_pct
        if risk_per_unit <= 0:
            logger.warning("Risk per unit is zero or negative")
            return 0, None, None
        max_units      = (equity * 0.15) / entry_price   # ← computed once
        units          = min(risk_amount / risk_per_unit, max_units)  # ← capped immediately
        position_value  = units * entry_price
        actual_fee_cost = position_value * round_trip_fee
        expected_gain   = position_value * stop_loss_pct * rr
        if actual_fee_cost > expected_gain * 0.20:
            logger.debug(f"⏭Fee ratio too high: fees ${actual_fee_cost:.4f} vs gain ${expected_gain:.4f}")
            return 0, None, None


        # Position sizing
        units = risk_amount / risk_per_unit

        # Cap at 15% of equity per position
        max_units = (equity * 0.15) / entry_price
        units = min(units, max_units)

        # Minimum trade size ($10)
        min_units = 10 / entry_price
        if units < min_units:
            logger.debug(f"Position too small: {units:.6f}, required: {min_units:.6f}")
            return 0, None, None

        logger.info(
            f"Position calc [{side.upper()}]: Entry=${entry_price:.4f}, "
            f"SL=${stop_loss_price:.4f} ({stop_loss_pct:.2%}), "
            f"TP=${take_profit_price:.4f} ({take_profit_pct:.2%}), "
            f"R:R={take_profit_pct/stop_loss_pct:.1f}x, "
            f"Units={units:.6f}, Risk=${risk_amount:.2f}"
        )

        return units, stop_loss_price, take_profit_price

    except Exception as e:
        logger.error(f"Error in calculate_position_units: {e}")
        return 0, None, None

def detect_trend(df, symbol: str = "") -> tuple:
    cache_key = (symbol, df.index[-1] if not df.empty else None)
    if cache_key in _trend_cache:
        return _trend_cache[cache_key]
    try:
        close  = df['close']
        ema20  = close.ewm(span=20, adjust=False).mean()
        ema50  = close.ewm(span=50, adjust=False).mean()

        if ema20.iloc[-1] > ema50.iloc[-1]:
            direction = "up"
        elif ema20.iloc[-1] < ema50.iloc[-1]:
            direction = "down"
        else:
            direction = "side"

        adx      = ADXIndicator(df['high'], df['low'], close, window=14).adx()
        strength = min(float(adx.iloc[-1]) / 100.0, 1.0)
        ema200   = close.ewm(span=min(200, len(df) - 1), adjust=False).mean()
        long_dir = "up" if close.iloc[-1] > ema200.iloc[-1] else "down"

        confidence = 0.7 if direction == long_dir else 0.5
        if strength > 0.3:
            confidence += 0.2

        result = (direction, min(strength, 1.0), min(confidence, 0.95))
        _trend_cache[cache_key] = result
        return result
    except Exception as e:
        logger.debug(f"Trend detection error: {e}")
        return "side", 0.0, 0.5

# ===========================================================================
# Main signal Generator
# ===========================================================================
def generate_trade_signal(df, equity, risk_per_trade=0.02, symbol=None, trading_engine=None, regime=None):
    """
    Main function that combines multiple strategies.
    Enhanced with trend detection to filter counter‑trend signals.
    """
    try:
        if df.empty or len(df) < 50:
            logger.debug("Insufficient data")
            return None

        trend_dir, trend_strength, trend_conf = detect_trend(df, symbol=symbol or "")
        
        # Adjust risk based on trend strength
        if trend_strength > 0.4:
            risk_multiplier = 1.2
        elif trend_strength > 0.25:
            risk_multiplier = 1.0
        else:
            risk_multiplier = 0.6
        
        pair_config = {}
        if symbol:
            try:
                from config_loader import get_pair_config
                pair_config = get_pair_config(symbol)
            except Exception as e:
                logger.debug(f"Could not load pair config for {symbol}: {e}")

        per_pair_risk = pair_config.get('risk_per_trade')
        if per_pair_risk is not None:
            adjusted_risk = per_pair_risk * risk_multiplier
            logger.info(f"Using per‑pair risk for {symbol}: {per_pair_risk:.2%} (global {risk_per_trade:.2%})")
            logger.debug(f"Risk multiplier: {risk_multiplier:.2f}, Adjusted risk: {adjusted_risk:.2%}")
        else:
            adjusted_risk = risk_per_trade * risk_multiplier
            logger.debug(f"Risk multiplier: {risk_multiplier:.2f}, Adjusted risk: {adjusted_risk:.2%}")
        logger.debug(f"Trend: {trend_dir} | strength={trend_strength:.2f}")

        atr_series = calculate_atr(df)
        current_atr = atr_series.iloc[-1] if not atr_series.empty else 0
        current_price = df['close'].iloc[-1]

        regime_type = "unknown"
        trend_direction = "unknown"
        if regime:
            if "UPTREND" in regime:
                trend_direction = "up"
                regime = regime.replace(" UPTREND", "")
            elif "DOWNTREND" in regime:
                trend_direction = "down"
                regime = regime.replace(" DOWNTREND", "")
            elif "SIDEWAYS" in regime:
                trend_direction = "side"
                regime = regime.replace(" SIDEWAYS", "")

            if "Range" in regime:
                regime_type = "range"
            elif "Compression" in regime:
                regime_type = "compression"
            elif "Expansion" in regime:
                regime_type = "expansion"
            elif "Breakout" in regime:
                regime_type = "breakout"
            elif "Trend" in regime or "Trending" in regime:
                regime_type = "trend"

        signal = None
        signal_type = None
        multiplier = 2.0
        
        # ===== TRENDING MARKET =====
        if regime_type == "trend":
            logger.debug(f"Trending market - following {trend_direction} trend")
            
            signal = macd_bar_exhaustion_signal(df, 
                                                min_bars=4, 
                                                min_shrink_bars=3,
                                                use_rsi_confirm=True, 
                                                rsi_oversold=30, 
                                                rsi_overbought=70,
                                                use_volume_shrink=use_volume_shrinkage)
            
            if trend_direction == "up":
                if signal == 'long':
                    signal_type = "macd_bar_exhaustion_trend_up"
                    multiplier = 2.5
                else:
                    signal = None  # ✅ discard wrong-direction MACD before fallback
                    signal = breakout_signal(df)
                    if signal == 'long':
                        signal_type = "breakout_trend_up"
                        multiplier = 2.5
                    else:
                        signal = None  # ✅ discard if breakout also wrong direction

            elif trend_direction == "down":
                if signal == 'short':
                    signal_type = "macd_bar_exhaustion_trend_down"
                    multiplier = 2.5
                else:
                    signal = None  # ✅ discard wrong-direction MACD before fallback
                    signal = breakout_signal(df)
                    if signal == 'short':
                        signal_type = "macd_bar_exhaustion_trend_down"
                        multiplier = 2.5
                    else:
                        signal = None  # ✅ discard if breakout also wrong direction

            else:
                signal = None  # ✅ discard MACD signal, not relevant in sideways
                signal = rsi_signal(df)
                if signal:
                    signal_type = f"rsi_trend_side_{signal}"
                    multiplier = 1.5
                if not signal:
                    signal = bollinger_band_signal(df)
                    if signal:
                        signal_type = f"bollinger_trend_side_{signal}"
                        multiplier = 1.5

        # ===== RANGING MARKET ===== (unchanged)
        elif regime_type in ["range", "compression"]:
            logger.debug(f"Ranging market – favoring mean reversion")
            
            signal = macd_bar_exhaustion_signal(df, 
                                                min_bars=4, 
                                                min_shrink_bars=3,
                                                use_rsi_confirm=False, 
                                                use_volume_shrink=True)
            if signal:
                signal_type = f"macd_bar_exhaustion_range_{signal}"
                multiplier = 1.5 
            else:
                signal = rsi_signal(df)
                if signal:
                    signal_type = f"rsi_range_{signal}"
                    multiplier = 1.2
                if not signal:
                    signal = bollinger_band_signal(df)
                    if signal:
                        signal_type = f"bollinger_range_{signal}"
                        multiplier = 1.2

            if signal:
                fixed_tp_dist = current_atr * 2.0
                if signal == 'long':
                    take_profit_price = current_price + fixed_tp_dist
                else:
                    take_profit_price = current_price - fixed_tp_dist
                
                units, stop_loss, _ = calculate_position_units(
                    current_price, equity, adjusted_risk, current_atr, 
                    stop_atr_multiplier=multiplier, side=signal
                )
                
                if units > 0:
                    return {
                        'signal': signal,
                        'symbol': symbol,
                        'units': units,
                        'entry_price': current_price,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit_price,
                        'type': signal_type,
                        'regime': regime,
                        'exit_strategy': 'fixed'
                    }

        # ===== BREAKOUT MARKET ===== (unchanged)
        elif regime_type == "breakout":
            logger.debug(f"Breakout market")
            
            signal = macd_bar_exhaustion_signal(df, 
                                                min_bars=4, 
                                                min_shrink_bars=3,
                                                use_rsi_confirm=False, 
                                                use_volume_shrink=True)
            if signal:
                signal_dir = 'up' if signal == 'long' else 'down'
                if signal_dir == trend_dir or trend_strength < 0.2:
                    signal_type = f"macd_bar_exhaustion_breakout_{signal}"
                    multiplier = 2.5
                    logger.debug(f"Breakout MACD Bar Exhaustion {signal}")
                else:
                    signal = None
            if not signal:
                signal = breakout_signal(df)
                if signal:
                    signal_dir = 'up' if signal == 'long' else 'down'
                    if signal_dir == trend_dir or trend_strength < 0.2:
                        signal_type = f"breakout_{signal}"
                        multiplier = 3.0
                        logger.debug(f"Breakout {signal}")
                    else:
                        signal = None
            if not signal:
                signal = volume_breakout_signal(df)
                if signal:
                    signal_dir = 'up' if signal == 'long' else 'down'
                    if signal_dir == trend_dir or trend_strength < 0.2:
                        signal_type = f"volume_breakout_{signal}"
                        multiplier = 2.5
                        logger.debug(f"Volume breakout {signal}")
                    else:
                        signal = None

        # ===== EXPANSION MARKET ===== (unchanged)
        elif regime_type == "expansion":
            logger.debug(f"High volatility – cautious")
            adjusted_risk = risk_per_trade * 0.5
            
            signal = macd_bar_exhaustion_signal(df, 
                                                min_bars=4, 
                                                min_shrink_bars=3,
                                                use_rsi_confirm=True, 
                                                use_volume_shrink=True)
            if signal:
                signal_dir = 'up' if signal == 'long' else 'down'
                if signal_dir == trend_dir:
                    signal_type = f"macd_bar_exhaustion_expansion_{signal}"
                    multiplier = 2.0
                    logger.debug(f"Expansion MACD Bar Exhaustion {signal}")
                else:
                    signal = None
            if not signal:
                signal = breakout_signal(df, volume_confirmation=True)
                if signal:
                    signal_dir = 'up' if signal == 'long' else 'down'
                    if signal_dir == trend_dir:
                        signal_type = f"breakout_expansion_{signal}"
                        multiplier = 3.0
                        logger.debug(f"Breakout in high volatility with trend")
                    else:
                        signal = None
            if not signal:
                signal = volume_breakout_signal(df)
                if signal and abs(df['close'].pct_change().iloc[-1]) > 0.02:
                    signal_dir = 'up' if signal == 'long' else 'down'
                    if signal_dir == trend_dir:
                        signal_type = f"volume_expansion_{signal}"
                        multiplier = 2.5
                        logger.debug(f"Volume breakout in high volatility")
                    else:
                        signal = None

        # ===== UNKNOWN REGIME ===== (unchanged)
        else:
            logger.debug(f"Unknown regime – trying all strategies")
            for fn, stype, mult in [
                (lambda d: macd_bar_exhaustion_signal(d, min_bars=4, min_shrink_bars=3, use_volume_shrink=True), "macd_bar_exhaustion_unknown", 2.0),
                (breakout_signal, "breakout_unknown", 2.0),
                (rsi_signal, "rsi_unknown", 1.5),
                (bollinger_band_signal, "bollinger_unknown", 1.5),
                (volume_breakout_signal, "volume_unknown", 2.0),
            ]:
                sig = fn(df)
                if sig:
                    signal_dir = 'up' if sig == 'long' else 'down'
                    if trend_strength > 0.25 and signal_dir != trend_dir and trend_dir != 'side':
                        logger.debug(f"Skipping {stype} - counter-trend")
                        continue
                    signal = sig
                    signal_type = stype
                    multiplier = mult
                    break

        # ===== FINAL TREND VALIDATION & PRICING ===== (unchanged)
        if signal:
            signal_dir = 'up' if signal == 'long' else 'down'
            
            min_rr = _MIN_RR

            if signal == 'long':
                sl_price = current_price - (current_atr * multiplier)
                risk_distance = current_price - sl_price
                tp_price = current_price + (risk_distance * min_rr)
            else:
                sl_price = current_price + (current_atr * multiplier)
                risk_distance = sl_price - current_price
                tp_price = current_price - (risk_distance * min_rr)

            available_balance = get_available_balance(symbol, trading_engine) or equity

            units, sl_price, tp_price = calculate_position_units(
                entry_price=current_price,
                equity=available_balance,
                risk_per_trade=adjusted_risk,
                atr=current_atr,
                stop_atr_multiplier=multiplier,
                side=signal
            )
            
            if units > 0:
                return {
                    'symbol': symbol,
                    'side': signal,
                    'signal_type': signal_type,
                    'units': units,
                    'entry_price': current_price,
                    'stop_loss': sl_price,
                    'take_profit': tp_price,
                    'risk_pct': adjusted_risk,
                    'regime': regime_type,
                    'atr': current_atr
                }

        logger.debug("No valid signals or insufficient units")
        return None

    except Exception as e:
        logger.error(f"Error in generate_trade_signal: {e}", exc_info=True)
        return None

def get_confidence_from_regime(regime_type, signal_type):
    """Calculate confidence based on regime and signal type"""
    # Base confidence
    confidence = 70
    
    # Boost confidence when strategy matches regime
    if regime_type == "range" and "rsi" in signal_type:
        confidence += 20
    elif regime_type == "range" and "bollinger" in signal_type:
        confidence += 15
    elif regime_type == "trend" and ("ema" in signal_type or "macd" in signal_type):
        confidence += 20
    elif regime_type == "breakout" and "breakout" in signal_type:
        confidence += 25
    elif regime_type == "expansion" and "breakout" in signal_type:
        confidence += 10  # Still good but risky
    
    # Cap at 95
    return min(confidence, 95)
    
if __name__ == "__main__":
    import sys
    import os
    from datetime import datetime, timedelta
    import time
    
    print("=" * 60)
    print("🧪 STRATEGY TOOLS COMPREHENSIVE TEST")
    print("=" * 60)
    
    # Test configuration
    test_symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    test_timeframes = ["1h", "4h"]
    test_equity = 10000
    
    # Create a mock trading engine for balance tests
    class MockTradingEngine:
        def __init__(self):
            self.trading_mode = "paper"
            self.binance_client = None
    
    mock_engine = MockTradingEngine()
    
    print(f"\n📊 Test Parameters:")
    print(f"   Symbols: {test_symbols}")
    print(f"   Timeframes: {test_timeframes}")
    print(f"   Equity: ${test_equity}")
    print(f"   Mode: {mock_engine.trading_mode}")
    
    # Test 1: Individual strategy functions
    print("\n" + "=" * 60)
    print("📈 TEST 1: Individual Strategy Functions")
    print("=" * 60)
    
    for symbol in test_symbols:
        print(f"\n🔍 Testing {symbol}...")
        
        for interval in test_timeframes:
            print(f"   ⏱️  Timeframe: {interval}")
            
            # Fetch data
            from modules.data_feed import fetch_ohlcv
            df = fetch_ohlcv(symbol, interval=interval, limit=200)
            
            if df.empty or len(df) < 50:
                print(f"      ❌ Insufficient data ({len(df)} candles)")
                continue
            
            print(f"      ✅ Data fetched: {len(df)} candles")
            print(f"      Latest price: ${df['close'].iloc[-1]:.2f}")
            
            # Test each strategy
            strategies = [
                ("Breakout", lambda: breakout_signal(df)),
                ("RSI", lambda: rsi_signal(df)),
                ("MACD Bar Exhaustion", lambda: macd_bar_exhaustion_signal(df)),
                ("Bollinger", lambda: bollinger_band_signal(df)),
                ("Volume Breakout", lambda: volume_breakout_signal(df))
            ]
            
            for name, func in strategies:
                try:
                    signal = func()
                    if signal:
                        print(f"      ✅ {name}: {signal.upper()}")
                    else:
                        print(f"      ⚪ {name}: No signal")
                except Exception as e:
                    print(f"      ❌ {name} error: {e}")
            
            # Small delay to avoid rate limiting
            time.sleep(0.5)
    
    # Test 2: Position sizing
    print("\n" + "=" * 60)
    print("💰 TEST 2: Position Sizing Calculator")
    print("=" * 60)
    
    test_prices = [50000, 3000, 100, 10, 1]
    test_atr_values = [1000, 100, 10, 1, 0.1]
    
    for i, price in enumerate(test_prices):
        atr = test_atr_values[i] if i < len(test_atr_values) else price * 0.02
        
        print(f"\n📊 Entry Price: ${price:.2f}, ATR: ${atr:.2f}")
        
        units, sl, tp = calculate_position_units(
            entry_price=price,
            equity=test_equity,
            risk_per_trade=0.02,
            atr=atr,
            stop_atr_multiplier=2
        )
        
        if units > 0 and sl is not None and tp is not None:
            print(f"   ✅ Units: {units:.6f}")
            print(f"   ✅ Stop Loss: ${sl:.2f} ({(1-sl/price)*100:.1f}% loss)")
            print(f"   ✅ Take Profit: ${tp:.2f} ({(tp/price-1)*100:.1f}% gain)")
            print(f"   ✅ Position Value: ${units * price:.2f}")
            print(f"   ✅ Risk Amount: ${test_equity * 0.02:.2f}")
        else:
            print(f"   ⚪ Position too small (min ${10/price:.6f} units)")
    
    # Test 3: Full signal generation with balance awareness
    print("\n" + "=" * 60)
    print("🎯 TEST 3: Full Signal Generation with Balance Awareness")
    print("=" * 60)
    
    # We'll use the get_available_balance function to test different scenarios
    # by monkey-patching it temporarily
    
    original_get_available_balance = get_available_balance
    
    for symbol in test_symbols:
        print(f"\n🔍 Testing {symbol}...")
        
        # Fetch data
        from modules.data_feed import fetch_ohlcv
        df = fetch_ohlcv(symbol, interval="1h", limit=200)
        
        if df.empty:
            continue
        
        # Test with different balance scenarios by mocking the get_available_balance function
        balance_scenarios = [
            ("No balance (returns 0)", 0),
            ("Has balance (100 units)", 100),
            ("Small balance (10 units)", 10),
        ]
        
        for scenario_name, mock_balance in balance_scenarios:
            print(f"\n   📊 Scenario: {scenario_name}")
            # Mock the get_available_balance function for this test
            def mock_get_balance(sym, engine, balance=mock_balance):
                return balance
            
            # Temporarily replace the function
            import modules.strategy_tools
            modules.strategy_tools.get_available_balance = mock_get_balance
            
            try:
                signal = generate_trade_signal(
                    df=df,
                    equity=test_equity,
                    risk_per_trade=0.02,
                    symbol=symbol,
                    trading_engine=mock_engine
                )
                
                if signal:
                    print(f"      ✅ Signal generated!")
                    print(f"         Type: {signal.get('signal_type', 'unknown')}")
                    print(f"         Side: {signal.get('side', 'unknown')}")
                    print(f"         Entry: ${signal.get('entry', 0):.2f}")
                    print(f"         Units: {signal.get('units', 0):.6f}")
                    print(f"         Stop: ${signal.get('stop_loss', 0):.2f}")
                    print(f"         Target: ${signal.get('take_profit', 0):.2f}")
                else:
                    print(f"      ⚪ No signal generated")
            finally:
                # Restore original function
                modules.strategy_tools.get_available_balance = original_get_available_balance
    
    # Test 4: Multiple timeframes
    print("\n" + "=" * 60)
    print("⏱️  TEST 4: Multiple Timeframe Analysis")
    print("=" * 60)
    
    symbol = "BTC/USDT"
    print(f"\n🔍 Analyzing {symbol} across timeframes...")
    
    signals_by_tf = {}
    
    for interval in test_timeframes:
        from modules.data_feed import fetch_ohlcv
        df = fetch_ohlcv(symbol, interval=interval, limit=200)
        
        if df.empty:
            continue
        
        signal = generate_trade_signal(
            df=df,
            equity=test_equity,
            risk_per_trade=0.02,
            symbol=symbol,
            trading_engine=mock_engine
        )
        
        if signal:
            signals_by_tf[interval] = signal
            print(f"\n   ⏱️  {interval}: {signal['side'].upper()} {signal['signal_type']} at ${signal['entry']:.2f}")
        else:
            print(f"\n   ⏱️  {interval}: No signal")
    
    # Check for alignment across timeframes
    if len(signals_by_tf) > 1:
        sides = [s['side'] for s in signals_by_tf.values()]
        if all(side == sides[0] for side in sides):
            print(f"\n   ✅ All timeframes agree on {sides[0].upper()}!")
        else:
            print(f"\n   ⚠️  Timeframes show mixed signals: {sides}")
    
    # Test 5: Performance metrics
    print("\n" + "=" * 60)
    print("📊 TEST 5: Signal Generation Speed")
    print("=" * 60)
    
    symbol = "BTC/USDT"
    from modules.data_feed import fetch_ohlcv
    df = fetch_ohlcv(symbol, interval="1h", limit=200)
    
    if not df.empty:
        iterations = 10  # Reduced for speed
        start_time = time.time()
        
        for i in range(iterations):
            signal = generate_trade_signal(
                df=df,
                equity=test_equity,
                risk_per_trade=0.02,
                symbol=symbol,
                trading_engine=mock_engine
            )
        
        elapsed = time.time() - start_time
        print(f"\n   ⏱️  Generated {iterations} signals in {elapsed:.2f} seconds")
        print(f"   ⚡ Average: {elapsed/iterations*1000:.2f} ms per signal")
    
    # Test 6: Edge cases
    print("\n" + "=" * 60)
    print("⚠️  TEST 6: Edge Cases")
    print("=" * 60)
    
    from modules.data_feed import fetch_ohlcv
    df = fetch_ohlcv("BTC/USDT", interval="1h", limit=200)
    
    edge_cases = [
        ("Empty DataFrame", pd.DataFrame()),
        ("Insufficient data", pd.DataFrame({'close': [1,2,3]})),
        ("Zero equity", 0),
        ("Negative equity", -1000),
        ("Zero risk", 0),
    ]
    
    for case_name, case_data in edge_cases:
        print(f"\n   📋 Testing: {case_name}")
        
        if case_name == "Empty DataFrame":
            result = generate_trade_signal(case_data, test_equity, trading_engine=mock_engine)
        elif case_name == "Insufficient data":
            result = generate_trade_signal(case_data, test_equity, trading_engine=mock_engine)
        elif case_name == "Zero equity":
            result = generate_trade_signal(df, 0, trading_engine=mock_engine)
        elif case_name == "Negative equity":
            result = generate_trade_signal(df, -1000, trading_engine=mock_engine)
        elif case_name == "Zero risk":
            result = generate_trade_signal(df, test_equity, 0, trading_engine=mock_engine)
        
        if result is None:
            print(f"      ✅ Handled correctly (returned None)")
        else:
            print(f"      ⚠️  Returned signal: {result}")
    
    # Restore original function
    modules.strategy_tools.get_available_balance = original_get_available_balance
    
    print("\n" + "=" * 60)
    print("✅ STRATEGY TOOLS TEST COMPLETE")
    print("=" * 60)