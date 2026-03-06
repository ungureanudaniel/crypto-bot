import sys
import os
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.ml_integration import MLStrategy
import pandas as pd
import numpy as np
import logging
from typing import Dict
from modules.sentiment_agent import SentimentAgent
from ta.volatility import BollingerBands
from ta.trend import MACD
from ta.momentum import RSIIndicator
from modules.data_feed import fetch_ohlcv

logger = logging.getLogger(__name__)
ml_strategies = {}
sentiment_agent = None
# -------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------
async def get_ml_prediction(symbol: str, df: pd.DataFrame) -> Dict:
    """Get ML prediction for a symbol"""
    global ml_strategies
    
    if symbol not in ml_strategies:
        ml_strategies[symbol] = MLStrategy(symbol)
        await ml_strategies[symbol].ensure_trained()
    
    ml_strategy = ml_strategies[symbol]
    prediction = await ml_strategy.get_prediction(df)
    
    return prediction

def calculate_donchian(df, length=20):
    """Returns Donchian channel: high, low, mid"""
    high = df['high'].rolling(length, min_periods=1).max()
    low = df['low'].rolling(length, min_periods=1).min()
    mid = (high + low) / 2
    return high, low, mid

def calculate_atr(df, length=14):
    """Returns ATR series"""
    try:
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift(1))
        low_close = np.abs(df['low'] - df['close'].shift(1))
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(length, min_periods=1).mean()
        return atr
    except Exception as e:
        logger.error(f"Error calculating ATR: {e}")
        return pd.Series([0] * len(df), index=df.index)

def get_available_balance(symbol, trading_engine):
    """
    Get available balance for short positions.
    For shorts, we need to know how much of the base currency we have.
    """
    if not trading_engine:
        return 0
        
    base_currency = symbol.split('/')[0]

    # PAPER MODE: Check open positions for the asset
    if trading_engine.trading_mode == 'paper':
        try:
            # Look through open positions to find this asset
            available = 0
            for pos_symbol, position in trading_engine.open_positions.items():
                if pos_symbol.split('/')[0] == base_currency and position['side'] == 'long':
                    # If we already have a long position in this asset, we can use it for short
                    available = position.get('amount', 0)
                    break
            
            # Also check if we have the asset in cash? No, cash is only quote currencies
            # For paper mode, we only have what's in positions
            return available
            
        except Exception as e:
            logger.debug(f"Could not get paper balance for {base_currency}: {e}")
            return 0
    
    # LIVE/TESTNET MODE: Check real exchange balance
    if not trading_engine.binance_client:
        return 0
    
    try:
        account = trading_engine.binance_client.get_account()
        for balance in account['balances']:
            if balance['asset'] == base_currency:
                return float(balance['free'])
    except Exception as e:
        logger.debug(f"Could not get live balance for {base_currency}: {e}")
    
    return 0

# -------------------------------------------------------------------
# Signal Generation Functions
# -------------------------------------------------------------------

def breakout_signal(df, lookback=20, volume_confirmation=True):
    """
    Donchian breakout signal - MORE SENSITIVE
    Returns: 'long', 'short', or None
    """
    try:
        # Calculate Donchian channel
        highest_high = df['high'].rolling(lookback).max()
        lowest_low = df['low'].rolling(lookback).min()
        
        # Volume SMA
        volume_sma = df['volume'].rolling(20).mean()
        
        # Get current and previous values
        current_close = df['close'].iloc[-1]
        prev_close = df['close'].iloc[-2]
        prev_high = highest_high.iloc[-2]
        prev_low = lowest_low.iloc[-2]
        current_volume = df['volume'].iloc[-1]
        
        # LONG SIGNAL: Price breaks above previous period's high
        long_condition = current_close > prev_high
        
        # Add volume confirmation if requested
        if volume_confirmation:
            long_condition = long_condition and (current_volume > volume_sma.iloc[-2])
        
        # SHORT SIGNAL: Price breaks below previous period's low
        short_condition = current_close < prev_low
        
        if volume_confirmation:
            short_condition = short_condition and (current_volume > volume_sma.iloc[-2])
        
        if long_condition:
            return 'long'
        elif short_condition:
            return 'short'
        else:
            return None
            
    except Exception as e:
        logger.error(f"Error in breakout_signal: {e}")
        return None

def rsi_signal(df, oversold=25, overbought=75):
    """
    RSI mean reversion signal
    Returns: 'long', 'short', or None
    """
    try:
        # Calculate RSI
        rsi_indicator = RSIIndicator(df['close'], window=14)
        rsi = rsi_indicator.rsi()
        
        current_rsi = rsi.iloc[-1]
        
        if pd.isna(current_rsi):
            return None
        
        # LONG: Oversold
        if current_rsi < oversold:
            return 'long'
        # SHORT: Overbought
        elif current_rsi > overbought:
            return 'short'
        else:
            return None
            
    except Exception as e:
        logger.error(f"Error in rsi_signal: {e}")
        return None

def macd_signal(df):
    """
    MACD crossover signal
    Returns: 'long', 'short', or None
    """
    try:
        macd = MACD(df['close'])
        macd_line = macd.macd()
        signal_line = macd.macd_signal()
        
        # Check for crossover
        if macd_line.iloc[-2] < signal_line.iloc[-2] and macd_line.iloc[-1] > signal_line.iloc[-1]:
            return 'long'  # Bullish crossover
        elif macd_line.iloc[-2] > signal_line.iloc[-2] and macd_line.iloc[-1] < signal_line.iloc[-1]:
            return 'short'  # Bearish crossover
        else:
            return None
            
    except Exception as e:
        logger.error(f"Error in macd_signal: {e}")
        return None

def ema_crossover_signal(df, fast=9, slow=21):
    """
    EMA crossover signal
    Returns: 'long', 'short' or None
    """
    try:
        ema_fast = df['close'].ewm(span=fast).mean()
        ema_slow = df['close'].ewm(span=slow).mean()
        
        # Check for crossover
        if ema_fast.iloc[-2] < ema_slow.iloc[-2] and ema_fast.iloc[-1] > ema_slow.iloc[-1]:
            return 'long'  # Golden cross
        elif ema_fast.iloc[-2] > ema_slow.iloc[-2] and ema_fast.iloc[-1] < ema_slow.iloc[-1]:
            return 'short'  # Death cross
        else:
            return None
            
    except Exception as e:
        logger.error(f"Error in ema_crossover_signal: {e}")
        return None

def bollinger_band_signal(df, deviation=2):
    """
    Bollinger Band signal
    Returns: 'long', 'short', or None
    """
    try:
        bb = BollingerBands(df['close'], window=20, window_dev=deviation)
        upper = bb.bollinger_hband()
        lower = bb.bollinger_lband()
        
        current_close = df['close'].iloc[-1]
        
        # LONG: Price touches lower band
        if current_close <= lower.iloc[-1]:
            return 'long'
        # SHORT: Price touches upper band
        elif current_close >= upper.iloc[-1]:
            return 'short'
        else:
            return None
            
    except Exception as e:
        logger.error(f"Error in bollinger_band_signal: {e}")
        return None

def volume_breakout_signal(df):
    """
    Volume breakout signal
    Returns: 'long', 'short', or None
    """
    try:
        volume_sma = df['volume'].rolling(20).mean()
        price_change = df['close'].pct_change()
        
        current_volume = df['volume'].iloc[-1]
        current_volume_sma = volume_sma.iloc[-1]
        current_price_change = price_change.iloc[-1]
        
        # High volume + price up
        if current_volume > current_volume_sma * 1.5 and current_price_change > 0.01:
            return 'long'
        # High volume + price down
        elif current_volume > current_volume_sma * 1.5 and current_price_change < -0.01:
            return 'short'
        else:
            return None
            
    except Exception as e:
        logger.error(f"Error in volume_breakout_signal: {e}")
        return None

# -------------------------------------------------------------------
# Position Sizing (Only calculates, but execution in trade_engine.py)
# -------------------------------------------------------------------
def calculate_position_units(entry_price, equity, risk_per_trade=0.02, atr=None, stop_atr_multiplier: float = 2):
    """
    Calculate position size based on risk
    Returns: units, stop_loss_price, take_profit_price
    """
    try:
        # Calculate stop loss distance
        if atr and atr > 0:
            stop_distance = atr * stop_atr_multiplier
            stop_loss_pct = stop_distance / entry_price
        else:
            stop_loss_pct = 0.02  # Default 2%
        
        # Ensure stop isn't too tight or too wide
        stop_loss_pct = max(stop_loss_pct, 0.005)  # Min 0.5%
        stop_loss_pct = min(stop_loss_pct, 0.05)   # Max 5%
        
        # Calculate stop price
        stop_loss_price = entry_price * (1 - stop_loss_pct)
        
        # Take profit (2:1 reward:risk)
        take_profit_pct = stop_loss_pct * 3 # 3:1 reward:risk
        take_profit_price = entry_price * (1 + take_profit_pct)
        
        # Calculate units based on risk
        risk_amount = equity * risk_per_trade
        risk_per_unit = entry_price * stop_loss_pct
        
        if risk_per_unit <= 0:
            logger.warning(f"⚠️ Risk per unit is zero or negative")
            return 0, None, None
        
        units = risk_amount / risk_per_unit
        
        # Cap at 20% of equity
        max_units = (equity * 0.2) / entry_price
        units = min(units, max_units)
        
        # Minimum trade size ($10)
        min_units = 10 / entry_price
        if units < min_units:
            logger.debug(f"Position too small: {units:.6f}, required: {min_units:.6f}")
            return 0, None, None
        
        logger.info(f"Position calc: Entry=${entry_price:.2f}, SL={stop_loss_pct:.2%}, "
                    f"TP={take_profit_pct:.2%}, Units={units:.6f}, Risk=${risk_amount:.2f}")
        
        return units, stop_loss_price, take_profit_price
        
    except Exception as e:
        logger.error(f"Error in calculate_position_units: {e}")
        return 0, None, None

# -------------------------------------------------------------------
# Main Signal Generator
# -------------------------------------------------------------------
def generate_trade_signal(df, equity, risk_per_trade=0.02, symbol=None, trading_engine=None, regime=None):
    """
    Main function that combines multiple strategies
    Now with regime-based strategy selection
    """
    try:
        if df.empty or len(df) < 50:
            logger.debug("Insufficient data")
            return None
        
        # Calculate ATR for position sizing
        atr_series = calculate_atr(df)
        current_atr = atr_series.iloc[-1] if not atr_series.empty else 0
        
        current_price = df['close'].iloc[-1]
        
        # Parse regime to determine market condition
        regime_type = "unknown"
        if regime:
            if "Range" in regime:
                regime_type = "range"
            elif "Compression" in regime:
                regime_type = "compression"
            elif "Expansion" in regime:
                regime_type = "expansion"
            elif "Breakout" in regime:
                regime_type = "breakout"
            elif "Trend" in regime:
                regime_type = "trend"
        
        logger.debug(f"📊 Market regime for {symbol}: {regime_type}")
        
        # ===== REGIME-BASED STRATEGY SELECTION =====
        signal = None
        signal_type = None
        multiplier = 2.0  # Default ATR multiplier
        
        # RANGING MARKET - Mean reversion works best
        if regime_type in ["range", "compression"]:
            logger.debug(f"📈 Ranging market - favoring mean reversion strategies")
            
            # Try RSI first (best for oversold/overbought)
            signal = rsi_signal(df, oversold=30, overbought=70)
            if signal:
                signal_type = "rsi_range"
                multiplier = 1.5  # Tighter stops in range
                logger.info(f"📊 RSI mean reversion signal in ranging market: {signal}")
            
            # If no RSI signal, try Bollinger Bands
            if not signal:
                signal = bollinger_band_signal(df)
                if signal:
                    signal_type = "bollinger_range"
                    multiplier = 1.5
                    logger.info(f"📊 Bollinger Band signal in ranging market: {signal}")
            
            # If still no signal, try volume breakout (could indicate range breakout)
            if not signal:
                signal = volume_breakout_signal(df)
                if signal:
                    signal_type = "volume_breakout_range"
                    multiplier = 2.0
                    logger.info(f"📊 Volume breakout signal in ranging market: {signal}")
        
        # TRENDING MARKET - Trend following works best
        elif regime_type == "trend":
            logger.debug(f"📈 Trending market - favoring trend following strategies")
            
            # Try EMA crossover first
            signal = ema_crossover_signal(df)
            if signal:
                signal_type = "ema_trend"
                multiplier = 2.5  # Wider stops in trends
                logger.info(f"📊 EMA crossover signal in trending market: {signal}")
            
            # If no EMA signal, try MACD
            if not signal:
                signal = macd_signal(df)
                if signal:
                    signal_type = "macd_trend"
                    multiplier = 2.5
                    logger.info(f"📊 MACD signal in trending market: {signal}")
            
            # If still no signal, try breakout (trend continuation)
            if not signal:
                signal = breakout_signal(df)
                if signal:
                    signal_type = "breakout_trend"
                    multiplier = 2.5
                    logger.info(f"📊 Breakout signal in trending market: {signal}")
        
        # BREAKOUT MARKET - Breakout strategies work best
        elif regime_type == "breakout":
            logger.debug(f"🚀 Breakout market - favoring breakout strategies")
            
            # Try breakout signal first
            signal = breakout_signal(df)
            if signal:
                signal_type = "breakout_breakout"
                multiplier = 3.0  # Wider stops for breakouts
                logger.info(f"📊 Breakout signal in breakout market: {signal}")
            
            # If no breakout, try volume breakout
            if not signal:
                signal = volume_breakout_signal(df)
                if signal:
                    signal_type = "volume_breakout_breakout"
                    multiplier = 2.5
                    logger.info(f"📊 Volume breakout signal in breakout market: {signal}")
            
            # If still no signal, try momentum indicators
            if not signal:
                signal = macd_signal(df)
                if signal:
                    signal_type = "macd_breakout"
                    multiplier = 2.5
                    logger.info(f"📊 MACD signal in breakout market: {signal}")
        
        # EXPANSION (HIGH VOLATILITY) - Be cautious
        elif regime_type == "expansion":
            logger.debug(f"🌪️ High volatility market - being cautious")
            
            # Reduce position size in high volatility
            risk_per_trade = risk_per_trade * 0.5  # Half the normal risk
            
            # Only take very strong signals
            signal = breakout_signal(df, volume_confirmation=True)
            if signal:
                signal_type = "breakout_expansion"
                multiplier = 3.0  # Wider stops for volatility
                logger.info(f"📊 Strong breakout signal in high volatility: {signal}")
            
            if not signal:
                signal = volume_breakout_signal(df)
                if signal and abs(df['close'].pct_change().iloc[-1]) > 0.02:
                    signal_type = "volume_expansion"
                    multiplier = 2.5
                    logger.info(f"📊 Volume breakout in high volatility: {signal}")
        
        # UNKNOWN REGIME - Try all strategies in order
        else:
            logger.debug(f"❓ Unknown regime - trying all strategies")
            
            # Try breakout first (most powerful)
            signal = breakout_signal(df)
            if signal:
                signal_type = "breakout"
                multiplier = 2.5
            else:
                # Then volume breakout
                signal = volume_breakout_signal(df)
                if signal:
                    signal_type = "volume_breakout"
                    multiplier = 2.0
                else:
                    # Then EMA crossover
                    signal = ema_crossover_signal(df)
                    if signal:
                        signal_type = "ema_crossover"
                        multiplier = 2.0
                    else:
                        # Then MACD
                        signal = macd_signal(df)
                        if signal:
                            signal_type = "macd_crossover"
                            multiplier = 2.0
                        else:
                            # Then RSI
                            signal = rsi_signal(df)
                            if signal:
                                signal_type = "rsi"
                                multiplier = 1.5
                            else:
                                # Finally Bollinger
                                signal = bollinger_band_signal(df)
                                if signal:
                                    signal_type = "bollinger"
                                    multiplier = 1.5
        
        # If we have a signal, calculate position size
        if signal:
            units, sl, tp = calculate_position_units(
                current_price, 
                equity, 
                risk_per_trade, 
                current_atr, 
                multiplier
            )
            
            if units > 0:
                # For short signals, cap by available balance
                if signal == 'short' and trading_engine:
                    available = get_available_balance(symbol, trading_engine)
                    if units > available:
                        original_units = units
                        units = available
                        logger.info(f"🔄 Short position capped: {original_units:.6f} -> {units:.6f}")
                        if units <= 0:
                            return None
                
                return {
                    'side': signal,
                    'entry': current_price,
                    'units': units,
                    'stop_loss': sl,
                    'take_profit': tp,
                    'signal_type': signal_type,
                    'regime': regime_type,
                    'confidence': get_confidence_from_regime(regime_type, signal_type)
                }
        
        logger.debug("No signals detected")
        return None
        
    except Exception as e:
        logger.error(f"Error in generate_trade_signal: {e}")
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
    test_timeframes = ["15m", "1h", "4h"]
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
            from data_feed import fetch_ohlcv
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
                ("MACD", lambda: macd_signal(df)),
                ("EMA Crossover", lambda: ema_crossover_signal(df)),
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
        
        if units > 0:
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
        from data_feed import fetch_ohlcv
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
        from data_feed import fetch_ohlcv
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
    from data_feed import fetch_ohlcv
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