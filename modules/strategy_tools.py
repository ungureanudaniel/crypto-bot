from modules.ml_integration import MLStrategy
import pandas as pd
import numpy as np
import logging
from typing import Dict
from modules.sentiment_agent import SentimentAgent
from ta.volatility import BollingerBands
from ta.trend import MACD
from ta.momentum import RSIIndicator

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
    """Get available balance for short positions"""
    if not trading_engine or not trading_engine.binance_client:
        return 0
    
    base_currency = symbol.split('/')[0]
    try:
        account = trading_engine.binance_client.get_account()
        for balance in account['balances']:
            if balance['asset'] == base_currency:
                return float(balance['free'])
    except:
        pass
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

def rsi_signal(df, oversold=30, overbought=70):
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
    Returns: 'long', 'short', or None
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
# Position Sizing (Only calculates - execution in trade_engine)
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
        take_profit_pct = stop_loss_pct * 2
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
def generate_trade_signal(df, equity, risk_per_trade=0.02, symbol=None, available_balance=None):
    """
    Main function that combines multiple strategies
    symbol: the trading pair (e.g., "XLM/USDT")
    available_balance: for shorts, how much of the base currency we own
    """
    try:
        if df.empty or len(df) < 50:
            logger.debug("Insufficient data")
            return None
        
        # Calculate ATR for position sizing
        atr_series = calculate_atr(df)
        current_atr = atr_series.iloc[-1] if not atr_series.empty else 0
        
        current_price = df['close'].iloc[-1]
        
        # Try each strategy in order of preference
        
        # 1. Breakout signal (most powerful)
        signal = breakout_signal(df, lookback=20)
        if signal:
            logger.info(f"📊 Breakout signal: {signal}")
            units, sl, tp = calculate_position_units(
                current_price, equity, risk_per_trade, current_atr, 2.5
            )
            if units > 0:
                # For short signals, cap by available balance
                if signal == 'short' and available_balance is not None:
                    if units > available_balance:
                        original_units = units
                        units = available_balance
                        logger.info(f"🔄 Short position capped: {original_units:.6f} -> {units:.6f} (available {symbol} balance)")
                        
                        # If capping to zero, skip
                        if units <= 0:
                            logger.info(f"⏭️ No {symbol} balance available for short")
                            return None
                
                return {
                    'side': signal,
                    'entry': current_price,
                    'units': units,
                    'stop_loss': sl,
                    'take_profit': tp,
                    'signal_type': 'breakout'
                }
        
        # 2. Volume breakout
        signal = volume_breakout_signal(df)
        if signal:
            logger.info(f"📊 Volume breakout signal: {signal}")
            units, sl, tp = calculate_position_units(
                current_price, equity, risk_per_trade, current_atr, 2.0
            )
            if units > 0:
                if signal == 'short' and available_balance is not None:
                    if units > available_balance:
                        original_units = units
                        units = available_balance
                        logger.info(f"🔄 Short position capped: {original_units:.6f} -> {units:.6f} (available {symbol} balance)")
                        if units <= 0:
                            return None
                
                return {
                    'side': signal,
                    'entry': current_price,
                    'units': units,
                    'stop_loss': sl,
                    'take_profit': tp,
                    'signal_type': 'volume_breakout'
                }
        
        # 3. EMA crossover
        signal = ema_crossover_signal(df)
        if signal:
            logger.info(f"📊 EMA crossover signal: {signal}")
            units, sl, tp = calculate_position_units(
                current_price, equity, risk_per_trade, current_atr, 2.0
            )
            if units > 0:
                if signal == 'short' and available_balance is not None:
                    if units > available_balance:
                        original_units = units
                        units = available_balance
                        logger.info(f"🔄 Short position capped: {original_units:.6f} -> {units:.6f} (available {symbol} balance)")
                        if units <= 0:
                            return None
                
                return {
                    'side': signal,
                    'entry': current_price,
                    'units': units,
                    'stop_loss': sl,
                    'take_profit': tp,
                    'signal_type': 'ema_crossover'
                }
        
        # 4. MACD crossover
        signal = macd_signal(df)
        if signal:
            logger.info(f"📊 MACD crossover signal: {signal}")
            units, sl, tp = calculate_position_units(
                current_price, equity, risk_per_trade, current_atr, 2.0
            )
            if units > 0:
                if signal == 'short' and available_balance is not None:
                    if units > available_balance:
                        original_units = units
                        units = available_balance
                        logger.info(f"🔄 Short position capped: {original_units:.6f} -> {units:.6f} (available {symbol} balance)")
                        if units <= 0:
                            return None
                
                return {
                    'side': signal,
                    'entry': current_price,
                    'units': units,
                    'stop_loss': sl,
                    'take_profit': tp,
                    'signal_type': 'macd_crossover'
                }
        
        # 5. RSI (mean reversion)
        signal = rsi_signal(df)
        if signal:
            logger.info(f"📊 RSI signal: {signal}")
            units, sl, tp = calculate_position_units(
                current_price, equity, risk_per_trade * 0.8, current_atr, 1.5
            )
            if units > 0:
                if signal == 'short' and available_balance is not None:
                    if units > available_balance:
                        original_units = units
                        units = available_balance
                        logger.info(f"🔄 Short position capped: {original_units:.6f} -> {units:.6f} (available {symbol} balance)")
                        if units <= 0:
                            return None
                
                return {
                    'side': signal,
                    'entry': current_price,
                    'units': units,
                    'stop_loss': sl,
                    'take_profit': tp,
                    'signal_type': 'rsi'
                }
        
        # 6. Bollinger Bands (mean reversion)
        signal = bollinger_band_signal(df)
        if signal:
            logger.info(f"📊 Bollinger Band signal: {signal}")
            units, sl, tp = calculate_position_units(
                current_price, equity, risk_per_trade * 0.8, current_atr, 1.5
            )
            if units > 0:
                if signal == 'short' and available_balance is not None:
                    if units > available_balance:
                        original_units = units
                        units = available_balance
                        logger.info(f"🔄 Short position capped: {original_units:.6f} -> {units:.6f} (available {symbol} balance)")
                        if units <= 0:
                            return None
                
                return {
                    'side': signal,
                    'entry': current_price,
                    'units': units,
                    'stop_loss': sl,
                    'take_profit': tp,
                    'signal_type': 'bollinger'
                }
        
        # No signal
        logger.debug("No signals detected")
        return None
        
    except Exception as e:
        logger.error(f"Error in generate_trade_signal: {e}")
        return None
    
if __name__ == "__main__":
    # Test
    from data_feed import fetch_ohlcv
    
    df = fetch_ohlcv("BTC/USDC", interval='1h', limit=200)
    if not df.empty:
        signal = generate_trade_signal(df, equity=10000)
        if signal:
            print(f"✅ Signal: {signal}")
        else:
            print("❌ No signal")