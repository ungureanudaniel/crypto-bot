import pandas as pd
import numpy as np
import logging
from regime_switcher import ensure_features, label_regime
from ta.volatility import BollingerBands, AverageTrueRange

# -------------------------------------------------------------------
# Enhanced Helper functions for breakout strategy
# -------------------------------------------------------------------

def calculate_donchian(df, length=20):
    """
    Returns Donchian channel: high, low, mid
    """
    high = df['high'].rolling(length, min_periods=1).max()
    low = df['low'].rolling(length, min_periods=1).min()
    mid = (high + low) / 2
    return high, low, mid

def calculate_atr(df, length=14):
    """
    Returns ATR series with proper handling
    """
    try:
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift(1))
        low_close = np.abs(df['low'] - df['close'].shift(1))
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(length, min_periods=1).mean()
        return atr
    except Exception as e:
        logging.error(f"Error calculating ATR: {e}")
        return pd.Series([0] * len(df), index=df.index)

def check_breakout(df, ema_length=200, volume_length=20, donchian_length=20):
    """
    Returns DataFrame with breakout conditions WITHOUT modifying original
    """
    try:
        df = df.copy()  # Work on copy to avoid side effects
        
        # Technical indicators
        df['ema'] = df['close'].ewm(span=ema_length, min_periods=1).mean()
        df['atr'] = calculate_atr(df)
        df['highest_high'], df['lowest_low'], df['dc_mid'] = calculate_donchian(df, donchian_length)
        
        # Volume indicator
        df['volume_sma'] = df['volume'].rolling(volume_length, min_periods=1).mean()
        
        # Previous bar references
        df['highest_high_prev'] = df['highest_high'].shift(1)
        df['lowest_low_prev'] = df['lowest_low'].shift(1)
        df['close_prev'] = df['close'].shift(1)
        
        # Breakout conditions with improved logic
        df['long_condition'] = (
            (df['close_prev'] < df['highest_high_prev']) &
            (df['close'] > df['highest_high_prev']) &
            (df['close'] > df['ema']) &
            (df['volume'] > df['volume_sma'].shift(1)) &  # Volume above average
            ((df['high'] - df['close']) < (df['close'] - df['highest_high_prev']))  # Favor bullish close
        )
        
        df['short_condition'] = (
            (df['close_prev'] > df['lowest_low_prev']) &
            (df['close'] < df['lowest_low_prev']) &
            (df['close'] < df['ema']) &
            (df['volume'] > df['volume_sma'].shift(1)) &  # Volume above average
            ((df['close'] - df['low']) < (df['lowest_low_prev'] - df['close']))  # Favor bearish close
        )
        
        return df
        
    except Exception as e:
        logging.error(f"Error in check_breakout: {e}")
        # Return original df with default conditions
        df['long_condition'] = False
        df['short_condition'] = False
        return df

def calculate_position_size(entry_price, equity, risk_pct=0.01, atr=None, atr_multiplier=2, reward_ratio=2, stop_strategy="ATR", dc_mid=None):
    """
    Enhanced position sizing with safety checks
    """
    try:
        max_loss_amount = equity * risk_pct
        
        # Calculate stop loss percentage
        if stop_strategy == "ATR" and atr is not None and atr > 0:
            stop_loss_pct = (atr * atr_multiplier) / entry_price
        elif stop_strategy == "Donchian" and dc_mid is not None:
            stop_loss_pct = abs(entry_price - dc_mid) / entry_price
        else:
            stop_loss_pct = 0.02  # fallback 2%
        
        # Safety bounds for stop loss
        stop_loss_pct = max(stop_loss_pct, 0.005)   # Minimum 0.5% stop
        stop_loss_pct = min(stop_loss_pct, 0.10)    # Maximum 10% stop
        
        take_profit_pct = reward_ratio * stop_loss_pct
        stop_loss_price = entry_price * (1 - stop_loss_pct)
        take_profit_price = entry_price * (1 + take_profit_pct)
        
        # Calculate units safely
        risk_per_unit = entry_price * stop_loss_pct
        if risk_per_unit > 0:
            units = max_loss_amount / risk_per_unit
        else:
            units = 0
        
        # Position size limits
        max_position_value = equity * 0.1  # Max 10% of equity per trade
        max_units_by_value = max_position_value / entry_price
        units = min(units, max_units_by_value)
        
        # Minimum trade size (avoid dust)
        min_units = 10 / entry_price  # $10 minimum
        if units < min_units:
            logging.warning(f"Position too small: {units:.6f}, required: {min_units:.6f}")
            return None, None, 0
        
        logging.info(f"Position calc: Entry=${entry_price:.2f}, SL={stop_loss_pct:.2%}, "
                    f"TP={take_profit_pct:.2%}, Units={units:.6f}")
        
        return take_profit_price, stop_loss_price, units
        
    except Exception as e:
        logging.error(f"Error in calculate_position_size: {e}")
        return None, None, 0

def dominance_confirmation(dominance_open, dominance_close, dominance_ema):
    """
    BTC dominance confirmation for trend alignment
    """
    try:
        if dominance_open > dominance_ema and dominance_close > dominance_ema:
            return "above"
        elif dominance_open < dominance_ema and dominance_close < dominance_ema:
            return "below"
        else:
            return "neutral"
    except:
        return "neutral"

# -------------------------------------------------------------------
# Enhanced Signal Generation
# -------------------------------------------------------------------
def generate_trade_signal(df, equity, risk_pct=0.01):
    try:
        if df.empty or len(df) < 20:
            logging.warning("Insufficient data for signal generation")
            return None

        # --- Step 1: Add all features required by strategy ---
        df = ensure_features(df)  # ensures ATR, bb_width, returns, volatility, volume_ratio, etc.

        # --- Step 2: Prepare breakout-specific features ---
        df['volume_sma'] = df['volume'].rolling(20, min_periods=1).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma'].replace(0, 1e-10)
        df['highest_high'] = df['high'].rolling(20, min_periods=1).max()
        df['lowest_low'] = df['low'].rolling(20, min_periods=1).min()
        df['dc_mid'] = (df['highest_high'] + df['lowest_low']) / 2
        df['atr'] = AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()

        # --- Step 3: Check breakout conditions ---
        df_with_signals = check_breakout(df)
        last_row = df_with_signals.iloc[-1]

        # --- Step 4: Generate trade signal ---
        signal = None
        if last_row['long_condition']:
            tp, sl, units = calculate_position_size(
                last_row['close'], equity, risk_pct,
                last_row['atr'], 2, 2, "ATR", last_row['dc_mid']
            )
            if units > 0:
                signal = {
                    "side": "long",
                    "entry": last_row['close'],
                    "take_profit": tp,
                    "stop_loss": sl,
                    "units": units,
                    "signal_type": "breakout_long"
                }

        elif last_row['short_condition']:
            tp, sl, units = calculate_position_size(
                last_row['close'], equity, risk_pct,
                last_row['atr'], 2, 2, "ATR", last_row['dc_mid']
            )
            if units > 0:
                signal = {
                    "side": "short",
                    "entry": last_row['close'],
                    "take_profit": tp,
                    "stop_loss": sl,
                    "units": units,
                    "signal_type": "breakout_short"
                }

        return signal

    except Exception as e:
        logging.error(f"Error in generate_trade_signal: {e}")
        return None

if __name__ == "__main__":
    # Example usage
    import pandas as pd
    from data_feed import fetch_ohlcv
    
    symbol = "BTC/USDC"
    df = fetch_ohlcv(symbol, interval='15m')
    
    # Assume equity of $10,000 for position sizing
    equity = 10000
    
    signal = generate_trade_signal(df, equity)
    if signal:
        print(f"Trade Signal: {signal}")
    else:
        print("No trade signal generated.")