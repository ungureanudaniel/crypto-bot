import pandas as pd
import numpy as np

# -------------------------------------------------------------------
# Helper functions for breakout strategy
# -------------------------------------------------------------------

def calculate_donchian(df, length=20):
    """
    Returns Donchian channel: high, low, mid
    """
    high = df['high'].rolling(length).max()
    low = df['low'].rolling(length).min()
    mid = (high + low) / 2
    return high, low, mid

def calculate_atr(df, length=20):
    """
    Returns ATR series
    """
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift(1))
    low_close = np.abs(df['low'] - df['close'].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(length).mean()
    return atr

def check_breakout(df, ema_length=200, volume_length=20):
    """
    Returns booleans for long and short breakout conditions
    """
    df['ema'] = df['close'].ewm(span=ema_length).mean()
    df['atr'] = calculate_atr(df)
    df['highest_high'], df['lowest_low'], df['dc_mid'] = calculate_donchian(df)

    # Previous bar references
    df['highest_high_prev'] = df['highest_high'].shift(1)
    df['lowest_low_prev'] = df['lowest_low'].shift(1)
    df['close_prev'] = df['close'].shift(1)
    df['volume_prev'] = df['volume'].shift(1)

    # Breakout conditions
    df['long_condition'] = (
        (df['close_prev'] < df['highest_high_prev']) &
        (df['close'] > df['highest_high_prev']) &
        (df['close'] > df['ema']) &
        (df['volume'] > df['volume'].rolling(volume_length).max().shift(1)) &
        ((df['high'] - df['close']) < (df['close'] - df['highest_high_prev']))
    )

    df['short_condition'] = (
        (df['close_prev'] > df['lowest_low_prev']) &
        (df['close'] < df['lowest_low_prev']) &
        (df['close'] < df['ema']) &
        (df['volume'] > df['volume'].rolling(volume_length).max().shift(1)) &
        ((df['close'] - df['low']) < (df['lowest_low_prev'] - df['close']))
    )

    return df

def calculate_position_size(entry_price, equity, risk_pct=0.01, atr=None, atr_multiplier=2, reward_ratio=2, stop_strategy="ATR", dc_mid=None):
    """
    Returns: take_profit, stop_loss, units
    """
    max_loss_amount = equity * risk_pct

    if stop_strategy == "ATR" and atr is not None:
        stop_loss_pct = (atr * atr_multiplier) / entry_price
    elif stop_strategy == "Donchian" and dc_mid is not None:
        stop_loss_pct = abs(entry_price - dc_mid) / entry_price
    else:
        stop_loss_pct = 0.02  # fallback 2%

    take_profit_pct = reward_ratio * stop_loss_pct
    stop_loss_price = entry_price * (1 - stop_loss_pct)
    take_profit_price = entry_price * (1 + take_profit_pct)
    units = max_loss_amount / (stop_loss_pct * entry_price)

    return take_profit_price, stop_loss_price, units

def dominance_confirmation(dominance_open, dominance_close, dominance_ema):
    """
    Returns True if trend aligns with breakout
    """
    if dominance_open > dominance_ema and dominance_close > dominance_ema:
        return "above"
    elif dominance_open < dominance_ema and dominance_close < dominance_ema:
        return "below"
    else:
        return "none"

# -------------------------------------------------------------------
# Example integration
# -------------------------------------------------------------------
def generate_trade_signal(df, equity, risk_pct=0.01):
    df = check_breakout(df)
    last_row = df.iloc[-1]

    # Example dominance check (can fetch from API)
    dominance_open, dominance_close, dominance_ema = 0.5, 0.52, 0.51
    dominance = dominance_confirmation(dominance_open, dominance_close, dominance_ema)

    signal = None
    if last_row['long_condition'] and dominance in ["above", "none"]:
        tp, sl, units = calculate_position_size(last_row['close'], equity, risk_pct, last_row['atr'], 2, 2, "ATR", last_row['dc_mid'])
        signal = {"side": "long", "entry": last_row['close'], "take_profit": tp, "stop_loss": sl, "units": units}

    elif last_row['short_condition'] and dominance in ["below", "none"]:
        tp, sl, units = calculate_position_size(last_row['close'], equity, risk_pct, last_row['atr'], 2, 2, "ATR", last_row['dc_mid'])
        signal = {"side": "short", "entry": last_row['close'], "take_profit": tp, "stop_loss": sl, "units": units}

    return signal
