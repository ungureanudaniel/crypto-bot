# ================================
# Multi-Coin Strategy Dashboard
# RSI + MACD + ADX + BB + Cycle
# ================================
# Requirements: pip install streamlit ccxt ta pandas numpy matplotlib scipy

import streamlit as st
import pandas as pd
import numpy as np
import ccxt
import ta
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.fftpack import fft

# ==== CONFIGURATION ====
coins = ['JUP/USDT', 'ATOM/USDT', 'FLR/USDT', 'ICP/USDT', 'ADA/USDT', 
         'PYTH/USDT', 'ARB/USDT', 'TON/USDT', 'INJ/USDT', 'TIA/USDT',
         'DYDX/USDT', 'OP/USDT', 'OCEAN/USDT', 'DOT/USDT', 'GRT/USDT', 
         'DYM/USDT', 'STRAX/USDT', 'IMX/USDT', 'EWT/USDT', 'ICX/USDT']
timeframe = '1h'  # or '1d' for daily
lookback_candles = 500

# ==== INIT EXCHANGE ====
exchange = ccxt.binance()

# ==== FUNCTIONS ====
def fetch_ohlcv(symbol):
    try:
        bars = exchange.fetch_ohlcv(symbol, timeframe, limit=lookback_candles)
        df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {e}")
        return None

def add_indicators(df):
    df['RSI'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    macd = ta.trend.MACD(df['close'])
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    df['ADX'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close']).adx()
    bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
    df['BB_upper'] = bb.bollinger_hband()
    df['BB_lower'] = bb.bollinger_lband()
    df['Cycle'] = fft_cycle(df['close'])
    return df

def fft_cycle(series):
    fft_vals = fft((series - np.mean(series)).to_numpy())
    freqs = np.fft.fftfreq(len(fft_vals))
    peaks, _ = find_peaks(np.abs(fft_vals))
    if peaks.size > 0:
        dominant_freq = freqs[peaks[0]]
        cycle_length = 1 / abs(dominant_freq) if dominant_freq != 0 else np.nan
    else:
        cycle_length = np.nan
    return cycle_length

def generate_signal(row):
    if (
        row['RSI'] < 30 and
        row['MACD'] > row['MACD_signal'] and
        row['ADX'] > 25 and
        row['close'] < row['BB_lower']
    ):
        return 'Strong Buy'
    elif (
        row['RSI'] > 70 and
        row['MACD'] < row['MACD_signal'] and
        row['ADX'] > 25 and
        row['close'] > row['BB_upper']
    ):
        return 'Strong Sell'
    else:
        return 'Neutral'

# ==== STREAMLIT DASHBOARD ====
st.title("📈 Multi-Coin Trading Dashboard")
st.write(f"Tracking: {', '.join([c.split('/')[0] for c in coins])}")

for coin in coins:
    st.subheader(f"🪙 {coin}")
    df = fetch_ohlcv(coin)
    if df is not None:
        df = add_indicators(df)
        df['Signal'] = df.apply(generate_signal, axis=1)

        st.write(df.tail(1)[['close', 'RSI', 'MACD', 'MACD_signal', 'ADX', 'BB_upper', 'BB_lower', 'Cycle', 'Signal']])

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df['timestamp'], df['close'], label='Price')
        ax.plot(df['timestamp'], df['BB_upper'], linestyle='--', color='g', label='BB Upper')
        ax.plot(df['timestamp'], df['BB_lower'], linestyle='--', color='r', label='BB Lower')
        ax.set_title(f"{coin} Price & Bollinger Bands")
        ax.legend()
        st.pyplot(fig)

        st.info(f"📣 **Latest Signal for {coin}: {df['Signal'].iloc[-1]}**")
    st.markdown("---")
