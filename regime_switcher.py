import pandas as pd
import numpy as np
import json
import logging
from ta.trend import MACD, ADXIndicator, IchimokuIndicator
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from data_feed import fetch_ohlcv

logging.basicConfig(level=logging.INFO)

model = None

# ---------------------------
# Feature Engineering
# ---------------------------
def add_features(df):
    """
    Add technical indicators as features.
    """
    df = df.copy()
    
    # RSI
    df['rsi'] = RSIIndicator(df['close']).rsi()
    
    # MACD
    macd = MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    
    # ADX
    df['adx'] = ADXIndicator(df['high'], df['low'], df['close']).adx()
    
    # Bollinger Bands width
    bb = BollingerBands(df['close'])
    df['bb_width'] = bb.bollinger_wband()
    
    # Returns
    df['returns'] = df['close'].pct_change()
    
    # Ichimoku Cloud
    ichimoku = IchimokuIndicator(df['high'], df['low'], df['close'])
    df['ichi_a'] = ichimoku.ichimoku_a()
    df['ichi_b'] = ichimoku.ichimoku_b()
    df['ichi_trend'] = 0
    df.loc[df['close'] > df['ichi_a'], 'ichi_trend'] = 1      # bullish
    df.loc[df['close'] < df['ichi_b'], 'ichi_trend'] = -1     # bearish
    
    return df.dropna()


# ---------------------------
# Regime Labeling
# ---------------------------
def label_regime(df):
    """
    Label market regimes:
    0 = Rangebound
    1 = Trending
    2 = Breakout
    """
    df = df.copy()
    df['regime'] = 0  # default = rangebound
    
    # Breakout priority
    df.loc[df['returns'].abs() > 0.05, 'regime'] = 2
    
    # Trending (ADX >= 20 or Ichimoku confirms trend) without overwriting breakouts
    df.loc[((df['adx'] >= 20) | (df['ichi_trend'] != 0)) & (df['regime'] != 2), 'regime'] = 1
    
    return df


# ---------------------------
# Model Training
# ---------------------------
def train_model():
    """
    Train an XGBoost classifier for regime detection.
    """
    global model
    logging.info("Training XGBoost regime classifier...")
    
    with open("config.json") as f:
        config = json.load(f)
    
    all_features, all_labels = [], []
    
    for coin in config['coins']:
        try:
            df = fetch_ohlcv(coin, "1W")
            df = add_features(df)
            df = label_regime(df)
            
            features = df[['rsi', 'macd', 'macd_signal', 'adx', 'bb_width', 'returns', 'ichi_a', 'ichi_b', 'ichi_trend']]
            labels = df['regime']
            
            all_features.append(features)
            all_labels.append(labels)
        except Exception as e:
            logging.error(f"Error processing {coin}: {e}")
    
    X = pd.concat(all_features)
    y = pd.concat(all_labels)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    logging.info("Model training completed.\n" + str(classification_report(y_test, y_pred)))


# ---------------------------
# Regime Prediction
# ---------------------------
def predict_regime(df):
    """
    Predict the current regime for a given OHLCV dataframe.
    df must contain columns: open, high, low, close
    """
    global model
    if model is None:
        raise ValueError("Model not trained. Call train_model() first.")
    
    df = add_features(df)
    features = df[['rsi', 'macd', 'macd_signal', 'adx', 'bb_width', 'returns', 'ichi_a', 'ichi_b', 'ichi_trend']].iloc[-1].values.reshape(1, -1)
    prediction = model.predict(features)[0]
    
    # Optional: return probability for confidence
    prob = model.predict_proba(features).max()
    logging.info(f"Predicted regime: {prediction} (confidence: {prob:.2f})")
    
    return prediction
