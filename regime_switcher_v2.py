import pandas as pd
import numpy as np
import json
import logging
from ta.trend import MACD, ADXIndicator
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from data_feed import fetch_ohlcv

model = None

def add_features(df):
    df['rsi'] = RSIIndicator(df['close']).rsi()
    macd = MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['adx'] = ADXIndicator(df['high'], df['low'], df['close']).adx()
    bb = BollingerBands(df['close'])
    df['bb_width'] = bb.bollinger_wband()
    df['returns'] = df['close'].pct_change()
    return df.dropna()

def label_regime(df):
    conditions = [
        (df['adx'] < 20) & (df['bb_width'] < 5),
        (df['adx'] >= 20),
        (df['returns'].abs() > 0.05)
    ]
    choices = [0, 1, 2]  # 0=Rangebound, 1=Trending, 2=Breakout
    df['regime'] = np.select(conditions, choices, default=0)
    return df

def train_model():
    global model
    logging.info("Training XGBoost model...")
    with open("config.json") as f:
        config = json.load(f)
    all_features = []
    all_labels = []
    for coin in config['coins']:
        df = fetch_ohlcv(coin, "1W")
        df = add_features(df)
        df = label_regime(df)
        features = df[['rsi', 'macd', 'macd_signal', 'adx', 'bb_width', 'returns']]
        labels = df['regime']
        all_features.append(features)
        all_labels.append(labels)
    X = pd.concat(all_features)
    y = pd.concat(all_labels)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    logging.info("Model training completed.\n" + classification_report(y_test, y_pred))

def predict_regime(df):
    global model
    features = df[['rsi', 'macd', 'macd_signal', 'adx', 'bb_width', 'returns']].values.reshape(1, -1)
    return model.predict(features)[0]
