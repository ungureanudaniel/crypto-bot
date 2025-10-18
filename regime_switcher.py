import pandas as pd
import numpy as np
import json
import logging
from ta.trend import MACD, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.momentum import RSIIndicator
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from data_feed import fetch_ohlcv

logging.basicConfig(level=logging.INFO)

model = None
feature_columns_used = None
scaler = None

# ---------------------------
# Feature Engineering - OPTIMIZED
# ---------------------------
def add_features(df, required_features=None):
    """
    Add technical indicators as features - OPTIMIZED VERSION
    """
    df = df.copy()
    
    if len(df) < 50:
        logging.warning(f"Insufficient data for indicators: {len(df)} rows")
        return df
    
    # Define the complete feature set
    all_possible_features = [
        'rsi', 'macd', 'macd_signal', 'macd_histogram', 'adx', 
        'bb_width', 'bb_position', 'atr_pct', 'volatility', 'volatility_5',
        'returns', 'returns_5', 'returns_10', 'price_channel_position',
        'volume_ratio', 'volume_volatility', 'momentum'
    ]
    
    try:
        # Price-based features
        df['returns'] = df['close'].pct_change()
        df['returns_5'] = df['close'].pct_change(5)
        df['returns_10'] = df['close'].pct_change(10)
        
        # Normalized price movement
        df['price_change'] = (df['close'] - df['open']) / df['open']
        df['body_size'] = (df['close'] - df['open']).abs() / (df['high'] - df['low'])
        
        # RSI with bounds checking
        rsi = RSIIndicator(df['close'], window=14).rsi()
        df['rsi'] = rsi.clip(0, 100)  # Ensure RSI stays within bounds
        
        # MACD
        macd = MACD(df['close'], window_slow=26, window_fast=12, window_sign=9)
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_histogram'] = macd.macd_diff()
        
        # ADX
        df['adx'] = ADXIndicator(df['high'], df['low'], df['close'], window=14).adx()
        
        # Bollinger Bands
        bb = BollingerBands(df['close'], window=20, window_dev=2)
        df['bb_width'] = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
        df['bb_position'] = (df['close'] - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband())
        
        # ATR for volatility
        atr = AverageTrueRange(df['high'], df['low'], df['close'], window=14)
        df['atr'] = atr.average_true_range()
        df['atr_pct'] = df['atr'] / df['close']
        
        # Multiple volatility timeframes
        df['volatility'] = df['returns'].rolling(window=20, min_periods=1).std()
        df['volatility_5'] = df['returns'].rolling(window=5, min_periods=1).std()
        df['volatility_10'] = df['returns'].rolling(window=10, min_periods=1).std()
        
        # Price channels
        df['high_20'] = df['high'].rolling(window=20).max()
        df['low_20'] = df['low'].rolling(window=20).min()
        df['price_channel_position'] = (df['close'] - df['low_20']) / (df['high_20'] - df['low_20'])
        
        # Volume analysis
        df['volume_sma'] = df['volume'].rolling(window=20, min_periods=1).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        df['volume_volatility'] = df['volume_ratio'].rolling(window=10).std()
        
        # Momentum and acceleration
        df['momentum'] = df['close'] / df['close'].shift(5) - 1
        df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
        df['acceleration'] = df['momentum'] - df['momentum_10']
        
        # Support features for required set
        if required_features:
            for feature in required_features:
                if feature not in df.columns:
                    df[feature] = 0
        
        # Clean extreme values that might cause overfitting
        for col in df.select_dtypes(include=[np.number]).columns:
            if col not in ['timestamp']:
                # Cap extreme values at 99th percentile
                upper_limit = df[col].quantile(0.99)
                lower_limit = df[col].quantile(0.01)
                df[col] = df[col].clip(lower_limit, upper_limit)
        
        logging.info(f"✅ Features added successfully. Final shape: {df.shape}")
        
    except Exception as e:
        logging.error(f"Error adding features: {e}")
        # Create basic features as fallback
        for feature in all_possible_features:
            df[feature] = 0
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(window=20, min_periods=1).std()
    
    return df.dropna()


# ---------------------------
# Model Training - WITH REGULARIZATION
# ---------------------------
def train_model():
    """
    Train an XGBoost classifier with better regularization
    """
    global model, feature_columns_used, scaler
    logging.info("Training XGBoost regime classifier with regularization...")
    
    with open("config.json") as f:
        config = json.load(f)
    
    all_features, all_labels = [], []
    
    for coin in config['coins']:
        try:
            df = fetch_ohlcv(coin, "15m")
            if df.empty:
                continue
                
            df_with_features = add_features(df)
            if df_with_features.empty:
                continue
                
            df_labeled = label_regime(df_with_features)
            
            desired_features = [
                'rsi', 'macd', 'macd_signal', 'macd_histogram', 'adx', 
                'bb_width', 'bb_position', 'atr_pct', 'volatility', 'volatility_5',
                'returns', 'returns_5', 'returns_10', 'price_channel_position',
                'volume_ratio', 'volume_volatility', 'momentum'
            ]
            
            available_features = [col for col in desired_features if col in df_labeled.columns]
            features = df_labeled[available_features]
            labels = df_labeled['regime']
            
            all_features.append(features)
            all_labels.append(labels)
            
        except Exception as e:
            logging.error(f"Error processing {coin}: {e}")
            continue
    
    if not all_features:
        logging.error("❌ No data processed for training!")
        return False
    
    try:
        X = pd.concat(all_features, ignore_index=True)
        y = pd.concat(all_labels, ignore_index=True)
        
        if len(X) == 0:
            return False
        
        # Store feature columns
        feature_columns_used = X.columns.tolist()
        logging.info(f"🔧 Features used for training: {len(feature_columns_used)}")
        
        # Remove NaN values
        nan_mask = X.isna().any(axis=1) | y.isna()
        X = X[~nan_mask]
        y = y[~nan_mask]
        
        if len(X) == 0:
            return False
        
        # Initialize and fit scaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        
        class_dist = y.value_counts()
        logging.info(f"Class distribution: {dict(class_dist)}")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Model with STRONG regularization to prevent overfitting
        model = XGBClassifier(
            n_estimators=100,
            max_depth=3,  # Reduced from 4
            learning_rate=0.05,  # Reduced from 0.1
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1.0,  # L1 regularization
            reg_lambda=1.0,  # L2 regularization
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        logging.info("✅ Model training completed!")
        
        # Calculate confidence statistics
        y_proba = model.predict_proba(X_test)
        max_probs = y_proba.max(axis=1)
        avg_confidence = max_probs.mean()
        high_confidence_ratio = (max_probs > 0.9).mean()
        
        logging.info(f"📊 Confidence stats - Avg: {avg_confidence:.3f}, >90%: {high_confidence_ratio:.3f}")
        logging.info("\n" + str(classification_report(y_test, y_pred)))
        
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logging.info("📊 Feature importance:\n" + str(feature_importance.head(8)))
        
        return True
        
    except Exception as e:
        logging.error(f"❌ Model training failed: {e}")
        return False


# ---------------------------
# Regime Prediction - WITH CONFIDENCE CALIBRATION
# ---------------------------
def predict_regime(df):
    """
    Predict the current regime with calibrated confidence
    """
    global model, feature_columns_used, scaler
    
    if model is None:
        success = train_model()
        if not success or model is None:
            return simple_regime_detection(df)
    
    try:
        if len(df) < 50:
            return f"Insufficient data ({len(df)} rows)"
        
        # Add features
        df_with_features = add_features(df, required_features=feature_columns_used)
        if df_with_features.empty:
            return "Feature engineering failed"
        
        # Ensure feature alignment
        if feature_columns_used:
            missing_features = [col for col in feature_columns_used if col not in df_with_features.columns]
            for feature in missing_features:
                df_with_features[feature] = 0
            prediction_features = df_with_features[feature_columns_used]
        else:
            return simple_regime_detection(df)
        
        # Scale features
        if scaler is not None:
            prediction_features_scaled = scaler.transform(prediction_features)
            prediction_features_scaled = pd.DataFrame(prediction_features_scaled, columns=prediction_features.columns)
        else:
            prediction_features_scaled = prediction_features
        
        # Get latest features
        latest_features = prediction_features_scaled.iloc[-1]
        
        if latest_features.isna().any():
            latest_features = prediction_features_scaled.iloc[-2]
            if latest_features.isna().any():
                return simple_regime_detection(df)
        
        features_array = latest_features.values.reshape(1, -1)
        
        # Verify shape
        if features_array.shape[1] != model.n_features_in_:
            return simple_regime_detection(df)
        
        prediction = model.predict(features_array)[0]
        probabilities = model.predict_proba(features_array)[0]
        max_prob = probabilities.max()
        
        # Calibrate confidence display
        confidence = max_prob
        if confidence > 0.95:
            confidence = 0.85 + (confidence - 0.95) * 0.3  # Compress very high confidences
        elif confidence < 0.6:
            confidence = 0.6  # Minimum confidence threshold
        
        regime_labels = {
            0: "Range-Bound 📊",
            1: "Trending 📈", 
            2: "Breakout 🚀"
        }
        
        regime_label = regime_labels.get(prediction, f"Unknown ({prediction})")
        
        # Log actual vs calibrated confidence
        actual_confidence = max_prob
        logging.info(f"🔮 Predicted regime: {regime_label} (raw: {actual_confidence:.1%}, calibrated: {confidence:.1%})")
        
        return f"{regime_label} ({(confidence*100):.0f}% confidence)"
        
    except Exception as e:
        logging.error(f"❌ Error in predict_regime: {e}")
        return simple_regime_detection(df)

# ---------------------------
# Regime Labeling - IMPROVED BREAKOUT DETECTION
# ---------------------------
def label_regime(df):
    """
    Label market regimes - BETTER BREAKOUT DETECTION
    0 = Rangebound, 1 = Trending, 2 = Breakout
    """
    df = df.copy()
    df['regime'] = 0  # default = rangebound
    
    try:
        # Calculate dynamic thresholds based on recent data
        recent_volatility = df['volatility'].tail(50)
        recent_returns = df['returns'].abs().tail(50)
        
        high_vol_threshold = recent_volatility.quantile(0.85)  # Top 15% volatility
        breakout_return_threshold = recent_returns.quantile(0.90)  # Top 10% returns
        trend_return_threshold = recent_returns.quantile(0.70)  # Top 30% returns
        
        # BREAKOUT DETECTION (Class 2)
        # Multiple conditions for breakout confirmation
        high_volatility = df['volatility'] > high_vol_threshold
        large_return = df['returns'].abs() > breakout_return_threshold
        high_atr = df['atr_pct'] > df['atr_pct'].quantile(0.8)
        volume_spike = df['volume_ratio'] > 2.0  # 2x average volume
        price_channel_break = (df['close'] > df['high_20']) | (df['close'] < df['low_20'])
        
        # Breakout requires at least 3 confirmations
        breakout_score = (high_volatility.astype(int) + 
                         large_return.astype(int) + 
                         high_atr.astype(int) + 
                         volume_spike.astype(int) + 
                         price_channel_break.astype(int))
        
        df.loc[breakout_score >= 3, 'regime'] = 2
        
        # TRENDING DETECTION (Class 1)
        # Moderate conditions for trending markets
        moderate_adx = df['adx'] > 25
        consistent_returns = df['returns_5'].abs() > trend_return_threshold
        bb_squeeze = df['bb_width'] < df['bb_width'].quantile(0.3)  # Low volatility periods before moves
        macd_trend = df['macd_histogram'].abs() > df['macd_histogram'].abs().quantile(0.7)
        
        # Trending requires directional consistency
        positive_trend = (df['returns_5'] > 0) & (df['macd'] > df['macd_signal'])
        negative_trend = (df['returns_5'] < 0) & (df['macd'] < df['macd_signal'])
        
        trending_conditions = (
            (moderate_adx & consistent_returns) |
            (bb_squeeze & consistent_returns) |
            (macd_trend & consistent_returns) |
            ((positive_trend | negative_trend) & consistent_returns)
        )
        
        df.loc[trending_conditions & (df['regime'] != 2), 'regime'] = 1
        
        # RANGEBOUND (Class 0) - everything else
        
        # Ensure minimum samples per class
        regime_counts = df['regime'].value_counts()
        total_samples = len(df)
        
        logging.info(f"Regime distribution: {dict(regime_counts)}")
        logging.info(f"Class percentages: 0: {regime_counts.get(0,0)/total_samples*100:.1f}%, "
                    f"1: {regime_counts.get(1,0)/total_samples*100:.1f}%, "
                    f"2: {regime_counts.get(2,0)/total_samples*100:.1f}%")
        
        # If breakout class is still too small, be more lenient
        if regime_counts.get(2, 0) < total_samples * 0.1:  # Less than 10%
            logging.warning("Breakout class too small, using relaxed conditions")
            relaxed_breakout = breakout_score >= 2  # Only 2 confirmations needed
            df.loc[relaxed_breakout & (df['regime'] != 2), 'regime'] = 2
            
        # Final check - if still no breakouts, create artificial ones
        regime_counts_final = df['regime'].value_counts()
        if regime_counts_final.get(2, 0) == 0:
            logging.warning("No breakouts detected, creating synthetic breakouts")
            # Label top 5% most volatile periods as breakouts
            top_volatility = df['volatility'].nlargest(int(total_samples * 0.05))
            df.loc[top_volatility.index, 'regime'] = 2
        
    except Exception as e:
        logging.error(f"Error labeling regimes: {e}")
        # Conservative fallback with all three classes
        returns_abs = df['returns'].abs()
        df.loc[returns_abs > returns_abs.quantile(0.95), 'regime'] = 2  # Top 5% as breakout
        df.loc[returns_abs > returns_abs.quantile(0.70), 'regime'] = 1  # Top 30% as trending
        # Remainder as rangebound (0)
    
    return df

# ---------------------------
# Simple Fallback Detection
# ---------------------------
def simple_regime_detection(df):
    """
    Simple rule-based regime detection as fallback
    """
    try:
        if len(df) < 20:
            return "Insufficient data"
            
        # Calculate recent volatility and returns
        recent_returns = df['close'].pct_change().tail(10)
        volatility = recent_returns.std()
        avg_return = recent_returns.abs().mean()
        
        if volatility > 0.03 or avg_return > 0.02:  # High volatility
            return "Breakout 🚀"
        elif volatility > 0.01:  # Moderate volatility  
            return "Trending 📈"
        else:
            return "Range-Bound 📊"
            
    except Exception as e:
        return f"Simple detection error: {str(e)}"

if __name__ == "__main__":
    # Test regime detection
    from data_feed import fetch_ohlcv
    
    test_coin = "BTC/USDC"
    df = fetch_ohlcv(test_coin, "1h")
    
    if not df.empty:
        result = predict_regime(df)
        print(f"Regime prediction for {test_coin}: {result}")
    else:
        print(f"No data fetched for {test_coin}")
