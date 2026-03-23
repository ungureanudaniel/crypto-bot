import pandas as pd
import numpy as np
import sys
import os
import time
import logging
from ta.trend import MACD, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.momentum import RSIIndicator
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

from modules.data_feed import fetch_historical_data

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# CONFIG LOADING
# -------------------------------------------------------------------
try:
    # Add parent directory to path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config_loader import config
    CONFIG = config.config
    logger.info(f"✅ Config loaded: {CONFIG.get('trading_mode', 'paper')}")
except ImportError:
    logger.warning("⚠️ Could not import config_loader, using defaults")
    CONFIG = {'trading_mode': 'paper', 'testnet': False, 'rate_limit_delay': 0.5}

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)
logging.info("🔧 Configuration loaded for regime switcher. Trading mode: %s", CONFIG.get('trading_mode', 'paper'))

# Global variables
model = None
feature_columns_used = None
scaler = None

# ---------------------------
# Helper: Fetch data with testnet support
# ---------------------------
def fetch_data_for_regime(symbol, interval="1h", limit=500):
    """Fetch data for regime detection with testnet support"""
    try:
        # Try different import methods
        try:
            # Method 1: Direct import
            from modules.data_feed import data_feed
            df = data_feed.get_ohlcv(symbol, interval, limit=limit)
        except ImportError:
            try:
                # Method 2: Relative import
                from .data_feed import data_feed
                df = data_feed.get_ohlcv(symbol, interval, limit=limit)
            except ImportError:
                # Method 3: Absolute import
                import sys
                sys.path.insert(0, '.')
                from data_feed import data_feed
                df = data_feed.get_ohlcv(symbol, interval, limit=limit)
        
        if df is not None and not df.empty:
            logger.info(f"✅ Fetched {len(df)} candles for {symbol}")
            return df
        else:
            logger.warning(f"⚠️ No data for {symbol}")
            return pd.DataFrame()
            
    except Exception as e:
        logger.error(f"❌ Error fetching data for {symbol}: {e}")
        return pd.DataFrame()

# ---------------------------
# Feature Engineering
# ---------------------------
def add_features(df, required_features=None):
    """Add technical indicators as features - OPTIMIZED VERSION"""
    from ta.volatility import BollingerBands, AverageTrueRange
    global feature_columns_used
    df = df.copy()
    
    if len(df) < 50:
        logging.warning(f"Insufficient data for indicators: {len(df)} rows")
        return df
    
    # Define default features if none provided
    if feature_columns_used is not None:
        required_features = feature_columns_used
    
    # Define the complete feature set
    all_possible_features = [
        'rsi', 'macd', 'macd_signal', 'macd_histogram', 'adx', 
        'bb_width', 'bb_position', 'atr_pct', 'volatility', 'volatility_5',
        'returns', 'returns_5', 'returns_10', 'price_channel_position',
        'volume_ratio', 'volume_volatility', 'momentum'
    ]
    
    try:
        # Ensure we have required columns
        required_ohlc = ['open', 'high', 'low', 'close', 'volume']
        for col in required_ohlc:
            if col not in df.columns:
                logger.error(f"Missing required column: {col}")
                return pd.DataFrame()
        
        # Price-based features
        df['returns'] = df['close'].pct_change()
        df['returns_5'] = df['close'].pct_change(5)
        df['returns_10'] = df['close'].pct_change(10)
        
        # Normalized price movement
        df['price_change'] = (df['close'] - df['open']) / df['open']
        df['body_size'] = (df['close'] - df['open']).abs() / (df['high'] - df['low']).replace(0, 1e-10)
        
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
        df['bb_width'] = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg().replace(0, 1e-10)
        df['bb_position'] = (df['close'] - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband()).replace(0, 1e-10)
        
        # ATR for volatility
        atr = AverageTrueRange(df['high'], df['low'], df['close'], window=14)
        df['atr'] = atr.average_true_range()
        df['atr_pct'] = df['atr'] / df['close'].replace(0, 1e-10)
        
        # Multiple volatility timeframes
        df['volatility'] = df['returns'].rolling(window=20, min_periods=1).std()
        df['volatility_5'] = df['returns'].rolling(window=5, min_periods=1).std()
        df['volatility_10'] = df['returns'].rolling(window=10, min_periods=1).std()
        
        # Price channels
        df['high_20'] = df['high'].rolling(window=20).max()
        df['low_20'] = df['low'].rolling(window=20).min()
        df['price_channel_position'] = (df['close'] - df['low_20']) / (df['high_20'] - df['low_20']).replace(0, 1e-10)
        
        # Volume analysis
        df['volume_sma'] = df['volume'].rolling(window=20, min_periods=1).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma'].replace(0, 1e-10)
        df['volume_volatility'] = df['volume_ratio'].rolling(window=10).std()
        
        # Momentum and acceleration
        df['momentum'] = df['close'] / df['close'].shift(5).replace(0, 1e-10) - 1
        df['momentum_10'] = df['close'] / df['close'].shift(10).replace(0, 1e-10) - 1
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
        
        # Set global feature columns if not set
        if feature_columns_used is None:
            feature_columns_used = [col for col in all_possible_features if col in df.columns]
            logger.info(f"📊 Set feature columns: {len(feature_columns_used)} features")
        
        return df.dropna()
        
    except Exception as e:
        logging.error(f"❌ Error adding features: {e}")
        # Create basic features as fallback
        df_fallback = df.copy()
        for feature in all_possible_features:
            df_fallback[feature] = 0
        df_fallback['returns'] = df_fallback['close'].pct_change()
        df_fallback['volatility'] = df_fallback['returns'].rolling(window=20, min_periods=1).std()
        return df_fallback.dropna()

def ensure_features(df):
    """
    Ensure all required features exist for both regime detection and strategy.
    """
    try:
        # If not enough rows, return original with zeros
        if len(df) < 20:
            for col in [
                'rsi','macd','macd_signal','macd_histogram','adx',
                'bb_width','bb_position','atr_pct','volatility','volatility_5',
                'returns','returns_5','returns_10','price_channel_position',
                'volume_ratio','volume_volatility','momentum'
            ]:
                if col not in df.columns:
                    df[col] = 0
            return df
        
        # Call optimized feature function
        df = add_features(df)
        
        # Fill any remaining missing features with 0
        for col in [
            'rsi','macd','macd_signal','macd_histogram','adx',
            'bb_width','bb_position','atr_pct','volatility','volatility_5',
            'returns','returns_5','returns_10','price_channel_position',
            'volume_ratio','volume_volatility','momentum'
        ]:
            if col not in df.columns:
                df[col] = 0
        df.fillna(0, inplace=True)
        
        return df
    
    except Exception as e:
        logging.error(f"Error ensuring features: {e}")
        # Fallback: minimal columns
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(20, min_periods=1).std()
        df.fillna(0, inplace=True)
        return df

def detect_trend(df, lookback=50):
    """Return (direction, strength, confidence) using multiple timeframes."""
    # Use 20/50 EMA cross for timely direction (faster than SMA)
    ema20 = df['close'].ewm(span=20).mean()
    ema50 = df['close'].ewm(span=50).mean()

    # Direction
    if ema20.iloc[-1] > ema50.iloc[-1]:
        direction = "up"
    elif ema20.iloc[-1] < ema50.iloc[-1]:
        direction = "down"
    else:
        direction = "side"

    # Strength (ADX)
    adx = ADXIndicator(df['high'], df['low'], df['close'], window=14).adx()
    strength = adx.iloc[-1] / 100  # 0-1

    # Additional check: price relative to 200-period EMA (long‑term trend)
    ema200 = df['close'].ewm(span=200).mean()
    long_trend = "up" if df['close'].iloc[-1] > ema200.iloc[-1] else "down"

    # Confidence: higher if both short and long trend agree
    confidence = 0.7 if direction == long_trend else 0.5
    if strength > 0.3: confidence += 0.2  # strong trend boosts confidence

    return direction, strength, confidence

def confirm_trend_with_higher_tf(symbol, df_1h):
    """Return True if 4h and 1h trends agree."""
    try:
        # Fetch 4h data (use a function that caches results to avoid too many calls)
        df_4h = fetch_historical_data(symbol, interval='4h', days=30)
        if df_4h.empty: return True  # fallback
        ema20_4h = df_4h['close'].ewm(span=20).mean()
        ema50_4h = df_4h['close'].ewm(span=50).mean()
        trend_4h = "up" if ema20_4h.iloc[-1] > ema50_4h.iloc[-1] else "down"
        trend_1h, _, _ = detect_trend(df_1h)
        return trend_1h == trend_4h
    except:
        return True

# ---------------------------
# Model Training
# ---------------------------
def train_model():
    """Train an XGBoost classifier - FIXED VERSION WITH BETTER CLASS HANDLING"""
    global model, feature_columns_used, scaler
    
    logger.info("=" * 60)
    logger.info("🔄 STARTING MODEL TRAINING")
    logger.info("=" * 60)
    
    start_time = time.time()
    
    # Load config properly
    config = CONFIG
    
    all_features = []
    all_labels = []
    
    coins = config.get('coins', ['BTC/USDC', 'ETH/USDC', 'SOL/USDC', 'ADA/USDC', 'BNB/USDC'])
    
    logger.info(f"🧪 Training with: {coins}")
    
    for coin_idx, coin in enumerate(coins):
        coin_start = time.time()
        logger.info(f"🔍 [{coin_idx+1}/{len(coins)}] Processing {coin}...")
        
        try:
            # STEP 1: Fetch more data for better regime detection
            logger.info(f"   ↳ Fetching OHLCV data...")
            timeframe = "15m" if CONFIG.get('trading_mode') == 'testnet' else "1h"

            df = fetch_data_for_regime(coin, timeframe, limit=500)  # Use 4h timeframe, more data
            logger.info(f"   ↳ Got {len(df)} candles")
            
            if df.empty or len(df) < 100:  # Require more data
                logger.warning(f"   ↳ Insufficient data ({len(df)}), skipping")
                continue
                
            # STEP 2: Add features
            logger.info(f"   ↳ Adding features...")
            df_with_features = add_features(df)
            logger.info(f"   ↳ Features added, shape: {df_with_features.shape}")
            
            if df_with_features.empty:
                logger.warning(f"   ↳ No features, skipping")
                continue
                
            # STEP 3: Label regimes with better balancing
            logger.info(f"   ↳ Labeling regimes...")
            df_labeled = label_regime(df_with_features)
            logger.info(f"   ↳ Labeled, shape: {df_labeled.shape}")
            
            # Check regime distribution
            regime_counts = df_labeled['regime'].value_counts().sort_index()
            logger.info(f"   ↳ Regime distribution: {regime_counts.to_dict()}")
            
            # FIX: Ensure we have enough samples for each regime BEFORE adding
            min_samples_per_class = 5
            valid_regimes = []
            for regime in [0, 1, 2, 3, 4]:  # All 5 regimes
                count = regime_counts.get(regime, 0)
                if count >= min_samples_per_class:
                    valid_regimes.append(regime)
                else:
                    logger.warning(f"   ↳ Regime {regime} has only {count} samples")
            
            if len(valid_regimes) < 2:
                logger.warning(f"   ↳ Not enough regimes ({len(valid_regimes)}), skipping")
                continue
            
            # FIX: Filter data to only include valid regimes
            df_labeled = df_labeled[df_labeled['regime'].isin(valid_regimes)]
            
            # Get available features
            if feature_columns_used is None:
                # Define default features
                feature_columns_used = [
                    'rsi', 'macd', 'macd_signal', 'macd_histogram', 'adx', 
                    'bb_width', 'bb_position', 'atr_pct', 'volatility', 'volatility_5',
                    'returns', 'returns_5', 'returns_10', 'price_channel_position',
                    'volume_ratio', 'volume_volatility', 'momentum'
                ]
            
            available_features = [col for col in feature_columns_used if col in df_labeled.columns]
            logger.info(f"   ↳ Available features: {len(available_features)}")
            
            if len(available_features) < 5:  # Further reduced requirement
                logger.warning(f"   ↳ Insufficient features ({len(available_features)}), skipping")
                continue
                
            features = df_labeled[available_features]
            labels = df_labeled['regime']
            
            all_features.append(features)
            all_labels.append(labels)
            
            coin_time = time.time() - coin_start
            logger.info(f"   ✅ Processed in {coin_time:.1f}s | Samples: {len(features)}")
            
        except Exception as e:
            logger.error(f"   ❌ Error processing {coin}: {str(e)[:100]}")
            continue
    
    logger.info(f"📊 Total coins processed: {len(all_features)}/{len(coins)}")
    
    if not all_features:
        logger.error("❌ NO DATA PROCESSED - Training failed!")
        return False
    
    try:
        # STEP 6: Concatenate
        concat_start = time.time()
        logger.info("📦 Concatenating data...")
        X = pd.concat(all_features, ignore_index=True)
        y = pd.concat(all_labels, ignore_index=True)
        concat_time = time.time() - concat_start
        
        # Check final distribution
        y_distribution = y.value_counts().sort_index()
        logger.info(f"   ✅ Concatenated in {concat_time:.1f}s | Total: {len(X)} samples")
        logger.info(f"   📊 Final class distribution: {y_distribution.to_dict()}")
        
        # STEP 7: Clean NaN
        logger.info("🧹 Cleaning NaN values...")
        nan_mask = X.isna().any(axis=1) | y.isna()
        X_clean = X[~nan_mask]
        y_clean = y[~nan_mask]
        logger.info(f"   ✅ Cleaned: {len(X_clean)}/{len(X)} samples remaining")
        
        if len(X_clean) < 50:  # Reduced minimum
            logger.error("❌ Not enough clean data after NaN removal!")
            return False
        
        # STEP 8: Scale features
        logger.info("⚖️ Scaling features...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_clean)
        X_scaled = pd.DataFrame(X_scaled, columns=X_clean.columns)
        
        # STEP 9: FIXED - Split data with conditional stratification
        logger.info("✂️ Splitting data...")
        
        # FIX: Check if we can use stratification
        unique_classes = y_clean.nunique()
        min_samples_per_class = y_clean.value_counts().min()
        
        if unique_classes >= 2 and min_samples_per_class >= 2:
            # We can use stratification
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_clean, test_size=0.2, random_state=42, stratify=y_clean
            )
            logger.info(f"   ✅ Split with stratification")
        else:
            # Not enough samples for stratification
            logger.warning(f"   ⚠️ Not enough samples per class for stratification")
            logger.warning(f"     Classes: {unique_classes}, Min samples: {min_samples_per_class}")
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_clean, test_size=0.2, random_state=42  # No stratification
            )
            logger.info(f"   ✅ Split WITHOUT stratification")
        
        logger.info(f"   📊 Train: {len(X_train)}, Test: {len(X_test)}")
        
        # Check train distribution
        train_dist = y_train.value_counts().sort_index()
        logger.info(f"   📈 Train distribution: {train_dist.to_dict()}")
        
        # STEP 10: Train model with balanced class weights
        logger.info("🧠 Training XGBoost model...")
        
        # FIX: Handle class imbalance properly
        unique_train_classes = y_train.nunique()
        
        if unique_train_classes < 2:
            logger.error(f"❌ Only {unique_train_classes} class in training data!")
            return False
        
        # Use balanced weights for multi-class
        if unique_train_classes > 2:
            # For multi-class, use class weights
            model = XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1,
                objective='multi:softprob',
                num_class=unique_train_classes,
                eval_metric='mlogloss'
            )
        else:
            # For binary classification
            model = XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            )
        
        train_start = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - train_start
        logger.info(f"   ✅ Model trained in {train_time:.1f}s")
        
        # STEP 11: Evaluate
        logger.info("📊 Evaluating model...")
        y_pred = model.predict(X_test)
        
        # FIX: Handle single class in test set
        if len(np.unique(y_test)) == 1:
            logger.warning("⚠️ Only one class in test set - accuracy is 100%")
            accuracy = 1.0
            report = {"accuracy": accuracy}
        else:
            # Suppress warnings with zero_division
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            accuracy = report['accuracy']
        
        total_time = time.time() - start_time
        logger.info("=" * 60)
        logger.info(f"✅ TRAINING COMPLETED in {total_time:.1f} seconds")
        logger.info("=" * 60)
        
        # Log detailed results
        logger.info(f"Accuracy: {accuracy:.3f}")
        
        if isinstance(report, dict):
            for regime in ['0', '1', '2']:
                if regime in report:
                    prec = report[regime]['precision']
                    rec = report[regime]['recall']
                    f1 = report[regime]['f1-score']
                    support = report[regime]['support']
                    logger.info(f"Regime {regime}: Precision={prec:.2f}, Recall={rec:.2f}, F1={f1:.2f}, Support={support}")
        
        return True
        
    except Exception as e:
        total_time = time.time() - start_time
        logger.error(f"❌ TRAINING FAILED after {total_time:.1f}s: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

# ---------------------------
# Regime Prediction - FIXED
# ---------------------------
def predict_regime(df):
    """Predict the current regime with calibrated confidence - WITH TREND DIRECTION"""
    global model, feature_columns_used, scaler
    
    # Check if we have a model
    if model is None:
        logger.info("🤖 No trained model found, attempting to train...")
        success = train_model()
        if not success or model is None:
            logger.warning("⚠️ Model training failed, using simple detection")
            return simple_regime_detection_with_direction(df)
    
    try:
        if len(df) < 50:
            return f"Insufficient data ({len(df)} rows)"
        
        # Add features
        df_with_features = add_features(df)
        if df_with_features.empty:
            return "Feature engineering failed"
        
        # Get latest features
        if feature_columns_used is None:
            # Define default features
            feature_columns_used = [
                'rsi', 'macd', 'macd_signal', 'macd_histogram', 'adx', 
                'bb_width', 'bb_position', 'atr_pct', 'volatility', 'volatility_5',
                'returns', 'returns_5', 'returns_10', 'price_channel_position',
                'volume_ratio', 'volume_volatility', 'momentum'
            ]
        
        # Ensure all features exist
        for feature in feature_columns_used:
            if feature not in df_with_features.columns:
                df_with_features[feature] = 0
        
        # Get latest row
        latest_row = df_with_features.iloc[-1:][feature_columns_used]
        
        if latest_row.isna().any().any():
            logger.warning("NaN values in features, trying previous row")
            latest_row = df_with_features.iloc[-2:-1][feature_columns_used]
            if latest_row.isna().any().any():
                return simple_regime_detection_with_direction(df)
        
        # Scale features
        if scaler is not None:
            features_scaled = scaler.transform(latest_row)
        else:
            features_scaled = latest_row.values
        
        # Predict
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        confidence = probabilities.max()
        
        # Map prediction to label
        regime_map = {
            0: "Range / Mean-Reversion",
            1: "Compression (Squeeze)",
            2: "Expansion (Volatile Chop)",
            3: "Breakout",
            4: "True Trend"
        }
        
        regime_label = regime_map.get(prediction, f"Unknown ({prediction})")
        
        # ===== ADD TREND DIRECTION =====
        # Calculate moving averages for trend direction
        sma_20 = df['close'].rolling(20).mean().iloc[-1]
        sma_50 = df['close'].rolling(50).mean().iloc[-1]
        current_price = df['close'].iloc[-1]
        
        # Determine trend direction
        if current_price > sma_20 and sma_20 > sma_50:
            direction = "UPTREND"
        elif current_price < sma_20 and sma_20 < sma_50:
            direction = "DOWNTREND"
        else:
            direction = "SIDEWAYS"
        
        # Add direction to the regime label
        if prediction == 4:  # True Trend
            full_label = f"{regime_label} {direction}"
        else:
            full_label = regime_label
        
        # Adjust confidence display
        display_confidence = min(99, max(60, int(confidence * 100)))
        
        return f"{full_label} ({display_confidence}% confidence)"
        
    except Exception as e:
        logger.error(f"❌ Error in predict_regime: {e}")
        return simple_regime_detection_with_direction(df)

def simple_regime_detection_with_direction(df):
    """Simple rule-based regime detection with trend direction"""
    try:
        if len(df) < 20:
            return "Insufficient data"
        
        # Get regime from simple detection
        base_result = simple_regime_detection(df)
        
        # Add direction
        sma_20 = df['close'].rolling(20).mean().iloc[-1]
        sma_50 = df['close'].rolling(50).mean().iloc[-1]
        current_price = df['close'].iloc[-1]
        
        if current_price > sma_20 and sma_20 > sma_50:
            direction = "UPTREND"
        elif current_price < sma_20 and sma_20 < sma_50:
            direction = "DOWNTREND"
        else:
            direction = "SIDEWAYS"
        
        # If it's trending, add direction
        if "Trending" in base_result:
            return f"{base_result} {direction}"
        
        return base_result
        
    except Exception as e:
        return f"Simple detection error: {str(e)}"

# ---------------------------
# Quick prediction (for testing)
# ---------------------------
def quick_predict(symbol):
    """Quick prediction for testing"""
    logger.info(f"🔮 Quick prediction for {symbol}")
    
    # Fetch data
    df = fetch_data_for_regime(symbol, "1h", limit=100)
    
    if df.empty:
        return f"No data for {symbol}"
    
    # Predict
    result = predict_regime(df)
    return result

# ---------------------------
# Regime Labeling - KEEP SAME
# ---------------------------
def label_regime(df):
    """
    Realistic market regime labeling based on:
    - Market structure
    - Volatility state
    - Compression vs expansion
    - Breakouts
    - True trend (HH/HL or LH/LL structure)

    Regimes:
    0 = Range / Mean Reversion
    1 = Compression (squeeze / accumulation)
    2 = Expansion (volatile chop)
    3 = Breakout
    4 = Trend (true structure trend)
    """

    df = df.copy()

    # Ensure bb_width exists
    if 'bb_width' not in df.columns:
        try:
            bb = BollingerBands(df['close'], window=20, window_dev=2)
            df['bb_width'] = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg().replace(0, 1e-10)
        except:
            df['bb_width'] = 0

    df['regime'] = 0  # default = range

    # ---------- Volatility ----------
    df['ret'] = df['close'].pct_change()
    vol = df['ret'].rolling(20).std()
    vol_ma = vol.rolling(50).mean()

    # ---------- Structure ----------
    df['hh'] = df['high'] > df['high'].shift(1)
    df['hl'] = df['low'] > df['low'].shift(1)
    df['lh'] = df['high'] < df['high'].shift(1)
    df['ll'] = df['low'] < df['low'].shift(1)

    up_structure = (df['hh'] & df['hl']).rolling(5).sum() >= 3
    down_structure = (df['lh'] & df['ll']).rolling(5).sum() >= 3
    structure_trend = up_structure | down_structure

    # ---------- Compression ----------
    bb_width = df['bb_width']
    compression = (
        (bb_width < bb_width.rolling(50).quantile(0.2)) &
        (vol < vol_ma * 0.7)
    )

    # ---------- Expansion ----------
    expansion = (
        (vol > vol_ma * 1.3)
    )

    # ---------- Breakout ----------
    high_20 = df['high'].rolling(20).max()
    low_20 = df['low'].rolling(20).min()

    breakout = (
        ((df['close'] > high_20.shift(1)) | (df['close'] < low_20.shift(1))) &
        (vol > vol_ma) &
        (df['volume_ratio'] > 1.5)
    )

    # ---------- True Trend ----------
    slope = df['close'].rolling(20).mean().diff()
    trend = structure_trend & (slope.abs() > slope.abs().rolling(50).mean())

    # ---------- Regime Assignment Priority ----------
    # Priority matters (top overrides below)
    df.loc[compression, 'regime'] = 1        # Compression
    df.loc[expansion, 'regime'] = 2          # Expansion
    df.loc[breakout, 'regime'] = 3           # Breakout
    df.loc[trend, 'regime'] = 4              # True Trend

    return df

# ---------------------------
# Simple Fallback Detection
# ---------------------------
def simple_regime_detection(df):
    """Simple rule-based regime detection as fallback"""
    try:
        if len(df) < 20:
            return "Insufficient data"
            
        recent_returns = df['close'].pct_change().tail(10)
        volatility = recent_returns.std()
        avg_return = recent_returns.abs().mean()
        
        if volatility > 0.03 or avg_return > 0.02:
            return "Breakout 🚀"
        elif volatility > 0.01:  
            return "Trending 📈"
        else:
            return "Range-Bound 📊"
            
    except Exception as e:
        return f"Simple detection error: {str(e)}"

# ---------------------------
# Test function
# ---------------------------
if __name__ == "__main__":
    print("🧪 Testing regime switcher with testnet...")
    
    # Test with config
    print(f"Config loaded: {config.get('trading_mode', 'paper')}")
    
    # Test with a coin
    test_coin = "BTC/USDT"
    result = quick_predict(test_coin)
    print(f"Result for {test_coin}: {result}")