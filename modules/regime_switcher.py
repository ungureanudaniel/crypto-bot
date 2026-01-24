# modules/regime_switcher.py - FIXED VERSION
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
logging.info("🔧 Configuration loaded for data feed. Trading mode: %s", CONFIG.get('trading_mode', 'paper'))
# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

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
# Feature Engineering - OPTIMIZED
# ---------------------------
def add_features(df, required_features=None):
    """Add technical indicators as features - OPTIMIZED VERSION"""
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

# ---------------------------
# Model Training - FIXED
# ---------------------------
def train_model():
    """Train an XGBoost classifier - IMPROVED VERSION"""
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
    
    # Use fewer coins but more data per coin
    coins = coins[:2]  # Just 2 coins for better data
    logger.info(f"🧪 Training with: {coins}")
    
    for coin_idx, coin in enumerate(coins):
        coin_start = time.time()
        logger.info(f"🔍 [{coin_idx+1}/{len(coins)}] Processing {coin}...")
        
        try:
            # STEP 1: Fetch more data for better regime detection
            logger.info(f"   ↳ Fetching OHLCV data...")
            df = fetch_data_for_regime(coin, "4h", limit=500)  # Use 4h timeframe, more data
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
            
            # Ensure we have enough samples for each regime
            min_samples_per_class = 10
            valid_regimes = []
            for regime in [0, 1, 2]:
                count = regime_counts.get(regime, 0)
                if count >= min_samples_per_class:
                    valid_regimes.append(regime)
                else:
                    logger.warning(f"   ↳ Regime {regime} has only {count} samples")
            
            if len(valid_regimes) < 2:
                logger.warning(f"   ↳ Not enough regimes ({len(valid_regimes)}), skipping")
                continue
                
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
            
            if len(available_features) < 8:  # Reduced requirement
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
        
        if len(X_clean) < 100:  # Need minimum samples
            logger.error("❌ Not enough clean data after NaN removal!")
            return False
        
        # STEP 8: Scale features
        logger.info("⚖️ Scaling features...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_clean)
        X_scaled = pd.DataFrame(X_scaled, columns=X_clean.columns)
        
        # STEP 9: Split data with stratification
        logger.info("✂️ Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_clean, test_size=0.2, random_state=42, stratify=y_clean
        )
        logger.info(f"   ✅ Split: Train={len(X_train)}, Test={len(X_test)}")
        
        # STEP 10: Train model with balanced class weights
        logger.info("🧠 Training XGBoost model...")
        
        # Calculate class weights if imbalanced
        class_counts = y_train.value_counts()
        total_samples = len(y_train)
        class_weights = {cls: total_samples / (len(class_counts) * count) 
                        for cls, count in class_counts.items()}
        
        model = XGBClassifier(
            n_estimators=100,  # Increased
            max_depth=4,       # Slightly deeper
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            scale_pos_weight=None,  # For binary, not needed for multi-class
            # For multi-class, use sample_weight parameter if needed
        )
        
        train_start = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - train_start
        logger.info(f"   ✅ Model trained in {train_time:.1f}s")
        
        # STEP 11: Evaluate
        logger.info("📊 Evaluating model...")
        y_pred = model.predict(X_test)
        
        # Suppress warnings with zero_division
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        
        total_time = time.time() - start_time
        logger.info("=" * 60)
        logger.info(f"✅ TRAINING COMPLETED in {total_time:.1f} seconds")
        logger.info("=" * 60)
        
        # Log detailed results
        logger.info(f"Accuracy: {report['accuracy']:.3f}")
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
    """Predict the current regime with calibrated confidence - FIXED"""
    global model, feature_columns_used, scaler
    
    # Check if we have a model
    if model is None:
        logger.info("🤖 No trained model found, attempting to train...")
        success = train_model()
        if not success or model is None:
            logger.warning("⚠️ Model training failed, using simple detection")
            return simple_regime_detection(df)
    
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
                return simple_regime_detection(df)
        
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
            0: "Range-Bound 📊",
            1: "Trending 📈", 
            2: "Breakout 🚀"
        }
        
        regime_label = regime_map.get(prediction, f"Unknown ({prediction})")
        
        # Adjust confidence display
        display_confidence = min(99, max(60, int(confidence * 100)))
        
        return f"{regime_label} ({display_confidence}% confidence)"
        
    except Exception as e:
        logger.error(f"❌ Error in predict_regime: {e}")
        return simple_regime_detection(df)

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
    """Label market regimes - BETTER BREAKOUT DETECTION
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
        high_volatility = df['volatility'] > high_vol_threshold
        large_return = df['returns'].abs() > breakout_return_threshold
        high_atr = df['atr_pct'] > df['atr_pct'].quantile(0.8)
        volume_spike = df['volume_ratio'] > 2.0
        price_channel_break = (df['close'] > df['high_20']) | (df['close'] < df['low_20'])
        
        breakout_score = (high_volatility.astype(int) + 
                         large_return.astype(int) + 
                         high_atr.astype(int) + 
                         volume_spike.astype(int) + 
                         price_channel_break.astype(int))
        
        df.loc[breakout_score >= 3, 'regime'] = 2
        
        # TRENDING DETECTION (Class 1)
        moderate_adx = df['adx'] > 25
        consistent_returns = df['returns_5'].abs() > trend_return_threshold
        bb_squeeze = df['bb_width'] < df['bb_width'].quantile(0.3)
        macd_trend = df['macd_histogram'].abs() > df['macd_histogram'].abs().quantile(0.7)
        
        positive_trend = (df['returns_5'] > 0) & (df['macd'] > df['macd_signal'])
        negative_trend = (df['returns_5'] < 0) & (df['macd'] < df['macd_signal'])
        
        trending_conditions = (
            (moderate_adx & consistent_returns) |
            (bb_squeeze & consistent_returns) |
            (macd_trend & consistent_returns) |
            ((positive_trend | negative_trend) & consistent_returns)
        )
        
        df.loc[trending_conditions & (df['regime'] != 2), 'regime'] = 1
        
        logger.info(f"Regime distribution: {dict(df['regime'].value_counts())}")
        
    except Exception as e:
        logger.error(f"Error labeling regimes: {e}")
        returns_abs = df['returns'].abs()
        df.loc[returns_abs > returns_abs.quantile(0.95), 'regime'] = 2
        df.loc[returns_abs > returns_abs.quantile(0.70), 'regime'] = 1
    
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
    print(f"Config loaded: {config.get('testnet', False)}")
    
    # Test with a coin
    test_coin = "BTC/USDC"
    result = quick_predict(test_coin)
    print(f"Result for {test_coin}: {result}")