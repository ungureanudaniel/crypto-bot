import pandas as pd
import numpy as np
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from modules.ml_features import FeatureEngine
from modules.ml_model import PricePredictor
from modules.data_feed import data_feed

logger = logging.getLogger(__name__)

class MLStrategy:
    """Integrates ML predictions into trading strategy"""
    
    def __init__(self, symbol: str, retrain_interval_hours: int = 24):
        self.symbol = symbol
        self.retrain_interval = retrain_interval_hours * 3600  # Convert to seconds
        self.last_train_time = None
        
        self.feature_engine = FeatureEngine()
        self.predictor = PricePredictor(model_type='xgboost')
        
        self.is_trained = False
        self.prediction_confidence = 0
        
    async def ensure_trained(self, force_retrain: bool = False):
        """Ensure model is trained (retrain if needed)"""
        now = datetime.now().timestamp()
        
        needs_retrain = (
            force_retrain or
            self.last_train_time is None or
            (now - self.last_train_time) > self.retrain_interval
        )
        
        if needs_retrain:
            logger.info(f"🔄 Training ML model for {self.symbol}")
            success = await self.train_model()
            if success:
                self.last_train_time = now
                self.is_trained = True
            return success
        
        return self.is_trained
    
    async def train_model(self, days: int = 60):
        """Train model on historical data"""
        try:
            # Fetch historical data
            df = await self._fetch_historical_data(days)
            
            if df.empty or len(df) < 500:
                logger.warning(f"⚠️ Insufficient data for training: {len(df)} rows")
                return False
            
            # Create features
            df_features = self.feature_engine.create_features(df)
            
            # Create target (predict 5 periods ahead)
            df_features = self.feature_engine.create_target(df_features, horizon=5)
            
            # Prepare data
            X, y = self.predictor.prepare_data(
                df_features, 
                self.feature_engine.feature_columns
            )
            
            # Train model
            accuracy = self.predictor.train(X, y)
            
            logger.info(f"✅ ML model trained for {self.symbol} with accuracy {accuracy:.3f}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Training failed for {self.symbol}: {e}")
            return False
    
    async def get_prediction(self, current_df: pd.DataFrame) -> Dict:
        """Get ML prediction for current market state"""
        
        if not self.is_trained:
            await self.ensure_trained()
        
        if not self.is_trained:
            return {'direction': 1, 'confidence': 0, 'signal': 'neutral'}
        
        try:
            # Create features for current data
            df_features = self.feature_engine.create_features(current_df)
            
            if df_features.empty:
                return {'direction': 1, 'confidence': 0, 'signal': 'neutral'}
            
            # Get latest features
            latest_features = df_features[self.feature_engine.feature_columns].iloc[-1:]
            
            # Predict
            pred, probs = self.predictor.predict_single(latest_features)
            
            if pred is None:
                return {'direction': 1, 'confidence': 0, 'signal': 'neutral'}
            
            # Map prediction to signal
            # 0=down, 1=neutral, 2=up
            signal_map = {0: 'short', 1: 'neutral', 2: 'long'}
            confidence_map = {0: probs[0], 1: probs[1], 2: probs[2]}
            
            self.prediction_confidence = confidence_map[pred]
            
            result = {
                'direction': pred,
                'signal': signal_map[pred],
                'confidence': confidence_map[pred],
                'prob_down': probs[0],
                'prob_neutral': probs[1],
                'prob_up': probs[2]
            }
            
            logger.debug(f"🤖 ML Prediction for {self.symbol}: {result['signal']} "
                        f"({result['confidence']:.1%} confidence)")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Prediction failed for {self.symbol}: {e}")
            return {'direction': 1, 'confidence': 0, 'signal': 'neutral'}
    
    async def _fetch_historical_data(self, days: int) -> pd.DataFrame:
        """Fetch historical data for training"""
        try:
            df = data_feed.get_historical_data(
                symbol=self.symbol,
                interval='1h',  # Use 1h for training
                days=days
            )
            return df
        except Exception as e:
            logger.error(f"❌ Failed to fetch historical data: {e}")
            return pd.DataFrame()
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from trained model"""
        if self.predictor.feature_importance is not None:
            return self.predictor.feature_importance
        return pd.DataFrame()