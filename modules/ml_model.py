# modules/ml_model.py
import numpy as np
import pandas as pd
import logging
import joblib
import os
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb

logger = logging.getLogger(__name__)

class PricePredictor:
    """Machine Learning model for price direction prediction"""
    
    def __init__(self, model_type='xgboost', model_path='models/'):
        self.model_type = model_type
        self.model_path = model_path
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importance = None
        
        # Create model directory if it doesn't exist
        os.makedirs(model_path, exist_ok=True)
        
    def build_model(self):
        """Build the ML model"""
        if self.model_type == 'xgboost':
            self.model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        else:
            self.model = GradientBoostingClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                random_state=42
            )
        
        return self.model
    
    def prepare_data(self, df, feature_columns, target_column='target'):
        """Prepare features and target for training"""
        
        # Separate features and target
        X = df[feature_columns].copy()
        y = df[target_column].copy()
        
        # Handle any remaining NaN values
        X = X.fillna(method='ffill').fillna(method='bfill')
        
        return X, y
    
    def train(self, X, y, test_size=0.2):
        """Train the model"""
        try:
            # Split data chronologically (important for time series)
            split_idx = int(len(X) * (1 - test_size))
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Build and train model
            if self.model is None:
                self.build_model()
            
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred = self.model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            logger.info(f"✅ Model trained with accuracy: {accuracy:.3f}")
            
            # Feature importance
            if hasattr(self.model, 'feature_importances_'):
                self.feature_importance = pd.DataFrame({
                    'feature': X.columns,
                    'importance': self.model.feature_importances_
                }).sort_values('importance', ascending=False)
                logger.info(f"📊 Top features: {self.feature_importance.head(5)}")
            
            # Save model
            self.save_model()
            
            return accuracy
            
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            return 0
    
    def predict(self, X):
        """Predict price direction"""
        if self.model is None:
            logger.warning("⚠️ No trained model available")
            return None
        
        try:
            X_scaled = self.scaler.transform(X)
            predictions = self.model.predict(X_scaled)
            probabilities = self.model.predict_proba(X_scaled)
            
            return predictions, probabilities
            
        except Exception as e:
            logger.error(f"❌ Prediction failed: {e}")
            return None, None
    
    def predict_single(self, features):
        """Predict for a single sample"""
        if isinstance(features, dict):
            features = pd.DataFrame([features])
        
        pred, probs = self.predict(features)
        if pred is not None:
            return pred[0], probs[0]
        return None, None
    
    def save_model(self):
        """Save model to disk"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_file = f"{self.model_path}/{self.model_type}_{timestamp}.joblib"
            scaler_file = f"{self.model_path}/scaler_{timestamp}.joblib"
            
            joblib.dump(self.model, model_file)
            joblib.dump(self.scaler, scaler_file)
            
            logger.info(f"💾 Model saved to {model_file}")
            return model_file
            
        except Exception as e:
            logger.error(f"❌ Failed to save model: {e}")
            return None
    
    def load_model(self, model_file, scaler_file):
        """Load model from disk"""
        try:
            self.model = joblib.load(model_file)
            self.scaler = joblib.load(scaler_file)
            logger.info(f"📂 Model loaded from {model_file}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            return False