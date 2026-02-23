# modules/ml_features.py
import pandas as pd
import numpy as np
import logging
from ta import add_all_ta_features
from ta.volatility import BollingerBands, AverageTrueRange
from ta.trend import MACD, ADXIndicator
from ta.momentum import RSIIndicator

logger = logging.getLogger(__name__)

class FeatureEngine:
    """Engineers advanced features for ML models"""
    
    def __init__(self):
        self.feature_columns = None
        
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive feature set for ML models
        """
        try:
            df = df.copy()
            
            # Basic price features
            df = self._add_price_features(df)
            
            # Technical indicators
            df = self._add_technical_indicators(df)
            
            # Market microstructure
            df = self._add_market_microstructure(df)
            
            # Cross-asset features (if multiple symbols)
            # df = self._add_cross_asset_features(df)
            
            # Drop NaN values
            df = df.dropna()
            
            # Store feature columns
            if self.feature_columns is None:
                self.feature_columns = [col for col in df.columns if col not in ['timestamp', 'close', 'target']]
            
            return df
            
        except Exception as e:
            logger.error(f"Error creating features: {e}")
            return df
    
    def _add_price_features(self, df):
        """Add price-based features"""
        # Returns at different horizons
        for period in [1, 5, 10, 20, 50]:
            df[f'returns_{period}'] = df['close'].pct_change(period)
            df[f'returns_abs_{period}'] = df[f'returns_{period}'].abs()
        
        # Price position relative to highs/lows
        for period in [10, 20, 50]:
            df[f'high_{period}'] = df['high'].rolling(period).max()
            df[f'low_{period}'] = df['low'].rolling(period).min()
            df[f'position_{period}'] = (df['close'] - df[f'low_{period}']) / (df[f'high_{period}'] - df[f'low_{period}'])
        
        # Volatility features
        df['volatility_10'] = df['returns_1'].rolling(10).std()
        df['volatility_20'] = df['returns_1'].rolling(20).std()
        df['volatility_ratio'] = df['volatility_10'] / df['volatility_20']
        
        return df
    
    def _add_technical_indicators(self, df):
        """Add technical analysis indicators"""
        
        # RSI
        rsi = RSIIndicator(df['close'], window=14)
        df['rsi'] = rsi.rsi()
        df['rsi_ma'] = df['rsi'].rolling(5).mean()
        
        # MACD
        macd = MACD(df['close'])
        df['macd'] = macd.macd_diff()
        df['macd_signal'] = macd.macd_signal()
        df['macd_histogram'] = macd.macd_diff()
        
        # Bollinger Bands
        bb = BollingerBands(df['close'], window=20, window_dev=2)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['close']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # ATR for volatility
        atr = AverageTrueRange(df['high'], df['low'], df['close'], window=14)
        df['atr'] = atr.average_true_range()
        df['atr_pct'] = df['atr'] / df['close']
        
        # ADX for trend strength
        adx = ADXIndicator(df['high'], df['low'], df['close'], window=14)
        df['adx'] = adx.adx()
        df['trend_strength'] = df['adx'] / 100
        
        return df
    
    def _add_market_microstructure(self, df):
        """Add order flow and microstructure features"""
        
        # Volume features
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        df['volume_change'] = df['volume'].pct_change()
        
        # Price impact
        df['price_impact'] = df['returns_1'].abs() / (df['volume_ratio'] + 0.001)
        
        # Spread estimation (using high-low as proxy)
        df['spread'] = (df['high'] - df['low']) / df['close']
        df['spread_ma'] = df['spread'].rolling(10).mean()
        
        # Candle patterns
        df['upper_wick'] = (df['high'] - df[['open', 'close']].max(axis=1)) / (df['high'] - df['low'] + 0.001)
        df['lower_wick'] = (df[['open', 'close']].min(axis=1) - df['low']) / (df['high'] - df['low'] + 0.001)
        df['body_size'] = abs(df['close'] - df['open']) / (df['high'] - df['low'] + 0.001)
        
        return df
    
    def create_target(self, df, horizon=5, threshold=0.005):
        """
        Create target variable for supervised learning
        horizon: number of periods ahead to predict
        threshold: minimum change to consider significant
        """
        future_returns = df['close'].shift(-horizon) / df['close'] - 1
        
        # 3-class target: -1 (down), 0 (neutral), 1 (up)
        conditions = [
            future_returns < -threshold,
            future_returns > threshold
        ]
        choices = [0, 2]  # 0=down, 1=neutral, 2=up
        df['target'] = np.select(conditions, choices, default=1)
        
        return df