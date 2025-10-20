import logging
import time
import sys
import os
from data_feed import fetch_ohlcv
from strategy_tools import check_breakout, calculate_position_size, generate_trade_signal
import pandas as pd
from unittest.mock import Mock, patch, MagicMock, mock_open
import pytest
import json

from trade_engine import execute_limit_order

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scheduler import start_schedulers, weekly_trading_job, intraday_trading_job, data_refresh_job, limit_order_check_job

# Set up logging to see what's happening
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Print to console
        logging.FileHandler("test_scheduler.log")  # Also log to file
    ]
)

def test_scheduler():
    """Test the scheduler functions"""
    
    # Mock bot_data
    bot_data = {
        "run_bot": True,
        "trading_interval": "15m",
        "portfolio": {
            "cash_balance": 1000,
            "holdings": {},
            "positions": {},
            "pending_orders": []
        }
    }
    
    print("🚀 Starting Scheduler Test...")
    print("=" * 50)
    
    # Test 1: Start the scheduler
    print("1. Testing scheduler startup...")
    try:
        start_schedulers(bot_data)
        print("✅ Scheduler started successfully")
    except Exception as e:
        print(f"❌ Scheduler startup failed: {e}")
        return
    
    # Test 2: Run jobs manually
    print("\n2. Testing individual jobs...")
    
    # Test weekly trading job
    try:
        print("   Testing weekly_trading_job...")
        weekly_trading_job(bot_data)
        print("   ✅ Weekly trading job completed")
    except Exception as e:
        print(f"   ❌ Weekly trading job failed: {e}")
    
    # Test intraday trading job
    try:
        print("   Testing intraday_trading_job...")
        intraday_trading_job(bot_data)
        print("   ✅ Intraday trading job completed")
    except Exception as e:
        print(f"   ❌ Intraday trading job failed: {e}")
    
    # Test data refresh job
    try:
        print("   Testing data_refresh_job...")
        data_refresh_job(bot_data)
        print("   ✅ Data refresh job completed")
    except Exception as e:
        print(f"   ❌ Data refresh job failed: {e}")
    
    # Test limit order check job
    try:
        print("   Testing limit_order_check_job...")
        limit_order_check_job(bot_data)
        print("   ✅ Limit order check job completed")
    except Exception as e:
        print(f"   ❌ Limit order check job failed: {e}")
    
    print("\n3. Waiting for scheduled jobs to run (30 seconds)...")
    print("   Press Ctrl+C to stop early")
    
    # Let the scheduler run for a bit
    try:
        for i in range(30):
            time.sleep(1)
            if i % 10 == 0:
                print(f"   ...{i} seconds elapsed")
    except KeyboardInterrupt:
        print("\n   Test interrupted by user")
    
    print("\n✅ Scheduler test completed!")
    print("📋 Check 'test_scheduler.log' for detailed logs")

def test_regime_detection():
    """Test regime detection functionality"""
    from regime_switcher import train_model, predict_regime, model
    from data_feed import fetch_ohlcv
    
    print("🧪 Testing Fixed Regime Detection...")
    
    # Test training
    success = train_model()
    print(f"Training success: {success}")
    print(f"Model is None: {model is None}")
    
    # Test prediction
    if success and model is not None:
        df = fetch_ohlcv("BTC/USDC", "1h")
        result = predict_regime(df)
        print(f"Prediction: {result}")
    else:
        print("❌ Model training failed")

def test_feature_alignment():
    from regime_switcher import train_model, predict_regime, feature_columns_used, model
    from data_feed import fetch_ohlcv
    
    print("🧪 Testing Feature Alignment...")
    
    # Train model
    success = train_model()
    print(f"Training success: {success}")
    print(f"Features used in training: {feature_columns_used}")
    print(f"Model expects features: {model.n_features_in_ if model else 'No model'}")
    
    # Test prediction with multiple coins
    test_coins = ["BTC/USDC", "ETH/USDC", "ADA/USDC"]
    
    for coin in test_coins:
        df = fetch_ohlcv(coin, "1h")
        if not df.empty:
            result = predict_regime(df)
            print(f"{coin}: {result}")

def run_basic_tests():
    """Run basic functionality tests without mocks"""
    from trade_engine import load_portfolio, save_portfolio
    
    print("🧪 Running Basic Trade Engine Tests...")
    print("=" * 50)
    
    # Test 1: Portfolio operations
    try:
        portfolio = load_portfolio()
        print("✅ Portfolio loaded successfully")
        print(f"   Balance: ${portfolio['cash_balance']:.2f}")
        print(f"   Holdings: {len(portfolio.get('holdings', {}))}")
        print(f"   Positions: {len(portfolio.get('positions', {}))}")
    except Exception as e:
        print(f"❌ Portfolio load failed: {e}")
    
    # Test 2: Config loading
    try:
        from trade_engine import load_config
        config = load_config()
        print("✅ Config loaded successfully")
        print(f"   Trading mode: {'LIVE' if config.get('live_trading') else 'PAPER'}")
        print(f"   Coins monitored: {len(config.get('coins', []))}")
    except Exception as e:
        print(f"❌ Config load failed: {e}")
    
    # Test 3: Trading mode detection
    try:
        from trade_engine import get_trading_mode, is_paper_trading
        mode = get_trading_mode()
        is_paper = is_paper_trading()
        print("✅ Trading mode detection working")
        print(f"   Mode: {mode}")
        print(f"   Is paper trading: {is_paper}")
    except Exception as e:
        print(f"❌ Trading mode detection failed: {e}")
    
    print("=" * 50)
    print("✅ Basic tests completed!")

def test_breakout_detection():
    from regime_switcher import train_model, predict_regime
    from data_feed import fetch_ohlcv
    
    print("🧪 Testing Breakout Detection...")
    
    success = train_model()
    if success:
        # Test with volatile coins that might have breakouts
        test_coins = ["BTC/USDC", "ETH/USDC", "ADA/USDC", "SOL/USDC"]
        
        for coin in test_coins:
            df = fetch_ohlcv(coin, "1h")
            if not df.empty:
                result = predict_regime(df)
                print(f"{coin}: {result}")
                
                # Check if we ever get breakout predictions
                if "Breakout" in result:
                    print(f"🎉 BREAKOUT DETECTED for {coin}!")
def debug_strategy():
    """Debug why no trade signals are being generated"""
    print("🔍 Debugging Strategy Tools...")
    print("=" * 50)
    
    # Test with multiple coins and timeframes
    test_pairs = [
        ("BTC/USDC", "15m"),
        ("ETH/USDC", "15m"), 
        ("ADA/USDC", "15m"),
        ("SOL/USDC", "15m")
    ]
    
    for symbol, timeframe in test_pairs:
        print(f"\n📊 Analyzing {symbol} ({timeframe})...")
        
        try:
            # Fetch data
            df = fetch_ohlcv(symbol, timeframe)
            if df.empty:
                print(f"   ❌ No data for {symbol}")
                continue
                
            print(f"   ✅ Data: {len(df)} candles, Latest: ${df.iloc[-1]['close']:.2f}")
            
            # Check breakout conditions
            df_with_signals = check_breakout(df)
            last_row = df_with_signals.iloc[-1]
            
            # Print key indicators
            print(f"   📈 Indicators:")
            print(f"      Close: ${last_row['close']:.2f}")
            print(f"      EMA: ${last_row.get('ema', 0):.2f}")
            print(f"      Donchian High: ${last_row.get('highest_high', 0):.2f}")
            print(f"      Donchian Low: ${last_row.get('lowest_low', 0):.2f}")
            print(f"      ATR: ${last_row.get('atr', 0):.4f}")
            print(f"      Volume: {last_row['volume']:.0f} vs SMA: {last_row.get('volume_sma', 0):.0f}")
            
            # Check conditions
            long_cond = last_row.get('long_condition', False)
            short_cond = last_row.get('short_condition', False)
            
            print(f"   🔍 Conditions:")
            print(f"      Long Breakout: {long_cond}")
            print(f"      Short Breakout: {short_cond}")
            
            if long_cond:
                print("   🟢 LONG signal conditions met!")
                # Check individual long conditions
                close_prev = last_row.get('close_prev', 0)
                highest_high_prev = last_row.get('highest_high_prev', 0)
                close = last_row['close']
                ema = last_row.get('ema', 0)
                volume = last_row['volume']
                volume_sma_prev = last_row.get('volume_sma', 0)
                
                print(f"      Close_prev < Highest_high_prev: {close_prev:.2f} < {highest_high_prev:.2f} = {close_prev < highest_high_prev}")
                print(f"      Close > Highest_high_prev: {close:.2f} > {highest_high_prev:.2f} = {close > highest_high_prev}")
                print(f"      Close > EMA: {close:.2f} > {ema:.2f} = {close > ema}")
                print(f"      Volume > Volume_SMA_prev: {volume:.0f} > {volume_sma_prev:.0f} = {volume > volume_sma_prev}")
                
            if short_cond:
                print("   🔴 SHORT signal conditions met!")
                # Check individual short conditions
                close_prev = last_row.get('close_prev', 0)
                lowest_low_prev = last_row.get('lowest_low_prev', 0)
                close = last_row['close']
                ema = last_row.get('ema', 0)
                volume = last_row['volume']
                volume_sma_prev = last_row.get('volume_sma', 0)
                
                print(f"      Close_prev > Lowest_low_prev: {close_prev:.2f} > {lowest_low_prev:.2f} = {close_prev > lowest_low_prev}")
                print(f"      Close < Lowest_low_prev: {close:.2f} < {lowest_low_prev:.2f} = {close < lowest_low_prev}")
                print(f"      Close < EMA: {close:.2f} < {ema:.2f} = {close < ema}")
                print(f"      Volume > Volume_SMA_prev: {volume:.0f} > {volume_sma_prev:.0f} = {volume > volume_sma_prev}")
            
            # Test signal generation
            if long_cond or short_cond:
                equity = 1000  # Test with $1000
                signal = generate_trade_signal(df, equity, risk_pct=0.01)
                if signal:
                    print(f"   🎯 GENERATED SIGNAL: {signal}")
                else:
                    print("   ❌ Conditions met but no signal generated (position sizing issue?)")
            else:
                print("   ⚪ No breakout conditions met")
                
        except Exception as e:
            print(f"   ❌ Error analyzing {symbol}: {e}")
            import traceback
            traceback.print_exc()

def test_breakout_scenarios():
    """Test specific breakout scenarios"""
    print("\n" + "=" * 50)
    print("🧪 Testing Breakout Scenarios...")
    
    # Create a synthetic breakout scenario
    dates = pd.date_range(start='2024-01-01', periods=50, freq='15T')
    synthetic_data = {
        'open': [100 + i for i in range(50)],
        'high': [102 + i for i in range(50)], 
        'low': [98 + i for i in range(50)],
        'close': [101 + i for i in range(50)],
        'volume': [1000 + i*10 for i in range(50)]
    }
    
    # Create a clear breakout in the last candle
    synthetic_data['high'][-1] = 160  # Break above previous high
    synthetic_data['close'][-1] = 159  # Close near high
    synthetic_data['volume'][-1] = 5000  # High volume
    
    df_synthetic = pd.DataFrame(synthetic_data, index=dates)
    
    print("Testing synthetic breakout data...")
    df_with_signals = check_breakout(df_synthetic)
    last_row = df_with_signals.iloc[-1]
    
    print(f"Long condition: {last_row['long_condition']}")
    print(f"Short condition: {last_row['short_condition']}")
    
    if last_row['long_condition']:
        signal = generate_trade_signal(df_synthetic, 1000, 0.01)
        print(f"Synthetic signal: {signal}")

def debug_regime_model():
    """Debug the regime model to see why confidence is so high"""
    from regime_switcher import model, train_model, predict_regime
    
    print("🐛 Debugging Regime Model...")
    print("=" * 50)
    
    # Check if model exists
    if model is None:
        print("Model is None - training now...")
        train_model()
    
    if model is None:
        print("❌ Model still None after training")
        return
    
    # Test with multiple symbols to see predictions
    test_symbols = ["BTC/USDC", "ETH/USDC", "ADA/USDC"]
    
    for symbol in test_symbols:
        print(f"\nTesting {symbol}:")
        df = fetch_ohlcv(symbol, "1h")
        if not df.empty:
            prediction = predict_regime(df)
            print(f"   Prediction: {prediction}")
            
            # Check feature distribution
            from regime_switcher import add_features
            df_features = add_features(df)
            if not df_features.empty:
                feature_stats = df_features[['rsi', 'macd', 'adx', 'volatility']].describe()
                print(f"   Feature stats - RSI: {df_features['rsi'].iloc[-1]:.1f}, "
                      f"ADX: {df_features['adx'].iloc[-1]:.1f}")
        else:
            print(f"   ❌ No data for {symbol}")

def test_auto_protection():
    print("Testing auto-protection for manual buys...")
    
    # Test 1: Manual buy with auto-protection
    print("\n1. Testing manual buy WITH auto-protection...")
    success, message = execute_limit_order("BTC/USDC", "buy", 0.001, 50000.0, auto_protect=True)
    print(f"Result: {success} - {message}")
    
    # Check portfolio
    from trade_engine import load_portfolio
    portfolio = load_portfolio()
    print(f"Portfolio status:")
    print(f"  Cash: ${portfolio['cash_balance']:.2f}")
    print(f"  Holdings: {portfolio.get('holdings', {})}")
    print(f"  Protected Positions: {len(portfolio.get('positions', {}))}")
    
    for symbol, position in portfolio.get('positions', {}).items():
        print(f"    {symbol}: {position['amount']} units")
        print(f"      Entry: ${position['entry_price']:.2f}")
        print(f"      Stop Loss: ${position['stop_loss']:.2f}")
        print(f"      Take Profit: ${position['take_profit']:.2f}")
    
    # Test 2: Manual buy WITHOUT auto-protection
    print("\n2. Testing manual buy WITHOUT auto-protection...")
    success, message = execute_limit_order("ETH/USDC", "buy", 0.01, 2800.0, auto_protect=True)
    print(f"Result: {success} - {message}")
    
    # Check portfolio again
    portfolio = load_portfolio()
    print(f"Portfolio status:")
    print(f"  Cash: ${portfolio['cash_balance']:.2f}")
    print(f"  Holdings: {portfolio.get('holdings', {})}")
    print(f"  Protected Positions: {len(portfolio.get('positions', {}))}")
    
    print("\n✅ Auto-protection test completed!")

if __name__ == "__main__":
    # test_scheduler()
    # test_regime_detection()
    test_auto_protection()
    # test_feature_alignment()
    # debug_regime_model()
    # debug_strategy()
    # test_breakout_scenarios()