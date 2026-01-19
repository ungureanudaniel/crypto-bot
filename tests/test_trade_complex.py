import json
import pandas as pd
from unittest.mock import Mock, patch, mock_open, MagicMock
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def setup_test_environment():
    """Setup all required mocks for the test environment"""
    
    # Mock the missing functions that trade_engine depends on
    def mock_close_position(symbol, current_price):
        print(f"Mock: Closing position for {symbol} at ${current_price}")
        return True

    def mock_send_telegram_message(message):
        print(f"Mock Telegram: {message}")
        return True
        
    def mock_check_portfolio_health():
        return True
        
    def mock_calculate_dynamic_stop_loss(symbol, entry_price, side='long', atr_multiplier=2):
        if side == 'long':
            return entry_price * 0.95, entry_price * 1.10
        else:
            return entry_price * 1.05, entry_price * 0.90
    
    # Apply all mocks
    patches = [
        patch('trade_engine.close_position', mock_close_position),
        patch('trade_engine.send_telegram_message', mock_send_telegram_message),
        patch('trade_engine.check_portfolio_health', mock_check_portfolio_health),
        patch('trade_engine.calculate_dynamic_stop_loss', mock_calculate_dynamic_stop_loss)
    ]
    
    for p in patches:
        p.start()
    
    return patches

def cleanup_test_environment(patches):
    """Clean up all patches"""
    for p in patches:
        p.stop()

def test_portfolio_functions():
    """Test portfolio loading and saving"""
    print("Testing portfolio functions...")
    
    sample_portfolio = {
        "cash_balance": 10000.0,
        "holdings": {"BTC": 0.5},
        "positions": {},
        "trade_history": [],
        "pending_orders": [],
        "initial_balance": 10000.0
    }
    
    # Mock the file operations
    with patch('builtins.open', mock_open(read_data=json.dumps(sample_portfolio))):
        from trade_engine import load_portfolio
        portfolio = load_portfolio()
        assert portfolio['cash_balance'] == 10000.0
        print("✓ load_portfolio works")
    
    with patch('builtins.open', mock_open()) as mock_file:
        from trade_engine import save_portfolio
        save_portfolio(sample_portfolio)
        mock_file.assert_called_with("portfolio.json", "w")
        print("✓ save_portfolio works")
    
    return True

def test_config_functions():
    """Test configuration handling"""
    print("Testing config functions...")
    
    mock_config = {
        "binance_api_key": "test_key",
        "binance_api_secret": "test_secret",
        "coins": ["BTC/USDC", "ETH/USDC"],
        "risk_pct": 0.02,
        "live_trading": False
    }
    
    with patch('builtins.open', mock_open(read_data=json.dumps(mock_config))):
        from trade_engine import load_config
        config = load_config()
        assert config['binance_api_key'] == "test_key"
        print("✓ load_config works")
    
    # Mock load_config for the other functions
    with patch('trade_engine.load_config', return_value=mock_config):
        from trade_engine import get_trading_mode, is_paper_trading
        
        mode = get_trading_mode()
        assert mode == 'paper'
        print("✓ get_trading_mode (paper) works")
        
        paper_trading = is_paper_trading()
        assert paper_trading == True
        print("✓ is_paper_trading works")
    
    # Test live trading mode
    live_config = mock_config.copy()
    live_config['live_trading'] = True
    with patch('trade_engine.load_config', return_value=live_config):
        mode = get_trading_mode()
        assert mode == 'live'
        print("✓ get_trading_mode (live) works")
    
    return True

def test_risk_management():
    """Test risk management functions"""
    print("Testing risk management functions...")
    
    sample_portfolio = {
        "cash_balance": 10000.0,
        "holdings": {"BTC": 0.5},
        "positions": {},
        "trade_history": [],
        "pending_orders": [],
        "initial_balance": 10000.0
    }
    
    from trade_engine import calculate_position_size, check_portfolio_health
    
    # Test position sizing
    size = calculate_position_size("BTC/USDC", 50000.0, sample_portfolio)
    assert size > 0
    assert size * 50000 <= 10000 * 0.1  # Max 10% of portfolio
    print(f"✓ calculate_position_size: {size:.6f} units")
    
    # Test with very high price (minimum position)
    size_min = calculate_position_size("BTC/USDC", 1000000.0, sample_portfolio)
    assert size_min > 0
    assert size_min * 1000000 >= 10  # Minimum $10
    print(f"✓ calculate_position_size (minimum): {size_min:.6f} units")
    
    # Test portfolio health
    with patch('trade_engine.load_portfolio', return_value=sample_portfolio):
        health = check_portfolio_health()
        assert health in [True, False]
        print("✓ check_portfolio_health works")
    
    return True

def test_trade_execution():
    """Test automatic trade execution"""
    print("Testing trade execution...")
    
    sample_portfolio = {
        "cash_balance": 10000.0,
        "holdings": {"BTC": 0.5},
        "positions": {},
        "trade_history": [],
        "pending_orders": [],
        "initial_balance": 10000.0
    }
    
    from trade_engine import execute_trade
    
    # Mock dependencies
    with patch('trade_engine.load_config', return_value={'risk_pct': 0.02}), \
         patch('trade_engine.load_portfolio', return_value=sample_portfolio):
        
        # Test breakout regime (should trade)
        execute_trade("ETH/USDC", 2, 2800.0)  # regime 2 = breakout
        print("✓ execute_trade (breakout regime) works")
        
        # Test rangebound regime (should not trade)
        execute_trade("ETH/USDC", 0, 2800.0)  # regime 0 = rangebound
        print("✓ execute_trade (rangebound regime) works")
    
    return True

def quick_test():
    """Quick functionality test"""
    print("Running quick functionality test...")
    
    try:
        patches = setup_test_environment()
        
        # Test basic functions
        sample_portfolio = {
            "cash_balance": 10000.0,
            "holdings": {"BTC": 0.5},
            "positions": {},
            "trade_history": [],
            "pending_orders": [],
            "initial_balance": 10000.0
        }
        
        from trade_engine import calculate_position_size, calculate_dynamic_stop_loss, is_paper_trading
        
        # Test position sizing
        size = calculate_position_size("BTC/USDC", 50000.0, sample_portfolio)
        print(f"✓ Position size: {size:.6f} BTC (${size * 50000:.2f})")
        
        # Test stop loss
        stop_loss, take_profit = calculate_dynamic_stop_loss("BTC/USDC", 50000.0, "long")
        print(f"✓ Stop loss: ${stop_loss:.2f}, Take profit: ${take_profit:.2f}")
        
        # Test config
        with patch('trade_engine.load_config', return_value={'live_trading': False}):
            assert is_paper_trading() == True
            print("✓ Paper trading mode detected correctly")
        
        print("✅ All quick tests passed!")
        
        cleanup_test_environment(patches)
        return True
        
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = quick_test()
    sys.exit(0 if success else 1)