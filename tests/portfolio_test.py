# Complete diagnostic test
sudo docker exec binance_trader_bot python3 << 'EOF'
import sys
import os
import json

print("=" * 60)
print("PORTFOLIO DIAGNOSTIC TEST")
print("=" * 60)

# 1. Check file path
from modules.portfolio import PORTFOLIO_FILE
print(f"\n1. Portfolio file path: {PORTFOLIO_FILE}")

# 2. Check if directory exists and is writable
dir_path = os.path.dirname(PORTFOLIO_FILE)
print(f"   Directory exists: {os.path.exists(dir_path)}")
print(f"   Directory writable: {os.access(dir_path, os.W_OK)}")

# 3. Test direct file write
test_file = PORTFOLIO_FILE + ".test"
try:
    with open(test_file, 'w') as f:
        f.write('test')
    os.remove(test_file)
    print(f"   Direct file write: ✅ SUCCESS")
except Exception as e:
    print(f"   Direct file write: ❌ FAILED - {e}")

# 4. Test save_positions function
from modules.portfolio import save_positions, get_positions

print(f"\n2. Testing save_positions...")
test_positions = {'TEST/USDT': {'side': 'long', 'amount': 0.001, 'entry_price': 100}}
try:
    save_positions(test_positions)
    print(f"   save_positions() called - no exception")
except Exception as e:
    print(f"   save_positions() FAILED: {e}")

# 5. Verify it saved
loaded = get_positions()
print(f"\n3. After save, positions in file: {list(loaded.keys())}")

# 6. Check if there are any errors in the logs
print(f"\n4. Check if there were any error messages above")
print("   (Look for any red ERROR messages in the docker logs)")

# 7. Check file content directly
if os.path.exists(PORTFOLIO_FILE):
    with open(PORTFOLIO_FILE, 'r') as f:
        content = json.load(f)
        print(f"\n5. Current file content:")
        print(f"   Positions: {list(content.get('positions', {}).keys())}")
        print(f"   Cash: ${content.get('cash', {}).get('USDT', 0)}")
        print(f"   Last updated: {content.get('last_updated', 'unknown')}")
EOF