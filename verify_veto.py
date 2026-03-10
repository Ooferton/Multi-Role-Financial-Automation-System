import logging
import os
from core.risk_manager import RiskManager

def test_veto():
    logging.basicConfig(level=logging.INFO)
    rm = RiskManager({})
    
    lock_path = "data/system_veto.lock"
    
    print("\n--- VETO VERIFICATION TEST ---")
    
    # 1. Test without veto
    if os.path.exists(lock_path):
        os.remove(lock_path)
    
    is_safe = rm.check_trade_risk({"action":"BUY", "symbol":"AAPL", "price":150, "quantity":1}, {"equity":100000, "market_value":0})
    print(f"Normal Mode: Trade Safe? {is_safe}")
    
    # 2. Test with veto
    with open(lock_path, 'w') as f:
        f.write("Veto Test")
    
    is_safe = rm.check_trade_risk({"action":"BUY", "symbol":"AAPL", "price":150, "quantity":1}, {"equity":100000, "market_value":0})
    print(f"Veto Mode:   Trade Safe? {is_safe}")
    
    # Cleanup
    if os.path.exists(lock_path):
        os.remove(lock_path)
    
    if not is_safe:
        print("RESULT: SUCCESS - Veto correctly blocked the trade.")
    else:
        print("RESULT: FAILED - Veto did not block the trade.")

if __name__ == "__main__":
    test_veto()
