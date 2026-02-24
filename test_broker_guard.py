import os
import yaml
import logging
from agents.alpaca_broker import AlpacaBroker
from core.broker_interface import TradeOrder
from dotenv import load_dotenv

load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)

def test_guard():
    print("--- DIAGNOSTIC TEST: BITCOIN/ETF GUARD ---")
    
    # 1. Force config to True
    config_path = "config/config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    original_mode = config.get("system", {}).get("crypto_etf_only", False)
    config["system"]["crypto_etf_only"] = True
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f)
    
    print("Mode set to: BITCOIN/ETF ONLY")
    
    broker = AlpacaBroker(paper=True)
    
    # 2. Try to trade something restricted (AAPL)
    order = TradeOrder(symbol="AAPL", qty=1, side="BUY", price=150.0, order_type="MARKET")
    print("Attempting to trade AAPL (RESTRICTED)...")
    result = broker.submit_order(order, reasoning="Test Guard")
    print(f"Result: {result}")
    
    # 3. Try to trade something allowed (BTC/USD)
    order_ok = TradeOrder(symbol="BTC/USD", qty=0.0001, side="BUY", price=50000.0, order_type="MARKET")
    print("Attempting to trade BTC/USD (ALLOWED)...")
    result_ok = broker.submit_order(order_ok, reasoning="Test Guard")
    print(f"Result: {result_ok}")

    # Restore original mode
    # config["system"]["crypto_etf_only"] = original_mode
    # with open(config_path, "w") as f:
    #     yaml.safe_dump(config, f)

if __name__ == "__main__":
    test_guard()
