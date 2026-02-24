import os
import alpaca_trade_api as tradeapi
from dotenv import load_dotenv

def test_keys():
    load_dotenv()
    key = os.getenv("APCA_API_KEY_ID")
    secret = os.getenv("APCA_API_SECRET_KEY")
    
    print(f"Testing Key ID: {key[:5]}...")
    
    # Test Paper
    print("\n--- Testing Paper ---")
    paper_api = tradeapi.REST(key, secret, "https://paper-api.alpaca.markets", api_version='v2')
    try:
        acc = paper_api.get_account()
        print(f"SUCCESS: Paper account found. Status: {acc.status}")
    except Exception as e:
        print(f"FAILED: Paper check failed: {e}")

    # Test Live
    print("\n--- Testing Live ---")
    live_api = tradeapi.REST(key, secret, "https://api.alpaca.markets", api_version='v2')
    try:
        acc = live_api.get_account()
        print(f"SUCCESS: Live account found. Status: {acc.status}")
    except Exception as e:
        print(f"FAILED: Live check failed: {e}")

if __name__ == "__main__":
    test_keys()
