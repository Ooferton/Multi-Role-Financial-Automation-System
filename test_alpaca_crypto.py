import os
import alpaca_trade_api as tradeapi
from dotenv import load_dotenv

def test_crypto():
    load_dotenv()
    key = os.getenv("APCA_API_KEY_ID")
    secret = os.getenv("APCA_API_SECRET_KEY")
    api = tradeapi.REST(key, secret, "https://paper-api.alpaca.markets", api_version='v2')
    
    symbols = ["BTCUSD", "BTC/USD", "ETHUSD", "ETH/USD"]
    
    for s in symbols:
        print(f"\n--- Testing {s} ---")
        try:
            # Pluralized method
            trades = api.get_latest_crypto_trades([s])
            print(f"SUCCESS (get_latest_crypto_trades): {trades[s].price}")
        except Exception as e:
            print(f"FAILED (get_latest_crypto_trades): {e}")
            
        try:
            snapshot = api.get_crypto_snapshot(s)
            print(f"SUCCESS (get_crypto_snapshot): {snapshot.latest_trade.price}")
        except Exception as e:
            print(f"FAILED (get_crypto_snapshot): {e}")

if __name__ == "__main__":
    test_crypto()
