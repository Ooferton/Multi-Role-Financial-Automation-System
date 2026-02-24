import os
import alpaca_trade_api as tradeapi
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("APCA_API_KEY_ID")
api_secret = os.getenv("APCA_API_SECRET_KEY")
base_url = "https://paper-api.alpaca.markets"

api = tradeapi.REST(api_key, api_secret, base_url, api_version='v2')

symbols = ["IBIT", "BITO", "FBTC", "ARKB", "BTC/USD"]
print(f"Fetching snapshots for: {symbols}")

try:
    # Stocks
    stocks = [s for s in symbols if "/" not in s]
    snapshots = api.get_snapshots(stocks)
    for s in stocks:
        if s in snapshots:
            snap = snapshots[s]
            price = snap.latest_trade.price if snap.latest_trade else "NO TRADE DATA"
            print(f"{s}: {price}")
        else:
            print(f"{s}: NOT IN SNAPSHOTS")

    # Crypto
    crypto = [s for s in symbols if "/" in s]
    for s in crypto:
        try:
            trade = api.get_latest_crypto_trade(s, exchange='CBSE')
            print(f"{s}: {trade.price}")
        except Exception as e:
            print(f"{s}: Error -> {e}")

except Exception as e:
    print(f"Global Error: {e}")
