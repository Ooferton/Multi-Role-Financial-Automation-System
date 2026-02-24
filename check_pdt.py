import os
import alpaca_trade_api as tradeapi
from dotenv import load_dotenv

def check_account():
    load_dotenv()
    api_key = os.getenv("APCA_API_KEY_ID")
    api_secret = os.getenv("APCA_API_SECRET_KEY")
    base_url = "https://paper-api.alpaca.markets"
    
    if not api_key or not api_secret:
        print("Error: Alpaca credentials not found.")
        return

    api = tradeapi.REST(api_key, api_secret, base_url, api_version='v2')
    
    try:
        account = api.get_account()
        print(f"Account Status: {account.status}")
        print(f"Equity: {account.equity}")
        print(f"Pattern Day Trader: {account.pattern_day_trader}")
        print(f"Day Trade Count: {account.day_trade_count}")
        print(f"Day Trading Buying Power: {account.daytrading_buying_power}")
        print(f"Trading Blocked: {account.trading_blocked}")
        print(f"Transfers Blocked: {account.transfers_blocked}")
        print(f"Account Control Blocked: {account.account_blocked}")
    except Exception as e:
        print(f"Error fetching account: {e}")

if __name__ == "__main__":
    check_account()
