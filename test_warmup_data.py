import os
from datetime import datetime, timedelta
from agents.alpaca_broker import AlpacaBroker
from dotenv import load_dotenv

load_dotenv()

broker = AlpacaBroker(paper=True)
symbols = ["IBIT", "BITO", "FBTC", "ARKB", "BTC/USD"]

start = datetime.now() - timedelta(days=3)
end = datetime.now()

print(f"Warmup Test: Fetching 100 bars for {symbols}")

for s in symbols:
    try:
        bars = broker.get_historical_data(s, start=start, end=end, timeframe='1Min', limit=100)
        print(f"{s}: Got {len(bars)} bars")
    except Exception as e:
        print(f"{s}: Error -> {e}")
