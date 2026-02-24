
import yfinance as yf
import sys

symbol = "SPY"
try:
    ticker = yf.Ticker(symbol)
    news = ticker.news
    print(f"News type: {type(news)}")
    if news:
        print(f"First news item keys: {news[0].keys()}")
        if 'content' in news[0]:
            print(f"Content keys: {news[0]['content'].keys()}")
except Exception as e:
    print(f"Error: {e}")
