import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional
from data.feature_store import MarketTick, FeatureStore

class YahooDataConnector:
    """
    Connects to Yahoo Finance to fetch market data.
    Useful for historical backfilling and delayed live data.
    """
    def __init__(self):
        pass

    def fetch_historical_ticks(self, symbol: str, start_date: datetime, end_date: datetime) -> List[MarketTick]:
        """
        Fetches 1-minute resolution data (max 7 days history for 1m on Yahoo).
        Converts OHLCV to pseudo-ticks (using Close price) for the system.
        """
        # yfinance expects string dates YYYY-MM-DD
        # Limit to last 5 days if resolution is 1m
        
        print(f"Fetching data for {symbol}...")
        ticker = yf.Ticker(symbol)
        
        # dynamic interval based on range
        interval = "1m"
        duration = end_date - start_date
        if duration.days <= 7:
            interval = "1m"
        elif duration.days <= 60:
            interval = "5m"
        else:
            interval = "1h" 
        
        print(f"Fetching {interval} data for {symbol}...")
        df = ticker.history(start=start_date, end=end_date, interval=interval)
        
        ticks = []
        if df.empty:
            print(f"No data found for {symbol}")
            return ticks
            
        for index, row in df.iterrows():
            # Create a tick from the Close price (simplification)
            # In real system, we might want OHLCV bars directly
            tick = MarketTick(
                symbol=symbol,
                timestamp=index.to_pydatetime(),
                price=float(row['Close']),
                size=int(row['Volume']),
                exchange="Yahoo"
            )
            ticks.append(tick)
            
        print(f"Fetched {len(ticks)} data points.")
        return ticks

    def fetch_historical_df(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Fetches historical data as a Pandas DataFrame with standard columns.
        """
        print(f"Fetching DataFrame for {symbol}...")
        ticker = yf.Ticker(symbol)
        
        # dynamic interval based on range
        interval = "1m"
        duration = end_date - start_date
        if duration.days <= 7:
            interval = "1m"
        elif duration.days <= 60:
            interval = "5m"
        else:
            interval = "1h" 
            
        df = ticker.history(start=start_date, end=end_date, interval=interval)
        
        if df.empty:
            return pd.DataFrame()
            
        # Standardize columns
        df.reset_index(inplace=True)
        # Rename to lowercase
        df.rename(columns={
            "Date": "timestamp", "Datetime": "timestamp",
            "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"
        }, inplace=True)
        
        # Ensure timestamp is stripped of timezone or handled consistently if needed
        # For now, keep as is
        
        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

    def get_current_price(self, symbol: str) -> float:
        ticker = yf.Ticker(symbol)
        # Fast "current" price check (delayed 15m usually)
        return ticker.fast_info.last_price

if __name__ == "__main__":
    # Test
    connector = YahooDataConnector()
    ticks = connector.fetch_historical_ticks("SPY", datetime.now() - timedelta(days=2), datetime.now())
    print(f"Last Price: {ticks[-1].price if ticks else 'N/A'}")
