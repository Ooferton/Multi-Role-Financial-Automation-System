import time
import os
import requests
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

        # Try a few times — yfinance can intermittently fail with JSON/errors from Yahoo
        df = pd.DataFrame()
        for attempt in range(3):
            try:
                df = ticker.history(start=start_date, end=end_date, interval=interval)
                break
            except Exception as e:
                print(f"Attempt {attempt+1} failed: {e}")
                time.sleep(1 + attempt)

        # fallback to yf.download if ticker.history failed or returned malformed data
        if df is None or df.empty:
            try:
                df = yf.download(symbol, start=start_date, end=end_date, interval=interval, progress=False)
            except Exception as e:
                print(f"Fallback download also failed: {e}")
                df = pd.DataFrame()

        # If still empty, try Finnhub REST API as a provider fallback (requires FINNHUB_API_KEY env var)
        if df is None or df.empty:
            api_key = os.getenv('FINNHUB_API_KEY')
            if api_key:
                try:
                    fh_df = self._fetch_from_finnhub(symbol, start_date, end_date, interval, api_key)
                    if fh_df is not None and not fh_df.empty:
                        df = fh_df
                except Exception as e:
                    print(f"Finnhub fallback failed: {e}")
            else:
                print("FINNHUB_API_KEY not set; skipping Finnhub fallback.")
        
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
            
        df = pd.DataFrame()
        for attempt in range(3):
            try:
                df = ticker.history(start=start_date, end=end_date, interval=interval)
                break
            except Exception as e:
                print(f"Attempt {attempt+1} failed: {e}")
                time.sleep(1 + attempt)

        if df is None or df.empty:
            try:
                df = yf.download(symbol, start=start_date, end=end_date, interval=interval, progress=False)
            except Exception as e:
                print(f"Fallback download also failed: {e}")
                df = pd.DataFrame()

        # Try Finnhub if still empty
        if df is None or df.empty:
            api_key = os.getenv('FINNHUB_API_KEY')
            if api_key:
                try:
                    fh_df = self._fetch_from_finnhub(symbol, start_date, end_date, interval, api_key)
                    if fh_df is not None and not fh_df.empty:
                        df = fh_df
                except Exception as e:
                    print(f"Finnhub fallback failed: {e}")
            else:
                print("FINNHUB_API_KEY not set; skipping Finnhub fallback.")
            
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
        try:
            return float(ticker.fast_info.last_price)
        except Exception:
            # fallback: try history for the most recent close
            try:
                df = ticker.history(period="1d", interval="1m")
                if not df.empty:
                    return float(df['Close'].iloc[-1])
            except Exception:
                pass
        raise RuntimeError(f"Unable to fetch current price for {symbol}")

    def _fetch_from_finnhub(self, symbol: str, start_date: datetime, end_date: datetime, interval: str, api_key: str) -> pd.DataFrame:
        """
        Fetch candles from Finnhub as a fallback. Returns a DataFrame with timestamp, open, high, low, close, volume columns.
        Finnhub resolution mapping: '1m'->'1', '5m'->'5', '1h'->'60'
        """
        # Map interval to Finnhub resolution
        res_map = {'1m': '1', '5m': '5', '1h': '60'}
        resolution = res_map.get(interval, '1')

        # Finnhub expects integer unix timestamps (seconds)
        _from = int(start_date.timestamp())
        _to = int(end_date.timestamp())

        url = f"https://finnhub.io/api/v1/stock/candle"
        params = {
            'symbol': symbol,
            'resolution': resolution,
            'from': _from,
            'to': _to,
            'token': api_key
        }

        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        # Finnhub returns {'s':'ok','t':[...], 'c':[...], 'v':[...], 'o':[...], 'h':[...], 'l':[...]} on success
        if data.get('s') != 'ok' or 't' not in data:
            return pd.DataFrame()

        times = [datetime.fromtimestamp(int(t)) for t in data['t']]
        df = pd.DataFrame({
            'timestamp': times,
            'open': data.get('o', []),
            'high': data.get('h', []),
            'low': data.get('l', []),
            'close': data.get('c', []),
            'volume': data.get('v', [])
        })
        return df

if __name__ == "__main__":
    # Test
    connector = YahooDataConnector()
    ticks = connector.fetch_historical_ticks("SPY", datetime.now() - timedelta(days=2), datetime.now())
    print(f"Last Price: {ticks[-1].price if ticks else 'N/A'}")
