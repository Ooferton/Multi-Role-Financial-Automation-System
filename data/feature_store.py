import sqlite3
import pandas as pd
import json
from dataclasses import dataclass
from typing import List, Optional, Dict
from datetime import datetime
import logging

# --- Data Schemas ---

@dataclass
class MarketTick:
    symbol: str
    price: float
    size: float
    timestamp: datetime
    bid_size: float = 0.0
    ask_size: float = 0.0
    exchange: str

@dataclass
class OHLCV:
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

@dataclass
class OrderBookSnapshot:
    symbol: str
    timestamp: datetime
    bids: List[List[float]] # [[price, size], ...]
    asks: List[List[float]]

# --- Feature Store ---

class FeatureStore:
    """
    Centralized data access layer.
    Manages storage and retrieval of raw market data and computed features.
    
    Currently uses SQLite for prototype simplicity, but designed to swap for TimescaleDB/InfluxDB.
    """
    def __init__(self, db_path: str = "data/history.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self._init_db()

    def _init_db(self):
        """Initializes the database schema."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # OHLCV Table
        c.execute('''CREATE TABLE IF NOT EXISTS ohlcv (
                        symbol TEXT,
                        timestamp DATETIME,
                        open REAL,
                        high REAL,
                        low REAL,
                        close REAL,
                        volume REAL,
                        PRIMARY KEY (symbol, timestamp)
                    )''')
        
        # Ticks Table (Ideally this goes to a specialized TSDB)
        c.execute('''CREATE TABLE IF NOT EXISTS ticks (
                        symbol TEXT,
                        timestamp DATETIME,
                        price REAL,
                        size REAL,
                        exchange TEXT,
                        PRIMARY KEY (symbol, timestamp)
                    )''')

        conn.commit()
        conn.close()

    def save_ohlcv(self, data: List[OHLCV]):
        """Batch saves OHLCV data."""
        if not data:
            return
            
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        records = [(d.symbol, d.timestamp, d.open, d.high, d.low, d.close, d.volume) for d in data]
        
        c.executemany('''INSERT OR REPLACE INTO ohlcv 
                         (symbol, timestamp, open, high, low, close, volume) 
                         VALUES (?, ?, ?, ?, ?, ?, ?)''', records)
        
        conn.commit()
        conn.close()
        self.logger.debug(f"Saved {len(data)} OHLCV records.")

    def get_ohlcv(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Retrieves OHLCV data as a Pandas DataFrame."""
        conn = sqlite3.connect(self.db_path)
        
        query = '''SELECT timestamp, open, high, low, close, volume 
                   FROM ohlcv 
                   WHERE symbol = ? AND timestamp BETWEEN ? AND ?
                   ORDER BY timestamp ASC'''
                   
        df = pd.read_sql_query(query, conn, params=(symbol, start_date, end_date), parse_dates=['timestamp'])
        conn.close()
        
        if not df.empty:
            df.set_index('timestamp', inplace=True)
            
        return df

    async def ingest_tick(self, tick: MarketTick):
        """
        Async ingestion for high-frequency data.
        Writes to an in-memory buffer and flushes periodically or when full.
        """
        # For now, we'll write directly to SQLite to keep it simple but async-compatible
        # In a real HFT system, this would push to Redis or a ring buffer
        # simple synchronous write wrapped in async for interface compatibility
        self.save_tick(tick)

    def save_tick(self, tick: MarketTick):
        """Saves a single tick with deduplication."""
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute("INSERT OR REPLACE INTO ticks (symbol, timestamp, price, size, exchange) VALUES (?, ?, ?, ?, ?)",
                      (tick.symbol, tick.timestamp, tick.price, tick.size, tick.exchange))
            conn.commit()
            conn.close()
        except Exception as e:
            self.logger.error(f"Failed to save tick: {e}")
