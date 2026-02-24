import sqlite3
import pandas as pd
import os

db_path = "data/feature_store.db"

def check_ticks(symbol):
    if not os.path.exists(db_path):
        print("Database not found.")
        return
    
    conn = sqlite3.connect(db_path)
    print(f"Checking ticks for {symbol}...")
    
    # Check count
    count = conn.execute(f"SELECT COUNT(*) FROM ticks WHERE symbol = '{symbol}'").fetchone()[0]
    print(f"Total ticks for {symbol}: {count}")
    
    # Check raw strings for the first few entries to see precision
    raw_ticks = conn.execute(f"SELECT * FROM ticks WHERE symbol = '{symbol}' LIMIT 10").fetchall()
    print("\nRaw Ticks (First 10):")
    for r in raw_ticks:
        print(r)
    
    # Check for multiple ticks per second (indicates OHLC preservation)
    df_precision = pd.read_sql_query(f"SELECT timestamp, COUNT(*) as c FROM ticks WHERE symbol = '{symbol}' GROUP BY timestamp HAVING c > 1 LIMIT 20", conn)
    print("\nTimestamps with multiple entries (should have 4 for WARMUP):")
    print(df_precision)
    
    if count > 0:
        # Check first and last entries
        first = conn.execute(f"SELECT * FROM ticks WHERE symbol = '{symbol}' ORDER BY timestamp ASC LIMIT 1").fetchone()
        last = conn.execute(f"SELECT * FROM ticks WHERE symbol = '{symbol}' ORDER BY timestamp DESC LIMIT 1").fetchone()
        print(f"First entry: {first}")
        print(f"Last entry: {last}")
        
        # Check if there are OHLC-like groups (multiple ticks per timestamp or near each other)
        # Using a subset of data
        df = pd.read_sql_query(f"SELECT timestamp, price FROM ticks WHERE symbol = '{symbol}' ORDER BY timestamp DESC LIMIT 100", conn)
        print("\nLast 100 ticks (tail):")
        print(df.tail(20))
        
    conn.close()

if __name__ == "__main__":
    check_ticks("BAC")
