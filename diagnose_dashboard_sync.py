import os
import json
import sqlite3
import time
from datetime import datetime

def check_sync():
    print("Dashboard Sync Due Diligence")
    print("=" * 30)

    # 1. Check sentience_status.json
    status_path = "data/sentience_status.json"
    if os.path.exists(status_path):
        mtime = os.path.getmtime(status_path)
        last_write = datetime.fromtimestamp(mtime)
        latency = time.time() - mtime
        
        print(f"[FILE] sentience_status.json")
        print(f"  - Last Modified: {last_write}")
        print(f"  - Latency: {latency:.2f} seconds")
        
        try:
            with open(status_path, "r") as f:
                data = json.load(f)
                print(f"  - Vibe: {data.get('vibe')}")
                print(f"  - News Score: {data.get('news_sentiment')}")
                print(f"  - Pulse Count: {len(data.get('pulse', {}))}")
        except Exception as e:
            print(f"  - Error reading JSON: {e}")
    else:
        print("[!] sentience_status.json MISSING")

    # 2. Check feature_store.db
    db_path = "data/feature_store.db"
    if os.path.exists(db_path):
        print(f"\n[DB] feature_store.db")
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Check last tick
            cursor.execute("SELECT timestamp, symbol, price FROM ticks ORDER BY timestamp DESC LIMIT 1")
            row = cursor.fetchone()
            if row:
                ts, sym, price = row
                print(f"  - Last Tick: {ts} | {sym} | ${price}")
                # Parse timestamp - usually '2026-02-21 22:45:01.123'
                try:
                    tick_ts = datetime.fromisoformat(ts.replace(' ', 'T'))
                    db_latency = (datetime.now() - tick_ts).total_seconds()
                    # print(f"  - DB Latency: {db_latency:.2f} seconds")
                except:
                    pass
            else:
                print("  - No ticks found in DB.")
            
            # Check row count
            cursor.execute("SELECT COUNT(*) FROM ticks")
            count = cursor.fetchone()[0]
            print(f"  - Total Ticks in DB: {count}")
            
            conn.close()
        except Exception as e:
            print(f"  - Error reading DB: {e}")
    else:
        print("[!] feature_store.db MISSING")

    # 3. Check ai_journal.md
    journal_path = "logs/ai_journal.md"
    if os.path.exists(journal_path):
        mtime = os.path.getmtime(journal_path)
        print(f"\n[LOG] ai_journal.md")
        print(f"  - Last Entry: {datetime.fromtimestamp(mtime)}")
    else:
        print("\n[!] ai_journal.md MISSING")

    print("=" * 30)

if __name__ == "__main__":
    check_sync()
