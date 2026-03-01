import json
import os
import sys

def get_status():
    status_path = os.path.join(os.getcwd(), 'data', 'sentience_status.json')
    if not os.path.exists(status_path):
        print("Trading Engine is currently OFFLINE (no status file found).")
        return

    try:
        with open(status_path, 'r') as f:
            data = json.load(f)
        
        print(f"--- TRADING ENGINE STATUS ---")
        print(f"Status: {'ACTIVE' if data.get('active') else 'INACTIVE'}")
        print(f"Mode: {data.get('mode')}")
        print(f"VIX: {data.get('vix', 'N/A')}")
        print(f"Market Vibe: {data.get('vibe', 'N/A')}")
        print(f"AI Summary: {data.get('macro_summary', 'N/A')}")
        print(f"Sentiment: {data.get('news_verdict', 'N/A')} ({data.get('news_sentiment', 0.0)})")
        
        pulse = data.get('pulse', {})
        warmup = [s for s, v in pulse.items() if v.get('status') == 'WARMUP']
        scanning = [s for s, v in pulse.items() if v.get('status') == 'SCANNING']
        
        if warmup:
            print(f"Currently Warming Up: {', '.join(warmup[:5])}...")
        if scanning:
            print(f"Actively Scanning: {len(scanning)} symbols.")
            
    except Exception as e:
        print(f"Error reading status: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "status":
        get_status()
    else:
        print("Usage: python monitor.py status")
