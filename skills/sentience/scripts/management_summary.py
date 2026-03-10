import json
import os
import pandas as pd
from datetime import datetime

def get_management_summary():
    # 1. System Status
    status_path = os.path.join(os.getcwd(), 'data', 'sentience_status.json')
    status_data = {}
    if os.path.exists(status_path):
        try:
            with open(status_path, 'r') as f:
                status_data = json.load(f)
        except: pass

    # 2. Performance Data
    trades_path = os.path.join(os.getcwd(), 'data', 'trades.csv')
    performance_summary = "No trade history available yet."
    if os.path.exists(trades_path):
        try:
            df = pd.read_csv(trades_path)
            if not df.empty:
                recent = df.tail(5)
                recent_trades = []
                for idx, row in recent.iterrows():
                    recent_trades.append(f"{row['timestamp']} | {row['symbol']} {row['side']} @ {row['price']} | Reasoning: {row['reasoning']}")
                
                performance_summary = f"Total Trades: {len(df)}\nRecent Activity:\n" + "\n".join(recent_trades)
        except: pass

    # 3. Veto Status
    veto_path = os.path.join(os.getcwd(), 'data', 'system_veto.lock')
    veto_status = "ACTIVE (Trading Blocked)" if os.path.exists(veto_path) else "INACTIVE (Trading Allowed)"

    # Construct Output
    output = f"""
=== MANAGER'S OPERATING PICTURE ===
Timestamp: {datetime.now().isoformat()}
System Mode: {status_data.get('mode', 'UNKNOWN')}
Current Vibe: {status_data.get('vibe', 'UNKNOWN')}
VIX: {status_data.get('vix', 'N/A')}
VETO STATUS: {veto_status}

--- PERFORMANCE SUMMARY ---
{performance_summary}
==================================
"""
    print(output)

if __name__ == "__main__":
    get_management_summary()
