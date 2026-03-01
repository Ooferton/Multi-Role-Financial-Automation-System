import pandas as pd
import os
import sys

def review_performance():
    trades_path = os.path.join(os.getcwd(), 'data', 'trades.csv')
    if not os.path.exists(trades_path):
        print("No trades recorded yet.")
        return

    try:
        df = pd.read_csv(trades_path)
        if df.empty:
            print("Trade log is empty.")
            return

        # Simple Performance analysis
        print(f"--- PERFORMANCE REVIEW (Last {len(df)} Trades) ---")
        buy_count = len(df[df['side'] == 'BUY'])
        sell_count = len(df[df['side'] == 'SELL'])
        total_cost = df['cost'].sum()
        
        print(f"Trade Volume: {len(df)} total ({buy_count} BUY / {sell_count} SELL)")
        print(f"Total Portfolio Exposure (Cost): ${total_cost:.2f}")
        
        # Breakdown by symbol
        symbols = df.groupby('symbol').size().sort_values(ascending=False)
        print("\nTop Managed Assets:")
        for sym, count in symbols.head(5).items():
            print(f" - {sym}: {count} trades")
            
        print("\nRecent AI Reasoning:")
        for idx, row in df.tail(3).iterrows():
            print(f" [{row['timestamp']}] {row['symbol']} {row['side']}: {row['reasoning']}")

    except Exception as e:
        print(f"Error reviewing performance: {e}")

if __name__ == "__main__":
    review_performance()
