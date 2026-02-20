import streamlit as st
import pandas as pd
import sqlite3
import plotly.graph_objects as go
import os

# Page Config
st.set_page_config(page_title="Financial AI Dashboard", layout="wide")
st.title("🤖 Financial AI Orchestrator Dashboard")

# 1. Load Data
db_path = "data/feature_store.db"
trades_path = "data/trades.csv"

def load_market_data():
    if not os.path.exists(db_path):
        return pd.DataFrame()
    conn = sqlite3.connect(db_path)
    # Load last 1000 ticks
    df = pd.read_sql_query("SELECT * FROM ticks ORDER BY timestamp DESC LIMIT 1000", conn)
    conn.close()
    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
    return df

def load_trades():
    if not os.path.exists(trades_path):
        return pd.DataFrame()
    df = pd.read_csv(trades_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

market_df = load_market_data()
trades_df = load_trades()

# 2. Key Metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("System Condition", "SHADOW MODE", delta_color="off")
with col2:
    if not trades_df.empty:
        total_trades = len(trades_df)
        st.metric("Total Trades", total_trades)
    else:
        st.metric("Total Trades", 0)
with col3:
    if not market_df.empty:
        last_price = market_df.iloc[-1]['price']
        st.metric("Live Price", f"${last_price:,.2f}")
    else:
        st.metric("Live Price", "N/A")

# 3. Main Chart
st.subheader("Market Activity & Trade Execution")

if not market_df.empty:
    fig = go.Figure()
    
    # Price Line
    fig.add_trace(go.Scatter(
        x=market_df['timestamp'], 
        y=market_df['price'],
        mode='lines',
        name='Price'
    ))
    
    # Trade Markers
    if not trades_df.empty:
        buys = trades_df[trades_df['side'] == 'BUY']
        sells = trades_df[trades_df['side'] == 'SELL']
        
        fig.add_trace(go.Scatter(
            x=buys['timestamp'], y=buys['price'],
            mode='markers', marker_symbol='triangle-up', marker_color='green', marker_size=10,
            name='Buy'
        ))
        
        fig.add_trace(go.Scatter(
            x=sells['timestamp'], y=sells['price'],
            mode='markers', marker_symbol='triangle-down', marker_color='red', marker_size=10,
            name='Sell'
        ))

    fig.update_layout(height=600, xaxis_title="Time", yaxis_title="Price")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No market data found. Run the simulation/backtester to generate data.")

# 4. Recent Trades Table
st.subheader("Execution Log")
if not trades_df.empty:
    st.dataframe(trades_df.sort_values('timestamp', ascending=False).head(50), use_container_width=True)
else:
    st.info("No trades executed yet.")
