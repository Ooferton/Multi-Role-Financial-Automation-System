import streamlit as st
import pandas as pd
import sqlite3
import plotly.graph_objects as go
import os
import json
import yaml
import time
import pytz
from datetime import datetime

# Page Config
st.set_page_config(page_title="Sentience V3 | MARL Swarm Dashboard", layout="wide", page_icon="🧠")
st.title("🧠 Sentience V3 | MARL Swarm Orchestrator")

# 1. Load Data/Config
db_path = "data/feature_store.db"
trades_path = "data/trades.csv"
journal_path = "logs/ai_journal.md"
status_path = "data/sentience_status.json"
config_path = "config/config.yaml"

def load_config():
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return {}

def load_sentience_status():
    if os.path.exists(status_path):
        try:
            with open(status_path, "r", encoding="utf-8", errors="replace") as f:
                return json.load(f)
        except: pass
    return {}

def load_swarm_status():
    """Load MARL Swarm status for all sector agents."""
    swarm_path = "data/swarm_status.json"
    if os.path.exists(swarm_path):
        try:
            with open(swarm_path, "r", encoding="utf-8", errors="replace") as f:
                return json.load(f)
        except: pass
    return {}

def get_v3_model_info():
    """Check if V3 model is present and return metadata."""
    v3_path = "ml/models/ppo_v3_cyborg.zip"
    if os.path.exists(v3_path):
        size_mb = os.path.getsize(v3_path) / (1024 * 1024)
        mtime = os.path.getmtime(v3_path)
        trained_at = datetime.fromtimestamp(mtime).strftime("%m/%d %H:%M")
        return {"loaded": True, "size_mb": round(size_mb, 1), "trained_at": trained_at}
    return {"loaded": False}

def load_ai_journal():
    if os.path.exists(journal_path):
        try:
            with open(journal_path, "r", encoding="utf-8", errors="replace") as f:
                return f.read()
        except Exception as e:
            return f"Error loading journal: {e}"
    return "No journal entries yet."

def load_market_data(symbol=None):
    if not os.path.exists(db_path):
        return pd.DataFrame()
    conn = sqlite3.connect(db_path)
    # Use SQLite's native UTC datetime to avoid string formatting issues
    query = f"SELECT * FROM ticks WHERE symbol = '{symbol}' AND timestamp > datetime('now', '-3 days') ORDER BY timestamp ASC"
    df = pd.read_sql_query(query, conn)
    conn.close()
    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', utc=True)
        df = df.sort_values('timestamp')
    return df

def load_trades(symbol=None):
    if not os.path.exists(trades_path):
        return pd.DataFrame()
    df = pd.read_csv(trades_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', utc=True)
    if symbol:
        df = df[df['symbol'] == symbol]
    return df

def get_all_symbols():
    if not os.path.exists(db_path):
        return ["SPY"]
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query("SELECT DISTINCT symbol FROM ticks", conn)
        symbols = sorted(df['symbol'].tolist())
        return symbols if symbols else ["SPY"]
    except:
        return ["SPY"]
    finally:
        conn.close()

config = load_config()
sentience = load_sentience_status()
swarm_status = load_swarm_status()
v3_info = get_v3_model_info()
all_symbols = get_all_symbols()

# 2. Sidebar Controls
st.sidebar.header("🕹️ Controls")
selected_ticker = st.sidebar.selectbox("Select Ticker", all_symbols, index=0)

# Emergency System
st.sidebar.divider()
st.sidebar.subheader("🚨 Emergency System")
lock_file = "data/circuit_breaker.lock"
is_triggered = os.path.exists(lock_file)

if not is_triggered:
    if st.sidebar.button("TRIGGER CIRCUIT BREAKER", type="primary", width="stretch"):
        st.sidebar.warning("Halting System & Liquidating...")
        
        # 1. Create Lock File IMMEDIATELY to stop the runners
        with open(lock_file, "w") as f:
            f.write(f"Triggered via dashboard at {datetime.now()}")
        
        # 2. Attempt Liquidation
        try:
            from agents.alpaca_broker import AlpacaBroker
            b = AlpacaBroker()
            b.liquidate_all()
            st.sidebar.success("SENTIENCE HALTED. All positions closed.")
        except Exception as e:
            st.sidebar.error(f"Halted, but liquidation failed: {e}")
            
        time.sleep(2)
        st.rerun()
else:
    st.sidebar.error("SYSTEM HALTED")
    if st.sidebar.button("Reset Circuit Breaker", width="stretch"):
        if os.path.exists(lock_file):
            os.remove(lock_file)
            st.sidebar.success("System Reset. Ready for operation.")
            time.sleep(2)
            st.rerun()

# 2. Multi-Process Metrics Bar
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    portfolio_active = sentience.get("active", False)
    st.metric("🤖 MARL Swarm", "ACTIVE" if portfolio_active else "OFFLINE",
              delta=f"PID {sentience.get('pid', 'N/A')}" if portfolio_active else None)
with col2:
    if v3_info["loaded"]:
        st.metric("🧠 V3 Model", "LOADED",
                  delta=f"{v3_info['size_mb']} MB | {v3_info['trained_at']}")
    else:
        st.metric("🧠 V3 Model", "NOT FOUND", delta="Run train_v3.py", delta_color="inverse")
with col3:
    vibe = sentience.get("vibe", "Neutral")
    st.metric("📊 Economic Vibe", vibe)
with col4:
    sentiment = sentience.get("news_sentiment", 0)
    verdict = sentience.get("news_verdict", "Neutral")
    st.metric("📰 News Sentiment", f"{verdict}", delta=f"{sentiment:.2f}")
with col5:
    full_trades = load_trades()
    total_trades = len(full_trades) if not full_trades.empty else 0
    st.metric("💼 Total Trades", total_trades)

# 2b. MARL Swarm Sector Overview
st.divider()
swarm_cols = st.columns(5)
sectors = [
    ("💻 TECH",    ["AAPL", "MSFT", "NVDA", "GOOGL", "AMD"]),
    ("🏦 FINANCE", ["JPM", "BAC", "GS", "V", "MA"]),
    ("🏥 HEALTH",  ["JNJ", "UNH", "PFE", "LLY", "MRK"]),
    ("🛢️ ENERGY",  ["XOM", "CVX", "COP", "SLB", "LIN"]),
    ("₿ CRYPTO",  ["BTC/USD", "ETH/USD", "DOGE/USD", "COIN", "MSTR"]),
]
for i, (sector_name, tickers) in enumerate(sectors):
    with swarm_cols[i]:
        # Compute trade count for this sector
        sector_trades = 0
        if not full_trades.empty and 'symbol' in full_trades.columns:
            sector_trades = full_trades[full_trades['symbol'].isin(tickers)].shape[0]
        # Allocation from swarm_status if available
        alloc = swarm_status.get(sector_name.split()[-1], {}).get("allocation", "—")
        st.metric(sector_name, f"{sector_trades} trades", delta=f"Alloc: {alloc}")

st.divider()

# 3. Main Dashboard Layout (Split)
main_col, side_col = st.columns([2, 1])

with main_col:
    st.subheader(f"📊 {selected_ticker} Activity & Trade Execution")

    market_df = load_market_data(selected_ticker)
    trades_df = load_trades(selected_ticker)
    if not market_df.empty:
        # Candle Aggregation (Resample ticks to 1-Min Bars)
        df_resampled = market_df.set_index('timestamp').resample('1min').agg({
            'price': ['first', 'max', 'min', 'last'],
            'size': 'sum'
        })
        df_resampled.columns = ['open', 'high', 'low', 'close', 'volume']
        df_resampled = df_resampled.dropna().reset_index()

        hide_gaps = st.sidebar.checkbox("Hide Market Gaps (Weekend/Night)", value=False)
        
        fig = go.Figure()
        
        # 1. Candlestick Trace
        fig.add_trace(go.Candlestick(
            x=df_resampled['timestamp'].dt.strftime('%m-%d %H:%M') if hide_gaps else df_resampled['timestamp'],
            open=df_resampled['open'],
            high=df_resampled['high'],
            low=df_resampled['low'],
            close=df_resampled['close'],
            name='OHLC'
        ))
        
        # 2. Trade Markers
        if not trades_df.empty:
            buys = trades_df[trades_df['side'] == 'BUY']
            sells = trades_df[trades_df['side'] == 'SELL']
            if not buys.empty:
                fig.add_trace(go.Scatter(x=buys['timestamp'].dt.strftime('%m-%d %H:%M') if hide_gaps else buys['timestamp'], 
                                         y=buys['price'], mode='markers', 
                                         marker_symbol='triangle-up', marker_color='#00ff88', marker_size=12, 
                                         name='Buy', text=buys['reasoning'], hoverinfo='text+x+y'))
            if not sells.empty:
                fig.add_trace(go.Scatter(x=sells['timestamp'].dt.strftime('%m-%d %H:%M') if hide_gaps else sells['timestamp'], 
                                         y=sells['price'], mode='markers', 
                                         marker_symbol='triangle-down', marker_color='#ff3366', marker_size=12, 
                                         name='Sell', text=sells['reasoning'], hoverinfo='text+x+y'))
        
        fig.update_layout(
            template="plotly_dark",
            height=500, 
            xaxis_rangeslider_visible=False,
            xaxis_type='category' if hide_gaps else 'date',
            xaxis_title="Time", 
            yaxis_title="Price", 
            margin=dict(l=0, r=0, t=30, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, width="stretch")
    else:
        st.info(f"Waiting for market data for {selected_ticker}...")

    # 4. Debug Data Inspection
    with st.sidebar.expander("🛠️ Debug Data View"):
        if not market_df.empty:
            st.write(f"Raw Ticks: {len(market_df)}")
            st.dataframe(market_df.tail(10))
            if 'df_resampled' in locals():
                st.write(f"OHLC Bars: {len(df_resampled)}")
                st.dataframe(df_resampled.tail(5))
        else:
            st.write("No market data loaded.")

    st.subheader(f"📜 {selected_ticker} Execution Log")
    if not trades_df.empty:
        st.dataframe(trades_df.sort_values('timestamp', ascending=False).head(20), width="stretch")
    else:
        st.info(f"No trades executed for {selected_ticker} yet.")

with side_col:
    st.subheader("🧠 Inner Monologue")
    journal_text = load_ai_journal()
    # Display the tail of the journal for focus
    st.markdown(f'<div style="height: 600px; overflow-y: scroll; border: 1px solid #444; padding: 10px; border-radius: 5px;">{journal_text}</div>', unsafe_allow_html=True)
    
    if sentience.get("macro_summary"):
        st.info(f"**Economist Outlook:** {sentience['macro_summary']}")

    st.subheader("📡 MARL Swarm Activity")
    combined_pulse = sentience.get("pulse", {})
    
    crypto_symbols  = {"BTC/USD", "ETH/USD", "DOGE/USD", "IBIT", "BITO", "COIN", "MSTR", "MARA", "RIOT"}
    finance_symbols = {"JPM", "BAC", "WFC", "C", "GS", "MS", "V", "MA", "AXP", "BLK"}
    health_symbols  = {"JNJ", "UNH", "PFE", "ABBV", "TMO", "MRK", "DHR", "LLY"}
    energy_symbols  = {"XOM", "CVX", "COP", "SLB", "LIN", "NEM"}
    
    def get_sector(sym):
        if sym in crypto_symbols:  return "₿ CRYPTO"
        if sym in finance_symbols: return "🏦 FINANCE"
        if sym in health_symbols:  return "🏥 HEALTH"
        if sym in energy_symbols:  return "🛢️ ENERGY"
        return "💻 TECH"
    
    if combined_pulse:
        pulse_data = []
        for sym, data in combined_pulse.items():
            pulse_data.append({
                "Sector": get_sector(sym),
                "Symbol": sym,
                "Status": data.get("status"),
                "Detail": data.get("detail"),
                "Time": data.get("timestamp")
            })
        pulse_df = pd.DataFrame(pulse_data)
        if not pulse_df.empty:
            st.dataframe(pulse_df.sort_values("Time", ascending=False), height=400, width="stretch")
    else:
        st.info("No activity pulses received yet.")

# Auto-refresh logic (5s)
if st.checkbox("Enable Auto-Refresh (5s)", value=True):
    time.sleep(5)
    st.rerun()
