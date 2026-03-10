import streamlit as st
import pandas as pd
import sqlite3
import plotly.graph_objects as go
import plotly.express as px
import os
import json
import yaml
import time
import time
from datetime import datetime

# Sentience Core Imports
from core.llm_supervisor import LLMSupervisor
from core.tool_registry import ToolRegistry
from core.agent_router import AgentRouter
from core.agent_memory import AgentMemory
from agents.alpaca_broker import AlpacaBroker
from data.feature_store import FeatureStore
from core.risk_manager import RiskManager
from core.telegram_notifier import TelegramNotifier
from ml.research_lab import run_regime_detection, run_factor_ranking, run_cointegration_scan, run_backtest

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Sentience V3 | MARL Swarm",
    layout="wide",
    page_icon="🧠",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# GLOBAL THEME
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background: linear-gradient(135deg, #0a0a1a 0%, #0d1117 50%, #0a0f23 100%); }

    /* ── Sidebar as Module Navigator ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #080d1a 100%);
        border-right: 1px solid rgba(99,102,241,0.25);
    }
    section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }

    /* Module nav buttons */
    .module-nav-btn {
        display: flex; align-items: center; gap: 12px;
        padding: 11px 16px; border-radius: 10px; margin-bottom: 4px;
        cursor: pointer; font-size: 0.88rem; font-weight: 500; color: #64748b;
        border: 1px solid transparent; transition: all 0.2s ease;
        text-decoration: none;
    }
    .module-nav-btn:hover { background: rgba(99,102,241,0.1); color: #c4b5fd; border-color: rgba(99,102,241,0.2); }
    .module-nav-btn.active { background: linear-gradient(135deg,rgba(99,102,241,0.2),rgba(139,92,246,0.2)); color: #a78bfa; border-color: rgba(99,102,241,0.4); }
    .module-badge { font-size:0.65rem; background:rgba(99,102,241,0.25); color:#a78bfa; border-radius:999px; padding:1px 7px; margin-left:auto; }
    .module-badge-soon { font-size:0.65rem; background:rgba(30,30,50,0.8); color:#475569; border-radius:999px; padding:1px 7px; margin-left:auto; border:1px solid #1e293b; }

    /* ── Metrics ── */
    div[data-testid="metric-container"] {
        background: rgba(255,255,255,0.03); border: 1px solid rgba(99,102,241,0.25);
        border-radius: 12px; padding: 14px 18px; backdrop-filter: blur(12px);
        transition: border-color 0.2s, box-shadow 0.2s;
    }
    div[data-testid="metric-container"]:hover { border-color:rgba(99,102,241,0.6); box-shadow:0 0 18px rgba(99,102,241,0.15); }
    div[data-testid="metric-container"] label { color:#94a3b8!important; font-size:0.72rem!important; letter-spacing:0.08em; text-transform:uppercase; }
    div[data-testid="metric-container"] [data-testid="stMetricValue"] { color:#f1f5f9!important; font-size:1.35rem!important; font-weight:600; }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255,255,255,0.02); border-radius: 12px;
        border: 1px solid rgba(99,102,241,0.15); padding: 4px; gap: 4px;
    }
    .stTabs [data-baseweb="tab"] { border-radius:8px; color:#64748b!important; font-weight:500; font-size:0.83rem; padding:8px 18px; transition:all 0.2s; }
    .stTabs [aria-selected="true"] { background: linear-gradient(135deg,#6366f1,#8b5cf6)!important; color:#fff!important; }

    /* ── Dataframes ── */
    .stDataFrame { border-radius:10px; overflow:hidden; border:1px solid rgba(99,102,241,0.12); }
    .stDataFrame thead tr th { background:rgba(99,102,241,0.12)!important; color:#c4b5fd!important; font-size:0.73rem; }

    /* ── Buttons ── */
    .stButton>button { background:linear-gradient(135deg,#6366f1,#8b5cf6); border:none; border-radius:8px; color:white; font-weight:600; transition:all 0.2s; }
    .stButton>button:hover { transform:translateY(-1px); box-shadow:0 8px 25px rgba(99,102,241,0.4); }
    .stButton>button[kind="primary"] { background:linear-gradient(135deg,#ef4444,#b91c1c)!important; box-shadow:0 0 18px rgba(239,68,68,0.3); }

    /* ── Utility ── */
    .section-hdr { font-size:0.75rem; font-weight:600; color:#6366f1; letter-spacing:0.1em; text-transform:uppercase; margin-bottom:10px; padding-bottom:6px; border-bottom:1px solid rgba(99,102,241,0.2); }
    .glass-card { background:rgba(255,255,255,0.03); border:1px solid rgba(99,102,241,0.18); border-radius:14px; padding:20px; backdrop-filter:blur(12px); margin-bottom:14px; }
    .journal-box { background:rgba(0,0,0,0.3); border:1px solid rgba(99,102,241,0.15); border-radius:10px; padding:14px; height:420px; overflow-y:auto; font-family:'JetBrains Mono',monospace; font-size:0.76rem; color:#94a3b8; line-height:1.65; }
    .status-active { background:rgba(34,197,94,0.12); color:#4ade80; border:1px solid rgba(34,197,94,0.3); border-radius:999px; padding:2px 10px; font-size:0.72rem; font-weight:600; }
    .status-offline { background:rgba(239,68,68,0.12); color:#f87171; border:1px solid rgba(239,68,68,0.3); border-radius:999px; padding:2px 10px; font-size:0.72rem; font-weight:600; }
    .coming-tag { font-size:0.65rem; background:rgba(30,30,50,0.9); color:#475569; border:1px solid #1e293b; border-radius:999px; padding:1px 8px; margin-left:8px; }

    hr { border-color:rgba(99,102,241,0.15)!important; }
    ::-webkit-scrollbar { width:4px; height:4px; }
    ::-webkit-scrollbar-track { background:transparent; }
    ::-webkit-scrollbar-thumb { background:rgba(99,102,241,0.4); border-radius:4px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DATA PATHS
# ─────────────────────────────────────────────
db_path     = "data/feature_store.db"
trades_path = "data/trades.csv"
journal_path= "logs/ai_journal.md"
status_path = "data/sentience_status.json"
config_path = "config/config.yaml"
lock_file   = "data/circuit_breaker.lock"

# ─────────────────────────────────────────────
# DATA LOADERS
# ─────────────────────────────────────────────
def load_config():
    if os.path.exists(config_path):
        with open(config_path,"r") as f: return yaml.safe_load(f)
    return {}

@st.cache_data(ttl=5)
def load_sentience_status():
    if os.path.exists(status_path):
        try:
            with open(status_path,"r",encoding="utf-8",errors="replace") as f: return json.load(f)
        except: pass
    return {}

@st.cache_data(ttl=5)
def load_swarm_status():
    p = "data/swarm_status.json"
    if os.path.exists(p):
        try:
            with open(p,"r",encoding="utf-8",errors="replace") as f: return json.load(f)
        except: pass
    return {}

def get_v3_model_info():
    p = "ml/models/ppo_v3_cyborg.zip"
    if os.path.exists(p):
        mb = os.path.getsize(p)/(1024*1024)
        ts = datetime.fromtimestamp(os.path.getmtime(p)).strftime("%m/%d %H:%M")
        return {"loaded":True,"size_mb":round(mb,1),"trained_at":ts}
    return {"loaded":False}

def get_sentience_model_info():
    p = "ml/models/sentience_core_lora"
    if os.path.exists(os.path.join(p, "adapter_model.safetensors")):
        mb = sum(os.path.getsize(os.path.join(p, f)) for f in os.listdir(p) if os.path.isfile(os.path.join(p, f)))/(1024*1024)
        ts = datetime.fromtimestamp(os.path.getmtime(os.path.join(p, "adapter_model.safetensors"))).strftime("%m/%d %H:%M")
        return {"loaded":True,"size_mb":round(mb,1),"trained_at":ts}
    return {"loaded":False}

@st.cache_data(ttl=5)
def load_ai_journal():
    if os.path.exists(journal_path):
        try:
            with open(journal_path,"r",encoding="utf-8",errors="replace") as f: return f.read()
        except Exception as e: return f"Error: {e}"
    return "No journal entries yet."

@st.cache_data(ttl=5)
def load_market_data(symbol=None):
    if not os.path.exists(db_path): return pd.DataFrame()
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(f"SELECT * FROM ticks WHERE symbol='{symbol}' AND timestamp>datetime('now','-3 days') ORDER BY timestamp ASC", conn)
    conn.close()
    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'],format='mixed',utc=True)
        df = df.sort_values('timestamp')
    return df

@st.cache_data(ttl=5)
def load_trades(symbol=None):
    if not os.path.exists(trades_path): return pd.DataFrame()
    df = pd.read_csv(trades_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'],format='mixed',utc=True)
    if symbol: df = df[df['symbol']==symbol]
    return df

@st.cache_data(ttl=30)
def get_all_symbols():
    if not os.path.exists(db_path): return ["SPY"]
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query("SELECT DISTINCT symbol FROM ticks",conn)
        s = sorted(df['symbol'].tolist())
        return s if s else ["SPY"]
    except: return ["SPY"]
    finally: conn.close()

def get_sector(sym):
    crypto  = {"BTC/USD","ETH/USD","DOGE/USD","IBIT","BITO","COIN","MSTR","MARA","RIOT"}
    finance = {"JPM","BAC","WFC","C","GS","MS","V","MA","AXP","BLK"}
    health  = {"JNJ","UNH","PFE","ABBV","TMO","MRK","DHR","LLY"}
    energy  = {"XOM","CVX","COP","SLB","LIN","NEM"}
    if sym in crypto:  return "₿ CRYPTO"
    if sym in finance: return "🏦 FINANCE"
    if sym in health:  return "🏥 HEALTH"
    if sym in energy:  return "🛢️ ENERGY"
    return "💻 TECH"

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
config      = load_config()
sentience   = load_sentience_status()
swarm       = load_swarm_status()
v3_info     = get_v3_model_info()
sentience_info = get_sentience_model_info()
ai_journal  = load_ai_journal()
all_symbs   = get_all_symbols()
full_trades = load_trades()
is_halted   = os.path.exists(lock_file)

# ─────────────────────────────────────────────
# MODULE DEFINITIONS
# ─────────────────────────────────────────────
MODULES = [
    {"id": "stocks",      "icon": "📈", "label": "Stock Trading",    "live": True},
    {"id": "portfolio",   "icon": "💼", "label": "Portfolio",         "live": True},
    {"id": "research",    "icon": "🔬", "label": "Research Lab",      "live": True},
    {"id": "real_estate", "icon": "🏠", "label": "Real Estate",       "live": False},
    {"id": "debt",        "icon": "💳", "label": "Debt Optimizer",    "live": False},
    {"id": "settings",    "icon": "⚙️",  "label": "Settings",          "live": True},
]

# ─────────────────────────────────────────────
# SIDEBAR — MODULE NAVIGATOR
# ─────────────────────────────────────────────
with st.sidebar:
    # Logo
    st.markdown("""
    <div style="padding:20px 4px 16px 4px;">
        <div style="font-size:1.6rem; font-weight:700; background:linear-gradient(135deg,#6366f1,#a78bfa);
             -webkit-background-clip:text;-webkit-text-fill-color:transparent;">🧠 Sentience</div>
        <div style="font-size:0.72rem; color:#475569; margin-top:2px; letter-spacing:0.08em;">V3 · MARL SWARM PLATFORM</div>
    </div>
    """, unsafe_allow_html=True)

    # Swarm status pill
    swarm_active = sentience.get("active", False)
    pill = '<span class="status-active">● SWARM ONLINE</span>' if swarm_active else '<span class="status-offline">● SWARM OFFLINE</span>'
    st.markdown(f'<div style="padding:0 4px 12px 4px;">{pill}</div>', unsafe_allow_html=True)

    st.divider()

    # Module nav
    st.markdown('<div style="font-size:0.68rem;color:#374151;letter-spacing:0.12em;text-transform:uppercase;padding:0 4px 8px 4px;font-weight:600;">Modules</div>', unsafe_allow_html=True)

    if "active_module" not in st.session_state:
        st.session_state.active_module = "stocks"

    # Initialize Sentience Core
    if "agent" not in st.session_state:
        try:
            broker = AlpacaBroker()
            fs = FeatureStore()
            rm = RiskManager(load_config())
            tn = TelegramNotifier()
            
            supervisor = LLMSupervisor()
            registry = ToolRegistry(broker, fs, rm, tn)
            memory = AgentMemory()
            
            st.session_state.agent = AgentRouter(supervisor, registry)
            st.session_state.memory = memory
        except Exception as e:
            st.error(f"Sentience Core Init Failed: {e}")

    for mod in MODULES:
        badge = '<span class="module-badge">LIVE</span>' if mod["live"] else '<span class="module-badge-soon">SOON</span>'
        is_active = st.session_state.active_module == mod["id"]
        active_cls = "active" if is_active else ""
        if st.button(
            f'{mod["icon"]}  {mod["label"]}',
            key=f'mod_{mod["id"]}',
            use_container_width=True,
            disabled=(not mod["live"] and not is_active)
        ):
            st.session_state.active_module = mod["id"]
            st.rerun()

    # Emergency system (always visible in sidebar — critical action)
    st.markdown('<div style="font-size:0.68rem;color:#374151;letter-spacing:0.12em;text-transform:uppercase;padding:0 4px 8px 4px;font-weight:600;">Emergency</div>', unsafe_allow_html=True)
    if not is_halted:
        if st.button("🚨 CIRCUIT BREAKER", type="primary", use_container_width=True):
            with open(lock_file,"w") as f: f.write(f"Triggered at {datetime.now()}")
            try:
                from agents.alpaca_broker import AlpacaBroker
                AlpacaBroker().liquidate_all()
                st.success("All positions closed.")
            except Exception as e:
                st.error(f"Liquidation failed: {e}")
            time.sleep(2); st.rerun()
    else:
        st.error("⛔ SYSTEM HALTED")
        if st.button("↺ Reset Breaker", use_container_width=True):
            if os.path.exists(lock_file): os.remove(lock_file)
            time.sleep(1); st.rerun()

    st.divider()

    # Auto-refresh at the bottom
    auto_refresh = st.checkbox("🔄 Auto-refresh (5s)", value=True)


# ─────────────────────────────────────────────
# MAIN CONTENT — MODULE ROUTER
# ─────────────────────────────────────────────
active = st.session_state.get("active_module", "stocks")

# ══════════════════════════════════════════════
# MODULE: STOCK TRADING
# ══════════════════════════════════════════════
if active == "stocks":
    # Global header
    hc1, hc2 = st.columns([3,1])
    with hc1:
        st.markdown("""
        <div style="padding:6px 0 2px 0;">
            <span style="font-size:2rem;font-weight:700;background:linear-gradient(135deg,#6366f1,#a78bfa,#38bdf8);
            -webkit-background-clip:text;-webkit-text-fill-color:transparent;">📈 Stock Trading</span>
            <span style="font-size:0.85rem;color:#475569;margin-left:10px;">MARL Swarm · V3 Cyborg Brain</span>
        </div>
        """, unsafe_allow_html=True)
    with hc2:
        st.markdown(f'<div style="text-align:right;padding-top:14px;margin-bottom:10px;">{pill}</div>', unsafe_allow_html=True)
        # Strategy Mode Override positioned in the top right of the main view
        mode_path = "data/trading_mode.json"
        current_mode = "Auto / Dynamic"
        if os.path.exists(mode_path):
            try:
                with open(mode_path, "r") as f:
                    saved = json.load(f)
                    current_mode = saved.get("mode", "Auto / Dynamic")
            except: pass
        
        modes = ["Auto / Dynamic", "Day Trading", "High Frequency Trading", "Swing Trading"]
        try: idx = modes.index(current_mode)
        except: idx = 0
            
        new_mode = st.selectbox("⚡ Active Strategy Override", options=modes, index=idx, label_visibility="collapsed")
        if new_mode != current_mode:
            with open(mode_path, "w") as f:
                json.dump({"mode": new_mode, "updated_at": str(datetime.now())}, f)
            st.toast(f"Swarm Strategy updated to: {new_mode}", icon="✅")
            time.sleep(1)
            st.rerun()

    st.divider()

    # ── Top metrics
    m1,m2,m3,m4,m5,m6 = st.columns(6)
    with m1:
        st.metric("🤖 MARL Swarm","ACTIVE" if swarm_active else "OFFLINE",
                  delta=f"PID {sentience.get('pid','N/A')}" if swarm_active else None)
    with m2:
        if v3_info.get("loaded"):
            st.metric("🧠 V3 Brain","LOADED",delta=f"{v3_info['size_mb']} MB · {v3_info['trained_at']}")
        else:
            st.metric("🧠 V3 Brain","NOT FOUND",delta="Run train_v3.py",delta_color="inverse")
    with m3:
        st.metric("📊 Market Vibe", sentience.get("vibe","Neutral"))
    with m4:
        st.metric("📰 Sentiment", sentience.get("news_verdict","Neutral"),
                  delta=f"{sentience.get('news_sentiment',0):.2f}")
    with m5:
        total_trades = len(full_trades) if not full_trades.empty else 0
        st.metric("💼 Trades", total_trades)
    with m6:
        pnl = full_trades['pnl'].sum() if not full_trades.empty and 'pnl' in full_trades.columns else 0
        st.metric("💰 Total P&L", f"${pnl:,.2f}")

    st.divider()

    # ── Sector bar
    st.markdown('<div class="section-hdr">MARL Swarm — Sector Allocation</div>', unsafe_allow_html=True)
    sector_defs = [
        ("💻 TECH",   ["AAPL","MSFT","NVDA","GOOGL","AMD","CRM","AVGO","INTC","CSCO","META"]),
        ("🏦 FIN",    ["JPM","BAC","GS","V","MA","WFC","C","MS","AXP","BLK"]),
        ("🏥 HEALTH", ["JNJ","UNH","PFE","LLY","MRK","ABBV","TMO","DHR"]),
        ("🛢️ ENERGY", ["XOM","CVX","COP","SLB","LIN","NEM"]),
        ("₿ CRYPTO", ["BTC/USD","ETH/USD","DOGE/USD","COIN","MSTR","MARA","RIOT","IBIT","BITO"]),
    ]
    sc = st.columns(5)
    for i,(name,tickers) in enumerate(sector_defs):
        with sc[i]:
            st = sc[i]  # reuse scoping trick avoided — using index directly
            n_t = full_trades[full_trades['symbol'].isin(tickers)].shape[0] if not full_trades.empty and 'symbol' in full_trades.columns else 0
            key = name.split()[-1]
            alloc = swarm.get(key,{}).get("allocation","—")
            sharpe= swarm.get(key,{}).get("sharpe",None)
            sc[i].metric(name, f"{n_t} trades", delta=f"Alloc: {alloc} · Sharpe: {sharpe:.2f}" if sharpe else f"Alloc: {alloc}")

    # ── restore st
    import streamlit as st  # re-import after column scope trick
    st.divider()

    # ── Inner tabs
    tab_chart, tab_swarm, tab_journal = st.tabs([
        "📊 Price & Trades",
        "🤖 Swarm Monitor",
        "🧠 AI Journal",
    ])

    # ─── TAB: Price & Trades
    with tab_chart:
        # Controls inline
        ctrl1, ctrl2, ctrl3 = st.columns([2,1,1])
        with ctrl1:
            selected_ticker = st.selectbox("Active Ticker", all_symbs, index=0, label_visibility="collapsed")
        with ctrl2:
            hide_gaps = st.checkbox("Hide Market Gaps", value=False)
        with ctrl3:
            st.markdown(f'<div style="padding-top:8px; color:#64748b; font-size:0.8rem;">Sector: <b style="color:#a78bfa">{get_sector(selected_ticker)}</b></div>', unsafe_allow_html=True)

        main_col, side_col = st.columns([2,1])

        with main_col:
            market_df = load_market_data(selected_ticker)
            trades_df = load_trades(selected_ticker)

            if not market_df.empty:
                df_r = market_df.set_index('timestamp').resample('1min').agg({'price':['first','max','min','last'],'size':'sum'})
                df_r.columns = ['open','high','low','close','volume']
                df_r = df_r.dropna().reset_index()

                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=df_r['timestamp'].dt.strftime('%m-%d %H:%M') if hide_gaps else df_r['timestamp'],
                    open=df_r['open'], high=df_r['high'], low=df_r['low'], close=df_r['close'],
                    increasing_line_color='#4ade80', decreasing_line_color='#f87171', name='OHLC'
                ))
                if not trades_df.empty:
                    buys  = trades_df[trades_df['side']=='BUY']
                    sells = trades_df[trades_df['side']=='SELL']
                    if not buys.empty:
                        fig.add_trace(go.Scatter(
                            x=buys['timestamp'].dt.strftime('%m-%d %H:%M') if hide_gaps else buys['timestamp'],
                            y=buys['price'], mode='markers',
                            marker=dict(symbol='triangle-up',color='#4ade80',size=14,line=dict(color='#166534',width=1)),
                            name='BUY', text=buys.get('reasoning',''), hoverinfo='text+x+y'))
                    if not sells.empty:
                        fig.add_trace(go.Scatter(
                            x=sells['timestamp'].dt.strftime('%m-%d %H:%M') if hide_gaps else sells['timestamp'],
                            y=sells['price'], mode='markers',
                            marker=dict(symbol='triangle-down',color='#f87171',size=14,line=dict(color='#7f1d1d',width=1)),
                            name='SELL', text=sells.get('reasoning',''), hoverinfo='text+x+y'))

                fig.update_layout(
                    template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    height=460, xaxis_rangeslider_visible=False,
                    xaxis=dict(type='category' if hide_gaps else 'date', gridcolor='rgba(99,102,241,0.08)', title="Time"),
                    yaxis=dict(gridcolor='rgba(99,102,241,0.08)', title="Price ($)"),
                    margin=dict(l=0,r=0,t=16,b=0),
                    legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1),
                    font=dict(family="Inter",color="#94a3b8")
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.markdown(f"""
                <div class="glass-card" style="text-align:center;padding:70px;">
                    <div style="font-size:2.5rem;">📡</div>
                    <div style="color:#64748b;margin-top:8px;">Waiting for live data for <b>{selected_ticker}</b>...</div>
                </div>""", unsafe_allow_html=True)

            # Execution log
            st.markdown('<div class="section-hdr" style="margin-top:12px;">📜 Execution Log</div>', unsafe_allow_html=True)
            if not trades_df.empty:
                display_cols = [c for c in ['timestamp','side','price','qty','pnl','reasoning'] if c in trades_df.columns]
                st.dataframe(trades_df[display_cols].sort_values('timestamp',ascending=False).head(25), use_container_width=True, height=260)
            else:
                st.markdown(f'<div class="glass-card" style="color:#64748b;text-align:center;">No trades for {selected_ticker} yet.</div>', unsafe_allow_html=True)

        with side_col:
            # Quick ticker switch
            st.markdown('<div class="section-hdr">⚡ Quick Switch</div>', unsafe_allow_html=True)
            quick = ["AAPL","MSFT","NVDA","BTC/USD","ETH/USD","JPM","TSLA","GOOGL"]
            qc = st.columns(4)
            for idx,qt in enumerate(quick):
                qc[idx%4].button(qt, key=f"q_{qt}", use_container_width=True)

            st.divider()

            # Sector donut
            st.markdown('<div class="section-hdr">🥧 Trade Distribution</div>', unsafe_allow_html=True)
            if not full_trades.empty and 'symbol' in full_trades.columns:
                full_trades['sector'] = full_trades['symbol'].apply(get_sector)
                sc_counts = full_trades['sector'].value_counts().reset_index()
                sc_counts.columns = ['Sector','Trades']
                fig_pie = px.pie(sc_counts, values='Trades', names='Sector',
                                 color_discrete_sequence=['#6366f1','#8b5cf6','#a78bfa','#38bdf8','#4ade80'], hole=0.55)
                fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)",margin=dict(l=0,r=0,t=4,b=0),height=210,
                                      showlegend=True,legend=dict(font=dict(color="#94a3b8",size=9)),
                                      font=dict(family="Inter",color="#94a3b8"))
                fig_pie.update_traces(textfont_color="#f1f5f9")
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.markdown('<div class="glass-card" style="color:#64748b;text-align:center;padding:30px;">No trades yet.</div>', unsafe_allow_html=True)

            st.divider()

            # SOUL.md
            st.markdown('<div class="section-hdr">📜 Active Rules (SOUL.md)</div>', unsafe_allow_html=True)
            soul_path = "SOUL.md"
            if os.path.exists(soul_path):
                with open(soul_path,"r") as f: soul_text = f.read()
                st.markdown(f'<div class="journal-box" style="height:180px;">{soul_text}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="glass-card" style="color:#64748b;text-align:center;">No SOUL.md found.</div>', unsafe_allow_html=True)

            st.divider()

            # Macro from sentience
            if sentience.get("macro_summary"):
                st.markdown('<div class="section-hdr">🌐 Macro Outlook</div>', unsafe_allow_html=True)
                st.info(sentience["macro_summary"])

            # Debug
            with st.expander("🛠️ Debug"):
                mkt = load_market_data(selected_ticker)
                if not mkt.empty:
                    st.write(f"Raw Ticks: {len(mkt)}")
                    st.dataframe(mkt.tail(5))
                else:
                    st.write("No market data.")

    # ─── TAB: Swarm Monitor
    with tab_swarm:
        st.markdown('<div class="section-hdr">🤖 Live Agent Pulse</div>', unsafe_allow_html=True)
        combined_pulse = sentience.get("pulse",{})
        if combined_pulse:
            pulse_data = [{"Sector":get_sector(s),"Symbol":s,"Status":d.get("status"),
                           "Detail":d.get("detail"),"Action":d.get("action","—"),
                           "Confidence":d.get("confidence","—"),"Time":d.get("timestamp")}
                          for s,d in combined_pulse.items()]
            st.dataframe(pd.DataFrame(pulse_data).sort_values("Time",ascending=False), use_container_width=True, height=320)
        else:
            st.markdown('<div class="glass-card" style="text-align:center;padding:60px;"><div style="font-size:2.5rem;">🤖</div><div style="color:#64748b;margin-top:8px;">No swarm pulses yet.</div></div>', unsafe_allow_html=True)

        st.divider()
        st.markdown('<div class="section-hdr">📊 P&L by Sector</div>', unsafe_allow_html=True)
        if not full_trades.empty and 'symbol' in full_trades.columns:
            full_trades['sector'] = full_trades['symbol'].apply(get_sector)
            if 'pnl' in full_trades.columns:
                gb = full_trades.groupby('sector')['pnl'].sum().reset_index()
                fig_b = px.bar(gb, x='sector', y='pnl', color='pnl',
                               color_continuous_scale=['#ef4444','#f97316','#4ade80'])
            else:
                gb = full_trades.groupby('sector').size().reset_index(name='count')
                fig_b = px.bar(gb, x='sector', y='count', color_discrete_sequence=['#6366f1'])
            fig_b.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                                 xaxis=dict(gridcolor='rgba(99,102,241,0.08)'),
                                 yaxis=dict(gridcolor='rgba(99,102,241,0.08)'),
                                 margin=dict(l=0,r=0,t=16,b=0),height=280,coloraxis_showscale=False,
                                 font=dict(family="Inter",color="#94a3b8"))
            st.plotly_chart(fig_b, use_container_width=True)

    # ─── TAB: AI Journal
    with tab_journal:
        j1,j2 = st.columns([2,1])
        with j1:
            st.markdown('<div class="section-hdr">🧠 Inner Monologue</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="journal-box">\n\n{load_ai_journal()}\n\n</div>', unsafe_allow_html=True)
        with j2:
            st.markdown('<div class="section-hdr">🌐 Macro Summary</div>', unsafe_allow_html=True)
            macro = sentience.get("macro_summary","No macro analysis available.")
            st.markdown(f'<div class="glass-card"><p style="color:#e2e8f0;font-size:0.85rem;line-height:1.7;">{macro}</p></div>', unsafe_allow_html=True)
            st.markdown('<div class="section-hdr" style="margin-top:8px;">📅 SOUL Update Log</div>', unsafe_allow_html=True)
            soul_log = "logs/soul_update.log"
            if os.path.exists(soul_log):
                with open(soul_log,"r") as f: txt = f.read()
                st.markdown(f'<div class="journal-box" style="height:200px;">\n\n{txt}\n\n</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="glass-card" style="color:#64748b;">No SOUL update logs yet.</div>', unsafe_allow_html=True)




# ══════════════════════════════════════════════
# MODULE: PORTFOLIO
# ══════════════════════════════════════════════
elif active == "portfolio":
    st.markdown('<span style="font-size:2rem;font-weight:700;background:linear-gradient(135deg,#6366f1,#a78bfa);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">💼 Portfolio Overview</span>', unsafe_allow_html=True)
    st.divider()

    if not full_trades.empty:
        p1,p2,p3,p4 = st.columns(4)
        pnl_total = full_trades['pnl'].sum() if 'pnl' in full_trades.columns else 0
        win_rate  = (full_trades['pnl']>0).mean()*100 if 'pnl' in full_trades.columns else 0
        avg_pnl   = full_trades['pnl'].mean() if 'pnl' in full_trades.columns else 0
        with p1: st.metric("Total P&L",f"${pnl_total:,.2f}")
        with p2: st.metric("Total Trades",len(full_trades))
        with p3: st.metric("Win Rate",f"{win_rate:.1f}%")
        with p4: st.metric("Avg P&L/Trade",f"${avg_pnl:,.2f}")

        if 'pnl' in full_trades.columns:
            st.divider()
            ft_s = full_trades.sort_values('timestamp')
            ft_s['cum_pnl'] = ft_s['pnl'].cumsum()
            fig_eq = px.area(ft_s, x='timestamp', y='cum_pnl', title="Cumulative P&L — Equity Curve",
                             color_discrete_sequence=['#6366f1'])
            fig_eq.update_traces(fillcolor='rgba(99,102,241,0.12)',line_color='#6366f1')
            fig_eq.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                                  xaxis=dict(gridcolor='rgba(99,102,241,0.08)'),
                                  yaxis=dict(gridcolor='rgba(99,102,241,0.08)'),
                                  margin=dict(l=0,r=0,t=40,b=0),height=300,
                                  font=dict(family="Inter",color="#94a3b8"))
            st.plotly_chart(fig_eq, use_container_width=True)

        st.markdown('<div class="section-hdr">📋 All Trades</div>', unsafe_allow_html=True)
        st.dataframe(full_trades.sort_values('timestamp',ascending=False), use_container_width=True, height=380)
    else:
        st.markdown("""
        <div class="glass-card" style="text-align:center;padding:80px;">
            <div style="font-size:3rem;">💼</div>
            <div style="color:#64748b;margin-top:10px;">No trade history yet. Portfolio metrics will appear once trading begins.</div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# MODULE: RESEARCH (scaffold)
# ══════════════════════════════════════════════
elif active == "research":
    st.markdown('<span style="font-size:2rem;font-weight:700;background:linear-gradient(135deg,#6366f1,#a78bfa);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">🔬 Research Lab</span>', unsafe_allow_html=True)
    st.divider()

    tab_regime, tab_factor, tab_pairs, tab_backtest = st.tabs([
        "🎭 Regime Detection", "📊 Factor Analysis", "📉 Pairs Trading", "🔁 Backtester"
    ])

    with tab_regime:
        st.markdown('<div class="section-hdr">HMM Market State Visualization</div>', unsafe_allow_html=True)
        symbol_regime = st.selectbox("Market Proxy", ["SPY", "QQQ", "IWM"], index=0, key="regime_sym")
        if st.button("Run Regime Detection", key="btn_regime"):
            with st.spinner("Training Hidden Markov Model on recent vol and returns..."):
                rdf = run_regime_detection(symbol_regime, days=180)
                if not rdf.empty:
                    fig = px.scatter(rdf, x="timestamp", y="close", color="regime", 
                                     title=f"{symbol_regime} Price by Detected Regime (Last 6 Months)",
                                     color_discrete_map={"LOW_VOL_BULL": "#4ade80", "HIGH_VOL_BEAR": "#ef4444", "SIDEWAYS_CHOP": "#f97316", "UNKNOWN": "#94a3b8"})
                    fig.update_traces(mode='markers')
                    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8"))
                    st.plotly_chart(fig, use_container_width=True)
                    
                    curr_reg = rdf.iloc[-1]['regime']
                    st.metric("Current Regime", curr_reg)
                else:
                    st.warning("Not enough data to run regime detection. Ensure DB is populated.")

    with tab_factor:
        st.markdown('<div class="section-hdr">Quantitative Factor Scoring</div>', unsafe_allow_html=True)
        if st.button("Run Full Universe Scan", type="primary", key="btn_factor"):
            with st.spinner("Scoring Universe for Momentum, Volatility, and Trend..."):
                symbols = get_all_symbols()
                fdf = run_factor_ranking(symbols, days=90)
                if not fdf.empty:
                    st.dataframe(fdf.head(20), use_container_width=True, height=400)
                else:
                    st.warning("Insufficient data across universe.")
                    
    with tab_pairs:
        st.markdown('<div class="section-hdr">Cointegration Scanner</div>', unsafe_allow_html=True)
        st.info("Scanning for statistically cointegrated pairs suitable for mean-reversion.")
        if st.button("Run Cointegration Scan", key="btn_pairs"):
            with st.spinner("Computing correlation and z-scores for all pairs..."):
                symbols = get_all_symbols()
                if len(symbols) > 25: 
                    symbols = symbols[:25]
                pdf = run_cointegration_scan(symbols, days=180)
                if not pdf.empty:
                    st.dataframe(pdf.head(15), use_container_width=True)
                else:
                    st.warning("No cointegrated pairs found or insufficient data.")

    with tab_backtest:
        st.markdown('<div class="section-hdr">Historical Strategy Replay</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1: b_strat = st.selectbox("Strategy Baseline", ["MACD Crossover", "Mean Reversion (Bollinger)"])
        with c2: b_sym = st.selectbox("Target Symbol", get_all_symbols(), key="bt_sym")
        with c3: b_days = st.slider("Lookback Days", 30, 365, 180)
        
        if st.button("Run Strategy Simulation", type="primary", key="btn_bt"):
            with st.spinner(f"Simulating {b_strat} on {b_sym} over {b_days} days..."):
                res = run_backtest(b_strat, b_sym, b_days)
                if "error" in res:
                    st.error(res["error"])
                else:
                    b1, b2, b3, b4 = st.columns(4)
                    with b1: st.metric("Strategy Return", f"{res['total_return']}%")
                    with b2: st.metric("Market Return", f"{res['market_return']}%")
                    with b3: st.metric("Win Rate", f"{res['win_rate']}%")
                    with b4: st.metric("Trades Taken", res['trades_taken'])
                    
                    df_out = res["df"]
                    fig = px.line(df_out, x="timestamp", y=["cumulative_market", "cumulative_strategy"],
                                  title="Equity Curve vs Benchmark ($10k Initial)",
                                  color_discrete_map={"cumulative_market": "#64748b", "cumulative_strategy": "#6366f1"})
                    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8"),
                                      legend_title="Portfolio")
                    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════
# MODULE: SENTIENCE CHAT
# ══════════════════════════════════════════════
elif active == "sentience":
    st.markdown('<span style="font-size:2rem;font-weight:700;background:linear-gradient(135deg,#6366f1,#a78bfa);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">🧠 Sentience Core Chat</span>', unsafe_allow_html=True)
    st.divider()

    # Chat UI
    session_id = "default_user"
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = st.session_state.memory.get_history(session_id)

    # Display history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat Input
    if prompt := st.chat_input("How can I help with your portfolio today?"):
        with st.chat_message("user"):
            st.markdown(prompt)
            st.session_state.memory.add_message(session_id, "user", prompt)
            st.session_state.chat_history.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("Sentience Core is thinking..."):
                response = st.session_state.agent.chat(prompt)
                st.markdown(response)
                st.session_state.memory.add_message(session_id, "assistant", response)
                st.session_state.chat_history.append({"role": "assistant", "content": response})

    # Sidebar Tools Display
    with st.sidebar:
        st.markdown('<div class="section-hdr">🛠️ Available Tools</div>', unsafe_allow_html=True)
        for tool in st.session_state.agent.tool_registry.get_tool_schemas():
            with st.expander(f"🔧 {tool['name']}"):
                st.caption(tool['description'])
                st.json(tool['parameters'])

elif active == "real_estate":
    st.markdown('<span style="font-size:2rem;font-weight:700;background:linear-gradient(135deg,#6366f1,#a78bfa);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">🏠 Real Estate Intelligence</span><span class="coming-tag">COMING SOON</span>', unsafe_allow_html=True)
    st.divider()
    r1,r2 = st.columns(2)
    cards = [
        ("🏗️ Property Valuation","Cap Rate, DCF, and Gross Rent Multiplier calculators. Compare properties by market and asset class."),
        ("📊 REIT Analytics","REIT yield comparison vs direct ownership vs equity portfolio via integrated allocation optimizer."),
        ("🗺️ Market Heatmaps","Zip-code level price appreciation, rental yield, and population growth overlays."),
        ("💰 Debt vs Equity","Side-by-side ROI comparison between mortgaged properties and stock portfolio equivalents."),
    ]
    for i,(title,desc) in enumerate(cards):
        with (r1 if i%2==0 else r2):
            st.markdown(f'<div class="glass-card"><div style="color:#a78bfa;font-weight:600;margin-bottom:6px;">{title}</div><div style="color:#64748b;font-size:0.83rem;line-height:1.6;">{desc}</div></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════
# MODULE: SETTINGS & TRAINING
# ══════════════════════════════════════════════
elif active == "settings":
    st.markdown('<span style="font-size:2rem;font-weight:700;background:linear-gradient(135deg,#6366f1,#a78bfa);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">⚙️ System Settings & Training</span>', unsafe_allow_html=True)
    st.divider()

    st.markdown('<div class="section-hdr">⚡ Sentience Core (QLoRA) Status</div>', unsafe_allow_html=True)
    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        if sentience_info.get("loaded"):
            st.success(f"✅ Sentience Lora Adapter\n{sentience_info['size_mb']} MB · trained {sentience_info['trained_at']}")
        else:
            st.warning("⚠️ No LoRA adapter found. Running on base Phi-3.")
    with sc2:
        st.markdown("**Core Learning**")
        st.code("python training/train_qlora.py", language="bash")
    with sc3:
        st.markdown("**Hot Reload**")
        if st.session_state.get("agent"):
            if st.button("🔄 Reload Model"):
                try:
                    import requests
                    requests.post("http://localhost:8000/v1/model/reload")
                    st.success("Reload trigger sent.")
                except: st.error("Model server unreachable.")

    st.divider()
    st.markdown('<div class="section-hdr">⚡ V3 Model (PPO) Training Status</div>', unsafe_allow_html=True)
    t1,t2,t3 = st.columns(3)
    with t1:
        if v3_info.get("loaded"):
            st.success(f"✅ ppo_v3_cyborg.zip\n{v3_info['size_mb']} MB · trained {v3_info['trained_at']}")
        else:
            st.error("❌ V3 model not found.")
    with t2:
        st.markdown("**Next Training Run**")
        st.code("python train_v3.py --lr 0.0003 --time 90 --seeds 3", language="bash")
    with t3:
        st.markdown("**Architecture**")
        st.markdown("""
        - Network: `[256, 256, 128]`
        - Observation: 23-dim
        - Algo: PPO + VecNormalize
        - Curriculum: 3-phase
        - Population: best-of-3 seeds
        """)

    st.divider()
    ckpt_dir = "ml/checkpoints"
    if os.path.exists(ckpt_dir):
        ckpts = sorted([f for f in os.listdir(ckpt_dir) if f.endswith('.zip')], reverse=True)[:10]
        if ckpts:
            st.markdown('<div class="section-hdr">🗂️ Recent Checkpoints</div>', unsafe_allow_html=True)
            data = [{"File":f,"Size MB":round(os.path.getsize(os.path.join(ckpt_dir,f))/(1024*1024),1),
                     "Modified":datetime.fromtimestamp(os.path.getmtime(os.path.join(ckpt_dir,f))).strftime("%m/%d %H:%M")} for f in ckpts]
            st.dataframe(pd.DataFrame(data), use_container_width=True)

    st.divider()
    s1,s2 = st.columns(2)
    with s1:
        st.markdown('<div class="section-hdr">📋 config.yaml</div>', unsafe_allow_html=True)
        if config: st.json(config)
        else: st.warning("No config.yaml found.")

    with s2:
        st.markdown('<div class="section-hdr">🔑 Environment Variables</div>', unsafe_allow_html=True)
        env_keys = ["APCA_API_KEY_ID","APCA_API_SECRET_KEY","TELEGRAM_BOT_TOKEN","BRAVE_API_KEY"]
        for key in env_keys:
            val = os.environ.get(key)
            if val: st.metric(key, "✅ SET", delta=f"{val[:6]}***")
            else:   st.metric(key, "❌ MISSING", delta_color="inverse", delta="Not configured")

# ─────────────────────────────────────────────
# AUTO-REFRESH
# ─────────────────────────────────────────────
if auto_refresh:
    time.sleep(5)
    st.rerun()
