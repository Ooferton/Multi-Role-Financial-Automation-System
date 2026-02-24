import time
import os
import logging
import json
import subprocess
import sys
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

from agents.trading_agent import TradingAgent
from agents.alpaca_broker import AlpacaBroker
from agents.coinbase_broker import CoinbaseBroker
from strategies.rl_strategy_v2 import RLStrategyV2
from strategies.hft import HFTStrategy
from core.orchestrator import Orchestrator
from data.feature_store import FeatureStore, MarketTick
from agents.mock_broker import MockBroker
import pytz
import argparse

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("live_trading.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("LiveRunner")

def run_warmup(broker, agent, feature_store, symbols):
    """Fetches historical data to prime the agent and dashboard."""
    logger.info("Starting Historical Warmup (Last 100 bars within 72-hour window)...")
    
    # Use a 3-day lookback to ensure we catch the last trading day even on weekends
    start = datetime.now() - timedelta(days=3)
    end = datetime.now()
    
    total_bars = 0
    for symbol in symbols:
        # Emergency Circuit Breaker Check (Inside Warmup)
        if os.path.exists("data/circuit_breaker.lock"):
            logger.critical(f"[CIRCUIT BREAKER] HALTING WARMUP for {symbol}!")
            while os.path.exists("data/circuit_breaker.lock"):
                time.sleep(2)
            logger.info("[CIRCUIT BREAKER] Reset detected. Resuming warmup...")

        try:
            # Fetch last 100 bars from the window
            bars = broker.get_historical_data(symbol, start=start, end=end, timeframe='1Min', limit=100)
            if bars:
                for bar in bars:
                    # Save OHLC as separate ticks so dashboard resample can reconstruct candles
                    # Use 1-second offsets to ensure uniqueness in SQLite PK (symbol, timestamp)
                    for i, attr in enumerate(['open', 'high', 'low', 'close']):
                        # Ensure bar.timestamp is UTC
                        base_ts = bar.timestamp
                        if base_ts.tzinfo is None:
                            base_ts = pytz.UTC.localize(base_ts)
                            
                        tick = MarketTick(
                            symbol=symbol,
                            timestamp=base_ts + timedelta(seconds=i),
                            price=getattr(bar, attr),
                            size=bar.volume / 4.0,
                            exchange="WARMUP"
                        )
                        if attr == 'close':
                            agent.update_market_state({'tick': tick})
                        feature_store.save_tick(tick)
                total_bars += len(bars)
        except Exception as e:
            logger.error(f"Warmup failed for {symbol}: {e}")
            
    logger.info(f"Warmup complete. Ingested {total_bars} historical data points across {len(symbols)} symbols.")

def start_dashboard():
    """Launches the Streamlit dashboard in the background."""
    logger.info("Launching Sentience Dashboard (http://localhost:8501)...")
    try:
        # Create logs directory if missing
        os.makedirs("logs", exist_ok=True)
        
        # Open a log file for the dashboard
        log_f = open("logs/dashboard.log", "w")
        
        # Using sys.executable to ensure we use the same environment
        # Removed DEVNULL redirect to allow errors to go to dashboard.log
        subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", "dashboard.py", "--server.port", "8501", "--server.headless", "true"],
            stdout=log_f,
            stderr=log_f,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0,
            start_new_session=True if os.name != 'nt' else False
        )
        logger.info("✅ Dashboard process started in background.")
        logger.info("💡 If the browser doesn't open, visit: http://localhost:8501")
        logger.info("⚠️ To run manually if background fails: 'python -m streamlit run dashboard.py'")
    except Exception as e:
        logger.error(f"❌ Failed to launch dashboard: {e}")

def main():
    parser = argparse.ArgumentParser(description="Multi-Role Financial Trading Runner")
    parser.add_argument("--one-shot", action="store_true", help="Run once and exit (for GitHub Actions)")
    parser.add_argument("--crypto-etf-only", action="store_true", help="Only trade BTC and Bitcoin/Market ETFs (Bypasses PDT)")
    parser.add_argument("--skip-dashboard", action="store_true", help="Do not auto-launch the dashboard (for HF Spaces)")
    args = parser.parse_args()

    load_dotenv()
    
    # 0. Process ID Tracking
    os.makedirs("data", exist_ok=True)
    
    # 0.1 Auto-Reset Circuit Breaker on Startup
    if os.path.exists("data/circuit_breaker.lock"):
        try:
            os.remove("data/circuit_breaker.lock")
            logger.info("♻️ System Startup: Circuit Breaker auto-reset.")
        except: pass

    with open("data/live_runner.pid", "w") as f:
        f.write(str(os.getpid()))
    logger.info(f"Process ID: {os.getpid()}")

    # 1. Initialize Components
    logger.info("Initializing Live Trading System...")
    
    # Auto-launch Dashboard (only if not in one-shot mode and not skipped)
    if not args.one_shot and not args.skip_dashboard:
        start_dashboard()
    
    # Orchestrator with config
    orchestrator = Orchestrator("config/config.yaml")

    # 1. Initialize Broker based on Config
    broker_name = orchestrator.config.get('brokerage', {}).get('name', 'MOCK')
    
    if broker_name == "ALPACA" and os.getenv("APCA_API_KEY_ID"):
        logger.info("Initializing Alpaca Broker (Paper Trading)...")
        broker = AlpacaBroker(paper=True)
    elif broker_name == "COINBASE":
        logger.info("Initializing Coinbase Broker (Crypto)...")
        broker = CoinbaseBroker()
    else:
        logger.info("Initializing Mock Broker for simulation...")
        broker = MockBroker(initial_cash=100000)
    
    # Connect Orchestrator
    orchestrator.set_broker(broker)
    
    # Initial PDT Check
    try:
        if hasattr(broker, 'get_account_summary'):
            acc_info = broker.get_account_summary()
            if acc_info:
                logger.info(f"Initial Account Load | Equity: ${acc_info.get('equity')} | PDT Status: {acc_info.get('pattern_day_trader')} | Day Trades: {acc_info.get('day_trade_count')}")
                if acc_info.get('equity', 0) < 25000 and acc_info.get('day_trade_count', 0) >= 3:
                    logger.warning("PDT LIMIT APPROACHED! System will throttle high-frequency trades to avoid lock-out.")
    except Exception as e:
        logger.error(f"Failed to perform initial PDT check: {e}")
    
    # Feature Store (for logging history if needed)
    feature_store = FeatureStore(db_path="data/feature_store.db")
    
    # Agent
    agent = TradingAgent("Live_RL_Agent_v2", {}, feature_store, broker)
    
    # Strategy — PPO Trading Real v2 (10 indicators)
    model_path = "ml/models/ppo_trading_real_v2"
    
    # Broadcast INITIAL state immediately
    try:
        status_path = "data/sentience_status.json"
        with open(status_path, "w") as f:
            json.dump({
                "pid": os.getpid(),
                "mode": "FULL_PORTFOLIO",
                "active": True,
                "timestamp": datetime.now().isoformat()
            }, f)
    except: pass

    # Wait for model if not ready
    while not os.path.exists(model_path + ".zip") and not os.path.exists(model_path):
        logger.info("Waiting for model file...")
        time.sleep(10)

    strategy = RLStrategyV2("RL_Brain_v2", orchestrator.config, broker, model_path.replace(".zip",""))
    agent.add_strategy(strategy)
    
    # HFT Strategy
    hft_strategy = HFTStrategy("HFT_Momentum", {"window_size": 20}, broker)
    agent.add_strategy(hft_strategy)
    
    target_symbols = [
        # Indices & Broad Market
        "SPY", "QQQ", "DIA", "IWM", 
        "SQQQ", "UVXY", # Crisis Alpha Hedge Tickers
        
        # Sector ETFs
        "XLF", "XLK", "XLE", "XLV", "XLI", "XLB", "XLY", "XLP", "XLU", "XLC", "XLRE",
        
        # Treasury Bonds (ETFs)
        "TLT", "IEF", "IEI", "SHY", "BND", "AGG", "TIP",

        # Tech Titans & Growth
        "AAPL", "NVDA", "TSLA", "MSFT", "AMZN", "META", "GOOGL", "GOOG", "AVGO", "COST", "ORCL", "CRM", "ADBE", "PLTR", "RTX", "TSM", "MU", "CSCO", "SNPS",
        
        # Finance & Value
        "BAC", "JPM", "WFC", "C", "BLK", "GS", "MS", "AXP", "MA", "V", 
        
        # Industrials & Energy
        "CAT", "GE", "BA", "LMT", "NOC", "XOM", "CVX", "HON", "GIS", 
        
        # Consumer & Healthcare
        "KO", "PEP", "MCD", "NKE", "SBUX", "HD", "LOW", "JNJ", "PFE", "UNH", "DIS",
    ]

    # 1. Determine Asset Universe
    target_symbols = [
        # Indices & Broad Market
        "SPY", "QQQ", "DIA", "IWM", 
        "SQQQ", "UVXY", # Crisis Alpha Hedge Tickers
        
        # Sector ETFs
        "XLF", "XLK", "XLE", "XLV", "XLI", "XLB", "XLY", "XLP", "XLU", "XLC", "XLRE",
        
        # Treasury Bonds (ETFs)
        "TLT", "IEF", "IEI", "SHY", "BND", "AGG", "TIP",

        # Tech Titans & Growth
        "AAPL", "NVDA", "TSLA", "MSFT", "AMZN", "META", "GOOGL", "GOOG", "AVGO", "COST", "ORCL", "CRM", "ADBE", "PLTR", "RTX", "TSM", "MU", "CSCO", "SNPS",
        
        # Finance & Value
        "BAC", "JPM", "WFC", "C", "BLK", "GS", "MS", "AXP", "MA", "V", 
        
        # Industrials & Energy
        "CAT", "GE", "BA", "LMT", "NOC", "XOM", "CVX", "HON", "GIS", 
        
        # Consumer & Healthcare
        "KO", "PEP", "MCD", "NKE", "SBUX", "HD", "LOW", "JNJ", "PFE", "UNH", "DIS"
    ]

    # 2. Performance Warmup (Historical Data)
    logger.info("Initializing Standard Portfolio Runner...")
    run_warmup(broker, agent, feature_store, target_symbols)
    
    try:
        while True:
            # Broadcast state to dashboard status JSON
            try:
                status_path = "data/sentience_status.json"
                if os.path.exists(status_path):
                    with open(status_path, "r") as f:
                        curr = json.load(f)
                else: curr = {}
                curr["pid"] = os.getpid()
                curr["mode"] = "FULL_PORTFOLIO"
                curr["active"] = True
                with open(status_path, "w") as f:
                    json.dump(curr, f)
            except: pass

            # 2. Emergency Circuit Breaker Check (Initial)
            if os.path.exists("data/circuit_breaker.lock"):
                logger.critical("[CIRCUIT BREAKER] TRIGGERED! Halting system for safety.")
                while os.path.exists("data/circuit_breaker.lock"): time.sleep(2)
                logger.info("[CIRCUIT BREAKER] Reset detected. Resuming operation...")

            # 3. Fetch Latest Prices
            prices = broker.get_latest_prices(target_symbols)
            
            for symbol in target_symbols:
                # Secondary Emergency Check (During long loops)
                if os.path.exists("data/circuit_breaker.lock"):
                    break
                
                price = prices.get(symbol, 0.0)
                
                if price > 0:
                    # 2. Create Tick
                    tick = MarketTick(
                        symbol=symbol,
                        timestamp=datetime.now(),
                        price=price,
                        size=100, # Placeholder volume
                        exchange=broker_name
                    )
                    
                    # 3. Update Agent (Predict & Execute)
                    # This calls strategy.on_tick -> process_signal -> execute_instruction -> broker.submit
                    agent.update_market_state({'tick': tick})
                    
                    # 3.5 Persist Tick for Dashboard Visualization
                    feature_store.save_tick(tick)
                    
                else:
                    logger.warning(f"Failed to fetch price for {symbol}.")

            # 4. Exit if in one-shot mode
            if args.one_shot:
                logger.info("One-shot execution complete. Exiting.")
                break

            # 5. Wait (High frequency for Crypto/ETF, 1s for standard)
            sleep_time = 0.1 if args.crypto_etf_only else 1
            time.sleep(sleep_time)
            
    except KeyboardInterrupt:
        logger.info("Live Trading Stopped by User.")
    except Exception as e:
        logger.exception(f"Crash: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
