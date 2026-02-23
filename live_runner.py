import time
import os
import logging
import subprocess
import sys
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

from agents.trading_agent import TradingAgent
from agents.alpaca_broker import AlpacaBroker
from strategies.rl_strategy_v2 import RLStrategyV2
from strategies.hft import HFTStrategy
from core.orchestrator import Orchestrator
from data.feature_store import FeatureStore, MarketTick
from agents.mock_broker import MockBroker
import pytz

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("live_trading.log"),
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
    logger.info("Launching Sentience Dashboard...")
    try:
        # Using sys.executable to ensure we use the same environment
        subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", "dashboard.py"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
        )
        logger.info("Dashboard active in background.")
    except Exception as e:
        logger.error(f"Failed to launch dashboard: {e}")

def main():
    load_dotenv()
    
    # 1. Initialize Components
    logger.info("Initializing Live Trading System...")
    
    # Auto-launch Dashboard
    start_dashboard()
    
    # Orchestrator with config
    orchestrator = Orchestrator("config/config.yaml")

    # 1. Initialize Broker based on Config
    broker_name = orchestrator.config.get('brokerage', {}).get('name', 'MOCK')
    
    if broker_name == "ALPACA" and os.getenv("APCA_API_KEY_ID"):
        logger.info("Initializing Alpaca Broker (Paper Trading)...")
        broker = AlpacaBroker(paper=True)
    else:
        logger.info("Initializing Mock Broker for simulation...")
        broker = MockBroker(initial_cash=100000)
    
    # Connect Orchestrator
    orchestrator.set_broker(broker)
    
    # Feature Store (for logging history if needed)
    feature_store = FeatureStore(db_path="data/feature_store.db")
    
    # Agent
    agent = TradingAgent("Live_RL_Agent_v2", {}, feature_store, broker)
    
    # Strategy — PPO Trading Real v2 (10 indicators)
    model_path = "ml/models/ppo_trading_real_v2"
    
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
        "KO", "PEP", "MCD", "NKE", "SBUX", "HD", "LOW", "JNJ", "PFE", "UNH", "DIS"
    ]
    logger.info(f"Starting Multi-Ticker Loop for {target_symbols}...")
    
    # 2. Performance Warmup (Historical Data)
    run_warmup(broker, agent, feature_store, target_symbols)
    
    try:
        while True:
            # Emergency Circuit Breaker Check
            if os.path.exists("data/circuit_breaker.lock"):
                logger.critical("[CIRCUIT BREAKER] TRIGGERED! Halting system for safety.")
                # Wait until reset on dashboard
                while os.path.exists("data/circuit_breaker.lock"):
                    time.sleep(2)
                logger.info("[CIRCUIT BREAKER] Reset detected. Resuming operation...")

            # 1. Fetch Latest Prices for ALL symbols in ONE call (Prevents Rate Limiting)
            prices = broker.get_latest_prices(target_symbols)
            
            for symbol in target_symbols:
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

            # 4. Wait (1 second frequency)
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Live Trading Stopped by User.")
    except Exception as e:
        logger.exception(f"Crash: {e}")

if __name__ == "__main__":
    main()
