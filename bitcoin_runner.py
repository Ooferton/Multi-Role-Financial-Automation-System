import time
import os
import logging
import json
import yaml
from datetime import datetime, timedelta
import pytz
import argparse
import subprocess
import sys
from dotenv import load_dotenv

from agents.trading_agent import TradingAgent
from agents.alpaca_broker import AlpacaBroker
from strategies.rl_strategy_v2 import RLStrategyV2
from strategies.hft import HFTStrategy
from core.orchestrator import Orchestrator
from data.feature_store import FeatureStore, MarketTick

# Setup Logging for Bitcoin Runner
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/bitcoin_trading.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("BitRunner")

def run_warmup(broker, agent, feature_store, symbols):
    """Fetches historical data to prime the agent and dashboard."""
    logger.info("Starting Historical Warmup for Bitcoin Assets...")
    start = datetime.now() - timedelta(days=3)
    end = datetime.now()
    
    total_bars = 0
    for symbol in symbols:
        # Emergency Circuit Breaker Check (Inside Warmup)
        if os.path.exists("data/circuit_breaker.lock"):
            logger.critical(f"[CIRCUIT BREAKER] HALTING BIT-WARMUP for {symbol}!")
            while os.path.exists("data/circuit_breaker.lock"):
                time.sleep(2)
            logger.info("[CIRCUIT BREAKER] Reset detected. Resuming bit-warmup...")

        try:
            bars = broker.get_historical_data(symbol, start=start, end=end, timeframe='1Min', limit=100)
            if bars:
                for bar in bars:
                    for i, attr in enumerate(['open', 'high', 'low', 'close']):
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
            
    logger.info(f"Warmup complete. Ingested {total_bars} bars.")

def start_dashboard():
    """Launches the Streamlit dashboard in the background."""
    logger.info("Launching Sentience Dashboard (http://localhost:8501)...")
    try:
        os.makedirs("logs", exist_ok=True)
        log_f = open("logs/dashboard.log", "w")
        subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", "dashboard.py", "--server.port", "8501", "--server.headless", "true"],
            stdout=log_f,
            stderr=log_f,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0,
            start_new_session=True if os.name != 'nt' else False
        )
        logger.info("✅ Dashboard process started in background.")
        logger.info("💡 If the browser doesn't open, visit: http://localhost:8501")
    except Exception as e:
        logger.error(f"❌ Failed to launch dashboard: {e}")

def main():
    parser = argparse.ArgumentParser(description="High-Speed Bitcoin/ETF Trading Runner")
    parser.add_argument("--one-shot", action="store_true", help="Run once and exit")
    parser.add_argument("--skip-dashboard", action="store_true", help="Do not auto-launch the dashboard (for HF Spaces)")
    args = parser.parse_args()

    load_dotenv()
    
    # 0. Process ID Tracking
    os.makedirs("data", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # 0.1 Auto-Reset Circuit Breaker on Startup
    if os.path.exists("data/circuit_breaker.lock"):
        try:
            os.remove("data/circuit_breaker.lock")
            logger.info("♻️ System Startup: Circuit Breaker auto-reset.")
        except: pass

    with open("data/bitcoin_runner.pid", "w") as f:
        f.write(str(os.getpid()))
    logger.info(f"🚀 BitRunner Started | PID: {os.getpid()}")
    
    # Dashboard Startup
    if not args.skip_dashboard:
        start_dashboard()

    # 1. Initialize Components
    logger.info("Initializing High-Speed Components...")
    target_symbols = ["BTC/USD", "ETH/USD", "DOGE/USD", "IBIT", "BITO", "FBTC", "ARKB", "HODL", "COIN", "MSTR", "MARA", "RIOT"]
    broker = AlpacaBroker(paper=True, authorized_tickers=target_symbols)
    feature_store = FeatureStore()
    
    orchestrator = Orchestrator("config/config.yaml")
    agent = TradingAgent("BitAgent", orchestrator.config, feature_store, broker)
    orchestrator.register_agent(agent)
    
    # RL Strategy v2 (Strict Bitcoin Mode)
    model_path = "ml/models/ppo_trading_real_v2"
    # Force strategy into Bitcoin mode BEFORE initialization to set correct status_path
    orchestrator.config.setdefault('system', {})['crypto_etf_only'] = True
    strategy = RLStrategyV2("RL_BitBrain", orchestrator.config, broker, model_path)
    agent.add_strategy(strategy)
    
    # Broadcast INITIAL state immediately
    try:
        with open("data/sentience_bitcoin.json", "w") as f:
            json.dump({
                "pid": os.getpid(),
                "mode": "HIGH_SPEED_BITCOIN",
                "active": True,
                "timestamp": datetime.now().isoformat()
            }, f)
    except: pass

    # 2. Warmup
    run_warmup(broker, agent, feature_store, target_symbols)
    
    logger.info(f"Main loop starting for: {target_symbols}")
    
    try:
        tick_count = 0
        while True:
            # Broadcast state to specialized status file
            try:
                with open("data/sentience_bitcoin.json", "w") as f:
                    json.dump({
                        "timestamp": datetime.now().isoformat(),
                        "mode": "HIGH_SPEED_BITCOIN",
                        "pid": os.getpid(),
                        "active": True,
                        "symbols": target_symbols,
                        "pulse": strategy.live_activity
                    }, f)
            except: pass

            # 2. Emergency Circuit Breaker Check
            if os.path.exists("data/circuit_breaker.lock"):
                logger.critical("[CIRCUIT BREAKER] TRIGGERED! Halting BitRunner for safety.")
                while os.path.exists("data/circuit_breaker.lock"): 
                    time.sleep(2)
                logger.info("[CIRCUIT BREAKER] Reset detected. Resuming crypto operation...")

            # 3. Fetch Latest Prices (High Speed)
            prices = broker.get_latest_prices(target_symbols)
            
            for symbol in target_symbols:
                price = prices.get(symbol, 0.0)
                if price > 0:
                    tick = MarketTick(
                        symbol=symbol,
                        timestamp=datetime.now(pytz.UTC),
                        price=price,
                        size=100,
                        exchange="BITRUNNER"
                    )
                    agent.update_market_state({'tick': tick})
                    feature_store.save_tick(tick)

            if args.one_shot: break
            
            # High speed: 0.2s polling
            time.sleep(0.2)
            
            tick_count += 1
            if tick_count % 50 == 0:
                logger.info(f"BitRunner Pulse | Scanned {len(target_symbols)} high-speed assets.")

    except KeyboardInterrupt:
        logger.info("BitRunner Stopped by User.")
    except Exception as e:
        logger.exception(f"BitRunner Crash: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
