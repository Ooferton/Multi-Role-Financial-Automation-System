import time
import os
import logging
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

from agents.trading_agent import TradingAgent
from agents.alpaca_broker import AlpacaBroker
from strategies.rl_strategy_v2 import RLStrategyV2
from core.orchestrator import Orchestrator
from data.feature_store import FeatureStore, MarketTick

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

def main():
    load_dotenv()
    
    # 1. Initialize Components
    logger.info("Initializing Live Trading System...")
    
    # Broker
    broker = AlpacaBroker(paper=True)
    
    # Check connection
    summary = broker.get_account_summary()
    logger.info(f"Connected to Alpaca. Equity: ${summary.get('equity', 0)}")
    
    # Feature Store (for logging history if needed)
    feature_store = FeatureStore()
    
    # Orchestrator with broker
    orchestrator = Orchestrator("config/config.yaml")
    orchestrator.set_broker(broker)
    
    # Agent
    agent = TradingAgent("Live_RL_Agent_v2", {}, feature_store, broker)
    
    # Strategy — PPO Trading Real v2 (10 indicators)
    model_path = "ml/models/ppo_trading_real_v2"
    
    # Wait for model if not ready
    while not os.path.exists(model_path + ".zip") and not os.path.exists(model_path):
        logger.info("Waiting for model file...")
        time.sleep(10)

    strategy = RLStrategyV2("RL_Brain_v2", {}, broker, model_path.replace(".zip",""))
    agent.add_strategy(strategy)
    
    target_symbol = "SPY"
    logger.info(f"Starting High-Frequency Loop for {target_symbol}...")
    
    try:
        while True:
            # 1. Fetch Real-time Price
            price = broker.get_current_price(target_symbol)
            
            if price > 0:
                # 2. Create Tick
                tick = MarketTick(
                    symbol=target_symbol,
                    timestamp=datetime.now(),
                    price=price,
                    size=100, # Placeholder volume
                    exchange="ALPACA"
                )
                
                # 3. Update Agent (Predict & Execute)
                # This calls strategy.on_tick -> process_signal -> execute_instruction -> broker.submit
                agent.update_market_state({'tick': tick})
                
            else:
                logger.warning("Failed to fetch price.")

            # 4. Wait (1 second frequency)
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Live Trading Stopped by User.")
    except Exception as e:
        logger.exception(f"Crash: {e}")

if __name__ == "__main__":
    main()
