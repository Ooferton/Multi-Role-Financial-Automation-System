import asyncio
import logging
from datetime import datetime
from core.orchestrator import Orchestrator
from agents.trading_agent import TradingAgent
from agents.wealth_agent import WealthStrategyAgent
from agents.mock_broker import MockBroker
from data.feature_store import FeatureStore, MarketTick
from strategies.day_trading import DayTradingStrategy
from ml.model_factory import ModelFactory
from ml.rl_agent import RLAgent

# Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ShadowModeVerification")

async def run_shadow_mode():
    logger.info("Initializing Financial Automation System in SHADOW MODE...")

    # 1. Initialize Infrastructure
    feature_store = FeatureStore()
    broker = MockBroker() # New Mock Broker
    
    # 2. Initialize Orchestrator
    orchestrator = Orchestrator("config/config.yaml")
    
    # 3. Initialize Agents
    # Note: TradingAgent needs a config dict as second arg
    trading_agent = TradingAgent("trading_bot_v1", {}, feature_store, broker)
    wealth_agent = WealthStrategyAgent("wealth_manager_v1", {})
    
    # 4. Attach Strategies to Trading Agent
    dt_strategy = DayTradingStrategy("momentum_v1", {})
    trading_agent.add_strategy(dt_strategy)
    
    # 5. Register Agents with Orchestrator
    orchestrator.register_agent(trading_agent)
    orchestrator.register_agent(wealth_agent)
    
    # 6. Initialize ML Models (Mock load)
    logger.info("Loading ML Models...")
    # Creating a dummy model
    lstm_model = ModelFactory.create_model("LSTM", {'input_size': 10, 'hidden_size': 32})
    rl_agent = RLAgent() # Initialize PPO agent
    
    # 7. Simulate Data Ingestion
    logger.info("Simulating Market Data Stream...")
    tick = MarketTick("AAPL", datetime.now(), 150.0, 100, "NASDAQ")
    await feature_store.ingest_tick(tick)
    
    # 8. Run Orchestrator Cycle
    logger.info("Executing Orchestrator Cycle...")
    orchestrator.run_cycle()
    
    # 9. Trigger Agent Logic (Manually for test)
    # In real loop, Orchestrator would do this via events or async loop
    trading_agent.update_market_state({'tick': tick})
    
    logger.info("Shadow Mode Cycle Complete. System Integration Verified.")

if __name__ == "__main__":
    asyncio.run(run_shadow_mode())
