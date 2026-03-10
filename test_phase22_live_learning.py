import os
import json
import logging
from unittest.mock import MagicMock
from datetime import datetime
import numpy as np

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger('TestPhase22')

# Mock configuration
config = {
    'system': {'crypto_etf_only': False},
    'risk': {'max_single_position_pct': 0.15}
}

try:
    from ml.rl_agent_v2 import RLAgentV2
    from strategies.rl_strategy_v2 import RLStrategyV2
    from data.feature_store import MarketTick
    
    # 1. Initialize Mock Broker and Agent
    logger.info("Initializing Agent and Strategy...")
    broker = MagicMock()
    broker.get_account_summary.return_value = {'equity': 10000, 'buying_power': 20000, 'cash': 10000}
    broker.get_positions.return_value = []
    
    # For this test, we don't strictly need a pre-trained model. 
    # RLAgentV2 will initialize a fresh default PPO model if none is found.
    agent = RLStrategyV2("RL_TEST", config, broker, model_path="ml/models/ppo_v3_cyborg")
    
    # Fast-forward the batch size to 3 for testing (default is 50)
    agent.train_batch_size = 3
    
    # 2. Simulate 3 closed trades
    logger.info("Simulating live trades...")
    symbols = ["SPY", "QQQ", "TSLA"]
    for i, sym in enumerate(symbols):
        # We manually inject a pending experience to simulate the on_tick cycle
        dummy_obs = np.random.rand(14).astype(np.float32)
        agent.pending_experiences[sym] = {'obs': dummy_obs, 'action': 0.5}
        
        # Simulate closing the position (win for even, loss for odd to test both)
        entry_price = 100.0
        agent.entry_prices[sym] = entry_price
        agent.entry_sides[sym] = 'LONG'
        
        exit_price = 105.0 if i % 2 == 0 else 95.0 # Win, Loss, Win
        logger.info(f"Closing trade {i+1} for {sym} at ${exit_price}")
        
        result = agent._close_position(sym, exit_price, 10, "TEST EXIT")
        logger.info(f"Close Result: {result['action']} {sym}")
        
    logger.info("Test completed successfully.")
    
except Exception as e:
    logger.error(f"Test failed with exception: {e}")
