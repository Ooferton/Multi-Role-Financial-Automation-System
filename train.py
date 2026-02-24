import os
import time
import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback
from ml.rl_agent import RLAgent
from data.yahoo_connector import YahooDataConnector
from datetime import datetime, timedelta
import logging

# Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TrainingJob")

class TimeLimitCallback(BaseCallback):
    """
    Stops training when the time limit is reached.
    """
    def __init__(self, time_limit: int, verbose=0):
        super(TimeLimitCallback, self).__init__(verbose)
        self.time_limit = time_limit
        self.start_time = None

    def _on_training_start(self) -> None:
        self.start_time = time.time()

    def _on_step(self) -> bool:
        elapsed = time.time() - self.start_time
        if elapsed > self.time_limit:
            print(f"\nTime limit reached ({self.time_limit}s). Stopping training.")
            return False # Stop training
        return True

def train_real():
    logger.info("Initializing Real-World Training...")
    
    # 1. Fetch Data
    connector = YahooDataConnector()
    symbol = "SPY"
    logger.info(f"Fetching 60 days of 5m data for {symbol}...")
    
    end = datetime.now()
    start = end - timedelta(days=59) # Max 60d for 5m
    
    # Fetch Dataframe directly
    df = connector.fetch_historical_df(symbol, start, end)
    
    if df.empty:
        logger.error("No data fetched. Aborting.")
        return

    logger.info(f"Loaded {len(df)} data points.")

    # 2. Initialize Agent with Data
    # Increased Learning Rate to 0.001 (approx 3x default) for faster convergence
    agent = RLAgent(training_data=df, learning_rate=0.001)
    
    # 3. Train
    # Set a massive timestep count, let the callback stop us
    training_duration_minutes = 10 
    training_duration_seconds = training_duration_minutes * 60
    
    logger.info(f"Starting Training for {training_duration_minutes} minutes with LR=0.001...")
    
    # Callbacks
    from stable_baselines3.common.callbacks import CheckpointCallback
    
    # Save every ~10k steps (approx 5-10 mins)
    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./ml/checkpoints/', name_prefix='ppo_trading')
    time_limit_callback = TimeLimitCallback(time_limit=training_duration_seconds)
    
    callbacks = [checkpoint_callback, time_limit_callback]
    
    try:
        agent.model.learn(total_timesteps=10_000_000, callback=callbacks)
    except KeyboardInterrupt:
        logger.info("Training interrupted by user. Saving current state...")
    
    # 4. Save Model
    os.makedirs("ml/models", exist_ok=True)
    model_path = "ml/models/ppo_trading_real_v1"
    agent.model.save(model_path)
    logger.info(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_real()
