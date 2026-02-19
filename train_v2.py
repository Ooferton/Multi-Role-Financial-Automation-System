"""
PPO Trading Real v2 — Training Script.

Trains a PPO model with 10 market indicators (14-dim observation).
Uses real market data from Yahoo Finance.
Saves model to ml/models/ppo_trading_real_v2.
"""
import os
import time
import torch
import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from ml.rl_agent_v2 import RLAgentV2
from data.yahoo_connector import YahooDataConnector
from datetime import datetime, timedelta
import logging

# Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TrainingJob_v2")

class TimeLimitCallback(BaseCallback):
    """Stops training when the time limit is reached."""
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
            return False
        return True

class PerformanceLogCallback(BaseCallback):
    """Logs training metrics periodically."""
    def __init__(self, log_freq: int = 1000, verbose=0):
        super(PerformanceLogCallback, self).__init__(verbose)
        self.log_freq = log_freq
        
    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq == 0:
            if 'net_worth' in self.locals.get('infos', [{}])[-1]:
                net_worth = self.locals['infos'][-1]['net_worth']
                print(f"  Step {self.n_calls}: Net Worth = ${net_worth:,.2f}")
        return True

def train_v2():
    logger.info("=" * 60)
    logger.info("PPO Trading Real v2 — Training with 10 Market Indicators")
    logger.info("=" * 60)
    
    # 1. Fetch Data
    connector = YahooDataConnector()
    symbol = "SPY"
    logger.info(f"Fetching 60 days of 5m data for {symbol}...")
    
    end = datetime.now()
    start = end - timedelta(days=59)  # Max 60d for 5m
    
    df = connector.fetch_historical_df(symbol, start, end)
    
    if df.empty:
        logger.error("No data fetched. Aborting.")
        return
    
    logger.info(f"Loaded {len(df)} data points.")
    logger.info(f"Date range: {df.index[0] if hasattr(df.index, '__getitem__') else 'N/A'} to {df.index[-1] if hasattr(df.index, '__getitem__') else 'N/A'}")

    # GPU Detection
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"🚀 GPU DETECTED: {gpu_name} — CUDA acceleration enabled!")
    else:
        logger.info("⚠️ No GPU detected — training on CPU")

    # 2. Initialize v2 Agent with Data
    # Higher LR for faster convergence with GPU
    agent = RLAgentV2(training_data=df, learning_rate=0.003)
    
    logger.info(f"Observation space: {agent.env.observation_space.shape}")
    logger.info(f"Action space: {agent.env.action_space.shape}")
    
    # 3. Train
    training_duration_minutes = 15  # Longer for richer model
    training_duration_seconds = training_duration_minutes * 60
    
    logger.info(f"Starting Training for {training_duration_minutes} minutes with LR=0.003...")
    logger.info(f"Device: {agent.device}")
    logger.info(f"Observation dimensions: 14 (10 indicators + price + returns + position + cash)")
    
    # Callbacks
    os.makedirs('./ml/checkpoints/', exist_ok=True)
    checkpoint_callback = CheckpointCallback(
        save_freq=10000, 
        save_path='./ml/checkpoints/', 
        name_prefix='ppo_trading_v2'
    )
    time_limit_callback = TimeLimitCallback(time_limit=training_duration_seconds)
    perf_callback = PerformanceLogCallback(log_freq=2000)
    
    callbacks = [checkpoint_callback, time_limit_callback, perf_callback]
    
    try:
        agent.train(total_timesteps=10_000_000, callbacks=callbacks)
    except KeyboardInterrupt:
        logger.info("Training interrupted by user. Saving current state...")
    
    # 4. Save Model
    os.makedirs("ml/models", exist_ok=True)
    model_path = "ml/models/ppo_trading_real_v2"
    agent.save(model_path)
    logger.info(f"✅ Model saved to {model_path}")
    logger.info(f"   Observation shape: {agent.env.observation_space.shape}")
    logger.info(f"   Indicators: 10 (RSI, MACD, Signal, BB_Width, SMA20_Dist, "
                f"Histogram, BB_Pos, SMA50_Dist, ATR_norm, OBV_norm)")

if __name__ == "__main__":
    train_v2()
