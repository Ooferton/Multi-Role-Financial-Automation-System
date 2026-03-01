import os
import argparse
import logging
import yfinance as yf
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import VecNormalize
import time
import numpy as np

from ml.v3_custom_env import V3TradingEnv
from ml.features import TechnicalIndicators

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TrainV3")

class TimeLimitCallback(BaseCallback):
    """Callback that stops training after a specified number of minutes."""
    def __init__(self, time_limit_minutes: float, verbose=1):
        super().__init__(verbose)
        self.time_limit_seconds = time_limit_minutes * 60
        self.start_time = None
        
    def _on_training_start(self) -> None:
        self.start_time = time.time()
        self.logger.info(f"Training started. Time limit set to {self.time_limit_seconds / 60} minutes.")
        
    def _on_step(self) -> bool:
        if (time.time() - self.start_time) > self.time_limit_seconds:
            self.logger.info(f"Halting training: Time limit of {self.time_limit_seconds / 60} minutes reached.")
            return False
        return True

def add_v3_state_features(df):
    """
    Precomputes the NLP Sentiment proxies and mathematical Regime states
    for historical training data, avoiding massive computational bottleneck.
    """
    df = df.copy()
    df['returns'] = df['close'].pct_change()
    df['vol_20'] = df['returns'].rolling(20).std()
    
    median_vol = df['vol_20'].median()
    
    # HMM Proxy: Bull = Price > SMA200 + low/med vol, Bear = Price < SMA200 + high vol
    df['is_bull'] = ((df['close'] > df['sma_200']) & (df['vol_20'] < median_vol * 1.5)).astype(float)
    df['is_bear'] = ((df['close'] < df['sma_200']) & (df['vol_20'] > median_vol)).astype(float)
    df['is_choppy'] = ((df['is_bull'] == 0) & (df['is_bear'] == 0)).astype(float)
    
    # NLP Proxy: Historical sentiment simulated via price momentum + noise
    recent_return = df['close'].pct_change(5)
    sentiment = (recent_return / 0.05).clip(-1, 1) # ~5% move = max sentiment
    noise = np.random.normal(0, 0.2, len(df))
    df['macro_sentiment'] = (sentiment + noise).clip(-1, 1)
    
    # OFI Proxy: Historical OFI simulated via return direction scaled by relative volume
    rolling_vol = df['volume'].rolling(20).mean()
    vol_ratio = (df['volume'] / rolling_vol).replace([np.inf, -np.inf], 1.0).fillna(1.0).clip(0, 5)
    df['order_flow_imbalance'] = (np.sign(df['returns']) * (vol_ratio * 0.2)).clip(-1.0, 1.0).fillna(0.0)
    
    df['macro_sentiment'] = df['macro_sentiment'].fillna(0.0)
    
    df.replace([np.inf, -np.inf], 0, inplace=True)
    df.fillna(0, inplace=True)
    return df

def fetch_training_data(tickers):
    logger.info(f"Downloading historical data for {len(tickers)} tickers via yfinance...")
    # Download 2 years of daily data for broad learning
    data = yf.download(tickers, period="2y", interval="1d", group_by="ticker", threads=True, progress=False)
    
    formatted_dfs = []
    
    # Process each ticker
    for ticker in tickers:
        try:
            if len(tickers) == 1:
                df = data.copy()
            else:
                df = data[ticker].copy()
                
            df.dropna(inplace=True)
            if len(df) < 100:
                continue
                
            # Rename columns to match what TechnicalIndicators expects
            df = df.rename(columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume"
            })
            
            # Add technical features
            df = TechnicalIndicators.add_all_features(df)
            
            # Add V3 Phase D specific features (Regime & Sentiment)
            df = add_v3_state_features(df)
            
            formatted_dfs.append(df)
        except Exception as e:
            logger.warning(f"Failed to process {ticker}: {e}")
            
    if not formatted_dfs:
        logger.error("Failed to download or process any valid ticker data.")
        return None
        
    # Concatenate all tickers into one massive sequence for the environment
    # The agent will learn generalized patterns across all sectors
    final_df = pd.concat(formatted_dfs, ignore_index=True)
    logger.info(f"Successfully built training dataset with {len(final_df)} market candles.")
    return final_df

def linear_schedule(initial_lr: float):
    """Returns a function that linearly decays the learning rate from initial_lr → 0."""
    def schedule(progress_remaining: float) -> float:
        return progress_remaining * initial_lr
    return schedule

def train_single_agent(historical_df, total_timesteps, lr, time_limit_mins, seed=0):
    """Trains a single PPO agent and returns its model + mean reward."""
    logger.info(f"  [Seed {seed}] Initializing V3 Trading Environment...")
    
    raw_env = V3TradingEnv(df=historical_df, crisis_prob=0.01, constraint_change_prob=0.05)
    
    # Wrap in VecNormalize for automatic observation & reward normalization
    vec_env = DummyVecEnv([lambda: raw_env])
    env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)
    
    # #4: Larger Neural Network (256-256-128) for 23-dim input
    policy_kwargs = dict(net_arch=[256, 256, 128])
    
    logger.info(f"  [Seed {seed}] Initializing PPO (Net: 256-256-128, LR: {lr} → 0 decay)...")
    model = PPO(
        "MlpPolicy", env, verbose=1, 
        learning_rate=linear_schedule(lr),  # #1: LR Scheduling (linear decay)
        n_steps=2048, 
        batch_size=64, 
        max_grad_norm=0.5, 
        clip_range=0.2,
        ent_coef=0.01,
        gamma=0.99,
        gae_lambda=0.95,
        seed=seed,
        policy_kwargs=policy_kwargs,
    )
    
    # Save checkpoints every 10k steps
    os.makedirs("ml/checkpoints", exist_ok=True)
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="ml/checkpoints/",
        name_prefix=f"v3_cyborg_seed{seed}"
    )
    
    time_callback = TimeLimitCallback(time_limit_minutes=time_limit_mins)
    callbacks = CallbackList([checkpoint_callback, time_callback])
    
    logger.info(f"  [Seed {seed}] Starting training for {total_timesteps} timesteps or {time_limit_mins} mins.")
    model.learn(total_timesteps=total_timesteps, callback=callbacks)
    
    # Evaluate: run 10 episodes and compute mean reward
    total_reward = 0.0
    n_eval = 10
    for _ in range(n_eval):
        obs = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            ep_reward += reward[0]
        total_reward += ep_reward
    
    mean_reward = total_reward / n_eval
    logger.info(f"  [Seed {seed}] Training complete. Mean Eval Reward: {mean_reward:.2f}")
    return model, mean_reward


def train_v3(total_timesteps=5000000, model_path="ml/models/ppo_v3_cyborg", lr=0.0003, time_limit_mins=90, n_seeds=3):
    # 59 Diversified Tickers Across All Sectors
    target_symbols = [
        # Tech & Comms
        "AAPL", "MSFT", "GOOGL", "META", "NVDA", "AMD", "INTC", "CSCO", "CRM", "AVGO",
        # Finance
        "JPM", "BAC", "WFC", "C", "GS", "MS", "V", "MA", "AXP", "BLK",
        # Healthcare
        "JNJ", "UNH", "PFE", "ABBV", "TMO", "MRK", "DHR", "LLY",
        # Consumer
        "AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "WMT", "PG", "KO", "PEP",
        # Industrials & Defense
        "HON", "UPS", "BA", "LMT", "CAT", "GE",
        # Energy & Materials
        "XOM", "CVX", "COP", "SLB", "LIN", "NEM",
        # Crypto & Bitcoin Proxies (yfinance format)
        "BTC-USD", "ETH-USD", "DOGE-USD", "MSTR", "COIN", "MARA", "RIOT", "IBIT", "BITO"
    ]
    
    historical_df = fetch_training_data(target_symbols)

    logger.info("=" * 60)
    logger.info(f"V3 POPULATION TRAINING: {n_seeds} agents, best-of-{n_seeds} selection")
    logger.info(f"LR: {lr} → 0 (linear decay) | Net: 256-256-128 | Curriculum: 3-phase")
    logger.info("=" * 60)
    
    # #5: Population Training — train N agents with different seeds, keep the best
    time_per_agent = time_limit_mins / n_seeds
    best_model = None
    best_reward = -float('inf')
    best_seed = 0
    
    for seed in range(n_seeds):
        logger.info(f"\n{'='*40} AGENT {seed+1}/{n_seeds} (Seed: {seed}) {'='*40}")
        model, mean_reward = train_single_agent(
            historical_df, total_timesteps, lr, time_per_agent, seed=seed
        )
        
        if mean_reward > best_reward:
            best_reward = mean_reward
            best_model = model
            best_seed = seed
            logger.info(f"  ★ New best agent! Seed {seed} with reward {mean_reward:.2f}")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"POPULATION WINNER: Seed {best_seed} | Reward: {best_reward:.2f}")
    logger.info(f"Saving best V3 Model to {model_path}...")
    logger.info(f"{'='*60}")
    
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    best_model.save(model_path)
    logger.info("V3 Model Saved Successfully. Ready for production.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the V3 Cyborg Neural Network.")
    parser.add_argument("--timesteps", type=int, default=5000000, help="Total training timesteps per agent.")
    parser.add_argument("--lr", type=float, default=0.0003, help="Initial learning rate (decays to 0).")
    parser.add_argument("--time", type=int, default=90, help="Total time limit in minutes (split across agents).")
    parser.add_argument("--seeds", type=int, default=3, help="Number of population agents to train.")
    args = parser.parse_args()
    
    train_v3(total_timesteps=args.timesteps, lr=args.lr, time_limit_mins=args.time, n_seeds=args.seeds)
