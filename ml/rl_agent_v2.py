"""
PPO Trading Real v2 — RL Agent with 10 Market Indicators.

Observation space expanded from 9-dim (v1) to 14-dim:
  [Price, Returns, RSI, MACD, MACD_Signal, BB_Width, Dist_SMA_20,
   MACD_Histogram, BB_Position, Dist_SMA_50, ATR_norm, OBV_norm,
   Position, Cash]

This provides richer market context for higher returns.
Supports NVIDIA GPU acceleration via CUDA.
"""
import sys
import numpy as np
import numpy

# --- NumPy 2.0 Compatibility Shim ---
# Enables loading models saved with NumPy 2.x on systems with NumPy 1.x (e.g. Linux Codespaces)
# We omit the 2.x -> 1.x alias as it causes RecursionErrors in some environments (like Python 3.14).
if numpy.__version__.startswith("1."):
    try:
        import numpy.core as _core
        sys.modules["numpy._core"] = _core
    except ImportError:
        pass
# ------------------------------------

import gymnasium as gym
from stable_baselines3 import PPO
from typing import Any
import numpy as np
import torch
from ml.features import TechnicalIndicators

# The 10 indicator columns in the order they appear in the observation vector
V2_INDICATOR_COLUMNS = [
    'close',           # Price
    'rsi_14',          # 1. Momentum
    'macd',            # 2. Momentum
    'macd_signal',     # 3. Momentum
    'bb_width',        # 4. Volatility
    'dist_sma_20',     # 5. Trend
    'macd_histogram',  # 6. Momentum (NEW)
    'bb_position',     # 7. Volatility (NEW)
    'dist_sma_50',     # 8. Trend (NEW)
    'atr_norm',        # 9. Volatility (NEW)
    'obv_norm',        # 10. Volume (NEW)
]


class TradingEnvironmentV2(gym.Env):
    """
    Enhanced trading environment with 10 market indicators.
    Observation: 14-dim vector (10 indicators + price + returns + position + cash).
    Action: Continuous [-1, 1] representing target portfolio fraction.
    """
    def __init__(self, df=None):
        super(TradingEnvironmentV2, self).__init__()
        
        # ACTION: [Position Change] (-1 to 1)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        
        # OBSERVATION: 14 dimensions
        # [Price, Returns, 10 indicators, Position, Cash]
        self.obs_dim = 14
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )
        
        # Data
        self.df = df
        if self.df is None:
            self.prices = self._generate_prices()
            self.features = np.zeros((len(self.prices), len(V2_INDICATOR_COLUMNS)))
        else:
            self.df = TechnicalIndicators.add_all_features(self.df)
            self.prices = self.df['close'].values.astype(np.float32)
            self._precompute_features()
            
        # Simulation State
        self.current_step = 0
        self.max_steps = len(self.prices) - 1
        
        self.position = 0.0
        self.cash = 10000.0
        self.initial_value = 10000.0

    def _precompute_features(self):
        """
        Extracts all 10 indicator columns into a numpy array for fast stepping.
        """
        self.feature_array = self.df[V2_INDICATOR_COLUMNS].values.astype(np.float32)

    def _generate_prices(self):
        """Generate random walk for training without real data."""
        np.random.seed(42)
        self.max_steps = 1000
        returns = np.random.normal(0, 0.01, self.max_steps + 1)
        prices = 100 * np.cumprod(1 + returns)
        return prices.astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.position = 0.0
        self.cash = 10000.0
        return self._get_obs(), {}

    def _get_obs(self):
        if self.df is None:
            return np.zeros(self.obs_dim, dtype=np.float32)

        feats = self.feature_array[self.current_step]
        
        price = feats[0]  # close
        prev_price = self.feature_array[self.current_step - 1][0] if self.current_step > 0 else price
        ret = (price - prev_price) / prev_price if prev_price > 0 else 0.0
        
        # Construct 14-dim vector:
        # [Price, Returns, RSI, MACD, Signal, BB_Width, Dist_SMA_20,
        #  Histogram, BB_Pos, Dist_SMA_50, ATR_norm, OBV_norm,
        #  Position, Cash]
        obs = np.array([
            price,
            ret,
            feats[1],   # RSI
            feats[2],   # MACD
            feats[3],   # MACD Signal
            feats[4],   # BB Width
            feats[5],   # Dist SMA 20
            feats[6],   # MACD Histogram (NEW)
            feats[7],   # BB Position (NEW)
            feats[8],   # Dist SMA 50 (NEW)
            feats[9],   # ATR Normalized (NEW)
            feats[10],  # OBV Normalized (NEW)
            self.position,
            self.cash
        ], dtype=np.float32)
            
        return obs

    def step(self, action):
        # 1. Execute Action
        current_price = self.prices[self.current_step]
        trade_size_fraction = float(action[0])
        
        total_value = self.cash + (self.position * current_price)
        target_position_value = total_value * trade_size_fraction
        target_position_units = target_position_value / current_price if current_price > 0 else 0
        
        # Transaction Cost (0.1%)
        trade_value = abs((target_position_units - self.position) * current_price)
        cost = trade_value * 0.001
        
        self.position = target_position_units
        self.cash = total_value - (self.position * current_price) - cost
        
        # 2. Advance Time
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        # 3. Calculate Reward
        new_price = self.prices[self.current_step]
        new_total_value = self.cash + (self.position * new_price)
        
        # Sharpe-inspired reward: risk-adjusted return
        raw_return = (new_total_value - total_value) / total_value
        reward = raw_return * 100  # Scale for learning stability
        
        # Penalty for excessive trading (reduces churn)
        if trade_value > total_value * 0.3:
            reward -= 0.05
        
        return self._get_obs(), reward, terminated, truncated, {"net_worth": new_total_value}


class RLAgentV2:
    """
    PPO Trading Real v2 — Reinforcement Learning Agent with 10 market indicators.
    Wraps stable-baselines3 PPO model with the enhanced 14-dim observation space.
    Supports NVIDIA GPU acceleration via CUDA.
    """
    def __init__(self, model_path: str = None, training_data=None, learning_rate: float = 0.001):
        self.env = TradingEnvironmentV2(df=training_data)
        
        # Auto-detect GPU
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if model_path:
            # Allow overriding learning rate when loading
            self.model = PPO.load(model_path, env=self.env, device=self.device, custom_objects={'learning_rate': learning_rate})
        else:
            self.model = PPO(
                "MlpPolicy", 
                self.env, 
                verbose=1, 
                learning_rate=learning_rate,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,
                device=self.device,
            )

    def predict(self, observation: np.ndarray) -> Any:
        action, _states = self.model.predict(observation)
        return action

    def train(self, total_timesteps: int = 50000, callbacks=None):
        self.model.learn(total_timesteps=total_timesteps, callback=callbacks)

    def save(self, path: str):
        self.model.save(path)
