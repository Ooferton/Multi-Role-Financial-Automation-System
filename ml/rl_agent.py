import gymnasium as gym
from stable_baselines3 import PPO
from typing import Dict, Any
import numpy as np
from ml.features import TechnicalIndicators

class TradingEnvironment(gym.Env):
    """
    Custom Environment that follows gym interface.
    This connects the RL agent to the FeatureStore and execution logic.
    """
    def __init__(self, df=None):
        super(TradingEnvironment, self).__init__()
        # ACTION: [Position Change] (-1 to 1)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        
        # OBSERVATION: Expanded for Technical Indicators
        # [Price, Returns, RSI, MACD, Signal, BB_Width, SMA_Dist, Position, Cash]
        self.obs_dim = 9
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
        
        # Data
        self.df = df
        if self.df is None:
            self.prices = self._generate_prices()
            self.features = np.zeros((len(self.prices), self.obs_dim)) # Placeholder
        else:
            # Feature Engineering
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
        Extracts relevant features into a numpy array for fast access during stepping.
        Columns: [close, returns (calc on fly/pre), rsi_14, macd, macd_signal, bb_width, dist_sma_20]
        """
        # We need to calculate returns here or rely on the feature lib
        # Let's rely on feature lib or simple diff
        
        # Select columns in specific order
        # [0: Price, 1: RSI, 2: MACD, 3: Signal, 4: BB_Width, 5: SMA_Dist]
        # Returns, Pos, Cash are dynamic
        
        self.feature_array = self.df[[
            'close', 'rsi_14', 'macd', 'macd_signal', 'bb_width', 'dist_sma_20'
        ]].values.astype(np.float32)

    def _generate_prices(self):
        # Generate a random walk for training
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
        
        # If using real data, maybe start at random index?
        # For simplicity, just reset to 0
        
        return self._get_obs(), {}




    def _get_obs(self):
        if self.df is None:
            # Fallback for random walk/tests without features
            price = self.prices[self.current_step]
            return np.zeros(self.obs_dim, dtype=np.float32)

        # Retrieve precomputed features for current step
        # [0: Price, 1: RSI, 2: MACD, 3: Signal, 4: BB_Width, 5: SMA_Dist]
        feats = self.feature_array[self.current_step]
        
        price = feats[0]
        prev_price = self.feature_array[self.current_step - 1][0] if self.current_step > 0 else price
        ret = (price - prev_price) / prev_price if prev_price > 0 else 0.0
        
        # Construct full vector
        # [Price, Returns, RSI, MACD, Signal, BB_Width, SMA_Dist, Position, Cash]
        obs = np.array([
            price,
            ret,
            feats[1], # RSI
            feats[2], # MACD
            feats[3], # MACD Signal
            feats[4], # BB Width
            feats[5], # Dist SMA 20
            self.position,
            self.cash
        ], dtype=np.float32)
            
        return obs

    def step(self, action):
        # 1. Execute Action
        current_price = self.prices[self.current_step]
        trade_size_fraction = float(action[0])
        
        # Simplified execution: Can go Long/Short up to 100% of equity
        # Rebalance portfolio to match target fraction
        total_value = self.cash + (self.position * current_price)
        target_position_value = total_value * trade_size_fraction
        target_position_units = target_position_value / current_price
        
        # Transaction Cost (0.1%)
        trade_value = abs((target_position_units - self.position) * current_price)
        cost = trade_value * 0.001
        
        self.position = target_position_units
        self.cash = total_value - (self.position * current_price) - cost
        
        # 2. Advance Time
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        # 3. Calculate Reward (Change in Net Worth)
        new_total_value = self.cash + (self.position * self.prices[self.current_step])
        reward = new_total_value - total_value
        
        # Optional: Normalize reward
        reward /= 100.0 
        
        return self._get_obs(), reward, terminated, truncated, {"net_worth": new_total_value}

class RLAgent:
    """
    Reinforcement Learning Agent for strategic decision making.
    Wraps stable-baselines3 PPO model.
    """
    def __init__(self, model_path: str = None, training_data=None, learning_rate: float = 0.0003):
        self.env = TradingEnvironment(df=training_data)
        if model_path:
            # When loading, we might not need data if just predicting, 
            # but env creation usually requires it or defaults.
            self.model = PPO.load(model_path, env=self.env)
        else:
            self.model = PPO("MlpPolicy", self.env, verbose=1, learning_rate=learning_rate)

    def predict(self, observation: np.ndarray) -> Any:
        action, _states = self.model.predict(observation)
        return action

    def train(self, total_timesteps: int = 10000):
        self.model.learn(total_timesteps=total_timesteps)
