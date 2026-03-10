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
import os
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

try:
    import gymnasium as gym
    _GYM_AVAILABLE = True
except ImportError:
    _GYM_AVAILABLE = False

try:
    from stable_baselines3 import PPO
    _SB3_AVAILABLE = True
except ImportError:
    _SB3_AVAILABLE = False

try:
    from huggingface_hub import hf_hub_download, HfApi
    _HF_HUB_AVAILABLE = True
except ImportError:
    _HF_HUB_AVAILABLE = False

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

from typing import Any
import numpy as np
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


# Conditional base class: use gym.Env if available, otherwise plain object
_GymEnvBase = gym.Env if _GYM_AVAILABLE else object

class TradingEnvironmentV2(_GymEnvBase):
    """
    Enhanced trading environment with 10 market indicators.
    Observation: 14-dim vector (10 indicators + price + returns + position + cash).
    Action: Continuous [-1, 1] representing target portfolio fraction.
    """
    def __init__(self, df=None):
        if _GYM_AVAILABLE:
            super(TradingEnvironmentV2, self).__init__()
        
        # ACTION: [Position Change] (-1 to 1)
        if _GYM_AVAILABLE:
            self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        
        # OBSERVATION: 14 dimensions
        # [Price, Returns, 10 indicators, Position, Cash]
        self.obs_dim = 14
        if _GYM_AVAILABLE:
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
        self.initial_price = self.prices[0] if len(self.prices) > 0 else 100.0

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
        self.initial_price = self.prices[0] if len(self.prices) > 0 else 100.0
        return self._get_obs(), {}

    def _get_obs(self):
        if self.df is None:
            return np.zeros(self.obs_dim, dtype=np.float32)

        feats = self.feature_array[self.current_step]
        
        price = feats[0]  # close
        prev_price = self.feature_array[self.current_step - 1][0] if self.current_step > 0 else price
        ret = (price - prev_price) / prev_price if prev_price > 0 else 0.0
        
        # Normalize price and cash relative to initial values so all features are ~[0, 2]
        norm_price = price / self.initial_price if self.initial_price > 0 else 0.0
        norm_cash = self.cash / self.initial_value
        norm_position = np.clip(self.position * price / self.initial_value, -2.0, 2.0)
        
        # Construct 14-dim vector:
        # [NormPrice, Returns, RSI, MACD, Signal, BB_Width, Dist_SMA_20,
        #  Histogram, BB_Pos, Dist_SMA_50, ATR_norm, OBV_norm,
        #  NormPosition, NormCash]
        obs = np.array([
            norm_price,
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
            norm_position,
            norm_cash
        ], dtype=np.float32)
        
        # Sanitize: replace any NaN/Inf with 0 before PyTorch sees it
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
            
        return obs

    def step(self, action):
        # 1. Execute Action
        current_price = self.prices[self.current_step]
        trade_size_fraction = float(action[0])
        
        total_value = self.cash + (self.position * current_price)
        
        # Guard against zero/negative total value
        if not np.isfinite(total_value) or total_value <= 0:
            total_value = self.initial_value
            self.cash = self.initial_value
            self.position = 0.0
        
        target_position_value = total_value * trade_size_fraction
        target_position_units = target_position_value / current_price if current_price > 1e-6 else 0.0
        
        # Clamp position units to prevent float overflow
        target_position_units = np.clip(target_position_units, -1e8, 1e8)
        
        # Transaction Cost (0.1%)
        trade_value = abs((target_position_units - self.position) * current_price)
        if not np.isfinite(trade_value):
            trade_value = 0.0
        cost = trade_value * 0.001
        
        self.position = target_position_units
        self.cash = total_value - (self.position * current_price) - cost
        
        # Clamp cash to prevent overflow
        self.cash = np.clip(self.cash, -1e10, 1e10)
        
        # 2. Advance Time
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        # 3. Calculate Reward
        new_price = self.prices[self.current_step]
        new_total_value = self.cash + (self.position * new_price)
        
        if not np.isfinite(new_total_value):
            new_total_value = total_value
        
        # Sharpe-inspired reward: log-return for scale invariance
        raw_return = (new_total_value - total_value) / total_value if total_value > 0 else 0.0
        reward = float(np.clip(raw_return * 100, -10, 10))  # Clamped for stability
        
        # Penalty for excessive trading (reduces churn)
        if trade_value > total_value * 0.3:
            reward -= 0.05
        
        return self._get_obs(), reward, terminated, truncated, {"net_worth": new_total_value}


class LiveReplayEnv(_GymEnvBase):
    """
    A specialized Gymnasium environment that replays a fixed set of real-world experiences
    so that the PPO model can train on live data dynamically.
    """
    def __init__(self, experiences: list):
        if _GYM_AVAILABLE:
            super().__init__()
            self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
            # Obs dim must match RLAgentV2. Expecting 14~23 depending on V3 extensions
            obs_dim = len(experiences[0]['obs']) if experiences else 14
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
            
        self.experiences = experiences
        self.current_step = 0

    def reset(self, seed=None, options=None):
        if _GYM_AVAILABLE:
            super().reset(seed=seed)
        self.current_step = 0
        if not self.experiences:
            # Fallback zero obs if empty
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            return obs, {} if _GYM_AVAILABLE else obs
            
        obs = np.array(self.experiences[0]['obs'], dtype=np.float32)
        return obs, {} if _GYM_AVAILABLE else obs

    def step(self, action):
        if self.current_step >= len(self.experiences):
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            return obs, 0.0, True, False, {}

        # The reward is exactly what we logged from the real world outcome
        reward = float(self.experiences[self.current_step]['reward'])
        
        self.current_step += 1
        terminated = self.current_step >= len(self.experiences)
        
        if terminated:
            next_obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        else:
            next_obs = np.array(self.experiences[self.current_step]['obs'], dtype=np.float32)
            
        return next_obs, reward, terminated, False, {}



class RLAgentV2:
    """
    PPO Trading Real v2 — Reinforcement Learning Agent with 10 market indicators.
    Wraps stable-baselines3 PPO model with the enhanced 14-dim observation space.
    Supports heuristic fallback if SB3 is missing (e.g. cloud environments).
    """
    def __init__(self, model_path: str = None, training_data=None, learning_rate: float = 0.001):
        self.env = TradingEnvironmentV2(df=training_data)
        self.model = None
        
        if _SB3_AVAILABLE:
            # Auto-detect GPU
            self.device = 'cuda' if (_TORCH_AVAILABLE and torch.cuda.is_available()) else 'cpu'
            
            hf_repo_id = os.environ.get("HF_MODEL_REPO_ID")
            hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
            
            # --- HF Hub Download Attempt ---
            if hf_repo_id and _HF_HUB_AVAILABLE and model_path and not (os.path.exists(model_path) or os.path.exists(model_path + ".zip")):
                try:
                    filename = os.path.basename(model_path) + ".zip"
                    print(f"☁️ Downloading {filename} from HF Hub ({hf_repo_id})...")
                    downloaded_path = hf_hub_download(repo_id=hf_repo_id, filename=filename, token=hf_token)
                    import shutil
                    os.makedirs(os.path.dirname(model_path), exist_ok=True)
                    shutil.copy(downloaded_path, model_path + ".zip")
                    print(f"✅ Successfully downloaded {filename} to local cache.")
                except Exception as e:
                    print(f"⚠️ Could not download model from HF Hub: {e}")
            # -------------------------------
            
            if model_path and (os.path.exists(model_path) or os.path.exists(model_path + ".zip")):
                try:
                    self.model = PPO.load(model_path, env=self.env, device=self.device, custom_objects={'learning_rate': learning_rate})
                except Exception as e:
                    print(f"Warning: Failed to load RL model, using heuristic: {e}")
            elif not model_path:
                self.model = PPO(
                    "MlpPolicy", 
                    self.env, 
                    verbose=1, 
                    learning_rate=learning_rate,
                    device=self.device,
                )
        else:
            print("RL Brain (SB3) not found. Initializing Heuristic Fallback Agent.")

    def predict(self, observation: np.ndarray) -> Any:
        if self.model:
            action, _states = self.model.predict(observation)
            return action
        
        # Heuristic Fallback (Wait if RSI neutral, Buy if low, Sell if high)
        # obs order: [Price, Returns, RSI, MACD, Signal, BB_Width, Trend, ...]
        rsi = observation[2]
        if rsi < 30: return np.array([0.8], dtype=np.float32) # Oversold -> Buy
        if rsi > 70: return np.array([-0.8], dtype=np.float32) # Overbought -> Sell
        return np.array([0.0], dtype=np.float32) # Neutral

    def train(self, total_timesteps: int = 50000, callbacks=None):
        if self.model:
            self.model.learn(total_timesteps=total_timesteps, callback=callbacks)

    def train_on_live_experiences(self, experiences: list, epochs: int = 1):
        """
        Triggers an online PPO micro-batch update using real-world trade outcomes.
        """
        if not self.model or getattr(self.model, 'env', None) is None:
            print("Cannot train live: Model or base environment not loaded.")
            return

        if not experiences:
            return

        print(f"🧠 [LIVE LEARNING] Triggering PPO update on {len(experiences)} real-world trades over {epochs} epochs...")
        
        # 1. Store the original environment so we can restore it after
        original_env = self.model.env
        
        # 2. Create the replay environment with our live buffer
        replay_env = LiveReplayEnv(experiences)
        
        if _SB3_AVAILABLE:
            from stable_baselines3.common.vec_env import DummyVecEnv
            vec_env = DummyVecEnv([lambda: replay_env])
            
            # 3. Temporarily swap the model's environment
            self.model.set_env(vec_env)
            
            # 4. Run the learning process (Timesteps = Epochs * Experiences)
            timesteps = len(experiences) * epochs
            try:
                self.model.learn(total_timesteps=timesteps)
                print(f"✅ [LIVE LEARNING] Successfully updated neural weights across {timesteps} steps.")
            except Exception as e:
                print(f"❌ [LIVE LEARNING] Training failed: {e}")
            finally:
                # 5. Restore the original environment to prevent state bleeding
                self.model.set_env(original_env)

    def save(self, path: str):
        if self.model:
            self.model.save(path)
            
            # --- HF Hub Upload ---
            if _HF_HUB_AVAILABLE:
                hf_repo_id = os.environ.get("HF_MODEL_REPO_ID")
                hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
                
                if hf_repo_id and hf_token:
                    try:
                        api = HfApi()
                        filename = os.path.basename(path) + ".zip"
                        print(f"☁️ Uploading {filename} back to HF Hub ({hf_repo_id})...")
                        api.upload_file(
                            path_or_fileobj=path + ".zip",
                            path_in_repo=filename,
                            repo_id=hf_repo_id,
                            token=hf_token,
                            commit_message="Live AI Training Update 🧠"
                        )
                        print("✅ HF Hub upload complete.")
                    except Exception as e:
                        print(f"❌ HF Hub upload failed: {e}")
