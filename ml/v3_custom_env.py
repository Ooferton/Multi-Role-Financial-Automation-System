import numpy as np
import gymnasium as gym
from ml.rl_agent_v2 import TradingEnvironmentV2, V2_INDICATOR_COLUMNS

class V3TradingEnv(TradingEnvironmentV2):
    """
    V3 Environment with Adversarial Crisis Injection and Hierarchical LLM Simulation.
    Extends V2 by:
    1. Periodically injecting artificial market crashes (Black Swans).
    2. Dynamically changing risk constraints mid-episode (simulating the LLM Supervisor).
    3. Expanding the observation space to include the current constraints and crisis state.
    4. Curriculum learning — crisis intensity phases in over time.
    5. Random start offset + noise augmentation to prevent memorization.
    """
    def __init__(self, df=None, crisis_prob=0.005, constraint_change_prob=0.01):
        super(V3TradingEnv, self).__init__(df)
        self.base_crisis_prob = crisis_prob
        self.crisis_prob = 0.0  # Starts at zero — curriculum phases it in
        self.constraint_change_prob = constraint_change_prob
        
        # State tracking for V3 features
        self.in_crisis = False
        self.crisis_step_count = 0
        
        # Curriculum Learning: track global steps across all episodes
        self.global_step_count = 0
        self.curriculum_phase = 0  # 0=Easy, 1=Medium, 2=Hard
        
        # LLM Supervisor constraints (simulated)
        self.max_leverage = 1.0
        self.max_position_frac = 1.0
        self.allow_shorting = 1.0 # 1.0 allows shorting, 0.0 bans shorting
        
        # Expand observation space definition by 9 dimensions:
        # [original_14_dims, in_crisis, max_leverage, max_position_frac, allow_shorting, macro_sentiment, is_bull, is_bear, is_choppy, order_flow_imbalance]
        self.v3_extra_dims = 9
        
        if hasattr(gym.spaces, 'Box'):
            orig_shape = 14
            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, 
                shape=(orig_shape + self.v3_extra_dims,), 
                dtype=np.float32
            )

    def reset(self, seed=None, options=None):
        # We MUST clone prices so adversarial drops don't permanently alter the full dataset
        if not hasattr(self, 'original_prices'):
            self.original_prices = self.prices.copy()
        self.prices = self.original_prices.copy()
        
        # --- Curriculum Learning: Phase in crisis difficulty ---
        if self.global_step_count < 500_000:
            self.curriculum_phase = 0
            self.crisis_prob = 0.0  # Phase 1: No crises, learn basic trading
        elif self.global_step_count < 1_000_000:
            self.curriculum_phase = 1
            self.crisis_prob = self.base_crisis_prob * 0.5  # Phase 2: Light crises
        else:
            self.curriculum_phase = 2
            self.crisis_prob = self.base_crisis_prob  # Phase 3: Full adversarial
        
        # --- Data Augmentation: Random Start Offset ---
        # Instead of always starting at step 0, start at a random point
        # This prevents the agent from memorizing the exact sequence
        buffer = min(200, self.max_steps // 4)  # Leave room for at least 75% of data
        random_offset = np.random.randint(0, buffer) if buffer > 0 else 0
        
        obs, info = super().reset(seed=seed, options=options)
        self.current_step = random_offset
        
        # --- Data Augmentation: Price Noise ---
        # Add slight random perturbation to prices each episode (±0.1%)
        noise = 1.0 + np.random.normal(0, 0.001, len(self.prices))
        self.prices = self.prices * noise.astype(np.float32)
        
        self.in_crisis = False
        self.crisis_step_count = 0
        self._simulate_llm_constraints(force_reset=True)
        return self._get_obs(), info

    def _simulate_llm_constraints(self, force_reset=False):
        """Simulates the LLM Supervisor changing rules based on a schedule or anomaly."""
        if force_reset or np.random.random() < self.constraint_change_prob:
            # Randomly select a new regime constraint
            scenario = np.random.choice(["NORMAL", "CONSERVATIVE", "PANIC"])
            if scenario == "NORMAL":
                self.max_leverage = 1.0
                self.max_position_frac = 1.0
                self.allow_shorting = 1.0
            elif scenario == "CONSERVATIVE":
                self.max_leverage = 0.5
                self.max_position_frac = 0.5
                self.allow_shorting = 0.0 # Ban shorting in conservative mode
            elif scenario == "PANIC":
                self.max_leverage = 0.1
                self.max_position_frac = 0.1
                self.allow_shorting = 1.0 # Allow shorting to profit from drop

    def _get_obs(self):
        # Get baseline 14-dim observation
        obs_v2 = super()._get_obs()
        
        # Read precomputed features from dataframe if available
        macro_sentiment = 0.0
        is_bull, is_bear, is_choppy = 0.0, 0.0, 1.0
        order_flow_imbalance = 0.0
        
        if self.df is not None:
            if 'macro_sentiment' in self.df.columns:
                macro_sentiment = self.df['macro_sentiment'].values[self.current_step]
            if 'is_bull' in self.df.columns:
                is_bull = self.df['is_bull'].values[self.current_step]
                is_bear = self.df['is_bear'].values[self.current_step]
                is_choppy = self.df['is_choppy'].values[self.current_step]
            if 'order_flow_imbalance' in self.df.columns:
                order_flow_imbalance = self.df['order_flow_imbalance'].values[self.current_step]
        
        # Append V3 dimensions
        v3_features = np.array([
            1.0 if self.in_crisis else 0.0,
            self.max_leverage,
            self.max_position_frac,
            self.allow_shorting,
            macro_sentiment,
            is_bull,
            is_bear,
            is_choppy,
            order_flow_imbalance
        ], dtype=np.float32)
        
        return np.nan_to_num(np.concatenate([obs_v2, v3_features]), nan=0.0, posinf=0.0, neginf=0.0)

    def step(self, action):
        # Track global progress for curriculum learning
        self.global_step_count += 1
        
        # 1. Simulate LLM Rule Updates
        self._simulate_llm_constraints()
        
        # 2. Enforce LLM Risk Constraints on Action
        # Action is usually [-1, 1]. We scale it by constraints.
        target_fraction = float(action[0])
        
        if target_fraction < 0 and self.allow_shorting == 0.0:
            target_fraction = 0.0 # Banned from shorting
            
        target_fraction = np.clip(target_fraction, -self.max_position_frac, self.max_position_frac)
        target_fraction *= self.max_leverage
        
        constrained_action = np.array([target_fraction], dtype=np.float32)

        # 3. Simulate Adversarial Crisis
        if not self.in_crisis and np.random.random() < self.crisis_prob:
            self.in_crisis = True
            self.crisis_step_count = 0
            
        original_price = self.prices[self.current_step]
        if self.in_crisis:
            # Drop price aggressively for a fake liquidation event
            crash_depth = np.random.uniform(0.95, 0.98) # 2% to 5% drop per step
            self.prices[self.current_step:] *= crash_depth
            self.crisis_step_count += 1
            if self.crisis_step_count > 10: # Crisis ends after 10 steps
                self.in_crisis = False
                
        # 4. Standard step execution
        # We temporarily override the action passed to super() with our constrained action
        obs, reward, terminated, truncated, info = super().step(constrained_action)
        
        # 5. Reward Shaping for Crisis/Compliance
        price_drop = (self.prices[self.current_step] - original_price) / original_price
        
        if self.in_crisis or price_drop < -0.02:
            if self.position > 0:
                reward -= 10.0 # Massive penalty for holding long during a crash
            elif self.position <= 0:
                reward += 5.0 # Bounty for Crisis Alpha
                
        # Penalty for trying to violate simulated LLM constraints
        # Even if constraints blocked it, the AI should learn not to *try*
        if abs(float(action[0]) - target_fraction) > 0.1:
            reward -= 1.0 # Mild penalty for non-compliance intent
            
        info["v3_crisis"] = self.in_crisis
        info["v3_constraint_scenario"] = "COMPLIANT" if abs(float(action[0]) - target_fraction) <= 0.1 else "VIOLATION"
        
        # Fix the returned observation size (super().step returns normal obs, we need to append)
        obs = self._get_obs()

        return obs, reward, terminated, truncated, info

if __name__ == "__main__":
    print("V3 Custom Environment initialized.")
    env = V3TradingEnv()
    obs, info = env.reset()
    print(f"Observation Shape: {obs.shape}")
