import numpy as np
import gymnasium as gym
from ml.rl_agent_v2 import TradingEnvironmentV2, V2_INDICATOR_COLUMNS

class TradingEnvironmentV3(TradingEnvironmentV2):
    """
    Experimental V3 Environment with Adversarial Crisis Injection.
    """
    def __init__(self, df=None, crisis_prob=0.01):
        super(TradingEnvironmentV3, self).__init__(df)
        self.crisis_prob = crisis_prob
        self.in_crisis = False
        self.crisis_step_count = 0
        
    def step(self, action):
        # 1. Randomly trigger a "Flash Crash"
        if not self.in_crisis and np.random.random() < self.crisis_prob:
            self.in_crisis = True
            self.crisis_step_count = 0
            
        # 2. If in crisis, modify the current price to simulate a crash
        # This deviates from history to create the "Adversarial" training sample
        original_price = self.prices[self.current_step]
        if self.in_crisis:
            # Drop price by 1-5% per step for a sudden liquidation event
            crash_depth = np.random.uniform(0.95, 0.99)
            self.prices[self.current_step:] *= crash_depth # Permanent (intra-episode) drop
            self.crisis_step_count += 1
            
            if self.crisis_step_count > 5: # Crisis event window
                self.in_crisis = False
        
        # 3. Standard step logic
        obs, reward, terminated, truncated, info = super().step(action)
        
        # 4. Crisis-Aware Reward Modification
        # Heavily penalize drawdowns during a crash to force defensiveness
        price_drop = (self.prices[self.current_step] - original_price) / original_price
        if price_drop < -0.03: # 3% drop detected
            # If the agent is LONG, hit them with a massive penalty
            if self.position > 0:
                reward -= 5.0 # Survival over Profit
            # If the agent is SHORT or in CASH, give a bounty
            elif self.position <= 0:
                reward += 2.0 # Reward for "Crisis Alpha" (Short/Cash preservation)
                
        return obs, reward, terminated, truncated, info

if __name__ == "__main__":
    print("V3 Crisis Prototype Loaded.")
    print("This environment can inject synthetic volatility events to train for survival.")
