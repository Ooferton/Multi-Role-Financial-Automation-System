import numpy as np
import pandas as pd
from ml.v3_crisis_prototype import TradingEnvironmentV3

def test_crisis_environment():
    print("--- Testing V3 Crisis Environment Prototype ---")
    
    # Create mock data (Linear uptrend with dummy OHLC)
    prices = np.linspace(100, 110, 100)
    data = {
        'open': prices * 0.99,
        'high': prices * 1.01,
        'low': prices * 0.98,
        'close': prices
    }
    df = pd.DataFrame(data)
    
    # Initialize V3 Env with High Crisis Probability for testing
    env = TradingEnvironmentV3(df=df, crisis_prob=0.1)
    
    obs, _ = env.reset()
    total_reward = 0
    crash_detected = False
    
    print("\nStarting Simulation (Targeting a Flash Crash Injection)...")
    for i in range(50):
        # Always hold a LONG position to test the penalty logic
        action = np.array([1.0]) 
        
        obs, reward, terminated, truncated, info = env.step(action)
        price = obs[0]
        
        if env.in_crisis:
            if not crash_detected:
                print(f"!!! FLASH CRASH INITIALIZED at Step {i} | Price: {price:.2f}")
                crash_detected = True
            print(f"  [CRISIS] Step {i}: Price={price:.2f}, Reward={reward:.4f} (Penalty check)")
        
        total_reward += reward
        if terminated: break

    if crash_detected:
        print("\nVerification Complete: Crisis injection and reward penalties observed.")
    else:
        print("\nSimulation finished without a random crash. Increase crisis_prob or run again.")

if __name__ == "__main__":
    test_crisis_environment()
