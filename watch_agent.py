import time
import os
import logging
from ml.rl_agent import RLAgent
from data.news_service import NewsService
import numpy as np

# Configure logging to print to console only
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("WatchMode")

def watch_agent():
    model_path = "ml/models/ppo_trading_v1"
    
    if not os.path.exists(model_path + ".zip"):
        logger.error("No trained model found! Please run train.py first.")
        return

    logger.info(f"Loading model from {model_path}...")
    agent = RLAgent(model_path)
    env = agent.env
    
    # Initialize News Service
    news_service = NewsService()
    logger.info("Fetching live news sentiment...")
    # Mocking fetching for speed in loop, in real app this runs async in background
    current_sentiment = 0.15 # Slightly bullish default
    
    obs, _ = env.reset()
    done = False
    step = 0
    
    logger.info("\n--- STARTING LIVE SIMULATION ---\n")
    logger.info(f"{'Step':<6} | {'Price':<10} | {'Action':<10} | {'Position':<8} | {'Sentiment':<10} | {'Net Worth':<12}")
    logger.info("-" * 75)

    try:
        while not done:
            # Simulate changing sentiment
            if step % 20 == 0:
                # Every 20 steps, shift sentiment randomly
                current_sentiment += np.random.normal(0, 0.1)
                current_sentiment = np.clip(current_sentiment, -1.0, 1.0)
            
            # Inject sentiment into observation (Hack for demo: overwrite volatility slot)
            obs[2] = current_sentiment 
            
            action = agent.predict(obs) # Action is array [-1 to 1]
            
            # Execute step
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Extract info for logging
            price = obs[0]
            position = obs[3]
            net_worth = info['net_worth']
            
            # Interpret action
            act_val = float(action[0])
            act_str = "HOLD"
            if act_val > 0.1: act_str = f"BUY {act_val:.2f}"
            elif act_val < -0.1: act_str = f"SELL {act_val:.2f}"
            
            sent_str = f"{current_sentiment:+.2f}"
            
            logger.info(f"{step:<6} | {price:<10.2f} | {act_str:<10} | {position:<8.2f} | {sent_str:<10} | ${net_worth:<12.2f}")
            
            step += 1
            time.sleep(0.05) # Small delay to make it readable
            
            if step >= 100: # Limit to 100 steps for demo
                break
                
    except KeyboardInterrupt:
        logger.info("\nStopped by user.")

    logger.info("-" * 60)
    logger.info(f"Simulation Complete. Final Net Worth: ${net_worth:.2f}")

if __name__ == "__main__":
    watch_agent()
