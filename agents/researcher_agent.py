"""
Autonomous Researcher Agent
Runs daily before market open to scan the historical database and generate data-driven daily targets.
"""

import os
import json
import logging
from datetime import datetime
from ml.research_lab import run_factor_ranking, run_cointegration_scan
from data.feature_store import FeatureStore

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - [RESEARCHER] - %(levelname)s - %(message)s")
logger = logging.getLogger("ResearcherAgent")

TARGETS_PATH = "data/daily_targets.json"

def get_universe_symbols():
    """Retrieve all distinct symbols from the feature store database."""
    try:
        store = FeatureStore()
        symbols = store.get_all_symbols()
        if not symbols:
            # Fallback if DB is empty or missing
            symbols = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA"]
        return symbols
    except Exception as e:
        logger.error(f"Failed to fetch universe symbols: {e}")
        return ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA"]

def run_daily_scan():
    """Executes the quantitative scan across the universe and saves top targets."""
    logger.info("Starting Daily Autonomous Research Scan...")
    
    symbols = get_universe_symbols()
    logger.info(f"Scanning universe of {len(symbols)} symbols...")
    
    # Run Factor Analysis (Momentum, Volatility, Trend)
    logger.info("Running Factor Analysis Scoring...")
    factor_df = run_factor_ranking(symbols, days=90)
    
    top_5_targets = []
    if not factor_df.empty:
        # Get the top 5 highest conviction targets
        top_5_targets = factor_df.head(5)["Symbol"].tolist()
        logger.info(f"Top 5 Conviction Targets Isolated: {top_5_targets}")
    else:
        logger.warning("Factor Analysis returned empty DataFrame. Proceeding with fallback indices.")
        top_5_targets = ["SPY", "QQQ", "IWM", "DIA"]
        
    # Run Cointegration Scan to find statistical arbitrage pairs
    logger.info("Running Cointegration Pairs Scanner...")
    # Scan max 30 symbols to avoid combinatorial explosion
    pair_scan_pool = symbols[:30] 
    pairs_df = run_cointegration_scan(pair_scan_pool, days=180)
    
    top_pair = {}
    if not pairs_df.empty:
        best_pair_row = pairs_df.iloc[0]
        # Only suggest taking action if z-score is significant (> 2.0 or < -2.0)
        z_score = best_pair_row["Spread Z-Score"]
        if abs(z_score) > 2.0:
            top_pair = {
                "pair": best_pair_row["Pair"],
                "z_score": z_score,
                "signal": best_pair_row["Signal"]
            }
            logger.info(f"Actionable Arbitrage Pair Found: {top_pair}")
        else:
            logger.info("No highly-divergent cointegrated pairs found today.")

    # Save to daily_targets.json
    output = {
        "timestamp": datetime.now().isoformat(),
        "dynamic_targets": top_5_targets,
        "arbitrage_pair": top_pair
    }
    
    os.makedirs(os.path.dirname(TARGETS_PATH), exist_ok=True)
    with open(TARGETS_PATH, "w") as f:
        json.dump(output, f, indent=4)
        
    logger.info(f"🎉 Daily Research complete. Targets saved to {TARGETS_PATH}.")

if __name__ == "__main__":
    run_daily_scan()
