"""
DEPRECATED: bitcoin_runner.py
==============================
This module has been DEPRECATED as of V3 MARL Swarm Architecture.

Bitcoin and crypto trading (BTC/USD, ETH/USD, DOGE/USD, COIN, MSTR, MARA, RIOT, IBIT, BITO)
is now handled by the dedicated CRYPTO sector swarm agent inside `live_runner.py`.

The MARL Swarm manages crypto with:
  - V3 Cyborg Neural Network (ppo_v3_cyborg.zip)
  - 23-dimensional observation space including OFI and macro sentiment
  - Sector-specific Sharpe Ratio capital allocation from the Meta-Orchestrator

To run the live trading system (including crypto), use:
    python live_runner.py

To retrain the V3 model powering all swarm agents, use:
    python train_v3.py --lr 0.0003 --time 90 --seeds 3
"""

import sys
import logging

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("BitRunner")

if __name__ == "__main__":
    logger.warning("=" * 70)
    logger.warning("DEPRECATED: bitcoin_runner.py has been retired.")
    logger.warning("Crypto trading is now managed by the V3 MARL CRYPTO Swarm.")
    logger.warning("Please run: python live_runner.py")
    logger.warning("=" * 70)
    sys.exit(0)
