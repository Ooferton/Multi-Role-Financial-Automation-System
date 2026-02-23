import os
import logging
from datetime import datetime
from dotenv import load_dotenv
from agents.alpaca_broker import AlpacaBroker
from agents.economist_agent import EconomistAgent
from ml.news_sentiment import NewsSentimentEngine

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger("DueDiligence")

def run_diagnostics():
    load_dotenv()
    logger.info("=== 🔍 QUANT SYSTEM DATA DUE-DILIGENCE REPORT ===")
    
    # 1. Verify Alpaca Marketplace Connection (Real-time Prices)
    logger.info("\n--- 1. Alpaca Snapshots (Live Prices) ---")
    broker = AlpacaBroker(paper=True)
    symbols = ["SPY", "AAPL", "TLT", "XLK"]
    prices = broker.get_latest_prices(symbols)
    
    if prices:
        for s, p in prices.items():
            logger.info(f"✅ Captured {s}: ${p:.2f}")
    else:
        logger.error("❌ FAILED to pull live prices from Alpaca. Check API keys.")

    # 2. Verify Economist Agent (Macro Telemetry)
    logger.info("\n--- 2. Economist Agent (yfinance Macro) ---")
    economist = EconomistAgent({}) # Empty config is fine for now
    outlook = economist.update_outlook()
    
    vix = outlook['metrics'].get('VIX')
    if vix:
        logger.info(f"✅ Captured VIX: {vix:.2f}")
        logger.info(f"✅ Market Regime: {outlook['vibe']}")
    else:
        logger.error("❌ FAILED to pull macro data from yfinance.")

    # 3. Verify News Sentiment Engine (Hearing Layer)
    logger.info("\n--- 3. News Sentiment (Alpaca/yfinance Feed) ---")
    news_engine = NewsSentimentEngine()
    sentiment = news_engine.get_sentiment("AAPL")
    
    if sentiment and sentiment['headlines']:
        logger.info(f"✅ Captured {len(sentiment['headlines'])} headlines for AAPL.")
        logger.info(f"✅ Sentiment Score: {sentiment['score']:.4f} ({sentiment['verdict']})")
        logger.info(f"Latest Headline: {sentiment['headlines'][0][:70]}...")
    else:
        logger.error("❌ FAILED to pull news headlines.")

    logger.info("\n=== ✅ DUE-DILIGENCE COMPLETE ===")

if __name__ == "__main__":
    run_diagnostics()
