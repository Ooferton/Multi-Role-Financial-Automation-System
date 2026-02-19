import time
import logging
import random
from datetime import datetime
from dotenv import load_dotenv

from agents.mock_broker import MockBroker
from agents.trading_agent import TradingAgent, TradingStrategy
from data.feature_store import FeatureStore, MarketTick

# Simple runner that uses MockBroker and a tiny random strategy
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("StartPaperMock")


class RandomStrategy(TradingStrategy):
    def __init__(self, name: str, config: dict = None):
        super().__init__(name, config or {})

    def on_tick(self, tick: MarketTick):
        # 5% chance to emit a buy or sell signal
        r = random.random()
        if r < 0.025:
            return {"action": "BUY", "symbol": tick.symbol, "quantity": 1, "type": "MARKET", "price": tick.price}
        elif r < 0.05:
            return {"action": "SELL", "symbol": tick.symbol, "quantity": 1, "type": "MARKET", "price": tick.price}
        return None

    def on_bar(self, bar: dict):
        return None


def main(ticks: int = 20, symbol: str = "SPY"):
    load_dotenv()

    broker = MockBroker(initial_cash=100000.0)
    feature_store = FeatureStore()
    agent = TradingAgent("DemoAgent", {}, feature_store, broker)

    strategy = RandomStrategy("RandomNoise")
    agent.add_strategy(strategy)

    logger.info("Starting mock paper runner (short demo)...")

    try:
        for i in range(ticks):
            price = 100.0 + random.uniform(-1.0, 1.0)  # synthetic price
            tick = MarketTick(symbol=symbol, timestamp=datetime.now(), price=price, size=100, exchange="MOCK")

            # Persist tick
            feature_store.save_tick(tick)

            # Pass to agent
            agent.update_market_state({"tick": tick})

            # Print account summary occasionally
            if i % 5 == 0:
                logger.info(f"Account: {broker.get_account_summary()}")

            time.sleep(1)

        logger.info("Demo complete.")

    except KeyboardInterrupt:
        logger.info("Stopped by user.")


if __name__ == "__main__":
    main()
