import time
import os
import logging
import argparse
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

from agents.trading_agent import TradingAgent
from agents.alpaca_broker import AlpacaBroker
from agents.mock_broker import MockBroker
from core.orchestrator import Orchestrator
from data.feature_store import FeatureStore, MarketTick

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("live_trading.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("LiveRunner")

def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Live runner for trading system")
    parser.add_argument('--confirm-live', action='store_true', help='Enable live Alpaca broker (unsafe)')
    parser.add_argument('--symbols', default='SPY', help='Comma-separated target trading symbols')
    parser.add_argument('--model', default='ml/models/ppo_trading_real_v2', help='Model path prefix')
    parser.add_argument('--order-throttle', type=float, default=None, help='Seconds between Alpaca order submissions (overrides ALPACA_ORDER_THROTTLE)')
    parser.add_argument('--min-trade-value', type=float, default=None, help='Minimum trade value in USD to allow orders (overrides ALPACA_MIN_COST_BASIS)')
    parser.add_argument('--strategy', default='rl_v2', choices=['rl_v2', 'day'], help='Which strategy to enable')
    args = parser.parse_args()

    # 1. Initialize Components
    logger.info("Initializing Live Trading System...")

    # Optionally override env knobs before broker construction
    if args.order_throttle is not None:
        os.environ['ALPACA_ORDER_THROTTLE'] = str(args.order_throttle)
        logger.info(f"Setting ALPACA_ORDER_THROTTLE={os.environ['ALPACA_ORDER_THROTTLE']}")
    if args.min_trade_value is not None:
        os.environ['ALPACA_MIN_COST_BASIS'] = str(args.min_trade_value)
        logger.info(f"Setting ALPACA_MIN_COST_BASIS={os.environ['ALPACA_MIN_COST_BASIS']}")

    # Broker selection: default to MockBroker unless explicitly confirmed
    if args.confirm_live:
        logger.warning('confirm-live flag present: using AlpacaBroker (paper mode if configured)')
        broker = AlpacaBroker(paper=True)
    else:
        logger.info('Dry-run mode: using MockBroker. Pass --confirm-live to enable AlpacaBroker.')
        broker = MockBroker()
    
    # Check connection
    summary = broker.get_account_summary()
    logger.info(f"Connected to Alpaca. Equity: ${summary.get('equity', 0)}")
    
    # Feature Store (for logging history if needed)
    feature_store = FeatureStore()
    
    # Orchestrator with broker
    orchestrator = Orchestrator("config/config.yaml")
    orchestrator.set_broker(broker)
    
    # Agent
    agent = TradingAgent("Live_RL_Agent_v2", {}, feature_store, broker)
    
    # Strategy — PPO Trading Real v2 (10 indicators)
    model_path = args.model
    
    # Wait for model if not ready
    while not os.path.exists(model_path + ".zip") and not os.path.exists(model_path):
        logger.info("Waiting for model file...")
        time.sleep(10)

    # Instantiate chosen strategy
    if args.strategy == 'day':
        from strategies.day_trading import DayTradingStrategy
        strategy = DayTradingStrategy("Day_Trading", {})
    else:
        from strategies.rl_strategy_v2 import RLStrategyV2
        strategy_config = {}
        if args.min_trade_value is not None:
            strategy_config['min_trade_value'] = args.min_trade_value
        strategy = RLStrategyV2("RL_Brain_v2", strategy_config, broker, model_path.replace(".zip",""))

    agent.add_strategy(strategy)
    
    # Support multiple symbols
    symbols = [s.strip().upper() for s in args.symbols.split(',') if s.strip()]
    if not symbols:
        symbols = ['SPY']
    logger.info(f"Starting High-Frequency Loop for {', '.join(symbols)}...")
    
    try:
        # If using live Alpaca broker, prefer the broker's streaming feed
        if args.confirm_live and hasattr(broker, 'start_price_stream'):
            import queue as _queue

            tick_queue: _queue.Queue = _queue.Queue()

            def _on_price(sym: str, price: float):
                tick = MarketTick(symbol=sym, timestamp=datetime.now(), price=price, size=100, exchange='ALPACA')
                try:
                    tick_queue.put_nowait(tick)
                except Exception:
                    pass

            broker.start_price_stream(symbols, _on_price, use_ws=True)

            # Process ticks from stream
            while True:
                try:
                    tick = tick_queue.get(timeout=5)
                    agent.update_market_state({'tick': tick})
                except _queue.Empty:
                    logger.debug('No tick received in timeout window')

        else:
            # Polling loop (mock or fallback)
            while True:
                for sym in symbols:
                    if hasattr(broker, 'get_current_price'):
                        price = broker.get_current_price(sym)
                    else:
                        import random
                        base = float(os.environ.get('MOCK_PRICE', 100.0))
                        price = base + random.uniform(-1.0, 1.0)

                    if price > 0:
                        tick = MarketTick(symbol=sym, timestamp=datetime.now(), price=price, size=100, exchange='ALPACA')
                        agent.update_market_state({'tick': tick})
                    else:
                        logger.warning(f'Failed to fetch price for {sym}.')
                # Small sleep to allow rapid multi-symbol looping
                time.sleep(0.2)

    except KeyboardInterrupt:
        logger.info("Live Trading Stopped by User.")
    except Exception as e:
        logger.exception(f"Crash: {e}")

if __name__ == "__main__":
    main()
