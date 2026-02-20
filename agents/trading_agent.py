from abc import ABC, abstractmethod
from typing import Dict, List, Any
import logging
import asyncio
from core.base_agent import BaseAgent
from core.broker_interface import BrokerInterface, TradeOrder
from data.feature_store import FeatureStore, MarketTick

class TradingStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    Strategies generate signals but do NOT execute them directly.
    """
    def __init__(self, name: str, config: Dict):
        self.name = name
        self.config = config

    @abstractmethod
    def on_tick(self, tick: MarketTick) -> Optional[Dict]:
        """
        Called on every market tick. Returns a signal or None.
        """
        pass

    @abstractmethod
    def on_bar(self, bar: Dict) -> Optional[Dict]:
        """
        Called on every OHLCV bar closure.
        """
        pass

class TradingAgent(BaseAgent):
    """
    Specialized agent for active trading.
    It manages a portfolio of strategies (Long-term, Day, HFT).
    """
    def __init__(self, name: str, config: Dict[str, Any], feature_store: FeatureStore, broker: BrokerInterface):
        super().__init__(name, config)
        self.feature_store = feature_store
        self.broker = broker
        self.logger = logging.getLogger(__name__)
        self.strategies: List[TradingStrategy] = []
        self.active_orders = []

    def add_strategy(self, strategy: TradingStrategy):
        self.strategies.append(strategy)
        self.logger.info(f"Strategy added to {self.name}: {strategy.name}")

    def update_market_state(self, market_data: Dict[str, Any]):
        """
        Receives market data from Orchestrator or Feed.
        Passes data to all active strategies.
        """
        # Example assumes market_data contains 'tick' or 'bar'
        if 'tick' in market_data:
            tick = market_data['tick']
            # Async processing of strategies could happen here
            for strategy in self.strategies:
                signal = strategy.on_tick(tick)
                if signal:
                    self._process_signal(signal, strategy.name)
        
        # Also update broker with latest prices if it's a simulation
        if hasattr(self.broker, 'update_market_prices') and 'tick' in market_data:
             self.broker.update_market_prices({market_data['tick'].symbol: market_data['tick'].price})

    def _process_signal(self, signal: Dict, strategy_name: str):
        """
        Internal logic to convert a raw signal into a formal proposal.
        """
        self.logger.info(f"Signal received from {strategy_name}: {signal}")
        # Logic to validate signal against local constraints would go here
        
        # Create a proposal (which might be auto-executed if Orchestrator allows)
        # For now, we'll try to execute it directly as if approved
        self.execute_instruction(signal)

    def generate_proposals(self) -> List[Dict]:
        """
        Collects high-conviction signals and converts them to Orchestrator proposals.
        """
        # Placeholder: returning an empty list for now until strategy logic is implemented
        return []

    def execute_instruction(self, instruction: Dict[str, Any]):
        """
        Executes a trade approved by the Orchestrator.
        """
        self.logger.info(f"TradingAgent executing: {instruction}")
        
        try:
            order = TradeOrder(
                symbol=instruction.get('symbol'),
                qty=instruction.get('quantity', 0),
                side=instruction.get('action'),
                order_type=instruction.get('type', 'MARKET'),
                price=instruction.get('price')
            )
            
            # Extract reasoning if present
            reasoning_text = instruction.get('reason', '')
            
            result = self.broker.submit_order(order, reasoning=reasoning_text)
            self.logger.info(f"Order Submitted: {result}")
            
        except Exception as e:
            self.logger.error(f"Failed to execute instruction: {e}")
