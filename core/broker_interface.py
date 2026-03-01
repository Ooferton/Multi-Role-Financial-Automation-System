from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime

@dataclass
class TradeOrder:
    symbol: str
    qty: float
    side: str # 'BUY' or 'SELL'
    order_type: str # 'MARKET', 'LIMIT'
    price: Optional[float] = None
    time_in_force: str = 'DAY'

@dataclass
class Position:
    symbol: str
    qty: float
    avg_entry_price: float
    current_price: float
    market_value: float
    unrealized_pl: float

class BrokerInterface(ABC):
    """
    Standard interface for all brokerage connections.
    Enforces a unified way to submit orders and fetch account data.
    """
    
    @abstractmethod
    def get_account_summary(self) -> Dict:
        """
        Returns account equity, cash, buying power, etc.
        """
        pass

    @abstractmethod
    def get_positions(self) -> List[Position]:
        """
        Returns a list of all current open positions.
        """
        pass

    @abstractmethod
    def submit_order(self, order: TradeOrder, reasoning: str = "") -> Dict:
        """
        Submits an order to the broker with optional reasoning for logging.
        Returns the order ID and status.
        """
        pass

    @abstractmethod
    def cancel_order(self, order_id: str):
        """
        Cancels an open order.
        """
        pass

    @abstractmethod
    def liquidate_all(self):
        """
        Immediately closes all open positions and cancels all open orders.
        """
        pass

    @abstractmethod
    def get_latest_prices(self, symbols: List[str]) -> Dict[str, float]:
        """
        Fetches latest prices for multiple symbols in a single request.
        """
        pass

    def get_latest_quotes(self, symbols: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Fetches latest Level 1/1.5 quotes (bid_size, ask_size).
        Optional method, defaults to returning empty data.
        """
        return {s: {'bid_size': 0.0, 'ask_size': 0.0} for s in symbols}

    @abstractmethod
    def get_historical_data(self, symbol: str, start: Optional[datetime] = None, end: Optional[datetime] = None, timeframe: str = '1Min', limit: Optional[int] = None) -> List[Any]:
        """
        Fetches historical OHLCV data.
        """
        pass
