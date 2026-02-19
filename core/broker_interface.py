from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from dataclasses import dataclass

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
    def submit_order(self, order: TradeOrder) -> Dict:
        """
        Submits an order to the broker.
        Returns the order ID and status.
        """
        pass

    @abstractmethod
    def cancel_order(self, order_id: str):
        """
        Cancels an open order.
        """
        pass
