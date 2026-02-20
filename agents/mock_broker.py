import logging
import uuid
import asyncio
import random
import csv
import os
from datetime import datetime
from typing import Dict, List, Optional
from core.broker_interface import BrokerInterface, TradeOrder, Position

class MockBroker(BrokerInterface):
    """
    In-memory broker simulation for testing and paper trading.
    """
    def __init__(self, initial_cash: float = 100000.0):
        self.logger = logging.getLogger(__name__)
        self.cash = initial_cash
        self.positions: Dict[str, Position] = {} # Symbol -> Position
        self.orders = {}
        self.latency_ms = 50 # Simulate network latency
        
        # Trade Log
        self.trade_log_path = "data/trades.csv"
        os.makedirs("data", exist_ok=True)
        if not os.path.exists(self.trade_log_path):
            with open(self.trade_log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "symbol", "side", "qty", "price", "cost", "reasoning"])

    def get_account_summary(self) -> Dict:
        equity = self.cash + sum(p.market_value for p in self.positions.values())
        return {
            "cash": self.cash,
            "equity": equity,
            "buying_power": self.cash * 2, # 2x leverage
            "currency": "USD"
        }

    def get_positions(self) -> List[Position]:
        return list(self.positions.values())

    def submit_order(self, order: TradeOrder, reasoning: str = "") -> Dict:
        """
        Simulates order submission and immediate execution (for market orders).
        """
        order_id = str(uuid.uuid4())
        self.orders[order_id] = order
        self.logger.info(f"MockBroker: Received order {order.side} {order.qty} {order.symbol}")
        
        # Simulate Fill (Market Order assumption)
        if order.order_type == 'MARKET':
             fill_price = order.price if order.price else 100.0 
             self._execute_trade(order, fill_price, reasoning)
             
        return {"order_id": order_id, "status": "FILLED"}

    def cancel_order(self, order_id: str):
        if order_id in self.orders:
            self.logger.info(f"MockBroker: Order {order_id} cancelled.")
            del self.orders[order_id]

    def _execute_trade(self, order: TradeOrder, price: float, reasoning: str = ""):
        cost = price * order.qty
        
        if order.side == 'BUY':
            if self.cash >= cost:
                self.cash -= cost
                self._update_position(order.symbol, order.qty, price)
            else:
                self.logger.warning("MockBroker: Insufficient funds")
                return
        elif order.side == 'SELL':
            self.cash += cost
            self._update_position(order.symbol, -order.qty, price)
            
        self.logger.info(f"MockBroker: Filled {order.side} {order.qty} @ {price}")
        
        # Log to CSV with reasoning
        try:
            with open(self.trade_log_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([datetime.now(), order.symbol, order.side, order.qty, price, cost, reasoning])
        except Exception as e:
            self.logger.error(f"Failed to log trade: {e}")

    def _update_position(self, symbol: str, quantity: str, price: float):
        if symbol not in self.positions:
            self.positions[symbol] = Position(
                symbol=symbol, qty=0, avg_entry_price=0, 
                current_price=price, market_value=0, unrealized_pl=0
            )
        
        pos = self.positions[symbol]
        
        # Weighted Average Price Logic (Simplified)
        if quantity > 0: # Adding to position
            total_value = (pos.qty * pos.avg_entry_price) + (quantity * price)
            new_qty = pos.qty + quantity
            if new_qty != 0:
                pos.avg_entry_price = total_value / new_qty
            pos.qty = new_qty
        else: # Reducing position
            pos.qty += quantity
            
        # Remove if closed
        if abs(pos.qty) < 0.0001:
            del self.positions[symbol]

    def update_market_prices(self, price_map: Dict[str, float]):
        """
        Helper to update PnL based on latest prices.
        """
        for symbol, price in price_map.items():
            if symbol in self.positions:
                pos = self.positions[symbol]
                pos.current_price = price
                pos.market_value = pos.qty * price
                pos.unrealized_pl = (price - pos.avg_entry_price) * pos.qty
