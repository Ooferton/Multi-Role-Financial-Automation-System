import logging
import os
import alpaca_trade_api as tradeapi
from typing import Dict, List, Optional
from core.broker_interface import BrokerInterface, TradeOrder, Position
from datetime import datetime

class AlpacaBroker(BrokerInterface):
    """
    Real/Paper Broker Adapter for Alpaca Markets.
    Requires APCA_API_KEY_ID and APCA_API_SECRET_KEY env vars.
    """
    def __init__(self, paper: bool = True):
        self.logger = logging.getLogger(__name__)
        
        self.api_key = os.getenv("APCA_API_KEY_ID")
        self.api_secret = os.getenv("APCA_API_SECRET_KEY")
        self.base_url = "https://paper-api.alpaca.markets" if paper else "https://api.alpaca.markets"
        
        if not self.api_key or not self.api_secret:
            self.logger.warning("Alpaca API Credentials not found in Environment!")
            
        self.api = tradeapi.REST(
            self.api_key,
            self.api_secret,
            self.base_url,
            api_version='v2'
        )
        
        self.logger.info(f"Connected to Alpaca ({'Paper' if paper else 'Live'}).")

    def submit_order(self, order: TradeOrder, **kwargs) -> bool:
        try:
            # Alpaca expects 'buy' or 'sell'
            side = order.side.lower()
            qty = float(order.qty)
            
            if qty <= 0:
                self.logger.warning(f"Invalid order quantity: {qty}")
                return False

            # Alpaca requires time_in_force='day' for fractional orders
            is_fractional = (qty % 1) != 0
            tif = 'day' if is_fractional else 'gtc'

            self.api.submit_order(
                symbol=order.symbol,
                qty=str(qty), # Alpaca accepts string for fractional
                side=side,
                type='market',
                time_in_force=tif
            )
            self.logger.info(f"Alpaca Order Submitted: {side.upper()} {qty} {order.symbol}")
            return True
        except Exception as e:
            self.logger.error(f"Alpaca Order Failed: {e}")
            return False

    def get_positions(self) -> List[Position]:
        try:
            alpaca_positions = self.api.list_positions()
            positions = []
            for p in alpaca_positions:
                pos = Position(
                    symbol=p.symbol,
                    qty=float(p.qty),
                    avg_entry_price=float(p.avg_entry_price),
                    current_price=float(p.current_price),
                    market_value=float(p.market_value),
                    unrealized_pnl=float(p.unrealized_pl)
                )
                positions.append(pos)
            return positions
        except Exception as e:
            self.logger.error(f"Failed to fetch positions: {e}")
            return []

    def get_account_summary(self) -> Dict:
        try:
            account = self.api.get_account()
            return {
                "equity": float(account.equity),
                "cash": float(account.cash),
                "buying_power": float(account.buying_power),
                "initial_margin": float(account.initial_margin),
                "status": account.status
            }
        except Exception as e:
            self.logger.error(f"Failed to fetch account summary: {e}")
            return {"equity": 0.0, "cash": 0.0}

    def get_current_price(self, symbol: str) -> float:
        """
        Fetches real-time price from Alpaca.
        """
        try:
            trade = self.api.get_latest_trade(symbol)
            return float(trade.price)
        except Exception as e:
            self.logger.error(f"Failed to fetch price for {symbol}: {e}")
            return 0.0

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancels an open order.
        """
        try:
            self.api.cancel_order(order_id)
            self.logger.info(f"Order {order_id} cancelled.")
            return True
        except Exception as e:
            self.logger.error(f"Failed to cancel order {order_id}: {e}")
            return False
