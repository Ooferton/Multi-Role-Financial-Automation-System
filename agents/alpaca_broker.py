import logging
import os
import csv
import alpaca_trade_api as tradeapi
from typing import Dict, List, Optional, Any
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
            self.logger.error("Alpaca API Credentials (APCA_API_KEY_ID, APCA_API_SECRET_KEY) not found in Environment!")
            
        self.api = tradeapi.REST(
            self.api_key,
            self.api_secret,
            self.base_url,
            api_version='v2'
        )
        
        self.logger.info(f"Connected to Alpaca ({'Paper' if paper else 'Live'}).")

    def submit_order(self, order: TradeOrder, reasoning: str = "") -> Dict:
        try:
            # Alpaca expects 'buy' or 'sell'
            side = order.side.lower()
            qty = float(order.qty)
            
            if qty <= 0:
                self.logger.warning(f"Invalid order quantity: {qty}")
                return {"status": "failed", "error": "Invalid quantity"}

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

            # LOG TO CSV for Dashboard Execution Log
            try:
                trades_path = "data/trades.csv"
                os.makedirs("data", exist_ok=True)
                file_exists = os.path.exists(trades_path)
                
                with open(trades_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    if not file_exists:
                        writer.writerow(["timestamp", "symbol", "side", "qty", "price", "cost", "reasoning"])
                    
                    # For market orders, use the price passed in the instruction if available
                    price = order.price if order.price else 0.0
                    cost = price * qty
                    writer.writerow([datetime.now(), order.symbol, side.upper(), qty, price, cost, reasoning])
            except Exception as e:
                self.logger.error(f"Failed to log Alpaca trade to CSV: {e}")

            return {"status": "submitted", "symbol": order.symbol, "side": side}
        except Exception as e:
            self.logger.error(f"Alpaca Order Failed: {e}")
            return {"status": "failed", "error": str(e)}

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
                    unrealized_pl=float(p.unrealized_pl)
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

    def liquidate_all(self):
        """
        Emergency liquidation of all positions and orders.
        """
        try:
            self.logger.warning("[LIQUIDATION] EMERGENCY LIQUIDATION TRIGGERED!")
            # Close all positions
            self.api.close_all_positions()
            # Cancel all orders
            self.api.cancel_all_orders()
            self.logger.info("[LIQUIDATION] All Alpaca positions closed and orders cancelled.")
            return True
        except Exception as e:
            self.logger.error(f"[LIQUIDATION] Failed to liquidate all: {e}")
            return False

    def get_latest_prices(self, symbols: List[str]) -> Dict[str, float]:
        """
        Efficient bulk price fetching via Snapshots.
        Reduces API overhead and prevents rate-limiting.
        """
        if not symbols:
            return {}
        try:
            # get_snapshots takes a list and returns a dict-like object
            snapshots = self.api.get_snapshots(symbols)
            prices = {}
            for symbol in symbols:
                if symbol in snapshots:
                    prices[symbol] = float(snapshots[symbol].latest_trade.price)
            return prices
        except Exception as e:
            self.logger.error(f"Alpaca Snapshots Failed: {e}")
            return {}

    def get_historical_data(self, symbol: str, start: Optional[datetime] = None, end: Optional[datetime] = None, timeframe: str = '1Min', limit: Optional[int] = None) -> List[Any]:
        """
        Fetches historical OHLCV data from Alpaca.
        """
        try:
            from data.feature_store import OHLCV
            # timeframe map
            tf_map = {
                '1Min': tradeapi.rest.TimeFrame.Minute,
                '5Min': tradeapi.rest.TimeFrame(5, tradeapi.rest.TimeFrameUnit.Minute),
                '1Hour': tradeapi.rest.TimeFrame.Hour,
                '1Day': tradeapi.rest.TimeFrame.Day
            }
            tf = tf_map.get(timeframe, tradeapi.rest.TimeFrame.Minute)
            
            # Use RFC3339 if provided, otherwise None
            start_str = start.strftime('%Y-%m-%dT%H:%M:%SZ') if start else None
            end_str = end.strftime('%Y-%m-%dT%H:%M:%SZ') if end else None
            
            # Fetch the window. We apply the limit client-side to ensure we get the 'tail' (most recent bars).
            bars = self.api.get_bars(symbol, tf, start_str, end_str, adjustment='all').df
            
            if limit and not bars.empty:
                bars = bars.tail(limit)
            
            ohlcv_list = []
            if not bars.empty:
                for ts, row in bars.iterrows():
                    ohlcv_list.append(OHLCV(
                        symbol=symbol,
                        timestamp=ts.to_pydatetime(),
                        open=float(row['open']),
                        high=float(row['high']),
                        low=float(row['low']),
                        close=float(row['close']),
                        volume=float(row['volume'])
                    ))
            return ohlcv_list
        except Exception as e:
            self.logger.error(f"Failed to fetch historical data for {symbol}: {e}")
            return []
