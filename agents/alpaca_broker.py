import logging
import os
import time
import threading
import queue
import alpaca_trade_api as tradeapi
from typing import Dict, List, Optional, Callable
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

        # Order queue and throttling
        self.order_queue: "queue.Queue[Dict]" = queue.Queue()
        self.orders: Dict[str, Dict] = {}
        # Throttle delay between API order submissions (seconds)
        try:
            self.throttle_delay = float(os.getenv('ALPACA_ORDER_THROTTLE', '0.5'))
        except Exception:
            self.throttle_delay = 0.5

        self._order_worker_thread = threading.Thread(target=self._order_worker, daemon=True)
        self._order_worker_thread.start()

        # Price stream thread handle (polling fallback if streaming not available)
        self._price_stream_thread: Optional[threading.Thread] = None
        self._price_stream_stop = threading.Event()

    def submit_order(self, order: TradeOrder, reasoning: str = "") -> Dict:
        """
        Place order onto an internal queue; the background worker will submit
        to Alpaca respecting `throttle_delay`. Returns a queued token dict.
        """
        try:
            side = order.side.lower()
            qty = float(order.qty)

            if qty <= 0:
                self.logger.warning(f"Invalid order quantity: {qty}")
                return {"order_id": None, "status": "REJECTED", "reason": "invalid_qty"}

            # Estimate cost and enforce local minimum
            min_cost_basis = float(os.getenv('ALPACA_MIN_COST_BASIS', '1.0'))
            price_for_cost = order.price if order.price else self.get_current_price(order.symbol)
            try:
                cost_basis = float(qty) * float(price_for_cost)
            except Exception:
                cost_basis = 0.0

            if cost_basis < min_cost_basis:
                self.logger.warning(
                    f"Alpaca Order Rejected: cost basis ${cost_basis:.4f} < minimum ${min_cost_basis:.2f}"
                )
                return {"order_id": None, "status": "REJECTED", "error": "cost_basis_too_small", "cost": cost_basis}

            # Queue the order for the worker
            token = {
                'order': order,
                'reasoning': reasoning,
                'submitted_at': datetime.now().isoformat()
            }
            self.order_queue.put(token)
            qid = id(token)
            self.orders[str(qid)] = {'status': 'QUEUED', 'token': token}
            return {"order_id": str(qid), "status": "QUEUED"}
        except Exception as e:
            self.logger.error(f"Alpaca Queueing Failed: {e}")
            return {"order_id": None, "status": "ERROR", "error": str(e)}

    def get_positions(self) -> List[Position]:
        try:
            alpaca_positions = self.api.list_positions()
            positions = []
            for p in alpaca_positions:
                try:
                    sym = str(p.symbol).strip().upper()
                    qty = float(p.qty) if p.qty is not None else 0.0
                    avg_entry = float(p.avg_entry_price) if p.avg_entry_price not in (None, '') else 0.0
                    curr_price = None
                    try:
                        curr_price = float(p.current_price) if p.current_price not in (None, '') else None
                    except Exception:
                        curr_price = None

                    market_value = float(p.market_value) if getattr(p, 'market_value', None) not in (None, '') else qty * (curr_price or 0.0)
                    unrealized = float(getattr(p, 'unrealized_pl', getattr(p, 'unrealized_pl', 0.0))) if getattr(p, 'unrealized_pl', None) not in (None, '') else 0.0

                    # If current price missing, attempt to fetch latest trade
                    if curr_price is None:
                        try:
                            latest = self.api.get_latest_trade(sym)
                            curr_price = float(latest.price)
                        except Exception:
                            curr_price = 0.0

                    pos = Position(
                        symbol=sym,
                        qty=qty,
                        avg_entry_price=avg_entry,
                        current_price=curr_price,
                        market_value=market_value,
                        unrealized_pl=unrealized
                    )
                    positions.append(pos)
                except Exception as inner_e:
                    self.logger.warning(f"Skipping malformed position from Alpaca: {inner_e}")
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

    def start_price_stream(self, symbols: List[str], callback: Callable[[str, float], None], use_ws: bool = True):
        """
        Start a background price feed. If Alpaca websocket streaming is available
        the method will use it; otherwise it falls back to a polling thread.
        The callback signature is callback(symbol, price).
        """
        # Try multiple Stream constructor signatures and robustly reconnect on error
        if use_ws:
            try:
                from alpaca_trade_api.stream import Stream

                # Try several constructor signatures to be compatible across versions
                stream = None
                try:
                    stream = Stream(self.api_key, self.api_secret, base_url=self.base_url, data_stream='alpacadatav1')
                except TypeError:
                    try:
                        stream = Stream(self.api_key, self.api_secret, base_url=self.base_url)
                    except TypeError:
                        try:
                            stream = Stream(self.api_key, self.api_secret, self.base_url)
                        except Exception as e:
                            raise e

                async def on_trade(data):
                    try:
                        sym = getattr(data, 'symbol', None) or getattr(data, 'S', None)
                        price = float(getattr(data, 'price', getattr(data, 'p', 0.0)))
                        if sym and price:
                            callback(sym, price)
                    except Exception:
                        pass

                # Subscribe to trades; tolerate subscription errors and fallback
                for s in symbols:
                    try:
                        stream.subscribe_trades(on_trade, s)
                    except Exception:
                        try:
                            # older API naming
                            stream.subscribe_trade_updates(on_trade, s)
                        except Exception:
                            self.logger.debug(f"Could not subscribe trades for {s} via WS")

                # Run stream with reconnect/backoff in separate thread
                def _run_stream():
                    backoff = 1.0
                    while not self._price_stream_stop.is_set():
                        try:
                            stream.run()
                        except Exception as e:
                            self.logger.warning(f"Alpaca stream error: {e}. Reconnecting in {backoff}s")
                            time.sleep(backoff)
                            backoff = min(backoff * 2, 60.0)
                        else:
                            backoff = 1.0

                    # If stopped, attempt to gracefully close stream
                    try:
                        stream.close()
                    except Exception:
                        pass

                t = threading.Thread(target=_run_stream, daemon=True)
                t.start()
                self._price_stream_thread = t
                return
            except Exception as e:
                self.logger.info(f"WebSocket stream unavailable or failed to start: {e}. Using polling fallback.")

        # Start polling fallback
        self._start_polling_price_stream(symbols, callback)

    def _start_polling_price_stream(self, symbols: List[str], callback: Callable[[str, float], None], interval: float = 1.0):
        self._price_stream_stop.clear()

        def _poll():
            import random
            while not self._price_stream_stop.is_set():
                for s in symbols:
                    try:
                        price = self.get_current_price(s)
                        if price and price > 0:
                            callback(s, price)
                        else:
                            # occasional simulated jitter fallback
                            base = float(os.environ.get('MOCK_PRICE', 100.0))
                            callback(s, base + random.uniform(-1.0, 1.0))
                    except Exception as e:
                        self.logger.debug(f"Polling price failed for {s}: {e}")
                time.sleep(interval)

        t = threading.Thread(target=_poll, daemon=True)
        t.start()
        self._price_stream_thread = t

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancels an order by ID using Alpaca REST API.
        Returns True if cancelled or False on failure.
        """
        try:
            self.api.cancel_order(order_id)
            self.logger.info(f"Alpaca: Cancelled order {order_id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to cancel order {order_id}: {e}")
            return False

    def _order_worker(self):
        """Background worker that consumes the order queue and submits to Alpaca respecting throttle."""
        while True:
            try:
                token = self.order_queue.get()
                if token is None:
                    break
                order: TradeOrder = token['order']
                reasoning = token.get('reasoning', '')

                # Submit via REST API
                try:
                    side = order.side.lower()
                    qty = float(order.qty)
                    is_fractional = (qty % 1) != 0
                    tif = 'day' if is_fractional else 'gtc'
                    alpaca_order = self.api.submit_order(
                        symbol=order.symbol,
                        qty=str(qty),
                        side=side,
                        type='market',
                        time_in_force=tif
                    )
                    oid = getattr(alpaca_order, 'id', None)
                    self.logger.info(f"Alpaca Order Submitted (worker): {side.upper()} {qty} {order.symbol} (id={oid})")
                    self.orders[str(id(token))] = {'status': 'SUBMITTED', 'order_id': oid}
                except Exception as e:
                    self.logger.error(f"Alpaca worker failed to submit order: {e}")
                    self.orders[str(id(token))] = {'status': 'ERROR', 'error': str(e)}

                # Throttle
                time.sleep(self.throttle_delay)
            except Exception as e:
                self.logger.exception(f"Order worker crash: {e}")
                time.sleep(1)
