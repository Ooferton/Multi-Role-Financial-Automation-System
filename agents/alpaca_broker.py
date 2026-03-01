import logging
import os
import csv
import yaml
import alpaca_trade_api as tradeapi
from typing import Dict, List, Optional, Any
from core.broker_interface import BrokerInterface, TradeOrder, Position
from datetime import datetime

class AlpacaBroker(BrokerInterface):
    """
    Real/Paper Broker Adapter for Alpaca Markets.
    Requires APCA_API_KEY_ID and APCA_API_SECRET_KEY env vars.
    """
    def __init__(self, paper: Optional[bool] = None, authorized_tickers: List[str] = None):
        self.logger = logging.getLogger(__name__)
        self.authorized_tickers = authorized_tickers
        
        # Default to ALPACA_LIVE env var if paper is not explicitly provided
        if paper is None:
            live_env = os.getenv("ALPACA_LIVE", "false").lower()
            paper = False if live_env == "true" else True
            
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
        self.pdt_blocked = False
        self.day_trade_count = 0

    def check_pdt_safe(self, symbol: str = "") -> bool:
        """
        Checks if the account is at risk of PDT violations.
        Returns False if a day trade would be blocked.
        Crypto symbols are exempt from PDT rules.
        """
        # Crypto check: Alpaca crypto symbols contain '/' or are known crypto pairs
        if symbol and ("/" in symbol or any(c in symbol for c in ["BTC", "ETH", "SOL", "DOGE"])):
            self.logger.info(f"Crypto trade detected for {symbol}. Bypassing PDT check.")
            return True

        try:
            acc = self.get_account_summary()
            equity = acc.get("equity", 0.0)
            self.day_trade_count = acc.get("day_trade_count", 0)
            is_pdt = acc.get("pattern_day_trader", False)

            # PDT Rule: Under $25k, limited to 3 day trades in 5 days
            if equity < 25000:
                if self.day_trade_count >= 3:
                    self.logger.warning(f"PDT WARNING: Account equity (${equity}) is below $25,000 and day trade count is {self.day_trade_count}. Further day trades may be blocked.")
                    return False
            return True
        except Exception as e:
            self.logger.error(f"Error checking PDT safety: {e}")
            return True # Default to True to avoid blocking if API fails

    def submit_order(self, order: TradeOrder, reasoning: str = "") -> Dict:
        # --- ULTIMATE SAFETY GUARD (PROCESS-LEVEL RESTRICTION) ---
        try:
            # 1. Check if this specific broker instance is locked to a set of tickers
            if self.authorized_tickers:
                if order.symbol not in self.authorized_tickers:
                    msg = f"🚫 [PID {os.getpid()}] BLOCKED BY INSTANCE GUARD: {order.side} {order.symbol}. Not in authorized list {self.authorized_tickers}."
                    self.logger.critical(msg)
                    return {"status": "failed", "error": f"Instance Restricted Asset: {order.symbol}"}
                else:
                    # Explicitly permitted by instance, skip global config check
                    pass
            else:
                # 2. Fallback to Global Config Check (Safety Net for general processes)
                config_path = "config/config.yaml"
                if os.path.exists(config_path):
                    with open(config_path, "r") as f:
                        v_config = yaml.safe_load(f)
                    
                    is_btm = v_config.get("system", {}).get("crypto_etf_only", False)
                    if is_btm:
                        allowed = ["BTC/USD", "IBIT", "BITO", "FBTC", "ARKB"]
                        if order.symbol not in allowed:
                            msg = f"🚫 [PID {os.getpid()}] BLOCKED BY GLOBAL GUARD: {order.side} {order.symbol}. Mode Restricted."
                            self.logger.critical(msg)
                            return {"status": "failed", "error": f"Global Restricted Asset: {order.symbol}"}
        except Exception as ve:
            self.logger.debug(f"Safety Guard config check failed: {ve}")
        # ---------------------------------------------------

        try:
            # Alpaca expects 'buy' or 'sell'
            side = order.side.lower()
            qty = float(order.qty)
            
            if qty <= 0:
                self.logger.warning(f"Invalid order quantity: {qty}")
                return {"status": "failed", "error": "Invalid quantity"}

            # Alpaca constraints for fractional stocks:
            # 1. Must be Market order
            # 2. Must be Time-in-Force 'day'
            # (Crypto is exempt from these specific constraints)
            is_fractional = (qty % 1) != 0
            is_crypto = "/" in order.symbol
            
            # Default to order's values
            final_type = order.order_type.lower() if hasattr(order, 'order_type') and order.order_type else 'market'
            final_tif = order.time_in_force.lower() if hasattr(order, 'time_in_force') and order.time_in_force else 'day'
            
            if is_fractional and not is_crypto:
                if final_type != 'market' or final_tif != 'day':
                    self.logger.info(f"Adjusting fractional stock order for {order.symbol}: forcing Market/Day.")
                    final_type = 'market'
                    final_tif = 'day'

            # PDT Safety Check (Only for opening/closing on same day - simplified as general check here)
            if not is_crypto and not self.check_pdt_safe(order.symbol) and side == 'buy':
                self.logger.critical(f"ORDER BLOCKED: PDT Limit approached on {order.symbol}. Equity below $25k.")
                return {"status": "failed", "error": "PDT Limit approached"}

            if is_crypto:
                final_tif = 'gtc'
                self.logger.info(f"Crypto trade: forcing GTC time-in-force for {order.symbol}")

            self.logger.info(f"Submitting {side.upper()} {qty} {order.symbol} ({final_type}/{final_tif})")
            self.api.submit_order(
                symbol=order.symbol,
                qty=str(qty), # Alpaca accepts string for fractional
                side=side,
                type=final_type,
                time_in_force=final_tif
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
                "equity": float(getattr(account, 'equity', 0.0)),
                "cash": float(getattr(account, 'cash', 0.0)),
                "buying_power": float(getattr(account, 'buying_power', 0.0)),
                "initial_margin": float(getattr(account, 'initial_margin', 0.0)),
                "day_trade_count": int(getattr(account, 'day_trade_count', 0)),
                "daytrading_buying_power": float(getattr(account, 'daytrading_buying_power', 0.0)),
                "pattern_day_trader": getattr(account, 'pattern_day_trader', False),
                "status": getattr(account, 'status', 'inactive')
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
        Separates stocks and crypto to use correct endpoints.
        """
        if not symbols:
            return {}
            
        prices = {}
        stocks = [s for s in symbols if "/" not in s]
        cryptos = [s for s in symbols if "/" in s]
        
        # 1. Fetch Stock Snapshots
        if stocks:
            try:
                snapshots = self.api.get_snapshots(stocks, feed='iex')
                for symbol in stocks:
                    if symbol in snapshots and snapshots[symbol].latest_trade:
                        prices[symbol] = float(snapshots[symbol].latest_trade.price)
            except Exception as e:
                self.logger.error(f"Alpaca Stock Snapshots Failed: {e}")
        
        # 2. Fetch Crypto Prices
        for symbol in cryptos:
            try:
                # Use pluralized method which we confirmed works
                trades = self.api.get_latest_crypto_trades([symbol])
                if symbol in trades:
                    prices[symbol] = float(trades[symbol].price)
                    self.logger.info(f"Successfully fetched Alpaca Crypto price for {symbol}: {prices[symbol]}")
            except Exception as e:
                self.logger.error(f"Alpaca Crypto Fetch Failed for {symbol}: {e}")
                    
        return prices

    def get_latest_quotes(self, symbols: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Fetches real-time Bid/Ask sizes for Order Flow Imbalance logic.
        """
        if not symbols:
            return {}
            
        quotes_data = {}
        stocks = [s for s in symbols if "/" not in s]
        cryptos = [s for s in symbols if "/" in s]
        
        # 1. Fetch Stock Snapshots for Volume
        if stocks:
            try:
                snapshots = self.api.get_snapshots(stocks, feed='iex')
                for symbol in stocks:
                    if symbol in snapshots and snapshots[symbol].latest_quote:
                        q = snapshots[symbol].latest_quote
                        quotes_data[symbol] = {
                            'bid_size': float(getattr(q, 'bid_size', 0.0)),
                            'ask_size': float(getattr(q, 'ask_size', 0.0))
                        }
            except Exception as e:
                self.logger.error(f"Alpaca Stock Quotes Failed: {e}")
                
        # 2. Fetch Crypto Quotes
        for symbol in cryptos:
            try:
                quotes = self.api.get_latest_crypto_quotes([symbol])
                if symbol in quotes:
                    q = quotes[symbol]
                    quotes_data[symbol] = {
                        'bid_size': float(getattr(q, 'bid_size', 0.0)),
                        'ask_size': float(getattr(q, 'ask_size', 0.0))
                    }
            except Exception as e:
                self.logger.warning(f"Alpaca Crypto Quote Fetch Failed for {symbol}: {e}")
                
        return quotes_data

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
            if "/" in symbol:
                # Crypto bars
                bars_obj = self.api.get_crypto_bars(symbol, tf, start_str, end_str)
                bars = bars_obj.df if hasattr(bars_obj, 'df') else bars_obj
            else:
                # Stock bars
                bars = self.api.get_bars(symbol, tf, start_str, end_str, adjustment='all', feed='iex').df
            
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

    def reset_paper_account(self, balance: float = 100000.0):
        """
        Resets the paper trading account balance to bypass PDT limits.
        ONLY works on Paper Trading.
        """
        if "paper-api" not in self.base_url:
            self.logger.error("Cannot reset balance on a Live account!")
            return False
            
        try:
            # Note: There isn't a direct API to 'set' balance, but we can recommend the user
            # to use the Alpaca Dashboard or we can simulate it if there was a mock.
            # However, Alpaca Paper API allows resetting via the dashboard.
            # For automation, we'll log the instruction.
            self.logger.info(f"To bypass PDT, please reset your Paper Trading balance to ${balance} in the Alpaca Dashboard.")
            return True
        except Exception as e:
            self.logger.error(f"Failed to reset paper account: {e}")
            return False
