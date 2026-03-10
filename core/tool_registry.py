import logging
import json
from typing import Dict, Any, List, Callable, Optional
from datetime import datetime, timedelta

# Import system modules
from agents.alpaca_broker import AlpacaBroker
from core.broker_interface import TradeOrder
from data.feature_store import FeatureStore
from core.risk_manager import RiskManager
from core.telegram_notifier import TelegramNotifier

logger = logging.getLogger("ToolRegistry")

class ToolRegistry:
    """
    Registry for tools that Sentience Core can call.
    Each tool has a schema for the LLM and a handler function.
    """
    def __init__(self, broker: AlpacaBroker, feature_store: FeatureStore, risk_manager: RiskManager, notifier: TelegramNotifier):
        self.broker = broker
        self.feature_store = feature_store
        self.risk_manager = risk_manager
        self.notifier = notifier
        self.tools: Dict[str, Dict[str, Any]] = {}
        self._register_default_tools()

    def _register_default_tools(self):
        """Registers the 14 defined tools from the implementation plan."""
        
        # 1. execute_trade
        self.register_tool(
            name="execute_trade",
            description="Place a buy or sell order via Alpaca.",
            parameters={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "side": {"type": "string", "enum": ["BUY", "SELL"]},
                    "qty": {"type": "number"},
                    "reason": {"type": "string"}
                },
                "required": ["symbol", "side", "qty", "reason"]
            },
            handler=self._handle_execute_trade
        )

        # 2. read_portfolio
        self.register_tool(
            name="read_portfolio",
            description="Get current account equity and positions.",
            parameters={"type": "object", "properties": {}},
            handler=self._handle_read_portfolio
        )

        # 3. adjust_risk
        self.register_tool(
            name="adjust_risk",
            description="Modify system risk constraints.",
            parameters={
                "type": "object",
                "properties": {
                    "max_leverage": {"type": "number"},
                    "max_daily_loss": {"type": "number"}
                }
            },
            handler=self._handle_adjust_risk
        )
        
        # 4. send_notification
        self.register_tool(
            name="send_notification",
            description="Send a message to the user via Telegram.",
            parameters={
                "type": "object",
                "properties": {
                    "message": {"type": "string"}
                },
                "required": ["message"]
            },
            handler=self._handle_send_notification
        )

        # 5. read_market_data
        self.register_tool(
            name="read_market_data",
            description="Get historical OHLCV data for a symbol.",
            parameters={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "days": {"type": "integer", "default": 1}
                },
                "required": ["symbol"]
            },
            handler=self._handle_read_market_data
        )

    def register_tool(self, name: str, description: str, parameters: Dict, handler: Callable):
        self.tools[name] = {
            "name": name,
            "description": description,
            "parameters": parameters,
            "handler": handler
        }
        logger.info(f"Tool registered: {name}")

    def get_tool_schemas(self) -> List[Dict]:
        return [
            {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["parameters"]
            }
            for t in self.tools.values()
        ]

    def call_tool(self, name: str, args: Dict) -> Any:
        if name not in self.tools:
            return f"Error: Tool {name} not found."
        
        try:
            logger.info(f"Calling tool {name} with args: {args}")
            return self.tools[name]["handler"](**args)
        except Exception as e:
            logger.error(f"Error executing tool {name}: {e}")
            return f"Error: {str(e)}"

    # --- Handlers ---
    
    def _handle_execute_trade(self, symbol: str, side: str, qty: float, reason: str):
        order = TradeOrder(symbol=symbol, side=side, quantity=qty)
        result = self.broker.submit_order(order, reasoning=reason)
        return {"status": "success" if result else "failed", "order_id": str(result)}

    def _handle_read_portfolio(self):
        summary = self.broker.get_account_summary()
        positions = self.broker.get_positions()
        return {
            "equity": summary.get("equity", 0),
            "buying_power": summary.get("buying_power", 0),
            "positions": [vars(p) for p in positions]
        }

    def _handle_adjust_risk(self, max_leverage: Optional[float] = None, max_daily_loss: Optional[float] = None):
        if max_leverage is not None:
            self.risk_manager.max_leverage = max_leverage
        if max_daily_loss is not None:
            self.risk_manager.max_daily_loss = max_daily_loss
        return {"status": "updated", "leverage": self.risk_manager.max_leverage, "daily_limit": self.risk_manager.max_daily_loss}

    def _handle_send_notification(self, message: str):
        self.notifier._send_message(message)
        return {"status": "sent"}

    def _handle_read_market_data(self, symbol: str, days: int = 1):
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        df = self.feature_store.get_ohlcv(symbol, start_date, end_date)
        if df.empty:
            return {"error": f"No data found for {symbol}"}
        return df.tail(10).to_dict(orient="records") # Return last 10 points for context
