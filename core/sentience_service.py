import time
import logging
import json
import os
from datetime import datetime
from core.agent_router import AgentRouter
from core.llm_supervisor import LLMSupervisor
from core.tool_registry import ToolRegistry
from core.agent_memory import AgentMemory
from agents.alpaca_broker import AlpacaBroker
from data.feature_store import FeatureStore
from core.risk_manager import RiskManager
from core.telegram_notifier import TelegramNotifier

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [SENTIENCE-SERVICE] - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/sentience_service.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("SentienceService")

class SentienceService:
    """
    Heartbeat of the Sentience Core.
    Acts as the 24/7 Autonomous Overlord of all system modules.
    """
    def __init__(self, interval_seconds: int = 900): # 15 minutes
        self.interval = interval_seconds
        
        # Initialize Dependencies
        self.broker = AlpacaBroker()
        self.fs = FeatureStore()
        self.rm = RiskManager({}) # Will load from config inside
        self.tn = TelegramNotifier()
        
        self.llm = LLMSupervisor()
        self.registry = ToolRegistry(self.broker, self.fs, self.rm, self.tn)
        self.memory = AgentMemory()
        self.agent = AgentRouter(self.llm, self.registry)
        
        self.status_file = "data/sentience_status.json"

    def run(self):
        logger.info("Sentience Service (Layer 4) online. Starting Heartbeat Loop.")
        
        while True:
            try:
                self.heartbeat()
            except Exception as e:
                logger.error(f"Heartbeat Loop Error: {e}")
            
            time.sleep(self.interval)

    def heartbeat(self):
        """Perform a single autonomous pulse check."""
        logger.info("💓 Pulse check starting...")
        
        # 1. Gather System State
        state = self._gather_system_state()
        
        # 2. Strategic Autonomous Review
        prompt = f"""SYSTEM AUDIT REQUEST. Analyze the current state and take ANY necessary actions to optimize the system.
If everything is within limits and optimal, simply report 'System Optimal'.

Current State:
{json.dumps(state, indent=2)}
"""
        logger.info("Requesting autonomous strategy review from Sentience Core...")
        response = self.agent.chat(prompt, context=state)
        
        # 3. Update Status
        self._update_status(state, response)
        logger.info(f"Pulse check complete. Response: {response[:100]}...")

    def _gather_system_state(self) -> dict:
        summary = self.broker.get_account_summary()
        positions = self.broker.get_positions()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "equity": summary.get("equity"),
            "buying_power": summary.get("buying_power"),
            "daily_pnl": summary.get("daily_pnl"),
            "position_count": len(positions),
            "is_halted": self.rm.is_halted,
            "max_leverage": self.rm.max_leverage,
        }

    def _update_status(self, state: dict, last_response: str):
        status = {
            "last_heartbeat": datetime.now().isoformat(),
            "state": state,
            "ai_monologue": last_response
        }
        with open(self.status_file, "w") as f:
            json.dump(status, f, indent=2)

if __name__ == "__main__":
    service = SentienceService()
    service.run()
