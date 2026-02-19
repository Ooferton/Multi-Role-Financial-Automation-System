from abc import ABC, abstractmethod
from typing import Dict, Any, List

class BaseAgent(ABC):
    """
    Abstract base class for all specialized financial agents.
    Enforces a common interface for the Orchestrator to interact with.
    """
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.is_active = False
        self.capital_allocation = 0.0

    @abstractmethod
    def update_market_state(self, market_data: Dict[str, Any]):
        """
        Ingests new market data and updates internal state/models.
        """
        pass

    @abstractmethod
    def generate_proposals(self) -> List[Dict]:
        """
        Generates potential actions (trades, investments, etc.) for the Orchestrator to review.
        Returns a list of proposal dictionaries.
        """
        pass

    @abstractmethod
    def execute_instruction(self, instruction: Dict[str, Any]):
        """
        Executes a specific instruction approved/issued by the Orchestrator.
        """
        pass

    def set_budget(self, amount: float):
        """
        Updates the capital available to this agent.
        """
        self.capital_allocation = amount
        # Log or trigger internal rebalancing if needed

    def get_status(self) -> Dict[str, Any]:
        """
        Returns the health and status of the agent.
        """
        return {
            "name": self.name,
            "active": self.is_active,
            "capital": self.capital_allocation
        }
