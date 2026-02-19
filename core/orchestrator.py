import logging
import yaml
from typing import Dict, List
from core.risk_manager import RiskManager
from core.base_agent import BaseAgent
from core.global_optimizer import SimpleRuleBasedOptimizer

class Orchestrator:
    """
    The Central Nervous System of the Financial Platform.
    Responsible for allocating capital, managing global state, and coordinating agents.
    """
    def __init__(self, config_path: str):
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path)
        self.risk_manager = RiskManager(self.config)
        self.optimizer = SimpleRuleBasedOptimizer() # In future, load dynamically based on config
        
        self.capital_allocation = {}
        self.registered_agents: Dict[str, BaseAgent] = {}

    def _load_config(self, path: str) -> Dict:
        with open(path, 'r') as f:
            return yaml.safe_load(f)

    def register_agent(self, agent_instance: BaseAgent):
        """
        Registers a specialized agent with the orchestrator.
        """
        self.registered_agents[agent_instance.name] = agent_instance
        self.logger.info(f"Agent registered: {agent_instance.name}")

    def allocate_capital(self):
        """
        Determines how much capital each agent is allowed to deploy.
        Uses the GlobalObjectiveOptimizer to solve for optimal allocation.
        """
        self.logger.info("Running capital allocation cycle...")
        
        # 1. Get current market regime (Placeholder)
        current_regime = "NORMAL" # This would come from a tailored Model/Detector
        
        # 2. Get constraints from Risk Manager
        constraints = {
            "max_daily_loss": self.risk_manager.max_daily_loss,
            "max_leverage": self.risk_manager.max_leverage
        }

        # 3. Optimize
        # Note: In a real implementation, 'current_state' would be aggregated from all agents
        current_state = {"net_worth": 100000} 
        
        allocation_targets = self.optimizer.optimize_allocation(
            current_state, 
            current_regime, 
            constraints
        )
        
        self.capital_allocation = allocation_targets
        self.logger.info(f"New Allocation Targets: {self.capital_allocation}")
        
        # 4. Distribute Budgets
        for agent_name, agent in self.registered_agents.items():
            if agent_name in self.capital_allocation:
                # Convert percentage to absolute amount (simplified)
                # In reality, this needs to account for equity, margin, etc.
                total_equity = 100000 # Placeholder
                budget = self.capital_allocation[agent_name] * total_equity
                agent.set_budget(budget)
                self.logger.info(f"Updated budget for {agent_name}: ${budget:.2f}")

    def run_cycle(self):
        """
        Main execution loop.
        """
        if self.risk_manager.is_halted:
            self.logger.warning("Orchestrator skipping cycle: System is HALTED.")
            return

        self.allocate_capital()
        
        # Trigger agents to generate proposals
        all_proposals = []
        for agent in self.registered_agents.values():
            if agent.is_active:
                proposals = agent.generate_proposals()
                all_proposals.extend(proposals)
                
        # Evaluate and Execute proposals (Placeholder)
        self.logger.info(f"Collected {len(all_proposals)} proposals from agents.")

if __name__ == "__main__":
    # Simple test
    logging.basicConfig(level=logging.INFO)
    orch = Orchestrator("config/config.yaml")
    orch.run_cycle()
