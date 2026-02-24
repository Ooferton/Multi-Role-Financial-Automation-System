from abc import ABC, abstractmethod
from typing import Dict, List

class GlobalObjectiveOptimizer(ABC):
    """
    Interface for the strategic optimization engine.
    Responsible for solving the multi-objective problem:
    Maximize(Net Worth Growth) - Penalty(Risk) - Penalty(Liquidity Shortfall)
    """

    @abstractmethod
    def optimize_allocation(self, 
                          current_state: Dict, 
                          market_regime: str, 
                          constraints: Dict) -> Dict[str, float]:
        """
        Calculates the optimal capital allocation for each agent.
        
        Args:
            current_state: Global system state (Net Worth, Liquidity, Agent KPIs)
            market_regime: Current identified market regime (e.g., "HIGH_VOLATILITY", "BULL_TREND")
            constraints: Hard constraints from Risk Manager
            
        Returns:
            Dict mapping agent_names to capital_allocation_amounts
        """
        pass

    @abstractmethod
    def evaluate_proposal(self, proposal: Dict) -> float:
        """
        Scores a specific proposal (trade/investment) against the global objective function.
        Returns a score (higher is better).
        """
        pass

class SimpleRuleBasedOptimizer(GlobalObjectiveOptimizer):
    """
    A simple baseline implementation using heuristics.
    Useful for initial testing before RL deployment.
    """
    def optimize_allocation(self, current_state: Dict, market_regime: str, constraints: Dict) -> Dict[str, float]:
        # Simple heuristic: heavily weight trading in bull markets, cash in high volatility
        if market_regime == "HIGH_VOLATILITY":
            return {
                "trading_agent": 0.2,
                "wealth_agent": 0.2,
                "cash_reserve": 0.6
            }
        else:
            return {
                "trading_agent": 0.5,
                "wealth_agent": 0.4,
                "cash_reserve": 0.1
            }

    def evaluate_proposal(self, proposal: Dict) -> float:
        # Placeholder scoring
        return proposal.get('expected_return', 0) / (proposal.get('risk', 1) + 1e-6)
