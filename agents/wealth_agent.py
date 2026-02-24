import numpy as np
import logging
from typing import Dict, List, Any
from core.base_agent import BaseAgent
from ml.reasoning_engine import ReasoningEngine

class WealthStrategyAgent(BaseAgent):
    """
    Manages long-term wealth preservation and growth.
    
    Responsibilities:
    1. Monte Carlo retirement simulations.
    2. Tax-loss harvesting opportunities.
    3. Asset allocation rebalancing suggestions.
    
    All decisions include human-readable reasoning.
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.simulation_runs = 1000
        self.years_to_simulate = 30
        self.reasoner = ReasoningEngine()
        self.logger = logging.getLogger(__name__)
        
        # Target allocation
        self.target_allocation = config.get('target_allocation', {
            'stocks': 0.60,
            'bonds': 0.30,
            'cash': 0.10
        })

    def update_market_state(self, market_data: Dict[str, Any]):
        pass

    def run_monte_carlo(self, current_portfolio_value: float, annual_contribution: float) -> Dict:
        """
        Runs a Monte Carlo simulation to project future net worth.
        """
        mu = 0.07
        sigma = 0.15
        
        results = []
        for _ in range(self.simulation_runs):
            value = current_portfolio_value
            for _ in range(self.years_to_simulate):
                returns = np.random.normal(mu, sigma)
                value = value * (1 + returns) + annual_contribution
            results.append(value)
            
        avg_result = np.mean(results)
        p5_result = np.percentile(results, 5)
        
        return {
            "mean_outcome": avg_result,
            "worst_case_p5": p5_result,
            "success_probability": np.mean([r > 1000000 for r in results])
        }

    def check_rebalance(self, current_allocation: Dict[str, float]) -> Dict:
        """
        Checks if rebalancing is needed and provides reasoning.
        """
        needs_rebalance = False
        drift_threshold = 0.05  # 5% drift triggers rebalance
        
        for asset, target in self.target_allocation.items():
            current = current_allocation.get(asset, 0)
            if abs(current - target) > drift_threshold:
                needs_rebalance = True
                break
        
        if needs_rebalance:
            # Determine reason
            stock_drift = current_allocation.get('stocks', 0.6) - self.target_allocation.get('stocks', 0.6)
            if stock_drift > 0.05:
                reason = "Stocks overweight after rally — reducing risk"
            elif stock_drift < -0.05:
                reason = "Stocks underweight — buying the dip"
            else:
                reason = "Portfolio drift exceeded 5% threshold"
            
            reasoning = self.reasoner.explain_rebalance(
                current_allocation, self.target_allocation, reason
            )
            self.logger.info(f"\n{'='*50}\nWEALTH REBALANCE:\n{reasoning}\n{'='*50}")
            
            return {
                "needs_rebalance": True,
                "current": current_allocation,
                "target": self.target_allocation,
                "reasoning": reasoning
            }
        
        return {"needs_rebalance": False, "reasoning": None}

    def generate_proposals(self) -> List[Dict]:
        return []

    def execute_instruction(self, instruction: Dict[str, Any]):
        if instruction.get('type') == 'rebalance':
            self.logger.info(f"Executing rebalance: {instruction.get('reason', 'N/A')}")
