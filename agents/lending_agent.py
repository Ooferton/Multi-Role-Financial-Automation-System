import logging
from typing import Dict, List, Any
from core.base_agent import BaseAgent
from ml.reasoning_engine import ReasoningEngine

class LendingAgent(BaseAgent):
    """
    Optimizes debt structure and lending opportunities.
    
    Responsibilities:
    1. Monitor interest rates for refinance opportunities.
    2. Optimize debt payoff (Avalanche vs Snowball).
    3. Evaluate peer-to-peer lending or private credit deals.
    
    All decisions include human-readable reasoning.
    """
    
    def __init__(self, name: str = "LendingAgent", config: Dict[str, Any] = None):
        super().__init__(name, config or {})
        self.reasoner = ReasoningEngine()
        self.logger = logging.getLogger(__name__)
        self.watched_rates = {}  # Track rate history

    def update_market_state(self, market_data: Dict[str, Any]):
        # Update Fed Funds Rate, Mortgage Rates, etc.
        if 'mortgage_rate' in market_data:
            self.watched_rates['mortgage'] = market_data['mortgage_rate']
        if 'fed_funds_rate' in market_data:
            self.watched_rates['fed_funds'] = market_data['fed_funds_rate']

    def analyze_refinance(self, current_loan: Dict, new_rate: float) -> Dict:
        """
        Calculates break-even point for refinancing with full reasoning.
        """
        current_rate = current_loan.get('rate', 0.05)
        balance = current_loan.get('balance', 100000)
        remaining_months = current_loan.get('remaining_months', 360)
        closing_costs = current_loan.get('closing_costs', 3000)
        
        # Monthly payment calculations
        monthly_current = (balance * current_rate / 12) / (1 - (1 + current_rate / 12) ** -remaining_months)
        monthly_new = (balance * new_rate / 12) / (1 - (1 + new_rate / 12) ** -remaining_months)
        monthly_savings = monthly_current - monthly_new
        
        break_even_months = int(closing_costs / monthly_savings) if monthly_savings > 0 else 999
        
        # Generate reasoning
        reasoning = self.reasoner.explain_refinance(
            current_rate=current_rate,
            new_rate=new_rate,
            monthly_savings=monthly_savings,
            break_even_months=break_even_months
        )
        
        should_refinance = break_even_months < 24 and monthly_savings > 50
        
        self.logger.info(f"\n{'='*50}\nLENDING ANALYSIS:\n{reasoning}\n{'='*50}")
        
        return {
            "should_refinance": should_refinance,
            "monthly_savings": monthly_savings,
            "break_even_months": break_even_months,
            "total_savings": monthly_savings * remaining_months - closing_costs,
            "reasoning": reasoning
        }

    def compare_payoff_strategies(self, debts: List[Dict]) -> Dict:
        """
        Compares Avalanche (highest rate first) vs Snowball (lowest balance first).
        Returns reasoning for the recommended approach.
        """
        if not debts:
            return {"strategy": "none", "reasoning": "No debts to analyze"}
        
        # Sort for each strategy
        avalanche_order = sorted(debts, key=lambda d: d.get('rate', 0), reverse=True)
        snowball_order = sorted(debts, key=lambda d: d.get('balance', 0))
        
        total_interest_avalanche = sum(d['balance'] * d['rate'] for d in debts)
        highest_rate = avalanche_order[0]
        smallest_balance = snowball_order[0]
        
        factors = [
            f"Highest rate debt: {highest_rate.get('name', 'Unknown')} at {highest_rate['rate']:.1%}",
            f"Smallest balance: {smallest_balance.get('name', 'Unknown')} at ${smallest_balance['balance']:,.0f}",
            f"Total debts: {len(debts)}, weighted avg rate: {total_interest_avalanche / sum(d['balance'] for d in debts):.1%}"
        ]
        
        # Recommend avalanche if rate spread is large, snowball if balances are close
        rate_spread = avalanche_order[0]['rate'] - avalanche_order[-1]['rate']
        
        if rate_spread > 0.05:
            strategy = "avalanche"
            summary = f"Use Avalanche: Rate spread of {rate_spread:.1%} makes interest savings significant"
            factors.append("Large rate spread favors minimizing interest payments")
        else:
            strategy = "snowball"
            summary = f"Use Snowball: Similar rates — quick wins from small balances boost motivation"
            factors.append("Tight rate spread means psychological wins outweigh small interest differences")
        
        from ml.reasoning_engine import TradeReasoning
        reasoning = TradeReasoning(
            summary=summary,
            factors=factors,
            confidence=0.75,
            risk_notes=["Debt payoff strategies assume no new debt is added"]
        )
        
        self.logger.info(f"\n{'='*50}\nDEBT STRATEGY:\n{reasoning}\n{'='*50}")
        
        return {
            "strategy": strategy,
            "order": avalanche_order if strategy == "avalanche" else snowball_order,
            "reasoning": reasoning
        }

    def generate_proposals(self) -> List[Dict]:
        return []

    def execute_instruction(self, instruction: Dict[str, Any]):
        if instruction.get('type') == 'refinance':
            self.logger.info(f"Executing refinance: {instruction.get('reason', 'N/A')}")
