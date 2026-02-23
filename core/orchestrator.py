import logging
import yaml
from typing import Dict, List, Optional
from core.risk_manager import RiskManager
from core.base_agent import BaseAgent
from core.global_optimizer import SimpleRuleBasedOptimizer
from core.broker_interface import BrokerInterface, TradeOrder

class Orchestrator:
    """
    The Central Nervous System of the Financial Platform.
    Responsible for allocating capital, managing global state, and coordinating agents.
    """
    def __init__(self, config_path: str):
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path)
        self.risk_manager = RiskManager(self.config)
        self.optimizer = SimpleRuleBasedOptimizer()
        self.broker: Optional[BrokerInterface] = None
        
        self.capital_allocation = {}
        self.registered_agents: Dict[str, BaseAgent] = {}
        self.execution_log: List[Dict] = []

    def _load_config(self, path: str) -> Dict:
        with open(path, 'r') as f:
            return yaml.safe_load(f)

    def set_broker(self, broker: BrokerInterface):
        """Sets the broker used for trade execution."""
        self.broker = broker
        self.logger.info(f"Broker connected: {type(broker).__name__}")

    def register_agent(self, agent_instance: BaseAgent):
        """Registers a specialized agent with the orchestrator."""
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

    def run_cycle(self) -> List[Dict]:
        """
        Main execution loop.
        1. Allocate capital
        2. Collect proposals from agents
        3. Risk-check each proposal
        4. Score approved proposals
        5. Execute top proposals via broker
        6. Track PnL
        
        Returns list of execution results.
        """
        if self.risk_manager.is_halted:
            self.logger.warning("Orchestrator skipping cycle: System is HALTED.")
            return []

        self.allocate_capital()
        
        # 1. Collect proposals from all active agents
        all_proposals = []
        for agent in self.registered_agents.values():
            if agent.is_active:
                proposals = agent.generate_proposals()
                for p in proposals:
                    p['source_agent'] = agent.name
                all_proposals.extend(proposals)
                
        self.logger.info(f"Collected {len(all_proposals)} proposals from agents.")
        
        if not all_proposals:
            return []
        
        # 2. Risk-check each proposal
        portfolio_state = self._get_portfolio_state()
        approved = []
        for proposal in all_proposals:
            if self.risk_manager.check_trade_risk(proposal, portfolio_state):
                approved.append(proposal)
            else:
                self.logger.info(f"Proposal REJECTED by risk manager: {proposal.get('action')} {proposal.get('symbol')}")
        
        self.logger.info(f"{len(approved)}/{len(all_proposals)} proposals passed risk checks.")
        
        if not approved:
            return []
        
        # 3. Score and rank approved proposals
        scored = []
        for proposal in approved:
            score = self.optimizer.evaluate_proposal(proposal)
            scored.append((score, proposal))
        scored.sort(key=lambda x: x[0], reverse=True)
        
        # 4. Execute proposals via broker
        results = []
        for score, proposal in scored:
            result = self._execute_proposal(proposal, score)
            if result:
                results.append(result)
        
        self.execution_log.extend(results)
        self.logger.info(f"Executed {len(results)} trades this cycle.")
        return results
    
    def _get_portfolio_state(self) -> Dict:
        """Builds current portfolio state from broker."""
        if self.broker:
            summary = self.broker.get_account_summary()
            positions = self.broker.get_positions()
            return {
                'equity': summary.get('equity', 100000),
                'cash': summary.get('cash', 100000),
                'market_value': sum(p.market_value for p in positions),
                'num_positions': len(positions)
            }
        return {'equity': 100000, 'cash': 100000, 'market_value': 0, 'num_positions': 0}
    
    def _execute_proposal(self, proposal: Dict, score: float) -> Optional[Dict]:
        """
        Converts an approved proposal into a TradeOrder and submits it.
        Returns execution result or None on failure.
        """
        if not self.broker:
            self.logger.warning("No broker connected. Cannot execute trades.")
            return None
        
        action = proposal.get('action', '').upper()
        if action not in ('BUY', 'SELL'):
            return None
        
        order = TradeOrder(
            symbol=proposal.get('symbol', ''),
            qty=proposal.get('quantity', 0),
            side=action,
            order_type=proposal.get('type', 'MARKET'),
            price=proposal.get('price'),
        )
        
        reasoning = str(proposal.get('reason', ''))
        
        try:
            broker_result = self.broker.submit_order(order, reasoning)
            
            # Calculate PnL estimate for risk tracking
            trade_value = (order.price or 0) * order.qty
            if action == 'SELL':
                # Simplified PnL: compare to entry if available
                pnl = proposal.get('pnl', 0)
                self.risk_manager.update_daily_pnl(pnl)
            
            result = {
                'order_id': broker_result.get('order_id'),
                'status': broker_result.get('status'),
                'action': action,
                'symbol': order.symbol,
                'quantity': order.qty,
                'price': order.price,
                'score': score,
                'source_agent': proposal.get('source_agent', 'unknown'),
                'reason': reasoning
            }
            
            self.logger.info(
                f"✅ EXECUTED: {action} {order.qty} {order.symbol} @ ${order.price or 0:.2f} "
                f"(score: {score:.2f}, agent: {result['source_agent']})"
            )
            return result
            
        except Exception as e:
            self.logger.error(f"Trade execution failed: {e}")
            return None

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    orch = Orchestrator("config/config.yaml")
    orch.run_cycle()
