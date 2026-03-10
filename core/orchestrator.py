import logging
import yaml
from typing import Dict, List, Optional
from agents.risk_manager import RiskManager
from core.base_agent import BaseAgent
from core.global_optimizer import SimpleRuleBasedOptimizer
from core.broker_interface import BrokerInterface, TradeOrder
from core.telegram_notifier import TelegramNotifier
from core.llm_supervisor import LLMSupervisor

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
        self.telegram = TelegramNotifier()
        self.llm_supervisor = LLMSupervisor()
        
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

    def run_llm_supervision(self, macro_sentiment: str, regime: str):
        """
        Triggers the LLM Supervisor to assess the market context and 
        dynamically adjust RiskManager constraints.
        """
        if not self.llm_supervisor.enabled:
            return

        context = {
            "macro_sentiment": macro_sentiment,
            "regime": regime,
            "daily_pnl": self.risk_manager.daily_pnl,
            "global_drawdown": self.risk_manager.max_global_drawdown,  # Ideally current drawdown, but using threshold for now
            "current_constraints": {
                "max_leverage": self.risk_manager.max_leverage,
                "max_position_size_pct": self.risk_manager.max_single_position_pct
            }
        }
        
        self.logger.info("🤖 Requesting Strategic Assessment from LLM Supervisor...")
        decision = self.llm_supervisor.analyze_market_context(context)
        
        if decision:
            reasoning = decision.get("reasoning", "")
            summary = decision.get("executive_summary", "")
            
            # Apply dynamic constraints
            if "max_leverage" in decision:
                self.risk_manager.max_leverage = decision["max_leverage"]
            if "max_position_size_pct" in decision:
                self.risk_manager.max_single_position_pct = decision["max_position_size_pct"]
            if decision.get("emergency_stop"):
                self.risk_manager.set_emergency_stop(True)
                
            self.logger.info(f"LLM Supervisor Adjustments Applied: {decision}")
            
            # Broadcast to Telegram
            alert_text = (
                "<b>🧠 LLM SUPERVISOR UPDATE</b>\n\n"
                f"<b>Summary:</b> {summary}\n"
                f"<b>Reasoning:</b> {reasoning}\n\n"
                f"<b>New Max Leverage:</b> {self.risk_manager.max_leverage}x\n"
                f"<b>New Max Pos Size:</b> {self.risk_manager.max_single_position_pct * 100:.1f}%\n"
                f"<b>Emergency Stop:</b> {self.risk_manager.emergency_stop}"
            )
            self.telegram._send_message(alert_text)

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
            
            # Send Telegram Alert
            self.telegram.send_trade_alert(
                symbol=order.symbol,
                action=action,
                quantity=order.qty,
                price=order.price or 0.0,
                strategy=result['source_agent'],
                reason=reasoning
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Trade execution failed: {e}")
            self.telegram.send_error_alert("Orchestrator Trade Execution", str(e))
            return None

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    orch = Orchestrator("config/config.yaml")
    orch.run_cycle()
