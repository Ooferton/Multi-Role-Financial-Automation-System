"""
Central Financial AI — The Brain.

A conversational AI that users interact with naturally.
It understands financial intent, delegates to specialized agents,
and synthesizes responses with reasoning.

Architecture:
  User <-> FinancialAI <-> [TradingAgent, WealthAgent, LendingAgent]
                       <-> ReasoningEngine
                       <-> StrategyEvolver
"""
import logging
import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta

from core.orchestrator import Orchestrator
from agents.trading_agent import TradingAgent
from agents.wealth_agent import WealthStrategyAgent
from agents.lending_agent import LendingAgent
from ml.reasoning_engine import ReasoningEngine, TradeReasoning
from ml.strategy_evolver import StrategyEvolver


# ── Intent Categories ────────────────────────────────────────────
INTENTS = {
    'TRADE':      ['buy', 'sell', 'trade', 'short', 'long', 'position', 'order', 'stock', 'shares'],
    'PORTFOLIO':  ['portfolio', 'holdings', 'positions', 'balance', 'equity', 'account', 'how am i doing', 'performance'],
    'WEALTH':     ['retire', 'retirement', 'monte carlo', 'rebalance', 'allocation', 'long term', 'save', 'invest'],
    'LENDING':    ['loan', 'mortgage', 'refinance', 'debt', 'interest rate', 'payoff', 'owe', 'credit'],
    'STRATEGY':   ['strategy', 'genome', 'parameters', 'adapt', 'evolve', 'modify', 'threshold', 'stop loss', 'take profit'],
    'MARKET':     ['market', 'price', 'quote', 'spy', 'aapl', 'what is', 'how is'],
    'HELP':       ['help', 'what can you do', 'commands', 'capabilities'],
}


class FinancialAI:
    """
    The central conversational financial assistant.
    Parses user intent and delegates to the right agent.
    """

    def __init__(self, orchestrator: Orchestrator):
        self.logger = logging.getLogger("FinancialAI")
        self.orchestrator = orchestrator
        self.reasoner = ReasoningEngine()
        
        # Agent references (populated via orchestrator)
        self.trading_agent: Optional[TradingAgent] = None
        self.wealth_agent: Optional[WealthStrategyAgent] = None
        self.lending_agent: Optional[LendingAgent] = None
        
        # Conversation memory
        self.conversation_history: List[Dict] = []
        self.context: Dict[str, Any] = {}
        
        # Discover agents from orchestrator
        self._discover_agents()
        
        self.logger.info("Financial AI initialized. All agents connected.")

    def _discover_agents(self):
        """Find and reference registered agents by type."""
        for name, agent in self.orchestrator.registered_agents.items():
            if isinstance(agent, TradingAgent):
                self.trading_agent = agent
            elif isinstance(agent, WealthStrategyAgent):
                self.wealth_agent = agent
            elif isinstance(agent, LendingAgent):
                self.lending_agent = agent

    # ── Intent Detection ─────────────────────────────────────────
    def _detect_intent(self, message: str) -> Tuple[str, float]:
        """
        Detects the user's intent from natural language.
        Returns (intent_category, confidence).
        """
        msg = message.lower().strip()
        scores = {}

        for intent, keywords in INTENTS.items():
            score = sum(1 for kw in keywords if kw in msg)
            if score > 0:
                scores[intent] = score

        if not scores:
            return ('UNKNOWN', 0.0)

        best = max(scores, key=scores.get)
        confidence = min(1.0, scores[best] / 3.0)
        return (best, confidence)

    # ── Entity Extraction ────────────────────────────────────────
    def _extract_entities(self, message: str) -> Dict:
        """Extracts financial entities from the message."""
        entities = {}
        msg = message.upper()

        # Ticker symbols (1-5 uppercase letters)
        tickers = re.findall(r'\b([A-Z]{1,5})\b', msg)
        # Filter out common words
        noise = {'I', 'A', 'THE', 'MY', 'ME', 'IS', 'IT', 'IN', 'ON', 'TO', 'OF',
                 'AND', 'OR', 'FOR', 'AT', 'DO', 'IF', 'SO', 'AM', 'HOW', 'CAN',
                 'BUY', 'SELL', 'WHAT', 'WITH', 'MUCH', 'WORTH', 'ABOUT', 'FROM'}
        tickers = [t for t in tickers if t not in noise and len(t) >= 2]
        if tickers:
            entities['symbols'] = tickers

        # Dollar amounts
        amounts = re.findall(r'\$?([\d,]+\.?\d*)', message)
        if amounts:
            entities['amounts'] = [float(a.replace(',', '')) for a in amounts]

        # Percentages
        pcts = re.findall(r'([\d.]+)%', message)
        if pcts:
            entities['percentages'] = [float(p) / 100 for p in pcts]

        # Quantities
        qty_match = re.findall(r'(\d+)\s*(?:shares|units|lots)', message.lower())
        if qty_match:
            entities['quantity'] = int(qty_match[0])

        return entities

    # ── Main Chat Interface ──────────────────────────────────────
    def chat(self, message: str) -> str:
        """
        Main entry point. User says something, AI responds.
        """
        self.conversation_history.append({
            'role': 'user', 'content': message, 'timestamp': datetime.now()
        })

        intent, confidence = self._detect_intent(message)
        entities = self._extract_entities(message)

        self.logger.info(f"Intent: {intent} ({confidence:.0%}) | Entities: {entities}")

        # Route to handler
        handlers = {
            'TRADE':     self._handle_trade,
            'PORTFOLIO': self._handle_portfolio,
            'WEALTH':    self._handle_wealth,
            'LENDING':   self._handle_lending,
            'STRATEGY':  self._handle_strategy,
            'MARKET':    self._handle_market,
            'HELP':      self._handle_help,
        }

        handler = handlers.get(intent, self._handle_unknown)
        response = handler(message, entities)

        self.conversation_history.append({
            'role': 'assistant', 'content': response, 'timestamp': datetime.now()
        })

        return response

    # ── Intent Handlers ──────────────────────────────────────────

    def _handle_trade(self, message: str, entities: Dict) -> str:
        """Handle trade requests."""
        if not self.trading_agent:
            return "⚠️ Trading agent is not connected. Please initialize the system first."

        msg = message.lower()
        symbols = entities.get('symbols', ['SPY'])
        amounts = entities.get('amounts', [])
        quantity = entities.get('quantity', 0)

        # Determine action
        if any(w in msg for w in ['buy', 'long', 'purchase']):
            action = "BUY"
        elif any(w in msg for w in ['sell', 'short', 'dump', 'exit']):
            action = "SELL"
        else:
            # Ask the AI what it thinks
            return self._get_ai_recommendation(symbols[0] if symbols else 'SPY')

        symbol = symbols[0] if symbols else 'SPY'

        # Get current price
        broker = self.trading_agent.broker
        summary = broker.get_account_summary()
        cash = summary.get('cash', 0)

        # Determine quantity
        if amounts:
            qty = amounts[0]  # Use dollar amount as qty placeholder
        elif quantity:
            qty = quantity
        else:
            qty = round(cash * 0.1 / 600, 4)  # Default: 10% of cash

        # Generate reasoning
        reasoning = self.reasoner.explain_trade(
            action_val=0.8 if action == "BUY" else -0.8,
            indicators={'rsi_14': 50, 'macd': 0, 'macd_signal': 0, 'bb_width': 0.03, 'dist_sma_20': 0},
            portfolio={'cash': cash, 'position_qty': 0},
            symbol=symbol
        )

        response = (
            f"📊 **{action} Order for {symbol}**\n"
            f"   Quantity: {qty}\n"
            f"   Account Cash: ${cash:,.2f}\n\n"
            f"🧠 **AI Reasoning:**\n"
            f"   {reasoning.summary}\n"
            f"   Factors: {' • '.join(reasoning.factors[:3])}\n"
        )

        if reasoning.risk_notes:
            response += f"   ⚠️ Risks: {' | '.join(reasoning.risk_notes)}\n"

        response += f"\n💡 To confirm, I would execute this through the broker."
        return response

    def _handle_portfolio(self, message: str, entities: Dict) -> str:
        """Show portfolio status."""
        if not self.trading_agent:
            return "⚠️ Trading agent not connected."

        broker = self.trading_agent.broker
        summary = broker.get_account_summary()
        positions = broker.get_positions()

        response = (
            f"📈 **Portfolio Summary**\n"
            f"   Equity: ${summary.get('equity', 0):,.2f}\n"
            f"   Cash:   ${summary.get('cash', 0):,.2f}\n"
            f"   Buying Power: ${summary.get('buying_power', 0):,.2f}\n\n"
        )

        if positions:
            response += "📋 **Open Positions:**\n"
            for p in positions:
                pnl_sign = "+" if p.unrealized_pl >= 0 else ""
                response += (
                    f"   {p.symbol}: {p.qty} shares @ ${p.avg_entry_price:.2f} "
                    f"(now ${p.current_price:.2f}, {pnl_sign}${p.unrealized_pl:.2f})\n"
                )
        else:
            response += "   No open positions.\n"

        # Strategy status
        for strategy in self.trading_agent.strategies:
            if hasattr(strategy, 'evolver'):
                response += f"\n🧬 **Strategy Genome:**\n"
                response += "   " + strategy.evolver.get_status().replace("\n", "\n   ")

        return response

    def _handle_wealth(self, message: str, entities: Dict) -> str:
        """Handle wealth/retirement queries."""
        if not self.wealth_agent:
            return "⚠️ Wealth agent not connected."

        msg = message.lower()
        amounts = entities.get('amounts', [])

        if 'monte carlo' in msg or 'retire' in msg or 'simulation' in msg:
            portfolio_value = amounts[0] if amounts else 100000
            contribution = amounts[1] if len(amounts) > 1 else 12000

            results = self.wealth_agent.run_monte_carlo(portfolio_value, contribution)

            return (
                f"🎯 **Retirement Monte Carlo Simulation**\n"
                f"   Starting Value: ${portfolio_value:,.0f}\n"
                f"   Annual Contribution: ${contribution:,.0f}\n"
                f"   Horizon: {self.wealth_agent.years_to_simulate} years\n"
                f"   Simulations: {self.wealth_agent.simulation_runs}\n\n"
                f"📊 **Results:**\n"
                f"   Average Outcome: ${results['mean_outcome']:,.0f}\n"
                f"   Worst Case (5th pctile): ${results['worst_case_p5']:,.0f}\n"
                f"   Probability of reaching $1M: {results['success_probability']:.0%}\n"
            )

        elif 'rebalance' in msg or 'allocation' in msg:
            # Example current allocation
            current = {'stocks': 0.70, 'bonds': 0.20, 'cash': 0.10}
            result = self.wealth_agent.check_rebalance(current)

            if result['needs_rebalance']:
                reasoning = result['reasoning']
                return (
                    f"⚖️ **Rebalancing Recommended**\n\n"
                    f"🧠 {reasoning.summary}\n"
                    f"   Factors: {' • '.join(reasoning.factors)}\n"
                )
            else:
                return "✅ Portfolio is well-balanced. No rebalancing needed right now."

        return (
            f"💰 **Wealth Management**\n"
            f"I can help with:\n"
            f"  • \"Run a Monte Carlo simulation with $50000\"\n"
            f"  • \"Should I rebalance my portfolio?\"\n"
            f"  • \"What's my allocation look like?\"\n"
        )

    def _handle_lending(self, message: str, entities: Dict) -> str:
        """Handle lending/debt queries."""
        if not self.lending_agent:
            return "⚠️ Lending agent not connected."

        msg = message.lower()
        amounts = entities.get('amounts', [])
        pcts = entities.get('percentages', [])

        if 'refinance' in msg or 'mortgage' in msg:
            balance = amounts[0] if amounts else 300000
            current_rate = pcts[0] if pcts else 0.065
            new_rate = pcts[1] if len(pcts) > 1 else current_rate - 0.01

            loan = {'balance': balance, 'rate': current_rate, 'remaining_months': 300}
            result = self.lending_agent.analyze_refinance(loan, new_rate)
            reasoning = result['reasoning']

            verdict = "✅ Yes, refinance!" if result['should_refinance'] else "❌ Not recommended right now."

            return (
                f"🏦 **Refinance Analysis**\n"
                f"   {verdict}\n\n"
                f"🧠 {reasoning.summary}\n"
                f"   {chr(10).join('   • ' + f for f in reasoning.factors)}\n\n"
                f"   Monthly Savings: ${result['monthly_savings']:,.2f}\n"
                f"   Break-even: {result['break_even_months']} months\n"
                f"   Total Savings: ${result['total_savings']:,.0f}\n"
            )

        elif 'debt' in msg or 'payoff' in msg or 'owe' in msg:
            # Example debts
            debts = [
                {'name': 'Credit Card', 'balance': 8000, 'rate': 0.22},
                {'name': 'Car Loan', 'balance': 15000, 'rate': 0.06},
                {'name': 'Student Loan', 'balance': 35000, 'rate': 0.045},
            ]
            result = self.lending_agent.compare_payoff_strategies(debts)
            reasoning = result['reasoning']

            return (
                f"💳 **Debt Payoff Strategy**\n\n"
                f"🧠 {reasoning.summary}\n"
                f"   {chr(10).join('   • ' + f for f in reasoning.factors)}\n\n"
                f"   Recommended order:\n"
                + "\n".join(f"   {i+1}. {d['name']} (${d['balance']:,.0f} @ {d['rate']:.1%})"
                           for i, d in enumerate(result['order']))
                + "\n"
            )

        return (
            f"🏦 **Lending & Debt**\n"
            f"I can help with:\n"
            f"  • \"Should I refinance my $300k mortgage at 6.5%?\"\n"
            f"  • \"What's the best debt payoff strategy?\"\n"
            f"  • \"Analyze my loans\"\n"
        )

    def _handle_strategy(self, message: str, entities: Dict) -> str:
        """Handle strategy/genome queries."""
        if not self.trading_agent:
            return "⚠️ Trading agent not connected."

        msg = message.lower()

        for strategy in self.trading_agent.strategies:
            if hasattr(strategy, 'evolver'):
                evolver = strategy.evolver

                if 'adapt' in msg or 'evolve' in msg:
                    mutations = strategy.self_adapt()
                    if mutations:
                        return (
                            f"🧬 **Self-Adaptation Complete**\n"
                            f"   Applied {len(mutations)} mutations:\n"
                            + "\n".join(f"   • {m}" for m in mutations)
                            + f"\n\n   {evolver.get_status()}"
                        )
                    else:
                        return "🧬 No adaptations needed right now. Strategy is performing within bounds."

                elif 'rollback' in msg:
                    if evolver.rollback():
                        return f"⏪ Rolled back to generation {evolver.genome.generation}.\n{evolver.get_status()}"
                    return "No history to rollback to."

                else:
                    return f"🧬 **Current Strategy DNA:**\n{evolver.get_status()}"

        return "No adaptive strategy found."

    def _handle_market(self, message: str, entities: Dict) -> str:
        """Handle market/price queries."""
        symbols = entities.get('symbols', ['SPY'])
        response = "📊 **Market Data**\n"
        for sym in symbols[:3]:
            response += f"   {sym}: (connect Alpaca API for real-time quotes)\n"
        return response

    def _handle_help(self, message: str, entities: Dict) -> str:
        """Show capabilities."""
        return (
            "🤖 **I'm your Financial AI. Here's what I can do:**\n\n"
            "📈 **Trading**\n"
            "   • \"Buy 10 shares of AAPL\"\n"
            "   • \"What should I do with SPY?\"\n"
            "   • \"Show my portfolio\"\n\n"
            "💰 **Wealth Management**\n"
            "   • \"Run a retirement simulation with $50k\"\n"
            "   • \"Should I rebalance?\"\n\n"
            "🏦 **Lending & Debt**\n"
            "   • \"Should I refinance my mortgage at 5.5%?\"\n"
            "   • \"Best strategy to pay off my debts?\"\n\n"
            "🧬 **Strategy Evolution**\n"
            "   • \"Show my strategy parameters\"\n"
            "   • \"Adapt the strategy based on performance\"\n"
            "   • \"Rollback the last mutation\"\n\n"
            "💬 Just talk to me naturally — I'll figure out the rest!"
        )

    def _handle_unknown(self, message: str, entities: Dict) -> str:
        return (
            f"🤔 I'm not sure what you mean. Try asking about:\n"
            f"   • Trading (buy/sell stocks)\n"
            f"   • Your portfolio\n"
            f"   • Retirement planning\n"
            f"   • Debt optimization\n"
            f"   • Strategy parameters\n"
            f"\nOr say \"help\" to see all my capabilities."
        )

    def _get_ai_recommendation(self, symbol: str) -> str:
        """Let the AI recommend an action for a symbol."""
        return (
            f"🤖 **AI Analysis for {symbol}**\n"
            f"   To get a real-time recommendation, the trading engine needs live data.\n"
            f"   Run `python live_runner.py` to start the AI trading loop.\n"
            f"   The AI will analyze RSI, MACD, trend, and volatility to decide.\n"
        )
