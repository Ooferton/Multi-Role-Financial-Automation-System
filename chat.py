"""
Financial AI — Interactive Chat Interface.

Talk to your AI financial advisor naturally.
"""
import logging
import sys
import os

# Setup path
sys.path.insert(0, os.path.dirname(__file__))

from core.orchestrator import Orchestrator
from core.financial_ai import FinancialAI
from agents.trading_agent import TradingAgent
from agents.wealth_agent import WealthStrategyAgent
from agents.lending_agent import LendingAgent
from agents.mock_broker import MockBroker
from data.feature_store import FeatureStore

# Logging
logging.basicConfig(
    level=logging.WARNING,  # Quiet mode for chat
    format='%(message)s'
)

BANNER = """
╔══════════════════════════════════════════════════════╗
║          🤖  FINANCIAL AI  v1.0                      ║
║          Your AI-Powered Financial Advisor           ║
║                                                      ║
║  Type naturally — I understand trading, investing,   ║
║  retirement planning, debt optimization, and more.   ║
║                                                      ║
║  Type 'help' for capabilities  |  'quit' to exit     ║
╚══════════════════════════════════════════════════════╝
"""

def main():
    print(BANNER)
    
    # 1. Initialize System
    print("  Initializing system...", end=" ")
    
    orchestrator = Orchestrator("config/config.yaml")
    
    # 2. Create and register agents
    feature_store = FeatureStore()
    broker = MockBroker(initial_cash=10000.0)
    
    trading_agent = TradingAgent("Trader", {}, feature_store, broker)
    wealth_agent = WealthStrategyAgent("Wealth", {})
    lending_agent = LendingAgent("Lending", {})
    
    orchestrator.register_agent(trading_agent)
    orchestrator.register_agent(wealth_agent)
    orchestrator.register_agent(lending_agent)
    
    # 3. Try loading RL strategy if model exists
    model_path = "production_models/ppo_trading_real_v1"
    if os.path.exists(model_path + ".zip") or os.path.exists(model_path):
        try:
            from strategies.rl_strategy import RLStrategy
            strategy = RLStrategy("RL_Brain", {}, broker, model_path)
            trading_agent.add_strategy(strategy)
            print("✅ (AI model loaded)")
        except Exception as e:
            print(f"⚠️ (AI model failed: {e})")
    else:
        print("⚠️ (No trained model found — trading analysis limited)")
    
    # 4. Create the Financial AI
    ai = FinancialAI(orchestrator)
    
    # 5. Chat Loop
    print()
    while True:
        try:
            user_input = input("  You: ").strip()
            
            if not user_input:
                continue
            if user_input.lower() in ('quit', 'exit', 'q'):
                print("\n  👋 Goodbye! Stay financially healthy.\n")
                break
            
            response = ai.chat(user_input)
            print(f"\n  AI: {response}\n")
            
        except KeyboardInterrupt:
            print("\n\n  👋 Goodbye!\n")
            break
        except Exception as e:
            print(f"\n  ❌ Error: {e}\n")

if __name__ == "__main__":
    main()
