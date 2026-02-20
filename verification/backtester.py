from datetime import datetime, timedelta
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from core.broker_interface import BrokerInterface, TradeOrder, Position
from agents.trading_agent import TradingAgent, TradingStrategy
from data.feature_store import FeatureStore, MarketTick

class Backtester:
    """
    Simulates the passage of time and replays historical data to the TradingAgent.
    Uses a MockBroker for execution.
    """
    def __init__(self, agent: TradingAgent, start_date: datetime, end_date: datetime, initial_capital: float = 100000.0):
        self.agent = agent
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        
        # Ensure agent uses a MockBroker
        if not isinstance(self.agent.broker, BrokerInterface):
             raise ValueError("Agent must have a compatible Broker Interface")
        
        # Reset Broker for Backtest
        if hasattr(self.agent.broker, 'cash'):
            self.agent.broker.cash = initial_capital
            self.agent.broker.positions = {}
            self.agent.broker.orders = {}

        self.history: List[Dict] = [] # Track equity curve

    def run(self, data_feed: List[MarketTick]):
        """
        Replays the provided list of ticks.
        """
        print(f"Starting Backtest from {self.start_date} to {self.end_date}...")
        
        for tick in data_feed:
            if tick.timestamp < self.start_date or tick.timestamp > self.end_date:
                continue
            
            # 1. Update Market Logic
            self.agent.update_market_state({'tick': tick})
            self.agent.feature_store.save_tick(tick) # Persist for Dashboard
            
            # 2. Track Performance
            summary = self.agent.broker.get_account_summary()
            self.history.append({
                'timestamp': tick.timestamp,
                'equity': summary['equity'],
                'cash': summary['cash'],
                'price': tick.price
            })
            
        print("Backtest Complete.")
        self._calculate_metrics()

    def _calculate_metrics(self):
        if not self.history:
            print("No trades or data to analyze.")
            return

        df = pd.DataFrame(self.history)
        df.set_index('timestamp', inplace=True)
        
        initial = self.initial_capital
        final = df['equity'].iloc[-1]
        returns = (final - initial) / initial
        
        # Drawdown
        df['peak'] = df['equity'].cummax()
        df['drawdown'] = (df['equity'] - df['peak']) / df['peak']
        max_drawdown = df['drawdown'].min()
        
        print("\n--- Backtest Results ---")
        print(f"Initial Capital: ${initial:,.2f}")
        print(f"Final Equity:    ${final:,.2f}")
        print(f"Total Return:    {returns:.2%}")
        print(f"Max Drawdown:    {max_drawdown:.2%}")
        
        return df

if __name__ == "__main__":
    # Quick Test
    from agents.mock_broker import MockBroker
    from strategies.rl_strategy import RLStrategy # Use AI!
    from data.yahoo_connector import YahooDataConnector
    
    # Setup
    feature_store = FeatureStore()
    broker = MockBroker()
    agent = TradingAgent("ai_trader_v1", {}, feature_store, broker)
    
    # Use the trained brain
    strategy = RLStrategy("ppo_model_v1", {}, broker, "production_models/ppo_trading_real_v1")
    agent.add_strategy(strategy)
    
    # Fetch Real Data (SPY)
    connector = YahooDataConnector()
    print("Fetching real SPY data from Yahoo Finance...")
    # Get last 5 days of 1-minute data
    end = datetime.now()
    start = end - timedelta(days=5)
    ticks = connector.fetch_historical_ticks("SPY", start, end)
        
    # Run
    if ticks:
        backtester = Backtester(agent, ticks[0].timestamp, ticks[-1].timestamp, initial_capital=30.0)
        backtester.run(ticks)
    else:
        print("Failed to fetch data.")
