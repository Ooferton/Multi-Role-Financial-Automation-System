from typing import Dict, Optional, List
from agents.trading_agent import TradingStrategy
from data.feature_store import MarketTick

class LongTermStrategy(TradingStrategy):
    """
    Long-term portfolio management strategy.
    
    Logic:
    1. Analyzes macroeconomic factors and asset fundamentals (simulated for now).
    2. Rebalances portfolio monthly or quarterly.
    3. Targets specific asset allocation mix.
    """
    
    def on_tick(self, tick: MarketTick) -> Optional[Dict]:
        # Long-term strategies generally ignore tick-level data
        return None

    def on_bar(self, bar: Dict) -> Optional[Dict]:
        """
        Evaluates portfolio state on daily/weekly bars.
        """
        # Placeholder logic: simple rebalancing signal if cash is too high
        # In reality, this would run a factor model
        symbol = bar.get('symbol')
        close_price = bar.get('close')
        
        # Example signal: Buy if price drops significantly (Mean Reversion / Value factor)
        # This is just a dummy condition
        if close_price and close_price < 100: 
            return {
                "action": "BUY",
                "symbol": symbol,
                "type": "MARKET",
                "quantity": 10,
                "reason": "Value Factor Trigger"
            }
        return None
