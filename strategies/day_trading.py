from typing import Dict, Optional
from agents.trading_agent import TradingStrategy
from data.feature_store import MarketTick

class DayTradingStrategy(TradingStrategy):
    """
    Intraday trading strategy.
    
    Logic:
    1. Momentum breakouts.
    2. VWAP mean reversion.
    3. Closes all positions by end of day.
    """
    
    def __init__(self, name: str, config: Dict):
        super().__init__(name, config)
        self.current_vwap = 0.0
        self.tick_count = 0
        self.cum_pv = 0.0
        self.cum_vol = 0.0

    def on_tick(self, tick: MarketTick) -> Optional[Dict]:
        """
        Updates VWAP and checks for breakout signals.
        """
        self.tick_count += 1
        
        # Update VWAP
        price = tick.price
        size = tick.size
        self.cum_pv += price * size
        self.cum_vol += size
        
        if self.cum_vol > 0:
            self.current_vwap = self.cum_pv / self.cum_vol
            
        # Example Logic: Buy if price crosses above VWAP + threshold (Momentum)
        if self.tick_count > 100 and price > self.current_vwap * 1.005:
            return {
                "action": "BUY",
                "symbol": tick.symbol,
                "type": "MARKET",
                "price": price,
                "quantity": 100,
                "reason": "VWAP Breakout"
            }
            
        return None

    def on_bar(self, bar: Dict) -> Optional[Dict]:
        # Reset VWAP at start of day (logic would go here)
        return None
