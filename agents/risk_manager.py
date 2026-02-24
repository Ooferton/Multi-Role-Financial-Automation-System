import logging
from typing import Dict, Optional

class RiskManager:
    """
    Decentralized Risk Management agent.
    Provides global oversight for position sizing, drawdown limits, and veto power.
    """
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Risk constraints
        self.max_global_drawdown = config.get('max_global_drawdown', 0.20)
        self.max_single_position_pct = config.get('max_single_position_pct', 0.15)
        self.emergency_stop = False

    def review_trade(self, symbol: str, action: str, quantity: float, price: float, 
                     portfolio: Dict, current_drawdown: float, crisis_severity: float = 0.0) -> Dict:
        """
        Reviews a proposed trade and returns a modified or approved instruction.
        Now supports sizing checks for both LONG and SHORT entries.
        """
        if self.emergency_stop:
            return {"approved": False, "reason": "Emergency stop active"}
            
        if current_drawdown < -self.max_global_drawdown:
            self.logger.warning(f"Global drawdown {current_drawdown:.1%} exceeds limit!")
            return {"approved": False, "reason": "Global drawdown limit exceeded"}

        # Position size check
        equity = portfolio.get('equity', 10000)
        trade_value = quantity * price
        

        position_pct = trade_value / equity if equity > 0 else 0
        
        # Crisis Sizing Overrides
        dynamic_long_cap = self.max_single_position_pct
        dynamic_short_cap = self.max_single_position_pct * 0.7 
        
        if crisis_severity > 0.5:
            # Shift risk budget from Longs to Shorts/Hedges
            dynamic_long_cap *= 0.3 # Drastically reduce long exposure
            dynamic_short_cap = self.max_single_position_pct * 1.5 # Allow aggressive shorts
            
            # If symbol is a hedge (VIX/SQQQ), allow even more
            if any(h in symbol for h in ["SQQQ", "VIX", "SH", "UVXY"]):
                dynamic_long_cap = self.max_single_position_pct * 2.0 # Allow large hedge buys
        
        current_qty = portfolio.get('positions', {}).get(symbol, 0)
        
        # Opening Logic
        if action == "BUY" and current_qty >= 0: # Opening/Adding to LONG
            if position_pct > dynamic_long_cap:
                new_qty = (equity * dynamic_long_cap) / price
                return self._resized_response(symbol, quantity, new_qty, "LONG cap (Crisis-Adjusted)")
        
        elif action == "SELL" and current_qty <= 0: # Opening/Adding to SHORT
            if position_pct > dynamic_short_cap:
                new_qty = (equity * dynamic_short_cap) / price
                return self._resized_response(symbol, quantity, new_qty, "SHORT cap (Crisis-Adjusted)")

        return {"approved": True, "quantity": quantity, "reason": "Approved by RiskManager"}

    def _resized_response(self, symbol, old_qty, new_qty, reason_tag):
        self.logger.info(f"RiskManager: Resizing {symbol} from {old_qty} to {new_qty:.2f} ({reason_tag})")
        return {
            "approved": True, 
            "quantity": round(new_qty, 4),
            "reason": f"Sized down by RiskManager to {reason_tag}"
        }

    def set_emergency_stop(self, state: bool):
        self.emergency_stop = state
        self.logger.info(f"RiskManager: Emergency stop set to {state}")
