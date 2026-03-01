import logging
from typing import Dict, Optional, List
import numpy as np
from ml.quant_models import calculate_cvar
from core.soul_parser import SoulParser

class RiskManager:
    """
    Decentralized Risk Management agent.
    Provides global oversight for position sizing, drawdown limits, and veto power.
    """
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize SOUL.md Parser for natural language constraints
        self.soul_parser = SoulParser()
        soul_constraints = self.soul_parser.parse_constraints()
        
        # Risk constraints
        self.max_global_drawdown = config.get('max_global_drawdown', 0.20)
        self.max_single_position_pct = soul_constraints.get('max_position_size_pct', config.get('max_single_position_pct', 0.15))
        self.max_cvar_limit = config.get('max_cvar_limit', 0.05) # Max 5% expected shortfall
        self.max_daily_loss = soul_constraints.get('max_daily_loss', 500.0)
        self.max_leverage = soul_constraints.get('max_leverage', 1.0)
        self.max_open_positions = soul_constraints.get('max_open_positions', 5)
        
        self.daily_pnl = 0.0
        
        self.emergency_stop = False
        self.cvar_override = False # True if CVaR is too high
        self.parity_weights = {} # Dynamic Inverse-Vol allocations
        self.sector_allocations = {} # MARL Swarm Allocations
        
    def set_parity_weights(self, weights: Dict[str, float]):
        self.logger.info(f"RiskManager received new Parity Weights for {len(weights)} assets.")
        self.parity_weights = weights

    def set_sector_allocations(self, allocations: Dict[str, float]):
        self.logger.info(f"RiskManager received new Sector Allocations: {allocations}")
        self.sector_allocations = allocations
        
    def check_portfolio_risk(self, recent_returns: List[float]):
        """
        Calculates Conditional Value at Risk (CVaR).
        If CVaR > limit, enables strict override logic (halving limits).
        """
        if len(recent_returns) < 20:
            return
            
        cvar = calculate_cvar(np.array(recent_returns), confidence_level=0.99)
        
        if cvar > self.max_cvar_limit:
            if not self.cvar_override:
                self.logger.warning(f"🚨 CVaR ALERT: Expected shortfall {cvar:.2%} exceeds {self.max_cvar_limit:.2%} limit. Entering Defensive Mode.")
            self.cvar_override = True
        else:
            if self.cvar_override:
                self.logger.info(f"✅ CVaR normalized ({cvar:.2%}). Restoring normal risk limits.")
            self.cvar_override = False



    def review_trade(self, symbol: str, action: str, quantity: float, price: float, 
                     portfolio: Dict, current_drawdown: float, crisis_severity: float = 0.0,
                     sector: str = None) -> Dict:
        """
        Reviews a proposed trade and returns a modified or approved instruction.
        Now supports sizing checks for both LONG and SHORT entries.
        """
        if self.emergency_stop:
            return {"approved": False, "reason": "Emergency stop active"}
            
        if self.daily_pnl <= -self.max_daily_loss:
            self.logger.warning(f"🚨 DAILY LOSS LIMIT HIT (${self.daily_pnl:.2f} <= -${self.max_daily_loss}). HALTING!")
            self.emergency_stop = True
            return {"approved": False, "reason": "SOUL.md Daily Loss limit exceeded"}
            
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
        
        # Risk Parity Overlay (Inverse Volatility cap adjustment)
        if self.parity_weights and symbol in self.parity_weights:
            avg_weight = 1.0 / len(self.parity_weights)
            # If weight > avg (low vol), cap increases. If weight < avg (high vol), cap decreases.
            parity_multiplier = self.parity_weights[symbol] / avg_weight
            
            # Bound the multiplier between 0.5 (double risk) and 2.0 (half risk) so it doesn't go crazy
            parity_multiplier = max(0.5, min(2.0, parity_multiplier))
            
            dynamic_long_cap *= parity_multiplier
            dynamic_short_cap *= parity_multiplier
        
        # MARL Sector Swarm Overlay (Sharpe Ratio Cap Adjustment)
        if sector and sector in self.sector_allocations:
            sector_multiplier = self.sector_allocations[sector]
            dynamic_long_cap *= sector_multiplier
            dynamic_short_cap *= sector_multiplier

        if self.cvar_override:
            dynamic_long_cap *= 0.5
            dynamic_short_cap *= 0.5
        
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

    def update_daily_pnl(self, realized_pnl: float):
        self.daily_pnl += realized_pnl
        self.logger.info(f"Daily PnL updated tracking: ${self.daily_pnl:.2f}")

    def set_emergency_stop(self, state: bool):
        self.emergency_stop = state
        self.logger.info(f"RiskManager: Emergency stop set to {state}")
