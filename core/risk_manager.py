import logging
from typing import Dict, Optional
from datetime import datetime

class RiskManager:
    """
    Enforces hard risk limits and safety constraints for the financial system.
    This module is the 'brake system' that cannot be overridden by ML models.
    """
    def __init__(self, config: Dict):
        self.config = config
        self.max_daily_loss = config.get('risk', {}).get('max_daily_loss', 1000.0)
        self.max_leverage = config.get('risk', {}).get('max_leverage', 1.0)
        self.max_drawdown_pct = config.get('risk', {}).get('max_drawdown_percent', 0.02)
        
        self.current_daily_loss = 0.0
        self.is_halted = False
        self.logger = logging.getLogger(__name__)

    def check_trade_risk(self, proposed_trade: Dict, current_portfolio_state: Dict) -> bool:
        """
        Evaluates if a proposed trade violates any hard risk constraints.
        
        Args:
            proposed_trade: Dict containing trade details (action, symbol, quantity, price)
            current_portfolio_state: Dict containing current positions, equity, and margin used
            
        Returns:
            bool: True if trade is SAFE, False if it violates risk limits.
        """
        if self.is_halted:
            self.logger.warning("Risk Manager: System is HALTED. Trade rejected.")
            return False

        # 1. Check Max Daily Loss
        if self.current_daily_loss >= self.max_daily_loss:
            self.logger.error(f"Risk Manager: Max daily loss exceeded ({self.current_daily_loss} >= {self.max_daily_loss}). Halting system.")
            self.halt_system("Max Daily Loss Exceeded")
            return False

        # 2. Check Leverage Constraint
        projected_leverage = self._calculate_projected_leverage(proposed_trade, current_portfolio_state)
        if projected_leverage > self.max_leverage:
            self.logger.warning(f"Risk Manager: Leverage limit reached ({projected_leverage:.2f} > {self.max_leverage}). Trade rejected.")
            return False

        return True

    def update_daily_pnl(self, pnl: float):
        """
        Updates the daily PnL tracker. Should be called after every closed trade or mark-to-market update.
        """
        if pnl < 0:
            self.current_daily_loss += abs(pnl)
        
        # Check if updated loss triggers a halt
        if self.current_daily_loss >= self.max_daily_loss:
            self.halt_system("Max Daily Loss Exceeded after PnL update")

    def halt_system(self, reason: str):
        """
        Triggers a system-wide halt.
        """
        self.is_halted = True
        self.logger.critical(f"SYSTEM HALTED: {reason}")
        # In a real system, this would trigger emergency liquidation or order cancellation logic

    def _calculate_projected_leverage(self, trade: Dict, state: Dict) -> float:
        """
        Calculates what the account leverage would be if the trade executes.
        Accounts for BOTH Long and Short exposure (Gross Exposure).
        """
        total_equity = state.get('equity', 100000)
        current_market_value = state.get('market_value', 0)
        
        # We use absolute value because shorting also increases exposure/risk
        trade_value = abs(trade.get('price', 0) * trade.get('quantity', 0))
        
        # Gross Exposure = Absolute Market Value of Longs + Absolute Market Value of Shorts
        new_gross_exposure = abs(current_market_value) + trade_value
        
        return new_gross_exposure / total_equity if total_equity > 0 else 999.0
