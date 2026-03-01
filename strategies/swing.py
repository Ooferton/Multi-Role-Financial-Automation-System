from collections import defaultdict
from typing import Dict, Optional, List
import logging
from agents.trading_agent import TradingStrategy
from data.feature_store import MarketTick
from core.broker_interface import BrokerInterface
from ml.quant_models import calculate_kelly_fraction

class SwingStrategy(TradingStrategy):
    """
    Swing / Multi-day Momentum Strategy.
    
    Logic:
    1. Focuses on larger price movements over multiday or 4H periods.
    2. Uses Moving Averages and trailing stops to capture longer trends.
    3. Less sensitive than HFT, holds positions longer to avoid shakeouts.
    """
    
    def __init__(self, name: str, config: Dict, broker: BrokerInterface = None):
        super().__init__(name, config)
        self.broker = broker
        self.logger = logging.getLogger(__name__)
        
        # History for momentum calc
        self.history = defaultdict(list)
        # Swing usually looks at more bars (e.g. 100 1-hour bars or similar)
        self.window_size = config.get("window_size", 100)
        
        # Aggressive Settings
        self.buy_threshold_pct = config.get("buy_threshold_pct", 0.015)   # 1.5% momentum to buy
        self.stop_loss_pct = config.get("stop_loss_pct", 0.05)            # 5% stop loss
        self.take_profit_pct = config.get("take_profit_pct", 0.10)        # 10% target
        
        # Liquidity Protection Limits
        self.max_open_positions = config.get("max_open_positions", 3)     # Hold max 3 swings at once
        self.base_risk_pct = config.get("risk_per_trade_pct", 0.05)       # Base 5% risk if no history
        self.cash_reserve_pct = config.get("cash_reserve_pct", 0.10)      # Always keep 10% cash floor
        
        self.positions = {} # symbol -> { 'entry_price': float, 'qty': int }
        
        # Performance Tracking for Kelly Criterion
        self.trade_history_pnl = [] # Store PnL percentages

    def _record_trade(self, pnl_pct: float):
        self.trade_history_pnl.append(pnl_pct)
        if len(self.trade_history_pnl) > 50:
            self.trade_history_pnl.pop(0)

    def _get_kelly_fraction(self) -> float:
        if len(self.trade_history_pnl) < 5:
            return self.base_risk_pct
            
        wins = [p for p in self.trade_history_pnl if p > 0]
        losses = [p for p in self.trade_history_pnl if p <= 0]
        
        win_rate = len(wins) / len(self.trade_history_pnl)
        
        avg_win = sum(wins) / len(wins) if wins else 0.01
        avg_loss = abs(sum(losses) / len(losses)) if losses else 0.01
        
        win_loss_ratio = avg_win / avg_loss
        
        # Max out Kelly at 30% of portfolio, Half-Kelly for safety
        kelly_pct = calculate_kelly_fraction(win_rate, win_loss_ratio, fraction_multiplier=0.5)
        # Bounded between base risk and 30%
        return max(self.base_risk_pct, min(0.30, kelly_pct))

    def on_tick(self, tick: MarketTick) -> Optional[Dict]:
        """
        Evaluate swing setups based on rolling history.
        """
        sym = tick.symbol
        self.history[sym].append(tick)
        
        if len(self.history[sym]) > self.window_size:
            self.history[sym].pop(0)
            
        if len(self.history[sym]) < self.window_size:
            return None # Not enough data
            
        prices = [t.price for t in self.history[sym]]
        ma = sum(prices) / len(prices)
        current_price = prices[-1]
        
        momentum_pct = (current_price - ma) / ma
        
        # Manage existing positions
        if sym in self.positions:
            entry = self.positions[sym]['entry_price']
            qty = self.positions[sym]['qty']
            pnl_pct = (current_price - entry) / entry
            
            # Stop Loss
            if pnl_pct <= -self.stop_loss_pct:
                del self.positions[sym]
                self._record_trade(pnl_pct)
                return {
                    "action": "SELL",
                    "symbol": sym,
                    "type": "MARKET",
                    "quantity": qty,
                    "reason": f"Swing Stop Loss Triggered ({pnl_pct:.2%})"
                }
                
            # Take Profit
            if pnl_pct >= self.take_profit_pct:
                del self.positions[sym]
                self._record_trade(pnl_pct)
                return {
                    "action": "SELL",
                    "symbol": sym,
                    "type": "MARKET",
                    "quantity": qty,
                    "reason": f"Swing Take Profit Reached ({pnl_pct:.2%})"
                }
            
            # Momentum breakdown
            if momentum_pct < -self.buy_threshold_pct:
                del self.positions[sym]
                self._record_trade(pnl_pct)
                return {
                    "action": "SELL",
                    "symbol": sym,
                    "type": "MARKET",
                    "quantity": qty,
                    "reason": f"Swing Momentum Breakdown ({momentum_pct:.2%})"
                }
                
            return None

        # Look for new entries
        if momentum_pct > self.buy_threshold_pct:
             # Liquidity Protection 1: Max Open Positions
             if len(self.positions) >= self.max_open_positions:
                 return None
                 
             if self.broker:
                 try:
                     summary = self.broker.get_account_summary()
                     bp = summary.get('buying_power', 0)
                     equity = summary.get('equity', bp) # fallback to bp if equity missing
                     
                     # Liquidity Protection 2: Cash Reserve Floor
                     cash = summary.get('cash', bp)
                     required_reserve = equity * self.cash_reserve_pct
                     if cash <= required_reserve:
                         self.logger.info(f"Swing Veto: Cash ${cash:.2f} is below 10% reserve (${required_reserve:.2f})")
                         return None
                     
                     # Liquidity Protection 3: Kelly Sizing
                     cost_per_share = current_price
                     kelly_pct = self._get_kelly_fraction()
                     trade_value = equity * kelly_pct
                     # But don't exceed what we actually have available
                     trade_value = min(trade_value, bp)
                     
                     qty = int(trade_value // cost_per_share)
                     
                     if qty <= 0:
                         return None
                     
                     self.positions[sym] = {'entry_price': current_price, 'qty': qty}
                     return {
                        "action": "BUY",
                        "symbol": sym,
                        "type": "MARKET",
                        "quantity": qty,
                        "reason": f"Swing Breakout (+{momentum_pct:.2%}) | Kelly Sizing {kelly_pct:.1%}"
                    }
                 except Exception as e:
                     self.logger.error(f"Error checking BP for Swing trade: {e}")
                     return None
            
        return None

    def on_bar(self, bar: Dict) -> Optional[Dict]:
        return None
