"""
Self-Modifying Strategy Engine.

The AI can adjust its own trading parameters and rules at runtime
based on performance feedback, market regime detection, and indicator signals.

Safety: All modifications are bounded within safe ranges and logged.
"""
import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional

@dataclass
class StrategyGenome:
    """
    Mutable parameters that the AI can evolve on the fly.
    Every field has a safe min/max to prevent catastrophic self-modification.
    """
    # --- Signal Thresholds ---
    buy_threshold: float = 0.2       # Action value above this triggers BUY
    sell_threshold: float = -0.2     # Action value below this triggers SELL
    
    # --- Position Sizing ---
    max_position_pct: float = 0.5    # Max % of cash per trade
    conviction_multiplier: float = 1.0  # Scales position by conviction
    
    # --- Risk Management ---
    stop_loss_pct: float = 0.05      # Exit if position drops this much
    take_profit_pct: float = 0.10    # Exit if position gains this much
    max_drawdown_pct: float = 0.15   # Pause trading if drawdown exceeds this
    
    # --- Indicator Weights (how much each factor matters) ---
    rsi_weight: float = 1.0
    macd_weight: float = 1.0
    trend_weight: float = 1.0
    volatility_weight: float = 1.0
    
    # --- Meta ---
    generation: int = 0              # How many times this genome has been modified
    last_modified: str = ""
    modification_log: List[str] = field(default_factory=list)

# Safe bounds for each parameter
GENOME_BOUNDS = {
    'buy_threshold':        (0.05, 0.8),
    'sell_threshold':       (-0.8, -0.05),
    'max_position_pct':     (0.05, 0.8),
    'conviction_multiplier':(0.1, 3.0),
    'stop_loss_pct':        (0.01, 0.20),
    'take_profit_pct':      (0.02, 0.50),
    'max_drawdown_pct':     (0.05, 0.30),
    'rsi_weight':           (0.0, 3.0),
    'macd_weight':          (0.0, 3.0),
    'trend_weight':         (0.0, 3.0),
    'volatility_weight':    (0.0, 3.0),
}

class StrategyEvolver:
    """
    Allows the AI to modify its own strategy parameters based on performance.
    
    Key safety features:
    - All parameters are bounded within safe ranges
    - Every modification is logged with reasoning
    - Can rollback to any previous generation
    """
    
    def __init__(self, genome_path: str = "ml/strategy_genome.json"):
        self.logger = logging.getLogger(__name__)
        self.genome_path = genome_path
        self.genome = self._load_or_create()
        self.history: List[StrategyGenome] = []
    
    def _load_or_create(self) -> StrategyGenome:
        if os.path.exists(self.genome_path):
            try:
                with open(self.genome_path, 'r') as f:
                    data = json.load(f)
                    # Remove non-init fields before constructing
                    data.pop('modification_log', None)
                    genome = StrategyGenome(**{k: v for k, v in data.items() 
                                              if k in StrategyGenome.__dataclass_fields__})
                    self.logger.info(f"Loaded genome generation {genome.generation}")
                    return genome
            except Exception as e:
                self.logger.warning(f"Failed to load genome: {e}. Creating new.")
        return StrategyGenome()
    
    def save(self):
        os.makedirs(os.path.dirname(self.genome_path) or '.', exist_ok=True)
        with open(self.genome_path, 'w') as f:
            json.dump(asdict(self.genome), f, indent=2, default=str)
    
    def mutate(self, param: str, new_value: float, reason: str) -> bool:
        """
        Safely modify a single parameter.
        Returns True if modification was accepted.
        """
        if param not in GENOME_BOUNDS:
            self.logger.warning(f"Unknown parameter: {param}")
            return False
        
        low, high = GENOME_BOUNDS[param]
        clamped = max(low, min(high, new_value))
        
        old_value = getattr(self.genome, param)
        if abs(old_value - clamped) < 1e-6:
            return False  # No change
        
        # Save history for rollback
        self.history.append(StrategyGenome(**asdict(self.genome)))
        
        # Apply mutation
        setattr(self.genome, param, clamped)
        self.genome.generation += 1
        self.genome.last_modified = datetime.now().isoformat()
        
        log_entry = f"Gen {self.genome.generation}: {param} {old_value:.4f} → {clamped:.4f} | {reason}"
        self.genome.modification_log.append(log_entry)
        
        # Keep log manageable
        if len(self.genome.modification_log) > 50:
            self.genome.modification_log = self.genome.modification_log[-50:]
        
        self.logger.info(f"🧬 MUTATION: {log_entry}")
        self.save()
        return True
    
    def adapt(self, performance: Dict) -> List[str]:
        """
        Automatically adapt strategy based on recent performance.
        Returns list of mutations applied.
        
        Args:
            performance: Dict with keys like 'roi', 'win_rate', 'max_drawdown', 
                        'avg_trade_duration', 'total_trades', 'consecutive_losses'
        """
        mutations = []
        
        roi = performance.get('roi', 0)
        win_rate = performance.get('win_rate', 0.5)
        max_dd = performance.get('max_drawdown', 0)
        consecutive_losses = performance.get('consecutive_losses', 0)
        total_trades = performance.get('total_trades', 0)
        
        # --- Drawdown Protection ---
        if abs(max_dd) > self.genome.max_drawdown_pct:
            # Tighten risk
            if self.mutate('max_position_pct', self.genome.max_position_pct * 0.8,
                          f"Drawdown {max_dd:.1%} exceeded limit. Reducing position size."):
                mutations.append("Reduced position size due to drawdown")
            if self.mutate('stop_loss_pct', self.genome.stop_loss_pct * 0.8,
                          f"Tightening stop loss after {max_dd:.1%} drawdown"):
                mutations.append("Tightened stop loss")
        
        # --- Win Rate Adaptation ---
        if total_trades > 20:
            if win_rate < 0.35:
                # Too many losing trades, raise threshold (be more selective)
                if self.mutate('buy_threshold', self.genome.buy_threshold * 1.2,
                              f"Win rate {win_rate:.0%} too low. Raising entry bar."):
                    mutations.append("Raised buy threshold (more selective)")
            elif win_rate > 0.65:
                # Very high win rate, can be slightly more aggressive
                if self.mutate('buy_threshold', self.genome.buy_threshold * 0.9,
                              f"Win rate {win_rate:.0%} strong. Lowering entry bar."):
                    mutations.append("Lowered buy threshold (more aggressive)")
        
        # --- Consecutive Loss Protection ---
        if consecutive_losses >= 5:
            if self.mutate('conviction_multiplier', self.genome.conviction_multiplier * 0.7,
                          f"{consecutive_losses} consecutive losses. Reducing conviction."):
                mutations.append("Reduced conviction after losing streak")
        elif consecutive_losses == 0 and total_trades > 10:
            # Winning streak, slowly restore conviction
            if self.genome.conviction_multiplier < 1.0:
                if self.mutate('conviction_multiplier', 
                              min(1.0, self.genome.conviction_multiplier * 1.1),
                              "Winning streak. Restoring conviction."):
                    mutations.append("Restored conviction after wins")
        
        # --- Profitability Tuning ---
        if roi > 0.05 and total_trades > 20:
            # Profitable, widen take-profit to let winners run
            if self.mutate('take_profit_pct', self.genome.take_profit_pct * 1.1,
                          f"ROI {roi:.1%} positive. Letting winners run."):
                mutations.append("Widened take-profit target")
        elif roi < -0.03 and total_trades > 20:
            # Losing money, quick exits
            if self.mutate('take_profit_pct', self.genome.take_profit_pct * 0.85,
                          f"ROI {roi:.1%} negative. Taking profits faster."):
                mutations.append("Tightened take-profit for faster exits")
        
        if mutations:
            self.logger.info(f"🧬 ADAPTATION: {len(mutations)} mutations applied")
        
        return mutations
    
    def rollback(self, generations: int = 1) -> bool:
        """Roll back to a previous generation."""
        for _ in range(generations):
            if not self.history:
                self.logger.warning("No history to rollback to")
                return False
            self.genome = self.history.pop()
        
        self.save()
        self.logger.info(f"Rolled back to generation {self.genome.generation}")
        return True
    
    def get_status(self) -> str:
        """Human-readable status of current genome."""
        g = self.genome
        return (
            f"Strategy Genome (Gen {g.generation})\n"
            f"  Buy/Sell Thresholds: {g.buy_threshold:.2f} / {g.sell_threshold:.2f}\n"
            f"  Position Size: {g.max_position_pct:.0%} × {g.conviction_multiplier:.1f}x conviction\n"
            f"  Stop Loss / Take Profit: {g.stop_loss_pct:.1%} / {g.take_profit_pct:.1%}\n"
            f"  Max Drawdown Limit: {g.max_drawdown_pct:.1%}\n"
            f"  Indicator Weights: RSI={g.rsi_weight:.1f} MACD={g.macd_weight:.1f} "
            f"Trend={g.trend_weight:.1f} Vol={g.volatility_weight:.1f}\n"
            f"  Last Modified: {g.last_modified or 'Never'}"
        )
