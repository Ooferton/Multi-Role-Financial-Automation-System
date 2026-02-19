"""
PPO Trading Real v2 Strategy.

Uses all 10 market indicators for richer signal processing.
"""
from typing import Dict, Optional, Any
import numpy as np
import logging
from agents.trading_agent import TradingStrategy
from core.broker_interface import BrokerInterface
from ml.rl_agent_v2 import RLAgentV2
from ml.reasoning_engine import ReasoningEngine
from ml.strategy_evolver import StrategyEvolver
from data.feature_store import MarketTick
from ml.features import TechnicalIndicators
import pandas as pd

class RLStrategyV2(TradingStrategy):
    """
    Self-modifying RL trading strategy v2 with 10 market indicators.
    
    Uses:
    - RLAgentV2 for numeric action prediction (14-dim observation)
    - ReasoningEngine for human-readable explanations
    - StrategyEvolver for adaptive parameter tuning
    
    Indicators used (10):
    1. RSI (14)          - Momentum
    2. MACD              - Momentum  
    3. MACD Signal       - Momentum
    4. BB Width          - Volatility
    5. Dist SMA 20       - Trend
    6. MACD Histogram    - Momentum (NEW)
    7. BB Position       - Volatility (NEW)
    8. Dist SMA 50       - Trend (NEW)
    9. ATR Normalized    - Volatility (NEW)
    10. OBV Normalized   - Volume (NEW)
    """
    def __init__(self, name: str, config: Dict, broker: BrokerInterface, model_path: str):
        super().__init__(name, config)
        self.broker = broker
        self.agent = RLAgentV2(model_path)
        self.reasoner = ReasoningEngine()
        self.evolver = StrategyEvolver()
        self.logger = logging.getLogger(__name__)
        
        self.last_price = 0.0
        self.history_buffer = [] 
        self.min_history = 50
        
        # Performance tracking for self-modification
        self.trade_count = 0
        self.win_count = 0
        self.loss_count = 0
        self.consecutive_losses = 0
        self.peak_equity = 0.0
        self.entry_prices: Dict[str, float] = {}
        
        self.logger.info(f"Strategy v2 initialized with genome:\n{self.evolver.get_status()}")

    def on_tick(self, tick: MarketTick) -> Optional[Dict]:
        price = tick.price
        genome = self.evolver.genome
        
        # 1. Update Buffer
        self.history_buffer.append({
            'timestamp': tick.timestamp,
            'close': price,
            'high': price,
            'low': price,
            'volume': tick.size
        })
        
        if len(self.history_buffer) > 300:
            self.history_buffer.pop(0)
            
        if len(self.history_buffer) < self.min_history:
            return None
            
        # 2. Compute ALL Indicators (10 total)
        df = pd.DataFrame(self.history_buffer)
        df = TechnicalIndicators.add_all_features(df)
        current_row = df.iloc[-1]
        
        # 3. Calculate Returns
        ret = 0.0
        if self.last_price > 0:
            ret = (price - self.last_price) / self.last_price
        self.last_price = price
        
        # 4. Get Portfolio State
        summary = self.broker.get_account_summary()
        positions = self.broker.get_positions()
        
        cash = summary.get('cash', 0.0)
        equity = summary.get('equity', 0.0)
        current_qty = 0
        for p in positions:
            if p.symbol == tick.symbol:
                current_qty = p.qty
                break
        
        # Track peak equity for drawdown
        self.peak_equity = max(self.peak_equity, equity) if self.peak_equity > 0 else equity
        current_drawdown = (equity - self.peak_equity) / self.peak_equity if self.peak_equity > 0 else 0
        
        # 5. Extract all 10 indicators
        indicators = {
            'rsi_14': current_row['rsi_14'],
            'macd': current_row['macd'],
            'macd_signal': current_row['macd_signal'],
            'bb_width': current_row['bb_width'],
            'dist_sma_20': current_row['dist_sma_20'],
            # NEW v2 indicators
            'macd_histogram': current_row['macd_histogram'],
            'bb_position': current_row['bb_position'],
            'dist_sma_50': current_row['dist_sma_50'],
            'atr_norm': current_row['atr_norm'],
            'obv_norm': current_row['obv_norm'],
        }
        
        # 6. Check Stop Loss / Take Profit (self-modifying thresholds)
        if tick.symbol in self.entry_prices and current_qty > 0:
            entry = self.entry_prices[tick.symbol]
            pnl_pct = (price - entry) / entry
            
            if pnl_pct <= -genome.stop_loss_pct:
                reasoning = self.reasoner.explain_trade_v2(
                    -1.0, indicators,
                    {'cash': cash, 'position_qty': current_qty},
                    tick.symbol
                )
                reasoning.summary = f"STOP LOSS triggered at {pnl_pct:.1%} (limit: {genome.stop_loss_pct:.1%})"
                reasoning.risk_notes.append(f"Auto-exit: loss exceeded {genome.stop_loss_pct:.1%} threshold")
                self._record_loss()
                
                return {
                    "action": "SELL", "symbol": tick.symbol,
                    "quantity": current_qty, "type": "MARKET",
                    "price": price, "reason": reasoning.summary, "reasoning": reasoning
                }
            
            elif pnl_pct >= genome.take_profit_pct:
                reasoning = self.reasoner.explain_trade_v2(
                    -0.5, indicators,
                    {'cash': cash, 'position_qty': current_qty},
                    tick.symbol
                )
                reasoning.summary = f"TAKE PROFIT at {pnl_pct:.1%} (target: {genome.take_profit_pct:.1%})"
                self._record_win()
                
                return {
                    "action": "SELL", "symbol": tick.symbol,
                    "quantity": current_qty, "type": "MARKET",
                    "price": price, "reason": reasoning.summary, "reasoning": reasoning
                }
        
        # 7. Check drawdown circuit breaker
        if abs(current_drawdown) > genome.max_drawdown_pct:
            self.logger.warning(
                f"⚠️ CIRCUIT BREAKER: Drawdown {current_drawdown:.1%} "
                f"exceeds {genome.max_drawdown_pct:.1%}. Pausing new trades."
            )
            return None
                
        # 8. Construct 14-dim Observation Vector (10 weighted indicators)
        obs = np.array([
            price, ret,
            indicators['rsi_14'] * genome.rsi_weight,
            indicators['macd'] * genome.macd_weight,
            indicators['macd_signal'] * genome.macd_weight,
            indicators['bb_width'] * genome.volatility_weight,
            indicators['dist_sma_20'] * genome.trend_weight,
            # NEW v2 indicators with genome weights
            indicators['macd_histogram'] * genome.histogram_weight,
            indicators['bb_position'] * genome.bb_position_weight,
            indicators['dist_sma_50'] * genome.sma50_weight,
            indicators['atr_norm'] * genome.atr_weight,
            indicators['obv_norm'] * genome.obv_weight,
            float(current_qty), float(cash)
        ], dtype=np.float32)
        
        # 9. Ask the AI
        action = self.agent.predict(obs)
        action_val = float(action[0])
        
        # 10. Generate Reasoning (using all 10 indicators)
        reasoning = self.reasoner.explain_trade_v2(
            action_val=action_val,
            indicators=indicators,
            portfolio={'cash': cash, 'position_qty': current_qty},
            symbol=tick.symbol
        )
        
        # 11. Apply genome thresholds and sizing
        trade_fraction = min(abs(action_val), 1.0) * genome.conviction_multiplier
        trade_value = cash * min(trade_fraction, 1.0) * genome.max_position_pct
        quantity = round(trade_value / price, 4) if price > 0 else 0
        
        if quantity <= 0:
            return None
        
        if action_val > genome.buy_threshold:
            self.entry_prices[tick.symbol] = price
            self.logger.info(f"\n{'='*60}\n{reasoning}\n{'='*60}")
            return {
                "action": "BUY", "symbol": tick.symbol,
                "quantity": quantity, "type": "MARKET",
                "price": price, "reason": reasoning.summary, "reasoning": reasoning
            }
        elif action_val < genome.sell_threshold:
            if tick.symbol in self.entry_prices:
                if price > self.entry_prices[tick.symbol]:
                    self._record_win()
                else:
                    self._record_loss()
            
            self.logger.info(f"\n{'='*60}\n{reasoning}\n{'='*60}")
            return {
                "action": "SELL", "symbol": tick.symbol,
                "quantity": quantity, "type": "MARKET",
                "price": price, "reason": reasoning.summary, "reasoning": reasoning
            }
            
        return None
    
    def _record_win(self):
        self.trade_count += 1
        self.win_count += 1
        self.consecutive_losses = 0
    
    def _record_loss(self):
        self.trade_count += 1
        self.loss_count += 1
        self.consecutive_losses += 1
    
    def self_adapt(self):
        """Called periodically to let the AI evolve its own strategy."""
        if self.trade_count < 5:
            return []
        
        win_rate = self.win_count / self.trade_count if self.trade_count > 0 else 0.5
        current_dd = 0
        
        summary = self.broker.get_account_summary()
        equity = summary.get('equity', 0)
        if self.peak_equity > 0:
            current_dd = (equity - self.peak_equity) / self.peak_equity
        
        performance = {
            'roi': (equity - 10000) / 10000 if equity > 0 else 0,
            'win_rate': win_rate,
            'max_drawdown': current_dd,
            'total_trades': self.trade_count,
            'consecutive_losses': self.consecutive_losses
        }
        
        mutations = self.evolver.adapt(performance)
        if mutations:
            self.logger.info(f"🧬 Self-adaptation applied {len(mutations)} mutations")
        return mutations

    def on_bar(self, bar: Dict) -> Optional[Dict]:
        return None
