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
import time
from datetime import datetime
import os

from collections import defaultdict
from agents.risk_manager import RiskManager
from agents.sentinel import Sentinel
from ml.news_sentiment import NewsSentimentEngine
from agents.economist_agent import EconomistAgent
from ml.quant_models import calculate_kelly_fraction

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
    def __init__(self, name: str, config: Dict[str, Any], broker: BrokerInterface, model_path: str):
        super().__init__(name, config)
        self.broker = broker
        self.agent = RLAgentV2(model_path)
        self.reasoner = ReasoningEngine()
        self.evolver = StrategyEvolver()
        self.genome = self.evolver.genome # Load the strategy genome
        self.logger = logging.getLogger(__name__)
        
        # Sentience Layer Agents
        self.risk_manager = RiskManager(config)
        self.sentinel = Sentinel(config)
        self.news_engine = NewsSentimentEngine()
        self.economist = EconomistAgent(config)
        
        self.journal_path = "logs/ai_journal.md"
        self._last_journal_time = 0.0
        self._last_export_time = 0.0
        self._last_macro_time = 0.0
        self.macro_outlook = {}
        
        self.last_prices = {}
        self.history_buffers = defaultdict(list)
        self.min_history = 50
        
        # Performance tracking for self-modification
        self.trade_count = 0
        self.win_count = 0
        self.loss_count = 0
        self.consecutive_losses = 0
        self.peak_equity = 0.0
        self.entry_prices: Dict[str, float] = {}
        self.entry_sides: Dict[str, str] = {} # {symbol: 'LONG'|'SHORT'}
        self.live_activity = {} # {symbol: {status: str, detail: str, timestamp: str}}
        self._last_api_sync_time = 0.0
        self._cached_summary = {}
        self._cached_positions = []
        
        self.trade_history_pnl = [] # Track PnL for Kelly Sizing
        
        self.logger.info(f"Strategy v2 initialized with genome:\n{self.evolver.get_status()}")
        
        # Determine status file path based on mode
        is_btm = config.get('system', {}).get('crypto_etf_only', False)
        self.status_path = "data/sentience_bitcoin.json" if is_btm else "data/sentience_status.json"
        
        self._sync_positions()

    def _sync_positions(self):
        """
        Sync local state with broker's actual positions.
        """
        try:
            positions = self.broker.get_positions()
            for p in positions:
                self.entry_prices[p.symbol] = float(p.avg_entry_price)
                self.entry_sides[p.symbol] = 'LONG' if p.qty > 0 else 'SHORT'
                self.logger.info(f"Synced position for {p.symbol}: {p.qty} @ ${p.avg_entry_price} ({self.entry_sides[p.symbol]})")
        except Exception as e:
            self.logger.error(f"Failed to sync positions: {e}")

    def on_tick(self, tick: MarketTick) -> Optional[Dict]:
        """Core logic for v2 tick processing."""
        # 0. Periodic Macro Update (Every 15 minutes)
        if time.time() - self._last_macro_time > 900:
            self.macro_outlook = self.economist.update_outlook()
            self._last_macro_time = time.time()
            self.logger.info(f"Macro Outlook: {self.macro_outlook.get('summary')}")
            # Ensure macro updates reach the dashboard immediately
            self._update_sentience_export({}, None)

        price = tick.price
        symbol = tick.symbol
        genome = self.evolver.genome
        is_btm = self.config.get('system', {}).get('crypto_etf_only', False)
        
        # 1. Update Buffer (Per Symbol)
        self.history_buffers[symbol].append({
            'timestamp': tick.timestamp,
            'close': price,
            'high': price,
            'low': price,
            'volume': tick.size,
            'bid_size': getattr(tick, 'bid_size', 0.0),
            'ask_size': getattr(tick, 'ask_size', 0.0)
        })
        
        if len(self.history_buffers[symbol]) > 300:
            self.history_buffers[symbol].pop(0)
            
        if len(self.history_buffers[symbol]) < self.min_history:
            # Still update dashboard news/macro even during warm-up
            try:
                sentiment = self.news_engine.get_sentiment(symbol)
                self.live_activity[symbol] = {
                    "status": "WARMUP",
                    "detail": f"Gathering history ({len(self.history_buffers[symbol])}/{self.min_history})",
                    "timestamp": datetime.now().strftime("%H:%M:%S")
                }
                self._update_sentience_export({}, sentiment)
            except: pass
            return None
            
        # 2. Compute ALL Indicators (10 total)
        df = pd.DataFrame(self.history_buffers[symbol])
        df = TechnicalIndicators.add_all_features(df)
        current_row = df.iloc[-1]
        
        # 3. Calculate Returns
        ret = 0.0
        previous_price = self.last_prices.get(symbol, 0.0)
        if previous_price > 0:
            ret = (price - previous_price) / previous_price
        self.last_prices[symbol] = price
        
        # 4. Get Portfolio State (Rate limited to avoid 429)
        current_time = time.time()
        if current_time - self._last_api_sync_time < 3.0:
            summary = self._cached_summary
            positions = self._cached_positions
        else:
            summary = self.broker.get_account_summary()
            positions = self.broker.get_positions()
            self._cached_summary = summary
            self._cached_positions = positions
            self._last_api_sync_time = current_time
        
        # Use BUYING_POWER instead of cash for new trades
        buying_power = summary.get('buying_power', 0.0)
        
        # Still track cash/equity for reasoning/stats
        cash = summary.get('cash', 0.0)
        equity = summary.get('equity', 0.0)
        current_qty = 0
        for p in positions:
            if p.symbol == tick.symbol:
                current_qty = p.qty
                # Ensure entry price is synced if missed during init
                if symbol not in self.entry_prices:
                    self.entry_prices[symbol] = float(p.avg_entry_price)
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
        
        # 10. Sentience Layer: Risk Protections
        # 1. Check stop-loss / take-profit
        if symbol in self.entry_prices and current_qty != 0: # Only check if we have an open position
            entry_price = self.entry_prices[symbol]
            side = self.entry_sides.get(symbol, 'LONG') # Default to LONG if not set
            
            if side == 'LONG':
                pnl_pct = (price - entry_price) / entry_price
            else: # SHORT
                pnl_pct = (entry_price - price) / entry_price
                
            # --- AGGRESSIVE BITCOIN RULE: ZERO-TOLERANCE STOP LOSS ---
            if is_btm and pnl_pct < -0.0005: # -0.05%
                self.logger.warning(f"🔺 AGGRESSIVE STOP LOSS triggered for {symbol}: {pnl_pct:.3%} (Zero-Tolerance Mode)")
                return self._close_position(symbol, price, current_qty, "AGGRESSIVE STOP LOSS")

            if pnl_pct < -genome.stop_loss_pct:
                self.logger.warning(f"STOP LOSS triggered for {symbol} ({side}): {pnl_pct:.2%}")
                return self._close_position(symbol, price, current_qty, "STOP LOSS")
            if pnl_pct > genome.take_profit_pct:
                self.logger.info(f"TAKE PROFIT triggered for {symbol} ({side}): {pnl_pct:.2%}")
                return self._close_position(symbol, price, current_qty, "TAKE PROFIT")
        
        # 7. Check drawdown circuit breaker
        if abs(current_drawdown) > genome.max_drawdown_pct:
            self.logger.critical(f"Circuit Breaker: Global drawdown {current_drawdown:.1%} exceeds limit! Closing all positions.")
            return self._close_position(symbol, price, current_qty, "CIRCUIT BREAKER")
                
        # 6. Macro & Crisis Detection (Moved up for Sentinel access)
        if current_time - self._last_macro_time > 60:
            self.macro_outlook = self.economist.update_outlook()
            self._last_macro_time = current_time
        
        severity = self.economist.get_crisis_severity()

        # 8. Construct 14-dim Observation Vector (10 weighted indicators)
        obs_base = [
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
        ]
        
        expected_obs_dim = 14
        if self.agent.model and hasattr(self.agent.model.observation_space, 'shape'):
            expected_obs_dim = self.agent.model.observation_space.shape[0]

        if expected_obs_dim >= 22:
            # 10. Hearing Layer: News Sentiment (V3 NLP state)
            sentiment_label = self.news_engine.get_sentiment(tick.symbol)
            sentiment_score = 0.0
            if "BULLISH" in sentiment_label: sentiment_score = 0.8
            elif "BEARISH" in sentiment_label: sentiment_score = -0.8
            
            # V3 Regime State using existing indicators
            current_vol = indicators['atr_norm']
            is_bull, is_bear, is_choppy = 0.0, 0.0, 1.0
            if 'sma_200' in current_row:
                sma_200 = current_row['sma_200']
                if price > sma_200 and current_vol < 0.02:
                    is_bull = 1.0; is_choppy = 0.0
                elif price < sma_200 and current_vol > 0.02:
                    is_bear = 1.0; is_choppy = 0.0
            
            # Simulated LLM Risk Constraints from Risk Manager
            max_leverage = getattr(self.risk_manager, 'max_leverage', 1.0)
            max_position_frac = getattr(self.risk_manager, 'max_portfolio_risk', 0.1)
            allow_shorting = 1.0 # default
            in_crisis = 1.0 if severity == "CRITICAL" else 0.0
            
            # Order Flow Imbalance
            bid_size = current_row.get('bid_size', 0.0)
            ask_size = current_row.get('ask_size', 0.0)
            if (bid_size + ask_size) > 0:
                ofi = (bid_size - ask_size) / (bid_size + ask_size)
            else:
                ofi = 0.0
            
            v3_features = [
                in_crisis, max_leverage, max_position_frac, allow_shorting,
                sentiment_score, is_bull, is_bear, is_choppy
            ]
            if expected_obs_dim == 23:
                v3_features.append(ofi)
                
            obs_base.extend(v3_features)

        obs = np.array(obs_base, dtype=np.float32)
        
        # 9. Sentience Check: Sentinel (Safety)
        safety = self.sentinel.check_safety(tick.symbol, indicators, crisis_severity=severity)
        if not safety["safe"]:
            self.live_activity[tick.symbol] = {
                "status": "VETO (SAFETY)",
                "detail": safety["reason"],
                "timestamp": datetime.now().strftime("%H:%M:%S")
            }
            self.logger.warning(safety["reason"])
            # In a Black Swan, we pivot to cash if recommended
            if safety.get("recommendation") == "GO_TO_CASH" and current_qty != 0:
                return self._close_position(tick.symbol, price, current_qty, "BLACK SWAN EXIT")
            return None

        # 10. Hearing Layer: News Sentiment (Already calculated above for V3, but we keep it here for Dashboard export logic)
        sentiment = self.news_engine.get_sentiment(tick.symbol)
        
        # 11.5 Always update dashboard sentience metrics
        # Clear stale pulses if mode changed
        last_m = getattr(self, '_last_mode_cache', None)
        if last_m is not None and last_m != is_btm:
            self.logger.info(f"🧹 Mode Switch Detected in Strategy: Clearing {len(self.live_activity)} stale pulses.")
            self.live_activity.clear()
        self._last_mode_cache = is_btm

        self.live_activity[tick.symbol] = {
            "status": "SCANNING",
            "detail": f"Sentiment: {sentiment.get('verdict')} ({sentiment.get('score'):.2f})",
            "timestamp": datetime.now().strftime("%H:%M:%S")
        }
        # 11.6 Periodic Journaling / Dashboard Export
        self._check_journaling(indicators, summary, sentiment)

        # 12. Ask the AI Agent
        action = self.agent.predict(obs)
        action_val = float(action[0])

        # LOUD DIAGNOSTICS for Bitcoin/ETF
        if symbol in ["BTC/USD", "IBIT", "BITO", "FBTC", "ARKB"]:
            self.logger.info(f"🔍 [AI DIAGNOSTIC] {symbol} | Price: {price} | Action: {action_val:.4f} | RSI: {indicators['rsi_14']:.1f} | MACD: {indicators['macd']:.4f}")
            if abs(action_val) < 0.1:
                self.logger.warning(f"⚠️ {symbol} Signal is FLAT (Near 0). AI might be scale-choked by price/MACD.")
        
        # 12.5 Sentiment Sentinel: Action-Aware News Veto
        action_str = "BUY" if action_val > genome.buy_threshold else "SELL" if action_val < genome.sell_threshold else "WAIT"
        if action_str != "WAIT":
            sent_safety = self.sentinel.check_sentiment_safety(sentiment, action=action_str)
            if not sent_safety["safe"]:
                self.live_activity[tick.symbol] = {
                    "status": "VETO (NEWS)",
                    "detail": sent_safety["reason"],
                    "timestamp": datetime.now().strftime("%H:%M:%S")
                }
                self.logger.warning(sent_safety["reason"])
                return None
            
            # --- AGGRESSIVE BITCOIN RULE: MOMENTUM HOLD ---
            if is_btm and action_str == "SELL":
                # Look back at price 1 minute ago (previous bar)
                if len(self.history_buffers[symbol]) > 1:
                    prev_close = self.history_buffers[symbol][-2]['close']
                    if price > prev_close:
                        self.logger.info(f"💎 MOMENTUM HOLD for {symbol}: AI says SELL, but price is RISING (${price:.2f} > ${prev_close:.2f}). Vetoing exit.")
                        self.live_activity[tick.symbol] = {
                            "status": "MOMENTUM HOLD",
                            "detail": f"Vetoing AI SELL while price is climbing",
                            "timestamp": datetime.now().strftime("%H:%M:%S")
                        }
                        return None

        # 12.6 Generate Reasoning
        reasoning = self.reasoner.explain_trade_v2(
            action_val=action_val,
            indicators=indicators,
            portfolio={'cash': cash, 'position_qty': current_qty},
            symbol=tick.symbol,
            sentiment=sentiment,
            macro=self.macro_outlook
        )
        
        # 13. Apply Crisis Alpha Overrides
        severity = self.economist.get_crisis_severity()
        crisis_buy_threshold = genome.buy_threshold
        crisis_sell_threshold = genome.sell_threshold
        
        # If in crisis, we want to be "Short-Happy"
        if severity > 0.5:
            self.logger.warning(f"CRISIS MODE ACTIVE (Severity: {severity:.2f}). Pivoting strategy.")
            crisis_sell_threshold *= 0.7 # Lower threshold to enter shorts faster (e.g., -0.35 -> -0.245)
            # If the symbol is a known hedge (SQQQ, VIX), make it easier to BUY
            if any(h in tick.symbol for h in ["SQQQ", "VIX", "SH", "UVXY"]):
                crisis_buy_threshold *= 0.5 # Buy hedges aggressively
            else:
                crisis_buy_threshold *= 2.0 # Make it very hard to buy normal longs
            
            reasoning.risk_notes.append(f"Crisis Mode: Thresholds adjusted for Bearish Pivot (Severity {severity})")

        # 13.5 Apply genome thresholds and sizing
        trade_fraction = min(abs(action_val), 1.0) * genome.conviction_multiplier
        
        # Crisis Conviction Boost for shorts
        if severity > 0.5 and action_val < 0:
            trade_fraction *= (1.0 + severity) # Boost short size in a crash
            reasoning.risk_notes.append(f"Crisis Alpha: Boosting short conviction by {(severity*100):.0f}%")

        if self.economist.is_risk_off() and severity < 0.5:
            self.logger.info("Macro Risk-Off: Nervous market. Halving position size.")
            trade_fraction *= 0.5
            reasoning.risk_notes.append("Macro sizing override: Half-size due to Risk-Off regime")

        # --- KELLY POSITION SIZING ---
        # Instead of fixed genome.max_position_pct, we calculate Kelly Fraction
        if len(self.trade_history_pnl) >= 5:
            wins = [p for p in self.trade_history_pnl if p > 0]
            losses = [p for p in self.trade_history_pnl if p <= 0]
            
            win_rate = len(wins) / len(self.trade_history_pnl)
            avg_win = sum(wins) / len(wins) if wins else 0.01
            avg_loss = abs(sum(losses) / len(losses)) if losses else 0.01
            win_loss_ratio = avg_win / avg_loss
            
            # Half-Kelly for safety
            kelly_pct = calculate_kelly_fraction(win_rate, win_loss_ratio, fraction_multiplier=0.5)
            # Bound it between a minimum 2% risk and genome max
            base_risk = max(0.02, min(genome.max_position_pct, kelly_pct))
        else:
            base_risk = genome.max_position_pct

        trade_value = buying_power * min(trade_fraction, 1.0) * base_risk
        
        # --- BUYING POWER RESERVATION ---
        # Portfolio Mode (Standard Stocks) must leave at least $15 for Crypto Mode.
        if not is_btm and buying_power < 15.00 and action_val > genome.buy_threshold:
            # Only log if we were actually planning to buy
            self.logger.info(f"🔇 [PORTFOLIO MODE] Standing down. Buying power ${buying_power:.2f} is below $15.00 Crypto Reserve.")
            return None

        # --- CRYPTO-AWARE MINIMUM ORDER FLOOR ---
        # Direct coins (BTC/USD, DOGE/USD) require $10.00 min.
        # Stocks/ETFs (RIOT, IBIT) require $1.00 min.
        is_coin = "/" in tick.symbol
        floor = 10.05 if is_coin else 1.05
        
        if 0 < trade_value < floor:
            if buying_power >= floor:
                self.logger.info(f"📍 Bumping {tick.symbol} trade value from ${trade_value:.2f} to ${floor:.2f} (Broker Minimum Floor)")
                trade_value = floor
            else:
                self.logger.warning(f"⚠️ {tick.symbol} value ${trade_value:.2f} is below Alpaca floor (${floor}), and insufficient buying power to bump.")
                return None

        quantity = round(trade_value / price, 4) if price > 0 else 0
        
        if quantity <= 0:
            return None
        
        # 14. Sentience Check: Risk Manager (Veto/Resize)
        sector = self.name.split('_')[-1] if '_' in self.name else None
        
        risk_review = self.risk_manager.review_trade(
            symbol=tick.symbol,
            action=action_str,
            quantity=quantity,
            price=price,
            portfolio={
                'equity': getattr(self.risk_manager, 'total_equity', 100000), 
                'positions': {tick.symbol: current_qty}
            },
            current_drawdown=0.0,
            crisis_severity=1.0 if severity == "CRITICAL" else 0.0,
            sector=sector
        )
        
        if not risk_review["approved"]:
            self.live_activity[tick.symbol] = {
                "status": "VETO (RISK)",
                "detail": risk_review["reason"],
                "timestamp": datetime.now().strftime("%H:%M:%S")
            }
            self.logger.warning(f"RiskManager VETO: {risk_review['reason']}")
            return None
        
        quantity = risk_review["quantity"]
        reasoning.risk_notes.append(risk_review["reason"])

        # 15. Apply Strategy Signal: BUY (Long or Cover)
        if action_val > crisis_buy_threshold:
            # Check current position
            current_side = self.entry_sides.get(tick.symbol)
            if current_side == 'SHORT':
                # Signal says buy, but we are short -> Close Short
                return self._close_position(tick.symbol, price, current_qty, "SIGNAL FLIP")
            elif current_side == 'LONG':
                # Already long, ignore for now (or could add to position)
                return None
            else:
                # FLAT -> Open LONG
                self.live_activity[tick.symbol] = {
                    "status": "SIGNAL (LONG)",
                    "detail": reasoning.summary,
                    "timestamp": datetime.now().strftime("%H:%M:%S")
                }
                self.entry_prices[tick.symbol] = price
                self.entry_sides[tick.symbol] = 'LONG'
                self.logger.info(f"\n{'='*60}\n{reasoning}\n{'='*60}")
                return {
                    "action": "BUY", "symbol": tick.symbol,
                    "quantity": quantity, "type": "MARKET",
                    "price": price, "reason": reasoning.summary, "reasoning": reasoning
                }

        # 16. Apply Strategy Signal: SELL (Short or Close)
        elif action_val < crisis_sell_threshold:
            # Check current position
            current_side = self.entry_sides.get(tick.symbol)
            if current_side == 'LONG':
                # Signal says sell, and we are long -> Close Long
                return self._close_position(tick.symbol, price, current_qty, "SIGNAL FLIP")
            elif current_side == 'SHORT':
                # Already short, ignore
                return None
            else:
                # FLAT -> Open SHORT
                self.live_activity[tick.symbol] = {
                    "status": "SIGNAL (SHORT)",
                    "detail": reasoning.summary,
                    "timestamp": datetime.now().strftime("%H:%M:%S")
                }
                self.entry_prices[tick.symbol] = price
                self.entry_sides[tick.symbol] = 'SHORT'
                self.logger.info(f"\n{'='*60}\n{reasoning}\n{'='*60}")
                return {
                    "action": "SELL", "symbol": tick.symbol,
                    "quantity": quantity, "type": "MARKET",
                    "price": price, "reason": reasoning.summary, "reasoning": reasoning
                }
            
        return None
    
    def _record_win(self, pnl_pct: float):
        self.trade_count += 1
        self.win_count += 1
        self.consecutive_losses = 0
        self.trade_history_pnl.append(pnl_pct)
        if len(self.trade_history_pnl) > 50: self.trade_history_pnl.pop(0)
    
    def _record_loss(self, pnl_pct: float):
        self.trade_count += 1
        self.loss_count += 1
        self.consecutive_losses += 1
        self.trade_history_pnl.append(pnl_pct)
        if len(self.trade_history_pnl) > 50: self.trade_history_pnl.pop(0)
    
    def self_adapt(self):
        """Called periodically to let the AI evolve its own strategy."""
        if self.trade_count < 5:
            return []
        
        win_rate = self.win_count / self.trade_count if self.trade_count > 0 else 0.5
        current_dd = 0
        
        current_time = time.time()
        if current_time - self._last_api_sync_time < 3.0:
            summary = self._cached_summary
        else:
            summary = self.broker.get_account_summary()
            self._cached_summary = summary
            self._last_api_sync_time = current_time
        
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

    def _close_position(self, symbol: str, price: float, qty: float, reason: str) -> Dict:
        """Helper to close a position and record metrics."""
        entry_price = self.entry_prices.pop(symbol, price)
        side = self.entry_sides.pop(symbol, 'LONG')
        
        # side is the side of the entry. To close:
        # If we were LONG, we SELL.
        # If we were SHORT, we BUY.
        if side == 'LONG':
            is_win = price > entry_price
            pnl_pct = (price - entry_price) / entry_price
            action = "SELL"
        else: # SHORT
            is_win = price < entry_price
            pnl_pct = (entry_price - price) / entry_price
            action = "BUY"
            
        if is_win:
            self._record_win(pnl_pct)
        else:
            self._record_loss(pnl_pct)
            
        self.live_activity[symbol] = {
            "status": f"EXIT ({reason})",
            "detail": f"Closed {side} at {price:.2f} (Profit: {is_win})",
            "timestamp": datetime.now().strftime("%H:%M:%S")
        }
        
        return {
            "action": action, "symbol": symbol,
            "quantity": abs(qty), 
            "type": "MARKET", "price": price, "reason": f"{reason} ({side})"
        }

    def on_bar(self, bar: Dict) -> Optional[Dict]:
        return None

    def _check_journaling(self, indicators: Dict, summary: Dict, sentiment: Optional[Dict] = None):
        """Internal monologue logger."""
        current_time = time.time()
        # Prepare indicators for export/journal
        agg_indicators = {
            'avg_volatility': indicators.get('bb_width', 0),
            'avg_trend': indicators.get('dist_sma_20', 0)
        }

        # Journal every 5 minutes
        if current_time - self._last_journal_time > 300:
            try:
                # Aggregate data for journal
                performance = {
                    'win_rate': self.win_count / self.trade_count if self.trade_count > 0 else 0.5,
                }
                
                # Fetch recent headlines from sentiment dict
                headlines = sentiment.get('headlines', []) if sentiment else []
                
                entry = self.reasoner.generate_journal_entry(
                    agg_indicators, performance, 
                    news_agg=headlines, macro=self.macro_outlook
                )
                
                # Ensure directory exists and write
                os.makedirs(os.path.dirname(self.journal_path), exist_ok=True)
                with open(self.journal_path, "a") as f:
                    f.write(entry.to_markdown())
                
                self.logger.info("AI Journal updated.")
                self._last_journal_time = current_time
            except Exception as e:
                self.logger.error(f"Failed to update AI journal: {e}")

        # 16. Export Sentience Status for Dashboard (Every tick/symbol update)
        self._update_sentience_export(agg_indicators, sentiment)

    def _update_sentience_export(self, indicators: Dict, sentiment: Optional[Dict]):
        """Exports high-frequency metrics for the dashboard with throttling."""
        current_time = time.time()
        if current_time - self._last_export_time < 1.0: # Throttle to 1Hz
            return
            
        try:
            import json
            os.makedirs("data", exist_ok=True)
            
            # Read-Modify-Write to preserve 'pid' and 'active' set by the runner
            curr_data = {}
            if os.path.exists(self.status_path):
                try:
                    with open(self.status_path, "r", encoding="utf-8") as f:
                        curr_data = json.load(f)
                except: pass
            
            status = {
                "active": True,
                "pid": os.getpid(),
                "timestamp": datetime.now().isoformat(),
                "vix": self.macro_outlook.get("metrics", {}).get("VIX", 0),
                "vibe": self.macro_outlook.get("vibe", "Neutral"),
                "macro_summary": self.macro_outlook.get("summary", ""),
                "news_sentiment": sentiment.get("score", 0) if sentiment else 0,
                "news_verdict": sentiment.get("verdict", "Neutral") if sentiment else "Neutral",
                "pulse": self.live_activity # Include per-ticker pulse
            }
            
            # Merge with existing data (Runner metadata like 'pid' and 'active')
            curr_data.update(status)
            
            with open(self.status_path, "w", encoding="utf-8") as f:
                json.dump(curr_data, f)
            
            self._last_export_time = current_time
        except Exception:
            pass # Silent failure for dashboard export
