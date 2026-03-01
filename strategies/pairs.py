import logging
from typing import Dict, Optional, List
from collections import defaultdict
from agents.trading_agent import TradingStrategy
from data.feature_store import MarketTick
from core.broker_interface import BrokerInterface
from ml.quant_models import calculate_spread_zscore, check_cointegration
import numpy as np

class PairsStrategy(TradingStrategy):
    """
    Statistical Arbitrage (Pairs Trading) Strategy.
    
    Logic:
    1. Monitors predefined pairs of correlated assets (e.g., V / MA).
    2. Uses Engle-Granger Cointegration and Spread Z-Score.
    3. If Z-Score > 2.0: Short Asset A, Long Asset B (A is overvalued vs B).
    4. If Z-Score < -2.0: Long Asset A, Short Asset B (A is undervalued vs B).
    """
    
    def __init__(self, name: str, config: Dict, broker: BrokerInterface = None):
        super().__init__(name, config)
        self.broker = broker
        self.logger = logging.getLogger(__name__)
        
        # Predefined pairs to monitor (Asset A, Asset B)
        self.pairs = config.get("pairs", [
            ("V", "MA"),
            ("GOOGL", "GOOG"),
            ("AAPL", "MSFT"),
            ("KO", "PEP"),
            ("XOM", "CVX")
        ])
        
        self.window_size = config.get("window_size", 100)
        self.entry_zscore = config.get("entry_zscore", 2.0)
        self.exit_zscore = config.get("exit_zscore", 0.5)
        self.risk_per_pair_pct = config.get("risk_per_pair_pct", 0.10) # 10% of equity per pair
        
        # History for spread calculations
        self.history = defaultdict(list)
        
        # Active trades tracking
        # key: pair tuple e.g. ("V", "MA")
        # value: { 'long_symbol': 'MA', 'short_symbol': 'V', 'long_qty': 10, 'short_qty': 10, 'status': 'ACTIVE' }
        self.active_pairs = {}

    def on_tick(self, tick: MarketTick) -> Optional[Dict]:
        """
        Record ticks and evaluate pairs. We return None normally because this strategy
        needs to emit MULTIPLE orders (a buy and a sell). We will handle execution internally
        or return a special composite signal. Since the framework expects one action Dict, 
        we will execute via broker directly here or yield a list. 
        For standard integration, we'll execute directly via broker.
        """
        sym = tick.symbol
        
        # Only track history for symbols in our pairs
        pair_symbols = set([s for p in self.pairs for s in p])
        if sym not in pair_symbols:
            return None
            
        self.history[sym].append(tick.price)
        if len(self.history[sym]) > self.window_size:
            self.history[sym].pop(0)
            
        # Check pairs
        for pair in self.pairs:
            sym_A, sym_B = pair
            
            # Ensure we have enough data for both
            if len(self.history[sym_A]) < self.window_size or len(self.history[sym_B]) < self.window_size:
                continue
                
            prices_A = np.array(self.history[sym_A])
            prices_B = np.array(self.history[sym_B])
            
            current_A = prices_A[-1]
            current_B = prices_B[-1]
            
            # Only trade cointegrated pairs (re-check occasionally or assume predefined are coint)
            # For speed, we just rely on Z-Score of spread if predefined
            
            z_score = calculate_spread_zscore(prices_A, prices_B, window=self.window_size)
            
            # Manage active pair
            if pair in self.active_pairs:
                active = self.active_pairs[pair]
                # Exit condition: Z-Score reverts to mean (crosses exit_zscore)
                # If we shorted A (z > 2), we want z to drop below 0.5
                if active['short_symbol'] == sym_A and z_score < self.exit_zscore:
                    self._close_pair(pair)
                # If we shorted B (z < -2), we want z to rise above -0.5
                elif active['short_symbol'] == sym_B and z_score > -self.exit_zscore:
                    self._close_pair(pair)
                continue
                
            # Entry conditions
            if z_score > self.entry_zscore:
                # A is overvalued, B is undervalued. Short A, Long B.
                self._open_pair(pair, sym_B, sym_A, current_B, current_A, z_score)
            elif z_score < -self.entry_zscore:
                # A is undervalued, B is overvalued. Long A, Short B.
                self._open_pair(pair, sym_A, sym_B, current_A, current_B, z_score)
                
        return None
        
    def _open_pair(self, pair, long_sym, short_sym, long_price, short_price, z_score):
        if not self.broker: return
        
        try:
            summary = self.broker.get_account_summary()
            buying_power = summary.get('buying_power', 0)
            equity = summary.get('equity', buying_power)
            
            trade_value = equity * self.risk_per_pair_pct
            leg_value = trade_value / 2.0 # Half to long, half to short
            
            if leg_value > buying_power * 0.4: # Safety check
                return
                
            long_qty = int(leg_value // long_price)
            short_qty = int(leg_value // short_price)
            
            if long_qty <= 0 or short_qty <= 0:
                return
                
            # Execute Long Limit/Market
            self.broker.submit_order(long_sym, long_qty, "BUY", "MARKET")
            # Execute Short Limit/Market
            self.broker.submit_order(short_sym, short_qty, "SELL", "MARKET")
            
            self.logger.info(f"📈 [PAIRS] OPEN: Pair {pair} | Z={z_score:.2f} | Long {long_qty} {long_sym} & Short {short_qty} {short_sym}")
            
            self.active_pairs[pair] = {
                'long_symbol': long_sym,
                'short_symbol': short_sym,
                'long_qty': long_qty,
                'short_qty': short_qty,
                'z_entry': z_score
            }
        except Exception as e:
            self.logger.error(f"Failed to open pair {pair}: {e}")

    def _close_pair(self, pair):
        if not self.broker: return
        active = self.active_pairs.pop(pair, None)
        if not active: return
        
        long_sym = active['long_symbol']
        short_sym = active['short_symbol']
        
        try:
            # Sell the Long
            self.broker.submit_order(long_sym, active['long_qty'], "SELL", "MARKET")
            # Buy to cover the Short
            self.broker.submit_order(short_sym, active['short_qty'], "BUY", "MARKET")
            
            self.logger.info(f"📉 [PAIRS] CLOSE: Pair {pair} | Mean Reversion Reached.")
        except Exception as e:
            self.logger.error(f"Failed to close pair {pair}: {e}")

    def on_bar(self, bar: Dict) -> Optional[Dict]:
        return None
