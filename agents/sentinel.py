import logging
import time
from typing import Dict, List, Optional

class Sentinel:
    """
    Market Safety agent.
    Watches for 'Black Swan' events, high volatility spikes, and calendar constraints.
    """
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Safety thresholds
        self.volatility_limit = config.get('volatility_limit', 0.08) # 8% width limit
        self.forbidden_symbols: List[str] = []
        self.active_alert = False

    def check_safety(self, symbol: str, indicators: Dict, crisis_severity: float = 0.0) -> Dict:
        """
        Verifies if it is safe to trade a specific symbol.
        In a crisis, we are MORE restrictive on Longs but MORE permissive on Shorts/Hedges.
        """
        # 1. Volatility Check
        bb_width = indicators.get('bb_width', 0)
        
        # If BB Width is extreme (>15%), we have a potential Flash Crash / Black Swan
        if bb_width > 0.15:
            return {
                "safe": False, 
                "reason": f"Sentinel: BLACK SWAN detected ({bb_width:.1%}). Trading frozen for safety.",
                "recommendation": "GO_TO_CASH"
            }

        # Standard volatility limit check
        if bb_width > self.volatility_limit and crisis_severity < 0.7:
            return {
                "safe": False, 
                "reason": f"Sentinel: High volatility ({bb_width:.1%})",
                "recommendation": "WAIT"
            }

        # 3. Forbidden List
        if symbol in self.forbidden_symbols:
            return {"safe": False, "reason": f"Sentinel: Restricted", "recommendation": "WAIT"}

        return {
            "safe": True, 
            "reason": "Sentinel: Clear", 
            "recommendation": "CRISIS_SHORT" if crisis_severity > 0.5 else "OPTIMAL"
        }

    def check_sentiment_safety(self, sentiment: Dict, action: str = "BUY") -> Dict:
        """
        Vetoes trades based on news sentiment. 
        For SELL orders (Shorts), we allow deeper 'fear' as it represents momentum.
        """
        score = sentiment.get('score', 0.0)
        
        # If we are BUYING, we want positive/stable sentiment
        if action == "BUY":
            if score < -0.3:
                return {
                    "safe": False,
                    "reason": f"Sentinel: Negative sentiment ({score}) – Vetoing Long entry."
                }
        
        # If we are SELLING (Shorting), we actually expect negative sentiment
        elif action == "SELL":
            if score > 0.4:
                return {
                    "safe": False,
                    "reason": f"Sentinel: Sentiment too positive ({score}) for Short entry."
                }
        
        return {"safe": True, "reason": "Sentinel: Sentiment acceptable for action."}

    def restrict_symbol(self, symbol: str, duration_sec: int):
        self.forbidden_symbols.append(symbol)
        self.logger.warning(f"Sentinel: Restricting {symbol} for {duration_sec}s")
        # In a real system, we'd use a timer to clean this up
