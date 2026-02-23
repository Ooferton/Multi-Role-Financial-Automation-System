import yfinance as yf
import logging
from typing import Dict, List, Optional
from datetime import datetime

class EconomistAgent:
    """
    Macro-Economic Sense agent.
    Monitors global 'vital signs' like VIX, Yields, and Gold.
    """
    def __init__(self, config: Dict):
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # Macro Tickers
        self.tickers = {
            "VIX": "^VIX",       # Fear Index
            "10Y_Yield": "^TNX", # Interest Rates
            "Gold": "GC=F",      # Safe Haven
            "Dollar": "DX-Y.NYB" # Currency Strength
        }
        
        self.last_outlook = {
            "vibe": "Neutral",
            "metrics": {},
            "timestamp": None
        }

    def update_outlook(self) -> Dict:
        """
        Fetches latest macro data and computes the 'Economic Vibe'.
        """
        try:
            metrics = {}
            for name, ticker_sym in self.tickers.items():
                t = yf.Ticker(ticker_sym)
                # fast_info is efficient for current price
                metrics[name] = t.fast_info.last_price
            
            # 1. VIX Analysis (Granular Severity)
            vix = metrics.get("VIX", 15)
            crisis_severity = 0.0
            
            if vix > 45:
                vibe = "SUPER_BLACK_SWAN"
                crisis_severity = 1.0
            elif vix > 30:
                vibe = "Crisis / Extreme Fear"
                crisis_severity = 0.7
            elif vix > 22:
                vibe = "Risk-Off / Heightened Anxiety"
                crisis_severity = 0.3
            elif vix < 13:
                vibe = "Euphoria / Low Volatility"
                crisis_severity = 0.0
            else:
                vibe = "Risk-On / Stable"
                crisis_severity = 0.0
                
            self.last_outlook = {
                "vibe": vibe,
                "metrics": metrics,
                "timestamp": datetime.now(),
                "crisis_severity": crisis_severity,
                "summary": f"Market is currently in a {vibe} regime (VIX: {vix:.2f}, Crisis Score: {crisis_severity:.2f})"
            }
            
            return self.last_outlook
            
        except Exception as e:
            self.logger.error(f"Economist error: {e}")
            return self.last_outlook

    def get_crisis_severity(self) -> float:
        """Returns 0.0 (Safe) to 1.0 (Apocalyptic)."""
        return self.last_outlook.get("crisis_severity", 0.0)

    def get_vibe(self) -> str:
        return self.last_outlook["vibe"]

    def is_risk_off(self) -> bool:
        vix = self.last_outlook["metrics"].get("VIX", 15)
        return vix > 25
