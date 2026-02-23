import yfinance as yf
from textblob import TextBlob
import logging
from typing import Dict, List, Optional
from datetime import datetime

class NewsSentimentEngine:
    """
    The 'Hearing' layer of the AI.
    Fetches news headlines and calculates a sentiment score.
    """
    def __init__(self, cache_timeout_mins: int = 30):
        self.logger = logging.getLogger(__name__)
        self._cache = {} # symbol -> {score, headlines, verdict, timestamp}
        self.cache_timeout_mins = cache_timeout_mins

    def get_sentiment(self, symbol: str) -> Dict:
        """
        Fetches latest news for a symbol and returns a sentiment summary with caching.
        """
        # 1. Check Cache
        if symbol in self._cache:
            cache_entry = self._cache[symbol]
            age = (datetime.now() - cache_entry['timestamp']).total_seconds() / 60
            if age < self.cache_timeout_mins:
                return cache_entry

        try:
            ticker = yf.Ticker(symbol)
            news = ticker.news
            
            if not news:
                return {"score": 0.0, "headlines": [], "verdict": "Neutral"}

            # Take top 5 headlines - structural change in yfinance response
            headlines = []
            for n in news[:5]:
                if 'content' in n and 'title' in n['content']:
                    headlines.append(n['content']['title'])
                elif 'title' in n:
                    headlines.append(n['title'])
            
            total_polarity = 0.0
            for h in headlines:
                analysis = TextBlob(h)
                total_polarity += analysis.sentiment.polarity
            
            avg_polarity = total_polarity / len(headlines) if headlines else 0.0
            
            verdict = "Bullish" if avg_polarity > 0.1 else "Bearish" if avg_polarity < -0.1 else "Neutral"
            
            result = {
                "score": round(avg_polarity, 4),
                "headlines": headlines,
                "verdict": verdict,
                "timestamp": datetime.now()
            }
            
            self._cache[symbol] = result
            return result
            
        except Exception as e:
            self.logger.error(f"Error fetching sentiment for {symbol}: {e}")
            return {"score": 0.0, "headlines": [], "verdict": "Neutral (Error)"}

    def get_market_vibe(self, symbols: List[str]) -> str:
        """
        Aggregates sentiment across target symbols to get a broad market 'vibe'.
        """
        scores = []
        for s in symbols[:5]: # Cap at 5 for performance
            s_data = self.get_sentiment(s)
            scores.append(s_data['score'])
            
        avg = sum(scores) / len(scores) if scores else 0
        return "Bullish" if avg > 0.05 else "Bearish" if avg < -0.05 else "Neutral"
