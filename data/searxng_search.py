import os
import requests
import logging
from typing import List, Dict

class SearXNGClient:
    """
    Fetches real-time market news and sentiment from SearXNG,
    a free, self-hosted metasearch engine (no API key required).
    Queries the local SearXNG instance running inside the Docker container.
    """
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # Use a public SearXNG instance (no self-hosting needed, no API keys)
        self.base_url = os.environ.get("SEARXNG_URL", "https://searx.be")
        
        self.enabled = True
        self.logger.info(f"🌐 SearXNG Search Engine targeting: {self.base_url}")
            
    def search_news(self, query: str, count: int = 5) -> List[Dict]:
        """
        Searches the live internet for news related to the query.
        Returns a list of articles with title, description, and source.
        """
        url = f"{self.base_url}/search"
        params = {
            "q": query,
            "format": "json",
            "categories": "news",
            "time_range": "day",
            "language": "en",
            "pageno": 1
        }
        
        try:
            response = requests.get(url, params=params, timeout=10.0)
            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])
                
                parsed_news = []
                for res in results[:count]:
                    parsed_news.append({
                        "title": res.get("title", ""),
                        "description": res.get("content", ""),
                        "url": res.get("url", ""),
                        "source": res.get("engine", "unknown")
                    })
                return parsed_news
            else:
                self.logger.error(f"SearXNG API Error: {response.status_code} - {response.text}")
                return []
        except requests.exceptions.ConnectionError:
            self.logger.warning("SearXNG not reachable (may not be running locally). Skipping news search.")
            self.enabled = False
            return []
        except Exception as e:
            self.logger.error(f"SearXNG search failed: {e}")
            return []

    def get_macro_sentiment(self) -> str:
        """
        Runs a broad query for macroeconomic news to determine a simple sentiment.
        """
        if not self.enabled:
            return "NEUTRAL"
            
        news = self.search_news("stock market economy fed inflation recession")
        if not news:
            return "NEUTRAL"
            
        text_corpus = " ".join([n["title"] + " " + n["description"] for n in news]).lower()
        
        bear_words = ["crash", "selloff", "recession", "fear", "drop", "plunge", "inflation", "rate hike", "panic"]
        bull_words = ["rally", "soar", "surge", "record high", "optimism", "growth", "cut rates", "bullish"]
        
        bear_score = sum(text_corpus.count(w) for w in bear_words)
        bull_score = sum(text_corpus.count(w) for w in bull_words)
        
        self.logger.info(f"Macro Sentiment Read: Bull={bull_score}, Bear={bear_score}")
        
        if bear_score > bull_score * 1.5 and bear_score > 3:
            return "BEARISH_MACRO"
        elif bull_score > bear_score * 1.5 and bull_score > 3:
            return "BULLISH_MACRO"
        return "NEUTRAL"
