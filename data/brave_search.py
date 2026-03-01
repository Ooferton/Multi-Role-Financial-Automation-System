import os
import requests
import logging
from typing import List, Dict

class BraveSearchClient:
    """
    Fetches real-time market news and sentiment from the live internet 
    using the Brave Search API, similar to OpenClaw methodology.
    """
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.api_key = os.environ.get("BRAVE_API_KEY")
        
        self.enabled = bool(self.api_key)
        if not self.enabled:
            self.logger.warning("BraveSearchClient disabled. BRAVE_API_KEY missing.")
        else:
            self.logger.info("🌐 Brave Search Live Internet Access Online")
            
    def search_news(self, query: str, count: int = 5) -> List[Dict]:
        """
        Searches the live internet for news related to the query.
        Returns a list of articles with title, description, and source.
        """
        if not self.enabled:
            return []
            
        url = "https://api.search.brave.com/res/v1/news/search"
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self.api_key
        }
        params = {
            "q": query,
            "count": count,
            "freshness": "pd" # Past 24 hours
        }
        
        try:
            response = requests.get(url, headers=headers, params=params, timeout=5.0)
            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])
                
                parsed_news = []
                for res in results:
                    parsed_news.append({
                        "title": res.get("title", ""),
                        "description": res.get("description", ""),
                        "url": res.get("url", ""),
                        "source": res.get("meta_url", {}).get("hostname", "unknown")
                    })
                return parsed_news
            else:
                self.logger.error(f"Brave Search API Error: {response.status_code} - {response.text}")
                return []
        except Exception as e:
            self.logger.error(f"Brave Search connection failed: {e}")
            return []

    def get_macro_sentiment(self) -> str:
        """
        Runs a broad query for macroeconomic news to determine a simple sentiment.
        In a full version, this would be passed to an LLM. Here, we do keyword matching.
        """
        if not self.enabled:
            return "NEUTRAL"
            
        news = self.search_news("stock market crash recession inflation economy fed")
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
