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
        # Use a list of public SearXNG instances to fallback if one returns 403 (common for datacenter IPs)
        self.instances = [
            os.environ.get("SEARXNG_URL", "https://searx.be"),
            "https://searx.tiekoetter.com",
            "https://search.ononoki.org",
            "https://searx.work",
            "https://paulgo.io"
        ]
        
        self.enabled = True
        self.logger.info(f"🌐 SearXNG Search Engine targeting primary: {self.instances[0]}")
            
    def search_news(self, query: str, count: int = 5) -> List[Dict]:
        """
        Searches the live internet for news related to the query.
        Returns a list of articles with title, description, and source.
        """
        params = {
            "q": query,
            "format": "json",
            "categories": "news",
            "time_range": "day",
            "language": "en",
            "pageno": 1
        }
        
        # Public SearXNG instances often block default python-requests user agents.
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Sec-Ch-Ua": "\"Google Chrome\";v=\"122\", \"Chromium\";v=\"122\", \"Not=A?Brand\";v=\"24\"",
            "Sec-Ch-Ua-Mobile": "?0",
            "Sec-Ch-Ua-Platform": "\"Windows\""
        }
        
        for base_url in self.instances:
            url = f"{base_url}/search"
            try:
                response = requests.get(url, params=params, headers=headers, timeout=8.0)
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
                elif response.status_code == 403:
                    self.logger.warning(f"SearXNG 403 Forbidden at {base_url} (Likely IP block). Trying next instance...")
                    continue
                else:
                    self.logger.error(f"SearXNG API Error at {base_url}: {response.status_code}")
                    continue
            except requests.exceptions.Timeout:
                self.logger.warning(f"SearXNG timeout at {base_url}. Trying next...")
                continue
            except requests.exceptions.ConnectionError:
                self.logger.warning(f"SearXNG not reachable at {base_url}. Trying next...")
                continue
            except Exception as e:
                self.logger.error(f"SearXNG Exception at {base_url}: {e}")
                continue
                
        self.logger.error("❌ All SearXNG fallback instances failed or blocked the request.")
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
