import feedparser
from textblob import TextBlob
from typing import List, Dict
from dataclasses import dataclass
from datetime import datetime

@dataclass
class NewsItem:
    title: str
    source: str
    published_at: datetime
    sentiment_score: float # -1.0 to 1.0

class NewsService:
    """
    Ingests news from RSS feeds and analyzes sentiment.
    """
    def __init__(self):
        self.feeds = [
            "http://feeds.reuters.com/reuters/businessNews",
            "https://www.cnbc.com/id/10000664/device/rss/rss.html" # Finance
        ]
    
    def fetch_news(self) -> List[NewsItem]:
        news_items = []
        for feed_url in self.feeds:
            try:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries[:5]: # Top 5 per feed
                    sentiment = self._analyze_sentiment(entry.title)
                    item = NewsItem(
                        title=entry.title,
                        source=feed.feed.get('title', 'Unknown'),
                        published_at=datetime.now(), # Simplified, should parse entry.published
                        sentiment_score=sentiment
                    )
                    news_items.append(item)
            except Exception as e:
                print(f"Error fetching feed {feed_url}: {e}")
                
        return news_items

    def _analyze_sentiment(self, text: str) -> float:
        """
        Returns a score from -1.0 (Negative) to 1.0 (Positive).
        """
        blob = TextBlob(text)
        return blob.sentiment.polarity

if __name__ == "__main__":
    # Test
    ns = NewsService()
    # Mocking a fetch since real RSS might fail without internet/proxy
    print("News Service Initialized. Fetching logic ready.")
