import logging
import re
import xml.etree.ElementTree as ET
from typing import Dict, List
from datetime import datetime

import requests
import yfinance as yf

# Optional: textblob gives more accurate polarity if installed
try:
    from textblob import TextBlob
    _TEXTBLOB_AVAILABLE = True
except ImportError:
    _TEXTBLOB_AVAILABLE = False

# Optional: feedparser as secondary RSS parser
try:
    import feedparser
    _FEEDPARSER_AVAILABLE = True
except ImportError:
    _FEEDPARSER_AVAILABLE = False

# ---------------------------------------------------------------------------
# Built-in keyword sentiment scorer (works with no NLP libraries at all)
# ---------------------------------------------------------------------------
_BULLISH_WORDS = {
    'surge', 'soar', 'rally', 'gain', 'rise', 'jump', 'climb', 'beat', 'record',
    'profit', 'growth', 'strong', 'bullish', 'upgrade', 'boost', 'positive',
    'outperform', 'exceed', 'high', 'recover', 'winning', 'expansion', 'up',
    'breakthrough', 'new high', 'all-time', 'optimistic', 'opportunity',
}
_BEARISH_WORDS = {
    'fall', 'drop', 'decline', 'slump', 'plunge', 'crash', 'loss', 'miss',
    'weak', 'bearish', 'downgrade', 'cut', 'negative', 'underperform', 'low',
    'concern', 'fear', 'risk', 'recession', 'inflation', 'tariff', 'layoff',
    'bankrupt', 'investigation', 'fine', 'sell-off', 'down', 'warning',
    'disappointing', 'crisis', 'debt', 'default',
}

def _keyword_polarity(text: str) -> float:
    """Simple word-count polarity in [-1, 1]. No external deps required."""
    words = set(re.findall(r'\b\w+\b', text.lower()))
    bull = len(words & _BULLISH_WORDS)
    bear = len(words & _BEARISH_WORDS)
    total = bull + bear
    if total == 0:
        return 0.0
    return (bull - bear) / total


def _score_headline(headline: str) -> float:
    """Use textblob if available, otherwise fall back to keyword scorer."""
    if _TEXTBLOB_AVAILABLE:
        try:
            return TextBlob(headline).sentiment.polarity
        except Exception:
            pass
    return _keyword_polarity(headline)


# ---------------------------------------------------------------------------
# RSS feeds — tried in order, stdlib xml parser used first
# ---------------------------------------------------------------------------
_RSS_FEEDS = [
    # Yahoo Finance market news
    "https://finance.yahoo.com/news/rssindex",
    # Reuters business
    "https://feeds.reuters.com/reuters/businessNews",
    # MarketWatch top stories
    "https://feeds.content.dowjones.io/public/rss/mw_topstories",
    # Seeking Alpha (market news)
    "https://seekingalpha.com/market_currents.xml",
    # CNBC markets
    "https://search.cnbc.com/rs/search/view.xml?partnerId=2000&keywords=market",
]

_RSS_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; SentienceBot/1.0; +https://huggingface.co/spaces/ooferton/sentience-trading-system)"
    )
}


def _fetch_rss_headlines(max_headlines: int = 5) -> List[str]:
    """Try each RSS feed in order; return first successful batch of headlines."""
    for url in _RSS_FEEDS:
        try:
            resp = requests.get(url, timeout=8, headers=_RSS_HEADERS)
            resp.raise_for_status()
            root = ET.fromstring(resp.content)
            titles = []
            # Standard RSS 2.0 structure: rss > channel > item > title
            for item in root.iter("item"):
                title_el = item.find("title")
                if title_el is not None and title_el.text:
                    titles.append(title_el.text.strip())
                if len(titles) >= max_headlines:
                    break
            if titles:
                return titles
        except Exception:
            # Try feedparser as secondary parser for this URL
            if _FEEDPARSER_AVAILABLE:
                try:
                    feed = feedparser.parse(url)
                    titles = [e.title for e in feed.entries[:max_headlines] if hasattr(e, 'title')]
                    if titles:
                        return titles
                except Exception:
                    pass
            continue
    return []


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

class NewsSentimentEngine:
    """
    The 'Hearing' layer of the AI.
    Fetches news headlines and calculates a sentiment score.
    Works with zero optional dependencies using keyword scoring + stdlib RSS.
    """

    def __init__(self, cache_timeout_mins: int = 30):
        self.logger = logging.getLogger(__name__)
        self._cache: Dict = {}
        self.cache_timeout_mins = cache_timeout_mins

    def get_sentiment(self, symbol: str) -> Dict:
        """
        Fetches latest news for a symbol and returns a sentiment summary with caching.
        Falls back to generic market RSS if symbol-specific news unavailable.
        """
        # 1. Check cache
        if symbol in self._cache:
            entry = self._cache[symbol]
            age_mins = (datetime.now() - entry['timestamp']).total_seconds() / 60
            if age_mins < self.cache_timeout_mins:
                return entry

        # 2. Try yfinance symbol news
        headlines = self._fetch_yfinance_news(symbol)

        # 3. Fall back to generic RSS if symbol news unavailable
        if not headlines:
            headlines = _fetch_rss_headlines()

        # 4. Score headlines
        if not headlines:
            return {"score": 0.0, "headlines": [], "verdict": "Neutral", "timestamp": datetime.now()}

        total_polarity = sum(_score_headline(h) for h in headlines)
        avg_polarity = total_polarity / len(headlines)
        verdict = "Bullish" if avg_polarity > 0.1 else "Bearish" if avg_polarity < -0.1 else "Neutral"

        result = {
            "score": round(avg_polarity, 4),
            "headlines": headlines,
            "verdict": verdict,
            "timestamp": datetime.now(),
        }
        self._cache[symbol] = result
        return result

    def _fetch_yfinance_news(self, symbol: str) -> List[str]:
        """Fetch top 5 headlines for symbol via yfinance."""
        try:
            ticker = yf.Ticker(symbol)
            news = ticker.news or []
            headlines = []
            for n in news[:5]:
                # Handle both old and new yfinance response structures
                if isinstance(n, dict):
                    if 'content' in n and isinstance(n['content'], dict):
                        title = n['content'].get('title', '')
                    else:
                        title = n.get('title', '')
                    if title:
                        headlines.append(title)
            return headlines
        except Exception as e:
            self.logger.debug(f"yfinance news fetch failed for {symbol}: {e}")
            return []

    def get_market_vibe(self, symbols: List[str]) -> str:
        """
        Aggregates sentiment across target symbols for a broad market 'vibe'.
        """
        scores = []
        for s in symbols[:5]:
            s_data = self.get_sentiment(s)
            scores.append(s_data['score'])
        avg = sum(scores) / len(scores) if scores else 0
        return "Bullish" if avg > 0.05 else "Bearish" if avg < -0.05 else "Neutral"
