"""
Mock News/Data Fetchers
Returns sample articles instead of calling external APIs.
"""
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any
import random


def generate_sample_articles(symbol: str, count: int = 5) -> List[Dict[str, Any]]:
    """Generate sample news articles for testing."""
    headlines = [
        f"{symbol} Reports Strong Q4 Earnings, Beats Estimates",
        f"Analysts Upgrade {symbol} on Growth Prospects",
        f"{symbol} Announces New Product Launch",
        f"Market Watch: {symbol} Shows Bullish Momentum",
        f"{symbol} CEO Discusses Strategic Vision",
        f"Institutional Investors Increase {symbol} Holdings",
        f"{symbol} Faces Regulatory Scrutiny",
        f"Bear Case: Why {symbol} May Underperform",
        f"{symbol} Dividend Announcement Surprises Market",
        f"Technical Analysis: {symbol} Breaks Key Resistance",
    ]
    
    sources = ["Reuters", "Bloomberg", "CNBC", "MarketWatch", "Yahoo Finance"]
    
    articles = []
    for i in range(count):
        pub_date = datetime.now(timezone.utc) - timedelta(days=random.randint(0, 7))
        articles.append({
            'headline': random.choice(headlines),
            'summary': f"This is a sample summary for {symbol} news article {i+1}.",
            'source': random.choice(sources),
            'published': pub_date.isoformat(),
            'engagement': random.randint(100, 5000),
            'media_url': None,
        })
    
    return articles


async def mock_aggregate_sources(query: str, symbol: str, cutoff: datetime) -> List[Dict[str, Any]]:
    """Mock aggregate_sources from fetchers.py"""
    return generate_sample_articles(symbol, count=5)


def mock_fetch_reddit_posts(symbol: str, subreddits: List[str], limit: int = 10) -> List[Dict[str, Any]]:
    """Mock Reddit fetcher - returns empty for simplicity."""
    return []


# Sample entities for NER testing
SAMPLE_ENTITIES = {
    "organizations": ["TestCorp", "Rival Inc", "Market Authority"],
    "persons": ["John CEO", "Jane Analyst"],
    "locations": ["New York", "Silicon Valley"],
}


def mock_extract_entities(text: str) -> Dict[str, List[str]]:
    """Mock entity extraction."""
    return SAMPLE_ENTITIES.copy()
