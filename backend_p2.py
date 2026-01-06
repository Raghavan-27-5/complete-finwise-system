# backend_p2.py
from pipeline import get_news_sentiment_async, _run_async

def build_sentiment_intelligence(stock_symbol: str, days: int = 7):
    """
    Institutional sentiment engine (NLP-enabled).
    Safe sync adapter for Gradio / event-loop environments.
    """
    return _run_async(get_news_sentiment_async(stock_symbol, days))