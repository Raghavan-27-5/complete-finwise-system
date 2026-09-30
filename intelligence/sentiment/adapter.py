# backend_p2.py
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout

from intelligence.sentiment.pipeline import get_news_sentiment_async, _run_async

logger = logging.getLogger(__name__)

# Absolute cap for the whole sentiment engine. Even if a news source or a model
# download stalls, the dashboard must answer within this budget.
SENTIMENT_TIMEOUT_SECONDS = float(os.environ.get("FINWISE_SENTIMENT_TIMEOUT", "150"))

_EMPTY_SENTIMENT = {
    "global_score": 0.0,
    "label": None,
    "aspects": {},
    "impact_articles": [],
}


def build_sentiment_intelligence(stock_symbol: str, days: int = 7):
    """
    Institutional sentiment engine (NLP-enabled).
    Safe sync adapter for Gradio / event-loop environments.

    Hard-capped: a stalled data source can never hang the dashboard. On timeout
    or failure the caller receives an empty, well-formed sentiment block, and the
    price / risk analytics still render.
    """
    try:
        # NOTE: no context manager here — exiting one waits for the worker
        # thread, which would silently undo the timeout we are implementing.
        pool = ThreadPoolExecutor(max_workers=1)
        future = pool.submit(_run_async, get_news_sentiment_async(stock_symbol, days))
        try:
            return future.result(timeout=SENTIMENT_TIMEOUT_SECONDS)
        except FuturesTimeout:
            logger.warning(
                "Sentiment engine exceeded %.0fs for %s — returning an empty block",
                SENTIMENT_TIMEOUT_SECONDS,
                stock_symbol,
            )
            return dict(_EMPTY_SENTIMENT, error="sentiment timeout")
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
    except Exception as exc:
        logger.warning("Sentiment adapter failure for %s: %s", stock_symbol, exc)
        return dict(_EMPTY_SENTIMENT, error=f"sentiment error: {type(exc).__name__}")