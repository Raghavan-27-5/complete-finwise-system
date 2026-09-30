# orchestrator.py
import logging
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from datetime import datetime, timezone

from intelligence.price.adapter import build_price_intelligence
from intelligence.sentiment.adapter import build_sentiment_intelligence
from kg.state_adapter import adapt_state

logger = logging.getLogger(__name__)

# The two engines are independent (price = Yahoo/LSTM, sentiment = news/FinBERT),
# so they run concurrently: a click costs max(p1, p2) instead of p1 + p2.
_ENGINE_TIMEOUT_SECONDS = float(__import__("os").environ.get("FINWISE_ENGINE_TIMEOUT", "90"))
_NEUTRAL_SENTIMENT = {
    "global_score": 0.0,
    "label": None,
    "aspects": {},
    "impact_articles": [],
}


def compute_state(symbol: str, days: int, kg_writer=None):
    """Compute the combined FinWise state for a symbol.

    Args:
        symbol: Ticker symbol (US or Indian, e.g. "AAPL", "RELIANCE.NS").
        days: Forecast horizon in days.
        kg_writer: Optional KG writer (e.g. ``kg.kg_writer.KGWriter``).
            When ``None`` the Neo4j snapshot write is skipped, which keeps the
            dashboard runnable in standalone / demo mode without credentials.
    """
    # Run both engines concurrently and cap the pair: a click must never hang,
    # and each engine already degrades to a neutral block on its own failure.
    # NOTE: no context manager — exiting one waits for the workers and would
    # defeat the timeout.
    pool = ThreadPoolExecutor(max_workers=2)
    try:
        price_future = pool.submit(build_price_intelligence, symbol, days)
        sentiment_future = pool.submit(build_sentiment_intelligence, symbol, days)

        try:
            p1 = price_future.result(timeout=_ENGINE_TIMEOUT_SECONDS)
        except FuturesTimeout:
            raise RuntimeError(
                f"Price engine exceeded {_ENGINE_TIMEOUT_SECONDS:.0f}s for {symbol}"
            ) from None
        except Exception as exc:
            raise RuntimeError(
                f"Price engine failed for {symbol}: {type(exc).__name__}: {exc}"
            ) from exc

        try:
            p2 = sentiment_future.result(timeout=_ENGINE_TIMEOUT_SECONDS)
            if not isinstance(p2, dict):
                p2 = {}
        except FuturesTimeout:
            logger.warning(
                "Sentiment engine exceeded %.0fs for %s — using an empty block",
                _ENGINE_TIMEOUT_SECONDS,
                symbol,
            )
            p2 = dict(_NEUTRAL_SENTIMENT, error="sentiment timeout")
        except Exception as exc:
            logger.warning("Sentiment engine unavailable for %s: %s", symbol, exc)
            p2 = dict(
                _NEUTRAL_SENTIMENT,
                error=f"sentiment engine error: {type(exc).__name__}",
            )
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

    raw_state = {
        "symbol": symbol,
        "as_of": datetime.now(timezone.utc).isoformat(),
        # Defensive reads: the sentiment pipeline returns a minimal error dict
        # ({"global_score": 0.0, "error": "No articles"}) when no relevant news
        # was found for a ticker. The dashboard must still render the price and
        # risk analytics in that case instead of failing the whole evaluation.
        "price": {
            "trend": p1.get("trend"),
            "forecast_slope": p1.get("forecast_slope"),
            "downside_risk": p1.get("downside_risk"),
        },
        "sentiment": {
            "global_score": p2.get("global_score", 0.0) or 0.0,
            "label": p2.get("label"),
            "aspects": p2.get("aspects") or {},
        },
        "history": p1.get("history") or {},
        "forecast": p1.get("forecast") or {},
        "monte_carlo": p1.get("monte_carlo") or {},
        "indicators": p1.get("indicators") or {},
        "articles": p2.get("impact_articles") or [],
    }

    state = adapt_state(raw_state)
    if kg_writer is not None:
        kg_writer.write_snapshot(state)

    return raw_state