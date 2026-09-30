# orchestrator.py
import logging
from datetime import datetime, timezone

from intelligence.price.adapter import build_price_intelligence
from intelligence.sentiment.adapter import build_sentiment_intelligence
from kg.state_adapter import adapt_state

logger = logging.getLogger(__name__)


def compute_state(symbol: str, days: int, kg_writer=None):
    """Compute the combined FinWise state for a symbol.

    Args:
        symbol: Ticker symbol (US or Indian, e.g. "AAPL", "RELIANCE.NS").
        days: Forecast horizon in days.
        kg_writer: Optional KG writer (e.g. ``kg.kg_writer.KGWriter``).
            When ``None`` the Neo4j snapshot write is skipped, which keeps the
            dashboard runnable in standalone / demo mode without credentials.
    """
    p1 = build_price_intelligence(symbol, days)

    # Sentiment is best-effort: a model download failure, an offline runtime or a
    # ticker with no coverage must never take down the price/risk analytics, so
    # any failure degrades to an empty (but well-formed) sentiment block.
    try:
        p2 = build_sentiment_intelligence(symbol, days)
        if not isinstance(p2, dict):
            p2 = {}
    except Exception as exc:  # pragma: no cover - environment dependent
        logger.warning("Sentiment engine unavailable for %s: %s", symbol, exc)
        p2 = {
            "global_score": 0.0,
            "label": None,
            "aspects": {},
            "impact_articles": [],
            "error": f"sentiment engine error: {type(exc).__name__}",
        }

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