# orchestrator.py
from datetime import datetime, timezone
from intelligence.price.adapter import build_price_intelligence
from intelligence.sentiment.adapter import build_sentiment_intelligence
from kg.state_adapter import adapt_state


def compute_state(symbol: str, days: int, kg_writer):
    p1 = build_price_intelligence(symbol, days)
    p2 = build_sentiment_intelligence(symbol, days)

    raw_state = {
        "symbol": symbol,
        "as_of": datetime.now(timezone.utc).isoformat(),
        "price": {
            "trend": p1["trend"],
            "forecast_slope": p1.get("forecast_slope"),
            "downside_risk": p1.get("downside_risk"),
        },
        "sentiment": {
            "global_score": p2["global_score"],
            "label": p2.get("label"),
            "aspects": p2["aspects"],
        },
        "history": p1["history"],
        "forecast": p1["forecast"],
        "monte_carlo": p1["monte_carlo"],
        "indicators": p1["indicators"],
        "articles": p2.get("impact_articles", []),
    }

    state = adapt_state(raw_state)
    kg_writer.write_snapshot(state)

    return raw_state