# orchestrator.py
from backend_p1 import build_price_intelligence
from backend_p2 import build_sentiment_intelligence


def compute_state(symbol: str, days: int):
    p1 = build_price_intelligence(symbol, days)
    p2 = build_sentiment_intelligence(symbol, days)

    return {
        "symbol": symbol,

        # ---- PRICE INTELLIGENCE (REAL FINWISE) ----
        "history": p1["history"],          # actual vs predicted
        "forecast": p1["forecast"],        # LSTM future
        "monte_carlo": p1["monte_carlo"],  # Heston-like MC
        "indicators": p1["indicators"],
        "trend": p1["trend"],

        "sentiment_score": p2["global_score"],
        "sentiment_aspects": p2["aspects"],
        "articles": p2.get("impact_articles", []),

    }