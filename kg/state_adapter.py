from dataclasses import dataclass
from typing import Optional, Dict


@dataclass
class PriceState:
    trend: Optional[str]
    forecast_slope: Optional[float]
    downside_risk: Optional[float]


@dataclass
class SentimentState:
    global_score: Optional[float]
    label: Optional[str]
    aspects: Optional[Dict[str, float]]


@dataclass
class State:
    symbol: Optional[str]
    as_of: Optional[str]
    price: Optional[PriceState]
    sentiment: Optional[SentimentState]


def adapt_state(raw_state: Dict) -> State:
    raw_price = raw_state.get("price")
    price = None
    if raw_price is not None:
        price = PriceState(
            trend=raw_price.get("trend"),
            forecast_slope=raw_price.get("forecast_slope"),
            downside_risk=raw_price.get("downside_risk"),
        )

    raw_sentiment = raw_state.get("sentiment")
    sentiment = None
    if raw_sentiment is not None:
        sentiment = SentimentState(
            global_score=raw_sentiment.get("global_score"),
            label=raw_sentiment.get("label"),
            aspects=raw_sentiment.get("aspects"),
        )

    return State(
        symbol=raw_state.get("symbol"),
        as_of=raw_state.get("as_of"),
        price=price,
        sentiment=sentiment,
    )
