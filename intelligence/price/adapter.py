# backend_p1.py
# PM-GRADE THIN ADAPTER
# NO LOGIC, NO MATH, NO INDICATORS HERE

from intelligence.price.predictor import build_price_block


def build_price_intelligence(symbol: str, days: int):
    """
    Delegates 100% to original FinWise P1 logic.
    """

    return build_price_block(
        stock_symbol=symbol,
        prediction_days=days
    )