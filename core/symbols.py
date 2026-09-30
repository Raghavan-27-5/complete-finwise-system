"""Ticker symbol normalization and resolution for FinWise.

The dashboard must accept arbitrary US and Indian tickers typed by a user:

* US tickers are used as-is:            ``AAPL``, ``NVDA``, ``MSFT``, ``BRK.B``
* Indices are used as-is:               ``^NSEI``, ``^BSESN``, ``^GSPC``
* Indian tickers use Yahoo suffixes:    ``RELIANCE.NS`` (NSE), ``TCS.BO`` (BSE)

Bare alphabetic input (for example ``RELIANCE`` or ``TCS``) is probed against
Yahoo Finance in the order ``SYMBOL`` -> ``SYMBOL.NS`` -> ``SYMBOL.BO``.

Documented ambiguity: a few Indian issuers also trade in the US as ADRs, so
bare ``INFY`` resolves to the US ADR (probe order is bare-first). Use
``INFY.NS`` to target the NSE listing explicitly.

``yfinance`` is imported lazily inside the probe helper so importing this
module stays cheap and safe on constrained machines.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

INDIAN_EXCHANGE_SUFFIXES: Tuple[str, ...] = (".NS", ".BO")
_INDEX_PREFIX = "^"
# Symbols containing these markers are already fully qualified
# (e.g. BRK.B, BTC-USD, EURUSD=X) and must never be suffix-probed.
_STRUCTURED_MARKERS: Tuple[str, ...] = (".", "-", "=")

# Probe results are cached for the process lifetime: repeat clicks are free.
_PROBE_CACHE: Dict[str, bool] = {}


def normalize_symbol(raw: str) -> str:
    """Return the canonical (upper-cased, stripped) form of a raw ticker."""
    return str(raw or "").strip().upper()


def is_index_symbol(symbol: str) -> bool:
    """True for Yahoo index symbols such as ``^NSEI`` or ``^GSPC``."""
    return symbol.startswith(_INDEX_PREFIX)


def has_exchange_suffix(symbol: str) -> bool:
    """True when the symbol already carries an Indian exchange suffix."""
    return symbol.endswith(INDIAN_EXCHANGE_SUFFIXES)


def _probe_yahoo(symbol: str) -> bool:
    """Best-effort check that Yahoo Finance recognises ``symbol``.

    Uses the lightweight ``fast_info`` endpoint first and falls back to a
    five-day download. Any failure (including a missing/offline ``yfinance``)
    is treated as "unknown symbol" so the caller can try the next candidate.
    """
    if symbol in _PROBE_CACHE:
        return _PROBE_CACHE[symbol]

    resolved = False
    try:
        import yfinance as yf  # lazy import: keeps this module lightweight

        try:
            fast_info = yf.Ticker(symbol).fast_info
            try:
                last_price = fast_info["last_price"]
            except Exception:
                last_price = getattr(fast_info, "last_price", None)
            resolved = last_price is not None
        except Exception:
            resolved = False

        if not resolved:
            frame = yf.download(
                symbol,
                period="5d",
                progress=False,
                threads=False,
                auto_adjust=True,
            )
            resolved = frame is not None and not frame.empty
    except Exception:
        resolved = False

    _PROBE_CACHE[symbol] = resolved
    return resolved


def resolve_symbol(raw: str) -> Tuple[str, List[str]]:
    """Resolve raw user input to a tradeable Yahoo symbol.

    Returns:
        ``(resolved_symbol, tried_symbols)``. When nothing resolves, the
        normalized input is returned unchanged so the downstream pipeline
        raises its own error and the dashboard can surface it.
    """
    symbol = normalize_symbol(raw)
    if not symbol:
        return "", []

    tried: List[str] = [symbol]

    if is_index_symbol(symbol) or has_exchange_suffix(symbol):
        return symbol, tried
    if any(marker in symbol for marker in _STRUCTURED_MARKERS):
        return symbol, tried

    if _probe_yahoo(symbol):
        return symbol, tried

    for suffix in INDIAN_EXCHANGE_SUFFIXES:
        candidate = f"{symbol}{suffix}"
        tried.append(candidate)
        if _probe_yahoo(candidate):
            return candidate, tried

    return symbol, tried