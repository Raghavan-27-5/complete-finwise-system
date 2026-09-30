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

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from typing import Dict, List, Tuple

INDIAN_EXCHANGE_SUFFIXES: Tuple[str, ...] = (".NS", ".BO")
_INDEX_PREFIX = "^"
# Symbols containing these markers are already fully qualified
# (e.g. BRK.B, BTC-USD, EURUSD=X) and must never be suffix-probed.
_STRUCTURED_MARKERS: Tuple[str, ...] = (".", "-", "=")

# Probe results are cached for the process lifetime: repeat clicks are free.
_PROBE_CACHE: Dict[str, bool] = {}

# Hard cap on Yahoo symbol resolution. Resolution sits on the click path (the
# warm-up calls compute_state directly and never probes), so an uncapped probe
# chain would otherwise add minutes to a single UI click.
PROBE_TIMEOUT_SECONDS = float(os.environ.get("FINWISE_PROBE_TIMEOUT", "12"))
# Shared budget for the whole bare -> .NS -> .BO chain, so resolution can never
# stack three per-probe caps onto one click.
RESOLVE_BUDGET_SECONDS = float(os.environ.get("FINWISE_RESOLVE_BUDGET", "15"))


def normalize_symbol(raw: str) -> str:
    """Return the canonical (upper-cased, stripped) form of a raw ticker."""
    return str(raw or "").strip().upper()


def is_index_symbol(symbol: str) -> bool:
    """True for Yahoo index symbols such as ``^NSEI`` or ``^GSPC``."""
    return symbol.startswith(_INDEX_PREFIX)


def has_exchange_suffix(symbol: str) -> bool:
    """True when the symbol already carries an Indian exchange suffix."""
    return symbol.endswith(INDIAN_EXCHANGE_SUFFIXES)


def _probe_yahoo(symbol: str, deadline: float = None) -> bool:
    """Best-effort check that Yahoo Finance recognises ``symbol``.

    Uses the lightweight ``fast_info`` endpoint first and falls back to a
    five-day download, both under a hard wall-clock cap (the smaller of
    ``PROBE_TIMEOUT_SECONDS`` and whatever remains of ``deadline``). Any
    failure (including a missing/offline ``yfinance``) is treated as "unknown
    symbol" so the caller can try the next candidate without stalling the UI.
    """
    if symbol in _PROBE_CACHE:
        return _PROBE_CACHE[symbol]

    cap = PROBE_TIMEOUT_SECONDS
    if deadline is not None:
        cap = min(cap, max(deadline - time.monotonic(), 0.5))
        if deadline - time.monotonic() <= 0:
            _PROBE_CACHE[symbol] = False
            return False

    resolved = False
    try:
        import yfinance as yf  # lazy import: keeps this module lightweight

        def _check() -> bool:
            try:
                fast_info = yf.Ticker(symbol).fast_info
                try:
                    last_price = fast_info["last_price"]
                except Exception:
                    last_price = getattr(fast_info, "last_price", None)
                if last_price is not None:
                    return True
            except Exception:
                pass
            frame = yf.download(
                symbol,
                period="5d",
                progress=False,
                threads=False,
                auto_adjust=True,
                timeout=(10, 20),
            )
            return frame is not None and not frame.empty

        # NOTE: no context manager here — exiting one waits for the worker
        # thread and would silently defeat the timeout we are implementing.
        pool = ThreadPoolExecutor(max_workers=1)
        future = pool.submit(_check)
        try:
            resolved = bool(future.result(timeout=cap))
        except FuturesTimeout:
            logging.getLogger("symbols").warning(
                "Probe for %s exceeded %.1fs — treating as unknown",
                symbol,
                cap,
            )
            resolved = False
        except Exception as exc:
            logging.getLogger("symbols").warning("Probe for %s failed: %s", symbol, exc)
            resolved = False
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
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

    # One shared budget for the whole chain: three sequential 12s probes would
    # add 36s to a single click. The per-probe cap still applies inside
    # _probe_yahoo, this simply stops the chain early once the budget is gone.
    deadline = time.monotonic() + RESOLVE_BUDGET_SECONDS

    if _probe_yahoo(symbol, deadline):
        return symbol, tried

    for suffix in INDIAN_EXCHANGE_SUFFIXES:
        if time.monotonic() >= deadline:
            break
        candidate = f"{symbol}{suffix}"
        tried.append(candidate)
        if _probe_yahoo(candidate, deadline):
            return candidate, tried

    return symbol, tried