
# pipeline.py (FINAL v20) - fully corrected, relevance=R2
"""
Production-ready pipeline.py
- Relevance filter: R2 (strict + smart synonyms)
- Safe JSON loaders
- Normalized headline dedupe (single deterministic pass)
- Robust timestamp and timezone handling
- Daily sentiment history with metadata (article_count, relevant_rate)
- DSP calculation, stable weighting, Granger readiness
- Detailed debug prints controlled by env PIPELINE_DEBUG
"""
# ============================================================
# NOTE:
# get_news_sentiment_async() is the ONLY production sentiment path.
# Any other sentiment pipelines are legacy and must not be used.
# ============================================================

import asyncio
import os
import re
import json
import sqlite3
import logging
from html import unescape
from typing import Dict, Union, List, Any
from collections import Counter
from datetime import datetime, timedelta, timezone
import math
import pandas as pd
import numpy as np
import yfinance as yf
from statsmodels.tsa.stattools import grangercausalitytests
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")

from core.config import (
    BASE_SOURCE_WEIGHTS,
    ASPECT_CATEGORIES,
    ASPECT_KEYWORDS,
    CACHE_DB,
    CACHE_TTL_HOURS,
    SOURCE_NAME_MAP
)

from intelligence.sentiment.fetchers import aggregate_sources
from intelligence.sentiment.models import (
    nlp,
    extract_aspects_from_doc,
    compute_ensemble_sentiment,
    process_multimodal,
    explain_sentiment,
    extract_entities_transformer
)

logger = logging.getLogger("pipeline")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Local DSP risk threshold
RISK_THRESHOLD = 0.15
# Debug verbosity controlled by environment
FULL_DEBUG = os.environ.get("PIPELINE_DEBUG", "0") == "1"

# Hard wall-clock caps: the dashboard must never hang on a stalled data source.
FETCH_TIMEOUT_SECONDS = float(os.environ.get("FINWISE_FETCH_TIMEOUT", "75"))
YF_TIMEOUT_SECONDS = float(os.environ.get("FINWISE_YF_TIMEOUT", "12"))


def _call_with_timeout(func, timeout: float, default=None):
    """Run a blocking callable with a hard wall-clock cap (best effort).

    The worker thread cannot be killed, but the caller stops waiting — which is
    what keeps the dashboard responsive when Yahoo / SEC endpoints stall.
    """
    from concurrent.futures import ThreadPoolExecutor
    from concurrent.futures import TimeoutError as FuturesTimeout

    # NOTE: no context manager here — exiting one waits for the worker thread,
    # which would silently undo the timeout we are implementing.
    pool = ThreadPoolExecutor(max_workers=1)
    future = pool.submit(func)
    try:
        return future.result(timeout=timeout)
    except FuturesTimeout:
        logger.warning(
            "Blocking call %s exceeded %.0fs — continuing without it",
            getattr(func, "__qualname__", repr(func)),
            timeout,
        )
        return default
    except Exception as exc:
        logger.warning("Blocking call failed: %s", exc)
        return default
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

# -------------------------
# Helper utilities
# -------------------------

def safe_json_load(s: Any, default: Any):
    try:
        if isinstance(s, str) and s:
            return json.loads(s)
        return s if s is not None else default
    except Exception:
        return default

# Keep canonical map but do NOT overwrite original mapping keys used elsewhere
SOURCE_NAME_MAP_CANON = {}
try:
    if isinstance(SOURCE_NAME_MAP, dict):
        SOURCE_NAME_MAP_CANON = {str(k).lower(): v for k, v in SOURCE_NAME_MAP.items()}
except Exception:
    SOURCE_NAME_MAP_CANON = {}

# -------------------------
# Noise, normalization, and relevance helpers (R2)
# -------------------------


NOISE_PATTERNS = [
    r"\b(sale|sales|deal|deals|discount|coupon|free shipping|black friday|cyber monday)\b",
    r"\b(price cut|lowest price|hits all-time low|now only|best buy|gift guide)\b",
    r"\b(shopping|top picks|best of|review:|hands[- ]on|best \w+ for|top \d+)\b",
    r"\b(trailer|movie|tv|series|projector|gaming monitor|smart tv)\b",
    r"\b(download|apk|install|torrent)\b",
    r"\b(recipe|diet|fitness|workout)\b",
    r"\b(click here|read more|learn more|sign up|subscribe|sponsored by)\b",
]
NOISE_RE = re.compile("|".join(NOISE_PATTERNS), flags=re.I)

SYNONYM_BRANDS = {
    'apple': ['iphone', 'ipad', 'macbook', 'mac', 'airpods', 'apple watch', 'a14', 'a15', 'a16', 'a19'],
    'microsoft': ['windows', 'azure', 'surface', 'xbox', 'office'],
    'tesla': ['model s', 'model 3', 'model x', 'model y', 'robotaxi'],
    'nvidia': ['gpu', 'geforce', 'tensor core', 'blackwell'],
    'google': ['android', 'pixel', 'search', 'google cloud'],
    'amazon': ['aws', 'kindle', 'prime', 'alexa'],
    'intel': ['chip', 'foundry', 'packaging'],
    'meta': ['facebook', 'instagram', 'oculus', 'metaverse']
}


def normalize_headline(h: str) -> str:
    if not isinstance(h, str):
        return ""
    s = unescape(h)
    s = s.replace('…', ' ').replace('—', ' ').replace('–', ' ')
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def is_marketing_or_consumer_noise(text: str) -> bool:
    if not isinstance(text, str) or len(text) < 10:
        return True
    return bool(NOISE_RE.search(text))


def generate_synonyms_for_company(company_name: str, symbol: str) -> List[str]:
    """R2 synonym generator: exact company tokens + brand heuristics + symbol"""
    syns = set()
    if symbol:
        syns.add(str(symbol).lower())

    if not company_name or not isinstance(company_name, str):
        return list(syns)

    cname = company_name.lower()
    # strip common suffixes
    cname_clean = re.sub(r"\b(inc|inc\.|ltd|llc|corp|co|company|plc)\b", "", cname)
    tokens = [t for t in re.split(r"\W+", cname_clean) if t]
    for t in tokens:
        syns.add(t)
    # add n-grams (2-word phrases)
    for i in range(len(tokens) - 1):
        syns.add(f"{tokens[i]} {tokens[i+1]}")

    # brand heuristics
    for k, v in SYNONYM_BRANDS.items():
        if k in cname:
            for term in v:
                syns.add(term)
    return list(syns)

def is_relevant_article(
    symbol: str,
    company_name: str,
    headline: str,
    entities: dict,
    allow_synonyms: bool = True,
    source: str = ""
) -> bool:
    """
    V28 INSTITUTIONAL-GRADE RELEVANCE FILTER (UNIVERSAL)
    Works for ALL US + Indian tickers, ALL sectors.
    
    Removes:
      - Crime, accidents, tragedies (car crashes, murders, etc.)
      - MarketBeat 13F position-change spam
      - Blog/SEO/opinion fluff
      - Competitor-only noise without read-through
      - Irrelevant mentions of vehicles/products detached from business context

    Keeps:
      - Earnings, guidance, revenue, margins
      - Strategy, partnerships, M&A
      - Regulatory, legal, compliance
      - Product launches, technology
      - SEC filings + EDGAR
      - MATERIAL competitor read-through (EV, banking, tech, telecom, etc.)
    """

    # -----------------------------
    # 0. Validate Inputs
    # -----------------------------
    if not isinstance(headline, str) or not headline.strip():
        return False

    h = headline.lower().strip()
    s = (source or "").lower().strip()
    sym = (symbol or "").lower()
    cname = (company_name or "").lower()

    # Clean entity orgs
    orgs = [str(o).lower() for o in entities.get("organizations", []) if o]

    # -----------------------------
    # 1. Hard Noise / SEO / Junk Sources
    # -----------------------------
    noise_patterns = [
        r"price prediction",
        r"where.*could be",
        r"top \d+ stocks",
        r"hot stocks",
        r"should you buy",
        r"reddit",
        r"crypto",
        r"blog",
        r"opinion",
        r"stocktwits",
        r"forum",
        r"motley fool",
        r"investorplace",
        r"zacks",
        r"seeking alpha",
        r"youtuber",
        r"influencer",
    ]
    for pat in noise_patterns:
        if re.search(pat, h):
            return False

    # -----------------------------
    # 2. Crime / Accident / Fatality Filter
    # -----------------------------
    crime_terms = [
        "killed", "dead", "death", "murder", "shot", "shooting",
        "body found", "decapitated", "frozen", "assault",
        "accident", "crash", "collision", "fatal", "tragedy",
        "found in trunk", "found in car"
    ]
    if any(term in h for term in crime_terms):
        return False

    # -----------------------------
    # 3. MarketBeat 13F SPAM Filter
    # -----------------------------
    if "marketbeat" in s:
        spam = [
            "lowers position", "decreases", "trims",
            "reduced stake", "stake lowered", "cuts holdings",
            "reduces position", "sells stake",
            "grows position", "boosts stake", "increases position",
            "raises holdings", "adds shares",
            "makes new investment", "buys stake",
            "portfolio update"
        ]
        if any(p in h for p in spam):
            return False

    # -----------------------------
    # 4. SEC / EDGAR Whitelist
    # -----------------------------
    sec_forms = [
        "8-k", "10-k", "10-q", "20-f",
        "13d", "13g", "sc 13d", "sc 13g",
        "form 4", "form 5"
    ]
    if any(form in h for form in sec_forms):
        return True
    if "edgar" in s:
        return True

    # -----------------------------
    # 5. MUST mention symbol/company OR entity OR synonym
    # -----------------------------
    def has_symbol_or_company():
        if sym and re.search(rf"\b{re.escape(sym)}\b", h):
            return True
        if cname and cname in h:
            return True
        if any(cname == o or sym == o for o in orgs):
            return True

        # fallback synonyms
        if allow_synonyms:
            syns = generate_synonyms_for_company(company_name, symbol)
            for s2 in syns:
                if len(s2) >= 3 and s2 in h:
                    return True
        return False

    direct_hit = has_symbol_or_company()

    # -----------------------------
    # 6. Competitor Negative Space (Generic, Multi-Sector)
    # -----------------------------
    competitors = [
        # Auto/EV
        "ford", "gm", "general motors", "toyota", "volkswagen",
        "rivian", "lucid", "byd", "nio", "xpeng", "hyundai", "kia",

        # Tech
        "google", "alphabet", "amazon", "meta", "apple", "microsoft",
        "nvidia", "amd", "intel",

        # Banking (US)
        "jp morgan", "jpmorgan", "goldman", "morgan stanley", "wells fargo",
        "bank of america", "citigroup",

        # Banking (India)
        "hdfc", "icici", "sbi", "kotak", "axis bank",

        # Indian corporates
        "reliance", "tcs", "infosys", "wipro", "mahindra", "tata"
    ]

    competitor_hit = any(c in h for c in competitors)

    # -----------------------------
    # 7. MATERIAL COMPETITOR READ-THROUGH (Generic)
    # -----------------------------
    # If competitor news is about the SAME SECTOR and affects valuation,
    # we KEEP IT even if symbol not mentioned.
    sector_keywords = [
        "ev", "electric vehicle", "battery", "recall",
        "production halt", "factory delay", "chip shortage",
        "margin pressure", "regulatory probe",
        "price cuts", "demand slowdown",
        "earnings miss", "profit warning",
    ]

    material_readthrough = competitor_hit and any(kw in h for kw in sector_keywords)

    if material_readthrough:
        return True

    # If competitor mentioned but no symbol/company and no readthrough → reject
    if competitor_hit and not direct_hit:
        return False

    # -----------------------------
    # 8. Final Decision
    # -----------------------------
    return direct_hit


# -------------------------
# Cache / DB utilities
# -------------------------

def init_cache_db():
    # Ensure directory exists (if CACHE_DB is a path)
    db_dir = os.path.dirname(CACHE_DB) or None
    if db_dir:
        try:
            os.makedirs(db_dir, exist_ok=True)
        except Exception:
            pass

    with sqlite3.connect(CACHE_DB) as conn:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS articles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                headline TEXT NOT NULL,
                summary TEXT,
                source TEXT,
                published TEXT,
                engagement INTEGER,
                fetch_date TEXT,
                media_url TEXT,
                aspects TEXT,
                aspect_sentiment TEXT,
                entities TEXT,
                sentiment_label TEXT,
                sentiment_score REAL,
                sentiment_num REAL,
                weight REAL,
                weighted_score REAL,
                UNIQUE(symbol, headline)
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS daily_sentiment (
                symbol TEXT NOT NULL,
                date TEXT NOT NULL,
                score REAL NOT NULL,
                article_count INTEGER DEFAULT 0,
                relevant_rate REAL DEFAULT 0.0,
                updated_at TEXT,
                UNIQUE(symbol, date)
            )
        """)
        conn.commit()
        logger.info("Cache DB initialized: %s", CACHE_DB)


init_cache_db()


def load_cache(symbol: str, days: int) -> pd.DataFrame:
    cutoff = (datetime.utcnow().replace(tzinfo=timezone.utc) - timedelta(days=days)).isoformat()
    try:
        with sqlite3.connect(CACHE_DB) as conn:
            df = pd.read_sql_query(
                "SELECT * FROM articles WHERE symbol=? AND published >= ?",
                conn, params=(symbol, cutoff)
            )
        if df.empty:
            return pd.DataFrame()

        df['aspects'] = df.get('aspects', pd.Series([None] * len(df))).apply(lambda x: safe_json_load(x, {}))
        df['aspect_sentiment'] = df.get('aspect_sentiment', pd.Series([None] * len(df))).apply(lambda x: safe_json_load(x, {}))
        df['entities'] = df.get('entities', pd.Series([None] * len(df))).apply(lambda x: safe_json_load(x, {"organizations": [], "persons": [], "locations": []}))
        return df
    except Exception as e:
        logger.exception("Cache load error: %s", e)
        return pd.DataFrame()


def save_cache(symbol: str, df: pd.DataFrame):
    if df is None or df.empty:
        return

    df_to_save = df.copy()
    df_to_save['symbol'] = symbol

    # Convert Timestamp-like objects to iso strings for SQLite
    if 'published' in df_to_save.columns:
        df_to_save['published'] = df_to_save['published'].apply(
            lambda x: x.isoformat() if hasattr(x, 'isoformat') else str(x)
        )
    if 'fetch_date' in df_to_save.columns:
        df_to_save['fetch_date'] = df_to_save['fetch_date'].apply(
            lambda x: x.isoformat() if hasattr(x, 'isoformat') else str(x)
        )

    # JSON serialization for structured columns (preserve existing strings)
    df_to_save['aspects'] = df_to_save.get('aspects', pd.Series([{}] * len(df_to_save))).apply(lambda x: json.dumps(x) if not isinstance(x, str) else x)
    df_to_save['aspect_sentiment'] = df_to_save.get('aspect_sentiment', pd.Series([{}] * len(df_to_save))).apply(lambda x: json.dumps(x) if not isinstance(x, str) else x)
    df_to_save['entities'] = df_to_save.get('entities', pd.Series([{}] * len(df_to_save))).apply(lambda x: json.dumps(x) if not isinstance(x, str) else x)

    cols = ['symbol', 'headline', 'summary', 'source', 'published', 'engagement', 'fetch_date', 'media_url', 'aspects', 'aspect_sentiment', 'entities', 'sentiment_label', 'sentiment_score', 'sentiment_num', 'weight', 'weighted_score']
    df_to_save = df_to_save[[c for c in cols if c in df_to_save.columns]]

    try:
        with sqlite3.connect(CACHE_DB) as conn:
            cur = conn.cursor()
            colnames = ', '.join(df_to_save.columns)
            placeholders = ', '.join(['?'] * len(df_to_save.columns))

            insert_sql = f"INSERT OR IGNORE INTO articles ({colnames}) VALUES ({placeholders})"
            cur.executemany(insert_sql, df_to_save.values.tolist())
            conn.commit()
            logger.info("Saved %d rows to cache for %s", len(df_to_save), symbol)
    except Exception as e:
        logger.exception("Cache save error: %s", e)

# -------------------------
# Market helpers
# -------------------------

def fetch_yahoo_close(symbol: str, days: int = 30) -> pd.Series:
    end = pd.Timestamp.now(tz=timezone.utc).normalize()
    start = end - pd.Timedelta(days=days + 15)
    hist = _call_with_timeout(
        lambda: yf.download(
            symbol,
            start=start.date().isoformat(),
            end=(end + pd.Timedelta(days=1)).date().isoformat(),
            progress=False,
            auto_adjust=True,
        ),
        timeout=YF_TIMEOUT_SECONDS,
        default=None,
    )
    if hist is None or hist.empty or "Close" not in hist.columns:
        return pd.Series(dtype=float)

    # normalize index to UTC midnight
    idx = pd.to_datetime(hist.index, utc=True).normalize()
    hist.index = idx
    return hist["Close"].dropna()

# -------------------------
# Weighting, DSP & Granger
# -------------------------

def compute_dynamic_weight(
    source: str,
    published: Any,
    engagement: int,
    form_type: str = None
) -> float:
    """
    V32.1 PRODUCTION WEIGHTING ENGINE
    - Clean canonical lookup
    - Correct EDGAR/SEC/NSE/BSE detection
    - True half-life decay (HL=10 days)
    - Engagement small and stable
    - Institutional clamp [0.10, 2.00]
    """

    # ----------------------------------------------------------
    # 1. CANONICAL SOURCE LOOKUP (NO FUZZY MATCHING)
    # ----------------------------------------------------------
    s_raw = (source or "").strip()
    s_key = s_raw.title()  # canonical form for config

    # direct lookup first
    base = BASE_SOURCE_WEIGHTS.get(s_key, None)

    # lowercase fallback once
    if base is None:
        s_lower = s_raw.lower()
        for k, v in BASE_SOURCE_WEIGHTS.items():
            if k.lower() == s_lower:
                base = v
                break

    # if still missing: neutral
    if base is None:
        base = 1.0

    # ----------------------------------------------------------
    # 2. TRUE HALF-LIFE DECAY (HL = 10 days)
    # ----------------------------------------------------------
    try:
        pub_dt = (
            published if isinstance(published, datetime)
            else pd.to_datetime(published, utc=True, errors='coerce')
        )
    except:
        pub_dt = None

    if pub_dt is None or pd.isna(pub_dt):
        # median recency, not inflated fallback
        recency = math.exp(-1.0)   # ≈ 0.367
    else:
        now = pd.Timestamp.now(tz=timezone.utc)
        age_days = max(0.0, (now - pub_dt).total_seconds() / 86400.0)
        k = math.log(2) / 10.0     # HL=10
        recency = math.exp(-k * age_days)

    # ----------------------------------------------------------
    # 3. ENGAGEMENT (bounded)
    # ----------------------------------------------------------
    try:
        eng = min(max(int(engagement or 0), 0), 5000)
    except:
        eng = 0

    eng_score = math.log1p(eng) / 12.0

    # ----------------------------------------------------------
    # 4. SEC/NSE/BSE BOOST (fixed detection)
    # ----------------------------------------------------------
    sec_adj = 1.00
    s_upper = s_raw.upper()

    if any(tag in s_upper for tag in ["EDGAR", "SEC", "NSE", "BSE"]):
        if form_type:
            ft = str(form_type).upper().strip()

            if ft == "4":             sec_adj = 1.08
            elif "13D" in ft:         sec_adj = 1.12
            elif "13G" in ft:         sec_adj = 1.05
            elif ft in {"10-K", "10-Q"}: sec_adj = 1.04
            elif ft == "8-K":         sec_adj = 1.06

    # ----------------------------------------------------------
    # 5. FINAL CALC
    # ----------------------------------------------------------
    w = base * sec_adj * recency * (1.0 + eng_score)

    # institutional clamp
    w = max(0.10, min(w, 2.00))

    return float(w)
# ==========================================
# V25 DSP HELPERS
# ==========================================

def normalize_sentiment_direction(scores: np.ndarray, threshold: float):
    return np.where(scores > threshold, 1.0,
           np.where(scores < -threshold, -1.0, 0.0))

def compute_magnitude(scores: np.ndarray, threshold: float):
    mag = (np.abs(scores) - threshold) / (1.0 - threshold)
    return np.clip(mag, 0.0, 1.0)

def apply_decay_weights(weights: np.ndarray, ages: np.ndarray, halflife: float = 7.0):
    decay = np.exp(-ages / halflife)
    return weights * decay

def volatility_scaling(dsp_raw: float, price_vol: float, target_vol: float = 0.02):
    if price_vol is None or price_vol < 1e-8:
        return dsp_raw
    scale = target_vol / price_vol
    scale = np.clip(scale, 0.25, 4.0)
    return dsp_raw * scale

def compute_age_days(df: pd.DataFrame, time_col: str = "published"):
    now = pd.Timestamp.now(tz=timezone.utc)
    df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    df["age_days"] = (now - df[time_col]).dt.total_seconds() / 86400.0
    df["age_days"] = df["age_days"].clip(lower=0.0, upper=365.0)
    return df

def compute_price_vol(price_df: pd.DataFrame):
    if price_df is None or price_df.empty:
        return None
    returns = price_df["close"].pct_change()
    if len(returns) < 20:
        return None
    vol = returns.rolling(20).std().iloc[-1]
    if pd.isna(vol) or vol < 0:
        return None
    return float(vol)

def calculate_institutional_dsp(
    df: pd.DataFrame,
    price_vol: float = None,
    threshold: float = 0.20,
    neutral_discount: float = 0.15,
    clip_abs: float = 3.0
):
    if df is None or df.empty:
        return 0.0

    scores = df["sentiment_score"].fillna(0.0).astype(float).values
    weights = df["weight"].fillna(1.0).astype(float).values
    ages = df.get("age_days", pd.Series([0]*len(df))).astype(float).values

    # Direction & Magnitude
    direction = normalize_sentiment_direction(scores, threshold)
    magnitude = compute_magnitude(scores, threshold)

    # Decay-weighted weights
    decay_weights = apply_decay_weights(weights, ages, halflife=7.0)

    # Neutral discount
    effective_weights = np.where(
        magnitude == 0.0,
        decay_weights * neutral_discount,
        decay_weights
    )

    # Bearish asymmetry
    asym = np.where(direction < 0, 1.20, 1.00)
    effective_weights *= asym

    numerator = np.sum(direction * magnitude * effective_weights)
    denom = np.sqrt(np.sum(effective_weights ** 2))
    if denom < 1e-8:
        return 0.0

    dsp_raw = numerator / denom
    dsp_raw = float(np.clip(dsp_raw, -clip_abs, clip_abs))

    if price_vol is not None:
        dsp_raw = volatility_scaling(dsp_raw, price_vol)

    return dsp_raw



def causal_validation(symbol: str, df: pd.DataFrame, days: int) -> Union[float, None]:
    """Read daily_sentiment history and run Granger causality against returns.
    Aligns by normalized UTC date index. Requires >=10 matching days.
    """
    try:
        with sqlite3.connect(CACHE_DB) as conn:
            history = pd.read_sql_query(
                "SELECT date, score FROM daily_sentiment WHERE symbol=? ORDER BY date ASC",
                conn, params=(symbol,)
            )
    except Exception as e:
        logger.exception("Failed to load daily_sentiment: %s", e)
        return None

    if history.empty or len(history) < 10:
        return None

    # parse history dates safely to normalized UTC datetimes
    history['dt'] = pd.to_datetime(history['date'], utc=True).dt.normalize()
    history = history.set_index('dt')

    try:
        start = history.index.min().date()
        end = datetime.utcnow().date()

        hist = yf.download(symbol, start=start.isoformat(), end=(end + timedelta(days=1)).isoformat(), progress=False, auto_adjust=True)
        if hist.empty:
            return None

        price_idx = pd.to_datetime(hist.index, utc=True).normalize()
        price_returns = hist['Close'].pct_change().dropna()
        # align returns to normalized datetime index (skip first NaN)
        price_returns.index = price_idx[1: 1 + len(price_returns)]

        # intersection on DatetimeIndex
        common = price_returns.index.intersection(history.index)
        if len(common) < 10:
            return None

        sent_series = history.loc[common]['score'].astype(float)
        ret_series = price_returns.loc[common].astype(float)

        data = pd.DataFrame({
            "price": ret_series,
            "sent": sent_series
        })

        data = data.dropna()
        if data.shape[0] < 10:
            return None

        res = grangercausalitytests(data[["price", "sent"]], maxlag=2, verbose=False)
        pvals = [res[i + 1][0]['ssr_ftest'][1] for i in range(2)]
        return float(min(pvals))

    except Exception as e:
        logger.exception("Causal validation failed: %s", e)
        return None

# -------------------------
# Main pipeline
# -------------------------

def clean_text_for_sentiment(headline: str, summary: str) -> str:
    s = (str(headline or "") + ". " + str(summary or "")).strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"(click here|read more|learn more|sign up|subscribe|sponsored by)", "", s, flags=re.I)
    return s.strip()


async def get_news_sentiment_async(symbol: str, days: int = 10, debug: bool = False) -> Union[Dict, pd.DataFrame]:
    logger.info("Analyzing %s for %d days", symbol, days)

    cached_df = load_cache(symbol, days)

    try:
        company_name = _call_with_timeout(
            lambda: yf.Ticker(symbol).info.get("longName", symbol),
            timeout=YF_TIMEOUT_SECONDS,
            default=symbol,
        )
    except Exception:
        company_name = symbol

    query = f"{company_name} OR {symbol}"
    cutoff = datetime.utcnow().replace(tzinfo=timezone.utc) - timedelta(days=days)

    try:
        fetched = await asyncio.wait_for(
            aggregate_sources(query, symbol, cutoff),
            timeout=FETCH_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "News fetch exceeded %.0fs for %s — continuing with cached articles only",
            FETCH_TIMEOUT_SECONDS,
            symbol,
        )
        fetched = []
    if not fetched and (cached_df is None or cached_df.empty):
        return {"global_score": 0.0, "error": "No articles"}

    new_df = pd.DataFrame(fetched)

    # Strict timestamp cleaning
    if not new_df.empty:
        new_df['published'] = pd.to_datetime(new_df.get('published'), utc=True, errors='coerce')
        new_df = new_df.dropna(subset=['published'])

    # remove headlines already seen
    if not new_df.empty and cached_df is not None and not cached_df.empty:
        new_df = new_df[~new_df['headline'].isin(cached_df['headline'])]

    processed = pd.DataFrame()
    if not new_df.empty:
        processed = new_df.copy()
        processed['fetch_date'] = datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()

        # extract entities from original text (headline + summary) to preserve NER accuracy
        original_texts = (processed.get('headline', '').fillna('') + ". " + processed.get('summary', '').fillna('')).tolist()
        ents = []
        try:
            for t in original_texts:
                ent = extract_entities_transformer(t)
                if not isinstance(ent, dict):
                    ent = safe_json_load(ent, {"organizations": [], "persons": [], "locations": []})
                # ensure keys exist
                if 'organizations' not in ent:
                    ent['organizations'] = []
                if 'persons' not in ent:
                    ent['persons'] = []
                if 'locations' not in ent:
                    ent['locations'] = []
                ents.append(ent)
        except Exception:
            ents = [{"organizations": [], "persons": [], "locations": []} for _ in range(len(processed))]
        processed['entities'] = ents

        # cleaned text for sentiment models
        processed['text'] = [clean_text_for_sentiment(h, s) for h, s in zip(processed.get('headline', '').fillna(''), processed.get('summary', '').fillna(''))]

        processed['text_lower'] = processed['text'].astype(str)
        processed['entities_parsed'] = processed['entities'].apply(lambda x: x if isinstance(x, dict) else safe_json_load(x, {}))

        # relevance filter (R2). Uses original NER-derived entities.
        processed['is_relevant'] = processed.apply(
            lambda r: True if r.get('source') == 'EDGAR' else is_relevant_article(symbol, company_name, r.get('headline', ''), r['entities_parsed'], allow_synonyms=True, source=r.get('source', '')),
            axis=1
        )

        if FULL_DEBUG:
            for _, row in processed.iterrows():
                if not row['is_relevant']:
                    print(f"[FILTERED-OUT] {row.get('headline')}  -- Reason: not relevant to {symbol}")

        processed = processed[processed['is_relevant']].reset_index(drop=True)

        if not processed.empty:
            processed['aspects'] = [extract_aspects_from_doc(t, ASPECT_KEYWORDS) for t in processed['text'].tolist()]

            sentiments = [
                compute_ensemble_sentiment(text, snippets, ASPECT_KEYWORDS)
                for text, snippets in zip(processed['text'].tolist(), processed['aspects'].tolist())
            ]
            sentiments_df = pd.DataFrame(sentiments, index=processed.index)

            processed['sentiment_label'] = sentiments_df['label']
            processed['sentiment_score'] = sentiments_df['score']
            processed['sentiment_num'] = sentiments_df['sentiment_num']
            processed['aspect_sentiment'] = sentiments_df['aspect_sentiment']

            processed['weight'] = processed.apply(lambda r: compute_dynamic_weight(r.get('source', ''), r.get('published', ''), r.get('engagement', 0),r.get('form_type', None)), axis=1)

            save_cache(symbol, processed)

    # Merge cached + processed
    if processed is None or processed.empty:
        final_df = cached_df.copy() if cached_df is not None else pd.DataFrame()
    else:
        if cached_df is None or cached_df.empty:
            final_df = processed.copy()
        else:
            final_df = pd.concat([cached_df, processed], ignore_index=True)

    # Dedupe by normalized headline using deterministic ordering by fetch_date (most recent wins)
    if final_df is None or final_df.empty:
        return {"global_score": 0.0, "error": "No data post-process"}

    final_df['norm_headline'] = final_df['headline'].fillna('').apply(normalize_headline)
    if 'fetch_date' in final_df.columns:
        final_df['fetch_date'] = pd.to_datetime(final_df['fetch_date'], utc=True, errors='coerce')
    else:
        final_df['fetch_date'] = pd.NaT

    final_df = (
        final_df.sort_values(['norm_headline', 'fetch_date'], ascending=[True, True])
                .drop_duplicates(subset=['norm_headline'], keep='last')
                .reset_index(drop=True)
    )

    # Final hygiene
    final_df['published'] = pd.to_datetime(final_df['published'], utc=True, errors='coerce')
    final_df = final_df.dropna(subset=['published']).reset_index(drop=True)

    final_df['fused_text'] = final_df.apply(lambda r: process_multimodal(r.get('text', ''), r.get('media_url', None)), axis=1)

    # Compute global DSP
    # Add article ages for decay
    final_df = compute_age_days(final_df, time_col="published")
    global_sentiment_score = float(calculate_institutional_dsp(final_df, price_vol=None, threshold=RISK_THRESHOLD))

    # Ensure gated_direction exists for validation logic
    final_df['sentiment_score'] = final_df.get('sentiment_score', 0.0).fillna(0.0).astype(float)
    final_df['weight'] = final_df.get('weight', 1.0).fillna(1.0).astype(float)

    final_df['gated_direction'] = np.where(
        final_df['sentiment_score'] > RISK_THRESHOLD, 1.0,
        np.where(final_df['sentiment_score'] < -RISK_THRESHOLD, -1.0, 0.0)
    )

    # Attempt causal validation (Granger)
    granger_p = None
    try:
        granger_p = causal_validation(symbol, final_df, days)
    except Exception as e:
        logger.exception("Granger validation failed: %s", e)
        granger_p = None

    # Entities
    all_orgs = []
    for ent in final_df.get('entities', []):
        parsed = {}
        if isinstance(ent, str):
            try:
                parsed = json.loads(ent)
            except Exception:
                parsed = {}
        else:
            parsed = ent or {}
        all_orgs.extend(parsed.get('organizations', []))
    top_entities = [it for it, _ in Counter(all_orgs).most_common(5) if it and it.lower() != company_name.lower()]

    # Aspect aggregation
    aspect_summary = {a: {"label": "Neutral", "score": 0.0} for a in ASPECT_CATEGORIES}
    for a in ASPECT_CATEGORIES:
        mask = final_df['aspect_sentiment'].apply(lambda x: isinstance(x, dict) and isinstance(x.get(a), dict))
        if mask.any():
            labels = final_df.loc[mask, 'aspect_sentiment'].apply(lambda d: d[a]['label'])
            scores = final_df.loc[mask, 'aspect_sentiment'].apply(lambda d: float(d[a]['score']))
            aspect_summary[a] = {
                "label": labels.mode().iloc[0] if not labels.empty else "Neutral",
                "score": float(scores.mean()) if not scores.empty else 0.0
            }

        # DEBUG: Signal Validation prints (improved formatting + 5 items + entities)
    if (FULL_DEBUG or debug) and not final_df.empty:
        base_weights = final_df['weight'].fillna(1.0).astype(float).values
        sent_vals = final_df['sentiment_score'].fillna(0.0).astype(float).values
        direction = final_df['gated_direction'].astype(float).values

        magnitude = np.clip((np.abs(sent_vals) - RISK_THRESHOLD) / (1.0 - RISK_THRESHOLD + 1e-12),
                            0.0, 1.0)
        signed_weights = direction * magnitude * base_weights
        adjusted_weights = np.where(magnitude == 0,
                                    base_weights * 0.15,
                                    base_weights)
        denom = float(np.sqrt(np.sum(adjusted_weights ** 2))) or 1.0

        final_df = final_df.copy()
        final_df['real_impact_raw'] = direction * base_weights
        final_df['real_impact_signed'] = signed_weights
        final_df['real_impact_norm'] = final_df['real_impact_signed'] / denom

        bears = final_df[final_df['real_impact_norm'] < 0]\
                    .sort_values(by='real_impact_norm', ascending=True)\
                    .head(5)

        bulls = final_df[final_df['real_impact_norm'] > 0]\
                    .sort_values(by='real_impact_norm', ascending=False)\
                    .head(5)

        print("\n--- SIGNAL VALIDATION (BULLS VS BEARS) ---")

        print("--- TOP BEARS (Pulling Down) ---")
        for _, row in bears.iterrows():
            print(f"[-] {row.get('headline')} "
                  f"(Impact: {row['real_impact_norm']:.4f})")
            print(f"     • norm_headline: {row.get('norm_headline')}")
            print(f"     • entities: {row.get('entities')}")
            print()

        print("--- TOP BULLS (Pushing Up) ---")
        for _, row in bulls.iterrows():
            print(f"[+] {row.get('headline')} "
                  f"(Impact: {row['real_impact_norm']:.4f})")
            print(f"     • norm_headline: {row.get('norm_headline')}")
            print(f"     • entities: {row.get('entities')}")
            print()

        pos_count = (final_df['real_impact_raw'] > 0).sum()
        neg_count = (final_df['real_impact_raw'] < 0).sum()
        print(f"Total Positive Articles: {int(pos_count)}")
        print(f"Total Negative Articles: {int(neg_count)}")
        print("--------------------------------------\n")
        # ===========================================================
    # ALWAYS SHOW TOP IMPACT ARTICLES (AFFECTING GLOBAL DSP)
    # ===========================================================
    try:
        impact_df = final_df.copy()
        impact_df['sentiment_score'] = impact_df['sentiment_score'].astype(float)
        impact_df['weight'] = impact_df['weight'].astype(float)
        impact_df['direction'] = impact_df['gated_direction'].astype(float)

        magnitude = np.clip(
            (impact_df['sentiment_score'].abs() - RISK_THRESHOLD) /
            (1.0 - RISK_THRESHOLD + 1e-12),
            0.0, 1.0
        )
        impact_df['magnitude'] = magnitude

        impact_df['impact_raw'] = (
            impact_df['direction'] *
            impact_df['magnitude'] *
            impact_df['weight']
        )
        impact_df['impact_abs'] = impact_df['impact_raw'].abs()

        top_impact = impact_df.sort_values(
            'impact_abs', ascending=False
        ).head(5)
        
        impact_articles = (
        top_impact[[
            "headline",
            "source",
            "published",
            "impact_raw",
            "magnitude",
            "weight",
            "direction",
            "entities"
        ]]
        .rename(columns={
            "impact_raw": "impact"
        })
        .to_dict(orient="records")
        )

        print("\n--- TOP IMPACT ARTICLES (AFFECTING GLOBAL DSP) ---")
        for _, row in top_impact.iterrows():
            direction = "+" if row['impact_raw'] > 0 else "-"
            print(f"[{direction}] {row.get('headline')}  (impact: {row['impact_raw']:.4f})")
            print(f"     magnitude={row['magnitude']:.4f}, weight={row['weight']:.4f}, direction={row['direction']}")
            print(f"     entities: {row.get('entities')}")
        print("--------------------------------------------------\n")

    except Exception as e:
        print(f"[Impact Block Error] {e}")


    # Save daily snapshot with metadata (upsert)
    if not final_df.empty:
        today_str = datetime.now(tz=timezone.utc).date().isoformat()
        daily_score = global_sentiment_score
        article_count = int(len(final_df))
        relevant_rate = float(final_df.get('is_relevant', pd.Series([True] * len(final_df))).mean())
        now_iso = datetime.now(tz=timezone.utc).isoformat()

        params = (symbol, today_str, daily_score, article_count, relevant_rate, now_iso)
        try:
            with sqlite3.connect(CACHE_DB) as conn:
                cur = conn.cursor()
                try:
                    cur.execute("""
                        INSERT INTO daily_sentiment (symbol, date, score, article_count, relevant_rate, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                        ON CONFLICT(symbol, date) DO UPDATE SET
                            score=excluded.score,
                            article_count=excluded.article_count,
                            relevant_rate=excluded.relevant_rate,
                            updated_at=excluded.updated_at
                    """, params)
                except sqlite3.OperationalError:
                    # fallback for older SQLite: use INSERT OR REPLACE
                    cur.execute("""
                        INSERT OR REPLACE INTO daily_sentiment (symbol, date, score, article_count, relevant_rate, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, params)
                conn.commit()
                if FULL_DEBUG:
                    print(f"[DAILY] Saved daily sentiment {symbol} {today_str} -> {daily_score:.4f} (count={article_count}, relevant_rate={relevant_rate:.2f})")
        except Exception as e:
            logger.exception("[daily_sentiment] Failed to save history: %s", e)

    if debug:
        try:
            final_df['explanation'] = explain_sentiment(final_df['fused_text'].tolist())
        except Exception as e:
            logger.exception("explain_sentiment error: %s", e)
        return final_df.sort_values('published', ascending=False)
       
    


    return {
        "global_score": float(global_sentiment_score),
        "aspects": aspect_summary,
        "impact_articles": impact_articles,
        "granger_predictive_p_value": granger_p,
        "top_entities_mentioned": top_entities
    }


# Pretty print helper

def pretty_print_sentiment(result: dict, symbol: str):
    print("\n==============================")
    print(f"   NEWS SENTIMENT SUMMARY: {symbol}")
    print("==============================\n")

    g = result.get("global_score", 0.0)
    print(f"GLOBAL SENTIMENT SCORE: {g:.4f} ({'Positive' if g>0 else 'Negative' if g<0 else 'Neutral'})\n")

    print("ASPECT SENTIMENTS:")
    aspects = result.get("aspects", {})
    for a, v in aspects.items():
        label = v.get("label", "Neutral")
        score = v.get("score", 0.0)
        print(f"  - {a.capitalize():12s} : {label:8s} | {score:+.4f}")
    print()

    ents = result.get("top_entities_mentioned", [])
    if ents:
        print("TOP ENTITIES MENTIONED:")
        for e in ents:
            print(f"  - {e}")
        print()

    gr = result.get("granger_predictive_p_value", None)
    if gr is not None:
        print(f"Granger Predictive p-value: {gr:.5f}\n")
    else:
        print("Granger Predictive p-value: None\n")

    print("==============================\n")
    
import asyncio

def _run_async(coro):
    """
    Safely run async code from sync context (Gradio / Colab safe).
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        return asyncio.run_coroutine_threadsafe(coro, loop).result()
    else:
        return asyncio.run(coro)


def run_sentiment_pipeline(symbol: str, days: int = 7) -> dict:
    import numpy as np
    import pandas as pd
    
    symbol = str(symbol).upper()
    
    
    cutoff = datetime.utcnow().replace(tzinfo=timezone.utc) - timedelta(days=days)
    raw = _run_async(aggregate_sources(symbol, symbol, cutoff))

    # -------- HARD NORMALIZATION BOUNDARY --------
    if raw is None:
        return {"dsp_score": 0.0, "aspects": {}, "impact_articles": []}
    import pandas as pd
    if isinstance(raw, list):
        if len(raw) == 0:
            return {"dsp_score": 0.0, "aspects": {}, "impact_articles": []}
        df = pd.DataFrame(raw)
    elif isinstance(raw, pd.DataFrame):
        df = raw 
    else:
        raise TypeError(f"aggregate_sources returned unsupported type: {type(raw)}")
    if df.empty:
        return {"dsp_score": 0.0, "aspects": {}, "impact_articles": []}
    
    df = compute_age_days(df)

    df["weight"] = df.apply(
        lambda r: compute_dynamic_weight(
            r.get("source"),
            r.get("published"),
            r.get("engagement", 0),
            r.get("form_type")
        ),
        axis=1
    )
    # -------- SCHEMA HARDENING (REQUIRED) --------
    # Ensure required columns always exist, even if empty or keys missing

    if "sentiment_score" not in df.columns:
        df["sentiment_score"] = 0.0

    if "aspect_sentiment" not in df.columns:
        df["aspect_sentiment"] = [{} for _ in range(len(df))]

    if "weight" not in df.columns:
        df["weight"] = 1.0

# -------------------------------------------

    dsp_raw = calculate_institutional_dsp(df)
    dsp_score = float(np.clip(dsp_raw, -1.0, 1.0))

    aspect_scores = {}
    for aspect in ASPECT_CATEGORIES.keys():
        vals = []
        for x in df["aspect_sentiment"]:
            if isinstance(x, dict) and aspect in x:
                vals.append(float(x[aspect].get("score", 0.0)))
        aspect_scores[aspect] = float(np.clip(np.mean(vals) if vals else 0.0, -1.0, 1.0))

    df["impact"] = df["sentiment_score"].astype(float) * df["weight"].astype(float)
    top = df.reindex(df["impact"].abs().sort_values(ascending=False).index).head(5)

    articles = [
        {
            "headline": r["headline"],
            "source": r["source"],
            "impact": float(np.clip(r["impact"], -1.0, 1.0))
        }
        for _, r in top.iterrows()
    ]

    return {
        "dsp_score": dsp_score,
        "aspects": aspect_scores,
        "impact_articles": articles
    }
