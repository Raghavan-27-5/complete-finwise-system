# FinWise Core: Canonical System Specification

**Document Version:** 1.0.0  
**Generated:** 2026-01-17  
**Classification:** Internal Engineering Document  
**Status:** Technical Specification — Source of Truth

---

## Table of Contents

1. [Executive Overview](#1-executive-overview)
2. [High-Level System Architecture](#2-high-level-system-architecture)
3. [Repository & Folder Structure](#3-repository--folder-structure)
4. [Core Design Philosophy](#4-core-design-philosophy)
5. [Data Sources & Inputs](#5-data-sources--inputs)
6. [Data Pipeline (End-to-End)](#6-data-pipeline-end-to-end)
7. [Intelligence Layer](#7-intelligence-layer)
8. [Models](#8-models)
9. [Execution & Orchestration](#9-execution--orchestration)
10. [APIs & Interfaces](#10-apis--interfaces)
11. [State Management & Persistence](#11-state-management--persistence)
12. [Error Handling & Failure Modes](#12-error-handling--failure-modes)
13. [Security & Trust Boundaries](#13-security--trust-boundaries)
14. [Performance Characteristics](#14-performance-characteristics)
15. [Deployment & Environment](#15-deployment--environment)
16. [Current Project Status](#16-current-project-status)
17. [Known Limitations & Technical Debt](#17-known-limitations--technical-debt)
18. [Extension Points](#18-extension-points)
19. [Glossary](#19-glossary)

---

## 1. Executive Overview

### 1.1 What This Project Is

FinWise Core is a **financial intelligence platform** that combines quantitative price forecasting with qualitative sentiment analysis to produce actionable investment signals. The system is designed as a modular pipeline consisting of:

1. **Price Intelligence (P1)**: LSTM-based time-series forecasting with Monte Carlo risk simulation using a Heston-like stochastic volatility model.

2. **Sentiment Intelligence (P2)**: Multi-source news aggregation with FinBERT-based sentiment scoring, relevance filtering, and aspect-level decomposition.

3. **Knowledge Graph**: Neo4j AuraDB integration for temporal state tracking of financial snapshots.

4. **Conversational AI**: LLM-powered (OpenRouter/Gemini) natural language interface for querying the knowledge graph.

5. **Dual UI Applications**: Streamlit (chat-focused research) and Gradio (quant-focused risk terminal).

### 1.2 What Problem It Solves

The system attempts to synthesize **information from 11+ data sources** (APIs, RSS feeds, SEC filings, social media) into a unified "financial state" object containing:

- Trend direction (Bullish/Bearish/Sideways)
- Price forecast with uncertainty bounds
- Downside risk (5% Value at Risk)
- Sentiment score (Directional Sentiment Pressure)
- Aspect-level sentiment breakdown (earnings, leadership, legal, etc.)
- High-impact articles driving sentiment

This enables a user to obtain a holistic view of a stock's current condition without manually monitoring multiple sources.

### 1.3 Who It Is For

| Persona | Use Case |
|---------|----------|
| **Retail Investor** | Quick sentiment and trend check before trading |
| **Financial Analyst** | Research augmentation with structured sentiment data |
| **Quantitative Trader** | Risk metrics (VaR, Monte Carlo bounds) for position sizing |
| **Developer** | Extensible platform for building financial applications |

### 1.4 What It Explicitly Does NOT Try to Do

| Non-Goal | Explanation |
|----------|-------------|
| **Real-time trading** | No sub-second latency guarantees. Not designed for HFT. |
| **Portfolio management** | Does not track positions, P&L, or execute trades. |
| **Regulatory compliance** | No GDPR/CCPA/SEC compliance built-in. |
| **Options/derivatives analysis** | Price intelligence covers equities only. Options flow is not implemented. |
| **Multi-asset class** | Primarily equities (US and Indian markets). No crypto, forex, or commodities. |
| **Production-grade reliability** | Current state is prototype. Multiple brittleness points documented in audits. |

---

## 2. High-Level System Architecture

### 2.1 Major Subsystems

The system is organized into six logical layers:

```
┌─────────────────────────────────────────────────────────────────┐
│                     APPLICATION LAYER                            │
│  ┌─────────────────────┐  ┌─────────────────────┐               │
│  │  Streamlit App      │  │  Gradio App         │               │
│  │  (Chat + Research)  │  │  (Risk Terminal)    │               │
│  └─────────┬───────────┘  └─────────┬───────────┘               │
│            │                        │                            │
└────────────┼────────────────────────┼────────────────────────────┘
             │                        │
             ▼                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                        CORE LAYER                                │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  core/orchestrator.py                                    │    │
│  │  - Coordinates P1 (Price) and P2 (Sentiment) pipelines   │    │
│  │  - Writes combined state to Knowledge Graph              │    │
│  └─────────────────────────────────────────────────────────┘    │
│  ┌─────────────────────┐  ┌─────────────────────┐               │
│  │  core/config.py     │  │  core/llm_provider.py│              │
│  │  (Weights, Aspects) │  │  (OpenRouter/Gemini) │              │
│  └─────────────────────┘  └─────────────────────┘               │
└─────────────────────────────────────────────────────────────────┘
             │                        │
             ▼                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                    INTELLIGENCE LAYER                            │
│  ┌───────────────────────────┐  ┌───────────────────────────┐   │
│  │  intelligence/price/      │  │  intelligence/sentiment/  │   │
│  │  ├─ predictor.py          │  │  ├─ pipeline.py           │   │
│  │  │  - LSTM forecasting    │  │  │  - DSP calculation     │   │
│  │  │  - Heston Monte Carlo  │  │  │  - Aspect aggregation  │   │
│  │  │  - VaR calculation     │  │  ├─ fetchers.py           │   │
│  │  └─ adapter.py            │  │  │  - 11 data sources     │   │
│  │                           │  │  ├─ models.py             │   │
│  │                           │  │  │  - FinBERT sentiment   │   │
│  │                           │  │  │  - SBERT similarity    │   │
│  │                           │  │  └─ adapter.py            │   │
│  └───────────────────────────┘  └───────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
             │                        │
             ▼                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                       DATA LAYER                                 │
│  ┌───────────────────────────┐  ┌───────────────────────────┐   │
│  │  kg/ (Knowledge Graph)    │  │  recall_engine/           │   │
│  │  ├─ kg_writer.py          │  │  ├─ chatbot.py            │   │
│  │  │  - Neo4j write ops     │  │  ├─ conversation_manager  │   │
│  │  ├─ kg_schema.py          │  │  ├─ database_manager.py   │   │
│  │  │  - Node/Rel types      │  │  ├─ query_generator.py    │   │
│  │  └─ state_adapter.py      │  │  └─ nlp_processor.py      │   │
│  │      - Dataclass mapping  │  │                           │   │
│  └───────────────────────────┘  └───────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
             │                        │
             ▼                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                   EXTERNAL SERVICES                              │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐            │
│  │ Neo4j    │ │ Yahoo    │ │ NewsAPI  │ │ SEC      │            │
│  │ AuraDB   │ │ Finance  │ │          │ │ EDGAR    │ ... (11+)  │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘            │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Data Flow

**Primary Analysis Flow:**

1. **User Input**: Symbol (e.g., "AAPL") and forecast horizon (e.g., 7 days) entered via Gradio or Streamlit.

2. **Orchestration** (`core/orchestrator.py`):
   - Calls `build_price_intelligence(symbol, days)` → returns `p1` dict
   - Calls `build_sentiment_intelligence(symbol, days)` → returns `p2` dict
   - Constructs `raw_state` combining both
   - Adapts to typed `State` dataclass
   - Writes snapshot to Neo4j via `kg_writer.write_snapshot()`

3. **Price Intelligence** (`intelligence/price/predictor.py`):
   - Downloads 3 years of OHLCV data from Yahoo Finance
   - Scales data and runs through pre-trained LSTM model
   - Performs walk-forward prediction for `days` trading days
   - Runs Heston Monte Carlo simulation (5000 paths)
   - Calculates 5% VaR (Value at Risk)
   - Returns: history, forecast, monte_carlo, indicators, trend

4. **Sentiment Intelligence** (`intelligence/sentiment/pipeline.py`):
   - Fetches articles from 11 sources concurrently via `aggregate_sources()`
   - Applies R2 relevance filter to remove noise
   - Scores each article with FinBERT
   - Extracts aspect-level sentiment using SBERT similarity
   - Calculates institutional DSP (Directional Sentiment Pressure)
   - Caches results in SQLite
   - Returns: global_score, aspects, impact_articles, granger_p_value

5. **Knowledge Graph Write**:
   - Creates/updates `Stock` node
   - Creates `Snapshot` node with timestamp
   - Creates `Signal` nodes for each metric (trend, forecast_slope, downside_risk, global_score, label)
   - Creates/links `Aspect` nodes for aspect-level sentiment

6. **UI Rendering**:
   - Streamlit: Chat interface with context-aware LLM responses
   - Gradio: Risk terminal with Plotly visualizations

### 2.3 Control Flow and Orchestration

The orchestration layer (`core/orchestrator.compute_state()`) is **synchronous and blocking**. The sentiment pipeline uses async internally but is wrapped in a sync executor for compatibility.

```python
def compute_state(symbol: str, days: int, kg_writer):
    p1 = build_price_intelligence(symbol, days)  # Blocking
    p2 = build_sentiment_intelligence(symbol, days)  # Blocking (async wrapped)
    
    raw_state = {...}  # Combine p1 and p2
    
    state = adapt_state(raw_state)  # Convert to typed dataclass
    kg_writer.write_snapshot(state)  # Write to Neo4j
    
    return raw_state
```

---

## 3. Repository & Folder Structure

### 3.1 Full Directory Tree

```
finwise_core/
├── .env                        # Environment variables (secrets)
├── .env.example                # Template for environment variables
├── .gitignore                  # Git ignore rules
├── pytest.ini                  # Pytest configuration
├── requirements.txt            # Python dependencies (63 packages)
├── README.md                   # Project overview and quick start
├── FINWISE_STRATEGIC_ROADMAP.md # Future development roadmap
├── SYSTEM_SPECIFICATION.md     # This document
├── sentiment_cache_v20.db      # SQLite cache for sentiment articles
│
├── app/                        # Application entry points
│   ├── __init__.py
│   ├── streamlit_app.py        # Chat-focused research UI (309 lines)
│   └── gradio_app.py           # Quant-focused risk terminal (277 lines)
│
├── core/                       # Shared core modules
│   ├── __init__.py
│   ├── config.py               # Configuration constants (195 lines)
│   ├── orchestrator.py         # State computation logic (35 lines)
│   └── llm_provider.py         # LLM abstraction layer (173 lines)
│
├── intelligence/               # Intelligence engines
│   ├── __init__.py
│   ├── price/                  # P1: Price Intelligence
│   │   ├── __init__.py
│   │   ├── adapter.py          # Thin adapter layer
│   │   ├── predictor.py        # LSTM + Monte Carlo (729 lines)
│   │   └── stock_price_model.h5 # Pre-trained Keras LSTM model (682 KB)
│   │
│   └── sentiment/              # P2: Sentiment Intelligence
│       ├── __init__.py
│       ├── adapter.py          # Thin adapter layer
│       ├── pipeline.py         # Main sentiment pipeline (1175 lines)
│       ├── fetchers.py         # News/RSS/API data fetchers (585 lines)
│       └── models.py           # FinBERT + spaCy models (447 lines)
│
├── kg/                         # Knowledge Graph
│   ├── __init__.py
│   ├── kg_writer.py            # Neo4j write operations (133 lines)
│   ├── kg_schema.py            # Graph schema definitions (38 lines)
│   └── state_adapter.py        # State transformation (52 lines)
│
├── recall_engine/              # Conversational AI (READ-ONLY)
│   ├── __init__.py
│   ├── chatbot.py              # LLM chat functions (84 lines)
│   ├── conversation_manager.py # Session state management (130 lines)
│   ├── database_manager.py     # Neo4j read queries (46 lines)
│   ├── llm_query_generator.py  # NL → Cypher conversion (173 lines)
│   ├── nlp_processor.py        # Entity extraction (125 lines)
│   └── query_generator.py      # Query orchestration (207 lines)
│
├── audits/                     # Engineering audit documents
│   ├── SCRAPER_FIX_PLAN.md     # Plan to fix broken scrapers
│   ├── SCRAPER_LIVE_AUDIT.md   # Live test results for scrapers
│   ├── SYSTEM_FORENSIC_AUDIT.md # Comprehensive code audit
│   └── TEST_AUDIT_REPORT.md    # Test coverage analysis
│
└── tests/                      # Test suite
    ├── __init__.py
    ├── conftest.py             # Pytest fixtures (146 lines)
    ├── unit/                   # Unit tests
    │   ├── test_config.py
    │   ├── test_orchestrator.py
    │   ├── test_price_predictor.py
    │   ├── test_recall_engine.py
    │   └── test_sentiment_pipeline.py
    ├── integration/            # Integration tests
    │   └── test_full_pipeline.py
    ├── live/                   # Live API tests
    │   ├── test_api_connections.py
    │   └── test_scrapers.py
    ├── smoke/                  # Smoke tests for UIs
    │   ├── test_gradio_app.py
    │   └── test_streamlit_app.py
    └── mocks/                  # Test mock objects
        ├── mock_fetchers.py
        ├── mock_gemini.py
        ├── mock_neo4j.py
        └── mock_yfinance.py
```

### 3.2 Directory Purpose and Criticality

| Directory | Purpose | What Breaks If Removed |
|-----------|---------|------------------------|
| `app/` | User interfaces | No way to interact with system |
| `core/` | Orchestration and config | Entire pipeline fails to execute |
| `intelligence/price/` | Price forecasting | No price predictions, Monte Carlo, VaR |
| `intelligence/sentiment/` | Sentiment analysis | No sentiment scores, aspect analysis |
| `kg/` | Knowledge graph integration | State snapshots not persisted to Neo4j |
| `recall_engine/` | Conversational AI | Streamlit chat functionality broken |
| `audits/` | Documentation only | No runtime impact |
| `tests/` | Test suite | No runtime impact, but no verification |

---

## 4. Core Design Philosophy

### 4.1 Architectural Principles

1. **Modular Intelligence Engines**: Each "P" engine (P1: Price, P2: Sentiment) is self-contained with its own adapter layer. Future engines (P3: Options, P4: Alternative Data) can be added following the same pattern.

2. **Adapter Pattern**: Thin adapter modules (`adapter.py`) expose a standardized interface (`build_X_intelligence(symbol, days) → dict`) isolating implementation details.

3. **Read-Write Separation**: The `recall_engine/` is explicitly READ-ONLY from the knowledge graph. All writes go through `kg/kg_writer.py`.

4. **Configuration Externalization**: Weights, aspect keywords, and model parameters are defined in `core/config.py`, not scattered in implementation files.

5. **LLM Provider Abstraction**: The system supports multiple LLM backends (OpenRouter, Gemini) through a unified interface (`core/llm_provider.py`) with automatic fallback.

### 4.2 Constraints

| Constraint Type | Description |
|-----------------|-------------|
| **Technical** | Python 3.12+, TensorFlow 2.x, requires GPU for optimal FinBERT performance |
| **Financial** | Free-tier API limits (NewsAPI: 100/day, Alpha Vantage: 25/day, Reddit: 60/min) |
| **Operational** | Requires Neo4j AuraDB instance, at least 12 API keys for full functionality |
| **Data** | Primarily US and Indian equities. No real-time streaming. |

### 4.3 Trade-offs Consciously Made

| Trade-off | Chosen | Rejected | Rationale |
|-----------|--------|----------|-----------|
| **Sync vs Async** | Synchronous orchestration with async fetchers | Fully async pipeline | Simpler debugging, easier state management |
| **Model serving** | In-process model loading | External model server (TF Serving) | Reduced operational complexity for prototype |
| **Caching** | SQLite for articles | Redis/PostgreSQL | Zero infrastructure dependency for local dev |
| **Sentiment model** | FinBERT with SBERT hybrid | Pure LLM-based analysis | Deterministic, cost-effective, no API calls |

### 4.4 Trade-offs Explicitly Rejected

| Rejected Approach | Reason for Rejection |
|-------------------|----------------------|
| Real-time WebSocket feeds | Overkill for current batch-oriented use case |
| Microservices architecture | Prototype phase; monolith is faster to iterate |
| Custom model training | Pre-trained models sufficient for MVP |
| Multi-tenant SaaS design | Not a priority for initial development |

---

## 5. Data Sources & Inputs

### 5.1 External Data Sources

The sentiment pipeline fetches from 11 distinct data sources:

| Source | Type | Rate Limit | Status | Implementation |
|--------|------|------------|--------|----------------|
| **NewsAPI** | REST API | 100/day | ✅ Working | `fetch_newsapi()` |
| **Google RSS** | RSS Feed | Unlimited | ✅ Working | `fetch_google_rss_sync()` |
| **Yahoo Finance News** | yfinance lib | Unofficial | ✅ Working | `fetch_yahoo_sync()` |
| **Yahoo Earnings** | yfinance lib | Unofficial | ✅ Working | `fetch_yahoo_earnings_sync()` |
| **SEC EDGAR** | REST API | 10 req/sec | ✅ Working | `fetch_edgar_filings()` |
| **Reddit** | PRAW API | 60/min | ⚠️ Requires keys | `fetch_reddit_posts()` |
| **X/Twitter** | REST API | 5/day (Academic) | ❌ Not implemented | Placeholder only |
| **Alpha Vantage** | REST API | 25/day | ⚠️ Optional | `fetch_alpha_transcripts()` |
| **Moneycontrol** | Web Scraping | N/A | ⚠️ Flaky (CSS drift) | `fetch_moneycontrol()` |
| **SeekingAlpha** | Web Scraping | N/A | ❌ Broken (bot detection) | `fetch_seekingalpha()` |
| **MarketWatch** | Web Scraping | N/A | ❌ Broken (JS rendering) | `fetch_marketwatch()` |
| **NSE/BSE (India)** | REST/XML API | N/A | ⚠️ Flaky | `fetch_indian_filings()` |

### 5.2 Internal Data Generation

| Data | Source | Storage |
|------|--------|---------|
| LSTM predictions | `predictor.py` | In-memory only |
| Monte Carlo paths | `predictor.py` | In-memory only |
| Sentiment scores | `pipeline.py` | SQLite cache |
| Knowledge graph snapshots | `kg_writer.py` | Neo4j AuraDB |

### 5.3 Assumptions About Data Quality

| Assumption | Reality |
|------------|---------|
| Yahoo Finance returns complete OHLCV | Generally true for major equities |
| NewsAPI returns relevant financial news | Varies; includes general business news |
| SEC EDGAR filings are timely | Yes, but 8-K filings may lag hours |
| Company names resolve to correct symbols | Manual mapping required for ambiguous names |

### 5.4 Data Source Failure Modes

| Source | Failure Mode | Impact | Mitigation |
|--------|--------------|--------|------------|
| NewsAPI | API key exhausted (100/day) | No primary news data | Fallback to Google RSS |
| Yahoo Finance | yfinance library deprecated | No price data | Upgrade library or switch to Finnhub |
| SEC EDGAR | Rate limit exceeded | No filing data | Built-in rate limiter (8/sec) |
| Web Scrapers | CSS selector drift | 0 articles returned | **No mitigation currently** |
| Neo4j | Connection timeout | State not persisted | **No retry logic currently** |

---

## 6. Data Pipeline (End-to-End)

### 6.1 Ingestion

**Price Data Ingestion:**
```python
# intelligence/price/predictor.py
def get_stock_data(stock_symbol):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 3)  # 3 years
    df = safe_download(stock_symbol, start=start_date, end=end_date)
    return preprocess_data(df)  # Normalize column names, set DatetimeIndex
```

**Sentiment Data Ingestion:**
```python
# intelligence/sentiment/fetchers.py
async def aggregate_sources(query: str, symbol: str, cutoff: datetime):
    async with aiohttp.ClientSession() as session:
        tasks = [
            fetch_newsapi(session, query, cutoff),
            asyncio.to_thread(fetch_google_rss_sync, query, cutoff),
            # ... 9 more sources
        ]
        all_data = await asyncio.gather(*tasks, return_exceptions=True)
```

### 6.2 Validation

**Article Relevance Validation (R2 Filter):**
```python
# intelligence/sentiment/pipeline.py
def is_relevant_article(symbol, company_name, headline, entities, ...):
    # 1. Reject noise patterns (price prediction, top X stocks, etc.)
    # 2. Reject crime/accident headlines
    # 3. Reject MarketBeat 13F position-change spam
    # 4. Whitelist SEC filings (8-K, 10-K, etc.)
    # 5. Require symbol/company mention OR synonym match
    # 6. Allow material competitor read-through (EV, banking, etc.)
```

**Data Type Validation:**
- Timestamps converted to UTC-aware datetime
- NaN values in published dates cause article rejection
- JSON columns (`aspects`, `entities`) safely parsed with fallback to empty dict

### 6.3 Cleaning

**Headline Normalization:**
```python
def normalize_headline(h: str) -> str:
    s = unescape(h)  # HTML entities
    s = s.replace('…', ' ').replace('—', ' ')  # Unicode normalization
    s = re.sub(r"[^\w\s]", " ", s)  # Remove punctuation
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s
```

**Text Cleaning for Sentiment:**
```python
def clean_text_for_sentiment(headline: str, summary: str) -> str:
    s = headline + ". " + summary
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"(click here|read more|...)", "", s, flags=re.I)
    return s.strip()
```

### 6.4 Transformation

**Sentiment Scoring:**
```python
# intelligence/sentiment/models.py
def llm_sentiment(text: str) -> Dict[str, Any]:
    results = _aspect_model(text, top_k=None)  # FinBERT
    p_pos = scores.get('positive', 0.0)
    p_neg = scores.get('negative', 0.0)
    sentiment_val = p_pos - p_neg  # Continuous [-1, +1]
    return {"score": sentiment_val, "label": label}
```

**DSP (Directional Sentiment Pressure) Calculation:**
```python
# intelligence/sentiment/pipeline.py
def calculate_institutional_dsp(df, price_vol, threshold=0.20):
    direction = np.where(scores > threshold, 1.0,
               np.where(scores < -threshold, -1.0, 0.0))
    magnitude = (np.abs(scores) - threshold) / (1.0 - threshold)
    decay_weights = np.exp(-ages / 7.0)  # 7-day half-life
    
    # Bearish asymmetry (1.20x weight for negative)
    # Neutral discount (0.15x weight for neutral)
    
    numerator = np.sum(direction * magnitude * effective_weights)
    denom = np.sqrt(np.sum(effective_weights ** 2))
    return numerator / denom
```

### 6.5 Storage

**SQLite Sentiment Cache:**
```sql
CREATE TABLE articles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    headline TEXT NOT NULL,
    summary TEXT,
    source TEXT,
    published TEXT,
    engagement INTEGER,
    fetch_date TEXT,
    media_url TEXT,
    aspects TEXT,          -- JSON
    aspect_sentiment TEXT, -- JSON
    entities TEXT,         -- JSON
    sentiment_label TEXT,
    sentiment_score REAL,
    sentiment_num REAL,
    weight REAL,
    weighted_score REAL,
    UNIQUE(symbol, headline)
);

CREATE TABLE daily_sentiment (
    symbol TEXT NOT NULL,
    date TEXT NOT NULL,
    score REAL NOT NULL,
    article_count INTEGER DEFAULT 0,
    relevant_rate REAL DEFAULT 0.0,
    updated_at TEXT,
    UNIQUE(symbol, date)
);
```

**Neo4j Knowledge Graph:**
- See Section 11 for schema details.

### 6.6 Retrieval

**Cache Loading:**
```python
def load_cache(symbol: str, days: int) -> pd.DataFrame:
    cutoff = datetime.utcnow() - timedelta(days=days)
    with sqlite3.connect(CACHE_DB) as conn:
        df = pd.read_sql_query(
            "SELECT * FROM articles WHERE symbol=? AND published >= ?",
            conn, params=(symbol, cutoff.isoformat())
        )
```

### 6.7 Error Handling at Each Stage

| Stage | Error Type | Handling |
|-------|------------|----------|
| Ingestion | Network timeout | Retry 3x, then log error, return empty |
| Validation | Invalid date format | Skip article |
| Cleaning | Non-string input | Convert to string or skip |
| Transformation | Model inference failure | Return neutral (0.0) score |
| Storage | SQLite lock | `INSERT OR IGNORE` |
| Retrieval | Cache miss | Fetch from sources |

---

## 7. Intelligence Layer

### 7.1 Price Intelligence Module (`intelligence/price/`)

**Purpose:** Generate probabilistic price forecasts and risk metrics.

**Components:**

| Component | Function | Algorithm |
|-----------|----------|-----------|
| `get_stock_data()` | Fetch 3 years OHLCV | Yahoo Finance API |
| `prepare_data()` | Scale to [0,1] | MinMaxScaler |
| `predict_future()` | Walk-forward forecast | Pre-trained LSTM |
| `monte_carlo_heston()` | Risk simulation | Heston stochastic vol |
| `monte_carlo_heston_stats_only()` | VaR calculation | 5000 paths, 5th percentile |

**Deterministic vs Probabilistic:**
- LSTM prediction is **deterministic** (no dropout at inference)
- Monte Carlo is **probabilistic** (random walks with stochastic volatility)

**Decision Thresholds:**

| Signal | Bullish | Bearish | Sideways |
|--------|---------|---------|----------|
| Trend slope | > 0 | < 0 | = 0 |
| RSI | < 30 (oversold) | > 70 (overbought) | 30-70 |
| MACD | > 0 | < 0 | N/A |

### 7.2 Sentiment Intelligence Module (`intelligence/sentiment/`)

**Purpose:** Aggregate and score financial news for directional sentiment.

**Components:**

| Component | Function | Algorithm |
|-----------|----------|-----------|
| `aggregate_sources()` | Parallel fetch from 11 sources | asyncio.gather |
| `is_relevant_article()` | R2 relevance filter | Rule-based + synonym matching |
| `extract_aspects_from_doc()` | Map sentences to aspects | SBERT cosine similarity |
| `compute_ensemble_sentiment()` | Score article | FinBERT sentiment classification |
| `compute_dynamic_weight()` | Weight by source/recency | Half-life decay (10 days) |
| `calculate_institutional_dsp()` | Global sentiment score | Weighted, gated aggregation |
| `causal_validation()` | Validate predictive power | Granger causality test |

**Aspect Categories (15 total):**

| Aspect ID | Description |
|-----------|-------------|
| `earnings_revenue` | Earnings, revenue, forecasts |
| `costs_margins` | Costs, expenses, margin changes |
| `liquidity_balance` | Cash flow, debt, refinancing |
| `dividends_buybacks` | Dividends, share repurchases |
| `leadership` | CEO/CFO actions, management |
| `corporate_strategy` | Strategic pivots, restructuring |
| `mna_partnerships` | M&A, joint ventures |
| `legal_regulatory` | Lawsuits, regulatory actions |
| `competition` | Competitive pressure |
| `product_services` | Product launches, recalls |
| `technology_innovation` | Patents, R&D, breakthroughs |
| `supply_chain` | Manufacturing, logistics |
| `customer_experience` | Customer satisfaction |
| `brand_reputation` | Brand image, scandals |
| `labor_workforce` | Hiring, layoffs, unions |

**Source Weighting (Tier System):**

| Tier | Sources | Base Weight |
|------|---------|-------------|
| 1 (Institutional) | Reuters, Bloomberg, WSJ, FT, CNBC | 1.10 - 1.15 |
| 2 (Aggregators) | Yahoo Finance, MarketWatch, SeekingAlpha | 0.95 - 1.05 |
| 3 (Social) | Reddit, StockTwits, X | 0.20 - 0.30 |
| 4 (Low-quality) | MarketBeat, Zacks, InvestorPlace, Motley Fool | 0.40 - 0.50 |
| 5 (Official) | SEC EDGAR, NSE, BSE | 1.20 |

---

## 8. Models

### 8.1 LSTM Price Prediction Model

**Model Type:** Keras Sequential LSTM

**File:** `intelligence/price/stock_price_model.h5` (682 KB)

**Architecture:** [UNKNOWN — NOT SPECIFIED in code, only loaded via `load_model()`]

**Training Assumptions:**
- [UNKNOWN — NOT IMPLEMENTED] Training code not present in repository
- Model appears pre-trained externally

**Inputs:**
- Shape: `(batch_size, 60, 1)` — 60-day lookback of scaled close prices

**Outputs:**
- Shape: `(batch_size, 1)` — Next day scaled close price

**Inference Workflow:**
```python
y_pred = model.predict(X)  # Scaled prediction
y_pred = scaler.inverse_transform(y_pred)  # Convert to actual price
```

**Limitations:**
- Walk-forward prediction compounds error exponentially beyond 7 days
- No uncertainty quantification at model level

### 8.2 FinBERT Sentiment Model

**Model Type:** Hugging Face Transformers (`ProsusAI/finbert`)

**Loading:**
```python
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained(
    "ProsusAI/finbert", 
    device_map="auto"
)
```

**Inputs:**
- Text string (max 1024 characters)

**Outputs:**
- Label probabilities: `positive`, `negative`, `neutral`
- Continuous score: `p_positive - p_negative` ∈ [-1, +1]

**Fallback:**
- If FinBERT fails, falls back to generic HuggingFace sentiment pipeline

### 8.3 SBERT Similarity Model

**Model Type:** Sentence Transformers (`all-MiniLM-L6-v2`)

**Purpose:** Map sentences to aspects via cosine similarity

**Inputs:**
- Sentence embeddings (384 dimensions)
- Aspect keyword embeddings (pre-computed, cached)

**Outputs:**
- Best-matching aspect for each sentence (similarity threshold: 0.32)

### 8.4 spaCy NER Model

**Model Type:** spaCy (`en_core_web_sm`, fallback to blank English)

**Purpose:** Named Entity Recognition for relevance filtering

**Entity Types Used:**
- `ORG` / `NORP` → Organizations
- `PERSON` → Persons
- `GPE` / `LOC` / `FAC` → Locations

---

## 9. Execution & Orchestration

### 9.1 Entry Points

| Entry Point | Command | Purpose |
|-------------|---------|---------|
| Streamlit UI | `streamlit run app/streamlit_app.py` | Chat + research interface |
| Gradio UI | `python app/gradio_app.py` | Risk terminal interface |
| Direct Import | `from core.orchestrator import compute_state` | Programmatic access |

### 9.2 Runtime Dependencies

| Dependency | Version | Purpose |
|------------|---------|---------|
| Python | 3.12+ | Runtime |
| TensorFlow | 2.20+ | LSTM model loading |
| PyTorch | Latest | FinBERT inference |
| Neo4j Driver | Latest | Knowledge graph |
| spaCy | 3.7.6 | NER processing |

### 9.3 Scheduling / Triggering Logic

**Current State:** No automated scheduling. All analysis is on-demand.

**Potential Scheduling Points:**
- Daily sentiment cache refresh (not implemented)
- Scraper health checks (documented in roadmap, not implemented)

---

## 10. APIs & Interfaces

### 10.1 Internal Python API

**Orchestrator:**
```python
from core.orchestrator import compute_state

state = compute_state(
    symbol="AAPL",
    days=7,
    kg_writer=kg_writer_instance
)
# Returns dict with: symbol, as_of, price, sentiment, history, forecast, 
#                    monte_carlo, indicators, articles
```

**Price Intelligence:**
```python
from intelligence.price.adapter import build_price_intelligence

p1 = build_price_intelligence(symbol="AAPL", days=7)
# Returns dict with: trend, forecast_slope, downside_risk, history, 
#                    forecast, monte_carlo, indicators
```

**Sentiment Intelligence:**
```python
from intelligence.sentiment.adapter import build_sentiment_intelligence

p2 = build_sentiment_intelligence(symbol="AAPL", days=7)
# Returns dict with: global_score, label, aspects, impact_articles
```

### 10.2 LLM Provider Interface

```python
from core.llm_provider import get_llm_client, LLMClient

client: LLMClient = get_llm_client()  # Auto-selects OpenRouter or Gemini
response = client.generate("Your prompt here")  # Returns str
response = client.generate_content("Prompt")  # Backwards-compatible
```

### 10.3 No External HTTP API

[UNKNOWN — NOT IMPLEMENTED] The system does not expose an HTTP REST API. All interaction is through UI or programmatic Python imports.

---

## 11. State Management & Persistence

### 11.1 State Categories

| State Type | Persistence | Location | Lifetime |
|------------|-------------|----------|----------|
| Sentiment cache | Persistent | SQLite file | 4-hour TTL per article |
| Daily sentiment scores | Persistent | SQLite file | Indefinite |
| Knowledge graph snapshots | Persistent | Neo4j AuraDB | Indefinite |
| Price data cache | In-memory | `_YF_CACHE` dict | Never expires (memory leak) |
| Conversation context | In-memory | `deque(maxlen=5)` | Session only |
| Streamlit session state | In-memory | `st.session_state` | Session only |

### 11.2 Neo4j Knowledge Graph Schema

**Node Types:**

| Label | Properties | Description |
|-------|------------|-------------|
| `Stock` | `symbol: string` | Company/ticker entity |
| `Snapshot` | `symbol: string, as_of: string` | Point-in-time state |
| `Signal` | `symbol, as_of, name, value, direction` | Individual metric |
| `Aspect` | `name: string` | Sentiment aspect category |

**Relationship Types:**

| Type | From | To | Description |
|------|------|----|----|
| `HAS_SNAPSHOT` | Stock | Snapshot | Stock has temporal snapshots |
| `HAS_SIGNAL` | Snapshot | Signal | Snapshot contains signals |
| `OF_ASPECT` | Signal | Aspect | Signal relates to aspect |

**Constraints:**
```cypher
CREATE CONSTRAINT stock_symbol_unique FOR (s:Stock) REQUIRE s.symbol IS UNIQUE
CREATE CONSTRAINT aspect_name_unique FOR (a:Aspect) REQUIRE a.name IS UNIQUE
CREATE CONSTRAINT snapshot_symbol_as_of_unique FOR (sn:Snapshot) 
    REQUIRE (sn.symbol, sn.as_of) IS UNIQUE
```

### 11.3 Recovery Behavior

| Failure Scenario | Recovery Behavior |
|------------------|-------------------|
| SQLite file corrupted | Tables recreated on next init |
| Neo4j connection lost | **No automatic retry** — returns error |
| Conversation context lost | User starts fresh session |
| YF cache corrupted | N/A (in-memory only) |

---

## 12. Error Handling & Failure Modes

### 12.1 Expected Failures

| Failure | Cause | Handling |
|---------|-------|----------|
| API rate limit exceeded | NewsAPI 100/day limit | Log warning, skip source |
| Network timeout | Slow external service | Retry 3x, then skip |
| Invalid stock symbol | User typo | Return error message to UI |
| No articles found | New symbol or quiet news day | Return 0.0 sentiment score |

### 12.2 Unexpected Failures

| Failure | Cause | Handling |
|---------|-------|----------|
| FinBERT OOM | Large text on GPU | **No handling** — crashes |
| Neo4j SSL error | Certificate issues | **No handling** — raises exception |
| LSTM model file missing | Accidental deletion | **No handling** — crashes on import |

### 12.3 Graceful Degradation

| Component Failure | Degraded Behavior |
|-------------------|-------------------|
| NewsAPI unavailable | Google RSS provides headlines |
| FinBERT model fails | Falls back to generic sentiment pipeline |
| Yahoo Finance fails | **No fallback** — price intelligence fails |
| Neo4j unavailable | State not persisted, raw_state still returned |

### 12.4 Silent Failure Risks

| Risk | Description |
|------|-------------|
| Broken web scrapers | Return empty list, not exception. System appears functional with no sentiment data. |
| Stale cache | Articles with future cache TTL never refreshed |
| Memory leak | `_YF_CACHE` grows indefinitely, no monitoring |

---

## 13. Security & Trust Boundaries

### 13.1 What Is Trusted

| Component | Trust Level | Reason |
|-----------|-------------|--------|
| SEC EDGAR | High | Official government source |
| Yahoo Finance | Medium | Widely used, stable |
| FinBERT model | Medium | Pre-trained on financial corpus |
| User input (symbol) | Low | Must be validated |
| LLM responses | Low | Subject to hallucination |

### 13.2 What Is Not Trusted

| Component | Risk |
|-----------|------|
| Web-scraped content | SEO spam, marketing, fake news |
| Reddit posts | Manipulation, pump-and-dump |
| User-provided Cypher queries | Injection risk (via LLM generation) |

### 13.3 Attack Surfaces

| Surface | Risk | Mitigation |
|---------|------|------------|
| LLM prompt injection | Malicious context manipulation | **Not mitigated** |
| Cypher injection | DB modification via LLM query | **Not mitigated** — validated by LLM only |
| API key exposure | `.env` in git history | `.gitignore` present |
| Denial of service | Expensive LLM/model calls | **No rate limiting** |

### 13.4 Explicit Security Non-Goals

| Non-Goal | Reason |
|----------|--------|
| User authentication | Prototype phase, single-user |
| Encryption at rest | SQLite cache contains public data |
| Audit logging | Not required for prototype |
| Input sanitization | Partially implemented via R2 filter |

---

## 14. Performance Characteristics

### 14.1 Bottlenecks

| Operation | Time (Approximate) | Bottleneck |
|-----------|-------------------|------------|
| Yahoo Finance download | 2-5 seconds | Network I/O |
| LSTM prediction | 0.5-2 seconds | CPU/GPU bound |
| Monte Carlo (5000 paths) | 1-3 seconds | CPU bound (NumPy) |
| Sentiment fetch (11 sources) | 5-15 seconds | Network I/O (parallel) |
| FinBERT inference (per article) | 0.1-0.3 seconds | GPU bound |
| Neo4j write | 0.5-2 seconds | Network I/O |

**Total `compute_state()` latency:** 15-30 seconds typical

### 14.2 Scalability Limits

| Dimension | Limit | Reason |
|-----------|-------|--------|
| Concurrent users | 1-5 | Single-process, synchronous |
| Symbols per day | ~50 | API rate limits (NewsAPI, Alpha Vantage) |
| Articles cached | 10,000+ | SQLite performance degrades |
| Neo4j writes | ~100/min | Free-tier rate limits |

### 14.3 Resource Usage

| Resource | Typical Usage |
|----------|---------------|
| RAM (idle) | 1-2 GB (models loaded) |
| RAM (peak) | 4-6 GB (FinBERT + TensorFlow) |
| Disk | ~700 MB (h5 model + SQLite cache) |
| GPU VRAM | 2-4 GB (if CUDA available) |

---

## 15. Deployment & Environment

### 15.1 Expected Runtime Environment

| Requirement | Specification |
|-------------|---------------|
| OS | Windows, Linux, macOS |
| Python | 3.12+ |
| Memory | 8 GB minimum, 16 GB recommended |
| GPU | Optional (CUDA for faster inference) |
| Network | Outbound HTTPS to APIs |

### 15.2 Environment Variables

| Variable | Required | Purpose |
|----------|----------|---------|
| `AURA_CONNECTION_URI` | ✅ | Neo4j AuraDB connection string |
| `AURA_USERNAME` | ✅ | Neo4j username |
| `AURA_PASSWORD` | ✅ | Neo4j password |
| `OPENROUTER_API_KEY` | ⭐ Recommended | OpenRouter LLM access |
| `OPENROUTER_MODEL` | ❌ Optional | Model selection (default: deepseek/deepseek-chat) |
| `GEMINI_API_KEY` | ⚠️ Fallback | Google Gemini API (deprecated) |
| `NEWS_API_KEY` | ✅ | NewsAPI access |
| `FINNHUB_API_KEY` | ⭐ Recommended | Alternative data source |
| `ALPHA_VANTAGE_KEY` | ❌ Optional | Company overview data |
| `REDDIT_CLIENT_ID` | ❌ Optional | Reddit API |
| `REDDIT_CLIENT_SECRET` | ❌ Optional | Reddit API |
| `REDDIT_USER_AGENT` | ❌ Optional | Reddit API |
| `FRED_API_KEY` | ❌ Optional | Macroeconomic data |
| `PIPELINE_DEBUG` | ❌ Optional | Enable verbose logging (0 or 1) |

### 15.3 Configuration in Constrained Environments

| Constraint | Impact | Workaround |
|------------|--------|------------|
| No GPU | Slower FinBERT inference | CPU fallback automatic |
| Low memory (< 4GB) | OOM on model load | Not supported |
| No network | Cannot fetch data | Not supported |
| Read-only filesystem | Cannot write SQLite cache | Mount writable volume |

---

## 16. Current Project Status

### 16.1 Complete Features

| Feature | Status | Notes |
|---------|--------|-------|
| LSTM price prediction | ✅ Complete | Pre-trained model loaded |
| Monte Carlo risk simulation | ✅ Complete | Heston-like model |
| Technical indicators (RSI, MACD) | ✅ Complete | via `ta` library |
| Multi-source news aggregation | ✅ Complete | 11 sources implemented |
| FinBERT sentiment scoring | ✅ Complete | With SBERT hybrid |
| Aspect-level sentiment | ✅ Complete | 15 aspect categories |
| DSP calculation | ✅ Complete | Institutional-grade formula |
| SQLite caching | ✅ Complete | Articles + daily scores |
| Neo4j knowledge graph | ✅ Complete | Write-only currently |
| Streamlit chat UI | ✅ Complete | With conversation history |
| Gradio risk terminal | ✅ Complete | With Plotly visualizations |
| LLM provider abstraction | ✅ Complete | OpenRouter + Gemini |

### 16.2 Partial Features

| Feature | Status | Missing |
|---------|--------|---------|
| Conversational AI context | 🟡 Partial | Lost on restart (in-memory) |
| Indian market support | 🟡 Partial | NSE/BSE scrapers flaky |
| Granger causality validation | 🟡 Partial | Requires >10 days history |
| Chart display in Streamlit | 🟡 Partial | `display_chart()` is placeholder |

### 16.3 Stubbed Features

| Feature | Status | Location |
|---------|--------|----------|
| X/Twitter fetcher | ❌ Stub | `fetch_x_posts()` returns `[]` |
| Image captioning (BLIP) | ❌ Commented out | `models.py:process_multimodal()` |
| Explain sentiment | ❌ Stub | `models.py:explain_sentiment()` |

### 16.4 Planned But Not Implemented

| Feature | Source |
|---------|--------|
| Options flow analysis | FINWISE_STRATEGIC_ROADMAP.md |
| Insider transaction tracking | FINWISE_STRATEGIC_ROADMAP.md |
| Circuit breakers | FINWISE_STRATEGIC_ROADMAP.md |
| Redis caching | FINWISE_STRATEGIC_ROADMAP.md |
| CI/CD pipelines | FINWISE_STRATEGIC_ROADMAP.md |

---

## 17. Known Limitations & Technical Debt

### 17.1 Explicit Weaknesses

| Weakness | Severity | Description |
|----------|----------|-------------|
| Memory leak | 🔴 Critical | `_YF_CACHE` never expires |
| Broken scrapers | 🔴 Critical | SeekingAlpha, MarketWatch return 0 articles |
| No retry logic | 🟠 High | External API failures not retried |
| In-memory context | 🟠 High | Conversation lost on restart |
| Hardcoded Heston params | 🟠 High | kappa=2.0, xi=0.15, rho=-0.4 |
| Synchronous orchestration | 🟡 Medium | Blocks UI thread during processing |
| No input validation | 🟡 Medium | Cypher injection possible via LLM |

### 17.2 Implicit Risks

| Risk | Likelihood | Impact |
|------|------------|--------|
| CSS drift breaks scrapers | High | Silent degradation |
| API deprecation (yfinance) | Medium | Price intelligence fails |
| FinBERT model updates | Low | Sentiment scoring changes |
| Neo4j free tier limits | Medium | Graph writes throttled |

### 17.3 Areas Likely to Fail Under Scale

| Area | Failure Mode |
|------|--------------|
| SQLite cache | Lock contention with >5 concurrent users |
| In-process models | OOM with large batch processing |
| Single-threaded orchestration | UI freezes during analysis |

---

## 18. Extension Points

### 18.1 Safe Extension Areas

| Extension | Location | Impact |
|-----------|----------|--------|
| New data source | `fetchers.py:aggregate_sources()` | Add to `tasks` list |
| New aspect category | `config.py:ASPECT_CATEGORIES` | Add key + keywords |
| New source weight | `config.py:BASE_SOURCE_WEIGHTS` | Add source name |
| New LLM provider | `llm_provider.py` | Implement `LLMClient` interface |
| New intelligence engine | `intelligence/` | Create P3/P4 directory with adapter |

### 18.2 Dangerous Modification Areas

| Area | Risk | Contract |
|------|------|----------|
| `kg_schema.py` | Graph structure change | Breaking for existing snapshots |
| `state_adapter.py` | Dataclass structure | Breaking for orchestrator |
| `pipeline.py:calculate_institutional_dsp()` | Sentiment scoring formula | Changes all historical comparisons |
| `predictor.py:TIME_STEP` | Must match model training | LSTM input size fixed at 60 |

### 18.3 Contracts That Must Not Be Broken

| Contract | Location | Consumers |
|----------|----------|-----------|
| `build_price_intelligence(symbol, days) → dict` | `price/adapter.py` | orchestrator, gradio_app |
| `build_sentiment_intelligence(symbol, days) → dict` | `sentiment/adapter.py` | orchestrator |
| `State` dataclass structure | `state_adapter.py` | kg_writer |
| SQLite table schema | `pipeline.py:init_cache_db()` | load_cache, save_cache |

---

## 19. Glossary

| Term | Definition |
|------|------------|
| **DSP** | Directional Sentiment Pressure. A weighted, gated aggregate of article sentiment scores ranging from -3.0 to +3.0, clipped to [-1.0, +1.0] for output. |
| **P1** | Price Intelligence module. Handles LSTM forecasting and Monte Carlo simulation. |
| **P2** | Sentiment Intelligence module. Handles news aggregation and sentiment scoring. |
| **R2 Filter** | Relevance filter (version 2). Rule-based article filtering using synonym matching and noise pattern detection. |
| **VaR** | Value at Risk. The 5th percentile of simulated final prices, representing potential downside loss. |
| **Heston Model** | Stochastic volatility model where variance follows a mean-reverting square-root process. Used in Monte Carlo simulation. |
| **Walk-forward** | Prediction method where each new prediction is fed back as input for the next timestep. |
| **FinBERT** | BERT model fine-tuned on financial text (`ProsusAI/finbert`) for sentiment classification. |
| **SBERT** | Sentence-BERT. Used to generate sentence embeddings for aspect similarity matching. |
| **Knowledge Graph** | Neo4j database storing temporal snapshots of financial state as nodes and relationships. |
| **Recall Engine** | Conversational AI subsystem that reads from the knowledge graph via LLM-generated Cypher queries. |
| **Aspect** | A thematic category (e.g., earnings, leadership, legal) for decomposing sentiment signals. |
| **Snapshot** | A point-in-time record of a stock's state (price signals + sentiment signals) in the knowledge graph. |
| **Signal** | An individual metric (trend, forecast_slope, downside_risk, global_score, aspect score) stored in the graph. |
| **TTL** | Time-to-Live. Cache expiration period (4 hours for sentiment articles). |
| **CIK** | Central Index Key. SEC identifier for companies, used to lookup EDGAR filings. |
| **EDGAR** | Electronic Data Gathering, Analysis, and Retrieval. SEC's filing database. |

---

## Document Metadata

| Field | Value |
|-------|-------|
| **Total Source Files Analyzed** | 23 |
| **Total Lines of Code Analyzed** | ~4,500 |
| **Repository Root** | `H:\finwise_core` |
| **Primary Language** | Python 3.12 |
| **Key Dependencies** | TensorFlow, PyTorch, Transformers, Neo4j, Streamlit, Gradio |
| **Document Author** | AI System Analyst |
| **Last Updated** | 2026-01-17 |

---

*End of System Specification*
