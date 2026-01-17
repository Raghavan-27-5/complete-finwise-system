# FinWise Core: Strategic Roadmap
## From Prototype to Production-Grade Financial Platform

---

## Executive Summary

You've built a functional financial intelligence platform with impressive capabilities: LSTM forecasting, FinBERT sentiment analysis, Neo4j knowledge graph, and Gemini-powered conversational AI. However, the existing forensic audit reveals significant brittleness (70% scraper failure rate, hardcoded parameters, in-memory state loss).

This roadmap addresses your 10 key concerns with **specific, actionable guidance**.

---

## 1. Identifying & Fixing Brittleness

### Current Brittleness Points (from audits)

| Component | Issue | Severity |
|-----------|-------|----------|
| Web Scrapers | 70% failure rate, CSS selector drift | 🔴 CRITICAL |
| Monte Carlo | Hardcoded kappa=2.0, xi=0.15, rho=-0.4 | 🟠 HIGH |
| `_YF_CACHE` | Global dict, never expires → memory leak | 🟠 HIGH |
| Conversation Memory | In-memory `deque()` → lost on restart | 🟠 HIGH |
| LSTM Predictions | Recursive feedback compounds error 7+ days | 🟡 MEDIUM |

### Fix Strategy

```mermaid
flowchart LR
    subgraph "Phase 1: Stabilize"
        A[Replace brittle scrapers] --> B[Add cache TTL/LRU limits]
        B --> C[Persist conversation state]
    end
    subgraph "Phase 2: Robustify"
        D[Calculate Heston params from data] --> E[Add circuit breakers]
        E --> F[Implement health checks]
    end
    A --> D
```

#### Quick Wins (Do First)
1. **Replace `_YF_CACHE` with LRU Cache**: Add TTL and max size
   ```python
   from cachetools import TTLCache
   _YF_CACHE = TTLCache(maxsize=100, ttl=3600)  # 1 hour TTL
   ```
2. **Persist Conversation State**: Switch from `deque()` to SQLite/Redis
3. **Add Scraper Health Monitoring**: Log zero-result fetches as CRITICAL alerts

---

## 2. Spotting Broken Things

### Automated Detection Strategy

| Layer | Detection Method | Tool |
|-------|------------------|------|
| **Data Sources** | Scheduled scraper tests | pytest + GitHub Actions cron |
| **API Endpoints** | Response validation | pydantic + pytest |
| **ML Models** | Prediction drift monitoring | evidently AI / whylabs |
| **Database** | Connection health checks | Built-in Neo4j driver health |

### Recommended Testing Structure

```
tests/
├── unit/              # Fast, mocked tests
├── integration/       # Real DB, mocked APIs
├── live/              # ACTUAL network calls (scheduled)
│   └── test_scraper_health.py  # ← Daily cron job
└── benchmarks/        # Performance regression
```

> [!IMPORTANT]
> Create a **daily GitHub Actions workflow** that runs `test_scraper_health.py` against live endpoints and alerts on failures.

---

## 3. Expanding the System

### Recommended Expansion Roadmap

```mermaid
timeline
    title Feature Expansion Timeline
    section Phase 1 (Stabilize)
        Replace brittle scrapers : Week 1-2
        Add health monitoring : Week 2-3
    section Phase 2 (Enhance)
        Options flow analysis : Week 4-6
        Insider transaction tracking : Week 5-7
        Alternative data (satellite/shipping) : Week 8+
    section Phase 3 (Scale)
        Multi-asset classes (Crypto, Forex) : Week 10+
        Real-time WebSocket feeds : Week 12+
```

### High-Value Feature Ideas

| Feature | Data Source | Difficulty | Value |
|---------|-------------|------------|-------|
| **Dark Pool Activity** | FINRA ADF/TRF | Medium | Institutional flow |
| **Options Flow** | CBOE/OCC | High | Smart money tracking |
| **Insider Trades** | SEC Form 4 (already partial) | Low | Leadership confidence |
| **Earnings Whisper** | Estimize API | Medium | Crowd estimates |
| **Alternative Data** | Reddit sentiment, satellite imagery | High | Alpha generation |

---

## 4. Research Approach

### Systematic Research Framework

```mermaid
flowchart TB
    A[Identify Gap] --> B{Research Type}
    B -->|Data Source| C[API Docs + Rate Limits + Data Quality]
    B -->|Algorithm| D[Academic Papers + Backtesting]
    B -->|Infrastructure| E[Benchmarks + Cost Analysis]
    C --> F[Prototype + Validate]
    D --> F
    E --> F
    F --> G[Document in ADR]
```

### Research Documentation Template

Create Architecture Decision Records (ADRs) in `docs/decisions/`:

```markdown
# ADR-001: Replace Moneycontrol Scraper with Finnhub

## Status: Proposed

## Context
Moneycontrol scraper has 0/4 success rate in live audit.

## Decision
Replace with Finnhub News API (60 calls/min free tier).

## Consequences
- ✅ Stable JSON API with documented schema
- ✅ Includes sentiment scores
- ⚠️ Limited to Finnhub-supported symbols
```

---

## 5. Production Hardening

### Production Readiness Checklist

| Category | Requirement | Status |
|----------|-------------|--------|
| **Security** | Environment variable secrets | ✅ Done |
| **Security** | Input validation (Cypher injection) | ❌ Missing |
| **Reliability** | Retry with exponential backoff | ❌ Missing |
| **Reliability** | Circuit breakers | ❌ Missing |
| **Observability** | Structured logging (JSON) | ❌ Missing |
| **Observability** | Request tracing | ❌ Missing |
| **Performance** | Async I/O throughout | 🟡 Partial |
| **Performance** | Connection pooling | ❌ Missing |

### Priority Hardening Tasks

1. **Add Request Validation**
   ```python
   from pydantic import BaseModel, validator
   
   class SymbolRequest(BaseModel):
       symbol: str
       
       @validator('symbol')
       def validate_symbol(cls, v):
           if not re.match(r'^[A-Z0-9.]{1,10}$', v):
               raise ValueError('Invalid symbol format')
           return v
   ```

2. **Implement Circuit Breakers**
   ```python
   from circuitbreaker import circuit
   
   @circuit(failure_threshold=3, recovery_timeout=60)
   async def fetch_external_api(url):
       # ... 
   ```

3. **Structured Logging**
   ```python
   import structlog
   logger = structlog.get_logger()
   logger.info("fetch_complete", source="finnhub", articles=len(data))
   ```

---

## 6. Fixing Brittle Scrapers

> [!CAUTION]
> Web scraping is inherently fragile. **The solution is to move to stable APIs, not fix scrapers.**

### Replacement Strategy

| Current Scraper | Status | Replacement | Cost |
|-----------------|--------|-------------|------|
| `fetch_seekingalpha` | ❌ BROKEN | **Remove** (paywall, bot protection) | - |
| `fetch_marketwatch` | ❌ BROKEN | **Remove** (JavaScript rendering required) | - |
| `fetch_moneycontrol` | 🟡 Flaky | **Finnhub** for Indian market | Free |
| `fetch_indian_filings` | ❌ BROKEN | **BSE/NSE Official APIs** | Free |
| `fetch_google_rss_sync` | ✅ Working | **Keep** as fallback | Free |
| `fetch_edgar_filings` | ✅ Working | **Keep** (official SEC API) | Free |

### Recommended New Data Stack

#### Tier 1: Free & Stable APIs

| Provider | Data Type | Free Tier Limits | Integration Effort |
|----------|-----------|------------------|-------------------|
| **Finnhub** | News, Sentiment, Fundamentals | 60 calls/min | Low (REST JSON) |
| **SEC EDGAR** | Filings (10-K, 8-K, Form 4) | 10 req/sec | Already done |
| **Yahoo Finance** | Prices, News | Unofficial but stable | Already done |
| **Google RSS** | News Headlines | Unlimited | Already done |
| **FRED** | Macro data | 100 calls/day | Low |

#### Tier 2: OpenBB (Unified SDK)

OpenBB provides a **single unified Python API** to access 100+ data sources:

```python
from openbb import obb

# One line to fetch from multiple sources
news = obb.news.world(query="AAPL")
fundamentals = obb.equity.fundamental.overview("AAPL")
```

**Pros:** Single SDK, standardized output, actively maintained  
**Cons:** Some providers require keys, learning curve

### Implementation Plan for fetchers.py

```mermaid
flowchart LR
    A[Current: 11 sources] --> B[Phase 1: Remove broken scrapers]
    B --> C[Phase 2: Add Finnhub adapter]
    C --> D[Phase 3: Integrate OpenBB SDK]
    D --> E[Final: 6-8 stable sources]
```

#### New `fetchers.py` Structure

```python
# fetchers.py - v18 (API-First Architecture)

class DataSourceManager:
    """Manages multiple data sources with failover"""
    
    def __init__(self):
        self.sources = {
            'primary': [FinnhubAdapter(), SECEdgarAdapter()],
            'secondary': [YahooAdapter(), GoogleRSSAdapter()],
            'fallback': [OpenBBAdapter()]
        }
    
    async def fetch_all(self, symbol: str, cutoff: datetime) -> List[Article]:
        """Fetch with automatic failover"""
        results = []
        for tier in ['primary', 'secondary', 'fallback']:
            for source in self.sources[tier]:
                try:
                    data = await source.fetch(symbol, cutoff)
                    results.extend(data)
                except SourceUnavailableError:
                    logger.warning(f"{source.name} unavailable, trying next")
                    continue
        return deduplicate(results)
```

---

## 7. SQLite → Production Database Migration

### Free Database Options Comparison

| Provider | Type | Free Tier | Best For |
|----------|------|-----------|----------|
| **Neon** | Serverless PostgreSQL | 0.5 GB, scale-to-zero | ⭐ Recommended |
| **Supabase** | PostgreSQL + BaaS | 500 MB, 50K MAU | Full-stack |
| **Aiven** | Managed PostgreSQL | 5 GB (hobbyist) | Simple migration |
| **CockroachDB Serverless** | Distributed SQL | 5 GB, 250M RUs | High availability |
| **PlanetScale** | MySQL (Vitess) | 1 billion rows | MySQL preference |

> [!TIP]
> **Recommended: Neon** — True serverless (pay nothing when idle), branching for dev/prod, PostgreSQL-compatible.

### Migration Path

```mermaid
flowchart LR
    A[SQLite] --> B[Export schema + data]
    B --> C[Create Neon project]
    C --> D[Apply schema via Alembic]
    D --> E[Migrate data]
    E --> F[Update connection strings]
    F --> G[Test + Deploy]
```

#### Step-by-Step Migration

1. **Create Neon Project** (free): https://console.neon.tech
2. **Install Dependencies**
   ```bash
   pip install psycopg2-binary sqlalchemy[asyncio] alembic
   ```
3. **Update Connection String**
   ```python
   # .env
   DATABASE_URL=postgresql://user:pass@ep-cool-name-123456.us-east-2.aws.neon.tech/neondb?sslmode=require
   ```
4. **Use SQLAlchemy for Portability**
   ```python
   from sqlalchemy import create_engine
   engine = create_engine(os.environ["DATABASE_URL"])
   ```

---

## 8. CI/CD Pipeline Implementation

### GitHub Actions Strategy

| Workflow | Trigger | Purpose |
|----------|---------|---------|
| `ci.yml` | Push/PR | Lint, test, type-check |
| `scraper-health.yml` | Cron (daily) | Live scraper validation |
| `deploy.yml` | Push to main | Deploy to hosting |

### Sample CI Workflow

```yaml
# .github/workflows/ci.yml
name: CI Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      - run: pip install ruff mypy
      - run: ruff check .
      - run: mypy . --ignore-missing-imports

  test:
    runs-on: ubuntu-latest
    needs: lint
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      - run: pip install -r requirements.txt
      - run: pytest tests/unit tests/integration --cov=.

  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pip install bandit safety
      - run: bandit -r . -x ./tests
      - run: safety check
```

### Scraper Health Check (Daily Cron)

```yaml
# .github/workflows/scraper-health.yml
name: Scraper Health Check

on:
  schedule:
    - cron: '0 6 * * *'  # 6 AM UTC daily

jobs:
  health-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
      - run: pip install -r requirements.txt
      - run: pytest tests/live/test_scraper_health.py -v
      - name: Alert on Failure
        if: failure()
        run: |
          # Send Slack/Discord/Email notification
          curl -X POST ${{ secrets.ALERT_WEBHOOK }} -d '{"text":"Scraper health check failed!"}'
```

---

## 9. Free Hosting Options

### Hosting Comparison

| Platform | Best For | Free Tier | Limitations |
|----------|----------|-----------|-------------|
| **Streamlit Cloud** | Streamlit apps | ✅ Unlimited public | 1 GB RAM, public only |
| **Hugging Face Spaces** | Gradio apps | ✅ Unlimited | 16 GB RAM (CPU) |
| **Render** | Full apps | 750 hrs/mo | Sleeps after 15 min |
| **Railway** | Full stack | $5 credit | Expires in 30 days |
| **Fly.io** | Containers | 3 VMs free | Pay-as-you-go now |
| **Vercel** | Frontend + API | 100 GB bandwidth | Serverless only |

### Recommended Setup

```mermaid
flowchart TB
    subgraph "Free Hosting Stack"
        A[Streamlit Cloud] -->|Streamlit UI| B[Neon PostgreSQL]
        C[Hugging Face Spaces] -->|Gradio UI| B
        D[Neo4j Aura Free] -->|Knowledge Graph| E[User]
    end
    A --> E
    C --> E
```

| Component | Host | URL Pattern |
|-----------|------|-------------|
| Streamlit UI | Streamlit Cloud | `your-app.streamlit.app` |
| Gradio UI | Hugging Face Spaces | `huggingface.co/spaces/you/finwise` |
| PostgreSQL | Neon | Connection string in secrets |
| Knowledge Graph | Neo4j Aura | Already using |

---

## 10. Monetization Strategy

### Short Answer: **Yes, monetizable — with significant work.**

### Market Opportunity

> FinTech market: $226B (2023) → $917B (2032), 16.8% CAGR  
> Mobile finance apps: $1.87B revenue (2024)

### Viable Business Models for FinWise

| Model | Description | Potential Revenue |
|-------|-------------|-------------------|
| **Freemium SaaS** | Free basic analysis, paid premium features | $10-50/user/month |
| **Tiered API** | Sell API access by volume | $100-500/month |
| **Data Licensing** | Sell aggregated sentiment data | $1K-10K/month |
| **White-Label** | License to financial advisors/apps | $5K-50K/year |
| **Affiliate** | Partner with brokers (commission) | Variable |

### Recommended Monetization Path

```mermaid
flowchart LR
    A[Phase 1: Free Public App] --> B[Build user base]
    B --> C[Phase 2: Freemium]
    C --> D[Premium features:\n- API access\n- Historical data\n- Custom alerts]
    D --> E[Phase 3: B2B]
    E --> F[Enterprise licensing\nData syndication]
```

### Freemium Feature Matrix

| Feature | Free Tier | Pro ($19/mo) | Enterprise |
|---------|-----------|--------------|------------|
| Symbols tracked | 3 | 50 | Unlimited |
| Sentiment analysis | Daily | Real-time | Real-time + API |
| Monte Carlo | 500 paths | 5,000 paths | Custom |
| Historical data | 7 days | 1 year | 5+ years |
| Export | ❌ | CSV | JSON/API |
| Custom alerts | ❌ | Email | Webhook |

### Legal Considerations

> [!WARNING]
> **Critical before monetization:**
> - Financial data redistribution licenses (Yahoo Finance allows non-commercial only)
> - "Not investment advice" disclaimers
> - GDPR/CCPA compliance if collecting user data
> - Consider SEC regulations if providing recommendations

---

## Next Steps: Prioritized Action Items

### Week 1-2: Critical Fixes
- [ ] Replace `_YF_CACHE` with TTL cache
- [ ] Persist conversation state to SQLite/Redis
- [ ] Remove broken scrapers (SeekingAlpha, MarketWatch)
- [ ] Add Finnhub adapter

### Week 3-4: Infrastructure
- [ ] Set up Neon PostgreSQL for production
- [ ] Implement GitHub Actions CI pipeline
- [ ] Deploy to Streamlit Cloud + Hugging Face

### Week 5-8: Enhancement
- [ ] Integrate OpenBB SDK
- [ ] Add scraper health monitoring
- [ ] Calculate Heston params from historical data

### Week 9-12: Production Launch
- [ ] Implement tiered pricing (Stripe integration)
- [ ] Add user authentication
- [ ] Launch MVP to public

---

## Resources

### Documentation Links
- [Finnhub API Docs](https://finnhub.io/docs/api)
- [OpenBB Documentation](https://docs.openbb.co)
- [Neon PostgreSQL](https://neon.tech/docs)
- [Streamlit Cloud](https://streamlit.io/cloud)
- [GitHub Actions Python](https://docs.github.com/en/actions/automating-builds-and-tests/building-and-testing-python)

### Relevant Audits
- [SCRAPER_LIVE_AUDIT.md](file:///h:/finwise_core/audits/SCRAPER_LIVE_AUDIT.md)
- [SYSTEM_FORENSIC_AUDIT.md](file:///h:/finwise_core/audits/SYSTEM_FORENSIC_AUDIT.md)

---

*Document generated: 2026-01-17*  
*Version: 1.0*
