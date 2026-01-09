# Live Web Scraper Reliability Audit

## Metadata

| Field | Value |
|-------|-------|
| Execution Date | 2026-01-09 05:41:52 UTC |
| Environment | Google Antigravity (Live Network) |
| Cutoff Window | 14 days |
| US Symbols Tested | AAPL, MSFT |
| Indian Symbols Tested | RELIANCE.NS, TCS.NS |

## Explicit Exclusions

The following data sources were explicitly excluded from this audit as they are API-based fetchers requiring authentication credentials:

- fetch_newsapi (requires NEWS_API_KEY)
- fetch_alpha_transcripts (requires ALPHA_VANTAGE_KEY)
- fetch_reddit_posts (requires Reddit OAuth credentials)
- fetch_x_posts (requires X_BEARER_TOKEN)
- fetch_yahoo_sync (uses yfinance library, not a web scraper)
- fetch_yahoo_earnings_sync (uses yfinance library, not a web scraper)

---

## Scope and Methodology

### What Was Tested

Six web scrapers were tested against live endpoints with real network calls:

1. **fetch_google_rss_sync** - Google News RSS feed parser
2. **fetch_moneycontrol** - Moneycontrol.com HTML scraper
3. **fetch_seekingalpha** - SeekingAlpha.com HTML scraper
4. **fetch_marketwatch** - MarketWatch.com HTML scraper
5. **fetch_edgar_filings** - SEC EDGAR JSON API (US only)
6. **fetch_indian_filings** - NSE/BSE filing endpoints (India only)

### What Was NOT Tested

- API-based fetchers (see exclusions above)
- Aggregation functions (aggregate_sources)
- Rate limiting behavior under sustained load
- Concurrent execution at scale

### Why Live Execution Was Required

Mocked tests cannot detect:
- Silent failures (scraper returns empty but does not raise)
- HTML/CSS selector drift
- Endpoint deprecation
- Bot protection mechanisms
- Network-level blocking

### Failure Definition

A scraper is considered **FAILED** if any of the following are true:
- Unhandled exception during execution
- Return type is not a list
- Return list has zero items
- Any item is missing required fields (headline, summary, source, published)
- Published field is not parseable as datetime

---

## Results: US Markets

| Scraper | Symbol | Status | Items | Latency (s) | Notes |
|---------|--------|--------|-------|-------------|-------|
| fetch_google_rss_sync | AAPL | PASS | 85 | 1.914 | 85 articles validated |
| fetch_google_rss_sync | MSFT | PASS | 81 | 2.678 | 81 articles validated |
| fetch_moneycontrol | AAPL | PASS | 8 | 1.046 | 8 articles validated |
| fetch_seekingalpha | AAPL | FAIL | 0 | 0.701 | Zero articles returned (silent failure) |
| fetch_marketwatch | AAPL | FAIL | 0 | 0.217 | Zero articles returned (silent failure) |
| fetch_edgar_filings | AAPL | PASS | 1 | 2.561 | 1 articles validated |
| fetch_moneycontrol | MSFT | FAIL | 0 | 0.849 | Zero articles returned (silent failure) |
| fetch_seekingalpha | MSFT | FAIL | 0 | 0.66 | Zero articles returned (silent failure) |
| fetch_marketwatch | MSFT | FAIL | 0 | 0.166 | Zero articles returned (silent failure) |
| fetch_edgar_filings | MSFT | FAIL | 0 | 0.715 | Zero articles returned (silent failure) |

---

## Results: Indian Markets

| Scraper | Symbol | Status | Items | Latency (s) | Notes |
|---------|--------|--------|-------|-------------|-------|
| fetch_google_rss_sync | RELIANCE.NS | PASS | 87 | 2.08 | 87 articles validated |
| fetch_google_rss_sync | TCS.NS | PASS | 47 | 1.56 | 47 articles validated |
| fetch_moneycontrol | RELIANCE.NS | FAIL | 0 | 2.433 | Zero articles returned (silent failure) |
| fetch_seekingalpha | RELIANCE.NS | FAIL | 0 | 0.506 | Zero articles returned (silent failure) |
| fetch_marketwatch | RELIANCE.NS | FAIL | 0 | 0.28 | Zero articles returned (silent failure) |
| fetch_indian_filings | RELIANCE.NS | FAIL | 0 | 0.267 | Zero articles returned (silent failure) |
| fetch_moneycontrol | TCS.NS | FAIL | 0 | 1.353 | Zero articles returned (silent failure) |
| fetch_seekingalpha | TCS.NS | FAIL | 0 | 0.429 | Zero articles returned (silent failure) |
| fetch_marketwatch | TCS.NS | FAIL | 0 | 0.159 | Zero articles returned (silent failure) |
| fetch_indian_filings | TCS.NS | FAIL | 0 | 0.123 | Zero articles returned (silent failure) |

---

## Failure Analysis

### fetch_seekingalpha (AAPL)

- **Status**: FAIL
- **Exception**: N/A
- **Notes**: Zero articles returned (silent failure)

**Probable Cause**:
- HTML structure change or CSS selector drift
- Bot protection blocking automated requests
- Endpoint may require JavaScript rendering
- **Severity**: HIGH (silent failure risk)

### fetch_marketwatch (AAPL)

- **Status**: FAIL
- **Exception**: N/A
- **Notes**: Zero articles returned (silent failure)

**Probable Cause**:
- HTML structure change or CSS selector drift
- Bot protection blocking automated requests
- Endpoint may require JavaScript rendering
- **Severity**: HIGH (silent failure risk)

### fetch_moneycontrol (MSFT)

- **Status**: FAIL
- **Exception**: N/A
- **Notes**: Zero articles returned (silent failure)

**Probable Cause**:
- HTML structure change or CSS selector drift
- Bot protection blocking automated requests
- Endpoint may require JavaScript rendering
- **Severity**: HIGH (silent failure risk)

### fetch_seekingalpha (MSFT)

- **Status**: FAIL
- **Exception**: N/A
- **Notes**: Zero articles returned (silent failure)

**Probable Cause**:
- HTML structure change or CSS selector drift
- Bot protection blocking automated requests
- Endpoint may require JavaScript rendering
- **Severity**: HIGH (silent failure risk)

### fetch_marketwatch (MSFT)

- **Status**: FAIL
- **Exception**: N/A
- **Notes**: Zero articles returned (silent failure)

**Probable Cause**:
- HTML structure change or CSS selector drift
- Bot protection blocking automated requests
- Endpoint may require JavaScript rendering
- **Severity**: HIGH (silent failure risk)

### fetch_edgar_filings (MSFT)

- **Status**: FAIL
- **Exception**: N/A
- **Notes**: Zero articles returned (silent failure)

**Probable Cause**:
- HTML structure change or CSS selector drift
- Bot protection blocking automated requests
- Endpoint may require JavaScript rendering
- **Severity**: HIGH (silent failure risk)

### fetch_moneycontrol (RELIANCE.NS)

- **Status**: FAIL
- **Exception**: N/A
- **Notes**: Zero articles returned (silent failure)

**Probable Cause**:
- HTML structure change or CSS selector drift
- Bot protection blocking automated requests
- Endpoint may require JavaScript rendering
- **Severity**: HIGH (silent failure risk)

### fetch_seekingalpha (RELIANCE.NS)

- **Status**: FAIL
- **Exception**: N/A
- **Notes**: Zero articles returned (silent failure)

**Probable Cause**:
- HTML structure change or CSS selector drift
- Bot protection blocking automated requests
- Endpoint may require JavaScript rendering
- **Severity**: HIGH (silent failure risk)

### fetch_marketwatch (RELIANCE.NS)

- **Status**: FAIL
- **Exception**: N/A
- **Notes**: Zero articles returned (silent failure)

**Probable Cause**:
- HTML structure change or CSS selector drift
- Bot protection blocking automated requests
- Endpoint may require JavaScript rendering
- **Severity**: HIGH (silent failure risk)

### fetch_indian_filings (RELIANCE.NS)

- **Status**: FAIL
- **Exception**: N/A
- **Notes**: Zero articles returned (silent failure)

**Probable Cause**:
- HTML structure change or CSS selector drift
- Bot protection blocking automated requests
- Endpoint may require JavaScript rendering
- **Severity**: HIGH (silent failure risk)

### fetch_moneycontrol (TCS.NS)

- **Status**: FAIL
- **Exception**: N/A
- **Notes**: Zero articles returned (silent failure)

**Probable Cause**:
- HTML structure change or CSS selector drift
- Bot protection blocking automated requests
- Endpoint may require JavaScript rendering
- **Severity**: HIGH (silent failure risk)

### fetch_seekingalpha (TCS.NS)

- **Status**: FAIL
- **Exception**: N/A
- **Notes**: Zero articles returned (silent failure)

**Probable Cause**:
- HTML structure change or CSS selector drift
- Bot protection blocking automated requests
- Endpoint may require JavaScript rendering
- **Severity**: HIGH (silent failure risk)

### fetch_marketwatch (TCS.NS)

- **Status**: FAIL
- **Exception**: N/A
- **Notes**: Zero articles returned (silent failure)

**Probable Cause**:
- HTML structure change or CSS selector drift
- Bot protection blocking automated requests
- Endpoint may require JavaScript rendering
- **Severity**: HIGH (silent failure risk)

### fetch_indian_filings (TCS.NS)

- **Status**: FAIL
- **Exception**: N/A
- **Notes**: Zero articles returned (silent failure)

**Probable Cause**:
- HTML structure change or CSS selector drift
- Bot protection blocking automated requests
- Endpoint may require JavaScript rendering
- **Severity**: HIGH (silent failure risk)

---

## Reliability Classification

| Scraper | Classification | Justification |
|---------|----------------|---------------|
| fetch_edgar_filings | Brittle | Partial success (1/2), likely to break |
| fetch_google_rss_sync | Reliable | 4/4 tests passed |
| fetch_indian_filings | Unreliable | Low success rate (0/2), do not depend on |
| fetch_marketwatch | Unreliable | Low success rate (0/4), do not depend on |
| fetch_moneycontrol | Unreliable | Low success rate (1/4), do not depend on |
| fetch_seekingalpha | Unreliable | Low success rate (0/4), do not depend on |

---

## Final Verdict

### Overall Statistics

| Metric | Value |
|--------|-------|
| Total Tests | 20 |
| Passed | 6 |
| Failed | 14 |
| Errors | 0 |
| Partial | 0 |
| Pass Rate | 30.0% |

### System Readiness

The scraper layer is **DEGRADED** with significant failures across endpoints.
Immediate investigation is required. Consider disabling failed scrapers.

**Silent Failure Risk**: HIGH

### Recommendations Per Scraper

| Scraper | Recommendation |
|---------|----------------|
| fetch_edgar_filings | REFACTOR - Address selector brittleness |
| fetch_google_rss_sync | KEEP - Monitor for regressions |
| fetch_indian_filings | DELETE or DISABLE - Unreliable data source |
| fetch_marketwatch | DELETE or DISABLE - Unreliable data source |
| fetch_moneycontrol | DELETE or DISABLE - Unreliable data source |
| fetch_seekingalpha | DELETE or DISABLE - Unreliable data source |

---

*Report generated by automated live verification system.*
*Absence of failures does not guarantee future reliability.*
*Web scrapers require continuous monitoring.*
