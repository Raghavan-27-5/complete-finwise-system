# Web Scraper Reliability Restoration Plan

**Date**: 2026-01-09
**Author**: Raghavan-27-5
**Scope**: Fix failed scrapers and implement anti-brittleness measures.

---

## 1. Diagnosis & Specific Fixes

### A. MarketWatch (SSL Error)
- **Issue**: `SSLCertVerificationError`. Python's `aiohttp` is strict about certificate chains, which are often incomplete on legacy media sites or blocked by corporate firewalls.
- **Fix**: 
  - Disable SSL verification for scraping context (`ssl=False`).
  - This is acceptable for public data scraping (non-transactional).

### B. Moneycontrol (Zero Results)
- **Issue**: CSS Selectors are obsolete (`li.startup-videos-item`). The site layout has changed.
- **Fix**: 
  - Switch to the "Search" or "News" endpoint which is more stable.
  - Use **Loose Selectors**: Target semantic elements (e.g., `h2 > a`, `time`) rather than brittle class names (e.g., `.class_v2_red`).
  - **Fallback**: If Moneycontrol fails, scrape `Google RSS` with query `site:moneycontrol.com {symbol}`.

### C. SeekingAlpha (Bot Protection)
- **Issue**: Cloudflare/PerimeterX is detecting the `aiohttp` User-Agent and returning a challenge page (hence 0 articles parsed).
- **Fix**: 
  - **Header Impersonation**: Use a "Browser-Identical" header set (Chrome 120 on Windows).
  - **Method Change**: Switch to `fetch_google_rss_sync` with `site:seekingalpha.com`. This offloads the scraping difficulty to Google, which indexes SeekingAlpha effectively.

### D. NSE / BSE Filings (API Errors)
- **Issue**: API endpoints return 404 or require specific headers (Referer/Host).
- **Fix**: 
  - Update endpoints to verified 2026 APIs.
  - Enforce `Referer: https://www.nseindia.com/` in headers.
  - Implement a "Broad Search" fallback if the specific API fails.

---

## 2. Structural Improvements (Preventing Brittleness)

### Strategy 1: The "Google Shield" Fallback
Direct scraping is brittle. Google Indexing is robust.
- **Current**: Scraper tries `moneycontrol.com/search?q=...`
- **New Pattern**:
  1. Try Direct Scrape (High Fidelity).
  2. If Fail/Empty: Call `fetch_google_rss_sync` with `site:moneycontrol.com {query}`.
  This guarantees *some* coverage even if the target site changes its CSS.

### Strategy 2: Robust Header Rotation
- Implement a pool of 5-10 common User-Agents (Chrome, Firefox, Safari).
- Randomize per request to reduce "bot fingerprinting".

### Strategy 3: Loose Coupling Selectors
- Avoid: `div.content-wrapper > div.row > div.col-md-8 > h1`
- Prefer: `h1` (if unique) or `a[href*='/news/']`.
- Find items by *content pattern* (e.g., timestamp presence) rather than just CSS path.

---

## 3. Execution Roadmap

1. **Refactor `fetchers.py`**:
   - Add `get_random_header()` helper.
   - Implement SSL bypass context in `aiohttp`.
   - Update individual scraper logic (Moneycontrol, MarketWatch, NSE).
2. **Verify**:
   - Run `tests/test_scrapers_live.py` on `fix-scrapers-v1` branch.
3. **Merge**:
   - Merge to `main` only when Pass Rate > 80%.

---

## 4. Maintenance Protocol

- **Weekly**: Automated run of `test_scrapers_live.py`.
- **On Failure**:
  - If <80% pass: Alert Developer.
  - If specific scraper fails consistently: Disable temporarily via Config flag.

---
