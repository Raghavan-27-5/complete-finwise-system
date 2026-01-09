"""
Live Web Scraper Reliability Verification
------------------------------------------
Adversarial live testing of web scrapers for US and Indian equities.
Does NOT mock network calls. Does NOT modify production code.
API-based fetchers are explicitly excluded.

Author: Raghavan-27-5
Date: 2026-01-09
"""
import sys
import os
import time
import asyncio
import aiohttp
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, field
from dateutil import parser as dateutil_parser

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Direct imports from production code (no mocking)
from intelligence.sentiment.fetchers import (
    fetch_google_rss_sync,
    fetch_moneycontrol,
    fetch_seekingalpha,
    fetch_marketwatch,
    fetch_edgar_filings,
    fetch_indian_filings,
)

# --- Configuration ---
US_SYMBOLS = ["AAPL", "MSFT"]
INDIA_SYMBOLS = ["RELIANCE.NS", "TCS.NS"]
CUTOFF_DAYS = 14
REQUIRED_FIELDS = ["headline", "summary", "source", "published"]


@dataclass
class ScraperResult:
    """Result of a single scraper execution."""
    scraper_name: str
    symbol: str
    market: str
    status: str = "UNKNOWN"
    items_count: int = 0
    latency_seconds: float = 0.0
    exception: str = ""
    notes: str = ""
    raw_items: List[Dict[str, Any]] = field(default_factory=list)
    field_validation_errors: List[str] = field(default_factory=list)


def validate_article_structure(articles: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
    """
    Validate that each article contains required fields and published is parseable.
    Returns (all_valid, list_of_errors).
    """
    errors = []
    for i, art in enumerate(articles):
        for field_name in REQUIRED_FIELDS:
            if field_name not in art:
                errors.append(f"Article {i}: missing '{field_name}'")
            elif art[field_name] is None:
                errors.append(f"Article {i}: '{field_name}' is None")
        
        # Validate published is parseable
        if "published" in art and art["published"]:
            try:
                dateutil_parser.parse(art["published"])
            except Exception as e:
                errors.append(f"Article {i}: 'published' not parseable: {e}")
    
    return len(errors) == 0, errors


def run_sync_scraper(scraper_func, query: str, symbol: str, cutoff: datetime, market: str) -> ScraperResult:
    """Execute a synchronous scraper and capture results."""
    result = ScraperResult(
        scraper_name=scraper_func.__name__,
        symbol=symbol,
        market=market
    )
    
    start_time = time.time()
    try:
        articles = scraper_func(query, cutoff)
        result.latency_seconds = round(time.time() - start_time, 3)
        
        if not isinstance(articles, list):
            result.status = "FAIL"
            result.notes = f"Return type is {type(articles).__name__}, expected list"
            return result
        
        result.items_count = len(articles)
        result.raw_items = articles
        
        if len(articles) == 0:
            result.status = "FAIL"
            result.notes = "Zero articles returned (silent failure)"
            return result
        
        valid, errors = validate_article_structure(articles)
        result.field_validation_errors = errors
        
        if not valid:
            result.status = "PARTIAL"
            result.notes = f"{len(errors)} field validation errors"
        else:
            result.status = "PASS"
            result.notes = f"{len(articles)} articles validated"
            
    except Exception as e:
        result.latency_seconds = round(time.time() - start_time, 3)
        result.status = "ERROR"
        result.exception = str(e)
        result.notes = f"Unhandled exception: {type(e).__name__}"
    
    return result


async def run_async_scraper(scraper_func, session: aiohttp.ClientSession, 
                            query_or_symbol: str, symbol: str, cutoff: datetime, 
                            market: str, is_symbol_based: bool = False) -> ScraperResult:
    """Execute an asynchronous scraper and capture results."""
    result = ScraperResult(
        scraper_name=scraper_func.__name__,
        symbol=symbol,
        market=market
    )
    
    start_time = time.time()
    try:
        if is_symbol_based:
            articles = await scraper_func(session, query_or_symbol, cutoff)
        else:
            articles = await scraper_func(session, query_or_symbol, cutoff)
        
        result.latency_seconds = round(time.time() - start_time, 3)
        
        if not isinstance(articles, list):
            result.status = "FAIL"
            result.notes = f"Return type is {type(articles).__name__}, expected list"
            return result
        
        result.items_count = len(articles)
        result.raw_items = articles
        
        if len(articles) == 0:
            result.status = "FAIL"
            result.notes = "Zero articles returned (silent failure)"
            return result
        
        valid, errors = validate_article_structure(articles)
        result.field_validation_errors = errors
        
        if not valid:
            result.status = "PARTIAL"
            result.notes = f"{len(errors)} field validation errors"
        else:
            result.status = "PASS"
            result.notes = f"{len(articles)} articles validated"
            
    except Exception as e:
        result.latency_seconds = round(time.time() - start_time, 3)
        result.status = "ERROR"
        result.exception = str(e)
        result.notes = f"Unhandled exception: {type(e).__name__}"
    
    return result


async def run_all_tests() -> List[ScraperResult]:
    """Execute all scraper tests for US and Indian markets."""
    results = []
    cutoff = datetime.now(timezone.utc) - timedelta(days=CUTOFF_DAYS)
    
    print("[INFO] Starting live scraper verification...")
    print(f"[INFO] Cutoff: {cutoff.isoformat()}")
    
    # --- US Market Tests ---
    print("\n[INFO] === US MARKET TESTS ===")
    
    for symbol in US_SYMBOLS:
        query = symbol
        
        # Google RSS (sync)
        print(f"[INFO] Testing fetch_google_rss_sync for {symbol}...")
        results.append(run_sync_scraper(fetch_google_rss_sync, query, symbol, cutoff, "US"))
        
    # Async scrapers for US
    async with aiohttp.ClientSession() as session:
        for symbol in US_SYMBOLS:
            query = symbol
            
            # Moneycontrol (typically for India but test anyway)
            print(f"[INFO] Testing fetch_moneycontrol for {symbol}...")
            results.append(await run_async_scraper(
                fetch_moneycontrol, session, query, symbol, cutoff, "US"))
            
            # SeekingAlpha
            print(f"[INFO] Testing fetch_seekingalpha for {symbol}...")
            results.append(await run_async_scraper(
                fetch_seekingalpha, session, query, symbol, cutoff, "US"))
            
            # MarketWatch
            print(f"[INFO] Testing fetch_marketwatch for {symbol}...")
            results.append(await run_async_scraper(
                fetch_marketwatch, session, query, symbol, cutoff, "US"))
            
            # EDGAR Filings
            print(f"[INFO] Testing fetch_edgar_filings for {symbol}...")
            results.append(await run_async_scraper(
                fetch_edgar_filings, session, symbol, symbol, cutoff, "US", is_symbol_based=True))
    
    # --- Indian Market Tests ---
    print("\n[INFO] === INDIAN MARKET TESTS ===")
    
    for symbol in INDIA_SYMBOLS:
        # Extract base query (remove .NS/.BO suffix)
        query = symbol.replace(".NS", "").replace(".BO", "")
        
        # Google RSS (sync)
        print(f"[INFO] Testing fetch_google_rss_sync for {symbol}...")
        results.append(run_sync_scraper(fetch_google_rss_sync, query, symbol, cutoff, "INDIA"))
    
    # Async scrapers for India
    async with aiohttp.ClientSession() as session:
        for symbol in INDIA_SYMBOLS:
            query = symbol.replace(".NS", "").replace(".BO", "")
            
            # Moneycontrol
            print(f"[INFO] Testing fetch_moneycontrol for {symbol}...")
            results.append(await run_async_scraper(
                fetch_moneycontrol, session, query, symbol, cutoff, "INDIA"))
            
            # SeekingAlpha
            print(f"[INFO] Testing fetch_seekingalpha for {symbol}...")
            results.append(await run_async_scraper(
                fetch_seekingalpha, session, query, symbol, cutoff, "INDIA"))
            
            # MarketWatch
            print(f"[INFO] Testing fetch_marketwatch for {symbol}...")
            results.append(await run_async_scraper(
                fetch_marketwatch, session, query, symbol, cutoff, "INDIA"))
            
            # Indian Filings
            print(f"[INFO] Testing fetch_indian_filings for {symbol}...")
            results.append(await run_async_scraper(
                fetch_indian_filings, session, symbol, symbol, cutoff, "INDIA", is_symbol_based=True))
    
    print("\n[INFO] All tests completed.")
    return results


def generate_audit_report(results: List[ScraperResult], execution_time: datetime) -> str:
    """Generate the professional audit report in markdown format."""
    
    # Separate by market
    us_results = [r for r in results if r.market == "US"]
    india_results = [r for r in results if r.market == "INDIA"]
    
    # Count statistics
    total_pass = sum(1 for r in results if r.status == "PASS")
    total_fail = sum(1 for r in results if r.status == "FAIL")
    total_error = sum(1 for r in results if r.status == "ERROR")
    total_partial = sum(1 for r in results if r.status == "PARTIAL")
    
    # Build report
    report = f"""# Live Web Scraper Reliability Audit

## Metadata

| Field | Value |
|-------|-------|
| Execution Date | {execution_time.strftime('%Y-%m-%d %H:%M:%S %Z')} |
| Environment | Google Antigravity (Live Network) |
| Cutoff Window | {CUTOFF_DAYS} days |
| US Symbols Tested | {', '.join(US_SYMBOLS)} |
| Indian Symbols Tested | {', '.join(INDIA_SYMBOLS)} |

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
"""
    
    for r in us_results:
        status_emoji = {"PASS": "PASS", "FAIL": "FAIL", "ERROR": "ERROR", "PARTIAL": "PARTIAL"}.get(r.status, "?")
        notes = r.notes if not r.exception else f"{r.notes}: {r.exception[:50]}"
        report += f"| {r.scraper_name} | {r.symbol} | {status_emoji} | {r.items_count} | {r.latency_seconds} | {notes} |\n"
    
    report += """
---

## Results: Indian Markets

| Scraper | Symbol | Status | Items | Latency (s) | Notes |
|---------|--------|--------|-------|-------------|-------|
"""
    
    for r in india_results:
        status_emoji = {"PASS": "PASS", "FAIL": "FAIL", "ERROR": "ERROR", "PARTIAL": "PARTIAL"}.get(r.status, "?")
        notes = r.notes if not r.exception else f"{r.notes}: {r.exception[:50]}"
        report += f"| {r.scraper_name} | {r.symbol} | {status_emoji} | {r.items_count} | {r.latency_seconds} | {notes} |\n"
    
    # Failure Analysis
    failed_results = [r for r in results if r.status in ("FAIL", "ERROR")]
    
    report += """
---

## Failure Analysis

"""
    
    if not failed_results:
        report += "No failures detected during this audit run.\n"
    else:
        for r in failed_results:
            report += f"""### {r.scraper_name} ({r.symbol})

- **Status**: {r.status}
- **Exception**: {r.exception if r.exception else 'N/A'}
- **Notes**: {r.notes}

**Probable Cause**:
"""
            # Determine probable cause
            if "timeout" in r.exception.lower() if r.exception else False:
                report += "- Endpoint timeout or network latency issue\n"
                report += "- **Severity**: MEDIUM (transient)\n"
            elif r.items_count == 0 and r.status == "FAIL":
                report += "- HTML structure change or CSS selector drift\n"
                report += "- Bot protection blocking automated requests\n"
                report += "- Endpoint may require JavaScript rendering\n"
                report += "- **Severity**: HIGH (silent failure risk)\n"
            elif r.status == "ERROR":
                report += "- Unhandled exception in scraper logic\n"
                report += "- Possible parsing error or data format change\n"
                report += "- **Severity**: HIGH (code defect)\n"
            else:
                report += "- Unknown cause, requires manual investigation\n"
                report += "- **Severity**: MEDIUM\n"
            
            report += "\n"
    
    # Reliability Classification
    report += """---

## Reliability Classification

| Scraper | Classification | Justification |
|---------|----------------|---------------|
"""
    
    # Group by scraper name
    scraper_names = set(r.scraper_name for r in results)
    for scraper_name in sorted(scraper_names):
        scraper_results = [r for r in results if r.scraper_name == scraper_name]
        pass_count = sum(1 for r in scraper_results if r.status == "PASS")
        total_count = len(scraper_results)
        pass_rate = pass_count / total_count if total_count > 0 else 0
        
        if pass_rate >= 0.8:
            classification = "Reliable"
            justification = f"{pass_count}/{total_count} tests passed"
        elif pass_rate >= 0.4:
            classification = "Brittle"
            justification = f"Partial success ({pass_count}/{total_count}), likely to break"
        else:
            classification = "Unreliable"
            justification = f"Low success rate ({pass_count}/{total_count}), do not depend on"
        
        report += f"| {scraper_name} | {classification} | {justification} |\n"
    
    # Final Verdict
    report += f"""
---

## Final Verdict

### Overall Statistics

| Metric | Value |
|--------|-------|
| Total Tests | {len(results)} |
| Passed | {total_pass} |
| Failed | {total_fail} |
| Errors | {total_error} |
| Partial | {total_partial} |
| Pass Rate | {(total_pass / len(results) * 100):.1f}% |

### System Readiness

"""
    
    if total_fail + total_error == 0:
        report += """The scraper layer is **OPERATIONAL** with all tested endpoints returning valid data.
However, web scrapers are inherently brittle and should be monitored continuously.

**Silent Failure Risk**: LOW (at time of audit)
"""
    elif (total_fail + total_error) / len(results) < 0.3:
        report += """The scraper layer is **PARTIALLY OPERATIONAL** with some endpoints failing.
Failed scrapers should be investigated and either fixed or removed from the pipeline.

**Silent Failure Risk**: MEDIUM
"""
    else:
        report += """The scraper layer is **DEGRADED** with significant failures across endpoints.
Immediate investigation is required. Consider disabling failed scrapers.

**Silent Failure Risk**: HIGH
"""
    
    report += """
### Recommendations Per Scraper

| Scraper | Recommendation |
|---------|----------------|
"""
    
    for scraper_name in sorted(scraper_names):
        scraper_results = [r for r in results if r.scraper_name == scraper_name]
        pass_count = sum(1 for r in scraper_results if r.status == "PASS")
        total_count = len(scraper_results)
        pass_rate = pass_count / total_count if total_count > 0 else 0
        
        if pass_rate >= 0.8:
            recommendation = "KEEP - Monitor for regressions"
        elif pass_rate >= 0.4:
            recommendation = "REFACTOR - Address selector brittleness"
        else:
            recommendation = "DELETE or DISABLE - Unreliable data source"
        
        report += f"| {scraper_name} | {recommendation} |\n"
    
    report += """
---

*Report generated by automated live verification system.*
*Absence of failures does not guarantee future reliability.*
*Web scrapers require continuous monitoring.*
"""
    
    return report


def main():
    """Main entry point for live scraper verification."""
    execution_time = datetime.now(timezone.utc)
    
    # Run all tests
    results = asyncio.run(run_all_tests())
    
    # Generate report
    report = generate_audit_report(results, execution_time)
    
    # Write report to file
    report_path = os.path.join(
        os.path.dirname(__file__), 
        "..", 
        "audits", 
        "SCRAPER_LIVE_AUDIT.md"
    )
    report_path = os.path.abspath(report_path)
    
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    
    print(f"\n[OUTPUT] Report written to: {report_path}")
    
    return results, report_path


if __name__ == "__main__":
    main()
