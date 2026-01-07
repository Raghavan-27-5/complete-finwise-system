# fetchers.py - (v17.2, Final)

import asyncio
import aiohttp
from ratelimit import limits, sleep_and_retry
from datetime import datetime, timedelta, timezone
import feedparser
import yfinance as yf
import praw
import logging
import os
from bs4 import BeautifulSoup
from dateutil import parser as dateutil_parser
from urllib.parse import quote_plus
import xml.etree.ElementTree as ET
from io import BytesIO
import PyPDF2
import re
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from functools import lru_cache
from core.config import REDDIT_SUBREDDITS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- API Keys & Configuration ---
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
X_BEARER_TOKEN = os.getenv("X_BEARER_TOKEN")
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY")
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT")

MARKET_INDIA_SUFFIXES = [".NS", ".BO"]

# --- CIK Map Caching for EDGAR ---
@lru_cache(maxsize=1)
def get_cik_map() -> Dict[str, str]:
    """
    Downloads and caches the SEC's ticker-to-CIK mapping.
    This is a synchronous function intended to be run once.
    """
    import requests
    logger.info("Fetching and caching SEC CIK map for the first time...")
    url = "https://www.sec.gov/files/company_tickers.json"
    try:
        # SEC requires a real User-Agent with contact info for programmatic access
        response = requests.get(url, headers={'User-Agent': 'Raghavan r3398465@gmail.com'})
        response.raise_for_status()
        data = response.json()
        return {val['ticker']: str(val['cik_str']).zfill(10) for _, val in data.items()}
    except Exception as e:
        logger.error(f"Failed to fetch or process SEC CIK map: {e}")
        return {}

# --- General Helpers ---
def make_tz_aware(dt: datetime) -> datetime:
    """Make a datetime object timezone-aware (UTC) if it's naive."""
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt

def parse_relative_date(date_str: str) -> datetime:
    date_str = date_str.lower().strip()
    now = datetime.now(timezone.utc)
    try:
        if "ago" in date_str:
            num = int(re.search(r'\d+', date_str).group())
            if "hour" in date_str:
                return now - timedelta(hours=num)
            if "minute" in date_str:
                return now - timedelta(minutes=num)
            if "day" in date_str:
                return now - timedelta(days=num)
        parsed_date = dateutil_parser.parse(date_str)
        return make_tz_aware(parsed_date)
    except Exception:
        return now

def is_indian_market(symbol: str) -> bool:
    return any(symbol.upper().endswith(suffix.upper()) for suffix in MARKET_INDIA_SUFFIXES)

def extract_pdf_text(pdf_bytes: bytes, max_pages: int = 3) -> str:
    try:
        pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))
        text = "".join(page.extract_text() or "" for i, page in enumerate(pdf_reader.pages) if i < max_pages)
        return re.sub(r'\s+', ' ', text).strip()[:5000]
    except Exception as e:
        logger.warning(f"PDF extraction failed: {e}")
        return ""

# --- Parsing Function ---
_eps_regex = re.compile(r"reports EPS of \$?(-?\d+\.\d+)")
_est_regex = re.compile(r"estimate of \$?(-?\d+\.\d+)")
_rev_regex = re.compile(r"revenue of \$?([\d,]+\.?\d*)")

def parse_eps_revenue(summary_text: str) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    reported_eps = estimated_eps = revenue = surprise_pct = None
    try:
        eps_matches = _eps_regex.findall(summary_text)
        if eps_matches:
            try:
                reported_eps = float(eps_matches[0])
            except (ValueError, IndexError): pass

        est_matches = _est_regex.findall(summary_text)
        if est_matches:
            try:
                estimated_eps = float(est_matches[0])
            except (ValueError, IndexError): pass

        rev_matches = _rev_regex.findall(summary_text)
        if rev_matches:
            try:
                revenue = float(rev_matches[0].replace(",", ""))
            except (ValueError, IndexError): pass

        if reported_eps is not None and estimated_eps is not None and estimated_eps != 0:
            surprise_pct = ((reported_eps - estimated_eps) / abs(estimated_eps)) * 100
    except Exception as e:
        logger.warning(f"EPS/Revenue parsing failed: {e}")
    return reported_eps, estimated_eps, revenue, surprise_pct

# --- Data Fetcher Functions ---

# NewsAPI (async)
@sleep_and_retry
@limits(calls=90, period=86400)
async def fetch_newsapi(session: aiohttp.ClientSession, query: str, cutoff: datetime) -> list:
    if not NEWS_API_KEY:
        raise ValueError("NEWS_API_KEY is not configured.")
    url = f"https://newsapi.org/v2/everything?q={quote_plus(query)}&language=en&sortBy=publishedAt&apiKey={NEWS_API_KEY}"
    articles = []
    cutoff_aware = make_tz_aware(cutoff)
    try:
        async with session.get(url, timeout=10) as resp:
            data = await resp.json()
            for art in data.get("articles", []):
                pub_date_str = art.get("publishedAt")
                if not pub_date_str: continue
                pub_date = make_tz_aware(dateutil_parser.parse(pub_date_str))
                if pub_date >= cutoff_aware:
                    summary = art.get("description", "") or ""
                    reported_eps, estimated_eps, revenue, surprise_pct = parse_eps_revenue(summary)
                    articles.append({
                        "headline": art.get("title", ""), "summary": summary,
                        "source": art.get("source", {}).get("name", "NewsAPI"),
                        "published": pub_date.isoformat(), "engagement": 0,
                        "media_url": art.get("urlToImage", ""), "reported_eps": reported_eps,
                        "estimated_eps": estimated_eps, "revenue": revenue, "surprise_pct": surprise_pct
                    })
        return articles
    except Exception as e:
        logger.error(f"NewsAPI error: {e}")
        return []

# Google RSS (sync)
def fetch_google_rss_sync(query: str, cutoff: datetime) -> list:
    articles = []
    cutoff_aware = make_tz_aware(cutoff)
    try:
        feed = feedparser.parse(f"https://news.google.com/rss/search?q={quote_plus(query)}&hl=en-US&gl=US")
        for entry in feed.entries:
            pub_time = make_tz_aware(datetime(*entry.published_parsed[:6]))
            if pub_time >= cutoff_aware:
                summary = getattr(entry, "summary", "")
                reported_eps, estimated_eps, revenue, surprise_pct = parse_eps_revenue(summary)
                articles.append({
                    "headline": entry.title, "summary": summary,
                    "source": getattr(entry, "source", {}).get("title", "Google RSS"),
                    "published": pub_time.isoformat(), "engagement": 0, "media_url": "",
                    "reported_eps": reported_eps, "estimated_eps": estimated_eps,
                    "revenue": revenue, "surprise_pct": surprise_pct
                })
        return articles
    except Exception as e:
        logger.error(f"Google RSS error: {e}")
        return []

# Yahoo Finance news (sync)
def fetch_yahoo_sync(symbol: str, cutoff: datetime) -> list:
    articles = []
    cutoff_aware = make_tz_aware(cutoff)
    try:
        stock = yf.Ticker(symbol)
        for n in stock.news:
            pub_time_ts = n.get("providerPublishTime")
            if pub_time_ts and make_tz_aware(datetime.fromtimestamp(pub_time_ts)) >= cutoff_aware:
                pub_time = make_tz_aware(datetime.fromtimestamp(pub_time_ts))
                summary = n.get("summary", "")
                reported_eps, estimated_eps, revenue, surprise_pct = parse_eps_revenue(summary)
                articles.append({
                    "headline": n.get("title", ""), "summary": summary,
                    "source": n.get("publisher", "Yahoo"), "published": pub_time.isoformat(),
                    "engagement": 0, "media_url": n.get("thumbnail", {}).get("resolutions", [{}])[0].get("url", ""),
                    "reported_eps": reported_eps, "estimated_eps": estimated_eps,
                    "revenue": revenue, "surprise_pct": surprise_pct
                })
        return articles
    except Exception as e:
        logger.error(f"Yahoo Finance error: {e}")
        return []

# Yahoo Earnings (sync, structured)
def fetch_yahoo_earnings_sync(symbol: str, cutoff: datetime) -> list:
    articles = []
    cutoff_aware = make_tz_aware(cutoff)
    try:
        ticker = yf.Ticker(symbol)
        income_stmt = ticker.income_stmt
        if income_stmt.empty: return []
        earnings_dates = ticker.earnings_dates
        for col_date in income_stmt.columns[:4]:
            pub_date = make_tz_aware(pd.to_datetime(col_date).to_pydatetime())
            if pub_date >= cutoff_aware:
                net_income = income_stmt.loc['Net Income', col_date] if 'Net Income' in income_stmt.index else np.nan
                revenue = income_stmt.loc['Total Revenue', col_date] if 'Total Revenue' in income_stmt.index else np.nan
                reported_eps, estimated_eps = np.nan, np.nan
                if not earnings_dates.empty:
                    matching_earnings = earnings_dates[earnings_dates.index.date == col_date.date()]
                    if not matching_earnings.empty:
                        reported_eps = matching_earnings['Reported EPS'].iloc[0]
                        estimated_eps = matching_earnings['Estimate EPS'].iloc[0]
                surprise_pct = np.nan
                if pd.notna(reported_eps) and pd.notna(estimated_eps) and estimated_eps != 0:
                    surprise_pct = ((reported_eps - estimated_eps) / abs(estimated_eps)) * 100
                summary = f"Net Income: {net_income:,.0f}, Revenue: {revenue:,.0f}, Reported EPS: {reported_eps}"
                articles.append({
                    "headline": f"{symbol} Quarterly Financials {pub_date.strftime('%Y-%m-%d')}",
                    "summary": summary, "source": "Yahoo Earnings", "published": pub_date.isoformat(),
                    "engagement": 0, "media_url": "", "reported_eps": reported_eps,
                    "estimated_eps": estimated_eps, "revenue": revenue, "surprise_pct": surprise_pct
                })
        return articles
    except Exception as e:
        logger.error(f"Yahoo Earnings error for {symbol}: {e}")
        return []

# X Posts (async placeholder)
@sleep_and_retry
@limits(calls=5, period=86400)
async def fetch_x_posts(query: str, cutoff: datetime) -> list:
    if not X_BEARER_TOKEN: raise ValueError("X_BEARER_TOKEN is not configured.")
    return []

# Alpha Vantage (async)
@sleep_and_retry
@limits(calls=25, period=86400)
async def fetch_alpha_transcripts(session: aiohttp.ClientSession, symbol: str, cutoff: datetime) -> list:
    if not ALPHA_VANTAGE_KEY: raise ValueError("ALPHA_VANTAGE_KEY is not configured.")
    try:
        url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={symbol}&apikey={ALPHA_VANTAGE_KEY}"
        async with session.get(url, timeout=10) as resp:
            data = await resp.json()
            pub_date = datetime.now(timezone.utc)
            if make_tz_aware(cutoff) <= pub_date and data.get("Description"):
                summary = data.get("Description", "")
                reported_eps, estimated_eps, revenue, surprise_pct = parse_eps_revenue(summary)
                return [{"headline": f"{symbol} Company Overview", "summary": summary,
                         "source": "Alpha Vantage", "published": pub_date.isoformat(),
                         "engagement": 0, "media_url": "", "reported_eps": reported_eps,
                         "estimated_eps": estimated_eps, "revenue": revenue, "surprise_pct": surprise_pct}]
        return []
    except Exception as e:
        logger.error(f"Alpha Vantage error for {symbol}: {e}")
        return []

# Reddit (async)
async def fetch_reddit_posts(query: str, cutoff: datetime) -> list:
    if not all([REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT]):
        raise ValueError("Reddit API credentials are not fully configured.")
    cutoff_aware = make_tz_aware(cutoff)
    def reddit_fetch_sync():
        results = []
        try:
            reddit = praw.Reddit(client_id=REDDIT_CLIENT_ID, client_secret=REDDIT_CLIENT_SECRET,
                                 user_agent=REDDIT_USER_AGENT, check_for_async=False)
            for sub in REDDIT_SUBREDDITS:
                try:
                    for s in reddit.subreddit(sub).search(query, limit=12, sort="new"):
                        pub_time = make_tz_aware(datetime.fromtimestamp(s.created_utc))
                        if pub_time >= cutoff_aware:
                            media_url = s.url if s.url and any(ext in s.url.lower() for ext in ('.jpg','.png','.jpeg')) else ""
                            summary = s.selftext
                            reported_eps, estimated_eps, revenue, surprise_pct = parse_eps_revenue(summary)
                            results.append({
                                "headline": s.title, "summary": summary, "source": "Reddit",
                                "published": pub_time.isoformat(), "engagement": s.score,
                                "media_url": media_url, "reported_eps": reported_eps,
                                "estimated_eps": estimated_eps, "revenue": revenue, "surprise_pct": surprise_pct
                            })
                except Exception as e: logger.warning(f"Reddit sub {sub} error: {e}")
        except Exception as e: logger.error(f"Reddit top-level error: {e}")
        return results
    return await asyncio.to_thread(reddit_fetch_sync)
# Moneycontrol (async)
async def fetch_moneycontrol(session: aiohttp.ClientSession, query: str, cutoff: datetime) -> list:
    articles = []
    try:
        url = f"https://www.moneycontrol.com/news/tags/{quote_plus(query)}.html"
        async with session.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10) as resp:
            html = await resp.text()
            soup = BeautifulSoup(html, 'html.parser')
            items = soup.select('li.startup-videos-item')
            for item in items:
                try:
                    date_node = item.select_one('time.fleft')
                    date_str = date_node.text.strip() if date_node else ""
                    pub_date = parse_relative_date(date_str)
                    if pub_date >= cutoff:
                        headline_node = item.select_one('a')
                        headline_text = headline_node.text.strip() if headline_node else ""
                        summary = headline_text
                        reported_eps, estimated_eps, revenue, surprise_pct = parse_eps_revenue(summary)
                        articles.append({
                            "headline": headline_text, "summary": summary, "source": "Moneycontrol",
                            "published": pub_date.isoformat(), "engagement": 0, "media_url": "",
                            "reported_eps": reported_eps, "estimated_eps": estimated_eps,
                            "revenue": revenue, "surprise_pct": surprise_pct
                        })
                except Exception:
                    continue
        if not articles:
            logger.warning(f"[MONITOR] Moneycontrol scraper returned 0 articles for query '{query}'.")
        return articles[:20]
    except Exception as e:
        logger.error(f"Moneycontrol error: {e}")
        return []

# SeekingAlpha (async)
async def fetch_seekingalpha(session: aiohttp.ClientSession, query: str, cutoff: datetime) -> list:
    articles = []
    url = f"https://seekingalpha.com/search?q={quote_plus(query)}&type=news"
    try:
        async with session.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=15) as resp:
            html = await resp.text()
            soup = BeautifulSoup(html, 'html.parser')
            items = soup.select('li[class*="search-result"]')
            for item in items:
                try:
                    date_node = item.select_one('span[class*="search-result-date"]')
                    date_str = date_node.text.strip() if date_node else ""
                    pub_date = datetime.strptime(date_str, '%b. %d, %Y') if date_str else datetime.utcnow()
                    if pub_date >= cutoff:
                        headline_node = item.select_one('a')
                        summary_node = item.select_one('p')
                        summary = summary_node.text.strip() if summary_node else ""
                        reported_eps, estimated_eps, revenue, surprise_pct = parse_eps_revenue(summary)
                        articles.append({
                            "headline": headline_node.text.strip() if headline_node else "",
                            "summary": summary, "source": "SeekingAlpha", "published": pub_date.isoformat(),
                            "engagement": 0, "media_url": "", "reported_eps": reported_eps,
                            "estimated_eps": estimated_eps, "revenue": revenue, "surprise_pct": surprise_pct
                        })
                except Exception:
                    continue
        return articles
    except Exception as e:
        logger.error(f"SeekingAlpha error: {e}")
        return []

# MarketWatch (async)
async def fetch_marketwatch(session: aiohttp.ClientSession, query: str, cutoff: datetime) -> list:
    articles = []
    url = f"https://www.marketwatch.com/search?q={quote_plus(query)}"
    try:
        async with session.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=15) as resp:
            html = await resp.text()
            soup = BeautifulSoup(html, 'html.parser')
            items = soup.select('div.searchresult')
            for item in items:
                try:
                    date_node = item.select_one('span.timestamp')
                    date_str = date_node.text.strip() if date_node else ""
                    pub_date = parse_relative_date(date_str)
                    if pub_date >= cutoff:
                        headline_node = item.select_one('h3 a')
                        summary_node = item.select_one('p.article-summary')
                        summary = summary_node.text.strip() if summary_node else headline_node.text.strip() if headline_node else ""
                        reported_eps, estimated_eps, revenue, surprise_pct = parse_eps_revenue(summary)
                        articles.append({
                            "headline": headline_node.text.strip() if headline_node else "",
                            "summary": summary, "source": "MarketWatch", "published": pub_date.isoformat(),
                            "engagement": 0, "media_url": "", "reported_eps": reported_eps,
                            "estimated_eps": estimated_eps, "revenue": revenue, "surprise_pct": surprise_pct
                        })
                except Exception:
                    continue
        return articles
    except Exception as e:
        logger.error(f"MarketWatch error: {e}")
        return []

# EDGAR Filings (US, async) - FULLY CORRECTED
# --- SEC Form Definitions (Cleaner maintenance) ---
SEC_FORMS_CORE = {"8-K", "10-Q", "10-K"}
SEC_FORMS_OWNERSHIP = {"4", "5"}  # Insider trades
SEC_FORMS_BENEFICIAL = {"SC 13D", "SC 13G", "SC 13DA", "SC 13GA"} 
SEC_FORMS_ALL = SEC_FORMS_CORE | SEC_FORMS_OWNERSHIP | SEC_FORMS_BENEFICIAL

# EDGAR Filings (US, async) - V22 Final
@sleep_and_retry
@limits(calls=8, period=1) # Keep your 8/sec limit (Safe)
async def fetch_edgar_filings(session: aiohttp.ClientSession, symbol: str, cutoff: datetime) -> List[Dict[str, Any]]:
    cik_map = get_cik_map()
    cik = cik_map.get(symbol.upper())
    if not cik:
        return []

    articles = []
    cutoff_aware = make_tz_aware(cutoff)
    
    # Your headers
    headers = {'User-Agent': 'Raghavan r3398465@gmail.com'} 

    try:
        submissions_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        async with session.get(submissions_url, headers=headers, timeout=15) as resp:
            if resp.status != 200: return []
            data = await resp.json()

        filings = data.get('filings', {}).get('recent', {})
        if not filings: return []

        for i in range(len(filings.get('accessionNumber', []))):
            form_type = str(filings['form'][i]).strip().upper()
            
            # Use the Set lookup (Faster/Cleaner than list)
            if form_type not in SEC_FORMS_ALL: continue

            pub_date = make_tz_aware(datetime.strptime(filings['filingDate'][i], '%Y-%m-%d'))
            if pub_date < cutoff_aware: continue

            accession_no = filings['accessionNumber'][i].replace('-', '')
            doc_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_no}/{filings['primaryDocument'][i]}"
            desc = filings['primaryDocDescription'][i] or ""

            # Smart Summaries
            if form_type == '4':
                summary = f"SEC Form 4: Insider Trading Activity detected. {desc}"
            elif '13D' in form_type:
                summary = f"SEC {form_type}: Major shareholder accumulation (>5%). {desc}"
            else:
                summary = f"Filing of form {form_type}. {desc}"

            articles.append({
                "headline": f"{symbol} SEC Filings {form_type}: {desc}",
                "summary": summary, 
                "source": "EDGAR",
                "form_type": form_type,  # <--- THIS IS THE MISSING KEY YOU NEED
                "published": pub_date.isoformat(),
                "engagement": 0, 
                "media_url": doc_url, 
                "reported_eps": None, "estimated_eps": None, "revenue": None, "surprise_pct": None
            })
            
        return articles
    except Exception as e:
        logger.error(f"EDGAR error for {symbol}: {e}")
        return []
# Indian Filings (NSE/BSE, async - Resolved for 2025 Endpoints)
async def fetch_indian_filings(session: aiohttp.ClientSession, symbol: str, cutoff: datetime) -> list:
    articles = []
    # NSE (Financial Results Endpoint)
    try:
        nse_url = f"https://www.nseindia.com/api/corporates-financial-results?symbol={symbol.replace('.NS','')}"
        headers = {'User-Agent': 'Mozilla/5.0', 'Accept': 'application/json', 'Referer': 'https://www.nseindia.com'}
        async with session.get(nse_url, headers=headers, timeout=15) as resp:
            if resp.status != 200:
                logger.warning(f"NSE API {resp.status} for {symbol}")
            else:
                data = await resp.json()
                for filing in data.get('data', [])[:10]:
                    pub_date_str = filing.get('filingDate', '')
                    pub_date = dateutil_parser.parse(pub_date_str) if pub_date_str else datetime.now()
                    if pub_date < cutoff or 'financial' not in filing.get('filingDesc', '').lower():
                        continue
                    summary = filing.get('filingDesc', '')
                    attach_url = filing.get('attachUrl', '')
                    if attach_url and attach_url.startswith('/'):
                        attach_url = f"https://www.nseindia.com{attach_url}"
                    if attach_url:
                        async with session.get(attach_url, headers=headers) as pdf_resp:
                            if pdf_resp.status == 200:
                                pdf_bytes = await pdf_resp.read()
                                summary += " " + extract_pdf_text(pdf_bytes)
                    reported_eps, estimated_eps, revenue, surprise_pct = parse_eps_revenue(summary)
                    articles.append({
                        "headline": filing.get('filingDesc', f"{symbol} NSE Filing"), "summary": summary,
                        "source": "NSE", "published": pub_date.isoformat(), "engagement": 0,
                        "media_url": attach_url, "reported_eps": reported_eps,
                        "estimated_eps": estimated_eps, "revenue": revenue, "surprise_pct": surprise_pct
                    })
    except Exception as e:
        logger.warning(f"NSE filings error for {symbol}: {e}")

    # BSE (Bulk XML Endpoint + Filter)
    try:
        bse_url = "https://www.bseindia.com/xml-data/corpfiling/NewCorpAnnouncements.xml"
        headers = {'User-Agent': 'Mozilla/5.0'}
        async with session.get(bse_url, headers=headers, timeout=15) as resp:
            if resp.status != 200:
                logger.warning(f"BSE XML {resp.status}")
            else:
                xml_content = await resp.text()
                root = ET.fromstring(xml_content)
                for ann in root.findall(".//announcement"):
                    scrip_cd = ann.find("scripcd").text if ann.find("scripcd") is not None else ""
                    if scrip_cd != symbol.replace(".BO", ""):
                        continue
                    news_title = ann.find("newstitle").text or ""
                    pub_date_str = ann.find("dt_tm").text or ""
                    pub_date = dateutil_parser.parse(pub_date_str) if pub_date_str else datetime.now()
                    if pub_date < cutoff or ("financial" not in news_title.lower() and "result" not in news_title.lower()):
                        continue
                    summary = news_title
                    attach_name = ann.find("attachmentname").text or ""
                    attach_url = ""
                    if attach_name:
                        attach_url = f"https://www.bseindia.com/xml-data/corpfiling/AttachLive/{attach_name}"
                        async with session.get(attach_url, headers=headers) as attach_resp:
                            if attach_resp.status == 200:
                                content = await attach_resp.read()
                                if attach_name.lower().endswith(".pdf"):
                                    summary += " " + extract_pdf_text(content)
                    reported_eps, estimated_eps, revenue, surprise_pct = parse_eps_revenue(summary)
                    articles.append({
                        "headline": news_title, "summary": summary, "source": "BSE",
                        "published": pub_date.isoformat(), "engagement": 0,
                        "media_url": attach_url, "reported_eps": reported_eps,
                        "estimated_eps": estimated_eps, "revenue": revenue, "surprise_pct": surprise_pct
                    })
    except Exception as e:
        logger.warning(f"BSE filings error for {symbol}: {e}")

    if not articles:
        logger.warning(f"[MONITOR] Indian filings returned 0 for {symbol}.")
    return articles[:10]

# Unified Filings
async def fetch_filings(symbol: str, cutoff: datetime, session: aiohttp.ClientSession) -> list:
    if is_indian_market(symbol):
        return await fetch_indian_filings(session, symbol, cutoff)
    else:
        return await fetch_edgar_filings(session, symbol, cutoff)

# Aggregator (Full Parallel Fetch)
async def aggregate_sources(query: str, symbol: str = None, cutoff: datetime = None):
    if cutoff is None:
        cutoff = datetime.now(timezone.utc) - timedelta(days=7)
    if not symbol: symbol = query

    async with aiohttp.ClientSession() as session:
        tasks = [
            fetch_newsapi(session, query, cutoff),
            asyncio.to_thread(fetch_google_rss_sync, query, cutoff),
            asyncio.to_thread(fetch_yahoo_sync, symbol, cutoff),
            asyncio.to_thread(fetch_yahoo_earnings_sync, symbol, cutoff),
            fetch_x_posts(query, cutoff),
            fetch_alpha_transcripts(session, symbol, cutoff),
            fetch_reddit_posts(query, cutoff),
            fetch_moneycontrol(session, query, cutoff),
            fetch_seekingalpha(session, query, cutoff),
            fetch_marketwatch(session, query, cutoff),
            fetch_filings(symbol, cutoff, session)
        ]
        all_data = await asyncio.gather(*tasks, return_exceptions=True)

    flat = [item for sublist in all_data if isinstance(sublist, list) for item in sublist]

    seen = set()
    unique = []
    for art in flat:
        if not art.get('headline'): continue
        identifier = (art['headline'], art['source'])
        if identifier not in seen:
            unique.append(art)
            seen.add(identifier)

    # Handle and log exceptions from gather
    for result in all_data:
        if isinstance(result, Exception):
            logger.error(f"An error occurred in an async task: {result}", exc_info=False)

    return unique