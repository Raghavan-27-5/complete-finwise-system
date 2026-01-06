# config.py

# ================================================
# FINWISE ENTERPRISE CONFIGURATION (v20.0)
# ================================================

## Data Source Weights (Refined for Signal-to-Noise Ratio)

## Data Source Weights (Scaled for V24 Multipliers)
# Max Base = 1.20. Max Final Weight (after boosts) ≈ 1.80 - 2.00.
BASE_SOURCE_WEIGHTS = {
    # TIER 1 — Institutional Primary (High Trust)
    "Reuters": 1.15,
    "Bloomberg": 1.15,
    "WSJ": 1.12,
    "Financial Times": 1.12,
    "CNBC": 1.10,
    "Barrons": 1.10,

    # TIER 2 — Strong Aggregators
    "Yahoo": 1.00,
    "Yahoo Finance": 1.05,
    "MarketWatch": 1.05,
    "Investing.com": 1.00,
    "SeekingAlpha": 1.00,
    "Benzinga": 0.95,
    "Google RSS": 0.90,

    # INDIA
    "Moneycontrol": 1.10,
    "Economic Times": 1.10,
    "LiveMint": 1.05,
    "NDTV Profit": 1.00,

    # TIER 3 — Social (Noise)
    "Reddit": 0.30,
    "StockTwits": 0.30,
    "X": 0.30,
    "Twitter": 0.30,
    "YouTube": 0.20,

    # TIER 4 — Low-Quality Aggregators (The Spam List)
    "MarketBeat": 0.50,  # Nerfed to 50%
    "Zacks": 0.50,
    "InvestorPlace": 0.40,
    "The Motley Fool": 0.40,

    # TIER 5 — Official Filings (Signal)
    # Set to 1.20 so that with a 1.5x boost, they hit ~1.8 (High but not capped)
    "EDGAR": 1.20,
    "SEC": 1.20,
    "NSE": 1.20,
    "BSE": 1.20
}

## Caching and System Configuration
CACHE_DB = "sentiment_cache_v20.db"
CACHE_TTL_HOURS = 4 

## Source Specific Configuration
REDDIT_SUBREDDITS = [
    "investing", "stocks", "wallstreetbets", "SecurityAnalysis",
    "ValueInvesting", "StockMarket", "options", "pennystocks",
    "quantfinance", "financialindependence", "economics", "globalmarkets"
]

SOURCE_NAME_MAP = {
    "yahoo finance": "Yahoo", "yahoo": "Yahoo",
    "google rss": "Google RSS", "google news": "Google RSS",
    "newsapi": "NewsAPI", "moneycontrol": "Moneycontrol",
    "bloomberg": "Bloomberg", "reuters": "Reuters",
    "cnbc": "CNBC", "seeking alpha": "SeekingAlpha",
    "seekingalpha": "SeekingAlpha", "marketwatch": "MarketWatch",
    "wsj": "WSJ", "wall street journal": "WSJ",
    "ft": "Financial Times", "financial times": "Financial Times",
    "benzinga": "Benzinga"
}

# ================================================
# ENTERPRISE ASPECT SYSTEM (GENERIC, ALL STOCKS)
# ================================================

# Dictionary Keys = The Aspect ID used in the database
# Dictionary Values = Human readable descriptions (Useful for future LLM prompts)
ASPECT_CATEGORIES = {
    "earnings_revenue": "Financial performance: earnings, revenue, forecasts, beats/misses.",
    "costs_margins": "Costs, expenses, margin changes, opex/capex.",
    "liquidity_balance": "Cash flow, debt levels, refinancing, liquidity conditions.",
    "dividends_buybacks": "Dividend policy, payouts, share repurchases.",
    "leadership": "CEO/CFO actions, executive behavior, management decisions.",
    "corporate_strategy": "Strategic pivots, restructuring, expansion, business model changes.",
    "mna_partnerships": "Acquisitions, mergers, JVs, partnerships, collaborations.",
    "legal_regulatory": "Lawsuits, regulatory actions, compliance issues, fines.",
    "competition": "Competitive pressure, market share battles, rivals.",
    "product_services": "Product launches, updates, failures, quality issues, recalls.",
    "technology_innovation": "Technology advancements, patents, R&D, innovation milestones.",
    "supply_chain": "Manufacturing, production capacity, shortages, logistics.",
    "customer_experience": "Customer complaints, satisfaction, reviews, support quality.",
    "brand_reputation": "Brand image, scandals, public backlash, media perception.",
    "labor_workforce": "Hiring, layoffs, unions, strikes, employee sentiment."
}

# --------------------------------------------------------------------
# Keyword dictionary used as *initial triggers*
# --------------------------------------------------------------------

ASPECT_KEYWORDS = {
    "earnings_revenue": [
        "earnings", "revenue", "profit", "loss", "eps", "forecast",
        "guidance", "beat", "miss", "estimates", "top line", "bottom line",
        "financial results", "sales growth", "organic growth"
    ],

    "costs_margins": [
        "costs", "expenses", "margin", "opex", "capex",
        "pricing power", "cost-cutting", "profit margin", "operating leverage",
        "efficiency", "overhead"
    ],

    "liquidity_balance": [
        "debt", "cash flow", "liquidity", "solvency",
        "refinancing", "credit rating", "balance sheet", "cash burn",
        "liabilities", "bonds", "default"
    ],

    "dividends_buybacks": [
        "dividend", "payout", "yield", "buyback",
        "share repurchase", "capital return", "special dividend"
    ],

    "leadership": [
        "ceo", "cfo", "executive", "management", "board",
        "resigns", "steps down", "appointed", "fired", "leadership",
        "succession", "insider trading", "founder"
    ],

    "corporate_strategy": [
        "strategy", "pivot", "restructuring", "expansion",
        "focus", "market entry", "business model", "strategic plan",
        "spin-off", "divestiture"
    ],

    "mna_partnerships": [
        "acquisition", "merger", "joint venture", "partner",
        "partnership", "collaboration", "takeover", "buyout", "deal",
        "synergy", "hostile bid"
    ],

    "legal_regulatory": [
        "lawsuit", "litigation", "regulator", "regulation",
        "investigation", "compliance", "fine", "court",
        "settlement", "violation", "antitrust", "sec", "subpoena"
    ],

    "competition": [
        "competition", "competitor", "market share",
        "rival", "industry pressure", "competitive landscape",
        "disruptor", "benchmark", "market leader"
    ],

    "product_services": [
        "product", "service", "launch", "recall",
        "quality", "update", "feature", "design", "malfunction",
        "flagship", "adoption", "defect"
    ],

    "technology_innovation": [
        "technology", "software", "hardware", "patent",
        "ai", "algorithm", "innovation", "r&d", "proprietary",
        "breakthrough", "next-gen", "upgrade"
    ],

    "supply_chain": [
        "supply chain", "factory", "production", "manufacturing",
        "capacity", "output", "shortage", "logistics", "inventory",
        "supplier", "raw materials"
    ],

    "customer_experience": [
        "customer", "complaint", "review", "satisfaction",
        "service quality", "support", "feedback", "churn", "loyalty"
    ],

    "brand_reputation": [
        "brand", "reputation", "image", "scandal",
        "public backlash", "controversy", "media perception", "boycott",
        "pr crisis"
    ],

    "labor_workforce": [
        "labor", "employees", "hiring", "layoff",
        "union", "strike", "workforce", "staffing", "wages",
        "talent", "headcount"
    ]
}