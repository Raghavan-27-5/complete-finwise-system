# SYSTEM_FORENSIC_AUDIT.md

## 1. Executive System Summary

**Status**: "Pre-Alpha" Prototype masquerading as "Institutional-Grade" software.

**Reality vs Intent**:
The system claims to be a "modular, high-frequency financial intelligence engine" with "probabilistic" modeling. In reality, it is a **brittle, synchronous script collection** that wraps basic libraries (yfinance, TextBlob, Transformers) with hardcoded heuristics and fragile HTML scraping.

**Critical Finding**:
The "Recall Engine" (Conversational AI) has **zero persistence**. It forgets everything upon restart. The "Knowledge Graph" is written to but barely read from in a meaningful, analytical way during the core pipeline—it serves mostly as a data dump. The "Monte Carlo" simulation uses **hardcoded parameters** (kappa=2.0, xi=0.15) regardless of the asset's actual historical volatility profile, rendering the "risk metrics" mathematically completely arbitrary.

---

## 2. File-by-File Analysis

### Root & Configuration
- **Note**: `.env` contains critical keys (Neo4j, Gemini, NewsAPI). `requirements.txt` is standard but heavy (`tensorflow` + `torch`).
- **`core/config.py`**:
  - **Risk**: Hardcoded `BASE_SOURCE_WEIGHTS` and `ASPECT_KEYWORDS`. New source names or updated HTML structures will break the logic immediately.
  - **Fragility**: The "Enterprise Aspect System" is purely keyword-matching, not semantic understanding.

### App Layer
- **`app/streamlit_app.py`**:
  - **Risk**: `process_chart_data` and `display_chart` are **Placeholders** returning `None` or text. The "Insights" are just LLM hallucinations on top of a query result.
  - **Security**: Direct pass-through of user intent to LLM-generated Cypher queries (`QueryGenerator`). High injection risk.
- **`app/gradio_app.py`**:
  - **Performance**: Calls `compute_state` synchronously. This blocks the entire UI thread while fetching data from 10+ sources.
  - **Visuals**: Plots are generated server-side with Plotly, limiting interactivity.

### Core Layer
- **`core/orchestrator.py`**:
  - **Architecture**: A simple synchronous function `compute_state`. It orchestrates `price` and `sentiment` by blocking on them.
  - **State**: Constructs a dictionary object that couples the UI to the backend structure tightly.

### Intelligence Layer: Price (`intelligence/price`)
- **`predictor.py`**:
  - **CRITICAL MATH FLAW**: `predict_future` uses **recursive feedback** (feeding prediction back into input) for the LSTM. This compounds error exponentially. For 7+ days, the result is noise.
  - **CRITICAL MATH FLAW**: `monte_carlo_heston` initializes with `sigma0` from data but uses **hardcoded** mean reversion (`kappa=2.0`), vol-of-vol (`xi=0.15`), and correlation (`rho=-0.4`). It ignores the asset's actual stochastic properties.
  - **Memory Leak**: `_YF_CACHE` is a global dictionary that never expires. Infinite growth over time.
  - **Network**: `safe_download` disables threading due to a "curl bug", forcing slow serial downloads.

### Intelligence Layer: Sentiment (`intelligence/sentiment`)
- **`fetchers.py`**:
  - **Fragility**: "Production" scrapers for `Moneycontrol`, `SeekingAlpha`, `MarketWatch` use **raw HTML parsing** (`soup.select('li.startup-videos-item')`). One div change breaks the entire pipeline.
  - **Parsing**: `parse_eps_revenue` uses regex on English sentences. Will fail on "Earnings dropped *to* $5M" vs "dropped *by* $5M".
- **`models.py`**:
  - **Performance**: Loads `spacy` model and `FinBERT` pipeline on import. Heavy cold start.
  - **Logic**: `compute_ensemble_sentiment` mixes `SBERT` similarity with `FinBERT`. "Hybrid Engine" claim is just snippet filtering.
- **`pipeline.py`**:
  - **Complexity**: `calculate_institutional_dsp` is a "magic number" formula with arbitrary weights (`halflife=7.0`, `neutral_discount=0.15`).

### Recall Engine (`recall_engine`)
- **`chatbot.py`**:
  - **LLM Risk**: `chatbot_no_context` has a system prompt to "return only what's asked". Extremely susceptible to prompt injection.
- **`conversation_manager.py`**:
  - **Data Loss**: Stores history in `self.history = deque()`. **In-Memory Only**. All context is lost if the app restarts.
- **`llm_query_generator.py`**:
  - **Reliability**: Relies on `chatbot_no_context` to "validate" its own query. Uses a "Two-step" LLM hop for every DB query. Slow and expensive.

---

## 3. Architecture Assessment

**Data Flow**:
`User -> UI -> Orchestrator -> [Price (Sync), Sentiment (Async wrapped in Sync)] -> Neo4j`

**Control Flow**:
1.  **Monolithic & Blocking**: The `orchestrator` forces async sentiment fetchers to run via `_run_async` (loop closing/opening hacks), checking for IO-bound operations in a blocking way.
2.  **Coupling**: `fetchers.py` is tightly coupled to specific HTML layouts. `predictor.py` is coupled to a specific `.h5` model file structure.

**Complexity & Bugs**:
-   **Hidden**: `_YF_CACHE` in `predictor.py` will silently consume RAM until OOM.
-   **Silent Failure**: `fetchers.py` catches exceptions broadly and logs errors, often returning empty lists. The system will look "working" but essentially return zero intelligence if APIs change.

---

## 4. PM-Level Risks

-   **Product Lie**: The system is sold as "Institutional-Grade". It is currently a **fragile prototype**.
-   **Failure Mode**: If Google News changes RSS format or Moneycontrol changes CSS, the "Sensitivity Intelligence" drops to zero immediately.
-   **Trust**: The "Monte Carlo" simulation looks fancy but the math is rigged (hardcoded params). Traders relying on this for VaR (Value at Risk) could incur massive losses because the model ignores true tail risk of the specific asset.
-   **Onboarding**: New engineers will assume the "Enterprise Aspect System" is a smart model, but it's just a keyword list in `config.py`.

---

## 5. Engineering-Level Risks

-   **Technical Debt**: `intelligence.sentiment.fetchers` is 600 lines of brittle scraping script. Needs immediate refactoring to official APIs or reliable data vendors.
-   **Testing**: Tests rely on mocks (`integration/test_full_pipeline.py`). There are **no** live end-to-end tests that verify if the HTML parsers still work against real websites.
-   **Scalability**:
    -   **Zero**: The system runs in-process. `price` prediction loads TensorFlow in-process. `sentiment` runs NLP pipelines in-process. No worker queues (Celery/Redis).
    -   **Concurrency**: Global variables (`_YF_CACHE`, `_aspect_model`) make this thread-unsafe for high-load servers.

## 6. Explicit Non-Actions

-   **No Code Changes**: I have not fixed the `_YF_CACHE` leak or the brittle regexes.
-   **No Refactoring**: I left the synchronous orchestrator as is.
-   **Reason**: Strict "Forensic Analysis" mode. Modifying the code would obscure the existing architectural flaws necessary for the audit.
