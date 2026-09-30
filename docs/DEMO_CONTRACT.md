# FinWise Demo — FROZEN INTERFACE CONTRACT (Colab + UI/UX)

> Single source of truth for all agents working on the Sunday demo.
> Scope: `app/gradio_app.py` (dashboard) + minimal pipeline compatibility fixes + Colab notebook.
> Out of scope: `chat_app_v2.py`, `chat_app_v3.py`, `streamlit_app.py`, `recall_engine/`, `kg/` logic, `tests/`.

## 0. Machine safety (non-negotiable)
The local machine is a 10+ year old i5 with ~7.4 GB RAM and no GPU. It must never be stressed.
- DO NOT `pip install` anything.
- DO NOT import/run tensorflow, torch, transformers, sentence_transformers, spacy, gradio, yfinance, plotly, pandas.
- DO NOT launch any app or run any notebook locally.
- Allowed local commands ONLY: `python3 -m py_compile`, stdlib `ast`/`json` scripts, `git`, text reads/edits.
- Heavy execution happens exclusively on Google Colab (free tier, T4 GPU when offered).

## 1. Required fixes (contract-preserving)
1. `core/orchestrator.py`: signature becomes `compute_state(symbol: str, days: int, kg_writer=None)`.
   KG write is skipped when `kg_writer is None`. Existing 3-arg unit tests MUST keep passing unchanged.
2. `intelligence/price/predictor.py` (Colab hardening only, no logic change):
   - `os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")` at module top BEFORE the tensorflow import.
   - `load_model` import: try `tensorflow.keras.models` then fallback `tf_keras.models`.
   - Load the model with a `compile=False` retry if the first load fails.
   - `make_predict_function()` called only if it exists (`hasattr` guard).
3. Neo4j is NOT used for the demo. `app/gradio_app.py` calls `compute_state(resolved_symbol, days, None)`.
   No `neo4j` import anywhere in app code. No Aura secrets.

## 2. Symbol resolution (new `core/symbols.py`)
Goal: the demo works for arbitrary US + Indian tickers.
- `normalize_symbol(raw: str) -> str` — strip + uppercase.
- `resolve_symbol(raw: str) -> tuple[str, list[str]]` — returns `(resolved, tried)`:
  - `^...` indices and explicit `.NS`/`.BO` suffixes pass through untouched (no probe).
  - otherwise probe Yahoo in order: bare symbol, then `SYMBOL.NS`, then `SYMBOL.BO`.
  - probe = `yf.Ticker(sym).fast_info["last_price"]` in try/except, fallback `yf.download(sym, period="5d", progress=False, threads=False)` non-empty check.
  - results memoised in a module-level dict so repeat clicks do not re-probe.
  - if nothing resolves: return `(normalized, tried)` and let the pipeline raise its own error.
- `yfinance` is imported lazily inside functions (module import stays light).
- Known ambiguity to document in the UI: bare `INFY` resolves to the US ADR; NSE users type `INFY.NS`.
- US tickers (AAPL, NVDA, MSFT, TSLA...), indices (^NSEI, ^BSESN, ^GSPC) and explicit Indian tickers
  (`RELIANCE.NS`, `TCS.NS`, `HDFCBANK.NS`) must all work.

## 3. Dashboard callback contract (`app/gradio_app.py`)
`run_finwise(symbol: str, days: int) -> tuple` — 8 outputs, in this exact order:
1. `kpi_html` (gr.HTML) — KPI deck: Market Regime, DSP, 5% VaR, risk context, eval time, resolved symbol, elapsed seconds.
2. `status_html` (gr.HTML) — status chips: mode = STANDALONE (no KG), articles count, data window.
3. `hist_plot` (gr.Plot) — Observed vs Model Fit.
4. `mc_plot` (gr.Plot) — Monte Carlo fan (PRIMARY).
5. `fc_plot` (gr.Plot) — Forward path.
6. `aspects_html` (gr.HTML) — signed decomposition bars, sorted by |score|, top 6.
7. `articles_html` (gr.HTML) — top impact articles table (direction badge, impact, weight, magnitude, headline escaped, source, entities, relative time).
8. `error_html` (gr.HTML) — empty string on success, error banner otherwise.

Hard rules:
- Inputs clamped: `days = min(max(int(days), 1), 30)`; blank ticker -> error banner, no exception.
- Everything read defensively with `.get()` (the smoke test mocks `compute_state` with `{"symbol": "", "price": {}, "sentiment": {}}`).
- Any exception -> return a full 8-tuple with error banner + placeholders. NEVER raise.
- All user/article text HTML-escaped (`html.escape`).
- `downside_var_percentile(history_prices, current_var)` helper preserved.
- Launch guarded: `if __name__ == "__main__": demo.queue().launch(share=True)`.
- Module-level `demo` object stays importable for the notebook.
- Gradio target: 5.49.1 (pin). Only stable APIs: Blocks, Row, Column, HTML, Plot, Textbox, Slider, Button, Markdown.

State dict provided by `compute_state` (raw_state):
```
symbol, as_of,
price: {trend, forecast_slope, downside_risk},
sentiment: {global_score, label, aspects: {name: {label, score}}},   # may be missing/error dict
history: {dates[], actual[], predicted[]},
forecast: {dates[], predicted[]},
monte_carlo: {downside_var, figure},
indicators: {rsi, macd, signal, momentum},
articles: [{headline, source, published, impact, magnitude, weight, direction, entities}]
```

## 4. Colab notebook contract (`FinWise_Risk_Terminal_Colab.ipynb`, repo root)
Cell order (fixed):
1. env: `TF_USE_LEGACY_KERAS=1`, `PIPELINE_DEBUG=0`, `TOKENIZERS_PARALLELISM=false` — BEFORE any TF import.
2. install: `%pip install -q "gradio==5.49.1" plotly feedparser praw PyPDF2 ta ratelimit jinja2`
   (everything else is preinstalled in the Colab image: tf 2.20, tf_keras 2.20.1, torch 2.11, transformers 5.17,
   sentence-transformers 5.7, spacy 3.8.16 + en_core_web_sm, accelerate, statsmodels, textblob, yfinance 0.2.66).
3. optional secrets (NEWS_API_KEY only; silently skipped if absent).
4. clone or pull `https://github.com/Raghavan-27-5/complete-finwise-system.git` -> `/content/finwise`, print commit hash.
5. chdir + assert `intelligence/price/stock_price_model.h5` exists + timed `import app.gradio_app`.
6. UI self-test with a synthetic state (monkeypatch `app.gradio_app.compute_state`, call real `run_finwise`, assert 8 outputs) — no network.
7. warm-up: `compute_state("AAPL", 7, None)` timed, print trend/DSP/VaR/articles.
8. launch `share=True` (public gradio.live URL + inline iframe in Colab).
9. Plan-B launch cell (share=False + `google.colab.output.serve_kernel_port_as_window`) + keep-alive JS.
10. markdown: troubleshooting matrix + T4 notes + demo talk-track pointer to `docs/COLAB_DEMO_RUNBOOK.md`.

## 5. Local verification protocol (what "verified" means here)
- `python3 -m py_compile <changed files>` must pass.
- stdlib-AST checks: `run_finwise` has 2 params; orchestrator has `kg_writer` defaulted to None; app has no `neo4j` import;
  notebook JSON parses (`json.load`); every notebook code cell passes `ast.parse` after stripping `%`/`!` magics.
- No import of heavy modules, ever, locally. Runtime truth is established by the notebook's self-test + warm-up cells on Colab.
