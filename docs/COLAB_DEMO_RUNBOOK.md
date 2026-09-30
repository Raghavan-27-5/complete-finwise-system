# FinWise · Colab Demo Runbook

**Companion notebook:** `FinWise_Risk_Terminal_Colab.ipynb` (repo root) ·
**Dashboard:** `app/gradio_app.py` · **Contract:** `docs/DEMO_CONTRACT.md`

The demo runs the *real* pipeline on Google Colab (free tier, T4 GPU when offered):
a Keras LSTM price path, a Heston-style Monte-Carlo risk envelope and FinBERT
narrative pressure, in standalone mode (no Neo4j, no Aura credentials).

---

## 1. T-20 minutes — pre-flight checklist

Do this **before** the audience arrives. Each step is numbered because the demo is
timed from step 7 onwards.

1. **Open the notebook in Colab:** `File ▸ Open notebook ▸ GitHub ▸`
   `Raghavan-27-5/complete-finwise-system ▸ FinWise_Risk_Terminal_Colab.ipynb`
   (or upload the `.ipynb` from the repo root).
2. `Runtime ▸ Change runtime type ▸ T4 GPU ▸ Save`. If Colab says no GPU is
   available, stay on CPU — everything works, it is only slower.
3. `Runtime ▸ Run all`. **Do not** click through cells manually: cell 2 must
   finish before anything imports TensorFlow (`TF_USE_LEGACY_KERAS=1`).
4. Watch cell 3 print the GPU line, the pinned `gradio==5.49.1`, and the version
   audit (all `absent` lines are a red flag → re-run the `%pip` line and restart).
5. Watch cell 7: `✅ UI self-test passed in X.XXs (8/8 outputs)`. This proves the
   dashboard contract with **no** network and **no** models. If it fails, nothing
   downstream matters — fix it now.
6. Watch cell 8 finish: `✅ pipeline warm — live clicks are now fast`. First run
   downloads FinBERT + MiniLM (~500 MB); budget 3-8 minutes for it.
7. Cell 9 prints `Running on public URL: https://xxxx.gradio.live`.
   **Open that URL in a NEW browser tab** and keep the Colab tab open behind it.
8. **Do one dry click:** ticker `AAPL`, horizon `7`, `▶ RUN EVALUATION`. Confirm
   all five panels populate and the status strip says `mode STANDALONE · no graph DB`.
   This dry click is the single most valuable step in this list.
9. Optional: open the share URL on the phone as well (mobile render check), and
   start cell 11 (keep-alive).
10. Set a stopwatch/browser timer: from the first click you have ~20-60 s of
    talking to fill. Rehearse the talk track in section 3 at least once.

---

## 2. Timing budget (measured shape, not optimism)

| Stage | Cell | Typical | Worst case | Notes |
| --- | --- | --- | --- | --- |
| Env pinning | 2 | < 1 s | < 1 s | Must precede every TF import |
| `%pip install` (gradio pin + 6 deps) | 3 | 20-45 s | 90 s | One time per runtime |
| `git clone` (shallow) | 5 | 5-15 s | 60 s | Fast-forward pull on re-runs ≈ 5 s |
| `import app.gradio_app` (loads TF) | 6 | 15-40 s | 90 s | TF import dominates |
| Self-test (synthetic state) | 7 | 2-5 s | 10 s | No network, no models |
| **FinBERT + MiniLM download** | 8 | 2-6 min | 9 min | **~500 MB, first run only** |
| Warm-up `compute_state("AAPL")` | 8 | 40-120 s | 4 min | Includes LSTM + spaCy + news fetch |
| Warm-up `compute_state("RELIANCE.NS")` | 8 | 20-60 s | 2 min | Cheap once models are cached |
| **Per demo click (warm)** | 9 | **20-60 s** | 90 s | Network-bound; GPU only saves a few s |
| Launch + share tunnel | 9 | 5-20 s | 60 s | If it hangs → Plan B, cell 10 |

**Cold start total: ≈ 8-12 min.** Warm click total: **≈ 20-60 s**. That is why the
`T-20` checklist exists.

---

## 3. Demo talk track — three acts

One ticker per act, `Horizon = 7` for all three. Each click gives you 20-60 s of
narration, so narrate the *architecture* between acts and the *numbers* after each
render.

### Act 1 — `AAPL` · "the mega-cap with the richest news coverage"

Type `AAPL`, keep `7`, click `▶ RUN EVALUATION`.

- **What to point at:** *Market Regime*, *Global Directional Bias (DSP)*, *Downside
  Risk · 5% VaR*.
- **Why this ticker:** the densest English-language coverage, so the article table
  and aspect decomposition are at their most convincing. It is the recap act.
- **Line:** "AAPL has the deepest news flow of any listed company, so this is the
  best case for the narrative engine: more articles means a better-sampled DSP."

### Act 2 — `NVDA` · "the AI-driven margin story"

Type `NVDA`, click `▶ RUN EVALUATION`.

- **What to point at:** the DSP sign flipping versus Act 1, the aspect bars (which
  narratives dominate), and a **fatter Monte-Carlo fan** — higher realized vol
  widens the 2.5-97.5% envelope.
- **Line:** "Same model, same window, different company: the LSTM trend and the
  narrative pressure are computed independently, and here you can see the aspect
  decomposition explaining *why* the bias sits where it does."

### Act 3 — `RELIANCE` · "Indian market coverage + auto-resolution of the `.NS` suffix"

Type `RELIANCE` (bare), click `▶ RUN EVALUATION`.

- **What to point at:** the **status strip**, which shows the probe chain
  (`RELIANCE → RELIANCE.NS`), i.e. the resolver found the NSE listing for you.
- **Note to say out loud:** if the resolver cannot find coverage, type the explicit
  `RELIANCE.NS` — and bare `INFY` intentionally resolves to the US ADR; use
  `INFY.NS` for the NSE line.
- **Line:** "Nothing here is US-only: index symbols (`^NSEI`), explicit suffixes
  (`.NS`, `.BO`) pass straight through, bare tickers get probed in order."

### The three headline numbers (same for every act)

1. **Market Regime** — the LSTM price-trend classification (Bullish / Bearish /
   Neutral). Point at the coloured pill, not the word.
2. **DSP** (Global Directional Bias) — standardised FinBERT narrative pressure,
   signed. Positive = constructive coverage, negative = deteriorating.
3. **Downside Risk · 5% VaR** — *always* read it with its percentile chip, e.g.
   "the 5% VaR is 0.0523 — that is the 87.4th percentile of this asset's own rolling
   20-day VaR". The percentile is what turns a number into a statement, and it is
   asset-relative by construction.

### The Monte-Carlo fan, in one sentence

"Each faint line is one simulated future path; the mean is the drift-anchored
expectation and the dotted ribbon is the 2.5-97.5% envelope."

---

## 4. What to say while it computes (~20-60 s)

The spinner is not dead air — it is the best architecture slide you have. Walk the
call chain, top to bottom:

```
app/gradio_app.py  run_finwise(symbol, days)
      → core/orchestrator.py  compute_state(symbol, days, kg_writer=None)
            → intelligence/price    → LSTM (Keras 2 .h5) + Heston-style Monte-Carlo
            → intelligence/sentiment→ FinBERT + SBERT + spaCy NER + Granger causality
            → knowledge-graph adapter (kg_writer=None ⇒ skipped in standalone mode)
```

- **`app/gradio_app.py`** — a frozen callback contract: eight outputs, clamped
  horizon, blank ticker never raises, all article text HTML-escaped. "The UI is a
  thin shell; every number comes from the pipeline below it."
- **`core/orchestrator.py`** — one `compute_state` call, third argument `None`.
  "That `None` is standalone mode: the graph write is skipped, so there are no
  Neo4j credentials anywhere in this demo."
- **`intelligence/price`** — the trained LSTM classifies the regime and projects the
  forward path; a Heston-style Monte-Carlo simulates the risk envelope and returns
  the 5% VaR. "Deterministic model, stochastic risk layer — that is why the fan and
  the line disagree, and both are right."
- **`intelligence/sentiment`** — FinBERT scores each headline, SBERT clusters the
  narratives into aspects, spaCy extracts the entities you see as chips, and a
  Granger test asks whether the news actually leads the price. "The DSP is signed
  and standardised: it is a direction, not a probability."
- **Knowledge-graph adapter** — "unused in standalone mode by contract; the same
  call signature writes a Neo4j snapshot when a `kg_writer` is supplied."

Close with the honest GPU line: "the T4 accelerates the simulation and the LSTM
inference; the wall-clock you are watching is mostly network — news fetch and
model weight loading."

---

## 5. Fallback matrix

| Failure | Symptom in the room | Immediate action |
| --- | --- | --- |
| **share blocked** | Cell 9 never prints a `*.gradio.live` URL, or says "Could not create share link" | Stop cell 9 → run **cell 10** (Plan B: `serve_kernel_port_as_window(7860)` + `share=False`). Present the link Colab prints. |
| **T4 unavailable** | An info banner about CPU / "no GPU backend" | Do nothing different. Say "running on CPU — identical numbers, slightly slower" and continue. Never restart to hunt for a GPU mid-demo. |
| **model load error** | `ValueError: Unknown layer` / `could not deserialize` on the first click | Do **not** debug live. Start **cell 7's self-test** outcome as your proof the UI works, then `Runtime ▸ Restart session ▸ Run all` during a break. |
| **no-article ticker** | Sentiment panel shows the empty state ("no relevant coverage") | Expected behaviour, not a bug. Say: "price and risk are unaffected; the narrative layer reports low information." Switch to `AAPL`/`NVDA`. |
| **disconnect** | "Runtime disconnected" banner, the UI stops answering | Cell 11 must be running. Reconnect (`Runtime ▸ Reconnect`), then re-run cells **2 → 6** and **9**. Do not re-run cell 3 unless the pip line failed. |
| **Colab quota** | "You are out of GPU units" | Same code, CPU runtime. Warn the room, continue. If the whole session is unusable, fall back to the recorded screen capture / the notebook's cell 7 self-test output. |

---

## 6. Never do this

- **Don't re-run cell 3 (`%pip install`) mid-demo.** It can reinstall/downgrade a
  package under a live interpreter and break the running app. Install once, before
  the dry click.
- **Don't restart the runtime after warming up.** A restart throws away the ~500 MB
  of FinBERT/MiniLM weights and the LSTM in memory, and puts you back at an 8-12
  minute cold start.
- **Don't hardcode API keys in the notebook.** Use Colab Secrets (🔑 sidebar) named
  `NEWS_API_KEY`; the pipeline is fully functional key-free and cell 4 prints an
  informational line when the key is absent.
- **Don't click `▶ RUN EVALUATION` repeatedly while a job is in flight.** Each click
  is a new 20-60 s pipeline run; queueing them makes the UI look stuck. One click,
  then narrate.
- **Don't run the notebook and the local app at the same time**, and don't run
  cell 9 and cell 10 together — two servers on one runtime fight over the port.
- **Don't edit code in `/content/finwise`.** It is a throwaway clone; the edit
  disappears with the runtime. Fix it in the repo and re-run cell 5.
- **Don't hand over the `*.gradio.live` URL and then stop cell 9.** Stopping the
  cell kills the server; the audience's tab goes blank.
- **Don't close the Colab browser tab.** Cell 11 clicks the connect button, but a
  closed tab is still a dead session.

---

## 7. Post-demo

- **The code lives in the repo, not in Colab.** Colab runtimes are ephemeral: the
  `/content/finwise` clone, the pip layer and every downloaded weight vanish when the
  session ends. Anything worth keeping must be committed from your machine:

  ```bash
  cd ~/projects/complete-finwise-system
  git status --short          # review what actually changed
  git add -A && git commit -m "docs: Colab runbook + notebook"
  git push origin main
  ```

- Kill the runtime after the demo: `Runtime ▸ Disconnect and delete runtime` frees
  your Colab free-tier quota for the next session.
- Nothing was written to Neo4j during the demo (`kg_writer=None` by contract), so
  there is no graph state to clean up — the graph store is untouched by design.
- If the demo produced a genuinely interesting number, capture it from the KPI deck
  (regime, DSP, 5% VaR + percentile, eval time) into the "results" section of your
  notes explicitly — Colab will not remember it for you.