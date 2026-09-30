"""FinWise · Market Intelligence Terminal.

Dark-luxury Gradio dashboard for the FinWise pipeline: an LSTM price path, a
Heston-style Monte-Carlo risk envelope and FinBERT narrative pressure, rendered
in standalone mode (no graph database, no Aura credentials).

Module surface
--------------
``run_finwise(symbol, days)``
    The frozen dashboard callback. Returns an 8-tuple in the order mandated by
    ``docs/DEMO_CONTRACT.md`` section 3.
``demo``
    Module-level ``gr.Blocks`` object, importable by the Colab notebook.

Hard rules honoured here:
    * ``days`` is clamped to 1..30 and a blank ticker never raises.
    * Every state read goes through ``.get()`` with sane defaults, so the mock
      state used by the notebook self-test (``{"symbol": "", "price": {},
      "sentiment": {}}``) renders gracefully.
    * Any exception yields a full 8-tuple with an error banner, never a raise.
    * All user/article text is HTML-escaped.
"""

import os
import sys
import html
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as _FuturesTimeout

# Per-stage click timings land here so a slow click is diagnosable from the
# Colab log without switching on the pipeline's own debug output.
logging.basicConfig(level=logging.INFO, format="%(asctime)s [FW] %(message)s")
_perf_log = logging.getLogger("finwise.perf")

# ---------------------------------------------------------------------------
# UI click budget (gradio_app.py owns this — nothing downstream may exceed it).
# A single Gradio click = resolve + price/sentiment engines + render. The
# engines each carry their own generous timeouts (90s/150s), so without a UI
# ceiling one click can stack them into the 300s+ queue ETA in the screenshot.
# Clamp every downstream cap DOWN to a sub-60s envelope *before* importing the
# pipeline (all those modules read os.environ once at import time).
# ---------------------------------------------------------------------------
CLICK_BUDGET_SECONDS = 55.0
_CLICK_ENV_CAPS = {
    # price+sentiment pair as seen by the orchestrator
    "FINWISE_ENGINE_TIMEOUT": "40",
    # whole sentiment engine (adapter)
    "FINWISE_SENTIMENT_TIMEOUT": "35",
    # news aggregation inside the sentiment pipeline
    "FINWISE_FETCH_TIMEOUT": "20",
    # blocking yfinance calls inside the pipeline
    "FINWISE_YF_TIMEOUT": "8",
    # ticker probe chain (symbols.py)
    "FINWISE_PROBE_TIMEOUT": "5",
    "FINWISE_RESOLVE_BUDGET": "8",
    # FinBERT fan-out: articles × (1 global + N aspects)
    "FINWISE_MAX_ARTICLES": "8",
    "FINWISE_ASPECT_SCORE_LIMIT": "3",
}
for _k, _cap in _CLICK_ENV_CAPS.items():
    try:
        _cur = os.environ.get(_k)
        if _cur is None:
            os.environ[_k] = _cap
        elif float(_cur) > float(_cap):
            os.environ[_k] = _cap
    except Exception:
        try:
            os.environ[_k] = _cap
        except Exception:
            pass

# Ensure the project root is importable when this file is run from anywhere.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import gradio as gr
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timezone

from core.orchestrator import compute_state
from core.symbols import resolve_symbol

# Single-flight + result cache (queue-ETA fix, gradio_app.py only).
_CLICK_LOCK = threading.Lock()
_CACHE_LOCK = threading.Lock()
_ORPHAN_LOCK = threading.Lock()
_ORPHAN_RUNNING = False
_RESULT_CACHE = {}
_RESULT_CACHE_TTL_SECONDS = 120.0



# =============================================================================
# DESIGN TOKENS (mirrored in CUSTOM_CSS so charts and DOM never drift apart)
# =============================================================================
FONT_UI = "Inter, system-ui, -apple-system, 'Segoe UI', sans-serif"
FONT_MONO = "'JetBrains Mono', 'SF Mono', Menlo, Consolas, monospace"

C_TXT = "#e2e8f0"
C_MUTED = "#94a3b8"
C_DIM = "#64748b"
C_ACC = "#22d3ee"
C_ACC2 = "#8b5cf6"
C_ACC3 = "#ec4899"
C_OK = "#34d399"
C_BAD = "#fb7185"
C_WARN = "#fbbf24"
C_GRID = "rgba(148,163,184,0.10)"
C_ZERO = "rgba(148,163,184,0.25)"
C_FAN = "rgba(148,163,184,0.35)"

BLANK_TICKER_MESSAGE = "Enter a ticker to evaluate — e.g. AAPL, NVDA or RELIANCE.NS"


# =============================================================================
# ASSET-RELATIVE VAR CONTEXT (PM-GRADE) — preserved verbatim per contract
# =============================================================================
def downside_var_percentile(history_prices, current_var):
    """
    Compute asset-relative percentile for 5% VaR.
    Uses historical realized returns as empirical context.
    """
    try:
        prices = pd.Series(history_prices).astype(float)
        returns = np.log(prices / prices.shift(1)).dropna()

        if len(returns) < 100:
            return None

        rolling_var = returns.rolling(20).quantile(0.05).dropna().abs()

        if len(rolling_var) == 0:
            return None

        pct = (rolling_var < current_var).mean()
        return round(100 * pct, 1)

    except Exception:
        return None


# =============================================================================
# SMALL NUMERIC / TEXT UTILITIES (all defensive, none of them ever raise)
# =============================================================================
def _elapsed(started):
    """Seconds since ``started`` (perf_counter), clamped at zero."""
    try:
        return max(0.0, time.perf_counter() - started)
    except Exception:
        return 0.0


def _safe_float(value, default=0.0):
    """Best-effort float conversion that swallows NaN / inf / junk."""
    try:
        out = float(value)
        if out != out or out in (float("inf"), float("-inf")):  # NaN / inf
            return default
        return out
    except Exception:
        return default


def _as_int(value, default=0):
    try:
        return int(value)
    except Exception:
        return default


def _entity_list(value):
    """Normalise the many shapes ``entities`` arrives in into a list of str."""
    try:
        if isinstance(value, dict):
            for key in ("organizations", "orgs", "ORG", "entities", "people"):
                candidate = value.get(key)
                if isinstance(candidate, (list, tuple)):
                    return [str(item) for item in candidate if item]
            flat = []
            for item in value.values():
                if isinstance(item, (list, tuple)):
                    flat.extend(str(sub) for sub in item if sub)
            return flat
        if isinstance(value, (list, tuple)):
            return [str(item) for item in value if item]
        if isinstance(value, str) and value.strip():
            cleaned = (
                value.replace("[", " ").replace("]", " ")
                .replace("{", " ").replace("}", " ")
                .replace('"', " ").replace("'", " ")
                .replace(",", " ")
            )
            return [part.strip() for part in cleaned.split() if part.strip()]
    except Exception:
        return []
    return []


def _relative_time(published):
    """Turn an ISO ``published`` string into "3h ago" / "2d ago".

    Any parse failure falls back to the raw (escaped) string so the table never
    loses information.
    """
    raw = "" if published is None else str(published).strip()
    if not raw:
        return "—"
    try:
        stamp = raw.replace("Z", "+00:00")
        moment = datetime.fromisoformat(stamp)
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)
        delta = datetime.now(timezone.utc) - moment
        seconds = delta.total_seconds()
        if seconds < 0:
            seconds = 0.0
        if seconds < 90:
            return f"{int(seconds)}s ago"
        if seconds < 5400:
            return f"{int(seconds // 60)}m ago"
        if seconds < 86400 * 2:
            return f"{int(seconds // 3600)}h ago"
        if seconds < 86400 * 365:
            return f"{int(seconds // 86400)}d ago"
        return f"{int(seconds // (86400 * 365))}y ago"
    except Exception:
        return html.escape(raw[:24])


# =============================================================================
# FIGURE HARMONISATION
# =============================================================================
def _style_fig(fig, height=300, hover="x unified"):
    """Apply the FinWise terminal look to any Plotly figure, in place."""
    try:
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=height,
            margin=dict(l=8, r=8, t=30, b=8),
            hovermode=hover,
            font=dict(family=FONT_UI, size=11, color=C_MUTED),
            hoverlabel=dict(
                bgcolor="rgba(5,7,12,0.92)",
                bordercolor="rgba(148,163,184,0.25)",
                font=dict(family=FONT_MONO, size=11, color=C_TXT),
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1.0,
                bgcolor="rgba(0,0,0,0)",
                borderwidth=0,
                font=dict(family=FONT_UI, size=11, color=C_MUTED),
            ),
            showlegend=True,
        )
        fig.update_xaxes(
            showline=False,
            showgrid=False,
            zeroline=False,
            linecolor="rgba(148,163,184,0.10)",
            gridcolor=C_GRID,
            zerolinecolor=C_ZERO,
            tickfont=dict(family=FONT_MONO, size=10, color=C_DIM),
            title_font=dict(family=FONT_UI, size=10, color=C_DIM),
        )
        fig.update_yaxes(
            showline=False,
            showgrid=True,
            zeroline=False,
            linecolor="rgba(148,163,184,0.10)",
            gridcolor=C_GRID,
            zerolinecolor=C_ZERO,
            tickfont=dict(family=FONT_MONO, size=10, color=C_DIM),
            title_font=dict(family=FONT_UI, size=10, color=C_DIM),
        )
    except Exception:
        # Styling must never break an evaluation.
        pass
    return fig


def _style_mc_figure(fig, height=420):
    """Restyle the pipeline's Monte-Carlo fan without touching its data."""
    if fig is None:
        return None

    for trace in list(getattr(fig, "data", []) or []):
        try:
            name = (getattr(trace, "name", None) or "").strip()
            line = getattr(trace, "line", None)
            if name == "Mean Forecast":
                if line is not None:
                    line.color = C_TXT
                    line.width = 3.2
            elif name == "Upper 97.5%":
                if line is not None:
                    line.color = C_ACC
                    line.dash = "dot"
                    line.width = 1.4
            elif name == "Lower 2.5%":
                if line is not None:
                    line.color = C_ACC3
                    line.dash = "dot"
                    line.width = 1.4
            else:
                # Fanned sample paths: keep them quiet. Opacity is raised to 1.0
                # so the rgba() line colour is the single source of transparency.
                if line is not None:
                    line.color = C_FAN
                    line.width = 1
                trace.showlegend = False
            trace.opacity = 1.0
        except Exception:
            continue

    try:
        title_text = ""
        try:
            title_text = fig.layout.title.text or ""
        except Exception:
            title_text = ""
        fig.update_layout(
            title=dict(
                text=html.escape(str(title_text)) if title_text else "",
                font=dict(family=FONT_UI, size=11, color=C_DIM),
                x=0.012,
                xanchor="left",
                y=0.97,
                yanchor="top",
            )
        )
    except Exception:
        pass

    # 120+ overlapping paths make an x-unified hover unusable: use "closest".
    return _style_fig(fig, height, hover="closest")


# =============================================================================
# HTML BUILDERS — every value is escaped, every failure degrades gracefully
# =============================================================================
def _kpi_html(trend="Unknown", dsp=0.0, downside_var=0.0, risk_context="",
              days=7, articles_count=0, eval_time="—", elapsed=None,
              resolved="—", placeholder=False):
    """Render the 5-card KPI deck as a ``gr.HTML`` string."""
    horizon = _as_int(days, 7)
    articles = _as_int(articles_count, 0)
    symbol_text = html.escape(str(resolved)) if resolved else "—"

    if placeholder:
        regime_pill = (
            '<span class="kpi-pill" style="color:#94a3b8;'
            'background:rgba(148,163,184,0.10);'
            'border:1px solid rgba(148,163,184,0.22)">■ STANDBY</span>'
        )
        regime_note = "awaiting first evaluation"
        dsp_text, dsp_class = "—", "neu"
        var_text = "—"
        window_value = f"{horizon}d"
        window_note = "horizon clamped to 1–30 days"
        articles_note = "no fetch yet"
        time_text = "—"
    else:
        regime_text = str(trend or "Unknown").strip() or "Unknown"
        regime_low = regime_text.lower()
        if "bull" in regime_low:
            regime_color, regime_glyph = C_OK, "▲"
        elif "bear" in regime_low:
            regime_color, regime_glyph = C_BAD, "▼"
        else:
            regime_color, regime_glyph = C_WARN, "■"
        regime_pill = (
            f'<span class="kpi-pill" style="color:{regime_color};'
            f'background:{regime_color}1f;border:1px solid {regime_color}55">'
            f'{regime_glyph} {html.escape(regime_text.upper())}</span>'
        )
        regime_note = "LSTM price-trend classification"

        dsp_value = _safe_float(dsp)
        dsp_class = "pos" if dsp_value > 0 else ("neg" if dsp_value < 0 else "neu")
        dsp_text = f"{dsp_value:+.2f}"

        var_text = f"{_safe_float(downside_var):.4f}"
        window_value = f"{horizon}d"
        window_note = "forecast horizon"
        articles_note = f"{articles} articles in window"
        time_text = html.escape(str(eval_time))

    elapsed_text = "—" if elapsed is None else f"computed in {_safe_float(elapsed):.1f}s"

    return (
        '<div class="kpi-grid">'
        '<div class="kpi-card">'
        '<div class="kpi-label">Market Regime</div>'
        f'<div style="margin-top:10px">{regime_pill}</div>'
        f'<div class="kpi-sub">{regime_note}</div>'
        '</div>'
        '<div class="kpi-card">'
        '<div class="kpi-label">Global Directional Bias</div>'
        f'<div class="kpi-value {dsp_class}">{dsp_text}</div>'
        '<div class="kpi-sub">FinBERT narrative pressure · standardised</div>'
        '</div>'
        '<div class="kpi-card">'
        '<div class="kpi-label">Downside Risk · 5% VaR</div>'
        f'<div class="kpi-value neg">{var_text}</div>'
        '<div class="kpi-chips">'
        f'<span class="kpi-chip">{html.escape(str(risk_context))}</span>'
        '</div>'
        '</div>'
        '<div class="kpi-card">'
        '<div class="kpi-label">Data Window</div>'
        f'<div class="kpi-value neu">{window_value}</div>'
        f'<div class="kpi-sub">{window_note} · {articles_note}</div>'
        '</div>'
        '<div class="kpi-card">'
        '<div class="kpi-label">Evaluation Time · UTC</div>'
        f'<div class="kpi-value-sm">{time_text}</div>'
        '<div class="kpi-chips">'
        f'<span class="kpi-chip">{elapsed_text}</span>'
        f'<span class="kpi-chip">symbol {symbol_text}</span>'
        '</div>'
        '</div>'
        '</div>'
    )


def _status_html(resolved="", tried=None, articles_count=0, days=7,
                 mode="STANDALONE"):
    """Glass status strip: mode pill, resolved symbol, probe chain, window."""
    if isinstance(tried, (list, tuple)):
        tried_list = [str(item) for item in tried if str(item).strip()]
    else:
        tried_list = []

    symbol_text = html.escape(str(resolved)) if resolved else "unresolved"
    symbol_class = "chip chip-sym" if resolved else "chip chip-warn"

    chips = [
        '<span class="chip chip-mode">'
        f'<span class="chip-k">mode</span>{html.escape(str(mode))} · no graph DB</span>',
        f'<span class="{symbol_class}"><span class="chip-k">symbol</span>'
        f'{symbol_text}</span>',
    ]

    if len(tried_list) > 1:
        chain = html.escape(" → ".join(tried_list))
        chips.append(
            f'<span class="chip"><span class="chip-k">probed:</span>{chain}</span>'
        )

    chips.append(
        '<span class="chip"><span class="chip-k">articles</span>'
        f'{_as_int(articles_count, 0)}</span>'
    )
    chips.append(
        '<span class="chip"><span class="chip-k">horizon</span>'
        f'{_as_int(days, 7)}d</span>'
    )

    return '<div class="status-strip">' + "".join(chips) + "</div>"


def _aspects_html(aspects):
    """Signed, centre-anchored decomposition bars for the top 6 aspects."""
    items = []
    try:
        if isinstance(aspects, dict):
            for name, payload in aspects.items():
                if isinstance(payload, dict):
                    score = _safe_float(payload.get("score"))
                else:
                    score = _safe_float(payload)
                label = str(name).replace("_", " ").strip()
                if not label:
                    continue
                items.append((label, score))
    except Exception:
        items = []

    items.sort(key=lambda pair: abs(pair[1]), reverse=True)
    items = items[:6]

    if not items:
        return (
            '<div class="empty-state">No aspect signal — sentiment sources '
            'returned no relevant coverage</div>'
        )

    rows = []
    for label, score in items:
        magnitude = min(abs(score), 1.0) * 50.0
        if score >= 0:
            fill_class = "aspect-fill-pos"
            value_class = "pos"
        else:
            fill_class = "aspect-fill-neg"
            value_class = "neg"
        rows.append(
            '<div class="aspect-row">'
            f'<div class="aspect-label" title="{html.escape(label)}">'
            f'{html.escape(label.title())}</div>'
            '<div class="aspect-track">'
            f'<div class="{fill_class}" style="width:{magnitude:.2f}%"></div>'
            '</div>'
            f'<div class="aspect-val {value_class}">{score:+.2f}</div>'
            '</div>'
        )

    return '<div class="aspect-panel">' + "".join(rows) + "</div>"


def _articles_html(articles, limit=8):
    """Top impact articles: badge, signed impact, weight, magnitude, time."""
    rows = []
    try:
        if isinstance(articles, (list, tuple)):
            for item in articles:
                if not isinstance(item, dict):
                    continue
                impact = _safe_float(item.get("impact"))
                # A missing weight is neutral (1.0); an explicit 0.0 is respected.
                weight = _safe_float(item.get("weight"), 1.0)
                direction = _safe_float(
                    item.get("direction"), 1.0 if impact >= 0 else -1.0
                )
                if direction == 0:
                    direction = 1.0 if impact >= 0 else -1.0
                rows.append({
                    "headline": str(item.get("headline") or "—"),
                    "source": str(item.get("source") or "—"),
                    "published": item.get("published"),
                    "impact": impact,
                    "weight": weight,
                    "magnitude": _safe_float(item.get("magnitude")),
                    "direction": direction,
                    "entities": _entity_list(item.get("entities")),
                    "influence": abs(impact) * weight,
                })
    except Exception:
        rows = []

    if not rows:
        return (
            '<div class="empty-state">No relevant articles fetched for this '
            'window — showing price/risk analytics only</div>'
        )

    rows.sort(key=lambda row: row["influence"], reverse=True)
    rows = rows[:max(1, _as_int(limit, 8))]

    body = []
    for row in rows:
        positive = row["direction"] > 0
        badge_class = "pos" if positive else "neg"
        badge_glyph = "+" if positive else "−"
        headline = row["headline"].strip() or "—"
        if len(headline) > 150:
            headline = headline[:147].rstrip() + "…"

        entities = row["entities"][:3]
        entity_chips = "".join(
            f'<span class="ent-chip">{html.escape(entity[:28])}</span>'
            for entity in entities
        )

        body.append(
            '<div class="art-row">'
            '<div class="art-num">'
            f'<span class="dir-badge {badge_class}">{badge_glyph}</span>'
            f'<span class="art-impact {badge_class}">{row["impact"]:+.2f}</span>'
            '</div>'
            '<div class="art-body">'
            f'<div class="art-headline">{html.escape(headline)}</div>'
            '<div class="art-meta">'
            f'<span class="src-chip">{html.escape(row["source"][:26])}</span>'
            f'{entity_chips}'
            '<span class="art-num-meta">w '
            f'{row["weight"]:.2f} · m {row["magnitude"]:.2f}</span>'
            f'<span class="art-time">{_relative_time(row["published"])}</span>'
            '</div>'
            '</div>'
            '</div>'
        )

    return (
        '<div class="art-table">'
        '<div class="art-head">Top impact articles · ranked by |impact| × weight</div>'
        + "".join(body)
        + '</div>'
    )


def _error_html(message):
    """Red-tinted glass banner; an empty message renders nothing at all."""
    if not message:
        return ""
    return (
        '<div class="error-banner">'
        '<span class="error-icon">⚠</span>'
        f'<span>{html.escape(str(message))}</span>'
        '</div>'
    )


def _empty_fig(title: str):
    fig = go.Figure()
    try:
        fig.update_layout(
            template=None, paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=C_TXT, family=FONT_UI, size=11),
            title=dict(text=str(title), font=dict(size=12, color=C_MUTED)),
            margin=dict(l=8, r=8, t=36, b=8),
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False),
        )
    except Exception:
        pass
    return fig


def _busy_placeholders(reason: str):
    note = html.escape(str(reason))
    k = ("<div class='panel'><div class='panel-title'>KPI Deck <span>· busy</span></div>"
         f"<div class='empty-state'>BUSY — {note}. Wait, then click once.</div></div>")
    s = ("<div class='status-strip'><span class='chip warn'>BUSY</span>"
         "<span class='chip'>single-flight active</span></div>")
    h = _empty_fig("Busy — previous run in flight")
    m = _empty_fig("Busy — previous run in flight")
    f = _empty_fig("Busy — previous run in flight")
    asp = ("<div class='panel aspect-panel'><div class='panel-title'>Aspect Decomposition "
           "<span>· busy</span></div><div class='empty-state'>Previous run in flight.</div></div>")
    art = ("<div class='panel'><div class='panel-title'>Top Impact Articles <span>· busy</span>"
           "</div><div class='empty-state'>Previous run in flight.</div></div>")
    e = f"<div class='error-banner'>BUSY: {note} — wait, then click once.</div>"
    return (k, s, h, m, f, asp, art, e)


def _timeout_placeholders(elapsed: float):
    k = ("<div class='panel'><div class='panel-title'>KPI Deck <span>· timeout</span></div>"
         f"<div class='empty-state'>Click budget ({CLICK_BUDGET_SECONDS:.0f}s) exceeded "
         f"after {elapsed:.1f}s. Try once more.</div></div>")
    s = ("<div class='status-strip'><span class='chip warn'>TIMEOUT</span>"
         f"<span class='chip'>budget {CLICK_BUDGET_SECONDS:.0f}s</span>"
         f"<span class='chip'>elapsed {elapsed:.1f}s</span></div>")
    h = _empty_fig("Timeout — no price path")
    m = _empty_fig("Timeout — no risk envelope")
    f = _empty_fig("Timeout — no forward path")
    asp = ("<div class='panel aspect-panel'><div class='panel-title'>Aspect Decomposition "
           "<span>· timeout</span></div><div class='empty-state'>No narrative pressure.</div></div>")
    art = ("<div class='panel'><div class='panel-title'>Top Impact Articles <span>· timeout</span>"
           "</div><div class='empty-state'>No articles this click.</div></div>")
    e = (f"<div class='error-banner'>Click timeout after {elapsed:.1f}s (budget "
         f"{CLICK_BUDGET_SECONDS:.0f}s). Wait ~30s, then click once.</div>")
    return (k, s, h, m, f, asp, art, e)


def _as_list(value):
    """Coerce list/tuple/ndarray/Series into a plain list; junk becomes []."""
    try:
        if value is None:
            return []
        return list(value)
    except Exception:
        return []


# =============================================================================
# CORE CALLBACK (frozen contract: docs/DEMO_CONTRACT.md section 3)
# =============================================================================
def _run_finwise_uncapped(raw_symbol, horizon, t0, compute_fn=None, resolve_fn=None):
    """Evaluate one ticker and return the 8 dashboard outputs, never raising.

    Output order: kpi_html, status_html, hist_plot, mc_plot, fc_plot,
    aspects_html, articles_html, error_html.
    """
    started = time.perf_counter()

    def _mark(stage: str) -> None:
        """Log a per-stage timing line so a slow click is diagnosable."""
        _perf_log.info("[FW] %s", f"{stage:<28} {_elapsed(started):6.2f}s")

    try:
        horizon = min(max(int(horizon), 1), 30)
    except Exception:
        horizon = 7

    raw_symbol = "" if raw_symbol is None else str(raw_symbol).strip()
    if compute_fn is None:
        compute_fn = globals().get("compute_state")
    if resolve_fn is None:
        resolve_fn = globals().get("resolve_symbol")

    # ------------- BLANK TICKER: banner + placeholders, no pipeline call ------
    if not raw_symbol:
        return (
            _kpi_html(days=horizon, placeholder=True,
                      risk_context="awaiting first evaluation"),
            _status_html("", [], 0, horizon),
            None,
            None,
            None,
            _aspects_html({}),
            _articles_html([]),
            _error_html(BLANK_TICKER_MESSAGE),
        )

    try:
        resolved, tried = (resolve_fn or globals()["resolve_symbol"])(raw_symbol)
        _mark(f"resolve_symbol({raw_symbol})")
        if not resolved:
            resolved = raw_symbol

        # Probe chain exhausted (bare + .NS + .BO all unknown): fail fast with an
        # actionable message instead of waiting on three Yahoo retry cycles.
        if len(tried) > 1 and resolved == tried[0]:
            return (
                _kpi_html(days=horizon, placeholder=True, articles_count=0,
                          risk_context=f"unresolved: {raw_symbol}",
                          elapsed=_elapsed(started)),
                _status_html(resolved, tried, 0, horizon),
                None,
                None,
                None,
                _aspects_html({}),
                _articles_html([]),
                _error_html(
                    f"Could not resolve '{raw_symbol}' on Yahoo Finance "
                    f"(tried {' → '.join(tried)}). Check the ticker symbol, or use "
                    f"the explicit exchange suffix for Indian listings "
                    f"(e.g. RELIANCE.NS, TCS.NS, HDFCBANK.BO)."
                ),
            )

        # Third arg None -> skip the Neo4j/KG snapshot write (standalone mode).
        state = (compute_fn or globals()["compute_state"])(resolved, horizon, None)
        _mark("compute_state (both engines)")
        if not isinstance(state, dict):
            state = {}

        price = state.get("price") or {}
        trend = str(price.get("trend") or "Unknown")

        sentiment = state.get("sentiment") or {}
        dsp = _safe_float(sentiment.get("global_score"))
        aspects = sentiment.get("aspects") or {}

        history = state.get("history") or {}
        forecast = state.get("forecast") or {}
        monte_carlo = state.get("monte_carlo") or {}
        downside_var = _safe_float(monte_carlo.get("downside_var"))

        articles = state.get("articles") or []
        if not isinstance(articles, (list, tuple)):
            articles = []

        eval_symbol = str(state.get("symbol") or resolved or raw_symbol)

        # ---------------- OBSERVED vs MODEL FIT ----------------
        h_dates = _as_list(history.get("dates"))
        h_actual = _as_list(history.get("actual"))
        h_pred = _as_list(history.get("predicted"))

        fig_hist = go.Figure()
        overlap = min(len(h_actual), len(h_pred))
        if overlap >= 2:
            if len(h_dates) >= overlap:
                hist_x = h_dates[-overlap:]
            else:
                hist_x = list(range(overlap))
            fig_hist.add_trace(go.Scatter(
                x=hist_x, y=h_actual[-overlap:], mode="lines",
                name="Observed", line=dict(color=C_TXT, width=1.8),
                hovertemplate="%{y:.2f}<extra>observed</extra>",
            ))
            fig_hist.add_trace(go.Scatter(
                x=hist_x, y=h_pred[-overlap:], mode="lines",
                name="Model fit", line=dict(color=C_ACC2, width=1.6, dash="dot"),
                hovertemplate="%{y:.2f}<extra>model</extra>",
            ))
            fig_hist.update_xaxes(title=dict(text="Session date"))
            fig_hist.update_yaxes(title=dict(text="Price"))
        else:
            fig_hist.add_annotation(
                text="insufficient history for a model-fit overlay",
                showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5,
                font=dict(family=FONT_UI, size=12, color=C_DIM),
            )
            fig_hist.update_layout(showlegend=False)
        fig_hist = _style_fig(fig_hist, 260)

        # ---------------- MONTE CARLO RISK ENVELOPE (PRIMARY) ----------------
        # The pipeline figure is restyled trace-by-trace; its data is untouched.
        # When the pipeline yields no figure we pass None straight to gr.Plot.
        fig_mc = _style_mc_figure(monte_carlo.get("figure"), 420)

        # ---------------- CONDITIONAL FORWARD PATH ----------------
        f_dates = _as_list(forecast.get("dates"))
        f_pred = _as_list(forecast.get("predicted"))

        fig_fc = go.Figure()
        points = min(len(f_dates), len(f_pred))
        if points >= 1:
            fig_fc.add_trace(go.Scatter(
                x=f_dates[:points], y=f_pred[:points], mode="lines+markers",
                name="Forward path", line=dict(color=C_ACC, width=2.2),
                marker=dict(size=5, color=C_ACC, line=dict(width=0)),
                hovertemplate="%{y:.2f}<extra>forecast</extra>",
            ))
            fig_fc.update_xaxes(title=dict(text="Trading day ahead"))
            fig_fc.update_yaxes(title=dict(text="Price (projected)"))
        else:
            fig_fc.add_annotation(
                text="no forward projection available for this window",
                showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5,
                font=dict(family=FONT_UI, size=12, color=C_DIM),
            )
            fig_fc.update_layout(showlegend=False)
        fig_fc = _style_fig(fig_fc, 300)

        # ---------------- RISK CONTEXT + TIMING ----------------
        var_percentile = downside_var_percentile(h_actual, downside_var)
        if var_percentile is not None:
            risk_context = f"{var_percentile}th percentile vs own 20d VaR"
        else:
            risk_context = "insufficient history for percentile context"

        elapsed = _elapsed(started)
        eval_time = datetime.now(timezone.utc).strftime("%Y-%m-%d · %H:%M:%S UTC")

        kpi_html = _kpi_html(
            trend=trend,
            dsp=dsp,
            downside_var=downside_var,
            risk_context=risk_context,
            days=horizon,
            articles_count=len(articles),
            eval_time=eval_time,
            elapsed=elapsed,
            resolved=eval_symbol,
        )
        status_html = _status_html(
            eval_symbol, tried, len(articles), horizon, mode="STANDALONE"
        )
        aspects_html = _aspects_html(aspects)
        articles_html = _articles_html(articles)
        _mark("render + return")

        return (
            kpi_html,       # 1
            status_html,    # 2
            fig_hist,       # 3
            fig_mc,         # 4 (dominant chart)
            fig_fc,         # 5
            aspects_html,   # 6
            articles_html,  # 7
            "",             # 8
        )

    except Exception as exc:
        # Absolutely nothing escapes this handler: the dashboard always answers.
        elapsed = _elapsed(started)
        message = (
            f"Evaluation failed: {type(exc).__name__}: {exc} · after {elapsed:.1f}s"
        )
        return (
            _kpi_html(days=horizon, placeholder=True, articles_count=0,
                      risk_context="evaluation aborted", elapsed=elapsed),
            _status_html("", [], 0, horizon),
            None,
            None,
            None,
            _aspects_html({}),
            _articles_html([]),
            _error_html(message),
        )

# =============================================================================
# PUBLIC CALLBACK (frozen contract: docs/DEMO_CONTRACT.md section 3)
# Sub-60s envelope: cache -> single-flight -> hard click budget -> pipeline.
# =============================================================================
def run_finwise(symbol: str, days: int):
    t0 = time.monotonic()
    raw = "" if symbol is None else str(symbol).strip()
    try:
        horizon = min(max(int(days), 1), 30)
    except Exception:
        horizon = 7
    key = (raw.upper(), horizon)
    try:
        with _CACHE_LOCK:
            hit = _RESULT_CACHE.get(key)
            if hit is not None:
                ts, cached = hit
                if time.monotonic() - ts <= _RESULT_CACHE_TTL_SECONDS:
                    _perf_log.info("[FW] run_finwise cache-hit %s/%dd (%.2fs)",
                                   raw or "empty", horizon, time.monotonic() - t0)
                    return cached
                del _RESULT_CACHE[key]
    except Exception:
        pass
    if not _CLICK_LOCK.acquire(blocking=False):
        _perf_log.info("[FW] run_finwise single-flight BUSY %s/%dd",
                        raw or "empty", horizon)
        return _busy_placeholders(
            "evaluation for '%s' already in progress" % (raw or "ticker"))
    try:
        with _ORPHAN_LOCK:
            orphan = _ORPHAN_RUNNING
        if orphan:
            return _busy_placeholders(
                "previous evaluation still releasing - try again in ~30s")
        import sys as _sys
        mod = _sys.modules.get(__name__)
        _cf = getattr(mod, "compute_state", None)
        _rf = getattr(mod, "resolve_symbol", None)
        pool = ThreadPoolExecutor(max_workers=1)
        try:
            fut = pool.submit(_run_finwise_uncapped, raw, horizon, t0, _cf, _rf)
            try:
                result = fut.result(timeout=CLICK_BUDGET_SECONDS)
            except _FuturesTimeout:
                elapsed = time.monotonic() - t0
                with _ORPHAN_LOCK:
                    globals()["_ORPHAN_RUNNING"] = True
                def _reap(_fut=fut, _pool=pool):
                    try:
                        _fut.result(timeout=180)
                    except Exception:
                        pass
                    finally:
                        with _ORPHAN_LOCK:
                            globals()["_ORPHAN_RUNNING"] = False
                        try:
                            _pool.shutdown(wait=False, cancel_futures=True)
                        except Exception:
                            pass
                import threading as _th
                _th.Thread(target=_reap, daemon=True).start()
                _perf_log.warning("[FW] run_finwise CLICK TIMEOUT %s/%dd after %.1fs",
                                  raw or "empty", horizon, elapsed)
                return _timeout_placeholders(elapsed)
            try:
                with _CACHE_LOCK:
                    _RESULT_CACHE[key] = (time.monotonic(), result)
            except Exception:
                pass
            return result
        finally:
            try:
                if fut.done():
                    pool.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
    finally:
        try:
            _CLICK_LOCK.release()
        except Exception:
            pass




# =============================================================================
# STATIC MARKUP — hero, quick-pick hints, panel captions, footer, ready states
# =============================================================================
HERO_HTML = """
<div class="hero-row">
  <div class="hero-left">
    <div class="wordmark">◆ FINWISE · MARKET INTELLIGENCE TERMINAL</div>
    <div class="hero-sub">Knowledge-driven price forecasting · Heston Monte-Carlo risk · FinBERT narrative pressure</div>
  </div>
  <div class="hero-spacer"></div>
  <div class="hero-chips">
    <span class="live-pill"><span class="live-dot"></span>LIVE · PIPELINE READY</span>
    <span class="chip">MODELS: LSTM + MC + FinBERT</span>
    <span class="chip chip-mode">MODE: STANDALONE</span>
  </div>
</div>
"""

QUICK_PICK_HTML = """
<div class="chip-row">
  <span class="chip chip-k">quick pick</span>
  <span class="chip">AAPL · NVDA · MSFT · TSLA · AMZN · META</span>
  <span class="chip">RELIANCE.NS · TCS.NS · HDFCBANK.NS · INFY.NS · SBIN.NS</span>
  <span class="chip-note">US tickers work as typed; Indian tickers auto-resolve (RELIANCE → RELIANCE.NS)</span>
</div>
"""

PANEL_MC_TITLE = (
    '<div class="panel-title">MONTE-CARLO RISK ENVELOPE '
    '<span>· Heston-like · primary</span></div>'
)
PANEL_HIST_TITLE = (
    '<div class="panel-title">OBSERVED VS MODEL FIT '
    '<span>· LSTM back-fit</span></div>'
)
PANEL_FC_TITLE = (
    '<div class="panel-title">CONDITIONAL FORWARD PATH '
    '<span>· projected path</span></div>'
)
PANEL_ASPECTS_TITLE = (
    '<div class="panel-title">DIRECTIONAL BIAS DECOMPOSITION '
    '<span>· top 6 aspects</span></div>'
)
PANEL_ARTICLES_TITLE = (
    '<div class="panel-title">INFORMATION PRESSURE '
    '<span>· impact-weighted</span></div>'
)

FOOTER_HTML = (
    '<div class="foot">FINWISE CORE v1.0 · LSTM + Heston MC + FinBERT · '
    'Standalone mode (no graph DB) · '
    + datetime.now().strftime("%d %b %Y")
    + '</div>'
)

READY_STATUS = (
    '<div class="status-strip">'
    '<span class="chip chip-mode"><span class="chip-k">mode</span>STANDBY</span>'
    '<span class="chip"><span class="chip-k">symbol</span>awaiting ticker</span>'
    '<span class="chip"><span class="chip-k">models</span>loaded lazily on first run</span>'
    '<span class="chip"><span class="chip-k">horizon</span>1–30d</span>'
    '</div>'
)

READY_KPI = _kpi_html(days=7, placeholder=True,
                      risk_context="awaiting first evaluation")
READY_ASPECTS = _aspects_html({})
READY_ARTICLES = _articles_html([])


# =============================================================================
# CUSTOM CSS — dark luxury fintech terminal
# =============================================================================
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root{
  --bg:#05070c;
  --panel:rgba(15,23,42,0.55);
  --panel-brd:rgba(148,163,184,0.14);
  --txt:#e2e8f0;
  --muted:#94a3b8;
  --dim:#64748b;
  --acc:#22d3ee;
  --acc2:#8b5cf6;
  --acc3:#ec4899;
  --ok:#34d399;
  --bad:#fb7185;
  --warn:#fbbf24;
  --mono:'JetBrains Mono','SF Mono',Menlo,Consolas,monospace;
}

html,body{background:var(--bg) !important}

/* ---------- shell ---------- */
.gradio-container{
  max-width:100% !important;
  width:100% !important;
  background:
    radial-gradient(circle at 12% -12%, rgba(34,211,238,0.10), transparent 45%),
    radial-gradient(circle at 88% 4%, rgba(139,92,246,0.12), transparent 42%),
    #05070c !important;
  font-family:'Inter',system-ui,-apple-system,'Segoe UI',sans-serif !important;
  color:var(--txt);
  padding:18px 22px 40px 22px !important;
}
.gradio-container .main,
.gradio-container .wrap,
.gradio-container .gr-block,
.gradio-container .gr-form,
.gradio-container .block,
.gradio-container .form{background:transparent !important}
.gr-plot{background:transparent !important}
.modebar{display:none !important}
footer{display:none !important}
.gradio-container ::-webkit-scrollbar{width:8px;height:8px}
.gradio-container ::-webkit-scrollbar-track{background:transparent}
.gradio-container ::-webkit-scrollbar-thumb{background:rgba(148,163,184,0.25);border-radius:8px}
.gradio-container ::-webkit-scrollbar-thumb:hover{background:rgba(148,163,184,0.40)}
.gradio-container .prose{color:inherit !important}
.prose p{margin:0 !important}

/* ---------- hero ---------- */
.hero{
  position:relative;
  overflow:hidden;
  border-radius:18px;
  padding:20px 26px 18px 26px;
  margin-bottom:14px;
  background:linear-gradient(180deg,rgba(15,23,42,0.72),rgba(5,7,12,0.35));
  border:1px solid var(--panel-brd);
  backdrop-filter:blur(14px);
  -webkit-backdrop-filter:blur(14px);
  box-shadow:0 18px 44px rgba(0,0,0,0.45);
}
.hero::before{
  content:"";
  position:absolute;top:0;left:0;right:0;height:2px;
  background:linear-gradient(90deg,#22d3ee,#8b5cf6,#ec4899,#22d3ee);
  background-size:300% 100%;
  animation:fw-border 9s linear infinite;
}
@keyframes fw-border{
  0%{background-position:0% 50%}
  100%{background-position:300% 50%}
}
.hero-row{display:flex;align-items:center;gap:16px;flex-wrap:wrap}
.hero-left{min-width:260px}
.hero-spacer{flex:1 1 auto}
.wordmark{
  font-size:22px;font-weight:800;letter-spacing:2.2px;line-height:1.25;
  background:linear-gradient(95deg,#22d3ee 0%,#8b5cf6 52%,#ec4899 100%);
  -webkit-background-clip:text;background-clip:text;
  -webkit-text-fill-color:transparent;color:transparent;
}
.hero-sub{font-size:11.5px;color:var(--muted);letter-spacing:0.4px;margin-top:6px}
.hero-chips{display:flex;gap:8px;flex-wrap:wrap;justify-content:flex-end;align-items:center}

/* ---------- pills + chips ---------- */
.live-pill{
  display:inline-flex;align-items:center;gap:7px;
  padding:5px 12px;border-radius:999px;
  background:rgba(52,211,153,0.10);
  border:1px solid rgba(52,211,153,0.32);
  color:var(--ok);font-size:9.5px;font-weight:700;
  letter-spacing:1.6px;text-transform:uppercase;white-space:nowrap;
}
.live-dot{
  width:7px;height:7px;border-radius:50%;background:var(--ok);
  animation:fw-pulse 1.8s ease-in-out infinite;
}
@keyframes fw-pulse{
  0%,100%{opacity:1;box-shadow:0 0 0 0 rgba(52,211,153,0.55)}
  50%{opacity:0.45;box-shadow:0 0 0 6px rgba(52,211,153,0)}
}
.chip{
  display:inline-flex;align-items:center;gap:6px;
  padding:4px 11px;border-radius:999px;white-space:nowrap;
  background:rgba(148,163,184,0.09);
  border:1px solid rgba(148,163,184,0.18);
  color:var(--muted);font-size:10.5px;font-weight:500;letter-spacing:0.5px;
}
.chip-k{
  color:var(--dim);font-size:9px;font-weight:700;
  letter-spacing:1.2px;text-transform:uppercase;
}
.chip-mode{background:rgba(139,92,246,0.14);border-color:rgba(139,92,246,0.38);color:#c4b5fd}
.chip-sym{background:rgba(34,211,238,0.12);border-color:rgba(34,211,238,0.36);color:#67e8f9;font-family:var(--mono)}
.chip-warn{background:rgba(251,191,36,0.12);border-color:rgba(251,191,36,0.34);color:#fcd34d}
.chip-ok{background:rgba(52,211,153,0.12);border-color:rgba(52,211,153,0.34);color:#6ee7b7}
.status-strip{
  display:flex;flex-wrap:wrap;gap:8px;align-items:center;
  padding:10px 14px;margin-bottom:12px;border-radius:14px;
  background:var(--panel);border:1px solid var(--panel-brd);
  backdrop-filter:blur(12px);-webkit-backdrop-filter:blur(12px);
}
.chip-row{display:flex;flex-wrap:wrap;gap:8px;align-items:center;padding:10px 2px 12px 2px}
.chip-row span.chip{pointer-events:none;user-select:none;opacity:0.92}
.chip-note{font-size:10.5px;color:var(--dim);letter-spacing:0.3px;line-height:1.5}
"""

CUSTOM_CSS += """
/* ---------- KPI deck ---------- */
.kpi-grid{
  display:grid;
  grid-template-columns:repeat(auto-fit,minmax(180px,1fr));
  gap:12px;
  margin-bottom:14px;
}
.kpi-card{
  position:relative;overflow:hidden;
  padding:14px 15px 13px 15px;border-radius:15px;
  background:linear-gradient(180deg,rgba(15,23,42,0.66),rgba(5,7,12,0.42));
  border:1px solid var(--panel-brd);
  backdrop-filter:blur(12px);-webkit-backdrop-filter:blur(12px);
  transition:transform 220ms ease,border-color 220ms ease,box-shadow 220ms ease;
}
.kpi-card:hover{
  transform:translateY(-3px);
  border-color:rgba(34,211,238,0.35);
  box-shadow:0 14px 30px rgba(8,145,178,0.18);
}
.kpi-card::after{
  content:"";
  position:absolute;inset:0;pointer-events:none;
  background:radial-gradient(circle at 100% 0%,rgba(34,211,238,0.10),transparent 55%);
}
.kpi-label{
  font-size:9px;font-weight:700;letter-spacing:1.6px;
  text-transform:uppercase;color:var(--dim);
}
.kpi-value{
  margin-top:9px;font-family:var(--mono);font-size:30px;font-weight:600;
  line-height:1;letter-spacing:-0.5px;
  background:linear-gradient(120deg,#e2e8f0,#94a3b8);
  -webkit-background-clip:text;background-clip:text;
  -webkit-text-fill-color:transparent;color:transparent;
}
.kpi-value.pos{background:linear-gradient(120deg,#6ee7b7,#34d399)}
.kpi-value.neg{background:linear-gradient(120deg,#fda4af,#fb7185)}
.kpi-value.neu{background:linear-gradient(120deg,#e2e8f0,#94a3b8)}
.kpi-value-sm{
  margin-top:10px;font-family:var(--mono);font-size:13px;
  color:var(--txt);letter-spacing:0.3px;
}
.kpi-sub{margin-top:8px;font-size:10.5px;color:var(--muted);line-height:1.5}
.kpi-chips{margin-top:9px;display:flex;flex-wrap:wrap;gap:6px}
.kpi-chip{
  display:inline-flex;align-items:center;gap:5px;
  padding:3px 9px;border-radius:999px;
  background:rgba(148,163,184,0.10);
  border:1px solid rgba(148,163,184,0.18);
  color:var(--muted);font-size:9.5px;font-weight:500;letter-spacing:0.6px;
}
.kpi-pill{
  display:inline-flex;align-items:center;gap:7px;
  padding:6px 13px;border-radius:999px;
  font-size:12.5px;font-weight:700;letter-spacing:1.1px;text-transform:uppercase;
}

/* ---------- panels ---------- */
.panel{
  border-radius:16px;
  padding:14px 15px 10px 15px;
  margin-bottom:14px;
  background:var(--panel);
  border:1px solid var(--panel-brd);
  backdrop-filter:blur(12px);-webkit-backdrop-filter:blur(12px);
}
.panel-title{
  font-size:10px;font-weight:700;letter-spacing:2px;
  text-transform:uppercase;color:var(--muted);margin-bottom:10px;
}
.panel-title span{color:var(--dim);font-weight:500;letter-spacing:1.4px}
.panel-body{font-size:12px;color:var(--txt)}
.aspect-panel{display:flex;flex-direction:column;padding-bottom:6px}
.empty-state{
  padding:18px 14px;border-radius:12px;text-align:center;
  background:rgba(148,163,184,0.05);
  border:1px dashed rgba(148,163,184,0.20);
  color:var(--dim);font-size:11.5px;line-height:1.6;letter-spacing:0.2px;
}

/* ---------- aspect decomposition bars ---------- */
.aspect-row{
  display:flex;align-items:center;gap:10px;
  padding:7px 2px;border-bottom:1px solid rgba(148,163,184,0.07);
}
.aspect-row:last-child{border-bottom:none}
.aspect-label{
  flex:0 0 118px;font-size:12px;color:var(--txt);letter-spacing:0.2px;
  overflow:hidden;text-overflow:ellipsis;white-space:nowrap;
}
.aspect-track{
  position:relative;flex:1 1 auto;height:8px;border-radius:999px;
  background:rgba(148,163,184,0.10);overflow:hidden;
}
.aspect-track::after{
  content:"";
  position:absolute;left:50%;top:0;bottom:0;width:1px;
  background:rgba(148,163,184,0.30);
}
.aspect-fill-pos{
  position:absolute;left:50%;top:0;bottom:0;
  border-radius:0 999px 999px 0;
  background:linear-gradient(90deg,rgba(52,211,153,0.35),#34d399);
}
.aspect-fill-neg{
  position:absolute;right:50%;top:0;bottom:0;
  border-radius:999px 0 0 999px;
  background:linear-gradient(270deg,rgba(251,113,133,0.35),#fb7185);
}
.aspect-val{
  flex:0 0 56px;text-align:right;font-family:var(--mono);
  font-size:11.5px;color:var(--muted);
}
.aspect-val.pos{color:var(--ok)}
.aspect-val.neg{color:var(--bad)}
"""

CUSTOM_CSS += """
/* ---------- information pressure (articles) ---------- */
.art-table{display:flex;flex-direction:column;padding-bottom:4px}
.art-head{
  font-size:10px;font-weight:700;letter-spacing:1.6px;text-transform:uppercase;
  color:var(--dim);padding:4px 2px 8px 2px;
  border-bottom:1px solid rgba(148,163,184,0.12);
}
.art-row{
  display:grid;grid-template-columns:52px 1fr;gap:10px;
  padding:9px 4px;border-bottom:1px solid rgba(148,163,184,0.07);
  transition:background 160ms ease;
}
.art-row:nth-child(odd){background:transparent}
.art-row:nth-child(even){background:rgba(148,163,184,0.035)}
.art-row:hover{background:rgba(34,211,238,0.06)}
.art-num{
  display:flex;flex-direction:column;align-items:center;gap:4px;
  font-family:var(--mono);
}
.dir-badge{
  display:inline-flex;align-items:center;justify-content:center;
  width:22px;height:22px;border-radius:7px;font-size:13px;font-weight:700;
}
.dir-badge.pos{background:rgba(52,211,153,0.16);border:1px solid rgba(52,211,153,0.40);color:var(--ok)}
.dir-badge.neg{background:rgba(251,113,133,0.16);border:1px solid rgba(251,113,133,0.40);color:var(--bad)}
.art-impact{font-size:11px;font-weight:600}
.art-impact.pos{color:var(--ok)}
.art-impact.neg{color:var(--bad)}
.art-body{min-width:0}
.art-headline{font-size:12px;color:var(--txt);line-height:1.45;letter-spacing:0.1px}
.art-meta{display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin-top:6px}
.src-chip{
  padding:2px 8px;border-radius:999px;
  background:rgba(139,92,246,0.14);border:1px solid rgba(139,92,246,0.32);
  color:#c4b5fd;font-size:9.5px;font-weight:600;letter-spacing:0.6px;text-transform:uppercase;
}
.ent-chip{
  padding:2px 8px;border-radius:999px;
  background:rgba(34,211,238,0.10);border:1px solid rgba(34,211,238,0.26);
  color:#67e8f9;font-size:9.5px;letter-spacing:0.4px;
}
.art-num-meta{font-family:var(--mono);font-size:9.5px;color:var(--dim);letter-spacing:0.3px}
.art-time{font-family:var(--mono);font-size:9.5px;color:var(--dim)}

/* ---------- control deck ---------- */
#control-deck{gap:12px !important;align-items:stretch !important;margin-bottom:2px}
#ticker-input input, #ticker-input textarea{
  background:rgba(15,23,42,0.72) !important;
  border:1px solid var(--panel-brd) !important;
  border-radius:11px !important;
  color:var(--txt) !important;
  font-family:var(--mono) !important;
  font-size:14px !important;
  letter-spacing:1.2px !important;
  padding:11px 13px !important;
  transition:border-color 180ms ease, box-shadow 180ms ease !important;
}
#ticker-input input:focus, #ticker-input textarea:focus{
  outline:none !important;
  border-color:rgba(34,211,238,0.55) !important;
  box-shadow:0 0 0 3px rgba(34,211,238,0.18) !important;
}
#horizon-slider input[type=range]{accent-color:#22d3ee}
#horizon-slider input[type=number]{
  background:rgba(15,23,42,0.72) !important;
  border:1px solid var(--panel-brd) !important;
  border-radius:9px !important;
  color:var(--txt) !important;
  font-family:var(--mono) !important;
  font-size:13px !important;
}
#horizon-slider input[type=number]:focus{
  outline:none !important;
  border-color:rgba(34,211,238,0.5) !important;
  box-shadow:0 0 0 3px rgba(34,211,238,0.18) !important;
}
#ticker-input label span, #horizon-slider label span{
  font-size:10px !important;letter-spacing:1.5px !important;
  text-transform:uppercase !important;color:var(--dim) !important;
}
#run-btn{
  background:linear-gradient(135deg,#0891b2,#22d3ee) !important;
  border:none !important;
  color:#f8fafc !important;
  font-weight:800 !important;
  font-size:11.5px !important;
  letter-spacing:1.6px !important;
  border-radius:12px !important;
  padding:12px 18px !important;
  text-shadow:0 1px 2px rgba(3,20,26,0.55);
  box-shadow:0 10px 26px rgba(34,211,238,0.22) !important;
  transition:transform 180ms ease, box-shadow 180ms ease, filter 180ms ease !important;
}
#run-btn:hover{
  transform:translateY(-2px) !important;
  box-shadow:0 14px 32px rgba(34,211,238,0.34) !important;
  filter:brightness(1.06);
}
#run-btn:active{transform:translateY(0) !important}
#reset-btn{
  background:rgba(148,163,184,0.08) !important;
  border:1px solid var(--panel-brd) !important;
  color:var(--muted) !important;
  font-weight:600 !important;
  font-size:11px !important;
  letter-spacing:1.2px !important;
  border-radius:12px !important;
  padding:12px 14px !important;
}
#reset-btn:hover{background:rgba(148,163,184,0.14) !important;color:var(--txt) !important}

/* ---------- grid, error banner, footer ---------- */
#main-grid{gap:16px !important}
#error-slot{padding:0 !important;margin:0 !important}
.error-banner{
  display:flex;align-items:flex-start;gap:10px;
  padding:13px 16px;margin-bottom:12px;border-radius:13px;
  background:linear-gradient(180deg,rgba(251,113,133,0.14),rgba(251,113,133,0.05));
  border:1px solid rgba(251,113,133,0.38);
  color:#fecdd3;font-size:12px;line-height:1.55;
  backdrop-filter:blur(10px);-webkit-backdrop-filter:blur(10px);
}
.error-icon{font-size:14px;line-height:1.25}
.foot{
  padding:16px 8px 4px 8px;margin-top:6px;text-align:center;
  font-size:10.5px;color:var(--dim);letter-spacing:1.2px;text-transform:uppercase;
}

/* ---------- responsive ---------- */
@media (max-width:900px){
  .gradio-container{padding:12px 12px 28px 12px !important}
  .hero{padding:16px 16px 14px 16px;border-radius:15px}
  .wordmark{font-size:17px;letter-spacing:1.4px}
  .hero-left{min-width:0}
  .hero-chips{justify-content:flex-start}
  .kpi-grid{grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:9px}
  .kpi-value{font-size:24px}
  #main-grid{gap:10px !important}
  #control-deck{gap:9px !important}
  .aspect-label{flex:0 0 92px;font-size:11px}
  .art-row{grid-template-columns:44px 1fr;gap:8px}
}
"""


# =============================================================================
# UI — FINWISE MARKET INTELLIGENCE TERMINAL
# =============================================================================
with gr.Blocks(
    theme=gr.themes.Base(
        primary_hue="cyan",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter"),
        font_mono=gr.themes.GoogleFont("JetBrains Mono"),
    ),
    css=CUSTOM_CSS,
    title="FinWise · Market Intelligence Terminal",
) as demo:

    gr.HTML(HERO_HTML, elem_id="hero", elem_classes=["hero"])

    status_strip = gr.HTML(READY_STATUS, elem_id="status-strip")
    kpi_deck = gr.HTML(READY_KPI, elem_id="kpi-deck")
    error_slot = gr.HTML("", elem_id="error-slot")

    with gr.Row(elem_id="control-deck"):
        ticker = gr.Textbox(
            value="AAPL",
            label="Ticker",
            placeholder="AAPL / RELIANCE.NS",
            elem_id="ticker-input",
            scale=4,
        )
        horizon = gr.Slider(
            minimum=1,
            maximum=30,
            value=7,
            step=1,
            label="Horizon (days)",
            elem_id="horizon-slider",
            scale=3,
        )
        run_btn = gr.Button(
            "▶ RUN EVALUATION", elem_id="run-btn", variant="primary", scale=2
        )
        reset_btn = gr.Button("Reset", elem_id="reset-btn", scale=1)

    gr.HTML(QUICK_PICK_HTML, elem_id="quick-pick", elem_classes=["chip-row"])

    with gr.Row(elem_id="main-grid"):
        with gr.Column(scale=3, elem_id="col-plots"):
            with gr.Column(elem_classes=["panel"], elem_id="panel-mc"):
                gr.HTML(PANEL_MC_TITLE)
                mc_plot = gr.Plot(elem_id="mc-plot")
            with gr.Column(elem_classes=["panel"], elem_id="panel-hist"):
                gr.HTML(PANEL_HIST_TITLE)
                hist_plot = gr.Plot(elem_id="hist-plot")
            with gr.Column(elem_classes=["panel"], elem_id="panel-fc"):
                gr.HTML(PANEL_FC_TITLE)
                fc_plot = gr.Plot(elem_id="fc-plot")

        with gr.Column(scale=2, elem_id="col-signals"):
            with gr.Column(elem_classes=["panel"], elem_id="panel-aspects"):
                gr.HTML(PANEL_ASPECTS_TITLE)
                aspects_panel = gr.HTML(READY_ASPECTS, elem_id="aspects-panel")
            with gr.Column(elem_classes=["panel"], elem_id="panel-articles"):
                gr.HTML(PANEL_ARTICLES_TITLE)
                articles_panel = gr.HTML(READY_ARTICLES, elem_id="articles-panel")

    gr.HTML(FOOTER_HTML, elem_id="foot", elem_classes=["foot"])

    # ------------------------------- WIRING -------------------------------
    # Frozen order from docs/DEMO_CONTRACT.md section 3.
    dashboard_outputs = [
        kpi_deck,        # 1  kpi_html
        status_strip,    # 2  status_html
        hist_plot,       # 3  hist_plot
        mc_plot,         # 4  mc_plot (dominant chart)
        fc_plot,         # 5  fc_plot
        aspects_panel,   # 6  aspects_html
        articles_panel,  # 7  articles_html
        error_slot,      # 8  error_html
    ]

    run_btn.click(fn=run_finwise, inputs=[ticker, horizon],
                  outputs=dashboard_outputs,
                  concurrency_limit=1, concurrency_id="finwise-run")
    ticker.submit(fn=run_finwise, inputs=[ticker, horizon],
                  outputs=dashboard_outputs,
                  concurrency_limit=1, concurrency_id="finwise-run")

    reset_btn.click(
        fn=lambda: (
            "AAPL", 7, "", READY_STATUS, READY_KPI,
            None, None, None, READY_ASPECTS, READY_ARTICLES,
        ),
        inputs=None,
        outputs=[
            ticker, horizon, error_slot, status_strip, kpi_deck,
            mc_plot, hist_plot, fc_plot, aspects_panel, articles_panel,
        ],
    )


if __name__ == "__main__":
    # Single runner slot: concurrent clicks serialize here instead of stacking
    # parallel GPU/CPU pipeline runs; overlapping clicks get the BUSY 8-tuple
    # from run_finwise itself, so the visible queue ETA stays truthful.
    demo.queue(max_size=2).launch(share=True)
