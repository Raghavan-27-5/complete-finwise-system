import os
import sys
# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gradio as gr
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from datetime import datetime

from core.orchestrator import compute_state


# =========================================================
# ASSET-RELATIVE VAR CONTEXT (PM-GRADE)
# =========================================================
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


# =========================================================
# CORE CALLBACK
# =========================================================
def run_finwise(symbol: str, days: int):
    state = compute_state(symbol, int(days))
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    # ---------------- PRICE vs MODEL ----------------
    hist = state["history"]
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Scatter(
        x=hist["dates"],
        y=hist["actual"],
        mode="lines",
        name="Observed",
        line=dict(color="rgba(210,210,210,0.9)", width=2)
    ))
    fig_hist.add_trace(go.Scatter(
        x=hist["dates"],
        y=hist["predicted"],
        mode="lines",
        name="Model Fit",
        line=dict(color="rgba(120,180,255,0.7)", dash="dash", width=2)
    ))
    fig_hist.update_layout(
        template="plotly_dark",
        height=300,
        margin=dict(l=40, r=20, t=20, b=40),
        showlegend=True
    )

    # ---------------- FORECAST (SECONDARY) ----------------
    fc = state["forecast"]
    fig_fc = go.Figure()
    fig_fc.add_trace(go.Scatter(
        x=fc["dates"],
        y=fc["predicted"],
        mode="lines+markers",
        line=dict(color="rgba(255,165,0,0.7)", width=2)
    ))
    fig_fc.update_layout(
        template="plotly_dark",
        height=240,
        margin=dict(l=40, r=20, t=20, b=40),
        showlegend=False
    )

    # ---------------- MONTE CARLO (PRIMARY RISK) ----------------
    fig_mc = state["monte_carlo"]["figure"]
    fig_mc.update_layout(
        template="plotly_dark",
        height=380,   # intentionally dominant
        margin=dict(l=30, r=20, t=25, b=30)
    )

    # ---------------- DSP ----------------
    dsp_value = float(state["sentiment_score"])

    # ---------------- DOWNSIDE RISK CONTEXT ----------------
    downside_var = float(state["monte_carlo"]["downside_var"])
    var_pct = downside_var_percentile(hist["actual"], downside_var)

    if var_pct is not None:
        risk_context = f"{var_pct}th percentile vs asset history"
    else:
        risk_context = "insufficient history for percentile"

    # ---------------- SENTIMENT ASPECTS ----------------
    aspects_df = (
        pd.DataFrame(
            [
                (k, round(float(v.get("score", 0.0)), 2))
                for k, v in state["sentiment_aspects"].items()
            ],
            columns=["Aspect", "Signed Impact"]
        )
        .assign(abs_val=lambda df: df["Signed Impact"].abs())
        .sort_values("abs_val", ascending=False)
        .drop(columns="abs_val")
        .reset_index(drop=True)
    )

    # ---------------- ARTICLES ----------------
    articles_df = (
        pd.DataFrame(state["articles"])
        .assign(
            influence=lambda df: df["impact"].abs() * df["weight"]
        )
        .sort_values("influence", ascending=False)
        .assign(
            Dir=lambda df: df["direction"].map({1.0: "+", -1.0: "-"}),
            Impact=lambda df: df["impact"].map(lambda x: f"{x:+.2f}"),
            Weight=lambda df: df["weight"].map(lambda x: f"{x:.2f}"),
            Magnitude=lambda df: df["magnitude"].map(lambda x: f"{x:.2f}"),
            Article=lambda df: df["headline"],
            Entities=lambda df: df["entities"].apply(
                lambda x: ", ".join(x.get("organizations", []))
                if isinstance(x, dict) else ""
            ),
            Published=lambda df: df["published"]
        )
        [[
            "Dir",
            "Impact",
            "Weight",
            "Magnitude",
            "Article",
            "Entities",
            "Published"
        ]]
        .reset_index(drop=True)
    )

    return (
        state["trend"],        # Market Regime
        dsp_value,             # Global DSP
        downside_var,          # 5% VaR
        risk_context,          # Risk Context
        timestamp,             # Eval time
        fig_hist,              # Observed vs Model Fit
        fig_mc,                # Monte Carlo (PRIMARY)
        fig_fc,                # Forecast (SECONDARY)
        aspects_df,            # DSP decomposition
        articles_df,           # Information pressure
    )


# =========================================================
# UI — ELITE PM TERMINAL
# =========================================================
with gr.Blocks(
    title="FINWISE · INTERNAL RISK TERMINAL",
    css="""
    body { background:#0b0b0c; }
    label { font-size:11px; color:#8f8f8f; }
    .gr-markdown { color:#d0d0d0; }
    """
) as demo:

    gr.Markdown(
        "**FINWISE — INTERNAL MARKET INTELLIGENCE**  \n"
        "_Probabilistic · Regime-aware · Non-deterministic_"
    )
    gr.Markdown("""
    <style>

    /* Enable horizontal scrolling for dataframes */
    .gr-dataframe {
        overflow-x: auto !important;
    }

    /* Prevent column compression */
    .gr-dataframe table {
        table-layout: auto !important;
        min-width: 1400px;
    }

    /* Improve readability for long text */
    .gr-dataframe td {
        white-space: normal !important;
        line-height: 1.4em;
    }

    </style>
    """)

    # -------- PRIMARY SIGNALS --------
    with gr.Row():
        regime_box = gr.Textbox(label="Market Regime", scale=2)

        dsp_box = gr.Number(
            label="Global Directional Bias (standardized)",
            scale=4,
            precision=2
        )

        risk_box = gr.Number(
            label="Downside Risk · 5% VaR",
            scale=3,
            precision=3
        )

        risk_ctx_box = gr.Textbox(
            label="Risk Context",
            scale=3
        )

        time_box = gr.Textbox(label="Evaluation Time (UTC)", scale=2)

    # -------- CONTROLS --------
    with gr.Row():
        symbol = gr.Textbox(value="AAPL", label="Asset", scale=2)
        days = gr.Number(value=7, label="Horizon (days)", scale=2)
        run_btn = gr.Button("EVALUATE", scale=2)

    # -------- ANALYTICS --------
    with gr.Row():
        with gr.Column(scale=3):
            gr.Markdown("**Observed vs Model Fit** · diagnostic")
            hist_plot = gr.Plot()

            gr.Markdown("**Return Distribution Envelope** · downside risk focus")
            mc_plot = gr.Plot()

            gr.Markdown("**Conditional Forward Path** · projection")
            forecast_plot = gr.Plot()

        with gr.Column(scale=2):
            gr.Markdown(
                "**Directional Bias Decomposition**  \n"
                "_Aspect-level contributors to global DSP_"
            )
            aspects_table = gr.Dataframe(interactive=False, row_count=6)

            gr.Markdown("**Information Pressure** · narrative inputs")
            articles_table = gr.Dataframe(interactive=False, row_count=6, wrap=True,max_height=360)

    run_btn.click(
        fn=run_finwise,
        inputs=[symbol, days],
        outputs=[
            regime_box,
            dsp_box,
            risk_box,
            risk_ctx_box,
            time_box,
            hist_plot,
            mc_plot,
            forecast_plot,
            aspects_table,
            articles_table,
        ],
    )

demo.launch(share=True)
