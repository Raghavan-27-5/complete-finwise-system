# upd_final_app.py
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs

import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import gradio as gr
import warnings
import requests
from ta.momentum import RSIIndicator
from ta.trend import MACD
from bs4 import BeautifulSoup
from textblob import TextBlob
#from llm_utils import explain_stock_decision

warnings.filterwarnings("ignore")

import socket
import yfinance as yf

# Increase DNS timeout (critical for India / Gradio)
socket.setdefaulttimeout(30)

# In-memory cache to avoid repeated Yahoo calls
_YF_CACHE = {}

def safe_download(
    symbol: str,
    *,
    start=None,
    end=None,
    period=None,
    auto_adjust: bool = True,
    max_retries: int = 3,
):
    """
    PM-grade Yahoo wrapper:
    - retries
    - disables threading (curl bug source)
    - caches results per symbol+window
    """
    cache_key = (symbol, start, end, period, auto_adjust)

    if cache_key in _YF_CACHE:
        return _YF_CACHE[cache_key].copy()

    last_exc = None
    for _ in range(max_retries):
        try:
            df = yf.download(
                symbol,
                start=start,
                end=end,
                period=period,
                auto_adjust=auto_adjust,
                threads=False,
                progress=False,
            )
            if df is not None and not df.empty:
                _YF_CACHE[cache_key] = df.copy()
                return df
        except Exception as e:
            last_exc = e

    raise RuntimeError(f"Yahoo download failed for {symbol}") from last_exc

TIME_STEP = 60
DATA_YEARS = 3
import os
MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "stock_price_model.h5"
)
model = load_model(MODEL_PATH)

model.make_predict_function()
portfolio = {}

# ----------------- DATA PREPROCESSING -----------------
def preprocess_data(df):
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    df = df.reset_index().rename(columns={'index': 'Date'})
    df = df[['Date', 'High', 'Low', 'Open', 'Close', 'Volume']]
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df

def get_stock_data(stock_symbol):
    index_symbols = ['^NSEI', '^BSESN', '^NSEBANK', '^DJI', '^IXIC', '^GSPC']
    if stock_symbol not in index_symbols and not (stock_symbol.endswith('.NS') or stock_symbol.endswith('.BO') or stock_symbol.isupper()):
        stock_symbol += '.NS'
    end_date = dt.datetime.now()
    start_date = end_date - dt.timedelta(days=365 * DATA_YEARS)
    df = safe_download(
         stock_symbol,
         start=start_date,
         end=end_date,
         auto_adjust=True,
    )
    return preprocess_data(df)

def prepare_data(df):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))
    X = np.array([scaled_data[i:i + TIME_STEP, 0] for i in range(len(scaled_data) - TIME_STEP - 1)])
    y = scaled_data[TIME_STEP + 1:, 0]
    return X.reshape(X.shape[0], TIME_STEP, 1), y, scaler

# ----------------- FUTURE PREDICTION -----------------
def next_trading_day(last_date):
    next_day = last_date + pd.Timedelta(days=1)
    while next_day.weekday() >= 5:
        next_day += pd.Timedelta(days=1)
    return next_day

# ----------------- NEW FUTURE PREDICTION FUNCTION -----------------
def predict_future(model, df, scaler, prediction_days, include_last_n=3):
    """
    Predicts future stock prices for a given number of days, skipping weekends.
    """
    # Get the last TIME_STEP days from the real data to start the prediction
    last_window = df['Close'].values[-TIME_STEP:]
    
    # Scale this initial window
    last_window_scaled = scaler.transform(last_window.reshape(-1, 1))

    # Create empty lists to store the future predictions and dates
    # future_predictions = []
    future_dates = []

    # Get the last known date from the dataframe
    last_date = df.index[-1]

    # --- F3 FIX: Single-Step Prediction ---
    # We predict ONLY t+1 to avoid recursive error accumulation.
    # The result is projected flat for the requested horizon.
    
    # 1. Get the last window
    scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))
    last_window = scaled_data[-60:]
    last_window_reshaped = np.reshape(last_window, (1, last_window.shape[0], 1))
    
    # 2. Predict once
    next_scaled_pred = model.predict(last_window_reshaped, verbose=0)
    next_price = scaler.inverse_transform(next_scaled_pred)[0][0]
    
    # 3. Project flat
    future_predictions = [next_price] * prediction_days
    
    # Generate dates
    future_dates = []
    current_date = df.index[-1]
    for _ in range(prediction_days):
        current_date = next_trading_day(current_date)
        future_dates.append(current_date)

    # 4. Construct result arrays
    # Combine the last few days of real data with our flat projection
    final_prices = np.concatenate([df['Close'].values[-include_last_n:], future_predictions])
    final_dates = list(df.index[-include_last_n:]) + future_dates
    
    return final_dates, final_prices

# ----------------- SENTIMENT & TECHNICALS -----------------
def get_news(stock_symbol):
    try:
        search_query = stock_symbol.replace('.NS', '').replace('.BO', '')
        url = f'https://news.google.com/rss/search?q={search_query}+stock+when:7d&hl=en-IN&gl=IN&ceid=IN:en'
        response = requests.get(url)
        soup = BeautifulSoup(response.content, features='xml')
        items = soup.findAll('item')
        headlines = [item.title.text for item in items[:5]]
        return headlines if headlines else ["No recent news found"]
    except:
        return ["Failed to fetch news"]

def analyze_sentiment(news_headlines):
    polarity = sum(TextBlob(h).sentiment.polarity for h in news_headlines) / len(news_headlines)
    if polarity > 0.1:
        return "Positive"
    elif polarity < -0.1:
        return "Negative"
    else:
        return "Neutral"

def detect_candlestick_patterns(df):
    last = df.iloc[-1]
    open_ = last['Open']
    close = last['Close']
    high = last['High']
    low = last['Low']
    body = abs(close - open_)
    candle_range = high - low
    if candle_range == 0:
        return "Invalid candle"
    body_pct = body / candle_range
    upper_shadow_pct = (high - max(open_, close)) / candle_range
    lower_shadow_pct = (min(open_, close) - low) / candle_range
    if body_pct < 0.1 and upper_shadow_pct > 0.2 and lower_shadow_pct > 0.2:
        return "Doji – Indecision"
    elif body_pct < 0.3 and lower_shadow_pct > 2 * body_pct and upper_shadow_pct < 0.1 and close > open_:
        return "Bullish Hammer"
    elif body_pct < 0.3 and upper_shadow_pct > 2 * body_pct and lower_shadow_pct < 0.1 and open_ > close:
        return "Bearish Shooting Star"
    elif len(df) > 1:
        prev = df.iloc[-2]
        if prev['Close'] < prev['Open'] and close > open_ and close > prev['Open'] and open_ < prev['Close']:
            return "Bullish Engulfing"
        if prev['Close'] > prev['Open'] and close < open_ and open_ > prev['Close'] and close < prev['Open']:
            return "Bearish Engulfing"
    elif close > open_ and body_pct > 0.6:
        return "Strong Bullish Candle"
    elif close < open_ and body_pct > 0.6:
        return "Strong Bearish Candle"
    return "No Clear Pattern"

def decision_logic_v2(df, future_prices, sentiment):
    rsi = df['RSI'].iloc[-1]
    macd = df['MACD'].iloc[-1]
    trend_slope = future_prices[-1] - future_prices[0]
    decision = []
    if trend_slope > 0: decision.append("Upward Trend")
    elif trend_slope < 0: decision.append("Downward Trend")
    if rsi < 30: decision.append("RSI indicates Oversold (Buy)")
    elif rsi > 70: decision.append("RSI indicates Overbought (Sell)")
    if macd > 0: decision.append("MACD Positive")
    else: decision.append("MACD Negative")
    if sentiment == "Positive": decision.append("Positive News Sentiment")
    elif sentiment == "Negative": decision.append("Negative News Sentiment")
    score = 0
    if trend_slope > 0: score +=1
    if rsi < 30: score +=1
    if macd > 0: score +=1
    if sentiment == "Positive": score +=1
    if score >= 3: final_decision = "Buy"
    elif score == 2: final_decision = "Hold"
    else: final_decision = "Sell"
    explanation = " | ".join(decision)
    return final_decision, explanation

# ----------------- MONTE CARLO HESTON (returns fig + stats string) -----------------
def monte_carlo_heston(
    stock_symbol: str,
    last_price: float,
    lstm_forecast: float,
    days: int = 14,
    n_simulations: int = 5000,
    sample_paths: int = 100
):
    """
    Returns:
      fig -> Plotly figure (sample_paths plotted for clarity)
      stats_str -> small string with mean / 5% / 95% for final-day distribution
    """
    """
    Returns:
      fig -> Plotly figure (sample_paths plotted for clarity)
      stats_str -> small string with mean / 5% / 95% for final-day distribution
    """
    import yfinance as yf
    import numpy as np
    import plotly.graph_objects as go

    # Safety: force at least 2 days for plotting and avoid 1-day mismatches
    days = max(int(days), 2)

    # Fetch recent returns to calibrate volatility (Realized Volatility)
    try:
        stock_df = safe_download(
                   stock_symbol,
                   period="60d",
                   auto_adjust=True,
        )

        returns = np.log(stock_df['Close'] / stock_df['Close'].shift(1)).dropna()
        # Daily volatility
        sigma_daily = returns.std() if len(returns) > 0 else 0.02
    except Exception:
        sigma_daily = 0.02

    # Calculate Drift required to hit LSTM forecast (Geometric Drift)
    lstm_forecast = float(lstm_forecast)
    last_price = float(last_price)
    
    daily_drift = 0.0
    if last_price > 0 and days > 0:
        # Total log return required / number of days
        daily_drift = np.log(max(lstm_forecast, 1e-8) / last_price) / days

    # ----------------- GBM SIMULATION -----------------
    # Model: P_t = P_{t-1} * exp( r_t )
    # r_t ~ N(daily_drift - 0.5*sigma^2, sigma) ? 
    # To keep it simple and centered on LSTM forecast:
    # We sample log-returns around the required drift.
    
    # Generate random log-returns
    # Shape: (n_simulations, days)
    # We apply the drift needed to reach target on average, plus volatility noise.
    # Note: If we want E[P_T] = Forecast, we need to adjust for Jensen's inequality,
    # but for a "cone" visualization, direct log-normal sampling centered on drift is standard.
    
    # Random component: N(0, 1) * sigma
    random_shocks = np.random.normal(0, 1, (n_simulations, days)) * sigma_daily
    
    # Path evolution
    log_returns = daily_drift + random_shocks
    
    # Cumulative log returns
    cum_log_returns = np.cumsum(log_returns, axis=1)
    
    # Prices
    prices = last_price * np.exp(cum_log_returns)
    
    # Prepend starting price for plotting
    start_col = np.full((n_simulations, 1), last_price)
    prices = np.hstack([start_col, prices])

    # Stats on final-day distribution
    final_prices = prices[:, -1]
    mc_mean = np.mean(final_prices)
    mc_p5 = np.percentile(final_prices, 5)
    mc_p95 = np.percentile(final_prices, 95)

    stats_str = f"Simulated final day — mean: {mc_mean:.2f}, 5%: {mc_p5:.2f}, 95%: {mc_p95:.2f}"

    # Build Plotly figure (plot only sample_paths)
    x_axis = list(range(0, days + 1))
    fig = go.Figure()
    
    # Plot a subset of paths
    subset = prices[:min(sample_paths, n_simulations)]
    for i in range(len(subset)):
        fig.add_trace(go.Scatter(x=x_axis, y=subset[i], mode='lines',
                                 line=dict(width=1, color='gray'), opacity=0.35, showlegend=False))
                                 
    mean_path = np.mean(prices, axis=0)
    upper_bound = np.percentile(prices, 97.5, axis=0)
    lower_bound = np.percentile(prices, 2.5, axis=0)

    fig.add_trace(go.Scatter(x=x_axis, y=mean_path, mode="lines", name="Mean Projection",
                             line=dict(color="white", width=3)))
    fig.add_trace(go.Scatter(x=x_axis, y=upper_bound, mode="lines", name="Upper Vol Cone (97.5%)",
                             line=dict(color="lightgreen", dash='dot')))
    fig.add_trace(go.Scatter(x=x_axis, y=lower_bound, mode="lines", name="Lower Vol Cone (2.5%)",
                             line=dict(color="lightcoral", dash='dot')))

    fig.update_layout(title=f"Volatility Cone Simulation (GBM) for {stock_symbol}",
                      xaxis_title="Days Ahead", yaxis_title="Price",
                      template="plotly_dark", height=520)

    return fig, stats_str

def monte_carlo_heston_stats_only(
    stock_symbol: str,
    last_price: float,
    lstm_forecast: float,
    days: int = 14,
    n_simulations: int = 5000,
):
    """
    PM-grade downside risk estimator (distributional, volatility-aware)
    Returns: downside_risk (0.0 – 0.50)
    """
    import numpy as np

    days = max(int(days), 2)

    # -------------------------------
    # 1) Realized volatility (annualized)
    # -------------------------------
    try:
        stock_df = safe_download(
            stock_symbol,
            period="60d",
            auto_adjust=True,
        )
        returns = np.log(stock_df['Close'] / stock_df['Close'].shift(1)).dropna()
        # Daily volatility
        sigma_daily = returns.std() if len(returns) > 0 else 0.02
    except Exception:
        sigma_daily = 0.02

    # -------------------------------
    # 2) Drift from LSTM anchor
    # -------------------------------
    lstm_forecast = float(lstm_forecast)
    last_price = float(last_price)

    daily_drift = 0.0
    if last_price > 0 and days > 0:
        daily_drift = np.log(max(lstm_forecast, 1e-8) / last_price) / days

    # -------------------------------
    # 3) GBM Simulation
    # -------------------------------
    # Generate cumulative log returns
    random_shocks = np.random.normal(0, 1, (n_simulations, days)) * sigma_daily
    log_returns = daily_drift + random_shocks
    cum_log_returns = np.cumsum(log_returns, axis=1)
    
    # Final prices
    final_prices = last_price * np.exp(cum_log_returns[:, -1])

    # -------------------------------
    # 4) Downside Risk Calc
    # -------------------------------
    # Probability of loss > 5%
    downside_threshold = last_price * 0.95
    p_loss = np.mean(final_prices < downside_threshold)

    return float(min(p_loss, 0.50))



# ----------------- MAIN PREDICTION FUNCTION -----------------
def predict_stock(stock_symbol, prediction_days, invest_amount):
    try:
        prediction_days = int(prediction_days)
        df = get_stock_data(stock_symbol)
        X, y, scaler = prepare_data(df)
        y_pred = model.predict(X)
        y_pred = scaler.inverse_transform(y_pred)
        df['RSI'] = RSIIndicator(df['Close'], window=14).rsi()
        df['MACD'] = MACD(df['Close']).macd()
        news_headlines = get_news(stock_symbol)
        sentiment = analyze_sentiment(news_headlines)
        candlestick_pattern = detect_candlestick_patterns(df)
        future_dates, future_prices = predict_future(model, df, scaler, prediction_days, include_last_n=3)
        decision, explanation = decision_logic_v2(df, future_prices, sentiment)
        currency_symbol = '₹' if stock_symbol.endswith('.NS') or stock_symbol.endswith('.BO') or stock_symbol in ['^NSEI','^BSESN','^NSEBANK'] else '$'

        if invest_amount and float(invest_amount) > 0:
            amount = float(invest_amount)
            shares = amount / df['Close'].iloc[-1]
            portfolio[stock_symbol] = {
                'Invested': f"{currency_symbol}{amount:.2f}",
                'Buy Price': f"{currency_symbol}{df['Close'].iloc[-1]:.2f}",
                'Shares': f"{shares:.4f}"
            }

        # --- Actual vs Predicted plot ---
        display_days = 365
        df_display = df.tail(display_days)
        pred_dates = df.index[TIME_STEP+1:]
        pred_series = pd.Series(y_pred[:, 0], index=pred_dates)
        pred_display = pred_series.tail(display_days)
        actual_vs_pred_fig = go.Figure()
        actual_vs_pred_fig.add_trace(go.Scatter(x=df_display.index, y=df_display['Close'], name='Actual Price', line=dict(color='blue')))
        actual_vs_pred_fig.add_trace(go.Scatter(x=pred_display.index, y=pred_display.values, name='Predicted Price', line=dict(color='orange')))
        actual_vs_pred_fig.update_layout(title=f"{stock_symbol} Actual vs Predicted ({currency_symbol}) - Last {display_days} Days", template='plotly_dark')
        

        # --- Future Forecast plot ---
        future_fig = go.Figure()
        future_fig.add_trace(go.Scatter(x=future_dates, y=future_prices, mode='lines+markers', line=dict(color='green', width=3)))
        future_fig.update_layout(title=f"{stock_symbol} Future Forecast (Last 3 Days + Next Trading Day)", template='plotly_dark', xaxis_title='Date', yaxis_title='Price', height=500)

        # --- Monte Carlo plot (Heston-like) + stats ---
        # --- Monte Carlo plot (Heston) ---
        mc_fig, mc_stats = monte_carlo_heston(
                stock_symbol,
                last_price=float(df['Close'].iloc[-1]),
                lstm_forecast=float(future_prices.iloc[-1]) if hasattr(future_prices, "iloc") else float(future_prices[-1]),
                days=int(prediction_days) if prediction_days > 1 else 14,
                n_simulations=10000
                )
        explanation_with_mc = f"{explanation}\n{mc_stats}"


        return (
            gr.update(value=f"{currency_symbol}{df['Close'].iloc[-1]:.2f}"),
            df.index[-1].strftime('%Y-%m-%d'),
            decision,
            candlestick_pattern,
            "\n".join(news_headlines),
            sentiment,
            actual_vs_pred_fig,
            future_fig,
            mc_fig,
            str(portfolio),
            explanation_with_mc
              # show MC summary stats in the Decision Explanation box (last output)
        )

    except Exception as e:
        return (
        gr.update(value="Error"),      # 1. Price Textbox
        "N/A",                         # 2. Date Textbox
        "Error",                       # 3. Decision Textbox
        "N/A",                         # 4. Candlestick Textbox
        "N/A",                         # 5. News Textbox
        "N/A",                         # 6. Sentiment Textbox
        None,                          # 7. First Plot (Actual vs. Pred)
        None,                          # 8. Second Plot (Future)
        None,                          # 9. Third Plot (Monte Carlo)
        "{}",                          # 10. Portfolio Textbox
        f"An error occurred: {str(e)}" # 11. Explanation Textbox
    )

# ----------------- Chat Reset -----------------
def reset_conversation():
    import llm_utils
    llm_utils.chat_history = []
    return "🧹 Chat reset. Ready to start again!"


# ========= BACKEND API FOR FINWISE COPILOT =========
from typing import Dict, Any, List

from typing import Dict, Any

def build_price_block(stock_symbol: str, prediction_days: int = 7) -> Dict[str, Any]:
    """
    Backend-only price intelligence for FinWise Copilot.
    No Gradio, no LLM, no UI. Pure numbers.
    """

    # 1) Get cleaned OHLCV
    df = get_stock_data(stock_symbol)  # uses preprocess_data() internally

    # 2) Prepare data + model input
    X, y, scaler = prepare_data(df)    # uses TIME_STEP global
    y_pred_scaled = model.predict(X)
    y_pred = scaler.inverse_transform(y_pred_scaled)

    # 3) Technical indicators
    df['RSI'] = RSIIndicator(df['Close'], window=14).rsi()
    df['MACD'] = MACD(df['Close']).macd()

    # 4) Walk-forward future prediction
    prediction_days = int(prediction_days)
    future_dates, future_prices = predict_future(
        model,
        df,
        scaler,
        prediction_days,
        include_last_n=3
    )

    # 5) History: Actual vs Predicted (last 365 days)
    display_days = 365
    df_display = df.tail(display_days)

    # y_pred aligned with df index starting at TIME_STEP+1
    pred_dates = df.index[TIME_STEP + 1:]
    pred_series = pd.Series(y_pred[:, 0], index=pred_dates)
    pred_display = pred_series.tail(display_days)

    history = {
        "dates": [d.strftime("%Y-%m-%d") for d in df_display.index],
        "actual": df_display["Close"].astype(float).tolist(),
        "predicted": pred_display.astype(float).tolist(),
    }

    # 6) Forecast block (future)
    # future_dates is list of Timestamps, future_prices is np.array or list
    # We want only the FUTURE part (exclude last_n history)
    # predict_future concatenates last include_last_n actuals + predicted.
    # So split:
    include_last_n = 3
    future_only_dates = future_dates[include_last_n:]
    future_only_prices = future_prices[include_last_n:]

    forecast = {
        "dates": [d.strftime("%Y-%m-%d") for d in future_only_dates],
        "predicted": [float(p) for p in future_only_prices],
    }

    # 7) Monte Carlo risk stats (no figure)
    last_price = float(df["Close"].iloc[-1])
    # Use last forecasted price as anchor
    lstm_forecast_price = float(future_only_prices[-1]) if len(future_only_prices) > 0 else last_price

    downside_risk = monte_carlo_heston_stats_only(
        stock_symbol=stock_symbol,
        last_price=last_price,
        lstm_forecast=lstm_forecast_price,
        days=prediction_days if prediction_days > 1 else 14,
        n_simulations=5000,
    )
    # 7b) Monte Carlo path fan (figure for terminal visualization)
    mc_fig, _ = monte_carlo_heston(
        stock_symbol=stock_symbol,
        last_price=last_price,
        lstm_forecast=lstm_forecast_price,
        days=prediction_days if prediction_days > 1 else 14,
        n_simulations=3000,
        sample_paths=120
    )

    monte_carlo = {
        "downside_var": downside_risk,
        "figure": mc_fig 
    }
    
    # 8) Indicators + momentum
    rsi_val = float(df["RSI"].iloc[-1])
    macd_val = float(df["MACD"].iloc[-1])
    signal_val = 0.0  # you can add MACD signal column later if needed

    trend_slope = forecast["predicted"][-1] - forecast["predicted"][0] if len(forecast["predicted"]) > 1 else 0.0

    if trend_slope > 0 and macd_val > 0:
        momentum = "Bullish"
    elif trend_slope < 0 and macd_val < 0:
        momentum = "Bearish"
    else:
        momentum = "Neutral"

    indicators = {
        "rsi": rsi_val,
        "macd": macd_val,
        "signal": signal_val,
        "momentum": momentum,
    }

    # 9) Trend tag
    if trend_slope > 0:
        trend = "Bullish"
    elif trend_slope < 0:
        trend = "Bearish"
    else:
        trend = "Sideways"

    return {
        "history": history,
        "forecast": forecast,
        "monte_carlo": monte_carlo,
        "indicators": indicators,
        "trend": trend,
    }


# ----------------- GRADIO UI -----------------
if __name__ == "__main__":
    demo = gr.Blocks()

    with demo:
        gr.Markdown("# 🌏 FinWise AI: Global Stock & Index Predictor 📊")
        with gr.Row():
            stock_input = gr.Textbox(label="Enter Stock Symbol", value="RELIANCE")
            days_input = gr.Textbox(label="Days to Predict", value="1")
            invest_input = gr.Textbox(label="Virtual Investment Amount", value="0")
            submit_btn = gr.Button("Predict")

        with gr.Row():
            last_price = gr.Textbox(label="Last Price", interactive=False)
            last_date = gr.Textbox(label="Last Date", interactive=False)

        with gr.Row():
            decision = gr.Textbox(label="Investment Decision", interactive=False)
            pattern = gr.Textbox(label="Candlestick Pattern", interactive=False)
            explanation_box = gr.Textbox(label="Decision Explanation / MC Summary", lines=3, interactive=False)

        with gr.Row():
            headlines = gr.Textbox(label="Recent News Headlines", lines=4, interactive=False)
            sentiment_box = gr.Textbox(label="Sentiment Analysis", interactive=False)

        with gr.Tabs():
            with gr.Tab("Actual vs Predicted"):
                actual_vs_pred_plot = gr.Plot()
            with gr.Tab("Future Forecast"):
                future_plot = gr.Plot()
            with gr.Tab("Monte Carlo (Heston)"):
                mc_plot = gr.Plot()
            with gr.Tab("Portfolio"):
                portfolio_output = gr.Textbox(label="Portfolio Summary", lines=4, interactive=False)
            with gr.Tab("💬 Ask AI About Stock"):
                user_question = gr.Textbox(label="Ask FinWise AI", placeholder=" Ask me anything about finance")
                chat_display = gr.Textbox(label="Conversation", lines=15, interactive=False, show_copy_button=True)
                ask_btn = gr.Button("Ask")
                reset_btn = gr.Button("🔄 Reset Chat")

        submit_btn.click(
            fn=predict_stock,
            inputs=[stock_input, days_input, invest_input],
            outputs=[last_price, last_date, decision, pattern, headlines, sentiment_box,
                     actual_vs_pred_plot, future_plot, mc_plot, portfolio_output, explanation_box]
        )

        ask_btn.click(
            #fn=explain_stock_decision,
            inputs=[stock_input, explanation_box, user_question],
            outputs=[chat_display]
        )

        reset_btn.click(
            fn=reset_conversation,
            inputs=[],
            outputs=[chat_display]
        )

    demo.launch(share=True)


    
