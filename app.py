#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ==============================================================================
# MAGNIFICENT 7 QUANTITATIVE TRADING ENGINE & DASHBOARD
# Streamlit Web Application (app.py)
# ==============================================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.calibration import CalibratedClassifierCV

# ------------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & STYLING
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Mag 7 Quantitative Trading Engine",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Institutional CSS Injection
st.markdown("""
<style>
    /* Dark Theme Base Rules */
    .stApp {
        background-color: #0E1117;
        color: #E0E0E0;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Clean Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #161922;
        border-right: 1px solid #262C3A;
    }
    
    /* Custom Metric Display Cards */
    .metric-card {
        background-color: #161922;
        border: 1px solid #262C3A;
        border-radius: 4px;
        padding: 16px;
        text-align: left;
    }
    .metric-label {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #8A92A6;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #FFFFFF;
    }
    .metric-sub {
        font-size: 0.8rem;
        font-weight: 500;
        margin-top: 4px;
    }
    
    /* Status Signal Typography */
    .status-bullish { color: #00E676; }
    .status-caution { color: #FFB300; }
    .status-bearish { color: #FF5252; }
    .status-neutral { color: #9E9E9E; }
    
    /* Divider Rules */
    hr {
        border-color: #262C3A !important;
        margin: 1.5rem 0 !important;
    }
</style>
""", unsafe_allow_html=True)

# Application Header
st.title("Mag 7 Algorithmic Trading & ML Prediction Engine")
st.caption("Quantitative Execution Engine | XGBoost Probability Calibration & Dynamic Sizing")

# ------------------------------------------------------------------------------
# 2. PIPELINE EXECUTION ENGINE (Cached for Performance)
# ------------------------------------------------------------------------------
MAG_7 = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA"]

@st.cache_data(ttl=3600)
def load_and_process_data():
    price_data = yf.download(MAG_7, start="2019-01-01", progress=False)
    df_ml = price_data['Close'].stack().reset_index()
    df_ml.columns = ['Date', 'Ticker', 'Close']
    df_ml = df_ml.sort_values(['Ticker', 'Date']).reset_index(drop=True)

    # Technical Features
    delta = df_ml.groupby('Ticker')['Close'].diff()
    gain = delta.where(delta > 0, 0).groupby(df_ml['Ticker']).rolling(14).mean().reset_index(level=0, drop=True)
    loss = (-delta.where(delta < 0, 0)).groupby(df_ml['Ticker']).rolling(14).mean().reset_index(level=0, drop=True)
    rs = gain / (loss + 1e-9)
    df_ml['RSI_14'] = 100 - (100 / (1 + rs))

    ema12 = df_ml.groupby('Ticker')['Close'].transform(lambda x: x.ewm(span=12, adjust=False).mean())
    ema26 = df_ml.groupby('Ticker')['Close'].transform(lambda x: x.ewm(span=26, adjust=False).mean())
    df_ml['MACD'] = ema12 - ema26

    df_ml['MA_20'] = df_ml.groupby('Ticker')['Close'].transform(lambda x: x.rolling(20).mean())
    df_ml['MA_50'] = df_ml.groupby('Ticker')['Close'].transform(lambda x: x.rolling(50).mean())
    df_ml['MA_200'] = df_ml.groupby('Ticker')['Close'].transform(lambda x: x.rolling(200).mean())
    df_ml['MA_Ratio_20_50'] = df_ml['MA_20'] / df_ml['MA_50']

    df_ml['Returns_1D'] = df_ml.groupby('Ticker')['Close'].pct_change()
    df_ml['Vol_20D'] = df_ml.groupby('Ticker')['Returns_1D'].transform(lambda x: x.rolling(20).std())

    # Target
    df_ml['Returns_5D_Fwd'] = df_ml.groupby('Ticker')['Close'].transform(lambda x: x.pct_change(5, fill_method=None).shift(-5))
    df_ml['Target'] = (df_ml['Returns_5D_Fwd'] > 0.0).astype(int)

    return df_ml.dropna().reset_index(drop=True)

@st.cache_data(ttl=1800)
def fetch_live_sentiment():
    analyzer = SentimentIntensityAnalyzer()
    sentiment_dict = {}
    for ticker in MAG_7:
        try:
            t = yf.Ticker(ticker)
            news = t.news
            scores = []
            if news:
                for item in news:
                    content = item.get('content', {})
                    title = content.get('title') or item.get('title', '')
                    if title:
                        scores.append(analyzer.polarity_scores(title)['compound'])
            sentiment_dict[ticker] = sum(scores) / len(scores) if scores else 0.0
        except Exception:
            sentiment_dict[ticker] = 0.0
    return sentiment_dict

# Pipeline Initialization
with st.spinner("Compiling quantitative features..."):
    df_ml = load_and_process_data()
    sentiment_results = fetch_live_sentiment()

df_ml['News_Sentiment'] = df_ml['Ticker'].map(sentiment_results)
feature_cols = ['RSI_14', 'MACD', 'MA_Ratio_20_50', 'Vol_20D', 'News_Sentiment']

X = df_ml[feature_cols]
y = df_ml['Target']

base_model = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42, eval_metric='logloss')
calibrated_model = CalibratedClassifierCV(estimator=base_model, method='isotonic', cv=5)
calibrated_model.fit(X, y)

df_ml['Bullish_Prob'] = calibrated_model.predict_proba(X)[:, 1]

# ------------------------------------------------------------------------------
# 3. INTERACTIVE SIDEBAR CONTROLS
# ------------------------------------------------------------------------------
st.sidebar.markdown("### System Controls")
selected_ticker = st.sidebar.selectbox("Asset / Ticker", MAG_7)

st.sidebar.markdown("---")
st.sidebar.markdown("### Execution Parameters")
execution_mode = st.sidebar.radio("Strategy Framework", ["Dynamic Position Sizing", "Binary Thresholds"])

if execution_mode == "Binary Thresholds":
    buy_threshold = st.sidebar.slider("Buy Threshold", 0.50, 0.70, 0.52, step=0.01)
    sell_threshold = st.sidebar.slider("Sell Threshold", 0.30, 0.50, 0.48, step=0.01)
else:
    min_allocation = st.sidebar.slider("Bull Market Base Allocation", 0.0, 1.0, 0.35, step=0.05)
    use_regime_filter = st.sidebar.checkbox("Enable 200 MA Regime Filter", value=True)

# Asset Data Processing
ticker_df = df_ml[df_ml['Ticker'] == selected_ticker].sort_values('Date').reset_index(drop=True)
ticker_df['Prob_Rank'] = ticker_df['Bullish_Prob'].rank(pct=True)

latest_row = ticker_df.iloc[-1]
prob = latest_row['Bullish_Prob']
latest_rank = latest_row['Prob_Rank']

# Dynamic Position Logic
if execution_mode == "Binary Thresholds":
    if prob >= buy_threshold:
        signal = "BUY"
        sub_signal = "Bullish Conviction"
        status_class = "status-bullish"
    elif prob <= sell_threshold:
        signal = "SELL"
        sub_signal = "Bearish Conviction"
        status_class = "status-bearish"
    else:
        signal = "HOLD"
        sub_signal = "Neutral Regime"
        status_class = "status-neutral"
else:
    is_bull_regime = latest_row['Close'] > latest_row['MA_200'] if use_regime_filter else True
    
    if is_bull_regime:
        if latest_rank >= 0.20:
            alloc = 1.0
            signal = "ALLOCATE 100.0%"
            sub_signal = "Full Bull Regime"
            status_class = "status-bullish"
        else:
            alloc = min_allocation
            signal = f"ALLOCATE {alloc * 100:.1f}%"
            sub_signal = "Defensive Risk Trim"
            status_class = "status-caution"
    else:
        if latest_rank >= 0.70:
            alloc = latest_rank
            signal = f"ALLOCATE {alloc * 100:.1f}%"
            sub_signal = "Counter-Trend Exposure"
            status_class = "status-caution"
        else:
            alloc = 0.0
            signal = "ALLOCATE 0.0%"
            sub_signal = "Bear Regime Filter"
            status_class = "status-bearish"

# ------------------------------------------------------------------------------
# 4. CUSTOM INFERENCE METRIC DISPLAY CARDS
# ------------------------------------------------------------------------------
st.markdown("---")
c1, c2, c3, c4, c5 = st.columns(5)

c1.markdown(f"""
<div class="metric-card">
    <div class="metric-label">Latest Close</div>
    <div class="metric-value">${latest_row['Close']:.2f}</div>
</div>
""", unsafe_allow_html=True)

c2.markdown(f"""
<div class="metric-card">
    <div class="metric-label">RSI (14)</div>
    <div class="metric-value">{latest_row['RSI_14']:.2f}</div>
</div>
""", unsafe_allow_html=True)

c3.markdown(f"""
<div class="metric-card">
    <div class="metric-label">News Sentiment</div>
    <div class="metric-value">{latest_row['News_Sentiment']:.4f}</div>
</div>
""", unsafe_allow_html=True)

c4.markdown(f"""
<div class="metric-card">
    <div class="metric-label">Calibrated Probability</div>
    <div class="metric-value">{prob * 100:.2f}%</div>
</div>
""", unsafe_allow_html=True)

c5.markdown(f"""
<div class="metric-card">
    <div class="metric-label">Execution Signal</div>
    <div class="metric-value {status_class}">{signal}</div>
    <div class="metric-sub {status_class}">{sub_signal}</div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ------------------------------------------------------------------------------
# 5. CHARTS: PRICE, TECHNICALS & PREDICTION PROBABILITIES
# ------------------------------------------------------------------------------
st.markdown("### Asset Performance & Signal Trajectory")

fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.7, 0.3])

# Dark Chart Theme Styling
plotly_template = {
    "layout": {
        "paper_bgcolor": "#0E1117",
        "plot_bgcolor": "#0E1117",
        "font": {"color": "#8A92A6", "family": "Arial, sans-serif"},
        "xaxis": {"gridcolor": "#1F2430", "showline": False},
        "yaxis": {"gridcolor": "#1F2430", "showline": False}
    }
}

fig.add_trace(go.Scatter(x=ticker_df['Date'], y=ticker_df['Close'], name='Close Price', line=dict(color='#2962FF', width=1.5)), row=1, col=1)
fig.add_trace(go.Scatter(x=ticker_df['Date'], y=ticker_df['MA_20'], name='20 MA', line=dict(color='#FF6D00', width=1, dash='dot')), row=1, col=1)
fig.add_trace(go.Scatter(x=ticker_df['Date'], y=ticker_df['MA_50'], name='50 MA', line=dict(color='#AA00FF', width=1, dash='dot')), row=1, col=1)
fig.add_trace(go.Scatter(x=ticker_df['Date'], y=ticker_df['MA_200'], name='200 MA', line=dict(color='#FF5252', width=1.2)), row=1, col=1)

fig.add_trace(go.Scatter(x=ticker_df['Date'], y=ticker_df['Bullish_Prob'], name='Calibrated Probability', line=dict(color='#00E676', width=1.2)), row=2, col=1)

if execution_mode == "Binary Thresholds":
    fig.add_hline(y=buy_threshold, line_dash="dash", line_color="#00E676", row=2, col=1)
    fig.add_hline(y=sell_threshold, line_dash="dash", line_color="#FF5252", row=2, col=1)

fig.update_layout(
    template=plotly_template,
    height=500,
    margin=dict(l=10, r=10, t=10, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1)
)
st.plotly_chart(fig, width="stretch")

# ------------------------------------------------------------------------------
# 6. DYNAMIC STRATEGY BACKTESTER
# ------------------------------------------------------------------------------
st.markdown("---")
st.markdown("### Strategy Backtest vs. Buy & Hold Benchmark")

ticker_df['Market_Return'] = ticker_df['Close'].pct_change()

if execution_mode == "Binary Thresholds":
    ticker_df['Signal_State'] = 0.0
    ticker_df.loc[ticker_df['Bullish_Prob'] >= buy_threshold, 'Signal_State'] = 1.0
    ticker_df.loc[ticker_df['Bullish_Prob'] <= sell_threshold, 'Signal_State'] = 0.0
else:
    is_bull = ticker_df['Close'] > ticker_df['MA_200'] if use_regime_filter else True
    if use_regime_filter:
        ticker_df['Signal_State'] = np.where(
            is_bull,
            np.where(ticker_df['Prob_Rank'] >= 0.20, 1.0, min_allocation),
            np.where(ticker_df['Prob_Rank'] >= 0.70, ticker_df['Prob_Rank'], 0.0)
        )
    else:
        ticker_df['Signal_State'] = ticker_df['Prob_Rank']

ticker_df['Strategy_Return'] = ticker_df['Signal_State'].shift(1) * ticker_df['Market_Return']

ticker_df['Cum_Market_Return'] = (1 + ticker_df['Market_Return'].fillna(0)).cumprod()
ticker_df['Cum_Strategy_Return'] = (1 + ticker_df['Strategy_Return'].fillna(0)).cumprod()

total_strat_return = (ticker_df['Cum_Strategy_Return'].iloc[-1] - 1) * 100
total_market_return = (ticker_df['Cum_Market_Return'].iloc[-1] - 1) * 100

sharpe_strat = (ticker_df['Strategy_Return'].mean() / (ticker_df['Strategy_Return'].std() + 1e-9)) * np.sqrt(252)
sharpe_market = (ticker_df['Market_Return'].mean() / (ticker_df['Market_Return'].std() + 1e-9)) * np.sqrt(252)

# Backtest Performance Metrics Display
m1, m2, m3, m4 = st.columns(4)

m1.markdown(f"""
<div class="metric-card">
    <div class="metric-label">Strategy Total Return</div>
    <div class="metric-value status-bullish">{total_strat_return:.2f}%</div>
</div>
""", unsafe_allow_html=True)

m2.markdown(f"""
<div class="metric-card">
    <div class="metric-label">Buy & Hold Return</div>
    <div class="metric-value">{total_market_return:.2f}%</div>
</div>
""", unsafe_allow_html=True)

m3.markdown(f"""
<div class="metric-card">
    <div class="metric-label">Strategy Sharpe Ratio</div>
    <div class="metric-value status-bullish">{sharpe_strat:.2f}</div>
</div>
""", unsafe_allow_html=True)

m4.markdown(f"""
<div class="metric-card">
    <div class="metric-label">Market Sharpe Ratio</div>
    <div class="metric-value">{sharpe_market:.2f}</div>
</div>
""", unsafe_allow_html=True)

# Performance Chart
fig_bt = go.Figure()
fig_bt.add_trace(go.Scatter(x=ticker_df['Date'], y=ticker_df['Cum_Strategy_Return'], name='XGBoost Dynamic Strategy', line=dict(color='#00E676', width=1.8)))
fig_bt.add_trace(go.Scatter(x=ticker_df['Date'], y=ticker_df['Cum_Market_Return'], name='Buy & Hold Benchmark', line=dict(color='#8A92A6', width=1.2, dash='dash')))

fig_bt.update_layout(
    template=plotly_template,
    height=380,
    margin=dict(l=10, r=10, t=20, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1)
)

st.plotly_chart(fig_bt, width="stretch")