# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from src.data_loader import load_data
from src.feature_engineering import engineer_features, add_features
from src.model_training import load_model, predict_proba_direction
from src.backtest import backtest_with_sltp

st.set_page_config(page_title="BitPredictor Dashboard", layout="wide")

@st.cache_data
def load_data_and_features(path):
    df = load_data(path)
    df_feats = engineer_features(df)
    df_feats = add_features(df_feats)
    return df, df_feats

@st.cache_resource
def load_trained_model(path="models/model.pkl"):
    return load_model(path)

# Sidebar controls
st.sidebar.header("Controls")
threshold = st.sidebar.slider("Probability threshold", 0.30, 0.80, 0.50, 0.05)
fee_rate = st.sidebar.slider("Trading fee rate", 0.0, 0.005, 0.001, 0.0005)
slippage_rate = st.sidebar.slider("Slippage rate", 0.0, 0.005, 0.001, 0.0005)
sl_pct = st.sidebar.slider("Stop-loss %", 0.0, 0.02, 0.005, 0.001)
tp_pct = st.sidebar.slider("Take-profit %", 0.0, 0.05, 0.01, 0.005)
n_last = st.sidebar.number_input("Show last N bars", min_value=50, max_value=500, value=200, step=50)

# Load data, features, and model
df, df_feats = load_data_and_features("data/bitpredict.csv")
model = load_trained_model()

# Prepare X for prediction
X = df_feats.drop(columns=['close','returns','log_returns'], errors='ignore')

# Generate signals
signals = predict_proba_direction(model, X, threshold=threshold)

# Backtest
bt = backtest_with_sltp(
    df_feats, signals,
    fee_rate=fee_rate,
    slippage_rate=slippage_rate,
    sl_pct=sl_pct,
    tp_pct=tp_pct,
    initial_capital=10000
)

# Layout
col1, col2 = st.columns((2,3))

with col1:
    st.subheader("Latest Prediction")
    last_signal = signals[-1]
    direction = "📈 UP" if last_signal==1 else "📉 DOWN"
    st.metric(label="Predicted Direction", value=direction)

    st.subheader(f"Last {n_last} Bars Price Chart")
    fig, ax = plt.subplots(figsize=(6,3))
    ax.plot(df['close'].iloc[-n_last:], label='Close')
    ax.set_title("Close Price")
    ax.legend()
    st.pyplot(fig)

with col2:
    st.subheader("Equity Curve vs Buy-and-Hold")
    fig2, ax2 = plt.subplots(figsize=(8,4))
    ax2.plot(bt['cumulative_market_return'], label='Buy & Hold')
    ax2.plot(bt['cumulative_strategy_return'], label='Strategy')
    ax2.set_ylabel("Cumulative Return")
    ax2.legend()
    st.pyplot(fig2)

st.markdown("---")
st.subheader("Performance Summary")
col3, col4, col5 = st.columns(3)
with col3:
    st.metric("Buy & Hold Return ×", f"{bt['cumulative_market_return'].iloc[-1]:.2f}")
with col4:
    st.metric("Strategy Return ×", f"{bt['cumulative_strategy_return'].iloc[-1]:.2f}")
with col5:
    trades = int(signals.sum())
    st.metric("Total Long Trades", f"{trades}")

st.markdown("**Adjust the threshold, fees, and SL/TP in the sidebar to see how strategy performance changes in real time.**")
