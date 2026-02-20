import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pandas as pd
import numpy as np
import yfinance as yf
import ta
import joblib
import matplotlib.pyplot as plt
import os
from datetime import datetime

# ---------------------------------------
# Telegram Function (DEFINE FIRST)
# ---------------------------------------
def send_telegram_alert(message):
    token = os.getenv("BOT_TOKEN")
    chat_id = os.getenv("CHAT_ID")
    
    if token and chat_id:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {"chat_id": chat_id, "text": message}
        requests.post(url, data=payload)

# ---------------------------------------
# Page Config
# ---------------------------------------
st.set_page_config(page_title="XAUUSD AI Trading Dashboard", layout="wide")

st.title("🥇 XAUUSD AI Trading Dashboard")
st.markdown("Live Machine Learning Gold Prediction System")

# ---------------------------------------
# AUTO REFRESH (60 seconds)
# ---------------------------------------
st_autorefresh(interval=60000, key="datarefresh")

# ---------------------------------------
# Load Model
# ---------------------------------------
model = joblib.load("xauusd_rf_model.pkl")

# ---------------------------------------
# Sidebar Settings
# ---------------------------------------
st.sidebar.header("Settings")

interval = st.sidebar.selectbox("Timeframe", ["1h", "4h", "1d"])
period = st.sidebar.selectbox("Data Period", ["1mo", "3mo", "6mo", "1y"])

# ---------------------------------------
# Download Data
# ---------------------------------------
symbol = "GC=F"
data = yf.download(symbol, period=period, interval=interval)

if data.empty:
    st.error("No data retrieved.")
    st.stop()

# ---------------------------------------
# Indicator Calculation
# ---------------------------------------
data["EMA50"] = ta.trend.ema_indicator(data["Close"], window=50)
data["EMA200"] = ta.trend.ema_indicator(data["Close"], window=200)
data["RSI"] = ta.momentum.rsi(data["Close"], window=14)
data["ATR"] = ta.volatility.average_true_range(
    data["High"], data["Low"], data["Close"], window=14
)

data["EMA_Diff"] = data["EMA50"] - data["EMA200"]
data["Momentum"] = data["Close"] - data["Close"].shift(5)
data["Range"] = data["High"] - data["Low"]

data.dropna(inplace=True)

if len(data) < 10:
    st.warning("Not enough data for prediction.")
    st.stop()

# ---------------------------------------
# Prediction
# ---------------------------------------
features = ["EMA50","EMA200","EMA_Diff","RSI","ATR","Momentum","Range"]

latest_data = data[features].iloc[-1:].values

prediction = model.predict(latest_data)[0]
probability = model.predict_proba(latest_data)[0][1]

current_price = data["Close"].iloc[-1]
atr = data["ATR"].iloc[-1]
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# ---------------------------------------
# Trading Logic
# ---------------------------------------
if prediction == 1:
    direction = "BUY 📈"
    stop_loss = current_price - (1.5 * atr)
    take_profit = current_price + (3 * atr)
else:
    direction = "SELL 📉"
    stop_loss = current_price + (1.5 * atr)
    take_profit = current_price - (3 * atr)

# ---------------------------------------
# Display Metrics
# ---------------------------------------
col1, col2, col3 = st.columns(3)
col1.metric("Current Price", f"${current_price:.2f}")
col2.metric("AI Prediction", direction)
col3.metric("Confidence", f"{probability*100:.2f}%")

st.write(f"Last Updated: {timestamp}")

st.subheader("Suggested Risk Management")
st.write(f"Stop Loss: ${stop_loss:.2f}")
st.write(f"Take Profit: ${take_profit:.2f}")

# ---------------------------------------
# Logging
# ---------------------------------------
log_data = {
    "Timestamp": timestamp,
    "Price": current_price,
    "Prediction": direction,
    "Confidence": probability,
    "Stop_Loss": stop_loss,
    "Take_Profit": take_profit
}

log_df = pd.DataFrame([log_data])
log_file = "trade_log.csv"

if os.path.exists(log_file):
    history = pd.read_csv(log_file)
    last_prediction = history.iloc[-1]["Prediction"]
else:
    history = pd.DataFrame()
    last_prediction = None

if direction != last_prediction:
    message = f"""
🥇 XAUUSD AI Signal

Time: {timestamp}
Direction: {direction}
Price: ${current_price:.2f}
Confidence: {probability*100:.2f}%

Stop Loss: ${stop_loss:.2f}
Take Profit: ${take_profit:.2f}
"""
    send_telegram_alert(message)

if os.path.exists(log_file):
    log_df.to_csv(log_file, mode='a', header=False, index=False)
else:
    log_df.to_csv(log_file, index=False)

# ---------------------------------------
# Show History
# ---------------------------------------
st.subheader("Trade History Log")

if os.path.exists(log_file):
    history = pd.read_csv(log_file)
    st.dataframe(history.tail(20))
else:
    st.write("No trade history yet.")

# ---------------------------------------
# Chart
# ---------------------------------------
st.subheader("XAUUSD Price Chart")

fig, ax = plt.subplots(figsize=(12,6))
ax.plot(data.index, data["Close"])
ax.set_title("Gold Price (GC=F)")
ax.set_xlabel("Date")
ax.set_ylabel("Price")
st.pyplot(fig)
