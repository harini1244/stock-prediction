import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

st.title('📈 Stock Trend Prediction App')

# User input
user_input = st.text_input('Enter Stock Ticker (e.g., AAPL, TSLA, MSFT)', 'AAPL')

# Date range
start_date = '2020-01-01'
end_date = '2025-02-01'

# Load data
df = yf.download(user_input, start=start_date, end=end_date)

# Display the data
st.subheader('Stock Data from 2020 to 2025')
st.write(df.tail())

# --- Chart 1: Closing Price vs Time ---
st.subheader('1. Closing Price vs Time Chart')
fig1 = plt.figure(figsize=(12, 6))
plt.plot(df['Close'], label='Closing Price')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.title('Closing Price Over Time')
plt.legend()
st.pyplot(fig1)

st.markdown("""
This chart shows the **daily closing price** of the stock over the selected time period.
You can observe periods where the stock is in an **uptrend** and **downtrend**.

- An **uptrend** is marked by higher highs and higher lows.
- A **downtrend** shows lower highs and lower lows.
""")
st.info("""
💡 **Beginner Insight**: 
If the stock shows a general upward direction over a long period, it can be a **positive sign** for long-term investors. However, sharp dips suggest caution.
""")


# --- Chart 2: Closing Price with 100-Day MA ---
st.subheader('2. Closing Price with 100-Day Moving Average')
ma100 = df['Close'].rolling(100).mean()
fig2 = plt.figure(figsize=(12, 6))
plt.plot(df['Close'], label='Closing Price')
plt.plot(ma100, 'r', label='100-Day MA')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.title('Closing Price vs 100-Day Moving Average')
plt.legend()
st.pyplot(fig2)

st.markdown("""
The **100-day Moving Average (MA)** helps smooth short-term volatility.

- If the **price is above** the MA: strong upward trend.
- If **below**: it may indicate the start of a downtrend.
""")
st.info("""
💡 **Beginner Insight**: 
If the **current price is above the 100-day MA**, it suggests **positive momentum**.
""")


# --- Chart 3: 100 & 200-Day Moving Averages ---
st.subheader('3. Closing Price with 100 & 200-Day Moving Averages')
ma200 = df['Close'].rolling(200).mean()
fig3 = plt.figure(figsize=(12, 6))
plt.plot(df['Close'], label='Closing Price')
plt.plot(ma100, 'r', label='100-Day MA')
plt.plot(ma200, 'g', label='200-Day MA')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.title('Closing Price vs 100 & 200-Day Moving Averages')
plt.legend()
st.pyplot(fig3)

st.markdown("""
- A **Golden Cross**: 100-day MA crosses **above** the 200-day — bullish sign.
- A **Death Cross**: 100-day MA crosses **below** the 200-day — bearish signal.
""")
st.info("""
💡 **Beginner Insight**: 
Golden Cross = Buy Signal 📈 | Death Cross = Caution ⚠️
""")


# --- Prepare data for LSTM prediction ---
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):])

scaler = MinMaxScaler(feature_range=(0, 1))
data_training_array = scaler.fit_transform(data_training)

# Load pre-trained model
model = load_model('keras_model.h5')

# Prepare test data
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.transform(final_df)

x_test = []
y_test = []
for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

# Make predictions
y_predicted = model.predict(x_test)

# Reverse scaling
scale_factor = 1 / scaler.scale_[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

# --- Chart 4: Prediction vs Actual ---
st.subheader('4. Predicted vs Actual Stock Price')
fig4 = plt.figure(figsize=(12, 6))
plt.plot(y_test, label='Actual Price', color='blue')
plt.plot(y_predicted, label='Predicted Price', color='red')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.title('LSTM Model Prediction vs Actual Price')
plt.legend()
st.pyplot(fig4)

st.markdown("""
This graph compares the model’s **predicted prices** (red) with the **actual prices** (blue).
""")
st.info("""
💡 **Beginner Insight**: 
If the predicted line follows the actual closely, the model has learned well.
However, always combine predictions with other analysis tools.
""")


# --- Chart 5: Explainable AI Style Investment Insight ---
st.subheader("5. Investment Insight Based on Prediction")

if y_predicted[-1] > y_test[-1]:
    trend = "📈 The model predicts an upward trend."
    advice = "This may indicate a potential opportunity to invest."
else:
    trend = "📉 The model predicts a downward trend."
    advice = "You might want to hold off on investing until signs of recovery."

st.success(trend)
st.warning(advice)

st.markdown("""
### How to Interpret This:
- If the **predicted price** at the end is **higher than the actual**, the model expects growth.
- If it's **lower**, it may suggest a decline.

Use this along with:
- Company financials
- Recent news
- Risk level & goals
""")
st.info("""
🧠 **Note**: This model uses historical price patterns, not real-world events. Use it as **supportive insight**, not as a guarantee.
""")
