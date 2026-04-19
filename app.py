import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# =========================
# PAGE
# =========================
st.set_page_config(layout="wide")
st.title("📊 Predictive Forecasting of HHS Care Load")

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv("HHS_Unaccompanied_Alien_Children_Program.csv")

    df["Date"] = pd.to_datetime(df["Date"])

    df["Children in HHS Care"] = df["Children in HHS Care"].astype(str).str.replace(",", "")
    df["Children in HHS Care"] = pd.to_numeric(df["Children in HHS Care"], errors="coerce")

    df["Children discharged from HHS Care"] = pd.to_numeric(
        df["Children discharged from HHS Care"], errors="coerce"
    )

    df["Children transferred out of CBP custody"] = pd.to_numeric(
        df["Children transferred out of CBP custody"], errors="coerce"
    )

    df = df.dropna()
    df = df.set_index("Date").sort_index()

    df = df.asfreq("D")
    df = df.ffill()

    return df

data = load_data()

# =========================
# FEATURE ENGINEERING
# =========================
df = data.copy()

df["lag1"] = df["Children in HHS Care"].shift(1)
df["lag7"] = df["Children in HHS Care"].shift(7)
df["rolling7"] = df["Children in HHS Care"].rolling(7).mean()

df["dayofweek"] = df.index.dayofweek
df["month"] = df.index.month

df["net_flow"] = (
    df["Children transferred out of CBP custody"]
    - df["Children discharged from HHS Care"]
)

df = df.dropna()

# =========================
# SIDEBAR
# =========================
model_choice = st.sidebar.selectbox(
    "Select Model",
    ["Naive", "Moving Average", "ARIMA", "Exponential Smoothing", "Random Forest", "Gradient Boosting"]
)

horizon = st.sidebar.slider("Forecast Horizon", 7, 60, 30)

# =========================
# SPLIT
# =========================
train = df.iloc[:-horizon]
test = df.iloc[-horizon:]

features = ["lag1","lag7","rolling7","dayofweek","month","net_flow"]

X_train = train[features]
y_train = train["Children in HHS Care"]

X_test = test[features]
y_test = test["Children in HHS Care"]

# =========================
# MODEL LOGIC (CLEAN FIX)
# =========================
if model_choice == "Naive":
    preds = test["lag1"]

elif model_choice == "Moving Average":
    preds = test["Children in HHS Care"].rolling(3).mean()

elif model_choice == "ARIMA":
    model = ARIMA(train["Children in HHS Care"], order=(1,1,1))
    model_fit = model.fit()
    preds = model_fit.forecast(steps=len(test))
    preds.index = test.index

elif model_choice == "Exponential Smoothing":
    model = ExponentialSmoothing(train["Children in HHS Care"], trend="add")
    model_fit = model.fit()
    preds = model_fit.forecast(len(test))
    preds.index = test.index

elif model_choice == "Random Forest":
    model = RandomForestRegressor(random_state=42)
    preds = model.fit(X_train, y_train).predict(X_test)

elif model_choice == "Gradient Boosting":
    model = GradientBoostingRegressor(random_state=42)
    preds = model.fit(X_train, y_train).predict(X_test)

# =========================
# ENSURE SERIES FORMAT
# =========================
preds = pd.Series(preds, index=y_test.index)

# =========================
# REMOVE NaN SAFELY
# =========================
df_eval = pd.DataFrame({"actual": y_test, "pred": preds}).dropna()

y_clean = df_eval["actual"]
p_clean = df_eval["pred"]

# =========================
# METRICS
# =========================
mae = mean_absolute_error(y_clean, p_clean)
rmse = np.sqrt(mean_squared_error(y_clean, p_clean))
mape = np.mean(np.abs((y_clean - p_clean)/y_clean))*100
accuracy = 100 - mape

st.subheader("📌 Model Metrics")
st.write("Model:", model_choice)
st.write("MAE:", round(mae,2))
st.write("RMSE:", round(rmse,2))
st.write("MAPE:", round(mape,2), "%")
st.write("Accuracy:", round(accuracy,2), "%")

# =========================
# PLOT
# =========================
st.subheader("📈 Forecast vs Actual")

fig, ax = plt.subplots(figsize=(12,5))

ax.plot(y_clean.index, y_clean, label="Actual")
ax.plot(y_clean.index, p_clean, label=f"Prediction ({model_choice})")

ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

plt.xticks(rotation=45)
plt.tight_layout()
ax.legend()

st.pyplot(fig)