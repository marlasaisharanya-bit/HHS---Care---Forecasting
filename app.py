import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose

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

    # continuity
    df = df.asfreq("D")
    df = df.ffill()

    return df

data = load_data()

# =========================
# DECOMPOSITION
# =========================
st.subheader("📉 Time Series Decomposition")
decomp = seasonal_decompose(data["Children in HHS Care"], model="additive", period=30)
st.pyplot(decomp.plot())

# =========================
# FEATURE ENGINEERING
# =========================
df = data.copy()

df["lag1"] = df["Children in HHS Care"].shift(1)
df["lag7"] = df["Children in HHS Care"].shift(7)
df["lag14"] = df["Children in HHS Care"].shift(14)

df["rolling7"] = df["Children in HHS Care"].rolling(7).mean()
df["rolling14"] = df["Children in HHS Care"].rolling(14).mean()

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
st.sidebar.header("Controls")

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

features = ["lag1","lag7","lag14","rolling7","rolling14","dayofweek","month","net_flow"]

X_train = train[features]
y_train = train["Children in HHS Care"]

X_test = test[features]
y_test = test["Children in HHS Care"]

# =========================
# WALK FORWARD (RF/GB)
# =========================
def walk_forward(df, features, target, model, window=100):
    preds, actuals, idx = [], [], []

    for i in range(window, len(df)):
        train = df.iloc[:i]
        test = df.iloc[i:i+1]

        model.fit(train[features], train[target])
        pred = model.predict(test[features])[0]

        preds.append(pred)
        actuals.append(test[target].values[0])
        idx.append(test.index[0])

    return pd.Series(preds, index=idx), pd.Series(actuals, index=idx)

# =========================
# MODELS
# =========================
if model_choice == "Naive":
    preds = test["lag1"]

elif model_choice == "Moving Average":
    preds = test["Children in HHS Care"].rolling(3).mean()

elif model_choice == "ARIMA":
    model = ARIMA(train["Children in HHS Care"], order=(2,1,2))
    model_fit = model.fit()
    preds = model_fit.forecast(len(test))
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
# CLEAN FOR METRICS
# =========================
preds = pd.Series(preds, index=y_test.index)

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

# =========================
# HORIZON METRICS
# =========================
st.subheader("📊 Horizon Error")

st.write("7-day MAE:", round(mean_absolute_error(y_clean[-7:], p_clean[-7:]),2))
st.write("30-day MAE:", round(mean_absolute_error(y_clean[-30:], p_clean[-30:]),2))

# =========================
# PLOT
# =========================
st.subheader("📈 Forecast vs Actual")

fig, ax = plt.subplots(figsize=(12,5))
ax.plot(y_clean.index, y_clean, label="Actual")
ax.plot(y_clean.index, p_clean, label="Prediction")

ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

plt.xticks(rotation=45)
plt.tight_layout()
ax.legend()
st.pyplot(fig)

# =========================
# CONFIDENCE INTERVAL
# =========================
st.subheader("📊 Confidence Interval")

std = np.std(y_clean - p_clean)
upper = p_clean + 1.96*std
lower = p_clean - 1.96*std

fig2, ax2 = plt.subplots(figsize=(12,5))
ax2.plot(y_clean.index, y_clean)
ax2.plot(y_clean.index, p_clean)
ax2.fill_between(y_clean.index, lower, upper, alpha=0.2)

ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig2)

# =========================
# DISCHARGE MODEL
# =========================
st.subheader("📉 Discharge Forecast")

d = data["Children discharged from HHS Care"]

df_d = pd.DataFrame({
    "target": d,
    "lag1": d.shift(1),
    "lag7": d.shift(7)
}).dropna()

train_d = df_d.iloc[:-horizon]
test_d = df_d.iloc[-horizon:]

model_d = RandomForestRegressor(random_state=42)
model_d.fit(train_d[["lag1","lag7"]], train_d["target"])

pred_d = model_d.predict(test_d[["lag1","lag7"]])

fig3, ax3 = plt.subplots(figsize=(12,5))
ax3.plot(test_d.index, test_d["target"], label="Actual")
ax3.plot(test_d.index, pred_d, label="Prediction")

ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
plt.xticks(rotation=45)
plt.tight_layout()
ax3.legend()
st.pyplot(fig3)

# =========================
# KPIs
# =========================
st.subheader("🚨 KPIs")

threshold = y_clean.mean() + 2*y_clean.std()

surge_days = np.where(p_clean > threshold)[0]
lead_time = surge_days[0] if len(surge_days)>0 else 0

st.write("Forecast Accuracy:", round(accuracy,2), "%")
st.write("Capacity Breach Probability:", round(np.mean(p_clean>threshold)*100,2), "%")
st.write("Surge Lead Time:", lead_time, "days")

stability = np.std(p_clean)/np.mean(p_clean)
st.write("Forecast Stability Index:", round(stability,3))

# =========================
# MODEL COMPARISON
# =========================
st.subheader("🔍 Model Comparison")

models = {
    "RF": RandomForestRegressor(),
    "GB": GradientBoostingRegressor()
}

fig_cmp, ax_cmp = plt.subplots(figsize=(12,5))

for name,m in models.items():
    m.fit(X_train,y_train)
    p = m.predict(X_test)
    ax_cmp.plot(y_test.index, p, label=name)

ax_cmp.plot(y_test.index, y_test, label="Actual", linewidth=2)
ax_cmp.legend()

plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig_cmp)