import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from prophet import Prophet
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm
from datetime import datetime

# Streamlit app configuration
st.set_page_config(page_title="🐅 Tiger Population Forecast", layout="centered")
st.title("🐅 Tiger Population Forecast (Next 5 Years)")
st.markdown("Forecast India's tiger population using **Prophet**, **Linear Regression**, **ARIMA**, and **SARIMA** models.")

# Sidebar information
st.sidebar.markdown("### ℹ️ Model Info")
st.sidebar.markdown("- **Prophet**: Captures trend/seasonality")
st.sidebar.markdown("- **ARIMA/SARIMA**: Time-series patterns")
st.sidebar.markdown("- **Linear Regression**: Simple linear trend")

# File uploader
data_file = st.file_uploader("📂 Upload cleaned tiger population CSV", type=["csv"])

# Cache preprocessing
@st.cache_data(show_spinner=False)
def preprocess_data(df):
    yearly = df.groupby("Year", as_index=False)["Tiger Population"].sum()
    yearly["ds"] = pd.to_datetime(yearly["Year"], format="%Y")
    yearly["y"] = yearly["Tiger Population"]
    return yearly

# Cache Prophet
@st.cache_resource(show_spinner=False)
def fit_prophet_model(data):
    model = Prophet()
    model.fit(data[["ds", "y"]])
    return model

# Cache ARIMA/SARIMA
@st.cache_resource(show_spinner=False)
def fit_auto_arima_model(data, seasonal=False):
    return pm.auto_arima(data, seasonal=seasonal, stepwise=True, suppress_warnings=True)

if data_file:
    df = pd.read_csv(data_file)
    yearly = preprocess_data(df)

    st.subheader("📊 EDA")
    st.line_chart(yearly.set_index("Year")["Tiger Population"])

    model_choice = st.radio("Choose Forecast Model:", ["Prophet", "Linear Regression", "ARIMA", "SARIMA"])

    # Limit forecast to 5 years
    future_years = list(range(yearly["Year"].max() + 1, yearly["Year"].max() + 6))
    future_df = pd.DataFrame({"Year": future_years})
    future_df["ds"] = pd.to_datetime(future_df["Year"], format="%Y")

    forecast_df = None

    if model_choice == "Prophet":
        model = fit_prophet_model(yearly)
        forecast = model.predict(future_df[["ds"]])
        forecast_df = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
        forecast_df["Year"] = forecast_df["ds"].dt.year
        merged = pd.merge(yearly, forecast_df, on="Year", how="inner")
        mae = mean_absolute_error(merged["y"], merged["yhat"])
        rmse = np.sqrt(mean_squared_error(merged["y"], merged["yhat"]))
        mse = mean_squared_error(merged["y"], merged["yhat"])

    elif model_choice == "Linear Regression":
        lr = LinearRegression()
        lr.fit(yearly[["Year"]], yearly["y"])
        pred_years = yearly["Year"].tolist() + future_years
        pred_vals = lr.predict(np.array(pred_years).reshape(-1, 1))
        forecast_df = pd.DataFrame({"Year": pred_years, "yhat": pred_vals})
        forecast_df["ds"] = pd.to_datetime(forecast_df["Year"], format="%Y")
        known_preds = forecast_df[forecast_df["Year"] <= yearly["Year"].max()]
        mae = mean_absolute_error(yearly["y"], known_preds["yhat"])
        rmse = np.sqrt(mean_squared_error(yearly["y"], known_preds["yhat"]))
        mse = mean_squared_error(yearly["y"], known_preds["yhat"])
        st.markdown(f"**Equation:** y = {lr.intercept_:.2f} + {lr.coef_[0]:.2f} * Year")

    elif model_choice == "ARIMA":
        model = fit_auto_arima_model(yearly["y"], seasonal=False)
        forecast_vals = model.predict(n_periods=len(future_years))
        forecast_df = pd.DataFrame({"Year": future_years, "yhat": forecast_vals})
        forecast_df["ds"] = pd.to_datetime(forecast_df["Year"], format="%Y")
        mae = rmse = mse = np.nan  # skip metric for unseen test

    elif model_choice == "SARIMA":
        model = fit_auto_arima_model(yearly["y"], seasonal=True)
        forecast_vals = model.predict(n_periods=len(future_years))
        forecast_df = pd.DataFrame({"Year": future_years, "yhat": forecast_vals})
        forecast_df["ds"] = pd.to_datetime(forecast_df["Year"], format="%Y")
        mae = rmse = mse = np.nan

    # Plot
    st.subheader("📈 Forecast")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=yearly["Year"], y=yearly["y"], name="Actual", mode="lines+markers"))
    fig.add_trace(go.Scatter(x=forecast_df["Year"], y=forecast_df["yhat"], name="Forecast", mode="lines+markers"))
    if model_choice == "Prophet":
        fig.add_trace(go.Scatter(x=forecast_df["Year"], y=forecast_df["yhat_upper"], name="Upper", line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=forecast_df["Year"], y=forecast_df["yhat_lower"], name="Lower", fill='tonexty', line=dict(width=0), fillcolor='rgba(0,100,80,0.2)', showlegend=False))
    fig.update_layout(xaxis_title="Year", yaxis_title="Tiger Population", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    # Show metrics
    if not np.isnan(mae):
        st.subheader("📊 Model Fit (Historical)")
        metrics_df = pd.DataFrame({"Metric": ["MAE", "RMSE", "MSE"], "Value": [mae, rmse, mse]})
        st.table(metrics_df)

    # Downloadable forecast
    st.subheader("📥 Download Forecast")
    future_forecast = forecast_df[["Year", "yhat"]].rename(columns={"yhat": "Predicted Population"})
    st.dataframe(future_forecast)
    csv = future_forecast.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", data=csv, file_name="tiger_forecast.csv", mime="text/csv")

else:
    st.info("Please upload a cleaned CSV file to begin.")
