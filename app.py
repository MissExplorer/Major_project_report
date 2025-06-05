import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import pmdarima as pm
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import datetime

# Set up the Streamlit page
st.set_page_config(page_title="🐅 Tiger Population Forecast", layout="centered")
st.title("🐅 Tiger Population Forecast using ARIMA & SARIMA")
st.markdown("Upload cleaned tiger population data to forecast with ARIMA and SARIMA models.")

# File uploader
uploaded_file = st.file_uploader("📂 Upload cleaned CSV file (must have 'Year' and 'Tiger Population')", type=["csv"])

@st.cache_data(show_spinner=False)
def preprocess_data(df):
    # Group by year and sum population
    yearly = df.groupby("Year", as_index=False)["Tiger Population"].sum()
    yearly["ds"] = pd.to_datetime(yearly["Year"], format="%Y")
    yearly["y"] = yearly["Tiger Population"]
    return yearly

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Validate required columns
    if df.empty or "Year" not in df.columns or "Tiger Population" not in df.columns:
        st.error("CSV must contain 'Year' and 'Tiger Population' columns.")
        st.stop()

    yearly = preprocess_data(df)

    # Forecasting horizon
    future_years = list(range(yearly["Year"].max() + 1, yearly["Year"].max() + 6))
    forecast_df = None

    # Choose model
    model_type = st.radio("Choose Model", ["ARIMA", "SARIMA"])

    if model_type == "ARIMA":
        with st.spinner("Training ARIMA model..."):
            model = pm.auto_arima(yearly["y"], seasonal=False, stepwise=True, suppress_warnings=True)
            forecast_vals = model.predict(n_periods=len(future_years))
    elif model_type == "SARIMA":
        with st.spinner("Training SARIMA model..."):
            model = pm.auto_arima(yearly["y"], seasonal=True, stepwise=True, suppress_warnings=True)
            forecast_vals = model.predict(n_periods=len(future_years))

    # Prepare forecast data
    forecast_df = pd.DataFrame({
        "Year": future_years,
        "Predicted Population": forecast_vals
    })
    forecast_df["ds"] = pd.to_datetime(forecast_df["Year"], format="%Y")

    # Combine historical and predicted data for plotting
    full_df = pd.concat([
        yearly[["Year", "y"]].rename(columns={"y": "Population"}).assign(Type="Actual"),
        forecast_df[["Year", "Predicted Population"]].rename(columns={"Predicted Population": "Population"}).assign(Type="Forecast")
    ])

    # Plot
    st.subheader("📈 Forecast Plot")
    fig = go.Figure()
    for label, group in full_df.groupby("Type"):
        fig.add_trace(go.Scatter(x=group["Year"], y=group["Population"], mode="lines+markers", name=label))
    fig.update_layout(xaxis_title="Year", yaxis_title="Tiger Population", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    # Model metrics (on historical data only)
    st.subheader("📊 Model Evaluation")
    backtest = model.predict_in_sample()
    mae = mean_absolute_error(yearly["y"], backtest)
    rmse = np.sqrt(mean_squared_error(yearly["y"], backtest))
    mse = mean_squared_error(yearly["y"], backtest)
    metrics_df = pd.DataFrame({
        "Metric": ["MAE", "RMSE", "MSE"],
        "Value": [mae, rmse, mse]
    })
    st.table(metrics_df)

    # Forecast table and download
    st.subheader("📥 Download Forecast")
    st.dataframe(forecast_df[["Year", "Predicted Population"]])
    csv = forecast_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", data=csv, file_name="tiger_arima_sarima_forecast.csv", mime="text/csv")

else:
    st.info("Please upload a cleaned CSV file to begin.")
