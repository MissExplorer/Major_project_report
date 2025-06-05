import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import pmdarima as pm
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import datetime

# Page configuration
st.set_page_config(page_title="🐅 Tiger Population Forecast (ARIMA/SARIMA)", layout="centered")
st.title("🐅 Tiger Population Forecast (ARIMA/SARIMA)")
st.markdown("This app uses **ARIMA** and **SARIMA** to model and forecast tiger population trends in India.")

# Sidebar
st.sidebar.title("📘 Model Info")
st.sidebar.markdown("- **ARIMA**: Autoregressive Integrated Moving Average\n- **SARIMA**: Seasonal ARIMA\n\nOnly historical trends are used to make future predictions.")

# File uploader
uploaded_file = st.file_uploader("📂 Upload your cleaned tiger population CSV", type=["csv"])

@st.cache_data(show_spinner=False)
def preprocess(df):
    # Summarize data year-wise
    yearly = df.groupby("Year", as_index=False)["Tiger Population"].sum()
    return yearly

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Validate required columns
    if "Year" not in df.columns or "Tiger Population" not in df.columns:
        st.error("CSV must contain 'Year' and 'Tiger Population' columns.")
        st.stop()

    yearly = preprocess(df)
    years = yearly["Year"]
    values = yearly["Tiger Population"]

    st.subheader("📈 Historical Trend")
    st.line_chart(data=yearly.set_index("Year"))

    model_type = st.radio("Select Model:", ["ARIMA", "SARIMA"], horizontal=True)

    # Future years to forecast
    future_years = list(range(years.max() + 1, years.max() + 6))

    if model_type == "ARIMA":
        with st.spinner("Training ARIMA model..."):
            model = pm.auto_arima(values, seasonal=False, stepwise=True, suppress_warnings=True)
            forecast = model.predict(n_periods=len(future_years))

    elif model_type == "SARIMA":
        with st.spinner("Training SARIMA model..."):
            model = pm.auto_arima(values, seasonal=True, stepwise=True, suppress_warnings=True)
            forecast = model.predict(n_periods=len(future_years))

    # Combine forecast
    forecast_df = pd.DataFrame({
        "Year": future_years,
        "Predicted Population": forecast
    })

    # Plot results
    st.subheader("📉 Forecast Plot")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=years, y=values, mode="lines+markers", name="Actual"))
    fig.add_trace(go.Scatter(x=forecast_df["Year"], y=forecast_df["Predicted Population"], mode="lines+markers", name="Forecast"))
    fig.update_layout(xaxis_title="Year", yaxis_title="Tiger Population", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    # Display forecast table
    st.subheader("📋 Forecast Table")
    st.dataframe(forecast_df)

    # Download option
    csv = forecast_df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Forecast as CSV", csv, file_name="tiger_population_forecast.csv", mime="text/csv")

else:
    st.info("Please upload a cleaned dataset to begin.")
