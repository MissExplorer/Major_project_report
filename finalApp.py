import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from prophet import Prophet
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pmdarima as pm

# Set up the Streamlit page
st.set_page_config(page_title="🐅 Tiger Population Forecast", layout="centered")
st.title("🐅 Tiger Population Forecast")
st.markdown("Forecast India's tiger population using **Prophet**, **Linear Regression**, **ARIMA**, and **SARIMA** models.")

# Sidebar model info
st.sidebar.markdown("### ℹ️ Model Info")
st.sidebar.markdown("- **Prophet**: Captures trend/seasonality")
st.sidebar.markdown("- **ARIMA/SARIMA**: Time-series models")
st.sidebar.markdown("- **Linear Regression**: Linear trend")

# File uploader
uploaded_file = st.file_uploader("📂 Upload cleaned tiger population CSV", type=["csv"])

@st.cache_data(show_spinner=False)
def preprocess_data(df):
    # Aggregate population by year
    yearly = df.groupby("Year", as_index=False)["Tiger Population"].sum()
    yearly["ds"] = pd.to_datetime(yearly["Year"], format="%Y")  # For Prophet
    yearly["y"] = yearly["Tiger Population"]
    return yearly

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if df.empty or "Year" not in df.columns or "Tiger Population" not in df.columns:
        st.error("Uploaded file must contain 'Year' and 'Tiger Population' columns.")
        st.stop()

    yearly = preprocess_data(df)

    # Line chart of actual values
    st.subheader("📊 Trend Overview")
    st.line_chart(yearly.set_index("Year")["Tiger Population"])

    # Model selection
    model_choice = st.radio("Choose Forecast Model:", ["Prophet", "Linear Regression", "ARIMA", "SARIMA"], index=0)

    # Forecast up to 5 years ahead
    future_years = list(range(yearly["Year"].max() + 1, yearly["Year"].max() + 6))
    future_df = pd.DataFrame({"Year": future_years})
    future_df["ds"] = pd.to_datetime(future_df["Year"], format="%Y")

    forecast_df = None

    if model_choice == "Prophet":
        with st.spinner("Training Prophet model..."):
            model = Prophet()
            model.fit(yearly[["ds", "y"]])
            forecast = model.predict(future_df[["ds"]])
        forecast_df = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
        forecast_df["Year"] = forecast_df["ds"].dt.year
        merged = pd.merge(yearly, forecast_df, on="Year", how="inner")
        # Compute metrics but don't show
        _ = mean_absolute_error(merged["y"], merged["yhat"])
        _ = np.sqrt(mean_squared_error(merged["y"], merged["yhat"]))
        _ = mean_squared_error(merged["y"], merged["yhat"])

    elif model_choice == "Linear Regression":
        with st.spinner("Training Linear Regression model..."):
            lr = LinearRegression()
            lr.fit(yearly[["Year"]], yearly["y"])
            pred_years = yearly["Year"].tolist() + future_years
            pred_vals = lr.predict(np.array(pred_years).reshape(-1, 1))
        forecast_df = pd.DataFrame({"Year": pred_years, "yhat": pred_vals})
        forecast_df["ds"] = pd.to_datetime(forecast_df["Year"], format="%Y")
        known_preds = forecast_df[forecast_df["Year"] <= yearly["Year"].max()]
        _ = mean_absolute_error(yearly["y"], known_preds["yhat"])
        _ = np.sqrt(mean_squared_error(yearly["y"], known_preds["yhat"]))
        _ = mean_squared_error(yearly["y"], known_preds["yhat"])

    elif model_choice == "ARIMA":
        with st.spinner("Training ARIMA model..."):
            model = pm.auto_arima(yearly["y"], seasonal=False, stepwise=True, suppress_warnings=True)
            forecast_vals = model.predict(n_periods=len(future_years))
        forecast_df = pd.DataFrame({"Year": future_years, "yhat": forecast_vals})
        forecast_df["ds"] = pd.to_datetime(forecast_df["Year"], format="%Y")

    elif model_choice == "SARIMA":
        with st.spinner("Training SARIMA model..."):
            model = pm.auto_arima(yearly["y"], seasonal=True, stepwise=True, suppress_warnings=True)
            forecast_vals = model.predict(n_periods=len(future_years))
        forecast_df = pd.DataFrame({"Year": future_years, "yhat": forecast_vals})
        forecast_df["ds"] = pd.to_datetime(forecast_df["Year"], format="%Y")

    # 📈 Plot results
    st.subheader("📈 Forecast")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=yearly["Year"], y=yearly["y"], name="Actual", mode="lines+markers"))
    fig.add_trace(go.Scatter(x=forecast_df["Year"], y=forecast_df["yhat"], name="Forecast", mode="lines+markers"))
    if model_choice == "Prophet":
        fig.add_trace(go.Scatter(x=forecast_df["Year"], y=forecast_df["yhat_upper"], name="Upper", line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=forecast_df["Year"], y=forecast_df["yhat_lower"], name="Lower", fill='tonexty', line=dict(width=0), fillcolor='rgba(0,100,80,0.2)', showlegend=False))
    fig.update_layout(xaxis_title="Year", yaxis_title="Tiger Population", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    # 📥 Downloadable CSV
    st.subheader("📥 Download Forecast")
    future_forecast = forecast_df[["Year", "yhat"]].rename(columns={"yhat": "Predicted Population"})
    st.dataframe(future_forecast)
    csv = future_forecast.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", data=csv, file_name="tiger_forecast.csv", mime="text/csv")

else:
    st.info("Please upload a cleaned CSV file to begin.")
