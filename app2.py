import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import plotly.graph_objs as go
import pmdarima as pm
from datetime import datetime

# Streamlit page setup
st.set_page_config(page_title="🐅 Tiger Population Forecast", layout="centered")
st.title("🐅 Tiger Population Forecast")
st.markdown("Forecast India's tiger population using **ARIMA**, **SARIMA**, **Prophet**, and **Linear Regression**.")

# Sidebar info
st.sidebar.markdown("### 📘 Model Descriptions")
st.sidebar.markdown("- **Prophet**: Seasonality + trend modeling by Facebook")
st.sidebar.markdown("- **Linear Regression**: Straight-line trend")
st.sidebar.markdown("- **ARIMA**: Auto-regressive temporal model")
st.sidebar.markdown("- **SARIMA**: Seasonal ARIMA")

# Upload CSV
uploaded_file = st.file_uploader("📂 Upload cleaned CSV file (must have 'Year' and 'Tiger Population')", type="csv")

@st.cache_data(show_spinner=False)
def preprocess_data(df):
    df = df.groupby("Year", as_index=False)["Tiger Population"].sum()
    df["ds"] = pd.to_datetime(df["Year"], format="%Y")
    df["y"] = df["Tiger Population"]
    return df

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if "Year" not in df.columns or "Tiger Population" not in df.columns:
        st.error("❌ File must contain 'Year' and 'Tiger Population' columns.")
        st.stop()

    yearly_df = preprocess_data(df)

    # Display line chart
    st.subheader("📊 Tiger Population Over Time")
    st.line_chart(yearly_df.set_index("Year")["Tiger Population"])

    # Model selection
    model_choice = st.radio("🔍 Select Forecasting Model", ["Prophet", "Linear Regression", "ARIMA", "SARIMA"], index=0)

    # Future forecast range
    future_years = list(range(yearly_df["Year"].max() + 1, yearly_df["Year"].max() + 6))
    future_df = pd.DataFrame({"Year": future_years})
    future_df["ds"] = pd.to_datetime(future_df["Year"], format="%Y")

    forecast_df = None
    mae = rmse = mse = np.nan

    if model_choice == "Prophet":
        with st.spinner("Training Prophet model..."):
            model = Prophet()
            model.fit(yearly_df[["ds", "y"]])
            forecast = model.predict(future_df[["ds"]])
        forecast_df = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
        forecast_df["Year"] = forecast_df["ds"].dt.year
        merged = pd.merge(yearly_df, forecast_df, on="Year", how="inner")
        mae = mean_absolute_error(merged["y"], merged["yhat"])
        rmse = np.sqrt(mean_squared_error(merged["y"], merged["yhat"]))
        mse = mean_squared_error(merged["y"], merged["yhat"])

    elif model_choice == "Linear Regression":
        with st.spinner("Training Linear Regression model..."):
            model = LinearRegression()
            model.fit(yearly_df[["Year"]], yearly_df["y"])
            pred_years = yearly_df["Year"].tolist() + future_years
            pred_vals = model.predict(np.array(pred_years).reshape(-1, 1))
        forecast_df = pd.DataFrame({"Year": pred_years, "yhat": pred_vals})
        forecast_df["ds"] = pd.to_datetime(forecast_df["Year"], format="%Y")
        known_preds = forecast_df[forecast_df["Year"] <= yearly_df["Year"].max()]
        mae = mean_absolute_error(yearly_df["y"], known_preds["yhat"])
        rmse = np.sqrt(mean_squared_error(yearly_df["y"], known_preds["yhat"]))
        mse = mean_squared_error(yearly_df["y"], known_preds["yhat"])
        st.markdown(f"**Linear Equation:** y = {model.intercept_:.2f} + {model.coef_[0]:.2f} * Year")

    elif model_choice == "ARIMA":
        with st.spinner("Training ARIMA model..."):
            model = pm.auto_arima(yearly_df["y"], seasonal=False, stepwise=True, suppress_warnings=True)
            forecast_vals = model.predict(n_periods=len(future_years))
        forecast_df = pd.DataFrame({"Year": future_years, "yhat": forecast_vals})
        forecast_df["ds"] = pd.to_datetime(forecast_df["Year"], format="%Y")

    elif model_choice == "SARIMA":
        with st.spinner("Training SARIMA model..."):
            model = pm.auto_arima(yearly_df["y"], seasonal=True, stepwise=True, suppress_warnings=True)
            forecast_vals = model.predict(n_periods=len(future_years))
        forecast_df = pd.DataFrame({"Year": future_years, "yhat": forecast_vals})
        forecast_df["ds"] = pd.to_datetime(forecast_df["Year"], format="%Y")

    # Plot interactive chart
    st.subheader("📈 Forecast Visualization")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=yearly_df["Year"], y=yearly_df["y"], name="Actual", mode="lines+markers"))
    fig.add_trace(go.Scatter(x=forecast_df["Year"], y=forecast_df["yhat"], name="Forecast", mode="lines+markers"))
    if model_choice == "Prophet":
        fig.add_trace(go.Scatter(x=forecast_df["Year"], y=forecast_df["yhat_upper"], name="Upper Bound", line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=forecast_df["Year"], y=forecast_df["yhat_lower"], name="Lower Bound", fill='tonexty', line=dict(width=0), fillcolor='rgba(0,100,80,0.2)', showlegend=False))
    fig.update_layout(xaxis_title="Year", yaxis_title="Tiger Population", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    # Show metrics if available
    if not np.isnan(mae):
        st.subheader("📏 Model Evaluation")
        st.write("Performance on historical data:")
        st.metric("MAE", f"{mae:.2f}")
        st.metric("RMSE", f"{rmse:.2f}")
        st.metric("MSE", f"{mse:.2f}")

    # Show forecast and download option
    st.subheader("📁 Forecast Output")
    output = forecast_df[["Year", "yhat"]].rename(columns={"yhat": "Predicted Population"})
    st.dataframe(output)
    csv = output.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Forecast CSV", data=csv, file_name="tiger_forecast.csv", mime="text/csv")

else:
    st.info("🔼 Please upload a CSV file to proceed.")
