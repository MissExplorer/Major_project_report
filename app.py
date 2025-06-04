import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from prophet import Prophet
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm
from sklearn.model_selection import cross_val_score
from datetime import datetime

# Streamlit app configuration
st.set_page_config(page_title="🐅 Tiger Population Forecast", layout="centered")
st.title("🐅 Tiger Population Forecast (Up to 2030)")
st.markdown("This tool forecasts India's tiger population using **Prophet**, **Linear Regression**, **ARIMA**, and **SARIMA** models.")

# Sidebar model information
st.sidebar.markdown("### ℹ️ Model Guidance")
st.sidebar.markdown("- **Prophet**: Captures trend and seasonality")
st.sidebar.markdown("- **ARIMA/SARIMA**: Best for time series with autocorrelation")
st.sidebar.markdown("- **Linear Regression**: Good for simple linear trends")

# File uploader for CSV
data_file = st.file_uploader("📂 Upload your cleaned tiger population CSV", type=["csv"])

if data_file:
    df = pd.read_csv(data_file)

    # Preprocess uploaded data
    yearly = df.groupby("Year", as_index=False)["Tiger Population"].sum()
    yearly["ds"] = pd.to_datetime(yearly["Year"], format="%Y")
    yearly["y"] = yearly["Tiger Population"]

    # EDA visualization
    st.subheader("📊 Exploratory Data Analysis (EDA)")
    st.line_chart(yearly.set_index("Year")["Tiger Population"])

    # Model selection radio button
    model_choice = st.radio("Select model for forecasting:", ["Prophet", "Linear Regression", "ARIMA", "SARIMA"])

    # Forecast year range and placeholder for forecast
    future_years = list(range(yearly["Year"].max() + 1, 2031))
    future_df = pd.DataFrame({"Year": future_years})
    future_df["ds"] = pd.to_datetime(future_df["Year"], format="%Y")

    forecast_df = None

    if model_choice == "Prophet":
        model = Prophet()
        model.fit(yearly[["ds", "y"]])
        full_future = pd.concat([yearly[["ds"]], future_df[["ds"]]], ignore_index=True)
        forecast = model.predict(full_future)
        forecast_df = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
        forecast_df["Year"] = forecast_df["ds"].dt.year
        merged = pd.merge(yearly, forecast_df, on="Year", how="left")
        merged = merged[merged["Year"] <= yearly["Year"].max()]
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
        scores = cross_val_score(lr, yearly[["Year"]], yearly["y"], scoring='neg_mean_squared_error', cv=10)
        st.markdown(f"**Cross-validated RMSE (avg):** {np.sqrt(-scores.mean()):.2f}")

    elif model_choice == "ARIMA":
        model = pm.auto_arima(yearly["y"], seasonal=False, stepwise=True, suppress_warnings=True)
        forecast_vals = model.predict(n_periods=len(future_years))
        full_years = yearly["Year"].tolist() + future_years
        full_vals = list(yearly["y"]) + list(forecast_vals)
        forecast_df = pd.DataFrame({"Year": full_years, "yhat": full_vals})
        forecast_df["ds"] = pd.to_datetime(forecast_df["Year"], format="%Y")
        known_preds = forecast_df[forecast_df["Year"] <= yearly["Year"].max()]
        mae = mean_absolute_error(yearly["y"], known_preds["yhat"])
        rmse = np.sqrt(mean_squared_error(yearly["y"], known_preds["yhat"]))
        mse = mean_squared_error(yearly["y"], known_preds["yhat"])

    elif model_choice == "SARIMA":
        model = pm.auto_arima(yearly["y"], seasonal=True, m=2, stepwise=True, suppress_warnings=True)
        forecast_vals = model.predict(n_periods=len(future_years))
        full_years = yearly["Year"].tolist() + future_years
        full_vals = list(yearly["y"]) + list(forecast_vals)
        forecast_df = pd.DataFrame({"Year": full_years, "yhat": full_vals})
        forecast_df["ds"] = pd.to_datetime(forecast_df["Year"], format="%Y")
        known_preds = forecast_df[forecast_df["Year"] <= yearly["Year"].max()]
        mae = mean_absolute_error(yearly["y"], known_preds["yhat"])
        rmse = np.sqrt(mean_squared_error(yearly["y"], known_preds["yhat"]))
        mse = mean_squared_error(yearly["y"], known_preds["yhat"])

    # Filter year range for plot
    year_range = st.slider("Select year range for visualization", int(forecast_df["Year"].min()), int(forecast_df["Year"].max()), (2010, 2030))
    filtered = forecast_df[(forecast_df["Year"] >= year_range[0]) & (forecast_df["Year"] <= year_range[1])]

    # Forecast plot
    st.subheader("📈 Forecast vs Actual")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=yearly["Year"], y=yearly["y"], mode="lines+markers", name="Actual"))
    fig.add_trace(go.Scatter(x=filtered["Year"], y=filtered["yhat"], mode="lines+markers", name="Forecast"))

    if model_choice == "Prophet":
        fig.add_trace(go.Scatter(x=filtered["Year"], y=filtered["yhat_upper"], name="Upper Bound", line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=filtered["Year"], y=filtered["yhat_lower"], name="Lower Bound", fill='tonexty', line=dict(width=0), fillcolor='rgba(0,100,80,0.2)', showlegend=False))

    fig.update_layout(xaxis_title="Year", yaxis_title="Tiger Population", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    # Display evaluation metrics
    st.subheader("📊 Model Fit on Historical Data")
    metrics_df = pd.DataFrame({
        "Metric": ["MAE", "RMSE", "MSE"],
        "Value": [mae, rmse, mse]
    })
    st.table(metrics_df)

    # Download forecast CSV
    st.subheader("📥 Download Forecast")
    future_forecast = forecast_df[forecast_df["Year"].isin(future_years)][["Year", "yhat"]]
    future_forecast = future_forecast.rename(columns={"yhat": "Predicted Population"})
    st.dataframe(future_forecast)
    csv = future_forecast.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", data=csv, file_name="tiger_population_forecast_2030.csv", mime="text/csv")

else:
    st.info("Upload a cleaned CSV file to start.")
