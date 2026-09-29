import os

import joblib
import pandas as pd
import streamlit as st

D = "data/processed"


@st.cache_data
def segments():
    return pd.read_csv(f"{D}/customer_segments.csv")


@st.cache_data
def forecast():
    f = pd.read_csv(f"{D}/prophet_forecast.csv")
    f["ds"] = pd.to_datetime(f["ds"])
    return f


@st.cache_data
def series():
    t = pd.read_csv(f"{D}/daily_sales_timeseries.csv")
    t.columns = ["ds", "y"]
    t["ds"] = pd.to_datetime(t["ds"])
    return t


@st.cache_data
def inventory():
    i = pd.read_csv(f"{D}/inventory_recommendations.csv")
    i["StockCode"] = i["StockCode"].astype(str)
    return i


@st.cache_data
def churn_scores():
    r = segments().copy()
    model = joblib.load("models/churn_model.pkl")
    r["Churn_Probability"] = model.predict_proba(r[["Frequency_log", "Monetary_log"]])[:, 1]
    return r


def image(path, caption):
    """Show a saved chart only if it exists in the repo."""
    if os.path.exists(path):
        st.image(path, caption=caption)
