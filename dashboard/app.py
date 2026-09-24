import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

st.set_page_config(page_title="RetailPulse Analytics", layout="wide")

st.title("RetailPulse: AI-Powered Customer Analytics & Demand Forecasting")
st.markdown("Predictive Demand - Customer Segmentation - Churn Analysis - Inventory Optimization")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Demand Forecasting", "Customer Segments", "Churn Risk", "Inventory"])

with tab1:
    st.header("Business Overview")
    rfm = pd.read_csv("data/processed/customer_segments.csv")
    total_customers = len(rfm)
    total_revenue = rfm["Monetary"].sum()
    champions_revenue = rfm[rfm["Segment_Label"] == "Champions"]["Monetary"].sum()
    champions_pct_customers = (rfm["Segment_Label"] == "Champions").mean() * 100
    champions_pct_revenue = (champions_revenue / total_revenue) * 100

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", f"{total_customers:,}")
    col2.metric("Total Revenue", f"GBP {total_revenue:,.0f}")
    col3.metric("Champions (% customers)", f"{champions_pct_customers:.1f}%")
    col4.metric("Champions (% revenue)", f"{champions_pct_revenue:.1f}%")

    st.info(f"Key Insight: {champions_pct_customers:.1f}% of customers generate {champions_pct_revenue:.1f}% of total revenue.")

    st.subheader("Customer Segment Distribution")
    seg_counts = rfm["Segment_Label"].value_counts()
    fig, ax = plt.subplots(figsize=(10, 4))
    seg_counts.plot(kind="bar", ax=ax, color="steelblue")
    ax.set_ylabel("Number of Customers")
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)

with tab2:
    st.header("Demand Forecasting")
    forecast = pd.read_csv("data/processed/prophet_forecast.csv")
    forecast["ds"] = pd.to_datetime(forecast["ds"])
    ts = pd.read_csv("data/processed/daily_sales_timeseries.csv")
    ts.columns = ["ds", "y"]
    ts["ds"] = pd.to_datetime(ts["ds"])

    st.metric("Model MAPE (Prophet)", "24.06%")

    days_to_show = st.slider("Days of history to display", 30, 400, 120)
    recent_actual = ts.tail(days_to_show)
    recent_forecast = forecast[forecast["ds"] >= recent_actual["ds"].min()]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(recent_actual["ds"], recent_actual["y"], label="Actual Sales", color="black")
    ax.plot(recent_forecast["ds"], recent_forecast["yhat"], label="Forecast", color="orange")
    ax.fill_between(recent_forecast["ds"], recent_forecast["yhat_lower"], recent_forecast["yhat_upper"], alpha=0.2, color="orange")
    ax.set_ylabel("Daily Sales (GBP)")
    ax.legend()
    st.pyplot(fig)

    st.subheader("What-If: Promotion Impact Simulator")
    boost_pct = st.slider("Simulated promotion boost (%)", 0, 50, 0)
    last_30 = forecast.tail(30).copy()
    last_30["yhat_boosted"] = last_30["yhat"] * (1 + boost_pct/100)
    baseline_sum = last_30["yhat"].sum()
    boosted_sum = last_30["yhat_boosted"].sum()
    st.write(f"Projected next-30-day revenue with {boost_pct}% boost: GBP {boosted_sum:,.0f} (baseline: GBP {baseline_sum:,.0f})")

with tab3:
    st.header("Customer Segmentation (RFM + K-Means)")
    selected_segment = st.selectbox("Filter by segment", ["All"] + list(rfm["Segment_Label"].unique()))

    if selected_segment != "All":
        filtered = rfm[rfm["Segment_Label"] == selected_segment]
    else:
        filtered = rfm

    st.dataframe(filtered[["Customer ID", "Recency", "Frequency", "Monetary", "Segment_Label"]].head(50))

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots()
        ax.scatter(rfm["Frequency"], rfm["Monetary"], c=rfm["Segment"], cmap="viridis", alpha=0.5)
        ax.set_xlabel("Frequency")
        ax.set_ylabel("Monetary (GBP)")
        st.pyplot(fig)
    with col2:
        segment_summary = rfm.groupby("Segment_Label").agg(
            Avg_Recency=("Recency", "mean"),
            Avg_Frequency=("Frequency", "mean"),
            Avg_Monetary=("Monetary", "mean"),
            Count=("Customer ID", "count")
        ).round(1)
        st.dataframe(segment_summary)

    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button("Download Segment Data (CSV)", csv, "customer_segments.csv", "text/csv")

with tab4:
    st.header("Churn Risk Analysis")
    churn_model = joblib.load("models/churn_model.pkl")
    rfm["Churn_Probability"] = churn_model.predict_proba(rfm[["Frequency_log", "Monetary_log"]])[:, 1]

    st.metric("Model AUC-ROC", "0.77")

    risk_threshold = st.slider("Churn risk threshold", 0.0, 1.0, 0.5)
    at_risk = rfm[rfm["Churn_Probability"] >= risk_threshold].sort_values("Churn_Probability", ascending=False)

    st.write(f"Customers above risk threshold: {len(at_risk)}")
    st.dataframe(at_risk[["Customer ID", "Recency", "Frequency", "Monetary", "Churn_Probability"]].head(50))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(rfm["Churn_Probability"], bins=30, color="salmon")
    ax.axvline(risk_threshold, color="black", linestyle="--")
    ax.set_xlabel("Churn Probability")
    ax.set_ylabel("Number of Customers")
    st.pyplot(fig)

with tab5:
    st.header("Inventory Optimization")
    inv = pd.read_csv("data/processed/inventory_recommendations.csv")
    st.write(f"Recommendations generated for {len(inv)} products")
    st.dataframe(inv.sort_values("recommended_order_qty", ascending=False).head(50))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(inv["recommended_order_qty"], bins=30, color="seagreen")
    ax.set_xlabel("Recommended Order Quantity")
    ax.set_ylabel("Number of Products")
    st.pyplot(fig)

    csv2 = inv.to_csv(index=False).encode("utf-8")
    st.download_button("Download Inventory Recommendations (CSV)", csv2, "inventory_recommendations.csv", "text/csv")

st.markdown("---")
st.caption("RetailPulse - Zidio Development Data Science Project")