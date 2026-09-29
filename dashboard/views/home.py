import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from utils import segments

st.title("RetailPulse")
st.caption("Predictive Demand | Customer Segmentation | Churn Analysis | Inventory Optimization")

rfm = segments()
rev = rfm["Monetary"].sum()
champ = rfm[rfm["Segment_Label"] == "Champions"]
c = st.columns(4)
c[0].metric("Customers", f"{len(rfm):,}")
c[1].metric("Revenue (GBP)", f"{rev:,.0f}")
c[2].metric("Champions (% of customers)", f"{len(champ) / len(rfm) * 100:.1f}%")
c[3].metric("Champions (% of revenue)", f"{champ['Monetary'].sum() / rev * 100:.1f}%")
st.info("Use the sidebar to open each analysis page. Demo data: UCI Online Retail II (2009-2011). No sign-in needed.")

left, right = st.columns(2)
with left:
    st.subheader("Customers by segment")
    fig, ax = plt.subplots(figsize=(6, 4))
    rfm["Segment_Label"].value_counts().plot(kind="barh", ax=ax, color="#1F4E78")
    ax.invert_yaxis()
    ax.set_xlabel("Customers")
    st.pyplot(fig)
with right:
    st.subheader("Model performance")
    perf = pd.DataFrame({
        "Model": ["Prophet forecast", "LSTM forecast", "XGBoost churn", "XGBoost churn"],
        "Metric": ["MAPE", "MAPE", "AUC-ROC", "Precision@top20%"],
        "Result": ["24.06%", "51.25%", "0.77", "0.72"],
        "Target": ["<= 12%", "<= 12%", ">= 0.88", ">= 0.75"],
    })
    st.dataframe(perf, hide_index=True)
    st.caption("Reported honestly: targets were not met. Churn AUC is realistic after removing a target-leakage bug.")
