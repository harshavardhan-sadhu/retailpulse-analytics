import matplotlib.pyplot as plt
import streamlit as st
from utils import churn_scores, image

st.title("Churn Risk")
st.caption("XGBoost on Frequency and Monetary. Churn = no purchase in 180+ days. Recency is excluded to avoid target leakage.")

r = churn_scores()
thr = st.slider("Risk threshold", 0.0, 1.0, 0.5, 0.05)
risky = r[r["Churn_Probability"] >= thr].sort_values("Churn_Probability", ascending=False)
valuable = risky[risky["Monetary"] >= r["Monetary"].median()]

a, b, c, d = st.columns(4)
a.metric("AUC-ROC", "0.77")
b.metric("Precision@top20%", "0.72")
c.metric("Customers above threshold", f"{len(risky):,}")
d.metric("High-value at risk", f"{len(valuable):,}")

fig, ax = plt.subplots(figsize=(9, 3.5))
ax.hist(r["Churn_Probability"], bins=30, color="#C0504D")
ax.axvline(thr, color="black", linestyle="--")
ax.set_xlabel("Churn probability")
ax.set_ylabel("Customers")
st.pyplot(fig)

cols = ["Customer ID", "Recency", "Frequency", "Monetary", "Churn_Probability"]
st.dataframe(risky[cols].head(50), hide_index=True)
st.download_button("Download at-risk customers", risky[cols].to_csv(index=False).encode("utf-8"), "at_risk_customers.csv", "text/csv")
with st.expander("How the model was validated"):
    st.write("The first version scored AUC 1.0 because Recency defined the label and was also a feature (target leakage). Removing it gave the realistic 0.77.")
    image("docs/shap_churn_summary.png", "SHAP feature importance")
