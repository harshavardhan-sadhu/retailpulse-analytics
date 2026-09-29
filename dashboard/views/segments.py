import matplotlib.pyplot as plt
import streamlit as st
from utils import segments, image

ACTIONS = {
    "Champions": "Reward and retain: VIP perks, early access.",
    "Loyal Customers": "Upsell and enrol in loyalty programmes.",
    "Big Spenders": "Re-engage before they drift: personalised offers.",
    "New/Occasional": "Nurture: onboarding and second-purchase incentives.",
    "At Risk": "Win-back campaigns with targeted discounts.",
    "Lost/Churned": "Low-cost reactivation only.",
}

st.title("Customer Segments")
st.caption("RFM features + K-Means (k=6, log-scaled). Labels come from cluster profiles.")

rfm = segments()
summary = rfm.groupby("Segment_Label").agg(
    Customers=("Customer ID", "count"),
    Avg_Recency=("Recency", "mean"),
    Avg_Frequency=("Frequency", "mean"),
    Avg_Monetary=("Monetary", "mean"),
).round(1).reset_index()
summary["Suggested action"] = summary["Segment_Label"].map(ACTIONS)
st.dataframe(summary, hide_index=True)

choice = st.selectbox("Filter customers", ["All"] + sorted(rfm["Segment_Label"].unique()))
view = rfm if choice == "All" else rfm[rfm["Segment_Label"] == choice]

fig, ax = plt.subplots(figsize=(9, 4.5))
for name, g in view.groupby("Segment_Label"):
    ax.scatter(g["Frequency"], g["Monetary"], s=12, alpha=0.5, label=name)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Frequency (orders, log)")
ax.set_ylabel("Monetary (GBP, log)")
ax.legend()
st.pyplot(fig)

cols = ["Customer ID", "Recency", "Frequency", "Monetary", "Segment_Label"]
st.dataframe(view[cols].head(100), hide_index=True)
st.download_button("Download segment data", view[cols].to_csv(index=False).encode("utf-8"), "customer_segments.csv", "text/csv")
with st.expander("Clustering diagnostics"):
    image("docs/elbow_method.png", "Elbow method")
