import matplotlib.pyplot as plt
import streamlit as st
from utils import inventory

st.title("Inventory Optimization")
st.caption("Reorder point = demand over a 7-day lead time + safety stock (95% service level, Z = 1.65).")

inv = inventory()
q = st.text_input("Search StockCode")
sort = st.selectbox("Sort by", ["recommended_order_qty", "reorder_point", "safety_stock", "avg_daily_demand"])
n = st.slider("Rows to show", 10, 200, 50)

view = inv[inv["StockCode"].str.contains(q, case=False, regex=False)] if q else inv
view = view.sort_values(sort, ascending=False)

a, b, c = st.columns(3)
a.metric("Products", f"{len(inv):,}")
b.metric("Median reorder point", f"{inv['reorder_point'].median():,.0f}")
c.metric("Matches", f"{len(view):,}")
st.dataframe(view.head(n), hide_index=True)

qty = inv["recommended_order_qty"]
fig, ax = plt.subplots(figsize=(9, 3.5))
ax.hist(qty.clip(upper=qty.quantile(0.95)), bins=30, color="#2E8B57")
ax.set_xlabel("Recommended order quantity (clipped at 95th percentile)")
ax.set_ylabel("Products")
st.pyplot(fig)
st.download_button("Download recommendations", view.to_csv(index=False).encode("utf-8"), "inventory_recommendations.csv", "text/csv")
