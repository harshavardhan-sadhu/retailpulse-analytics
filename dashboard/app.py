import streamlit as st

st.set_page_config(page_title="RetailPulse Analytics", page_icon=":material/insights:", layout="wide")

pages = [
    st.Page("views/home.py", title="Home", icon=":material/home:", default=True),
    st.Page("views/forecasting.py", title="Demand Forecasting", icon=":material/show_chart:"),
    st.Page("views/segments.py", title="Customer Segments", icon=":material/groups:"),
    st.Page("views/churn.py", title="Churn Risk", icon=":material/warning:"),
    st.Page("views/inventory.py", title="Inventory", icon=":material/inventory_2:"),
]

with st.sidebar:
    st.markdown("**RetailPulse**  \nAI-Powered Customer Analytics & Demand Forecasting")
    st.caption("Harshavardhan Sadhu - Zidio Data Science & Analytics")
    st.link_button("GitHub repository", "https://github.com/harshavardhan-sadhu/retailpulse-analytics")

st.navigation(pages).run()
