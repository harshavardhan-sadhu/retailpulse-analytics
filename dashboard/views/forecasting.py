import matplotlib.pyplot as plt
import streamlit as st
from utils import forecast, series, image

st.title("Demand Forecasting")
st.caption("Prophet trained on 95th-percentile-capped daily sales. MAPE 24.06% on a 30-day holdout.")

fc, ts = forecast(), series()
days = st.slider("Days of history to display", 30, 400, 120)
hist = ts.tail(days)
view = fc[fc["ds"] >= hist["ds"].min()]

fig, ax = plt.subplots(figsize=(12, 4.5))
ax.plot(hist["ds"], hist["y"], color="black", label="Actual")
ax.plot(view["ds"], view["yhat"], color="orange", label="Forecast")
ax.fill_between(view["ds"], view["yhat_lower"], view["yhat_upper"], color="orange", alpha=0.2)
ax.set_ylabel("Daily sales (GBP)")
ax.legend()
st.pyplot(fig)

st.subheader("What-if: promotion impact")
boost = st.slider("Promotion uplift (%)", 0, 50, 10)
base = fc.tail(30)["yhat"].sum()
new = base * (1 + boost / 100)
a, b, c = st.columns(3)
a.metric("Baseline, final 30 forecast days (GBP)", f"{base:,.0f}")
b.metric("With uplift (GBP)", f"{new:,.0f}", f"{new - base:,.0f}")
c.metric("Prophet MAPE", "24.06%")

st.download_button("Download forecast CSV", fc.to_csv(index=False).encode("utf-8"), "prophet_forecast.csv", "text/csv")
with st.expander("Model diagnostics"):
    image("docs/seasonal_decomposition.png", "Seasonal decomposition")
    image("docs/lstm_forecast_vs_actual.png", "LSTM comparison (MAPE 51.25%)")
