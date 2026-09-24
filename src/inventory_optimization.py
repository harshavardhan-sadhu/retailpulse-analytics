import pandas as pd
import numpy as np

# Load forecast and product-level sales data
df = pd.read_csv("data/processed/sales_for_forecasting.csv")

# Aggregate demand by product (StockCode)
product_demand = df.groupby("StockCode").agg(
    avg_daily_demand=("Quantity", lambda x: x.sum() / 739),
    demand_std=("Quantity", "std"),
    total_units_sold=("Quantity", "sum")
).reset_index()

product_demand = product_demand.dropna()
product_demand = product_demand[product_demand["total_units_sold"] > 10]  # filter very rare items

# Simple safety stock formula: safety_stock = Z * std_dev * sqrt(lead_time)
# Assume lead_time = 7 days, Z = 1.65 (95% service level)
lead_time_days = 7
z_score = 1.65

product_demand["safety_stock"] = z_score * product_demand["demand_std"] * np.sqrt(lead_time_days)
product_demand["reorder_point"] = (product_demand["avg_daily_demand"] * lead_time_days) + product_demand["safety_stock"]
product_demand["recommended_order_qty"] = product_demand["avg_daily_demand"] * 30  # 30-day supply

product_demand = product_demand.round(2)
product_demand.to_csv("data/processed/inventory_recommendations.csv", index=False)

print(f"Generated inventory recommendations for {len(product_demand)} products")
print(product_demand.head(10))
print("Saved inventory_recommendations.csv")
