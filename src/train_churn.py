import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.metrics import roc_auc_score, precision_score, classification_report
import shap
import matplotlib.pyplot as plt
import joblib

# Load segmented customer data (has RFM already)
rfm = pd.read_csv("data/processed/customer_segments.csv")

# Define churn: customers with Recency > 180 days are considered churned
rfm["Churned"] = (rfm["Recency"] > 180).astype(int)

print(f"Total customers: {len(rfm)}")
print(f"Churned: {rfm[\"Churned\"].sum()} ({rfm[\"Churned\"].mean()*100:.1f}%)")

# Features for prediction (using log versions to avoid skew issues)
features = ["Recency_log", "Frequency_log", "Monetary_log"]
X = rfm[features]
y = rfm["Churned"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train XGBoost
model = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42, eval_metric="logloss")
model.fit(X_train, y_train)

# Evaluate
y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

auc = roc_auc_score(y_test, y_pred_proba)
print(f"\nAUC-ROC: {auc:.4f}")
print(f"Target: >= 0.88")

# Precision at top 20%
top_20_pct = int(len(y_test) * 0.2)
top_indices = np.argsort(y_pred_proba)[-top_20_pct:]
precision_top20 = y_test.iloc[top_indices].mean()
print(f"Precision@top20%: {precision_top20:.4f}")
print(f"Target: >= 0.75")

print("\n", classification_report(y_test, y_pred))

# SHAP explainability
explainer = shap.Explainer(model)
shap_values = explainer(X_test)

plt.figure()
shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()
plt.savefig("docs/shap_churn_summary.png", dpi=100, bbox_inches="tight")
print("Saved SHAP summary plot")

# Save model
joblib.dump(model, "models/churn_model.pkl")
print("Saved churn_model.pkl")
