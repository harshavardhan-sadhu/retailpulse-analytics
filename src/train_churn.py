import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import roc_auc_score, classification_report
import shap
import matplotlib.pyplot as plt
import joblib

rfm = pd.read_csv("data/processed/customer_segments.csv")
rfm["Churned"] = (rfm["Recency"] > 180).astype(int)

churned_count = rfm["Churned"].sum()
churned_pct = rfm["Churned"].mean() * 100
print(f"Total customers: {len(rfm)}")
print(f"Churned: {churned_count} ({churned_pct:.1f}%)")

# IMPORTANT FIX: removed Recency_log since it directly determines the label (data leakage)
features = ["Frequency_log", "Monetary_log"]
X = rfm[features]
y = rfm["Churned"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42, eval_metric="logloss")
model.fit(X_train, y_train)

y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

auc = roc_auc_score(y_test, y_pred_proba)
print(f"AUC-ROC: {auc:.4f} (target >= 0.88)")

top_20_pct = int(len(y_test) * 0.2)
top_indices = np.argsort(y_pred_proba)[-top_20_pct:]
precision_top20 = y_test.iloc[top_indices].mean()
print(f"Precision@top20%: {precision_top20:.4f} (target >= 0.75)")

print(classification_report(y_test, y_pred))

explainer = shap.Explainer(model)
shap_values = explainer(X_test)
plt.figure()
shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()
plt.savefig("docs/shap_churn_summary.png", dpi=100, bbox_inches="tight")

joblib.dump(model, "models/churn_model.pkl")
print("Saved churn_model.pkl and SHAP plot")
