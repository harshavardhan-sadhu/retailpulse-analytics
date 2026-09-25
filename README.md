# RetailPulse - AI-Powered Customer Analytics & Demand Forecasting Platform

## Overview
RetailPulse is an end-to-end data science platform that analyzes retail sales and customer data to deliver demand forecasting, customer segmentation, churn prediction, and inventory optimization.

**Live Demo:** https://retailpulse-analytics-yyfpzpdzr7l2cgtuuvnpjm.streamlit.app/

## Features
- Demand Forecasting: Prophet-based time series forecasting (24.06% MAPE)
- Customer Segmentation: RFM analysis + K-Means clustering (6 segments)
- Churn Prediction: XGBoost classifier with SHAP explainability (0.77 AUC-ROC)
- Inventory Optimization: Safety stock and reorder point recommendations
- Interactive Dashboard: Streamlit app with real-time filtering and what-if analysis

## Key Insight
7.6% of customers (Champions segment) generate 54.3% of total revenue - a clear Pareto pattern that informs retention strategy.

## Tech Stack
- Python 3.14, Pandas, Scikit-learn
- Prophet, PyTorch/PyTorch Lightning (LSTM)
- XGBoost, SHAP
- Streamlit (dashboard)
- MLflow (experiment tracking)

## Project Structure

## Setup
```bash
pip install -r requirements.txt
streamlit run dashboard/app.py
```

## Model Performance
| Model | Metric | Result | Target |
|---|---|---|---|
| Demand Forecasting (Prophet) | MAPE | 24.06% | <=12% |
| Churn Prediction (XGBoost) | AUC-ROC | 0.77 | >=0.88 |
| Churn Prediction | Precision@top20% | 0.72 | >=0.75 |

## Key Challenges Solved
1. **Target leakage in churn model**: Initial model showed perfect AUC (1.0) due to using Recency as both the churn-definition variable and a feature. Fixed by restricting features to Frequency and Monetary only.
2. **Non-stationary time series**: Identified that 77% of "missing" days were Saturdays (store closure), not data errors. Handled by encoding as zero-sales days rather than interpolating.
3. **Outlier-driven volatility**: Large wholesale orders caused high forecast error. Addressed via 95th-percentile capping on training data.

## Author
Harshavardhan Sadhu - Zidio Development Data Science & Analytics Internship
