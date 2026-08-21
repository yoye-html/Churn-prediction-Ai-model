# Churn Prediction Project — INSA Summer Camp 5th Batch

## What this does
Predicts customer churn with calibrated probabilities trustworthy enough to drive
a retention budget. Includes SHAP-based driver analysis and top-k EV targeting simulation.

## Dataset
IBM Telco Customer Churn (Kaggle, ~7k rows)

## Team
- Member 1: EDA & Setup
- Member 2: Feature Engineering
- Member 3: Model Training & Calibration
- Member 4: SHAP Report
- Member 5: Budget Simulation & Model Card

## How to run
1. `python3 -m venv venv && source venv/bin/activate`
2. `pip install -r requirements.txt`
3. Run notebooks in order: 01_eda → 02_features → 03_model → 04_shap → 05_simulation
Churn rate is ~26.5% — mild class imbalance. We do NOT need SMOTE or oversampling at this level. The guide (M07) tells us to use precision/recall not accuracy."