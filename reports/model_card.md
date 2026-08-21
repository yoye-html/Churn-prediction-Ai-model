# Model Card — Trustworthy Churn Prediction

**Project:** INSA Summer Camp 5th Batch, AI Engineering Track
**Dataset:** IBM Telco Customer Churn (Kaggle: blastchar/telco-customer-churn), 7,043 rows, 26.5% churn

## Intended use

Predicts customer churn probability for the IBM Telco dataset to support a
top-k retention-targeting campaign. Calibrated probabilities are the primary
output — this model is designed to be trusted numerically (e.g. "this
customer has a 42% churn probability"), not just used for ranking.

**Not intended for:** deployment on other telecom providers' customer bases
without re-validation; use as the sole basis for individual customer-facing
decisions without human review; extrapolation beyond the feature ranges
present in the training data.

## Model details

- **Algorithm:** LightGBM (LGBMClassifier), hyperparameters tuned via
  RandomizedSearchCV (5-fold stratified, scored on negative Brier score)
- **Calibration:** CalibratedClassifierCV, **platt_sigmoid** selected
  after comparing Platt (sigmoid) and Isotonic — platt_sigmoid won on
  both Brier score and Expected Calibration Error
- **Training data split:** 80/20 stratified on Churn_binary, random_state=42

## Performance (held-out test set)

| Metric | Value | Target |
|---|---|---|
| Brier Score Loss | 0.1351 | < 0.14 |
| ROC-AUC | 0.8464 | — |
| Expected Calibration Error (10-bin) | 0.0112 | — |

Gate status: PASSED — KKBox dataset unlocked.

## Top churn drivers (SHAP TreeExplainer, pre-calibration model)

- `Contract` — increases churn risk
- `tenure` — decreases churn risk
- `InternetService_Fiber_optic` — see dependence plot — direction under review
- `PaymentMethod_Electronic_check` — increases churn risk
- `InternetService_No` — decreases churn risk

See `reports/shap_churn_driver_report.md` for the full analysis, dependence
plots, and important caveats about features whose SHAP direction disagreed
with raw churn-rate patterns and required manual verification.

## Business simulation: top-k targeting

**Framework:** Expected Value (cost-benefit matrix relative to a "contact
nobody" baseline). Contacted customers ranked by predicted churn probability.

**Assumptions (not derived from churn data — override with real business
numbers if available):**
- CLV proxy = MonthlyCharges x 12 months
- Contact cost = 10% of MonthlyCharges
- Save rate = 30% of contacted actual-churners successfully retained

**Recommended targeting depth:** contact the top **77%**
of customers by predicted churn risk (1085 customers),
capturing an estimated **371 true churners**
at **34.2% precision**, for an estimated expected
profit of **$89,762** under the assumptions above.

**Sensitivity warning:** the optimal k and profit figure are directly
proportional to the save_rate assumption, which is not derived from the churn
model or dataset. Re-run `src/budget_sim.py`'s `profit_curve()` with the
team's actual retention-campaign save rate before using this number for
budget decisions.

## Limitations & ethical considerations

- SeniorCitizen-correlated features (e.g. `is_senior_with_no_support`, r=0.86
  with SeniorCitizen) mean targeting decisions may disproportionately flag
  senior customers — review for fairness before operationalizing.
- The save_rate assumption (30%) is a placeholder pending real
  campaign data; treat all downstream profit figures as illustrative.
- Model trained and calibrated on a single historical snapshot; churn
  drivers and calibration should be periodically re-validated against new
  data (concept drift).
