# SHAP Churn-Driver Report

INSA Churn Project — Phase 4 (Member 4)

**Model explained:** LightGBM (pre-calibration base model, `lgbm_base_tuned.pkl`).

**Production model:** Platt-calibrated LightGBM (`lgbm_calibrated_sigmoid.pkl`), Brier Score Loss 0.1351 on held-out test set (gate: < 0.14, PASSED).

Explains `models/lgbm_base_tuned.pkl` (pre-calibration LightGBM) via SHAP TreeExplainer. Calibration reshapes output probabilities but not feature ranking, so this reflects the true drivers behind the model's calibrated predictions as well.

## Top churn drivers

| Rank | Feature | Mean \|SHAP\| | Direction |
|---|---|---|---|
| 1 | `Contract` | 0.7821 | increases churn risk |
| 2 | `tenure` | 0.5059 | decreases churn risk |
| 3 | `InternetService_Fiber_optic` | 0.2479 | decreases churn risk |
| 4 | `PaymentMethod_Electronic_check` | 0.1837 | increases churn risk |
| 5 | `InternetService_No` | 0.1522 | decreases churn risk |
| 6 | `charges_per_month` | 0.1255 | increases churn risk |
| 7 | `PaperlessBilling` | 0.1226 | decreases churn risk |
| 8 | `OnlineSecurity` | 0.1050 | increases churn risk |
| 9 | `MonthlyCharges` | 0.1029 | decreases churn risk |
| 10 | `StreamingMovies` | 0.1002 | decreases churn risk |

*Mean \|SHAP\| = average magnitude of each feature's contribution to individual churn-risk predictions, across the explained sample. Direction is the sign of the average signed SHAP value — a feature can still push risk in both directions for different customers even if its average leans one way; see the beeswarm plot for the full spread.*
