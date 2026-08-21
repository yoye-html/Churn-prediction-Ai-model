# Trustworthy Churn Prediction

INSA Summer Camp — 5th Batch, AI Engineering Track
5-member team project | Dataset: [IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (Kaggle: `blastchar/telco-customer-churn`), 7,043 rows, 26.5% churn rate

## Objective

Build a **trustworthy** churn prediction system — not just accurate, but one whose output probabilities are reliable enough to drive real budget decisions. Three deliverables:

1. A calibrated LightGBM classifier (Brier Score Loss < 0.14 — this threshold also unlocks a follow-on KKBox dataset phase)
2. A SHAP-based churn-driver report explaining *why* the model predicts what it predicts
3. A top-k targeting simulation using an Expected Value framework, translating model output into an actual retention-campaign recommendation

## Results summary

| Deliverable | Status | Key number |
|---|---|---|
| Calibrated model | ✅ Gate passed | Brier = 0.1351 (Platt/sigmoid calibration, beat isotonic) |
| SHAP driver report | ⚠️ In review | Top drivers: `Contract`, `tenure`; one direction (`InternetService_Fiber_optic`) pending verification |
| Targeting simulation | ✅ Complete | Recommended: contact top 25% of customers by predicted risk (61.9% precision, ~$52K expected profit under current assumptions) |

## Project structure

```
churn_project/
├── data/
│   ├── telco_clean.csv              # raw cleaned dataset
│   └── processed/
│       ├── X_train.csv, X_test.csv          # pre-encoding, real dollar values
│       ├── X_train_encoded.csv, X_test_encoded.csv  # post-pipeline, scaled/encoded
│       └── y_train.csv, y_test.csv          # targets (Churn_binary)
├── src/
│   ├── features.py       # Phase 2: feature engineering pipeline
│   ├── train.py          # Phase 3: LightGBM tuning + calibration
│   ├── shap_report.py    # Phase 4: SHAP TreeExplainer analysis
│   └── budget_sim.py     # Phase 5: Expected Value targeting simulation
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_features.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_shap_report.ipynb
│   └── 05_budget_sim.ipynb
├── models/
│   ├── lgbm_base_tuned.pkl              # pre-calibration (used by SHAP)
│   ├── lgbm_calibrated_sigmoid.pkl      # PRODUCTION model (used by targeting sim)
│   └── lgbm_calibrated_isotonic.pkl     # comparison only, not used downstream
└── reports/
    ├── feature_report.md
    ├── phase3_calibration_metrics.csv
    ├── shap_churn_driver_report.md
    ├── model_card.md
    ├── top_k_targeting_list.csv
    └── figures/
```

## Team & phases

| Phase | Owner | Deliverable | Status |
|---|---|---|---|
| M1 | Environment / EDA | — | Complete |
| M2 | Feature Engineering | `src/features.py`, `feature_report.md` | Complete |
| M3 | Model Training / Calibration | `src/train.py`, calibrated model | Complete |
| M4 | SHAP Report | `src/shap_report.py`, driver report | In review (see caveats) |
| M5 | Budget Sim + Model Card | `src/budget_sim.py`, `model_card.md` | Complete |

## Locked technical decisions

- **Model:** LightGBM only (not XGBoost)
- **Calibration:** `CalibratedClassifierCV`, both Platt (sigmoid) and Isotonic compared — sigmoid won on Brier and ECE
- **Imputation:** `KNNImputer(k=5)`, fit on train only
- **Encoding:** `Contract` as ordinal [M2M=0, 1yr=1, 2yr=2]; all other multi-class nominal columns via `OneHotEncoder(handle_unknown='ignore')`, not `drop_first` — every category stays visible for SHAP
- **Split:** 80/20 stratified on `Churn_binary`, `random_state=42`
- **SHAP:** `TreeExplainer` on the pre-calibration model (`lgbm_base_tuned.pkl`) — calibration wrappers break direct booster access
- **Targeting economics:** CLV proxy = MonthlyCharges × 12 months; contact cost = flat $25/customer (validated against a percentage-of-bill model, which produced an unrealistic 77%-of-customers recommendation); save rate = 30% (⚠️ assumption, not derived from data — see Limitations)

## How to run

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt   # lightgbm, shap, scikit-learn, pandas, matplotlib, joblib

jupyter notebook notebooks/03_model_training.ipynb  # re-run from Phase 3 onward if needed
```

Run notebooks in numeric order — each phase depends on the previous phase's output files in `data/processed/` and `models/`.

## Known limitations / open items

- **`InternetService_Fiber_optic`'s SHAP direction is unverified.** Its measured direction (decreases churn risk) conflicts with the commonly reported pattern for this dataset. Pending: raw churn-rate cross-check and dependence-plot review before treating this as confirmed.
- **`save_rate` (30%) is a placeholder**, not derived from any real retention campaign. All targeting-simulation dollar figures scale directly with this assumption — see the sensitivity table in `reports/model_card.md` before using these numbers for actual budget decisions.
- **`is_senior_with_no_support`** is highly correlated with `SeniorCitizen` (r=0.86) by construction (77% of senior citizens lack TechSupport); retained for SHAP interpretability despite modest incremental predictive lift. Targeting decisions built on this feature should be reviewed for disproportionate impact on senior customers.
- Model trained on a single historical snapshot — recommend periodic re-validation against new data for concept drift.

## License / attribution

Dataset: IBM Telco Customer Churn, redistributed via Kaggle user `blastchar`. Project work licensed per INSA Summer Camp course terms.
