# Phase 3 Model Card Inputs (raw numbers)

- Best LightGBM hyperparameters:
```json
{
  "subsample": 0.9,
  "scale_pos_weight": 1.0,
  "reg_lambda": 1.0,
  "reg_alpha": 0.1,
  "num_leaves": 15,
  "n_estimators": 300,
  "min_child_samples": 20,
  "max_depth": 3,
  "learning_rate": 0.03,
  "colsample_bytree": 0.8
}
```

- Chosen calibration method: **platt_sigmoid**
- Brier Score Loss: 0.1351 (target < 0.14, PASSED)
- Log-Loss: 0.4145
- Expected Calibration Error (10-bin): 0.0112
- ROC-AUC: 0.8464
- Average Precision (PR-AUC): 0.6628
- Train/test rows: 5634 / 1409
