"""
src/budget_sim.py — Phase 5 (Member 5): Top-K Targeting Simulation + Model Card
INSA Summer Camp 5th Batch — Trustworthy Churn Prediction

Reusable, importable functions only — no top-level execution.
Mirrors src/features.py (Phase 2), src/train.py (Phase 3), src/shap_report.py (Phase 4).

FRAMEWORK: Expected Value / cost-benefit matrix (Provost & Fawcett), evaluated
relative to a "do nothing" baseline (contacting nobody = $0 net effect).
For a customer contacted based on the model's ranking:
    actual churner,   contacted -> profit = save_rate * CLV - contact_cost
    actual non-churn, contacted -> profit = -contact_cost
    not contacted (either class)    -> profit = 0  (baseline reference)

ASSUMPTIONS (not derivable from the data — override with real numbers if available):
    - CLV proxy      = MonthlyCharges * RETENTION_HORIZON_MONTHS (default 12)
    - Contact cost   = CONTACT_COST_PCT * MonthlyCharges (default 10%)
    - Save rate      = SAVE_RATE (default 0.30) — retention-campaign effectiveness,
                        cannot be derived from churn data alone.

IMPORTANT: MonthlyCharges in X_test_encoded.csv is STANDARDIZED (scaled), not
real dollars. This module pulls raw dollar MonthlyCharges from X_test.csv
(the pre-encoding file) and aligns it by row position to X_test_encoded /
y_test / predicted probabilities, which were confirmed row-aligned in the
Phase 3 sanity check.

Usage from the Phase 5 notebook:
    from src.budget_sim import (
        get_raw_monthly_charges, compute_customer_economics,
        expected_profit_at_k, profit_curve, find_optimal_k,
        plot_profit_curve, top_k_targeting_table, write_model_card,
    )
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RETENTION_HORIZON_MONTHS = 12   # CLV proxy = MonthlyCharges * this
CONTACT_COST_PCT = 0.10         # contact cost = this * MonthlyCharges
SAVE_RATE = 0.30                # fraction of contacted actual-churners retained — ASSUMPTION


# --------------------------------------------------------------------------
# Data prep
# --------------------------------------------------------------------------
def get_raw_monthly_charges(raw_csv_path, n_rows_expected=None):
    """
    Loads real dollar MonthlyCharges from the pre-encoding CSV (e.g. X_test.csv).
    Do NOT use X_test_encoded.csv's MonthlyCharges column for money math — it's
    standardized/scaled, not dollars.
    """
    df = pd.read_csv(raw_csv_path)
    if n_rows_expected is not None:
        assert len(df) == n_rows_expected, (
            f"{raw_csv_path} has {len(df)} rows, expected {n_rows_expected} — "
            "row alignment with X_test_encoded/y_test would be wrong. Check the file."
        )
    assert "MonthlyCharges" in df.columns, f"MonthlyCharges column not found in {raw_csv_path}"
    return df["MonthlyCharges"].values


def compute_customer_economics(monthly_charges, retention_horizon_months=RETENTION_HORIZON_MONTHS,
                                 contact_cost_pct=CONTACT_COST_PCT):
    """
    Per-customer CLV proxy and contact cost, both derived from real dollar
    MonthlyCharges. Returns (clv, contact_cost) arrays, same length as input.
    """
    monthly_charges = np.asarray(monthly_charges, dtype=float)
    clv = monthly_charges * retention_horizon_months
    contact_cost = monthly_charges * contact_cost_pct
    return clv, contact_cost


# --------------------------------------------------------------------------
# Expected Value simulation
# --------------------------------------------------------------------------
def expected_profit_at_k(y_true, y_prob, clv, contact_cost, k_frac, save_rate=SAVE_RATE):
    """
    Contacts the top k_frac (0-1) of customers ranked by predicted churn
    probability. Returns (total_profit, n_contacted, n_true_churners_contacted).

    Profit per contacted customer:
        actual churner (y=1):     save_rate * clv_i - contact_cost_i
        actual non-churner (y=0): -contact_cost_i
    Not contacted: profit = 0 (baseline).
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    clv = np.asarray(clv, dtype=float)
    contact_cost = np.asarray(contact_cost, dtype=float)

    n = len(y_true)
    n_contact = max(1, int(round(k_frac * n)))
    order = np.argsort(-y_prob)  # descending by predicted churn probability
    contacted_idx = order[:n_contact]

    contacted_y = y_true[contacted_idx]
    contacted_clv = clv[contacted_idx]
    contacted_cost = contact_cost[contacted_idx]

    is_churner = contacted_y == 1
    profit_churners = np.sum(save_rate * contacted_clv[is_churner] - contacted_cost[is_churner])
    profit_non_churners = np.sum(-contacted_cost[~is_churner])
    total_profit = profit_churners + profit_non_churners

    return total_profit, n_contact, int(is_churner.sum())


def profit_curve(y_true, y_prob, clv, contact_cost, save_rate=SAVE_RATE,
                  k_values=None):
    """
    Sweeps k from 1% to 100% of the population and computes expected profit
    at each. Returns a DataFrame: k_pct, n_contacted, n_true_churners_contacted,
    total_profit, precision_at_k.
    """
    if k_values is None:
        k_values = np.arange(0.01, 1.01, 0.01)

    rows = []
    for k in k_values:
        profit, n_contact, n_churners = expected_profit_at_k(
            y_true, y_prob, clv, contact_cost, k, save_rate=save_rate
        )
        precision = n_churners / n_contact if n_contact > 0 else 0.0
        rows.append({
            "k_pct": round(k * 100, 1),
            "n_contacted": n_contact,
            "n_true_churners_contacted": n_churners,
            "total_profit": profit,
            "precision_at_k": precision,
        })
    return pd.DataFrame(rows)


def find_optimal_k(curve_df):
    """Returns the row of the profit curve with maximum total_profit."""
    return curve_df.loc[curve_df["total_profit"].idxmax()]


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------
def plot_profit_curve(curve_df, optimal_row=None, out_path=None):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(curve_df["k_pct"], curve_df["total_profit"], color="#1D9E75", linewidth=2)
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)

    if optimal_row is not None:
        ax.axvline(optimal_row["k_pct"], color="#D85A30", linestyle=":", linewidth=1.5)
        ax.scatter([optimal_row["k_pct"]], [optimal_row["total_profit"]],
                   color="#D85A30", zorder=5, s=50,
                   label=f"Optimal k = {optimal_row['k_pct']:.0f}% "
                         f"(${optimal_row['total_profit']:,.0f})")
        ax.legend(fontsize=9)

    ax.set_xlabel("% of customers contacted (ranked by predicted churn risk)")
    ax.set_ylabel("Expected profit ($)")
    ax.set_title("Top-K Targeting: Expected Profit Curve")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if out_path is not None:
        fig.savefig(out_path, dpi=200)
    return fig


# --------------------------------------------------------------------------
# Business-facing targeting table
# --------------------------------------------------------------------------
def top_k_targeting_table(y_prob, monthly_charges, clv, contact_cost, k_frac,
                           customer_ids=None):
    """
    Returns a DataFrame of the top-k_frac customers by predicted churn risk,
    with their economics — the actual list an ops/retention team would work from.
    """
    n = len(y_prob)
    n_contact = max(1, int(round(k_frac * n)))
    order = np.argsort(-np.asarray(y_prob))[:n_contact]

    df = pd.DataFrame({
        "customer_id": customer_ids[order] if customer_ids is not None else order,
        "predicted_churn_prob": np.asarray(y_prob)[order],
        "monthly_charges": np.asarray(monthly_charges)[order],
        "clv_proxy": np.asarray(clv)[order],
        "contact_cost": np.asarray(contact_cost)[order],
    }).sort_values("predicted_churn_prob", ascending=False).reset_index(drop=True)

    return df


# --------------------------------------------------------------------------
# Model card
# --------------------------------------------------------------------------
def write_model_card(report_dir, brier_score, calibration_method, roc_auc, ece,
                      optimal_k_row, top_shap_drivers, save_rate=SAVE_RATE,
                      retention_horizon_months=RETENTION_HORIZON_MONTHS,
                      contact_cost_pct=CONTACT_COST_PCT):
    """
    Writes reports/model_card.md — the Phase 5 deliverable, pulling together
    Phase 3 (calibration), Phase 4 (SHAP), and Phase 5 (targeting) results.
    top_shap_drivers: a list of (feature, direction) tuples, top 5 recommended.
    """
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / "model_card.md"

    driver_lines = "\n".join(f"- `{feat}` — {direction}" for feat, direction in top_shap_drivers)

    content = f"""# Model Card — Trustworthy Churn Prediction

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
- **Calibration:** CalibratedClassifierCV, **{calibration_method}** selected
  after comparing Platt (sigmoid) and Isotonic — {calibration_method} won on
  both Brier score and Expected Calibration Error
- **Training data split:** 80/20 stratified on Churn_binary, random_state=42

## Performance (held-out test set)

| Metric | Value | Target |
|---|---|---|
| Brier Score Loss | {brier_score:.4f} | < 0.14 |
| ROC-AUC | {roc_auc:.4f} | — |
| Expected Calibration Error (10-bin) | {ece:.4f} | — |

Gate status: {"PASSED" if brier_score < 0.14 else "NOT MET"} — {"KKBox dataset unlocked" if brier_score < 0.14 else "do not proceed to KKBox"}.

## Top churn drivers (SHAP TreeExplainer, pre-calibration model)

{driver_lines}

See `reports/shap_churn_driver_report.md` for the full analysis, dependence
plots, and important caveats about features whose SHAP direction disagreed
with raw churn-rate patterns and required manual verification.

## Business simulation: top-k targeting

**Framework:** Expected Value (cost-benefit matrix relative to a "contact
nobody" baseline). Contacted customers ranked by predicted churn probability.

**Assumptions (not derived from churn data — override with real business
numbers if available):**
- CLV proxy = MonthlyCharges x {retention_horizon_months} months
- Contact cost = {contact_cost_pct:.0%} of MonthlyCharges
- Save rate = {save_rate:.0%} of contacted actual-churners successfully retained

**Recommended targeting depth:** contact the top **{optimal_k_row['k_pct']:.0f}%**
of customers by predicted churn risk ({int(optimal_k_row['n_contacted'])} customers),
capturing an estimated **{int(optimal_k_row['n_true_churners_contacted'])} true churners**
at **{optimal_k_row['precision_at_k']:.1%} precision**, for an estimated expected
profit of **${optimal_k_row['total_profit']:,.0f}** under the assumptions above.

**Sensitivity warning:** the optimal k and profit figure are directly
proportional to the save_rate assumption, which is not derived from the churn
model or dataset. Re-run `src/budget_sim.py`'s `profit_curve()` with the
team's actual retention-campaign save rate before using this number for
budget decisions.

## Limitations & ethical considerations

- SeniorCitizen-correlated features (e.g. `is_senior_with_no_support`, r=0.86
  with SeniorCitizen) mean targeting decisions may disproportionately flag
  senior customers — review for fairness before operationalizing.
- The save_rate assumption ({save_rate:.0%}) is a placeholder pending real
  campaign data; treat all downstream profit figures as illustrative.
- Model trained and calibrated on a single historical snapshot; churn
  drivers and calibration should be periodically re-validated against new
  data (concept drift).
"""

    with open(path, "w") as f:
        f.write(content)

    return path