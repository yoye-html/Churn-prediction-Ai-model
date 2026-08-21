"""
src/shap_report.py — Phase 4 (Member 4): SHAP Churn-Driver Report
INSA Summer Camp 5th Batch — Trustworthy Churn Prediction

Reusable, importable functions only — no top-level execution.
Mirrors src/features.py (Phase 2) and src/train.py (Phase 3).

LOCKED DECISIONS respected here:
  - SHAP TreeExplainer (not KernelExplainer — model is tree-based, exact
    and fast).
  - Explains the PRE-CALIBRATION model (models/lgbm_base_tuned.pkl), not
    the calibrated wrapper. CalibratedClassifierCV wraps the estimator,
    which breaks TreeExplainer's direct access to the booster. Calibration
    only rescales output probabilities — it doesn't change which features
    drive the ranking — so explaining the base tree model is both the
    correct choice and the only one TreeExplainer supports directly.
  - OneHotEncoder(handle_unknown='ignore'), NOT drop_first, was used in
    Phase 2 specifically so every category is visible here as its own
    SHAP feature/bar — do not collapse or drop categories before plotting.

Usage from the Phase 4 notebook:
    from src.shap_report import (
        load_base_model, compute_shap_values, global_importance_table,
        top_drivers_with_direction, plot_summary_beeswarm,
        plot_global_importance_bar, plot_dependence, write_driver_report,
    )
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import shap


# --------------------------------------------------------------------------
# Load model
# --------------------------------------------------------------------------
def load_base_model(model_dir):
    """
    Loads the pre-calibration tuned LightGBM model — the one SHAP explains.
    Raises a clear error if someone points this at the calibrated wrapper.
    """
    path = Path(model_dir) / "lgbm_base_tuned.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Phase 4 needs the PRE-calibration model from Phase 3, "
            "not lgbm_calibrated_sigmoid.pkl / lgbm_calibrated_isotonic.pkl — "
            "CalibratedClassifierCV wraps the estimator and breaks TreeExplainer's "
            "direct access to the booster."
        )
    model = joblib.load(path)
    if type(model).__name__ != "LGBMClassifier":
        raise TypeError(
            f"Expected a raw LGBMClassifier, got {type(model).__name__}. "
            "Did you accidentally load a calibrated wrapper?"
        )
    return model


# --------------------------------------------------------------------------
# SHAP computation
# --------------------------------------------------------------------------
def compute_shap_values(model, X):
    """
    Runs SHAP TreeExplainer on X (typically X_test, to explain held-out
    predictions rather than training-fit predictions).
    Returns (explainer, shap_values, expected_value).
    For binary classification LightGBM, shap_values is a 2D array
    (n_samples, n_features) for the positive (churn) class in recent
    shap versions returning an Explanation object — handle both cases.
    """
    explainer = shap.TreeExplainer(model)
    raw = explainer.shap_values(X)

    # shap.TreeExplainer can return a list [class0, class1] (older API)
    # or a single 2D array for binary classification (newer API) —
    # normalize to a single 2D array for the positive/churn class.
    if isinstance(raw, list):
        shap_values = raw[1]
        expected_value = explainer.expected_value[1]
    else:
        shap_values = raw
        expected_value = explainer.expected_value
        if isinstance(expected_value, (list, np.ndarray)) and np.ndim(expected_value) > 0:
            expected_value = expected_value[-1]

    return explainer, shap_values, expected_value


# --------------------------------------------------------------------------
# Global importance
# --------------------------------------------------------------------------
def global_importance_table(shap_values, X):
    """
    Mean |SHAP value| per feature, sorted descending — the standard
    global feature-importance ranking from SHAP.
    """
    mean_abs = np.abs(shap_values).mean(axis=0)
    df = pd.DataFrame({
        "feature": X.columns,
        "mean_abs_shap": mean_abs,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    return df


def top_drivers_with_direction(shap_values, X, top_n=10):
    """
    For each of the top-N features by mean |SHAP|, also reports the mean
    SIGNED SHAP value — tells you not just "this feature matters" but
    "this feature pushes risk up or down on average". Useful for the
    business-facing narrative in the driver report.
    """
    mean_abs = np.abs(shap_values).mean(axis=0)
    mean_signed = shap_values.mean(axis=0)
    df = pd.DataFrame({
        "feature": X.columns,
        "mean_abs_shap": mean_abs,
        "mean_signed_shap": mean_signed,
        "direction": np.where(mean_signed > 0, "increases churn risk", "decreases churn risk"),
    }).sort_values("mean_abs_shap", ascending=False).head(top_n).reset_index(drop=True)
    return df


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------
def plot_global_importance_bar(importance_df, top_n=15, out_path=None):
    top = importance_df.head(top_n).iloc[::-1]  # reverse for horizontal bar (largest on top)
    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * top_n)))
    ax.barh(top["feature"], top["mean_abs_shap"], color="#1D9E75")
    ax.set_xlabel("Mean |SHAP value| (average impact on model output)")
    ax.set_title(f"Top {top_n} Churn Drivers — Global Feature Importance")
    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=200)
    return fig


def plot_summary_beeswarm(shap_values, X, max_display=15, out_path=None):
    """Standard SHAP beeswarm: shows importance AND direction/spread per feature."""
    fig = plt.figure(figsize=(8, max(4, 0.35 * max_display)))
    shap.summary_plot(shap_values, X, max_display=max_display, show=False)
    if out_path is not None:
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
    return fig


def plot_dependence(feature, shap_values, X, interaction_feature="auto", out_path=None):
    """
    SHAP dependence plot for one feature — shows how that feature's value
    relates to its SHAP value (impact on churn risk), colored by the
    feature it interacts with most.
    """
    fig = plt.figure(figsize=(7, 5))
    shap.dependence_plot(feature, shap_values, X, interaction_index=interaction_feature, show=False)
    if out_path is not None:
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
    return fig


# --------------------------------------------------------------------------
# Report writing
# --------------------------------------------------------------------------
def write_driver_report(top_drivers_df, report_dir, model_context: str = ""):
    """
    Writes reports/shap_churn_driver_report.md — the Phase 4 deliverable.
    top_drivers_df: output of top_drivers_with_direction().
    model_context: optional free-text block (e.g. Brier score, calibration
    method used) to anchor the report to which model was actually explained.
    """
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / "shap_churn_driver_report.md"

    with open(path, "w") as f:
        f.write("# SHAP Churn-Driver Report\n\n")
        f.write("INSA Churn Project — Phase 4 (Member 4)\n\n")
        if model_context:
            f.write(f"{model_context}\n\n")
        f.write(
            "Explains `models/lgbm_base_tuned.pkl` (pre-calibration LightGBM) via "
            "SHAP TreeExplainer. Calibration reshapes output probabilities but not "
            "feature ranking, so this reflects the true drivers behind the model's "
            "calibrated predictions as well.\n\n"
        )
        f.write("## Top churn drivers\n\n")
        f.write("| Rank | Feature | Mean \\|SHAP\\| | Direction |\n")
        f.write("|---|---|---|---|\n")
        for i, row in top_drivers_df.iterrows():
            f.write(f"| {i+1} | `{row['feature']}` | {row['mean_abs_shap']:.4f} | {row['direction']} |\n")
        f.write(
            "\n*Mean \\|SHAP\\| = average magnitude of each feature's contribution to "
            "individual churn-risk predictions, across the explained sample. "
            "Direction is the sign of the average signed SHAP value — a feature can "
            "still push risk in both directions for different customers even if its "
            "average leans one way; see the beeswarm plot for the full spread.*\n"
        )

    return path