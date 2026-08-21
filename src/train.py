"""
src/train.py — Phase 3 (Member 3): Model Training & Calibration
INSA Summer Camp 5th Batch — Trustworthy Churn Prediction

Reusable, importable functions only — no top-level execution.
Mirrors the src/features.py pattern from Phase 2 (Member 2): the notebook
imports from here and calls these functions cell by cell, rather than
redefining everything inline.

LOCKED DECISIONS respected here:
  - Model: LightGBM (LGBMClassifier), not XGBoost
  - Calibration: CalibratedClassifierCV, method='sigmoid' (Platt) AND
    method='isotonic' are both fit and compared — never sigmoid-only.
  - Gate: Brier Score Loss < 0.14 on held-out test set.

Usage from the Phase 3 notebook:
    from src.train import (
        load_target, sanity_check, tune_lightgbm, calibrate_both,
        evaluate, expected_calibration_error, plot_diagnostics,
        save_artifacts, write_model_card_inputs,
    )
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    log_loss,
    brier_score_loss,
    roc_auc_score,
    average_precision_score,
)
from lightgbm import LGBMClassifier

RANDOM_STATE = 42
BRIER_TARGET = 0.14


# --------------------------------------------------------------------------
# Data loading / validation
# --------------------------------------------------------------------------
def load_target(path) -> np.ndarray:
    """
    Safely load a target CSV whether or not it has a stray index column.
    Same index-safety logic as Phase 2's features.py::load_target.
    Target is assumed to be the last column.
    """
    df = pd.read_csv(path)
    y = df.iloc[:, -1] if df.shape[1] > 1 else df.iloc[:, 0]
    return y.astype(int).values


def sanity_check(X_train, X_test, y_train, y_test):
    """
    Validates the Phase 2 -> Phase 3 handoff. Raises AssertionError with a
    descriptive message on the first problem found, so failures are loud
    and specific rather than a downstream shape-mismatch traceback.
    """
    assert X_train.shape[0] == len(y_train), "X_train/y_train row mismatch — re-check Phase 2 handoff"
    assert X_test.shape[0] == len(y_test), "X_test/y_test row mismatch"
    assert X_train.shape[1] == X_test.shape[1], "train/test column mismatch — encoding drift between splits"
    assert not X_train.isna().any().any(), "NaNs in X_train — imputer step may have been skipped"
    assert not X_test.isna().any().any(), "NaNs in X_test"
    assert set(np.unique(y_train)) == {0, 1}, "y_train is not binary — check load_target()"

    neg, pos = np.bincount(y_train)
    churn_rate = pos / (pos + neg)
    assert 0.24 < churn_rate < 0.29, (
        f"Train churn rate {churn_rate:.4f} drifted from expected ~26.5% — check the split"
    )
    return churn_rate


# --------------------------------------------------------------------------
# Hyperparameter search
# --------------------------------------------------------------------------
def tune_lightgbm(X_train, y_train, n_iter: int = 40, verbose: int = 1):
    """
    RandomizedSearchCV over LightGBM, 5-fold stratified, scored on
    neg_brier_score (directly optimizes the team's gate metric).
    n_iter kept modest by default for the shared 8GB laptop.
    Returns (best_estimator_refit_on_full_train, best_params_dict, search_object).
    """
    neg, pos = np.bincount(y_train)
    scale_pos_weight = neg / pos

    param_dist = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [3, 4, 5, 6, -1],
        "num_leaves": [15, 31, 63, 127],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "min_child_samples": [10, 20, 30],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "reg_alpha": [0.0, 0.1, 0.5, 1.0],
        "reg_lambda": [0.0, 0.1, 0.5, 1.0],
        "scale_pos_weight": [1.0, scale_pos_weight],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    base = LGBMClassifier(objective="binary", random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1)

    search = RandomizedSearchCV(
        base, param_dist, n_iter=n_iter, scoring="neg_brier_score",
        cv=cv, random_state=RANDOM_STATE, n_jobs=1, verbose=verbose,
    )
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    best_model.fit(X_train, y_train)  # refit on full train set with best params
    return best_model, search.best_params_, search


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------
def calibrate_both(tuned_model, X_train, y_train):
    """
    Fits both Platt (sigmoid) and Isotonic calibration on top of the tuned
    base model, per the locked decision to compare rather than assume.
    Returns (cal_sigmoid, cal_isotonic).
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    cal_sigmoid = CalibratedClassifierCV(estimator=tuned_model, method="sigmoid", cv=cv)
    cal_isotonic = CalibratedClassifierCV(estimator=tuned_model, method="isotonic", cv=cv)

    cal_sigmoid.fit(X_train, y_train)
    cal_isotonic.fit(X_train, y_train)

    return cal_sigmoid, cal_isotonic


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    """Standard binned ECE: weighted gap between predicted confidence and observed accuracy."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece, n = 0.0, len(y_true)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_prob > lo) & (y_prob <= hi) if lo > 0 else (y_prob >= lo) & (y_prob <= hi)
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / n) * abs(y_true[mask].mean() - y_prob[mask].mean())
    return ece


def evaluate(name: str, model, X_test, y_test):
    """Returns (metrics_dict, y_prob) for a fitted model with predict_proba."""
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = {
        "model": name,
        "brier_score": brier_score_loss(y_test, y_prob),
        "log_loss": log_loss(y_test, y_prob),
        "ece": expected_calibration_error(np.asarray(y_test), y_prob),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "avg_precision": average_precision_score(y_test, y_prob),
    }
    return metrics, y_prob


def check_gate(metrics_df: pd.DataFrame, brier_target: float = BRIER_TARGET):
    """Returns (best_row, gate_passed: bool). Best = lowest Brier score."""
    best_row = metrics_df.sort_values("brier_score").iloc[0]
    return best_row, bool(best_row["brier_score"] < brier_target)


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------
def plot_diagnostics(y_test, prob_uncal, prob_sigmoid, prob_isotonic, out_path=None):
    """
    Two-panel figure: reliability diagram (uncalibrated vs. both calibrated
    methods vs. perfect calibration) and probability-shift histogram.
    Saves to out_path if given, and returns the figure for inline display.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    ax = axes[0]
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfectly calibrated")
    for label, probs, style in [
        ("Uncalibrated", prob_uncal, "o-"),
        ("Platt (sigmoid)", prob_sigmoid, "s-"),
        ("Isotonic", prob_isotonic, "^-"),
    ]:
        frac_pos, mean_pred = calibration_curve(y_test, probs, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac_pos, style, label=label, linewidth=1.8, markersize=5)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed churn rate")
    ax.set_title("Reliability Diagram")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    ax2 = axes[1]
    bins = np.linspace(0, 1, 31)
    for label, probs in [("Uncalibrated", prob_uncal), ("Platt", prob_sigmoid), ("Isotonic", prob_isotonic)]:
        ax2.hist(probs, bins=bins, alpha=0.5, label=label, density=True)
    ax2.set_xlabel("Predicted churn probability")
    ax2.set_ylabel("Density")
    ax2.set_title("Predicted-Risk Distribution Shift")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    fig.suptitle("Phase 3 — Calibration Diagnostics (LightGBM, Telco Churn)")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if out_path is not None:
        fig.savefig(out_path, dpi=200)

    return fig


# --------------------------------------------------------------------------
# Saving artifacts
# --------------------------------------------------------------------------
def save_artifacts(tuned_model, cal_sigmoid, cal_isotonic, metrics_df, model_dir, report_dir):
    """Saves the 3 model .pkl files and the metrics comparison CSV."""
    model_dir = Path(model_dir)
    report_dir = Path(report_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(tuned_model, model_dir / "lgbm_base_tuned.pkl")
    joblib.dump(cal_sigmoid, model_dir / "lgbm_calibrated_sigmoid.pkl")
    joblib.dump(cal_isotonic, model_dir / "lgbm_calibrated_isotonic.pkl")
    metrics_df.to_csv(report_dir / "phase3_calibration_metrics.csv", index=False)


def write_model_card_inputs(best_params, best_row, gate_passed, X_train, X_test, report_dir,
                             brier_target: float = BRIER_TARGET):
    """Writes raw numbers for Member 5's model card, in Phase 3's own words."""
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / "phase3_model_card_inputs.md"

    with open(path, "w") as f:
        f.write("# Phase 3 Model Card Inputs (raw numbers)\n\n")
        f.write(f"- Best LightGBM hyperparameters:\n```json\n{json.dumps(best_params, indent=2)}\n```\n\n")
        f.write(f"- Chosen calibration method: **{best_row['model']}**\n")
        f.write(f"- Brier Score Loss: {best_row['brier_score']:.4f} "
                f"(target < {brier_target}, {'PASSED' if gate_passed else 'NOT MET'})\n")
        f.write(f"- Log-Loss: {best_row['log_loss']:.4f}\n")
        f.write(f"- Expected Calibration Error (10-bin): {best_row['ece']:.4f}\n")
        f.write(f"- ROC-AUC: {best_row['roc_auc']:.4f}\n")
        f.write(f"- Average Precision (PR-AUC): {best_row['avg_precision']:.4f}\n")
        f.write(f"- Train/test rows: {X_train.shape[0]} / {X_test.shape[0]}\n")

    return path
