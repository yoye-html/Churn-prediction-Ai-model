"""
src/features.py — Member 2: Feature Engineering + Encoding
Trustworthy Churn Prediction — INSA Summer Camp 5th Batch

Rebuilt against a new draft prompt, reconciled against the LOCKED Phase 1/2
decisions in HANDOFF_SUMMARY.md:
  - Model is LightGBM (tree-based)      -> handle_unknown='ignore', no drop_first
  - Logistic Regression is ALSO benchmarked as a baseline (Section 8)
                                          -> RobustScaler still earns its place
  - TotalCharges already dropped in Phase 1, replaced by charges_per_month
                                          -> this module does NOT recreate it
  - Contract is ordinal (locked)          -> OrdinalEncoder, not OHE
  - "No internet service" -> "No" consolidation is a locked, load-bearing step
                                          -> re-added here; skipping it crashes
                                             the binary OrdinalEncoder downstream

INSA module anchors: M06 (no leakage), M07 (feature justification), M08
(Pipeline/ColumnTransformer discipline).
"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, RobustScaler, FunctionTransformer

# ---------------------------------------------------------------------------
# Columns whose "No internet service" value is redundant with
# InternetService == "No" (locked decision — 6 columns, verified in Phase 2 EDA)
# ---------------------------------------------------------------------------
_INTERNET_DEPENDENT_COLS = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]

# The 6 columns above, once consolidated, become genuine Yes/No columns and
# join the rest of the binary group for ordinal 0/1 encoding.
_BINARY_COLS = [
    "gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling",
    *_INTERNET_DEPENDENT_COLS,
]
_BINARY_CATEGORIES = [
    ["Female", "Male"] if col == "gender" else ["No", "Yes"]
    for col in _BINARY_COLS
]

_ORDINAL_COLS = ["Contract"]
_ORDINAL_CATEGORIES = [["Month-to-month", "One year", "Two year"]]

_OHE_COLS = ["InternetService", "MultipleLines", "PaymentMethod"]

# Continuous columns that a scale-sensitive baseline model (Logistic
# Regression, Section 8) benefits from — LightGBM ignores scaling entirely,
# so this only exists for the baseline comparison, not for the tree model.
_SCALED_NUMERIC_COLS = ["tenure", "MonthlyCharges", "charges_per_month", "cost_per_service"]

# Engineered features that are counts/booleans — deliberately NOT scaled,
# consistent with the original Phase 2 'passthrough' bucket.
_PASSTHROUGH_COLS = [
    "SeniorCitizen", "num_services", "is_senior_with_no_support",
]


def load_target(path: str, default_name: str = "Churn_binary") -> pd.Series:
    """
    Load target labels as a 1D Series.

    Processed y CSV files in this project are often saved with a leading index
    column. This loader drops that index at read-time and returns only the
    target values.
    """
    raw = pd.read_csv(path)

    if default_name in raw.columns:
        y = raw[default_name]
    elif raw.shape[1] == 1:
        y = raw.iloc[:, 0]
    elif raw.shape[1] == 2 and str(raw.columns[0]).startswith("Unnamed"):
        # Legacy format: index column + target column.
        y = raw.iloc[:, 1]
    else:
        raise ValueError(
            f"Could not identify target column in {path}. "
            f"Columns found: {list(raw.columns)}"
        )

    y.name = y.name or default_name
    return y


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pure, stateless feature construction. No fitting happens here — safe to
    apply identically to train and test (M06: no leakage).
    """
    df = df.copy()

    # --- Load-bearing prerequisite: consolidate the redundant 3rd category
    # before anything downstream assumes these columns are binary Yes/No.
    for col in _INTERNET_DEPENDENT_COLS:
        df[col] = df[col].replace("No internet service", "No")
    if "MultipleLines" in df.columns:
        df["MultipleLines"] = df["MultipleLines"].replace("No phone service", "No")

    # --- num_services (the prompt's "TotalServices", renamed to match the
    # already-validated Phase 2 feature — same concept, not a duplicate).
    service_cols = _INTERNET_DEPENDENT_COLS
    df["num_services"] = (df[service_cols] == "Yes").sum(axis=1).astype(int)

    # --- is_senior_with_no_support — NEW, not yet validated against churn
    # rate. Included, but flagged: see validate_new_features() below before
    # trusting this in the final feature_report.md.
    df["is_senior_with_no_support"] = (
        (df["SeniorCitizen"] == 1) & (df["TechSupport"] == "No")
    ).astype(int)

    # --- cost_per_service — genuinely new, doesn't overlap with
    # charges_per_month (that one is tenure-based, from Phase 1).
    # +1 smoothing avoids div-by-zero for customers with 0 add-on services,
    # without a conditional branch, and keeps ordering monotonic.
    df["cost_per_service"] = df["MonthlyCharges"] / (df["num_services"] + 1)

    return df


def validate_new_features(
    X: pd.DataFrame,
    y: pd.Series | None = None,
    target_col: str = "Churn_binary",
    feature_col: str = "is_senior_with_no_support",
) -> pd.DataFrame:
    """
    M07 discipline: don't trust an engineered feature until its churn-rate
    lift is checked, the same way high_value_at_risk / new_fiber_customer /
    no_safety_net were validated in the original Phase 2 table.

    Run this BEFORE finalizing feature_report.md. is_senior_with_no_support
    in particular has not been validated yet on real data.
    """
    if feature_col not in X.columns:
        raise ValueError(
            f"Feature '{feature_col}' not found in input columns. "
            f"Columns include: {list(X.columns)}"
        )

    work = X[[feature_col]].copy()

    if y is None:
        if target_col not in X.columns:
            raise ValueError(
                f"Target '{target_col}' not found in input columns. "
                "Pass y explicitly as a Series or include target in X."
            )
        work[target_col] = X[target_col].values
    else:
        if isinstance(y, pd.DataFrame):
            if y.shape[1] != 1:
                raise ValueError(
                    f"Expected y with a single column, got shape {y.shape}."
                )
            y_series = y.iloc[:, 0]
        elif isinstance(y, pd.Series):
            y_series = y
        else:
            y_series = pd.Series(y)
        y_series = y_series.rename(target_col)
        work = work.join(y_series, how="inner")

    work[feature_col] = (work[feature_col] > 0).astype(int)

    base_rate = work[target_col].mean()
    flagged = work[work[feature_col] == 1][target_col]
    not_flagged = work[work[feature_col] == 0][target_col]

    flagged_rate = flagged.mean() if len(flagged) else np.nan
    not_flagged_rate = not_flagged.mean() if len(not_flagged) else np.nan
    lift_ratio = (flagged_rate / not_flagged_rate) if not_flagged_rate and not np.isnan(not_flagged_rate) else np.nan
    lift_pct_vs_base = ((flagged_rate - base_rate) / base_rate * 100) if base_rate else np.nan

    return pd.DataFrame([
        {
            "feature": feature_col,
            "n_total": int(len(work)),
            "n_flagged": int((work[feature_col] == 1).sum()),
            "flag_rate": (work[feature_col] == 1).mean(),
            "churn_rate_flagged": flagged_rate,
            "churn_rate_not_flagged": not_flagged_rate,
            "base_rate": base_rate,
            "lift_ratio_flagged_vs_not_flagged": lift_ratio,
            "lift_pct_vs_base": lift_pct_vs_base,
        }
    ])


def build_preprocessor() -> ColumnTransformer:
    """
    Encoding strategy, reconciled against the locked LightGBM decision:
      - binary Yes/No  -> OrdinalEncoder to 0/1 (unchanged from original)
      - Contract       -> OrdinalEncoder, explicit monotone order (locked)
      - nominal cats   -> OneHotEncoder(handle_unknown='ignore'), NO
                          drop_first — tree models don't need collinearity
                          protection, and Phase 4's SHAP plots need every
                          category visible to attribute effects correctly.
      - continuous     -> RobustScaler, scoped only to the Logistic
                          Regression baseline's benefit — inert for LightGBM.
      - counts/bools   -> passthrough, unchanged.
    """
    return ColumnTransformer(
        transformers=[
            ("binary", OrdinalEncoder(categories=_BINARY_CATEGORIES), _BINARY_COLS),
            ("ordinal", OrdinalEncoder(categories=_ORDINAL_CATEGORIES), _ORDINAL_COLS),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False), _OHE_COLS),
            ("scaled_numeric", RobustScaler(), _SCALED_NUMERIC_COLS),
            ("passthrough", "passthrough", _PASSTHROUGH_COLS),
        ],
        remainder="drop",  # explicit: customerID / raw target must not leak in
        verbose_feature_names_out=False,
    )


def run_full_pipeline() -> Pipeline:
    """
    The single pipeline object: raw text-and-numeric DataFrame in,
    fully transformed numeric matrix out. Matches the new prompt's
    requirement while reusing the locked encoding decisions above.

    Usage:
        pipe = run_full_pipeline()
        X_train_transformed = pipe.fit_transform(X_train)
        X_test_transformed  = pipe.transform(X_test)   # no re-fitting (M06)
    """
    return Pipeline(steps=[
        ("engineer", FunctionTransformer(engineer_features, validate=False)),
        ("preprocess", build_preprocessor()),
    ])


if __name__ == "__main__":
    # Standalone smoke test against the REAL Phase 1 handoff files.
    # (If you're working through 02_features.ipynb instead, you don't need
    # to run this file directly at all -- the notebook imports these
    # functions and uses the real train/test split there.)
    import os

    train_path = "data/processed/X_train.csv"
    y_path = "data/processed/y_train.csv"

    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"Could not find {train_path}. Run this script from the "
            f"project root (churn_project/), e.g.:\n"
            f"    cd ~/churn_project && python3 src/features.py\n"
            f"Not from inside src/ -- the path above is relative to the "
            f"project root, matching how the notebook loads it."
        )

    X = pd.read_csv(train_path)
    y = load_target(y_path)
    y_col = y.name if hasattr(y, "name") and y.name else "Churn_binary"
    df = X.copy()
    df[y_col] = y.values

    # Validate the new, not-yet-trusted feature BEFORE finalizing anything.
    engineered = engineer_features(df)
    validation_report = validate_new_features(engineered, target_col=y_col)
    print("=== New feature validation (check before trusting) ===")
    print(validation_report.to_string(index=False))
    print()

    drop_cols = [c for c in [y_col, "customerID"] if c in X.columns]
    X_features = X.drop(columns=drop_cols)
    pipe = run_full_pipeline()
    X_transformed = pipe.fit_transform(X_features)

    feature_names = pipe.named_steps["preprocess"].get_feature_names_out()
    print("=== Pipeline smoke test ===")
    print("Input shape: ", X_features.shape)
    print("Output shape:", X_transformed.shape)
    print("Output dtype:", X_transformed.dtype)
    print("Feature names:", list(feature_names))
    print("Any NaNs in output:", np.isnan(X_transformed).any())