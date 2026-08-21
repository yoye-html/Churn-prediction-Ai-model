"""
Data validation script — Member 1
Run this first before any notebook work.
"""
import pandas as pd
import sys

def validate_telco(path):
    print("=" * 50)
    print("TELCO DATA VALIDATION")
    print("=" * 50)

    df = pd.read_csv(path)

    # 1. Shape check
    print(f"\n[1] Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    assert df.shape[0] > 6000, "Too few rows — file may be incomplete"
    assert df.shape[1] == 21, f"Expected 21 columns, got {df.shape[1]}"
    print("    ✓ Shape is correct")

    # 2. Target column check
    assert "Churn" in df.columns, "Missing target column 'Churn'"
    churn_rate = (df["Churn"] == "Yes").mean()
    print(f"\n[2] Target 'Churn': {churn_rate:.1%} positive rate")
    assert 0.20 < churn_rate < 0.35, "Churn rate looks wrong"
    print("    ✓ Churn rate in expected range (~26%)")

    # 3. Duplicates
    n_dup = df.duplicated(subset="customerID").sum()
    print(f"\n[3] Duplicate customerIDs: {n_dup}")
    assert n_dup == 0, "Found duplicate customer IDs"
    print("    ✓ No duplicates")

    # 4. Missing values
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        print(f"\n[4] Missing values found:")
        print(missing)
    else:
        print(f"\n[4] No null values ✓")

    # 5. TotalCharges type issue (known Kaggle artifact)
    print(f"\n[5] TotalCharges dtype: {df['TotalCharges'].dtype}")
    print("    Note: TotalCharges is stored as string — spaces instead of NaN")
    spaces = (df["TotalCharges"].str.strip() == "").sum()
    print(f"    Empty TotalCharges (new customers): {spaces}")

    print("\n" + "=" * 50)
    print("VALIDATION PASSED — data is ready for EDA")
    print("=" * 50)
    return df

if __name__ == "__main__":
    path = "data/raw/telco_churn.csv"
    df = validate_telco(path)