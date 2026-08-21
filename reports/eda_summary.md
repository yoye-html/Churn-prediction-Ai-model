
========================================
EDA SUMMARY — TELCO CHURN
========================================

Dataset: 7,043 customers × 23 features
Churn rate: 26.5% positive (mild imbalance — no oversampling needed)

KEY FINDINGS FOR THE TEAM:
--------------------------

1. TOP CHURN DRIVERS (from categorical analysis):
   - Contract type: Month-to-month customers churn at ~42% vs 3% for two-year
   - Internet service: Fiber optic churns at ~42% vs DSL at ~19%
   - Tech Support / Online Security: "No" → ~40% churn, "Yes" → ~15% churn

2. NUMERIC INSIGHT:
   - Tenure: Short-tenure customers churn far more (0-1yr: ~47%)
   - MonthlyCharges & TotalCharges are 0.83 correlated → use SHAP not raw importance

3. DATA QUALITY:
   - TotalCharges had 11 spaces (new customers) → fixed to 0
   - No other missing values
   - No duplicate customerIDs

4. ENCODING SIGNALS FOR MEMBER 2:
   - Contract: ordinal (Month < 1yr < 2yr)
   - SeniorCitizen: already 0/1
   - All other categoricals: binary Yes/No or nominal → OHE or target encoding

5. BUSINESS CONTEXT:
   - High-risk profile: Month-to-month + Fiber optic + no TechSupport + short tenure
   - These 4 factors together may push churn probability above 70%
   - This is what Member 5's budget simulation will target

CHECKPOINT STATUS:
  ✓ Data loaded and validated
  ✓ 6 EDA figures saved to reports/figures/
  ✓ Clean dataset saved to data/processed/telco_clean.csv
  ✓ Encoding signals documented for Member 2
  ✓ Repository updated
========================================
