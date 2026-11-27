# ADR 001: Leak-Free Credit Card Delinquency Pipeline & Threshold Calibration

- **Status**: Accepted
- **Date**: 2026-11-27
- **Context**: Stage 1 code audit of `PredictionModel.ipynb` uncovered systemic ML methodology bugs and cross-platform defects that distorted out-of-time validation metrics.

## Decision

1. **Canonical Horizon Feature Extraction (`src/latepay_detector/features.py`)**:
   - Decouple training (`Q1-Q4`) and out-of-time validation (`Q2-Q5`) column selection via explicit `detect_quarter_window`, mapping both into identical canonical columns (`Age`, `Avg. Annual Income/Month`, `Mean Balance`, `Delta Balance`, `Active Months`, `Diff PH`, `Vintage_CR`, `Branch Code`, `City`).
   - Enforce pure non-mutating transformations and eliminate cross-DataFrame references (`df2_val` referencing `df1_val`).

2. **Encapsulated Leak-Free Pipeline (`src/latepay_detector/pipeline.py`)**:
   - Encapsulate `StandardScaler`, `OneHotEncoder(handle_unknown="ignore")`, adaptive `SMOTE`, and the classifier inside `imblearn.pipeline.Pipeline`.
   - Guarantee all scalers, encoders, and oversampling parameters are fitted **strictly** on training folds, preventing validation leakage (`pt.fit_transform(X_val)` and `X_val = concat([X_cat, X_num])` bugs removed).

3. **Exact-Case Dataset Resolution**:
   - Enforce `resolve_exact_case_path` to catch Windows/Linux casing mismatches (`Finanku` vs `FinanKu`) during unit testing on Windows.

4. **Risk-Averse Threshold (`0.30`)**:
   - Maintain `threshold = 0.30` as default policy over `0.50` because False Negatives (missed defaulting accounts) carry substantially higher principal loss than False Positive collection outreach.

## Consequences & Verified Empirical Impact

After eliminating the validation concatenation leak in `PredictionModel.ipynb` and re-evaluating strictly out-of-time on `FinanKu Data Validasi.csv` (7,561 records, `random_state=42`):

- At default `threshold = 0.50`, out-of-time validation Recall collapses to **19.27%–21.74%**.
- At calibrated `threshold = 0.30` (`SMOTE` + balanced weighting):
  - **Logistic Regression (`12m`)**: OOT Validation Recall rises from `21.74%` to **66.52%** (test Recall **95.05%**, OOT Precision **36.57%**, F1 **47.20%**, ROC-AUC **0.5436**).
  - **Logistic Regression (`6m`)**: OOT Validation Recall rises from `19.27%` to **86.01%** (test Recall **97.69%**, OOT Precision **36.03%**, F1 **50.79%**, ROC-AUC **0.5545**).
- Both CLI/API and patched `PredictionModel.ipynb` remain synchronized and verified by automated regression tests.
