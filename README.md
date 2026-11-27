# FinanKu Credit Card Late-Payment Detector

[![CI Pipeline](https://github.com/subki72/credit-card-latepay-detector/actions/workflows/ci.yml/badge.svg)](https://github.com/subki72/credit-card-latepay-detector/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Recall-oriented credit card delinquency prediction system engineered to identify high-risk banking customers across 12-month (`Q1-Q4`) and 6-month (`Q3-Q4`) quarterly behavioral horizons, validated strictly out-of-time (`Q2-Q5`) using a leak-free `scikit-learn` + `SMOTE` pipeline, CLI runner, and FastAPI inference service.

---

## Business Overview & Risk Philosophy

In retail banking collections, missing an actual delinquent customer (**False Negative**) incurs direct principal charge-off losses, whereas contacting a punctual customer (**False Positive**) only incurs operational outreach cost. Standard classifiers trained on imbalanced portfolios achieve high nominal accuracy (~76-80%) while catching under **22%** of actual out-of-time defaults at threshold `0.50`.

This project combines:
1. **Horizon-Based Behavioral Features**: Compares 12-month (`12m`) vs. 6-month (`6m`) balance velocity (`Mean Balance`, `Delta Balance`), customer activity (`Active Months`), product holding delta (`Diff PH`), and card tenure (`Vintage_CR`).
2. **Leak-Free SMOTE Oversampling**: Applies `StandardScaler`, `OneHotEncoder(handle_unknown="ignore")`, and `SMOTE` strictly inside `imblearn.pipeline.Pipeline` during training folds.
3. **Risk-Averse Threshold Tuning (`0.30`)**: Lowers the decision cutoff from `0.50` to `0.30` to prioritize delinquency Recall.

---

## Stage 1 Audit Remediation & Leak-Free Benchmark

During production-readiness remediation, critical defects in the original `PredictionModel.ipynb` were diagnosed and fixed directly in both the notebook and modular `src/latepay_detector/` package:

- **Fixed Validation Data Leakage (Cells 106-108)**: Replaced `pt.fit_transform` on validation data with training-fitted `pt1.transform` / `pt2.transform`, and fixed `X1_val` / `X2_val` mistakenly concatenating `X1_cat` / `X1_num` training rows instead of validation rows.
- **Fixed Linux Case-Sensitivity Bug (Cell 84)**: Corrected `"Finanku Data Validasi.csv"` to `"FinanKu Data Validasi.csv"` and enforced exact-case resolution in `resolve_exact_case_path`.
- **Fixed Cross-DataFrame Copy-Paste References (Cells 88, 91, 93)**: Corrected `df2_val` feature formulas referencing `df1_val`.
- **Removed Broken/Deprecated Dependencies**: Replaced `%pip install`, `jcopml` (`mean_score_decrease`), `max_features='auto'`, and invalid `metrics.completeness_score` calls with `sklearn.inspection.permutation_importance` and `precision_score` / `f1_score(..., zero_division=0)`.

### Verified Out-of-Time Validation Benchmarks (`FinanKu Data All.csv` -> `FinanKu Data Validasi.csv`, 7,561 rows)

*Historical pre-fix notebook baseline reported 81.65% recall under leaky validation alignment; below are verified leak-free out-of-time (`Q2-Q5`) benchmarks reproduced via `python main.py --demo --horizon both --threshold 0.30` (`random_state=42`):*

| Horizon | Model | Threshold | Test Recall | OOT Val Recall | OOT Val Precision | OOT Val Accuracy | OOT Val F1 | ROC-AUC |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `12m` | Logistic Regression | `0.50` | 45.87% | 21.74% | 50.13% | 64.21% | 30.33% | 0.5436 |
| `12m` | Logistic Regression | **`0.30`** | **95.05%** | **66.52%** | 36.57% | 46.67% | 47.20% | 0.5436 |
| `6m` | Logistic Regression | `0.50` | 42.74% | 19.27% | 54.32% | 65.27% | 28.45% | 0.5545 |
| `6m` | Logistic Regression | **`0.30`** | **97.69%** | **86.01%** | 36.03% | 40.29% | **50.79%** | **0.5545** |

---

## Project Architecture

```text
credit-card-latepay-detector/
├── src/latepay_detector/
│   ├── config.py          # Typed env settings & structured logging
│   ├── features.py        # Leak-free 12m (Q1-Q4) / 6m (Q3-Q4) & OOT (Q2-Q5) feature builder
│   ├── pipeline.py        # ImbPipeline (Scaler + OHE + SMOTE + LogReg/RF/XGB) & CSV loader
│   ├── evaluation.py      # Threshold-adjusted metrics & permutation importance
│   └── api.py             # FastAPI service (/health, /predict, /analyze)
├── tests/                 # Offline unit & API/CLI integration tests (>= 75% coverage)
├── docs/
│   ├── adr/001-leak-free-delinquency-pipeline.md
│   └── runbook.md
├── main.py                # CLI entrypoint (--health, --demo, --serve)
├── PredictionModel.ipynb  # Patched exploratory modeling notebook
├── Dockerfile             # Multi-stage Python 3.11-slim non-root container
├── docker-compose.yml     # Single-command service orchestration
└── Makefile               # Standard dev/lint/test/run targets
```

---

## Quickstart

### 1. Local Installation

```bash
git clone https://github.com/subki72/credit-card-latepay-detector.git
cd credit-card-latepay-detector
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

### 2. CLI Commands

```bash
# Service & dataset readiness check
python main.py --health

# Run leak-free Experiment 1 (12m) & Experiment 2 (6m) benchmark
python main.py --demo --horizon both --model logreg --threshold 0.30

# Fast offline synthetic smoke demo
python main.py --demo --synthetic --threshold 0.30

# Start FastAPI server on port 8000
python main.py --serve
```

### 3. FastAPI Endpoints & Example Request

- `GET /health` — Service liveness and active configuration defaults.
- `POST /predict` — Batch delinquency scoring with configurable risk threshold (`0.30` default).
- `POST /analyze` — On-demand leak-free `12m`/`6m` comparative benchmark (`use_synthetic=true` supported).

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "threshold": 0.30,
    "horizon": "6m",
    "records": [{
      "customer_id": 15565701,
      "age": 48,
      "avg_annual_income_month": 18000000,
      "mean_balance": 1200000,
      "delta_balance": -2500000,
      "active_months": 0,
      "diff_ph": -1,
      "vintage_cr": 6,
      "branch_code": "1001",
      "city": "Jakarta"
    }]
  }'
```

### 4. Docker & Quality Verification

```bash
# Container build & run
docker compose up --build

# Code quality & offline tests
make lint
make test
```

---

## Documentation & Governance

- [Architecture Decision Record (ADR 001)](docs/adr/001-leak-free-delinquency-pipeline.md)
- [Operational Runbook](docs/runbook.md)
- [Contributing Guide](CONTRIBUTING.md)
- [Security Policy](SECURITY.md)
- [Changelog](CHANGELOG.md)

**Author**: Muhammad Syafii Assubki ([@subki72](https://github.com/subki72)) · **License**: [MIT](LICENSE)
