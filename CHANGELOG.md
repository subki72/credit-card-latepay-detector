# Changelog

All notable changes to this project are documented in this file.

The format adheres to [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-11-27

### Added
- Modular Python package `src/latepay_detector/` (`config.py`, `features.py`, `pipeline.py`, `evaluation.py`, `api.py`) and CLI entrypoint `main.py` (`--health`, `--demo`, `--serve`, `--threshold`, `--synthetic`).
- FastAPI inference and benchmarking service exposing `GET /health`, `POST /predict`, and `POST /analyze` with Pydantic v2 validation, `CORSMiddleware`, thread-safe model caching, and centralized exception handling.
- Automated offline test suite (`tests/test_features.py`, `tests/test_evaluation.py`, `tests/test_pipeline.py`, `tests/test_api.py`, `tests/test_cli_integration.py`) covering exact-case CSV loading, leak-free scaling, threshold tuning, API contracts, and notebook regression guards.
- Multi-stage non-root `Dockerfile`, `docker-compose.yml`, `Makefile`, GitHub Actions CI/CD (`.github/workflows/ci.yml`, `.github/workflows/release.yml`), Dependabot, pre-commit hooks, ADR, and operational runbook.

### Fixed
- **Severe Validation Data Leakage (`PredictionModel.ipynb`)**: Fixed Cells 106-108 so validation features are transformed with training-fitted `pt1`/`pt2` scalers and `X1_val`/`X2_val` concatenate validation features rather than `X1_cat`/`X1_num` training rows.
- **Linux Case-Sensitivity Crash (`PredictionModel.ipynb`)**: Fixed `"Finanku Data Validasi.csv"` -> `"FinanKu Data Validasi.csv"`.
- **Copy-Paste DataFrame Mix (`PredictionModel.ipynb`)**: Fixed `df2_val` expressions referencing `df1_val` columns.
- **Modern Scikit-Learn Compatibility**: Replaced deprecated `max_features='auto'` with `['sqrt', 'log2', None]`, replaced invalid `completeness_score` lines with `precision_score`/`f1_score(..., zero_division=0)`, and replaced `jcopml` with `sklearn.inspection.permutation_importance`.
- **Security Hygiene**: Created `.gitignore` excluding `.env`, `.env.*`, caches, and runtime artifacts, and pinned `requirements.txt` dependencies (`==`) for Python 3.11+ parity.
