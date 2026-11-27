# Contributing Guide

Thank you for contributing to `credit-card-latepay-detector`.

## Development Setup

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows:     .venv\Scripts\activate
pip install -r requirements.txt
pip install ruff flake8 mypy pre-commit
pre-commit install
```

## Engineering Standards

1. **Zero Data Leakage**: Never fit `StandardScaler`, `OneHotEncoder`, or `SMOTE` outside the training fold. Keep transformations inside `LatePayPipeline` / `imblearn.pipeline.Pipeline`.
2. **Linux Case Sensitivity**: Verify exact filenames (`FinanKu Data All.csv`, `FinanKu Data Validasi.csv`) via `resolve_exact_case_path`.
3. **Offline-Safe Tests**: All unit and integration tests under `tests/` must run deterministically offline without network dependencies.
4. **Code Quality Gates**: Every pull request must pass all local checks:

```bash
python -m ruff check src tests main.py
python -m ruff format --check src tests main.py
python -m flake8 --max-line-length=120 src tests main.py
python -m pytest tests/ -v --cov=src --cov-fail-under=75
```
