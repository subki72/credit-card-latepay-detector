## Summary

Describe what changed and why. Link any related issue or ADR.

## Checklist

- [ ] `python -m ruff check src tests main.py` and `python -m ruff format --check src tests main.py` pass
- [ ] `python -m flake8 --max-line-length=120 src tests main.py` passes
- [ ] `python -m pytest tests/ -v --cov=src --cov-fail-under=75` passes offline
- [ ] No training/validation data leakage (`fit` called strictly on training split)
- [ ] Linux case-sensitive filenames verified (`FinanKu Data All.csv`, `FinanKu Data Validasi.csv`)
- [ ] No secrets or `.env` files committed
