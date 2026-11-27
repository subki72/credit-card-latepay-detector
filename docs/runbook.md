# Operational Runbook: FinanKu Late-Payment Detector

## 1. Service Health Verification

Check local CLI or HTTP liveness probe:

```bash
python main.py --health
curl -sf http://127.0.0.1:8000/health
```

Expected response (`200 OK`):

```json
{
  "status": "ok",
  "service": "credit-card-latepay-detector",
  "version": "1.0.0",
  "default_horizon": "6m",
  "default_model_type": "logreg",
  "default_threshold": 0.3
}
```

## 2. Common Failure Scenarios & Triage

| Symptom | Root Cause | Immediate Remediation |
|---|---|---|
| `FileNotFoundError: Exact filename 'FinanKu Data Validasi.csv' not found` | Case mismatch on Linux containers (`Finanku` vs `FinanKu`) or missing mounted CSV | Verify exact casing of CSVs in `LATEPAY_DATA_DIR` or pass `--synthetic` for smoke runs |
| HTTP `422` on `/predict` | Input record violates Pydantic schema bounds (`age` outside `[18, 105]`, empty batch, or `threshold` outside `(0, 1)`) | Inspect JSON payload against `CustomerFeatureInput` schema |
| Recall drops below SLA (`< 65%`) on OOT validation | Decision threshold reset to default `0.50` or concept drift across quarters | Verify `LATEPAY_THRESHOLD=0.30` in environment; run `POST /analyze` comparing `logreg` vs `rf` on latest quarterly holdout |
| Slow container cold start | First `POST /predict` trains pipeline in-memory when `artifacts/` is empty | Pre-warm or mount serialized `artifacts/latepay_6m_logreg.joblib` |

## 3. Threshold & Model Rollback

1. Adjust `LATEPAY_THRESHOLD` in `.env` (lower threshold increases Recall / collection sensitivity; higher threshold raises Precision).
2. Switch `LATEPAY_MODEL_TYPE` between `logreg`, `rf`, and `xgb` via env configuration without code changes.
3. Restart service (`docker compose up -d --build`) and verify `GET /health` reports updated defaults.
