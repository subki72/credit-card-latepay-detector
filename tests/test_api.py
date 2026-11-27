"""API integration tests for FastAPI inference and benchmarking service."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from src.latepay_detector.api import create_app
from src.latepay_detector.config import Settings


def _make_test_client(tmp_path: Path) -> TestClient:
    """Create isolated offline TestClient backed by synthetic fallback data."""
    settings = Settings(
        data_dir=tmp_path / "missing_csv_dir",
        artifacts_dir=tmp_path / "artifacts",
        default_horizon="6m",
        default_model_type="logreg",
        default_threshold=0.30,
        random_state=42,
    )
    return TestClient(create_app(settings=settings), raise_server_exceptions=False)


def test_health_endpoint_returns_service_metadata(tmp_path: Path) -> None:
    # Arrange
    client = _make_test_client(tmp_path)

    # Act
    response = client.get("/health")

    # Assert
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["default_threshold"] == 0.3
    assert body["default_horizon"] == "6m"


def test_predict_endpoint_scores_records_and_respects_threshold(tmp_path: Path) -> None:
    # Arrange
    client = _make_test_client(tmp_path)
    payload = {
        "threshold": 0.30,
        "horizon": "6m",
        "records": [
            {
                "customer_id": 15565701,
                "age": 48,
                "avg_annual_income_month": 18_000_000,
                "mean_balance": 1_200_000,
                "delta_balance": -2_500_000,
                "active_months": 0,
                "diff_ph": -1,
                "vintage_cr": 6,
                "branch_code": "1001",
                "city": "Jakarta",
            },
            {
                "customer_id": 15565878,
                "age": 29,
                "avg_annual_income_month": 33_000_000,
                "mean_balance": 18_000_000,
                "delta_balance": 3_000_000,
                "active_months": 6,
                "diff_ph": 1,
                "vintage_cr": 12,
                "branch_code": "1005",
                "city": "Bandung",
            },
        ],
    }

    # Act
    response = client.post("/predict", json=payload)

    # Assert
    assert response.status_code == 200
    data = response.json()
    assert data["threshold_applied"] == 0.3
    assert len(data["predictions"]) == 2
    for pred in data["predictions"]:
        assert 0.0 <= pred["delinquency_probability"] <= 1.0
        assert pred["predicted_late_pay"] in (0, 1)
        assert pred["risk_tier"] in ("HIGH", "MEDIUM", "LOW")


def test_predict_endpoint_validates_bad_payload_and_analyze_runs_offline(tmp_path: Path) -> None:
    # Arrange
    client = _make_test_client(tmp_path)

    # Act
    bad_res = client.post("/predict", json={"records": []})
    analyze_res = client.post(
        "/analyze",
        json={
            "horizons": ["12m", "6m"],
            "model_type": "logreg",
            "threshold": 0.30,
            "use_synthetic": True,
        },
    )

    # Assert
    assert bad_res.status_code == 422
    assert analyze_res.status_code == 200
    results = analyze_res.json()["results"]
    assert "12m" in results and "6m" in results
    assert results["6m"]["val_metrics_threshold"]["recall"] >= results["6m"]["val_metrics_default"]["recall"]


def test_global_exception_handler_masks_unexpected_errors(tmp_path: Path, monkeypatch: object) -> None:
    # Arrange
    client = _make_test_client(tmp_path)

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("sensitive internal traceback detail")

    monkeypatch.setattr("src.latepay_detector.api.run_benchmark", _boom)  # type: ignore[attr-defined]

    # Act
    res = client.post("/analyze", json={"use_synthetic": True})

    # Assert
    assert res.status_code == 500
    assert res.json() == {"detail": "Internal server error"}
