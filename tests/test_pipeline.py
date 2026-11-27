"""Unit and regression tests for leak-free ML pipeline and exact-case CSV paths."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.latepay_detector.features import build_horizon_features
from src.latepay_detector.pipeline import (
    LatePayPipeline,
    generate_synthetic_finanku_data,
    resolve_exact_case_path,
    run_experiment,
)


def test_resolve_exact_case_path_catches_linux_casing_typo(tmp_path: Path) -> None:
    # Arrange
    correct_file = tmp_path / "FinanKu Data Validasi.csv"
    correct_file.write_text("col\n1\n", encoding="utf-8")

    # Act
    resolved = resolve_exact_case_path(tmp_path, "FinanKu Data Validasi.csv")

    # Assert
    assert resolved.name == "FinanKu Data Validasi.csv"
    with pytest.raises(FileNotFoundError, match="Finanku Data Validasi.csv"):
        resolve_exact_case_path(tmp_path, "Finanku Data Validasi.csv")


def test_repo_validation_csv_exact_filename_when_present() -> None:
    # Arrange
    repo_root = Path(__file__).resolve().parent.parent
    if not (repo_root / "FinanKu Data Validasi.csv").exists():
        pytest.skip("Repository CSV datasets not mounted in isolated directory.")

    # Act
    resolved = resolve_exact_case_path(repo_root, "FinanKu Data Validasi.csv")

    # Assert
    assert resolved.exists()
    with pytest.raises(FileNotFoundError):
        resolve_exact_case_path(repo_root, "Finanku Data Validasi.csv")


def test_pipeline_does_not_refit_scaler_on_validation_and_handles_unseen_categories() -> None:
    # Arrange
    train_df = generate_synthetic_finanku_data(n_rows=100, random_state=11, validation=False)
    val_df = generate_synthetic_finanku_data(n_rows=20, random_state=12, validation=True)
    x_train, y_train = build_horizon_features(train_df, horizon="6m")
    x_val, _ = build_horizon_features(val_df, horizon="6m")
    assert y_train is not None

    model = LatePayPipeline(horizon="6m", model_type="logreg", threshold=0.30).fit(x_train, y_train)
    baseline_probs = model.predict_proba(x_val)

    # Shift a single validation row dramatically and introduce unseen Branch/City values;
    # other rows' predictions must remain strictly invariant (proving no val refit).
    shifted_val = x_val.copy()
    shifted_val.iloc[0, shifted_val.columns.get_loc("Mean Balance")] = 9.0e9
    shifted_val.iloc[1, shifted_val.columns.get_loc("Branch Code")] = "9999"
    shifted_val.iloc[1, shifted_val.columns.get_loc("City")] = "Medan"

    # Act
    shifted_probs = model.predict_proba(shifted_val)

    # Assert
    assert np.allclose(baseline_probs[2:], shifted_probs[2:])


def test_run_experiment_and_artifact_roundtrip(tmp_path: Path) -> None:
    # Arrange
    train_df = generate_synthetic_finanku_data(n_rows=120, random_state=42, validation=False)
    val_df = generate_synthetic_finanku_data(n_rows=80, random_state=43, validation=True)

    # Act
    result, fitted = run_experiment(
        train_df=train_df,
        val_df=val_df,
        horizon="6m",
        model_type="rf",
        threshold=0.30,
        random_state=42,
    )
    artifact_file = fitted.save(tmp_path / "model.joblib")
    reloaded = LatePayPipeline.load(artifact_file)
    x_val, _ = build_horizon_features(val_df, horizon="6m")

    # Assert
    assert result.val_metrics_threshold.recall >= result.val_metrics_default.recall
    assert np.allclose(fitted.predict_proba(x_val), reloaded.predict_proba(x_val))


def test_pipeline_error_guards() -> None:
    # Arrange
    unfitted = LatePayPipeline(horizon="6m", model_type="logreg", threshold=0.30)
    train_df = generate_synthetic_finanku_data(n_rows=40, random_state=5, validation=False)
    x_train, y_train = build_horizon_features(train_df, horizon="6m")
    assert y_train is not None

    # Act & Assert
    with pytest.raises(RuntimeError, match="fitted"):
        unfitted.predict_proba(x_train)
    with pytest.raises(ValueError, match="Threshold"):
        LatePayPipeline(threshold=0.0)
    with pytest.raises(ValueError, match="both classes"):
        unfitted.fit(x_train, y_train * 0)
    with pytest.raises(ValueError, match="Unsupported model_type"):
        LatePayPipeline(model_type="svm").fit(x_train, y_train)  # type: ignore[arg-type]
