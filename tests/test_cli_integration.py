"""CLI integration and notebook regression safety tests."""

from __future__ import annotations

import json
from pathlib import Path

from main import main


def test_cli_health_returns_zero_and_valid_json(capsys: object) -> None:
    # Act
    exit_code = main(["--health"])
    captured = capsys.readouterr()  # type: ignore[attr-defined]

    # Assert
    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["status"] == "ok"
    assert payload["default_threshold"] == 0.3


def test_cli_demo_synthetic_runs_both_horizons(capsys: object) -> None:
    # Act
    exit_code = main(["--demo", "--synthetic", "--horizon", "both", "--threshold", "0.3"])
    captured = capsys.readouterr()  # type: ignore[attr-defined]

    # Assert
    assert exit_code == 0
    summary = json.loads(captured.out)
    assert "12m" in summary and "6m" in summary
    assert summary["6m"]["val_metrics_threshold"]["recall"] >= summary["6m"]["val_metrics_default"]["recall"]


def test_cli_rejects_invalid_threshold() -> None:
    # Act
    exit_code = main(["--threshold", "1.5"])

    # Assert
    assert exit_code == 2


def test_notebook_patches_prevent_leakage_and_linux_case_regressions() -> None:
    # Arrange
    nb_path = Path(__file__).resolve().parent.parent / "PredictionModel.ipynb"
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    all_source = "\n".join("".join(cell.get("source", [])) for cell in nb["cells"])

    # Assert
    forbidden_patterns = [
        "%pip install",
        "jcopml",
        "Finanku Data Validasi.csv",
        "pt.fit_transform(X1_num_val)",
        "X1_val = pd.concat([X1_cat, X1_num]",
        "X2_val = pd.concat([X2_cat, X2_num]",
        "completeness_score",
        "'auto'",
    ]
    for pattern in forbidden_patterns:
        assert pattern not in all_source, f"Regression found in PredictionModel.ipynb: {pattern}"
    for line in all_source.splitlines():
        if line.strip().startswith("df2_val["):
            assert "df1_val" not in line, f"Cross-DataFrame copy-paste leak in df2_val line: {line}"
    assert "FinanKu Data Validasi.csv" in all_source
    assert "pt1.transform(X1_num_val)" in all_source
    assert "pt2.transform(X2_num_val)" in all_source
    assert "permutation_importance" in all_source
