"""Unit tests for leak-free quarterly horizon feature engineering."""

from __future__ import annotations

import pandas as pd
import pytest

from src.latepay_detector.features import (
    CANONICAL_FEATURES,
    build_horizon_features,
    compute_credit_vintage,
    detect_quarter_window,
)
from src.latepay_detector.pipeline import generate_synthetic_finanku_data


def test_compute_credit_vintage_prioritizes_earliest_quarter() -> None:
    # Arrange
    df = pd.DataFrame(
        {
            "HasCrCard Q1": [1, 0, 0, 0],
            "HasCrCard Q2": [1, 1, 0, 0],
            "HasCrCard Q3": [1, 1, 1, 0],
        }
    )

    # Act
    vintage = compute_credit_vintage(df, ("HasCrCard Q1", "HasCrCard Q2", "HasCrCard Q3"))

    # Assert
    assert vintage.tolist() == [12, 9, 6, 3]


def test_build_horizon_features_aligns_train_and_validation_schemas() -> None:
    # Arrange
    train_df = generate_synthetic_finanku_data(n_rows=40, random_state=7, validation=False)
    val_df = generate_synthetic_finanku_data(n_rows=30, random_state=8, validation=True)

    # Act
    x_train_12m, y_train_12m = build_horizon_features(train_df, horizon="12m")
    x_val_12m, y_val_12m = build_horizon_features(val_df, horizon="12m")
    x_train_6m, _ = build_horizon_features(train_df, horizon="6m")
    x_val_6m, _ = build_horizon_features(val_df, horizon="6m")

    # Assert
    assert list(x_train_12m.columns) == CANONICAL_FEATURES
    assert list(x_val_12m.columns) == CANONICAL_FEATURES
    assert list(x_train_6m.columns) == CANONICAL_FEATURES
    assert list(x_val_6m.columns) == CANONICAL_FEATURES
    assert y_train_12m is not None and len(y_train_12m) == 40
    assert y_val_12m is not None and len(y_val_12m) == 30


def test_build_horizon_features_uses_exact_6m_validation_window_without_mutation() -> None:
    # Arrange
    val_df = pd.DataFrame(
        {
            "Customer ID": [101],
            "Branch Code": [1001.0],
            "City": [" Jakarta "],
            "Age": [35],
            "Avg. Annual Income/Month": [12_000_000],
            "Balance Q2": [1_000.0],
            "Balance Q3": [2_000.0],
            "Balance Q4": [4_000.0],
            "Balance Q5": [8_000.0],
            "NumOfProducts Q2": [1],
            "NumOfProducts Q3": [1],
            "NumOfProducts Q4": [2],
            "NumOfProducts Q5": [3],
            "HasCrCard Q2": [0],
            "HasCrCard Q3": [1],
            "HasCrCard Q4": [1],
            "HasCrCard Q5": [1],
            "ActiveMember Q2": [0],
            "ActiveMember Q3": [0],
            "ActiveMember Q4": [1],
            "ActiveMember Q5": [1],
            "Unpaid Tagging": [1],
        }
    )
    original_cols = list(val_df.columns)

    # Act
    features, target = build_horizon_features(val_df, horizon="6m", window="validation")

    # Assert
    assert list(val_df.columns) == original_cols
    assert float(features.loc[0, "Mean Balance"]) == pytest.approx(6_000.0)
    assert float(features.loc[0, "Delta Balance"]) == pytest.approx(6_000.0)
    assert float(features.loc[0, "Active Months"]) == pytest.approx(6.0)
    assert float(features.loc[0, "Diff PH"]) == pytest.approx(2.0)
    assert float(features.loc[0, "Vintage_CR"]) == pytest.approx(9.0)
    assert features.loc[0, "Branch Code"] == "1001"
    assert features.loc[0, "City"] == "Jakarta"
    assert target is not None and int(target.iloc[0]) == 1


def test_build_horizon_features_rejects_invalid_inputs() -> None:
    # Arrange
    empty_df = pd.DataFrame()
    valid_df = generate_synthetic_finanku_data(n_rows=25, random_state=1, validation=False)
    null_df = valid_df.copy()
    null_df.loc[0, "Balance Q4"] = None

    # Act & Assert
    with pytest.raises(ValueError, match="Unsupported horizon"):
        build_horizon_features(valid_df, horizon="3m")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="missing required"):
        detect_quarter_window(empty_df)

    with pytest.raises(ValueError, match="null values"):
        build_horizon_features(null_df, horizon="6m")
