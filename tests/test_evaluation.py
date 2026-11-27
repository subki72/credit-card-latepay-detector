"""Unit tests for threshold-aware evaluation and permutation importance."""

from __future__ import annotations

import numpy as np
import pytest

from src.latepay_detector.evaluation import (
    compute_feature_importance,
    evaluate_predictions,
    predict_with_threshold,
)
from src.latepay_detector.features import build_horizon_features
from src.latepay_detector.pipeline import LatePayPipeline, generate_synthetic_finanku_data


def test_predict_with_threshold_increases_sensitivity_at_030() -> None:
    # Arrange
    probabilities = np.array([0.15, 0.32, 0.45, 0.80])

    # Act
    preds_050 = predict_with_threshold(probabilities, threshold=0.50)
    preds_030 = predict_with_threshold(probabilities, threshold=0.30)

    # Assert
    assert preds_050.tolist() == [0, 0, 0, 1]
    assert preds_030.tolist() == [0, 1, 1, 1]


def test_evaluate_predictions_computes_recall_and_precision_accurately() -> None:
    # Arrange
    y_true = np.array([0, 1, 1, 0])
    probabilities = np.array([0.20, 0.35, 0.80, 0.40])

    # Act
    metrics_030 = evaluate_predictions(y_true, probabilities, threshold=0.30)
    metrics_050 = evaluate_predictions(y_true, probabilities, threshold=0.50)

    # Assert
    assert metrics_030.recall == pytest.approx(1.0)
    assert metrics_030.precision == pytest.approx(2.0 / 3.0)
    assert metrics_050.recall == pytest.approx(0.5)
    assert metrics_030.recall > metrics_050.recall
    assert metrics_030.sample_count == 4


def test_evaluate_predictions_handles_single_class_and_invalid_inputs() -> None:
    # Arrange
    y_single = np.array([0, 0, 0])
    probs = np.array([0.1, 0.2, 0.3])

    # Act
    single_metrics = evaluate_predictions(y_single, probs, threshold=0.30)

    # Assert
    assert single_metrics.roc_auc == 0.0
    with pytest.raises(ValueError, match="Threshold"):
        predict_with_threshold(probs, threshold=1.0)
    with pytest.raises(ValueError, match="finite values"):
        predict_with_threshold(np.array([0.2, 1.2]), threshold=0.3)
    with pytest.raises(ValueError, match="length"):
        evaluate_predictions(np.array([0, 1]), probs, threshold=0.3)


def test_compute_feature_importance_returns_ranked_features() -> None:
    # Arrange
    train_df = generate_synthetic_finanku_data(n_rows=80, random_state=42, validation=False)
    features, target = build_horizon_features(train_df, horizon="6m")
    assert target is not None
    pipeline = LatePayPipeline(horizon="6m", model_type="logreg", threshold=0.30).fit(features, target)

    # Act
    importances = compute_feature_importance(pipeline.estimator, features, target, top_k=5)

    # Assert
    assert len(importances) == 5
    assert set(importances[0].keys()) == {"feature", "importance_mean", "importance_std"}
    scores = [float(item["importance_mean"]) for item in importances]
    assert scores == sorted(scores, reverse=True)
    with pytest.raises(ValueError, match="top_k"):
        compute_feature_importance(pipeline.estimator, features, target, top_k=0)
