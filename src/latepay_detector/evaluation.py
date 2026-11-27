"""Threshold-aware evaluation metrics and permutation feature importance."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvaluationMetrics:
    """Immutable binary classification metrics computed at a decision threshold."""

    accuracy: float
    recall: float
    precision: float
    f1: float
    roc_auc: float
    positive_rate: float
    threshold: float
    sample_count: int

    def to_dict(self) -> dict[str, float | int]:
        """Serialize metrics with rounded floating-point values for reporting."""
        payload = asdict(self)
        for key, value in payload.items():
            if isinstance(value, float):
                payload[key] = round(value, 4)
        return payload


def predict_with_threshold(probabilities: np.ndarray, threshold: float = 0.30) -> np.ndarray:
    """Convert positive-class probabilities into binary predictions.

    Args:
        probabilities: 1D array of predicted delinquency probabilities in ``[0, 1]``.
        threshold: Risk cutoff strictly within ``(0, 1)``.

    Returns:
        1D integer array of binary risk flags (``0`` or ``1``).

    Raises:
        ValueError: If threshold or probability shape/bounds are invalid.
    """
    if not 0.0 < threshold < 1.0:
        raise ValueError(f"Threshold must be strictly between 0 and 1, got {threshold}.")

    probs = np.asarray(probabilities, dtype=float)
    if probs.ndim != 1 or probs.size == 0:
        raise ValueError("Probabilities must be a non-empty 1D array.")
    if np.any(np.isnan(probs)) or np.any((probs < 0.0) | (probs > 1.0)):
        raise ValueError("Probabilities must contain finite values within [0, 1].")

    return (probs >= threshold).astype(int)


def evaluate_predictions(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.30,
) -> EvaluationMetrics:
    """Compute recall-first classification metrics at a specified probability threshold.

    Args:
        y_true: Ground-truth binary delinquency labels (0/1).
        probabilities: Predicted positive-class probabilities.
        threshold: Decision threshold applied to ``probabilities``.

    Returns:
        Populated ``EvaluationMetrics`` summary.

    Raises:
        ValueError: When ground truth and probability lengths do not match.
    """
    y_arr = np.asarray(y_true, dtype=int)
    probs = np.asarray(probabilities, dtype=float)
    if y_arr.shape[0] != probs.shape[0]:
        raise ValueError(f"Target length ({y_arr.shape[0]}) does not match probabilities length ({probs.shape[0]}).")

    preds = predict_with_threshold(probs, threshold=threshold)
    unique_classes = np.unique(y_arr)
    roc_auc = float(roc_auc_score(y_arr, probs)) if unique_classes.size >= 2 else 0.0

    return EvaluationMetrics(
        accuracy=float(accuracy_score(y_arr, preds)),
        recall=float(recall_score(y_arr, preds, zero_division=0)),
        precision=float(precision_score(y_arr, preds, zero_division=0)),
        f1=float(f1_score(y_arr, preds, zero_division=0)),
        roc_auc=roc_auc,
        positive_rate=float(np.mean(preds)),
        threshold=float(threshold),
        sample_count=int(y_arr.shape[0]),
    )


def compute_feature_importance(
    estimator: Any,
    features: pd.DataFrame,
    target: pd.Series | np.ndarray,
    top_k: int = 10,
    random_state: int = 42,
) -> list[dict[str, float | str]]:
    """Compute permutation feature importance replacing legacy jcopml calls.

    Args:
        estimator: Fitted estimator or pipeline supporting ``predict``.
        features: Feature DataFrame used for permutation scoring.
        target: Binary target labels.
        top_k: Maximum number of ranked features to return.
        random_state: Seed for reproducible feature permutations.

    Returns:
        List of feature importance records sorted descending by mean recall decrease.
    """
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    result = permutation_importance(
        estimator,
        features,
        target,
        scoring="recall",
        n_repeats=5,
        random_state=random_state,
        n_jobs=1,
    )
    records = [
        {
            "feature": str(col),
            "importance_mean": round(float(mean), 6),
            "importance_std": round(float(std), 6),
        }
        for col, mean, std in zip(
            features.columns,
            result.importances_mean,
            result.importances_std,
            strict=True,
        )
    ]
    records.sort(key=lambda item: float(item["importance_mean"]), reverse=True)
    logger.debug("Computed permutation importance across %d features.", len(records))
    return records[:top_k]
