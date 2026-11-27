"""Modular credit card delinquency detection package."""

from src.latepay_detector.api import create_app
from src.latepay_detector.config import Settings, configure_logging, load_settings
from src.latepay_detector.evaluation import (
    EvaluationMetrics,
    compute_feature_importance,
    evaluate_predictions,
    predict_with_threshold,
)
from src.latepay_detector.features import (
    CANONICAL_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    build_horizon_features,
    compute_credit_vintage,
)
from src.latepay_detector.pipeline import (
    ExperimentResult,
    LatePayPipeline,
    generate_synthetic_finanku_data,
    load_datasets,
    run_benchmark,
    run_experiment,
)

__version__ = "1.0.0"

__all__ = [
    "CANONICAL_FEATURES",
    "CATEGORICAL_FEATURES",
    "EvaluationMetrics",
    "ExperimentResult",
    "LatePayPipeline",
    "NUMERIC_FEATURES",
    "Settings",
    "__version__",
    "build_horizon_features",
    "compute_credit_vintage",
    "compute_feature_importance",
    "configure_logging",
    "create_app",
    "evaluate_predictions",
    "generate_synthetic_finanku_data",
    "load_datasets",
    "load_settings",
    "predict_with_threshold",
    "run_benchmark",
    "run_experiment",
]
