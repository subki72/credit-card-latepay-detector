"""Leak-free ML training, SMOTE resampling, persistence, and benchmarking pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.latepay_detector.config import Settings
from src.latepay_detector.evaluation import (
    EvaluationMetrics,
    compute_feature_importance,
    evaluate_predictions,
    predict_with_threshold,
)
from src.latepay_detector.features import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    HorizonType,
    build_horizon_features,
)

logger = logging.getLogger(__name__)

ModelType = Literal["logreg", "rf", "xgb"]


@dataclass(frozen=True)
class ExperimentResult:
    """Structured output from a single leak-free train/test/OOT-validation experiment."""

    horizon: HorizonType
    model_type: ModelType
    threshold: float
    train_rows: int
    test_rows: int
    val_rows: int
    test_metrics_threshold: EvaluationMetrics
    test_metrics_default: EvaluationMetrics
    val_metrics_threshold: EvaluationMetrics
    val_metrics_default: EvaluationMetrics
    top_features: list[dict[str, float | str]]

    def to_dict(self) -> dict[str, Any]:
        """Convert experiment summary to JSON-serializable dictionary."""
        return {
            "horizon": self.horizon,
            "model_type": self.model_type,
            "threshold": self.threshold,
            "train_rows": self.train_rows,
            "test_rows": self.test_rows,
            "val_rows": self.val_rows,
            "test_metrics_threshold": self.test_metrics_threshold.to_dict(),
            "test_metrics_default": self.test_metrics_default.to_dict(),
            "val_metrics_threshold": self.val_metrics_threshold.to_dict(),
            "val_metrics_default": self.val_metrics_default.to_dict(),
            "top_features": self.top_features,
        }


def resolve_exact_case_path(directory: Path, filename: str) -> Path:
    """Resolve a dataset file verifying exact case-sensitive filename match.

    Prevents Windows NTFS case-insensitivity from masking Linux FileNotFoundError
    bugs (e.g., ``Finanku`` vs ``FinanKu``).

    Args:
        directory: Parent directory containing dataset files.
        filename: Expected exact filename.

    Returns:
        Resolved ``Path`` if exact match exists.

    Raises:
        FileNotFoundError: If directory or exact-case filename does not exist.
    """
    if not directory.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {directory}")

    available = {entry.name: entry for entry in directory.iterdir()}
    if filename not in available:
        raise FileNotFoundError(
            f"Exact filename '{filename}' not found in {directory}. Available CSVs: "
            f"{sorted(name for name in available if name.endswith('.csv'))}"
        )
    return available[filename]


def load_datasets(settings: Settings) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load training and out-of-time validation CSV files safely.

    Args:
        settings: Application configuration containing dataset directory and filenames.

    Returns:
        Tuple ``(train_df, val_df)``.
    """
    train_path = resolve_exact_case_path(settings.data_dir, settings.train_filename)
    val_path = resolve_exact_case_path(settings.data_dir, settings.val_filename)
    logger.info("Loading datasets from %s and %s.", train_path.name, val_path.name)
    return pd.read_csv(train_path), pd.read_csv(val_path)


def generate_synthetic_finanku_data(
    n_rows: int = 300,
    random_state: int = 42,
    validation: bool = False,
) -> pd.DataFrame:
    """Generate deterministic offline FinanKu-compatible customer records for tests/demo.

    Args:
        n_rows: Number of synthetic customer rows to create.
        random_state: Deterministic NumPy RNG seed.
        validation: If True, emit ``Q2..Q5`` schema matching ``FinanKu Data Validasi.csv``.

    Returns:
        Synthetic DataFrame with realistic quarterly financial signals.
    """
    if n_rows < 20:
        raise ValueError("n_rows must be at least 20 for stratified synthetic sampling.")

    rng = np.random.default_rng(random_state)
    quarters = ("Q2", "Q3", "Q4", "Q5") if validation else ("Q1", "Q2", "Q3", "Q4")

    data: dict[str, Any] = {
        "Customer ID": np.arange(15565000, 15565000 + n_rows),
        "Branch Code": rng.choice(["1001", "1005", "1011", "1021"], size=n_rows),
        "City": rng.choice(["Jakarta", "Bandung", "Surabaya"], size=n_rows),
        "Age": rng.integers(21, 72, size=n_rows),
        "Avg. Annual Income/Month": rng.integers(5_000_000, 45_000_000, size=n_rows),
    }

    risk_signal = np.zeros(n_rows, dtype=float)
    for idx, q in enumerate(quarters):
        bal = rng.uniform(0.0, 25_000_000.0, size=n_rows)
        act = rng.binomial(1, 0.58, size=n_rows)
        prods = rng.integers(1, 4, size=n_rows)
        crcard = rng.binomial(1, 0.72, size=n_rows)
        data[f"Balance {q}"] = np.round(bal, 2)
        data[f"NumOfProducts {q}"] = prods
        data[f"HasCrCard {q}"] = crcard
        data[f"ActiveMember {q}"] = act
        if idx >= 2:
            risk_signal += (bal < 6_000_000.0).astype(float) * 0.9 - act * 0.7

    logits = -0.8 + risk_signal + (data["Age"] > 50).astype(float) * 0.4
    probs = 1.0 / (1.0 + np.exp(-logits))
    data["Unpaid Tagging"] = (rng.uniform(0.0, 1.0, size=n_rows) < probs).astype(int)
    return pd.DataFrame(data)


def build_classifier(model_type: ModelType = "logreg", random_state: int = 42) -> Any:
    """Instantiate configured sklearn-compatible binary classifier.

    Args:
        model_type: One of ``"logreg"``, ``"rf"``, or ``"xgb"``.
        random_state: Random seed for reproducible training.

    Returns:
        Unfitted estimator instance.

    Raises:
        ValueError: If ``model_type`` is unsupported.
    """
    if model_type == "logreg":
        return LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0, random_state=random_state)
    if model_type == "rf":
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            max_features="sqrt",
            random_state=random_state,
            n_jobs=1,
        )
    if model_type == "xgb":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=75,
            max_depth=5,
            learning_rate=0.1,
            random_state=random_state,
            eval_metric="logloss",
            n_jobs=1,
        )
    raise ValueError(f"Unsupported model_type '{model_type}'. Choose 'logreg', 'rf', or 'xgb'.")


class LatePayPipeline:
    """Leak-free scikit-learn + imbalanced-learn SMOTE pipeline wrapper."""

    def __init__(
        self,
        horizon: HorizonType = "6m",
        model_type: ModelType = "logreg",
        threshold: float = 0.30,
        random_state: int = 42,
    ) -> None:
        if not 0.0 < threshold < 1.0:
            raise ValueError(f"Threshold must be strictly between 0 and 1, got {threshold}.")
        self.horizon: HorizonType = horizon
        self.model_type: ModelType = model_type
        self.threshold: float = threshold
        self.random_state: int = random_state
        self._pipeline: ImbPipeline | None = None

    @property
    def is_fitted(self) -> bool:
        """Return whether the underlying ML pipeline has been trained."""
        return self._pipeline is not None

    @property
    def estimator(self) -> ImbPipeline:
        """Return fitted underlying imbalanced-learn pipeline."""
        if self._pipeline is None:
            raise RuntimeError("LatePayPipeline has not been fitted.")
        return self._pipeline

    def _build_pipeline(self, minority_count: int) -> ImbPipeline:
        """Assemble preprocessor, adaptive SMOTE sampler, and classifier."""
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), NUMERIC_FEATURES),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
            ],
            remainder="drop",
        )
        k_neighbors = max(1, min(5, minority_count - 1))
        smote = SMOTE(random_state=self.random_state, k_neighbors=k_neighbors)
        classifier = build_classifier(self.model_type, random_state=self.random_state)
        return ImbPipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("smote", smote),
                ("classifier", classifier),
            ]
        )

    def fit(self, features: pd.DataFrame, target: pd.Series) -> LatePayPipeline:
        """Fit preprocessing, SMOTE resampling, and classifier strictly on training data.

        Args:
            features: Canonical training features.
            target: Binary training labels (0/1).

        Returns:
            Self after fitting.

        Raises:
            ValueError: If fewer than 2 minority samples exist or classes are unary.
        """
        counts = target.value_counts()
        if len(counts) < 2:
            raise ValueError("Training target must contain both classes (0 and 1).")
        minority_count = int(counts.min())
        if minority_count < 2:
            raise ValueError(f"SMOTE requires at least 2 minority-class samples, found {minority_count}.")

        self._pipeline = self._build_pipeline(minority_count=minority_count)
        self._pipeline.fit(features, target)
        logger.info(
            "Trained LatePayPipeline (horizon=%s, model=%s, rows=%d, threshold=%.2f).",
            self.horizon,
            self.model_type,
            len(features),
            self.threshold,
        )
        return self

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Predict delinquency probabilities using training-fitted transformers only."""
        if self._pipeline is None:
            raise RuntimeError("LatePayPipeline must be fitted before calling predict_proba.")
        return np.asarray(self._pipeline.predict_proba(features)[:, 1], dtype=float)

    def predict(self, features: pd.DataFrame, threshold: float | None = None) -> np.ndarray:
        """Predict binary delinquency risk flags at configured threshold."""
        active_threshold = self.threshold if threshold is None else threshold
        probs = self.predict_proba(features)
        return predict_with_threshold(probs, threshold=active_threshold)

    def save(self, artifact_path: Path) -> Path:
        """Persist trained pipeline bundle with joblib."""
        if self._pipeline is None:
            raise RuntimeError("Cannot save an unfitted LatePayPipeline.")
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, artifact_path)
        logger.info("Saved model artifact to %s.", artifact_path)
        return artifact_path

    @classmethod
    def load(cls, artifact_path: Path) -> LatePayPipeline:
        """Load persisted ``LatePayPipeline`` bundle from disk."""
        if not artifact_path.exists():
            raise FileNotFoundError(f"Model artifact not found: {artifact_path}")
        loaded = joblib.load(artifact_path)
        if not isinstance(loaded, cls):
            raise TypeError(f"Expected LatePayPipeline instance, got {type(loaded).__name__}.")
        return loaded


def _evaluate_split(
    model: LatePayPipeline,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    train_rows: int,
    random_state: int,
) -> ExperimentResult:
    """Compute thresholded/baseline metrics and permutation importance for an experiment."""
    test_probs = model.predict_proba(x_test)
    val_probs = model.predict_proba(x_val)
    sample_x = x_val.iloc[: min(len(x_val), 500)]
    sample_y = y_val.iloc[: min(len(y_val), 500)]
    top_features = compute_feature_importance(model.estimator, sample_x, sample_y, top_k=7, random_state=random_state)
    return ExperimentResult(
        horizon=model.horizon,
        model_type=model.model_type,
        threshold=model.threshold,
        train_rows=train_rows,
        test_rows=len(x_test),
        val_rows=len(x_val),
        test_metrics_threshold=evaluate_predictions(y_test, test_probs, threshold=model.threshold),
        test_metrics_default=evaluate_predictions(y_test, test_probs, threshold=0.50),
        val_metrics_threshold=evaluate_predictions(y_val, val_probs, threshold=model.threshold),
        val_metrics_default=evaluate_predictions(y_val, val_probs, threshold=0.50),
        top_features=top_features,
    )


def run_experiment(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    horizon: HorizonType = "6m",
    model_type: ModelType = "logreg",
    threshold: float = 0.30,
    test_size: float = 0.30,
    random_state: int = 42,
) -> tuple[ExperimentResult, LatePayPipeline]:
    """Execute one leak-free train/test split + out-of-time validation benchmark."""
    x_all, y_all = build_horizon_features(train_df, horizon=horizon, window="train", require_target=True)
    x_val, y_val = build_horizon_features(val_df, horizon=horizon, window="validation", require_target=True)
    assert y_all is not None and y_val is not None

    x_train, x_test, y_train, y_test = train_test_split(
        x_all, y_all, test_size=test_size, stratify=y_all, random_state=random_state
    )
    model = LatePayPipeline(
        horizon=horizon,
        model_type=model_type,
        threshold=threshold,
        random_state=random_state,
    ).fit(x_train, y_train)
    result = _evaluate_split(model, x_test, y_test, x_val, y_val, len(x_train), random_state)
    return result, model


def run_benchmark(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    horizons: tuple[HorizonType, ...] = ("12m", "6m"),
    model_type: ModelType = "logreg",
    threshold: float = 0.30,
    random_state: int = 42,
) -> dict[HorizonType, ExperimentResult]:
    """Run multi-horizon benchmark returning structured results per horizon."""
    results: dict[HorizonType, ExperimentResult] = {}
    for horizon in horizons:
        res, _ = run_experiment(
            train_df=train_df,
            val_df=val_df,
            horizon=horizon,
            model_type=model_type,
            threshold=threshold,
            random_state=random_state,
        )
        results[horizon] = res
    return results
