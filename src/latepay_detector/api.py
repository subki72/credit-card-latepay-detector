"""FastAPI inference and experiment benchmarking microservice."""

from __future__ import annotations

import logging
import threading
from typing import Any, Literal

import pandas as pd
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.latepay_detector.config import Settings, configure_logging, load_settings
from src.latepay_detector.features import CANONICAL_FEATURES, HorizonType, build_horizon_features
from src.latepay_detector.pipeline import (
    LatePayPipeline,
    ModelType,
    generate_synthetic_finanku_data,
    load_datasets,
    run_benchmark,
)

logger = logging.getLogger(__name__)

MAX_PREDICT_BATCH = 500


class HealthResponse(BaseModel):
    """Service liveness and readiness status schema."""

    status: Literal["ok"]
    service: str
    version: str
    model_loaded: bool
    default_horizon: HorizonType
    default_model_type: ModelType
    default_threshold: float


class CustomerFeatureInput(BaseModel):
    """Canonical credit card customer feature vector for delinquency scoring."""

    customer_id: int | None = Field(default=None, description="Optional customer identifier")
    age: float = Field(..., ge=18, le=105)
    avg_annual_income_month: float = Field(..., ge=0.0)
    mean_balance: float
    delta_balance: float
    active_months: float = Field(..., ge=0.0, le=12.0)
    diff_ph: float
    vintage_cr: float = Field(..., ge=0.0, le=12.0)
    branch_code: str = Field(..., min_length=1, max_length=20)
    city: str = Field(..., min_length=1, max_length=60)


class PredictRequest(BaseModel):
    """Batch delinquency scoring payload."""

    records: list[CustomerFeatureInput] = Field(..., min_length=1, max_length=MAX_PREDICT_BATCH)
    threshold: float | None = Field(default=None, gt=0.0, lt=1.0)
    horizon: HorizonType | None = Field(default=None)


class PredictionItem(BaseModel):
    """Individual customer delinquency risk score output."""

    customer_id: int | None
    delinquency_probability: float
    predicted_late_pay: int
    risk_tier: Literal["HIGH", "MEDIUM", "LOW"]


class PredictResponse(BaseModel):
    """Batch scoring response payload."""

    threshold_applied: float
    horizon: HorizonType
    model_type: ModelType
    predictions: list[PredictionItem]


class AnalyzeRequest(BaseModel):
    """Experiment comparison trigger payload."""

    horizons: list[HorizonType] = Field(default_factory=lambda: ["12m", "6m"], min_length=1)
    model_type: ModelType = Field(default="logreg")
    threshold: float = Field(default=0.30, gt=0.0, lt=1.0)
    use_synthetic: bool = Field(default=False)


class AnalyzeResponse(BaseModel):
    """Leak-free experiment benchmark response."""

    results: dict[str, Any]


class _ModelCache:
    """Thread-safe in-memory model cache avoiding redundant retraining on requests."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._lock = threading.RLock()
        self._models: dict[tuple[HorizonType, ModelType], LatePayPipeline] = {}

    def has_any_loaded(self) -> bool:
        """Return True if at least one trained model is cached in memory."""
        with self._lock:
            return len(self._models) > 0

    def get_or_train(
        self,
        horizon: HorizonType,
        model_type: ModelType,
    ) -> LatePayPipeline:
        """Fetch cached model or fit offline on local/synthetic data."""
        key = (horizon, model_type)
        with self._lock:
            if key in self._models:
                return self._models[key]

            artifact_path = self._settings.artifacts_dir / f"latepay_{horizon}_{model_type}.joblib"
            if artifact_path.exists():
                loaded = LatePayPipeline.load(artifact_path)
                self._models[key] = loaded
                return loaded

            try:
                train_df, _ = load_datasets(self._settings)
                window: Literal["train", "validation"] = "train"
            except FileNotFoundError:
                logger.warning("Local CSV datasets unavailable; training fallback synthetic model.")
                train_df = generate_synthetic_finanku_data(
                    n_rows=320,
                    random_state=self._settings.random_state,
                    validation=False,
                )
                window = "train"

            features, target = build_horizon_features(
                train_df,
                horizon=horizon,
                window=window,
                require_target=True,
            )
            assert target is not None
            model = LatePayPipeline(
                horizon=horizon,
                model_type=model_type,
                threshold=self._settings.default_threshold,
                random_state=self._settings.random_state,
            ).fit(features, target)
            self._models[key] = model
            return model


def _classify_risk_tier(prob: float, threshold: float) -> Literal["HIGH", "MEDIUM", "LOW"]:
    """Assign actionable collection risk tier from predicted probability."""
    if prob >= max(0.60, threshold + 0.25):
        return "HIGH"
    if prob >= threshold:
        return "MEDIUM"
    return "LOW"


def _records_to_dataframe(records: list[CustomerFeatureInput]) -> pd.DataFrame:
    """Convert validated Pydantic input records into canonical feature DataFrame."""
    rows = [
        {
            "Age": float(item.age),
            "Avg. Annual Income/Month": float(item.avg_annual_income_month),
            "Mean Balance": float(item.mean_balance),
            "Delta Balance": float(item.delta_balance),
            "Active Months": float(item.active_months),
            "Diff PH": float(item.diff_ph),
            "Vintage_CR": float(item.vintage_cr),
            "Branch Code": item.branch_code.strip(),
            "City": item.city.strip(),
        }
        for item in records
    ]
    return pd.DataFrame(rows, columns=CANONICAL_FEATURES)


def _load_benchmark_frames(active_settings: Settings, use_synthetic: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load real CSVs or fallback synthetic frames for /analyze."""
    if not use_synthetic:
        try:
            return load_datasets(active_settings)
        except FileNotFoundError:
            logger.warning("Dataset CSVs missing; falling back to synthetic benchmark data.")
    train_df = generate_synthetic_finanku_data(300, active_settings.random_state, validation=False)
    val_df = generate_synthetic_finanku_data(200, active_settings.random_state + 1, validation=True)
    return train_df, val_df


def _register_exception_handlers(app: FastAPI) -> None:
    """Attach validation and global fallback JSON error handlers."""

    @app.exception_handler(ValueError)
    async def value_error_handler(_request: Request, exc: ValueError) -> JSONResponse:
        logger.warning("Request validation error: %s", exc)
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.exception_handler(Exception)
    async def global_exception_handler(_request: Request, exc: Exception) -> JSONResponse:
        logger.error("Unhandled internal API exception: %s", exc, exc_info=True)
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})


def _register_routes(app: FastAPI, active_settings: Settings, cache: _ModelCache) -> None:
    """Register liveness, inference, and experiment benchmark endpoints."""

    @app.get("/health", response_model=HealthResponse, tags=["Ops"])
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            service="credit-card-latepay-detector",
            version="1.0.0",
            model_loaded=cache.has_any_loaded(),
            default_horizon=active_settings.default_horizon,
            default_model_type=active_settings.default_model_type,
            default_threshold=active_settings.default_threshold,
        )

    @app.post("/predict", response_model=PredictResponse, tags=["Inference"])
    def predict(payload: PredictRequest) -> PredictResponse:
        horizon = payload.horizon or active_settings.default_horizon
        threshold = payload.threshold if payload.threshold is not None else active_settings.default_threshold
        model = cache.get_or_train(horizon=horizon, model_type=active_settings.default_model_type)
        probs = model.predict_proba(_records_to_dataframe(payload.records))
        items = [
            PredictionItem(
                customer_id=record.customer_id,
                delinquency_probability=round(float(prob), 4),
                predicted_late_pay=int(prob >= threshold),
                risk_tier=_classify_risk_tier(float(prob), threshold),
            )
            for record, prob in zip(payload.records, probs, strict=True)
        ]
        return PredictResponse(
            threshold_applied=threshold,
            horizon=horizon,
            model_type=active_settings.default_model_type,
            predictions=items,
        )

    @app.post("/analyze", response_model=AnalyzeResponse, tags=["Benchmarking"])
    def analyze(payload: AnalyzeRequest) -> AnalyzeResponse:
        train_df, val_df = _load_benchmark_frames(active_settings, payload.use_synthetic)
        results = run_benchmark(
            train_df=train_df,
            val_df=val_df,
            horizons=tuple(payload.horizons),
            model_type=payload.model_type,
            threshold=payload.threshold,
            random_state=active_settings.random_state,
        )
        return AnalyzeResponse(results={h: res.to_dict() for h, res in results.items()})


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build configured FastAPI application with CORSMiddleware and global error handlers."""
    active_settings = settings or load_settings()
    configure_logging(active_settings.log_level)
    cache = _ModelCache(active_settings)

    app = FastAPI(
        title="FinanKu Credit Card Late-Payment Detector API",
        version="1.0.0",
        description="Recall-focused credit card delinquency inference & leak-free benchmarking API.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=active_settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    _register_exception_handlers(app)
    _register_routes(app, active_settings, cache)
    return app


app = create_app()
