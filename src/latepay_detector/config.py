"""Externalized runtime configuration and structured logging setup."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field

HorizonType = Literal["12m", "6m"]
ModelType = Literal["logreg", "rf", "xgb"]


class Settings(BaseModel):
    """Typed application settings loaded from environment variables."""

    app_env: str = Field(default="development", min_length=1)
    log_level: str = Field(default="INFO", min_length=1)
    api_host: str = Field(default="0.0.0.0", min_length=1)
    api_port: int = Field(default=8000, ge=1, le=65535)
    data_dir: Path = Field(default=Path("."))
    train_filename: str = Field(default="FinanKu Data All.csv", min_length=1)
    val_filename: str = Field(default="FinanKu Data Validasi.csv", min_length=1)
    default_horizon: HorizonType = Field(default="6m")
    default_model_type: ModelType = Field(default="logreg")
    default_threshold: float = Field(default=0.30, gt=0.0, lt=1.0)
    random_state: int = Field(default=42, ge=0)
    artifacts_dir: Path = Field(default=Path("artifacts"))
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
    request_timeout_seconds: float = Field(default=15.0, gt=0.0)


def _parse_cors_origins(raw_value: str | None) -> list[str]:
    """Parse comma-separated CORS origins safely."""
    if not raw_value or not raw_value.strip():
        return ["*"]
    return [origin.strip() for origin in raw_value.split(",") if origin.strip()]


def load_settings(env_file: Path | None = None) -> Settings:
    """Load application settings from `.env` and process environment.

    Args:
        env_file: Optional path to a `.env` file.

    Returns:
        Validated ``Settings`` instance.
    """
    if env_file is not None and env_file.exists():
        load_dotenv(dotenv_path=env_file, override=False)
    else:
        load_dotenv(override=False)

    horizon_raw = os.getenv("LATEPAY_HORIZON", "6m").strip().lower()
    horizon: HorizonType = "12m" if horizon_raw == "12m" else "6m"

    model_map: dict[str, ModelType] = {"logreg": "logreg", "rf": "rf", "xgb": "xgb"}
    model_raw = os.getenv("LATEPAY_MODEL_TYPE", "logreg").strip().lower()
    model_type: ModelType = model_map.get(model_raw, "logreg")

    return Settings(
        app_env=os.getenv("APP_ENV", "development").strip(),
        log_level=os.getenv("LOG_LEVEL", "INFO").strip().upper(),
        api_host=os.getenv("API_HOST", "0.0.0.0").strip(),
        api_port=int(os.getenv("API_PORT", "8000")),
        data_dir=Path(os.getenv("LATEPAY_DATA_DIR", ".")),
        train_filename=os.getenv("LATEPAY_TRAIN_FILE", "FinanKu Data All.csv").strip(),
        val_filename=os.getenv("LATEPAY_VAL_FILE", "FinanKu Data Validasi.csv").strip(),
        default_horizon=horizon,
        default_model_type=model_type,
        default_threshold=float(os.getenv("LATEPAY_THRESHOLD", "0.30")),
        random_state=int(os.getenv("LATEPAY_RANDOM_STATE", "42")),
        artifacts_dir=Path(os.getenv("LATEPAY_ARTIFACTS_DIR", "artifacts")),
        cors_origins=_parse_cors_origins(os.getenv("CORS_ORIGINS")),
        request_timeout_seconds=float(os.getenv("REQUEST_TIMEOUT_SECONDS", "15.0")),
    )


def configure_logging(level: str = "INFO") -> logging.Logger:
    """Configure root structured leveled logging for the package.

    Args:
        level: Standard Python logging level name.

    Returns:
        Package logger instance.
    """
    resolved_level = getattr(logging, level.upper(), logging.INFO)
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(
            level=resolved_level,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )
    else:
        root_logger.setLevel(resolved_level)

    pkg_logger = logging.getLogger("latepay_detector")
    pkg_logger.setLevel(resolved_level)
    return pkg_logger
