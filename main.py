"""CLI and ASGI entrypoint for FinanKu credit card late-payment detector."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence

import uvicorn

from src.latepay_detector.config import Settings, configure_logging, load_settings
from src.latepay_detector.features import HorizonType
from src.latepay_detector.pipeline import (
    ModelType,
    generate_synthetic_finanku_data,
    load_datasets,
    run_benchmark,
)

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser supporting health, demo, and API serving."""
    parser = argparse.ArgumentParser(
        description="FinanKu Credit Card Late-Payment Detector CLI & API Runner",
    )
    parser.add_argument("--health", action="store_true", help="Print service & dataset health JSON and exit")
    parser.add_argument("--demo", action="store_true", help="Run Experiment 1 & 2 leak-free evaluation demo")
    parser.add_argument("--serve", action="store_true", help="Launch FastAPI uvicorn inference server")
    parser.add_argument("--synthetic", action="store_true", help="Force deterministic synthetic demo dataset")
    parser.add_argument(
        "--horizon",
        choices=["12m", "6m", "both"],
        default="both",
        help="Observation horizon for --demo (default: both)",
    )
    parser.add_argument(
        "--model",
        choices=["logreg", "rf", "xgb"],
        default="logreg",
        help="Classifier family for --demo (default: logreg)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Decision threshold override in (0, 1) (default from settings: 0.30)",
    )
    return parser


def _resolve_horizons(horizon_arg: str) -> tuple[HorizonType, ...]:
    """Expand CLI horizon flag into explicit horizon tuple."""
    if horizon_arg == "both":
        return ("12m", "6m")
    return ("12m",) if horizon_arg == "12m" else ("6m",)


def run_health_check(settings: Settings) -> int:
    """Emit structured health payload verifying config and local datasets."""
    train_exists = (settings.data_dir / settings.train_filename).exists()
    val_exists = (settings.data_dir / settings.val_filename).exists()
    payload = {
        "status": "ok",
        "service": "credit-card-latepay-detector",
        "version": "1.0.0",
        "app_env": settings.app_env,
        "default_horizon": settings.default_horizon,
        "default_model_type": settings.default_model_type,
        "default_threshold": settings.default_threshold,
        "datasets_present": {
            "train": train_exists,
            "validation": val_exists,
        },
    }
    logger.info("Health check OK (train=%s, val=%s).", train_exists, val_exists)
    sys.stdout.write(json.dumps(payload, indent=2) + "\n")
    return 0


def run_demo(
    settings: Settings,
    horizon_arg: str,
    model_type: ModelType,
    threshold: float,
    use_synthetic: bool,
) -> int:
    """Execute leak-free Experiment 1 & Experiment 2 benchmark and output summary."""
    if use_synthetic:
        train_df = generate_synthetic_finanku_data(360, settings.random_state, validation=False)
        val_df = generate_synthetic_finanku_data(240, settings.random_state + 1, validation=True)
    else:
        try:
            train_df, val_df = load_datasets(settings)
        except FileNotFoundError:
            logger.warning("Real CSV datasets missing; using deterministic synthetic fallback.")
            train_df = generate_synthetic_finanku_data(360, settings.random_state, validation=False)
            val_df = generate_synthetic_finanku_data(240, settings.random_state + 1, validation=True)

    results = run_benchmark(
        train_df=train_df,
        val_df=val_df,
        horizons=_resolve_horizons(horizon_arg),
        model_type=model_type,
        threshold=threshold,
        random_state=settings.random_state,
    )

    summary = {horizon: result.to_dict() for horizon, result in results.items()}
    for horizon, res in results.items():
        val_t = res.val_metrics_threshold
        val_d = res.val_metrics_default
        logger.info(
            "Horizon=%s | Model=%s | Val t=%.2f Recall=%.2f%% Precision=%.2f%% | Baseline t=0.50 Recall=%.2f%%",
            horizon,
            model_type,
            threshold,
            val_t.recall * 100.0,
            val_t.precision * 100.0,
            val_d.recall * 100.0,
        )
    sys.stdout.write(json.dumps(summary, indent=2) + "\n")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint parsing command flags and dispatching actions."""
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    settings = load_settings()
    configure_logging(settings.log_level)

    threshold = settings.default_threshold if args.threshold is None else args.threshold
    if not 0.0 < threshold < 1.0:
        logger.error("Invalid threshold %.3f; must be strictly inside (0, 1).", threshold)
        return 2

    if args.serve:
        logger.info("Starting uvicorn server at %s:%d.", settings.api_host, settings.api_port)
        uvicorn.run(
            "src.latepay_detector.api:app",
            host=settings.api_host,
            port=settings.api_port,
            reload=False,
        )
        return 0

    if args.demo:
        return run_demo(
            settings=settings,
            horizon_arg=args.horizon,
            model_type=args.model,
            threshold=threshold,
            use_synthetic=args.synthetic,
        )

    return run_health_check(settings)


if __name__ == "__main__":
    raise SystemExit(main())
