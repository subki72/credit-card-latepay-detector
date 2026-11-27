"""Leak-free quarterly horizon feature engineering for credit card delinquency."""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TARGET_COLUMN = "Unpaid Tagging"
ID_COLUMN = "Customer ID"
BASE_COLUMNS = ["Age", "Avg. Annual Income/Month", "Branch Code", "City"]

NUMERIC_FEATURES = [
    "Age",
    "Avg. Annual Income/Month",
    "Mean Balance",
    "Delta Balance",
    "Active Months",
    "Diff PH",
    "Vintage_CR",
]
CATEGORICAL_FEATURES = ["Branch Code", "City"]
CANONICAL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

TRAIN_QUARTERS: tuple[str, str, str, str] = ("Q1", "Q2", "Q3", "Q4")
VALIDATION_QUARTERS: tuple[str, str, str, str] = ("Q2", "Q3", "Q4", "Q5")

HorizonType = Literal["12m", "6m"]
WindowType = Literal["auto", "train", "validation"]


def _quarter_columns(quarters: tuple[str, str, str, str]) -> list[str]:
    """Build required raw quarterly column names for a 4-quarter window."""
    prefixes = ("Balance", "NumOfProducts", "HasCrCard", "ActiveMember")
    return [f"{prefix} {quarter}" for quarter in quarters for prefix in prefixes]


def detect_quarter_window(
    df: pd.DataFrame,
    window: WindowType = "auto",
) -> tuple[str, str, str, str]:
    """Determine the 4-quarter column tuple (`Q1..Q4` vs `Q2..Q5`).

    Args:
        df: Raw customer DataFrame.
        window: Explicit window override or ``"auto"`` detection.

    Returns:
        Ordered 4-tuple of quarter suffixes.

    Raises:
        ValueError: If required quarterly columns are missing.
    """
    if window == "train":
        quarters = TRAIN_QUARTERS
    elif window == "validation":
        quarters = VALIDATION_QUARTERS
    else:
        has_q5 = all(col in df.columns for col in _quarter_columns(VALIDATION_QUARTERS))
        has_q1 = all(col in df.columns for col in _quarter_columns(TRAIN_QUARTERS))
        if has_q5:
            quarters = VALIDATION_QUARTERS
        elif has_q1:
            quarters = TRAIN_QUARTERS
        else:
            raise ValueError("Dataset is missing required Q1-Q4 or Q2-Q5 quarterly columns.")

    missing = [col for col in _quarter_columns(quarters) if col not in df.columns]
    if missing:
        raise ValueError(f"Missing quarterly columns for window '{window}': {missing}")
    return quarters


def validate_raw_dataset(
    df: pd.DataFrame,
    quarters: tuple[str, str, str, str],
    require_target: bool = True,
) -> None:
    """Validate input DataFrame structure before feature transformations.

    Args:
        df: Raw customer dataset.
        quarters: Expected quarter suffix tuple.
        require_target: Whether ``Unpaid Tagging`` must exist.

    Raises:
        ValueError: When dataset is empty, incomplete, or contains nulls.
    """
    if df.empty:
        raise ValueError("Input dataset must not be empty.")

    required = list(BASE_COLUMNS) + _quarter_columns(quarters)
    if require_target:
        required.append(TARGET_COLUMN)

    missing_cols = [col for col in required if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Input dataset missing required columns: {missing_cols}")

    null_counts = df[required].isnull().sum()
    null_cols = {col: int(cnt) for col, cnt in null_counts.items() if cnt > 0}
    if null_cols:
        raise ValueError(f"Input dataset contains null values in required columns: {null_cols}")


def compute_credit_vintage(
    df: pd.DataFrame,
    card_columns: tuple[str, str, str],
) -> pd.Series:
    """Compute credit card tenure proxy (`Vintage_CR`) in months (3, 6, 9, or 12).

    Evaluates earliest observed credit card ownership across the first three
    quarters of the active observation window to match training/validation alignment.

    Args:
        df: Customer DataFrame containing credit card indicator columns.
        card_columns: Three ordered quarter column names, earliest to latest.

    Returns:
        Integer ``pd.Series`` aligned to ``df.index``.
    """
    c_early, c_mid, c_recent = card_columns
    vintage = np.full(len(df), 3, dtype=np.int64)
    vintage = np.where(df[c_recent].to_numpy() == 1, 6, vintage)
    vintage = np.where(df[c_mid].to_numpy() == 1, 9, vintage)
    vintage = np.where(df[c_early].to_numpy() == 1, 12, vintage)
    return pd.Series(vintage, index=df.index, name="Vintage_CR")


def _normalize_branch_code(series: pd.Series) -> pd.Series:
    """Convert numeric or string branch codes to clean categorical strings."""
    return series.astype(str).str.replace(r"\.0$", "", regex=True).str.strip()


def _compute_horizon_aggregates(
    df: pd.DataFrame,
    quarters: tuple[str, str, str, str],
    horizon: HorizonType,
) -> dict[str, pd.Series]:
    """Compute horizon-specific balance, activity, and product holding metrics."""
    q1, q2, q3, q4 = quarters
    if horizon == "12m":
        mean_bal = (df[f"Balance {q1}"] + df[f"Balance {q2}"] + df[f"Balance {q3}"] + df[f"Balance {q4}"]) / 4.0
        delta_bal = df[f"Balance {q4}"] - df[f"Balance {q1}"]
        active_months = (
            df[f"ActiveMember {q1}"] + df[f"ActiveMember {q2}"] + df[f"ActiveMember {q3}"] + df[f"ActiveMember {q4}"]
        ) * 3.0
        diff_ph = (df[f"NumOfProducts {q4}"] - df[f"NumOfProducts {q1}"]).astype(float)
    else:
        mean_bal = (df[f"Balance {q3}"] + df[f"Balance {q4}"]) / 2.0
        delta_bal = df[f"Balance {q4}"] - df[f"Balance {q2}"]
        active_months = (df[f"ActiveMember {q3}"] + df[f"ActiveMember {q4}"]) * 3.0
        diff_ph = (df[f"NumOfProducts {q4}"] - df[f"NumOfProducts {q2}"]).astype(float)

    return {
        "Mean Balance": mean_bal,
        "Delta Balance": delta_bal,
        "Active Months": active_months,
        "Diff PH": diff_ph,
    }


def build_horizon_features(
    df: pd.DataFrame,
    horizon: HorizonType = "6m",
    window: WindowType = "auto",
    require_target: bool = True,
) -> tuple[pd.DataFrame, pd.Series | None]:
    """Engineer canonical leak-free features for the requested observation horizon.

    Args:
        df: Raw customer quarterly DataFrame.
        horizon: ``"12m"`` (4 quarters) or ``"6m"`` (last 2 quarters + 6m delta).
        window: ``"auto"``, ``"train"`` (`Q1-Q4`), or ``"validation"`` (`Q2-Q5`).
        require_target: Whether to extract ``Unpaid Tagging`` target vector.

    Returns:
        Tuple of ``(features_df, target_series_or_none)``.
    """
    if horizon not in {"12m", "6m"}:
        raise ValueError(f"Unsupported horizon '{horizon}'. Expected '12m' or '6m'.")

    quarters = detect_quarter_window(df, window=window)
    validate_raw_dataset(df, quarters=quarters, require_target=require_target)
    q1, q2, q3, _ = quarters

    aggregates = _compute_horizon_aggregates(df, quarters=quarters, horizon=horizon)
    features = pd.DataFrame(aggregates, index=df.index)
    features["Age"] = df["Age"].astype(float)
    features["Avg. Annual Income/Month"] = df["Avg. Annual Income/Month"].astype(float)
    features["Vintage_CR"] = compute_credit_vintage(
        df, (f"HasCrCard {q1}", f"HasCrCard {q2}", f"HasCrCard {q3}")
    ).astype(float)
    features["Branch Code"] = _normalize_branch_code(df["Branch Code"])
    features["City"] = df["City"].astype(str).str.strip()

    target = df[TARGET_COLUMN].astype(int).copy() if require_target else None
    logger.debug("Built %d rows of '%s' features from window %s.", len(features), horizon, quarters)
    return features[CANONICAL_FEATURES].copy(), target
