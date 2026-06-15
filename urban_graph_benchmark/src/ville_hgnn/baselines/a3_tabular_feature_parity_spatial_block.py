#!/usr/bin/env python3
"""
A3 feature-parity tabular baselines for the Montréal 311 water/drainage benchmark.

A3 establishes the strongest non-graph tabular baseline before GraphSAGE/HGNN.

Research role
-------------
A3 is not "just another model." It is the non-graph ML control that future graph
models must beat. The feature-parity rule is:

    Any node/month feature later given to a graph model should first be tested
    in a flat, non-graph tabular model under the same splits and target.

This module separates:
  1. strict train/static forecasting features,
  2. rolling observed-history features,
  3. retrospective same-month reporting controls.

Target
------
    water_drainage_count

Primary split
-------------
    temporal

Important leakage policy
------------------------
Strict train/static feature sets may use static, SVI, calendar, and train-period
summaries only.

Rolling observed-history feature sets may additionally use past target/reporting
lags and rolling features. These features are safe for a one-step/rolling monthly
forecasting protocol, but they are not the same as forecasting the entire future
horizon from the end of the training period.

Retrospective feature sets may additionally use same-month
``total_311_count_non_water_drainage`` and are labeled
``retrospective_explanatory_v0``. They are not strict forecasting baselines.
"""

from __future__ import annotations

import argparse
import itertools
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover
    np = None  # type: ignore[assignment]
    _NUMPY_IMPORT_ERROR = exc
else:
    _NUMPY_IMPORT_ERROR = None

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover
    pd = None  # type: ignore[assignment]
    _PANDAS_IMPORT_ERROR = exc
else:
    _PANDAS_IMPORT_ERROR = None

try:  # Optional scikit-learn models.
    import sklearn
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
except Exception:  # pragma: no cover
    sklearn = None  # type: ignore[assignment]
    HistGradientBoostingRegressor = None  # type: ignore[assignment]
    RandomForestRegressor = None  # type: ignore[assignment]

from ville_hgnn.baselines.a1_svi_direct_ranking import (
    SviScoreSpec,
    build_svi_score_specs,
    validate_static_svi_scores,
)
from ville_hgnn.baselines.common import (
    BINARY_TARGET_COLUMN,
    DATASET_VERSION_DEFAULT,
    DEFAULT_CONFIG_PATH,
    TARGET_COLUMN,
    BaselineError,
    build_run_context,
    evaluate_prediction_frame,
    get_baseline_paths,
    load_benchmark_frame,
    split_column_for_scheme,
    split_counts,
    split_type_for_scheme,
)
from ville_hgnn.utils.io import config_hash, file_hash, to_jsonable, write_json, write_markdown


STAGE_SLUG = "A3_feature_parity_tabular_spatial_block"
MODEL_STAGE = "A3_feature_parity_tabular_spatial_block"


DEFAULT_SPLIT_SCHEME = "temporal"

ZONE_COL = "zone_id"
PERIOD_COL = "period_month"
MONTH_COL = "month"
TARGET_NAME = TARGET_COLUMN

OBSERVED_ALIAS_COL = "observed_water_drainage_count"
PREDICTED_COL = "predicted_water_drainage_count"
BINARY_SCORE_COL = "predicted_binary_probability"

RANDOM_SEED = 42
MIN_PREDICTION = 0.0

SAME_MONTH_NON_WATER_COL = "total_311_count_non_water_drainage"
TARGET_CONTAINING_REPORTING_COL = "total_311_count_all"

STRICT_TRAIN_STATIC_SETTING = "forecasting_v0"
ROLLING_HISTORY_SETTING = "rolling_observed_history_v0"
RETROSPECTIVE_SETTING = "retrospective_explanatory_v0"

DEFAULT_RIDGE_ALPHAS = (0.1, 1.0, 10.0)

STATIC_NUMERIC_SOURCE_COLUMNS = [
    "population_total_2021",
    "land_area_km2",
    "population_density",
    "population_density_per_km2",
    "tract_centroid_x",
    "tract_centroid_y",
    "tract_centroid_lon",
    "tract_centroid_lat",
]

TARGET_LAG_STEPS = (1, 2, 3, 6, 12)
REPORTING_LAG_STEPS = (1, 2, 3, 12)
REQUESTS_TOTAL_LAG_STEPS = (1, 3, 12)
ROLLING_WINDOWS = (3, 6, 12)

ALWAYS_FORBIDDEN_SUBSTRINGS = ["sovi"]

TARGET_DERIVED_SAME_MONTH_COLUMNS = {
    TARGET_COLUMN,
    BINARY_TARGET_COLUMN,
    "water_drainage_requests",
    "share_water_drainage_requests",
}

TARGET_DERIVED_AND_TARGET_CONTAINING = {
    *TARGET_DERIVED_SAME_MONTH_COLUMNS,
    TARGET_CONTAINING_REPORTING_COL,
}


class A3BaselineError(BaselineError):
    """Raised when A3 tabular feature-parity baseline generation fails."""


@dataclass(frozen=True)
class FeatureMeta:
    """Per-feature lineage metadata."""

    feature_name: str
    feature_family: str
    source_column: str
    transformation: str
    prediction_setting_allowed: str
    uses_target_history: bool = False
    uses_reporting_history: bool = False
    uses_same_month_information: bool = False
    uses_train_summary: bool = False
    lag_months: int | None = None
    rolling_window: int | None = None
    shift_applied: bool = False
    is_primary_svi: bool | None = None
    is_diagnostic_svi: bool | None = None
    is_strict_forecasting_safe: bool = True


@dataclass(frozen=True)
class FeatureSetSpec:
    """A3 feature-set specification."""

    name: str
    prediction_setting: str
    feature_columns: list[str]
    description: str
    is_primary_feature_set: bool = True


@dataclass(frozen=True)
class CandidateSpec:
    """Model candidate specification."""

    model_family: str
    hyperparameter_id: str
    parameters: dict[str, Any]


@dataclass(frozen=True)
class FittedTabularModel:
    """Fitted A3 tabular model."""

    model_name: str
    model_family: str
    hyperparameter_id: str
    feature_set_name: str
    prediction_setting: str
    feature_columns: list[str]
    feature_medians: dict[str, float]
    feature_means: dict[str, float]
    feature_stds: dict[str, float]
    parameters: dict[str, Any]
    coefficients: list[float] | None = None
    sklearn_model: Any | None = None
    standardize: bool = True
    target_transform: str = "none"
    inverse_transform: str = "identity_clipped_at_zero"

    def feature_importance_table(self) -> pd.DataFrame:
        """Return feature importance rows when available."""

        rows: list[dict[str, Any]] = []

        if self.coefficients is not None:
            for feature, coefficient in zip(["intercept", *self.feature_columns], self.coefficients):
                rows.append(
                    {
                        "model_name": self.model_name,
                        "model_family": self.model_family,
                        "hyperparameter_id": self.hyperparameter_id,
                        "feature_set_name": self.feature_set_name,
                        "prediction_setting": self.prediction_setting,
                        "feature": feature,
                        "importance_type": "signed_standardized_coefficient",
                        "importance": float(coefficient),
                        "absolute_importance": abs(float(coefficient)),
                        "interpretation_note": (
                            "Ridge coefficient on standardized features; predictive association, not causality."
                        ),
                    }
                )

        if self.sklearn_model is not None and hasattr(self.sklearn_model, "feature_importances_"):
            importances = getattr(self.sklearn_model, "feature_importances_")
            for feature, importance in zip(self.feature_columns, importances):
                rows.append(
                    {
                        "model_name": self.model_name,
                        "model_family": self.model_family,
                        "hyperparameter_id": self.hyperparameter_id,
                        "feature_set_name": self.feature_set_name,
                        "prediction_setting": self.prediction_setting,
                        "feature": feature,
                        "importance_type": "sklearn_impurity_feature_importance",
                        "importance": float(importance),
                        "absolute_importance": abs(float(importance)),
                        "interpretation_note": (
                            "Impurity-based tree importance; predictive association, not causality."
                        ),
                    }
                )

        return pd.DataFrame(rows)


def require_runtime_dependencies() -> None:
    """Fail clearly if required dependencies are unavailable."""

    if pd is None:
        raise A3BaselineError("pandas is required for A3 baselines.") from _PANDAS_IMPORT_ERROR
    if np is None:
        raise A3BaselineError("numpy is required for A3 baselines.") from _NUMPY_IMPORT_ERROR


def parse_float_list(value: str | Sequence[float]) -> list[float]:
    """Parse comma-separated float list."""

    if isinstance(value, str):
        parsed = [float(part.strip()) for part in value.split(",") if part.strip()]
    else:
        parsed = [float(x) for x in value]

    if not parsed:
        raise A3BaselineError("At least one ridge alpha must be provided.")

    if any(x < 0 for x in parsed):
        raise A3BaselineError(f"Ridge alphas must be nonnegative: {parsed}")

    return parsed


def normalize_frame_for_a3(frame: pd.DataFrame, split_scheme: str) -> pd.DataFrame:
    """Normalize columns required by A3."""

    split_col = split_column_for_scheme(split_scheme)
    required = [ZONE_COL, PERIOD_COL, TARGET_COLUMN, split_col]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise A3BaselineError(f"A3 input frame missing required columns: {missing}")

    out = frame.copy()
    out[ZONE_COL] = out[ZONE_COL].astype(str)

    parsed = pd.to_datetime(out[PERIOD_COL].astype(str), errors="coerce")
    if parsed.isna().any():
        bad = out.loc[parsed.isna(), PERIOD_COL].drop_duplicates().head(20).tolist()
        raise A3BaselineError(f"Could not parse period_month values: {bad}")

    out[PERIOD_COL] = parsed.dt.to_period("M").astype(str)
    period_index = pd.PeriodIndex(out[PERIOD_COL], freq="M")
    out["_period_ordinal"] = period_index.astype(int)

    if MONTH_COL not in out.columns:
        out[MONTH_COL] = parsed.dt.month.astype(int)
    else:
        month = pd.to_numeric(out[MONTH_COL], errors="coerce")
        out[MONTH_COL] = month.fillna(parsed.dt.month).astype(int)

    if "year" not in out.columns:
        out["year"] = parsed.dt.year.astype(int)
    else:
        year = pd.to_numeric(out["year"], errors="coerce")
        out["year"] = year.fillna(parsed.dt.year).astype(int)

    out[TARGET_COLUMN] = pd.to_numeric(out[TARGET_COLUMN], errors="coerce")
    if out[TARGET_COLUMN].isna().any():
        n_missing = int(out[TARGET_COLUMN].isna().sum())
        raise A3BaselineError(f"{TARGET_COLUMN} contains missing/non-numeric rows: {n_missing}")
    if (out[TARGET_COLUMN] < 0).any():
        n_negative = int((out[TARGET_COLUMN] < 0).sum())
        raise A3BaselineError(f"{TARGET_COLUMN} contains negative rows: {n_negative}")

    if BINARY_TARGET_COLUMN not in out.columns:
        out[BINARY_TARGET_COLUMN] = (out[TARGET_COLUMN] > 0).astype(int)
    else:
        out[BINARY_TARGET_COLUMN] = pd.to_numeric(
            out[BINARY_TARGET_COLUMN],
            errors="coerce",
        ).fillna(0).astype(int)

    return out.sort_values([ZONE_COL, "_period_ordinal"]).reset_index(drop=True)


def train_frame(frame: pd.DataFrame, split_scheme: str) -> pd.DataFrame:
    """Return train rows."""

    split_col = split_column_for_scheme(split_scheme)
    train = frame[frame[split_col].astype(str) == "train"].copy()
    if train.empty:
        raise A3BaselineError(f"No train rows found for split_scheme={split_scheme!r}.")
    return train


def safe_numeric(series: pd.Series) -> pd.Series:
    """Convert series to numeric float."""

    return pd.to_numeric(series, errors="coerce").astype(float)


def log1p_nonnegative(series: pd.Series) -> pd.Series:
    """Convert nonnegative series to log1p, clipping invalid values to zero."""

    return np.log1p(safe_numeric(series).fillna(0).clip(lower=0))


def append_feature(
    lineage: list[FeatureMeta],
    *,
    feature_name: str,
    feature_family: str,
    source_column: str,
    transformation: str,
    prediction_setting_allowed: str,
    uses_target_history: bool = False,
    uses_reporting_history: bool = False,
    uses_same_month_information: bool = False,
    uses_train_summary: bool = False,
    lag_months: int | None = None,
    rolling_window: int | None = None,
    shift_applied: bool = False,
    is_primary_svi: bool | None = None,
    is_diagnostic_svi: bool | None = None,
    is_strict_forecasting_safe: bool = True,
) -> None:
    """Append feature lineage metadata."""

    lineage.append(
        FeatureMeta(
            feature_name=feature_name,
            feature_family=feature_family,
            source_column=source_column,
            transformation=transformation,
            prediction_setting_allowed=prediction_setting_allowed,
            uses_target_history=uses_target_history,
            uses_reporting_history=uses_reporting_history,
            uses_same_month_information=uses_same_month_information,
            uses_train_summary=uses_train_summary,
            lag_months=lag_months,
            rolling_window=rolling_window,
            shift_applied=shift_applied,
            is_primary_svi=is_primary_svi,
            is_diagnostic_svi=is_diagnostic_svi,
            is_strict_forecasting_safe=is_strict_forecasting_safe,
        )
    )


def add_calendar_features(frame: pd.DataFrame, lineage: list[FeatureMeta]) -> tuple[pd.DataFrame, list[str]]:
    """Add calendar features known at prediction time."""

    out = frame.copy()
    added: list[str] = []

    month = pd.to_numeric(out[MONTH_COL], errors="coerce").fillna(1).astype(int)
    for month_value in range(2, 13):
        col = f"calendar__month_is_{month_value:02d}"
        out[col] = (month == month_value).astype(float)
        added.append(col)
        append_feature(
            lineage,
            feature_name=col,
            feature_family="calendar",
            source_column=MONTH_COL,
            transformation=f"indicator_month_{month_value:02d}",
            prediction_setting_allowed=STRICT_TRAIN_STATIC_SETTING,
        )

    out["calendar__month_sin"] = np.sin(2.0 * np.pi * month / 12.0)
    out["calendar__month_cos"] = np.cos(2.0 * np.pi * month / 12.0)
    for col, transform in [
        ("calendar__month_sin", "sin_2pi_month_over_12"),
        ("calendar__month_cos", "cos_2pi_month_over_12"),
    ]:
        added.append(col)
        append_feature(
            lineage,
            feature_name=col,
            feature_family="calendar",
            source_column=MONTH_COL,
            transformation=transform,
            prediction_setting_allowed=STRICT_TRAIN_STATIC_SETTING,
        )

    min_period = int(out["_period_ordinal"].min())
    out["calendar__period_index_since_start"] = (out["_period_ordinal"] - min_period).astype(float)
    added.append("calendar__period_index_since_start")
    append_feature(
        lineage,
        feature_name="calendar__period_index_since_start",
        feature_family="calendar",
        source_column=PERIOD_COL,
        transformation="period_ordinal_minus_dataset_min_period",
        prediction_setting_allowed=STRICT_TRAIN_STATIC_SETTING,
    )

    return out, added


def add_static_features(frame: pd.DataFrame, lineage: list[FeatureMeta]) -> tuple[pd.DataFrame, list[str]]:
    """Add transformed static tract and spatial-coordinate features."""

    out = frame.copy()
    added: list[str] = []

    for col in STATIC_NUMERIC_SOURCE_COLUMNS:
        if col not in out.columns:
            continue

        if col in {
            "population_total_2021",
            "land_area_km2",
            "population_density",
            "population_density_per_km2",
        }:
            feature_col = f"static__log1p_{col}"
            out[feature_col] = log1p_nonnegative(out[col])
            transformation = "log1p_nonnegative"
            family = "static"
        else:
            feature_col = f"static_spatial__{col}"
            out[feature_col] = safe_numeric(out[col])
            transformation = "numeric_coordinate"
            family = "static_spatial"

        added.append(feature_col)
        append_feature(
            lineage,
            feature_name=feature_col,
            feature_family=family,
            source_column=col,
            transformation=transformation,
            prediction_setting_allowed=STRICT_TRAIN_STATIC_SETTING,
        )

    return out, added


def add_svi_features(
    frame: pd.DataFrame,
    svi_specs: Sequence[SviScoreSpec],
    lineage: list[FeatureMeta],
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Add primary and diagnostic oriented SVI feature columns."""

    out = frame.copy()
    primary_cols: list[str] = []
    diagnostic_cols: list[str] = []

    for spec in svi_specs:
        score_role = str(getattr(spec, "score_role", ""))
        is_primary = score_role == "primary_continuous_svi_score_candidate"
        family = "svi_primary" if is_primary else "svi_diagnostic"
        feature_col = f"{family}__{spec.source_column}"
        out[feature_col] = safe_numeric(out[spec.score_column])

        if is_primary:
            primary_cols.append(feature_col)
        else:
            diagnostic_cols.append(feature_col)

        append_feature(
            lineage,
            feature_name=feature_col,
            feature_family=family,
            source_column=spec.source_column,
            transformation=f"oriented_score_from_{spec.score_column}",
            prediction_setting_allowed=STRICT_TRAIN_STATIC_SETTING,
            is_primary_svi=is_primary,
            is_diagnostic_svi=not is_primary,
        )

    return out, primary_cols, diagnostic_cols


def _assign_series_by_original_index(
    target: pd.DataFrame,
    sorted_frame: pd.DataFrame,
    columns: Sequence[str],
) -> pd.DataFrame:
    """Assign newly generated columns from sorted frame back to original row order."""

    out = target.copy()
    for col in columns:
        out.loc[sorted_frame.index, col] = sorted_frame[col]
    return out


def add_target_history_features(
    frame: pd.DataFrame,
    split_scheme: str,
    lineage: list[FeatureMeta],
) -> tuple[pd.DataFrame, list[str]]:
    """Add target-history features safe under a rolling observed-history protocol."""

    out = frame.copy()
    sorted_out = out.sort_values([ZONE_COL, "_period_ordinal"]).copy()
    grouped = sorted_out.groupby(ZONE_COL, sort=False)
    added: list[str] = []

    for lag in TARGET_LAG_STEPS:
        col = f"target_history__water_drainage_count_lag_{lag}"
        sorted_out[col] = grouped[TARGET_COLUMN].shift(lag)
        added.append(col)
        append_feature(
            lineage,
            feature_name=col,
            feature_family="target_history",
            source_column=TARGET_COLUMN,
            transformation="grouped_lag",
            prediction_setting_allowed=ROLLING_HISTORY_SETTING,
            uses_target_history=True,
            lag_months=lag,
            shift_applied=True,
            is_strict_forecasting_safe=True,
        )

    shifted_target = grouped[TARGET_COLUMN].shift(1)
    for window in ROLLING_WINDOWS:
        mean_col = f"target_history__water_drainage_count_roll{window}_mean_shift1"
        sum_col = f"target_history__water_drainage_count_roll{window}_sum_shift1"

        sorted_out[mean_col] = (
            shifted_target
            .groupby(sorted_out[ZONE_COL], sort=False)
            .rolling(window=window, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        sorted_out[sum_col] = (
            shifted_target
            .groupby(sorted_out[ZONE_COL], sort=False)
            .rolling(window=window, min_periods=1)
            .sum()
            .reset_index(level=0, drop=True)
        )

        for col, transform in [
            (mean_col, "grouped_shift1_rolling_mean"),
            (sum_col, "grouped_shift1_rolling_sum"),
        ]:
            added.append(col)
            append_feature(
                lineage,
                feature_name=col,
                feature_family="target_history",
                source_column=TARGET_COLUMN,
                transformation=transform,
                prediction_setting_allowed=ROLLING_HISTORY_SETTING,
                uses_target_history=True,
                lag_months=1,
                rolling_window=window,
                shift_applied=True,
                is_strict_forecasting_safe=True,
            )

    expanding_col = "target_history__water_drainage_count_expanding_mean_shift1"
    sorted_out[expanding_col] = (
        shifted_target
        .groupby(sorted_out[ZONE_COL], sort=False)
        .expanding(min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    added.append(expanding_col)
    append_feature(
        lineage,
        feature_name=expanding_col,
        feature_family="target_history",
        source_column=TARGET_COLUMN,
        transformation="grouped_shift1_expanding_mean",
        prediction_setting_allowed=ROLLING_HISTORY_SETTING,
        uses_target_history=True,
        lag_months=1,
        shift_applied=True,
        is_strict_forecasting_safe=True,
    )

    out = _assign_series_by_original_index(out, sorted_out, added)

    # Train-period tract summaries are known after training and safe for temporal forecasting.
    split_col = split_column_for_scheme(split_scheme)
    train_rows = out[out[split_col].astype(str) == "train"].copy()
    train_rows[TARGET_COLUMN] = safe_numeric(train_rows[TARGET_COLUMN]).clip(lower=0)

    by_zone = train_rows.groupby(ZONE_COL)[TARGET_COLUMN]
    summary = pd.DataFrame(
        {
            "target_train_summary__mean": by_zone.mean(),
            "target_train_summary__median": by_zone.median(),
            "target_train_summary__p90": by_zone.quantile(0.90),
            "target_train_summary__positive_rate": by_zone.apply(lambda s: float((s > 0).mean())),
        }
    ).reset_index()

    out = out.merge(summary, on=ZONE_COL, how="left")
    train_summary_cols = [col for col in summary.columns if col != ZONE_COL]

    for col in train_summary_cols:
        added.append(col)
        append_feature(
            lineage,
            feature_name=col,
            feature_family="target_train_summary",
            source_column=TARGET_COLUMN,
            transformation="train_period_grouped_summary_by_zone",
            prediction_setting_allowed=STRICT_TRAIN_STATIC_SETTING,
            uses_target_history=True,
            uses_train_summary=True,
            is_strict_forecasting_safe=True,
        )

    return out, added


def add_lagged_reporting_features(
    frame: pd.DataFrame,
    lineage: list[FeatureMeta],
) -> tuple[pd.DataFrame, list[str]]:
    """Add lagged non-water reporting features for rolling observed-history forecasting."""

    out = frame.copy()
    if SAME_MONTH_NON_WATER_COL not in out.columns:
        return out, []

    sorted_out = out.sort_values([ZONE_COL, "_period_ordinal"]).copy()
    grouped = sorted_out.groupby(ZONE_COL, sort=False)[SAME_MONTH_NON_WATER_COL]
    added: list[str] = []

    for lag in REPORTING_LAG_STEPS:
        col = f"reporting_history__{SAME_MONTH_NON_WATER_COL}_lag_{lag}"
        sorted_out[col] = grouped.shift(lag)
        added.append(col)
        append_feature(
            lineage,
            feature_name=col,
            feature_family="lagged_reporting",
            source_column=SAME_MONTH_NON_WATER_COL,
            transformation="grouped_lag",
            prediction_setting_allowed=ROLLING_HISTORY_SETTING,
            uses_reporting_history=True,
            lag_months=lag,
            shift_applied=True,
            is_strict_forecasting_safe=True,
        )

    shifted = grouped.shift(1)
    for window in ROLLING_WINDOWS:
        col = f"reporting_history__{SAME_MONTH_NON_WATER_COL}_roll{window}_mean_shift1"
        sorted_out[col] = (
            shifted
            .groupby(sorted_out[ZONE_COL], sort=False)
            .rolling(window=window, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        added.append(col)
        append_feature(
            lineage,
            feature_name=col,
            feature_family="lagged_reporting",
            source_column=SAME_MONTH_NON_WATER_COL,
            transformation="grouped_shift1_rolling_mean",
            prediction_setting_allowed=ROLLING_HISTORY_SETTING,
            uses_reporting_history=True,
            lag_months=1,
            rolling_window=window,
            shift_applied=True,
            is_strict_forecasting_safe=True,
        )

    out = _assign_series_by_original_index(out, sorted_out, added)
    return out, added


def add_lagged_requests_total_features(
    frame: pd.DataFrame,
    lineage: list[FeatureMeta],
) -> tuple[pd.DataFrame, list[str]]:
    """Add lagged total 311 reporting features; same-month total is never used."""

    out = frame.copy()
    if "requests_total" not in out.columns:
        return out, []

    sorted_out = out.sort_values([ZONE_COL, "_period_ordinal"]).copy()
    grouped = sorted_out.groupby(ZONE_COL, sort=False)["requests_total"]
    added: list[str] = []

    for lag in REQUESTS_TOTAL_LAG_STEPS:
        col = f"requests_history__requests_total_lag_{lag}"
        sorted_out[col] = grouped.shift(lag)
        added.append(col)
        append_feature(
            lineage,
            feature_name=col,
            feature_family="lagged_requests_total",
            source_column="requests_total",
            transformation="grouped_lag",
            prediction_setting_allowed=ROLLING_HISTORY_SETTING,
            uses_reporting_history=True,
            lag_months=lag,
            shift_applied=True,
            is_strict_forecasting_safe=True,
        )

    shifted = grouped.shift(1)
    for window in ROLLING_WINDOWS:
        col = f"requests_history__requests_total_roll{window}_mean_shift1"
        sorted_out[col] = (
            shifted
            .groupby(sorted_out[ZONE_COL], sort=False)
            .rolling(window=window, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        added.append(col)
        append_feature(
            lineage,
            feature_name=col,
            feature_family="lagged_requests_total",
            source_column="requests_total",
            transformation="grouped_shift1_rolling_mean",
            prediction_setting_allowed=ROLLING_HISTORY_SETTING,
            uses_reporting_history=True,
            lag_months=1,
            rolling_window=window,
            shift_applied=True,
            is_strict_forecasting_safe=True,
        )

    out = _assign_series_by_original_index(out, sorted_out, added)
    return out, added


def add_retrospective_reporting_features(
    frame: pd.DataFrame,
    lineage: list[FeatureMeta],
) -> tuple[pd.DataFrame, list[str]]:
    """Add same-month reporting-control features for retrospective models only."""

    out = frame.copy()
    added: list[str] = []

    if SAME_MONTH_NON_WATER_COL in out.columns:
        col = f"reporting_retro__log1p_{SAME_MONTH_NON_WATER_COL}"
        out[col] = log1p_nonnegative(out[SAME_MONTH_NON_WATER_COL])
        added.append(col)
        append_feature(
            lineage,
            feature_name=col,
            feature_family="same_month_reporting_retrospective",
            source_column=SAME_MONTH_NON_WATER_COL,
            transformation="log1p_nonnegative_same_month",
            prediction_setting_allowed=RETROSPECTIVE_SETTING,
            uses_same_month_information=True,
            is_strict_forecasting_safe=False,
        )

    return out, added


def prepare_feature_frame(
    frame: pd.DataFrame,
    split_scheme: str,
    svi_specs: Sequence[SviScoreSpec],
) -> tuple[pd.DataFrame, dict[str, list[str]], pd.DataFrame]:
    """Add all reusable feature groups to the frame."""

    out = frame.copy()
    lineage: list[FeatureMeta] = []
    groups: dict[str, list[str]] = {}

    out, groups["calendar"] = add_calendar_features(out, lineage)
    out, groups["static"] = add_static_features(out, lineage)
    out, groups["svi_primary"], groups["svi_diagnostic"] = add_svi_features(out, svi_specs, lineage)
    out, groups["target_history"] = add_target_history_features(out, split_scheme, lineage)
    out, groups["lagged_reporting"] = add_lagged_reporting_features(out, lineage)
    out, groups["lagged_requests_total"] = add_lagged_requests_total_features(out, lineage)
    out, groups["reporting_retrospective"] = add_retrospective_reporting_features(out, lineage)

    lineage_df = pd.DataFrame([meta.__dict__ for meta in lineage])
    return out, groups, lineage_df


def unique_cols(*groups: Sequence[str]) -> list[str]:
    """Concatenate column groups while preserving order and uniqueness."""

    out: list[str] = []
    for group in groups:
        out.extend(list(group))
    return list(dict.fromkeys(out))


def build_feature_set_specs(
    feature_groups: Mapping[str, list[str]],
    *,
    include_diagnostic_svi_sets: bool = True,
) -> list[FeatureSetSpec]:
    """Build A3 feature sets with clean ablations."""

    calendar = feature_groups.get("calendar", [])
    static = feature_groups.get("static", [])
    svi_primary = feature_groups.get("svi_primary", [])
    svi_diagnostic = feature_groups.get("svi_diagnostic", [])
    target_history = feature_groups.get("target_history", [])
    lagged_reporting = feature_groups.get("lagged_reporting", [])
    lagged_requests_total = feature_groups.get("lagged_requests_total", [])
    retrospective = feature_groups.get("reporting_retrospective", [])

    specs: list[FeatureSetSpec] = [
        FeatureSetSpec(
            name="A3_static_svi_calendar_forecasting",
            prediction_setting=STRICT_TRAIN_STATIC_SETTING,
            feature_columns=unique_cols(svi_primary, static, calendar),
            description=(
                "Static primary SVI, static tract/spatial features, and calendar controls. No rolling target lags."
            ),
        ),
        FeatureSetSpec(
            name="A3_target_history_forecasting",
            prediction_setting=ROLLING_HISTORY_SETTING,
            feature_columns=unique_cols(target_history, calendar),
            description=(
                "Past target lags, shifted rolling target history, train-period tract target summaries, and calendar controls."
            ),
        ),
        FeatureSetSpec(
            name="A3_target_history_svi_static_forecasting",
            prediction_setting=ROLLING_HISTORY_SETTING,
            feature_columns=unique_cols(target_history, svi_primary, static, calendar),
            description="Target history plus primary SVI, static tract/spatial features, and calendar controls.",
        ),
        FeatureSetSpec(
            name="A3_lagged_reporting_forecasting",
            prediction_setting=ROLLING_HISTORY_SETTING,
            feature_columns=unique_cols(lagged_reporting, lagged_requests_total, calendar),
            description="Past non-water/total reporting lags and shifted rolling reporting history, without target history.",
        ),
        FeatureSetSpec(
            name="A3_target_history_lagged_reporting_forecasting",
            prediction_setting=ROLLING_HISTORY_SETTING,
            feature_columns=unique_cols(target_history, lagged_reporting, lagged_requests_total, calendar),
            description="Target history plus past non-water/total reporting history and calendar controls.",
        ),
        FeatureSetSpec(
            name="A3_all_forecasting",
            prediction_setting=ROLLING_HISTORY_SETTING,
            feature_columns=unique_cols(
                target_history, lagged_reporting, lagged_requests_total, svi_primary, static, calendar
            ),
            description=(
                "Main strict non-graph ML baseline: target history, lagged reporting history, primary SVI, "
                "static/spatial features, and calendar controls."
            ),
        ),
    ]

    if include_diagnostic_svi_sets and svi_diagnostic:
        specs.append(
            FeatureSetSpec(
                name="A3_all_forecasting_diagnostic_svi_expanded",
                prediction_setting=ROLLING_HISTORY_SETTING,
                feature_columns=unique_cols(
                    target_history,
                    lagged_reporting,
                    lagged_requests_total,
                    svi_primary,
                    svi_diagnostic,
                    static,
                    calendar,
                ),
                description="Diagnostic SVI-expanded version of A3_all_forecasting, including rank/class SVI encodings.",
                is_primary_feature_set=False,
            )
        )

    if retrospective:
        specs.append(
            FeatureSetSpec(
                name="A3_reporting_retrospective",
                prediction_setting=RETROSPECTIVE_SETTING,
                feature_columns=unique_cols(
                    target_history,
                    lagged_reporting,
                    lagged_requests_total,
                    svi_primary,
                    static,
                    calendar,
                    retrospective,
                ),
                description="A3_all_forecasting plus same-month non-water 311 reporting control. Retrospective only.",
            )
        )

        if include_diagnostic_svi_sets and svi_diagnostic:
            specs.append(
                FeatureSetSpec(
                    name="A3_reporting_retrospective_diagnostic_svi_expanded",
                    prediction_setting=RETROSPECTIVE_SETTING,
                    feature_columns=unique_cols(
                        target_history,
                        lagged_reporting,
                        lagged_requests_total,
                        svi_primary,
                        svi_diagnostic,
                        static,
                        calendar,
                        retrospective,
                    ),
                    description="Diagnostic SVI-expanded retrospective feature set with same-month non-water reporting control.",
                    is_primary_feature_set=False,
                )
            )

    cleaned: list[FeatureSetSpec] = []
    for spec in specs:
        feature_columns = [col for col in spec.feature_columns if col]
        if not feature_columns:
            continue
        cleaned.append(
            FeatureSetSpec(
                name=spec.name,
                prediction_setting=spec.prediction_setting,
                feature_columns=feature_columns,
                description=spec.description,
                is_primary_feature_set=spec.is_primary_feature_set,
            )
        )

    if not cleaned:
        raise A3BaselineError("No valid A3 feature sets could be constructed.")

    return cleaned


def assert_a3_feature_leakage_policy(feature_set: FeatureSetSpec, feature_lineage: pd.DataFrame) -> None:
    """Enforce A3 feature leakage rules using per-feature lineage."""

    if feature_lineage.empty:
        raise A3BaselineError("Feature lineage audit is empty; cannot verify leakage policy.")

    missing = [col for col in feature_set.feature_columns if col not in set(feature_lineage["feature_name"])]
    if missing:
        raise A3BaselineError(
            f"Feature set {feature_set.name} contains features missing from lineage audit: {missing}"
        )

    lineage = feature_lineage[feature_lineage["feature_name"].isin(feature_set.feature_columns)].copy()

    for col in [*feature_set.feature_columns, *lineage["source_column"].astype(str).tolist()]:
        lower = col.lower()
        if any(token in lower for token in ALWAYS_FORBIDDEN_SUBSTRINGS):
            raise A3BaselineError(f"Track A feature set contains forbidden SoVI column/source: {col}")

    same_month_target = lineage[
        (lineage["uses_same_month_information"].astype(bool))
        & (lineage["source_column"].isin(TARGET_DERIVED_AND_TARGET_CONTAINING))
    ]
    if not same_month_target.empty:
        raise A3BaselineError(
            f"Feature set {feature_set.name} contains same-month target-derived/containing features: "
            f"{same_month_target['feature_name'].tolist()}"
        )

    target_all = lineage[lineage["source_column"] == TARGET_CONTAINING_REPORTING_COL]
    if not target_all.empty:
        raise A3BaselineError(
            f"Feature set {feature_set.name} uses {TARGET_CONTAINING_REPORTING_COL}, which contains the target."
        )

    retrospective_features = lineage[lineage["uses_same_month_information"].astype(bool)]
    if feature_set.prediction_setting != RETROSPECTIVE_SETTING and not retrospective_features.empty:
        raise A3BaselineError(
            f"Non-retrospective feature set {feature_set.name} contains same-month features: "
            f"{retrospective_features['feature_name'].tolist()}"
        )

    if feature_set.prediction_setting == STRICT_TRAIN_STATIC_SETTING:
        history_features = lineage[
            lineage["uses_target_history"].astype(bool) | lineage["uses_reporting_history"].astype(bool)
        ]
        disallowed_history = history_features[~history_features["uses_train_summary"].astype(bool)]
        if not disallowed_history.empty:
            raise A3BaselineError(
                f"Strict train/static feature set {feature_set.name} contains rolling history features: "
                f"{disallowed_history['feature_name'].tolist()}"
            )

    if feature_set.prediction_setting == ROLLING_HISTORY_SETTING:
        rolling_target = lineage[
            lineage["uses_target_history"].astype(bool) & ~lineage["uses_train_summary"].astype(bool)
        ]
        unshifted = rolling_target[~rolling_target["shift_applied"].astype(bool)]
        if not unshifted.empty:
            raise A3BaselineError(
                f"Rolling-history feature set {feature_set.name} contains unshifted target-history features: "
                f"{unshifted['feature_name'].tolist()}"
            )

    unsafe_strict = lineage[
        (feature_set.prediction_setting != RETROSPECTIVE_SETTING)
        & (~lineage["is_strict_forecasting_safe"].astype(bool))
    ]
    if not unsafe_strict.empty:
        raise A3BaselineError(
            f"Feature set {feature_set.name} contains features not strict-safe: "
            f"{unsafe_strict['feature_name'].tolist()}"
        )


def validate_feature_sets(feature_sets: Sequence[FeatureSetSpec], feature_lineage: pd.DataFrame) -> None:
    """Validate all feature sets."""

    for feature_set in feature_sets:
        assert_a3_feature_leakage_policy(feature_set, feature_lineage)


def fit_imputer_scaler(
    train_features: pd.DataFrame,
    full_features: pd.DataFrame,
    *,
    standardize: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, float], dict[str, float], dict[str, float]]:
    """Fit train-only median imputation and optional standardization."""

    if train_features.empty or full_features.empty:
        raise A3BaselineError("Cannot fit model on empty feature matrix.")

    medians: dict[str, float] = {}
    means: dict[str, float] = {}
    stds: dict[str, float] = {}

    train_out = pd.DataFrame(index=train_features.index)
    full_out = pd.DataFrame(index=full_features.index)

    for col in train_features.columns:
        train_col = safe_numeric(train_features[col])
        full_col = safe_numeric(full_features[col])

        median = float(train_col.median(skipna=True)) if train_col.notna().any() else 0.0
        train_filled = train_col.fillna(median)
        full_filled = full_col.fillna(median)

        if standardize:
            mean = float(train_filled.mean())
            std = float(train_filled.std(ddof=0))
            if not math.isfinite(std) or std <= 1e-12:
                std = 1.0
            train_out[col] = (train_filled - mean) / std
            full_out[col] = (full_filled - mean) / std
        else:
            mean = 0.0
            std = 1.0
            train_out[col] = train_filled
            full_out[col] = full_filled

        medians[col] = median
        means[col] = mean
        stds[col] = std

    return train_out.to_numpy(dtype=float), full_out.to_numpy(dtype=float), medians, means, stds


def candidate_specs(
    *,
    ridge_alphas: Sequence[float],
    include_sklearn_models: bool,
    include_random_forest: bool,
    hgb_grid: str = "small",
) -> list[CandidateSpec]:
    """Return candidate model specifications."""

    specs: list[CandidateSpec] = []

    for alpha in ridge_alphas:
        specs.append(
            CandidateSpec(
                model_family="ridge_log_count",
                hyperparameter_id=f"alpha_{alpha:g}",
                parameters={
                    "ridge_alpha": float(alpha),
                    "target_transform": "log1p",
                    "inverse_transform": "expm1_clipped_at_zero",
                },
            )
        )

    if include_sklearn_models and HistGradientBoostingRegressor is not None:
        hgb_params = [
            {
                "loss": "poisson",
                "learning_rate": 0.05,
                "max_iter": 300,
                "max_leaf_nodes": 15,
                "min_samples_leaf": 20,
                "l2_regularization": 0.1,
            },
            {
                "loss": "poisson",
                "learning_rate": 0.05,
                "max_iter": 300,
                "max_leaf_nodes": 31,
                "min_samples_leaf": 20,
                "l2_regularization": 0.1,
            },
        ]

        if hgb_grid == "medium":
            hgb_params = []
            for max_leaf_nodes, min_samples_leaf, l2 in itertools.product([15, 31], [20, 50], [0.0, 0.1]):
                hgb_params.append(
                    {
                        "loss": "poisson",
                        "learning_rate": 0.05,
                        "max_iter": 300,
                        "max_leaf_nodes": max_leaf_nodes,
                        "min_samples_leaf": min_samples_leaf,
                        "l2_regularization": l2,
                    }
                )

        for i, params in enumerate(hgb_params, start=1):
            specs.append(
                CandidateSpec(
                    model_family="hist_gradient_boosting_poisson",
                    hyperparameter_id=f"hgb_poisson_{i:02d}",
                    parameters={**params, "random_state": RANDOM_SEED},
                )
            )

    if include_sklearn_models and include_random_forest and RandomForestRegressor is not None:
        specs.append(
            CandidateSpec(
                model_family="random_forest_log_count",
                hyperparameter_id="rf_log_count_conservative",
                parameters={
                    "n_estimators": 300,
                    "max_depth": 12,
                    "min_samples_leaf": 20,
                    "max_features": 0.7,
                    "random_state": RANDOM_SEED,
                    "n_jobs": -1,
                    "target_transform": "log1p",
                    "inverse_transform": "expm1_clipped_at_zero",
                },
            )
        )

    return specs


def fit_candidate_model(
    frame: pd.DataFrame,
    train: pd.DataFrame,
    feature_set: FeatureSetSpec,
    candidate: CandidateSpec,
) -> tuple[FittedTabularModel, pd.Series]:
    """Fit one candidate model and predict all rows."""

    train_features = frame.loc[train.index, feature_set.feature_columns]
    full_features = frame[feature_set.feature_columns]

    if candidate.model_family == "ridge_log_count":
        ridge_alpha = float(candidate.parameters["ridge_alpha"])
        X_train_raw, X_full_raw, medians, means, stds = fit_imputer_scaler(
            train_features,
            full_features,
            standardize=True,
        )
        X_train = np.column_stack([np.ones(len(X_train_raw)), X_train_raw])
        X_full = np.column_stack([np.ones(len(X_full_raw)), X_full_raw])

        y_train = np.log1p(safe_numeric(train[TARGET_COLUMN]).clip(lower=0).to_numpy(dtype=float))

        penalty = np.eye(X_train.shape[1]) * ridge_alpha
        penalty[0, 0] = 0.0

        xtx = X_train.T @ X_train
        xty = X_train.T @ y_train

        try:
            beta = np.linalg.solve(xtx + penalty, xty)
        except np.linalg.LinAlgError:
            beta = np.linalg.pinv(xtx + penalty) @ xty

        pred = np.expm1(X_full @ beta)
        pred = np.clip(pred, MIN_PREDICTION, None)

        fitted = FittedTabularModel(
            model_name=f"ridge_log_count__{feature_set.name}__{candidate.hyperparameter_id}",
            model_family=candidate.model_family,
            hyperparameter_id=candidate.hyperparameter_id,
            feature_set_name=feature_set.name,
            prediction_setting=feature_set.prediction_setting,
            feature_columns=list(feature_set.feature_columns),
            feature_medians=medians,
            feature_means=means,
            feature_stds=stds,
            parameters=dict(candidate.parameters),
            coefficients=[float(value) for value in beta],
            sklearn_model=None,
            standardize=True,
            target_transform="log1p",
            inverse_transform="expm1_clipped_at_zero",
        )

        return fitted, pd.Series(pred, index=frame.index, dtype=float)

    if candidate.model_family == "hist_gradient_boosting_poisson":
        if HistGradientBoostingRegressor is None:
            raise A3BaselineError("HistGradientBoostingRegressor is unavailable.")

        X_train, X_full, medians, means, stds = fit_imputer_scaler(
            train_features,
            full_features,
            standardize=False,
        )
        y_train = safe_numeric(train[TARGET_COLUMN]).clip(lower=0).to_numpy(dtype=float)

        params = {
            key: value
            for key, value in candidate.parameters.items()
            if key not in {"target_transform", "inverse_transform"}
        }
        estimator = HistGradientBoostingRegressor(**params)
        estimator.fit(X_train, y_train)

        pred = estimator.predict(X_full)
        pred = np.clip(pred, MIN_PREDICTION, None)

        fitted = FittedTabularModel(
            model_name=f"hist_gradient_boosting_poisson__{feature_set.name}__{candidate.hyperparameter_id}",
            model_family=candidate.model_family,
            hyperparameter_id=candidate.hyperparameter_id,
            feature_set_name=feature_set.name,
            prediction_setting=feature_set.prediction_setting,
            feature_columns=list(feature_set.feature_columns),
            feature_medians=medians,
            feature_means=means,
            feature_stds=stds,
            parameters=dict(candidate.parameters),
            coefficients=None,
            sklearn_model=estimator,
            standardize=False,
            target_transform="none_count_scale",
            inverse_transform="identity_clipped_at_zero",
        )

        return fitted, pd.Series(pred, index=frame.index, dtype=float)

    if candidate.model_family == "random_forest_log_count":
        if RandomForestRegressor is None:
            raise A3BaselineError("RandomForestRegressor is unavailable.")

        X_train, X_full, medians, means, stds = fit_imputer_scaler(
            train_features,
            full_features,
            standardize=False,
        )
        y_train = np.log1p(safe_numeric(train[TARGET_COLUMN]).clip(lower=0).to_numpy(dtype=float))

        params = {
            key: value
            for key, value in candidate.parameters.items()
            if key not in {"target_transform", "inverse_transform"}
        }
        estimator = RandomForestRegressor(**params)
        estimator.fit(X_train, y_train)

        pred = np.expm1(estimator.predict(X_full))
        pred = np.clip(pred, MIN_PREDICTION, None)

        fitted = FittedTabularModel(
            model_name=f"random_forest_log_count__{feature_set.name}__{candidate.hyperparameter_id}",
            model_family=candidate.model_family,
            hyperparameter_id=candidate.hyperparameter_id,
            feature_set_name=feature_set.name,
            prediction_setting=feature_set.prediction_setting,
            feature_columns=list(feature_set.feature_columns),
            feature_medians=medians,
            feature_means=means,
            feature_stds=stds,
            parameters=dict(candidate.parameters),
            coefficients=None,
            sklearn_model=estimator,
            standardize=False,
            target_transform="log1p",
            inverse_transform="expm1_clipped_at_zero",
        )

        return fitted, pd.Series(pred, index=frame.index, dtype=float)

    raise A3BaselineError(f"Unknown candidate model family: {candidate.model_family}")


def poisson_any_probability(mu: pd.Series) -> pd.Series:
    """Convert predicted count mean to P(Y > 0) under a Poisson assumption."""

    values = safe_numeric(mu).fillna(0).clip(lower=0)
    return 1.0 - np.exp(-values)


def prediction_frame_for_model(
    frame: pd.DataFrame,
    *,
    split_scheme: str,
    fitted: FittedTabularModel,
    predictions: pd.Series,
) -> pd.DataFrame:
    """Create standardized long prediction rows for one A3 model."""

    split_col = split_column_for_scheme(split_scheme)

    cols = [
        ZONE_COL,
        PERIOD_COL,
        split_col,
        TARGET_COLUMN,
        BINARY_TARGET_COLUMN,
        "year",
        MONTH_COL,
        "population_total_2021",
        "land_area_km2",
        "population_density",
        "population_density_per_km2",
        SAME_MONTH_NON_WATER_COL,
    ]
    cols = [col for col in cols if col in frame.columns]

    out = frame[cols].copy()
    pred = safe_numeric(predictions).fillna(0).clip(lower=0)
    pred.index = out.index

    out[OBSERVED_ALIAS_COL] = out[TARGET_COLUMN]
    out[PREDICTED_COL] = pred
    out["predicted_score"] = pred
    out[BINARY_SCORE_COL] = poisson_any_probability(pred)
    out["model_stage"] = MODEL_STAGE
    out["model_name"] = fitted.model_name
    out["model_family"] = fitted.model_family
    out["hyperparameter_id"] = fitted.hyperparameter_id
    out["feature_set_name"] = fitted.feature_set_name
    out["prediction_setting"] = fitted.prediction_setting
    out["target_transform"] = fitted.target_transform

    return out


def standardize_a3_metric_schema(metrics: pd.DataFrame) -> pd.DataFrame:
    """Ensure current A3 metric schema aliases exist."""

    if metrics.empty:
        return metrics

    out = metrics.copy()

    if "n_rows" not in out.columns and "n_eval" in out.columns:
        out["n_rows"] = out["n_eval"]

    if "metric_name" in out.columns:
        alias_source = out[out["metric_name"] == "ranking__top_10pct_overlap_precision"].copy()
        if not alias_source.empty:
            alias_source["metric_name"] = "ranking__top_10pct_overlap_rate"
            if "notes" in alias_source.columns:
                alias_source["notes"] = alias_source["notes"].fillna("").astype(str)
                alias_source["notes"] = (
                    alias_source["notes"]
                    + "; alias of top-10pct overlap precision because predicted and observed top sets have equal size"
                )
            out = pd.concat([out, alias_source], ignore_index=True)

    preferred = [
        "benchmark_id",
        "dataset_version",
        "split_name",
        "split_type",
        "prediction_setting",
        "model_stage",
        "model_name",
        "target_name",
        "target_type",
        "feature_set_name",
        "metric_name",
        "metric_value",
        "higher_is_better",
        "n_rows",
        "n_eval",
        "n_train",
        "n_validation",
        "n_test",
        "notes",
    ]
    ordered = [col for col in preferred if col in out.columns]
    rest = [col for col in out.columns if col not in ordered]
    return out[ordered + rest]


def evaluate_one_model(
    prediction_frame: pd.DataFrame,
    *,
    config: Mapping[str, Any],
    full_frame: pd.DataFrame,
    split_scheme: str,
    fitted: FittedTabularModel,
) -> pd.DataFrame:
    """Evaluate one A3 model using shared metrics conventions."""

    context = build_run_context(
        config=config,
        frame=full_frame,
        split_scheme=split_scheme,
        prediction_setting=fitted.prediction_setting,
        model_stage=MODEL_STAGE,
        model_name=fitted.model_name,
        target_name=TARGET_NAME,
        target_type="count",
        feature_set_name=fitted.feature_set_name,
        dataset_version=DATASET_VERSION_DEFAULT,
    )

    metrics = evaluate_prediction_frame(
        prediction_frame,
        context=context,
        split_scheme=split_scheme,
        observed_col=TARGET_COLUMN,
        predicted_col=PREDICTED_COL,
        binary_observed_col=BINARY_TARGET_COLUMN,
        binary_score_col=BINARY_SCORE_COL,
        ranking_score_col="predicted_score",
    )

    out = standardize_a3_metric_schema(metrics)
    out["model_family"] = fitted.model_family
    out["hyperparameter_id"] = fitted.hyperparameter_id
    return out


def write_long_prediction_partitions(
    predictions_long: pd.DataFrame,
    output_dir: Path,
    *,
    split_scheme: str,
) -> dict[str, str]:
    """Write validation/test prediction files for all A3 models in long format."""

    split_col = split_column_for_scheme(split_scheme)
    validation_path = output_dir / "predictions_validation.parquet"
    test_path = output_dir / "predictions_test.parquet"

    preferred_cols = [
        ZONE_COL,
        PERIOD_COL,
        split_col,
        OBSERVED_ALIAS_COL,
        TARGET_COLUMN,
        BINARY_TARGET_COLUMN,
        PREDICTED_COL,
        "predicted_score",
        BINARY_SCORE_COL,
        "model_stage",
        "model_name",
        "model_family",
        "hyperparameter_id",
        "feature_set_name",
        "prediction_setting",
        "target_transform",
        "year",
        MONTH_COL,
        "population_total_2021",
        "land_area_km2",
        "population_density",
        "population_density_per_km2",
        SAME_MONTH_NON_WATER_COL,
    ]
    cols = [col for col in preferred_cols if col in predictions_long.columns]

    validation = predictions_long[predictions_long[split_col].astype(str) == "validation"][cols].copy()
    test = predictions_long[predictions_long[split_col].astype(str) == "test"][cols].copy()

    validation.to_parquet(validation_path, index=False)
    test.to_parquet(test_path, index=False)

    return {
        "predictions_validation": str(validation_path),
        "predictions_test": str(test_path),
    }


def feature_set_audit_rows(
    feature_sets: Sequence[FeatureSetSpec],
    feature_lineage: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Create feature-set audit rows."""

    rows: list[dict[str, Any]] = []

    for spec in feature_sets:
        lineage = feature_lineage[feature_lineage["feature_name"].isin(spec.feature_columns)].copy()
        rows.append(
            {
                "feature_set_name": spec.name,
                "prediction_setting": spec.prediction_setting,
                "n_features": len(spec.feature_columns),
                "is_primary_feature_set": spec.is_primary_feature_set,
                "feature_families": ";".join(sorted(lineage["feature_family"].dropna().astype(str).unique())),
                "source_columns": ";".join(sorted(lineage["source_column"].dropna().astype(str).unique())),
                "feature_columns": ";".join(spec.feature_columns),
                "description": spec.description,
                "uses_target_history": bool(lineage["uses_target_history"].astype(bool).any()) if not lineage.empty else False,
                "uses_reporting_history": bool(lineage["uses_reporting_history"].astype(bool).any()) if not lineage.empty else False,
                "uses_same_month_information": bool(lineage["uses_same_month_information"].astype(bool).any()) if not lineage.empty else False,
                "uses_retrospective_reporting_control": bool(
                    lineage["source_column"].astype(str).eq(SAME_MONTH_NON_WATER_COL).any()
                    and lineage["uses_same_month_information"].astype(bool).any()
                ) if not lineage.empty else False,
                "leakage_policy_status": "passed",
            }
        )

    return rows


def _metric_value(metrics: pd.DataFrame, model_name: str, metric_name: str, split_name: str) -> float | None:
    subset = metrics[
        (metrics["model_name"] == model_name)
        & (metrics["split_name"] == split_name)
        & (metrics["metric_name"] == metric_name)
    ].copy()
    if subset.empty:
        return None
    value = pd.to_numeric(subset["metric_value"], errors="coerce").dropna()
    if value.empty:
        return None
    return float(value.iloc[0])


def build_model_selection_audit(
    model_summaries: Sequence[Mapping[str, Any]],
    metrics: pd.DataFrame,
    split_scheme: str,
) -> pd.DataFrame:
    """Build validation-only model-selection audit."""

    rows: list[dict[str, Any]] = []
    validation_split = f"{split_scheme}_validation"
    test_split = f"{split_scheme}_test"

    for summary in model_summaries:
        model_name = str(summary["model_name"])
        rows.append(
            {
                "model_name": model_name,
                "model_family": summary["model_family"],
                "feature_set_name": summary["feature_set_name"],
                "prediction_setting": summary["prediction_setting"],
                "hyperparameter_id": summary["hyperparameter_id"],
                "parameters": summary["parameters"],
                "validation_mae": _metric_value(metrics, model_name, "count__mae", validation_split),
                "validation_spearman": _metric_value(metrics, model_name, "ranking__spearman_corr", validation_split),
                "validation_top_10pct_overlap_rate": _metric_value(
                    metrics, model_name, "ranking__top_10pct_overlap_rate", validation_split
                ),
                "test_mae": _metric_value(metrics, model_name, "count__mae", test_split),
                "test_spearman": _metric_value(metrics, model_name, "ranking__spearman_corr", test_split),
                "selection_rule": "min_validation_mae_within_feature_set_and_model_family",
                "selected_for_test_summary": False,
                "selected_overall_strict_forecasting": False,
                "selected_overall_retrospective": False,
            }
        )

    audit = pd.DataFrame(rows)
    if audit.empty:
        return audit

    audit["validation_mae_numeric"] = pd.to_numeric(audit["validation_mae"], errors="coerce")

    for (_, _), group in audit.groupby(["feature_set_name", "model_family"]):
        valid = group.dropna(subset=["validation_mae_numeric"])
        if valid.empty:
            continue
        best_idx = valid.sort_values("validation_mae_numeric", ascending=True).index[0]
        audit.loc[best_idx, "selected_for_test_summary"] = True

    strict = audit[
        audit["prediction_setting"].isin([STRICT_TRAIN_STATIC_SETTING, ROLLING_HISTORY_SETTING])
        & audit["validation_mae_numeric"].notna()
    ].copy()
    if not strict.empty:
        best_idx = strict.sort_values("validation_mae_numeric", ascending=True).index[0]
        audit.loc[best_idx, "selected_overall_strict_forecasting"] = True

    retro = audit[
        (audit["prediction_setting"] == RETROSPECTIVE_SETTING)
        & audit["validation_mae_numeric"].notna()
    ].copy()
    if not retro.empty:
        best_idx = retro.sort_values("validation_mae_numeric", ascending=True).index[0]
        audit.loc[best_idx, "selected_overall_retrospective"] = True

    return audit.drop(columns=["validation_mae_numeric"])


def dataframe_to_markdown(df: pd.DataFrame, max_rows: int = 80) -> str:
    """Render dataframe to Markdown with fallback when tabulate is unavailable."""

    if df.empty:
        return "_No rows._"

    display = df.head(max_rows).copy()
    try:
        return display.to_markdown(index=False)
    except Exception:
        return "```text\n" + display.to_string(index=False) + "\n```"


def compact_metrics_summary(metrics: pd.DataFrame, selected_models: Sequence[str] | None = None) -> pd.DataFrame:
    """Create compact metrics summary for the report."""

    if metrics.empty:
        return metrics

    wanted = [
        "count__mae",
        "count__rmse",
        "count__mean_poisson_deviance",
        "ranking__spearman_corr",
        "ranking__kendall_corr",
        "ranking__ndcg_at_10",
        "ranking__ndcg_at_25",
        "ranking__ndcg_at_50",
        "ranking__ndcg_at_100",
        "ranking__top10_overlap_rate",
        "ranking__top25_overlap_rate",
        "ranking__top50_overlap_rate",
        "ranking__top100_overlap_rate",
        "ranking__top_5pct_overlap_rate",
        "ranking__top_10pct_overlap_rate",
    ]

    out = metrics[metrics["metric_name"].isin(wanted)].copy()
    if selected_models is not None:
        out = out[out["model_name"].isin(set(selected_models))].copy()

    cols = [
        "split_name",
        "prediction_setting",
        "model_name",
        "model_family",
        "hyperparameter_id",
        "feature_set_name",
        "metric_name",
        "metric_value",
        "higher_is_better",
        "n_rows",
    ]
    return out[[col for col in cols if col in out.columns]].reset_index(drop=True)


def selected_test_table(selection_audit: pd.DataFrame) -> pd.DataFrame:
    """Return selected model rows for report."""

    if selection_audit.empty:
        return selection_audit

    cols = [
        "model_name",
        "model_family",
        "feature_set_name",
        "prediction_setting",
        "hyperparameter_id",
        "validation_mae",
        "validation_spearman",
        "test_mae",
        "test_spearman",
        "selected_overall_strict_forecasting",
        "selected_overall_retrospective",
    ]
    selected = selection_audit[selection_audit["selected_for_test_summary"].astype(bool)].copy()
    return selected[[col for col in cols if col in selected.columns]].sort_values(
        ["prediction_setting", "feature_set_name", "model_family"]
    )


def render_a3_report(
    *,
    metrics: pd.DataFrame,
    feature_set_audit: pd.DataFrame,
    feature_lineage_audit: pd.DataFrame,
    feature_importance: pd.DataFrame,
    model_selection_audit: pd.DataFrame,
    score_audit: Sequence[Mapping[str, Any]],
    static_audit: Sequence[Mapping[str, Any]],
    row_counts: Mapping[str, Any],
    outputs: Mapping[str, str],
    split_scheme: str,
    generated_at: str,
    include_sklearn_models: bool,
    include_random_forest: bool,
    hgb_grid: str,
) -> str:
    """Render A3 report."""

    selected_models = (
        model_selection_audit.loc[
            model_selection_audit["selected_for_test_summary"].astype(bool),
            "model_name",
        ].tolist()
        if not model_selection_audit.empty else []
    )
    compact_selected = compact_metrics_summary(metrics, selected_models=selected_models)
    selected_table = selected_test_table(model_selection_audit)

    lines: list[str] = []
    lines.append("# A3 Feature-Parity Tabular Baselines — Montréal 311 Water/Drainage v0\n")
    lines.append(f"Generated at: `{generated_at}`\n")
    lines.append(f"Split scheme: `{split_scheme}`\n")
    lines.append(f"Split type: `{split_type_for_scheme(split_scheme)}`\n")

    lines.append("## Purpose\n")
    lines.append(
        "A3 establishes strong non-graph tabular baselines before GraphSAGE/HGNN. "
        "The goal is to determine whether ordinary tabular ML can beat A0/A1/A2 "
        "before claiming value from graph structure.\n"
    )

    lines.append("## Prediction-setting interpretation\n")
    lines.append(
        "- `forecasting_v0`: static/calendar/train-summary features only.\n"
        "- `rolling_observed_history_v0`: uses lagged/rolling observed history. This is valid for rolling monthly forecasting, not for forecasting the whole future horizon from the train endpoint.\n"
        "- `retrospective_explanatory_v0`: uses same-month non-water 311 reporting exposure and is not a forecasting baseline.\n"
    )

    lines.append("## Row counts\n")
    lines.append("| Partition | Rows |")
    lines.append("|---|---:|")
    for key in ["train", "validation", "test"]:
        lines.append(f"| `{key}` | {row_counts.get(key)} |")
    lines.append("")

    lines.append("## Model families\n")
    lines.append("| Family | Included | Notes |")
    lines.append("|---|:---:|---|")
    lines.append("| `ridge_log_count` | `True` | NumPy ridge model fit on `log1p(y)`. |")
    lines.append(
        f"| `hist_gradient_boosting_poisson` | `{include_sklearn_models and HistGradientBoostingRegressor is not None}` | "
        f"scikit-learn HGB with Poisson loss; grid=`{hgb_grid}`. |"
    )
    lines.append(
        f"| `random_forest_log_count` | `{include_sklearn_models and include_random_forest and RandomForestRegressor is not None}` | "
        "Conservative optional diagnostic model fit on `log1p(y)`. |"
    )
    lines.append("")

    lines.append("## Feature sets\n")
    lines.append(dataframe_to_markdown(feature_set_audit, max_rows=30))
    lines.append("")

    lines.append("## Validation-only model selection audit\n")
    lines.append(
        "Selection uses validation MAE only. Test metrics are reported after selection and are not used for choosing models.\n"
    )
    lines.append(dataframe_to_markdown(selected_table, max_rows=80))
    lines.append("")

    lines.append("## Compact metrics for selected candidates\n")
    lines.append(dataframe_to_markdown(compact_selected, max_rows=180))
    lines.append("")

    lines.append("## Feature-lineage audit preview\n")
    lineage_cols = [
        "feature_name",
        "feature_family",
        "source_column",
        "transformation",
        "prediction_setting_allowed",
        "uses_target_history",
        "uses_reporting_history",
        "uses_same_month_information",
        "lag_months",
        "rolling_window",
        "shift_applied",
        "is_strict_forecasting_safe",
    ]
    lineage_cols = [col for col in lineage_cols if col in feature_lineage_audit.columns]
    lines.append(dataframe_to_markdown(feature_lineage_audit[lineage_cols], max_rows=80))
    lines.append("")

    lines.append("## SVI score audit\n")
    lines.append(dataframe_to_markdown(pd.DataFrame(score_audit), max_rows=40))
    lines.append("")

    lines.append("## Static SVI audit\n")
    lines.append(dataframe_to_markdown(pd.DataFrame(static_audit), max_rows=40))
    lines.append("")

    lines.append("## Feature importance preview\n")
    if feature_importance.empty:
        lines.append("_No feature importance available._\n")
    else:
        cols = [
            "model_name",
            "model_family",
            "feature_set_name",
            "feature",
            "importance_type",
            "importance",
            "absolute_importance",
        ]
        cols = [col for col in cols if col in feature_importance.columns]
        preview = feature_importance[cols].copy()
        if "absolute_importance" in preview.columns:
            preview = preview.sort_values("absolute_importance", ascending=False)
        lines.append(dataframe_to_markdown(preview, max_rows=80))
        lines.append("")

    lines.append("## Output artifacts\n")
    lines.append("| Artifact | Path |")
    lines.append("|---|---|")
    for key, value in outputs.items():
        lines.append(f"| `{key}` | `{value}` |")
    lines.append("")

    lines.append("## Leakage notes\n")
    lines.append(
        "- Strict train/static feature sets exclude rolling target/reporting history and same-month reporting controls.\n"
        "- Rolling observed-history feature sets use lag and rolling features shifted by at least one month within tract.\n"
        "- Retrospective feature sets using `total_311_count_non_water_drainage` are labeled `retrospective_explanatory_v0`.\n"
        "- `total_311_count_all` is not used because it contains the target.\n"
        "- Primary SVI features are `svi_percentile`/`svi_score_raw`; rank/class SVI are included only in diagnostic expanded feature sets.\n"
        "- No SoVI columns are used in Track A.\n"
    )

    lines.append("## Interpretation warning\n")
    lines.append(
        "Feature importance and coefficients indicate predictive association within this benchmark, not causal influence. "
        "A3 defines the non-graph ML floor; graph models should be compared against the best selected strict A3 model, "
        "not only against raw SVI or naive baselines.\n"
    )

    return "\n".join(lines)


def build_metadata(
    *,
    config: Mapping[str, Any],
    config_path: Path,
    panel_path: Path,
    split_path: Path,
    outputs: Mapping[str, str],
    split_scheme: str,
    row_counts: Mapping[str, Any],
    score_audit: Sequence[Mapping[str, Any]],
    static_audit: Sequence[Mapping[str, Any]],
    model_summaries: Sequence[Mapping[str, Any]],
    feature_sets: Sequence[FeatureSetSpec],
    metrics: pd.DataFrame,
    generated_at: str,
    ridge_alphas: Sequence[float],
    include_sklearn_models: bool,
    include_random_forest: bool,
    hgb_grid: str,
) -> dict[str, Any]:
    """Build A3 metadata."""

    return to_jsonable(
        {
            "benchmark_id": str(config.get("benchmark_id", "mtl_311_water_v0")),
            "dataset_version": DATASET_VERSION_DEFAULT,
            "generated_at": generated_at,
            "config_path": str(config_path),
            "config_hash": config_hash(config),
            "panel_path": str(panel_path),
            "panel_sha256": file_hash(panel_path),
            "split_assignments_path": str(split_path),
            "split_assignments_sha256": file_hash(split_path),
            "model_stage": MODEL_STAGE,
            "model_name": "A3_feature_parity_tabular_suite",
            "split_scheme": split_scheme,
            "split_type": split_type_for_scheme(split_scheme),
            "target_name": TARGET_NAME,
            "target_type": "count",
            "row_counts": row_counts,
            "ridge_alphas": list(ridge_alphas),
            "include_sklearn_models": include_sklearn_models,
            "include_random_forest": include_random_forest,
            "hgb_grid": hgb_grid,
            "random_seed": RANDOM_SEED,
            "sklearn_version": getattr(sklearn, "__version__", None) if sklearn is not None else None,
            "sklearn_random_forest_available": RandomForestRegressor is not None,
            "sklearn_hist_gradient_boosting_available": HistGradientBoostingRegressor is not None,
            "prediction_settings": {
                "forecasting_v0": "static/calendar/train-summary features only",
                "rolling_observed_history_v0": "lagged and rolling observed-history features shifted by at least one month",
                "retrospective_explanatory_v0": "same-month non-water 311 reporting exposure allowed",
            },
            "feature_sets": [
                {
                    "name": spec.name,
                    "prediction_setting": spec.prediction_setting,
                    "n_features": len(spec.feature_columns),
                    "is_primary_feature_set": spec.is_primary_feature_set,
                    "feature_columns": spec.feature_columns,
                    "description": spec.description,
                }
                for spec in feature_sets
            ],
            "score_audit": list(score_audit),
            "static_score_audit": list(static_audit),
            "models": list(model_summaries),
            "metric_rows": int(len(metrics)),
            "outputs": dict(outputs),
            "notes": (
                "A3 implements feature-parity non-graph tabular baselines with explicit separation of "
                "static forecasting, rolling observed-history forecasting, and retrospective reporting controls. "
                "Model selection uses validation MAE."
            ),
        }
    )


def run_a3_tabular_feature_parity(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    repo_root: str | Path | None = None,
    split_scheme: str = DEFAULT_SPLIT_SCHEME,
    ridge_alphas: Sequence[float] = DEFAULT_RIDGE_ALPHAS,
    include_sklearn_models: bool = True,
    include_random_forest: bool = True,
    include_diagnostic_svi_sets: bool = True,
    hgb_grid: str = "small",
) -> dict[str, Any]:
    """Run A3 feature-parity tabular baselines and write standard artifacts."""

    require_runtime_dependencies()
    ridge_alphas = parse_float_list(ridge_alphas)

    config, root, resolved_config_path, panel_path, split_path, frame = load_benchmark_frame(
        config_path=config_path,
        repo_root=repo_root,
    )

    frame = normalize_frame_for_a3(frame, split_scheme=split_scheme)
    frame, svi_specs, score_audit = build_svi_score_specs(frame)
    static_audit = validate_static_svi_scores(frame, svi_specs)

    row_counts = split_counts(frame, split_scheme=split_scheme)
    missing_required = [part for part in ["train", "validation", "test"] if row_counts.get(part, 0) <= 0]
    if missing_required:
        raise A3BaselineError(
            f"Split scheme {split_scheme!r} missing required partitions: {missing_required}. Counts: {row_counts}"
        )

    train = train_frame(frame, split_scheme=split_scheme)
    frame, feature_groups, feature_lineage = prepare_feature_frame(frame, split_scheme, svi_specs)

    feature_sets = build_feature_set_specs(
        feature_groups,
        include_diagnostic_svi_sets=include_diagnostic_svi_sets,
    )
    validate_feature_sets(feature_sets, feature_lineage)

    candidates = candidate_specs(
        ridge_alphas=ridge_alphas,
        include_sklearn_models=include_sklearn_models,
        include_random_forest=include_random_forest,
        hgb_grid=hgb_grid,
    )
    if not candidates:
        raise A3BaselineError("No model candidates available for A3.")

    paths = get_baseline_paths(config, root, STAGE_SLUG)
    output_dir = paths.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    all_predictions: list[pd.DataFrame] = []
    all_metrics: list[pd.DataFrame] = []
    all_importances: list[pd.DataFrame] = []
    model_summaries: list[dict[str, Any]] = []

    for feature_set in feature_sets:
        for candidate in candidates:
            fitted, pred = fit_candidate_model(frame, train, feature_set, candidate)

            pred_frame = prediction_frame_for_model(
                frame,
                split_scheme=split_scheme,
                fitted=fitted,
                predictions=pred,
            )
            metrics = evaluate_one_model(
                pred_frame,
                config=config,
                full_frame=frame,
                split_scheme=split_scheme,
                fitted=fitted,
            )
            importance = fitted.feature_importance_table()

            all_predictions.append(pred_frame)
            all_metrics.append(metrics)
            if not importance.empty:
                all_importances.append(importance)

            model_summaries.append(
                {
                    "model_name": fitted.model_name,
                    "model_family": fitted.model_family,
                    "hyperparameter_id": fitted.hyperparameter_id,
                    "feature_set_name": fitted.feature_set_name,
                    "prediction_setting": fitted.prediction_setting,
                    "n_features": len(fitted.feature_columns),
                    "feature_columns": fitted.feature_columns,
                    "parameters": fitted.parameters,
                    "standardize": fitted.standardize,
                    "target_transform": fitted.target_transform,
                    "inverse_transform": fitted.inverse_transform,
                }
            )

    predictions_long = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
    metrics = pd.concat(all_metrics, ignore_index=True) if all_metrics else pd.DataFrame()
    metrics = standardize_a3_metric_schema(metrics)
    feature_importance = pd.concat(all_importances, ignore_index=True) if all_importances else pd.DataFrame()

    feature_set_audit = pd.DataFrame(feature_set_audit_rows(feature_sets, feature_lineage))
    model_selection_audit = build_model_selection_audit(model_summaries, metrics, split_scheme)

    feature_set_audit_path = output_dir / "feature_set_audit.csv"
    feature_lineage_audit_path = output_dir / "feature_lineage_audit.csv"
    feature_importance_path = output_dir / "feature_importance.csv"
    model_audit_path = output_dir / "model_audit.csv"
    model_selection_audit_path = output_dir / "model_selection_audit.csv"
    score_audit_path = output_dir / "svi_score_audit.csv"
    static_audit_path = output_dir / "svi_static_score_audit.csv"

    feature_set_audit.to_csv(feature_set_audit_path, index=False)
    feature_lineage.to_csv(feature_lineage_audit_path, index=False)
    feature_importance.to_csv(feature_importance_path, index=False)
    pd.DataFrame(model_summaries).to_csv(model_audit_path, index=False)
    model_selection_audit.to_csv(model_selection_audit_path, index=False)
    pd.DataFrame(score_audit).to_csv(score_audit_path, index=False)
    pd.DataFrame(static_audit).to_csv(static_audit_path, index=False)

    written_predictions = write_long_prediction_partitions(predictions_long, output_dir, split_scheme=split_scheme)

    generated_at = datetime.now(timezone.utc).isoformat()

    outputs = {
        "metrics": str(paths.metrics),
        "model_metadata": str(paths.model_metadata),
        "baseline_report": str(paths.baseline_report),
        "feature_set_audit": str(feature_set_audit_path),
        "feature_lineage_audit": str(feature_lineage_audit_path),
        "feature_importance": str(feature_importance_path),
        "model_audit": str(model_audit_path),
        "model_selection_audit": str(model_selection_audit_path),
        "svi_score_audit": str(score_audit_path),
        "svi_static_score_audit": str(static_audit_path),
        **written_predictions,
    }

    metadata = build_metadata(
        config=config,
        config_path=resolved_config_path,
        panel_path=panel_path,
        split_path=split_path,
        outputs=outputs,
        split_scheme=split_scheme,
        row_counts=row_counts,
        score_audit=score_audit,
        static_audit=static_audit,
        model_summaries=model_summaries,
        feature_sets=feature_sets,
        metrics=metrics,
        generated_at=generated_at,
        ridge_alphas=ridge_alphas,
        include_sklearn_models=include_sklearn_models,
        include_random_forest=include_random_forest,
        hgb_grid=hgb_grid,
    )

    report = render_a3_report(
        metrics=metrics,
        feature_set_audit=feature_set_audit,
        feature_lineage_audit=feature_lineage,
        feature_importance=feature_importance,
        model_selection_audit=model_selection_audit,
        score_audit=score_audit,
        static_audit=static_audit,
        row_counts=row_counts,
        outputs=outputs,
        split_scheme=split_scheme,
        generated_at=generated_at,
        include_sklearn_models=include_sklearn_models,
        include_random_forest=include_random_forest,
        hgb_grid=hgb_grid,
    )

    metrics.to_csv(paths.metrics, index=False)
    write_json(paths.model_metadata, metadata)
    write_markdown(paths.baseline_report, report)

    return {
        "status": "completed",
        "model_stage": MODEL_STAGE,
        "model_name": "A3_feature_parity_tabular_suite",
        "split_scheme": split_scheme,
        "outputs": outputs,
        "row_counts": row_counts,
        "feature_set_count": len(feature_sets),
        "candidate_count": len(candidates),
        "model_count": len(model_summaries),
        "metric_rows": int(len(metrics)),
        "prediction_rows": int(len(predictions_long)),
        "feature_importance_rows": int(len(feature_importance)),
        "selected_models": (
            model_selection_audit.loc[
                model_selection_audit["selected_for_test_summary"].astype(bool), "model_name"
            ].tolist()
            if not model_selection_audit.empty else []
        ),
    }


def a3_brief(result: Mapping[str, Any]) -> str:
    """Return concise A3 run summary."""

    outputs = result.get("outputs", {})
    return (
        "A3 feature-parity tabular baselines completed.\n"
        f"Status: {result.get('status')}\n"
        f"Split scheme: {result.get('split_scheme')}\n"
        f"Feature sets: {result.get('feature_set_count')}\n"
        f"Candidates per feature set: {result.get('candidate_count')}\n"
        f"Models: {result.get('model_count')}\n"
        f"Metric rows: {result.get('metric_rows')}\n"
        f"Prediction rows: {result.get('prediction_rows')}\n"
        f"Feature importance rows: {result.get('feature_importance_rows')}\n"
        f"Metrics: {outputs.get('metrics')}\n"
        f"Report: {outputs.get('baseline_report')}\n"
    )


def main() -> None:
    """CLI entry point for direct module execution."""

    parser = argparse.ArgumentParser(
        description="Run A3 feature-parity tabular baselines for Montréal 311 water/drainage."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help=f"Config path. Default: {DEFAULT_CONFIG_PATH}")
    parser.add_argument("--repo-root", default=None, help="Repository root. Defaults to automatic detection.")
    parser.add_argument(
        "--split-scheme",
        default=DEFAULT_SPLIT_SCHEME,
        choices=sorted(["temporal", "random_debug", "spatial_block"]),
        help=f"Split scheme to evaluate. Default: {DEFAULT_SPLIT_SCHEME}",
    )
    parser.add_argument(
        "--ridge-alphas",
        default=",".join(str(x) for x in DEFAULT_RIDGE_ALPHAS),
        help="Comma-separated ridge alphas. Default: 0.1,1.0,10.0",
    )
    parser.add_argument(
        "--no-sklearn-models",
        action="store_true",
        help="Disable optional scikit-learn random forest / gradient boosting models.",
    )
    parser.add_argument(
        "--no-random-forest",
        action="store_true",
        help="Disable optional random forest diagnostic model.",
    )
    parser.add_argument(
        "--no-diagnostic-svi-sets",
        action="store_true",
        help="Disable diagnostic SVI-expanded feature sets that include rank/class SVI encodings.",
    )
    parser.add_argument(
        "--hgb-grid",
        default="small",
        choices=["small", "medium"],
        help="HistGradientBoosting Poisson grid size. Default: small.",
    )

    args = parser.parse_args()

    result = run_a3_tabular_feature_parity(
        config_path=args.config,
        repo_root=args.repo_root,
        split_scheme=args.split_scheme,
        ridge_alphas=parse_float_list(args.ridge_alphas),
        include_sklearn_models=not args.no_sklearn_models,
        include_random_forest=not args.no_random_forest,
        include_diagnostic_svi_sets=not args.no_diagnostic_svi_sets,
        hgb_grid=args.hgb_grid,
    )

    print(a3_brief(result).rstrip())
    print("\nWritten outputs:")
    for label, path in result.get("outputs", {}).items():
        print(f"  {label}: {path}")


if __name__ == "__main__":
    main()


__all__ = [
    "A3BaselineError",
    "DEFAULT_SPLIT_SCHEME",
    "FeatureMeta",
    "FeatureSetSpec",
    "FittedTabularModel",
    "MODEL_STAGE",
    "STAGE_SLUG",
    "a3_brief",
    "build_feature_set_specs",
    "candidate_specs",
    "prepare_feature_frame",
    "run_a3_tabular_feature_parity",
]
