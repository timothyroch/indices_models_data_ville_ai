"""
A0 naive temporal/exposure baselines for the Montréal 311 water/drainage benchmark.

This module implements the first baseline ladder stage from
``baseline_plan_mtl_311_v0.md``.

Implemented baselines:

- A0.1 global train mean
- A0.2 month-of-year train mean
- A0.3 tract train mean
- A0.4 tract × month-of-year train mean with fallback
- A0.5 previous-month persistence with fallback
- A0.6 previous-year same-month persistence with fallback
- A0.7 population exposure train-rate baseline
- A0.8 same-month non-water 311 reporting exposure baseline, retrospective only

The module intentionally does not implement calibrated SVI, tabular ML, graph
models, explainability, feature engineering, or split construction.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

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

from ville_hgnn.baselines.common import (
    BINARY_TARGET_COLUMN,
    DATASET_VERSION_DEFAULT,
    DEFAULT_CONFIG_PATH,
    TARGET_COLUMN,
    BaselineError,
    BaselinePaths,
    build_run_context,
    evaluate_prediction_frame,
    get_baseline_paths,
    load_benchmark_frame,
    split_column_for_scheme,
    split_counts,
    split_type_for_scheme,
)
from ville_hgnn.utils.io import config_hash, file_hash, to_jsonable, write_json, write_markdown


STAGE_SLUG = "A0_naive_temporal"
MODEL_STAGE = "A0_naive_temporal"
DEFAULT_SPLIT_SCHEME = "temporal"
PREDICTED_COL = "predicted_water_drainage_count"
BINARY_SCORE_COL = "predicted_binary_probability"
OBSERVED_ALIAS_COL = "observed_water_drainage_count"

COUNT_TARGET = TARGET_COLUMN
PERIOD_COL = "period_month"
ZONE_COL = "zone_id"
MONTH_COL = "month"

MIN_PREDICTION = 0.0


class A0BaselineError(BaselineError):
    """Raised when A0 baseline generation fails."""


def require_runtime_dependencies() -> None:
    """Fail clearly if numpy/pandas are unavailable."""

    if pd is None:
        raise A0BaselineError("pandas is required for A0 baselines.") from _PANDAS_IMPORT_ERROR
    if np is None:
        raise A0BaselineError("numpy is required for A0 baselines.") from _NUMPY_IMPORT_ERROR


def normalize_panel_for_a0(frame: pd.DataFrame, split_scheme: str) -> pd.DataFrame:
    """Normalize panel columns required by A0 baselines."""

    required = [ZONE_COL, PERIOD_COL, COUNT_TARGET, split_column_for_scheme(split_scheme)]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise A0BaselineError(f"A0 input frame missing required columns: {missing}")

    out = frame.copy()
    out[ZONE_COL] = out[ZONE_COL].astype(str)

    parsed = pd.to_datetime(out[PERIOD_COL].astype(str), errors="coerce")
    if parsed.isna().any():
        bad = out.loc[parsed.isna(), PERIOD_COL].drop_duplicates().head(20).tolist()
        raise A0BaselineError(f"Could not parse period_month values: {bad}")

    out[PERIOD_COL] = parsed.dt.to_period("M").astype(str)
    periods = pd.PeriodIndex(out[PERIOD_COL].astype(str), freq="M")
    out["_period"] = periods
    out["_period_ordinal"] = [period.ordinal for period in periods]

    if MONTH_COL not in out.columns:
        out[MONTH_COL] = parsed.dt.month.astype(int)
    else:
        out[MONTH_COL] = pd.to_numeric(out[MONTH_COL], errors="coerce").astype("Int64")

    if out[MONTH_COL].isna().any():
        bad = out.loc[out[MONTH_COL].isna(), [ZONE_COL, PERIOD_COL]].head(20).to_dict(orient="records")
        raise A0BaselineError(f"Could not obtain month-of-year values. Examples: {bad}")

    out[MONTH_COL] = out[MONTH_COL].astype(int)

    out[COUNT_TARGET] = pd.to_numeric(out[COUNT_TARGET], errors="coerce")
    if out[COUNT_TARGET].isna().any():
        n_missing = int(out[COUNT_TARGET].isna().sum())
        raise A0BaselineError(f"{COUNT_TARGET} contains missing/non-numeric rows: {n_missing}")

    if (out[COUNT_TARGET] < 0).any():
        n_negative = int((out[COUNT_TARGET] < 0).sum())
        raise A0BaselineError(f"{COUNT_TARGET} contains negative rows: {n_negative}")

    if BINARY_TARGET_COLUMN not in out.columns:
        out[BINARY_TARGET_COLUMN] = (out[COUNT_TARGET] > 0).astype(int)
    else:
        out[BINARY_TARGET_COLUMN] = (
            pd.to_numeric(out[BINARY_TARGET_COLUMN], errors="coerce")
            .fillna(0)
            .astype(int)
        )

    return out.sort_values([ZONE_COL, "_period_ordinal"]).reset_index(drop=True)


def train_frame(frame: pd.DataFrame, split_scheme: str) -> pd.DataFrame:
    """Return training rows for the selected split scheme."""

    split_col = split_column_for_scheme(split_scheme)
    train = frame[frame[split_col].astype(str) == "train"].copy()

    if train.empty:
        raise A0BaselineError(f"No training rows found for split_scheme={split_scheme!r}.")

    return train


def clip_prediction_values(values: Any) -> pd.Series:
    """Convert predictions to numeric nonnegative pandas Series."""

    series = pd.Series(values)
    series = pd.to_numeric(series, errors="coerce")

    if series.isna().any():
        n_missing = int(series.isna().sum())
        raise A0BaselineError(f"Generated predictions contain missing/non-numeric rows: {n_missing}")

    return series.clip(lower=MIN_PREDICTION)


def poisson_any_probability(mu: Any) -> pd.Series:
    """Convert count-rate predictions into P(Y > 0) using a Poisson assumption."""

    values = pd.to_numeric(pd.Series(mu), errors="coerce").fillna(0).clip(lower=0)
    return 1.0 - np.exp(-values)


def safe_global_train_mean(train: pd.DataFrame) -> float:
    """Return global train mean target."""

    value = float(train[COUNT_TARGET].mean())
    if math.isnan(value) or math.isinf(value):
        raise A0BaselineError("Global train mean is not finite.")
    return max(value, MIN_PREDICTION)


def make_global_train_mean_predictor(train: pd.DataFrame) -> tuple[Callable[[pd.DataFrame], pd.Series], dict[str, Any]]:
    """A0.1 global train mean."""

    global_mean = safe_global_train_mean(train)

    def predict(frame: pd.DataFrame) -> pd.Series:
        return pd.Series(global_mean, index=frame.index, dtype=float)

    metadata = {
        "formula": "E_train[y]",
        "global_train_mean": global_mean,
        "uses_target_history": False,
        "uses_same_month_reporting_control": False,
    }
    return predict, metadata


def make_month_of_year_train_mean_predictor(train: pd.DataFrame) -> tuple[Callable[[pd.DataFrame], pd.Series], dict[str, Any]]:
    """A0.2 month-of-year train mean with global fallback."""

    global_mean = safe_global_train_mean(train)
    month_mean = train.groupby(MONTH_COL)[COUNT_TARGET].mean().to_dict()

    def predict(frame: pd.DataFrame) -> pd.Series:
        pred = frame[MONTH_COL].map(month_mean).fillna(global_mean)
        return pred.astype(float)

    metadata = {
        "formula": "E_train[y | month_of_year]",
        "fallback": "global_train_mean",
        "global_train_mean": global_mean,
        "n_months_with_train_mean": len(month_mean),
        "uses_target_history": False,
        "uses_same_month_reporting_control": False,
    }
    return predict, metadata


def make_tract_train_mean_predictor(train: pd.DataFrame) -> tuple[Callable[[pd.DataFrame], pd.Series], dict[str, Any]]:
    """A0.3 tract train mean with global fallback."""

    global_mean = safe_global_train_mean(train)
    tract_mean = train.groupby(ZONE_COL)[COUNT_TARGET].mean().to_dict()

    def predict(frame: pd.DataFrame) -> pd.Series:
        pred = frame[ZONE_COL].map(tract_mean).fillna(global_mean)
        return pred.astype(float)

    metadata = {
        "formula": "E_train[y | zone_id]",
        "fallback": "global_train_mean",
        "global_train_mean": global_mean,
        "n_tracts_with_train_mean": len(tract_mean),
        "uses_target_history": False,
        "uses_same_month_reporting_control": False,
    }
    return predict, metadata


def make_tract_month_of_year_train_mean_predictor(
    train: pd.DataFrame,
) -> tuple[Callable[[pd.DataFrame], pd.Series], dict[str, Any]]:
    """A0.4 tract × month-of-year train mean with hierarchical fallback."""

    global_mean = safe_global_train_mean(train)
    tract_month_mean = train.groupby([ZONE_COL, MONTH_COL])[COUNT_TARGET].mean().rename("_tract_month_mean")
    tract_mean = train.groupby(ZONE_COL)[COUNT_TARGET].mean().rename("_tract_mean")
    month_mean = train.groupby(MONTH_COL)[COUNT_TARGET].mean().rename("_month_mean")

    def predict(frame: pd.DataFrame) -> pd.Series:
        temp = frame[[ZONE_COL, MONTH_COL]].copy()
        temp = temp.merge(
            tract_month_mean.reset_index(),
            on=[ZONE_COL, MONTH_COL],
            how="left",
            validate="many_to_one",
        )
        temp = temp.merge(
            tract_mean.reset_index(),
            on=ZONE_COL,
            how="left",
            validate="many_to_one",
        )
        temp = temp.merge(
            month_mean.reset_index(),
            on=MONTH_COL,
            how="left",
            validate="many_to_one",
        )
        pred = temp["_tract_month_mean"].fillna(temp["_tract_mean"]).fillna(temp["_month_mean"]).fillna(global_mean)
        pred.index = frame.index
        return pred.astype(float)

    metadata = {
        "formula": "E_train[y | zone_id, month_of_year]",
        "fallback_order": ["tract_train_mean", "month_of_year_train_mean", "global_train_mean"],
        "global_train_mean": global_mean,
        "n_tract_month_cells_with_train_mean": int(len(tract_month_mean)),
        "uses_target_history": False,
        "uses_same_month_reporting_control": False,
    }
    return predict, metadata


def make_previous_month_persistence_predictor(
    full_frame: pd.DataFrame,
    train: pd.DataFrame,
) -> tuple[Callable[[pd.DataFrame], pd.Series], dict[str, Any]]:
    """
    A0.5 previous-month persistence.

    Uses the observed value from the same tract in the previous month. This is a
    one-step observed-history baseline. Missing lags fall back to train-only
    tract mean, then global train mean.
    """

    global_mean = safe_global_train_mean(train)
    tract_mean = train.groupby(ZONE_COL)[COUNT_TARGET].mean().rename("_tract_mean")

    lag_source = full_frame[[ZONE_COL, "_period_ordinal", COUNT_TARGET]].copy()
    lag_source["_period_ordinal"] = lag_source["_period_ordinal"] + 1
    lag_source = lag_source.rename(columns={COUNT_TARGET: "_lag1_count"})

    def predict(frame: pd.DataFrame) -> pd.Series:
        temp = frame[[ZONE_COL, "_period_ordinal"]].copy()
        temp = temp.merge(
            lag_source,
            on=[ZONE_COL, "_period_ordinal"],
            how="left",
            validate="one_to_one",
        )
        temp = temp.merge(
            tract_mean.reset_index(),
            on=ZONE_COL,
            how="left",
            validate="many_to_one",
        )
        pred = temp["_lag1_count"].fillna(temp["_tract_mean"]).fillna(global_mean)
        pred.index = frame.index
        return pred.astype(float)

    n_available = int(lag_source["_lag1_count"].notna().sum())
    metadata = {
        "formula": "y_{zone,t-1}",
        "fallback_order": ["tract_train_mean", "global_train_mean"],
        "global_train_mean": global_mean,
        "uses_target_history": True,
        "target_history_rule": "strictly_past_observed_previous_month",
        "uses_same_month_reporting_control": False,
        "lag_rows_available_before_filtering": n_available,
    }
    return predict, metadata


def make_previous_year_same_month_persistence_predictor(
    full_frame: pd.DataFrame,
    train: pd.DataFrame,
) -> tuple[Callable[[pd.DataFrame], pd.Series], dict[str, Any]]:
    """
    A0.6 previous-year same-month persistence.

    Uses the observed value from the same tract 12 months earlier. Missing lags
    fall back to train-only tract-month mean, tract mean, month mean, then global
    train mean.
    """

    global_mean = safe_global_train_mean(train)

    lag_source = full_frame[[ZONE_COL, "_period_ordinal", COUNT_TARGET]].copy()
    lag_source["_period_ordinal"] = lag_source["_period_ordinal"] + 12
    lag_source = lag_source.rename(columns={COUNT_TARGET: "_lag12_count"})

    tract_month_mean = train.groupby([ZONE_COL, MONTH_COL])[COUNT_TARGET].mean().rename("_tract_month_mean")
    tract_mean = train.groupby(ZONE_COL)[COUNT_TARGET].mean().rename("_tract_mean")
    month_mean = train.groupby(MONTH_COL)[COUNT_TARGET].mean().rename("_month_mean")

    def predict(frame: pd.DataFrame) -> pd.Series:
        temp = frame[[ZONE_COL, MONTH_COL, "_period_ordinal"]].copy()
        temp = temp.merge(
            lag_source,
            on=[ZONE_COL, "_period_ordinal"],
            how="left",
            validate="one_to_one",
        )
        temp = temp.merge(
            tract_month_mean.reset_index(),
            on=[ZONE_COL, MONTH_COL],
            how="left",
            validate="many_to_one",
        )
        temp = temp.merge(
            tract_mean.reset_index(),
            on=ZONE_COL,
            how="left",
            validate="many_to_one",
        )
        temp = temp.merge(
            month_mean.reset_index(),
            on=MONTH_COL,
            how="left",
            validate="many_to_one",
        )
        pred = (
            temp["_lag12_count"]
            .fillna(temp["_tract_month_mean"])
            .fillna(temp["_tract_mean"])
            .fillna(temp["_month_mean"])
            .fillna(global_mean)
        )
        pred.index = frame.index
        return pred.astype(float)

    metadata = {
        "formula": "y_{zone,t-12}",
        "fallback_order": [
            "tract_month_of_year_train_mean",
            "tract_train_mean",
            "month_of_year_train_mean",
            "global_train_mean",
        ],
        "global_train_mean": global_mean,
        "uses_target_history": True,
        "target_history_rule": "strictly_past_observed_previous_year_same_month",
        "uses_same_month_reporting_control": False,
        "lag_rows_available_before_filtering": int(lag_source["_lag12_count"].notna().sum()),
    }
    return predict, metadata


def make_population_exposure_train_rate_predictor(
    train: pd.DataFrame,
) -> tuple[Callable[[pd.DataFrame], pd.Series], dict[str, Any]]:
    """
    A0.7 population exposure baseline.

    Fits a global train-period water/drainage request rate per person-month and
    predicts:

        population_total_2021 × train_rate
    """

    population_col = "population_total_2021"
    global_mean = safe_global_train_mean(train)

    if population_col not in train.columns:
        def fallback_predict(frame: pd.DataFrame) -> pd.Series:
            return pd.Series(global_mean, index=frame.index, dtype=float)

        return fallback_predict, {
            "formula": "fallback_global_train_mean",
            "fallback_reason": f"missing_column:{population_col}",
            "global_train_mean": global_mean,
            "uses_target_history": False,
            "uses_same_month_reporting_control": False,
        }

    train_pop = pd.to_numeric(train[population_col], errors="coerce").fillna(0).clip(lower=0)
    denominator = float(train_pop.sum())
    numerator = float(train[COUNT_TARGET].sum())

    if denominator <= 0:
        def fallback_predict(frame: pd.DataFrame) -> pd.Series:
            return pd.Series(global_mean, index=frame.index, dtype=float)

        return fallback_predict, {
            "formula": "fallback_global_train_mean",
            "fallback_reason": "nonpositive_train_population_denominator",
            "global_train_mean": global_mean,
            "uses_target_history": False,
            "uses_same_month_reporting_control": False,
        }

    rate = numerator / denominator

    def predict(frame: pd.DataFrame) -> pd.Series:
        pop = pd.to_numeric(frame[population_col], errors="coerce").fillna(0).clip(lower=0)
        return (pop * rate).astype(float)

    metadata = {
        "formula": "population_total_2021 * sum_train(y) / sum_train(population_total_2021)",
        "train_rate_per_person_month": rate,
        "train_target_sum": numerator,
        "train_population_row_sum": denominator,
        "uses_target_history": False,
        "uses_same_month_reporting_control": False,
    }
    return predict, metadata


def make_non_water_reporting_exposure_predictor(
    train: pd.DataFrame,
) -> tuple[Callable[[pd.DataFrame], pd.Series], dict[str, Any]]:
    """
    A0.8 same-month non-water 311 reporting exposure baseline.

    This is explicitly retrospective, not a strict forecasting baseline.

    Fits:

        train water/drainage count per train non-water 311 count

    and predicts:

        total_311_count_non_water_drainage × fitted_rate
    """

    exposure_col = "total_311_count_non_water_drainage"
    global_mean = safe_global_train_mean(train)

    if exposure_col not in train.columns:
        def fallback_predict(frame: pd.DataFrame) -> pd.Series:
            return pd.Series(global_mean, index=frame.index, dtype=float)

        return fallback_predict, {
            "formula": "fallback_global_train_mean",
            "fallback_reason": f"missing_column:{exposure_col}",
            "global_train_mean": global_mean,
            "uses_target_history": False,
            "uses_same_month_reporting_control": False,
        }

    train_exposure = pd.to_numeric(train[exposure_col], errors="coerce").fillna(0).clip(lower=0)
    denominator = float(train_exposure.sum())
    numerator = float(train[COUNT_TARGET].sum())

    if denominator <= 0:
        def fallback_predict(frame: pd.DataFrame) -> pd.Series:
            return pd.Series(global_mean, index=frame.index, dtype=float)

        return fallback_predict, {
            "formula": "fallback_global_train_mean",
            "fallback_reason": "nonpositive_train_non_water_311_denominator",
            "global_train_mean": global_mean,
            "uses_target_history": False,
            "uses_same_month_reporting_control": False,
        }

    rate = numerator / denominator

    def predict(frame: pd.DataFrame) -> pd.Series:
        exposure = pd.to_numeric(frame[exposure_col], errors="coerce").fillna(0).clip(lower=0)
        return (exposure * rate).astype(float)

    metadata = {
        "formula": "total_311_count_non_water_drainage * sum_train(y) / sum_train(total_311_count_non_water_drainage)",
        "train_rate_per_non_water_311_request": rate,
        "train_target_sum": numerator,
        "train_non_water_311_sum": denominator,
        "uses_target_history": False,
        "uses_same_month_reporting_control": True,
        "prediction_setting_note": "retrospective_only_same_month_reporting_control",
    }
    return predict, metadata


def baseline_specs(
    full_frame: pd.DataFrame,
    train: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Construct A0 baseline specifications."""

    return [
        {
            "model_name": "A0_1_global_train_mean",
            "feature_set_name": "global_train_target_mean",
            "prediction_setting": "forecasting_v0",
            "description": "Constant prediction equal to the training-set mean count.",
            "factory": lambda: make_global_train_mean_predictor(train),
        },
        {
            "model_name": "A0_2_month_of_year_train_mean",
            "feature_set_name": "month_of_year_train_target_mean",
            "prediction_setting": "forecasting_v0",
            "description": "Training-set month-of-year mean with global fallback.",
            "factory": lambda: make_month_of_year_train_mean_predictor(train),
        },
        {
            "model_name": "A0_3_tract_train_mean",
            "feature_set_name": "tract_train_target_mean",
            "prediction_setting": "forecasting_v0",
            "description": "Training-set tract mean with global fallback.",
            "factory": lambda: make_tract_train_mean_predictor(train),
        },
        {
            "model_name": "A0_4_tract_month_of_year_train_mean",
            "feature_set_name": "tract_month_of_year_train_target_mean",
            "prediction_setting": "forecasting_v0",
            "description": "Training-set tract × month-of-year mean with hierarchical fallback.",
            "factory": lambda: make_tract_month_of_year_train_mean_predictor(train),
        },
        {
            "model_name": "A0_5_previous_month_persistence",
            "feature_set_name": "lag1_observed_target_with_train_fallback",
            "prediction_setting": "one_step_observed_history_v0",
            "description": "Previous observed month for the same tract with train-only fallback.",
            "factory": lambda: make_previous_month_persistence_predictor(full_frame, train),
        },
        {
            "model_name": "A0_6_previous_year_same_month_persistence",
            "feature_set_name": "lag12_observed_target_with_train_fallback",
            "prediction_setting": "one_step_observed_history_v0",
            "description": "Previous-year same-month observed count with train-only fallback.",
            "factory": lambda: make_previous_year_same_month_persistence_predictor(full_frame, train),
        },
        {
            "model_name": "A0_7_population_exposure_train_rate",
            "feature_set_name": "population_exposure_train_rate",
            "prediction_setting": "forecasting_v0",
            "description": "Global train target rate per person-month multiplied by tract population.",
            "factory": lambda: make_population_exposure_train_rate_predictor(train),
        },
        {
            "model_name": "A0_8_non_water_311_reporting_exposure_retrospective",
            "feature_set_name": "same_month_non_water_311_reporting_exposure",
            "prediction_setting": "retrospective_explanatory_v0",
            "description": "Same-month non-water 311 reporting exposure. Retrospective only.",
            "factory": lambda: make_non_water_reporting_exposure_predictor(train),
        },
    ]


def prediction_frame_for_model(
    frame: pd.DataFrame,
    *,
    split_scheme: str,
    model_name: str,
    feature_set_name: str,
    predictions: pd.Series,
) -> pd.DataFrame:
    """Create standardized long prediction rows for one A0 model."""

    split_col = split_column_for_scheme(split_scheme)

    cols = [
        ZONE_COL,
        PERIOD_COL,
        split_col,
        COUNT_TARGET,
        BINARY_TARGET_COLUMN,
        "year",
        MONTH_COL,
        "population_total_2021",
        "total_311_count_non_water_drainage",
    ]
    cols = [col for col in cols if col in frame.columns]

    out = frame[cols].copy()
    pred = clip_prediction_values(predictions)
    pred.index = out.index

    out[OBSERVED_ALIAS_COL] = out[COUNT_TARGET]
    out[PREDICTED_COL] = pred
    out["predicted_score"] = pred
    out[BINARY_SCORE_COL] = poisson_any_probability(pred)
    out["model_stage"] = MODEL_STAGE
    out["model_name"] = model_name
    out["feature_set_name"] = feature_set_name

    return out


def evaluate_one_model(
    prediction_frame: pd.DataFrame,
    *,
    config: Mapping[str, Any],
    full_frame: pd.DataFrame,
    split_scheme: str,
    model_name: str,
    feature_set_name: str,
    prediction_setting: str,
) -> pd.DataFrame:
    """Evaluate one A0 model and return standardized metrics rows."""

    context = build_run_context(
        config=config,
        frame=full_frame,
        split_scheme=split_scheme,
        prediction_setting=prediction_setting,
        model_stage=MODEL_STAGE,
        model_name=model_name,
        target_name=COUNT_TARGET,
        target_type="count",
        feature_set_name=feature_set_name,
        dataset_version=DATASET_VERSION_DEFAULT,
    )

    return evaluate_prediction_frame(
        prediction_frame,
        context=context,
        split_scheme=split_scheme,
        observed_col=COUNT_TARGET,
        predicted_col=PREDICTED_COL,
        binary_observed_col=BINARY_TARGET_COLUMN,
        binary_score_col=BINARY_SCORE_COL,
        ranking_score_col="predicted_score",
    )


def write_long_prediction_partitions(
    predictions_long: pd.DataFrame,
    paths: BaselinePaths,
    *,
    split_scheme: str,
) -> dict[str, str]:
    """Write validation/test prediction files for all A0 models in long format."""

    split_col = split_column_for_scheme(split_scheme)

    preferred_cols = [
        ZONE_COL,
        PERIOD_COL,
        split_col,
        OBSERVED_ALIAS_COL,
        COUNT_TARGET,
        BINARY_TARGET_COLUMN,
        PREDICTED_COL,
        "predicted_score",
        BINARY_SCORE_COL,
        "model_stage",
        "model_name",
        "feature_set_name",
        "year",
        MONTH_COL,
        "population_total_2021",
        "total_311_count_non_water_drainage",
    ]
    cols = [col for col in preferred_cols if col in predictions_long.columns]

    validation = predictions_long[predictions_long[split_col].astype(str) == "validation"][cols].copy()
    test = predictions_long[predictions_long[split_col].astype(str) == "test"][cols].copy()

    validation.to_parquet(paths.predictions_validation, index=False)
    test.to_parquet(paths.predictions_test, index=False)

    return {
        "predictions_validation": str(paths.predictions_validation),
        "predictions_test": str(paths.predictions_test),
    }


def metrics_summary_table(metrics: pd.DataFrame) -> pd.DataFrame:
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
    if out.empty:
        return out

    available_cols = [
        col for col in [
            "split_name",
            "prediction_setting",
            "model_name",
            "metric_name",
            "metric_value",
            "higher_is_better",
            "n_rows",
        ]
        if col in out.columns
    ]
    out = out[available_cols]
    return out.sort_values(["split_name", "metric_name", "model_name"]).reset_index(drop=True)


def dataframe_to_markdown(df: pd.DataFrame, max_rows: int = 80) -> str:
    """Render dataframe to Markdown with fallback when tabulate is unavailable."""

    if df.empty:
        return "_No rows._"

    display = df.head(max_rows).copy()
    try:
        return display.to_markdown(index=False)
    except Exception:
        return "```text\n" + display.to_string(index=False) + "\n```"


def render_a0_report(
    *,
    metrics: pd.DataFrame,
    model_summaries: Sequence[Mapping[str, Any]],
    row_counts: Mapping[str, Any],
    outputs: Mapping[str, str],
    split_scheme: str,
    split_type: str,
    generated_at: str,
) -> str:
    """Render A0 baseline report."""

    summary = metrics_summary_table(metrics)

    lines: list[str] = []
    lines.append("# A0 Naive Temporal/Exposure Baselines — Montréal 311 Water/Drainage v0\n")
    lines.append(f"Generated at: `{generated_at}`\n")
    lines.append(f"Split scheme: `{split_scheme}`\n")
    lines.append(f"Split type: `{split_type}`\n")

    lines.append("## Purpose\n")
    lines.append(
        "A0 establishes the minimum non-graph benchmark: train-set means, seasonality, "
        "tract history, temporal persistence, population exposure, and a retrospective "
        "same-month non-water 311 reporting exposure baseline. These models provide the "
        "floor that calibrated SVI, tabular ML, GraphSAGE, and HGNN variants must beat.\n"
    )

    lines.append("## Row counts\n")
    lines.append("| Partition | Rows |")
    lines.append("|---|---:|")
    for key in ["train", "validation", "test"]:
        lines.append(f"| `{key}` | {row_counts.get(key)} |")
    lines.append("")

    lines.append("## Implemented baselines\n")
    lines.append("| Model | Prediction setting | Feature set | Notes |")
    lines.append("|---|---|---|---|")
    for item in model_summaries:
        lines.append(
            f"| `{item.get('model_name')}` | `{item.get('prediction_setting')}` | "
            f"`{item.get('feature_set_name')}` | {item.get('description')} |"
        )
    lines.append("")

    lines.append("## Compact metrics summary\n")
    lines.append(dataframe_to_markdown(summary))
    lines.append("")

    lines.append("## Output artifacts\n")
    lines.append("| Artifact | Path |")
    lines.append("|---|---|")
    for key, value in outputs.items():
        lines.append(f"| `{key}` | `{value}` |")
    lines.append("")

    lines.append("## Leakage notes\n")
    lines.append(
        "- A0.1–A0.4 and A0.7 are fitted using training rows only.\n"
        "- A0.5 and A0.6 use strictly past observed target history for persistence baselines.\n"
        "- A0.8 uses same-month `total_311_count_non_water_drainage`; it is retrospective/explanatory only, not a strict forecasting baseline.\n"
        "- No SoVI columns are used in Track A.\n"
    )

    return "\n".join(lines)


def build_a0_metadata(
    *,
    config: Mapping[str, Any],
    config_path: Path,
    panel_path: Path,
    split_path: Path,
    paths: BaselinePaths,
    split_scheme: str,
    row_counts: Mapping[str, Any],
    model_summaries: Sequence[Mapping[str, Any]],
    metrics: pd.DataFrame,
    generated_at: str,
) -> dict[str, Any]:
    """Build A0 suite metadata."""

    metadata = {
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
        "model_name": "A0_naive_temporal_suite",
        "split_scheme": split_scheme,
        "split_type": split_type_for_scheme(split_scheme),
        "target_name": COUNT_TARGET,
        "target_type": "count",
        "row_counts": to_jsonable(row_counts),
        "models": to_jsonable(list(model_summaries)),
        "metric_rows": int(len(metrics)),
        "outputs": paths.to_dict(),
        "notes": (
            "A0 contains multiple naive temporal/exposure baselines. "
            "Metrics are reported per model under the same Dataset v0 split artifacts."
        ),
    }
    return to_jsonable(metadata)


def run_a0_naive_temporal(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    repo_root: str | Path | None = None,
    split_scheme: str = DEFAULT_SPLIT_SCHEME,
) -> dict[str, Any]:
    """
    Run A0 naive temporal/exposure baselines and write standard artifacts.

    Returns a dictionary containing output paths, row counts, and model summaries.
    """

    require_runtime_dependencies()

    config, root, resolved_config_path, panel_path, split_path, frame = load_benchmark_frame(
        config_path=config_path,
        repo_root=repo_root,
    )

    frame = normalize_panel_for_a0(frame, split_scheme=split_scheme)

    counts = split_counts(frame, split_scheme=split_scheme)

    missing_eval = [part for part in ["train", "validation", "test"] if counts.get(part, 0) <= 0]
    if missing_eval:
        raise A0BaselineError(
            f"Split scheme {split_scheme!r} missing required partitions: {missing_eval}. "
            f"Counts: {counts}"
        )

    train = train_frame(frame, split_scheme=split_scheme)
    paths = get_baseline_paths(config, root, STAGE_SLUG)

    all_predictions: list[pd.DataFrame] = []
    all_metrics: list[pd.DataFrame] = []
    model_summaries: list[dict[str, Any]] = []

    for spec in baseline_specs(frame, train):
        predictor, model_metadata = spec["factory"]()
        pred = predictor(frame)
        pred_frame = prediction_frame_for_model(
            frame,
            split_scheme=split_scheme,
            model_name=spec["model_name"],
            feature_set_name=spec["feature_set_name"],
            predictions=pred,
        )

        metric_frame = evaluate_one_model(
            pred_frame,
            config=config,
            full_frame=frame,
            split_scheme=split_scheme,
            model_name=spec["model_name"],
            feature_set_name=spec["feature_set_name"],
            prediction_setting=spec["prediction_setting"],
        )

        all_predictions.append(pred_frame)
        all_metrics.append(metric_frame)
        model_summaries.append(
            {
                "model_name": spec["model_name"],
                "feature_set_name": spec["feature_set_name"],
                "prediction_setting": spec["prediction_setting"],
                "description": spec["description"],
                "parameters": to_jsonable(model_metadata),
            }
        )

    predictions_long = pd.concat(all_predictions, ignore_index=True)
    metrics = pd.concat(all_metrics, ignore_index=True)

    generated_at = datetime.now(timezone.utc).isoformat()

    metadata = build_a0_metadata(
        config=config,
        config_path=resolved_config_path,
        panel_path=panel_path,
        split_path=split_path,
        paths=paths,
        split_scheme=split_scheme,
        row_counts=counts,
        model_summaries=model_summaries,
        metrics=metrics,
        generated_at=generated_at,
    )

    output_paths_preliminary = {
        "metrics": str(paths.metrics),
        "model_metadata": str(paths.model_metadata),
        "baseline_report": str(paths.baseline_report),
        "predictions_validation": str(paths.predictions_validation),
        "predictions_test": str(paths.predictions_test),
    }

    report = render_a0_report(
        metrics=metrics,
        model_summaries=model_summaries,
        row_counts=counts,
        outputs=output_paths_preliminary,
        split_scheme=split_scheme,
        split_type=split_type_for_scheme(split_scheme),
        generated_at=generated_at,
    )

    paths.output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(paths.metrics, index=False)
    write_json(paths.model_metadata, metadata)
    write_markdown(paths.baseline_report, report)
    written_predictions = write_long_prediction_partitions(
        predictions_long,
        paths,
        split_scheme=split_scheme,
    )

    outputs = {
        "metrics": str(paths.metrics),
        "model_metadata": str(paths.model_metadata),
        "baseline_report": str(paths.baseline_report),
        **written_predictions,
    }

    return {
        "status": "completed",
        "model_stage": MODEL_STAGE,
        "model_name": "A0_naive_temporal_suite",
        "split_scheme": split_scheme,
        "outputs": outputs,
        "row_counts": counts,
        "model_count": len(model_summaries),
        "models": model_summaries,
        "metric_rows": int(len(metrics)),
        "prediction_rows": int(len(predictions_long)),
    }


def a0_brief(result: Mapping[str, Any]) -> str:
    """Return concise A0 run summary."""

    outputs = result.get("outputs", {})
    return (
        "A0 naive temporal baselines completed.\n"
        f"Status: {result.get('status')}\n"
        f"Split scheme: {result.get('split_scheme')}\n"
        f"Models: {result.get('model_count')}\n"
        f"Metric rows: {result.get('metric_rows')}\n"
        f"Prediction rows: {result.get('prediction_rows')}\n"
        f"Metrics: {outputs.get('metrics')}\n"
        f"Report: {outputs.get('baseline_report')}\n"
    )


def main() -> None:
    """CLI entry point for direct module execution."""

    import argparse

    parser = argparse.ArgumentParser(
        description="Run A0 naive temporal/exposure baselines for Montréal 311 water/drainage."
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help=f"Config path. Default: {DEFAULT_CONFIG_PATH}",
    )
    parser.add_argument(
        "--repo-root",
        default=None,
        help="Repository root. Defaults to automatic detection.",
    )
    parser.add_argument(
        "--split-scheme",
        default=DEFAULT_SPLIT_SCHEME,
        choices=sorted(["temporal", "random_debug", "spatial_block"]),
        help=f"Split scheme to evaluate. Default: {DEFAULT_SPLIT_SCHEME}",
    )

    args = parser.parse_args()

    result = run_a0_naive_temporal(
        config_path=args.config,
        repo_root=args.repo_root,
        split_scheme=args.split_scheme,
    )

    print(a0_brief(result).rstrip())
    print("\nWritten outputs:")
    for label, path in result.get("outputs", {}).items():
        print(f"  {label}: {path}")


if __name__ == "__main__":
    main()


__all__ = [
    "A0BaselineError",
    "COUNT_TARGET",
    "DEFAULT_SPLIT_SCHEME",
    "MODEL_STAGE",
    "PREDICTED_COL",
    "STAGE_SLUG",
    "a0_brief",
    "baseline_specs",
    "make_global_train_mean_predictor",
    "make_month_of_year_train_mean_predictor",
    "make_non_water_reporting_exposure_predictor",
    "make_population_exposure_train_rate_predictor",
    "make_previous_month_persistence_predictor",
    "make_previous_year_same_month_persistence_predictor",
    "make_tract_month_of_year_train_mean_predictor",
    "make_tract_train_mean_predictor",
    "run_a0_naive_temporal",
]