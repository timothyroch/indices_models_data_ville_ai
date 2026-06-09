"""
A2 calibrated SVI predictors for the Montréal 311 water/drainage benchmark.

A2 tests whether static SVI has predictive value after simple train-only
calibration and basic controls. Unlike A1, this stage produces calibrated count
predictions.

Implemented model family:

    ridge-regularized log-count linear model

Target transform:

    log1p(water_drainage_count)

Prediction inverse transform:

    expm1(predicted_log_count), clipped at zero

Implemented feature sets for each usable SVI score column:

- A2_svi_only
- A2_svi_plus_calendar
- A2_svi_plus_static
- A2_svi_plus_reporting_retrospective

The reporting-control feature set uses same-month
``total_311_count_non_water_drainage`` and is therefore retrospective only, not a
strict forecasting baseline.

This module intentionally does not implement feature-parity tabular ML, graph
models, explainability, split building, or dataset construction.
"""

from __future__ import annotations

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

from ville_hgnn.baselines.a1_svi_direct_ranking import (
    SviScoreSpec,
    build_svi_score_specs,
    validate_static_svi_scores,
)
from ville_hgnn.baselines.common import (
    BINARY_TARGET_COLUMN,
    DATASET_VERSION_DEFAULT,
    DEFAULT_CONFIG_PATH,
    EVAL_PARTITIONS,
    TARGET_COLUMN,
    BaselineError,
    assert_no_forbidden_feature_columns,
    build_run_context,
    evaluate_prediction_frame,
    get_baseline_paths,
    load_benchmark_frame,
    split_column_for_scheme,
    split_counts,
    split_type_for_scheme,
)
from ville_hgnn.utils.io import config_hash, file_hash, to_jsonable, write_json, write_markdown


STAGE_SLUG = "A2_calibrated_svi"
MODEL_STAGE = "A2_calibrated_svi"
DEFAULT_SPLIT_SCHEME = "temporal"

ZONE_COL = "zone_id"
PERIOD_COL = "period_month"
MONTH_COL = "month"

PREDICTED_COL = "predicted_water_drainage_count"
BINARY_SCORE_COL = "predicted_binary_probability"
OBSERVED_ALIAS_COL = "observed_water_drainage_count"

RIDGE_ALPHA_DEFAULT = 1.0
MIN_PREDICTION = 0.0

STATIC_NUMERIC_CANDIDATES = [
    "population_total_2021",
    "land_area_km2",
    "population_density",
    "population_density_per_km2",
]

REPORTING_CONTROL_COL = "total_311_count_non_water_drainage"


class A2BaselineError(BaselineError):
    """Raised when A2 calibrated SVI baseline generation fails."""


@dataclass(frozen=True)
class FeatureSetSpec:
    """A2 feature-set specification."""

    name: str
    prediction_setting: str
    include_calendar: bool
    include_static: bool
    include_reporting: bool
    description: str


@dataclass(frozen=True)
class FittedRidgeLogCountModel:
    """Fitted ridge log-count linear model."""

    model_name: str
    feature_set_name: str
    prediction_setting: str
    source_svi_column: str
    oriented_svi_score_column: str
    score_role: str
    feature_columns: list[str]
    beta: list[float]
    ridge_alpha: float
    train_target_log_mean: float
    train_target_count_mean: float
    feature_medians: dict[str, float]
    feature_means: dict[str, float]
    feature_stds: dict[str, float]
    feature_raw_columns: list[str]

    def coefficients_table(self) -> pd.DataFrame:
        """Return coefficient table including intercept."""

        return pd.DataFrame(
            {
                "model_name": self.model_name,
                "feature_set_name": self.feature_set_name,
                "prediction_setting": self.prediction_setting,
                "source_svi_column": self.source_svi_column,
                "oriented_svi_score_column": self.oriented_svi_score_column,
                "score_role": self.score_role,
                "feature": self.feature_columns,
                "coefficient": self.beta,
                "ridge_alpha": self.ridge_alpha,
            }
        )


def require_runtime_dependencies() -> None:
    """Fail clearly if numpy/pandas are unavailable."""

    if pd is None:
        raise A2BaselineError("pandas is required for A2 baselines.") from _PANDAS_IMPORT_ERROR
    if np is None:
        raise A2BaselineError("numpy is required for A2 baselines.") from _NUMPY_IMPORT_ERROR


def normalize_frame_for_a2(frame: pd.DataFrame, split_scheme: str) -> pd.DataFrame:
    """Normalize required A2 columns."""

    split_col = split_column_for_scheme(split_scheme)
    required = [ZONE_COL, PERIOD_COL, TARGET_COLUMN, split_col]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise A2BaselineError(f"A2 input frame missing required columns: {missing}")

    out = frame.copy()
    out[ZONE_COL] = out[ZONE_COL].astype(str)

    parsed = pd.to_datetime(out[PERIOD_COL].astype(str), errors="coerce")
    if parsed.isna().any():
        bad = out.loc[parsed.isna(), PERIOD_COL].drop_duplicates().head(20).tolist()
        raise A2BaselineError(f"Could not parse period_month values: {bad}")

    out[PERIOD_COL] = parsed.dt.to_period("M").astype(str)

    if MONTH_COL not in out.columns:
        out[MONTH_COL] = parsed.dt.month.astype(int)
    else:
        out[MONTH_COL] = pd.to_numeric(out[MONTH_COL], errors="coerce").astype("Int64")
        if out[MONTH_COL].isna().any():
            out[MONTH_COL] = parsed.dt.month.astype(int)
        out[MONTH_COL] = out[MONTH_COL].astype(int)

    out[TARGET_COLUMN] = pd.to_numeric(out[TARGET_COLUMN], errors="coerce")
    if out[TARGET_COLUMN].isna().any():
        n_missing = int(out[TARGET_COLUMN].isna().sum())
        raise A2BaselineError(f"{TARGET_COLUMN} contains missing/non-numeric rows: {n_missing}")
    if (out[TARGET_COLUMN] < 0).any():
        n_negative = int((out[TARGET_COLUMN] < 0).sum())
        raise A2BaselineError(f"{TARGET_COLUMN} contains negative rows: {n_negative}")

    if BINARY_TARGET_COLUMN not in out.columns:
        out[BINARY_TARGET_COLUMN] = (out[TARGET_COLUMN] > 0).astype(int)
    else:
        out[BINARY_TARGET_COLUMN] = pd.to_numeric(
            out[BINARY_TARGET_COLUMN],
            errors="coerce",
        ).fillna(0).astype(int)

    return out.reset_index(drop=True)


def train_frame(frame: pd.DataFrame, split_scheme: str) -> pd.DataFrame:
    """Return train rows."""

    split_col = split_column_for_scheme(split_scheme)
    train = frame[frame[split_col].astype(str) == "train"].copy()

    if train.empty:
        raise A2BaselineError(f"No train rows found for split_scheme={split_scheme!r}.")

    return train


def filter_svi_specs(
    svi_specs: Sequence[SviScoreSpec],
    *,
    primary_only: bool = False,
) -> list[SviScoreSpec]:
    """
    Optionally keep only primary continuous SVI scores.

    By default A2 evaluates both primary and diagnostic SVI score columns, but
    reports their roles explicitly. With ``primary_only=True``, only
    ``svi_percentile`` / ``svi_score_raw`` style continuous candidates are kept.
    """

    if not primary_only:
        return list(svi_specs)

    primary = [
        spec for spec in svi_specs
        if spec.score_role == "primary_continuous_svi_score_candidate"
    ]

    if not primary:
        raise A2BaselineError(
            "primary_only=True was requested, but no primary continuous SVI scores were found."
        )

    return primary


def feature_set_specs(frame: pd.DataFrame) -> list[FeatureSetSpec]:
    """Return A2 feature sets available for the frame."""

    specs = [
        FeatureSetSpec(
            name="A2_svi_only",
            prediction_setting="forecasting_v0",
            include_calendar=False,
            include_static=False,
            include_reporting=False,
            description="SVI score only.",
        ),
        FeatureSetSpec(
            name="A2_svi_plus_calendar",
            prediction_setting="forecasting_v0",
            include_calendar=True,
            include_static=False,
            include_reporting=False,
            description="SVI score plus month-of-year controls.",
        ),
        FeatureSetSpec(
            name="A2_svi_plus_static",
            prediction_setting="forecasting_v0",
            include_calendar=True,
            include_static=True,
            include_reporting=False,
            description=(
                "SVI score plus month-of-year, population, land area, and density controls "
                "when available."
            ),
        ),
    ]

    if REPORTING_CONTROL_COL in frame.columns:
        specs.append(
            FeatureSetSpec(
                name="A2_svi_plus_reporting_retrospective",
                prediction_setting="retrospective_explanatory_v0",
                include_calendar=True,
                include_static=True,
                include_reporting=True,
                description=(
                    "SVI score plus static/calendar controls and same-month non-water 311 "
                    "reporting exposure. Retrospective only."
                ),
            )
        )

    return specs


def safe_numeric(series: pd.Series) -> pd.Series:
    """Convert a series to numeric float."""

    return pd.to_numeric(series, errors="coerce").astype(float)


def log1p_nonnegative(series: pd.Series) -> pd.Series:
    """Convert a nonnegative numeric series to log1p, clipping negative noise to zero."""

    numeric = safe_numeric(series).fillna(0).clip(lower=0)
    return np.log1p(numeric)


def raw_feature_frame(
    frame: pd.DataFrame,
    *,
    svi_spec: SviScoreSpec,
    feature_spec: FeatureSetSpec,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Build raw unstandardized features for one SVI score and feature set.

    The returned dataframe does not include the intercept. The model-building
    function adds it after train-only imputation and standardization.
    """

    features = pd.DataFrame(index=frame.index)
    raw_columns: list[str] = []

    svi_feature = f"svi__{svi_spec.source_column}"
    features[svi_feature] = safe_numeric(frame[svi_spec.score_column])
    raw_columns.append(svi_spec.score_column)

    if feature_spec.include_calendar:
        month = pd.to_numeric(frame[MONTH_COL], errors="coerce").fillna(-1).astype(int)
        # February–December dummies; January is absorbed by intercept.
        for month_value in range(2, 13):
            col = f"month_is_{month_value:02d}"
            features[col] = (month == month_value).astype(float)
        raw_columns.append(MONTH_COL)

    if feature_spec.include_static:
        seen_static_sources: set[str] = set()
        for col in STATIC_NUMERIC_CANDIDATES:
            if col not in frame.columns or col in seen_static_sources:
                continue
            feature_col = f"log1p_{col}"
            features[feature_col] = log1p_nonnegative(frame[col])
            raw_columns.append(col)
            seen_static_sources.add(col)

    if feature_spec.include_reporting:
        if REPORTING_CONTROL_COL not in frame.columns:
            raise A2BaselineError(
                f"Feature set {feature_spec.name} requires missing column {REPORTING_CONTROL_COL}."
            )
        features[f"log1p_{REPORTING_CONTROL_COL}"] = log1p_nonnegative(frame[REPORTING_CONTROL_COL])
        raw_columns.append(REPORTING_CONTROL_COL)

    return features, raw_columns


def fit_transform_features(
    train_raw: pd.DataFrame,
    full_raw: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, float], dict[str, float], dict[str, float]]:
    """
    Fit train-only median imputation and standardization, then transform train/full.

    Binary dummy features are also standardized. This keeps implementation simple
    and makes coefficients comparable inside a fitted model.
    """

    if train_raw.empty or full_raw.empty:
        raise A2BaselineError("Cannot fit transform features on empty data.")

    feature_names = list(train_raw.columns)
    medians: dict[str, float] = {}
    means: dict[str, float] = {}
    stds: dict[str, float] = {}

    train_std = pd.DataFrame(index=train_raw.index)
    full_std = pd.DataFrame(index=full_raw.index)

    for col in feature_names:
        train_col = safe_numeric(train_raw[col])
        full_col = safe_numeric(full_raw[col])

        median = float(train_col.median(skipna=True)) if train_col.notna().any() else 0.0
        train_filled = train_col.fillna(median)
        full_filled = full_col.fillna(median)

        mean = float(train_filled.mean())
        std = float(train_filled.std(ddof=0))
        if not math.isfinite(std) or std <= 1e-12:
            std = 1.0

        train_std[col] = (train_filled - mean) / std
        full_std[col] = (full_filled - mean) / std

        medians[col] = median
        means[col] = mean
        stds[col] = std

    # Add intercept as first column.
    train_matrix = np.column_stack([np.ones(len(train_std)), train_std.to_numpy(dtype=float)])
    full_matrix = np.column_stack([np.ones(len(full_std)), full_std.to_numpy(dtype=float)])
    feature_columns = ["intercept", *feature_names]

    return train_matrix, full_matrix, feature_columns, medians, means, stds


def fit_ridge_log_count_model(
    frame: pd.DataFrame,
    train: pd.DataFrame,
    *,
    svi_spec: SviScoreSpec,
    feature_spec: FeatureSetSpec,
    ridge_alpha: float = RIDGE_ALPHA_DEFAULT,
) -> tuple[FittedRidgeLogCountModel, pd.Series]:
    """Fit ridge log-count model and predict for all rows."""

    if ridge_alpha < 0:
        raise A2BaselineError("ridge_alpha must be nonnegative.")

    raw_full, raw_columns = raw_feature_frame(frame, svi_spec=svi_spec, feature_spec=feature_spec)
    raw_train = raw_full.loc[train.index].copy()

    # Column-name leakage guard. raw_columns are the source columns that informed the feature set.
    assert_no_forbidden_feature_columns(
        raw_columns,
        prediction_setting=feature_spec.prediction_setting,
    )

    X_train, X_full, feature_columns, medians, means, stds = fit_transform_features(
        raw_train,
        raw_full,
    )

    y_train_count = safe_numeric(train[TARGET_COLUMN]).clip(lower=0)
    y_train = np.log1p(y_train_count.to_numpy(dtype=float))

    penalty = np.eye(X_train.shape[1]) * float(ridge_alpha)
    penalty[0, 0] = 0.0  # Do not penalize intercept.

    xtx = X_train.T @ X_train
    xty = X_train.T @ y_train

    try:
        beta = np.linalg.solve(xtx + penalty, xty)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(xtx + penalty) @ xty

    pred_log = X_full @ beta
    pred_count = np.expm1(pred_log)
    pred_count = np.clip(pred_count, MIN_PREDICTION, None)

    model_name = f"{feature_spec.name}__{svi_spec.source_column}"
    if svi_spec.score_role != "primary_continuous_svi_score_candidate":
        model_name += "__diagnostic_svi"

    fitted = FittedRidgeLogCountModel(
        model_name=model_name,
        feature_set_name=feature_spec.name,
        prediction_setting=feature_spec.prediction_setting,
        source_svi_column=svi_spec.source_column,
        oriented_svi_score_column=svi_spec.score_column,
        score_role=svi_spec.score_role,
        feature_columns=feature_columns,
        beta=[float(value) for value in beta],
        ridge_alpha=float(ridge_alpha),
        train_target_log_mean=float(np.mean(y_train)),
        train_target_count_mean=float(y_train_count.mean()),
        feature_medians=medians,
        feature_means=means,
        feature_stds=stds,
        feature_raw_columns=list(dict.fromkeys(raw_columns)),
    )

    return fitted, pd.Series(pred_count, index=frame.index, dtype=float)


def poisson_any_probability(mu: pd.Series) -> pd.Series:
    """Convert predicted count mean to P(Y > 0) under a Poisson assumption."""

    values = safe_numeric(mu).fillna(0).clip(lower=0)
    return 1.0 - np.exp(-values)


def prediction_frame_for_model(
    frame: pd.DataFrame,
    *,
    split_scheme: str,
    fitted: FittedRidgeLogCountModel,
    predictions: pd.Series,
) -> pd.DataFrame:
    """Create standardized long prediction rows for one A2 model."""

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
        REPORTING_CONTROL_COL,
        fitted.oriented_svi_score_column,
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
    out["feature_set_name"] = fitted.feature_set_name
    out["prediction_setting"] = fitted.prediction_setting
    out["source_svi_column"] = fitted.source_svi_column
    out["oriented_svi_score_column"] = fitted.oriented_svi_score_column
    out["score_role"] = fitted.score_role
    out["ridge_alpha"] = fitted.ridge_alpha

    return out


def evaluate_one_model(
    prediction_frame: pd.DataFrame,
    *,
    config: Mapping[str, Any],
    full_frame: pd.DataFrame,
    split_scheme: str,
    fitted: FittedRidgeLogCountModel,
) -> pd.DataFrame:
    """Evaluate one A2 model."""

    context = build_run_context(
        config=config,
        frame=full_frame,
        split_scheme=split_scheme,
        prediction_setting=fitted.prediction_setting,
        model_stage=MODEL_STAGE,
        model_name=fitted.model_name,
        target_name=TARGET_COLUMN,
        target_type="count",
        feature_set_name=fitted.feature_set_name,
        dataset_version=DATASET_VERSION_DEFAULT,
    )

    return evaluate_prediction_frame(
        prediction_frame,
        context=context,
        split_scheme=split_scheme,
        observed_col=TARGET_COLUMN,
        predicted_col=PREDICTED_COL,
        binary_observed_col=BINARY_TARGET_COLUMN,
        binary_score_col=BINARY_SCORE_COL,
        ranking_score_col="predicted_score",
    )


def write_long_prediction_partitions(
    predictions_long: pd.DataFrame,
    output_dir: Path,
    *,
    split_scheme: str,
) -> dict[str, str]:
    """Write validation/test prediction files for all A2 models in long format."""

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
        "feature_set_name",
        "prediction_setting",
        "source_svi_column",
        "oriented_svi_score_column",
        "score_role",
        "ridge_alpha",
        "year",
        MONTH_COL,
        "population_total_2021",
        "land_area_km2",
        "population_density",
        "population_density_per_km2",
        REPORTING_CONTROL_COL,
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


def dataframe_to_markdown(df: pd.DataFrame, max_rows: int = 80) -> str:
    """Render dataframe to Markdown with fallback when tabulate is unavailable."""

    if df.empty:
        return "_No rows._"

    display = df.head(max_rows).copy()
    try:
        return display.to_markdown(index=False)
    except Exception:
        return "```text\n" + display.to_string(index=False) + "\n```"


def compact_metrics_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    """Create compact metrics summary for the A2 report."""

    if metrics.empty:
        return metrics

    wanted = [
        "count__mae",
        "count__rmse",
        "count__mean_poisson_deviance",
        "ranking__spearman_corr",
        "ranking__ndcg_at_100",
        "ranking__top_10pct_overlap_rate",
    ]

    out = metrics[metrics["metric_name"].isin(wanted)].copy()
    cols = [
        "split_name",
        "prediction_setting",
        "model_name",
        "feature_set_name",
        "metric_name",
        "metric_value",
        "higher_is_better",
        "n_rows",
    ]
    return out[[col for col in cols if col in out.columns]].reset_index(drop=True)


def best_metric_table(metrics: pd.DataFrame, metric_name: str = "count__mae") -> pd.DataFrame:
    """Return best model per validation/test split for a selected metric."""

    if metrics.empty:
        return metrics

    subset = metrics[metrics["metric_name"] == metric_name].copy()
    if subset.empty:
        return subset

    subset["metric_value"] = pd.to_numeric(subset["metric_value"], errors="coerce")
    subset = subset.dropna(subset=["metric_value"])
    if subset.empty:
        return subset

    rows = []
    for split_name, part in subset.groupby("split_name"):
        higher = bool(part["higher_is_better"].iloc[0])
        best = part.sort_values("metric_value", ascending=not higher).head(10)
        rows.append(best)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def render_a2_report(
    *,
    metrics: pd.DataFrame,
    coefficients: pd.DataFrame,
    score_audit: Sequence[Mapping[str, Any]],
    static_audit: Sequence[Mapping[str, Any]],
    model_summaries: Sequence[Mapping[str, Any]],
    row_counts: Mapping[str, Any],
    outputs: Mapping[str, str],
    split_scheme: str,
    generated_at: str,
    ridge_alpha: float,
    primary_only: bool,
) -> str:
    """Render A2 report."""

    compact = compact_metrics_summary(metrics)
    best_mae = best_metric_table(metrics, metric_name="count__mae")
    included_scores = [row for row in score_audit if row.get("status") == "included"]
    primary_scores = [row for row in included_scores if row.get("is_primary_recommended")]
    diagnostic_scores = [row for row in included_scores if not row.get("is_primary_recommended")]

    lines: list[str] = []
    lines.append("# A2 Calibrated SVI Predictors — Montréal 311 Water/Drainage v0\n")
    lines.append(f"Generated at: `{generated_at}`\n")
    lines.append(f"Split scheme: `{split_scheme}`\n")
    lines.append(f"Split type: `{split_type_for_scheme(split_scheme)}`\n")
    lines.append(f"Ridge alpha: `{ridge_alpha}`\n")
    lines.append(f"Primary-only mode: `{primary_only}`\n")

    lines.append("## Purpose\n")
    lines.append(
        "A2 tests whether static SVI has predictive value after simple calibration "
        "and basic controls. The model family is a ridge-regularized linear model "
        "fit on `log1p(water_drainage_count)` using training rows only. Predictions "
        "are converted back to count scale with `expm1` and clipped at zero.\n"
    )

    lines.append("## Row counts\n")
    lines.append("| Partition | Rows |")
    lines.append("|---|---:|")
    for key in ["train", "validation", "test"]:
        lines.append(f"| `{key}` | {row_counts.get(key)} |")
    lines.append("")

    lines.append("## Primary SVI scores used\n")
    if primary_scores:
        lines.append("| Source column | Oriented score column | Orientation |")
        lines.append("|---|---|---|")
        for row in primary_scores:
            lines.append(
                f"| `{row.get('source_column')}` | `{row.get('score_column')}` | `{row.get('orientation')}` |"
            )
        lines.append("")
    else:
        lines.append("_No primary continuous SVI score columns included._\n")

    lines.append("## Diagnostic SVI scores used\n")
    if diagnostic_scores:
        lines.append("| Source column | Score role | Oriented score column | Orientation |")
        lines.append("|---|---|---|---|")
        for row in diagnostic_scores:
            lines.append(
                f"| `{row.get('source_column')}` | `{row.get('score_role')}` | "
                f"`{row.get('score_column')}` | `{row.get('orientation')}` |"
            )
        lines.append("")
    else:
        lines.append("_No diagnostic SVI score columns included._\n")

    lines.append("## Model families\n")
    lines.append("| Feature set | Prediction setting | Count | Description |")
    lines.append("|---|---|---:|---|")
    summary_df = pd.DataFrame(model_summaries)
    if not summary_df.empty:
        feature_rows = (
            summary_df.groupby(["feature_set_name", "prediction_setting", "description"], as_index=False)
            .agg(model_count=("model_name", "nunique"))
        )
        for _, row in feature_rows.iterrows():
            lines.append(
                f"| `{row['feature_set_name']}` | `{row['prediction_setting']}` | "
                f"{row['model_count']} | {row['description']} |"
            )
    lines.append("")

    lines.append("## Best models by validation/test MAE\n")
    lines.append(dataframe_to_markdown(best_mae, max_rows=40))
    lines.append("")

    lines.append("## Compact metrics summary\n")
    lines.append(dataframe_to_markdown(compact, max_rows=120))
    lines.append("")

    lines.append("## Static-score audit\n")
    lines.append(dataframe_to_markdown(pd.DataFrame(static_audit)))
    lines.append("")

    lines.append("## Coefficients preview\n")
    coef_preview_cols = [
        "model_name",
        "feature_set_name",
        "prediction_setting",
        "source_svi_column",
        "score_role",
        "feature",
        "coefficient",
    ]
    coef_preview_cols = [col for col in coef_preview_cols if col in coefficients.columns]
    lines.append(dataframe_to_markdown(coefficients[coef_preview_cols], max_rows=80))
    lines.append("")

    lines.append("## Output artifacts\n")
    lines.append("| Artifact | Path |")
    lines.append("|---|---|")
    for key, value in outputs.items():
        lines.append(f"| `{key}` | `{value}` |")
    lines.append("")

    lines.append("## Leakage and interpretation notes\n")
    lines.append(
        "- All calibration coefficients, imputations, and standardization parameters are fitted on training rows only.\n"
        "- `A2_svi_plus_reporting_retrospective` uses same-month `total_311_count_non_water_drainage`; it is retrospective/explanatory only.\n"
        "- The strict forecasting A2 feature sets do not use same-month reporting controls or target-derived share/count columns.\n"
        "- Primary A2 interpretation should focus on `svi_percentile` and `svi_score_raw` models.\n"
        "- `svi_rank` and `svi_class`, when included, are diagnostic robustness checks.\n"
        "- No SoVI columns are used in Track A.\n"
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
    metrics: pd.DataFrame,
    generated_at: str,
    ridge_alpha: float,
    primary_only: bool,
) -> dict[str, Any]:
    """Build A2 metadata."""

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
            "model_name": "A2_calibrated_svi_suite",
            "split_scheme": split_scheme,
            "split_type": split_type_for_scheme(split_scheme),
            "target_name": TARGET_COLUMN,
            "target_type": "count",
            "model_family": "ridge_log_count_linear_model",
            "target_transform": "log1p",
            "inverse_transform": "expm1_clipped_at_zero",
            "ridge_alpha": float(ridge_alpha),
            "primary_only": bool(primary_only),
            "row_counts": row_counts,
            "score_audit": list(score_audit),
            "static_score_audit": list(static_audit),
            "models": list(model_summaries),
            "metric_rows": int(len(metrics)),
            "outputs": dict(outputs),
            "notes": (
                "A2 calibrates static SVI scores into count predictions using simple "
                "train-only ridge log-count linear models. Same-month reporting-control "
                "models are labeled retrospective. Primary interpretation should focus on "
                "continuous SVI scores; rank/class scores are diagnostic when included."
            ),
        }
    )


def run_a2_calibrated_svi(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    repo_root: str | Path | None = None,
    split_scheme: str = DEFAULT_SPLIT_SCHEME,
    ridge_alpha: float = RIDGE_ALPHA_DEFAULT,
    primary_only: bool = False,
) -> dict[str, Any]:
    """
    Run A2 calibrated SVI predictors and write standard artifacts.

    Returns a dictionary containing output paths, row counts, and model summaries.
    """

    require_runtime_dependencies()

    config, root, resolved_config_path, panel_path, split_path, frame = load_benchmark_frame(
        config_path=config_path,
        repo_root=repo_root,
    )

    frame = normalize_frame_for_a2(frame, split_scheme=split_scheme)
    frame, svi_specs_all, score_audit = build_svi_score_specs(frame)
    svi_specs = filter_svi_specs(svi_specs_all, primary_only=primary_only)
    static_audit = validate_static_svi_scores(frame, svi_specs)

    row_counts = split_counts(frame, split_scheme=split_scheme)
    missing_required = [part for part in ["train", "validation", "test"] if row_counts.get(part, 0) <= 0]
    if missing_required:
        raise A2BaselineError(
            f"Split scheme {split_scheme!r} missing required partitions: {missing_required}. "
            f"Counts: {row_counts}"
        )

    train = train_frame(frame, split_scheme=split_scheme)
    fset_specs = feature_set_specs(frame)

    paths = get_baseline_paths(config, root, STAGE_SLUG)
    output_dir = paths.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    all_predictions: list[pd.DataFrame] = []
    all_metrics: list[pd.DataFrame] = []
    all_coefficients: list[pd.DataFrame] = []
    model_summaries: list[dict[str, Any]] = []

    for svi_spec in svi_specs:
        for fset in fset_specs:
            fitted, pred = fit_ridge_log_count_model(
                frame,
                train,
                svi_spec=svi_spec,
                feature_spec=fset,
                ridge_alpha=ridge_alpha,
            )

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

            all_predictions.append(pred_frame)
            all_metrics.append(metrics)
            all_coefficients.append(fitted.coefficients_table())
            model_summaries.append(
                {
                    "model_name": fitted.model_name,
                    "feature_set_name": fitted.feature_set_name,
                    "prediction_setting": fitted.prediction_setting,
                    "source_svi_column": fitted.source_svi_column,
                    "oriented_svi_score_column": fitted.oriented_svi_score_column,
                    "score_role": fitted.score_role,
                    "description": fset.description,
                    "feature_columns": fitted.feature_columns,
                    "feature_raw_columns": fitted.feature_raw_columns,
                    "ridge_alpha": fitted.ridge_alpha,
                    "train_target_count_mean": fitted.train_target_count_mean,
                    "train_target_log_mean": fitted.train_target_log_mean,
                }
            )

    predictions_long = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
    metrics = pd.concat(all_metrics, ignore_index=True) if all_metrics else pd.DataFrame()
    coefficients = pd.concat(all_coefficients, ignore_index=True) if all_coefficients else pd.DataFrame()

    coefficients_path = output_dir / "coefficients.csv"
    score_audit_path = output_dir / "svi_score_audit.csv"
    static_audit_path = output_dir / "svi_static_score_audit.csv"
    feature_set_audit_path = output_dir / "feature_set_audit.csv"

    coefficients.to_csv(coefficients_path, index=False)
    pd.DataFrame(score_audit).to_csv(score_audit_path, index=False)
    pd.DataFrame(static_audit).to_csv(static_audit_path, index=False)
    pd.DataFrame(model_summaries).to_csv(feature_set_audit_path, index=False)

    written_predictions = write_long_prediction_partitions(
        predictions_long,
        output_dir,
        split_scheme=split_scheme,
    )

    generated_at = datetime.now(timezone.utc).isoformat()

    outputs = {
        "metrics": str(paths.metrics),
        "model_metadata": str(paths.model_metadata),
        "baseline_report": str(paths.baseline_report),
        "coefficients": str(coefficients_path),
        "svi_score_audit": str(score_audit_path),
        "svi_static_score_audit": str(static_audit_path),
        "feature_set_audit": str(feature_set_audit_path),
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
        metrics=metrics,
        generated_at=generated_at,
        ridge_alpha=ridge_alpha,
        primary_only=primary_only,
    )

    report = render_a2_report(
        metrics=metrics,
        coefficients=coefficients,
        score_audit=score_audit,
        static_audit=static_audit,
        model_summaries=model_summaries,
        row_counts=row_counts,
        outputs=outputs,
        split_scheme=split_scheme,
        generated_at=generated_at,
        ridge_alpha=ridge_alpha,
        primary_only=primary_only,
    )

    metrics.to_csv(paths.metrics, index=False)
    write_json(paths.model_metadata, metadata)
    write_markdown(paths.baseline_report, report)

    return {
        "status": "completed",
        "model_stage": MODEL_STAGE,
        "model_name": "A2_calibrated_svi_suite",
        "split_scheme": split_scheme,
        "outputs": outputs,
        "row_counts": row_counts,
        "svi_score_count_all": len(svi_specs_all),
        "svi_score_count_used": len(svi_specs),
        "primary_only": primary_only,
        "feature_set_count": len(fset_specs),
        "model_count": len(model_summaries),
        "metric_rows": int(len(metrics)),
        "prediction_rows": int(len(predictions_long)),
        "coefficient_rows": int(len(coefficients)),
    }


def a2_brief(result: Mapping[str, Any]) -> str:
    """Return concise A2 run summary."""

    outputs = result.get("outputs", {})
    return (
        "A2 calibrated SVI predictors completed.\n"
        f"Status: {result.get('status')}\n"
        f"Split scheme: {result.get('split_scheme')}\n"
        f"SVI score columns available: {result.get('svi_score_count_all')}\n"
        f"SVI score columns used: {result.get('svi_score_count_used')}\n"
        f"Primary only: {result.get('primary_only')}\n"
        f"Feature sets: {result.get('feature_set_count')}\n"
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
        description="Run A2 calibrated SVI predictors for Montréal 311 water/drainage."
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
    parser.add_argument(
        "--ridge-alpha",
        type=float,
        default=RIDGE_ALPHA_DEFAULT,
        help=f"Ridge penalty. Default: {RIDGE_ALPHA_DEFAULT}",
    )
    parser.add_argument(
        "--primary-only",
        action="store_true",
        help="Evaluate only primary continuous SVI scores, excluding rank/class diagnostics.",
    )

    args = parser.parse_args()

    result = run_a2_calibrated_svi(
        config_path=args.config,
        repo_root=args.repo_root,
        split_scheme=args.split_scheme,
        ridge_alpha=args.ridge_alpha,
        primary_only=args.primary_only,
    )

    print(a2_brief(result).rstrip())
    print("\nWritten outputs:")
    for label, path in result.get("outputs", {}).items():
        print(f"  {label}: {path}")


if __name__ == "__main__":
    main()


__all__ = [
    "A2BaselineError",
    "DEFAULT_SPLIT_SCHEME",
    "FittedRidgeLogCountModel",
    "FeatureSetSpec",
    "MODEL_STAGE",
    "RIDGE_ALPHA_DEFAULT",
    "STAGE_SLUG",
    "a2_brief",
    "feature_set_specs",
    "filter_svi_specs",
    "fit_ridge_log_count_model",
    "run_a2_calibrated_svi",
]
