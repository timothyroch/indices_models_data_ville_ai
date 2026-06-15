"""
A1 SVI sensitivity baseline for the Montréal 311 water/drainage benchmark.

This module extends the raw A1 SVI direct-ranking stage with methodological
sensitivity tests that are more aligned with the static tract-level nature of
SVI. We want to answer:

    Is raw SVI weak because it truly has little alignment with this
    water/drainage 311 outcome, or because the first A1 test compared a static
    tract-level score to a dynamic monthly operational target?

The methodological boundary is strict:

- SVI remains the score/prediction used for ranking.
- The observed target definition changes across variants.
- No parameters are fitted from SVI to the 311 target.
- Therefore all variants remain A1-style direct-ranking index tests.

Implemented variants
--------------------

A1a strict tract-month raw-count ranking
    Compare repeated static SVI score ``s_i`` to monthly tract target
    ``y_{i,t}``.

A1b tract-level mean burden
    Aggregate water/drainage calls to mean monthly burden per tract.

A1c tract-level total burden
    Aggregate water/drainage calls to total burden per tract.

A1d population-normalized burden
    Total water/drainage reports per 1,000 residents.

A1e reporting-normalized water share
    Water/drainage reports divided by all 311 reports, when a usable all-311
    denominator is available.

A1f reporting-baseline excess burden
    Observed water/drainage reports above the expected count implied by the
    tract's all-311 reporting volume and a train-derived water-share baseline.

A1g data-defined water/drainage surge-window burden
    Optional surge-window direct ranking. The default data-defined version
    selects high-water-burden calendar months from the training partition only,
    then evaluates held-out rows whose month-of-year is in that set. A user can
    also pass explicit month numbers. This remains a transparent 311-defined
    surge diagnostic, not an external rainfall/hazard-window test.

Recommended use
---------------

Run this as a separate benchmark block, preserving the original A1 strict
baseline as a frozen result.

Example:

    PYTHONPATH=urban_graph_benchmark/src python -m ville_hgnn.baselines.a1_svi_sensitivity \
      --split-schemes temporal spatial_block \
      --enable-surge-window \
      --overwrite
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

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

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover
    plt = None  # type: ignore[assignment]
    _MATPLOTLIB_IMPORT_ERROR = exc
else:
    _MATPLOTLIB_IMPORT_ERROR = None

from ville_hgnn.baselines.common import (
    BINARY_TARGET_COLUMN,
    DATASET_VERSION_DEFAULT,
    DEFAULT_CONFIG_PATH,
    EVAL_PARTITIONS,
    TARGET_COLUMN,
    BaselineError,
    build_run_context,
    get_baseline_paths,
    load_benchmark_frame,
    split_column_for_scheme,
    split_counts,
    split_type_for_scheme,
)
from ville_hgnn.evaluation.metrics import (
    MetricResult,
    evaluate_ranking_metrics,
    make_metric_row,
    make_metrics_dataframe,
    ndcg_at_k,
    top_fraction_k,
    top_k_overlap,
)
from ville_hgnn.utils.io import config_hash, file_hash, to_jsonable, write_json, write_markdown


STAGE_SLUG = "A1_svi_sensitivity"
MODEL_STAGE = "A1_svi_sensitivity"
DEFAULT_SPLIT_SCHEMES = ("temporal", "spatial_block")
TARGET_NAME = TARGET_COLUMN

ZONE_COL = "zone_id"
PERIOD_COL = "period_month"
POPULATION_COL_CANDIDATES = (
    "population_total_2021",
    "population_total",
    "population",
    "pop_total",
    "total_population",
)
ALL_311_COL_CANDIDATES = (
    "total_311_count_all",
    "requests_total",
    "total_311_count",
    "all_311_count",
    "all311_count",
    "total_requests",
    "request_total",
)
NON_WATER_311_COL_CANDIDATES = (
    "total_311_count_non_water_drainage",
    "non_water_drainage_311_count",
    "total_311_non_water_drainage",
)

TRACT_MONTH_K_VALUES = (10, 25, 50, 100)
TRACT_LEVEL_K_VALUES = (10, 25, 50, 100)
FRACTION_K_VALUES = (0.05, 0.10)

DEFAULT_MIN_POPULATION = 100.0
DEFAULT_MIN_ALL311 = 10.0
DEFAULT_RATE_WINSORIZE_QUANTILE = 0.99
DEFAULT_SURGE_QUANTILE = 0.75

# These are SVI metadata, quality, availability, or status columns.
# They are not vulnerability scores and must not be evaluated as A1 scores.
EXCLUDED_SVI_COLUMNS = {
    "svi_missing_count",
    "svi_quality_flag",
    "svi_reproduction_level",
    "svi_input_missing_count",
    "svi_scored",
    "svi_available_variable_count",
    "svi_missing_variable_count",
    "svi_available_ready_variable_count",
    "svi_missing_ready_variable_count",
    "svi_has_all_15_canonical_variables",
    "svi_vulnerability_label",
    "svi_color",
    "svi_color_value_0_1",
}

SCORE_PRIORITY = [
    "svi_percentile",
    "svi_score_normalized_0_1",
    "svi_score",
    "svi_score_raw",
    "svi_score_z",
    "svi_overall_percentile",
    "svi_rank",
    "svi_class",
]

PRIMARY_CONTINUOUS_SCORE_TOKENS = [
    "percentile",
    "score",
    "z",
]


class A1SensitivityError(BaselineError):
    """Raised when A1 SVI sensitivity evaluation fails."""


@dataclass(frozen=True)
class SviScoreSpec:
    """SVI score column and deterministic orientation rule."""

    source_column: str
    score_column: str
    model_name: str
    feature_set_name: str
    orientation: str
    interpretation: str
    score_role: str


@dataclass(frozen=True)
class SensitivityConfig:
    """Runtime options for A1 SVI sensitivity evaluation."""

    split_schemes: tuple[str, ...] = DEFAULT_SPLIT_SCHEMES
    min_population: float = DEFAULT_MIN_POPULATION
    min_all311: float = DEFAULT_MIN_ALL311
    winsorize_rate_quantile: float | None = DEFAULT_RATE_WINSORIZE_QUANTILE
    tract_ndcg_k: int = 50
    enable_surge_window: bool = False
    hazard_months: tuple[int, ...] | None = None
    surge_quantile: float = DEFAULT_SURGE_QUANTILE
    make_plots: bool = True
    overwrite: bool = False


@dataclass(frozen=True)
class TargetSpec:
    """A target variant evaluated against a raw SVI ranking score."""

    variant_id: str
    variant_label: str
    target_column: str
    unit: str
    target_family: str
    description: str
    requires_population: bool = False
    requires_all311: bool = False
    is_optional: bool = False


def require_runtime_dependencies() -> None:
    """Fail clearly if numpy/pandas are unavailable."""

    if pd is None:
        raise A1SensitivityError("pandas is required for A1 SVI sensitivity.") from _PANDAS_IMPORT_ERROR
    if np is None:
        raise A1SensitivityError("numpy is required for A1 SVI sensitivity.") from _NUMPY_IMPORT_ERROR


def require_plotting_dependencies() -> None:
    """Fail clearly if plotting is requested but matplotlib is unavailable."""

    if plt is None:
        raise A1SensitivityError("matplotlib is required when make_plots=True.") from _MATPLOTLIB_IMPORT_ERROR


def normalize_frame_for_sensitivity(frame: pd.DataFrame, split_schemes: Sequence[str]) -> pd.DataFrame:
    """Normalize columns required by A1 sensitivity evaluation."""

    required = [ZONE_COL, PERIOD_COL, TARGET_COLUMN]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise A1SensitivityError(f"A1 sensitivity input frame missing required columns: {missing}")

    split_cols: list[str] = []
    for scheme in split_schemes:
        col = split_column_for_scheme(scheme)
        if col not in frame.columns:
            raise A1SensitivityError(f"Split scheme {scheme!r} requires missing column {col!r}.")
        split_cols.append(col)

    out = frame.copy()
    out[ZONE_COL] = out[ZONE_COL].astype(str)

    parsed = pd.to_datetime(out[PERIOD_COL].astype(str), errors="coerce")
    if parsed.isna().any():
        bad = out.loc[parsed.isna(), PERIOD_COL].drop_duplicates().head(20).tolist()
        raise A1SensitivityError(f"Could not parse period_month values: {bad}")
    out[PERIOD_COL] = parsed.dt.to_period("M").astype(str)
    out["period_datetime"] = parsed.dt.to_period("M").dt.to_timestamp()
    out["year"] = out["period_datetime"].dt.year.astype(int)
    out["month"] = out["period_datetime"].dt.month.astype(int)

    out[TARGET_COLUMN] = pd.to_numeric(out[TARGET_COLUMN], errors="coerce")
    if out[TARGET_COLUMN].isna().any():
        n_missing = int(out[TARGET_COLUMN].isna().sum())
        raise A1SensitivityError(f"{TARGET_COLUMN} contains missing/non-numeric rows: {n_missing}")
    if (out[TARGET_COLUMN] < 0).any():
        n_negative = int((out[TARGET_COLUMN] < 0).sum())
        raise A1SensitivityError(f"{TARGET_COLUMN} contains negative rows: {n_negative}")

    if BINARY_TARGET_COLUMN not in out.columns:
        out[BINARY_TARGET_COLUMN] = (out[TARGET_COLUMN] > 0).astype(int)
    else:
        out[BINARY_TARGET_COLUMN] = (
            pd.to_numeric(out[BINARY_TARGET_COLUMN], errors="coerce").fillna(0).astype(int)
        )

    for col in split_cols:
        out[col] = out[col].astype(str)

    return out.reset_index(drop=True)


def try_numeric_score(series: pd.Series) -> pd.Series | None:
    """Convert a candidate score column to numeric if possible."""

    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() == 0:
        return None
    return numeric.astype(float)


def try_class_score(series: pd.Series) -> pd.Series | None:
    """Convert an SVI class-like column into an ordinal diagnostic score."""

    mapping = {
        "very low": 0,
        "low": 1,
        "medium low": 1,
        "moderate": 2,
        "medium": 2,
        "middle": 2,
        "medium high": 3,
        "high": 4,
        "very high": 5,
    }

    normalized = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .str.replace("-", " ", regex=False)
        .str.replace("_", " ", regex=False)
        .str.replace(r"\s+", " ", regex=True)
    )
    mapped = normalized.map(mapping)
    if mapped.notna().sum() == 0:
        return None
    return mapped.astype(float)


def candidate_svi_columns(frame: pd.DataFrame) -> list[str]:
    """Return ordered SVI score-like columns available in the panel."""

    columns = [str(col) for col in frame.columns]
    lower_to_original = {col.lower(): col for col in columns}

    ordered: list[str] = []
    for col in SCORE_PRIORITY:
        if col.lower() in lower_to_original:
            ordered.append(lower_to_original[col.lower()])

    for col in columns:
        lower = col.lower()
        if not lower.startswith("svi_"):
            continue
        if col in ordered:
            continue
        if lower in EXCLUDED_SVI_COLUMNS:
            continue
        if any(token in lower for token in ["score", "percentile", "rank", "class", "theme"]):
            ordered.append(col)

    return list(dict.fromkeys(ordered))


def infer_score_role(source_col: str, conversion: str, orientation: str) -> str:
    """Classify score role for reporting."""

    lower = source_col.lower()

    if conversion == "class_label_mapping" or "class" in lower:
        return "diagnostic_ordinal_class_score_not_primary"
    if "rank" in lower and "percentile" not in lower:
        return "diagnostic_rank_reversed_score"
    if "theme" in lower:
        return "diagnostic_theme_score"
    if any(token in lower for token in PRIMARY_CONTINUOUS_SCORE_TOKENS):
        return "primary_continuous_svi_score_candidate"
    return "diagnostic_svi_score_candidate"


def build_svi_score_specs(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[SviScoreSpec], list[dict[str, Any]]]:
    """
    Create numeric oriented SVI score columns and corresponding score specs.

    Direction is deterministic from column semantics, not fitted to the target.
    """

    out = frame.copy()
    specs: list[SviScoreSpec] = []
    audit: list[dict[str, Any]] = []

    for source_col in candidate_svi_columns(out):
        lower = source_col.lower()
        numeric = try_numeric_score(out[source_col])
        conversion = "numeric"

        if numeric is None and "class" in lower:
            numeric = try_class_score(out[source_col])
            conversion = "class_label_mapping"

        if numeric is None or numeric.notna().sum() == 0:
            audit.append(
                {
                    "source_column": source_col,
                    "status": "skipped_non_numeric_or_all_missing",
                    "non_missing": int(out[source_col].notna().sum()),
                    "conversion": None,
                    "orientation": None,
                    "score_role": None,
                }
            )
            continue

        if "rank" in lower and "percentile" not in lower:
            score_col = f"{source_col}__rank_reversed_for_vulnerability"
            score = -numeric
            orientation = "negative_rank_reversed"
            interpretation = (
                "Rank column converted to vulnerability score by multiplying by -1, "
                "under the deterministic convention that lower rank means higher vulnerability."
            )
        else:
            score_col = f"{source_col}__higher_more_vulnerable"
            score = numeric
            orientation = "positive_higher_more_vulnerable"
            interpretation = "Column used directly as a static SVI vulnerability score."

        score_role = infer_score_role(source_col, conversion=conversion, orientation=orientation)
        out[score_col] = score

        model_name = f"A1_svi_sensitivity__{source_col}"
        if orientation == "negative_rank_reversed":
            model_name += "__rank_reversed"
        if score_role == "diagnostic_ordinal_class_score_not_primary":
            model_name += "__ordinal_class_diagnostic"

        specs.append(
            SviScoreSpec(
                source_column=source_col,
                score_column=score_col,
                model_name=model_name,
                feature_set_name=source_col,
                orientation=orientation,
                interpretation=interpretation,
                score_role=score_role,
            )
        )
        audit.append(
            {
                "source_column": source_col,
                "score_column": score_col,
                "status": "included",
                "score_role": score_role,
                "is_primary_recommended": score_role == "primary_continuous_svi_score_candidate",
                "non_missing": int(score.notna().sum()),
                "missing": int(score.isna().sum()),
                "conversion": conversion,
                "orientation": orientation,
                "min": float(score.min(skipna=True)),
                "max": float(score.max(skipna=True)),
                "mean": float(score.mean(skipna=True)),
            }
        )

    if not specs:
        raise A1SensitivityError(
            "No usable SVI score-like columns found. Expected joined columns such as "
            "svi_percentile, svi_score_raw, svi_rank, or svi_class."
        )

    return out, specs, audit


def validate_static_svi_scores(frame: pd.DataFrame, specs: Sequence[SviScoreSpec]) -> list[dict[str, Any]]:
    """Audit whether each SVI score is effectively static within zone."""

    rows: list[dict[str, Any]] = []
    for spec in specs:
        nunique = frame.groupby(ZONE_COL)[spec.score_column].nunique(dropna=True)
        variable_zones = nunique[nunique > 1]
        rows.append(
            {
                "source_column": spec.source_column,
                "score_column": spec.score_column,
                "score_role": spec.score_role,
                "zones": int(nunique.shape[0]),
                "zones_with_multiple_values": int(len(variable_zones)),
                "max_unique_values_within_zone": int(nunique.max()) if len(nunique) else 0,
                "status": "warning_not_static" if len(variable_zones) else "ok_static_within_zone",
                "examples": variable_zones.head(20).index.astype(str).tolist(),
            }
        )
    return rows


def first_available_column(frame: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    """Return first available column from a candidate list."""

    lower_to_original = {str(col).lower(): str(col) for col in frame.columns}
    for candidate in candidates:
        if candidate.lower() in lower_to_original:
            return lower_to_original[candidate.lower()]
    return None


def find_population_column(frame: pd.DataFrame) -> str | None:
    """Find the population denominator column, if available."""

    return first_available_column(frame, POPULATION_COL_CANDIDATES)


def find_all311_columns(frame: pd.DataFrame) -> tuple[str | None, str | None, str]:
    """
    Find or infer the all-311 denominator source.

    Returns
    -------
    all_col:
        Direct all-311 count column, if available.
    non_water_col:
        Non-water 311 count column used to infer all-311 as target + non-water.
    method:
        Human-readable method string.
    """

    all_col = first_available_column(frame, ALL_311_COL_CANDIDATES)
    if all_col is not None:
        return all_col, None, "direct_all311_column"

    non_water_col = first_available_column(frame, NON_WATER_311_COL_CANDIDATES)
    if non_water_col is not None:
        return None, non_water_col, "target_plus_non_water311_column"

    return None, None, "unavailable"


def coerce_numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
    """Return numeric version of a frame column."""

    return pd.to_numeric(frame[column], errors="coerce")


def add_denominator_columns(
    frame: pd.DataFrame,
    *,
    min_population: float,
    min_all311: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Add standardized population and all-311 denominator columns."""

    out = frame.copy()
    population_col = find_population_column(out)
    all_col, non_water_col, all311_method = find_all311_columns(out)

    audit: dict[str, Any] = {
        "population_source_column": population_col,
        "all311_source_column": all_col,
        "non_water_311_source_column": non_water_col,
        "all311_method": all311_method,
        "min_population": min_population,
        "min_all311": min_all311,
    }

    if population_col is not None:
        out["__population__"] = coerce_numeric_column(out, population_col)
    else:
        out["__population__"] = np.nan

    if all_col is not None:
        out["__all311_count__"] = coerce_numeric_column(out, all_col)
    elif non_water_col is not None:
        out["__all311_count__"] = coerce_numeric_column(out, non_water_col) + coerce_numeric_column(out, TARGET_COLUMN)
    else:
        out["__all311_count__"] = np.nan

    if (out["__all311_count__"].dropna() < out[TARGET_COLUMN].loc[out["__all311_count__"].notna()]).any():
        audit["all311_warning"] = (
            "Some all-311 denominator rows are smaller than water/drainage target rows. "
            "Rate/share variants should be interpreted cautiously."
        )
    else:
        audit["all311_warning"] = None

    audit["population_non_missing_rows"] = int(out["__population__"].notna().sum())
    audit["all311_non_missing_rows"] = int(out["__all311_count__"].notna().sum())
    return out, audit


def score_nonmissing_frame(frame: pd.DataFrame, score_col: str, target_col: str = TARGET_COLUMN) -> pd.DataFrame:
    """Return rows with non-missing numeric score and observed target."""

    out = frame.copy()
    out[score_col] = pd.to_numeric(out[score_col], errors="coerce")
    out[target_col] = pd.to_numeric(out[target_col], errors="coerce")
    keep = out[score_col].notna() & out[target_col].notna()
    return out[keep].copy()


def safe_rate(numerator: pd.Series, denominator: pd.Series, multiplier: float = 1.0) -> pd.Series:
    """Compute ratio with nonpositive denominator mapped to NaN."""

    den = pd.to_numeric(denominator, errors="coerce")
    num = pd.to_numeric(numerator, errors="coerce")
    return pd.Series(np.where(den > 0, num / den * multiplier, np.nan), index=num.index, dtype="float64")


def winsorize_series(series: pd.Series, quantile: float | None) -> pd.Series:
    """Winsorize upper tail of a numeric series, if requested."""

    numeric = pd.to_numeric(series, errors="coerce").astype(float)
    if quantile is None:
        return numeric
    if not (0.0 < quantile <= 1.0):
        raise A1SensitivityError(f"Invalid winsorize quantile: {quantile}")
    if numeric.notna().sum() == 0:
        return numeric
    cap = float(numeric.quantile(quantile))
    return numeric.clip(upper=cap)


def partition_frame(frame: pd.DataFrame, *, split_scheme: str, partition: str) -> pd.DataFrame:
    """Return rows for one split partition."""

    split_col = split_column_for_scheme(split_scheme)
    return frame[frame[split_col].astype(str) == partition].copy()


def derive_surge_months_from_training(
    frame: pd.DataFrame,
    *,
    split_scheme: str,
    surge_quantile: float,
) -> tuple[int, ...]:
    """
    Derive hazard/surge month-of-year values from training rows only.

    The procedure aggregates training water/drainage burden by calendar
    month-of-year and keeps months whose total is at or above the requested
    quantile. This is a transparent proxy, not an external rainfall definition.
    """

    train = partition_frame(frame, split_scheme=split_scheme, partition="train")
    if train.empty:
        return tuple()
    monthly = train.groupby("month", as_index=False)[TARGET_COLUMN].sum()
    if monthly.empty:
        return tuple()
    threshold = float(monthly[TARGET_COLUMN].quantile(surge_quantile))
    months = sorted(monthly.loc[monthly[TARGET_COLUMN] >= threshold, "month"].astype(int).tolist())
    return tuple(months)


def train_reporting_water_share(
    frame: pd.DataFrame,
    *,
    split_scheme: str,
) -> float:
    """Return train-derived water/drainage share of all 311 activity.

    This baseline is used by A1f for expected water counts in validation/test
    partitions. It is not fitted to SVI and does not use held-out water/share
    outcomes to define the reporting baseline.
    """

    train = partition_frame(frame, split_scheme=split_scheme, partition="train")
    if train.empty or "__all311_count__" not in train.columns:
        return float("nan")
    water = pd.to_numeric(train[TARGET_COLUMN], errors="coerce")
    all311 = pd.to_numeric(train["__all311_count__"], errors="coerce")
    valid = water.notna() & all311.notna()
    if not bool(valid.any()):
        return float("nan")
    water_sum = float(water[valid].sum(skipna=True))
    all311_sum = float(all311[valid].sum(skipna=True))
    if all311_sum <= 0:
        return float("nan")
    return water_sum / all311_sum


def aggregate_tract_targets(
    frame: pd.DataFrame,
    *,
    split_scheme: str,
    partition: str,
    sensitivity_config: SensitivityConfig,
    hazard_months: Sequence[int] | None,
    reporting_baseline_share: float | None,
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[TargetSpec]]:
    """Aggregate one split partition to tract-level sensitivity targets."""

    part = partition_frame(frame, split_scheme=split_scheme, partition=partition)
    if part.empty:
        raise A1SensitivityError(f"No rows found for split_scheme={split_scheme!r}, partition={partition!r}.")

    agg = part.groupby(ZONE_COL, as_index=False).agg(
        water_total=(TARGET_COLUMN, "sum"),
        water_mean=(TARGET_COLUMN, "mean"),
        water_positive_months=(TARGET_COLUMN, lambda s: int((s > 0).sum())),
        n_months=(PERIOD_COL, "nunique"),
        population=("__population__", "first"),
        all311_total=("__all311_count__", "sum"),
        all311_mean=("__all311_count__", "mean"),
    )

    agg["population"] = pd.to_numeric(agg["population"], errors="coerce")
    agg["all311_total"] = pd.to_numeric(agg["all311_total"], errors="coerce")
    agg["water_total"] = pd.to_numeric(agg["water_total"], errors="coerce")
    agg["water_mean"] = pd.to_numeric(agg["water_mean"], errors="coerce")

    agg["low_population_flag"] = agg["population"].isna() | (agg["population"] < sensitivity_config.min_population)
    agg["low_all311_flag"] = agg["all311_total"].isna() | (agg["all311_total"] < sensitivity_config.min_all311)

    agg["water_per_1000_population_raw"] = safe_rate(agg["water_total"], agg["population"], multiplier=1000.0)
    agg.loc[agg["low_population_flag"], "water_per_1000_population_raw"] = np.nan
    agg["water_per_1000_population"] = winsorize_series(
        agg["water_per_1000_population_raw"],
        sensitivity_config.winsorize_rate_quantile,
    )
    agg["water_per_1000_population_winsorized_flag"] = (
        agg["water_per_1000_population_raw"].notna()
        & agg["water_per_1000_population"].notna()
        & (agg["water_per_1000_population"] < agg["water_per_1000_population_raw"])
    )

    agg["water_share_of_all_311_raw"] = safe_rate(agg["water_total"], agg["all311_total"], multiplier=1.0)
    agg.loc[agg["low_all311_flag"], "water_share_of_all_311_raw"] = np.nan
    agg["water_share_of_all_311"] = winsorize_series(
        agg["water_share_of_all_311_raw"],
        sensitivity_config.winsorize_rate_quantile,
    )
    agg["water_share_winsorized_flag"] = (
        agg["water_share_of_all_311_raw"].notna()
        & agg["water_share_of_all_311"].notna()
        & (agg["water_share_of_all_311"] < agg["water_share_of_all_311_raw"])
    )

    all311_total_sum = float(agg["all311_total"].sum(skipna=True))
    water_total_sum = float(agg["water_total"].sum(skipna=True))
    partition_water_share = water_total_sum / all311_total_sum if all311_total_sum > 0 else np.nan
    baseline_water_share = (
        float(reporting_baseline_share)
        if reporting_baseline_share is not None and np.isfinite(float(reporting_baseline_share))
        else np.nan
    )
    agg["partition_water_share_diagnostic"] = partition_water_share
    agg["train_reporting_water_share_for_expected"] = baseline_water_share
    agg["expected_water_from_reporting"] = agg["all311_total"] * baseline_water_share
    agg.loc[agg["low_all311_flag"], "expected_water_from_reporting"] = np.nan
    agg["water_excess_over_reporting_expected"] = agg["water_total"] - agg["expected_water_from_reporting"]
    agg["water_excess_ratio_over_reporting_expected_raw"] = safe_rate(
        agg["water_total"],
        agg["expected_water_from_reporting"],
        multiplier=1.0,
    )
    agg["water_excess_ratio_over_reporting_expected"] = winsorize_series(
        agg["water_excess_ratio_over_reporting_expected_raw"],
        sensitivity_config.winsorize_rate_quantile,
    )

    target_specs: list[TargetSpec] = [
        TargetSpec(
            variant_id="A1b_tract_mean_burden",
            variant_label="A1b tract-level mean burden",
            target_column="water_mean",
            unit="tract",
            target_family="raw_aggregate",
            description="Mean monthly water/drainage 311 count per tract over the evaluation partition.",
        ),
        TargetSpec(
            variant_id="A1c_tract_total_burden",
            variant_label="A1c tract-level total burden",
            target_column="water_total",
            unit="tract",
            target_family="raw_aggregate",
            description="Total water/drainage 311 count per tract over the evaluation partition.",
        ),
        TargetSpec(
            variant_id="A1d_population_normalized_burden",
            variant_label="A1d population-normalized burden",
            target_column="water_per_1000_population",
            unit="tract",
            target_family="population_normalized",
            description="Water/drainage 311 reports per 1,000 residents, with low-population tracts excluded and optional upper-tail winsorization.",
            requires_population=True,
        ),
        TargetSpec(
            variant_id="A1e_reporting_normalized_share",
            variant_label="A1e water share of all 311",
            target_column="water_share_of_all_311",
            unit="tract",
            target_family="reporting_normalized",
            description="Share of all 311 reports that are water/drainage-related, using a minimum all-311 denominator and optional upper-tail winsorization.",
            requires_all311=True,
        ),
        TargetSpec(
            variant_id="A1f_reporting_excess_difference",
            variant_label="A1f excess water burden over reporting baseline",
            target_column="water_excess_over_reporting_expected",
            unit="tract",
            target_family="reporting_excess",
            description="Observed water/drainage reports minus expected reports from tract all-311 volume and the train-derived water share.",
            requires_all311=True,
        ),
        TargetSpec(
            variant_id="A1f_reporting_excess_ratio",
            variant_label="A1f ratio over reporting baseline",
            target_column="water_excess_ratio_over_reporting_expected",
            unit="tract",
            target_family="reporting_excess",
            description="Observed water/drainage reports divided by expected reports from tract all-311 volume and the train-derived water share.",
            requires_all311=True,
        ),
    ]

    hazard_audit: list[dict[str, Any]] = []
    if hazard_months:
        hazard_part = part[part["month"].astype(int).isin([int(m) for m in hazard_months])].copy()
        hazard_agg = hazard_part.groupby(ZONE_COL, as_index=False).agg(
            hazard_water_total=(TARGET_COLUMN, "sum"),
            hazard_water_mean=(TARGET_COLUMN, "mean"),
            hazard_n_months=(PERIOD_COL, "nunique"),
        )
        agg = agg.merge(hazard_agg, on=ZONE_COL, how="left")
        agg["hazard_water_total"] = agg["hazard_water_total"].fillna(0.0)
        agg["hazard_water_mean"] = pd.to_numeric(agg["hazard_water_mean"], errors="coerce")
        agg["hazard_n_months"] = agg["hazard_n_months"].fillna(0).astype(int)
        target_specs.extend(
            [
                TargetSpec(
                    variant_id="A1g_data_defined_surge_window_mean_burden",
                    variant_label="A1g data-defined surge-window mean burden",
                    target_column="hazard_water_mean",
                    unit="tract",
                    target_family="data_defined_surge_window",
                    description="Mean monthly water/drainage burden restricted to training-defined high-water-burden month-of-year values; this is a 311 surge diagnostic, not an external rainfall/hazard test.",
                    is_optional=True,
                ),
                TargetSpec(
                    variant_id="A1g_data_defined_surge_window_total_burden",
                    variant_label="A1g data-defined surge-window total burden",
                    target_column="hazard_water_total",
                    unit="tract",
                    target_family="data_defined_surge_window",
                    description="Total water/drainage burden restricted to training-defined high-water-burden month-of-year values; this is a 311 surge diagnostic, not an external rainfall/hazard test.",
                    is_optional=True,
                ),
            ]
        )
        hazard_audit.append(
            {
                "split_scheme": split_scheme,
                "split_partition": partition,
                "hazard_months": list(hazard_months),
                "hazard_rows": int(len(hazard_part)),
                "hazard_tracts": int(hazard_part[ZONE_COL].nunique()) if not hazard_part.empty else 0,
            }
        )

    agg["split_scheme"] = split_scheme
    agg["split_partition"] = partition
    agg["partition_water_share_diagnostic"] = partition_water_share

    audit_rows = build_target_audit_rows(
        agg,
        target_specs=target_specs,
        split_scheme=split_scheme,
        partition=partition,
        extra_rows=hazard_audit,
    )
    return agg, audit_rows, target_specs


def build_target_audit_rows(
    table: pd.DataFrame,
    *,
    target_specs: Sequence[TargetSpec],
    split_scheme: str,
    partition: str,
    extra_rows: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Build audit rows for target definitions and denominator stability."""

    rows: list[dict[str, Any]] = []
    n = int(len(table))
    for spec in target_specs:
        if spec.target_column not in table.columns:
            rows.append(
                {
                    "split_scheme": split_scheme,
                    "split_partition": partition,
                    "variant_id": spec.variant_id,
                    "target_column": spec.target_column,
                    "status": "missing_target_column",
                    "n": n,
                    "non_missing": 0,
                }
            )
            continue
        values = pd.to_numeric(table[spec.target_column], errors="coerce")
        rows.append(
            {
                "split_scheme": split_scheme,
                "split_partition": partition,
                "variant_id": spec.variant_id,
                "variant_label": spec.variant_label,
                "target_column": spec.target_column,
                "unit": spec.unit,
                "target_family": spec.target_family,
                "requires_population": spec.requires_population,
                "requires_all311": spec.requires_all311,
                "status": "ok" if values.notna().sum() else "all_missing",
                "n": n,
                "non_missing": int(values.notna().sum()),
                "missing": int(values.isna().sum()),
                "zero_count": int((values.fillna(np.nan) == 0).sum()),
                "min": float(values.min(skipna=True)) if values.notna().sum() else np.nan,
                "max": float(values.max(skipna=True)) if values.notna().sum() else np.nan,
                "mean": float(values.mean(skipna=True)) if values.notna().sum() else np.nan,
                "median": float(values.median(skipna=True)) if values.notna().sum() else np.nan,
                "description": spec.description,
            }
        )

    if "low_population_flag" in table.columns:
        rows.append(
            {
                "split_scheme": split_scheme,
                "split_partition": partition,
                "variant_id": "denominator_audit_population",
                "target_column": "population",
                "status": "audit",
                "n": n,
                "low_denominator_count": int(table["low_population_flag"].sum()),
            }
        )
    if "low_all311_flag" in table.columns:
        rows.append(
            {
                "split_scheme": split_scheme,
                "split_partition": partition,
                "variant_id": "denominator_audit_all311",
                "target_column": "all311_total",
                "status": "audit",
                "n": n,
                "low_denominator_count": int(table["low_all311_flag"].sum()),
            }
        )
    for extra in extra_rows or []:
        row = dict(extra)
        row.setdefault("variant_id", "data_defined_surge_window_audit")
        row.setdefault("status", "audit")
        rows.append(row)
    return rows


def make_tract_month_target_spec() -> TargetSpec:
    """Return strict A1a tract-month target spec."""

    return TargetSpec(
        variant_id="A1a_strict_tract_month_raw_count",
        variant_label="A1a strict tract-month raw count",
        target_column=TARGET_COLUMN,
        unit="tract_month",
        target_family="raw_monthly",
        description="Monthly tract-level water/drainage 311 count. This is the original strict A1 direct-ranking test.",
    )


def evaluate_score_target_ranking(
    table: pd.DataFrame,
    *,
    config: Mapping[str, Any],
    full_frame: pd.DataFrame,
    split_scheme: str,
    partition: str,
    spec: SviScoreSpec,
    target_spec: TargetSpec,
    score_col: str,
    k_values: Sequence[int],
    fractions: Sequence[float],
) -> pd.DataFrame:
    """Evaluate ranking metrics for one SVI score and one target.

    This function intentionally computes the ranking metrics locally instead of
    relying only on ``evaluate_ranking_metrics`` because some older versions of
    the project helper return only Spearman. The A1 sensitivity block needs a
    stable, auditable metric surface: Spearman, Kendall, NDCG@K, top-K overlap,
    and top-fraction overlap.
    """

    valid = table[[target_spec.target_column, score_col]].copy()
    valid[target_spec.target_column] = pd.to_numeric(valid[target_spec.target_column], errors="coerce")
    valid[score_col] = pd.to_numeric(valid[score_col], errors="coerce")
    valid = valid.dropna()

    y_true = valid[target_spec.target_column]
    y_score = valid[score_col]
    n_valid = int(len(valid))

    context = build_run_context(
        config=config,
        frame=full_frame,
        split_scheme=split_scheme,
        prediction_setting="raw_svi_sensitivity_direct_ranking_v0",
        model_stage=MODEL_STAGE,
        model_name=spec.model_name,
        target_name=target_spec.variant_id,
        target_type=f"{target_spec.unit}_ranking",
        feature_set_name=spec.feature_set_name,
        dataset_version=DATASET_VERSION_DEFAULT,
    )

    metric_results: list[MetricResult] = []

    def _rank_corr(method: str) -> float:
        if n_valid < 2:
            return float("nan")
        y = pd.to_numeric(y_true, errors="coerce")
        s = pd.to_numeric(y_score, errors="coerce")
        if y.nunique(dropna=True) < 2 or s.nunique(dropna=True) < 2:
            return float("nan")
        try:
            return float(y.corr(s, method=method))
        except Exception:
            return float("nan")

    metric_results.append(
        MetricResult(
            metric_name="spearman_corr",
            metric_value=_rank_corr("spearman"),
            higher_is_better=True,
            n=n_valid,
            notes="Spearman rank correlation between the target and the raw SVI ranking score.",
        )
    )
    metric_results.append(
        MetricResult(
            metric_name="kendall_corr",
            metric_value=_rank_corr("kendall"),
            higher_is_better=True,
            n=n_valid,
            notes="Kendall rank correlation between the target and the raw SVI ranking score.",
        )
    )

    for raw_k in k_values:
        requested_k = int(raw_k)
        effective_k = min(requested_k, n_valid)
        if effective_k <= 0:
            ndcg_value = float("nan")
            overlap_rate = float("nan")
            overlap_note = f"Top-{requested_k} overlap undefined because no valid rows are available."
        else:
            ndcg_value = ndcg_at_k(y_true, y_score, effective_k)
            overlap = top_k_overlap(y_true, y_score, effective_k)
            overlap_rate = overlap.get("overlap_rate")
            overlap_note = (
                f"Overlap rate between top-{effective_k} observed and SVI-ranked units; "
                f"requested K was {requested_k}."
            )

        metric_results.append(
            MetricResult(
                metric_name=f"ndcg_at_{requested_k}",
                metric_value=ndcg_value,
                higher_is_better=True,
                n=n_valid,
                notes=(
                    f"NDCG@{requested_k} using {target_spec.target_column} as relevance. "
                    f"Effective K is {effective_k} when fewer than {requested_k} valid rows exist."
                ),
            )
        )
        metric_results.append(
            MetricResult(
                metric_name=f"top_{requested_k}_overlap_rate",
                metric_value=overlap_rate,
                higher_is_better=True,
                n=n_valid,
                notes=overlap_note,
            )
        )

    for frac in fractions:
        pct = int(round(float(frac) * 100))
        if n_valid <= 0:
            k_frac = 0
            overlap_rate = float("nan")
        else:
            k_frac = top_fraction_k(n_valid, float(frac))
            overlap = top_k_overlap(y_true, y_score, k_frac)
            overlap_rate = overlap.get("overlap_rate")
        metric_results.append(
            MetricResult(
                metric_name=f"top_{pct}pct_overlap_rate",
                metric_value=overlap_rate,
                higher_is_better=True,
                n=n_valid,
                notes=(
                    f"Overlap rate between top-{pct}% observed and SVI-ranked units "
                    f"(effective K={k_frac})."
                ),
            )
        )

    rows: list[dict[str, Any]] = []
    for metric in metric_results:
        metric_prefixed = MetricResult(
            metric_name=f"{target_spec.variant_id}__{metric.metric_name}",
            metric_value=metric.metric_value,
            higher_is_better=metric.higher_is_better,
            n=metric.n,
            notes=metric.notes,
        )
        row = make_metric_row(
            metric=metric_prefixed,
            **context.metric_row_kwargs(eval_partition=partition),
        )
        row.update(
            {
                "sensitivity_variant_id": target_spec.variant_id,
                "sensitivity_variant_label": target_spec.variant_label,
                "sensitivity_unit": target_spec.unit,
                "target_family": target_spec.target_family,
                "target_column": target_spec.target_column,
                "source_svi_column": spec.source_column,
                "oriented_score_column": spec.score_column,
                "score_role": spec.score_role,
                "score_orientation": spec.orientation,
                "n_valid_for_variant": n_valid,
            }
        )
        rows.append(row)

    out = make_metrics_dataframe(rows)

    # Some versions of make_metrics_dataframe normalize the metric rows and may
    # drop custom columns. Reattach the A1-sensitivity metadata so that compact
    # summaries and reports can group correctly.
    metadata = {
        "sensitivity_variant_id": target_spec.variant_id,
        "sensitivity_variant_label": target_spec.variant_label,
        "sensitivity_unit": target_spec.unit,
        "target_family": target_spec.target_family,
        "target_column": target_spec.target_column,
        "source_svi_column": spec.source_column,
        "oriented_score_column": spec.score_column,
        "score_role": spec.score_role,
        "score_orientation": spec.orientation,
        "n_valid_for_variant": n_valid,
    }
    for key, value in metadata.items():
        out[key] = value

    return out

def topk_diagnostics(
    table: pd.DataFrame,
    *,
    split_scheme: str,
    partition: str,
    spec: SviScoreSpec,
    target_spec: TargetSpec,
    score_col: str,
    k_values: Sequence[int],
    fractions: Sequence[float],
) -> list[dict[str, Any]]:
    """Compute explicit top-K overlap diagnostics for one target variant."""

    valid = table[[ZONE_COL, target_spec.target_column, score_col]].copy()
    valid[target_spec.target_column] = pd.to_numeric(valid[target_spec.target_column], errors="coerce")
    valid[score_col] = pd.to_numeric(valid[score_col], errors="coerce")
    valid = valid.dropna()
    n = len(valid)
    if n == 0:
        return []

    k_specs: list[tuple[str, int]] = []
    for k in k_values:
        k_specs.append((f"top_{min(int(k), n)}", min(int(k), n)))
    for frac in fractions:
        k_specs.append((f"top_{int(round(frac * 100))}pct", top_fraction_k(n, float(frac))))

    rows: list[dict[str, Any]] = []
    for label, k in k_specs:
        overlap = top_k_overlap(valid[target_spec.target_column], valid[score_col], k)
        ndcg = ndcg_at_k(valid[target_spec.target_column], valid[score_col], k)
        observed_top_ids = set(
            valid.nlargest(k, target_spec.target_column, keep="all")[ZONE_COL].astype(str).head(k).tolist()
        )
        svi_top_ids = set(valid.nlargest(k, score_col, keep="all")[ZONE_COL].astype(str).head(k).tolist())
        rows.append(
            {
                "split_scheme": split_scheme,
                "split_partition": partition,
                "model_name": spec.model_name,
                "feature_set_name": spec.feature_set_name,
                "score_role": spec.score_role,
                "source_svi_column": spec.source_column,
                "oriented_score_column": spec.score_column,
                "variant_id": target_spec.variant_id,
                "variant_label": target_spec.variant_label,
                "target_column": target_spec.target_column,
                "unit": target_spec.unit,
                "target_family": target_spec.target_family,
                "k_label": label,
                "k": k,
                "n": n,
                "overlap": overlap.get("overlap"),
                "overlap_rate": overlap.get("overlap_rate"),
                "jaccard": overlap.get("jaccard"),
                "ndcg": ndcg,
                "observed_top_zone_ids": ";".join(sorted(observed_top_ids)),
                "svi_top_zone_ids": ";".join(sorted(svi_top_ids)),
                "intersection_zone_ids": ";".join(sorted(observed_top_ids & svi_top_ids)),
            }
        )
    return rows


def attach_static_scores_to_tract_targets(
    tract_targets: pd.DataFrame,
    frame: pd.DataFrame,
    specs: Sequence[SviScoreSpec],
) -> pd.DataFrame:
    """Attach one static SVI score value per tract for every score spec."""

    out = tract_targets.copy()
    static_cols = [ZONE_COL] + [spec.score_column for spec in specs]
    static = frame[static_cols].copy()
    static = static.groupby(ZONE_COL, as_index=False).first()
    out = out.merge(static, on=ZONE_COL, how="left", validate="one_to_one")
    return out


def compact_sensitivity_table(metrics: pd.DataFrame, preferred_score_role: str = "primary_continuous_svi_score_candidate") -> pd.DataFrame:
    """Return one compact table with key metrics by variant."""

    if metrics.empty:
        return metrics.copy()

    key_patterns = [
        "spearman_corr",
        "kendall_corr",
        "ndcg_at_10",
        "ndcg_at_25",
        "ndcg_at_50",
        "ndcg_at_100",
        "top_10_overlap_rate",
        "top_25_overlap_rate",
        "top_50_overlap_rate",
        "top_100_overlap_rate",
        "top_5pct_overlap_rate",
        "top_10pct_overlap_rate",
    ]

    mask = metrics["metric_name"].astype(str).apply(lambda x: any(p in x for p in key_patterns))
    sub = metrics[mask].copy()
    if sub.empty:
        return pd.DataFrame()

    # Prefer continuous primary score rows, but keep all score rows in metrics.csv.
    if "score_role" in sub.columns:
        primary = sub[sub["score_role"].astype(str) == preferred_score_role].copy()
    else:
        primary = sub.copy()



    if primary.empty:
        primary = sub.copy()

    id_cols = [
        "split_name",
        "eval_partition",
        "sensitivity_variant_id",
        "sensitivity_variant_label",
        "sensitivity_unit",
        "target_family",
        "source_svi_column",
        "score_role",
        "n_valid_for_variant",
    ]
    id_cols = [col for col in id_cols if col in primary.columns]

    rows: list[dict[str, Any]] = []
    group_cols = [
        col
        for col in [
            "split_name",
            "eval_partition",
            "sensitivity_variant_id",
            "sensitivity_variant_label",
            "sensitivity_unit",
            "target_family",
            "source_svi_column",
            "score_role",
        ]
        if col in primary.columns
    ]

    for _, group in primary.groupby(group_cols, dropna=False):
        row = {col: group[col].iloc[0] for col in id_cols if col in group.columns}
        row["n_metric_rows"] = int(len(group))


        for _, metric_row in group.iterrows():
            name = str(metric_row.get("metric_name"))
            value = metric_row.get("metric_value")

            # Metric names look like:
            # A1e_reporting_normalized_share__ndcg_at_100
            # We only want the part after the final "__".
            short_name = name.split("__")[-1]

            if short_name == "spearman_corr":
                row["spearman_corr"] = value
            elif short_name == "kendall_corr":
                row["kendall_corr"] = value
            elif short_name == "ndcg_at_10":
                row["ndcg_at_10"] = value
            elif short_name == "ndcg_at_25":
                row["ndcg_at_25"] = value
            elif short_name == "ndcg_at_50":
                row["ndcg_at_50"] = value
            elif short_name == "ndcg_at_100":
                row["ndcg_at_100"] = value
            elif short_name == "top_10_overlap_rate":
                row["top_10_overlap_rate"] = value
            elif short_name == "top_25_overlap_rate":
                row["top_25_overlap_rate"] = value
            elif short_name == "top_50_overlap_rate":
                row["top_50_overlap_rate"] = value
            elif short_name == "top_100_overlap_rate":
                row["top_100_overlap_rate"] = value
            elif short_name == "top_5pct_overlap_rate":
                row["top_5pct_overlap_rate"] = value
            elif short_name == "top_10pct_overlap_rate":
                row["top_10pct_overlap_rate"] = value



        rows.append(row)




    out = pd.DataFrame(rows)
    sort_cols = [col for col in ["split_name", "eval_partition", "sensitivity_variant_id", "source_svi_column"] if col in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols).reset_index(drop=True)
    return out


def dataframe_to_markdown(df: pd.DataFrame, max_rows: int = 80) -> str:
    """Render dataframe to Markdown with fallback when tabulate is unavailable."""

    if df.empty:
        return "_No rows._"

    display = df.head(max_rows).copy()
    try:
        return display.to_markdown(index=False)
    except Exception:
        return "```text\n" + display.to_string(index=False) + "\n```"


def plot_metric_bars(summary: pd.DataFrame, output_dir: Path) -> list[str]:
    """Write compact bar plots for key sensitivity metrics."""

    if summary.empty:
        return []
    require_plotting_dependencies()

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    written: list[str] = []
    metrics = [
        ("spearman_corr", "SVI sensitivity Spearman correlation"),
        ("ndcg_at_10", "SVI sensitivity NDCG@10"),
        ("ndcg_at_25", "SVI sensitivity NDCG@25"),
        ("ndcg_at_50", "SVI sensitivity NDCG@50"),
        ("ndcg_at_100", "SVI sensitivity NDCG@100"),
        ("top_10pct_overlap_rate", "SVI sensitivity top-10% overlap"),
    ]

    for metric_col, title in metrics:
        if metric_col not in summary.columns:
            continue
        data = summary.copy()
        data[metric_col] = pd.to_numeric(data[metric_col], errors="coerce")
        data = data.dropna(subset=[metric_col])
        if data.empty:
            continue
        # Keep figure readable: primary score, validation/test rows, top variants.
        if "score_role" in data.columns:
            primary = data[data["score_role"] == "primary_continuous_svi_score_candidate"]
            if not primary.empty:
                data = primary
        label_cols = [col for col in ["split_name", "eval_partition", "sensitivity_variant_id"] if col in data.columns]
        data["label"] = data[label_cols].astype(str).agg(" | ".join, axis=1)
        data = data.sort_values(metric_col, ascending=True).tail(30)

        fig_height = max(5.0, 0.32 * len(data) + 1.5)
        fig, ax = plt.subplots(figsize=(12, fig_height))
        ax.barh(data["label"], data[metric_col])
        ax.set_title(title)
        ax.set_xlabel(metric_col)
        ax.set_ylabel("")
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        path = plot_dir / f"{metric_col}.png"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        written.append(str(path))
    return written


def plot_primary_scatter_examples(
    tract_targets: pd.DataFrame,
    specs: Sequence[SviScoreSpec],
    output_dir: Path,
) -> list[str]:
    """Write a few scatter diagnostics for the primary SVI score."""

    primary_specs = [spec for spec in specs if spec.score_role == "primary_continuous_svi_score_candidate"]
    if not primary_specs:
        return []
    require_plotting_dependencies()

    spec = primary_specs[0]
    if spec.score_column not in tract_targets.columns:
        return []

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    targets = [
        "water_mean",
        "water_total",
        "water_per_1000_population",
        "water_share_of_all_311",
        "water_excess_over_reporting_expected",
    ]
    written: list[str] = []
    for target in targets:
        if target not in tract_targets.columns:
            continue
        data = tract_targets[["split_scheme", "split_partition", spec.score_column, target]].copy()
        data[spec.score_column] = pd.to_numeric(data[spec.score_column], errors="coerce")
        data[target] = pd.to_numeric(data[target], errors="coerce")
        data = data.dropna()
        if data.empty:
            continue

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(data[spec.score_column], data[target], alpha=0.45, s=18)
        ax.set_title(f"SVI vs {target}")
        ax.set_xlabel(spec.source_column)
        ax.set_ylabel(target)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        path = plot_dir / f"scatter__{spec.source_column}__vs__{target}.png"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        written.append(str(path))
    return written


def render_report(
    *,
    metrics: pd.DataFrame,
    sensitivity_summary: pd.DataFrame,
    score_audit: Sequence[Mapping[str, Any]],
    static_audit: Sequence[Mapping[str, Any]],
    target_audit: pd.DataFrame,
    denominator_audit: Mapping[str, Any],
    row_counts_by_scheme: Mapping[str, Mapping[str, Any]],
    outputs: Mapping[str, str],
    sensitivity_config: SensitivityConfig,
    generated_at: str,
) -> str:
    """Render Markdown report for A1 sensitivity analysis."""

    included = [row for row in score_audit if row.get("status") == "included"]
    primary = [row for row in included if row.get("is_primary_recommended")]


    compact_cols = [
        "split_name",
        "eval_partition",
        "sensitivity_variant_id",
        "source_svi_column",
        "spearman_corr",
        "kendall_corr",
        "ndcg_at_10",
        "ndcg_at_25",
        "ndcg_at_50",
        "ndcg_at_100",
        "top_10_overlap_rate",
        "top_25_overlap_rate",
        "top_50_overlap_rate",
        "top_100_overlap_rate",
        "top_5pct_overlap_rate",
        "top_10pct_overlap_rate",
        "n_valid_for_variant",
    ]


    compact = sensitivity_summary[[col for col in compact_cols if col in sensitivity_summary.columns]].copy()

    lines: list[str] = []
    lines.append("# A1 SVI Sensitivity Analysis — Montréal 311 Water/Drainage v0\n")
    lines.append(f"Generated at: `{generated_at}`\n")

    lines.append("## Purpose\n")
    lines.append(
        "This block expands the raw A1 SVI benchmark with non-fitted sensitivity variants. "
        "The primary A1 test compares static tract-level SVI to monthly tract-level water/drainage 311 burden. "
        "That strict tract-month test is useful, but harsh: SVI is static and social-vulnerability-oriented, while 311 burden is dynamic, hazard-dependent, and affected by reporting behavior. "
        "The variants below test whether the weak strict A1 result is robust to tract-level, population-normalized, and reporting-normalized burden definitions.\n"
    )
    lines.append(
        "**Methodological boundary:** no regression, calibration, or ML model is fitted from SVI to the 311 target. "
        "SVI remains the direct ranking score in every variant, so these remain A1-style index baselines rather than A2 calibrated SVI models.\n"
    )

    lines.append("## Sensitivity variants\n")
    lines.append("| Variant | Unit | Target definition | Fitted model? |")
    lines.append("|---|---|---|---|")
    variant_rows = [
        ("A1a", "tract-month", "monthly raw water/drainage count", "no"),
        ("A1b", "tract", "mean monthly water/drainage count", "no"),
        ("A1c", "tract", "total water/drainage count", "no"),
        ("A1d", "tract", "water/drainage reports per 1,000 residents", "no"),
        ("A1e", "tract", "water/drainage reports as a share of all 311 reports", "no"),
        ("A1f", "tract", "excess water/drainage burden over the all-311 reporting baseline", "no"),
    ]
    if sensitivity_config.enable_surge_window or sensitivity_config.hazard_months:
        variant_rows.append(("A1g", "tract", "data-defined water/drainage surge-window burden", "no"))
    for row in variant_rows:
        lines.append(f"| `{row[0]}` | {row[1]} | {row[2]} | {row[3]} |")
    lines.append("")

    lines.append("## Row counts by split scheme\n")
    lines.append("| Split scheme | Train | Validation | Test |")
    lines.append("|---|---:|---:|---:|")
    for scheme, counts in row_counts_by_scheme.items():
        lines.append(
            f"| `{scheme}` | {counts.get('train')} | {counts.get('validation')} | {counts.get('test')} |"
        )
    lines.append("")

    lines.append("## Primary SVI score columns\n")
    if primary:
        lines.append("| Source column | Oriented score column | Non-missing |")
        lines.append("|---|---|---:|")
        for row in primary:
            lines.append(
                f"| `{row.get('source_column')}` | `{row.get('score_column')}` | {row.get('non_missing')} |"
            )
        lines.append("")
    else:
        lines.append("_No primary continuous SVI score/percentile column detected._\n")

    lines.append("## Denominator audit\n")
    lines.append("| Field | Value |")
    lines.append("|---|---|")
    for key, value in denominator_audit.items():
        lines.append(f"| `{key}` | `{value}` |")
    lines.append("")

    lines.append("## Compact sensitivity metrics\n")
    lines.append(dataframe_to_markdown(compact, max_rows=160))
    lines.append("")

    lines.append("## Target audit\n")
    audit_display_cols = [
        "split_scheme",
        "split_partition",
        "variant_id",
        "target_column",
        "status",
        "non_missing",
        "missing",
        "min",
        "max",
        "mean",
    ]
    target_display = target_audit[[col for col in audit_display_cols if col in target_audit.columns]].copy()
    lines.append(dataframe_to_markdown(target_display, max_rows=120))
    lines.append("")

    lines.append("## Static-score audit\n")
    lines.append(dataframe_to_markdown(pd.DataFrame(static_audit), max_rows=80))
    lines.append("")

    lines.append("## Output artifacts\n")
    lines.append("| Artifact | Path |")
    lines.append("|---|---|")
    for key, value in outputs.items():
        lines.append(f"| `{key}` | `{value}` |")
    lines.append("")

    lines.append("## Interpretation guide\n")
    lines.append(
        "- If A1a is weak but A1b/A1c improve, SVI may contain static spatial burden signal while remaining weak for monthly operational prioritization.\n"
        "- If A1d improves, raw call volume may be partly dominated by population size rather than relative social burden.\n"
        "- If A1e/A1f improve, raw water/drainage call volume may be partly dominated by general 311 reporting intensity.\n"
        "- If all variants remain weak, then raw SVI is weakly aligned with this specific Montréal water/drainage 311 target even under more SVI-native target definitions.\n"
        "- A1 sensitivity results should not be read as invalidating SVI. The target is reported municipal service burden, not objective flood occurrence or disaster impact.\n"
    )

    return "\n".join(lines)


def build_metadata(
    *,
    config: Mapping[str, Any],
    config_path: Path,
    panel_path: Path,
    split_path: Path,
    outputs: Mapping[str, str],
    sensitivity_config: SensitivityConfig,
    row_counts_by_scheme: Mapping[str, Mapping[str, Any]],
    score_audit: Sequence[Mapping[str, Any]],
    static_audit: Sequence[Mapping[str, Any]],
    target_audit: pd.DataFrame,
    denominator_audit: Mapping[str, Any],
    metrics: pd.DataFrame,
    generated_at: str,
) -> dict[str, Any]:
    """Build metadata for A1 sensitivity."""

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
            "model_name": "A1_svi_sensitivity_suite",
            "target_name": TARGET_COLUMN,
            "target_type": "direct_ranking_sensitivity",
            "prediction_setting": "raw_svi_sensitivity_direct_ranking_v0",
            "sensitivity_config": sensitivity_config,
            "row_counts_by_scheme": row_counts_by_scheme,
            "score_audit": list(score_audit),
            "static_score_audit": list(static_audit),
            "target_audit_rows": int(len(target_audit)),
            "denominator_audit": dict(denominator_audit),
            "metric_rows": int(len(metrics)),
            "outputs": dict(outputs),
            "notes": (
                "A1 sensitivity evaluates raw SVI as a direct ranking signal under alternative "
                "target definitions. No parameters are fitted from SVI to the 311 target. "
                "These variants therefore remain A1-style index baselines, not A2 calibrated models."
            ),
        }
    )


def run_a1_svi_sensitivity(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    repo_root: str | Path | None = None,
    sensitivity_config: SensitivityConfig | None = None,
) -> dict[str, Any]:
    """Run A1 SVI sensitivity analysis and write standard artifacts."""

    require_runtime_dependencies()
    sensitivity_config = sensitivity_config or SensitivityConfig()
    if sensitivity_config.make_plots:
        require_plotting_dependencies()

    config, root, resolved_config_path, panel_path, split_path, frame = load_benchmark_frame(
        config_path=config_path,
        repo_root=repo_root,
    )

    frame = normalize_frame_for_sensitivity(frame, split_schemes=sensitivity_config.split_schemes)
    frame, denominator_audit = add_denominator_columns(
        frame,
        min_population=sensitivity_config.min_population,
        min_all311=sensitivity_config.min_all311,
    )
    frame, specs, score_audit = build_svi_score_specs(frame)
    static_audit = validate_static_svi_scores(frame, specs)

    paths = get_baseline_paths(config, root, STAGE_SLUG)
    output_dir = paths.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not sensitivity_config.overwrite:
        existing = [paths.metrics, paths.baseline_report, paths.model_metadata]
        if any(path.exists() for path in existing):
            raise A1SensitivityError(
                f"Output files already exist in {output_dir}. Re-run with --overwrite to replace them."
            )

    row_counts_by_scheme: dict[str, Mapping[str, Any]] = {}
    all_metric_frames: list[pd.DataFrame] = []
    all_topk_rows: list[dict[str, Any]] = []
    all_tract_targets: list[pd.DataFrame] = []
    all_target_audit_rows: list[dict[str, Any]] = []
    surge_months_by_scheme: dict[str, tuple[int, ...]] = {}

    tract_month_target = make_tract_month_target_spec()

    for split_scheme in sensitivity_config.split_schemes:
        row_counts_by_scheme[split_scheme] = split_counts(frame, split_scheme=split_scheme)
        missing_required = [
            part for part in ["train", "validation", "test"] if row_counts_by_scheme[split_scheme].get(part, 0) <= 0
        ]
        if missing_required:
            raise A1SensitivityError(
                f"Split scheme {split_scheme!r} missing required partitions: {missing_required}. "
                f"Counts: {row_counts_by_scheme[split_scheme]}"
            )

        if sensitivity_config.hazard_months:
            hazard_months = tuple(sorted({int(m) for m in sensitivity_config.hazard_months}))
        elif sensitivity_config.enable_surge_window:
            hazard_months = derive_surge_months_from_training(
                frame,
                split_scheme=split_scheme,
                surge_quantile=sensitivity_config.surge_quantile,
            )
        else:
            hazard_months = tuple()
        surge_months_by_scheme[split_scheme] = hazard_months

        reporting_baseline_share = train_reporting_water_share(frame, split_scheme=split_scheme)

        for partition in EVAL_PARTITIONS:
            # A1a: strict tract-month monthly raw-count ranking.
            tm_part = partition_frame(frame, split_scheme=split_scheme, partition=partition)
            for spec in specs:
                tm_scored = score_nonmissing_frame(tm_part, spec.score_column, TARGET_COLUMN)
                all_metric_frames.append(
                    evaluate_score_target_ranking(
                        tm_scored,
                        config=config,
                        full_frame=frame,
                        split_scheme=split_scheme,
                        partition=partition,
                        spec=spec,
                        target_spec=tract_month_target,
                        score_col=spec.score_column,
                        k_values=TRACT_MONTH_K_VALUES,
                        fractions=FRACTION_K_VALUES,
                    )
                )

            # A1b-A1g: tract-level target variants. Targets are computed once
            # per split/partition and then evaluated against every SVI score.
            tract_targets, target_audit_rows, target_specs = aggregate_tract_targets(
                frame,
                split_scheme=split_scheme,
                partition=partition,
                sensitivity_config=sensitivity_config,
                hazard_months=hazard_months,
                reporting_baseline_share=reporting_baseline_share,
            )
            tract_targets = attach_static_scores_to_tract_targets(tract_targets, frame, specs)
            all_tract_targets.append(tract_targets)
            all_target_audit_rows.extend(target_audit_rows)

            for spec in specs:
                for target_spec in target_specs:
                    if target_spec.requires_population and tract_targets["population"].notna().sum() == 0:
                        continue
                    if target_spec.requires_all311 and tract_targets["all311_total"].notna().sum() == 0:
                        continue
                    all_metric_frames.append(
                        evaluate_score_target_ranking(
                            tract_targets,
                            config=config,
                            full_frame=frame,
                            split_scheme=split_scheme,
                            partition=partition,
                            spec=spec,
                            target_spec=target_spec,
                            score_col=spec.score_column,
                            k_values=TRACT_LEVEL_K_VALUES,
                            fractions=FRACTION_K_VALUES,
                        )
                    )
                    all_topk_rows.extend(
                        topk_diagnostics(
                            tract_targets,
                            split_scheme=split_scheme,
                            partition=partition,
                            spec=spec,
                            target_spec=target_spec,
                            score_col=spec.score_column,
                            k_values=TRACT_LEVEL_K_VALUES,
                            fractions=FRACTION_K_VALUES,
                        )
                    )

    metrics = pd.concat(all_metric_frames, ignore_index=True) if all_metric_frames else make_metrics_dataframe([])
    topk = pd.DataFrame(all_topk_rows)
    tract_targets_all = pd.concat(all_tract_targets, ignore_index=True) if all_tract_targets else pd.DataFrame()
    target_audit = pd.DataFrame(all_target_audit_rows)
    sensitivity_summary = compact_sensitivity_table(metrics)

    tract_targets_path = output_dir / "tract_level_targets.csv"
    sensitivity_table_path = output_dir / "sensitivity_table.csv"
    topk_path = output_dir / "top_decile_overlap_by_variant.csv"
    target_audit_path = output_dir / "target_audit.csv"
    score_audit_path = output_dir / "svi_score_audit.csv"
    static_audit_path = output_dir / "svi_static_score_audit.csv"

    tract_targets_all.to_csv(tract_targets_path, index=False)
    sensitivity_summary.to_csv(sensitivity_table_path, index=False)
    topk.to_csv(topk_path, index=False)
    target_audit.to_csv(target_audit_path, index=False)
    pd.DataFrame(score_audit).to_csv(score_audit_path, index=False)
    pd.DataFrame(static_audit).to_csv(static_audit_path, index=False)

    plot_paths: list[str] = []
    if sensitivity_config.make_plots:
        plot_paths.extend(plot_metric_bars(sensitivity_summary, output_dir))
        plot_paths.extend(plot_primary_scatter_examples(tract_targets_all, specs, output_dir))

    generated_at = datetime.now(timezone.utc).isoformat()

    outputs: dict[str, str] = {
        "metrics": str(paths.metrics),
        "model_metadata": str(paths.model_metadata),
        "baseline_report": str(paths.baseline_report),
        "sensitivity_table": str(sensitivity_table_path),
        "tract_level_targets": str(tract_targets_path),
        "top_decile_overlap_by_variant": str(topk_path),
        "target_audit": str(target_audit_path),
        "svi_score_audit": str(score_audit_path),
        "svi_static_score_audit": str(static_audit_path),
    }
    for idx, plot_path in enumerate(plot_paths, start=1):
        outputs[f"plot_{idx:02d}"] = plot_path

    metadata = build_metadata(
        config=config,
        config_path=resolved_config_path,
        panel_path=panel_path,
        split_path=split_path,
        outputs=outputs,
        sensitivity_config=sensitivity_config,
        row_counts_by_scheme=row_counts_by_scheme,
        score_audit=score_audit,
        static_audit=static_audit,
        target_audit=target_audit,
        denominator_audit={
            **denominator_audit,
            "surge_months_by_scheme": surge_months_by_scheme,
            "a1f_reporting_baseline": "train-derived water share of all 311 activity",
        },
        metrics=metrics,
        generated_at=generated_at,
    )

    report = render_report(
        metrics=metrics,
        sensitivity_summary=sensitivity_summary,
        score_audit=score_audit,
        static_audit=static_audit,
        target_audit=target_audit,
        denominator_audit={
            **denominator_audit,
            "surge_months_by_scheme": surge_months_by_scheme,
            "a1f_reporting_baseline": "train-derived water share of all 311 activity",
        },
        row_counts_by_scheme=row_counts_by_scheme,
        outputs=outputs,
        sensitivity_config=sensitivity_config,
        generated_at=generated_at,
    )

    metrics.to_csv(paths.metrics, index=False)
    write_json(paths.model_metadata, metadata)
    write_markdown(paths.baseline_report, report)

    return {
        "status": "completed",
        "model_stage": MODEL_STAGE,
        "model_name": "A1_svi_sensitivity_suite",
        "split_schemes": list(sensitivity_config.split_schemes),
        "outputs": outputs,
        "row_counts_by_scheme": row_counts_by_scheme,
        "svi_score_count": len(specs),
        "svi_score_columns": [spec.source_column for spec in specs],
        "svi_primary_score_columns": [
            spec.source_column for spec in specs if spec.score_role == "primary_continuous_svi_score_candidate"
        ],
        "metric_rows": int(len(metrics)),
        "sensitivity_rows": int(len(sensitivity_summary)),
        "tract_target_rows": int(len(tract_targets_all)),
        "topk_rows": int(len(topk)),
        "plot_count": len(plot_paths),
        "surge_months_by_scheme": {key: list(value) for key, value in surge_months_by_scheme.items()},
    }


def a1_sensitivity_brief(result: Mapping[str, Any]) -> str:
    """Return concise A1 sensitivity run summary."""

    outputs = result.get("outputs", {})
    return (
        "A1 SVI sensitivity analysis completed.\n"
        f"Status: {result.get('status')}\n"
        f"Split schemes: {result.get('split_schemes')}\n"
        f"SVI score columns: {result.get('svi_score_count')}\n"
        f"Primary SVI columns: {result.get('svi_primary_score_columns')}\n"
        f"Metric rows: {result.get('metric_rows')}\n"
        f"Sensitivity summary rows: {result.get('sensitivity_rows')}\n"
        f"Tract target rows: {result.get('tract_target_rows')}\n"
        f"Top-K rows: {result.get('topk_rows')}\n"
        f"Plot count: {result.get('plot_count')}\n"
        f"Surge months by scheme: {result.get('surge_months_by_scheme')}\n"
        f"Metrics: {outputs.get('metrics')}\n"
        f"Sensitivity table: {outputs.get('sensitivity_table')}\n"
        f"Report: {outputs.get('baseline_report')}\n"
    )


def parse_months(values: Sequence[str] | None) -> tuple[int, ...] | None:
    """Parse optional explicit hazard month numbers."""

    if not values:
        return None
    months = sorted({int(value) for value in values})
    invalid = [month for month in months if month < 1 or month > 12]
    if invalid:
        raise A1SensitivityError(f"Invalid hazard month values: {invalid}")
    return tuple(months)


def main() -> None:
    """CLI entry point for direct module execution."""

    import argparse

    parser = argparse.ArgumentParser(
        description="Run A1 SVI sensitivity analysis for Montréal 311 water/drainage."
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
        "--split-schemes",
        nargs="+",
        default=list(DEFAULT_SPLIT_SCHEMES),
        choices=sorted(["temporal", "random_debug", "spatial_block"]),
        help="Split schemes to evaluate. Default: temporal spatial_block.",
    )
    parser.add_argument(
        "--min-population",
        type=float,
        default=DEFAULT_MIN_POPULATION,
        help="Minimum population denominator for per-capita rate variants.",
    )
    parser.add_argument(
        "--min-all311",
        type=float,
        default=DEFAULT_MIN_ALL311,
        help="Minimum all-311 denominator for reporting-normalized variants.",
    )
    parser.add_argument(
        "--winsorize-rate-quantile",
        type=float,
        default=DEFAULT_RATE_WINSORIZE_QUANTILE,
        help="Upper-tail winsorization quantile for unstable rate/ratio variants. Use 1.0 to effectively disable.",
    )
    parser.add_argument(
        "--tract-ndcg-k",
        type=int,
        default=50,
        help="Primary tract-level NDCG K for report emphasis. Metrics also include fixed K values.",
    )
    parser.add_argument(
        "--enable-surge-window",
        action="store_true",
        help="Enable data-defined A1g water/drainage surge-window variants using training-only high-burden month-of-year selection.",
    )
    parser.add_argument(
        "--hazard-months",
        nargs="*",
        default=None,
        help="Explicit month numbers 1-12. Overrides data-defined surge months when provided; use external rainfall/hazard months only if available.",
    )
    parser.add_argument(
        "--surge-quantile",
        type=float,
        default=DEFAULT_SURGE_QUANTILE,
        help="Training monthly burden quantile used to define surge months when --enable-surge-window is set.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip PNG plot generation.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing A1_svi_sensitivity output files.",
    )

    args = parser.parse_args()

    winsorize = args.winsorize_rate_quantile
    if winsorize is not None and winsorize >= 1.0:
        winsorize = None

    sensitivity_config = SensitivityConfig(
        split_schemes=tuple(args.split_schemes),
        min_population=float(args.min_population),
        min_all311=float(args.min_all311),
        winsorize_rate_quantile=winsorize,
        tract_ndcg_k=int(args.tract_ndcg_k),
        enable_surge_window=bool(args.enable_surge_window),
        hazard_months=parse_months(args.hazard_months),
        surge_quantile=float(args.surge_quantile),
        make_plots=not bool(args.no_plots),
        overwrite=bool(args.overwrite),
    )

    result = run_a1_svi_sensitivity(
        config_path=args.config,
        repo_root=args.repo_root,
        sensitivity_config=sensitivity_config,
    )

    print(a1_sensitivity_brief(result).rstrip())
    print("\nWritten outputs:")
    for label, path in result.get("outputs", {}).items():
        print(f"  {label}: {path}")


if __name__ == "__main__":
    main()


__all__ = [
    "A1SensitivityError",
    "DEFAULT_SPLIT_SCHEMES",
    "MODEL_STAGE",
    "STAGE_SLUG",
    "SensitivityConfig",
    "SviScoreSpec",
    "TargetSpec",
    "a1_sensitivity_brief",
    "aggregate_tract_targets",
    "build_svi_score_specs",
    "candidate_svi_columns",
    "compact_sensitivity_table",
    "derive_surge_months_from_training",
    "evaluate_score_target_ranking",
    "run_a1_svi_sensitivity",
]
