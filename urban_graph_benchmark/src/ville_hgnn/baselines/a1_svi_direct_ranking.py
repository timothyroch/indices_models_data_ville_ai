"""
A1 SVI direct-ranking baseline for the Montréal 311 water/drainage benchmark.

This module implements the A1 baseline stage from
``baseline_plan_mtl_311_v0.md``.

A1 asks a simple question:

    Do static SVI vulnerability scores rank the same tracts that later show high
    reported water/drainage 311 burden?

Implemented evaluation views:

- tract-month ranking: repeat each tract's static SVI score for every month
- tract-level total burden ranking over validation/test windows
- tract-level mean monthly burden ranking over validation/test windows
- tract-level burden rate per 1,000 population, when population is available
- top-K and top-percent overlap diagnostics

This module intentionally does not calibrate SVI into count predictions. That
belongs to A2. Here SVI is treated as a static prioritization/ranking score.

Important methodology note:

- Continuous SVI columns such as ``svi_percentile`` or ``svi_score_raw`` should
  be interpreted as the primary A1 signal when available.
- Class-label columns such as ``svi_class`` are allowed only as secondary
  diagnostic ordinal fallbacks. Their spacing is ordinal, not interval-scaled.
- Administrative/status columns such as ``svi_scored`` are excluded. In the SVI
  map output, ``svi_scored`` is a boolean-like scoring-status flag, not a
  vulnerability score.
"""

from __future__ import annotations

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


STAGE_SLUG = "A1_svi_direct_ranking"
MODEL_STAGE = "A1_svi_direct_ranking"
DEFAULT_SPLIT_SCHEME = "temporal"
TARGET_NAME = TARGET_COLUMN
OBSERVED_ALIAS_COL = "observed_water_drainage_count"
PREDICTED_SCORE_COL = "predicted_score"

ZONE_COL = "zone_id"
PERIOD_COL = "period_month"
POPULATION_COL = "population_total_2021"

ABSOLUTE_K_VALUES = (25, 50, 100)
FRACTION_K_VALUES = (0.05, 0.10)

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

# Preferred order. The script will still scan additional score-like SVI columns.
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


class A1BaselineError(BaselineError):
    """Raised when A1 SVI direct-ranking evaluation fails."""


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


def require_runtime_dependencies() -> None:
    """Fail clearly if numpy/pandas are unavailable."""

    if pd is None:
        raise A1BaselineError("pandas is required for A1 baselines.") from _PANDAS_IMPORT_ERROR
    if np is None:
        raise A1BaselineError("numpy is required for A1 baselines.") from _NUMPY_IMPORT_ERROR


def normalize_frame_for_a1(frame: pd.DataFrame, split_scheme: str) -> pd.DataFrame:
    """Normalize columns required by A1 evaluation."""

    split_col = split_column_for_scheme(split_scheme)
    required = [ZONE_COL, PERIOD_COL, TARGET_COLUMN, split_col]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise A1BaselineError(f"A1 input frame missing required columns: {missing}")

    out = frame.copy()
    out[ZONE_COL] = out[ZONE_COL].astype(str)

    parsed = pd.to_datetime(out[PERIOD_COL].astype(str), errors="coerce")
    if parsed.isna().any():
        bad = out.loc[parsed.isna(), PERIOD_COL].drop_duplicates().head(20).tolist()
        raise A1BaselineError(f"Could not parse period_month values: {bad}")
    out[PERIOD_COL] = parsed.dt.to_period("M").astype(str)

    out[TARGET_COLUMN] = pd.to_numeric(out[TARGET_COLUMN], errors="coerce")
    if out[TARGET_COLUMN].isna().any():
        n_missing = int(out[TARGET_COLUMN].isna().sum())
        raise A1BaselineError(f"{TARGET_COLUMN} contains missing/non-numeric rows: {n_missing}")
    if (out[TARGET_COLUMN] < 0).any():
        n_negative = int((out[TARGET_COLUMN] < 0).sum())
        raise A1BaselineError(f"{TARGET_COLUMN} contains negative rows: {n_negative}")

    if BINARY_TARGET_COLUMN not in out.columns:
        out[BINARY_TARGET_COLUMN] = (out[TARGET_COLUMN] > 0).astype(int)
    else:
        out[BINARY_TARGET_COLUMN] = (
            pd.to_numeric(out[BINARY_TARGET_COLUMN], errors="coerce")
            .fillna(0)
            .astype(int)
        )

    return out.reset_index(drop=True)


def try_numeric_score(series: pd.Series) -> pd.Series | None:
    """Convert a candidate score column to numeric if possible."""

    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() == 0:
        return None
    return numeric.astype(float)


def try_class_score(series: pd.Series) -> pd.Series | None:
    """
    Convert an SVI class-like column into an ordinal diagnostic score.

    This is acceptable only as a fallback/diagnostic ranking signal, not as the
    primary A1 SVI score when continuous ``svi_percentile`` or ``svi_score_*``
    columns are available.

    The mapping is deterministic and not target-tuned. The numeric gaps should
    be interpreted as ordinal labels, not interval-scaled vulnerability units.
    Numeric classes are handled by ``try_numeric_score`` before this function.
    """

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
        lower = col.lower()
        if lower in EXCLUDED_SVI_COLUMNS:
            continue
        if lower in lower_to_original:
            ordered.append(lower_to_original[lower])

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

    # Stable de-duplication while preserving priority order.
    return list(dict.fromkeys(ordered))


def infer_score_role(source_col: str, conversion: str, orientation: str) -> str:
    """
    Classify score role for reporting.

    Primary candidate means a continuous SVI score/percentile-like signal.
    Diagnostic means rank/class/theme-derived or otherwise less direct signal.
    """

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

    Direction is deterministic from column semantics, not fitted to the 311 target.

    - score/percentile/theme columns are kept positive: higher = more vulnerable.
    - rank columns are multiplied by -1 under the convention that rank 1 is most vulnerable.
    - class columns are interpreted as diagnostic ordinal scores when possible.
    """

    out = frame.copy()
    specs: list[SviScoreSpec] = []
    audit: list[dict[str, Any]] = []

    candidates = candidate_svi_columns(out)

    for source_col in candidates:
        lower = source_col.lower()

        numeric = try_numeric_score(out[source_col])
        conversion = "numeric"

        if numeric is None and "class" in lower:
            numeric = try_class_score(out[source_col])
            conversion = "class_label_mapping"

        if numeric is None:
            audit.append(
                {
                    "source_column": source_col,
                    "status": "skipped_non_numeric",
                    "non_missing": int(out[source_col].notna().sum()),
                    "conversion": None,
                    "orientation": None,
                    "score_role": None,
                }
            )
            continue

        if numeric.notna().sum() == 0:
            audit.append(
                {
                    "source_column": source_col,
                    "status": "skipped_all_missing_after_conversion",
                    "non_missing": 0,
                    "conversion": conversion,
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
                "Rank column converted to a vulnerability score by multiplying by -1, "
                "under the deterministic convention that lower rank means higher vulnerability."
            )
        else:
            score_col = f"{source_col}__higher_more_vulnerable"
            score = numeric
            orientation = "positive_higher_more_vulnerable"
            interpretation = "Column used directly as a static SVI vulnerability score."

        score_role = infer_score_role(source_col, conversion=conversion, orientation=orientation)
        out[score_col] = score

        model_name = f"A1_svi_direct_ranking__{source_col}"
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
        raise A1BaselineError(
            "No usable SVI score-like columns found in tract_month_panel. Expected joined "
            "columns such as svi_percentile, svi_score_raw, svi_rank, or svi_class."
        )

    return out, specs, audit


def validate_static_svi_scores(frame: pd.DataFrame, specs: Sequence[SviScoreSpec]) -> list[dict[str, Any]]:
    """
    Audit whether each SVI score is effectively static within zone.

    A1 assumes SVI is a static tract-level score. Repeated values across months
    are expected; multiple values within a zone indicate an upstream issue.
    """

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


def score_nonmissing_frame(frame: pd.DataFrame, score_col: str) -> pd.DataFrame:
    """Return rows with non-missing numeric score and observed target."""

    out = frame.copy()
    out[score_col] = pd.to_numeric(out[score_col], errors="coerce")
    keep = out[score_col].notna() & out[TARGET_COLUMN].notna()
    return out[keep].copy()


def evaluate_tract_month_ranking(
    frame: pd.DataFrame,
    *,
    config: Mapping[str, Any],
    split_scheme: str,
    spec: SviScoreSpec,
) -> pd.DataFrame:
    """Evaluate tract-month ranking using repeated static SVI score."""

    context = build_run_context(
        config=config,
        frame=frame,
        split_scheme=split_scheme,
        prediction_setting="static_svi_direct_ranking_v0",
        model_stage=MODEL_STAGE,
        model_name=spec.model_name,
        target_name=TARGET_COLUMN,
        target_type="tract_month_ranking",
        feature_set_name=spec.feature_set_name,
        dataset_version=DATASET_VERSION_DEFAULT,
    )

    split_col = split_column_for_scheme(split_scheme)
    rows: list[dict[str, Any]] = []

    for partition in EVAL_PARTITIONS:
        part = frame[frame[split_col].astype(str) == partition].copy()
        part = score_nonmissing_frame(part, spec.score_column)

        metrics = evaluate_ranking_metrics(
            part[TARGET_COLUMN],
            part[spec.score_column],
            k_values=ABSOLUTE_K_VALUES,
            fractions=FRACTION_K_VALUES,
        )

        for metric in metrics:
            metric_prefixed = MetricResult(
                metric_name=f"tract_month_ranking__{metric.metric_name}",
                metric_value=metric.metric_value,
                higher_is_better=metric.higher_is_better,
                n=metric.n,
                notes=metric.notes,
            )
            rows.append(
                make_metric_row(
                    metric=metric_prefixed,
                    **context.metric_row_kwargs(eval_partition=partition),
                )
            )

    return make_metrics_dataframe(rows)


def aggregate_tract_burden(
    frame: pd.DataFrame,
    *,
    split_scheme: str,
    partition: str,
    spec: SviScoreSpec,
) -> pd.DataFrame:
    """Aggregate validation/test burden to one row per tract for A1 ranking."""

    split_col = split_column_for_scheme(split_scheme)
    part = frame[frame[split_col].astype(str) == partition].copy()

    if part.empty:
        raise A1BaselineError(f"No rows found for partition={partition!r}.")

    part = score_nonmissing_frame(part, spec.score_column)

    agg_dict: dict[str, tuple[str, str]] = {
        "observed_total_water_drainage_count": (TARGET_COLUMN, "sum"),
        "observed_mean_water_drainage_count": (TARGET_COLUMN, "mean"),
        "observed_positive_months": (TARGET_COLUMN, lambda s: int((s > 0).sum())),
        "n_months": (PERIOD_COL, "nunique"),
        "svi_score": (spec.score_column, "first"),
    }

    if POPULATION_COL in part.columns:
        agg_dict["population_total_2021"] = (POPULATION_COL, "first")

    tract = part.groupby(ZONE_COL, as_index=False).agg(**agg_dict)

    if POPULATION_COL in tract.columns:
        population = pd.to_numeric(tract[POPULATION_COL], errors="coerce")
        tract["observed_total_rate_per_1000_population"] = np.where(
            population > 0,
            tract["observed_total_water_drainage_count"] / population * 1000.0,
            np.nan,
        )

    tract["split_partition"] = partition
    tract["model_name"] = spec.model_name
    tract["feature_set_name"] = spec.feature_set_name
    tract["score_role"] = spec.score_role
    tract["source_svi_column"] = spec.source_column
    tract["oriented_score_column"] = spec.score_column

    tract["observed_total_rank_desc"] = tract["observed_total_water_drainage_count"].rank(
        method="min",
        ascending=False,
    )
    tract["svi_score_rank_desc"] = tract["svi_score"].rank(method="min", ascending=False)

    if "observed_total_rate_per_1000_population" in tract.columns:
        tract["observed_rate_rank_desc"] = tract["observed_total_rate_per_1000_population"].rank(
            method="min",
            ascending=False,
        )

    return tract


def evaluate_tract_level_ranking(
    tract_table: pd.DataFrame,
    *,
    config: Mapping[str, Any],
    frame: pd.DataFrame,
    split_scheme: str,
    partition: str,
    spec: SviScoreSpec,
    target_col: str,
    target_label: str,
) -> pd.DataFrame:
    """Evaluate tract-level ranking metrics for one burden definition."""

    valid = tract_table[[target_col, "svi_score"]].dropna().copy()

    context = build_run_context(
        config=config,
        frame=frame,
        split_scheme=split_scheme,
        prediction_setting="static_svi_direct_ranking_v0",
        model_stage=MODEL_STAGE,
        model_name=spec.model_name,
        target_name=target_label,
        target_type="tract_level_ranking",
        feature_set_name=spec.feature_set_name,
        dataset_version=DATASET_VERSION_DEFAULT,
    )

    metrics = evaluate_ranking_metrics(
        valid[target_col],
        valid["svi_score"],
        k_values=ABSOLUTE_K_VALUES,
        fractions=FRACTION_K_VALUES,
    )

    rows = []
    for metric in metrics:
        metric_prefixed = MetricResult(
            metric_name=f"tract_level_ranking__{target_label}__{metric.metric_name}",
            metric_value=metric.metric_value,
            higher_is_better=metric.higher_is_better,
            n=metric.n,
            notes=metric.notes,
        )
        rows.append(
            make_metric_row(
                metric=metric_prefixed,
                **context.metric_row_kwargs(eval_partition=partition),
            )
        )

    return make_metrics_dataframe(rows)


def topk_diagnostics_for_tract_table(
    tract_table: pd.DataFrame,
    *,
    split_scheme: str,
    partition: str,
    spec: SviScoreSpec,
    target_col: str,
    target_label: str,
) -> list[dict[str, Any]]:
    """Compute explicit top-K overlap diagnostics for tract-level ranking."""

    valid = tract_table[[ZONE_COL, target_col, "svi_score"]].dropna().copy()
    n = len(valid)
    if n == 0:
        return []

    k_specs: list[tuple[str, int]] = []
    for k in ABSOLUTE_K_VALUES:
        k_specs.append((f"top_{min(k, n)}", min(k, n)))
    for frac in FRACTION_K_VALUES:
        k_specs.append((f"top_{int(round(frac * 100))}pct", top_fraction_k(n, frac)))

    rows: list[dict[str, Any]] = []

    for label, k in k_specs:
        overlap = top_k_overlap(valid[target_col], valid["svi_score"], k)
        ndcg = ndcg_at_k(valid[target_col], valid["svi_score"], k)

        observed_top_ids = set(
            valid.nlargest(k, target_col, keep="all")[ZONE_COL].astype(str).head(k).tolist()
        )
        svi_top_ids = set(
            valid.nlargest(k, "svi_score", keep="all")[ZONE_COL].astype(str).head(k).tolist()
        )

        rows.append(
            {
                "split_scheme": split_scheme,
                "split_partition": partition,
                "model_name": spec.model_name,
                "feature_set_name": spec.feature_set_name,
                "score_role": spec.score_role,
                "source_svi_column": spec.source_column,
                "oriented_score_column": spec.score_column,
                "target_label": target_label,
                "target_column": target_col,
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


def make_predictions_long(
    frame: pd.DataFrame,
    *,
    split_scheme: str,
    specs: Sequence[SviScoreSpec],
) -> pd.DataFrame:
    """Create long-format tract-month SVI ranking predictions."""

    split_col = split_column_for_scheme(split_scheme)

    base_cols = [
        ZONE_COL,
        PERIOD_COL,
        split_col,
        TARGET_COLUMN,
        BINARY_TARGET_COLUMN,
        "year",
        "month",
        POPULATION_COL,
    ]
    base_cols = [col for col in base_cols if col in frame.columns]

    parts: list[pd.DataFrame] = []
    for spec in specs:
        part = frame[base_cols + [spec.score_column]].copy()
        part[OBSERVED_ALIAS_COL] = part[TARGET_COLUMN]
        part[PREDICTED_SCORE_COL] = pd.to_numeric(part[spec.score_column], errors="coerce")
        part["model_stage"] = MODEL_STAGE
        part["model_name"] = spec.model_name
        part["feature_set_name"] = spec.feature_set_name
        part["score_role"] = spec.score_role
        part["source_svi_column"] = spec.source_column
        part["oriented_score_column"] = spec.score_column
        part["score_orientation"] = spec.orientation
        parts.append(part.drop(columns=[spec.score_column], errors="ignore"))

    return pd.concat(parts, ignore_index=True)


def write_prediction_partitions(
    predictions_long: pd.DataFrame,
    output_dir: Path,
    *,
    split_scheme: str,
) -> dict[str, str]:
    """Write validation and test long-format prediction files."""

    split_col = split_column_for_scheme(split_scheme)
    validation_path = output_dir / "predictions_validation.parquet"
    test_path = output_dir / "predictions_test.parquet"

    validation = predictions_long[predictions_long[split_col].astype(str) == "validation"].copy()
    test = predictions_long[predictions_long[split_col].astype(str) == "test"].copy()

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
    """Return compact A1 metric summary for the report."""

    if metrics.empty:
        return metrics

    keep_patterns = [
        "spearman_corr",
        "kendall_corr",
        "ndcg_at_100",
        "top_10pct_overlap_rate",
        "top100_overlap_rate",
    ]

    mask = metrics["metric_name"].apply(
        lambda name: any(pattern in str(name) for pattern in keep_patterns)
    )
    out = metrics[mask].copy()

    wanted = [
        "split_name",
        "model_name",
        "target_name",
        "target_type",
        "metric_name",
        "metric_value",
        "higher_is_better",
        "n_rows",
    ]
    return out[[col for col in wanted if col in out.columns]].reset_index(drop=True)


def render_a1_report(
    *,
    metrics: pd.DataFrame,
    score_audit: Sequence[Mapping[str, Any]],
    static_audit: Sequence[Mapping[str, Any]],
    row_counts: Mapping[str, Any],
    outputs: Mapping[str, str],
    split_scheme: str,
    generated_at: str,
) -> str:
    """Render A1 SVI direct-ranking report."""

    included = [row for row in score_audit if row.get("status") == "included"]
    skipped = [row for row in score_audit if row.get("status") != "included"]
    primary = [row for row in included if row.get("is_primary_recommended")]
    diagnostic = [row for row in included if not row.get("is_primary_recommended")]
    compact = compact_metrics_summary(metrics)

    lines: list[str] = []
    lines.append("# A1 SVI Direct-Ranking Baseline — Montréal 311 Water/Drainage v0\n")
    lines.append(f"Generated at: `{generated_at}`\n")
    lines.append(f"Split scheme: `{split_scheme}`\n")
    lines.append(f"Split type: `{split_type_for_scheme(split_scheme)}`\n")

    lines.append("## Purpose\n")
    lines.append(
        "A1 evaluates static SVI as a direct prioritization score. It does not "
        "fit a count model and does not calibrate SVI to 311 outcomes. It asks "
        "whether tracts with higher SVI scores are also tracts with higher "
        "reported water/drainage 311 burden in validation/test windows.\n"
    )

    lines.append("## Row counts\n")
    lines.append("| Partition | Rows |")
    lines.append("|---|---:|")
    for key in ["train", "validation", "test"]:
        lines.append(f"| `{key}` | {row_counts.get(key)} |")
    lines.append("")

    lines.append("## Primary recommended SVI score columns\n")
    if primary:
        lines.append("| Source column | Oriented score column | Orientation | Non-missing |")
        lines.append("|---|---|---|---:|")
        for row in primary:
            lines.append(
                f"| `{row.get('source_column')}` | `{row.get('score_column')}` | "
                f"`{row.get('orientation')}` | {row.get('non_missing')} |"
            )
        lines.append("")
    else:
        lines.append(
            "_No continuous primary SVI score/percentile columns were detected. "
            "Diagnostic SVI columns may still have been evaluated._\n"
        )

    lines.append("## Diagnostic SVI score columns\n")
    if diagnostic:
        lines.append("| Source column | Score role | Oriented score column | Non-missing |")
        lines.append("|---|---|---|---:|")
        for row in diagnostic:
            lines.append(
                f"| `{row.get('source_column')}` | `{row.get('score_role')}` | "
                f"`{row.get('score_column')}` | {row.get('non_missing')} |"
            )
        lines.append("")
    else:
        lines.append("_No diagnostic SVI score columns included._\n")

    if skipped:
        lines.append("## Skipped SVI candidate columns\n")
        lines.append("| Source column | Status |")
        lines.append("|---|---|")
        for row in skipped:
            lines.append(f"| `{row.get('source_column')}` | `{row.get('status')}` |")
        lines.append("")

    lines.append("## Static-score audit\n")
    lines.append(dataframe_to_markdown(pd.DataFrame(static_audit)))
    lines.append("")

    lines.append("## Compact metrics summary\n")
    lines.append(dataframe_to_markdown(compact, max_rows=120))
    lines.append("")

    lines.append("## Output artifacts\n")
    lines.append("| Artifact | Path |")
    lines.append("|---|---|")
    for key, value in outputs.items():
        lines.append(f"| `{key}` | `{value}` |")
    lines.append("")

    lines.append("## Interpretation notes\n")
    lines.append(
        "- A1 is a static ranking baseline, not a calibrated predictor.\n"
        "- Continuous SVI score/percentile columns should be interpreted as the primary A1 result when available.\n"
        "- SVI class-label columns are diagnostic ordinal fallbacks only; their spacing is not interval-scaled.\n"
        "- SVI status/quality fields such as `svi_scored` are excluded from evaluation as vulnerability scores.\n"
        "- SVI is not hazard-specific; weak alignment with water/drainage 311 burden does not invalidate SVI.\n"
        "- The target is reported municipal 311 burden, not objective flood occurrence.\n"
        "- Rank-column orientation is deterministic and not tuned on validation/test outcomes.\n"
        "- SoVI is excluded from Track A because the SoVI reproduction is census-division scale, while this benchmark is census tract × month.\n"
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
    metrics: pd.DataFrame,
    generated_at: str,
) -> dict[str, Any]:
    """Build A1 metadata."""

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
            "model_name": "A1_svi_direct_ranking_suite",
            "split_scheme": split_scheme,
            "split_type": split_type_for_scheme(split_scheme),
            "target_name": TARGET_COLUMN,
            "target_type": "ranking",
            "prediction_setting": "static_svi_direct_ranking_v0",
            "row_counts": row_counts,
            "score_audit": list(score_audit),
            "static_score_audit": list(static_audit),
            "metric_rows": int(len(metrics)),
            "outputs": dict(outputs),
            "excluded_svi_columns": sorted(EXCLUDED_SVI_COLUMNS),
            "notes": (
                "A1 evaluates SVI as a static direct-ranking baseline. "
                "No calibration or target fitting is performed. Continuous "
                "SVI score/percentile columns are primary candidates; class-derived "
                "scores are diagnostic ordinal fallbacks only. Status/quality columns "
                "such as svi_scored are excluded."
            ),
        }
    )


def run_a1_svi_direct_ranking(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    repo_root: str | Path | None = None,
    split_scheme: str = DEFAULT_SPLIT_SCHEME,
) -> dict[str, Any]:
    """
    Run A1 SVI direct-ranking baseline and write standard artifacts.

    Returns a dictionary containing output paths, row counts, and score audits.
    """

    require_runtime_dependencies()

    config, root, resolved_config_path, panel_path, split_path, frame = load_benchmark_frame(
        config_path=config_path,
        repo_root=repo_root,
    )

    frame = normalize_frame_for_a1(frame, split_scheme=split_scheme)
    frame, specs, score_audit = build_svi_score_specs(frame)
    static_audit = validate_static_svi_scores(frame, specs)

    row_counts = split_counts(frame, split_scheme=split_scheme)
    missing_required = [part for part in ["train", "validation", "test"] if row_counts.get(part, 0) <= 0]
    if missing_required:
        raise A1BaselineError(
            f"Split scheme {split_scheme!r} missing required partitions: {missing_required}. "
            f"Counts: {row_counts}"
        )

    paths = get_baseline_paths(config, root, STAGE_SLUG)
    output_dir = paths.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    all_metric_frames: list[pd.DataFrame] = []
    all_tract_ranking_tables: list[pd.DataFrame] = []
    all_topk_rows: list[dict[str, Any]] = []

    tract_level_target_specs = [
        ("observed_total_water_drainage_count", "water_drainage_count_tract_total"),
        ("observed_mean_water_drainage_count", "water_drainage_count_tract_month_mean"),
    ]

    for spec in specs:
        all_metric_frames.append(
            evaluate_tract_month_ranking(
                frame,
                config=config,
                split_scheme=split_scheme,
                spec=spec,
            )
        )

        for partition in EVAL_PARTITIONS:
            tract_table = aggregate_tract_burden(
                frame,
                split_scheme=split_scheme,
                partition=partition,
                spec=spec,
            )

            local_target_specs = list(tract_level_target_specs)
            if "observed_total_rate_per_1000_population" in tract_table.columns:
                local_target_specs.append(
                    (
                        "observed_total_rate_per_1000_population",
                        "water_drainage_count_tract_total_rate_per_1000_population",
                    )
                )

            for target_col, target_label in local_target_specs:
                all_metric_frames.append(
                    evaluate_tract_level_ranking(
                        tract_table,
                        config=config,
                        frame=frame,
                        split_scheme=split_scheme,
                        partition=partition,
                        spec=spec,
                        target_col=target_col,
                        target_label=target_label,
                    )
                )

                all_topk_rows.extend(
                    topk_diagnostics_for_tract_table(
                        tract_table,
                        split_scheme=split_scheme,
                        partition=partition,
                        spec=spec,
                        target_col=target_col,
                        target_label=target_label,
                    )
                )

            all_tract_ranking_tables.append(tract_table)

    metrics = pd.concat(all_metric_frames, ignore_index=True) if all_metric_frames else make_metrics_dataframe([])
    tract_rankings = pd.concat(all_tract_ranking_tables, ignore_index=True) if all_tract_ranking_tables else pd.DataFrame()
    topk = pd.DataFrame(all_topk_rows)

    predictions_long = make_predictions_long(frame, split_scheme=split_scheme, specs=specs)

    tract_validation_path = output_dir / "tract_ranking_validation.csv"
    tract_test_path = output_dir / "tract_ranking_test.csv"
    topk_path = output_dir / "topk_overlap.csv"
    score_audit_path = output_dir / "svi_score_audit.csv"
    static_audit_path = output_dir / "svi_static_score_audit.csv"

    tract_rankings[tract_rankings["split_partition"] == "validation"].to_csv(tract_validation_path, index=False)
    tract_rankings[tract_rankings["split_partition"] == "test"].to_csv(tract_test_path, index=False)
    topk.to_csv(topk_path, index=False)
    pd.DataFrame(score_audit).to_csv(score_audit_path, index=False)
    pd.DataFrame(static_audit).to_csv(static_audit_path, index=False)

    written_predictions = write_prediction_partitions(
        predictions_long,
        output_dir,
        split_scheme=split_scheme,
    )

    generated_at = datetime.now(timezone.utc).isoformat()

    outputs = {
        "metrics": str(paths.metrics),
        "model_metadata": str(paths.model_metadata),
        "baseline_report": str(paths.baseline_report),
        "tract_ranking_validation": str(tract_validation_path),
        "tract_ranking_test": str(tract_test_path),
        "topk_overlap": str(topk_path),
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
        metrics=metrics,
        generated_at=generated_at,
    )

    report = render_a1_report(
        metrics=metrics,
        score_audit=score_audit,
        static_audit=static_audit,
        row_counts=row_counts,
        outputs=outputs,
        split_scheme=split_scheme,
        generated_at=generated_at,
    )

    metrics.to_csv(paths.metrics, index=False)
    write_json(paths.model_metadata, metadata)
    write_markdown(paths.baseline_report, report)

    return {
        "status": "completed",
        "model_stage": MODEL_STAGE,
        "model_name": "A1_svi_direct_ranking_suite",
        "split_scheme": split_scheme,
        "outputs": outputs,
        "row_counts": row_counts,
        "svi_score_count": len(specs),
        "svi_score_columns": [spec.source_column for spec in specs],
        "svi_primary_score_columns": [
            spec.source_column
            for spec in specs
            if spec.score_role == "primary_continuous_svi_score_candidate"
        ],
        "svi_diagnostic_score_columns": [
            spec.source_column
            for spec in specs
            if spec.score_role != "primary_continuous_svi_score_candidate"
        ],
        "metric_rows": int(len(metrics)),
        "prediction_rows": int(len(predictions_long)),
        "tract_ranking_rows": int(len(tract_rankings)),
        "topk_rows": int(len(topk)),
    }


def a1_brief(result: Mapping[str, Any]) -> str:
    """Return concise A1 run summary."""

    outputs = result.get("outputs", {})
    return (
        "A1 SVI direct-ranking baseline completed.\n"
        f"Status: {result.get('status')}\n"
        f"Split scheme: {result.get('split_scheme')}\n"
        f"SVI score columns: {result.get('svi_score_count')}\n"
        f"Primary SVI columns: {result.get('svi_primary_score_columns')}\n"
        f"Diagnostic SVI columns: {result.get('svi_diagnostic_score_columns')}\n"
        f"Metric rows: {result.get('metric_rows')}\n"
        f"Prediction rows: {result.get('prediction_rows')}\n"
        f"Tract ranking rows: {result.get('tract_ranking_rows')}\n"
        f"Top-K rows: {result.get('topk_rows')}\n"
        f"Metrics: {outputs.get('metrics')}\n"
        f"Report: {outputs.get('baseline_report')}\n"
    )


def main() -> None:
    """CLI entry point for direct module execution."""

    import argparse

    parser = argparse.ArgumentParser(
        description="Run A1 SVI direct-ranking baseline for Montréal 311 water/drainage."
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

    result = run_a1_svi_direct_ranking(
        config_path=args.config,
        repo_root=args.repo_root,
        split_scheme=args.split_scheme,
    )

    print(a1_brief(result).rstrip())
    print("\nWritten outputs:")
    for label, path in result.get("outputs", {}).items():
        print(f"  {label}: {path}")


if __name__ == "__main__":
    main()


__all__ = [
    "A1BaselineError",
    "DEFAULT_SPLIT_SCHEME",
    "MODEL_STAGE",
    "STAGE_SLUG",
    "SviScoreSpec",
    "a1_brief",
    "aggregate_tract_burden",
    "build_svi_score_specs",
    "candidate_svi_columns",
    "evaluate_tract_level_ranking",
    "evaluate_tract_month_ranking",
    "run_a1_svi_direct_ranking",
]