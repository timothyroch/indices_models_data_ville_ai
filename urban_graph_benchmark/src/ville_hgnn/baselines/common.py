"""
Common utilities for Montréal 311 water/drainage baselines.

Shared infrastructure for A0/A1/A2/A3 baselines:
- load the validated Dataset v0 panel and split assignments
- join panel + splits on zone_id × period_month
- resolve standard baseline output directories
- validate split/key consistency
- enforce basic leakage guards
- evaluate prediction dataframes using the shared metrics layer
- write standardized metrics, predictions, metadata, and reports

This module contains no A0/A1/A2/A3 model logic and no graph logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover
    pd = None  # type: ignore[assignment]
    _PANDAS_IMPORT_ERROR = exc
else:
    _PANDAS_IMPORT_ERROR = None

from ville_hgnn.evaluation.metrics import (
    MetricResult,
    evaluate_binary_metrics,
    evaluate_count_metrics,
    evaluate_ranking_metrics,
    make_metric_row,
    make_metrics_dataframe,
)
from ville_hgnn.utils.io import (
    config_hash,
    file_hash,
    load_config,
    to_jsonable,
    write_json,
    write_markdown,
)
from ville_hgnn.utils.paths import (
    find_repo_root,
    get_nested,
    is_unresolved_value,
    resolve_path,
)


DEFAULT_CONFIG_PATH = "urban_graph_benchmark/configs/mtl_311_water_v0.1.yaml"
DEFAULT_PANEL_PATH = "urban_graph_benchmark/outputs/mtl_311_water_v0/datasets/tract_month_panel.parquet"
DEFAULT_SPLIT_ASSIGNMENTS_PATH = "urban_graph_benchmark/outputs/mtl_311_water_v0/datasets/splits/split_assignments.parquet"
DEFAULT_BASELINE_ROOT = "urban_graph_benchmark/outputs/mtl_311_water_v0/baselines"

BENCHMARK_ID_DEFAULT = "mtl_311_water_v0"
DATASET_VERSION_DEFAULT = "dataset_v0"

TARGET_COLUMN = "water_drainage_count"
BINARY_TARGET_COLUMN = "water_drainage_binary"
KEY_COLUMNS = ["zone_id", "period_month"]

# Must match build_splits.py output columns.
SPLIT_COLUMNS = {
    "temporal": "temporal_split",
    "random_debug": "random_debug_split",
    "spatial_block": "spatial_block_split",
}

SPLIT_TYPES = {
    "temporal": "temporal",
    "random_debug": "random_tract_month_debug_only",
    "spatial_block": "spatial_block_preliminary",
}

EVAL_PARTITIONS = ("validation", "test")

TARGET_DERIVED_FORBIDDEN_COLUMNS = {
    "water_drainage_count",
    "water_drainage_binary",
    "water_drainage_requests",
    "share_water_drainage_requests",
}

STRICT_FORECASTING_FORBIDDEN_COLUMNS = {
    "total_311_count_all",
    "total_311_count_non_water_drainage",
}

RETROSPECTIVE_TARGET_CONTAINING_COLUMNS = {
    "total_311_count_all",
}


class BaselineError(RuntimeError):
    """Raised when baseline utilities detect an invalid state."""


@dataclass(frozen=True)
class BaselinePaths:
    """Standard output paths for a baseline stage."""

    output_dir: Path
    metrics: Path
    predictions_validation: Path
    predictions_test: Path
    model_metadata: Path
    baseline_report: Path

    def to_dict(self) -> dict[str, str]:
        return {key: str(value) for key, value in self.__dict__.items()}


@dataclass(frozen=True)
class BaselineRunContext:
    """Standard metadata context for metric rows."""

    benchmark_id: str
    dataset_version: str
    split_scheme: str
    split_type: str
    prediction_setting: str
    model_stage: str
    model_name: str
    target_name: str
    target_type: str
    feature_set_name: str
    n_train: int | None = None
    n_validation: int | None = None
    n_test: int | None = None

    def metric_row_kwargs(self, eval_partition: str) -> dict[str, Any]:
        """
        Return keyword arguments for shared metric-row creation.

        The standardized metrics schema has a split_name column but not a
        separate eval_partition column. We encode both, e.g. temporal_validation.
        """

        return {
            "benchmark_id": self.benchmark_id,
            "dataset_version": self.dataset_version,
            "split_name": f"{self.split_scheme}_{eval_partition}",
            "split_type": self.split_type,
            "prediction_setting": self.prediction_setting,
            "model_stage": self.model_stage,
            "model_name": self.model_name,
            "target_name": self.target_name,
            "target_type": self.target_type,
            "feature_set_name": self.feature_set_name,
            "n_train": self.n_train,
            "n_validation": self.n_validation,
            "n_test": self.n_test,
            "extra_notes": f"eval_partition={eval_partition}",
        }


def require_pandas() -> None:
    """Fail clearly if pandas is unavailable."""

    if pd is None:
        raise BaselineError("pandas is required for baseline utilities.") from _PANDAS_IMPORT_ERROR


def read_table(path: Path) -> pd.DataFrame:
    """Read a parquet or CSV table."""

    require_pandas()
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path, low_memory=False)
    raise BaselineError(f"Unsupported table format: {path}")


def write_dataframe(df: pd.DataFrame, path: Path) -> None:
    """Write DataFrame to parquet or CSV based on extension."""

    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        df.to_parquet(path, index=False)
        return
    if suffix == ".csv":
        df.to_csv(path, index=False)
        return
    raise BaselineError(f"Unsupported output table format: {path}")


def resolve_existing_path(value: str | Path | None, repo_root: Path, label: str) -> Path:
    """Resolve and verify an existing path."""

    if is_unresolved_value(value):
        raise BaselineError(f"Missing required path for {label}: {value!r}")

    resolved = resolve_path(value, repo_root=repo_root, allow_unresolved=False)
    if resolved is None:
        raise BaselineError(f"Could not resolve {label} path: {value!r}")
    if not resolved.exists():
        raise BaselineError(f"{label} path does not exist: {resolved}")
    return resolved


def resolve_panel_path(config: Mapping[str, Any], repo_root: Path) -> Path:
    """Resolve the validated Dataset v0 panel path."""

    configured = get_nested(config, ["paths", "expected_output_files", "tract_month_panel"], default=None)
    value = configured if not is_unresolved_value(configured) else DEFAULT_PANEL_PATH
    return resolve_existing_path(value, repo_root, label="tract_month_panel")


def resolve_split_assignments_path(config: Mapping[str, Any], repo_root: Path) -> Path:
    """Resolve split assignments path."""

    configured = get_nested(config, ["splits", "split_artifacts_planned", "split_assignments"], default=None)
    if is_unresolved_value(configured):
        split_dir = get_nested(config, ["splits", "output_dir"], default=None)
        if not is_unresolved_value(split_dir):
            configured = str(Path(str(split_dir)) / "split_assignments.parquet")
        else:
            configured = DEFAULT_SPLIT_ASSIGNMENTS_PATH
    return resolve_existing_path(configured, repo_root, label="split_assignments")


def resolve_baseline_root(config: Mapping[str, Any], repo_root: Path) -> Path:
    """Resolve baseline output root directory."""

    configured = get_nested(config, ["paths", "outputs", "baseline_dir"], default=None)
    value = configured if not is_unresolved_value(configured) else DEFAULT_BASELINE_ROOT
    resolved = resolve_path(value, repo_root=repo_root, allow_unresolved=False)
    if resolved is None:
        raise BaselineError("Could not resolve baseline output root.")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def get_baseline_paths(config: Mapping[str, Any], repo_root: Path, stage_slug: str) -> BaselinePaths:
    """Return standard output paths for a baseline stage."""

    root = resolve_baseline_root(config, repo_root)
    output_dir = root / stage_slug
    output_dir.mkdir(parents=True, exist_ok=True)
    return BaselinePaths(
        output_dir=output_dir,
        metrics=output_dir / "metrics.csv",
        predictions_validation=output_dir / "predictions_validation.parquet",
        predictions_test=output_dir / "predictions_test.parquet",
        model_metadata=output_dir / "model_metadata.json",
        baseline_report=output_dir / "baseline_report.md",
    )


def load_config_and_inputs(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    repo_root: str | Path | None = None,
) -> tuple[dict[str, Any], Path, Path, Path, Path, pd.DataFrame, pd.DataFrame]:
    """
    Load config, panel, and split assignments.

    Returns:
        config, repo_root, resolved_config_path, panel_path, split_assignments_path, panel, splits
    """

    require_pandas()
    root = Path(repo_root).resolve() if repo_root is not None else find_repo_root()
    resolved_config_path = resolve_path(config_path, repo_root=root, allow_unresolved=False)
    if resolved_config_path is None:
        raise BaselineError(f"Could not resolve config path: {config_path}")

    config = load_config(resolved_config_path)
    panel_path = resolve_panel_path(config, root)
    split_path = resolve_split_assignments_path(config, root)
    panel = read_table(panel_path)
    splits = read_table(split_path)
    panel.columns = [str(col) for col in panel.columns]
    splits.columns = [str(col) for col in splits.columns]
    return config, root, resolved_config_path, panel_path, split_path, panel, splits


def load_panel(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    repo_root: str | Path | None = None,
) -> tuple[dict[str, Any], Path, Path, Path, pd.DataFrame]:
    """Load config and validated Dataset v0 panel only."""

    config, root, resolved_config_path, panel_path, _split_path, panel, _splits = load_config_and_inputs(
        config_path=config_path,
        repo_root=repo_root,
    )
    return config, root, resolved_config_path, panel_path, panel


def load_split_assignments(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    repo_root: str | Path | None = None,
) -> tuple[dict[str, Any], Path, Path, Path, pd.DataFrame]:
    """Load config and split assignments only."""

    config, root, resolved_config_path, _panel_path, split_path, _panel, splits = load_config_and_inputs(
        config_path=config_path,
        repo_root=repo_root,
    )
    return config, root, resolved_config_path, split_path, splits


def ensure_unique_keys(df: pd.DataFrame, keys: Sequence[str], label: str) -> None:
    """Ensure DataFrame has unique key rows."""

    missing = [col for col in keys if col not in df.columns]
    if missing:
        raise BaselineError(f"{label} missing key columns: {missing}")

    dupes = int(df.duplicated(list(keys)).sum())
    if dupes:
        examples = (
            df.loc[df.duplicated(list(keys), keep=False), list(keys)]
            .head(20)
            .to_dict(orient="records")
        )
        raise BaselineError(
            f"{label} has duplicate rows for keys {list(keys)}: {dupes}. Examples: {examples}"
        )


def normalize_key_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize key columns used for joins."""

    out = df.copy()
    if "zone_id" in out.columns:
        out["zone_id"] = out["zone_id"].astype(str)
    if "period_month" in out.columns:
        parsed = pd.to_datetime(out["period_month"].astype(str), errors="coerce")
        if parsed.isna().any():
            bad = out.loc[parsed.isna(), "period_month"].drop_duplicates().head(20).tolist()
            raise BaselineError(f"Could not parse period_month values: {bad}")
        out["period_month"] = parsed.dt.to_period("M").astype(str)
    return out


def merge_panel_with_splits(panel: pd.DataFrame, splits: pd.DataFrame) -> pd.DataFrame:
    """Merge panel and split assignments one-to-one on zone_id × period_month."""

    panel_norm = normalize_key_columns(panel)
    splits_norm = normalize_key_columns(splits)
    ensure_unique_keys(panel_norm, KEY_COLUMNS, label="panel")
    ensure_unique_keys(splits_norm, KEY_COLUMNS, label="split assignments")

    expected_split_cols = set(SPLIT_COLUMNS.values())
    split_cols = [
        col
        for col in splits_norm.columns
        if col in KEY_COLUMNS or col in expected_split_cols or col.startswith("magnitude_class_")
    ]

    missing_expected = sorted(expected_split_cols - set(split_cols))
    if missing_expected:
        raise BaselineError(
            "Split assignments are missing expected split columns: "
            f"{missing_expected}. Available columns: {list(splits_norm.columns)}"
        )

    merged = panel_norm.merge(
        splits_norm[split_cols],
        on=KEY_COLUMNS,
        how="left",
        validate="one_to_one",
    )

    actual_split_cols = [col for col in split_cols if col in expected_split_cols]
    missing_split_rows = int(merged[actual_split_cols].isna().all(axis=1).sum())
    if missing_split_rows:
        raise BaselineError(f"{missing_split_rows} panel rows did not receive any split assignment.")

    return merged


def validate_benchmark_frame(frame: pd.DataFrame) -> None:
    """Validate the merged benchmark frame."""

    ensure_unique_keys(frame, KEY_COLUMNS, label="benchmark frame")
    missing_required = [
        col for col in [*KEY_COLUMNS, TARGET_COLUMN, *SPLIT_COLUMNS.values()]
        if col not in frame.columns
    ]
    if missing_required:
        raise BaselineError(f"Benchmark frame missing required columns: {missing_required}")

    target_missing = int(pd.to_numeric(frame[TARGET_COLUMN], errors="coerce").isna().sum())
    if target_missing:
        raise BaselineError(f"Benchmark frame target has missing/non-numeric rows: {target_missing}")

    for split_col in SPLIT_COLUMNS.values():
        missing = int(frame[split_col].isna().sum())
        if missing:
            raise BaselineError(f"Split column {split_col} has missing rows: {missing}")


def load_benchmark_frame(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    repo_root: str | Path | None = None,
) -> tuple[dict[str, Any], Path, Path, Path, Path, pd.DataFrame]:
    """
    Load config, panel, split assignments, and return the merged benchmark frame.

    Returns:
        config, repo_root, resolved_config_path, panel_path, split_assignments_path, frame
    """

    config, root, resolved_config_path, panel_path, split_path, panel, splits = load_config_and_inputs(
        config_path=config_path,
        repo_root=repo_root,
    )
    frame = merge_panel_with_splits(panel, splits)
    validate_benchmark_frame(frame)
    return config, root, resolved_config_path, panel_path, split_path, frame


def split_column_for_scheme(split_scheme: str) -> str:
    """Return split column for a split scheme."""

    if split_scheme not in SPLIT_COLUMNS:
        raise BaselineError(f"Unknown split_scheme={split_scheme!r}. Allowed: {sorted(SPLIT_COLUMNS)}")
    return SPLIT_COLUMNS[split_scheme]


def split_type_for_scheme(split_scheme: str) -> str:
    """Return split type label for a split scheme."""

    return SPLIT_TYPES.get(split_scheme, split_scheme)


def split_counts(frame: pd.DataFrame, split_scheme: str = "temporal") -> dict[str, int]:
    """Return row counts by split partition."""

    col = split_column_for_scheme(split_scheme)
    if col not in frame.columns:
        raise BaselineError(f"Missing split column: {col}")
    return {str(k): int(v) for k, v in frame[col].value_counts(dropna=False).to_dict().items()}


def get_split_frame(frame: pd.DataFrame, split_scheme: str, partition: str) -> pd.DataFrame:
    """Return rows for a split partition."""

    col = split_column_for_scheme(split_scheme)
    if col not in frame.columns:
        raise BaselineError(f"Missing split column: {col}")
    return frame[frame[col].astype(str) == partition].copy()


def get_split_masks(frame: pd.DataFrame, split_scheme: str = "temporal") -> dict[str, pd.Series]:
    """Return boolean masks for train/validation/test partitions."""

    col = split_column_for_scheme(split_scheme)
    if col not in frame.columns:
        raise BaselineError(f"Missing split column: {col}")
    return {
        "train": frame[col].astype(str) == "train",
        "validation": frame[col].astype(str) == "validation",
        "test": frame[col].astype(str) == "test",
    }


def build_run_context(
    *,
    config: Mapping[str, Any],
    frame: pd.DataFrame,
    split_scheme: str,
    prediction_setting: str,
    model_stage: str,
    model_name: str,
    target_name: str = TARGET_COLUMN,
    target_type: str = "count",
    feature_set_name: str = "none",
    dataset_version: str = DATASET_VERSION_DEFAULT,
) -> BaselineRunContext:
    """Build standardized metric-row context from frame and config."""

    counts = split_counts(frame, split_scheme=split_scheme)
    return BaselineRunContext(
        benchmark_id=str(config.get("benchmark_id", BENCHMARK_ID_DEFAULT)),
        dataset_version=dataset_version,
        split_scheme=split_scheme,
        split_type=split_type_for_scheme(split_scheme),
        prediction_setting=prediction_setting,
        model_stage=model_stage,
        model_name=model_name,
        target_name=target_name,
        target_type=target_type,
        feature_set_name=feature_set_name,
        n_train=counts.get("train"),
        n_validation=counts.get("validation"),
        n_test=counts.get("test"),
    )


def assert_no_forbidden_feature_columns(feature_columns: Iterable[str], *, prediction_setting: str) -> None:
    """Guard against obvious leakage-prone feature columns by name."""

    features = [str(col) for col in feature_columns]
    lower_map = {col: col.lower() for col in features}

    exact_hits = [col for col in features if col in TARGET_DERIVED_FORBIDDEN_COLUMNS]
    if exact_hits:
        raise BaselineError(f"Feature set contains target-derived leakage columns: {exact_hits}")

    sovi_hits = [col for col, lower in lower_map.items() if "sovi" in lower]
    if sovi_hits:
        raise BaselineError(f"Feature set contains SoVI columns, excluded from Track A: {sovi_hits}")

    target_share_hits = [col for col, lower in lower_map.items() if lower == "share_water_drainage_requests"]
    if target_share_hits:
        raise BaselineError(f"Feature set contains target-derived share columns: {target_share_hits}")

    if prediction_setting == "forecasting_v0":
        forecasting_hits = [col for col in features if col in STRICT_FORECASTING_FORBIDDEN_COLUMNS]
        if forecasting_hits:
            raise BaselineError(
                "Strict forecasting feature set contains same-month reporting-control columns: "
                f"{forecasting_hits}"
            )


def validate_prediction_frame(
    predictions: pd.DataFrame,
    *,
    observed_col: str,
    predicted_col: str,
    split_col: str,
    require_partitions: Sequence[str] = EVAL_PARTITIONS,
) -> None:
    """Validate a prediction dataframe before evaluation."""

    required = [*KEY_COLUMNS, split_col, observed_col, predicted_col]
    missing = [col for col in required if col not in predictions.columns]
    if missing:
        raise BaselineError(f"Prediction dataframe missing required columns: {missing}")

    ensure_unique_keys(predictions, KEY_COLUMNS, label="predictions")
    observed_missing = int(pd.to_numeric(predictions[observed_col], errors="coerce").isna().sum())
    predicted_missing = int(pd.to_numeric(predictions[predicted_col], errors="coerce").isna().sum())
    if observed_missing:
        raise BaselineError(f"Observed column has nonnumeric/missing rows: {observed_missing}")
    if predicted_missing:
        raise BaselineError(f"Predicted column has nonnumeric/missing rows: {predicted_missing}")

    partitions = set(predictions[split_col].dropna().astype(str).unique().tolist())
    missing_partitions = [partition for partition in require_partitions if partition not in partitions]
    if missing_partitions:
        raise BaselineError(
            f"Prediction dataframe missing required eval partitions {missing_partitions} in {split_col}."
        )


def prefix_metric_results(metrics: Sequence[MetricResult], prefix: str) -> list[MetricResult]:
    """Prefix metric names to avoid duplicate count/ranking names."""

    return [
        MetricResult(
            metric_name=f"{prefix}__{metric.metric_name}",
            metric_value=metric.metric_value,
            higher_is_better=metric.higher_is_better,
            n=metric.n,
            notes=metric.notes,
        )
        for metric in metrics
    ]


def evaluate_prediction_frame(
    predictions: pd.DataFrame,
    *,
    context: BaselineRunContext,
    split_scheme: str = "temporal",
    observed_col: str = TARGET_COLUMN,
    predicted_col: str = "predicted_water_drainage_count",
    binary_observed_col: str | None = BINARY_TARGET_COLUMN,
    binary_score_col: str | None = None,
    ranking_score_col: str | None = None,
    eval_partitions: Sequence[str] = EVAL_PARTITIONS,
    ranking_k_values: Sequence[int] = (10, 25, 50, 100),
    ranking_fractions: Sequence[float] = (0.05, 0.10),
) -> pd.DataFrame:
    """Evaluate predictions on validation/test partitions and return a metrics table."""

    require_pandas()
    split_col = split_column_for_scheme(split_scheme)
    validate_prediction_frame(
        predictions,
        observed_col=observed_col,
        predicted_col=predicted_col,
        split_col=split_col,
        require_partitions=eval_partitions,
    )

    rows: list[dict[str, Any]] = []
    for partition in eval_partitions:
        part = predictions[predictions[split_col].astype(str) == partition].copy()
        if part.empty:
            continue

        y_true = part[observed_col]
        y_pred = part[predicted_col]
        y_rank = part[ranking_score_col] if ranking_score_col else y_pred

        metric_groups: list[MetricResult] = []
        metric_groups.extend(prefix_metric_results(evaluate_count_metrics(y_true, y_pred), "count"))
        metric_groups.extend(
            prefix_metric_results(
                evaluate_ranking_metrics(
                    y_true,
                    y_rank,
                    k_values=ranking_k_values,
                    fractions=ranking_fractions,
                ),
                "ranking",
            )
        )

        if (
            binary_observed_col is not None
            and binary_score_col is not None
            and binary_observed_col in part.columns
            and binary_score_col in part.columns
        ):
            metric_groups.extend(
                prefix_metric_results(
                    evaluate_binary_metrics(part[binary_observed_col], part[binary_score_col]),
                    "binary",
                )
            )

        for metric in metric_groups:
            rows.append(make_metric_row(metric=metric, **context.metric_row_kwargs(eval_partition=partition)))

    return make_metrics_dataframe(rows)


def prediction_output_columns(
    predictions: pd.DataFrame,
    *,
    split_scheme: str,
    observed_col: str,
    predicted_col: str,
    extra_columns: Sequence[str] = (),
) -> list[str]:
    """Return stable prediction output columns available in a prediction dataframe."""

    split_col = split_column_for_scheme(split_scheme)
    preferred = [
        "zone_id",
        "period_month",
        split_col,
        observed_col,
        predicted_col,
        "predicted_score",
        "model_name",
        "feature_set_name",
        *extra_columns,
    ]
    return [col for col in preferred if col in predictions.columns]


def write_prediction_partitions(
    predictions: pd.DataFrame,
    paths: BaselinePaths,
    *,
    split_scheme: str = "temporal",
    observed_col: str = TARGET_COLUMN,
    predicted_col: str = "predicted_water_drainage_count",
    extra_columns: Sequence[str] = (),
) -> dict[str, str]:
    """Write validation and test prediction parquet files."""

    split_col = split_column_for_scheme(split_scheme)
    columns = prediction_output_columns(
        predictions,
        split_scheme=split_scheme,
        observed_col=observed_col,
        predicted_col=predicted_col,
        extra_columns=extra_columns,
    )

    validation = predictions[predictions[split_col].astype(str) == "validation"][columns].copy()
    test = predictions[predictions[split_col].astype(str) == "test"][columns].copy()
    write_dataframe(validation, paths.predictions_validation)
    write_dataframe(test, paths.predictions_test)
    return {
        "predictions_validation": str(paths.predictions_validation),
        "predictions_test": str(paths.predictions_test),
    }


def build_model_metadata(
    *,
    config: Mapping[str, Any],
    config_path: str | Path,
    panel_path: Path,
    split_assignments_path: Path,
    paths: BaselinePaths,
    context: BaselineRunContext,
    row_counts: Mapping[str, Any],
    feature_columns: Sequence[str] = (),
    model_parameters: Mapping[str, Any] | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    """Build standardized model/baseline metadata."""

    return to_jsonable(
        {
            "benchmark_id": context.benchmark_id,
            "dataset_version": context.dataset_version,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "config_path": str(config_path),
            "config_hash": config_hash(config),
            "panel_path": str(panel_path),
            "panel_sha256": file_hash(panel_path),
            "split_assignments_path": str(split_assignments_path),
            "split_assignments_sha256": file_hash(split_assignments_path),
            "model_stage": context.model_stage,
            "model_name": context.model_name,
            "feature_set_name": context.feature_set_name,
            "target_name": context.target_name,
            "target_type": context.target_type,
            "split_scheme": context.split_scheme,
            "split_type": context.split_type,
            "prediction_setting": context.prediction_setting,
            "row_counts": row_counts,
            "feature_columns": list(feature_columns),
            "model_parameters": model_parameters or {},
            "outputs": paths.to_dict(),
            "notes": notes,
        }
    )


def dataframe_to_markdown_table(df: pd.DataFrame) -> str:
    """Render a dataframe to Markdown, with a fallback if tabulate is unavailable."""

    if df.empty:
        return "_No rows._"
    try:
        return df.to_markdown(index=False)
    except Exception:
        cols = list(df.columns)
        lines = [
            "| " + " | ".join(cols) + " |",
            "| " + " | ".join(["---"] * len(cols)) + " |",
        ]
        for _, row in df.iterrows():
            lines.append("| " + " | ".join(str(row[col]) for col in cols) + " |")
        return "\n".join(lines)


def render_baseline_report(
    *,
    title: str,
    context: BaselineRunContext,
    metrics: pd.DataFrame,
    metadata: Mapping[str, Any],
    paths: BaselinePaths,
    additional_sections: Mapping[str, str] | None = None,
) -> str:
    """Render a compact generic baseline report."""

    lines: list[str] = []
    lines.append(f"# {title}\n")
    lines.append(f"Generated at: `{metadata.get('generated_at')}`\n")
    lines.append(f"Benchmark ID: `{context.benchmark_id}`\n")
    lines.append(f"Model stage: `{context.model_stage}`\n")
    lines.append(f"Model name: `{context.model_name}`\n")
    lines.append(f"Feature set: `{context.feature_set_name}`\n")
    lines.append(f"Prediction setting: `{context.prediction_setting}`\n")
    lines.append(f"Split scheme: `{context.split_scheme}`\n")

    lines.append("## Row counts\n")
    lines.append("| Partition | Rows |")
    lines.append("|---|---:|")
    lines.append(f"| train | {context.n_train} |")
    lines.append(f"| validation | {context.n_validation} |")
    lines.append(f"| test | {context.n_test} |")
    lines.append("")

    if metrics.empty:
        lines.append("## Metrics\n")
        lines.append("_No metrics were produced._\n")
    else:
        lines.append("## Metrics summary\n")
        display_cols = ["split_name", "metric_name", "metric_value", "higher_is_better", "n_rows"]
        available_cols = [col for col in display_cols if col in metrics.columns]
        lines.append(dataframe_to_markdown_table(metrics[available_cols]))
        lines.append("")

    if additional_sections:
        for section_title, section_body in additional_sections.items():
            lines.append(f"## {section_title}\n")
            lines.append(str(section_body).rstrip() + "\n")

    lines.append("## Output artifacts\n")
    lines.append("| Artifact | Path |")
    lines.append("|---|---|")
    for key, value in paths.to_dict().items():
        lines.append(f"| `{key}` | `{value}` |")
    lines.append("")

    lines.append("## Leakage notes\n")
    lines.append(
        "- Same-month target-derived columns must not be used as features.\n"
        "- `total_311_count_all` contains the water/drainage target and is not a strict forecasting feature.\n"
        "- `total_311_count_non_water_drainage` is retrospective when used in the same month.\n"
        "- SoVI columns are excluded from Track A.\n"
    )
    return "\n".join(lines)


def write_baseline_artifacts(
    *,
    paths: BaselinePaths,
    metrics: pd.DataFrame,
    predictions: pd.DataFrame | None,
    metadata: Mapping[str, Any],
    report_markdown: str,
    split_scheme: str = "temporal",
    observed_col: str = TARGET_COLUMN,
    predicted_col: str = "predicted_water_drainage_count",
    prediction_extra_columns: Sequence[str] = (),
) -> dict[str, str]:
    """Write standard baseline artifacts."""

    paths.output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(paths.metrics, index=False)
    write_json(paths.model_metadata, metadata)
    write_markdown(paths.baseline_report, report_markdown)

    written = {
        "metrics": str(paths.metrics),
        "model_metadata": str(paths.model_metadata),
        "baseline_report": str(paths.baseline_report),
    }
    if predictions is not None:
        written.update(
            write_prediction_partitions(
                predictions,
                paths,
                split_scheme=split_scheme,
                observed_col=observed_col,
                predicted_col=predicted_col,
                extra_columns=prediction_extra_columns,
            )
        )
    return written


def make_prediction_frame(
    frame: pd.DataFrame,
    *,
    split_scheme: str,
    predicted_values: Any,
    model_name: str,
    feature_set_name: str,
    observed_col: str = TARGET_COLUMN,
    predicted_col: str = "predicted_water_drainage_count",
    prediction_score_values: Any | None = None,
    extra_columns: Sequence[str] = (),
) -> pd.DataFrame:
    """Create a standardized prediction dataframe from a source frame."""

    require_pandas()
    split_col = split_column_for_scheme(split_scheme)
    required = [*KEY_COLUMNS, split_col, observed_col]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise BaselineError(f"Source frame missing required prediction columns: {missing}")

    out_cols = [*required]
    if BINARY_TARGET_COLUMN in frame.columns:
        out_cols.append(BINARY_TARGET_COLUMN)
    for col in extra_columns:
        if col in frame.columns and col not in out_cols:
            out_cols.append(col)

    out = frame[out_cols].copy()
    out[predicted_col] = predicted_values
    out["predicted_score"] = prediction_score_values if prediction_score_values is not None else predicted_values
    out["model_name"] = model_name
    out["feature_set_name"] = feature_set_name
    return out


def baseline_result_brief(result: Mapping[str, Any]) -> str:
    """Return concise baseline-run summary."""

    outputs = result.get("outputs", {})
    status = result.get("status", "completed")
    model_stage = result.get("model_stage")
    model_name = result.get("model_name")
    return (
        "Baseline run completed.\n"
        f"Status: {status}\n"
        f"Model stage: {model_stage}\n"
        f"Model name: {model_name}\n"
        f"Metrics: {outputs.get('metrics')}\n"
        f"Report: {outputs.get('baseline_report')}\n"
    )


__all__ = [
    "BENCHMARK_ID_DEFAULT",
    "BINARY_TARGET_COLUMN",
    "BaselineError",
    "BaselinePaths",
    "BaselineRunContext",
    "DATASET_VERSION_DEFAULT",
    "DEFAULT_BASELINE_ROOT",
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_PANEL_PATH",
    "DEFAULT_SPLIT_ASSIGNMENTS_PATH",
    "EVAL_PARTITIONS",
    "KEY_COLUMNS",
    "RETROSPECTIVE_TARGET_CONTAINING_COLUMNS",
    "SPLIT_COLUMNS",
    "SPLIT_TYPES",
    "STRICT_FORECASTING_FORBIDDEN_COLUMNS",
    "TARGET_COLUMN",
    "TARGET_DERIVED_FORBIDDEN_COLUMNS",
    "assert_no_forbidden_feature_columns",
    "baseline_result_brief",
    "build_model_metadata",
    "build_run_context",
    "dataframe_to_markdown_table",
    "evaluate_prediction_frame",
    "get_baseline_paths",
    "get_split_frame",
    "get_split_masks",
    "load_benchmark_frame",
    "load_config_and_inputs",
    "load_panel",
    "load_split_assignments",
    "make_prediction_frame",
    "merge_panel_with_splits",
    "prefix_metric_results",
    "read_table",
    "render_baseline_report",
    "resolve_baseline_root",
    "resolve_panel_path",
    "resolve_split_assignments_path",
    "split_column_for_scheme",
    "split_counts",
    "split_type_for_scheme",
    "validate_benchmark_frame",
    "validate_prediction_frame",
    "write_baseline_artifacts",
    "write_dataframe",
    "write_prediction_partitions",
]