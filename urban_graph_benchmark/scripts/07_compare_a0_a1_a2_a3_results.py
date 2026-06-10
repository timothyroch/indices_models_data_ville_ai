#!/usr/bin/env python3
"""
Compare A0/A1/A2/A3 baseline results for the Montréal 311 water/drainage benchmark.

This script reads already-produced baseline metric artifacts. It does not retrain
models. It creates consolidated tables, a narrative Markdown report, and optional
PNG visuals for the canonical A0/A1/A2/A3 comparison.

Primary purpose
---------------
Make the non-graph baseline hierarchy explicit before GraphSAGE/HGNN:

- A0: naive temporal/history baselines
- A1: raw static SVI direct ranking
- A2: calibrated SVI predictors
- A3: feature-parity tabular ML baselines

The script separates forecasting, rolling observed-history, and retrospective
explanatory settings whenever the information is available in the metric files.

Default output directory
------------------------
urban_graph_benchmark/outputs/<benchmark_id>/baselines/A0_A1_A2_A3_comparison/

Main outputs
------------
- a0_a1_a2_a3_metrics_long.csv
- a0_a1_a2_a3_test_canonical_table.csv
- a0_a1_a2_a3_test_leaderboard.csv
- a3_selected_candidates.csv
- a3_all_candidates_validation_test.csv
- a0_a1_a2_a3_comparison_report.md
- comparison_metadata.json
- plots/*.png
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

try:  # Optional. Used only for plot generation.
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None  # type: ignore[assignment]

try:  # Optional. Used only to read benchmark_id from config if available.
    import yaml
except Exception:  # pragma: no cover
    yaml = None  # type: ignore[assignment]


DEFAULT_CONFIG_PATH = "urban_graph_benchmark/configs/mtl_311_water_v0.1.yaml"
DEFAULT_BENCHMARK_ID = "mtl_311_water_v0"
STAGE_SLUG = "A0_A1_A2_A3_comparison"
TEST_SPLIT_NAME = "temporal_test"
VALIDATION_SPLIT_NAME = "temporal_validation"

KEY_METRICS = [
    "count_prediction__mae",
    "count_prediction__rmse",
    "count_prediction__mean_poisson_deviance",
    "tract_month_ranking__spearman_corr",
    "tract_month_ranking__ndcg_at_100",
    "tract_month_ranking__top_10pct_overlap_rate",
]

METRIC_SHORT_NAMES = {
    "count_prediction__mae": "MAE",
    "count_prediction__rmse": "RMSE",
    "count_prediction__mean_poisson_deviance": "Poisson deviance",
    "tract_month_ranking__spearman_corr": "Spearman",
    "tract_month_ranking__ndcg_at_100": "NDCG@100",
    "tract_month_ranking__top_10pct_overlap_rate": "Top-10% overlap",
}

METRIC_DIRECTIONS = {
    "count_prediction__mae": False,
    "count_prediction__rmse": False,
    "count_prediction__mean_poisson_deviance": False,
    "tract_month_ranking__spearman_corr": True,
    "tract_month_ranking__ndcg_at_100": True,
    "tract_month_ranking__top_10pct_overlap_rate": True,
}

PREDICTION_SETTING_ORDER = {
    "static_svi_direct_ranking_v0": 0,
    "forecasting_v0": 1,
    "one_step_observed_history_v0": 2,
    "rolling_observed_history_v0": 3,
    "retrospective_explanatory_v0": 4,
}


class ComparisonError(RuntimeError):
    """Raised when the comparison cannot be produced."""


@dataclass(frozen=True)
class InputPaths:
    """Resolved input paths."""

    repo_root: Path
    config_path: Path
    output_dir: Path
    a0_metrics: Path
    a1_metrics: Path
    a2_metrics: Path
    a3_metrics: Path
    a3_model_selection: Path | None
    a3_feature_importance: Path | None


@dataclass(frozen=True)
class CanonicalSpec:
    """Specification for one canonical comparison row."""

    label: str
    source_stage: str
    model_name: str | None = None
    feature_set_name: str | None = None
    model_family: str | None = None
    prediction_setting: str | None = None
    role: str = ""
    prefer_selected: bool = True


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Compare A0/A1/A2/A3 baseline results and generate tables/plots."
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help=f"Config path. Default: {DEFAULT_CONFIG_PATH}",
    )
    parser.add_argument(
        "--repo-root",
        default=None,
        help="Repository root. Defaults to automatic detection from the current directory.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to outputs/<benchmark_id>/baselines/A0_A1_A2_A3_comparison.",
    )
    parser.add_argument("--a0-metrics", default=None, help="Path to A0 metrics.csv.")
    parser.add_argument("--a1-metrics", default=None, help="Path to A1 metrics.csv.")
    parser.add_argument("--a2-metrics", default=None, help="Path to A2 metrics.csv.")
    parser.add_argument("--a3-metrics", default=None, help="Path to A3 metrics.csv.")
    parser.add_argument("--a3-model-selection", default=None, help="Path to A3 model_selection_audit.csv.")
    parser.add_argument("--a3-feature-importance", default=None, help="Path to A3 feature_importance.csv.")
    parser.add_argument(
        "--split-name",
        default=TEST_SPLIT_NAME,
        help=f"Test split name to summarize. Default: {TEST_SPLIT_NAME}",
    )
    parser.add_argument(
        "--validation-split-name",
        default=VALIDATION_SPLIT_NAME,
        help=f"Validation split name. Default: {VALIDATION_SPLIT_NAME}",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip matplotlib plot generation.",
    )
    return parser.parse_args()


def find_repo_root(explicit: str | Path | None = None) -> Path:
    """Find repository root."""

    if explicit is not None:
        root = Path(explicit).expanduser().resolve()
        if not root.exists():
            raise ComparisonError(f"repo root does not exist: {root}")
        return root

    current = Path.cwd().resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "urban_graph_benchmark").is_dir():
            return candidate

    raise ComparisonError(
        "Could not detect repository root. Run from the repo or pass --repo-root."
    )


def resolve_under_root(path: str | Path | None, root: Path, default: str | Path | None = None) -> Path | None:
    """Resolve a path under root when relative."""

    raw = path if path is not None else default
    if raw is None:
        return None
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = root / p
    return p.resolve()


def load_config(config_path: Path) -> dict[str, Any]:
    """Load YAML config if possible."""

    if not config_path.exists() or yaml is None:
        return {}
    with config_path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    return dict(loaded) if isinstance(loaded, Mapping) else {}


def benchmark_id_from_config(config: Mapping[str, Any]) -> str:
    """Extract benchmark_id with robust fallback."""

    value = config.get("benchmark_id") or config.get("dataset_id") or DEFAULT_BENCHMARK_ID
    return str(value)


def default_baseline_paths(root: Path, benchmark_id: str) -> dict[str, Path]:
    """Return default baseline artifact paths."""

    base = root / "urban_graph_benchmark" / "outputs" / benchmark_id / "baselines"
    return {
        "a0_metrics": base / "A0_naive_temporal" / "metrics.csv",
        "a1_metrics": base / "A1_svi_direct_ranking" / "metrics.csv",
        "a2_metrics": base / "A2_calibrated_svi" / "metrics.csv",
        "a3_metrics": base / "A3_feature_parity_tabular" / "metrics.csv",
        "a3_model_selection": base / "A3_feature_parity_tabular" / "model_selection_audit.csv",
        "a3_feature_importance": base / "A3_feature_parity_tabular" / "feature_importance.csv",
        "output_dir": base / STAGE_SLUG,
    }


def resolve_input_paths(args: argparse.Namespace) -> InputPaths:
    """Resolve all input/output paths."""

    root = find_repo_root(args.repo_root)
    config_path = resolve_under_root(args.config, root)
    if config_path is None:
        raise ComparisonError("Config path could not be resolved.")

    config = load_config(config_path)
    benchmark_id = benchmark_id_from_config(config)
    defaults = default_baseline_paths(root, benchmark_id)

    output_dir = resolve_under_root(args.output_dir, root, defaults["output_dir"])
    if output_dir is None:
        raise ComparisonError("Output directory could not be resolved.")

    paths = InputPaths(
        repo_root=root,
        config_path=config_path,
        output_dir=output_dir,
        a0_metrics=resolve_under_root(args.a0_metrics, root, defaults["a0_metrics"]),
        a1_metrics=resolve_under_root(args.a1_metrics, root, defaults["a1_metrics"]),
        a2_metrics=resolve_under_root(args.a2_metrics, root, defaults["a2_metrics"]),
        a3_metrics=resolve_under_root(args.a3_metrics, root, defaults["a3_metrics"]),
        a3_model_selection=resolve_under_root(
            args.a3_model_selection,
            root,
            defaults["a3_model_selection"],
        ),
        a3_feature_importance=resolve_under_root(
            args.a3_feature_importance,
            root,
            defaults["a3_feature_importance"],
        ),
    )

    required = {
        "A0 metrics": paths.a0_metrics,
        "A1 metrics": paths.a1_metrics,
        "A2 metrics": paths.a2_metrics,
        "A3 metrics": paths.a3_metrics,
    }
    missing = [f"{label}: {path}" for label, path in required.items() if path is None or not path.exists()]
    if missing:
        raise ComparisonError("Missing required input metric files:\n" + "\n".join(missing))

    return paths


def file_sha256(path: Path) -> str:
    """Compute SHA256 for metadata."""

    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_metrics(path: Path, source_stage: str) -> pd.DataFrame:
    """Read and minimally enrich a metrics CSV."""

    df = pd.read_csv(path)
    if df.empty:
        raise ComparisonError(f"Metric file is empty: {path}")
    if "metric_name" not in df.columns or "model_name" not in df.columns:
        raise ComparisonError(f"Metric file missing metric_name/model_name columns: {path}")

    out = df.copy()
    out["source_stage"] = source_stage
    out["source_metrics_path"] = str(path)

    if "metric_value" in out.columns:
        out["metric_value"] = pd.to_numeric(out["metric_value"], errors="coerce")
    if "n_rows" not in out.columns and "n_eval" in out.columns:
        out["n_rows"] = out["n_eval"]
    if "prediction_setting" not in out.columns:
        out["prediction_setting"] = out["model_name"].map(infer_prediction_setting)
    else:
        missing = out["prediction_setting"].isna() | (out["prediction_setting"].astype(str).str.len() == 0)
        out.loc[missing, "prediction_setting"] = out.loc[missing, "model_name"].map(infer_prediction_setting)
    if "model_family" not in out.columns:
        out["model_family"] = out["model_name"].map(infer_model_family)
    else:
        missing = out["model_family"].isna() | (out["model_family"].astype(str).str.len() == 0)
        out.loc[missing, "model_family"] = out.loc[missing, "model_name"].map(infer_model_family)
    if "feature_set_name" not in out.columns:
        out["feature_set_name"] = out["model_name"].map(infer_feature_set_name)
    else:
        missing = out["feature_set_name"].isna() | (out["feature_set_name"].astype(str).str.len() == 0)
        out.loc[missing, "feature_set_name"] = out.loc[missing, "model_name"].map(infer_feature_set_name)

    normalized = out["metric_name"].apply(normalize_metric_name).apply(pd.Series)
    out["metric_scope"] = normalized["scope"]
    out["metric_short_name"] = normalized["metric"]
    out["display_metric"] = out["metric_scope"] + "__" + out["metric_short_name"]
    out["higher_is_better"] = out.apply(
        lambda row: METRIC_DIRECTIONS.get(
            row["display_metric"],
            bool(row.get("higher_is_better", True)),
        ),
        axis=1,
    )

    # Drop duplicate aliases if both old/new top-10 names are present.
    dedup_cols = [
        "source_stage",
        "split_name",
        "model_name",
        "feature_set_name",
        "prediction_setting",
        "display_metric",
    ]
    available = [c for c in dedup_cols if c in out.columns]
    out = out.sort_values("metric_name").drop_duplicates(available, keep="last")

    return out.reset_index(drop=True)


def normalize_metric_name(metric_name: str) -> dict[str, str]:
    """Normalize historical metric-name variants."""

    name = str(metric_name)

    if name.startswith("count__"):
        return {"scope": "count_prediction", "metric": name.removeprefix("count__")}
    if name.startswith("count_prediction__"):
        return {"scope": "count_prediction", "metric": name.removeprefix("count_prediction__")}

    if name.startswith("ranking__"):
        metric = name.removeprefix("ranking__")
        if metric == "top_10pct_overlap_precision":
            metric = "top_10pct_overlap_rate"
        return {"scope": "tract_month_ranking", "metric": metric}

    if name.startswith("tract_month_ranking__"):
        metric = name.removeprefix("tract_month_ranking__")
        if metric == "top_10pct_overlap_precision":
            metric = "top_10pct_overlap_rate"
        return {"scope": "tract_month_ranking", "metric": metric}

    if name.startswith("tract_level_ranking__"):
        return {"scope": "tract_level_ranking", "metric": name.removeprefix("tract_level_ranking__")}

    if name.startswith("binary__"):
        return {"scope": "binary_diagnostic", "metric": name.removeprefix("binary__")}

    return {"scope": "other", "metric": name}


def infer_prediction_setting(model_name: str) -> str:
    """Infer prediction setting from model name when missing."""

    name = str(model_name)
    if "direct_ranking" in name:
        return "static_svi_direct_ranking_v0"
    if "reporting_retrospective" in name or "reporting_exposure_retrospective" in name:
        return "retrospective_explanatory_v0"
    if "previous_month" in name or "previous_year" in name:
        return "one_step_observed_history_v0"
    if name.startswith(("ridge_log_count__A3_", "random_forest_log_count__A3_", "hist_gradient_boosting")):
        if "retrospective" in name:
            return "retrospective_explanatory_v0"
        if "target_history" in name or "lagged_reporting" in name or "all_forecasting" in name:
            return "rolling_observed_history_v0"
    return "forecasting_v0"


def infer_model_family(model_name: str) -> str:
    """Infer model family from model name."""

    name = str(model_name)
    if name.startswith("ridge_log_count"):
        return "ridge_log_count"
    if name.startswith("random_forest_log_count"):
        return "random_forest_log_count"
    if name.startswith("hist_gradient_boosting_poisson"):
        return "hist_gradient_boosting_poisson"
    if name.startswith("A0"):
        return "naive_temporal"
    if name.startswith("A1"):
        return "static_svi_ranking"
    if name.startswith("A2"):
        return "ridge_log_count"
    return "unknown"


def infer_feature_set_name(model_name: str) -> str:
    """Infer feature set from known model-name patterns."""

    name = str(model_name)
    if "__" in name:
        parts = name.split("__")
        if len(parts) >= 2 and parts[1].startswith("A3_"):
            return parts[1]
        if len(parts) >= 2 and parts[0].startswith("A2_"):
            return parts[0]
        if len(parts) >= 2 and parts[0].startswith("A1_"):
            return parts[0]
    if name.startswith("A0_"):
        return name
    return ""


def combine_metrics(paths: InputPaths) -> pd.DataFrame:
    """Read and combine A0/A1/A2/A3 metrics."""

    frames = [
        read_metrics(paths.a0_metrics, "A0"),
        read_metrics(paths.a1_metrics, "A1"),
        read_metrics(paths.a2_metrics, "A2"),
        read_metrics(paths.a3_metrics, "A3"),
    ]
    combined = pd.concat(frames, ignore_index=True)
    combined["prediction_setting_order"] = combined["prediction_setting"].map(
        lambda x: PREDICTION_SETTING_ORDER.get(str(x), 99)
    )
    return combined


def load_optional_csv(path: Path | None) -> pd.DataFrame:
    """Read optional CSV or return empty DataFrame."""

    if path is None or not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def metrics_wide(
    metrics: pd.DataFrame,
    split_name: str,
    metrics_to_keep: Sequence[str] = KEY_METRICS,
) -> pd.DataFrame:
    """Create one row per model with key metrics as columns."""

    subset = metrics[
        (metrics["split_name"].astype(str) == split_name)
        & (metrics["display_metric"].isin(metrics_to_keep))
    ].copy()
    if subset.empty:
        return pd.DataFrame()

    id_cols = [
        "source_stage",
        "model_name",
        "model_family",
        "feature_set_name",
        "prediction_setting",
    ]
    id_cols = [c for c in id_cols if c in subset.columns]
    pivot = subset.pivot_table(
        index=id_cols,
        columns="display_metric",
        values="metric_value",
        aggfunc="first",
    ).reset_index()
    pivot.columns.name = None

    for metric in metrics_to_keep:
        if metric not in pivot.columns:
            pivot[metric] = math.nan

    pivot["prediction_setting_order"] = pivot["prediction_setting"].map(
        lambda x: PREDICTION_SETTING_ORDER.get(str(x), 99)
    )
    return pivot


def metric_value_for_model(
    metrics: pd.DataFrame,
    *,
    model_name: str,
    split_name: str,
    display_metric: str,
) -> float | None:
    """Fetch metric value for one model/split/display metric."""

    subset = metrics[
        (metrics["model_name"].astype(str) == str(model_name))
        & (metrics["split_name"].astype(str) == str(split_name))
        & (metrics["display_metric"].astype(str) == str(display_metric))
    ]
    if subset.empty:
        return None
    values = pd.to_numeric(subset["metric_value"], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.iloc[0])


def select_a3_model(
    metrics: pd.DataFrame,
    selection_audit: pd.DataFrame,
    *,
    feature_set_name: str,
    model_family: str,
    validation_split_name: str,
) -> str | None:
    """Select an A3 model by validation MAE for a feature set and family."""

    if not selection_audit.empty:
        subset = selection_audit[
            (selection_audit["feature_set_name"].astype(str) == feature_set_name)
            & (selection_audit["model_family"].astype(str) == model_family)
        ].copy()
        if not subset.empty:
            if "selected_for_test_summary" in subset.columns:
                selected = subset[subset["selected_for_test_summary"].astype(str).str.lower().isin(["true", "1"])]
                if not selected.empty:
                    return str(selected.iloc[0]["model_name"])
            if "validation_mae" in subset.columns:
                subset["validation_mae_numeric"] = pd.to_numeric(subset["validation_mae"], errors="coerce")
                subset = subset.dropna(subset=["validation_mae_numeric"])
                if not subset.empty:
                    return str(subset.sort_values("validation_mae_numeric").iloc[0]["model_name"])

    subset = metrics[
        (metrics["source_stage"] == "A3")
        & (metrics["feature_set_name"].astype(str) == feature_set_name)
        & (metrics["model_family"].astype(str) == model_family)
        & (metrics["split_name"].astype(str) == validation_split_name)
        & (metrics["display_metric"].astype(str) == "count_prediction__mae")
    ].copy()
    if subset.empty:
        return None
    subset["metric_value"] = pd.to_numeric(subset["metric_value"], errors="coerce")
    subset = subset.dropna(subset=["metric_value"])
    if subset.empty:
        return None
    return str(subset.sort_values("metric_value").iloc[0]["model_name"])


def select_a3_overall(
    selection_audit: pd.DataFrame,
    flag_col: str,
) -> str | None:
    """Select overall strict/retrospective model from A3 selection audit."""

    if selection_audit.empty or flag_col not in selection_audit.columns:
        return None
    subset = selection_audit[selection_audit[flag_col].astype(str).str.lower().isin(["true", "1"])].copy()
    if subset.empty:
        return None
    return str(subset.iloc[0]["model_name"])


def base_canonical_specs(
    metrics: pd.DataFrame,
    selection_audit: pd.DataFrame,
    validation_split_name: str,
) -> list[CanonicalSpec]:
    """Create canonical comparison specification list."""

    specs: list[CanonicalSpec] = [
        CanonicalSpec(
            label="A0 tract history",
            source_stage="A0",
            model_name="A0_3_tract_train_mean",
            role="strong naive forecasting baseline",
        ),
        CanonicalSpec(
            label="A0 seasonal tract history",
            source_stage="A0",
            model_name="A0_4_tract_month_of_year_train_mean",
            role="seasonal naive forecasting baseline",
        ),
        CanonicalSpec(
            label="A0 previous-month persistence",
            source_stage="A0",
            model_name="A0_5_previous_month_persistence",
            role="one-step observed-history baseline",
        ),
        CanonicalSpec(
            label="A1 raw SVI percentile",
            source_stage="A1",
            model_name="A1_svi_direct_ranking__svi_percentile",
            role="raw primary static SVI ranking",
        ),
        CanonicalSpec(
            label="A2 SVI + static percentile",
            source_stage="A2",
            model_name="A2_svi_plus_static__svi_percentile",
            role="calibrated primary SVI forecasting",
        ),
        CanonicalSpec(
            label="A2 SVI + reporting retrospective",
            source_stage="A2",
            model_name="A2_svi_plus_reporting_retrospective__svi_percentile",
            role="calibrated primary SVI retrospective",
        ),
    ]

    a3_specs = [
        ("A3 RF static SVI/calendar", "A3_static_svi_calendar_forecasting", "random_forest_log_count", "static/spatial nonlinear forecasting"),
        ("A3 HGB static SVI/calendar", "A3_static_svi_calendar_forecasting", "hist_gradient_boosting_poisson", "static/spatial Poisson boosting forecasting"),
        ("A3 RF target history", "A3_target_history_forecasting", "random_forest_log_count", "target-history nonlinear rolling forecast"),
        ("A3 RF lagged reporting", "A3_lagged_reporting_forecasting", "random_forest_log_count", "lagged reporting nonlinear rolling forecast"),
        ("A3 RF target + reporting history", "A3_target_history_lagged_reporting_forecasting", "random_forest_log_count", "target and reporting history nonlinear rolling forecast"),
        ("A3 RF all forecasting", "A3_all_forecasting", "random_forest_log_count", "main primary A3 rolling forecast"),
        ("A3 RF all forecasting + diagnostic SVI", "A3_all_forecasting_diagnostic_svi_expanded", "random_forest_log_count", "diagnostic SVI-expanded A3 rolling forecast"),
        ("A3 HGB all forecasting", "A3_all_forecasting", "hist_gradient_boosting_poisson", "main HGB A3 rolling forecast"),
        ("A3 RF retrospective", "A3_reporting_retrospective", "random_forest_log_count", "main primary A3 retrospective"),
        ("A3 RF retrospective + diagnostic SVI", "A3_reporting_retrospective_diagnostic_svi_expanded", "random_forest_log_count", "diagnostic SVI-expanded A3 retrospective"),
    ]

    for label, feature_set, family, role in a3_specs:
        model = select_a3_model(
            metrics,
            selection_audit,
            feature_set_name=feature_set,
            model_family=family,
            validation_split_name=validation_split_name,
        )
        if model:
            specs.append(
                CanonicalSpec(
                    label=label,
                    source_stage="A3",
                    model_name=model,
                    feature_set_name=feature_set,
                    model_family=family,
                    role=role,
                )
            )

    strict_overall = select_a3_overall(selection_audit, "selected_overall_strict_forecasting")
    if strict_overall:
        specs.append(
            CanonicalSpec(
                label="A3 selected strict/rolling model",
                source_stage="A3",
                model_name=strict_overall,
                role="validation-selected best strict/rolling A3 model",
            )
        )

    retro_overall = select_a3_overall(selection_audit, "selected_overall_retrospective")
    if retro_overall:
        specs.append(
            CanonicalSpec(
                label="A3 selected retrospective model",
                source_stage="A3",
                model_name=retro_overall,
                role="validation-selected best retrospective A3 model",
            )
        )

    # Deduplicate by model_name while preserving first descriptive label, except exact duplicates
    # in selected-overall rows add no new metric information.
    seen: set[str] = set()
    deduped: list[CanonicalSpec] = []
    for spec in specs:
        key = spec.model_name or f"{spec.feature_set_name}:{spec.model_family}:{spec.label}"
        if key in seen:
            continue
        seen.add(key)
        deduped.append(spec)

    return deduped


def canonical_table(
    metrics: pd.DataFrame,
    selection_audit: pd.DataFrame,
    *,
    split_name: str,
    validation_split_name: str,
) -> pd.DataFrame:
    """Build canonical test comparison table."""

    wide = metrics_wide(metrics, split_name)
    if wide.empty:
        return wide

    specs = base_canonical_specs(metrics, selection_audit, validation_split_name)
    rows: list[dict[str, Any]] = []

    for spec in specs:
        if spec.model_name is None:
            continue
        subset = wide[wide["model_name"].astype(str) == str(spec.model_name)].copy()
        if subset.empty:
            continue
        row = subset.iloc[0].to_dict()
        row["label"] = spec.label
        row["role"] = spec.role
        rows.append(row)

    table = pd.DataFrame(rows)
    if table.empty:
        return table

    display_cols = [
        "label",
        "source_stage",
        "prediction_setting",
        "model_family",
        "feature_set_name",
        "model_name",
        "role",
        *KEY_METRICS,
    ]
    display_cols = [c for c in display_cols if c in table.columns]
    table = table[display_cols].copy()

    table["prediction_setting_order"] = table["prediction_setting"].map(
        lambda x: PREDICTION_SETTING_ORDER.get(str(x), 99)
    )
    table = table.sort_values(["prediction_setting_order", "source_stage", "label"]).drop(columns=["prediction_setting_order"])
    return table.reset_index(drop=True)


def leaderboard_table(metrics: pd.DataFrame, split_name: str) -> pd.DataFrame:
    """Build leaderboard rows for each key metric."""

    wide = metrics_wide(metrics, split_name)
    if wide.empty:
        return pd.DataFrame()

    rows: list[pd.DataFrame] = []
    for metric in KEY_METRICS:
        part = wide.dropna(subset=[metric]).copy()
        if part.empty:
            continue
        higher = METRIC_DIRECTIONS.get(metric, True)
        part = part.sort_values(metric, ascending=not higher).head(20)
        part.insert(0, "rank", range(1, len(part) + 1))
        part.insert(0, "leaderboard_metric", metric)
        part.insert(1, "leaderboard_metric_label", METRIC_SHORT_NAMES.get(metric, metric))
        rows.append(part)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def a3_validation_test_table(metrics: pd.DataFrame, selection_audit: pd.DataFrame) -> pd.DataFrame:
    """Summarize A3 candidate validation/test metrics."""

    a3 = metrics[metrics["source_stage"] == "A3"].copy()
    if a3.empty:
        return pd.DataFrame()

    wanted = [
        "count_prediction__mae",
        "tract_month_ranking__spearman_corr",
        "tract_month_ranking__ndcg_at_100",
        "tract_month_ranking__top_10pct_overlap_rate",
    ]
    subset = a3[
        a3["split_name"].isin([VALIDATION_SPLIT_NAME, TEST_SPLIT_NAME])
        & a3["display_metric"].isin(wanted)
    ].copy()
    if subset.empty:
        return pd.DataFrame()

    subset["split_metric"] = subset["split_name"].astype(str).str.replace("temporal_", "", regex=False) + "__" + subset["display_metric"]
    id_cols = [
        "model_name",
        "model_family",
        "feature_set_name",
        "prediction_setting",
    ]
    table = subset.pivot_table(
        index=id_cols,
        columns="split_metric",
        values="metric_value",
        aggfunc="first",
    ).reset_index()
    table.columns.name = None

    if not selection_audit.empty:
        flags = [
            "model_name",
            "hyperparameter_id",
            "selected_for_test_summary",
            "selected_overall_strict_forecasting",
            "selected_overall_retrospective",
        ]
        flags = [c for c in flags if c in selection_audit.columns]
        table = table.merge(selection_audit[flags], on="model_name", how="left")

    sort_cols = [c for c in ["prediction_setting", "feature_set_name", "model_family", "test__count_prediction__mae"] if c in table.columns]
    if sort_cols:
        table = table.sort_values(sort_cols)
    return table.reset_index(drop=True)


def selected_a3_candidates(selection_audit: pd.DataFrame) -> pd.DataFrame:
    """Return selected A3 candidates from selection audit."""

    if selection_audit.empty:
        return pd.DataFrame()
    if "selected_for_test_summary" not in selection_audit.columns:
        return selection_audit.copy()
    selected = selection_audit[
        selection_audit["selected_for_test_summary"].astype(str).str.lower().isin(["true", "1"])
    ].copy()
    if selected.empty:
        return selected
    cols = [
        "model_name",
        "model_family",
        "feature_set_name",
        "prediction_setting",
        "hyperparameter_id",
        "validation_mae",
        "validation_spearman",
        "validation_top_10pct_overlap_rate",
        "test_mae",
        "test_spearman",
        "selected_overall_strict_forecasting",
        "selected_overall_retrospective",
        "selection_rule",
    ]
    cols = [c for c in cols if c in selected.columns]
    return selected[cols].reset_index(drop=True)


def format_float(value: Any, digits: int = 4) -> str:
    """Format float for Markdown."""

    if value is None:
        return "—"
    try:
        f = float(value)
    except Exception:
        return "—"
    if not math.isfinite(f):
        return "—"
    return f"{f:.{digits}f}"


def table_to_markdown(df: pd.DataFrame, max_rows: int = 80) -> str:
    """Render dataframe as Markdown."""

    if df.empty:
        return "_No rows._"
    display = df.head(max_rows).copy()
    try:
        return display.to_markdown(index=False)
    except Exception:
        return "```text\n" + display.to_string(index=False) + "\n```"


def short_label(label: str, max_len: int = 48) -> str:
    """Shorten a label for plot axes."""

    text = str(label)
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


def make_bar_plot(
    df: pd.DataFrame,
    *,
    metric_col: str,
    title: str,
    output_path: Path,
    higher_is_better: bool,
    max_rows: int = 16,
) -> None:
    """Create one horizontal bar chart."""

    if plt is None or df.empty or metric_col not in df.columns:
        return

    data = df.dropna(subset=[metric_col]).copy()
    if data.empty:
        return
    data = data.sort_values(metric_col, ascending=not higher_is_better).head(max_rows)
    labels = [short_label(x) for x in data["label"].astype(str).tolist()]
    values = pd.to_numeric(data[metric_col], errors="coerce").tolist()

    height = max(4.0, 0.38 * len(data) + 1.8)
    fig, ax = plt.subplots(figsize=(10.5, height))
    ax.barh(labels, values)
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel(METRIC_SHORT_NAMES.get(metric_col, metric_col))
    ax.grid(axis="x", alpha=0.3)

    for i, value in enumerate(values):
        if value is None or not math.isfinite(float(value)):
            continue
        ax.text(float(value), i, f" {float(value):.3f}", va="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_a3_validation_test_scatter(
    a3_table: pd.DataFrame,
    output_path: Path,
) -> None:
    """Create validation-vs-test MAE scatter for A3 candidates."""

    if plt is None or a3_table.empty:
        return
    x_col = "validation__count_prediction__mae"
    y_col = "test__count_prediction__mae"
    if x_col not in a3_table.columns or y_col not in a3_table.columns:
        return
    data = a3_table.dropna(subset=[x_col, y_col]).copy()
    if data.empty:
        return

    x = pd.to_numeric(data[x_col], errors="coerce")
    y = pd.to_numeric(data[y_col], errors="coerce")

    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    ax.scatter(x, y, alpha=0.75)
    min_value = float(min(x.min(), y.min()))
    max_value = float(max(x.max(), y.max()))
    ax.plot([min_value, max_value], [min_value, max_value], linestyle="--", linewidth=1)
    ax.set_title("A3 candidates: validation MAE vs test MAE")
    ax.set_xlabel("Validation MAE")
    ax.set_ylabel("Test MAE")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_a3_selected_by_feature_plot(
    selected: pd.DataFrame,
    output_path: Path,
) -> None:
    """Create selected A3 test MAE by feature set/family plot."""

    if plt is None or selected.empty:
        return
    if "test_mae" not in selected.columns:
        return
    data = selected.copy()
    data["test_mae"] = pd.to_numeric(data["test_mae"], errors="coerce")
    data = data.dropna(subset=["test_mae"])
    if data.empty:
        return
    data["label"] = data["feature_set_name"].astype(str) + " / " + data["model_family"].astype(str)
    data = data.sort_values("test_mae", ascending=True).head(20)

    fig, ax = plt.subplots(figsize=(11.0, max(4.0, 0.38 * len(data) + 1.8)))
    ax.barh([short_label(x, 64) for x in data["label"]], data["test_mae"])
    ax.invert_yaxis()
    ax.set_title("A3 selected candidates by test MAE")
    ax.set_xlabel("Test MAE")
    ax.grid(axis="x", alpha=0.3)
    for i, value in enumerate(data["test_mae"].tolist()):
        ax.text(float(value), i, f" {float(value):.3f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_plots(
    canonical: pd.DataFrame,
    a3_table: pd.DataFrame,
    selected_a3: pd.DataFrame,
    output_dir: Path,
    *,
    skip_plots: bool,
) -> dict[str, str]:
    """Generate plot PNGs and return artifact mapping."""

    if skip_plots or plt is None:
        return {}

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plots: dict[str, str] = {}

    plot_specs = [
        ("canonical_test_mae", "count_prediction__mae", "Canonical models: temporal-test MAE"),
        ("canonical_test_spearman", "tract_month_ranking__spearman_corr", "Canonical models: temporal-test Spearman"),
        ("canonical_test_ndcg_at_100", "tract_month_ranking__ndcg_at_100", "Canonical models: temporal-test NDCG@100"),
        ("canonical_test_top10_overlap", "tract_month_ranking__top_10pct_overlap_rate", "Canonical models: temporal-test top-10% overlap"),
    ]

    for key, metric, title in plot_specs:
        path = plot_dir / f"{key}.png"
        make_bar_plot(
            canonical,
            metric_col=metric,
            title=title,
            output_path=path,
            higher_is_better=METRIC_DIRECTIONS.get(metric, True),
        )
        if path.exists():
            plots[key] = str(path)

    scatter_path = plot_dir / "a3_validation_vs_test_mae.png"
    make_a3_validation_test_scatter(a3_table, scatter_path)
    if scatter_path.exists():
        plots["a3_validation_vs_test_mae"] = str(scatter_path)

    selected_path = plot_dir / "a3_selected_test_mae_by_feature_set_family.png"
    make_a3_selected_by_feature_plot(selected_a3, selected_path)
    if selected_path.exists():
        plots["a3_selected_test_mae_by_feature_set_family"] = str(selected_path)

    return plots


def best_overall_by_metric(canonical: pd.DataFrame) -> dict[str, str]:
    """Return best canonical label by metric."""

    winners: dict[str, str] = {}
    if canonical.empty:
        return winners
    for metric in KEY_METRICS:
        if metric not in canonical.columns:
            continue
        data = canonical.dropna(subset=[metric]).copy()
        if data.empty:
            continue
        higher = METRIC_DIRECTIONS.get(metric, True)
        best = data.sort_values(metric, ascending=not higher).iloc[0]
        winners[metric] = str(best["label"])
    return winners


def interpret_main_result(canonical: pd.DataFrame) -> list[str]:
    """Create concise interpretation bullets based on canonical table."""

    lines: list[str] = []
    if canonical.empty:
        return ["No canonical rows were available for interpretation."]

    def row_for(label: str) -> pd.Series | None:
        rows = canonical[canonical["label"] == label]
        if rows.empty:
            return None
        return rows.iloc[0]

    a0 = row_for("A0 tract history")
    a3_primary = row_for("A3 RF all forecasting")
    a3_diag = row_for("A3 RF all forecasting + diagnostic SVI")
    a3_retro = row_for("A3 RF retrospective")
    a3_static = row_for("A3 RF static SVI/calendar")

    if a0 is not None and a3_primary is not None:
        a0_mae = a0.get("count_prediction__mae")
        a3_mae = a3_primary.get("count_prediction__mae")
        a0_s = a0.get("tract_month_ranking__spearman_corr")
        a3_s = a3_primary.get("tract_month_ranking__spearman_corr")
        lines.append(
            "A3 RF all-forecasting nearly reaches A0 tract-history on count error "
            f"(MAE {format_float(a3_mae)} vs {format_float(a0_mae)}), "
            "but A0 remains stronger on ranking "
            f"(Spearman {format_float(a0_s)} vs {format_float(a3_s)})."
        )

    if a3_diag is not None and a3_primary is not None:
        lines.append(
            "The diagnostic SVI-expanded A3 model is essentially tied with the primary A3 model, "
            "so rank/class SVI should remain a robustness diagnostic rather than the main claim."
        )

    if a3_static is not None:
        lines.append(
            "The static SVI/calendar/spatial RF baseline is surprisingly strong, which means the "
            "tabular baseline already captures substantial static spatial structure."
        )

    if a3_retro is not None and a0 is not None:
        lines.append(
            "A3 retrospective improves count error by using same-month non-water 311 reporting, "
            "but it is explanatory/retrospective and should not be mixed with forecasting claims."
        )

    return lines


def render_report(
    *,
    generated_at: str,
    paths: InputPaths,
    metrics: pd.DataFrame,
    canonical: pd.DataFrame,
    leaderboard: pd.DataFrame,
    a3_selected: pd.DataFrame,
    a3_table: pd.DataFrame,
    plots: Mapping[str, str],
    split_name: str,
    validation_split_name: str,
) -> str:
    """Render Markdown comparison report."""

    lines: list[str] = []
    lines.append("# A0/A1/A2/A3 Baseline Comparison — Montréal 311 Water/Drainage v0\n")
    lines.append(f"Generated at: `{generated_at}`\n")
    lines.append(f"Test split: `{split_name}`\n")
    lines.append(f"Validation split used by A3 selection: `{validation_split_name}`\n")

    lines.append("## Purpose\n")
    lines.append(
        "This report compares already-produced A0, A1, A2, and A3 baseline metrics. "
        "It does not retrain models. Its goal is to establish the non-graph benchmark "
        "floor before GraphSAGE/HGNN.\n"
    )

    lines.append("## Prediction-setting guardrails\n")
    lines.append(
        "- `forecasting_v0`: static/calendar/train-derived information only.\n"
        "- `one_step_observed_history_v0`: observed-history baseline using prior observations.\n"
        "- `rolling_observed_history_v0`: A3 lag/rolling features; valid for rolling monthly forecasting, not for forecasting the whole future horizon from the train endpoint.\n"
        "- `retrospective_explanatory_v0`: same-month reporting controls; not a forecasting setting.\n"
    )

    lines.append("## Headline interpretation\n")
    for bullet in interpret_main_result(canonical):
        lines.append(f"- {bullet}")
    lines.append("")

    lines.append("## Canonical temporal-test comparison\n")
    show = canonical.copy()
    rename = {metric: METRIC_SHORT_NAMES.get(metric, metric) for metric in KEY_METRICS}
    show = show.rename(columns=rename)
    lines.append(table_to_markdown(show, max_rows=40))
    lines.append("")

    if plots:
        lines.append("## Visual summaries\n")
        rel_plot_items = []
        for key, path in plots.items():
            try:
                rel = Path(path).relative_to(paths.output_dir)
            except Exception:
                rel = Path(path)
            rel_plot_items.append((key, rel))
        for key, rel in rel_plot_items:
            title = key.replace("_", " ").title()
            lines.append(f"### {title}\n")
            lines.append(f"![{title}]({rel.as_posix()})\n")

    lines.append("## Best canonical model by metric\n")
    winners = best_overall_by_metric(canonical)
    if winners:
        lines.append("| Metric | Best canonical model |")
        lines.append("|---|---|")
        for metric, label in winners.items():
            lines.append(f"| {METRIC_SHORT_NAMES.get(metric, metric)} | `{label}` |")
        lines.append("")
    else:
        lines.append("_No winners available._\n")

    lines.append("## A3 validation-selected candidates\n")
    lines.append(
        "A3 candidate selection uses validation MAE. Test metrics are reported after selection.\n"
    )
    lines.append(table_to_markdown(a3_selected, max_rows=80))
    lines.append("")

    lines.append("## Test leaderboard preview\n")
    keep_cols = [
        "leaderboard_metric_label",
        "rank",
        "source_stage",
        "prediction_setting",
        "model_family",
        "feature_set_name",
        "model_name",
        "count_prediction__mae",
        "tract_month_ranking__spearman_corr",
        "tract_month_ranking__ndcg_at_100",
        "tract_month_ranking__top_10pct_overlap_rate",
    ]
    keep_cols = [c for c in keep_cols if c in leaderboard.columns]
    lines.append(table_to_markdown(leaderboard[keep_cols], max_rows=120))
    lines.append("")

    lines.append("## A3 all-candidate validation/test table preview\n")
    lines.append(table_to_markdown(a3_table, max_rows=100))
    lines.append("")

    lines.append("## Output artifacts\n")
    artifact_rows = [
        ("metrics_long", paths.output_dir / "a0_a1_a2_a3_metrics_long.csv"),
        ("test_canonical_table", paths.output_dir / "a0_a1_a2_a3_test_canonical_table.csv"),
        ("test_leaderboard", paths.output_dir / "a0_a1_a2_a3_test_leaderboard.csv"),
        ("a3_selected_candidates", paths.output_dir / "a3_selected_candidates.csv"),
        ("a3_all_candidates_validation_test", paths.output_dir / "a3_all_candidates_validation_test.csv"),
        ("comparison_report", paths.output_dir / "a0_a1_a2_a3_comparison_report.md"),
        ("comparison_metadata", paths.output_dir / "comparison_metadata.json"),
    ]
    lines.append("| Artifact | Path |")
    lines.append("|---|---|")
    for label, path in artifact_rows:
        lines.append(f"| `{label}` | `{path}` |")
    for label, path in plots.items():
        lines.append(f"| `plot:{label}` | `{path}` |")
    lines.append("")

    lines.append("## Input artifact hashes\n")
    lines.append("| Input | Path | SHA256 |")
    lines.append("|---|---|---|")
    for label, path in [
        ("A0 metrics", paths.a0_metrics),
        ("A1 metrics", paths.a1_metrics),
        ("A2 metrics", paths.a2_metrics),
        ("A3 metrics", paths.a3_metrics),
    ]:
        lines.append(f"| `{label}` | `{path}` | `{file_sha256(path)}` |")
    lines.append("")

    lines.append("## Bottom line\n")
    lines.append(
        "A3 gives a strong non-graph ML floor. Future graph models should be compared "
        "against both A0 tract history and the best A3 strict/rolling tabular model. "
        "A3 retrospective results are useful for explanation but not for forecasting claims.\n"
    )

    return "\n".join(lines)


def build_metadata(
    *,
    generated_at: str,
    paths: InputPaths,
    metrics: pd.DataFrame,
    canonical: pd.DataFrame,
    leaderboard: pd.DataFrame,
    a3_table: pd.DataFrame,
    plots: Mapping[str, str],
    split_name: str,
    validation_split_name: str,
) -> dict[str, Any]:
    """Build JSON metadata."""

    inputs = {
        "a0_metrics": str(paths.a0_metrics),
        "a1_metrics": str(paths.a1_metrics),
        "a2_metrics": str(paths.a2_metrics),
        "a3_metrics": str(paths.a3_metrics),
        "a3_model_selection": str(paths.a3_model_selection) if paths.a3_model_selection else None,
        "a3_feature_importance": str(paths.a3_feature_importance) if paths.a3_feature_importance else None,
    }
    hashes = {
        key: file_sha256(Path(value))
        for key, value in inputs.items()
        if value is not None and Path(value).exists()
    }

    return {
        "generated_at": generated_at,
        "stage_slug": STAGE_SLUG,
        "config_path": str(paths.config_path),
        "repo_root": str(paths.repo_root),
        "output_dir": str(paths.output_dir),
        "split_name": split_name,
        "validation_split_name": validation_split_name,
        "input_paths": inputs,
        "input_sha256": hashes,
        "metric_rows": int(len(metrics)),
        "canonical_rows": int(len(canonical)),
        "leaderboard_rows": int(len(leaderboard)),
        "a3_candidate_rows": int(len(a3_table)),
        "plots": dict(plots),
        "notes": (
            "This comparison reads existing metrics only. It does not retrain models. "
            "A3 model selection is validation-based when model_selection_audit.csv is available."
        ),
    }


def run_compare(args: argparse.Namespace) -> dict[str, Any]:
    """Run comparison and write outputs."""

    paths = resolve_input_paths(args)
    paths.output_dir.mkdir(parents=True, exist_ok=True)

    metrics = combine_metrics(paths)
    selection_audit = load_optional_csv(paths.a3_model_selection)

    canonical = canonical_table(
        metrics,
        selection_audit,
        split_name=args.split_name,
        validation_split_name=args.validation_split_name,
    )
    leaderboard = leaderboard_table(metrics, args.split_name)
    a3_selected = selected_a3_candidates(selection_audit)
    a3_table = a3_validation_test_table(metrics, selection_audit)

    plots = make_plots(
        canonical,
        a3_table,
        a3_selected,
        paths.output_dir,
        skip_plots=args.no_plots,
    )

    generated_at = datetime.now(timezone.utc).isoformat()

    out_metrics = paths.output_dir / "a0_a1_a2_a3_metrics_long.csv"
    out_canonical = paths.output_dir / "a0_a1_a2_a3_test_canonical_table.csv"
    out_leaderboard = paths.output_dir / "a0_a1_a2_a3_test_leaderboard.csv"
    out_a3_selected = paths.output_dir / "a3_selected_candidates.csv"
    out_a3_table = paths.output_dir / "a3_all_candidates_validation_test.csv"
    out_report = paths.output_dir / "a0_a1_a2_a3_comparison_report.md"
    out_metadata = paths.output_dir / "comparison_metadata.json"

    metrics.to_csv(out_metrics, index=False)
    canonical.to_csv(out_canonical, index=False)
    leaderboard.to_csv(out_leaderboard, index=False)
    a3_selected.to_csv(out_a3_selected, index=False)
    a3_table.to_csv(out_a3_table, index=False)

    report = render_report(
        generated_at=generated_at,
        paths=paths,
        metrics=metrics,
        canonical=canonical,
        leaderboard=leaderboard,
        a3_selected=a3_selected,
        a3_table=a3_table,
        plots=plots,
        split_name=args.split_name,
        validation_split_name=args.validation_split_name,
    )
    out_report.write_text(report, encoding="utf-8")

    metadata = build_metadata(
        generated_at=generated_at,
        paths=paths,
        metrics=metrics,
        canonical=canonical,
        leaderboard=leaderboard,
        a3_table=a3_table,
        plots=plots,
        split_name=args.split_name,
        validation_split_name=args.validation_split_name,
    )
    out_metadata.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "status": "completed",
        "output_dir": str(paths.output_dir),
        "metric_rows": int(len(metrics)),
        "canonical_rows": int(len(canonical)),
        "leaderboard_rows": int(len(leaderboard)),
        "a3_candidate_rows": int(len(a3_table)),
        "plots": plots,
        "outputs": {
            "metrics_long": str(out_metrics),
            "test_canonical_table": str(out_canonical),
            "test_leaderboard": str(out_leaderboard),
            "a3_selected_candidates": str(out_a3_selected),
            "a3_all_candidates_validation_test": str(out_a3_table),
            "comparison_report": str(out_report),
            "comparison_metadata": str(out_metadata),
        },
    }


def brief(result: Mapping[str, Any]) -> str:
    """Return run summary."""

    lines = [
        "A0/A1/A2/A3 comparison completed.",
        f"Status: {result.get('status')}",
        f"Metric rows: {result.get('metric_rows')}",
        f"Canonical rows: {result.get('canonical_rows')}",
        f"Leaderboard rows: {result.get('leaderboard_rows')}",
        f"A3 candidate rows: {result.get('a3_candidate_rows')}",
        f"Plots: {len(result.get('plots') or {})}",
        f"Output directory: {result.get('output_dir')}",
    ]
    outputs = result.get("outputs") or {}
    for key, path in outputs.items():
        lines.append(f"  {key}: {path}")
    if result.get("plots"):
        lines.append("Plot artifacts:")
        for key, path in (result.get("plots") or {}).items():
            lines.append(f"  {key}: {path}")
    return "\n".join(lines)


def main() -> None:
    """CLI entry point."""

    args = parse_args()
    result = run_compare(args)
    print(brief(result))


if __name__ == "__main__":
    main()
