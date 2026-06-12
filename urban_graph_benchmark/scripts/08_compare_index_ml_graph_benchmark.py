#!/usr/bin/env python3
"""
Compare index, tabular-ML, and graph/neural benchmark results.

This script is a post-hoc benchmark consolidation layer. It reads already-produced
artifacts and does not train any model.

Repository role
---------------
Place this file next to the previous comparison scripts, for example:

    urban_graph_benchmark/scripts/08_compare_index_ml_graph_benchmark.py

The script consolidates the current VILLE_IA benchmark hierarchy:

- composite vulnerability indices, currently SVI-style and optionally SoVI-style;
- calibrated index baselines;
- A3 feature-parity tabular ML baselines;
- G1 / G1.5 graph-neural baselines and their no-edge/placebo controls.

Main outputs
------------
- index_ml_graph_metrics_long.csv
- benchmark_comparison.csv
- benchmark_comparison_compact.csv
- metric_winners.csv
- family_margin_table.csv
- missing_input_audit.csv
- benchmark_interpretation.md
- comparison_metadata.json
- plots/*.png, unless --no-plots is used

Interpretation policy
---------------------
The report deliberately separates three claims:

1. Static composite-index performance.
2. Supervised tabular ML performance.
3. Graph/neural performance relative to no-edge and placebo controls.

A graph/neural model beating SVI/SoVI is not by itself evidence that the spatial
edge topology is meaningful. A topology-specific claim requires comparison to
no_edges and random_spatial_placebo controls.
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

try:  # Optional: plots only.
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None  # type: ignore[assignment]

try:  # Optional: config parsing only.
    import yaml
except Exception:  # pragma: no cover
    yaml = None  # type: ignore[assignment]


DEFAULT_CONFIG_PATH = "urban_graph_benchmark/configs/mtl_311_water_v0.1.yaml"
DEFAULT_BENCHMARK_ID = "mtl_311_water_v0"
DEFAULT_OUTPUT_DIR = (
    "urban_graph_benchmark/outputs/mtl_311_water_v0/comparisons/"
    "08_index_ml_graph_benchmark"
)
DEFAULT_BASELINES_DIR = "urban_graph_benchmark/outputs/mtl_311_water_v0/baselines"

DEFAULT_A0_METRICS = f"{DEFAULT_BASELINES_DIR}/A0_naive_temporal/metrics.csv"
DEFAULT_A1_SVI_METRICS = f"{DEFAULT_BASELINES_DIR}/A1_svi_direct_ranking/metrics.csv"
DEFAULT_A2_SVI_METRICS = f"{DEFAULT_BASELINES_DIR}/A2_calibrated_svi/metrics.csv"
DEFAULT_A3_SPATIAL_DIR = f"{DEFAULT_BASELINES_DIR}/A3_feature_parity_tabular_spatial_block"
DEFAULT_G1_SPATIAL_DIR = f"{DEFAULT_BASELINES_DIR}/G1_spatiotemporal_gnn_spatial_core_ndcg_monitor"
DEFAULT_G1_5_DIR = f"{DEFAULT_BASELINES_DIR}/G1_5_validation_sweep_spatial_ndcg"
DEFAULT_SOVI_METRICS = f"{DEFAULT_BASELINES_DIR}/A1_sovi_direct_ranking/metrics.csv"

STAGE_SLUG = "08_index_ml_graph_benchmark"

CANONICAL_METRICS = [
    "mae",
    "rmse",
    "mean_poisson_deviance",
    "spearman",
    "ndcg_at_100",
    "top_10pct_overlap_rate",
]

METRIC_LABELS = {
    "mae": "MAE",
    "rmse": "RMSE",
    "mean_poisson_deviance": "Poisson deviance",
    "spearman": "Spearman",
    "kendall": "Kendall",
    "ndcg_at_100": "NDCG@100",
    "top_10pct_overlap_rate": "Top-10% overlap",
}

METRIC_DIRECTIONS = {
    "mae": False,
    "rmse": False,
    "mean_poisson_deviance": False,
    "spearman": True,
    "kendall": True,
    "ndcg_at_100": True,
    "top_10pct_overlap_rate": True,
}

# Canonical long-metric aliases used across A1/A2/A3/G1 outputs.
DISPLAY_TO_CANONICAL = {
    "count_prediction__mae": "mae",
    "count_prediction__rmse": "rmse",
    "count_prediction__mean_poisson_deviance": "mean_poisson_deviance",
    "tract_month_ranking__spearman_corr": "spearman",
    "tract_month_ranking__kendall_corr": "kendall",
    "tract_month_ranking__ndcg_at_100": "ndcg_at_100",
    "tract_month_ranking__top_10pct_overlap_rate": "top_10pct_overlap_rate",
    "tract_month_ranking__top_10pct_overlap_precision": "top_10pct_overlap_rate",
}

G15_TO_CANONICAL = {
    "test_mae": "mae",
    "test_rmse": "rmse",
    "test_mean_poisson_deviance": "mean_poisson_deviance",
    "test_spearman": "spearman",
    "test_kendall": "kendall",
    "test_ndcg_at_100": "ndcg_at_100",
    "test_top_10pct_overlap_rate": "top_10pct_overlap_rate",
}

PREFERRED_TEST_SPLITS = (
    "test",
    "spatial_block_test",
    "temporal_test",
    "validation",  # last-resort diagnostic fallback only
)


class BenchmarkComparisonError(RuntimeError):
    """Raised when the benchmark comparison cannot be produced."""


@dataclass(frozen=True)
class ResolvedPaths:
    """Resolved input/output paths."""

    repo_root: Path
    config_path: Path
    output_dir: Path
    a0_metrics: Path | None
    a1_svi_metrics: Path | None
    a2_svi_metrics: Path | None
    sovi_metrics: Path | None
    a3_spatial_dir: Path | None
    g1_spatial_dir: Path | None
    g1_5_dir: Path | None


@dataclass(frozen=True)
class CandidateSpec:
    """Specification for a row extracted from a long metrics CSV."""

    label: str
    source_stage: str
    comparison_group: str
    family: str
    model_name_exact: str | None = None
    model_name_contains: tuple[str, ...] = ()
    role: str = ""
    selection_policy: str = "predefined canonical model"
    is_index: bool = False
    is_tabular_ml: bool = False
    is_graph_family: bool = False
    is_control_family: bool = False


# ---------------------------------------------------------------------------
# CLI and path helpers
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare index, tabular-ML, and graph/neural benchmark outputs."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)

    parser.add_argument("--a0-metrics", default=DEFAULT_A0_METRICS)
    parser.add_argument("--a1-svi-metrics", default=DEFAULT_A1_SVI_METRICS)
    parser.add_argument("--a2-svi-metrics", default=DEFAULT_A2_SVI_METRICS)
    parser.add_argument("--sovi-metrics", default=DEFAULT_SOVI_METRICS)
    parser.add_argument("--a3-spatial-dir", default=DEFAULT_A3_SPATIAL_DIR)
    parser.add_argument("--g1-spatial-dir", default=DEFAULT_G1_SPATIAL_DIR)
    parser.add_argument("--g1-5-dir", default=DEFAULT_G1_5_DIR)

    parser.add_argument(
        "--preferred-test-splits",
        default=",".join(PREFERRED_TEST_SPLITS),
        help=(
            "Comma-separated split names to try, in order, when extracting test metrics "
            "from long metrics files."
        ),
    )
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--strict-inputs",
        action="store_true",
        help="Fail if optional inputs are missing instead of writing a missing-input audit.",
    )
    return parser.parse_args()


def find_repo_root(explicit: str | Path | None = None) -> Path:
    """Find repository root by looking for urban_graph_benchmark/."""

    if explicit is not None:
        root = Path(explicit).expanduser().resolve()
        if not root.exists():
            raise BenchmarkComparisonError(f"repo root does not exist: {root}")
        return root

    current = Path.cwd().resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "urban_graph_benchmark").is_dir():
            return candidate

    raise BenchmarkComparisonError(
        "Could not detect repository root. Run from the repository root or pass --repo-root."
    )


def resolve_optional_path(root: Path, value: str | Path | None) -> Path | None:
    """Resolve a path; return None for blank strings."""

    if value is None:
        return None
    raw = str(value).strip()
    if raw == "" or raw.lower() in {"none", "null", "skip"}:
        return None
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def load_config(config_path: Path) -> dict[str, Any]:
    """Load config if possible."""

    if not config_path.exists() or yaml is None:
        return {}
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return dict(data) if isinstance(data, Mapping) else {}


def resolve_paths(args: argparse.Namespace) -> ResolvedPaths:
    root = find_repo_root(args.repo_root)
    config_path = resolve_optional_path(root, args.config)
    if config_path is None:
        raise BenchmarkComparisonError("Config path resolved to None.")
    output_dir = resolve_optional_path(root, args.output_dir)
    if output_dir is None:
        raise BenchmarkComparisonError("Output path resolved to None.")

    paths = ResolvedPaths(
        repo_root=root,
        config_path=config_path,
        output_dir=output_dir,
        a0_metrics=resolve_optional_path(root, args.a0_metrics),
        a1_svi_metrics=resolve_optional_path(root, args.a1_svi_metrics),
        a2_svi_metrics=resolve_optional_path(root, args.a2_svi_metrics),
        sovi_metrics=resolve_optional_path(root, args.sovi_metrics),
        a3_spatial_dir=resolve_optional_path(root, args.a3_spatial_dir),
        g1_spatial_dir=resolve_optional_path(root, args.g1_spatial_dir),
        g1_5_dir=resolve_optional_path(root, args.g1_5_dir),
    )
    return paths


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_csv_list(value: str | None, default: Sequence[str]) -> tuple[str, ...]:
    if value is None:
        return tuple(default)
    items = [x.strip() for x in str(value).split(",") if x.strip()]
    return tuple(items) if items else tuple(default)


def path_exists(path: Path | None) -> bool:
    return path is not None and path.exists()


# ---------------------------------------------------------------------------
# Metrics normalization and extraction
# ---------------------------------------------------------------------------


def normalize_metric_name(metric_name: str) -> str:
    """Normalize metric-name variants to a canonical display metric."""

    name = str(metric_name)

    if name.startswith("count__"):
        return "count_prediction__" + name.removeprefix("count__")
    if name.startswith("count_prediction__"):
        return name

    if name.startswith("ranking__"):
        metric = name.removeprefix("ranking__")
        if metric == "top_10pct_overlap_precision":
            metric = "top_10pct_overlap_rate"
        return "tract_month_ranking__" + metric

    if name.startswith("tract_month_ranking__"):
        metric = name.removeprefix("tract_month_ranking__")
        if metric == "top_10pct_overlap_precision":
            metric = "top_10pct_overlap_rate"
        return "tract_month_ranking__" + metric

    return name


def canonical_metric_from_display(display_metric: str) -> str | None:
    return DISPLAY_TO_CANONICAL.get(str(display_metric))


def read_long_metrics(path: Path, source_stage: str) -> pd.DataFrame:
    """Read a long metrics.csv file and add normalized fields."""

    df = pd.read_csv(path)
    if df.empty:
        raise BenchmarkComparisonError(f"Metric file is empty: {path}")
    required = {"model_name", "metric_name", "metric_value"}
    missing = required - set(df.columns)
    if missing:
        raise BenchmarkComparisonError(f"Metric file {path} is missing columns: {sorted(missing)}")

    out = df.copy()
    out["source_stage"] = source_stage
    out["source_metrics_path"] = str(path)
    out["metric_value"] = pd.to_numeric(out["metric_value"], errors="coerce")
    if "split_name" not in out.columns:
        out["split_name"] = "test"
    out["display_metric"] = out["metric_name"].map(normalize_metric_name)
    out["canonical_metric"] = out["display_metric"].map(canonical_metric_from_display)
    out["higher_is_better"] = out["canonical_metric"].map(
        lambda m: METRIC_DIRECTIONS.get(str(m), True) if pd.notna(m) else True
    )
    return out.reset_index(drop=True)


def available_split_for_model(
    metrics: pd.DataFrame,
    *,
    model_name: str,
    preferred_splits: Sequence[str],
) -> str | None:
    """Return the first preferred split available for model_name."""

    sub = metrics[metrics["model_name"].astype(str) == str(model_name)]
    if sub.empty:
        return None
    available = set(sub["split_name"].astype(str))
    for split in preferred_splits:
        if split in available:
            return split
    # Last fallback: any split containing 'test', then any available split.
    test_like = sorted([s for s in available if "test" in s.lower()])
    if test_like:
        return test_like[0]
    return sorted(available)[0] if available else None


def metrics_for_model(
    metrics: pd.DataFrame,
    *,
    model_name: str,
    preferred_splits: Sequence[str],
) -> dict[str, Any] | None:
    """Extract canonical metrics for one model from a long metrics dataframe."""

    split = available_split_for_model(metrics, model_name=model_name, preferred_splits=preferred_splits)
    if split is None:
        return None
    sub = metrics[
        (metrics["model_name"].astype(str) == str(model_name))
        & (metrics["split_name"].astype(str) == str(split))
        & (metrics["canonical_metric"].notna())
    ].copy()
    if sub.empty:
        return None

    row: dict[str, Any] = {"model_name": model_name, "split_name": split}
    for metric in CANONICAL_METRICS:
        vals = sub.loc[sub["canonical_metric"].astype(str) == metric, "metric_value"].dropna()
        row[metric] = float(vals.iloc[0]) if not vals.empty else math.nan
    return row


def find_model_name(
    metrics: pd.DataFrame,
    *,
    exact: str | None = None,
    contains: Sequence[str] = (),
) -> str | None:
    """Find a model by exact name or case-insensitive contains tokens."""

    names = list(metrics["model_name"].dropna().astype(str).unique())
    if exact is not None and exact in names:
        return exact
    if contains:
        tokens = [t.lower() for t in contains]
        for name in names:
            low = name.lower()
            if all(t in low for t in tokens):
                return name
    return None


def build_row_from_long_metrics(
    metrics: pd.DataFrame,
    spec: CandidateSpec,
    preferred_splits: Sequence[str],
) -> tuple[dict[str, Any] | None, str | None]:
    """Build comparison row for one CandidateSpec."""

    model_name = find_model_name(
        metrics,
        exact=spec.model_name_exact,
        contains=spec.model_name_contains,
    )
    if model_name is None:
        return None, f"Could not find model for {spec.label!r}."
    extracted = metrics_for_model(metrics, model_name=model_name, preferred_splits=preferred_splits)
    if extracted is None:
        return None, f"Could not extract test metrics for {spec.label!r} model {model_name!r}."

    row = {
        "label": spec.label,
        "comparison_group": spec.comparison_group,
        "source_stage": spec.source_stage,
        "source": spec.source_stage,
        "family": spec.family,
        "model_name": model_name,
        "split_name": extracted.get("split_name"),
        "selection_policy": spec.selection_policy,
        "role": spec.role,
        "is_index": spec.is_index,
        "is_tabular_ml": spec.is_tabular_ml,
        "is_graph_family": spec.is_graph_family,
        "is_control_family": spec.is_control_family,
    }
    for metric in CANONICAL_METRICS:
        row[metric] = extracted.get(metric, math.nan)
    return row, None


# ---------------------------------------------------------------------------
# Source-specific loading
# ---------------------------------------------------------------------------


def index_candidate_specs() -> list[CandidateSpec]:
    """Canonical SVI/A2/SoVI specs. Some may be absent and reported as missing."""

    return [
        CandidateSpec(
            label="A1 raw SVI percentile",
            source_stage="A1_SVI",
            comparison_group="Composite index",
            family="SVI",
            model_name_exact="A1_svi_direct_ranking__svi_percentile",
            model_name_contains=("svi", "percentile"),
            role="Raw static SVI-style composite score used directly as risk ranking.",
            selection_policy="predefined primary SVI score",
            is_index=True,
        ),
        CandidateSpec(
            label="A1 raw SVI class",
            source_stage="A1_SVI",
            comparison_group="Composite index",
            family="SVI",
            model_name_exact="A1_svi_direct_ranking__svi_class",
            model_name_contains=("svi", "class"),
            role="Raw static SVI-style vulnerability class used directly as risk ranking.",
            selection_policy="diagnostic SVI class score",
            is_index=True,
        ),
        CandidateSpec(
            label="A2 calibrated SVI static",
            source_stage="A2_SVI",
            comparison_group="Calibrated index",
            family="SVI + calibration",
            model_name_exact="A2_svi_plus_static__svi_percentile",
            model_name_contains=("A2", "svi", "static", "percentile"),
            role="Supervised calibration of static SVI-style score.",
            selection_policy="predefined calibrated SVI model",
            is_index=True,
        ),
        CandidateSpec(
            label="A2 calibrated SVI retrospective",
            source_stage="A2_SVI",
            comparison_group="Calibrated index",
            family="SVI + retrospective reporting",
            model_name_exact="A2_svi_plus_reporting_retrospective__svi_percentile",
            model_name_contains=("A2", "svi", "reporting", "retrospective"),
            role="Retrospective calibrated SVI/reporting diagnostic.",
            selection_policy="diagnostic retrospective calibrated SVI model",
            is_index=True,
        ),
    ]


def sovi_candidate_specs() -> list[CandidateSpec]:
    return [
        CandidateSpec(
            label="SoVI-like composite index",
            source_stage="SOVI",
            comparison_group="Composite index",
            family="SoVI",
            model_name_contains=("sovi",),
            role="SoVI-like static composite score; optional until SoVI benchmark is produced.",
            selection_policy="first available SoVI-like score",
            is_index=True,
        )
    ]


def a0_candidate_specs() -> list[CandidateSpec]:
    return [
        CandidateSpec(
            label="A0 tract historical mean",
            source_stage="A0",
            comparison_group="Naive temporal baseline",
            family="A0",
            model_name_exact="A0_3_tract_train_mean",
            model_name_contains=("A0_3",),
            role="Strong naive tract-history baseline.",
            selection_policy="predefined strong A0 baseline",
        )
    ]


def load_a3_selected_from_dir(a3_dir: Path, preferred_splits: Sequence[str]) -> tuple[list[dict[str, Any]], list[str]]:
    """Load the frozen selected A3 spatial-block row."""

    missing: list[str] = []
    metrics_path = a3_dir / "metrics.csv"
    selection_path = a3_dir / "model_selection_audit.csv"
    if not metrics_path.exists():
        return [], [f"Missing A3 metrics: {metrics_path}"]

    metrics = read_long_metrics(metrics_path, "A3_spatial")
    selection = pd.read_csv(selection_path) if selection_path.exists() else pd.DataFrame()

    model_name: str | None = None
    selection_policy = "A3 spatial selected model"

    if not selection.empty:
        selected = selection.copy()
        for flag in ["selected_overall_for_split", "selected_for_test_summary"]:
            if flag in selected.columns:
                tmp = selected[selected[flag].astype(str).str.lower().isin({"true", "1"})].copy()
                if not tmp.empty:
                    selected = tmp
                    break
        if "validation_mae" in selected.columns:
            selected["validation_mae_num"] = pd.to_numeric(selected["validation_mae"], errors="coerce")
            selected = selected.sort_values("validation_mae_num", na_position="last")
        if not selected.empty and "model_name" in selected.columns:
            model_name = str(selected.iloc[0]["model_name"])
            selection_policy = "A3 validation-selected spatial-block model"

    if model_name is None:
        model_name = find_model_name(
            metrics,
            exact="hist_gradient_boosting_poisson__A3_lagged_reporting_forecasting__hgb_poisson_02",
            contains=("hist_gradient_boosting", "lagged_reporting"),
        )
        selection_policy = "fallback to known A3 spatial HGB candidate"

    if model_name is None:
        return [], [f"Could not identify selected A3 model in {a3_dir}"]

    extracted = metrics_for_model(metrics, model_name=model_name, preferred_splits=preferred_splits)
    if extracted is None:
        return [], [f"Could not extract metrics for selected A3 model {model_name!r}"]

    row: dict[str, Any] = {
        "label": "A3 selected tabular ML",
        "comparison_group": "Tabular ML",
        "source_stage": "A3",
        "source": "A3_spatial_block",
        "family": "A3 tabular ML",
        "model_name": model_name,
        "split_name": extracted.get("split_name"),
        "selection_policy": selection_policy,
        "role": "Frozen feature-parity tabular ML baseline for spatial-block comparison.",
        "is_index": False,
        "is_tabular_ml": True,
        "is_graph_family": False,
        "is_control_family": False,
    }
    for metric in CANONICAL_METRICS:
        row[metric] = extracted.get(metric, math.nan)
    return [row], missing


def load_g1_selected_from_dir(g1_dir: Path, preferred_splits: Sequence[str]) -> tuple[list[dict[str, Any]], list[str]]:
    """Load selected G1 pilot rows when available."""

    missing: list[str] = []
    selection_path = g1_dir / "model_selection_audit.csv"
    if not selection_path.exists():
        return [], [f"Missing G1 pilot selection audit: {selection_path}"]
    selection = pd.read_csv(selection_path)
    if selection.empty:
        return [], [f"Empty G1 pilot selection audit: {selection_path}"]

    # Keep only selected summary rows, but also prefer all_forecasting if present.
    selected = selection.copy()
    if "selected_for_test_summary" in selected.columns:
        tmp = selected[selected["selected_for_test_summary"].astype(str).str.lower().isin({"true", "1"})].copy()
        if not tmp.empty:
            selected = tmp
    if "feature_regime" in selected.columns:
        all_forecasting = selected[selected["feature_regime"].astype(str) == "all_forecasting"]
        if not all_forecasting.empty:
            selected = all_forecasting

    rows: list[dict[str, Any]] = []
    for _, item in selected.iterrows():
        edge_regime = str(item.get("edge_regime", ""))
        if edge_regime not in {"no_edges", "temporal_only", "spatial_temporal", "random_spatial_placebo"}:
            continue
        label = f"G1 pilot {edge_regime}"
        if edge_regime == "no_edges":
            group = "Neural control"
            is_control = True
            is_graph = False
        elif edge_regime == "random_spatial_placebo":
            group = "Placebo control"
            is_control = True
            is_graph = False
        else:
            group = "Graph/neural"
            is_control = False
            is_graph = True
        row = {
            "label": label,
            "comparison_group": group,
            "source_stage": "G1",
            "source": "G1_pilot",
            "family": edge_regime,
            "model_name": item.get("model_name"),
            "split_name": "test",
            "selection_policy": str(item.get("selection_metric", item.get("best_monitor_metric", "G1 validation-selected"))),
            "role": "Selected G1 pilot family representative.",
            "is_index": False,
            "is_tabular_ml": False,
            "is_graph_family": bool(is_graph),
            "is_control_family": bool(is_control),
        }
        mapping = {
            "test_mae": "mae",
            "test_rmse": "rmse",
            "test_mean_poisson_deviance": "mean_poisson_deviance",
            "test_spearman": "spearman",
            "test_ndcg_at_100": "ndcg_at_100",
            "test_top_10pct_overlap_rate": "top_10pct_overlap_rate",
        }
        for src, dst in mapping.items():
            row[dst] = pd.to_numeric(pd.Series([item.get(src, math.nan)]), errors="coerce").iloc[0]
        rows.append(row)

    if not rows:
        missing.append(f"No selected G1 pilot rows could be extracted from {selection_path}")
    return rows, missing


def load_g1_5_from_dir(g1_5_dir: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Load G1.5 final comparison rows."""

    missing: list[str] = []
    final_path = g1_5_dir / "final_comparison.csv"
    if not final_path.exists():
        return [], [f"Missing G1.5 final comparison: {final_path}"]
    final = pd.read_csv(final_path)
    if final.empty:
        return [], [f"Empty G1.5 final comparison: {final_path}"]

    rows: list[dict[str, Any]] = []
    for _, item in final.iterrows():
        family = str(item.get("family", ""))
        role = str(item.get("comparison_role", ""))
        if family.lower() == "nan" or family == "":
            # Usually the A3 row from G1.5. Keep it out to avoid duplicate A3 rows;
            # A3 is loaded from the frozen A3 directory directly.
            continue
        if family == "no_edges":
            label = "G1.5 selected no-edge neural control"
            group = "Neural control"
            is_control = True
            is_graph = False
        elif family == "random_spatial_placebo":
            label = "G1.5 selected random spatial placebo"
            group = "Placebo control"
            is_control = True
            is_graph = False
        elif family == "temporal_only":
            label = "G1.5 selected temporal graph"
            group = "Graph/neural"
            is_control = False
            is_graph = True
        elif family == "spatial_temporal":
            label = "G1.5 selected spatial-temporal graph"
            group = "Graph/neural"
            is_control = False
            is_graph = True
        else:
            label = f"G1.5 selected {family}"
            group = "Graph/neural"
            is_control = False
            is_graph = True

        row: dict[str, Any] = {
            "label": label,
            "comparison_group": group,
            "source_stage": "G1.5",
            "source": "G1.5_validation_sweep",
            "family": family,
            "model_name": item.get("model_name"),
            "split_name": "test",
            "selection_policy": str(item.get("selection_metric", "validation-selected G1.5 family representative")),
            "role": role or "Validation-selected G1.5 family representative.",
            "is_index": False,
            "is_tabular_ml": False,
            "is_graph_family": bool(is_graph),
            "is_control_family": bool(is_control),
        }
        for src, dst in G15_TO_CANONICAL.items():
            row[dst] = pd.to_numeric(pd.Series([item.get(src, math.nan)]), errors="coerce").iloc[0]
        rows.append(row)

    if not rows:
        missing.append(f"No G1.5 family rows could be extracted from {final_path}")
    return rows, missing


# ---------------------------------------------------------------------------
# Comparison tables
# ---------------------------------------------------------------------------


def build_long_metrics_from_comparison(comparison: pd.DataFrame) -> pd.DataFrame:
    """Convert compact comparison rows back into a long metric table."""

    rows: list[dict[str, Any]] = []
    for _, item in comparison.iterrows():
        for metric in CANONICAL_METRICS:
            value = pd.to_numeric(pd.Series([item.get(metric, math.nan)]), errors="coerce").iloc[0]
            rows.append(
                {
                    "label": item.get("label"),
                    "comparison_group": item.get("comparison_group"),
                    "source_stage": item.get("source_stage"),
                    "source": item.get("source"),
                    "family": item.get("family"),
                    "model_name": item.get("model_name"),
                    "split_name": item.get("split_name"),
                    "metric": metric,
                    "metric_label": METRIC_LABELS.get(metric, metric),
                    "metric_value": value,
                    "higher_is_better": METRIC_DIRECTIONS.get(metric, True),
                    "selection_policy": item.get("selection_policy"),
                    "role": item.get("role"),
                    "is_index": item.get("is_index", False),
                    "is_tabular_ml": item.get("is_tabular_ml", False),
                    "is_graph_family": item.get("is_graph_family", False),
                    "is_control_family": item.get("is_control_family", False),
                }
            )
    return pd.DataFrame(rows)


def compact_comparison(comparison: pd.DataFrame) -> pd.DataFrame:
    """Return a concise table for humans/reports."""

    cols = [
        "label",
        "comparison_group",
        "family",
        "selection_policy",
        "split_name",
        "mae",
        "spearman",
        "ndcg_at_100",
        "top_10pct_overlap_rate",
        "role",
    ]
    cols = [c for c in cols if c in comparison.columns]
    return comparison[cols].copy()


def metric_winners(comparison: pd.DataFrame) -> pd.DataFrame:
    """Winners among comparison rows for core metrics."""

    rows: list[dict[str, Any]] = []
    for metric in ["mae", "spearman", "ndcg_at_100", "top_10pct_overlap_rate"]:
        if metric not in comparison.columns:
            continue
        tmp = comparison.copy()
        tmp[metric] = pd.to_numeric(tmp[metric], errors="coerce")
        tmp = tmp[tmp[metric].notna()].copy()
        if tmp.empty:
            continue
        higher = METRIC_DIRECTIONS[metric]
        winner = tmp.sort_values(metric, ascending=not higher).iloc[0]
        rows.append(
            {
                "metric": metric,
                "metric_label": METRIC_LABELS.get(metric, metric),
                "higher_is_better": bool(higher),
                "winner_label": winner.get("label"),
                "winner_group": winner.get("comparison_group"),
                "winner_family": winner.get("family"),
                "winner_model_name": winner.get("model_name"),
                "winner_value": float(winner[metric]),
            }
        )
    return pd.DataFrame(rows)


def best_row_by_group(comparison: pd.DataFrame, group_predicate: pd.Series, metric: str) -> pd.Series | None:
    """Return best row in a group for metric."""

    tmp = comparison[group_predicate].copy()
    if tmp.empty or metric not in tmp.columns:
        return None
    tmp[metric] = pd.to_numeric(tmp[metric], errors="coerce")
    tmp = tmp[tmp[metric].notna()].copy()
    if tmp.empty:
        return None
    higher = METRIC_DIRECTIONS.get(metric, True)
    return tmp.sort_values(metric, ascending=not higher).iloc[0]


def metric_margin(a: float, b: float, metric: str) -> float:
    """Positive means a is better than b under metric direction."""

    if not (math.isfinite(float(a)) and math.isfinite(float(b))):
        return math.nan
    if METRIC_DIRECTIONS.get(metric, True):
        return float(a) - float(b)
    return float(b) - float(a)


def build_family_margin_table(comparison: pd.DataFrame) -> pd.DataFrame:
    """Build interpretable margins between key benchmark layers."""

    rows: list[dict[str, Any]] = []
    predicates = {
        "best_index": comparison["is_index"].astype(bool),
        "A3_tabular": comparison["comparison_group"].astype(str).eq("Tabular ML"),
        "best_graph_family": comparison["is_graph_family"].astype(bool),
        "best_control_family": comparison["is_control_family"].astype(bool),
    }
    comparisons = [
        ("best_graph_family", "best_index", "Graph family vs composite/calibrated index"),
        ("best_graph_family", "A3_tabular", "Graph family vs A3 tabular ML"),
        ("best_graph_family", "best_control_family", "Graph family vs no-edge/placebo controls"),
        ("best_control_family", "A3_tabular", "Control family vs A3 tabular ML"),
    ]
    for metric in ["mae", "spearman", "ndcg_at_100", "top_10pct_overlap_rate"]:
        for left_key, right_key, description in comparisons:
            left = best_row_by_group(comparison, predicates[left_key], metric)
            right = best_row_by_group(comparison, predicates[right_key], metric)
            if left is None or right is None:
                continue
            left_val = float(left.get(metric, math.nan))
            right_val = float(right.get(metric, math.nan))
            rows.append(
                {
                    "comparison": description,
                    "metric": metric,
                    "metric_label": METRIC_LABELS.get(metric, metric),
                    "higher_is_better": METRIC_DIRECTIONS.get(metric, True),
                    "left_label": left.get("label"),
                    "left_value": left_val,
                    "right_label": right.get("label"),
                    "right_value": right_val,
                    "positive_margin_means_left_better": metric_margin(left_val, right_val, metric),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Reports and plots
# ---------------------------------------------------------------------------


def dataframe_to_markdown(df: pd.DataFrame, max_rows: int = 80) -> str:
    if df.empty:
        return "_No rows._"
    out = df.head(max_rows).copy()
    try:
        return out.to_markdown(index=False)
    except Exception:
        return "```text\n" + out.to_string(index=False) + "\n```"


def fmt_num(value: Any, digits: int = 4) -> str:
    try:
        val = float(value)
    except Exception:
        return "—"
    if not math.isfinite(val):
        return "—"
    return f"{val:.{digits}f}"


def best_label(comparison: pd.DataFrame, predicate: pd.Series, metric: str) -> str | None:
    row = best_row_by_group(comparison, predicate, metric)
    if row is None:
        return None
    return str(row.get("label"))


def render_interpretation_report(
    *,
    generated_at: str,
    paths: ResolvedPaths,
    comparison: pd.DataFrame,
    compact: pd.DataFrame,
    winners: pd.DataFrame,
    margins: pd.DataFrame,
    missing: pd.DataFrame,
    metadata: Mapping[str, Any],
) -> str:
    """Render Markdown interpretation report."""

    lines: list[str] = []
    lines.append("# Index vs ML vs Graph Benchmark Comparison\n")
    lines.append(f"Generated at: `{generated_at}`\n")
    lines.append(f"Output directory: `{paths.output_dir}`\n")

    lines.append("## Purpose\n")
    lines.append(
        "This report consolidates already-produced benchmark outputs into a single "
        "index → tabular ML → graph/neural comparison. It does not retrain models. "
        "The goal is to summarize whether learned graph/neural benchmarks improve over "
        "static composite vulnerability indices and the A3 tabular ML layer.\n"
    )

    lines.append("## Compact comparison\n")
    display = compact.copy()
    rename = {
        "mae": "MAE",
        "spearman": "Spearman",
        "ndcg_at_100": "NDCG@100",
        "top_10pct_overlap_rate": "Top-10% overlap",
    }
    display = display.rename(columns=rename)
    lines.append(dataframe_to_markdown(display, max_rows=120))
    lines.append("")

    lines.append("## Metric winners\n")
    lines.append(dataframe_to_markdown(winners, max_rows=40))
    lines.append("")

    lines.append("## Key margins\n")
    lines.append(dataframe_to_markdown(margins, max_rows=120))
    lines.append("")

    # Narrative summary.
    lines.append("## Interpretation\n")
    if comparison.empty:
        lines.append("No comparison rows were produced. Check the missing-input audit.\n")
    else:
        index_pred = comparison["is_index"].astype(bool)
        graph_pred = comparison["is_graph_family"].astype(bool)
        control_pred = comparison["is_control_family"].astype(bool)
        a3_pred = comparison["comparison_group"].astype(str).eq("Tabular ML")

        graph_ndcg = best_row_by_group(comparison, graph_pred, "ndcg_at_100")
        index_ndcg = best_row_by_group(comparison, index_pred, "ndcg_at_100")
        a3_ndcg = best_row_by_group(comparison, a3_pred, "ndcg_at_100")
        control_ndcg = best_row_by_group(comparison, control_pred, "ndcg_at_100")

        if graph_ndcg is not None and index_ndcg is not None:
            margin = metric_margin(
                float(graph_ndcg.get("ndcg_at_100", math.nan)),
                float(index_ndcg.get("ndcg_at_100", math.nan)),
                "ndcg_at_100",
            )
            lines.append(
                f"- Best graph-family NDCG@100 row: `{graph_ndcg.get('label')}` "
                f"({fmt_num(graph_ndcg.get('ndcg_at_100'))}). Best index row: "
                f"`{index_ndcg.get('label')}` ({fmt_num(index_ndcg.get('ndcg_at_100'))}). "
                f"Graph-family margin: `{fmt_num(margin)}`.\n"
            )

        if graph_ndcg is not None and a3_ndcg is not None:
            margin = metric_margin(
                float(graph_ndcg.get("ndcg_at_100", math.nan)),
                float(a3_ndcg.get("ndcg_at_100", math.nan)),
                "ndcg_at_100",
            )
            lines.append(
                f"- Against A3 on NDCG@100, best graph-family row is "
                f"`{graph_ndcg.get('label')}` ({fmt_num(graph_ndcg.get('ndcg_at_100'))}) "
                f"versus `{a3_ndcg.get('label')}` ({fmt_num(a3_ndcg.get('ndcg_at_100'))}); "
                f"margin `{fmt_num(margin)}`.\n"
            )

        if graph_ndcg is not None and control_ndcg is not None:
            margin = metric_margin(
                float(graph_ndcg.get("ndcg_at_100", math.nan)),
                float(control_ndcg.get("ndcg_at_100", math.nan)),
                "ndcg_at_100",
            )
            if math.isfinite(margin) and margin > 0:
                lines.append(
                    "- The selected graph family beats the best no-edge/placebo control on "
                    "NDCG@100 in this table. This is the stronger topology-specific pattern, "
                    "but it should still be checked across seeds/splits.\n"
                )
            else:
                lines.append(
                    "- The best no-edge/placebo control remains competitive with or stronger "
                    "than the graph-family rows on NDCG@100. This means the benchmark supports "
                    "moving beyond static indices, but it does not yet isolate a clean spatial-topology "
                    "effect.\n"
                )

        lines.append(
            "- Public wording should distinguish `graph/neural benchmark improves over static "
            "composite indices` from `real spatial topology is validated`. The former may be "
            "supported even when no-edge/placebo controls remain strong; the latter requires "
            "the graph family to beat those controls.\n"
        )

    if not missing.empty:
        lines.append("## Missing or unavailable inputs\n")
        lines.append(dataframe_to_markdown(missing, max_rows=80))
        lines.append("")
        lines.append(
            "Missing optional rows are expected while SoVI or future benchmark layers are still "
            "being produced. Re-run this script after adding the corresponding metrics files.\n"
        )

    lines.append("## Inputs\n")
    input_rows = pd.DataFrame(metadata.get("inputs", []))
    lines.append(dataframe_to_markdown(input_rows, max_rows=80))
    lines.append("")

    lines.append("## Output files\n")
    output_rows = pd.DataFrame(metadata.get("outputs", []))
    lines.append(dataframe_to_markdown(output_rows, max_rows=80))
    lines.append("")

    return "\n".join(lines)


def write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def plot_metric_bar(
    comparison: pd.DataFrame,
    *,
    metric: str,
    output_path: Path,
    title: str,
    higher_is_better: bool,
    max_rows: int = 12,
) -> None:
    if plt is None or comparison.empty or metric not in comparison.columns:
        return
    data = comparison.dropna(subset=[metric]).copy()
    if data.empty:
        return
    data[metric] = pd.to_numeric(data[metric], errors="coerce")
    data = data.dropna(subset=[metric])
    data = data.sort_values(metric, ascending=not higher_is_better).head(max_rows)
    if data.empty:
        return

    fig, ax = plt.subplots(figsize=(11, max(4, 0.42 * len(data) + 1.5)))
    y = list(range(len(data)))
    ax.barh(y, data[metric].to_numpy())
    ax.set_yticks(y)
    ax.set_yticklabels(data["label"].astype(str).tolist())
    ax.invert_yaxis()
    ax.set_xlabel(METRIC_LABELS.get(metric, metric))
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def make_plots(comparison: pd.DataFrame, plots_dir: Path) -> list[dict[str, str]]:
    if plt is None:
        return []
    specs = [
        ("mae", "MAE by benchmark row", False),
        ("spearman", "Spearman by benchmark row", True),
        ("ndcg_at_100", "NDCG@100 by benchmark row", True),
        ("top_10pct_overlap_rate", "Top-10% overlap by benchmark row", True),
    ]
    outputs: list[dict[str, str]] = []
    for metric, title, higher in specs:
        path = plots_dir / f"{metric}.png"
        plot_metric_bar(comparison, metric=metric, output_path=path, title=title, higher_is_better=higher)
        if path.exists():
            outputs.append({"artifact": f"plot_{metric}", "path": str(path)})
    return outputs


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def missing_input_rows(paths: ResolvedPaths) -> list[dict[str, Any]]:
    specs = [
        ("A0 metrics", paths.a0_metrics, False),
        ("A1 SVI metrics", paths.a1_svi_metrics, True),
        ("A2 calibrated SVI metrics", paths.a2_svi_metrics, False),
        ("SoVI metrics", paths.sovi_metrics, False),
        ("A3 spatial directory", paths.a3_spatial_dir, True),
        ("G1 spatial directory", paths.g1_spatial_dir, False),
        ("G1.5 validation-sweep directory", paths.g1_5_dir, True),
    ]
    rows = []
    for label, path, required in specs:
        exists = path_exists(path)
        rows.append(
            {
                "input": label,
                "path": str(path) if path is not None else "",
                "exists": bool(exists),
                "required_for_core_comparison": bool(required),
                "status": "available" if exists else "missing",
            }
        )
    return rows


def collect_rows(paths: ResolvedPaths, preferred_splits: Sequence[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Collect all available comparison rows and missing-input notes."""

    rows: list[dict[str, Any]] = []
    missing_notes: list[dict[str, Any]] = []

    # A0 optional.
    if path_exists(paths.a0_metrics):
        metrics = read_long_metrics(paths.a0_metrics, "A0")
        for spec in a0_candidate_specs():
            row, note = build_row_from_long_metrics(metrics, spec, preferred_splits)
            if row is not None:
                rows.append(row)
            elif note:
                missing_notes.append({"source": "A0", "issue": note})

    # A1 SVI required for the original index comparison.
    if path_exists(paths.a1_svi_metrics):
        metrics = read_long_metrics(paths.a1_svi_metrics, "A1_SVI")
        for spec in index_candidate_specs()[:2]:
            row, note = build_row_from_long_metrics(metrics, spec, preferred_splits)
            if row is not None:
                rows.append(row)
            elif note:
                missing_notes.append({"source": "A1_SVI", "issue": note})
    else:
        missing_notes.append({"source": "A1_SVI", "issue": f"Missing SVI metrics: {paths.a1_svi_metrics}"})

    # A2 SVI optional but expected in current benchmark.
    if path_exists(paths.a2_svi_metrics):
        metrics = read_long_metrics(paths.a2_svi_metrics, "A2_SVI")
        for spec in index_candidate_specs()[2:]:
            row, note = build_row_from_long_metrics(metrics, spec, preferred_splits)
            if row is not None:
                rows.append(row)
            elif note:
                missing_notes.append({"source": "A2_SVI", "issue": note})
    else:
        missing_notes.append({"source": "A2_SVI", "issue": f"Missing A2 metrics: {paths.a2_svi_metrics}"})

    # SoVI optional/future.
    if path_exists(paths.sovi_metrics):
        metrics = read_long_metrics(paths.sovi_metrics, "SOVI")
        for spec in sovi_candidate_specs():
            row, note = build_row_from_long_metrics(metrics, spec, preferred_splits)
            if row is not None:
                rows.append(row)
            elif note:
                missing_notes.append({"source": "SOVI", "issue": note})
    else:
        missing_notes.append(
            {
                "source": "SOVI",
                "issue": (
                    f"SoVI metrics not found at {paths.sovi_metrics}. This is expected until the "
                    "SoVI benchmark adapter is generated."
                ),
            }
        )

    # A3 spatial selected row.
    if path_exists(paths.a3_spatial_dir):
        a3_rows, notes = load_a3_selected_from_dir(paths.a3_spatial_dir, preferred_splits)
        rows.extend(a3_rows)
        missing_notes.extend({"source": "A3", "issue": note} for note in notes)
    else:
        missing_notes.append({"source": "A3", "issue": f"Missing A3 spatial dir: {paths.a3_spatial_dir}"})

    # G1 pilot optional.
    if path_exists(paths.g1_spatial_dir):
        g1_rows, notes = load_g1_selected_from_dir(paths.g1_spatial_dir, preferred_splits)
        rows.extend(g1_rows)
        missing_notes.extend({"source": "G1", "issue": note} for note in notes)
    else:
        missing_notes.append({"source": "G1", "issue": f"Missing G1 spatial dir: {paths.g1_spatial_dir}"})

    # G1.5 final comparison.
    if path_exists(paths.g1_5_dir):
        g15_rows, notes = load_g1_5_from_dir(paths.g1_5_dir)
        rows.extend(g15_rows)
        missing_notes.extend({"source": "G1.5", "issue": note} for note in notes)
    else:
        missing_notes.append({"source": "G1.5", "issue": f"Missing G1.5 dir: {paths.g1_5_dir}"})

    comparison = pd.DataFrame(rows)
    missing = pd.DataFrame(missing_notes)

    if comparison.empty:
        return comparison, missing

    # Stable, conceptual ordering.
    order = {
        "Naive temporal baseline": 0,
        "Composite index": 1,
        "Calibrated index": 2,
        "Tabular ML": 3,
        "Neural control": 4,
        "Graph/neural": 5,
        "Placebo control": 6,
    }
    comparison["comparison_group_order"] = comparison["comparison_group"].map(lambda x: order.get(str(x), 99))
    comparison = comparison.sort_values(["comparison_group_order", "label"]).drop(columns=["comparison_group_order"])
    return comparison.reset_index(drop=True), missing.reset_index(drop=True)


def input_metadata(paths: ResolvedPaths, missing_audit: pd.DataFrame) -> list[dict[str, Any]]:
    rows = missing_input_rows(paths)
    # Add explicit notes for missing logical rows, not only missing files.
    if not missing_audit.empty:
        for _, row in missing_audit.iterrows():
            rows.append(
                {
                    "input": f"logical_row:{row.get('source')}",
                    "path": "",
                    "exists": False,
                    "required_for_core_comparison": False,
                    "status": str(row.get("issue")),
                }
            )
    return rows


def run_comparison(args: argparse.Namespace) -> dict[str, Any]:
    paths = resolve_paths(args)
    ensure_dir(paths.output_dir)
    preferred_splits = parse_csv_list(args.preferred_test_splits, PREFERRED_TEST_SPLITS)

    file_missing = pd.DataFrame(missing_input_rows(paths))
    if args.strict_inputs:
        required_missing = file_missing[
            file_missing["required_for_core_comparison"].astype(bool)
            & ~file_missing["exists"].astype(bool)
        ]
        if not required_missing.empty:
            raise BenchmarkComparisonError(
                "Missing required inputs:\n" + required_missing.to_string(index=False)
            )

    comparison, logical_missing = collect_rows(paths, preferred_splits)
    if comparison.empty:
        raise BenchmarkComparisonError(
            "No comparison rows were produced. Check paths or run upstream baselines first."
        )

    metrics_long = build_long_metrics_from_comparison(comparison)
    compact = compact_comparison(comparison)
    winners = metric_winners(comparison)
    margins = build_family_margin_table(comparison)

    missing_audit = pd.concat(
        [file_missing, logical_missing.assign(input="logical_row", path="", exists=False, required_for_core_comparison=False, status=logical_missing.get("issue", ""))]
        if not logical_missing.empty
        else [file_missing],
        ignore_index=True,
        sort=False,
    )

    outputs = {
        "metrics_long": paths.output_dir / "index_ml_graph_metrics_long.csv",
        "comparison": paths.output_dir / "benchmark_comparison.csv",
        "compact": paths.output_dir / "benchmark_comparison_compact.csv",
        "winners": paths.output_dir / "metric_winners.csv",
        "margins": paths.output_dir / "family_margin_table.csv",
        "missing": paths.output_dir / "missing_input_audit.csv",
        "metadata": paths.output_dir / "comparison_metadata.json",
        "report": paths.output_dir / "benchmark_interpretation.md",
    }

    metrics_long.to_csv(outputs["metrics_long"], index=False)
    comparison.to_csv(outputs["comparison"], index=False)
    compact.to_csv(outputs["compact"], index=False)
    winners.to_csv(outputs["winners"], index=False)
    margins.to_csv(outputs["margins"], index=False)
    missing_audit.to_csv(outputs["missing"], index=False)

    plot_outputs: list[dict[str, str]] = []
    if not args.no_plots:
        plot_outputs = make_plots(comparison, paths.output_dir / "plots")

    generated_at = now_utc()
    metadata = {
        "stage_slug": STAGE_SLUG,
        "generated_at": generated_at,
        "repo_root": str(paths.repo_root),
        "config_path": str(paths.config_path),
        "output_dir": str(paths.output_dir),
        "preferred_test_splits": list(preferred_splits),
        "n_comparison_rows": int(len(comparison)),
        "n_long_metric_rows": int(len(metrics_long)),
        "inputs": input_metadata(paths, logical_missing),
        "outputs": [
            {"artifact": key, "path": str(value)}
            for key, value in outputs.items()
        ]
        + plot_outputs,
        "interpretation_policy": (
            "Graph/neural wins over SVI/SoVI/A3 are separated from topology-specific claims. "
            "Topology-specific evidence requires graph-family rows to beat no-edge and placebo controls."
        ),
    }
    write_json(outputs["metadata"], metadata)

    report = render_interpretation_report(
        generated_at=generated_at,
        paths=paths,
        comparison=comparison,
        compact=compact,
        winners=winners,
        margins=margins,
        missing=logical_missing,
        metadata=metadata,
    )
    outputs["report"].write_text(report, encoding="utf-8")

    return {
        "status": "completed",
        "output_dir": str(paths.output_dir),
        "n_comparison_rows": int(len(comparison)),
        "n_long_metric_rows": int(len(metrics_long)),
        "outputs": {k: str(v) for k, v in outputs.items()},
        "plot_outputs": plot_outputs,
    }


def brief(result: Mapping[str, Any]) -> str:
    lines = [
        "Index/ML/graph benchmark comparison completed.",
        f"Status: {result.get('status')}",
        f"Output dir: {result.get('output_dir')}",
        f"Comparison rows: {result.get('n_comparison_rows')}",
        f"Long metric rows: {result.get('n_long_metric_rows')}",
    ]
    outputs = result.get("outputs", {}) or {}
    for key in ["compact", "comparison", "winners", "margins", "report", "metadata"]:
        lines.append(f"{key}: {outputs.get(key)}")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    result = run_comparison(args)
    print(brief(result))


if __name__ == "__main__":
    main()
