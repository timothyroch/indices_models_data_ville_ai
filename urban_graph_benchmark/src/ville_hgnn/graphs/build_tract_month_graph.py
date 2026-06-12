
#!/usr/bin/env python3
"""
Build an auditable tract-month spatiotemporal graph for the Montréal 311
water/drainage benchmark.

This module is the first graph-stage artifact builder after the frozen A0--A3
non-graph benchmark layer. It does not train a GNN. Instead, it converts the
A3-compatible tract-month panel into reproducible graph artifacts that can be
used by G1 spatiotemporal tract GNN experiments.

Purpose
------------------
The core graph-value question is:

    Does message passing over urban topology and time improve prediction/ranking
    of reported water/drainage 311 burden beyond the frozen A3 feature-parity
    tabular baselines?

This builder supports that question by creating:

- one node per census tract x month;
- A3-compatible feature regimes, especially:
    * all_forecasting
    * lagged_reporting
    * static_svi_calendar
    * target_history
    * no_target_history
- typed edges:
    * temporal_self_lag_1
    * temporal_self_lag_12, optional
    * spatial_knn_same_month
    * spatial_adjacency_same_month, optional when geometry is available
    * randomized spatial placebo edges
- split masks for temporal, random_debug, and spatial_block evaluation;
- leakage, feature, and edge audits;
- diagnostic plots for graph QA;
- a Markdown graph report with reproduction metadata.

Design discipline
-----------------
The builder intentionally exports plain, auditable artifacts (parquet, npy, npz,
json, csv) before any PyTorch/PyG-specific object. This makes graph construction
reviewable and avoids hiding leakage assumptions inside a model-training script.

Recommended command
-------------------
PYTHONPATH=urban_graph_benchmark/src python -m ville_hgnn.graphs.build_tract_month_graph \
  --config urban_graph_benchmark/configs/mtl_311_water_v0.1.yaml \
  --feature-regimes all_forecasting,lagged_reporting,no_target_history \
  --spatial-knn-k 8 \
  --include-temporal-lag1 \
  --include-temporal-lag12 \
  --include-knn-edges \
  --include-random-placebo \
  --generate-diagnostic-plots
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    import numpy as np
except Exception as exc:  # pragma: no cover
    np = None  # type: ignore[assignment]
    _NUMPY_IMPORT_ERROR = exc
else:
    _NUMPY_IMPORT_ERROR = None

try:
    import pandas as pd
except Exception as exc:  # pragma: no cover
    pd = None  # type: ignore[assignment]
    _PANDAS_IMPORT_ERROR = exc
else:
    _PANDAS_IMPORT_ERROR = None

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    plt = None  # type: ignore[assignment]
    _MATPLOTLIB_IMPORT_ERROR = exc
else:
    _MATPLOTLIB_IMPORT_ERROR = None

try:
    import geopandas as gpd  # type: ignore
except Exception:
    gpd = None  # type: ignore[assignment]

try:
    from shapely import wkt as shapely_wkt  # type: ignore
except Exception:
    shapely_wkt = None  # type: ignore[assignment]

try:
    from ville_hgnn.baselines import a3_tabular_feature_parity as a3
    from ville_hgnn.baselines.a1_svi_direct_ranking import build_svi_score_specs
    from ville_hgnn.baselines.common import (
        BINARY_TARGET_COLUMN,
        DEFAULT_CONFIG_PATH,
        TARGET_COLUMN,
        load_benchmark_frame,
        split_column_for_scheme,
        split_counts,
    )
    from ville_hgnn.utils.io import config_hash, file_hash, to_jsonable, write_json, write_markdown
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "This graph builder must be run inside the urban_graph_benchmark environment "
        "with ville_hgnn modules on PYTHONPATH."
    ) from exc


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STAGE_SLUG = "G1_tract_month_graph"
GRAPH_NAME = "G1 typed spatiotemporal tract-month graph"
DATASET_VERSION_DEFAULT = "mtl_311_water_v0"

ZONE_COL = getattr(a3, "ZONE_COL", "zone_id")
PERIOD_COL = getattr(a3, "PERIOD_COL", "period_month")
MONTH_COL = getattr(a3, "MONTH_COL", "month")
SAME_MONTH_NON_WATER_COL = getattr(a3, "SAME_MONTH_NON_WATER_COL", "total_311_count_non_water_drainage")

STRICT_TRAIN_STATIC_SETTING = getattr(a3, "STRICT_TRAIN_STATIC_SETTING", "forecasting_v0")
ROLLING_HISTORY_SETTING = getattr(a3, "ROLLING_HISTORY_SETTING", "rolling_observed_history_v0")
RETROSPECTIVE_SETTING = getattr(a3, "RETROSPECTIVE_SETTING", "retrospective_explanatory_v0")

RANDOM_SEED = getattr(a3, "RANDOM_SEED", 20240610)

DEFAULT_FEATURE_REGIMES = (
    "all_forecasting",
    "lagged_reporting",
    "static_svi_calendar",
    "target_history",
    "no_target_history",
)

DEFAULT_SPLIT_SCHEMES = ("temporal", "random_debug", "spatial_block")

FORBIDDEN_FORECASTING_TOKENS = (
    "same_month_water_drainage",
    "water_drainage_binary_same_month",
    "share_water_drainage",
    "share_water_drainage_requests",
    "total_311_count_all",
    "reporting_retro__",
    "same_month_reporting_retrospective",
)

ABSOLUTE_FORBIDDEN_FEATURE_NAMES = {
    TARGET_COLUMN,
    BINARY_TARGET_COLUMN,
    "water_drainage_requests",
    "share_water_drainage_requests",
    "total_311_count_all",
}

FEATURE_REGIME_TO_A3_SET = {
    "all_forecasting": "A3_all_forecasting",
    "lagged_reporting": "A3_lagged_reporting_forecasting",
    "static_svi_calendar": "A3_static_svi_calendar_forecasting",
    "target_history": "A3_target_history_forecasting",
    "target_history_svi_static": "A3_target_history_svi_static_forecasting",
    "target_history_lagged_reporting": "A3_target_history_lagged_reporting_forecasting",
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

class GraphBuildError(RuntimeError):
    """Raised when graph construction cannot proceed safely."""


@dataclass(frozen=True)
class GraphBuildConfig:
    """Configuration for graph artifact construction."""

    config_path: str | Path = DEFAULT_CONFIG_PATH
    repo_root: str | Path | None = None
    output_suffix: str | None = None
    feature_regimes: tuple[str, ...] = DEFAULT_FEATURE_REGIMES
    split_schemes: tuple[str, ...] = DEFAULT_SPLIT_SCHEMES
    spatial_knn_k: int = 8
    spatial_weighting: str = "rbf"  # one of: rbf, inverse_distance, binary
    include_knn_edges: bool = True
    include_adjacency_edges: bool = False
    include_temporal_lag1_edges: bool = True
    include_temporal_lag12_edges: bool = True
    include_random_placebo_edges: bool = True
    random_seed: int = RANDOM_SEED
    tract_geometry_path: str | Path | None = None
    tract_geometry_id_col: str | None = None
    generate_diagnostic_plots: bool = True
    plot_format: str = "png"
    strict_leakage: bool = True
    write_pyg_placeholders: bool = True


@dataclass(frozen=True)
class GraphArtifacts:
    """Paths written by graph construction."""

    output_dir: Path
    node_table: Path
    edge_table: Path
    target_vector: Path
    binary_target_vector: Path
    split_masks: Path
    edge_mask_by_split_regime: Path
    edge_index_by_type: Path
    edge_weight_by_type: Path
    graph_metadata: Path
    graph_report: Path
    feature_audit: Path
    edge_audit: Path
    leakage_audit: Path
    feature_matrix_metadata: Path
    plots_dir: Path
    feature_matrix_paths: Mapping[str, Path] = field(default_factory=dict)
    feature_columns_paths: Mapping[str, Path] = field(default_factory=dict)
    feature_stats_paths: Mapping[str, Path] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Dependency and serialization helpers
# ---------------------------------------------------------------------------

def require_runtime_dependencies() -> None:
    """Fail clearly if runtime dependencies are missing."""

    if pd is None:
        raise GraphBuildError("pandas is required to build graph artifacts.") from _PANDAS_IMPORT_ERROR
    if np is None:
        raise GraphBuildError("numpy is required to build graph artifacts.") from _NUMPY_IMPORT_ERROR


def ensure_dir(path: Path) -> Path:
    """Create a directory if needed."""

    path.mkdir(parents=True, exist_ok=True)
    return path


def now_utc() -> str:
    """Current UTC timestamp."""

    return datetime.now(timezone.utc).isoformat()


def safe_numeric(series: pd.Series) -> pd.Series:
    """Convert a series to numeric float with NaNs for invalid values."""

    return pd.to_numeric(series, errors="coerce").astype(float)


def parse_csv_list(value: str | Sequence[str] | None, default: Sequence[str]) -> tuple[str, ...]:
    """Parse a comma-separated string or list into a tuple."""

    if value is None:
        return tuple(default)
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",") if item.strip()]
        return tuple(items) if items else tuple(default)
    return tuple(str(item).strip() for item in value if str(item).strip())


def safe_filename(value: str, max_len: int = 120) -> str:
    """Make a filesystem-safe filename stem."""

    keep: list[str] = []
    for ch in str(value):
        if ch.isalnum() or ch in {"_", "-", "."}:
            keep.append(ch)
        else:
            keep.append("_")
    out = "".join(keep).strip("_")
    while "__" in out:
        out = out.replace("__", "_")
    if len(out) > max_len:
        out = out[:max_len].rstrip("_")
    return out or "unnamed"


def dataframe_to_markdown(df: pd.DataFrame, max_rows: int = 60) -> str:
    """Render dataframe to Markdown, with a robust fallback."""

    if df.empty:
        return "_No rows._"
    display = df.head(max_rows).copy()
    try:
        return display.to_markdown(index=False)
    except Exception:
        return "```text\n" + display.to_string(index=False) + "\n```"


def write_text(path: Path, text: str) -> None:
    """Write UTF-8 text."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json_fallback(path: Path, data: Mapping[str, Any]) -> None:
    """Write JSON using project helper when possible."""

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        write_json(path, to_jsonable(data))
    except Exception:
        path.write_text(json.dumps(to_jsonable(data), indent=2, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# Input loading and A3-compatible feature preparation
# ---------------------------------------------------------------------------

def load_inputs(config: GraphBuildConfig) -> tuple[Mapping[str, Any], Path, Path, Path, Path, pd.DataFrame]:
    """Load the benchmark frame using the same loader as A3."""

    loaded = load_benchmark_frame(config_path=config.config_path, repo_root=config.repo_root)
    if len(loaded) != 6:
        raise GraphBuildError(
            "Expected load_benchmark_frame to return 6 objects: config, root, config_path, "
            "panel_path, split_path, frame."
        )
    cfg, root, resolved_config_path, panel_path, split_path, frame = loaded
    return cfg, Path(root), Path(resolved_config_path), Path(panel_path), Path(split_path), frame.copy()


def output_dir_for_graph(config: Mapping[str, Any], root: Path, build_config: GraphBuildConfig) -> Path:
    """Resolve output directory for graph artifacts."""

    benchmark_id = str(config.get("benchmark_id", DATASET_VERSION_DEFAULT))
    slug = STAGE_SLUG if not build_config.output_suffix else f"{STAGE_SLUG}_{safe_filename(build_config.output_suffix)}"
    return root / "urban_graph_benchmark" / "outputs" / benchmark_id / "graphs" / slug


def normalize_period_column(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize period column and add period index/year/month fields."""

    if PERIOD_COL not in frame.columns:
        raise GraphBuildError(f"Required period column {PERIOD_COL!r} not found.")
    if ZONE_COL not in frame.columns:
        raise GraphBuildError(f"Required zone column {ZONE_COL!r} not found.")

    out = frame.copy()
    out[ZONE_COL] = out[ZONE_COL].astype(str)
    period_dt = pd.to_datetime(out[PERIOD_COL].astype(str), errors="coerce")
    if period_dt.isna().any():
        bad = out.loc[period_dt.isna(), PERIOD_COL].head(5).tolist()
        raise GraphBuildError(f"Could not parse some period values: {bad}")
    out["period_dt"] = period_dt.dt.to_period("M").dt.to_timestamp()
    out["period_month"] = out["period_dt"].dt.strftime("%Y-%m")
    out["year"] = out["period_dt"].dt.year.astype(int)
    out["month"] = out["period_dt"].dt.month.astype(int)

    ordered_months = {m: i for i, m in enumerate(sorted(out["period_month"].unique()))}
    out["period_index"] = out["period_month"].map(ordered_months).astype(int)
    return out


def train_frame_for_split(frame: pd.DataFrame, split_scheme: str = "temporal") -> pd.DataFrame:
    """Return train rows for a split scheme using A3 helper when possible."""

    if hasattr(a3, "train_frame"):
        try:
            return a3.train_frame(frame, split_scheme=split_scheme).copy()
        except TypeError:
            try:
                return a3.train_frame(frame, split_scheme).copy()
            except Exception:
                pass
        except Exception:
            pass

    split_col = split_column_for_scheme(split_scheme)
    if split_col not in frame.columns:
        raise GraphBuildError(f"Split column {split_col!r} not found for split {split_scheme!r}.")
    labels = frame[split_col].astype(str).str.lower()
    return frame[labels.str.endswith("train") | labels.eq("train")].copy()


def prepare_a3_feature_frame_compat(
    frame: pd.DataFrame,
    train: pd.DataFrame,
    split_scheme: str,
    svi_specs: Sequence[Any],
) -> tuple[pd.DataFrame, Mapping[str, Sequence[str]], pd.DataFrame]:
    """Call A3 feature preparation across known local signature variants."""

    attempts = [
        (frame, train, split_scheme, svi_specs),
        (frame, train, svi_specs),
        (frame, split_scheme, svi_specs),
    ]
    last_error: Exception | None = None
    for args in attempts:
        try:
            result = a3.prepare_feature_frame(*args)
            if len(result) != 3:
                raise GraphBuildError("A3 prepare_feature_frame did not return 3 objects.")
            prepared_frame, feature_groups, feature_lineage = result
            return prepared_frame.copy(), feature_groups, feature_lineage.copy()
        except TypeError as exc:
            last_error = exc
            continue
    raise GraphBuildError(
        "Could not call a3.prepare_feature_frame with any known signature. "
        "Tried (frame, train, split_scheme, svi_specs), (frame, train, svi_specs), "
        "and (frame, split_scheme, svi_specs)."
    ) from last_error


def prepare_a3_compatible_features(
    frame: pd.DataFrame,
    split_scheme_for_train_summary: str = "temporal",
    include_diagnostic_svi_sets: bool = False,
) -> tuple[pd.DataFrame, list[Any], pd.DataFrame, list[Mapping[str, Any]]]:
    """Normalize frame and construct A3 feature groups/sets for graph features."""

    if hasattr(a3, "normalize_frame_for_a3"):
        frame = a3.normalize_frame_for_a3(frame.copy(), split_scheme=split_scheme_for_train_summary)
    else:
        frame = normalize_period_column(frame)

    frame, svi_specs, score_audit = build_svi_score_specs(frame)
    train = train_frame_for_split(frame, split_scheme=split_scheme_for_train_summary)
    frame, feature_groups, feature_lineage = prepare_a3_feature_frame_compat(
        frame,
        train,
        split_scheme_for_train_summary,
        svi_specs,
    )

    feature_sets = a3.build_feature_set_specs(
        feature_groups,
        include_diagnostic_svi_sets=include_diagnostic_svi_sets,
    )
    try:
        a3.validate_feature_sets(feature_sets, feature_lineage)
    except Exception as exc:
        raise GraphBuildError(f"A3 feature-set validation failed: {exc}") from exc

    return frame, list(feature_sets), feature_lineage, list(score_audit)


# ---------------------------------------------------------------------------
# Node table construction
# ---------------------------------------------------------------------------

def find_coordinate_columns(frame: pd.DataFrame) -> tuple[str | None, str | None, str | None, str | None]:
    """Find projected and lon/lat coordinate columns if available."""

    x_col = next((c for c in ["tract_centroid_x", "centroid_x", "x", "x_centroid"] if c in frame.columns), None)
    y_col = next((c for c in ["tract_centroid_y", "centroid_y", "y", "y_centroid"] if c in frame.columns), None)
    lon_col = next((c for c in ["tract_centroid_lon", "lon_centroid", "longitude", "lon"] if c in frame.columns), None)
    lat_col = next((c for c in ["tract_centroid_lat", "lat_centroid", "latitude", "lat"] if c in frame.columns), None)
    return x_col, y_col, lon_col, lat_col


def build_node_table(frame: pd.DataFrame, split_schemes: Sequence[str]) -> pd.DataFrame:
    """Create stable node table: one node per tract-month."""

    frame = normalize_period_column(frame)
    duplicate_mask = frame.duplicated([ZONE_COL, "period_month"], keep=False)
    if duplicate_mask.any():
        examples = frame.loc[duplicate_mask, [ZONE_COL, "period_month"]].head(10)
        raise GraphBuildError(
            "Expected one row per zone-month, but duplicates were found:\n"
            + examples.to_string(index=False)
        )

    x_col, y_col, lon_col, lat_col = find_coordinate_columns(frame)

    base_cols = [
        ZONE_COL,
        "period_month",
        "period_dt",
        "period_index",
        "year",
        "month",
        TARGET_COLUMN,
        BINARY_TARGET_COLUMN,
        "population_total_2021",
        "land_area_km2",
        "population_density",
        "population_density_per_km2",
        "svi_percentile",
        "svi_score_raw",
        "svi_rank",
        "svi_class",
        SAME_MONTH_NON_WATER_COL,
        "requests_total",
    ]
    for coord in [x_col, y_col, lon_col, lat_col]:
        if coord is not None:
            base_cols.append(coord)

    for scheme in split_schemes:
        try:
            split_col = split_column_for_scheme(scheme)
            if split_col in frame.columns:
                base_cols.append(split_col)
        except Exception:
            continue

    cols = [c for c in dict.fromkeys(base_cols) if c in frame.columns]
    node_table = frame[cols].copy().sort_values([ZONE_COL, "period_dt"]).reset_index(drop=True)
    node_table.insert(0, "node_id", np.arange(len(node_table), dtype=np.int64))
    node_table["node_key"] = node_table[ZONE_COL].astype(str) + "__" + node_table["period_month"].astype(str)

    # Standard coordinate aliases for downstream graph/visual code.
    if x_col is not None:
        node_table["graph_x"] = safe_numeric(node_table[x_col])
    elif lon_col is not None:
        node_table["graph_x"] = safe_numeric(node_table[lon_col])
    else:
        node_table["graph_x"] = np.nan

    if y_col is not None:
        node_table["graph_y"] = safe_numeric(node_table[y_col])
    elif lat_col is not None:
        node_table["graph_y"] = safe_numeric(node_table[lat_col])
    else:
        node_table["graph_y"] = np.nan

    node_table["coordinate_source"] = (
        "projected_xy" if x_col is not None and y_col is not None
        else "lon_lat" if lon_col is not None and lat_col is not None
        else "missing"
    )

    return node_table


# ---------------------------------------------------------------------------
# Feature regimes and feature audits
# ---------------------------------------------------------------------------

def feature_set_name(spec: Any) -> str:
    """Return feature set name from A3 FeatureSetSpec-like object."""

    return str(getattr(spec, "name", spec.get("name") if isinstance(spec, Mapping) else ""))


def feature_set_columns(spec: Any) -> list[str]:
    """Return feature columns from A3 FeatureSetSpec-like object."""

    cols = getattr(spec, "feature_columns", None)
    if cols is None and isinstance(spec, Mapping):
        cols = spec.get("feature_columns")
    if cols is None:
        return []
    return list(cols)


def get_a3_feature_set_map(feature_sets: Sequence[Any]) -> dict[str, list[str]]:
    """Map A3 feature set names to columns."""

    out: dict[str, list[str]] = {}
    for spec in feature_sets:
        name = feature_set_name(spec)
        cols = feature_set_columns(spec)
        if name and cols:
            out[name] = cols
    return out


def feature_family_from_name(feature: str) -> str:
    """Infer feature family from A3-style feature prefix."""

    if "__" in feature:
        return feature.split("__", 1)[0]
    if feature.startswith("calendar_") or feature in {"month", "year"}:
        return "calendar"
    if "lag" in feature or "roll" in feature:
        return "history_or_reporting"
    return "unknown"


def build_feature_regime_columns(
    requested_regimes: Sequence[str],
    feature_sets: Sequence[Any],
    feature_lineage: pd.DataFrame,
) -> dict[str, list[str]]:
    """Build feature columns for requested graph feature regimes."""

    a3_map = get_a3_feature_set_map(feature_sets)
    regimes: dict[str, list[str]] = {}

    for regime in requested_regimes:
        if regime in FEATURE_REGIME_TO_A3_SET:
            a3_name = FEATURE_REGIME_TO_A3_SET[regime]
            if a3_name not in a3_map:
                raise GraphBuildError(f"Requested regime {regime!r}, but A3 feature set {a3_name!r} was not found.")
            regimes[regime] = a3_map[a3_name]
            continue

        if regime == "no_target_history":
            base_cols = a3_map.get("A3_all_forecasting")
            if not base_cols:
                raise GraphBuildError("Cannot build no_target_history because A3_all_forecasting was not found.")
            regimes[regime] = [
                c for c in base_cols
                if not c.startswith("target_history__") and not c.startswith("target_train_summary__")
            ]
            continue

        if regime == "all_forecasting_no_spatial_coordinates":
            base_cols = a3_map.get("A3_all_forecasting")
            if not base_cols:
                raise GraphBuildError("Cannot build all_forecasting_no_spatial_coordinates because A3_all_forecasting was not found.")
            regimes[regime] = [
                c for c in base_cols
                if not c.startswith("static_spatial__") and "centroid" not in c.lower()
            ]
            continue

        raise GraphBuildError(
            f"Unknown feature regime {regime!r}. Supported regimes include: "
            f"{sorted(set(FEATURE_REGIME_TO_A3_SET) | {'no_target_history', 'all_forecasting_no_spatial_coordinates'})}"
        )

    # Preserve order while removing duplicates within each regime.
    return {name: list(dict.fromkeys(cols)) for name, cols in regimes.items()}


def build_feature_matrix(frame: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    """Build raw float32 feature matrix, preserving NaNs for training-time imputation."""

    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise GraphBuildError(f"Feature columns missing from frame: {missing[:10]}{'...' if len(missing) > 10 else ''}")
    X = frame[list(columns)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    return X


def normalize_split_label(series: pd.Series) -> pd.Series:
    """Normalize split label values to train/validation/test/other."""

    labels = series.astype(str).str.lower()
    out = pd.Series("other", index=series.index, dtype="object")
    out[labels.eq("train") | labels.str.endswith("_train")] = "train"
    out[labels.eq("validation") | labels.eq("val") | labels.str.endswith("_validation") | labels.str.endswith("_val")] = "validation"
    out[labels.eq("test") | labels.str.endswith("_test")] = "test"
    return out


def train_mask_for_scheme(node_table: pd.DataFrame, split_scheme: str) -> np.ndarray:
    """Return train mask for a split scheme."""

    split_col = split_column_for_scheme(split_scheme)
    if split_col not in node_table.columns:
        raise GraphBuildError(f"Split column {split_col!r} missing from node table.")
    return normalize_split_label(node_table[split_col]).eq("train").to_numpy()


def build_feature_stats(
    node_table: pd.DataFrame,
    frame: pd.DataFrame,
    feature_regimes: Mapping[str, Sequence[str]],
    split_schemes: Sequence[str],
) -> dict[str, pd.DataFrame]:
    """Compute train-only imputation/scaling stats for each feature regime and split."""

    stats: dict[str, pd.DataFrame] = {}
    for regime, cols in feature_regimes.items():
        rows: list[dict[str, Any]] = []
        for split_scheme in split_schemes:
            try:
                train_mask = train_mask_for_scheme(node_table, split_scheme)
            except Exception:
                continue
            for feature in cols:
                values = safe_numeric(frame[feature])
                train_values = values.loc[train_mask]
                rows.append(
                    {
                        "feature_regime": regime,
                        "split_scheme": split_scheme,
                        "feature": feature,
                        "train_missing_rate": float(train_values.isna().mean()),
                        "train_median": float(train_values.median(skipna=True)) if train_values.notna().any() else 0.0,
                        "train_mean": float(train_values.mean(skipna=True)) if train_values.notna().any() else 0.0,
                        "train_std": float(train_values.std(skipna=True)) if train_values.notna().sum() > 1 else 1.0,
                        "global_missing_rate": float(values.isna().mean()),
                        "global_min": float(values.min(skipna=True)) if values.notna().any() else np.nan,
                        "global_max": float(values.max(skipna=True)) if values.notna().any() else np.nan,
                    }
                )
        stats[regime] = pd.DataFrame(rows)
    return stats


def build_feature_audit(
    frame: pd.DataFrame,
    node_table: pd.DataFrame,
    feature_regimes: Mapping[str, Sequence[str]],
    feature_lineage: pd.DataFrame,
    split_schemes: Sequence[str],
) -> pd.DataFrame:
    """Build feature audit with leakage flags and missingness by split."""

    lineage_by_feature: dict[str, Mapping[str, Any]] = {}
    feature_name_col = None
    for candidate in ["feature_name", "feature", "name"]:
        if candidate in feature_lineage.columns:
            feature_name_col = candidate
            break
    if feature_name_col is not None:
        for _, row in feature_lineage.iterrows():
            lineage_by_feature[str(row[feature_name_col])] = row.to_dict()

    rows: list[dict[str, Any]] = []
    for regime, cols in feature_regimes.items():
        for feature in cols:
            values = safe_numeric(frame[feature]) if feature in frame.columns else pd.Series(np.nan, index=frame.index)
            lineage = lineage_by_feature.get(feature, {})
            source_column = lineage.get("source_column", None)
            feature_family = lineage.get("feature_family", None) or feature_family_from_name(feature)
            uses_target_history = bool(lineage.get("uses_target_history", feature.startswith("target_history__") or feature.startswith("target_train_summary__")))
            uses_reporting_history = bool(lineage.get("uses_reporting_history", feature.startswith("reporting_history__") or feature.startswith("requests_history__")))
            uses_same_month_info = bool(lineage.get("uses_same_month_information", "same_month" in feature or feature.startswith("reporting_retro__")))
            strict_safe = bool(lineage.get("is_strict_forecasting_safe", not uses_same_month_info))

            forbidden_by_name = feature in ABSOLUTE_FORBIDDEN_FEATURE_NAMES
            forbidden_by_token = any(tok in feature for tok in FORBIDDEN_FORECASTING_TOKENS)
            leakage_status = "passed"
            if forbidden_by_name or forbidden_by_token or (uses_same_month_info and regime != "retrospective"):
                leakage_status = "failed"

            base = {
                "feature_regime": regime,
                "feature": feature,
                "feature_family": feature_family,
                "source_column": source_column,
                "uses_target_history": uses_target_history,
                "uses_reporting_history": uses_reporting_history,
                "uses_same_month_information": uses_same_month_info,
                "is_strict_forecasting_safe": strict_safe,
                "global_missing_rate": float(values.isna().mean()),
                "global_mean": float(values.mean(skipna=True)) if values.notna().any() else np.nan,
                "global_std": float(values.std(skipna=True)) if values.notna().sum() > 1 else np.nan,
                "leakage_status": leakage_status,
            }

            for split_scheme in split_schemes:
                try:
                    split_col = split_column_for_scheme(split_scheme)
                except Exception:
                    continue
                if split_col not in node_table.columns:
                    continue
                labels = normalize_split_label(node_table[split_col])
                for part in ["train", "validation", "test"]:
                    mask = labels.eq(part).to_numpy()
                    part_values = values.loc[mask]
                    base[f"{split_scheme}_{part}_missing_rate"] = float(part_values.isna().mean()) if len(part_values) else np.nan
                    base[f"{split_scheme}_{part}_mean"] = float(part_values.mean(skipna=True)) if part_values.notna().any() else np.nan
            rows.append(base)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Edge construction
# ---------------------------------------------------------------------------

def node_lookup(node_table: pd.DataFrame) -> dict[tuple[str, int], int]:
    """Map (zone_id, period_index) to node_id."""

    return {
        (str(row[ZONE_COL]), int(row["period_index"])): int(row["node_id"])
        for _, row in node_table[[ZONE_COL, "period_index", "node_id"]].iterrows()
    }


def build_temporal_self_edges(node_table: pd.DataFrame, lag: int) -> pd.DataFrame:
    """Build directed same-tract temporal edges from t-lag to t."""

    lookup = node_lookup(node_table)
    rows: list[dict[str, Any]] = []
    meta = node_table.set_index("node_id")[[ZONE_COL, "period_month", "period_index"]]

    for _, dst in node_table.iterrows():
        zone = str(dst[ZONE_COL])
        dst_period = int(dst["period_index"])
        src_key = (zone, dst_period - lag)
        if src_key not in lookup:
            continue
        src_id = int(lookup[src_key])
        dst_id = int(dst["node_id"])
        src_meta = meta.loc[src_id]
        rows.append(
            {
                "source_node_id": src_id,
                "target_node_id": dst_id,
                "edge_type": f"temporal_self_lag_{lag}",
                "edge_weight": 1.0,
                "source_zone_id": zone,
                "target_zone_id": zone,
                "source_period_month": str(src_meta["period_month"]),
                "target_period_month": str(dst["period_month"]),
                "source_period_index": int(src_meta["period_index"]),
                "target_period_index": dst_period,
                "distance_m": np.nan,
                "is_directed": True,
                "is_temporal": True,
                "is_spatial": False,
                "is_placebo": False,
            }
        )
    return pd.DataFrame(rows)


def unique_tract_coordinate_table(node_table: pd.DataFrame) -> pd.DataFrame:
    """Return one coordinate row per tract."""

    tract = (
        node_table[[ZONE_COL, "graph_x", "graph_y", "coordinate_source"]]
        .drop_duplicates(subset=[ZONE_COL])
        .copy()
        .sort_values(ZONE_COL)
        .reset_index(drop=True)
    )
    if tract["graph_x"].isna().any() or tract["graph_y"].isna().any():
        missing = tract[tract["graph_x"].isna() | tract["graph_y"].isna()][ZONE_COL].head(10).tolist()
        raise GraphBuildError(
            "Cannot build spatial edges because some tract coordinates are missing. "
            f"Examples: {missing}"
        )
    return tract


def pairwise_distances(coords: np.ndarray) -> np.ndarray:
    """Compute pairwise Euclidean distances for tract centroids."""

    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    return dist


def compute_spatial_weight(distance: float, bandwidth: float, weighting: str) -> float:
    """Compute edge weight from distance."""

    if weighting == "binary":
        return 1.0
    if weighting == "inverse_distance":
        return float(1.0 / max(distance, 1e-6))
    if weighting == "rbf":
        bw = max(float(bandwidth), 1e-6)
        return float(math.exp(-float(distance) / bw))
    raise GraphBuildError(f"Unknown spatial weighting scheme: {weighting!r}")


def build_spatial_knn_tract_pairs(
    tract_table: pd.DataFrame,
    k: int,
    weighting: str = "rbf",
) -> pd.DataFrame:
    """Build tract-level directed kNN pairs."""

    if k <= 0:
        raise GraphBuildError("spatial_knn_k must be positive.")
    coords = tract_table[["graph_x", "graph_y"]].to_numpy(dtype=float)
    dist = pairwise_distances(coords)
    n = len(tract_table)
    if n <= 1:
        raise GraphBuildError("Need at least two tracts for kNN edges.")

    k_eff = min(k, n - 1)
    # Distance to kth neighbor for rough bandwidth.
    kth_distances = []
    for i in range(n):
        order = np.argsort(dist[i])
        neighbors = [j for j in order if j != i][:k_eff]
        if neighbors:
            kth_distances.append(dist[i, neighbors[-1]])
    bandwidth = float(np.nanmedian(kth_distances)) if kth_distances else 1.0
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        bandwidth = float(np.nanmedian(dist[dist > 0])) if np.any(dist > 0) else 1.0

    rows: list[dict[str, Any]] = []
    zones = tract_table[ZONE_COL].astype(str).tolist()
    for i in range(n):
        order = np.argsort(dist[i])
        neighbors = [j for j in order if j != i][:k_eff]
        for j in neighbors:
            d = float(dist[i, j])
            rows.append(
                {
                    "source_zone_id": zones[i],
                    "target_zone_id": zones[j],
                    "edge_type_base": "spatial_knn_same_month",
                    "distance_m": d,
                    "edge_weight": compute_spatial_weight(d, bandwidth, weighting),
                    "knn_k": k_eff,
                    "spatial_weighting": weighting,
                    "is_directed": True,
                }
            )
    return pd.DataFrame(rows)


def expand_tract_pairs_to_month_edges(
    tract_pairs: pd.DataFrame,
    node_table: pd.DataFrame,
    edge_type: str,
    is_placebo: bool = False,
) -> pd.DataFrame:
    """Expand tract-level same-month pairs into tract-month node edges."""

    if tract_pairs.empty:
        return pd.DataFrame()

    months = node_table[["period_month", "period_index"]].drop_duplicates().sort_values("period_index")
    node_ids = node_table.set_index([ZONE_COL, "period_month"])["node_id"].to_dict()

    rows: list[dict[str, Any]] = []
    for _, pair in tract_pairs.iterrows():
        src_zone = str(pair["source_zone_id"])
        dst_zone = str(pair["target_zone_id"])
        for _, month in months.iterrows():
            period_month = str(month["period_month"])
            src_key = (src_zone, period_month)
            dst_key = (dst_zone, period_month)
            if src_key not in node_ids or dst_key not in node_ids:
                continue
            rows.append(
                {
                    "source_node_id": int(node_ids[src_key]),
                    "target_node_id": int(node_ids[dst_key]),
                    "edge_type": edge_type,
                    "edge_weight": float(pair.get("edge_weight", 1.0)),
                    "source_zone_id": src_zone,
                    "target_zone_id": dst_zone,
                    "source_period_month": period_month,
                    "target_period_month": period_month,
                    "source_period_index": int(month["period_index"]),
                    "target_period_index": int(month["period_index"]),
                    "distance_m": float(pair.get("distance_m", np.nan)) if pd.notna(pair.get("distance_m", np.nan)) else np.nan,
                    "is_directed": bool(pair.get("is_directed", True)),
                    "is_temporal": False,
                    "is_spatial": True,
                    "is_placebo": bool(is_placebo),
                }
            )
    return pd.DataFrame(rows)


def load_tract_geometry(path: str | Path, id_col: str | None = None) -> pd.DataFrame:
    """Load tract geometry if available for adjacency edges."""

    if gpd is None:
        raise GraphBuildError("geopandas is required for geometry adjacency edges but is not installed.")
    geom_path = Path(path)
    if not geom_path.exists():
        raise GraphBuildError(f"Geometry path does not exist: {geom_path}")

    gdf = gpd.read_file(geom_path) if geom_path.suffix.lower() not in {".parquet", ".pq"} else gpd.read_parquet(geom_path)
    if id_col is None:
        candidates = [ZONE_COL, "zone_id", "tract_id", "CTUID", "ctuid", "DGUID", "dguid", "GEO_UID", "geo_uid"]
        id_col = next((c for c in candidates if c in gdf.columns), None)
    if id_col is None or id_col not in gdf.columns:
        raise GraphBuildError(
            "Could not identify tract ID column in geometry. Provide --tract-geometry-id-col."
        )
    if "geometry" not in gdf.columns:
        raise GraphBuildError("Geometry file has no geometry column.")

    out = gdf[[id_col, "geometry"]].copy()
    out = out.rename(columns={id_col: ZONE_COL})
    out[ZONE_COL] = out[ZONE_COL].astype(str)
    return out


def build_spatial_adjacency_tract_pairs(
    tract_geometry: pd.DataFrame,
    tract_table: pd.DataFrame,
) -> pd.DataFrame:
    """Build tract-level adjacency pairs from polygon touching/intersection."""

    if gpd is None:
        raise GraphBuildError("geopandas is required for adjacency edges.")
    if "geometry" not in tract_geometry.columns:
        raise GraphBuildError("tract_geometry must contain a geometry column.")

    zones_needed = set(tract_table[ZONE_COL].astype(str))
    gdf = gpd.GeoDataFrame(tract_geometry.copy(), geometry="geometry")
    gdf[ZONE_COL] = gdf[ZONE_COL].astype(str)
    gdf = gdf[gdf[ZONE_COL].isin(zones_needed)].copy().reset_index(drop=True)
    if gdf.empty:
        raise GraphBuildError("No geometry rows matched graph tract IDs.")

    rows: list[dict[str, Any]] = []
    # Use spatial index when available, fallback to pairwise for 540 tracts.
    try:
        sindex = gdf.sindex
        for i, geom in enumerate(gdf.geometry):
            candidates = list(sindex.intersection(geom.bounds))
            for j in candidates:
                if i == j:
                    continue
                other = gdf.geometry.iloc[j]
                if geom.touches(other) or geom.intersects(other):
                    rows.append(
                        {
                            "source_zone_id": str(gdf[ZONE_COL].iloc[i]),
                            "target_zone_id": str(gdf[ZONE_COL].iloc[j]),
                            "edge_type_base": "spatial_adjacency_same_month",
                            "distance_m": np.nan,
                            "edge_weight": 1.0,
                            "is_directed": True,
                        }
                    )
    except Exception:
        n = len(gdf)
        for i in range(n):
            geom = gdf.geometry.iloc[i]
            for j in range(n):
                if i == j:
                    continue
                other = gdf.geometry.iloc[j]
                if geom.touches(other) or geom.intersects(other):
                    rows.append(
                        {
                            "source_zone_id": str(gdf[ZONE_COL].iloc[i]),
                            "target_zone_id": str(gdf[ZONE_COL].iloc[j]),
                            "edge_type_base": "spatial_adjacency_same_month",
                            "distance_m": np.nan,
                            "edge_weight": 1.0,
                            "is_directed": True,
                        }
                    )
    return pd.DataFrame(rows).drop_duplicates()


def build_randomized_spatial_placebo_edges(
    edge_table: pd.DataFrame,
    node_table: pd.DataFrame,
    edge_type_to_randomize: str,
    random_seed: int,
) -> pd.DataFrame:
    """Randomize same-month spatial targets while preserving source-node degree."""

    base_edges = edge_table[edge_table["edge_type"] == edge_type_to_randomize].copy()
    if base_edges.empty:
        return pd.DataFrame()

    rng = np.random.default_rng(random_seed)
    nodes_by_period = {
        int(period): group[["node_id", ZONE_COL]].copy()
        for period, group in node_table.groupby("period_index")
    }

    rows: list[dict[str, Any]] = []
    for _, edge in base_edges.iterrows():
        period = int(edge["source_period_index"])
        source_node = int(edge["source_node_id"])
        source_zone = str(edge["source_zone_id"])
        candidates = nodes_by_period.get(period)
        if candidates is None or candidates.empty:
            continue
        candidates = candidates[candidates[ZONE_COL].astype(str) != source_zone]
        if candidates.empty:
            continue
        sampled = candidates.sample(n=1, random_state=int(rng.integers(0, 2**31 - 1))).iloc[0]
        target_node = int(sampled["node_id"])
        target_zone = str(sampled[ZONE_COL])
        rows.append(
            {
                "source_node_id": source_node,
                "target_node_id": target_node,
                "edge_type": f"{edge_type_to_randomize}_random_placebo",
                "edge_weight": float(edge.get("edge_weight", 1.0)),
                "source_zone_id": source_zone,
                "target_zone_id": target_zone,
                "source_period_month": str(edge["source_period_month"]),
                "target_period_month": str(edge["target_period_month"]),
                "source_period_index": period,
                "target_period_index": int(edge["target_period_index"]),
                "distance_m": np.nan,
                "is_directed": True,
                "is_temporal": False,
                "is_spatial": True,
                "is_placebo": True,
            }
        )
    return pd.DataFrame(rows)


def build_edge_table(
    node_table: pd.DataFrame,
    build_config: GraphBuildConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build full typed edge table."""

    edge_parts: list[pd.DataFrame] = []
    edge_build_notes: dict[str, Any] = {}

    if build_config.include_temporal_lag1_edges:
        edge_parts.append(build_temporal_self_edges(node_table, lag=1))
    if build_config.include_temporal_lag12_edges:
        edge_parts.append(build_temporal_self_edges(node_table, lag=12))

    tract_table = None
    if build_config.include_knn_edges or build_config.include_adjacency_edges:
        tract_table = unique_tract_coordinate_table(node_table)

    if build_config.include_knn_edges:
        assert tract_table is not None
        tract_pairs = build_spatial_knn_tract_pairs(
            tract_table,
            k=build_config.spatial_knn_k,
            weighting=build_config.spatial_weighting,
        )
        edge_build_notes["spatial_knn_tract_pairs"] = int(len(tract_pairs))
        edge_parts.append(
            expand_tract_pairs_to_month_edges(
                tract_pairs,
                node_table,
                edge_type="spatial_knn_same_month",
                is_placebo=False,
            )
        )

    if build_config.include_adjacency_edges:
        if build_config.tract_geometry_path is None:
            warnings.warn(
                "--include-adjacency-edges was requested, but no --tract-geometry-path was provided. "
                "Skipping adjacency edges."
            )
            edge_build_notes["spatial_adjacency_skipped"] = "no_geometry_path"
        else:
            assert tract_table is not None
            geometry = load_tract_geometry(build_config.tract_geometry_path, build_config.tract_geometry_id_col)
            tract_pairs = build_spatial_adjacency_tract_pairs(geometry, tract_table)
            edge_build_notes["spatial_adjacency_tract_pairs"] = int(len(tract_pairs))
            edge_parts.append(
                expand_tract_pairs_to_month_edges(
                    tract_pairs,
                    node_table,
                    edge_type="spatial_adjacency_same_month",
                    is_placebo=False,
                )
            )

    edge_table = pd.concat([part for part in edge_parts if part is not None and not part.empty], ignore_index=True) if edge_parts else pd.DataFrame()
    if edge_table.empty:
        raise GraphBuildError("No edges were constructed. Enable at least one edge type.")

    if build_config.include_random_placebo_edges:
        placebo_parts = []
        for edge_type in ["spatial_knn_same_month", "spatial_adjacency_same_month"]:
            if edge_type in set(edge_table["edge_type"].astype(str)):
                placebo_parts.append(
                    build_randomized_spatial_placebo_edges(
                        edge_table,
                        node_table,
                        edge_type_to_randomize=edge_type,
                        random_seed=build_config.random_seed + len(placebo_parts),
                    )
                )
        if placebo_parts:
            edge_table = pd.concat([edge_table] + [p for p in placebo_parts if not p.empty], ignore_index=True)

    # Stable ordering and integer dtypes.
    edge_table = edge_table.sort_values(["edge_type", "source_node_id", "target_node_id"]).reset_index(drop=True)
    edge_table.insert(0, "edge_id", np.arange(len(edge_table), dtype=np.int64))
    for col in ["source_node_id", "target_node_id", "source_period_index", "target_period_index"]:
        edge_table[col] = edge_table[col].astype(np.int64)
    edge_table["edge_weight"] = safe_numeric(edge_table["edge_weight"]).fillna(1.0).astype(float)
    return edge_table, edge_build_notes


def annotate_edge_splits(
    edge_table: pd.DataFrame,
    node_table: pd.DataFrame,
    split_schemes: Sequence[str],
) -> pd.DataFrame:
    """Annotate edges with source/target split labels and edge-filter flags.

    The resulting columns are aligned with the final edge_table row order. This
    keeps graph message-passing regime definitions inside the graph artifact
    rather than forcing every training script to reconstruct them.
    """

    if edge_table.empty:
        return edge_table.copy()

    node_splits = node_table[["node_id"]].copy()
    active_schemes: list[str] = []

    for scheme in split_schemes:
        try:
            split_col = split_column_for_scheme(scheme)
        except Exception:
            continue

        if split_col not in node_table.columns:
            warnings.warn(
                f"Split column {split_col!r} not present in node table; "
                f"skipping edge split annotations for {scheme!r}."
            )
            continue

        node_splits[f"{scheme}_split"] = normalize_split_label(node_table[split_col])
        active_schemes.append(scheme)

    if not active_schemes:
        return edge_table.copy()

    source_splits = node_splits.rename(
        columns={
            "node_id": "source_node_id",
            **{f"{scheme}_split": f"source_{scheme}_split" for scheme in active_schemes},
        }
    )
    target_splits = node_splits.rename(
        columns={
            "node_id": "target_node_id",
            **{f"{scheme}_split": f"target_{scheme}_split" for scheme in active_schemes},
        }
    )

    annotation_cols: list[str] = []
    for scheme in active_schemes:
        annotation_cols.extend(
            [
                f"source_{scheme}_split",
                f"target_{scheme}_split",
                f"crosses_{scheme}_split",
                f"is_train_train_{scheme}",
                f"has_test_endpoint_{scheme}",
                f"source_is_{scheme}_train",
                f"target_is_{scheme}_train",
                f"source_is_{scheme}_test",
                f"target_is_{scheme}_test",
            ]
        )
    out = edge_table.drop(columns=[c for c in annotation_cols if c in edge_table.columns], errors="ignore")

    out = out.merge(source_splits, on="source_node_id", how="left", validate="many_to_one")
    out = out.merge(target_splits, on="target_node_id", how="left", validate="many_to_one")

    for scheme in active_schemes:
        source_col = f"source_{scheme}_split"
        target_col = f"target_{scheme}_split"

        source_label = out[source_col].astype("string").fillna("missing")
        target_label = out[target_col].astype("string").fillna("missing")

        out[f"crosses_{scheme}_split"] = source_label.ne(target_label)
        out[f"is_train_train_{scheme}"] = source_label.eq("train") & target_label.eq("train")
        out[f"has_test_endpoint_{scheme}"] = source_label.eq("test") | target_label.eq("test")

        out[f"source_is_{scheme}_train"] = source_label.eq("train")
        out[f"target_is_{scheme}_train"] = target_label.eq("train")
        out[f"source_is_{scheme}_test"] = source_label.eq("test")
        out[f"target_is_{scheme}_test"] = target_label.eq("test")

    return out


def build_edge_mask_by_split_regime(
    edge_table: pd.DataFrame,
    split_schemes: Sequence[str],
) -> dict[str, np.ndarray]:
    """Build edge masks for transductive and leakage-controlled regimes.

    Each mask is a boolean vector in the exact row order of edge_table.parquet.
    Training code can either filter the full edge table directly or map these
    masks to edge_index arrays after selecting edge types.
    """

    masks: dict[str, np.ndarray] = {}
    n_edges = len(edge_table)

    for scheme in split_schemes:
        source_split_col = f"source_{scheme}_split"
        target_split_col = f"target_{scheme}_split"
        if source_split_col not in edge_table.columns or target_split_col not in edge_table.columns:
            continue

        masks[f"{scheme}_all_edges"] = np.ones(n_edges, dtype=bool)

        train_train_col = f"is_train_train_{scheme}"
        if train_train_col in edge_table.columns:
            masks[f"{scheme}_train_train_edges"] = (
                edge_table[train_train_col]
                .fillna(False)
                .to_numpy(dtype=bool)
            )

        test_endpoint_col = f"has_test_endpoint_{scheme}"
        if test_endpoint_col in edge_table.columns:
            masks[f"{scheme}_no_test_incident_edges"] = (
                ~edge_table[test_endpoint_col]
                .fillna(True)
                .to_numpy(dtype=bool)
            )

    return masks


# ---------------------------------------------------------------------------
# Split masks and graph arrays
# ---------------------------------------------------------------------------

def build_split_masks(node_table: pd.DataFrame, split_schemes: Sequence[str]) -> dict[str, np.ndarray]:
    """Build boolean masks for every split scheme and partition."""

    masks: dict[str, np.ndarray] = {}
    for scheme in split_schemes:
        try:
            split_col = split_column_for_scheme(scheme)
        except Exception:
            continue
        if split_col not in node_table.columns:
            warnings.warn(f"Split column {split_col!r} not present in node table; skipping {scheme!r} masks.")
            continue
        labels = normalize_split_label(node_table[split_col])
        for part in ["train", "validation", "test"]:
            masks[f"{scheme}_{part}"] = labels.eq(part).to_numpy(dtype=bool)
    return masks


def build_edge_arrays(edge_table: pd.DataFrame) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Build edge_index and edge_weight arrays by edge type."""

    edge_index_by_type: dict[str, np.ndarray] = {}
    edge_weight_by_type: dict[str, np.ndarray] = {}
    for edge_type, part in edge_table.groupby("edge_type"):
        edge_index = part[["source_node_id", "target_node_id"]].to_numpy(dtype=np.int64).T
        edge_weight = safe_numeric(part["edge_weight"]).to_numpy(dtype=np.float32)
        edge_index_by_type[str(edge_type)] = edge_index
        edge_weight_by_type[str(edge_type)] = edge_weight
    return edge_index_by_type, edge_weight_by_type


def save_npz_dict(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Save a dictionary of arrays to compressed npz."""

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


# ---------------------------------------------------------------------------
# Audits
# ---------------------------------------------------------------------------

def build_edge_audit(edge_table: pd.DataFrame, node_table: pd.DataFrame, split_schemes: Sequence[str]) -> pd.DataFrame:
    """Build edge-type audit including degree stats and split-crossing counts."""

    rows: list[dict[str, Any]] = []
    n_nodes = len(node_table)
    split_labels_by_scheme: dict[str, pd.Series] = {}
    for scheme in split_schemes:
        try:
            split_col = split_column_for_scheme(scheme)
        except Exception:
            continue
        if split_col in node_table.columns:
            split_labels_by_scheme[scheme] = normalize_split_label(node_table[split_col])

    labels_lookup = {
        scheme: labels.reset_index(drop=True)
        for scheme, labels in split_labels_by_scheme.items()
    }

    for edge_type, part in edge_table.groupby("edge_type"):
        source_degree = part.groupby("source_node_id").size()
        target_degree = part.groupby("target_node_id").size()
        temporal_violations = int(
            (part["is_temporal"].astype(bool) & (part["source_period_index"] > part["target_period_index"])).sum()
        )
        same_month_spatial_violations = int(
            (part["is_spatial"].astype(bool) & (part["source_period_index"] != part["target_period_index"])).sum()
        )

        row: dict[str, Any] = {
            "edge_type": str(edge_type),
            "n_edges": int(len(part)),
            "n_unique_sources": int(part["source_node_id"].nunique()),
            "n_unique_targets": int(part["target_node_id"].nunique()),
            "mean_out_degree_over_active_sources": float(source_degree.mean()) if len(source_degree) else 0.0,
            "max_out_degree": int(source_degree.max()) if len(source_degree) else 0,
            "mean_in_degree_over_active_targets": float(target_degree.mean()) if len(target_degree) else 0.0,
            "max_in_degree": int(target_degree.max()) if len(target_degree) else 0,
            "density_directed": float(len(part) / max(n_nodes * max(n_nodes - 1, 1), 1)),
            "edge_weight_min": float(part["edge_weight"].min()) if len(part) else np.nan,
            "edge_weight_max": float(part["edge_weight"].max()) if len(part) else np.nan,
            "distance_m_min": float(part["distance_m"].min(skipna=True)) if part["distance_m"].notna().any() else np.nan,
            "distance_m_median": float(part["distance_m"].median(skipna=True)) if part["distance_m"].notna().any() else np.nan,
            "distance_m_max": float(part["distance_m"].max(skipna=True)) if part["distance_m"].notna().any() else np.nan,
            "temporal_direction_violations": temporal_violations,
            "same_month_spatial_violations": same_month_spatial_violations,
            "is_placebo": bool(part["is_placebo"].astype(bool).any()),
        }

        for scheme, labels in labels_lookup.items():
            source_labels = labels.iloc[part["source_node_id"].to_numpy()].to_numpy()
            target_labels = labels.iloc[part["target_node_id"].to_numpy()].to_numpy()
            for src_part in ["train", "validation", "test"]:
                for dst_part in ["train", "validation", "test"]:
                    key = f"{scheme}_edges_{src_part}_to_{dst_part}"
                    row[key] = int(((source_labels == src_part) & (target_labels == dst_part)).sum())
        rows.append(row)

    return pd.DataFrame(rows).sort_values("edge_type").reset_index(drop=True)


def build_leakage_audit(
    node_table: pd.DataFrame,
    edge_table: pd.DataFrame,
    feature_audit: pd.DataFrame,
    split_schemes: Sequence[str],
) -> pd.DataFrame:
    """Build graph leakage audit rows."""

    rows: list[dict[str, Any]] = []

    # Feature leakage checks.
    failed_features = feature_audit[feature_audit["leakage_status"].astype(str) == "failed"].copy()
    rows.append(
        {
            "check_name": "forecasting_feature_forbidden_tokens",
            "status": "failed" if not failed_features.empty else "passed",
            "severity": "critical",
            "n_violations": int(len(failed_features)),
            "details": "; ".join(failed_features["feature"].astype(str).head(12).tolist()),
        }
    )

    # Temporal edge direction.
    temporal_edges = edge_table[edge_table["is_temporal"].astype(bool)].copy()
    direction_violations = temporal_edges[temporal_edges["source_period_index"] > temporal_edges["target_period_index"]]
    rows.append(
        {
            "check_name": "temporal_edges_do_not_point_from_future_to_past",
            "status": "failed" if not direction_violations.empty else "passed",
            "severity": "critical",
            "n_violations": int(len(direction_violations)),
            "details": "source_period_index > target_period_index",
        }
    )

    # Same-month spatial edge consistency.
    spatial_edges = edge_table[edge_table["is_spatial"].astype(bool)].copy()
    spatial_time_violations = spatial_edges[spatial_edges["source_period_index"] != spatial_edges["target_period_index"]]
    rows.append(
        {
            "check_name": "spatial_edges_are_same_month",
            "status": "failed" if not spatial_time_violations.empty else "passed",
            "severity": "critical",
            "n_violations": int(len(spatial_time_violations)),
            "details": "source_period_index != target_period_index",
        }
    )

    # Cross-split edge disclosure. Not failure, but important.
    for scheme in split_schemes:
        try:
            split_col = split_column_for_scheme(scheme)
        except Exception:
            continue
        if split_col not in node_table.columns:
            continue
        labels = normalize_split_label(node_table[split_col]).reset_index(drop=True)
        source_labels = labels.iloc[edge_table["source_node_id"].to_numpy()].to_numpy()
        target_labels = labels.iloc[edge_table["target_node_id"].to_numpy()].to_numpy()
        cross = source_labels != target_labels
        rows.append(
            {
                "check_name": f"{scheme}_cross_split_edges_disclosure",
                "status": "info",
                "severity": "disclosure",
                "n_violations": int(cross.sum()),
                "details": "Cross-split edges are allowed only under explicit transductive/inference assumptions; labels must remain masked.",
            }
        )

    return pd.DataFrame(rows)


def maybe_raise_on_leakage(leakage_audit: pd.DataFrame, strict: bool) -> None:
    """Raise if critical leakage checks fail and strict mode is enabled."""

    if not strict:
        return
    failed = leakage_audit[
        (leakage_audit["severity"].astype(str) == "critical")
        & (leakage_audit["status"].astype(str) == "failed")
    ]
    if not failed.empty:
        raise GraphBuildError(
            "Critical leakage audit failed:\n" + failed.to_string(index=False)
        )


# ---------------------------------------------------------------------------
# Diagnostic plots
# ---------------------------------------------------------------------------

def require_plotting() -> bool:
    """Return whether plotting is available."""

    return plt is not None


def save_plot(path: Path, dpi: int = 160) -> str:
    """Save current matplotlib plot."""

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()
    return str(path)


def plot_edge_type_counts(edge_audit: pd.DataFrame, plots_dir: Path, plot_format: str) -> str | None:
    """Plot edge counts by type."""

    if not require_plotting() or edge_audit.empty:
        return None
    df = edge_audit.sort_values("n_edges", ascending=True)
    fig, ax = plt.subplots(figsize=(9.5, max(4.5, 0.45 * len(df))))
    ax.barh(df["edge_type"], df["n_edges"])
    ax.set_title("G1 graph edge counts by type")
    ax.set_xlabel("Number of directed edges")
    ax.grid(axis="x", alpha=0.25)
    for y, value in enumerate(df["n_edges"]):
        ax.text(value, y, f" {int(value):,}", va="center", fontsize=8)
    return save_plot(plots_dir / f"edge_type_counts.{plot_format}")


def plot_node_counts_by_split(node_table: pd.DataFrame, split_schemes: Sequence[str], plots_dir: Path, plot_format: str) -> str | None:
    """Plot node counts by split."""

    if not require_plotting():
        return None
    rows: list[dict[str, Any]] = []
    for scheme in split_schemes:
        try:
            split_col = split_column_for_scheme(scheme)
        except Exception:
            continue
        if split_col not in node_table.columns:
            continue
        labels = normalize_split_label(node_table[split_col])
        for part, n in labels.value_counts().items():
            rows.append({"split_scheme": scheme, "partition": part, "n_nodes": int(n)})
    if not rows:
        return None
    df = pd.DataFrame(rows)
    pivot = df.pivot_table(index="split_scheme", columns="partition", values="n_nodes", aggfunc="sum").fillna(0)
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    pivot[[c for c in ["train", "validation", "test", "other"] if c in pivot.columns]].plot(kind="bar", ax=ax)
    ax.set_title("G1 graph node counts by split")
    ax.set_xlabel("Split scheme")
    ax.set_ylabel("Number of nodes")
    ax.tick_params(axis="x", rotation=0)
    ax.grid(axis="y", alpha=0.25)
    return save_plot(plots_dir / f"node_counts_by_split.{plot_format}")


def plot_degree_distribution(edge_table: pd.DataFrame, plots_dir: Path, plot_format: str) -> str | None:
    """Plot out-degree distribution by edge type."""

    if not require_plotting() or edge_table.empty:
        return None
    edge_types = list(edge_table["edge_type"].dropna().astype(str).unique())
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    for edge_type in edge_types:
        deg = edge_table[edge_table["edge_type"] == edge_type].groupby("source_node_id").size()
        if deg.empty:
            continue
        ax.hist(deg, bins=30, alpha=0.35, label=clean_edge_label(edge_type))
    ax.set_title("G1 graph out-degree distribution by edge type")
    ax.set_xlabel("Out-degree among active source nodes")
    ax.set_ylabel("Node count")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    return save_plot(plots_dir / f"degree_distribution_by_edge_type.{plot_format}")


def clean_edge_label(value: str, max_len: int = 42) -> str:
    """Clean edge label for plots."""

    out = str(value).replace("_same_month", "").replace("_", " ")
    if len(out) > max_len:
        out = out[:max_len - 1] + "…"
    return out


def plot_target_mean_by_month(node_table: pd.DataFrame, plots_dir: Path, plot_format: str) -> str | None:
    """Plot average target by month."""

    if not require_plotting() or TARGET_COLUMN not in node_table.columns:
        return None
    df = node_table.copy()
    df[TARGET_COLUMN] = safe_numeric(df[TARGET_COLUMN])
    monthly = df.groupby("period_month", as_index=False)[TARGET_COLUMN].mean().sort_values("period_month")
    fig, ax = plt.subplots(figsize=(11.5, 5.0))
    ax.plot(monthly["period_month"], monthly[TARGET_COLUMN], marker="o", linewidth=2)
    ax.set_title("Mean water/drainage burden by month")
    ax.set_xlabel("Month")
    ax.set_ylabel("Mean count per tract")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(alpha=0.25)
    return save_plot(plots_dir / f"target_mean_by_month.{plot_format}")


def plot_spatial_graph_preview(
    node_table: pd.DataFrame,
    edge_table: pd.DataFrame,
    plots_dir: Path,
    plot_format: str,
    edge_type: str = "spatial_knn_same_month",
    max_edges: int = 5000,
) -> str | None:
    """Plot a geographic preview of spatial edges for the first month."""

    if not require_plotting():
        return None
    if node_table["graph_x"].isna().any() or node_table["graph_y"].isna().any():
        return None
    if edge_type not in set(edge_table["edge_type"].astype(str)):
        return None

    first_period = int(node_table["period_index"].min())
    nodes = node_table[node_table["period_index"] == first_period].copy()
    edges = edge_table[(edge_table["edge_type"] == edge_type) & (edge_table["source_period_index"] == first_period)].copy()
    if edges.empty:
        return None
    if len(edges) > max_edges:
        edges = edges.sample(n=max_edges, random_state=RANDOM_SEED)

    coords = nodes.set_index("node_id")[["graph_x", "graph_y"]]
    fig, ax = plt.subplots(figsize=(8.0, 8.0))
    for _, edge in edges.iterrows():
        src = int(edge["source_node_id"])
        dst = int(edge["target_node_id"])
        if src not in coords.index or dst not in coords.index:
            continue
        x0, y0 = coords.loc[src]
        x1, y1 = coords.loc[dst]
        ax.plot([x0, x1], [y0, y1], linewidth=0.35, alpha=0.18)
    ax.scatter(nodes["graph_x"], nodes["graph_y"], s=8, alpha=0.85)
    ax.set_title(f"Spatial graph preview: {edge_type}, {nodes['period_month'].iloc[0]}")
    ax.set_xlabel("Graph x")
    ax.set_ylabel("Graph y")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.18)
    return save_plot(plots_dir / f"spatial_preview__{safe_filename(edge_type)}.{plot_format}")


def generate_diagnostic_plots(
    node_table: pd.DataFrame,
    edge_table: pd.DataFrame,
    edge_audit: pd.DataFrame,
    split_schemes: Sequence[str],
    plots_dir: Path,
    plot_format: str,
) -> dict[str, str]:
    """Generate graph construction QA plots."""

    plot_paths: dict[str, str] = {}
    if not require_plotting():
        warnings.warn("matplotlib is not available; skipping diagnostic plots.")
        return plot_paths

    for key, maybe in [
        ("edge_type_counts", plot_edge_type_counts(edge_audit, plots_dir, plot_format)),
        ("node_counts_by_split", plot_node_counts_by_split(node_table, split_schemes, plots_dir, plot_format)),
        ("degree_distribution", plot_degree_distribution(edge_table, plots_dir, plot_format)),
        ("target_mean_by_month", plot_target_mean_by_month(node_table, plots_dir, plot_format)),
        ("spatial_knn_preview", plot_spatial_graph_preview(node_table, edge_table, plots_dir, plot_format, "spatial_knn_same_month")),
        ("spatial_adjacency_preview", plot_spatial_graph_preview(node_table, edge_table, plots_dir, plot_format, "spatial_adjacency_same_month")),
    ]:
        if maybe:
            plot_paths[key] = maybe
    return plot_paths


# ---------------------------------------------------------------------------
# Report and metadata
# ---------------------------------------------------------------------------

def split_summary_table(node_table: pd.DataFrame, split_schemes: Sequence[str]) -> pd.DataFrame:
    """Build split node/target summary."""

    rows: list[dict[str, Any]] = []
    for scheme in split_schemes:
        try:
            split_col = split_column_for_scheme(scheme)
        except Exception:
            continue
        if split_col not in node_table.columns:
            continue
        labels = normalize_split_label(node_table[split_col])
        for part in ["train", "validation", "test", "other"]:
            mask = labels.eq(part)
            if not mask.any():
                continue
            target = safe_numeric(node_table.loc[mask, TARGET_COLUMN]) if TARGET_COLUMN in node_table.columns else pd.Series(dtype=float)
            binary = safe_numeric(node_table.loc[mask, BINARY_TARGET_COLUMN]) if BINARY_TARGET_COLUMN in node_table.columns else pd.Series(dtype=float)
            rows.append(
                {
                    "split_scheme": scheme,
                    "partition": part,
                    "n_nodes": int(mask.sum()),
                    "target_total": float(target.sum(skipna=True)) if len(target) else np.nan,
                    "target_mean": float(target.mean(skipna=True)) if target.notna().any() else np.nan,
                    "target_positive_rate": float(binary.mean(skipna=True)) if binary.notna().any() else np.nan,
                }
            )
    return pd.DataFrame(rows)


def render_graph_report(
    *,
    graph_config: GraphBuildConfig,
    config: Mapping[str, Any],
    config_path: Path,
    panel_path: Path,
    split_path: Path,
    node_table: pd.DataFrame,
    edge_table: pd.DataFrame,
    feature_regimes: Mapping[str, Sequence[str]],
    split_masks: Mapping[str, np.ndarray],
    edge_masks: Mapping[str, np.ndarray],
    feature_audit: pd.DataFrame,
    edge_audit: pd.DataFrame,
    leakage_audit: pd.DataFrame,
    feature_stats_paths: Mapping[str, Path],
    plot_paths: Mapping[str, str],
    artifacts: GraphArtifacts,
    edge_build_notes: Mapping[str, Any],
) -> str:
    """Render Markdown graph construction report."""

    unique_tracts = int(node_table[ZONE_COL].nunique())
    unique_months = int(node_table["period_month"].nunique())
    split_summary = split_summary_table(node_table, graph_config.split_schemes)

    lines: list[str] = []
    lines.append(f"# {GRAPH_NAME}\n")
    lines.append(f"Generated at: `{now_utc()}`\n")
    lines.append("## Purpose\n")
    lines.append(
        "This artifact converts the A3 tract-month panel into an auditable typed "
        "spatiotemporal graph for G1 graph baselines. It does not train a GNN. "
        "It freezes graph nodes, feature regimes, edge types, split masks, and leakage audits.\n"
    )

    lines.append("## Core graph definition\n")
    lines.append("```text")
    lines.append("node = census tract × month")
    lines.append("target = water_drainage_count")
    lines.append("primary graph question = does typed message passing improve beyond frozen A3 tabular baselines?")
    lines.append("```\n")

    lines.append("## Graph dimensions\n")
    lines.append("| Quantity | Value |")
    lines.append("|---|---:|")
    lines.append(f"| Nodes | {len(node_table):,} |")
    lines.append(f"| Unique tracts | {unique_tracts:,} |")
    lines.append(f"| Unique months | {unique_months:,} |")
    lines.append(f"| Edges | {len(edge_table):,} |")
    lines.append("")

    lines.append("## Feature regimes\n")
    lines.append("| Regime | Number of features |")
    lines.append("|---|---:|")
    for name, cols in feature_regimes.items():
        lines.append(f"| `{name}` | {len(cols):,} |")
    lines.append("")

    lines.append("## Edge audit\n")
    edge_cols = [
        "edge_type",
        "n_edges",
        "n_unique_sources",
        "n_unique_targets",
        "mean_out_degree_over_active_sources",
        "max_out_degree",
        "temporal_direction_violations",
        "same_month_spatial_violations",
        "is_placebo",
    ]
    lines.append(dataframe_to_markdown(edge_audit[[c for c in edge_cols if c in edge_audit.columns]], max_rows=30))
    lines.append("")

    lines.append("## Split summary\n")
    lines.append(dataframe_to_markdown(split_summary, max_rows=30))
    lines.append("")

    lines.append("## Message-passing edge masks\n")
    lines.append(
        "This graph artifact supports multiple message-passing regimes. "
        "Edge masks are stored in `edge_mask_by_split_regime.npz` and are aligned "
        "with the row order of `edge_table.parquet`.\n"
    )
    if edge_masks:
        edge_mask_rows = pd.DataFrame(
            [
                {
                    "edge_mask": name,
                    "n_edges_allowed": int(mask.sum()),
                    "n_edges_total": int(len(mask)),
                    "share_edges_allowed": float(mask.mean()) if len(mask) else np.nan,
                }
                for name, mask in edge_masks.items()
            ]
        )
        lines.append(dataframe_to_markdown(edge_mask_rows, max_rows=60))
        lines.append("")
    else:
        lines.append("_No edge masks were generated._\n")

    lines.append("## Leakage audit\n")
    lines.append(dataframe_to_markdown(leakage_audit, max_rows=40))
    lines.append("")

    lines.append("## Feature audit preview\n")
    feature_cols = [
        "feature_regime",
        "feature",
        "feature_family",
        "uses_target_history",
        "uses_reporting_history",
        "uses_same_month_information",
        "is_strict_forecasting_safe",
        "global_missing_rate",
        "leakage_status",
    ]
    lines.append(dataframe_to_markdown(feature_audit[[c for c in feature_cols if c in feature_audit.columns]], max_rows=80))
    lines.append("")

    lines.append("## Edge construction notes\n")
    if edge_build_notes:
        lines.append("```json")
        lines.append(json.dumps(to_jsonable(edge_build_notes), indent=2, ensure_ascii=False))
        lines.append("```\n")
    else:
        lines.append("_No additional edge construction notes._\n")

    lines.append("## Diagnostic plots\n")
    if plot_paths:
        lines.append("| Plot | Path |")
        lines.append("|---|---|")
        for key, value in plot_paths.items():
            lines.append(f"| `{key}` | `{value}` |")
        lines.append("")
    else:
        lines.append("_No diagnostic plots generated._\n")

    lines.append("## Output artifacts\n")
    artifact_rows = {
        "node_table": artifacts.node_table,
        "edge_table": artifacts.edge_table,
        "target_vector": artifacts.target_vector,
        "binary_target_vector": artifacts.binary_target_vector,
        "split_masks": artifacts.split_masks,
        "edge_mask_by_split_regime": artifacts.edge_mask_by_split_regime,
        "edge_index_by_type": artifacts.edge_index_by_type,
        "edge_weight_by_type": artifacts.edge_weight_by_type,
        "graph_metadata": artifacts.graph_metadata,
        "feature_audit": artifacts.feature_audit,
        "edge_audit": artifacts.edge_audit,
        "leakage_audit": artifacts.leakage_audit,
        "feature_matrix_metadata": artifacts.feature_matrix_metadata,
        "plots_dir": artifacts.plots_dir,
    }
    lines.append("| Artifact | Path |")
    lines.append("|---|---|")
    for key, value in artifact_rows.items():
        lines.append(f"| `{key}` | `{value}` |")
    for regime, path in artifacts.feature_matrix_paths.items():
        lines.append(f"| `feature_matrix:{regime}` | `{path}` |")
    for regime, path in artifacts.feature_columns_paths.items():
        lines.append(f"| `feature_columns:{regime}` | `{path}` |")
    for regime, path in feature_stats_paths.items():
        lines.append(f"| `feature_stats:{regime}` | `{path}` |")
    lines.append("")

    lines.append("## Reproduction metadata\n")
    lines.append("```json")
    lines.append(
        json.dumps(
            to_jsonable(
                {
                    "config_path": str(config_path),
                    "config_hash": config_hash(config),
                    "panel_path": str(panel_path),
                    "panel_sha256": file_hash(panel_path) if panel_path.exists() else None,
                    "split_path": str(split_path),
                    "split_sha256": file_hash(split_path) if split_path.exists() else None,
                    "graph_build_config": asdict(graph_config),
                }
            ),
            indent=2,
            ensure_ascii=False,
        )
    )
    lines.append("```\n")

    lines.append("## Interpretation warnings\n")
    lines.append(
        "- This graph artifact supports multiple message-passing regimes. "
        "The default temporal experiment may use transductive node features with masked labels. "
        "Spatial-block experiments must report whether they use all edges, train-train edges, "
        "or no-test-incident edges.\n"
        "- Cross-split edges may exist in the graph artifact. This is acceptable only under explicit "
        "transductive/inference assumptions where labels remain masked.\n"
        "- Feature matrices are raw numeric matrices with NaNs preserved. Training scripts should fit "
        "imputation/scaling on train nodes only.\n"
        "- Randomized placebo edges are for topology ablation, not for final predictive deployment.\n"
        "- This graph is a typed spatiotemporal tract graph, not yet a full environmental HGNN with "
        "roads, drainage assets, green infrastructure, or critical-facility nodes.\n"
    )

    return "\n".join(lines)


def build_metadata(
    *,
    graph_config: GraphBuildConfig,
    config: Mapping[str, Any],
    config_path: Path,
    panel_path: Path,
    split_path: Path,
    node_table: pd.DataFrame,
    edge_table: pd.DataFrame,
    feature_regimes: Mapping[str, Sequence[str]],
    split_masks: Mapping[str, np.ndarray],
    edge_masks: Mapping[str, np.ndarray],
    edge_audit: pd.DataFrame,
    leakage_audit: pd.DataFrame,
    plot_paths: Mapping[str, str],
    artifacts: GraphArtifacts,
    edge_build_notes: Mapping[str, Any],
) -> dict[str, Any]:
    """Build graph metadata dictionary."""

    return to_jsonable(
        {
            "graph_name": GRAPH_NAME,
            "stage_slug": STAGE_SLUG,
            "generated_at": now_utc(),
            "benchmark_id": str(config.get("benchmark_id", DATASET_VERSION_DEFAULT)),
            "config_path": str(config_path),
            "config_hash": config_hash(config),
            "panel_path": str(panel_path),
            "panel_sha256": file_hash(panel_path) if panel_path.exists() else None,
            "split_path": str(split_path),
            "split_sha256": file_hash(split_path) if split_path.exists() else None,
            "node_definition": "census tract x month",
            "target": TARGET_COLUMN,
            "n_nodes": int(len(node_table)),
            "n_edges": int(len(edge_table)),
            "n_unique_tracts": int(node_table[ZONE_COL].nunique()),
            "n_unique_months": int(node_table["period_month"].nunique()),
            "edge_counts_by_type": edge_table["edge_type"].value_counts().to_dict(),
            "feature_regimes": {name: list(cols) for name, cols in feature_regimes.items()},
            "split_masks": {name: int(mask.sum()) for name, mask in split_masks.items()},
            "edge_masks": {name: int(mask.sum()) for name, mask in edge_masks.items()},
            "edge_build_notes": dict(edge_build_notes),
            "leakage_audit_status": leakage_audit["status"].value_counts().to_dict() if not leakage_audit.empty else {},
            "graph_build_config": asdict(graph_config),
            "plot_paths": dict(plot_paths),
            "artifacts": {
                "output_dir": str(artifacts.output_dir),
                "node_table": str(artifacts.node_table),
                "edge_table": str(artifacts.edge_table),
                "target_vector": str(artifacts.target_vector),
                "binary_target_vector": str(artifacts.binary_target_vector),
                "split_masks": str(artifacts.split_masks),
                "edge_mask_by_split_regime": str(artifacts.edge_mask_by_split_regime),
                "edge_index_by_type": str(artifacts.edge_index_by_type),
                "edge_weight_by_type": str(artifacts.edge_weight_by_type),
                "graph_report": str(artifacts.graph_report),
                "feature_audit": str(artifacts.feature_audit),
                "edge_audit": str(artifacts.edge_audit),
                "leakage_audit": str(artifacts.leakage_audit),
                "feature_matrix_paths": {k: str(v) for k, v in artifacts.feature_matrix_paths.items()},
                "feature_columns_paths": {k: str(v) for k, v in artifacts.feature_columns_paths.items()},
                "feature_stats_paths": {k: str(v) for k, v in artifacts.feature_stats_paths.items()},
            },
        }
    )


# ---------------------------------------------------------------------------
# Artifact writing
# ---------------------------------------------------------------------------

def initialize_artifact_paths(output_dir: Path, feature_regimes: Sequence[str]) -> GraphArtifacts:
    """Create artifact path object."""

    feature_matrix_paths = {
        regime: output_dir / f"feature_matrix__{safe_filename(regime)}__raw.npy"
        for regime in feature_regimes
    }
    feature_columns_paths = {
        regime: output_dir / f"feature_columns__{safe_filename(regime)}.json"
        for regime in feature_regimes
    }
    feature_stats_paths = {
        regime: output_dir / f"feature_stats__{safe_filename(regime)}.csv"
        for regime in feature_regimes
    }
    return GraphArtifacts(
        output_dir=output_dir,
        node_table=output_dir / "node_table.parquet",
        edge_table=output_dir / "edge_table.parquet",
        target_vector=output_dir / "target_vector.npy",
        binary_target_vector=output_dir / "binary_target_vector.npy",
        split_masks=output_dir / "split_masks.npz",
        edge_mask_by_split_regime=output_dir / "edge_mask_by_split_regime.npz",
        edge_index_by_type=output_dir / "edge_index_by_type.npz",
        edge_weight_by_type=output_dir / "edge_weight_by_type.npz",
        graph_metadata=output_dir / "graph_metadata.json",
        graph_report=output_dir / "graph_report.md",
        feature_audit=output_dir / "feature_audit.csv",
        edge_audit=output_dir / "edge_audit.csv",
        leakage_audit=output_dir / "leakage_audit.csv",
        feature_matrix_metadata=output_dir / "feature_matrix_metadata.json",
        plots_dir=output_dir / "plots",
        feature_matrix_paths=feature_matrix_paths,
        feature_columns_paths=feature_columns_paths,
        feature_stats_paths=feature_stats_paths,
    )


def write_feature_artifacts(
    frame: pd.DataFrame,
    feature_regimes: Mapping[str, Sequence[str]],
    feature_stats: Mapping[str, pd.DataFrame],
    artifacts: GraphArtifacts,
) -> dict[str, Any]:
    """Write feature matrices, column lists, and train-only stats."""

    metadata: dict[str, Any] = {}
    for regime, columns in feature_regimes.items():
        X = build_feature_matrix(frame, columns)
        np.save(artifacts.feature_matrix_paths[regime], X)
        artifacts.feature_columns_paths[regime].write_text(
            json.dumps(list(columns), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        stats = feature_stats.get(regime, pd.DataFrame())
        stats.to_csv(artifacts.feature_stats_paths[regime], index=False)
        metadata[regime] = {
            "matrix_path": str(artifacts.feature_matrix_paths[regime]),
            "columns_path": str(artifacts.feature_columns_paths[regime]),
            "stats_path": str(artifacts.feature_stats_paths[regime]),
            "n_rows": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "nan_fraction": float(np.isnan(X).mean()) if X.size else 0.0,
        }
    artifacts.feature_matrix_metadata.write_text(
        json.dumps(to_jsonable(metadata), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return metadata


def write_core_artifacts(
    *,
    frame: pd.DataFrame,
    node_table: pd.DataFrame,
    edge_table: pd.DataFrame,
    feature_regimes: Mapping[str, Sequence[str]],
    split_masks: Mapping[str, np.ndarray],
    edge_masks: Mapping[str, np.ndarray],
    edge_index_by_type: Mapping[str, np.ndarray],
    edge_weight_by_type: Mapping[str, np.ndarray],
    feature_audit: pd.DataFrame,
    edge_audit: pd.DataFrame,
    leakage_audit: pd.DataFrame,
    feature_stats: Mapping[str, pd.DataFrame],
    metadata: Mapping[str, Any],
    report: str,
    artifacts: GraphArtifacts,
) -> None:
    """Write all core artifacts."""

    ensure_dir(artifacts.output_dir)
    ensure_dir(artifacts.plots_dir)

    node_table.to_parquet(artifacts.node_table, index=False)
    edge_table.to_parquet(artifacts.edge_table, index=False)

    np.save(artifacts.target_vector, safe_numeric(node_table[TARGET_COLUMN]).to_numpy(dtype=np.float32))
    if BINARY_TARGET_COLUMN in node_table.columns:
        np.save(artifacts.binary_target_vector, safe_numeric(node_table[BINARY_TARGET_COLUMN]).to_numpy(dtype=np.float32))
    else:
        np.save(artifacts.binary_target_vector, (safe_numeric(node_table[TARGET_COLUMN]) > 0).to_numpy(dtype=np.float32))

    save_npz_dict(artifacts.split_masks, split_masks)
    save_npz_dict(artifacts.edge_mask_by_split_regime, edge_masks)
    save_npz_dict(artifacts.edge_index_by_type, edge_index_by_type)
    save_npz_dict(artifacts.edge_weight_by_type, edge_weight_by_type)

    feature_audit.to_csv(artifacts.feature_audit, index=False)
    edge_audit.to_csv(artifacts.edge_audit, index=False)
    leakage_audit.to_csv(artifacts.leakage_audit, index=False)

    write_feature_artifacts(frame, feature_regimes, feature_stats, artifacts)
    write_json_fallback(artifacts.graph_metadata, metadata)
    write_text(artifacts.graph_report, report)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_tract_month_graph(build_config: GraphBuildConfig) -> dict[str, Any]:
    """Build all tract-month graph artifacts."""

    require_runtime_dependencies()
    cfg, root, config_path, panel_path, split_path, raw_frame = load_inputs(build_config)

    prepared_frame, feature_sets, feature_lineage, svi_score_audit = prepare_a3_compatible_features(
        raw_frame,
        split_scheme_for_train_summary="temporal",
        include_diagnostic_svi_sets=False,
    )
    prepared_frame = normalize_period_column(prepared_frame)

    node_table = build_node_table(prepared_frame, build_config.split_schemes)
    # Reorder prepared_frame to the same node order for feature matrices.
    frame_keyed = prepared_frame.copy()
    frame_keyed["node_key"] = frame_keyed[ZONE_COL].astype(str) + "__" + frame_keyed["period_month"].astype(str)
    frame_ordered = node_table[["node_key"]].merge(frame_keyed, on="node_key", how="left", validate="one_to_one")
    if len(frame_ordered) != len(node_table):
        raise GraphBuildError("Internal ordering failure: feature frame rows do not match node table rows.")

    feature_regimes = build_feature_regime_columns(
        build_config.feature_regimes,
        feature_sets,
        feature_lineage,
    )

    feature_audit = build_feature_audit(
        frame_ordered,
        node_table,
        feature_regimes,
        feature_lineage,
        build_config.split_schemes,
    )
    feature_stats = build_feature_stats(node_table, frame_ordered, feature_regimes, build_config.split_schemes)

    edge_table, edge_build_notes = build_edge_table(node_table, build_config)
    edge_table = annotate_edge_splits(edge_table, node_table, build_config.split_schemes)
    edge_masks = build_edge_mask_by_split_regime(edge_table, build_config.split_schemes)

    edge_audit = build_edge_audit(edge_table, node_table, build_config.split_schemes)
    leakage_audit = build_leakage_audit(node_table, edge_table, feature_audit, build_config.split_schemes)
    maybe_raise_on_leakage(leakage_audit, strict=build_config.strict_leakage)

    split_masks = build_split_masks(node_table, build_config.split_schemes)
    edge_index_by_type, edge_weight_by_type = build_edge_arrays(edge_table)

    output_dir = output_dir_for_graph(cfg, root, build_config)
    artifacts = initialize_artifact_paths(output_dir, feature_regimes.keys())
    ensure_dir(output_dir)
    ensure_dir(artifacts.plots_dir)

    plot_paths: dict[str, str] = {}
    if build_config.generate_diagnostic_plots:
        plot_paths = generate_diagnostic_plots(
            node_table,
            edge_table,
            edge_audit,
            build_config.split_schemes,
            artifacts.plots_dir,
            build_config.plot_format,
        )

    metadata = build_metadata(
        graph_config=build_config,
        config=cfg,
        config_path=config_path,
        panel_path=panel_path,
        split_path=split_path,
        node_table=node_table,
        edge_table=edge_table,
        feature_regimes=feature_regimes,
        split_masks=split_masks,
        edge_masks=edge_masks,
        edge_audit=edge_audit,
        leakage_audit=leakage_audit,
        plot_paths=plot_paths,
        artifacts=artifacts,
        edge_build_notes=edge_build_notes,
    )

    report = render_graph_report(
        graph_config=build_config,
        config=cfg,
        config_path=config_path,
        panel_path=panel_path,
        split_path=split_path,
        node_table=node_table,
        edge_table=edge_table,
        feature_regimes=feature_regimes,
        split_masks=split_masks,
        edge_masks=edge_masks,
        feature_audit=feature_audit,
        edge_audit=edge_audit,
        leakage_audit=leakage_audit,
        feature_stats_paths=artifacts.feature_stats_paths,
        plot_paths=plot_paths,
        artifacts=artifacts,
        edge_build_notes=edge_build_notes,
    )

    write_core_artifacts(
        frame=frame_ordered,
        node_table=node_table,
        edge_table=edge_table,
        feature_regimes=feature_regimes,
        split_masks=split_masks,
        edge_masks=edge_masks,
        edge_index_by_type=edge_index_by_type,
        edge_weight_by_type=edge_weight_by_type,
        feature_audit=feature_audit,
        edge_audit=edge_audit,
        leakage_audit=leakage_audit,
        feature_stats=feature_stats,
        metadata=metadata,
        report=report,
        artifacts=artifacts,
    )

    return {
        "status": "completed",
        "graph_name": GRAPH_NAME,
        "stage_slug": STAGE_SLUG,
        "output_dir": str(output_dir),
        "n_nodes": int(len(node_table)),
        "n_edges": int(len(edge_table)),
        "n_edge_types": int(edge_table["edge_type"].nunique()),
        "feature_regimes": {name: len(cols) for name, cols in feature_regimes.items()},
        "split_masks": {name: int(mask.sum()) for name, mask in split_masks.items()},
        "edge_masks": {name: int(mask.sum()) for name, mask in edge_masks.items()},
        "artifacts": {k: str(v) for k, v in asdict(artifacts).items() if not isinstance(v, Mapping)},
        "graph_report": str(artifacts.graph_report),
        "graph_metadata": str(artifacts.graph_metadata),
        "plot_paths": plot_paths,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse CLI args."""

    parser = argparse.ArgumentParser(
        description="Build auditable G1 tract-month graph artifacts for the Montréal 311 benchmark."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Benchmark config path.")
    parser.add_argument("--repo-root", default=None, help="Repository root. Defaults to current working directory.")
    parser.add_argument("--output-suffix", default=None, help="Optional suffix for graph output directory.")
    parser.add_argument(
        "--feature-regimes",
        default=",".join(DEFAULT_FEATURE_REGIMES),
        help="Comma-separated feature regimes, e.g. all_forecasting,lagged_reporting,no_target_history.",
    )
    parser.add_argument(
        "--split-schemes",
        default=",".join(DEFAULT_SPLIT_SCHEMES),
        help="Comma-separated split schemes for masks and audits.",
    )
    parser.add_argument("--spatial-knn-k", type=int, default=8, help="Number of nearest tract neighbors.")
    parser.add_argument(
        "--spatial-weighting",
        default="rbf",
        choices=["rbf", "inverse_distance", "binary"],
        help="Spatial edge weighting scheme.",
    )

    edge_group = parser.add_argument_group("edge options")
    edge_group.add_argument("--include-knn-edges", action="store_true", default=True)
    edge_group.add_argument("--no-knn-edges", action="store_false", dest="include_knn_edges")
    edge_group.add_argument("--include-adjacency-edges", action="store_true", default=False)
    edge_group.add_argument("--include-temporal-lag1", action="store_true", default=True)
    edge_group.add_argument("--no-temporal-lag1", action="store_false", dest="include_temporal_lag1")
    edge_group.add_argument("--include-temporal-lag12", action="store_true", default=True)
    edge_group.add_argument("--no-temporal-lag12", action="store_false", dest="include_temporal_lag12")
    edge_group.add_argument("--include-random-placebo", action="store_true", default=True)
    edge_group.add_argument("--no-random-placebo", action="store_false", dest="include_random_placebo")

    parser.add_argument("--tract-geometry-path", default=None, help="Optional tract geometry file for adjacency edges.")
    parser.add_argument("--tract-geometry-id-col", default=None, help="Optional tract id column in geometry file.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--generate-diagnostic-plots", action="store_true", default=True)
    parser.add_argument("--no-diagnostic-plots", action="store_false", dest="generate_diagnostic_plots")
    parser.add_argument("--plot-format", default="png", choices=["png", "pdf", "svg"])
    parser.add_argument("--strict-leakage", action="store_true", default=True)
    parser.add_argument("--no-strict-leakage", action="store_false", dest="strict_leakage")

    return parser.parse_args()


def result_brief(result: Mapping[str, Any]) -> str:
    """Render short CLI result summary."""

    lines = [
        "G1 tract-month graph construction completed.",
        f"Status: {result.get('status')}",
        f"Output dir: {result.get('output_dir')}",
        f"Nodes: {result.get('n_nodes')}",
        f"Edges: {result.get('n_edges')}",
        f"Edge types: {result.get('n_edge_types')}",
        f"Graph report: {result.get('graph_report')}",
        f"Graph metadata: {result.get('graph_metadata')}",
    ]
    regimes = result.get("feature_regimes", {})
    if regimes:
        lines.append("Feature regimes:")
        for name, n_features in regimes.items():
            lines.append(f"  {name}: {n_features} features")
    edge_masks = result.get("edge_masks", {})
    if edge_masks:
        lines.append("Edge masks:")
        for name, n_edges in edge_masks.items():
            lines.append(f"  {name}: {n_edges} allowed edges")
    return "\n".join(lines)


def main() -> None:
    """CLI entry point."""

    args = parse_args()
    config = GraphBuildConfig(
        config_path=args.config,
        repo_root=args.repo_root,
        output_suffix=args.output_suffix,
        feature_regimes=parse_csv_list(args.feature_regimes, DEFAULT_FEATURE_REGIMES),
        split_schemes=parse_csv_list(args.split_schemes, DEFAULT_SPLIT_SCHEMES),
        spatial_knn_k=args.spatial_knn_k,
        spatial_weighting=args.spatial_weighting,
        include_knn_edges=args.include_knn_edges,
        include_adjacency_edges=args.include_adjacency_edges,
        include_temporal_lag1_edges=args.include_temporal_lag1,
        include_temporal_lag12_edges=args.include_temporal_lag12,
        include_random_placebo_edges=args.include_random_placebo,
        random_seed=args.random_seed,
        tract_geometry_path=args.tract_geometry_path,
        tract_geometry_id_col=args.tract_geometry_id_col,
        generate_diagnostic_plots=args.generate_diagnostic_plots,
        plot_format=args.plot_format,
        strict_leakage=args.strict_leakage,
    )

    result = build_tract_month_graph(config)
    print(result_brief(result).rstrip())


if __name__ == "__main__":
    main()


__all__ = [
    "GraphArtifacts",
    "GraphBuildConfig",
    "GraphBuildError",
    "STAGE_SLUG",
    "GRAPH_NAME",
    "build_tract_month_graph",
    "build_node_table",
    "build_edge_table",
    "annotate_edge_splits",
    "build_edge_mask_by_split_regime",
    "build_split_masks",
]