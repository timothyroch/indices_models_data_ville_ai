"""
Build leakage-aware split artifacts for the Montréal 311 water/drainage benchmark.

This module reads the validated Dataset v0 tract-month panel and writes split
artifacts required before any baseline metrics are reported.

It does not train models and does not implement baseline logic.

Outputs:

    urban_graph_benchmark/outputs/mtl_311_water_v0/datasets/splits/
      split_assignments.parquet
      split_metadata.json
      split_report.md
      target_thresholds_temporal.json
      target_thresholds_random_debug.json
      target_thresholds_spatial_block.json
      split_validation.json

Primary scientific split:

    train:      2022-01 to 2024-12
    validation: 2025-01 to 2025-08
    test:       2025-09 to 2026-05

Random split is debugging only. Spatial block split is preliminary and required
before graph claims, but graph-specific transductive/inductive handling belongs
in later graph-training code.

Magnitude-class rule, aligned with baseline_plan_mtl_311_v0.md:

    class 0: water_drainage_count == 0
    class 1: 0 < y <= positive-train Q25
    class 2: Q25 < y <= positive-train Q50
    class 3: Q50 < y <= positive-train Q75
    class 4: y > positive-train Q75
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

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

DEFAULT_PANEL_PATH = (
    "urban_graph_benchmark/outputs/mtl_311_water_v0/datasets/tract_month_panel.parquet"
)

DEFAULT_SPLIT_DIR = (
    "urban_graph_benchmark/outputs/mtl_311_water_v0/datasets/splits"
)

TEMPORAL_SPLIT_SPEC = {
    "train": {"start": "2022-01", "end": "2024-12"},
    "validation": {"start": "2025-01", "end": "2025-08"},
    "test": {"start": "2025-09", "end": "2026-05"},
}

RANDOM_DEBUG_SPEC = {
    "seed": 42,
    "train_fraction": 0.70,
    "validation_fraction": 0.15,
    "test_fraction": 0.15,
    "purpose": "debugging_only_not_main_scientific_evidence",
}

SPATIAL_BLOCK_SPEC = {
    "seed": 42,
    "x_bins": 5,
    "y_bins": 5,
    "train_fraction": 0.70,
    "validation_fraction": 0.15,
    "test_fraction": 0.15,
    "purpose": "preliminary_spatial_generalization_split_before_graph_claims",
}

TARGET_COLUMN = "water_drainage_count"
ID_COLUMNS = ["zone_id", "period_month"]

TEMPORAL_SPLIT_COL = "temporal_split"
RANDOM_DEBUG_SPLIT_COL = "random_debug_split"
SPATIAL_BLOCK_SPLIT_COL = "spatial_block_split"

TARGET_DERIVED_COLUMNS = {
    "water_drainage_count",
    "water_drainage_binary",
    "water_drainage_requests",
    "share_water_drainage_requests",
}


class SplitBuildError(RuntimeError):
    """Raised when split artifacts cannot be built safely."""


@dataclass(frozen=True)
class SplitOutputs:
    """Paths to split-builder output artifacts."""

    split_assignments: Path
    split_metadata: Path
    split_report: Path
    split_validation: Path
    target_thresholds_temporal: Path
    target_thresholds_random_debug: Path
    target_thresholds_spatial_block: Path

    def to_dict(self) -> dict[str, str]:
        return {key: str(value) for key, value in self.__dict__.items()}


def require_runtime_dependencies() -> None:
    """Fail clearly if required dataframe dependencies are missing."""

    if pd is None:
        raise SplitBuildError(
            "pandas is required to build split artifacts. Install pandas first."
        ) from _PANDAS_IMPORT_ERROR

    if np is None:
        raise SplitBuildError(
            "numpy is required to build split artifacts. Install numpy first."
        ) from _NUMPY_IMPORT_ERROR


def resolve_panel_path(config: Mapping[str, Any], repo_root: Path) -> Path:
    """Resolve the validated tract-month panel path."""

    configured = get_nested(
        config,
        ["paths", "expected_output_files", "tract_month_panel"],
        default=None,
    )

    value = configured if not is_unresolved_value(configured) else DEFAULT_PANEL_PATH
    resolved = resolve_path(value, repo_root=repo_root, allow_unresolved=False)

    if resolved is None:
        raise SplitBuildError("Could not resolve tract_month_panel path.")

    if not resolved.exists():
        raise SplitBuildError(f"tract_month_panel does not exist: {resolved}")

    return resolved


def resolve_split_dir(config: Mapping[str, Any], repo_root: Path) -> Path:
    """Resolve and create the split output directory."""

    configured = get_nested(
        config,
        ["splits", "output_dir"],
        default=None,
    )

    if is_unresolved_value(configured):
        # Derive from dataset_dir when possible.
        dataset_dir = get_nested(config, ["paths", "outputs", "dataset_dir"], default=None)
        if not is_unresolved_value(dataset_dir):
            configured = str(Path(str(dataset_dir)) / "splits")
        else:
            configured = DEFAULT_SPLIT_DIR

    resolved = resolve_path(configured, repo_root=repo_root, allow_unresolved=False)

    if resolved is None:
        raise SplitBuildError("Could not resolve split output directory.")

    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def get_split_output_paths(split_dir: Path) -> SplitOutputs:
    """Return all split output paths."""

    return SplitOutputs(
        split_assignments=split_dir / "split_assignments.parquet",
        split_metadata=split_dir / "split_metadata.json",
        split_report=split_dir / "split_report.md",
        split_validation=split_dir / "split_validation.json",
        target_thresholds_temporal=split_dir / "target_thresholds_temporal.json",
        target_thresholds_random_debug=split_dir / "target_thresholds_random_debug.json",
        target_thresholds_spatial_block=split_dir / "target_thresholds_spatial_block.json",
    )


def read_panel(path: Path) -> pd.DataFrame:
    """Read the validated tract-month panel."""

    require_runtime_dependencies()

    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        panel = pd.read_parquet(path)
    elif suffix == ".csv":
        panel = pd.read_csv(path, low_memory=False)
    else:
        raise SplitBuildError(f"Unsupported panel format: {path}")

    panel.columns = [str(col) for col in panel.columns]
    return panel


def normalize_period_month(series: pd.Series) -> pd.Series:
    """Normalize period_month values to YYYY-MM strings."""

    parsed = pd.to_datetime(series.astype(str), errors="coerce")
    if parsed.isna().any():
        bad = series[parsed.isna()].drop_duplicates().head(20).tolist()
        raise SplitBuildError(f"Could not parse period_month values: {bad}")
    return parsed.dt.to_period("M").astype(str)


def period_mask(periods: pd.Series, start: str, end: str) -> pd.Series:
    """Return inclusive period mask for YYYY-MM strings."""

    period_index = pd.PeriodIndex(periods.astype(str), freq="M")
    return (period_index >= pd.Period(start, freq="M")) & (
        period_index <= pd.Period(end, freq="M")
    )


def finalize_validation(
    checks: list[dict[str, Any]],
    extra_summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Finalize validation status from checks."""

    hard_failures = [
        check for check in checks
        if not check["passed"] and check["severity"] == "error"
    ]
    warnings = [
        check for check in checks
        if not check["passed"] and check["severity"] == "warning"
    ]

    return {
        "overall_status": "fail" if hard_failures else ("warning" if warnings else "pass"),
        "hard_failure_count": len(hard_failures),
        "warning_count": len(warnings),
        "checks": checks,
        "summary": to_jsonable(extra_summary),
    }


def validate_input_panel(panel: pd.DataFrame) -> dict[str, Any]:
    """Validate the panel before split construction."""

    checks: list[dict[str, Any]] = []

    def add_check(name: str, passed: bool, severity: str = "error", details: Any = None) -> None:
        checks.append(
            {
                "name": name,
                "passed": bool(passed),
                "severity": severity,
                "details": to_jsonable(details),
            }
        )

    missing_required = [col for col in [*ID_COLUMNS, TARGET_COLUMN] if col not in panel.columns]
    add_check(
        "required_columns_present",
        len(missing_required) == 0,
        details={"missing_required_columns": missing_required},
    )

    if missing_required:
        return finalize_validation(checks, extra_summary={})

    duplicate_rows = int(panel.duplicated(ID_COLUMNS).sum())
    add_check(
        "one_row_per_zone_month",
        duplicate_rows == 0,
        details={"duplicate_rows": duplicate_rows},
    )

    missing_zone = int(panel["zone_id"].isna().sum())
    add_check(
        "no_missing_zone_id",
        missing_zone == 0,
        details={"missing_zone_id_rows": missing_zone},
    )

    missing_period = int(panel["period_month"].isna().sum())
    add_check(
        "no_missing_period_month",
        missing_period == 0,
        details={"missing_period_month_rows": missing_period},
    )

    target = pd.to_numeric(panel[TARGET_COLUMN], errors="coerce")
    missing_target = int(target.isna().sum())
    negative_target = int((target < 0).sum())

    add_check(
        "target_is_numeric_non_missing",
        missing_target == 0,
        details={"missing_or_non_numeric_target_rows": missing_target},
    )
    add_check(
        "target_is_nonnegative",
        negative_target == 0,
        details={"negative_target_rows": negative_target},
    )

    sovi_cols = [col for col in panel.columns if "sovi" in col.lower()]
    add_check(
        "no_sovi_columns_in_track_a",
        len(sovi_cols) == 0,
        details={"sovi_like_columns": sovi_cols},
    )

    periods = sorted(panel["period_month"].astype(str).unique().tolist())
    zones = sorted(panel["zone_id"].astype(str).unique().tolist())

    expected_rows = len(periods) * len(zones)
    add_check(
        "complete_zone_month_panel_shape",
        len(panel) == expected_rows,
        severity="warning",
        details={
            "rows": len(panel),
            "n_zones": len(zones),
            "n_months": len(periods),
            "expected_rows_if_complete": expected_rows,
        },
    )

    return finalize_validation(
        checks,
        extra_summary={
            "panel_rows": len(panel),
            "n_zones": len(zones),
            "n_months": len(periods),
            "period_month_min": min(periods) if periods else None,
            "period_month_max": max(periods) if periods else None,
            "target_sum": float(target.sum()) if missing_target == 0 else None,
            "target_mean": float(target.mean()) if missing_target == 0 else None,
            "target_positive_rate": float((target > 0).mean()) if missing_target == 0 else None,
        },
    )


def assign_temporal_split(panel: pd.DataFrame) -> tuple[pd.Series, dict[str, Any]]:
    """Assign the frozen primary temporal split."""

    split = pd.Series(pd.NA, index=panel.index, dtype="object")

    for split_name, window in TEMPORAL_SPLIT_SPEC.items():
        mask = period_mask(panel["period_month"], window["start"], window["end"])
        split.loc[mask] = split_name

    unassigned = int(split.isna().sum())
    counts = split.value_counts(dropna=False).to_dict()
    month_counts = (
        panel.assign(_temporal_split=split)
        .groupby("_temporal_split", dropna=False)["period_month"]
        .nunique()
        .to_dict()
    )

    metadata = {
        "split_name": "temporal",
        "split_column": TEMPORAL_SPLIT_COL,
        "split_type": "temporal",
        "is_primary_scientific_split": True,
        "spec": TEMPORAL_SPLIT_SPEC,
        "row_counts": to_jsonable(counts),
        "month_counts": to_jsonable(month_counts),
        "unassigned_rows": unassigned,
        "notes": (
            "Frozen temporal split from baseline_plan_mtl_311_v0.md. "
            "No baseline metrics should be reported without this split artifact."
        ),
    }

    if unassigned:
        bad_periods = sorted(
            panel.loc[split.isna(), "period_month"].astype(str).unique().tolist()
        )
        raise SplitBuildError(
            "Temporal split left rows unassigned. "
            f"Unassigned periods: {bad_periods}"
        )

    return split, metadata


def assign_random_debug_split(panel: pd.DataFrame) -> tuple[pd.Series, dict[str, Any]]:
    """Assign a random tract-month split for debugging only."""

    seed = int(RANDOM_DEBUG_SPEC["seed"])
    rng = np.random.default_rng(seed)

    n = len(panel)
    order = rng.permutation(n)

    n_train = int(round(n * float(RANDOM_DEBUG_SPEC["train_fraction"])))
    n_validation = int(round(n * float(RANDOM_DEBUG_SPEC["validation_fraction"])))
    n_test = n - n_train - n_validation

    split_values = np.empty(n, dtype=object)
    split_values[order[:n_train]] = "train"
    split_values[order[n_train : n_train + n_validation]] = "validation"
    split_values[order[n_train + n_validation :]] = "test"

    split = pd.Series(split_values, index=panel.index, dtype="object")

    metadata = {
        "split_name": "random_debug",
        "split_column": RANDOM_DEBUG_SPLIT_COL,
        "split_type": "random_tract_month",
        "is_primary_scientific_split": False,
        "debug_only": True,
        "spec": RANDOM_DEBUG_SPEC,
        "row_counts": to_jsonable(split.value_counts().to_dict()),
        "notes": (
            "Random split is allowed for implementation checks only. "
            "It leaks spatial/temporal structure and is not main scientific evidence."
        ),
    }

    return split, metadata


def quantile_bins_for_coordinates(values: pd.Series, n_bins: int) -> pd.Series:
    """Create deterministic quantile-like bins for coordinate values."""

    if values.isna().any():
        raise SplitBuildError("Cannot build spatial blocks with missing centroid coordinates.")

    unique_count = values.nunique()
    effective_bins = min(int(n_bins), int(unique_count))

    if effective_bins <= 1:
        return pd.Series(0, index=values.index, dtype=int)

    ranks = values.rank(method="first")
    # qcut on ranks is stable even when coordinate values have ties.
    return pd.qcut(ranks, q=effective_bins, labels=False, duplicates="drop").astype(int)


def assign_spatial_block_split(panel: pd.DataFrame) -> tuple[pd.Series, dict[str, Any]]:
    """
    Assign a preliminary spatial block split.

    All months for a tract share the same split. Blocks are created from
    geometric tract centroid quantile bins. This is not a graph-training
    leakage-control implementation; graph code must still specify inductive or
    transductive handling.
    """

    required = ["zone_id", "tract_centroid_x", "tract_centroid_y"]
    missing = [col for col in required if col not in panel.columns]
    if missing:
        split = pd.Series(pd.NA, index=panel.index, dtype="object")
        metadata = {
            "split_name": "spatial_block",
            "split_column": SPATIAL_BLOCK_SPLIT_COL,
            "split_type": "spatial_block",
            "available": False,
            "reason": f"Missing required centroid columns: {missing}",
            "is_primary_scientific_split": False,
            "required_before_graph_claims": True,
        }
        return split, metadata

    zone_table = (
        panel[["zone_id", "tract_centroid_x", "tract_centroid_y"]]
        .drop_duplicates("zone_id")
        .copy()
    )
    zone_table["tract_centroid_x"] = pd.to_numeric(zone_table["tract_centroid_x"], errors="coerce")
    zone_table["tract_centroid_y"] = pd.to_numeric(zone_table["tract_centroid_y"], errors="coerce")

    if zone_table[["tract_centroid_x", "tract_centroid_y"]].isna().any().any():
        split = pd.Series(pd.NA, index=panel.index, dtype="object")
        metadata = {
            "split_name": "spatial_block",
            "split_column": SPATIAL_BLOCK_SPLIT_COL,
            "split_type": "spatial_block",
            "available": False,
            "reason": "Missing/non-numeric centroid values.",
            "is_primary_scientific_split": False,
            "required_before_graph_claims": True,
        }
        return split, metadata

    x_bins = int(SPATIAL_BLOCK_SPEC["x_bins"])
    y_bins = int(SPATIAL_BLOCK_SPEC["y_bins"])
    zone_table["x_block"] = quantile_bins_for_coordinates(zone_table["tract_centroid_x"], x_bins)
    zone_table["y_block"] = quantile_bins_for_coordinates(zone_table["tract_centroid_y"], y_bins)
    zone_table["spatial_block_id"] = (
        "x" + zone_table["x_block"].astype(str) + "_y" + zone_table["y_block"].astype(str)
    )

    block_table = (
        zone_table.groupby("spatial_block_id", as_index=False)
        .agg(n_zones=("zone_id", "nunique"))
    )

    block_table = (
        block_table
        .sample(frac=1.0, random_state=int(SPATIAL_BLOCK_SPEC["seed"]))
        .reset_index(drop=True)
    )

    total_zones = int(block_table["n_zones"].sum())
    train_target = total_zones * float(SPATIAL_BLOCK_SPEC["train_fraction"])
    validation_target = total_zones * float(SPATIAL_BLOCK_SPEC["validation_fraction"])

    assigned_blocks: list[str] = []
    train_zones = 0
    validation_zones = 0

    for _, row in block_table.iterrows():
        block_id = str(row["spatial_block_id"])
        n_zones = int(row["n_zones"])
        _ = block_id  # keep explicit for readability/debugging.

        if train_zones < train_target:
            split_name = "train"
            train_zones += n_zones
        elif validation_zones < validation_target:
            split_name = "validation"
            validation_zones += n_zones
        else:
            split_name = "test"

        assigned_blocks.append(split_name)

    block_table["split_spatial_block"] = assigned_blocks
    zone_table = zone_table.merge(
        block_table[["spatial_block_id", "split_spatial_block"]],
        on="spatial_block_id",
        how="left",
    )

    zone_to_split = dict(zip(zone_table["zone_id"].astype(str), zone_table["split_spatial_block"].astype(str)))
    split = panel["zone_id"].astype(str).map(zone_to_split)

    metadata = {
        "split_name": "spatial_block",
        "split_column": SPATIAL_BLOCK_SPLIT_COL,
        "split_type": "spatial_quantile_grid_blocks",
        "available": True,
        "is_primary_scientific_split": False,
        "required_before_graph_claims": True,
        "spec": SPATIAL_BLOCK_SPEC,
        "n_blocks": int(block_table["spatial_block_id"].nunique()),
        "n_zones": int(zone_table["zone_id"].nunique()),
        "row_counts": to_jsonable(split.value_counts(dropna=False).to_dict()),
        "zone_counts": to_jsonable(zone_table["split_spatial_block"].value_counts(dropna=False).to_dict()),
        "block_counts": to_jsonable(block_table["split_spatial_block"].value_counts(dropna=False).to_dict()),
        "notes": (
            "Preliminary spatial block split using tract centroid quantile-grid blocks. "
            "For graph models, later training code must still document transductive vs inductive evaluation."
        ),
    }

    return split, metadata


def assign_magnitude_classes(values: pd.Series, positive_thresholds: list[float]) -> pd.Series:
    """
    Assign magnitude classes 0–4.

    Baseline-plan rule:
      - class 0: y == 0
      - class 1: 0 < y <= positive_train_q25
      - class 2: q25 < y <= positive_train_q50
      - class 3: q50 < y <= positive_train_q75
      - class 4: y > positive_train_q75
    """

    y = pd.to_numeric(values, errors="coerce")
    if y.isna().any():
        raise SplitBuildError("Cannot assign magnitude classes with missing target values.")

    q25, q50, q75 = positive_thresholds

    classes = pd.Series(4, index=values.index, dtype=int)
    classes.loc[y <= q75] = 3
    classes.loc[y <= q50] = 2
    classes.loc[y <= q25] = 1
    classes.loc[y == 0] = 0
    return classes


def compute_train_only_magnitude_thresholds(
    panel: pd.DataFrame,
    split_series: pd.Series,
    split_name: str,
    target_col: str = TARGET_COLUMN,
) -> dict[str, Any]:
    """
    Fit magnitude-class thresholds on training rows only.

    This follows the final baseline plan:
      - class 0 is exactly water_drainage_count == 0
      - positive classes 1–4 are based on Q25/Q50/Q75 among positive train counts
    """

    train_mask = split_series == "train"
    train_values = pd.to_numeric(panel.loc[train_mask, target_col], errors="coerce").dropna()

    if train_values.empty:
        raise SplitBuildError(f"Cannot compute thresholds for {split_name}; no training rows.")

    positive_train_values = train_values.loc[train_values > 0]

    if positive_train_values.empty:
        raise SplitBuildError(
            f"Cannot compute positive-count thresholds for {split_name}; "
            "training target has no positive values."
        )

    quantile_levels = [0.25, 0.50, 0.75]
    quantiles = {
        str(level): float(np.quantile(positive_train_values.to_numpy(dtype=float), level))
        for level in quantile_levels
    }

    positive_thresholds = [
        quantiles["0.25"],
        quantiles["0.5"],
        quantiles["0.75"],
    ]

    classes = assign_magnitude_classes(panel[target_col], positive_thresholds)

    class_counts_by_split: dict[str, dict[str, int]] = {}
    temp = pd.DataFrame(
        {
            "split": split_series,
            "magnitude_class": classes,
        }
    )
    for split_value, part in temp.groupby("split", dropna=False):
        class_counts_by_split[str(split_value)] = {
            str(k): int(v)
            for k, v in part["magnitude_class"].value_counts().sort_index().to_dict().items()
        }

    return {
        "split_name": split_name,
        "target_column": target_col,
        "strategy": "class0_zero_positive_train_quantiles_25_50_75",
        "fitted_on": "train_rows_only",
        "train_row_count": int(train_mask.sum()),
        "train_positive_row_count": int((train_values > 0).sum()),
        "train_zero_row_count": int((train_values == 0).sum()),
        "train_target_summary": {
            "min": float(train_values.min()),
            "mean": float(train_values.mean()),
            "median": float(train_values.median()),
            "max": float(train_values.max()),
            "positive_rate": float((train_values > 0).mean()),
        },
        "positive_train_target_summary": {
            "min": float(positive_train_values.min()),
            "mean": float(positive_train_values.mean()),
            "median": float(positive_train_values.median()),
            "max": float(positive_train_values.max()),
        },
        "positive_quantiles": quantiles,
        "thresholds": {
            "class_0_rule": "y == 0",
            "class_1_max": positive_thresholds[0],
            "class_2_max": positive_thresholds[1],
            "class_3_max": positive_thresholds[2],
            "class_4_rule": f">{positive_thresholds[2]}",
        },
        "class_rule": (
            "0 if y == 0; 1 if 0 < y <= positive_train_q25; "
            "2 if q25 < y <= positive_train_q50; "
            "3 if q50 < y <= positive_train_q75; 4 if y > positive_train_q75."
        ),
        "class_counts_by_split": class_counts_by_split,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def validate_splits(assignments: pd.DataFrame, metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Validate split assignments and threshold artifacts."""

    checks: list[dict[str, Any]] = []

    def add_check(name: str, passed: bool, severity: str = "error", details: Any = None) -> None:
        checks.append(
            {
                "name": name,
                "passed": bool(passed),
                "severity": severity,
                "details": to_jsonable(details),
            }
        )

    duplicate_rows = int(assignments.duplicated(ID_COLUMNS).sum())
    add_check(
        "one_row_per_zone_month_in_split_assignments",
        duplicate_rows == 0,
        details={"duplicate_rows": duplicate_rows},
    )

    for col in [TEMPORAL_SPLIT_COL, RANDOM_DEBUG_SPLIT_COL]:
        missing = int(assignments[col].isna().sum())
        valid_values = sorted(assignments[col].dropna().unique().tolist())
        add_check(
            f"{col}_covers_every_row",
            missing == 0,
            details={"missing_rows": missing, "values": valid_values},
        )

        counts = assignments[col].value_counts().to_dict()
        add_check(
            f"{col}_has_train_validation_test",
            all(label in counts and counts[label] > 0 for label in ["train", "validation", "test"]),
            details={"counts": counts},
        )

    if SPATIAL_BLOCK_SPLIT_COL in assignments.columns:
        missing_spatial = int(assignments[SPATIAL_BLOCK_SPLIT_COL].isna().sum())
        spatial_available = bool(metadata.get("splits", {}).get("spatial_block", {}).get("available", False))
        add_check(
            "spatial_block_split_status_documented",
            True,
            severity="info",
            details={
                "available": spatial_available,
                "missing_rows": missing_spatial,
            },
        )

        if spatial_available:
            counts = assignments[SPATIAL_BLOCK_SPLIT_COL].value_counts().to_dict()
            add_check(
                "spatial_block_split_has_train_validation_test",
                all(label in counts and counts[label] > 0 for label in ["train", "validation", "test"]),
                severity="warning",
                details={"counts": counts},
            )

            # All months for a zone should share the same spatial split.
            per_zone = assignments.groupby("zone_id")[SPATIAL_BLOCK_SPLIT_COL].nunique(dropna=True)
            bad_zones = per_zone[per_zone > 1]
            add_check(
                "spatial_block_split_constant_within_zone",
                len(bad_zones) == 0,
                details={"bad_zone_count": int(len(bad_zones)), "examples": bad_zones.head(20).index.tolist()},
            )

    temporal_periods = assignments.groupby(TEMPORAL_SPLIT_COL)["period_month"].nunique().to_dict()
    add_check(
        "temporal_split_month_counts_match_plan",
        temporal_periods.get("train") == 36
        and temporal_periods.get("validation") == 8
        and temporal_periods.get("test") == 9,
        details={"month_counts": temporal_periods},
    )

    for col in [
        "magnitude_class_temporal",
        "magnitude_class_random_debug",
        "magnitude_class_spatial_block",
    ]:
        if col not in assignments.columns:
            continue

        missing = int(assignments[col].isna().sum())
        values = sorted(assignments[col].dropna().unique().tolist())
        add_check(
            f"{col}_valid_0_to_4",
            missing == 0 and set(values).issubset({0, 1, 2, 3, 4}),
            details={"missing_rows": missing, "values": values},
        )

    return finalize_validation(
        checks,
        extra_summary={
            "split_rows": len(assignments),
            "n_zones": int(assignments["zone_id"].nunique()),
            "n_months": int(assignments["period_month"].nunique()),
        },
    )


def build_split_assignments(panel: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any], dict[str, dict[str, Any]]]:
    """Build all split assignment columns and train-only threshold metadata."""

    panel = panel.copy()
    panel["period_month"] = normalize_period_month(panel["period_month"])

    input_validation = validate_input_panel(panel)
    if input_validation["overall_status"] == "fail":
        raise SplitBuildError(
            f"Input panel failed validation: {input_validation['checks']}"
        )

    assignments = panel[["zone_id", "period_month", TARGET_COLUMN]].copy()
    if "year" in panel.columns:
        assignments["year"] = panel["year"]
    if "month" in panel.columns:
        assignments["month"] = panel["month"]

    temporal_split, temporal_meta = assign_temporal_split(panel)
    random_split, random_meta = assign_random_debug_split(panel)
    spatial_split, spatial_meta = assign_spatial_block_split(panel)

    assignments[TEMPORAL_SPLIT_COL] = temporal_split
    assignments[RANDOM_DEBUG_SPLIT_COL] = random_split
    assignments[SPATIAL_BLOCK_SPLIT_COL] = spatial_split

    thresholds: dict[str, dict[str, Any]] = {}

    thresholds["temporal"] = compute_train_only_magnitude_thresholds(
        panel, temporal_split, split_name="temporal"
    )
    thresholds["random_debug"] = compute_train_only_magnitude_thresholds(
        panel, random_split, split_name="random_debug"
    )

    assignments["magnitude_class_temporal"] = assign_magnitude_classes(
        assignments[TARGET_COLUMN],
        [
            thresholds["temporal"]["thresholds"]["class_1_max"],
            thresholds["temporal"]["thresholds"]["class_2_max"],
            thresholds["temporal"]["thresholds"]["class_3_max"],
        ],
    )
    assignments["magnitude_class_random_debug"] = assign_magnitude_classes(
        assignments[TARGET_COLUMN],
        [
            thresholds["random_debug"]["thresholds"]["class_1_max"],
            thresholds["random_debug"]["thresholds"]["class_2_max"],
            thresholds["random_debug"]["thresholds"]["class_3_max"],
        ],
    )

    if spatial_meta.get("available", False) and not spatial_split.isna().any():
        thresholds["spatial_block"] = compute_train_only_magnitude_thresholds(
            panel, spatial_split, split_name="spatial_block"
        )
        assignments["magnitude_class_spatial_block"] = assign_magnitude_classes(
            assignments[TARGET_COLUMN],
            [
                thresholds["spatial_block"]["thresholds"]["class_1_max"],
                thresholds["spatial_block"]["thresholds"]["class_2_max"],
                thresholds["spatial_block"]["thresholds"]["class_3_max"],
            ],
        )
    else:
        thresholds["spatial_block"] = {
            "split_name": "spatial_block",
            "available": False,
            "reason": spatial_meta.get("reason", "spatial split unavailable"),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        assignments["magnitude_class_spatial_block"] = pd.NA

    metadata = {
        "splits": {
            "temporal": temporal_meta,
            "random_debug": random_meta,
            "spatial_block": spatial_meta,
        },
        "input_validation": input_validation,
        "target_column": TARGET_COLUMN,
        "target_derived_columns_excluded_from_features": sorted(TARGET_DERIVED_COLUMNS),
        "notes": {
            "primary_scientific_split": "temporal",
            "random_split": "debugging_only",
            "spatial_block_split": (
                "preliminary; graph models must still specify transductive/inductive "
                "message-passing handling."
            ),
            "magnitude_classes": (
                "Classes are split-specific artifacts. Thresholds are fitted on "
                "training rows only and are not written back into Dataset v0."
            ),
        },
    }

    return assignments, metadata, thresholds


def write_dataframe_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write DataFrame to parquet."""

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def render_split_report(
    assignments: pd.DataFrame,
    metadata: Mapping[str, Any],
    validation: Mapping[str, Any],
    thresholds: Mapping[str, Mapping[str, Any]],
    outputs: SplitOutputs,
    panel_path: Path,
) -> str:
    """Render split report as Markdown."""

    lines: list[str] = []

    lines.append("# Split Report — Montréal 311 Water/Drainage v0\n")
    lines.append(f"Generated at: `{metadata.get('generated_at')}`\n")
    lines.append(f"Benchmark ID: `{metadata.get('benchmark_id')}`\n")
    lines.append(f"Panel path: `{panel_path}`\n")
    lines.append(f"Validation status: `{validation.get('overall_status')}`\n")

    lines.append("## Summary\n")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    lines.append(f"| rows | {len(assignments)} |")
    lines.append(f"| zones | {assignments['zone_id'].nunique()} |")
    lines.append(f"| months | {assignments['period_month'].nunique()} |")
    lines.append(f"| period min | {assignments['period_month'].min()} |")
    lines.append(f"| period max | {assignments['period_month'].max()} |")
    lines.append("")

    lines.append("## Temporal split — primary scientific split\n")
    lines.append("| Split | Period | Rows | Months |")
    lines.append("|---|---|---:|---:|")
    temporal_spec = metadata["splits"]["temporal"]["spec"]
    for split_name in ["train", "validation", "test"]:
        rows = int((assignments[TEMPORAL_SPLIT_COL] == split_name).sum())
        months = int(assignments.loc[assignments[TEMPORAL_SPLIT_COL] == split_name, "period_month"].nunique())
        window = temporal_spec[split_name]
        lines.append(
            f"| `{split_name}` | {window['start']} to {window['end']} | {rows} | {months} |"
        )
    lines.append("")

    lines.append("## Random debug split\n")
    lines.append("Random split is for implementation checks only and is not main scientific evidence.\n")
    lines.append("| Split | Rows |")
    lines.append("|---|---:|")
    for split_name, count in assignments[RANDOM_DEBUG_SPLIT_COL].value_counts().to_dict().items():
        lines.append(f"| `{split_name}` | {count} |")
    lines.append("")

    lines.append("## Spatial block split\n")
    spatial_meta = metadata["splits"]["spatial_block"]
    lines.append(f"Available: `{spatial_meta.get('available')}`\n")
    if spatial_meta.get("available"):
        lines.append(
            "This is a preliminary spatial split based on tract centroid quantile-grid blocks. "
            "Graph-specific leakage control remains a later modeling responsibility.\n"
        )
        lines.append("| Split | Rows |")
        lines.append("|---|---:|")
        for split_name, count in assignments[SPATIAL_BLOCK_SPLIT_COL].value_counts().to_dict().items():
            lines.append(f"| `{split_name}` | {count} |")
        lines.append("")
    else:
        lines.append(f"Reason: `{spatial_meta.get('reason')}`\n")

    lines.append("## Train-only magnitude thresholds\n")
    for split_name, payload in thresholds.items():
        lines.append(f"### `{split_name}`\n")
        if not payload.get("available", True):
            lines.append(f"Unavailable: `{payload.get('reason')}`\n")
            continue

        lines.append("| Threshold | Value |")
        lines.append("|---|---:|")
        for key, value in payload.get("thresholds", {}).items():
            lines.append(f"| `{key}` | `{value}` |")
        lines.append("")

    lines.append("## Validation checks\n")
    lines.append("| Check | Passed | Severity | Details |")
    lines.append("|---|:---:|---|---|")
    for check in validation.get("checks", []):
        details = json.dumps(check.get("details"), ensure_ascii=False)[:300]
        lines.append(
            f"| `{check.get('name')}` | `{check.get('passed')}` | "
            f"`{check.get('severity')}` | `{details}` |"
        )
    lines.append("")

    lines.append("## Output artifacts\n")
    lines.append("| Artifact | Path |")
    lines.append("|---|---|")
    for key, value in outputs.to_dict().items():
        lines.append(f"| `{key}` | `{value}` |")
    lines.append("")

    lines.append("## Leakage notes\n")
    lines.append(
        "- Magnitude thresholds are fitted on training rows only for each split scheme.\n"
        "- Magnitude class 0 is strictly `water_drainage_count == 0`.\n"
        "- Positive magnitude classes 1–4 use quantiles of positive training counts only.\n"
        "- Random split is debugging-only and must not be used as primary scientific evidence.\n"
        "- Same-month target-derived columns must be excluded from model features.\n"
        "- Spatial block split is preliminary; graph evaluation must later document inductive vs transductive handling.\n"
    )

    return "\n".join(lines)


def build_metadata(
    config: Mapping[str, Any],
    config_path: Path,
    repo_root: Path,
    panel_path: Path,
    assignments: pd.DataFrame,
    split_metadata: Mapping[str, Any],
    outputs: SplitOutputs,
) -> dict[str, Any]:
    """Build split metadata."""

    metadata = {
        "benchmark_id": config.get("benchmark_id", "mtl_311_water_v0"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(repo_root),
        "config_path": str(config_path),
        "config_hash": config_hash(config),
        "panel_path": str(panel_path),
        "panel_sha256": file_hash(panel_path),
        "n_rows": int(len(assignments)),
        "n_zones": int(assignments["zone_id"].nunique()),
        "n_months": int(assignments["period_month"].nunique()),
        "period_month_min": str(assignments["period_month"].min()),
        "period_month_max": str(assignments["period_month"].max()),
        "outputs": outputs.to_dict(),
        **split_metadata,
    }

    return to_jsonable(metadata)


def run_build_splits(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """
    Build split artifacts for Dataset v0.

    Returns a dictionary with output paths, validation status, and split metadata.
    """

    require_runtime_dependencies()

    root = Path(repo_root).resolve() if repo_root is not None else find_repo_root()
    resolved_config_path = resolve_path(config_path, repo_root=root, allow_unresolved=False)

    if resolved_config_path is None:
        raise SplitBuildError(f"Could not resolve config path: {config_path}")

    config = load_config(resolved_config_path)
    panel_path = resolve_panel_path(config, repo_root=root)
    split_dir = resolve_split_dir(config, repo_root=root)
    outputs = get_split_output_paths(split_dir)

    panel = read_panel(panel_path)
    assignments, split_metadata, thresholds = build_split_assignments(panel)

    metadata = build_metadata(
        config=config,
        config_path=resolved_config_path,
        repo_root=root,
        panel_path=panel_path,
        assignments=assignments,
        split_metadata=split_metadata,
        outputs=outputs,
    )

    validation = validate_splits(assignments, metadata)

    # Write artifacts.
    write_dataframe_parquet(assignments, outputs.split_assignments)
    write_json(outputs.target_thresholds_temporal, thresholds["temporal"])
    write_json(outputs.target_thresholds_random_debug, thresholds["random_debug"])
    write_json(outputs.target_thresholds_spatial_block, thresholds["spatial_block"])
    write_json(outputs.split_metadata, metadata)
    write_json(outputs.split_validation, validation)
    write_markdown(
        outputs.split_report,
        render_split_report(
            assignments=assignments,
            metadata=metadata,
            validation=validation,
            thresholds=thresholds,
            outputs=outputs,
            panel_path=panel_path,
        ),
    )

    result = {
        "status": validation.get("overall_status"),
        "outputs": outputs.to_dict(),
        "validation": validation,
        "metadata": metadata,
        "thresholds": thresholds,
    }

    if validation.get("overall_status") == "fail":
        hard_failures = [
            check for check in validation.get("checks", [])
            if not check.get("passed") and check.get("severity") == "error"
        ]
        raise SplitBuildError(f"Split build failed validation. Hard failures: {hard_failures}")

    return result


def split_brief(result: Mapping[str, Any]) -> str:
    """Return a concise split-build summary."""

    metadata = result.get("metadata", {})
    splits = metadata.get("splits", {})
    temporal = splits.get("temporal", {})
    random_debug = splits.get("random_debug", {})
    spatial = splits.get("spatial_block", {})

    return (
        "Split artifacts built.\n"
        f"Status: {result.get('status')}\n"
        f"Rows: {metadata.get('n_rows')}\n"
        f"Zones: {metadata.get('n_zones')}\n"
        f"Months: {metadata.get('n_months')}\n"
        f"Period: {metadata.get('period_month_min')} to {metadata.get('period_month_max')}\n"
        f"Temporal rows: {temporal.get('row_counts')}\n"
        f"Random-debug rows: {random_debug.get('row_counts')}\n"
        f"Spatial block available: {spatial.get('available')}\n"
    )


def main() -> None:
    """CLI entry point for direct module execution."""

    import argparse

    parser = argparse.ArgumentParser(
        description="Build split artifacts for the Montréal 311 water/drainage benchmark."
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

    args = parser.parse_args()

    result = run_build_splits(
        config_path=args.config,
        repo_root=args.repo_root,
    )

    print(split_brief(result).rstrip())
    print("\nWritten outputs:")
    for label, path in result.get("outputs", {}).items():
        print(f"  {label}: {path}")


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_PANEL_PATH",
    "DEFAULT_SPLIT_DIR",
    "RANDOM_DEBUG_SPEC",
    "SPATIAL_BLOCK_SPEC",
    "TARGET_COLUMN",
    "TEMPORAL_SPLIT_SPEC",
    "SplitBuildError",
    "SplitOutputs",
    "assign_magnitude_classes",
    "assign_random_debug_split",
    "assign_spatial_block_split",
    "assign_temporal_split",
    "build_split_assignments",
    "compute_train_only_magnitude_thresholds",
    "run_build_splits",
    "split_brief",
]