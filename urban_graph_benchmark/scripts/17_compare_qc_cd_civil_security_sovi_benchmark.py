#!/usr/bin/env python3
"""
Final comparison script for the Québec CD civil-security / SoVI benchmark.

Purpose:
    Merge B1/B0/B2/B3/B4 metrics into final comparison tables and write a
    Markdown report answering the central benchmark question:

        Does graph structure add value beyond SoVI, history, tabular ML,
        no-edge neural controls, and random/placebo graph controls?

Outputs:
    urban_graph_benchmark/outputs/qc_cd_civil_security_sovi_v0/comparisons/benchmark_comparison.csv
    urban_graph_benchmark/outputs/qc_cd_civil_security_sovi_v0/comparisons/benchmark_comparison_compact.csv
    urban_graph_benchmark/outputs/qc_cd_civil_security_sovi_v0/comparisons/metrics_long.csv
    urban_graph_benchmark/outputs/qc_cd_civil_security_sovi_v0/comparisons/metric_winners.csv
    urban_graph_benchmark/outputs/qc_cd_civil_security_sovi_v0/reports/qc_cd_civil_security_sovi_benchmark_report.md

Run from repository root:
    PYTHONPATH=urban_graph_benchmark/src python \
      urban_graph_benchmark/scripts/17_compare_qc_cd_civil_security_sovi_benchmark.py
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

try:
    from ville_hgnn.baselines.qc_cd_sovi_common import (
        BASELINES_DIR,
        COMPARISONS_DIR,
        REPORTS_DIR,
        OUTPUT_ROOT,
        CD_ID_COL,
        SPLIT_COL,
        ensure_dir,
        write_metadata_json,
    )
except Exception:  # pragma: no cover - fallback for early bootstrapping only
    OUTPUT_ROOT = Path("urban_graph_benchmark/outputs/qc_cd_civil_security_sovi_v0")
    BASELINES_DIR = OUTPUT_ROOT / "baselines"
    COMPARISONS_DIR = OUTPUT_ROOT / "comparisons"
    REPORTS_DIR = OUTPUT_ROOT / "reports"
    CD_ID_COL = "cd_id_norm"
    SPLIT_COL = "split"

    def ensure_dir(path: str | Path) -> Path:
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        return out

    def write_metadata_json(metadata: dict[str, Any], path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
        return out


BENCHMARK_SPECS = [
    {
        "baseline_family": "B1_sovi_direct_validation",
        "baseline_stage": "B1",
        "method_class": "static_index",
        "display_name": "B1 direct SoVI validation",
        "directory": "B1_sovi_direct_validation",
        "graph_kind": "none",
        "uses_sovi": True,
        "uses_history": False,
        "uses_tabular_ml": False,
        "uses_neural": False,
        "uses_graph": False,
        "primary_role": "Static index validation",
    },
    {
        "baseline_family": "B0_history_only",
        "baseline_stage": "B0",
        "method_class": "history_baseline",
        "display_name": "B0 history-only",
        "directory": "B0_history_only",
        "graph_kind": "none",
        "uses_sovi": False,
        "uses_history": True,
        "uses_tabular_ml": False,
        "uses_neural": False,
        "uses_graph": False,
        "primary_role": "History baseline",
    },
    {
        "baseline_family": "B2_calibrated_sovi",
        "baseline_stage": "B2",
        "method_class": "calibrated_index",
        "display_name": "B2 calibrated SoVI",
        "directory": "B2_calibrated_sovi",
        "graph_kind": "none",
        "uses_sovi": True,
        "uses_history": False,
        "uses_tabular_ml": True,
        "uses_neural": False,
        "uses_graph": False,
        "primary_role": "Calibrated index baseline",
    },
    {
        "baseline_family": "B3_tabular_feature_parity",
        "baseline_stage": "B3",
        "method_class": "tabular_feature_parity",
        "display_name": "B3 tabular feature parity",
        "directory": "B3_tabular_feature_parity",
        "graph_kind": "none",
        "uses_sovi": True,
        "uses_history": True,
        "uses_tabular_ml": True,
        "uses_neural": False,
        "uses_graph": False,
        "primary_role": "Non-graph feature-parity baseline",
    },
    {
        "baseline_family": "B4_no_edge_neural",
        "baseline_stage": "B4",
        "method_class": "neural_no_edge",
        "display_name": "B4 no-edge neural",
        "directory": "B4_no_edge_neural",
        "graph_kind": "none",
        "uses_sovi": True,
        "uses_history": True,
        "uses_tabular_ml": False,
        "uses_neural": True,
        "uses_graph": False,
        "primary_role": "Neural no-topology control",
    },
    {
        "baseline_family": "B4_random_edge_graph",
        "baseline_stage": "B4",
        "method_class": "graph_placebo",
        "display_name": "B4 random-edge graph",
        "directory": "B4_random_edge_graph",
        "graph_kind": "random_placebo",
        "uses_sovi": True,
        "uses_history": True,
        "uses_tabular_ml": False,
        "uses_neural": True,
        "uses_graph": True,
        "primary_role": "Placebo topology control",
    },
    {
        "baseline_family": "B4_knn_graph",
        "baseline_stage": "B4",
        "method_class": "graph_knn",
        "display_name": "B4 kNN graph",
        "directory": "B4_knn_graph",
        "graph_kind": "centroid_knn",
        "uses_sovi": True,
        "uses_history": True,
        "uses_tabular_ml": False,
        "uses_neural": True,
        "uses_graph": True,
        "primary_role": "Spatial-proximity graph control",
    },
    {
        "baseline_family": "B4_real_cd_graph",
        "baseline_stage": "B4",
        "method_class": "graph_real_adjacency",
        "display_name": "B4 real CD adjacency graph",
        "directory": "B4_real_cd_graph",
        "graph_kind": "real_adjacency",
        "uses_sovi": True,
        "uses_history": True,
        "uses_tabular_ml": False,
        "uses_neural": True,
        "uses_graph": True,
        "primary_role": "Real topology graph model",
    },
]

LOWER_IS_BETTER = {
    "mae",
    "rmse",
    "mean_poisson_deviance",
    "poisson_deviance",
    "mse",
}
HIGHER_IS_BETTER = {
    "spearman",
    "spearman_corr",
    "kendall",
    "kendall_tau",
    "kendall_tau_b",
    "ndcg_at_10",
    "ndcg_at_25",
    "ndcg_at_100",
    "top10_overlap",
    "top10_overlap_rate",
    "top_10_percent_overlap",
    "top25_overlap",
    "top25_overlap_rate",
    "top_25_percent_overlap",
    "r2",
}

CORE_METRICS = [
    "mae",
    "rmse",
    "mean_poisson_deviance",
    "spearman",
    "kendall_tau",
    "ndcg_at_10",
    "ndcg_at_25",
    "ndcg_at_100",
    "top10_overlap_rate",
    "top25_overlap_rate",
]

COMPACT_METRICS = [
    "mae",
    "rmse",
    "mean_poisson_deviance",
    "spearman",
    "ndcg_at_25",
    "top10_overlap_rate",
]

METRIC_ALIASES = {
    "spearman_corr": "spearman",
    "spearmanrho": "spearman",
    "kendall": "kendall_tau",
    "kendall_tau_b": "kendall_tau",
    "kendalltau": "kendall_tau",
    "ndcg@10": "ndcg_at_10",
    "ndcg_10": "ndcg_at_10",
    "ndcg@25": "ndcg_at_25",
    "ndcg_25": "ndcg_at_25",
    "ndcg@100": "ndcg_at_100",
    "ndcg_100": "ndcg_at_100",
    "top10_overlap": "top10_overlap_rate",
    "top_10_percent_overlap": "top10_overlap_rate",
    "top10pct_overlap": "top10_overlap_rate",
    "top25_overlap": "top25_overlap_rate",
    "top_25_percent_overlap": "top25_overlap_rate",
    "poisson_deviance": "mean_poisson_deviance",
}


@dataclass
class Config:
    """Comparison-script configuration."""

    baselines_dir: Path = BASELINES_DIR
    comparisons_dir: Path = COMPARISONS_DIR
    reports_dir: Path = REPORTS_DIR

    primary_split: str = "test"
    fallback_splits: tuple[str, ...] = ("validation", "val", "all", "static", "cumulative_static")
    primary_metric: str = "mae"

    # B1 is a cumulative/static validation, not a forecast test split. Keep it
    # in the comparison but do not force it into the graph-value verdict.
    include_b1_in_graph_value_verdict: bool = False

    # Recompute metrics from predictions when possible, because this is more
    # robust to earlier metric-file format differences.
    prefer_recomputed_prediction_metrics: bool = True


def ensure_path_fields(config: Config) -> None:
    for name in ["baselines_dir", "comparisons_dir", "reports_dir"]:
        value = getattr(config, name)
        if not isinstance(value, Path):
            setattr(config, name, Path(value))


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table suffix: {path}")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_col_name(col: str) -> str:
    out = str(col).strip()
    out = out.replace("@", "_at_")
    out = re.sub(r"[^A-Za-z0-9]+", "_", out)
    out = re.sub(r"_+", "_", out).strip("_").lower()
    return METRIC_ALIASES.get(out, out)


def canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [normalize_col_name(c) for c in out.columns]
    return out


def metric_direction(metric: str) -> str:
    metric = normalize_col_name(metric)
    if metric in LOWER_IS_BETTER:
        return "lower_is_better"
    if metric in HIGHER_IS_BETTER:
        return "higher_is_better"
    if metric.startswith("ndcg") or "overlap" in metric or "spearman" in metric or "kendall" in metric:
        return "higher_is_better"
    if "mae" in metric or "rmse" in metric or "deviance" in metric or metric == "mse":
        return "lower_is_better"
    return "unknown"


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def rmse(y: np.ndarray, p: np.ndarray) -> float:
    mask = np.isfinite(y) & np.isfinite(p)
    if not mask.any():
        return np.nan
    return float(np.sqrt(np.mean((y[mask] - p[mask]) ** 2)))


def mae(y: np.ndarray, p: np.ndarray) -> float:
    mask = np.isfinite(y) & np.isfinite(p)
    if not mask.any():
        return np.nan
    return float(np.mean(np.abs(y[mask] - p[mask])))


def mean_poisson_deviance(y: np.ndarray, p: np.ndarray) -> float:
    mask = np.isfinite(y) & np.isfinite(p) & (y >= 0)
    if not mask.any():
        return np.nan

    y = y[mask].astype(float)
    p = np.clip(p[mask].astype(float), 1e-12, None)

    terms = np.where(y == 0, p, y * np.log(np.clip(y / p, 1e-12, None)) - y + p)
    return float(2.0 * np.mean(terms))


def rank_corr(y: np.ndarray, p: np.ndarray, method: str) -> float:
    mask = np.isfinite(y) & np.isfinite(p)
    if mask.sum() < 3:
        return np.nan

    try:
        return float(pd.Series(y[mask]).corr(pd.Series(p[mask]), method=method))
    except Exception:
        return np.nan


def ndcg_at_k(y: np.ndarray, p: np.ndarray, k: int) -> float:
    mask = np.isfinite(y) & np.isfinite(p)
    if mask.sum() == 0:
        return np.nan

    y = np.clip(y[mask].astype(float), 0.0, None)
    p = p[mask].astype(float)

    n = len(y)
    k_eff = min(int(k), n)
    if k_eff <= 0:
        return np.nan

    order = np.argsort(-p)
    ideal = np.argsort(-y)

    discounts = 1.0 / np.log2(np.arange(2, k_eff + 2))
    dcg = float(np.sum(y[order[:k_eff]] * discounts))
    idcg = float(np.sum(y[ideal[:k_eff]] * discounts))

    if idcg <= 0:
        return np.nan
    return dcg / idcg


def top_overlap_rate(y: np.ndarray, p: np.ndarray, frac: float) -> float:
    mask = np.isfinite(y) & np.isfinite(p)
    if mask.sum() == 0:
        return np.nan

    y = y[mask].astype(float)
    p = p[mask].astype(float)
    n = len(y)
    k = max(1, int(math.ceil(frac * n)))

    true_top = set(np.argsort(-y)[:k].tolist())
    pred_top = set(np.argsort(-p)[:k].tolist())

    return float(len(true_top & pred_top) / k)


def compute_metrics_for_group(df: pd.DataFrame) -> dict[str, Any]:
    y = safe_numeric(df["target"]).to_numpy(dtype=float)
    p = safe_numeric(df["prediction"]).to_numpy(dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)

    return {
        "n": int(mask.sum()),
        "target_sum": float(np.nansum(y[mask])) if mask.any() else np.nan,
        "prediction_sum": float(np.nansum(p[mask])) if mask.any() else np.nan,
        "target_mean": float(np.nanmean(y[mask])) if mask.any() else np.nan,
        "prediction_mean": float(np.nanmean(p[mask])) if mask.any() else np.nan,
        "mae": mae(y, p),
        "rmse": rmse(y, p),
        "mean_poisson_deviance": mean_poisson_deviance(y, p),
        "spearman": rank_corr(y, p, "spearman"),
        "kendall_tau": rank_corr(y, p, "kendall"),
        "ndcg_at_10": ndcg_at_k(y, p, 10),
        "ndcg_at_25": ndcg_at_k(y, p, 25),
        "ndcg_at_100": ndcg_at_k(y, p, 100),
        "top10_overlap_rate": top_overlap_rate(y, p, 0.10),
        "top25_overlap_rate": top_overlap_rate(y, p, 0.25),
    }


def find_prediction_file(directory: Path) -> Path | None:
    for name in ["predictions.parquet", "predictions.csv"]:
        path = directory / name
        if path.exists():
            return path
    return None


def find_metric_file_candidates(directory: Path) -> list[Path]:
    if not directory.exists():
        return []

    priority = [
        "metrics.csv",
        "validation_metrics.csv",
        "sovi_validation_metrics.csv",
        "target_sensitivity_metrics.csv",
        "rank_metrics.csv",
    ]

    out: list[Path] = []
    for name in priority:
        path = directory / name
        if path.exists():
            out.append(path)

    exclude_tokens = [
        "candidate",
        "bootstrap",
        "permutation",
        "null",
        "failure",
        "selection",
        "summary",
        "audit",
        "schema",
        "top",
        "feature",
        "metadata",
    ]

    for path in sorted(directory.glob("*metrics*.csv")):
        lower = path.name.lower()
        if path in out:
            continue
        if any(tok in lower for tok in exclude_tokens):
            continue
        out.append(path)

    return out


def metadata_for_spec(spec: Mapping[str, Any], directory: Path) -> dict[str, Any]:
    meta = read_json(directory / "metadata.json")
    out = {
        **spec,
        "source_directory": str(directory),
    }

    # Trust module metadata when available, but keep canonical names from spec.
    if meta:
        out["metadata_json"] = str(directory / "metadata.json")
        if "selected_model" in meta:
            out["selected_model_from_metadata"] = meta["selected_model"]
        if "training" in meta:
            out["training"] = meta["training"]
        if "features" in meta:
            out["features"] = meta["features"]
        if "graph_audit" in meta:
            out["graph_audit"] = meta["graph_audit"]

    return out


def add_spec_columns(df: pd.DataFrame, spec: Mapping[str, Any], source_file: Path) -> pd.DataFrame:
    out = df.copy()

    for key in [
        "baseline_family",
        "baseline_stage",
        "method_class",
        "display_name",
        "graph_kind",
        "uses_sovi",
        "uses_history",
        "uses_tabular_ml",
        "uses_neural",
        "uses_graph",
        "primary_role",
    ]:
        if key not in out.columns or out[key].isna().all():
            out[key] = spec.get(key)

    out["source_file"] = str(source_file)

    return out


def normalize_split_value(value: Any, *, default: str) -> str:
    if pd.isna(value):
        return default
    text = str(value).strip().lower()
    if text in {"", "nan", "none", "<na>"}:
        return default
    if text == "val":
        return "validation"
    return text


def recompute_metrics_from_predictions(
    directory: Path,
    spec: Mapping[str, Any],
) -> tuple[pd.DataFrame | None, Path | None]:
    pred_path = find_prediction_file(directory)
    if pred_path is None:
        return None, None

    pred = read_table(pred_path)
    pred = canonicalize_columns(pred)

    if "target" not in pred.columns or "prediction" not in pred.columns:
        return None, pred_path

    if "model_name" not in pred.columns:
        pred["model_name"] = spec["baseline_family"]
    if SPLIT_COL not in pred.columns:
        pred[SPLIT_COL] = "static" if spec["baseline_stage"] == "B1" else "all"
    if "candidate_name" not in pred.columns:
        pred["candidate_name"] = pd.NA
    if "target_col" not in pred.columns:
        pred["target_col"] = pd.NA
    if "graph_kind" not in pred.columns:
        pred["graph_kind"] = spec.get("graph_kind", "none")

    group_cols = ["model_name", "candidate_name", "graph_kind", "target_col", SPLIT_COL]

    rows: list[dict[str, Any]] = []
    for key, sub in pred.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, key))
        row[SPLIT_COL] = normalize_split_value(row[SPLIT_COL], default="all")
        row.update(compute_metrics_for_group(sub))
        rows.append(row)

    # Add overall/all row per model, except when already all-only.
    for key, sub in pred.groupby(["model_name", "candidate_name", "graph_kind", "target_col"], dropna=False):
        row = dict(zip(["model_name", "candidate_name", "graph_kind", "target_col"], key))
        row[SPLIT_COL] = "all"
        row.update(compute_metrics_for_group(sub))
        rows.append(row)

    metrics = pd.DataFrame(rows)
    metrics = add_spec_columns(metrics, spec, pred_path)
    metrics["metric_source_type"] = "recomputed_from_predictions"
    return metrics, pred_path


def standardize_metrics_file(
    metric_path: Path,
    spec: Mapping[str, Any],
) -> pd.DataFrame | None:
    try:
        raw = pd.read_csv(metric_path)
    except Exception:
        return None

    if raw.empty:
        return None

    df = canonicalize_columns(raw)

    # Avoid long-form files being double-longed; keep them but standardize.
    if "metric" in df.columns and "value" in df.columns:
        wide = df.pivot_table(
            index=[c for c in df.columns if c not in {"metric", "value"}],
            columns="metric",
            values="value",
            aggfunc="first",
        ).reset_index()
        wide.columns = [normalize_col_name(c) for c in wide.columns]
        df = wide

    # If no recognizable performance metric exists, skip.
    recognized = [m for m in CORE_METRICS if m in df.columns]
    if not recognized:
        # Try aliases already handled by canonicalize; if still no core metric, skip.
        return None

    if "model_name" not in df.columns:
        df["model_name"] = spec["baseline_family"]
    if "candidate_name" not in df.columns:
        df["candidate_name"] = pd.NA
    if "graph_kind" not in df.columns:
        df["graph_kind"] = spec.get("graph_kind", "none")
    if "target_col" not in df.columns:
        # B1 direct validation may have target or target_name columns.
        if "target" in df.columns:
            df["target_col"] = df["target"]
        elif "target_name" in df.columns:
            df["target_col"] = df["target_name"]
        else:
            df["target_col"] = pd.NA

    if SPLIT_COL not in df.columns:
        if "evaluation_split" in df.columns:
            df[SPLIT_COL] = df["evaluation_split"]
        elif "scope" in df.columns:
            df[SPLIT_COL] = df["scope"]
        else:
            df[SPLIT_COL] = "cumulative_static" if spec["baseline_stage"] == "B1" else "all"

    df[SPLIT_COL] = df[SPLIT_COL].map(
        lambda x: normalize_split_value(x, default="cumulative_static" if spec["baseline_stage"] == "B1" else "all")
    )

    for col in CORE_METRICS + ["n", "target_sum", "prediction_sum", "target_mean", "prediction_mean"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = add_spec_columns(df, spec, metric_path)
    df["metric_source_type"] = "reported_metrics_file"
    return df


def collect_metrics_for_baseline(
    spec: Mapping[str, Any],
    config: Config,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    directory = config.baselines_dir / str(spec["directory"])
    audit: list[dict[str, Any]] = []

    if not directory.exists():
        audit.append(
            {
                "baseline_family": spec["baseline_family"],
                "status": "missing_directory",
                "path": str(directory),
            }
        )
        return pd.DataFrame(), audit

    # Prefer recomputed prediction metrics for B0/B2/B3/B4 because those outputs
    # have a stable prediction schema. B1 may not have predictions.
    if config.prefer_recomputed_prediction_metrics:
        recomputed, pred_path = recompute_metrics_from_predictions(directory, spec)
        if recomputed is not None:
            audit.append(
                {
                    "baseline_family": spec["baseline_family"],
                    "status": "loaded_predictions_recomputed_metrics",
                    "path": str(pred_path),
                    "rows": int(len(recomputed)),
                }
            )
            return recomputed, audit
        elif pred_path is not None:
            audit.append(
                {
                    "baseline_family": spec["baseline_family"],
                    "status": "prediction_file_found_but_not_usable",
                    "path": str(pred_path),
                }
            )

    metric_frames: list[pd.DataFrame] = []
    for metric_path in find_metric_file_candidates(directory):
        standardized = standardize_metrics_file(metric_path, spec)
        if standardized is None:
            audit.append(
                {
                    "baseline_family": spec["baseline_family"],
                    "status": "metric_file_skipped_no_core_metrics",
                    "path": str(metric_path),
                }
            )
            continue
        audit.append(
            {
                "baseline_family": spec["baseline_family"],
                "status": "loaded_metrics_file",
                "path": str(metric_path),
                "rows": int(len(standardized)),
            }
        )
        metric_frames.append(standardized)

    if not metric_frames:
        audit.append(
            {
                "baseline_family": spec["baseline_family"],
                "status": "no_usable_metrics",
                "path": str(directory),
            }
        )
        return pd.DataFrame(), audit

    combined = pd.concat(metric_frames, ignore_index=True)
    return combined, audit


def collect_all_metrics(config: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    audit_rows: list[dict[str, Any]] = []

    for spec in BENCHMARK_SPECS:
        metrics, audit = collect_metrics_for_baseline(spec, config)
        audit_rows.extend(audit)
        if not metrics.empty:
            frames.append(metrics)

    if not frames:
        return pd.DataFrame(), pd.DataFrame(audit_rows)

    all_metrics = pd.concat(frames, ignore_index=True)

    # Preserve only the selected model per model family if duplicates from
    # reported files slipped in; recomputed metrics already reflect selected
    # prediction outputs.
    all_metrics = all_metrics.loc[:, ~all_metrics.columns.duplicated()].copy()

    # Ensure all core metric columns exist.
    for metric in CORE_METRICS:
        if metric not in all_metrics.columns:
            all_metrics[metric] = np.nan

    # Stable ordering.
    stage_order = {spec["baseline_family"]: i for i, spec in enumerate(BENCHMARK_SPECS)}
    all_metrics["_baseline_order"] = all_metrics["baseline_family"].map(stage_order).fillna(999).astype(int)
    all_metrics = all_metrics.sort_values(
        ["_baseline_order", "model_name", SPLIT_COL, "target_col"],
        na_position="last",
    ).drop(columns=["_baseline_order"]).reset_index(drop=True)

    return all_metrics, pd.DataFrame(audit_rows)


def build_metrics_long(comparison: pd.DataFrame) -> pd.DataFrame:
    id_cols = [
        "baseline_family",
        "baseline_stage",
        "method_class",
        "display_name",
        "model_name",
        "candidate_name",
        "graph_kind",
        SPLIT_COL,
        "target_col",
        "metric_source_type",
        "source_file",
    ]
    id_cols = [c for c in id_cols if c in comparison.columns]

    rows: list[dict[str, Any]] = []
    for _, row in comparison.iterrows():
        for metric in CORE_METRICS:
            if metric not in comparison.columns:
                continue
            value = row.get(metric)
            if pd.isna(value):
                continue
            out = {c: row.get(c) for c in id_cols}
            out["metric"] = metric
            out["value"] = float(value)
            out["direction"] = metric_direction(metric)
            rows.append(out)

    return pd.DataFrame(rows)


def choose_primary_rows(comparison: pd.DataFrame, config: Config) -> pd.DataFrame:
    """
    Choose one primary comparison row per model/candidate.

    Priority:
        primary split (test by default) -> validation -> val -> all/static.
    """
    if comparison.empty:
        return pd.DataFrame()

    split_priority = [config.primary_split, *config.fallback_splits]
    split_priority = [normalize_split_value(s, default=s) for s in split_priority]

    group_cols = ["baseline_family", "model_name"]
    if "candidate_name" in comparison.columns:
        group_cols.append("candidate_name")
    if "target_col" in comparison.columns:
        group_cols.append("target_col")

    rows = []
    for _, sub in comparison.groupby(group_cols, dropna=False):
        sub = sub.copy()
        sub["_split_norm"] = sub[SPLIT_COL].map(lambda x: normalize_split_value(x, default="all"))

        chosen = None
        for split_name in split_priority:
            cand = sub[sub["_split_norm"].eq(split_name)]
            if not cand.empty:
                chosen = cand.iloc[0]
                break

        if chosen is None:
            chosen = sub.iloc[0]

        rows.append(chosen.drop(labels=["_split_norm"], errors="ignore").to_dict())

    compact = pd.DataFrame(rows)
    if compact.empty:
        return compact

    # If a method produces multiple model rows, keep them all; this is compact,
    # not necessarily one per stage. Add rank by primary metric.
    if config.primary_metric in compact.columns:
        direction = metric_direction(config.primary_metric)
        ascending = direction != "higher_is_better"
        compact = compact.sort_values(
            [config.primary_metric, "baseline_family", "model_name"],
            ascending=[ascending, True, True],
            na_position="last",
        ).reset_index(drop=True)
        compact.insert(0, "primary_metric_rank", np.arange(1, len(compact) + 1))

    keep_front = [
        "primary_metric_rank",
        "baseline_stage",
        "baseline_family",
        "display_name",
        "model_name",
        "candidate_name",
        "method_class",
        "graph_kind",
        SPLIT_COL,
        "target_col",
        *COMPACT_METRICS,
        "uses_sovi",
        "uses_history",
        "uses_tabular_ml",
        "uses_neural",
        "uses_graph",
        "metric_source_type",
        "source_file",
    ]
    keep_front = [c for c in keep_front if c in compact.columns]
    rest = [c for c in compact.columns if c not in keep_front and c in CORE_METRICS]
    return compact[keep_front + rest]


def build_metric_winners(metrics_long: pd.DataFrame) -> pd.DataFrame:
    if metrics_long.empty:
        return pd.DataFrame()

    rows = []
    group_cols = [SPLIT_COL, "metric"]
    for (split_name, metric), sub in metrics_long.groupby(group_cols, dropna=False):
        direction = metric_direction(metric)
        if direction == "unknown":
            continue

        sub = sub.copy()
        sub = sub[pd.to_numeric(sub["value"], errors="coerce").notna()]
        if sub.empty:
            continue

        ascending = direction == "lower_is_better"
        sub = sub.sort_values("value", ascending=ascending).reset_index(drop=True)
        winner = sub.iloc[0].to_dict()

        # Add runner-up and margin when available.
        runner_up_value = np.nan
        margin = np.nan
        relative_margin = np.nan
        if len(sub) >= 2:
            runner_up_value = float(sub.iloc[1]["value"])
            winner_value = float(winner["value"])
            if direction == "lower_is_better":
                margin = runner_up_value - winner_value
                relative_margin = margin / abs(runner_up_value) if runner_up_value != 0 else np.nan
            else:
                margin = winner_value - runner_up_value
                relative_margin = margin / abs(runner_up_value) if runner_up_value != 0 else np.nan

        rows.append(
            {
                "split": split_name,
                "metric": metric,
                "direction": direction,
                "winner_baseline_family": winner.get("baseline_family"),
                "winner_display_name": winner.get("display_name"),
                "winner_model_name": winner.get("model_name"),
                "winner_candidate_name": winner.get("candidate_name"),
                "winner_graph_kind": winner.get("graph_kind"),
                "winner_value": float(winner["value"]),
                "runner_up_value": runner_up_value,
                "winner_margin": margin,
                "winner_relative_margin": relative_margin,
                "n_methods_compared": int(len(sub)),
            }
        )

    return pd.DataFrame(rows)


def best_row_for_family(compact: pd.DataFrame, family: str, metric: str = "mae") -> pd.Series | None:
    sub = compact[compact["baseline_family"].astype("string").eq(family)].copy()
    if sub.empty or metric not in sub.columns:
        return None

    sub[metric] = pd.to_numeric(sub[metric], errors="coerce")
    sub = sub[sub[metric].notna()]
    if sub.empty:
        return None

    ascending = metric_direction(metric) != "higher_is_better"
    return sub.sort_values(metric, ascending=ascending).iloc[0]


def best_row_by_stage_or_family(
    compact: pd.DataFrame,
    *,
    families: Sequence[str],
    metric: str = "mae",
) -> pd.Series | None:
    sub = compact[compact["baseline_family"].astype("string").isin(families)].copy()
    if sub.empty or metric not in sub.columns:
        return None

    sub[metric] = pd.to_numeric(sub[metric], errors="coerce")
    sub = sub[sub[metric].notna()]
    if sub.empty:
        return None

    ascending = metric_direction(metric) != "higher_is_better"
    return sub.sort_values(metric, ascending=ascending).iloc[0]


def compare_metric(
    a: pd.Series | None,
    b: pd.Series | None,
    metric: str,
) -> dict[str, Any]:
    if a is None or b is None:
        return {
            "metric": metric,
            "available": False,
            "a_value": np.nan,
            "b_value": np.nan,
            "delta": np.nan,
            "relative_delta": np.nan,
            "a_beats_b": False,
        }

    av = pd.to_numeric(pd.Series([a.get(metric)]), errors="coerce").iloc[0]
    bv = pd.to_numeric(pd.Series([b.get(metric)]), errors="coerce").iloc[0]

    if pd.isna(av) or pd.isna(bv):
        return {
            "metric": metric,
            "available": False,
            "a_value": av,
            "b_value": bv,
            "delta": np.nan,
            "relative_delta": np.nan,
            "a_beats_b": False,
        }

    direction = metric_direction(metric)
    if direction == "lower_is_better":
        delta = bv - av  # positive means a improves over b
        relative_delta = delta / abs(bv) if bv != 0 else np.nan
        beats = av < bv
    elif direction == "higher_is_better":
        delta = av - bv  # positive means a improves over b
        relative_delta = delta / abs(bv) if bv != 0 else np.nan
        beats = av > bv
    else:
        delta = np.nan
        relative_delta = np.nan
        beats = False

    return {
        "metric": metric,
        "available": True,
        "a_value": float(av),
        "b_value": float(bv),
        "delta": float(delta),
        "relative_delta": float(relative_delta) if pd.notna(relative_delta) else np.nan,
        "a_beats_b": bool(beats),
    }


def build_graph_value_checks(compact: pd.DataFrame) -> pd.DataFrame:
    """
    Build graph-value checklist comparing real adjacency graph to key baselines.
    """
    real = best_row_for_family(compact, "B4_real_cd_graph", "mae")

    comparators = [
        ("B3_tabular_feature_parity", "B3 tabular feature parity"),
        ("B4_no_edge_neural", "B4 no-edge neural"),
        ("B4_random_edge_graph", "B4 random/placebo graph"),
        ("B4_knn_graph", "B4 kNN graph"),
        ("B2_calibrated_sovi", "B2 calibrated SoVI"),
        ("B0_history_only", "B0 history-only"),
    ]

    metrics = ["mae", "rmse", "spearman", "ndcg_at_25", "top10_overlap_rate"]

    rows = []
    for family, label in comparators:
        other = best_row_for_family(compact, family, "mae")

        for metric in metrics:
            comp = compare_metric(real, other, metric)
            rows.append(
                {
                    "comparison": f"B4_real_cd_graph vs {family}",
                    "comparator_family": family,
                    "comparator_label": label,
                    "metric": metric,
                    "direction": metric_direction(metric),
                    "real_graph_value": comp["a_value"],
                    "comparator_value": comp["b_value"],
                    "improvement_absolute": comp["delta"],
                    "improvement_relative": comp["relative_delta"],
                    "real_graph_beats_comparator": comp["a_beats_b"],
                    "available": comp["available"],
                }
            )

    return pd.DataFrame(rows)


def verdict_from_checks(checks: pd.DataFrame) -> dict[str, Any]:
    """
    Produce a cautious graph-value verdict from the checklist.
    """
    if checks.empty:
        return {
            "verdict": "Insufficient evidence",
            "reason": "No graph-value checks were available.",
            "strength": "missing",
        }

    required = checks[
        checks["comparator_family"].isin(
            ["B3_tabular_feature_parity", "B4_no_edge_neural", "B4_random_edge_graph"]
        )
        & checks["metric"].isin(["mae", "rmse", "spearman", "ndcg_at_25"])
        & checks["available"].astype(bool)
    ].copy()

    if required.empty:
        return {
            "verdict": "Insufficient evidence",
            "reason": "Required comparisons against B3/no-edge/random controls were unavailable.",
            "strength": "missing",
        }

    # Count wins by comparator.
    comparator_wins = (
        required.groupby("comparator_family")["real_graph_beats_comparator"]
        .agg(["sum", "count"])
        .reset_index()
    )
    comparator_wins["win_rate"] = comparator_wins["sum"] / comparator_wins["count"]

    def win_rate(family: str) -> float:
        row = comparator_wins[comparator_wins["comparator_family"].eq(family)]
        if row.empty:
            return np.nan
        return float(row["win_rate"].iloc[0])

    wr_b3 = win_rate("B3_tabular_feature_parity")
    wr_no_edge = win_rate("B4_no_edge_neural")
    wr_random = win_rate("B4_random_edge_graph")

    mae_checks = checks[
        checks["metric"].eq("mae")
        & checks["comparator_family"].isin(
            ["B3_tabular_feature_parity", "B4_no_edge_neural", "B4_random_edge_graph"]
        )
        & checks["available"].astype(bool)
    ]

    mae_wins = int(mae_checks["real_graph_beats_comparator"].sum())
    mae_total = int(len(mae_checks))

    if (
        pd.notna(wr_b3)
        and pd.notna(wr_no_edge)
        and pd.notna(wr_random)
        and wr_b3 >= 0.75
        and wr_no_edge >= 0.75
        and wr_random >= 0.75
        and mae_wins == mae_total
    ):
        return {
            "verdict": "Strong evidence that real graph structure adds value",
            "reason": (
                "The real adjacency graph beats B3 feature-parity, no-edge neural, "
                "and random/placebo graph controls on most primary metrics, including MAE."
            ),
            "strength": "strong",
        }

    if pd.notna(wr_no_edge) and pd.notna(wr_random) and wr_no_edge >= 0.75 and wr_random >= 0.75:
        if pd.notna(wr_b3) and wr_b3 < 0.50:
            return {
                "verdict": "Graph topology helps the neural model, but not beyond tabular feature parity",
                "reason": (
                    "The real graph improves over no-edge and random/placebo neural controls, "
                    "but does not consistently beat the B3 non-graph feature-parity baseline."
                ),
                "strength": "mixed",
            }
        return {
            "verdict": "Moderate evidence that graph topology adds value within the neural family",
            "reason": (
                "The real graph improves over no-edge and random/placebo controls, "
                "but the comparison against B3 feature parity is mixed or incomplete."
            ),
            "strength": "moderate",
        }

    if pd.notna(wr_random) and wr_random < 0.50:
        return {
            "verdict": "No reliable evidence that real graph structure adds value",
            "reason": (
                "The real graph does not consistently beat the random/placebo graph control. "
                "This weakens the claim that the specific topology is informative."
            ),
            "strength": "weak_or_negative",
        }

    if pd.notna(wr_no_edge) and wr_no_edge < 0.50:
        return {
            "verdict": "No reliable evidence that message passing adds value",
            "reason": (
                "The real graph does not consistently beat the no-edge neural control. "
                "This suggests the gain, if any, may come from features rather than topology."
            ),
            "strength": "weak_or_negative",
        }

    return {
        "verdict": "Mixed or inconclusive evidence",
        "reason": (
            "The real graph wins some comparisons but not enough to support a strong topology-value claim."
        ),
        "strength": "mixed",
    }


def format_float(value: Any, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return "—"
    try:
        v = float(value)
    except Exception:
        return str(value)
    if math.isinf(v):
        return "∞" if v > 0 else "-∞"
    return f"{v:.{digits}f}"


def markdown_table(df: pd.DataFrame, columns: Sequence[str], max_rows: int = 20) -> str:
    if df.empty:
        return "_No rows available._"

    cols = [c for c in columns if c in df.columns]
    if not cols:
        return "_No requested columns available._"

    shown = df[cols].head(max_rows).copy()

    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")

    for _, row in shown.iterrows():
        vals = []
        for col in cols:
            value = row[col]
            if isinstance(value, (float, int, np.floating, np.integer)):
                vals.append(format_float(value))
            elif pd.isna(value):
                vals.append("—")
            else:
                vals.append(str(value).replace("\n", " "))
        lines.append("| " + " | ".join(vals) + " |")

    if len(df) > max_rows:
        lines.append(f"\n_Showing first {max_rows} of {len(df)} rows._")

    return "\n".join(lines)


def build_delta_table(checks: pd.DataFrame) -> pd.DataFrame:
    if checks.empty:
        return pd.DataFrame()

    keep_metrics = ["mae", "rmse", "spearman", "ndcg_at_25", "top10_overlap_rate"]
    out = checks[checks["metric"].isin(keep_metrics)].copy()
    out["improvement_relative_pct"] = out["improvement_relative"] * 100.0
    out = out.sort_values(["comparator_family", "metric"]).reset_index(drop=True)
    return out


def write_report(
    *,
    comparison: pd.DataFrame,
    compact: pd.DataFrame,
    metrics_long: pd.DataFrame,
    winners: pd.DataFrame,
    graph_checks: pd.DataFrame,
    audit: pd.DataFrame,
    verdict: dict[str, Any],
    config: Config,
    report_path: Path,
) -> None:
    """Write Markdown benchmark report."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    compact_report = compact.copy()
    if "primary_metric_rank" in compact_report.columns:
        compact_report = compact_report.sort_values("primary_metric_rank")

    winner_report = winners[
        winners["split"].astype("string").isin([config.primary_split, "validation", "val", "all", "cumulative_static"])
    ].copy() if not winners.empty else pd.DataFrame()

    delta_table = build_delta_table(graph_checks)

    real_vs_key = graph_checks[
        graph_checks["comparator_family"].isin(
            ["B3_tabular_feature_parity", "B4_no_edge_neural", "B4_random_edge_graph", "B4_knn_graph"]
        )
        & graph_checks["metric"].isin(["mae", "rmse", "spearman", "ndcg_at_25"])
    ].copy()

    report_lines = [
        "# Québec CD civil-security / SoVI benchmark report",
        "",
        f"Generated: **{now}**",
        "",
        "## Executive answer",
        "",
        f"**Verdict:** {verdict.get('verdict', 'Unavailable')}.",
        "",
        verdict.get("reason", "No reason available."),
        "",
        "This report compares direct SoVI validation, history-only baselines, calibrated SoVI, "
        "tabular feature-parity ML, no-edge neural controls, random/placebo graph controls, "
        "kNN graph controls, and the real CD adjacency graph.",
        "",
        "## What would count as graph value?",
        "",
        "The real graph claim is strongest only if **B4_real_cd_graph** improves over all of the following:",
        "",
        "1. **B3_tabular_feature_parity**: same non-graph node features with strong tabular ML.",
        "2. **B4_no_edge_neural**: same neural architecture family but no message passing.",
        "3. **B4_random_edge_graph**: same graph architecture but placebo topology.",
        "4. **B4_knn_graph**: generic spatial-proximity topology.",
        "",
        "If the real graph only beats B1/B2, that is not enough: it may simply be learning from history/features. "
        "If it beats no-edge but not random graph, topology-specific value is weak. "
        "If it beats random/no-edge but not B3, graph message passing may help neural modeling but not yet surpass feature-parity tabular ML.",
        "",
        "## Compact benchmark comparison",
        "",
        markdown_table(
            compact_report,
            [
                "primary_metric_rank",
                "baseline_stage",
                "display_name",
                "model_name",
                "graph_kind",
                "split",
                "mae",
                "rmse",
                "mean_poisson_deviance",
                "spearman",
                "ndcg_at_25",
                "top10_overlap_rate",
            ],
            max_rows=30,
        ),
        "",
        "## Graph-value checklist",
        "",
        markdown_table(
            real_vs_key,
            [
                "comparator_label",
                "metric",
                "direction",
                "real_graph_value",
                "comparator_value",
                "improvement_absolute",
                "improvement_relative",
                "real_graph_beats_comparator",
                "available",
            ],
            max_rows=40,
        ),
        "",
        "## Metric winners",
        "",
        markdown_table(
            winner_report,
            [
                "split",
                "metric",
                "direction",
                "winner_display_name",
                "winner_model_name",
                "winner_graph_kind",
                "winner_value",
                "runner_up_value",
                "winner_margin",
                "n_methods_compared",
            ],
            max_rows=40,
        ),
        "",
        "## Interpretation guide",
        "",
        "Read the benchmark in layers:",
        "",
        "- **B1 → B2** tests whether raw/static SoVI becomes operationally useful after calibration.",
        "- **B0** tests how much simple temporal history already explains.",
        "- **B3** is the key non-graph feature-parity baseline. It is the hardest non-graph benchmark.",
        "- **B4_no_edge_neural** tests whether neural capacity alone helps.",
        "- **B4_random_edge_graph** tests whether arbitrary graph smoothing helps; the real graph should beat this.",
        "- **B4_knn_graph** tests whether generic spatial proximity is enough.",
        "- **B4_real_cd_graph** supports a topology-value claim only if it improves over the controls above.",
        "",
        "## Missing or skipped inputs",
        "",
        markdown_table(
            audit,
            ["baseline_family", "status", "path", "rows"],
            max_rows=60,
        ),
        "",
        "## Output files",
        "",
        f"- `benchmark_comparison.csv`: wide comparison table, one row per method/model/split where available.",
        f"- `benchmark_comparison_compact.csv`: primary comparison rows used for report ranking.",
        f"- `metrics_long.csv`: long-form metric table.",
        f"- `metric_winners.csv`: winner by metric and split.",
        f"- `qc_cd_civil_security_sovi_benchmark_report.md`: this report.",
        "",
        "## Reproducibility notes",
        "",
        f"- Baselines directory: `{config.baselines_dir}`",
        f"- Comparisons directory: `{config.comparisons_dir}`",
        f"- Reports directory: `{config.reports_dir}`",
        f"- Primary split for compact ranking: `{config.primary_split}`",
        f"- Primary metric for compact ranking: `{config.primary_metric}`",
        "",
    ]

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines), encoding="utf-8")


def write_outputs(
    *,
    comparison: pd.DataFrame,
    compact: pd.DataFrame,
    metrics_long: pd.DataFrame,
    winners: pd.DataFrame,
    graph_checks: pd.DataFrame,
    audit: pd.DataFrame,
    verdict: dict[str, Any],
    config: Config,
) -> dict[str, str]:
    ensure_dir(config.comparisons_dir)
    ensure_dir(config.reports_dir)

    outputs: dict[str, str] = {}

    comparison_path = config.comparisons_dir / "benchmark_comparison.csv"
    comparison.to_csv(comparison_path, index=False)
    outputs["benchmark_comparison_csv"] = str(comparison_path)

    compact_path = config.comparisons_dir / "benchmark_comparison_compact.csv"
    compact.to_csv(compact_path, index=False)
    outputs["benchmark_comparison_compact_csv"] = str(compact_path)

    long_path = config.comparisons_dir / "metrics_long.csv"
    metrics_long.to_csv(long_path, index=False)
    outputs["metrics_long_csv"] = str(long_path)

    winners_path = config.comparisons_dir / "metric_winners.csv"
    winners.to_csv(winners_path, index=False)
    outputs["metric_winners_csv"] = str(winners_path)

    graph_checks_path = config.comparisons_dir / "graph_value_checks.csv"
    graph_checks.to_csv(graph_checks_path, index=False)
    outputs["graph_value_checks_csv"] = str(graph_checks_path)

    audit_path = config.comparisons_dir / "benchmark_collection_audit.csv"
    audit.to_csv(audit_path, index=False)
    outputs["collection_audit_csv"] = str(audit_path)

    report_path = config.reports_dir / "qc_cd_civil_security_sovi_benchmark_report.md"
    write_report(
        comparison=comparison,
        compact=compact,
        metrics_long=metrics_long,
        winners=winners,
        graph_checks=graph_checks,
        audit=audit,
        verdict=verdict,
        config=config,
        report_path=report_path,
    )
    outputs["markdown_report"] = str(report_path)

    metadata = {
        "script": "urban_graph_benchmark/scripts/17_compare_qc_cd_civil_security_sovi_benchmark.py",
        "purpose": "Merge B1/B0/B2/B3/B4 benchmark metrics and write final report.",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "baselines_dir": str(config.baselines_dir),
            "comparisons_dir": str(config.comparisons_dir),
            "reports_dir": str(config.reports_dir),
            "primary_split": config.primary_split,
            "fallback_splits": list(config.fallback_splits),
            "primary_metric": config.primary_metric,
            "prefer_recomputed_prediction_metrics": bool(config.prefer_recomputed_prediction_metrics),
        },
        "rows": {
            "benchmark_comparison": int(len(comparison)),
            "benchmark_comparison_compact": int(len(compact)),
            "metrics_long": int(len(metrics_long)),
            "metric_winners": int(len(winners)),
            "graph_value_checks": int(len(graph_checks)),
            "collection_audit": int(len(audit)),
        },
        "verdict": verdict,
        "outputs": outputs,
    }
    metadata_path = write_metadata_json(metadata, config.comparisons_dir / "benchmark_comparison_metadata.json")
    outputs["metadata_json"] = str(metadata_path)

    return outputs


def run_comparison(config: Config) -> dict[str, Any]:
    ensure_path_fields(config)
    ensure_dir(config.comparisons_dir)
    ensure_dir(config.reports_dir)

    comparison, audit = collect_all_metrics(config)

    if comparison.empty:
        # Still write an audit/report so failures are visible.
        metrics_long = pd.DataFrame()
        compact = pd.DataFrame()
        winners = pd.DataFrame()
        graph_checks = pd.DataFrame()
        verdict = {
            "verdict": "Insufficient evidence",
            "reason": "No usable metrics were collected from baseline directories.",
            "strength": "missing",
        }
    else:
        metrics_long = build_metrics_long(comparison)
        compact = choose_primary_rows(comparison, config)
        winners = build_metric_winners(metrics_long)
        graph_checks = build_graph_value_checks(compact)
        verdict = verdict_from_checks(graph_checks)

    outputs = write_outputs(
        comparison=comparison,
        compact=compact,
        metrics_long=metrics_long,
        winners=winners,
        graph_checks=graph_checks,
        audit=audit,
        verdict=verdict,
        config=config,
    )

    return {
        "outputs": outputs,
        "verdict": verdict,
        "rows": {
            "benchmark_comparison": int(len(comparison)),
            "benchmark_comparison_compact": int(len(compact)),
            "metrics_long": int(len(metrics_long)),
            "metric_winners": int(len(winners)),
            "graph_value_checks": int(len(graph_checks)),
            "collection_audit": int(len(audit)),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge B1/B0/B2/B3/B4 metrics into final comparison tables and "
            "write a Markdown benchmark report."
        )
    )
    parser.add_argument(
        "--baselines-dir",
        type=Path,
        default=BASELINES_DIR,
        help="Baseline output directory.",
    )
    parser.add_argument(
        "--comparisons-dir",
        type=Path,
        default=COMPARISONS_DIR,
        help="Directory for comparison CSV outputs.",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=REPORTS_DIR,
        help="Directory for Markdown report output.",
    )
    parser.add_argument(
        "--primary-split",
        default="test",
        help="Primary split used for compact ranking. Default: test.",
    )
    parser.add_argument(
        "--primary-metric",
        default="mae",
        choices=CORE_METRICS,
        help="Primary metric used for compact ranking. Default: mae.",
    )
    parser.add_argument(
        "--use-reported-metrics-first",
        action="store_true",
        help=(
            "Use reported metrics.csv files before recomputing from predictions. "
            "Default is to recompute from predictions when possible."
        ),
    )
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> Config:
    return Config(
        baselines_dir=args.baselines_dir,
        comparisons_dir=args.comparisons_dir,
        reports_dir=args.reports_dir,
        primary_split=args.primary_split,
        primary_metric=args.primary_metric,
        prefer_recomputed_prediction_metrics=not args.use_reported_metrics_first,
    )


def main() -> None:
    args = parse_args()
    config = config_from_args(args)

    result = run_comparison(config)

    print("Québec CD civil-security / SoVI benchmark comparison completed.")
    print(f"Baselines directory: {config.baselines_dir}")
    print(f"Comparisons directory: {config.comparisons_dir}")
    print(f"Reports directory: {config.reports_dir}")
    print("Rows:")
    for key, value in result["rows"].items():
        print(f"  {key}: {value}")

    print("Verdict:")
    print(f"  {result['verdict'].get('verdict')}")
    print(f"  {result['verdict'].get('reason')}")

    print("Outputs:")
    for key, value in result["outputs"].items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
