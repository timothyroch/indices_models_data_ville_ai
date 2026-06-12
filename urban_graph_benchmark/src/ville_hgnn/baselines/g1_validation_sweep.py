#!/usr/bin/env python3
"""
G1.5 validation-selected architecture sweep for the G1 spatiotemporal graph baseline.

This wrapper does not implement a new model. It orchestrates repeated calls to 
``g1_spatiotemporal_gnn.py`` as the single-configuration training/evaluation 
engine, then aggregates the resulting A3-compatible metrics into a 
validation-only model-selection audit.

Purpose
-------
G1.5 asks whether the initial G1 result was limited by a narrow architecture /
monitor choice rather than by the graph hypothesis itself.

Protocol
--------
- Keep A3 frozen.
- Use the existing G1 graph artifact and G1 training engine.
- Select G1 configurations by validation metric only, defaulting to
  validation NDCG@100.
- Compare selected family representatives on held-out spatial-block test metrics.
- Preserve no-edge and random-placebo controls.

This file is a sweep wrapper, not a replacement for the G1 runner.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import shutil
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from ville_hgnn.baselines.g1_spatiotemporal_gnn import (
    DEFAULT_GRAPH_DIR,
    G1BaselineError,
    MONITOR_HIGHER_IS_BETTER,
    TrainConfig,
    run_g1_spatiotemporal_gnn,
)


DEFAULT_BASELINES_DIR = "urban_graph_benchmark/outputs/mtl_311_water_v0/baselines"
DEFAULT_A3_SPATIAL_DIR = (
    "urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/"
    "A3_feature_parity_tabular_spatial_block"
)
DEFAULT_SWEEP_OUTPUT_DIR = (
    "urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/"
    "G1_5_validation_sweep_spatial_ndcg"
)

MODEL_STAGE = "G1_5_validation_sweep"
DEFAULT_MONITOR = "validation_ndcg_at_100"

TEST_METRICS = {
    "count__mae": "test_mae",
    "count__rmse": "test_rmse",
    "count__mean_poisson_deviance": "test_mean_poisson_deviance",
    "ranking__spearman_corr": "test_spearman",
    "ranking__kendall_corr": "test_kendall",
    "ranking__ndcg_at_100": "test_ndcg_at_100",
    "ranking__top_10pct_overlap_rate": "test_top_10pct_overlap_rate",
}

VALIDATION_METRICS = {
    "count__mae": "validation_mae",
    "count__rmse": "validation_rmse",
    "count__mean_poisson_deviance": "validation_mean_poisson_deviance",
    "ranking__spearman_corr": "validation_spearman",
    "ranking__kendall_corr": "validation_kendall",
    "ranking__ndcg_at_100": "validation_ndcg_at_100",
    "ranking__top_10pct_overlap_rate": "validation_top_10pct_overlap_rate",
}

FAMILY_CONTROL_EDGE_REGIMES = {"no_edges", "random_spatial_placebo"}


@dataclass(frozen=True)
class ArchitectureConfig:
    """Architecture/hyperparameter config for one G1 engine call."""

    hidden_dim: int
    num_layers: int
    dropout: float
    normalization: str
    residual: bool
    relation_combine: str
    backend: str
    learning_rate: float
    weight_decay: float

    def run_id(self) -> str:
        """Stable identifier for the architecture run directory."""

        residual_tag = "res" if self.residual else "nores"
        return (
            f"h{self.hidden_dim}_L{self.num_layers}_"
            f"do{float_tag(self.dropout)}_{self.normalization}_{residual_tag}_"
            f"rel{self.relation_combine}_{self.backend}_"
            f"lr{float_tag(self.learning_rate)}_wd{float_tag(self.weight_decay)}"
        )


@dataclass(frozen=True)
class SweepSpace:
    """Resolved sweep dimensions."""

    feature_regimes: tuple[str, ...]
    split_scheme: str
    edge_regimes: tuple[str, ...]
    edge_mask_regimes: tuple[str, ...]
    hidden_dims: tuple[int, ...]
    num_layers: tuple[int, ...]
    dropouts: tuple[float, ...]
    normalizations: tuple[str, ...]
    residual_options: tuple[bool, ...]
    relation_combines: tuple[str, ...]
    backends: tuple[str, ...]
    seeds: tuple[int, ...]

    def architecture_configs(
        self,
        *,
        learning_rate: float,
        weight_decay: float,
    ) -> list[ArchitectureConfig]:
        """Return architecture configs in deterministic order."""

        configs: list[ArchitectureConfig] = []
        for hidden_dim, num_layers, dropout, norm, residual, rel_combine, backend in itertools.product(
            self.hidden_dims,
            self.num_layers,
            self.dropouts,
            self.normalizations,
            self.residual_options,
            self.relation_combines,
            self.backends,
        ):
            configs.append(
                ArchitectureConfig(
                    hidden_dim=int(hidden_dim),
                    num_layers=int(num_layers),
                    dropout=float(dropout),
                    normalization=str(norm),
                    residual=bool(residual),
                    relation_combine=str(rel_combine),
                    backend=str(backend),
                    learning_rate=float(learning_rate),
                    weight_decay=float(weight_decay),
                )
            )
        return configs


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def float_tag(value: float) -> str:
    """Turn a float into a filesystem/model-name friendly tag."""

    text = f"{float(value):g}"
    return text.replace("-", "m").replace(".", "p")


def parse_csv(value: str | None, default: Sequence[str]) -> tuple[str, ...]:
    if value is None or str(value).strip() == "":
        return tuple(default)
    return tuple(part.strip() for part in str(value).split(",") if part.strip())


def parse_int_csv(value: str | None, default: Sequence[int]) -> tuple[int, ...]:
    return tuple(int(v) for v in parse_csv(value, [str(x) for x in default]))


def parse_float_csv(value: str | None, default: Sequence[float]) -> tuple[float, ...]:
    return tuple(float(v) for v in parse_csv(value, [str(x) for x in default]))


def parse_bool_csv(value: str | None, default: Sequence[bool]) -> tuple[bool, ...]:
    if value is None or str(value).strip() == "":
        return tuple(default)
    out: list[bool] = []
    for raw in str(value).split(","):
        token = raw.strip().lower()
        if token in {"true", "t", "1", "yes", "y", "res", "residual"}:
            out.append(True)
        elif token in {"false", "f", "0", "no", "n", "nores", "no_residual"}:
            out.append(False)
        else:
            raise ValueError(f"Invalid boolean token {raw!r}. Use true,false.")
    return tuple(out)


def write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(data), indent=2, ensure_ascii=False), encoding="utf-8")


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, Mapping):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return obj


def dataframe_to_markdown(df: pd.DataFrame, max_rows: int = 80) -> str:
    if df.empty:
        return "_No rows._"
    display = df.head(max_rows).copy()
    try:
        return display.to_markdown(index=False)
    except Exception:
        return "```text\n" + display.to_string(index=False) + "\n```"


# ---------------------------------------------------------------------------
# Sweep presets
# ---------------------------------------------------------------------------


def preset_defaults(name: str) -> dict[str, Any]:
    """Return default dimensions for a named sweep preset."""

    if name == "smoke":
        return {
            "feature_regimes": ("all_forecasting",),
            "split_scheme": "spatial_block",
            "edge_regimes": ("no_edges", "temporal_only", "spatial_temporal", "random_spatial_placebo"),
            "edge_mask_regimes": ("all_edges", "no_test_incident_edges"),
            "hidden_dims": (128,),
            "num_layers": (1, 2),
            "dropouts": (0.15,),
            "normalizations": ("layernorm",),
            "residual_options": (True,),
            "relation_combines": ("mean",),
            "backends": ("manual",),
            "seeds": (20240610,),
        }

    if name == "compact":
        return {
            "feature_regimes": ("all_forecasting",),
            "split_scheme": "spatial_block",
            "edge_regimes": ("no_edges", "temporal_only", "spatial_temporal", "random_spatial_placebo"),
            "edge_mask_regimes": ("all_edges", "no_test_incident_edges"),
            "hidden_dims": (64, 128),
            "num_layers": (1, 2, 3),
            "dropouts": (0.0, 0.05, 0.15),
            "normalizations": ("layernorm",),
            "residual_options": (True,),
            "relation_combines": ("mean", "sum"),
            "backends": ("manual",),
            "seeds": (20240610,),
        }

    if name == "expanded":
        return {
            "feature_regimes": ("all_forecasting",),
            "split_scheme": "spatial_block",
            "edge_regimes": ("no_edges", "temporal_only", "spatial_temporal", "random_spatial_placebo"),
            "edge_mask_regimes": ("all_edges", "no_test_incident_edges"),
            "hidden_dims": (64, 128, 256),
            "num_layers": (1, 2, 3),
            "dropouts": (0.0, 0.05, 0.15, 0.30),
            "normalizations": ("layernorm", "none"),
            "residual_options": (True, False),
            "relation_combines": ("mean", "sum"),
            "backends": ("manual",),
            "seeds": (20240610, 20240611, 20240612),
        }

    if name == "custom":
        return preset_defaults("compact")

    raise ValueError(f"Unknown sweep preset {name!r}.")


def resolve_sweep_space(args: argparse.Namespace) -> SweepSpace:
    defaults = preset_defaults(args.sweep_preset)
    return SweepSpace(
        feature_regimes=parse_csv(args.feature_regimes, defaults["feature_regimes"]),
        split_scheme=str(args.split_scheme or defaults["split_scheme"]),
        edge_regimes=parse_csv(args.edge_regimes, defaults["edge_regimes"]),
        edge_mask_regimes=parse_csv(args.edge_mask_regimes, defaults["edge_mask_regimes"]),
        hidden_dims=parse_int_csv(args.hidden_dims, defaults["hidden_dims"]),
        num_layers=parse_int_csv(args.num_layers_list, defaults["num_layers"]),
        dropouts=parse_float_csv(args.dropouts, defaults["dropouts"]),
        normalizations=parse_csv(args.normalizations, defaults["normalizations"]),
        residual_options=parse_bool_csv(args.residual_options, defaults["residual_options"]),
        relation_combines=parse_csv(args.relation_combines, defaults["relation_combines"]),
        backends=parse_csv(args.backends, defaults["backends"]),
        seeds=parse_int_csv(args.seeds, defaults["seeds"]),
    )


# ---------------------------------------------------------------------------
# Metrics and selection helpers
# ---------------------------------------------------------------------------


def metric_lookup(
    metrics: pd.DataFrame,
    *,
    model_name: str,
    split_contains: str,
    metric_name: str,
) -> float:
    """Fetch metric value for a model/split/metric, accepting split labels containing a token."""

    if metrics.empty:
        return float("nan")
    split = metrics["split_name"].astype(str).str.lower()
    sub = metrics[
        (metrics["model_name"].astype(str) == str(model_name))
        & split.str.contains(split_contains.lower(), regex=False)
        & (metrics["metric_name"].astype(str) == metric_name)
    ]
    if sub.empty:
        return float("nan")
    values = pd.to_numeric(sub["metric_value"], errors="coerce").dropna()
    if values.empty:
        return float("nan")
    return float(values.iloc[0])


def completed_candidates(selection: pd.DataFrame) -> pd.DataFrame:
    if selection.empty:
        return selection.copy()
    out = selection[selection["status"].astype(str) == "completed"].copy()
    return out


def select_best_rows(
    audit: pd.DataFrame,
    *,
    group_cols: Sequence[str],
    metric_col: str,
    higher_is_better: bool,
) -> pd.DataFrame:
    """Select one best row per group by a metric column."""

    valid = completed_candidates(audit)
    if valid.empty:
        return pd.DataFrame()
    valid["_selection_value"] = pd.to_numeric(valid[metric_col], errors="coerce")
    valid = valid[valid["_selection_value"].notna()].copy()
    if valid.empty:
        return pd.DataFrame()

    rows = []
    for _, group in valid.groupby(list(group_cols), dropna=False):
        best = group.sort_values("_selection_value", ascending=not higher_is_better).iloc[0].copy()
        rows.append(best)
    out = pd.DataFrame(rows).drop(columns=["_selection_value"], errors="ignore")
    return out.reset_index(drop=True)


def load_a3_selected_row(a3_dir: Path) -> dict[str, Any]:
    """Load frozen A3 selected spatial-block model and its test metrics."""

    metrics_path = a3_dir / "metrics.csv"
    selection_path = a3_dir / "model_selection_audit.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"A3 metrics not found: {metrics_path}")

    metrics = pd.read_csv(metrics_path)
    selection = pd.read_csv(selection_path) if selection_path.exists() else pd.DataFrame()

    model_name: str | None = None
    if not selection.empty and "model_name" in selection.columns:
        flag_candidates = [
            "selected_overall_strict_forecasting",
            "selected_for_test_summary",
            "selected_overall_spatial_block",
            "selected_overall",
        ]
        flagged = pd.DataFrame()
        for flag in flag_candidates:
            if flag in selection.columns:
                sub = selection[selection[flag].astype(bool)].copy()
                if not sub.empty:
                    flagged = sub
                    break
        if not flagged.empty:
            hgb = flagged[
                flagged["model_name"].astype(str).str.contains(
                    "hist_gradient_boosting|hgb", case=False, regex=True
                )
            ]
            model_name = str((hgb if not hgb.empty else flagged).iloc[0]["model_name"])

    if model_name is None:
        candidates = metrics[
            metrics["model_name"].astype(str).str.contains(
                "hist_gradient_boosting|hgb", case=False, regex=True
            )
        ]["model_name"].astype(str).drop_duplicates().tolist()
        if candidates:
            model_name = candidates[0]
        else:
            model_name = str(metrics["model_name"].astype(str).iloc[0])

    row: dict[str, Any] = {
        "comparison_role": "A3 frozen selected spatial-block baseline",
        "source": "A3",
        "family": "A3_frozen",
        "model_name": model_name,
        "feature_regime": np.nan,
        "edge_regime": "A3_tabular",
        "edge_mask_regime": np.nan,
        "hidden_dim": np.nan,
        "num_layers": np.nan,
        "dropout": np.nan,
        "normalization": np.nan,
        "residual": np.nan,
        "relation_combine": np.nan,
        "backend": np.nan,
        "seed": np.nan,
        "selection_metric": "frozen_A3_selection",
        "selection_metric_value": np.nan,
    }
    for metric_name, col in TEST_METRICS.items():
        row[col] = metric_lookup(
            metrics,
            model_name=model_name,
            split_contains="test",
            metric_name=metric_name,
        )
    return row


def add_family_labels(audit: pd.DataFrame) -> pd.DataFrame:
    """Add G1.5 family labels used for validation selection."""

    out = audit.copy()
    if out.empty:
        return out
    out["family"] = out["edge_regime"].astype(str)
    out["is_graph_family"] = ~out["edge_regime"].astype(str).isin(FAMILY_CONTROL_EDGE_REGIMES)
    out["is_control_family"] = out["edge_regime"].astype(str).isin(FAMILY_CONTROL_EDGE_REGIMES)
    return out


def build_family_selection(
    audit: pd.DataFrame,
    *,
    selection_metric: str,
) -> pd.DataFrame:
    """Select best row per edge-regime family by validation-only metric."""

    if selection_metric not in MONITOR_HIGHER_IS_BETTER:
        raise ValueError(f"Unknown selection metric {selection_metric!r}.")
    higher_is_better = bool(MONITOR_HIGHER_IS_BETTER[selection_metric])
    if selection_metric not in audit.columns:
        raise ValueError(
            f"Selection metric column {selection_metric!r} is absent from sweep audit. "
            f"Columns: {list(audit.columns)}"
        )

    selected = select_best_rows(
        audit,
        group_cols=["family"],
        metric_col=selection_metric,
        higher_is_better=higher_is_better,
    )
    if selected.empty:
        return selected

    selected["selected_within_family"] = True
    selected["comparison_role"] = "G1 selected " + selected["family"].astype(str)
    selected["source"] = "G1.5_validation_sweep"
    return selected


def build_final_comparison(
    *,
    a3_row: dict[str, Any] | None,
    family_selection: pd.DataFrame,
    selection_metric: str,
) -> pd.DataFrame:
    """Build final comparison table with A3 and selected G1 family representatives."""

    rows: list[dict[str, Any]] = []
    if a3_row is not None:
        rows.append(a3_row)

    if not family_selection.empty:
        for _, row in family_selection.iterrows():
            item = row.to_dict()
            item["source"] = "G1.5_validation_sweep"
            item["selection_metric"] = selection_metric
            item["selection_metric_value"] = item.get(selection_metric, np.nan)
            item["comparison_role"] = f"G1 selected {item.get('family')}"
            rows.append(item)

    comparison = pd.DataFrame(rows)
    wanted_cols = [
        "comparison_role",
        "source",
        "family",
        "model_name",
        "feature_regime",
        "edge_regime",
        "edge_mask_regime",
        "hidden_dim",
        "num_layers",
        "dropout",
        "normalization",
        "residual",
        "relation_combine",
        "backend",
        "seed",
        "selection_metric",
        "selection_metric_value",
        "test_mae",
        "test_rmse",
        "test_mean_poisson_deviance",
        "test_spearman",
        "test_kendall",
        "test_ndcg_at_100",
        "test_top_10pct_overlap_rate",
    ]
    for col in wanted_cols:
        if col not in comparison.columns:
            comparison[col] = np.nan
    return comparison[wanted_cols].reset_index(drop=True)


def build_factor_summary(audit: pd.DataFrame, *, selection_metric: str) -> pd.DataFrame:
    """Summarize performance by simple factors."""

    if audit.empty:
        return pd.DataFrame()
    factors = [
        "feature_regime",
        "edge_regime",
        "edge_mask_regime",
        "hidden_dim",
        "num_layers",
        "dropout",
        "normalization",
        "residual",
        "relation_combine",
        "backend",
        "seed",
    ]
    numeric_cols = [
        selection_metric,
        "validation_ndcg_at_100",
        "validation_spearman",
        "validation_top_10pct_overlap_rate",
        "validation_mae",
        "test_ndcg_at_100",
        "test_spearman",
        "test_top_10pct_overlap_rate",
        "test_mae",
    ]
    rows: list[dict[str, Any]] = []
    completed = completed_candidates(audit)
    for col in numeric_cols:
        if col in completed.columns:
            completed[col] = pd.to_numeric(completed[col], errors="coerce")
    for factor in factors:
        if factor not in completed.columns:
            continue
        for value, group in completed.groupby(factor, dropna=False):
            row: dict[str, Any] = {
                "factor": factor,
                "value": value,
                "n_trials": int(len(group)),
            }
            for metric in numeric_cols:
                if metric in group.columns:
                    vals = pd.to_numeric(group[metric], errors="coerce").dropna()
                    row[f"mean_{metric}"] = float(vals.mean()) if not vals.empty else np.nan
                    row[f"median_{metric}"] = float(vals.median()) if not vals.empty else np.nan
                    row[f"best_{metric}"] = float(vals.max()) if not vals.empty else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def metric_winners(final_comparison: pd.DataFrame) -> pd.DataFrame:
    """Return winners for core test metrics."""

    if final_comparison.empty:
        return pd.DataFrame()
    specs = [
        ("test_mae", False),
        ("test_spearman", True),
        ("test_ndcg_at_100", True),
        ("test_top_10pct_overlap_rate", True),
    ]
    rows = []
    for metric, higher in specs:
        if metric not in final_comparison.columns:
            continue
        tmp = final_comparison[["comparison_role", "family", "model_name", metric]].copy()
        tmp[metric] = pd.to_numeric(tmp[metric], errors="coerce")
        tmp = tmp[tmp[metric].notna()]
        if tmp.empty:
            continue
        winner = tmp.sort_values(metric, ascending=not higher).iloc[0]
        rows.append(
            {
                "metric": metric,
                "higher_is_better": bool(higher),
                "winner_role": winner["comparison_role"],
                "winner_family": winner["family"],
                "winner_model_name": winner["model_name"],
                "winner_value": float(winner[metric]),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Sweep execution
# ---------------------------------------------------------------------------


def run_architecture(
    *,
    arch: ArchitectureConfig,
    graph_dir: Path,
    output_dir: Path,
    space: SweepSpace,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Run one architecture through the existing G1 engine."""

    train_config = TrainConfig(
        hidden_dim=arch.hidden_dim,
        num_layers=arch.num_layers,
        dropout=arch.dropout,
        activation=args.activation,
        normalization=arch.normalization,
        residual=arch.residual,
        backend=arch.backend,
        relation_combine=arch.relation_combine,
        max_epochs=args.max_epochs,
        patience=args.patience,
        learning_rate=arch.learning_rate,
        weight_decay=arch.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        min_delta=args.min_delta,
        seed=space.seeds[0] if space.seeds else 20240610,
        device=args.device,
        monitor_metric=args.monitor_metric,
        save_checkpoints=not args.no_checkpoints,
        save_embeddings=args.save_embeddings,
    )

    result = run_g1_spatiotemporal_gnn(
        graph_dir=graph_dir,
        output_dir=output_dir,
        preset="custom",
        feature_regimes=space.feature_regimes,
        split_schemes=(space.split_scheme,),
        edge_regimes=space.edge_regimes,
        edge_mask_regimes=space.edge_mask_regimes,
        seeds=space.seeds,
        train_config=train_config,
        overwrite=True,
    )
    return dict(result)


def load_subrun_tables(subdir: Path, arch: ArchitectureConfig, run_id: str) -> dict[str, pd.DataFrame]:
    """Load output tables from one architecture sub-run and add architecture columns."""

    tables: dict[str, pd.DataFrame] = {}
    file_map = {
        "metrics": "metrics.csv",
        "model_selection": "model_selection_audit.csv",
        "trial_audit": "trial_audit.csv",
        "training_curves": "training_curves.csv",
        "graph_regime_audit": "graph_regime_audit.csv",
    }
    arch_cols = asdict(arch)
    arch_cols["run_id"] = run_id
    arch_cols["run_output_dir"] = str(subdir)

    for key, filename in file_map.items():
        path = subdir / filename
        if not path.exists():
            tables[key] = pd.DataFrame()
            continue
        df = pd.read_csv(path)
        for col, value in arch_cols.items():
            if col not in df.columns:
                df[col] = value
        tables[key] = df
    return tables


def run_sweep(args: argparse.Namespace) -> dict[str, Any]:
    graph_dir = Path(args.graph_dir)
    output_dir = Path(args.output_dir)
    a3_dir = Path(args.a3_dir) if args.a3_dir else None

    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    ensure_dir(output_dir)
    runs_dir = ensure_dir(output_dir / "runs")

    if args.monitor_metric not in MONITOR_HIGHER_IS_BETTER:
        raise ValueError(
            f"Unsupported monitor metric {args.monitor_metric!r}. "
            f"Supported: {sorted(MONITOR_HIGHER_IS_BETTER)}"
        )

    space = resolve_sweep_space(args)
    arch_configs = space.architecture_configs(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    if args.run_limit is not None:
        arch_configs = arch_configs[: int(args.run_limit)]

    manifest_rows: list[dict[str, Any]] = []
    for idx, arch in enumerate(arch_configs, start=1):
        run_id = arch.run_id()
        subdir = runs_dir / run_id
        manifest_rows.append(
            {
                "run_index": idx,
                "run_id": run_id,
                "run_output_dir": str(subdir),
                **asdict(arch),
                "status": "planned",
            }
        )
    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(output_dir / "sweep_manifest.csv", index=False)

    if args.dry_run:
        write_json(
            output_dir / "sweep_metadata.json",
            {
                "model_stage": MODEL_STAGE,
                "generated_at": now_utc(),
                "dry_run": True,
                "graph_dir": str(graph_dir),
                "output_dir": str(output_dir),
                "sweep_space": asdict(space),
                "n_architecture_configs": len(arch_configs),
                "expected_g1_engine_runs": len(arch_configs),
            },
        )
        return {
            "status": "dry_run_completed",
            "output_dir": str(output_dir),
            "n_architecture_configs": len(arch_configs),
        }

    run_results: list[dict[str, Any]] = []
    metrics_tables: list[pd.DataFrame] = []
    selection_tables: list[pd.DataFrame] = []
    trial_tables: list[pd.DataFrame] = []
    curve_tables: list[pd.DataFrame] = []
    graph_tables: list[pd.DataFrame] = []

    sweep_start = time.time()
    for idx, arch in enumerate(arch_configs, start=1):
        run_id = arch.run_id()
        subdir = runs_dir / run_id
        already_done = (subdir / "model_selection_audit.csv").exists() and not args.rerun_existing

        print(
            f"[{idx}/{len(arch_configs)}] G1.5 architecture {run_id}"
            + (" [existing; loading]" if already_done else ""),
            flush=True,
        )

        result: dict[str, Any]
        if already_done:
            result = {
                "status": "loaded_existing",
                "output_dir": str(subdir),
                "run_id": run_id,
            }
        else:
            try:
                result = run_architecture(
                    arch=arch,
                    graph_dir=graph_dir,
                    output_dir=subdir,
                    space=space,
                    args=args,
                )
                result["run_id"] = run_id
            except Exception as exc:
                result = {
                    "status": "failed",
                    "output_dir": str(subdir),
                    "run_id": run_id,
                    "failure_reason": repr(exc),
                }
                print(f"  failed: {exc}", flush=True)

        run_results.append({**asdict(arch), **result})

        tables = load_subrun_tables(subdir, arch, run_id)
        if not tables["metrics"].empty:
            metrics_tables.append(tables["metrics"])
        if not tables["model_selection"].empty:
            selection_tables.append(tables["model_selection"])
        if not tables["trial_audit"].empty:
            trial_tables.append(tables["trial_audit"])
        if not tables["training_curves"].empty:
            curve_tables.append(tables["training_curves"])
        if not tables["graph_regime_audit"].empty:
            graph_tables.append(tables["graph_regime_audit"])

    run_results_df = pd.DataFrame(run_results)
    run_results_df.to_csv(output_dir / "sweep_run_results.csv", index=False)

    sweep_metrics = pd.concat(metrics_tables, ignore_index=True) if metrics_tables else pd.DataFrame()
    sweep_selection = pd.concat(selection_tables, ignore_index=True) if selection_tables else pd.DataFrame()
    sweep_trial_audit = pd.concat(trial_tables, ignore_index=True) if trial_tables else pd.DataFrame()
    sweep_curves = pd.concat(curve_tables, ignore_index=True) if curve_tables else pd.DataFrame()
    sweep_graph_audit = pd.concat(graph_tables, ignore_index=True) if graph_tables else pd.DataFrame()

    sweep_selection = add_family_labels(sweep_selection)
    family_selection = build_family_selection(
        sweep_selection,
        selection_metric=args.monitor_metric,
    )

    # Mark family-selected rows in the full selection audit.
    sweep_selection["selected_within_family"] = False
    if not family_selection.empty and "model_name" in family_selection.columns:
        selected_names = set(family_selection["model_name"].astype(str))
        sweep_selection.loc[
            sweep_selection["model_name"].astype(str).isin(selected_names),
            "selected_within_family",
        ] = True

    a3_row = None
    if a3_dir is not None and a3_dir.exists():
        try:
            a3_row = load_a3_selected_row(a3_dir)
        except Exception as exc:
            print(f"Warning: failed to load A3 comparison row from {a3_dir}: {exc}", flush=True)

    final_comparison = build_final_comparison(
        a3_row=a3_row,
        family_selection=family_selection,
        selection_metric=args.monitor_metric,
    )
    winners = metric_winners(final_comparison)
    factor_summary = build_factor_summary(sweep_selection, selection_metric=args.monitor_metric)

    # Write sweep-level artifacts.
    outputs = {
        "sweep_manifest": str(output_dir / "sweep_manifest.csv"),
        "sweep_run_results": str(output_dir / "sweep_run_results.csv"),
        "sweep_metrics": str(output_dir / "sweep_metrics.csv"),
        "sweep_model_selection_audit": str(output_dir / "sweep_model_selection_audit.csv"),
        "sweep_trial_audit": str(output_dir / "sweep_trial_audit.csv"),
        "sweep_training_curves": str(output_dir / "sweep_training_curves.csv"),
        "sweep_graph_regime_audit": str(output_dir / "sweep_graph_regime_audit.csv"),
        "selection_by_family": str(output_dir / "selection_by_family.csv"),
        "final_comparison": str(output_dir / "final_comparison.csv"),
        "metric_winners": str(output_dir / "metric_winners.csv"),
        "factor_summary": str(output_dir / "factor_summary.csv"),
        "sweep_report": str(output_dir / "g1_validation_sweep_report.md"),
        "sweep_metadata": str(output_dir / "sweep_metadata.json"),
    }

    sweep_metrics.to_csv(outputs["sweep_metrics"], index=False)
    sweep_selection.to_csv(outputs["sweep_model_selection_audit"], index=False)
    sweep_trial_audit.to_csv(outputs["sweep_trial_audit"], index=False)
    sweep_curves.to_csv(outputs["sweep_training_curves"], index=False)
    sweep_graph_audit.to_csv(outputs["sweep_graph_regime_audit"], index=False)
    family_selection.to_csv(outputs["selection_by_family"], index=False)
    final_comparison.to_csv(outputs["final_comparison"], index=False)
    winners.to_csv(outputs["metric_winners"], index=False)
    factor_summary.to_csv(outputs["factor_summary"], index=False)

    metadata = {
        "model_stage": MODEL_STAGE,
        "generated_at": now_utc(),
        "graph_dir": str(graph_dir),
        "a3_dir": str(a3_dir) if a3_dir is not None else None,
        "output_dir": str(output_dir),
        "monitor_metric": args.monitor_metric,
        "monitor_higher_is_better": bool(MONITOR_HIGHER_IS_BETTER[args.monitor_metric]),
        "sweep_preset": args.sweep_preset,
        "sweep_space": asdict(space),
        "n_architecture_configs": len(arch_configs),
        "elapsed_seconds": time.time() - sweep_start,
        "outputs": outputs,
    }
    write_json(Path(outputs["sweep_metadata"]), metadata)

    report = render_report(
        metadata=metadata,
        run_results=run_results_df,
        family_selection=family_selection,
        final_comparison=final_comparison,
        winners=winners,
        factor_summary=factor_summary,
        outputs=outputs,
    )
    Path(outputs["sweep_report"]).write_text(report, encoding="utf-8")

    return {
        "status": "completed",
        "output_dir": str(output_dir),
        "n_architecture_configs": len(arch_configs),
        "n_completed_runs": int((run_results_df["status"].astype(str) != "failed").sum()) if not run_results_df.empty else 0,
        "n_failed_runs": int((run_results_df["status"].astype(str) == "failed").sum()) if not run_results_df.empty else 0,
        "elapsed_seconds": time.time() - sweep_start,
        "outputs": outputs,
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def render_report(
    *,
    metadata: Mapping[str, Any],
    run_results: pd.DataFrame,
    family_selection: pd.DataFrame,
    final_comparison: pd.DataFrame,
    winners: pd.DataFrame,
    factor_summary: pd.DataFrame,
    outputs: Mapping[str, str],
) -> str:
    lines: list[str] = []
    lines.append("# G1.5 Validation-Selected Architecture Sweep\n")
    lines.append(f"Generated at: `{metadata.get('generated_at')}`\n")
    lines.append(f"Graph directory: `{metadata.get('graph_dir')}`\n")
    lines.append(f"A3 comparison directory: `{metadata.get('a3_dir')}`\n")
    lines.append(f"Output directory: `{metadata.get('output_dir')}`\n")

    lines.append("## Purpose\n")
    lines.append(
        "This sweep is a bounded validation-selected G1 architecture/model-selection step. "
        "It tests whether G1 performance under spatial-block evaluation is sensitive to architecture "
        "and validation monitor choice. It is not a new graph construction and does not replace the "
        "frozen A3 baseline.\n"
    )

    lines.append("## Selection protocol\n")
    lines.append(
        f"Primary validation selection metric: `{metadata.get('monitor_metric')}` "
        f"({'higher is better' if metadata.get('monitor_higher_is_better') else 'lower is better'}).\n"
    )
    lines.append(
        "Each edge-regime family is selected using validation data only. Test metrics are joined "
        "only after selection for reporting. The no-edge neural control and random spatial placebo "
        "are retained as first-class controls.\n"
    )

    lines.append("## Sweep space\n")
    lines.append("```json")
    lines.append(json.dumps(to_jsonable(metadata.get("sweep_space", {})), indent=2))
    lines.append("```\n")

    lines.append("## Run summary\n")
    summary_cols = [
        "run_id",
        "status",
        "hidden_dim",
        "num_layers",
        "dropout",
        "normalization",
        "residual",
        "relation_combine",
        "backend",
        "trial_count",
        "completed_trial_count",
        "failed_trial_count",
        "elapsed_seconds",
    ]
    lines.append(dataframe_to_markdown(run_results[[c for c in summary_cols if c in run_results.columns]], max_rows=80))
    lines.append("")

    lines.append("## Selected representatives by family\n")
    family_cols = [
        "family",
        "model_name",
        "feature_regime",
        "edge_regime",
        "edge_mask_regime",
        "hidden_dim",
        "num_layers",
        "dropout",
        "normalization",
        "residual",
        "relation_combine",
        "seed",
        "validation_ndcg_at_100",
        "validation_spearman",
        "validation_mae",
        "test_ndcg_at_100",
        "test_spearman",
        "test_top_10pct_overlap_rate",
        "test_mae",
    ]
    lines.append(dataframe_to_markdown(family_selection[[c for c in family_cols if c in family_selection.columns]], max_rows=80))
    lines.append("")

    lines.append("## Final comparison\n")
    comparison_cols = [
        "comparison_role",
        "family",
        "edge_regime",
        "edge_mask_regime",
        "hidden_dim",
        "num_layers",
        "dropout",
        "test_mae",
        "test_spearman",
        "test_ndcg_at_100",
        "test_top_10pct_overlap_rate",
    ]
    lines.append(dataframe_to_markdown(final_comparison[[c for c in comparison_cols if c in final_comparison.columns]], max_rows=80))
    lines.append("")

    lines.append("## Metric winners\n")
    lines.append(dataframe_to_markdown(winners, max_rows=20))
    lines.append("")

    lines.append("## Factor summary preview\n")
    preview_cols = [
        "factor",
        "value",
        "n_trials",
        "mean_validation_ndcg_at_100",
        "best_validation_ndcg_at_100",
        "mean_test_ndcg_at_100",
        "best_test_ndcg_at_100",
        "mean_test_mae",
    ]
    lines.append(dataframe_to_markdown(factor_summary[[c for c in preview_cols if c in factor_summary.columns]], max_rows=120))
    lines.append("")

    lines.append("## Output artifacts\n")
    lines.append("| Artifact | Path |")
    lines.append("|---|---|")
    for key, path in outputs.items():
        lines.append(f"| `{key}` | `{path}` |")
    lines.append("")

    lines.append("## Interpretation guardrails\n")
    lines.append(
        "A G1.5 result should be interpreted as graph-specific only if a graph edge-regime family "
        "beats the validation-selected no-edge neural control and the random spatial placebo, not only "
        "the frozen A3 tabular model. If the no-edge control also wins, the result supports neural "
        "ranking-oriented model selection more strongly than graph-specific message passing.\n"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a validation-selected G1 architecture sweep using the existing G1 engine."
    )

    parser.add_argument("--graph-dir", default=DEFAULT_GRAPH_DIR)
    parser.add_argument("--a3-dir", default=DEFAULT_A3_SPATIAL_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_SWEEP_OUTPUT_DIR)
    parser.add_argument("--sweep-preset", default="compact", choices=["smoke", "compact", "expanded", "custom"])

    parser.add_argument("--feature-regimes", default=None, help="Comma-separated feature regimes.")
    parser.add_argument("--split-scheme", default=None, help="Split scheme, default from preset.")
    parser.add_argument("--edge-regimes", default=None, help="Comma-separated edge regimes.")
    parser.add_argument("--edge-mask-regimes", default=None, help="Comma-separated edge mask regimes.")
    parser.add_argument("--hidden-dims", default=None, help="Comma-separated hidden dimensions.")
    parser.add_argument("--num-layers-list", default=None, help="Comma-separated layer counts.")
    parser.add_argument("--dropouts", default=None, help="Comma-separated dropout values.")
    parser.add_argument("--normalizations", default=None, help="Comma-separated normalizations: layernorm,batchnorm,none.")
    parser.add_argument("--residual-options", default=None, help="Comma-separated booleans, e.g. true,false.")
    parser.add_argument("--relation-combines", default=None, help="Comma-separated relation combine modes: mean,sum.")
    parser.add_argument("--backends", default=None, help="Comma-separated backends: manual,pyg,auto.")
    parser.add_argument("--seeds", default=None, help="Comma-separated seeds.")

    parser.add_argument("--activation", default="relu")
    parser.add_argument("--max-epochs", type=int, default=250)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=5.0)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--monitor-metric", default=DEFAULT_MONITOR, choices=sorted(MONITOR_HIGHER_IS_BETTER))
    parser.add_argument("--no-checkpoints", action="store_true")
    parser.add_argument("--save-embeddings", default="none", choices=["none", "all"])

    parser.add_argument("--run-limit", type=int, default=None, help="Limit number of architecture configs, useful for debugging.")
    parser.add_argument("--dry-run", action="store_true", help="Write manifest/metadata without running models.")
    parser.add_argument("--overwrite", action="store_true", help="Delete sweep output directory before running.")
    parser.add_argument("--rerun-existing", action="store_true", help="Rerun sub-runs even if outputs already exist.")

    return parser.parse_args()


def brief(result: Mapping[str, Any]) -> str:
    outputs = result.get("outputs", {}) or {}
    lines = [
        "G1.5 validation sweep completed.",
        f"Status: {result.get('status')}",
        f"Output dir: {result.get('output_dir')}",
        f"Architecture configs: {result.get('n_architecture_configs')}",
        f"Completed/loaded runs: {result.get('n_completed_runs')}",
        f"Failed runs: {result.get('n_failed_runs')}",
        f"Elapsed seconds: {float(result.get('elapsed_seconds', 0.0)):.1f}",
    ]
    if outputs:
        lines.extend(
            [
                f"Final comparison: {outputs.get('final_comparison')}",
                f"Selection by family: {outputs.get('selection_by_family')}",
                f"Metric winners: {outputs.get('metric_winners')}",
                f"Report: {outputs.get('sweep_report')}",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    result = run_sweep(args)
    print(brief(result))


if __name__ == "__main__":
    main()
