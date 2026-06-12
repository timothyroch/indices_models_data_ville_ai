#!/usr/bin/env python3
"""
G1 spatiotemporal tract GNN baseline runner.

This module trains and evaluates the first controlled graph-value benchmark after
the frozen A0--A3 non-graph baseline layer.

It consumes the auditable graph artifacts produced by:

    ville_hgnn.graphs.build_tract_month_graph

and evaluates whether typed message passing over tract topology/time improves
prediction and ranking of reported water/drainage 311 burden beyond the A3
feature-parity tabular baselines.

Core benchmark question
-----------------------
Does message passing over urban topology and time improve prediction/ranking of
reported water/drainage 311 burden beyond a feature-parity tabular baseline?

Design principles
-----------------
1. Same task as A3:
   - node = census tract x month
   - target = water_drainage_count
   - predictions evaluated on the same temporal/spatial-block masks

2. Feature parity:
   - G1 uses graph-builder feature regimes derived from A3 feature sets
   - e.g. all_forecasting, lagged_reporting, no_target_history

3. Graph-value controls:
   - no_edges MLP
   - temporal_only
   - spatial_only
   - spatial_temporal
   - randomized spatial placebo

4. Split-safe message passing:
   - edge masks are read from edge_mask_by_split_regime.npz
   - temporal/spatial-block transductive and leakage-controlled regimes are
     explicit experiment dimensions, not hidden training-script behavior

5. A3-compatible target transform:
   - train on MSE(log1p(y_count), predicted_log_count)
   - report count-space metrics after expm1 and clipping at zero

Outputs
-------
The runner writes:

    metrics.csv
    predictions_validation.parquet
    predictions_test.parquet
    predictions_all_evaluated.parquet
    training_curves.csv
    trial_audit.csv
    model_selection_audit.csv
    graph_regime_audit.csv
    feature_preprocessing_audit.csv
    model_metadata.json
    baseline_report.md
    checkpoints/

Recommended test
----------------------
PYTHONPATH=urban_graph_benchmark/src python -m ville_hgnn.baselines.g1_spatiotemporal_gnn \\
  --graph-dir urban_graph_benchmark/outputs/mtl_311_water_v0/graphs/G1_tract_month_graph \\
  --preset smoke \\
  --max-epochs 20 \\
  --patience 5

Recommended first run
------------------------------
PYTHONPATH=urban_graph_benchmark/src python -m ville_hgnn.baselines.g1_spatiotemporal_gnn \\
  --graph-dir urban_graph_benchmark/outputs/mtl_311_water_v0/graphs/G1_tract_month_graph \\
  --preset core \\
  --max-epochs 250 \\
  --patience 30
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import time
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

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
    import torch
    from torch import Tensor, nn
except Exception as exc:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    Tensor = Any  # type: ignore[misc,assignment]
    nn = None  # type: ignore[assignment]
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None

try:
    from scipy import stats as scipy_stats  # type: ignore
except Exception:  # pragma: no cover
    scipy_stats = None  # type: ignore[assignment]

try:
    from ville_hgnn.models.spatiotemporal_graphsage import (
        DEFAULT_EDGE_REGIMES,
        G1ModelError,
        NoEdgesMLP,
        SpatioTemporalGraphSAGE,
        available_pyg,
        build_no_edges_mlp,
        build_spatiotemporal_graphsage,
        count_parameters,
        detach_outputs,
        invert_log_count_prediction,
        masked_log_count_mse_loss,
        masked_mae_in_count_space,
        model_summary,
        relation_kind,
        save_model_checkpoint,
        select_relations_for_regime,
    )
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "g1_spatiotemporal_gnn.py requires ville_hgnn.models.spatiotemporal_graphsage "
        "to be available on PYTHONPATH."
    ) from exc

try:
    from ville_hgnn.baselines.common import (
        BINARY_TARGET_COLUMN,
        DEFAULT_CONFIG_PATH,
        TARGET_COLUMN,
    )
except Exception:  # pragma: no cover
    DEFAULT_CONFIG_PATH = "urban_graph_benchmark/configs/mtl_311_water_v0.1.yaml"
    TARGET_COLUMN = "water_drainage_count"
    BINARY_TARGET_COLUMN = "water_drainage_binary"

try:
    from ville_hgnn.utils.io import to_jsonable, write_json, write_markdown
except Exception:  # pragma: no cover
    to_jsonable = None  # type: ignore[assignment]
    write_json = None  # type: ignore[assignment]
    write_markdown = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STAGE_SLUG = "G1_spatiotemporal_gnn"
MODEL_STAGE = "G1_spatiotemporal_gnn"

DEFAULT_GRAPH_DIR = "urban_graph_benchmark/outputs/mtl_311_water_v0/graphs/G1_tract_month_graph"
DEFAULT_OUTPUT_DIR = "urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_spatiotemporal_gnn"

ZONE_COL = "zone_id"
PERIOD_COL = "period_month"
NODE_ID_COL = "node_id"

SPLIT_PARTITIONS = ("train", "validation", "test")

DEFAULT_FEATURE_REGIMES = ("all_forecasting", "lagged_reporting", "no_target_history")
DEFAULT_SPLIT_SCHEMES = ("temporal", "spatial_block")
DEFAULT_EDGE_REGIME_GRID = (
    "no_edges",
    "temporal_only",
    "spatial_only",
    "spatial_temporal",
    "random_spatial_placebo",
)
DEFAULT_EDGE_MASK_REGIMES = ("all_edges", "no_test_incident_edges")

METRIC_HIGHER_IS_BETTER = {
    "count__mae": False,
    "count__rmse": False,
    "count__mean_poisson_deviance": False,
    "ranking__spearman_corr": True,
    "ranking__kendall_corr": True,
    "ranking__ndcg_at_100": True,
    "ranking__top_10pct_overlap_rate": True,
}

EPS = 1e-8


class G1BaselineError(RuntimeError):
    """Raised when the G1 baseline runner fails."""


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrainConfig:
    """Training hyperparameters shared by all trials."""

    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.15
    activation: str = "relu"
    normalization: str = "layernorm"
    residual: bool = True
    backend: str = "manual"
    relation_combine: str = "mean"

    max_epochs: int = 250
    patience: int = 30
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip_norm: float = 5.0
    min_delta: float = 1e-5

    seed: int = 20240610
    device: str = "auto"
    count_min: float = 0.0
    monitor_metric: str = "validation_mae"
    save_checkpoints: bool = True
    save_embeddings: str = "none"  # none, all


@dataclass(frozen=True)
class TrialSpec:
    """One G1 experiment trial."""

    feature_regime: str
    split_scheme: str
    edge_regime: str
    edge_mask_regime: str
    seed: int

    def model_name(self, train_config: TrainConfig) -> str:
        """Stable model name used in metrics and artifacts."""

        return (
            f"G1__{self.split_scheme}__{self.feature_regime}__"
            f"{self.edge_regime}__{self.edge_mask_regime}__"
            f"h{train_config.hidden_dim}_L{train_config.num_layers}_seed{self.seed}"
        )


@dataclass
class FeaturePreprocessor:
    """Train-only median imputation and standardization."""

    feature_regime: str
    split_scheme: str
    feature_columns: list[str]
    train_median: np.ndarray
    train_mean: np.ndarray
    train_std: np.ndarray

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply train-only imputation and scaling."""

        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2:
            raise G1BaselineError(f"Expected 2D feature matrix, got shape {X.shape}.")
        if X.shape[1] != len(self.feature_columns):
            raise G1BaselineError(
                f"Feature matrix has {X.shape[1]} columns but preprocessor expects "
                f"{len(self.feature_columns)}."
            )
        out = X.copy()
        nan_mask = ~np.isfinite(out)
        if nan_mask.any():
            out[nan_mask] = np.take(self.train_median, np.where(nan_mask)[1])
        out = (out - self.train_mean) / self.train_std
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        return out

    def audit_frame(self) -> pd.DataFrame:
        """Return feature preprocessing audit table."""

        return pd.DataFrame(
            {
                "feature_regime": self.feature_regime,
                "split_scheme": self.split_scheme,
                "feature": self.feature_columns,
                "train_median": self.train_median,
                "train_mean": self.train_mean,
                "train_std": self.train_std,
            }
        )


@dataclass
class GraphBundle:
    """Loaded graph artifacts."""

    graph_dir: Path
    node_table: pd.DataFrame
    edge_table: pd.DataFrame
    targets: np.ndarray
    binary_targets: np.ndarray
    split_masks: dict[str, np.ndarray]
    edge_masks: dict[str, np.ndarray]
    feature_matrix_paths: dict[str, Path]
    feature_columns: dict[str, list[str]]
    metadata: dict[str, Any]


@dataclass
class TrialResult:
    """Result artifacts for one trained trial."""

    spec: TrialSpec
    model_name: str
    status: str
    best_epoch: int
    best_validation_mae: float
    best_validation_loss: float
    elapsed_seconds: float
    n_parameters: int
    n_edges_used: int
    n_relations_used: int
    checkpoint_path: str | None
    embedding_path: str | None
    failure_reason: str | None = None


# ---------------------------------------------------------------------------
# Dependency and IO helpers
# ---------------------------------------------------------------------------

def require_runtime_dependencies() -> None:
    """Fail clearly if dependencies are missing."""

    if np is None:
        raise G1BaselineError("numpy is required.") from _NUMPY_IMPORT_ERROR
    if pd is None:
        raise G1BaselineError("pandas is required.") from _PANDAS_IMPORT_ERROR
    if torch is None:
        raise G1BaselineError("torch is required.") from _TORCH_IMPORT_ERROR


def now_utc() -> str:
    """Return current UTC timestamp."""

    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: Path) -> Path:
    """Create directory and return it."""

    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_filename(value: str, max_len: int = 160) -> str:
    """Create a safe filename stem."""

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


def parse_csv_list(value: str | Sequence[str] | None, default: Sequence[str]) -> tuple[str, ...]:
    """Parse comma-separated list."""

    if value is None:
        return tuple(default)
    if isinstance(value, str):
        items = [v.strip() for v in value.split(",") if v.strip()]
        return tuple(items) if items else tuple(default)
    return tuple(str(v).strip() for v in value if str(v).strip())


def jsonable(obj: Any) -> Any:
    """Best-effort JSON conversion."""

    if to_jsonable is not None:
        try:
            return to_jsonable(obj)
        except Exception:
            pass
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
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    return obj


def write_json_file(path: Path, data: Mapping[str, Any]) -> None:
    """Write JSON with project helper fallback."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if write_json is not None:
        try:
            write_json(path, jsonable(data))
            return
        except Exception:
            pass
    path.write_text(json.dumps(jsonable(data), indent=2, ensure_ascii=False), encoding="utf-8")


def write_markdown_file(path: Path, text: str) -> None:
    """Write Markdown with project helper fallback."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if write_markdown is not None:
        try:
            write_markdown(path, text)
            return
        except Exception:
            pass
    path.write_text(text, encoding="utf-8")


def read_json_file(path: Path) -> dict[str, Any]:
    """Read JSON file if present."""

    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(False)
    except Exception:
        pass


def resolve_device(device_arg: str) -> torch.device:
    """Resolve device string."""

    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise G1BaselineError("CUDA was requested but torch.cuda.is_available() is False.")
    return device


def dataframe_to_markdown(df: pd.DataFrame, max_rows: int = 60) -> str:
    """Render dataframe as Markdown."""

    if df.empty:
        return "_No rows._"
    display = df.head(max_rows).copy()
    try:
        return display.to_markdown(index=False)
    except Exception:
        return "```text\n" + display.to_string(index=False) + "\n```"


# ---------------------------------------------------------------------------
# Graph artifact loading
# ---------------------------------------------------------------------------

def discover_feature_matrices(graph_dir: Path) -> tuple[dict[str, Path], dict[str, list[str]]]:
    """Discover feature matrices and column lists in graph artifact folder."""

    matrix_paths: dict[str, Path] = {}
    columns: dict[str, list[str]] = {}

    for path in graph_dir.glob("feature_matrix__*__raw.npy"):
        stem = path.name
        regime = stem.removeprefix("feature_matrix__").removesuffix("__raw.npy")
        matrix_paths[regime] = path

    for path in graph_dir.glob("feature_columns__*.json"):
        regime = path.name.removeprefix("feature_columns__").removesuffix(".json")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise G1BaselineError(f"Failed reading feature columns {path}: {exc}") from exc
        if isinstance(data, Mapping) and "feature_columns" in data:
            cols = data["feature_columns"]
        else:
            cols = data
        columns[regime] = [str(c) for c in cols]

    missing_cols = sorted(set(matrix_paths) - set(columns))
    if missing_cols:
        raise G1BaselineError(f"Missing feature_columns JSON for regimes: {missing_cols}")

    if not matrix_paths:
        raise G1BaselineError(f"No feature_matrix__*__raw.npy files found in {graph_dir}")

    return matrix_paths, columns


def load_npz_bool_masks(path: Path) -> dict[str, np.ndarray]:
    """Load boolean masks from npz."""

    if not path.exists():
        raise G1BaselineError(f"Required mask file not found: {path}")
    loaded = np.load(path)
    out: dict[str, np.ndarray] = {}
    for key in loaded.files:
        arr = np.asarray(loaded[key]).astype(bool)
        out[str(key)] = arr
    return out


def load_graph_bundle(graph_dir: str | Path) -> GraphBundle:
    """Load graph artifacts produced by build_tract_month_graph.py."""

    require_runtime_dependencies()
    graph_dir = Path(graph_dir)

    node_table_path = graph_dir / "node_table.parquet"
    edge_table_path = graph_dir / "edge_table.parquet"
    target_vector_path = graph_dir / "target_vector.npy"
    binary_target_path = graph_dir / "binary_target_vector.npy"
    split_masks_path = graph_dir / "split_masks.npz"
    edge_masks_path = graph_dir / "edge_mask_by_split_regime.npz"
    metadata_path = graph_dir / "graph_metadata.json"

    required = [
        node_table_path,
        edge_table_path,
        target_vector_path,
        binary_target_path,
        split_masks_path,
        edge_masks_path,
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise G1BaselineError("Missing required graph artifacts:\n" + "\n".join(missing))

    node_table = pd.read_parquet(node_table_path)
    edge_table = pd.read_parquet(edge_table_path)
    targets = np.load(target_vector_path).astype(np.float32)
    binary_targets = np.load(binary_target_path).astype(np.float32)
    split_masks = load_npz_bool_masks(split_masks_path)
    edge_masks = load_npz_bool_masks(edge_masks_path)
    matrix_paths, columns = discover_feature_matrices(graph_dir)
    metadata = read_json_file(metadata_path)

    n_nodes = len(node_table)
    if targets.shape[0] != n_nodes:
        raise G1BaselineError(f"target_vector length {targets.shape[0]} != node count {n_nodes}.")
    if binary_targets.shape[0] != n_nodes:
        raise G1BaselineError(f"binary_target_vector length {binary_targets.shape[0]} != node count {n_nodes}.")

    for key, mask in split_masks.items():
        if mask.shape[0] != n_nodes:
            raise G1BaselineError(f"split mask {key!r} length {mask.shape[0]} != node count {n_nodes}.")

    n_edges = len(edge_table)
    for key, mask in edge_masks.items():
        if mask.shape[0] != n_edges:
            raise G1BaselineError(f"edge mask {key!r} length {mask.shape[0]} != edge count {n_edges}.")

    if "edge_type" not in edge_table.columns:
        raise G1BaselineError("edge_table is missing required column 'edge_type'.")

    return GraphBundle(
        graph_dir=graph_dir,
        node_table=node_table,
        edge_table=edge_table,
        targets=targets,
        binary_targets=binary_targets,
        split_masks=split_masks,
        edge_masks=edge_masks,
        feature_matrix_paths=matrix_paths,
        feature_columns=columns,
        metadata=metadata,
    )


def load_feature_matrix(bundle: GraphBundle, feature_regime: str) -> np.ndarray:
    """Load raw feature matrix for a feature regime."""

    if feature_regime not in bundle.feature_matrix_paths:
        raise G1BaselineError(
            f"Feature regime {feature_regime!r} not found. Available: {sorted(bundle.feature_matrix_paths)}"
        )
    X = np.load(bundle.feature_matrix_paths[feature_regime]).astype(np.float32)
    if X.ndim != 2:
        raise G1BaselineError(f"Feature matrix for {feature_regime!r} is not 2D: {X.shape}")
    if X.shape[0] != len(bundle.node_table):
        raise G1BaselineError(
            f"Feature matrix {feature_regime!r} has {X.shape[0]} rows but node table has {len(bundle.node_table)}."
        )
    expected_cols = len(bundle.feature_columns[feature_regime])
    if X.shape[1] != expected_cols:
        raise G1BaselineError(
            f"Feature matrix {feature_regime!r} has {X.shape[1]} cols but columns JSON has {expected_cols}."
        )
    return X


# ---------------------------------------------------------------------------
# Feature preprocessing and edge tensors
# ---------------------------------------------------------------------------

def fit_feature_preprocessor(
    X: np.ndarray,
    feature_columns: Sequence[str],
    train_mask: np.ndarray,
    *,
    feature_regime: str,
    split_scheme: str,
) -> FeaturePreprocessor:
    """Fit train-only median imputer and standardizer."""

    X = np.asarray(X, dtype=np.float32)
    train_mask = np.asarray(train_mask, dtype=bool)
    if X.ndim != 2:
        raise G1BaselineError(f"Expected X shape [n, p], got {X.shape}.")
    if train_mask.ndim != 1 or train_mask.shape[0] != X.shape[0]:
        raise G1BaselineError("train_mask shape does not match X.")
    if int(train_mask.sum()) <= 0:
        raise G1BaselineError("train_mask contains no rows.")

    X_train = X[train_mask].astype(np.float64)
    finite_train = np.isfinite(X_train)

    medians = np.zeros(X.shape[1], dtype=np.float64)
    means = np.zeros(X.shape[1], dtype=np.float64)
    stds = np.ones(X.shape[1], dtype=np.float64)

    for j in range(X.shape[1]):
        col = X_train[:, j]
        finite = finite_train[:, j]
        if finite.any():
            med = float(np.nanmedian(col[finite]))
        else:
            med = 0.0
        medians[j] = med

        imputed = col.copy()
        imputed[~np.isfinite(imputed)] = med
        means[j] = float(np.mean(imputed))
        std = float(np.std(imputed, ddof=0))
        if not math.isfinite(std) or std <= EPS:
            std = 1.0
        stds[j] = std

    return FeaturePreprocessor(
        feature_regime=feature_regime,
        split_scheme=split_scheme,
        feature_columns=list(feature_columns),
        train_median=medians.astype(np.float32),
        train_mean=means.astype(np.float32),
        train_std=stds.astype(np.float32),
    )


def split_mask_key(split_scheme: str, partition: str) -> str:
    """Return split mask key."""

    return f"{split_scheme}_{partition}"


def require_split_masks(bundle: GraphBundle, split_scheme: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return train/validation/test masks for a split scheme."""

    keys = [split_mask_key(split_scheme, p) for p in SPLIT_PARTITIONS]
    missing = [k for k in keys if k not in bundle.split_masks]
    if missing:
        raise G1BaselineError(f"Missing split masks for {split_scheme!r}: {missing}")
    return (
        bundle.split_masks[keys[0]].astype(bool),
        bundle.split_masks[keys[1]].astype(bool),
        bundle.split_masks[keys[2]].astype(bool),
    )


def edge_mask_key(split_scheme: str, edge_mask_regime: str) -> str:
    """Return edge mask key from split and mask regime."""

    return f"{split_scheme}_{edge_mask_regime}"


def selected_edge_types_for_regime(edge_types: Sequence[str], edge_regime: str) -> list[str]:
    """Select edge types for an edge-regime ablation."""

    return select_relations_for_regime(edge_types, edge_regime)


def build_edge_tensors_for_trial(
    bundle: GraphBundle,
    spec: TrialSpec,
    device: torch.device,
) -> tuple[dict[str, Tensor], dict[str, Tensor], pd.DataFrame]:
    """Build typed edge tensors for a trial."""

    if spec.edge_regime == "no_edges":
        return {}, {}, bundle.edge_table.iloc[0:0].copy()

    mask_name = edge_mask_key(spec.split_scheme, spec.edge_mask_regime)
    if mask_name not in bundle.edge_masks:
        raise G1BaselineError(
            f"Edge mask {mask_name!r} not found. Available: {sorted(bundle.edge_masks)}"
        )

    global_edge_mask = bundle.edge_masks[mask_name].astype(bool)
    edge_table = bundle.edge_table.loc[global_edge_mask].copy()

    all_types = sorted(edge_table["edge_type"].astype(str).unique())
    selected_types = selected_edge_types_for_regime(all_types, spec.edge_regime)
    if not selected_types:
        raise G1BaselineError(
            f"Edge regime {spec.edge_regime!r} selected no edge types from {all_types}."
        )

    edge_table = edge_table[edge_table["edge_type"].astype(str).isin(selected_types)].copy()
    if edge_table.empty:
        raise G1BaselineError(f"No edges remain after applying {spec}.")

    edge_index_by_type: dict[str, Tensor] = {}
    edge_weight_by_type: dict[str, Tensor] = {}
    for edge_type, part in edge_table.groupby("edge_type", sort=True):
        edge_index = part[["source_node_id", "target_node_id"]].to_numpy(dtype=np.int64).T
        edge_weight = pd.to_numeric(part["edge_weight"], errors="coerce").fillna(1.0).to_numpy(dtype=np.float32)
        edge_index_by_type[str(edge_type)] = torch.as_tensor(edge_index, dtype=torch.long, device=device)
        edge_weight_by_type[str(edge_type)] = torch.as_tensor(edge_weight, dtype=torch.float32, device=device)

    return edge_index_by_type, edge_weight_by_type, edge_table


def trial_graph_regime_audit_row(
    spec: TrialSpec,
    edge_table_used: pd.DataFrame,
    all_edge_count: int,
) -> dict[str, Any]:
    """Summarize graph regime for one trial."""

    if edge_table_used.empty:
        return {
            "feature_regime": spec.feature_regime,
            "split_scheme": spec.split_scheme,
            "edge_regime": spec.edge_regime,
            "edge_mask_regime": spec.edge_mask_regime,
            "n_edges_total_graph": int(all_edge_count),
            "n_edges_used": 0,
            "n_edge_types_used": 0,
            "edge_types_used": "",
            "uses_temporal_edges": False,
            "uses_spatial_edges": False,
            "uses_placebo_edges": False,
        }

    edge_types = sorted(edge_table_used["edge_type"].astype(str).unique())
    return {
        "feature_regime": spec.feature_regime,
        "split_scheme": spec.split_scheme,
        "edge_regime": spec.edge_regime,
        "edge_mask_regime": spec.edge_mask_regime,
        "n_edges_total_graph": int(all_edge_count),
        "n_edges_used": int(len(edge_table_used)),
        "n_edge_types_used": int(len(edge_types)),
        "edge_types_used": ",".join(edge_types),
        "uses_temporal_edges": bool(edge_table_used["is_temporal"].astype(bool).any()) if "is_temporal" in edge_table_used.columns else any(relation_kind(e) == "temporal" for e in edge_types),
        "uses_spatial_edges": bool(edge_table_used["is_spatial"].astype(bool).any()) if "is_spatial" in edge_table_used.columns else any(relation_kind(e) == "spatial" for e in edge_types),
        "uses_placebo_edges": bool(edge_table_used["is_placebo"].astype(bool).any()) if "is_placebo" in edge_table_used.columns else any(relation_kind(e) == "placebo" for e in edge_types),
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _finite_arrays(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return finite target/prediction arrays."""

    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    y = y[mask]
    p = p[mask]
    p = np.clip(p, 0.0, None)
    return y, p


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error."""

    y, p = _finite_arrays(y_true, y_pred)
    if len(y) == 0:
        return np.nan
    return float(np.mean(np.abs(y - p)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error."""

    y, p = _finite_arrays(y_true, y_pred)
    if len(y) == 0:
        return np.nan
    return float(np.sqrt(np.mean((y - p) ** 2)))


def mean_poisson_deviance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Poisson deviance with safe clipping."""

    y, p = _finite_arrays(y_true, y_pred)
    if len(y) == 0:
        return np.nan
    p = np.clip(p, EPS, None)
    positive = y > 0
    term = np.zeros_like(y, dtype=float)
    term[positive] = y[positive] * np.log(y[positive] / p[positive])
    dev = 2.0 * (term - y + p)
    return float(np.mean(dev))


def spearman_corr(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Spearman correlation."""

    y, s = _finite_arrays(y_true, y_score)
    if len(y) < 2 or np.nanstd(y) <= EPS or np.nanstd(s) <= EPS:
        return np.nan
    if scipy_stats is not None:
        return float(scipy_stats.spearmanr(y, s, nan_policy="omit").correlation)
    return float(pd.Series(y).corr(pd.Series(s), method="spearman"))


def kendall_corr(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Kendall tau correlation."""

    y, s = _finite_arrays(y_true, y_score)
    if len(y) < 2 or np.nanstd(y) <= EPS or np.nanstd(s) <= EPS:
        return np.nan
    if scipy_stats is not None:
        return float(scipy_stats.kendalltau(y, s, nan_policy="omit").correlation)
    try:
        return float(pd.Series(y).corr(pd.Series(s), method="kendall"))
    except Exception:
        return np.nan


def dcg_at_k(relevance: np.ndarray, k: int) -> float:
    """Discounted cumulative gain."""

    rel = np.asarray(relevance, dtype=float)[:k]
    if rel.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, rel.size + 2))
    return float(np.sum(rel * discounts))


def ndcg_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int = 100) -> float:
    """NDCG@k using count target as relevance."""

    y, s = _finite_arrays(y_true, y_score)
    if len(y) == 0:
        return np.nan
    k_eff = min(k, len(y))
    order_pred = np.argsort(-s, kind="mergesort")
    order_ideal = np.argsort(-y, kind="mergesort")
    dcg = dcg_at_k(y[order_pred], k_eff)
    ideal = dcg_at_k(y[order_ideal], k_eff)
    if ideal <= EPS:
        return np.nan
    return float(dcg / ideal)


def top_fraction_overlap(y_true: np.ndarray, y_score: np.ndarray, fraction: float = 0.10) -> float:
    """Overlap between top-fraction predicted and observed sets."""

    y, s = _finite_arrays(y_true, y_score)
    if len(y) == 0:
        return np.nan
    k = max(1, int(math.ceil(fraction * len(y))))
    pred_top = set(np.argsort(-s, kind="mergesort")[:k].tolist())
    true_top = set(np.argsort(-y, kind="mergesort")[:k].tolist())
    return float(len(pred_top & true_top) / k)


def compute_metric_rows(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    split_name: str,
    spec: TrialSpec,
    model_name: str,
    n_rows: int,
) -> list[dict[str, Any]]:
    """Compute all benchmark metric rows."""

    values = {
        "count__mae": mae(y_true, y_pred),
        "count__rmse": rmse(y_true, y_pred),
        "count__mean_poisson_deviance": mean_poisson_deviance(y_true, y_pred),
        "ranking__spearman_corr": spearman_corr(y_true, y_pred),
        "ranking__kendall_corr": kendall_corr(y_true, y_pred),
        "ranking__ndcg_at_100": ndcg_at_k(y_true, y_pred, k=100),
        "ranking__top_10pct_overlap_rate": top_fraction_overlap(y_true, y_pred, fraction=0.10),
    }

    rows = []
    for metric_name, metric_value in values.items():
        rows.append(
            {
                "model_stage": MODEL_STAGE,
                "model_name": model_name,
                "split_scheme": spec.split_scheme,
                "split_name": split_name,
                "feature_regime": spec.feature_regime,
                "edge_regime": spec.edge_regime,
                "edge_mask_regime": spec.edge_mask_regime,
                "metric_name": metric_name,
                "metric_value": float(metric_value) if metric_value is not None and math.isfinite(metric_value) else np.nan,
                "higher_is_better": bool(METRIC_HIGHER_IS_BETTER[metric_name]),
                "n_rows": int(n_rows),
                "seed": int(spec.seed),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Trial grid
# ---------------------------------------------------------------------------

def preset_trial_dimensions(preset: str) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Return feature/split/edge/mask dimensions for a preset."""

    if preset == "smoke":
        return (
            ("lagged_reporting",),
            ("temporal",),
            ("no_edges", "temporal_only", "spatial_temporal"),
            ("all_edges",),
        )

    if preset == "temporal_core":
        return (
            ("all_forecasting", "lagged_reporting", "no_target_history"),
            ("temporal",),
            ("no_edges", "temporal_only", "spatial_only", "spatial_temporal", "random_spatial_placebo"),
            ("all_edges",),
        )

    if preset == "spatial_core":
        return (
            ("all_forecasting", "lagged_reporting", "no_target_history"),
            ("spatial_block",),
            ("no_edges", "temporal_only", "spatial_only", "spatial_temporal", "random_spatial_placebo"),
            ("all_edges", "no_test_incident_edges"),
        )

    if preset == "core":
        return (
            ("all_forecasting", "lagged_reporting", "no_target_history"),
            ("temporal", "spatial_block"),
            ("no_edges", "temporal_only", "spatial_only", "spatial_temporal", "random_spatial_placebo"),
            ("all_edges", "no_test_incident_edges"),
        )

    if preset == "full":
        return (
            ("all_forecasting", "lagged_reporting", "no_target_history"),
            ("temporal", "spatial_block", "random_debug"),
            ("no_edges", "temporal_only", "spatial_only", "spatial_temporal", "random_spatial_placebo", "all_edges"),
            ("all_edges", "train_train_edges", "no_test_incident_edges"),
        )

    if preset == "custom":
        return ((), (), (), ())

    raise G1BaselineError(f"Unknown preset {preset!r}.")


def build_trial_specs(
    *,
    preset: str,
    feature_regimes: Sequence[str] | None,
    split_schemes: Sequence[str] | None,
    edge_regimes: Sequence[str] | None,
    edge_mask_regimes: Sequence[str] | None,
    seeds: Sequence[int],
    available_feature_regimes: Sequence[str],
    available_split_mask_keys: Sequence[str],
    available_edge_mask_keys: Sequence[str],
) -> list[TrialSpec]:
    """Build trial specs from preset/custom dimensions."""

    preset_features, preset_splits, preset_edges, preset_masks = preset_trial_dimensions(preset)

    features = tuple(feature_regimes or preset_features or DEFAULT_FEATURE_REGIMES)
    splits = tuple(split_schemes or preset_splits or DEFAULT_SPLIT_SCHEMES)
    edges = tuple(edge_regimes or preset_edges or DEFAULT_EDGE_REGIME_GRID)
    masks = tuple(edge_mask_regimes or preset_masks or DEFAULT_EDGE_MASK_REGIMES)

    missing_features = sorted(set(features) - set(available_feature_regimes))
    if missing_features:
        raise G1BaselineError(
            f"Requested missing feature regimes: {missing_features}. "
            f"Available: {sorted(available_feature_regimes)}"
        )

    trial_specs: list[TrialSpec] = []
    for seed in seeds:
        for split in splits:
            # Ensure split masks exist.
            missing_split_masks = [
                split_mask_key(split, part)
                for part in SPLIT_PARTITIONS
                if split_mask_key(split, part) not in available_split_mask_keys
            ]
            if missing_split_masks:
                warnings.warn(f"Skipping split {split!r}; missing masks {missing_split_masks}.")
                continue

            for feature in features:
                for edge in edges:
                    if edge not in DEFAULT_EDGE_REGIMES:
                        raise G1BaselineError(f"Unsupported edge regime {edge!r}.")
                    if edge == "no_edges":
                        trial_specs.append(
                            TrialSpec(
                                feature_regime=feature,
                                split_scheme=split,
                                edge_regime=edge,
                                edge_mask_regime="all_edges",
                                seed=int(seed),
                            )
                        )
                        continue

                    for mask in masks:
                        key = edge_mask_key(split, mask)
                        if key not in available_edge_mask_keys:
                            warnings.warn(
                                f"Skipping split={split}, edge_regime={edge}, mask={mask}; "
                                f"missing edge mask {key!r}."
                            )
                            continue
                        trial_specs.append(
                            TrialSpec(
                                feature_regime=feature,
                                split_scheme=split,
                                edge_regime=edge,
                                edge_mask_regime=mask,
                                seed=int(seed),
                            )
                        )

    # Deduplicate while preserving order.
    seen = set()
    unique: list[TrialSpec] = []
    for spec in trial_specs:
        key = (spec.feature_regime, spec.split_scheme, spec.edge_regime, spec.edge_mask_regime, spec.seed)
        if key not in seen:
            unique.append(spec)
            seen.add(key)
    return unique


# ---------------------------------------------------------------------------
# Model construction and training
# ---------------------------------------------------------------------------

def build_model_for_trial(
    spec: TrialSpec,
    input_dim: int,
    relation_names: Sequence[str],
    train_config: TrainConfig,
) -> nn.Module:
    """Build MLP or GraphSAGE for one trial."""

    if spec.edge_regime == "no_edges":
        return build_no_edges_mlp(
            input_dim=input_dim,
            hidden_dim=train_config.hidden_dim,
            output_dim=1,
            num_hidden_layers=max(train_config.num_layers, 1),
            dropout=train_config.dropout,
            activation=train_config.activation,
            normalization=train_config.normalization,
            count_min=train_config.count_min,
        )

    return build_spatiotemporal_graphsage(
        input_dim=input_dim,
        relation_names=relation_names,
        hidden_dim=train_config.hidden_dim,
        output_dim=1,
        num_layers=train_config.num_layers,
        dropout=train_config.dropout,
        activation=train_config.activation,
        normalization=train_config.normalization,
        residual=train_config.residual,
        backend=train_config.backend,
        relation_combine=train_config.relation_combine,
        count_min=train_config.count_min,
    )


def forward_model(
    model: nn.Module,
    x: Tensor,
    *,
    spec: TrialSpec,
    edge_index_by_type: Mapping[str, Tensor],
    edge_weight_by_type: Mapping[str, Tensor],
) -> Mapping[str, Tensor]:
    """Forward pass for MLP or GraphSAGE."""

    if isinstance(model, NoEdgesMLP):
        return model(x, return_embeddings=True)

    if isinstance(model, SpatioTemporalGraphSAGE):
        return model(
            x,
            edge_index_by_type=edge_index_by_type,
            edge_weight_by_type=edge_weight_by_type,
            edge_regime=spec.edge_regime,
            return_embeddings=True,
        )

    # Generic fallback for compatible modules.
    return model(x)


def evaluate_epoch_monitor(
    model: nn.Module,
    x: Tensor,
    y: Tensor,
    val_mask: Tensor,
    *,
    spec: TrialSpec,
    edge_index_by_type: Mapping[str, Tensor],
    edge_weight_by_type: Mapping[str, Tensor],
    count_min: float,
) -> tuple[float, float]:
    """Return validation log-loss and count MAE."""

    model.eval()
    with torch.no_grad():
        out = forward_model(
            model,
            x,
            spec=spec,
            edge_index_by_type=edge_index_by_type,
            edge_weight_by_type=edge_weight_by_type,
        )
        val_loss = masked_log_count_mse_loss(out["log_count"], y, val_mask)
        val_mae = masked_mae_in_count_space(out["log_count"], y, val_mask, count_min=count_min)
    return float(val_loss.detach().cpu()), float(val_mae.detach().cpu())


def train_one_trial(
    bundle: GraphBundle,
    spec: TrialSpec,
    train_config: TrainConfig,
    output_dir: Path,
) -> tuple[TrialResult, pd.DataFrame, pd.DataFrame, list[dict[str, Any]], dict[str, Any], pd.DataFrame]:
    """Train and evaluate one trial.

    Returns
    -------
    TrialResult
    training_curve_df
    predictions_df
    metric_rows
    preprocessing_metadata
    graph_regime_row
    """

    start_time = time.time()
    set_global_seed(spec.seed)

    device = resolve_device(train_config.device)
    checkpoints_dir = ensure_dir(output_dir / "checkpoints")
    embeddings_dir = ensure_dir(output_dir / "embeddings")

    model_name = spec.model_name(train_config)

    train_mask_np, val_mask_np, test_mask_np = require_split_masks(bundle, spec.split_scheme)
    X_raw = load_feature_matrix(bundle, spec.feature_regime)
    feature_columns = bundle.feature_columns[spec.feature_regime]
    preprocessor = fit_feature_preprocessor(
        X_raw,
        feature_columns,
        train_mask_np,
        feature_regime=spec.feature_regime,
        split_scheme=spec.split_scheme,
    )
    X = preprocessor.transform(X_raw)

    edge_index_by_type, edge_weight_by_type, edge_table_used = build_edge_tensors_for_trial(bundle, spec, device)
    graph_regime_row = trial_graph_regime_audit_row(spec, edge_table_used, len(bundle.edge_table))

    x_t = torch.as_tensor(X, dtype=torch.float32, device=device)
    y_t = torch.as_tensor(bundle.targets, dtype=torch.float32, device=device)
    train_mask_t = torch.as_tensor(train_mask_np, dtype=torch.bool, device=device)
    val_mask_t = torch.as_tensor(val_mask_np, dtype=torch.bool, device=device)

    relation_names = sorted(edge_index_by_type.keys())
    model = build_model_for_trial(
        spec,
        input_dim=X.shape[1],
        relation_names=relation_names,
        train_config=train_config,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )

    best_state: dict[str, Tensor] | None = None
    best_epoch = -1
    best_val_mae = float("inf")
    best_val_loss = float("inf")
    best_monitor = float("inf")
    epochs_without_improvement = 0
    training_rows: list[dict[str, Any]] = []

    for epoch in range(1, train_config.max_epochs + 1):
        epoch_start = time.time()
        model.train()
        optimizer.zero_grad(set_to_none=True)

        out = forward_model(
            model,
            x_t,
            spec=spec,
            edge_index_by_type=edge_index_by_type,
            edge_weight_by_type=edge_weight_by_type,
        )
        train_loss = masked_log_count_mse_loss(out["log_count"], y_t, train_mask_t)
        train_loss.backward()

        if train_config.grad_clip_norm and train_config.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=train_config.grad_clip_norm)

        optimizer.step()

        val_loss, val_mae = evaluate_epoch_monitor(
            model,
            x_t,
            y_t,
            val_mask_t,
            spec=spec,
            edge_index_by_type=edge_index_by_type,
            edge_weight_by_type=edge_weight_by_type,
            count_min=train_config.count_min,
        )

        train_mae = float(masked_mae_in_count_space(out["log_count"].detach(), y_t, train_mask_t, train_config.count_min).detach().cpu())
        train_loss_value = float(train_loss.detach().cpu())

        monitor = val_mae if train_config.monitor_metric == "validation_mae" else val_loss
        improved = monitor < (best_monitor - train_config.min_delta)

        if improved:
            best_monitor = monitor
            best_val_mae = val_mae
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        training_rows.append(
            {
                "model_stage": MODEL_STAGE,
                "model_name": model_name,
                "epoch": epoch,
                "split_scheme": spec.split_scheme,
                "feature_regime": spec.feature_regime,
                "edge_regime": spec.edge_regime,
                "edge_mask_regime": spec.edge_mask_regime,
                "train_loss_log_mse": train_loss_value,
                "train_mae_count": train_mae,
                "validation_loss_log_mse": val_loss,
                "validation_mae_count": val_mae,
                "best_validation_mae_so_far": best_val_mae,
                "best_epoch_so_far": best_epoch,
                "epoch_seconds": time.time() - epoch_start,
                "learning_rate": train_config.learning_rate,
            }
        )

        if epochs_without_improvement >= train_config.patience:
            break

    if best_state is None:
        raise G1BaselineError(f"No best state was recorded for {model_name}.")

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    with torch.no_grad():
        final_out = forward_model(
            model,
            x_t,
            spec=spec,
            edge_index_by_type=edge_index_by_type,
            edge_weight_by_type=edge_weight_by_type,
        )
        detached = detach_outputs(final_out)

    pred_log = detached["log_count"].numpy().astype(np.float32)
    pred_count = detached["count"].numpy().astype(np.float32)
    embedding = detached.get("embedding")

    checkpoint_path: str | None = None
    if train_config.save_checkpoints:
        checkpoint = checkpoints_dir / f"{safe_filename(model_name)}.pt"
        save_model_checkpoint(
            str(checkpoint),
            model,
            optimizer=optimizer,
            epoch=best_epoch,
            validation_metric=best_val_mae,
            extra={
                "trial_spec": asdict(spec),
                "train_config": asdict(train_config),
                "feature_columns": feature_columns,
                "relation_names": relation_names,
                "preprocessor": {
                    "train_median": preprocessor.train_median.tolist(),
                    "train_mean": preprocessor.train_mean.tolist(),
                    "train_std": preprocessor.train_std.tolist(),
                },
            },
        )
        checkpoint_path = str(checkpoint)

    embedding_path: str | None = None
    if train_config.save_embeddings == "all" and embedding is not None:
        emb_path = embeddings_dir / f"{safe_filename(model_name)}__node_embeddings.npy"
        np.save(emb_path, embedding.numpy().astype(np.float32))
        embedding_path = str(emb_path)

    metrics: list[dict[str, Any]] = []
    pred_rows: list[pd.DataFrame] = []

    for split_name, mask_np in [
        ("train", train_mask_np),
        ("validation", val_mask_np),
        ("test", test_mask_np),
    ]:
        y_split = bundle.targets[mask_np]
        pred_split = pred_count[mask_np]
        metrics.extend(
            compute_metric_rows(
                y_split,
                pred_split,
                split_name=split_name,
                spec=spec,
                model_name=model_name,
                n_rows=int(mask_np.sum()),
            )
        )

        if split_name in {"validation", "test"}:
            rows = bundle.node_table.loc[mask_np].copy()
            rows["observed_count"] = bundle.targets[mask_np]
            rows["predicted_count"] = pred_count[mask_np]
            rows["predicted_log_count"] = pred_log[mask_np]
            rows["observed_binary"] = bundle.binary_targets[mask_np]
            rows["model_stage"] = MODEL_STAGE
            rows["model_name"] = model_name
            rows["split_scheme"] = spec.split_scheme
            rows["split_name"] = split_name
            rows["feature_regime"] = spec.feature_regime
            rows["edge_regime"] = spec.edge_regime
            rows["edge_mask_regime"] = spec.edge_mask_regime
            rows["seed"] = spec.seed
            pred_rows.append(rows)

    predictions = pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()
    training_curve = pd.DataFrame(training_rows)

    elapsed = time.time() - start_time
    result = TrialResult(
        spec=spec,
        model_name=model_name,
        status="completed",
        best_epoch=int(best_epoch),
        best_validation_mae=float(best_val_mae),
        best_validation_loss=float(best_val_loss),
        elapsed_seconds=float(elapsed),
        n_parameters=int(count_parameters(model)),
        n_edges_used=int(len(edge_table_used)),
        n_relations_used=int(len(relation_names)),
        checkpoint_path=checkpoint_path,
        embedding_path=embedding_path,
    )

    preprocessing_metadata = {
        "feature_regime": spec.feature_regime,
        "split_scheme": spec.split_scheme,
        "n_features": int(X.shape[1]),
        "feature_columns": feature_columns,
        "preprocessor_audit": preprocessor.audit_frame(),
    }

    return result, training_curve, predictions, metrics, preprocessing_metadata, pd.DataFrame([graph_regime_row])


# ---------------------------------------------------------------------------
# Model selection and reports
# ---------------------------------------------------------------------------

def metric_lookup(metrics: pd.DataFrame, model_name: str, split_name: str, metric_name: str) -> float:
    """Fetch one metric value."""

    sub = metrics[
        (metrics["model_name"].astype(str) == model_name)
        & (metrics["split_name"].astype(str) == split_name)
        & (metrics["metric_name"].astype(str) == metric_name)
    ]
    if sub.empty:
        return np.nan
    values = pd.to_numeric(sub["metric_value"], errors="coerce").dropna()
    if values.empty:
        return np.nan
    return float(values.iloc[0])


def build_trial_audit(results: Sequence[TrialResult]) -> pd.DataFrame:
    """Build trial audit table."""

    rows = []
    for r in results:
        row = asdict(r)
        spec = row.pop("spec")
        row.update(spec)
        rows.append(row)
    return pd.DataFrame(rows)


def build_model_selection_audit(metrics: pd.DataFrame, trial_audit: pd.DataFrame) -> pd.DataFrame:
    """Build validation-only model-selection audit."""

    if trial_audit.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for _, trial in trial_audit.iterrows():
        model_name = str(trial["model_name"])
        row = trial.to_dict()
        row["validation_mae"] = metric_lookup(metrics, model_name, "validation", "count__mae")
        row["validation_rmse"] = metric_lookup(metrics, model_name, "validation", "count__rmse")
        row["validation_spearman"] = metric_lookup(metrics, model_name, "validation", "ranking__spearman_corr")
        row["validation_ndcg_at_100"] = metric_lookup(metrics, model_name, "validation", "ranking__ndcg_at_100")
        row["test_mae"] = metric_lookup(metrics, model_name, "test", "count__mae")
        row["test_rmse"] = metric_lookup(metrics, model_name, "test", "count__rmse")
        row["test_spearman"] = metric_lookup(metrics, model_name, "test", "ranking__spearman_corr")
        row["test_ndcg_at_100"] = metric_lookup(metrics, model_name, "test", "ranking__ndcg_at_100")
        rows.append(row)

    audit = pd.DataFrame(rows)
    audit["selected_overall_for_split"] = False
    audit["selected_for_feature_regime"] = False
    audit["selected_for_test_summary"] = False

    valid = audit[audit["status"].astype(str) == "completed"].copy()
    valid["validation_mae_num"] = pd.to_numeric(valid["validation_mae"], errors="coerce")
    valid = valid[valid["validation_mae_num"].notna()].copy()

    for split, group in valid.groupby("split_scheme"):
        best_idx = group.sort_values("validation_mae_num").index[0]
        audit.loc[best_idx, "selected_overall_for_split"] = True
        audit.loc[best_idx, "selected_for_test_summary"] = True

    for (split, feature), group in valid.groupby(["split_scheme", "feature_regime"]):
        best_idx = group.sort_values("validation_mae_num").index[0]
        audit.loc[best_idx, "selected_for_feature_regime"] = True
        audit.loc[best_idx, "selected_for_test_summary"] = True

    return audit.sort_values(["split_scheme", "validation_mae"], na_position="last").reset_index(drop=True)


def compact_metric_table(metrics: pd.DataFrame, selection: pd.DataFrame) -> pd.DataFrame:
    """Metrics for selected test-summary models."""

    if metrics.empty or selection.empty:
        return pd.DataFrame()
    selected = set(
        selection.loc[selection["selected_for_test_summary"].astype(bool), "model_name"].astype(str)
    )
    wanted = [
        "count__mae",
        "count__rmse",
        "count__mean_poisson_deviance",
        "ranking__spearman_corr",
        "ranking__ndcg_at_100",
        "ranking__top_10pct_overlap_rate",
    ]
    out = metrics[
        metrics["model_name"].astype(str).isin(selected)
        & metrics["metric_name"].astype(str).isin(wanted)
    ].copy()
    return out.sort_values(["split_scheme", "model_name", "split_name", "metric_name"]).reset_index(drop=True)


def render_report(
    *,
    generated_at: str,
    graph_dir: Path,
    output_dir: Path,
    train_config: TrainConfig,
    trial_audit: pd.DataFrame,
    model_selection_audit: pd.DataFrame,
    metrics: pd.DataFrame,
    graph_regime_audit: pd.DataFrame,
    feature_preprocessing_audit: pd.DataFrame,
    outputs: Mapping[str, str],
) -> str:
    """Render baseline report."""

    compact = compact_metric_table(metrics, model_selection_audit)

    lines: list[str] = []
    lines.append("# G1 Spatiotemporal Tract GNN Baseline\n")
    lines.append(f"Generated at: `{generated_at}`\n")
    lines.append(f"Graph artifact directory: `{graph_dir}`\n")
    lines.append(f"Output directory: `{output_dir}`\n")

    lines.append("## Purpose\n")
    lines.append(
        "G1 is the first controlled graph-value test after the frozen A0--A3 tabular benchmark layer. "
        "It evaluates whether message passing over tract topology/time improves prediction and ranking of "
        "reported water/drainage 311 burden beyond feature-parity tabular baselines.\n"
    )

    lines.append("## Important interpretation rule\n")
    lines.append(
        "A graph result is meaningful only relative to A3 feature-parity baselines and to no-edge neural controls. "
        "Beating raw SVI or weak naive baselines is not enough after A3.\n"
    )

    lines.append("## Training configuration\n")
    lines.append("```json")
    lines.append(json.dumps(jsonable(asdict(train_config)), indent=2))
    lines.append("```\n")

    lines.append("## Trial summary\n")
    summary_cols = [
        "model_name", "status", "split_scheme", "feature_regime", "edge_regime",
        "edge_mask_regime", "best_epoch", "best_validation_mae", "n_edges_used",
        "n_relations_used", "elapsed_seconds",
    ]
    lines.append(dataframe_to_markdown(trial_audit[[c for c in summary_cols if c in trial_audit.columns]], max_rows=100))
    lines.append("")

    lines.append("## Validation-only model selection\n")
    selection_cols = [
        "model_name", "split_scheme", "feature_regime", "edge_regime", "edge_mask_regime",
        "validation_mae", "validation_spearman", "test_mae", "test_spearman",
        "selected_overall_for_split", "selected_for_feature_regime", "selected_for_test_summary",
    ]
    lines.append(dataframe_to_markdown(model_selection_audit[[c for c in selection_cols if c in model_selection_audit.columns]], max_rows=120))
    lines.append("")

    lines.append("## Compact metrics for selected models\n")
    lines.append(dataframe_to_markdown(compact, max_rows=240))
    lines.append("")

    lines.append("## Graph-regime audit\n")
    lines.append(
        "Edge masks define the message-passing graph used by each trial. "
        "`all_edges` is transductive; `no_test_incident_edges` removes all edges touching test nodes; "
        "`train_train_edges` uses only train-train edges when requested.\n"
    )
    lines.append(dataframe_to_markdown(graph_regime_audit, max_rows=120))
    lines.append("")

    lines.append("## Feature preprocessing audit preview\n")
    lines.append(
        "Feature imputation and scaling are fit on train nodes only for each split/feature-regime combination.\n"
    )
    lines.append(dataframe_to_markdown(feature_preprocessing_audit, max_rows=80))
    lines.append("")

    lines.append("## Output artifacts\n")
    lines.append("| Artifact | Path |")
    lines.append("|---|---|")
    for key, path in outputs.items():
        lines.append(f"| `{key}` | `{path}` |")
    lines.append("")

    lines.append("## Benchmark handoff\n")
    lines.append(
        "Temporal graph results should be compared against A0_3_tract_train_mean and A3_all_forecasting RF. "
        "Spatial-block graph results should be compared against A3_lagged_reporting_forecasting HGB. "
        "The most convincing graph claim would improve both count error and high-burden ranking metrics under "
        "spatial-block evaluation.\n"
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_g1_spatiotemporal_gnn(
    *,
    graph_dir: str | Path = DEFAULT_GRAPH_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    preset: str = "smoke",
    feature_regimes: Sequence[str] | None = None,
    split_schemes: Sequence[str] | None = None,
    edge_regimes: Sequence[str] | None = None,
    edge_mask_regimes: Sequence[str] | None = None,
    seeds: Sequence[int] = (20240610,),
    train_config: TrainConfig = TrainConfig(),
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run G1 trials and write all artifacts."""

    require_runtime_dependencies()
    graph_dir = Path(graph_dir)
    output_dir = Path(output_dir)

    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    ensure_dir(output_dir)

    bundle = load_graph_bundle(graph_dir)

    trial_specs = build_trial_specs(
        preset=preset,
        feature_regimes=feature_regimes,
        split_schemes=split_schemes,
        edge_regimes=edge_regimes,
        edge_mask_regimes=edge_mask_regimes,
        seeds=seeds,
        available_feature_regimes=sorted(bundle.feature_matrix_paths),
        available_split_mask_keys=sorted(bundle.split_masks),
        available_edge_mask_keys=sorted(bundle.edge_masks),
    )
    if not trial_specs:
        raise G1BaselineError("No trials to run after applying preset/custom filters.")

    all_results: list[TrialResult] = []
    all_curves: list[pd.DataFrame] = []
    all_predictions: list[pd.DataFrame] = []
    all_metrics: list[dict[str, Any]] = []
    all_preproc_audits: list[pd.DataFrame] = []
    all_graph_regime_audits: list[pd.DataFrame] = []
    failures: list[TrialResult] = []

    run_start = time.time()
    for i, spec in enumerate(trial_specs, start=1):
        model_name = spec.model_name(train_config)
        print(f"[{i}/{len(trial_specs)}] Training {model_name}", flush=True)
        try:
            result, curve, predictions, metrics, preproc_meta, graph_regime = train_one_trial(
                bundle,
                spec,
                train_config,
                output_dir,
            )
            all_results.append(result)
            all_curves.append(curve)
            all_predictions.append(predictions)
            all_metrics.extend(metrics)
            all_preproc_audits.append(preproc_meta["preprocessor_audit"])
            all_graph_regime_audits.append(graph_regime)
            print(
                f"  completed: best_epoch={result.best_epoch}, "
                f"val_mae={result.best_validation_mae:.4f}, "
                f"edges={result.n_edges_used:,}",
                flush=True,
            )
        except Exception as exc:
            warnings.warn(f"Trial failed: {model_name}: {exc}")
            failure = TrialResult(
                spec=spec,
                model_name=model_name,
                status="failed",
                best_epoch=-1,
                best_validation_mae=float("nan"),
                best_validation_loss=float("nan"),
                elapsed_seconds=0.0,
                n_parameters=0,
                n_edges_used=0,
                n_relations_used=0,
                checkpoint_path=None,
                embedding_path=None,
                failure_reason=repr(exc),
            )
            all_results.append(failure)
            failures.append(failure)

    metrics_df = pd.DataFrame(all_metrics)
    training_curves = pd.concat(all_curves, ignore_index=True) if all_curves else pd.DataFrame()
    predictions_all = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
    trial_audit = build_trial_audit(all_results)
    model_selection = build_model_selection_audit(metrics_df, trial_audit)
    feature_preproc = pd.concat(all_preproc_audits, ignore_index=True) if all_preproc_audits else pd.DataFrame()
    graph_regime_audit = pd.concat(all_graph_regime_audits, ignore_index=True) if all_graph_regime_audits else pd.DataFrame()

    # Write predictions by split name across split schemes.
    pred_validation = predictions_all[predictions_all["split_name"].astype(str) == "validation"].copy() if not predictions_all.empty else pd.DataFrame()
    pred_test = predictions_all[predictions_all["split_name"].astype(str) == "test"].copy() if not predictions_all.empty else pd.DataFrame()

    outputs = {
        "metrics": str(output_dir / "metrics.csv"),
        "predictions_validation": str(output_dir / "predictions_validation.parquet"),
        "predictions_test": str(output_dir / "predictions_test.parquet"),
        "predictions_all_evaluated": str(output_dir / "predictions_all_evaluated.parquet"),
        "training_curves": str(output_dir / "training_curves.csv"),
        "trial_audit": str(output_dir / "trial_audit.csv"),
        "model_selection_audit": str(output_dir / "model_selection_audit.csv"),
        "feature_preprocessing_audit": str(output_dir / "feature_preprocessing_audit.csv"),
        "graph_regime_audit": str(output_dir / "graph_regime_audit.csv"),
        "model_metadata": str(output_dir / "model_metadata.json"),
        "baseline_report": str(output_dir / "baseline_report.md"),
        "checkpoints_dir": str(output_dir / "checkpoints"),
        "embeddings_dir": str(output_dir / "embeddings"),
    }

    metrics_df.to_csv(outputs["metrics"], index=False)
    training_curves.to_csv(outputs["training_curves"], index=False)
    trial_audit.to_csv(outputs["trial_audit"], index=False)
    model_selection.to_csv(outputs["model_selection_audit"], index=False)
    feature_preproc.to_csv(outputs["feature_preprocessing_audit"], index=False)
    graph_regime_audit.to_csv(outputs["graph_regime_audit"], index=False)

    if not pred_validation.empty:
        pred_validation.to_parquet(outputs["predictions_validation"], index=False)
    else:
        pd.DataFrame().to_parquet(outputs["predictions_validation"], index=False)

    if not pred_test.empty:
        pred_test.to_parquet(outputs["predictions_test"], index=False)
    else:
        pd.DataFrame().to_parquet(outputs["predictions_test"], index=False)

    if not predictions_all.empty:
        predictions_all.to_parquet(outputs["predictions_all_evaluated"], index=False)
    else:
        pd.DataFrame().to_parquet(outputs["predictions_all_evaluated"], index=False)

    generated_at = now_utc()
    metadata = {
        "model_stage": MODEL_STAGE,
        "generated_at": generated_at,
        "graph_dir": str(graph_dir),
        "output_dir": str(output_dir),
        "preset": preset,
        "train_config": asdict(train_config),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "device_resolved": str(resolve_device(train_config.device)),
        "pyg_available": available_pyg(),
        "n_nodes": int(len(bundle.node_table)),
        "n_edges": int(len(bundle.edge_table)),
        "feature_regimes_available": sorted(bundle.feature_matrix_paths),
        "split_masks_available": sorted(bundle.split_masks),
        "edge_masks_available": sorted(bundle.edge_masks),
        "trial_count": int(len(trial_specs)),
        "completed_trial_count": int((trial_audit["status"].astype(str) == "completed").sum()) if not trial_audit.empty else 0,
        "failed_trial_count": int(len(failures)),
        "outputs": outputs,
    }
    write_json_file(Path(outputs["model_metadata"]), metadata)

    report = render_report(
        generated_at=generated_at,
        graph_dir=graph_dir,
        output_dir=output_dir,
        train_config=train_config,
        trial_audit=trial_audit,
        model_selection_audit=model_selection,
        metrics=metrics_df,
        graph_regime_audit=graph_regime_audit,
        feature_preprocessing_audit=feature_preproc,
        outputs=outputs,
    )
    write_markdown_file(Path(outputs["baseline_report"]), report)

    return {
        "status": "completed",
        "model_stage": MODEL_STAGE,
        "preset": preset,
        "graph_dir": str(graph_dir),
        "output_dir": str(output_dir),
        "trial_count": len(trial_specs),
        "completed_trial_count": int((trial_audit["status"].astype(str) == "completed").sum()) if not trial_audit.empty else 0,
        "failed_trial_count": len(failures),
        "elapsed_seconds": time.time() - run_start,
        "outputs": outputs,
        "selected_models": (
            model_selection.loc[
                model_selection["selected_for_test_summary"].astype(bool),
                "model_name",
            ].tolist()
            if not model_selection.empty
            else []
        ),
    }


def g1_brief(result: Mapping[str, Any]) -> str:
    """Compact terminal summary."""

    outputs = result.get("outputs", {})
    lines = [
        "G1 spatiotemporal GNN baseline completed.",
        f"Status: {result.get('status')}",
        f"Preset: {result.get('preset')}",
        f"Graph dir: {result.get('graph_dir')}",
        f"Output dir: {result.get('output_dir')}",
        f"Trials: {result.get('completed_trial_count')}/{result.get('trial_count')} completed",
        f"Failures: {result.get('failed_trial_count')}",
        f"Elapsed seconds: {float(result.get('elapsed_seconds', 0.0)):.1f}",
        f"Metrics: {outputs.get('metrics')}",
        f"Model selection audit: {outputs.get('model_selection_audit')}",
        f"Report: {outputs.get('baseline_report')}",
    ]
    selected = result.get("selected_models") or []
    if selected:
        lines.append("Selected models:")
        for model in selected:
            lines.append(f"  {model}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse CLI args."""

    parser = argparse.ArgumentParser(
        description="Train/evaluate G1 spatiotemporal tract GNN baselines."
    )

    parser.add_argument("--graph-dir", default=DEFAULT_GRAPH_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--preset", default="smoke", choices=["smoke", "temporal_core", "spatial_core", "core", "full", "custom"])

    parser.add_argument("--feature-regimes", default=None, help="Comma-separated feature regimes for custom/preset override.")
    parser.add_argument("--split-schemes", default=None, help="Comma-separated split schemes.")
    parser.add_argument("--edge-regimes", default=None, help="Comma-separated edge regimes.")
    parser.add_argument("--edge-mask-regimes", default=None, help="Comma-separated edge mask regimes.")
    parser.add_argument("--seeds", default="20240610", help="Comma-separated random seeds.")

    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--activation", default="relu")
    parser.add_argument("--normalization", default="layernorm", choices=["layernorm", "batchnorm", "none"])
    parser.add_argument("--no-residual", action="store_true")
    parser.add_argument("--backend", default="manual", choices=["manual", "pyg", "auto"])
    parser.add_argument("--relation-combine", default="mean", choices=["mean", "sum"])

    parser.add_argument("--max-epochs", type=int, default=250)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=5.0)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--monitor-metric", default="validation_mae", choices=["validation_mae", "validation_loss"])
    parser.add_argument("--no-checkpoints", action="store_true")
    parser.add_argument("--save-embeddings", default="none", choices=["none", "all"])
    parser.add_argument("--overwrite", action="store_true")

    return parser.parse_args()


def main() -> None:
    """CLI entry point."""

    args = parse_args()

    seeds = tuple(int(s) for s in parse_csv_list(args.seeds, ("20240610",)))
    train_config = TrainConfig(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        activation=args.activation,
        normalization=args.normalization,
        residual=not args.no_residual,
        backend=args.backend,
        relation_combine=args.relation_combine,
        max_epochs=args.max_epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        min_delta=args.min_delta,
        seed=seeds[0] if seeds else 20240610,
        device=args.device,
        monitor_metric=args.monitor_metric,
        save_checkpoints=not args.no_checkpoints,
        save_embeddings=args.save_embeddings,
    )

    result = run_g1_spatiotemporal_gnn(
        graph_dir=args.graph_dir,
        output_dir=args.output_dir,
        preset=args.preset,
        feature_regimes=parse_csv_list(args.feature_regimes, ()) if args.feature_regimes else None,
        split_schemes=parse_csv_list(args.split_schemes, ()) if args.split_schemes else None,
        edge_regimes=parse_csv_list(args.edge_regimes, ()) if args.edge_regimes else None,
        edge_mask_regimes=parse_csv_list(args.edge_mask_regimes, ()) if args.edge_mask_regimes else None,
        seeds=seeds,
        train_config=train_config,
        overwrite=args.overwrite,
    )
    print(g1_brief(result))


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_GRAPH_DIR",
    "DEFAULT_OUTPUT_DIR",
    "G1BaselineError",
    "GraphBundle",
    "TrainConfig",
    "TrialResult",
    "TrialSpec",
    "build_model_selection_audit",
    "build_trial_specs",
    "compute_metric_rows",
    "g1_brief",
    "load_graph_bundle",
    "run_g1_spatiotemporal_gnn",
]
