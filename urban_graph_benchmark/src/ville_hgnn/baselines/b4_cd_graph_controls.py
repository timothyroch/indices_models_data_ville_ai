#!/usr/bin/env python3
"""
B4 graph-control baselines for the Québec CD civil-security / SoVI benchmark.

Purpose:
    Run neural controls using the same leakage-safe node features as B3, while
    changing only the graph topology:

        B4_no_edge_neural    : MLP on node-month features, no message passing
        B4_random_edge_graph : GraphSAGE-style model on placebo random edges
        B4_knn_graph         : GraphSAGE-style model on centroid kNN edges
        B4_real_cd_graph     : GraphSAGE-style model on real CD adjacency edges

Why this matters:
    B3 tests non-graph feature parity.
    B4 tests whether graph topology adds value beyond the same node features
    under controlled neural architectures.

Default inputs:
    urban_graph_benchmark/outputs/qc_cd_civil_security_sovi_v0/datasets/cd_month_panel.parquet
    urban_graph_benchmark/outputs/qc_cd_civil_security_sovi_v0/datasets/cd_graph_nodes.parquet
    urban_graph_benchmark/outputs/qc_cd_civil_security_sovi_v0/datasets/cd_graph_edges_adjacency.parquet
    urban_graph_benchmark/outputs/qc_cd_civil_security_sovi_v0/datasets/cd_graph_edges_knn.parquet
    urban_graph_benchmark/outputs/qc_cd_civil_security_sovi_v0/datasets/cd_graph_edges_random_placebo.parquet

Default outputs:
    urban_graph_benchmark/outputs/qc_cd_civil_security_sovi_v0/baselines/B4_no_edge_neural/
    urban_graph_benchmark/outputs/qc_cd_civil_security_sovi_v0/baselines/B4_random_edge_graph/
    urban_graph_benchmark/outputs/qc_cd_civil_security_sovi_v0/baselines/B4_knn_graph/
    urban_graph_benchmark/outputs/qc_cd_civil_security_sovi_v0/baselines/B4_real_cd_graph/

Implementation notes:
    - Uses PyTorch only, not PyTorch Geometric, to keep the dependency footprint small.
    - Message passing happens between CDs within the same origin month.
    - Training is leakage-safe: scaling/imputation uses train rows only.
    - Early stopping is based on validation MAE.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ville_hgnn.baselines.qc_cd_sovi_common import (
    BASELINES_DIR,
    CD_ID_COL,
    CD_NAME_COL,
    DATASETS_DIR,
    DEFAULT_PANEL_PATH,
    SPLIT_COL,
    ensure_dir,
    evaluate_standard_prediction_frame,
    write_metadata_json,
)

# Reuse B3 feature inference so B4 really is feature-parity with B3.
from ville_hgnn.baselines.b3_cd_tabular_feature_parity import (
    Config as B3FeatureConfig,
    add_seasonality_features,
    infer_feature_columns,
    make_design_matrix,
)


DEFAULT_NODES_PATH = DATASETS_DIR / "cd_graph_nodes.parquet"
DEFAULT_ADJACENCY_EDGES_PATH = DATASETS_DIR / "cd_graph_edges_adjacency.parquet"
DEFAULT_KNN_EDGES_PATH = DATASETS_DIR / "cd_graph_edges_knn.parquet"
DEFAULT_RANDOM_EDGES_PATH = DATASETS_DIR / "cd_graph_edges_random_placebo.parquet"

MODEL_OUTPUT_DIRS = {
    "B4_no_edge_neural": BASELINES_DIR / "B4_no_edge_neural",
    "B4_random_edge_graph": BASELINES_DIR / "B4_random_edge_graph",
    "B4_knn_graph": BASELINES_DIR / "B4_knn_graph",
    "B4_real_cd_graph": BASELINES_DIR / "B4_real_cd_graph",
}

MODEL_TO_EDGE_KIND = {
    "B4_no_edge_neural": "none",
    "B4_random_edge_graph": "random",
    "B4_knn_graph": "knn",
    "B4_real_cd_graph": "adjacency",
}

DEFAULT_MODELS = [
    "B4_no_edge_neural",
    "B4_random_edge_graph",
    "B4_knn_graph",
    "B4_real_cd_graph",
]


@dataclass
class Config:
    """Configuration for B4 graph-control baselines."""

    panel_path: Path = DEFAULT_PANEL_PATH
    nodes_path: Path = DEFAULT_NODES_PATH
    adjacency_edges_path: Path = DEFAULT_ADJACENCY_EDGES_PATH
    knn_edges_path: Path = DEFAULT_KNN_EDGES_PATH
    random_edges_path: Path = DEFAULT_RANDOM_EDGES_PATH

    base_output_dir: Path = BASELINES_DIR
    target_col: str = "target_next_3_months"

    cd_id_col: str = CD_ID_COL
    cd_name_col: str = CD_NAME_COL
    period_month_col: str = "period_month"
    split_col: str = SPLIT_COL

    models: list[str] = field(default_factory=lambda: list(DEFAULT_MODELS))

    # Feature parity flags. Defaults match the B3 baseline.
    include_sovi_features: bool = True
    include_history_features: bool = True
    include_hazard_history_features: bool = True
    include_current_month_counts: bool = True
    include_seasonality: bool = True
    include_year_trend: bool = True
    include_all_other_numeric_features: bool = True

    # Neural architecture.
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.10
    use_layer_norm: bool = True
    output_activation: str = "softplus"  # "softplus", "relu", or "identity"

    # Optimization.
    max_epochs: int = 500
    patience: int = 60
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    loss: str = "mse"  # "mse", "huber", or "poisson_nll"
    huber_delta: float = 1.0

    # Reproducibility/runtime.
    random_seed: int = 42
    device: str = "auto"

    # Count predictions should be nonnegative.
    clip_predictions_at_zero: bool = True
    drop_missing_target: bool = True

    # Edge handling.
    add_self_loops_to_graph_models: bool = False
    normalize_edge_weights: bool = True


def _lazy_import_torch() -> Any:
    """Import torch lazily to keep import-time behavior clean."""
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "B4 graph-control baselines require PyTorch. Install torch in the "
            "active environment or run inside the project .venv."
        ) from exc

    return torch, nn, F


def read_table(path: Path) -> pd.DataFrame:
    """Read a table by suffix."""
    if not path.exists():
        raise FileNotFoundError(f"Input table does not exist: {path}")

    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)

    raise ValueError(f"Unsupported table suffix: {path}")


def write_table_with_csv_copy(df: pd.DataFrame, path: Path) -> dict[str, str]:
    """Write parquet/csv and an inspection CSV copy."""
    path.parent.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}

    if path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=False)
        outputs["parquet"] = str(path)
        csv_path = path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        outputs["csv"] = str(csv_path)
        return outputs

    if path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
        outputs["csv"] = str(path)
        return outputs

    raise ValueError(f"Unsupported output suffix: {path}")


def set_global_seed(seed: int) -> None:
    """Set NumPy/Python/Torch seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch, _, _ = _lazy_import_torch()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> str:
    """Resolve device string."""
    torch, _, _ = _lazy_import_torch()
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def validate_config(config: Config) -> None:
    """Validate config before work."""
    path_fields = [
        "panel_path",
        "nodes_path",
        "adjacency_edges_path",
        "knn_edges_path",
        "random_edges_path",
        "base_output_dir",
    ]
    for field_name in path_fields:
        value = getattr(config, field_name)
        if not isinstance(value, Path):
            setattr(config, field_name, Path(value))

    unknown = sorted(set(config.models) - set(DEFAULT_MODELS))
    if unknown:
        raise ValueError(f"Unknown B4 models: {unknown}. Allowed models: {DEFAULT_MODELS}")

    if not config.models:
        raise ValueError("At least one B4 model must be requested.")

    if config.hidden_dim < 1:
        raise ValueError("hidden_dim must be positive.")
    if config.num_layers not in {1, 2}:
        raise ValueError("This implementation supports num_layers = 1 or 2.")
    if config.max_epochs < 1:
        raise ValueError("max_epochs must be positive.")
    if config.patience < 1:
        raise ValueError("patience must be positive.")

    if config.loss not in {"mse", "huber", "poisson_nll"}:
        raise ValueError("loss must be one of: mse, huber, poisson_nll.")

    if config.output_activation not in {"softplus", "relu", "identity"}:
        raise ValueError("output_activation must be one of: softplus, relu, identity.")


def validate_panel_columns(panel: pd.DataFrame, config: Config) -> None:
    """Validate required panel columns."""
    required = [
        config.cd_id_col,
        config.target_col,
        config.split_col,
        config.period_month_col,
    ]
    missing = [c for c in required if c not in panel.columns]
    if missing:
        raise KeyError(
            f"Panel missing required columns: {missing}. "
            f"Available columns: {list(panel.columns)}"
        )


def build_b3_feature_config(config: Config) -> B3FeatureConfig:
    """Build a B3 config so B4 feature inference matches B3."""
    return B3FeatureConfig(
        panel_path=config.panel_path,
        output_dir=config.base_output_dir / "_unused_b3_feature_config",
        target_col=config.target_col,
        cd_id_col=config.cd_id_col,
        cd_name_col=config.cd_name_col,
        period_month_col=config.period_month_col,
        split_col=config.split_col,
        include_sovi_features=config.include_sovi_features,
        include_history_features=config.include_history_features,
        include_hazard_history_features=config.include_hazard_history_features,
        include_current_month_counts=config.include_current_month_counts,
        include_seasonality=config.include_seasonality,
        include_year_trend=config.include_year_trend,
        include_all_other_numeric_features=config.include_all_other_numeric_features,
    )


def load_panel_and_features(config: Config) -> tuple[pd.DataFrame, pd.DataFrame, list[str], pd.DataFrame]:
    """Load panel and infer B3-compatible features."""
    panel = read_table(config.panel_path)
    validate_panel_columns(panel, config)

    panel = panel.copy()
    panel[CD_ID_COL] = panel[config.cd_id_col].astype("string")
    panel[SPLIT_COL] = panel[config.split_col].astype("string")
    panel["target"] = pd.to_numeric(panel[config.target_col], errors="coerce")
    panel["period_month_str"] = panel[config.period_month_col].astype("string")

    b3_cfg = build_b3_feature_config(config)
    panel = add_seasonality_features(panel, b3_cfg)

    sort_cols = [CD_ID_COL, "period_month_str"]
    panel = panel.sort_values(sort_cols).reset_index(drop=True)

    feature_cols, feature_audit = infer_feature_columns(panel, b3_cfg)
    X_raw = make_design_matrix(panel, feature_cols)

    return panel, X_raw, feature_cols, feature_audit


def train_val_test_masks(panel: pd.DataFrame) -> dict[str, np.ndarray]:
    """Build boolean masks for train/val/test/all rows."""
    split = panel[SPLIT_COL].astype("string")
    target = pd.to_numeric(panel["target"], errors="coerce")
    has_target = target.notna()

    train = split.eq("train") & has_target
    val = split.isin(["val", "validation"]) & has_target
    test = split.eq("test") & has_target

    return {
        "train": train.to_numpy(dtype=bool),
        "val": val.to_numpy(dtype=bool),
        "validation": val.to_numpy(dtype=bool),
        "test": test.to_numpy(dtype=bool),
        "all_target": has_target.to_numpy(dtype=bool),
    }


@dataclass
class FeatureScaler:
    """Train-only median imputation and standardization."""

    medians: np.ndarray
    means: np.ndarray
    stds: np.ndarray

    @classmethod
    def fit(cls, X: pd.DataFrame, train_mask: np.ndarray) -> "FeatureScaler":
        X_train = X.loc[train_mask].to_numpy(dtype=float)

        medians = np.nanmedian(X_train, axis=0)
        medians = np.where(np.isfinite(medians), medians, 0.0)

        X_imp = np.where(np.isnan(X_train), medians[None, :], X_train)
        means = np.mean(X_imp, axis=0)
        stds = np.std(X_imp, axis=0)
        stds = np.where(stds < 1e-8, 1.0, stds)

        return cls(medians=medians.astype(float), means=means.astype(float), stds=stds.astype(float))

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        arr = X.to_numpy(dtype=float)
        arr = np.where(np.isnan(arr), self.medians[None, :], arr)
        arr = (arr - self.means[None, :]) / self.stds[None, :]
        return arr.astype(np.float32)


def load_nodes(config: Config) -> pd.DataFrame:
    """Load graph nodes."""
    nodes = read_table(config.nodes_path)
    if CD_ID_COL not in nodes.columns:
        raise KeyError(f"Node table missing {CD_ID_COL}. Available columns: {list(nodes.columns)}")
    if "node_index" not in nodes.columns:
        nodes = nodes.sort_values(CD_ID_COL).reset_index(drop=True)
        nodes.insert(0, "node_index", np.arange(len(nodes), dtype=int))

    nodes = nodes.copy()
    nodes[CD_ID_COL] = nodes[CD_ID_COL].astype("string")
    nodes["node_index"] = pd.to_numeric(nodes["node_index"], errors="raise").astype(int)
    return nodes.sort_values("node_index").reset_index(drop=True)


def load_edges_for_model(model_name: str, config: Config) -> pd.DataFrame | None:
    """Load the graph edge table for one B4 model."""
    kind = MODEL_TO_EDGE_KIND[model_name]
    if kind == "none":
        return None
    if kind == "random":
        return read_table(config.random_edges_path)
    if kind == "knn":
        return read_table(config.knn_edges_path)
    if kind == "adjacency":
        return read_table(config.adjacency_edges_path)
    raise ValueError(f"Unknown edge kind: {kind}")


def infer_edge_node_columns(edges: pd.DataFrame) -> tuple[str, str]:
    """Infer source/target node-index columns from an edge file."""
    source_candidates = ["source_node_index", "src_node_index", "source", "src"]
    target_candidates = ["target_node_index", "dst_node_index", "target", "dst"]

    src_col = next((c for c in source_candidates if c in edges.columns), None)
    dst_col = next((c for c in target_candidates if c in edges.columns), None)

    if src_col is None or dst_col is None:
        raise KeyError(
            "Could not infer source/target node index columns from edge file. "
            f"Available columns: {list(edges.columns)}"
        )

    return src_col, dst_col


def build_row_edge_index(
    panel: pd.DataFrame,
    nodes: pd.DataFrame,
    edges: pd.DataFrame | None,
    *,
    config: Config,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """
    Expand static CD graph edges into row-level same-month edges.

    Each static edge src_cd -> dst_cd becomes one row edge for every origin month
    in which both src and dst CD rows exist.
    """
    if edges is None:
        return None, {
            "edge_kind": "none",
            "static_directed_edges": 0,
            "row_directed_edges": 0,
        }

    src_col, dst_col = infer_edge_node_columns(edges)

    node_index_to_cd = dict(zip(nodes["node_index"].astype(int), nodes[CD_ID_COL].astype(str)))

    # Row lookup: (period_month, cd_id) -> row index.
    row_lookup: dict[tuple[str, str], int] = {}
    for row_idx, (period, cd_id) in enumerate(zip(panel["period_month_str"], panel[CD_ID_COL])):
        row_lookup[(str(period), str(cd_id))] = int(row_idx)

    months = sorted(panel["period_month_str"].astype(str).unique().tolist())

    edge_pairs: list[tuple[int, int]] = []
    skipped_static_edges = 0

    for _, edge in edges.iterrows():
        try:
            src_node = int(edge[src_col])
            dst_node = int(edge[dst_col])
        except Exception:
            skipped_static_edges += 1
            continue

        src_cd = node_index_to_cd.get(src_node)
        dst_cd = node_index_to_cd.get(dst_node)
        if src_cd is None or dst_cd is None:
            skipped_static_edges += 1
            continue

        for period in months:
            src_row = row_lookup.get((period, src_cd))
            dst_row = row_lookup.get((period, dst_cd))
            if src_row is None or dst_row is None:
                continue
            if src_row == dst_row:
                continue
            edge_pairs.append((src_row, dst_row))

    if config.add_self_loops_to_graph_models:
        for row_idx in range(len(panel)):
            edge_pairs.append((row_idx, row_idx))

    if not edge_pairs:
        raise RuntimeError("Graph model requested but no row-level edges were constructed.")

    edge_index = np.array(edge_pairs, dtype=np.int64).T

    audit = {
        "edge_kind": "graph",
        "static_directed_edges": int(len(edges)),
        "row_directed_edges": int(edge_index.shape[1]),
        "skipped_static_edges": int(skipped_static_edges),
        "months_expanded": int(len(months)),
        "add_self_loops": bool(config.add_self_loops_to_graph_models),
    }
    return edge_index, audit


def graph_degree_audit(edge_index: np.ndarray | None, n_rows: int) -> dict[str, Any]:
    """Audit row-level graph degrees."""
    if edge_index is None:
        return {
            "row_edge_count": 0,
            "mean_in_degree": 0.0,
            "min_in_degree": 0,
            "max_in_degree": 0,
            "row_isolates": int(n_rows),
        }

    dst = edge_index[1]
    deg = np.bincount(dst, minlength=n_rows).astype(float)

    return {
        "row_edge_count": int(edge_index.shape[1]),
        "mean_in_degree": float(deg.mean()),
        "min_in_degree": int(deg.min()),
        "max_in_degree": int(deg.max()),
        "row_isolates": int((deg == 0).sum()),
    }


def make_loss_fn(config: Config) -> Any:
    """Create a torch loss function."""
    torch, _, F = _lazy_import_torch()

    if config.loss == "mse":
        return lambda pred, y: F.mse_loss(pred, y)

    if config.loss == "huber":
        return lambda pred, y: F.huber_loss(pred, y, delta=float(config.huber_delta))

    if config.loss == "poisson_nll":
        # pred is expected to be nonnegative rate after softplus/relu.
        return lambda pred, y: F.poisson_nll_loss(
            pred.clamp_min(1e-8),
            y,
            log_input=False,
            full=True,
            reduction="mean",
        )

    raise ValueError(f"Unknown loss: {config.loss}")


class MLPRegressorBase:
    """
    Lightweight wrapper class marker.

    Real neural classes are created after torch import inside build_model.
    """


def build_model(
    *,
    model_name: str,
    input_dim: int,
    config: Config,
) -> Any:
    """Build no-edge MLP or GraphSAGE-style model."""
    torch, nn, F = _lazy_import_torch()

    class NoEdgeMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            layers = []
            dim = input_dim
            for layer_idx in range(config.num_layers):
                layers.append(nn.Linear(dim, config.hidden_dim))
                if config.use_layer_norm:
                    layers.append(nn.LayerNorm(config.hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(config.dropout))
                dim = config.hidden_dim
            layers.append(nn.Linear(dim, 1))
            self.net = nn.Sequential(*layers)

        def forward(self, x: Any, edge_index: Any | None = None) -> Any:
            out = self.net(x).squeeze(-1)
            if config.output_activation == "softplus":
                return F.softplus(out)
            if config.output_activation == "relu":
                return F.relu(out)
            return out

    class GraphSAGERegressor(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.self1 = nn.Linear(input_dim, config.hidden_dim)
            self.neigh1 = nn.Linear(input_dim, config.hidden_dim)

            if config.num_layers == 2:
                self.self2 = nn.Linear(config.hidden_dim, config.hidden_dim)
                self.neigh2 = nn.Linear(config.hidden_dim, config.hidden_dim)
            else:
                self.self2 = None
                self.neigh2 = None

            self.norm1 = nn.LayerNorm(config.hidden_dim) if config.use_layer_norm else nn.Identity()
            self.norm2 = nn.LayerNorm(config.hidden_dim) if config.use_layer_norm else nn.Identity()
            self.dropout = nn.Dropout(config.dropout)
            self.out = nn.Linear(config.hidden_dim, 1)

        def aggregate(self, h: Any, edge_index: Any) -> Any:
            src = edge_index[0]
            dst = edge_index[1]

            agg = torch.zeros_like(h)
            agg.index_add_(0, dst, h[src])

            deg = torch.zeros(h.shape[0], device=h.device, dtype=h.dtype)
            one = torch.ones(dst.shape[0], device=h.device, dtype=h.dtype)
            deg.index_add_(0, dst, one)
            deg = deg.clamp_min(1.0).unsqueeze(-1)

            return agg / deg

        def forward(self, x: Any, edge_index: Any | None = None) -> Any:
            if edge_index is None:
                raise ValueError("GraphSAGERegressor requires edge_index.")

            neigh_x = self.aggregate(x, edge_index)
            h = self.self1(x) + self.neigh1(neigh_x)
            h = self.norm1(h)
            h = F.relu(h)
            h = self.dropout(h)

            if self.self2 is not None and self.neigh2 is not None:
                neigh_h = self.aggregate(h, edge_index)
                h = self.self2(h) + self.neigh2(neigh_h)
                h = self.norm2(h)
                h = F.relu(h)
                h = self.dropout(h)

            out = self.out(h).squeeze(-1)
            if config.output_activation == "softplus":
                return F.softplus(out)
            if config.output_activation == "relu":
                return F.relu(out)
            return out

    if model_name == "B4_no_edge_neural":
        return NoEdgeMLP()

    return GraphSAGERegressor()


def mae_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not mask.any():
        return float("inf")
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))


def train_one_model(
    *,
    model_name: str,
    panel: pd.DataFrame,
    X_scaled: np.ndarray,
    edge_index_np: np.ndarray | None,
    config: Config,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    """Train one B4 model and return predictions, training history, run metadata."""
    torch, _, _ = _lazy_import_torch()

    set_global_seed(config.random_seed)
    device = resolve_device(config.device)

    y_np = pd.to_numeric(panel["target"], errors="coerce").to_numpy(dtype=np.float32)
    masks = train_val_test_masks(panel)

    x = torch.tensor(X_scaled, dtype=torch.float32, device=device)
    y = torch.tensor(np.nan_to_num(y_np, nan=0.0), dtype=torch.float32, device=device)

    train_idx = torch.tensor(np.where(masks["train"])[0], dtype=torch.long, device=device)
    val_idx = torch.tensor(np.where(masks["val"])[0], dtype=torch.long, device=device)

    if len(train_idx) == 0:
        raise RuntimeError("No train rows with nonmissing targets.")
    if len(val_idx) == 0:
        raise RuntimeError("No validation rows with nonmissing targets.")

    if edge_index_np is None:
        edge_index = None
    else:
        edge_index = torch.tensor(edge_index_np, dtype=torch.long, device=device)

    model = build_model(model_name=model_name, input_dim=X_scaled.shape[1], config=config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )
    loss_fn = make_loss_fn(config)

    best_state = None
    best_epoch = -1
    best_val_mae = float("inf")
    patience_left = int(config.patience)

    history_rows: list[dict[str, Any]] = []

    for epoch in range(1, int(config.max_epochs) + 1):
        model.train()
        optimizer.zero_grad()

        pred = model(x, edge_index)
        loss = loss_fn(pred[train_idx], y[train_idx])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            pred_eval = model(x, edge_index).detach().cpu().numpy()

        train_mae = mae_np(y_np[masks["train"]], pred_eval[masks["train"]])
        val_mae = mae_np(y_np[masks["val"]], pred_eval[masks["val"]])
        test_mae_observed = mae_np(y_np[masks["test"]], pred_eval[masks["test"]])

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": float(loss.detach().cpu().item()),
                "train_mae": train_mae,
                "val_mae": val_mae,
                "test_mae_observed_not_for_selection": test_mae_observed,
                "learning_rate": float(config.learning_rate),
                "weight_decay": float(config.weight_decay),
            }
        )

        improved = val_mae < best_val_mae - 1e-8
        if improved:
            best_val_mae = val_mae
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = int(config.patience)
        else:
            patience_left -= 1

        if patience_left <= 0:
            break

    if best_state is None:
        raise RuntimeError(f"Training failed to produce a best state for {model_name}.")

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        final_pred = model(x, edge_index).detach().cpu().numpy().astype(float)

    if config.clip_predictions_at_zero:
        final_pred = np.clip(final_pred, 0.0, None)

    history = pd.DataFrame(history_rows)

    train_meta = {
        "device": device,
        "best_epoch": int(best_epoch),
        "best_val_mae": float(best_val_mae),
        "epochs_ran": int(len(history)),
        "stopped_by_patience": bool(len(history) < int(config.max_epochs)),
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
    }

    return final_pred, history, train_meta


def build_prediction_frame(
    panel: pd.DataFrame,
    *,
    prediction: np.ndarray,
    model_name: str,
    graph_kind: str,
    feature_count: int,
    config: Config,
) -> pd.DataFrame:
    """Build standard prediction output."""
    out = pd.DataFrame(index=panel.index)

    out[CD_ID_COL] = panel[CD_ID_COL].astype("string")
    if config.cd_name_col in panel.columns:
        out[CD_NAME_COL] = panel[config.cd_name_col].astype("string")
    elif CD_NAME_COL in panel.columns:
        out[CD_NAME_COL] = panel[CD_NAME_COL].astype("string")

    out["period_month"] = panel[config.period_month_col].astype("string")
    if "year" in panel.columns:
        out["year"] = panel["year"]
    if "month" in panel.columns:
        out["month"] = panel["month"]

    out[SPLIT_COL] = panel[SPLIT_COL].astype("string")
    out["model_name"] = model_name
    out["graph_kind"] = graph_kind
    out["target_col"] = config.target_col
    out["feature_count"] = int(feature_count)
    out["target"] = pd.to_numeric(panel["target"], errors="coerce")
    out["prediction"] = pd.to_numeric(prediction, errors="coerce")

    return out.reset_index(drop=True)


def write_split_predictions(predictions: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    """Write train/validation/test split prediction files."""
    outputs: dict[str, str] = {}

    for split_name in ["train", "val", "validation", "test"]:
        sub = predictions[predictions[SPLIT_COL].astype("string").eq(split_name)]
        if sub.empty:
            continue

        split_label = "validation" if split_name == "val" else split_name
        written = write_table_with_csv_copy(sub, output_dir / f"predictions_{split_label}.parquet")
        for kind, path in written.items():
            outputs[f"predictions_{split_label}_{kind}"] = path

    return outputs


def summarize_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Compact prediction audit by split."""
    rows: list[dict[str, Any]] = []

    for split_name, sub in predictions.groupby(SPLIT_COL, dropna=False):
        target = pd.to_numeric(sub["target"], errors="coerce")
        pred = pd.to_numeric(sub["prediction"], errors="coerce")

        rows.append(
            {
                "split": split_name,
                "n": int(len(sub)),
                "target_nonmissing": int(target.notna().sum()),
                "prediction_nonmissing": int(pred.notna().sum()),
                "target_mean": float(target.mean()) if target.notna().any() else np.nan,
                "prediction_mean": float(pred.mean()) if pred.notna().any() else np.nan,
                "target_sum": float(target.sum()) if target.notna().any() else 0.0,
                "prediction_sum": float(pred.sum()) if pred.notna().any() else 0.0,
            }
        )

    return pd.DataFrame(rows)


def run_single_b4_model(
    *,
    model_name: str,
    panel: pd.DataFrame,
    X_scaled: np.ndarray,
    feature_cols: list[str],
    feature_audit: pd.DataFrame,
    nodes: pd.DataFrame,
    config: Config,
) -> dict[str, Any]:
    """Run one of the four B4 baselines."""
    output_dir = ensure_dir(MODEL_OUTPUT_DIRS[model_name])

    edge_df = load_edges_for_model(model_name, config)
    edge_index_np, edge_audit = build_row_edge_index(panel, nodes, edge_df, config=config)

    preds, history, train_meta = train_one_model(
        model_name=model_name,
        panel=panel,
        X_scaled=X_scaled,
        edge_index_np=edge_index_np,
        config=config,
    )

    graph_kind = MODEL_TO_EDGE_KIND[model_name]
    predictions = build_prediction_frame(
        panel,
        prediction=preds,
        model_name=model_name,
        graph_kind=graph_kind,
        feature_count=len(feature_cols),
        config=config,
    )

    if config.drop_missing_target:
        predictions = predictions[predictions["target"].notna()].copy()

    metrics = evaluate_standard_prediction_frame(
        predictions,
        target_col="target",
        prediction_col="prediction",
        split_col=SPLIT_COL,
        id_col=CD_ID_COL,
    )

    summary = summarize_predictions(predictions)

    prediction_paths = write_table_with_csv_copy(predictions, output_dir / "predictions.parquet")
    split_outputs = write_split_predictions(predictions, output_dir)

    metrics_path = output_dir / "metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    history_path = output_dir / "training_history.csv"
    history.to_csv(history_path, index=False)

    summary_path = output_dir / "prediction_summary.csv"
    summary.to_csv(summary_path, index=False)

    feature_path = output_dir / "feature_columns.csv"
    feature_audit.to_csv(feature_path, index=False)

    row_edge_audit = graph_degree_audit(edge_index_np, len(panel))
    graph_audit = {
        **edge_audit,
        **row_edge_audit,
    }

    metadata = {
        "baseline_family": model_name,
        "module": "ville_hgnn.baselines.b4_cd_graph_controls",
        "purpose": (
            "Neural graph-control baseline with B3 feature parity. "
            "Only graph topology changes across B4 models."
        ),
        "model_name": model_name,
        "graph_kind": graph_kind,
        "config": {
            **asdict(config),
            "panel_path": str(config.panel_path),
            "nodes_path": str(config.nodes_path),
            "adjacency_edges_path": str(config.adjacency_edges_path),
            "knn_edges_path": str(config.knn_edges_path),
            "random_edges_path": str(config.random_edges_path),
            "base_output_dir": str(config.base_output_dir),
        },
        "inputs": {
            "panel_path": str(config.panel_path),
            "nodes_path": str(config.nodes_path),
            "edge_path": None if edge_df is None else edge_path_for_model(model_name, config),
            "panel_rows": int(len(panel)),
            "node_count": int(len(nodes)),
        },
        "features": {
            "feature_count": int(len(feature_cols)),
            "feature_columns": list(feature_cols),
            "feature_category_summary": (
                feature_audit[feature_audit["use_as_feature"]]
                .groupby("category", dropna=False)
                .size()
                .rename("feature_count")
                .reset_index()
                .sort_values("feature_count", ascending=False)
                .to_dict(orient="records")
            ),
        },
        "graph_audit": graph_audit,
        "training": train_meta,
        "outputs": {
            **{f"predictions_{k}": v for k, v in prediction_paths.items()},
            **split_outputs,
            "metrics_csv": str(metrics_path),
            "training_history_csv": str(history_path),
            "prediction_summary_csv": str(summary_path),
            "feature_columns_csv": str(feature_path),
        },
    }

    # Convenience selected metric fields.
    val_metrics = metrics[metrics[SPLIT_COL].astype("string").isin(["val", "validation"])]
    if not val_metrics.empty:
        row = val_metrics.iloc[0].to_dict()
        metadata["validation_metrics"] = row

    metadata_path = write_metadata_json(metadata, output_dir / "metadata.json")
    metadata["outputs"]["metadata_json"] = str(metadata_path)

    return metadata


def edge_path_for_model(model_name: str, config: Config) -> str | None:
    """Return the edge path associated with a model."""
    kind = MODEL_TO_EDGE_KIND[model_name]
    if kind == "none":
        return None
    if kind == "random":
        return str(config.random_edges_path)
    if kind == "knn":
        return str(config.knn_edges_path)
    if kind == "adjacency":
        return str(config.adjacency_edges_path)
    return None


def write_family_comparison(results: dict[str, dict[str, Any]], config: Config) -> Path:
    """Write a compact B4 model comparison table."""
    rows = []

    for model_name, meta in results.items():
        metrics_path = Path(meta["outputs"]["metrics_csv"])
        if not metrics_path.exists():
            continue

        metrics = pd.read_csv(metrics_path)
        for _, row in metrics.iterrows():
            rows.append(
                {
                    "model_name": model_name,
                    "graph_kind": meta.get("graph_kind"),
                    "split": row.get(SPLIT_COL),
                    "mae": row.get("mae"),
                    "rmse": row.get("rmse"),
                    "mean_poisson_deviance": row.get("mean_poisson_deviance"),
                    "spearman": row.get("spearman"),
                    "ndcg_at_25": row.get("ndcg_at_25"),
                    "top10_overlap": row.get("top10_overlap"),
                    "feature_count": meta.get("features", {}).get("feature_count"),
                    "row_edge_count": meta.get("graph_audit", {}).get("row_edge_count"),
                    "best_epoch": meta.get("training", {}).get("best_epoch"),
                    "best_val_mae": meta.get("training", {}).get("best_val_mae"),
                }
            )

    comparison = pd.DataFrame(rows)
    out_dir = ensure_dir(config.base_output_dir / "B4_graph_control_comparison")
    path = out_dir / "b4_graph_control_comparison.csv"
    comparison.to_csv(path, index=False)

    return path


def run_b4_cd_graph_controls(config: Config) -> dict[str, Any]:
    """
    Run all requested B4 graph-control models.
    """
    validate_config(config)
    set_global_seed(config.random_seed)

    panel, X_raw, feature_cols, feature_audit = load_panel_and_features(config)
    nodes = load_nodes(config)

    masks = train_val_test_masks(panel)
    scaler = FeatureScaler.fit(X_raw, masks["train"])
    X_scaled = scaler.transform(X_raw)

    results: dict[str, dict[str, Any]] = {}

    for model_name in config.models:
        results[model_name] = run_single_b4_model(
            model_name=model_name,
            panel=panel,
            X_scaled=X_scaled,
            feature_cols=feature_cols,
            feature_audit=feature_audit,
            nodes=nodes,
            config=config,
        )

    comparison_path = write_family_comparison(results, config)

    family_metadata = {
        "baseline_family": "B4_graph_controls",
        "module": "ville_hgnn.baselines.b4_cd_graph_controls",
        "purpose": "Run no-edge, random-edge, kNN-edge, and real-adjacency neural graph controls.",
        "models": list(config.models),
        "target_col": config.target_col,
        "feature_count": int(len(feature_cols)),
        "outputs": {
            "comparison_csv": str(comparison_path),
            **{
                f"{model_name}_metadata_json": meta["outputs"]["metadata_json"]
                for model_name, meta in results.items()
            },
        },
        "model_summaries": {
            model_name: {
                "graph_kind": meta.get("graph_kind"),
                "best_epoch": meta.get("training", {}).get("best_epoch"),
                "best_val_mae": meta.get("training", {}).get("best_val_mae"),
                "row_edge_count": meta.get("graph_audit", {}).get("row_edge_count"),
                "feature_count": meta.get("features", {}).get("feature_count"),
            }
            for model_name, meta in results.items()
        },
    }

    family_meta_path = write_metadata_json(
        family_metadata,
        config.base_output_dir / "B4_graph_control_comparison" / "metadata.json",
    )
    family_metadata["outputs"]["metadata_json"] = str(family_meta_path)

    return family_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run B4 no-edge, random-edge, kNN, and real-adjacency graph-control "
            "baselines for the Québec CD civil-security / SoVI benchmark."
        )
    )

    parser.add_argument("--panel-path", type=Path, default=DEFAULT_PANEL_PATH)
    parser.add_argument("--nodes-path", type=Path, default=DEFAULT_NODES_PATH)
    parser.add_argument("--adjacency-edges-path", type=Path, default=DEFAULT_ADJACENCY_EDGES_PATH)
    parser.add_argument("--knn-edges-path", type=Path, default=DEFAULT_KNN_EDGES_PATH)
    parser.add_argument("--random-edges-path", type=Path, default=DEFAULT_RANDOM_EDGES_PATH)
    parser.add_argument("--base-output-dir", type=Path, default=BASELINES_DIR)

    parser.add_argument("--target-col", default="target_next_3_months")
    parser.add_argument("--cd-id-col", default=CD_ID_COL)
    parser.add_argument("--cd-name-col", default=CD_NAME_COL)
    parser.add_argument("--period-month-col", default="period_month")
    parser.add_argument("--split-col", default=SPLIT_COL)

    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODELS),
        choices=list(DEFAULT_MODELS),
        help="B4 models to run.",
    )

    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2, choices=[1, 2])
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--no-layer-norm", action="store_true")
    parser.add_argument("--output-activation", default="softplus", choices=["softplus", "relu", "identity"])

    parser.add_argument("--max-epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--loss", default="mse", choices=["mse", "huber", "poisson_nll"])
    parser.add_argument("--huber-delta", type=float, default=1.0)

    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--device", default="auto")

    parser.add_argument("--keep-missing-targets", action="store_true")
    parser.add_argument("--no-clip-predictions", action="store_true")
    parser.add_argument("--add-self-loops", action="store_true")

    feature_group = parser.add_argument_group("feature parity / ablation controls")
    feature_group.add_argument("--no-sovi-features", action="store_true")
    feature_group.add_argument("--no-history-features", action="store_true")
    feature_group.add_argument("--no-hazard-history-features", action="store_true")
    feature_group.add_argument("--no-current-month-counts", action="store_true")
    feature_group.add_argument("--no-seasonality", action="store_true")
    feature_group.add_argument("--no-year-trend", action="store_true")
    feature_group.add_argument("--no-other-numeric-features", action="store_true")

    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> Config:
    return Config(
        panel_path=args.panel_path,
        nodes_path=args.nodes_path,
        adjacency_edges_path=args.adjacency_edges_path,
        knn_edges_path=args.knn_edges_path,
        random_edges_path=args.random_edges_path,
        base_output_dir=args.base_output_dir,
        target_col=args.target_col,
        cd_id_col=args.cd_id_col,
        cd_name_col=args.cd_name_col,
        period_month_col=args.period_month_col,
        split_col=args.split_col,
        models=list(args.models),
        include_sovi_features=not args.no_sovi_features,
        include_history_features=not args.no_history_features,
        include_hazard_history_features=not args.no_hazard_history_features,
        include_current_month_counts=not args.no_current_month_counts,
        include_seasonality=not args.no_seasonality,
        include_year_trend=not args.no_year_trend,
        include_all_other_numeric_features=not args.no_other_numeric_features,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        use_layer_norm=not args.no_layer_norm,
        output_activation=args.output_activation,
        max_epochs=args.max_epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        loss=args.loss,
        huber_delta=args.huber_delta,
        random_seed=args.random_seed,
        device=args.device,
        drop_missing_target=not args.keep_missing_targets,
        clip_predictions_at_zero=not args.no_clip_predictions,
        add_self_loops_to_graph_models=args.add_self_loops,
    )


def main() -> None:
    args = parse_args()
    config = config_from_args(args)

    metadata = run_b4_cd_graph_controls(config)

    print("B4 graph-control baselines completed.")
    print(f"Panel: {config.panel_path}")
    print(f"Nodes: {config.nodes_path}")
    print(f"Base output directory: {config.base_output_dir}")
    print(f"Target column: {config.target_col}")
    print(f"Feature count: {metadata.get('feature_count')}")
    print("Models:")
    for model_name, summary in metadata.get("model_summaries", {}).items():
        print(
            "  "
            f"{model_name}: "
            f"graph={summary.get('graph_kind')}, "
            f"best_epoch={summary.get('best_epoch')}, "
            f"best_val_mae={summary.get('best_val_mae')}, "
            f"row_edges={summary.get('row_edge_count')}"
        )

    print("Outputs:")
    for key, value in metadata["outputs"].items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
