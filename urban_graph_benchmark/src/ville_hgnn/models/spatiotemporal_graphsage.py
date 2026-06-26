#!/usr/bin/env python3
"""
Model definitions for the G1 typed spatiotemporal tract graph.

This module intentionally contains *model code only*. It does not load graph
artifacts, choose train/validation/test masks, run ablations, write metrics, or
make paper figures. Those responsibilities belong to the G1 training/evaluation
runner.

The design target is the graph artifact produced by:

    ville_hgnn.graphs.build_tract_month_graph

where:

    node = census tract x month
    target = water_drainage_count
    edge direction = source node sends a message to target node

The main model is a small, auditable GraphSAGE-style architecture that supports:

- no-edge MLP control;
- homogeneous edge_index message passing;
- typed relation-specific message passing over edge_index_by_type;
- temporal-only, spatial-only, spatial+temporal, and placebo edge regimes;
- optional edge weights in the manual backend;
- log-count prediction aligned with A3 tabular baselines.

Why this module has a manual backend
------------------------------------
PyTorch Geometric is excellent when available, and this module can use
``torch_geometric.nn.SAGEConv`` for homogeneous unweighted message passing.
However, the graph artifacts also include edge weights and typed edge
regimes. To keep the G1 benchmark runnable and auditable without requiring
PyG-specific installation details, the default backend is a small manual
mean-aggregation GraphSAGE implementation using standard PyTorch
``index_add_``.

The training runner can still choose ``backend="pyg"`` if PyG is installed.

Expected prediction contract
----------------------------
Models return a dictionary with at least:

    log_count: Tensor[n_nodes]
    count:     Tensor[n_nodes]
    embedding: Tensor[n_nodes, hidden_dim]

The training loss should normally be:

    MSE(log_count, log1p(y_count))

and benchmark metrics should be computed after:

    count = expm1(log_count).clip(min=0)

This mirrors the log-count A3 baseline convention and keeps model comparison
clean.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

try:  # Optional dependency; the manual backend remains the default.
    from torch_geometric.nn import SAGEConv as PyGSAGEConv  # type: ignore
except Exception:  # pragma: no cover
    PyGSAGEConv = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Constants and edge-regime helpers
# ---------------------------------------------------------------------------

SPATIAL_EDGE_TOKENS = ("spatial", "knn", "adjacency")
TEMPORAL_EDGE_TOKENS = ("temporal", "lag_1", "lag_12")
PLACEBO_EDGE_TOKENS = ("placebo", "random")
DEFAULT_EDGE_REGIMES = (
    "no_edges",
    "temporal_only",
    "spatial_only",
    "spatial_temporal",
    "spatial_temporal_no_placebo",
    "random_spatial_placebo",
    "all_edges",
)


class G1ModelError(RuntimeError):
    """Raised when the G1 model receives an invalid configuration/input."""


@dataclass(frozen=True)
class GraphSAGEConfig:
    """Configuration for :class:`SpatioTemporalGraphSAGE`.

    Parameters
    ----------
    input_dim:
        Number of input node features.
    hidden_dim:
        Hidden representation dimension.
    output_dim:
        Number of output channels. For the benchmark, this should normally be 1.
    num_layers:
        Number of graph message-passing layers after input projection.
    dropout:
        Dropout probability applied after activations and in the prediction head.
    activation:
        Activation name. Supported: relu, gelu, elu, tanh, silu, leaky_relu.
    normalization:
        Hidden normalization. Supported: layernorm, batchnorm, none.
    residual:
        Whether to use residual connections between hidden graph layers.
    backend:
        Message-passing backend. Supported: manual, pyg, auto.
        ``manual`` supports edge weights and typed relations.
        ``pyg`` currently supports homogeneous unweighted SAGEConv here.
    aggregation:
        Neighbor aggregation. The manual backend currently implements weighted
        mean aggregation. The name is still recorded for metadata clarity.
    relation_combine:
        How to combine relation-specific aggregate tensors. Supported: mean, sum.
    relation_names:
        Optional relation names expected for typed message passing. If empty,
        the model can still accept any relation dictionary at forward time, but
        explicit names are preferred for checkpoint stability.
    count_min:
        Minimum value for inverse-transformed count predictions.
    """

    input_dim: int
    hidden_dim: int = 128
    output_dim: int = 1
    num_layers: int = 2
    dropout: float = 0.15
    activation: str = "relu"
    normalization: str = "layernorm"
    residual: bool = True
    backend: str = "manual"
    aggregation: str = "mean"
    relation_combine: str = "mean"
    relation_names: tuple[str, ...] = field(default_factory=tuple)
    count_min: float = 0.0

    def validate(self) -> None:
        """Validate config values."""

        if self.input_dim <= 0:
            raise G1ModelError("input_dim must be positive.")
        if self.hidden_dim <= 0:
            raise G1ModelError("hidden_dim must be positive.")
        if self.output_dim <= 0:
            raise G1ModelError("output_dim must be positive.")
        if self.num_layers < 0:
            raise G1ModelError("num_layers must be non-negative.")
        if not (0.0 <= self.dropout < 1.0):
            raise G1ModelError("dropout must satisfy 0 <= dropout < 1.")
        if self.backend not in {"manual", "pyg", "auto"}:
            raise G1ModelError("backend must be one of: manual, pyg, auto.")
        if self.normalization not in {"layernorm", "batchnorm", "none"}:
            raise G1ModelError("normalization must be one of: layernorm, batchnorm, none.")
        if self.relation_combine not in {"mean", "sum"}:
            raise G1ModelError("relation_combine must be one of: mean, sum.")
        if self.aggregation != "mean":
            raise G1ModelError("Only mean aggregation is implemented in this module.")


@dataclass(frozen=True)
class MLPConfig:
    """Configuration for the no-edge MLP baseline."""

    input_dim: int
    hidden_dim: int = 128
    output_dim: int = 1
    num_hidden_layers: int = 2
    dropout: float = 0.15
    activation: str = "relu"
    normalization: str = "layernorm"
    count_min: float = 0.0

    def validate(self) -> None:
        """Validate config values."""

        if self.input_dim <= 0:
            raise G1ModelError("input_dim must be positive.")
        if self.hidden_dim <= 0:
            raise G1ModelError("hidden_dim must be positive.")
        if self.output_dim <= 0:
            raise G1ModelError("output_dim must be positive.")
        if self.num_hidden_layers < 0:
            raise G1ModelError("num_hidden_layers must be non-negative.")
        if not (0.0 <= self.dropout < 1.0):
            raise G1ModelError("dropout must satisfy 0 <= dropout < 1.")
        if self.normalization not in {"layernorm", "batchnorm", "none"}:
            raise G1ModelError("normalization must be one of: layernorm, batchnorm, none.")


def available_pyg() -> bool:
    """Return whether PyTorch Geometric SAGEConv is importable."""

    return PyGSAGEConv is not None


def normalize_relation_name(name: str) -> str:
    """Make relation names safe for ModuleDict keys."""

    out = str(name).replace(".", "_").replace("-", "_").replace("/", "_")
    out = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in out)
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_") or "relation"


def relation_kind(name: str) -> str:
    """Classify a relation name into temporal/spatial/placebo/other."""

    lower = str(name).lower()
    if any(tok in lower for tok in PLACEBO_EDGE_TOKENS):
        return "placebo"
    if any(tok in lower for tok in TEMPORAL_EDGE_TOKENS):
        return "temporal"
    if any(tok in lower for tok in SPATIAL_EDGE_TOKENS):
        return "spatial"
    return "other"


def select_relations_for_regime(
    relation_names: Iterable[str],
    edge_regime: str,
) -> list[str]:
    """Select relation names for a named graph ablation regime.

    This helper keeps edge-regime naming consistent between the future training
    script and paper reports. It does not inspect edge masks; it only filters
    relation/edge types.

    Parameters
    ----------
    relation_names:
        Names such as ``temporal_self_lag_1``, ``spatial_knn_same_month``,
        or ``spatial_knn_same_month_random_placebo``.
    edge_regime:
        One of ``no_edges``, ``temporal_only``, ``spatial_only``,
        ``spatial_temporal``, ``spatial_temporal_no_placebo``,
        ``random_spatial_placebo``, or ``all_edges``.
    """

    names = list(relation_names)
    if edge_regime not in DEFAULT_EDGE_REGIMES:
        raise G1ModelError(
            f"Unknown edge_regime={edge_regime!r}. Supported: {sorted(DEFAULT_EDGE_REGIMES)}"
        )

    if edge_regime == "no_edges":
        return []
    if edge_regime == "all_edges":
        return names

    selected: list[str] = []
    for name in names:
        kind = relation_kind(name)
        is_placebo = kind == "placebo"
        is_spatial = kind == "spatial"
        is_temporal = kind == "temporal"

        if edge_regime == "temporal_only" and is_temporal:
            selected.append(name)
        elif edge_regime == "spatial_only" and is_spatial:
            selected.append(name)
        elif edge_regime == "spatial_temporal" and (is_spatial or is_temporal):
            selected.append(name)
        elif edge_regime == "spatial_temporal_no_placebo" and (is_spatial or is_temporal) and not is_placebo:
            selected.append(name)
        elif edge_regime == "random_spatial_placebo" and (is_placebo or is_temporal):
            selected.append(name)

    return selected


# ---------------------------------------------------------------------------
# Numeric / tensor validation helpers
# ---------------------------------------------------------------------------

def require_floating_node_features(x: Tensor) -> None:
    """Validate node feature tensor."""

    if not isinstance(x, Tensor):
        raise G1ModelError("x must be a torch.Tensor.")
    if x.ndim != 2:
        raise G1ModelError(f"x must have shape [n_nodes, n_features], got {tuple(x.shape)}.")
    if not x.is_floating_point():
        raise G1ModelError("x must be a floating point tensor.")
    if x.shape[0] <= 0 or x.shape[1] <= 0:
        raise G1ModelError(f"x must be non-empty, got {tuple(x.shape)}.")


def validate_edge_index(edge_index: Tensor, n_nodes: int, name: str = "edge_index") -> None:
    """Validate a PyG-style edge_index tensor."""

    if not isinstance(edge_index, Tensor):
        raise G1ModelError(f"{name} must be a torch.Tensor.")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise G1ModelError(f"{name} must have shape [2, n_edges], got {tuple(edge_index.shape)}.")
    if edge_index.dtype not in (torch.long, torch.int64):
        raise G1ModelError(f"{name} must have dtype torch.long/int64.")
    if edge_index.numel() == 0:
        return
    if int(edge_index.min().item()) < 0:
        raise G1ModelError(f"{name} contains negative node ids.")
    if int(edge_index.max().item()) >= n_nodes:
        raise G1ModelError(
            f"{name} contains node id {int(edge_index.max().item())}, but n_nodes={n_nodes}."
        )


def validate_edge_weight(edge_weight: Tensor | None, n_edges: int, name: str = "edge_weight") -> None:
    """Validate an optional edge-weight vector."""

    if edge_weight is None:
        return
    if not isinstance(edge_weight, Tensor):
        raise G1ModelError(f"{name} must be a torch.Tensor or None.")
    if edge_weight.ndim != 1:
        raise G1ModelError(f"{name} must have shape [n_edges], got {tuple(edge_weight.shape)}.")
    if edge_weight.shape[0] != n_edges:
        raise G1ModelError(f"{name} length {edge_weight.shape[0]} does not match n_edges={n_edges}.")
    if not edge_weight.is_floating_point():
        raise G1ModelError(f"{name} must be floating point.")
    if not torch.isfinite(edge_weight).all():
        raise G1ModelError(f"{name} contains non-finite values.")


def to_device_like(t: Tensor | None, reference: Tensor) -> Tensor | None:
    """Move optional tensor to reference device."""

    if t is None:
        return None
    return t.to(device=reference.device)


def assert_same_device(*tensors: Tensor | None) -> None:
    """Raise if tensors are on different devices."""

    devices = {t.device for t in tensors if t is not None}
    if len(devices) > 1:
        raise G1ModelError(f"Expected tensors on same device, found {devices}.")


def masked_tensor(values: Tensor, mask: Tensor | None) -> Tensor:
    """Return values restricted by a boolean mask."""

    if mask is None:
        return values
    if mask.dtype != torch.bool:
        raise G1ModelError("mask must be boolean.")
    if mask.ndim != 1 or mask.shape[0] != values.shape[0]:
        raise G1ModelError(
            f"mask must have shape [{values.shape[0]}], got {tuple(mask.shape)}."
        )
    return values[mask]


# ---------------------------------------------------------------------------
# Target transforms and losses
# ---------------------------------------------------------------------------

def log1p_count_target(y_count: Tensor) -> Tensor:
    """Transform count targets with log1p after non-negative clipping."""

    if not y_count.is_floating_point():
        y_count = y_count.float()
    return torch.log1p(torch.clamp(y_count, min=0.0))


def invert_log_count_prediction(log_count: Tensor, count_min: float = 0.0) -> Tensor:
    """Invert log-count predictions to count space."""

    count = torch.expm1(log_count)
    return torch.clamp(count, min=float(count_min))


def masked_log_count_mse_loss(
    pred_log_count: Tensor,
    y_count: Tensor,
    mask: Tensor | None = None,
    reduction: str = "mean",
) -> Tensor:
    """MSE loss on log1p count target, optionally masked."""

    if pred_log_count.ndim > 1 and pred_log_count.shape[-1] == 1:
        pred_log_count = pred_log_count.squeeze(-1)
    if y_count.ndim > 1 and y_count.shape[-1] == 1:
        y_count = y_count.squeeze(-1)
    if pred_log_count.shape != y_count.shape:
        raise G1ModelError(
            f"pred_log_count and y_count shapes differ: {tuple(pred_log_count.shape)} vs {tuple(y_count.shape)}."
        )

    pred = masked_tensor(pred_log_count, mask)
    target = masked_tensor(log1p_count_target(y_count), mask)

    if pred.numel() == 0:
        raise G1ModelError("Loss mask selects zero nodes.")

    return F.mse_loss(pred, target, reduction=reduction)


def masked_mae_in_count_space(
    pred_log_count: Tensor,
    y_count: Tensor,
    mask: Tensor | None = None,
    count_min: float = 0.0,
) -> Tensor:
    """MAE in count space, useful for monitoring but not as primary loss."""

    pred_count = invert_log_count_prediction(pred_log_count, count_min=count_min)
    if pred_count.ndim > 1 and pred_count.shape[-1] == 1:
        pred_count = pred_count.squeeze(-1)
    if y_count.ndim > 1 and y_count.shape[-1] == 1:
        y_count = y_count.squeeze(-1)
    pred = masked_tensor(pred_count, mask)
    target = masked_tensor(y_count.float(), mask)
    if pred.numel() == 0:
        raise G1ModelError("MAE mask selects zero nodes.")
    return torch.mean(torch.abs(pred - target))


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def make_activation(name: str) -> nn.Module:
    """Create activation module."""

    key = str(name).lower()
    if key == "relu":
        return nn.ReLU()
    if key == "gelu":
        return nn.GELU()
    if key == "elu":
        return nn.ELU()
    if key == "tanh":
        return nn.Tanh()
    if key == "silu" or key == "swish":
        return nn.SiLU()
    if key == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.1)
    raise G1ModelError(f"Unsupported activation: {name!r}")


def make_norm(kind: str, dim: int) -> nn.Module:
    """Create normalization layer."""

    key = str(kind).lower()
    if key == "layernorm":
        return nn.LayerNorm(dim)
    if key == "batchnorm":
        return nn.BatchNorm1d(dim)
    if key == "none":
        return nn.Identity()
    raise G1ModelError(f"Unsupported normalization: {kind!r}")


def reset_parameters(module: nn.Module) -> None:
    """Default initialization for linear layers."""

    for sub in module.modules():
        if isinstance(sub, nn.Linear):
            nn.init.xavier_uniform_(sub.weight)
            if sub.bias is not None:
                nn.init.zeros_(sub.bias)


class PredictionHead(nn.Module):
    """Small prediction head from hidden embeddings to log-count output."""

    def __init__(
        self,
        hidden_dim: int,
        output_dim: int = 1,
        dropout: float = 0.15,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            make_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        reset_parameters(self)

    def forward(self, h: Tensor) -> Tensor:
        """Return raw log-count prediction."""

        return self.net(h)


class NoEdgesMLP(nn.Module):
    """No-edge neural control for G1.

    This model sees exactly the same node features as the GNN, but no graph
    edges. It is the neural architecture control for claims about message
    passing. If G1 only beats A3 but not this MLP, the result is not really a
    graph-structure result.
    """

    def __init__(self, config: MLPConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config

        layers: list[nn.Module] = []
        in_dim = config.input_dim
        for _ in range(config.num_hidden_layers):
            layers.append(nn.Linear(in_dim, config.hidden_dim))
            layers.append(make_norm(config.normalization, config.hidden_dim))
            layers.append(make_activation(config.activation))
            layers.append(nn.Dropout(config.dropout))
            in_dim = config.hidden_dim

        self.encoder = nn.Sequential(*layers) if layers else nn.Identity()
        self.input_projection = (
            nn.Linear(config.input_dim, config.hidden_dim)
            if config.num_hidden_layers == 0
            else nn.Identity()
        )
        self.head = PredictionHead(
            hidden_dim=config.hidden_dim,
            output_dim=config.output_dim,
            dropout=config.dropout,
            activation=config.activation,
        )
        reset_parameters(self)

    def forward(
        self,
        x: Tensor,
        *,
        return_embeddings: bool = True,
        **_: Any,
    ) -> dict[str, Tensor]:
        """Predict log-count/count for all nodes."""

        require_floating_node_features(x)
        if self.config.num_hidden_layers == 0:
            h = self.input_projection(x)
        else:
            h = self.encoder(x)

        log_count = self.head(h)
        if log_count.shape[-1] == 1:
            log_count_flat = log_count.squeeze(-1)
        else:
            log_count_flat = log_count

        out = {
            "log_count": log_count_flat,
            "count": invert_log_count_prediction(log_count_flat, self.config.count_min),
        }
        if return_embeddings:
            out["embedding"] = h
        return out

    def metadata(self) -> dict[str, Any]:
        """Return serializable model metadata."""

        return {
            "model_class": self.__class__.__name__,
            "config": asdict(self.config),
            "uses_edges": False,
            "prediction_target": "log1p_count",
        }


class ManualWeightedMeanGraphSAGELayer(nn.Module):
    """Homogeneous manual weighted-mean GraphSAGE layer.

    For each directed edge ``source -> target`` this layer sends a transformed
    message from source to target. The target representation combines a learned
    root/self transformation with a weighted mean of incoming neighbor messages.

    This layer uses standard PyTorch operations only:

        index_add_ over target node ids

    so it does not require PyTorch Geometric or torch_scatter.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.root_linear = nn.Linear(in_dim, out_dim, bias=bias)
        self.neighbor_linear = nn.Linear(in_dim, out_dim, bias=False)
        reset_parameters(self)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor | None = None,
    ) -> Tensor:
        """Run one homogeneous weighted mean-aggregation layer."""

        require_floating_node_features(x)
        validate_edge_index(edge_index, n_nodes=x.shape[0])
        if edge_weight is not None:
            edge_weight = edge_weight.to(device=x.device, dtype=x.dtype)
        validate_edge_weight(edge_weight, n_edges=edge_index.shape[1])

        root = self.root_linear(x)
        if edge_index.numel() == 0:
            return root

        src = edge_index[0]
        dst = edge_index[1]
        messages = self.neighbor_linear(x[src])

        if edge_weight is None:
            weights = torch.ones(edge_index.shape[1], device=x.device, dtype=x.dtype)
        else:
            weights = torch.clamp(edge_weight, min=0.0)

        messages = messages * weights.unsqueeze(-1)
        agg = torch.zeros_like(root)
        agg.index_add_(0, dst, messages)

        denom = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        denom.index_add_(0, dst, weights)
        denom = torch.clamp(denom, min=1.0).unsqueeze(-1)

        return root + agg / denom


class ManualTypedWeightedMeanGraphSAGELayer(nn.Module):
    """Typed-edge weighted mean GraphSAGE layer.

    A single root transformation is combined with one transformed aggregate per
    relation type. This avoids multiplying the self/root contribution when using
    many edge types.

    Relation aggregates are combined by mean or sum. Mean is the safer default
    because it makes the hidden scale less sensitive to the number of active
    relation types in an ablation.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        relation_names: Sequence[str],
        *,
        relation_combine: str = "mean",
        bias: bool = True,
    ) -> None:
        super().__init__()
        if relation_combine not in {"mean", "sum"}:
            raise G1ModelError("relation_combine must be one of: mean, sum.")
        self.relation_combine = relation_combine
        self.relation_names = tuple(str(r) for r in relation_names)
        self._relation_key_to_name = {
            normalize_relation_name(name): str(name)
            for name in self.relation_names
        }

        self.root_linear = nn.Linear(in_dim, out_dim, bias=bias)
        self.relation_linears = nn.ModuleDict(
            {
                normalize_relation_name(name): nn.Linear(in_dim, out_dim, bias=False)
                for name in self.relation_names
            }
        )
        reset_parameters(self)

    def _aggregate_one_relation(
        self,
        x: Tensor,
        edge_index: Tensor,
        relation_key: str,
        edge_weight: Tensor | None = None,
    ) -> Tensor:
        """Aggregate one relation into target nodes."""

        validate_edge_index(edge_index, n_nodes=x.shape[0], name=f"edge_index[{relation_key}]")
        if edge_weight is not None:
            edge_weight = edge_weight.to(device=x.device, dtype=x.dtype)
        validate_edge_weight(edge_weight, edge_index.shape[1], name=f"edge_weight[{relation_key}]")

        out_dim = self.root_linear.out_features
        agg = torch.zeros((x.shape[0], out_dim), device=x.device, dtype=x.dtype)
        if edge_index.numel() == 0:
            return agg

        src = edge_index[0]
        dst = edge_index[1]
        messages = self.relation_linears[relation_key](x[src])

        if edge_weight is None:
            weights = torch.ones(edge_index.shape[1], device=x.device, dtype=x.dtype)
        else:
            weights = torch.clamp(edge_weight, min=0.0)

        messages = messages * weights.unsqueeze(-1)
        agg.index_add_(0, dst, messages)

        denom = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        denom.index_add_(0, dst, weights)
        denom = torch.clamp(denom, min=1.0).unsqueeze(-1)
        return agg / denom

    def forward(
        self,
        x: Tensor,
        edge_index_by_type: Mapping[str, Tensor],
        edge_weight_by_type: Mapping[str, Tensor] | None = None,
        active_relations: Sequence[str] | None = None,
    ) -> Tensor:
        """Run one typed weighted mean-aggregation layer."""

        require_floating_node_features(x)
        root = self.root_linear(x)

        if not edge_index_by_type:
            return root

        active = list(active_relations) if active_relations is not None else list(edge_index_by_type.keys())
        aggregates: list[Tensor] = []

        for relation_name in active:
            relation_key = normalize_relation_name(relation_name)
            if relation_key not in self.relation_linears:
                # The training runner may provide extra edge types that are not
                # active for this model. Ignore them explicitly rather than fail.
                continue
            if relation_name not in edge_index_by_type:
                continue

            edge_index = edge_index_by_type[relation_name]
            edge_weight = None
            if edge_weight_by_type is not None:
                edge_weight = edge_weight_by_type.get(relation_name)
            aggregates.append(self._aggregate_one_relation(x, edge_index, relation_key, edge_weight))

        if not aggregates:
            return root

        stacked = torch.stack(aggregates, dim=0)
        if self.relation_combine == "sum":
            return root + stacked.sum(dim=0)
        return root + stacked.mean(dim=0)


class PyGHomogeneousGraphSAGELayer(nn.Module):
    """Thin wrapper around PyG SAGEConv for homogeneous edge_index use.

    This wrapper intentionally ignores edge weights because PyG SAGEConv support
    for weighted edges varies by version. Use the manual backend when edge
    weights are methodologically important.
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        if PyGSAGEConv is None:
            raise G1ModelError(
                "backend='pyg' requested, but torch_geometric.nn.SAGEConv is not available."
            )
        self.conv = PyGSAGEConv(in_dim, out_dim)
        self._warned_weight_ignored = False

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor | None = None,
    ) -> Tensor:
        """Forward pass using PyG SAGEConv."""

        require_floating_node_features(x)
        validate_edge_index(edge_index, n_nodes=x.shape[0])
        if edge_weight is not None and not self._warned_weight_ignored:
            warnings.warn(
                "PyG SAGEConv backend in this module ignores edge_weight. "
                "Use backend='manual' for weighted edges.",
                RuntimeWarning,
                stacklevel=2,
            )
            self._warned_weight_ignored = True
        return self.conv(x, edge_index)


# ---------------------------------------------------------------------------
# Main GraphSAGE model
# ---------------------------------------------------------------------------

class SpatioTemporalGraphSAGE(nn.Module):
    """Small GraphSAGE-style model for the G1 tract-month graph.

    The model supports two input modes:

    1. homogeneous mode:
        forward(x, edge_index=..., edge_weight=...)

    2. typed mode:
        forward(x, edge_index_by_type={...}, edge_weight_by_type={...})

    If ``edge_index_by_type`` is provided, typed manual relation-specific
    message passing is used. If only ``edge_index`` is provided, a homogeneous
    layer stack is used. The future training script can create either view from
    the graph artifacts.
    """

    def __init__(self, config: GraphSAGEConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config

        backend = config.backend
        if backend == "auto":
            backend = "pyg" if PyGSAGEConv is not None else "manual"
        self.backend = backend

        if self.backend == "pyg" and PyGSAGEConv is None:
            raise G1ModelError("backend='pyg' requested, but torch_geometric is not available.")
        if self.backend == "pyg" and config.relation_names:
            warnings.warn(
                "backend='pyg' is used only for homogeneous edge_index in this module. "
                "Typed edge_index_by_type forward calls will use manual typed layers.",
                RuntimeWarning,
            )

        self.input_projection = nn.Linear(config.input_dim, config.hidden_dim)
        self.input_norm = make_norm(config.normalization, config.hidden_dim)
        self.activation = make_activation(config.activation)
        self.dropout = nn.Dropout(config.dropout)

        self.relation_names = tuple(config.relation_names)
        self.homogeneous_layers = nn.ModuleList()
        self.typed_layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(config.num_layers):
            if self.backend == "pyg":
                self.homogeneous_layers.append(
                    PyGHomogeneousGraphSAGELayer(config.hidden_dim, config.hidden_dim)
                )
            else:
                self.homogeneous_layers.append(
                    ManualWeightedMeanGraphSAGELayer(config.hidden_dim, config.hidden_dim)
                )

            self.typed_layers.append(
                ManualTypedWeightedMeanGraphSAGELayer(
                    config.hidden_dim,
                    config.hidden_dim,
                    relation_names=self.relation_names,
                    relation_combine=config.relation_combine,
                )
            )
            self.norms.append(make_norm(config.normalization, config.hidden_dim))

        self.head = PredictionHead(
            hidden_dim=config.hidden_dim,
            output_dim=config.output_dim,
            dropout=config.dropout,
            activation=config.activation,
        )
        reset_parameters(self)

    def _forward_homogeneous(
        self,
        h: Tensor,
        edge_index: Tensor | None,
        edge_weight: Tensor | None,
    ) -> Tensor:
        """Run homogeneous GraphSAGE layers."""

        if self.config.num_layers == 0:
            return h
        if edge_index is None:
            # No-edge path through graph model: layers are skipped. The training
            # script should normally use NoEdgesMLP for the official no-edge
            # control, but this fallback makes ablations easy to run.
            return h

        edge_index = edge_index.to(device=h.device)
        edge_weight = to_device_like(edge_weight, h)

        for layer, norm in zip(self.homogeneous_layers, self.norms):
            residual = h
            h_new = layer(h, edge_index, edge_weight)
            h_new = norm(h_new)
            h_new = self.activation(h_new)
            h_new = self.dropout(h_new)
            if self.config.residual and h_new.shape == residual.shape:
                h = h_new + residual
            else:
                h = h_new
        return h

    def _forward_typed(
        self,
        h: Tensor,
        edge_index_by_type: Mapping[str, Tensor],
        edge_weight_by_type: Mapping[str, Tensor] | None,
        active_relations: Sequence[str] | None,
    ) -> Tensor:
        """Run typed relation-specific GraphSAGE layers."""

        if self.config.num_layers == 0:
            return h
        if not edge_index_by_type:
            return h

        # Move tensors once. Keep relation names unchanged for dictionary lookup.
        edge_index_by_type = {
            str(k): v.to(device=h.device)
            for k, v in edge_index_by_type.items()
        }
        if edge_weight_by_type is not None:
            edge_weight_by_type = {
                str(k): v.to(device=h.device, dtype=h.dtype)
                for k, v in edge_weight_by_type.items()
            }

        for layer, norm in zip(self.typed_layers, self.norms):
            residual = h
            h_new = layer(
                h,
                edge_index_by_type=edge_index_by_type,
                edge_weight_by_type=edge_weight_by_type,
                active_relations=active_relations,
            )
            h_new = norm(h_new)
            h_new = self.activation(h_new)
            h_new = self.dropout(h_new)
            if self.config.residual and h_new.shape == residual.shape:
                h = h_new + residual
            else:
                h = h_new
        return h

    def forward(
        self,
        x: Tensor,
        *,
        edge_index: Tensor | None = None,
        edge_weight: Tensor | None = None,
        edge_index_by_type: Mapping[str, Tensor] | None = None,
        edge_weight_by_type: Mapping[str, Tensor] | None = None,
        active_relations: Sequence[str] | None = None,
        edge_regime: str | None = None,
        return_embeddings: bool = True,
    ) -> dict[str, Tensor]:
        """Predict log-count/count for all nodes.

        Parameters
        ----------
        x:
            Node feature matrix ``[n_nodes, n_features]``.
        edge_index:
            Homogeneous edge index ``[2, n_edges]``.
        edge_weight:
            Optional homogeneous edge weights ``[n_edges]``.
        edge_index_by_type:
            Mapping from relation name to edge index. Takes precedence over
            homogeneous ``edge_index``.
        edge_weight_by_type:
            Optional mapping from relation name to edge weights.
        active_relations:
            Optional explicit relation subset.
        edge_regime:
            Optional named relation filter. If provided with typed edges, it is
            applied to available relation names unless ``active_relations`` is
            already provided.
        return_embeddings:
            Whether to include hidden embeddings in the output dictionary.
        """

        require_floating_node_features(x)
        h = self.input_projection(x)
        h = self.input_norm(h)
        h = self.activation(h)
        h = self.dropout(h)

        if edge_index_by_type is not None:
            relation_names = list(edge_index_by_type.keys())
            if active_relations is None and edge_regime is not None:
                active_relations = select_relations_for_regime(relation_names, edge_regime)
            h = self._forward_typed(
                h,
                edge_index_by_type=edge_index_by_type,
                edge_weight_by_type=edge_weight_by_type,
                active_relations=active_relations,
            )
        else:
            h = self._forward_homogeneous(h, edge_index=edge_index, edge_weight=edge_weight)

        log_count = self.head(h)
        if log_count.shape[-1] == 1:
            log_count_flat = log_count.squeeze(-1)
        else:
            log_count_flat = log_count

        out = {
            "log_count": log_count_flat,
            "count": invert_log_count_prediction(log_count_flat, self.config.count_min),
        }
        if return_embeddings:
            out["embedding"] = h
        return out

    def metadata(self) -> dict[str, Any]:
        """Return serializable model metadata."""

        return {
            "model_class": self.__class__.__name__,
            "config": asdict(self.config),
            "backend_resolved": self.backend,
            "uses_edges": True,
            "supports_typed_edges": True,
            "supports_edge_weights": self.backend == "manual",
            "pyg_available": available_pyg(),
            "prediction_target": "log1p_count",
        }


# ---------------------------------------------------------------------------
# Model factories and checkpoint helpers
# ---------------------------------------------------------------------------

def build_no_edges_mlp(
    input_dim: int,
    *,
    hidden_dim: int = 128,
    output_dim: int = 1,
    num_hidden_layers: int = 2,
    dropout: float = 0.15,
    activation: str = "relu",
    normalization: str = "layernorm",
    count_min: float = 0.0,
) -> NoEdgesMLP:
    """Factory for the no-edge MLP control."""

    return NoEdgesMLP(
        MLPConfig(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_hidden_layers=num_hidden_layers,
            dropout=dropout,
            activation=activation,
            normalization=normalization,
            count_min=count_min,
        )
    )


def build_spatiotemporal_graphsage(
    input_dim: int,
    relation_names: Sequence[str] | None = None,
    *,
    hidden_dim: int = 128,
    output_dim: int = 1,
    num_layers: int = 2,
    dropout: float = 0.15,
    activation: str = "relu",
    normalization: str = "layernorm",
    residual: bool = True,
    backend: str = "manual",
    relation_combine: str = "mean",
    count_min: float = 0.0,
) -> SpatioTemporalGraphSAGE:
    """Factory for the G1 GraphSAGE model."""

    return SpatioTemporalGraphSAGE(
        GraphSAGEConfig(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation,
            normalization=normalization,
            residual=residual,
            backend=backend,
            relation_combine=relation_combine,
            relation_names=tuple(str(r) for r in (relation_names or ())),
            count_min=count_min,
        )
    )


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count model parameters."""

    params = model.parameters()
    if trainable_only:
        return int(sum(p.numel() for p in params if p.requires_grad))
    return int(sum(p.numel() for p in params))


def model_summary(model: nn.Module) -> dict[str, Any]:
    """Return small serializable model summary."""

    meta = model.metadata() if hasattr(model, "metadata") else {}
    return {
        "class_name": model.__class__.__name__,
        "n_parameters_trainable": count_parameters(model, trainable_only=True),
        "n_parameters_total": count_parameters(model, trainable_only=False),
        "metadata": meta,
    }


def detach_outputs(outputs: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """Detach output dictionary tensors to CPU."""

    return {k: v.detach().cpu() for k, v in outputs.items()}


def save_model_checkpoint(
    path: str,
    model: nn.Module,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    epoch: int | None = None,
    validation_metric: float | None = None,
    extra: Mapping[str, Any] | None = None,
) -> None:
    """Save a model checkpoint with metadata."""

    payload: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "model_summary": model_summary(model),
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if epoch is not None:
        payload["epoch"] = int(epoch)
    if validation_metric is not None:
        payload["validation_metric"] = float(validation_metric)
    if extra:
        payload["extra"] = dict(extra)
    torch.save(payload, path)


# ---------------------------------------------------------------------------
# Tiny smoke-test helper
# ---------------------------------------------------------------------------

def smoke_test_forward(device: str | torch.device = "cpu") -> dict[str, Any]:
    """Run a small synthetic forward pass to verify the module works.

    This is useful after installation:

        python - <<'PY'
        from ville_hgnn.models.spatiotemporal_graphsage import smoke_test_forward
        print(smoke_test_forward())
        PY
    """

    device = torch.device(device)
    torch.manual_seed(7)

    n_nodes = 10
    input_dim = 5
    x = torch.randn(n_nodes, input_dim, device=device)
    edge_index_by_type = {
        "temporal_self_lag_1": torch.tensor(
            [[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6]],
            dtype=torch.long,
            device=device,
        ),
        "spatial_knn_same_month": torch.tensor(
            [[0, 2, 4, 6, 8], [2, 4, 6, 8, 0]],
            dtype=torch.long,
            device=device,
        ),
    }
    edge_weight_by_type = {
        "temporal_self_lag_1": torch.ones(6, device=device),
        "spatial_knn_same_month": torch.ones(5, device=device) * 0.5,
    }

    model = build_spatiotemporal_graphsage(
        input_dim=input_dim,
        relation_names=list(edge_index_by_type),
        hidden_dim=16,
        num_layers=2,
        dropout=0.0,
        backend="manual",
    ).to(device)

    out = model(
        x,
        edge_index_by_type=edge_index_by_type,
        edge_weight_by_type=edge_weight_by_type,
        edge_regime="spatial_temporal",
    )

    y = torch.poisson(torch.ones(n_nodes, device=device) * 2.0)
    loss = masked_log_count_mse_loss(out["log_count"], y)

    return {
        "status": "ok",
        "torch_version": torch.__version__,
        "pyg_available": available_pyg(),
        "log_count_shape": tuple(out["log_count"].shape),
        "count_shape": tuple(out["count"].shape),
        "embedding_shape": tuple(out["embedding"].shape),
        "loss": float(loss.detach().cpu()),
        "n_parameters": count_parameters(model),
    }


__all__ = [
    "DEFAULT_EDGE_REGIMES",
    "G1ModelError",
    "GraphSAGEConfig",
    "ManualTypedWeightedMeanGraphSAGELayer",
    "ManualWeightedMeanGraphSAGELayer",
    "MLPConfig",
    "NoEdgesMLP",
    "PredictionHead",
    "SpatioTemporalGraphSAGE",
    "available_pyg",
    "build_no_edges_mlp",
    "build_spatiotemporal_graphsage",
    "count_parameters",
    "detach_outputs",
    "invert_log_count_prediction",
    "log1p_count_target",
    "masked_log_count_mse_loss",
    "masked_mae_in_count_space",
    "model_summary",
    "relation_kind",
    "save_model_checkpoint",
    "select_relations_for_regime",
    "smoke_test_forward",
]
