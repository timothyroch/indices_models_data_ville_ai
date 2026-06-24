"""
Research-grade descriptive diagnostics for one functional message-passing layer.

Target path:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            functional_message_passing/
                layer/
                    diagnostics.py

This module assembles tensor-free diagnostics across the node-level portion of
one complete functional message-passing layer:

    edge messages
        -> target-node aggregation
        -> update-branch dropout
        -> optional additive residual
        -> optional post-residual layer normalization
        -> updated node state

When a full layer trace is retained, the report may also embed the existing
message-builder diagnostic report for the edge-level stage chain. The edge
report is reused rather than reimplemented.

Interpretation
-------------------------
The diagnostics describe:

- incoming-edge coverage;
- isolated nodes;
- target-node aggregate scale;
- dropout and residual state transitions;
- post-residual normalization behavior;
- graph-batch slices;
- regularization scalars;
- exact object lineage and architecture provenance;
- bounded numerical alerts.

They do not claim:

- causal importance;
- explanation faithfulness;
- calibrated uncertainty;
- counterfactual effects;
- mechanistic identifiability;
- relation necessity;
- intervention validity.

A large norm, a sparse update, a gate value, an attention value, or a strong
state change is therefore reported only as an observed model quantity.

Trace-aware reporting
---------------------
``none``
    The internal output contains no optional trace. Node-level stage summaries
    remain available because aggregation, residual update, and normalization
    are mandatory internal outputs.

``node``
    The retained trace contains the exact node-level stages but no edge-level
    message-builder objects.

``full``
    The retained trace contains the complete exact edge- and node-level chain.
    An existing message-builder report may be embedded when explicitly enabled
    in the diagnostics configuration.

The diagnostic configuration is separate from the numerical layer
architecture. Enabling graph slices, full edge reports, or different alert
thresholds must not change model computation or numerical architecture
fingerprints.

Serialization
-------------
Reports contain only Python scalars, strings, lists, dictionaries, and ``None``.
No tensor, module, source object, or graph object is retained. Reports can be
serialized with strict JSON settings using ``allow_nan=False``.

Performance
-----------
Diagnostics are explicit research/debugging operations rather than implicit
forward-pass behavior. Tensor values are detached only when converted to Python
statistics. Model tensors, gradients, ordering, dtype, and device are never
modified.

This module is parameter-free and buffer-free.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from typing import Any, Final, Mapping

import torch
from torch import nn

from ..message_builders.diagnostics import (
    DEFAULT_MESSAGE_BUILDER_DIAGNOSTIC_THRESHOLDS,
    MessageBuilderDiagnosticThresholds,
    build_public_edge_message_diagnostic_report,
)
from ..schemas import (
    AggregationOutput,
    FunctionalMessagePassingLayerOutput,
)
from .normalization import (
    LAYER_NORMALIZER_SCHEMA_VERSION,
    layer_normalization_diagnostic_summary,
    validate_layer_normalization_output,
)
from .residual_update import (
    LAYER_RESIDUAL_UPDATER_SCHEMA_VERSION,
    layer_residual_update_diagnostic_summary,
    validate_layer_residual_update_output,
)
from .schemas import (
    LAYER_COMPUTATION_OUTPUT_SCHEMA_VERSION,
    LAYER_NORMALIZATION_LAYER_NORM,
    LAYER_TRACE_FULL,
    LAYER_TRACE_NODE,
    LAYER_TRACE_NONE,
    FunctionalMessagePassingLayerTrace,
    LayerComputationOutput,
    validate_layer_stage_chain,
    validate_public_layer_output,
)


# =============================================================================
# Public identity
# =============================================================================


LAYER_DIAGNOSTICS_SCHEMA_VERSION: Final[str] = "0.1"

LAYER_DIAGNOSTICS_INTERPRETATION: Final[str] = (
    "descriptive_layer_state_transition_and_lineage_diagnostics_only"
)

LAYER_DIAGNOSTICS_OPERATION_ORDER: Final[
    tuple[str, ...]
] = (
    "validate_exact_layer_stage_chain",
    "collect_aggregation_summary",
    "collect_residual_update_summary",
    "collect_normalization_summary",
    "collect_global_state_transition_statistics",
    "collect_optional_graph_batch_summaries",
    "collect_trace_retention_and_optional_edge_report",
    "collect_regularization_terms",
    "collect_exact_lineage_and_architecture_metadata",
    "derive_bounded_descriptive_alerts",
    "assert_tensor_free_report",
    "fingerprint_report",
)

LAYER_DIAGNOSTICS_PARAMETER_FREE: Final[bool] = True
LAYER_DIAGNOSTICS_BUFFER_FREE: Final[bool] = True
LAYER_DIAGNOSTICS_IMPLICIT_FORWARD_EXECUTION: Final[bool] = False

LAYER_DIAGNOSTIC_SECTION_STAGE_SUMMARIES: Final[str] = (
    "stage_summaries"
)
LAYER_DIAGNOSTIC_SECTION_GLOBAL: Final[str] = "global"
LAYER_DIAGNOSTIC_SECTION_BY_GRAPH: Final[str] = "by_graph"
LAYER_DIAGNOSTIC_SECTION_TRACE: Final[str] = "trace"
LAYER_DIAGNOSTIC_SECTION_REGULARIZATION: Final[str] = (
    "regularization"
)
LAYER_DIAGNOSTIC_SECTION_LINEAGE: Final[str] = "lineage"
LAYER_DIAGNOSTIC_SECTION_ALERTS: Final[str] = "alerts"

LAYER_DIAGNOSTIC_REQUIRED_SECTIONS: Final[
    tuple[str, ...]
] = (
    LAYER_DIAGNOSTIC_SECTION_STAGE_SUMMARIES,
    LAYER_DIAGNOSTIC_SECTION_GLOBAL,
    LAYER_DIAGNOSTIC_SECTION_BY_GRAPH,
    LAYER_DIAGNOSTIC_SECTION_TRACE,
    LAYER_DIAGNOSTIC_SECTION_REGULARIZATION,
    LAYER_DIAGNOSTIC_SECTION_LINEAGE,
    LAYER_DIAGNOSTIC_SECTION_ALERTS,
)


# =============================================================================
# Diagnostic thresholds
# =============================================================================


@dataclass(frozen=True, slots=True)
class LayerDiagnosticThresholds:
    """
    Bounded thresholds used only for descriptive alert flags.

    These values are not training constraints, acceptance criteria,
    uncertainty intervals, or causal tests.
    """

    near_zero_absolute: float = 1e-8
    large_node_state_l2_norm: float = 100.0
    large_global_update_to_source_ratio: float = 10.0
    large_global_output_to_source_ratio: float = 10.0
    high_isolated_node_fraction: float = 0.5
    high_near_zero_aggregate_fraction: float = 0.95
    high_near_zero_output_fraction: float = 0.95

    def __post_init__(self) -> None:
        for name in (
            "near_zero_absolute",
            "large_node_state_l2_norm",
            "large_global_update_to_source_ratio",
            "large_global_output_to_source_ratio",
        ):
            value = getattr(
                self,
                name,
            )

            if isinstance(
                value,
                bool,
            ) or not isinstance(
                value,
                (int, float),
            ):
                raise TypeError(
                    f"{name} must be numeric."
                )

            numeric = float(
                value
            )

            if not math.isfinite(
                numeric
            ):
                raise ValueError(
                    f"{name} must be finite."
                )

            if numeric <= 0.0:
                raise ValueError(
                    f"{name} must be strictly positive."
                )

        for name in (
            "high_isolated_node_fraction",
            "high_near_zero_aggregate_fraction",
            "high_near_zero_output_fraction",
        ):
            value = getattr(
                self,
                name,
            )

            if isinstance(
                value,
                bool,
            ) or not isinstance(
                value,
                (int, float),
            ):
                raise TypeError(
                    f"{name} must be numeric."
                )

            numeric = float(
                value
            )

            if not math.isfinite(
                numeric
            ):
                raise ValueError(
                    f"{name} must be finite."
                )

            if not 0.0 <= numeric <= 1.0:
                raise ValueError(
                    f"{name} must lie in [0, 1]."
                )

    def architecture_dict(
        self,
    ) -> dict[str, float]:
        return {
            key: float(value)
            for key, value
            in asdict(self).items()
        }


DEFAULT_LAYER_DIAGNOSTIC_THRESHOLDS: Final[
    LayerDiagnosticThresholds
] = LayerDiagnosticThresholds()


# =============================================================================
# Generic helpers
# =============================================================================


def _canonical_json(
    payload: Mapping[str, Any],
) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _fingerprint(
    payload: Mapping[str, Any],
) -> str:
    return sha256(
        _canonical_json(payload)
        .encode("utf-8")
    ).hexdigest()


def _require_nonempty_string(
    name: str,
    value: str,
) -> None:
    if not isinstance(
        value,
        str,
    ) or not value.strip():
        raise ValueError(
            f"{name} must be a non-empty string."
        )


def _require_boolean(
    name: str,
    value: bool,
) -> None:
    if not isinstance(
        value,
        bool,
    ):
        raise TypeError(
            f"{name} must be Boolean."
        )


def _require_nonnegative_int(
    name: str,
    value: int,
) -> None:
    if isinstance(
        value,
        bool,
    ) or not isinstance(
        value,
        int,
    ):
        raise TypeError(
            f"{name} must be an integer."
        )

    if value < 0:
        raise ValueError(
            f"{name} must be nonnegative."
        )


def _require_thresholds(
    thresholds: LayerDiagnosticThresholds,
) -> None:
    if not isinstance(
        thresholds,
        LayerDiagnosticThresholds,
    ):
        raise TypeError(
            "thresholds must be a "
            "LayerDiagnosticThresholds."
        )


def _require_edge_thresholds(
    edge_thresholds: MessageBuilderDiagnosticThresholds,
) -> None:
    if not isinstance(
        edge_thresholds,
        MessageBuilderDiagnosticThresholds,
    ):
        raise TypeError(
            "edge_thresholds must be a "
            "MessageBuilderDiagnosticThresholds."
        )


def _require_internal_output(
    internal_output: LayerComputationOutput,
) -> None:
    if not isinstance(
        internal_output,
        LayerComputationOutput,
    ):
        raise TypeError(
            "internal_output must be a "
            "LayerComputationOutput."
        )


def _require_public_output(
    public_output: FunctionalMessagePassingLayerOutput,
) -> None:
    if not isinstance(
        public_output,
        FunctionalMessagePassingLayerOutput,
    ):
        raise TypeError(
            "public_output must be a "
            "FunctionalMessagePassingLayerOutput."
        )


def _require_float_tensor(
    name: str,
    value: torch.Tensor,
) -> None:
    if not isinstance(
        value,
        torch.Tensor,
    ):
        raise TypeError(
            f"{name} must be a tensor."
        )

    if not value.dtype.is_floating_point:
        raise ValueError(
            f"{name} must use a floating-point dtype."
        )

    if not bool(
        torch.isfinite(value)
        .all()
        .item()
    ):
        raise FloatingPointError(
            f"{name} must contain only finite values."
        )


def _require_float_matrix(
    name: str,
    value: torch.Tensor,
) -> None:
    _require_float_tensor(
        name,
        value,
    )

    if value.ndim != 2:
        raise ValueError(
            f"{name} must have rank 2 and shape [N, H]; "
            f"observed {tuple(value.shape)}."
        )


def _require_long_vector(
    name: str,
    value: torch.Tensor,
) -> None:
    if not isinstance(
        value,
        torch.Tensor,
    ):
        raise TypeError(
            f"{name} must be a tensor."
        )

    if value.dtype != torch.long:
        raise ValueError(
            f"{name} must use torch.long dtype."
        )

    if value.ndim != 1:
        raise ValueError(
            f"{name} must have rank 1."
        )


def _safe_fraction(
    numerator: int,
    denominator: int,
) -> float:
    if denominator == 0:
        return 0.0

    return (
        float(numerator)
        / float(denominator)
    )


def _safe_ratio(
    numerator: float,
    denominator: float,
    *,
    near_zero_absolute: float,
) -> float | None:
    if abs(
        denominator
    ) <= near_zero_absolute:
        return None

    return (
        numerator
        / denominator
    )


def _assert_tensor_free(
    value: Any,
    *,
    path: str = "report",
) -> None:
    """
    Ensure no tensor, module, or source object leaks into a report.
    """

    if isinstance(
        value,
        torch.Tensor,
    ):
        raise TypeError(
            f"{path} must not retain tensors."
        )

    if isinstance(
        value,
        nn.Module,
    ):
        raise TypeError(
            f"{path} must not retain modules."
        )

    if isinstance(
        value,
        Mapping,
    ):
        for key, nested in (
            value.items()
        ):
            _assert_tensor_free(
                nested,
                path=f"{path}.{key}",
            )
        return

    if isinstance(
        value,
        (list, tuple),
    ):
        for index, nested in enumerate(
            value
        ):
            _assert_tensor_free(
                nested,
                path=f"{path}[{index}]",
            )
        return

    if (
        value is None
        or isinstance(
            value,
            (
                str,
                bool,
                int,
                float,
            ),
        )
    ):
        return

    raise TypeError(
        f"{path} contains unsupported value type "
        f"{type(value).__name__}."
    )


def _validate_internal_output(
    internal_output: LayerComputationOutput,
) -> None:
    _require_internal_output(
        internal_output
    )

    validate_layer_stage_chain(
        layer_inputs=(
            internal_output
            .layer_inputs
        ),
        aggregation=(
            internal_output
            .aggregation
        ),
        residual_update=(
            internal_output
            .residual_update
        ),
        normalization=(
            internal_output
            .normalization
        ),
        computation_output=(
            internal_output
        ),
    )
    validate_layer_residual_update_output(
        output=(
            internal_output
            .residual_update
        ),
        aggregation=(
            internal_output
            .aggregation
        ),
        layer_inputs=(
            internal_output
            .layer_inputs
        ),
    )
    validate_layer_normalization_output(
        output=(
            internal_output
            .normalization
        ),
        residual_update=(
            internal_output
            .residual_update
        ),
    )


# =============================================================================
# General tensor statistics
# =============================================================================


def scalar_tensor_statistics(
    value: torch.Tensor,
    *,
    near_zero_absolute: float,
) -> dict[str, Any]:
    """
    Describe a floating tensor as one flattened population.
    """

    _require_float_tensor(
        "value",
        value,
    )

    if not math.isfinite(
        float(near_zero_absolute)
    ) or float(
        near_zero_absolute
    ) <= 0.0:
        raise ValueError(
            "near_zero_absolute must be finite and strictly positive."
        )

    count = int(
        value.numel()
    )

    if count == 0:
        return {
            "count": 0,
            "minimum": None,
            "maximum": None,
            "mean": None,
            "standard_deviation": None,
            "mean_absolute_value": None,
            "l1_norm": 0.0,
            "l2_norm": 0.0,
            "zero_count": 0,
            "zero_fraction": 0.0,
            "near_zero_count": 0,
            "near_zero_fraction": 0.0,
            "positive_count": 0,
            "negative_count": 0,
            "finite": True,
        }

    detached = (
        value.detach()
    )
    zero_count = int(
        (detached == 0)
        .sum()
        .item()
    )
    near_zero_count = int(
        (
            detached.abs()
            <= float(
                near_zero_absolute
            )
        )
        .sum()
        .item()
    )
    positive_count = int(
        (detached > 0)
        .sum()
        .item()
    )
    negative_count = int(
        (detached < 0)
        .sum()
        .item()
    )

    return {
        "count": count,
        "minimum": float(
            detached.min().item()
        ),
        "maximum": float(
            detached.max().item()
        ),
        "mean": float(
            detached.mean().item()
        ),
        "standard_deviation": float(
            detached.std(
                unbiased=False
            ).item()
        ),
        "mean_absolute_value": float(
            detached.abs()
            .mean()
            .item()
        ),
        "l1_norm": float(
            torch.linalg.vector_norm(
                detached.reshape(-1),
                ord=1,
            ).item()
        ),
        "l2_norm": float(
            torch.linalg.vector_norm(
                detached.reshape(-1),
                ord=2,
            ).item()
        ),
        "zero_count": zero_count,
        "zero_fraction": (
            _safe_fraction(
                zero_count,
                count,
            )
        ),
        "near_zero_count": (
            near_zero_count
        ),
        "near_zero_fraction": (
            _safe_fraction(
                near_zero_count,
                count,
            )
        ),
        "positive_count": (
            positive_count
        ),
        "negative_count": (
            negative_count
        ),
        "finite": True,
    }


def matrix_statistics(
    value: torch.Tensor,
    *,
    near_zero_absolute: float,
) -> dict[str, Any]:
    """
    Describe a node-aligned matrix and its per-node L2 norms.
    """

    _require_float_matrix(
        "value",
        value,
    )

    element_summary = (
        scalar_tensor_statistics(
            value,
            near_zero_absolute=(
                near_zero_absolute
            ),
        )
    )
    node_norms = (
        torch.linalg.vector_norm(
            value,
            ord=2,
            dim=1,
        )
    )
    node_norm_summary = (
        scalar_tensor_statistics(
            node_norms,
            near_zero_absolute=(
                near_zero_absolute
            ),
        )
    )

    return {
        "shape": [
            int(size)
            for size in value.shape
        ],
        "node_count": int(
            value.shape[0]
        ),
        "hidden_dim": int(
            value.shape[1]
        ),
        "dtype": str(
            value.dtype
        ),
        "device": str(
            value.device
        ),
        "elements": (
            element_summary
        ),
        "per_node_l2_norm": (
            node_norm_summary
        ),
    }


def state_transition_statistics(
    *,
    source_state: torch.Tensor,
    target_state: torch.Tensor,
    near_zero_absolute: float,
) -> dict[str, Any]:
    """
    Describe one node-state transition without assigning causal meaning.
    """

    _require_float_matrix(
        "source_state",
        source_state,
    )
    _require_float_matrix(
        "target_state",
        target_state,
    )

    if tuple(
        source_state.shape
    ) != tuple(
        target_state.shape
    ):
        raise ValueError(
            "source_state and target_state must share one shape."
        )

    if source_state.dtype != (
        target_state.dtype
    ):
        raise ValueError(
            "source_state and target_state must share one dtype."
        )

    if source_state.device != (
        target_state.device
    ):
        raise ValueError(
            "source_state and target_state must share one device."
        )

    delta = (
        target_state
        - source_state
    )

    source_norm = float(
        torch.linalg.vector_norm(
            source_state.detach()
        ).item()
    )
    target_norm = float(
        torch.linalg.vector_norm(
            target_state.detach()
        ).item()
    )
    delta_norm = float(
        torch.linalg.vector_norm(
            delta.detach()
        ).item()
    )

    per_node_source_norm = (
        torch.linalg.vector_norm(
            source_state.detach(),
            ord=2,
            dim=1,
        )
    )
    per_node_target_norm = (
        torch.linalg.vector_norm(
            target_state.detach(),
            ord=2,
            dim=1,
        )
    )
    per_node_delta_norm = (
        torch.linalg.vector_norm(
            delta.detach(),
            ord=2,
            dim=1,
        )
    )

    denominator = (
        per_node_source_norm
        .clamp_min(
            float(
                near_zero_absolute
            )
        )
    )
    per_node_relative_change = (
        per_node_delta_norm
        / denominator
    )

    source_is_near_zero = (
        per_node_source_norm
        <= float(
            near_zero_absolute
        )
    )

    return {
        "source_global_l2_norm": (
            source_norm
        ),
        "target_global_l2_norm": (
            target_norm
        ),
        "delta_global_l2_norm": (
            delta_norm
        ),
        "target_to_source_global_norm_ratio": (
            _safe_ratio(
                target_norm,
                source_norm,
                near_zero_absolute=(
                    near_zero_absolute
                ),
            )
        ),
        "delta_to_source_global_norm_ratio": (
            _safe_ratio(
                delta_norm,
                source_norm,
                near_zero_absolute=(
                    near_zero_absolute
                ),
            )
        ),
        "source_near_zero_node_count": int(
            source_is_near_zero
            .sum()
            .item()
        ),
        "source_near_zero_node_fraction": (
            _safe_fraction(
                int(
                    source_is_near_zero
                    .sum()
                    .item()
                ),
                int(
                    source_state.shape[0]
                ),
            )
        ),
        "per_node_source_l2_norm": (
            scalar_tensor_statistics(
                per_node_source_norm,
                near_zero_absolute=(
                    near_zero_absolute
                ),
            )
        ),
        "per_node_target_l2_norm": (
            scalar_tensor_statistics(
                per_node_target_norm,
                near_zero_absolute=(
                    near_zero_absolute
                ),
            )
        ),
        "per_node_delta_l2_norm": (
            scalar_tensor_statistics(
                per_node_delta_norm,
                near_zero_absolute=(
                    near_zero_absolute
                ),
            )
        ),
        "per_node_relative_change": (
            scalar_tensor_statistics(
                per_node_relative_change,
                near_zero_absolute=(
                    near_zero_absolute
                ),
            )
        ),
    }


# =============================================================================
# Aggregation diagnostics
# =============================================================================


def incoming_edge_count_statistics(
    incoming_edge_count: torch.Tensor,
) -> dict[str, Any]:
    """
    Describe target-node incoming-edge coverage.
    """

    _require_long_vector(
        "incoming_edge_count",
        incoming_edge_count,
    )

    count = int(
        incoming_edge_count.numel()
    )

    if count == 0:
        return {
            "node_count": 0,
            "total_incoming_edges": 0,
            "minimum": None,
            "maximum": None,
            "mean": None,
            "standard_deviation": None,
            "isolated_node_count": 0,
            "isolated_node_fraction": 0.0,
            "nonisolated_node_count": 0,
        }

    detached = (
        incoming_edge_count
        .detach()
    )
    isolated_count = int(
        (detached == 0)
        .sum()
        .item()
    )

    return {
        "node_count": count,
        "total_incoming_edges": int(
            detached.sum().item()
        ),
        "minimum": int(
            detached.min().item()
        ),
        "maximum": int(
            detached.max().item()
        ),
        "mean": float(
            detached
            .to(torch.float64)
            .mean()
            .item()
        ),
        "standard_deviation": float(
            detached
            .to(torch.float64)
            .std(
                unbiased=False
            )
            .item()
        ),
        "isolated_node_count": (
            isolated_count
        ),
        "isolated_node_fraction": (
            _safe_fraction(
                isolated_count,
                count,
            )
        ),
        "nonisolated_node_count": (
            count
            - isolated_count
        ),
    }


def aggregation_diagnostic_summary(
    aggregation: AggregationOutput,
    *,
    thresholds: LayerDiagnosticThresholds = (
        DEFAULT_LAYER_DIAGNOSTIC_THRESHOLDS
    ),
) -> dict[str, Any]:
    """
    Return descriptive aggregation coverage and scale statistics.
    """

    if not isinstance(
        aggregation,
        AggregationOutput,
    ):
        raise TypeError(
            "aggregation must be an AggregationOutput."
        )

    _require_thresholds(
        thresholds
    )

    source_inputs = (
        aggregation
        .source_messages
        .source_inputs
    )

    if tuple(
        aggregation
        .node_aggregate
        .shape
    ) != (
        source_inputs.num_nodes,
        source_inputs.hidden_dim,
    ):
        raise ValueError(
            "aggregation.node_aggregate has an unexpected shape."
        )

    return {
        "aggregation_mode": (
            aggregation
            .aggregation_mode
        ),
        "encoder_architecture_fingerprint": (
            aggregation
            .encoder_architecture_fingerprint
        ),
        "num_nodes": (
            source_inputs.num_nodes
        ),
        "num_edges": (
            source_inputs.num_edges
        ),
        "hidden_dim": (
            source_inputs.hidden_dim
        ),
        "dtype": str(
            aggregation
            .node_aggregate
            .dtype
        ),
        "device": str(
            aggregation
            .node_aggregate
            .device
        ),
        "incoming_edge_count": (
            incoming_edge_count_statistics(
                aggregation
                .incoming_edge_count
            )
        ),
        "node_aggregate": (
            matrix_statistics(
                aggregation
                .node_aggregate,
                near_zero_absolute=(
                    thresholds
                    .near_zero_absolute
                ),
            )
        ),
        "exact_source_message_lineage": {
            "source_inputs_preserved": (
                aggregation
                .source_messages
                .source_inputs
                is source_inputs
            ),
        },
        "residual_performed_here": False,
        "normalization_performed_here": False,
        "causal_importance_claim": False,
        "explanation_faithfulness_claim": False,
    }


# =============================================================================
# Graph-batch slices
# =============================================================================


def _masked_matrix_statistics(
    value: torch.Tensor,
    mask: torch.Tensor,
    *,
    near_zero_absolute: float,
) -> dict[str, Any]:
    if not isinstance(
        mask,
        torch.Tensor,
    ):
        raise TypeError(
            "mask must be a tensor."
        )

    if mask.dtype != torch.bool:
        raise ValueError(
            "mask must use Boolean dtype."
        )

    if mask.ndim != 1:
        raise ValueError(
            "mask must have rank 1."
        )

    if tuple(
        mask.shape
    ) != (
        int(
            value.shape[0]
        ),
    ):
        raise ValueError(
            "mask must align with the node axis."
        )

    if mask.device != (
        value.device
    ):
        raise ValueError(
            "mask and value must share one device."
        )

    return matrix_statistics(
        value[mask],
        near_zero_absolute=(
            near_zero_absolute
        ),
    )


def graph_batch_diagnostics(
    internal_output: LayerComputationOutput,
    *,
    thresholds: LayerDiagnosticThresholds = (
        DEFAULT_LAYER_DIAGNOSTIC_THRESHOLDS
    ),
) -> list[dict[str, Any]]:
    """
    Describe each graph in the exact input batch independently.
    """

    _validate_internal_output(
        internal_output
    )
    _require_thresholds(
        thresholds
    )

    source_inputs = (
        internal_output
        .source_inputs
    )
    source_state = (
        internal_output
        .layer_inputs
        .input_node_state
    )
    aggregate = (
        internal_output
        .aggregation
        .node_aggregate
    )
    post_dropout = (
        internal_output
        .residual_update
        .post_dropout_update
    )
    post_residual = (
        internal_output
        .residual_update
        .post_residual_state
    )
    updated = (
        internal_output
        .updated_node_state
    )

    reports: list[
        dict[str, Any]
    ] = []

    for graph_index in range(
        source_inputs.num_graphs
    ):
        node_mask = (
            source_inputs
            .node_batch_index
            == graph_index
        )
        edge_mask = (
            source_inputs
            .edge_batch_index
            == graph_index
        )

        node_count = int(
            node_mask.sum().item()
        )
        edge_count = int(
            edge_mask.sum().item()
        )

        graph_incoming = (
            internal_output
            .incoming_edge_count[
                node_mask
            ]
        )
        isolated_count = int(
            (graph_incoming == 0)
            .sum()
            .item()
        )

        reports.append(
            {
                "graph_index": (
                    graph_index
                ),
                "node_count": node_count,
                "edge_count": edge_count,
                "node_fraction": (
                    _safe_fraction(
                        node_count,
                        source_inputs
                        .num_nodes,
                    )
                ),
                "edge_fraction": (
                    _safe_fraction(
                        edge_count,
                        source_inputs
                        .num_edges,
                    )
                ),
                "isolated_node_count": (
                    isolated_count
                ),
                "isolated_node_fraction": (
                    _safe_fraction(
                        isolated_count,
                        node_count,
                    )
                ),
                "incoming_edge_count": (
                    incoming_edge_count_statistics(
                        graph_incoming
                    )
                ),
                "source_node_state": (
                    _masked_matrix_statistics(
                        source_state,
                        node_mask,
                        near_zero_absolute=(
                            thresholds
                            .near_zero_absolute
                        ),
                    )
                ),
                "node_aggregate": (
                    _masked_matrix_statistics(
                        aggregate,
                        node_mask,
                        near_zero_absolute=(
                            thresholds
                            .near_zero_absolute
                        ),
                    )
                ),
                "post_dropout_update": (
                    _masked_matrix_statistics(
                        post_dropout,
                        node_mask,
                        near_zero_absolute=(
                            thresholds
                            .near_zero_absolute
                        ),
                    )
                ),
                "post_residual_state": (
                    _masked_matrix_statistics(
                        post_residual,
                        node_mask,
                        near_zero_absolute=(
                            thresholds
                            .near_zero_absolute
                        ),
                    )
                ),
                "updated_node_state": (
                    _masked_matrix_statistics(
                        updated,
                        node_mask,
                        near_zero_absolute=(
                            thresholds
                            .near_zero_absolute
                        ),
                    )
                ),
                "source_to_output_transition": (
                    state_transition_statistics(
                        source_state=(
                            source_state[
                                node_mask
                            ]
                        ),
                        target_state=(
                            updated[
                                node_mask
                            ]
                        ),
                        near_zero_absolute=(
                            thresholds
                            .near_zero_absolute
                        ),
                    )
                ),
            }
        )

    return reports


# =============================================================================
# Trace and lineage diagnostics
# =============================================================================


def layer_trace_diagnostic_summary(
    internal_output: LayerComputationOutput,
    *,
    include_edge_report: bool,
    edge_thresholds: MessageBuilderDiagnosticThresholds,
) -> dict[str, Any]:
    """
    Describe retained trace policy and optionally embed edge diagnostics.
    """

    _validate_internal_output(
        internal_output
    )
    _require_boolean(
        "include_edge_report",
        include_edge_report,
    )
    _require_edge_thresholds(
        edge_thresholds
    )

    policy = (
        internal_output
        .trace_policy
    )
    trace = (
        internal_output
        .trace
    )

    if policy.mode == (
        LAYER_TRACE_NONE
    ):
        if trace is not None:
            raise ValueError(
                "Trace policy 'none' must not retain a trace object."
            )

        return {
            "trace_mode": (
                LAYER_TRACE_NONE
            ),
            "trace_retained": False,
            "node_stages_retained": False,
            "edge_stages_retained": False,
            "edge_report_requested": (
                include_edge_report
            ),
            "edge_report_available": False,
            "edge_report": None,
        }

    if not isinstance(
        trace,
        FunctionalMessagePassingLayerTrace,
    ):
        raise TypeError(
            "Enabled tracing requires a "
            "FunctionalMessagePassingLayerTrace."
        )

    if trace.layer_inputs is not (
        internal_output.layer_inputs
    ):
        raise ValueError(
            "Trace must preserve exact layer_inputs."
        )

    if trace.aggregation is not (
        internal_output.aggregation
    ):
        raise ValueError(
            "Trace must preserve exact aggregation."
        )

    if trace.residual_update is not (
        internal_output.residual_update
    ):
        raise ValueError(
            "Trace must preserve exact residual_update."
        )

    if trace.normalization is not (
        internal_output.normalization
    ):
        raise ValueError(
            "Trace must preserve exact normalization."
        )

    edge_report: (
        dict[str, Any]
        | None
    ) = None

    if (
        include_edge_report
        and policy.mode
        == LAYER_TRACE_FULL
    ):
        if trace.message_builder_run is None:
            raise ValueError(
                "Full trace is missing message_builder_run."
            )

        if trace.edge_messages is None:
            raise ValueError(
                "Full trace is missing edge_messages."
            )

        edge_report = (
            build_public_edge_message_diagnostic_report(
                public_output=(
                    trace.edge_messages
                ),
                composition_output=(
                    trace
                    .message_builder_run
                    .composition_output
                ),
                include_per_relation=True,
                include_per_graph=True,
                thresholds=edge_thresholds,
            )
        )

    return {
        "trace_mode": (
            policy.mode
        ),
        "trace_retained": True,
        "node_stages_retained": (
            policy.retain_node_stages
        ),
        "edge_stages_retained": (
            policy.retain_edge_stages
        ),
        "trace_lineage_fingerprint": (
            trace.lineage_fingerprint()
        ),
        "edge_report_requested": (
            include_edge_report
        ),
        "edge_report_available": (
            edge_report is not None
        ),
        "edge_report": (
            edge_report
        ),
        "exact_identity": {
            "layer_inputs_preserved": (
                trace.layer_inputs
                is internal_output
                .layer_inputs
            ),
            "aggregation_preserved": (
                trace.aggregation
                is internal_output
                .aggregation
            ),
            "residual_update_preserved": (
                trace.residual_update
                is internal_output
                .residual_update
            ),
            "normalization_preserved": (
                trace.normalization
                is internal_output
                .normalization
            ),
            "updated_node_state_preserved": (
                trace.updated_node_state
                is internal_output
                .updated_node_state
            ),
        },
    }


def layer_lineage_summary(
    internal_output: LayerComputationOutput,
    *,
    public_output: (
        FunctionalMessagePassingLayerOutput
        | None
    ) = None,
) -> dict[str, Any]:
    """
    Return exact stage provenance without retaining source objects.
    """

    _validate_internal_output(
        internal_output
    )

    if public_output is not None:
        _require_public_output(
            public_output
        )
        validate_public_layer_output(
            public_output=(
                public_output
            ),
            internal_output=(
                internal_output
            ),
        )

    source_inputs = (
        internal_output
        .source_inputs
    )

    summary: dict[str, Any] = {
        "source_inputs_lineage_fingerprint": (
            source_inputs
            .lineage_fingerprint()
        ),
        "source_fingerprint": (
            source_inputs
            .source_fingerprint
        ),
        "compiled_relation_registry_fingerprint": (
            source_inputs
            .compiled_relation_registry
            .fingerprint()
        ),
        "layer_inputs": {
            "schema_version": (
                internal_output
                .layer_inputs
                .schema_version
            ),
            "layer_index": (
                internal_output
                .layer_index
            ),
            "training": (
                internal_output
                .layer_inputs
                .training
            ),
            "trace_policy": (
                internal_output
                .trace_policy
                .architecture_dict()
            ),
            "lineage_fingerprint": (
                internal_output
                .layer_inputs
                .lineage_fingerprint()
            ),
            "source_stack_fingerprint": (
                internal_output
                .layer_inputs
                .source_stack_fingerprint
            ),
        },
        "aggregation": {
            "schema_version": (
                internal_output
                .aggregation
                .schema_version
            ),
            "aggregation_mode": (
                internal_output
                .aggregation
                .aggregation_mode
            ),
            "architecture_fingerprint": (
                internal_output
                .aggregation
                .encoder_architecture_fingerprint
            ),
        },
        "residual_update": {
            "schema_version": (
                internal_output
                .residual_update
                .schema_version
            ),
            "architecture_fingerprint": (
                internal_output
                .residual_update
                .updater_architecture_fingerprint
            ),
            "parameter_fingerprint": (
                internal_output
                .residual_update
                .updater_parameter_fingerprint
            ),
            "lineage_fingerprint": (
                internal_output
                .residual_update
                .lineage_fingerprint()
            ),
        },
        "normalization": {
            "schema_version": (
                internal_output
                .normalization
                .schema_version
            ),
            "architecture_fingerprint": (
                internal_output
                .normalization
                .normalizer_architecture_fingerprint
            ),
            "parameter_fingerprint": (
                internal_output
                .normalization
                .normalizer_parameter_fingerprint
            ),
            "lineage_fingerprint": (
                internal_output
                .normalization
                .lineage_fingerprint()
            ),
        },
        "layer_computation": {
            "schema_version": (
                internal_output
                .schema_version
            ),
            "architecture_fingerprint": (
                internal_output
                .layer_architecture_fingerprint
            ),
            "parameter_fingerprint": (
                internal_output
                .layer_parameter_fingerprint
            ),
            "declared_lineage_fingerprint": (
                internal_output
                .lineage_fingerprint
            ),
            "value_lineage": (
                internal_output
                .value_lineage_dict()
            ),
        },
        "exact_object_identity": {
            "aggregation_source_inputs_preserved": (
                internal_output
                .aggregation
                .source_messages
                .source_inputs
                is source_inputs
            ),
            "residual_layer_inputs_preserved": (
                internal_output
                .residual_update
                .layer_inputs
                is internal_output
                .layer_inputs
            ),
            "residual_aggregation_preserved": (
                internal_output
                .residual_update
                .aggregation
                is internal_output
                .aggregation
            ),
            "normalization_residual_preserved": (
                internal_output
                .normalization
                .residual_update
                is internal_output
                .residual_update
            ),
            "updated_state_preserved": (
                internal_output
                .updated_node_state
                is internal_output
                .normalization
                .output_state
            ),
            "aggregate_preserved": (
                internal_output
                .node_aggregate
                is internal_output
                .aggregation
                .node_aggregate
            ),
            "incoming_edge_count_preserved": (
                internal_output
                .incoming_edge_count
                is internal_output
                .aggregation
                .incoming_edge_count
            ),
        },
    }

    if public_output is not None:
        summary["public_output"] = {
            "schema_version": (
                public_output
                .schema_version
            ),
            "architecture_fingerprint": (
                public_output
                .encoder_architecture_fingerprint
            ),
            "lineage_fingerprint": (
                public_output
                .lineage_fingerprint
            ),
            "updated_state_preserved": (
                public_output
                .updated_node_state
                is internal_output
                .updated_node_state
            ),
            "aggregate_preserved": (
                public_output
                .node_aggregate
                is internal_output
                .node_aggregate
            ),
            "incoming_edge_count_preserved": (
                public_output
                .incoming_edge_count
                is internal_output
                .incoming_edge_count
            ),
            "source_inputs_preserved": (
                public_output
                .source_inputs
                is source_inputs
            ),
        }

    return summary


# =============================================================================
# Regularization and alerts
# =============================================================================


def regularization_diagnostic_summary(
    internal_output: LayerComputationOutput,
) -> dict[str, Any]:
    """
    Convert scalar regularization tensors into Python values.
    """

    _validate_internal_output(
        internal_output
    )

    terms: dict[str, float] = {}

    for name, value in (
        internal_output
        .regularization_terms
        .items()
    ):
        _require_nonempty_string(
            "regularization term name",
            name,
        )

        if not isinstance(
            value,
            torch.Tensor,
        ):
            raise TypeError(
                f"Regularization term {name!r} must be a tensor."
            )

        if value.ndim != 0:
            raise ValueError(
                f"Regularization term {name!r} must be scalar."
            )

        if not value.dtype.is_floating_point:
            raise ValueError(
                f"Regularization term {name!r} must use floating dtype."
            )

        if not bool(
            torch.isfinite(value)
            .item()
        ):
            raise FloatingPointError(
                f"Regularization term {name!r} must be finite."
            )

        terms[name] = float(
            value.detach().item()
        )

    return {
        "term_count": len(
            terms
        ),
        "terms": terms,
        "sum": float(
            sum(
                terms.values()
            )
        ),
        "all_finite": True,
    }


def derive_layer_alerts(
    internal_output: LayerComputationOutput,
    *,
    thresholds: LayerDiagnosticThresholds = (
        DEFAULT_LAYER_DIAGNOSTIC_THRESHOLDS
    ),
) -> list[dict[str, Any]]:
    """
    Derive bounded descriptive flags from one valid internal output.
    """

    _validate_internal_output(
        internal_output
    )
    _require_thresholds(
        thresholds
    )

    alerts: list[
        dict[str, Any]
    ] = []

    num_nodes = (
        internal_output.num_nodes
    )
    num_edges = (
        internal_output
        .source_inputs
        .num_edges
    )

    if num_nodes == 0:
        alerts.append(
            {
                "code": "empty_node_set",
                "severity": "info",
                "observation": (
                    "The layer processed zero nodes."
                ),
            }
        )

    if num_edges == 0:
        alerts.append(
            {
                "code": "empty_edge_set",
                "severity": "info",
                "observation": (
                    "The layer processed zero edges; every target-node "
                    "aggregate is therefore zero."
                ),
            }
        )

    incoming = (
        internal_output
        .incoming_edge_count
    )
    isolated_count = int(
        (incoming == 0)
        .sum()
        .item()
    )
    isolated_fraction = (
        _safe_fraction(
            isolated_count,
            num_nodes,
        )
    )

    if isolated_count > 0:
        alerts.append(
            {
                "code": "isolated_nodes_present",
                "severity": "info",
                "count": (
                    isolated_count
                ),
                "fraction": (
                    isolated_fraction
                ),
                "observation": (
                    "Some nodes have no incoming edges and receive an exact "
                    "zero aggregate before residual processing."
                ),
            }
        )

    if isolated_fraction >= (
        thresholds
        .high_isolated_node_fraction
    ) and num_nodes > 0:
        alerts.append(
            {
                "code": "high_isolated_node_fraction",
                "severity": "warning",
                "count": (
                    isolated_count
                ),
                "fraction": (
                    isolated_fraction
                ),
                "threshold": (
                    thresholds
                    .high_isolated_node_fraction
                ),
                "observation": (
                    "The isolated-node fraction meets or exceeds the "
                    "configured descriptive threshold."
                ),
            }
        )

    aggregate_norms = (
        torch.linalg.vector_norm(
            internal_output
            .node_aggregate
            .detach(),
            ord=2,
            dim=1,
        )
    )
    near_zero_aggregate_count = int(
        (
            aggregate_norms
            <= thresholds
            .near_zero_absolute
        )
        .sum()
        .item()
    )
    near_zero_aggregate_fraction = (
        _safe_fraction(
            near_zero_aggregate_count,
            num_nodes,
        )
    )

    if (
        num_nodes > 0
        and near_zero_aggregate_fraction
        >= thresholds
        .high_near_zero_aggregate_fraction
    ):
        alerts.append(
            {
                "code": "high_near_zero_aggregate_fraction",
                "severity": "warning",
                "count": (
                    near_zero_aggregate_count
                ),
                "fraction": (
                    near_zero_aggregate_fraction
                ),
                "threshold": (
                    thresholds
                    .high_near_zero_aggregate_fraction
                ),
                "observation": (
                    "A high fraction of target-node aggregate vectors are "
                    "near zero under the configured threshold."
                ),
            }
        )

    output_norms = (
        torch.linalg.vector_norm(
            internal_output
            .updated_node_state
            .detach(),
            ord=2,
            dim=1,
        )
    )
    near_zero_output_count = int(
        (
            output_norms
            <= thresholds
            .near_zero_absolute
        )
        .sum()
        .item()
    )
    near_zero_output_fraction = (
        _safe_fraction(
            near_zero_output_count,
            num_nodes,
        )
    )

    if (
        num_nodes > 0
        and near_zero_output_fraction
        >= thresholds
        .high_near_zero_output_fraction
    ):
        alerts.append(
            {
                "code": "high_near_zero_output_fraction",
                "severity": "warning",
                "count": (
                    near_zero_output_count
                ),
                "fraction": (
                    near_zero_output_fraction
                ),
                "threshold": (
                    thresholds
                    .high_near_zero_output_fraction
                ),
                "observation": (
                    "A high fraction of updated node-state vectors are near "
                    "zero under the configured threshold."
                ),
            }
        )

    large_output_count = int(
        (
            output_norms
            >= thresholds
            .large_node_state_l2_norm
        )
        .sum()
        .item()
    )

    if large_output_count > 0:
        alerts.append(
            {
                "code": "large_updated_node_state_norm",
                "severity": "warning",
                "count": (
                    large_output_count
                ),
                "fraction": (
                    _safe_fraction(
                        large_output_count,
                        num_nodes,
                    )
                ),
                "threshold": (
                    thresholds
                    .large_node_state_l2_norm
                ),
                "observation": (
                    "Some updated node-state vectors exceed the configured "
                    "L2-norm threshold."
                ),
            }
        )

    source_state = (
        internal_output
        .layer_inputs
        .input_node_state
    )
    post_dropout = (
        internal_output
        .residual_update
        .post_dropout_update
    )
    updated = (
        internal_output
        .updated_node_state
    )

    source_norm = float(
        torch.linalg.vector_norm(
            source_state.detach()
        ).item()
    )
    update_norm = float(
        torch.linalg.vector_norm(
            post_dropout.detach()
        ).item()
    )
    output_norm = float(
        torch.linalg.vector_norm(
            updated.detach()
        ).item()
    )

    update_ratio = _safe_ratio(
        update_norm,
        source_norm,
        near_zero_absolute=(
            thresholds
            .near_zero_absolute
        ),
    )
    output_ratio = _safe_ratio(
        output_norm,
        source_norm,
        near_zero_absolute=(
            thresholds
            .near_zero_absolute
        ),
    )

    if (
        update_ratio is not None
        and update_ratio
        >= thresholds
        .large_global_update_to_source_ratio
    ):
        alerts.append(
            {
                "code": "large_global_update_to_source_ratio",
                "severity": "warning",
                "ratio": (
                    update_ratio
                ),
                "threshold": (
                    thresholds
                    .large_global_update_to_source_ratio
                ),
                "observation": (
                    "The global realized update-branch norm is large relative "
                    "to the input node-state norm."
                ),
            }
        )

    if (
        output_ratio is not None
        and output_ratio
        >= thresholds
        .large_global_output_to_source_ratio
    ):
        alerts.append(
            {
                "code": "large_global_output_to_source_ratio",
                "severity": "warning",
                "ratio": (
                    output_ratio
                ),
                "threshold": (
                    thresholds
                    .large_global_output_to_source_ratio
                ),
                "observation": (
                    "The global updated-state norm is large relative to the "
                    "input node-state norm."
                ),
            }
        )

    if (
        internal_output
        .normalization
        .normalization_mode
        == LAYER_NORMALIZATION_LAYER_NORM
        and internal_output.hidden_dim == 1
    ):
        alerts.append(
            {
                "code": "single_feature_layer_normalization",
                "severity": "info",
                "observation": (
                    "Layer normalization over one hidden feature removes all "
                    "within-node variation before affine transformation."
                ),
            }
        )

    return alerts


# =============================================================================
# Architecture identity
# =============================================================================


def layer_diagnostics_architecture_dict(
    *,
    include_per_graph: bool,
    include_edge_report: bool,
    thresholds: LayerDiagnosticThresholds,
    edge_thresholds: MessageBuilderDiagnosticThresholds,
) -> dict[str, Any]:
    """
    Return the non-numerical reporting architecture.
    """

    _require_boolean(
        "include_per_graph",
        include_per_graph,
    )
    _require_boolean(
        "include_edge_report",
        include_edge_report,
    )
    _require_thresholds(
        thresholds
    )
    _require_edge_thresholds(
        edge_thresholds
    )

    return {
        "schema_version": (
            LAYER_DIAGNOSTICS_SCHEMA_VERSION
        ),
        "interpretation": (
            LAYER_DIAGNOSTICS_INTERPRETATION
        ),
        "operation_order": list(
            LAYER_DIAGNOSTICS_OPERATION_ORDER
        ),
        "required_sections": list(
            LAYER_DIAGNOSTIC_REQUIRED_SECTIONS
        ),
        "include_per_graph": (
            include_per_graph
        ),
        "include_edge_report": (
            include_edge_report
        ),
        "thresholds": (
            thresholds
            .architecture_dict()
        ),
        "edge_thresholds": (
            edge_thresholds
            .architecture_dict()
        ),
        "parameter_free": (
            LAYER_DIAGNOSTICS_PARAMETER_FREE
        ),
        "buffer_free": (
            LAYER_DIAGNOSTICS_BUFFER_FREE
        ),
        "implicit_forward_execution": (
            LAYER_DIAGNOSTICS_IMPLICIT_FORWARD_EXECUTION
        ),
        "retains_tensors": False,
        "retains_modules": False,
        "retains_source_objects": False,
        "changes_numerical_layer_architecture": False,
        "claims_causal_importance": False,
        "claims_explanation_faithfulness": False,
        "claims_uncertainty_calibration": False,
    }


def layer_diagnostics_architecture_fingerprint(
    *,
    include_per_graph: bool,
    include_edge_report: bool,
    thresholds: LayerDiagnosticThresholds,
    edge_thresholds: MessageBuilderDiagnosticThresholds,
) -> str:
    return _fingerprint(
        layer_diagnostics_architecture_dict(
            include_per_graph=(
                include_per_graph
            ),
            include_edge_report=(
                include_edge_report
            ),
            thresholds=thresholds,
            edge_thresholds=(
                edge_thresholds
            ),
        )
    )


# =============================================================================
# Complete reports
# =============================================================================


def build_layer_diagnostic_report(
    internal_output: LayerComputationOutput,
    *,
    public_output: (
        FunctionalMessagePassingLayerOutput
        | None
    ) = None,
    include_per_graph: bool = True,
    include_edge_report: bool = True,
    thresholds: LayerDiagnosticThresholds = (
        DEFAULT_LAYER_DIAGNOSTIC_THRESHOLDS
    ),
    edge_thresholds: MessageBuilderDiagnosticThresholds = (
        DEFAULT_MESSAGE_BUILDER_DIAGNOSTIC_THRESHOLDS
    ),
) -> dict[str, Any]:
    """
    Build one complete tensor-free layer diagnostic report.
    """

    _validate_internal_output(
        internal_output
    )
    _require_boolean(
        "include_per_graph",
        include_per_graph,
    )
    _require_boolean(
        "include_edge_report",
        include_edge_report,
    )
    _require_thresholds(
        thresholds
    )
    _require_edge_thresholds(
        edge_thresholds
    )

    if public_output is not None:
        _require_public_output(
            public_output
        )
        validate_public_layer_output(
            public_output=(
                public_output
            ),
            internal_output=(
                internal_output
            ),
        )

    source_state = (
        internal_output
        .layer_inputs
        .input_node_state
    )
    aggregate = (
        internal_output
        .node_aggregate
    )
    pre_dropout = (
        internal_output
        .residual_update
        .pre_dropout_update
    )
    post_dropout = (
        internal_output
        .residual_update
        .post_dropout_update
    )
    post_residual = (
        internal_output
        .residual_update
        .post_residual_state
    )
    updated = (
        internal_output
        .updated_node_state
    )

    architecture = (
        layer_diagnostics_architecture_dict(
            include_per_graph=(
                include_per_graph
            ),
            include_edge_report=(
                include_edge_report
            ),
            thresholds=thresholds,
            edge_thresholds=(
                edge_thresholds
            ),
        )
    )

    stage_summaries = {
        "aggregation": (
            aggregation_diagnostic_summary(
                internal_output
                .aggregation,
                thresholds=thresholds,
            )
        ),
        "residual_update": (
            layer_residual_update_diagnostic_summary(
                internal_output
                .residual_update
            )
        ),
        "normalization": (
            layer_normalization_diagnostic_summary(
                internal_output
                .normalization
            )
        ),
    }

    global_summary = {
        "layer_index": (
            internal_output.layer_index
        ),
        "training": (
            internal_output
            .layer_inputs
            .training
        ),
        "trace_mode": (
            internal_output
            .trace_policy
            .mode
        ),
        "num_nodes": (
            internal_output.num_nodes
        ),
        "num_edges": (
            internal_output
            .source_inputs
            .num_edges
        ),
        "num_graphs": (
            internal_output
            .source_inputs
            .num_graphs
        ),
        "hidden_dim": (
            internal_output.hidden_dim
        ),
        "dtype": str(
            internal_output.dtype
        ),
        "device": str(
            internal_output.device
        ),
        "aggregation_mode": (
            internal_output
            .aggregation
            .aggregation_mode
        ),
        "residual_mode": (
            internal_output
            .residual_update
            .residual_mode
        ),
        "residual_enabled": (
            internal_output
            .residual_enabled
        ),
        "dropout_probability": float(
            internal_output
            .residual_update
            .dropout_probability
        ),
        "dropout_active": bool(
            internal_output
            .residual_update
            .training
            and internal_output
            .residual_update
            .dropout_probability
            > 0.0
        ),
        "normalization_mode": (
            internal_output
            .normalization
            .normalization_mode
        ),
        "normalization_enabled": (
            internal_output
            .layer_norm_enabled
        ),
        "incoming_edge_count": (
            incoming_edge_count_statistics(
                internal_output
                .incoming_edge_count
            )
        ),
        "states": {
            "source_node_state": (
                matrix_statistics(
                    source_state,
                    near_zero_absolute=(
                        thresholds
                        .near_zero_absolute
                    ),
                )
            ),
            "node_aggregate": (
                matrix_statistics(
                    aggregate,
                    near_zero_absolute=(
                        thresholds
                        .near_zero_absolute
                    ),
                )
            ),
            "pre_dropout_update": (
                matrix_statistics(
                    pre_dropout,
                    near_zero_absolute=(
                        thresholds
                        .near_zero_absolute
                    ),
                )
            ),
            "post_dropout_update": (
                matrix_statistics(
                    post_dropout,
                    near_zero_absolute=(
                        thresholds
                        .near_zero_absolute
                    ),
                )
            ),
            "post_residual_state": (
                matrix_statistics(
                    post_residual,
                    near_zero_absolute=(
                        thresholds
                        .near_zero_absolute
                    ),
                )
            ),
            "updated_node_state": (
                matrix_statistics(
                    updated,
                    near_zero_absolute=(
                        thresholds
                        .near_zero_absolute
                    ),
                )
            ),
        },
        "transitions": {
            "source_to_aggregate": (
                state_transition_statistics(
                    source_state=(
                        source_state
                    ),
                    target_state=(
                        aggregate
                    ),
                    near_zero_absolute=(
                        thresholds
                        .near_zero_absolute
                    ),
                )
            ),
            "aggregate_to_post_dropout": (
                state_transition_statistics(
                    source_state=(
                        aggregate
                    ),
                    target_state=(
                        post_dropout
                    ),
                    near_zero_absolute=(
                        thresholds
                        .near_zero_absolute
                    ),
                )
            ),
            "source_to_post_residual": (
                state_transition_statistics(
                    source_state=(
                        source_state
                    ),
                    target_state=(
                        post_residual
                    ),
                    near_zero_absolute=(
                        thresholds
                        .near_zero_absolute
                    ),
                )
            ),
            "post_residual_to_updated": (
                state_transition_statistics(
                    source_state=(
                        post_residual
                    ),
                    target_state=(
                        updated
                    ),
                    near_zero_absolute=(
                        thresholds
                        .near_zero_absolute
                    ),
                )
            ),
            "source_to_updated": (
                state_transition_statistics(
                    source_state=(
                        source_state
                    ),
                    target_state=(
                        updated
                    ),
                    near_zero_absolute=(
                        thresholds
                        .near_zero_absolute
                    ),
                )
            ),
        },
    }

    graph_reports = (
        graph_batch_diagnostics(
            internal_output,
            thresholds=thresholds,
        )
        if include_per_graph
        else []
    )

    report: dict[str, Any] = {
        "schema_version": (
            LAYER_DIAGNOSTICS_SCHEMA_VERSION
        ),
        "interpretation": (
            LAYER_DIAGNOSTICS_INTERPRETATION
        ),
        "diagnostics_architecture": (
            architecture
        ),
        "diagnostics_architecture_fingerprint": (
            layer_diagnostics_architecture_fingerprint(
                include_per_graph=(
                    include_per_graph
                ),
                include_edge_report=(
                    include_edge_report
                ),
                thresholds=thresholds,
                edge_thresholds=(
                    edge_thresholds
                ),
            )
        ),
        "upstream_schema_versions": {
            "layer_computation_output": (
                LAYER_COMPUTATION_OUTPUT_SCHEMA_VERSION
            ),
            "residual_updater": (
                LAYER_RESIDUAL_UPDATER_SCHEMA_VERSION
            ),
            "normalizer": (
                LAYER_NORMALIZER_SCHEMA_VERSION
            ),
        },
        LAYER_DIAGNOSTIC_SECTION_STAGE_SUMMARIES: (
            stage_summaries
        ),
        LAYER_DIAGNOSTIC_SECTION_GLOBAL: (
            global_summary
        ),
        LAYER_DIAGNOSTIC_SECTION_BY_GRAPH: (
            graph_reports
        ),
        LAYER_DIAGNOSTIC_SECTION_TRACE: (
            layer_trace_diagnostic_summary(
                internal_output,
                include_edge_report=(
                    include_edge_report
                ),
                edge_thresholds=(
                    edge_thresholds
                ),
            )
        ),
        LAYER_DIAGNOSTIC_SECTION_REGULARIZATION: (
            regularization_diagnostic_summary(
                internal_output
            )
        ),
        LAYER_DIAGNOSTIC_SECTION_LINEAGE: (
            layer_lineage_summary(
                internal_output,
                public_output=(
                    public_output
                ),
            )
        ),
        LAYER_DIAGNOSTIC_SECTION_ALERTS: (
            derive_layer_alerts(
                internal_output,
                thresholds=thresholds,
            )
        ),
        "scientific_claims": {
            "causal_importance": False,
            "explanation_faithfulness": False,
            "uncertainty_calibration": False,
            "counterfactual_effect": False,
            "mechanistic_identifiability": False,
            "relation_necessity": False,
        },
    }

    if public_output is not None:
        report["public_output"] = {
            "schema_version": (
                public_output
                .schema_version
            ),
            "layer_index": (
                public_output
                .layer_index
            ),
            "residual_enabled": (
                public_output
                .residual_enabled
            ),
            "layer_norm_enabled": (
                public_output
                .layer_norm_enabled
            ),
            "architecture_fingerprint": (
                public_output
                .encoder_architecture_fingerprint
            ),
            "lineage_fingerprint": (
                public_output
                .lineage_fingerprint
            ),
            "intermediates_retained": (
                public_output
                .intermediates
                is not None
            ),
            "exact_identity": {
                "updated_node_state_preserved": (
                    public_output
                    .updated_node_state
                    is internal_output
                    .updated_node_state
                ),
                "node_aggregate_preserved": (
                    public_output
                    .node_aggregate
                    is internal_output
                    .node_aggregate
                ),
                "incoming_edge_count_preserved": (
                    public_output
                    .incoming_edge_count
                    is internal_output
                    .incoming_edge_count
                ),
                "source_inputs_preserved": (
                    public_output
                    .source_inputs
                    is internal_output
                    .source_inputs
                ),
            },
        }

    _assert_tensor_free(
        report
    )

    report["report_fingerprint"] = (
        layer_diagnostic_report_fingerprint(
            report
        )
    )

    validate_layer_diagnostic_report(
        report,
        expected_num_graphs=(
            internal_output
            .source_inputs
            .num_graphs
            if include_per_graph
            else 0
        ),
    )

    return report


def build_public_layer_diagnostic_report(
    *,
    public_output: FunctionalMessagePassingLayerOutput,
    internal_output: LayerComputationOutput,
    include_per_graph: bool = True,
    include_edge_report: bool = True,
    thresholds: LayerDiagnosticThresholds = (
        DEFAULT_LAYER_DIAGNOSTIC_THRESHOLDS
    ),
    edge_thresholds: MessageBuilderDiagnosticThresholds = (
        DEFAULT_MESSAGE_BUILDER_DIAGNOSTIC_THRESHOLDS
    ),
) -> dict[str, Any]:
    """
    Build a complete report anchored to the public layer output.
    """

    _require_public_output(
        public_output
    )
    _validate_internal_output(
        internal_output
    )
    validate_public_layer_output(
        public_output=public_output,
        internal_output=internal_output,
    )

    return build_layer_diagnostic_report(
        internal_output,
        public_output=public_output,
        include_per_graph=(
            include_per_graph
        ),
        include_edge_report=(
            include_edge_report
        ),
        thresholds=thresholds,
        edge_thresholds=(
            edge_thresholds
        ),
    )


# =============================================================================
# Report validation and fingerprinting
# =============================================================================


def layer_diagnostic_report_fingerprint(
    report: Mapping[str, Any],
) -> str:
    """
    Fingerprint a tensor-free report without recursive self-reference.
    """

    if not isinstance(
        report,
        Mapping,
    ):
        raise TypeError(
            "report must be a mapping."
        )

    payload = {
        key: value
        for key, value
        in report.items()
        if key != "report_fingerprint"
    }

    _assert_tensor_free(
        payload
    )

    return _fingerprint(
        payload
    )


def validate_layer_diagnostic_report(
    report: Mapping[str, Any],
    *,
    expected_num_graphs: int | None = None,
) -> None:
    """
    Validate report completeness, serialization safety, and fingerprint.
    """

    if not isinstance(
        report,
        Mapping,
    ):
        raise TypeError(
            "report must be a mapping."
        )

    _assert_tensor_free(
        report
    )

    if report.get(
        "schema_version"
    ) != LAYER_DIAGNOSTICS_SCHEMA_VERSION:
        raise ValueError(
            "Layer diagnostic schema version is missing or unexpected."
        )

    if report.get(
        "interpretation"
    ) != LAYER_DIAGNOSTICS_INTERPRETATION:
        raise ValueError(
            "Layer diagnostic interpretation is missing or unexpected."
        )

    for section in (
        LAYER_DIAGNOSTIC_REQUIRED_SECTIONS
    ):
        if section not in report:
            raise ValueError(
                "Layer diagnostic report is missing required section "
                f"{section!r}."
            )

    by_graph = report[
        LAYER_DIAGNOSTIC_SECTION_BY_GRAPH
    ]

    if not isinstance(
        by_graph,
        list,
    ):
        raise TypeError(
            "by_graph must be a list."
        )

    if expected_num_graphs is not None:
        _require_nonnegative_int(
            "expected_num_graphs",
            expected_num_graphs,
        )

        if len(
            by_graph
        ) != expected_num_graphs:
            raise ValueError(
                "Layer diagnostic graph count differs from the expected "
                "count."
            )

    scientific_claims = report.get(
        "scientific_claims"
    )

    if not isinstance(
        scientific_claims,
        Mapping,
    ):
        raise TypeError(
            "scientific_claims must be a mapping."
        )

    unsupported = [
        name
        for name, value
        in scientific_claims.items()
        if value is not False
    ]

    if unsupported:
        raise ValueError(
            "Layer diagnostics must not assert unsupported scientific "
            f"claims. Non-false entries: {unsupported}."
        )

    if "report_fingerprint" in report:
        fingerprint = report[
            "report_fingerprint"
        ]
        _require_nonempty_string(
            "report_fingerprint",
            fingerprint,
        )

        expected = (
            layer_diagnostic_report_fingerprint(
                report
            )
        )

        if fingerprint != expected:
            raise ValueError(
                "Layer diagnostic report fingerprint does not match "
                "report contents."
            )

    json.dumps(
        report,
        sort_keys=True,
        allow_nan=False,
    )


# =============================================================================
# Parameter-free module wrapper
# =============================================================================


class LayerDiagnostics(nn.Module):
    """
    Explicit parameter-free layer diagnostic orchestrator.

    Parameters
    ----------
    include_per_graph:
        Include one node-state summary for every graph in the input batch.
    include_edge_report:
        Embed the existing message-builder report when a full trace exists.
    thresholds:
        Layer-level descriptive alert thresholds.
    edge_thresholds:
        Existing message-builder thresholds used only by the nested edge
        report.
    """

    include_per_graph: bool
    include_edge_report: bool
    thresholds: LayerDiagnosticThresholds
    edge_thresholds: (
        MessageBuilderDiagnosticThresholds
    )

    def __init__(
        self,
        *,
        include_per_graph: bool = True,
        include_edge_report: bool = True,
        thresholds: LayerDiagnosticThresholds = (
            DEFAULT_LAYER_DIAGNOSTIC_THRESHOLDS
        ),
        edge_thresholds: (
            MessageBuilderDiagnosticThresholds
        ) = (
            DEFAULT_MESSAGE_BUILDER_DIAGNOSTIC_THRESHOLDS
        ),
    ) -> None:
        super().__init__()

        _require_boolean(
            "include_per_graph",
            include_per_graph,
        )
        _require_boolean(
            "include_edge_report",
            include_edge_report,
        )
        _require_thresholds(
            thresholds
        )
        _require_edge_thresholds(
            edge_thresholds
        )

        self.include_per_graph = (
            include_per_graph
        )
        self.include_edge_report = (
            include_edge_report
        )
        self.thresholds = thresholds
        self.edge_thresholds = (
            edge_thresholds
        )

        self.assert_parameter_free()

    @property
    def parameter_count(self) -> int:
        return sum(
            int(parameter.numel())
            for parameter in self.parameters()
        )

    @property
    def trainable_parameter_count(
        self,
    ) -> int:
        return sum(
            int(parameter.numel())
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    @property
    def buffer_count(self) -> int:
        return sum(
            int(buffer.numel())
            for buffer in self.buffers()
        )

    @property
    def parameter_fingerprint(
        self,
    ) -> None:
        return None

    def architecture_dict(
        self,
    ) -> dict[str, Any]:
        return (
            layer_diagnostics_architecture_dict(
                include_per_graph=(
                    self.include_per_graph
                ),
                include_edge_report=(
                    self.include_edge_report
                ),
                thresholds=(
                    self.thresholds
                ),
                edge_thresholds=(
                    self.edge_thresholds
                ),
            )
        )

    def architecture_fingerprint(
        self,
    ) -> str:
        return (
            layer_diagnostics_architecture_fingerprint(
                include_per_graph=(
                    self.include_per_graph
                ),
                include_edge_report=(
                    self.include_edge_report
                ),
                thresholds=(
                    self.thresholds
                ),
                edge_thresholds=(
                    self.edge_thresholds
                ),
            )
        )

    def assert_parameter_free(
        self,
    ) -> None:
        parameters = tuple(
            self.named_parameters()
        )
        buffers = tuple(
            self.named_buffers()
        )

        if parameters:
            raise RuntimeError(
                "LayerDiagnostics must remain parameter-free."
            )

        if buffers:
            raise RuntimeError(
                "LayerDiagnostics must remain buffer-free."
            )

        if self.state_dict():
            raise RuntimeError(
                "LayerDiagnostics must have an empty state_dict."
            )

        if self.parameter_count != 0:
            raise RuntimeError(
                "LayerDiagnostics parameter_count must be zero."
            )

        if self.trainable_parameter_count != 0:
            raise RuntimeError(
                "LayerDiagnostics trainable_parameter_count must be zero."
            )

        if self.buffer_count != 0:
            raise RuntimeError(
                "LayerDiagnostics buffer_count must be zero."
            )

    def report(
        self,
        internal_output: LayerComputationOutput,
    ) -> dict[str, Any]:
        """
        Build one internal layer report.
        """

        self.assert_parameter_free()

        report = (
            build_layer_diagnostic_report(
                internal_output,
                include_per_graph=(
                    self.include_per_graph
                ),
                include_edge_report=(
                    self.include_edge_report
                ),
                thresholds=(
                    self.thresholds
                ),
                edge_thresholds=(
                    self.edge_thresholds
                ),
            )
        )

        if report[
            "diagnostics_architecture_fingerprint"
        ] != self.architecture_fingerprint():
            raise RuntimeError(
                "Layer diagnostic architecture fingerprint differs from "
                "the diagnostics module."
            )

        return report

    def public_report(
        self,
        *,
        public_output: FunctionalMessagePassingLayerOutput,
        internal_output: LayerComputationOutput,
    ) -> dict[str, Any]:
        """
        Build one report anchored to the public layer output.
        """

        self.assert_parameter_free()

        report = (
            build_public_layer_diagnostic_report(
                public_output=(
                    public_output
                ),
                internal_output=(
                    internal_output
                ),
                include_per_graph=(
                    self.include_per_graph
                ),
                include_edge_report=(
                    self.include_edge_report
                ),
                thresholds=(
                    self.thresholds
                ),
                edge_thresholds=(
                    self.edge_thresholds
                ),
            )
        )

        if report[
            "diagnostics_architecture_fingerprint"
        ] != self.architecture_fingerprint():
            raise RuntimeError(
                "Public layer diagnostic architecture fingerprint differs "
                "from the diagnostics module."
            )

        return report

    def forward(
        self,
        internal_output: LayerComputationOutput,
    ) -> dict[str, Any]:
        return self.report(
            internal_output
        )

    def extra_repr(self) -> str:
        return (
            f"include_per_graph={self.include_per_graph}, "
            f"include_edge_report={self.include_edge_report}, "
            "retains_tensors=False, "
            "parameter_free=True"
        )


# =============================================================================
# Builders and aliases
# =============================================================================


def build_layer_diagnostics(
    *,
    include_per_graph: bool = True,
    include_edge_report: bool = True,
    thresholds: LayerDiagnosticThresholds = (
        DEFAULT_LAYER_DIAGNOSTIC_THRESHOLDS
    ),
    edge_thresholds: MessageBuilderDiagnosticThresholds = (
        DEFAULT_MESSAGE_BUILDER_DIAGNOSTIC_THRESHOLDS
    ),
) -> LayerDiagnostics:
    """
    Construct the explicit parameter-free layer diagnostics orchestrator.
    """

    return LayerDiagnostics(
        include_per_graph=(
            include_per_graph
        ),
        include_edge_report=(
            include_edge_report
        ),
        thresholds=thresholds,
        edge_thresholds=(
            edge_thresholds
        ),
    )


FunctionalLayerDiagnostics = (
    LayerDiagnostics
)
MessagePassingLayerDiagnostics = (
    LayerDiagnostics
)

build_functional_layer_diagnostics = (
    build_layer_diagnostics
)
build_message_passing_layer_diagnostics = (
    build_layer_diagnostics
)

layer_diagnostic_report = (
    build_layer_diagnostic_report
)
public_layer_diagnostic_report = (
    build_public_layer_diagnostic_report
)


__all__ = (
    # Public identity.
    "LAYER_DIAGNOSTICS_SCHEMA_VERSION",
    "LAYER_DIAGNOSTICS_INTERPRETATION",
    "LAYER_DIAGNOSTICS_OPERATION_ORDER",
    "LAYER_DIAGNOSTICS_PARAMETER_FREE",
    "LAYER_DIAGNOSTICS_BUFFER_FREE",
    "LAYER_DIAGNOSTICS_IMPLICIT_FORWARD_EXECUTION",
    "LAYER_DIAGNOSTIC_SECTION_STAGE_SUMMARIES",
    "LAYER_DIAGNOSTIC_SECTION_GLOBAL",
    "LAYER_DIAGNOSTIC_SECTION_BY_GRAPH",
    "LAYER_DIAGNOSTIC_SECTION_TRACE",
    "LAYER_DIAGNOSTIC_SECTION_REGULARIZATION",
    "LAYER_DIAGNOSTIC_SECTION_LINEAGE",
    "LAYER_DIAGNOSTIC_SECTION_ALERTS",
    "LAYER_DIAGNOSTIC_REQUIRED_SECTIONS",
    # Thresholds.
    "LayerDiagnosticThresholds",
    "DEFAULT_LAYER_DIAGNOSTIC_THRESHOLDS",
    # Generic statistics.
    "scalar_tensor_statistics",
    "matrix_statistics",
    "state_transition_statistics",
    "incoming_edge_count_statistics",
    # Stage summaries.
    "aggregation_diagnostic_summary",
    "graph_batch_diagnostics",
    "layer_trace_diagnostic_summary",
    "layer_lineage_summary",
    "regularization_diagnostic_summary",
    "derive_layer_alerts",
    # Architecture.
    "layer_diagnostics_architecture_dict",
    "layer_diagnostics_architecture_fingerprint",
    # Reports.
    "build_layer_diagnostic_report",
    "layer_diagnostic_report",
    "build_public_layer_diagnostic_report",
    "public_layer_diagnostic_report",
    "layer_diagnostic_report_fingerprint",
    "validate_layer_diagnostic_report",
    # Module API.
    "LayerDiagnostics",
    "FunctionalLayerDiagnostics",
    "MessagePassingLayerDiagnostics",
    "build_layer_diagnostics",
    "build_functional_layer_diagnostics",
    "build_message_passing_layer_diagnostics",
)
