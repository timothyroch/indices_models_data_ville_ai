"""
Research-grade descriptive diagnostics for functional edge-message building.

Target path:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            functional_message_passing/
                message_builders/
                    diagnostics.py

This module assembles diagnostics across the completed message-builder stage
chain:

    RelationTransformOutput
        -> ResolvedMessageCoefficients
        -> MessageCompositionOutput
        -> optional public EdgeMessageOutput

The diagnostics are deliberately descriptive. They summarize numerical scale,
sparsity, edge coverage, exact-relation slices, graph-batch slices, lineage,
and architectural provenance. They do not claim:

- causal importance;
- explanation faithfulness;
- calibrated uncertainty;
- mechanistic identifiability;
- relation necessity;
- counterfactual validity.

Role
---------------
For every edge ``e``:

    m_e = u_e * n_e * g_e * alpha_e * w_e

where:

``u_e``
    Exact relation-transformed source state.

``n_e``
    Structural edge-normalization factor.

``g_e``
    Exact relation-gate factor, or one when disabled.

``alpha_e``
    Reduced edge-attention factor, or one when disabled.

``w_e``
    Explicitly consumed semantic edge factor, or one.

This module reports each factor separately and never collapses them into an
ambiguous "importance" score.

Diagnostic levels
-----------------
``stage summaries``
    Reuse the validated stage-specific summaries from relation-state
    resolution, coefficient resolution, and message composition.

``global summary``
    Describes all edges jointly.

``exact-relation summary``
    Describes edge counts, factor distributions, and message norms separately
    for every exact compiled relation. Semantic-family pooling is not used.

``graph summary``
    Describes each graph in a batched input separately.

``alerts``
    Emits bounded descriptive flags for empty edge sets, signed semantic
    factors, zero coefficients, unusually large absolute coefficients, and
    unusually large message norms. Alerts are not scientific conclusions.

Lineage
-------
The report records architecture and lineage fingerprints from every stage and
requires exact object identity across the stage chain. It does not reconstruct
or independently infer provenance.

Performance
-----------
Diagnostics are intended for explicit research evaluation, debugging, and
artifact generation. They detach tensors only when converting to Python
scalars and do not modify the model graph. No tensor-valued state is retained
by the diagnostics module.

The diagnostics module is parameter-free and buffer-free.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from typing import Any, Final, Mapping, Sequence

import torch
from torch import nn

from ..schemas import (
    EdgeMessageOutput,
    FunctionalMessagePassingInputs,
    RelationTransformOutput,
)
from .coefficient_resolution import (
    MESSAGE_COEFFICIENT_RESOLUTION_SCHEMA_VERSION,
    message_coefficient_diagnostic_summary,
    validate_resolved_message_coefficients,
)
from .message_composition import (
    MESSAGE_COMPOSER_SCHEMA_VERSION,
    message_composition_diagnostic_summary,
    validate_message_composition_output,
)
from .relation_state_gather import (
    RELATION_STATE_GATHER_SCHEMA_VERSION,
    relation_state_gather_diagnostic_summary,
    validate_edge_aligned_relation_state,
)
from .schemas import (
    MESSAGE_FACTOR_EDGE_ATTENTION,
    MESSAGE_FACTOR_ORDER,
    MESSAGE_FACTOR_RELATION_GATE,
    MESSAGE_FACTOR_SEMANTIC_EDGE_WEIGHT,
    MESSAGE_FACTOR_STRUCTURAL_NORMALIZATION,
    MessageCompositionOutput,
    ResolvedMessageCoefficients,
    validate_message_builder_stage_chain,
    validate_public_edge_message_output,
)


# =============================================================================
# Public identity
# =============================================================================


MESSAGE_BUILDER_DIAGNOSTICS_SCHEMA_VERSION: Final[str] = "0.1"

MESSAGE_BUILDER_DIAGNOSTICS_INTERPRETATION: Final[str] = (
    "descriptive_numerical_and_lineage_diagnostics_only"
)

MESSAGE_BUILDER_DIAGNOSTICS_OPERATION_ORDER: Final[
    tuple[str, ...]
] = (
    "validate_exact_internal_stage_chain",
    "collect_stage_specific_summaries",
    "collect_global_factor_and_message_statistics",
    "collect_exact_relation_summaries",
    "collect_graph_batch_summaries",
    "collect_lineage_and_architecture_metadata",
    "derive_bounded_descriptive_alerts",
    "assert_python_scalar_report",
    "fingerprint_report",
)

MESSAGE_BUILDER_DIAGNOSTICS_PARAMETER_FREE: Final[bool] = True
MESSAGE_BUILDER_DIAGNOSTICS_BUFFER_FREE: Final[bool] = True

MESSAGE_BUILDER_DIAGNOSTIC_SECTION_STAGE_SUMMARIES: Final[str] = (
    "stage_summaries"
)
MESSAGE_BUILDER_DIAGNOSTIC_SECTION_GLOBAL: Final[str] = "global"
MESSAGE_BUILDER_DIAGNOSTIC_SECTION_BY_RELATION: Final[str] = (
    "by_exact_relation"
)
MESSAGE_BUILDER_DIAGNOSTIC_SECTION_BY_GRAPH: Final[str] = (
    "by_graph"
)
MESSAGE_BUILDER_DIAGNOSTIC_SECTION_LINEAGE: Final[str] = "lineage"
MESSAGE_BUILDER_DIAGNOSTIC_SECTION_ALERTS: Final[str] = "alerts"

MESSAGE_BUILDER_DIAGNOSTIC_REQUIRED_SECTIONS: Final[
    tuple[str, ...]
] = (
    MESSAGE_BUILDER_DIAGNOSTIC_SECTION_STAGE_SUMMARIES,
    MESSAGE_BUILDER_DIAGNOSTIC_SECTION_GLOBAL,
    MESSAGE_BUILDER_DIAGNOSTIC_SECTION_BY_RELATION,
    MESSAGE_BUILDER_DIAGNOSTIC_SECTION_BY_GRAPH,
    MESSAGE_BUILDER_DIAGNOSTIC_SECTION_LINEAGE,
    MESSAGE_BUILDER_DIAGNOSTIC_SECTION_ALERTS,
)


# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True, slots=True)
class MessageBuilderDiagnosticThresholds:
    """
    Bounded thresholds used only to produce descriptive alert flags.

    These thresholds are not training constraints, acceptance criteria,
    uncertainty estimates, or causal tests.
    """

    near_zero_absolute: float = 1e-8
    large_absolute_coefficient: float = 10.0
    large_message_l2_norm: float = 100.0
    high_near_zero_fraction: float = 0.95
    high_zero_message_fraction: float = 0.95

    def __post_init__(self) -> None:
        for name in (
            "near_zero_absolute",
            "large_absolute_coefficient",
            "large_message_l2_norm",
        ):
            value = getattr(self, name)

            if not isinstance(value, (int, float)):
                raise TypeError(
                    f"{name} must be numeric."
                )

            if not math.isfinite(float(value)):
                raise ValueError(
                    f"{name} must be finite."
                )

            if float(value) <= 0:
                raise ValueError(
                    f"{name} must be strictly positive."
                )

        for name in (
            "high_near_zero_fraction",
            "high_zero_message_fraction",
        ):
            value = getattr(self, name)

            if not isinstance(value, (int, float)):
                raise TypeError(
                    f"{name} must be numeric."
                )

            if not math.isfinite(float(value)):
                raise ValueError(
                    f"{name} must be finite."
                )

            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(
                    f"{name} must lie in [0, 1]."
                )

    def architecture_dict(self) -> dict[str, float]:
        return {
            key: float(value)
            for key, value in asdict(self).items()
        }


DEFAULT_MESSAGE_BUILDER_DIAGNOSTIC_THRESHOLDS: Final[
    MessageBuilderDiagnosticThresholds
] = MessageBuilderDiagnosticThresholds()


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
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"{name} must be a non-empty string."
        )


def _require_source_inputs(
    source_inputs: FunctionalMessagePassingInputs,
) -> None:
    if not isinstance(
        source_inputs,
        FunctionalMessagePassingInputs,
    ):
        raise TypeError(
            "source_inputs must be a "
            "FunctionalMessagePassingInputs."
        )


def _require_relation_transform(
    relation_transform: RelationTransformOutput,
) -> None:
    if not isinstance(
        relation_transform,
        RelationTransformOutput,
    ):
        raise TypeError(
            "relation_transform must be a "
            "RelationTransformOutput."
        )


def _require_resolved_coefficients(
    resolved_coefficients: ResolvedMessageCoefficients,
) -> None:
    if not isinstance(
        resolved_coefficients,
        ResolvedMessageCoefficients,
    ):
        raise TypeError(
            "resolved_coefficients must be a "
            "ResolvedMessageCoefficients."
        )


def _require_composition_output(
    composition_output: MessageCompositionOutput,
) -> None:
    if not isinstance(
        composition_output,
        MessageCompositionOutput,
    ):
        raise TypeError(
            "composition_output must be a "
            "MessageCompositionOutput."
        )


def _require_public_output(
    public_output: EdgeMessageOutput,
) -> None:
    if not isinstance(
        public_output,
        EdgeMessageOutput,
    ):
        raise TypeError(
            "public_output must be an EdgeMessageOutput."
        )


def _require_thresholds(
    thresholds: MessageBuilderDiagnosticThresholds,
) -> None:
    if not isinstance(
        thresholds,
        MessageBuilderDiagnosticThresholds,
    ):
        raise TypeError(
            "thresholds must be a "
            "MessageBuilderDiagnosticThresholds."
        )


def _python_scalar(
    value: torch.Tensor,
) -> float:
    return float(
        value.detach().item()
    )


def _safe_fraction(
    numerator: int,
    denominator: int,
) -> float:
    if denominator == 0:
        return 0.0

    return float(numerator) / float(denominator)


def _assert_tensor_free(
    value: Any,
    *,
    path: str = "report",
) -> None:
    """
    Ensure no tensor or module object leaks into a serialized report.
    """

    if isinstance(value, torch.Tensor):
        raise TypeError(
            f"{path} must not retain tensor-valued diagnostics."
        )

    if isinstance(value, nn.Module):
        raise TypeError(
            f"{path} must not retain module objects."
        )

    if isinstance(value, Mapping):
        for key, nested in value.items():
            _assert_tensor_free(
                nested,
                path=f"{path}.{key}",
            )
        return

    if isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _assert_tensor_free(
                nested,
                path=f"{path}[{index}]",
            )
        return

    if value is None or isinstance(
        value,
        (str, bool, int, float),
    ):
        return

    raise TypeError(
        f"{path} contains unsupported diagnostic value type "
        f"{type(value).__name__}."
    )


def _require_stage_chain(
    *,
    relation_transform: RelationTransformOutput,
    resolved_coefficients: ResolvedMessageCoefficients,
    composition_output: MessageCompositionOutput,
    source_inputs: FunctionalMessagePassingInputs | None,
) -> FunctionalMessagePassingInputs:
    _require_relation_transform(
        relation_transform
    )
    _require_resolved_coefficients(
        resolved_coefficients
    )
    _require_composition_output(
        composition_output
    )

    validate_message_builder_stage_chain(
        resolved_coefficients=resolved_coefficients,
        composition_output=composition_output,
    )

    if composition_output.relation_transform is not (
        relation_transform
    ):
        raise ValueError(
            "composition_output must preserve the exact supplied "
            "relation_transform object."
        )

    resolved_inputs = (
        relation_transform.source_inputs
    )
    _require_source_inputs(
        resolved_inputs
    )

    if resolved_coefficients.source_inputs is not (
        resolved_inputs
    ):
        raise ValueError(
            "relation_transform and resolved_coefficients must share "
            "the exact same FunctionalMessagePassingInputs object."
        )

    if composition_output.source_inputs is not (
        resolved_inputs
    ):
        raise ValueError(
            "composition_output must preserve the exact source-input "
            "lineage."
        )

    if source_inputs is not None:
        _require_source_inputs(
            source_inputs
        )

        if resolved_inputs is not source_inputs:
            raise ValueError(
                "The diagnostic stage chain must reference the exact "
                "supplied source_inputs object."
            )

    validate_edge_aligned_relation_state(
        relation_transform=relation_transform,
        source_inputs=resolved_inputs,
    )
    validate_resolved_message_coefficients(
        output=resolved_coefficients,
        source_inputs=resolved_inputs,
    )
    validate_message_composition_output(
        output=composition_output,
        relation_transform=relation_transform,
        resolved_coefficients=resolved_coefficients,
        source_inputs=resolved_inputs,
    )

    return resolved_inputs


# =============================================================================
# Scalar and vector statistics
# =============================================================================


def scalar_tensor_statistics(
    value: torch.Tensor,
    *,
    near_zero_absolute: float,
) -> dict[str, Any]:
    """
    Describe one scalar tensor using detached Python values.

    The function accepts tensors of any shape and treats all elements as one
    descriptive population. Standard deviation uses ``unbiased=False`` so
    singleton tensors remain defined.
    """

    if not isinstance(value, torch.Tensor):
        raise TypeError(
            "value must be a tensor."
        )

    if not value.dtype.is_floating_point:
        raise ValueError(
            "value must use a floating-point dtype."
        )

    if not bool(
        torch.isfinite(value)
        .all()
        .item()
    ):
        raise FloatingPointError(
            "value must contain only finite values."
        )

    if not math.isfinite(
        float(near_zero_absolute)
    ) or float(near_zero_absolute) <= 0:
        raise ValueError(
            "near_zero_absolute must be finite and strictly positive."
        )

    count = int(value.numel())

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

    detached = value.detach()

    zero_count = int(
        (detached == 0)
        .sum()
        .item()
    )
    near_zero_count = int(
        (
            detached.abs()
            <= float(near_zero_absolute)
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
        "minimum": _python_scalar(
            detached.min()
        ),
        "maximum": _python_scalar(
            detached.max()
        ),
        "mean": _python_scalar(
            detached.mean()
        ),
        "standard_deviation": _python_scalar(
            detached.std(
                unbiased=False
            )
        ),
        "mean_absolute_value": _python_scalar(
            detached.abs().mean()
        ),
        "l1_norm": _python_scalar(
            torch.linalg.vector_norm(
                detached.reshape(-1),
                ord=1,
            )
        ),
        "l2_norm": _python_scalar(
            torch.linalg.vector_norm(
                detached.reshape(-1),
                ord=2,
            )
        ),
        "zero_count": zero_count,
        "zero_fraction": _safe_fraction(
            zero_count,
            count,
        ),
        "near_zero_count": near_zero_count,
        "near_zero_fraction": _safe_fraction(
            near_zero_count,
            count,
        ),
        "positive_count": positive_count,
        "negative_count": negative_count,
        "finite": True,
    }


def edge_vector_norm_statistics(
    value: torch.Tensor,
    *,
    near_zero_absolute: float,
) -> dict[str, Any]:
    """
    Describe per-edge L2 norms for one edge-aligned matrix ``[E, H]``.
    """

    if not isinstance(value, torch.Tensor):
        raise TypeError(
            "value must be a tensor."
        )

    if value.ndim != 2:
        raise ValueError(
            "value must have rank 2 and shape [E, H]."
        )

    if not value.dtype.is_floating_point:
        raise ValueError(
            "value must use a floating-point dtype."
        )

    if not bool(
        torch.isfinite(value)
        .all()
        .item()
    ):
        raise FloatingPointError(
            "value must contain only finite values."
        )

    edge_norms = torch.linalg.vector_norm(
        value,
        ord=2,
        dim=1,
    )

    summary = scalar_tensor_statistics(
        edge_norms,
        near_zero_absolute=near_zero_absolute,
    )
    summary["edge_count"] = int(
        value.shape[0]
    )
    summary["hidden_dim"] = int(
        value.shape[1]
    )
    summary["statistic"] = (
        "per_edge_l2_norm"
    )

    return summary


def factor_statistics(
    resolved_coefficients: ResolvedMessageCoefficients,
    *,
    thresholds: MessageBuilderDiagnosticThresholds,
) -> dict[str, dict[str, Any]]:
    """
    Return one independent descriptive summary per message factor.
    """

    _require_resolved_coefficients(
        resolved_coefficients
    )
    _require_thresholds(
        thresholds
    )

    summaries: dict[str, dict[str, Any]] = {}

    for name, factor in (
        (
            MESSAGE_FACTOR_STRUCTURAL_NORMALIZATION,
            resolved_coefficients
            .structural_normalization_factor,
        ),
        (
            MESSAGE_FACTOR_RELATION_GATE,
            resolved_coefficients
            .relation_gate_factor,
        ),
        (
            MESSAGE_FACTOR_EDGE_ATTENTION,
            resolved_coefficients
            .edge_attention_factor,
        ),
        (
            MESSAGE_FACTOR_SEMANTIC_EDGE_WEIGHT,
            resolved_coefficients
            .semantic_edge_factor,
        ),
    ):
        summaries[name] = scalar_tensor_statistics(
            factor,
            near_zero_absolute=(
                thresholds.near_zero_absolute
            ),
        )
        summaries[name]["active"] = (
            name
            in resolved_coefficients
            .active_factor_names
        )
        summaries[name]["disabled_identity"] = (
            name
            in resolved_coefficients
            .disabled_factor_names
        )

    summaries["combined_coefficient"] = (
        scalar_tensor_statistics(
            resolved_coefficients
            .combined_coefficient,
            near_zero_absolute=(
                thresholds.near_zero_absolute
            ),
        )
    )

    return summaries


# =============================================================================
# Exact-relation and graph slicing
# =============================================================================


def _slice_statistics(
    *,
    mask: torch.Tensor,
    transformed_source_state: torch.Tensor,
    resolved_coefficients: ResolvedMessageCoefficients,
    edge_messages: torch.Tensor,
    thresholds: MessageBuilderDiagnosticThresholds,
) -> dict[str, Any]:
    if not isinstance(mask, torch.Tensor):
        raise TypeError(
            "mask must be a tensor."
        )

    if mask.ndim != 1:
        raise ValueError(
            "mask must have rank 1 and shape [E]."
        )

    if mask.dtype != torch.bool:
        raise ValueError(
            "mask must use boolean dtype."
        )

    expected_edges = (
        resolved_coefficients.num_edges
    )

    if tuple(mask.shape) != (
        expected_edges,
    ):
        raise ValueError(
            "mask must align exactly with the edge axis."
        )

    edge_count = int(
        mask.sum().item()
    )

    sliced_factors = {
        MESSAGE_FACTOR_STRUCTURAL_NORMALIZATION: (
            resolved_coefficients
            .structural_normalization_factor[mask]
        ),
        MESSAGE_FACTOR_RELATION_GATE: (
            resolved_coefficients
            .relation_gate_factor[mask]
        ),
        MESSAGE_FACTOR_EDGE_ATTENTION: (
            resolved_coefficients
            .edge_attention_factor[mask]
        ),
        MESSAGE_FACTOR_SEMANTIC_EDGE_WEIGHT: (
            resolved_coefficients
            .semantic_edge_factor[mask]
        ),
        "combined_coefficient": (
            resolved_coefficients
            .combined_coefficient[mask]
        ),
    }

    return {
        "edge_count": edge_count,
        "edge_fraction": _safe_fraction(
            edge_count,
            expected_edges,
        ),
        "factors": {
            name: scalar_tensor_statistics(
                factor,
                near_zero_absolute=(
                    thresholds
                    .near_zero_absolute
                ),
            )
            for name, factor
            in sliced_factors.items()
        },
        "transformed_source_state": (
            edge_vector_norm_statistics(
                transformed_source_state[mask],
                near_zero_absolute=(
                    thresholds
                    .near_zero_absolute
                ),
            )
        ),
        "edge_messages": (
            edge_vector_norm_statistics(
                edge_messages[mask],
                near_zero_absolute=(
                    thresholds
                    .near_zero_absolute
                ),
            )
        ),
    }


def exact_relation_diagnostics(
    *,
    source_inputs: FunctionalMessagePassingInputs,
    relation_transform: RelationTransformOutput,
    resolved_coefficients: ResolvedMessageCoefficients,
    composition_output: MessageCompositionOutput,
    thresholds: MessageBuilderDiagnosticThresholds = (
        DEFAULT_MESSAGE_BUILDER_DIAGNOSTIC_THRESHOLDS
    ),
) -> list[dict[str, Any]]:
    """
    Describe every exact compiled relation independently.

    The result includes relations with zero observed edges so the report
    remains aligned with the full compiled relation axis.
    """

    _require_thresholds(
        thresholds
    )
    _require_stage_chain(
        relation_transform=relation_transform,
        resolved_coefficients=resolved_coefficients,
        composition_output=composition_output,
        source_inputs=source_inputs,
    )

    relation_index = (
        source_inputs.edge_relation_index
    )
    control_relation_mask = (
        source_inputs.control_relation_mask
    )

    reports: list[dict[str, Any]] = []

    for dense_index, (
        relation_name,
        stable_relation_id,
    ) in enumerate(
        zip(
            source_inputs.relation_names,
            source_inputs.stable_relation_ids,
            strict=True,
        )
    ):
        mask = (
            relation_index
            == dense_index
        )

        report = {
            "dense_relation_index": dense_index,
            "stable_relation_id": int(
                stable_relation_id
            ),
            "relation_name": relation_name,
            "is_control_relation": bool(
                control_relation_mask[
                    dense_index
                ].item()
            ),
        }
        report.update(
            _slice_statistics(
                mask=mask,
                transformed_source_state=(
                    relation_transform
                    .transformed_source_state
                ),
                resolved_coefficients=(
                    resolved_coefficients
                ),
                edge_messages=(
                    composition_output
                    .edge_messages
                ),
                thresholds=thresholds,
            )
        )
        reports.append(report)

    return reports


def graph_batch_diagnostics(
    *,
    source_inputs: FunctionalMessagePassingInputs,
    relation_transform: RelationTransformOutput,
    resolved_coefficients: ResolvedMessageCoefficients,
    composition_output: MessageCompositionOutput,
    thresholds: MessageBuilderDiagnosticThresholds = (
        DEFAULT_MESSAGE_BUILDER_DIAGNOSTIC_THRESHOLDS
    ),
) -> list[dict[str, Any]]:
    """
    Describe every graph in a batched input independently.
    """

    _require_thresholds(
        thresholds
    )
    _require_stage_chain(
        relation_transform=relation_transform,
        resolved_coefficients=resolved_coefficients,
        composition_output=composition_output,
        source_inputs=source_inputs,
    )

    reports: list[dict[str, Any]] = []

    for graph_index in range(
        source_inputs.num_graphs
    ):
        edge_mask = (
            source_inputs.edge_batch_index
            == graph_index
        )
        node_count = int(
            (
                source_inputs
                .node_batch_index
                == graph_index
            )
            .sum()
            .item()
        )

        report = {
            "graph_index": graph_index,
            "node_count": node_count,
        }
        report.update(
            _slice_statistics(
                mask=edge_mask,
                transformed_source_state=(
                    relation_transform
                    .transformed_source_state
                ),
                resolved_coefficients=(
                    resolved_coefficients
                ),
                edge_messages=(
                    composition_output
                    .edge_messages
                ),
                thresholds=thresholds,
            )
        )
        reports.append(report)

    return reports


# =============================================================================
# Lineage and alert derivation
# =============================================================================


def message_builder_lineage_summary(
    *,
    relation_transform: RelationTransformOutput,
    resolved_coefficients: ResolvedMessageCoefficients,
    composition_output: MessageCompositionOutput,
) -> dict[str, Any]:
    """
    Return exact stage provenance without retaining source objects.
    """

    source_inputs = _require_stage_chain(
        relation_transform=relation_transform,
        resolved_coefficients=resolved_coefficients,
        composition_output=composition_output,
        source_inputs=None,
    )

    return {
        "source_inputs_lineage_fingerprint": (
            source_inputs.lineage_fingerprint()
        ),
        "source_fingerprint": (
            source_inputs.source_fingerprint
        ),
        "compiled_relation_registry_fingerprint": (
            source_inputs
            .compiled_relation_registry
            .fingerprint()
        ),
        "relation_transform": {
            "schema_version": (
                relation_transform.schema_version
            ),
            "transform_mode": (
                relation_transform.transform_mode
            ),
            "architecture_fingerprint": (
                relation_transform
                .encoder_architecture_fingerprint
            ),
            "parameter_fingerprint": (
                relation_transform
                .parameter_fingerprint
            ),
        },
        "coefficient_resolution": {
            "schema_version": (
                resolved_coefficients
                .schema_version
            ),
            "architecture_fingerprint": (
                resolved_coefficients
                .architecture_fingerprint()
            ),
            "resolver_architecture_fingerprint": (
                resolved_coefficients
                .resolver_architecture_fingerprint
            ),
            "lineage_fingerprint": (
                resolved_coefficients
                .lineage_fingerprint()
            ),
            "value_fingerprint": (
                resolved_coefficients
                .value_fingerprint()
            ),
            "parameter_fingerprint": None,
        },
        "message_composition": {
            "schema_version": (
                composition_output
                .schema_version
            ),
            "architecture_fingerprint": (
                composition_output
                .architecture_fingerprint()
            ),
            "composer_architecture_fingerprint": (
                composition_output
                .composer_architecture_fingerprint
            ),
            "lineage_fingerprint": (
                composition_output
                .lineage_fingerprint()
            ),
            "value_fingerprint": (
                composition_output
                .value_fingerprint()
            ),
            "parameter_fingerprint": None,
        },
        "exact_object_identity": {
            "relation_transform_preserved": (
                composition_output
                .relation_transform
                is relation_transform
            ),
            "resolved_coefficients_preserved": (
                composition_output
                .resolved_coefficients
                is resolved_coefficients
            ),
            "source_inputs_preserved": (
                composition_output
                .source_inputs
                is source_inputs
            ),
            "transformed_state_tensor_preserved": (
                composition_output
                .relation_transform
                .transformed_source_state
                is relation_transform
                .transformed_source_state
            ),
            "combined_coefficient_tensor_preserved": (
                composition_output
                .combined_coefficient
                is resolved_coefficients
                .combined_coefficient
            ),
        },
    }


def derive_message_builder_alerts(
    *,
    resolved_coefficients: ResolvedMessageCoefficients,
    composition_output: MessageCompositionOutput,
    thresholds: MessageBuilderDiagnosticThresholds = (
        DEFAULT_MESSAGE_BUILDER_DIAGNOSTIC_THRESHOLDS
    ),
) -> list[dict[str, Any]]:
    """
    Derive bounded descriptive flags.

    Alerts are deliberately phrased as observations, never diagnoses or causal
    conclusions.
    """

    _require_resolved_coefficients(
        resolved_coefficients
    )
    _require_composition_output(
        composition_output
    )
    _require_thresholds(
        thresholds
    )

    if composition_output.resolved_coefficients is not (
        resolved_coefficients
    ):
        raise ValueError(
            "composition_output must preserve the exact supplied "
            "resolved_coefficients object."
        )

    alerts: list[dict[str, Any]] = []

    num_edges = resolved_coefficients.num_edges

    if num_edges == 0:
        alerts.append(
            {
                "code": "empty_edge_set",
                "severity": "info",
                "observation": (
                    "The message-builder stage received zero edges."
                ),
            }
        )
        return alerts

    semantic_negative_count = int(
        (
            resolved_coefficients
            .semantic_edge_factor
            < 0
        )
        .sum()
        .item()
    )

    if semantic_negative_count > 0:
        alerts.append(
            {
                "code": "signed_semantic_edge_factor",
                "severity": "info",
                "count": semantic_negative_count,
                "fraction": _safe_fraction(
                    semantic_negative_count,
                    num_edges,
                ),
                "observation": (
                    "Some consumed semantic edge factors are negative; "
                    "message directions may therefore be sign-reversed."
                ),
            }
        )

    combined = (
        resolved_coefficients
        .combined_coefficient
        .detach()
    )
    absolute_combined = (
        combined.abs()
    )

    zero_count = int(
        (combined == 0)
        .sum()
        .item()
    )

    if zero_count > 0:
        alerts.append(
            {
                "code": "zero_combined_coefficient",
                "severity": "info",
                "count": zero_count,
                "fraction": _safe_fraction(
                    zero_count,
                    num_edges,
                ),
                "observation": (
                    "Some edge messages are deterministically zeroed by "
                    "their combined scalar coefficient."
                ),
            }
        )

    near_zero_count = int(
        (
            absolute_combined
            <= thresholds.near_zero_absolute
        )
        .sum()
        .item()
    )
    near_zero_fraction = _safe_fraction(
        near_zero_count,
        num_edges,
    )

    if near_zero_fraction >= (
        thresholds.high_near_zero_fraction
    ):
        alerts.append(
            {
                "code": "high_near_zero_coefficient_fraction",
                "severity": "warning",
                "count": near_zero_count,
                "fraction": near_zero_fraction,
                "threshold": (
                    thresholds
                    .high_near_zero_fraction
                ),
                "observation": (
                    "A high fraction of combined edge coefficients are "
                    "near zero under the configured descriptive threshold."
                ),
            }
        )

    large_coefficient_count = int(
        (
            absolute_combined
            >= thresholds
            .large_absolute_coefficient
        )
        .sum()
        .item()
    )

    if large_coefficient_count > 0:
        alerts.append(
            {
                "code": "large_absolute_coefficient",
                "severity": "warning",
                "count": large_coefficient_count,
                "fraction": _safe_fraction(
                    large_coefficient_count,
                    num_edges,
                ),
                "threshold": (
                    thresholds
                    .large_absolute_coefficient
                ),
                "observation": (
                    "Some combined edge coefficients exceed the configured "
                    "absolute-magnitude threshold."
                ),
            }
        )

    message_norms = torch.linalg.vector_norm(
        composition_output
        .edge_messages
        .detach(),
        ord=2,
        dim=1,
    )
    zero_message_count = int(
        (
            message_norms
            <= thresholds.near_zero_absolute
        )
        .sum()
        .item()
    )
    zero_message_fraction = _safe_fraction(
        zero_message_count,
        num_edges,
    )

    if zero_message_fraction >= (
        thresholds.high_zero_message_fraction
    ):
        alerts.append(
            {
                "code": "high_zero_message_fraction",
                "severity": "warning",
                "count": zero_message_count,
                "fraction": zero_message_fraction,
                "threshold": (
                    thresholds
                    .high_zero_message_fraction
                ),
                "observation": (
                    "A high fraction of edge-message vectors have near-zero "
                    "L2 norm under the configured descriptive threshold."
                ),
            }
        )

    large_message_count = int(
        (
            message_norms
            >= thresholds
            .large_message_l2_norm
        )
        .sum()
        .item()
    )

    if large_message_count > 0:
        alerts.append(
            {
                "code": "large_message_l2_norm",
                "severity": "warning",
                "count": large_message_count,
                "fraction": _safe_fraction(
                    large_message_count,
                    num_edges,
                ),
                "threshold": (
                    thresholds
                    .large_message_l2_norm
                ),
                "observation": (
                    "Some edge-message vectors exceed the configured L2-norm "
                    "threshold."
                ),
            }
        )

    return alerts


# =============================================================================
# Complete reports
# =============================================================================


def message_builder_diagnostics_architecture_dict(
    *,
    include_per_relation: bool,
    include_per_graph: bool,
    thresholds: MessageBuilderDiagnosticThresholds,
) -> dict[str, Any]:
    _require_thresholds(
        thresholds
    )

    if not isinstance(
        include_per_relation,
        bool,
    ):
        raise TypeError(
            "include_per_relation must be boolean."
        )

    if not isinstance(
        include_per_graph,
        bool,
    ):
        raise TypeError(
            "include_per_graph must be boolean."
        )

    return {
        "schema_version": (
            MESSAGE_BUILDER_DIAGNOSTICS_SCHEMA_VERSION
        ),
        "interpretation": (
            MESSAGE_BUILDER_DIAGNOSTICS_INTERPRETATION
        ),
        "operation_order": list(
            MESSAGE_BUILDER_DIAGNOSTICS_OPERATION_ORDER
        ),
        "required_sections": list(
            MESSAGE_BUILDER_DIAGNOSTIC_REQUIRED_SECTIONS
        ),
        "include_per_relation": include_per_relation,
        "include_per_graph": include_per_graph,
        "thresholds": (
            thresholds.architecture_dict()
        ),
        "parameter_free": (
            MESSAGE_BUILDER_DIAGNOSTICS_PARAMETER_FREE
        ),
        "buffer_free": (
            MESSAGE_BUILDER_DIAGNOSTICS_BUFFER_FREE
        ),
        "retains_tensors": False,
        "retains_source_objects": False,
        "claims_causal_importance": False,
        "claims_explanation_faithfulness": False,
        "claims_uncertainty_calibration": False,
    }


def message_builder_diagnostics_architecture_fingerprint(
    *,
    include_per_relation: bool,
    include_per_graph: bool,
    thresholds: MessageBuilderDiagnosticThresholds,
) -> str:
    return _fingerprint(
        message_builder_diagnostics_architecture_dict(
            include_per_relation=(
                include_per_relation
            ),
            include_per_graph=(
                include_per_graph
            ),
            thresholds=thresholds,
        )
    )


def build_message_builder_diagnostic_report(
    *,
    relation_transform: RelationTransformOutput,
    resolved_coefficients: ResolvedMessageCoefficients,
    composition_output: MessageCompositionOutput,
    source_inputs: FunctionalMessagePassingInputs | None = None,
    include_per_relation: bool = True,
    include_per_graph: bool = True,
    thresholds: MessageBuilderDiagnosticThresholds = (
        DEFAULT_MESSAGE_BUILDER_DIAGNOSTIC_THRESHOLDS
    ),
) -> dict[str, Any]:
    """
    Build one complete internal-stage diagnostic report.
    """

    _require_thresholds(
        thresholds
    )

    resolved_inputs = _require_stage_chain(
        relation_transform=relation_transform,
        resolved_coefficients=resolved_coefficients,
        composition_output=composition_output,
        source_inputs=source_inputs,
    )

    architecture = (
        message_builder_diagnostics_architecture_dict(
            include_per_relation=(
                include_per_relation
            ),
            include_per_graph=(
                include_per_graph
            ),
            thresholds=thresholds,
        )
    )

    stage_summaries = {
        "relation_state_gather": (
            relation_state_gather_diagnostic_summary(
                relation_transform=(
                    relation_transform
                ),
                source_inputs=resolved_inputs,
            )
        ),
        "coefficient_resolution": (
            message_coefficient_diagnostic_summary(
                resolved_coefficients
            )
        ),
        "message_composition": (
            message_composition_diagnostic_summary(
                composition_output
            )
        ),
    }

    transformed_state = (
        relation_transform
        .transformed_source_state
    )
    edge_messages = (
        composition_output.edge_messages
    )

    global_summary = {
        "num_nodes": resolved_inputs.num_nodes,
        "num_edges": resolved_inputs.num_edges,
        "num_graphs": resolved_inputs.num_graphs,
        "hidden_dim": resolved_inputs.hidden_dim,
        "num_exact_relations": (
            resolved_inputs.num_relations
        ),
        "dtype": str(
            composition_output.dtype
        ),
        "device": str(
            composition_output.device
        ),
        "relation_gate_enabled": (
            resolved_coefficients
            .relation_gate_enabled
        ),
        "edge_attention_enabled": (
            resolved_coefficients
            .edge_attention_enabled
        ),
        "semantic_edge_weight_enabled": (
            resolved_coefficients
            .semantic_edge_weight_enabled
        ),
        "active_factor_names": list(
            resolved_coefficients
            .active_factor_names
        ),
        "disabled_factor_names": list(
            resolved_coefficients
            .disabled_factor_names
        ),
        "factors": factor_statistics(
            resolved_coefficients,
            thresholds=thresholds,
        ),
        "transformed_source_state": (
            edge_vector_norm_statistics(
                transformed_state,
                near_zero_absolute=(
                    thresholds
                    .near_zero_absolute
                ),
            )
        ),
        "edge_messages": (
            edge_vector_norm_statistics(
                edge_messages,
                near_zero_absolute=(
                    thresholds
                    .near_zero_absolute
                ),
            )
        ),
    }

    relation_reports = (
        exact_relation_diagnostics(
            source_inputs=resolved_inputs,
            relation_transform=relation_transform,
            resolved_coefficients=(
                resolved_coefficients
            ),
            composition_output=composition_output,
            thresholds=thresholds,
        )
        if include_per_relation
        else []
    )

    graph_reports = (
        graph_batch_diagnostics(
            source_inputs=resolved_inputs,
            relation_transform=relation_transform,
            resolved_coefficients=(
                resolved_coefficients
            ),
            composition_output=composition_output,
            thresholds=thresholds,
        )
        if include_per_graph
        else []
    )

    report: dict[str, Any] = {
        "schema_version": (
            MESSAGE_BUILDER_DIAGNOSTICS_SCHEMA_VERSION
        ),
        "interpretation": (
            MESSAGE_BUILDER_DIAGNOSTICS_INTERPRETATION
        ),
        "diagnostics_architecture": architecture,
        "diagnostics_architecture_fingerprint": (
            message_builder_diagnostics_architecture_fingerprint(
                include_per_relation=(
                    include_per_relation
                ),
                include_per_graph=(
                    include_per_graph
                ),
                thresholds=thresholds,
            )
        ),
        "upstream_schema_versions": {
            "relation_state_gather": (
                RELATION_STATE_GATHER_SCHEMA_VERSION
            ),
            "coefficient_resolution": (
                MESSAGE_COEFFICIENT_RESOLUTION_SCHEMA_VERSION
            ),
            "message_composition": (
                MESSAGE_COMPOSER_SCHEMA_VERSION
            ),
        },
        MESSAGE_BUILDER_DIAGNOSTIC_SECTION_STAGE_SUMMARIES: (
            stage_summaries
        ),
        MESSAGE_BUILDER_DIAGNOSTIC_SECTION_GLOBAL: (
            global_summary
        ),
        MESSAGE_BUILDER_DIAGNOSTIC_SECTION_BY_RELATION: (
            relation_reports
        ),
        MESSAGE_BUILDER_DIAGNOSTIC_SECTION_BY_GRAPH: (
            graph_reports
        ),
        MESSAGE_BUILDER_DIAGNOSTIC_SECTION_LINEAGE: (
            message_builder_lineage_summary(
                relation_transform=(
                    relation_transform
                ),
                resolved_coefficients=(
                    resolved_coefficients
                ),
                composition_output=(
                    composition_output
                ),
            )
        ),
        MESSAGE_BUILDER_DIAGNOSTIC_SECTION_ALERTS: (
            derive_message_builder_alerts(
                resolved_coefficients=(
                    resolved_coefficients
                ),
                composition_output=(
                    composition_output
                ),
                thresholds=thresholds,
            )
        ),
        "scientific_claims": {
            "causal_importance": False,
            "explanation_faithfulness": False,
            "uncertainty_calibration": False,
            "mechanistic_identifiability": False,
            "relation_necessity": False,
        },
    }

    _assert_tensor_free(
        report
    )

    report["report_fingerprint"] = (
        diagnostic_report_fingerprint(
            report
        )
    )

    validate_message_builder_diagnostic_report(
        report,
        expected_num_relations=(
            resolved_inputs.num_relations
            if include_per_relation
            else 0
        ),
        expected_num_graphs=(
            resolved_inputs.num_graphs
            if include_per_graph
            else 0
        ),
    )

    return report


def build_public_edge_message_diagnostic_report(
    *,
    public_output: EdgeMessageOutput,
    composition_output: MessageCompositionOutput,
    include_per_relation: bool = True,
    include_per_graph: bool = True,
    thresholds: MessageBuilderDiagnosticThresholds = (
        DEFAULT_MESSAGE_BUILDER_DIAGNOSTIC_THRESHOLDS
    ),
) -> dict[str, Any]:
    """
    Build a report while validating exact compatibility with the public output.
    """

    _require_public_output(
        public_output
    )
    _require_composition_output(
        composition_output
    )

    validate_public_edge_message_output(
        public_output=public_output,
        composition_output=composition_output,
    )

    report = build_message_builder_diagnostic_report(
        relation_transform=(
            composition_output
            .relation_transform
        ),
        resolved_coefficients=(
            composition_output
            .resolved_coefficients
        ),
        composition_output=(
            composition_output
        ),
        source_inputs=(
            composition_output
            .source_inputs
        ),
        include_per_relation=(
            include_per_relation
        ),
        include_per_graph=(
            include_per_graph
        ),
        thresholds=thresholds,
    )

    report["public_output"] = {
        "schema_version": (
            public_output.schema_version
        ),
        "encoder_architecture_fingerprint": (
            public_output
            .encoder_architecture_fingerprint
        ),
        "exact_edge_messages_tensor_preserved": (
            public_output.edge_messages
            is composition_output.edge_messages
        ),
        "exact_relation_transform_preserved": (
            public_output.relation_transform
            is composition_output
            .relation_transform
        ),
        "exact_edge_normalization_preserved": (
            public_output.edge_normalization
            is composition_output
            .resolved_coefficients
            .edge_normalization
        ),
        "exact_relation_gate_preserved": (
            public_output.relation_gate
            is composition_output
            .resolved_coefficients
            .relation_gate
        ),
        "exact_edge_attention_preserved": (
            public_output.edge_attention
            is composition_output
            .resolved_coefficients
            .edge_attention
        ),
        "exact_semantic_edge_weight_preserved": (
            public_output.semantic_edge_weight
            is composition_output
            .resolved_coefficients
            .semantic_edge_weight
        ),
    }

    report.pop(
        "report_fingerprint",
        None,
    )
    _assert_tensor_free(
        report
    )
    report["report_fingerprint"] = (
        diagnostic_report_fingerprint(
            report
        )
    )

    return report


# =============================================================================
# Report validation and fingerprinting
# =============================================================================


def diagnostic_report_fingerprint(
    report: Mapping[str, Any],
) -> str:
    """
    Fingerprint a tensor-free diagnostic report.

    An existing ``report_fingerprint`` field is excluded to avoid recursive
    self-reference.
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
        for key, value in report.items()
        if key != "report_fingerprint"
    }

    _assert_tensor_free(
        payload
    )

    return _fingerprint(
        payload
    )


def validate_message_builder_diagnostic_report(
    report: Mapping[str, Any],
    *,
    expected_num_relations: int | None = None,
    expected_num_graphs: int | None = None,
) -> None:
    """
    Validate structural completeness and serialization safety.
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

    if report.get("schema_version") != (
        MESSAGE_BUILDER_DIAGNOSTICS_SCHEMA_VERSION
    ):
        raise ValueError(
            "Diagnostic report schema version is missing or unexpected."
        )

    if report.get("interpretation") != (
        MESSAGE_BUILDER_DIAGNOSTICS_INTERPRETATION
    ):
        raise ValueError(
            "Diagnostic report interpretation is missing or unexpected."
        )

    for section in (
        MESSAGE_BUILDER_DIAGNOSTIC_REQUIRED_SECTIONS
    ):
        if section not in report:
            raise ValueError(
                f"Diagnostic report is missing required section "
                f"{section!r}."
            )

    if not isinstance(
        report[
            MESSAGE_BUILDER_DIAGNOSTIC_SECTION_BY_RELATION
        ],
        list,
    ):
        raise TypeError(
            "by_exact_relation must be a list."
        )

    if not isinstance(
        report[
            MESSAGE_BUILDER_DIAGNOSTIC_SECTION_BY_GRAPH
        ],
        list,
    ):
        raise TypeError(
            "by_graph must be a list."
        )

    if expected_num_relations is not None:
        if not isinstance(
            expected_num_relations,
            int,
        ) or expected_num_relations < 0:
            raise ValueError(
                "expected_num_relations must be a nonnegative integer."
            )

        observed_relations = len(
            report[
                MESSAGE_BUILDER_DIAGNOSTIC_SECTION_BY_RELATION
            ]
        )

        if observed_relations != (
            expected_num_relations
        ):
            raise ValueError(
                "Diagnostic report exact-relation count differs from "
                "the expected count."
            )

    if expected_num_graphs is not None:
        if not isinstance(
            expected_num_graphs,
            int,
        ) or expected_num_graphs < 0:
            raise ValueError(
                "expected_num_graphs must be a nonnegative integer."
            )

        observed_graphs = len(
            report[
                MESSAGE_BUILDER_DIAGNOSTIC_SECTION_BY_GRAPH
            ]
        )

        if observed_graphs != (
            expected_num_graphs
        ):
            raise ValueError(
                "Diagnostic report graph count differs from the "
                "expected count."
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

    prohibited_true_claims = [
        name
        for name, value
        in scientific_claims.items()
        if value is not False
    ]

    if prohibited_true_claims:
        raise ValueError(
            "Message-builder diagnostics must not assert unsupported "
            "scientific claims. Non-false entries: "
            f"{prohibited_true_claims}."
        )

    if "report_fingerprint" in report:
        fingerprint = report[
            "report_fingerprint"
        ]
        _require_nonempty_string(
            "report_fingerprint",
            fingerprint,
        )

        expected = diagnostic_report_fingerprint(
            report
        )

        if fingerprint != expected:
            raise ValueError(
                "Diagnostic report fingerprint does not match report "
                "contents."
            )


# =============================================================================
# Parameter-free module wrapper
# =============================================================================


class MessageBuilderDiagnostics(nn.Module):
    """
    Parameter-free diagnostics orchestrator.

    Parameters
    ----------
    include_per_relation:
        Include one report entry for every exact compiled relation.
    include_per_graph:
        Include one report entry for every graph in the input batch.
    thresholds:
        Descriptive alert thresholds.
    """

    include_per_relation: bool
    include_per_graph: bool
    thresholds: MessageBuilderDiagnosticThresholds

    def __init__(
        self,
        *,
        include_per_relation: bool = True,
        include_per_graph: bool = True,
        thresholds: MessageBuilderDiagnosticThresholds = (
            DEFAULT_MESSAGE_BUILDER_DIAGNOSTIC_THRESHOLDS
        ),
    ) -> None:
        super().__init__()

        if not isinstance(
            include_per_relation,
            bool,
        ):
            raise TypeError(
                "include_per_relation must be boolean."
            )

        if not isinstance(
            include_per_graph,
            bool,
        ):
            raise TypeError(
                "include_per_graph must be boolean."
            )

        _require_thresholds(
            thresholds
        )

        self.include_per_relation = (
            include_per_relation
        )
        self.include_per_graph = (
            include_per_graph
        )
        self.thresholds = thresholds

        self.assert_parameter_free()

    @property
    def parameter_count(self) -> int:
        return sum(
            int(parameter.numel())
            for parameter in self.parameters()
        )

    @property
    def trainable_parameter_count(self) -> int:
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
            message_builder_diagnostics_architecture_dict(
                include_per_relation=(
                    self.include_per_relation
                ),
                include_per_graph=(
                    self.include_per_graph
                ),
                thresholds=self.thresholds,
            )
        )

    def architecture_fingerprint(
        self,
    ) -> str:
        return (
            message_builder_diagnostics_architecture_fingerprint(
                include_per_relation=(
                    self.include_per_relation
                ),
                include_per_graph=(
                    self.include_per_graph
                ),
                thresholds=self.thresholds,
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
                "MessageBuilderDiagnostics must remain parameter-free."
            )

        if buffers:
            raise RuntimeError(
                "MessageBuilderDiagnostics must remain buffer-free."
            )

        if self.state_dict():
            raise RuntimeError(
                "MessageBuilderDiagnostics must have an empty state_dict."
            )

        if self.parameter_count != 0:
            raise RuntimeError(
                "MessageBuilderDiagnostics parameter_count must be zero."
            )

        if self.trainable_parameter_count != 0:
            raise RuntimeError(
                "MessageBuilderDiagnostics trainable_parameter_count must "
                "be zero."
            )

        if self.buffer_count != 0:
            raise RuntimeError(
                "MessageBuilderDiagnostics buffer_count must be zero."
            )

    def report(
        self,
        *,
        relation_transform: RelationTransformOutput,
        resolved_coefficients: ResolvedMessageCoefficients,
        composition_output: MessageCompositionOutput,
        source_inputs: FunctionalMessagePassingInputs | None = None,
    ) -> dict[str, Any]:
        """
        Build one complete internal-stage report.
        """

        self.assert_parameter_free()

        report = (
            build_message_builder_diagnostic_report(
                relation_transform=(
                    relation_transform
                ),
                resolved_coefficients=(
                    resolved_coefficients
                ),
                composition_output=(
                    composition_output
                ),
                source_inputs=source_inputs,
                include_per_relation=(
                    self.include_per_relation
                ),
                include_per_graph=(
                    self.include_per_graph
                ),
                thresholds=self.thresholds,
            )
        )

        if report[
            "diagnostics_architecture_fingerprint"
        ] != self.architecture_fingerprint():
            raise RuntimeError(
                "Diagnostic report architecture fingerprint differs from "
                "the diagnostics module."
            )

        return report

    def public_report(
        self,
        *,
        public_output: EdgeMessageOutput,
        composition_output: MessageCompositionOutput,
    ) -> dict[str, Any]:
        """
        Build a report anchored to the final public edge-message output.
        """

        self.assert_parameter_free()

        report = (
            build_public_edge_message_diagnostic_report(
                public_output=public_output,
                composition_output=(
                    composition_output
                ),
                include_per_relation=(
                    self.include_per_relation
                ),
                include_per_graph=(
                    self.include_per_graph
                ),
                thresholds=self.thresholds,
            )
        )

        if report[
            "diagnostics_architecture_fingerprint"
        ] != self.architecture_fingerprint():
            raise RuntimeError(
                "Public diagnostic report architecture fingerprint differs "
                "from the diagnostics module."
            )

        return report

    def forward(
        self,
        *,
        relation_transform: RelationTransformOutput,
        resolved_coefficients: ResolvedMessageCoefficients,
        composition_output: MessageCompositionOutput,
        source_inputs: FunctionalMessagePassingInputs | None = None,
    ) -> dict[str, Any]:
        return self.report(
            relation_transform=(
                relation_transform
            ),
            resolved_coefficients=(
                resolved_coefficients
            ),
            composition_output=(
                composition_output
            ),
            source_inputs=source_inputs,
        )

    def extra_repr(self) -> str:
        return (
            f"include_per_relation={self.include_per_relation}, "
            f"include_per_graph={self.include_per_graph}, "
            "retains_tensors=False, "
            "parameter_free=True"
        )


# =============================================================================
# Builders and aliases
# =============================================================================


def build_message_builder_diagnostics(
    *,
    include_per_relation: bool = True,
    include_per_graph: bool = True,
    thresholds: MessageBuilderDiagnosticThresholds = (
        DEFAULT_MESSAGE_BUILDER_DIAGNOSTIC_THRESHOLDS
    ),
) -> MessageBuilderDiagnostics:
    """
    Construct the parameter-free diagnostics orchestrator.
    """

    return MessageBuilderDiagnostics(
        include_per_relation=(
            include_per_relation
        ),
        include_per_graph=(
            include_per_graph
        ),
        thresholds=thresholds,
    )


MessageDiagnostics = MessageBuilderDiagnostics
EdgeMessageDiagnostics = MessageBuilderDiagnostics
build_edge_message_diagnostics = (
    build_message_builder_diagnostics
)
message_builder_diagnostic_report = (
    build_message_builder_diagnostic_report
)
public_edge_message_diagnostic_report = (
    build_public_edge_message_diagnostic_report
)


__all__ = (
    # Public identity.
    "MESSAGE_BUILDER_DIAGNOSTICS_SCHEMA_VERSION",
    "MESSAGE_BUILDER_DIAGNOSTICS_INTERPRETATION",
    "MESSAGE_BUILDER_DIAGNOSTICS_OPERATION_ORDER",
    "MESSAGE_BUILDER_DIAGNOSTICS_PARAMETER_FREE",
    "MESSAGE_BUILDER_DIAGNOSTICS_BUFFER_FREE",
    "MESSAGE_BUILDER_DIAGNOSTIC_SECTION_STAGE_SUMMARIES",
    "MESSAGE_BUILDER_DIAGNOSTIC_SECTION_GLOBAL",
    "MESSAGE_BUILDER_DIAGNOSTIC_SECTION_BY_RELATION",
    "MESSAGE_BUILDER_DIAGNOSTIC_SECTION_BY_GRAPH",
    "MESSAGE_BUILDER_DIAGNOSTIC_SECTION_LINEAGE",
    "MESSAGE_BUILDER_DIAGNOSTIC_SECTION_ALERTS",
    "MESSAGE_BUILDER_DIAGNOSTIC_REQUIRED_SECTIONS",
    # Configuration.
    "MessageBuilderDiagnosticThresholds",
    "DEFAULT_MESSAGE_BUILDER_DIAGNOSTIC_THRESHOLDS",
    # Statistics and slicing.
    "scalar_tensor_statistics",
    "edge_vector_norm_statistics",
    "factor_statistics",
    "exact_relation_diagnostics",
    "graph_batch_diagnostics",
    # Lineage and alerts.
    "message_builder_lineage_summary",
    "derive_message_builder_alerts",
    # Complete reports.
    "message_builder_diagnostics_architecture_dict",
    "message_builder_diagnostics_architecture_fingerprint",
    "build_message_builder_diagnostic_report",
    "message_builder_diagnostic_report",
    "build_public_edge_message_diagnostic_report",
    "public_edge_message_diagnostic_report",
    "diagnostic_report_fingerprint",
    "validate_message_builder_diagnostic_report",
    # Module API.
    "MessageBuilderDiagnostics",
    "MessageDiagnostics",
    "EdgeMessageDiagnostics",
    "build_message_builder_diagnostics",
    "build_edge_message_diagnostics",
)
