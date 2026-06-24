"""
Explicit resolution of scalar edge-message coefficients.

Target path:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            functional_message_passing/
                message_builders/
                    coefficient_resolution.py

This module resolves every scalar edge coefficient used by message
composition while preserving the identity and exact source lineage
of each factor.

For every stored directed edge ``e``:

    n_e
        = structural edge-normalization coefficient

    g_e
        = edge-aligned relation-gate coefficient, or exact one when relation
          gating is disabled

    alpha_e
        = final reduced edge-attention coefficient, or exact one when edge
          attention is disabled

    w_e
        = source-graph semantic edge coefficient when explicitly consumed,
          otherwise exact one

The combined coefficient is:

    c_e = n_e * g_e * alpha_e * w_e

The later message-composition stage applies:

    m_e = transformed_source_state[e] * c_e

Separation
---------------------
The four factors remain distinct because they encode different mechanisms:

``structural normalization``
    Graph-structural scaling.

``relation gate``
    Target-node activation of one exact relation mechanism.

``edge attention``
    Within-relation routing among concrete incoming edges.

``semantic edge weight``
    Externally supplied data coefficient.

This module does not reinterpret one factor as another and does not
renormalize any upstream output.

Disabled mechanisms
-------------------
Disabled relation gating and disabled edge attention are represented upstream
by ``None`` and resolve here to exact all-one tensors:

    disabled mechanism -> multiplicative identity one

Therefore:

    disabled edge attention != enabled uniform edge attention

Enabled uniform attention may assign reciprocal group-size weights. Disabled
attention contributes one.

Semantic-edge policy
--------------------
Two bounded policies are supported:

``ignore``
    Do not consume ``source_graph.semantic_edge_weight`` even when the field is
    present. Resolve the semantic factor to exact one and record
    ``semantic_edge_weight=None`` in the output.

``use_source_graph``
    Consume the exact source-graph semantic-edge tensor. Clones, overrides,
    casts, and independently reconstructed equivalent tensors are not accepted.

The policy is an architecture choice and is included in architecture
fingerprints.

Lineage and tensor identity
---------------------------
Enabled factors preserve exact tensor identity:

    structural factor
        is edge_normalization.coefficients

    relation-gate factor
        is relation_gate.edge_gate_values

    edge-attention factor
        is edge_attention.edge_weights

    semantic factor
        is source_inputs.source_graph.semantic_edge_weight

Identity factors for disabled mechanisms are newly allocated all-one tensors
with the same edge shape, dtype, and device as structural normalization.

The combined coefficient is the only new numerical tensor produced by this
module.

Scope exclusions
----------------
This module does not own:

- relation transforms;
- source-state gathering;
- structural-normalization computation;
- relation-gate prediction or activation;
- edge-attention scoring, normalization, or head reduction;
- semantic-weight estimation;
- message-vector composition;
- target-node aggregation;
- residual updates;
- causal or explanation claims.

The resolver is parameter-free and buffer-free.
"""

from __future__ import annotations

from hashlib import sha256
import json
from typing import Any, Final, Mapping, NamedTuple

import torch
from torch import nn

from ..schemas import (
    EdgeAttentionOutput,
    FunctionalMessagePassingInputs,
    RelationGateOutput,
    StructuralEdgeNormalizationOutput,
)
from .schemas import (
    CANONICAL_MESSAGE_SEMANTIC_EDGE_POLICIES,
    IMPLEMENTED_MESSAGE_SEMANTIC_EDGE_POLICIES,
    MESSAGE_COMBINED_COEFFICIENT_FORMULA,
    MESSAGE_DISABLED_FACTOR_POLICY,
    MESSAGE_FACTOR_EDGE_ATTENTION,
    MESSAGE_FACTOR_ORDER,
    MESSAGE_FACTOR_RELATION_GATE,
    MESSAGE_FACTOR_SEMANTIC_EDGE_WEIGHT,
    MESSAGE_FACTOR_STRUCTURAL_NORMALIZATION,
    MESSAGE_SEMANTIC_EDGE_POLICY_IGNORE,
    MESSAGE_SEMANTIC_EDGE_POLICY_USE_SOURCE_GRAPH,
    ResolvedMessageCoefficients,
)


# =============================================================================
# Public identity
# =============================================================================


MESSAGE_COEFFICIENT_RESOLUTION_SCHEMA_VERSION: Final[str] = "0.1"

MESSAGE_COEFFICIENT_RESOLUTION_OPERATION_ORDER: Final[
    tuple[str, ...]
] = (
    "validate_exact_source_input_lineage",
    "preserve_structural_normalization_tensor",
    "resolve_relation_gate_or_exact_identity_one",
    "resolve_edge_attention_or_exact_identity_one",
    "resolve_semantic_edge_policy",
    "multiply_factors_in_frozen_order",
    "construct_resolved_message_coefficients",
)

MESSAGE_COEFFICIENT_RESOLUTION_INTERPRETATION: Final[str] = (
    "explicit_multiplicative_edge_message_scaling"
)

MESSAGE_COEFFICIENT_RESOLUTION_PARAMETER_FREE: Final[bool] = True
MESSAGE_COEFFICIENT_RESOLUTION_BUFFER_FREE: Final[bool] = True


# =============================================================================
# Small internal return contracts
# =============================================================================


class OptionalFactorResolution(NamedTuple):
    """
    Resolved tensor plus the optional source object retained in metadata.
    """

    factor: torch.Tensor
    source: RelationGateOutput | EdgeAttentionOutput | None


class SemanticFactorResolution(NamedTuple):
    """
    Resolved semantic factor and exact consumed source tensor, if any.
    """

    factor: torch.Tensor
    source_weight: torch.Tensor | None
    policy: str


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


def _require_edge_normalization(
    edge_normalization: StructuralEdgeNormalizationOutput,
) -> None:
    if not isinstance(
        edge_normalization,
        StructuralEdgeNormalizationOutput,
    ):
        raise TypeError(
            "edge_normalization must be a "
            "StructuralEdgeNormalizationOutput."
        )


def _require_optional_relation_gate(
    relation_gate: RelationGateOutput | None,
) -> None:
    if (
        relation_gate is not None
        and not isinstance(
            relation_gate,
            RelationGateOutput,
        )
    ):
        raise TypeError(
            "relation_gate must be a RelationGateOutput or None."
        )


def _require_optional_edge_attention(
    edge_attention: EdgeAttentionOutput | None,
) -> None:
    if (
        edge_attention is not None
        and not isinstance(
            edge_attention,
            EdgeAttentionOutput,
        )
    ):
        raise TypeError(
            "edge_attention must be an EdgeAttentionOutput or None."
        )


def _require_float_vector(
    name: str,
    value: torch.Tensor,
    *,
    num_edges: int,
) -> None:
    if not isinstance(
        value,
        torch.Tensor,
    ):
        raise TypeError(
            f"{name} must be a tensor."
        )

    if value.ndim != 1:
        raise ValueError(
            f"{name} must have rank 1 and shape [E]; "
            f"observed {tuple(value.shape)}."
        )

    if tuple(value.shape) != (
        num_edges,
    ):
        raise ValueError(
            f"{name} must have shape ({num_edges},); "
            f"observed {tuple(value.shape)}."
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


def _require_same_dtype_device(
    *,
    name: str,
    value: torch.Tensor,
    reference_name: str,
    reference: torch.Tensor,
) -> None:
    if value.dtype != reference.dtype:
        raise ValueError(
            f"{name} and {reference_name} must share one dtype. "
            f"Observed {value.dtype} and {reference.dtype}."
        )

    if value.device != reference.device:
        raise ValueError(
            f"{name} and {reference_name} must share one device. "
            f"Observed {value.device} and {reference.device}."
        )


def _require_nonnegative(
    name: str,
    value: torch.Tensor,
) -> None:
    if bool(
        (value < 0)
        .any()
        .item()
    ):
        raise ValueError(
            f"{name} must be nonnegative."
        )


def _normalize_semantic_edge_policy(
    semantic_edge_policy: str,
) -> str:
    if not isinstance(
        semantic_edge_policy,
        str,
    ):
        raise TypeError(
            "semantic_edge_policy must be a string."
        )

    normalized = semantic_edge_policy.strip()

    if not normalized:
        raise ValueError(
            "semantic_edge_policy must be a non-empty string."
        )

    if normalized not in (
        CANONICAL_MESSAGE_SEMANTIC_EDGE_POLICIES
    ):
        raise ValueError(
            "Unknown semantic-edge policy "
            f"{normalized!r}. Expected one of "
            f"{CANONICAL_MESSAGE_SEMANTIC_EDGE_POLICIES!r}."
        )

    if normalized not in (
        IMPLEMENTED_MESSAGE_SEMANTIC_EDGE_POLICIES
    ):
        raise NotImplementedError(
            "Semantic-edge policy "
            f"{normalized!r} is canonical but not implemented."
        )

    return normalized


def _identity_factor_like(
    reference: torch.Tensor,
) -> torch.Tensor:
    """
    Construct an exact all-one edge factor matching dtype and device.

    The identity tensor is intentionally independent of the reference values
    and introduces no trainable state.
    """

    return torch.ones_like(
        reference,
        memory_format=torch.preserve_format,
    )


def _require_exact_identity(
    name: str,
    value: torch.Tensor,
) -> None:
    if not torch.equal(
        value,
        torch.ones_like(value),
    ):
        raise RuntimeError(
            f"{name} failed to resolve to exact multiplicative identity "
            "one."
        )


def _validate_source_lineage(
    *,
    source_inputs: FunctionalMessagePassingInputs,
    edge_normalization: StructuralEdgeNormalizationOutput,
    relation_gate: RelationGateOutput | None,
    edge_attention: EdgeAttentionOutput | None,
) -> None:
    _require_source_inputs(
        source_inputs
    )
    _require_edge_normalization(
        edge_normalization
    )
    _require_optional_relation_gate(
        relation_gate
    )
    _require_optional_edge_attention(
        edge_attention
    )

    if edge_normalization.source_inputs is not (
        source_inputs
    ):
        raise ValueError(
            "edge_normalization must reference the exact supplied "
            "source_inputs object."
        )

    if (
        relation_gate is not None
        and relation_gate.source_inputs
        is not source_inputs
    ):
        raise ValueError(
            "relation_gate must reference the exact supplied "
            "source_inputs object."
        )

    if (
        edge_attention is not None
        and edge_attention.source_inputs
        is not source_inputs
    ):
        raise ValueError(
            "edge_attention must reference the exact supplied "
            "source_inputs object."
        )


# =============================================================================
# Individual factor resolution
# =============================================================================


def resolve_structural_normalization_factor(
    edge_normalization: StructuralEdgeNormalizationOutput,
    *,
    source_inputs: FunctionalMessagePassingInputs | None = None,
) -> torch.Tensor:
    """
    Preserve the exact upstream structural-normalization tensor.
    """

    _require_edge_normalization(
        edge_normalization
    )

    resolved_inputs = (
        edge_normalization.source_inputs
    )
    _require_source_inputs(
        resolved_inputs
    )

    if source_inputs is not None:
        _require_source_inputs(
            source_inputs
        )

        if resolved_inputs is not (
            source_inputs
        ):
            raise ValueError(
                "edge_normalization must reference the exact supplied "
                "source_inputs object."
            )

    factor = edge_normalization.coefficients

    _require_float_vector(
        "edge_normalization.coefficients",
        factor,
        num_edges=resolved_inputs.num_edges,
    )
    _require_same_dtype_device(
        name="edge_normalization.coefficients",
        value=factor,
        reference_name="source_inputs.node_state.fused_state",
        reference=(
            resolved_inputs
            .node_state
            .fused_state
        ),
    )
    _require_nonnegative(
        "edge_normalization.coefficients",
        factor,
    )

    return factor


def resolve_relation_gate_factor(
    *,
    source_inputs: FunctionalMessagePassingInputs,
    reference_factor: torch.Tensor,
    relation_gate: RelationGateOutput | None,
) -> OptionalFactorResolution:
    """
    Resolve an enabled exact relation gate or the disabled identity factor.
    """

    _require_source_inputs(
        source_inputs
    )
    _require_optional_relation_gate(
        relation_gate
    )
    _require_float_vector(
        "reference_factor",
        reference_factor,
        num_edges=source_inputs.num_edges,
    )

    if relation_gate is None:
        factor = _identity_factor_like(
            reference_factor
        )
        _require_exact_identity(
            "relation_gate_factor",
            factor,
        )

        return OptionalFactorResolution(
            factor=factor,
            source=None,
        )

    if relation_gate.source_inputs is not (
        source_inputs
    ):
        raise ValueError(
            "relation_gate must reference the exact supplied "
            "source_inputs object."
        )

    factor = relation_gate.edge_gate_values

    _require_float_vector(
        "relation_gate.edge_gate_values",
        factor,
        num_edges=source_inputs.num_edges,
    )
    _require_same_dtype_device(
        name="relation_gate.edge_gate_values",
        value=factor,
        reference_name="reference_factor",
        reference=reference_factor,
    )
    _require_nonnegative(
        "relation_gate.edge_gate_values",
        factor,
    )

    return OptionalFactorResolution(
        factor=factor,
        source=relation_gate,
    )


def resolve_edge_attention_factor(
    *,
    source_inputs: FunctionalMessagePassingInputs,
    reference_factor: torch.Tensor,
    edge_attention: EdgeAttentionOutput | None,
) -> OptionalFactorResolution:
    """
    Resolve enabled final reduced edge attention or the disabled identity.
    """

    _require_source_inputs(
        source_inputs
    )
    _require_optional_edge_attention(
        edge_attention
    )
    _require_float_vector(
        "reference_factor",
        reference_factor,
        num_edges=source_inputs.num_edges,
    )

    if edge_attention is None:
        factor = _identity_factor_like(
            reference_factor
        )
        _require_exact_identity(
            "edge_attention_factor",
            factor,
        )

        return OptionalFactorResolution(
            factor=factor,
            source=None,
        )

    if edge_attention.source_inputs is not (
        source_inputs
    ):
        raise ValueError(
            "edge_attention must reference the exact supplied "
            "source_inputs object."
        )

    factor = edge_attention.edge_weights

    _require_float_vector(
        "edge_attention.edge_weights",
        factor,
        num_edges=source_inputs.num_edges,
    )
    _require_same_dtype_device(
        name="edge_attention.edge_weights",
        value=factor,
        reference_name="reference_factor",
        reference=reference_factor,
    )
    _require_nonnegative(
        "edge_attention.edge_weights",
        factor,
    )

    return OptionalFactorResolution(
        factor=factor,
        source=edge_attention,
    )


def resolve_semantic_edge_factor(
    *,
    source_inputs: FunctionalMessagePassingInputs,
    reference_factor: torch.Tensor,
    semantic_edge_policy: str,
) -> SemanticFactorResolution:
    """
    Resolve the explicit semantic-edge policy.

    No caller-provided override tensor is accepted. Under
    ``use_source_graph``, the exact source-graph tensor is preserved.
    """

    _require_source_inputs(
        source_inputs
    )
    _require_float_vector(
        "reference_factor",
        reference_factor,
        num_edges=source_inputs.num_edges,
    )

    policy = _normalize_semantic_edge_policy(
        semantic_edge_policy
    )

    if policy == (
        MESSAGE_SEMANTIC_EDGE_POLICY_IGNORE
    ):
        factor = _identity_factor_like(
            reference_factor
        )
        _require_exact_identity(
            "semantic_edge_factor",
            factor,
        )

        return SemanticFactorResolution(
            factor=factor,
            source_weight=None,
            policy=policy,
        )

    if policy == (
        MESSAGE_SEMANTIC_EDGE_POLICY_USE_SOURCE_GRAPH
    ):
        source_weight = (
            source_inputs
            .source_graph
            .semantic_edge_weight
        )

        if source_weight is None:
            raise ValueError(
                "semantic_edge_policy='use_source_graph' requires "
                "source_graph.semantic_edge_weight."
            )

        _require_float_vector(
            "source_graph.semantic_edge_weight",
            source_weight,
            num_edges=source_inputs.num_edges,
        )
        _require_same_dtype_device(
            name="source_graph.semantic_edge_weight",
            value=source_weight,
            reference_name="reference_factor",
            reference=reference_factor,
        )

        return SemanticFactorResolution(
            factor=source_weight,
            source_weight=source_weight,
            policy=policy,
        )

    raise RuntimeError(
        "Unreachable semantic-edge policy branch."
    )


# =============================================================================
# Explicit coefficient composition
# =============================================================================


def combine_message_coefficients(
    *,
    structural_normalization_factor: torch.Tensor,
    relation_gate_factor: torch.Tensor,
    edge_attention_factor: torch.Tensor,
    semantic_edge_factor: torch.Tensor,
) -> torch.Tensor:
    """
    Multiply scalar factors in the frozen order.

    The operation is deliberately written in sequential statements so the
    implemented order matches ``MESSAGE_FACTOR_ORDER`` exactly.
    """

    if not isinstance(
        structural_normalization_factor,
        torch.Tensor,
    ):
        raise TypeError(
            "structural_normalization_factor must be a tensor."
        )

    if structural_normalization_factor.ndim != 1:
        raise ValueError(
            "structural_normalization_factor must have shape [E]."
        )

    num_edges = int(
        structural_normalization_factor.shape[0]
    )

    for name, factor in (
        (
            "structural_normalization_factor",
            structural_normalization_factor,
        ),
        (
            "relation_gate_factor",
            relation_gate_factor,
        ),
        (
            "edge_attention_factor",
            edge_attention_factor,
        ),
        (
            "semantic_edge_factor",
            semantic_edge_factor,
        ),
    ):
        _require_float_vector(
            name,
            factor,
            num_edges=num_edges,
        )
        _require_same_dtype_device(
            name=name,
            value=factor,
            reference_name=(
                "structural_normalization_factor"
            ),
            reference=(
                structural_normalization_factor
            ),
        )

    _require_nonnegative(
        "structural_normalization_factor",
        structural_normalization_factor,
    )
    _require_nonnegative(
        "relation_gate_factor",
        relation_gate_factor,
    )
    _require_nonnegative(
        "edge_attention_factor",
        edge_attention_factor,
    )

    combined = (
        structural_normalization_factor
        * relation_gate_factor
    )
    combined = (
        combined
        * edge_attention_factor
    )
    combined = (
        combined
        * semantic_edge_factor
    )

    if not bool(
        torch.isfinite(combined)
        .all()
        .item()
    ):
        raise FloatingPointError(
            "Combined message coefficient contains non-finite values."
        )

    return combined


# =============================================================================
# Complete functional resolver
# =============================================================================


def coefficient_resolution_architecture_dict(
    *,
    semantic_edge_policy: str,
) -> dict[str, Any]:
    """
    Return the parameter-free coefficient-resolver architecture.
    """

    policy = _normalize_semantic_edge_policy(
        semantic_edge_policy
    )

    return {
        "schema_version": (
            MESSAGE_COEFFICIENT_RESOLUTION_SCHEMA_VERSION
        ),
        "operation": (
            MESSAGE_COEFFICIENT_RESOLUTION_INTERPRETATION
        ),
        "operation_order": list(
            MESSAGE_COEFFICIENT_RESOLUTION_OPERATION_ORDER
        ),
        "factor_order": list(
            MESSAGE_FACTOR_ORDER
        ),
        "combined_coefficient_formula": (
            MESSAGE_COMBINED_COEFFICIENT_FORMULA
        ),
        "disabled_factor_policy": (
            MESSAGE_DISABLED_FACTOR_POLICY
        ),
        "semantic_edge_policy": policy,
        "structural_normalization_required": True,
        "relation_gate_optional": True,
        "edge_attention_optional": True,
        "semantic_edge_weight_estimation_owned_here": False,
        "preserves_enabled_source_tensor_identity": True,
        "parameter_free": (
            MESSAGE_COEFFICIENT_RESOLUTION_PARAMETER_FREE
        ),
        "buffer_free": (
            MESSAGE_COEFFICIENT_RESOLUTION_BUFFER_FREE
        ),
        "message_composition_owned_here": False,
        "aggregation_owned_here": False,
        "claims_causal_importance": False,
        "claims_explanation_faithfulness": False,
    }


def coefficient_resolution_architecture_fingerprint(
    *,
    semantic_edge_policy: str,
) -> str:
    """
    Fingerprint one semantic-edge-policy-specific resolver architecture.
    """

    return _fingerprint(
        coefficient_resolution_architecture_dict(
            semantic_edge_policy=(
                semantic_edge_policy
            )
        )
    )


def resolve_message_coefficients(
    *,
    edge_normalization: StructuralEdgeNormalizationOutput,
    relation_gate: RelationGateOutput | None = None,
    edge_attention: EdgeAttentionOutput | None = None,
    semantic_edge_policy: str = (
        MESSAGE_SEMANTIC_EDGE_POLICY_IGNORE
    ),
    source_inputs: FunctionalMessagePassingInputs | None = None,
    resolver_architecture_fingerprint: str | None = None,
) -> ResolvedMessageCoefficients:
    """
    Resolve every scalar edge-message coefficient.

    Parameters
    ----------
    edge_normalization:
        Required structural coefficient source and canonical source-input
        owner.
    relation_gate:
        Optional exact relation-gate output. ``None`` means disabled and
        resolves to one.
    edge_attention:
        Optional final edge-attention output. ``None`` means disabled and
        resolves to one.
    semantic_edge_policy:
        Explicit ``ignore`` or ``use_source_graph`` policy.
    source_inputs:
        Optional exact expected source-input object. When omitted, the
        edge-normalization source is authoritative.
    resolver_architecture_fingerprint:
        Optional caller-provided architecture fingerprint. When omitted, the
        deterministic functional fingerprint for the selected policy is used.
    """

    _require_edge_normalization(
        edge_normalization
    )

    resolved_inputs = (
        edge_normalization.source_inputs
    )

    if source_inputs is not None:
        _require_source_inputs(
            source_inputs
        )

        if resolved_inputs is not (
            source_inputs
        ):
            raise ValueError(
                "edge_normalization must reference the exact supplied "
                "source_inputs object."
            )
    else:
        source_inputs = resolved_inputs

    policy = _normalize_semantic_edge_policy(
        semantic_edge_policy
    )

    _validate_source_lineage(
        source_inputs=source_inputs,
        edge_normalization=edge_normalization,
        relation_gate=relation_gate,
        edge_attention=edge_attention,
    )

    structural_factor = (
        resolve_structural_normalization_factor(
            edge_normalization,
            source_inputs=source_inputs,
        )
    )
    relation_resolution = (
        resolve_relation_gate_factor(
            source_inputs=source_inputs,
            reference_factor=structural_factor,
            relation_gate=relation_gate,
        )
    )
    attention_resolution = (
        resolve_edge_attention_factor(
            source_inputs=source_inputs,
            reference_factor=structural_factor,
            edge_attention=edge_attention,
        )
    )
    semantic_resolution = (
        resolve_semantic_edge_factor(
            source_inputs=source_inputs,
            reference_factor=structural_factor,
            semantic_edge_policy=policy,
        )
    )

    combined_coefficient = (
        combine_message_coefficients(
            structural_normalization_factor=(
                structural_factor
            ),
            relation_gate_factor=(
                relation_resolution.factor
            ),
            edge_attention_factor=(
                attention_resolution.factor
            ),
            semantic_edge_factor=(
                semantic_resolution.factor
            ),
        )
    )

    architecture_fingerprint = (
        coefficient_resolution_architecture_fingerprint(
            semantic_edge_policy=policy
        )
        if resolver_architecture_fingerprint
        is None
        else resolver_architecture_fingerprint
    )
    _require_nonempty_string(
        "resolver_architecture_fingerprint",
        architecture_fingerprint,
    )

    output = ResolvedMessageCoefficients(
        structural_normalization_factor=(
            structural_factor
        ),
        relation_gate_factor=(
            relation_resolution.factor
        ),
        edge_attention_factor=(
            attention_resolution.factor
        ),
        semantic_edge_factor=(
            semantic_resolution.factor
        ),
        combined_coefficient=(
            combined_coefficient
        ),
        source_inputs=source_inputs,
        edge_normalization=edge_normalization,
        relation_gate=relation_resolution.source,
        edge_attention=attention_resolution.source,
        semantic_edge_weight=(
            semantic_resolution.source_weight
        ),
        semantic_edge_policy=policy,
        resolver_architecture_fingerprint=(
            architecture_fingerprint
        ),
    )

    validate_resolved_message_coefficients(
        output=output,
        source_inputs=source_inputs,
        edge_normalization=edge_normalization,
        relation_gate=relation_gate,
        edge_attention=edge_attention,
        semantic_edge_policy=policy,
    )

    return output


# =============================================================================
# Output validation
# =============================================================================


def validate_resolved_message_coefficients(
    *,
    output: ResolvedMessageCoefficients,
    source_inputs: FunctionalMessagePassingInputs | None = None,
    edge_normalization: StructuralEdgeNormalizationOutput | None = None,
    relation_gate: RelationGateOutput | None = None,
    edge_attention: EdgeAttentionOutput | None = None,
    semantic_edge_policy: str | None = None,
) -> None:
    """
    Validate one complete resolved-coefficient output and optional expectations.
    """

    if not isinstance(
        output,
        ResolvedMessageCoefficients,
    ):
        raise TypeError(
            "output must be a ResolvedMessageCoefficients."
        )

    if source_inputs is not None:
        _require_source_inputs(
            source_inputs
        )

        if output.source_inputs is not (
            source_inputs
        ):
            raise ValueError(
                "output must preserve the exact expected source_inputs "
                "object."
            )

    if edge_normalization is not None:
        _require_edge_normalization(
            edge_normalization
        )

        if output.edge_normalization is not (
            edge_normalization
        ):
            raise ValueError(
                "output must preserve the exact expected "
                "edge_normalization object."
            )

    if relation_gate is not None:
        _require_optional_relation_gate(
            relation_gate
        )

        if output.relation_gate is not (
            relation_gate
        ):
            raise ValueError(
                "output must preserve the exact expected relation_gate "
                "object."
            )
    elif (
        relation_gate is None
        and output.relation_gate is None
    ):
        _require_exact_identity(
            "output.relation_gate_factor",
            output.relation_gate_factor,
        )

    if edge_attention is not None:
        _require_optional_edge_attention(
            edge_attention
        )

        if output.edge_attention is not (
            edge_attention
        ):
            raise ValueError(
                "output must preserve the exact expected edge_attention "
                "object."
            )
    elif (
        edge_attention is None
        and output.edge_attention is None
    ):
        _require_exact_identity(
            "output.edge_attention_factor",
            output.edge_attention_factor,
        )

    if semantic_edge_policy is not None:
        expected_policy = (
            _normalize_semantic_edge_policy(
                semantic_edge_policy
            )
        )

        if output.semantic_edge_policy != (
            expected_policy
        ):
            raise ValueError(
                "output semantic-edge policy differs from the expected "
                "policy."
            )

    if output.structural_normalization_factor is not (
        output.edge_normalization.coefficients
    ):
        raise ValueError(
            "Resolved structural factor lost exact source tensor "
            "identity."
        )

    if output.relation_gate is not None:
        if output.relation_gate_factor is not (
            output.relation_gate.edge_gate_values
        ):
            raise ValueError(
                "Resolved relation-gate factor lost exact source tensor "
                "identity."
            )
    else:
        _require_exact_identity(
            "output.relation_gate_factor",
            output.relation_gate_factor,
        )

    if output.edge_attention is not None:
        if output.edge_attention_factor is not (
            output.edge_attention.edge_weights
        ):
            raise ValueError(
                "Resolved edge-attention factor lost exact source tensor "
                "identity."
            )
    else:
        _require_exact_identity(
            "output.edge_attention_factor",
            output.edge_attention_factor,
        )

    if output.semantic_edge_policy == (
        MESSAGE_SEMANTIC_EDGE_POLICY_IGNORE
    ):
        if output.semantic_edge_weight is not None:
            raise ValueError(
                "Ignored semantic edge weights must not be retained as "
                "a consumed source tensor."
            )

        _require_exact_identity(
            "output.semantic_edge_factor",
            output.semantic_edge_factor,
        )
    elif output.semantic_edge_policy == (
        MESSAGE_SEMANTIC_EDGE_POLICY_USE_SOURCE_GRAPH
    ):
        expected_weight = (
            output
            .source_inputs
            .source_graph
            .semantic_edge_weight
        )

        if expected_weight is None:
            raise ValueError(
                "Consumed semantic-edge policy has no source-graph "
                "semantic weight."
            )

        if output.semantic_edge_weight is not (
            expected_weight
        ):
            raise ValueError(
                "Resolved semantic-edge source lost exact source-graph "
                "tensor identity."
            )

        if output.semantic_edge_factor is not (
            expected_weight
        ):
            raise ValueError(
                "Resolved semantic-edge factor lost exact source-graph "
                "tensor identity."
            )

    expected_combined = (
        combine_message_coefficients(
            structural_normalization_factor=(
                output
                .structural_normalization_factor
            ),
            relation_gate_factor=(
                output.relation_gate_factor
            ),
            edge_attention_factor=(
                output.edge_attention_factor
            ),
            semantic_edge_factor=(
                output.semantic_edge_factor
            ),
        )
    )

    if not torch.equal(
        output.combined_coefficient,
        expected_combined,
    ):
        # The schema permits normal floating-point tolerance. This validator
        # first prefers exact equality and then mirrors that numerical
        # contract without demanding a particular accumulation bit pattern.
        if output.combined_coefficient.dtype in (
            torch.float16,
            torch.bfloat16,
        ):
            atol, rtol = 1e-3, 1e-3
        elif output.combined_coefficient.dtype == (
            torch.float64
        ):
            atol, rtol = 1e-10, 1e-9
        else:
            atol, rtol = 1e-6, 1e-5

        if not torch.allclose(
            output.combined_coefficient,
            expected_combined,
            atol=atol,
            rtol=rtol,
        ):
            raise ValueError(
                "output.combined_coefficient differs from the explicit "
                "factor product."
            )


# =============================================================================
# Descriptive diagnostics
# =============================================================================


def _safe_factor_statistics(
    factor: torch.Tensor,
) -> dict[str, Any]:
    if factor.numel() == 0:
        return {
            "count": 0,
            "minimum": None,
            "maximum": None,
            "mean": None,
            "zero_count": 0,
            "negative_count": 0,
            "finite": True,
        }

    detached = factor.detach()

    return {
        "count": int(detached.numel()),
        "minimum": float(
            detached.min().item()
        ),
        "maximum": float(
            detached.max().item()
        ),
        "mean": float(
            detached.mean().item()
        ),
        "zero_count": int(
            (detached == 0)
            .sum()
            .item()
        ),
        "negative_count": int(
            (detached < 0)
            .sum()
            .item()
        ),
        "finite": bool(
            torch.isfinite(detached)
            .all()
            .item()
        ),
    }


def message_coefficient_diagnostic_summary(
    output: ResolvedMessageCoefficients,
) -> dict[str, Any]:
    """
    Return compact descriptive factor statistics.

    These diagnostics describe numerical scaling. They do not decompose
    causal importance or explanation faithfulness.
    """

    validate_resolved_message_coefficients(
        output=output
    )

    return {
        "schema_version": (
            MESSAGE_COEFFICIENT_RESOLUTION_SCHEMA_VERSION
        ),
        "operation": (
            MESSAGE_COEFFICIENT_RESOLUTION_INTERPRETATION
        ),
        "num_edges": output.num_edges,
        "dtype": str(output.dtype),
        "device": str(output.device),
        "semantic_edge_policy": (
            output.semantic_edge_policy
        ),
        "relation_gate_enabled": (
            output.relation_gate_enabled
        ),
        "edge_attention_enabled": (
            output.edge_attention_enabled
        ),
        "semantic_edge_weight_enabled": (
            output.semantic_edge_weight_enabled
        ),
        "active_factor_names": list(
            output.active_factor_names
        ),
        "disabled_factor_names": list(
            output.disabled_factor_names
        ),
        "factors": {
            MESSAGE_FACTOR_STRUCTURAL_NORMALIZATION: (
                _safe_factor_statistics(
                    output
                    .structural_normalization_factor
                )
            ),
            MESSAGE_FACTOR_RELATION_GATE: (
                _safe_factor_statistics(
                    output.relation_gate_factor
                )
            ),
            MESSAGE_FACTOR_EDGE_ATTENTION: (
                _safe_factor_statistics(
                    output.edge_attention_factor
                )
            ),
            MESSAGE_FACTOR_SEMANTIC_EDGE_WEIGHT: (
                _safe_factor_statistics(
                    output.semantic_edge_factor
                )
            ),
        },
        "combined_coefficient": (
            _safe_factor_statistics(
                output.combined_coefficient
            )
        ),
        "parameter_free": True,
        "aggregation_performed_here": False,
        "causal_importance_claim": False,
        "explanation_faithfulness_claim": False,
    }


# =============================================================================
# Parameter-free module wrapper
# =============================================================================


class MessageCoefficientResolver(nn.Module):
    """
    Parameter-free resolver with a frozen semantic-edge policy.
    """

    semantic_edge_policy: str

    def __init__(
        self,
        *,
        semantic_edge_policy: str = (
            MESSAGE_SEMANTIC_EDGE_POLICY_IGNORE
        ),
    ) -> None:
        super().__init__()

        self.semantic_edge_policy = (
            _normalize_semantic_edge_policy(
                semantic_edge_policy
            )
        )

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
        return coefficient_resolution_architecture_dict(
            semantic_edge_policy=(
                self.semantic_edge_policy
            )
        )

    def architecture_fingerprint(
        self,
    ) -> str:
        return coefficient_resolution_architecture_fingerprint(
            semantic_edge_policy=(
                self.semantic_edge_policy
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
        state = self.state_dict()

        if parameters:
            raise RuntimeError(
                "MessageCoefficientResolver must remain parameter-free. "
                f"Observed parameters: "
                f"{tuple(name for name, _ in parameters)}."
            )

        if buffers:
            raise RuntimeError(
                "MessageCoefficientResolver must remain buffer-free. "
                f"Observed buffers: "
                f"{tuple(name for name, _ in buffers)}."
            )

        if state:
            raise RuntimeError(
                "MessageCoefficientResolver must have an empty "
                "state_dict."
            )

        if self.parameter_count != 0:
            raise RuntimeError(
                "MessageCoefficientResolver parameter_count must be "
                "zero."
            )

        if self.trainable_parameter_count != 0:
            raise RuntimeError(
                "MessageCoefficientResolver trainable_parameter_count "
                "must be zero."
            )

        if self.buffer_count != 0:
            raise RuntimeError(
                "MessageCoefficientResolver buffer_count must be zero."
            )

    def resolve(
        self,
        *,
        edge_normalization: StructuralEdgeNormalizationOutput,
        relation_gate: RelationGateOutput | None = None,
        edge_attention: EdgeAttentionOutput | None = None,
        source_inputs: FunctionalMessagePassingInputs | None = None,
    ) -> ResolvedMessageCoefficients:
        """
        Resolve coefficients under the module's frozen semantic-edge policy.
        """

        self.assert_parameter_free()

        return resolve_message_coefficients(
            edge_normalization=edge_normalization,
            relation_gate=relation_gate,
            edge_attention=edge_attention,
            semantic_edge_policy=(
                self.semantic_edge_policy
            ),
            source_inputs=source_inputs,
            resolver_architecture_fingerprint=(
                self.architecture_fingerprint()
            ),
        )

    def diagnostic_summary(
        self,
        output: ResolvedMessageCoefficients,
    ) -> dict[str, Any]:
        """
        Return diagnostics after validating policy and architecture identity.
        """

        self.assert_parameter_free()

        validate_resolved_message_coefficients(
            output=output,
            semantic_edge_policy=(
                self.semantic_edge_policy
            ),
        )

        if (
            output.resolver_architecture_fingerprint
            != self.architecture_fingerprint()
        ):
            raise ValueError(
                "Resolved coefficients were not produced under this "
                "resolver architecture."
            )

        return message_coefficient_diagnostic_summary(
            output
        )

    def forward(
        self,
        *,
        edge_normalization: StructuralEdgeNormalizationOutput,
        relation_gate: RelationGateOutput | None = None,
        edge_attention: EdgeAttentionOutput | None = None,
        source_inputs: FunctionalMessagePassingInputs | None = None,
    ) -> ResolvedMessageCoefficients:
        return self.resolve(
            edge_normalization=edge_normalization,
            relation_gate=relation_gate,
            edge_attention=edge_attention,
            source_inputs=source_inputs,
        )

    def extra_repr(self) -> str:
        return (
            f"semantic_edge_policy={self.semantic_edge_policy!r}, "
            f"factor_order={MESSAGE_FACTOR_ORDER!r}, "
            "parameter_free=True"
        )


# =============================================================================
# Builders and aliases
# =============================================================================


def build_message_coefficient_resolver(
    *,
    semantic_edge_policy: str = (
        MESSAGE_SEMANTIC_EDGE_POLICY_IGNORE
    ),
) -> MessageCoefficientResolver:
    """
    Construct a parameter-free resolver with one frozen semantic policy.
    """

    return MessageCoefficientResolver(
        semantic_edge_policy=(
            semantic_edge_policy
        )
    )


CoefficientResolver = MessageCoefficientResolver
EdgeMessageCoefficientResolver = MessageCoefficientResolver
resolve_edge_message_coefficients = resolve_message_coefficients
build_coefficient_resolver = build_message_coefficient_resolver


__all__ = (
    # Public identity.
    "MESSAGE_COEFFICIENT_RESOLUTION_SCHEMA_VERSION",
    "MESSAGE_COEFFICIENT_RESOLUTION_OPERATION_ORDER",
    "MESSAGE_COEFFICIENT_RESOLUTION_INTERPRETATION",
    "MESSAGE_COEFFICIENT_RESOLUTION_PARAMETER_FREE",
    "MESSAGE_COEFFICIENT_RESOLUTION_BUFFER_FREE",
    # Small resolution contracts.
    "OptionalFactorResolution",
    "SemanticFactorResolution",
    # Individual factors.
    "resolve_structural_normalization_factor",
    "resolve_relation_gate_factor",
    "resolve_edge_attention_factor",
    "resolve_semantic_edge_factor",
    "combine_message_coefficients",
    # Complete functional resolution.
    "coefficient_resolution_architecture_dict",
    "coefficient_resolution_architecture_fingerprint",
    "resolve_message_coefficients",
    "resolve_edge_message_coefficients",
    "validate_resolved_message_coefficients",
    "message_coefficient_diagnostic_summary",
    # Module API.
    "MessageCoefficientResolver",
    "CoefficientResolver",
    "EdgeMessageCoefficientResolver",
    "build_message_coefficient_resolver",
    "build_coefficient_resolver",
)
