"""
Parameter-free composition of final edge-message vectors.

Target path:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            functional_message_passing/
                message_builders/
                    message_composition.py

This module owns one numerical operation:

    edge_messages
        = transformed_source_state
        * combined_coefficient.unsqueeze(-1)

where:

    transformed_source_state  has shape [E, H]
    combined_coefficient      has shape [E]
    edge_messages             has shape [E, H]

The relation-transform subsystem already owns exact relation dispatch and
publishes an edge-aligned transformed source-state tensor. The coefficient
resolver already owns structural normalization, relation gates, edge
attention, semantic-edge policy, disabled-factor identities, and coefficient
multiplication. This module combines those two completed stages without
reinterpreting either one.

Equation
-------------------
For every stored directed edge ``e``:

    u_e
        = relation_transform.transformed_source_state[e]

    c_e
        = resolved_coefficients.combined_coefficient[e]

    m_e
        = u_e * c_e

The scalar coefficient is broadcast only across the final hidden-feature
axis. No target-node aggregation occurs here.

Separation
---------------------
The module does not:

- gather source nodes;
- dispatch exact relation transforms;
- recompute structural normalization;
- predict or activate relation gates;
- score or normalize edge attention;
- estimate semantic edge weights;
- renormalize combined coefficients;
- aggregate messages into target nodes;
- add residual connections;
- apply layer normalization or dropout;
- assign causal meaning to factors or message dimensions.

The output remains edge aligned and preserves exact source-object lineage.

Numerical behavior
------------------
The implementation:

- validates exact ``FunctionalMessagePassingInputs`` lineage;
- preserves the exact upstream transformed-state tensor;
- preserves the exact resolved-coefficient object;
- supports empty edge sets;
- preserves dtype, device, and autograd connectivity;
- detects non-finite input values;
- detects overflow or other non-finite output values;
- performs no cloning, detaching, casting, or device movement;
- introduces no parameters or buffers.

The final immutable result is ``MessageCompositionOutput`` from
``message_builders.schemas``.
"""

from __future__ import annotations

from hashlib import sha256
import json
from typing import Any, Final, Mapping

import torch
from torch import nn

from ..schemas import (
    FunctionalMessagePassingInputs,
    RelationTransformOutput,
)
from .coefficient_resolution import (
    validate_resolved_message_coefficients,
)
from .relation_state_gather import (
    RelationStateGather,
    assert_zero_copy_relation_state,
    build_relation_state_gather,
    resolve_edge_aligned_relation_state,
    validate_edge_aligned_relation_state,
)
from .schemas import (
    MESSAGE_COMPOSITION_FORMULA,
    MESSAGE_FACTOR_ORDER,
    MESSAGE_TRANSFORM_INPUT_LAYOUT,
    MessageCompositionOutput,
    ResolvedMessageCoefficients,
)


# =============================================================================
# Public identity
# =============================================================================


MESSAGE_COMPOSER_SCHEMA_VERSION: Final[str] = "0.1"

MESSAGE_COMPOSER_OPERATION: Final[str] = (
    "broadcast_scalar_edge_coefficient_over_hidden_features"
)

MESSAGE_COMPOSER_OPERATION_ORDER: Final[
    tuple[str, ...]
] = (
    "validate_relation_transform_output",
    "resolve_exact_edge_aligned_transformed_source_state",
    "validate_resolved_message_coefficients",
    "validate_exact_source_input_lineage",
    "broadcast_combined_coefficient_over_hidden_axis",
    "multiply_transformed_state_by_combined_coefficient",
    "validate_finite_edge_messages",
    "construct_message_composition_output",
)

MESSAGE_COMPOSER_INPUT_STATE_LAYOUT: Final[str] = (
    MESSAGE_TRANSFORM_INPUT_LAYOUT
)
MESSAGE_COMPOSER_INPUT_COEFFICIENT_LAYOUT: Final[str] = (
    "edge_aligned_combined_coefficient_[E]"
)
MESSAGE_COMPOSER_OUTPUT_LAYOUT: Final[str] = (
    "edge_aligned_messages_[E,H]"
)

MESSAGE_COMPOSER_BROADCAST_AXIS: Final[int] = -1
MESSAGE_COMPOSER_PARAMETER_FREE: Final[bool] = True
MESSAGE_COMPOSER_BUFFER_FREE: Final[bool] = True
MESSAGE_COMPOSER_AGGREGATION_OWNED_HERE: Final[bool] = False


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


def _require_float_matrix(
    name: str,
    value: torch.Tensor,
    *,
    num_edges: int,
    hidden_dim: int,
) -> None:
    if not isinstance(
        value,
        torch.Tensor,
    ):
        raise TypeError(
            f"{name} must be a tensor."
        )

    if value.ndim != 2:
        raise ValueError(
            f"{name} must have rank 2 and shape [E, H]; "
            f"observed {tuple(value.shape)}."
        )

    expected = (
        num_edges,
        hidden_dim,
    )
    observed = tuple(
        int(size)
        for size in value.shape
    )

    if observed != expected:
        raise ValueError(
            f"{name} must have shape {expected}; "
            f"observed {observed}."
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

    expected = (
        num_edges,
    )
    observed = tuple(
        int(size)
        for size in value.shape
    )

    if observed != expected:
        raise ValueError(
            f"{name} must have shape {expected}; "
            f"observed {observed}."
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
    transformed_source_state: torch.Tensor,
    combined_coefficient: torch.Tensor,
) -> None:
    if (
        transformed_source_state.dtype
        != combined_coefficient.dtype
    ):
        raise ValueError(
            "transformed_source_state and combined_coefficient must "
            "share one dtype. Observed "
            f"{transformed_source_state.dtype} and "
            f"{combined_coefficient.dtype}."
        )

    if (
        transformed_source_state.device
        != combined_coefficient.device
    ):
        raise ValueError(
            "transformed_source_state and combined_coefficient must "
            "share one device. Observed "
            f"{transformed_source_state.device} and "
            f"{combined_coefficient.device}."
        )


def _require_exact_source_lineage(
    *,
    relation_transform: RelationTransformOutput,
    resolved_coefficients: ResolvedMessageCoefficients,
    source_inputs: FunctionalMessagePassingInputs | None,
) -> FunctionalMessagePassingInputs:
    transform_inputs = (
        relation_transform.source_inputs
    )
    coefficient_inputs = (
        resolved_coefficients.source_inputs
    )

    _require_source_inputs(
        transform_inputs
    )
    _require_source_inputs(
        coefficient_inputs
    )

    if transform_inputs is not (
        coefficient_inputs
    ):
        raise ValueError(
            "relation_transform and resolved_coefficients must share "
            "the exact same FunctionalMessagePassingInputs object."
        )

    if source_inputs is not None:
        _require_source_inputs(
            source_inputs
        )

        if transform_inputs is not (
            source_inputs
        ):
            raise ValueError(
                "relation_transform and resolved_coefficients must "
                "reference the exact supplied source_inputs object."
            )

    return transform_inputs


# =============================================================================
# Static architecture identity
# =============================================================================


def message_composer_architecture_dict() -> dict[str, Any]:
    """
    Return the frozen architecture of the parameter-free composition stage.
    """

    return {
        "schema_version": (
            MESSAGE_COMPOSER_SCHEMA_VERSION
        ),
        "operation": (
            MESSAGE_COMPOSER_OPERATION
        ),
        "operation_order": list(
            MESSAGE_COMPOSER_OPERATION_ORDER
        ),
        "input_state_layout": (
            MESSAGE_COMPOSER_INPUT_STATE_LAYOUT
        ),
        "input_coefficient_layout": (
            MESSAGE_COMPOSER_INPUT_COEFFICIENT_LAYOUT
        ),
        "output_layout": (
            MESSAGE_COMPOSER_OUTPUT_LAYOUT
        ),
        "composition_formula": (
            MESSAGE_COMPOSITION_FORMULA
        ),
        "factor_order": list(
            MESSAGE_FACTOR_ORDER
        ),
        "broadcast_axis": (
            MESSAGE_COMPOSER_BROADCAST_AXIS
        ),
        "parameter_free": (
            MESSAGE_COMPOSER_PARAMETER_FREE
        ),
        "buffer_free": (
            MESSAGE_COMPOSER_BUFFER_FREE
        ),
        "relation_state_recomputed_here": False,
        "coefficient_recomputed_here": False,
        "aggregation_owned_here": (
            MESSAGE_COMPOSER_AGGREGATION_OWNED_HERE
        ),
        "residual_update_owned_here": False,
        "layer_normalization_owned_here": False,
        "dropout_owned_here": False,
        "claims_causal_importance": False,
        "claims_explanation_faithfulness": False,
    }


def message_composer_architecture_fingerprint() -> str:
    """
    Fingerprint the static composition architecture.
    """

    return _fingerprint(
        message_composer_architecture_dict()
    )


# =============================================================================
# Low-level numerical kernel
# =============================================================================


def compose_edge_message_tensor(
    *,
    transformed_source_state: torch.Tensor,
    combined_coefficient: torch.Tensor,
) -> torch.Tensor:
    """
    Compose edge-message vectors from edge-aligned states and coefficients.

    This low-level kernel validates only numerical tensor contracts. Exact
    source-object lineage is validated by the higher-level composition helper.
    """

    if not isinstance(
        transformed_source_state,
        torch.Tensor,
    ):
        raise TypeError(
            "transformed_source_state must be a tensor."
        )

    if transformed_source_state.ndim != 2:
        raise ValueError(
            "transformed_source_state must have rank 2 and shape [E, H]."
        )

    num_edges = int(
        transformed_source_state.shape[0]
    )
    hidden_dim = int(
        transformed_source_state.shape[1]
    )

    _require_float_matrix(
        "transformed_source_state",
        transformed_source_state,
        num_edges=num_edges,
        hidden_dim=hidden_dim,
    )
    _require_float_vector(
        "combined_coefficient",
        combined_coefficient,
        num_edges=num_edges,
    )
    _require_same_dtype_device(
        transformed_source_state=(
            transformed_source_state
        ),
        combined_coefficient=(
            combined_coefficient
        ),
    )

    edge_messages = (
        transformed_source_state
        * combined_coefficient.unsqueeze(
            MESSAGE_COMPOSER_BROADCAST_AXIS
        )
    )

    if tuple(edge_messages.shape) != (
        num_edges,
        hidden_dim,
    ):
        raise RuntimeError(
            "Edge-message composition produced an unexpected shape. "
            f"Expected {(num_edges, hidden_dim)}, observed "
            f"{tuple(edge_messages.shape)}."
        )

    if edge_messages.dtype != (
        transformed_source_state.dtype
    ):
        raise RuntimeError(
            "Edge-message composition changed dtype unexpectedly."
        )

    if edge_messages.device != (
        transformed_source_state.device
    ):
        raise RuntimeError(
            "Edge-message composition changed device unexpectedly."
        )

    if not bool(
        torch.isfinite(edge_messages)
        .all()
        .item()
    ):
        raise FloatingPointError(
            "Edge-message composition produced non-finite values. "
            "This may indicate overflow in transformed states, resolved "
            "coefficients, or their product."
        )

    return edge_messages


# =============================================================================
# High-level composition
# =============================================================================


def compose_message_output(
    *,
    relation_transform: RelationTransformOutput,
    resolved_coefficients: ResolvedMessageCoefficients,
    source_inputs: FunctionalMessagePassingInputs | None = None,
    composer_architecture_fingerprint: str | None = None,
) -> MessageCompositionOutput:
    """
    Compose one immutable edge-message output from exact upstream stages.
    """

    _require_relation_transform(
        relation_transform
    )
    _require_resolved_coefficients(
        resolved_coefficients
    )

    resolved_inputs = (
        _require_exact_source_lineage(
            relation_transform=relation_transform,
            resolved_coefficients=resolved_coefficients,
            source_inputs=source_inputs,
        )
    )

    validate_edge_aligned_relation_state(
        relation_transform=relation_transform,
        source_inputs=resolved_inputs,
    )
    validate_resolved_message_coefficients(
        output=resolved_coefficients,
        source_inputs=resolved_inputs,
    )

    transformed_source_state = (
        resolve_edge_aligned_relation_state(
            relation_transform,
            source_inputs=resolved_inputs,
        )
    )

    assert_zero_copy_relation_state(
        relation_transform=relation_transform,
        resolved_state=transformed_source_state,
    )

    combined_coefficient = (
        resolved_coefficients
        .combined_coefficient
    )

    _require_float_matrix(
        "transformed_source_state",
        transformed_source_state,
        num_edges=resolved_inputs.num_edges,
        hidden_dim=resolved_inputs.hidden_dim,
    )
    _require_float_vector(
        "resolved_coefficients.combined_coefficient",
        combined_coefficient,
        num_edges=resolved_inputs.num_edges,
    )
    _require_same_dtype_device(
        transformed_source_state=(
            transformed_source_state
        ),
        combined_coefficient=(
            combined_coefficient
        ),
    )

    edge_messages = compose_edge_message_tensor(
        transformed_source_state=(
            transformed_source_state
        ),
        combined_coefficient=(
            combined_coefficient
        ),
    )

    architecture_fingerprint = (
        message_composer_architecture_fingerprint()
        if composer_architecture_fingerprint
        is None
        else composer_architecture_fingerprint
    )
    _require_nonempty_string(
        "composer_architecture_fingerprint",
        architecture_fingerprint,
    )

    output = MessageCompositionOutput(
        edge_messages=edge_messages,
        relation_transform=relation_transform,
        resolved_coefficients=(
            resolved_coefficients
        ),
        composer_architecture_fingerprint=(
            architecture_fingerprint
        ),
    )

    validate_message_composition_output(
        output=output,
        relation_transform=relation_transform,
        resolved_coefficients=(
            resolved_coefficients
        ),
        source_inputs=resolved_inputs,
        composer_architecture_fingerprint=(
            architecture_fingerprint
        ),
    )

    return output


# =============================================================================
# Output validation
# =============================================================================


def validate_message_composition_output(
    *,
    output: MessageCompositionOutput,
    relation_transform: RelationTransformOutput | None = None,
    resolved_coefficients: ResolvedMessageCoefficients | None = None,
    source_inputs: FunctionalMessagePassingInputs | None = None,
    composer_architecture_fingerprint: str | None = None,
) -> None:
    """
    Validate one complete message-composition result and optional expectations.
    """

    if not isinstance(
        output,
        MessageCompositionOutput,
    ):
        raise TypeError(
            "output must be a MessageCompositionOutput."
        )

    _require_relation_transform(
        output.relation_transform
    )
    _require_resolved_coefficients(
        output.resolved_coefficients
    )

    resolved_inputs = _require_exact_source_lineage(
        relation_transform=(
            output.relation_transform
        ),
        resolved_coefficients=(
            output.resolved_coefficients
        ),
        source_inputs=source_inputs,
    )

    if relation_transform is not None:
        _require_relation_transform(
            relation_transform
        )

        if output.relation_transform is not (
            relation_transform
        ):
            raise ValueError(
                "output must preserve the exact expected "
                "relation_transform object."
            )

    if resolved_coefficients is not None:
        _require_resolved_coefficients(
            resolved_coefficients
        )

        if output.resolved_coefficients is not (
            resolved_coefficients
        ):
            raise ValueError(
                "output must preserve the exact expected "
                "resolved_coefficients object."
            )

    validate_edge_aligned_relation_state(
        relation_transform=(
            output.relation_transform
        ),
        source_inputs=resolved_inputs,
    )
    validate_resolved_message_coefficients(
        output=output.resolved_coefficients,
        source_inputs=resolved_inputs,
    )

    transformed_source_state = (
        output
        .relation_transform
        .transformed_source_state
    )
    combined_coefficient = (
        output
        .resolved_coefficients
        .combined_coefficient
    )

    _require_float_matrix(
        "output.edge_messages",
        output.edge_messages,
        num_edges=resolved_inputs.num_edges,
        hidden_dim=resolved_inputs.hidden_dim,
    )
    _require_same_dtype_device(
        transformed_source_state=(
            output.edge_messages
        ),
        combined_coefficient=(
            combined_coefficient
        ),
    )

    expected = compose_edge_message_tensor(
        transformed_source_state=(
            transformed_source_state
        ),
        combined_coefficient=(
            combined_coefficient
        ),
    )

    if output.edge_messages.dtype in (
        torch.float16,
        torch.bfloat16,
    ):
        atol, rtol = 1e-3, 1e-3
    elif output.edge_messages.dtype == (
        torch.float64
    ):
        atol, rtol = 1e-10, 1e-9
    else:
        atol, rtol = 1e-6, 1e-5

    if not torch.equal(
        output.edge_messages,
        expected,
    ) and not torch.allclose(
        output.edge_messages,
        expected,
        atol=atol,
        rtol=rtol,
    ):
        raise ValueError(
            "output.edge_messages differs from transformed_source_state "
            "multiplied by combined_coefficient."
        )

    _require_nonempty_string(
        "output.composer_architecture_fingerprint",
        output.composer_architecture_fingerprint,
    )

    if (
        composer_architecture_fingerprint
        is not None
    ):
        _require_nonempty_string(
            "composer_architecture_fingerprint",
            composer_architecture_fingerprint,
        )

        if (
            output
            .composer_architecture_fingerprint
            != composer_architecture_fingerprint
        ):
            raise ValueError(
                "output composer architecture fingerprint differs from "
                "the expected fingerprint."
            )


# =============================================================================
# Descriptive diagnostics
# =============================================================================


def _safe_message_statistics(
    edge_messages: torch.Tensor,
) -> dict[str, Any]:
    if edge_messages.numel() == 0:
        return {
            "element_count": 0,
            "edge_count": int(
                edge_messages.shape[0]
            ),
            "hidden_dim": int(
                edge_messages.shape[1]
            ),
            "minimum": None,
            "maximum": None,
            "mean": None,
            "mean_absolute_value": None,
            "l2_norm": 0.0,
            "zero_count": 0,
            "finite": True,
        }

    detached = edge_messages.detach()

    return {
        "element_count": int(
            detached.numel()
        ),
        "edge_count": int(
            detached.shape[0]
        ),
        "hidden_dim": int(
            detached.shape[1]
        ),
        "minimum": float(
            detached.min().item()
        ),
        "maximum": float(
            detached.max().item()
        ),
        "mean": float(
            detached.mean().item()
        ),
        "mean_absolute_value": float(
            detached.abs().mean().item()
        ),
        "l2_norm": float(
            torch.linalg.vector_norm(
                detached
            ).item()
        ),
        "zero_count": int(
            (detached == 0)
            .sum()
            .item()
        ),
        "finite": bool(
            torch.isfinite(detached)
            .all()
            .item()
        ),
    }


def message_composition_diagnostic_summary(
    output: MessageCompositionOutput,
) -> dict[str, Any]:
    """
    Return compact descriptive diagnostics for one composition output.

    Values describe numerical scaling only. They do not establish causal
    importance, explanation faithfulness, or relation-mechanism attribution.
    """

    validate_message_composition_output(
        output=output
    )

    transformed = (
        output
        .relation_transform
        .transformed_source_state
    )
    coefficient = (
        output
        .resolved_coefficients
        .combined_coefficient
    )

    if coefficient.numel() == 0:
        coefficient_summary = {
            "count": 0,
            "minimum": None,
            "maximum": None,
            "mean": None,
            "zero_count": 0,
            "finite": True,
        }
    else:
        detached_coefficient = (
            coefficient.detach()
        )
        coefficient_summary = {
            "count": int(
                detached_coefficient.numel()
            ),
            "minimum": float(
                detached_coefficient.min().item()
            ),
            "maximum": float(
                detached_coefficient.max().item()
            ),
            "mean": float(
                detached_coefficient.mean().item()
            ),
            "zero_count": int(
                (detached_coefficient == 0)
                .sum()
                .item()
            ),
            "finite": bool(
                torch.isfinite(
                    detached_coefficient
                )
                .all()
                .item()
            ),
        }

    return {
        "schema_version": (
            MESSAGE_COMPOSER_SCHEMA_VERSION
        ),
        "operation": (
            MESSAGE_COMPOSER_OPERATION
        ),
        "composition_formula": (
            MESSAGE_COMPOSITION_FORMULA
        ),
        "num_edges": output.num_edges,
        "hidden_dim": output.hidden_dim,
        "dtype": str(output.dtype),
        "device": str(output.device),
        "requires_grad": (
            output.edge_messages.requires_grad
        ),
        "transformed_state_zero_copy_identity_preserved": (
            transformed
            is output
            .relation_transform
            .transformed_source_state
        ),
        "resolved_coefficients_object_identity_preserved": (
            output.resolved_coefficients
            is output.resolved_coefficients
        ),
        "combined_coefficient": (
            coefficient_summary
        ),
        "edge_messages": (
            _safe_message_statistics(
                output.edge_messages
            )
        ),
        "parameter_free": True,
        "buffer_free": True,
        "aggregation_performed_here": False,
        "residual_update_performed_here": False,
        "causal_importance_claim": False,
        "explanation_faithfulness_claim": False,
    }


# =============================================================================
# Parameter-free module wrapper
# =============================================================================


class EdgeMessageComposer(nn.Module):
    """
    Parameter-free orchestrator for edge-message vector composition.

    Parameters
    ----------
    relation_state_gather:
        Optional zero-copy relation-state boundary. A default
        ``RelationStateGather`` is constructed when omitted.
    """

    relation_state_gather: RelationStateGather

    def __init__(
        self,
        *,
        relation_state_gather: RelationStateGather | None = None,
    ) -> None:
        super().__init__()

        if relation_state_gather is None:
            relation_state_gather = (
                build_relation_state_gather()
            )

        if not isinstance(
            relation_state_gather,
            RelationStateGather,
        ):
            raise TypeError(
                "relation_state_gather must be a "
                "RelationStateGather."
            )

        relation_state_gather.assert_parameter_free()
        self.relation_state_gather = (
            relation_state_gather
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
        architecture = (
            message_composer_architecture_dict()
        )
        architecture[
            "relation_state_gather"
        ] = (
            self
            .relation_state_gather
            .architecture_dict()
        )
        return architecture

    def architecture_fingerprint(
        self,
    ) -> str:
        return _fingerprint(
            self.architecture_dict()
        )

    def assert_parameter_free(
        self,
    ) -> None:
        self.relation_state_gather.assert_parameter_free()

        parameters = tuple(
            self.named_parameters()
        )
        buffers = tuple(
            self.named_buffers()
        )
        state = self.state_dict()

        if parameters:
            raise RuntimeError(
                "EdgeMessageComposer must remain parameter-free. "
                f"Observed parameters: "
                f"{tuple(name for name, _ in parameters)}."
            )

        if buffers:
            raise RuntimeError(
                "EdgeMessageComposer must remain buffer-free. "
                f"Observed buffers: "
                f"{tuple(name for name, _ in buffers)}."
            )

        if state:
            raise RuntimeError(
                "EdgeMessageComposer must have an empty state_dict."
            )

        if self.parameter_count != 0:
            raise RuntimeError(
                "EdgeMessageComposer parameter_count must be zero."
            )

        if (
            self.trainable_parameter_count
            != 0
        ):
            raise RuntimeError(
                "EdgeMessageComposer trainable_parameter_count must be "
                "zero."
            )

        if self.buffer_count != 0:
            raise RuntimeError(
                "EdgeMessageComposer buffer_count must be zero."
            )

    def compose(
        self,
        *,
        relation_transform: RelationTransformOutput,
        resolved_coefficients: ResolvedMessageCoefficients,
        source_inputs: FunctionalMessagePassingInputs | None = None,
    ) -> MessageCompositionOutput:
        """
        Compose one immutable edge-message output.
        """

        self.assert_parameter_free()

        _require_relation_transform(
            relation_transform
        )
        _require_resolved_coefficients(
            resolved_coefficients
        )

        resolved_inputs = (
            _require_exact_source_lineage(
                relation_transform=(
                    relation_transform
                ),
                resolved_coefficients=(
                    resolved_coefficients
                ),
                source_inputs=source_inputs,
            )
        )

        resolved_state = (
            self.relation_state_gather(
                relation_transform,
                source_inputs=resolved_inputs,
            )
        )
        assert_zero_copy_relation_state(
            relation_transform=(
                relation_transform
            ),
            resolved_state=resolved_state,
        )

        edge_messages = (
            compose_edge_message_tensor(
                transformed_source_state=(
                    resolved_state
                ),
                combined_coefficient=(
                    resolved_coefficients
                    .combined_coefficient
                ),
            )
        )

        output = MessageCompositionOutput(
            edge_messages=edge_messages,
            relation_transform=(
                relation_transform
            ),
            resolved_coefficients=(
                resolved_coefficients
            ),
            composer_architecture_fingerprint=(
                self.architecture_fingerprint()
            ),
        )

        validate_message_composition_output(
            output=output,
            relation_transform=(
                relation_transform
            ),
            resolved_coefficients=(
                resolved_coefficients
            ),
            source_inputs=resolved_inputs,
            composer_architecture_fingerprint=(
                self.architecture_fingerprint()
            ),
        )

        return output

    def diagnostic_summary(
        self,
        output: MessageCompositionOutput,
    ) -> dict[str, Any]:
        """
        Validate architecture identity and return compact diagnostics.
        """

        self.assert_parameter_free()

        validate_message_composition_output(
            output=output,
            composer_architecture_fingerprint=(
                self.architecture_fingerprint()
            ),
        )

        return message_composition_diagnostic_summary(
            output
        )

    def forward(
        self,
        *,
        relation_transform: RelationTransformOutput,
        resolved_coefficients: ResolvedMessageCoefficients,
        source_inputs: FunctionalMessagePassingInputs | None = None,
    ) -> MessageCompositionOutput:
        return self.compose(
            relation_transform=(
                relation_transform
            ),
            resolved_coefficients=(
                resolved_coefficients
            ),
            source_inputs=source_inputs,
        )

    def extra_repr(self) -> str:
        return (
            "formula='state * coefficient.unsqueeze(-1)', "
            "input_state_layout='[E, H]', "
            "input_coefficient_layout='[E]', "
            "aggregation_owned_here=False, "
            "parameter_free=True"
        )


# =============================================================================
# Builders and aliases
# =============================================================================


def build_edge_message_composer(
    *,
    relation_state_gather: RelationStateGather | None = None,
) -> EdgeMessageComposer:
    """
    Construct the parameter-free edge-message composer.
    """

    return EdgeMessageComposer(
        relation_state_gather=(
            relation_state_gather
        )
    )


def compose_edge_messages(
    *,
    relation_transform: RelationTransformOutput,
    resolved_coefficients: ResolvedMessageCoefficients,
    source_inputs: FunctionalMessagePassingInputs | None = None,
) -> MessageCompositionOutput:
    """
    Functional spelling for complete edge-message composition.
    """

    return compose_message_output(
        relation_transform=(
            relation_transform
        ),
        resolved_coefficients=(
            resolved_coefficients
        ),
        source_inputs=source_inputs,
    )


MessageComposer = EdgeMessageComposer
FunctionalMessageComposer = EdgeMessageComposer
build_message_composer = build_edge_message_composer
compose_message_vectors = compose_edge_message_tensor


__all__ = (
    # Public identity.
    "MESSAGE_COMPOSER_SCHEMA_VERSION",
    "MESSAGE_COMPOSER_OPERATION",
    "MESSAGE_COMPOSER_OPERATION_ORDER",
    "MESSAGE_COMPOSER_INPUT_STATE_LAYOUT",
    "MESSAGE_COMPOSER_INPUT_COEFFICIENT_LAYOUT",
    "MESSAGE_COMPOSER_OUTPUT_LAYOUT",
    "MESSAGE_COMPOSER_BROADCAST_AXIS",
    "MESSAGE_COMPOSER_PARAMETER_FREE",
    "MESSAGE_COMPOSER_BUFFER_FREE",
    "MESSAGE_COMPOSER_AGGREGATION_OWNED_HERE",
    # Architecture helpers.
    "message_composer_architecture_dict",
    "message_composer_architecture_fingerprint",
    # Numerical and complete composition.
    "compose_edge_message_tensor",
    "compose_message_vectors",
    "compose_message_output",
    "compose_edge_messages",
    "validate_message_composition_output",
    "message_composition_diagnostic_summary",
    # Module API.
    "EdgeMessageComposer",
    "MessageComposer",
    "FunctionalMessageComposer",
    "build_edge_message_composer",
    "build_message_composer",
)
