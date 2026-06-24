"""
Zero-copy resolution of edge-aligned relation-transformed source states.

Target path:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            functional_message_passing/
                message_builders/
                    relation_state_gather.py

Historical name
---------------
The planned message-builder decomposition originally included a
``relation_state_gather.py`` stage that would select, for every edge, the
correct relation-transformed source representation from a node/relation tensor.

The completed relation-transform subsystem already performs that operation and
publishes:

    RelationTransformOutput.transformed_source_state  # [E, H]

Consequently, repeating the indexing operation here would be computationally 
undesirable. It would create two possible owners of exact
relation dispatch and two opportunities for source/relation-axis
misalignment.

The historical file name is retained because this module still owns the
message-builder boundary at which relation-transformed source states are
resolved. In bounded V2.0, that resolution is deliberately:

    validate -> preserve exact tensor identity -> return

No tensor is indexed, cloned, detached, cast, moved, copied, transformed, or
reordered.

Contract
-------------------
For every stored directed edge ``e = (s_e -> t_e)`` with dense exact relation
index ``r_e``, the upstream relation-transform subsystem has already produced:

    u_e = T_{r_e}(h_{s_e})

and stored the result in edge order:

    transformed_source_state[e] = u_e

This module verifies that:

- the object is a ``RelationTransformOutput``;
- the source-input lineage is exact;
- the tensor has shape ``[E, H]``;
- edge order agrees with the source graph;
- hidden width agrees with the fused node-state width;
- dtype and device agree with ``FunctionalMessagePassingInputs``;
- all values are finite;
- relation names and stable IDs remain those of the compiled registry;
- the exact tensor object is returned without recomputation.

The final message-composition stage may then compute:

    edge_messages
        = transformed_source_state
        * combined_coefficient.unsqueeze(-1)

Scope exclusions
----------------
This module does not own:

- source-node gathering;
- exact-relation dispatch;
- relation-transform parameterization;
- relation-family pooling;
- structural edge normalization;
- relation gating;
- edge attention;
- semantic edge weighting;
- message coefficient multiplication;
- target-node aggregation;
- residual updates;
- explanation or causal claims.

Those responsibilities remain in their dedicated subsystems.

Why retain a module for an identity boundary?
---------------------------------------------
The boundary remains valuable because it makes an important architectural
invariant executable:

    the message builder consumes the exact edge-aligned output of the
    relation-transform subsystem

rather than a reconstructed, copied, or independently gathered equivalent.

That invariant improves:

- lineage auditing;
- gradient continuity;
- memory efficiency;
- edge-order safety;
- separation of scientific responsibilities;
- future refactoring safety.

The module is parameter-free and buffer-free.
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
from .schemas import (
    MESSAGE_TRANSFORM_INPUT_LAYOUT,
)


# =============================================================================
# Public identity
# =============================================================================


RELATION_STATE_GATHER_SCHEMA_VERSION: Final[str] = "0.1"

RELATION_STATE_GATHER_OPERATION: Final[str] = (
    "zero_copy_resolution_of_edge_aligned_relation_transform_output"
)

RELATION_STATE_GATHER_INPUT_LAYOUT: Final[str] = (
    MESSAGE_TRANSFORM_INPUT_LAYOUT
)

RELATION_STATE_GATHER_OUTPUT_LAYOUT: Final[str] = (
    "edge_aligned_transformed_source_state_[E,H]"
)

RELATION_STATE_GATHER_OWNER: Final[str] = (
    "relation_transform_subsystem"
)

RELATION_STATE_GATHER_INDEXING_OWNED_HERE: Final[bool] = False
RELATION_STATE_GATHER_ZERO_COPY_REQUIRED: Final[bool] = True
RELATION_STATE_GATHER_PARAMETER_FREE: Final[bool] = True

RELATION_STATE_GATHER_OPERATION_ORDER: Final[
    tuple[str, ...]
] = (
    "validate_relation_transform_output",
    "validate_exact_source_input_lineage",
    "validate_edge_aligned_shape_dtype_device_and_finiteness",
    "validate_exact_relation_axis_metadata",
    "return_exact_transformed_source_state_tensor",
)


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


def _require_float_matrix(
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

    if value.ndim != 2:
        raise ValueError(
            f"{name} must have rank 2 and shape [E, H]; "
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


def _require_exact_shape(
    name: str,
    value: torch.Tensor,
    expected: tuple[int, int],
) -> None:
    observed = tuple(
        int(size)
        for size in value.shape
    )

    if observed != expected:
        raise ValueError(
            f"{name} must have shape {expected}; "
            f"observed {observed}."
        )


def _require_nonempty_relation_axis(
    source_inputs: FunctionalMessagePassingInputs,
) -> None:
    if source_inputs.num_relations <= 0:
        raise ValueError(
            "Relation-state resolution requires at least one exact "
            "compiled relation."
        )

    if len(
        source_inputs.relation_names
    ) != source_inputs.num_relations:
        raise ValueError(
            "source_inputs.relation_names length differs from "
            "source_inputs.num_relations."
        )

    if len(
        source_inputs.stable_relation_ids
    ) != source_inputs.num_relations:
        raise ValueError(
            "source_inputs.stable_relation_ids length differs from "
            "source_inputs.num_relations."
        )

    if len(
        set(source_inputs.relation_names)
    ) != source_inputs.num_relations:
        raise ValueError(
            "source_inputs.relation_names must be unique."
        )

    if len(
        set(source_inputs.stable_relation_ids)
    ) != source_inputs.num_relations:
        raise ValueError(
            "source_inputs.stable_relation_ids must be unique."
        )

    for index, name in enumerate(
        source_inputs.relation_names
    ):
        _require_nonempty_string(
            f"source_inputs.relation_names[{index}]",
            name,
        )


# =============================================================================
# Functional architecture metadata
# =============================================================================


def relation_state_gather_architecture_dict() -> dict[str, Any]:
    """
    Return the frozen architecture contract for this zero-copy boundary.
    """

    return {
        "schema_version": (
            RELATION_STATE_GATHER_SCHEMA_VERSION
        ),
        "operation": (
            RELATION_STATE_GATHER_OPERATION
        ),
        "operation_order": list(
            RELATION_STATE_GATHER_OPERATION_ORDER
        ),
        "input_layout": (
            RELATION_STATE_GATHER_INPUT_LAYOUT
        ),
        "output_layout": (
            RELATION_STATE_GATHER_OUTPUT_LAYOUT
        ),
        "edge_alignment_owner": (
            RELATION_STATE_GATHER_OWNER
        ),
        "indexing_owned_here": (
            RELATION_STATE_GATHER_INDEXING_OWNED_HERE
        ),
        "zero_copy_required": (
            RELATION_STATE_GATHER_ZERO_COPY_REQUIRED
        ),
        "parameter_free": (
            RELATION_STATE_GATHER_PARAMETER_FREE
        ),
        "buffer_free": True,
        "source_node_gather_owned_here": False,
        "relation_dispatch_owned_here": False,
        "relation_transform_owned_here": False,
        "relation_family_pooling_owned_here": False,
        "message_composition_owned_here": False,
        "aggregation_owned_here": False,
        "claims_causal_importance": False,
    }


def relation_state_gather_architecture_fingerprint() -> str:
    """
    Fingerprint the static resolver architecture.
    """

    return _fingerprint(
        relation_state_gather_architecture_dict()
    )


# =============================================================================
# Exact edge-aligned boundary validation
# =============================================================================


def validate_edge_aligned_relation_state(
    *,
    relation_transform: RelationTransformOutput,
    source_inputs: FunctionalMessagePassingInputs | None = None,
) -> None:
    """
    Validate the relation-transform output consumed by message construction.

    Parameters
    ----------
    relation_transform:
        Complete upstream relation-transform output.
    source_inputs:
        Optional exact source-input object expected by the caller. When
        supplied, object identity is required rather than merely equivalent
        values.
    """

    _require_relation_transform(
        relation_transform
    )
    _require_source_inputs(
        relation_transform.source_inputs
    )

    resolved_source_inputs = (
        relation_transform.source_inputs
    )

    if source_inputs is not None:
        _require_source_inputs(
            source_inputs
        )

        if resolved_source_inputs is not (
            source_inputs
        ):
            raise ValueError(
                "relation_transform must reference the exact supplied "
                "source_inputs object."
            )

    _require_nonempty_relation_axis(
        resolved_source_inputs
    )

    transformed = (
        relation_transform
        .transformed_source_state
    )

    _require_float_matrix(
        "relation_transform.transformed_source_state",
        transformed,
    )
    _require_exact_shape(
        "relation_transform.transformed_source_state",
        transformed,
        (
            resolved_source_inputs.num_edges,
            resolved_source_inputs.hidden_dim,
        ),
    )

    if transformed.device != (
        resolved_source_inputs.device
    ):
        raise ValueError(
            "relation_transform.transformed_source_state and "
            "source_inputs must share one device."
        )

    if transformed.dtype != (
        resolved_source_inputs.dtype
    ):
        raise ValueError(
            "relation_transform.transformed_source_state and "
            "source_inputs must share one dtype."
        )

    if relation_transform.num_edges != (
        resolved_source_inputs.num_edges
    ):
        raise ValueError(
            "relation_transform.num_edges differs from "
            "source_inputs.num_edges."
        )

    if relation_transform.hidden_dim != (
        resolved_source_inputs.hidden_dim
    ):
        raise ValueError(
            "relation_transform.hidden_dim differs from "
            "source_inputs.hidden_dim."
        )

    _require_nonempty_string(
        "relation_transform.transform_mode",
        relation_transform.transform_mode,
    )
    _require_nonempty_string(
        "relation_transform.encoder_architecture_fingerprint",
        relation_transform.encoder_architecture_fingerprint,
    )

    if (
        relation_transform.parameter_fingerprint
        is not None
    ):
        _require_nonempty_string(
            "relation_transform.parameter_fingerprint",
            relation_transform.parameter_fingerprint,
        )

    unexpected_relations = sorted(
        set(
            relation_transform
            .relation_parameter_fingerprints
        )
        - set(
            resolved_source_inputs
            .relation_names
        )
    )

    if unexpected_relations:
        raise ValueError(
            "relation_transform.relation_parameter_fingerprints "
            "contains relations outside the exact compiled relation "
            f"axis: {unexpected_relations}."
        )

    if resolved_source_inputs.source_index.shape != (
        resolved_source_inputs.num_edges,
    ):
        raise ValueError(
            "source_inputs.source_index must remain edge aligned with "
            "shape [E]."
        )

    if (
        resolved_source_inputs
        .edge_relation_index
        .shape
        != (
            resolved_source_inputs.num_edges,
        )
    ):
        raise ValueError(
            "source_inputs.edge_relation_index must remain edge aligned "
            "with shape [E]."
        )


def assert_zero_copy_relation_state(
    *,
    relation_transform: RelationTransformOutput,
    resolved_state: torch.Tensor,
) -> None:
    """
    Require exact tensor-object preservation across the boundary.
    """

    _require_relation_transform(
        relation_transform
    )
    _require_float_matrix(
        "resolved_state",
        resolved_state,
    )

    if resolved_state is not (
        relation_transform
        .transformed_source_state
    ):
        raise ValueError(
            "resolved_state must be the exact "
            "relation_transform.transformed_source_state tensor object. "
            "Cloning, detaching, casting, moving, or recomputing the "
            "edge-aligned relation state is not permitted."
        )


def resolve_edge_aligned_relation_state(
    relation_transform: RelationTransformOutput,
    *,
    source_inputs: FunctionalMessagePassingInputs | None = None,
) -> torch.Tensor:
    """
    Return the exact edge-aligned relation-transformed source-state tensor.

    This function deliberately performs no numerical operation.
    """

    validate_edge_aligned_relation_state(
        relation_transform=relation_transform,
        source_inputs=source_inputs,
    )

    resolved = (
        relation_transform
        .transformed_source_state
    )

    assert_zero_copy_relation_state(
        relation_transform=relation_transform,
        resolved_state=resolved,
    )

    return resolved


# =============================================================================
# Descriptive diagnostics
# =============================================================================


def relation_state_gather_diagnostic_summary(
    *,
    relation_transform: RelationTransformOutput,
    source_inputs: FunctionalMessagePassingInputs | None = None,
) -> dict[str, Any]:
    """
    Return compact descriptive metadata without retaining new tensors.

    Diagnostics do not assign causal or explanatory meaning to transformed
    dimensions or relation-specific parameterization.
    """

    resolved = resolve_edge_aligned_relation_state(
        relation_transform,
        source_inputs=source_inputs,
    )
    inputs = relation_transform.source_inputs

    relation_specific_parameter_count = len(
        relation_transform
        .relation_parameter_fingerprints
    )

    return {
        "schema_version": (
            RELATION_STATE_GATHER_SCHEMA_VERSION
        ),
        "operation": (
            RELATION_STATE_GATHER_OPERATION
        ),
        "input_layout": (
            RELATION_STATE_GATHER_INPUT_LAYOUT
        ),
        "output_layout": (
            RELATION_STATE_GATHER_OUTPUT_LAYOUT
        ),
        "num_edges": inputs.num_edges,
        "hidden_dim": inputs.hidden_dim,
        "num_relations": inputs.num_relations,
        "relation_names": list(
            inputs.relation_names
        ),
        "stable_relation_ids": list(
            inputs.stable_relation_ids
        ),
        "transform_mode": (
            relation_transform.transform_mode
        ),
        "relation_specific_parameter_fingerprint_count": (
            relation_specific_parameter_count
        ),
        "dtype": str(resolved.dtype),
        "device": str(resolved.device),
        "zero_copy_identity_preserved": (
            resolved
            is relation_transform
            .transformed_source_state
        ),
        "requires_grad": (
            resolved.requires_grad
        ),
        "parameter_free_boundary": True,
        "indexing_performed_here": False,
        "relation_dispatch_performed_here": False,
        "finite": bool(
            torch.isfinite(resolved)
            .all()
            .item()
        ),
        "causal_importance_claim": False,
        "explanation_faithfulness_claim": False,
    }


# =============================================================================
# Parameter-free module wrapper
# =============================================================================


class RelationStateGather(nn.Module):
    """
    Parameter-free, zero-copy message-builder boundary.

    The class exists for uniform orchestration with the later message-builder
    module and for explicit architecture fingerprinting. It does not gather
    from node tensors because that work is already complete upstream.
    """

    def __init__(self) -> None:
        super().__init__()
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
        return relation_state_gather_architecture_dict()

    def architecture_fingerprint(
        self,
    ) -> str:
        return relation_state_gather_architecture_fingerprint()

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
                "RelationStateGather must remain parameter-free. "
                f"Observed parameters: "
                f"{tuple(name for name, _ in parameters)}."
            )

        if buffers:
            raise RuntimeError(
                "RelationStateGather must remain buffer-free. "
                f"Observed buffers: "
                f"{tuple(name for name, _ in buffers)}."
            )

        if state:
            raise RuntimeError(
                "RelationStateGather must have an empty state_dict."
            )

        if self.parameter_count != 0:
            raise RuntimeError(
                "RelationStateGather parameter_count must be zero."
            )

        if self.trainable_parameter_count != 0:
            raise RuntimeError(
                "RelationStateGather trainable_parameter_count must be "
                "zero."
            )

        if self.buffer_count != 0:
            raise RuntimeError(
                "RelationStateGather buffer_count must be zero."
            )

    def resolve(
        self,
        relation_transform: RelationTransformOutput,
        *,
        source_inputs: FunctionalMessagePassingInputs | None = None,
    ) -> torch.Tensor:
        """
        Resolve the exact edge-aligned tensor.
        """

        self.assert_parameter_free()

        return resolve_edge_aligned_relation_state(
            relation_transform,
            source_inputs=source_inputs,
        )

    def diagnostic_summary(
        self,
        relation_transform: RelationTransformOutput,
        *,
        source_inputs: FunctionalMessagePassingInputs | None = None,
    ) -> dict[str, Any]:
        """
        Return compact boundary diagnostics.
        """

        self.assert_parameter_free()

        return relation_state_gather_diagnostic_summary(
            relation_transform=relation_transform,
            source_inputs=source_inputs,
        )

    def forward(
        self,
        relation_transform: RelationTransformOutput,
        *,
        source_inputs: FunctionalMessagePassingInputs | None = None,
    ) -> torch.Tensor:
        """
        Return the exact upstream transformed-source-state tensor.
        """

        return self.resolve(
            relation_transform,
            source_inputs=source_inputs,
        )

    def extra_repr(self) -> str:
        return (
            "operation='zero_copy_resolution', "
            "input_layout='[E, H]', "
            "indexing_owned_here=False, "
            "parameter_free=True"
        )


# =============================================================================
# Construction and compact aliases
# =============================================================================


def build_relation_state_gather() -> RelationStateGather:
    """
    Construct the parameter-free relation-state boundary.
    """

    return RelationStateGather()


def gather_relation_state(
    relation_transform: RelationTransformOutput,
    *,
    source_inputs: FunctionalMessagePassingInputs | None = None,
) -> torch.Tensor:
    """
    Compatibility spelling for zero-copy relation-state resolution.

    Despite the historical verb ``gather``, no indexing occurs here.
    """

    return resolve_edge_aligned_relation_state(
        relation_transform,
        source_inputs=source_inputs,
    )


def validate_relation_state_gather_contract(
    *,
    gatherer: RelationStateGather,
    relation_transform: RelationTransformOutput,
    resolved_state: torch.Tensor,
    source_inputs: FunctionalMessagePassingInputs | None = None,
) -> None:
    """
    Validate one complete resolver invocation.
    """

    if not isinstance(
        gatherer,
        RelationStateGather,
    ):
        raise TypeError(
            "gatherer must be a RelationStateGather."
        )

    gatherer.assert_parameter_free()

    validate_edge_aligned_relation_state(
        relation_transform=relation_transform,
        source_inputs=source_inputs,
    )
    assert_zero_copy_relation_state(
        relation_transform=relation_transform,
        resolved_state=resolved_state,
    )


EdgeAlignedRelationStateResolver = RelationStateGather
RelationStateResolver = RelationStateGather
build_relation_state_resolver = build_relation_state_gather
resolve_relation_state = resolve_edge_aligned_relation_state


__all__ = (
    # Public identity.
    "RELATION_STATE_GATHER_SCHEMA_VERSION",
    "RELATION_STATE_GATHER_OPERATION",
    "RELATION_STATE_GATHER_INPUT_LAYOUT",
    "RELATION_STATE_GATHER_OUTPUT_LAYOUT",
    "RELATION_STATE_GATHER_OWNER",
    "RELATION_STATE_GATHER_INDEXING_OWNED_HERE",
    "RELATION_STATE_GATHER_ZERO_COPY_REQUIRED",
    "RELATION_STATE_GATHER_PARAMETER_FREE",
    "RELATION_STATE_GATHER_OPERATION_ORDER",
    # Functional helpers.
    "relation_state_gather_architecture_dict",
    "relation_state_gather_architecture_fingerprint",
    "validate_edge_aligned_relation_state",
    "assert_zero_copy_relation_state",
    "resolve_edge_aligned_relation_state",
    "resolve_relation_state",
    "gather_relation_state",
    "relation_state_gather_diagnostic_summary",
    "validate_relation_state_gather_contract",
    # Module API.
    "RelationStateGather",
    "RelationStateResolver",
    "EdgeAlignedRelationStateResolver",
    "build_relation_state_gather",
    "build_relation_state_resolver",
)
