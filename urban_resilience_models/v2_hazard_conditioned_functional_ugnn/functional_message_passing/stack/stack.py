"""
Complete orchestration for multi-layer functional message passing.

Target path:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            functional_message_passing/
                stack/
                    stack.py

This module coordinates repeated execution of already-complete
``FunctionalMessagePassingLayer`` modules:

    original FunctionalMessagePassingInputs
        -> layer 0
        -> immutable node-state rebinding
        -> layer 1
        -> ...
        -> layer L - 1
        -> FunctionalMessagePassingStackOutput

Ownership boundary
------------------
This file owns:

- stack module registration;
- independent versus fully shared layer ownership;
- zero-based runtime layer indices;
- repeated complete layer execution;
- immutable depth-wise node-state rebinding;
- policy-selected public layer-output retention;
- optional explicit complete audit traces;
- preservation of one regularization mapping per depth;
- stack architecture, parameter, execution, and lineage fingerprints;
- public stack-output assembly;
- optional explicit stack diagnostics.

It does not reimplement:

- relation transforms;
- structural edge normalization;
- relation gating;
- edge attention;
- message composition;
- target-node aggregation;
- residual updates;
- layer normalization;
- prediction or readout;
- training-loss reduction.

State rebinding
---------------
Layer 0 consumes the exact original ``FunctionalMessagePassingInputs`` object.

For every later depth, this module constructs a new immutable
``FunctionalMessagePassingInputs`` object. It preserves exact identity for the
graph, compiled relation registry, relation-family alignment, hazard query,
and optional compiled relation priors. Only ``node_state`` and its lineage
change.

The evolved node-state tensor is the exact previous layer
``updated_node_state`` tensor. It is never cloned, detached, cast, or moved.
This preserves end-to-end autograd across stack depth.

Sharing
-------
``independent``
    Layers are registered under ``layers.<depth>`` and own distinct parameters.

``fully_shared``
    One layer is registered under ``shared_layer`` and reused at every depth.

The stack relies on ``StackLayerSharingPlan`` to reject accidental module,
Parameter-object, and parameter-storage aliases.

Trace and retention
-------------------
Stack output retention and one-layer trace detail remain orthogonal.

Ordinary execution retains only policy-selected public layer outputs.
Complete layer runs are retained only when explicit audit mode is enabled.

Diagnostics
-----------
Diagnostics are never invoked by ordinary ``forward``. They are generated only
through explicit methods and return tensor-free reports.

Limits
-----------------
Repeated state transitions, retained layer outputs, attention values, gates,
regularization terms, update magnitudes, and diagnostic alerts are descriptive
model quantities. They do not automatically establish causal influence,
faithful explanation, calibrated uncertainty, counterfactual validity, or
mechanistic identifiability.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from hashlib import sha256
import json
import math
from typing import Any, Final, Mapping

import torch
from torch import nn

from ...constants import (
    STACK_RETENTION_NONE,
    STACK_SHARING_FULLY_SHARED,
    STACK_SHARING_INDEPENDENT,
)
from ..layer.layer import (
    FunctionalMessagePassingLayer,
    FunctionalMessagePassingLayerRun,
    validate_functional_message_passing_layer_run,
)
from ..layer.schemas import (
    LAYER_TRACE_NONE,
    LayerTracePolicy,
)
from ..schemas import (
    FMP_NODE_STATE_SOURCE_LAYER_OUTPUT,
    FunctionalMessagePassingInputs,
    FunctionalMessagePassingNodeState,
)
from .diagnostics import (
    FunctionalMessagePassingStackDiagnostics,
)
from .schemas import (
    FUNCTIONAL_MESSAGE_PASSING_STACK_OPERATION_ORDER,
    FUNCTIONAL_MESSAGE_PASSING_STACK_SCIENTIFIC_INTERPRETATION,
    FunctionalMessagePassingStackComputationOutput,
    FunctionalMessagePassingStackDepthRecord,
    FunctionalMessagePassingStackInputs,
    FunctionalMessagePassingStackOutput,
    FunctionalMessagePassingStackRun,
    FunctionalMessagePassingStackRunWithDiagnostics,
    FunctionalMessagePassingStackTrace,
    assemble_functional_message_passing_stack_output,
    validate_functional_message_passing_stack_run,
)
from .sharing_policy import (
    StackLayerSharingPlan,
    StackSharingPolicy,
    build_stack_layer_sharing_plan,
    build_stack_layer_sharing_plan_from_factory,
    validate_stack_layer_sharing_plan,
)
from .trace_policy import (
    StackRetentionPolicy,
    StackTracePolicy,
    resolve_stack_trace_policy,
)


# =============================================================================
# Public identity
# =============================================================================


FUNCTIONAL_MESSAGE_PASSING_STACK_MODULE_SCHEMA_VERSION: Final[str] = "0.1"

FUNCTIONAL_MESSAGE_PASSING_STACK_STACKING_OWNED_HERE: Final[bool] = True
FUNCTIONAL_MESSAGE_PASSING_STACK_LAYER_MATH_OWNED_HERE: Final[bool] = False
FUNCTIONAL_MESSAGE_PASSING_STACK_PREDICTION_OWNED_HERE: Final[bool] = False

FUNCTIONAL_MESSAGE_PASSING_STACK_STATE_REBINDING_CLONES_TENSOR: Final[
    bool
] = False
FUNCTIONAL_MESSAGE_PASSING_STACK_STATE_REBINDING_DETACHES_TENSOR: Final[
    bool
] = False
FUNCTIONAL_MESSAGE_PASSING_STACK_STATE_REBINDING_CASTS_TENSOR: Final[
    bool
] = False
FUNCTIONAL_MESSAGE_PASSING_STACK_STATE_REBINDING_MOVES_TENSOR: Final[
    bool
] = False

FUNCTIONAL_MESSAGE_PASSING_STACK_REGULARIZATION_REDUCTION: Final[str] = (
    "preserve_one_mapping_per_depth_without_reduction"
)
FUNCTIONAL_MESSAGE_PASSING_STACK_AUDIT_RETENTION: Final[str] = (
    "complete_layer_runs_retained_only_when_audit_mode_enabled"
)

FUNCTIONAL_MESSAGE_PASSING_STACK_INDEPENDENT_REGISTRATION_PREFIX: Final[
    str
] = "layers"
FUNCTIONAL_MESSAGE_PASSING_STACK_FULLY_SHARED_REGISTRATION_PREFIX: Final[
    str
] = "shared_layer"

FUNCTIONAL_MESSAGE_PASSING_STACK_OUTPUT_SCHEMA: Final[str] = (
    "FunctionalMessagePassingStackOutput"
)


StackLayerFactory = Callable[
    [int],
    FunctionalMessagePassingLayer,
]


# =============================================================================
# Generic helpers
# =============================================================================


def _to_plain_json_value(
    value: Any,
) -> Any:
    if isinstance(
        value,
        Mapping,
    ):
        return {
            str(key): _to_plain_json_value(
                child
            )
            for key, child in value.items()
        }

    if isinstance(
        value,
        (
            tuple,
            list,
        ),
    ):
        return [
            _to_plain_json_value(
                child
            )
            for child in value
        ]

    return value


def _canonical_json(
    payload: Mapping[str, Any],
) -> str:
    return json.dumps(
        _to_plain_json_value(
            payload
        ),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )



def _fingerprint(
    payload: dict[str, Any],
) -> str:
    return sha256(
        _canonical_json(payload).encode(
            "utf-8"
        )
    ).hexdigest()


def _tensor_value_fingerprint(
    value: torch.Tensor,
    *,
    name: str,
) -> str:
    if not isinstance(
        value,
        torch.Tensor,
    ):
        raise TypeError(
            f"{name} must be a tensor."
        )

    detached = (
        value
        .detach()
        .cpu()
        .contiguous()
    )

    digest = sha256()
    digest.update(
        name.encode("utf-8")
    )
    digest.update(
        str(detached.dtype).encode(
            "utf-8"
        )
    )
    digest.update(
        json.dumps(
            list(
                detached.shape
            ),
            separators=(",", ":"),
        ).encode("utf-8")
    )
    digest.update(
        detached
        .view(torch.uint8)
        .numpy()
        .tobytes()
    )
    return digest.hexdigest()


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


def _require_optional_nonempty_string(
    name: str,
    value: str | None,
) -> None:
    if value is not None:
        _require_nonempty_string(
            name,
            value,
        )


def _require_positive_int(
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

    if value <= 0:
        raise ValueError(
            f"{name} must be strictly positive."
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


def _require_source_inputs(
    value: FunctionalMessagePassingInputs,
    *,
    name: str = "source_inputs",
) -> None:
    if not isinstance(
        value,
        FunctionalMessagePassingInputs,
    ):
        raise TypeError(
            f"{name} must be a FunctionalMessagePassingInputs."
        )


def _require_layer_run(
    value: FunctionalMessagePassingLayerRun,
    *,
    name: str = "previous_layer_run",
) -> None:
    if not isinstance(
        value,
        FunctionalMessagePassingLayerRun,
    ):
        raise TypeError(
            f"{name} must be a FunctionalMessagePassingLayerRun."
        )

    validate_functional_message_passing_layer_run(
        value
    )


def _devices_match(
    left: torch.device,
    right: torch.device,
) -> bool:
    if left.type != right.type:
        return False

    if left.type != "cuda":
        return left == right

    if left.index is None or right.index is None:
        return True

    return left.index == right.index


def _resolve_method_or_property(
    value: object,
    name: str,
) -> Any:
    resolved = getattr(
        value,
        name,
        None,
    )

    if callable(
        resolved
    ):
        resolved = resolved()

    return resolved


def _resolve_nonempty_fingerprint(
    value: object,
    name: str,
    *,
    owner_name: str,
) -> str:
    fingerprint = (
        _resolve_method_or_property(
            value,
            name,
        )
    )

    _require_nonempty_string(
        f"{owner_name}.{name}",
        fingerprint,
    )
    return fingerprint


def _resolve_lineage_fingerprint(
    value: object,
    *,
    name: str,
) -> str:
    return _resolve_nonempty_fingerprint(
        value,
        "lineage_fingerprint",
        owner_name=name,
    )


def _assert_finite_module(
    module: nn.Module,
    *,
    name: str,
) -> None:
    if not isinstance(
        module,
        nn.Module,
    ):
        raise TypeError(
            f"{name} must be an nn.Module."
        )

    method = getattr(
        module,
        "assert_finite_parameters",
        None,
    )
    if callable(
        method
    ):
        method()

    for parameter_name, parameter in (
        module.named_parameters()
    ):
        if not bool(
            torch.isfinite(
                parameter
            ).all().item()
        ):
            raise FloatingPointError(
                f"{name} parameter {parameter_name!r} contains "
                "non-finite values."
            )

    for buffer_name, buffer in (
        module.named_buffers()
    ):
        if (
            buffer.dtype.is_floating_point
            and not bool(
                torch.isfinite(
                    buffer
                ).all().item()
            )
        ):
            raise FloatingPointError(
                f"{name} buffer {buffer_name!r} contains non-finite values."
            )


def _validate_exact_structural_context(
    *,
    original_inputs: FunctionalMessagePassingInputs,
    candidate_inputs: FunctionalMessagePassingInputs,
    name: str,
) -> None:
    _require_source_inputs(
        original_inputs,
        name="original_inputs",
    )
    _require_source_inputs(
        candidate_inputs,
        name=name,
    )

    identity_fields = (
        (
            "source_graph",
            original_inputs.source_graph,
            candidate_inputs.source_graph,
        ),
        (
            "compiled_relation_registry",
            (
                original_inputs
                .compiled_relation_registry
            ),
            (
                candidate_inputs
                .compiled_relation_registry
            ),
        ),
        (
            "relation_families",
            original_inputs.relation_families,
            candidate_inputs.relation_families,
        ),
        (
            "hazard_query",
            original_inputs.hazard_query,
            candidate_inputs.hazard_query,
        ),
        (
            "compiled_relation_priors",
            (
                original_inputs
                .compiled_relation_priors
            ),
            (
                candidate_inputs
                .compiled_relation_priors
            ),
        ),
    )

    for (
        field_name,
        expected,
        observed,
    ) in identity_fields:
        if observed is not expected:
            raise ValueError(
                f"{name}.{field_name} must preserve exact structural "
                "context object identity."
            )

    if candidate_inputs.num_nodes != (
        original_inputs.num_nodes
    ):
        raise ValueError(
            f"{name} must preserve the original node count."
        )

    if candidate_inputs.num_edges != (
        original_inputs.num_edges
    ):
        raise ValueError(
            f"{name} must preserve the original edge count."
        )

    if candidate_inputs.num_graphs != (
        original_inputs.num_graphs
    ):
        raise ValueError(
            f"{name} must preserve the original graph count."
        )

    if candidate_inputs.hidden_dim != (
        original_inputs.hidden_dim
    ):
        raise ValueError(
            f"{name} must preserve the original hidden width."
        )

    if candidate_inputs.dtype != (
        original_inputs.dtype
    ):
        raise ValueError(
            f"{name} must preserve the original floating-point dtype."
        )

    if not _devices_match(
        candidate_inputs.device,
        original_inputs.device,
    ):
        raise ValueError(
            f"{name} must preserve the original device."
        )

    if candidate_inputs.relation_names != (
        original_inputs.relation_names
    ):
        raise ValueError(
            f"{name} must preserve exact relation ordering."
        )

    if candidate_inputs.stable_relation_ids != (
        original_inputs.stable_relation_ids
    ):
        raise ValueError(
            f"{name} must preserve stable relation identities."
        )

    if candidate_inputs.source_graph.edge_index is not (
        original_inputs.source_graph.edge_index
    ):
        raise ValueError(
            f"{name} must preserve exact edge_index tensor identity."
        )

    if (
        candidate_inputs
        .source_graph
        .edge_relation_type
        is not original_inputs
        .source_graph
        .edge_relation_type
    ):
        raise ValueError(
            f"{name} must preserve exact edge-relation tensor identity."
        )

    if candidate_inputs.node_batch_index is not (
        original_inputs.node_batch_index
    ):
        raise ValueError(
            f"{name} must preserve exact node_batch_index tensor identity."
        )

    if (
        candidate_inputs
        .node_state
        .alignment
        is not original_inputs
        .node_state
        .alignment
    ):
        raise ValueError(
            f"{name} must preserve exact node-alignment object identity."
        )


# =============================================================================
# State rebinding
# =============================================================================


def derive_next_functional_message_passing_inputs(
    *,
    original_stack_inputs: FunctionalMessagePassingInputs,
    previous_layer_run: FunctionalMessagePassingLayerRun,
    source_stack_fingerprint: str,
    next_layer_index: int | None = None,
) -> FunctionalMessagePassingInputs:
    """
    Derive the immutable source inputs for the next stack depth.

    The returned node-state tensor is the exact previous
    ``updated_node_state`` tensor. No clone, detach, cast, or device movement
    occurs.
    """

    _require_source_inputs(
        original_stack_inputs,
        name="original_stack_inputs",
    )
    _require_layer_run(
        previous_layer_run
    )
    _require_nonempty_string(
        "source_stack_fingerprint",
        source_stack_fingerprint,
    )

    expected_next_index = (
        previous_layer_run.layer_index
        + 1
    )

    if next_layer_index is None:
        resolved_next_index = (
            expected_next_index
        )
    else:
        _require_nonnegative_int(
            "next_layer_index",
            next_layer_index,
        )
        resolved_next_index = (
            next_layer_index
        )

    if resolved_next_index != (
        expected_next_index
    ):
        raise ValueError(
            "next_layer_index must immediately follow the previous layer "
            "run index."
        )

    _validate_exact_structural_context(
        original_inputs=(
            original_stack_inputs
        ),
        candidate_inputs=(
            previous_layer_run
            .source_inputs
        ),
        name="previous_layer_run.source_inputs",
    )

    previous_output = (
        previous_layer_run
        .public_output
    )
    previous_internal = (
        previous_layer_run
        .internal_output
    )
    previous_state = (
        previous_output
        .updated_node_state
    )

    if previous_state is not (
        previous_layer_run
        .updated_node_state
    ):
        raise ValueError(
            "previous_layer_run lost exact updated-node-state identity."
        )

    if tuple(
        previous_state.shape
    ) != (
        original_stack_inputs.num_nodes,
        original_stack_inputs.hidden_dim,
    ):
        raise ValueError(
            "Previous layer updated state does not preserve stack shape."
        )

    if previous_state.dtype != (
        original_stack_inputs.dtype
    ):
        raise ValueError(
            "Previous layer updated state does not preserve stack dtype."
        )

    if not _devices_match(
        previous_state.device,
        original_stack_inputs.device,
    ):
        raise ValueError(
            "Previous layer updated state does not preserve stack device."
        )

    if not bool(
        torch.isfinite(
            previous_state
        ).all().item()
    ):
        raise FloatingPointError(
            "Previous layer updated state contains non-finite values."
        )

    source_architecture_fingerprint = (
        previous_output
        .encoder_architecture_fingerprint
    )
    source_parameter_fingerprint = (
        previous_internal
        .layer_parameter_fingerprint
    )
    source_lineage_fingerprint = (
        previous_output
        .lineage_fingerprint
    )

    _require_nonempty_string(
        "source_architecture_fingerprint",
        source_architecture_fingerprint,
    )
    _require_nonempty_string(
        "source_parameter_fingerprint",
        source_parameter_fingerprint,
    )
    _require_nonempty_string(
        "source_lineage_fingerprint",
        source_lineage_fingerprint,
    )

    node_state = (
        FunctionalMessagePassingNodeState(
            state=previous_state,
            alignment=(
                original_stack_inputs
                .node_state
                .alignment
            ),
            source_kind=(
                FMP_NODE_STATE_SOURCE_LAYER_OUTPUT
            ),
            source_layer_index=(
                previous_layer_run
                .layer_index
            ),
            source_architecture_fingerprint=(
                source_architecture_fingerprint
            ),
            source_lineage_fingerprint=(
                source_lineage_fingerprint
            ),
            source_parameter_fingerprint=(
                source_parameter_fingerprint
            ),
        )
    )

    rebound_source_fingerprint = (
        _fingerprint(
            {
                "contract": (
                    "functional_message_passing_stack_state_rebinding"
                ),
                "source_stack_fingerprint": (
                    source_stack_fingerprint
                ),
                "original_inputs_lineage_fingerprint": (
                    original_stack_inputs
                    .lineage_fingerprint()
                ),
                "previous_layer_index": (
                    previous_layer_run
                    .layer_index
                ),
                "next_layer_index": (
                    resolved_next_index
                ),
                "previous_output_architecture_fingerprint": (
                    source_architecture_fingerprint
                ),
                "previous_output_parameter_fingerprint": (
                    source_parameter_fingerprint
                ),
                "previous_output_lineage_fingerprint": (
                    source_lineage_fingerprint
                ),
                "node_state_lineage_fingerprint": (
                    node_state
                    .lineage_fingerprint
                ),
            }
        )
    )

    rebound = FunctionalMessagePassingInputs(
        source_graph=(
            original_stack_inputs
            .source_graph
        ),
        node_state=node_state,
        compiled_relation_registry=(
            original_stack_inputs
            .compiled_relation_registry
        ),
        relation_families=(
            original_stack_inputs
            .relation_families
        ),
        hazard_query=(
            original_stack_inputs
            .hazard_query
        ),
        compiled_relation_priors=(
            original_stack_inputs
            .compiled_relation_priors
        ),
        source_fingerprint=(
            rebound_source_fingerprint
        ),
    )

    validate_rebound_functional_message_passing_inputs(
        original_stack_inputs=(
            original_stack_inputs
        ),
        rebound_inputs=rebound,
        previous_layer_run=(
            previous_layer_run
        ),
        next_layer_index=(
            resolved_next_index
        ),
    )

    return rebound


def validate_rebound_functional_message_passing_inputs(
    *,
    original_stack_inputs: FunctionalMessagePassingInputs,
    rebound_inputs: FunctionalMessagePassingInputs,
    previous_layer_run: FunctionalMessagePassingLayerRun,
    next_layer_index: int,
) -> None:
    """
    Validate exact structural preservation and previous-state lineage.
    """

    _require_source_inputs(
        original_stack_inputs,
        name="original_stack_inputs",
    )
    _require_source_inputs(
        rebound_inputs,
        name="rebound_inputs",
    )
    _require_layer_run(
        previous_layer_run
    )
    _require_nonnegative_int(
        "next_layer_index",
        next_layer_index,
    )

    if next_layer_index != (
        previous_layer_run.layer_index
        + 1
    ):
        raise ValueError(
            "next_layer_index must immediately follow previous_layer_run."
        )

    _validate_exact_structural_context(
        original_inputs=(
            original_stack_inputs
        ),
        candidate_inputs=(
            rebound_inputs
        ),
        name="rebound_inputs",
    )

    if rebound_inputs is (
        original_stack_inputs
    ):
        raise ValueError(
            "Rebound inputs must be a distinct immutable input object."
        )

    node_state = (
        rebound_inputs.node_state
    )

    if not isinstance(
        node_state,
        FunctionalMessagePassingNodeState,
    ):
        raise TypeError(
            "rebound_inputs.node_state must be a "
            "FunctionalMessagePassingNodeState."
        )

    if node_state.source_kind != (
        FMP_NODE_STATE_SOURCE_LAYER_OUTPUT
    ):
        raise ValueError(
            "Rebound node state must declare source_kind='layer_output'."
        )

    if node_state.source_layer_index != (
        previous_layer_run.layer_index
    ):
        raise ValueError(
            "Rebound node-state source_layer_index differs from the "
            "previous layer run."
        )

    if node_state.state is not (
        previous_layer_run
        .public_output
        .updated_node_state
    ):
        raise ValueError(
            "Rebound node state must preserve exact previous updated-state "
            "tensor identity."
        )

    if node_state.fused_state is not (
        previous_layer_run
        .updated_node_state
    ):
        raise ValueError(
            "Rebound fused_state compatibility view lost tensor identity."
        )

    if (
        node_state
        .source_architecture_fingerprint
        != previous_layer_run
        .public_output
        .encoder_architecture_fingerprint
    ):
        raise ValueError(
            "Rebound node-state source architecture fingerprint differs "
            "from the previous layer output."
        )

    if (
        node_state
        .source_parameter_fingerprint
        != previous_layer_run
        .internal_output
        .layer_parameter_fingerprint
    ):
        raise ValueError(
            "Rebound node-state source parameter fingerprint differs from "
            "the previous layer output."
        )

    if (
        node_state
        .source_lineage_fingerprint
        != previous_layer_run
        .public_output
        .lineage_fingerprint
    ):
        raise ValueError(
            "Rebound node-state source lineage fingerprint differs from "
            "the previous layer output."
        )


# =============================================================================
# Stack orchestrator
# =============================================================================


class FunctionalMessagePassingStack(
    nn.Module
):
    """
    Coordinate a positive-depth sequence of complete FMP layers.

    Parameters
    ----------
    sharing_plan:
        Validated exact ownership plan for all stack depths.

    default_trace_policy:
        Default stack output-retention, layer-trace, and audit policy.
        Observability policy does not alter numerical architecture.

    diagnostics:
        Optional explicit tensor-free stack diagnostics. Ordinary ``forward``
        never invokes diagnostics.
    """

    sharing_plan: StackLayerSharingPlan
    default_trace_policy: StackTracePolicy
    diagnostics: (
        FunctionalMessagePassingStackDiagnostics
        | None
    )

    def __init__(
        self,
        *,
        sharing_plan: StackLayerSharingPlan,
        default_trace_policy: (
            StackTracePolicy
            | None
        ) = None,
        diagnostics: (
            FunctionalMessagePassingStackDiagnostics
            | None
        ) = None,
    ) -> None:
        super().__init__()

        if not isinstance(
            sharing_plan,
            StackLayerSharingPlan,
        ):
            raise TypeError(
                "sharing_plan must be a StackLayerSharingPlan."
            )

        validate_stack_layer_sharing_plan(
            sharing_plan
        )

        resolved_trace_policy = (
            StackTracePolicy()
            if default_trace_policy
            is None
            else default_trace_policy
        )

        if not isinstance(
            resolved_trace_policy,
            StackTracePolicy,
        ):
            raise TypeError(
                "default_trace_policy must be a StackTracePolicy."
            )

        if (
            diagnostics is not None
            and not isinstance(
                diagnostics,
                FunctionalMessagePassingStackDiagnostics,
            )
        ):
            raise TypeError(
                "diagnostics must be a "
                "FunctionalMessagePassingStackDiagnostics or None."
            )

        self.sharing_plan = (
            sharing_plan
        )
        self.default_trace_policy = (
            resolved_trace_policy
        )
        self.diagnostics = diagnostics

        if (
            sharing_plan.sharing_policy
            == STACK_SHARING_INDEPENDENT
        ):
            self.layers = nn.ModuleList(
                sharing_plan
                .layers_by_depth
            )
        elif (
            sharing_plan.sharing_policy
            == STACK_SHARING_FULLY_SHARED
        ):
            self.shared_layer = (
                sharing_plan
                .unique_layers[0]
            )
        else:
            raise RuntimeError(
                "Unreachable stack-sharing registration branch."
            )

        self._assert_static_contract()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_layers(
        cls,
        layers: (
            FunctionalMessagePassingLayer
            | Sequence[
                FunctionalMessagePassingLayer
            ]
        ),
        *,
        num_layers: int,
        sharing_policy: (
            StackSharingPolicy
            | str
        ) = STACK_SHARING_INDEPENDENT,
        default_trace_policy: (
            StackTracePolicy
            | None
        ) = None,
        diagnostics: (
            FunctionalMessagePassingStackDiagnostics
            | None
        ) = None,
        require_uniform_training_mode: bool = True,
    ) -> "FunctionalMessagePassingStack":
        plan = build_stack_layer_sharing_plan(
            layers,
            num_layers=num_layers,
            sharing_policy=(
                sharing_policy
            ),
            require_uniform_training_mode=(
                require_uniform_training_mode
            ),
        )

        return cls(
            sharing_plan=plan,
            default_trace_policy=(
                default_trace_policy
            ),
            diagnostics=diagnostics,
        )

    @classmethod
    def from_factory(
        cls,
        layer_factory: StackLayerFactory,
        *,
        num_layers: int,
        sharing_policy: (
            StackSharingPolicy
            | str
        ) = STACK_SHARING_INDEPENDENT,
        default_trace_policy: (
            StackTracePolicy
            | None
        ) = None,
        diagnostics: (
            FunctionalMessagePassingStackDiagnostics
            | None
        ) = None,
        require_uniform_training_mode: bool = True,
    ) -> "FunctionalMessagePassingStack":
        plan = (
            build_stack_layer_sharing_plan_from_factory(
                layer_factory,
                num_layers=num_layers,
                sharing_policy=(
                    sharing_policy
                ),
                require_uniform_training_mode=(
                    require_uniform_training_mode
                ),
            )
        )

        return cls(
            sharing_plan=plan,
            default_trace_policy=(
                default_trace_policy
            ),
            diagnostics=diagnostics,
        )

    # ------------------------------------------------------------------
    # Public identity
    # ------------------------------------------------------------------

    @property
    def num_layers(self) -> int:
        return self.sharing_plan.num_layers

    @property
    def sharing_policy(self) -> str:
        return (
            self.sharing_plan
            .sharing_policy
        )

    @property
    def hidden_dim(self) -> int:
        return self.sharing_plan.hidden_dim

    @property
    def relation_names(
        self,
    ) -> tuple[str, ...]:
        return (
            self.sharing_plan
            .relation_names
        )

    @property
    def stable_relation_ids(
        self,
    ) -> tuple[int, ...]:
        return (
            self.sharing_plan
            .stable_relation_ids
        )

    @property
    def num_relations(self) -> int:
        return len(
            self.relation_names
        )

    @property
    def compiled_relation_registry_fingerprint(
        self,
    ) -> str:
        return (
            self
            .sharing_plan
            .compiled_relation_registry_fingerprint
        )

    @property
    def num_unique_layers(self) -> int:
        return (
            self.sharing_plan
            .num_unique_layers
        )

    @property
    def depth_to_unique_layer_index(
        self,
    ) -> tuple[int, ...]:
        return (
            self
            .sharing_plan
            .depth_to_unique_layer_index
        )

    @property
    def diagnostics_enabled(self) -> bool:
        return self.diagnostics is not None

    @property
    def parameter_count(self) -> int:
        return sum(
            int(
                parameter.numel()
            )
            for parameter
            in self.parameters()
        )

    @property
    def trainable_parameter_count(
        self,
    ) -> int:
        return sum(
            int(
                parameter.numel()
            )
            for parameter
            in self.parameters()
            if parameter.requires_grad
        )

    @property
    def buffer_count(self) -> int:
        return sum(
            int(
                buffer.numel()
            )
            for buffer
            in self.buffers()
        )

    @property
    def layers_by_depth(
        self,
    ) -> tuple[
        FunctionalMessagePassingLayer,
        ...,
    ]:
        if (
            self.sharing_policy
            == STACK_SHARING_INDEPENDENT
        ):
            return tuple(
                self.layers
            )

        return tuple(
            self.shared_layer
            for _ in range(
                self.num_layers
            )
        )

    @property
    def unique_layers(
        self,
    ) -> tuple[
        FunctionalMessagePassingLayer,
        ...,
    ]:
        if (
            self.sharing_policy
            == STACK_SHARING_INDEPENDENT
        ):
            return tuple(
                self.layers
            )

        return (
            self.shared_layer,
        )

    def layer_for_depth(
        self,
        depth: int,
    ) -> FunctionalMessagePassingLayer:
        _require_nonnegative_int(
            "depth",
            depth,
        )

        if depth >= self.num_layers:
            raise IndexError(
                "depth lies outside the configured stack."
            )

        if (
            self.sharing_policy
            == STACK_SHARING_INDEPENDENT
        ):
            return self.layers[
                depth
            ]

        return self.shared_layer

    # ------------------------------------------------------------------
    # Architecture and parameter provenance
    # ------------------------------------------------------------------

    def numerical_architecture_dict(
        self,
    ) -> dict[str, Any]:
        return {
            "schema_version": (
                FUNCTIONAL_MESSAGE_PASSING_STACK_MODULE_SCHEMA_VERSION
            ),
            "scientific_interpretation": (
                FUNCTIONAL_MESSAGE_PASSING_STACK_SCIENTIFIC_INTERPRETATION
            ),
            "operation_order": list(
                FUNCTIONAL_MESSAGE_PASSING_STACK_OPERATION_ORDER
            ),
            "sharing_plan": (
                self
                .sharing_plan
                .numerical_architecture_dict()
            ),
            "num_layers": (
                self.num_layers
            ),
            "num_unique_layers": (
                self.num_unique_layers
            ),
            "depth_to_unique_layer_index": list(
                self.depth_to_unique_layer_index
            ),
            "hidden_dim": (
                self.hidden_dim
            ),
            "num_relations": (
                self.num_relations
            ),
            "relation_names": list(
                self.relation_names
            ),
            "stable_relation_ids": list(
                self.stable_relation_ids
            ),
            "compiled_relation_registry_fingerprint": (
                self
                .compiled_relation_registry_fingerprint
            ),
            "state_rebinding": {
                "clones_tensor": (
                    FUNCTIONAL_MESSAGE_PASSING_STACK_STATE_REBINDING_CLONES_TENSOR
                ),
                "detaches_tensor": (
                    FUNCTIONAL_MESSAGE_PASSING_STACK_STATE_REBINDING_DETACHES_TENSOR
                ),
                "casts_tensor": (
                    FUNCTIONAL_MESSAGE_PASSING_STACK_STATE_REBINDING_CASTS_TENSOR
                ),
                "moves_tensor": (
                    FUNCTIONAL_MESSAGE_PASSING_STACK_STATE_REBINDING_MOVES_TENSOR
                ),
            },
            "regularization_reduction": (
                FUNCTIONAL_MESSAGE_PASSING_STACK_REGULARIZATION_REDUCTION
            ),
            "audit_retention": (
                FUNCTIONAL_MESSAGE_PASSING_STACK_AUDIT_RETENTION
            ),
            "output_schema": (
                FUNCTIONAL_MESSAGE_PASSING_STACK_OUTPUT_SCHEMA
            ),
            "stacking_owned_here": (
                FUNCTIONAL_MESSAGE_PASSING_STACK_STACKING_OWNED_HERE
            ),
            "layer_math_owned_here": (
                FUNCTIONAL_MESSAGE_PASSING_STACK_LAYER_MATH_OWNED_HERE
            ),
            "prediction_owned_here": (
                FUNCTIONAL_MESSAGE_PASSING_STACK_PREDICTION_OWNED_HERE
            ),
            "parameter_count": (
                self.parameter_count
            ),
            "trainable_parameter_count": (
                self
                .trainable_parameter_count
            ),
            "buffer_count": (
                self.buffer_count
            ),
            "claims_causal_importance": False,
            "claims_explanation_faithfulness": False,
        }

    def architecture_dict(
        self,
    ) -> dict[str, Any]:
        return (
            self
            .numerical_architecture_dict()
        )

    def architecture_fingerprint(
        self,
    ) -> str:
        return _fingerprint(
            self
            .numerical_architecture_dict()
        )

    def parameter_fingerprint(
        self,
    ) -> str:
        return _fingerprint(
            {
                "schema_version": (
                    FUNCTIONAL_MESSAGE_PASSING_STACK_MODULE_SCHEMA_VERSION
                ),
                "sharing_policy": (
                    self.sharing_policy
                ),
                "sharing_plan_parameter_fingerprint": (
                    self
                    .sharing_plan
                    .parameter_fingerprint()
                ),
                "ordered_state_dict_keys": list(
                    self.state_dict()
                ),
                "parameter_count": (
                    self.parameter_count
                ),
                "trainable_parameter_count": (
                    self
                    .trainable_parameter_count
                ),
            }
        )

    def diagnostics_architecture_dict(
        self,
    ) -> dict[str, Any] | None:
        if self.diagnostics is None:
            return None

        return (
            self.diagnostics
            .architecture_dict()
        )

    def runtime_dict(
        self,
    ) -> dict[str, Any]:
        return {
            "numerical_architecture": (
                self
                .numerical_architecture_dict()
            ),
            "architecture_fingerprint": (
                self
                .architecture_fingerprint()
            ),
            "parameter_fingerprint": (
                self
                .parameter_fingerprint()
            ),
            "training": (
                self.training
            ),
            "default_trace_policy": (
                self
                .default_trace_policy
                .execution_contract_dict()
            ),
            "diagnostics_enabled": (
                self.diagnostics_enabled
            ),
            "diagnostics_architecture": (
                self
                .diagnostics_architecture_dict()
            ),
        }


    def assert_finite_parameters(
        self,
    ) -> None:
        for index, layer in enumerate(
            self.unique_layers
        ):
            _assert_finite_module(
                layer,
                name=(
                    f"unique_layers[{index}]"
                ),
            )



    # ------------------------------------------------------------------
    # Static and runtime validation
    # ------------------------------------------------------------------

    def _assert_static_contract(
        self,
    ) -> None:
        validate_stack_layer_sharing_plan(
            self.sharing_plan
        )

        if tuple(
            id(layer)
            for layer
            in self.layers_by_depth
        ) != tuple(
            id(layer)
            for layer
            in self
            .sharing_plan
            .layers_by_depth
        ):
            raise ValueError(
                "Registered stack modules differ from the sharing plan."
            )

        if (
            self.sharing_policy
            == STACK_SHARING_INDEPENDENT
        ):
            if not hasattr(
                self,
                "layers",
            ):
                raise RuntimeError(
                    "Independent stack must register an nn.ModuleList "
                    "named 'layers'."
                )

            if hasattr(
                self,
                "shared_layer",
            ):
                raise RuntimeError(
                    "Independent stack must not register shared_layer."
                )
        elif (
            self.sharing_policy
            == STACK_SHARING_FULLY_SHARED
        ):
            if not hasattr(
                self,
                "shared_layer",
            ):
                raise RuntimeError(
                    "Fully shared stack must register shared_layer."
                )

            if hasattr(
                self,
                "layers",
            ):
                raise RuntimeError(
                    "Fully shared stack must not register an independent "
                    "ModuleList."
                )
        else:
            raise RuntimeError(
                "Unreachable stack-sharing contract branch."
            )

        self.assert_finite_parameters()

    def _validate_source_inputs(
        self,
        source_inputs: FunctionalMessagePassingInputs,
    ) -> None:
        _require_source_inputs(
            source_inputs
        )

        if source_inputs.hidden_dim != (
            self.hidden_dim
        ):
            raise ValueError(
                "source_inputs hidden width differs from the stack."
            )

        if source_inputs.relation_names != (
            self.relation_names
        ):
            raise ValueError(
                "source_inputs relation ordering differs from the stack."
            )

        if source_inputs.stable_relation_ids != (
            self.stable_relation_ids
        ):
            raise ValueError(
                "source_inputs stable relation IDs differ from the stack."
            )

        if source_inputs.num_relations != (
            self.num_relations
        ):
            raise ValueError(
                "source_inputs relation count differs from the stack."
            )

        input_registry_fingerprint = (
            source_inputs
            .compiled_relation_registry
            .fingerprint()
        )

        if input_registry_fingerprint != (
            self
            .compiled_relation_registry_fingerprint
        ):
            raise ValueError(
                "source_inputs compiled relation registry differs from the "
                "stack."
            )

        for depth, layer in enumerate(
            self.layers_by_depth
        ):
            if layer.hidden_dim != (
                source_inputs.hidden_dim
            ):
                raise ValueError(
                    f"Layer {depth} hidden width differs from source inputs."
                )

            if layer.relation_names != (
                source_inputs.relation_names
            ):
                raise ValueError(
                    f"Layer {depth} relation ordering differs from source "
                    "inputs."
                )

            if layer.stable_relation_ids != (
                source_inputs.stable_relation_ids
            ):
                raise ValueError(
                    f"Layer {depth} stable relation IDs differ from source "
                    "inputs."
                )

    def _validate_stack_inputs(
        self,
        stack_inputs: FunctionalMessagePassingStackInputs,
    ) -> None:
        if not isinstance(
            stack_inputs,
            FunctionalMessagePassingStackInputs,
        ):
            raise TypeError(
                "stack_inputs must be a "
                "FunctionalMessagePassingStackInputs."
            )

        self._validate_source_inputs(
            stack_inputs.source_inputs
        )

        if stack_inputs.num_layers != (
            self.num_layers
        ):
            raise ValueError(
                "stack_inputs.num_layers differs from the stack."
            )

        if stack_inputs.sharing_policy != (
            self.sharing_policy
        ):
            raise ValueError(
                "stack_inputs.sharing_policy differs from the stack."
            )

        if stack_inputs.training is not (
            self.training
        ):
            raise ValueError(
                "stack_inputs.training must match the stack runtime mode."
            )

        for depth, layer in enumerate(
            self.layers_by_depth
        ):
            if layer.training is not (
                self.training
            ):
                raise ValueError(
                    f"Layer {depth} train/eval mode differs from the stack."
                )

        self.sharing_plan.validate_current_ownership(
            require_uniform_training_mode=True,
        )
        self.assert_finite_parameters()

    # ------------------------------------------------------------------
    # Trace and runtime-input construction
    # ------------------------------------------------------------------

    def resolve_trace_policy(
        self,
        *,
        trace_policy: StackTracePolicy | None = None,
        retention_policy: (
            StackRetentionPolicy
            | str
            | None
        ) = None,
        layer_trace_policy: (
            LayerTracePolicy
            | str
            | None
        ) = None,
        audit_mode: bool | None = None,
    ) -> StackTracePolicy:
        if trace_policy is not None:
            return resolve_stack_trace_policy(
                trace_policy,
                retention_policy=(
                    retention_policy
                ),
                layer_trace_policy=(
                    layer_trace_policy
                ),
                audit_mode=(
                    audit_mode
                ),
            )

        return resolve_stack_trace_policy(
            None,
            retention_policy=(
                self.default_trace_policy.retention_policy
                if retention_policy is None
                else retention_policy
            ),
            layer_trace_policy=(
                self.default_trace_policy.layer_trace_policy
                if layer_trace_policy is None
                else layer_trace_policy
            ),
            audit_mode=(
                self.default_trace_policy.audit_mode
                if audit_mode is None
                else audit_mode
            ),
        )


    def build_stack_inputs(
        self,
        source_inputs: FunctionalMessagePassingInputs,
        *,
        trace_policy: StackTracePolicy | None = None,
        retention_policy: (
            StackRetentionPolicy
            | str
            | None
        ) = None,
        layer_trace_policy: (
            LayerTracePolicy
            | str
            | None
        ) = None,
        audit_mode: bool | None = None,
        source_model_fingerprint: str | None = None,
    ) -> FunctionalMessagePassingStackInputs:
        self._validate_source_inputs(
            source_inputs
        )
        _require_optional_nonempty_string(
            "source_model_fingerprint",
            source_model_fingerprint,
        )

        resolved = self.resolve_trace_policy(
            trace_policy=trace_policy,
            retention_policy=(
                retention_policy
            ),
            layer_trace_policy=(
                layer_trace_policy
            ),
            audit_mode=(
                audit_mode
            ),
        )

        stack_inputs = (
            FunctionalMessagePassingStackInputs(
                source_inputs=(
                    source_inputs
                ),
                num_layers=(
                    self.num_layers
                ),
                sharing_policy=(
                    self.sharing_policy
                ),
                retention_policy=(
                    resolved
                    .retention_name
                ),
                layer_trace_policy=(
                    resolved
                    .layer_trace_policy
                ),
                training=(
                    self.training
                ),
                audit_mode=(
                    resolved.audit_mode
                ),
                source_model_fingerprint=(
                    source_model_fingerprint
                ),
            )
        )

        self._validate_stack_inputs(
            stack_inputs
        )
        return stack_inputs

    def _source_stack_fingerprint(
        self,
        stack_inputs: FunctionalMessagePassingStackInputs,
    ) -> str:
        return _fingerprint(
            {
                "stack_architecture_fingerprint": (
                    self
                    .architecture_fingerprint()
                ),
                "stack_parameter_fingerprint": (
                    self
                    .parameter_fingerprint()
                ),
                "stack_inputs_lineage_fingerprint": (
                    stack_inputs
                    .lineage_fingerprint()
                ),
                "execution_contract_fingerprint": (
                    stack_inputs
                    .execution_contract_fingerprint()
                ),
            }
        )

    # ------------------------------------------------------------------
    # Complete stack execution
    # ------------------------------------------------------------------

    def run_from_stack_inputs(
        self,
        stack_inputs: FunctionalMessagePassingStackInputs,
    ) -> FunctionalMessagePassingStackRun:
        """
        Execute the complete stack from a preconstructed runtime contract.
        """

        self._validate_stack_inputs(
            stack_inputs
        )

        original_inputs = (
            stack_inputs.source_inputs
        )
        current_inputs = (
            original_inputs
        )
        source_stack_fingerprint = (
            self
            ._source_stack_fingerprint(
                stack_inputs
            )
        )

        depth_records: list[
            FunctionalMessagePassingStackDepthRecord
        ] = []
        audit_runs: list[
            FunctionalMessagePassingLayerRun
        ] = []

        final_node_state: (
            torch.Tensor
            | None
        ) = None

        retention_policy = (
            StackRetentionPolicy(
                name=(
                    stack_inputs
                    .retention_policy
                )
            )
        )

        for depth in range(
            self.num_layers
        ):
            layer = self.layer_for_depth(
                depth
            )

            if layer.training is not (
                stack_inputs.training
            ):
                raise ValueError(
                    f"Layer {depth} training mode differs from stack_inputs."
                )

            layer_run = layer.run_complete(
                current_inputs,
                layer_index=depth,
                trace_policy=(
                    stack_inputs
                    .layer_trace_policy
                ),
                source_stack_fingerprint=(
                    source_stack_fingerprint
                ),
            )
            validate_functional_message_passing_layer_run(
                layer_run
            )

            if layer_run.layer_index != (
                depth
            ):
                raise ValueError(
                    "Layer run emitted the wrong runtime layer index."
                )

            if layer_run.source_inputs is not (
                current_inputs
            ):
                raise ValueError(
                    "Layer run did not preserve exact current source inputs."
                )

            if (
                layer_run
                .internal_output
                .layer_architecture_fingerprint
                != _resolve_nonempty_fingerprint(
                    layer,
                    "architecture_fingerprint",
                    owner_name=(
                        f"layer[{depth}]"
                    ),
                )
            ):
                raise ValueError(
                    "Layer-run architecture fingerprint differs from the "
                    "executed layer."
                )

            if (
                layer_run
                .internal_output
                .layer_parameter_fingerprint
                != _resolve_nonempty_fingerprint(
                    layer,
                    "parameter_fingerprint",
                    owner_name=(
                        f"layer[{depth}]"
                    ),
                )
            ):
                raise ValueError(
                    "Layer-run parameter fingerprint differs from the "
                    "executed layer."
                )

            retain_output = (
                retention_policy
                .should_retain(
                    layer_index=depth,
                    num_layers=(
                        self.num_layers
                    ),
                )
            )

            depth_record = (
                FunctionalMessagePassingStackDepthRecord
                .from_layer_run(
                    layer_run,
                    retain_output=(
                        retain_output
                    ),
                )
            )
            depth_records.append(
                depth_record
            )

            if stack_inputs.audit_mode:
                audit_runs.append(
                    layer_run
                )

            final_node_state = (
                layer_run.updated_node_state
            )

            if depth + 1 < (
                self.num_layers
            ):
                current_inputs = (
                    derive_next_functional_message_passing_inputs(
                        original_stack_inputs=(
                            original_inputs
                        ),
                        previous_layer_run=(
                            layer_run
                        ),
                        source_stack_fingerprint=(
                            source_stack_fingerprint
                        ),
                        next_layer_index=(
                            depth + 1
                        ),
                    )
                )

        if final_node_state is None:
            raise RuntimeError(
                "Stack execution completed without a final node state."
            )

        audit_trace = (
            FunctionalMessagePassingStackTrace(
                stack_inputs=(
                    stack_inputs
                ),
                layer_runs=tuple(
                    audit_runs
                ),
            )
            if stack_inputs.audit_mode
            else None
        )

        stack_architecture_fingerprint = (
            self.architecture_fingerprint()
        )
        stack_parameter_fingerprint = (
            self.parameter_fingerprint()
        )
        execution_contract_fingerprint = (
            stack_inputs
            .execution_contract_fingerprint()
        )

        lineage_fingerprint = (
            self._lineage_fingerprint(
                stack_inputs=(
                    stack_inputs
                ),
                depth_records=tuple(
                    depth_records
                ),
                final_node_state=(
                    final_node_state
                ),
                stack_architecture_fingerprint=(
                    stack_architecture_fingerprint
                ),
                stack_parameter_fingerprint=(
                    stack_parameter_fingerprint
                ),
                execution_contract_fingerprint=(
                    execution_contract_fingerprint
                ),
                audit_trace=(
                    audit_trace
                ),
            )
        )

        internal_output = (
            FunctionalMessagePassingStackComputationOutput(
                stack_inputs=(
                    stack_inputs
                ),
                final_node_state=(
                    final_node_state
                ),
                depth_records=tuple(
                    depth_records
                ),
                stack_architecture_fingerprint=(
                    stack_architecture_fingerprint
                ),
                stack_parameter_fingerprint=(
                    stack_parameter_fingerprint
                ),
                execution_contract_fingerprint=(
                    execution_contract_fingerprint
                ),
                lineage_fingerprint=(
                    lineage_fingerprint
                ),
                audit_trace=(
                    audit_trace
                ),
            )
        )
        public_output = (
            assemble_functional_message_passing_stack_output(
                internal_output
            )
        )
        run = (
            FunctionalMessagePassingStackRun(
                stack_inputs=(
                    stack_inputs
                ),
                internal_output=(
                    internal_output
                ),
                public_output=(
                    public_output
                ),
            )
        )

        self._validate_owned_run(
            run
        )
        return run

    def run_complete(
        self,
        source_inputs: FunctionalMessagePassingInputs,
        *,
        trace_policy: StackTracePolicy | None = None,
        retention_policy: (
            StackRetentionPolicy
            | str
            | None
        ) = None,
        layer_trace_policy: (
            LayerTracePolicy
            | str
            | None
        ) = None,
        audit_mode: bool | None = None,
        source_model_fingerprint: str | None = None,
    ) -> FunctionalMessagePassingStackRun:
        stack_inputs = (
            self.build_stack_inputs(
                source_inputs,
                trace_policy=trace_policy,
                retention_policy=(
                    retention_policy
                ),
                layer_trace_policy=(
                    layer_trace_policy
                ),
                audit_mode=(
                    audit_mode
                ),
                source_model_fingerprint=(
                    source_model_fingerprint
                ),
            )
        )

        return self.run_from_stack_inputs(
            stack_inputs
        )

    def _lineage_fingerprint(
        self,
        *,
        stack_inputs: FunctionalMessagePassingStackInputs,
        depth_records: tuple[
            FunctionalMessagePassingStackDepthRecord,
            ...,
        ],
        final_node_state: torch.Tensor,
        stack_architecture_fingerprint: str,
        stack_parameter_fingerprint: str,
        execution_contract_fingerprint: str,
        audit_trace: (
            FunctionalMessagePassingStackTrace
            | None
        ),
    ) -> str:
        _require_nonempty_string(
            "stack_architecture_fingerprint",
            stack_architecture_fingerprint,
        )
        _require_nonempty_string(
            "stack_parameter_fingerprint",
            stack_parameter_fingerprint,
        )
        _require_nonempty_string(
            "execution_contract_fingerprint",
            execution_contract_fingerprint,
        )

        return _fingerprint(
            {
                "schema_version": (
                    FUNCTIONAL_MESSAGE_PASSING_STACK_MODULE_SCHEMA_VERSION
                ),
                "stack_inputs_lineage_fingerprint": (
                    stack_inputs
                    .lineage_fingerprint()
                ),
                "stack_architecture_fingerprint": (
                    stack_architecture_fingerprint
                ),
                "stack_parameter_fingerprint": (
                    stack_parameter_fingerprint
                ),
                "execution_contract_fingerprint": (
                    execution_contract_fingerprint
                ),
                "depth_record_lineage_fingerprints": [
                    record
                    .lineage_fingerprint()
                    for record
                    in depth_records
                ],
                "final_node_state_value_fingerprint": (
                    _tensor_value_fingerprint(
                        final_node_state,
                        name=(
                            "final_node_state"
                        ),
                    )
                ),
                "audit_trace_lineage_fingerprint": (
                    audit_trace
                    .lineage_fingerprint()
                    if audit_trace
                    is not None
                    else None
                ),
            }
        )

    def _validate_owned_run(
        self,
        run: FunctionalMessagePassingStackRun,
    ) -> None:
        validate_functional_message_passing_stack_run(
            run
        )

        if run.stack_inputs.num_layers != (
            self.num_layers
        ):
            raise ValueError(
                "Owned run depth differs from the stack."
            )

        if run.stack_inputs.sharing_policy != (
            self.sharing_policy
        ):
            raise ValueError(
                "Owned run sharing policy differs from the stack."
            )

        if run.stack_inputs.training is not (
            self.training
        ):
            raise ValueError(
                "Owned run training mode differs from the stack."
            )

        if (
            run
            .internal_output
            .stack_architecture_fingerprint
            != self.architecture_fingerprint()
        ):
            raise ValueError(
                "Owned run architecture fingerprint differs from the stack."
            )

        if (
            run
            .internal_output
            .stack_parameter_fingerprint
            != self.parameter_fingerprint()
        ):
            raise ValueError(
                "Owned run parameter fingerprint differs from the stack."
            )

        expected_layer_architectures = tuple(
            _resolve_nonempty_fingerprint(
                layer,
                "architecture_fingerprint",
                owner_name=(
                    f"layer[{depth}]"
                ),
            )
            for depth, layer in enumerate(
                self.layers_by_depth
            )
        )
        expected_layer_parameters = tuple(
            _resolve_nonempty_fingerprint(
                layer,
                "parameter_fingerprint",
                owner_name=(
                    f"layer[{depth}]"
                ),
            )
            for depth, layer in enumerate(
                self.layers_by_depth
            )
        )

        if (
            run
            .internal_output
            .layer_architecture_fingerprints
            != expected_layer_architectures
        ):
            raise ValueError(
                "Owned run layer architecture fingerprints differ from "
                "the registered stack layers."
            )

        if (
            run
            .internal_output
            .layer_parameter_fingerprints
            != expected_layer_parameters
        ):
            raise ValueError(
                "Owned run layer parameter fingerprints differ from the "
                "registered stack layers."
            )

        original_inputs = (
            run.source_inputs
        )
        previous_run: (
            FunctionalMessagePassingLayerRun
            | None
        ) = None

        if run.trace is not None:
            layer_runs = (
                run.trace.layer_runs
            )
        else:
            layer_runs = ()

        for depth, layer_run in enumerate(
            layer_runs
        ):
            if depth == 0:
                if layer_run.source_inputs is not (
                    original_inputs
                ):
                    raise ValueError(
                        "Audit layer 0 must consume exact original stack "
                        "inputs."
                    )
            else:
                validate_rebound_functional_message_passing_inputs(
                    original_stack_inputs=(
                        original_inputs
                    ),
                    rebound_inputs=(
                        layer_run.source_inputs
                    ),
                    previous_layer_run=(
                        previous_run
                    ),
                    next_layer_index=depth,
                )

            previous_run = layer_run

    # ------------------------------------------------------------------
    # Public forward
    # ------------------------------------------------------------------

    def forward(
        self,
        source_inputs: FunctionalMessagePassingInputs,
        *,
        trace_policy: StackTracePolicy | None = None,
        retention_policy: (
            StackRetentionPolicy
            | str
            | None
        ) = None,
        layer_trace_policy: (
            LayerTracePolicy
            | str
            | None
        ) = None,
        audit_mode: bool | None = None,
        source_model_fingerprint: str | None = None,
    ) -> FunctionalMessagePassingStackOutput:
        """
        Return only the public output for ordinary model execution.
        """

        return self.run_complete(
            source_inputs,
            trace_policy=trace_policy,
            retention_policy=(
                retention_policy
            ),
            layer_trace_policy=(
                layer_trace_policy
            ),
            audit_mode=(
                audit_mode
            ),
            source_model_fingerprint=(
                source_model_fingerprint
            ),
        ).public_output

    # ------------------------------------------------------------------
    # Explicit diagnostics
    # ------------------------------------------------------------------

    def diagnostic_report(
        self,
        *,
        run: FunctionalMessagePassingStackRun,
    ) -> dict[str, Any]:
        if not isinstance(
            run,
            FunctionalMessagePassingStackRun,
        ):
            raise TypeError(
                "run must be a FunctionalMessagePassingStackRun."
            )

        if self.diagnostics is None:
            raise RuntimeError(
                "Diagnostics are not configured for this "
                "FunctionalMessagePassingStack."
            )

        self._validate_owned_run(
            run
        )

        return dict(
            self.diagnostics
            .public_report(
                run
            )
        )

    def forward_with_diagnostics(
        self,
        source_inputs: FunctionalMessagePassingInputs,
        *,
        trace_policy: StackTracePolicy | None = None,
        retention_policy: (
            StackRetentionPolicy
            | str
            | None
        ) = None,
        layer_trace_policy: (
            LayerTracePolicy
            | str
            | None
        ) = None,
        audit_mode: bool | None = None,
        source_model_fingerprint: str | None = None,
    ) -> FunctionalMessagePassingStackRunWithDiagnostics:
        run = self.run_complete(
            source_inputs,
            trace_policy=trace_policy,
            retention_policy=(
                retention_policy
            ),
            layer_trace_policy=(
                layer_trace_policy
            ),
            audit_mode=(
                audit_mode
            ),
            source_model_fingerprint=(
                source_model_fingerprint
            ),
        )

        report = self.diagnostic_report(
            run=run
        )

        return (
            FunctionalMessagePassingStackRunWithDiagnostics(
                run=run,
                diagnostic_report=report,
            )
        )

    def extra_repr(self) -> str:
        return (
            f"num_layers={self.num_layers}, "
            f"num_unique_layers={self.num_unique_layers}, "
            f"sharing_policy={self.sharing_policy!r}, "
            f"hidden_dim={self.hidden_dim}, "
            f"num_relations={self.num_relations}, "
            f"default_retention_policy="
            f"{self.default_trace_policy.retention_name!r}, "
            f"default_layer_trace_mode="
            f"{self.default_trace_policy.layer_trace_mode!r}, "
            f"default_audit_mode="
            f"{self.default_trace_policy.audit_mode}, "
            f"diagnostics_enabled={self.diagnostics_enabled}, "
            "prediction_owned_here=False"
        )


# =============================================================================
# Construction helpers
# =============================================================================


def build_functional_message_passing_stack(
    layers: (
        FunctionalMessagePassingLayer
        | Sequence[
            FunctionalMessagePassingLayer
        ]
    ),
    *,
    num_layers: int,
    sharing_policy: (
        StackSharingPolicy
        | str
    ) = STACK_SHARING_INDEPENDENT,
    default_trace_policy: (
        StackTracePolicy
        | None
    ) = None,
    diagnostics: (
        FunctionalMessagePassingStackDiagnostics
        | None
    ) = None,
    require_uniform_training_mode: bool = True,
) -> FunctionalMessagePassingStack:
    return (
        FunctionalMessagePassingStack
        .from_layers(
            layers,
            num_layers=num_layers,
            sharing_policy=(
                sharing_policy
            ),
            default_trace_policy=(
                default_trace_policy
            ),
            diagnostics=diagnostics,
            require_uniform_training_mode=(
                require_uniform_training_mode
            ),
        )
    )


def build_functional_message_passing_stack_from_factory(
    layer_factory: StackLayerFactory,
    *,
    num_layers: int,
    sharing_policy: (
        StackSharingPolicy
        | str
    ) = STACK_SHARING_INDEPENDENT,
    default_trace_policy: (
        StackTracePolicy
        | None
    ) = None,
    diagnostics: (
        FunctionalMessagePassingStackDiagnostics
        | None
    ) = None,
    require_uniform_training_mode: bool = True,
) -> FunctionalMessagePassingStack:
    return (
        FunctionalMessagePassingStack
        .from_factory(
            layer_factory,
            num_layers=num_layers,
            sharing_policy=(
                sharing_policy
            ),
            default_trace_policy=(
                default_trace_policy
            ),
            diagnostics=diagnostics,
            require_uniform_training_mode=(
                require_uniform_training_mode
            ),
        )
    )


# =============================================================================
# Functional execution helpers
# =============================================================================


def run_functional_message_passing_stack(
    stack: FunctionalMessagePassingStack,
    source_inputs: FunctionalMessagePassingInputs,
    *,
    trace_policy: StackTracePolicy | None = None,
    retention_policy: (
        StackRetentionPolicy
        | str
        | None
    ) = None,
    layer_trace_policy: (
        LayerTracePolicy
        | str
        | None
    ) = None,
    audit_mode: bool | None = None,
    source_model_fingerprint: str | None = None,
) -> FunctionalMessagePassingStackOutput:
    if not isinstance(
        stack,
        FunctionalMessagePassingStack,
    ):
        raise TypeError(
            "stack must be a FunctionalMessagePassingStack."
        )

    return stack(
        source_inputs,
        trace_policy=trace_policy,
        retention_policy=(
            retention_policy
        ),
        layer_trace_policy=(
            layer_trace_policy
        ),
        audit_mode=(
            audit_mode
        ),
        source_model_fingerprint=(
            source_model_fingerprint
        ),
    )


def run_functional_message_passing_stack_complete(
    stack: FunctionalMessagePassingStack,
    source_inputs: FunctionalMessagePassingInputs,
    *,
    trace_policy: StackTracePolicy | None = None,
    retention_policy: (
        StackRetentionPolicy
        | str
        | None
    ) = None,
    layer_trace_policy: (
        LayerTracePolicy
        | str
        | None
    ) = None,
    audit_mode: bool | None = None,
    source_model_fingerprint: str | None = None,
) -> FunctionalMessagePassingStackRun:
    if not isinstance(
        stack,
        FunctionalMessagePassingStack,
    ):
        raise TypeError(
            "stack must be a FunctionalMessagePassingStack."
        )

    return stack.run_complete(
        source_inputs,
        trace_policy=trace_policy,
        retention_policy=(
            retention_policy
        ),
        layer_trace_policy=(
            layer_trace_policy
        ),
        audit_mode=(
            audit_mode
        ),
        source_model_fingerprint=(
            source_model_fingerprint
        ),
    )


# =============================================================================
# Compact aliases
# =============================================================================


HazardConditionedFunctionalMessagePassingStack = (
    FunctionalMessagePassingStack
)
FunctionalStack = FunctionalMessagePassingStack
MessagePassingStack = FunctionalMessagePassingStack

build_stack = (
    build_functional_message_passing_stack
)
build_stack_from_factory = (
    build_functional_message_passing_stack_from_factory
)
run_stack = (
    run_functional_message_passing_stack
)
run_stack_complete = (
    run_functional_message_passing_stack_complete
)

derive_next_layer_inputs = (
    derive_next_functional_message_passing_inputs
)
validate_rebound_inputs = (
    validate_rebound_functional_message_passing_inputs
)


__all__ = (
    # Public identity.
    "FUNCTIONAL_MESSAGE_PASSING_STACK_MODULE_SCHEMA_VERSION",
    "FUNCTIONAL_MESSAGE_PASSING_STACK_STACKING_OWNED_HERE",
    "FUNCTIONAL_MESSAGE_PASSING_STACK_LAYER_MATH_OWNED_HERE",
    "FUNCTIONAL_MESSAGE_PASSING_STACK_PREDICTION_OWNED_HERE",
    "FUNCTIONAL_MESSAGE_PASSING_STACK_STATE_REBINDING_CLONES_TENSOR",
    "FUNCTIONAL_MESSAGE_PASSING_STACK_STATE_REBINDING_DETACHES_TENSOR",
    "FUNCTIONAL_MESSAGE_PASSING_STACK_STATE_REBINDING_CASTS_TENSOR",
    "FUNCTIONAL_MESSAGE_PASSING_STACK_STATE_REBINDING_MOVES_TENSOR",
    "FUNCTIONAL_MESSAGE_PASSING_STACK_REGULARIZATION_REDUCTION",
    "FUNCTIONAL_MESSAGE_PASSING_STACK_AUDIT_RETENTION",
    "FUNCTIONAL_MESSAGE_PASSING_STACK_INDEPENDENT_REGISTRATION_PREFIX",
    "FUNCTIONAL_MESSAGE_PASSING_STACK_FULLY_SHARED_REGISTRATION_PREFIX",
    "FUNCTIONAL_MESSAGE_PASSING_STACK_OUTPUT_SCHEMA",
    # State rebinding.
    "derive_next_functional_message_passing_inputs",
    "derive_next_layer_inputs",
    "validate_rebound_functional_message_passing_inputs",
    "validate_rebound_inputs",
    # Stack module.
    "FunctionalMessagePassingStack",
    "HazardConditionedFunctionalMessagePassingStack",
    "FunctionalStack",
    "MessagePassingStack",
    # Construction.
    "build_functional_message_passing_stack",
    "build_stack",
    "build_functional_message_passing_stack_from_factory",
    "build_stack_from_factory",
    # Execution.
    "run_functional_message_passing_stack",
    "run_stack",
    "run_functional_message_passing_stack_complete",
    "run_stack_complete",
)
