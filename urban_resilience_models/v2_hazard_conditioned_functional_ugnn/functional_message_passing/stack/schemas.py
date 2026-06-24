"""
Immutable contracts for multi-layer functional message passing.

Target path:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            functional_message_passing/
                stack/
                    schemas.py

This module freezes the execution, retention, lineage, regularization, audit,
and public-output contracts for a depth-wise stack of already-complete
``FunctionalMessagePassingLayer`` modules.

The stack owns repeated application of one-layer state transitions. It does
not reimplement relation transforms, structural edge normalization, relation
gating, edge attention, message construction, target-node aggregation,
residual updates, or layer normalization.

Bounded V2.0 stack contract
---------------------------
A stack contains a strictly positive number of layers and preserves one fixed
node-state width:

    initial node state      [N, H]
    every layer output      [N, H]
    final node state        [N, H]

Only the current node representation evolves across depth. The following
structural context remains unchanged:

- graph object and edge ordering;
- node ordering and graph membership;
- compiled relation registry;
- optional relation-family alignment;
- optional hazard query;
- optional compiled hazard-relation priors;
- dtype, device, node count, and hidden width.

Sharing policy
--------------
The bounded implementation recognizes:

``independent``
    Every depth owns a distinct layer module and distinct parameter set.

``fully_shared``
    One exact layer module and one parameter set are reused at every depth.

Partial sharing is not silently approximated.

Retention policy
----------------
Stack output retention is distinct from one-layer trace detail.

``none``
    Retain no public layer outputs.

``final_layer``
    Retain only the final public layer output.

``all_layers``
    Retain every public layer output in zero-based depth order.

The existing ``LayerTracePolicy`` independently controls how much internal
detail each retained layer output exposes.

Regularization
--------------
Scalar regularization mappings are preserved once per executed depth. This
module does not sum, average, or deduplicate them. In particular, it does not
decide how parameter-only penalties should be reduced under fully shared
parameters. Loss construction remains a later training responsibility.

Memory discipline
-----------------
Ordinary stack execution need not retain complete layer runs. A
``FunctionalMessagePassingStackTrace`` is an explicit high-cost audit object
that retains every complete layer run and validates exact state propagation.

Scientific limits
-----------------
Depth-wise state changes, retained gates, attention values, regularization
terms, and diagnostics are descriptive model quantities. They do not
automatically establish causal importance, faithful explanation, calibrated
uncertainty, counterfactual effect, or mechanistic identifiability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
from types import MappingProxyType
from typing import Any, Final, Mapping, Sequence

import torch

from ...constants import (
    CANONICAL_STACK_RETENTION_POLICIES,
    CANONICAL_STACK_SHARING_POLICIES,
    STACK_RETENTION_ALL_LAYERS,
    STACK_RETENTION_FINAL_LAYER,
    STACK_RETENTION_NONE,
    STACK_SHARING_FULLY_SHARED,
    STACK_SHARING_INDEPENDENT,
    V2_0_IMPLEMENTED_STACK_RETENTION_POLICIES,
    V2_0_IMPLEMENTED_STACK_SHARING_POLICIES,
)
from ..layer.layer import (
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
    FunctionalMessagePassingLayerOutput,
    FunctionalMessagePassingNodeState,
)


# =============================================================================
# Public identity
# =============================================================================


FUNCTIONAL_MESSAGE_PASSING_STACK_SCHEMA_VERSION: Final[str] = "0.1"
STACK_INPUTS_SCHEMA_VERSION: Final[str] = "0.1"
STACK_DEPTH_RECORD_SCHEMA_VERSION: Final[str] = "0.1"
STACK_TRACE_SCHEMA_VERSION: Final[str] = "0.1"
STACK_COMPUTATION_OUTPUT_SCHEMA_VERSION: Final[str] = "0.1"
STACK_PUBLIC_OUTPUT_SCHEMA_VERSION: Final[str] = "0.1"
STACK_RUN_SCHEMA_VERSION: Final[str] = "0.1"
STACK_RUN_WITH_DIAGNOSTICS_SCHEMA_VERSION: Final[str] = "0.1"

FUNCTIONAL_MESSAGE_PASSING_STACK_OPERATION_ORDER: Final[
    tuple[str, ...]
] = (
    "validate_stack_and_layer_contracts",
    "select_layer_for_depth",
    "execute_complete_layer_transition",
    "preserve_depth_regularization_mapping",
    "retain_public_layer_output_when_requested",
    "derive_next_functional_message_passing_node_state",
    "rebind_next_layer_inputs_without_mutation",
    "repeat_until_configured_depth",
    "assemble_stack_computation_output",
    "assemble_public_stack_output",
    "optionally_construct_explicit_audit_trace",
    "optionally_generate_explicit_tensor_free_diagnostics",
)

FUNCTIONAL_MESSAGE_PASSING_STACK_SCIENTIFIC_INTERPRETATION: Final[str] = (
    "depth_wise_repeated_hazard_conditioned_functional_state_transition"
)

FUNCTIONAL_MESSAGE_PASSING_STACK_OUTPUT_SCHEMA: Final[str] = (
    "FunctionalMessagePassingStackOutput"
)

FUNCTIONAL_MESSAGE_PASSING_STACK_REIMPLEMENTS_LAYER_MATH: Final[bool] = False
FUNCTIONAL_MESSAGE_PASSING_STACK_RETENTION_AFFECTS_NUMERICS: Final[
    bool
] = False
FUNCTIONAL_MESSAGE_PASSING_STACK_LAYER_TRACE_AFFECTS_NUMERICS: Final[
    bool
] = False
FUNCTIONAL_MESSAGE_PASSING_STACK_DIAGNOSTICS_AFFECT_NUMERICS: Final[
    bool
] = False

FUNCTIONAL_MESSAGE_PASSING_STACK_HIDDEN_WIDTH_CHANGES_SUPPORTED: Final[
    bool
] = False
FUNCTIONAL_MESSAGE_PASSING_STACK_ZERO_LAYER_SUPPORTED: Final[bool] = False
FUNCTIONAL_MESSAGE_PASSING_STACK_PARTIAL_SHARING_SUPPORTED: Final[
    bool
] = False


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
        _canonical_json(payload).encode(
            "utf-8"
        )
    ).hexdigest()


def _require_nonempty_string(
    name: str,
    value: str,
) -> None:
    if not isinstance(value, str) or not value.strip():
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
    if isinstance(value, bool) or not isinstance(
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
    if isinstance(value, bool) or not isinstance(
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
    if not isinstance(value, bool):
        raise TypeError(
            f"{name} must be Boolean."
        )


def _require_choice(
    name: str,
    value: str,
    choices: tuple[str, ...],
) -> None:
    _require_nonempty_string(
        name,
        value,
    )

    if value not in choices:
        raise ValueError(
            f"{name} must be one of {choices!r}; "
            f"observed {value!r}."
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


def _require_float_matrix(
    name: str,
    value: torch.Tensor,
    *,
    num_nodes: int,
    hidden_dim: int,
    dtype: torch.dtype,
    device: torch.device,
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
            f"{name} must have rank 2 and shape [N, H]; "
            f"observed {tuple(value.shape)}."
        )

    expected = (
        num_nodes,
        hidden_dim,
    )
    observed = tuple(
        int(size)
        for size in value.shape
    )

    if observed != expected:
        raise ValueError(
            f"{name} must have shape {expected}; observed {observed}."
        )

    if not value.dtype.is_floating_point:
        raise ValueError(
            f"{name} must use a floating-point dtype."
        )

    if value.dtype != dtype:
        raise ValueError(
            f"{name} must use dtype {dtype}; observed {value.dtype}."
        )

    if not _devices_match(
        value.device,
        device,
    ):
        raise ValueError(
            f"{name} must be on device {device}; "
            f"observed {value.device}."
        )

    if not bool(
        torch.isfinite(value).all().item()
    ):
        raise FloatingPointError(
            f"{name} must contain only finite values."
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


def _resolve_lineage_fingerprint(
    value: object,
    *,
    name: str,
) -> str:
    fingerprint = getattr(
        value,
        "lineage_fingerprint",
        None,
    )

    if callable(fingerprint):
        fingerprint = fingerprint()

    _require_nonempty_string(
        name,
        fingerprint,
    )

    return fingerprint


def _resolve_layer_trace_policy(
    value: LayerTracePolicy | str,
) -> LayerTracePolicy:
    if isinstance(
        value,
        LayerTracePolicy,
    ):
        policy = value
    elif isinstance(
        value,
        str,
    ):
        policy = LayerTracePolicy(
            mode=value
        )
    else:
        raise TypeError(
            "layer_trace_policy must be a LayerTracePolicy or string."
        )

    policy.assert_implemented()
    return policy


def _normalize_sharing_policy(
    value: str,
) -> str:
    _require_choice(
        "sharing_policy",
        value,
        CANONICAL_STACK_SHARING_POLICIES,
    )

    if value not in (
        V2_0_IMPLEMENTED_STACK_SHARING_POLICIES
    ):
        raise NotImplementedError(
            f"Stack sharing policy {value!r} is canonical but not "
            "implemented in bounded V2.0."
        )

    return value


def _normalize_retention_policy(
    value: str,
) -> str:
    _require_choice(
        "retention_policy",
        value,
        CANONICAL_STACK_RETENTION_POLICIES,
    )

    if value not in (
        V2_0_IMPLEMENTED_STACK_RETENTION_POLICIES
    ):
        raise NotImplementedError(
            f"Stack retention policy {value!r} is canonical but not "
            "implemented in bounded V2.0."
        )

    return value


def expected_retained_layer_indices(
    *,
    retention_policy: str,
    num_layers: int,
) -> tuple[int, ...]:
    """
    Return the exact zero-based layer indices required by one retention mode.
    """

    policy = _normalize_retention_policy(
        retention_policy
    )
    _require_positive_int(
        "num_layers",
        num_layers,
    )

    if policy == STACK_RETENTION_NONE:
        return ()

    if policy == STACK_RETENTION_FINAL_LAYER:
        return (
            num_layers - 1,
        )

    if policy == STACK_RETENTION_ALL_LAYERS:
        return tuple(
            range(num_layers)
        )

    raise RuntimeError(
        "Unreachable stack-retention branch."
    )


def _immutable_scalar_tensor_mapping(
    name: str,
    values: Mapping[str, torch.Tensor],
    *,
    device: torch.device,
) -> Mapping[str, torch.Tensor]:
    if not isinstance(
        values,
        Mapping,
    ):
        raise TypeError(
            f"{name} must be a mapping."
        )

    copied: dict[
        str,
        torch.Tensor,
    ] = {}

    for key, value in values.items():
        _require_nonempty_string(
            f"{name} key",
            key,
        )

        if not isinstance(
            value,
            torch.Tensor,
        ):
            raise TypeError(
                f"{name}[{key!r}] must be a tensor."
            )

        if not value.dtype.is_floating_point:
            raise ValueError(
                f"{name}[{key!r}] must use a floating-point dtype."
            )

        if value.numel() != 1:
            raise ValueError(
                f"{name}[{key!r}] must contain exactly one scalar value."
            )

        if not _devices_match(
            value.device,
            device,
        ):
            raise ValueError(
                f"{name}[{key!r}] must be on device {device}; "
                f"observed {value.device}."
            )

        if not bool(
            torch.isfinite(value).all().item()
        ):
            raise FloatingPointError(
                f"{name}[{key!r}] must be finite."
            )

        copied[key] = value

    return MappingProxyType(
        copied
    )


def _immutable_layer_regularization_terms(
    values: Sequence[
        Mapping[str, torch.Tensor]
    ],
    *,
    num_layers: int,
    device: torch.device,
) -> tuple[
    Mapping[str, torch.Tensor],
    ...,
]:
    if isinstance(
        values,
        (
            str,
            bytes,
        ),
    ) or not isinstance(
        values,
        Sequence,
    ):
        raise TypeError(
            "layer_regularization_terms must be a sequence of mappings."
        )

    if len(values) != num_layers:
        raise ValueError(
            "layer_regularization_terms must contain exactly one mapping "
            "per executed layer."
        )

    return tuple(
        _immutable_scalar_tensor_mapping(
            f"layer_regularization_terms[{index}]",
            mapping,
            device=device,
        )
        for index, mapping in enumerate(
            values
        )
    )


def _same_scalar_mapping_identity(
    left: Mapping[str, torch.Tensor],
    right: Mapping[str, torch.Tensor],
) -> bool:
    if tuple(left.keys()) != tuple(
        right.keys()
    ):
        return False

    return all(
        left[key] is right[key]
        for key in left
    )


def _assert_tensor_free_value(
    value: Any,
    *,
    path: str,
) -> None:
    if isinstance(
        value,
        torch.Tensor,
    ):
        raise TypeError(
            f"{path} must not contain tensors."
        )

    if isinstance(
        value,
        torch.nn.Module,
    ):
        raise TypeError(
            f"{path} must not contain modules."
        )

    if value is None or isinstance(
        value,
        (
            str,
            bool,
            int,
            float,
        ),
    ):
        return

    if isinstance(
        value,
        Mapping,
    ):
        for key, child in value.items():
            if not isinstance(
                key,
                str,
            ):
                raise TypeError(
                    f"{path} mapping keys must be strings."
                )
            _assert_tensor_free_value(
                child,
                path=f"{path}.{key}",
            )
        return

    if isinstance(
        value,
        (
            tuple,
            list,
        ),
    ):
        for index, child in enumerate(
            value
        ):
            _assert_tensor_free_value(
                child,
                path=f"{path}[{index}]",
            )
        return

    raise TypeError(
        f"{path} contains unsupported value type "
        f"{type(value).__name__!r}."
    )


def _immutable_tensor_free_mapping(
    values: Mapping[str, Any],
) -> Mapping[str, Any]:
    if not isinstance(
        values,
        Mapping,
    ):
        raise TypeError(
            "diagnostic_report must be a mapping."
        )

    _assert_tensor_free_value(
        values,
        path="diagnostic_report",
    )

    return MappingProxyType(
        dict(values)
    )


def _validate_structural_context(
    *,
    original: FunctionalMessagePassingInputs,
    candidate: FunctionalMessagePassingInputs,
    name: str,
) -> None:
    _require_source_inputs(
        original,
        name="original",
    )
    _require_source_inputs(
        candidate,
        name=name,
    )

    identity_fields = (
        (
            "source_graph",
            original.source_graph,
            candidate.source_graph,
        ),
        (
            "compiled_relation_registry",
            original.compiled_relation_registry,
            candidate.compiled_relation_registry,
        ),
        (
            "relation_families",
            original.relation_families,
            candidate.relation_families,
        ),
        (
            "hazard_query",
            original.hazard_query,
            candidate.hazard_query,
        ),
        (
            "compiled_relation_priors",
            original.compiled_relation_priors,
            candidate.compiled_relation_priors,
        ),
    )

    for field_name, expected, observed in (
        identity_fields
    ):
        if observed is not expected:
            raise ValueError(
                f"{name}.{field_name} must preserve exact stack structural "
                "context identity."
            )

    if candidate.num_nodes != original.num_nodes:
        raise ValueError(
            f"{name} must preserve the stack node count."
        )

    if candidate.hidden_dim != original.hidden_dim:
        raise ValueError(
            f"{name} must preserve the stack hidden width."
        )

    if candidate.dtype != original.dtype:
        raise ValueError(
            f"{name} must preserve the stack floating-point dtype."
        )

    if not _devices_match(
        candidate.device,
        original.device,
    ):
        raise ValueError(
            f"{name} must preserve the stack device."
        )

    if candidate.relation_names != (
        original.relation_names
    ):
        raise ValueError(
            f"{name} must preserve exact relation ordering."
        )

    if candidate.stable_relation_ids != (
        original.stable_relation_ids
    ):
        raise ValueError(
            f"{name} must preserve stable relation identities."
        )


def _validate_rebound_layer_source(
    *,
    original: FunctionalMessagePassingInputs,
    candidate: FunctionalMessagePassingInputs,
    layer_index: int,
    previous_run: FunctionalMessagePassingLayerRun | None,
) -> None:
    _validate_structural_context(
        original=original,
        candidate=candidate,
        name=f"layer_source_inputs[{layer_index}]",
    )

    if layer_index == 0:
        if candidate is not original:
            raise ValueError(
                "Layer 0 must consume the exact original stack source "
                "inputs object."
            )
        return

    if previous_run is None:
        raise ValueError(
            "A previous layer run is required to validate a rebound source."
        )

    node_state = candidate.node_state

    if not isinstance(
        node_state,
        FunctionalMessagePassingNodeState,
    ):
        raise TypeError(
            "Every layer after depth 0 must consume a "
            "FunctionalMessagePassingNodeState."
        )

    if node_state.source_kind != (
        FMP_NODE_STATE_SOURCE_LAYER_OUTPUT
    ):
        raise ValueError(
            "Rebound node state must declare source_kind='layer_output'."
        )

    if node_state.source_layer_index != (
        layer_index - 1
    ):
        raise ValueError(
            "Rebound node-state source_layer_index must identify the "
            "immediately preceding layer."
        )

    if node_state.state is not (
        previous_run.updated_node_state
    ):
        raise ValueError(
            "Rebound node state must preserve exact previous "
            "updated_node_state tensor identity."
        )

    if (
        node_state.source_architecture_fingerprint
        != previous_run.public_output.encoder_architecture_fingerprint
    ):
        raise ValueError(
            "Rebound node state carries the wrong source architecture "
            "fingerprint."
        )

    if (
        node_state.source_parameter_fingerprint
        != previous_run.internal_output.layer_parameter_fingerprint
    ):
        raise ValueError(
            "Rebound node state carries the wrong source parameter "
            "fingerprint."
        )

    if (
        node_state.source_lineage_fingerprint
        != previous_run.public_output.lineage_fingerprint
    ):
        raise ValueError(
            "Rebound node state carries the wrong source lineage "
            "fingerprint."
        )


# =============================================================================
# Stack execution inputs
# =============================================================================


@dataclass(slots=True, frozen=True)
class FunctionalMessagePassingStackInputs:
    """
    Immutable numerical and execution contract for one stack invocation.
    """

    source_inputs: FunctionalMessagePassingInputs
    num_layers: int

    sharing_policy: str = (
        STACK_SHARING_INDEPENDENT
    )
    retention_policy: str = (
        STACK_RETENTION_NONE
    )
    layer_trace_policy: (
        LayerTracePolicy | str
    ) = field(
        default_factory=LayerTracePolicy
    )

    training: bool = True
    audit_mode: bool = False
    source_model_fingerprint: str | None = None

    schema_version: str = (
        STACK_INPUTS_SCHEMA_VERSION
    )

    def __post_init__(self) -> None:
        _require_source_inputs(
            self.source_inputs
        )
        _require_positive_int(
            "num_layers",
            self.num_layers,
        )

        object.__setattr__(
            self,
            "sharing_policy",
            _normalize_sharing_policy(
                self.sharing_policy
            ),
        )
        object.__setattr__(
            self,
            "retention_policy",
            _normalize_retention_policy(
                self.retention_policy
            ),
        )
        object.__setattr__(
            self,
            "layer_trace_policy",
            _resolve_layer_trace_policy(
                self.layer_trace_policy
            ),
        )

        _require_boolean(
            "training",
            self.training,
        )
        _require_boolean(
            "audit_mode",
            self.audit_mode,
        )
        _require_optional_nonempty_string(
            "source_model_fingerprint",
            self.source_model_fingerprint,
        )
        _require_nonempty_string(
            "schema_version",
            self.schema_version,
        )

    @property
    def num_nodes(self) -> int:
        return self.source_inputs.num_nodes

    @property
    def hidden_dim(self) -> int:
        return self.source_inputs.hidden_dim

    @property
    def dtype(self) -> torch.dtype:
        return self.source_inputs.dtype

    @property
    def device(self) -> torch.device:
        return self.source_inputs.device

    @property
    def layer_trace_mode(self) -> str:
        return self.layer_trace_policy.mode

    @property
    def expected_retained_layer_indices(
        self,
    ) -> tuple[int, ...]:
        return expected_retained_layer_indices(
            retention_policy=(
                self.retention_policy
            ),
            num_layers=self.num_layers,
        )

    def numerical_architecture_dict(
        self,
    ) -> dict[str, Any]:
        return {
            "schema_version": (
                self.schema_version
            ),
            "num_layers": (
                self.num_layers
            ),
            "sharing_policy": (
                self.sharing_policy
            ),
            "num_nodes": (
                self.num_nodes
            ),
            "hidden_dim": (
                self.hidden_dim
            ),
            "dtype": (
                str(self.dtype)
            ),
        }

    def execution_contract_dict(
        self,
    ) -> dict[str, Any]:
        return {
            "schema_version": (
                self.schema_version
            ),
            "retention_policy": (
                self.retention_policy
            ),
            "layer_trace_policy": (
                self
                .layer_trace_policy
                .architecture_dict()
            ),
            "training": (
                self.training
            ),
            "audit_mode": (
                self.audit_mode
            ),
        }

    def execution_contract_fingerprint(
        self,
    ) -> str:
        return _fingerprint(
            self.execution_contract_dict()
        )

    def lineage_dict(
        self,
    ) -> dict[str, Any]:
        return {
            "schema_version": (
                self.schema_version
            ),
            "source_inputs_lineage_fingerprint": (
                self
                .source_inputs
                .lineage_fingerprint()
            ),
            "source_model_fingerprint": (
                self.source_model_fingerprint
            ),
            "execution_contract_fingerprint": (
                self.execution_contract_fingerprint()
            ),
        }

    def lineage_fingerprint(
        self,
    ) -> str:
        return _fingerprint(
            self.lineage_dict()
        )


# =============================================================================
# Per-depth compact execution record
# =============================================================================


@dataclass(slots=True, frozen=True)
class FunctionalMessagePassingStackDepthRecord:
    """
    Compact provenance and regularization record for one executed depth.

    The record does not retain a complete layer run unless its public output is
    selected by stack retention. It therefore preserves auditable metadata
    without defeating the ordinary memory-saving path.
    """

    layer_index: int

    source_inputs_lineage_fingerprint: str
    source_node_state_lineage_fingerprint: str

    output_architecture_fingerprint: str
    output_parameter_fingerprint: str
    output_lineage_fingerprint: str

    regularization_terms: Mapping[
        str,
        torch.Tensor,
    ] = field(default_factory=dict)

    retained_output: (
        FunctionalMessagePassingLayerOutput
        | None
    ) = None

    schema_version: str = (
        STACK_DEPTH_RECORD_SCHEMA_VERSION
    )

    def __post_init__(self) -> None:
        _require_nonnegative_int(
            "layer_index",
            self.layer_index,
        )
        _require_nonempty_string(
            "source_inputs_lineage_fingerprint",
            self.source_inputs_lineage_fingerprint,
        )
        _require_nonempty_string(
            "source_node_state_lineage_fingerprint",
            self.source_node_state_lineage_fingerprint,
        )
        _require_nonempty_string(
            "output_architecture_fingerprint",
            self.output_architecture_fingerprint,
        )
        _require_nonempty_string(
            "output_parameter_fingerprint",
            self.output_parameter_fingerprint,
        )
        _require_nonempty_string(
            "output_lineage_fingerprint",
            self.output_lineage_fingerprint,
        )
        _require_nonempty_string(
            "schema_version",
            self.schema_version,
        )

        if self.retained_output is not None:
            if not isinstance(
                self.retained_output,
                FunctionalMessagePassingLayerOutput,
            ):
                raise TypeError(
                    "retained_output must be a "
                    "FunctionalMessagePassingLayerOutput or None."
                )

            output = self.retained_output

            if output.layer_index != (
                self.layer_index
            ):
                raise ValueError(
                    "retained_output.layer_index differs from the depth "
                    "record index."
                )

            if (
                output.encoder_architecture_fingerprint
                != self.output_architecture_fingerprint
            ):
                raise ValueError(
                    "retained_output architecture fingerprint differs from "
                    "the compact depth record."
                )

            if (
                output.lineage_fingerprint
                != self.output_lineage_fingerprint
            ):
                raise ValueError(
                    "retained_output lineage fingerprint differs from the "
                    "compact depth record."
                )

            device = output.updated_node_state.device
        else:
            if self.regularization_terms:
                first = next(
                    iter(
                        self
                        .regularization_terms
                        .values()
                    )
                )
                device = first.device
            else:
                device = torch.device(
                    "cpu"
                )

        immutable_regularization = (
            _immutable_scalar_tensor_mapping(
                "regularization_terms",
                self.regularization_terms,
                device=device,
            )
        )

        if (
            self.retained_output is not None
            and not _same_scalar_mapping_identity(
                immutable_regularization,
                self
                .retained_output
                .regularization_terms,
            )
        ):
            raise ValueError(
                "Depth-record regularization terms must preserve exact "
                "retained-output scalar tensor identity."
            )

        object.__setattr__(
            self,
            "regularization_terms",
            immutable_regularization,
        )

    @classmethod
    def from_layer_run(
        cls,
        run: FunctionalMessagePassingLayerRun,
        *,
        retain_output: bool,
    ) -> "FunctionalMessagePassingStackDepthRecord":
        validate_functional_message_passing_layer_run(
            run
        )
        _require_boolean(
            "retain_output",
            retain_output,
        )

        return cls(
            layer_index=(
                run.layer_index
            ),
            source_inputs_lineage_fingerprint=(
                run.source_inputs
                .lineage_fingerprint()
            ),
            source_node_state_lineage_fingerprint=(
                _resolve_lineage_fingerprint(
                    run
                    .source_inputs
                    .node_state,
                    name=(
                        "source node-state lineage fingerprint"
                    ),
                )
            ),
            output_architecture_fingerprint=(
                run
                .public_output
                .encoder_architecture_fingerprint
            ),
            output_parameter_fingerprint=(
                run
                .internal_output
                .layer_parameter_fingerprint
            ),
            output_lineage_fingerprint=(
                run
                .public_output
                .lineage_fingerprint
            ),
            regularization_terms=(
                run
                .public_output
                .regularization_terms
            ),
            retained_output=(
                run.public_output
                if retain_output
                else None
            ),
        )

    @property
    def retained(self) -> bool:
        return self.retained_output is not None

    @property
    def regularization_names(
        self,
    ) -> tuple[str, ...]:
        return tuple(
            self.regularization_terms
        )

    def lineage_dict(
        self,
    ) -> dict[str, Any]:
        return {
            "schema_version": (
                self.schema_version
            ),
            "layer_index": (
                self.layer_index
            ),
            "source_inputs_lineage_fingerprint": (
                self.source_inputs_lineage_fingerprint
            ),
            "source_node_state_lineage_fingerprint": (
                self.source_node_state_lineage_fingerprint
            ),
            "output_architecture_fingerprint": (
                self.output_architecture_fingerprint
            ),
            "output_parameter_fingerprint": (
                self.output_parameter_fingerprint
            ),
            "output_lineage_fingerprint": (
                self.output_lineage_fingerprint
            ),
            "regularization_names": list(
                self.regularization_names
            ),
            "retained": (
                self.retained
            ),
        }

    def lineage_fingerprint(
        self,
    ) -> str:
        return _fingerprint(
            self.lineage_dict()
        )


# =============================================================================
# Explicit high-cost complete stack trace
# =============================================================================


@dataclass(slots=True, frozen=True)
class FunctionalMessagePassingStackTrace:
    """
    Explicit complete audit trace retaining every exact layer run.
    """

    stack_inputs: FunctionalMessagePassingStackInputs
    layer_runs: tuple[
        FunctionalMessagePassingLayerRun,
        ...,
    ]

    schema_version: str = (
        STACK_TRACE_SCHEMA_VERSION
    )

    def __post_init__(self) -> None:
        if not isinstance(
            self.stack_inputs,
            FunctionalMessagePassingStackInputs,
        ):
            raise TypeError(
                "stack_inputs must be a "
                "FunctionalMessagePassingStackInputs."
            )

        if not self.stack_inputs.audit_mode:
            raise ValueError(
                "A complete stack trace is permitted only when audit_mode "
                "is enabled."
            )

        if not isinstance(
            self.layer_runs,
            tuple,
        ):
            raise TypeError(
                "layer_runs must be a tuple."
            )

        if len(self.layer_runs) != (
            self.stack_inputs.num_layers
        ):
            raise ValueError(
                "A complete stack trace must retain exactly num_layers "
                "layer runs."
            )

        previous_run: (
            FunctionalMessagePassingLayerRun
            | None
        ) = None

        for index, run in enumerate(
            self.layer_runs
        ):
            if not isinstance(
                run,
                FunctionalMessagePassingLayerRun,
            ):
                raise TypeError(
                    f"layer_runs[{index}] must be a "
                    "FunctionalMessagePassingLayerRun."
                )

            validate_functional_message_passing_layer_run(
                run
            )

            if run.layer_index != index:
                raise ValueError(
                    "Complete stack trace layer runs must use contiguous "
                    "zero-based layer indices."
                )

            if (
                run.layer_inputs.training
                != self.stack_inputs.training
            ):
                raise ValueError(
                    "Layer-run training mode differs from stack_inputs."
                )

            if (
                run
                .layer_inputs
                .trace_policy
                != self.stack_inputs.layer_trace_policy
            ):
                raise ValueError(
                    "Layer-run trace policy differs from stack_inputs."
                )

            _validate_rebound_layer_source(
                original=(
                    self.stack_inputs
                    .source_inputs
                ),
                candidate=(
                    run.source_inputs
                ),
                layer_index=index,
                previous_run=previous_run,
            )

            previous_run = run

        _require_nonempty_string(
            "schema_version",
            self.schema_version,
        )

    @property
    def final_run(
        self,
    ) -> FunctionalMessagePassingLayerRun:
        return self.layer_runs[-1]

    @property
    def final_node_state(
        self,
    ) -> torch.Tensor:
        return self.final_run.updated_node_state

    @property
    def layer_outputs(
        self,
    ) -> tuple[
        FunctionalMessagePassingLayerOutput,
        ...,
    ]:
        return tuple(
            run.public_output
            for run in self.layer_runs
        )

    @property
    def layer_indices(
        self,
    ) -> tuple[int, ...]:
        return tuple(
            run.layer_index
            for run in self.layer_runs
        )

    def lineage_dict(
        self,
    ) -> dict[str, Any]:
        return {
            "schema_version": (
                self.schema_version
            ),
            "stack_inputs_lineage_fingerprint": (
                self
                .stack_inputs
                .lineage_fingerprint()
            ),
            "layer_run_lineage_fingerprints": [
                run
                .internal_output
                .lineage_fingerprint
                for run in self.layer_runs
            ],
        }

    def lineage_fingerprint(
        self,
    ) -> str:
        return _fingerprint(
            self.lineage_dict()
        )


# =============================================================================
# Complete internal stack computation output
# =============================================================================


@dataclass(slots=True, frozen=True)
class FunctionalMessagePassingStackComputationOutput:
    """
    Complete bounded internal stack result without mandatory heavy run retention.
    """

    stack_inputs: FunctionalMessagePassingStackInputs
    final_node_state: torch.Tensor

    depth_records: tuple[
        FunctionalMessagePassingStackDepthRecord,
        ...,
    ]

    stack_architecture_fingerprint: str
    stack_parameter_fingerprint: str
    execution_contract_fingerprint: str
    lineage_fingerprint: str

    audit_trace: (
        FunctionalMessagePassingStackTrace
        | None
    ) = None

    schema_version: str = (
        STACK_COMPUTATION_OUTPUT_SCHEMA_VERSION
    )

    def __post_init__(self) -> None:
        if not isinstance(
            self.stack_inputs,
            FunctionalMessagePassingStackInputs,
        ):
            raise TypeError(
                "stack_inputs must be a "
                "FunctionalMessagePassingStackInputs."
            )

        _require_float_matrix(
            "final_node_state",
            self.final_node_state,
            num_nodes=(
                self.stack_inputs.num_nodes
            ),
            hidden_dim=(
                self.stack_inputs.hidden_dim
            ),
            dtype=(
                self.stack_inputs.dtype
            ),
            device=(
                self.stack_inputs.device
            ),
        )

        if not isinstance(
            self.depth_records,
            tuple,
        ):
            raise TypeError(
                "depth_records must be a tuple."
            )

        if len(self.depth_records) != (
            self.stack_inputs.num_layers
        ):
            raise ValueError(
                "depth_records must contain exactly one record per "
                "executed layer."
            )

        retained_indices: list[int] = []

        for expected_index, record in enumerate(
            self.depth_records
        ):
            if not isinstance(
                record,
                FunctionalMessagePassingStackDepthRecord,
            ):
                raise TypeError(
                    f"depth_records[{expected_index}] must be a "
                    "FunctionalMessagePassingStackDepthRecord."
                )

            if record.layer_index != (
                expected_index
            ):
                raise ValueError(
                    "depth_records must use contiguous zero-based layer "
                    "indices."
                )

            if record.retained:
                retained_indices.append(
                    expected_index
                )

        expected_indices = (
            self.stack_inputs
            .expected_retained_layer_indices
        )

        if tuple(retained_indices) != (
            expected_indices
        ):
            raise ValueError(
                "Retained depth records do not match the configured stack "
                "retention policy."
            )

        final_retained = (
            self.depth_records[-1]
            .retained_output
        )

        if final_retained is not None:
            if (
                self.final_node_state
                is not final_retained
                .updated_node_state
            ):
                raise ValueError(
                    "final_node_state must preserve exact final retained "
                    "layer-output tensor identity."
                )

        _require_nonempty_string(
            "stack_architecture_fingerprint",
            self.stack_architecture_fingerprint,
        )
        _require_nonempty_string(
            "stack_parameter_fingerprint",
            self.stack_parameter_fingerprint,
        )
        _require_nonempty_string(
            "execution_contract_fingerprint",
            self.execution_contract_fingerprint,
        )
        _require_nonempty_string(
            "lineage_fingerprint",
            self.lineage_fingerprint,
        )
        _require_nonempty_string(
            "schema_version",
            self.schema_version,
        )

        if (
            self.execution_contract_fingerprint
            != self.stack_inputs
            .execution_contract_fingerprint()
        ):
            raise ValueError(
                "execution_contract_fingerprint differs from stack_inputs."
            )

        if self.audit_trace is not None:
            if not isinstance(
                self.audit_trace,
                FunctionalMessagePassingStackTrace,
            ):
                raise TypeError(
                    "audit_trace must be a "
                    "FunctionalMessagePassingStackTrace or None."
                )

            if (
                self.audit_trace.stack_inputs
                is not self.stack_inputs
            ):
                raise ValueError(
                    "audit_trace must preserve the exact stack_inputs "
                    "object."
                )

            if (
                self.final_node_state
                is not self.audit_trace
                .final_node_state
            ):
                raise ValueError(
                    "final_node_state must preserve exact audit-trace "
                    "final-state tensor identity."
                )

            for index, (
                record,
                run,
            ) in enumerate(
                zip(
                    self.depth_records,
                    self.audit_trace.layer_runs,
                    strict=True,
                )
            ):
                if (
                    record.output_architecture_fingerprint
                    != run
                    .public_output
                    .encoder_architecture_fingerprint
                ):
                    raise ValueError(
                        f"depth_records[{index}] architecture differs from "
                        "the audit trace."
                    )

                if (
                    record.output_parameter_fingerprint
                    != run
                    .internal_output
                    .layer_parameter_fingerprint
                ):
                    raise ValueError(
                        f"depth_records[{index}] parameter fingerprint "
                        "differs from the audit trace."
                    )

                if (
                    record.output_lineage_fingerprint
                    != run
                    .public_output
                    .lineage_fingerprint
                ):
                    raise ValueError(
                        f"depth_records[{index}] lineage differs from the "
                        "audit trace."
                    )

                if not _same_scalar_mapping_identity(
                    record.regularization_terms,
                    run
                    .public_output
                    .regularization_terms,
                ):
                    raise ValueError(
                        f"depth_records[{index}] regularization terms "
                        "differ from the audit trace."
                    )

                if (
                    record.retained_output is not None
                    and record.retained_output
                    is not run.public_output
                ):
                    raise ValueError(
                        f"depth_records[{index}] retained output must be "
                        "the exact audit-trace public output."
                    )

        elif self.stack_inputs.audit_mode:
            raise ValueError(
                "audit_mode requires a complete FunctionalMessagePassingStackTrace."
            )

    @property
    def retained_layer_outputs(
        self,
    ) -> tuple[
        FunctionalMessagePassingLayerOutput,
        ...,
    ]:
        return tuple(
            record.retained_output
            for record in self.depth_records
            if record.retained_output
            is not None
        )

    @property
    def retained_layer_indices(
        self,
    ) -> tuple[int, ...]:
        return tuple(
            output.layer_index
            for output
            in self.retained_layer_outputs
        )

    @property
    def layer_regularization_terms(
        self,
    ) -> tuple[
        Mapping[str, torch.Tensor],
        ...,
    ]:
        return tuple(
            record.regularization_terms
            for record in self.depth_records
        )

    @property
    def layer_architecture_fingerprints(
        self,
    ) -> tuple[str, ...]:
        return tuple(
            record.output_architecture_fingerprint
            for record in self.depth_records
        )

    @property
    def layer_parameter_fingerprints(
        self,
    ) -> tuple[str, ...]:
        return tuple(
            record.output_parameter_fingerprint
            for record in self.depth_records
        )

    @property
    def layer_lineage_fingerprints(
        self,
    ) -> tuple[str, ...]:
        return tuple(
            record.output_lineage_fingerprint
            for record in self.depth_records
        )


# =============================================================================
# Public stack output
# =============================================================================


@dataclass(slots=True, frozen=True)
class FunctionalMessagePassingStackOutput:
    """
    Public final stack state plus policy-selected public layer outputs.
    """

    final_node_state: torch.Tensor
    source_inputs: FunctionalMessagePassingInputs

    num_layers: int
    sharing_policy: str
    retention_policy: str
    layer_trace_policy: (
        LayerTracePolicy | str
    )

    retained_layer_outputs: tuple[
        FunctionalMessagePassingLayerOutput,
        ...,
    ] = ()

    layer_regularization_terms: tuple[
        Mapping[str, torch.Tensor],
        ...,
    ] = ()

    encoder_architecture_fingerprint: str = ""
    execution_contract_fingerprint: str = ""
    lineage_fingerprint: str = ""

    schema_version: str = (
        STACK_PUBLIC_OUTPUT_SCHEMA_VERSION
    )

    def __post_init__(self) -> None:
        _require_source_inputs(
            self.source_inputs
        )
        _require_positive_int(
            "num_layers",
            self.num_layers,
        )

        object.__setattr__(
            self,
            "sharing_policy",
            _normalize_sharing_policy(
                self.sharing_policy
            ),
        )
        object.__setattr__(
            self,
            "retention_policy",
            _normalize_retention_policy(
                self.retention_policy
            ),
        )
        object.__setattr__(
            self,
            "layer_trace_policy",
            _resolve_layer_trace_policy(
                self.layer_trace_policy
            ),
        )

        _require_float_matrix(
            "final_node_state",
            self.final_node_state,
            num_nodes=(
                self.source_inputs.num_nodes
            ),
            hidden_dim=(
                self.source_inputs.hidden_dim
            ),
            dtype=(
                self.source_inputs.dtype
            ),
            device=(
                self.source_inputs.device
            ),
        )

        if not isinstance(
            self.retained_layer_outputs,
            tuple,
        ):
            raise TypeError(
                "retained_layer_outputs must be a tuple."
            )

        expected_indices = (
            expected_retained_layer_indices(
                retention_policy=(
                    self.retention_policy
                ),
                num_layers=(
                    self.num_layers
                ),
            )
        )

        observed_indices: list[int] = []

        for position, output in enumerate(
            self.retained_layer_outputs
        ):
            if not isinstance(
                output,
                FunctionalMessagePassingLayerOutput,
            ):
                raise TypeError(
                    f"retained_layer_outputs[{position}] must be a "
                    "FunctionalMessagePassingLayerOutput."
                )

            observed_indices.append(
                output.layer_index
            )

            _validate_structural_context(
                original=(
                    self.source_inputs
                ),
                candidate=(
                    output.source_inputs
                ),
                name=(
                    f"retained_layer_outputs[{position}].source_inputs"
                ),
            )

            if (
                output.source_inputs.dtype
                != self.source_inputs.dtype
            ):
                raise ValueError(
                    "Retained layer output dtype differs from stack input."
                )

            if not _devices_match(
                output.source_inputs.device,
                self.source_inputs.device,
            ):
                raise ValueError(
                    "Retained layer output device differs from stack input."
                )

        if tuple(observed_indices) != (
            expected_indices
        ):
            raise ValueError(
                "retained_layer_outputs do not match the configured "
                "retention policy."
            )

        immutable_regularization = (
            _immutable_layer_regularization_terms(
                self.layer_regularization_terms,
                num_layers=(
                    self.num_layers
                ),
                device=(
                    self.final_node_state
                    .device
                ),
            )
        )
        object.__setattr__(
            self,
            "layer_regularization_terms",
            immutable_regularization,
        )

        for output in (
            self.retained_layer_outputs
        ):
            if not _same_scalar_mapping_identity(
                self
                .layer_regularization_terms[
                    output.layer_index
                ],
                output.regularization_terms,
            ):
                raise ValueError(
                    "Retained layer-output regularization terms must "
                    "preserve exact per-depth scalar tensor identity."
                )

        final_output = (
            self.final_layer_output
        )

        if final_output is not None:
            if self.final_node_state is not (
                final_output.updated_node_state
            ):
                raise ValueError(
                    "final_node_state must preserve exact retained final "
                    "layer-output tensor identity."
                )

        _require_nonempty_string(
            "encoder_architecture_fingerprint",
            self.encoder_architecture_fingerprint,
        )
        _require_nonempty_string(
            "execution_contract_fingerprint",
            self.execution_contract_fingerprint,
        )
        _require_nonempty_string(
            "lineage_fingerprint",
            self.lineage_fingerprint,
        )
        _require_nonempty_string(
            "schema_version",
            self.schema_version,
        )

    @property
    def num_nodes(self) -> int:
        return int(
            self.final_node_state.shape[0]
        )

    @property
    def hidden_dim(self) -> int:
        return int(
            self.final_node_state.shape[1]
        )

    @property
    def dtype(self) -> torch.dtype:
        return self.final_node_state.dtype

    @property
    def device(self) -> torch.device:
        return self.final_node_state.device

    @property
    def layer_trace_mode(self) -> str:
        return self.layer_trace_policy.mode

    @property
    def num_retained_layers(self) -> int:
        return len(
            self.retained_layer_outputs
        )

    @property
    def has_retained_layers(self) -> bool:
        return bool(
            self.retained_layer_outputs
        )

    @property
    def retained_layer_indices(
        self,
    ) -> tuple[int, ...]:
        return tuple(
            output.layer_index
            for output
            in self.retained_layer_outputs
        )

    @property
    def final_layer_output(
        self,
    ) -> (
        FunctionalMessagePassingLayerOutput
        | None
    ):
        if not self.retained_layer_outputs:
            return None

        candidate = (
            self.retained_layer_outputs[-1]
        )

        if candidate.layer_index != (
            self.num_layers - 1
        ):
            return None

        return candidate


def assemble_functional_message_passing_stack_output(
    internal_output: FunctionalMessagePassingStackComputationOutput,
) -> FunctionalMessagePassingStackOutput:
    """
    Assemble the bounded public output from one validated internal result.
    """

    if not isinstance(
        internal_output,
        FunctionalMessagePassingStackComputationOutput,
    ):
        raise TypeError(
            "internal_output must be a "
            "FunctionalMessagePassingStackComputationOutput."
        )

    inputs = internal_output.stack_inputs

    return FunctionalMessagePassingStackOutput(
        final_node_state=(
            internal_output
            .final_node_state
        ),
        source_inputs=(
            inputs.source_inputs
        ),
        num_layers=(
            inputs.num_layers
        ),
        sharing_policy=(
            inputs.sharing_policy
        ),
        retention_policy=(
            inputs.retention_policy
        ),
        layer_trace_policy=(
            inputs.layer_trace_policy
        ),
        retained_layer_outputs=(
            internal_output
            .retained_layer_outputs
        ),
        layer_regularization_terms=(
            internal_output
            .layer_regularization_terms
        ),
        encoder_architecture_fingerprint=(
            internal_output
            .stack_architecture_fingerprint
        ),
        execution_contract_fingerprint=(
            internal_output
            .execution_contract_fingerprint
        ),
        lineage_fingerprint=(
            internal_output
            .lineage_fingerprint
        ),
    )


def validate_public_stack_output(
    *,
    public_output: FunctionalMessagePassingStackOutput,
    internal_output: FunctionalMessagePassingStackComputationOutput,
) -> None:
    """
    Validate exact public/internal identity for one stack computation.
    """

    if not isinstance(
        public_output,
        FunctionalMessagePassingStackOutput,
    ):
        raise TypeError(
            "public_output must be a "
            "FunctionalMessagePassingStackOutput."
        )

    if not isinstance(
        internal_output,
        FunctionalMessagePassingStackComputationOutput,
    ):
        raise TypeError(
            "internal_output must be a "
            "FunctionalMessagePassingStackComputationOutput."
        )

    inputs = internal_output.stack_inputs

    if public_output.source_inputs is not (
        inputs.source_inputs
    ):
        raise ValueError(
            "public_output must preserve exact original stack source inputs."
        )

    if public_output.final_node_state is not (
        internal_output.final_node_state
    ):
        raise ValueError(
            "public_output must preserve exact internal final-node-state "
            "tensor identity."
        )

    if public_output.num_layers != (
        inputs.num_layers
    ):
        raise ValueError(
            "public_output.num_layers differs from stack_inputs."
        )

    if public_output.sharing_policy != (
        inputs.sharing_policy
    ):
        raise ValueError(
            "public_output sharing policy differs from stack_inputs."
        )

    if public_output.retention_policy != (
        inputs.retention_policy
    ):
        raise ValueError(
            "public_output retention policy differs from stack_inputs."
        )

    if public_output.layer_trace_policy != (
        inputs.layer_trace_policy
    ):
        raise ValueError(
            "public_output layer trace policy differs from stack_inputs."
        )

    if len(
        public_output.retained_layer_outputs
    ) != len(
        internal_output.retained_layer_outputs
    ):
        raise ValueError(
            "public_output retained-layer count differs from internal output."
        )

    for index, (
        public_layer,
        internal_layer,
    ) in enumerate(
        zip(
            public_output
            .retained_layer_outputs,
            internal_output
            .retained_layer_outputs,
            strict=True,
        )
    ):
        if public_layer is not internal_layer:
            raise ValueError(
                f"public_output.retained_layer_outputs[{index}] must "
                "preserve exact internal object identity."
            )

    for index, (
        public_terms,
        internal_terms,
    ) in enumerate(
        zip(
            public_output
            .layer_regularization_terms,
            internal_output
            .layer_regularization_terms,
            strict=True,
        )
    ):
        if not _same_scalar_mapping_identity(
            public_terms,
            internal_terms,
        ):
            raise ValueError(
                f"public layer regularization mapping {index} differs "
                "from the internal output."
            )

    if (
        public_output.encoder_architecture_fingerprint
        != internal_output.stack_architecture_fingerprint
    ):
        raise ValueError(
            "public stack architecture fingerprint differs from the "
            "internal output."
        )

    if (
        public_output.execution_contract_fingerprint
        != internal_output.execution_contract_fingerprint
    ):
        raise ValueError(
            "public execution-contract fingerprint differs from the "
            "internal output."
        )

    if (
        public_output.lineage_fingerprint
        != internal_output.lineage_fingerprint
    ):
        raise ValueError(
            "public lineage fingerprint differs from the internal output."
        )


# =============================================================================
# Complete stack run
# =============================================================================


@dataclass(slots=True, frozen=True)
class FunctionalMessagePassingStackRun:
    """
    Complete exact stack execution record without mandatory diagnostic report.
    """

    stack_inputs: FunctionalMessagePassingStackInputs
    internal_output: FunctionalMessagePassingStackComputationOutput
    public_output: FunctionalMessagePassingStackOutput

    schema_version: str = (
        STACK_RUN_SCHEMA_VERSION
    )

    def __post_init__(self) -> None:
        if not isinstance(
            self.stack_inputs,
            FunctionalMessagePassingStackInputs,
        ):
            raise TypeError(
                "stack_inputs must be a "
                "FunctionalMessagePassingStackInputs."
            )

        if not isinstance(
            self.internal_output,
            FunctionalMessagePassingStackComputationOutput,
        ):
            raise TypeError(
                "internal_output must be a "
                "FunctionalMessagePassingStackComputationOutput."
            )

        if not isinstance(
            self.public_output,
            FunctionalMessagePassingStackOutput,
        ):
            raise TypeError(
                "public_output must be a "
                "FunctionalMessagePassingStackOutput."
            )

        if self.internal_output.stack_inputs is not (
            self.stack_inputs
        ):
            raise ValueError(
                "internal_output must preserve exact stack_inputs."
            )

        validate_public_stack_output(
            public_output=(
                self.public_output
            ),
            internal_output=(
                self.internal_output
            ),
        )

        _require_nonempty_string(
            "schema_version",
            self.schema_version,
        )

    @property
    def source_inputs(
        self,
    ) -> FunctionalMessagePassingInputs:
        return self.stack_inputs.source_inputs

    @property
    def final_node_state(
        self,
    ) -> torch.Tensor:
        return self.public_output.final_node_state

    @property
    def retained_layer_outputs(
        self,
    ) -> tuple[
        FunctionalMessagePassingLayerOutput,
        ...,
    ]:
        return (
            self.public_output
            .retained_layer_outputs
        )

    @property
    def trace(
        self,
    ) -> (
        FunctionalMessagePassingStackTrace
        | None
    ):
        return self.internal_output.audit_trace


def validate_functional_message_passing_stack_run(
    run: FunctionalMessagePassingStackRun,
) -> None:
    if not isinstance(
        run,
        FunctionalMessagePassingStackRun,
    ):
        raise TypeError(
            "run must be a FunctionalMessagePassingStackRun."
        )

    if run.internal_output.stack_inputs is not (
        run.stack_inputs
    ):
        raise ValueError(
            "run.internal_output must preserve exact stack_inputs."
        )

    validate_public_stack_output(
        public_output=(
            run.public_output
        ),
        internal_output=(
            run.internal_output
        ),
    )


@dataclass(slots=True, frozen=True)
class FunctionalMessagePassingStackRunWithDiagnostics:
    """
    Complete stack run plus an explicit tensor-free diagnostic report.
    """

    run: FunctionalMessagePassingStackRun
    diagnostic_report: Mapping[str, Any]

    schema_version: str = (
        STACK_RUN_WITH_DIAGNOSTICS_SCHEMA_VERSION
    )

    def __post_init__(self) -> None:
        validate_functional_message_passing_stack_run(
            self.run
        )

        object.__setattr__(
            self,
            "diagnostic_report",
            _immutable_tensor_free_mapping(
                self.diagnostic_report
            ),
        )

        _require_nonempty_string(
            "schema_version",
            self.schema_version,
        )

    @property
    def public_output(
        self,
    ) -> FunctionalMessagePassingStackOutput:
        return self.run.public_output

    @property
    def internal_output(
        self,
    ) -> FunctionalMessagePassingStackComputationOutput:
        return self.run.internal_output


# =============================================================================
# Compact aliases
# =============================================================================


StackInputs = FunctionalMessagePassingStackInputs
StackDepthRecord = FunctionalMessagePassingStackDepthRecord
StackTrace = FunctionalMessagePassingStackTrace
StackComputationOutput = (
    FunctionalMessagePassingStackComputationOutput
)
StackOutput = FunctionalMessagePassingStackOutput
StackRun = FunctionalMessagePassingStackRun
StackRunWithDiagnostics = (
    FunctionalMessagePassingStackRunWithDiagnostics
)

assemble_stack_output = (
    assemble_functional_message_passing_stack_output
)
validate_stack_run = (
    validate_functional_message_passing_stack_run
)


__all__ = (
    # Public identity.
    "FUNCTIONAL_MESSAGE_PASSING_STACK_SCHEMA_VERSION",
    "STACK_INPUTS_SCHEMA_VERSION",
    "STACK_DEPTH_RECORD_SCHEMA_VERSION",
    "STACK_TRACE_SCHEMA_VERSION",
    "STACK_COMPUTATION_OUTPUT_SCHEMA_VERSION",
    "STACK_PUBLIC_OUTPUT_SCHEMA_VERSION",
    "STACK_RUN_SCHEMA_VERSION",
    "STACK_RUN_WITH_DIAGNOSTICS_SCHEMA_VERSION",
    "FUNCTIONAL_MESSAGE_PASSING_STACK_OPERATION_ORDER",
    "FUNCTIONAL_MESSAGE_PASSING_STACK_SCIENTIFIC_INTERPRETATION",
    "FUNCTIONAL_MESSAGE_PASSING_STACK_OUTPUT_SCHEMA",
    "FUNCTIONAL_MESSAGE_PASSING_STACK_REIMPLEMENTS_LAYER_MATH",
    "FUNCTIONAL_MESSAGE_PASSING_STACK_RETENTION_AFFECTS_NUMERICS",
    "FUNCTIONAL_MESSAGE_PASSING_STACK_LAYER_TRACE_AFFECTS_NUMERICS",
    "FUNCTIONAL_MESSAGE_PASSING_STACK_DIAGNOSTICS_AFFECT_NUMERICS",
    "FUNCTIONAL_MESSAGE_PASSING_STACK_HIDDEN_WIDTH_CHANGES_SUPPORTED",
    "FUNCTIONAL_MESSAGE_PASSING_STACK_ZERO_LAYER_SUPPORTED",
    "FUNCTIONAL_MESSAGE_PASSING_STACK_PARTIAL_SHARING_SUPPORTED",
    # Policy-facing helpers.
    "expected_retained_layer_indices",
    # Stack schemas.
    "FunctionalMessagePassingStackInputs",
    "StackInputs",
    "FunctionalMessagePassingStackDepthRecord",
    "StackDepthRecord",
    "FunctionalMessagePassingStackTrace",
    "StackTrace",
    "FunctionalMessagePassingStackComputationOutput",
    "StackComputationOutput",
    "FunctionalMessagePassingStackOutput",
    "StackOutput",
    "FunctionalMessagePassingStackRun",
    "StackRun",
    "FunctionalMessagePassingStackRunWithDiagnostics",
    "StackRunWithDiagnostics",
    # Assembly and validation.
    "assemble_functional_message_passing_stack_output",
    "assemble_stack_output",
    "validate_public_stack_output",
    "validate_functional_message_passing_stack_run",
    "validate_stack_run",
)
