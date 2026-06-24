"""
Integration and failure tests for the functional-message-passing stack.

Target repository path:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            tests/
                stack/
                    test_functional_message_passing_stack.py

Implementations under test:
    functional_message_passing/
        stack/
            schemas.py
            diagnostics.py
            stack.py

The policy vocabulary and ownership-plan surface are tested separately in
``test_stack_policies.py``. This suite exercises the complete stack lifecycle:

- construction and module registration;
- one-layer equivalence;
- depth-wise numerical execution;
- immutable state rebinding;
- independent and fully shared gradients;
- output retention and one-layer trace detail;
- explicit audit traces;
- per-depth regularization preservation;
- deterministic evaluation and stochastic training;
- state-dict serialization;
- architecture, parameter, execution, and lineage fingerprints;
- explicit tensor-free diagnostics;
- CPU, float64, empty-edge, and optional CUDA execution;
- defensive validation and tamper rejection;
- functional builders, execution helpers, aliases, and module exports.

Controlled boundary
-------------------
The stack is tested with a small differentiable layer double and lightweight
metadata-preserving FMP input contracts. The doubles expose the exact public
surfaces consumed by the stack while avoiding a second end-to-end test of
relation transforms, gating, attention, message construction, aggregation,
residual updates, and normalization. Those one-layer numerical contracts are
covered by the completed layer suites.

The stack schemas themselves remain the real production dataclasses. Only
their upstream one-layer and top-level FMP type sentinels are patched so this
suite isolates stack behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import io
import json
import math
from types import MappingProxyType
from typing import Any, Final, Mapping

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.constants import (
    STACK_RETENTION_ALL_LAYERS,
    STACK_RETENTION_FINAL_LAYER,
    STACK_RETENTION_NONE,
    STACK_SHARING_FULLY_SHARED,
    STACK_SHARING_INDEPENDENT,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.functional_message_passing.layer.schemas import (
    LAYER_TRACE_FULL,
    LAYER_TRACE_NODE,
    LAYER_TRACE_NONE,
    LayerTracePolicy,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.functional_message_passing.stack import (
    diagnostics as diagnostics_module,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.functional_message_passing.stack import (
    schemas as stack_schemas_module,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.functional_message_passing.stack import (
    sharing_policy as sharing_module,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.functional_message_passing.stack import (
    stack as stack_module,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.functional_message_passing.stack.diagnostics import (
    STACK_DIAGNOSTICS_AFFECT_NUMERICAL_RESULTS,
    STACK_DIAGNOSTICS_ESTABLISH_CALIBRATION,
    STACK_DIAGNOSTICS_ESTABLISH_CAUSALITY,
    STACK_DIAGNOSTICS_ESTABLISH_FAITHFULNESS,
    STACK_DIAGNOSTICS_ESTABLISH_IDENTIFIABILITY,
    STACK_DIAGNOSTICS_RETAIN_AUTOGRAD_REFERENCES,
    STACK_DIAGNOSTICS_RETURN_TENSORS,
    STACK_DIAGNOSTICS_SCHEMA_VERSION,
    STACK_DIAGNOSTIC_ALERT_INCOMPLETE_DEPTH_COVERAGE,
    STACK_DIAGNOSTIC_DEPTH_SOURCE_AUDIT_TRACE,
    STACK_DIAGNOSTIC_DEPTH_SOURCE_RETAINED_OUTPUT,
    STACK_DIAGNOSTIC_DEPTH_SOURCE_UNAVAILABLE,
    STACK_DIAGNOSTIC_REPORT_SCHEMA_VERSION,
    STACK_DIAGNOSTIC_SCIENTIFIC_CLAIMS,
    STACK_DIAGNOSTIC_THRESHOLDS_SCHEMA_VERSION,
    STACK_DEPTH_DIAGNOSTIC_SCHEMA_VERSION,
    DepthDiagnostic,
    DiagnosticThresholds,
    FunctionalMessagePassingStackDiagnosticReport,
    FunctionalMessagePassingStackDiagnostics,
    StackDepthDiagnostic,
    StackDiagnosticReport,
    StackDiagnosticThresholds,
    StackDiagnostics,
    build_diagnostic_report,
    build_functional_message_passing_stack_diagnostic_report,
    build_stack_diagnostic_report,
    validate_diagnostic_report,
    validate_functional_message_passing_stack_diagnostic_report,
    validate_stack_diagnostic_report,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.functional_message_passing.stack.schemas import (
    FUNCTIONAL_MESSAGE_PASSING_STACK_DIAGNOSTICS_AFFECT_NUMERICS,
    FUNCTIONAL_MESSAGE_PASSING_STACK_HIDDEN_WIDTH_CHANGES_SUPPORTED,
    FUNCTIONAL_MESSAGE_PASSING_STACK_LAYER_TRACE_AFFECTS_NUMERICS,
    FUNCTIONAL_MESSAGE_PASSING_STACK_OPERATION_ORDER,
    FUNCTIONAL_MESSAGE_PASSING_STACK_OUTPUT_SCHEMA,
    FUNCTIONAL_MESSAGE_PASSING_STACK_PARTIAL_SHARING_SUPPORTED,
    FUNCTIONAL_MESSAGE_PASSING_STACK_REIMPLEMENTS_LAYER_MATH,
    FUNCTIONAL_MESSAGE_PASSING_STACK_RETENTION_AFFECTS_NUMERICS,
    FUNCTIONAL_MESSAGE_PASSING_STACK_SCHEMA_VERSION,
    FUNCTIONAL_MESSAGE_PASSING_STACK_SCIENTIFIC_INTERPRETATION,
    FUNCTIONAL_MESSAGE_PASSING_STACK_ZERO_LAYER_SUPPORTED,
    STACK_COMPUTATION_OUTPUT_SCHEMA_VERSION,
    STACK_DEPTH_RECORD_SCHEMA_VERSION,
    STACK_INPUTS_SCHEMA_VERSION,
    STACK_PUBLIC_OUTPUT_SCHEMA_VERSION,
    STACK_RUN_SCHEMA_VERSION,
    STACK_RUN_WITH_DIAGNOSTICS_SCHEMA_VERSION,
    STACK_TRACE_SCHEMA_VERSION,
    FunctionalMessagePassingStackComputationOutput,
    FunctionalMessagePassingStackDepthRecord,
    FunctionalMessagePassingStackInputs,
    FunctionalMessagePassingStackOutput,
    FunctionalMessagePassingStackRun,
    FunctionalMessagePassingStackRunWithDiagnostics,
    FunctionalMessagePassingStackTrace,
    StackComputationOutput,
    StackDepthRecord,
    StackInputs,
    StackOutput,
    StackRun,
    StackRunWithDiagnostics,
    StackTrace,
    assemble_functional_message_passing_stack_output,
    assemble_stack_output,
    expected_retained_layer_indices,
    validate_functional_message_passing_stack_run,
    validate_public_stack_output,
    validate_stack_run,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.functional_message_passing.stack.sharing_policy import (
    StackLayerSharingPlan,
    StackSharingPolicy,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.functional_message_passing.stack.stack import (
    FUNCTIONAL_MESSAGE_PASSING_STACK_AUDIT_RETENTION,
    FUNCTIONAL_MESSAGE_PASSING_STACK_FULLY_SHARED_REGISTRATION_PREFIX,
    FUNCTIONAL_MESSAGE_PASSING_STACK_INDEPENDENT_REGISTRATION_PREFIX,
    FUNCTIONAL_MESSAGE_PASSING_STACK_LAYER_MATH_OWNED_HERE,
    FUNCTIONAL_MESSAGE_PASSING_STACK_MODULE_SCHEMA_VERSION,
    FUNCTIONAL_MESSAGE_PASSING_STACK_OUTPUT_SCHEMA as STACK_MODULE_OUTPUT_SCHEMA,
    FUNCTIONAL_MESSAGE_PASSING_STACK_PREDICTION_OWNED_HERE,
    FUNCTIONAL_MESSAGE_PASSING_STACK_REGULARIZATION_REDUCTION,
    FUNCTIONAL_MESSAGE_PASSING_STACK_STACKING_OWNED_HERE,
    FUNCTIONAL_MESSAGE_PASSING_STACK_STATE_REBINDING_CASTS_TENSOR,
    FUNCTIONAL_MESSAGE_PASSING_STACK_STATE_REBINDING_CLONES_TENSOR,
    FUNCTIONAL_MESSAGE_PASSING_STACK_STATE_REBINDING_DETACHES_TENSOR,
    FUNCTIONAL_MESSAGE_PASSING_STACK_STATE_REBINDING_MOVES_TENSOR,
    FunctionalMessagePassingStack,
    FunctionalStack,
    HazardConditionedFunctionalMessagePassingStack,
    MessagePassingStack,
    build_functional_message_passing_stack,
    build_functional_message_passing_stack_from_factory,
    build_stack,
    build_stack_from_factory,
    derive_next_functional_message_passing_inputs,
    derive_next_layer_inputs,
    run_functional_message_passing_stack,
    run_functional_message_passing_stack_complete,
    run_stack,
    run_stack_complete,
    validate_rebound_functional_message_passing_inputs,
    validate_rebound_inputs,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.functional_message_passing.stack.trace_policy import (
    StackRetentionPolicy,
    StackTracePolicy,
)


NODES: Final[int] = 5
EDGES: Final[int] = 6
GRAPHS: Final[int] = 2
HIDDEN_DIM: Final[int] = 4
RELATIONS: Final[int] = 3
NUM_LAYERS: Final[int] = 3

RELATION_NAMES: Final[tuple[str, ...]] = (
    "spatial_adjacency",
    "temporal_lag",
    "random_placebo",
)
STABLE_RELATION_IDS: Final[tuple[int, ...]] = (
    100,
    200,
    900,
)
REGISTRY_FINGERPRINT: Final[str] = "compiled-registry"

FMP_NODE_STATE_SOURCE_LAYER_OUTPUT: Final[str] = "layer_output"


# =============================================================================
# Controlled upstream contracts
# =============================================================================


def _canonical_fingerprint(
    payload: Mapping[str, Any],
) -> str:
    return sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _tensor_fingerprint(
    value: torch.Tensor,
) -> str:
    tensor = (
        value
        .detach()
        .cpu()
        .contiguous()
    )
    digest = sha256()
    digest.update(
        str(tensor.dtype).encode(
            "utf-8"
        )
    )
    digest.update(
        json.dumps(
            list(tensor.shape),
            separators=(",", ":"),
        ).encode("utf-8")
    )
    digest.update(
        tensor
        .view(torch.uint8)
        .numpy()
        .tobytes()
    )
    return digest.hexdigest()


@dataclass(frozen=True)
class ControlledAlignment:
    item_ids: tuple[str, ...]
    node_batch_index: torch.Tensor
    graph_count: int

    @property
    def item_count(self) -> int:
        return len(
            self.item_ids
        )

    def fingerprint(self) -> str:
        return _canonical_fingerprint(
            {
                "item_ids": list(
                    self.item_ids
                ),
                "node_batch_index": (
                    self
                    .node_batch_index
                    .detach()
                    .cpu()
                    .tolist()
                ),
                "graph_count": (
                    self.graph_count
                ),
            }
        )


@dataclass(frozen=True)
class ControlledInitialNodeState:
    fused_state: torch.Tensor
    alignment: ControlledAlignment
    encoder_architecture_fingerprint: str = (
        "fusion-architecture"
    )
    lineage_fingerprint: str = (
        "fusion-lineage"
    )

    @property
    def item_count(self) -> int:
        return int(
            self.fused_state.shape[0]
        )

    @property
    def output_dim(self) -> int:
        return int(
            self.fused_state.shape[1]
        )


@dataclass(frozen=True)
class ControlledFunctionalNodeState:
    state: torch.Tensor
    alignment: ControlledAlignment

    source_kind: str
    source_layer_index: int

    source_architecture_fingerprint: str
    source_lineage_fingerprint: str
    source_parameter_fingerprint: str | None = None

    schema_version: str = "0.1"

    def __post_init__(self) -> None:
        if self.source_kind != (
            FMP_NODE_STATE_SOURCE_LAYER_OUTPUT
        ):
            raise ValueError(
                "source_kind must be layer_output."
            )
        if self.source_layer_index < 0:
            raise ValueError(
                "source_layer_index must be nonnegative."
            )
        if self.state.ndim != 2:
            raise ValueError(
                "state must have rank 2."
            )
        if self.alignment.item_count != (
            int(self.state.shape[0])
        ):
            raise ValueError(
                "alignment row count mismatch."
            )

    @property
    def fused_state(self) -> torch.Tensor:
        return self.state

    @property
    def item_count(self) -> int:
        return int(
            self.state.shape[0]
        )

    @property
    def output_dim(self) -> int:
        return int(
            self.state.shape[1]
        )

    @property
    def encoder_architecture_fingerprint(
        self,
    ) -> str:
        return (
            self
            .source_architecture_fingerprint
        )

    @property
    def lineage_fingerprint(self) -> str:
        return _canonical_fingerprint(
            {
                "source_kind": (
                    self.source_kind
                ),
                "source_layer_index": (
                    self.source_layer_index
                ),
                "source_architecture_fingerprint": (
                    self
                    .source_architecture_fingerprint
                ),
                "source_parameter_fingerprint": (
                    self
                    .source_parameter_fingerprint
                ),
                "source_lineage_fingerprint": (
                    self
                    .source_lineage_fingerprint
                ),
                "alignment_fingerprint": (
                    self
                    .alignment
                    .fingerprint()
                ),
            }
        )


class ControlledCompiledRegistry:
    def __init__(
        self,
        *,
        fingerprint: str = (
            REGISTRY_FINGERPRINT
        ),
    ) -> None:
        self._fingerprint = fingerprint

    def fingerprint(self) -> str:
        return self._fingerprint


class ControlledGraph:
    def __init__(
        self,
        *,
        device: torch.device | str = "cpu",
        empty_edges: bool = False,
    ) -> None:
        resolved_device = torch.device(
            device
        )

        self.external_node_ids = tuple(
            f"node-{index}"
            for index in range(
                NODES
            )
        )
        self.node_batch_index = torch.tensor(
            [0, 0, 0, 1, 1],
            dtype=torch.long,
            device=resolved_device,
        )

        if empty_edges:
            self.edge_index = torch.empty(
                (2, 0),
                dtype=torch.long,
                device=resolved_device,
            )
            self.edge_relation_type = (
                torch.empty(
                    (0,),
                    dtype=torch.long,
                    device=resolved_device,
                )
            )
        else:
            self.edge_index = torch.tensor(
                [
                    [0, 1, 2, 3, 4, 3],
                    [1, 2, 0, 4, 3, 3],
                ],
                dtype=torch.long,
                device=resolved_device,
            )
            self.edge_relation_type = (
                torch.tensor(
                    [0, 1, 0, 2, 1, 0],
                    dtype=torch.long,
                    device=resolved_device,
                )
            )

        self.edge_attributes = None
        self.semantic_edge_weight = None
        self.edge_batch_index = None
        self.allow_cross_graph_edges = False

    @property
    def num_nodes(self) -> int:
        return NODES

    @property
    def num_edges(self) -> int:
        return int(
            self.edge_index.shape[1]
        )

    @property
    def batch_size(self) -> int:
        return GRAPHS


class ControlledInputs:
    def __init__(
        self,
        *,
        source_graph: ControlledGraph,
        node_state: (
            ControlledInitialNodeState
            | ControlledFunctionalNodeState
        ),
        compiled_relation_registry: (
            ControlledCompiledRegistry
        ),
        relation_families: object | None = None,
        hazard_query: object | None = None,
        compiled_relation_priors: (
            object | None
        ) = None,
        source_fingerprint: str | None = None,
    ) -> None:
        self.source_graph = source_graph
        self.node_state = node_state
        self.compiled_relation_registry = (
            compiled_relation_registry
        )
        self.relation_families = (
            relation_families
        )
        self.hazard_query = hazard_query
        self.compiled_relation_priors = (
            compiled_relation_priors
        )
        self.source_fingerprint = (
            source_fingerprint
        )

        if (
            node_state.item_count
            != source_graph.num_nodes
        ):
            raise ValueError(
                "node-state rows differ from graph nodes."
            )

        if (
            node_state.alignment.item_ids
            != source_graph.external_node_ids
        ):
            raise ValueError(
                "alignment item IDs differ."
            )

        if not torch.equal(
            node_state
            .alignment
            .node_batch_index,
            source_graph
            .node_batch_index,
        ):
            raise ValueError(
                "alignment graph membership differs."
            )

    @property
    def num_nodes(self) -> int:
        return self.source_graph.num_nodes

    @property
    def num_edges(self) -> int:
        return self.source_graph.num_edges

    @property
    def num_graphs(self) -> int:
        return self.source_graph.batch_size

    @property
    def hidden_dim(self) -> int:
        return self.node_state.output_dim

    @property
    def dtype(self) -> torch.dtype:
        return (
            self
            .node_state
            .fused_state
            .dtype
        )

    @property
    def device(self) -> torch.device:
        return (
            self
            .node_state
            .fused_state
            .device
        )

    @property
    def relation_names(
        self,
    ) -> tuple[str, ...]:
        return RELATION_NAMES

    @property
    def stable_relation_ids(
        self,
    ) -> tuple[int, ...]:
        return STABLE_RELATION_IDS

    @property
    def num_relations(self) -> int:
        return RELATIONS

    @property
    def node_batch_index(
        self,
    ) -> torch.Tensor:
        return (
            self
            .source_graph
            .node_batch_index
        )

    def lineage_fingerprint(
        self,
    ) -> str:
        node_lineage = getattr(
            self.node_state,
            "lineage_fingerprint",
        )
        if callable(
            node_lineage
        ):
            node_lineage = (
                node_lineage()
            )

        return _canonical_fingerprint(
            {
                "node_state_lineage": (
                    node_lineage
                ),
                "registry": (
                    self
                    .compiled_relation_registry
                    .fingerprint()
                ),
                "source_fingerprint": (
                    self.source_fingerprint
                ),
                "relation_names": list(
                    self.relation_names
                ),
                "stable_relation_ids": list(
                    self.stable_relation_ids
                ),
            }
        )


@dataclass(frozen=True)
class ControlledLayerInputs:
    source_inputs: ControlledInputs
    layer_index: int
    trace_policy: LayerTracePolicy
    training: bool
    source_stack_fingerprint: str | None

    @property
    def input_node_state(
        self,
    ) -> torch.Tensor:
        return (
            self
            .source_inputs
            .node_state
            .fused_state
        )


@dataclass(frozen=True)
class ControlledLayerOutput:
    updated_node_state: torch.Tensor
    node_aggregate: torch.Tensor
    incoming_edge_count: torch.Tensor
    source_inputs: ControlledInputs

    layer_index: int
    residual_enabled: bool
    layer_norm_enabled: bool

    encoder_architecture_fingerprint: str
    lineage_fingerprint: str

    intermediates: object | None
    regularization_terms: Mapping[
        str,
        torch.Tensor
    ]


@dataclass(frozen=True)
class ControlledInternalOutput:
    updated_node_state: torch.Tensor
    layer_inputs: ControlledLayerInputs

    layer_architecture_fingerprint: str
    layer_parameter_fingerprint: str
    lineage_fingerprint: str

    regularization_terms: Mapping[
        str,
        torch.Tensor
    ]


@dataclass(frozen=True)
class ControlledLayerRun:
    layer_inputs: ControlledLayerInputs
    internal_output: ControlledInternalOutput
    public_output: ControlledLayerOutput

    edge_stages: object | None = None
    node_stages: object | None = None

    @property
    def source_inputs(
        self,
    ) -> ControlledInputs:
        return self.layer_inputs.source_inputs

    @property
    def layer_index(self) -> int:
        return self.layer_inputs.layer_index

    @property
    def updated_node_state(
        self,
    ) -> torch.Tensor:
        return (
            self
            .public_output
            .updated_node_state
        )


def _validate_controlled_layer_run(
    run: ControlledLayerRun,
) -> None:
    if not isinstance(
        run,
        ControlledLayerRun,
    ):
        raise TypeError(
            "run must be a ControlledLayerRun."
        )

    if (
        run.public_output.source_inputs
        is not run.source_inputs
    ):
        raise ValueError(
            "public output lost source-input identity."
        )

    if (
        run.internal_output.layer_inputs
        is not run.layer_inputs
    ):
        raise ValueError(
            "internal output lost layer-input identity."
        )

    if (
        run.internal_output.updated_node_state
        is not run.public_output.updated_node_state
    ):
        raise ValueError(
            "internal/public updated-state identity differs."
        )

    if (
        run.public_output.layer_index
        != run.layer_index
    ):
        raise ValueError(
            "public output layer index differs."
        )

    if (
        run.public_output
        .encoder_architecture_fingerprint
        != run.internal_output
        .layer_architecture_fingerprint
    ):
        raise ValueError(
            "layer architecture fingerprints differ."
        )

    if tuple(
        run.updated_node_state.shape
    ) != (
        run.source_inputs.num_nodes,
        run.source_inputs.hidden_dim,
    ):
        raise ValueError(
            "updated state shape differs."
        )

    if run.updated_node_state.dtype != (
        run.source_inputs.dtype
    ):
        raise ValueError(
            "updated state dtype differs."
        )

    if run.updated_node_state.device != (
        run.source_inputs.device
    ):
        raise ValueError(
            "updated state device differs."
        )


class ControlledFunctionalMessagePassingLayer(
    nn.Module
):
    """
    Small differentiable layer exposing the production stack-facing API.
    """

    def __init__(
        self,
        *,
        tag: str,
        hidden_dim: int = HIDDEN_DIM,
        relation_names: tuple[str, ...] = (
            RELATION_NAMES
        ),
        stable_relation_ids: tuple[int, ...] = (
            STABLE_RELATION_IDS
        ),
        registry_fingerprint: str = (
            REGISTRY_FINGERPRINT
        ),
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
        dropout_probability: float = 0.0,
        fill_value: float = 0.05,
        architecture_override: str | None = None,
        parameter_override: str | None = None,
        emitted_layer_index_offset: int = 0,
        replace_source_inputs: bool = False,
        emit_nonfinite_state: bool = False,
    ) -> None:
        super().__init__()

        self.tag = tag
        self.hidden_dim = hidden_dim
        self.relation_names = relation_names
        self.stable_relation_ids = (
            stable_relation_ids
        )
        self.compiled_relation_registry_fingerprint = (
            registry_fingerprint
        )
        self.dropout_probability = (
            dropout_probability
        )
        self._architecture_override = (
            architecture_override
        )
        self._parameter_override = (
            parameter_override
        )
        self.emitted_layer_index_offset = (
            emitted_layer_index_offset
        )
        self.replace_source_inputs = (
            replace_source_inputs
        )
        self.emit_nonfinite_state = (
            emit_nonfinite_state
        )

        resolved_device = torch.device(
            device
        )

        self.weight = nn.Parameter(
            torch.eye(
                hidden_dim,
                dtype=dtype,
                device=resolved_device,
            )
            * fill_value
        )
        self.bias = nn.Parameter(
            torch.full(
                (hidden_dim,),
                fill_value,
                dtype=dtype,
                device=resolved_device,
            )
        )
        self.dropout = nn.Dropout(
            p=dropout_probability
        )

        self.call_count = 0
        self.observed_layer_indices: list[int] = []
        self.observed_source_inputs: list[
            ControlledInputs
        ] = []
        self.observed_trace_modes: list[str] = []
        self.observed_stack_fingerprints: list[
            str | None
        ] = []

    def architecture_fingerprint(
        self,
    ) -> str:
        if self._architecture_override is not None:
            return self._architecture_override

        return _canonical_fingerprint(
            {
                "hidden_dim": (
                    self.hidden_dim
                ),
                "relation_names": list(
                    self.relation_names
                ),
                "stable_relation_ids": list(
                    self.stable_relation_ids
                ),
                "registry_fingerprint": (
                    self
                    .compiled_relation_registry_fingerprint
                ),
                "dropout_probability": (
                    self.dropout_probability
                ),
                "update": (
                    "x_plus_dropout_linear_x"
                ),
            }
        )

    def parameter_fingerprint(
        self,
    ) -> str:
        if self._parameter_override is not None:
            return self._parameter_override

        return _canonical_fingerprint(
            {
                "weight": (
                    _tensor_fingerprint(
                        self.weight
                    )
                ),
                "bias": (
                    _tensor_fingerprint(
                        self.bias
                    )
                ),
            }
        )

    def assert_finite_parameters(
        self,
    ) -> None:
        for name, parameter in (
            self.named_parameters()
        ):
            if not bool(
                torch.isfinite(
                    parameter
                ).all().item()
            ):
                raise FloatingPointError(
                    f"{name} contains non-finite values."
                )

    def run_complete(
        self,
        source_inputs: ControlledInputs,
        *,
        layer_index: int = 0,
        trace_policy: (
            LayerTracePolicy
            | str
            | None
        ) = None,
        capture_intermediate_messages: (
            bool | None
        ) = None,
        source_stack_fingerprint: (
            str | None
        ) = None,
    ) -> ControlledLayerRun:
        del capture_intermediate_messages

        if trace_policy is None:
            policy = LayerTracePolicy(
                mode=LAYER_TRACE_NONE
            )
        elif isinstance(
            trace_policy,
            LayerTracePolicy,
        ):
            policy = trace_policy
        elif isinstance(
            trace_policy,
            str,
        ):
            policy = LayerTracePolicy(
                mode=trace_policy
            )
        else:
            raise TypeError(
                "trace_policy must be a LayerTracePolicy, string, or None."
            )

        policy.assert_implemented()

        self.call_count += 1
        self.observed_layer_indices.append(
            layer_index
        )
        self.observed_source_inputs.append(
            source_inputs
        )
        self.observed_trace_modes.append(
            policy.mode
        )
        self.observed_stack_fingerprints.append(
            source_stack_fingerprint
        )

        input_state = (
            source_inputs
            .node_state
            .fused_state
        )
        projected = F.linear(
            input_state,
            self.weight,
            self.bias,
        )
        update = self.dropout(
            torch.tanh(
                projected
            )
        )
        updated = (
            input_state
            + update
        )

        if self.emit_nonfinite_state:
            updated = (
                updated
                * torch.tensor(
                    float("nan"),
                    dtype=updated.dtype,
                    device=updated.device,
                )
            )

        emitted_index = (
            layer_index
            + self.emitted_layer_index_offset
        )

        architecture = (
            self.architecture_fingerprint()
        )
        parameter = (
            self.parameter_fingerprint()
        )
        lineage = _canonical_fingerprint(
            {
                "source_inputs": (
                    source_inputs
                    .lineage_fingerprint()
                ),
                "layer_index": (
                    emitted_index
                ),
                "architecture": (
                    architecture
                ),
                "parameter": (
                    parameter
                ),
                "trace_mode": (
                    policy.mode
                ),
                "source_stack_fingerprint": (
                    source_stack_fingerprint
                ),
                "updated_state": (
                    _tensor_fingerprint(
                        updated
                    )
                ),
            }
        )

        regularization = {
            "weight_l2": (
                self.weight
                .square()
                .mean()
            )
        }

        layer_inputs = (
            ControlledLayerInputs(
                source_inputs=(
                    source_inputs
                ),
                layer_index=(
                    emitted_index
                ),
                trace_policy=policy,
                training=self.training,
                source_stack_fingerprint=(
                    source_stack_fingerprint
                ),
            )
        )

        output_source = source_inputs

        if self.replace_source_inputs:
            output_source = ControlledInputs(
                source_graph=(
                    source_inputs
                    .source_graph
                ),
                node_state=(
                    source_inputs
                    .node_state
                ),
                compiled_relation_registry=(
                    source_inputs
                    .compiled_relation_registry
                ),
                relation_families=(
                    source_inputs
                    .relation_families
                ),
                hazard_query=(
                    source_inputs
                    .hazard_query
                ),
                compiled_relation_priors=(
                    source_inputs
                    .compiled_relation_priors
                ),
                source_fingerprint=(
                    source_inputs
                    .source_fingerprint
                ),
            )

        public = ControlledLayerOutput(
            updated_node_state=updated,
            node_aggregate=update,
            incoming_edge_count=torch.ones(
                source_inputs.num_nodes,
                dtype=torch.long,
                device=(
                    source_inputs.device
                ),
            ),
            source_inputs=output_source,
            layer_index=emitted_index,
            residual_enabled=True,
            layer_norm_enabled=False,
            encoder_architecture_fingerprint=(
                architecture
            ),
            lineage_fingerprint=lineage,
            intermediates=(
                None
                if policy.mode
                == LAYER_TRACE_NONE
                else {
                    "mode": policy.mode,
                    "node_update": update,
                }
            ),
            regularization_terms=(
                regularization
            ),
        )
        internal = ControlledInternalOutput(
            updated_node_state=updated,
            layer_inputs=layer_inputs,
            layer_architecture_fingerprint=(
                architecture
            ),
            layer_parameter_fingerprint=(
                parameter
            ),
            lineage_fingerprint=(
                lineage
            ),
            regularization_terms=(
                regularization
            ),
        )
        run = ControlledLayerRun(
            layer_inputs=layer_inputs,
            internal_output=internal,
            public_output=public,
        )

        _validate_controlled_layer_run(
            run
        )
        return run


# =============================================================================
# Patching and construction helpers
# =============================================================================


@pytest.fixture(autouse=True)
def _patch_stack_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sharing_module,
        "FunctionalMessagePassingLayer",
        ControlledFunctionalMessagePassingLayer,
    )
    monkeypatch.setattr(
        stack_module,
        "FunctionalMessagePassingLayer",
        ControlledFunctionalMessagePassingLayer,
    )

    monkeypatch.setattr(
        stack_module,
        "FunctionalMessagePassingInputs",
        ControlledInputs,
    )
    monkeypatch.setattr(
        stack_schemas_module,
        "FunctionalMessagePassingInputs",
        ControlledInputs,
    )

    monkeypatch.setattr(
        stack_module,
        "FunctionalMessagePassingNodeState",
        ControlledFunctionalNodeState,
    )
    monkeypatch.setattr(
        stack_schemas_module,
        "FunctionalMessagePassingNodeState",
        ControlledFunctionalNodeState,
    )

    monkeypatch.setattr(
        stack_module,
        "FMP_NODE_STATE_SOURCE_LAYER_OUTPUT",
        FMP_NODE_STATE_SOURCE_LAYER_OUTPUT,
    )
    monkeypatch.setattr(
        stack_schemas_module,
        "FMP_NODE_STATE_SOURCE_LAYER_OUTPUT",
        FMP_NODE_STATE_SOURCE_LAYER_OUTPUT,
    )

    monkeypatch.setattr(
        stack_module,
        "FunctionalMessagePassingLayerRun",
        ControlledLayerRun,
    )
    monkeypatch.setattr(
        stack_schemas_module,
        "FunctionalMessagePassingLayerRun",
        ControlledLayerRun,
    )
    monkeypatch.setattr(
        stack_schemas_module,
        "FunctionalMessagePassingLayerOutput",
        ControlledLayerOutput,
    )

    monkeypatch.setattr(
        stack_module,
        "validate_functional_message_passing_layer_run",
        _validate_controlled_layer_run,
    )
    monkeypatch.setattr(
        stack_schemas_module,
        "validate_functional_message_passing_layer_run",
        _validate_controlled_layer_run,
    )



def _inputs(
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
    empty_edges: bool = False,
    requires_grad: bool = False,
    registry_fingerprint: str = (
        REGISTRY_FINGERPRINT
    ),
    source_fingerprint: str = (
        "stack-test-input"
    ),
) -> ControlledInputs:
    resolved_device = torch.device(
        device
    )
    graph = ControlledGraph(
        device=resolved_device,
        empty_edges=empty_edges,
    )
    values = (
        torch.arange(
            NODES * HIDDEN_DIM,
            dtype=dtype,
            device=resolved_device,
        )
        .reshape(
            NODES,
            HIDDEN_DIM,
        )
        / 10.0
    )

    if requires_grad:
        values.requires_grad_()

    alignment = ControlledAlignment(
        item_ids=(
            graph.external_node_ids
        ),
        node_batch_index=(
            graph.node_batch_index
        ),
        graph_count=GRAPHS,
    )
    node_state = ControlledInitialNodeState(
        fused_state=values,
        alignment=alignment,
    )

    return ControlledInputs(
        source_graph=graph,
        node_state=node_state,
        compiled_relation_registry=(
            ControlledCompiledRegistry(
                fingerprint=(
                    registry_fingerprint
                )
            )
        ),
        relation_families=object(),
        hazard_query=object(),
        compiled_relation_priors=object(),
        source_fingerprint=(
            source_fingerprint
        ),
    )


def _layer(
    index: int,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
    dropout_probability: float = 0.0,
    fill_value: float | None = None,
    hidden_dim: int = HIDDEN_DIM,
    relation_names: tuple[str, ...] = (
        RELATION_NAMES
    ),
    stable_relation_ids: tuple[int, ...] = (
        STABLE_RELATION_IDS
    ),
    registry_fingerprint: str = (
        REGISTRY_FINGERPRINT
    ),
    emitted_layer_index_offset: int = 0,
    replace_source_inputs: bool = False,
    emit_nonfinite_state: bool = False,
) -> ControlledFunctionalMessagePassingLayer:
    return ControlledFunctionalMessagePassingLayer(
        tag=f"layer-{index}",
        hidden_dim=hidden_dim,
        relation_names=relation_names,
        stable_relation_ids=(
            stable_relation_ids
        ),
        registry_fingerprint=(
            registry_fingerprint
        ),
        dtype=dtype,
        device=device,
        dropout_probability=(
            dropout_probability
        ),
        fill_value=(
            0.05 * (index + 1)
            if fill_value is None
            else fill_value
        ),
        emitted_layer_index_offset=(
            emitted_layer_index_offset
        ),
        replace_source_inputs=(
            replace_source_inputs
        ),
        emit_nonfinite_state=(
            emit_nonfinite_state
        ),
    )


def _layers(
    count: int = NUM_LAYERS,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
    dropout_probability: float = 0.0,
    equal_initialization: bool = False,
) -> tuple[
    ControlledFunctionalMessagePassingLayer,
    ...,
]:
    return tuple(
        _layer(
            index,
            dtype=dtype,
            device=device,
            dropout_probability=(
                dropout_probability
            ),
            fill_value=(
                0.05
                if equal_initialization
                else None
            ),
        )
        for index in range(
            count
        )
    )


def _stack(
    *,
    num_layers: int = NUM_LAYERS,
    sharing_policy: str = (
        STACK_SHARING_INDEPENDENT
    ),
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
    dropout_probability: float = 0.0,
    default_trace_policy: (
        StackTracePolicy | None
    ) = None,
    diagnostics: (
        FunctionalMessagePassingStackDiagnostics
        | None
    ) = None,
) -> FunctionalMessagePassingStack:
    if sharing_policy == (
        STACK_SHARING_INDEPENDENT
    ):
        layer_value: Any = _layers(
            num_layers,
            dtype=dtype,
            device=device,
            dropout_probability=(
                dropout_probability
            ),
        )
    else:
        layer_value = _layer(
            0,
            dtype=dtype,
            device=device,
            dropout_probability=(
                dropout_probability
            ),
        )

    return FunctionalMessagePassingStack.from_layers(
        layer_value,
        num_layers=num_layers,
        sharing_policy=(
            sharing_policy
        ),
        default_trace_policy=(
            default_trace_policy
        ),
        diagnostics=diagnostics,
    )


def _assert_tensor_free(
    value: Any,
) -> None:
    if isinstance(
        value,
        torch.Tensor,
    ):
        raise AssertionError(
            "Diagnostic output contains a tensor."
        )

    if isinstance(
        value,
        nn.Module,
    ):
        raise AssertionError(
            "Diagnostic output contains a module."
        )

    if isinstance(
        value,
        Mapping,
    ):
        for key, child in value.items():
            assert isinstance(
                key,
                str,
            )
            _assert_tensor_free(
                child
            )
        return

    if isinstance(
        value,
        (
            tuple,
            list,
        ),
    ):
        for child in value:
            _assert_tensor_free(
                child
            )
        return

    assert value is None or isinstance(
        value,
        (
            str,
            bool,
            int,
            float,
        ),
    )


def _torch_load_state_dict(
    buffer: io.BytesIO,
) -> Mapping[str, torch.Tensor]:
    try:
        loaded = torch.load(
            buffer,
            weights_only=True,
        )
    except TypeError:
        loaded = torch.load(
            buffer
        )

    assert isinstance(
        loaded,
        Mapping,
    )
    return loaded


# =============================================================================
# Published identity and aliases
# =============================================================================


def test_stack_schema_versions_are_nonempty() -> None:
    versions = (
        FUNCTIONAL_MESSAGE_PASSING_STACK_SCHEMA_VERSION,
        STACK_INPUTS_SCHEMA_VERSION,
        STACK_DEPTH_RECORD_SCHEMA_VERSION,
        STACK_TRACE_SCHEMA_VERSION,
        STACK_COMPUTATION_OUTPUT_SCHEMA_VERSION,
        STACK_PUBLIC_OUTPUT_SCHEMA_VERSION,
        STACK_RUN_SCHEMA_VERSION,
        STACK_RUN_WITH_DIAGNOSTICS_SCHEMA_VERSION,
        FUNCTIONAL_MESSAGE_PASSING_STACK_MODULE_SCHEMA_VERSION,
        STACK_DIAGNOSTIC_THRESHOLDS_SCHEMA_VERSION,
        STACK_DEPTH_DIAGNOSTIC_SCHEMA_VERSION,
        STACK_DIAGNOSTIC_REPORT_SCHEMA_VERSION,
        STACK_DIAGNOSTICS_SCHEMA_VERSION,
    )

    for version in versions:
        assert isinstance(
            version,
            str,
        )
        assert version.strip()


def test_stack_scope_flags() -> None:
    assert (
        FUNCTIONAL_MESSAGE_PASSING_STACK_REIMPLEMENTS_LAYER_MATH
        is False
    )
    assert (
        FUNCTIONAL_MESSAGE_PASSING_STACK_RETENTION_AFFECTS_NUMERICS
        is False
    )
    assert (
        FUNCTIONAL_MESSAGE_PASSING_STACK_LAYER_TRACE_AFFECTS_NUMERICS
        is False
    )
    assert (
        FUNCTIONAL_MESSAGE_PASSING_STACK_DIAGNOSTICS_AFFECT_NUMERICS
        is False
    )
    assert (
        FUNCTIONAL_MESSAGE_PASSING_STACK_HIDDEN_WIDTH_CHANGES_SUPPORTED
        is False
    )
    assert (
        FUNCTIONAL_MESSAGE_PASSING_STACK_ZERO_LAYER_SUPPORTED
        is False
    )
    assert (
        FUNCTIONAL_MESSAGE_PASSING_STACK_PARTIAL_SHARING_SUPPORTED
        is False
    )

    assert (
        FUNCTIONAL_MESSAGE_PASSING_STACK_STACKING_OWNED_HERE
        is True
    )
    assert (
        FUNCTIONAL_MESSAGE_PASSING_STACK_LAYER_MATH_OWNED_HERE
        is False
    )
    assert (
        FUNCTIONAL_MESSAGE_PASSING_STACK_PREDICTION_OWNED_HERE
        is False
    )


def test_state_rebinding_flags() -> None:
    assert (
        FUNCTIONAL_MESSAGE_PASSING_STACK_STATE_REBINDING_CLONES_TENSOR
        is False
    )
    assert (
        FUNCTIONAL_MESSAGE_PASSING_STACK_STATE_REBINDING_DETACHES_TENSOR
        is False
    )
    assert (
        FUNCTIONAL_MESSAGE_PASSING_STACK_STATE_REBINDING_CASTS_TENSOR
        is False
    )
    assert (
        FUNCTIONAL_MESSAGE_PASSING_STACK_STATE_REBINDING_MOVES_TENSOR
        is False
    )
    assert (
        FUNCTIONAL_MESSAGE_PASSING_STACK_REGULARIZATION_REDUCTION
        == "preserve_one_mapping_per_depth_without_reduction"
    )
    assert (
        FUNCTIONAL_MESSAGE_PASSING_STACK_AUDIT_RETENTION
        == "complete_layer_runs_retained_only_when_audit_mode_enabled"
    )


def test_stack_identity_strings() -> None:
    assert isinstance(
        FUNCTIONAL_MESSAGE_PASSING_STACK_SCIENTIFIC_INTERPRETATION,
        str,
    )
    assert (
        FUNCTIONAL_MESSAGE_PASSING_STACK_SCIENTIFIC_INTERPRETATION
    )
    assert isinstance(
        FUNCTIONAL_MESSAGE_PASSING_STACK_OPERATION_ORDER,
        tuple,
    )
    assert len(
        FUNCTIONAL_MESSAGE_PASSING_STACK_OPERATION_ORDER
    ) >= 5
    assert (
        FUNCTIONAL_MESSAGE_PASSING_STACK_OUTPUT_SCHEMA
        == "FunctionalMessagePassingStackOutput"
    )
    assert (
        STACK_MODULE_OUTPUT_SCHEMA
        == "FunctionalMessagePassingStackOutput"
    )


def test_stack_aliases_are_exact() -> None:
    assert (
        HazardConditionedFunctionalMessagePassingStack
        is FunctionalMessagePassingStack
    )
    assert FunctionalStack is (
        FunctionalMessagePassingStack
    )
    assert MessagePassingStack is (
        FunctionalMessagePassingStack
    )
    assert build_stack is (
        build_functional_message_passing_stack
    )
    assert build_stack_from_factory is (
        build_functional_message_passing_stack_from_factory
    )
    assert run_stack is (
        run_functional_message_passing_stack
    )
    assert run_stack_complete is (
        run_functional_message_passing_stack_complete
    )
    assert derive_next_layer_inputs is (
        derive_next_functional_message_passing_inputs
    )
    assert validate_rebound_inputs is (
        validate_rebound_functional_message_passing_inputs
    )


def test_stack_schema_aliases_are_exact() -> None:
    assert StackInputs is (
        FunctionalMessagePassingStackInputs
    )
    assert StackDepthRecord is (
        FunctionalMessagePassingStackDepthRecord
    )
    assert StackTrace is (
        FunctionalMessagePassingStackTrace
    )
    assert StackComputationOutput is (
        FunctionalMessagePassingStackComputationOutput
    )
    assert StackOutput is (
        FunctionalMessagePassingStackOutput
    )
    assert StackRun is (
        FunctionalMessagePassingStackRun
    )
    assert StackRunWithDiagnostics is (
        FunctionalMessagePassingStackRunWithDiagnostics
    )
    assert assemble_stack_output is (
        assemble_functional_message_passing_stack_output
    )
    assert validate_stack_run is (
        validate_functional_message_passing_stack_run
    )


def test_diagnostic_aliases_are_exact() -> None:
    assert StackDiagnostics is (
        FunctionalMessagePassingStackDiagnostics
    )
    assert StackDiagnosticReport is (
        FunctionalMessagePassingStackDiagnosticReport
    )
    assert DepthDiagnostic is (
        StackDepthDiagnostic
    )
    assert DiagnosticThresholds is (
        StackDiagnosticThresholds
    )
    assert build_diagnostic_report is (
        build_functional_message_passing_stack_diagnostic_report
    )
    assert validate_diagnostic_report is (
        validate_functional_message_passing_stack_diagnostic_report
    )


def test_stack_modules_have_unique_bound_all_exports() -> None:
    for module in (
        stack_module,
        stack_schemas_module,
        diagnostics_module,
    ):
        exported = tuple(
            module.__all__
        )
        assert len(exported) == len(
            set(exported)
        )
        for name in exported:
            assert hasattr(
                module,
                name,
            ), (
                module.__name__,
                name,
            )


# =============================================================================
# Construction and registration
# =============================================================================


def test_independent_stack_registration() -> None:
    stack = _stack(
        sharing_policy=(
            STACK_SHARING_INDEPENDENT
        )
    )

    assert stack.num_layers == (
        NUM_LAYERS
    )
    assert stack.num_unique_layers == (
        NUM_LAYERS
    )
    assert stack.sharing_policy == (
        STACK_SHARING_INDEPENDENT
    )
    assert isinstance(
        stack.layers,
        nn.ModuleList,
    )
    assert not hasattr(
        stack,
        "shared_layer",
    )
    assert (
        stack.depth_to_unique_layer_index
        == (0, 1, 2)
    )

    for depth in range(
        NUM_LAYERS
    ):
        assert (
            stack.layer_for_depth(
                depth
            )
            is stack.layers[depth]
        )


def test_fully_shared_stack_registration() -> None:
    stack = _stack(
        sharing_policy=(
            STACK_SHARING_FULLY_SHARED
        )
    )

    assert stack.num_layers == (
        NUM_LAYERS
    )
    assert stack.num_unique_layers == 1
    assert stack.sharing_policy == (
        STACK_SHARING_FULLY_SHARED
    )
    assert hasattr(
        stack,
        "shared_layer",
    )
    assert not hasattr(
        stack,
        "layers",
    )
    assert (
        stack.depth_to_unique_layer_index
        == (0, 0, 0)
    )

    for depth in range(
        NUM_LAYERS
    ):
        assert (
            stack.layer_for_depth(
                depth
            )
            is stack.shared_layer
        )


def test_independent_state_dict_registration_names() -> None:
    stack = _stack(
        sharing_policy=(
            STACK_SHARING_INDEPENDENT
        )
    )

    keys = tuple(
        stack.state_dict()
    )

    assert keys == (
        "layers.0.weight",
        "layers.0.bias",
        "layers.1.weight",
        "layers.1.bias",
        "layers.2.weight",
        "layers.2.bias",
    )
    assert (
        FUNCTIONAL_MESSAGE_PASSING_STACK_INDEPENDENT_REGISTRATION_PREFIX
        == "layers"
    )


def test_fully_shared_state_dict_registration_names() -> None:
    stack = _stack(
        sharing_policy=(
            STACK_SHARING_FULLY_SHARED
        )
    )

    keys = tuple(
        stack.state_dict()
    )

    assert keys == (
        "shared_layer.weight",
        "shared_layer.bias",
    )
    assert (
        FUNCTIONAL_MESSAGE_PASSING_STACK_FULLY_SHARED_REGISTRATION_PREFIX
        == "shared_layer"
    )


def test_stack_public_metadata() -> None:
    stack = _stack()

    assert stack.hidden_dim == HIDDEN_DIM
    assert stack.relation_names == (
        RELATION_NAMES
    )
    assert stack.stable_relation_ids == (
        STABLE_RELATION_IDS
    )
    assert stack.num_relations == (
        RELATIONS
    )
    assert (
        stack.compiled_relation_registry_fingerprint
        == REGISTRY_FINGERPRINT
    )
    assert stack.parameter_count == (
        NUM_LAYERS
        * (
            HIDDEN_DIM * HIDDEN_DIM
            + HIDDEN_DIM
        )
    )
    assert (
        stack.trainable_parameter_count
        == stack.parameter_count
    )
    assert stack.buffer_count == 0
    assert stack.diagnostics_enabled is False


def test_shared_stack_parameter_count_counts_unique_parameters_once() -> None:
    independent = _stack(
        sharing_policy=(
            STACK_SHARING_INDEPENDENT
        )
    )
    shared = _stack(
        sharing_policy=(
            STACK_SHARING_FULLY_SHARED
        )
    )

    assert independent.parameter_count == (
        NUM_LAYERS
        * shared.parameter_count
    )


def test_stack_extra_repr() -> None:
    stack = _stack(
        sharing_policy=(
            STACK_SHARING_FULLY_SHARED
        ),
        default_trace_policy=(
            StackTracePolicy.full_audit()
        ),
        diagnostics=(
            FunctionalMessagePassingStackDiagnostics()
        ),
    )

    text = stack.extra_repr()

    assert "num_layers=3" in text
    assert "num_unique_layers=1" in text
    assert "fully_shared" in text
    assert "full" in text
    assert "diagnostics_enabled=True" in text
    assert "prediction_owned_here=False" in text


def test_stack_rejects_wrong_sharing_plan_type() -> None:
    with pytest.raises(
        TypeError,
        match="StackLayerSharingPlan",
    ):
        FunctionalMessagePassingStack(
            sharing_plan=object(),  # type: ignore[arg-type]
        )


def test_stack_rejects_wrong_trace_policy_type() -> None:
    plan = StackLayerSharingPlan(
        policy=(
            StackSharingPolicy.independent()
        ),
        num_layers=1,
        layers_by_depth=(
            _layer(0),
        ),
    )

    with pytest.raises(
        TypeError,
        match="StackTracePolicy",
    ):
        FunctionalMessagePassingStack(
            sharing_plan=plan,
            default_trace_policy=object(),  # type: ignore[arg-type]
        )


def test_stack_rejects_wrong_diagnostics_type() -> None:
    plan = StackLayerSharingPlan(
        policy=(
            StackSharingPolicy.independent()
        ),
        num_layers=1,
        layers_by_depth=(
            _layer(0),
        ),
    )

    with pytest.raises(
        TypeError,
        match="Diagnostics",
    ):
        FunctionalMessagePassingStack(
            sharing_plan=plan,
            diagnostics=object(),  # type: ignore[arg-type]
        )


def test_stack_layer_for_depth_rejects_invalid_values() -> None:
    stack = _stack()

    with pytest.raises(
        ValueError,
        match="nonnegative",
    ):
        stack.layer_for_depth(
            -1
        )

    with pytest.raises(
        TypeError,
        match="integer",
    ):
        stack.layer_for_depth(
            True  # type: ignore[arg-type]
        )

    with pytest.raises(
        IndexError,
        match="outside",
    ):
        stack.layer_for_depth(
            NUM_LAYERS
        )


# =============================================================================
# Architecture and parameter fingerprints
# =============================================================================


def test_stack_architecture_fingerprint_is_deterministic() -> None:
    first = _stack()
    second = _stack()

    assert (
        first.numerical_architecture_dict()
        == second.numerical_architecture_dict()
    )
    assert (
        first.architecture_dict()
        == second.architecture_dict()
    )
    assert (
        first.architecture_fingerprint()
        == second.architecture_fingerprint()
    )


def test_stack_architecture_fingerprint_changes_with_sharing() -> None:
    independent = _stack(
        sharing_policy=(
            STACK_SHARING_INDEPENDENT
        )
    )
    shared = _stack(
        sharing_policy=(
            STACK_SHARING_FULLY_SHARED
        )
    )

    assert (
        independent.architecture_fingerprint()
        != shared.architecture_fingerprint()
    )


def test_stack_parameter_fingerprint_is_deterministic() -> None:
    first = _stack()
    second = _stack()

    assert (
        first.parameter_fingerprint()
        == second.parameter_fingerprint()
    )


def test_stack_parameter_fingerprint_changes_after_update() -> None:
    stack = _stack()
    before = stack.parameter_fingerprint()

    with torch.no_grad():
        stack.layers[1].weight.add_(
            1.0
        )

    after = stack.parameter_fingerprint()

    assert after != before


def test_stack_runtime_dict_separates_observability() -> None:
    stack = _stack(
        default_trace_policy=(
            StackTracePolicy.full_audit()
        ),
        diagnostics=(
            FunctionalMessagePassingStackDiagnostics()
        ),
    )

    runtime = stack.runtime_dict()

    assert runtime[
        "architecture_fingerprint"
    ] == stack.architecture_fingerprint()
    assert runtime[
        "parameter_fingerprint"
    ] == stack.parameter_fingerprint()
    assert runtime[
        "default_trace_policy"
    ] == (
        stack.default_trace_policy
        .execution_contract_dict()
    )
    assert runtime[
        "diagnostics_enabled"
    ] is True
    assert runtime[
        "diagnostics_architecture"
    ] == (
        stack.diagnostics
        .architecture_dict()
    )


# =============================================================================
# Source-input validation
# =============================================================================


def test_stack_rejects_wrong_source_input_type() -> None:
    stack = _stack()

    with pytest.raises(
        TypeError,
        match="FunctionalMessagePassingInputs",
    ):
        stack(
            object()  # type: ignore[arg-type]
        )


def test_stack_rejects_hidden_width_mismatch() -> None:
    stack = _stack()
    inputs = _inputs()
    wrong_values = torch.zeros(
        NODES,
        HIDDEN_DIM + 1,
    )
    wrong_state = ControlledInitialNodeState(
        fused_state=wrong_values,
        alignment=(
            inputs
            .node_state
            .alignment
        ),
    )
    wrong_inputs = ControlledInputs(
        source_graph=(
            inputs.source_graph
        ),
        node_state=wrong_state,
        compiled_relation_registry=(
            inputs
            .compiled_relation_registry
        ),
        relation_families=(
            inputs.relation_families
        ),
        hazard_query=(
            inputs.hazard_query
        ),
        compiled_relation_priors=(
            inputs.compiled_relation_priors
        ),
    )

    with pytest.raises(
        ValueError,
        match="hidden width",
    ):
        stack(
            wrong_inputs
        )


def test_stack_rejects_registry_fingerprint_mismatch() -> None:
    stack = _stack()
    inputs = _inputs(
        registry_fingerprint=(
            "different-registry"
        )
    )

    with pytest.raises(
        ValueError,
        match="compiled relation registry",
    ):
        stack(
            inputs
        )


def test_stack_rejects_relation_order_mismatch() -> None:
    layers = (
        _layer(
            0,
            relation_names=(
                "temporal_lag",
                "spatial_adjacency",
                "random_placebo",
            ),
        ),
    )
    stack = (
        FunctionalMessagePassingStack.from_layers(
            layers,
            num_layers=1,
            sharing_policy=(
                STACK_SHARING_INDEPENDENT
            ),
        )
    )

    with pytest.raises(
        ValueError,
        match="relation ordering",
    ):
        stack(
            _inputs()
        )


def test_stack_rejects_stable_relation_id_mismatch() -> None:
    layers = (
        _layer(
            0,
            stable_relation_ids=(
                100,
                201,
                900,
            ),
        ),
    )
    stack = (
        FunctionalMessagePassingStack.from_layers(
            layers,
            num_layers=1,
            sharing_policy=(
                STACK_SHARING_INDEPENDENT
            ),
        )
    )

    with pytest.raises(
        ValueError,
        match="stable relation IDs",
    ):
        stack(
            _inputs()
        )


# =============================================================================
# Stack inputs and execution contracts
# =============================================================================


def test_build_stack_inputs_valid_contract() -> None:
    stack = _stack()
    inputs = _inputs()
    policy = StackTracePolicy(
        retention_policy=(
            STACK_RETENTION_FINAL_LAYER
        ),
        layer_trace_policy=(
            LAYER_TRACE_NODE
        ),
        audit_mode=True,
    )

    stack_inputs = (
        stack.build_stack_inputs(
            inputs,
            trace_policy=policy,
            source_model_fingerprint=(
                "source-model"
            ),
        )
    )

    assert isinstance(
        stack_inputs,
        FunctionalMessagePassingStackInputs,
    )
    assert stack_inputs.source_inputs is (
        inputs
    )
    assert stack_inputs.num_layers == (
        NUM_LAYERS
    )
    assert stack_inputs.sharing_policy == (
        STACK_SHARING_INDEPENDENT
    )
    assert stack_inputs.retention_policy == (
        STACK_RETENTION_FINAL_LAYER
    )
    assert stack_inputs.layer_trace_mode == (
        LAYER_TRACE_NODE
    )
    assert stack_inputs.audit_mode is True
    assert stack_inputs.training is (
        stack.training
    )
    assert (
        stack_inputs.source_model_fingerprint
        == "source-model"
    )
    assert (
        stack_inputs.expected_retained_layer_indices
        == (2,)
    )
    assert (
        stack_inputs.execution_contract_fingerprint()
        == stack_inputs.execution_contract_fingerprint()
    )
    assert (
        stack_inputs.lineage_fingerprint()
        == stack_inputs.lineage_fingerprint()
    )


def test_build_stack_inputs_rejects_blank_source_model_fingerprint() -> None:
    stack = _stack()

    with pytest.raises(
        ValueError,
        match="source_model_fingerprint",
    ):
        stack.build_stack_inputs(
            _inputs(),
            source_model_fingerprint="",
        )


def test_run_from_stack_inputs_rejects_depth_mismatch() -> None:
    stack = _stack()
    wrong = FunctionalMessagePassingStackInputs(
        source_inputs=_inputs(),
        num_layers=2,
        sharing_policy=(
            STACK_SHARING_INDEPENDENT
        ),
        retention_policy=(
            STACK_RETENTION_NONE
        ),
        layer_trace_policy=(
            LAYER_TRACE_NONE
        ),
        training=stack.training,
    )

    with pytest.raises(
        ValueError,
        match="num_layers differs",
    ):
        stack.run_from_stack_inputs(
            wrong
        )


def test_run_from_stack_inputs_rejects_sharing_mismatch() -> None:
    stack = _stack(
        sharing_policy=(
            STACK_SHARING_INDEPENDENT
        )
    )
    wrong = FunctionalMessagePassingStackInputs(
        source_inputs=_inputs(),
        num_layers=NUM_LAYERS,
        sharing_policy=(
            STACK_SHARING_FULLY_SHARED
        ),
        retention_policy=(
            STACK_RETENTION_NONE
        ),
        layer_trace_policy=(
            LAYER_TRACE_NONE
        ),
        training=stack.training,
    )

    with pytest.raises(
        ValueError,
        match="sharing_policy differs",
    ):
        stack.run_from_stack_inputs(
            wrong
        )


def test_run_from_stack_inputs_rejects_training_mismatch() -> None:
    stack = _stack()
    stack.train()
    wrong = FunctionalMessagePassingStackInputs(
        source_inputs=_inputs(),
        num_layers=NUM_LAYERS,
        sharing_policy=(
            STACK_SHARING_INDEPENDENT
        ),
        retention_policy=(
            STACK_RETENTION_NONE
        ),
        layer_trace_policy=(
            LAYER_TRACE_NONE
        ),
        training=False,
    )

    with pytest.raises(
        ValueError,
        match="training",
    ):
        stack.run_from_stack_inputs(
            wrong
        )


def test_explicit_trace_policy_controls_stack_inputs() -> None:
    stack = _stack()
    policy = StackTracePolicy(
        retention_policy=(
            STACK_RETENTION_ALL_LAYERS
        ),
        layer_trace_policy=(
            LAYER_TRACE_FULL
        ),
        audit_mode=True,
    )

    run = stack.run_complete(
        _inputs(),
        trace_policy=policy,
    )

    assert run.stack_inputs.retention_policy == (
        STACK_RETENTION_ALL_LAYERS
    )
    assert run.stack_inputs.layer_trace_mode == (
        LAYER_TRACE_FULL
    )
    assert run.stack_inputs.audit_mode is True


def test_runtime_overrides_can_replace_default_policy() -> None:
    stack = _stack(
        default_trace_policy=(
            StackTracePolicy.minimal()
        )
    )

    run = stack.run_complete(
        _inputs(),
        retention_policy=(
            STACK_RETENTION_ALL_LAYERS
        ),
        layer_trace_policy=(
            LAYER_TRACE_NODE
        ),
        audit_mode=True,
    )

    assert run.stack_inputs.retention_policy == (
        STACK_RETENTION_ALL_LAYERS
    )
    assert run.stack_inputs.layer_trace_mode == (
        LAYER_TRACE_NODE
    )
    assert run.stack_inputs.audit_mode is True


# =============================================================================
# One-layer and multi-layer numerical execution
# =============================================================================


def test_one_layer_stack_matches_direct_layer_execution() -> None:
    layer = _layer(0)
    stack = (
        FunctionalMessagePassingStack.from_layers(
            (layer,),
            num_layers=1,
            sharing_policy=(
                STACK_SHARING_INDEPENDENT
            ),
        )
    )
    stack.eval()
    inputs = _inputs()

    direct = layer.run_complete(
        inputs,
        layer_index=0,
        trace_policy=(
            LAYER_TRACE_NONE
        ),
        source_stack_fingerprint=(
            "direct-comparison"
        ),
    )
    stacked = stack(
        inputs,
        trace_policy=(
            StackTracePolicy.minimal()
        ),
    )

    assert torch.equal(
        stacked.final_node_state,
        direct.updated_node_state,
    )


def test_multi_layer_stack_matches_manual_depth_iteration() -> None:
    layers = _layers()
    stack = (
        FunctionalMessagePassingStack.from_layers(
            layers,
            num_layers=NUM_LAYERS,
            sharing_policy=(
                STACK_SHARING_INDEPENDENT
            ),
        )
    )
    stack.eval()
    inputs = _inputs()

    current = inputs
    manual_final: torch.Tensor | None = None

    for depth, layer in enumerate(
        layers
    ):
        layer_run = layer.run_complete(
            current,
            layer_index=depth,
            trace_policy=(
                LAYER_TRACE_NONE
            ),
            source_stack_fingerprint=(
                "manual-stack"
            ),
        )
        manual_final = (
            layer_run.updated_node_state
        )

        if depth + 1 < NUM_LAYERS:
            current = (
                derive_next_functional_message_passing_inputs(
                    original_stack_inputs=(
                        inputs
                    ),
                    previous_layer_run=(
                        layer_run
                    ),
                    source_stack_fingerprint=(
                        "manual-stack"
                    ),
                    next_layer_index=(
                        depth + 1
                    ),
                )
            )

    stacked = stack(
        inputs,
        trace_policy=(
            StackTracePolicy.minimal()
        ),
    )

    assert manual_final is not None
    assert torch.equal(
        stacked.final_node_state,
        manual_final,
    )


def test_all_layers_receive_zero_based_runtime_indices() -> None:
    stack = _stack()
    stack(
        _inputs()
    )

    assert tuple(
        layer.observed_layer_indices
        for layer in stack.layers
    ) == (
        [0],
        [1],
        [2],
    )


def test_fully_shared_layer_receives_every_runtime_index() -> None:
    stack = _stack(
        sharing_policy=(
            STACK_SHARING_FULLY_SHARED
        )
    )
    stack(
        _inputs()
    )

    assert (
        stack.shared_layer
        .observed_layer_indices
        == [0, 1, 2]
    )
    assert stack.shared_layer.call_count == (
        NUM_LAYERS
    )


def test_source_stack_fingerprint_is_constant_across_depth() -> None:
    stack = _stack(
        sharing_policy=(
            STACK_SHARING_FULLY_SHARED
        )
    )
    stack(
        _inputs()
    )

    observed = (
        stack.shared_layer
        .observed_stack_fingerprints
    )

    assert len(observed) == (
        NUM_LAYERS
    )
    assert all(
        value is not None
        for value in observed
    )
    assert len(
        set(observed)
    ) == 1


def test_empty_edge_graph_executes() -> None:
    stack = _stack()
    inputs = _inputs(
        empty_edges=True
    )

    output = stack(
        inputs
    )

    assert output.final_node_state.shape == (
        NODES,
        HIDDEN_DIM,
    )
    assert torch.isfinite(
        output.final_node_state
    ).all()


# =============================================================================
# Immutable state rebinding and audit lineage
# =============================================================================


def test_derive_next_inputs_preserves_exact_state_tensor_identity() -> None:
    layer = _layer(0)
    inputs = _inputs()
    layer_run = layer.run_complete(
        inputs,
        layer_index=0,
        source_stack_fingerprint=(
            "stack-fingerprint"
        ),
    )

    rebound = (
        derive_next_functional_message_passing_inputs(
            original_stack_inputs=(
                inputs
            ),
            previous_layer_run=(
                layer_run
            ),
            source_stack_fingerprint=(
                "stack-fingerprint"
            ),
            next_layer_index=1,
        )
    )

    assert rebound is not inputs
    assert isinstance(
        rebound.node_state,
        ControlledFunctionalNodeState,
    )
    assert (
        rebound.node_state.state
        is layer_run.updated_node_state
    )
    assert (
        rebound.node_state.fused_state
        is layer_run.updated_node_state
    )
    assert (
        rebound.node_state.state.grad_fn
        is layer_run.updated_node_state.grad_fn
    )


def test_derive_next_inputs_preserves_structural_object_identity() -> None:
    layer = _layer(0)
    inputs = _inputs()
    layer_run = layer.run_complete(
        inputs,
        layer_index=0,
        source_stack_fingerprint=(
            "stack-fingerprint"
        ),
    )
    rebound = derive_next_layer_inputs(
        original_stack_inputs=inputs,
        previous_layer_run=layer_run,
        source_stack_fingerprint=(
            "stack-fingerprint"
        ),
    )

    assert rebound.source_graph is (
        inputs.source_graph
    )
    assert (
        rebound.compiled_relation_registry
        is inputs.compiled_relation_registry
    )
    assert rebound.relation_families is (
        inputs.relation_families
    )
    assert rebound.hazard_query is (
        inputs.hazard_query
    )
    assert (
        rebound.compiled_relation_priors
        is inputs.compiled_relation_priors
    )
    assert (
        rebound.node_state.alignment
        is inputs.node_state.alignment
    )
    assert (
        rebound.source_graph.edge_index
        is inputs.source_graph.edge_index
    )
    assert (
        rebound.source_graph.edge_relation_type
        is inputs.source_graph.edge_relation_type
    )
    assert (
        rebound.node_batch_index
        is inputs.node_batch_index
    )


def test_derive_next_inputs_records_previous_layer_provenance() -> None:
    layer = _layer(0)
    inputs = _inputs()
    layer_run = layer.run_complete(
        inputs,
        layer_index=2,
        source_stack_fingerprint=(
            "stack-fingerprint"
        ),
    )
    rebound = (
        derive_next_functional_message_passing_inputs(
            original_stack_inputs=(
                inputs
            ),
            previous_layer_run=(
                layer_run
            ),
            source_stack_fingerprint=(
                "stack-fingerprint"
            ),
            next_layer_index=3,
        )
    )
    node_state = rebound.node_state

    assert node_state.source_kind == (
        FMP_NODE_STATE_SOURCE_LAYER_OUTPUT
    )
    assert node_state.source_layer_index == 2
    assert (
        node_state.source_architecture_fingerprint
        == layer_run
        .public_output
        .encoder_architecture_fingerprint
    )
    assert (
        node_state.source_parameter_fingerprint
        == layer_run
        .internal_output
        .layer_parameter_fingerprint
    )
    assert (
        node_state.source_lineage_fingerprint
        == layer_run
        .public_output
        .lineage_fingerprint
    )


def test_derive_next_inputs_rejects_noncontiguous_next_index() -> None:
    layer_run = _layer(0).run_complete(
        _inputs(),
        layer_index=0,
        source_stack_fingerprint=(
            "stack-fingerprint"
        ),
    )

    with pytest.raises(
        ValueError,
        match="immediately follow",
    ):
        derive_next_functional_message_passing_inputs(
            original_stack_inputs=(
                layer_run.source_inputs
            ),
            previous_layer_run=(
                layer_run
            ),
            source_stack_fingerprint=(
                "stack-fingerprint"
            ),
            next_layer_index=2,
        )


def test_derive_next_inputs_rejects_blank_stack_fingerprint() -> None:
    inputs = _inputs()
    layer_run = _layer(0).run_complete(
        inputs,
        layer_index=0,
    )

    with pytest.raises(
        ValueError,
        match="source_stack_fingerprint",
    ):
        derive_next_functional_message_passing_inputs(
            original_stack_inputs=inputs,
            previous_layer_run=layer_run,
            source_stack_fingerprint="",
        )


def test_audit_trace_proves_depthwise_state_identity() -> None:
    stack = _stack(
        sharing_policy=(
            STACK_SHARING_FULLY_SHARED
        )
    )
    run = stack.run_complete(
        _inputs(),
        trace_policy=(
            StackTracePolicy.full_audit()
        ),
    )

    assert isinstance(
        run.trace,
        FunctionalMessagePassingStackTrace,
    )
    assert run.trace.layer_indices == (
        0,
        1,
        2,
    )

    layer_runs = run.trace.layer_runs

    assert (
        layer_runs[0].source_inputs
        is run.source_inputs
    )

    for depth in range(
        1,
        NUM_LAYERS
    ):
        previous = layer_runs[
            depth - 1
        ]
        current = layer_runs[
            depth
        ]

        assert isinstance(
            current.source_inputs.node_state,
            ControlledFunctionalNodeState,
        )
        assert (
            current
            .source_inputs
            .node_state
            .state
            is previous.updated_node_state
        )
        assert (
            current
            .source_inputs
            .node_state
            .source_layer_index
            == depth - 1
        )


def test_original_inputs_remain_unchanged_after_stack_execution() -> None:
    stack = _stack()
    inputs = _inputs()
    original_state = (
        inputs
        .node_state
        .fused_state
        .detach()
        .clone()
    )
    original_node_state = (
        inputs.node_state
    )

    stack.run_complete(
        inputs,
        trace_policy=(
            StackTracePolicy.full_audit()
        ),
    )

    assert inputs.node_state is (
        original_node_state
    )
    assert torch.equal(
        inputs.node_state.fused_state,
        original_state,
    )


# =============================================================================
# Output retention and trace detail
# =============================================================================


@pytest.mark.parametrize(
    (
        "retention",
        "expected_indices",
    ),
    (
        (
            STACK_RETENTION_NONE,
            (),
        ),
        (
            STACK_RETENTION_FINAL_LAYER,
            (2,),
        ),
        (
            STACK_RETENTION_ALL_LAYERS,
            (0, 1, 2),
        ),
    ),
)
def test_stack_retention_modes(
    retention: str,
    expected_indices: tuple[int, ...],
) -> None:
    stack = _stack()
    policy = StackTracePolicy(
        retention_policy=retention,
        layer_trace_policy=(
            LAYER_TRACE_NONE
        ),
        audit_mode=False,
    )

    run = stack.run_complete(
        _inputs(),
        trace_policy=policy,
    )
    output = run.public_output

    assert output.retention_policy == (
        retention
    )
    assert output.retained_layer_indices == (
        expected_indices
    )
    assert output.num_retained_layers == (
        len(expected_indices)
    )
    assert output.has_retained_layers is (
        bool(expected_indices)
    )
    assert run.trace is None

    if retention == STACK_RETENTION_NONE:
        assert output.final_layer_output is None
    else:
        assert (
            output.final_layer_output
            is output.retained_layer_outputs[-1]
        )
        assert (
            output.final_node_state
            is output.final_layer_output
            .updated_node_state
        )


@pytest.mark.parametrize(
    "trace_mode",
    (
        LAYER_TRACE_NONE,
        LAYER_TRACE_NODE,
        LAYER_TRACE_FULL,
    ),
)
def test_layer_trace_mode_is_propagated_to_every_depth(
    trace_mode: str,
) -> None:
    stack = _stack(
        sharing_policy=(
            STACK_SHARING_FULLY_SHARED
        )
    )
    policy = StackTracePolicy(
        retention_policy=(
            STACK_RETENTION_ALL_LAYERS
        ),
        layer_trace_policy=(
            trace_mode
        ),
    )

    run = stack.run_complete(
        _inputs(),
        trace_policy=policy,
    )

    assert (
        stack.shared_layer
        .observed_trace_modes
        == [trace_mode] * NUM_LAYERS
    )
    assert run.public_output.layer_trace_mode == (
        trace_mode
    )

    for output in (
        run.public_output
        .retained_layer_outputs
    ):
        if trace_mode == LAYER_TRACE_NONE:
            assert output.intermediates is None
        else:
            assert output.intermediates is not None


def test_retention_and_trace_detail_do_not_change_numerics_in_eval() -> None:
    stack = _stack()
    stack.eval()
    inputs = _inputs()

    minimal = stack(
        inputs,
        trace_policy=(
            StackTracePolicy.minimal()
        ),
    )
    verbose = stack(
        inputs,
        trace_policy=(
            StackTracePolicy.full_audit()
        ),
    )

    assert torch.equal(
        minimal.final_node_state,
        verbose.final_node_state,
    )
    assert (
        minimal.encoder_architecture_fingerprint
        == verbose.encoder_architecture_fingerprint
    )
    assert (
        minimal.execution_contract_fingerprint
        != verbose.execution_contract_fingerprint
    )


def test_audit_mode_retains_complete_runs_without_forcing_public_outputs() -> None:
    stack = _stack()
    policy = StackTracePolicy(
        retention_policy=(
            STACK_RETENTION_NONE
        ),
        layer_trace_policy=(
            LAYER_TRACE_FULL
        ),
        audit_mode=True,
    )

    run = stack.run_complete(
        _inputs(),
        trace_policy=policy,
    )

    assert run.public_output.retained_layer_outputs == ()
    assert run.trace is not None
    assert len(
        run.trace.layer_runs
    ) == NUM_LAYERS


# =============================================================================
# Stack schema and run contracts
# =============================================================================


def test_complete_stack_run_contract() -> None:
    stack = _stack()
    inputs = _inputs()
    run = stack.run_complete(
        inputs,
        trace_policy=(
            StackTracePolicy.full_audit()
        ),
        source_model_fingerprint=(
            "source-model"
        ),
    )

    assert isinstance(
        run,
        FunctionalMessagePassingStackRun,
    )
    assert isinstance(
        run.internal_output,
        FunctionalMessagePassingStackComputationOutput,
    )
    assert isinstance(
        run.public_output,
        FunctionalMessagePassingStackOutput,
    )
    assert run.source_inputs is inputs
    assert run.final_node_state is (
        run.public_output.final_node_state
    )
    assert run.internal_output.stack_inputs is (
        run.stack_inputs
    )
    assert run.public_output.source_inputs is (
        inputs
    )
    assert run.public_output.num_layers == (
        NUM_LAYERS
    )
    assert run.public_output.num_nodes == (
        NODES
    )
    assert run.public_output.hidden_dim == (
        HIDDEN_DIM
    )
    assert run.public_output.dtype == (
        inputs.dtype
    )
    assert run.public_output.device == (
        inputs.device
    )

    validate_functional_message_passing_stack_run(
        run
    )
    validate_stack_run(
        run
    )
    validate_public_stack_output(
        public_output=(
            run.public_output
        ),
        internal_output=(
            run.internal_output
        ),
    )


def test_depth_records_are_contiguous_and_complete() -> None:
    run = _stack().run_complete(
        _inputs(),
        trace_policy=(
            StackTracePolicy(
                retention_policy=(
                    STACK_RETENTION_FINAL_LAYER
                ),
                layer_trace_policy=(
                    LAYER_TRACE_NONE
                ),
            )
        ),
    )

    records = (
        run.internal_output.depth_records
    )

    assert len(records) == NUM_LAYERS
    assert tuple(
        record.layer_index
        for record in records
    ) == (0, 1, 2)
    assert tuple(
        record.retained
        for record in records
    ) == (
        False,
        False,
        True,
    )
    assert (
        run.internal_output
        .retained_layer_indices
        == (2,)
    )


def test_public_output_assembly_preserves_exact_identity() -> None:
    run = _stack().run_complete(
        _inputs(),
        trace_policy=(
            StackTracePolicy.full_audit()
        ),
    )

    rebuilt = (
        assemble_functional_message_passing_stack_output(
            run.internal_output
        )
    )
    alias_rebuilt = assemble_stack_output(
        run.internal_output
    )

    for output in (
        rebuilt,
        alias_rebuilt,
    ):
        assert output.final_node_state is (
            run.internal_output
            .final_node_state
        )
        assert output.source_inputs is (
            run.source_inputs
        )
        assert (
            output.retained_layer_outputs
            == run.public_output
            .retained_layer_outputs
        )
        validate_public_stack_output(
            public_output=output,
            internal_output=(
                run.internal_output
            ),
        )


def test_expected_retained_layer_indices_helper() -> None:
    assert expected_retained_layer_indices(
        retention_policy=(
            STACK_RETENTION_NONE
        ),
        num_layers=3,
    ) == ()
    assert expected_retained_layer_indices(
        retention_policy=(
            STACK_RETENTION_FINAL_LAYER
        ),
        num_layers=3,
    ) == (2,)
    assert expected_retained_layer_indices(
        retention_policy=(
            STACK_RETENTION_ALL_LAYERS
        ),
        num_layers=3,
    ) == (0, 1, 2)


def test_stack_fingerprints_are_nonempty() -> None:
    run = _stack().run_complete(
        _inputs()
    )
    internal = run.internal_output
    public = run.public_output

    for value in (
        internal.stack_architecture_fingerprint,
        internal.stack_parameter_fingerprint,
        internal.execution_contract_fingerprint,
        internal.lineage_fingerprint,
        public.encoder_architecture_fingerprint,
        public.execution_contract_fingerprint,
        public.lineage_fingerprint,
    ):
        assert isinstance(
            value,
            str,
        )
        assert value


def test_lineage_changes_when_source_model_fingerprint_changes() -> None:
    stack = _stack()
    stack.eval()
    inputs = _inputs()

    first = stack.run_complete(
        inputs,
        source_model_fingerprint=(
            "model-a"
        ),
    )
    second = stack.run_complete(
        inputs,
        source_model_fingerprint=(
            "model-b"
        ),
    )

    assert torch.equal(
        first.final_node_state,
        second.final_node_state,
    )
    assert (
        first.public_output.lineage_fingerprint
        != second.public_output.lineage_fingerprint
    )


# =============================================================================
# Regularization preservation
# =============================================================================


def test_regularization_mapping_is_preserved_per_depth() -> None:
    run = _stack().run_complete(
        _inputs(),
        trace_policy=(
            StackTracePolicy(
                retention_policy=(
                    STACK_RETENTION_ALL_LAYERS
                ),
                layer_trace_policy=(
                    LAYER_TRACE_NONE
                ),
            )
        ),
    )

    mappings = (
        run.public_output
        .layer_regularization_terms
    )

    assert len(mappings) == NUM_LAYERS

    for depth, mapping in enumerate(
        mappings
    ):
        output = (
            run.public_output
            .retained_layer_outputs[
                depth
            ]
        )
        assert isinstance(
            mapping,
            MappingProxyType,
        )
        assert tuple(mapping) == (
            "weight_l2",
        )
        assert (
            mapping["weight_l2"]
            is output
            .regularization_terms[
                "weight_l2"
            ]
        )


def test_regularization_is_not_reduced_by_stack() -> None:
    run = _stack(
        sharing_policy=(
            STACK_SHARING_FULLY_SHARED
        )
    ).run_complete(
        _inputs(),
        trace_policy=(
            StackTracePolicy(
                retention_policy=(
                    STACK_RETENTION_NONE
                ),
            )
        ),
    )

    terms = (
        run.public_output
        .layer_regularization_terms
    )

    assert len(terms) == NUM_LAYERS
    assert all(
        tuple(mapping) == (
            "weight_l2",
        )
        for mapping in terms
    )


# =============================================================================
# Gradients and sharing
# =============================================================================


def test_independent_stack_backpropagates_to_every_layer() -> None:
    stack = _stack(
        sharing_policy=(
            STACK_SHARING_INDEPENDENT
        )
    )
    stack.train()
    inputs = _inputs(
        requires_grad=True
    )

    output = stack(
        inputs
    )
    loss = (
        output.final_node_state
        .square()
        .mean()
    )
    loss.backward()

    assert (
        inputs
        .node_state
        .fused_state
        .grad
        is not None
    )

    for layer in stack.layers:
        assert layer.weight.grad is not None
        assert layer.bias.grad is not None
        assert torch.isfinite(
            layer.weight.grad
        ).all()
        assert torch.isfinite(
            layer.bias.grad
        ).all()


def test_fully_shared_stack_accumulates_gradient_on_one_parameter_set() -> None:
    stack = _stack(
        sharing_policy=(
            STACK_SHARING_FULLY_SHARED
        )
    )
    stack.train()
    inputs = _inputs(
        requires_grad=True
    )

    output = stack(
        inputs
    )
    output.final_node_state.sum().backward()

    assert stack.num_unique_layers == 1
    assert (
        stack.shared_layer.weight.grad
        is not None
    )
    assert (
        stack.shared_layer.bias.grad
        is not None
    )
    assert torch.isfinite(
        stack.shared_layer.weight.grad
    ).all()


def test_state_rebinding_does_not_break_autograd_chain() -> None:
    stack = _stack(
        sharing_policy=(
            STACK_SHARING_INDEPENDENT
        )
    )
    inputs = _inputs(
        requires_grad=True
    )

    run = stack.run_complete(
        inputs,
        trace_policy=(
            StackTracePolicy.full_audit()
        ),
    )
    run.final_node_state.sum().backward()

    first_output = (
        run.trace
        .layer_runs[0]
        .updated_node_state
    )

    assert first_output.grad_fn is not None
    assert (
        inputs
        .node_state
        .fused_state
        .grad
        is not None
    )
    assert stack.layers[0].weight.grad is not None
    assert stack.layers[-1].weight.grad is not None


def test_independent_layers_have_distinct_parameter_objects() -> None:
    stack = _stack(
        sharing_policy=(
            STACK_SHARING_INDEPENDENT
        )
    )

    parameter_ids = tuple(
        id(layer.weight)
        for layer in stack.layers
    )
    storage_pointers = tuple(
        layer.weight
        .untyped_storage()
        .data_ptr()
        for layer in stack.layers
    )

    assert len(set(parameter_ids)) == (
        NUM_LAYERS
    )
    assert len(set(storage_pointers)) == (
        NUM_LAYERS
    )


def test_fully_shared_depths_use_exact_same_parameter_objects() -> None:
    stack = _stack(
        sharing_policy=(
            STACK_SHARING_FULLY_SHARED
        )
    )

    layers = stack.layers_by_depth

    assert all(
        layer is layers[0]
        for layer in layers
    )
    assert all(
        layer.weight is layers[0].weight
        for layer in layers
    )


# =============================================================================
# Evaluation determinism and training stochasticity
# =============================================================================


def test_eval_mode_is_deterministic_with_dropout_layers() -> None:
    stack = _stack(
        dropout_probability=0.5
    )
    stack.eval()
    inputs = _inputs()

    first = stack(
        inputs
    )
    second = stack(
        inputs
    )

    assert torch.equal(
        first.final_node_state,
        second.final_node_state,
    )


def test_train_mode_dropout_changes_output_for_different_rng_states() -> None:
    stack = _stack(
        dropout_probability=0.5
    )
    stack.train()
    inputs = _inputs()

    torch.manual_seed(
        1
    )
    first = stack(
        inputs
    )
    torch.manual_seed(
        2
    )
    second = stack(
        inputs
    )

    assert not torch.equal(
        first.final_node_state,
        second.final_node_state,
    )


def test_train_mode_dropout_is_reproducible_with_reset_seed() -> None:
    stack = _stack(
        dropout_probability=0.5
    )
    stack.train()
    inputs = _inputs()

    torch.manual_seed(
        123
    )
    first = stack(
        inputs
    )
    torch.manual_seed(
        123
    )
    second = stack(
        inputs
    )

    assert torch.equal(
        first.final_node_state,
        second.final_node_state,
    )


# =============================================================================
# Serialization
# =============================================================================


@pytest.mark.parametrize(
    "sharing_policy",
    (
        STACK_SHARING_INDEPENDENT,
        STACK_SHARING_FULLY_SHARED,
    ),
)
def test_state_dict_round_trip_preserves_eval_output(
    sharing_policy: str,
) -> None:
    source = _stack(
        sharing_policy=(
            sharing_policy
        )
    )
    target = _stack(
        sharing_policy=(
            sharing_policy
        )
    )
    source.eval()
    target.eval()
    inputs = _inputs()

    with torch.no_grad():
        for index, parameter in enumerate(
            source.parameters()
        ):
            parameter.add_(
                float(index + 1)
                / 10.0
            )

    expected = source(
        inputs
    ).final_node_state

    buffer = io.BytesIO()
    torch.save(
        source.state_dict(),
        buffer,
    )
    buffer.seek(0)
    loaded = _torch_load_state_dict(
        buffer
    )
    result = target.load_state_dict(
        loaded
    )

    assert result.missing_keys == []
    assert result.unexpected_keys == []

    observed = target(
        inputs
    ).final_node_state

    assert torch.equal(
        observed,
        expected,
    )
    assert (
        target.parameter_fingerprint()
        == source.parameter_fingerprint()
    )


def test_incompatible_sharing_state_dict_is_rejected() -> None:
    independent = _stack(
        sharing_policy=(
            STACK_SHARING_INDEPENDENT
        )
    )
    shared = _stack(
        sharing_policy=(
            STACK_SHARING_FULLY_SHARED
        )
    )

    with pytest.raises(
        RuntimeError,
    ):
        shared.load_state_dict(
            independent.state_dict()
        )


# =============================================================================
# Functional builders and execution helpers
# =============================================================================


def test_build_functional_stack_helper() -> None:
    layers = _layers()

    stack = (
        build_functional_message_passing_stack(
            layers,
            num_layers=NUM_LAYERS,
            sharing_policy=(
                STACK_SHARING_INDEPENDENT
            ),
        )
    )

    assert isinstance(
        stack,
        FunctionalMessagePassingStack,
    )
    assert stack.layers_by_depth == (
        layers
    )


def test_build_stack_from_factory_independent() -> None:
    calls: list[int] = []

    def factory(
        depth: int,
    ) -> ControlledFunctionalMessagePassingLayer:
        calls.append(
            depth
        )
        return _layer(
            depth
        )

    stack = (
        build_functional_message_passing_stack_from_factory(
            factory,
            num_layers=NUM_LAYERS,
            sharing_policy=(
                STACK_SHARING_INDEPENDENT
            ),
        )
    )

    assert calls == [
        0,
        1,
        2,
    ]
    assert stack.num_unique_layers == (
        NUM_LAYERS
    )


def test_build_stack_from_factory_fully_shared() -> None:
    calls: list[int] = []

    def factory(
        depth: int,
    ) -> ControlledFunctionalMessagePassingLayer:
        calls.append(
            depth
        )
        return _layer(
            depth
        )

    stack = build_stack_from_factory(
        factory,
        num_layers=NUM_LAYERS,
        sharing_policy=(
            STACK_SHARING_FULLY_SHARED
        ),
    )

    assert calls == [0]
    assert stack.num_unique_layers == 1


def test_functional_execution_helpers() -> None:
    stack = _stack()
    inputs = _inputs()

    public = (
        run_functional_message_passing_stack(
            stack,
            inputs,
        )
    )
    complete = (
        run_functional_message_passing_stack_complete(
            stack,
            inputs,
        )
    )

    assert isinstance(
        public,
        FunctionalMessagePassingStackOutput,
    )
    assert isinstance(
        complete,
        FunctionalMessagePassingStackRun,
    )


@pytest.mark.parametrize(
    "helper",
    (
        run_functional_message_passing_stack,
        run_functional_message_passing_stack_complete,
    ),
)
def test_functional_execution_helpers_reject_wrong_stack(
    helper: Any,
) -> None:
    with pytest.raises(
        TypeError,
        match="FunctionalMessagePassingStack",
    ):
        helper(
            object(),
            _inputs(),
        )


# =============================================================================
# Diagnostics
# =============================================================================


def test_diagnostic_scientific_flags() -> None:
    assert (
        STACK_DIAGNOSTICS_AFFECT_NUMERICAL_RESULTS
        is False
    )
    assert (
        STACK_DIAGNOSTICS_RETAIN_AUTOGRAD_REFERENCES
        is False
    )
    assert (
        STACK_DIAGNOSTICS_RETURN_TENSORS
        is False
    )
    assert (
        STACK_DIAGNOSTICS_ESTABLISH_CAUSALITY
        is False
    )
    assert (
        STACK_DIAGNOSTICS_ESTABLISH_FAITHFULNESS
        is False
    )
    assert (
        STACK_DIAGNOSTICS_ESTABLISH_CALIBRATION
        is False
    )
    assert (
        STACK_DIAGNOSTICS_ESTABLISH_IDENTIFIABILITY
        is False
    )

    assert (
        STACK_DIAGNOSTIC_SCIENTIFIC_CLAIMS[
            "descriptive_state_statistics"
        ]
        is True
    )
    assert (
        STACK_DIAGNOSTIC_SCIENTIFIC_CLAIMS[
            "causal_attribution"
        ]
        is False
    )


def test_diagnostic_thresholds_valid_contract() -> None:
    thresholds = (
        StackDiagnosticThresholds()
    )

    assert thresholds.norm_epsilon > 0.0
    assert (
        thresholds
        .collapse_output_to_input_ratio
        < thresholds
        .explosion_output_to_input_ratio
    )
    assert (
        thresholds
        .tiny_update_to_input_ratio
        < thresholds
        .large_update_to_input_ratio
    )
    assert (
        thresholds.fingerprint()
        == thresholds.fingerprint()
    )


@pytest.mark.parametrize(
    (
        "field",
        "value",
    ),
    (
        (
            "norm_epsilon",
            0.0,
        ),
        (
            "collapse_output_to_input_ratio",
            -1.0,
        ),
        (
            "explosion_output_to_input_ratio",
            float("inf"),
        ),
        (
            "direction_reversal_cosine_threshold",
            2.0,
        ),
    ),
)
def test_diagnostic_thresholds_reject_invalid_values(
    field: str,
    value: float,
) -> None:
    kwargs = {
        field: value,
    }

    with pytest.raises(
        (
            ValueError,
            TypeError,
        ),
    ):
        StackDiagnosticThresholds(
            **kwargs
        )


def test_ordinary_forward_does_not_invoke_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    diagnostics = (
        FunctionalMessagePassingStackDiagnostics()
    )
    stack = _stack(
        diagnostics=diagnostics
    )

    def forbidden(
        *args: Any,
        **kwargs: Any,
    ) -> None:
        raise AssertionError(
            "ordinary forward invoked diagnostics"
        )

    monkeypatch.setattr(
        FunctionalMessagePassingStackDiagnostics,
        "public_report",
        forbidden,
    )

    output = stack(
        _inputs()
    )

    assert isinstance(
        output,
        FunctionalMessagePassingStackOutput,
    )


def test_explicit_diagnostics_without_audit_reports_partial_coverage() -> None:
    diagnostics = (
        FunctionalMessagePassingStackDiagnostics()
    )
    stack = _stack(
        diagnostics=diagnostics
    )
    run = stack.run_complete(
        _inputs(),
        trace_policy=(
            StackTracePolicy(
                retention_policy=(
                    STACK_RETENTION_FINAL_LAYER
                ),
                layer_trace_policy=(
                    LAYER_TRACE_NONE
                ),
                audit_mode=False,
            )
        ),
    )

    report = diagnostics.report(
        run
    )

    assert isinstance(
        report,
        FunctionalMessagePassingStackDiagnosticReport,
    )
    assert report.available_depth_indices == (
        2,
    )
    assert not report.complete_depth_coverage
    assert (
        STACK_DIAGNOSTIC_ALERT_INCOMPLETE_DEPTH_COVERAGE
        in report.alerts
    )
    assert tuple(
        summary.source
        for summary in report.depth_summaries
    ) == (
        STACK_DIAGNOSTIC_DEPTH_SOURCE_UNAVAILABLE,
        STACK_DIAGNOSTIC_DEPTH_SOURCE_UNAVAILABLE,
        STACK_DIAGNOSTIC_DEPTH_SOURCE_RETAINED_OUTPUT,
    )


def test_audit_diagnostics_have_complete_depth_coverage() -> None:
    diagnostics = (
        FunctionalMessagePassingStackDiagnostics()
    )
    stack = _stack(
        diagnostics=diagnostics
    )
    run = stack.run_complete(
        _inputs(),
        trace_policy=(
            StackTracePolicy.full_audit(
                retention_policy=(
                    STACK_RETENTION_NONE
                )
            )
        ),
    )

    report = (
        build_functional_message_passing_stack_diagnostic_report(
            run
        )
    )

    assert report.available_depth_indices == (
        0,
        1,
        2,
    )
    assert report.complete_depth_coverage
    assert all(
        summary.source
        == STACK_DIAGNOSTIC_DEPTH_SOURCE_AUDIT_TRACE
        for summary in report.depth_summaries
    )
    assert (
        STACK_DIAGNOSTIC_ALERT_INCOMPLETE_DEPTH_COVERAGE
        not in report.alerts
    )


def test_diagnostic_report_is_tensor_free_and_deterministic() -> None:
    diagnostics = (
        FunctionalMessagePassingStackDiagnostics()
    )
    stack = _stack(
        diagnostics=diagnostics
    )
    stack.eval()
    run = stack.run_complete(
        _inputs(),
        trace_policy=(
            StackTracePolicy.full_audit()
        ),
    )

    first = diagnostics.report(
        run
    )
    second = (
        build_stack_diagnostic_report(
            run
        )
    )

    public = first.public_report()

    _assert_tensor_free(
        public
    )
    assert (
        first.report_fingerprint()
        == second.report_fingerprint()
    )
    assert public[
        "report_fingerprint"
    ] == first.report_fingerprint()

    validate_functional_message_passing_stack_diagnostic_report(
        first
    )
    validate_stack_diagnostic_report(
        first
    )


def test_forward_with_diagnostics_returns_wrapped_run() -> None:
    stack = _stack(
        diagnostics=(
            FunctionalMessagePassingStackDiagnostics()
        )
    )

    result = stack.forward_with_diagnostics(
        _inputs(),
        trace_policy=(
            StackTracePolicy.full_audit()
        ),
    )

    assert isinstance(
        result,
        FunctionalMessagePassingStackRunWithDiagnostics,
    )
    assert isinstance(
        result.run,
        FunctionalMessagePassingStackRun,
    )
    assert (
        result.public_output
        is result.run.public_output
    )
    assert (
        result.internal_output
        is result.run.internal_output
    )
    _assert_tensor_free(
        result.diagnostic_report
    )


def test_stack_diagnostic_report_requires_configured_diagnostics() -> None:
    stack = _stack()
    run = stack.run_complete(
        _inputs()
    )

    with pytest.raises(
        RuntimeError,
        match="not configured",
    ):
        stack.diagnostic_report(
            run=run
        )


def test_disabled_diagnostics_reject_report_generation() -> None:
    diagnostics = (
        FunctionalMessagePassingStackDiagnostics(
            enabled=False
        )
    )
    stack = _stack(
        diagnostics=diagnostics
    )
    run = stack.run_complete(
        _inputs()
    )

    with pytest.raises(
        RuntimeError,
        match="disabled",
    ):
        diagnostics.report(
            run
        )


def test_regularization_diagnostic_summary_is_descriptive_only() -> None:
    diagnostics = (
        FunctionalMessagePassingStackDiagnostics()
    )
    stack = _stack(
        diagnostics=diagnostics,
        sharing_policy=(
            STACK_SHARING_FULLY_SHARED
        ),
    )
    run = stack.run_complete(
        _inputs(),
        trace_policy=(
            StackTracePolicy.full_audit()
        ),
    )
    report = diagnostics.report(
        run
    )
    summary = (
        report.regularization_summary
    )

    assert summary[
        "training_reduction_applied"
    ] is False
    assert summary[
        "shared_parameter_deduplication_applied"
    ] is False
    assert len(
        summary["per_depth"]
    ) == NUM_LAYERS
    assert summary[
        "occurrence_count_by_name"
    ][
        "weight_l2"
    ] == NUM_LAYERS


# =============================================================================
# Defensive stack execution failures
# =============================================================================


def test_stack_rejects_layer_emitting_wrong_runtime_index() -> None:
    bad = _layer(
        0,
        emitted_layer_index_offset=1,
    )
    stack = (
        FunctionalMessagePassingStack.from_layers(
            (bad,),
            num_layers=1,
            sharing_policy=(
                STACK_SHARING_INDEPENDENT
            ),
        )
    )

    with pytest.raises(
        ValueError,
        match="wrong runtime layer index",
    ):
        stack(
            _inputs()
        )


def test_stack_rejects_layer_losing_source_input_identity() -> None:
    bad = _layer(
        0,
        replace_source_inputs=True,
    )
    stack = (
        FunctionalMessagePassingStack.from_layers(
            (bad,),
            num_layers=1,
            sharing_policy=(
                STACK_SHARING_INDEPENDENT
            ),
        )
    )

    with pytest.raises(
        ValueError,
        match=r"source[- ]input",
    ):
        stack(
            _inputs()
        )


def test_stack_rejects_nonfinite_layer_output() -> None:
    bad = _layer(
        0,
        emit_nonfinite_state=True,
    )
    stack = (
        FunctionalMessagePassingStack.from_layers(
            (bad,),
            num_layers=1,
            sharing_policy=(
                STACK_SHARING_INDEPENDENT
            ),
        )
    )

    with pytest.raises(
        (
            FloatingPointError,
            ValueError,
        ),
    ):
        stack(
            _inputs()
        )


def test_stack_rejects_nonfinite_parameters() -> None:
    stack = _stack()

    with torch.no_grad():
        stack.layers[0].weight[
            0,
            0,
        ] = float("nan")

    with pytest.raises(
        FloatingPointError,
    ):
        stack(
            _inputs()
        )


def test_stack_rejects_layer_training_mode_mismatch() -> None:
    stack = _stack()
    stack.train()
    stack.layers[1].eval()

    with pytest.raises(
        ValueError,
        match="train/eval mode",
    ):
        stack(
            _inputs()
        )


def test_validate_public_output_rejects_tampered_final_state() -> None:
    run = _stack().run_complete(
        _inputs(),
        trace_policy=(
            StackTracePolicy(
                retention_policy=(
                    STACK_RETENTION_FINAL_LAYER
                ),
            )
        ),
    )

    object.__setattr__(
        run.public_output,
        "final_node_state",
        run.public_output
        .final_node_state
        .clone(),
    )

    with pytest.raises(
        ValueError,
        match="exact internal final-node-state",
    ):
        validate_public_stack_output(
            public_output=(
                run.public_output
            ),
            internal_output=(
                run.internal_output
            ),
        )


def test_validate_stack_run_rejects_wrong_type() -> None:
    with pytest.raises(
        TypeError,
        match="FunctionalMessagePassingStackRun",
    ):
        validate_functional_message_passing_stack_run(
            object()  # type: ignore[arg-type]
        )


# =============================================================================
# Dtype and device coverage
# =============================================================================


def test_float64_stack_execution_and_gradients() -> None:
    stack = _stack(
        dtype=torch.float64
    )
    inputs = _inputs(
        dtype=torch.float64,
        requires_grad=True,
    )

    output = stack(
        inputs
    )

    assert output.dtype == (
        torch.float64
    )
    assert output.final_node_state.dtype == (
        torch.float64
    )

    output.final_node_state.sum().backward()

    assert (
        inputs
        .node_state
        .fused_state
        .grad
        is not None
    )
    for layer in stack.layers:
        assert layer.weight.grad is not None
        assert layer.weight.grad.dtype == (
            torch.float64
        )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is not available.",
)
def test_cuda_stack_execution_and_gradients() -> None:
    device = torch.device(
        "cuda"
    )
    stack = _stack(
        device=device
    )
    inputs = _inputs(
        device=device,
        requires_grad=True,
    )

    output = stack(
        inputs,
        trace_policy=(
            StackTracePolicy.full_audit()
        ),
    )

    assert output.device.type == "cuda"
    assert (
        output.final_node_state.device.type
        == "cuda"
    )

    output.final_node_state.sum().backward()

    assert (
        inputs
        .node_state
        .fused_state
        .grad
        is not None
    )
    assert (
        inputs
        .node_state
        .fused_state
        .grad
        .device
        .type
        == "cuda"
    )


# =============================================================================
# Final public export inventories
# =============================================================================


def test_stack_module_expected_core_exports() -> None:
    expected = {
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
        "derive_next_functional_message_passing_inputs",
        "derive_next_layer_inputs",
        "validate_rebound_functional_message_passing_inputs",
        "validate_rebound_inputs",
        "FunctionalMessagePassingStack",
        "HazardConditionedFunctionalMessagePassingStack",
        "FunctionalStack",
        "MessagePassingStack",
        "build_functional_message_passing_stack",
        "build_stack",
        "build_functional_message_passing_stack_from_factory",
        "build_stack_from_factory",
        "run_functional_message_passing_stack",
        "run_stack",
        "run_functional_message_passing_stack_complete",
        "run_stack_complete",
    }

    assert set(
        stack_module.__all__
    ) == expected


def test_diagnostics_expected_core_exports() -> None:
    expected = {
        "STACK_DIAGNOSTIC_THRESHOLDS_SCHEMA_VERSION",
        "STACK_DEPTH_DIAGNOSTIC_SCHEMA_VERSION",
        "STACK_DIAGNOSTIC_REPORT_SCHEMA_VERSION",
        "STACK_DIAGNOSTICS_SCHEMA_VERSION",
        "STACK_DIAGNOSTICS_AFFECT_NUMERICAL_RESULTS",
        "STACK_DIAGNOSTICS_RETAIN_AUTOGRAD_REFERENCES",
        "STACK_DIAGNOSTICS_RETURN_TENSORS",
        "STACK_DIAGNOSTICS_ESTABLISH_CAUSALITY",
        "STACK_DIAGNOSTICS_ESTABLISH_FAITHFULNESS",
        "STACK_DIAGNOSTICS_ESTABLISH_CALIBRATION",
        "STACK_DIAGNOSTICS_ESTABLISH_IDENTIFIABILITY",
        "STACK_DIAGNOSTIC_DEPTH_SOURCE_AUDIT_TRACE",
        "STACK_DIAGNOSTIC_DEPTH_SOURCE_RETAINED_OUTPUT",
        "STACK_DIAGNOSTIC_DEPTH_SOURCE_UNAVAILABLE",
        "CANONICAL_STACK_DIAGNOSTIC_DEPTH_SOURCES",
        "STACK_DIAGNOSTIC_ALERT_TOTAL_STATE_COLLAPSE",
        "STACK_DIAGNOSTIC_ALERT_TOTAL_STATE_EXPLOSION",
        "STACK_DIAGNOSTIC_ALERT_TOTAL_UPDATE_LARGE",
        "STACK_DIAGNOSTIC_ALERT_DEPTH_STATE_COLLAPSE",
        "STACK_DIAGNOSTIC_ALERT_DEPTH_STATE_EXPLOSION",
        "STACK_DIAGNOSTIC_ALERT_DEPTH_UPDATE_LARGE",
        "STACK_DIAGNOSTIC_ALERT_DEPTH_UPDATE_TINY",
        "STACK_DIAGNOSTIC_ALERT_DEPTH_DIRECTION_REVERSAL",
        "STACK_DIAGNOSTIC_ALERT_INCOMPLETE_DEPTH_COVERAGE",
        "STACK_DIAGNOSTIC_ALERT_SHARED_PARAMETER_DRIFT",
        "STACK_DIAGNOSTIC_ALERT_SHARED_ARCHITECTURE_DRIFT",
        "STACK_DIAGNOSTIC_SCIENTIFIC_CLAIMS",
        "StackDiagnosticThresholds",
        "DiagnosticThresholds",
        "StackDepthDiagnostic",
        "DepthDiagnostic",
        "FunctionalMessagePassingStackDiagnosticReport",
        "StackDiagnosticReport",
        "build_functional_message_passing_stack_diagnostic_report",
        "build_stack_diagnostic_report",
        "build_diagnostic_report",
        "validate_functional_message_passing_stack_diagnostic_report",
        "validate_stack_diagnostic_report",
        "validate_diagnostic_report",
        "FunctionalMessagePassingStackDiagnostics",
        "StackDiagnostics",
    }

    assert set(
        diagnostics_module.__all__
    ) == expected
