"""
Failure-hardening tests for one functional message-passing layer.

Target repository path:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            tests/
                layer/
                    test_layer_failures.py

Implementations under test:
    functional_message_passing/layer/schemas.py
    functional_message_passing/layer/residual_update.py
    functional_message_passing/layer/normalization.py
    functional_message_passing/layer/diagnostics.py
    functional_message_passing/layer/layer.py

This suite centralizes negative contracts that should not be mixed into the
successful component and integration suites. It deliberately constructs stale,
crossed, malformed, non-finite, or semantically inconsistent objects and checks
that the layer fails loudly rather than silently accepting corrupted lineage.

Failure classes covered here include:

- invalid constructor component types;
- incompatible hidden widths and exact-relation axes;
- source-input schema, relation-order, stable-ID, and registry mismatches;
- invalid runtime layer identity, trace policy, and train/eval state;
- disabled configurations and malformed configuration overrides;
- edge-stage outputs sourced from a different execution;
- gate and attention presence inconsistent with layer configuration;
- message-builder and aggregation lineage crossings;
- residual and normalization outputs from stale runs;
- forbidden trace retention combinations;
- malformed internal and public outputs;
- stale complete runs and post-run parameter mutation;
- malformed regularization mappings;
- non-finite trainable parameters;
- disabled diagnostics, non-serializable reports, report tampering, and
  unsupported claims.

The controlled modules are real ``nn.Module`` objects producing the actual
immutable public schemas. The layer module's imported component classes are
patched to these controlled implementations, isolating orchestration and
cross-component contracts from the mathematical unit tests in
``test_layer_components.py``.
"""


from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Any, Mapping

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.constants import (
    AGGREGATION_MEAN,
    ATTENTION_HEAD_REDUCTION_MEAN,
    ATTENTION_MODE_MULTIHEAD_HAZARD_CONDITIONED,
    ATTENTION_NORMALIZATION_TARGET_NODE_RELATION,
    EDGE_NORMALIZATION_NONE,
    RELATION_GATE_ACTIVATION_SIGMOID,
    RELATION_GATE_SCOPE_TARGET_NODE,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.functional_message_passing import (
    schemas as fmp_schemas,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.functional_message_passing.layer import (
    layer as layer_module,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.functional_message_passing.layer.diagnostics import (
    LayerDiagnosticThresholds,
    LayerDiagnostics,
    build_layer_diagnostic_report,
    build_layer_diagnostics,
    validate_layer_diagnostic_report,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.functional_message_passing.layer.layer import (
    FUNCTIONAL_MESSAGE_PASSING_LAYER_AGGREGATION_MATH_OWNED_HERE,
    FUNCTIONAL_MESSAGE_PASSING_LAYER_AGGREGATION_ORCHESTRATED_HERE,
    FUNCTIONAL_MESSAGE_PASSING_LAYER_DIAGNOSTICS_AFFECT_NUMERICS,
    FUNCTIONAL_MESSAGE_PASSING_LAYER_DISABLED_ATTENTION_REPRESENTATION,
    FUNCTIONAL_MESSAGE_PASSING_LAYER_DISABLED_GATE_REPRESENTATION,
    FUNCTIONAL_MESSAGE_PASSING_LAYER_OPERATION_ORDER,
    FUNCTIONAL_MESSAGE_PASSING_LAYER_OUTPUT_SCHEMA,
    FUNCTIONAL_MESSAGE_PASSING_LAYER_PREDICTION_OWNED_HERE,
    FUNCTIONAL_MESSAGE_PASSING_LAYER_SCHEMA_VERSION,
    FUNCTIONAL_MESSAGE_PASSING_LAYER_SCIENTIFIC_INTERPRETATION,
    FUNCTIONAL_MESSAGE_PASSING_LAYER_STACKING_OWNED_HERE,
    FUNCTIONAL_MESSAGE_PASSING_LAYER_TRACE_AFFECTS_NUMERICS,
    FUNCTIONAL_MESSAGE_PASSING_LAYER_UNIFORM_ATTENTION_REPRESENTATION,
    FunctionalLayer,
    FunctionalMessagePassingLayer,
    FunctionalMessagePassingLayerEdgeStages,
    FunctionalMessagePassingLayerNodeStages,
    FunctionalMessagePassingLayerRun,
    FunctionalMessagePassingLayerRunWithDiagnostics,
    HazardConditionedFunctionalMessagePassingLayer,
    LayerEdgeStages,
    LayerNodeStages,
    LayerRun,
    LayerRunWithDiagnostics,
    MessagePassingLayer,
    assemble_functional_message_passing_layer_output,
    build_functional_message_passing_layer,
    build_functional_message_passing_layer_from_config,
    build_layer,
    build_layer_from_config,
    run_functional_message_passing_layer,
    run_functional_message_passing_layer_complete,
    run_layer,
    run_layer_complete,
    validate_functional_message_passing_layer_run,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.functional_message_passing.layer.normalization import (
    LAYER_NORMALIZATION_LAYER_NORM,
    LAYER_NORMALIZATION_NONE,
    LAYER_NORMALIZATION_POST_RESIDUAL,
    LayerNormalizer,
    build_layer_normalization_output,
    build_layer_normalizer_from_flag,
    validate_layer_normalization_output,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.functional_message_passing.layer.residual_update import (
    LAYER_RESIDUAL_ADDITIVE,
    LAYER_RESIDUAL_DISABLED,
    LayerResidualUpdater,
    build_layer_residual_update_output,
    build_layer_residual_updater_from_flags,
    validate_layer_residual_update_output,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.functional_message_passing.layer.schemas import (
    LAYER_NORMALIZATION_PRE_RESIDUAL,
    LAYER_TRACE_FULL,
    LAYER_TRACE_NODE,
    LAYER_TRACE_NONE,
    FunctionalMessagePassingLayerInputs,
    FunctionalMessagePassingLayerStages,
    FunctionalMessagePassingLayerTrace,
    LayerComputationOutput,
    LayerNormalizationOutput,
    LayerResidualUpdateOutput,
    LayerTracePolicy,
    validate_layer_stage_chain,
    validate_public_layer_output,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.functional_message_passing.message_builders import (
    MESSAGE_SEMANTIC_EDGE_POLICY_IGNORE,
    EdgeMessageBuilder,
    MessageBuilderRun,
    build_edge_message_builder,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.functional_message_passing.schemas import (
    AggregationOutput,
    EdgeAttentionOutput,
    EdgeMessageOutput,
    FunctionalMessagePassingInputs,
    FunctionalMessagePassingLayerOutput,
    RelationGateOutput,
    RelationTransformOutput,
    StructuralEdgeNormalizationOutput,
)


NODES = 5
EDGES = 7
GRAPHS = 2
RELATIONS = 3
HIDDEN_DIM = 4
HEADS = 2
NUM_GROUPS = NODES * RELATIONS

RELATION_NAMES = (
    "spatial_adjacency",
    "drainage_dependency",
    "random_placebo",
)
STABLE_RELATION_IDS = (
    100,
    310,
    900,
)
CONTROL_RELATIONS = (
    False,
    False,
    True,
)

_DEFAULT = object()


# Controlled upstream contracts
# =============================================================================


@dataclass
class FakeSpecification:
    is_control: bool = False


@dataclass
class FakeCompiledEntry:
    name: str
    relation_id: int
    specification: FakeSpecification


class FakeCompiledRelationRegistry:
    def __init__(
        self,
        *,
        names: tuple[str, ...] = RELATION_NAMES,
        stable_ids: tuple[int, ...] = STABLE_RELATION_IDS,
        controls: tuple[bool, ...] = CONTROL_RELATIONS,
        fingerprint: str = "compiled-relation-registry",
    ) -> None:
        self.relation_names = names
        self.stable_relation_ids = stable_ids
        self.entries = tuple(
            FakeCompiledEntry(
                name=name,
                relation_id=relation_id,
                specification=FakeSpecification(
                    is_control=is_control,
                ),
            )
            for name, relation_id, is_control in zip(
                names,
                stable_ids,
                controls,
                strict=True,
            )
        )
        self._fingerprint = fingerprint
        self.validated = False

    def __len__(self) -> int:
        return len(self.entries)

    def validate(self) -> None:
        self.validated = True

    def fingerprint(self) -> str:
        return self._fingerprint


@dataclass
class FakeAlignment:
    item_ids: tuple[str, ...]
    node_batch_index: torch.Tensor | None
    graph_count: int | None


class FakeNodeStateFusionOutput:
    def __init__(
        self,
        *,
        fused_state: torch.Tensor,
        alignment: FakeAlignment,
        lineage_fingerprint: str = "node-state-lineage",
        encoder_architecture_fingerprint: str = (
            "node-state-architecture"
        ),
    ) -> None:
        self.fused_state = fused_state
        self.alignment = alignment
        self.lineage_fingerprint = lineage_fingerprint
        self.encoder_architecture_fingerprint = (
            encoder_architecture_fingerprint
        )

    @property
    def item_count(self) -> int:
        return int(self.fused_state.shape[0])

    @property
    def output_dim(self) -> int:
        return int(self.fused_state.shape[1])


class FakeUrbanGraphBatch:
    def __init__(
        self,
        *,
        external_node_ids: tuple[str, ...],
        node_batch_index: torch.Tensor,
        edge_index: torch.Tensor,
        edge_relation_type: torch.Tensor,
        edge_attributes: torch.Tensor | None = None,
        semantic_edge_weight: torch.Tensor | None = None,
        edge_batch_index: torch.Tensor | None = None,
        allow_cross_graph_edges: bool = False,
    ) -> None:
        self.external_node_ids = external_node_ids
        self.node_batch_index = node_batch_index
        self.edge_index = edge_index
        self.edge_relation_type = edge_relation_type
        self.edge_attributes = edge_attributes
        self.semantic_edge_weight = semantic_edge_weight
        self.edge_batch_index = edge_batch_index
        self.allow_cross_graph_edges = allow_cross_graph_edges
        self.validated = False

    def validate(self) -> None:
        self.validated = True

    @property
    def num_nodes(self) -> int:
        return len(self.external_node_ids)

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[1])

    @property
    def batch_size(self) -> int:
        return int(
            self.node_batch_index.max().item()
        ) + 1


@pytest.fixture(autouse=True)
def _patch_upstream_contracts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        fmp_schemas,
        "UrbanGraphBatch",
        FakeUrbanGraphBatch,
    )
    monkeypatch.setattr(
        fmp_schemas,
        "NodeStateFusionOutput",
        FakeNodeStateFusionOutput,
    )
    monkeypatch.setattr(
        fmp_schemas,
        "CompiledRelationRegistry",
        FakeCompiledRelationRegistry,
    )


# =============================================================================
# Controlled source-input and source-output factories
# =============================================================================


def _node_ids() -> tuple[str, ...]:
    return tuple(
        f"node-{index}"
        for index in range(NODES)
    )


def _node_batch_index(
    *,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    return torch.tensor(
        [0, 0, 0, 1, 1],
        dtype=torch.long,
        device=device,
    )


def _edge_index(
    *,
    device: torch.device | str = "cpu",
    empty_edges: bool = False,
) -> torch.Tensor:
    if empty_edges:
        return torch.empty(
            (2, 0),
            dtype=torch.long,
            device=device,
        )

    return torch.tensor(
        [
            [0, 2, 1, 0, 3, 4, 4],
            [1, 1, 2, 2, 4, 4, 3],
        ],
        dtype=torch.long,
        device=device,
    )


def _edge_relations(
    *,
    device: torch.device | str = "cpu",
    empty_edges: bool = False,
) -> torch.Tensor:
    if empty_edges:
        return torch.empty(
            (0,),
            dtype=torch.long,
            device=device,
        )

    return torch.tensor(
        [0, 0, 1, 2, 2, 2, 1],
        dtype=torch.long,
        device=device,
    )


def _edge_batch_index(
    *,
    device: torch.device | str = "cpu",
    empty_edges: bool = False,
) -> torch.Tensor:
    if empty_edges:
        return torch.empty(
            (0,),
            dtype=torch.long,
            device=device,
        )

    return torch.tensor(
        [0, 0, 0, 0, 1, 1, 1],
        dtype=torch.long,
        device=device,
    )


def _state_tensor(
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
    requires_grad: bool = False,
) -> torch.Tensor:
    value = (
        torch.arange(
            NODES * HIDDEN_DIM,
            dtype=dtype,
            device=device,
        )
        .reshape(NODES, HIDDEN_DIM)
        / 10.0
    )
    value = value.detach().clone()

    if requires_grad:
        value.requires_grad_(True)

    return value


def _semantic_weight(
    *,
    num_edges: int = EDGES,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
    requires_grad: bool = False,
) -> torch.Tensor:
    value = torch.linspace(
        0.5,
        1.5,
        num_edges,
        dtype=dtype,
        device=device,
    )
    value = value.detach().clone()

    if requires_grad:
        value.requires_grad_(True)

    return value


def _inputs(
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
    empty_edges: bool = False,
    semantic_edge_weight: (
        torch.Tensor | None | object
    ) = _DEFAULT,
    state: torch.Tensor | None = None,
    source_fingerprint: str = "message-builder-test-input",
) -> FunctionalMessagePassingInputs:
    node_batch = _node_batch_index(
        device=device,
    )
    edge_index = _edge_index(
        device=device,
        empty_edges=empty_edges,
    )
    relation_index = _edge_relations(
        device=device,
        empty_edges=empty_edges,
    )
    edge_batch = _edge_batch_index(
        device=device,
        empty_edges=empty_edges,
    )
    num_edges = int(
        edge_index.shape[1]
    )

    if semantic_edge_weight is _DEFAULT:
        resolved_semantic = _semantic_weight(
            num_edges=num_edges,
            dtype=dtype,
            device=device,
        )
    else:
        resolved_semantic = semantic_edge_weight

    graph = FakeUrbanGraphBatch(
        external_node_ids=_node_ids(),
        node_batch_index=node_batch,
        edge_index=edge_index,
        edge_relation_type=relation_index,
        edge_attributes=torch.zeros(
            (num_edges, 2),
            dtype=dtype,
            device=device,
        ),
        semantic_edge_weight=resolved_semantic,  # type: ignore[arg-type]
        edge_batch_index=edge_batch,
    )

    node_state = FakeNodeStateFusionOutput(
        fused_state=(
            _state_tensor(
                dtype=dtype,
                device=device,
            )
            if state is None
            else state
        ),
        alignment=FakeAlignment(
            item_ids=_node_ids(),
            node_batch_index=node_batch,
            graph_count=GRAPHS,
        ),
    )

    return FunctionalMessagePassingInputs(
        source_graph=graph,
        node_state=node_state,
        compiled_relation_registry=(
            FakeCompiledRelationRegistry()
        ),
        source_fingerprint=source_fingerprint,
    )


def _relation_transform(
    inputs: FunctionalMessagePassingInputs,
    *,
    tensor: torch.Tensor | None = None,
    architecture: str = "relation-transform-architecture",
    parameter: str | None = "relation-transform-parameters",
) -> RelationTransformOutput:
    if tensor is None:
        tensor = (
            torch.arange(
                inputs.num_edges
                * inputs.hidden_dim,
                dtype=inputs.dtype,
                device=inputs.device,
            )
            .reshape(
                inputs.num_edges,
                inputs.hidden_dim,
            )
            / 5.0
        )

    return RelationTransformOutput(
        transformed_source_state=tensor,
        source_inputs=inputs,
        transform_mode="exact_relation_linear",
        encoder_architecture_fingerprint=architecture,
        parameter_fingerprint=parameter,
    )


def _edge_normalization(
    inputs: FunctionalMessagePassingInputs,
    *,
    coefficients: torch.Tensor | None = None,
    architecture: str = "edge-normalization-architecture",
) -> StructuralEdgeNormalizationOutput:
    if coefficients is None:
        coefficients = torch.linspace(
            0.4,
            1.0,
            inputs.num_edges,
            dtype=inputs.dtype,
            device=inputs.device,
        )

    return StructuralEdgeNormalizationOutput(
        coefficients=coefficients,
        source_inputs=inputs,
        normalization_mode=(
            EDGE_NORMALIZATION_NONE
        ),
        encoder_architecture_fingerprint=architecture,
    )


def _relation_gate(
    inputs: FunctionalMessagePassingInputs,
    *,
    architecture: str = "relation-gate-architecture",
    parameter: str | None = "relation-gate-parameters",
) -> RelationGateOutput:
    gate_logits = (
        torch.arange(
            inputs.num_nodes
            * inputs.num_relations,
            dtype=inputs.dtype,
            device=inputs.device,
        )
        .reshape(
            inputs.num_nodes,
            inputs.num_relations,
        )
        / 10.0
        - 0.5
    )
    gate_values = torch.sigmoid(
        gate_logits
    )
    edge_values = gate_values[
        inputs.target_index,
        inputs.edge_relation_index,
    ]

    return RelationGateOutput(
        gate_logits=gate_logits,
        gate_values=gate_values,
        edge_gate_values=edge_values,
        source_inputs=inputs,
        scope=RELATION_GATE_SCOPE_TARGET_NODE,
        activation=RELATION_GATE_ACTIVATION_SIGMOID,
        encoder_architecture_fingerprint=architecture,
        parameter_fingerprint=parameter,
    )


def _attention_weights(
    inputs: FunctionalMessagePassingInputs,
) -> torch.Tensor:
    group_ids = inputs.attention_group_id
    counts = torch.bincount(
        group_ids,
        minlength=inputs.attention_num_groups,
    )
    return (
        torch.ones(
            inputs.num_edges,
            dtype=inputs.dtype,
            device=inputs.device,
        )
        / counts[group_ids].to(
            dtype=inputs.dtype
        )
    )


def _edge_attention(
    inputs: FunctionalMessagePassingInputs,
    *,
    architecture: str = "edge-attention-architecture",
    parameter: str | None = "edge-attention-parameters",
) -> EdgeAttentionOutput:
    group_ids = inputs.attention_group_id
    counts = torch.bincount(
        group_ids,
        minlength=inputs.attention_num_groups,
    )
    one_head = _attention_weights(
        inputs
    )
    normalized = one_head.unsqueeze(
        1
    ).repeat(
        1,
        HEADS,
    )
    raw_scores = torch.zeros_like(
        normalized
    )
    reduced = normalized.mean(
        dim=1
    )

    return EdgeAttentionOutput(
        raw_scores_by_head=raw_scores,
        normalized_weights_by_head=normalized,
        edge_weights=reduced,
        group_ids=group_ids,
        group_counts=counts,
        source_inputs=inputs,
        attention_mode=(
            ATTENTION_MODE_MULTIHEAD_HAZARD_CONDITIONED
        ),
        normalization_mode=(
            ATTENTION_NORMALIZATION_TARGET_NODE_RELATION
        ),
        head_reduction=(
            ATTENTION_HEAD_REDUCTION_MEAN
        ),
        encoder_architecture_fingerprint=architecture,
        parameter_fingerprint=parameter,
    )



# =============================================================================
# Controlled orchestrated components
# =============================================================================


def _json_fingerprint(
    payload: Mapping[str, Any],
) -> str:
    return sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _tensor_value_fingerprint(
    value: torch.Tensor,
) -> str:
    detached = (
        value.detach()
        .cpu()
        .contiguous()
    )
    return sha256(
        detached
        .view(torch.uint8)
        .numpy()
        .tobytes()
    ).hexdigest()


class ControlledRelationTransforms(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int,
        relation_names: tuple[str, ...],
        stable_relation_ids: tuple[int, ...],
        registry_fingerprint: str,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.relation_names = relation_names
        self.stable_relation_ids = stable_relation_ids
        self.compiled_relation_registry_fingerprint = (
            registry_fingerprint
        )
        self.scale = nn.Parameter(
            torch.ones(
                hidden_dim,
                device=device,
                dtype=dtype,
            )
        )

    @classmethod
    def from_config(
        cls,
        *,
        config: Any,
        hidden_dim: int,
        compiled_relation_registry: FakeCompiledRelationRegistry,
        bias: bool,
    ) -> "ControlledRelationTransforms":
        del config, bias
        return cls(
            hidden_dim=hidden_dim,
            relation_names=(
                compiled_relation_registry
                .relation_names
            ),
            stable_relation_ids=(
                compiled_relation_registry
                .stable_relation_ids
            ),
            registry_fingerprint=(
                compiled_relation_registry
                .fingerprint()
            ),
        )

    @property
    def relation_count(self) -> int:
        return len(
            self.relation_names
        )

    @property
    def parameter_count(self) -> int:
        return int(
            self.scale.numel()
        )

    def architecture_dict(
        self,
    ) -> dict[str, Any]:
        return {
            "component": "controlled_relation_transforms",
            "hidden_dim": self.hidden_dim,
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
            "parameter_count": (
                self.parameter_count
            ),
        }

    def architecture_fingerprint(
        self,
    ) -> str:
        return _json_fingerprint(
            self.architecture_dict()
        )

    def parameter_fingerprint(
        self,
    ) -> str:
        return _json_fingerprint(
            {
                "component": (
                    "controlled_relation_transforms"
                ),
                "scale": (
                    _tensor_value_fingerprint(
                        self.scale
                    )
                ),
            }
        )

    def assert_finite_parameters(
        self,
    ) -> None:
        if not bool(
            torch.isfinite(
                self.scale
            )
            .all()
            .item()
        ):
            raise FloatingPointError(
                "Controlled relation-transform scale must be finite."
            )

    def forward(
        self,
        inputs: FunctionalMessagePassingInputs,
    ) -> RelationTransformOutput:
        transformed = (
            inputs
            .node_state
            .fused_state[
                inputs.source_index
            ]
            * self.scale
        )

        return RelationTransformOutput(
            transformed_source_state=(
                transformed
            ),
            source_inputs=inputs,
            transform_mode=(
                "controlled_exact_relation_diagonal"
            ),
            encoder_architecture_fingerprint=(
                self.architecture_fingerprint()
            ),
            parameter_fingerprint=(
                self.parameter_fingerprint()
            ),
        )


class ControlledEdgeNormalization(nn.Module):
    def __init__(
        self,
        *,
        mode: str = EDGE_NORMALIZATION_NONE,
    ) -> None:
        super().__init__()
        self.mode = mode

    @classmethod
    def from_config(
        cls,
        *,
        config: Any,
    ) -> "ControlledEdgeNormalization":
        return cls(
            mode=config.edge_normalization_type
        )

    @property
    def parameter_count(self) -> int:
        return 0

    def architecture_dict(
        self,
    ) -> dict[str, Any]:
        return {
            "component": (
                "controlled_edge_normalization"
            ),
            "mode": self.mode,
            "parameter_count": 0,
        }

    def architecture_fingerprint(
        self,
    ) -> str:
        return _json_fingerprint(
            self.architecture_dict()
        )

    def parameter_fingerprint(
        self,
    ) -> str:
        return _json_fingerprint(
            {
                "component": (
                    "controlled_edge_normalization"
                ),
                "parameter_count": 0,
            }
        )

    def assert_finite_parameters(
        self,
    ) -> None:
        return None

    def forward(
        self,
        inputs: FunctionalMessagePassingInputs,
    ) -> StructuralEdgeNormalizationOutput:
        coefficients = torch.ones(
            inputs.num_edges,
            dtype=inputs.dtype,
            device=inputs.device,
        )

        return StructuralEdgeNormalizationOutput(
            coefficients=coefficients,
            source_inputs=inputs,
            normalization_mode=self.mode,
            encoder_architecture_fingerprint=(
                self.architecture_fingerprint()
            ),
        )


class ControlledRelationGate(nn.Module):
    def __init__(
        self,
        *,
        source_inputs: FunctionalMessagePassingInputs,
    ) -> None:
        super().__init__()
        self.relation_names = (
            source_inputs.relation_names
        )
        self.stable_relation_ids = (
            source_inputs
            .stable_relation_ids
        )
        self.bias = nn.Parameter(
            torch.linspace(
                -0.3,
                0.3,
                source_inputs.num_relations,
                dtype=source_inputs.dtype,
                device=source_inputs.device,
            )
        )

    @classmethod
    def from_config(
        cls,
        *,
        config: Any,
        source_inputs: FunctionalMessagePassingInputs,
        **kwargs: Any,
    ) -> "ControlledRelationGate":
        del config, kwargs
        return cls(
            source_inputs=source_inputs
        )

    @property
    def num_relations(self) -> int:
        return len(
            self.relation_names
        )

    @property
    def parameter_count(self) -> int:
        return int(
            self.bias.numel()
        )

    def architecture_dict(
        self,
    ) -> dict[str, Any]:
        return {
            "component": (
                "controlled_relation_gate"
            ),
            "relation_names": list(
                self.relation_names
            ),
            "stable_relation_ids": list(
                self.stable_relation_ids
            ),
            "scope": (
                RELATION_GATE_SCOPE_TARGET_NODE
            ),
            "activation": (
                RELATION_GATE_ACTIVATION_SIGMOID
            ),
            "parameter_count": (
                self.parameter_count
            ),
        }

    def architecture_fingerprint(
        self,
    ) -> str:
        return _json_fingerprint(
            self.architecture_dict()
        )

    def parameter_fingerprint(
        self,
    ) -> str:
        return _json_fingerprint(
            {
                "component": (
                    "controlled_relation_gate"
                ),
                "bias": (
                    _tensor_value_fingerprint(
                        self.bias
                    )
                ),
            }
        )

    def assert_finite_parameters(
        self,
    ) -> None:
        if not bool(
            torch.isfinite(
                self.bias
            )
            .all()
            .item()
        ):
            raise FloatingPointError(
                "Controlled relation-gate bias must be finite."
            )

    def forward(
        self,
        inputs: FunctionalMessagePassingInputs,
    ) -> RelationGateOutput:
        node_signal = (
            inputs
            .node_state
            .fused_state
            .mean(
                dim=-1,
                keepdim=True,
            )
        )
        logits = (
            node_signal
            + self.bias.unsqueeze(0)
        )
        values = torch.sigmoid(
            logits
        )
        edge_values = values[
            inputs.target_index,
            inputs.edge_relation_index,
        ]

        return RelationGateOutput(
            gate_logits=logits,
            gate_values=values,
            edge_gate_values=(
                edge_values
            ),
            source_inputs=inputs,
            scope=(
                RELATION_GATE_SCOPE_TARGET_NODE
            ),
            activation=(
                RELATION_GATE_ACTIVATION_SIGMOID
            ),
            encoder_architecture_fingerprint=(
                self.architecture_fingerprint()
            ),
            parameter_fingerprint=(
                self.parameter_fingerprint()
            ),
            regularization_terms={
                "mean_gate": (
                    values.mean()
                    * 0.01
                )
            },
        )


def _grouped_softmax(
    scores: torch.Tensor,
    group_ids: torch.Tensor,
) -> torch.Tensor:
    if scores.numel() == 0:
        return scores

    output = torch.zeros_like(
        scores
    )

    for group_id in torch.unique(
        group_ids,
        sorted=True,
    ):
        indices = torch.nonzero(
            group_ids == group_id,
            as_tuple=False,
        ).squeeze(-1)
        output = output.index_copy(
            0,
            indices,
            torch.softmax(
                scores[indices],
                dim=0,
            ),
        )

    return output


class ControlledEdgeAttention(nn.Module):
    def __init__(
        self,
        *,
        source_inputs: FunctionalMessagePassingInputs,
        num_heads: int = HEADS,
    ) -> None:
        super().__init__()
        self.relation_names = (
            source_inputs.relation_names
        )
        self.stable_relation_ids = (
            source_inputs
            .stable_relation_ids
        )
        self.num_heads = num_heads
        self.head_scale = nn.Parameter(
            torch.linspace(
                0.7,
                1.3,
                num_heads,
                dtype=source_inputs.dtype,
                device=source_inputs.device,
            )
        )

    @classmethod
    def from_config(
        cls,
        *,
        config: Any,
        source_inputs: FunctionalMessagePassingInputs,
    ) -> "ControlledEdgeAttention":
        return cls(
            source_inputs=source_inputs,
            num_heads=config.attention_heads,
        )

    @property
    def num_relations(self) -> int:
        return len(
            self.relation_names
        )

    @property
    def parameter_count(self) -> int:
        return int(
            self.head_scale.numel()
        )

    def architecture_dict(
        self,
    ) -> dict[str, Any]:
        return {
            "component": (
                "controlled_edge_attention"
            ),
            "relation_names": list(
                self.relation_names
            ),
            "stable_relation_ids": list(
                self.stable_relation_ids
            ),
            "num_heads": (
                self.num_heads
            ),
            "normalization": (
                ATTENTION_NORMALIZATION_TARGET_NODE_RELATION
            ),
            "head_reduction": (
                ATTENTION_HEAD_REDUCTION_MEAN
            ),
            "parameter_count": (
                self.parameter_count
            ),
        }

    def architecture_fingerprint(
        self,
    ) -> str:
        return _json_fingerprint(
            self.architecture_dict()
        )

    def parameter_fingerprint(
        self,
    ) -> str:
        return _json_fingerprint(
            {
                "component": (
                    "controlled_edge_attention"
                ),
                "head_scale": (
                    _tensor_value_fingerprint(
                        self.head_scale
                    )
                ),
            }
        )

    def assert_finite_parameters(
        self,
    ) -> None:
        if not bool(
            torch.isfinite(
                self.head_scale
            )
            .all()
            .item()
        ):
            raise FloatingPointError(
                "Controlled attention scales must be finite."
            )

    def forward(
        self,
        inputs: FunctionalMessagePassingInputs,
    ) -> EdgeAttentionOutput:
        source_signal = (
            inputs
            .node_state
            .fused_state[
                inputs.source_index
            ]
            .mean(
                dim=-1,
            )
        )
        target_signal = (
            inputs
            .node_state
            .fused_state[
                inputs.target_index
            ]
            .mean(
                dim=-1,
            )
        )
        base = (
            source_signal
            + 0.25 * target_signal
        )
        raw_scores = (
            base.unsqueeze(-1)
            * self
            .head_scale
            .unsqueeze(0)
        )
        weights_by_head = torch.stack(
            tuple(
                _grouped_softmax(
                    raw_scores[:, head],
                    inputs.attention_group_id,
                )
                for head in range(
                    self.num_heads
                )
            ),
            dim=1,
        )
        edge_weights = (
            weights_by_head.mean(
                dim=1
            )
        )
        group_counts = torch.bincount(
            inputs.attention_group_id,
            minlength=(
                inputs
                .attention_num_groups
            ),
        )

        return EdgeAttentionOutput(
            raw_scores_by_head=(
                raw_scores
            ),
            normalized_weights_by_head=(
                weights_by_head
            ),
            edge_weights=edge_weights,
            group_ids=(
                inputs
                .attention_group_id
            ),
            group_counts=group_counts,
            source_inputs=inputs,
            attention_mode=(
                ATTENTION_MODE_MULTIHEAD_HAZARD_CONDITIONED
            ),
            normalization_mode=(
                ATTENTION_NORMALIZATION_TARGET_NODE_RELATION
            ),
            head_reduction=(
                ATTENTION_HEAD_REDUCTION_MEAN
            ),
            encoder_architecture_fingerprint=(
                self.architecture_fingerprint()
            ),
            parameter_fingerprint=(
                self.parameter_fingerprint()
            ),
        )


class ControlledMessageAggregator(nn.Module):
    def __init__(
        self,
        *,
        mode: str = AGGREGATION_MEAN,
    ) -> None:
        super().__init__()
        self.mode = mode

    @classmethod
    def from_config(
        cls,
        *,
        config: Any,
    ) -> "ControlledMessageAggregator":
        return cls(
            mode=config.aggregation_type
        )

    @property
    def parameter_count(self) -> int:
        return 0

    def architecture_dict(
        self,
    ) -> dict[str, Any]:
        return {
            "component": (
                "controlled_message_aggregator"
            ),
            "mode": self.mode,
            "parameter_count": 0,
        }

    def architecture_fingerprint(
        self,
    ) -> str:
        return _json_fingerprint(
            self.architecture_dict()
        )

    def parameter_fingerprint(
        self,
    ) -> str:
        return _json_fingerprint(
            {
                "component": (
                    "controlled_message_aggregator"
                ),
                "parameter_count": 0,
            }
        )

    def assert_finite_parameters(
        self,
    ) -> None:
        return None

    def forward(
        self,
        messages: EdgeMessageOutput,
    ) -> AggregationOutput:
        inputs = messages.source_inputs
        aggregate_sum = torch.zeros(
            (
                inputs.num_nodes,
                inputs.hidden_dim,
            ),
            dtype=messages.edge_messages.dtype,
            device=messages.edge_messages.device,
        )
        aggregate_sum.index_add_(
            0,
            inputs.target_index,
            messages.edge_messages,
        )
        counts = torch.bincount(
            inputs.target_index,
            minlength=inputs.num_nodes,
        )
        aggregate = aggregate_sum / (
            counts
            .clamp_min(1)
            .to(
                dtype=aggregate_sum.dtype
            )
            .unsqueeze(-1)
        )

        return AggregationOutput(
            node_aggregate=aggregate,
            incoming_edge_count=counts,
            source_messages=messages,
            aggregation_mode=self.mode,
            encoder_architecture_fingerprint=(
                self.architecture_fingerprint()
            ),
        )


@dataclass
class ControlledFunctionalMessagePassingConfig:
    enabled: bool = True
    relation_transform_type: str = "controlled"
    aggregation_type: str = AGGREGATION_MEAN
    edge_normalization_type: str = EDGE_NORMALIZATION_NONE
    attention_enabled: bool = True
    attention_heads: int = HEADS
    residual: bool = True
    layer_norm: bool = True
    dropout: float = 0.0
    capture_intermediate_messages: bool = False

    def validate(self) -> None:
        if not isinstance(
            self.enabled,
            bool,
        ):
            raise TypeError(
                "enabled must be Boolean."
            )

    def assert_implemented(self) -> None:
        return None


@dataclass
class ControlledRelationConfig:
    gate_enabled: bool = True

    def validate(self) -> None:
        if not isinstance(
            self.gate_enabled,
            bool,
        ):
            raise TypeError(
                "gate_enabled must be Boolean."
            )


@pytest.fixture(autouse=True)
def _patch_layer_component_types(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        layer_module,
        "RelationTransforms",
        ControlledRelationTransforms,
    )
    monkeypatch.setattr(
        layer_module,
        "EdgeNormalization",
        ControlledEdgeNormalization,
    )
    monkeypatch.setattr(
        layer_module,
        "RelationFamilyGate",
        ControlledRelationGate,
    )
    monkeypatch.setattr(
        layer_module,
        "EdgeAttention",
        ControlledEdgeAttention,
    )
    monkeypatch.setattr(
        layer_module,
        "MessageAggregator",
        ControlledMessageAggregator,
    )
    monkeypatch.setattr(
        layer_module,
        "FunctionalMessagePassingConfig",
        ControlledFunctionalMessagePassingConfig,
    )
    monkeypatch.setattr(
        layer_module,
        "RelationConfig",
        ControlledRelationConfig,
    )


# =============================================================================
# Layer factories and analytical helpers
# =============================================================================


def _explicit_layer(
    inputs: FunctionalMessagePassingInputs,
    *,
    gate_enabled: bool = True,
    attention_enabled: bool = True,
    residual_enabled: bool = True,
    layer_norm_enabled: bool = True,
    dropout_probability: float = 0.0,
    trace_mode: str = LAYER_TRACE_NONE,
    diagnostics_enabled: bool = False,
    semantic_edge_policy: str = (
        MESSAGE_SEMANTIC_EDGE_POLICY_IGNORE
    ),
) -> FunctionalMessagePassingLayer:
    transforms = (
        ControlledRelationTransforms(
            hidden_dim=(
                inputs.hidden_dim
            ),
            relation_names=(
                inputs.relation_names
            ),
            stable_relation_ids=(
                inputs.stable_relation_ids
            ),
            registry_fingerprint=(
                inputs
                .compiled_relation_registry
                .fingerprint()
            ),
            device=inputs.device,
            dtype=inputs.dtype,
        )
    )
    edge_normalization = (
        ControlledEdgeNormalization()
    )
    gate = (
        ControlledRelationGate(
            source_inputs=inputs
        )
        if gate_enabled
        else None
    )
    attention = (
        ControlledEdgeAttention(
            source_inputs=inputs
        )
        if attention_enabled
        else None
    )
    message_builder = (
        build_edge_message_builder(
            semantic_edge_policy=(
                semantic_edge_policy
            ),
            diagnostics_enabled=False,
        )
    )
    aggregator = (
        ControlledMessageAggregator()
    )
    residual_updater = (
        build_layer_residual_updater_from_flags(
            residual_enabled=(
                residual_enabled
            ),
            dropout_probability=(
                dropout_probability
            ),
        )
    )
    normalizer = (
        build_layer_normalizer_from_flag(
            inputs.hidden_dim,
            layer_norm_enabled=(
                layer_norm_enabled
            ),
            device=inputs.device,
            dtype=inputs.dtype,
        )
    )
    diagnostics = (
        build_layer_diagnostics(
            include_per_graph=True,
            include_edge_report=True,
        )
        if diagnostics_enabled
        else None
    )

    return FunctionalMessagePassingLayer(
        relation_transforms=transforms,
        edge_normalization=(
            edge_normalization
        ),
        relation_gate=gate,
        edge_attention=attention,
        message_builder=message_builder,
        aggregator=aggregator,
        residual_updater=(
            residual_updater
        ),
        normalizer=normalizer,
        default_trace_policy=(
            LayerTracePolicy(
                mode=trace_mode
            )
        ),
        diagnostics=diagnostics,
    )


def _configured_layer(
    inputs: FunctionalMessagePassingInputs,
    *,
    gate_enabled: bool = True,
    attention_enabled: bool = True,
    residual_enabled: bool = True,
    layer_norm_enabled: bool = True,
    dropout_probability: float = 0.0,
    capture_intermediates: bool = False,
    diagnostics_enabled: bool = False,
) -> FunctionalMessagePassingLayer:
    config = (
        ControlledFunctionalMessagePassingConfig(
            enabled=True,
            attention_enabled=(
                attention_enabled
            ),
            residual=(
                residual_enabled
            ),
            layer_norm=(
                layer_norm_enabled
            ),
            dropout=(
                dropout_probability
            ),
            capture_intermediate_messages=(
                capture_intermediates
            ),
        )
    )
    relation_config = (
        ControlledRelationConfig(
            gate_enabled=(
                gate_enabled
            )
        )
    )

    return (
        FunctionalMessagePassingLayer
        .from_config(
            config=config,
            relation_config=(
                relation_config
            ),
            source_inputs=inputs,
            diagnostics_enabled=(
                diagnostics_enabled
            ),
        )
    )


def _expected_mean_aggregate(
    messages: EdgeMessageOutput,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
]:
    inputs = messages.source_inputs
    total = torch.zeros(
        (
            inputs.num_nodes,
            inputs.hidden_dim,
        ),
        dtype=inputs.dtype,
        device=inputs.device,
    )
    total.index_add_(
        0,
        inputs.target_index,
        messages.edge_messages,
    )
    counts = torch.bincount(
        inputs.target_index,
        minlength=inputs.num_nodes,
    )
    mean = total / (
        counts
        .clamp_min(1)
        .to(
            dtype=inputs.dtype
        )
        .unsqueeze(-1)
    )
    return mean, counts


def _assert_tensor_free(
    value: Any,
) -> None:
    if isinstance(
        value,
        torch.Tensor,
    ):
        raise AssertionError(
            "Diagnostic report retained a tensor."
        )

    if isinstance(
        value,
        nn.Module,
    ):
        raise AssertionError(
            "Diagnostic report retained a module."
        )

    if isinstance(
        value,
        Mapping,
    ):
        for nested in value.values():
            _assert_tensor_free(
                nested
            )
        return

    if isinstance(
        value,
        (list, tuple),
    ):
        for nested in value:
            _assert_tensor_free(
                nested
            )



# =============================================================================
# Failure-suite helpers
# =============================================================================


def _component_kwargs(
    inputs: FunctionalMessagePassingInputs,
    *,
    gate_enabled: bool = True,
    attention_enabled: bool = True,
    diagnostics_enabled: bool = False,
) -> dict[str, Any]:
    reference = _explicit_layer(
        inputs,
        gate_enabled=gate_enabled,
        attention_enabled=(
            attention_enabled
        ),
        diagnostics_enabled=(
            diagnostics_enabled
        ),
    )

    return {
        "relation_transforms": (
            reference.relation_transforms
        ),
        "edge_normalization": (
            reference.edge_normalization
        ),
        "relation_gate": (
            reference.relation_gate
        ),
        "edge_attention": (
            reference.edge_attention
        ),
        "message_builder": (
            reference.message_builder
        ),
        "aggregator": (
            reference.aggregator
        ),
        "residual_updater": (
            reference.residual_updater
        ),
        "normalizer": (
            reference.normalizer
        ),
        "default_trace_policy": (
            reference.default_trace_policy
        ),
        "diagnostics": (
            reference.diagnostics
        ),
    }


def _full_run(
    *,
    inputs: FunctionalMessagePassingInputs | None = None,
    diagnostics_enabled: bool = False,
    gate_enabled: bool = True,
    attention_enabled: bool = True,
) -> tuple[
    FunctionalMessagePassingLayer,
    FunctionalMessagePassingInputs,
    FunctionalMessagePassingLayerRun,
]:
    source_inputs = (
        _inputs()
        if inputs is None
        else inputs
    )
    layer = _explicit_layer(
        source_inputs,
        gate_enabled=gate_enabled,
        attention_enabled=(
            attention_enabled
        ),
        trace_mode=LAYER_TRACE_FULL,
        diagnostics_enabled=(
            diagnostics_enabled
        ),
    )
    run = layer.run_complete(
        source_inputs,
        layer_index=2,
        source_stack_fingerprint=(
            "failure-suite-stack"
        ),
    )
    return layer, source_inputs, run


def _different_source_inputs(
    *,
    source_fingerprint: str = (
        "different-source-input"
    ),
) -> FunctionalMessagePassingInputs:
    return _inputs(
        source_fingerprint=(
            source_fingerprint
        )
    )


def _tamper(
    value: Any,
    name: str,
    replacement: Any,
) -> Any:
    object.__setattr__(
        value,
        name,
        replacement,
    )
    return value


def _valid_node_stages(
    layer: FunctionalMessagePassingLayer,
    inputs: FunctionalMessagePassingInputs,
) -> tuple[
    FunctionalMessagePassingLayerInputs,
    FunctionalMessagePassingLayerEdgeStages,
    FunctionalMessagePassingLayerNodeStages,
]:
    layer_inputs = (
        layer.build_layer_inputs(
            inputs,
            layer_index=0,
            trace_policy=LAYER_TRACE_FULL,
        )
    )
    edge_stages = (
        layer.compute_edge_stages(
            inputs
        )
    )
    node_stages = (
        layer.compute_node_stages(
            edge_stages=edge_stages,
            layer_inputs=layer_inputs,
        )
    )
    return (
        layer_inputs,
        edge_stages,
        node_stages,
    )


# =============================================================================
# Constructor type failures
# =============================================================================


@pytest.mark.parametrize(
    "component_name",
    (
        "relation_transforms",
        "edge_normalization",
        "relation_gate",
        "edge_attention",
        "message_builder",
        "aggregator",
        "residual_updater",
        "normalizer",
        "diagnostics",
    ),
)
def test_layer_constructor_rejects_wrong_component_types(
    component_name: str,
) -> None:
    inputs = _inputs()
    kwargs = _component_kwargs(
        inputs,
        diagnostics_enabled=True,
    )
    kwargs[component_name] = object()

    with pytest.raises(
        TypeError,
    ):
        FunctionalMessagePassingLayer(
            **kwargs
        )


def test_layer_constructor_rejects_wrong_trace_policy_type() -> None:
    inputs = _inputs()
    kwargs = _component_kwargs(
        inputs
    )
    kwargs[
        "default_trace_policy"
    ] = "full"

    with pytest.raises(
        TypeError,
    ):
        FunctionalMessagePassingLayer(
            **kwargs
        )


def test_explicit_builder_rejects_wrong_component_type() -> None:
    inputs = _inputs()
    kwargs = _component_kwargs(
        inputs
    )
    kwargs["aggregator"] = object()

    with pytest.raises(
        TypeError,
    ):
        build_functional_message_passing_layer(
            **kwargs
        )


# =============================================================================
# Static cross-component incompatibilities
# =============================================================================


def test_layer_rejects_normalizer_hidden_width_mismatch() -> None:
    inputs = _inputs()
    kwargs = _component_kwargs(
        inputs
    )
    kwargs["normalizer"] = (
        LayerNormalizer(
            HIDDEN_DIM + 1
        )
    )

    with pytest.raises(
        ValueError,
    ):
        FunctionalMessagePassingLayer(
            **kwargs
        )


def test_layer_rejects_pre_residual_normalization_position() -> None:
    inputs = _inputs()
    kwargs = _component_kwargs(
        inputs
    )
    normalizer = kwargs[
        "normalizer"
    ]
    _tamper(
        normalizer,
        "normalization_position",
        LAYER_NORMALIZATION_PRE_RESIDUAL,
    )

    with pytest.raises(
        ValueError,
    ):
        FunctionalMessagePassingLayer(
            **kwargs
        )


def test_layer_rejects_gate_relation_order_mismatch() -> None:
    inputs = _inputs()
    kwargs = _component_kwargs(
        inputs
    )
    gate = kwargs[
        "relation_gate"
    ]
    assert gate is not None
    gate.relation_names = tuple(
        reversed(
            gate.relation_names
        )
    )

    with pytest.raises(
        ValueError,
    ):
        FunctionalMessagePassingLayer(
            **kwargs
        )


def test_layer_rejects_gate_stable_id_mismatch() -> None:
    inputs = _inputs()
    kwargs = _component_kwargs(
        inputs
    )
    gate = kwargs[
        "relation_gate"
    ]
    assert gate is not None
    gate.stable_relation_ids = (
        1,
        2,
        3,
    )

    with pytest.raises(
        ValueError,
    ):
        FunctionalMessagePassingLayer(
            **kwargs
        )


def test_layer_rejects_attention_relation_order_mismatch() -> None:
    inputs = _inputs()
    kwargs = _component_kwargs(
        inputs
    )
    attention = kwargs[
        "edge_attention"
    ]
    assert attention is not None
    attention.relation_names = tuple(
        reversed(
            attention.relation_names
        )
    )

    with pytest.raises(
        ValueError,
    ):
        FunctionalMessagePassingLayer(
            **kwargs
        )


def test_layer_rejects_attention_stable_id_mismatch() -> None:
    inputs = _inputs()
    kwargs = _component_kwargs(
        inputs
    )
    attention = kwargs[
        "edge_attention"
    ]
    assert attention is not None
    attention.stable_relation_ids = (
        10,
        20,
        30,
    )

    with pytest.raises(
        ValueError,
    ):
        FunctionalMessagePassingLayer(
            **kwargs
        )


# =============================================================================
# Configuration-construction failures
# =============================================================================


def test_from_config_rejects_wrong_config_type() -> None:
    inputs = _inputs()

    with pytest.raises(
        TypeError,
    ):
        FunctionalMessagePassingLayer.from_config(
            config=object(),
            relation_config=(
                ControlledRelationConfig()
            ),
            source_inputs=inputs,
        )


def test_from_config_rejects_wrong_relation_config_type() -> None:
    inputs = _inputs()

    with pytest.raises(
        TypeError,
    ):
        FunctionalMessagePassingLayer.from_config(
            config=(
                ControlledFunctionalMessagePassingConfig()
            ),
            relation_config=object(),
            source_inputs=inputs,
        )


def test_from_config_rejects_wrong_source_input_type() -> None:
    with pytest.raises(
        TypeError,
    ):
        FunctionalMessagePassingLayer.from_config(
            config=(
                ControlledFunctionalMessagePassingConfig()
            ),
            relation_config=(
                ControlledRelationConfig()
            ),
            source_inputs=object(),
        )


def test_from_config_rejects_disabled_functional_message_passing() -> None:
    inputs = _inputs()
    config = (
        ControlledFunctionalMessagePassingConfig(
            enabled=False
        )
    )

    with pytest.raises(
        ValueError,
    ):
        FunctionalMessagePassingLayer.from_config(
            config=config,
            relation_config=(
                ControlledRelationConfig()
            ),
            source_inputs=inputs,
        )


@pytest.mark.parametrize(
    ("argument", "value"),
    (
        (
            "relation_transform_bias",
            1,
        ),
        (
            "relation_gate_use_node_state",
            1,
        ),
        (
            "relation_gate_use_hazard_query",
            "yes",
        ),
        (
            "relation_gate_layer_norm",
            0,
        ),
        (
            "relation_gate_bias",
            None,
        ),
        (
            "layer_norm_elementwise_affine",
            1,
        ),
        (
            "layer_norm_bias_enabled",
            "true",
        ),
        (
            "diagnostics_enabled",
            1,
        ),
        (
            "diagnostics_include_per_graph",
            1,
        ),
        (
            "diagnostics_include_edge_report",
            1,
        ),
    ),
)
def test_from_config_rejects_non_boolean_overrides(
    argument: str,
    value: Any,
) -> None:
    inputs = _inputs()
    kwargs = {
        argument: value
    }

    with pytest.raises(
        TypeError,
    ):
        FunctionalMessagePassingLayer.from_config(
            config=(
                ControlledFunctionalMessagePassingConfig()
            ),
            relation_config=(
                ControlledRelationConfig()
            ),
            source_inputs=inputs,
            **kwargs,
        )


@pytest.mark.parametrize(
    "epsilon",
    (
        0.0,
        -1e-5,
        float("inf"),
        float("nan"),
    ),
)
def test_from_config_rejects_invalid_layer_norm_epsilon(
    epsilon: float,
) -> None:
    inputs = _inputs()

    with pytest.raises(
        (TypeError, ValueError),
    ):
        FunctionalMessagePassingLayer.from_config(
            config=(
                ControlledFunctionalMessagePassingConfig()
            ),
            relation_config=(
                ControlledRelationConfig()
            ),
            source_inputs=inputs,
            layer_norm_epsilon=epsilon,
        )


@pytest.mark.parametrize(
    "epsilon",
    (
        0.0,
        -1e-4,
        float("inf"),
        float("nan"),
    ),
)
def test_from_config_rejects_invalid_relation_prior_epsilon(
    epsilon: float,
) -> None:
    inputs = _inputs()

    with pytest.raises(
        ValueError,
    ):
        FunctionalMessagePassingLayer.from_config(
            config=(
                ControlledFunctionalMessagePassingConfig()
            ),
            relation_config=(
                ControlledRelationConfig()
            ),
            source_inputs=inputs,
            relation_prior_epsilon=epsilon,
        )


def test_from_config_rejects_wrong_diagnostic_threshold_type() -> None:
    inputs = _inputs()

    with pytest.raises(
        TypeError,
    ):
        FunctionalMessagePassingLayer.from_config(
            config=(
                ControlledFunctionalMessagePassingConfig()
            ),
            relation_config=(
                ControlledRelationConfig()
            ),
            source_inputs=inputs,
            diagnostics_enabled=True,
            diagnostic_thresholds=object(),
        )


# =============================================================================
# Runtime identity and source-input failures
# =============================================================================


@pytest.mark.parametrize(
    "mode",
    (
        "",
        "edge",
        "all",
    ),
)
def test_trace_policy_rejects_unknown_modes(
    mode: str,
) -> None:
    with pytest.raises(
        ValueError,
    ):
        LayerTracePolicy(
            mode=mode
        )


def test_trace_policy_capture_constructor_rejects_non_boolean() -> None:
    with pytest.raises(
        TypeError,
    ):
        LayerTracePolicy.from_capture_intermediate_messages(
            1
        )


def test_resolve_trace_policy_rejects_conflicting_interfaces() -> None:
    layer = _explicit_layer(
        _inputs()
    )

    with pytest.raises(
        ValueError,
    ):
        layer.resolve_trace_policy(
            trace_policy=LAYER_TRACE_NODE,
            capture_intermediate_messages=True,
        )


@pytest.mark.parametrize(
    "trace_policy",
    (
        1,
        object(),
        False,
    ),
)
def test_resolve_trace_policy_rejects_wrong_type(
    trace_policy: Any,
) -> None:
    layer = _explicit_layer(
        _inputs()
    )

    with pytest.raises(
        TypeError,
    ):
        layer.resolve_trace_policy(
            trace_policy=trace_policy
        )


@pytest.mark.parametrize(
    "layer_index",
    (
        -1,
        True,
        1.5,
        "2",
    ),
)
def test_build_layer_inputs_rejects_invalid_layer_index(
    layer_index: Any,
) -> None:
    inputs = _inputs()
    layer = _explicit_layer(
        inputs
    )

    with pytest.raises(
        (TypeError, ValueError),
    ):
        layer.build_layer_inputs(
            inputs,
            layer_index=layer_index,
        )


@pytest.mark.parametrize(
    "fingerprint",
    (
        "",
        "   ",
    ),
)
def test_build_layer_inputs_rejects_empty_stack_fingerprint(
    fingerprint: str,
) -> None:
    inputs = _inputs()
    layer = _explicit_layer(
        inputs
    )

    with pytest.raises(
        ValueError,
    ):
        layer.build_layer_inputs(
            inputs,
            layer_index=0,
            source_stack_fingerprint=(
                fingerprint
            ),
        )


def test_run_from_layer_inputs_rejects_train_eval_mismatch() -> None:
    inputs = _inputs()
    layer = _explicit_layer(
        inputs
    )
    layer_inputs = (
        layer.build_layer_inputs(
            inputs,
            layer_index=0,
        )
    )
    layer.eval()

    with pytest.raises(
        ValueError,
    ):
        layer.run_from_layer_inputs(
            layer_inputs
        )


def test_source_validation_rejects_wrong_type() -> None:
    layer = _explicit_layer(
        _inputs()
    )

    with pytest.raises(
        TypeError,
    ):
        layer.run_complete(
            object()
        )


def test_source_validation_rejects_hidden_width_mismatch() -> None:
    reference_inputs = _inputs()
    layer = _explicit_layer(
        reference_inputs
    )
    wider_state = torch.randn(
        NODES,
        HIDDEN_DIM + 1,
    )
    mismatched_inputs = _inputs(
        state=wider_state,
        source_fingerprint=(
            "wider-state"
        ),
    )

    with pytest.raises(
        ValueError,
    ):
        layer.run_complete(
            mismatched_inputs
        )


def test_source_validation_rejects_relation_order_mismatch() -> None:
    reference_inputs = _inputs()
    layer = _explicit_layer(
        reference_inputs
    )
    mismatched_inputs = (
        _different_source_inputs()
    )
    registry = (
        mismatched_inputs
        .compiled_relation_registry
    )
    registry.relation_names = tuple(
        reversed(
            registry.relation_names
        )
    )

    with pytest.raises(
        ValueError,
    ):
        layer.run_complete(
            mismatched_inputs
        )


def test_source_validation_rejects_stable_relation_id_mismatch() -> None:
    reference_inputs = _inputs()
    layer = _explicit_layer(
        reference_inputs
    )
    mismatched_inputs = (
        _different_source_inputs()
    )
    mismatched_inputs.compiled_relation_registry.stable_relation_ids = (
        7,
        8,
        9,
    )

    with pytest.raises(
        ValueError,
    ):
        layer.run_complete(
            mismatched_inputs
        )


def test_source_validation_rejects_registry_fingerprint_mismatch() -> None:
    reference_inputs = _inputs()
    layer = _explicit_layer(
        reference_inputs
    )
    mismatched_inputs = (
        _different_source_inputs()
    )
    mismatched_inputs.compiled_relation_registry._fingerprint = (
        "different-registry"
    )

    with pytest.raises(
        ValueError,
    ):
        layer.run_complete(
            mismatched_inputs
        )


def test_source_validation_rejects_child_training_mode_divergence() -> None:
    inputs = _inputs()
    layer = _explicit_layer(
        inputs
    )
    layer.train()
    layer.residual_updater.eval()

    with pytest.raises(
        RuntimeError,
    ):
        layer.run_complete(
            inputs
        )


# =============================================================================
# Non-finite parameter failures
# =============================================================================


@pytest.mark.parametrize(
    "component",
    (
        "relation_transforms",
        "relation_gate",
        "edge_attention",
        "normalizer_weight",
        "normalizer_bias",
    ),
)
def test_layer_rejects_non_finite_parameters(
    component: str,
) -> None:
    inputs = _inputs()
    layer = _explicit_layer(
        inputs
    )

    with torch.no_grad():
        if component == (
            "relation_transforms"
        ):
            layer.relation_transforms.scale[
                0
            ] = float("nan")
        elif component == (
            "relation_gate"
        ):
            assert (
                layer.relation_gate
                is not None
            )
            layer.relation_gate.bias[
                0
            ] = float("inf")
        elif component == (
            "edge_attention"
        ):
            assert (
                layer.edge_attention
                is not None
            )
            layer.edge_attention.head_scale[
                0
            ] = float("nan")
        elif component == (
            "normalizer_weight"
        ):
            assert (
                layer.normalizer.weight
                is not None
            )
            layer.normalizer.weight[
                0
            ] = float("inf")
        else:
            assert (
                layer.normalizer.bias
                is not None
            )
            layer.normalizer.bias[
                0
            ] = float("nan")

    with pytest.raises(
        FloatingPointError,
    ):
        layer.run_complete(
            inputs
        )


# =============================================================================
# Edge-stage lineage and presence failures
# =============================================================================


def test_edge_stage_rejects_transform_from_different_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _inputs()
    foreign_inputs = (
        _different_source_inputs()
    )
    layer = _explicit_layer(
        inputs
    )

    def bad_forward(
        _inputs_value: FunctionalMessagePassingInputs,
    ) -> RelationTransformOutput:
        return (
            layer.relation_transforms(
                foreign_inputs
            )
        )

    original = (
        layer.relation_transforms.forward
    )

    def replacement(
        _inputs_value: FunctionalMessagePassingInputs,
    ) -> RelationTransformOutput:
        return original(
            foreign_inputs
        )

    monkeypatch.setattr(
        layer.relation_transforms,
        "forward",
        replacement,
    )

    with pytest.raises(
        ValueError,
    ):
        layer.compute_edge_stages(
            inputs
        )


def test_edge_stage_rejects_normalization_from_different_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _inputs()
    foreign_inputs = (
        _different_source_inputs()
    )
    layer = _explicit_layer(
        inputs
    )
    original = (
        layer.edge_normalization.forward
    )

    def replacement(
        _inputs_value: FunctionalMessagePassingInputs,
    ) -> StructuralEdgeNormalizationOutput:
        return original(
            foreign_inputs
        )

    monkeypatch.setattr(
        layer.edge_normalization,
        "forward",
        replacement,
    )

    with pytest.raises(
        ValueError,
    ):
        layer.compute_edge_stages(
            inputs
        )


def test_enabled_gate_rejects_missing_gate_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _inputs()
    layer = _explicit_layer(
        inputs,
        gate_enabled=True,
    )
    assert layer.relation_gate is not None

    monkeypatch.setattr(
        layer.relation_gate,
        "forward",
        lambda _inputs_value: None,
    )

    with pytest.raises(
        ValueError,
    ):
        layer.compute_edge_stages(
            inputs
        )


def test_enabled_attention_rejects_missing_attention_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _inputs()
    layer = _explicit_layer(
        inputs,
        attention_enabled=True,
    )
    assert layer.edge_attention is not None

    monkeypatch.setattr(
        layer.edge_attention,
        "forward",
        lambda _inputs_value: None,
    )

    with pytest.raises(
        ValueError,
    ):
        layer.compute_edge_stages(
            inputs
        )


def test_disabled_gate_rejects_present_gate_stage() -> None:
    inputs = _inputs()
    enabled_layer = _explicit_layer(
        inputs,
        gate_enabled=True,
    )
    disabled_layer = _explicit_layer(
        inputs,
        gate_enabled=False,
    )
    stages = (
        enabled_layer
        .compute_edge_stages(
            inputs
        )
    )

    with pytest.raises(
        ValueError,
    ):
        disabled_layer._validate_edge_stages(
            stages,
            source_inputs=inputs,
        )


def test_disabled_attention_rejects_present_attention_stage() -> None:
    inputs = _inputs()
    enabled_layer = _explicit_layer(
        inputs,
        attention_enabled=True,
    )
    disabled_layer = _explicit_layer(
        inputs,
        attention_enabled=False,
    )
    stages = (
        enabled_layer
        .compute_edge_stages(
            inputs
        )
    )

    with pytest.raises(
        ValueError,
    ):
        disabled_layer._validate_edge_stages(
            stages,
            source_inputs=inputs,
        )


def test_edge_stage_rejects_foreign_message_builder_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _inputs()
    layer = _explicit_layer(
        inputs
    )
    foreign_inputs = (
        _different_source_inputs()
    )
    foreign_layer = _explicit_layer(
        foreign_inputs
    )
    foreign_stages = (
        foreign_layer
        .compute_edge_stages(
            foreign_inputs
        )
    )

    monkeypatch.setattr(
        layer.message_builder,
        "run_complete",
        lambda **_kwargs: (
            foreign_stages
            .message_builder_run
        ),
    )

    with pytest.raises(
        ValueError,
    ):
        layer.compute_edge_stages(
            inputs
        )


# =============================================================================
# Node-stage lineage failures
# =============================================================================


def test_node_stage_rejects_foreign_aggregation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _inputs()
    layer = _explicit_layer(
        inputs
    )
    layer_inputs, edge_stages, _ = (
        _valid_node_stages(
            layer,
            inputs,
        )
    )

    foreign_inputs = (
        _different_source_inputs()
    )
    foreign_layer = _explicit_layer(
        foreign_inputs
    )
    foreign_run = (
        foreign_layer.run_complete(
            foreign_inputs
        )
    )

    monkeypatch.setattr(
        layer.aggregator,
        "forward",
        lambda _messages: (
            foreign_run
            .node_stages
            .aggregation
        ),
    )

    with pytest.raises(
        ValueError,
    ):
        layer.compute_node_stages(
            edge_stages=edge_stages,
            layer_inputs=layer_inputs,
        )


def test_node_stage_rejects_stale_residual_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _inputs()
    layer = _explicit_layer(
        inputs
    )
    layer_inputs, edge_stages, _ = (
        _valid_node_stages(
            layer,
            inputs,
        )
    )
    stale_run = layer.run_complete(
        inputs,
        layer_index=9,
    )

    monkeypatch.setattr(
        layer.residual_updater,
        "forward",
        lambda **_kwargs: (
            stale_run
            .node_stages
            .residual_update
        ),
    )

    with pytest.raises(
        ValueError,
    ):
        layer.compute_node_stages(
            edge_stages=edge_stages,
            layer_inputs=layer_inputs,
        )


def test_node_stage_rejects_stale_normalization_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _inputs()
    layer = _explicit_layer(
        inputs
    )
    layer_inputs, edge_stages, _ = (
        _valid_node_stages(
            layer,
            inputs,
        )
    )
    stale_run = layer.run_complete(
        inputs,
        layer_index=10,
    )

    monkeypatch.setattr(
        layer.normalizer,
        "forward",
        lambda _residual: (
            stale_run
            .node_stages
            .normalization
        ),
    )

    with pytest.raises(
        ValueError,
    ):
        layer.compute_node_stages(
            edge_stages=edge_stages,
            layer_inputs=layer_inputs,
        )


def test_compute_node_stages_rejects_wrong_edge_stage_type() -> None:
    inputs = _inputs()
    layer = _explicit_layer(
        inputs
    )
    layer_inputs = (
        layer.build_layer_inputs(
            inputs,
            layer_index=0,
        )
    )

    with pytest.raises(
        TypeError,
    ):
        layer.compute_node_stages(
            edge_stages=object(),
            layer_inputs=layer_inputs,
        )


def test_compute_node_stages_rejects_wrong_layer_input_type() -> None:
    inputs = _inputs()
    layer = _explicit_layer(
        inputs
    )
    edge_stages = (
        layer.compute_edge_stages(
            inputs
        )
    )

    with pytest.raises(
        TypeError,
    ):
        layer.compute_node_stages(
            edge_stages=edge_stages,
            layer_inputs=object(),
        )


# =============================================================================
# Residual and normalization schema failures
# =============================================================================


def test_residual_output_rejects_crossed_source_lineage() -> None:
    layer, inputs, run = _full_run()
    del layer
    foreign_inputs = (
        _different_source_inputs()
    )
    foreign_layer = _explicit_layer(
        foreign_inputs
    )
    foreign_aggregation = (
        foreign_layer
        .run_complete(
            foreign_inputs
        )
        .node_stages
        .aggregation
    )

    with pytest.raises(
        ValueError,
    ):
        build_layer_residual_update_output(
            aggregation=(
                foreign_aggregation
            ),
            layer_inputs=(
                run.layer_inputs
            ),
            residual_mode=(
                LAYER_RESIDUAL_ADDITIVE
            ),
            dropout_probability=0.0,
            training=True,
        )


def test_residual_output_rejects_training_mismatch() -> None:
    _layer, _inputs_value, run = (
        _full_run()
    )

    with pytest.raises(
        ValueError,
    ):
        build_layer_residual_update_output(
            aggregation=(
                run.node_stages
                .aggregation
            ),
            layer_inputs=(
                run.layer_inputs
            ),
            residual_mode=(
                LAYER_RESIDUAL_ADDITIVE
            ),
            dropout_probability=0.0,
            training=False,
        )


def test_residual_validator_rejects_tampered_equation() -> None:
    _layer, _inputs_value, run = (
        _full_run()
    )
    residual = (
        run.node_stages
        .residual_update
    )
    _tamper(
        residual,
        "post_residual_state",
        torch.zeros_like(
            residual
            .post_residual_state
        ),
    )

    with pytest.raises(
        ValueError,
    ):
        validate_layer_residual_update_output(
            output=residual
        )


def test_residual_validator_rejects_tampered_architecture_fingerprint() -> None:
    _layer, _inputs_value, run = (
        _full_run()
    )
    residual = (
        run.node_stages
        .residual_update
    )
    _tamper(
        residual,
        "updater_architecture_fingerprint",
        "tampered-residual-architecture",
    )

    with pytest.raises(
        ValueError,
    ):
        validate_layer_residual_update_output(
            output=residual
        )


def test_normalization_output_rejects_wrong_position() -> None:
    _layer, _inputs_value, run = (
        _full_run()
    )

    with pytest.raises(
        (NotImplementedError, ValueError),
    ):
        build_layer_normalization_output(
            residual_update=(
                run.node_stages
                .residual_update
            ),
            normalization_mode=(
                LAYER_NORMALIZATION_NONE
            ),
            normalization_position=(
                LAYER_NORMALIZATION_PRE_RESIDUAL
            ),
        )


def test_normalization_validator_rejects_tampered_architecture_fingerprint() -> None:
    _layer, _inputs_value, run = (
        _full_run()
    )
    normalization = (
        run.node_stages
        .normalization
    )
    _tamper(
        normalization,
        "normalizer_architecture_fingerprint",
        "tampered-normalization-architecture",
    )

    with pytest.raises(
        ValueError,
    ):
        validate_layer_normalization_output(
            output=normalization,
            residual_update=(
                run.node_stages
                .residual_update
            ),
        )


def test_normalization_validator_rejects_tampered_values_with_explicit_parameters() -> None:
    layer, _inputs_value, run = (
        _full_run()
    )
    normalization = (
        run.node_stages
        .normalization
    )
    _tamper(
        normalization,
        "output_state",
        normalization.output_state
        + 1.0,
    )

    with pytest.raises(
        ValueError,
    ):
        validate_layer_normalization_output(
            output=normalization,
            residual_update=(
                run.node_stages
                .residual_update
            ),
            normalization_mode=(
                layer.normalizer
                .normalization_mode
            ),
            normalization_position=(
                LAYER_NORMALIZATION_POST_RESIDUAL
            ),
            epsilon=(
                layer.normalizer.epsilon
            ),
            weight=(
                layer.normalizer.weight
            ),
            bias=(
                layer.normalizer.bias
            ),
            normalizer_architecture_fingerprint=(
                layer.normalizer
                .architecture_fingerprint()
            ),
            normalizer_parameter_fingerprint=(
                layer.normalizer
                .parameter_fingerprint()
            ),
        )


# =============================================================================
# Trace schema failures
# =============================================================================


def test_trace_object_is_forbidden_under_none_policy() -> None:
    layer, inputs, run = _full_run()
    del layer
    none_inputs = (
        FunctionalMessagePassingLayerInputs(
            source_inputs=inputs,
            layer_index=0,
            trace_policy=(
                LayerTracePolicy(
                    mode=(
                        LAYER_TRACE_NONE
                    )
                )
            ),
            training=True,
        )
    )

    with pytest.raises(
        ValueError,
    ):
        FunctionalMessagePassingLayerTrace(
            layer_inputs=none_inputs,
            aggregation=(
                run.node_stages
                .aggregation
            ),
            residual_update=(
                run.node_stages
                .residual_update
            ),
            normalization=(
                run.node_stages
                .normalization
            ),
        )


def test_node_trace_rejects_edge_level_objects() -> None:
    inputs = _inputs()
    layer = _explicit_layer(
        inputs
    )
    node_inputs = (
        layer.build_layer_inputs(
            inputs,
            layer_index=0,
            trace_policy=(
                LAYER_TRACE_NODE
            ),
        )
    )
    edge_stages = (
        layer.compute_edge_stages(
            inputs
        )
    )
    node_stages = (
        layer.compute_node_stages(
            edge_stages=edge_stages,
            layer_inputs=node_inputs,
        )
    )

    with pytest.raises(
        ValueError,
    ):
        FunctionalMessagePassingLayerTrace(
            layer_inputs=node_inputs,
            aggregation=(
                node_stages.aggregation
            ),
            residual_update=(
                node_stages.residual_update
            ),
            normalization=(
                node_stages.normalization
            ),
            relation_transform=(
                edge_stages
                .relation_transform
            ),
        )


def test_full_trace_rejects_missing_relation_transform() -> None:
    _layer, _inputs_value, run = (
        _full_run()
    )

    with pytest.raises(
        TypeError,
    ):
        FunctionalMessagePassingLayerTrace(
            layer_inputs=(
                run.layer_inputs
            ),
            aggregation=(
                run.node_stages
                .aggregation
            ),
            residual_update=(
                run.node_stages
                .residual_update
            ),
            normalization=(
                run.node_stages
                .normalization
            ),
            relation_transform=None,
            edge_normalization=(
                run.edge_stages
                .edge_normalization
            ),
            relation_gate=(
                run.edge_stages
                .relation_gate
            ),
            edge_attention=(
                run.edge_stages
                .edge_attention
            ),
            edge_messages=(
                run.edge_stages
                .message_builder_run
                .public_output
            ),
            message_builder_run=(
                run.edge_stages
                .message_builder_run
            ),
        )


def test_full_trace_rejects_stale_edge_messages() -> None:
    _layer, _inputs_value, run = (
        _full_run()
    )
    foreign_layer, foreign_inputs, foreign_run = (
        _full_run(
            inputs=(
                _different_source_inputs()
            )
        )
    )
    del foreign_layer, foreign_inputs

    with pytest.raises(
        ValueError,
    ):
        FunctionalMessagePassingLayerTrace(
            layer_inputs=(
                run.layer_inputs
            ),
            aggregation=(
                run.node_stages
                .aggregation
            ),
            residual_update=(
                run.node_stages
                .residual_update
            ),
            normalization=(
                run.node_stages
                .normalization
            ),
            relation_transform=(
                run.edge_stages
                .relation_transform
            ),
            edge_normalization=(
                run.edge_stages
                .edge_normalization
            ),
            relation_gate=(
                run.edge_stages
                .relation_gate
            ),
            edge_attention=(
                run.edge_stages
                .edge_attention
            ),
            edge_messages=(
                foreign_run
                .edge_stages
                .message_builder_run
                .public_output
            ),
            message_builder_run=(
                run.edge_stages
                .message_builder_run
            ),
        )


# =============================================================================
# Internal output and stage-chain failures
# =============================================================================


def test_internal_output_rejects_cloned_updated_state() -> None:
    _layer, _inputs_value, run = (
        _full_run()
    )

    with pytest.raises(
        ValueError,
    ):
        LayerComputationOutput(
            updated_node_state=(
                run.internal_output
                .updated_node_state
                .clone()
            ),
            layer_inputs=(
                run.layer_inputs
            ),
            aggregation=(
                run.node_stages
                .aggregation
            ),
            residual_update=(
                run.node_stages
                .residual_update
            ),
            normalization=(
                run.node_stages
                .normalization
            ),
            layer_architecture_fingerprint=(
                "architecture"
            ),
            layer_parameter_fingerprint=(
                "parameters"
            ),
            lineage_fingerprint=(
                "lineage"
            ),
            trace=run.trace,
        )


def test_internal_output_rejects_trace_when_policy_is_none() -> None:
    inputs = _inputs()
    layer = _explicit_layer(
        inputs,
        trace_mode=LAYER_TRACE_NONE,
    )
    run = layer.run_complete(
        inputs
    )
    full_layer = _explicit_layer(
        inputs,
        trace_mode=LAYER_TRACE_FULL,
    )
    full_trace = (
        full_layer
        .run_complete(
            inputs
        )
        .trace
    )
    assert full_trace is not None

    with pytest.raises(
        ValueError,
    ):
        LayerComputationOutput(
            updated_node_state=(
                run.internal_output
                .updated_node_state
            ),
            layer_inputs=(
                run.layer_inputs
            ),
            aggregation=(
                run.node_stages
                .aggregation
            ),
            residual_update=(
                run.node_stages
                .residual_update
            ),
            normalization=(
                run.node_stages
                .normalization
            ),
            layer_architecture_fingerprint=(
                "architecture"
            ),
            layer_parameter_fingerprint=(
                "parameters"
            ),
            lineage_fingerprint=(
                "lineage"
            ),
            trace=full_trace,
        )


@pytest.mark.parametrize(
    ("name", "value"),
    (
        (
            "",
            torch.tensor(1.0),
        ),
        (
            "vector",
            torch.ones(2),
        ),
        (
            "integer",
            torch.tensor(1),
        ),
        (
            "nonfinite",
            torch.tensor(float("nan")),
        ),
        (
            "not_tensor",
            1.0,
        ),
    ),
)
def test_internal_output_rejects_malformed_regularization_terms(
    name: str,
    value: Any,
) -> None:
    _layer, _inputs_value, run = (
        _full_run()
    )

    with pytest.raises(
        (TypeError, ValueError, FloatingPointError),
    ):
        LayerComputationOutput(
            updated_node_state=(
                run.internal_output
                .updated_node_state
            ),
            layer_inputs=(
                run.layer_inputs
            ),
            aggregation=(
                run.node_stages
                .aggregation
            ),
            residual_update=(
                run.node_stages
                .residual_update
            ),
            normalization=(
                run.node_stages
                .normalization
            ),
            layer_architecture_fingerprint=(
                "architecture"
            ),
            layer_parameter_fingerprint=(
                "parameters"
            ),
            lineage_fingerprint=(
                "lineage"
            ),
            trace=run.trace,
            regularization_terms={
                name: value
            },
        )


def test_stage_chain_rejects_crossed_aggregation() -> None:
    _layer, _inputs_value, run = (
        _full_run()
    )
    _foreign_layer, _foreign_inputs, foreign_run = (
        _full_run(
            inputs=(
                _different_source_inputs()
            )
        )
    )

    with pytest.raises(
        ValueError,
    ):
        validate_layer_stage_chain(
            layer_inputs=(
                run.layer_inputs
            ),
            aggregation=(
                foreign_run
                .node_stages
                .aggregation
            ),
            residual_update=(
                run.node_stages
                .residual_update
            ),
            normalization=(
                run.node_stages
                .normalization
            ),
        )


def test_stage_chain_rejects_crossed_normalization() -> None:
    _layer, _inputs_value, run = (
        _full_run()
    )
    _foreign_layer, _foreign_inputs, foreign_run = (
        _full_run()
    )

    with pytest.raises(
        ValueError,
    ):
        validate_layer_stage_chain(
            layer_inputs=(
                run.layer_inputs
            ),
            aggregation=(
                run.node_stages
                .aggregation
            ),
            residual_update=(
                run.node_stages
                .residual_update
            ),
            normalization=(
                foreign_run
                .node_stages
                .normalization
            ),
        )


# =============================================================================
# Public-output failures
# =============================================================================


def test_public_validator_rejects_cloned_updated_state() -> None:
    _layer, _inputs_value, run = (
        _full_run()
    )
    public = run.public_output
    _tamper(
        public,
        "updated_node_state",
        public
        .updated_node_state
        .clone(),
    )

    with pytest.raises(
        ValueError,
    ):
        validate_public_layer_output(
            public_output=public,
            internal_output=(
                run.internal_output
            ),
        )


def test_public_validator_rejects_wrong_architecture_fingerprint() -> None:
    _layer, _inputs_value, run = (
        _full_run()
    )
    public = run.public_output
    _tamper(
        public,
        "encoder_architecture_fingerprint",
        "wrong-architecture",
    )

    with pytest.raises(
        ValueError,
    ):
        validate_public_layer_output(
            public_output=public,
            internal_output=(
                run.internal_output
            ),
        )


def test_public_validator_rejects_wrong_lineage_fingerprint() -> None:
    _layer, _inputs_value, run = (
        _full_run()
    )
    public = run.public_output
    _tamper(
        public,
        "lineage_fingerprint",
        "wrong-lineage",
    )

    with pytest.raises(
        ValueError,
    ):
        validate_public_layer_output(
            public_output=public,
            internal_output=(
                run.internal_output
            ),
        )


def test_public_validator_rejects_missing_full_intermediates() -> None:
    _layer, _inputs_value, run = (
        _full_run()
    )
    public = run.public_output
    _tamper(
        public,
        "intermediates",
        None,
    )

    with pytest.raises(
        ValueError,
    ):
        validate_public_layer_output(
            public_output=public,
            internal_output=(
                run.internal_output
            ),
        )


def test_public_assembly_rejects_full_policy_without_trace() -> None:
    _layer, _inputs_value, run = (
        _full_run()
    )
    internal = run.internal_output
    _tamper(
        internal,
        "trace",
        None,
    )

    with pytest.raises(
        ValueError,
    ):
        assemble_functional_message_passing_layer_output(
            internal_output=internal
        )


# =============================================================================
# Complete-run and owning-layer failures
# =============================================================================


def test_complete_run_validator_rejects_wrong_type() -> None:
    with pytest.raises(
        TypeError,
    ):
        validate_functional_message_passing_layer_run(
            object()
        )


def test_complete_run_rejects_crossed_node_stages() -> None:
    _layer, _inputs_value, run = (
        _full_run()
    )
    _foreign_layer, _foreign_inputs, foreign_run = (
        _full_run()
    )

    with pytest.raises(
        ValueError,
    ):
        FunctionalMessagePassingLayerRun(
            layer_inputs=(
                run.layer_inputs
            ),
            edge_stages=(
                run.edge_stages
            ),
            node_stages=(
                foreign_run.node_stages
            ),
            internal_output=(
                run.internal_output
            ),
            public_output=(
                run.public_output
            ),
        )


def test_complete_run_rejects_crossed_edge_stages() -> None:
    _layer, _inputs_value, run = (
        _full_run()
    )
    _foreign_layer, _foreign_inputs, foreign_run = (
        _full_run(
            inputs=(
                _different_source_inputs()
            )
        )
    )

    with pytest.raises(
        ValueError,
    ):
        FunctionalMessagePassingLayerRun(
            layer_inputs=(
                run.layer_inputs
            ),
            edge_stages=(
                foreign_run.edge_stages
            ),
            node_stages=(
                run.node_stages
            ),
            internal_output=(
                run.internal_output
            ),
            public_output=(
                run.public_output
            ),
        )


def test_owning_layer_rejects_run_after_parameter_mutation() -> None:
    layer, _inputs_value, run = (
        _full_run(
            diagnostics_enabled=True
        )
    )

    with torch.no_grad():
        layer.relation_transforms.scale[
            0
        ] += 1.0

    with pytest.raises(
        ValueError,
    ):
        layer.diagnostic_report(
            run=run
        )


def test_run_from_layer_inputs_rejects_wrong_type() -> None:
    layer = _explicit_layer(
        _inputs()
    )

    with pytest.raises(
        TypeError,
    ):
        layer.run_from_layer_inputs(
            object()
        )


@pytest.mark.parametrize(
    "helper",
    (
        run_functional_message_passing_layer,
        run_functional_message_passing_layer_complete,
    ),
)
def test_functional_execution_helpers_reject_wrong_layer(
    helper: Any,
) -> None:
    with pytest.raises(
        TypeError,
    ):
        helper(
            object(),
            _inputs(),
        )


# =============================================================================
# Relation-gate regularization failures
# =============================================================================


@pytest.mark.parametrize(
    ("name", "value_factory"),
    (
        (
            "",
            lambda inputs: torch.tensor(
                1.0,
                dtype=inputs.dtype,
                device=inputs.device,
            ),
        ),
        (
            "not_tensor",
            lambda _inputs_value: 1.0,
        ),
        (
            "vector",
            lambda inputs: torch.ones(
                2,
                dtype=inputs.dtype,
                device=inputs.device,
            ),
        ),
        (
            "integer",
            lambda inputs: torch.tensor(
                1,
                device=inputs.device,
            ),
        ),
        (
            "nonfinite",
            lambda inputs: torch.tensor(
                float("nan"),
                dtype=inputs.dtype,
                device=inputs.device,
            ),
        ),
    ),
)
def test_layer_rejects_malformed_gate_regularization(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    value_factory: Any,
) -> None:
    inputs = _inputs()
    layer = _explicit_layer(
        inputs,
        gate_enabled=True,
    )
    assert layer.relation_gate is not None
    original = (
        layer.relation_gate.forward
    )

    def replacement(
        inputs_value: FunctionalMessagePassingInputs,
    ) -> RelationGateOutput:
        output = original(
            inputs_value
        )
        _tamper(
            output,
            "regularization_terms",
            {
                name: value_factory(
                    inputs_value
                )
            },
        )
        return output

    monkeypatch.setattr(
        layer.relation_gate,
        "forward",
        replacement,
    )

    with pytest.raises(
        (TypeError, ValueError, FloatingPointError),
    ):
        layer.run_complete(
            inputs
        )


# =============================================================================
# Diagnostics and report hardening
# =============================================================================


def test_diagnostic_report_rejects_when_diagnostics_disabled() -> None:
    layer, _inputs_value, run = (
        _full_run(
            diagnostics_enabled=False
        )
    )

    with pytest.raises(
        RuntimeError,
    ):
        layer.diagnostic_report(
            run=run
        )


def test_run_with_diagnostics_rejects_tensor_report() -> None:
    _layer, _inputs_value, run = (
        _full_run()
    )

    with pytest.raises(
        TypeError,
    ):
        FunctionalMessagePassingLayerRunWithDiagnostics(
            run=run,
            diagnostic_report={
                "tensor": torch.tensor(
                    1.0
                )
            },
        )


def test_run_with_diagnostics_rejects_module_report() -> None:
    _layer, _inputs_value, run = (
        _full_run()
    )

    with pytest.raises(
        TypeError,
    ):
        FunctionalMessagePassingLayerRunWithDiagnostics(
            run=run,
            diagnostic_report={
                "module": nn.Identity()
            },
        )


def test_diagnostic_report_tampering_is_detected() -> None:
    layer, _inputs_value, run = (
        _full_run(
            diagnostics_enabled=True
        )
    )
    report = layer.diagnostic_report(
        run=run
    )
    mutable = dict(
        report
    )
    mutable["global"] = dict(
        mutable["global"]
    )
    mutable["global"][
        "num_edges"
    ] += 1

    with pytest.raises(
        ValueError,
    ):
        validate_layer_diagnostic_report(
            mutable
        )


def test_diagnostic_report_rejects_unsupported_scientific_claim() -> None:
    layer, _inputs_value, run = (
        _full_run(
            diagnostics_enabled=True
        )
    )
    report = layer.diagnostic_report(
        run=run
    )
    mutable = dict(
        report
    )
    mutable[
        "scientific_claims"
    ] = dict(
        mutable[
            "scientific_claims"
        ]
    )
    mutable[
        "scientific_claims"
    ][
        "causal_importance"
    ] = True
    mutable.pop(
        "report_fingerprint",
        None,
    )

    with pytest.raises(
        ValueError,
    ):
        validate_layer_diagnostic_report(
            mutable
        )


def test_diagnostic_report_rejects_wrong_expected_graph_count() -> None:
    layer, _inputs_value, run = (
        _full_run(
            diagnostics_enabled=True
        )
    )
    report = layer.diagnostic_report(
        run=run
    )

    with pytest.raises(
        ValueError,
    ):
        validate_layer_diagnostic_report(
            report,
            expected_num_graphs=(
                GRAPHS + 1
            ),
        )


def test_diagnostic_report_rejects_injected_tensor() -> None:
    layer, _inputs_value, run = (
        _full_run(
            diagnostics_enabled=True
        )
    )
    report = dict(
        layer.diagnostic_report(
            run=run
        )
    )
    report["injected"] = torch.tensor(
        1.0
    )
    report.pop(
        "report_fingerprint",
        None,
    )

    with pytest.raises(
        TypeError,
    ):
        validate_layer_diagnostic_report(
            report
        )


def test_layer_diagnostics_rejects_invalid_threshold_object() -> None:
    with pytest.raises(
        TypeError,
    ):
        LayerDiagnostics(
            thresholds=object()
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        (
            "near_zero_absolute",
            0.0,
        ),
        (
            "large_node_state_l2_norm",
            -1.0,
        ),
        (
            "high_isolated_node_fraction",
            1.1,
        ),
    ),
)
def test_layer_diagnostic_thresholds_reject_invalid_values(
    field: str,
    value: float,
) -> None:
    kwargs = {
        field: value
    }

    with pytest.raises(
        (TypeError, ValueError),
    ):
        LayerDiagnosticThresholds(
            **kwargs
        )


# =============================================================================
# Post-construction tamper detection
# =============================================================================


def test_owned_run_rejects_tampered_layer_architecture_fingerprint() -> None:
    layer, _inputs_value, run = (
        _full_run()
    )
    _tamper(
        run.internal_output,
        "layer_architecture_fingerprint",
        "tampered-layer-architecture",
    )

    with pytest.raises(
        ValueError,
    ):
        layer._validate_owned_run(
            run
        )


def test_owned_run_rejects_tampered_layer_parameter_fingerprint() -> None:
    layer, _inputs_value, run = (
        _full_run()
    )
    _tamper(
        run.internal_output,
        "layer_parameter_fingerprint",
        "tampered-layer-parameters",
    )

    with pytest.raises(
        ValueError,
    ):
        layer._validate_owned_run(
            run
        )


def test_complete_run_validator_rejects_tampered_public_output() -> None:
    _layer, _inputs_value, run = (
        _full_run()
    )
    _tamper(
        run.public_output,
        "node_aggregate",
        run.public_output
        .node_aggregate
        .clone(),
    )

    with pytest.raises(
        ValueError,
    ):
        validate_functional_message_passing_layer_run(
            run
        )


def test_complete_run_validator_rejects_tampered_aggregation_source_messages() -> None:
    _layer, _inputs_value, run = (
        _full_run()
    )
    _foreign_layer, _foreign_inputs, foreign_run = (
        _full_run()
    )
    _tamper(
        run.node_stages.aggregation,
        "source_messages",
        foreign_run
        .edge_stages
        .message_builder_run
        .public_output,
    )

    with pytest.raises(
        ValueError,
    ):
        validate_functional_message_passing_layer_run(
            run
        )


def test_stage_named_tuple_rejects_wrong_constructor_arity() -> None:
    with pytest.raises(
        TypeError,
    ):
        FunctionalMessagePassingLayerStages(
            aggregation=object(),
            residual_update=object(),
            normalization=object(),
        )


# =============================================================================
# Serialization and wrapper input failures
# =============================================================================


def test_run_with_diagnostics_rejects_wrong_run_type() -> None:
    with pytest.raises(
        TypeError,
    ):
        FunctionalMessagePassingLayerRunWithDiagnostics(
            run=object(),
            diagnostic_report={},
        )


def test_run_with_diagnostics_rejects_non_mapping_report() -> None:
    _layer, _inputs_value, run = (
        _full_run()
    )

    with pytest.raises(
        TypeError,
    ):
        FunctionalMessagePassingLayerRunWithDiagnostics(
            run=run,
            diagnostic_report=[],
        )


def test_build_layer_diagnostic_report_rejects_wrong_internal_type() -> None:
    with pytest.raises(
        TypeError,
    ):
        build_layer_diagnostic_report(
            object()
        )
