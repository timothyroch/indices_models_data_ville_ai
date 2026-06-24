"""
Integration tests for the complete functional message-passing layer.

Target repository path:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            tests/
                layer/
                    test_functional_message_passing_layer.py

Implementation under test:
    functional_message_passing/layer/layer.py

This suite tests successful complete-layer execution. Component-local validation
and detailed negative contracts belong respectively to:

    test_layer_components.py
    test_layer_failures.py

The layer is tested as an orchestrator over controlled implementations of the
already established component interfaces:

    relation transform
        -> structural edge normalization
        -> optional exact-relation gate
        -> optional exact-relation edge attention
        -> existing message builder
        -> existing target-node mean aggregation
        -> residual update
        -> post-residual normalization
        -> public layer output

The controlled components are real ``nn.Module`` objects and produce the real
immutable public schemas. The layer module's imported component type symbols
are patched to these controlled implementations so that the tests isolate
orchestration, exact lineage, runtime trace behavior, architecture provenance,
parameter provenance, autograd, and diagnostics without duplicating the
component suites.

Successful contracts covered here include:

- explicit-component and configuration-driven construction;
- exact edge-stage and node-stage ordering;
- disabled gate and attention represented by ``None``;
- enabled gate and attention preserved through the message builder;
- relation-gate regularization namespacing;
- exact target-node mean aggregation;
- residual and normalization equations;
- ``none``, ``node``, and ``full`` trace policies;
- historical Boolean intermediate-capture compatibility;
- runtime-supplied layer indices for future shared stacks;
- public-output assembly and complete-run validation;
- architecture fingerprints independent of runtime trace and diagnostics;
- parameter fingerprints responsive to learned values;
- ordinary forward passes free of implicit diagnostics;
- explicit tensor-free diagnostics;
- empty-edge, train/eval, float64, autograd, and optional CUDA behavior;
- preservation limits: retained values remain descriptive rather
  than causal or explanation-faithfulness claims.
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
    LayerDiagnostics,
    build_layer_diagnostics,
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
    LayerNormalizer,
    build_layer_normalizer_from_flag,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.functional_message_passing.layer.residual_update import (
    LayerResidualUpdater,
    build_layer_residual_updater_from_flags,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.functional_message_passing.layer.schemas import (
    LAYER_TRACE_FULL,
    LAYER_TRACE_NODE,
    LAYER_TRACE_NONE,
    FunctionalMessagePassingLayerInputs,
    FunctionalMessagePassingLayerTrace,
    LayerComputationOutput,
    LayerTracePolicy,
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
# Public identity and aliases
# =============================================================================


def test_layer_public_identity_constants() -> None:
    assert (
        FUNCTIONAL_MESSAGE_PASSING_LAYER_SCHEMA_VERSION
        .strip()
    )
    assert (
        FUNCTIONAL_MESSAGE_PASSING_LAYER_OPERATION_ORDER[
            0
        ]
        == "validate_component_and_runtime_contracts"
    )
    assert (
        FUNCTIONAL_MESSAGE_PASSING_LAYER_OPERATION_ORDER[
            -1
        ]
        == "validate_exact_complete_run_lineage"
    )
    assert (
        FUNCTIONAL_MESSAGE_PASSING_LAYER_SCIENTIFIC_INTERPRETATION
        == "one_hazard_conditioned_functional_graph_state_transition"
    )
    assert (
        FUNCTIONAL_MESSAGE_PASSING_LAYER_OUTPUT_SCHEMA
        == "FunctionalMessagePassingLayerOutput"
    )
    assert (
        FUNCTIONAL_MESSAGE_PASSING_LAYER_AGGREGATION_ORCHESTRATED_HERE
        is True
    )
    assert (
        FUNCTIONAL_MESSAGE_PASSING_LAYER_AGGREGATION_MATH_OWNED_HERE
        is False
    )
    assert (
        FUNCTIONAL_MESSAGE_PASSING_LAYER_STACKING_OWNED_HERE
        is False
    )
    assert (
        FUNCTIONAL_MESSAGE_PASSING_LAYER_PREDICTION_OWNED_HERE
        is False
    )
    assert (
        FUNCTIONAL_MESSAGE_PASSING_LAYER_TRACE_AFFECTS_NUMERICS
        is False
    )
    assert (
        FUNCTIONAL_MESSAGE_PASSING_LAYER_DIAGNOSTICS_AFFECT_NUMERICS
        is False
    )


def test_layer_disabled_and_uniform_attention_representations_are_distinct() -> None:
    assert "None" in (
        FUNCTIONAL_MESSAGE_PASSING_LAYER_DISABLED_GATE_REPRESENTATION
    )
    assert "identity_one" in (
        FUNCTIONAL_MESSAGE_PASSING_LAYER_DISABLED_GATE_REPRESENTATION
    )
    assert "None" in (
        FUNCTIONAL_MESSAGE_PASSING_LAYER_DISABLED_ATTENTION_REPRESENTATION
    )
    assert "identity_one" in (
        FUNCTIONAL_MESSAGE_PASSING_LAYER_DISABLED_ATTENTION_REPRESENTATION
    )
    assert "zero_logits" in (
        FUNCTIONAL_MESSAGE_PASSING_LAYER_UNIFORM_ATTENTION_REPRESENTATION
    )
    assert (
        FUNCTIONAL_MESSAGE_PASSING_LAYER_DISABLED_ATTENTION_REPRESENTATION
        != FUNCTIONAL_MESSAGE_PASSING_LAYER_UNIFORM_ATTENTION_REPRESENTATION
    )


def test_layer_aliases_are_exact() -> None:
    assert (
        HazardConditionedFunctionalMessagePassingLayer
        is FunctionalMessagePassingLayer
    )
    assert FunctionalLayer is (
        FunctionalMessagePassingLayer
    )
    assert MessagePassingLayer is (
        FunctionalMessagePassingLayer
    )
    assert LayerEdgeStages is (
        FunctionalMessagePassingLayerEdgeStages
    )
    assert LayerNodeStages is (
        FunctionalMessagePassingLayerNodeStages
    )
    assert LayerRun is (
        FunctionalMessagePassingLayerRun
    )
    assert LayerRunWithDiagnostics is (
        FunctionalMessagePassingLayerRunWithDiagnostics
    )
    assert build_layer is (
        build_functional_message_passing_layer
    )
    assert build_layer_from_config is (
        build_functional_message_passing_layer_from_config
    )
    assert run_layer is (
        run_functional_message_passing_layer
    )
    assert run_layer_complete is (
        run_functional_message_passing_layer_complete
    )


# =============================================================================
# Construction and component ownership
# =============================================================================


def test_explicit_component_construction() -> None:
    inputs = _inputs()
    layer = _explicit_layer(
        inputs,
        diagnostics_enabled=True,
    )

    assert layer.hidden_dim == HIDDEN_DIM
    assert layer.relation_names == (
        RELATION_NAMES
    )
    assert layer.stable_relation_ids == (
        STABLE_RELATION_IDS
    )
    assert layer.num_relations == RELATIONS
    assert layer.gate_enabled is True
    assert layer.attention_enabled is True
    assert layer.residual_enabled is True
    assert layer.layer_norm_enabled is True
    assert layer.diagnostics_enabled is True
    assert (
        layer.semantic_edge_policy
        == MESSAGE_SEMANTIC_EDGE_POLICY_IGNORE
    )
    assert isinstance(
        layer.relation_transforms,
        ControlledRelationTransforms,
    )
    assert isinstance(
        layer.edge_normalization,
        ControlledEdgeNormalization,
    )
    assert isinstance(
        layer.relation_gate,
        ControlledRelationGate,
    )
    assert isinstance(
        layer.edge_attention,
        ControlledEdgeAttention,
    )
    assert isinstance(
        layer.aggregator,
        ControlledMessageAggregator,
    )
    assert isinstance(
        layer.residual_updater,
        LayerResidualUpdater,
    )
    assert isinstance(
        layer.normalizer,
        LayerNormalizer,
    )
    assert isinstance(
        layer.diagnostics,
        LayerDiagnostics,
    )


def test_explicit_builder_returns_layer() -> None:
    inputs = _inputs()
    reference = _explicit_layer(
        inputs
    )

    rebuilt = (
        build_functional_message_passing_layer(
            relation_transforms=(
                reference
                .relation_transforms
            ),
            edge_normalization=(
                reference
                .edge_normalization
            ),
            relation_gate=(
                reference
                .relation_gate
            ),
            edge_attention=(
                reference
                .edge_attention
            ),
            message_builder=(
                reference
                .message_builder
            ),
            aggregator=(
                reference.aggregator
            ),
            residual_updater=(
                reference
                .residual_updater
            ),
            normalizer=(
                reference.normalizer
            ),
            default_trace_policy=(
                reference
                .default_trace_policy
            ),
            diagnostics=(
                reference.diagnostics
            ),
        )
    )

    assert isinstance(
        rebuilt,
        FunctionalMessagePassingLayer,
    )
    assert (
        rebuilt.relation_transforms
        is reference.relation_transforms
    )
    assert rebuilt.aggregator is (
        reference.aggregator
    )


@pytest.mark.parametrize(
    (
        "gate_enabled",
        "attention_enabled",
        "residual_enabled",
        "layer_norm_enabled",
        "capture_intermediates",
    ),
    (
        (
            False,
            False,
            False,
            False,
            False,
        ),
        (
            True,
            False,
            True,
            False,
            False,
        ),
        (
            False,
            True,
            False,
            True,
            True,
        ),
        (
            True,
            True,
            True,
            True,
            True,
        ),
    ),
)
def test_configuration_driven_construction(
    gate_enabled: bool,
    attention_enabled: bool,
    residual_enabled: bool,
    layer_norm_enabled: bool,
    capture_intermediates: bool,
) -> None:
    inputs = _inputs()
    layer = _configured_layer(
        inputs,
        gate_enabled=gate_enabled,
        attention_enabled=(
            attention_enabled
        ),
        residual_enabled=(
            residual_enabled
        ),
        layer_norm_enabled=(
            layer_norm_enabled
        ),
        capture_intermediates=(
            capture_intermediates
        ),
    )

    assert layer.gate_enabled is (
        gate_enabled
    )
    assert layer.attention_enabled is (
        attention_enabled
    )
    assert layer.residual_enabled is (
        residual_enabled
    )
    assert layer.layer_norm_enabled is (
        layer_norm_enabled
    )
    assert (
        layer.default_trace_policy.mode
        == (
            LAYER_TRACE_FULL
            if capture_intermediates
            else LAYER_TRACE_NONE
        )
    )


def test_configuration_builder_alias() -> None:
    inputs = _inputs()
    config = (
        ControlledFunctionalMessagePassingConfig()
    )
    relation_config = (
        ControlledRelationConfig()
    )

    layer = (
        build_functional_message_passing_layer_from_config(
            config=config,
            relation_config=(
                relation_config
            ),
            source_inputs=inputs,
        )
    )
    alias = build_layer_from_config(
        config=config,
        relation_config=(
            relation_config
        ),
        source_inputs=inputs,
    )

    assert isinstance(
        layer,
        FunctionalMessagePassingLayer,
    )
    assert isinstance(
        alias,
        FunctionalMessagePassingLayer,
    )


def test_extra_repr_exposes_scope_without_claiming_stack_ownership() -> None:
    layer = _explicit_layer(
        _inputs()
    )
    representation = (
        layer.extra_repr()
    )

    assert "hidden_dim=4" in representation
    assert "num_relations=3" in representation
    assert "layer_index_runtime_supplied=True" in (
        representation
    )
    assert "stacking_owned_here=False" in (
        representation
    )


# =============================================================================
# Numerical architecture and parameter provenance
# =============================================================================


def test_numerical_architecture_is_deterministic() -> None:
    layer = _explicit_layer(
        _inputs(),
        diagnostics_enabled=True,
    )

    first = (
        layer.numerical_architecture_dict()
    )
    second = (
        layer.numerical_architecture_dict()
    )

    assert first == second
    assert (
        layer.architecture_dict()
        == first
    )
    assert (
        layer.architecture_fingerprint()
        == layer.architecture_fingerprint()
    )
    assert first[
        "aggregation_orchestrated_here"
    ] is True
    assert first[
        "aggregation_math_owned_here"
    ] is False
    assert first[
        "stacking_owned_here"
    ] is False
    assert first[
        "prediction_owned_here"
    ] is False
    assert first[
        "claims_causal_importance"
    ] is False
    assert first[
        "claims_explanation_faithfulness"
    ] is False


def test_trace_policy_and_diagnostics_do_not_change_numerical_architecture() -> None:
    inputs = _inputs()
    no_trace = _explicit_layer(
        inputs,
        trace_mode=LAYER_TRACE_NONE,
        diagnostics_enabled=False,
    )
    full_trace = _explicit_layer(
        inputs,
        trace_mode=LAYER_TRACE_FULL,
        diagnostics_enabled=True,
    )

    full_trace.load_state_dict(
        no_trace.state_dict()
    )

    assert (
        no_trace
        .architecture_fingerprint()
        == full_trace
        .architecture_fingerprint()
    )
    assert (
        no_trace
        .parameter_fingerprint()
        == full_trace
        .parameter_fingerprint()
    )


def test_parameter_fingerprint_is_deterministic_and_value_sensitive() -> None:
    layer = _explicit_layer(
        _inputs()
    )
    before = (
        layer.parameter_fingerprint()
    )
    repeated = (
        layer.parameter_fingerprint()
    )

    assert before == repeated
    assert isinstance(
        before,
        str,
    )
    assert before.strip()

    with torch.no_grad():
        layer.relation_transforms.scale[
            0
        ] = 2.0

    after = (
        layer.parameter_fingerprint()
    )
    assert before != after


def test_runtime_dict_separates_runtime_trace_and_diagnostics() -> None:
    layer = _explicit_layer(
        _inputs(),
        trace_mode=LAYER_TRACE_NODE,
        diagnostics_enabled=True,
    )
    runtime = layer.runtime_dict()

    assert runtime[
        "architecture_fingerprint"
    ] == layer.architecture_fingerprint()
    assert runtime[
        "parameter_fingerprint"
    ] == layer.parameter_fingerprint()
    assert runtime["training"] is True
    assert runtime[
        "default_trace_policy"
    ]["mode"] == LAYER_TRACE_NODE
    assert runtime[
        "diagnostics_enabled"
    ] is True
    assert isinstance(
        runtime[
            "diagnostics_architecture"
        ],
        dict,
    )


def test_parameter_and_buffer_counts_match_module_tree() -> None:
    layer = _explicit_layer(
        _inputs()
    )

    assert layer.parameter_count == sum(
        int(parameter.numel())
        for parameter in layer.parameters()
    )
    assert (
        layer.trainable_parameter_count
        == sum(
            int(parameter.numel())
            for parameter in layer.parameters()
            if parameter.requires_grad
        )
    )
    assert layer.buffer_count == sum(
        int(buffer.numel())
        for buffer in layer.buffers()
    )
    assert layer.parameter_count > 0
    layer.assert_finite_parameters()


# =============================================================================
# Edge-level and node-level orchestration
# =============================================================================


def test_compute_edge_stages_preserves_exact_lineage() -> None:
    inputs = _inputs()
    layer = _explicit_layer(
        inputs
    )
    stages = layer.compute_edge_stages(
        inputs
    )

    assert isinstance(
        stages,
        FunctionalMessagePassingLayerEdgeStages,
    )
    assert (
        stages.relation_transform
        .source_inputs
        is inputs
    )
    assert (
        stages.edge_normalization
        .source_inputs
        is inputs
    )
    assert stages.relation_gate is not None
    assert (
        stages.relation_gate.source_inputs
        is inputs
    )
    assert stages.edge_attention is not None
    assert (
        stages.edge_attention.source_inputs
        is inputs
    )
    assert (
        stages
        .message_builder_run
        .composition_output
        .relation_transform
        is stages.relation_transform
    )
    assert (
        stages
        .message_builder_run
        .resolved_coefficients
        .edge_normalization
        is stages.edge_normalization
    )
    assert (
        stages
        .message_builder_run
        .resolved_coefficients
        .relation_gate
        is stages.relation_gate
    )
    assert (
        stages
        .message_builder_run
        .resolved_coefficients
        .edge_attention
        is stages.edge_attention
    )


def test_disabled_gate_and_attention_remain_none() -> None:
    inputs = _inputs()
    layer = _explicit_layer(
        inputs,
        gate_enabled=False,
        attention_enabled=False,
    )
    stages = layer.compute_edge_stages(
        inputs
    )

    assert stages.relation_gate is None
    assert stages.edge_attention is None
    assert (
        stages
        .message_builder_run
        .resolved_coefficients
        .relation_gate
        is None
    )
    assert (
        stages
        .message_builder_run
        .resolved_coefficients
        .edge_attention
        is None
    )
    assert torch.equal(
        stages
        .message_builder_run
        .resolved_coefficients
        .relation_gate_factor,
        torch.ones_like(
            stages
            .message_builder_run
            .resolved_coefficients
            .relation_gate_factor
        ),
    )
    assert torch.equal(
        stages
        .message_builder_run
        .resolved_coefficients
        .edge_attention_factor,
        torch.ones_like(
            stages
            .message_builder_run
            .resolved_coefficients
            .edge_attention_factor
        ),
    )


def test_enabled_attention_is_group_normalized() -> None:
    inputs = _inputs()
    layer = _explicit_layer(
        inputs,
        attention_enabled=True,
    )
    stages = layer.compute_edge_stages(
        inputs
    )
    assert stages.edge_attention is not None

    attention = stages.edge_attention

    for group_id in torch.unique(
        attention.group_ids,
        sorted=True,
    ):
        mask = (
            attention.group_ids
            == group_id
        )
        torch.testing.assert_close(
            attention
            .normalized_weights_by_head[
                mask
            ]
            .sum(
                dim=0
            ),
            torch.ones(
                HEADS,
                dtype=inputs.dtype,
                device=inputs.device,
            ),
        )


def test_compute_node_stages_matches_exact_mean_aggregation() -> None:
    inputs = _inputs()
    layer = _explicit_layer(
        inputs,
        residual_enabled=False,
        layer_norm_enabled=False,
    )
    layer_inputs = (
        layer.build_layer_inputs(
            inputs,
            layer_index=4,
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
    expected, counts = (
        _expected_mean_aggregate(
            edge_stages
            .message_builder_run
            .public_output
        )
    )

    assert isinstance(
        node_stages,
        FunctionalMessagePassingLayerNodeStages,
    )
    torch.testing.assert_close(
        node_stages
        .aggregation
        .node_aggregate,
        expected,
    )
    assert torch.equal(
        node_stages
        .aggregation
        .incoming_edge_count,
        counts,
    )
    assert (
        node_stages
        .aggregation
        .source_messages
        is edge_stages
        .message_builder_run
        .public_output
    )
    assert (
        node_stages
        .residual_update
        .post_residual_state
        is node_stages
        .aggregation
        .node_aggregate
    )
    assert (
        node_stages
        .normalization
        .output_state
        is node_stages
        .residual_update
        .post_residual_state
    )


def test_additive_residual_and_layer_norm_match_equations() -> None:
    inputs = _inputs()
    layer = _explicit_layer(
        inputs,
        residual_enabled=True,
        layer_norm_enabled=True,
        dropout_probability=0.0,
    )
    run = layer.run_complete(
        inputs,
        layer_index=1,
    )
    aggregate = (
        run.node_stages
        .aggregation
        .node_aggregate
    )
    expected_residual = (
        inputs
        .node_state
        .fused_state
        + aggregate
    )
    expected_output = F.layer_norm(
        expected_residual,
        normalized_shape=(
            HIDDEN_DIM,
        ),
        weight=(
            layer.normalizer.weight
        ),
        bias=(
            layer.normalizer.bias
        ),
        eps=(
            layer.normalizer.epsilon
        ),
    )

    torch.testing.assert_close(
        run.node_stages
        .residual_update
        .post_residual_state,
        expected_residual,
    )
    torch.testing.assert_close(
        run.public_output
        .updated_node_state,
        expected_output,
    )


# =============================================================================
# Complete run and public output
# =============================================================================


def test_complete_run_contract() -> None:
    inputs = _inputs()
    layer = _explicit_layer(
        inputs,
        trace_mode=LAYER_TRACE_FULL,
    )
    run = layer.run_complete(
        inputs,
        layer_index=3,
        source_stack_fingerprint=(
            "shared-stack"
        ),
    )

    assert isinstance(
        run,
        FunctionalMessagePassingLayerRun,
    )
    assert run.source_inputs is inputs
    assert run.layer_index == 3
    assert (
        run.layer_inputs
        .source_stack_fingerprint
        == "shared-stack"
    )
    assert (
        run.updated_node_state
        is run.public_output
        .updated_node_state
    )
    assert (
        run.node_stages
        .aggregation
        .source_messages
        is run.edge_stages
        .message_builder_run
        .public_output
    )
    assert (
        run.internal_output
        .updated_node_state
        is run.node_stages
        .normalization
        .output_state
    )
    assert (
        run.public_output
        .source_inputs
        is inputs
    )

    validate_functional_message_passing_layer_run(
        run
    )


def test_forward_returns_only_public_output() -> None:
    inputs = _inputs()
    layer = _explicit_layer(
        inputs
    )

    output = layer(
        inputs,
        layer_index=2,
    )

    assert isinstance(
        output,
        FunctionalMessagePassingLayerOutput,
    )
    assert output.layer_index == 2
    assert output.source_inputs is inputs


def test_forward_and_complete_run_are_numerically_identical() -> None:
    inputs = _inputs()
    layer = _explicit_layer(
        inputs,
        dropout_probability=0.0,
    )

    public = layer(
        inputs,
        layer_index=5,
    )
    complete = (
        layer.run_complete(
            inputs,
            layer_index=5,
        )
    )

    torch.testing.assert_close(
        public.updated_node_state,
        complete
        .public_output
        .updated_node_state,
    )
    torch.testing.assert_close(
        public.node_aggregate,
        complete
        .public_output
        .node_aggregate,
    )
    assert public.layer_index == (
        complete.layer_index
    )


def test_public_assembly_preserves_exact_objects() -> None:
    inputs = _inputs()
    layer = _explicit_layer(
        inputs,
        trace_mode=LAYER_TRACE_FULL,
    )
    run = layer.run_complete(
        inputs
    )

    rebuilt = (
        assemble_functional_message_passing_layer_output(
            internal_output=(
                run.internal_output
            )
        )
    )

    assert (
        rebuilt.updated_node_state
        is run.internal_output
        .updated_node_state
    )
    assert (
        rebuilt.node_aggregate
        is run.internal_output
        .node_aggregate
    )
    assert (
        rebuilt.incoming_edge_count
        is run.internal_output
        .incoming_edge_count
    )
    assert rebuilt.source_inputs is inputs
    assert rebuilt.intermediates is not None


def test_functional_execution_helpers() -> None:
    inputs = _inputs()
    layer = _explicit_layer(
        inputs
    )

    public = (
        run_functional_message_passing_layer(
            layer,
            inputs,
            layer_index=2,
        )
    )
    public_alias = run_layer(
        layer,
        inputs,
        layer_index=2,
    )
    complete = (
        run_functional_message_passing_layer_complete(
            layer,
            inputs,
            layer_index=2,
        )
    )
    complete_alias = run_layer_complete(
        layer,
        inputs,
        layer_index=2,
    )

    torch.testing.assert_close(
        public.updated_node_state,
        public_alias.updated_node_state,
    )
    torch.testing.assert_close(
        complete.updated_node_state,
        complete_alias.updated_node_state,
    )


# =============================================================================
# Trace policies and shared-layer runtime identity
# =============================================================================


@pytest.mark.parametrize(
    "trace_mode",
    (
        LAYER_TRACE_NONE,
        LAYER_TRACE_NODE,
        LAYER_TRACE_FULL,
    ),
)
def test_runtime_trace_modes(
    trace_mode: str,
) -> None:
    inputs = _inputs()
    layer = _explicit_layer(
        inputs,
        trace_mode=LAYER_TRACE_NONE,
    )
    run = layer.run_complete(
        inputs,
        trace_policy=trace_mode,
    )

    assert (
        run.layer_inputs
        .trace_policy
        .mode
        == trace_mode
    )

    if trace_mode == LAYER_TRACE_NONE:
        assert run.trace is None
        assert (
            run.public_output
            .intermediates
            is None
        )
    elif trace_mode == LAYER_TRACE_NODE:
        assert isinstance(
            run.trace,
            FunctionalMessagePassingLayerTrace,
        )
        assert (
            run.trace
            .relation_transform
            is None
        )
        assert (
            run.trace
            .edge_messages
            is None
        )
        assert (
            run.public_output
            .intermediates
            is None
        )
    else:
        assert isinstance(
            run.trace,
            FunctionalMessagePassingLayerTrace,
        )
        assert (
            run.trace
            .relation_transform
            is run.edge_stages
            .relation_transform
        )
        assert (
            run.trace
            .edge_messages
            is run.edge_stages
            .message_builder_run
            .public_output
        )
        assert (
            run.public_output
            .intermediates
            is not None
        )


@pytest.mark.parametrize(
    ("capture", "expected"),
    (
        (
            False,
            LAYER_TRACE_NONE,
        ),
        (
            True,
            LAYER_TRACE_FULL,
        ),
    ),
)
def test_historical_capture_flag(
    capture: bool,
    expected: str,
) -> None:
    inputs = _inputs()
    layer = _explicit_layer(
        inputs,
        trace_mode=LAYER_TRACE_NODE,
    )
    run = layer.run_complete(
        inputs,
        capture_intermediate_messages=(
            capture
        ),
    )

    assert (
        run.layer_inputs
        .trace_policy
        .mode
        == expected
    )


def test_runtime_layer_index_supports_shared_layer_reuse() -> None:
    inputs = _inputs()
    layer = _explicit_layer(
        inputs,
        dropout_probability=0.0,
    )

    first = layer.run_complete(
        inputs,
        layer_index=0,
        source_stack_fingerprint=(
            "shared-stack"
        ),
    )
    second = layer.run_complete(
        inputs,
        layer_index=4,
        source_stack_fingerprint=(
            "shared-stack"
        ),
    )

    torch.testing.assert_close(
        first.updated_node_state,
        second.updated_node_state,
    )
    assert first.layer_index == 0
    assert second.layer_index == 4
    assert (
        first.internal_output
        .layer_architecture_fingerprint
        == second.internal_output
        .layer_architecture_fingerprint
    )
    assert (
        first.internal_output
        .lineage_fingerprint
        != second.internal_output
        .lineage_fingerprint
    )


def test_build_layer_inputs_preserves_runtime_identity() -> None:
    inputs = _inputs()
    layer = _explicit_layer(
        inputs
    )
    layer_inputs = (
        layer.build_layer_inputs(
            inputs,
            layer_index=7,
            trace_policy=(
                LayerTracePolicy(
                    mode=(
                        LAYER_TRACE_NODE
                    )
                )
            ),
            source_stack_fingerprint=(
                "stack-fingerprint"
            ),
        )
    )

    assert isinstance(
        layer_inputs,
        FunctionalMessagePassingLayerInputs,
    )
    assert layer_inputs.source_inputs is (
        inputs
    )
    assert layer_inputs.layer_index == 7
    assert (
        layer_inputs.trace_policy.mode
        == LAYER_TRACE_NODE
    )
    assert (
        layer_inputs
        .source_stack_fingerprint
        == "stack-fingerprint"
    )


# =============================================================================
# Regularization and diagnostics
# =============================================================================


def test_relation_gate_regularization_is_namespaced() -> None:
    inputs = _inputs()
    layer = _explicit_layer(
        inputs,
        gate_enabled=True,
    )
    run = layer.run_complete(
        inputs
    )

    assert tuple(
        run.internal_output
        .regularization_terms
    ) == (
        "relation_gate.mean_gate",
    )
    term = (
        run.internal_output
        .regularization_terms[
            "relation_gate.mean_gate"
        ]
    )
    assert term.ndim == 0
    assert term.device == (
        inputs.device
    )
    assert bool(
        torch.isfinite(term)
        .item()
    )


def test_disabled_gate_has_no_regularization_terms() -> None:
    inputs = _inputs()
    layer = _explicit_layer(
        inputs,
        gate_enabled=False,
    )
    run = layer.run_complete(
        inputs
    )

    assert dict(
        run.internal_output
        .regularization_terms
    ) == {}


def test_ordinary_forward_does_not_run_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _inputs()
    layer = _explicit_layer(
        inputs,
        diagnostics_enabled=True,
    )
    assert layer.diagnostics is not None

    def forbidden(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise AssertionError(
            "Diagnostics ran during ordinary forward."
        )

    monkeypatch.setattr(
        layer.diagnostics,
        "public_report",
        forbidden,
    )

    output = layer(
        inputs
    )
    assert isinstance(
        output,
        FunctionalMessagePassingLayerOutput,
    )


def test_explicit_diagnostic_report_is_tensor_free() -> None:
    inputs = _inputs()
    layer = _explicit_layer(
        inputs,
        trace_mode=LAYER_TRACE_FULL,
        diagnostics_enabled=True,
    )
    run = layer.run_complete(
        inputs
    )
    report = layer.diagnostic_report(
        run=run
    )

    _assert_tensor_free(
        report
    )
    json.dumps(
        report,
        sort_keys=True,
        allow_nan=False,
    )
    assert report[
        "scientific_claims"
    ] == {
        "causal_importance": False,
        "explanation_faithfulness": False,
        "uncertainty_calibration": False,
        "counterfactual_effect": False,
        "mechanistic_identifiability": False,
        "relation_necessity": False,
    }


def test_forward_with_diagnostics_returns_complete_wrapper() -> None:
    inputs = _inputs()
    layer = _explicit_layer(
        inputs,
        trace_mode=LAYER_TRACE_FULL,
        diagnostics_enabled=True,
    )

    result = (
        layer.forward_with_diagnostics(
            inputs,
            layer_index=6,
        )
    )

    assert isinstance(
        result,
        FunctionalMessagePassingLayerRunWithDiagnostics,
    )
    assert result.run.layer_index == 6
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


# =============================================================================
# Train/eval, empty edges, dtype, gradients, and CUDA
# =============================================================================


def test_eval_mode_bypasses_dropout_by_exact_identity() -> None:
    inputs = _inputs()
    layer = _explicit_layer(
        inputs,
        dropout_probability=0.6,
        layer_norm_enabled=False,
    )
    layer.eval()

    run = layer.run_complete(
        inputs
    )

    assert (
        run.node_stages
        .residual_update
        .post_dropout_update
        is run.node_stages
        .residual_update
        .pre_dropout_update
    )
    assert (
        run.layer_inputs.training
        is False
    )


def test_training_dropout_is_reproducible_with_fixed_seed() -> None:
    inputs = _inputs()
    layer = _explicit_layer(
        inputs,
        dropout_probability=0.4,
        layer_norm_enabled=False,
    )
    layer.train()

    torch.manual_seed(1234)
    first = layer.run_complete(
        inputs
    )
    torch.manual_seed(1234)
    second = layer.run_complete(
        inputs
    )

    torch.testing.assert_close(
        first
        .node_stages
        .residual_update
        .post_dropout_update,
        second
        .node_stages
        .residual_update
        .post_dropout_update,
    )


def test_empty_edges_preserve_source_state_under_additive_residual() -> None:
    inputs = _inputs(
        empty_edges=True
    )
    layer = _explicit_layer(
        inputs,
        gate_enabled=True,
        attention_enabled=True,
        residual_enabled=True,
        layer_norm_enabled=False,
    )

    run = layer.run_complete(
        inputs,
        trace_policy=LAYER_TRACE_FULL,
    )

    assert (
        run.edge_stages
        .message_builder_run
        .public_output
        .edge_messages
        .shape
        == (
            0,
            HIDDEN_DIM,
        )
    )
    assert torch.equal(
        run.public_output
        .node_aggregate,
        torch.zeros_like(
            run.public_output
            .node_aggregate
        ),
    )
    torch.testing.assert_close(
        run.public_output
        .updated_node_state,
        inputs
        .node_state
        .fused_state,
    )


def test_complete_layer_preserves_float64() -> None:
    inputs = _inputs(
        dtype=torch.float64
    )
    layer = _explicit_layer(
        inputs
    )

    run = layer.run_complete(
        inputs
    )

    assert (
        run.public_output
        .updated_node_state
        .dtype
        == torch.float64
    )
    assert (
        run.public_output
        .node_aggregate
        .dtype
        == torch.float64
    )
    assert (
        layer
        .relation_transforms
        .scale
        .dtype
        == torch.float64
    )


def test_complete_layer_preserves_autograd_to_all_trainable_components() -> None:
    source_state = (
        torch.arange(
            NODES * HIDDEN_DIM,
            dtype=torch.float32,
        )
        .reshape(
            NODES,
            HIDDEN_DIM,
        )
        .div(7.0)
        .detach()
        .requires_grad_(True)
    )
    inputs = _inputs(
        state=source_state
    )
    layer = _explicit_layer(
        inputs,
        gate_enabled=True,
        attention_enabled=True,
        residual_enabled=True,
        layer_norm_enabled=True,
        dropout_probability=0.0,
    )

    output = layer(
        inputs
    )
    loss = (
        output
        .updated_node_state
        .square()
        .sum()
        + sum(
            output
            .regularization_terms
            .values()
        )
    )
    loss.backward()

    assert source_state.grad is not None
    assert (
        layer
        .relation_transforms
        .scale
        .grad
        is not None
    )
    assert layer.relation_gate is not None
    assert layer.relation_gate.bias.grad is not None
    assert layer.edge_attention is not None
    assert (
        layer
        .edge_attention
        .head_scale
        .grad
        is not None
    )
    assert layer.normalizer.weight is not None
    assert (
        layer.normalizer.weight.grad
        is not None
    )
    assert layer.normalizer.bias is not None
    assert (
        layer.normalizer.bias.grad
        is not None
    )

    for gradient in (
        source_state.grad,
        layer
        .relation_transforms
        .scale
        .grad,
        layer
        .relation_gate
        .bias
        .grad,
        layer
        .edge_attention
        .head_scale
        .grad,
        layer
        .normalizer
        .weight
        .grad,
        layer
        .normalizer
        .bias
        .grad,
    ):
        assert gradient is not None
        assert bool(
            torch.isfinite(
                gradient
            )
            .all()
            .item()
        )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is not available.",
)
def test_complete_layer_supports_cuda() -> None:
    inputs = _inputs(
        device="cuda"
    )
    layer = _explicit_layer(
        inputs
    )

    run = layer.run_complete(
        inputs
    )

    assert (
        run.public_output
        .updated_node_state
        .device
        .type
        == "cuda"
    )
    assert (
        run.edge_stages
        .message_builder_run
        .public_output
        .edge_messages
        .device
        .type
        == "cuda"
    )
    assert (
        layer
        .relation_transforms
        .scale
        .device
        .type
        == "cuda"
    )
