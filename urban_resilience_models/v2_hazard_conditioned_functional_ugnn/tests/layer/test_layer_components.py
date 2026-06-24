"""
Component tests for one functional message-passing layer.

Target repository path:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            tests/
                layer/
                    test_layer_components.py

Implementations under test:
    functional_message_passing/layer/schemas.py
    functional_message_passing/layer/residual_update.py
    functional_message_passing/layer/normalization.py
    functional_message_passing/layer/diagnostics.py

This suite deliberately stops before testing the complete layer orchestrator in
``layer/layer.py``. It tests the stable component boundaries that the
orchestrator will later compose:

    existing edge-message builder
        -> existing mean target-node aggregation
        -> dropout and optional additive residual
        -> optional post-residual layer normalization
        -> immutable internal layer output
        -> optional tensor-free diagnostics

Primary contracts
-----------------
- aggregation is consumed, not reimplemented inside ``layer/``;
- the residual update uses the exact aggregate and exact source node state;
- inactive dropout preserves exact tensor identity;
- disabled residuals preserve exact post-dropout tensor identity;
- additive residuals implement source state plus realized update;
- disabled normalization preserves exact input tensor identity;
- layer normalization operates independently per node over the hidden axis;
- affine parameters have explicit initialization, counts, and fingerprints;
- ``none``, ``node``, and ``full`` trace policies retain only their declared
  stage objects;
- internal and public schemas preserve exact lineage;
- diagnostics remain explicit, tensor-free, JSON-safe, and descriptive only;
- graph slices, isolated-node coverage, regularization scalars, alerts, and
  report fingerprints are deterministic;
- dtype, device, empty-edge behavior, autograd, train/eval behavior, and
  optional CUDA remain valid.

Controlled upstream doubles are patched into
``functional_message_passing.schemas`` so failures remain localized to the
layer components.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import json
from typing import Any

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.constants import (
    AGGREGATION_MEAN,
    ATTENTION_HEAD_REDUCTION_MEAN,
    ATTENTION_MODE_MULTIHEAD_HAZARD_CONDITIONED,
    ATTENTION_NORMALIZATION_TARGET_NODE_RELATION,
    EDGE_NORMALIZATION_TARGET_DEGREE,
    RELATION_GATE_ACTIVATION_SIGMOID,
    RELATION_GATE_SCOPE_TARGET_NODE,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.functional_message_passing import (
    schemas as fmp_schemas,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.functional_message_passing.layer.diagnostics import (
    DEFAULT_LAYER_DIAGNOSTIC_THRESHOLDS,
    LAYER_DIAGNOSTICS_BUFFER_FREE,
    LAYER_DIAGNOSTICS_IMPLICIT_FORWARD_EXECUTION,
    LAYER_DIAGNOSTICS_INTERPRETATION,
    LAYER_DIAGNOSTICS_PARAMETER_FREE,
    LAYER_DIAGNOSTICS_SCHEMA_VERSION,
    LAYER_DIAGNOSTIC_SECTION_ALERTS,
    LAYER_DIAGNOSTIC_SECTION_BY_GRAPH,
    LAYER_DIAGNOSTIC_SECTION_GLOBAL,
    LAYER_DIAGNOSTIC_SECTION_LINEAGE,
    LAYER_DIAGNOSTIC_SECTION_REGULARIZATION,
    LAYER_DIAGNOSTIC_SECTION_STAGE_SUMMARIES,
    LAYER_DIAGNOSTIC_SECTION_TRACE,
    LayerDiagnosticThresholds,
    LayerDiagnostics,
    aggregation_diagnostic_summary,
    build_layer_diagnostic_report,
    build_layer_diagnostics,
    build_public_layer_diagnostic_report,
    derive_layer_alerts,
    graph_batch_diagnostics,
    incoming_edge_count_statistics,
    layer_diagnostic_report_fingerprint,
    layer_diagnostics_architecture_dict,
    layer_diagnostics_architecture_fingerprint,
    layer_lineage_summary,
    layer_trace_diagnostic_summary,
    matrix_statistics,
    regularization_diagnostic_summary,
    scalar_tensor_statistics,
    state_transition_statistics,
    validate_layer_diagnostic_report,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.functional_message_passing.layer.normalization import (
    LAYER_DISABLED_NORMALIZATION_IDENTITY_POLICY,
    LAYER_NORMALIZER_AGGREGATION_OWNED_HERE,
    LAYER_NORMALIZER_DEFAULT_BIAS_ENABLED,
    LAYER_NORMALIZER_DEFAULT_ELEMENTWISE_AFFINE,
    LAYER_NORMALIZER_DEFAULT_EPSILON,
    LAYER_NORMALIZER_DROPOUT_OWNED_HERE,
    LAYER_NORMALIZER_MULTI_LAYER_ITERATION_OWNED_HERE,
    LAYER_NORMALIZER_NORMALIZED_AXIS,
    LAYER_NORMALIZER_OPERATION,
    LAYER_NORMALIZER_OPERATION_ORDER,
    LAYER_NORMALIZER_RESIDUAL_OWNED_HERE,
    LAYER_NORMALIZER_SCHEMA_VERSION,
    LAYER_NORMALIZER_STATISTIC_SCOPE,
    LAYER_NORMALIZER_VARIANCE_ESTIMATOR,
    FunctionalLayerNormalizer,
    LayerNormalizer,
    MessagePassingLayerNormalizer,
    PostResidualLayerNormalizer,
    apply_layer_normalization,
    apply_normalization,
    build_layer_normalization_output,
    build_layer_normalizer,
    build_layer_normalizer_from_flag,
    build_normalizer,
    build_normalizer_from_flag,
    layer_normalization_diagnostic_summary,
    layer_normalizer_architecture_dict,
    layer_normalizer_architecture_fingerprint,
    normalization_enabled_from_mode,
    normalization_mode_from_enabled,
    normalize_layer_state,
    resolve_layer_normalization,
    validate_layer_normalization_output,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.functional_message_passing.layer.residual_update import (
    LAYER_DISABLED_DROPOUT_IDENTITY_POLICY,
    LAYER_DISABLED_RESIDUAL_IDENTITY_POLICY,
    LAYER_DROPOUT_SEMANTICS,
    LAYER_RESIDUAL_UPDATER_AGGREGATION_OWNED_HERE,
    LAYER_RESIDUAL_UPDATER_BUFFER_FREE,
    LAYER_RESIDUAL_UPDATER_NORMALIZATION_OWNED_HERE,
    LAYER_RESIDUAL_UPDATER_OPERATION,
    LAYER_RESIDUAL_UPDATER_OPERATION_ORDER,
    LAYER_RESIDUAL_UPDATER_PARAMETER_FREE,
    LAYER_RESIDUAL_UPDATER_PROJECTION_OWNED_HERE,
    LAYER_RESIDUAL_UPDATER_SCHEMA_VERSION,
    FunctionalResidualUpdater,
    LayerResidualUpdater,
    MessagePassingResidualUpdater,
    ResidualUpdater,
    apply_layer_residual,
    apply_layer_residual_update,
    apply_layer_update_dropout,
    apply_residual,
    apply_update_dropout,
    build_layer_residual_update_output,
    build_layer_residual_updater,
    build_layer_residual_updater_from_flags,
    build_residual_updater,
    build_residual_updater_from_flags,
    layer_residual_update_diagnostic_summary,
    layer_residual_updater_architecture_dict,
    layer_residual_updater_architecture_fingerprint,
    residual_enabled_from_mode,
    residual_mode_from_enabled,
    resolve_layer_residual_update,
    validate_layer_residual_update_output,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.functional_message_passing.layer.schemas import (
    CANONICAL_LAYER_NORMALIZATION_MODES,
    CANONICAL_LAYER_NORMALIZATION_POSITIONS,
    CANONICAL_LAYER_RESIDUAL_MODES,
    CANONICAL_LAYER_TRACE_MODES,
    LAYER_ADDITIVE_RESIDUAL_FORMULA,
    LAYER_AGGREGATE_LAYOUT,
    LAYER_COMPUTATION_OUTPUT_SCHEMA_VERSION,
    LAYER_DISABLED_RESIDUAL_FORMULA,
    LAYER_INPUT_LAYOUT,
    LAYER_INPUTS_SCHEMA_VERSION,
    LAYER_INTERMEDIATE_TRACE_SCHEMA_VERSION,
    LAYER_NORMALIZATION_LAYER_NORM,
    LAYER_NORMALIZATION_NONE,
    LAYER_NORMALIZATION_OUTPUT_SCHEMA_VERSION,
    LAYER_NORMALIZATION_POST_RESIDUAL,
    LAYER_NORMALIZATION_PRE_RESIDUAL,
    LAYER_OUTPUT_LAYOUT,
    LAYER_POST_NORMALIZATION_FORMULA,
    LAYER_RESIDUAL_ADDITIVE,
    LAYER_RESIDUAL_DISABLED,
    LAYER_RESIDUAL_UPDATE_SCHEMA_VERSION,
    LAYER_SCIENTIFIC_INTERPRETATION,
    LAYER_TRACE_FULL,
    LAYER_TRACE_NODE,
    LAYER_TRACE_NONE,
    LAYER_TRACE_POLICY_SCHEMA_VERSION,
    LAYER_UPDATE_BRANCH_FORMULA,
    FunctionalMessagePassingLayerComputation,
    FunctionalMessagePassingLayerInputs,
    FunctionalMessagePassingLayerStages,
    FunctionalMessagePassingLayerTrace,
    LayerComputationOutput,
    LayerInputs,
    LayerNormalizationOutput,
    LayerResidualUpdateOutput,
    LayerStages,
    LayerTrace,
    LayerTracePolicy,
    NormalizationOutput,
    ResidualUpdateOutput,
    build_public_layer_intermediates,
    layer_schema_architecture_dict,
    layer_schema_architecture_fingerprint,
    validate_layer_stage_chain,
    validate_public_layer_output,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.functional_message_passing.message_builders import (
    MESSAGE_SEMANTIC_EDGE_POLICY_IGNORE,
    MESSAGE_SEMANTIC_EDGE_POLICY_USE_SOURCE_GRAPH,
    MessageBuilderRun,
    build_edge_message_builder,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.functional_message_passing.message_builders.diagnostics import (
    DEFAULT_MESSAGE_BUILDER_DIAGNOSTIC_THRESHOLDS,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.functional_message_passing.schemas import (
    AggregationOutput,
    EdgeAttentionOutput,
    EdgeMessageOutput,
    FunctionalMessagePassingInputs,
    FunctionalMessagePassingIntermediates,
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
            EDGE_NORMALIZATION_TARGET_DEGREE
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
# Layer-component fixture helpers
# =============================================================================


def _message_builder_run(
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
    empty_edges: bool = False,
    semantic_policy: str = MESSAGE_SEMANTIC_EDGE_POLICY_IGNORE,
    state: torch.Tensor | None = None,
    transformed_state: torch.Tensor | None = None,
    source_fingerprint: str = "layer-component-input",
) -> tuple[
    MessageBuilderRun,
    FunctionalMessagePassingInputs,
]:
    inputs = _inputs(
        dtype=dtype,
        device=device,
        empty_edges=empty_edges,
        state=state,
        source_fingerprint=source_fingerprint,
    )
    transform = _relation_transform(
        inputs,
        tensor=transformed_state,
    )
    normalization = _edge_normalization(
        inputs
    )
    gate = _relation_gate(
        inputs
    )
    attention = _edge_attention(
        inputs
    )
    builder = build_edge_message_builder(
        semantic_edge_policy=(
            semantic_policy
        ),
        diagnostics_enabled=False,
    )

    run = builder.run_complete(
        relation_transform=transform,
        edge_normalization=normalization,
        relation_gate=gate,
        edge_attention=attention,
        source_inputs=inputs,
    )

    return run, inputs


def _aggregation(
    run: MessageBuilderRun,
    *,
    architecture_fingerprint: str = "aggregation-architecture",
) -> AggregationOutput:
    messages = run.public_output
    inputs = messages.source_inputs
    node_aggregate = torch.zeros(
        (
            inputs.num_nodes,
            inputs.hidden_dim,
        ),
        dtype=messages.edge_messages.dtype,
        device=messages.edge_messages.device,
    )
    node_aggregate.index_add_(
        0,
        inputs.target_index,
        messages.edge_messages,
    )
    incoming_edge_count = torch.bincount(
        inputs.target_index,
        minlength=inputs.num_nodes,
    )
    denominator = (
        incoming_edge_count
        .clamp_min(1)
        .to(
            dtype=node_aggregate.dtype
        )
        .unsqueeze(-1)
    )
    node_aggregate = (
        node_aggregate
        / denominator
    )

    return AggregationOutput(
        node_aggregate=node_aggregate,
        incoming_edge_count=(
            incoming_edge_count
        ),
        source_messages=messages,
        aggregation_mode=(
            AGGREGATION_MEAN
        ),
        encoder_architecture_fingerprint=(
            architecture_fingerprint
        ),
    )


def _layer_inputs(
    inputs: FunctionalMessagePassingInputs,
    *,
    layer_index: int = 2,
    trace_mode: str = LAYER_TRACE_NONE,
    training: bool = True,
    source_stack_fingerprint: str | None = (
        "stack-architecture"
    ),
) -> FunctionalMessagePassingLayerInputs:
    return FunctionalMessagePassingLayerInputs(
        source_inputs=inputs,
        layer_index=layer_index,
        trace_policy=LayerTracePolicy(
            mode=trace_mode
        ),
        training=training,
        source_stack_fingerprint=(
            source_stack_fingerprint
        ),
    )


def _residual_output(
    aggregation: AggregationOutput,
    layer_inputs: FunctionalMessagePassingLayerInputs,
    *,
    residual_mode: str = LAYER_RESIDUAL_ADDITIVE,
    dropout_probability: float = 0.0,
    training: bool | None = None,
) -> LayerResidualUpdateOutput:
    resolved_training = (
        layer_inputs.training
        if training is None
        else training
    )
    return build_layer_residual_update_output(
        aggregation=aggregation,
        layer_inputs=layer_inputs,
        residual_mode=residual_mode,
        dropout_probability=(
            dropout_probability
        ),
        training=resolved_training,
    )


def _normalization_output(
    residual_update: LayerResidualUpdateOutput,
    *,
    normalization_mode: str = LAYER_NORMALIZATION_NONE,
    epsilon: float = LAYER_NORMALIZER_DEFAULT_EPSILON,
    affine: bool = False,
) -> tuple[
    LayerNormalizationOutput,
    LayerNormalizer | None,
]:
    if normalization_mode == (
        LAYER_NORMALIZATION_NONE
    ):
        return (
            build_layer_normalization_output(
                residual_update=residual_update,
                normalization_mode=(
                    normalization_mode
                ),
                epsilon=epsilon,
            ),
            None,
        )

    normalizer = LayerNormalizer(
        residual_update.hidden_dim,
        normalization_mode=(
            normalization_mode
        ),
        epsilon=epsilon,
        elementwise_affine=affine,
        bias_enabled=affine,
        device=residual_update.device,
        dtype=residual_update.dtype,
    )
    return (
        normalizer(
            residual_update
        ),
        normalizer,
    )


def _internal_output(
    *,
    trace_mode: str = LAYER_TRACE_NONE,
    residual_mode: str = LAYER_RESIDUAL_ADDITIVE,
    normalization_mode: str = LAYER_NORMALIZATION_NONE,
    normalization_affine: bool = False,
    dropout_probability: float = 0.0,
    training: bool = True,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
    empty_edges: bool = False,
    regularization_terms: dict[
        str,
        torch.Tensor,
    ] | None = None,
) -> tuple[
    LayerComputationOutput,
    MessageBuilderRun,
    LayerNormalizer | None,
]:
    run, inputs = _message_builder_run(
        dtype=dtype,
        device=device,
        empty_edges=empty_edges,
    )
    aggregation = _aggregation(
        run
    )
    layer_inputs = _layer_inputs(
        inputs,
        trace_mode=trace_mode,
        training=training,
    )
    residual = _residual_output(
        aggregation,
        layer_inputs,
        residual_mode=residual_mode,
        dropout_probability=(
            dropout_probability
        ),
        training=training,
    )
    normalization, normalizer = (
        _normalization_output(
            residual,
            normalization_mode=(
                normalization_mode
            ),
            affine=normalization_affine,
        )
    )

    trace: (
        FunctionalMessagePassingLayerTrace
        | None
    )

    if trace_mode == LAYER_TRACE_NONE:
        trace = None
    elif trace_mode == LAYER_TRACE_NODE:
        trace = (
            FunctionalMessagePassingLayerTrace(
                layer_inputs=layer_inputs,
                aggregation=aggregation,
                residual_update=residual,
                normalization=normalization,
            )
        )
    elif trace_mode == LAYER_TRACE_FULL:
        trace = (
            FunctionalMessagePassingLayerTrace(
                layer_inputs=layer_inputs,
                aggregation=aggregation,
                residual_update=residual,
                normalization=normalization,
                relation_transform=(
                    run
                    .composition_output
                    .relation_transform
                ),
                edge_normalization=(
                    run
                    .resolved_coefficients
                    .edge_normalization
                ),
                relation_gate=(
                    run
                    .resolved_coefficients
                    .relation_gate
                ),
                edge_attention=(
                    run
                    .resolved_coefficients
                    .edge_attention
                ),
                edge_messages=(
                    run.public_output
                ),
                message_builder_run=run,
            )
        )
    else:
        raise AssertionError(
            "Unsupported test trace mode."
        )

    terms = (
        {
            "layer_regularizer": torch.tensor(
                0.125,
                dtype=dtype,
                device=device,
            )
        }
        if regularization_terms is None
        else regularization_terms
    )

    internal = LayerComputationOutput(
        updated_node_state=(
            normalization.output_state
        ),
        layer_inputs=layer_inputs,
        aggregation=aggregation,
        residual_update=residual,
        normalization=normalization,
        layer_architecture_fingerprint=(
            "layer-component-architecture"
        ),
        layer_parameter_fingerprint=(
            normalizer.parameter_fingerprint()
            if normalizer is not None
            else None
        ),
        lineage_fingerprint=(
            "layer-component-lineage"
        ),
        trace=trace,
        regularization_terms=terms,
    )

    return internal, run, normalizer


def _public_output(
    internal: LayerComputationOutput,
) -> FunctionalMessagePassingLayerOutput:
    intermediates = (
        build_public_layer_intermediates(
            internal.trace
        )
        if (
            internal.trace is not None
            and internal.trace.trace_mode
            == LAYER_TRACE_FULL
        )
        else None
    )

    return FunctionalMessagePassingLayerOutput(
        updated_node_state=(
            internal.updated_node_state
        ),
        node_aggregate=(
            internal.node_aggregate
        ),
        incoming_edge_count=(
            internal.incoming_edge_count
        ),
        source_inputs=(
            internal.source_inputs
        ),
        layer_index=(
            internal.layer_index
        ),
        residual_enabled=(
            internal.residual_enabled
        ),
        layer_norm_enabled=(
            internal.layer_norm_enabled
        ),
        encoder_architecture_fingerprint=(
            internal
            .layer_architecture_fingerprint
        ),
        lineage_fingerprint=(
            internal.lineage_fingerprint
        ),
        intermediates=intermediates,
        regularization_terms=(
            internal
            .regularization_terms
        ),
    )


def _assert_parameter_and_buffer_free(
    module: nn.Module,
) -> None:
    assert tuple(
        module.named_parameters()
    ) == ()
    assert tuple(
        module.named_buffers()
    ) == ()
    assert len(
        module.state_dict()
    ) == 0


def _assert_tensor_free(
    value: Any,
) -> None:
    if isinstance(
        value,
        torch.Tensor,
    ):
        raise AssertionError(
            "Report unexpectedly retained a tensor."
        )

    if isinstance(
        value,
        nn.Module,
    ):
        raise AssertionError(
            "Report unexpectedly retained a module."
        )

    if isinstance(
        value,
        dict,
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
# Schema identity and trace policy
# =============================================================================


def test_layer_schema_versions_and_layouts() -> None:
    assert LAYER_INPUTS_SCHEMA_VERSION.strip()
    assert LAYER_TRACE_POLICY_SCHEMA_VERSION.strip()
    assert LAYER_RESIDUAL_UPDATE_SCHEMA_VERSION.strip()
    assert LAYER_NORMALIZATION_OUTPUT_SCHEMA_VERSION.strip()
    assert LAYER_INTERMEDIATE_TRACE_SCHEMA_VERSION.strip()
    assert LAYER_COMPUTATION_OUTPUT_SCHEMA_VERSION.strip()
    assert LAYER_INPUT_LAYOUT == "node_state_[N,H]"
    assert LAYER_AGGREGATE_LAYOUT == (
        "target_node_aggregate_[N,H]"
    )
    assert LAYER_OUTPUT_LAYOUT == (
        "updated_node_state_[N,H]"
    )
    assert LAYER_SCIENTIFIC_INTERPRETATION == (
        "one_functional_message_passing_state_update"
    )


def test_layer_schema_equations_are_explicit() -> None:
    assert "dropout" in (
        LAYER_UPDATE_BRANCH_FORMULA
    )
    assert "+" in (
        LAYER_ADDITIVE_RESIDUAL_FORMULA
    )
    assert LAYER_DISABLED_RESIDUAL_FORMULA == (
        "post_residual_state = post_dropout_update"
    )
    assert "normalization" in (
        LAYER_POST_NORMALIZATION_FORMULA
    )


def test_layer_schema_aliases_are_exact() -> None:
    assert LayerInputs is (
        FunctionalMessagePassingLayerInputs
    )
    assert ResidualUpdateOutput is (
        LayerResidualUpdateOutput
    )
    assert NormalizationOutput is (
        LayerNormalizationOutput
    )
    assert LayerTrace is (
        FunctionalMessagePassingLayerTrace
    )
    assert (
        FunctionalMessagePassingLayerComputation
        is LayerComputationOutput
    )
    assert LayerStages is (
        FunctionalMessagePassingLayerStages
    )


@pytest.mark.parametrize(
    "mode",
    CANONICAL_LAYER_TRACE_MODES,
)
def test_trace_policy_modes(
    mode: str,
) -> None:
    policy = LayerTracePolicy(
        mode=mode
    )

    assert policy.mode == mode
    assert policy.enabled is (
        mode != LAYER_TRACE_NONE
    )
    assert policy.retain_node_stages is (
        mode in (
            LAYER_TRACE_NODE,
            LAYER_TRACE_FULL,
        )
    )
    assert policy.retain_edge_stages is (
        mode == LAYER_TRACE_FULL
    )
    assert (
        policy.capture_intermediate_messages
        is (
            mode == LAYER_TRACE_FULL
        )
    )
    policy.assert_implemented()


@pytest.mark.parametrize(
    ("capture", "expected"),
    (
        (False, LAYER_TRACE_NONE),
        (True, LAYER_TRACE_FULL),
    ),
)
def test_trace_policy_from_historical_capture_flag(
    capture: bool,
    expected: str,
) -> None:
    policy = (
        LayerTracePolicy
        .from_capture_intermediate_messages(
            capture
        )
    )
    assert policy.mode == expected


def test_trace_policy_architecture_is_deterministic() -> None:
    policy = LayerTracePolicy(
        mode=LAYER_TRACE_FULL
    )

    assert (
        policy.architecture_dict()
        == policy.architecture_dict()
    )
    assert (
        policy.architecture_fingerprint()
        == policy.architecture_fingerprint()
    )


def test_layer_inputs_preserve_exact_source_identity() -> None:
    _run_value, inputs = (
        _message_builder_run()
    )
    policy = LayerTracePolicy(
        mode=LAYER_TRACE_NODE
    )
    layer_inputs = (
        FunctionalMessagePassingLayerInputs(
            source_inputs=inputs,
            layer_index=3,
            trace_policy=policy,
            training=False,
            source_stack_fingerprint=(
                "stack-fingerprint"
            ),
        )
    )

    assert layer_inputs.source_inputs is inputs
    assert (
        layer_inputs.input_node_state
        is inputs.node_state.fused_state
    )
    assert layer_inputs.num_nodes == NODES
    assert layer_inputs.num_edges == EDGES
    assert layer_inputs.hidden_dim == HIDDEN_DIM
    assert layer_inputs.dtype == (
        inputs.dtype
    )
    assert layer_inputs.device == (
        inputs.device
    )
    assert layer_inputs.layer_index == 3
    assert layer_inputs.trace_policy is policy
    assert layer_inputs.training is False
    assert (
        layer_inputs.lineage_fingerprint()
        == layer_inputs.lineage_fingerprint()
    )


def test_layer_schema_architecture_separates_scope() -> None:
    architecture = (
        layer_schema_architecture_dict()
    )

    assert architecture[
        "canonical_trace_modes"
    ] == list(
        CANONICAL_LAYER_TRACE_MODES
    )
    assert architecture[
        "canonical_residual_modes"
    ] == list(
        CANONICAL_LAYER_RESIDUAL_MODES
    )
    assert architecture[
        "canonical_normalization_modes"
    ] == list(
        CANONICAL_LAYER_NORMALIZATION_MODES
    )
    assert architecture[
        "canonical_normalization_positions"
    ] == list(
        CANONICAL_LAYER_NORMALIZATION_POSITIONS
    )
    assert architecture[
        "multi_layer_iteration_owned_here"
    ] is False
    assert architecture[
        "prediction_owned_here"
    ] is False
    assert architecture[
        "claims_causal_importance"
    ] is False
    assert (
        layer_schema_architecture_fingerprint()
        == layer_schema_architecture_fingerprint()
    )


# =============================================================================
# Existing aggregation boundary
# =============================================================================


def test_mean_aggregation_helper_matches_exact_equation() -> None:
    run, inputs = _message_builder_run()
    aggregation = _aggregation(
        run
    )

    expected_sum = torch.zeros(
        (
            inputs.num_nodes,
            inputs.hidden_dim,
        ),
        dtype=inputs.dtype,
        device=inputs.device,
    )
    expected_sum.index_add_(
        0,
        inputs.target_index,
        run.public_output.edge_messages,
    )
    counts = torch.bincount(
        inputs.target_index,
        minlength=inputs.num_nodes,
    )
    expected = expected_sum / (
        counts
        .clamp_min(1)
        .to(
            dtype=inputs.dtype
        )
        .unsqueeze(-1)
    )

    torch.testing.assert_close(
        aggregation.node_aggregate,
        expected,
    )
    assert torch.equal(
        aggregation.incoming_edge_count,
        counts,
    )
    assert aggregation.source_messages is (
        run.public_output
    )


def test_empty_edges_produce_exact_zero_aggregates() -> None:
    run, inputs = _message_builder_run(
        empty_edges=True
    )
    aggregation = _aggregation(
        run
    )

    assert aggregation.node_aggregate.shape == (
        NODES,
        HIDDEN_DIM,
    )
    assert torch.equal(
        aggregation.node_aggregate,
        torch.zeros_like(
            aggregation.node_aggregate
        ),
    )
    assert torch.equal(
        aggregation.incoming_edge_count,
        torch.zeros(
            NODES,
            dtype=torch.long,
            device=inputs.device,
        ),
    )


def test_aggregation_diagnostics_report_isolated_nodes() -> None:
    run, _inputs_value = (
        _message_builder_run()
    )
    aggregation = _aggregation(
        run
    )
    summary = (
        aggregation_diagnostic_summary(
            aggregation
        )
    )

    assert summary[
        "aggregation_mode"
    ] == AGGREGATION_MEAN
    assert summary["num_nodes"] == NODES
    assert summary["num_edges"] == EDGES
    assert summary[
        "incoming_edge_count"
    ][
        "total_incoming_edges"
    ] == EDGES
    assert summary[
        "residual_performed_here"
    ] is False
    assert summary[
        "normalization_performed_here"
    ] is False
    assert summary[
        "causal_importance_claim"
    ] is False


# =============================================================================
# Residual mode and architecture
# =============================================================================


@pytest.mark.parametrize(
    ("enabled", "mode"),
    (
        (False, LAYER_RESIDUAL_DISABLED),
        (True, LAYER_RESIDUAL_ADDITIVE),
    ),
)
def test_residual_mode_conversion(
    enabled: bool,
    mode: str,
) -> None:
    assert residual_mode_from_enabled(
        enabled
    ) == mode
    assert residual_enabled_from_mode(
        mode
    ) is enabled


def test_residual_public_identity() -> None:
    assert (
        LAYER_RESIDUAL_UPDATER_SCHEMA_VERSION
        .strip()
    )
    assert LAYER_RESIDUAL_UPDATER_OPERATION == (
        "dropout_then_optional_additive_residual_update"
    )
    assert (
        LAYER_RESIDUAL_UPDATER_OPERATION_ORDER[
            -1
        ]
        == "construct_layer_residual_update_output"
    )
    assert LAYER_RESIDUAL_UPDATER_PARAMETER_FREE is True
    assert LAYER_RESIDUAL_UPDATER_BUFFER_FREE is True
    assert (
        LAYER_RESIDUAL_UPDATER_PROJECTION_OWNED_HERE
        is False
    )
    assert (
        LAYER_RESIDUAL_UPDATER_AGGREGATION_OWNED_HERE
        is False
    )
    assert (
        LAYER_RESIDUAL_UPDATER_NORMALIZATION_OWNED_HERE
        is False
    )
    assert "inverted_dropout" in (
        LAYER_DROPOUT_SEMANTICS
    )
    assert (
        LAYER_DISABLED_DROPOUT_IDENTITY_POLICY
        == "exact_input_tensor_identity"
    )
    assert (
        LAYER_DISABLED_RESIDUAL_IDENTITY_POLICY
        == "exact_post_dropout_tensor_identity"
    )


def test_residual_aliases_are_exact() -> None:
    assert ResidualUpdater is (
        LayerResidualUpdater
    )
    assert FunctionalResidualUpdater is (
        LayerResidualUpdater
    )
    assert MessagePassingResidualUpdater is (
        LayerResidualUpdater
    )
    assert apply_update_dropout is (
        apply_layer_update_dropout
    )
    assert apply_residual is (
        apply_layer_residual
    )
    assert build_residual_updater is (
        build_layer_residual_updater
    )
    assert (
        build_residual_updater_from_flags
        is build_layer_residual_updater_from_flags
    )


@pytest.mark.parametrize(
    "mode",
    CANONICAL_LAYER_RESIDUAL_MODES,
)
def test_residual_architecture_is_deterministic(
    mode: str,
) -> None:
    first = (
        layer_residual_updater_architecture_dict(
            residual_mode=mode,
            dropout_probability=0.25,
        )
    )
    second = (
        layer_residual_updater_architecture_dict(
            residual_mode=mode,
            dropout_probability=0.25,
        )
    )

    assert first == second
    assert (
        layer_residual_updater_architecture_fingerprint(
            residual_mode=mode,
            dropout_probability=0.25,
        )
        == layer_residual_updater_architecture_fingerprint(
            residual_mode=mode,
            dropout_probability=0.25,
        )
    )
    assert first[
        "aggregation_owned_here"
    ] is False
    assert first[
        "normalization_owned_here"
    ] is False


def test_residual_architecture_changes_with_mode_or_dropout() -> None:
    additive = (
        layer_residual_updater_architecture_fingerprint(
            residual_mode=(
                LAYER_RESIDUAL_ADDITIVE
            ),
            dropout_probability=0.0,
        )
    )
    disabled = (
        layer_residual_updater_architecture_fingerprint(
            residual_mode=(
                LAYER_RESIDUAL_DISABLED
            ),
            dropout_probability=0.0,
        )
    )
    dropout = (
        layer_residual_updater_architecture_fingerprint(
            residual_mode=(
                LAYER_RESIDUAL_ADDITIVE
            ),
            dropout_probability=0.5,
        )
    )

    assert additive != disabled
    assert additive != dropout


# =============================================================================
# Residual low-level numerics
# =============================================================================


def test_zero_dropout_preserves_exact_identity() -> None:
    update = torch.randn(
        5,
        4,
        requires_grad=True,
    )

    output = apply_layer_update_dropout(
        update,
        dropout_probability=0.0,
        training=True,
    )

    assert output is update


def test_eval_dropout_preserves_exact_identity() -> None:
    update = torch.randn(
        5,
        4,
        requires_grad=True,
    )

    output = apply_layer_update_dropout(
        update,
        dropout_probability=0.75,
        training=False,
    )

    assert output is update


def test_training_dropout_matches_pytorch_with_fixed_seed() -> None:
    update = torch.arange(
        1,
        21,
        dtype=torch.float32,
    ).reshape(5, 4)

    torch.manual_seed(41)
    expected = F.dropout(
        update,
        p=0.4,
        training=True,
        inplace=False,
    )
    torch.manual_seed(41)
    observed = (
        apply_layer_update_dropout(
            update,
            dropout_probability=0.4,
            training=True,
        )
    )

    torch.testing.assert_close(
        observed,
        expected,
    )
    assert observed is not update


def test_disabled_residual_preserves_exact_update_identity() -> None:
    source = torch.randn(
        5,
        4,
    )
    update = torch.randn(
        5,
        4,
    )

    output = apply_layer_residual(
        residual_source_state=source,
        post_dropout_update=update,
        residual_mode=(
            LAYER_RESIDUAL_DISABLED
        ),
    )

    assert output is update


def test_additive_residual_matches_elementwise_sum() -> None:
    source = torch.tensor(
        [
            [1.0, 2.0],
            [-3.0, 4.0],
        ]
    )
    update = torch.tensor(
        [
            [0.5, -1.0],
            [2.0, 3.0],
        ]
    )

    output = apply_layer_residual(
        residual_source_state=source,
        post_dropout_update=update,
        residual_mode=(
            LAYER_RESIDUAL_ADDITIVE
        ),
    )

    torch.testing.assert_close(
        output,
        source + update,
    )
    assert output is not source
    assert output is not update


def test_additive_residual_analytical_gradients() -> None:
    source = torch.randn(
        3,
        4,
        requires_grad=True,
    )
    update = torch.randn(
        3,
        4,
        requires_grad=True,
    )

    output = apply_layer_residual(
        residual_source_state=source,
        post_dropout_update=update,
        residual_mode=(
            LAYER_RESIDUAL_ADDITIVE
        ),
    )
    output.sum().backward()

    torch.testing.assert_close(
        source.grad,
        torch.ones_like(source),
    )
    torch.testing.assert_close(
        update.grad,
        torch.ones_like(update),
    )


# =============================================================================
# Complete residual stage and module
# =============================================================================


@pytest.mark.parametrize(
    "residual_mode",
    CANONICAL_LAYER_RESIDUAL_MODES,
)
def test_complete_residual_stage_preserves_lineage(
    residual_mode: str,
) -> None:
    run, inputs = _message_builder_run()
    aggregation = _aggregation(
        run
    )
    layer_inputs = _layer_inputs(
        inputs,
        training=True,
    )

    output = (
        build_layer_residual_update_output(
            aggregation=aggregation,
            layer_inputs=layer_inputs,
            residual_mode=residual_mode,
            dropout_probability=0.0,
            training=True,
        )
    )

    assert output.aggregation is aggregation
    assert output.layer_inputs is layer_inputs
    assert (
        output.pre_dropout_update
        is aggregation.node_aggregate
    )
    assert (
        output.post_dropout_update
        is output.pre_dropout_update
    )
    assert (
        output.pre_residual_state
        is output.post_dropout_update
    )
    assert (
        output.residual_source_state
        is inputs.node_state.fused_state
    )
    assert output.residual_enabled is (
        residual_mode
        == LAYER_RESIDUAL_ADDITIVE
    )

    validate_layer_residual_update_output(
        output=output,
        aggregation=aggregation,
        layer_inputs=layer_inputs,
        residual_mode=residual_mode,
        dropout_probability=0.0,
        training=True,
        updater_architecture_fingerprint=(
            output
            .updater_architecture_fingerprint
        ),
    )


def test_complete_disabled_residual_preserves_output_identity() -> None:
    run, inputs = _message_builder_run()
    aggregation = _aggregation(
        run
    )
    layer_inputs = _layer_inputs(
        inputs
    )
    output = _residual_output(
        aggregation,
        layer_inputs,
        residual_mode=(
            LAYER_RESIDUAL_DISABLED
        ),
    )

    assert (
        output.post_residual_state
        is output.post_dropout_update
    )


def test_residual_module_builders_and_flags() -> None:
    direct = build_layer_residual_updater(
        residual_mode=(
            LAYER_RESIDUAL_DISABLED
        ),
        dropout_probability=0.2,
    )
    alias = build_residual_updater(
        residual_mode=(
            LAYER_RESIDUAL_DISABLED
        ),
        dropout_probability=0.2,
    )
    flagged = (
        build_layer_residual_updater_from_flags(
            residual_enabled=True,
            dropout_probability=0.3,
        )
    )
    flagged_alias = (
        build_residual_updater_from_flags(
            residual_enabled=True,
            dropout_probability=0.3,
        )
    )

    assert isinstance(
        direct,
        LayerResidualUpdater,
    )
    assert isinstance(
        alias,
        LayerResidualUpdater,
    )
    assert direct.residual_enabled is False
    assert flagged.residual_enabled is True
    assert flagged_alias.residual_enabled is True


def test_residual_module_is_parameter_and_buffer_free() -> None:
    updater = LayerResidualUpdater(
        residual_mode=(
            LAYER_RESIDUAL_ADDITIVE
        ),
        dropout_probability=0.25,
    )

    _assert_parameter_and_buffer_free(
        updater
    )
    assert updater.parameter_count == 0
    assert (
        updater.trainable_parameter_count
        == 0
    )
    assert updater.buffer_count == 0
    assert updater.parameter_fingerprint is None
    updater.assert_parameter_free()


def test_residual_module_train_eval_contract() -> None:
    run, inputs = _message_builder_run()
    aggregation = _aggregation(
        run
    )
    updater = LayerResidualUpdater(
        dropout_probability=0.5
    )

    updater.eval()
    eval_inputs = _layer_inputs(
        inputs,
        training=False,
    )
    eval_output = updater(
        aggregation=aggregation,
        layer_inputs=eval_inputs,
    )
    assert (
        eval_output.post_dropout_update
        is aggregation.node_aggregate
    )

    updater.train()
    train_inputs = _layer_inputs(
        inputs,
        training=True,
    )
    torch.manual_seed(17)
    train_output = updater(
        aggregation=aggregation,
        layer_inputs=train_inputs,
    )
    assert train_output.training is True
    assert (
        train_output
        .updater_architecture_fingerprint
        == updater
        .architecture_fingerprint()
    )


def test_residual_functional_alias_matches_complete_stage() -> None:
    run, inputs = _message_builder_run()
    aggregation = _aggregation(
        run
    )
    layer_inputs = _layer_inputs(
        inputs
    )

    first = apply_layer_residual_update(
        aggregation=aggregation,
        layer_inputs=layer_inputs,
        residual_mode=(
            LAYER_RESIDUAL_ADDITIVE
        ),
        dropout_probability=0.0,
        training=True,
    )
    second = resolve_layer_residual_update(
        aggregation=aggregation,
        layer_inputs=layer_inputs,
        residual_mode=(
            LAYER_RESIDUAL_ADDITIVE
        ),
        dropout_probability=0.0,
        training=True,
    )

    torch.testing.assert_close(
        first.post_residual_state,
        second.post_residual_state,
    )


def test_residual_diagnostics_are_descriptive() -> None:
    run, inputs = _message_builder_run()
    aggregation = _aggregation(
        run
    )
    layer_inputs = _layer_inputs(
        inputs
    )
    output = _residual_output(
        aggregation,
        layer_inputs,
    )
    summary = (
        layer_residual_update_diagnostic_summary(
            output
        )
    )

    assert summary["num_nodes"] == NODES
    assert summary["hidden_dim"] == HIDDEN_DIM
    assert summary["residual_enabled"] is True
    assert summary[
        "exact_identity"
    ][
        "pre_dropout_is_exact_aggregate"
    ] is True
    assert summary[
        "aggregation_performed_here"
    ] is False
    assert summary[
        "normalization_performed_here"
    ] is False
    assert summary[
        "causal_importance_claim"
    ] is False


def test_residual_stage_supports_empty_edges_and_float64() -> None:
    run, inputs = _message_builder_run(
        empty_edges=True,
        dtype=torch.float64,
    )
    aggregation = _aggregation(
        run
    )
    layer_inputs = _layer_inputs(
        inputs
    )
    output = _residual_output(
        aggregation,
        layer_inputs,
    )

    assert output.dtype == torch.float64
    torch.testing.assert_close(
        output.post_residual_state,
        inputs.node_state.fused_state,
    )


# =============================================================================
# Normalization mode and architecture
# =============================================================================


@pytest.mark.parametrize(
    ("enabled", "mode"),
    (
        (False, LAYER_NORMALIZATION_NONE),
        (True, LAYER_NORMALIZATION_LAYER_NORM),
    ),
)
def test_normalization_mode_conversion(
    enabled: bool,
    mode: str,
) -> None:
    assert normalization_mode_from_enabled(
        enabled
    ) == mode
    assert normalization_enabled_from_mode(
        mode
    ) is enabled


def test_normalizer_public_identity() -> None:
    assert LAYER_NORMALIZER_SCHEMA_VERSION.strip()
    assert LAYER_NORMALIZER_OPERATION == (
        "optional_post_residual_feature_layer_normalization"
    )
    assert LAYER_NORMALIZER_OPERATION_ORDER[-1] == (
        "construct_layer_normalization_output"
    )
    assert LAYER_NORMALIZER_NORMALIZED_AXIS == -1
    assert "per_node" in (
        LAYER_NORMALIZER_STATISTIC_SCOPE
    )
    assert (
        LAYER_NORMALIZER_VARIANCE_ESTIMATOR
        == "biased_population_variance"
    )
    assert (
        LAYER_NORMALIZER_AGGREGATION_OWNED_HERE
        is False
    )
    assert LAYER_NORMALIZER_DROPOUT_OWNED_HERE is False
    assert LAYER_NORMALIZER_RESIDUAL_OWNED_HERE is False
    assert (
        LAYER_NORMALIZER_MULTI_LAYER_ITERATION_OWNED_HERE
        is False
    )
    assert (
        LAYER_DISABLED_NORMALIZATION_IDENTITY_POLICY
        == "exact_input_tensor_identity"
    )
    assert LAYER_NORMALIZER_DEFAULT_EPSILON > 0.0
    assert (
        LAYER_NORMALIZER_DEFAULT_ELEMENTWISE_AFFINE
        is True
    )
    assert LAYER_NORMALIZER_DEFAULT_BIAS_ENABLED is True


def test_normalizer_aliases_are_exact() -> None:
    assert FunctionalLayerNormalizer is (
        LayerNormalizer
    )
    assert MessagePassingLayerNormalizer is (
        LayerNormalizer
    )
    assert PostResidualLayerNormalizer is (
        LayerNormalizer
    )
    assert apply_normalization is (
        apply_layer_normalization
    )
    assert build_normalizer is (
        build_layer_normalizer
    )
    assert build_normalizer_from_flag is (
        build_layer_normalizer_from_flag
    )


@pytest.mark.parametrize(
    (
        "mode",
        "affine",
        "bias",
        "expected_parameters",
    ),
    (
        (
            LAYER_NORMALIZATION_NONE,
            False,
            False,
            0,
        ),
        (
            LAYER_NORMALIZATION_LAYER_NORM,
            False,
            False,
            0,
        ),
        (
            LAYER_NORMALIZATION_LAYER_NORM,
            True,
            False,
            HIDDEN_DIM,
        ),
        (
            LAYER_NORMALIZATION_LAYER_NORM,
            True,
            True,
            2 * HIDDEN_DIM,
        ),
    ),
)
def test_normalizer_architecture_parameter_counts(
    mode: str,
    affine: bool,
    bias: bool,
    expected_parameters: int,
) -> None:
    architecture = (
        layer_normalizer_architecture_dict(
            normalization_mode=mode,
            normalization_position=(
                LAYER_NORMALIZATION_POST_RESIDUAL
            ),
            hidden_dim=HIDDEN_DIM,
            epsilon=1e-5,
            elementwise_affine=affine,
            bias_enabled=bias,
        )
    )

    assert architecture[
        "parameter_count"
    ] == expected_parameters
    assert architecture[
        "aggregation_owned_here"
    ] is False
    assert architecture[
        "residual_owned_here"
    ] is False


def test_normalizer_architecture_is_deterministic() -> None:
    first = (
        layer_normalizer_architecture_fingerprint(
            normalization_mode=(
                LAYER_NORMALIZATION_LAYER_NORM
            ),
            normalization_position=(
                LAYER_NORMALIZATION_POST_RESIDUAL
            ),
            hidden_dim=HIDDEN_DIM,
            epsilon=1e-5,
            elementwise_affine=True,
            bias_enabled=True,
        )
    )
    second = (
        layer_normalizer_architecture_fingerprint(
            normalization_mode=(
                LAYER_NORMALIZATION_LAYER_NORM
            ),
            normalization_position=(
                LAYER_NORMALIZATION_POST_RESIDUAL
            ),
            hidden_dim=HIDDEN_DIM,
            epsilon=1e-5,
            elementwise_affine=True,
            bias_enabled=True,
        )
    )

    assert first == second


# =============================================================================
# Normalization low-level numerics
# =============================================================================


def test_disabled_normalization_preserves_exact_identity() -> None:
    state = torch.randn(
        5,
        4,
        requires_grad=True,
    )

    output = apply_layer_normalization(
        state,
        normalization_mode=(
            LAYER_NORMALIZATION_NONE
        ),
        epsilon=1e-5,
    )

    assert output is state


def test_non_affine_layer_norm_matches_pytorch() -> None:
    state = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [-2.0, 0.0, 2.0, 4.0],
        ]
    )

    observed = apply_layer_normalization(
        state,
        normalization_mode=(
            LAYER_NORMALIZATION_LAYER_NORM
        ),
        epsilon=1e-5,
    )
    expected = F.layer_norm(
        state,
        normalized_shape=(
            state.shape[-1],
        ),
        weight=None,
        bias=None,
        eps=1e-5,
    )

    torch.testing.assert_close(
        observed,
        expected,
    )


def test_affine_layer_norm_matches_pytorch() -> None:
    state = torch.randn(
        5,
        4,
    )
    weight = torch.tensor(
        [0.5, 1.0, 1.5, 2.0],
    )
    bias = torch.tensor(
        [-1.0, 0.0, 0.5, 1.0],
    )

    observed = apply_layer_normalization(
        state,
        normalization_mode=(
            LAYER_NORMALIZATION_LAYER_NORM
        ),
        epsilon=1e-4,
        weight=weight,
        bias=bias,
    )
    expected = F.layer_norm(
        state,
        normalized_shape=(4,),
        weight=weight,
        bias=bias,
        eps=1e-4,
    )

    torch.testing.assert_close(
        observed,
        expected,
    )


def test_non_affine_layer_norm_has_per_node_zero_mean_unit_variance() -> None:
    state = torch.randn(
        8,
        6,
        dtype=torch.float64,
    )

    output = apply_layer_normalization(
        state,
        normalization_mode=(
            LAYER_NORMALIZATION_LAYER_NORM
        ),
        epsilon=1e-12,
    )

    torch.testing.assert_close(
        output.mean(dim=-1),
        torch.zeros(
            8,
            dtype=torch.float64,
        ),
        atol=1e-10,
        rtol=1e-9,
    )
    torch.testing.assert_close(
        output.var(
            dim=-1,
            unbiased=False,
        ),
        torch.ones(
            8,
            dtype=torch.float64,
        ),
        atol=1e-8,
        rtol=1e-8,
    )


def test_layer_normalization_supports_empty_node_batch() -> None:
    state = torch.empty(
        0,
        HIDDEN_DIM,
    )

    output = apply_layer_normalization(
        state,
        normalization_mode=(
            LAYER_NORMALIZATION_LAYER_NORM
        ),
        epsilon=1e-5,
    )

    assert output.shape == (
        0,
        HIDDEN_DIM,
    )


def test_layer_normalization_preserves_autograd() -> None:
    state = torch.randn(
        5,
        4,
        requires_grad=True,
    )
    weight = torch.ones(
        4,
        requires_grad=True,
    )
    bias = torch.zeros(
        4,
        requires_grad=True,
    )

    output = apply_layer_normalization(
        state,
        normalization_mode=(
            LAYER_NORMALIZATION_LAYER_NORM
        ),
        epsilon=1e-5,
        weight=weight,
        bias=bias,
    )
    output.square().sum().backward()

    assert state.grad is not None
    assert weight.grad is not None
    assert bias.grad is not None
    assert bool(
        torch.isfinite(state.grad)
        .all()
        .item()
    )


# =============================================================================
# Complete normalization stage and module
# =============================================================================


def test_disabled_normalization_output_preserves_lineage_and_identity() -> None:
    run, inputs = _message_builder_run()
    aggregation = _aggregation(
        run
    )
    layer_inputs = _layer_inputs(
        inputs
    )
    residual = _residual_output(
        aggregation,
        layer_inputs,
    )

    output = build_layer_normalization_output(
        residual_update=residual,
        normalization_mode=(
            LAYER_NORMALIZATION_NONE
        ),
    )

    assert output.residual_update is residual
    assert (
        output.input_state
        is residual.post_residual_state
    )
    assert output.output_state is (
        output.input_state
    )
    assert output.normalization_enabled is False
    assert (
        output.normalizer_parameter_fingerprint
        is None
    )

    validate_layer_normalization_output(
        output=output,
        residual_update=residual,
        normalization_mode=(
            LAYER_NORMALIZATION_NONE
        ),
        normalization_position=(
            LAYER_NORMALIZATION_POST_RESIDUAL
        ),
        epsilon=(
            LAYER_NORMALIZER_DEFAULT_EPSILON
        ),
        normalizer_architecture_fingerprint=(
            output
            .normalizer_architecture_fingerprint
        ),
    )


def test_non_affine_normalization_output_matches_equation() -> None:
    run, inputs = _message_builder_run()
    aggregation = _aggregation(
        run
    )
    layer_inputs = _layer_inputs(
        inputs
    )
    residual = _residual_output(
        aggregation,
        layer_inputs,
    )

    output = build_layer_normalization_output(
        residual_update=residual,
        normalization_mode=(
            LAYER_NORMALIZATION_LAYER_NORM
        ),
        epsilon=1e-5,
    )

    expected = F.layer_norm(
        residual.post_residual_state,
        normalized_shape=(HIDDEN_DIM,),
        weight=None,
        bias=None,
        eps=1e-5,
    )
    torch.testing.assert_close(
        output.output_state,
        expected,
    )
    assert output.normalization_enabled is True
    assert (
        output.normalizer_parameter_fingerprint
        is not None
    )


def test_normalizer_default_parameter_initialization() -> None:
    normalizer = LayerNormalizer(
        HIDDEN_DIM
    )

    assert normalizer.weight is not None
    assert normalizer.bias is not None
    assert torch.equal(
        normalizer.weight,
        torch.ones_like(
            normalizer.weight
        ),
    )
    assert torch.equal(
        normalizer.bias,
        torch.zeros_like(
            normalizer.bias
        ),
    )
    assert normalizer.parameter_count == (
        2 * HIDDEN_DIM
    )
    assert normalizer.buffer_count == 0
    normalizer.assert_parameter_contract()


def test_normalizer_non_affine_and_disabled_parameter_contracts() -> None:
    non_affine = LayerNormalizer(
        HIDDEN_DIM,
        normalization_mode=(
            LAYER_NORMALIZATION_LAYER_NORM
        ),
        elementwise_affine=False,
        bias_enabled=False,
    )
    disabled = LayerNormalizer.from_flag(
        HIDDEN_DIM,
        layer_norm_enabled=False,
    )

    assert non_affine.weight is None
    assert non_affine.bias is None
    assert non_affine.parameter_count == 0
    assert len(
        non_affine.state_dict()
    ) == 0
    non_affine.assert_parameter_contract()

    assert disabled.normalization_enabled is False
    assert disabled.weight is None
    assert disabled.bias is None
    assert disabled.parameter_count == 0
    assert len(
        disabled.state_dict()
    ) == 0
    disabled.assert_parameter_contract()


def test_normalizer_parameter_fingerprint_tracks_values() -> None:
    normalizer = LayerNormalizer(
        HIDDEN_DIM
    )
    before = (
        normalizer.parameter_fingerprint()
    )

    with torch.no_grad():
        assert normalizer.weight is not None
        normalizer.weight[0] = 2.0

    after = (
        normalizer.parameter_fingerprint()
    )

    assert before is not None
    assert after is not None
    assert before != after


def test_normalizer_module_output_preserves_provenance() -> None:
    run, inputs = _message_builder_run()
    aggregation = _aggregation(
        run
    )
    layer_inputs = _layer_inputs(
        inputs
    )
    residual = _residual_output(
        aggregation,
        layer_inputs,
    )
    normalizer = LayerNormalizer(
        HIDDEN_DIM,
        elementwise_affine=True,
        bias_enabled=True,
    )

    output = normalizer(
        residual
    )

    assert output.residual_update is residual
    assert (
        output
        .normalizer_architecture_fingerprint
        == normalizer
        .architecture_fingerprint()
    )
    assert (
        output
        .normalizer_parameter_fingerprint
        == normalizer
        .parameter_fingerprint()
    )

    validate_layer_normalization_output(
        output=output,
        residual_update=residual,
        normalization_mode=(
            normalizer
            .normalization_mode
        ),
        normalization_position=(
            normalizer
            .normalization_position
        ),
        epsilon=normalizer.epsilon,
        weight=normalizer.weight,
        bias=normalizer.bias,
        normalizer_architecture_fingerprint=(
            normalizer
            .architecture_fingerprint()
        ),
        normalizer_parameter_fingerprint=(
            normalizer
            .parameter_fingerprint()
        ),
    )


def test_normalizer_builders() -> None:
    direct = build_layer_normalizer(
        HIDDEN_DIM
    )
    alias = build_normalizer(
        HIDDEN_DIM
    )
    flagged = (
        build_layer_normalizer_from_flag(
            HIDDEN_DIM,
            layer_norm_enabled=False,
        )
    )
    flagged_alias = (
        build_normalizer_from_flag(
            HIDDEN_DIM,
            layer_norm_enabled=False,
        )
    )

    assert isinstance(
        direct,
        LayerNormalizer,
    )
    assert isinstance(
        alias,
        LayerNormalizer,
    )
    assert flagged.normalization_enabled is False
    assert (
        flagged_alias.normalization_enabled
        is False
    )


def test_normalizer_train_eval_parity() -> None:
    run, inputs = _message_builder_run()
    aggregation = _aggregation(
        run
    )
    layer_inputs = _layer_inputs(
        inputs
    )
    residual = _residual_output(
        aggregation,
        layer_inputs,
    )
    normalizer = LayerNormalizer(
        HIDDEN_DIM
    )

    normalizer.train()
    train_output = normalizer(
        residual
    )
    normalizer.eval()
    eval_output = normalizer(
        residual
    )

    torch.testing.assert_close(
        train_output.output_state,
        eval_output.output_state,
    )


def test_normalization_functional_aliases() -> None:
    run, inputs = _message_builder_run()
    aggregation = _aggregation(
        run
    )
    layer_inputs = _layer_inputs(
        inputs
    )
    residual = _residual_output(
        aggregation,
        layer_inputs,
    )

    first = normalize_layer_state(
        residual_update=residual,
        normalization_mode=(
            LAYER_NORMALIZATION_NONE
        ),
    )
    second = resolve_layer_normalization(
        residual_update=residual,
        normalization_mode=(
            LAYER_NORMALIZATION_NONE
        ),
    )

    assert (
        first.output_state
        is residual.post_residual_state
    )
    assert (
        second.output_state
        is residual.post_residual_state
    )


def test_normalization_diagnostics_are_descriptive() -> None:
    internal, _run_value, _normalizer = (
        _internal_output(
            normalization_mode=(
                LAYER_NORMALIZATION_LAYER_NORM
            ),
            normalization_affine=False,
        )
    )
    summary = (
        layer_normalization_diagnostic_summary(
            internal.normalization
        )
    )

    assert summary["num_nodes"] == NODES
    assert summary["hidden_dim"] == HIDDEN_DIM
    assert summary[
        "normalization_enabled"
    ] is True
    assert summary[
        "aggregation_performed_here"
    ] is False
    assert summary[
        "dropout_performed_here"
    ] is False
    assert summary[
        "residual_performed_here"
    ] is False
    assert summary[
        "causal_importance_claim"
    ] is False


# =============================================================================
# Trace and internal output schemas
# =============================================================================


@pytest.mark.parametrize(
    "trace_mode",
    (
        LAYER_TRACE_NONE,
        LAYER_TRACE_NODE,
        LAYER_TRACE_FULL,
    ),
)
def test_internal_output_trace_modes(
    trace_mode: str,
) -> None:
    internal, _run_value, _normalizer = (
        _internal_output(
            trace_mode=trace_mode,
        )
    )

    if trace_mode == LAYER_TRACE_NONE:
        assert internal.trace is None
    else:
        assert isinstance(
            internal.trace,
            FunctionalMessagePassingLayerTrace,
        )
        assert (
            internal.trace.trace_mode
            == trace_mode
        )
        assert (
            internal.trace.layer_inputs
            is internal.layer_inputs
        )
        assert (
            internal.trace.aggregation
            is internal.aggregation
        )
        assert (
            internal.trace.residual_update
            is internal.residual_update
        )
        assert (
            internal.trace.normalization
            is internal.normalization
        )

    assert (
        internal.updated_node_state
        is internal
        .normalization
        .output_state
    )
    assert (
        internal.node_aggregate
        is internal
        .aggregation
        .node_aggregate
    )
    assert (
        internal.incoming_edge_count
        is internal
        .aggregation
        .incoming_edge_count
    )


def test_node_trace_retains_no_edge_objects() -> None:
    internal, _run_value, _normalizer = (
        _internal_output(
            trace_mode=(
                LAYER_TRACE_NODE
            )
        )
    )
    assert internal.trace is not None
    trace = internal.trace

    assert trace.relation_transform is None
    assert trace.edge_normalization is None
    assert trace.relation_gate is None
    assert trace.edge_attention is None
    assert trace.edge_messages is None
    assert trace.message_builder_run is None


def test_full_trace_preserves_complete_edge_lineage() -> None:
    internal, run, _normalizer = (
        _internal_output(
            trace_mode=(
                LAYER_TRACE_FULL
            )
        )
    )
    assert internal.trace is not None
    trace = internal.trace

    assert (
        trace.message_builder_run
        is run
    )
    assert trace.edge_messages is (
        run.public_output
    )
    assert (
        trace.relation_transform
        is run
        .composition_output
        .relation_transform
    )
    assert (
        trace.edge_normalization
        is run
        .resolved_coefficients
        .edge_normalization
    )
    assert trace.relation_gate is (
        run
        .resolved_coefficients
        .relation_gate
    )
    assert trace.edge_attention is (
        run
        .resolved_coefficients
        .edge_attention
    )
    assert (
        trace.updated_node_state
        is internal.updated_node_state
    )
    assert (
        trace.lineage_fingerprint()
        == trace.lineage_fingerprint()
    )


def test_layer_stage_chain_validator_accepts_exact_chain() -> None:
    internal, _run_value, _normalizer = (
        _internal_output()
    )

    validate_layer_stage_chain(
        layer_inputs=(
            internal.layer_inputs
        ),
        aggregation=(
            internal.aggregation
        ),
        residual_update=(
            internal.residual_update
        ),
        normalization=(
            internal.normalization
        ),
        computation_output=internal,
    )


def test_regularization_terms_are_immutable_scalar_mapping() -> None:
    internal, _run_value, _normalizer = (
        _internal_output()
    )

    assert dict(
        internal.regularization_terms
    ) == {
        "layer_regularizer": (
            internal
            .regularization_terms[
                "layer_regularizer"
            ]
        )
    }

    with pytest.raises(
        TypeError,
    ):
        internal.regularization_terms[
            "new"
        ] = torch.tensor(1.0)  # type: ignore[index]


def test_public_intermediates_from_full_trace_preserve_identity() -> None:
    internal, run, _normalizer = (
        _internal_output(
            trace_mode=(
                LAYER_TRACE_FULL
            )
        )
    )
    assert internal.trace is not None

    public_trace = (
        build_public_layer_intermediates(
            internal.trace
        )
    )

    assert isinstance(
        public_trace,
        FunctionalMessagePassingIntermediates,
    )
    assert (
        public_trace.relation_transform
        is run
        .composition_output
        .relation_transform
    )
    assert public_trace.edge_messages is (
        run.public_output
    )
    assert public_trace.aggregation is (
        internal.aggregation
    )
    assert (
        public_trace.pre_residual_state
        is internal
        .residual_update
        .pre_residual_state
    )
    assert (
        public_trace.post_residual_state
        is internal
        .residual_update
        .post_residual_state
    )


@pytest.mark.parametrize(
    "trace_mode",
    (
        LAYER_TRACE_NONE,
        LAYER_TRACE_NODE,
        LAYER_TRACE_FULL,
    ),
)
def test_public_layer_output_compatibility(
    trace_mode: str,
) -> None:
    internal, _run_value, _normalizer = (
        _internal_output(
            trace_mode=trace_mode,
        )
    )
    public = _public_output(
        internal
    )

    validate_public_layer_output(
        public_output=public,
        internal_output=internal,
    )

    assert (
        public.updated_node_state
        is internal.updated_node_state
    )
    assert public.node_aggregate is (
        internal.node_aggregate
    )
    assert (
        public.incoming_edge_count
        is internal.incoming_edge_count
    )
    assert public.source_inputs is (
        internal.source_inputs
    )
    assert (
        public.intermediates is not None
    ) is (
        trace_mode == LAYER_TRACE_FULL
    )


def test_layer_stages_named_tuple() -> None:
    internal, _run_value, _normalizer = (
        _internal_output()
    )
    stages = (
        FunctionalMessagePassingLayerStages(
            aggregation=(
                internal.aggregation
            ),
            residual_update=(
                internal.residual_update
            ),
            normalization=(
                internal.normalization
            ),
            computation_output=internal,
        )
    )

    assert stages.aggregation is (
        internal.aggregation
    )
    assert stages.computation_output is (
        internal
    )


# =============================================================================
# Generic diagnostic statistics
# =============================================================================


def test_scalar_tensor_statistics() -> None:
    value = torch.tensor(
        [-2.0, 0.0, 1e-10, 4.0],
    )
    summary = scalar_tensor_statistics(
        value,
        near_zero_absolute=1e-8,
    )

    assert summary["count"] == 4
    assert summary["minimum"] == -2.0
    assert summary["maximum"] == 4.0
    assert summary["zero_count"] == 1
    assert summary["near_zero_count"] == 2
    assert summary["positive_count"] == 2
    assert summary["negative_count"] == 1
    assert summary["finite"] is True


def test_matrix_statistics_include_per_node_norms() -> None:
    value = torch.tensor(
        [
            [3.0, 4.0],
            [0.0, 0.0],
        ]
    )
    summary = matrix_statistics(
        value,
        near_zero_absolute=1e-8,
    )

    assert summary["shape"] == [2, 2]
    assert summary["node_count"] == 2
    assert summary["hidden_dim"] == 2
    assert summary[
        "per_node_l2_norm"
    ]["maximum"] == 5.0
    assert summary[
        "per_node_l2_norm"
    ]["zero_count"] == 1


def test_state_transition_statistics_match_simple_case() -> None:
    source = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 2.0],
        ]
    )
    target = torch.tensor(
        [
            [2.0, 0.0],
            [0.0, 4.0],
        ]
    )

    summary = state_transition_statistics(
        source_state=source,
        target_state=target,
        near_zero_absolute=1e-8,
    )

    assert summary[
        "target_to_source_global_norm_ratio"
    ] == pytest.approx(2.0)
    assert summary[
        "delta_to_source_global_norm_ratio"
    ] == pytest.approx(1.0)


def test_incoming_edge_count_statistics() -> None:
    counts = torch.tensor(
        [0, 1, 2, 0, 4],
        dtype=torch.long,
    )
    summary = (
        incoming_edge_count_statistics(
            counts
        )
    )

    assert summary["node_count"] == 5
    assert summary[
        "total_incoming_edges"
    ] == 7
    assert summary[
        "isolated_node_count"
    ] == 2
    assert summary[
        "isolated_node_fraction"
    ] == pytest.approx(0.4)


# =============================================================================
# Complete layer diagnostics
# =============================================================================


def test_layer_diagnostics_public_identity() -> None:
    assert LAYER_DIAGNOSTICS_SCHEMA_VERSION.strip()
    assert LAYER_DIAGNOSTICS_INTERPRETATION == (
        "descriptive_layer_state_transition_and_lineage_diagnostics_only"
    )
    assert LAYER_DIAGNOSTICS_PARAMETER_FREE is True
    assert LAYER_DIAGNOSTICS_BUFFER_FREE is True
    assert (
        LAYER_DIAGNOSTICS_IMPLICIT_FORWARD_EXECUTION
        is False
    )


def test_layer_diagnostic_threshold_validation() -> None:
    with pytest.raises(
        ValueError,
        match="strictly positive",
    ):
        LayerDiagnosticThresholds(
            near_zero_absolute=0.0
        )

    with pytest.raises(
        ValueError,
        match=r"\[0, 1\]",
    ):
        LayerDiagnosticThresholds(
            high_isolated_node_fraction=1.1
        )


def test_layer_diagnostics_architecture_is_deterministic() -> None:
    first = (
        layer_diagnostics_architecture_dict(
            include_per_graph=True,
            include_edge_report=False,
            thresholds=(
                DEFAULT_LAYER_DIAGNOSTIC_THRESHOLDS
            ),
            edge_thresholds=(
                DEFAULT_MESSAGE_BUILDER_DIAGNOSTIC_THRESHOLDS
            ),
        )
    )
    second = (
        layer_diagnostics_architecture_dict(
            include_per_graph=True,
            include_edge_report=False,
            thresholds=(
                DEFAULT_LAYER_DIAGNOSTIC_THRESHOLDS
            ),
            edge_thresholds=(
                DEFAULT_MESSAGE_BUILDER_DIAGNOSTIC_THRESHOLDS
            ),
        )
    )

    assert first == second
    assert first[
        "changes_numerical_layer_architecture"
    ] is False
    assert first[
        "implicit_forward_execution"
    ] is False
    assert (
        layer_diagnostics_architecture_fingerprint(
            include_per_graph=True,
            include_edge_report=False,
            thresholds=(
                DEFAULT_LAYER_DIAGNOSTIC_THRESHOLDS
            ),
            edge_thresholds=(
                DEFAULT_MESSAGE_BUILDER_DIAGNOSTIC_THRESHOLDS
            ),
        )
        == layer_diagnostics_architecture_fingerprint(
            include_per_graph=True,
            include_edge_report=False,
            thresholds=(
                DEFAULT_LAYER_DIAGNOSTIC_THRESHOLDS
            ),
            edge_thresholds=(
                DEFAULT_MESSAGE_BUILDER_DIAGNOSTIC_THRESHOLDS
            ),
        )
    )


@pytest.mark.parametrize(
    "trace_mode",
    (
        LAYER_TRACE_NONE,
        LAYER_TRACE_NODE,
        LAYER_TRACE_FULL,
    ),
)
def test_trace_diagnostic_summary_respects_policy(
    trace_mode: str,
) -> None:
    internal, _run_value, _normalizer = (
        _internal_output(
            trace_mode=trace_mode,
        )
    )
    summary = (
        layer_trace_diagnostic_summary(
            internal,
            include_edge_report=False,
            edge_thresholds=(
                DEFAULT_MESSAGE_BUILDER_DIAGNOSTIC_THRESHOLDS
            ),
        )
    )

    assert summary[
        "trace_mode"
    ] == trace_mode
    assert summary[
        "trace_retained"
    ] is (
        trace_mode != LAYER_TRACE_NONE
    )
    assert summary[
        "edge_stages_retained"
    ] is (
        trace_mode == LAYER_TRACE_FULL
    )
    assert summary[
        "edge_report_available"
    ] is False


def test_full_trace_can_embed_existing_edge_report() -> None:
    internal, _run_value, _normalizer = (
        _internal_output(
            trace_mode=(
                LAYER_TRACE_FULL
            )
        )
    )
    summary = (
        layer_trace_diagnostic_summary(
            internal,
            include_edge_report=True,
            edge_thresholds=(
                DEFAULT_MESSAGE_BUILDER_DIAGNOSTIC_THRESHOLDS
            ),
        )
    )

    assert summary[
        "edge_report_available"
    ] is True
    assert isinstance(
        summary["edge_report"],
        dict,
    )
    assert summary[
        "edge_report"
    ][
        "public_output"
    ][
        "exact_edge_messages_tensor_preserved"
    ] is True


def test_graph_batch_diagnostics_cover_all_graphs() -> None:
    internal, _run_value, _normalizer = (
        _internal_output()
    )
    reports = graph_batch_diagnostics(
        internal
    )

    assert len(reports) == GRAPHS
    assert [
        report["graph_index"]
        for report in reports
    ] == list(range(GRAPHS))
    assert sum(
        report["node_count"]
        for report in reports
    ) == NODES
    assert sum(
        report["edge_count"]
        for report in reports
    ) == EDGES


def test_regularization_diagnostics_convert_scalars() -> None:
    internal, _run_value, _normalizer = (
        _internal_output()
    )
    summary = (
        regularization_diagnostic_summary(
            internal
        )
    )

    assert summary["term_count"] == 1
    assert summary["terms"] == {
        "layer_regularizer": pytest.approx(
            0.125
        )
    }
    assert summary["sum"] == pytest.approx(
        0.125
    )


def test_lineage_summary_preserves_internal_and_public_identity() -> None:
    internal, _run_value, _normalizer = (
        _internal_output(
            trace_mode=(
                LAYER_TRACE_FULL
            )
        )
    )
    public = _public_output(
        internal
    )
    lineage = layer_lineage_summary(
        internal,
        public_output=public,
    )

    identity = lineage[
        "exact_object_identity"
    ]
    assert identity[
        "aggregation_source_inputs_preserved"
    ] is True
    assert identity[
        "residual_layer_inputs_preserved"
    ] is True
    assert identity[
        "residual_aggregation_preserved"
    ] is True
    assert identity[
        "normalization_residual_preserved"
    ] is True
    assert identity[
        "updated_state_preserved"
    ] is True
    assert lineage[
        "public_output"
    ][
        "updated_state_preserved"
    ] is True


def test_complete_layer_report_is_tensor_free_json_safe_and_fingerprinted() -> None:
    internal, _run_value, _normalizer = (
        _internal_output(
            trace_mode=(
                LAYER_TRACE_FULL
            ),
            normalization_mode=(
                LAYER_NORMALIZATION_LAYER_NORM
            ),
            normalization_affine=False,
        )
    )
    public = _public_output(
        internal
    )

    report = (
        build_public_layer_diagnostic_report(
            public_output=public,
            internal_output=internal,
            include_per_graph=True,
            include_edge_report=True,
        )
    )

    assert report["schema_version"] == (
        LAYER_DIAGNOSTICS_SCHEMA_VERSION
    )
    assert (
        LAYER_DIAGNOSTIC_SECTION_STAGE_SUMMARIES
        in report
    )
    assert (
        LAYER_DIAGNOSTIC_SECTION_GLOBAL
        in report
    )
    assert (
        LAYER_DIAGNOSTIC_SECTION_BY_GRAPH
        in report
    )
    assert (
        LAYER_DIAGNOSTIC_SECTION_TRACE
        in report
    )
    assert (
        LAYER_DIAGNOSTIC_SECTION_REGULARIZATION
        in report
    )
    assert (
        LAYER_DIAGNOSTIC_SECTION_LINEAGE
        in report
    )
    assert (
        LAYER_DIAGNOSTIC_SECTION_ALERTS
        in report
    )
    assert "public_output" in report

    _assert_tensor_free(
        report
    )
    json.dumps(
        report,
        sort_keys=True,
        allow_nan=False,
    )

    assert (
        report["report_fingerprint"]
        == layer_diagnostic_report_fingerprint(
            report
        )
    )
    validate_layer_diagnostic_report(
        report,
        expected_num_graphs=GRAPHS,
    )


def test_internal_report_can_disable_graph_and_edge_slices() -> None:
    internal, _run_value, _normalizer = (
        _internal_output(
            trace_mode=(
                LAYER_TRACE_FULL
            )
        )
    )
    report = build_layer_diagnostic_report(
        internal,
        include_per_graph=False,
        include_edge_report=False,
    )

    assert report[
        LAYER_DIAGNOSTIC_SECTION_BY_GRAPH
    ] == []
    assert report[
        LAYER_DIAGNOSTIC_SECTION_TRACE
    ][
        "edge_report_available"
    ] is False


def test_layer_diagnostic_report_tampering_is_detected() -> None:
    internal, _run_value, _normalizer = (
        _internal_output()
    )
    report = build_layer_diagnostic_report(
        internal
    )
    tampered = copy.deepcopy(
        report
    )
    tampered[
        LAYER_DIAGNOSTIC_SECTION_GLOBAL
    ]["num_edges"] += 1

    with pytest.raises(
        ValueError,
        match="fingerprint",
    ):
        validate_layer_diagnostic_report(
            tampered
        )


def test_layer_diagnostic_scientific_claims_are_all_false() -> None:
    internal, _run_value, _normalizer = (
        _internal_output()
    )
    report = build_layer_diagnostic_report(
        internal
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


def test_empty_edges_generate_expected_layer_alerts() -> None:
    internal, _run_value, _normalizer = (
        _internal_output(
            empty_edges=True,
            residual_mode=(
                LAYER_RESIDUAL_DISABLED
            ),
            normalization_mode=(
                LAYER_NORMALIZATION_NONE
            ),
        )
    )
    alerts = derive_layer_alerts(
        internal
    )
    codes = {
        alert["code"]
        for alert in alerts
    }

    assert "empty_edge_set" in codes
    assert "isolated_nodes_present" in codes
    assert (
        "high_near_zero_aggregate_fraction"
        in codes
    )
    assert (
        "high_near_zero_output_fraction"
        in codes
    )


def test_layer_diagnostics_module_is_parameter_and_buffer_free() -> None:
    diagnostics = LayerDiagnostics(
        include_per_graph=True,
        include_edge_report=False,
    )

    _assert_parameter_and_buffer_free(
        diagnostics
    )
    assert diagnostics.parameter_count == 0
    assert (
        diagnostics
        .trainable_parameter_count
        == 0
    )
    assert diagnostics.buffer_count == 0
    assert diagnostics.parameter_fingerprint is None
    diagnostics.assert_parameter_free()


def test_layer_diagnostics_builder_and_module_reports() -> None:
    diagnostics = build_layer_diagnostics(
        include_per_graph=True,
        include_edge_report=False,
    )
    internal, _run_value, _normalizer = (
        _internal_output(
            trace_mode=(
                LAYER_TRACE_NODE
            )
        )
    )
    public = _public_output(
        internal
    )

    internal_report = diagnostics(
        internal
    )
    public_report = (
        diagnostics.public_report(
            public_output=public,
            internal_output=internal,
        )
    )

    assert (
        internal_report[
            "diagnostics_architecture_fingerprint"
        ]
        == diagnostics
        .architecture_fingerprint()
    )
    assert (
        public_report[
            "diagnostics_architecture_fingerprint"
        ]
        == diagnostics
        .architecture_fingerprint()
    )
    assert "public_output" in (
        public_report
    )


# =============================================================================
# End-to-end component-chain properties
# =============================================================================


@pytest.mark.parametrize(
    (
        "residual_mode",
        "normalization_mode",
    ),
    (
        (
            LAYER_RESIDUAL_DISABLED,
            LAYER_NORMALIZATION_NONE,
        ),
        (
            LAYER_RESIDUAL_ADDITIVE,
            LAYER_NORMALIZATION_NONE,
        ),
        (
            LAYER_RESIDUAL_DISABLED,
            LAYER_NORMALIZATION_LAYER_NORM,
        ),
        (
            LAYER_RESIDUAL_ADDITIVE,
            LAYER_NORMALIZATION_LAYER_NORM,
        ),
    ),
)
def test_component_chain_mode_combinations(
    residual_mode: str,
    normalization_mode: str,
) -> None:
    internal, _run_value, _normalizer = (
        _internal_output(
            residual_mode=residual_mode,
            normalization_mode=(
                normalization_mode
            ),
            normalization_affine=False,
        )
    )

    if residual_mode == (
        LAYER_RESIDUAL_DISABLED
    ):
        assert (
            internal
            .residual_update
            .post_residual_state
            is internal
            .residual_update
            .post_dropout_update
        )
    else:
        torch.testing.assert_close(
            internal
            .residual_update
            .post_residual_state,
            internal
            .layer_inputs
            .input_node_state
            + internal
            .residual_update
            .post_dropout_update,
        )

    if normalization_mode == (
        LAYER_NORMALIZATION_NONE
    ):
        assert (
            internal.updated_node_state
            is internal
            .residual_update
            .post_residual_state
        )
    else:
        assert (
            internal.updated_node_state
            is not internal
            .residual_update
            .post_residual_state
        )


def test_full_component_chain_preserves_autograd() -> None:
    source_state = torch.randn(
        NODES,
        HIDDEN_DIM,
        requires_grad=True,
    )
    transformed_state = torch.randn(
        EDGES,
        HIDDEN_DIM,
        requires_grad=True,
    )
    run, inputs = _message_builder_run(
        state=source_state,
        transformed_state=(
            transformed_state
        ),
    )
    aggregation = _aggregation(
        run
    )
    layer_inputs = _layer_inputs(
        inputs
    )
    residual = _residual_output(
        aggregation,
        layer_inputs,
        residual_mode=(
            LAYER_RESIDUAL_ADDITIVE
        ),
    )
    normalizer = LayerNormalizer(
        HIDDEN_DIM,
        elementwise_affine=True,
        bias_enabled=True,
    )
    normalized = normalizer(
        residual
    )

    normalized.output_state.square().sum().backward()

    assert source_state.grad is not None
    assert transformed_state.grad is not None
    assert normalizer.weight is not None
    assert normalizer.weight.grad is not None
    assert normalizer.bias is not None
    assert normalizer.bias.grad is not None
    assert bool(
        torch.isfinite(
            source_state.grad
        )
        .all()
        .item()
    )


def test_component_chain_supports_float64() -> None:
    internal, _run_value, normalizer = (
        _internal_output(
            dtype=torch.float64,
            normalization_mode=(
                LAYER_NORMALIZATION_LAYER_NORM
            ),
            normalization_affine=True,
        )
    )

    assert internal.dtype == torch.float64
    assert (
        internal.updated_node_state.dtype
        == torch.float64
    )
    assert normalizer is not None
    assert normalizer.weight is not None
    assert normalizer.weight.dtype == (
        torch.float64
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is not available.",
)
def test_component_chain_supports_cuda() -> None:
    internal, _run_value, normalizer = (
        _internal_output(
            device="cuda",
            normalization_mode=(
                LAYER_NORMALIZATION_LAYER_NORM
            ),
            normalization_affine=True,
        )
    )

    assert internal.device.type == "cuda"
    assert (
        internal.updated_node_state
        .device.type
        == "cuda"
    )
    assert normalizer is not None
    assert normalizer.weight is not None
    assert normalizer.weight.device.type == (
        "cuda"
    )
