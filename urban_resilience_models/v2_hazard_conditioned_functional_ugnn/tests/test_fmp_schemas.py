"""
Contract tests for functional-message-passing schemas.

Target repository path:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            tests/
                test_fmp_schemas.py

Implementation under test:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            functional_message_passing/
                schemas.py

This suite isolates schema semantics from trainable modules and graph-loading
implementations. Lightweight controlled upstream doubles are injected into the
schema module so failures identify the FMP boundary itself rather than a
different subsystem's constructor.

Covered contracts:

- schema-version identity;
- semantic relation-family alignment;
- compiled relation, control, node, edge, graph, and hazard-query alignment;
- exact target-node + dense-relation attention grouping;
- relation-transform, structural-normalization, gate, attention, message, and
  aggregation outputs;
- disabled gate and attention identity behavior;
- isolated-node and empty-edge behavior;
- optional retained intermediates;
- one-layer and stack outputs;
- deterministic semantic, lineage, and value fingerprints;
- immutable metadata and regularization mappings;
- dtype, device, shape, index-range, and finiteness validation.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import pytest
import torch

from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.constants import (
    AGGREGATION_MEAN,
    ATTENTION_HEAD_REDUCTION_MEAN,
    ATTENTION_NORMALIZATION_TARGET_NODE_RELATION,
    EDGE_NORMALIZATION_NONE,
    RELATION_GATE_ACTIVATION_SIGMOID,
    RELATION_GATE_SCOPE_TARGET_NODE,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.functional_message_passing import (
    schemas as fmp_schemas,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.functional_message_passing.schemas import (
    AGGREGATION_OUTPUT_SCHEMA_VERSION,
    EDGE_ATTENTION_OUTPUT_SCHEMA_VERSION,
    EDGE_MESSAGE_OUTPUT_SCHEMA_VERSION,
    EDGE_NORMALIZATION_OUTPUT_SCHEMA_VERSION,
    FMP_INTERMEDIATES_SCHEMA_VERSION,
    FMP_LAYER_OUTPUT_SCHEMA_VERSION,
    FMP_STACK_OUTPUT_SCHEMA_VERSION,
    FUNCTIONAL_MESSAGE_PASSING_INPUT_SCHEMA_VERSION,
    RELATION_FAMILY_ALIGNMENT_SCHEMA_VERSION,
    RELATION_GATE_OUTPUT_SCHEMA_VERSION,
    RELATION_TRANSFORM_OUTPUT_SCHEMA_VERSION,
    AggregationOutput,
    EdgeAttentionOutput,
    EdgeMessageOutput,
    FunctionalMessagePassingInputs,
    FunctionalMessagePassingIntermediates,
    FunctionalMessagePassingLayerOutput,
    FunctionalMessagePassingStackOutput,
    RelationFamilyAlignment,
    RelationGateOutput,
    RelationTransformOutput,
    StructuralEdgeNormalizationOutput,
)


NODES = 5
EDGES = 5
GRAPHS = 2
RELATIONS = 3
FAMILIES = 2
HIDDEN_DIM = 4
QUERY_DIM = 3
HEADS = 2


# =============================================================================
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
        names: tuple[str, ...] = (
            "spatial_adjacency",
            "temporal_lag",
            "random_placebo",
        ),
        stable_ids: tuple[int, ...] = (
            100,
            200,
            900,
        ),
        controls: tuple[bool, ...] = (
            False,
            False,
            True,
        ),
        fingerprint: str = "compiled-registry",
    ) -> None:
        self.relation_names = names
        self.stable_relation_ids = stable_ids
        self.entries = tuple(
            FakeCompiledEntry(
                name=name,
                relation_id=relation_id,
                specification=FakeSpecification(
                    is_control=is_control
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

    def assert_matches_source_registry(
        self,
        source_registry: object,
        *,
        require_operational_match: bool,
    ) -> None:
        assert require_operational_match is False
        if getattr(
            source_registry,
            "reject_compiled_match",
            False,
        ):
            raise ValueError(
                "registry mismatch"
            )


@dataclass
class FakeRegistryEntry:
    name: str
    relation_id: int


class FakeRelationRegistry:
    def __init__(self) -> None:
        self.reject_compiled_match = False
        self._entries = {
            "spatial": FakeRegistryEntry(
                "spatial",
                10,
            ),
            "spatial_adjacency": FakeRegistryEntry(
                "spatial_adjacency",
                100,
            ),
            "temporal": FakeRegistryEntry(
                "temporal",
                20,
            ),
            "temporal_lag": FakeRegistryEntry(
                "temporal_lag",
                200,
            ),
            "random_placebo": FakeRegistryEntry(
                "random_placebo",
                900,
            ),
        }

    def ancestors_of(
        self,
        name: str,
    ) -> tuple[FakeRegistryEntry, ...]:
        if name == "spatial_adjacency":
            return (
                self._entries["spatial"],
            )
        if name == "temporal_lag":
            return (
                self._entries["temporal"],
            )
        return ()

    def get_entry_by_name(
        self,
        name: str,
    ) -> FakeRegistryEntry:
        return self._entries[name]

    def semantic_fingerprint(self) -> str:
        return "source-registry"


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
        lineage_fingerprint: str = "fusion-lineage",
        encoder_architecture_fingerprint: str = (
            "fusion-architecture"
        ),
    ) -> None:
        self.fused_state = fused_state
        self.alignment = alignment
        self.lineage_fingerprint = (
            lineage_fingerprint
        )
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
        validation_error: Exception | None = None,
    ) -> None:
        self.external_node_ids = (
            external_node_ids
        )
        self.node_batch_index = (
            node_batch_index
        )
        self.edge_index = edge_index
        self.edge_relation_type = (
            edge_relation_type
        )
        self.edge_attributes = edge_attributes
        self.semantic_edge_weight = (
            semantic_edge_weight
        )
        self.edge_batch_index = (
            edge_batch_index
        )
        self.allow_cross_graph_edges = (
            allow_cross_graph_edges
        )
        self.validation_error = validation_error
        self.validated = False

    def validate(self) -> None:
        self.validated = True
        if self.validation_error is not None:
            raise self.validation_error

    @property
    def num_nodes(self) -> int:
        return len(self.external_node_ids)

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[1])

    @property
    def batch_size(self) -> int:
        if self.node_batch_index.numel() == 0:
            return 0
        return (
            int(
                self.node_batch_index
                .max()
                .item()
            )
            + 1
        )


class FakeHazardEmbeddingLookup:
    pass


class FakeNodeAlignedHazardEmbeddingLookup:
    def __init__(
        self,
        node_batch_index: torch.Tensor,
    ) -> None:
        self.node_batch_index = (
            node_batch_index
        )


class FakeHazardQueryEncoding:
    def __init__(
        self,
        *,
        query: torch.Tensor,
        source_embedding: object,
        lineage_fingerprint: str = (
            "hazard-lineage"
        ),
    ) -> None:
        self.query = query
        self.source_embedding = (
            source_embedding
        )
        self.lineage_fingerprint = (
            lineage_fingerprint
        )

    @property
    def item_count(self) -> int:
        return int(self.query.shape[0])


class FakeCompiledHazardRelationPriors:
    def __init__(
        self,
        *,
        relation_names: tuple[str, ...],
        stable_relation_ids: tuple[int, ...],
        source_compiled_relation_fingerprint: str,
        fingerprint: str = "prior-fingerprint",
    ) -> None:
        self.relation_names = relation_names
        self.stable_relation_ids = (
            stable_relation_ids
        )
        self.source_compiled_relation_fingerprint = (
            source_compiled_relation_fingerprint
        )
        self._fingerprint = fingerprint

    def fingerprint(self) -> str:
        return self._fingerprint


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
    monkeypatch.setattr(
        fmp_schemas,
        "RelationRegistry",
        FakeRelationRegistry,
    )
    monkeypatch.setattr(
        fmp_schemas,
        "HazardEmbeddingLookup",
        FakeHazardEmbeddingLookup,
    )
    monkeypatch.setattr(
        fmp_schemas,
        "NodeAlignedHazardEmbeddingLookup",
        FakeNodeAlignedHazardEmbeddingLookup,
    )
    monkeypatch.setattr(
        fmp_schemas,
        "HazardQueryEncoding",
        FakeHazardQueryEncoding,
    )
    monkeypatch.setattr(
        fmp_schemas,
        "CompiledHazardRelationPriors",
        FakeCompiledHazardRelationPriors,
    )


# =============================================================================
# Helpers
# =============================================================================


def _node_ids(
    count: int = NODES,
) -> tuple[str, ...]:
    return tuple(
        f"node-{index}"
        for index in range(count)
    )


def _node_batch(
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
) -> torch.Tensor:
    # 0→1, 2→1, 1→2 in graph 0; 3→4, 4→3 in graph 1.
    return torch.tensor(
        [
            [0, 2, 1, 3, 4],
            [1, 1, 2, 4, 3],
        ],
        dtype=torch.long,
        device=device,
    )


def _edge_relation_index(
    *,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    return torch.tensor(
        [0, 0, 1, 2, 1],
        dtype=torch.long,
        device=device,
    )


def _edge_batch(
    *,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    return torch.tensor(
        [0, 0, 0, 1, 1],
        dtype=torch.long,
        device=device,
    )


def _graph(
    *,
    node_count: int = NODES,
    node_batch_index: torch.Tensor | None = None,
    edge_index: torch.Tensor | None = None,
    edge_relation_type: torch.Tensor | None = None,
    edge_attributes: torch.Tensor | None = None,
    semantic_edge_weight: torch.Tensor | None = None,
    edge_batch_index: torch.Tensor | None = None,
    allow_cross_graph_edges: bool = False,
) -> FakeUrbanGraphBatch:
    if node_count == NODES:
        resolved_batch = (
            _node_batch()
            if node_batch_index is None
            else node_batch_index
        )
        resolved_edge_index = (
            _edge_index()
            if edge_index is None
            else edge_index
        )
        resolved_relation = (
            _edge_relation_index()
            if edge_relation_type is None
            else edge_relation_type
        )
    else:
        resolved_batch = (
            torch.zeros(
                node_count,
                dtype=torch.long,
            )
            if node_batch_index is None
            else node_batch_index
        )
        resolved_edge_index = (
            torch.empty(
                2,
                0,
                dtype=torch.long,
            )
            if edge_index is None
            else edge_index
        )
        resolved_relation = (
            torch.empty(
                0,
                dtype=torch.long,
            )
            if edge_relation_type is None
            else edge_relation_type
        )

    return FakeUrbanGraphBatch(
        external_node_ids=_node_ids(
            node_count
        ),
        node_batch_index=resolved_batch,
        edge_index=resolved_edge_index,
        edge_relation_type=resolved_relation,
        edge_attributes=edge_attributes,
        semantic_edge_weight=(
            semantic_edge_weight
        ),
        edge_batch_index=edge_batch_index,
        allow_cross_graph_edges=(
            allow_cross_graph_edges
        ),
    )


def _node_state(
    *,
    node_count: int = NODES,
    hidden_dim: int = HIDDEN_DIM,
    node_batch_index: torch.Tensor | None = None,
    item_ids: tuple[str, ...] | None = None,
    graph_count: int | None = None,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
    offset: float = 0.0,
) -> FakeNodeStateFusionOutput:
    values = (
        torch.arange(
            node_count * hidden_dim,
            dtype=dtype,
            device=device,
        )
        .reshape(node_count, hidden_dim)
        / 10.0
        + offset
    )
    resolved_batch = (
        _node_batch(device=device)
        if (
            node_batch_index is None
            and node_count == NODES
        )
        else node_batch_index
    )
    resolved_graph_count = (
        GRAPHS
        if graph_count is None
        and node_count == NODES
        else graph_count
    )

    return FakeNodeStateFusionOutput(
        fused_state=values,
        alignment=FakeAlignment(
            item_ids=(
                _node_ids(node_count)
                if item_ids is None
                else item_ids
            ),
            node_batch_index=resolved_batch,
            graph_count=resolved_graph_count,
        ),
    )


def _registry(
    *,
    fingerprint: str = "compiled-registry",
) -> FakeCompiledRelationRegistry:
    return FakeCompiledRelationRegistry(
        fingerprint=fingerprint,
    )


def _families(
    *,
    registry_fingerprint: str = (
        "compiled-registry"
    ),
    mapping: torch.Tensor | None = None,
    relation_names: tuple[str, ...] | None = None,
    stable_relation_ids: tuple[int, ...] | None = None,
) -> RelationFamilyAlignment:
    return RelationFamilyAlignment(
        family_names=(
            "spatial",
            "temporal_control",
        ),
        stable_family_ids=(10, 20),
        relation_family_index_by_relation=(
            torch.tensor(
                [0, 1, 1],
                dtype=torch.long,
            )
            if mapping is None
            else mapping
        ),
        relation_names=(
            (
                "spatial_adjacency",
                "temporal_lag",
                "random_placebo",
            )
            if relation_names is None
            else relation_names
        ),
        stable_relation_ids=(
            (100, 200, 900)
            if stable_relation_ids is None
            else stable_relation_ids
        ),
        source_relation_registry_fingerprint=(
            "source-registry"
        ),
        compiled_relation_registry_fingerprint=(
            registry_fingerprint
        ),
    )


def _inputs(
    *,
    graph: FakeUrbanGraphBatch | None = None,
    node_state: FakeNodeStateFusionOutput | None = None,
    registry: FakeCompiledRelationRegistry | None = None,
    relation_families: RelationFamilyAlignment | None = None,
    hazard_query: FakeHazardQueryEncoding | None = None,
    priors: FakeCompiledHazardRelationPriors | None = None,
    source_fingerprint: str | None = (
        "fmp-source"
    ),
) -> FunctionalMessagePassingInputs:
    return FunctionalMessagePassingInputs(
        source_graph=(
            _graph()
            if graph is None
            else graph
        ),
        node_state=(
            _node_state()
            if node_state is None
            else node_state
        ),
        compiled_relation_registry=(
            _registry()
            if registry is None
            else registry
        ),
        relation_families=(
            relation_families
        ),
        hazard_query=hazard_query,
        compiled_relation_priors=priors,
        source_fingerprint=source_fingerprint,
    )


def _transform(
    inputs: FunctionalMessagePassingInputs,
    *,
    values: torch.Tensor | None = None,
) -> RelationTransformOutput:
    transformed = (
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
        / 10.0
        if values is None
        else values
    )
    return RelationTransformOutput(
        transformed_source_state=transformed,
        source_inputs=inputs,
        transform_mode="shared",
        encoder_architecture_fingerprint=(
            "transform-architecture"
        ),
        parameter_fingerprint=(
            "transform-parameters"
        ),
        relation_parameter_fingerprints={
            inputs.relation_names[0]: (
                "relation-parameters"
            )
        },
    )


def _normalization(
    inputs: FunctionalMessagePassingInputs,
    *,
    coefficients: torch.Tensor | None = None,
) -> StructuralEdgeNormalizationOutput:
    return StructuralEdgeNormalizationOutput(
        coefficients=(
            torch.ones(
                inputs.num_edges,
                dtype=inputs.dtype,
                device=inputs.device,
            )
            if coefficients is None
            else coefficients
        ),
        source_inputs=inputs,
        normalization_mode=(
            EDGE_NORMALIZATION_NONE
        ),
        encoder_architecture_fingerprint=(
            "normalization-architecture"
        ),
    )


def _gate(
    inputs: FunctionalMessagePassingInputs,
) -> RelationGateOutput:
    logits = torch.linspace(
        -1.0,
        1.0,
        steps=(
            inputs.num_nodes
            * inputs.num_relations
        ),
        dtype=inputs.dtype,
        device=inputs.device,
    ).reshape(
        inputs.num_nodes,
        inputs.num_relations,
    )
    values = torch.sigmoid(logits)
    edge_values = values[
        inputs.target_index,
        inputs.edge_relation_index,
    ]

    return RelationGateOutput(
        gate_logits=logits,
        gate_values=values,
        edge_gate_values=edge_values,
        source_inputs=inputs,
        scope=RELATION_GATE_SCOPE_TARGET_NODE,
        activation=(
            RELATION_GATE_ACTIVATION_SIGMOID
        ),
        encoder_architecture_fingerprint=(
            "gate-architecture"
        ),
        parameter_fingerprint="gate-parameters",
        regularization_terms={
            "prior_penalty": torch.tensor(
                0.25,
                dtype=inputs.dtype,
                device=inputs.device,
            )
        },
    )


def _uniform_attention_weights(
    inputs: FunctionalMessagePassingInputs,
    *,
    heads: int = HEADS,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    group_ids = inputs.attention_group_id
    counts = torch.bincount(
        group_ids,
        minlength=inputs.attention_num_groups,
    )

    denominators = counts[
        group_ids
    ].to(dtype=inputs.dtype)
    one_head = torch.ones(
        inputs.num_edges,
        dtype=inputs.dtype,
        device=inputs.device,
    ) / denominators
    by_head = one_head.unsqueeze(-1).repeat(
        1,
        heads,
    )
    return (
        by_head,
        group_ids,
        counts,
    )


def _attention(
    inputs: FunctionalMessagePassingInputs,
    *,
    heads: int = HEADS,
) -> EdgeAttentionOutput:
    normalized, group_ids, counts = (
        _uniform_attention_weights(
            inputs,
            heads=heads,
        )
    )
    scores = torch.zeros_like(normalized)

    return EdgeAttentionOutput(
        raw_scores_by_head=scores,
        normalized_weights_by_head=normalized,
        edge_weights=normalized.mean(
            dim=1
        ),
        group_ids=group_ids,
        group_counts=counts,
        source_inputs=inputs,
        attention_mode="uniform",
        normalization_mode=(
            ATTENTION_NORMALIZATION_TARGET_NODE_RELATION
        ),
        head_reduction=(
            ATTENTION_HEAD_REDUCTION_MEAN
        ),
        encoder_architecture_fingerprint=(
            "attention-architecture"
        ),
        parameter_fingerprint=(
            "attention-parameters"
        ),
    )


def _messages(
    inputs: FunctionalMessagePassingInputs,
    *,
    with_gate: bool = True,
    with_attention: bool = True,
    semantic: torch.Tensor | None = None,
) -> EdgeMessageOutput:
    transform = _transform(inputs)
    normalization = _normalization(inputs)
    gate = _gate(inputs) if with_gate else None
    attention = (
        _attention(inputs)
        if with_attention
        else None
    )

    expected = (
        transform.transformed_source_state
        * normalization.coefficients.unsqueeze(
            -1
        )
    )
    if gate is not None:
        expected = (
            expected
            * gate.edge_gate_values.unsqueeze(
                -1
            )
        )
    if attention is not None:
        expected = (
            expected
            * attention.edge_weights.unsqueeze(
                -1
            )
        )
    if semantic is not None:
        expected = (
            expected
            * semantic.unsqueeze(-1)
        )

    return EdgeMessageOutput(
        edge_messages=expected,
        relation_transform=transform,
        edge_normalization=normalization,
        relation_gate=gate,
        edge_attention=attention,
        semantic_edge_weight=semantic,
        encoder_architecture_fingerprint=(
            "message-architecture"
        ),
    )


def _aggregation(
    messages: EdgeMessageOutput,
) -> AggregationOutput:
    inputs = messages.source_inputs
    counts = torch.bincount(
        inputs.target_index,
        minlength=inputs.num_nodes,
    )
    sums = torch.zeros(
        (
            inputs.num_nodes,
            inputs.hidden_dim,
        ),
        dtype=inputs.dtype,
        device=inputs.device,
    )
    sums.index_add_(
        0,
        inputs.target_index,
        messages.edge_messages,
    )
    mean = sums / (
        counts
        .clamp_min(1)
        .to(dtype=inputs.dtype)
        .unsqueeze(-1)
    )

    return AggregationOutput(
        node_aggregate=mean,
        incoming_edge_count=counts,
        source_messages=messages,
        aggregation_mode=AGGREGATION_MEAN,
        encoder_architecture_fingerprint=(
            "aggregation-architecture"
        ),
    )


def _intermediates(
    inputs: FunctionalMessagePassingInputs,
) -> FunctionalMessagePassingIntermediates:
    transform = _transform(inputs)
    normalization = _normalization(inputs)
    gate = _gate(inputs)
    attention = _attention(inputs)

    expected_messages = (
        transform.transformed_source_state
        * normalization.coefficients.unsqueeze(
            -1
        )
        * gate.edge_gate_values.unsqueeze(-1)
        * attention.edge_weights.unsqueeze(
            -1
        )
    )
    messages = EdgeMessageOutput(
        edge_messages=expected_messages,
        relation_transform=transform,
        edge_normalization=normalization,
        relation_gate=gate,
        edge_attention=attention,
        semantic_edge_weight=None,
        encoder_architecture_fingerprint=(
            "message-architecture"
        ),
    )
    aggregation = _aggregation(messages)

    return FunctionalMessagePassingIntermediates(
        relation_transform=transform,
        edge_normalization=normalization,
        relation_gate=gate,
        edge_attention=attention,
        edge_messages=messages,
        aggregation=aggregation,
        pre_residual_state=(
            aggregation.node_aggregate
        ),
        post_residual_state=(
            aggregation.node_aggregate
            + inputs.node_state.fused_state
        ),
    )


def _layer_output(
    inputs: FunctionalMessagePassingInputs,
    *,
    layer_index: int = 0,
    retain_intermediates: bool = True,
    offset: float = 0.0,
) -> FunctionalMessagePassingLayerOutput:
    intermediates = (
        _intermediates(inputs)
        if retain_intermediates
        else None
    )

    if intermediates is not None:
        aggregate = (
            intermediates
            .aggregation
            .node_aggregate
        )
        counts = (
            intermediates
            .aggregation
            .incoming_edge_count
        )
    else:
        messages = _messages(inputs)
        aggregation = _aggregation(messages)
        aggregate = aggregation.node_aggregate
        counts = aggregation.incoming_edge_count

    updated = (
        inputs.node_state.fused_state
        + aggregate
        + offset
    )

    return FunctionalMessagePassingLayerOutput(
        updated_node_state=updated,
        node_aggregate=aggregate,
        incoming_edge_count=counts,
        source_inputs=inputs,
        layer_index=layer_index,
        residual_enabled=True,
        layer_norm_enabled=False,
        encoder_architecture_fingerprint=(
            f"layer-architecture-{layer_index}"
        ),
        lineage_fingerprint=(
            f"layer-lineage-{layer_index}"
        ),
        intermediates=intermediates,
        regularization_terms={
            "gate_penalty": torch.tensor(
                0.1,
                dtype=inputs.dtype,
                device=inputs.device,
            )
        },
    )


# =============================================================================
# Published schema identity
# =============================================================================


def test_all_schema_versions_are_nonempty() -> None:
    versions = (
        RELATION_FAMILY_ALIGNMENT_SCHEMA_VERSION,
        FUNCTIONAL_MESSAGE_PASSING_INPUT_SCHEMA_VERSION,
        RELATION_TRANSFORM_OUTPUT_SCHEMA_VERSION,
        EDGE_NORMALIZATION_OUTPUT_SCHEMA_VERSION,
        RELATION_GATE_OUTPUT_SCHEMA_VERSION,
        EDGE_ATTENTION_OUTPUT_SCHEMA_VERSION,
        EDGE_MESSAGE_OUTPUT_SCHEMA_VERSION,
        AGGREGATION_OUTPUT_SCHEMA_VERSION,
        FMP_INTERMEDIATES_SCHEMA_VERSION,
        FMP_LAYER_OUTPUT_SCHEMA_VERSION,
        FMP_STACK_OUTPUT_SCHEMA_VERSION,
    )

    assert len(set(versions)) >= 1
    for version in versions:
        assert isinstance(version, str)
        assert version.strip()


# =============================================================================
# RelationFamilyAlignment
# =============================================================================


def test_relation_family_alignment_valid_contract() -> None:
    alignment = _families()

    assert alignment.num_relations == RELATIONS
    assert alignment.num_families == FAMILIES
    assert alignment.device == torch.device(
        "cpu"
    )
    assert alignment.family_names == (
        "spatial",
        "temporal_control",
    )
    assert alignment.stable_family_ids == (
        10,
        20,
    )


def test_relation_family_fingerprints_are_deterministic() -> None:
    first = _families()
    second = _families()

    assert first.semantic_dict() == (
        second.semantic_dict()
    )
    assert first.semantic_fingerprint() == (
        second.semantic_fingerprint()
    )
    assert first.value_fingerprint() == (
        second.value_fingerprint()
    )
    assert first.fingerprint() == (
        second.fingerprint()
    )


def test_relation_family_value_fingerprint_changes_with_mapping() -> None:
    first = _families()
    second = _families(
        mapping=torch.tensor(
            [1, 0, 1],
            dtype=torch.long,
        )
    )

    assert first.semantic_fingerprint() == (
        second.semantic_fingerprint()
    )
    assert first.value_fingerprint() != (
        second.value_fingerprint()
    )


def test_relation_family_from_registries_compiles_root_order() -> None:
    source = FakeRelationRegistry()
    compiled = _registry()

    alignment = (
        RelationFamilyAlignment
        .from_registries(
            source_registry=source,
            compiled_registry=compiled,
        )
    )

    assert alignment.family_names == (
        "spatial",
        "temporal",
        "random_placebo",
    )
    assert alignment.stable_family_ids == (
        10,
        20,
        900,
    )
    assert torch.equal(
        alignment.relation_family_index_by_relation,
        torch.tensor(
            [0, 1, 2],
            dtype=torch.long,
        ),
    )
    assert (
        alignment
        .compiled_relation_registry_fingerprint
        == compiled.fingerprint()
    )


def test_relation_family_from_registries_rejects_wrong_types() -> None:
    with pytest.raises(
        TypeError,
        match="RelationRegistry",
    ):
        RelationFamilyAlignment.from_registries(
            source_registry=object(),  # type: ignore[arg-type]
            compiled_registry=_registry(),
        )

    with pytest.raises(
        TypeError,
        match="CompiledRelationRegistry",
    ):
        RelationFamilyAlignment.from_registries(
            source_registry=FakeRelationRegistry(),
            compiled_registry=object(),  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        (
            "family_names",
            ("spatial", "spatial"),
        ),
        (
            "relation_names",
            ("r", "r", "q"),
        ),
        (
            "stable_family_ids",
            (10, 10),
        ),
        (
            "stable_relation_ids",
            (1, 1, 2),
        ),
    ),
)
def test_relation_family_rejects_duplicate_identity(
    field: str,
    value: Any,
) -> None:
    kwargs: dict[str, Any] = {
        "family_names": (
            "spatial",
            "temporal_control",
        ),
        "stable_family_ids": (
            10,
            20,
        ),
        "relation_family_index_by_relation": (
            torch.tensor(
                [0, 1, 1],
                dtype=torch.long,
            )
        ),
        "relation_names": (
            "r0",
            "r1",
            "r2",
        ),
        "stable_relation_ids": (
            100,
            200,
            900,
        ),
        "source_relation_registry_fingerprint": (
            "source"
        ),
        "compiled_relation_registry_fingerprint": (
            "compiled"
        ),
    }
    kwargs[field] = value

    with pytest.raises(
        ValueError,
        match="duplicates",
    ):
        RelationFamilyAlignment(**kwargs)


def test_relation_family_requires_at_least_one_family() -> None:
    with pytest.raises(
        ValueError,
        match="At least one",
    ):
        RelationFamilyAlignment(
            family_names=(),
            stable_family_ids=(),
            relation_family_index_by_relation=(
                torch.empty(
                    0,
                    dtype=torch.long,
                )
            ),
            relation_names=(),
            stable_relation_ids=(),
            source_relation_registry_fingerprint=(
                "source"
            ),
            compiled_relation_registry_fingerprint=(
                "compiled"
            ),
        )


def test_relation_family_rejects_length_mismatches() -> None:
    with pytest.raises(
        ValueError,
        match="family_names",
    ):
        RelationFamilyAlignment(
            family_names=("a", "b"),
            stable_family_ids=(1,),
            relation_family_index_by_relation=(
                torch.tensor(
                    [0],
                    dtype=torch.long,
                )
            ),
            relation_names=("r",),
            stable_relation_ids=(5,),
            source_relation_registry_fingerprint=(
                "source"
            ),
            compiled_relation_registry_fingerprint=(
                "compiled"
            ),
        )

    with pytest.raises(
        ValueError,
        match="relation_names",
    ):
        RelationFamilyAlignment(
            family_names=("a",),
            stable_family_ids=(1,),
            relation_family_index_by_relation=(
                torch.tensor(
                    [0],
                    dtype=torch.long,
                )
            ),
            relation_names=("r",),
            stable_relation_ids=(5, 6),
            source_relation_registry_fingerprint=(
                "source"
            ),
            compiled_relation_registry_fingerprint=(
                "compiled"
            ),
        )


@pytest.mark.parametrize(
    "mapping",
    (
        torch.tensor(
            [0, 1, 1],
            dtype=torch.int32,
        ),
        torch.tensor(
            [[0, 1, 1]],
            dtype=torch.long,
        ),
        torch.tensor(
            [0, 1],
            dtype=torch.long,
        ),
        torch.tensor(
            [0, 2, 1],
            dtype=torch.long,
        ),
    ),
)
def test_relation_family_rejects_invalid_mapping(
    mapping: torch.Tensor,
) -> None:
    with pytest.raises(ValueError):
        _families(mapping=mapping)


def test_relation_family_requires_every_family_to_be_represented() -> None:
    with pytest.raises(
        ValueError,
        match="Every declared",
    ):
        _families(
            mapping=torch.tensor(
                [0, 0, 0],
                dtype=torch.long,
            )
        )


@pytest.mark.parametrize(
    "field",
    (
        "source_relation_registry_fingerprint",
        "compiled_relation_registry_fingerprint",
        "schema_version",
    ),
)
def test_relation_family_rejects_blank_fingerprint_fields(
    field: str,
) -> None:
    kwargs = {
        "family_names": (
            "spatial",
            "temporal_control",
        ),
        "stable_family_ids": (
            10,
            20,
        ),
        "relation_family_index_by_relation": (
            torch.tensor(
                [0, 1, 1],
                dtype=torch.long,
            )
        ),
        "relation_names": (
            "r0",
            "r1",
            "r2",
        ),
        "stable_relation_ids": (
            100,
            200,
            900,
        ),
        "source_relation_registry_fingerprint": (
            "source"
        ),
        "compiled_relation_registry_fingerprint": (
            "compiled"
        ),
        field: "",
    }

    with pytest.raises(
        ValueError,
        match="non-empty",
    ):
        RelationFamilyAlignment(**kwargs)


# =============================================================================
# FunctionalMessagePassingInputs
# =============================================================================


def test_inputs_valid_contract_and_derived_views() -> None:
    families = _families()
    inputs = _inputs(
        relation_families=families,
    )

    assert inputs.num_nodes == NODES
    assert inputs.num_edges == EDGES
    assert inputs.num_graphs == GRAPHS
    assert inputs.hidden_dim == HIDDEN_DIM
    assert inputs.num_relations == RELATIONS
    assert inputs.num_relation_families == (
        FAMILIES
    )
    assert inputs.device == torch.device(
        "cpu"
    )
    assert inputs.dtype == torch.float32

    assert torch.equal(
        inputs.source_index,
        _edge_index()[0],
    )
    assert torch.equal(
        inputs.target_index,
        _edge_index()[1],
    )
    assert torch.equal(
        inputs.edge_relation_index,
        _edge_relation_index(),
    )
    assert torch.equal(
        inputs.edge_batch_index,
        _edge_batch(),
    )
    assert torch.equal(
        inputs.control_relation_mask,
        torch.tensor(
            [False, False, True],
            dtype=torch.bool,
        ),
    )
    assert torch.equal(
        inputs.control_edge_mask,
        torch.tensor(
            [False, False, False, True, False],
            dtype=torch.bool,
        ),
    )
    assert torch.equal(
        inputs.edge_relation_family_index,
        torch.tensor(
            [0, 0, 1, 1, 1],
            dtype=torch.long,
        ),
    )


def test_inputs_attention_group_ids_are_target_relation_pairs() -> None:
    inputs = _inputs()

    expected = (
        inputs.target_index
        * RELATIONS
        + inputs.edge_relation_index
    )

    assert torch.equal(
        inputs.attention_group_id,
        expected,
    )
    assert inputs.attention_num_groups == (
        NODES * RELATIONS
    )


def test_inputs_without_family_metadata_returns_none() -> None:
    inputs = _inputs()

    assert (
        inputs.num_relation_families
        is None
    )
    assert (
        inputs.edge_relation_family_index
        is None
    )


def test_inputs_derives_edge_batch_when_source_omits_it() -> None:
    inputs = _inputs(
        graph=_graph(
            edge_batch_index=None,
        )
    )

    assert torch.equal(
        inputs.edge_batch_index,
        _edge_batch(),
    )


def test_inputs_accepts_matching_explicit_edge_batch() -> None:
    inputs = _inputs(
        graph=_graph(
            edge_batch_index=_edge_batch(),
        )
    )

    assert torch.equal(
        inputs.edge_batch_index,
        _edge_batch(),
    )


def test_inputs_expands_graph_scoped_hazard_query() -> None:
    graph_query = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        dtype=torch.float32,
    )
    hazard = FakeHazardQueryEncoding(
        query=graph_query,
        source_embedding=(
            FakeHazardEmbeddingLookup()
        ),
    )

    inputs = _inputs(
        hazard_query=hazard,
    )

    assert torch.equal(
        inputs.node_hazard_query,
        graph_query[
            inputs.node_batch_index
        ],
    )


def test_inputs_preserves_node_aligned_hazard_query() -> None:
    query = torch.arange(
        NODES * QUERY_DIM,
        dtype=torch.float32,
    ).reshape(NODES, QUERY_DIM)
    hazard = FakeHazardQueryEncoding(
        query=query,
        source_embedding=(
            FakeNodeAlignedHazardEmbeddingLookup(
                _node_batch()
            )
        ),
    )

    inputs = _inputs(
        hazard_query=hazard,
    )

    assert inputs.node_hazard_query is query


def test_inputs_accepts_matching_compiled_priors() -> None:
    registry = _registry()
    priors = FakeCompiledHazardRelationPriors(
        relation_names=(
            registry.relation_names
        ),
        stable_relation_ids=(
            registry.stable_relation_ids
        ),
        source_compiled_relation_fingerprint=(
            registry.fingerprint()
        ),
    )

    inputs = _inputs(
        registry=registry,
        priors=priors,
    )

    assert (
        inputs.compiled_relation_priors
        is priors
    )


def test_input_lineage_and_value_fingerprints_are_deterministic() -> None:
    first = _inputs(
        relation_families=_families(),
    )
    second = _inputs(
        relation_families=_families(),
    )

    assert first.lineage_dict() == (
        second.lineage_dict()
    )
    assert first.lineage_fingerprint() == (
        second.lineage_fingerprint()
    )
    assert first.value_fingerprint() == (
        second.value_fingerprint()
    )


def test_input_value_fingerprint_changes_with_node_state() -> None:
    first = _inputs()
    second = _inputs(
        node_state=_node_state(
            offset=1.0
        )
    )

    assert first.value_fingerprint() != (
        second.value_fingerprint()
    )


def test_inputs_reject_wrong_top_level_types() -> None:
    with pytest.raises(
        TypeError,
        match="UrbanGraphBatch",
    ):
        FunctionalMessagePassingInputs(
            source_graph=object(),  # type: ignore[arg-type]
            node_state=_node_state(),
            compiled_relation_registry=_registry(),
        )

    with pytest.raises(
        TypeError,
        match="NodeStateFusionOutput",
    ):
        FunctionalMessagePassingInputs(
            source_graph=_graph(),
            node_state=object(),  # type: ignore[arg-type]
            compiled_relation_registry=_registry(),
        )

    with pytest.raises(
        TypeError,
        match="CompiledRelationRegistry",
    ):
        FunctionalMessagePassingInputs(
            source_graph=_graph(),
            node_state=_node_state(),
            compiled_relation_registry=object(),  # type: ignore[arg-type]
        )


def test_inputs_calls_upstream_validation() -> None:
    graph = _graph()
    registry = _registry()

    _inputs(
        graph=graph,
        registry=registry,
    )

    assert graph.validated
    assert registry.validated


def test_inputs_rejects_cross_graph_permission() -> None:
    with pytest.raises(
        ValueError,
        match="forbids cross-graph",
    ):
        _inputs(
            graph=_graph(
                allow_cross_graph_edges=True,
            )
        )


def test_inputs_rejects_zero_node_graph() -> None:
    graph = _graph(
        node_count=0,
        node_batch_index=torch.empty(
            0,
            dtype=torch.long,
        ),
    )
    state = _node_state(
        node_count=0,
        node_batch_index=torch.empty(
            0,
            dtype=torch.long,
        ),
        item_ids=(),
        graph_count=0,
    )

    with pytest.raises(
        ValueError,
        match="at least one node",
    ):
        _inputs(
            graph=graph,
            node_state=state,
        )


def test_inputs_accepts_positive_node_zero_edge_graph() -> None:
    graph = _graph(
        node_count=3,
    )
    state = _node_state(
        node_count=3,
        node_batch_index=torch.zeros(
            3,
            dtype=torch.long,
        ),
        graph_count=1,
    )

    inputs = _inputs(
        graph=graph,
        node_state=state,
    )

    assert inputs.num_nodes == 3
    assert inputs.num_edges == 0
    assert inputs.attention_group_id.shape == (
        0,
    )
    assert inputs.control_edge_mask.shape == (
        0,
    )


def test_inputs_rejects_node_state_row_mismatch() -> None:
    with pytest.raises(
        ValueError,
        match="rows",
    ):
        _inputs(
            node_state=_node_state(
                node_count=4,
                node_batch_index=torch.tensor(
                    [0, 0, 1, 1],
                    dtype=torch.long,
                ),
                graph_count=2,
            )
        )


def test_inputs_requires_stable_item_ids() -> None:
    state = _node_state(
        item_ids=(),
    )

    with pytest.raises(
        ValueError,
        match="stable item_ids",
    ):
        _inputs(node_state=state)


def test_inputs_rejects_item_id_order_mismatch() -> None:
    state = _node_state(
        item_ids=tuple(
            reversed(_node_ids())
        )
    )

    with pytest.raises(
        ValueError,
        match="item_ids differ",
    ):
        _inputs(node_state=state)


def test_inputs_requires_node_batch_alignment() -> None:
    state = _node_state(
        node_batch_index=None,
        graph_count=GRAPHS,
    )
    # Override helper's default.
    state.alignment.node_batch_index = None

    with pytest.raises(
        ValueError,
        match="must preserve node_batch_index",
    ):
        _inputs(node_state=state)


def test_inputs_rejects_node_batch_alignment_mismatch() -> None:
    state = _node_state(
        node_batch_index=torch.tensor(
            [0, 0, 1, 1, 1],
            dtype=torch.long,
        )
    )

    with pytest.raises(
        ValueError,
        match="node_batch_index differs",
    ):
        _inputs(node_state=state)


def test_inputs_rejects_graph_count_mismatch() -> None:
    state = _node_state(
        graph_count=3,
    )

    with pytest.raises(
        ValueError,
        match="graph_count differs",
    ):
        _inputs(node_state=state)


def test_inputs_rejects_relation_index_out_of_range() -> None:
    relation = _edge_relation_index()
    relation[0] = RELATIONS

    with pytest.raises(
        ValueError,
        match="out-of-range",
    ):
        _inputs(
            graph=_graph(
                edge_relation_type=relation,
            )
        )


def test_inputs_rejects_node_index_out_of_range() -> None:
    edges = _edge_index()
    edges[0, 0] = NODES

    with pytest.raises(
        ValueError,
        match="out-of-range",
    ):
        _inputs(
            graph=_graph(
                edge_index=edges,
            )
        )


def test_inputs_rejects_actual_cross_graph_edge() -> None:
    edges = _edge_index()
    edges[:, 0] = torch.tensor(
        [0, 4],
        dtype=torch.long,
    )

    with pytest.raises(
        ValueError,
        match="Cross-graph",
    ):
        _inputs(
            graph=_graph(
                edge_index=edges,
            )
        )


def test_inputs_rejects_incorrect_explicit_edge_batch() -> None:
    wrong = _edge_batch()
    wrong[0] = 1

    with pytest.raises(
        ValueError,
        match="edge_batch_index differs",
    ):
        _inputs(
            graph=_graph(
                edge_batch_index=wrong,
            )
        )


def test_inputs_rejects_family_registry_mismatches() -> None:
    with pytest.raises(
        ValueError,
        match="relation ordering differs",
    ):
        _inputs(
            relation_families=_families(
                relation_names=(
                    "temporal_lag",
                    "spatial_adjacency",
                    "random_placebo",
                )
            )
        )

    with pytest.raises(
        ValueError,
        match="stable relation IDs differ",
    ):
        _inputs(
            relation_families=_families(
                stable_relation_ids=(
                    100,
                    201,
                    900,
                )
            )
        )

    with pytest.raises(
        ValueError,
        match="different compiled",
    ):
        _inputs(
            relation_families=_families(
                registry_fingerprint="other"
            )
        )


def test_inputs_rejects_wrong_family_type() -> None:
    with pytest.raises(
        TypeError,
        match="RelationFamilyAlignment",
    ):
        _inputs(
            relation_families=object(),  # type: ignore[arg-type]
        )


def test_inputs_rejects_unsupported_hazard_source() -> None:
    hazard = FakeHazardQueryEncoding(
        query=torch.zeros(
            NODES,
            QUERY_DIM,
        ),
        source_embedding=object(),
    )

    with pytest.raises(
        TypeError,
        match="unsupported",
    ):
        _inputs(
            hazard_query=hazard,
        )


def test_inputs_rejects_node_hazard_row_mismatch() -> None:
    hazard = FakeHazardQueryEncoding(
        query=torch.zeros(
            NODES - 1,
            QUERY_DIM,
        ),
        source_embedding=(
            FakeNodeAlignedHazardEmbeddingLookup(
                _node_batch()
            )
        ),
    )

    with pytest.raises(
        ValueError,
        match="rows must match",
    ):
        _inputs(
            hazard_query=hazard,
        )


def test_inputs_rejects_node_hazard_membership_mismatch() -> None:
    hazard = FakeHazardQueryEncoding(
        query=torch.zeros(
            NODES,
            QUERY_DIM,
        ),
        source_embedding=(
            FakeNodeAlignedHazardEmbeddingLookup(
                torch.tensor(
                    [0, 0, 1, 1, 1],
                    dtype=torch.long,
                )
            )
        ),
    )

    with pytest.raises(
        ValueError,
        match="membership differs",
    ):
        _inputs(
            hazard_query=hazard,
        )


def test_inputs_rejects_graph_hazard_row_mismatch() -> None:
    hazard = FakeHazardQueryEncoding(
        query=torch.zeros(
            GRAPHS + 1,
            QUERY_DIM,
        ),
        source_embedding=(
            FakeHazardEmbeddingLookup()
        ),
    )

    with pytest.raises(
        ValueError,
        match="rows must match",
    ):
        _inputs(
            hazard_query=hazard,
        )


def test_inputs_rejects_prior_alignment_mismatches() -> None:
    registry = _registry()

    with pytest.raises(
        ValueError,
        match="relation ordering differs",
    ):
        _inputs(
            registry=registry,
            priors=FakeCompiledHazardRelationPriors(
                relation_names=tuple(
                    reversed(
                        registry.relation_names
                    )
                ),
                stable_relation_ids=(
                    registry.stable_relation_ids
                ),
                source_compiled_relation_fingerprint=(
                    registry.fingerprint()
                ),
            ),
        )

    with pytest.raises(
        ValueError,
        match="stable relation IDs differ",
    ):
        _inputs(
            registry=registry,
            priors=FakeCompiledHazardRelationPriors(
                relation_names=(
                    registry.relation_names
                ),
                stable_relation_ids=(
                    100,
                    201,
                    900,
                ),
                source_compiled_relation_fingerprint=(
                    registry.fingerprint()
                ),
            ),
        )

    with pytest.raises(
        ValueError,
        match="different compiled",
    ):
        _inputs(
            registry=registry,
            priors=FakeCompiledHazardRelationPriors(
                relation_names=(
                    registry.relation_names
                ),
                stable_relation_ids=(
                    registry.stable_relation_ids
                ),
                source_compiled_relation_fingerprint=(
                    "other"
                ),
            ),
        )


def test_inputs_rejects_floating_dtype_mismatch() -> None:
    graph = _graph(
        edge_attributes=torch.zeros(
            EDGES,
            2,
            dtype=torch.float64,
        )
    )

    with pytest.raises(
        ValueError,
        match="share one dtype",
    ):
        _inputs(graph=graph)


def test_inputs_rejects_nonfinite_floating_values() -> None:
    state = _node_state()
    state.fused_state[0, 0] = float(
        "nan"
    )

    with pytest.raises(
        ValueError,
        match="finite",
    ):
        _inputs(node_state=state)


@pytest.mark.parametrize(
    "field",
    (
        "source_fingerprint",
        "schema_version",
    ),
)
def test_inputs_rejects_blank_identity_fields(
    field: str,
) -> None:
    kwargs: dict[str, Any] = {
        "source_graph": _graph(),
        "node_state": _node_state(),
        "compiled_relation_registry": (
            _registry()
        ),
        "source_fingerprint": "source",
        field: "",
    }

    with pytest.raises(
        ValueError,
        match="non-empty",
    ):
        FunctionalMessagePassingInputs(**kwargs)


# =============================================================================
# RelationTransformOutput
# =============================================================================


def test_relation_transform_output_valid_contract() -> None:
    inputs = _inputs()
    output = _transform(inputs)

    assert output.num_edges == EDGES
    assert output.hidden_dim == HIDDEN_DIM
    assert output.source_inputs is inputs
    assert isinstance(
        output.relation_parameter_fingerprints,
        MappingProxyType,
    )


def test_relation_transform_mapping_is_read_only() -> None:
    output = _transform(_inputs())

    with pytest.raises(TypeError):
        output.relation_parameter_fingerprints[
            "x"
        ] = "y"  # type: ignore[index]


@pytest.mark.parametrize(
    "values",
    (
        torch.zeros(
            EDGES - 1,
            HIDDEN_DIM,
        ),
        torch.zeros(
            EDGES,
            HIDDEN_DIM + 1,
        ),
        torch.zeros(
            EDGES,
            HIDDEN_DIM,
            dtype=torch.long,
        ),
    ),
)
def test_relation_transform_rejects_invalid_tensor(
    values: torch.Tensor,
) -> None:
    with pytest.raises(ValueError):
        _transform(
            _inputs(),
            values=values,
        )


def test_relation_transform_rejects_unexpected_relation_fingerprint() -> None:
    inputs = _inputs()
    values = torch.zeros(
        EDGES,
        HIDDEN_DIM,
    )

    with pytest.raises(
        ValueError,
        match="outside the compiled registry",
    ):
        RelationTransformOutput(
            transformed_source_state=values,
            source_inputs=inputs,
            transform_mode="shared",
            encoder_architecture_fingerprint=(
                "architecture"
            ),
            relation_parameter_fingerprints={
                "unknown": "fingerprint"
            },
        )


def test_relation_transform_rejects_blank_identity() -> None:
    with pytest.raises(
        ValueError,
        match="non-empty",
    ):
        RelationTransformOutput(
            transformed_source_state=torch.zeros(
                EDGES,
                HIDDEN_DIM,
            ),
            source_inputs=_inputs(),
            transform_mode="",
            encoder_architecture_fingerprint=(
                "architecture"
            ),
        )


# =============================================================================
# StructuralEdgeNormalizationOutput
# =============================================================================


def test_none_normalization_requires_exact_identity() -> None:
    output = _normalization(
        _inputs()
    )

    assert torch.equal(
        output.coefficients,
        torch.ones(EDGES),
    )


def test_normalization_accepts_degree_diagnostics() -> None:
    inputs = _inputs()
    output = (
        StructuralEdgeNormalizationOutput(
            coefficients=torch.ones(
                EDGES
            ),
            source_inputs=inputs,
            normalization_mode=(
                EDGE_NORMALIZATION_NONE
            ),
            encoder_architecture_fingerprint=(
                "architecture"
            ),
            source_degree=torch.tensor(
                [1, 1, 1, 1, 1],
                dtype=torch.long,
            ),
            target_degree=torch.tensor(
                [0, 2, 1, 1, 1],
                dtype=torch.long,
            ),
        )
    )

    assert output.source_degree is not None
    assert output.target_degree is not None


def test_none_normalization_rejects_nonidentity_coefficients() -> None:
    with pytest.raises(
        ValueError,
        match="exact multiplicative identity",
    ):
        _normalization(
            _inputs(),
            coefficients=torch.full(
                (EDGES,),
                0.5,
            ),
        )


def test_normalization_rejects_negative_coefficients() -> None:
    coefficients = torch.ones(
        EDGES
    )
    coefficients[0] = -1.0

    with pytest.raises(
        ValueError,
        match="nonnegative",
    ):
        StructuralEdgeNormalizationOutput(
            coefficients=coefficients,
            source_inputs=_inputs(),
            normalization_mode="future_mode",
            encoder_architecture_fingerprint=(
                "architecture"
            ),
        )


def test_normalization_rejects_invalid_degree_shape() -> None:
    with pytest.raises(ValueError):
        StructuralEdgeNormalizationOutput(
            coefficients=torch.ones(
                EDGES
            ),
            source_inputs=_inputs(),
            normalization_mode=(
                EDGE_NORMALIZATION_NONE
            ),
            encoder_architecture_fingerprint=(
                "architecture"
            ),
            source_degree=torch.zeros(
                NODES - 1,
                dtype=torch.long,
            ),
        )


# =============================================================================
# RelationGateOutput
# =============================================================================


def test_gate_output_valid_exact_relation_axis() -> None:
    inputs = _inputs()
    gate = _gate(inputs)

    assert gate.gate_logits.shape == (
        NODES,
        RELATIONS,
    )
    assert gate.edge_gate_values.shape == (
        EDGES,
    )
    assert torch.equal(
        gate.control_relation_mask,
        inputs.control_relation_mask,
    )
    assert isinstance(
        gate.regularization_terms,
        MappingProxyType,
    )


def test_gate_edge_lookup_is_exact() -> None:
    inputs = _inputs()
    gate = _gate(inputs)

    expected = gate.gate_values[
        inputs.target_index,
        inputs.edge_relation_index,
    ]
    assert torch.allclose(
        gate.edge_gate_values,
        expected,
    )


@pytest.mark.parametrize(
    ("scope", "activation"),
    (
        ("graph", RELATION_GATE_ACTIVATION_SIGMOID),
        (RELATION_GATE_SCOPE_TARGET_NODE, "softmax"),
    ),
)
def test_gate_rejects_unsupported_scope_or_activation(
    scope: str,
    activation: str,
) -> None:
    inputs = _inputs()
    logits = torch.zeros(
        NODES,
        RELATIONS,
    )
    values = torch.sigmoid(logits)

    with pytest.raises(
        ValueError,
        match="bounded V2.0",
    ):
        RelationGateOutput(
            gate_logits=logits,
            gate_values=values,
            edge_gate_values=values[
                inputs.target_index,
                inputs.edge_relation_index,
            ],
            source_inputs=inputs,
            scope=scope,
            activation=activation,
            encoder_architecture_fingerprint=(
                "architecture"
            ),
        )


def test_gate_rejects_values_outside_unit_interval() -> None:
    inputs = _inputs()
    logits = torch.zeros(
        NODES,
        RELATIONS,
    )
    values = torch.ones(
        NODES,
        RELATIONS,
    )
    values[0, 0] = 1.1

    with pytest.raises(
        ValueError,
        match=r"\[0, 1\]",
    ):
        RelationGateOutput(
            gate_logits=logits,
            gate_values=values,
            edge_gate_values=torch.ones(
                EDGES
            ),
            source_inputs=inputs,
            scope=RELATION_GATE_SCOPE_TARGET_NODE,
            activation=(
                RELATION_GATE_ACTIVATION_SIGMOID
            ),
            encoder_architecture_fingerprint=(
                "architecture"
            ),
        )


def test_gate_rejects_wrong_edge_lookup() -> None:
    inputs = _inputs()
    logits = torch.zeros(
        NODES,
        RELATIONS,
    )
    values = torch.sigmoid(logits)

    with pytest.raises(
        ValueError,
        match="must equal target-node gate lookup",
    ):
        RelationGateOutput(
            gate_logits=logits,
            gate_values=values,
            edge_gate_values=torch.zeros(
                EDGES
            ),
            source_inputs=inputs,
            scope=RELATION_GATE_SCOPE_TARGET_NODE,
            activation=(
                RELATION_GATE_ACTIVATION_SIGMOID
            ),
            encoder_architecture_fingerprint=(
                "architecture"
            ),
        )


def test_gate_regularization_mapping_is_read_only() -> None:
    gate = _gate(_inputs())

    with pytest.raises(TypeError):
        gate.regularization_terms[
            "new"
        ] = torch.tensor(1.0)  # type: ignore[index]


# =============================================================================
# EdgeAttentionOutput
# =============================================================================


def test_attention_output_valid_grouped_contract() -> None:
    inputs = _inputs()
    attention = _attention(inputs)

    assert attention.num_heads == HEADS
    assert attention.raw_scores_by_head.shape == (
        EDGES,
        HEADS,
    )
    assert torch.equal(
        attention.group_ids,
        inputs.attention_group_id,
    )
    assert attention.group_counts.shape == (
        NODES * RELATIONS,
    )


def test_attention_nonempty_groups_sum_to_one_per_head() -> None:
    inputs = _inputs()
    attention = _attention(inputs)

    sums = torch.zeros(
        inputs.attention_num_groups,
        HEADS,
    )
    sums.index_add_(
        0,
        attention.group_ids,
        attention.normalized_weights_by_head,
    )
    nonempty = (
        attention.group_counts > 0
    )

    assert torch.allclose(
        sums[nonempty],
        torch.ones_like(
            sums[nonempty]
        ),
    )


def test_attention_single_edge_groups_receive_one() -> None:
    inputs = _inputs()
    attention = _attention(inputs)
    edge_group_counts = (
        attention.group_counts[
            attention.group_ids
        ]
    )
    single = edge_group_counts == 1

    assert torch.equal(
        attention.normalized_weights_by_head[
            single
        ],
        torch.ones_like(
            attention
            .normalized_weights_by_head[
                single
            ]
        ),
    )


def test_attention_rejects_wrong_group_ids() -> None:
    inputs = _inputs()
    normalized, group_ids, counts = (
        _uniform_attention_weights(inputs)
    )
    wrong = group_ids.clone()
    wrong[0] += 1

    with pytest.raises(
        ValueError,
        match="must encode target node",
    ):
        EdgeAttentionOutput(
            raw_scores_by_head=torch.zeros_like(
                normalized
            ),
            normalized_weights_by_head=(
                normalized
            ),
            edge_weights=normalized.mean(
                dim=1
            ),
            group_ids=wrong,
            group_counts=counts,
            source_inputs=inputs,
            attention_mode="uniform",
            normalization_mode=(
                ATTENTION_NORMALIZATION_TARGET_NODE_RELATION
            ),
            head_reduction=(
                ATTENTION_HEAD_REDUCTION_MEAN
            ),
            encoder_architecture_fingerprint=(
                "architecture"
            ),
        )


def test_attention_rejects_wrong_group_counts() -> None:
    inputs = _inputs()
    normalized, group_ids, counts = (
        _uniform_attention_weights(inputs)
    )
    wrong = counts.clone()
    wrong[group_ids[0]] += 1

    with pytest.raises(
        ValueError,
        match="do not match",
    ):
        EdgeAttentionOutput(
            raw_scores_by_head=torch.zeros_like(
                normalized
            ),
            normalized_weights_by_head=(
                normalized
            ),
            edge_weights=normalized.mean(
                dim=1
            ),
            group_ids=group_ids,
            group_counts=wrong,
            source_inputs=inputs,
            attention_mode="uniform",
            normalization_mode=(
                ATTENTION_NORMALIZATION_TARGET_NODE_RELATION
            ),
            head_reduction=(
                ATTENTION_HEAD_REDUCTION_MEAN
            ),
            encoder_architecture_fingerprint=(
                "architecture"
            ),
        )


def test_attention_rejects_group_sum_violation() -> None:
    inputs = _inputs()
    normalized, group_ids, counts = (
        _uniform_attention_weights(inputs)
    )
    normalized = normalized.clone()
    normalized[0] = 0.0

    with pytest.raises(
        ValueError,
        match="must sum to one",
    ):
        EdgeAttentionOutput(
            raw_scores_by_head=torch.zeros_like(
                normalized
            ),
            normalized_weights_by_head=(
                normalized
            ),
            edge_weights=normalized.mean(
                dim=1
            ),
            group_ids=group_ids,
            group_counts=counts,
            source_inputs=inputs,
            attention_mode="uniform",
            normalization_mode=(
                ATTENTION_NORMALIZATION_TARGET_NODE_RELATION
            ),
            head_reduction=(
                ATTENTION_HEAD_REDUCTION_MEAN
            ),
            encoder_architecture_fingerprint=(
                "architecture"
            ),
        )


def test_attention_rejects_wrong_mean_reduction() -> None:
    inputs = _inputs()
    normalized, group_ids, counts = (
        _uniform_attention_weights(inputs)
    )

    with pytest.raises(
        ValueError,
        match="arithmetic mean",
    ):
        EdgeAttentionOutput(
            raw_scores_by_head=torch.zeros_like(
                normalized
            ),
            normalized_weights_by_head=(
                normalized
            ),
            edge_weights=torch.zeros(
                EDGES
            ),
            group_ids=group_ids,
            group_counts=counts,
            source_inputs=inputs,
            attention_mode="uniform",
            normalization_mode=(
                ATTENTION_NORMALIZATION_TARGET_NODE_RELATION
            ),
            head_reduction=(
                ATTENTION_HEAD_REDUCTION_MEAN
            ),
            encoder_architecture_fingerprint=(
                "architecture"
            ),
        )


def test_attention_rejects_unsupported_normalization() -> None:
    inputs = _inputs()
    normalized, group_ids, counts = (
        _uniform_attention_weights(inputs)
    )

    with pytest.raises(
        ValueError,
        match="exact target-node",
    ):
        EdgeAttentionOutput(
            raw_scores_by_head=torch.zeros_like(
                normalized
            ),
            normalized_weights_by_head=(
                normalized
            ),
            edge_weights=normalized.mean(
                dim=1
            ),
            group_ids=group_ids,
            group_counts=counts,
            source_inputs=inputs,
            attention_mode="uniform",
            normalization_mode="target_node_family",
            head_reduction=(
                ATTENTION_HEAD_REDUCTION_MEAN
            ),
            encoder_architecture_fingerprint=(
                "architecture"
            ),
        )


def test_attention_supports_zero_edge_graph() -> None:
    graph = _graph(node_count=3)
    state = _node_state(
        node_count=3,
        node_batch_index=torch.zeros(
            3,
            dtype=torch.long,
        ),
        graph_count=1,
    )
    inputs = _inputs(
        graph=graph,
        node_state=state,
    )
    attention = _attention(inputs)

    assert attention.raw_scores_by_head.shape == (
        0,
        HEADS,
    )
    assert attention.edge_weights.shape == (
        0,
    )
    assert attention.group_counts.sum().item() == 0


# =============================================================================
# EdgeMessageOutput
# =============================================================================


def test_edge_message_output_valid_full_product() -> None:
    inputs = _inputs()
    semantic = torch.linspace(
        0.5,
        1.0,
        steps=EDGES,
    )
    output = _messages(
        inputs,
        semantic=semantic,
    )

    assert output.edge_messages.shape == (
        EDGES,
        HIDDEN_DIM,
    )
    assert output.source_inputs is inputs
    assert output.gate_factor is not None
    assert output.attention_factor is not None
    assert torch.equal(
        output.structural_factor,
        torch.ones(EDGES),
    )


def test_edge_message_disabled_gate_and_attention_use_identity() -> None:
    inputs = _inputs()
    output = _messages(
        inputs,
        with_gate=False,
        with_attention=False,
    )

    assert output.relation_gate is None
    assert output.edge_attention is None
    assert output.gate_factor is None
    assert output.attention_factor is None
    assert torch.equal(
        output.edge_messages,
        output
        .relation_transform
        .transformed_source_state,
    )


def test_edge_message_rejects_incorrect_product() -> None:
    inputs = _inputs()
    transform = _transform(inputs)
    normalization = _normalization(inputs)

    with pytest.raises(
        ValueError,
        match="explicit product",
    ):
        EdgeMessageOutput(
            edge_messages=torch.zeros(
                EDGES,
                HIDDEN_DIM,
            ),
            relation_transform=transform,
            edge_normalization=normalization,
            relation_gate=None,
            edge_attention=None,
            semantic_edge_weight=None,
            encoder_architecture_fingerprint=(
                "architecture"
            ),
        )


def test_edge_message_rejects_different_source_objects() -> None:
    first = _inputs()
    second = _inputs()

    with pytest.raises(
        ValueError,
        match="same FunctionalMessagePassingInputs",
    ):
        EdgeMessageOutput(
            edge_messages=torch.zeros(
                EDGES,
                HIDDEN_DIM,
            ),
            relation_transform=_transform(
                first,
                values=torch.zeros(
                    EDGES,
                    HIDDEN_DIM,
                ),
            ),
            edge_normalization=_normalization(
                second
            ),
            relation_gate=None,
            edge_attention=None,
            semantic_edge_weight=None,
            encoder_architecture_fingerprint=(
                "architecture"
            ),
        )


# =============================================================================
# AggregationOutput
# =============================================================================


def test_aggregation_output_valid_mean_and_isolated_nodes() -> None:
    inputs = _inputs()
    messages = _messages(inputs)
    output = _aggregation(messages)

    assert output.node_aggregate.shape == (
        NODES,
        HIDDEN_DIM,
    )
    assert torch.equal(
        output.incoming_edge_count,
        torch.tensor(
            [0, 2, 1, 1, 1],
            dtype=torch.long,
        ),
    )
    assert torch.equal(
        output.node_aggregate[0],
        torch.zeros(HIDDEN_DIM),
    )


def test_aggregation_rejects_wrong_counts() -> None:
    messages = _messages(_inputs())
    wrong = torch.zeros(
        NODES,
        dtype=torch.long,
    )

    with pytest.raises(
        ValueError,
        match="does not match",
    ):
        AggregationOutput(
            node_aggregate=torch.zeros(
                NODES,
                HIDDEN_DIM,
            ),
            incoming_edge_count=wrong,
            source_messages=messages,
            aggregation_mode=AGGREGATION_MEAN,
            encoder_architecture_fingerprint=(
                "architecture"
            ),
        )


def test_aggregation_rejects_wrong_mean() -> None:
    messages = _messages(_inputs())
    counts = torch.bincount(
        messages.source_inputs.target_index,
        minlength=NODES,
    )

    with pytest.raises(
        ValueError,
        match="does not match mean aggregation",
    ):
        AggregationOutput(
            node_aggregate=torch.zeros(
                NODES,
                HIDDEN_DIM,
            ),
            incoming_edge_count=counts,
            source_messages=messages,
            aggregation_mode=AGGREGATION_MEAN,
            encoder_architecture_fingerprint=(
                "architecture"
            ),
        )


def test_aggregation_rejects_unimplemented_mode() -> None:
    messages = _messages(_inputs())
    valid = _aggregation(messages)

    with pytest.raises(
        ValueError,
        match="bounded V2.0",
    ):
        AggregationOutput(
            node_aggregate=(
                valid.node_aggregate
            ),
            incoming_edge_count=(
                valid.incoming_edge_count
            ),
            source_messages=messages,
            aggregation_mode="sum",
            encoder_architecture_fingerprint=(
                "architecture"
            ),
        )


def test_zero_edge_aggregation_is_exact_zero() -> None:
    graph = _graph(node_count=3)
    state = _node_state(
        node_count=3,
        node_batch_index=torch.zeros(
            3,
            dtype=torch.long,
        ),
        graph_count=1,
    )
    inputs = _inputs(
        graph=graph,
        node_state=state,
    )
    output = _aggregation(
        _messages(
            inputs,
            with_gate=False,
            with_attention=False,
        )
    )

    assert torch.equal(
        output.incoming_edge_count,
        torch.zeros(
            3,
            dtype=torch.long,
        ),
    )
    assert torch.equal(
        output.node_aggregate,
        torch.zeros(
            3,
            HIDDEN_DIM,
        ),
    )


# =============================================================================
# FunctionalMessagePassingIntermediates
# =============================================================================


def test_intermediates_valid_contract() -> None:
    inputs = _inputs()
    intermediates = _intermediates(
        inputs
    )

    assert (
        intermediates
        .relation_transform
        .source_inputs
        is inputs
    )
    assert (
        intermediates
        .aggregation
        .source_messages
        is intermediates.edge_messages
    )


def test_intermediates_rejects_different_source_objects() -> None:
    first = _inputs()
    second = _inputs()
    base = _intermediates(first)

    with pytest.raises(
        ValueError,
        match="different source inputs",
    ):
        FunctionalMessagePassingIntermediates(
            relation_transform=(
                base.relation_transform
            ),
            edge_normalization=_normalization(
                second
            ),
            relation_gate=base.relation_gate,
            edge_attention=base.edge_attention,
            edge_messages=base.edge_messages,
            aggregation=base.aggregation,
            pre_residual_state=(
                base.pre_residual_state
            ),
            post_residual_state=(
                base.post_residual_state
            ),
        )


def test_intermediates_rejects_state_shape_mismatch() -> None:
    base = _intermediates(_inputs())

    with pytest.raises(ValueError):
        FunctionalMessagePassingIntermediates(
            relation_transform=(
                base.relation_transform
            ),
            edge_normalization=(
                base.edge_normalization
            ),
            relation_gate=base.relation_gate,
            edge_attention=base.edge_attention,
            edge_messages=base.edge_messages,
            aggregation=base.aggregation,
            pre_residual_state=torch.zeros(
                NODES - 1,
                HIDDEN_DIM,
            ),
            post_residual_state=(
                base.post_residual_state
            ),
        )


# =============================================================================
# FunctionalMessagePassingLayerOutput
# =============================================================================


def test_layer_output_valid_with_intermediates() -> None:
    inputs = _inputs()
    output = _layer_output(inputs)

    assert output.num_nodes == NODES
    assert output.hidden_dim == HIDDEN_DIM
    assert output.layer_index == 0
    assert output.intermediates is not None
    assert isinstance(
        output.regularization_terms,
        MappingProxyType,
    )


def test_layer_output_valid_without_intermediates() -> None:
    output = _layer_output(
        _inputs(),
        retain_intermediates=False,
    )

    assert output.intermediates is None


@pytest.mark.parametrize(
    "field",
    (
        "residual_enabled",
        "layer_norm_enabled",
    ),
)
def test_layer_output_rejects_nonboolean_flags(
    field: str,
) -> None:
    inputs = _inputs()
    base = _layer_output(inputs)
    kwargs = {
        "updated_node_state": (
            base.updated_node_state
        ),
        "node_aggregate": (
            base.node_aggregate
        ),
        "incoming_edge_count": (
            base.incoming_edge_count
        ),
        "source_inputs": inputs,
        "layer_index": 0,
        "residual_enabled": True,
        "layer_norm_enabled": False,
        "encoder_architecture_fingerprint": (
            "architecture"
        ),
        "lineage_fingerprint": "lineage",
        field: 1,
    }

    with pytest.raises(
        TypeError,
        match="Boolean",
    ):
        FunctionalMessagePassingLayerOutput(
            **kwargs
        )


def test_layer_output_rejects_wrong_edge_counts() -> None:
    inputs = _inputs()
    base = _layer_output(inputs)

    with pytest.raises(
        ValueError,
        match="does not match",
    ):
        FunctionalMessagePassingLayerOutput(
            updated_node_state=(
                base.updated_node_state
            ),
            node_aggregate=(
                base.node_aggregate
            ),
            incoming_edge_count=torch.zeros(
                NODES,
                dtype=torch.long,
            ),
            source_inputs=inputs,
            layer_index=0,
            residual_enabled=True,
            layer_norm_enabled=False,
            encoder_architecture_fingerprint=(
                "architecture"
            ),
            lineage_fingerprint="lineage",
        )


def test_layer_output_rejects_intermediate_aggregate_mismatch() -> None:
    inputs = _inputs()
    base = _layer_output(inputs)
    wrong_aggregate = (
        base.node_aggregate + 1.0
    )

    with pytest.raises(
        ValueError,
        match="differs from retained aggregation",
    ):
        FunctionalMessagePassingLayerOutput(
            updated_node_state=(
                base.updated_node_state
            ),
            node_aggregate=wrong_aggregate,
            incoming_edge_count=(
                base.incoming_edge_count
            ),
            source_inputs=inputs,
            layer_index=0,
            residual_enabled=True,
            layer_norm_enabled=False,
            encoder_architecture_fingerprint=(
                "architecture"
            ),
            lineage_fingerprint="lineage",
            intermediates=base.intermediates,
        )


def test_layer_regularization_mapping_is_read_only() -> None:
    output = _layer_output(_inputs())

    with pytest.raises(TypeError):
        output.regularization_terms[
            "x"
        ] = torch.tensor(1.0)  # type: ignore[index]


# =============================================================================
# FunctionalMessagePassingStackOutput
# =============================================================================


def test_stack_output_valid_without_retained_layers() -> None:
    inputs = _inputs()
    final = _layer_output(
        inputs,
        retain_intermediates=False,
    )

    output = (
        FunctionalMessagePassingStackOutput(
            final_node_state=(
                final.updated_node_state
            ),
            source_inputs=inputs,
            num_layers=1,
            layer_outputs=(),
            encoder_architecture_fingerprint=(
                "stack-architecture"
            ),
            lineage_fingerprint=(
                "stack-lineage"
            ),
        )
    )

    assert output.num_layers == 1
    assert output.layer_outputs == ()


def test_stack_output_valid_with_retained_layers() -> None:
    inputs = _inputs()
    first = _layer_output(
        inputs,
        layer_index=0,
        offset=0.0,
    )
    second = _layer_output(
        inputs,
        layer_index=1,
        offset=1.0,
    )

    output = (
        FunctionalMessagePassingStackOutput(
            final_node_state=(
                second.updated_node_state
            ),
            source_inputs=inputs,
            num_layers=2,
            layer_outputs=(
                first,
                second,
            ),
            encoder_architecture_fingerprint=(
                "stack-architecture"
            ),
            lineage_fingerprint=(
                "stack-lineage"
            ),
            regularization_terms={
                "total_penalty": torch.tensor(
                    0.2
                )
            },
        )
    )

    assert len(output.layer_outputs) == 2
    assert isinstance(
        output.regularization_terms,
        MappingProxyType,
    )


@pytest.mark.parametrize(
    "num_layers",
    (
        0,
        -1,
        True,
        1.5,
    ),
)
def test_stack_output_rejects_invalid_layer_count(
    num_layers: Any,
) -> None:
    with pytest.raises(
        ValueError,
        match="positive integer",
    ):
        FunctionalMessagePassingStackOutput(
            final_node_state=(
                _node_state().fused_state
            ),
            source_inputs=_inputs(),
            num_layers=num_layers,
            encoder_architecture_fingerprint=(
                "architecture"
            ),
            lineage_fingerprint="lineage",
        )


def test_stack_output_rejects_wrong_retained_count() -> None:
    inputs = _inputs()
    layer = _layer_output(inputs)

    with pytest.raises(
        ValueError,
        match="exactly num_layers",
    ):
        FunctionalMessagePassingStackOutput(
            final_node_state=(
                layer.updated_node_state
            ),
            source_inputs=inputs,
            num_layers=2,
            layer_outputs=(layer,),
            encoder_architecture_fingerprint=(
                "architecture"
            ),
            lineage_fingerprint="lineage",
        )


def test_stack_output_rejects_noncontiguous_indices() -> None:
    inputs = _inputs()
    wrong = _layer_output(
        inputs,
        layer_index=1,
    )

    with pytest.raises(
        ValueError,
        match="contiguous",
    ):
        FunctionalMessagePassingStackOutput(
            final_node_state=(
                wrong.updated_node_state
            ),
            source_inputs=inputs,
            num_layers=1,
            layer_outputs=(wrong,),
            encoder_architecture_fingerprint=(
                "architecture"
            ),
            lineage_fingerprint="lineage",
        )


def test_stack_output_rejects_wrong_final_state() -> None:
    inputs = _inputs()
    layer = _layer_output(inputs)

    with pytest.raises(
        ValueError,
        match="differs from the final retained",
    ):
        FunctionalMessagePassingStackOutput(
            final_node_state=(
                layer.updated_node_state
                + 1.0
            ),
            source_inputs=inputs,
            num_layers=1,
            layer_outputs=(layer,),
            encoder_architecture_fingerprint=(
                "architecture"
            ),
            lineage_fingerprint="lineage",
        )


def test_stack_regularization_mapping_is_read_only() -> None:
    inputs = _inputs()
    layer = _layer_output(inputs)
    output = (
        FunctionalMessagePassingStackOutput(
            final_node_state=(
                layer.updated_node_state
            ),
            source_inputs=inputs,
            num_layers=1,
            encoder_architecture_fingerprint=(
                "architecture"
            ),
            lineage_fingerprint="lineage",
            regularization_terms={
                "penalty": torch.tensor(0.1)
            },
        )
    )

    with pytest.raises(TypeError):
        output.regularization_terms[
            "x"
        ] = torch.tensor(1.0)  # type: ignore[index]


# =============================================================================
# Optional CUDA device checks
# =============================================================================


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is unavailable.",
)
def test_inputs_rejects_device_mismatch() -> None:
    graph = _graph()
    state = _node_state(
        device="cuda",
    )

    with pytest.raises(
        ValueError,
        match="share one device",
    ):
        _inputs(
            graph=graph,
            node_state=state,
        )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is unavailable.",
)
def test_valid_cuda_contracts() -> None:
    device = torch.device("cuda")
    node_batch = _node_batch(
        device=device
    )
    graph = FakeUrbanGraphBatch(
        external_node_ids=_node_ids(),
        node_batch_index=node_batch,
        edge_index=_edge_index(
            device=device
        ),
        edge_relation_type=(
            _edge_relation_index(
                device=device
            )
        ),
        edge_batch_index=_edge_batch(
            device=device
        ),
    )
    state = _node_state(
        node_batch_index=node_batch,
        device=device,
    )
    families = _families(
        mapping=torch.tensor(
            [0, 1, 1],
            dtype=torch.long,
            device=device,
        )
    )
    inputs = _inputs(
        graph=graph,
        node_state=state,
        relation_families=families,
    )

    assert inputs.device.type == "cuda"
    assert _attention(
        inputs
    ).edge_weights.device.type == "cuda"
