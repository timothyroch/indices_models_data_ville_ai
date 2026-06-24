"""
Focused tests for immutable message-builder schemas.

Target repository path:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            tests/
                message_builders/
                    test_message_builder_schemas.py

Implementation under test:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            functional_message_passing/
                message_builders/
                    schemas.py

The suite freezes the contracts that sit between already completed functional
message-passing components and the later message-builder orchestrator.

The tested message equation is:

    edge_messages
        = transformed_source_state
        * structural_normalization_factor
        * relation_gate_factor
        * edge_attention_factor
        * semantic_edge_factor

where scalar edge factors are broadcast only across the hidden-feature axis.

Scientific contracts
--------------------
- ``RelationTransformOutput.transformed_source_state`` is already edge aligned
  ``[E, H]``; no second relation-state gather schema is introduced.
- Structural normalization is always explicit.
- Disabled relation gating resolves to exact one.
- Disabled edge attention resolves to exact one.
- Enabled uniform attention is not equivalent to disabled attention.
- Semantic edge weights are consumed only under an explicit source-graph
  policy.
- Every enabled source tensor must be preserved by exact object identity.
- Internal stages preserve one exact ``FunctionalMessagePassingInputs`` object.
- Message composition performs no aggregation.
- Internal schema stages are parameter-free.
- Architecture, lineage, and value fingerprints represent distinct concepts.
- Empty edge sets, float64, autograd, and optional CUDA are supported.

Controlled upstream doubles are patched into
``functional_message_passing.schemas`` so failures remain localized to these
contracts rather than graph loading, fusion, hazard encoding, or registry
construction.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass, replace
from types import MappingProxyType
from typing import Any

import pytest
import torch

from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.constants import (
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
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.functional_message_passing.message_builders.schemas import (
    CANONICAL_MESSAGE_SEMANTIC_EDGE_POLICIES,
    IMPLEMENTED_MESSAGE_SEMANTIC_EDGE_POLICIES,
    MESSAGE_COMBINED_COEFFICIENT_FORMULA,
    MESSAGE_COMPOSITION_FORMULA,
    MESSAGE_COMPOSITION_OUTPUT_SCHEMA_VERSION,
    MESSAGE_DISABLED_FACTOR_POLICY,
    MESSAGE_FACTOR_EDGE_ATTENTION,
    MESSAGE_FACTOR_ORDER,
    MESSAGE_FACTOR_RELATION_GATE,
    MESSAGE_FACTOR_SEMANTIC_EDGE_WEIGHT,
    MESSAGE_FACTOR_STRUCTURAL_NORMALIZATION,
    MESSAGE_SEMANTIC_EDGE_POLICY_IGNORE,
    MESSAGE_SEMANTIC_EDGE_POLICY_USE_SOURCE_GRAPH,
    MESSAGE_TRANSFORM_INPUT_LAYOUT,
    RESOLVED_MESSAGE_COEFFICIENTS_SCHEMA_VERSION,
    EdgeMessageCompositionOutput,
    EdgeMessageOutput,
    MessageBuilderStages,
    MessageCoefficients,
    MessageCompositionOutput,
    ResolvedMessageCoefficients,
    validate_message_builder_stage_chain,
    validate_public_edge_message_output,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.functional_message_passing.schemas import (
    EdgeAttentionOutput,
    FunctionalMessagePassingInputs,
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


def _resolved(
    inputs: FunctionalMessagePassingInputs,
    *,
    edge_normalization: (
        StructuralEdgeNormalizationOutput | None
    ) = None,
    relation_gate: RelationGateOutput | None | object = _DEFAULT,
    edge_attention: EdgeAttentionOutput | None | object = _DEFAULT,
    semantic_policy: str = (
        MESSAGE_SEMANTIC_EDGE_POLICY_USE_SOURCE_GRAPH
    ),
    structural_factor: torch.Tensor | None = None,
    relation_gate_factor: torch.Tensor | None = None,
    edge_attention_factor: torch.Tensor | None = None,
    semantic_edge_factor: torch.Tensor | None = None,
    semantic_edge_weight: torch.Tensor | None | object = _DEFAULT,
    combined_coefficient: torch.Tensor | None = None,
    resolver_architecture: str = "coefficient-resolver-architecture",
) -> ResolvedMessageCoefficients:
    normalization = (
        _edge_normalization(
            inputs
        )
        if edge_normalization is None
        else edge_normalization
    )

    resolved_gate = (
        _relation_gate(
            inputs
        )
        if relation_gate is _DEFAULT
        else relation_gate
    )
    resolved_attention = (
        _edge_attention(
            inputs
        )
        if edge_attention is _DEFAULT
        else edge_attention
    )

    structural = (
        normalization.coefficients
        if structural_factor is None
        else structural_factor
    )

    if relation_gate_factor is None:
        relation_factor = (
            resolved_gate.edge_gate_values
            if isinstance(
                resolved_gate,
                RelationGateOutput,
            )
            else torch.ones_like(
                structural
            )
        )
    else:
        relation_factor = relation_gate_factor

    if edge_attention_factor is None:
        attention_factor = (
            resolved_attention.edge_weights
            if isinstance(
                resolved_attention,
                EdgeAttentionOutput,
            )
            else torch.ones_like(
                structural
            )
        )
    else:
        attention_factor = edge_attention_factor

    if semantic_policy == (
        MESSAGE_SEMANTIC_EDGE_POLICY_USE_SOURCE_GRAPH
    ):
        graph_semantic = (
            inputs
            .source_graph
            .semantic_edge_weight
        )
        resolved_semantic_weight = (
            graph_semantic
            if semantic_edge_weight is _DEFAULT
            else semantic_edge_weight
        )
        if semantic_edge_factor is None:
            # The factory must remain capable of constructing an invalid
            # use_source_graph case so the schema itself can reject the
            # missing source tensor. Avoid failing prematurely while
            # computing the synthetic combined coefficient.
            semantic_factor = (
                resolved_semantic_weight
                if isinstance(
                    resolved_semantic_weight,
                    torch.Tensor,
                )
                else torch.ones_like(
                    structural
                )
            )
        else:
            semantic_factor = semantic_edge_factor
    else:
        resolved_semantic_weight = (
            None
            if semantic_edge_weight is _DEFAULT
            else semantic_edge_weight
        )
        semantic_factor = (
            torch.ones_like(
                structural
            )
            if semantic_edge_factor is None
            else semantic_edge_factor
        )

    if combined_coefficient is None:
        combined = (
            structural
            * relation_factor
            * attention_factor
            * semantic_factor
        )
    else:
        combined = combined_coefficient

    return ResolvedMessageCoefficients(
        structural_normalization_factor=structural,
        relation_gate_factor=relation_factor,
        edge_attention_factor=attention_factor,
        semantic_edge_factor=semantic_factor,
        combined_coefficient=combined,
        source_inputs=inputs,
        edge_normalization=normalization,
        relation_gate=resolved_gate,  # type: ignore[arg-type]
        edge_attention=resolved_attention,  # type: ignore[arg-type]
        semantic_edge_weight=resolved_semantic_weight,  # type: ignore[arg-type]
        semantic_edge_policy=semantic_policy,
        resolver_architecture_fingerprint=(
            resolver_architecture
        ),
    )


def _composition(
    inputs: FunctionalMessagePassingInputs,
    *,
    relation_transform: RelationTransformOutput | None = None,
    resolved: ResolvedMessageCoefficients | None = None,
    edge_messages: torch.Tensor | None = None,
    composer_architecture: str = "message-composer-architecture",
) -> MessageCompositionOutput:
    transform = (
        _relation_transform(
            inputs
        )
        if relation_transform is None
        else relation_transform
    )
    coefficients = (
        _resolved(
            inputs
        )
        if resolved is None
        else resolved
    )
    messages = (
        transform.transformed_source_state
        * coefficients
        .combined_coefficient
        .unsqueeze(-1)
        if edge_messages is None
        else edge_messages
    )

    return MessageCompositionOutput(
        edge_messages=messages,
        relation_transform=transform,
        resolved_coefficients=coefficients,
        composer_architecture_fingerprint=(
            composer_architecture
        ),
    )


def _public_output(
    composition: MessageCompositionOutput,
    *,
    relation_transform: RelationTransformOutput | None = None,
    edge_normalization: StructuralEdgeNormalizationOutput | None = None,
    relation_gate: RelationGateOutput | None | object = _DEFAULT,
    edge_attention: EdgeAttentionOutput | None | object = _DEFAULT,
    semantic_edge_weight: torch.Tensor | None | object = _DEFAULT,
    edge_messages: torch.Tensor | None = None,
    architecture: str = "public-message-architecture",
) -> EdgeMessageOutput:
    resolved = (
        composition
        .resolved_coefficients
    )

    transform = (
        composition.relation_transform
        if relation_transform is None
        else relation_transform
    )
    normalization = (
        resolved.edge_normalization
        if edge_normalization is None
        else edge_normalization
    )
    gate = (
        resolved.relation_gate
        if relation_gate is _DEFAULT
        else relation_gate
    )
    attention = (
        resolved.edge_attention
        if edge_attention is _DEFAULT
        else edge_attention
    )
    semantic = (
        resolved.semantic_edge_weight
        if semantic_edge_weight is _DEFAULT
        else semantic_edge_weight
    )
    messages = (
        composition.edge_messages
        if edge_messages is None
        else edge_messages
    )

    return EdgeMessageOutput(
        edge_messages=messages,
        relation_transform=transform,
        edge_normalization=normalization,
        relation_gate=gate,  # type: ignore[arg-type]
        edge_attention=attention,  # type: ignore[arg-type]
        semantic_edge_weight=semantic,  # type: ignore[arg-type]
        encoder_architecture_fingerprint=architecture,
    )


# =============================================================================
# Constant and alias contracts
# =============================================================================


def test_schema_versions_are_nonempty() -> None:
    assert (
        RESOLVED_MESSAGE_COEFFICIENTS_SCHEMA_VERSION
        .strip()
    )
    assert (
        MESSAGE_COMPOSITION_OUTPUT_SCHEMA_VERSION
        .strip()
    )


def test_semantic_edge_policy_vocabulary() -> None:
    assert (
        CANONICAL_MESSAGE_SEMANTIC_EDGE_POLICIES
        == (
            MESSAGE_SEMANTIC_EDGE_POLICY_IGNORE,
            MESSAGE_SEMANTIC_EDGE_POLICY_USE_SOURCE_GRAPH,
        )
    )
    assert (
        IMPLEMENTED_MESSAGE_SEMANTIC_EDGE_POLICIES
        == CANONICAL_MESSAGE_SEMANTIC_EDGE_POLICIES
    )
    assert len(
        CANONICAL_MESSAGE_SEMANTIC_EDGE_POLICIES
    ) == len(
        set(
            CANONICAL_MESSAGE_SEMANTIC_EDGE_POLICIES
        )
    )


def test_factor_order_is_complete_and_unique() -> None:
    assert MESSAGE_FACTOR_ORDER == (
        MESSAGE_FACTOR_STRUCTURAL_NORMALIZATION,
        MESSAGE_FACTOR_RELATION_GATE,
        MESSAGE_FACTOR_EDGE_ATTENTION,
        MESSAGE_FACTOR_SEMANTIC_EDGE_WEIGHT,
    )
    assert len(
        MESSAGE_FACTOR_ORDER
    ) == len(
        set(MESSAGE_FACTOR_ORDER)
    )


def test_public_equation_constants() -> None:
    assert MESSAGE_DISABLED_FACTOR_POLICY == (
        "exact_multiplicative_identity_one"
    )
    assert MESSAGE_TRANSFORM_INPUT_LAYOUT == (
        "edge_aligned_transformed_source_state_[E,H]"
    )
    assert "structural_normalization_factor" in (
        MESSAGE_COMBINED_COEFFICIENT_FORMULA
    )
    assert "relation_gate_factor" in (
        MESSAGE_COMBINED_COEFFICIENT_FORMULA
    )
    assert "edge_attention_factor" in (
        MESSAGE_COMBINED_COEFFICIENT_FORMULA
    )
    assert "semantic_edge_factor" in (
        MESSAGE_COMBINED_COEFFICIENT_FORMULA
    )
    assert "unsqueeze(-1)" in (
        MESSAGE_COMPOSITION_FORMULA
    )


def test_compact_aliases_are_exact() -> None:
    assert MessageCoefficients is (
        ResolvedMessageCoefficients
    )
    assert EdgeMessageCompositionOutput is (
        MessageCompositionOutput
    )


def test_message_builder_stages_type_alias_is_importable() -> None:
    assert MessageBuilderStages is not None


def test_public_edge_message_output_is_reexported() -> None:
    assert EdgeMessageOutput is (
        fmp_schemas.EdgeMessageOutput
    )


# =============================================================================
# Valid resolved-coefficient construction
# =============================================================================


def test_resolved_coefficients_full_contract() -> None:
    inputs = _inputs()
    output = _resolved(inputs)

    assert output.source_inputs is inputs
    assert (
        output.structural_normalization_factor
        is output.edge_normalization.coefficients
    )
    assert (
        output.relation_gate_factor
        is output.relation_gate.edge_gate_values
    )
    assert (
        output.edge_attention_factor
        is output.edge_attention.edge_weights
    )
    assert (
        output.semantic_edge_weight
        is inputs.source_graph.semantic_edge_weight
    )
    assert (
        output.semantic_edge_factor
        is output.semantic_edge_weight
    )
    assert output.num_edges == EDGES
    assert output.dtype == inputs.dtype
    assert output.device == inputs.device
    assert output.relation_gate_enabled is True
    assert output.edge_attention_enabled is True
    assert (
        output.semantic_edge_weight_enabled
        is True
    )
    assert output.parameter_fingerprint is None


@pytest.mark.parametrize(
    (
        "relation_gate_enabled",
        "attention_enabled",
        "semantic_policy",
    ),
    (
        (
            False,
            False,
            MESSAGE_SEMANTIC_EDGE_POLICY_IGNORE,
        ),
        (
            False,
            True,
            MESSAGE_SEMANTIC_EDGE_POLICY_IGNORE,
        ),
        (
            True,
            False,
            MESSAGE_SEMANTIC_EDGE_POLICY_IGNORE,
        ),
        (
            True,
            True,
            MESSAGE_SEMANTIC_EDGE_POLICY_IGNORE,
        ),
        (
            False,
            False,
            MESSAGE_SEMANTIC_EDGE_POLICY_USE_SOURCE_GRAPH,
        ),
        (
            False,
            True,
            MESSAGE_SEMANTIC_EDGE_POLICY_USE_SOURCE_GRAPH,
        ),
        (
            True,
            False,
            MESSAGE_SEMANTIC_EDGE_POLICY_USE_SOURCE_GRAPH,
        ),
        (
            True,
            True,
            MESSAGE_SEMANTIC_EDGE_POLICY_USE_SOURCE_GRAPH,
        ),
    ),
)
def test_resolved_coefficients_all_enabled_disabled_combinations(
    relation_gate_enabled: bool,
    attention_enabled: bool,
    semantic_policy: str,
) -> None:
    inputs = _inputs()
    output = _resolved(
        inputs,
        relation_gate=(
            _DEFAULT
            if relation_gate_enabled
            else None
        ),
        edge_attention=(
            _DEFAULT
            if attention_enabled
            else None
        ),
        semantic_policy=semantic_policy,
    )

    assert (
        output.relation_gate_enabled
        is relation_gate_enabled
    )
    assert (
        output.edge_attention_enabled
        is attention_enabled
    )
    assert (
        output.semantic_edge_weight_enabled
        is (
            semantic_policy
            == MESSAGE_SEMANTIC_EDGE_POLICY_USE_SOURCE_GRAPH
        )
    )

    if not relation_gate_enabled:
        assert torch.equal(
            output.relation_gate_factor,
            torch.ones_like(
                output.relation_gate_factor
            ),
        )

    if not attention_enabled:
        assert torch.equal(
            output.edge_attention_factor,
            torch.ones_like(
                output.edge_attention_factor
            ),
        )

    if semantic_policy == (
        MESSAGE_SEMANTIC_EDGE_POLICY_IGNORE
    ):
        assert output.semantic_edge_weight is None
        assert torch.equal(
            output.semantic_edge_factor,
            torch.ones_like(
                output.semantic_edge_factor
            ),
        )


def test_combined_coefficient_is_explicit_product() -> None:
    output = _resolved(
        _inputs()
    )
    expected = (
        output.structural_normalization_factor
        * output.relation_gate_factor
        * output.edge_attention_factor
        * output.semantic_edge_factor
    )

    torch.testing.assert_close(
        output.combined_coefficient,
        expected,
    )


def test_disabled_gate_and_attention_are_not_uniform_attention() -> None:
    inputs = _inputs()
    disabled = _resolved(
        inputs,
        relation_gate=None,
        edge_attention=None,
        semantic_policy=(
            MESSAGE_SEMANTIC_EDGE_POLICY_IGNORE
        ),
    )
    enabled_uniform = _resolved(
        inputs,
        relation_gate=None,
        edge_attention=_edge_attention(
            inputs
        ),
        semantic_policy=(
            MESSAGE_SEMANTIC_EDGE_POLICY_IGNORE
        ),
    )

    assert torch.equal(
        disabled.edge_attention_factor,
        torch.ones_like(
            disabled.edge_attention_factor
        ),
    )
    assert not torch.equal(
        enabled_uniform.edge_attention_factor,
        torch.ones_like(
            enabled_uniform.edge_attention_factor
        ),
    )


def test_semantic_ignore_preserves_graph_field_but_resolves_one() -> None:
    inputs = _inputs()
    graph_semantic = (
        inputs
        .source_graph
        .semantic_edge_weight
    )
    output = _resolved(
        inputs,
        semantic_policy=(
            MESSAGE_SEMANTIC_EDGE_POLICY_IGNORE
        ),
    )

    assert graph_semantic is not None
    assert output.semantic_edge_weight is None
    assert torch.equal(
        output.semantic_edge_factor,
        torch.ones_like(
            output.semantic_edge_factor
        ),
    )


def test_semantic_use_source_graph_preserves_exact_tensor() -> None:
    inputs = _inputs()
    output = _resolved(
        inputs,
        semantic_policy=(
            MESSAGE_SEMANTIC_EDGE_POLICY_USE_SOURCE_GRAPH
        ),
    )

    assert (
        output.semantic_edge_weight
        is inputs.source_graph.semantic_edge_weight
    )
    assert (
        output.semantic_edge_factor
        is inputs.source_graph.semantic_edge_weight
    )


def test_semantic_policy_is_trimmed() -> None:
    output = _resolved(
        _inputs(),
        semantic_policy="  ignore  ",
    )

    assert output.semantic_edge_policy == (
        MESSAGE_SEMANTIC_EDGE_POLICY_IGNORE
    )


def test_factor_mapping_is_read_only_and_identity_preserving() -> None:
    output = _resolved(
        _inputs()
    )
    mapping = output.factor_mapping

    assert isinstance(
        mapping,
        MappingProxyType,
    )
    assert tuple(mapping) == (
        MESSAGE_FACTOR_ORDER
    )
    assert mapping[
        MESSAGE_FACTOR_STRUCTURAL_NORMALIZATION
    ] is output.structural_normalization_factor
    assert mapping[
        MESSAGE_FACTOR_RELATION_GATE
    ] is output.relation_gate_factor
    assert mapping[
        MESSAGE_FACTOR_EDGE_ATTENTION
    ] is output.edge_attention_factor
    assert mapping[
        MESSAGE_FACTOR_SEMANTIC_EDGE_WEIGHT
    ] is output.semantic_edge_factor

    with pytest.raises(
        TypeError,
    ):
        mapping[
            MESSAGE_FACTOR_RELATION_GATE
        ] = torch.ones(EDGES)  # type: ignore[index]


def test_active_and_disabled_factor_names() -> None:
    full = _resolved(
        _inputs()
    )
    reduced = _resolved(
        _inputs(),
        relation_gate=None,
        edge_attention=None,
        semantic_policy=(
            MESSAGE_SEMANTIC_EDGE_POLICY_IGNORE
        ),
    )

    assert full.active_factor_names == (
        MESSAGE_FACTOR_ORDER
    )
    assert full.disabled_factor_names == ()

    assert reduced.active_factor_names == (
        MESSAGE_FACTOR_STRUCTURAL_NORMALIZATION,
    )
    assert reduced.disabled_factor_names == (
        MESSAGE_FACTOR_RELATION_GATE,
        MESSAGE_FACTOR_EDGE_ATTENTION,
        MESSAGE_FACTOR_SEMANTIC_EDGE_WEIGHT,
    )


def test_resolved_coefficients_are_frozen() -> None:
    output = _resolved(
        _inputs()
    )

    with pytest.raises(
        FrozenInstanceError,
    ):
        output.semantic_edge_policy = "changed"  # type: ignore[misc]


# =============================================================================
# Resolved coefficient metadata and fingerprints
# =============================================================================


def test_resolved_architecture_metadata() -> None:
    output = _resolved(
        _inputs()
    )
    metadata = output.architecture_dict()

    assert metadata["schema_version"] == (
        RESOLVED_MESSAGE_COEFFICIENTS_SCHEMA_VERSION
    )
    assert metadata["module_contract"] == (
        "ResolvedMessageCoefficients"
    )
    assert metadata["factor_order"] == list(
        MESSAGE_FACTOR_ORDER
    )
    assert metadata[
        "combined_coefficient_formula"
    ] == MESSAGE_COMBINED_COEFFICIENT_FORMULA
    assert metadata["disabled_factor_policy"] == (
        MESSAGE_DISABLED_FACTOR_POLICY
    )
    assert metadata["semantic_edge_policy"] == (
        MESSAGE_SEMANTIC_EDGE_POLICY_USE_SOURCE_GRAPH
    )
    assert metadata["relation_gate_enabled"] is True
    assert metadata["edge_attention_enabled"] is True
    assert (
        metadata["semantic_edge_weight_enabled"]
        is True
    )
    assert metadata["parameter_free"] is True


def test_resolved_architecture_fingerprint_is_deterministic() -> None:
    inputs = _inputs()
    first = _resolved(inputs)
    second = _resolved(inputs)

    assert (
        first.architecture_fingerprint()
        == second.architecture_fingerprint()
    )


def test_resolved_architecture_fingerprint_changes_with_policy() -> None:
    inputs = _inputs()
    ignored = _resolved(
        inputs,
        semantic_policy=(
            MESSAGE_SEMANTIC_EDGE_POLICY_IGNORE
        ),
    )
    consumed = _resolved(
        inputs,
        semantic_policy=(
            MESSAGE_SEMANTIC_EDGE_POLICY_USE_SOURCE_GRAPH
        ),
    )

    assert (
        ignored.architecture_fingerprint()
        != consumed.architecture_fingerprint()
    )


def test_resolved_architecture_fingerprint_changes_with_enabled_factors() -> None:
    inputs = _inputs()
    full = _resolved(inputs)
    disabled = _resolved(
        inputs,
        relation_gate=None,
        edge_attention=None,
    )

    assert (
        full.architecture_fingerprint()
        != disabled.architecture_fingerprint()
    )


def test_resolved_lineage_metadata_contains_source_provenance() -> None:
    output = _resolved(
        _inputs()
    )
    lineage = output.lineage_dict()

    assert lineage[
        "source_inputs_lineage_fingerprint"
    ] == output.source_inputs.lineage_fingerprint()
    assert lineage[
        "edge_normalization_architecture_fingerprint"
    ] == (
        output
        .edge_normalization
        .encoder_architecture_fingerprint
    )
    assert lineage[
        "relation_gate_architecture_fingerprint"
    ] == (
        output
        .relation_gate
        .encoder_architecture_fingerprint
    )
    assert lineage[
        "relation_gate_parameter_fingerprint"
    ] == (
        output
        .relation_gate
        .parameter_fingerprint
    )
    assert lineage[
        "edge_attention_architecture_fingerprint"
    ] == (
        output
        .edge_attention
        .encoder_architecture_fingerprint
    )


def test_resolved_lineage_fingerprint_is_deterministic() -> None:
    inputs = _inputs()
    first = _resolved(inputs)
    second = _resolved(inputs)

    assert (
        first.lineage_fingerprint()
        == second.lineage_fingerprint()
    )


def test_resolved_lineage_fingerprint_changes_with_source_lineage() -> None:
    first = _resolved(
        _inputs(
            source_fingerprint="source-a",
        )
    )
    second = _resolved(
        _inputs(
            source_fingerprint="source-b",
        )
    )

    assert (
        first.lineage_fingerprint()
        != second.lineage_fingerprint()
    )


def test_resolved_value_fingerprint_is_deterministic() -> None:
    inputs = _inputs()
    first = _resolved(inputs)
    second = _resolved(inputs)

    assert (
        first.value_fingerprint()
        == second.value_fingerprint()
    )


def test_resolved_value_fingerprint_changes_with_values() -> None:
    inputs = _inputs()
    first = _resolved(inputs)
    changed_coefficients = (
        first
        .edge_normalization
        .coefficients
        .clone()
    )
    changed_coefficients[0] += 0.1
    second_normalization = _edge_normalization(
        inputs,
        coefficients=changed_coefficients,
    )
    second = _resolved(
        inputs,
        edge_normalization=second_normalization,
    )

    assert (
        first.value_fingerprint()
        != second.value_fingerprint()
    )


# =============================================================================
# Empty-edge, dtype, device, and autograd contracts
# =============================================================================


@pytest.mark.parametrize(
    "semantic_policy",
    (
        MESSAGE_SEMANTIC_EDGE_POLICY_IGNORE,
        MESSAGE_SEMANTIC_EDGE_POLICY_USE_SOURCE_GRAPH,
    ),
)
def test_resolved_coefficients_support_empty_edges(
    semantic_policy: str,
) -> None:
    inputs = _inputs(
        empty_edges=True,
    )
    output = _resolved(
        inputs,
        semantic_policy=semantic_policy,
    )

    assert output.num_edges == 0
    assert output.combined_coefficient.shape == (
        0,
    )
    assert output.factor_mapping[
        MESSAGE_FACTOR_STRUCTURAL_NORMALIZATION
    ].shape == (0,)
    assert output.value_fingerprint()


def test_resolved_coefficients_support_float64() -> None:
    inputs = _inputs(
        dtype=torch.float64,
    )
    output = _resolved(inputs)

    assert output.dtype == torch.float64
    assert all(
        tensor.dtype == torch.float64
        for tensor in output.factor_mapping.values()
    )
    assert (
        output.combined_coefficient.dtype
        == torch.float64
    )


def test_schema_preserves_autograd_connectivity() -> None:
    inputs = _inputs(
        semantic_edge_weight=None,
    )
    transformed = torch.randn(
        (EDGES, HIDDEN_DIM),
        requires_grad=True,
    )
    structural = torch.linspace(
        0.4,
        1.0,
        EDGES,
        requires_grad=True,
    )
    transform = _relation_transform(
        inputs,
        tensor=transformed,
    )
    normalization = _edge_normalization(
        inputs,
        coefficients=structural,
    )
    resolved = _resolved(
        inputs,
        edge_normalization=normalization,
        relation_gate=None,
        edge_attention=None,
        semantic_policy=(
            MESSAGE_SEMANTIC_EDGE_POLICY_IGNORE
        ),
    )
    composition = _composition(
        inputs,
        relation_transform=transform,
        resolved=resolved,
    )

    composition.edge_messages.sum().backward()

    assert transformed.grad is not None
    assert structural.grad is not None
    assert torch.isfinite(
        transformed.grad
    ).all()
    assert torch.isfinite(
        structural.grad
    ).all()


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is not available.",
)
def test_resolved_and_composition_support_cuda() -> None:
    inputs = _inputs(
        device="cuda",
    )
    resolved = _resolved(inputs)
    composition = _composition(
        inputs,
        resolved=resolved,
    )

    assert resolved.device.type == "cuda"
    assert composition.device.type == "cuda"
    assert (
        composition.edge_messages.device.type
        == "cuda"
    )


# =============================================================================
# Invalid semantic-edge policy and source contracts
# =============================================================================


@pytest.mark.parametrize(
    "policy",
    (
        "",
        " ",
        "unknown",
        "source",
        "semantic_weight",
    ),
)
def test_unknown_or_blank_semantic_policy_is_rejected(
    policy: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="semantic-edge policy|non-empty",
    ):
        _resolved(
            _inputs(),
            semantic_policy=policy,
        )


def test_nonstring_semantic_policy_is_rejected() -> None:
    with pytest.raises(
        TypeError,
        match="semantic_edge_policy",
    ):
        _resolved(
            _inputs(),
            semantic_policy=1,  # type: ignore[arg-type]
        )


def test_use_source_graph_requires_semantic_tensor() -> None:
    inputs = _inputs(
        semantic_edge_weight=None,
    )

    with pytest.raises(
        ValueError,
        match="requires source_graph.semantic_edge_weight",
    ):
        _resolved(
            inputs,
            semantic_policy=(
                MESSAGE_SEMANTIC_EDGE_POLICY_USE_SOURCE_GRAPH
            ),
        )


def test_ignore_policy_rejects_semantic_source_object() -> None:
    inputs = _inputs()

    with pytest.raises(
        ValueError,
        match="must be None",
    ):
        _resolved(
            inputs,
            semantic_policy=(
                MESSAGE_SEMANTIC_EDGE_POLICY_IGNORE
            ),
            semantic_edge_weight=(
                inputs
                .source_graph
                .semantic_edge_weight
            ),
        )


def test_ignore_policy_requires_exact_one_factor() -> None:
    inputs = _inputs()

    with pytest.raises(
        ValueError,
        match="exact multiplicative identity",
    ):
        _resolved(
            inputs,
            semantic_policy=(
                MESSAGE_SEMANTIC_EDGE_POLICY_IGNORE
            ),
            semantic_edge_factor=torch.full(
                (EDGES,),
                0.5,
            ),
        )


def test_use_source_graph_requires_exact_source_object() -> None:
    inputs = _inputs()
    clone = (
        inputs
        .source_graph
        .semantic_edge_weight
        .clone()
    )

    with pytest.raises(
        ValueError,
        match="exact source_graph.semantic_edge_weight",
    ):
        _resolved(
            inputs,
            semantic_policy=(
                MESSAGE_SEMANTIC_EDGE_POLICY_USE_SOURCE_GRAPH
            ),
            semantic_edge_weight=clone,
            semantic_edge_factor=clone,
        )


def test_enabled_semantic_factor_requires_exact_semantic_object() -> None:
    inputs = _inputs()
    semantic = (
        inputs
        .source_graph
        .semantic_edge_weight
    )

    with pytest.raises(
        ValueError,
        match="exact semantic_edge_weight",
    ):
        _resolved(
            inputs,
            semantic_policy=(
                MESSAGE_SEMANTIC_EDGE_POLICY_USE_SOURCE_GRAPH
            ),
            semantic_edge_weight=semantic,
            semantic_edge_factor=semantic.clone(),
        )


# =============================================================================
# Invalid source-output types and lineage
# =============================================================================


def test_wrong_source_inputs_type_is_rejected() -> None:
    inputs = _inputs()
    normalization = _edge_normalization(
        inputs
    )

    with pytest.raises(
        TypeError,
        match="FunctionalMessagePassingInputs",
    ):
        replace(
            _resolved(inputs),
            source_inputs=object(),  # type: ignore[arg-type]
            edge_normalization=normalization,
        )


def test_wrong_edge_normalization_type_is_rejected() -> None:
    output = _resolved(
        _inputs()
    )

    with pytest.raises(
        TypeError,
        match="StructuralEdgeNormalizationOutput",
    ):
        replace(
            output,
            edge_normalization=object(),  # type: ignore[arg-type]
        )


def test_wrong_relation_gate_type_is_rejected() -> None:
    output = _resolved(
        _inputs()
    )

    with pytest.raises(
        TypeError,
        match="RelationGateOutput or None",
    ):
        replace(
            output,
            relation_gate=object(),  # type: ignore[arg-type]
        )


def test_wrong_edge_attention_type_is_rejected() -> None:
    output = _resolved(
        _inputs()
    )

    with pytest.raises(
        TypeError,
        match="EdgeAttentionOutput or None",
    ):
        replace(
            output,
            edge_attention=object(),  # type: ignore[arg-type]
        )


def test_edge_normalization_from_other_inputs_is_rejected() -> None:
    first_inputs = _inputs(
        source_fingerprint="first",
    )
    second_inputs = _inputs(
        source_fingerprint="second",
    )
    output = _resolved(
        first_inputs
    )
    other_normalization = (
        _edge_normalization(
            second_inputs
        )
    )

    with pytest.raises(
        ValueError,
        match="exact supplied source_inputs",
    ):
        replace(
            output,
            edge_normalization=other_normalization,
        )


def test_relation_gate_from_other_inputs_is_rejected() -> None:
    first_inputs = _inputs(
        source_fingerprint="first",
    )
    second_inputs = _inputs(
        source_fingerprint="second",
    )
    output = _resolved(
        first_inputs
    )
    other_gate = _relation_gate(
        second_inputs
    )

    with pytest.raises(
        ValueError,
        match="exact same FunctionalMessagePassingInputs",
    ):
        replace(
            output,
            relation_gate=other_gate,
            relation_gate_factor=(
                other_gate.edge_gate_values
            ),
        )


def test_edge_attention_from_other_inputs_is_rejected() -> None:
    first_inputs = _inputs(
        source_fingerprint="first",
    )
    second_inputs = _inputs(
        source_fingerprint="second",
    )
    output = _resolved(
        first_inputs
    )
    other_attention = _edge_attention(
        second_inputs
    )

    with pytest.raises(
        ValueError,
        match="exact same FunctionalMessagePassingInputs",
    ):
        replace(
            output,
            edge_attention=other_attention,
            edge_attention_factor=(
                other_attention.edge_weights
            ),
        )


# =============================================================================
# Invalid factor tensor contracts
# =============================================================================


@pytest.mark.parametrize(
    "field",
    (
        "structural_normalization_factor",
        "relation_gate_factor",
        "edge_attention_factor",
        "semantic_edge_factor",
        "combined_coefficient",
    ),
)
def test_factor_rank_is_validated(
    field: str,
) -> None:
    output = _resolved(
        _inputs()
    )
    invalid = torch.ones(
        (EDGES, 1),
        dtype=output.dtype,
    )

    with pytest.raises(
        ValueError,
        match="rank 1",
    ):
        replace(
            output,
            **{field: invalid},
        )


@pytest.mark.parametrize(
    "field",
    (
        "structural_normalization_factor",
        "relation_gate_factor",
        "edge_attention_factor",
        "semantic_edge_factor",
        "combined_coefficient",
    ),
)
def test_factor_shape_is_validated(
    field: str,
) -> None:
    output = _resolved(
        _inputs()
    )
    invalid = torch.ones(
        EDGES + 1,
        dtype=output.dtype,
    )

    with pytest.raises(
        ValueError,
        match="shape",
    ):
        replace(
            output,
            **{field: invalid},
        )


@pytest.mark.parametrize(
    "field",
    (
        "structural_normalization_factor",
        "relation_gate_factor",
        "edge_attention_factor",
        "semantic_edge_factor",
        "combined_coefficient",
    ),
)
def test_factor_requires_floating_dtype(
    field: str,
) -> None:
    output = _resolved(
        _inputs()
    )
    invalid = torch.ones(
        EDGES,
        dtype=torch.long,
    )

    with pytest.raises(
        ValueError,
        match="floating-point",
    ):
        replace(
            output,
            **{field: invalid},
        )


@pytest.mark.parametrize(
    "field",
    (
        "structural_normalization_factor",
        "relation_gate_factor",
        "edge_attention_factor",
        "semantic_edge_factor",
        "combined_coefficient",
    ),
)
def test_factor_requires_finite_values(
    field: str,
) -> None:
    output = _resolved(
        _inputs()
    )
    invalid = getattr(
        output,
        field,
    ).clone()
    invalid[0] = float("nan")

    with pytest.raises(
        ValueError,
        match="finite",
    ):
        replace(
            output,
            **{field: invalid},
        )


def test_structural_factor_requires_exact_normalization_tensor() -> None:
    output = _resolved(
        _inputs()
    )

    with pytest.raises(
        ValueError,
        match="exact edge_normalization.coefficients",
    ):
        replace(
            output,
            structural_normalization_factor=(
                output
                .structural_normalization_factor
                .clone()
            ),
        )


def test_enabled_gate_factor_requires_exact_gate_tensor() -> None:
    output = _resolved(
        _inputs()
    )

    with pytest.raises(
        ValueError,
        match="exact relation_gate.edge_gate_values",
    ):
        replace(
            output,
            relation_gate_factor=(
                output
                .relation_gate_factor
                .clone()
            ),
        )


def test_enabled_attention_factor_requires_exact_attention_tensor() -> None:
    output = _resolved(
        _inputs()
    )

    with pytest.raises(
        ValueError,
        match="exact edge_attention.edge_weights",
    ):
        replace(
            output,
            edge_attention_factor=(
                output
                .edge_attention_factor
                .clone()
            ),
        )


def test_disabled_gate_requires_exact_ones() -> None:
    inputs = _inputs()
    valid = _resolved(
        inputs,
        relation_gate=None,
    )

    with pytest.raises(
        ValueError,
        match="exact multiplicative identity",
    ):
        replace(
            valid,
            relation_gate_factor=torch.full(
                (EDGES,),
                0.9,
            ),
        )


def test_disabled_attention_requires_exact_ones() -> None:
    inputs = _inputs()
    valid = _resolved(
        inputs,
        edge_attention=None,
    )

    with pytest.raises(
        ValueError,
        match="exact multiplicative identity",
    ):
        replace(
            valid,
            edge_attention_factor=torch.full(
                (EDGES,),
                0.9,
            ),
        )


@pytest.mark.parametrize(
    "field",
    (
        "structural_normalization_factor",
        "relation_gate_factor",
        "edge_attention_factor",
    ),
)
def test_nonnegative_factor_contract(
    field: str,
) -> None:
    inputs = _inputs()
    resolved_kwargs: dict[str, Any] = {}

    if field == "structural_normalization_factor":
        normalization = _edge_normalization(
            inputs
        )
        with torch.no_grad():
            normalization.coefficients[0] = -0.1
        resolved_kwargs[
            "edge_normalization"
        ] = normalization
    elif field == "relation_gate_factor":
        gate = _relation_gate(
            inputs
        )
        with torch.no_grad():
            gate.edge_gate_values[0] = -0.1
        resolved_kwargs[
            "relation_gate"
        ] = gate
    else:
        attention = _edge_attention(
            inputs
        )
        with torch.no_grad():
            attention.edge_weights[0] = -0.1
        resolved_kwargs[
            "edge_attention"
        ] = attention

    with pytest.raises(
        ValueError,
        match="nonnegative",
    ):
        _resolved(
            inputs,
            **resolved_kwargs,
        )


def test_semantic_factor_may_be_signed_when_source_contract_is_signed() -> None:
    signed = torch.linspace(
        -1.0,
        1.0,
        EDGES,
    )
    inputs = _inputs(
        semantic_edge_weight=signed,
    )
    output = _resolved(
        inputs
    )

    assert output.semantic_edge_factor is signed
    assert bool(
        (output.semantic_edge_factor < 0)
        .any()
        .item()
    )


def test_combined_coefficient_must_match_factor_product() -> None:
    output = _resolved(
        _inputs()
    )
    invalid = (
        output.combined_coefficient
        .clone()
    )
    invalid[0] += 0.2

    with pytest.raises(
        ValueError,
        match="explicit product",
    ):
        replace(
            output,
            combined_coefficient=invalid,
        )


def test_blank_resolver_fingerprint_is_rejected() -> None:
    output = _resolved(
        _inputs()
    )

    with pytest.raises(
        ValueError,
        match="non-empty",
    ):
        replace(
            output,
            resolver_architecture_fingerprint=" ",
        )


def test_blank_resolved_schema_version_is_rejected() -> None:
    output = _resolved(
        _inputs()
    )

    with pytest.raises(
        ValueError,
        match="non-empty",
    ):
        replace(
            output,
            schema_version="",
        )


def test_semantic_tensor_shape_is_validated() -> None:
    inputs = _inputs()
    output = _resolved(inputs)

    with pytest.raises(
        ValueError,
        match="shape",
    ):
        replace(
            output,
            semantic_edge_weight=torch.ones(
                EDGES + 1,
            ),
            semantic_edge_factor=torch.ones(
                EDGES + 1,
            ),
        )


def test_factor_dtype_mismatch_is_rejected() -> None:
    output = _resolved(
        _inputs()
    )
    invalid = (
        output
        .combined_coefficient
        .to(torch.float64)
    )

    with pytest.raises(
        ValueError,
        match="share one dtype",
    ):
        replace(
            output,
            combined_coefficient=invalid,
        )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is not available.",
)
def test_factor_device_mismatch_is_rejected() -> None:
    output = _resolved(
        _inputs()
    )
    invalid = (
        output
        .combined_coefficient
        .to("cuda")
    )

    with pytest.raises(
        ValueError,
        match="share one device",
    ):
        replace(
            output,
            combined_coefficient=invalid,
        )


# =============================================================================
# Valid message-composition output
# =============================================================================


def test_message_composition_full_contract() -> None:
    inputs = _inputs()
    resolved = _resolved(inputs)
    transform = _relation_transform(
        inputs
    )
    output = _composition(
        inputs,
        relation_transform=transform,
        resolved=resolved,
    )

    assert output.source_inputs is inputs
    assert output.relation_transform is transform
    assert output.resolved_coefficients is resolved
    assert output.num_edges == EDGES
    assert output.hidden_dim == HIDDEN_DIM
    assert output.dtype == inputs.dtype
    assert output.device == inputs.device
    assert (
        output.combined_coefficient
        is resolved.combined_coefficient
    )
    assert output.parameter_fingerprint is None


def test_composition_equation_is_exact_public_contract() -> None:
    inputs = _inputs()
    output = _composition(inputs)
    expected = (
        output
        .relation_transform
        .transformed_source_state
        * output
        .combined_coefficient
        .unsqueeze(-1)
    )

    torch.testing.assert_close(
        output.edge_messages,
        expected,
    )


def test_composition_supports_empty_edges() -> None:
    inputs = _inputs(
        empty_edges=True,
    )
    output = _composition(inputs)

    assert output.edge_messages.shape == (
        0,
        HIDDEN_DIM,
    )
    assert output.num_edges == 0
    assert output.hidden_dim == HIDDEN_DIM


def test_composition_supports_float64() -> None:
    inputs = _inputs(
        dtype=torch.float64,
    )
    output = _composition(inputs)

    assert output.dtype == torch.float64
    assert (
        output.edge_messages.dtype
        == torch.float64
    )


def test_composition_is_frozen() -> None:
    output = _composition(
        _inputs()
    )

    with pytest.raises(
        FrozenInstanceError,
    ):
        output.edge_messages = torch.empty(0)  # type: ignore[misc]


def test_composition_architecture_metadata() -> None:
    output = _composition(
        _inputs()
    )
    metadata = output.architecture_dict()

    assert metadata["schema_version"] == (
        MESSAGE_COMPOSITION_OUTPUT_SCHEMA_VERSION
    )
    assert metadata["module_contract"] == (
        "MessageCompositionOutput"
    )
    assert metadata["transform_input_layout"] == (
        MESSAGE_TRANSFORM_INPUT_LAYOUT
    )
    assert metadata["composition_formula"] == (
        MESSAGE_COMPOSITION_FORMULA
    )
    assert metadata["factor_order"] == list(
        MESSAGE_FACTOR_ORDER
    )
    assert metadata["parameter_free"] is True
    assert metadata["aggregation_owned_here"] is False
    assert (
        metadata["residual_update_owned_here"]
        is False
    )
    assert (
        metadata["layer_normalization_owned_here"]
        is False
    )


def test_composition_architecture_fingerprint_is_deterministic() -> None:
    inputs = _inputs()
    first = _composition(inputs)
    second = _composition(inputs)

    assert (
        first.architecture_fingerprint()
        == second.architecture_fingerprint()
    )


def test_composition_lineage_fingerprint_is_deterministic() -> None:
    inputs = _inputs()
    first = _composition(inputs)
    second = _composition(inputs)

    assert (
        first.lineage_fingerprint()
        == second.lineage_fingerprint()
    )


def test_composition_lineage_changes_with_transform_parameter_provenance() -> None:
    inputs = _inputs()
    first = _composition(
        inputs,
        relation_transform=(
            _relation_transform(
                inputs,
                parameter="parameters-a",
            )
        ),
    )
    second = _composition(
        inputs,
        relation_transform=(
            _relation_transform(
                inputs,
                parameter="parameters-b",
            )
        ),
    )

    assert (
        first.lineage_fingerprint()
        != second.lineage_fingerprint()
    )


def test_composition_value_fingerprint_is_deterministic() -> None:
    inputs = _inputs()
    first = _composition(inputs)
    second = _composition(inputs)

    assert (
        first.value_fingerprint()
        == second.value_fingerprint()
    )


def test_composition_value_fingerprint_changes_with_messages() -> None:
    inputs = _inputs()
    first = _composition(inputs)
    changed_transform = _relation_transform(
        inputs,
        tensor=(
            first
            .relation_transform
            .transformed_source_state
            + 0.25
        ),
    )
    second = _composition(
        inputs,
        relation_transform=changed_transform,
        resolved=first.resolved_coefficients,
    )

    assert (
        first.value_fingerprint()
        != second.value_fingerprint()
    )


# =============================================================================
# Invalid message-composition contracts
# =============================================================================


def test_wrong_relation_transform_type_is_rejected() -> None:
    output = _composition(
        _inputs()
    )

    with pytest.raises(
        TypeError,
        match="RelationTransformOutput",
    ):
        replace(
            output,
            relation_transform=object(),  # type: ignore[arg-type]
        )


def test_wrong_resolved_coefficients_type_is_rejected() -> None:
    output = _composition(
        _inputs()
    )

    with pytest.raises(
        TypeError,
        match="ResolvedMessageCoefficients",
    ):
        replace(
            output,
            resolved_coefficients=object(),  # type: ignore[arg-type]
        )


def test_composition_rejects_crossed_input_lineage() -> None:
    first_inputs = _inputs(
        source_fingerprint="first",
    )
    second_inputs = _inputs(
        source_fingerprint="second",
    )
    transform = _relation_transform(
        first_inputs
    )
    resolved = _resolved(
        second_inputs
    )

    with pytest.raises(
        ValueError,
        match="exact same FunctionalMessagePassingInputs",
    ):
        _composition(
            first_inputs,
            relation_transform=transform,
            resolved=resolved,
        )


def test_edge_message_rank_is_validated() -> None:
    output = _composition(
        _inputs()
    )

    with pytest.raises(
        ValueError,
        match="rank 2",
    ):
        replace(
            output,
            edge_messages=torch.ones(
                EDGES,
            ),
        )


def test_edge_message_shape_is_validated() -> None:
    output = _composition(
        _inputs()
    )

    with pytest.raises(
        ValueError,
        match="shape",
    ):
        replace(
            output,
            edge_messages=torch.ones(
                EDGES,
                HIDDEN_DIM + 1,
            ),
        )


def test_edge_message_requires_floating_dtype() -> None:
    output = _composition(
        _inputs()
    )

    with pytest.raises(
        ValueError,
        match="floating-point",
    ):
        replace(
            output,
            edge_messages=torch.ones(
                (EDGES, HIDDEN_DIM),
                dtype=torch.long,
            ),
        )


def test_edge_message_requires_finite_values() -> None:
    output = _composition(
        _inputs()
    )
    invalid = output.edge_messages.clone()
    invalid[0, 0] = float("inf")

    with pytest.raises(
        ValueError,
        match="finite",
    ):
        replace(
            output,
            edge_messages=invalid,
        )


def test_edge_message_dtype_mismatch_is_rejected() -> None:
    output = _composition(
        _inputs()
    )

    with pytest.raises(
        ValueError,
        match="share one dtype",
    ):
        replace(
            output,
            edge_messages=(
                output
                .edge_messages
                .to(torch.float64)
            ),
        )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is not available.",
)
def test_edge_message_device_mismatch_is_rejected() -> None:
    output = _composition(
        _inputs()
    )

    with pytest.raises(
        ValueError,
        match="share one device",
    ):
        replace(
            output,
            edge_messages=(
                output
                .edge_messages
                .to("cuda")
            ),
        )


def test_edge_message_equation_is_validated() -> None:
    output = _composition(
        _inputs()
    )
    invalid = output.edge_messages.clone()
    invalid[0, 0] += 0.2

    with pytest.raises(
        ValueError,
        match="do not equal",
    ):
        replace(
            output,
            edge_messages=invalid,
        )


def test_blank_composer_fingerprint_is_rejected() -> None:
    output = _composition(
        _inputs()
    )

    with pytest.raises(
        ValueError,
        match="non-empty",
    ):
        replace(
            output,
            composer_architecture_fingerprint="",
        )


def test_blank_composition_schema_version_is_rejected() -> None:
    output = _composition(
        _inputs()
    )

    with pytest.raises(
        ValueError,
        match="non-empty",
    ):
        replace(
            output,
            schema_version=" ",
        )


# =============================================================================
# Exact internal stage-chain validation
# =============================================================================


def test_valid_message_builder_stage_chain() -> None:
    inputs = _inputs()
    resolved = _resolved(inputs)
    composition = _composition(
        inputs,
        resolved=resolved,
    )

    validate_message_builder_stage_chain(
        resolved_coefficients=resolved,
        composition_output=composition,
    )


def test_stage_chain_rejects_wrong_resolved_type() -> None:
    composition = _composition(
        _inputs()
    )

    with pytest.raises(
        TypeError,
        match="ResolvedMessageCoefficients",
    ):
        validate_message_builder_stage_chain(
            resolved_coefficients=object(),  # type: ignore[arg-type]
            composition_output=composition,
        )


def test_stage_chain_rejects_wrong_composition_type() -> None:
    resolved = _resolved(
        _inputs()
    )

    with pytest.raises(
        TypeError,
        match="MessageCompositionOutput",
    ):
        validate_message_builder_stage_chain(
            resolved_coefficients=resolved,
            composition_output=object(),  # type: ignore[arg-type]
        )


def test_stage_chain_rejects_equal_but_distinct_resolved_object() -> None:
    inputs = _inputs()
    first = _resolved(inputs)
    second = _resolved(inputs)
    composition = _composition(
        inputs,
        resolved=second,
    )

    with pytest.raises(
        ValueError,
        match="exact supplied resolved_coefficients",
    ):
        validate_message_builder_stage_chain(
            resolved_coefficients=first,
            composition_output=composition,
        )


# =============================================================================
# Public EdgeMessageOutput compatibility validation
# =============================================================================


def test_valid_public_edge_message_output() -> None:
    composition = _composition(
        _inputs()
    )
    public = _public_output(
        composition
    )

    validate_public_edge_message_output(
        public_output=public,
        composition_output=composition,
    )


def test_public_validator_rejects_wrong_public_type() -> None:
    composition = _composition(
        _inputs()
    )

    with pytest.raises(
        TypeError,
        match="EdgeMessageOutput",
    ):
        validate_public_edge_message_output(
            public_output=object(),  # type: ignore[arg-type]
            composition_output=composition,
        )


def test_public_validator_rejects_wrong_composition_type() -> None:
    public = _public_output(
        _composition(
            _inputs()
        )
    )

    with pytest.raises(
        TypeError,
        match="MessageCompositionOutput",
    ):
        validate_public_edge_message_output(
            public_output=public,
            composition_output=object(),  # type: ignore[arg-type]
        )


def test_public_validator_requires_exact_relation_transform() -> None:
    inputs = _inputs()
    composition = _composition(inputs)
    other_transform = _relation_transform(
        inputs,
        tensor=(
            composition
            .relation_transform
            .transformed_source_state
            .clone()
        ),
    )
    other_messages = (
        other_transform.transformed_source_state
        * composition
        .combined_coefficient
        .unsqueeze(-1)
    )
    public = _public_output(
        composition,
        relation_transform=other_transform,
        edge_messages=other_messages,
    )

    with pytest.raises(
        ValueError,
        match="exact relation_transform",
    ):
        validate_public_edge_message_output(
            public_output=public,
            composition_output=composition,
        )


def test_public_validator_requires_exact_edge_normalization() -> None:
    inputs = _inputs()
    composition = _composition(inputs)
    resolved = composition.resolved_coefficients
    other_normalization = (
        _edge_normalization(
            inputs,
            coefficients=(
                resolved
                .edge_normalization
                .coefficients
                .clone()
            ),
        )
    )
    public = EdgeMessageOutput(
        edge_messages=composition.edge_messages,
        relation_transform=(
            composition.relation_transform
        ),
        edge_normalization=other_normalization,
        relation_gate=resolved.relation_gate,
        edge_attention=resolved.edge_attention,
        semantic_edge_weight=(
            resolved.semantic_edge_weight
        ),
        encoder_architecture_fingerprint=(
            "other-public-output"
        ),
    )

    with pytest.raises(
        ValueError,
        match="exact edge_normalization",
    ):
        validate_public_edge_message_output(
            public_output=public,
            composition_output=composition,
        )


def test_public_validator_requires_exact_relation_gate() -> None:
    inputs = _inputs()
    composition = _composition(inputs)
    resolved = composition.resolved_coefficients
    other_gate = _relation_gate(
        inputs
    )
    other_messages = (
        composition
        .relation_transform
        .transformed_source_state
        * resolved
        .edge_normalization
        .coefficients
        .unsqueeze(-1)
        * other_gate
        .edge_gate_values
        .unsqueeze(-1)
        * resolved
        .edge_attention
        .edge_weights
        .unsqueeze(-1)
        * resolved
        .semantic_edge_weight
        .unsqueeze(-1)
    )
    public = EdgeMessageOutput(
        edge_messages=other_messages,
        relation_transform=(
            composition.relation_transform
        ),
        edge_normalization=(
            resolved.edge_normalization
        ),
        relation_gate=other_gate,
        edge_attention=resolved.edge_attention,
        semantic_edge_weight=(
            resolved.semantic_edge_weight
        ),
        encoder_architecture_fingerprint=(
            "other-public-output"
        ),
    )

    with pytest.raises(
        ValueError,
        match="exact relation_gate",
    ):
        validate_public_edge_message_output(
            public_output=public,
            composition_output=composition,
        )


def test_public_validator_requires_exact_edge_attention() -> None:
    inputs = _inputs()
    composition = _composition(inputs)
    resolved = composition.resolved_coefficients
    other_attention = _edge_attention(
        inputs
    )
    other_messages = (
        composition
        .relation_transform
        .transformed_source_state
        * resolved
        .edge_normalization
        .coefficients
        .unsqueeze(-1)
        * resolved
        .relation_gate
        .edge_gate_values
        .unsqueeze(-1)
        * other_attention
        .edge_weights
        .unsqueeze(-1)
        * resolved
        .semantic_edge_weight
        .unsqueeze(-1)
    )
    public = EdgeMessageOutput(
        edge_messages=other_messages,
        relation_transform=(
            composition.relation_transform
        ),
        edge_normalization=(
            resolved.edge_normalization
        ),
        relation_gate=resolved.relation_gate,
        edge_attention=other_attention,
        semantic_edge_weight=(
            resolved.semantic_edge_weight
        ),
        encoder_architecture_fingerprint=(
            "other-public-output"
        ),
    )

    with pytest.raises(
        ValueError,
        match="exact edge_attention",
    ):
        validate_public_edge_message_output(
            public_output=public,
            composition_output=composition,
        )


def test_public_validator_requires_exact_semantic_tensor() -> None:
    inputs = _inputs()
    composition = _composition(inputs)
    resolved = composition.resolved_coefficients
    semantic_clone = (
        resolved
        .semantic_edge_weight
        .clone()
    )
    other_messages = (
        composition
        .relation_transform
        .transformed_source_state
        * resolved
        .edge_normalization
        .coefficients
        .unsqueeze(-1)
        * resolved
        .relation_gate
        .edge_gate_values
        .unsqueeze(-1)
        * resolved
        .edge_attention
        .edge_weights
        .unsqueeze(-1)
        * semantic_clone
        .unsqueeze(-1)
    )
    public = EdgeMessageOutput(
        edge_messages=other_messages,
        relation_transform=(
            composition.relation_transform
        ),
        edge_normalization=(
            resolved.edge_normalization
        ),
        relation_gate=resolved.relation_gate,
        edge_attention=resolved.edge_attention,
        semantic_edge_weight=semantic_clone,
        encoder_architecture_fingerprint=(
            "other-public-output"
        ),
    )

    with pytest.raises(
        ValueError,
        match="exact semantic_edge_weight",
    ):
        validate_public_edge_message_output(
            public_output=public,
            composition_output=composition,
        )


def test_public_validator_requires_exact_edge_message_tensor() -> None:
    composition = _composition(
        _inputs()
    )
    cloned_messages = (
        composition.edge_messages.clone()
    )
    public = _public_output(
        composition,
        edge_messages=cloned_messages,
    )

    with pytest.raises(
        ValueError,
        match="exact edge_messages tensor",
    ):
        validate_public_edge_message_output(
            public_output=public,
            composition_output=composition,
        )


def test_public_output_with_all_optional_factors_disabled() -> None:
    inputs = _inputs()
    resolved = _resolved(
        inputs,
        relation_gate=None,
        edge_attention=None,
        semantic_policy=(
            MESSAGE_SEMANTIC_EDGE_POLICY_IGNORE
        ),
    )
    composition = _composition(
        inputs,
        resolved=resolved,
    )
    public = _public_output(
        composition
    )

    assert public.relation_gate is None
    assert public.edge_attention is None
    assert public.semantic_edge_weight is None

    validate_public_edge_message_output(
        public_output=public,
        composition_output=composition,
    )


# =============================================================================
# Schema separation from aggregation and explanation claims
# =============================================================================


def test_resolved_schema_contains_no_messages_or_aggregation() -> None:
    output = _resolved(
        _inputs()
    )

    assert not hasattr(
        output,
        "edge_messages",
    )
    assert not hasattr(
        output,
        "aggregated_messages",
    )
    assert not hasattr(
        output,
        "target_updates",
    )


def test_composition_schema_contains_no_aggregation_or_causal_claims() -> None:
    output = _composition(
        _inputs()
    )

    assert not hasattr(
        output,
        "aggregated_messages",
    )
    assert not hasattr(
        output,
        "target_updates",
    )
    assert not hasattr(
        output,
        "causal_importance",
    )
    assert not hasattr(
        output,
        "explanation_faithfulness",
    )
