"""
Numerical tests for functional edge-message construction.

Target repository path:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            tests/
                message_builders/
                    test_message_builder_numerics.py

Implementations under test:
    functional_message_passing/message_builders/relation_state_gather.py
    functional_message_passing/message_builders/coefficient_resolution.py
    functional_message_passing/message_builders/message_composition.py

This suite is organized by numerical and contracts rather than by
a one-test-file-per-source-file mirror.

Covered contracts
-----------------
1. Relation-state boundary

   ``RelationTransformOutput.transformed_source_state`` is already edge
   aligned ``[E, H]``. Resolution must therefore preserve exact tensor
   identity and perform no second gather, clone, cast, detach, move, or
   reorder.

2. Scalar coefficient resolution

   For every edge ``e``:

       c_e = n_e * g_e * alpha_e * w_e

   where:

   - ``n_e`` is structural normalization;
   - ``g_e`` is the exact relation-gate coefficient or one when disabled;
   - ``alpha_e`` is reduced edge attention or one when disabled;
   - ``w_e`` is the explicitly consumed semantic edge weight or one.

3. Message composition

       m_e = u_e * c_e

   where ``u_e`` is the exact edge-aligned relation-transformed source state.
   Scalar coefficients broadcast only over the final hidden-feature axis.

4. Separation of responsibilities

   No target-node aggregation, residual update, normalization, dropout,
   relation dispatch, or causal interpretation occurs in these modules.

5. Numerical robustness

   The suite checks:

   - exact hand-computed values;
   - all optional-factor combinations;
   - enabled-uniform versus disabled attention;
   - signed semantic coefficients;
   - empty edges;
   - float64 and optional CUDA;
   - autograd and analytical gradients;
   - edge-permutation equivariance;
   - zero coefficients;
   - non-finite input and overflow rejection;
   - exact source-object and tensor lineage;
   - parameter-free and buffer-free module wrappers;
   - deterministic architecture fingerprints;
   - diagnostics that remain descriptive rather than causal.

Controlled upstream doubles are patched into
``functional_message_passing.schemas`` so failures remain localized to the
three numerical message-builder stages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
import torch
from torch import nn

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
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.functional_message_passing.message_builders.coefficient_resolution import (
    MESSAGE_COEFFICIENT_RESOLUTION_BUFFER_FREE,
    MESSAGE_COEFFICIENT_RESOLUTION_INTERPRETATION,
    MESSAGE_COEFFICIENT_RESOLUTION_OPERATION_ORDER,
    MESSAGE_COEFFICIENT_RESOLUTION_PARAMETER_FREE,
    MESSAGE_COEFFICIENT_RESOLUTION_SCHEMA_VERSION,
    CoefficientResolver,
    EdgeMessageCoefficientResolver,
    MessageCoefficientResolver,
    OptionalFactorResolution,
    SemanticFactorResolution,
    build_coefficient_resolver,
    build_message_coefficient_resolver,
    coefficient_resolution_architecture_dict,
    coefficient_resolution_architecture_fingerprint,
    combine_message_coefficients,
    message_coefficient_diagnostic_summary,
    resolve_edge_attention_factor,
    resolve_edge_message_coefficients,
    resolve_message_coefficients,
    resolve_relation_gate_factor,
    resolve_semantic_edge_factor,
    resolve_structural_normalization_factor,
    validate_resolved_message_coefficients,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.functional_message_passing.message_builders.message_composition import (
    MESSAGE_COMPOSER_AGGREGATION_OWNED_HERE,
    MESSAGE_COMPOSER_BROADCAST_AXIS,
    MESSAGE_COMPOSER_BUFFER_FREE,
    MESSAGE_COMPOSER_INPUT_COEFFICIENT_LAYOUT,
    MESSAGE_COMPOSER_INPUT_STATE_LAYOUT,
    MESSAGE_COMPOSER_OPERATION,
    MESSAGE_COMPOSER_OPERATION_ORDER,
    MESSAGE_COMPOSER_OUTPUT_LAYOUT,
    MESSAGE_COMPOSER_PARAMETER_FREE,
    MESSAGE_COMPOSER_SCHEMA_VERSION,
    EdgeMessageComposer,
    FunctionalMessageComposer,
    MessageComposer,
    build_edge_message_composer,
    build_message_composer,
    compose_edge_message_tensor,
    compose_edge_messages,
    compose_message_output,
    compose_message_vectors,
    message_composer_architecture_dict,
    message_composer_architecture_fingerprint,
    message_composition_diagnostic_summary,
    validate_message_composition_output,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.functional_message_passing.message_builders.relation_state_gather import (
    RELATION_STATE_GATHER_INDEXING_OWNED_HERE,
    RELATION_STATE_GATHER_INPUT_LAYOUT,
    RELATION_STATE_GATHER_OPERATION,
    RELATION_STATE_GATHER_OPERATION_ORDER,
    RELATION_STATE_GATHER_OUTPUT_LAYOUT,
    RELATION_STATE_GATHER_OWNER,
    RELATION_STATE_GATHER_PARAMETER_FREE,
    RELATION_STATE_GATHER_SCHEMA_VERSION,
    RELATION_STATE_GATHER_ZERO_COPY_REQUIRED,
    EdgeAlignedRelationStateResolver,
    RelationStateGather,
    RelationStateResolver,
    assert_zero_copy_relation_state,
    build_relation_state_gather,
    build_relation_state_resolver,
    gather_relation_state,
    relation_state_gather_architecture_dict,
    relation_state_gather_architecture_fingerprint,
    relation_state_gather_diagnostic_summary,
    resolve_edge_aligned_relation_state,
    resolve_relation_state,
    validate_edge_aligned_relation_state,
    validate_relation_state_gather_contract,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.functional_message_passing.message_builders.schemas import (
    MESSAGE_DISABLED_FACTOR_POLICY,
    MESSAGE_FACTOR_EDGE_ATTENTION,
    MESSAGE_FACTOR_ORDER,
    MESSAGE_FACTOR_RELATION_GATE,
    MESSAGE_FACTOR_SEMANTIC_EDGE_WEIGHT,
    MESSAGE_FACTOR_STRUCTURAL_NORMALIZATION,
    MESSAGE_SEMANTIC_EDGE_POLICY_IGNORE,
    MESSAGE_SEMANTIC_EDGE_POLICY_USE_SOURCE_GRAPH,
    MESSAGE_TRANSFORM_INPUT_LAYOUT,
    MessageCompositionOutput,
    ResolvedMessageCoefficients,
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
# Additional numerical helpers
# =============================================================================


def _resolve_actual(
    inputs: FunctionalMessagePassingInputs,
    *,
    relation_gate: RelationGateOutput | None | object = _DEFAULT,
    edge_attention: EdgeAttentionOutput | None | object = _DEFAULT,
    semantic_policy: str = MESSAGE_SEMANTIC_EDGE_POLICY_IGNORE,
    edge_normalization: StructuralEdgeNormalizationOutput | None = None,
) -> ResolvedMessageCoefficients:
    normalization = (
        _edge_normalization(inputs)
        if edge_normalization is None
        else edge_normalization
    )
    gate = (
        _relation_gate(inputs)
        if relation_gate is _DEFAULT
        else relation_gate
    )
    attention = (
        _edge_attention(inputs)
        if edge_attention is _DEFAULT
        else edge_attention
    )

    return resolve_message_coefficients(
        edge_normalization=normalization,
        relation_gate=gate,  # type: ignore[arg-type]
        edge_attention=attention,  # type: ignore[arg-type]
        semantic_edge_policy=semantic_policy,
        source_inputs=inputs,
    )


def _compose_actual(
    inputs: FunctionalMessagePassingInputs,
    *,
    relation_transform: RelationTransformOutput | None = None,
    resolved_coefficients: ResolvedMessageCoefficients | None = None,
) -> MessageCompositionOutput:
    transform = (
        _relation_transform(inputs)
        if relation_transform is None
        else relation_transform
    )
    coefficients = (
        _resolve_actual(inputs)
        if resolved_coefficients is None
        else resolved_coefficients
    )

    return compose_message_output(
        relation_transform=transform,
        resolved_coefficients=coefficients,
        source_inputs=inputs,
    )


def _manual_factor_product(
    resolved: ResolvedMessageCoefficients,
) -> torch.Tensor:
    return (
        resolved.structural_normalization_factor
        * resolved.relation_gate_factor
        * resolved.edge_attention_factor
        * resolved.semantic_edge_factor
    )


def _assert_parameter_and_buffer_free(
    module: nn.Module,
) -> None:
    assert tuple(module.named_parameters()) == ()
    assert tuple(module.named_buffers()) == ()
    assert module.state_dict() == {}
    assert sum(
        parameter.numel()
        for parameter in module.parameters()
    ) == 0
    assert sum(
        buffer.numel()
        for buffer in module.buffers()
    ) == 0


# =============================================================================
# Public identity and architecture constants
# =============================================================================


def test_relation_state_gather_public_identity() -> None:
    assert RELATION_STATE_GATHER_SCHEMA_VERSION.strip()
    assert RELATION_STATE_GATHER_OPERATION == (
        "zero_copy_resolution_of_edge_aligned_relation_transform_output"
    )
    assert RELATION_STATE_GATHER_INPUT_LAYOUT == (
        MESSAGE_TRANSFORM_INPUT_LAYOUT
    )
    assert RELATION_STATE_GATHER_OUTPUT_LAYOUT == (
        "edge_aligned_transformed_source_state_[E,H]"
    )
    assert RELATION_STATE_GATHER_OWNER == (
        "relation_transform_subsystem"
    )
    assert RELATION_STATE_GATHER_INDEXING_OWNED_HERE is False
    assert RELATION_STATE_GATHER_ZERO_COPY_REQUIRED is True
    assert RELATION_STATE_GATHER_PARAMETER_FREE is True
    assert RELATION_STATE_GATHER_OPERATION_ORDER[-1] == (
        "return_exact_transformed_source_state_tensor"
    )


def test_coefficient_resolution_public_identity() -> None:
    assert MESSAGE_COEFFICIENT_RESOLUTION_SCHEMA_VERSION.strip()
    assert MESSAGE_COEFFICIENT_RESOLUTION_INTERPRETATION == (
        "explicit_multiplicative_edge_message_scaling"
    )
    assert MESSAGE_COEFFICIENT_RESOLUTION_PARAMETER_FREE is True
    assert MESSAGE_COEFFICIENT_RESOLUTION_BUFFER_FREE is True
    assert MESSAGE_COEFFICIENT_RESOLUTION_OPERATION_ORDER[-1] == (
        "construct_resolved_message_coefficients"
    )
    assert MESSAGE_DISABLED_FACTOR_POLICY == (
        "exact_multiplicative_identity_one"
    )


def test_message_composer_public_identity() -> None:
    assert MESSAGE_COMPOSER_SCHEMA_VERSION.strip()
    assert MESSAGE_COMPOSER_OPERATION == (
        "broadcast_scalar_edge_coefficient_over_hidden_features"
    )
    assert MESSAGE_COMPOSER_INPUT_STATE_LAYOUT == (
        MESSAGE_TRANSFORM_INPUT_LAYOUT
    )
    assert MESSAGE_COMPOSER_INPUT_COEFFICIENT_LAYOUT == (
        "edge_aligned_combined_coefficient_[E]"
    )
    assert MESSAGE_COMPOSER_OUTPUT_LAYOUT == (
        "edge_aligned_messages_[E,H]"
    )
    assert MESSAGE_COMPOSER_BROADCAST_AXIS == -1
    assert MESSAGE_COMPOSER_PARAMETER_FREE is True
    assert MESSAGE_COMPOSER_BUFFER_FREE is True
    assert MESSAGE_COMPOSER_AGGREGATION_OWNED_HERE is False
    assert MESSAGE_COMPOSER_OPERATION_ORDER[-1] == (
        "construct_message_composition_output"
    )


def test_compact_aliases_are_exact() -> None:
    assert RelationStateResolver is RelationStateGather
    assert EdgeAlignedRelationStateResolver is RelationStateGather
    assert CoefficientResolver is MessageCoefficientResolver
    assert EdgeMessageCoefficientResolver is MessageCoefficientResolver
    assert MessageComposer is EdgeMessageComposer
    assert FunctionalMessageComposer is EdgeMessageComposer
    assert compose_message_vectors is compose_edge_message_tensor
    assert resolve_edge_message_coefficients is resolve_message_coefficients


# =============================================================================
# Relation-state zero-copy numerical boundary
# =============================================================================


def test_relation_state_resolution_returns_exact_tensor_object() -> None:
    inputs = _inputs()
    transform = _relation_transform(inputs)

    resolved = resolve_edge_aligned_relation_state(
        transform,
        source_inputs=inputs,
    )

    assert resolved is transform.transformed_source_state
    assert resolved.data_ptr() == (
        transform.transformed_source_state.data_ptr()
    )


@pytest.mark.parametrize(
    "resolver",
    (
        resolve_edge_aligned_relation_state,
        resolve_relation_state,
        gather_relation_state,
    ),
)
def test_relation_state_functional_aliases_are_zero_copy(
    resolver: Any,
) -> None:
    inputs = _inputs()
    transform = _relation_transform(inputs)

    resolved = resolver(
        transform,
        source_inputs=inputs,
    )

    assert resolved is transform.transformed_source_state


def test_relation_state_module_is_zero_copy() -> None:
    inputs = _inputs()
    transform = _relation_transform(inputs)
    gatherer = RelationStateGather()

    resolved = gatherer(
        transform,
        source_inputs=inputs,
    )

    assert resolved is transform.transformed_source_state
    assert gatherer.resolve(
        transform,
        source_inputs=inputs,
    ) is resolved


def test_relation_state_builder_aliases() -> None:
    first = build_relation_state_gather()
    second = build_relation_state_resolver()

    assert isinstance(first, RelationStateGather)
    assert isinstance(second, RelationStateGather)
    _assert_parameter_and_buffer_free(first)
    _assert_parameter_and_buffer_free(second)


def test_relation_state_module_is_parameter_and_buffer_free() -> None:
    gatherer = RelationStateGather()

    _assert_parameter_and_buffer_free(gatherer)
    assert gatherer.parameter_count == 0
    assert gatherer.trainable_parameter_count == 0
    assert gatherer.buffer_count == 0
    assert gatherer.parameter_fingerprint is None
    gatherer.assert_parameter_free()


def test_relation_state_architecture_is_deterministic() -> None:
    first = relation_state_gather_architecture_dict()
    second = relation_state_gather_architecture_dict()

    assert first == second
    assert (
        relation_state_gather_architecture_fingerprint()
        == relation_state_gather_architecture_fingerprint()
    )
    assert first["indexing_owned_here"] is False
    assert first["zero_copy_required"] is True
    assert first["parameter_free"] is True
    assert first["message_composition_owned_here"] is False
    assert first["aggregation_owned_here"] is False


def test_relation_state_module_architecture_matches_functional_contract() -> None:
    gatherer = RelationStateGather()

    assert gatherer.architecture_dict() == (
        relation_state_gather_architecture_dict()
    )
    assert gatherer.architecture_fingerprint() == (
        relation_state_gather_architecture_fingerprint()
    )


def test_relation_state_diagnostics_are_descriptive() -> None:
    inputs = _inputs()
    transform = _relation_transform(inputs)
    summary = relation_state_gather_diagnostic_summary(
        relation_transform=transform,
        source_inputs=inputs,
    )

    assert summary["num_edges"] == EDGES
    assert summary["hidden_dim"] == HIDDEN_DIM
    assert summary["num_relations"] == RELATIONS
    assert summary["relation_names"] == list(RELATION_NAMES)
    assert summary["stable_relation_ids"] == list(
        STABLE_RELATION_IDS
    )
    assert summary["zero_copy_identity_preserved"] is True
    assert summary["indexing_performed_here"] is False
    assert summary["relation_dispatch_performed_here"] is False
    assert summary["causal_importance_claim"] is False
    assert summary["explanation_faithfulness_claim"] is False


def test_relation_state_contract_validator_accepts_exact_output() -> None:
    inputs = _inputs()
    transform = _relation_transform(inputs)
    gatherer = RelationStateGather()
    resolved = gatherer(
        transform,
        source_inputs=inputs,
    )

    validate_relation_state_gather_contract(
        gatherer=gatherer,
        relation_transform=transform,
        resolved_state=resolved,
        source_inputs=inputs,
    )


def test_zero_copy_assertion_rejects_clone() -> None:
    inputs = _inputs()
    transform = _relation_transform(inputs)

    with pytest.raises(
        ValueError,
        match="exact.*tensor object",
    ):
        assert_zero_copy_relation_state(
            relation_transform=transform,
            resolved_state=(
                transform
                .transformed_source_state
                .clone()
            ),
        )


def test_relation_state_rejects_crossed_source_input_lineage() -> None:
    first = _inputs(
        source_fingerprint="first",
    )
    second = _inputs(
        source_fingerprint="second",
    )
    transform = _relation_transform(first)

    with pytest.raises(
        ValueError,
        match="exact supplied source_inputs",
    ):
        resolve_edge_aligned_relation_state(
            transform,
            source_inputs=second,
        )


def test_relation_state_supports_empty_edges() -> None:
    inputs = _inputs(
        empty_edges=True,
    )
    transform = _relation_transform(inputs)

    resolved = resolve_edge_aligned_relation_state(
        transform,
        source_inputs=inputs,
    )

    assert resolved is transform.transformed_source_state
    assert resolved.shape == (
        0,
        HIDDEN_DIM,
    )


def test_relation_state_preserves_float64() -> None:
    inputs = _inputs(
        dtype=torch.float64,
    )
    transform = _relation_transform(inputs)

    resolved = resolve_edge_aligned_relation_state(
        transform,
        source_inputs=inputs,
    )

    assert resolved.dtype == torch.float64
    assert resolved.device == inputs.device


def test_relation_state_preserves_autograd_identity() -> None:
    inputs = _inputs()
    transformed = torch.randn(
        EDGES,
        HIDDEN_DIM,
        requires_grad=True,
    )
    transform = _relation_transform(
        inputs,
        tensor=transformed,
    )

    resolved = resolve_edge_aligned_relation_state(
        transform,
        source_inputs=inputs,
    )
    resolved.square().sum().backward()

    assert resolved is transformed
    torch.testing.assert_close(
        transformed.grad,
        2.0 * transformed.detach(),
    )


def test_relation_state_rejects_nonfinite_transform() -> None:
    inputs = _inputs()
    tensor = torch.randn(
        EDGES,
        HIDDEN_DIM,
    )
    tensor[0, 0] = float("nan")

    # The upstream schema itself is expected to reject this malformed output.
    with pytest.raises(
        (ValueError, FloatingPointError),
        match="finite",
    ):
        transform = _relation_transform(
            inputs,
            tensor=tensor,
        )
        validate_edge_aligned_relation_state(
            relation_transform=transform,
            source_inputs=inputs,
        )


# =============================================================================
# Individual coefficient-factor resolution
# =============================================================================


def test_structural_factor_preserves_exact_tensor() -> None:
    inputs = _inputs()
    normalization = _edge_normalization(inputs)

    factor = resolve_structural_normalization_factor(
        normalization,
        source_inputs=inputs,
    )

    assert factor is normalization.coefficients


def test_enabled_relation_gate_factor_preserves_exact_tensor() -> None:
    inputs = _inputs()
    structural = _edge_normalization(
        inputs
    ).coefficients
    gate = _relation_gate(inputs)

    resolution = resolve_relation_gate_factor(
        source_inputs=inputs,
        reference_factor=structural,
        relation_gate=gate,
    )

    assert isinstance(
        resolution,
        OptionalFactorResolution,
    )
    assert resolution.source is gate
    assert resolution.factor is gate.edge_gate_values


def test_disabled_relation_gate_factor_is_exact_identity() -> None:
    inputs = _inputs()
    structural = _edge_normalization(
        inputs
    ).coefficients

    resolution = resolve_relation_gate_factor(
        source_inputs=inputs,
        reference_factor=structural,
        relation_gate=None,
    )

    assert resolution.source is None
    assert torch.equal(
        resolution.factor,
        torch.ones_like(structural),
    )
    assert resolution.factor.dtype == structural.dtype
    assert resolution.factor.device == structural.device


def test_enabled_attention_factor_preserves_exact_tensor() -> None:
    inputs = _inputs()
    structural = _edge_normalization(
        inputs
    ).coefficients
    attention = _edge_attention(inputs)

    resolution = resolve_edge_attention_factor(
        source_inputs=inputs,
        reference_factor=structural,
        edge_attention=attention,
    )

    assert resolution.source is attention
    assert resolution.factor is attention.edge_weights


def test_disabled_attention_factor_is_exact_identity() -> None:
    inputs = _inputs()
    structural = _edge_normalization(
        inputs
    ).coefficients

    resolution = resolve_edge_attention_factor(
        source_inputs=inputs,
        reference_factor=structural,
        edge_attention=None,
    )

    assert resolution.source is None
    assert torch.equal(
        resolution.factor,
        torch.ones_like(structural),
    )


def test_semantic_ignore_resolves_exact_identity_without_source() -> None:
    inputs = _inputs()
    structural = _edge_normalization(
        inputs
    ).coefficients

    resolution = resolve_semantic_edge_factor(
        source_inputs=inputs,
        reference_factor=structural,
        semantic_edge_policy=(
            MESSAGE_SEMANTIC_EDGE_POLICY_IGNORE
        ),
    )

    assert isinstance(
        resolution,
        SemanticFactorResolution,
    )
    assert resolution.source_weight is None
    assert resolution.policy == (
        MESSAGE_SEMANTIC_EDGE_POLICY_IGNORE
    )
    assert torch.equal(
        resolution.factor,
        torch.ones_like(structural),
    )


def test_semantic_use_source_graph_preserves_exact_tensor() -> None:
    inputs = _inputs()
    structural = _edge_normalization(
        inputs
    ).coefficients

    resolution = resolve_semantic_edge_factor(
        source_inputs=inputs,
        reference_factor=structural,
        semantic_edge_policy=(
            MESSAGE_SEMANTIC_EDGE_POLICY_USE_SOURCE_GRAPH
        ),
    )

    assert resolution.source_weight is (
        inputs.source_graph.semantic_edge_weight
    )
    assert resolution.factor is (
        inputs.source_graph.semantic_edge_weight
    )


def test_semantic_use_source_graph_supports_signed_weights() -> None:
    signed = torch.linspace(
        -1.5,
        1.5,
        EDGES,
    )
    inputs = _inputs(
        semantic_edge_weight=signed,
    )
    structural = _edge_normalization(
        inputs
    ).coefficients

    resolution = resolve_semantic_edge_factor(
        source_inputs=inputs,
        reference_factor=structural,
        semantic_edge_policy=(
            MESSAGE_SEMANTIC_EDGE_POLICY_USE_SOURCE_GRAPH
        ),
    )

    assert resolution.factor is signed
    assert bool(
        (resolution.factor < 0)
        .any()
        .item()
    )


def test_semantic_use_source_graph_requires_tensor() -> None:
    inputs = _inputs(
        semantic_edge_weight=None,
    )
    structural = _edge_normalization(
        inputs
    ).coefficients

    with pytest.raises(
        ValueError,
        match="requires source_graph.semantic_edge_weight",
    ):
        resolve_semantic_edge_factor(
            source_inputs=inputs,
            reference_factor=structural,
            semantic_edge_policy=(
                MESSAGE_SEMANTIC_EDGE_POLICY_USE_SOURCE_GRAPH
            ),
        )


def test_enabled_uniform_attention_differs_from_disabled_attention() -> None:
    inputs = _inputs()
    structural = _edge_normalization(
        inputs
    ).coefficients
    enabled = resolve_edge_attention_factor(
        source_inputs=inputs,
        reference_factor=structural,
        edge_attention=_edge_attention(inputs),
    )
    disabled = resolve_edge_attention_factor(
        source_inputs=inputs,
        reference_factor=structural,
        edge_attention=None,
    )

    assert not torch.equal(
        enabled.factor,
        disabled.factor,
    )
    assert torch.equal(
        disabled.factor,
        torch.ones_like(disabled.factor),
    )


# =============================================================================
# Coefficient multiplication kernel
# =============================================================================


def test_combine_message_coefficients_matches_hand_calculation() -> None:
    structural = torch.tensor(
        [0.5, 1.0, 2.0],
    )
    gate = torch.tensor(
        [0.2, 0.4, 0.5],
    )
    attention = torch.tensor(
        [0.5, 0.25, 1.0],
    )
    semantic = torch.tensor(
        [2.0, -1.0, 0.5],
    )

    combined = combine_message_coefficients(
        structural_normalization_factor=structural,
        relation_gate_factor=gate,
        edge_attention_factor=attention,
        semantic_edge_factor=semantic,
    )

    expected = torch.tensor(
        [0.1, -0.1, 0.5],
    )
    torch.testing.assert_close(
        combined,
        expected,
    )


def test_combine_message_coefficients_does_not_renormalize() -> None:
    structural = torch.tensor(
        [2.0, 3.0],
    )
    ones = torch.ones(2)

    combined = combine_message_coefficients(
        structural_normalization_factor=structural,
        relation_gate_factor=ones,
        edge_attention_factor=ones,
        semantic_edge_factor=ones,
    )

    torch.testing.assert_close(
        combined,
        structural,
    )
    assert not torch.isclose(
        combined.sum(),
        torch.tensor(1.0),
    )


def test_combine_message_coefficients_all_identities() -> None:
    ones = torch.ones(5)

    combined = combine_message_coefficients(
        structural_normalization_factor=ones,
        relation_gate_factor=ones,
        edge_attention_factor=ones,
        semantic_edge_factor=ones,
    )

    assert torch.equal(
        combined,
        ones,
    )


def test_combine_message_coefficients_supports_empty_edges() -> None:
    empty = torch.empty(0)

    combined = combine_message_coefficients(
        structural_normalization_factor=empty,
        relation_gate_factor=empty,
        edge_attention_factor=empty,
        semantic_edge_factor=empty,
    )

    assert combined.shape == (0,)


def test_combine_message_coefficients_preserves_float64() -> None:
    factor = torch.linspace(
        0.5,
        1.0,
        5,
        dtype=torch.float64,
    )

    combined = combine_message_coefficients(
        structural_normalization_factor=factor,
        relation_gate_factor=factor,
        edge_attention_factor=factor,
        semantic_edge_factor=factor,
    )

    assert combined.dtype == torch.float64


def test_combine_message_coefficients_analytical_gradients() -> None:
    structural = torch.tensor(
        [0.5, 0.7],
        requires_grad=True,
    )
    gate = torch.tensor(
        [0.2, 0.3],
        requires_grad=True,
    )
    attention = torch.tensor(
        [0.4, 0.6],
        requires_grad=True,
    )
    semantic = torch.tensor(
        [1.5, -2.0],
        requires_grad=True,
    )

    combined = combine_message_coefficients(
        structural_normalization_factor=structural,
        relation_gate_factor=gate,
        edge_attention_factor=attention,
        semantic_edge_factor=semantic,
    )
    combined.sum().backward()

    torch.testing.assert_close(
        structural.grad,
        gate.detach()
        * attention.detach()
        * semantic.detach(),
    )
    torch.testing.assert_close(
        gate.grad,
        structural.detach()
        * attention.detach()
        * semantic.detach(),
    )
    torch.testing.assert_close(
        attention.grad,
        structural.detach()
        * gate.detach()
        * semantic.detach(),
    )
    torch.testing.assert_close(
        semantic.grad,
        structural.detach()
        * gate.detach()
        * attention.detach(),
    )


@pytest.mark.parametrize(
    "negative_field",
    (
        "structural",
        "gate",
        "attention",
    ),
)
def test_combine_rejects_negative_nonsemantic_factor(
    negative_field: str,
) -> None:
    values = {
        "structural": torch.ones(3),
        "gate": torch.ones(3),
        "attention": torch.ones(3),
    }
    values[negative_field][0] = -0.1

    with pytest.raises(
        ValueError,
        match="nonnegative",
    ):
        combine_message_coefficients(
            structural_normalization_factor=values["structural"],
            relation_gate_factor=values["gate"],
            edge_attention_factor=values["attention"],
            semantic_edge_factor=torch.ones(3),
        )


def test_combine_rejects_nonfinite_output_overflow() -> None:
    huge = torch.full(
        (2,),
        torch.finfo(torch.float32).max,
    )

    with pytest.raises(
        FloatingPointError,
        match="non-finite",
    ):
        combine_message_coefficients(
            structural_normalization_factor=huge,
            relation_gate_factor=torch.full(
                (2,),
                2.0,
            ),
            edge_attention_factor=torch.ones(2),
            semantic_edge_factor=torch.ones(2),
        )


# =============================================================================
# Complete coefficient resolver
# =============================================================================


@pytest.mark.parametrize(
    (
        "gate_enabled",
        "attention_enabled",
        "semantic_policy",
    ),
    (
        (False, False, MESSAGE_SEMANTIC_EDGE_POLICY_IGNORE),
        (False, True, MESSAGE_SEMANTIC_EDGE_POLICY_IGNORE),
        (True, False, MESSAGE_SEMANTIC_EDGE_POLICY_IGNORE),
        (True, True, MESSAGE_SEMANTIC_EDGE_POLICY_IGNORE),
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
def test_complete_resolution_all_optional_factor_combinations(
    gate_enabled: bool,
    attention_enabled: bool,
    semantic_policy: str,
) -> None:
    inputs = _inputs()
    output = _resolve_actual(
        inputs,
        relation_gate=(
            _DEFAULT
            if gate_enabled
            else None
        ),
        edge_attention=(
            _DEFAULT
            if attention_enabled
            else None
        ),
        semantic_policy=semantic_policy,
    )

    assert output.relation_gate_enabled is gate_enabled
    assert output.edge_attention_enabled is attention_enabled
    assert output.semantic_edge_weight_enabled is (
        semantic_policy
        == MESSAGE_SEMANTIC_EDGE_POLICY_USE_SOURCE_GRAPH
    )
    torch.testing.assert_close(
        output.combined_coefficient,
        _manual_factor_product(output),
    )


def test_complete_resolution_preserves_enabled_source_identity() -> None:
    inputs = _inputs()
    normalization = _edge_normalization(inputs)
    gate = _relation_gate(inputs)
    attention = _edge_attention(inputs)

    output = resolve_message_coefficients(
        edge_normalization=normalization,
        relation_gate=gate,
        edge_attention=attention,
        semantic_edge_policy=(
            MESSAGE_SEMANTIC_EDGE_POLICY_USE_SOURCE_GRAPH
        ),
        source_inputs=inputs,
    )

    assert output.edge_normalization is normalization
    assert output.relation_gate is gate
    assert output.edge_attention is attention
    assert output.structural_normalization_factor is (
        normalization.coefficients
    )
    assert output.relation_gate_factor is (
        gate.edge_gate_values
    )
    assert output.edge_attention_factor is (
        attention.edge_weights
    )
    assert output.semantic_edge_factor is (
        inputs.source_graph.semantic_edge_weight
    )


def test_complete_resolution_disabled_factors_are_independent_identities() -> None:
    inputs = _inputs()
    output = _resolve_actual(
        inputs,
        relation_gate=None,
        edge_attention=None,
        semantic_policy=(
            MESSAGE_SEMANTIC_EDGE_POLICY_IGNORE
        ),
    )

    assert torch.equal(
        output.relation_gate_factor,
        torch.ones_like(
            output.relation_gate_factor
        ),
    )
    assert torch.equal(
        output.edge_attention_factor,
        torch.ones_like(
            output.edge_attention_factor
        ),
    )
    assert torch.equal(
        output.semantic_edge_factor,
        torch.ones_like(
            output.semantic_edge_factor
        ),
    )
    assert (
        output.relation_gate_factor
        is not output.edge_attention_factor
    )
    assert (
        output.relation_gate_factor
        is not output.semantic_edge_factor
    )


def test_complete_resolution_function_alias() -> None:
    inputs = _inputs()
    normalization = _edge_normalization(inputs)

    output = resolve_edge_message_coefficients(
        edge_normalization=normalization,
        semantic_edge_policy=(
            MESSAGE_SEMANTIC_EDGE_POLICY_IGNORE
        ),
        source_inputs=inputs,
    )

    assert isinstance(
        output,
        ResolvedMessageCoefficients,
    )


def test_coefficient_resolver_module_matches_functional_output() -> None:
    inputs = _inputs()
    normalization = _edge_normalization(inputs)
    gate = _relation_gate(inputs)
    attention = _edge_attention(inputs)
    resolver = MessageCoefficientResolver(
        semantic_edge_policy=(
            MESSAGE_SEMANTIC_EDGE_POLICY_USE_SOURCE_GRAPH
        )
    )

    module_output = resolver(
        edge_normalization=normalization,
        relation_gate=gate,
        edge_attention=attention,
        source_inputs=inputs,
    )
    functional_output = resolve_message_coefficients(
        edge_normalization=normalization,
        relation_gate=gate,
        edge_attention=attention,
        semantic_edge_policy=(
            MESSAGE_SEMANTIC_EDGE_POLICY_USE_SOURCE_GRAPH
        ),
        source_inputs=inputs,
    )

    torch.testing.assert_close(
        module_output.combined_coefficient,
        functional_output.combined_coefficient,
    )
    assert module_output.structural_normalization_factor is (
        functional_output.structural_normalization_factor
    )
    assert module_output.relation_gate_factor is (
        functional_output.relation_gate_factor
    )
    assert module_output.edge_attention_factor is (
        functional_output.edge_attention_factor
    )
    assert module_output.semantic_edge_factor is (
        functional_output.semantic_edge_factor
    )


def test_coefficient_resolver_builders() -> None:
    first = build_message_coefficient_resolver(
        semantic_edge_policy=(
            MESSAGE_SEMANTIC_EDGE_POLICY_IGNORE
        )
    )
    second = build_coefficient_resolver(
        semantic_edge_policy=(
            MESSAGE_SEMANTIC_EDGE_POLICY_USE_SOURCE_GRAPH
        )
    )

    assert isinstance(
        first,
        MessageCoefficientResolver,
    )
    assert isinstance(
        second,
        MessageCoefficientResolver,
    )
    assert first.semantic_edge_policy == (
        MESSAGE_SEMANTIC_EDGE_POLICY_IGNORE
    )
    assert second.semantic_edge_policy == (
        MESSAGE_SEMANTIC_EDGE_POLICY_USE_SOURCE_GRAPH
    )


def test_coefficient_resolver_is_parameter_and_buffer_free() -> None:
    resolver = MessageCoefficientResolver()

    _assert_parameter_and_buffer_free(resolver)
    assert resolver.parameter_count == 0
    assert resolver.trainable_parameter_count == 0
    assert resolver.buffer_count == 0
    assert resolver.parameter_fingerprint is None
    resolver.assert_parameter_free()


def test_coefficient_architecture_fingerprint_depends_on_policy() -> None:
    ignored = (
        coefficient_resolution_architecture_fingerprint(
            semantic_edge_policy=(
                MESSAGE_SEMANTIC_EDGE_POLICY_IGNORE
            )
        )
    )
    consumed = (
        coefficient_resolution_architecture_fingerprint(
            semantic_edge_policy=(
                MESSAGE_SEMANTIC_EDGE_POLICY_USE_SOURCE_GRAPH
            )
        )
    )

    assert ignored != consumed
    assert ignored == (
        coefficient_resolution_architecture_fingerprint(
            semantic_edge_policy=(
                MESSAGE_SEMANTIC_EDGE_POLICY_IGNORE
            )
        )
    )


def test_coefficient_architecture_metadata_separates_responsibilities() -> None:
    metadata = coefficient_resolution_architecture_dict(
        semantic_edge_policy=(
            MESSAGE_SEMANTIC_EDGE_POLICY_IGNORE
        )
    )

    assert metadata["factor_order"] == list(
        MESSAGE_FACTOR_ORDER
    )
    assert metadata["parameter_free"] is True
    assert metadata["buffer_free"] is True
    assert metadata["message_composition_owned_here"] is False
    assert metadata["aggregation_owned_here"] is False
    assert metadata["claims_causal_importance"] is False


def test_complete_resolution_supports_empty_edges() -> None:
    inputs = _inputs(
        empty_edges=True,
    )
    output = _resolve_actual(
        inputs,
        relation_gate=None,
        edge_attention=None,
        semantic_policy=(
            MESSAGE_SEMANTIC_EDGE_POLICY_USE_SOURCE_GRAPH
        ),
    )

    assert output.num_edges == 0
    assert output.combined_coefficient.shape == (0,)


def test_complete_resolution_supports_float64() -> None:
    inputs = _inputs(
        dtype=torch.float64,
    )
    output = _resolve_actual(
        inputs,
        semantic_policy=(
            MESSAGE_SEMANTIC_EDGE_POLICY_USE_SOURCE_GRAPH
        ),
    )

    assert output.dtype == torch.float64
    assert output.combined_coefficient.dtype == torch.float64


def test_complete_resolution_preserves_gradients_from_structural_and_semantic() -> None:
    structural = torch.linspace(
        0.4,
        1.0,
        EDGES,
        requires_grad=True,
    )
    semantic = torch.linspace(
        0.5,
        1.5,
        EDGES,
        requires_grad=True,
    )
    inputs = _inputs(
        semantic_edge_weight=semantic,
    )
    normalization = _edge_normalization(
        inputs,
        coefficients=structural,
    )

    output = resolve_message_coefficients(
        edge_normalization=normalization,
        relation_gate=None,
        edge_attention=None,
        semantic_edge_policy=(
            MESSAGE_SEMANTIC_EDGE_POLICY_USE_SOURCE_GRAPH
        ),
        source_inputs=inputs,
    )
    output.combined_coefficient.sum().backward()

    torch.testing.assert_close(
        structural.grad,
        semantic.detach(),
    )
    torch.testing.assert_close(
        semantic.grad,
        structural.detach(),
    )


def test_complete_resolution_rejects_crossed_gate_lineage() -> None:
    first = _inputs(
        source_fingerprint="first",
    )
    second = _inputs(
        source_fingerprint="second",
    )

    with pytest.raises(
        ValueError,
        match="exact supplied source_inputs",
    ):
        resolve_message_coefficients(
            edge_normalization=(
                _edge_normalization(first)
            ),
            relation_gate=_relation_gate(second),
            semantic_edge_policy=(
                MESSAGE_SEMANTIC_EDGE_POLICY_IGNORE
            ),
            source_inputs=first,
        )


def test_complete_resolution_rejects_crossed_attention_lineage() -> None:
    first = _inputs(
        source_fingerprint="first",
    )
    second = _inputs(
        source_fingerprint="second",
    )

    with pytest.raises(
        ValueError,
        match="exact supplied source_inputs",
    ):
        resolve_message_coefficients(
            edge_normalization=(
                _edge_normalization(first)
            ),
            edge_attention=(
                _edge_attention(second)
            ),
            semantic_edge_policy=(
                MESSAGE_SEMANTIC_EDGE_POLICY_IGNORE
            ),
            source_inputs=first,
        )


def test_complete_resolution_validator_accepts_output() -> None:
    inputs = _inputs()
    normalization = _edge_normalization(inputs)
    gate = _relation_gate(inputs)
    attention = _edge_attention(inputs)
    output = resolve_message_coefficients(
        edge_normalization=normalization,
        relation_gate=gate,
        edge_attention=attention,
        semantic_edge_policy=(
            MESSAGE_SEMANTIC_EDGE_POLICY_USE_SOURCE_GRAPH
        ),
        source_inputs=inputs,
    )

    validate_resolved_message_coefficients(
        output=output,
        source_inputs=inputs,
        edge_normalization=normalization,
        relation_gate=gate,
        edge_attention=attention,
        semantic_edge_policy=(
            MESSAGE_SEMANTIC_EDGE_POLICY_USE_SOURCE_GRAPH
        ),
    )


def test_coefficient_diagnostics_report_factor_statistics() -> None:
    output = _resolve_actual(
        _inputs(),
        semantic_policy=(
            MESSAGE_SEMANTIC_EDGE_POLICY_USE_SOURCE_GRAPH
        ),
    )
    summary = message_coefficient_diagnostic_summary(
        output
    )

    assert summary["num_edges"] == EDGES
    assert summary["relation_gate_enabled"] is True
    assert summary["edge_attention_enabled"] is True
    assert summary["semantic_edge_weight_enabled"] is True
    assert summary["active_factor_names"] == list(
        MESSAGE_FACTOR_ORDER
    )
    assert set(summary["factors"]) == {
        MESSAGE_FACTOR_STRUCTURAL_NORMALIZATION,
        MESSAGE_FACTOR_RELATION_GATE,
        MESSAGE_FACTOR_EDGE_ATTENTION,
        MESSAGE_FACTOR_SEMANTIC_EDGE_WEIGHT,
    }
    assert summary["combined_coefficient"]["finite"] is True
    assert summary["aggregation_performed_here"] is False
    assert summary["causal_importance_claim"] is False


# =============================================================================
# Low-level message-composition kernel
# =============================================================================


def test_message_tensor_composition_matches_hand_calculation() -> None:
    state = torch.tensor(
        [
            [1.0, 2.0, -1.0],
            [4.0, -2.0, 0.5],
        ]
    )
    coefficient = torch.tensor(
        [0.5, -2.0],
    )

    messages = compose_edge_message_tensor(
        transformed_source_state=state,
        combined_coefficient=coefficient,
    )

    expected = torch.tensor(
        [
            [0.5, 1.0, -0.5],
            [-8.0, 4.0, -1.0],
        ]
    )
    torch.testing.assert_close(
        messages,
        expected,
    )


def test_message_composition_broadcasts_only_hidden_axis() -> None:
    state = torch.arange(
        12,
        dtype=torch.float32,
    ).reshape(3, 4)
    coefficient = torch.tensor(
        [1.0, 10.0, -1.0],
    )

    messages = compose_edge_message_tensor(
        transformed_source_state=state,
        combined_coefficient=coefficient,
    )

    for edge in range(3):
        torch.testing.assert_close(
            messages[edge],
            state[edge] * coefficient[edge],
        )


def test_message_composition_zero_coefficient_zeroes_only_its_edge() -> None:
    state = torch.ones(
        3,
        4,
    )
    coefficient = torch.tensor(
        [1.0, 0.0, 2.0],
    )

    messages = compose_edge_message_tensor(
        transformed_source_state=state,
        combined_coefficient=coefficient,
    )

    assert torch.equal(
        messages[1],
        torch.zeros(4),
    )
    assert torch.equal(
        messages[0],
        torch.ones(4),
    )
    assert torch.equal(
        messages[2],
        torch.full((4,), 2.0),
    )


def test_signed_coefficient_flips_message_direction() -> None:
    state = torch.tensor(
        [[1.0, -2.0, 3.0]]
    )
    coefficient = torch.tensor(
        [-0.5]
    )

    messages = compose_edge_message_tensor(
        transformed_source_state=state,
        combined_coefficient=coefficient,
    )

    torch.testing.assert_close(
        messages,
        torch.tensor(
            [[-0.5, 1.0, -1.5]]
        ),
    )


def test_message_composition_does_not_aggregate_equal_targets() -> None:
    # Two edge rows may target the same node upstream, but this kernel retains
    # both edge rows independently.
    state = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )
    coefficient = torch.tensor(
        [0.5, 2.0],
    )

    messages = compose_edge_message_tensor(
        transformed_source_state=state,
        combined_coefficient=coefficient,
    )

    assert messages.shape == (2, 2)
    torch.testing.assert_close(
        messages[0],
        torch.tensor([0.5, 1.0]),
    )
    torch.testing.assert_close(
        messages[1],
        torch.tensor([6.0, 8.0]),
    )


def test_message_composition_is_edge_permutation_equivariant() -> None:
    generator = torch.Generator().manual_seed(7)
    state = torch.randn(
        8,
        5,
        generator=generator,
    )
    coefficient = torch.randn(
        8,
        generator=generator,
    )
    permutation = torch.tensor(
        [4, 0, 7, 2, 5, 1, 6, 3],
    )

    original = compose_edge_message_tensor(
        transformed_source_state=state,
        combined_coefficient=coefficient,
    )
    permuted = compose_edge_message_tensor(
        transformed_source_state=state[permutation],
        combined_coefficient=coefficient[permutation],
    )

    torch.testing.assert_close(
        permuted,
        original[permutation],
    )


def test_message_composition_supports_empty_edges() -> None:
    state = torch.empty(
        0,
        HIDDEN_DIM,
    )
    coefficient = torch.empty(0)

    messages = compose_edge_message_tensor(
        transformed_source_state=state,
        combined_coefficient=coefficient,
    )

    assert messages.shape == (
        0,
        HIDDEN_DIM,
    )


def test_message_composition_preserves_float64() -> None:
    state = torch.randn(
        4,
        3,
        dtype=torch.float64,
    )
    coefficient = torch.randn(
        4,
        dtype=torch.float64,
    )

    messages = compose_edge_message_tensor(
        transformed_source_state=state,
        combined_coefficient=coefficient,
    )

    assert messages.dtype == torch.float64


def test_message_composition_analytical_gradients() -> None:
    state = torch.tensor(
        [
            [1.0, 2.0],
            [-3.0, 4.0],
        ],
        requires_grad=True,
    )
    coefficient = torch.tensor(
        [0.5, -2.0],
        requires_grad=True,
    )

    messages = compose_edge_message_tensor(
        transformed_source_state=state,
        combined_coefficient=coefficient,
    )
    messages.sum().backward()

    expected_state_grad = (
        coefficient.detach()
        .unsqueeze(-1)
        .expand_as(state)
    )
    expected_coefficient_grad = (
        state.detach().sum(dim=1)
    )

    torch.testing.assert_close(
        state.grad,
        expected_state_grad,
    )
    torch.testing.assert_close(
        coefficient.grad,
        expected_coefficient_grad,
    )


def test_message_composition_rejects_nonfinite_input() -> None:
    state = torch.ones(
        2,
        3,
    )
    state[0, 0] = float("nan")

    with pytest.raises(
        FloatingPointError,
        match="finite",
    ):
        compose_edge_message_tensor(
            transformed_source_state=state,
            combined_coefficient=torch.ones(2),
        )


def test_message_composition_rejects_overflow() -> None:
    state = torch.full(
        (2, 3),
        torch.finfo(torch.float32).max,
    )
    coefficient = torch.full(
        (2,),
        2.0,
    )

    with pytest.raises(
        FloatingPointError,
        match="non-finite",
    ):
        compose_edge_message_tensor(
            transformed_source_state=state,
            combined_coefficient=coefficient,
        )


# =============================================================================
# Complete composition outputs and module wrapper
# =============================================================================


def test_complete_composition_matches_manual_equation() -> None:
    inputs = _inputs()
    transform = _relation_transform(inputs)
    resolved = _resolve_actual(
        inputs,
        semantic_policy=(
            MESSAGE_SEMANTIC_EDGE_POLICY_USE_SOURCE_GRAPH
        ),
    )

    output = compose_message_output(
        relation_transform=transform,
        resolved_coefficients=resolved,
        source_inputs=inputs,
    )

    expected = (
        transform.transformed_source_state
        * resolved
        .combined_coefficient
        .unsqueeze(-1)
    )
    torch.testing.assert_close(
        output.edge_messages,
        expected,
    )
    assert output.relation_transform is transform
    assert output.resolved_coefficients is resolved
    assert output.source_inputs is inputs


def test_complete_composition_function_alias() -> None:
    inputs = _inputs()
    transform = _relation_transform(inputs)
    resolved = _resolve_actual(inputs)

    output = compose_edge_messages(
        relation_transform=transform,
        resolved_coefficients=resolved,
        source_inputs=inputs,
    )

    assert isinstance(
        output,
        MessageCompositionOutput,
    )


def test_message_composer_module_matches_functional_output() -> None:
    inputs = _inputs()
    transform = _relation_transform(inputs)
    resolved = _resolve_actual(
        inputs,
        semantic_policy=(
            MESSAGE_SEMANTIC_EDGE_POLICY_USE_SOURCE_GRAPH
        ),
    )
    composer = EdgeMessageComposer()

    module_output = composer(
        relation_transform=transform,
        resolved_coefficients=resolved,
        source_inputs=inputs,
    )
    functional_output = compose_message_output(
        relation_transform=transform,
        resolved_coefficients=resolved,
        source_inputs=inputs,
    )

    torch.testing.assert_close(
        module_output.edge_messages,
        functional_output.edge_messages,
    )
    assert module_output.relation_transform is transform
    assert module_output.resolved_coefficients is resolved


def test_message_composer_builders() -> None:
    gatherer = RelationStateGather()
    first = build_edge_message_composer(
        relation_state_gather=gatherer,
    )
    second = build_message_composer()

    assert isinstance(first, EdgeMessageComposer)
    assert isinstance(second, EdgeMessageComposer)
    assert first.relation_state_gather is gatherer


def test_message_composer_is_parameter_and_buffer_free() -> None:
    composer = EdgeMessageComposer()

    _assert_parameter_and_buffer_free(composer)
    assert composer.parameter_count == 0
    assert composer.trainable_parameter_count == 0
    assert composer.buffer_count == 0
    assert composer.parameter_fingerprint is None
    composer.assert_parameter_free()


def test_message_composer_architecture_is_deterministic() -> None:
    assert message_composer_architecture_dict() == (
        message_composer_architecture_dict()
    )
    assert message_composer_architecture_fingerprint() == (
        message_composer_architecture_fingerprint()
    )

    metadata = message_composer_architecture_dict()
    assert metadata["parameter_free"] is True
    assert metadata["buffer_free"] is True
    assert metadata["aggregation_owned_here"] is False
    assert metadata["relation_state_recomputed_here"] is False
    assert metadata["coefficient_recomputed_here"] is False


def test_module_architecture_includes_relation_state_boundary() -> None:
    composer = EdgeMessageComposer()
    metadata = composer.architecture_dict()

    assert "relation_state_gather" in metadata
    assert metadata[
        "relation_state_gather"
    ] == relation_state_gather_architecture_dict()
    assert composer.architecture_fingerprint() == (
        composer.architecture_fingerprint()
    )


def test_complete_composition_supports_empty_edges() -> None:
    inputs = _inputs(
        empty_edges=True,
    )
    transform = _relation_transform(inputs)
    resolved = _resolve_actual(
        inputs,
        relation_gate=None,
        edge_attention=None,
        semantic_policy=(
            MESSAGE_SEMANTIC_EDGE_POLICY_IGNORE
        ),
    )

    output = compose_message_output(
        relation_transform=transform,
        resolved_coefficients=resolved,
        source_inputs=inputs,
    )

    assert output.edge_messages.shape == (
        0,
        HIDDEN_DIM,
    )


def test_complete_composition_supports_float64() -> None:
    inputs = _inputs(
        dtype=torch.float64,
    )
    output = _compose_actual(inputs)

    assert output.dtype == torch.float64
    assert output.edge_messages.dtype == torch.float64


def test_complete_pipeline_preserves_autograd() -> None:
    transformed = torch.randn(
        EDGES,
        HIDDEN_DIM,
        requires_grad=True,
    )
    structural = torch.linspace(
        0.4,
        1.0,
        EDGES,
        requires_grad=True,
    )
    semantic = torch.linspace(
        0.5,
        1.5,
        EDGES,
        requires_grad=True,
    )
    inputs = _inputs(
        semantic_edge_weight=semantic,
    )
    transform = _relation_transform(
        inputs,
        tensor=transformed,
    )
    normalization = _edge_normalization(
        inputs,
        coefficients=structural,
    )
    resolved = resolve_message_coefficients(
        edge_normalization=normalization,
        relation_gate=None,
        edge_attention=None,
        semantic_edge_policy=(
            MESSAGE_SEMANTIC_EDGE_POLICY_USE_SOURCE_GRAPH
        ),
        source_inputs=inputs,
    )
    output = compose_message_output(
        relation_transform=transform,
        resolved_coefficients=resolved,
        source_inputs=inputs,
    )

    output.edge_messages.sum().backward()

    expected_transformed_grad = (
        structural.detach()
        * semantic.detach()
    ).unsqueeze(-1).expand_as(
        transformed
    )
    expected_structural_grad = (
        transformed.detach().sum(dim=1)
        * semantic.detach()
    )
    expected_semantic_grad = (
        transformed.detach().sum(dim=1)
        * structural.detach()
    )

    torch.testing.assert_close(
        transformed.grad,
        expected_transformed_grad,
    )
    torch.testing.assert_close(
        structural.grad,
        expected_structural_grad,
    )
    torch.testing.assert_close(
        semantic.grad,
        expected_semantic_grad,
    )


def test_complete_composition_rejects_crossed_lineage() -> None:
    first = _inputs(
        source_fingerprint="first",
    )
    second = _inputs(
        source_fingerprint="second",
    )

    with pytest.raises(
        ValueError,
        match="exact same FunctionalMessagePassingInputs",
    ):
        compose_message_output(
            relation_transform=(
                _relation_transform(first)
            ),
            resolved_coefficients=(
                _resolve_actual(second)
            ),
        )


def test_complete_composition_validator_accepts_output() -> None:
    inputs = _inputs()
    transform = _relation_transform(inputs)
    resolved = _resolve_actual(inputs)
    output = compose_message_output(
        relation_transform=transform,
        resolved_coefficients=resolved,
        source_inputs=inputs,
    )

    validate_message_composition_output(
        output=output,
        relation_transform=transform,
        resolved_coefficients=resolved,
        source_inputs=inputs,
        composer_architecture_fingerprint=(
            output.composer_architecture_fingerprint
        ),
    )


def test_message_composition_diagnostics_are_descriptive() -> None:
    output = _compose_actual(
        _inputs()
    )
    summary = message_composition_diagnostic_summary(
        output
    )

    assert summary["num_edges"] == EDGES
    assert summary["hidden_dim"] == HIDDEN_DIM
    assert summary[
        "transformed_state_zero_copy_identity_preserved"
    ] is True
    assert summary["combined_coefficient"]["finite"] is True
    assert summary["edge_messages"]["finite"] is True
    assert summary["aggregation_performed_here"] is False
    assert summary["residual_update_performed_here"] is False
    assert summary["causal_importance_claim"] is False
    assert summary["explanation_faithfulness_claim"] is False


def test_module_diagnostic_summary_requires_its_architecture() -> None:
    inputs = _inputs()
    transform = _relation_transform(inputs)
    resolved = _resolve_actual(inputs)
    composer = EdgeMessageComposer()
    output = composer(
        relation_transform=transform,
        resolved_coefficients=resolved,
        source_inputs=inputs,
    )

    summary = composer.diagnostic_summary(
        output
    )

    assert summary["num_edges"] == EDGES


# =============================================================================
# Cross-stage invariants
# =============================================================================


def test_disabled_optional_factors_reduce_to_structural_only() -> None:
    inputs = _inputs()
    normalization = _edge_normalization(inputs)
    resolved = resolve_message_coefficients(
        edge_normalization=normalization,
        relation_gate=None,
        edge_attention=None,
        semantic_edge_policy=(
            MESSAGE_SEMANTIC_EDGE_POLICY_IGNORE
        ),
        source_inputs=inputs,
    )

    torch.testing.assert_close(
        resolved.combined_coefficient,
        normalization.coefficients,
    )


def test_full_pipeline_matches_expanded_five_factor_equation() -> None:
    inputs = _inputs()
    transform = _relation_transform(inputs)
    normalization = _edge_normalization(inputs)
    gate = _relation_gate(inputs)
    attention = _edge_attention(inputs)
    resolved = resolve_message_coefficients(
        edge_normalization=normalization,
        relation_gate=gate,
        edge_attention=attention,
        semantic_edge_policy=(
            MESSAGE_SEMANTIC_EDGE_POLICY_USE_SOURCE_GRAPH
        ),
        source_inputs=inputs,
    )
    output = compose_message_output(
        relation_transform=transform,
        resolved_coefficients=resolved,
        source_inputs=inputs,
    )

    expected = (
        transform.transformed_source_state
        * normalization.coefficients.unsqueeze(-1)
        * gate.edge_gate_values.unsqueeze(-1)
        * attention.edge_weights.unsqueeze(-1)
        * inputs
        .source_graph
        .semantic_edge_weight
        .unsqueeze(-1)
    )

    torch.testing.assert_close(
        output.edge_messages,
        expected,
    )


def test_semantic_ignore_and_use_source_graph_change_only_semantic_factor() -> None:
    inputs = _inputs()
    normalization = _edge_normalization(inputs)
    gate = _relation_gate(inputs)
    attention = _edge_attention(inputs)

    ignored = resolve_message_coefficients(
        edge_normalization=normalization,
        relation_gate=gate,
        edge_attention=attention,
        semantic_edge_policy=(
            MESSAGE_SEMANTIC_EDGE_POLICY_IGNORE
        ),
        source_inputs=inputs,
    )
    consumed = resolve_message_coefficients(
        edge_normalization=normalization,
        relation_gate=gate,
        edge_attention=attention,
        semantic_edge_policy=(
            MESSAGE_SEMANTIC_EDGE_POLICY_USE_SOURCE_GRAPH
        ),
        source_inputs=inputs,
    )

    assert (
        ignored.structural_normalization_factor
        is consumed.structural_normalization_factor
    )
    assert (
        ignored.relation_gate_factor
        is consumed.relation_gate_factor
    )
    assert (
        ignored.edge_attention_factor
        is consumed.edge_attention_factor
    )
    assert ignored.semantic_edge_weight is None
    assert consumed.semantic_edge_weight is (
        inputs.source_graph.semantic_edge_weight
    )
    torch.testing.assert_close(
        consumed.combined_coefficient,
        ignored.combined_coefficient
        * inputs.source_graph.semantic_edge_weight,
    )


def test_message_builder_stages_do_not_expose_aggregation_outputs() -> None:
    inputs = _inputs()
    resolved = _resolve_actual(inputs)
    composition = _compose_actual(
        inputs,
        resolved_coefficients=resolved,
    )

    for value in (
        resolved,
        composition,
        RelationStateGather(),
        MessageCoefficientResolver(),
        EdgeMessageComposer(),
    ):
        assert not hasattr(
            value,
            "aggregated_messages",
        )
        assert not hasattr(
            value,
            "target_updates",
        )
        assert not hasattr(
            value,
            "causal_importance",
        )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is not available.",
)
def test_full_numerical_pipeline_supports_cuda() -> None:
    inputs = _inputs(
        device="cuda",
    )
    transform = _relation_transform(inputs)
    resolved = _resolve_actual(
        inputs,
        semantic_policy=(
            MESSAGE_SEMANTIC_EDGE_POLICY_USE_SOURCE_GRAPH
        ),
    )
    output = compose_message_output(
        relation_transform=transform,
        resolved_coefficients=resolved,
        source_inputs=inputs,
    )

    assert output.device.type == "cuda"
    assert output.edge_messages.device.type == "cuda"
    assert resolved.device.type == "cuda"
