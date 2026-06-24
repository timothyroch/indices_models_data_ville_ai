"""
Integration and public-contract tests for complete edge-message builders.

Target repository path:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            tests/
                message_builders/
                    test_message_builders.py

Implementation under test:
    functional_message_passing/message_builders/message_builders.py

Supporting integration surfaces:
    message_builders/diagnostics.py
    message_builders/coefficient_resolution.py
    message_builders/message_composition.py
    message_builders/relation_state_gather.py
    message_builders/schemas.py

This suite tests the complete package-level orchestration rather than repeating
the low-level numerical tests in ``test_message_builder_numerics.py``.

Primary contracts
-----------------
- one exact ``FunctionalMessagePassingInputs`` object is shared by every stage;
- resolved coefficients preserve exact upstream source tensors;
- composed edge messages preserve exact internal-stage lineage;
- the public ``EdgeMessageOutput`` reuses the exact composed message tensor;
- disabled gates and attention remain ``None`` publicly and identity one
  internally;
- semantic-edge policy is frozen by the builder architecture;
- ordinary ``forward`` never executes diagnostics implicitly;
- diagnostics are explicit, tensor-free, JSON-safe, and descriptive only;
- diagnostics configuration does not change the numerical model fingerprint;
- the complete message-builder subsystem is parameter-free and buffer-free;
- empty edges, float64, autograd, train/eval parity, and optional CUDA work;
- target-node aggregation and causal claims remain outside this subsystem.

Controlled upstream doubles are patched into
``functional_message_passing.schemas`` so failures remain localized to the
message-builder package.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import json
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
    MessageCoefficientResolver,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.functional_message_passing.message_builders.diagnostics import (
    DEFAULT_MESSAGE_BUILDER_DIAGNOSTIC_THRESHOLDS,
    MESSAGE_BUILDER_DIAGNOSTICS_BUFFER_FREE,
    MESSAGE_BUILDER_DIAGNOSTICS_INTERPRETATION,
    MESSAGE_BUILDER_DIAGNOSTICS_PARAMETER_FREE,
    MESSAGE_BUILDER_DIAGNOSTICS_SCHEMA_VERSION,
    MESSAGE_BUILDER_DIAGNOSTIC_SECTION_ALERTS,
    MESSAGE_BUILDER_DIAGNOSTIC_SECTION_BY_GRAPH,
    MESSAGE_BUILDER_DIAGNOSTIC_SECTION_BY_RELATION,
    MESSAGE_BUILDER_DIAGNOSTIC_SECTION_GLOBAL,
    MESSAGE_BUILDER_DIAGNOSTIC_SECTION_LINEAGE,
    MESSAGE_BUILDER_DIAGNOSTIC_SECTION_STAGE_SUMMARIES,
    MessageBuilderDiagnostics,
    MessageBuilderDiagnosticThresholds,
    build_message_builder_diagnostic_report,
    build_public_edge_message_diagnostic_report,
    diagnostic_report_fingerprint,
    validate_message_builder_diagnostic_report,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.functional_message_passing.message_builders.message_builders import (
    MESSAGE_BUILDERS_AGGREGATION_OWNED_HERE,
    MESSAGE_BUILDERS_BUFFER_FREE,
    MESSAGE_BUILDERS_DISABLED_ATTENTION_REPRESENTATION,
    MESSAGE_BUILDERS_DISABLED_GATE_REPRESENTATION,
    MESSAGE_BUILDERS_OPERATION_ORDER,
    MESSAGE_BUILDERS_ORCHESTRATOR_SCHEMA_VERSION,
    MESSAGE_BUILDERS_OUTPUT_SCHEMA,
    MESSAGE_BUILDERS_PARAMETER_FREE,
    MESSAGE_BUILDERS_SCIENTIFIC_INTERPRETATION,
    EdgeMessageBuilder,
    FunctionalEdgeMessageBuilder,
    FunctionalMessageBuilder,
    MessageBuilder,
    MessageBuilderRun,
    MessageBuilderRunWithDiagnostics,
    assemble_edge_message_output,
    build_edge_message_builder,
    build_message_builder,
    build_message_builders,
    run_edge_message_builder,
    run_edge_message_builder_stages,
    run_message_builder,
    run_message_builder_stages,
    validate_complete_message_builder_run,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.functional_message_passing.message_builders.message_composition import (
    EdgeMessageComposer,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.functional_message_passing.message_builders.relation_state_gather import (
    RelationStateGather,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.functional_message_passing.message_builders.schemas import (
    MESSAGE_FACTOR_ORDER,
    MESSAGE_SEMANTIC_EDGE_POLICY_IGNORE,
    MESSAGE_SEMANTIC_EDGE_POLICY_USE_SOURCE_GRAPH,
    MessageCompositionOutput,
    ResolvedMessageCoefficients,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.functional_message_passing.schemas import (
    EdgeAttentionOutput,
    EdgeMessageOutput,
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
# Integration helpers
# =============================================================================


def _builder(
    *,
    semantic_policy: str = MESSAGE_SEMANTIC_EDGE_POLICY_IGNORE,
    diagnostics_enabled: bool = False,
    include_per_relation: bool = True,
    include_per_graph: bool = True,
    thresholds: MessageBuilderDiagnosticThresholds = (
        DEFAULT_MESSAGE_BUILDER_DIAGNOSTIC_THRESHOLDS
    ),
) -> EdgeMessageBuilder:
    return build_edge_message_builder(
        semantic_edge_policy=semantic_policy,
        diagnostics_enabled=diagnostics_enabled,
        diagnostics_include_per_relation=include_per_relation,
        diagnostics_include_per_graph=include_per_graph,
        diagnostic_thresholds=thresholds,
    )


def _upstream(
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
    empty_edges: bool = False,
    semantic_edge_weight: (
        torch.Tensor | None | object
    ) = _DEFAULT,
    transformed_state: torch.Tensor | None = None,
    structural_coefficients: torch.Tensor | None = None,
    gate_enabled: bool = True,
    attention_enabled: bool = True,
    source_fingerprint: str = "message-builder-integration",
) -> tuple[
    FunctionalMessagePassingInputs,
    RelationTransformOutput,
    StructuralEdgeNormalizationOutput,
    RelationGateOutput | None,
    EdgeAttentionOutput | None,
]:
    inputs = _inputs(
        dtype=dtype,
        device=device,
        empty_edges=empty_edges,
        semantic_edge_weight=semantic_edge_weight,
        source_fingerprint=source_fingerprint,
    )
    transform = _relation_transform(
        inputs,
        tensor=transformed_state,
    )
    normalization = _edge_normalization(
        inputs,
        coefficients=structural_coefficients,
    )
    gate = (
        _relation_gate(inputs)
        if gate_enabled
        else None
    )
    attention = (
        _edge_attention(inputs)
        if attention_enabled
        else None
    )

    return (
        inputs,
        transform,
        normalization,
        gate,
        attention,
    )


def _run(
    builder: EdgeMessageBuilder,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
    empty_edges: bool = False,
    semantic_edge_weight: (
        torch.Tensor | None | object
    ) = _DEFAULT,
    transformed_state: torch.Tensor | None = None,
    structural_coefficients: torch.Tensor | None = None,
    gate_enabled: bool = True,
    attention_enabled: bool = True,
    source_fingerprint: str = "message-builder-integration",
) -> tuple[
    MessageBuilderRun,
    FunctionalMessagePassingInputs,
    RelationTransformOutput,
    StructuralEdgeNormalizationOutput,
    RelationGateOutput | None,
    EdgeAttentionOutput | None,
]:
    (
        inputs,
        transform,
        normalization,
        gate,
        attention,
    ) = _upstream(
        dtype=dtype,
        device=device,
        empty_edges=empty_edges,
        semantic_edge_weight=semantic_edge_weight,
        transformed_state=transformed_state,
        structural_coefficients=structural_coefficients,
        gate_enabled=gate_enabled,
        attention_enabled=attention_enabled,
        source_fingerprint=source_fingerprint,
    )

    run = builder.run_complete(
        relation_transform=transform,
        edge_normalization=normalization,
        relation_gate=gate,
        edge_attention=attention,
        source_inputs=inputs,
    )

    return (
        run,
        inputs,
        transform,
        normalization,
        gate,
        attention,
    )


def _assert_parameter_and_buffer_free(
    module: nn.Module,
) -> None:
    assert tuple(module.named_parameters()) == ()
    assert tuple(module.named_buffers()) == ()
    assert len(module.state_dict()) == 0
    assert sum(
        int(parameter.numel())
        for parameter in module.parameters()
    ) == 0
    assert sum(
        int(buffer.numel())
        for buffer in module.buffers()
    ) == 0


def _assert_tensor_free(
    value: Any,
) -> None:
    if isinstance(value, torch.Tensor):
        raise AssertionError(
            "Diagnostic report unexpectedly retained a tensor."
        )

    if isinstance(value, dict):
        for nested in value.values():
            _assert_tensor_free(nested)
        return

    if isinstance(value, (list, tuple)):
        for nested in value:
            _assert_tensor_free(nested)


def _manual_messages(
    *,
    transform: RelationTransformOutput,
    normalization: StructuralEdgeNormalizationOutput,
    gate: RelationGateOutput | None,
    attention: EdgeAttentionOutput | None,
    semantic_weight: torch.Tensor | None,
) -> torch.Tensor:
    factor = normalization.coefficients

    if gate is not None:
        factor = factor * gate.edge_gate_values

    if attention is not None:
        factor = factor * attention.edge_weights

    if semantic_weight is not None:
        factor = factor * semantic_weight

    return (
        transform.transformed_source_state
        * factor.unsqueeze(-1)
    )


# =============================================================================
# Public identity, aliases, and builders
# =============================================================================


def test_orchestrator_public_identity_constants() -> None:
    assert (
        MESSAGE_BUILDERS_ORCHESTRATOR_SCHEMA_VERSION
        .strip()
    )
    assert MESSAGE_BUILDERS_SCIENTIFIC_INTERPRETATION == (
        "hierarchical_functional_edge_message_construction"
    )
    assert MESSAGE_BUILDERS_OPERATION_ORDER[-1] == (
        "validate_exact_public_output_lineage"
    )
    assert MESSAGE_BUILDERS_DISABLED_GATE_REPRESENTATION == (
        "None_publicly_and_exact_identity_one_internally"
    )
    assert MESSAGE_BUILDERS_DISABLED_ATTENTION_REPRESENTATION == (
        "None_publicly_and_exact_identity_one_internally"
    )
    assert MESSAGE_BUILDERS_PARAMETER_FREE is True
    assert MESSAGE_BUILDERS_BUFFER_FREE is True
    assert MESSAGE_BUILDERS_AGGREGATION_OWNED_HERE is False
    assert MESSAGE_BUILDERS_OUTPUT_SCHEMA == "EdgeMessageOutput"


def test_diagnostics_public_identity_constants() -> None:
    assert MESSAGE_BUILDER_DIAGNOSTICS_SCHEMA_VERSION.strip()
    assert MESSAGE_BUILDER_DIAGNOSTICS_INTERPRETATION == (
        "descriptive_numerical_and_lineage_diagnostics_only"
    )
    assert MESSAGE_BUILDER_DIAGNOSTICS_PARAMETER_FREE is True
    assert MESSAGE_BUILDER_DIAGNOSTICS_BUFFER_FREE is True


def test_orchestrator_aliases_are_exact() -> None:
    assert MessageBuilder is EdgeMessageBuilder
    assert FunctionalMessageBuilder is EdgeMessageBuilder
    assert FunctionalEdgeMessageBuilder is EdgeMessageBuilder
    assert build_message_builder is build_edge_message_builder
    assert build_message_builders is build_edge_message_builder
    assert run_message_builder is run_edge_message_builder
    assert (
        run_message_builder_stages
        is run_edge_message_builder_stages
    )


@pytest.mark.parametrize(
    "factory",
    (
        build_edge_message_builder,
        build_message_builder,
        build_message_builders,
    ),
)
def test_builder_factories_construct_complete_subsystem(
    factory: Any,
) -> None:
    builder = factory(
        semantic_edge_policy=(
            MESSAGE_SEMANTIC_EDGE_POLICY_IGNORE
        ),
        diagnostics_enabled=True,
    )

    assert isinstance(builder, EdgeMessageBuilder)
    assert isinstance(
        builder.coefficient_resolver,
        MessageCoefficientResolver,
    )
    assert isinstance(
        builder.message_composer,
        EdgeMessageComposer,
    )
    assert isinstance(
        builder.relation_state_gather,
        RelationStateGather,
    )
    assert isinstance(
        builder.diagnostics,
        MessageBuilderDiagnostics,
    )


def test_from_policy_constructs_expected_policy() -> None:
    builder = EdgeMessageBuilder.from_policy(
        semantic_edge_policy=(
            MESSAGE_SEMANTIC_EDGE_POLICY_USE_SOURCE_GRAPH
        ),
        diagnostics_enabled=False,
    )

    assert builder.semantic_edge_policy == (
        MESSAGE_SEMANTIC_EDGE_POLICY_USE_SOURCE_GRAPH
    )
    assert builder.diagnostics_enabled is False
    assert builder.diagnostics is None


def test_builder_component_constructor() -> None:
    resolver = MessageCoefficientResolver(
        semantic_edge_policy=(
            MESSAGE_SEMANTIC_EDGE_POLICY_IGNORE
        )
    )
    composer = EdgeMessageComposer()
    diagnostics = MessageBuilderDiagnostics()

    builder = EdgeMessageBuilder(
        coefficient_resolver=resolver,
        message_composer=composer,
        diagnostics=diagnostics,
    )

    assert builder.coefficient_resolver is resolver
    assert builder.message_composer is composer
    assert builder.diagnostics is diagnostics
    assert (
        builder.relation_state_gather
        is composer.relation_state_gather
    )


def test_builder_rejects_wrong_component_types() -> None:
    resolver = MessageCoefficientResolver()
    composer = EdgeMessageComposer()

    with pytest.raises(
        TypeError,
        match="coefficient_resolver",
    ):
        EdgeMessageBuilder(
            coefficient_resolver=object(),  # type: ignore[arg-type]
            message_composer=composer,
        )

    with pytest.raises(
        TypeError,
        match="message_composer",
    ):
        EdgeMessageBuilder(
            coefficient_resolver=resolver,
            message_composer=object(),  # type: ignore[arg-type]
        )

    with pytest.raises(
        TypeError,
        match="diagnostics",
    ):
        EdgeMessageBuilder(
            coefficient_resolver=resolver,
            message_composer=composer,
            diagnostics=object(),  # type: ignore[arg-type]
        )


def test_from_policy_rejects_nonboolean_diagnostics_flag() -> None:
    with pytest.raises(
        TypeError,
        match="diagnostics_enabled",
    ):
        EdgeMessageBuilder.from_policy(
            diagnostics_enabled=1,  # type: ignore[arg-type]
        )


# =============================================================================
# Parameter, buffer, state, and architecture identity
# =============================================================================


@pytest.mark.parametrize(
    "diagnostics_enabled",
    (
        False,
        True,
    ),
)
def test_complete_builder_is_parameter_and_buffer_free(
    diagnostics_enabled: bool,
) -> None:
    builder = _builder(
        diagnostics_enabled=diagnostics_enabled,
    )

    _assert_parameter_and_buffer_free(builder)
    assert builder.parameter_count == 0
    assert builder.trainable_parameter_count == 0
    assert builder.buffer_count == 0
    assert builder.parameter_fingerprint is None
    builder.assert_parameter_free()


def test_empty_state_dict_roundtrip() -> None:
    first = _builder(
        diagnostics_enabled=True,
    )
    second = _builder(
        diagnostics_enabled=True,
    )

    state = first.state_dict()
    assert len(state) == 0

    result = second.load_state_dict(
        state,
        strict=True,
    )
    assert result.missing_keys == []
    assert result.unexpected_keys == []


def test_numerical_architecture_is_deterministic() -> None:
    first = _builder()
    second = _builder()

    assert (
        first.numerical_architecture_dict()
        == second.numerical_architecture_dict()
    )
    assert (
        first.architecture_fingerprint()
        == second.architecture_fingerprint()
    )
    assert (
        first.architecture_dict()
        == first.numerical_architecture_dict()
    )


def test_diagnostics_configuration_does_not_change_numerical_fingerprint() -> None:
    without_diagnostics = _builder(
        diagnostics_enabled=False,
    )
    with_diagnostics = _builder(
        diagnostics_enabled=True,
        include_per_relation=False,
        include_per_graph=False,
        thresholds=MessageBuilderDiagnosticThresholds(
            near_zero_absolute=1e-5,
            large_absolute_coefficient=3.0,
            large_message_l2_norm=7.0,
            high_near_zero_fraction=0.5,
            high_zero_message_fraction=0.5,
        ),
    )

    assert (
        without_diagnostics.architecture_fingerprint()
        == with_diagnostics.architecture_fingerprint()
    )
    assert (
        without_diagnostics.numerical_architecture_dict()
        == with_diagnostics.numerical_architecture_dict()
    )


def test_semantic_policy_changes_numerical_fingerprint() -> None:
    ignored = _builder(
        semantic_policy=(
            MESSAGE_SEMANTIC_EDGE_POLICY_IGNORE
        )
    )
    consumed = _builder(
        semantic_policy=(
            MESSAGE_SEMANTIC_EDGE_POLICY_USE_SOURCE_GRAPH
        )
    )

    assert (
        ignored.architecture_fingerprint()
        != consumed.architecture_fingerprint()
    )


def test_numerical_architecture_separates_responsibilities() -> None:
    builder = _builder()
    architecture = builder.numerical_architecture_dict()

    assert architecture["factor_order"] == list(
        MESSAGE_FACTOR_ORDER
    )
    assert architecture["output_schema"] == (
        "EdgeMessageOutput"
    )
    assert architecture["parameter_free"] is True
    assert architecture["buffer_free"] is True
    assert architecture["aggregation_owned_here"] is False
    assert architecture["residual_update_owned_here"] is False
    assert architecture["dropout_owned_here"] is False
    assert architecture["layer_normalization_owned_here"] is False
    assert architecture["claims_causal_importance"] is False
    assert architecture["claims_explanation_faithfulness"] is False


def test_full_runtime_dict_includes_non_numerical_reporting_settings() -> None:
    builder = _builder(
        diagnostics_enabled=True,
        include_per_relation=False,
    )
    runtime = builder.full_runtime_dict()

    assert runtime["numerical_architecture"] == (
        builder.numerical_architecture_dict()
    )
    assert runtime[
        "numerical_architecture_fingerprint"
    ] == builder.architecture_fingerprint()
    assert runtime["diagnostics_enabled"] is True
    assert runtime["diagnostics_architecture"] == (
        builder.diagnostics.architecture_dict()
    )
    assert runtime["parameter_count"] == 0
    assert runtime["buffer_count"] == 0


def test_repr_exposes_bounded_policy() -> None:
    builder = _builder(
        semantic_policy=(
            MESSAGE_SEMANTIC_EDGE_POLICY_USE_SOURCE_GRAPH
        ),
        diagnostics_enabled=True,
    )
    text = repr(builder)

    assert "use_source_graph" in text
    assert "diagnostics_enabled=True" in text
    assert "aggregation_owned_here=False" in text
    assert "parameter_free=True" in text


# =============================================================================
# Internal-stage execution
# =============================================================================


def test_run_stages_returns_exact_two_stage_chain() -> None:
    builder = _builder(
        semantic_policy=(
            MESSAGE_SEMANTIC_EDGE_POLICY_USE_SOURCE_GRAPH
        )
    )
    (
        inputs,
        transform,
        normalization,
        gate,
        attention,
    ) = _upstream()

    stages = builder.run_stages(
        relation_transform=transform,
        edge_normalization=normalization,
        relation_gate=gate,
        edge_attention=attention,
        source_inputs=inputs,
    )

    assert isinstance(stages, tuple)
    assert len(stages) == 2

    resolved, composition = stages
    assert isinstance(
        resolved,
        ResolvedMessageCoefficients,
    )
    assert isinstance(
        composition,
        MessageCompositionOutput,
    )
    assert composition.resolved_coefficients is resolved
    assert composition.relation_transform is transform
    assert resolved.source_inputs is inputs


def test_run_stages_preserves_all_enabled_source_objects() -> None:
    builder = _builder(
        semantic_policy=(
            MESSAGE_SEMANTIC_EDGE_POLICY_USE_SOURCE_GRAPH
        )
    )
    (
        inputs,
        transform,
        normalization,
        gate,
        attention,
    ) = _upstream()

    resolved, composition = builder.run_stages(
        relation_transform=transform,
        edge_normalization=normalization,
        relation_gate=gate,
        edge_attention=attention,
        source_inputs=inputs,
    )

    assert resolved.edge_normalization is normalization
    assert resolved.relation_gate is gate
    assert resolved.edge_attention is attention
    assert resolved.structural_normalization_factor is (
        normalization.coefficients
    )
    assert resolved.relation_gate_factor is (
        gate.edge_gate_values
    )
    assert resolved.edge_attention_factor is (
        attention.edge_weights
    )
    assert resolved.semantic_edge_weight is (
        inputs.source_graph.semantic_edge_weight
    )
    assert composition.relation_transform is transform
    assert (
        composition.combined_coefficient
        is resolved.combined_coefficient
    )


@pytest.mark.parametrize(
    (
        "gate_enabled",
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
def test_complete_builder_all_optional_factor_combinations(
    gate_enabled: bool,
    attention_enabled: bool,
    semantic_policy: str,
) -> None:
    builder = _builder(
        semantic_policy=semantic_policy,
    )
    (
        run,
        inputs,
        transform,
        normalization,
        gate,
        attention,
    ) = _run(
        builder,
        gate_enabled=gate_enabled,
        attention_enabled=attention_enabled,
    )

    semantic = (
        inputs.source_graph.semantic_edge_weight
        if semantic_policy
        == MESSAGE_SEMANTIC_EDGE_POLICY_USE_SOURCE_GRAPH
        else None
    )
    expected = _manual_messages(
        transform=transform,
        normalization=normalization,
        gate=gate,
        attention=attention,
        semantic_weight=semantic,
    )

    torch.testing.assert_close(
        run.public_output.edge_messages,
        expected,
    )
    assert (
        run.public_output.relation_gate
        is gate
    )
    assert (
        run.public_output.edge_attention
        is attention
    )
    assert (
        run.public_output.semantic_edge_weight
        is semantic
    )


def test_disabled_optional_factors_are_none_publicly_and_one_internally() -> None:
    builder = _builder()
    (
        run,
        _inputs_value,
        _transform,
        normalization,
        _gate,
        _attention,
    ) = _run(
        builder,
        gate_enabled=False,
        attention_enabled=False,
    )

    resolved = run.resolved_coefficients

    assert resolved.relation_gate is None
    assert resolved.edge_attention is None
    assert resolved.semantic_edge_weight is None
    assert torch.equal(
        resolved.relation_gate_factor,
        torch.ones_like(
            normalization.coefficients
        ),
    )
    assert torch.equal(
        resolved.edge_attention_factor,
        torch.ones_like(
            normalization.coefficients
        ),
    )
    assert torch.equal(
        resolved.semantic_edge_factor,
        torch.ones_like(
            normalization.coefficients
        ),
    )
    assert run.public_output.relation_gate is None
    assert run.public_output.edge_attention is None
    assert run.public_output.semantic_edge_weight is None


def test_enabled_uniform_attention_differs_from_disabled_attention() -> None:
    builder = _builder()
    (
        inputs,
        transform,
        normalization,
        _gate,
        attention,
    ) = _upstream(
        gate_enabled=False,
        attention_enabled=True,
    )

    enabled = builder(
        relation_transform=transform,
        edge_normalization=normalization,
        relation_gate=None,
        edge_attention=attention,
        source_inputs=inputs,
    )
    disabled = builder(
        relation_transform=transform,
        edge_normalization=normalization,
        relation_gate=None,
        edge_attention=None,
        source_inputs=inputs,
    )

    assert not torch.equal(
        enabled.edge_messages,
        disabled.edge_messages,
    )


# =============================================================================
# Complete run and public output assembly
# =============================================================================


def test_run_complete_returns_exact_public_chain() -> None:
    builder = _builder(
        semantic_policy=(
            MESSAGE_SEMANTIC_EDGE_POLICY_USE_SOURCE_GRAPH
        )
    )
    (
        run,
        inputs,
        transform,
        normalization,
        gate,
        attention,
    ) = _run(builder)

    assert isinstance(run, MessageBuilderRun)
    assert isinstance(
        run.public_output,
        EdgeMessageOutput,
    )
    assert (
        run.composition_output.resolved_coefficients
        is run.resolved_coefficients
    )
    assert (
        run.public_output.edge_messages
        is run.composition_output.edge_messages
    )
    assert (
        run.public_output.relation_transform
        is transform
    )
    assert (
        run.public_output.edge_normalization
        is normalization
    )
    assert run.public_output.relation_gate is gate
    assert run.public_output.edge_attention is attention
    assert (
        run.public_output.semantic_edge_weight
        is inputs.source_graph.semantic_edge_weight
    )
    assert run.public_output.source_inputs is inputs
    assert (
        run.public_output.encoder_architecture_fingerprint
        == builder.architecture_fingerprint()
    )


def test_forward_returns_public_output_only() -> None:
    builder = _builder()
    (
        inputs,
        transform,
        normalization,
        gate,
        attention,
    ) = _upstream()

    output = builder(
        relation_transform=transform,
        edge_normalization=normalization,
        relation_gate=gate,
        edge_attention=attention,
        source_inputs=inputs,
    )

    assert isinstance(output, EdgeMessageOutput)
    assert not isinstance(output, MessageBuilderRun)


def test_forward_and_run_complete_are_numerically_equivalent() -> None:
    builder = _builder(
        semantic_policy=(
            MESSAGE_SEMANTIC_EDGE_POLICY_USE_SOURCE_GRAPH
        )
    )
    (
        inputs,
        transform,
        normalization,
        gate,
        attention,
    ) = _upstream()

    forward_output = builder(
        relation_transform=transform,
        edge_normalization=normalization,
        relation_gate=gate,
        edge_attention=attention,
        source_inputs=inputs,
    )
    run = builder.run_complete(
        relation_transform=transform,
        edge_normalization=normalization,
        relation_gate=gate,
        edge_attention=attention,
        source_inputs=inputs,
    )

    torch.testing.assert_close(
        forward_output.edge_messages,
        run.public_output.edge_messages,
    )
    assert (
        forward_output.encoder_architecture_fingerprint
        == run.public_output.encoder_architecture_fingerprint
    )


def test_functional_execution_helpers_match_methods() -> None:
    builder = _builder()
    (
        inputs,
        transform,
        normalization,
        gate,
        attention,
    ) = _upstream()

    output = run_edge_message_builder(
        builder,
        relation_transform=transform,
        edge_normalization=normalization,
        relation_gate=gate,
        edge_attention=attention,
        source_inputs=inputs,
    )
    alias_output = run_message_builder(
        builder,
        relation_transform=transform,
        edge_normalization=normalization,
        relation_gate=gate,
        edge_attention=attention,
        source_inputs=inputs,
    )
    stages = run_edge_message_builder_stages(
        builder,
        relation_transform=transform,
        edge_normalization=normalization,
        relation_gate=gate,
        edge_attention=attention,
        source_inputs=inputs,
    )
    alias_stages = run_message_builder_stages(
        builder,
        relation_transform=transform,
        edge_normalization=normalization,
        relation_gate=gate,
        edge_attention=attention,
        source_inputs=inputs,
    )

    torch.testing.assert_close(
        output.edge_messages,
        alias_output.edge_messages,
    )
    torch.testing.assert_close(
        stages[1].edge_messages,
        alias_stages[1].edge_messages,
    )


def test_assemble_edge_message_output_preserves_exact_objects() -> None:
    builder = _builder()
    (
        inputs,
        transform,
        normalization,
        gate,
        attention,
    ) = _upstream()
    resolved, composition = builder.run_stages(
        relation_transform=transform,
        edge_normalization=normalization,
        relation_gate=gate,
        edge_attention=attention,
        source_inputs=inputs,
    )

    output = assemble_edge_message_output(
        composition_output=composition,
        encoder_architecture_fingerprint=(
            "manual-message-builder-architecture"
        ),
    )

    assert output.edge_messages is composition.edge_messages
    assert output.relation_transform is transform
    assert output.edge_normalization is normalization
    assert output.relation_gate is gate
    assert output.edge_attention is attention
    assert (
        output.semantic_edge_weight
        is resolved.semantic_edge_weight
    )


def test_complete_run_validator_accepts_exact_chain() -> None:
    builder = _builder()
    run, inputs, *_ = _run(builder)

    validate_complete_message_builder_run(
        resolved_coefficients=(
            run.resolved_coefficients
        ),
        composition_output=(
            run.composition_output
        ),
        public_output=run.public_output,
        source_inputs=inputs,
        encoder_architecture_fingerprint=(
            builder.architecture_fingerprint()
        ),
    )


def test_complete_run_validator_rejects_public_output_from_distinct_run() -> None:
    builder = _builder()
    (
        inputs,
        transform,
        normalization,
        gate,
        attention,
    ) = _upstream()

    first = builder.run_complete(
        relation_transform=transform,
        edge_normalization=normalization,
        relation_gate=gate,
        edge_attention=attention,
        source_inputs=inputs,
    )
    second = builder.run_complete(
        relation_transform=transform,
        edge_normalization=normalization,
        relation_gate=gate,
        edge_attention=attention,
        source_inputs=inputs,
    )

    with pytest.raises(
        ValueError,
        match="exact edge_messages tensor",
    ):
        validate_complete_message_builder_run(
            resolved_coefficients=(
                first.resolved_coefficients
            ),
            composition_output=(
                first.composition_output
            ),
            public_output=(
                second.public_output
            ),
        )


# =============================================================================
# Semantic policy and lineage failures
# =============================================================================


def test_use_source_graph_requires_semantic_weight() -> None:
    builder = _builder(
        semantic_policy=(
            MESSAGE_SEMANTIC_EDGE_POLICY_USE_SOURCE_GRAPH
        )
    )
    (
        inputs,
        transform,
        normalization,
        gate,
        attention,
    ) = _upstream(
        semantic_edge_weight=None,
    )

    with pytest.raises(
        ValueError,
        match="requires source_graph.semantic_edge_weight",
    ):
        builder(
            relation_transform=transform,
            edge_normalization=normalization,
            relation_gate=gate,
            edge_attention=attention,
            source_inputs=inputs,
        )


def test_ignore_policy_does_not_consume_present_graph_weight() -> None:
    builder = _builder(
        semantic_policy=(
            MESSAGE_SEMANTIC_EDGE_POLICY_IGNORE
        )
    )
    run, inputs, *_ = _run(builder)

    assert (
        inputs.source_graph.semantic_edge_weight
        is not None
    )
    assert (
        run.resolved_coefficients.semantic_edge_weight
        is None
    )
    assert run.public_output.semantic_edge_weight is None


def test_crossed_transform_and_normalization_lineage_is_rejected() -> None:
    first = _upstream(
        source_fingerprint="first",
    )
    second = _upstream(
        source_fingerprint="second",
    )
    builder = _builder()

    with pytest.raises(
        ValueError,
        match="exact same FunctionalMessagePassingInputs",
    ):
        builder(
            relation_transform=first[1],
            edge_normalization=second[2],
        )


def test_crossed_gate_lineage_is_rejected() -> None:
    first = _upstream(
        source_fingerprint="first",
    )
    second = _upstream(
        source_fingerprint="second",
    )
    builder = _builder()

    with pytest.raises(
        ValueError,
        match="exact same FunctionalMessagePassingInputs",
    ):
        builder(
            relation_transform=first[1],
            edge_normalization=first[2],
            relation_gate=second[3],
        )


def test_crossed_attention_lineage_is_rejected() -> None:
    first = _upstream(
        source_fingerprint="first",
    )
    second = _upstream(
        source_fingerprint="second",
    )
    builder = _builder()

    with pytest.raises(
        ValueError,
        match="exact same FunctionalMessagePassingInputs",
    ):
        builder(
            relation_transform=first[1],
            edge_normalization=first[2],
            edge_attention=second[4],
        )


def test_explicit_wrong_source_inputs_is_rejected() -> None:
    first = _upstream(
        source_fingerprint="first",
    )
    second = _upstream(
        source_fingerprint="second",
    )
    builder = _builder()

    with pytest.raises(
        ValueError,
        match="exact supplied source_inputs",
    ):
        builder(
            relation_transform=first[1],
            edge_normalization=first[2],
            source_inputs=second[0],
        )


def test_functional_helpers_reject_wrong_builder_type() -> None:
    (
        inputs,
        transform,
        normalization,
        gate,
        attention,
    ) = _upstream()

    with pytest.raises(
        TypeError,
        match="EdgeMessageBuilder",
    ):
        run_edge_message_builder(
            object(),  # type: ignore[arg-type]
            relation_transform=transform,
            edge_normalization=normalization,
            relation_gate=gate,
            edge_attention=attention,
            source_inputs=inputs,
        )

    with pytest.raises(
        TypeError,
        match="EdgeMessageBuilder",
    ):
        run_edge_message_builder_stages(
            object(),  # type: ignore[arg-type]
            relation_transform=transform,
            edge_normalization=normalization,
            relation_gate=gate,
            edge_attention=attention,
            source_inputs=inputs,
        )


def test_assemble_public_output_rejects_blank_fingerprint() -> None:
    builder = _builder()
    run, *_ = _run(builder)

    with pytest.raises(
        ValueError,
        match="non-empty",
    ):
        assemble_edge_message_output(
            composition_output=(
                run.composition_output
            ),
            encoder_architecture_fingerprint=" ",
        )


# =============================================================================
# Empty edges, dtype, evaluation mode, autograd, and CUDA
# =============================================================================


def test_complete_builder_supports_empty_edges() -> None:
    builder = _builder(
        semantic_policy=(
            MESSAGE_SEMANTIC_EDGE_POLICY_USE_SOURCE_GRAPH
        ),
        diagnostics_enabled=True,
    )
    run, inputs, *_ = _run(
        builder,
        empty_edges=True,
        gate_enabled=False,
        attention_enabled=False,
    )

    assert run.public_output.edge_messages.shape == (
        0,
        HIDDEN_DIM,
    )
    assert run.public_output.edge_messages.shape[0] == 0
    assert run.public_output.source_inputs is inputs

    report = builder.diagnostic_report(
        run=run
    )
    alert_codes = {
        alert["code"]
        for alert in report[
            MESSAGE_BUILDER_DIAGNOSTIC_SECTION_ALERTS
        ]
    }
    assert "empty_edge_set" in alert_codes


def test_complete_builder_supports_float64() -> None:
    builder = _builder(
        semantic_policy=(
            MESSAGE_SEMANTIC_EDGE_POLICY_USE_SOURCE_GRAPH
        )
    )
    run, *_ = _run(
        builder,
        dtype=torch.float64,
    )

    assert run.public_output.edge_messages.dtype == torch.float64
    assert (
        run.public_output.edge_messages.dtype
        == torch.float64
    )


def test_train_and_eval_modes_are_numerically_identical() -> None:
    builder = _builder(
        semantic_policy=(
            MESSAGE_SEMANTIC_EDGE_POLICY_USE_SOURCE_GRAPH
        )
    )
    (
        inputs,
        transform,
        normalization,
        gate,
        attention,
    ) = _upstream()

    builder.train()
    train_output = builder(
        relation_transform=transform,
        edge_normalization=normalization,
        relation_gate=gate,
        edge_attention=attention,
        source_inputs=inputs,
    )

    builder.eval()
    eval_output = builder(
        relation_transform=transform,
        edge_normalization=normalization,
        relation_gate=gate,
        edge_attention=attention,
        source_inputs=inputs,
    )

    torch.testing.assert_close(
        train_output.edge_messages,
        eval_output.edge_messages,
    )
    assert (
        train_output.encoder_architecture_fingerprint
        == eval_output.encoder_architecture_fingerprint
    )


def test_complete_builder_preserves_autograd() -> None:
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
    builder = _builder(
        semantic_policy=(
            MESSAGE_SEMANTIC_EDGE_POLICY_USE_SOURCE_GRAPH
        )
    )
    (
        inputs,
        transform,
        normalization,
        _gate,
        _attention,
    ) = _upstream(
        semantic_edge_weight=semantic,
        transformed_state=transformed,
        structural_coefficients=structural,
        gate_enabled=False,
        attention_enabled=False,
    )

    output = builder(
        relation_transform=transform,
        edge_normalization=normalization,
        relation_gate=None,
        edge_attention=None,
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


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is not available.",
)
def test_complete_builder_supports_cuda() -> None:
    builder = _builder(
        semantic_policy=(
            MESSAGE_SEMANTIC_EDGE_POLICY_USE_SOURCE_GRAPH
        )
    ).to("cuda")
    run, *_ = _run(
        builder,
        device="cuda",
    )

    assert run.public_output.edge_messages.device.type == "cuda"
    assert (
        run.public_output.edge_messages.device.type
        == "cuda"
    )


# =============================================================================
# Explicit diagnostics integration
# =============================================================================


def test_forward_does_not_require_or_return_diagnostics() -> None:
    builder = _builder(
        diagnostics_enabled=True,
    )
    (
        inputs,
        transform,
        normalization,
        gate,
        attention,
    ) = _upstream()

    output = builder(
        relation_transform=transform,
        edge_normalization=normalization,
        relation_gate=gate,
        edge_attention=attention,
        source_inputs=inputs,
    )

    assert isinstance(output, EdgeMessageOutput)
    assert not isinstance(
        output,
        MessageBuilderRunWithDiagnostics,
    )


def test_diagnostic_report_requires_configured_diagnostics() -> None:
    builder = _builder(
        diagnostics_enabled=False,
    )
    run, *_ = _run(builder)

    with pytest.raises(
        RuntimeError,
        match="Diagnostics are not configured",
    ):
        builder.diagnostic_report(
            run=run
        )


def test_diagnostic_report_rejects_wrong_run_type() -> None:
    builder = _builder(
        diagnostics_enabled=True,
    )

    with pytest.raises(
        TypeError,
        match="MessageBuilderRun",
    ):
        builder.diagnostic_report(
            run=object(),  # type: ignore[arg-type]
        )


def test_explicit_diagnostic_report_is_complete_and_tensor_free() -> None:
    builder = _builder(
        semantic_policy=(
            MESSAGE_SEMANTIC_EDGE_POLICY_USE_SOURCE_GRAPH
        ),
        diagnostics_enabled=True,
    )
    run, *_ = _run(builder)

    report = builder.diagnostic_report(
        run=run
    )

    assert report["schema_version"] == (
        MESSAGE_BUILDER_DIAGNOSTICS_SCHEMA_VERSION
    )
    assert report["interpretation"] == (
        MESSAGE_BUILDER_DIAGNOSTICS_INTERPRETATION
    )
    assert (
        MESSAGE_BUILDER_DIAGNOSTIC_SECTION_STAGE_SUMMARIES
        in report
    )
    assert (
        MESSAGE_BUILDER_DIAGNOSTIC_SECTION_GLOBAL
        in report
    )
    assert (
        MESSAGE_BUILDER_DIAGNOSTIC_SECTION_BY_RELATION
        in report
    )
    assert (
        MESSAGE_BUILDER_DIAGNOSTIC_SECTION_BY_GRAPH
        in report
    )
    assert (
        MESSAGE_BUILDER_DIAGNOSTIC_SECTION_LINEAGE
        in report
    )
    assert (
        MESSAGE_BUILDER_DIAGNOSTIC_SECTION_ALERTS
        in report
    )
    assert "public_output" in report

    _assert_tensor_free(report)
    json.dumps(
        report,
        sort_keys=True,
        allow_nan=False,
    )


def test_diagnostic_report_exact_relation_and_graph_coverage() -> None:
    builder = _builder(
        diagnostics_enabled=True,
    )
    run, *_ = _run(builder)
    report = builder.diagnostic_report(
        run=run
    )

    by_relation = report[
        MESSAGE_BUILDER_DIAGNOSTIC_SECTION_BY_RELATION
    ]
    by_graph = report[
        MESSAGE_BUILDER_DIAGNOSTIC_SECTION_BY_GRAPH
    ]

    assert len(by_relation) == RELATIONS
    assert len(by_graph) == GRAPHS
    assert [
        item["relation_name"]
        for item in by_relation
    ] == list(RELATION_NAMES)
    assert [
        item["stable_relation_id"]
        for item in by_relation
    ] == list(STABLE_RELATION_IDS)
    assert by_relation[2][
        "is_control_relation"
    ] is True
    assert sum(
        item["edge_count"]
        for item in by_relation
    ) == EDGES
    assert sum(
        item["edge_count"]
        for item in by_graph
    ) == EDGES
    assert sum(
        item["node_count"]
        for item in by_graph
    ) == NODES


def test_diagnostic_slices_can_be_disabled() -> None:
    builder = _builder(
        diagnostics_enabled=True,
        include_per_relation=False,
        include_per_graph=False,
    )
    run, *_ = _run(builder)
    report = builder.diagnostic_report(
        run=run
    )

    assert report[
        MESSAGE_BUILDER_DIAGNOSTIC_SECTION_BY_RELATION
    ] == []
    assert report[
        MESSAGE_BUILDER_DIAGNOSTIC_SECTION_BY_GRAPH
    ] == []


def test_forward_with_diagnostics_returns_public_output_and_report() -> None:
    builder = _builder(
        diagnostics_enabled=True,
    )
    (
        inputs,
        transform,
        normalization,
        gate,
        attention,
    ) = _upstream()

    result = builder.forward_with_diagnostics(
        relation_transform=transform,
        edge_normalization=normalization,
        relation_gate=gate,
        edge_attention=attention,
        source_inputs=inputs,
    )

    assert isinstance(
        result,
        MessageBuilderRunWithDiagnostics,
    )
    assert isinstance(
        result.public_output,
        EdgeMessageOutput,
    )
    assert isinstance(
        result.diagnostic_report,
        dict,
    )
    assert result.diagnostic_report[
        "public_output"
    ][
        "exact_edge_messages_tensor_preserved"
    ] is True


def test_functional_internal_diagnostic_report_matches_module_shape() -> None:
    builder = _builder(
        diagnostics_enabled=True,
    )
    run, inputs, *_ = _run(builder)

    report = build_message_builder_diagnostic_report(
        relation_transform=(
            run.composition_output
            .relation_transform
        ),
        resolved_coefficients=(
            run.resolved_coefficients
        ),
        composition_output=(
            run.composition_output
        ),
        source_inputs=inputs,
    )

    assert len(
        report[
            MESSAGE_BUILDER_DIAGNOSTIC_SECTION_BY_RELATION
        ]
    ) == RELATIONS
    assert len(
        report[
            MESSAGE_BUILDER_DIAGNOSTIC_SECTION_BY_GRAPH
        ]
    ) == GRAPHS


def test_public_diagnostic_report_validates_public_identity() -> None:
    builder = _builder(
        diagnostics_enabled=True,
    )
    run, *_ = _run(builder)

    report = build_public_edge_message_diagnostic_report(
        public_output=run.public_output,
        composition_output=(
            run.composition_output
        ),
    )

    public_section = report["public_output"]
    assert public_section[
        "exact_edge_messages_tensor_preserved"
    ] is True
    assert public_section[
        "exact_relation_transform_preserved"
    ] is True
    assert public_section[
        "exact_edge_normalization_preserved"
    ] is True


def test_diagnostic_fingerprint_is_deterministic_and_validated() -> None:
    builder = _builder(
        diagnostics_enabled=True,
    )
    run, *_ = _run(builder)

    first = builder.diagnostic_report(
        run=run
    )
    second = builder.diagnostic_report(
        run=run
    )

    assert (
        first["report_fingerprint"]
        == second["report_fingerprint"]
    )
    assert first["report_fingerprint"] == (
        diagnostic_report_fingerprint(
            first
        )
    )

    validate_message_builder_diagnostic_report(
        first,
        expected_num_relations=RELATIONS,
        expected_num_graphs=GRAPHS,
    )


def test_diagnostic_validator_detects_tampering() -> None:
    builder = _builder(
        diagnostics_enabled=True,
    )
    run, *_ = _run(builder)
    report = builder.diagnostic_report(
        run=run
    )
    tampered = copy.deepcopy(report)
    tampered[
        MESSAGE_BUILDER_DIAGNOSTIC_SECTION_GLOBAL
    ]["num_edges"] += 1

    with pytest.raises(
        ValueError,
        match="fingerprint",
    ):
        validate_message_builder_diagnostic_report(
            tampered
        )


def test_diagnostic_validator_rejects_unsupported_scientific_claim() -> None:
    builder = _builder(
        diagnostics_enabled=True,
    )
    run, *_ = _run(builder)
    report = builder.diagnostic_report(
        run=run
    )
    tampered = copy.deepcopy(report)
    tampered.pop(
        "report_fingerprint",
        None,
    )
    tampered["scientific_claims"][
        "causal_importance"
    ] = True

    with pytest.raises(
        ValueError,
        match="unsupported scientific claims",
    ):
        validate_message_builder_diagnostic_report(
            tampered
        )


def test_diagnostic_lineage_preserves_complete_stage_identity() -> None:
    builder = _builder(
        diagnostics_enabled=True,
    )
    run, inputs, *_ = _run(builder)
    report = builder.diagnostic_report(
        run=run
    )
    lineage = report[
        MESSAGE_BUILDER_DIAGNOSTIC_SECTION_LINEAGE
    ]

    assert lineage[
        "source_inputs_lineage_fingerprint"
    ] == inputs.lineage_fingerprint()
    identity = lineage["exact_object_identity"]
    assert identity[
        "relation_transform_preserved"
    ] is True
    assert identity[
        "resolved_coefficients_preserved"
    ] is True
    assert identity[
        "source_inputs_preserved"
    ] is True
    assert identity[
        "transformed_state_tensor_preserved"
    ] is True
    assert identity[
        "combined_coefficient_tensor_preserved"
    ] is True


def test_signed_semantic_factor_produces_descriptive_alert() -> None:
    signed = torch.linspace(
        -1.0,
        1.0,
        EDGES,
    )
    builder = _builder(
        semantic_policy=(
            MESSAGE_SEMANTIC_EDGE_POLICY_USE_SOURCE_GRAPH
        ),
        diagnostics_enabled=True,
    )
    run, *_ = _run(
        builder,
        semantic_edge_weight=signed,
    )
    report = builder.diagnostic_report(
        run=run
    )
    codes = {
        alert["code"]
        for alert in report[
            MESSAGE_BUILDER_DIAGNOSTIC_SECTION_ALERTS
        ]
    }

    assert "signed_semantic_edge_factor" in codes


def test_zero_and_large_coefficients_produce_bounded_alerts() -> None:
    structural = torch.tensor(
        [0.0, 20.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    )
    thresholds = MessageBuilderDiagnosticThresholds(
        near_zero_absolute=1e-8,
        large_absolute_coefficient=10.0,
        large_message_l2_norm=1e9,
        high_near_zero_fraction=1.0,
        high_zero_message_fraction=1.0,
    )
    builder = _builder(
        diagnostics_enabled=True,
        thresholds=thresholds,
    )
    run, *_ = _run(
        builder,
        structural_coefficients=structural,
        gate_enabled=False,
        attention_enabled=False,
    )
    report = builder.diagnostic_report(
        run=run
    )
    codes = {
        alert["code"]
        for alert in report[
            MESSAGE_BUILDER_DIAGNOSTIC_SECTION_ALERTS
        ]
    }

    assert "zero_combined_coefficient" in codes
    assert "large_absolute_coefficient" in codes


def test_diagnostics_threshold_validation() -> None:
    with pytest.raises(
        ValueError,
        match="strictly positive",
    ):
        MessageBuilderDiagnosticThresholds(
            near_zero_absolute=0.0,
        )

    with pytest.raises(
        ValueError,
        match=r"\[0, 1\]",
    ):
        MessageBuilderDiagnosticThresholds(
            high_near_zero_fraction=1.1,
        )


def test_diagnostics_module_is_parameter_and_buffer_free() -> None:
    diagnostics = MessageBuilderDiagnostics()

    _assert_parameter_and_buffer_free(
        diagnostics
    )
    assert diagnostics.parameter_count == 0
    assert diagnostics.trainable_parameter_count == 0
    assert diagnostics.buffer_count == 0
    assert diagnostics.parameter_fingerprint is None
    diagnostics.assert_parameter_free()


# =============================================================================
# Boundary assertions
# =============================================================================


def test_public_and_internal_outputs_do_not_expose_aggregation() -> None:
    builder = _builder()
    run, *_ = _run(builder)

    for value in (
        run.resolved_coefficients,
        run.composition_output,
        run.public_output,
        builder,
    ):
        assert not hasattr(
            value,
            "aggregated_messages",
        )
        assert not hasattr(
            value,
            "target_updates",
        )


def test_diagnostics_never_claim_causality_or_faithfulness() -> None:
    builder = _builder(
        diagnostics_enabled=True,
    )
    run, *_ = _run(builder)
    report = builder.diagnostic_report(
        run=run
    )

    assert report["scientific_claims"] == {
        "causal_importance": False,
        "explanation_faithfulness": False,
        "uncertainty_calibration": False,
        "mechanistic_identifiability": False,
        "relation_necessity": False,
    }
