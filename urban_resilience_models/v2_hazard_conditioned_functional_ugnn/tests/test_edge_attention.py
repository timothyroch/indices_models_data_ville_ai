"""
Focused integration tests for complete edge-attention orchestration and package API.

Target repository path:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            tests/
                test_edge_attention.py

Implementations under test:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            functional_message_passing/
                edge_attention/
                    __init__.py
                    edge_attention.py

The suite tests the complete enabled edge-attention chain:

    score prediction
        -> exact target-node/relation grouped normalization
        -> arithmetic-mean head reduction
        -> public EdgeAttentionOutput

The lower-level score-function, normalization, and multihead suites establish
their isolated numerical contracts. This file focuses on cross-stage
composition, exact lineage, configuration construction, final public output,
package exports, end-to-end gradients, metamorphic behavior, and scientific
scope boundaries.

Scientific contracts frozen here
--------------------------------
- Edge attention is enabled routing within one exact relation mechanism at one
  target node.
- Relation gates and edge attention remain distinct.
- Group IDs are target-node + exact dense compiled relation.
- Every head is normalized independently within each exact group.
- Mean head reduction preserves group normalization.
- Enabled uniform attention produces reciprocal group-size weights.
- Disabled attention is represented outside this package by ``None`` and the
  multiplicative identity one; it is not silently replaced by uniform
  attention.
- Hazard-blind attention is invariant to hazard-query values.
- Hazard-conditioned attention can depend on the target hazard query.
- Exact relation order is frozen into the architecture.
- Edge attributes and semantic edge weights are not consumed by the bounded
  score function.
- The complete trainable parameter state belongs to the score stage;
  normalization and reduction remain parameter-free.
- The final output preserves the exact tensors and input object from its stage
  chain.
- Attention is a routing coefficient, not causal importance, calibrated
  uncertainty, or proof of head specialization.

Controlled upstream doubles are patched into
``functional_message_passing.schemas`` so failures remain localized to the
edge-attention package rather than graph loading, fusion, hazard encoding, or
registry construction.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass
from typing import Any

import pytest
import torch
from torch import nn

from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.config import (
    FunctionalMessagePassingConfig,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.constants import (
    ATTENTION_HEAD_REDUCTION_MAX,
    ATTENTION_HEAD_REDUCTION_MEAN,
    ATTENTION_HEAD_REDUCTION_WEIGHTED_MEAN,
    ATTENTION_MODE_HAZARD_BLIND,
    ATTENTION_MODE_HAZARD_CONDITIONED,
    ATTENTION_MODE_MULTIHEAD_HAZARD_CONDITIONED,
    ATTENTION_MODE_SEMANTIC_WEIGHT,
    ATTENTION_MODE_UNIFORM,
    ATTENTION_NORMALIZATION_TARGET_NODE,
    ATTENTION_NORMALIZATION_TARGET_NODE_RELATION,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.functional_message_passing import (
    edge_attention as edge_attention_package,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.functional_message_passing import (
    schemas as fmp_schemas,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.functional_message_passing.edge_attention.attention_normalization import (
    TargetNodeRelationAttentionNormalization,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.functional_message_passing.edge_attention.edge_attention import (
    EDGE_ATTENTION_GROUP_SEMANTICS,
    EDGE_ATTENTION_OPERATION_ORDER,
    EDGE_ATTENTION_ORCHESTRATOR_SCHEMA_VERSION,
    EDGE_ATTENTION_SCIENTIFIC_INTERPRETATION,
    EdgeAttention,
    EdgeAttentionModule,
    EdgeAttentionStages,
    FunctionalEdgeAttention,
    HazardConditionedEdgeAttention,
    assemble_edge_attention_output,
    build_edge_attention,
    build_edge_attention_module,
    run_edge_attention,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.functional_message_passing.edge_attention.multihead import (
    MeanAttentionHeadReduction,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.functional_message_passing.edge_attention.schemas import (
    AttentionHeadReductionOutput,
    AttentionNormalizationOutput,
    EdgeAttentionScoreOutput,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.functional_message_passing.edge_attention.score_functions import (
    AdditiveEdgeAttentionScoreFunction,
    UniformEdgeAttentionScoreFunction,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.functional_message_passing.schemas import (
    EdgeAttentionOutput,
    FunctionalMessagePassingInputs,
)


NODES = 5
EDGES = 7
GRAPHS = 2
RELATIONS = 3
HIDDEN_DIM = 4
QUERY_DIM = 3
SCORE_HIDDEN_DIM = 5
MULTIHEAD_COUNT = 3
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


class FakeHazardEmbeddingLookup:
    pass


class FakeNodeAlignedHazardEmbeddingLookup:
    def __init__(
        self,
        *,
        node_batch_index: torch.Tensor,
    ) -> None:
        self.node_batch_index = node_batch_index


class FakeHazardQueryEncoding:
    def __init__(
        self,
        *,
        query: torch.Tensor,
        source_embedding: object,
        lineage_fingerprint: str = "hazard-query-lineage",
    ) -> None:
        self.query = query
        self.source_embedding = source_embedding
        self.lineage_fingerprint = lineage_fingerprint

    @property
    def item_count(self) -> int:
        return int(self.query.shape[0])


class FakeCompiledHazardRelationPriors:
    pass


class FakeRelationRegistry:
    pass


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
    monkeypatch.setattr(
        fmp_schemas,
        "RelationRegistry",
        FakeRelationRegistry,
    )


# =============================================================================
# Controlled graph/input/configuration helpers
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
) -> torch.Tensor:
    # Exact target/relation groups:
    #   group 3  = target 1, relation 0: edges 0 and 1
    #   group 7  = target 2, relation 1: edge 2
    #   group 8  = target 2, relation 2: edge 3
    #   group 14 = target 4, relation 2: edges 4 and 5
    #   group 10 = target 3, relation 1: edge 6
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
) -> torch.Tensor:
    return torch.tensor(
        [0, 0, 1, 2, 2, 2, 1],
        dtype=torch.long,
        device=device,
    )


def _edge_batch_index(
    *,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
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
    offset: float = 0.0,
) -> torch.Tensor:
    state = (
        torch.arange(
            NODES * HIDDEN_DIM,
            dtype=dtype,
            device=device,
        )
        .reshape(NODES, HIDDEN_DIM)
        / 10.0
        + offset
    )
    state = state.detach().clone()

    if requires_grad:
        state.requires_grad_(True)

    return state


def _query_tensor(
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
    requires_grad: bool = False,
    offset: float = 0.0,
) -> torch.Tensor:
    query = (
        torch.arange(
            NODES * QUERY_DIM,
            dtype=dtype,
            device=device,
        )
        .reshape(NODES, QUERY_DIM)
        / 7.0
        + offset
    )
    query = query.detach().clone()

    if requires_grad:
        query.requires_grad_(True)

    return query


def _registry(
    *,
    names: tuple[str, ...] = RELATION_NAMES,
    stable_ids: tuple[int, ...] = STABLE_RELATION_IDS,
    fingerprint: str = "compiled-relation-registry",
) -> FakeCompiledRelationRegistry:
    controls = tuple(
        CONTROL_RELATIONS[index]
        if index < len(CONTROL_RELATIONS)
        else False
        for index in range(len(names))
    )
    return FakeCompiledRelationRegistry(
        names=names,
        stable_ids=stable_ids,
        controls=controls,
        fingerprint=fingerprint,
    )


def _graph(
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
    empty_edges: bool = False,
    permutation: torch.Tensor | None = None,
    edge_attributes: torch.Tensor | None = None,
    semantic_edge_weight: torch.Tensor | None = None,
) -> FakeUrbanGraphBatch:
    node_batch = _node_batch_index(
        device=device,
    )

    if empty_edges:
        edge_index = torch.empty(
            (2, 0),
            dtype=torch.long,
            device=device,
        )
        relations = torch.empty(
            (0,),
            dtype=torch.long,
            device=device,
        )
        edge_batch = torch.empty(
            (0,),
            dtype=torch.long,
            device=device,
        )
        resolved_attributes = (
            torch.empty(
                (0, 2),
                dtype=dtype,
                device=device,
            )
            if edge_attributes is None
            else edge_attributes
        )
        resolved_semantic_weight = (
            torch.empty(
                (0,),
                dtype=dtype,
                device=device,
            )
            if semantic_edge_weight is None
            else semantic_edge_weight
        )
    else:
        edge_index = _edge_index(
            device=device,
        )
        relations = _edge_relations(
            device=device,
        )
        edge_batch = _edge_batch_index(
            device=device,
        )
        resolved_attributes = (
            torch.arange(
                EDGES * 2,
                dtype=dtype,
                device=device,
            ).reshape(EDGES, 2)
            if edge_attributes is None
            else edge_attributes
        )
        resolved_semantic_weight = (
            torch.linspace(
                0.5,
                1.5,
                EDGES,
                dtype=dtype,
                device=device,
            )
            if semantic_edge_weight is None
            else semantic_edge_weight
        )

        if permutation is not None:
            edge_index = edge_index[
                :,
                permutation,
            ]
            relations = relations[
                permutation
            ]
            edge_batch = edge_batch[
                permutation
            ]
            resolved_attributes = (
                resolved_attributes[
                    permutation
                ]
            )
            resolved_semantic_weight = (
                resolved_semantic_weight[
                    permutation
                ]
            )

    return FakeUrbanGraphBatch(
        external_node_ids=_node_ids(),
        node_batch_index=node_batch,
        edge_index=edge_index,
        edge_relation_type=relations,
        edge_attributes=resolved_attributes,
        semantic_edge_weight=resolved_semantic_weight,
        edge_batch_index=edge_batch,
    )


def _inputs(
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
    state: torch.Tensor | None = None,
    query: torch.Tensor | None = None,
    with_hazard_query: bool = True,
    registry: FakeCompiledRelationRegistry | None = None,
    empty_edges: bool = False,
    permutation: torch.Tensor | None = None,
    edge_attributes: torch.Tensor | None = None,
    semantic_edge_weight: torch.Tensor | None = None,
) -> FunctionalMessagePassingInputs:
    node_batch = _node_batch_index(
        device=device,
    )
    resolved_state = (
        _state_tensor(
            dtype=dtype,
            device=device,
        )
        if state is None
        else state
    )

    node_state = FakeNodeStateFusionOutput(
        fused_state=resolved_state,
        alignment=FakeAlignment(
            item_ids=_node_ids(),
            node_batch_index=node_batch,
            graph_count=GRAPHS,
        ),
    )

    if with_hazard_query:
        resolved_query = (
            _query_tensor(
                dtype=dtype,
                device=device,
            )
            if query is None
            else query
        )
        hazard_query = FakeHazardQueryEncoding(
            query=resolved_query,
            source_embedding=(
                FakeNodeAlignedHazardEmbeddingLookup(
                    node_batch_index=node_batch,
                )
            ),
        )
    else:
        hazard_query = None

    return FunctionalMessagePassingInputs(
        source_graph=_graph(
            dtype=dtype,
            device=device,
            empty_edges=empty_edges,
            permutation=permutation,
            edge_attributes=edge_attributes,
            semantic_edge_weight=semantic_edge_weight,
        ),
        node_state=node_state,
        compiled_relation_registry=(
            _registry()
            if registry is None
            else registry
        ),
        hazard_query=hazard_query,
        source_fingerprint="edge-attention-orchestrator-test-input",
    )


def _config(
    *,
    mode: str,
    heads: int = 1,
    enabled: bool = True,
    attention_enabled: bool = True,
    normalization: str = (
        ATTENTION_NORMALIZATION_TARGET_NODE_RELATION
    ),
    reduction: str = (
        ATTENTION_HEAD_REDUCTION_MEAN
    ),
) -> FunctionalMessagePassingConfig:
    return FunctionalMessagePassingConfig(
        enabled=enabled,
        attention_enabled=attention_enabled,
        attention_mode=mode,
        attention_heads=heads,
        attention_normalization=normalization,
        attention_head_reduction=reduction,
    )


def _module(
    inputs: FunctionalMessagePassingInputs,
    *,
    mode: str = ATTENTION_MODE_HAZARD_CONDITIONED,
    heads: int = 1,
) -> EdgeAttention:
    return EdgeAttention.from_config(
        config=_config(
            mode=mode,
            heads=heads,
        ),
        source_inputs=inputs,
        score_hidden_dim=SCORE_HIDDEN_DIM,
    )


def _fill_score_parameters(
    module: EdgeAttention,
    value: float,
) -> None:
    with torch.no_grad():
        for parameter in (
            module
            .score_function
            .parameters()
        ):
            parameter.fill_(value)


def _group_sums(
    edge_values: torch.Tensor,
    group_ids: torch.Tensor,
    *,
    num_groups: int,
) -> torch.Tensor:
    if edge_values.ndim == 1:
        output = torch.zeros(
            num_groups,
            dtype=edge_values.dtype,
            device=edge_values.device,
        )
    else:
        output = torch.zeros(
            (
                num_groups,
                int(edge_values.shape[1]),
            ),
            dtype=edge_values.dtype,
            device=edge_values.device,
        )

    if edge_values.numel() > 0:
        output.index_add_(
            0,
            group_ids,
            edge_values,
        )

    return output


def _stage_chain(
    module: EdgeAttention,
    inputs: FunctionalMessagePassingInputs,
) -> EdgeAttentionStages:
    return module.compute_stages(
        inputs
    )


# =============================================================================
# Package API and import-resolution tests
# =============================================================================


def test_edge_attention_import_resolves_to_package() -> None:
    assert hasattr(
        edge_attention_package,
        "__path__",
    )
    assert edge_attention_package.__name__.endswith(
        ".functional_message_passing.edge_attention"
    )


def test_package_all_has_no_duplicates_and_all_names_resolve() -> None:
    exported = tuple(
        edge_attention_package.__all__
    )

    assert exported
    assert len(exported) == len(set(exported))

    for name in exported:
        assert isinstance(name, str)
        assert name
        assert hasattr(
            edge_attention_package,
            name,
        ), name


def test_package_exports_core_orchestrator_symbols() -> None:
    expected = {
        "EdgeAttention",
        "EdgeAttentionModule",
        "FunctionalEdgeAttention",
        "HazardConditionedEdgeAttention",
        "EdgeAttentionOutput",
        "EdgeAttentionScoreOutput",
        "AttentionNormalizationOutput",
        "AttentionHeadReductionOutput",
        "assemble_edge_attention_output",
        "build_edge_attention",
        "build_edge_attention_module",
        "run_edge_attention",
    }

    assert expected.issubset(
        set(edge_attention_package.__all__)
    )


def test_package_export_identities() -> None:
    assert edge_attention_package.EdgeAttention is (
        EdgeAttention
    )
    assert edge_attention_package.EdgeAttentionOutput is (
        EdgeAttentionOutput
    )
    assert (
        edge_attention_package
        .EdgeAttentionScoreOutput
        is EdgeAttentionScoreOutput
    )
    assert (
        edge_attention_package
        .AttentionNormalizationOutput
        is AttentionNormalizationOutput
    )
    assert (
        edge_attention_package
        .AttentionHeadReductionOutput
        is AttentionHeadReductionOutput
    )
    assert (
        edge_attention_package
        .build_edge_attention
        is build_edge_attention
    )


def test_orchestrator_aliases_are_exact() -> None:
    assert EdgeAttentionModule is EdgeAttention
    assert FunctionalEdgeAttention is EdgeAttention
    assert HazardConditionedEdgeAttention is EdgeAttention
    assert build_edge_attention_module is (
        build_edge_attention
    )
    assert run_edge_attention is (
        EdgeAttention.forward
    )


def test_package_does_not_export_private_names_through_all() -> None:
    assert all(
        not name.startswith("_")
        for name in edge_attention_package.__all__
    )


# =============================================================================
# Public constants and scientific identity
# =============================================================================


def test_orchestrator_public_constants() -> None:
    assert isinstance(
        EDGE_ATTENTION_ORCHESTRATOR_SCHEMA_VERSION,
        str,
    )
    assert (
        EDGE_ATTENTION_ORCHESTRATOR_SCHEMA_VERSION
        .strip()
    )
    assert EDGE_ATTENTION_GROUP_SEMANTICS == (
        "target_node_exact_compiled_relation"
    )
    assert EDGE_ATTENTION_SCIENTIFIC_INTERPRETATION == (
        "within_relation_incoming_edge_routing"
    )


def test_operation_order_is_complete_and_unique() -> None:
    assert EDGE_ATTENTION_OPERATION_ORDER == (
        "validate_functional_message_passing_inputs",
        "predict_edge_attention_logits",
        "construct_exact_target_node_relation_groups",
        "normalize_logits_independently_per_head",
        "reduce_attention_heads_by_arithmetic_mean",
        "validate_exact_stage_lineage",
        "construct_public_edge_attention_output",
    )
    assert len(
        EDGE_ATTENTION_OPERATION_ORDER
    ) == len(
        set(EDGE_ATTENTION_OPERATION_ORDER)
    )


# =============================================================================
# Construction and component identity
# =============================================================================


@pytest.mark.parametrize(
    (
        "mode",
        "heads",
        "score_type",
        "uses_hazard",
        "parameterized",
    ),
    (
        (
            ATTENTION_MODE_UNIFORM,
            1,
            UniformEdgeAttentionScoreFunction,
            False,
            False,
        ),
        (
            ATTENTION_MODE_HAZARD_BLIND,
            1,
            AdditiveEdgeAttentionScoreFunction,
            False,
            True,
        ),
        (
            ATTENTION_MODE_HAZARD_CONDITIONED,
            1,
            AdditiveEdgeAttentionScoreFunction,
            True,
            True,
        ),
        (
            ATTENTION_MODE_MULTIHEAD_HAZARD_CONDITIONED,
            MULTIHEAD_COUNT,
            AdditiveEdgeAttentionScoreFunction,
            True,
            True,
        ),
    ),
)
def test_from_config_constructs_complete_stack(
    mode: str,
    heads: int,
    score_type: type[nn.Module],
    uses_hazard: bool,
    parameterized: bool,
) -> None:
    inputs = _inputs(
        with_hazard_query=uses_hazard,
    )
    module = _module(
        inputs,
        mode=mode,
        heads=heads,
    )

    assert isinstance(
        module.score_function,
        score_type,
    )
    assert isinstance(
        module.normalizer,
        TargetNodeRelationAttentionNormalization,
    )
    assert isinstance(
        module.head_reducer,
        MeanAttentionHeadReduction,
    )
    assert module.attention_mode == mode
    assert module.normalization_mode == (
        ATTENTION_NORMALIZATION_TARGET_NODE_RELATION
    )
    assert module.head_reduction == (
        ATTENTION_HEAD_REDUCTION_MEAN
    )
    assert module.num_heads == heads
    assert module.num_relations == RELATIONS
    assert module.relation_names == RELATION_NAMES
    assert module.stable_relation_ids == (
        STABLE_RELATION_IDS
    )
    assert module.uses_hazard_query is uses_hazard
    assert (module.parameter_count > 0) is parameterized
    assert (
        module.trainable_parameter_count > 0
    ) is parameterized


def test_direct_constructor_accepts_valid_components() -> None:
    inputs = _inputs()
    score = AdditiveEdgeAttentionScoreFunction(
        node_state_dim=HIDDEN_DIM,
        hazard_query_dim=QUERY_DIM,
        relation_names=RELATION_NAMES,
        stable_relation_ids=(
            STABLE_RELATION_IDS
        ),
        hidden_dim=SCORE_HIDDEN_DIM,
        mode=(
            ATTENTION_MODE_HAZARD_CONDITIONED
        ),
        num_heads=1,
    ).to(
        dtype=inputs.dtype,
        device=inputs.device,
    )
    normalizer = (
        TargetNodeRelationAttentionNormalization()
    )
    reducer = MeanAttentionHeadReduction(
        num_heads=1,
    )

    module = EdgeAttention(
        score_function=score,
        normalizer=normalizer,
        head_reducer=reducer,
    )

    assert module.score_function is score
    assert module.normalizer is normalizer
    assert module.head_reducer is reducer


def test_build_edge_attention_matches_from_config_contract() -> None:
    inputs = _inputs()
    config = _config(
        mode=(
            ATTENTION_MODE_HAZARD_CONDITIONED
        ),
    )

    built = build_edge_attention(
        config=config,
        source_inputs=inputs,
        score_hidden_dim=SCORE_HIDDEN_DIM,
    )
    direct = EdgeAttention.from_config(
        config=config,
        source_inputs=inputs,
        score_hidden_dim=SCORE_HIDDEN_DIM,
    )

    assert type(built) is type(direct)
    assert (
        built.architecture_fingerprint()
        == direct.architecture_fingerprint()
    )


def test_uniform_stack_is_fully_parameter_free() -> None:
    module = _module(
        _inputs(
            with_hazard_query=False,
        ),
        mode=ATTENTION_MODE_UNIFORM,
    )

    assert module.parameter_count == 0
    assert module.trainable_parameter_count == 0
    assert tuple(module.parameters()) == ()
    assert module.parameter_fingerprint() is None


def test_learned_stack_parameters_belong_only_to_score_function() -> None:
    module = _module(
        _inputs(),
    )

    assert module.parameter_count == (
        module.score_function.parameter_count
    )
    assert (
        module.trainable_parameter_count
        == module
        .score_function
        .trainable_parameter_count
    )
    assert module.normalizer.parameter_count == 0
    assert module.head_reducer.parameter_count == 0
    assert tuple(
        module.normalizer.parameters()
    ) == ()
    assert tuple(
        module.head_reducer.parameters()
    ) == ()


# =============================================================================
# Architecture and parameter provenance
# =============================================================================


def test_architecture_metadata_is_scientifically_explicit() -> None:
    module = _module(
        _inputs(),
    )
    architecture = module.architecture_dict()

    assert architecture["attention_mode"] == (
        ATTENTION_MODE_HAZARD_CONDITIONED
    )
    assert architecture["normalization_mode"] == (
        ATTENTION_NORMALIZATION_TARGET_NODE_RELATION
    )
    assert architecture["head_reduction"] == (
        ATTENTION_HEAD_REDUCTION_MEAN
    )
    assert architecture["num_heads"] == 1
    assert architecture["num_relations"] == (
        RELATIONS
    )
    assert architecture["group_semantics"] == (
        EDGE_ATTENTION_GROUP_SEMANTICS
    )
    assert architecture[
        "scientific_interpretation"
    ] == (
        EDGE_ATTENTION_SCIENTIFIC_INTERPRETATION
    )
    assert architecture[
        "relation_gate_owned_here"
    ] is False
    assert architecture["edge_attributes_used"] is False
    assert architecture[
        "semantic_edge_weight_owned_here"
    ] is False
    assert architecture[
        "structural_normalization_owned_here"
    ] is False
    assert architecture[
        "message_construction_owned_here"
    ] is False
    assert architecture["aggregation_owned_here"] is False
    assert architecture[
        "claims_causal_importance"
    ] is False
    assert architecture[
        "claims_head_specialization"
    ] is False
    assert architecture["operation_order"] == list(
        EDGE_ATTENTION_OPERATION_ORDER
    )


def test_architecture_fingerprint_is_deterministic() -> None:
    inputs = _inputs()
    first = _module(inputs)
    second = _module(inputs)

    assert (
        first.architecture_fingerprint()
        == second.architecture_fingerprint()
    )


def test_architecture_fingerprint_changes_with_mode() -> None:
    inputs = _inputs()
    blind = _module(
        inputs,
        mode=ATTENTION_MODE_HAZARD_BLIND,
    )
    conditioned = _module(
        inputs,
        mode=(
            ATTENTION_MODE_HAZARD_CONDITIONED
        ),
    )

    assert (
        blind.architecture_fingerprint()
        != conditioned.architecture_fingerprint()
    )


def test_architecture_fingerprint_changes_with_head_count() -> None:
    inputs = _inputs()
    single = _module(
        inputs,
        mode=(
            ATTENTION_MODE_HAZARD_CONDITIONED
        ),
        heads=1,
    )
    multi = _module(
        inputs,
        mode=(
            ATTENTION_MODE_MULTIHEAD_HAZARD_CONDITIONED
        ),
        heads=MULTIHEAD_COUNT,
    )

    assert (
        single.architecture_fingerprint()
        != multi.architecture_fingerprint()
    )


def test_architecture_fingerprint_ignores_parameter_values() -> None:
    inputs = _inputs()
    first = _module(inputs)
    second = _module(inputs)
    _fill_score_parameters(
        first,
        0.1,
    )
    _fill_score_parameters(
        second,
        0.9,
    )

    assert (
        first.architecture_fingerprint()
        == second.architecture_fingerprint()
    )
    assert (
        first.parameter_fingerprint()
        != second.parameter_fingerprint()
    )


def test_learned_parameter_fingerprint_changes_with_parameter() -> None:
    module = _module(
        _inputs(),
    )
    before = module.parameter_fingerprint()

    with torch.no_grad():
        assert isinstance(
            module.score_function,
            AdditiveEdgeAttentionScoreFunction,
        )
        module.score_function.score_vectors[
            0,
            0,
        ] += 1.0

    after = module.parameter_fingerprint()

    assert before is not None
    assert after is not None
    assert before != after


def test_uniform_parameter_fingerprint_remains_none() -> None:
    module = _module(
        _inputs(
            with_hazard_query=False,
        ),
        mode=ATTENTION_MODE_UNIFORM,
    )

    assert module.parameter_fingerprint() is None


def test_state_dict_round_trip_preserves_output_and_parameter_fingerprint() -> None:
    inputs = _inputs()
    original = _module(inputs)
    restored = _module(inputs)
    restored.load_state_dict(
        original.state_dict()
    )

    original_output = original(inputs)
    restored_output = restored(inputs)

    torch.testing.assert_close(
        restored_output.raw_scores_by_head,
        original_output.raw_scores_by_head,
    )
    torch.testing.assert_close(
        restored_output.normalized_weights_by_head,
        original_output.normalized_weights_by_head,
    )
    torch.testing.assert_close(
        restored_output.edge_weights,
        original_output.edge_weights,
    )
    assert (
        restored.parameter_fingerprint()
        == original.parameter_fingerprint()
    )


# =============================================================================
# Individually auditable stage methods
# =============================================================================


@pytest.mark.parametrize(
    (
        "mode",
        "heads",
    ),
    (
        (
            ATTENTION_MODE_UNIFORM,
            1,
        ),
        (
            ATTENTION_MODE_HAZARD_BLIND,
            1,
        ),
        (
            ATTENTION_MODE_HAZARD_CONDITIONED,
            1,
        ),
        (
            ATTENTION_MODE_MULTIHEAD_HAZARD_CONDITIONED,
            MULTIHEAD_COUNT,
        ),
    ),
)
def test_compute_stages_returns_exact_lineage(
    mode: str,
    heads: int,
) -> None:
    inputs = _inputs(
        with_hazard_query=(
            mode
            in (
                ATTENTION_MODE_HAZARD_CONDITIONED,
                ATTENTION_MODE_MULTIHEAD_HAZARD_CONDITIONED,
            )
        ),
    )
    module = _module(
        inputs,
        mode=mode,
        heads=heads,
    )

    (
        score_output,
        normalization_output,
        reduction_output,
    ) = module.compute_stages(inputs)

    assert isinstance(
        score_output,
        EdgeAttentionScoreOutput,
    )
    assert isinstance(
        normalization_output,
        AttentionNormalizationOutput,
    )
    assert isinstance(
        reduction_output,
        AttentionHeadReductionOutput,
    )
    assert score_output.source_inputs is inputs
    assert (
        normalization_output.source_score_output
        is score_output
    )
    assert (
        reduction_output.source_normalization_output
        is normalization_output
    )
    assert score_output.raw_scores_by_head.shape == (
        inputs.num_edges,
        heads,
    )
    assert (
        normalization_output
        .normalized_weights_by_head
        .shape
        == (
            inputs.num_edges,
            heads,
        )
    )
    assert reduction_output.edge_weights.shape == (
        inputs.num_edges,
    )


def test_individual_stage_methods_match_compute_stages() -> None:
    inputs = _inputs()
    module = _module(inputs)

    score = module.score_edges(inputs)
    normalization = module.normalize_scores(
        score
    )
    reduction = module.reduce_heads(
        normalization
    )
    computed = module.compute_stages(
        inputs
    )

    torch.testing.assert_close(
        score.raw_scores_by_head,
        computed[0].raw_scores_by_head,
    )
    torch.testing.assert_close(
        normalization.normalized_weights_by_head,
        computed[1].normalized_weights_by_head,
    )
    torch.testing.assert_close(
        reduction.edge_weights,
        computed[2].edge_weights,
    )


def test_stage_group_ids_and_counts_match_source_inputs() -> None:
    inputs = _inputs()
    module = _module(inputs)
    _, normalization, reduction = (
        module.compute_stages(inputs)
    )

    assert torch.equal(
        normalization.group_ids,
        inputs.attention_group_id,
    )
    assert torch.equal(
        reduction.group_ids,
        inputs.attention_group_id,
    )
    expected_counts = torch.bincount(
        inputs.attention_group_id,
        minlength=NUM_GROUPS,
    )
    assert torch.equal(
        normalization.group_counts,
        expected_counts,
    )
    assert torch.equal(
        reduction.group_counts,
        expected_counts,
    )


# =============================================================================
# Final public output
# =============================================================================


@pytest.mark.parametrize(
    (
        "mode",
        "heads",
    ),
    (
        (
            ATTENTION_MODE_UNIFORM,
            1,
        ),
        (
            ATTENTION_MODE_HAZARD_BLIND,
            1,
        ),
        (
            ATTENTION_MODE_HAZARD_CONDITIONED,
            1,
        ),
        (
            ATTENTION_MODE_MULTIHEAD_HAZARD_CONDITIONED,
            MULTIHEAD_COUNT,
        ),
    ),
)
def test_forward_public_output_contract(
    mode: str,
    heads: int,
) -> None:
    inputs = _inputs(
        with_hazard_query=(
            mode
            in (
                ATTENTION_MODE_HAZARD_CONDITIONED,
                ATTENTION_MODE_MULTIHEAD_HAZARD_CONDITIONED,
            )
        ),
    )
    module = _module(
        inputs,
        mode=mode,
        heads=heads,
    )

    output = module(inputs)

    assert isinstance(
        output,
        EdgeAttentionOutput,
    )
    assert output.source_inputs is inputs
    assert output.attention_mode == mode
    assert output.normalization_mode == (
        ATTENTION_NORMALIZATION_TARGET_NODE_RELATION
    )
    assert output.head_reduction == (
        ATTENTION_HEAD_REDUCTION_MEAN
    )
    assert output.num_heads == heads
    assert output.raw_scores_by_head.shape == (
        inputs.num_edges,
        heads,
    )
    assert (
        output.normalized_weights_by_head.shape
        == (
            inputs.num_edges,
            heads,
        )
    )
    assert output.edge_weights.shape == (
        inputs.num_edges,
    )
    assert torch.equal(
        output.group_ids,
        inputs.attention_group_id,
    )
    assert output.encoder_architecture_fingerprint == (
        module.architecture_fingerprint()
    )
    assert output.parameter_fingerprint == (
        module.parameter_fingerprint()
    )


def test_forward_output_uses_exact_stage_tensors() -> None:
    inputs = _inputs()
    module = _module(inputs)
    score, normalization, reduction = (
        module.compute_stages(inputs)
    )
    output = module.assemble_output(
        score,
        normalization,
        reduction,
    )

    assert (
        output.raw_scores_by_head
        is score.raw_scores_by_head
    )
    assert (
        output.normalized_weights_by_head
        is normalization.normalized_weights_by_head
    )
    assert (
        output.edge_weights
        is reduction.edge_weights
    )
    assert output.group_ids is (
        normalization.group_ids
    )
    assert output.group_counts is (
        normalization.group_counts
    )


def test_forward_matches_manual_stage_assembly() -> None:
    inputs = _inputs()
    module = _module(inputs)

    direct = module(inputs)
    score, normalization, reduction = (
        module.compute_stages(inputs)
    )
    assembled = module.assemble_output(
        score,
        normalization,
        reduction,
    )

    torch.testing.assert_close(
        direct.raw_scores_by_head,
        assembled.raw_scores_by_head,
    )
    torch.testing.assert_close(
        direct.normalized_weights_by_head,
        assembled.normalized_weights_by_head,
    )
    torch.testing.assert_close(
        direct.edge_weights,
        assembled.edge_weights,
    )
    assert direct.attention_mode == (
        assembled.attention_mode
    )


def test_final_output_is_frozen() -> None:
    output = _module(
        _inputs(),
    )(
        _inputs()
    )

    with pytest.raises(
        FrozenInstanceError,
    ):
        output.attention_mode = "changed"  # type: ignore[misc]


def test_output_contains_no_gate_or_causal_fields() -> None:
    output = _module(
        _inputs(),
    )(
        _inputs()
    )

    assert not hasattr(
        output,
        "relation_gate",
    )
    assert not hasattr(
        output,
        "causal_importance",
    )
    assert not hasattr(
        output,
        "aggregation_denominator",
    )
    assert not hasattr(
        output,
        "edge_messages",
    )


# =============================================================================
# Exact normalization and reduction behavior
# =============================================================================


def test_every_nonempty_group_sums_to_one_per_head_and_after_reduction() -> None:
    inputs = _inputs()
    output = _module(
        inputs,
        mode=(
            ATTENTION_MODE_MULTIHEAD_HAZARD_CONDITIONED
        ),
        heads=MULTIHEAD_COUNT,
    )(inputs)

    head_sums = _group_sums(
        output.normalized_weights_by_head,
        output.group_ids,
        num_groups=NUM_GROUPS,
    )
    reduced_sums = _group_sums(
        output.edge_weights,
        output.group_ids,
        num_groups=NUM_GROUPS,
    )
    present = output.group_counts > 0

    torch.testing.assert_close(
        head_sums[present],
        torch.ones_like(
            head_sums[present]
        ),
    )
    torch.testing.assert_close(
        reduced_sums[present],
        torch.ones_like(
            reduced_sums[present]
        ),
    )
    assert torch.equal(
        head_sums[~present],
        torch.zeros_like(
            head_sums[~present]
        ),
    )
    assert torch.equal(
        reduced_sums[~present],
        torch.zeros_like(
            reduced_sums[~present]
        ),
    )


def test_uniform_attention_produces_reciprocal_group_size() -> None:
    inputs = _inputs(
        with_hazard_query=False,
    )
    output = _module(
        inputs,
        mode=ATTENTION_MODE_UNIFORM,
    )(inputs)

    edge_counts = output.group_counts[
        output.group_ids
    ].to(dtype=output.edge_weights.dtype)
    expected = (
        torch.ones_like(
            output.edge_weights
        )
        / edge_counts
    )

    assert torch.equal(
        output.raw_scores_by_head,
        torch.zeros_like(
            output.raw_scores_by_head
        ),
    )
    assert torch.equal(
        output.edge_weights,
        expected,
    )


def test_single_head_output_is_exact_normalized_head_identity() -> None:
    inputs = _inputs()
    output = _module(
        inputs,
        mode=(
            ATTENTION_MODE_HAZARD_CONDITIONED
        ),
        heads=1,
    )(inputs)

    assert torch.equal(
        output.edge_weights,
        output.normalized_weights_by_head[
            :,
            0,
        ],
    )


def test_multihead_output_equals_arithmetic_mean() -> None:
    inputs = _inputs()
    output = _module(
        inputs,
        mode=(
            ATTENTION_MODE_MULTIHEAD_HAZARD_CONDITIONED
        ),
        heads=MULTIHEAD_COUNT,
    )(inputs)

    torch.testing.assert_close(
        output.edge_weights,
        output.normalized_weights_by_head.mean(
            dim=1
        ),
    )


def test_singleton_groups_retain_exact_one() -> None:
    inputs = _inputs()
    output = _module(
        inputs,
        mode=(
            ATTENTION_MODE_MULTIHEAD_HAZARD_CONDITIONED
        ),
        heads=MULTIHEAD_COUNT,
    )(inputs)
    singleton_edges = (
        output.group_counts[
            output.group_ids
        ]
        == 1
    )

    assert torch.equal(
        output.normalized_weights_by_head[
            singleton_edges
        ],
        torch.ones_like(
            output.normalized_weights_by_head[
                singleton_edges
            ]
        ),
    )
    assert torch.equal(
        output.edge_weights[
            singleton_edges
        ],
        torch.ones_like(
            output.edge_weights[
                singleton_edges
            ]
        ),
    )


# =============================================================================
# Scientific metamorphic behavior
# =============================================================================


def test_hazard_blind_output_is_query_invariant() -> None:
    inputs_a = _inputs(
        query=_query_tensor(
            offset=0.0,
        ),
    )
    inputs_b = _inputs(
        query=_query_tensor(
            offset=100.0,
        ),
    )
    module = _module(
        inputs_a,
        mode=ATTENTION_MODE_HAZARD_BLIND,
    )

    output_a = module(inputs_a)
    output_b = module(inputs_b)

    assert torch.equal(
        output_a.raw_scores_by_head,
        output_b.raw_scores_by_head,
    )
    assert torch.equal(
        output_a.normalized_weights_by_head,
        output_b.normalized_weights_by_head,
    )
    assert torch.equal(
        output_a.edge_weights,
        output_b.edge_weights,
    )


def test_conditioned_output_can_change_with_query() -> None:
    inputs_a = _inputs(
        query=torch.zeros(
            (NODES, QUERY_DIM),
        ),
    )
    inputs_b = _inputs(
        query=torch.ones(
            (NODES, QUERY_DIM),
        ),
    )
    module = _module(
        inputs_a,
        mode=(
            ATTENTION_MODE_HAZARD_CONDITIONED
        ),
    )
    _fill_score_parameters(
        module,
        0.1,
    )

    output_a = module(inputs_a)
    output_b = module(inputs_b)

    assert not torch.allclose(
        output_a.raw_scores_by_head,
        output_b.raw_scores_by_head,
    )


def test_edge_order_equivariance_end_to_end() -> None:
    permutation = torch.tensor(
        [6, 2, 0, 5, 1, 4, 3],
        dtype=torch.long,
    )
    original_inputs = _inputs()
    permuted_inputs = _inputs(
        permutation=permutation,
    )
    module = _module(
        original_inputs,
        mode=(
            ATTENTION_MODE_MULTIHEAD_HAZARD_CONDITIONED
        ),
        heads=MULTIHEAD_COUNT,
    )

    original = module(original_inputs)
    permuted = module(permuted_inputs)

    torch.testing.assert_close(
        permuted.raw_scores_by_head,
        original.raw_scores_by_head[
            permutation
        ],
    )
    torch.testing.assert_close(
        permuted.normalized_weights_by_head,
        original.normalized_weights_by_head[
            permutation
        ],
    )
    torch.testing.assert_close(
        permuted.edge_weights,
        original.edge_weights[
            permutation
        ],
    )
    assert torch.equal(
        permuted.group_ids,
        original.group_ids[
            permutation
        ],
    )
    assert torch.equal(
        permuted.group_counts,
        original.group_counts,
    )


def test_edge_attributes_do_not_change_bounded_attention() -> None:
    attributes_a = torch.zeros(
        (EDGES, 2),
    )
    attributes_b = torch.arange(
        EDGES * 2,
        dtype=torch.float32,
    ).reshape(EDGES, 2)
    inputs_a = _inputs(
        edge_attributes=attributes_a,
    )
    inputs_b = _inputs(
        edge_attributes=attributes_b,
    )
    module = _module(inputs_a)

    output_a = module(inputs_a)
    output_b = module(inputs_b)

    assert torch.equal(
        output_a.raw_scores_by_head,
        output_b.raw_scores_by_head,
    )
    assert torch.equal(
        output_a.edge_weights,
        output_b.edge_weights,
    )


def test_semantic_edge_weights_do_not_change_bounded_attention() -> None:
    semantic_a = torch.ones(
        EDGES,
    )
    semantic_b = torch.linspace(
        0.1,
        5.0,
        EDGES,
    )
    inputs_a = _inputs(
        semantic_edge_weight=semantic_a,
    )
    inputs_b = _inputs(
        semantic_edge_weight=semantic_b,
    )
    module = _module(inputs_a)

    output_a = module(inputs_a)
    output_b = module(inputs_b)

    assert torch.equal(
        output_a.raw_scores_by_head,
        output_b.raw_scores_by_head,
    )
    assert torch.equal(
        output_a.edge_weights,
        output_b.edge_weights,
    )


def test_forward_does_not_mutate_source_inputs() -> None:
    inputs = _inputs()
    state_before = (
        inputs.node_state.fused_state
        .clone()
    )
    query_before = (
        inputs.node_hazard_query
        .clone()
    )
    source_before = (
        inputs.source_index.clone()
    )
    target_before = (
        inputs.target_index.clone()
    )
    relation_before = (
        inputs.edge_relation_index
        .clone()
    )

    _ = _module(inputs)(inputs)

    assert torch.equal(
        inputs.node_state.fused_state,
        state_before,
    )
    assert torch.equal(
        inputs.node_hazard_query,
        query_before,
    )
    assert torch.equal(
        inputs.source_index,
        source_before,
    )
    assert torch.equal(
        inputs.target_index,
        target_before,
    )
    assert torch.equal(
        inputs.edge_relation_index,
        relation_before,
    )


# =============================================================================
# Empty edges, dtype, device, and autograd
# =============================================================================


@pytest.mark.parametrize(
    (
        "mode",
        "heads",
    ),
    (
        (
            ATTENTION_MODE_UNIFORM,
            1,
        ),
        (
            ATTENTION_MODE_HAZARD_CONDITIONED,
            1,
        ),
        (
            ATTENTION_MODE_MULTIHEAD_HAZARD_CONDITIONED,
            MULTIHEAD_COUNT,
        ),
    ),
)
def test_empty_edge_end_to_end(
    mode: str,
    heads: int,
) -> None:
    inputs = _inputs(
        empty_edges=True,
        with_hazard_query=(
            mode != ATTENTION_MODE_UNIFORM
        ),
    )
    output = _module(
        inputs,
        mode=mode,
        heads=heads,
    )(inputs)

    assert output.raw_scores_by_head.shape == (
        0,
        heads,
    )
    assert (
        output.normalized_weights_by_head.shape
        == (
            0,
            heads,
        )
    )
    assert output.edge_weights.shape == (0,)
    assert output.group_ids.shape == (0,)
    assert output.group_counts.shape == (
        NUM_GROUPS,
    )
    assert torch.equal(
        output.group_counts,
        torch.zeros_like(
            output.group_counts
        ),
    )


def test_float64_end_to_end() -> None:
    inputs = _inputs(
        dtype=torch.float64,
    )
    module = _module(inputs)
    output = module(inputs)

    assert output.raw_scores_by_head.dtype == (
        torch.float64
    )
    assert (
        output.normalized_weights_by_head.dtype
        == torch.float64
    )
    assert output.edge_weights.dtype == (
        torch.float64
    )
    assert all(
        parameter.dtype == torch.float64
        for parameter in module.parameters()
    )


def test_end_to_end_gradients_reach_state_query_and_score_parameters() -> None:
    state = _state_tensor(
        requires_grad=True,
    )
    query = _query_tensor(
        requires_grad=True,
    )
    inputs = _inputs(
        state=state,
        query=query,
    )
    module = _module(
        inputs,
        mode=(
            ATTENTION_MODE_MULTIHEAD_HAZARD_CONDITIONED
        ),
        heads=MULTIHEAD_COUNT,
    )
    _fill_score_parameters(
        module,
        0.1,
    )

    output = module(inputs)
    coefficients = torch.arange(
        1,
        EDGES + 1,
        dtype=inputs.dtype,
    )
    loss = (
        output.edge_weights
        * coefficients
    ).sum()
    loss.backward()

    assert state.grad is not None
    assert query.grad is not None
    assert torch.isfinite(
        state.grad
    ).all()
    assert torch.isfinite(
        query.grad
    ).all()
    assert float(
        state.grad.abs().sum().item()
    ) > 0.0
    assert float(
        query.grad.abs().sum().item()
    ) > 0.0

    for name, parameter in (
        module
        .score_function
        .named_parameters()
    ):
        assert parameter.grad is not None, name
        assert torch.isfinite(
            parameter.grad
        ).all(), name


def test_hazard_blind_end_to_end_creates_no_query_gradient() -> None:
    state = _state_tensor(
        requires_grad=True,
    )
    query = _query_tensor(
        requires_grad=True,
    )
    inputs = _inputs(
        state=state,
        query=query,
    )
    module = _module(
        inputs,
        mode=ATTENTION_MODE_HAZARD_BLIND,
    )
    _fill_score_parameters(
        module,
        0.1,
    )

    output = module(inputs)
    coefficients = torch.arange(
        1,
        EDGES + 1,
        dtype=inputs.dtype,
    )
    (
        output.edge_weights
        * coefficients
    ).sum().backward()

    assert state.grad is not None
    assert query.grad is None


# =============================================================================
# Final-output assembly helper and exact stage lineage
# =============================================================================


def test_free_assembly_helper_constructs_public_output() -> None:
    inputs = _inputs()
    module = _module(inputs)
    score, normalization, reduction = (
        module.compute_stages(inputs)
    )

    output = assemble_edge_attention_output(
        score_output=score,
        normalization_output=normalization,
        reduction_output=reduction,
        encoder_architecture_fingerprint=(
            "assembled-architecture"
        ),
        parameter_fingerprint=(
            "assembled-parameters"
        ),
    )

    assert isinstance(
        output,
        EdgeAttentionOutput,
    )
    assert output.source_inputs is inputs
    assert output.raw_scores_by_head is (
        score.raw_scores_by_head
    )
    assert (
        output.normalized_weights_by_head
        is normalization
        .normalized_weights_by_head
    )
    assert output.edge_weights is (
        reduction.edge_weights
    )
    assert (
        output.encoder_architecture_fingerprint
        == "assembled-architecture"
    )
    assert output.parameter_fingerprint == (
        "assembled-parameters"
    )


def test_free_assembly_helper_rejects_crossed_score_chain() -> None:
    inputs = _inputs()
    module = _module(inputs)
    score_a, _, _ = module.compute_stages(
        inputs
    )
    _, normalization_b, reduction_b = (
        module.compute_stages(inputs)
    )

    with pytest.raises(
        ValueError,
        match="exact supplied score_output",
    ):
        assemble_edge_attention_output(
            score_output=score_a,
            normalization_output=(
                normalization_b
            ),
            reduction_output=reduction_b,
            encoder_architecture_fingerprint=(
                "architecture"
            ),
        )


def test_free_assembly_helper_rejects_crossed_reduction_chain() -> None:
    inputs = _inputs()
    module = _module(inputs)
    score_a, normalization_a, _ = (
        module.compute_stages(inputs)
    )
    _, _, reduction_b = (
        module.compute_stages(inputs)
    )

    with pytest.raises(
        ValueError,
        match="exact supplied normalization_output",
    ):
        assemble_edge_attention_output(
            score_output=score_a,
            normalization_output=(
                normalization_a
            ),
            reduction_output=reduction_b,
            encoder_architecture_fingerprint=(
                "architecture"
            ),
        )


@pytest.mark.parametrize(
    (
        "field",
        "value",
    ),
    (
        (
            "architecture",
            "",
        ),
        (
            "parameter",
            "",
        ),
    ),
)
def test_free_assembly_helper_rejects_blank_fingerprints(
    field: str,
    value: str,
) -> None:
    inputs = _inputs()
    module = _module(inputs)
    score, normalization, reduction = (
        module.compute_stages(inputs)
    )

    kwargs: dict[str, Any] = {
        "score_output": score,
        "normalization_output": (
            normalization
        ),
        "reduction_output": reduction,
        "encoder_architecture_fingerprint": (
            "architecture"
        ),
        "parameter_fingerprint": None,
    }
    if field == "architecture":
        kwargs[
            "encoder_architecture_fingerprint"
        ] = value
    else:
        kwargs[
            "parameter_fingerprint"
        ] = value

    with pytest.raises(
        ValueError,
        match="non-empty",
    ):
        assemble_edge_attention_output(
            **kwargs
        )


def test_free_assembly_helper_rejects_wrong_stage_types() -> None:
    inputs = _inputs()
    module = _module(inputs)
    score, normalization, reduction = (
        module.compute_stages(inputs)
    )

    with pytest.raises(
        TypeError,
        match="EdgeAttentionScoreOutput",
    ):
        assemble_edge_attention_output(
            score_output=object(),  # type: ignore[arg-type]
            normalization_output=(
                normalization
            ),
            reduction_output=reduction,
            encoder_architecture_fingerprint=(
                "architecture"
            ),
        )

    with pytest.raises(
        TypeError,
        match="AttentionNormalizationOutput",
    ):
        assemble_edge_attention_output(
            score_output=score,
            normalization_output=object(),  # type: ignore[arg-type]
            reduction_output=reduction,
            encoder_architecture_fingerprint=(
                "architecture"
            ),
        )

    with pytest.raises(
        TypeError,
        match="AttentionHeadReductionOutput",
    ):
        assemble_edge_attention_output(
            score_output=score,
            normalization_output=(
                normalization
            ),
            reduction_output=object(),  # type: ignore[arg-type]
            encoder_architecture_fingerprint=(
                "architecture"
            ),
        )


def test_module_assemble_output_rejects_chain_from_other_mode() -> None:
    inputs = _inputs()
    conditioned = _module(
        inputs,
        mode=(
            ATTENTION_MODE_HAZARD_CONDITIONED
        ),
    )
    blind = _module(
        inputs,
        mode=ATTENTION_MODE_HAZARD_BLIND,
    )
    score, normalization, reduction = (
        blind.compute_stages(inputs)
    )

    with pytest.raises(
        ValueError,
        match="attention mode differs",
    ):
        conditioned.assemble_output(
            score,
            normalization,
            reduction,
        )


# =============================================================================
# Diagnostics
# =============================================================================


def test_diagnostic_summary_contract() -> None:
    inputs = _inputs()
    module = _module(
        inputs,
        mode=(
            ATTENTION_MODE_MULTIHEAD_HAZARD_CONDITIONED
        ),
        heads=MULTIHEAD_COUNT,
    )
    _, _, reduction = (
        module.compute_stages(inputs)
    )

    summary = module.diagnostic_summary(
        reduction
    )

    assert summary["schema_version"] == (
        EDGE_ATTENTION_ORCHESTRATOR_SCHEMA_VERSION
    )
    assert summary["attention_mode"] == (
        ATTENTION_MODE_MULTIHEAD_HAZARD_CONDITIONED
    )
    assert summary["normalization_mode"] == (
        ATTENTION_NORMALIZATION_TARGET_NODE_RELATION
    )
    assert summary["head_reduction"] == (
        ATTENTION_HEAD_REDUCTION_MEAN
    )
    assert summary["num_edges"] == EDGES
    assert summary["num_heads"] == (
        MULTIHEAD_COUNT
    )
    assert summary["num_groups"] == (
        NUM_GROUPS
    )
    assert summary["num_nonempty_groups"] == 5
    assert (
        summary[
            "maximum_head_level_normalization_error"
        ]
        <= 1e-6
    )
    assert (
        summary[
            "maximum_reduced_normalization_error"
        ]
        <= 1e-6
    )
    assert summary["head_disagreement"][
        "num_heads"
    ] == MULTIHEAD_COUNT
    assert summary["causal_importance_claim"] is False
    assert (
        summary["uncertainty_calibration_claim"]
        is False
    )
    assert (
        summary["head_specialization_claim"]
        is False
    )


def test_diagnostic_summary_empty_edges() -> None:
    inputs = _inputs(
        empty_edges=True,
    )
    module = _module(
        inputs,
        mode=(
            ATTENTION_MODE_MULTIHEAD_HAZARD_CONDITIONED
        ),
        heads=MULTIHEAD_COUNT,
    )
    _, _, reduction = (
        module.compute_stages(inputs)
    )

    summary = module.diagnostic_summary(
        reduction
    )

    assert summary["num_edges"] == 0
    assert summary["num_nonempty_groups"] == 0
    assert summary[
        "maximum_head_level_normalization_error"
    ] == 0.0
    assert summary[
        "maximum_reduced_normalization_error"
    ] == 0.0
    assert summary["head_disagreement"][
        "edge_count"
    ] == 0


def test_diagnostic_summary_rejects_other_mode_chain() -> None:
    inputs = _inputs()
    conditioned = _module(
        inputs,
        mode=(
            ATTENTION_MODE_HAZARD_CONDITIONED
        ),
    )
    blind = _module(
        inputs,
        mode=ATTENTION_MODE_HAZARD_BLIND,
    )
    _, _, reduction = blind.compute_stages(
        inputs
    )

    with pytest.raises(
        ValueError,
        match="differs from the orchestrator attention mode",
    ):
        conditioned.diagnostic_summary(
            reduction
        )


def test_diagnostic_summary_rejects_wrong_type() -> None:
    module = _module(
        _inputs(),
    )

    with pytest.raises(
        TypeError,
        match="AttentionHeadReductionOutput",
    ):
        module.diagnostic_summary(
            object()  # type: ignore[arg-type]
        )


# =============================================================================
# Invalid direct construction
# =============================================================================


def test_constructor_rejects_wrong_score_function_type() -> None:
    with pytest.raises(
        TypeError,
        match="score_function",
    ):
        EdgeAttention(
            score_function=object(),  # type: ignore[arg-type]
            normalizer=(
                TargetNodeRelationAttentionNormalization()
            ),
            head_reducer=(
                MeanAttentionHeadReduction(
                    num_heads=1,
                )
            ),
        )


def test_constructor_rejects_wrong_normalizer_type() -> None:
    inputs = _inputs()
    score = (
        UniformEdgeAttentionScoreFunction(
            relation_names=RELATION_NAMES,
            stable_relation_ids=(
                STABLE_RELATION_IDS
            ),
        )
    )

    with pytest.raises(
        TypeError,
        match="normalizer",
    ):
        EdgeAttention(
            score_function=score,
            normalizer=object(),  # type: ignore[arg-type]
            head_reducer=(
                MeanAttentionHeadReduction(
                    num_heads=1,
                )
            ),
        )


def test_constructor_rejects_wrong_head_reducer_type() -> None:
    score = (
        UniformEdgeAttentionScoreFunction(
            relation_names=RELATION_NAMES,
            stable_relation_ids=(
                STABLE_RELATION_IDS
            ),
        )
    )

    with pytest.raises(
        TypeError,
        match="head_reducer",
    ):
        EdgeAttention(
            score_function=score,
            normalizer=(
                TargetNodeRelationAttentionNormalization()
            ),
            head_reducer=object(),  # type: ignore[arg-type]
        )


def test_constructor_rejects_head_count_mismatch() -> None:
    score = AdditiveEdgeAttentionScoreFunction(
        node_state_dim=HIDDEN_DIM,
        hazard_query_dim=QUERY_DIM,
        relation_names=RELATION_NAMES,
        stable_relation_ids=(
            STABLE_RELATION_IDS
        ),
        hidden_dim=SCORE_HIDDEN_DIM,
        mode=(
            ATTENTION_MODE_MULTIHEAD_HAZARD_CONDITIONED
        ),
        num_heads=MULTIHEAD_COUNT,
    )

    with pytest.raises(
        ValueError,
        match="agree on the exact attention head count",
    ):
        EdgeAttention(
            score_function=score,
            normalizer=(
                TargetNodeRelationAttentionNormalization()
            ),
            head_reducer=(
                MeanAttentionHeadReduction(
                    num_heads=2,
                )
            ),
        )


def test_constructor_rejects_mutated_normalization_policy() -> None:
    score = (
        UniformEdgeAttentionScoreFunction(
            relation_names=RELATION_NAMES,
            stable_relation_ids=(
                STABLE_RELATION_IDS
            ),
        )
    )
    normalizer = (
        TargetNodeRelationAttentionNormalization()
    )
    normalizer.normalization_mode = (
        ATTENTION_NORMALIZATION_TARGET_NODE
    )

    with pytest.raises(
        ValueError,
        match="requires exact target-node/relation normalization",
    ):
        EdgeAttention(
            score_function=score,
            normalizer=normalizer,
            head_reducer=(
                MeanAttentionHeadReduction(
                    num_heads=1,
                )
            ),
        )


def test_constructor_rejects_mutated_head_reduction_policy() -> None:
    score = (
        UniformEdgeAttentionScoreFunction(
            relation_names=RELATION_NAMES,
            stable_relation_ids=(
                STABLE_RELATION_IDS
            ),
        )
    )
    reducer = MeanAttentionHeadReduction(
        num_heads=1,
    )
    reducer.head_reduction = (
        ATTENTION_HEAD_REDUCTION_MAX
    )

    with pytest.raises(
        ValueError,
        match="requires mean attention-head reduction",
    ):
        EdgeAttention(
            score_function=score,
            normalizer=(
                TargetNodeRelationAttentionNormalization()
            ),
            head_reducer=reducer,
        )


# =============================================================================
# Invalid configuration construction
# =============================================================================


def test_from_config_rejects_wrong_config_type() -> None:
    with pytest.raises(
        TypeError,
        match="FunctionalMessagePassingConfig",
    ):
        EdgeAttention.from_config(
            config=object(),  # type: ignore[arg-type]
            source_inputs=_inputs(),
        )


def test_from_config_rejects_wrong_input_type() -> None:
    with pytest.raises(
        TypeError,
        match="FunctionalMessagePassingInputs",
    ):
        EdgeAttention.from_config(
            config=_config(
                mode=(
                    ATTENTION_MODE_HAZARD_CONDITIONED
                ),
            ),
            source_inputs=object(),  # type: ignore[arg-type]
        )


def test_from_config_rejects_disabled_message_passing() -> None:
    config = _config(
        mode=ATTENTION_MODE_UNIFORM,
        enabled=False,
        attention_enabled=False,
    )

    with pytest.raises(
        ValueError,
        match="message passing to be enabled",
    ):
        EdgeAttention.from_config(
            config=config,
            source_inputs=_inputs(
                with_hazard_query=False,
            ),
        )


def test_from_config_rejects_disabled_attention() -> None:
    config = _config(
        mode=ATTENTION_MODE_UNIFORM,
        enabled=True,
        attention_enabled=False,
    )

    with pytest.raises(
        ValueError,
        match="attention_enabled=True",
    ):
        EdgeAttention.from_config(
            config=config,
            source_inputs=_inputs(
                with_hazard_query=False,
            ),
        )


def test_from_config_rejects_semantic_weight_mode() -> None:
    with pytest.raises(
        NotImplementedError,
        match="data-coefficient mode",
    ):
        EdgeAttention.from_config(
            config=_config(
                mode=(
                    ATTENTION_MODE_SEMANTIC_WEIGHT
                ),
            ),
            source_inputs=_inputs(),
        )


def test_from_config_rejects_unimplemented_normalization() -> None:
    with pytest.raises(
        NotImplementedError,
        match="not implemented",
    ):
        EdgeAttention.from_config(
            config=_config(
                mode=ATTENTION_MODE_UNIFORM,
                normalization=(
                    ATTENTION_NORMALIZATION_TARGET_NODE
                ),
            ),
            source_inputs=_inputs(
                with_hazard_query=False,
            ),
        )


def test_from_config_rejects_unimplemented_head_reduction() -> None:
    with pytest.raises(
        NotImplementedError,
    ):
        EdgeAttention.from_config(
            config=_config(
                mode=(
                    ATTENTION_MODE_HAZARD_CONDITIONED
                ),
                reduction=(
                    ATTENTION_HEAD_REDUCTION_WEIGHTED_MEAN
                ),
            ),
            source_inputs=_inputs(),
        )


def test_conditioned_from_config_requires_hazard_query() -> None:
    with pytest.raises(
        ValueError,
        match="node_hazard_query",
    ):
        EdgeAttention.from_config(
            config=_config(
                mode=(
                    ATTENTION_MODE_HAZARD_CONDITIONED
                ),
            ),
            source_inputs=_inputs(
                with_hazard_query=False,
            ),
        )


# =============================================================================
# Runtime validation and component-integrity failures
# =============================================================================


def test_forward_rejects_wrong_input_type() -> None:
    module = _module(
        _inputs(),
    )

    with pytest.raises(
        TypeError,
        match="FunctionalMessagePassingInputs",
    ):
        module(
            object()  # type: ignore[arg-type]
        )


def test_forward_rejects_relation_name_mismatch() -> None:
    original_inputs = _inputs()
    module = _module(
        original_inputs
    )
    changed_inputs = _inputs(
        registry=_registry(
            names=(
                "drainage_dependency",
                "spatial_adjacency",
                "random_placebo",
            ),
        )
    )

    with pytest.raises(
        ValueError,
        match="relation ordering differs",
    ):
        module(changed_inputs)


def test_forward_rejects_stable_relation_id_mismatch() -> None:
    original_inputs = _inputs()
    module = _module(
        original_inputs
    )
    changed_inputs = _inputs(
        registry=_registry(
            stable_ids=(
                101,
                310,
                900,
            ),
        )
    )

    with pytest.raises(
        ValueError,
        match="stable relation IDs differ",
    ):
        module(changed_inputs)


def test_conditioned_forward_rejects_missing_query() -> None:
    inputs = _inputs()
    module = _module(
        inputs,
        mode=(
            ATTENTION_MODE_HAZARD_CONDITIONED
        ),
    )
    no_query_inputs = _inputs(
        with_hazard_query=False,
    )

    with pytest.raises(
        ValueError,
        match="requires a node-aligned hazard query",
    ):
        module(no_query_inputs)


def test_nonfinite_score_parameter_is_rejected() -> None:
    module = _module(
        _inputs(),
    )
    assert isinstance(
        module.score_function,
        AdditiveEdgeAttentionScoreFunction,
    )

    with torch.no_grad():
        module.score_function.score_vectors[
            0,
            0,
        ] = float("nan")

    with pytest.raises(
        FloatingPointError,
        match="parameter",
    ):
        module(_inputs())


def test_component_head_count_mutation_is_detected() -> None:
    module = _module(
        _inputs(),
    )
    module.head_reducer.num_heads = 2

    with pytest.raises(
        RuntimeError,
        match="head counts are inconsistent",
    ):
        module.assert_finite_parameters()


def test_component_normalization_policy_mutation_is_detected() -> None:
    module = _module(
        _inputs(),
    )
    module.normalizer.normalization_mode = (
        ATTENTION_NORMALIZATION_TARGET_NODE
    )

    with pytest.raises(
        RuntimeError,
        match="normalization contract changed",
    ):
        module.assert_finite_parameters()


def test_component_head_policy_mutation_is_detected() -> None:
    module = _module(
        _inputs(),
    )
    module.head_reducer.head_reduction = (
        ATTENTION_HEAD_REDUCTION_MAX
    )

    with pytest.raises(
        RuntimeError,
        match="head-reduction contract changed",
    ):
        module.assert_finite_parameters()


def test_normalizer_parameter_injection_is_detected() -> None:
    module = _module(
        _inputs(),
    )
    module.normalizer.register_parameter(
        "unexpected",
        nn.Parameter(torch.ones(1)),
    )

    with pytest.raises(
        RuntimeError,
        match="parameter-free",
    ):
        module.assert_finite_parameters()


def test_head_reducer_buffer_injection_is_detected() -> None:
    module = _module(
        _inputs(),
    )
    module.head_reducer.register_buffer(
        "unexpected",
        torch.ones(1),
    )

    with pytest.raises(
        RuntimeError,
        match="data-dependent buffers",
    ):
        module.assert_finite_parameters()


# =============================================================================
# Stage-method invalid contracts
# =============================================================================


def test_normalize_scores_rejects_wrong_type() -> None:
    module = _module(
        _inputs(),
    )

    with pytest.raises(
        TypeError,
        match="EdgeAttentionScoreOutput",
    ):
        module.normalize_scores(
            object()  # type: ignore[arg-type]
        )


def test_reduce_heads_rejects_wrong_type() -> None:
    module = _module(
        _inputs(),
    )

    with pytest.raises(
        TypeError,
        match="AttentionNormalizationOutput",
    ):
        module.reduce_heads(
            object()  # type: ignore[arg-type]
        )


def test_normalize_scores_rejects_other_mode_score() -> None:
    inputs = _inputs()
    conditioned = _module(
        inputs,
        mode=(
            ATTENTION_MODE_HAZARD_CONDITIONED
        ),
    )
    blind = _module(
        inputs,
        mode=ATTENTION_MODE_HAZARD_BLIND,
    )
    blind_score = blind.score_edges(
        inputs
    )

    with pytest.raises(
        ValueError,
        match="attention mode differs",
    ):
        conditioned.normalize_scores(
            blind_score
        )


def test_reduce_heads_rejects_other_head_count() -> None:
    inputs = _inputs()
    single = _module(
        inputs,
        mode=(
            ATTENTION_MODE_HAZARD_CONDITIONED
        ),
        heads=1,
    )
    multi = _module(
        inputs,
        mode=(
            ATTENTION_MODE_MULTIHEAD_HAZARD_CONDITIONED
        ),
        heads=MULTIHEAD_COUNT,
    )
    _, multi_normalization, _ = (
        multi.compute_stages(inputs)
    )

    with pytest.raises(
        ValueError,
        match="head count differs",
    ):
        single.reduce_heads(
            multi_normalization
        )


# =============================================================================
# Representation
# =============================================================================


def test_extra_repr_is_informative() -> None:
    module = _module(
        _inputs(),
        mode=(
            ATTENTION_MODE_MULTIHEAD_HAZARD_CONDITIONED
        ),
        heads=MULTIHEAD_COUNT,
    )
    text = repr(module)

    assert (
        ATTENTION_MODE_MULTIHEAD_HAZARD_CONDITIONED
        in text
    )
    assert (
        ATTENTION_NORMALIZATION_TARGET_NODE_RELATION
        in text
    )
    assert ATTENTION_HEAD_REDUCTION_MEAN in text
    assert "num_heads=3" in text
    assert "num_relations=3" in text
    assert "attention_enabled=True" in text


# =============================================================================
# Optional CUDA contracts
# =============================================================================


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is not available.",
)
def test_cuda_end_to_end_and_gradients() -> None:
    state = _state_tensor(
        device="cuda",
        requires_grad=True,
    )
    query = _query_tensor(
        device="cuda",
        requires_grad=True,
    )
    inputs = _inputs(
        device="cuda",
        state=state,
        query=query,
    )
    module = _module(
        inputs,
        mode=(
            ATTENTION_MODE_MULTIHEAD_HAZARD_CONDITIONED
        ),
        heads=MULTIHEAD_COUNT,
    )
    output = module(inputs)
    loss = (
        output.edge_weights
        * torch.arange(
            1,
            EDGES + 1,
            dtype=inputs.dtype,
            device="cuda",
        )
    ).sum()
    loss.backward()

    assert output.raw_scores_by_head.device.type == (
        "cuda"
    )
    assert (
        output.normalized_weights_by_head.device.type
        == "cuda"
    )
    assert output.edge_weights.device.type == (
        "cuda"
    )
    assert output.group_ids.device.type == "cuda"
    assert output.group_counts.device.type == (
        "cuda"
    )
    assert state.grad is not None
    assert query.grad is not None


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is not available.",
)
def test_cpu_module_rejects_cuda_inputs() -> None:
    cpu_inputs = _inputs()
    module = _module(
        cpu_inputs
    )
    cuda_inputs = _inputs(
        device="cuda",
    )

    with pytest.raises(
        ValueError,
        match="share one device|device",
    ):
        module(cuda_inputs)
