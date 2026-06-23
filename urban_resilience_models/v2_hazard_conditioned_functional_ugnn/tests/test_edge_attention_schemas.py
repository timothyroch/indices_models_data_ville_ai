"""
Contract tests for exact-relation edge-attention schemas.

Target repository path:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            tests/
                test_edge_attention_schemas.py

Implementation under test:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            functional_message_passing/
                edge_attention/
                    schemas.py

This suite freezes the metadata and tensor contracts of the three internal
edge-attention stages independently from trainable score functions,
grouped-softmax orchestration, head-reduction modules, message construction,
aggregation, and the final functional-message-passing layer.

Covered contracts
-----------------
``EdgeAttentionScoreOutput``
    - exact compiled-relation identity and ordering;
    - canonical score-formula identity;
    - mode-specific input-feature identity;
    - bounded single-head versus future multihead semantics;
    - uniform exact-zero logits;
    - target-node hazard-query requirement;
    - shape, dtype, device, finiteness, and empty-edge behavior;
    - deterministic relation-axis, lineage, value, and complete fingerprints.

``AttentionNormalizationOutput``
    - exact target-node + dense-relation group IDs;
    - exact dense group counts over ``N * R`` possible groups;
    - independent per-head nonnegative normalization;
    - empty-group and empty-edge behavior;
    - singleton-group exactness;
    - reciprocal group-size identity for uniform attention;
    - learned nonuniform distributions;
    - explicit separation between schema validation and recomputation of
      grouped softmax from raw logits.

``AttentionHeadReductionOutput``
    - bounded arithmetic-mean head reduction;
    - preservation of group normalization;
    - singleton and empty-edge behavior;
    - complete stage provenance and fingerprints.

The suite also freezes several boundaries:

- relation gates are not part of edge-attention schema inputs;
- edge attributes remain preserved upstream but are not consumed by the
  bounded score contract;
- stable ontology IDs are metadata, never dense tensor indices;
- attention is a routing coefficient, not a causal-importance field;
- multihead tensor support does not imply head specialization.

Lightweight controlled upstream doubles are patched into the existing
``functional_message_passing.schemas`` module. This isolates failures at the
edge-attention schema boundary instead of coupling these tests to graph
loading, fusion, hazard encoding, or registry construction.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass, replace
from typing import Any

import pytest
import torch

from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.constants import (
    ATTENTION_HEAD_REDUCTION_MAX,
    ATTENTION_HEAD_REDUCTION_MEAN,
    ATTENTION_HEAD_REDUCTION_WEIGHTED_MEAN,
    ATTENTION_MODE_HAZARD_BLIND,
    ATTENTION_MODE_HAZARD_CONDITIONED,
    ATTENTION_MODE_MULTIHEAD_HAZARD_CONDITIONED,
    ATTENTION_MODE_UNIFORM,
    ATTENTION_NORMALIZATION_GLOBAL_RELATION,
    ATTENTION_NORMALIZATION_TARGET_NODE_RELATION,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.functional_message_passing import (
    schemas as fmp_schemas,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.functional_message_passing.edge_attention.schemas import (
    ATTENTION_HEAD_REDUCTION_OUTPUT_SCHEMA_VERSION,
    ATTENTION_NORMALIZATION_OUTPUT_SCHEMA_VERSION,
    EDGE_ATTENTION_INPUT_EXACT_RELATION_EMBEDDING,
    EDGE_ATTENTION_INPUT_SOURCE_NODE_STATE,
    EDGE_ATTENTION_INPUT_TARGET_HAZARD_QUERY,
    EDGE_ATTENTION_INPUT_TARGET_NODE_STATE,
    EDGE_ATTENTION_SCHEMA_MODES,
    EDGE_ATTENTION_SCORE_FUNCTION_ADDITIVE,
    EDGE_ATTENTION_SCORE_FUNCTION_UNIFORM,
    EDGE_ATTENTION_SCORE_OUTPUT_SCHEMA_VERSION,
    AttentionHeadReductionOutput,
    AttentionNormalizationOutput,
    EdgeAttentionOutput,
    EdgeAttentionScoreOutput,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.functional_message_passing.schemas import (
    EdgeAttentionOutput as PublicEdgeAttentionOutput,
    FunctionalMessagePassingInputs,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.functional_message_passing.segment_ops import (
    grouped_softmax,
    segment_counts,
)


NODES = 5
EDGES = 7
GRAPHS = 2
RELATIONS = 3
HIDDEN_DIM = 4
QUERY_DIM = 3
MULTIHEAD_COUNT = 3

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

UNIFORM_FEATURES: tuple[str, ...] = ()
HAZARD_BLIND_FEATURES = (
    EDGE_ATTENTION_INPUT_SOURCE_NODE_STATE,
    EDGE_ATTENTION_INPUT_TARGET_NODE_STATE,
    EDGE_ATTENTION_INPUT_EXACT_RELATION_EMBEDDING,
)
HAZARD_CONDITIONED_FEATURES = (
    EDGE_ATTENTION_INPUT_SOURCE_NODE_STATE,
    EDGE_ATTENTION_INPUT_TARGET_NODE_STATE,
    EDGE_ATTENTION_INPUT_TARGET_HAZARD_QUERY,
    EDGE_ATTENTION_INPUT_EXACT_RELATION_EMBEDDING,
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
        *,
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
            "hazard-query-lineage"
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
# Controlled graph/input helpers
# =============================================================================


def _node_ids(
    count: int = NODES,
) -> tuple[str, ...]:
    return tuple(
        f"node-{index}"
        for index in range(count)
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
    # Group structure:
    #   target 1, relation 0: edges 0 and 1 (size 2)
    #   target 2, relation 1: edge 2       (singleton)
    #   target 2, relation 2: edge 3       (singleton)
    #   target 4, relation 2: edges 4 and 5 (size 2)
    #   target 3, relation 1: edge 6       (singleton)
    #
    # Edge 5 is a valid within-graph self-loop. Nothing in the bounded FMP
    # contract forbids self-loops.
    return torch.tensor(
        [
            [0, 2, 1, 0, 3, 4, 4],
            [1, 1, 2, 2, 4, 4, 3],
        ],
        dtype=torch.long,
        device=device,
    )


def _edge_relation_index(
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


def _node_state(
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
    node_count: int = NODES,
    hidden_dim: int = HIDDEN_DIM,
    node_batch_index: torch.Tensor | None = None,
) -> FakeNodeStateFusionOutput:
    state = (
        torch.arange(
            node_count * hidden_dim,
            dtype=dtype,
            device=device,
        )
        .reshape(node_count, hidden_dim)
        / 10.0
    )

    resolved_batch = (
        _node_batch_index(device=device)
        if node_batch_index is None
        else node_batch_index
    )

    graph_count = (
        int(resolved_batch.max().item())
        + 1
    )

    return FakeNodeStateFusionOutput(
        fused_state=state,
        alignment=FakeAlignment(
            item_ids=_node_ids(node_count),
            node_batch_index=resolved_batch,
            graph_count=graph_count,
        ),
    )


def _hazard_query(
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
    node_batch_index: torch.Tensor | None = None,
    query_dim: int = QUERY_DIM,
) -> FakeHazardQueryEncoding:
    resolved_batch = (
        _node_batch_index(device=device)
        if node_batch_index is None
        else node_batch_index
    )

    query = (
        torch.arange(
            NODES * query_dim,
            dtype=dtype,
            device=device,
        )
        .reshape(NODES, query_dim)
        / 7.0
    )

    return FakeHazardQueryEncoding(
        query=query,
        source_embedding=(
            FakeNodeAlignedHazardEmbeddingLookup(
                node_batch_index=resolved_batch,
            )
        ),
    )


def _registry(
    *,
    fingerprint: str = (
        "compiled-relation-registry"
    ),
    names: tuple[str, ...] = RELATION_NAMES,
    stable_ids: tuple[int, ...] = (
        STABLE_RELATION_IDS
    ),
) -> FakeCompiledRelationRegistry:
    return FakeCompiledRelationRegistry(
        names=names,
        stable_ids=stable_ids,
        controls=CONTROL_RELATIONS,
        fingerprint=fingerprint,
    )


def _graph(
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
    empty_edges: bool = False,
    with_edge_attributes: bool = False,
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
        edge_relations = torch.empty(
            (0,),
            dtype=torch.long,
            device=device,
        )
        edge_batch = torch.empty(
            (0,),
            dtype=torch.long,
            device=device,
        )
        edge_attributes = (
            torch.empty(
                (0, 2),
                dtype=dtype,
                device=device,
            )
            if with_edge_attributes
            else None
        )
    else:
        edge_index = _edge_index(
            device=device,
        )
        edge_relations = (
            _edge_relation_index(
                device=device,
            )
        )
        edge_batch = _edge_batch_index(
            device=device,
        )
        edge_attributes = (
            torch.arange(
                EDGES * 2,
                dtype=dtype,
                device=device,
            ).reshape(EDGES, 2)
            if with_edge_attributes
            else None
        )

    return FakeUrbanGraphBatch(
        external_node_ids=_node_ids(),
        node_batch_index=node_batch,
        edge_index=edge_index,
        edge_relation_type=edge_relations,
        edge_attributes=edge_attributes,
        semantic_edge_weight=None,
        edge_batch_index=edge_batch,
    )


def _inputs(
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
    empty_edges: bool = False,
    with_hazard_query: bool = True,
    with_edge_attributes: bool = False,
    registry: FakeCompiledRelationRegistry | None = None,
) -> FunctionalMessagePassingInputs:
    node_batch = _node_batch_index(
        device=device,
    )

    return FunctionalMessagePassingInputs(
        source_graph=_graph(
            dtype=dtype,
            device=device,
            empty_edges=empty_edges,
            with_edge_attributes=(
                with_edge_attributes
            ),
        ),
        node_state=_node_state(
            dtype=dtype,
            device=device,
            node_batch_index=node_batch,
        ),
        compiled_relation_registry=(
            _registry()
            if registry is None
            else registry
        ),
        hazard_query=(
            _hazard_query(
                dtype=dtype,
                device=device,
                node_batch_index=node_batch,
            )
            if with_hazard_query
            else None
        ),
        source_fingerprint="edge-attention-test-input",
    )


def _feature_names_for_mode(
    mode: str,
) -> tuple[str, ...]:
    if mode == ATTENTION_MODE_UNIFORM:
        return UNIFORM_FEATURES

    if mode == ATTENTION_MODE_HAZARD_BLIND:
        return HAZARD_BLIND_FEATURES

    return HAZARD_CONDITIONED_FEATURES


def _score_function_for_mode(
    mode: str,
) -> str:
    if mode == ATTENTION_MODE_UNIFORM:
        return EDGE_ATTENTION_SCORE_FUNCTION_UNIFORM

    return EDGE_ATTENTION_SCORE_FUNCTION_ADDITIVE


def _head_count_for_mode(
    mode: str,
) -> int:
    if mode == (
        ATTENTION_MODE_MULTIHEAD_HAZARD_CONDITIONED
    ):
        return MULTIHEAD_COUNT

    return 1


def _raw_scores(
    inputs: FunctionalMessagePassingInputs,
    *,
    mode: str,
    num_heads: int | None = None,
    offset: float = 0.0,
) -> torch.Tensor:
    heads = (
        _head_count_for_mode(mode)
        if num_heads is None
        else num_heads
    )

    if mode == ATTENTION_MODE_UNIFORM:
        return torch.zeros(
            (inputs.num_edges, heads),
            dtype=inputs.dtype,
            device=inputs.device,
        )

    if inputs.num_edges == 0:
        return torch.empty(
            (0, heads),
            dtype=inputs.dtype,
            device=inputs.device,
        )

    return (
        torch.arange(
            inputs.num_edges * heads,
            dtype=inputs.dtype,
            device=inputs.device,
        )
        .reshape(inputs.num_edges, heads)
        / 5.0
        + offset
    )


def _score(
    inputs: FunctionalMessagePassingInputs,
    *,
    mode: str = (
        ATTENTION_MODE_HAZARD_CONDITIONED
    ),
    raw_scores: torch.Tensor | None = None,
    num_heads: int | None = None,
    relation_names: tuple[str, ...] | list[str] | None = None,
    stable_relation_ids: tuple[int, ...] | list[int] | None = None,
    compiled_fingerprint: str | None = None,
    input_feature_names: tuple[str, ...] | list[str] | None = None,
    score_function: str | None = None,
    architecture_fingerprint: str = (
        "score-architecture"
    ),
    parameter_fingerprint: str | None = (
        "score-parameters"
    ),
) -> EdgeAttentionScoreOutput:
    resolved_scores = (
        _raw_scores(
            inputs,
            mode=mode,
            num_heads=num_heads,
        )
        if raw_scores is None
        else raw_scores
    )

    return EdgeAttentionScoreOutput(
        raw_scores_by_head=resolved_scores,
        source_inputs=inputs,
        relation_names=(
            inputs.relation_names
            if relation_names is None
            else relation_names
        ),
        stable_relation_ids=(
            inputs.stable_relation_ids
            if stable_relation_ids is None
            else stable_relation_ids
        ),
        compiled_relation_registry_fingerprint=(
            inputs
            .compiled_relation_registry
            .fingerprint()
            if compiled_fingerprint is None
            else compiled_fingerprint
        ),
        attention_mode=mode,
        score_function=(
            _score_function_for_mode(mode)
            if score_function is None
            else score_function
        ),
        input_feature_names=(
            _feature_names_for_mode(mode)
            if input_feature_names is None
            else input_feature_names
        ),
        encoder_architecture_fingerprint=(
            architecture_fingerprint
        ),
        parameter_fingerprint=(
            parameter_fingerprint
        ),
    )


def _normalization(
    score: EdgeAttentionScoreOutput,
    *,
    weights: torch.Tensor | None = None,
    group_ids: torch.Tensor | None = None,
    group_counts: torch.Tensor | None = None,
    normalization_mode: str = (
        ATTENTION_NORMALIZATION_TARGET_NODE_RELATION
    ),
    architecture_fingerprint: str = (
        "normalization-architecture"
    ),
    parameter_fingerprint: str | None = None,
) -> AttentionNormalizationOutput:
    inputs = score.source_inputs

    # Use canonical valid grouping to construct every dependent field that is
    # not itself under test. A supplied group_ids value may intentionally be
    # malformed and must reach AttentionNormalizationOutput unchanged.
    canonical_group_ids = (
        inputs.attention_group_id
    )

    resolved_group_ids = (
        canonical_group_ids
        if group_ids is None
        else group_ids
    )

    resolved_counts = (
        segment_counts(
            canonical_group_ids,
            num_segments=(
                inputs.attention_num_groups
            ),
        )
        if group_counts is None
        else group_counts
    )

    resolved_weights = (
        grouped_softmax(
            score.raw_scores_by_head,
            canonical_group_ids,
            num_segments=(
                inputs.attention_num_groups
            ),
        )
        if weights is None
        else weights
    )

    return AttentionNormalizationOutput(
        normalized_weights_by_head=(
            resolved_weights
        ),
        group_ids=resolved_group_ids,
        group_counts=resolved_counts,
        source_score_output=score,
        normalization_mode=normalization_mode,
        encoder_architecture_fingerprint=(
            architecture_fingerprint
        ),
        parameter_fingerprint=(
            parameter_fingerprint
        ),
    )


def _reduction(
    normalization: AttentionNormalizationOutput,
    *,
    edge_weights: torch.Tensor | None = None,
    head_reduction: str = (
        ATTENTION_HEAD_REDUCTION_MEAN
    ),
    architecture_fingerprint: str = (
        "head-reduction-architecture"
    ),
    parameter_fingerprint: str | None = None,
) -> AttentionHeadReductionOutput:
    resolved_edge_weights = (
        normalization
        .normalized_weights_by_head
        .mean(dim=1)
        if edge_weights is None
        else edge_weights
    )

    return AttentionHeadReductionOutput(
        edge_weights=resolved_edge_weights,
        source_normalization_output=(
            normalization
        ),
        head_reduction=head_reduction,
        encoder_architecture_fingerprint=(
            architecture_fingerprint
        ),
        parameter_fingerprint=(
            parameter_fingerprint
        ),
    )


def _nonuniform_group_normalized_weights(
    score: EdgeAttentionScoreOutput,
) -> torch.Tensor:
    """
    Construct valid learned weights that intentionally need not equal the
    grouped softmax of ``score.raw_scores_by_head``.

    This helper is used to freeze the schema boundary: mathematical
    normalization invariants are validated here, while exact scorer-to-softmax
    equivalence belongs to attention_normalization.py tests.
    """

    inputs = score.source_inputs
    weights = torch.ones(
        (
            inputs.num_edges,
            score.num_heads,
        ),
        dtype=inputs.dtype,
        device=inputs.device,
    )

    counts = segment_counts(
        inputs.attention_group_id,
        num_segments=(
            inputs.attention_num_groups
        ),
    )

    for group_id in range(
        inputs.attention_num_groups
    ):
        indices = torch.nonzero(
            inputs.attention_group_id
            == group_id,
            as_tuple=False,
        ).flatten()

        count = int(
            counts[group_id].item()
        )

        if count == 0:
            continue

        if count == 1:
            weights[indices] = 1.0
            continue

        # The controlled graph has only size-two non-singleton groups.
        # Use an asymmetric but normalized learned routing distribution.
        assert count == 2
        weights[indices[0]] = 0.8
        weights[indices[1]] = 0.2

    return weights


# =============================================================================
# Published identity and public boundary
# =============================================================================


def test_schema_versions_are_nonempty() -> None:
    versions = (
        EDGE_ATTENTION_SCORE_OUTPUT_SCHEMA_VERSION,
        ATTENTION_NORMALIZATION_OUTPUT_SCHEMA_VERSION,
        ATTENTION_HEAD_REDUCTION_OUTPUT_SCHEMA_VERSION,
    )

    for version in versions:
        assert isinstance(version, str)
        assert version.strip()


def test_schema_mode_vocabulary_is_exact_and_unique() -> None:
    assert EDGE_ATTENTION_SCHEMA_MODES == (
        ATTENTION_MODE_UNIFORM,
        ATTENTION_MODE_HAZARD_BLIND,
        ATTENTION_MODE_HAZARD_CONDITIONED,
        ATTENTION_MODE_MULTIHEAD_HAZARD_CONDITIONED,
    )
    assert len(
        set(EDGE_ATTENTION_SCHEMA_MODES)
    ) == len(EDGE_ATTENTION_SCHEMA_MODES)


def test_internal_schema_reexports_public_final_output() -> None:
    assert EdgeAttentionOutput is (
        PublicEdgeAttentionOutput
    )


def test_feature_identity_constants_are_unique() -> None:
    values = (
        EDGE_ATTENTION_INPUT_SOURCE_NODE_STATE,
        EDGE_ATTENTION_INPUT_TARGET_NODE_STATE,
        EDGE_ATTENTION_INPUT_TARGET_HAZARD_QUERY,
        EDGE_ATTENTION_INPUT_EXACT_RELATION_EMBEDDING,
    )

    assert len(set(values)) == len(values)
    assert all(value.strip() for value in values)


def test_score_formula_identity_constants_are_distinct() -> None:
    assert (
        EDGE_ATTENTION_SCORE_FUNCTION_UNIFORM
        != EDGE_ATTENTION_SCORE_FUNCTION_ADDITIVE
    )


# =============================================================================
# EdgeAttentionScoreOutput — valid contracts
# =============================================================================


@pytest.mark.parametrize(
    "mode",
    (
        ATTENTION_MODE_UNIFORM,
        ATTENTION_MODE_HAZARD_BLIND,
        ATTENTION_MODE_HAZARD_CONDITIONED,
        ATTENTION_MODE_MULTIHEAD_HAZARD_CONDITIONED,
    ),
)
def test_score_output_valid_mode_contract(
    mode: str,
) -> None:
    inputs = _inputs(
        with_hazard_query=(
            mode
            not in (
                ATTENTION_MODE_UNIFORM,
                ATTENTION_MODE_HAZARD_BLIND,
            )
        )
    )
    output = _score(
        inputs,
        mode=mode,
    )

    assert output.source_inputs is inputs
    assert output.attention_mode == mode
    assert output.score_function == (
        _score_function_for_mode(mode)
    )
    assert output.input_feature_names == (
        _feature_names_for_mode(mode)
    )
    assert output.num_edges == inputs.num_edges
    assert output.num_heads == (
        _head_count_for_mode(mode)
    )
    assert output.num_relations == RELATIONS
    assert output.relation_names == (
        inputs.relation_names
    )
    assert output.stable_relation_ids == (
        inputs.stable_relation_ids
    )
    assert output.device == inputs.device
    assert output.dtype == inputs.dtype


@pytest.mark.parametrize(
    (
        "mode",
        "uses_source",
        "uses_target",
        "uses_hazard",
        "uses_relation",
    ),
    (
        (
            ATTENTION_MODE_UNIFORM,
            False,
            False,
            False,
            False,
        ),
        (
            ATTENTION_MODE_HAZARD_BLIND,
            True,
            True,
            False,
            True,
        ),
        (
            ATTENTION_MODE_HAZARD_CONDITIONED,
            True,
            True,
            True,
            True,
        ),
        (
            ATTENTION_MODE_MULTIHEAD_HAZARD_CONDITIONED,
            True,
            True,
            True,
            True,
        ),
    ),
)
def test_score_output_feature_use_properties(
    mode: str,
    uses_source: bool,
    uses_target: bool,
    uses_hazard: bool,
    uses_relation: bool,
) -> None:
    inputs = _inputs(
        with_hazard_query=uses_hazard,
    )
    output = _score(
        inputs,
        mode=mode,
    )

    assert (
        output.uses_source_node_state
        is uses_source
    )
    assert (
        output.uses_target_node_state
        is uses_target
    )
    assert output.uses_hazard_query is uses_hazard
    assert (
        output.uses_relation_embedding
        is uses_relation
    )
    assert output.uses_edge_attributes is False


def test_score_output_preserves_edge_attributes_without_consuming_them() -> None:
    inputs = _inputs(
        with_edge_attributes=True,
    )
    output = _score(inputs)

    assert (
        inputs.source_graph.edge_attributes
        is not None
    )
    assert output.uses_edge_attributes is False
    assert output.source_inputs is inputs


def test_score_output_uniform_requires_and_preserves_exact_zero_logits() -> None:
    inputs = _inputs(
        with_hazard_query=False,
    )
    output = _score(
        inputs,
        mode=ATTENTION_MODE_UNIFORM,
    )

    assert torch.equal(
        output.raw_scores_by_head,
        torch.zeros_like(
            output.raw_scores_by_head
        ),
    )


def test_score_output_hazard_blind_does_not_require_query() -> None:
    inputs = _inputs(
        with_hazard_query=False,
    )
    output = _score(
        inputs,
        mode=ATTENTION_MODE_HAZARD_BLIND,
    )

    assert output.uses_hazard_query is False


@pytest.mark.parametrize(
    "mode",
    (
        ATTENTION_MODE_HAZARD_CONDITIONED,
        ATTENTION_MODE_MULTIHEAD_HAZARD_CONDITIONED,
    ),
)
def test_score_output_conditioned_modes_require_node_query(
    mode: str,
) -> None:
    inputs = _inputs(
        with_hazard_query=True,
    )
    output = _score(
        inputs,
        mode=mode,
    )

    assert (
        output.source_inputs.node_hazard_query
        is not None
    )
    assert output.uses_hazard_query is True


def test_score_output_relation_identity_helper() -> None:
    inputs = _inputs()

    identity = (
        EdgeAttentionScoreOutput
        .relation_identity_from_inputs(
            source_inputs=inputs,
        )
    )

    assert identity == {
        "relation_names": (
            inputs.relation_names
        ),
        "stable_relation_ids": (
            inputs.stable_relation_ids
        ),
        "compiled_relation_registry_fingerprint": (
            inputs
            .compiled_relation_registry
            .fingerprint()
        ),
    }


def test_score_output_converts_sequence_metadata_to_tuples() -> None:
    inputs = _inputs()
    output = _score(
        inputs,
        relation_names=list(
            inputs.relation_names
        ),
        stable_relation_ids=list(
            inputs.stable_relation_ids
        ),
        input_feature_names=list(
            HAZARD_CONDITIONED_FEATURES
        ),
    )

    assert isinstance(
        output.relation_names,
        tuple,
    )
    assert isinstance(
        output.stable_relation_ids,
        tuple,
    )
    assert isinstance(
        output.input_feature_names,
        tuple,
    )


def test_score_output_relation_axis_metadata_and_fingerprint() -> None:
    output = _score(_inputs())

    assert output.relation_axis_dict == {
        "relation_names": list(
            RELATION_NAMES
        ),
        "stable_relation_ids": list(
            STABLE_RELATION_IDS
        ),
        "compiled_relation_registry_fingerprint": (
            "compiled-relation-registry"
        ),
    }
    assert (
        output.relation_axis_fingerprint()
        == output.relation_axis_fingerprint()
    )


def test_score_output_empty_edges_are_valid() -> None:
    inputs = _inputs(
        empty_edges=True,
    )
    output = _score(inputs)

    assert output.raw_scores_by_head.shape == (
        0,
        1,
    )
    assert output.num_edges == 0
    assert torch.isfinite(
        output.raw_scores_by_head
    ).all()


def test_score_output_float64_contract() -> None:
    inputs = _inputs(
        dtype=torch.float64,
    )
    output = _score(inputs)

    assert output.dtype == torch.float64
    assert (
        output.raw_scores_by_head.dtype
        == torch.float64
    )


def test_score_output_parameter_fingerprint_may_be_none() -> None:
    output = _score(
        _inputs(),
        parameter_fingerprint=None,
    )

    assert output.parameter_fingerprint is None


def test_score_output_is_frozen() -> None:
    output = _score(_inputs())

    with pytest.raises(FrozenInstanceError):
        output.attention_mode = "changed"  # type: ignore[misc]


# =============================================================================
# EdgeAttentionScoreOutput — fingerprint behavior
# =============================================================================


def test_score_output_fingerprints_are_deterministic() -> None:
    inputs = _inputs()
    raw = _raw_scores(
        inputs,
        mode=ATTENTION_MODE_HAZARD_CONDITIONED,
    )

    first = _score(
        inputs,
        raw_scores=raw.clone(),
    )
    second = _score(
        inputs,
        raw_scores=raw.clone(),
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
    assert first.fingerprint() == (
        second.fingerprint()
    )


def test_score_value_fingerprint_changes_with_logits() -> None:
    inputs = _inputs()
    first = _score(inputs)
    changed = (
        first.raw_scores_by_head
        .clone()
    )
    changed[0, 0] += 1.0
    second = _score(
        inputs,
        raw_scores=changed,
    )

    assert first.lineage_fingerprint() == (
        second.lineage_fingerprint()
    )
    assert first.value_fingerprint() != (
        second.value_fingerprint()
    )
    assert first.fingerprint() != (
        second.fingerprint()
    )


def test_score_lineage_changes_with_architecture_identity() -> None:
    inputs = _inputs()
    first = _score(
        inputs,
        architecture_fingerprint="architecture-a",
    )
    second = _score(
        inputs,
        architecture_fingerprint="architecture-b",
    )

    assert first.value_fingerprint() == (
        second.value_fingerprint()
    )
    assert first.lineage_fingerprint() != (
        second.lineage_fingerprint()
    )


def test_score_value_fingerprint_ignores_requires_grad_flag() -> None:
    inputs = _inputs()
    raw = _raw_scores(
        inputs,
        mode=ATTENTION_MODE_HAZARD_CONDITIONED,
    )

    first = _score(
        inputs,
        raw_scores=raw.clone(),
    )
    second = _score(
        inputs,
        raw_scores=(
            raw.clone().requires_grad_(True)
        ),
    )

    assert first.value_fingerprint() == (
        second.value_fingerprint()
    )


def test_score_lineage_records_scientific_boundary() -> None:
    output = _score(_inputs())
    lineage = output.lineage_dict()

    assert lineage["uses_hazard_query"] is True
    assert lineage["uses_edge_attributes"] is False
    assert lineage["score_function"] == (
        EDGE_ATTENTION_SCORE_FUNCTION_ADDITIVE
    )
    assert lineage["num_heads"] == 1
    assert (
        "relation_axis_fingerprint"
        in lineage
    )


# =============================================================================
# EdgeAttentionScoreOutput — invalid identity and modes
# =============================================================================


def test_score_output_rejects_wrong_source_input_type() -> None:
    with pytest.raises(
        TypeError,
        match="FunctionalMessagePassingInputs",
    ):
        EdgeAttentionScoreOutput(
            raw_scores_by_head=torch.zeros(
                (1, 1),
            ),
            source_inputs=object(),  # type: ignore[arg-type]
            relation_names=("relation",),
            stable_relation_ids=(1,),
            compiled_relation_registry_fingerprint=(
                "compiled"
            ),
            attention_mode=(
                ATTENTION_MODE_UNIFORM
            ),
            score_function=(
                EDGE_ATTENTION_SCORE_FUNCTION_UNIFORM
            ),
            input_feature_names=(),
            encoder_architecture_fingerprint=(
                "architecture"
            ),
        )


def test_relation_identity_helper_rejects_wrong_type() -> None:
    with pytest.raises(
        TypeError,
        match="FunctionalMessagePassingInputs",
    ):
        (
            EdgeAttentionScoreOutput
            .relation_identity_from_inputs(
                source_inputs=object(),  # type: ignore[arg-type]
            )
        )


@pytest.mark.parametrize(
    "mode",
    (
        "",
        "   ",
        "unknown_attention",
    ),
)
def test_score_output_rejects_invalid_attention_mode(
    mode: str,
) -> None:
    inputs = _inputs()

    with pytest.raises(
        ValueError,
        match="attention_mode|does not support",
    ):
        _score(
            inputs,
            mode=mode,
            raw_scores=torch.zeros(
                (inputs.num_edges, 1),
                dtype=inputs.dtype,
            ),
            input_feature_names=(),
            score_function=(
                EDGE_ATTENTION_SCORE_FUNCTION_UNIFORM
            ),
        )


def test_score_output_rejects_non_string_attention_mode() -> None:
    inputs = _inputs()

    with pytest.raises(
        TypeError,
        match="attention_mode",
    ):
        _score(
            inputs,
            mode=3,  # type: ignore[arg-type]
            raw_scores=torch.zeros(
                (inputs.num_edges, 1),
                dtype=inputs.dtype,
            ),
            input_feature_names=(),
            score_function=(
                EDGE_ATTENTION_SCORE_FUNCTION_UNIFORM
            ),
        )


def test_score_output_rejects_noncanonical_whitespace_mode() -> None:
    inputs = _inputs()

    with pytest.raises(
        ValueError,
        match="canonical spelling",
    ):
        _score(
            inputs,
            mode=(
                f" {ATTENTION_MODE_HAZARD_CONDITIONED} "
            ),
            raw_scores=torch.zeros(
                (inputs.num_edges, 1),
                dtype=inputs.dtype,
            ),
            input_feature_names=(
                HAZARD_CONDITIONED_FEATURES
            ),
            score_function=(
                EDGE_ATTENTION_SCORE_FUNCTION_ADDITIVE
            ),
        )


@pytest.mark.parametrize(
    (
        "mode",
        "num_heads",
        "match",
    ),
    (
        (
            ATTENTION_MODE_UNIFORM,
            2,
            "exactly one",
        ),
        (
            ATTENTION_MODE_HAZARD_BLIND,
            2,
            "exactly one",
        ),
        (
            ATTENTION_MODE_HAZARD_CONDITIONED,
            2,
            "exactly one",
        ),
        (
            ATTENTION_MODE_MULTIHEAD_HAZARD_CONDITIONED,
            1,
            "at least two",
        ),
    ),
)
def test_score_output_rejects_invalid_mode_head_combination(
    mode: str,
    num_heads: int,
    match: str,
) -> None:
    inputs = _inputs()
    raw = torch.zeros(
        (inputs.num_edges, num_heads),
        dtype=inputs.dtype,
    )

    with pytest.raises(
        ValueError,
        match=match,
    ):
        _score(
            inputs,
            mode=mode,
            raw_scores=raw,
            num_heads=num_heads,
        )


def test_score_output_rejects_zero_heads() -> None:
    inputs = _inputs()

    with pytest.raises(
        ValueError,
        match="positive integer",
    ):
        _score(
            inputs,
            raw_scores=torch.empty(
                (inputs.num_edges, 0),
                dtype=inputs.dtype,
            ),
        )


@pytest.mark.parametrize(
    "mode",
    (
        ATTENTION_MODE_HAZARD_CONDITIONED,
        ATTENTION_MODE_MULTIHEAD_HAZARD_CONDITIONED,
    ),
)
def test_score_output_rejects_missing_hazard_query(
    mode: str,
) -> None:
    inputs = _inputs(
        with_hazard_query=False,
    )

    with pytest.raises(
        ValueError,
        match="node-aligned hazard query",
    ):
        _score(
            inputs,
            mode=mode,
        )


def test_score_output_rejects_nonzero_uniform_logits() -> None:
    inputs = _inputs(
        with_hazard_query=False,
    )
    raw = torch.zeros(
        (inputs.num_edges, 1),
        dtype=inputs.dtype,
    )
    raw[0, 0] = 0.01

    with pytest.raises(
        ValueError,
        match="exact zero",
    ):
        _score(
            inputs,
            mode=ATTENTION_MODE_UNIFORM,
            raw_scores=raw,
        )


@pytest.mark.parametrize(
    (
        "mode",
        "wrong_function",
    ),
    (
        (
            ATTENTION_MODE_UNIFORM,
            EDGE_ATTENTION_SCORE_FUNCTION_ADDITIVE,
        ),
        (
            ATTENTION_MODE_HAZARD_BLIND,
            EDGE_ATTENTION_SCORE_FUNCTION_UNIFORM,
        ),
        (
            ATTENTION_MODE_HAZARD_CONDITIONED,
            EDGE_ATTENTION_SCORE_FUNCTION_UNIFORM,
        ),
    ),
)
def test_score_output_rejects_wrong_score_formula(
    mode: str,
    wrong_function: str,
) -> None:
    inputs = _inputs()

    with pytest.raises(
        ValueError,
        match="requires score_function",
    ):
        _score(
            inputs,
            mode=mode,
            score_function=wrong_function,
        )


@pytest.mark.parametrize(
    (
        "mode",
        "features",
    ),
    (
        (
            ATTENTION_MODE_UNIFORM,
            HAZARD_BLIND_FEATURES,
        ),
        (
            ATTENTION_MODE_HAZARD_BLIND,
            HAZARD_CONDITIONED_FEATURES,
        ),
        (
            ATTENTION_MODE_HAZARD_CONDITIONED,
            HAZARD_BLIND_FEATURES,
        ),
    ),
)
def test_score_output_rejects_wrong_input_feature_identity(
    mode: str,
    features: tuple[str, ...],
) -> None:
    inputs = _inputs()

    with pytest.raises(
        ValueError,
        match="requires input_feature_names",
    ):
        _score(
            inputs,
            mode=mode,
            input_feature_names=features,
        )


def test_score_output_rejects_duplicate_input_feature_names() -> None:
    inputs = _inputs(
        with_hazard_query=False,
    )

    with pytest.raises(
        ValueError,
        match="duplicates",
    ):
        _score(
            inputs,
            mode=ATTENTION_MODE_HAZARD_BLIND,
            input_feature_names=(
                EDGE_ATTENTION_INPUT_SOURCE_NODE_STATE,
                EDGE_ATTENTION_INPUT_SOURCE_NODE_STATE,
                EDGE_ATTENTION_INPUT_EXACT_RELATION_EMBEDDING,
            ),
        )


def test_score_output_rejects_wrong_relation_order() -> None:
    inputs = _inputs()

    with pytest.raises(
        ValueError,
        match="relation_names",
    ):
        _score(
            inputs,
            relation_names=(
                RELATION_NAMES[1],
                RELATION_NAMES[0],
                RELATION_NAMES[2],
            ),
        )


def test_score_output_rejects_wrong_stable_relation_order() -> None:
    inputs = _inputs()

    with pytest.raises(
        ValueError,
        match="stable_relation_ids",
    ):
        _score(
            inputs,
            stable_relation_ids=(
                STABLE_RELATION_IDS[1],
                STABLE_RELATION_IDS[0],
                STABLE_RELATION_IDS[2],
            ),
        )


def test_score_output_rejects_different_compiled_registry() -> None:
    inputs = _inputs()

    with pytest.raises(
        ValueError,
        match="different compiled relation registry",
    ):
        _score(
            inputs,
            compiled_fingerprint=(
                "different-registry"
            ),
        )


def test_score_output_rejects_duplicate_relation_names() -> None:
    inputs = _inputs()

    with pytest.raises(
        ValueError,
        match="duplicates",
    ):
        _score(
            inputs,
            relation_names=(
                RELATION_NAMES[0],
                RELATION_NAMES[0],
                RELATION_NAMES[2],
            ),
        )


def test_score_output_rejects_duplicate_stable_ids() -> None:
    inputs = _inputs()

    with pytest.raises(
        ValueError,
        match="duplicates",
    ):
        _score(
            inputs,
            stable_relation_ids=(
                STABLE_RELATION_IDS[0],
                STABLE_RELATION_IDS[0],
                STABLE_RELATION_IDS[2],
            ),
        )


def test_score_output_rejects_negative_stable_id() -> None:
    inputs = _inputs()

    with pytest.raises(
        ValueError,
        match="nonnegative",
    ):
        _score(
            inputs,
            stable_relation_ids=(
                -1,
                STABLE_RELATION_IDS[1],
                STABLE_RELATION_IDS[2],
            ),
        )


def test_score_output_rejects_boolean_stable_id() -> None:
    inputs = _inputs()

    with pytest.raises(
        TypeError,
        match="integer",
    ):
        _score(
            inputs,
            stable_relation_ids=(
                True,  # type: ignore[arg-type]
                STABLE_RELATION_IDS[1],
                STABLE_RELATION_IDS[2],
            ),
        )


def test_score_output_rejects_relation_identity_length_mismatch() -> None:
    inputs = _inputs()

    with pytest.raises(
        ValueError,
        match="must align",
    ):
        _score(
            inputs,
            stable_relation_ids=(
                STABLE_RELATION_IDS[0],
                STABLE_RELATION_IDS[1],
            ),
        )


@pytest.mark.parametrize(
    "field",
    (
        "compiled_relation_registry_fingerprint",
        "encoder_architecture_fingerprint",
        "schema_version",
    ),
)
def test_score_output_rejects_blank_required_string(
    field: str,
) -> None:
    output = _score(_inputs())

    with pytest.raises(
        ValueError,
        match="non-empty",
    ):
        replace(
            output,
            **{field: ""},
        )


def test_score_output_rejects_blank_parameter_fingerprint() -> None:
    with pytest.raises(
        ValueError,
        match="non-empty",
    ):
        _score(
            _inputs(),
            parameter_fingerprint="",
        )


# =============================================================================
# EdgeAttentionScoreOutput — invalid raw-score tensors
# =============================================================================


def test_score_output_rejects_non_tensor_scores() -> None:
    inputs = _inputs()

    with pytest.raises(
        TypeError,
        match="tensor",
    ):
        _score(
            inputs,
            raw_scores=[[0.0]],  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "raw",
    (
        torch.zeros(EDGES),
        torch.zeros(1, EDGES, 1),
    ),
)
def test_score_output_rejects_wrong_score_rank(
    raw: torch.Tensor,
) -> None:
    inputs = _inputs()

    with pytest.raises(
        ValueError,
        match="rank 2",
    ):
        _score(
            inputs,
            raw_scores=raw,
        )


def test_score_output_rejects_wrong_edge_count() -> None:
    inputs = _inputs()

    with pytest.raises(
        ValueError,
        match="shape",
    ):
        _score(
            inputs,
            raw_scores=torch.zeros(
                (inputs.num_edges + 1, 1),
                dtype=inputs.dtype,
            ),
        )


def test_score_output_rejects_integer_scores() -> None:
    inputs = _inputs()

    with pytest.raises(
        ValueError,
        match="floating-point",
    ):
        _score(
            inputs,
            raw_scores=torch.zeros(
                (inputs.num_edges, 1),
                dtype=torch.long,
            ),
        )


@pytest.mark.parametrize(
    "invalid_value",
    (
        float("nan"),
        float("inf"),
        float("-inf"),
    ),
)
def test_score_output_rejects_nonfinite_scores(
    invalid_value: float,
) -> None:
    inputs = _inputs()
    raw = _raw_scores(
        inputs,
        mode=ATTENTION_MODE_HAZARD_CONDITIONED,
    )
    raw[0, 0] = invalid_value

    with pytest.raises(
        ValueError,
        match="finite",
    ):
        _score(
            inputs,
            raw_scores=raw,
        )


def test_score_output_rejects_float_dtype_mismatch() -> None:
    inputs = _inputs(
        dtype=torch.float32,
    )
    raw = _raw_scores(
        inputs,
        mode=ATTENTION_MODE_HAZARD_CONDITIONED,
    ).to(dtype=torch.float64)

    with pytest.raises(
        ValueError,
        match="dtype",
    ):
        _score(
            inputs,
            raw_scores=raw,
        )


# =============================================================================
# AttentionNormalizationOutput — valid contracts
# =============================================================================


@pytest.mark.parametrize(
    "mode",
    (
        ATTENTION_MODE_UNIFORM,
        ATTENTION_MODE_HAZARD_BLIND,
        ATTENTION_MODE_HAZARD_CONDITIONED,
        ATTENTION_MODE_MULTIHEAD_HAZARD_CONDITIONED,
    ),
)
def test_normalization_output_valid_contract(
    mode: str,
) -> None:
    inputs = _inputs(
        with_hazard_query=(
            mode
            not in (
                ATTENTION_MODE_UNIFORM,
                ATTENTION_MODE_HAZARD_BLIND,
            )
        )
    )
    score = _score(
        inputs,
        mode=mode,
    )
    output = _normalization(score)

    assert output.source_score_output is score
    assert output.source_inputs is inputs
    assert output.raw_scores_by_head is (
        score.raw_scores_by_head
    )
    assert output.attention_mode == mode
    assert output.score_function == (
        score.score_function
    )
    assert output.num_heads == score.num_heads
    assert output.num_groups == (
        inputs.attention_num_groups
    )
    assert output.normalization_mode == (
        ATTENTION_NORMALIZATION_TARGET_NODE_RELATION
    )
    assert output.device == inputs.device
    assert output.dtype == inputs.dtype
    assert torch.equal(
        output.group_ids,
        inputs.attention_group_id,
    )
    assert torch.equal(
        output.group_counts,
        segment_counts(
            inputs.attention_group_id,
            num_segments=(
                inputs.attention_num_groups
            ),
        ),
    )


def test_normalization_group_ids_use_target_and_exact_relation() -> None:
    inputs = _inputs()
    output = _normalization(
        _score(inputs)
    )

    expected = (
        inputs.target_index
        * inputs.num_relations
        + inputs.edge_relation_index
    )

    assert torch.equal(
        output.group_ids,
        expected,
    )


def test_normalization_group_count_shape_includes_absent_groups() -> None:
    inputs = _inputs()
    output = _normalization(
        _score(inputs)
    )

    assert output.group_counts.shape == (
        NODES * RELATIONS,
    )
    assert int(
        output.group_counts.sum().item()
    ) == EDGES
    assert output.num_nonempty_groups == 5
    assert int(
        output.group_presence.sum().item()
    ) == 5
    assert bool(
        (output.group_counts == 0)
        .any()
        .item()
    )


def test_uniform_normalization_is_reciprocal_group_size() -> None:
    inputs = _inputs(
        with_hazard_query=False,
    )
    output = _normalization(
        _score(
            inputs,
            mode=ATTENTION_MODE_UNIFORM,
        )
    )

    expected = (
        torch.ones_like(
            output.normalized_weights_by_head
        )
        / output.group_counts[
            output.group_ids
        ].to(
            dtype=inputs.dtype,
        ).unsqueeze(1)
    )

    assert torch.equal(
        output.normalized_weights_by_head,
        expected,
    )


def test_learned_normalization_accepts_nonuniform_valid_distribution() -> None:
    score = _score(_inputs())
    custom = (
        _nonuniform_group_normalized_weights(
            score
        )
    )
    output = _normalization(
        score,
        weights=custom,
    )

    repeated_group = (
        output.group_ids
        == output.group_ids[0]
    )
    observed = (
        output
        .normalized_weights_by_head[
            repeated_group,
            0,
        ]
    )

    assert torch.allclose(
        observed,
        torch.tensor(
            [0.8, 0.2],
            dtype=output.dtype,
            device=output.device,
        ),
    )


def test_normalization_schema_does_not_recompute_softmax_from_logits() -> None:
    score = _score(_inputs())
    custom = (
        _nonuniform_group_normalized_weights(
            score
        )
    )
    actual_softmax = grouped_softmax(
        score.raw_scores_by_head,
        score.source_inputs.attention_group_id,
        num_segments=(
            score
            .source_inputs
            .attention_num_groups
        ),
    )

    assert not torch.allclose(
        custom,
        actual_softmax,
    )

    output = _normalization(
        score,
        weights=custom,
    )

    assert torch.equal(
        output.normalized_weights_by_head,
        custom,
    )


def test_normalization_singleton_groups_receive_exact_one() -> None:
    output = _normalization(
        _score(_inputs())
    )
    singleton_edges = (
        output.group_counts[
            output.group_ids
        ]
        == 1
    )

    assert bool(singleton_edges.any().item())
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


def test_normalization_empty_edges_are_valid() -> None:
    inputs = _inputs(
        empty_edges=True,
    )
    output = _normalization(
        _score(inputs)
    )

    assert output.normalized_weights_by_head.shape == (
        0,
        1,
    )
    assert output.group_ids.shape == (0,)
    assert output.group_counts.shape == (
        NODES * RELATIONS,
    )
    assert torch.equal(
        output.group_counts,
        torch.zeros_like(
            output.group_counts
        ),
    )
    assert output.num_nonempty_groups == 0


def test_normalization_float64_contract() -> None:
    inputs = _inputs(
        dtype=torch.float64,
    )
    output = _normalization(
        _score(inputs)
    )

    assert output.dtype == torch.float64
    assert (
        output.normalized_weights_by_head.dtype
        == torch.float64
    )


def test_normalization_parameter_fingerprint_may_be_none() -> None:
    output = _normalization(
        _score(_inputs()),
        parameter_fingerprint=None,
    )

    assert output.parameter_fingerprint is None


def test_normalization_is_frozen() -> None:
    output = _normalization(
        _score(_inputs())
    )

    with pytest.raises(FrozenInstanceError):
        output.normalization_mode = "changed"  # type: ignore[misc]


# =============================================================================
# AttentionNormalizationOutput — fingerprints
# =============================================================================


def test_normalization_fingerprints_are_deterministic() -> None:
    score = _score(_inputs())
    first = _normalization(score)
    second = _normalization(
        _score(
            score.source_inputs,
            raw_scores=(
                score.raw_scores_by_head.clone()
            ),
        )
    )

    assert first.lineage_fingerprint() == (
        second.lineage_fingerprint()
    )
    assert first.value_fingerprint() == (
        second.value_fingerprint()
    )
    assert first.fingerprint() == (
        second.fingerprint()
    )


def test_normalization_value_fingerprint_changes_with_weights() -> None:
    score = _score(_inputs())
    first = _normalization(score)
    custom = (
        _nonuniform_group_normalized_weights(
            score
        )
    )
    second = _normalization(
        score,
        weights=custom,
    )

    assert first.value_fingerprint() != (
        second.value_fingerprint()
    )
    assert first.fingerprint() != (
        second.fingerprint()
    )


def test_normalization_lineage_changes_with_architecture() -> None:
    score = _score(_inputs())
    first = _normalization(
        score,
        architecture_fingerprint="normalizer-a",
    )
    second = _normalization(
        score,
        architecture_fingerprint="normalizer-b",
    )

    assert first.value_fingerprint() == (
        second.value_fingerprint()
    )
    assert first.lineage_fingerprint() != (
        second.lineage_fingerprint()
    )


def test_normalization_lineage_records_group_semantics() -> None:
    output = _normalization(
        _score(_inputs())
    )
    lineage = output.lineage_dict()

    assert lineage["group_key"] == (
        "target_node_exact_relation"
    )
    assert lineage["num_groups"] == (
        NODES * RELATIONS
    )
    assert lineage["num_nonempty_groups"] == 5


# =============================================================================
# AttentionNormalizationOutput — invalid tensors and grouping
# =============================================================================


def test_normalization_rejects_wrong_source_type() -> None:
    inputs = _inputs()

    with pytest.raises(
        TypeError,
        match="EdgeAttentionScoreOutput",
    ):
        AttentionNormalizationOutput(
            normalized_weights_by_head=(
                torch.ones(
                    (inputs.num_edges, 1),
                    dtype=inputs.dtype,
                )
            ),
            group_ids=inputs.attention_group_id,
            group_counts=segment_counts(
                inputs.attention_group_id,
                num_segments=(
                    inputs.attention_num_groups
                ),
            ),
            source_score_output=object(),  # type: ignore[arg-type]
            normalization_mode=(
                ATTENTION_NORMALIZATION_TARGET_NODE_RELATION
            ),
            encoder_architecture_fingerprint=(
                "architecture"
            ),
        )


def test_normalization_rejects_non_tensor_weights() -> None:
    score = _score(_inputs())

    with pytest.raises(
        TypeError,
        match="tensor",
    ):
        _normalization(
            score,
            weights=[[1.0]],  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "weights",
    (
        torch.ones(EDGES),
        torch.ones(1, EDGES, 1),
    ),
)
def test_normalization_rejects_wrong_weight_rank(
    weights: torch.Tensor,
) -> None:
    score = _score(_inputs())

    with pytest.raises(
        ValueError,
        match="rank 2",
    ):
        _normalization(
            score,
            weights=weights,
        )


def test_normalization_rejects_wrong_weight_shape() -> None:
    score = _score(_inputs())

    with pytest.raises(
        ValueError,
        match="shape",
    ):
        _normalization(
            score,
            weights=torch.ones(
                (
                    score.source_inputs.num_edges
                    + 1,
                    score.num_heads,
                ),
                dtype=score.dtype,
            ),
        )


def test_normalization_rejects_integer_weights() -> None:
    score = _score(_inputs())

    with pytest.raises(
        ValueError,
        match="floating-point",
    ):
        _normalization(
            score,
            weights=torch.ones(
                (
                    score.source_inputs.num_edges,
                    score.num_heads,
                ),
                dtype=torch.long,
            ),
        )


@pytest.mark.parametrize(
    "invalid_value",
    (
        float("nan"),
        float("inf"),
        float("-inf"),
    ),
)
def test_normalization_rejects_nonfinite_weights(
    invalid_value: float,
) -> None:
    score = _score(_inputs())
    weights = grouped_softmax(
        score.raw_scores_by_head,
        score.source_inputs.attention_group_id,
        num_segments=(
            score
            .source_inputs
            .attention_num_groups
        ),
    )
    weights[0, 0] = invalid_value

    with pytest.raises(
        ValueError,
        match="finite",
    ):
        _normalization(
            score,
            weights=weights,
        )


def test_normalization_rejects_float_dtype_mismatch() -> None:
    score = _score(
        _inputs(dtype=torch.float32)
    )
    weights = grouped_softmax(
        score.raw_scores_by_head,
        score.source_inputs.attention_group_id,
        num_segments=(
            score
            .source_inputs
            .attention_num_groups
        ),
    ).to(dtype=torch.float64)

    with pytest.raises(
        ValueError,
        match="dtype",
    ):
        _normalization(
            score,
            weights=weights,
        )


def test_normalization_rejects_negative_weights() -> None:
    score = _score(_inputs())
    custom = (
        _nonuniform_group_normalized_weights(
            score
        )
    )
    custom[0, 0] = -0.1
    custom[1, 0] = 1.1

    with pytest.raises(
        ValueError,
        match="nonnegative",
    ):
        _normalization(
            score,
            weights=custom,
        )


def test_normalization_rejects_group_sum_not_one() -> None:
    score = _score(_inputs())
    weights = (
        _nonuniform_group_normalized_weights(
            score
        )
    )
    weights[0, 0] = 0.6
    weights[1, 0] = 0.2

    with pytest.raises(
        ValueError,
        match="sum to one",
    ):
        _normalization(
            score,
            weights=weights,
        )


def test_uniform_normalization_rejects_nonuniform_valid_distribution() -> None:
    inputs = _inputs(
        with_hazard_query=False,
    )
    score = _score(
        inputs,
        mode=ATTENTION_MODE_UNIFORM,
    )
    custom = (
        _nonuniform_group_normalized_weights(
            score
        )
    )

    with pytest.raises(
        ValueError,
        match="reciprocal group-size",
    ):
        _normalization(
            score,
            weights=custom,
        )


def test_normalization_rejects_non_tensor_group_ids() -> None:
    score = _score(_inputs())

    with pytest.raises(
        TypeError,
        match="group_ids.*tensor",
    ):
        _normalization(
            score,
            group_ids=[0] * EDGES,  # type: ignore[arg-type]
        )


def test_normalization_rejects_wrong_group_id_rank() -> None:
    score = _score(_inputs())

    with pytest.raises(
        ValueError,
        match="rank 1",
    ):
        _normalization(
            score,
            group_ids=(
                score
                .source_inputs
                .attention_group_id
                .unsqueeze(0)
            ),
        )


def test_normalization_rejects_non_long_group_ids() -> None:
    score = _score(_inputs())

    with pytest.raises(
        ValueError,
        match="torch.long",
    ):
        _normalization(
            score,
            group_ids=(
                score
                .source_inputs
                .attention_group_id
                .to(dtype=torch.int32)
            ),
        )


def test_normalization_rejects_wrong_group_id_shape() -> None:
    score = _score(_inputs())

    with pytest.raises(
        ValueError,
        match="shape",
    ):
        _normalization(
            score,
            group_ids=(
                score
                .source_inputs
                .attention_group_id[:-1]
            ),
        )


def test_normalization_rejects_out_of_range_group_ids() -> None:
    score = _score(_inputs())
    group_ids = (
        score
        .source_inputs
        .attention_group_id
        .clone()
    )
    group_ids[0] = (
        score
        .source_inputs
        .attention_num_groups
    )

    with pytest.raises(
        ValueError,
        match="out-of-range",
    ):
        _normalization(
            score,
            group_ids=group_ids,
            weights=torch.ones_like(
                score.raw_scores_by_head
            ),
        )


def test_normalization_rejects_semantically_wrong_group_ids() -> None:
    score = _score(_inputs())
    group_ids = (
        score
        .source_inputs
        .attention_group_id
        .clone()
    )
    group_ids[0], group_ids[2] = (
        group_ids[2].clone(),
        group_ids[0].clone(),
    )

    with pytest.raises(
        ValueError,
        match="target node.*relation",
    ):
        _normalization(
            score,
            group_ids=group_ids,
            weights=grouped_softmax(
                score.raw_scores_by_head,
                group_ids,
                num_segments=(
                    score
                    .source_inputs
                    .attention_num_groups
                ),
            ),
        )


def test_normalization_rejects_non_tensor_group_counts() -> None:
    score = _score(_inputs())

    with pytest.raises(
        TypeError,
        match="group_counts.*tensor",
    ):
        _normalization(
            score,
            group_counts=[0] * (
                score
                .source_inputs
                .attention_num_groups
            ),  # type: ignore[arg-type]
        )


def test_normalization_rejects_wrong_group_count_rank() -> None:
    score = _score(_inputs())
    counts = segment_counts(
        score.source_inputs.attention_group_id,
        num_segments=(
            score
            .source_inputs
            .attention_num_groups
        ),
    ).unsqueeze(0)

    with pytest.raises(
        ValueError,
        match="rank 1",
    ):
        _normalization(
            score,
            group_counts=counts,
        )


def test_normalization_rejects_non_long_group_counts() -> None:
    score = _score(_inputs())
    counts = segment_counts(
        score.source_inputs.attention_group_id,
        num_segments=(
            score
            .source_inputs
            .attention_num_groups
        ),
    ).to(dtype=torch.int32)

    with pytest.raises(
        ValueError,
        match="torch.long",
    ):
        _normalization(
            score,
            group_counts=counts,
        )


def test_normalization_rejects_wrong_group_count_shape() -> None:
    score = _score(_inputs())
    counts = segment_counts(
        score.source_inputs.attention_group_id,
        num_segments=(
            score
            .source_inputs
            .attention_num_groups
        ),
    )[:-1]

    with pytest.raises(
        ValueError,
        match="shape",
    ):
        _normalization(
            score,
            group_counts=counts,
        )


def test_normalization_rejects_negative_group_count() -> None:
    score = _score(_inputs())
    counts = segment_counts(
        score.source_inputs.attention_group_id,
        num_segments=(
            score
            .source_inputs
            .attention_num_groups
        ),
    )
    counts[0] = -1

    with pytest.raises(
        ValueError,
        match="nonnegative",
    ):
        _normalization(
            score,
            group_counts=counts,
        )


def test_normalization_rejects_counts_not_implied_by_groups() -> None:
    score = _score(_inputs())
    counts = segment_counts(
        score.source_inputs.attention_group_id,
        num_segments=(
            score
            .source_inputs
            .attention_num_groups
        ),
    )
    nonempty = torch.nonzero(
        counts > 0,
        as_tuple=False,
    ).flatten()
    counts[nonempty[0]] += 1

    with pytest.raises(
        ValueError,
        match="counts implied",
    ):
        _normalization(
            score,
            group_counts=counts,
        )


@pytest.mark.parametrize(
    "normalization_mode",
    (
        "",
        ATTENTION_NORMALIZATION_GLOBAL_RELATION,
    ),
)
def test_normalization_rejects_unsupported_mode(
    normalization_mode: str,
) -> None:
    score = _score(_inputs())

    with pytest.raises(ValueError):
        _normalization(
            score,
            normalization_mode=(
                normalization_mode
            ),
        )


@pytest.mark.parametrize(
    "field",
    (
        "encoder_architecture_fingerprint",
        "schema_version",
    ),
)
def test_normalization_rejects_blank_required_string(
    field: str,
) -> None:
    output = _normalization(
        _score(_inputs())
    )

    with pytest.raises(
        ValueError,
        match="non-empty",
    ):
        replace(
            output,
            **{field: ""},
        )


def test_normalization_rejects_blank_parameter_fingerprint() -> None:
    with pytest.raises(
        ValueError,
        match="non-empty",
    ):
        _normalization(
            _score(_inputs()),
            parameter_fingerprint="",
        )


# =============================================================================
# AttentionHeadReductionOutput — valid contracts
# =============================================================================


@pytest.mark.parametrize(
    "mode",
    (
        ATTENTION_MODE_UNIFORM,
        ATTENTION_MODE_HAZARD_BLIND,
        ATTENTION_MODE_HAZARD_CONDITIONED,
        ATTENTION_MODE_MULTIHEAD_HAZARD_CONDITIONED,
    ),
)
def test_head_reduction_valid_contract(
    mode: str,
) -> None:
    inputs = _inputs(
        with_hazard_query=(
            mode
            not in (
                ATTENTION_MODE_UNIFORM,
                ATTENTION_MODE_HAZARD_BLIND,
            )
        )
    )
    normalization = _normalization(
        _score(
            inputs,
            mode=mode,
        )
    )
    output = _reduction(
        normalization
    )

    assert (
        output.source_normalization_output
        is normalization
    )
    assert output.source_inputs is inputs
    assert output.source_score_output is (
        normalization.source_score_output
    )
    assert output.raw_scores_by_head is (
        normalization.raw_scores_by_head
    )
    assert output.normalized_weights_by_head is (
        normalization
        .normalized_weights_by_head
    )
    assert output.group_ids is (
        normalization.group_ids
    )
    assert output.group_counts is (
        normalization.group_counts
    )
    assert output.attention_mode == mode
    assert output.normalization_mode == (
        ATTENTION_NORMALIZATION_TARGET_NODE_RELATION
    )
    assert output.num_heads == (
        normalization.num_heads
    )
    assert output.num_groups == (
        normalization.num_groups
    )
    assert output.head_reduction == (
        ATTENTION_HEAD_REDUCTION_MEAN
    )
    assert output.device == inputs.device
    assert output.dtype == inputs.dtype


def test_head_reduction_equals_arithmetic_mean() -> None:
    normalization = _normalization(
        _score(
            _inputs(),
            mode=(
                ATTENTION_MODE_MULTIHEAD_HAZARD_CONDITIONED
            ),
        )
    )
    output = _reduction(
        normalization
    )

    assert torch.allclose(
        output.edge_weights,
        normalization
        .normalized_weights_by_head
        .mean(dim=1),
    )


def test_head_reduction_remains_group_normalized() -> None:
    output = _reduction(
        _normalization(
            _score(
                _inputs(),
                mode=(
                    ATTENTION_MODE_MULTIHEAD_HAZARD_CONDITIONED
                ),
            )
        )
    )
    sums = torch.zeros(
        output.num_groups,
        dtype=output.dtype,
        device=output.device,
    )
    sums.index_add_(
        0,
        output.group_ids,
        output.edge_weights,
    )
    present = output.group_counts > 0

    assert torch.allclose(
        sums[present],
        torch.ones_like(
            sums[present]
        ),
    )
    assert torch.equal(
        sums[~present],
        torch.zeros_like(
            sums[~present]
        ),
    )


def test_head_reduction_singletons_remain_exact_one() -> None:
    output = _reduction(
        _normalization(
            _score(_inputs())
        )
    )
    singleton_edges = (
        output.group_counts[
            output.group_ids
        ]
        == 1
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


def test_head_reduction_empty_edges_are_valid() -> None:
    output = _reduction(
        _normalization(
            _score(
                _inputs(
                    empty_edges=True,
                )
            )
        )
    )

    assert output.edge_weights.shape == (0,)
    assert output.num_groups == (
        NODES * RELATIONS
    )


def test_head_reduction_float64_contract() -> None:
    output = _reduction(
        _normalization(
            _score(
                _inputs(
                    dtype=torch.float64,
                )
            )
        )
    )

    assert output.dtype == torch.float64


def test_head_reduction_parameter_fingerprint_may_be_none() -> None:
    output = _reduction(
        _normalization(
            _score(_inputs())
        ),
        parameter_fingerprint=None,
    )

    assert output.parameter_fingerprint is None


def test_head_reduction_is_frozen() -> None:
    output = _reduction(
        _normalization(
            _score(_inputs())
        )
    )

    with pytest.raises(FrozenInstanceError):
        output.head_reduction = "changed"  # type: ignore[misc]


# =============================================================================
# AttentionHeadReductionOutput — fingerprints
# =============================================================================


def test_head_reduction_fingerprints_are_deterministic() -> None:
    score = _score(_inputs())
    first = _reduction(
        _normalization(score)
    )
    second = _reduction(
        _normalization(
            _score(
                score.source_inputs,
                raw_scores=(
                    score
                    .raw_scores_by_head
                    .clone()
                ),
            )
        )
    )

    assert first.lineage_fingerprint() == (
        second.lineage_fingerprint()
    )
    assert first.value_fingerprint() == (
        second.value_fingerprint()
    )
    assert first.fingerprint() == (
        second.fingerprint()
    )


def test_head_reduction_value_fingerprint_changes_with_weights() -> None:
    score = _score(_inputs())
    first_normalization = _normalization(
        score
    )
    second_normalization = _normalization(
        score,
        weights=(
            _nonuniform_group_normalized_weights(
                score
            )
        ),
    )

    first = _reduction(
        first_normalization
    )
    second = _reduction(
        second_normalization
    )

    assert first.value_fingerprint() != (
        second.value_fingerprint()
    )
    assert first.fingerprint() != (
        second.fingerprint()
    )


def test_head_reduction_lineage_changes_with_architecture() -> None:
    normalization = _normalization(
        _score(_inputs())
    )
    first = _reduction(
        normalization,
        architecture_fingerprint="reduction-a",
    )
    second = _reduction(
        normalization,
        architecture_fingerprint="reduction-b",
    )

    assert first.value_fingerprint() == (
        second.value_fingerprint()
    )
    assert first.lineage_fingerprint() != (
        second.lineage_fingerprint()
    )


def test_head_reduction_lineage_records_policy_and_head_count() -> None:
    output = _reduction(
        _normalization(
            _score(
                _inputs(),
                mode=(
                    ATTENTION_MODE_MULTIHEAD_HAZARD_CONDITIONED
                ),
            )
        )
    )
    lineage = output.lineage_dict()

    assert lineage["head_reduction"] == (
        ATTENTION_HEAD_REDUCTION_MEAN
    )
    assert lineage["num_heads"] == (
        MULTIHEAD_COUNT
    )


# =============================================================================
# AttentionHeadReductionOutput — invalid contracts
# =============================================================================


def test_head_reduction_rejects_wrong_source_type() -> None:
    with pytest.raises(
        TypeError,
        match="AttentionNormalizationOutput",
    ):
        AttentionHeadReductionOutput(
            edge_weights=torch.ones(EDGES),
            source_normalization_output=object(),  # type: ignore[arg-type]
            head_reduction=(
                ATTENTION_HEAD_REDUCTION_MEAN
            ),
            encoder_architecture_fingerprint=(
                "architecture"
            ),
        )


def test_head_reduction_rejects_non_tensor_weights() -> None:
    normalization = _normalization(
        _score(_inputs())
    )

    with pytest.raises(
        TypeError,
        match="tensor",
    ):
        _reduction(
            normalization,
            edge_weights=[1.0],  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "weights",
    (
        torch.ones(EDGES, 1),
        torch.ones(1, EDGES, 1),
    ),
)
def test_head_reduction_rejects_wrong_weight_rank(
    weights: torch.Tensor,
) -> None:
    normalization = _normalization(
        _score(_inputs())
    )

    with pytest.raises(
        ValueError,
        match="rank 1",
    ):
        _reduction(
            normalization,
            edge_weights=weights,
        )


def test_head_reduction_rejects_wrong_weight_shape() -> None:
    normalization = _normalization(
        _score(_inputs())
    )

    with pytest.raises(
        ValueError,
        match="shape",
    ):
        _reduction(
            normalization,
            edge_weights=torch.ones(
                EDGES + 1,
                dtype=normalization.dtype,
            ),
        )


def test_head_reduction_rejects_integer_weights() -> None:
    normalization = _normalization(
        _score(_inputs())
    )

    with pytest.raises(
        ValueError,
        match="floating-point",
    ):
        _reduction(
            normalization,
            edge_weights=torch.ones(
                EDGES,
                dtype=torch.long,
            ),
        )


@pytest.mark.parametrize(
    "invalid_value",
    (
        float("nan"),
        float("inf"),
        float("-inf"),
    ),
)
def test_head_reduction_rejects_nonfinite_weights(
    invalid_value: float,
) -> None:
    normalization = _normalization(
        _score(_inputs())
    )
    weights = (
        normalization
        .normalized_weights_by_head
        .mean(dim=1)
    )
    weights[0] = invalid_value

    with pytest.raises(
        ValueError,
        match="finite",
    ):
        _reduction(
            normalization,
            edge_weights=weights,
        )


def test_head_reduction_rejects_float_dtype_mismatch() -> None:
    normalization = _normalization(
        _score(
            _inputs(dtype=torch.float32)
        )
    )
    weights = (
        normalization
        .normalized_weights_by_head
        .mean(dim=1)
        .to(dtype=torch.float64)
    )

    with pytest.raises(
        ValueError,
        match="dtype",
    ):
        _reduction(
            normalization,
            edge_weights=weights,
        )


@pytest.mark.parametrize(
    "reduction",
    (
        "",
        ATTENTION_HEAD_REDUCTION_WEIGHTED_MEAN,
        ATTENTION_HEAD_REDUCTION_MAX,
    ),
)
def test_head_reduction_rejects_unsupported_policy(
    reduction: str,
) -> None:
    normalization = _normalization(
        _score(_inputs())
    )

    with pytest.raises(ValueError):
        _reduction(
            normalization,
            head_reduction=reduction,
        )


def test_head_reduction_rejects_values_not_equal_to_mean() -> None:
    normalization = _normalization(
        _score(_inputs())
    )
    weights = (
        normalization
        .normalized_weights_by_head
        .mean(dim=1)
        .clone()
    )
    weights[0] += 0.05
    weights[1] -= 0.05

    with pytest.raises(
        ValueError,
        match="arithmetic mean",
    ):
        _reduction(
            normalization,
            edge_weights=weights,
        )


@pytest.mark.parametrize(
    "field",
    (
        "encoder_architecture_fingerprint",
        "schema_version",
    ),
)
def test_head_reduction_rejects_blank_required_string(
    field: str,
) -> None:
    output = _reduction(
        _normalization(
            _score(_inputs())
        )
    )

    with pytest.raises(
        ValueError,
        match="non-empty",
    ):
        replace(
            output,
            **{field: ""},
        )


def test_head_reduction_rejects_blank_parameter_fingerprint() -> None:
    with pytest.raises(
        ValueError,
        match="non-empty",
    ):
        _reduction(
            _normalization(
                _score(_inputs())
            ),
            parameter_fingerprint="",
        )


# =============================================================================
# Cross-stage provenance and scientific boundaries
# =============================================================================


def test_complete_stage_chain_preserves_exact_source_object() -> None:
    inputs = _inputs()
    score = _score(inputs)
    normalization = _normalization(score)
    reduction = _reduction(
        normalization
    )

    assert score.source_inputs is inputs
    assert normalization.source_inputs is inputs
    assert reduction.source_inputs is inputs


def test_complete_stage_chain_preserves_exact_relation_order() -> None:
    inputs = _inputs()
    reduction = _reduction(
        _normalization(
            _score(inputs)
        )
    )

    assert (
        reduction
        .source_score_output
        .relation_names
        == inputs.relation_names
    )
    assert (
        reduction
        .source_score_output
        .stable_relation_ids
        == inputs.stable_relation_ids
    )


def test_stage_chain_contains_no_relation_gate_field() -> None:
    reduction = _reduction(
        _normalization(
            _score(_inputs())
        )
    )

    assert not hasattr(
        reduction.source_score_output,
        "relation_gate",
    )
    assert not hasattr(
        reduction.source_normalization_output,
        "relation_gate",
    )
    assert not hasattr(
        reduction,
        "relation_gate",
    )


def test_stage_chain_does_not_label_attention_as_causal_importance() -> None:
    reduction = _reduction(
        _normalization(
            _score(_inputs())
        )
    )

    for value in (
        reduction.source_score_output,
        reduction.source_normalization_output,
        reduction,
    ):
        assert not hasattr(
            value,
            "causal_importance",
        )


def test_uniform_attention_is_not_disabled_attention_identity() -> None:
    inputs = _inputs(
        with_hazard_query=False,
    )
    reduction = _reduction(
        _normalization(
            _score(
                inputs,
                mode=ATTENTION_MODE_UNIFORM,
            )
        )
    )

    repeated_group = (
        reduction.group_counts[
            reduction.group_ids
        ]
        > 1
    )

    assert bool(repeated_group.any().item())
    assert bool(
        (
            reduction.edge_weights[
                repeated_group
            ]
            < 1
        )
        .all()
        .item()
    )


def test_control_relation_edges_are_normalized_without_special_math() -> None:
    inputs = _inputs(
        with_hazard_query=False,
    )
    reduction = _reduction(
        _normalization(
            _score(
                inputs,
                mode=ATTENTION_MODE_UNIFORM,
            )
        )
    )
    control_edges = (
        inputs.control_edge_mask
    )

    assert bool(control_edges.any().item())
    assert torch.isfinite(
        reduction.edge_weights[
            control_edges
        ]
    ).all()
    assert bool(
        (
            reduction.edge_weights[
                control_edges
            ]
            >= 0
        )
        .all()
        .item()
    )


# =============================================================================
# Optional CUDA contract
# =============================================================================


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is not available.",
)
def test_complete_schema_chain_on_cuda() -> None:
    device = torch.device("cuda")
    inputs = _inputs(
        device=device,
    )
    score = _score(inputs)
    normalization = _normalization(score)
    reduction = _reduction(
        normalization
    )

    assert score.device.type == "cuda"
    assert normalization.device.type == "cuda"
    assert reduction.device.type == "cuda"
    assert reduction.group_ids.device.type == (
        "cuda"
    )
    assert reduction.group_counts.device.type == (
        "cuda"
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is not available.",
)
def test_score_output_rejects_cpu_tensor_for_cuda_inputs() -> None:
    inputs = _inputs(
        device="cuda",
    )
    raw = torch.zeros(
        (inputs.num_edges, 1),
        dtype=inputs.dtype,
        device="cpu",
    )

    with pytest.raises(
        ValueError,
        match="device",
    ):
        _score(
            inputs,
            raw_scores=raw,
        )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is not available.",
)
def test_normalization_rejects_cpu_group_metadata_for_cuda_inputs() -> None:
    score = _score(
        _inputs(device="cuda")
    )
    group_ids = (
        score
        .source_inputs
        .attention_group_id
        .cpu()
    )
    weights = grouped_softmax(
        score.raw_scores_by_head.cpu(),
        group_ids,
        num_segments=(
            score
            .source_inputs
            .attention_num_groups
        ),
    )

    with pytest.raises(
        ValueError,
        match="device",
    ):
        _normalization(
            score,
            weights=weights.to(
                device=score.device,
            ),
            group_ids=group_ids,
            group_counts=segment_counts(
                group_ids,
                num_segments=(
                    score
                    .source_inputs
                    .attention_num_groups
                ),
            ),
        )
