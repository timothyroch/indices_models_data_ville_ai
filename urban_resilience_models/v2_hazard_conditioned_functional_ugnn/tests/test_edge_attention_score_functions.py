"""
Focused tests for edge-attention score functions.

Target repository path:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            tests/
                test_edge_attention_score_functions.py

Implementation under test:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            functional_message_passing/
                edge_attention/
                    score_functions.py

The suite tests the score stage independently from grouped normalization,
head reduction, relation gates, relation transforms, message construction,
aggregation, and explanation exporters.

Scientific contracts frozen here
--------------------------------
- Uniform attention emits exact zero logits and owns no parameters.
- Learned attention uses exact-relation additive compatibility:
      v_a^T tanh(
          W_s,a h_source
          + W_t,a h_target
          + optional W_q,a q_target
          + E_relation,a
      )
- Hazard-blind attention does not read the hazard query.
- Hazard-conditioned attention can change pairwise logit differences among
  competing edges, rather than merely adding a group-constant bias.
- Exact dense relation indices select learned embeddings; sparse stable
  ontology IDs remain metadata only.
- Edge attributes, semantic edge weights, relation gates, normalization,
  transformed messages, and aggregation statistics do not enter this stage.
- Edge order is equivariant.
- Identical edge scorer inputs produce identical logits.
- Empty edge sets are valid.
- Parameters, runtime tensors, outputs, devices, dtypes, relation order, and
  fingerprints are validated explicitly.
- Multihead tensor support is tested without claiming head specialization.

Controlled upstream doubles are patched into the existing
``functional_message_passing.schemas`` module so that failures remain
localized to score-function behavior rather than graph loading, fusion, hazard
encoding, or registry construction.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Iterator

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.config import (
    FunctionalMessagePassingConfig,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.constants import (
    ATTENTION_MODE_HAZARD_BLIND,
    ATTENTION_MODE_HAZARD_CONDITIONED,
    ATTENTION_MODE_MULTIHEAD_HAZARD_CONDITIONED,
    ATTENTION_MODE_SEMANTIC_WEIGHT,
    ATTENTION_MODE_UNIFORM,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.functional_message_passing import (
    schemas as fmp_schemas,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.functional_message_passing.edge_attention.schemas import (
    EDGE_ATTENTION_INPUT_EXACT_RELATION_EMBEDDING,
    EDGE_ATTENTION_INPUT_SOURCE_NODE_STATE,
    EDGE_ATTENTION_INPUT_TARGET_HAZARD_QUERY,
    EDGE_ATTENTION_INPUT_TARGET_NODE_STATE,
    EDGE_ATTENTION_SCORE_FUNCTION_ADDITIVE,
    EDGE_ATTENTION_SCORE_FUNCTION_UNIFORM,
    EdgeAttentionScoreOutput,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.functional_message_passing.edge_attention.score_functions import (
    DEFAULT_EDGE_ATTENTION_HIDDEN_DIM,
    EDGE_ATTENTION_SCORE_FUNCTIONS_SCHEMA_VERSION,
    EDGE_ATTENTION_SCORE_MODES,
    LEARNED_EDGE_ATTENTION_MODES,
    AdditiveAttentionScoreFunction,
    AdditiveEdgeAttentionScoreFunction,
    EdgeAttentionScoreFunction,
    UniformAttentionScoreFunction,
    UniformEdgeAttentionScoreFunction,
    build_attention_score_function,
    build_edge_attention_score_function,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.functional_message_passing.schemas import (
    FunctionalMessagePassingInputs,
)


NODES = 5
EDGES = 7
GRAPHS = 2
RELATIONS = 3
HIDDEN_DIM = 4
QUERY_DIM = 3
ATTENTION_HIDDEN_DIM = 5
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


def _base_edge_index(
    *,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    # Same-group edges 0 and 1:
    #   target=1, relation=0, different sources.
    #
    # Same-group edges 4 and 5:
    #   target=4, relation=2, different sources.
    return torch.tensor(
        [
            [0, 2, 1, 0, 3, 4, 4],
            [1, 1, 2, 2, 4, 4, 3],
        ],
        dtype=torch.long,
        device=device,
    )


def _base_edge_relations(
    *,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    return torch.tensor(
        [0, 0, 1, 2, 2, 2, 1],
        dtype=torch.long,
        device=device,
    )


def _base_edge_batch_index(
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
    hidden_dim: int = HIDDEN_DIM,
    offset: float = 0.0,
) -> torch.Tensor:
    value = (
        torch.arange(
            NODES * hidden_dim,
            dtype=dtype,
            device=device,
        )
        .reshape(NODES, hidden_dim)
        / 10.0
        + offset
    )
    value = value.detach().clone()

    if requires_grad:
        value.requires_grad_(True)

    return value


def _query_tensor(
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
    requires_grad: bool = False,
    query_dim: int = QUERY_DIM,
    offset: float = 0.0,
) -> torch.Tensor:
    value = (
        torch.arange(
            NODES * query_dim,
            dtype=dtype,
            device=device,
        )
        .reshape(NODES, query_dim)
        / 7.0
        + offset
    )
    value = value.detach().clone()

    if requires_grad:
        value.requires_grad_(True)

    return value


def _registry(
    *,
    names: tuple[str, ...] = RELATION_NAMES,
    stable_ids: tuple[int, ...] = (
        STABLE_RELATION_IDS
    ),
    fingerprint: str = (
        "compiled-relation-registry"
    ),
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
    edge_index: torch.Tensor | None = None,
    edge_relations: torch.Tensor | None = None,
    edge_batch_index: torch.Tensor | None = None,
    edge_attributes: torch.Tensor | None = None,
    semantic_edge_weight: torch.Tensor | None = None,
    permutation: torch.Tensor | None = None,
    empty_edges: bool = False,
) -> FakeUrbanGraphBatch:
    node_batch = _node_batch_index(
        device=device,
    )

    if empty_edges:
        resolved_edge_index = torch.empty(
            (2, 0),
            dtype=torch.long,
            device=device,
        )
        resolved_relations = torch.empty(
            (0,),
            dtype=torch.long,
            device=device,
        )
        resolved_edge_batch = torch.empty(
            (0,),
            dtype=torch.long,
            device=device,
        )
    else:
        resolved_edge_index = (
            _base_edge_index(device=device)
            if edge_index is None
            else edge_index
        )
        resolved_relations = (
            _base_edge_relations(device=device)
            if edge_relations is None
            else edge_relations
        )
        resolved_edge_batch = (
            _base_edge_batch_index(
                device=device
            )
            if edge_batch_index is None
            else edge_batch_index
        )

        if permutation is not None:
            resolved_edge_index = (
                resolved_edge_index[
                    :,
                    permutation,
                ]
            )
            resolved_relations = (
                resolved_relations[
                    permutation
                ]
            )
            resolved_edge_batch = (
                resolved_edge_batch[
                    permutation
                ]
            )

            if edge_attributes is not None:
                edge_attributes = (
                    edge_attributes[
                        permutation
                    ]
                )

            if semantic_edge_weight is not None:
                semantic_edge_weight = (
                    semantic_edge_weight[
                        permutation
                    ]
                )

    return FakeUrbanGraphBatch(
        external_node_ids=_node_ids(),
        node_batch_index=node_batch,
        edge_index=resolved_edge_index,
        edge_relation_type=(
            resolved_relations
        ),
        edge_attributes=edge_attributes,
        semantic_edge_weight=(
            semantic_edge_weight
        ),
        edge_batch_index=(
            resolved_edge_batch
        ),
    )


def _inputs(
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
    state: torch.Tensor | None = None,
    query: torch.Tensor | None = None,
    with_hazard_query: bool = True,
    registry: FakeCompiledRelationRegistry | None = None,
    edge_index: torch.Tensor | None = None,
    edge_relations: torch.Tensor | None = None,
    edge_batch_index: torch.Tensor | None = None,
    edge_attributes: torch.Tensor | None = None,
    semantic_edge_weight: torch.Tensor | None = None,
    permutation: torch.Tensor | None = None,
    empty_edges: bool = False,
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

    graph = _graph(
        dtype=dtype,
        device=device,
        edge_index=edge_index,
        edge_relations=edge_relations,
        edge_batch_index=edge_batch_index,
        edge_attributes=edge_attributes,
        semantic_edge_weight=(
            semantic_edge_weight
        ),
        permutation=permutation,
        empty_edges=empty_edges,
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
        source_graph=graph,
        node_state=node_state,
        compiled_relation_registry=(
            _registry()
            if registry is None
            else registry
        ),
        hazard_query=hazard_query,
        source_fingerprint=(
            "edge-attention-score-test-input"
        ),
    )


def _config(
    *,
    mode: str,
    heads: int = 1,
) -> FunctionalMessagePassingConfig:
    return FunctionalMessagePassingConfig(
        enabled=True,
        attention_enabled=True,
        attention_mode=mode,
        attention_heads=heads,
    )


def _uniform_module(
    *,
    relation_names: tuple[str, ...] = RELATION_NAMES,
    stable_relation_ids: tuple[int, ...] = (
        STABLE_RELATION_IDS
    ),
) -> UniformEdgeAttentionScoreFunction:
    return UniformEdgeAttentionScoreFunction(
        relation_names=relation_names,
        stable_relation_ids=(
            stable_relation_ids
        ),
    )


def _additive_module(
    *,
    mode: str = (
        ATTENTION_MODE_HAZARD_CONDITIONED
    ),
    node_state_dim: int = HIDDEN_DIM,
    hazard_query_dim: int | None = QUERY_DIM,
    hidden_dim: int = (
        ATTENTION_HIDDEN_DIM
    ),
    num_heads: int = 1,
    relation_names: tuple[str, ...] = RELATION_NAMES,
    stable_relation_ids: tuple[int, ...] = (
        STABLE_RELATION_IDS
    ),
) -> AdditiveEdgeAttentionScoreFunction:
    if mode == ATTENTION_MODE_HAZARD_BLIND:
        hazard_query_dim = None

    return AdditiveEdgeAttentionScoreFunction(
        node_state_dim=node_state_dim,
        hazard_query_dim=hazard_query_dim,
        relation_names=relation_names,
        stable_relation_ids=(
            stable_relation_ids
        ),
        hidden_dim=hidden_dim,
        mode=mode,
        num_heads=num_heads,
    )


def _all_parameters(
    module: nn.Module,
) -> Iterator[torch.Tensor]:
    yield from module.parameters()


def _fill_parameters(
    module: nn.Module,
    value: float,
) -> None:
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.fill_(value)


def _manual_scores(
    module: AdditiveEdgeAttentionScoreFunction,
    inputs: FunctionalMessagePassingInputs,
) -> torch.Tensor:
    node_state = (
        inputs.node_state.fused_state
    )
    source_state = node_state[
        inputs.source_index
    ]
    target_state = node_state[
        inputs.target_index
    ]

    source_term = F.linear(
        source_state,
        module.source_projection.weight,
    ).reshape(
        inputs.num_edges,
        module.num_heads,
        module.hidden_dim,
    )
    target_term = F.linear(
        target_state,
        module.target_projection.weight,
    ).reshape(
        inputs.num_edges,
        module.num_heads,
        module.hidden_dim,
    )
    relation_term = (
        module.relation_embeddings[
            inputs.edge_relation_index
        ]
    )

    preactivation = (
        source_term
        + target_term
        + relation_term
    )

    if module.uses_hazard_query:
        assert (
            module.hazard_projection
            is not None
        )
        query = inputs.node_hazard_query
        assert query is not None
        hazard_term = F.linear(
            query[inputs.target_index],
            module.hazard_projection.weight,
        ).reshape(
            inputs.num_edges,
            module.num_heads,
            module.hidden_dim,
        )
        preactivation = (
            preactivation
            + hazard_term
        )

    state = torch.tanh(
        preactivation
    )

    return torch.einsum(
        "eah,ah->ea",
        state,
        module.score_vectors,
    )


# =============================================================================
# Public identity and aliases
# =============================================================================


def test_public_schema_version_and_defaults_are_valid() -> None:
    assert isinstance(
        EDGE_ATTENTION_SCORE_FUNCTIONS_SCHEMA_VERSION,
        str,
    )
    assert (
        EDGE_ATTENTION_SCORE_FUNCTIONS_SCHEMA_VERSION
        .strip()
    )
    assert (
        DEFAULT_EDGE_ATTENTION_HIDDEN_DIM
        > 0
    )


def test_score_mode_vocabularies_are_exact() -> None:
    assert LEARNED_EDGE_ATTENTION_MODES == (
        ATTENTION_MODE_HAZARD_BLIND,
        ATTENTION_MODE_HAZARD_CONDITIONED,
        ATTENTION_MODE_MULTIHEAD_HAZARD_CONDITIONED,
    )
    assert EDGE_ATTENTION_SCORE_MODES == (
        ATTENTION_MODE_UNIFORM,
        ATTENTION_MODE_HAZARD_BLIND,
        ATTENTION_MODE_HAZARD_CONDITIONED,
        ATTENTION_MODE_MULTIHEAD_HAZARD_CONDITIONED,
    )


def test_compact_class_aliases_are_exact() -> None:
    assert UniformAttentionScoreFunction is (
        UniformEdgeAttentionScoreFunction
    )
    assert AdditiveAttentionScoreFunction is (
        AdditiveEdgeAttentionScoreFunction
    )


def test_compact_builder_alias_is_exact() -> None:
    assert build_attention_score_function is (
        build_edge_attention_score_function
    )


def test_type_alias_accepts_both_implementations() -> None:
    uniform: EdgeAttentionScoreFunction = (
        _uniform_module()
    )
    additive: EdgeAttentionScoreFunction = (
        _additive_module()
    )

    assert isinstance(
        uniform,
        UniformEdgeAttentionScoreFunction,
    )
    assert isinstance(
        additive,
        AdditiveEdgeAttentionScoreFunction,
    )


# =============================================================================
# Uniform score function
# =============================================================================


def test_uniform_constructor_identity() -> None:
    module = _uniform_module()

    assert module.mode == (
        ATTENTION_MODE_UNIFORM
    )
    assert module.num_heads == 1
    assert module.num_relations == RELATIONS
    assert module.relation_names == (
        RELATION_NAMES
    )
    assert module.stable_relation_ids == (
        STABLE_RELATION_IDS
    )
    assert module.input_feature_names == ()
    assert module.parameter_count == 0
    assert module.trainable_parameter_count == 0
    assert module.parameter_fingerprint() is None


def test_uniform_architecture_metadata() -> None:
    module = _uniform_module()
    architecture = (
        module.architecture_dict()
    )

    assert architecture["mode"] == (
        ATTENTION_MODE_UNIFORM
    )
    assert architecture["score_function"] == (
        EDGE_ATTENTION_SCORE_FUNCTION_UNIFORM
    )
    assert architecture["raw_score_identity"] == (
        "exact_zero_logits"
    )
    assert architecture["uses_node_state"] is False
    assert architecture["uses_hazard_query"] is False
    assert (
        architecture["uses_relation_embedding"]
        is False
    )
    assert architecture["uses_edge_attributes"] is False
    assert architecture["uses_relation_gate"] is False
    assert architecture["parameter_count"] == 0


def test_uniform_forward_returns_exact_zero_output() -> None:
    inputs = _inputs(
        with_hazard_query=False,
    )
    module = _uniform_module()
    output = module(inputs)

    assert isinstance(
        output,
        EdgeAttentionScoreOutput,
    )
    assert output.source_inputs is inputs
    assert output.attention_mode == (
        ATTENTION_MODE_UNIFORM
    )
    assert output.score_function == (
        EDGE_ATTENTION_SCORE_FUNCTION_UNIFORM
    )
    assert output.input_feature_names == ()
    assert output.raw_scores_by_head.shape == (
        EDGES,
        1,
    )
    assert torch.equal(
        output.raw_scores_by_head,
        torch.zeros_like(
            output.raw_scores_by_head
        ),
    )
    assert output.parameter_fingerprint is None
    assert (
        output.encoder_architecture_fingerprint
        == module.architecture_fingerprint()
    )


def test_uniform_score_tensor_matches_forward() -> None:
    inputs = _inputs(
        with_hazard_query=False,
    )
    module = _uniform_module()

    assert torch.equal(
        module.score_tensor(inputs),
        module(inputs).raw_scores_by_head,
    )


def test_uniform_does_not_require_or_read_hazard_query() -> None:
    no_query = _inputs(
        with_hazard_query=False,
    )
    query_a = _inputs(
        query=_query_tensor(offset=0.0),
    )
    query_b = _inputs(
        query=_query_tensor(offset=100.0),
    )
    module = _uniform_module()

    expected = module(no_query).raw_scores_by_head
    assert torch.equal(
        expected,
        module(query_a).raw_scores_by_head,
    )
    assert torch.equal(
        expected,
        module(query_b).raw_scores_by_head,
    )


def test_uniform_empty_edges() -> None:
    inputs = _inputs(
        empty_edges=True,
        with_hazard_query=False,
    )
    output = _uniform_module()(inputs)

    assert output.raw_scores_by_head.shape == (
        0,
        1,
    )
    assert torch.isfinite(
        output.raw_scores_by_head
    ).all()


def test_uniform_float64_output() -> None:
    inputs = _inputs(
        dtype=torch.float64,
        with_hazard_query=False,
    )
    module = (
        _uniform_module()
        .to(dtype=torch.float64)
    )
    output = module(inputs)

    assert output.dtype == torch.float64


def test_uniform_from_config() -> None:
    inputs = _inputs(
        with_hazard_query=False,
    )
    config = _config(
        mode=ATTENTION_MODE_UNIFORM,
    )

    module = (
        UniformEdgeAttentionScoreFunction
        .from_config(
            config=config,
            source_inputs=inputs,
        )
    )

    assert isinstance(
        module,
        UniformEdgeAttentionScoreFunction,
    )
    assert module.relation_names == (
        inputs.relation_names
    )


def test_uniform_extra_repr_is_informative() -> None:
    text = repr(
        _uniform_module()
    )

    assert "num_relations=3" in text
    assert "uniform" in text
    assert "parameter_free=True" in text


# =============================================================================
# Additive constructor and architecture identity
# =============================================================================


@pytest.mark.parametrize(
    (
        "mode",
        "heads",
        "uses_hazard",
    ),
    (
        (
            ATTENTION_MODE_HAZARD_BLIND,
            1,
            False,
        ),
        (
            ATTENTION_MODE_HAZARD_CONDITIONED,
            1,
            True,
        ),
        (
            ATTENTION_MODE_MULTIHEAD_HAZARD_CONDITIONED,
            MULTIHEAD_COUNT,
            True,
        ),
    ),
)
def test_additive_valid_mode_identity(
    mode: str,
    heads: int,
    uses_hazard: bool,
) -> None:
    module = _additive_module(
        mode=mode,
        num_heads=heads,
    )

    assert module.mode == mode
    assert module.num_heads == heads
    assert module.num_relations == RELATIONS
    assert module.node_state_dim == HIDDEN_DIM
    assert module.hidden_dim == (
        ATTENTION_HIDDEN_DIM
    )
    assert (
        module.uses_hazard_query
        is uses_hazard
    )
    assert module.projection_width == (
        heads * ATTENTION_HIDDEN_DIM
    )
    assert module.hazard_query_dim == (
        QUERY_DIM if uses_hazard else None
    )
    assert (
        module.hazard_projection is not None
    ) is uses_hazard


@pytest.mark.parametrize(
    (
        "mode",
        "expected",
    ),
    (
        (
            ATTENTION_MODE_HAZARD_BLIND,
            (
                EDGE_ATTENTION_INPUT_SOURCE_NODE_STATE,
                EDGE_ATTENTION_INPUT_TARGET_NODE_STATE,
                EDGE_ATTENTION_INPUT_EXACT_RELATION_EMBEDDING,
            ),
        ),
        (
            ATTENTION_MODE_HAZARD_CONDITIONED,
            (
                EDGE_ATTENTION_INPUT_SOURCE_NODE_STATE,
                EDGE_ATTENTION_INPUT_TARGET_NODE_STATE,
                EDGE_ATTENTION_INPUT_TARGET_HAZARD_QUERY,
                EDGE_ATTENTION_INPUT_EXACT_RELATION_EMBEDDING,
            ),
        ),
    ),
)
def test_additive_input_feature_identity(
    mode: str,
    expected: tuple[str, ...],
) -> None:
    module = _additive_module(
        mode=mode,
    )

    assert module.input_feature_names == (
        expected
    )


@pytest.mark.parametrize(
    (
        "mode",
        "expected_components",
    ),
    (
        (
            ATTENTION_MODE_HAZARD_BLIND,
            3,
        ),
        (
            ATTENTION_MODE_HAZARD_CONDITIONED,
            4,
        ),
    ),
)
def test_additive_context_initialization_identity(
    mode: str,
    expected_components: int,
) -> None:
    module = _additive_module(
        mode=mode,
    )

    assert module.context_component_count == (
        expected_components
    )
    assert math.isclose(
        module.context_initialization_scale,
        1.0
        / math.sqrt(
            float(expected_components)
        ),
    )


def test_additive_parameter_shapes() -> None:
    module = _additive_module()

    assert module.source_projection.weight.shape == (
        MULTIHEAD_COUNT * 0 + ATTENTION_HIDDEN_DIM,
        HIDDEN_DIM,
    )
    assert module.target_projection.weight.shape == (
        ATTENTION_HIDDEN_DIM,
        HIDDEN_DIM,
    )
    assert module.hazard_projection is not None
    assert module.hazard_projection.weight.shape == (
        ATTENTION_HIDDEN_DIM,
        QUERY_DIM,
    )
    assert module.relation_embeddings.shape == (
        RELATIONS,
        1,
        ATTENTION_HIDDEN_DIM,
    )
    assert module.score_vectors.shape == (
        1,
        ATTENTION_HIDDEN_DIM,
    )


def test_multihead_parameter_shapes() -> None:
    module = _additive_module(
        mode=(
            ATTENTION_MODE_MULTIHEAD_HAZARD_CONDITIONED
        ),
        num_heads=MULTIHEAD_COUNT,
    )
    width = (
        MULTIHEAD_COUNT
        * ATTENTION_HIDDEN_DIM
    )

    assert module.source_projection.weight.shape == (
        width,
        HIDDEN_DIM,
    )
    assert module.target_projection.weight.shape == (
        width,
        HIDDEN_DIM,
    )
    assert module.hazard_projection is not None
    assert module.hazard_projection.weight.shape == (
        width,
        QUERY_DIM,
    )
    assert module.relation_embeddings.shape == (
        RELATIONS,
        MULTIHEAD_COUNT,
        ATTENTION_HIDDEN_DIM,
    )
    assert module.score_vectors.shape == (
        MULTIHEAD_COUNT,
        ATTENTION_HIDDEN_DIM,
    )


def test_additive_projections_have_no_bias() -> None:
    module = _additive_module()

    assert module.source_projection.bias is None
    assert module.target_projection.bias is None
    assert module.hazard_projection is not None
    assert module.hazard_projection.bias is None


@pytest.mark.parametrize(
    (
        "mode",
        "heads",
        "expected_count",
    ),
    (
        (
            ATTENTION_MODE_HAZARD_BLIND,
            1,
            (
                HIDDEN_DIM
                * ATTENTION_HIDDEN_DIM
                * 2
                + RELATIONS
                * ATTENTION_HIDDEN_DIM
                + ATTENTION_HIDDEN_DIM
            ),
        ),
        (
            ATTENTION_MODE_HAZARD_CONDITIONED,
            1,
            (
                HIDDEN_DIM
                * ATTENTION_HIDDEN_DIM
                * 2
                + QUERY_DIM
                * ATTENTION_HIDDEN_DIM
                + RELATIONS
                * ATTENTION_HIDDEN_DIM
                + ATTENTION_HIDDEN_DIM
            ),
        ),
        (
            ATTENTION_MODE_MULTIHEAD_HAZARD_CONDITIONED,
            MULTIHEAD_COUNT,
            (
                HIDDEN_DIM
                * MULTIHEAD_COUNT
                * ATTENTION_HIDDEN_DIM
                * 2
                + QUERY_DIM
                * MULTIHEAD_COUNT
                * ATTENTION_HIDDEN_DIM
                + RELATIONS
                * MULTIHEAD_COUNT
                * ATTENTION_HIDDEN_DIM
                + MULTIHEAD_COUNT
                * ATTENTION_HIDDEN_DIM
            ),
        ),
    ),
)
def test_additive_parameter_count_formula(
    mode: str,
    heads: int,
    expected_count: int,
) -> None:
    module = _additive_module(
        mode=mode,
        num_heads=heads,
    )

    assert module.parameter_count == (
        expected_count
    )
    assert (
        module.trainable_parameter_count
        == expected_count
    )


def test_trainable_parameter_count_respects_requires_grad() -> None:
    module = _additive_module()
    original = (
        module.trainable_parameter_count
    )
    module.score_vectors.requires_grad_(
        False
    )

    assert module.parameter_count == (
        original
    )
    assert (
        module.trainable_parameter_count
        == original
        - module.score_vectors.numel()
    )


def test_additive_architecture_metadata() -> None:
    module = _additive_module()
    architecture = (
        module.architecture_dict()
    )

    assert architecture["mode"] == (
        ATTENTION_MODE_HAZARD_CONDITIONED
    )
    assert architecture["score_function"] == (
        EDGE_ATTENTION_SCORE_FUNCTION_ADDITIVE
    )
    assert architecture["node_state_dim"] == (
        HIDDEN_DIM
    )
    assert architecture["hazard_query_dim"] == (
        QUERY_DIM
    )
    assert architecture["hidden_dim"] == (
        ATTENTION_HIDDEN_DIM
    )
    assert architecture["num_heads"] == 1
    assert architecture["uses_node_state"] is True
    assert architecture["uses_hazard_query"] is True
    assert (
        architecture["uses_relation_embedding"]
        is True
    )
    assert architecture["uses_edge_attributes"] is False
    assert architecture["uses_relation_gate"] is False
    assert architecture["projection_bias"] is False
    assert (
        architecture["standalone_relation_bias"]
        is False
    )
    assert (
        architecture["standalone_hazard_bias"]
        is False
    )
    assert architecture["compatibility_activation"] == (
        "tanh"
    )


def test_additive_extra_repr_is_informative() -> None:
    text = repr(
        _additive_module()
    )

    assert "node_state_dim=4" in text
    assert "hazard_query_dim=3" in text
    assert "hidden_dim=5" in text
    assert "num_relations=3" in text
    assert "num_heads=1" in text
    assert "uses_edge_attributes=False" in text


def test_additive_initial_parameters_are_finite() -> None:
    module = _additive_module()

    module.assert_finite_parameters()

    for parameter in module.parameters():
        assert torch.isfinite(
            parameter
        ).all()


# =============================================================================
# Additive tensor stages and exact formula
# =============================================================================


@pytest.mark.parametrize(
    (
        "mode",
        "heads",
    ),
    (
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
def test_additive_intermediate_shapes(
    mode: str,
    heads: int,
) -> None:
    inputs = _inputs()
    module = _additive_module(
        mode=mode,
        num_heads=heads,
    )

    assert (
        module.projected_source_state(
            inputs
        ).shape
        == (
            EDGES,
            heads,
            ATTENTION_HIDDEN_DIM,
        )
    )
    assert (
        module.projected_target_state(
            inputs
        ).shape
        == (
            EDGES,
            heads,
            ATTENTION_HIDDEN_DIM,
        )
    )
    assert (
        module.edge_relation_embeddings(
            inputs
        ).shape
        == (
            EDGES,
            heads,
            ATTENTION_HIDDEN_DIM,
        )
    )
    assert (
        module.compatibility_preactivations(
            inputs
        ).shape
        == (
            EDGES,
            heads,
            ATTENTION_HIDDEN_DIM,
        )
    )
    assert (
        module.compatibility_state(
            inputs
        ).shape
        == (
            EDGES,
            heads,
            ATTENTION_HIDDEN_DIM,
        )
    )
    assert module.score_tensor(inputs).shape == (
        EDGES,
        heads,
    )


def test_hazard_blind_projected_query_is_none() -> None:
    module = _additive_module(
        mode=ATTENTION_MODE_HAZARD_BLIND,
    )

    assert (
        module.projected_target_hazard_query(
            _inputs()
        )
        is None
    )


def test_conditioned_projected_query_shape() -> None:
    module = _additive_module()

    projected = (
        module.projected_target_hazard_query(
            _inputs()
        )
    )

    assert projected is not None
    assert projected.shape == (
        EDGES,
        1,
        ATTENTION_HIDDEN_DIM,
    )


def test_projected_source_state_matches_linear_projection() -> None:
    inputs = _inputs()
    module = _additive_module()
    observed = (
        module.projected_source_state(
            inputs
        )
    )
    expected = F.linear(
        inputs.node_state.fused_state[
            inputs.source_index
        ],
        module.source_projection.weight,
    ).reshape(
        EDGES,
        1,
        ATTENTION_HIDDEN_DIM,
    )

    assert torch.allclose(
        observed,
        expected,
    )


def test_projected_target_state_matches_linear_projection() -> None:
    inputs = _inputs()
    module = _additive_module()
    observed = (
        module.projected_target_state(
            inputs
        )
    )
    expected = F.linear(
        inputs.node_state.fused_state[
            inputs.target_index
        ],
        module.target_projection.weight,
    ).reshape(
        EDGES,
        1,
        ATTENTION_HIDDEN_DIM,
    )

    assert torch.allclose(
        observed,
        expected,
    )


def test_projected_hazard_matches_target_query_projection() -> None:
    inputs = _inputs()
    module = _additive_module()
    observed = (
        module.projected_target_hazard_query(
            inputs
        )
    )
    assert observed is not None
    assert module.hazard_projection is not None

    expected = F.linear(
        inputs.node_hazard_query[
            inputs.target_index
        ],
        module.hazard_projection.weight,
    ).reshape(
        EDGES,
        1,
        ATTENTION_HIDDEN_DIM,
    )

    assert torch.allclose(
        observed,
        expected,
    )


def test_relation_embedding_lookup_uses_dense_relation_indices() -> None:
    inputs = _inputs()
    module = _additive_module()

    with torch.no_grad():
        module.relation_embeddings.copy_(
            torch.tensor(
                [
                    [[1.0] * ATTENTION_HIDDEN_DIM],
                    [[2.0] * ATTENTION_HIDDEN_DIM],
                    [[3.0] * ATTENTION_HIDDEN_DIM],
                ],
                dtype=inputs.dtype,
            )
        )

    observed = (
        module.edge_relation_embeddings(
            inputs
        )
    )
    expected = (
        module.relation_embeddings[
            inputs.edge_relation_index
        ]
    )

    assert torch.equal(
        observed,
        expected,
    )
    assert STABLE_RELATION_IDS != (
        0,
        1,
        2,
    )


def test_compatibility_preactivation_is_exact_sum() -> None:
    inputs = _inputs()
    module = _additive_module()

    observed = (
        module.compatibility_preactivations(
            inputs
        )
    )
    expected = (
        module.projected_source_state(
            inputs
        )
        + module.projected_target_state(
            inputs
        )
        + module.edge_relation_embeddings(
            inputs
        )
    )
    hazard = (
        module.projected_target_hazard_query(
            inputs
        )
    )
    assert hazard is not None
    expected = expected + hazard

    assert torch.allclose(
        observed,
        expected,
    )


def test_hazard_blind_preactivation_has_no_hazard_term() -> None:
    inputs = _inputs()
    module = _additive_module(
        mode=ATTENTION_MODE_HAZARD_BLIND,
    )

    observed = (
        module.compatibility_preactivations(
            inputs
        )
    )
    expected = (
        module.projected_source_state(
            inputs
        )
        + module.projected_target_state(
            inputs
        )
        + module.edge_relation_embeddings(
            inputs
        )
    )

    assert torch.allclose(
        observed,
        expected,
    )


def test_compatibility_state_is_tanh() -> None:
    inputs = _inputs()
    module = _additive_module()

    assert torch.allclose(
        module.compatibility_state(
            inputs
        ),
        torch.tanh(
            module
            .compatibility_preactivations(
                inputs
            )
        ),
    )


@pytest.mark.parametrize(
    (
        "mode",
        "heads",
    ),
    (
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
def test_score_tensor_matches_manual_formula(
    mode: str,
    heads: int,
) -> None:
    inputs = _inputs()
    module = _additive_module(
        mode=mode,
        num_heads=heads,
    )

    assert torch.allclose(
        module.score_tensor(inputs),
        _manual_scores(
            module,
            inputs,
        ),
    )


def test_multihead_scores_are_head_separated() -> None:
    inputs = _inputs()
    module = _additive_module(
        mode=(
            ATTENTION_MODE_MULTIHEAD_HAZARD_CONDITIONED
        ),
        num_heads=2,
        hidden_dim=1,
    )

    with torch.no_grad():
        module.source_projection.weight.zero_()
        module.target_projection.weight.zero_()
        assert module.hazard_projection is not None
        module.hazard_projection.weight.zero_()
        module.relation_embeddings.zero_()
        module.score_vectors.copy_(
            torch.tensor(
                [[1.0], [2.0]],
                dtype=inputs.dtype,
            )
        )
        module.source_projection.weight[
            0,
            0,
        ] = 1.0
        module.source_projection.weight[
            1,
            1,
        ] = 1.0

    scores = module.score_tensor(inputs)

    expected_head_0 = torch.tanh(
        inputs.node_state.fused_state[
            inputs.source_index,
            0,
        ]
    )
    expected_head_1 = (
        2.0
        * torch.tanh(
            inputs.node_state.fused_state[
                inputs.source_index,
                1,
            ]
        )
    )

    assert torch.allclose(
        scores[:, 0],
        expected_head_0,
    )
    assert torch.allclose(
        scores[:, 1],
        expected_head_1,
    )


# =============================================================================
# Forward output contract
# =============================================================================


@pytest.mark.parametrize(
    (
        "mode",
        "heads",
    ),
    (
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
def test_additive_forward_output_contract(
    mode: str,
    heads: int,
) -> None:
    inputs = _inputs()
    module = _additive_module(
        mode=mode,
        num_heads=heads,
    )
    output = module(inputs)

    assert isinstance(
        output,
        EdgeAttentionScoreOutput,
    )
    assert output.source_inputs is inputs
    assert output.attention_mode == mode
    assert output.score_function == (
        EDGE_ATTENTION_SCORE_FUNCTION_ADDITIVE
    )
    assert output.input_feature_names == (
        module.input_feature_names
    )
    assert output.raw_scores_by_head.shape == (
        EDGES,
        heads,
    )
    assert torch.allclose(
        output.raw_scores_by_head,
        module.score_tensor(inputs),
    )
    assert output.relation_names == (
        module.relation_names
    )
    assert output.stable_relation_ids == (
        module.stable_relation_ids
    )
    assert (
        output.encoder_architecture_fingerprint
        == module.architecture_fingerprint()
    )
    assert output.parameter_fingerprint == (
        module.parameter_fingerprint()
    )


def test_forward_records_runtime_registry_fingerprint() -> None:
    inputs = _inputs(
        registry=_registry(
            fingerprint="runtime-registry",
        )
    )
    output = _additive_module()(inputs)

    assert (
        output
        .compiled_relation_registry_fingerprint
        == "runtime-registry"
    )


def test_additive_empty_edges() -> None:
    inputs = _inputs(
        empty_edges=True,
    )
    module = _additive_module()
    output = module(inputs)

    assert output.raw_scores_by_head.shape == (
        0,
        1,
    )
    assert (
        module.compatibility_preactivations(
            inputs
        ).shape
        == (
            0,
            1,
            ATTENTION_HIDDEN_DIM,
        )
    )
    assert torch.isfinite(
        output.raw_scores_by_head
    ).all()


def test_additive_float64_end_to_end() -> None:
    inputs = _inputs(
        dtype=torch.float64,
    )
    module = (
        _additive_module()
        .to(dtype=torch.float64)
    )
    output = module(inputs)

    assert output.dtype == torch.float64
    assert all(
        parameter.dtype == torch.float64
        for parameter in module.parameters()
    )


# =============================================================================
# Scientific behavior and metamorphic properties
# =============================================================================


def test_hazard_blind_scores_are_exactly_query_invariant() -> None:
    inputs_a = _inputs(
        query=_query_tensor(offset=0.0),
    )
    inputs_b = _inputs(
        query=_query_tensor(offset=100.0),
    )
    module = _additive_module(
        mode=ATTENTION_MODE_HAZARD_BLIND,
    )

    assert torch.equal(
        module.score_tensor(inputs_a),
        module.score_tensor(inputs_b),
    )


def test_hazard_conditioned_scores_can_change_with_query() -> None:
    inputs_a = _inputs(
        query=torch.zeros(
            (NODES, 1),
        ),
    )
    inputs_b = _inputs(
        query=torch.ones(
            (NODES, 1),
        ),
    )
    module = _additive_module(
        mode=(
            ATTENTION_MODE_HAZARD_CONDITIONED
        ),
        hazard_query_dim=1,
        hidden_dim=1,
    )

    with torch.no_grad():
        module.source_projection.weight.fill_(
            0.5
        )
        module.target_projection.weight.zero_()
        assert module.hazard_projection is not None
        module.hazard_projection.weight.fill_(
            1.0
        )
        module.relation_embeddings.zero_()
        module.score_vectors.fill_(1.0)

    scores_a = module.score_tensor(
        inputs_a
    )
    scores_b = module.score_tensor(
        inputs_b
    )

    assert not torch.allclose(
        scores_a,
        scores_b,
    )


def test_hazard_can_change_same_group_pairwise_logit_difference() -> None:
    state = torch.zeros(
        (NODES, 1),
    )
    state[0, 0] = -0.5
    state[2, 0] = 0.5

    inputs_a = _inputs(
        state=state,
        query=torch.zeros(
            (NODES, 1),
        ),
    )
    inputs_b = _inputs(
        state=state,
        query=torch.ones(
            (NODES, 1),
        ),
    )
    module = _additive_module(
        mode=(
            ATTENTION_MODE_HAZARD_CONDITIONED
        ),
        node_state_dim=1,
        hazard_query_dim=1,
        hidden_dim=1,
    )

    with torch.no_grad():
        module.source_projection.weight.fill_(
            1.0
        )
        module.target_projection.weight.zero_()
        assert module.hazard_projection is not None
        module.hazard_projection.weight.fill_(
            1.0
        )
        module.relation_embeddings.zero_()
        module.score_vectors.fill_(1.0)

    score_a = module.score_tensor(
        inputs_a
    )
    score_b = module.score_tensor(
        inputs_b
    )

    # Edges 0 and 1 share target 1 and relation 0.
    difference_a = (
        score_a[0, 0]
        - score_a[1, 0]
    )
    difference_b = (
        score_b[0, 0]
        - score_b[1, 0]
    )

    assert not torch.allclose(
        difference_a,
        difference_b,
    )


def test_zero_hazard_projection_reduces_to_hazard_blind_formula() -> None:
    inputs = _inputs()
    blind = _additive_module(
        mode=ATTENTION_MODE_HAZARD_BLIND,
    )
    conditioned = _additive_module(
        mode=(
            ATTENTION_MODE_HAZARD_CONDITIONED
        ),
    )

    with torch.no_grad():
        conditioned.source_projection.weight.copy_(
            blind.source_projection.weight
        )
        conditioned.target_projection.weight.copy_(
            blind.target_projection.weight
        )
        conditioned.relation_embeddings.copy_(
            blind.relation_embeddings
        )
        conditioned.score_vectors.copy_(
            blind.score_vectors
        )
        assert (
            conditioned.hazard_projection
            is not None
        )
        conditioned.hazard_projection.weight.zero_()

    assert torch.allclose(
        conditioned.score_tensor(inputs),
        blind.score_tensor(inputs),
    )


def test_edge_order_equivariance() -> None:
    permutation = torch.tensor(
        [6, 2, 0, 5, 1, 4, 3],
        dtype=torch.long,
    )
    original = _inputs()
    permuted = _inputs(
        permutation=permutation,
    )
    module = _additive_module()

    original_scores = (
        module.score_tensor(original)
    )
    permuted_scores = (
        module.score_tensor(permuted)
    )

    assert torch.allclose(
        permuted_scores,
        original_scores[permutation],
    )


def test_identical_edge_inputs_produce_identical_scores() -> None:
    edge_index = torch.tensor(
        [
            [0, 0, 1, 0, 3, 4, 4],
            [1, 1, 2, 2, 4, 4, 3],
        ],
        dtype=torch.long,
    )
    relations = _base_edge_relations()
    edge_batch = _base_edge_batch_index()
    inputs = _inputs(
        edge_index=edge_index,
        edge_relations=relations,
        edge_batch_index=edge_batch,
    )
    module = _additive_module()
    scores = module.score_tensor(inputs)

    assert torch.equal(
        scores[0],
        scores[1],
    )


def test_edge_attributes_do_not_change_scores() -> None:
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
    module = _additive_module()

    assert torch.equal(
        module.score_tensor(inputs_a),
        module.score_tensor(inputs_b),
    )


def test_semantic_edge_weights_do_not_change_scores() -> None:
    weights_a = torch.ones(
        EDGES,
    )
    weights_b = torch.linspace(
        0.1,
        2.0,
        EDGES,
    )
    inputs_a = _inputs(
        semantic_edge_weight=weights_a,
    )
    inputs_b = _inputs(
        semantic_edge_weight=weights_b,
    )
    module = _additive_module()

    assert torch.equal(
        module.score_tensor(inputs_a),
        module.score_tensor(inputs_b),
    )


def test_runtime_input_objects_are_not_mutated() -> None:
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

    _ = _additive_module()(inputs)

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
# Gradient behavior
# =============================================================================


def test_conditioned_gradients_reach_inputs_and_all_parameters() -> None:
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
    module = _additive_module(
        hidden_dim=3,
    )
    _fill_parameters(
        module,
        0.1,
    )

    loss = (
        module.score_tensor(inputs)
        .square()
        .sum()
    )
    loss.backward()

    assert state.grad is not None
    assert query.grad is not None
    assert torch.isfinite(state.grad).all()
    assert torch.isfinite(query.grad).all()
    assert float(
        state.grad.abs().sum().item()
    ) > 0.0
    assert float(
        query.grad.abs().sum().item()
    ) > 0.0

    for name, parameter in (
        module.named_parameters()
    ):
        assert parameter.grad is not None, name
        assert torch.isfinite(
            parameter.grad
        ).all(), name
        assert float(
            parameter.grad
            .abs()
            .sum()
            .item()
        ) > 0.0, name


def test_hazard_blind_does_not_create_query_gradient() -> None:
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
    module = _additive_module(
        mode=ATTENTION_MODE_HAZARD_BLIND,
    )
    _fill_parameters(
        module,
        0.1,
    )

    module.score_tensor(inputs).sum().backward()

    assert state.grad is not None
    assert query.grad is None


def test_uniform_output_has_no_trainable_gradient_path() -> None:
    inputs = _inputs(
        with_hazard_query=False,
    )
    module = _uniform_module()
    scores = module.score_tensor(inputs)

    assert scores.requires_grad is False
    assert tuple(module.parameters()) == ()


# =============================================================================
# Fingerprints, reproducibility, and state restoration
# =============================================================================


def test_uniform_architecture_fingerprint_is_deterministic() -> None:
    first = _uniform_module()
    second = _uniform_module()

    assert (
        first.architecture_fingerprint()
        == second.architecture_fingerprint()
    )
    assert first.parameter_fingerprint() is None
    assert second.parameter_fingerprint() is None


def test_additive_architecture_fingerprint_ignores_parameter_values() -> None:
    first = _additive_module()
    second = _additive_module()
    _fill_parameters(first, 0.1)
    _fill_parameters(second, 0.9)

    assert (
        first.architecture_fingerprint()
        == second.architecture_fingerprint()
    )
    assert (
        first.parameter_fingerprint()
        != second.parameter_fingerprint()
    )


@pytest.mark.parametrize(
    "change",
    (
        "mode",
        "hidden_dim",
        "num_heads",
        "relation_axis",
    ),
)
def test_additive_architecture_fingerprint_changes_with_architecture(
    change: str,
) -> None:
    baseline = _additive_module()

    if change == "mode":
        changed = _additive_module(
            mode=ATTENTION_MODE_HAZARD_BLIND,
        )
    elif change == "hidden_dim":
        changed = _additive_module(
            hidden_dim=(
                ATTENTION_HIDDEN_DIM + 1
            ),
        )
    elif change == "num_heads":
        changed = _additive_module(
            mode=(
                ATTENTION_MODE_MULTIHEAD_HAZARD_CONDITIONED
            ),
            num_heads=2,
        )
    else:
        changed = _additive_module(
            relation_names=(
                "spatial_adjacency",
                "drainage_dependency",
                "service_access",
            ),
            stable_relation_ids=(
                100,
                310,
                700,
            ),
        )

    assert (
        baseline.architecture_fingerprint()
        != changed.architecture_fingerprint()
    )


def test_parameter_fingerprint_changes_with_parameter_value() -> None:
    module = _additive_module()
    before = module.parameter_fingerprint()

    with torch.no_grad():
        module.score_vectors[0, 0] += 1.0

    after = module.parameter_fingerprint()

    assert before is not None
    assert after is not None
    assert before != after


def test_reset_parameters_changes_zeroed_parameter_state() -> None:
    module = _additive_module()
    _fill_parameters(module, 0.0)
    zero_fingerprint = (
        module.parameter_fingerprint()
    )

    module.reset_parameters()
    reset_fingerprint = (
        module.parameter_fingerprint()
    )

    assert zero_fingerprint != (
        reset_fingerprint
    )
    module.assert_finite_parameters()


def test_state_dict_round_trip_preserves_scores_and_fingerprint() -> None:
    inputs = _inputs()
    original = _additive_module()
    restored = _additive_module()

    restored.load_state_dict(
        original.state_dict()
    )

    assert torch.equal(
        original.score_tensor(inputs),
        restored.score_tensor(inputs),
    )
    assert (
        original.parameter_fingerprint()
        == restored.parameter_fingerprint()
    )


def test_manual_seed_reproduces_initial_parameters() -> None:
    torch.manual_seed(12345)
    first = _additive_module()

    torch.manual_seed(12345)
    second = _additive_module()

    assert (
        first.parameter_fingerprint()
        == second.parameter_fingerprint()
    )


def test_relation_axis_fingerprint_is_deterministic() -> None:
    first = _additive_module()
    second = _additive_module()

    assert (
        first.relation_axis_fingerprint
        == second.relation_axis_fingerprint
    )


# =============================================================================
# Construction from configuration and dispatcher
# =============================================================================


@pytest.mark.parametrize(
    (
        "mode",
        "heads",
        "expected_type",
    ),
    (
        (
            ATTENTION_MODE_UNIFORM,
            1,
            UniformEdgeAttentionScoreFunction,
        ),
        (
            ATTENTION_MODE_HAZARD_BLIND,
            1,
            AdditiveEdgeAttentionScoreFunction,
        ),
        (
            ATTENTION_MODE_HAZARD_CONDITIONED,
            1,
            AdditiveEdgeAttentionScoreFunction,
        ),
        (
            ATTENTION_MODE_MULTIHEAD_HAZARD_CONDITIONED,
            MULTIHEAD_COUNT,
            AdditiveEdgeAttentionScoreFunction,
        ),
    ),
)
def test_dispatcher_builds_selected_implementation(
    mode: str,
    heads: int,
    expected_type: type[nn.Module],
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
    module = (
        build_edge_attention_score_function(
            config=_config(
                mode=mode,
                heads=heads,
            ),
            source_inputs=inputs,
            hidden_dim=(
                ATTENTION_HIDDEN_DIM
            ),
        )
    )

    assert isinstance(
        module,
        expected_type,
    )
    assert module.mode == mode
    assert module.num_heads == heads


def test_additive_from_config_infers_dimensions_and_relation_axis() -> None:
    inputs = _inputs()
    module = (
        AdditiveEdgeAttentionScoreFunction
        .from_config(
            config=_config(
                mode=(
                    ATTENTION_MODE_HAZARD_CONDITIONED
                ),
            ),
            source_inputs=inputs,
            hidden_dim=7,
        )
    )

    assert module.node_state_dim == (
        HIDDEN_DIM
    )
    assert module.hazard_query_dim == (
        QUERY_DIM
    )
    assert module.hidden_dim == 7
    assert module.relation_names == (
        RELATION_NAMES
    )
    assert module.stable_relation_ids == (
        STABLE_RELATION_IDS
    )


def test_hazard_blind_from_config_does_not_require_query() -> None:
    inputs = _inputs(
        with_hazard_query=False,
    )
    module = (
        AdditiveEdgeAttentionScoreFunction
        .from_config(
            config=_config(
                mode=(
                    ATTENTION_MODE_HAZARD_BLIND
                ),
            ),
            source_inputs=inputs,
        )
    )

    assert module.hazard_query_dim is None
    assert module.uses_hazard_query is False


def test_multihead_from_config_uses_configured_head_count() -> None:
    inputs = _inputs()
    module = (
        AdditiveEdgeAttentionScoreFunction
        .from_config(
            config=_config(
                mode=(
                    ATTENTION_MODE_MULTIHEAD_HAZARD_CONDITIONED
                ),
                heads=MULTIHEAD_COUNT,
            ),
            source_inputs=inputs,
        )
    )

    assert module.num_heads == (
        MULTIHEAD_COUNT
    )


def test_from_config_moves_parameters_to_input_dtype() -> None:
    inputs = _inputs(
        dtype=torch.float64,
    )
    module = (
        AdditiveEdgeAttentionScoreFunction
        .from_config(
            config=_config(
                mode=(
                    ATTENTION_MODE_HAZARD_CONDITIONED
                ),
            ),
            source_inputs=inputs,
        )
    )

    assert all(
        parameter.dtype == torch.float64
        for parameter in module.parameters()
    )


# =============================================================================
# Invalid construction and dispatcher contracts
# =============================================================================


@pytest.mark.parametrize(
    "bad_value",
    (
        0,
        -1,
    ),
)
def test_constructor_rejects_nonpositive_dimensions(
    bad_value: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="positive",
    ):
        _additive_module(
            node_state_dim=bad_value,
        )

    with pytest.raises(
        ValueError,
        match="positive",
    ):
        _additive_module(
            hidden_dim=bad_value,
        )


@pytest.mark.parametrize(
    "bad_value",
    (
        True,
        1.5,
        "4",
    ),
)
def test_constructor_rejects_noninteger_dimensions(
    bad_value: object,
) -> None:
    with pytest.raises(
        TypeError,
        match="integer",
    ):
        _additive_module(
            node_state_dim=bad_value,  # type: ignore[arg-type]
        )


def test_conditioned_constructor_requires_hazard_dimension() -> None:
    with pytest.raises(
        ValueError,
        match="hazard_query_dim",
    ):
        AdditiveEdgeAttentionScoreFunction(
            node_state_dim=HIDDEN_DIM,
            hazard_query_dim=None,
            relation_names=RELATION_NAMES,
            stable_relation_ids=(
                STABLE_RELATION_IDS
            ),
            hidden_dim=(
                ATTENTION_HIDDEN_DIM
            ),
            mode=(
                ATTENTION_MODE_HAZARD_CONDITIONED
            ),
            num_heads=1,
        )


def test_hazard_blind_constructor_rejects_hazard_dimension() -> None:
    with pytest.raises(
        ValueError,
        match="must be None",
    ):
        AdditiveEdgeAttentionScoreFunction(
            node_state_dim=HIDDEN_DIM,
            hazard_query_dim=QUERY_DIM,
            relation_names=RELATION_NAMES,
            stable_relation_ids=(
                STABLE_RELATION_IDS
            ),
            hidden_dim=(
                ATTENTION_HIDDEN_DIM
            ),
            mode=(
                ATTENTION_MODE_HAZARD_BLIND
            ),
            num_heads=1,
        )


@pytest.mark.parametrize(
    (
        "mode",
        "heads",
        "match",
    ),
    (
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
def test_constructor_rejects_invalid_head_contract(
    mode: str,
    heads: int,
    match: str,
) -> None:
    with pytest.raises(
        ValueError,
        match=match,
    ):
        _additive_module(
            mode=mode,
            num_heads=heads,
        )


def test_additive_constructor_rejects_uniform_mode() -> None:
    with pytest.raises(
        ValueError,
        match="learned attention mode",
    ):
        AdditiveEdgeAttentionScoreFunction(
            node_state_dim=HIDDEN_DIM,
            hazard_query_dim=None,
            relation_names=RELATION_NAMES,
            stable_relation_ids=(
                STABLE_RELATION_IDS
            ),
            hidden_dim=(
                ATTENTION_HIDDEN_DIM
            ),
            mode=ATTENTION_MODE_UNIFORM,
            num_heads=1,
        )


def test_constructor_rejects_semantic_weight_mode() -> None:
    with pytest.raises(
        NotImplementedError,
        match="data-coefficient mode",
    ):
        AdditiveEdgeAttentionScoreFunction(
            node_state_dim=HIDDEN_DIM,
            hazard_query_dim=QUERY_DIM,
            relation_names=RELATION_NAMES,
            stable_relation_ids=(
                STABLE_RELATION_IDS
            ),
            hidden_dim=(
                ATTENTION_HIDDEN_DIM
            ),
            mode=(
                ATTENTION_MODE_SEMANTIC_WEIGHT
            ),
            num_heads=1,
        )


def test_constructor_rejects_unknown_mode() -> None:
    with pytest.raises(
        ValueError,
        match="Unknown attention mode",
    ):
        AdditiveEdgeAttentionScoreFunction(
            node_state_dim=HIDDEN_DIM,
            hazard_query_dim=QUERY_DIM,
            relation_names=RELATION_NAMES,
            stable_relation_ids=(
                STABLE_RELATION_IDS
            ),
            hidden_dim=(
                ATTENTION_HIDDEN_DIM
            ),
            mode="unknown-mode",
            num_heads=1,
        )


def test_constructor_rejects_empty_relation_axis() -> None:
    with pytest.raises(
        ValueError,
        match="At least one",
    ):
        UniformEdgeAttentionScoreFunction(
            relation_names=(),
            stable_relation_ids=(),
        )


def test_constructor_rejects_relation_axis_length_mismatch() -> None:
    with pytest.raises(
        ValueError,
        match="must align",
    ):
        UniformEdgeAttentionScoreFunction(
            relation_names=RELATION_NAMES,
            stable_relation_ids=(
                STABLE_RELATION_IDS[:-1]
            ),
        )


def test_constructor_rejects_duplicate_relation_names() -> None:
    with pytest.raises(
        ValueError,
        match="duplicates",
    ):
        UniformEdgeAttentionScoreFunction(
            relation_names=(
                "same",
                "same",
            ),
            stable_relation_ids=(
                1,
                2,
            ),
        )


def test_constructor_rejects_duplicate_stable_ids() -> None:
    with pytest.raises(
        ValueError,
        match="duplicates",
    ):
        UniformEdgeAttentionScoreFunction(
            relation_names=(
                "first",
                "second",
            ),
            stable_relation_ids=(
                1,
                1,
            ),
        )


def test_constructor_rejects_negative_stable_id() -> None:
    with pytest.raises(
        ValueError,
        match="nonnegative",
    ):
        UniformEdgeAttentionScoreFunction(
            relation_names=("relation",),
            stable_relation_ids=(-1,),
        )


def test_constructor_rejects_boolean_stable_id() -> None:
    with pytest.raises(
        TypeError,
        match="integer",
    ):
        UniformEdgeAttentionScoreFunction(
            relation_names=("relation",),
            stable_relation_ids=(True,),  # type: ignore[arg-type]
        )


def test_uniform_from_config_rejects_wrong_mode() -> None:
    with pytest.raises(
        ValueError,
        match="requires attention_mode='uniform'",
    ):
        (
            UniformEdgeAttentionScoreFunction
            .from_config(
                config=_config(
                    mode=(
                        ATTENTION_MODE_HAZARD_CONDITIONED
                    ),
                ),
                source_inputs=_inputs(),
            )
        )


def test_additive_from_config_rejects_uniform_mode() -> None:
    with pytest.raises(
        ValueError,
        match="requires a learned attention mode",
    ):
        (
            AdditiveEdgeAttentionScoreFunction
            .from_config(
                config=_config(
                    mode=ATTENTION_MODE_UNIFORM,
                ),
                source_inputs=_inputs(
                    with_hazard_query=False,
                ),
            )
        )


def test_from_config_rejects_wrong_config_type() -> None:
    with pytest.raises(
        TypeError,
        match="FunctionalMessagePassingConfig",
    ):
        (
            UniformEdgeAttentionScoreFunction
            .from_config(
                config=object(),  # type: ignore[arg-type]
                source_inputs=_inputs(
                    with_hazard_query=False,
                ),
            )
        )

    with pytest.raises(
        TypeError,
        match="FunctionalMessagePassingConfig",
    ):
        (
            AdditiveEdgeAttentionScoreFunction
            .from_config(
                config=object(),  # type: ignore[arg-type]
                source_inputs=_inputs(),
            )
        )


def test_dispatcher_rejects_wrong_config_type() -> None:
    with pytest.raises(
        TypeError,
        match="FunctionalMessagePassingConfig",
    ):
        build_edge_attention_score_function(
            config=object(),  # type: ignore[arg-type]
            source_inputs=_inputs(),
        )


def test_dispatcher_rejects_semantic_weight_mode() -> None:
    with pytest.raises(
        NotImplementedError,
        match="data-coefficient mode",
    ):
        build_edge_attention_score_function(
            config=_config(
                mode=(
                    ATTENTION_MODE_SEMANTIC_WEIGHT
                ),
            ),
            source_inputs=_inputs(),
        )


def test_conditioned_from_config_rejects_missing_query() -> None:
    with pytest.raises(
        ValueError,
        match="node_hazard_query",
    ):
        (
            AdditiveEdgeAttentionScoreFunction
            .from_config(
                config=_config(
                    mode=(
                        ATTENTION_MODE_HAZARD_CONDITIONED
                    ),
                ),
                source_inputs=_inputs(
                    with_hazard_query=False,
                ),
            )
        )


# =============================================================================
# Runtime validation failures
# =============================================================================


def test_uniform_rejects_wrong_source_input_type() -> None:
    with pytest.raises(
        TypeError,
        match="FunctionalMessagePassingInputs",
    ):
        _uniform_module().score_tensor(
            object()  # type: ignore[arg-type]
        )


def test_additive_rejects_wrong_source_input_type() -> None:
    with pytest.raises(
        TypeError,
        match="FunctionalMessagePassingInputs",
    ):
        _additive_module().score_tensor(
            object()  # type: ignore[arg-type]
        )


def test_runtime_relation_names_must_match_architecture() -> None:
    inputs = _inputs(
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
        match="relation_names",
    ):
        _additive_module().score_tensor(
            inputs
        )


def test_runtime_stable_ids_must_match_architecture() -> None:
    inputs = _inputs(
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
        match="stable_relation_ids",
    ):
        _additive_module().score_tensor(
            inputs
        )


def test_runtime_node_state_width_must_match_architecture() -> None:
    inputs = _inputs()

    with pytest.raises(
        ValueError,
        match="shape",
    ):
        _additive_module(
            node_state_dim=(
                HIDDEN_DIM + 1
            ),
        ).score_tensor(inputs)


def test_runtime_hazard_width_must_match_architecture() -> None:
    inputs = _inputs()

    with pytest.raises(
        ValueError,
        match="shape",
    ):
        _additive_module(
            hazard_query_dim=(
                QUERY_DIM + 1
            ),
        ).score_tensor(inputs)


def test_conditioned_runtime_requires_query() -> None:
    inputs = _inputs(
        with_hazard_query=False,
    )

    with pytest.raises(
        ValueError,
        match="node_hazard_query",
    ):
        _additive_module().score_tensor(
            inputs
        )


def test_runtime_parameter_dtype_must_match_input_dtype() -> None:
    inputs = _inputs(
        dtype=torch.float32,
    )
    module = (
        _additive_module()
        .to(dtype=torch.float64)
    )

    with pytest.raises(
        ValueError,
        match="floating-point dtype",
    ):
        module.score_tensor(inputs)


def test_runtime_rejects_nonfinite_node_state_after_construction() -> None:
    inputs = _inputs()
    inputs.node_state.fused_state[
        0,
        0,
    ] = float("nan")

    with pytest.raises(
        ValueError,
        match="finite",
    ):
        _additive_module().score_tensor(
            inputs
        )


def test_runtime_rejects_nonfinite_hazard_query_after_construction() -> None:
    inputs = _inputs()
    assert inputs.node_hazard_query is not None
    inputs.node_hazard_query[
        0,
        0,
    ] = float("inf")

    with pytest.raises(
        ValueError,
        match="finite",
    ):
        _additive_module().score_tensor(
            inputs
        )


@pytest.mark.parametrize(
    "invalid_value",
    (
        float("nan"),
        float("inf"),
        float("-inf"),
    ),
)
def test_forward_rejects_nonfinite_parameters(
    invalid_value: float,
) -> None:
    module = _additive_module()

    with torch.no_grad():
        module.score_vectors[
            0,
            0,
        ] = invalid_value

    with pytest.raises(
        FloatingPointError,
        match="parameter",
    ):
        module(_inputs())


def test_parameter_finiteness_check_names_bad_parameter() -> None:
    module = _additive_module()

    with torch.no_grad():
        module.relation_embeddings[
            0,
            0,
            0,
        ] = float("nan")

    with pytest.raises(
        FloatingPointError,
        match="relation_embeddings",
    ):
        module.assert_finite_parameters()


def test_mixed_parameter_dtypes_are_rejected() -> None:
    module = _additive_module()
    module.score_vectors = nn.Parameter(
        module.score_vectors
        .detach()
        .to(dtype=torch.float64)
    )

    with pytest.raises(
        RuntimeError,
        match="share one floating-point dtype",
    ):
        module.score_tensor(_inputs())


# =============================================================================
# Optional CUDA contracts
# =============================================================================


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is not available.",
)
def test_uniform_cuda_output() -> None:
    inputs = _inputs(
        device="cuda",
        with_hazard_query=False,
    )
    module = _uniform_module().to(
        device="cuda",
    )
    output = module(inputs)

    assert output.device.type == "cuda"


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is not available.",
)
def test_additive_cuda_forward_and_gradients() -> None:
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
    module = _additive_module().to(
        device="cuda",
    )

    loss = module.score_tensor(
        inputs
    ).square().sum()
    loss.backward()

    assert loss.device.type == "cuda"
    assert state.grad is not None
    assert query.grad is not None


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is not available.",
)
def test_additive_rejects_cpu_module_for_cuda_inputs() -> None:
    inputs = _inputs(
        device="cuda",
    )
    module = _additive_module()

    with pytest.raises(
        ValueError,
        match="share one device",
    ):
        module.score_tensor(inputs)
