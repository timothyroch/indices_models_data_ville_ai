"""
Focused tests for exact target-node/relation attention normalization.

Target repository path:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            tests/
                test_attention_normalization.py

Implementation under test:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            functional_message_passing/
                edge_attention/
                    attention_normalization.py

The suite isolates the normalization stage from score prediction, relation
gating, relation transforms, head reduction, message construction,
aggregation, and explanation exporters.

Scientific contracts frozen here
--------------------------------
- The normalization group is exactly ``(target node, exact compiled relation)``.
- Dense group IDs are:
      target_index * num_relations + edge_relation_index
- Stable ontology IDs never participate in group arithmetic.
- Every attention head is normalized independently.
- Every nonempty group sums to one per head.
- Absent groups remain explicit zero-count entries on the dense ``N * R`` axis.
- Singleton groups receive exact weight one.
- Uniform zero logits produce reciprocal group-size weights.
- Group-constant logit shifts do not alter attention.
- Changing one group cannot alter another group.
- Edge permutations permute outputs equivariantly.
- Extreme finite logits remain numerically stable.
- Empty edge sets are valid.
- Gradients remain connected to raw logits.
- Disabled attention is not implemented here; this module only normalizes
  enabled attention.
- The module is parameter-free and owns no persistent data-dependent buffers.
- Control/placebo relations use the same mathematics as substantive relations.
- Attention weights are routing coefficients, not causal importance scores.

Controlled upstream doubles are patched into the existing
``functional_message_passing.schemas`` module. This keeps failures local to the
attention-normalization boundary instead of coupling them to graph loading,
fusion, hazard encoding, or registry construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
import torch
from torch import nn

from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.config import (
    FunctionalMessagePassingConfig,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.constants import (
    ATTENTION_MODE_HAZARD_CONDITIONED,
    ATTENTION_MODE_MULTIHEAD_HAZARD_CONDITIONED,
    ATTENTION_MODE_UNIFORM,
    ATTENTION_NORMALIZATION_GLOBAL_RELATION,
    ATTENTION_NORMALIZATION_TARGET_NODE,
    ATTENTION_NORMALIZATION_TARGET_NODE_RELATION,
    ATTENTION_NORMALIZATION_UNNORMALIZED_SIGMOID,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.functional_message_passing import (
    schemas as fmp_schemas,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.functional_message_passing.edge_attention.attention_normalization import (
    ATTENTION_GROUP_ID_FORMULA,
    ATTENTION_GROUP_KEY_TARGET_NODE_EXACT_RELATION,
    ATTENTION_NORMALIZATION_SCHEMA_VERSION,
    IMPLEMENTED_ATTENTION_NORMALIZATION_MODES,
    AttentionNormalization,
    AttentionNormalizer,
    EdgeAttentionNormalizer,
    TargetNodeRelationAttentionNormalization,
    apply_attention_normalization,
    assert_attention_normalized,
    attention_group_sums,
    build_attention_normalizer,
    build_edge_attention_normalizer,
    build_target_node_relation_group_counts,
    build_target_node_relation_group_ids,
    maximum_attention_normalization_error,
    normalize_attention_logits,
    normalize_edge_attention_scores,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.functional_message_passing.edge_attention.schemas import (
    EDGE_ATTENTION_INPUT_EXACT_RELATION_EMBEDDING,
    EDGE_ATTENTION_INPUT_SOURCE_NODE_STATE,
    EDGE_ATTENTION_INPUT_TARGET_HAZARD_QUERY,
    EDGE_ATTENTION_INPUT_TARGET_NODE_STATE,
    EDGE_ATTENTION_SCORE_FUNCTION_ADDITIVE,
    EDGE_ATTENTION_SCORE_FUNCTION_UNIFORM,
    AttentionNormalizationOutput,
    EdgeAttentionScoreOutput,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.functional_message_passing.schemas import (
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
# Controlled graph, score, and configuration helpers
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
    # Dense target/relation groups:
    #   group 3  = target 1, relation 0: edges 0, 1
    #   group 7  = target 2, relation 1: edge 2
    #   group 8  = target 2, relation 2: edge 3
    #   group 14 = target 4, relation 2: edges 4, 5
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


def _expected_group_ids(
    *,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    return torch.tensor(
        [3, 3, 7, 8, 14, 14, 10],
        dtype=torch.long,
        device=device,
    )


def _node_state(
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> FakeNodeStateFusionOutput:
    node_batch = _node_batch_index(
        device=device,
    )
    state = (
        torch.arange(
            NODES * HIDDEN_DIM,
            dtype=dtype,
            device=device,
        )
        .reshape(NODES, HIDDEN_DIM)
        / 10.0
    )
    return FakeNodeStateFusionOutput(
        fused_state=state,
        alignment=FakeAlignment(
            item_ids=_node_ids(),
            node_batch_index=node_batch,
            graph_count=GRAPHS,
        ),
    )


def _hazard_query(
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> FakeHazardQueryEncoding:
    node_batch = _node_batch_index(
        device=device,
    )
    query = (
        torch.arange(
            NODES * QUERY_DIM,
            dtype=dtype,
            device=device,
        )
        .reshape(NODES, QUERY_DIM)
        / 7.0
    )
    return FakeHazardQueryEncoding(
        query=query,
        source_embedding=(
            FakeNodeAlignedHazardEmbeddingLookup(
                node_batch_index=node_batch,
            )
        ),
    )


def _registry() -> FakeCompiledRelationRegistry:
    return FakeCompiledRelationRegistry()


def _graph(
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
    empty_edges: bool = False,
    permutation: torch.Tensor | None = None,
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

    return FakeUrbanGraphBatch(
        external_node_ids=_node_ids(),
        node_batch_index=node_batch,
        edge_index=edge_index,
        edge_relation_type=relations,
        edge_attributes=(
            torch.empty(
                (0, 2),
                dtype=dtype,
                device=device,
            )
            if empty_edges
            else torch.arange(
                EDGES * 2,
                dtype=dtype,
                device=device,
            ).reshape(EDGES, 2)
        ),
        semantic_edge_weight=(
            torch.empty(
                (0,),
                dtype=dtype,
                device=device,
            )
            if empty_edges
            else torch.linspace(
                0.5,
                1.5,
                EDGES,
                dtype=dtype,
                device=device,
            )
        ),
        edge_batch_index=edge_batch,
    )


def _inputs(
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
    empty_edges: bool = False,
    permutation: torch.Tensor | None = None,
) -> FunctionalMessagePassingInputs:
    return FunctionalMessagePassingInputs(
        source_graph=_graph(
            dtype=dtype,
            device=device,
            empty_edges=empty_edges,
            permutation=permutation,
        ),
        node_state=_node_state(
            dtype=dtype,
            device=device,
        ),
        compiled_relation_registry=_registry(),
        hazard_query=_hazard_query(
            dtype=dtype,
            device=device,
        ),
        source_fingerprint="attention-normalization-test-input",
    )


def _raw_logits(
    inputs: FunctionalMessagePassingInputs,
    *,
    heads: int = 1,
    uniform: bool = False,
    requires_grad: bool = False,
) -> torch.Tensor:
    if uniform:
        logits = torch.zeros(
            (inputs.num_edges, heads),
            dtype=inputs.dtype,
            device=inputs.device,
        )
    elif inputs.num_edges == 0:
        logits = torch.empty(
            (0, heads),
            dtype=inputs.dtype,
            device=inputs.device,
        )
    else:
        logits = (
            torch.arange(
                inputs.num_edges * heads,
                dtype=inputs.dtype,
                device=inputs.device,
            )
            .reshape(inputs.num_edges, heads)
            / 4.0
        )

    logits = logits.detach().clone()

    if requires_grad:
        logits.requires_grad_(True)

    return logits


def _score_output(
    inputs: FunctionalMessagePassingInputs,
    *,
    mode: str = ATTENTION_MODE_HAZARD_CONDITIONED,
    heads: int = 1,
    logits: torch.Tensor | None = None,
) -> EdgeAttentionScoreOutput:
    if mode == ATTENTION_MODE_UNIFORM:
        resolved_logits = (
            _raw_logits(
                inputs,
                heads=1,
                uniform=True,
            )
            if logits is None
            else logits
        )
        score_function = (
            EDGE_ATTENTION_SCORE_FUNCTION_UNIFORM
        )
        input_features: tuple[str, ...] = ()
    else:
        resolved_logits = (
            _raw_logits(
                inputs,
                heads=heads,
            )
            if logits is None
            else logits
        )
        score_function = (
            EDGE_ATTENTION_SCORE_FUNCTION_ADDITIVE
        )
        input_features = (
            HAZARD_CONDITIONED_FEATURES
        )

    return EdgeAttentionScoreOutput(
        raw_scores_by_head=resolved_logits,
        source_inputs=inputs,
        relation_names=inputs.relation_names,
        stable_relation_ids=(
            inputs.stable_relation_ids
        ),
        compiled_relation_registry_fingerprint=(
            inputs
            .compiled_relation_registry
            .fingerprint()
        ),
        attention_mode=mode,
        score_function=score_function,
        input_feature_names=input_features,
        encoder_architecture_fingerprint=(
            "score-architecture"
        ),
        parameter_fingerprint=(
            None
            if mode == ATTENTION_MODE_UNIFORM
            else "score-parameters"
        ),
    )


def _multihead_score(
    inputs: FunctionalMessagePassingInputs,
    *,
    logits: torch.Tensor | None = None,
) -> EdgeAttentionScoreOutput:
    return _score_output(
        inputs,
        mode=(
            ATTENTION_MODE_MULTIHEAD_HAZARD_CONDITIONED
        ),
        heads=MULTIHEAD_COUNT,
        logits=logits,
    )


def _config(
    *,
    normalization: str = (
        ATTENTION_NORMALIZATION_TARGET_NODE_RELATION
    ),
) -> FunctionalMessagePassingConfig:
    return FunctionalMessagePassingConfig(
        enabled=True,
        attention_enabled=True,
        attention_mode=ATTENTION_MODE_UNIFORM,
        attention_heads=1,
        attention_normalization=normalization,
    )


def _manual_group_softmax(
    logits: torch.Tensor,
    group_ids: torch.Tensor,
    *,
    num_groups: int,
) -> torch.Tensor:
    output = torch.empty_like(logits)

    for group_id in range(num_groups):
        mask = group_ids == group_id
        if not bool(mask.any().item()):
            continue

        output[mask] = torch.softmax(
            logits[mask],
            dim=0,
        )

    return output


# =============================================================================
# Public identity and aliases
# =============================================================================


def test_public_identity_constants() -> None:
    assert isinstance(
        ATTENTION_NORMALIZATION_SCHEMA_VERSION,
        str,
    )
    assert (
        ATTENTION_NORMALIZATION_SCHEMA_VERSION
        .strip()
    )
    assert (
        ATTENTION_GROUP_KEY_TARGET_NODE_EXACT_RELATION
        == "target_node_exact_relation"
    )
    assert ATTENTION_GROUP_ID_FORMULA == (
        "target_index * num_relations + edge_relation_index"
    )
    assert (
        IMPLEMENTED_ATTENTION_NORMALIZATION_MODES
        == (
            ATTENTION_NORMALIZATION_TARGET_NODE_RELATION,
        )
    )


def test_class_aliases_are_exact() -> None:
    assert AttentionNormalization is (
        TargetNodeRelationAttentionNormalization
    )
    assert EdgeAttentionNormalizer is (
        TargetNodeRelationAttentionNormalization
    )
    module: AttentionNormalizer = (
        TargetNodeRelationAttentionNormalization()
    )
    assert isinstance(
        module,
        TargetNodeRelationAttentionNormalization,
    )


def test_function_and_builder_aliases_are_exact() -> None:
    assert apply_attention_normalization is (
        normalize_edge_attention_scores
    )
    assert build_edge_attention_normalizer is (
        build_attention_normalizer
    )


# =============================================================================
# Deterministic group construction
# =============================================================================


def test_group_ids_match_exact_formula() -> None:
    inputs = _inputs()

    observed = (
        build_target_node_relation_group_ids(
            inputs
        )
    )
    expected = (
        inputs.target_index
        * inputs.num_relations
        + inputs.edge_relation_index
    )

    assert torch.equal(
        observed,
        expected,
    )
    assert torch.equal(
        observed,
        _expected_group_ids(),
    )
    assert observed.dtype == torch.long
    assert observed.device == inputs.device


def test_group_ids_match_source_input_contract() -> None:
    inputs = _inputs()

    assert torch.equal(
        build_target_node_relation_group_ids(
            inputs
        ),
        inputs.attention_group_id,
    )


def test_group_ids_do_not_use_sparse_stable_relation_ids() -> None:
    inputs = _inputs()
    observed = (
        build_target_node_relation_group_ids(
            inputs
        )
    )

    assert STABLE_RELATION_IDS != (
        0,
        1,
        2,
    )
    assert int(observed.max().item()) < (
        inputs.num_nodes
        * inputs.num_relations
    )


def test_group_ids_preserve_control_relation_math() -> None:
    inputs = _inputs()
    group_ids = (
        build_target_node_relation_group_ids(
            inputs
        )
    )
    control_edges = inputs.control_edge_mask

    assert bool(control_edges.any().item())
    assert torch.equal(
        group_ids[control_edges],
        (
            inputs.target_index[control_edges]
            * inputs.num_relations
            + inputs.edge_relation_index[
                control_edges
            ]
        ),
    )


def test_group_ids_empty_edges() -> None:
    inputs = _inputs(
        empty_edges=True,
    )
    group_ids = (
        build_target_node_relation_group_ids(
            inputs
        )
    )

    assert group_ids.shape == (0,)
    assert group_ids.dtype == torch.long


def test_group_ids_reject_wrong_source_type() -> None:
    with pytest.raises(
        TypeError,
        match="FunctionalMessagePassingInputs",
    ):
        build_target_node_relation_group_ids(
            object()  # type: ignore[arg-type]
        )


def test_group_counts_cover_complete_dense_axis() -> None:
    inputs = _inputs()
    group_ids = (
        build_target_node_relation_group_ids(
            inputs
        )
    )
    counts = (
        build_target_node_relation_group_counts(
            inputs,
            group_ids=group_ids,
        )
    )

    assert counts.shape == (
        NUM_GROUPS,
    )
    assert counts.dtype == torch.long
    assert int(counts.sum().item()) == EDGES
    assert counts[3].item() == 2
    assert counts[7].item() == 1
    assert counts[8].item() == 1
    assert counts[10].item() == 1
    assert counts[14].item() == 2
    assert int((counts == 0).sum().item()) == (
        NUM_GROUPS - 5
    )


def test_group_counts_default_ids_match_explicit_ids() -> None:
    inputs = _inputs()
    group_ids = (
        build_target_node_relation_group_ids(
            inputs
        )
    )

    assert torch.equal(
        build_target_node_relation_group_counts(
            inputs
        ),
        build_target_node_relation_group_counts(
            inputs,
            group_ids=group_ids,
        ),
    )


def test_group_counts_empty_edges_are_all_zero() -> None:
    inputs = _inputs(
        empty_edges=True,
    )
    counts = (
        build_target_node_relation_group_counts(
            inputs
        )
    )

    assert counts.shape == (
        NUM_GROUPS,
    )
    assert torch.equal(
        counts,
        torch.zeros_like(counts),
    )


def test_group_counts_reject_non_tensor_ids() -> None:
    inputs = _inputs()

    with pytest.raises(
        TypeError,
        match="group_ids must be a tensor",
    ):
        build_target_node_relation_group_counts(
            inputs,
            group_ids=[3] * EDGES,  # type: ignore[arg-type]
        )


def test_group_counts_reject_wrong_id_rank() -> None:
    inputs = _inputs()

    with pytest.raises(
        ValueError,
        match=r"shape \[E\]",
    ):
        build_target_node_relation_group_counts(
            inputs,
            group_ids=(
                inputs.attention_group_id
                .unsqueeze(0)
            ),
        )


def test_group_counts_reject_wrong_id_length() -> None:
    inputs = _inputs()

    with pytest.raises(
        ValueError,
        match="align with the edge axis",
    ):
        build_target_node_relation_group_counts(
            inputs,
            group_ids=(
                inputs.attention_group_id[:-1]
            ),
        )


def test_group_counts_reject_wrong_id_dtype() -> None:
    inputs = _inputs()

    with pytest.raises(
        ValueError,
        match="torch.long",
    ):
        build_target_node_relation_group_counts(
            inputs,
            group_ids=(
                inputs.attention_group_id
                .to(dtype=torch.int32)
            ),
        )


def test_group_counts_reject_semantically_wrong_ids() -> None:
    inputs = _inputs()
    wrong = (
        inputs.attention_group_id.clone()
    )
    wrong[0] = 4

    with pytest.raises(
        ValueError,
        match="target node.*exact dense relation",
    ):
        build_target_node_relation_group_counts(
            inputs,
            group_ids=wrong,
        )


# =============================================================================
# Low-level grouped softmax: exact mathematics
# =============================================================================


def test_low_level_normalization_matches_manual_group_softmax() -> None:
    inputs = _inputs()
    logits = _raw_logits(inputs)
    group_ids = inputs.attention_group_id

    observed = normalize_attention_logits(
        logits,
        group_ids,
        num_groups=NUM_GROUPS,
    )
    expected = _manual_group_softmax(
        logits,
        group_ids,
        num_groups=NUM_GROUPS,
    )

    torch.testing.assert_close(
        observed,
        expected,
    )


def test_low_level_multihead_normalization_matches_manual_result() -> None:
    inputs = _inputs()
    logits = _raw_logits(
        inputs,
        heads=MULTIHEAD_COUNT,
    )
    group_ids = inputs.attention_group_id

    observed = normalize_attention_logits(
        logits,
        group_ids,
        num_groups=NUM_GROUPS,
    )
    expected = _manual_group_softmax(
        logits,
        group_ids,
        num_groups=NUM_GROUPS,
    )

    torch.testing.assert_close(
        observed,
        expected,
    )
    assert observed.shape == (
        EDGES,
        MULTIHEAD_COUNT,
    )


def test_heads_are_normalized_independently() -> None:
    inputs = _inputs()
    group_ids = inputs.attention_group_id
    logits = torch.tensor(
        [
            [0.0, 5.0],
            [2.0, -1.0],
            [3.0, 9.0],
            [4.0, -3.0],
            [1.0, 2.0],
            [1.0, 8.0],
            [-2.0, 4.0],
        ],
        dtype=inputs.dtype,
    )

    weights = normalize_attention_logits(
        logits,
        group_ids,
        num_groups=NUM_GROUPS,
    )

    group_three = group_ids == 3
    expected_head_zero = torch.softmax(
        torch.tensor(
            [0.0, 2.0],
            dtype=inputs.dtype,
        ),
        dim=0,
    )
    expected_head_one = torch.softmax(
        torch.tensor(
            [5.0, -1.0],
            dtype=inputs.dtype,
        ),
        dim=0,
    )

    torch.testing.assert_close(
        weights[group_three, 0],
        expected_head_zero,
    )
    torch.testing.assert_close(
        weights[group_three, 1],
        expected_head_one,
    )
    assert not torch.allclose(
        weights[group_three, 0],
        weights[group_three, 1],
    )


def test_uniform_logits_produce_reciprocal_group_size() -> None:
    inputs = _inputs()
    group_ids = inputs.attention_group_id
    logits = torch.zeros(
        (EDGES, MULTIHEAD_COUNT),
        dtype=inputs.dtype,
    )

    weights = normalize_attention_logits(
        logits,
        group_ids,
        num_groups=NUM_GROUPS,
    )
    counts = segment_counts(
        group_ids,
        num_segments=NUM_GROUPS,
    )
    expected = (
        torch.ones_like(weights)
        / counts[group_ids]
        .to(dtype=weights.dtype)
        .unsqueeze(1)
    )

    assert torch.equal(
        weights,
        expected,
    )


def test_singleton_groups_receive_exact_one() -> None:
    inputs = _inputs()
    group_ids = inputs.attention_group_id
    logits = _raw_logits(
        inputs,
        heads=MULTIHEAD_COUNT,
    )

    weights = normalize_attention_logits(
        logits,
        group_ids,
        num_groups=NUM_GROUPS,
    )
    counts = segment_counts(
        group_ids,
        num_segments=NUM_GROUPS,
    )
    singleton_edges = (
        counts[group_ids] == 1
    )

    assert torch.equal(
        weights[singleton_edges],
        torch.ones_like(
            weights[singleton_edges]
        ),
    )


def test_extreme_logits_are_numerically_stable() -> None:
    inputs = _inputs()
    group_ids = inputs.attention_group_id
    logits = torch.tensor(
        [
            [10000.0, -10000.0],
            [9999.0, -9999.0],
            [1e20, -1e20],
            [-1e20, 1e20],
            [50000.0, -50000.0],
            [49990.0, -49990.0],
            [0.0, 0.0],
        ],
        dtype=inputs.dtype,
    )

    weights = normalize_attention_logits(
        logits,
        group_ids,
        num_groups=NUM_GROUPS,
    )

    assert torch.isfinite(weights).all()
    assert bool((weights >= 0).all().item())
    assert_attention_normalized(
        weights,
        group_ids,
        num_groups=NUM_GROUPS,
    )


def test_group_constant_shift_invariance() -> None:
    inputs = _inputs()
    group_ids = inputs.attention_group_id
    logits = _raw_logits(
        inputs,
        heads=MULTIHEAD_COUNT,
    )
    shifts = torch.linspace(
        -100.0,
        100.0,
        NUM_GROUPS,
        dtype=inputs.dtype,
    ).unsqueeze(1)
    shifted = (
        logits
        + shifts[group_ids]
    )

    original_weights = (
        normalize_attention_logits(
            logits,
            group_ids,
            num_groups=NUM_GROUPS,
        )
    )
    shifted_weights = (
        normalize_attention_logits(
            shifted,
            group_ids,
            num_groups=NUM_GROUPS,
        )
    )

    torch.testing.assert_close(
        shifted_weights,
        original_weights,
        atol=1e-5,
        rtol=1e-5,
    )


def test_group_locality() -> None:
    inputs = _inputs()
    group_ids = inputs.attention_group_id
    logits = _raw_logits(
        inputs,
        heads=2,
    )
    changed = logits.clone()
    changed[group_ids == 3] += torch.tensor(
        [7.0, -4.0],
        dtype=inputs.dtype,
    )

    original_weights = normalize_attention_logits(
        logits,
        group_ids,
        num_groups=NUM_GROUPS,
    )
    changed_weights = normalize_attention_logits(
        changed,
        group_ids,
        num_groups=NUM_GROUPS,
    )

    outside_group = group_ids != 3
    assert torch.equal(
        original_weights[outside_group],
        changed_weights[outside_group],
    )


def test_edge_order_equivariance() -> None:
    inputs = _inputs()
    group_ids = inputs.attention_group_id
    logits = _raw_logits(
        inputs,
        heads=MULTIHEAD_COUNT,
    )
    permutation = torch.tensor(
        [6, 2, 0, 5, 1, 4, 3],
        dtype=torch.long,
    )

    original = normalize_attention_logits(
        logits,
        group_ids,
        num_groups=NUM_GROUPS,
    )
    permuted = normalize_attention_logits(
        logits[permutation],
        group_ids[permutation],
        num_groups=NUM_GROUPS,
    )

    torch.testing.assert_close(
        permuted,
        original[permutation],
    )


def test_identical_logits_within_group_receive_equal_weights() -> None:
    inputs = _inputs()
    group_ids = inputs.attention_group_id
    logits = torch.zeros(
        (EDGES, 2),
        dtype=inputs.dtype,
    )
    logits[group_ids == 3] = torch.tensor(
        [2.0, -4.0],
        dtype=inputs.dtype,
    )

    weights = normalize_attention_logits(
        logits,
        group_ids,
        num_groups=NUM_GROUPS,
    )

    assert torch.equal(
        weights[group_ids == 3],
        torch.full(
            (2, 2),
            0.5,
            dtype=inputs.dtype,
        ),
    )


def test_output_preserves_shape_dtype_and_device() -> None:
    inputs = _inputs(
        dtype=torch.float64,
    )
    logits = _raw_logits(
        inputs,
        heads=2,
    )

    weights = normalize_attention_logits(
        logits,
        inputs.attention_group_id,
        num_groups=NUM_GROUPS,
    )

    assert weights.shape == logits.shape
    assert weights.dtype == torch.float64
    assert weights.device == logits.device


def test_empty_edge_normalization_with_dense_group_axis() -> None:
    logits = torch.empty(
        (0, 2),
        dtype=torch.float32,
    )
    group_ids = torch.empty(
        (0,),
        dtype=torch.long,
    )

    weights = normalize_attention_logits(
        logits,
        group_ids,
        num_groups=NUM_GROUPS,
    )

    assert weights.shape == (0, 2)
    assert weights.dtype == logits.dtype


def test_empty_edge_normalization_with_zero_groups() -> None:
    logits = torch.empty(
        (0, 1),
        dtype=torch.float32,
    )
    group_ids = torch.empty(
        (0,),
        dtype=torch.long,
    )

    weights = normalize_attention_logits(
        logits,
        group_ids,
        num_groups=0,
    )

    assert weights.shape == (0, 1)


# =============================================================================
# Low-level normalization: autograd
# =============================================================================


def test_gradients_flow_to_logits() -> None:
    inputs = _inputs()
    logits = _raw_logits(
        inputs,
        heads=2,
        requires_grad=True,
    )
    weights = normalize_attention_logits(
        logits,
        inputs.attention_group_id,
        num_groups=NUM_GROUPS,
    )
    coefficients = torch.arange(
        1,
        EDGES * 2 + 1,
        dtype=inputs.dtype,
    ).reshape(EDGES, 2)
    loss = (
        weights
        * coefficients
    ).sum()
    loss.backward()

    assert logits.grad is not None
    assert torch.isfinite(
        logits.grad
    ).all()
    assert float(
        logits.grad.abs().sum().item()
    ) > 0.0


def test_singleton_group_logits_have_zero_softmax_gradient() -> None:
    inputs = _inputs()
    logits = _raw_logits(
        inputs,
        requires_grad=True,
    )
    weights = normalize_attention_logits(
        logits,
        inputs.attention_group_id,
        num_groups=NUM_GROUPS,
    )
    counts = segment_counts(
        inputs.attention_group_id,
        num_segments=NUM_GROUPS,
    )
    singleton_edges = (
        counts[
            inputs.attention_group_id
        ]
        == 1
    )
    repeated_edges = ~singleton_edges

    loss = (
        weights[repeated_edges]
        * torch.arange(
            1,
            int(repeated_edges.sum().item()) + 1,
            dtype=inputs.dtype,
        ).unsqueeze(1)
    ).sum()
    loss.backward()

    assert logits.grad is not None
    assert torch.equal(
        logits.grad[singleton_edges],
        torch.zeros_like(
            logits.grad[singleton_edges]
        ),
    )


def test_group_weight_sum_has_zero_logit_gradient() -> None:
    inputs = _inputs()
    logits = _raw_logits(
        inputs,
        heads=2,
        requires_grad=True,
    )
    weights = normalize_attention_logits(
        logits,
        inputs.attention_group_id,
        num_groups=NUM_GROUPS,
    )

    # Every nonempty group sum is mathematically constant at one.
    total = weights.sum()
    total.backward()

    assert logits.grad is not None
    torch.testing.assert_close(
        logits.grad,
        torch.zeros_like(logits.grad),
        atol=1e-6,
        rtol=0.0,
    )


def test_normalization_does_not_detach_output() -> None:
    inputs = _inputs()
    logits = _raw_logits(
        inputs,
        requires_grad=True,
    )

    weights = normalize_attention_logits(
        logits,
        inputs.attention_group_id,
        num_groups=NUM_GROUPS,
    )

    assert weights.requires_grad
    assert weights.grad_fn is not None


# =============================================================================
# Low-level normalization: invalid contracts
# =============================================================================


def test_normalization_rejects_non_tensor_logits() -> None:
    with pytest.raises(
        TypeError,
        match="logits_by_head must be a tensor",
    ):
        normalize_attention_logits(
            [[0.0]],  # type: ignore[arg-type]
            torch.tensor(
                [0],
                dtype=torch.long,
            ),
            num_groups=1,
        )


@pytest.mark.parametrize(
    "logits",
    (
        torch.zeros(EDGES),
        torch.zeros(1, EDGES, 1),
    ),
)
def test_normalization_rejects_wrong_logit_rank(
    logits: torch.Tensor,
) -> None:
    with pytest.raises(
        ValueError,
        match=r"shape \[E, A\]",
    ):
        normalize_attention_logits(
            logits,
            torch.zeros(
                EDGES,
                dtype=torch.long,
            ),
            num_groups=1,
        )


def test_normalization_rejects_zero_heads() -> None:
    with pytest.raises(
        ValueError,
        match="at least one attention head",
    ):
        normalize_attention_logits(
            torch.empty(
                (EDGES, 0),
            ),
            torch.zeros(
                EDGES,
                dtype=torch.long,
            ),
            num_groups=1,
        )


def test_normalization_rejects_integer_logits() -> None:
    with pytest.raises(
        ValueError,
        match="floating-point",
    ):
        normalize_attention_logits(
            torch.zeros(
                (EDGES, 1),
                dtype=torch.long,
            ),
            torch.zeros(
                EDGES,
                dtype=torch.long,
            ),
            num_groups=1,
        )


@pytest.mark.parametrize(
    "invalid_value",
    (
        float("nan"),
        float("inf"),
        float("-inf"),
    ),
)
def test_normalization_rejects_nonfinite_logits(
    invalid_value: float,
) -> None:
    logits = torch.zeros(
        (EDGES, 1),
    )
    logits[0, 0] = invalid_value

    with pytest.raises(
        ValueError,
        match="finite",
    ):
        normalize_attention_logits(
            logits,
            torch.zeros(
                EDGES,
                dtype=torch.long,
            ),
            num_groups=1,
        )


def test_normalization_rejects_non_tensor_group_ids() -> None:
    with pytest.raises(
        TypeError,
        match="group_ids must be a tensor",
    ):
        normalize_attention_logits(
            torch.zeros(
                (EDGES, 1),
            ),
            [0] * EDGES,  # type: ignore[arg-type]
            num_groups=1,
        )


def test_normalization_rejects_wrong_group_id_rank() -> None:
    with pytest.raises(
        ValueError,
        match=r"shape \[E\]",
    ):
        normalize_attention_logits(
            torch.zeros(
                (EDGES, 1),
            ),
            torch.zeros(
                (1, EDGES),
                dtype=torch.long,
            ),
            num_groups=1,
        )


def test_normalization_rejects_wrong_group_id_length() -> None:
    with pytest.raises(
        ValueError,
        match="align with the edge axis",
    ):
        normalize_attention_logits(
            torch.zeros(
                (EDGES, 1),
            ),
            torch.zeros(
                EDGES - 1,
                dtype=torch.long,
            ),
            num_groups=1,
        )


def test_normalization_rejects_non_long_group_ids() -> None:
    with pytest.raises(
        ValueError,
        match="torch.long",
    ):
        normalize_attention_logits(
            torch.zeros(
                (EDGES, 1),
            ),
            torch.zeros(
                EDGES,
                dtype=torch.int32,
            ),
            num_groups=1,
        )


@pytest.mark.parametrize(
    "group_ids",
    (
        torch.tensor(
            [-1, 0, 0, 0, 0, 0, 0],
            dtype=torch.long,
        ),
        torch.tensor(
            [1, 0, 0, 0, 0, 0, 0],
            dtype=torch.long,
        ),
    ),
)
def test_normalization_rejects_out_of_range_group_ids(
    group_ids: torch.Tensor,
) -> None:
    with pytest.raises(
        ValueError,
        match="segment_ids|range",
    ):
        normalize_attention_logits(
            torch.zeros(
                (EDGES, 1),
            ),
            group_ids,
            num_groups=1,
        )


@pytest.mark.parametrize(
    "num_groups",
    (
        -1,
        True,
        1.5,
    ),
)
def test_normalization_rejects_invalid_num_groups(
    num_groups: object,
) -> None:
    expected_error = (
        TypeError
        if isinstance(
            num_groups,
            (bool, float),
        )
        else ValueError
    )

    with pytest.raises(expected_error):
        normalize_attention_logits(
            torch.zeros(
                (0, 1),
            ),
            torch.empty(
                (0,),
                dtype=torch.long,
            ),
            num_groups=num_groups,  # type: ignore[arg-type]
        )


def test_nonempty_items_reject_zero_num_groups() -> None:
    with pytest.raises(
        ValueError,
        match="segment_ids|num_segments|range",
    ):
        normalize_attention_logits(
            torch.zeros(
                (1, 1),
            ),
            torch.zeros(
                1,
                dtype=torch.long,
            ),
            num_groups=0,
        )


# =============================================================================
# Diagnostic helpers
# =============================================================================


def test_attention_group_sums_shape_and_values() -> None:
    inputs = _inputs()
    logits = _raw_logits(
        inputs,
        heads=2,
    )
    weights = normalize_attention_logits(
        logits,
        inputs.attention_group_id,
        num_groups=NUM_GROUPS,
    )

    sums = attention_group_sums(
        weights,
        inputs.attention_group_id,
        num_groups=NUM_GROUPS,
    )
    counts = segment_counts(
        inputs.attention_group_id,
        num_segments=NUM_GROUPS,
    )
    present = counts > 0

    assert sums.shape == (
        NUM_GROUPS,
        2,
    )
    torch.testing.assert_close(
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


def test_group_sums_does_not_require_already_normalized_values() -> None:
    inputs = _inputs()
    candidate = torch.full(
        (EDGES, 2),
        2.0,
        dtype=inputs.dtype,
    )

    sums = attention_group_sums(
        candidate,
        inputs.attention_group_id,
        num_groups=NUM_GROUPS,
    )

    assert sums[3, 0].item() == 4.0
    assert sums[7, 0].item() == 2.0
    assert sums[14, 1].item() == 4.0


def test_maximum_normalization_error_is_zero_for_valid_weights() -> None:
    inputs = _inputs()
    weights = normalize_attention_logits(
        _raw_logits(
            inputs,
            heads=2,
        ),
        inputs.attention_group_id,
        num_groups=NUM_GROUPS,
    )

    assert (
        maximum_attention_normalization_error(
            weights,
            inputs.attention_group_id,
            num_groups=NUM_GROUPS,
        )
        <= 1e-6
    )


def test_maximum_normalization_error_detects_bad_group_sum() -> None:
    inputs = _inputs()
    weights = normalize_attention_logits(
        _raw_logits(inputs),
        inputs.attention_group_id,
        num_groups=NUM_GROUPS,
    )
    bad = weights.clone()
    bad[0, 0] += 0.25

    observed = (
        maximum_attention_normalization_error(
            bad,
            inputs.attention_group_id,
            num_groups=NUM_GROUPS,
        )
    )

    assert observed == pytest.approx(
        0.25,
        abs=1e-6,
    )


def test_maximum_normalization_error_empty_groups_only() -> None:
    weights = torch.empty(
        (0, 2),
    )
    group_ids = torch.empty(
        (0,),
        dtype=torch.long,
    )

    assert (
        maximum_attention_normalization_error(
            weights,
            group_ids,
            num_groups=5,
        )
        == 0.0
    )


def test_assert_attention_normalized_accepts_valid_weights() -> None:
    inputs = _inputs()
    weights = normalize_attention_logits(
        _raw_logits(inputs),
        inputs.attention_group_id,
        num_groups=NUM_GROUPS,
    )

    assert_attention_normalized(
        weights,
        inputs.attention_group_id,
        num_groups=NUM_GROUPS,
    )


def test_assert_attention_normalized_rejects_bad_weights() -> None:
    inputs = _inputs()
    weights = normalize_attention_logits(
        _raw_logits(inputs),
        inputs.attention_group_id,
        num_groups=NUM_GROUPS,
    )
    weights[0, 0] += 0.1

    with pytest.raises(
        ValueError,
        match="sum to one",
    ):
        assert_attention_normalized(
            weights,
            inputs.attention_group_id,
            num_groups=NUM_GROUPS,
        )


def test_assert_attention_normalized_respects_custom_tolerance() -> None:
    inputs = _inputs()
    weights = normalize_attention_logits(
        _raw_logits(inputs),
        inputs.attention_group_id,
        num_groups=NUM_GROUPS,
    )
    weights[0, 0] += 1e-4

    assert_attention_normalized(
        weights,
        inputs.attention_group_id,
        num_groups=NUM_GROUPS,
        atol=1e-3,
        rtol=0.0,
    )

    with pytest.raises(ValueError):
        assert_attention_normalized(
            weights,
            inputs.attention_group_id,
            num_groups=NUM_GROUPS,
            atol=1e-7,
            rtol=0.0,
        )


@pytest.mark.parametrize(
    "helper",
    (
        attention_group_sums,
        maximum_attention_normalization_error,
        assert_attention_normalized,
    ),
)
def test_diagnostic_helpers_reject_non_tensor_values(
    helper: Any,
) -> None:
    with pytest.raises(
        TypeError,
        match="tensor",
    ):
        helper(
            [[1.0]],  # type: ignore[arg-type]
            torch.tensor(
                [0],
                dtype=torch.long,
            ),
            num_groups=1,
        )


# =============================================================================
# Functional metadata-preserving normalization
# =============================================================================


@pytest.mark.parametrize(
    (
        "score_factory",
        "expected_heads",
    ),
    (
        (
            lambda inputs: _score_output(
                inputs,
                mode=ATTENTION_MODE_UNIFORM,
            ),
            1,
        ),
        (
            lambda inputs: _score_output(
                inputs,
                mode=(
                    ATTENTION_MODE_HAZARD_CONDITIONED
                ),
            ),
            1,
        ),
        (
            lambda inputs: _multihead_score(
                inputs
            ),
            MULTIHEAD_COUNT,
        ),
    ),
)
def test_functional_normalization_output_contract(
    score_factory: Any,
    expected_heads: int,
) -> None:
    inputs = _inputs()
    score = score_factory(inputs)
    output = normalize_edge_attention_scores(
        score
    )

    assert isinstance(
        output,
        AttentionNormalizationOutput,
    )
    assert output.source_score_output is score
    assert output.source_inputs is inputs
    assert output.normalization_mode == (
        ATTENTION_NORMALIZATION_TARGET_NODE_RELATION
    )
    assert output.normalized_weights_by_head.shape == (
        EDGES,
        expected_heads,
    )
    assert torch.equal(
        output.group_ids,
        inputs.attention_group_id,
    )
    assert torch.equal(
        output.group_counts,
        segment_counts(
            inputs.attention_group_id,
            num_segments=NUM_GROUPS,
        ),
    )
    assert output.parameter_fingerprint is None
    assert output.encoder_architecture_fingerprint


def test_functional_normalization_matches_low_level_result() -> None:
    inputs = _inputs()
    score = _multihead_score(
        inputs
    )

    output = normalize_edge_attention_scores(
        score
    )
    expected = normalize_attention_logits(
        score.raw_scores_by_head,
        inputs.attention_group_id,
        num_groups=NUM_GROUPS,
    )

    torch.testing.assert_close(
        output.normalized_weights_by_head,
        expected,
    )


def test_functional_uniform_attention_is_reciprocal_group_size() -> None:
    inputs = _inputs()
    score = _score_output(
        inputs,
        mode=ATTENTION_MODE_UNIFORM,
    )
    output = normalize_edge_attention_scores(
        score
    )
    counts = output.group_counts[
        output.group_ids
    ].to(dtype=output.dtype)
    expected = (
        torch.ones_like(
            output.normalized_weights_by_head
        )
        / counts.unsqueeze(1)
    )

    assert torch.equal(
        output.normalized_weights_by_head,
        expected,
    )


def test_functional_normalization_custom_architecture_fingerprint() -> None:
    output = normalize_edge_attention_scores(
        _score_output(_inputs()),
        encoder_architecture_fingerprint=(
            "custom-normalizer-architecture"
        ),
    )

    assert (
        output.encoder_architecture_fingerprint
        == "custom-normalizer-architecture"
    )


def test_functional_normalization_rejects_blank_custom_fingerprint() -> None:
    with pytest.raises(
        ValueError,
        match="non-empty",
    ):
        normalize_edge_attention_scores(
            _score_output(_inputs()),
            encoder_architecture_fingerprint="",
        )


def test_functional_normalization_default_fingerprint_is_deterministic() -> None:
    inputs = _inputs()
    first = normalize_edge_attention_scores(
        _score_output(inputs)
    )
    second = normalize_edge_attention_scores(
        _score_output(
            inputs,
            logits=_raw_logits(inputs),
        )
    )

    assert (
        first.encoder_architecture_fingerprint
        == second.encoder_architecture_fingerprint
    )


def test_functional_normalization_empty_edges() -> None:
    inputs = _inputs(
        empty_edges=True,
    )
    score = _multihead_score(
        inputs
    )
    output = normalize_edge_attention_scores(
        score
    )

    assert output.normalized_weights_by_head.shape == (
        0,
        MULTIHEAD_COUNT,
    )
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


def test_functional_normalization_preserves_autograd() -> None:
    inputs = _inputs()
    logits = _raw_logits(
        inputs,
        heads=MULTIHEAD_COUNT,
        requires_grad=True,
    )
    score = _multihead_score(
        inputs,
        logits=logits,
    )
    output = normalize_edge_attention_scores(
        score
    )

    loss = (
        output.normalized_weights_by_head
        * torch.arange(
            1,
            EDGES * MULTIHEAD_COUNT + 1,
            dtype=inputs.dtype,
        ).reshape(
            EDGES,
            MULTIHEAD_COUNT,
        )
    ).sum()
    loss.backward()

    assert logits.grad is not None
    assert torch.isfinite(
        logits.grad
    ).all()


def test_functional_normalization_rejects_wrong_score_type() -> None:
    with pytest.raises(
        TypeError,
        match="EdgeAttentionScoreOutput",
    ):
        normalize_edge_attention_scores(
            object()  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "mode",
    (
        ATTENTION_NORMALIZATION_TARGET_NODE,
        ATTENTION_NORMALIZATION_GLOBAL_RELATION,
        ATTENTION_NORMALIZATION_UNNORMALIZED_SIGMOID,
    ),
)
def test_functional_normalization_rejects_unimplemented_modes(
    mode: str,
) -> None:
    with pytest.raises(
        NotImplementedError,
        match="not implemented",
    ):
        normalize_edge_attention_scores(
            _score_output(_inputs()),
            normalization_mode=mode,
        )


def test_functional_normalization_rejects_unknown_mode() -> None:
    with pytest.raises(
        ValueError,
        match="Unknown attention normalization mode",
    ):
        normalize_edge_attention_scores(
            _score_output(_inputs()),
            normalization_mode="unknown-mode",
        )


def test_functional_normalization_rejects_non_string_mode() -> None:
    with pytest.raises(
        TypeError,
        match="must be a string",
    ):
        normalize_edge_attention_scores(
            _score_output(_inputs()),
            normalization_mode=3,  # type: ignore[arg-type]
        )


# =============================================================================
# Parameter-free module identity
# =============================================================================


def test_module_identity_and_parameter_free_contract() -> None:
    module = (
        TargetNodeRelationAttentionNormalization()
    )

    assert module.normalization_mode == (
        ATTENTION_NORMALIZATION_TARGET_NODE_RELATION
    )
    assert module.group_key == (
        ATTENTION_GROUP_KEY_TARGET_NODE_EXACT_RELATION
    )
    assert module.group_id_formula == (
        ATTENTION_GROUP_ID_FORMULA
    )
    assert module.parameter_count == 0
    assert module.trainable_parameter_count == 0
    assert module.parameter_fingerprint() is None
    assert tuple(module.parameters()) == ()
    assert tuple(module.buffers()) == ()
    assert module.state_dict() == {}
    module.assert_parameter_free()


def test_module_architecture_metadata_is_scientifically_explicit() -> None:
    module = (
        TargetNodeRelationAttentionNormalization()
    )
    architecture = (
        module.architecture_dict()
    )

    assert architecture["normalization_mode"] == (
        ATTENTION_NORMALIZATION_TARGET_NODE_RELATION
    )
    assert architecture["group_key"] == (
        ATTENTION_GROUP_KEY_TARGET_NODE_EXACT_RELATION
    )
    assert architecture["group_id_formula"] == (
        ATTENTION_GROUP_ID_FORMULA
    )
    assert architecture["relation_identity"] == (
        "exact_compiled_relation_index"
    )
    assert (
        architecture[
            "stable_relation_ids_used_as_indices"
        ]
        is False
    )
    assert (
        architecture["heads_normalized_independently"]
        is True
    )
    assert architecture["nonempty_group_sum"] == 1.0
    assert architecture["singleton_group_weight"] == 1.0
    assert architecture["absent_group_count"] == 0
    assert (
        architecture["attention_disabled_handled_here"]
        is False
    )
    assert architecture["edge_masking_owned_here"] is False
    assert architecture["relation_gate_owned_here"] is False
    assert architecture["head_reduction_owned_here"] is False
    assert architecture["aggregation_owned_here"] is False
    assert architecture["parameter_count"] == 0


def test_module_architecture_fingerprint_is_deterministic() -> None:
    first = (
        TargetNodeRelationAttentionNormalization()
    )
    second = (
        TargetNodeRelationAttentionNormalization()
    )

    assert (
        first.architecture_fingerprint()
        == second.architecture_fingerprint()
    )


def test_module_extra_repr_is_informative() -> None:
    text = repr(
        TargetNodeRelationAttentionNormalization()
    )

    assert (
        ATTENTION_NORMALIZATION_TARGET_NODE_RELATION
        in text
    )
    assert (
        ATTENTION_GROUP_KEY_TARGET_NODE_EXACT_RELATION
        in text
    )
    assert "parameter_free=True" in text


def test_module_group_helpers_match_free_functions() -> None:
    inputs = _inputs()
    module = (
        TargetNodeRelationAttentionNormalization()
    )
    group_ids = module.group_ids(inputs)
    counts = module.group_counts(
        inputs,
        group_ids=group_ids,
    )

    assert torch.equal(
        group_ids,
        build_target_node_relation_group_ids(
            inputs
        ),
    )
    assert torch.equal(
        counts,
        build_target_node_relation_group_counts(
            inputs
        ),
    )


def test_module_normalize_logits_matches_free_function() -> None:
    inputs = _inputs()
    logits = _raw_logits(
        inputs,
        heads=2,
    )
    group_ids = inputs.attention_group_id
    module = (
        TargetNodeRelationAttentionNormalization()
    )

    assert torch.equal(
        module.normalize_logits(
            logits,
            group_ids,
            num_groups=NUM_GROUPS,
        ),
        normalize_attention_logits(
            logits,
            group_ids,
            num_groups=NUM_GROUPS,
        ),
    )


def test_module_forward_contract() -> None:
    inputs = _inputs()
    score = _multihead_score(inputs)
    module = (
        TargetNodeRelationAttentionNormalization()
    )
    output = module(score)

    assert isinstance(
        output,
        AttentionNormalizationOutput,
    )
    assert output.source_score_output is score
    assert output.source_inputs is inputs
    assert output.normalization_mode == (
        module.normalization_mode
    )
    assert (
        output.encoder_architecture_fingerprint
        == module.architecture_fingerprint()
    )
    assert output.parameter_fingerprint is None
    assert output.normalized_weights_by_head.shape == (
        EDGES,
        MULTIHEAD_COUNT,
    )


def test_module_forward_matches_functional_wrapper() -> None:
    score = _multihead_score(
        _inputs()
    )
    module = (
        TargetNodeRelationAttentionNormalization()
    )

    class_output = module(score)
    functional_output = (
        normalize_edge_attention_scores(
            score,
            encoder_architecture_fingerprint=(
                module.architecture_fingerprint()
            ),
        )
    )

    assert torch.equal(
        class_output.normalized_weights_by_head,
        functional_output.normalized_weights_by_head,
    )
    assert torch.equal(
        class_output.group_ids,
        functional_output.group_ids,
    )
    assert torch.equal(
        class_output.group_counts,
        functional_output.group_counts,
    )
    assert (
        class_output.encoder_architecture_fingerprint
        == functional_output.encoder_architecture_fingerprint
    )


def test_module_forward_rejects_wrong_score_type() -> None:
    with pytest.raises(
        TypeError,
        match="EdgeAttentionScoreOutput",
    ):
        TargetNodeRelationAttentionNormalization()(
            object()  # type: ignore[arg-type]
        )


def test_module_forward_empty_edges() -> None:
    inputs = _inputs(
        empty_edges=True,
    )
    output = (
        TargetNodeRelationAttentionNormalization()(
            _multihead_score(inputs)
        )
    )

    assert output.normalized_weights_by_head.shape == (
        0,
        MULTIHEAD_COUNT,
    )
    assert output.group_counts.shape == (
        NUM_GROUPS,
    )


def test_module_rejects_unimplemented_mode() -> None:
    with pytest.raises(
        NotImplementedError,
        match="not implemented",
    ):
        TargetNodeRelationAttentionNormalization(
            normalization_mode=(
                ATTENTION_NORMALIZATION_TARGET_NODE
            )
        )


def test_module_rejects_unknown_mode() -> None:
    with pytest.raises(
        ValueError,
        match="Unknown attention normalization mode",
    ):
        TargetNodeRelationAttentionNormalization(
            normalization_mode="unknown",
        )


def test_module_rejects_non_string_mode() -> None:
    with pytest.raises(
        TypeError,
        match="must be a string",
    ):
        TargetNodeRelationAttentionNormalization(
            normalization_mode=1,  # type: ignore[arg-type]
        )


def test_assert_parameter_free_detects_injected_parameter() -> None:
    module = (
        TargetNodeRelationAttentionNormalization()
    )
    module.register_parameter(
        "unexpected_parameter",
        nn.Parameter(torch.ones(1)),
    )

    with pytest.raises(
        RuntimeError,
        match="parameter-free",
    ):
        module.assert_parameter_free()


def test_assert_parameter_free_detects_injected_buffer() -> None:
    module = (
        TargetNodeRelationAttentionNormalization()
    )
    module.register_buffer(
        "unexpected_buffer",
        torch.ones(1),
    )

    with pytest.raises(
        RuntimeError,
        match="data-dependent buffers",
    ):
        module.assert_parameter_free()


def test_assert_parameter_free_detects_nonempty_state_dict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = (
        TargetNodeRelationAttentionNormalization()
    )
    monkeypatch.setattr(
        module,
        "state_dict",
        lambda: {"unexpected": torch.ones(1)},
    )

    with pytest.raises(
        RuntimeError,
        match="empty state_dict",
    ):
        module.assert_parameter_free()


# =============================================================================
# Configuration constructors and dispatcher
# =============================================================================


def test_from_config_builds_normalizer() -> None:
    module = (
        TargetNodeRelationAttentionNormalization
        .from_config(
            config=_config(),
        )
    )

    assert isinstance(
        module,
        TargetNodeRelationAttentionNormalization,
    )
    assert module.normalization_mode == (
        ATTENTION_NORMALIZATION_TARGET_NODE_RELATION
    )


def test_dispatcher_builds_normalizer() -> None:
    module = build_attention_normalizer(
        config=_config(),
    )

    assert isinstance(
        module,
        TargetNodeRelationAttentionNormalization,
    )


def test_from_config_rejects_wrong_config_type() -> None:
    with pytest.raises(
        TypeError,
        match="FunctionalMessagePassingConfig",
    ):
        (
            TargetNodeRelationAttentionNormalization
            .from_config(
                config=object(),  # type: ignore[arg-type]
            )
        )


def test_dispatcher_rejects_wrong_config_type() -> None:
    with pytest.raises(
        TypeError,
        match="FunctionalMessagePassingConfig",
    ):
        build_attention_normalizer(
            config=object(),  # type: ignore[arg-type]
        )


def test_from_config_rejects_unimplemented_canonical_mode() -> None:
    with pytest.raises(
        NotImplementedError,
        match="not implemented",
    ):
        (
            TargetNodeRelationAttentionNormalization
            .from_config(
                config=_config(
                    normalization=(
                        ATTENTION_NORMALIZATION_TARGET_NODE
                    ),
                )
            )
        )


def test_dispatcher_rejects_unimplemented_canonical_mode() -> None:
    with pytest.raises(
        NotImplementedError,
        match="not implemented",
    ):
        build_attention_normalizer(
            config=_config(
                normalization=(
                    ATTENTION_NORMALIZATION_TARGET_NODE
                ),
            )
        )


# =============================================================================
# End-to-end metamorphic properties through module wrapper
# =============================================================================


def test_module_group_constant_shift_invariance() -> None:
    inputs = _inputs()
    group_ids = inputs.attention_group_id
    logits = _raw_logits(
        inputs,
        heads=MULTIHEAD_COUNT,
    )
    shifts = torch.randn(
        NUM_GROUPS,
        MULTIHEAD_COUNT,
        dtype=inputs.dtype,
    )
    shifted = (
        logits
        + shifts[group_ids]
    )
    module = (
        TargetNodeRelationAttentionNormalization()
    )

    original = module(
        _multihead_score(
            inputs,
            logits=logits,
        )
    )
    changed = module(
        _multihead_score(
            inputs,
            logits=shifted,
        )
    )

    torch.testing.assert_close(
        changed.normalized_weights_by_head,
        original.normalized_weights_by_head,
    )


def test_module_edge_order_equivariance() -> None:
    permutation = torch.tensor(
        [6, 2, 0, 5, 1, 4, 3],
        dtype=torch.long,
    )
    original_inputs = _inputs()
    original_logits = _raw_logits(
        original_inputs,
        heads=MULTIHEAD_COUNT,
    )
    permuted_inputs = _inputs(
        permutation=permutation,
    )
    permuted_logits = (
        original_logits[permutation]
    )
    module = (
        TargetNodeRelationAttentionNormalization()
    )

    original = module(
        _multihead_score(
            original_inputs,
            logits=original_logits,
        )
    )
    permuted = module(
        _multihead_score(
            permuted_inputs,
            logits=permuted_logits,
        )
    )

    torch.testing.assert_close(
        permuted.normalized_weights_by_head,
        original.normalized_weights_by_head[
            permutation
        ],
    )
    assert torch.equal(
        permuted.group_ids,
        original.group_ids[permutation],
    )
    assert torch.equal(
        permuted.group_counts,
        original.group_counts,
    )


def test_module_control_relation_normalization() -> None:
    inputs = _inputs()
    output = (
        TargetNodeRelationAttentionNormalization()(
            _score_output(inputs)
        )
    )
    control_edges = inputs.control_edge_mask

    assert bool(control_edges.any().item())
    assert torch.isfinite(
        output.normalized_weights_by_head[
            control_edges
        ]
    ).all()
    assert bool(
        (
            output.normalized_weights_by_head[
                control_edges
            ]
            >= 0
        ).all().item()
    )


def test_uniform_attention_is_not_disabled_identity() -> None:
    inputs = _inputs()
    output = (
        TargetNodeRelationAttentionNormalization()(
            _score_output(
                inputs,
                mode=ATTENTION_MODE_UNIFORM,
            )
        )
    )
    repeated_edges = (
        output.group_counts[
            output.group_ids
        ]
        > 1
    )

    assert bool(repeated_edges.any().item())
    assert bool(
        (
            output.normalized_weights_by_head[
                repeated_edges
            ]
            < 1.0
        ).all().item()
    )


def test_normalization_owns_no_causal_importance_field() -> None:
    output = (
        TargetNodeRelationAttentionNormalization()(
            _score_output(_inputs())
        )
    )

    assert not hasattr(
        output,
        "causal_importance",
    )
    assert not hasattr(
        output,
        "relation_gate",
    )
    assert not hasattr(
        output,
        "edge_message",
    )


# =============================================================================
# Optional CUDA contracts
# =============================================================================


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is not available.",
)
def test_low_level_cuda_normalization_and_gradients() -> None:
    inputs = _inputs(
        device="cuda",
    )
    logits = _raw_logits(
        inputs,
        heads=2,
        requires_grad=True,
    )

    weights = normalize_attention_logits(
        logits,
        inputs.attention_group_id,
        num_groups=NUM_GROUPS,
    )
    loss = (
        weights
        * torch.arange(
            1,
            EDGES * 2 + 1,
            device="cuda",
            dtype=inputs.dtype,
        ).reshape(EDGES, 2)
    ).sum()
    loss.backward()

    assert weights.device.type == "cuda"
    assert logits.grad is not None
    assert logits.grad.device.type == "cuda"


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is not available.",
)
def test_module_cuda_forward() -> None:
    inputs = _inputs(
        device="cuda",
    )
    score = _multihead_score(inputs)
    module = (
        TargetNodeRelationAttentionNormalization()
        .to(device="cuda")
    )
    output = module(score)

    assert output.device.type == "cuda"
    assert output.group_ids.device.type == "cuda"
    assert output.group_counts.device.type == "cuda"


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is not available.",
)
def test_low_level_rejects_cpu_group_ids_for_cuda_logits() -> None:
    inputs = _inputs(
        device="cuda",
    )
    logits = _raw_logits(inputs)
    cpu_group_ids = (
        inputs.attention_group_id.cpu()
    )

    with pytest.raises(
        ValueError,
        match="share one device",
    ):
        normalize_attention_logits(
            logits,
            cpu_group_ids,
            num_groups=NUM_GROUPS,
        )
