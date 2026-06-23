"""
Focused tests for attention-head reduction.

Target repository path:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            tests/
                test_edge_attention_multihead.py

Implementation under test:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            functional_message_passing/
                edge_attention/
                    multihead.py

The suite isolates the head-reduction stage from edge-score prediction,
target-node/relation grouped normalization, relation gating, relation
transforms, message construction, aggregation, and explanation exporters.

Contracts frozen here
--------------------------------
- The bounded reduction is the arithmetic mean over heads.
- Every head enters with equal weight ``1 / A``.
- A single head reduces exactly to itself.
- Head-axis permutations cannot change the reduced coefficient.
- Edge order is preserved.
- Nonnegativity is preserved.
- Independently group-normalized heads remain group-normalized after mean
  reduction.
- Empty edge sets are valid.
- Dtype, device, and autograd connectivity are preserved.
- The reduction stage is parameter-free and buffer-free.
- Runtime head count must match the architecture frozen in the module.
- Canonical alternatives such as weighted mean, max, and no reduction are
  rejected explicitly because they require different downstream contracts.
- Head-disagreement statistics are descriptive diagnostics only; they are not
  calibrated uncertainty, proof of specialization, or causal importance.
- The final reduced coefficient remains an attention-routing coefficient and
  is not a relation gate, semantic edge weight, structural normalizer, or
  aggregation denominator.

Controlled upstream doubles are patched into the existing
``functional_message_passing.schemas`` module so failures remain local to the
multihead reduction boundary rather than graph loading, fusion, hazard
encoding, or registry construction.
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
    ATTENTION_HEAD_REDUCTION_MAX,
    ATTENTION_HEAD_REDUCTION_MEAN,
    ATTENTION_HEAD_REDUCTION_NONE,
    ATTENTION_HEAD_REDUCTION_WEIGHTED_MEAN,
    ATTENTION_MODE_HAZARD_CONDITIONED,
    ATTENTION_MODE_MULTIHEAD_HAZARD_CONDITIONED,
    ATTENTION_NORMALIZATION_TARGET_NODE_RELATION,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.functional_message_passing import (
    schemas as fmp_schemas,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.functional_message_passing.edge_attention.multihead import (
    ATTENTION_HEAD_MEAN_FORMULA,
    ATTENTION_HEAD_REDUCTION_INTERPRETATION,
    ATTENTION_MULTIHEAD_SCHEMA_VERSION,
    IMPLEMENTED_ATTENTION_HEAD_REDUCTIONS,
    AttentionHeadReducer,
    AttentionHeadReduction,
    EdgeAttentionHeadReducer,
    MeanAttentionHeadReduction,
    MultiheadAttentionReduction,
    apply_attention_head_reduction,
    assert_reduced_attention_normalized,
    attention_head_mean,
    attention_head_mean_absolute_deviation,
    attention_head_range,
    attention_head_standard_deviation,
    attention_head_variance,
    build_attention_head_reducer,
    build_edge_attention_head_reducer,
    head_disagreement_summary,
    maximum_attention_head_range,
    maximum_reduced_attention_normalization_error,
    mean_attention_head_standard_deviation,
    mean_reduce_attention_heads,
    reduce_attention_heads,
    reduce_normalized_attention_heads,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.functional_message_passing.edge_attention.schemas import (
    EDGE_ATTENTION_INPUT_EXACT_RELATION_EMBEDDING,
    EDGE_ATTENTION_INPUT_SOURCE_NODE_STATE,
    EDGE_ATTENTION_INPUT_TARGET_HAZARD_QUERY,
    EDGE_ATTENTION_INPUT_TARGET_NODE_STATE,
    EDGE_ATTENTION_SCORE_FUNCTION_ADDITIVE,
    AttentionHeadReductionOutput,
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
# Controlled graph and attention helpers
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
        edge_attributes = torch.empty(
            (0, 2),
            dtype=dtype,
            device=device,
        )
        semantic_weight = torch.empty(
            (0,),
            dtype=dtype,
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
        edge_attributes = torch.arange(
            EDGES * 2,
            dtype=dtype,
            device=device,
        ).reshape(EDGES, 2)
        semantic_weight = torch.linspace(
            0.5,
            1.5,
            EDGES,
            dtype=dtype,
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
            edge_attributes = edge_attributes[
                permutation
            ]
            semantic_weight = semantic_weight[
                permutation
            ]

    return FakeUrbanGraphBatch(
        external_node_ids=_node_ids(),
        node_batch_index=node_batch,
        edge_index=edge_index,
        edge_relation_type=relations,
        edge_attributes=edge_attributes,
        semantic_edge_weight=semantic_weight,
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
        compiled_relation_registry=(
            FakeCompiledRelationRegistry()
        ),
        hazard_query=_hazard_query(
            dtype=dtype,
            device=device,
        ),
        source_fingerprint="multihead-test-input",
    )


def _raw_logits(
    inputs: FunctionalMessagePassingInputs,
    *,
    heads: int,
    requires_grad: bool = False,
) -> torch.Tensor:
    if inputs.num_edges == 0:
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
            .reshape(
                inputs.num_edges,
                heads,
            )
            / 4.0
        )

    logits = logits.detach().clone()

    if requires_grad:
        logits.requires_grad_(True)

    return logits


def _score_output(
    inputs: FunctionalMessagePassingInputs,
    *,
    heads: int,
    logits: torch.Tensor | None = None,
) -> EdgeAttentionScoreOutput:
    mode = (
        ATTENTION_MODE_HAZARD_CONDITIONED
        if heads == 1
        else ATTENTION_MODE_MULTIHEAD_HAZARD_CONDITIONED
    )
    resolved_logits = (
        _raw_logits(
            inputs,
            heads=heads,
        )
        if logits is None
        else logits
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
        score_function=(
            EDGE_ATTENTION_SCORE_FUNCTION_ADDITIVE
        ),
        input_feature_names=(
            HAZARD_CONDITIONED_FEATURES
        ),
        encoder_architecture_fingerprint=(
            "score-architecture"
        ),
        parameter_fingerprint=(
            "score-parameters"
        ),
    )


def _normalization_output(
    inputs: FunctionalMessagePassingInputs,
    *,
    heads: int,
    logits: torch.Tensor | None = None,
    normalized_weights: torch.Tensor | None = None,
) -> AttentionNormalizationOutput:
    score = _score_output(
        inputs,
        heads=heads,
        logits=logits,
    )
    group_ids = inputs.attention_group_id
    counts = segment_counts(
        group_ids,
        num_segments=(
            inputs.attention_num_groups
        ),
    )
    weights = (
        grouped_softmax(
            score.raw_scores_by_head,
            group_ids,
            num_segments=(
                inputs.attention_num_groups
            ),
        )
        if normalized_weights is None
        else normalized_weights
    )

    return AttentionNormalizationOutput(
        normalized_weights_by_head=weights,
        group_ids=group_ids,
        group_counts=counts,
        source_score_output=score,
        normalization_mode=(
            ATTENTION_NORMALIZATION_TARGET_NODE_RELATION
        ),
        encoder_architecture_fingerprint=(
            "normalization-architecture"
        ),
        parameter_fingerprint=None,
    )


def _config(
    *,
    heads: int,
    reduction: str = (
        ATTENTION_HEAD_REDUCTION_MEAN
    ),
) -> FunctionalMessagePassingConfig:
    mode = (
        ATTENTION_MODE_HAZARD_CONDITIONED
        if heads == 1
        else ATTENTION_MODE_MULTIHEAD_HAZARD_CONDITIONED
    )

    return FunctionalMessagePassingConfig(
        enabled=True,
        attention_enabled=True,
        attention_mode=mode,
        attention_heads=heads,
        attention_head_reduction=(
            reduction
        ),
    )


def _manual_mean(
    values: torch.Tensor,
) -> torch.Tensor:
    return values.sum(dim=1) / float(
        values.shape[1]
    )


def _arbitrary_nonnegative_weights(
    *,
    edges: int = EDGES,
    heads: int = MULTIHEAD_COUNT,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
    requires_grad: bool = False,
) -> torch.Tensor:
    values = (
        torch.arange(
            1,
            edges * heads + 1,
            dtype=dtype,
            device=device,
        )
        .reshape(edges, heads)
        / 10.0
    )
    values = values.detach().clone()

    if requires_grad:
        values.requires_grad_(True)

    return values


# =============================================================================
# Public identity and aliases
# =============================================================================


def test_public_identity_constants() -> None:
    assert isinstance(
        ATTENTION_MULTIHEAD_SCHEMA_VERSION,
        str,
    )
    assert ATTENTION_MULTIHEAD_SCHEMA_VERSION.strip()
    assert IMPLEMENTED_ATTENTION_HEAD_REDUCTIONS == (
        ATTENTION_HEAD_REDUCTION_MEAN,
    )
    assert ATTENTION_HEAD_MEAN_FORMULA == (
        "edge_weights = normalized_weights_by_head.mean(dim=1)"
    )
    assert ATTENTION_HEAD_REDUCTION_INTERPRETATION == (
        "equal_weight_ensemble_of_independently_normalized_attention_heads"
    )


def test_class_aliases_are_exact() -> None:
    assert AttentionHeadReduction is (
        MeanAttentionHeadReduction
    )
    assert MultiheadAttentionReduction is (
        MeanAttentionHeadReduction
    )
    assert EdgeAttentionHeadReducer is (
        MeanAttentionHeadReduction
    )
    reducer: AttentionHeadReducer = (
        MeanAttentionHeadReduction(
            num_heads=1,
        )
    )
    assert isinstance(
        reducer,
        MeanAttentionHeadReduction,
    )


def test_function_and_builder_aliases_are_exact() -> None:
    assert apply_attention_head_reduction is (
        reduce_normalized_attention_heads
    )
    assert build_edge_attention_head_reducer is (
        build_attention_head_reducer
    )


# =============================================================================
# Low-level arithmetic mean
# =============================================================================


def test_mean_reduce_matches_manual_arithmetic_mean() -> None:
    values = _arbitrary_nonnegative_weights()

    observed = mean_reduce_attention_heads(
        values
    )
    expected = _manual_mean(values)

    torch.testing.assert_close(
        observed,
        expected,
    )


def test_single_head_reduction_is_exact_identity() -> None:
    values = _arbitrary_nonnegative_weights(
        heads=1,
    )

    observed = mean_reduce_attention_heads(
        values
    )

    assert torch.equal(
        observed,
        values[:, 0],
    )


def test_two_head_reduction() -> None:
    values = torch.tensor(
        [
            [0.2, 0.8],
            [0.0, 1.0],
            [0.4, 0.4],
        ],
        dtype=torch.float32,
    )

    observed = mean_reduce_attention_heads(
        values
    )

    torch.testing.assert_close(
        observed,
        torch.tensor(
            [0.5, 0.5, 0.4],
            dtype=torch.float32,
        ),
    )


def test_head_permutation_invariance() -> None:
    values = _arbitrary_nonnegative_weights(
        heads=4,
    )
    permutation = torch.tensor(
        [3, 1, 0, 2],
        dtype=torch.long,
    )

    original = mean_reduce_attention_heads(
        values
    )
    permuted = mean_reduce_attention_heads(
        values[:, permutation]
    )

    torch.testing.assert_close(
        original,
        permuted,
        rtol=1e-6,
        atol=1e-7,
    )


def test_edge_order_equivariance_low_level() -> None:
    values = _arbitrary_nonnegative_weights()
    permutation = torch.tensor(
        [6, 2, 0, 5, 1, 4, 3],
        dtype=torch.long,
    )

    original = mean_reduce_attention_heads(
        values
    )
    permuted = mean_reduce_attention_heads(
        values[permutation]
    )

    assert torch.equal(
        permuted,
        original[permutation],
    )


def test_identical_heads_reduce_to_shared_head() -> None:
    base = torch.linspace(
        0.1,
        0.9,
        EDGES,
    )
    values = base.unsqueeze(1).repeat(
        1,
        MULTIHEAD_COUNT,
    )

    observed = mean_reduce_attention_heads(
        values
    )

    torch.testing.assert_close(
        observed,
        base,
        rtol=1e-6,
        atol=1e-7,
    )


def test_nonnegativity_is_preserved() -> None:
    values = _arbitrary_nonnegative_weights()

    reduced = mean_reduce_attention_heads(
        values
    )

    assert bool(
        (reduced >= 0)
        .all()
        .item()
    )


def test_empty_edge_reduction() -> None:
    values = torch.empty(
        (0, MULTIHEAD_COUNT),
        dtype=torch.float32,
    )

    reduced = mean_reduce_attention_heads(
        values
    )

    assert reduced.shape == (0,)
    assert reduced.dtype == values.dtype


def test_float64_dtype_is_preserved() -> None:
    values = _arbitrary_nonnegative_weights(
        dtype=torch.float64,
    )

    reduced = mean_reduce_attention_heads(
        values
    )

    assert reduced.dtype == torch.float64


def test_reduce_attention_heads_dispatches_mean() -> None:
    values = _arbitrary_nonnegative_weights()

    assert torch.equal(
        reduce_attention_heads(
            values,
            head_reduction=(
                ATTENTION_HEAD_REDUCTION_MEAN
            ),
        ),
        mean_reduce_attention_heads(
            values
        ),
    )


# =============================================================================
# Low-level autograd
# =============================================================================


def test_mean_reduction_preserves_autograd() -> None:
    values = _arbitrary_nonnegative_weights(
        requires_grad=True,
    )

    reduced = mean_reduce_attention_heads(
        values
    )
    coefficients = torch.arange(
        1,
        EDGES + 1,
        dtype=values.dtype,
    )
    loss = (
        reduced
        * coefficients
    ).sum()
    loss.backward()

    assert values.grad is not None
    assert torch.isfinite(
        values.grad
    ).all()

    expected = (
        coefficients.unsqueeze(1)
        / float(MULTIHEAD_COUNT)
    ).expand_as(values)

    torch.testing.assert_close(
        values.grad,
        expected,
    )


def test_single_head_identity_preserves_gradient_exactly() -> None:
    values = _arbitrary_nonnegative_weights(
        heads=1,
        requires_grad=True,
    )

    reduced = mean_reduce_attention_heads(
        values
    )
    coefficients = torch.arange(
        1,
        EDGES + 1,
        dtype=values.dtype,
    )
    (
        reduced
        * coefficients
    ).sum().backward()

    assert values.grad is not None
    assert torch.equal(
        values.grad[:, 0],
        coefficients,
    )


def test_empty_reduction_retains_autograd_connectivity() -> None:
    values = torch.empty(
        (0, 2),
        dtype=torch.float32,
        requires_grad=True,
    )

    reduced = mean_reduce_attention_heads(
        values
    )

    assert reduced.requires_grad
    assert reduced.grad_fn is not None


# =============================================================================
# Low-level invalid contracts
# =============================================================================


def test_mean_reduce_rejects_non_tensor() -> None:
    with pytest.raises(
        TypeError,
        match="tensor",
    ):
        mean_reduce_attention_heads(
            [[0.5, 0.5]]  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "values",
    (
        torch.ones(EDGES),
        torch.ones(1, EDGES, 2),
    ),
)
def test_mean_reduce_rejects_wrong_rank(
    values: torch.Tensor,
) -> None:
    with pytest.raises(
        ValueError,
        match=r"shape \[E, A\]",
    ):
        mean_reduce_attention_heads(
            values
        )


def test_mean_reduce_rejects_zero_heads() -> None:
    with pytest.raises(
        ValueError,
        match="at least one attention head",
    ):
        mean_reduce_attention_heads(
            torch.empty(
                (EDGES, 0),
            )
        )


def test_mean_reduce_rejects_integer_values() -> None:
    with pytest.raises(
        ValueError,
        match="floating-point",
    ):
        mean_reduce_attention_heads(
            torch.ones(
                (EDGES, 2),
                dtype=torch.long,
            )
        )


@pytest.mark.parametrize(
    "invalid_value",
    (
        float("nan"),
        float("inf"),
        float("-inf"),
    ),
)
def test_mean_reduce_rejects_nonfinite_values(
    invalid_value: float,
) -> None:
    values = torch.ones(
        (EDGES, 2),
    )
    values[0, 0] = invalid_value

    with pytest.raises(
        ValueError,
        match="finite",
    ):
        mean_reduce_attention_heads(
            values
        )


def test_mean_reduce_rejects_negative_values() -> None:
    values = torch.ones(
        (EDGES, 2),
    )
    values[0, 0] = -0.1

    with pytest.raises(
        ValueError,
        match="nonnegative",
    ):
        mean_reduce_attention_heads(
            values
        )


@pytest.mark.parametrize(
    (
        "reduction",
        "reason_fragment",
    ),
    (
        (
            ATTENTION_HEAD_REDUCTION_WEIGHTED_MEAN,
            "source",
        ),
        (
            ATTENTION_HEAD_REDUCTION_MAX,
            "does not generally preserve",
        ),
        (
            ATTENTION_HEAD_REDUCTION_NONE,
            "explicit head axis",
        ),
    ),
)
def test_reduction_dispatch_rejects_unimplemented_canonical_modes(
    reduction: str,
    reason_fragment: str,
) -> None:
    with pytest.raises(
        NotImplementedError,
        match=reason_fragment,
    ):
        reduce_attention_heads(
            torch.ones(
                (EDGES, 2),
            ),
            head_reduction=reduction,
        )


def test_reduction_dispatch_rejects_unknown_mode() -> None:
    with pytest.raises(
        ValueError,
        match="Unknown attention head reduction",
    ):
        reduce_attention_heads(
            torch.ones(
                (EDGES, 2),
            ),
            head_reduction="unknown",
        )


def test_reduction_dispatch_rejects_non_string_mode() -> None:
    with pytest.raises(
        TypeError,
        match="must be a string",
    ):
        reduce_attention_heads(
            torch.ones(
                (EDGES, 2),
            ),
            head_reduction=3,  # type: ignore[arg-type]
        )


# =============================================================================
# Preservation of target-relation group normalization
# =============================================================================


@pytest.mark.parametrize(
    "heads",
    (
        1,
        2,
        MULTIHEAD_COUNT,
        5,
    ),
)
def test_mean_reduction_preserves_group_normalization(
    heads: int,
) -> None:
    inputs = _inputs()
    normalization = _normalization_output(
        inputs,
        heads=heads,
    )
    reduced = mean_reduce_attention_heads(
        normalization
        .normalized_weights_by_head
    )

    assert_reduced_attention_normalized(
        reduced,
        normalization,
    )
    assert (
        maximum_reduced_attention_normalization_error(
            reduced,
            normalization,
        )
        <= 1e-6
    )


def test_reduced_group_sums_equal_one_and_absent_groups_zero() -> None:
    inputs = _inputs()
    normalization = _normalization_output(
        inputs,
        heads=MULTIHEAD_COUNT,
    )
    reduced = mean_reduce_attention_heads(
        normalization
        .normalized_weights_by_head
    )
    sums = torch.zeros(
        NUM_GROUPS,
        dtype=reduced.dtype,
    )
    sums.index_add_(
        0,
        normalization.group_ids,
        reduced,
    )
    present = normalization.group_counts > 0

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


def test_singleton_groups_remain_exact_one() -> None:
    inputs = _inputs()
    normalization = _normalization_output(
        inputs,
        heads=MULTIHEAD_COUNT,
    )
    reduced = mean_reduce_attention_heads(
        normalization
        .normalized_weights_by_head
    )
    singleton_edges = (
        normalization.group_counts[
            normalization.group_ids
        ]
        == 1
    )

    assert torch.equal(
        reduced[singleton_edges],
        torch.ones_like(
            reduced[singleton_edges]
        ),
    )


def test_empty_reduced_normalization_error_is_zero() -> None:
    inputs = _inputs(
        empty_edges=True,
    )
    normalization = _normalization_output(
        inputs,
        heads=MULTIHEAD_COUNT,
    )
    reduced = mean_reduce_attention_heads(
        normalization
        .normalized_weights_by_head
    )

    assert (
        maximum_reduced_attention_normalization_error(
            reduced,
            normalization,
        )
        == 0.0
    )


def test_assert_reduced_normalized_rejects_bad_group_sum() -> None:
    normalization = _normalization_output(
        _inputs(),
        heads=2,
    )
    reduced = mean_reduce_attention_heads(
        normalization
        .normalized_weights_by_head
    ).clone()
    reduced[0] += 0.1

    with pytest.raises(
        ValueError,
        match="sum to one",
    ):
        assert_reduced_attention_normalized(
            reduced,
            normalization,
        )


def test_maximum_reduced_normalization_error_detects_perturbation() -> None:
    normalization = _normalization_output(
        _inputs(),
        heads=2,
    )
    reduced = mean_reduce_attention_heads(
        normalization
        .normalized_weights_by_head
    ).clone()
    reduced[0] += 0.25

    observed = (
        maximum_reduced_attention_normalization_error(
            reduced,
            normalization,
        )
    )

    assert observed == pytest.approx(
        0.25,
        abs=1e-6,
    )


def test_assert_reduced_normalized_respects_custom_tolerance() -> None:
    normalization = _normalization_output(
        _inputs(),
        heads=2,
    )
    reduced = mean_reduce_attention_heads(
        normalization
        .normalized_weights_by_head
    ).clone()
    reduced[0] += 1e-4

    assert_reduced_attention_normalized(
        reduced,
        normalization,
        atol=1e-3,
        rtol=0.0,
    )

    with pytest.raises(ValueError):
        assert_reduced_attention_normalized(
            reduced,
            normalization,
            atol=1e-7,
            rtol=0.0,
        )


def test_reduced_normalization_helpers_reject_wrong_source_type() -> None:
    values = torch.ones(
        EDGES,
    )

    with pytest.raises(
        TypeError,
        match="AttentionNormalizationOutput",
    ):
        assert_reduced_attention_normalized(
            values,
            object(),  # type: ignore[arg-type]
        )

    with pytest.raises(
        TypeError,
        match="AttentionNormalizationOutput",
    ):
        maximum_reduced_attention_normalization_error(
            values,
            object(),  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "bad_values",
    (
        [1.0] * EDGES,
        torch.ones(EDGES, 1),
        torch.ones(EDGES - 1),
        torch.ones(EDGES, dtype=torch.long),
    ),
)
def test_reduced_normalization_helpers_reject_invalid_edge_vector(
    bad_values: Any,
) -> None:
    normalization = _normalization_output(
        _inputs(),
        heads=2,
    )

    with pytest.raises(
        (TypeError, ValueError),
    ):
        assert_reduced_attention_normalized(
            bad_values,
            normalization,
        )


# =============================================================================
# Head-disagreement tensor diagnostics
# =============================================================================


def test_attention_head_mean_matches_reduction() -> None:
    values = _arbitrary_nonnegative_weights()

    assert torch.equal(
        attention_head_mean(values),
        mean_reduce_attention_heads(
            values
        ),
    )


def test_attention_head_variance_matches_population_formula() -> None:
    values = torch.tensor(
        [
            [0.0, 2.0],
            [1.0, 1.0],
            [2.0, 6.0],
        ],
        dtype=torch.float32,
    )

    observed = attention_head_variance(
        values
    )
    expected = torch.tensor(
        [1.0, 0.0, 4.0],
        dtype=torch.float32,
    )

    torch.testing.assert_close(
        observed,
        expected,
    )


def test_attention_head_standard_deviation_is_sqrt_variance() -> None:
    values = _arbitrary_nonnegative_weights()

    torch.testing.assert_close(
        attention_head_standard_deviation(
            values
        ),
        torch.sqrt(
            attention_head_variance(
                values
            )
        ),
    )


def test_attention_head_range_matches_max_minus_min() -> None:
    values = torch.tensor(
        [
            [0.1, 0.9, 0.4],
            [0.3, 0.3, 0.3],
        ],
        dtype=torch.float32,
    )

    observed = attention_head_range(
        values
    )

    torch.testing.assert_close(
        observed,
        torch.tensor(
            [0.8, 0.0],
            dtype=torch.float32,
        ),
    )


def test_attention_head_mean_absolute_deviation() -> None:
    values = torch.tensor(
        [
            [0.0, 2.0],
            [1.0, 1.0],
        ],
        dtype=torch.float32,
    )

    observed = (
        attention_head_mean_absolute_deviation(
            values
        )
    )

    torch.testing.assert_close(
        observed,
        torch.tensor(
            [1.0, 0.0],
            dtype=torch.float32,
        ),
    )


def test_single_head_disagreement_diagnostics_are_zero() -> None:
    values = _arbitrary_nonnegative_weights(
        heads=1,
    )

    assert torch.equal(
        attention_head_variance(values),
        torch.zeros(EDGES),
    )
    assert torch.equal(
        attention_head_standard_deviation(
            values
        ),
        torch.zeros(EDGES),
    )
    assert torch.equal(
        attention_head_range(values),
        torch.zeros(EDGES),
    )
    assert torch.equal(
        attention_head_mean_absolute_deviation(
            values
        ),
        torch.zeros(EDGES),
    )


def test_identical_multihead_diagnostics_are_zero() -> None:
    base = torch.linspace(
        0.1,
        0.9,
        EDGES,
    )
    values = base.unsqueeze(1).repeat(
        1,
        MULTIHEAD_COUNT,
    )

    assert torch.equal(
        attention_head_variance(values),
        torch.zeros(EDGES),
    )
    assert torch.equal(
        attention_head_range(values),
        torch.zeros(EDGES),
    )


def test_empty_disagreement_diagnostics() -> None:
    values = torch.empty(
        (0, MULTIHEAD_COUNT),
    )

    assert attention_head_variance(
        values
    ).shape == (0,)
    assert attention_head_standard_deviation(
        values
    ).shape == (0,)
    assert attention_head_range(
        values
    ).shape == (0,)
    assert (
        attention_head_mean_absolute_deviation(
            values
        ).shape
        == (0,)
    )
    assert maximum_attention_head_range(
        values
    ) == 0.0
    assert (
        mean_attention_head_standard_deviation(
            values
        )
        == 0.0
    )


def test_tensor_diagnostics_preserve_autograd() -> None:
    values = _arbitrary_nonnegative_weights(
        requires_grad=True,
    )

    loss = (
        attention_head_variance(values).sum()
        + attention_head_range(values).sum()
        + attention_head_mean_absolute_deviation(
            values
        ).sum()
    )
    loss.backward()

    assert values.grad is not None
    assert torch.isfinite(
        values.grad
    ).all()


@pytest.mark.parametrize(
    "diagnostic",
    (
        attention_head_mean,
        attention_head_variance,
        attention_head_standard_deviation,
        attention_head_range,
        attention_head_mean_absolute_deviation,
        maximum_attention_head_range,
        mean_attention_head_standard_deviation,
        head_disagreement_summary,
    ),
)
def test_diagnostics_reject_negative_values(
    diagnostic: Any,
) -> None:
    values = torch.ones(
        (EDGES, 2),
    )
    values[0, 0] = -0.1

    with pytest.raises(
        ValueError,
        match="nonnegative",
    ):
        diagnostic(values)


# =============================================================================
# Scalar disagreement diagnostics and summary
# =============================================================================


def test_maximum_head_range() -> None:
    values = torch.tensor(
        [
            [0.1, 0.9, 0.4],
            [0.2, 0.4, 0.3],
        ],
        dtype=torch.float32,
    )

    assert maximum_attention_head_range(
        values
    ) == pytest.approx(
        0.8,
        abs=1e-6,
    )


def test_mean_head_standard_deviation() -> None:
    values = torch.tensor(
        [
            [0.0, 2.0],
            [1.0, 1.0],
        ],
        dtype=torch.float32,
    )

    assert (
        mean_attention_head_standard_deviation(
            values
        )
        == pytest.approx(
            0.5,
            abs=1e-6,
        )
    )


def test_head_disagreement_summary_fields() -> None:
    values = torch.tensor(
        [
            [0.1, 0.9, 0.4],
            [0.3, 0.3, 0.3],
        ],
        dtype=torch.float32,
    )

    summary = head_disagreement_summary(
        values
    )

    assert summary["edge_count"] == 2
    assert summary["num_heads"] == 3
    assert summary[
        "maximum_head_range"
    ] == pytest.approx(0.8)
    assert summary[
        "mean_head_range"
    ] == pytest.approx(0.4)
    assert summary[
        "mean_head_standard_deviation"
    ] >= 0.0
    assert summary[
        "mean_head_absolute_deviation"
    ] >= 0.0


def test_empty_head_disagreement_summary() -> None:
    summary = head_disagreement_summary(
        torch.empty(
            (0, MULTIHEAD_COUNT),
        )
    )

    assert summary == {
        "edge_count": 0,
        "num_heads": MULTIHEAD_COUNT,
        "maximum_head_range": 0.0,
        "mean_head_range": 0.0,
        "mean_head_standard_deviation": 0.0,
        "mean_head_absolute_deviation": 0.0,
    }


def test_single_head_summary_does_not_claim_disagreement() -> None:
    summary = head_disagreement_summary(
        _arbitrary_nonnegative_weights(
            heads=1,
        )
    )

    assert summary["num_heads"] == 1
    assert summary["maximum_head_range"] == 0.0
    assert summary[
        "mean_head_standard_deviation"
    ] == 0.0
    assert summary[
        "mean_head_absolute_deviation"
    ] == 0.0


# =============================================================================
# Functional metadata-preserving reduction
# =============================================================================


@pytest.mark.parametrize(
    "heads",
    (
        1,
        2,
        MULTIHEAD_COUNT,
        5,
    ),
)
def test_functional_reduction_output_contract(
    heads: int,
) -> None:
    normalization = _normalization_output(
        _inputs(),
        heads=heads,
    )

    output = reduce_normalized_attention_heads(
        normalization
    )

    assert isinstance(
        output,
        AttentionHeadReductionOutput,
    )
    assert (
        output.source_normalization_output
        is normalization
    )
    assert output.source_inputs is (
        normalization.source_inputs
    )
    assert output.head_reduction == (
        ATTENTION_HEAD_REDUCTION_MEAN
    )
    assert output.edge_weights.shape == (
        EDGES,
    )
    assert torch.equal(
        output.edge_weights,
        mean_reduce_attention_heads(
            normalization
            .normalized_weights_by_head
        ),
    )
    assert output.parameter_fingerprint is None
    assert output.encoder_architecture_fingerprint


def test_functional_single_head_is_exact_identity() -> None:
    normalization = _normalization_output(
        _inputs(),
        heads=1,
    )
    output = reduce_normalized_attention_heads(
        normalization
    )

    assert torch.equal(
        output.edge_weights,
        normalization
        .normalized_weights_by_head[
            :,
            0,
        ],
    )


def test_functional_empty_edges() -> None:
    normalization = _normalization_output(
        _inputs(
            empty_edges=True,
        ),
        heads=MULTIHEAD_COUNT,
    )

    output = reduce_normalized_attention_heads(
        normalization
    )

    assert output.edge_weights.shape == (0,)


def test_functional_preserves_autograd() -> None:
    inputs = _inputs()
    logits = _raw_logits(
        inputs,
        heads=MULTIHEAD_COUNT,
        requires_grad=True,
    )
    normalization = _normalization_output(
        inputs,
        heads=MULTIHEAD_COUNT,
        logits=logits,
    )

    output = reduce_normalized_attention_heads(
        normalization
    )
    coefficients = torch.arange(
        1,
        EDGES + 1,
        dtype=inputs.dtype,
    )
    (
        output.edge_weights
        * coefficients
    ).sum().backward()

    assert logits.grad is not None
    assert torch.isfinite(
        logits.grad
    ).all()


def test_functional_default_fingerprint_is_deterministic() -> None:
    normalization = _normalization_output(
        _inputs(),
        heads=MULTIHEAD_COUNT,
    )

    first = reduce_normalized_attention_heads(
        normalization
    )
    second = reduce_normalized_attention_heads(
        normalization
    )

    assert (
        first.encoder_architecture_fingerprint
        == second.encoder_architecture_fingerprint
    )


def test_functional_default_fingerprint_changes_with_head_count() -> None:
    first = reduce_normalized_attention_heads(
        _normalization_output(
            _inputs(),
            heads=1,
        )
    )
    second = reduce_normalized_attention_heads(
        _normalization_output(
            _inputs(),
            heads=2,
        )
    )

    assert (
        first.encoder_architecture_fingerprint
        != second.encoder_architecture_fingerprint
    )


def test_functional_custom_architecture_fingerprint() -> None:
    output = reduce_normalized_attention_heads(
        _normalization_output(
            _inputs(),
            heads=2,
        ),
        encoder_architecture_fingerprint=(
            "custom-reduction-architecture"
        ),
    )

    assert (
        output.encoder_architecture_fingerprint
        == "custom-reduction-architecture"
    )


def test_functional_rejects_blank_custom_fingerprint() -> None:
    with pytest.raises(
        ValueError,
        match="non-empty",
    ):
        reduce_normalized_attention_heads(
            _normalization_output(
                _inputs(),
                heads=2,
            ),
            encoder_architecture_fingerprint="",
        )


def test_functional_rejects_wrong_source_type() -> None:
    with pytest.raises(
        TypeError,
        match="AttentionNormalizationOutput",
    ):
        reduce_normalized_attention_heads(
            object()  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "reduction",
    (
        ATTENTION_HEAD_REDUCTION_WEIGHTED_MEAN,
        ATTENTION_HEAD_REDUCTION_MAX,
        ATTENTION_HEAD_REDUCTION_NONE,
    ),
)
def test_functional_rejects_unimplemented_reduction(
    reduction: str,
) -> None:
    with pytest.raises(
        NotImplementedError,
    ):
        reduce_normalized_attention_heads(
            _normalization_output(
                _inputs(),
                heads=2,
            ),
            head_reduction=reduction,
        )


# =============================================================================
# Parameter-free module identity
# =============================================================================


@pytest.mark.parametrize(
    "heads",
    (
        1,
        2,
        MULTIHEAD_COUNT,
    ),
)
def test_module_identity_and_parameter_free_contract(
    heads: int,
) -> None:
    module = MeanAttentionHeadReduction(
        num_heads=heads,
    )

    assert module.num_heads == heads
    assert module.head_reduction == (
        ATTENTION_HEAD_REDUCTION_MEAN
    )
    assert module.single_head_identity is (
        heads == 1
    )
    assert module.parameter_count == 0
    assert module.trainable_parameter_count == 0
    assert module.parameter_fingerprint() is None
    assert tuple(module.parameters()) == ()
    assert tuple(module.buffers()) == ()
    assert module.state_dict() == {}
    module.assert_parameter_free()


def test_module_architecture_metadata_is_explicit() -> None:
    module = MeanAttentionHeadReduction(
        num_heads=MULTIHEAD_COUNT,
    )
    architecture = module.architecture_dict()

    assert architecture["head_reduction"] == (
        ATTENTION_HEAD_REDUCTION_MEAN
    )
    assert architecture["num_heads"] == (
        MULTIHEAD_COUNT
    )
    assert architecture["input_shape"] == (
        "[E, A]"
    )
    assert architecture["output_shape"] == (
        "[E]"
    )
    assert architecture["reduction_axis"] == 1
    assert architecture["reduction_formula"] == (
        ATTENTION_HEAD_MEAN_FORMULA
    )
    assert architecture["head_mixture_weights"] == (
        "equal_1_over_num_heads"
    )
    assert architecture["learned_head_weights"] is False
    assert (
        architecture["head_permutation_invariant"]
        is True
    )
    assert (
        architecture[
            "preserves_target_relation_group_normalization"
        ]
        is True
    )
    assert architecture["claims_head_specialization"] is False
    assert (
        architecture["claims_uncertainty_calibration"]
        is False
    )
    assert architecture["claims_causal_importance"] is False
    assert architecture["relation_gate_owned_here"] is False
    assert architecture["normalization_owned_here"] is False
    assert (
        architecture["message_construction_owned_here"]
        is False
    )
    assert architecture["aggregation_owned_here"] is False


def test_module_architecture_fingerprint_is_deterministic() -> None:
    first = MeanAttentionHeadReduction(
        num_heads=3,
    )
    second = MeanAttentionHeadReduction(
        num_heads=3,
    )

    assert (
        first.architecture_fingerprint()
        == second.architecture_fingerprint()
    )


def test_module_architecture_fingerprint_changes_with_head_count() -> None:
    first = MeanAttentionHeadReduction(
        num_heads=1,
    )
    second = MeanAttentionHeadReduction(
        num_heads=2,
    )

    assert (
        first.architecture_fingerprint()
        != second.architecture_fingerprint()
    )


def test_module_extra_repr_is_informative() -> None:
    text = repr(
        MeanAttentionHeadReduction(
            num_heads=3,
        )
    )

    assert "num_heads=3" in text
    assert ATTENTION_HEAD_REDUCTION_MEAN in text
    assert "head_permutation_invariant=True" in text
    assert "parameter_free=True" in text


def test_module_reduce_tensor_matches_free_function() -> None:
    values = _arbitrary_nonnegative_weights()
    module = MeanAttentionHeadReduction(
        num_heads=MULTIHEAD_COUNT,
    )

    assert torch.equal(
        module.reduce_tensor(values),
        mean_reduce_attention_heads(
            values
        ),
    )


def test_module_reduce_tensor_rejects_wrong_head_count() -> None:
    module = MeanAttentionHeadReduction(
        num_heads=3,
    )

    with pytest.raises(
        ValueError,
        match="wrong head count",
    ):
        module.reduce_tensor(
            torch.ones(
                (EDGES, 2),
            )
        )


def test_module_forward_contract() -> None:
    normalization = _normalization_output(
        _inputs(),
        heads=MULTIHEAD_COUNT,
    )
    module = MeanAttentionHeadReduction(
        num_heads=MULTIHEAD_COUNT,
    )

    output = module(normalization)

    assert isinstance(
        output,
        AttentionHeadReductionOutput,
    )
    assert (
        output.source_normalization_output
        is normalization
    )
    assert output.head_reduction == (
        ATTENTION_HEAD_REDUCTION_MEAN
    )
    assert (
        output.encoder_architecture_fingerprint
        == module.architecture_fingerprint()
    )
    assert output.parameter_fingerprint is None


def test_module_forward_matches_functional_wrapper() -> None:
    normalization = _normalization_output(
        _inputs(),
        heads=MULTIHEAD_COUNT,
    )
    module = MeanAttentionHeadReduction(
        num_heads=MULTIHEAD_COUNT,
    )

    class_output = module(normalization)
    functional_output = (
        reduce_normalized_attention_heads(
            normalization,
            encoder_architecture_fingerprint=(
                module.architecture_fingerprint()
            ),
        )
    )

    assert torch.equal(
        class_output.edge_weights,
        functional_output.edge_weights,
    )
    assert (
        class_output.encoder_architecture_fingerprint
        == functional_output.encoder_architecture_fingerprint
    )


def test_module_runtime_head_count_must_match_architecture() -> None:
    normalization = _normalization_output(
        _inputs(),
        heads=2,
    )
    module = MeanAttentionHeadReduction(
        num_heads=3,
    )

    with pytest.raises(
        ValueError,
        match="Runtime attention head count differs",
    ):
        module(normalization)


def test_module_forward_rejects_wrong_source_type() -> None:
    module = MeanAttentionHeadReduction(
        num_heads=1,
    )

    with pytest.raises(
        TypeError,
        match="AttentionNormalizationOutput",
    ):
        module(
            object()  # type: ignore[arg-type]
        )


def test_module_empty_edges() -> None:
    normalization = _normalization_output(
        _inputs(
            empty_edges=True,
        ),
        heads=MULTIHEAD_COUNT,
    )
    output = MeanAttentionHeadReduction(
        num_heads=MULTIHEAD_COUNT,
    )(normalization)

    assert output.edge_weights.shape == (0,)


def test_module_disagreement_summary() -> None:
    normalization = _normalization_output(
        _inputs(),
        heads=MULTIHEAD_COUNT,
    )
    module = MeanAttentionHeadReduction(
        num_heads=MULTIHEAD_COUNT,
    )

    summary = module.disagreement_summary(
        normalization
    )

    assert summary["edge_count"] == EDGES
    assert summary["num_heads"] == (
        MULTIHEAD_COUNT
    )


def test_module_disagreement_summary_rejects_head_mismatch() -> None:
    normalization = _normalization_output(
        _inputs(),
        heads=2,
    )
    module = MeanAttentionHeadReduction(
        num_heads=3,
    )

    with pytest.raises(
        ValueError,
        match="Runtime attention head count differs",
    ):
        module.disagreement_summary(
            normalization
        )


@pytest.mark.parametrize(
    "bad_heads",
    (
        0,
        -1,
    ),
)
def test_module_rejects_nonpositive_head_count(
    bad_heads: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="positive",
    ):
        MeanAttentionHeadReduction(
            num_heads=bad_heads,
        )


@pytest.mark.parametrize(
    "bad_heads",
    (
        True,
        1.5,
        "2",
    ),
)
def test_module_rejects_noninteger_head_count(
    bad_heads: object,
) -> None:
    with pytest.raises(
        TypeError,
        match="integer",
    ):
        MeanAttentionHeadReduction(
            num_heads=bad_heads,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "reduction",
    (
        ATTENTION_HEAD_REDUCTION_WEIGHTED_MEAN,
        ATTENTION_HEAD_REDUCTION_MAX,
        ATTENTION_HEAD_REDUCTION_NONE,
    ),
)
def test_module_rejects_unimplemented_reduction(
    reduction: str,
) -> None:
    with pytest.raises(
        NotImplementedError,
    ):
        MeanAttentionHeadReduction(
            num_heads=2,
            head_reduction=reduction,
        )


def test_assert_parameter_free_detects_injected_parameter() -> None:
    module = MeanAttentionHeadReduction(
        num_heads=2,
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
    module = MeanAttentionHeadReduction(
        num_heads=2,
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
    module = MeanAttentionHeadReduction(
        num_heads=2,
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


@pytest.mark.parametrize(
    "heads",
    (
        1,
        2,
        MULTIHEAD_COUNT,
    ),
)
def test_from_config_builds_expected_reducer(
    heads: int,
) -> None:
    module = (
        MeanAttentionHeadReduction
        .from_config(
            config=_config(
                heads=heads,
            )
        )
    )

    assert isinstance(
        module,
        MeanAttentionHeadReduction,
    )
    assert module.num_heads == heads


@pytest.mark.parametrize(
    "heads",
    (
        1,
        2,
        MULTIHEAD_COUNT,
    ),
)
def test_dispatcher_builds_expected_reducer(
    heads: int,
) -> None:
    module = build_attention_head_reducer(
        config=_config(
            heads=heads,
        )
    )

    assert isinstance(
        module,
        MeanAttentionHeadReduction,
    )
    assert module.num_heads == heads


def test_from_config_rejects_wrong_config_type() -> None:
    with pytest.raises(
        TypeError,
        match="FunctionalMessagePassingConfig",
    ):
        MeanAttentionHeadReduction.from_config(
            config=object(),  # type: ignore[arg-type]
        )


def test_dispatcher_rejects_wrong_config_type() -> None:
    with pytest.raises(
        TypeError,
        match="FunctionalMessagePassingConfig",
    ):
        build_attention_head_reducer(
            config=object(),  # type: ignore[arg-type]
        )


# =============================================================================
# End-to-end metamorphic properties through module wrapper
# =============================================================================


def test_module_head_permutation_invariance() -> None:
    inputs = _inputs()
    normalization = _normalization_output(
        inputs,
        heads=MULTIHEAD_COUNT,
    )
    permutation = torch.tensor(
        [2, 0, 1],
        dtype=torch.long,
    )
    permuted_weights = (
        normalization
        .normalized_weights_by_head[
            :,
            permutation,
        ]
    )
    permuted_normalization = (
        _normalization_output(
            inputs,
            heads=MULTIHEAD_COUNT,
            normalized_weights=(
                permuted_weights
            ),
        )
    )
    module = MeanAttentionHeadReduction(
        num_heads=MULTIHEAD_COUNT,
    )

    original = module(normalization)
    permuted = module(
        permuted_normalization
    )

    assert torch.equal(
        original.edge_weights,
        permuted.edge_weights,
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
    original = _normalization_output(
        original_inputs,
        heads=MULTIHEAD_COUNT,
        logits=original_logits,
    )

    permuted_inputs = _inputs(
        permutation=permutation,
    )
    permuted = _normalization_output(
        permuted_inputs,
        heads=MULTIHEAD_COUNT,
        logits=(
            original_logits[
                permutation
            ]
        ),
    )

    module = MeanAttentionHeadReduction(
        num_heads=MULTIHEAD_COUNT,
    )
    original_output = module(original)
    permuted_output = module(permuted)

    torch.testing.assert_close(
        permuted_output.edge_weights,
        original_output.edge_weights[
            permutation
        ],
    )


def test_module_single_head_exact_identity() -> None:
    normalization = _normalization_output(
        _inputs(),
        heads=1,
    )
    output = MeanAttentionHeadReduction(
        num_heads=1,
    )(normalization)

    assert torch.equal(
        output.edge_weights,
        normalization
        .normalized_weights_by_head[
            :,
            0,
        ],
    )


def test_module_output_contains_no_causal_or_gate_fields() -> None:
    output = MeanAttentionHeadReduction(
        num_heads=2,
    )(
        _normalization_output(
            _inputs(),
            heads=2,
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
        "aggregation_denominator",
    )
    assert not hasattr(
        output,
        "semantic_edge_weight",
    )


# =============================================================================
# Optional CUDA contracts
# =============================================================================


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is not available.",
)
def test_low_level_cuda_reduction_and_gradients() -> None:
    values = _arbitrary_nonnegative_weights(
        device="cuda",
        requires_grad=True,
    )

    reduced = mean_reduce_attention_heads(
        values
    )
    reduced.sum().backward()

    assert reduced.device.type == "cuda"
    assert values.grad is not None
    assert values.grad.device.type == "cuda"


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is not available.",
)
def test_module_cuda_forward() -> None:
    inputs = _inputs(
        device="cuda",
    )
    normalization = _normalization_output(
        inputs,
        heads=MULTIHEAD_COUNT,
    )
    module = MeanAttentionHeadReduction(
        num_heads=MULTIHEAD_COUNT,
    ).to(device="cuda")

    output = module(normalization)

    assert output.device.type == "cuda"
    assert output.edge_weights.device.type == (
        "cuda"
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is not available.",
)
def test_reduced_normalization_rejects_cpu_vector_for_cuda_source() -> None:
    inputs = _inputs(
        device="cuda",
    )
    normalization = _normalization_output(
        inputs,
        heads=2,
    )
    cpu_edge_weights = (
        normalization
        .normalized_weights_by_head
        .mean(dim=1)
        .cpu()
    )

    with pytest.raises(
        ValueError,
        match="device",
    ):
        assert_reduced_attention_normalized(
            cpu_edge_weights,
            normalization,
        )
