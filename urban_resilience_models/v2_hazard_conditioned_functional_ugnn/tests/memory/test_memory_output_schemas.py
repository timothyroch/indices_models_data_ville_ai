"""
Consolidated contract tests for temporal-memory output schemas.

Target repository path:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            tests/
                memory/
                    test_memory_output_schemas.py

Implementations under test:
    memory/schemas/sequence_encoding.py
    memory/schemas/temporal_pooling.py
    memory/schemas/urban_memory.py
    memory/schemas/hazard_queried_memory.py

The suite freezes:

- sequence-preserving encoder outputs ``[N, T, H]``;
- exact source-history preservation;
- hazard-independent pooled memory ``[N, P]``;
- per-head temporal pooling weights ``[N, A, T]``;
- exact sequence preservation inside ``UrbanMemory``;
- query-neutral temporal retrieval;
- graph- and node-scoped hazard-query alignment;
- separation of retrieved context and fused memory;
- zero-history policies;
- alignment, architecture, parameter, and execution provenance;
- semantic, value, and lineage fingerprints;
- validated device reconstruction.

Hazard embedding and query-encoder internals are upstream concerns. This suite
uses small metadata-preserving fake hazard contracts and patches only the
three imported hazard result classes inside ``hazard_queried_memory``. The
memory schemas themselves are exercised without replacing any memory class.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import MappingProxyType
from typing import Any

import pytest
import torch

from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.memory.schemas import hazard_queried_memory as hazard_memory_schemas
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.memory.schemas.hazard_queried_memory import (
    CANONICAL_HAZARD_MEMORY_FUSION_POLICIES,
    CANONICAL_HAZARD_QUERY_ALIGNMENT_SCOPES,
    CANONICAL_TEMPORAL_QUERY_RETRIEVAL_HEAD_REDUCTIONS,
    CANONICAL_TEMPORAL_QUERY_RETRIEVAL_KINDS,
    CANONICAL_TEMPORAL_QUERY_RETRIEVAL_ZERO_HISTORY_POLICIES,
    HAZARD_QUERIED_MEMORY_SCHEMA_VERSION,
    HAZARD_QUERIED_MEMORY_SCIENTIFIC_INTERPRETATION,
    HAZARD_QUERY_ALIGNMENT_POLICY,
    TEMPORAL_QUERY_RETRIEVAL_NORMALIZATION_POLICY,
    TEMPORAL_QUERY_RETRIEVAL_OUTPUT_SCHEMA_VERSION,
    TEMPORAL_QUERY_RETRIEVAL_PADDING_POLICY,
    TEMPORAL_QUERY_RETRIEVAL_WEIGHT_SEMANTICS,
    HazardConditionedMemory,
    HazardMemoryFusionPolicy,
    HazardQueriedMemory,
    HazardQueryAlignmentScope,
    QueryRetrievalOutput,
    TemporalQueryRetrievalHeadReduction,
    TemporalQueryRetrievalKind,
    TemporalQueryRetrievalOutput,
    TemporalQueryRetrievalZeroHistoryPolicy,
    TemporalRetrievalOutput,
    hazard_query_alignment_fingerprint,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.memory.schemas.history_inputs import (
    HistoricalSequenceInputs,
    HistoryZeroLengthPolicy,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.memory.schemas.provenance import (
    MemoryArchitectureProvenance,
    MemoryComputationProvenance,
    MemoryExecutionLineage,
    MemorySourceProvenance,
    TemporalFeatureAxis,
    TemporalNodeAxis,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.memory.schemas.sequence_encoding import (
    CANONICAL_TEMPORAL_SEQUENCE_ENCODER_KINDS,
    TEMPORAL_SEQUENCE_ENCODING_PADDING_POLICY,
    TEMPORAL_SEQUENCE_ENCODING_SCHEMA_VERSION,
    TEMPORAL_SEQUENCE_ENCODING_SCIENTIFIC_INTERPRETATION,
    TEMPORAL_SEQUENCE_ENCODING_VALUE_SEMANTICS,
    SequenceEncoding,
    SharedTemporalSequenceEncoding,
    TemporalSequenceEncoderKind,
    TemporalSequenceEncoding,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.memory.schemas.temporal_coordinates import (
    RelativeTemporalCoordinates,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.memory.schemas.temporal_pooling import (
    CANONICAL_TEMPORAL_POOLING_HEAD_REDUCTIONS,
    CANONICAL_TEMPORAL_POOLING_KINDS,
    CANONICAL_TEMPORAL_POOLING_ZERO_HISTORY_POLICIES,
    TEMPORAL_POOLING_NORMALIZATION_POLICY,
    TEMPORAL_POOLING_OUTPUT_SCHEMA_VERSION,
    TEMPORAL_POOLING_PADDING_POLICY,
    TEMPORAL_POOLING_SCIENTIFIC_INTERPRETATION,
    TEMPORAL_POOLING_WEIGHT_SEMANTICS,
    PoolingOutput,
    TemporalMemoryPoolingOutput,
    TemporalPoolingHeadReduction,
    TemporalPoolingKind,
    TemporalPoolingOutput,
    TemporalPoolingZeroHistoryPolicy,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.memory.schemas.urban_memory import (
    CANONICAL_URBAN_MEMORY_ASSEMBLY_POLICIES,
    URBAN_MEMORY_POOLING_SCOPE,
    URBAN_MEMORY_SCHEMA_VERSION,
    URBAN_MEMORY_SCIENTIFIC_INTERPRETATION,
    URBAN_MEMORY_SEQUENCE_PRESERVATION_POLICY,
    SharedUrbanMemory,
    UrbanMemory,
    UrbanMemoryAssemblyPolicy,
    UrbanTemporalMemory,
)


N = 3
T = 4
D = 2
H = 3
QUERY_DIM = 2


# =============================================================================
# Isolated hazard-result contracts
# =============================================================================


class FakeHazardEmbeddingLookup:
    """Graph-scoped hazard embedding metadata."""

    def __init__(
        self,
        *,
        embeddings: torch.Tensor,
    ) -> None:
        self.embeddings = embeddings


class FakeNodeAlignedHazardEmbeddingLookup:
    """Node-scoped hazard embedding metadata with graph lineage."""

    def __init__(
        self,
        *,
        node_embeddings: torch.Tensor,
        graph_lookup: FakeHazardEmbeddingLookup,
        node_batch_index: torch.Tensor,
    ) -> None:
        self.node_embeddings = node_embeddings
        self.graph_lookup = graph_lookup
        self.node_batch_index = node_batch_index


class FakeHazardQueryEncoding:
    """Minimal metadata-preserving query output required by memory schemas."""

    def __init__(
        self,
        *,
        query: torch.Tensor,
        source_embedding: (
            FakeHazardEmbeddingLookup
            | FakeNodeAlignedHazardEmbeddingLookup
        ),
        lineage_fingerprint: str = "hazard-query-lineage-v1",
        query_encoder_architecture_fingerprint: str = (
            "hazard-query-architecture-v1"
        ),
    ) -> None:
        self.query = query
        self.source_embedding = source_embedding
        self.lineage_fingerprint = lineage_fingerprint
        self.query_encoder_architecture_fingerprint = (
            query_encoder_architecture_fingerprint
        )

    @property
    def item_count(self) -> int:
        return int(
            self.query.shape[0]
        )

    @property
    def query_dim(self) -> int:
        return int(
            self.query.shape[1]
        )


@pytest.fixture(autouse=True)
def _patch_hazard_contracts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        hazard_memory_schemas,
        "HazardEmbeddingLookup",
        FakeHazardEmbeddingLookup,
    )
    monkeypatch.setattr(
        hazard_memory_schemas,
        "NodeAlignedHazardEmbeddingLookup",
        FakeNodeAlignedHazardEmbeddingLookup,
    )
    monkeypatch.setattr(
        hazard_memory_schemas,
        "HazardQueryEncoding",
        FakeHazardQueryEncoding,
    )


# =============================================================================
# Core factories
# =============================================================================


def _mask() -> torch.Tensor:
    return torch.tensor(
        [
            [True, True, True, False],
            [True, True, False, False],
            [True, True, True, True],
        ],
        dtype=torch.bool,
    )


def _zero_history_mask() -> torch.Tensor:
    return torch.tensor(
        [
            [True, True, True, False],
            [False, False, False, False],
            [True, True, True, True],
        ],
        dtype=torch.bool,
    )


def _history_values(
    mask: torch.Tensor,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    values = (
        torch.arange(
            N * T * D,
            dtype=dtype,
        )
        .reshape(N, T, D)
        / 10.0
        + 0.1
    )
    return values * mask.unsqueeze(-1)


def _relative_values(
    mask: torch.Tensor,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    base = torch.tensor(
        [
            [-3.0, -2.0, -1.0, 0.0],
            [-2.0, -1.0, 0.0, 0.0],
            [-4.0, -3.0, -2.0, -1.0],
        ],
        dtype=dtype,
    )
    return base * mask


def _history(
    *,
    mask: torch.Tensor | None = None,
    dtype: torch.dtype = torch.float32,
    zero_history_policy: HistoryZeroLengthPolicy | str = (
        HistoryZeroLengthPolicy.ERROR
    ),
) -> HistoricalSequenceInputs:
    if mask is None:
        mask = _mask()

    return HistoricalSequenceInputs(
        history=_history_values(
            mask,
            dtype=dtype,
        ),
        timestep_mask=mask,
        node_axis=TemporalNodeAxis(
            node_ids=(
                "node-0",
                "node-1",
                "node-2",
            ),
            node_batch_index=torch.tensor(
                [0, 0, 1],
                dtype=torch.long,
            ),
            graph_count=2,
            graph_ids=(
                "graph-0",
                "graph-1",
            ),
            source_fingerprint="node-axis-source",
        ),
        feature_axis=TemporalFeatureAxis(
            feature_names=(
                "burden",
                "reporting_intensity",
            ),
            source_fingerprint="feature-axis-source",
        ),
        temporal_coordinates=RelativeTemporalCoordinates(
            values=_relative_values(
                mask,
                dtype=dtype,
            ),
            unit="months",
        ),
        source_provenance=MemorySourceProvenance(
            source_name="historical-panel",
            source_kind="node-month-history",
            source_fingerprint="history-source-v1",
        ),
        padding_direction="right",
        zero_length_policy=zero_history_policy,
    )


def _architecture(
    *,
    name: str,
    kind: str,
    fingerprint: str,
) -> MemoryArchitectureProvenance:
    return MemoryArchitectureProvenance(
        component_name=name,
        component_kind=kind,
        architecture_fingerprint=fingerprint,
    )


def _computation(
    *,
    source_fingerprints: tuple[str, ...],
    architecture_name: str,
    architecture_kind: str,
    architecture_fingerprint: str,
    source_history: HistoricalSequenceInputs,
) -> MemoryComputationProvenance:
    architecture = _architecture(
        name=architecture_name,
        kind=architecture_kind,
        fingerprint=architecture_fingerprint,
    )
    lineage = MemoryExecutionLineage(
        operation_name=architecture_name,
        source_lineage_fingerprints=source_fingerprints,
        architecture_fingerprint=architecture_fingerprint,
        node_axis_fingerprint=(
            source_history
            .node_axis
            .fingerprint()
        ),
        temporal_axis_fingerprint=(
            source_history
            .temporal_alignment_fingerprint()
        ),
        feature_axis_fingerprint=(
            source_history
            .feature_axis
            .fingerprint()
        ),
    )
    return MemoryComputationProvenance(
        architecture=architecture,
        lineage=lineage,
    )


def _encoded_values(
    source_history: HistoricalSequenceInputs,
    *,
    hidden_dim: int = H,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    resolved_dtype = (
        source_history.dtype
        if dtype is None
        else dtype
    )
    values = (
        torch.arange(
            source_history.node_count
            * source_history.sequence_length
            * hidden_dim,
            dtype=resolved_dtype,
        )
        .reshape(
            source_history.node_count,
            source_history.sequence_length,
            hidden_dim,
        )
        / 10.0
        + 0.1
    )
    return (
        values
        * source_history
        .timestep_mask
        .unsqueeze(-1)
    )


def _encoding(
    *,
    source_history: HistoricalSequenceInputs | None = None,
    encoded_sequence: torch.Tensor | None = None,
    encoder_kind: TemporalSequenceEncoderKind | str = (
        TemporalSequenceEncoderKind.GRU
    ),
    computation_provenance: (
        MemoryComputationProvenance
        | None
    ) = None,
) -> TemporalSequenceEncoding:
    if source_history is None:
        source_history = _history()

    if encoded_sequence is None:
        encoded_sequence = _encoded_values(
            source_history
        )

    if computation_provenance is None:
        computation_provenance = _computation(
            source_fingerprints=(
                source_history.lineage_fingerprint(),
            ),
            architecture_name="encode_history",
            architecture_kind="gru",
            architecture_fingerprint="encoder-architecture-v1",
            source_history=source_history,
        )

    return TemporalSequenceEncoding(
        encoded_sequence=encoded_sequence,
        source_history=source_history,
        encoder_kind=encoder_kind,
        computation_provenance=computation_provenance,
    )


def _single_head_weights(
    source_encoding: TemporalSequenceEncoding,
) -> torch.Tensor:
    weights = torch.zeros(
        (
            source_encoding.node_count,
            1,
            source_encoding.sequence_length,
        ),
        dtype=source_encoding.dtype,
        device=source_encoding.device,
    )
    for node_index, length in enumerate(
        source_encoding.valid_lengths.tolist()
    ):
        if length > 0:
            weights[
                node_index,
                0,
                :length,
            ] = 1.0 / float(length)
    return weights


def _multihead_weights(
    source_encoding: TemporalSequenceEncoding,
    *,
    heads: int = 2,
) -> torch.Tensor:
    return _single_head_weights(
        source_encoding
    ).expand(
        -1,
        heads,
        -1,
    ).clone()


def _pooled_values(
    source_encoding: TemporalSequenceEncoding,
    *,
    output_dim: int | None = None,
) -> torch.Tensor:
    hidden = (
        source_encoding.hidden_dim
        if output_dim is None
        else output_dim
    )
    values = torch.zeros(
        (
            source_encoding.node_count,
            hidden,
        ),
        dtype=source_encoding.dtype,
        device=source_encoding.device,
    )
    nonempty = (
        source_encoding
        .valid_lengths
        > 0
    )
    if hidden == source_encoding.hidden_dim:
        denominator = (
            source_encoding
            .valid_lengths
            .clamp_min(1)
            .to(
                dtype=source_encoding.dtype
            )
            .unsqueeze(-1)
        )
        values = (
            source_encoding
            .encoded_sequence
            .sum(dim=1)
            / denominator
        )
        values[
            ~nonempty
        ] = 0
        return values

    values[
        nonempty
    ] = 1.0
    return values


def _pooling(
    *,
    source_encoding: TemporalSequenceEncoding | None = None,
    pooled_memory: torch.Tensor | None = None,
    pooling_weights: torch.Tensor | None = None,
    pooling_kind: TemporalPoolingKind | str = (
        TemporalPoolingKind.MASKED_MEAN
    ),
    head_reduction: TemporalPoolingHeadReduction | str = (
        TemporalPoolingHeadReduction.SINGLE_HEAD
    ),
    zero_history_policy: TemporalPoolingZeroHistoryPolicy | str = (
        TemporalPoolingZeroHistoryPolicy.ERROR
    ),
    head_reduction_weights: torch.Tensor | None = None,
    computation_provenance: (
        MemoryComputationProvenance
        | None
    ) = None,
) -> TemporalPoolingOutput:
    if source_encoding is None:
        source_encoding = _encoding()

    if pooled_memory is None:
        pooled_memory = _pooled_values(
            source_encoding
        )

    if pooling_weights is None:
        pooling_weights = _single_head_weights(
            source_encoding
        )

    if computation_provenance is None:
        computation_provenance = _computation(
            source_fingerprints=(
                source_encoding
                .lineage_fingerprint(),
            ),
            architecture_name="pool_history",
            architecture_kind="masked_mean",
            architecture_fingerprint="pooling-architecture-v1",
            source_history=(
                source_encoding
                .source_history
            ),
        )

    return TemporalPoolingOutput(
        pooled_memory=pooled_memory,
        pooling_weights=pooling_weights,
        source_encoding=source_encoding,
        pooling_kind=pooling_kind,
        head_reduction=head_reduction,
        zero_history_policy=zero_history_policy,
        computation_provenance=computation_provenance,
        head_reduction_weights=head_reduction_weights,
    )


def _urban_computation(
    *,
    sequence: TemporalSequenceEncoding,
    pooling: TemporalPoolingOutput | None,
) -> MemoryComputationProvenance:
    sources = [
        sequence.lineage_fingerprint(),
    ]
    if pooling is not None:
        sources.append(
            pooling.lineage_fingerprint()
        )

    return _computation(
        source_fingerprints=tuple(sources),
        architecture_name="assemble_urban_memory",
        architecture_kind=(
            "sequence_only"
            if pooling is None
            else "sequence_with_generic_pooling"
        ),
        architecture_fingerprint="urban-memory-architecture-v1",
        source_history=sequence.source_history,
    )


def _urban_memory(
    *,
    sequence: TemporalSequenceEncoding | None = None,
    pooling: TemporalPoolingOutput | None = None,
    with_pooling: bool = False,
    computation_provenance: (
        MemoryComputationProvenance
        | None
    ) = None,
) -> UrbanMemory:
    if sequence is None:
        sequence = _encoding()

    if with_pooling and pooling is None:
        pooling = _pooling(
            source_encoding=sequence
        )

    if computation_provenance is None:
        computation_provenance = _urban_computation(
            sequence=sequence,
            pooling=pooling,
        )

    return UrbanMemory(
        sequence_encoding=sequence,
        assembly_policy=(
            UrbanMemoryAssemblyPolicy.SEQUENCE_WITH_GENERIC_POOLING
            if pooling is not None
            else UrbanMemoryAssemblyPolicy.SEQUENCE_ONLY
        ),
        computation_provenance=computation_provenance,
        temporal_pooling=pooling,
    )


def _graph_hazard_query(
    *,
    dtype: torch.dtype = torch.float32,
    lineage: str = "hazard-query-lineage-v1",
) -> FakeHazardQueryEncoding:
    query = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=dtype,
    )
    return FakeHazardQueryEncoding(
        query=query,
        source_embedding=FakeHazardEmbeddingLookup(
            embeddings=query,
        ),
        lineage_fingerprint=lineage,
    )


def _node_hazard_query(
    urban_memory: UrbanMemory,
    *,
    lineage: str = "node-hazard-query-lineage-v1",
) -> FakeHazardQueryEncoding:
    graph_query = _graph_hazard_query(
        dtype=urban_memory.dtype,
    )
    node_values = graph_query.query[
        urban_memory.node_batch_index
    ]
    return FakeHazardQueryEncoding(
        query=node_values,
        source_embedding=FakeNodeAlignedHazardEmbeddingLookup(
            node_embeddings=node_values,
            graph_lookup=graph_query.source_embedding,
            node_batch_index=(
                urban_memory
                .node_batch_index
            ),
        ),
        lineage_fingerprint=lineage,
    )


def _aligned_node_query(
    urban_memory: UrbanMemory,
    query: FakeHazardQueryEncoding,
) -> torch.Tensor:
    if isinstance(
        query.source_embedding,
        FakeNodeAlignedHazardEmbeddingLookup,
    ):
        return query.query

    return query.query[
        urban_memory.node_batch_index
    ]


def _retrieval(
    *,
    urban_memory: UrbanMemory | None = None,
    hazard_query: FakeHazardQueryEncoding | None = None,
    node_query: torch.Tensor | None = None,
    retrieved_context: torch.Tensor | None = None,
    attention_weights: torch.Tensor | None = None,
    retrieval_kind: TemporalQueryRetrievalKind | str = (
        TemporalQueryRetrievalKind.MULTIHEAD_CROSS_ATTENTION
    ),
    head_reduction: (
        TemporalQueryRetrievalHeadReduction
        | str
    ) = TemporalQueryRetrievalHeadReduction.SINGLE_HEAD,
    zero_history_policy: (
        TemporalQueryRetrievalZeroHistoryPolicy
        | str
    ) = TemporalQueryRetrievalZeroHistoryPolicy.ERROR,
    head_reduction_weights: torch.Tensor | None = None,
    query_alignment_fingerprint: str | None = None,
    query_lineage_fingerprint: str | None = None,
    computation_provenance: (
        MemoryComputationProvenance
        | None
    ) = None,
) -> tuple[
    TemporalQueryRetrievalOutput,
    FakeHazardQueryEncoding,
    torch.Tensor,
]:
    if urban_memory is None:
        urban_memory = _urban_memory()

    if hazard_query is None:
        hazard_query = _graph_hazard_query(
            dtype=urban_memory.dtype
        )

    if node_query is None:
        node_query = _aligned_node_query(
            urban_memory,
            hazard_query,
        )

    if query_alignment_fingerprint is None:
        query_alignment_fingerprint = (
            hazard_query_alignment_fingerprint(
                source_urban_memory=urban_memory,
                source_hazard_query=hazard_query,
                node_hazard_query=node_query,
            )
        )

    if query_lineage_fingerprint is None:
        query_lineage_fingerprint = (
            hazard_query.lineage_fingerprint
        )

    if retrieved_context is None:
        retrieved_context = _pooled_values(
            urban_memory.sequence_encoding
        )

    if attention_weights is None:
        attention_weights = _single_head_weights(
            urban_memory.sequence_encoding
        )

    if computation_provenance is None:
        computation_provenance = _computation(
            source_fingerprints=(
                urban_memory
                .sequence_encoding
                .lineage_fingerprint(),
                query_lineage_fingerprint,
            ),
            architecture_name="retrieve_history",
            architecture_kind="cross_attention",
            architecture_fingerprint="retrieval-architecture-v1",
            source_history=(
                urban_memory
                .sequence_encoding
                .source_history
            ),
        )

    output = TemporalQueryRetrievalOutput(
        retrieved_context=retrieved_context,
        attention_weights=attention_weights,
        source_sequence=(
            urban_memory
            .sequence_encoding
        ),
        query_alignment_fingerprint=(
            query_alignment_fingerprint
        ),
        query_lineage_fingerprint=(
            query_lineage_fingerprint
        ),
        retrieval_kind=retrieval_kind,
        head_reduction=head_reduction,
        zero_history_policy=zero_history_policy,
        computation_provenance=(
            computation_provenance
        ),
        head_reduction_weights=(
            head_reduction_weights
        ),
    )
    return output, hazard_query, node_query


def _hazard_memory(
    *,
    urban_memory: UrbanMemory | None = None,
    hazard_query: FakeHazardQueryEncoding | None = None,
    node_query: torch.Tensor | None = None,
    retrieval: TemporalQueryRetrievalOutput | None = None,
    fused_memory: torch.Tensor | None = None,
    fusion_policy: HazardMemoryFusionPolicy | str = (
        HazardMemoryFusionPolicy.CONCAT_PROJECTION
    ),
    fusion_components: dict[str, torch.Tensor] | None = None,
    computation_provenance: (
        MemoryComputationProvenance
        | None
    ) = None,
) -> HazardQueriedMemory:
    if urban_memory is None:
        urban_memory = _urban_memory(
            with_pooling=True
        )

    if retrieval is None:
        retrieval, resolved_query, resolved_node_query = _retrieval(
            urban_memory=urban_memory,
            hazard_query=hazard_query,
            node_query=node_query,
        )
        hazard_query = resolved_query
        node_query = resolved_node_query

    assert hazard_query is not None
    assert node_query is not None

    if fused_memory is None:
        if (
            fusion_policy
            == HazardMemoryFusionPolicy.RETRIEVED_ONLY
            or fusion_policy == "retrieved_only"
        ):
            fused_memory = retrieval.retrieved_context
        else:
            fused_memory = torch.cat(
                (
                    retrieval.retrieved_context,
                    node_query[:, :1],
                ),
                dim=-1,
            )

    if fusion_components is None:
        fusion_components = {
            "retrieved_projection": (
                retrieval
                .retrieved_context
            ),
        }

    if computation_provenance is None:
        computation_provenance = _computation(
            source_fingerprints=(
                urban_memory.lineage_fingerprint(),
                hazard_query.lineage_fingerprint,
                retrieval.lineage_fingerprint(),
            ),
            architecture_name="fuse_hazard_memory",
            architecture_kind=str(
                fusion_policy
            ),
            architecture_fingerprint="hazard-fusion-architecture-v1",
            source_history=(
                urban_memory
                .sequence_encoding
                .source_history
            ),
        )

    return HazardQueriedMemory(
        source_urban_memory=urban_memory,
        source_hazard_query=hazard_query,
        node_hazard_query=node_query,
        retrieval=retrieval,
        fused_memory=fused_memory,
        fusion_policy=fusion_policy,
        computation_provenance=(
            computation_provenance
        ),
        fusion_components=(
            fusion_components
        ),
    )


# =============================================================================
# Published constants, vocabularies, and aliases
# =============================================================================


def test_output_schema_versions_are_nonempty() -> None:
    for value in (
        TEMPORAL_SEQUENCE_ENCODING_SCHEMA_VERSION,
        TEMPORAL_POOLING_OUTPUT_SCHEMA_VERSION,
        URBAN_MEMORY_SCHEMA_VERSION,
        TEMPORAL_QUERY_RETRIEVAL_OUTPUT_SCHEMA_VERSION,
        HAZARD_QUERIED_MEMORY_SCHEMA_VERSION,
    ):
        assert isinstance(
            value,
            str,
        )
        assert value.strip()


def test_output_aliases_preserve_exact_classes() -> None:
    assert SequenceEncoding is TemporalSequenceEncoding
    assert SharedTemporalSequenceEncoding is TemporalSequenceEncoding
    assert PoolingOutput is TemporalPoolingOutput
    assert TemporalMemoryPoolingOutput is TemporalPoolingOutput
    assert UrbanTemporalMemory is UrbanMemory
    assert SharedUrbanMemory is UrbanMemory
    assert TemporalRetrievalOutput is TemporalQueryRetrievalOutput
    assert QueryRetrievalOutput is TemporalQueryRetrievalOutput
    assert HazardConditionedMemory is HazardQueriedMemory


def test_output_vocabularies_match_enums() -> None:
    assert CANONICAL_TEMPORAL_SEQUENCE_ENCODER_KINDS == tuple(
        value.value
        for value in TemporalSequenceEncoderKind
    )
    assert CANONICAL_TEMPORAL_POOLING_KINDS == tuple(
        value.value
        for value in TemporalPoolingKind
    )
    assert CANONICAL_TEMPORAL_POOLING_HEAD_REDUCTIONS == tuple(
        value.value
        for value in TemporalPoolingHeadReduction
    )
    assert CANONICAL_TEMPORAL_POOLING_ZERO_HISTORY_POLICIES == tuple(
        value.value
        for value in TemporalPoolingZeroHistoryPolicy
    )
    assert CANONICAL_URBAN_MEMORY_ASSEMBLY_POLICIES == tuple(
        value.value
        for value in UrbanMemoryAssemblyPolicy
    )
    assert CANONICAL_TEMPORAL_QUERY_RETRIEVAL_KINDS == tuple(
        value.value
        for value in TemporalQueryRetrievalKind
    )
    assert CANONICAL_TEMPORAL_QUERY_RETRIEVAL_HEAD_REDUCTIONS == tuple(
        value.value
        for value in TemporalQueryRetrievalHeadReduction
    )
    assert CANONICAL_TEMPORAL_QUERY_RETRIEVAL_ZERO_HISTORY_POLICIES == tuple(
        value.value
        for value in TemporalQueryRetrievalZeroHistoryPolicy
    )
    assert CANONICAL_HAZARD_QUERY_ALIGNMENT_SCOPES == tuple(
        value.value
        for value in HazardQueryAlignmentScope
    )
    assert CANONICAL_HAZARD_MEMORY_FUSION_POLICIES == tuple(
        value.value
        for value in HazardMemoryFusionPolicy
    )


def test_output_interpretation_constants_are_explicit() -> None:
    for value in (
        TEMPORAL_SEQUENCE_ENCODING_VALUE_SEMANTICS,
        TEMPORAL_SEQUENCE_ENCODING_PADDING_POLICY,
        TEMPORAL_SEQUENCE_ENCODING_SCIENTIFIC_INTERPRETATION,
        TEMPORAL_POOLING_WEIGHT_SEMANTICS,
        TEMPORAL_POOLING_PADDING_POLICY,
        TEMPORAL_POOLING_NORMALIZATION_POLICY,
        TEMPORAL_POOLING_SCIENTIFIC_INTERPRETATION,
        URBAN_MEMORY_SEQUENCE_PRESERVATION_POLICY,
        URBAN_MEMORY_POOLING_SCOPE,
        URBAN_MEMORY_SCIENTIFIC_INTERPRETATION,
        TEMPORAL_QUERY_RETRIEVAL_WEIGHT_SEMANTICS,
        TEMPORAL_QUERY_RETRIEVAL_PADDING_POLICY,
        TEMPORAL_QUERY_RETRIEVAL_NORMALIZATION_POLICY,
        HAZARD_QUERY_ALIGNMENT_POLICY,
        HAZARD_QUERIED_MEMORY_SCIENTIFIC_INTERPRETATION,
    ):
        assert isinstance(
            value,
            str,
        )
        assert value.strip()


# =============================================================================
# TemporalSequenceEncoding
# =============================================================================


def test_sequence_encoding_preserves_exact_source_contract() -> None:
    history = _history()
    encoding = _encoding(
        source_history=history
    )

    assert encoding.source_history is history
    assert encoding.timestep_mask is history.timestep_mask
    assert encoding.node_axis is history.node_axis
    assert encoding.feature_axis is history.feature_axis
    assert encoding.temporal_coordinates is history.temporal_coordinates
    assert encoding.source_provenance is history.source_provenance
    assert encoding.node_ids == history.node_ids
    assert encoding.node_batch_index is history.node_batch_index
    assert encoding.graph_ids == history.graph_ids


def test_sequence_encoding_structural_properties() -> None:
    encoding = _encoding()

    assert encoding.node_count == N
    assert encoding.item_count == N
    assert encoding.sequence_length == T
    assert encoding.hidden_dim == H
    assert encoding.encoded_shape == (
        N,
        T,
        H,
    )
    assert encoding.device == torch.device(
        "cpu"
    )
    assert encoding.dtype == torch.float32
    assert encoding.encoder_kind == TemporalSequenceEncoderKind.GRU
    assert not encoding.has_zero_history


def test_sequence_encoding_has_no_generic_final_state() -> None:
    encoding = _encoding()

    assert not hasattr(
        encoding,
        "final_state",
    )
    assert not hasattr(
        encoding,
        "hidden_state",
    )
    assert not hasattr(
        encoding,
        "cell_state",
    )


def test_sequence_encoding_provenance_properties() -> None:
    encoding = _encoding()

    assert encoding.architecture_fingerprint == (
        "encoder-architecture-v1"
    )
    assert encoding.parameter_snapshot_fingerprint is None
    assert encoding.computation_lineage_fingerprint == (
        encoding
        .computation_provenance
        .lineage_fingerprint
    )


def test_sequence_encoding_semantic_dictionary() -> None:
    encoding = _encoding()
    payload = encoding.semantic_dict()

    assert payload["encoder_kind"] == "gru"
    assert payload["encoded_shape"] == [
        N,
        T,
        H,
    ]
    assert payload["value_semantics"] == (
        TEMPORAL_SEQUENCE_ENCODING_VALUE_SEMANTICS
    )
    assert payload["padding_policy"] == (
        TEMPORAL_SEQUENCE_ENCODING_PADDING_POLICY
    )
    assert payload["scientific_interpretation"] == (
        TEMPORAL_SEQUENCE_ENCODING_SCIENTIFIC_INTERPRETATION
    )


def test_sequence_encoding_fingerprints_are_deterministic() -> None:
    first = _encoding()
    second = _encoding()

    assert first.alignment_fingerprint() == second.alignment_fingerprint()
    assert first.semantic_fingerprint() == second.semantic_fingerprint()
    assert first.value_fingerprint() == second.value_fingerprint()
    assert first.lineage_fingerprint() == second.lineage_fingerprint()
    assert first.fingerprint() == first.lineage_fingerprint()


def test_sequence_value_change_preserves_alignment_but_changes_lineage() -> None:
    first = _encoding()
    changed = first.encoded_sequence.clone()
    changed[0, 0, 0] += 1.0
    second = _encoding(
        source_history=first.source_history,
        encoded_sequence=changed,
        computation_provenance=(
            first.computation_provenance
        ),
    )

    assert first.alignment_fingerprint() == second.alignment_fingerprint()
    assert first.value_fingerprint() != second.value_fingerprint()
    assert first.lineage_fingerprint() != second.lineage_fingerprint()


@pytest.mark.parametrize(
    "value",
    [
        torch.zeros(
            (N, T),
            dtype=torch.float32,
        ),
        torch.zeros(
            (N, T, H),
            dtype=torch.long,
        ),
        torch.zeros(
            (0, T, H),
            dtype=torch.float32,
        ),
        torch.zeros(
            (N, 0, H),
            dtype=torch.float32,
        ),
        torch.zeros(
            (N, T, 0),
            dtype=torch.float32,
        ),
    ],
)
def test_sequence_encoding_rejects_invalid_tensor(
    value: torch.Tensor,
) -> None:
    with pytest.raises(
        (TypeError, ValueError),
    ):
        _encoding(
            encoded_sequence=value
        )


@pytest.mark.parametrize(
    "nonfinite",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_sequence_encoding_rejects_nonfinite_values(
    nonfinite: float,
) -> None:
    values = _encoded_values(
        _history()
    )
    values[0, 0, 0] = nonfinite

    with pytest.raises(ValueError):
        _encoding(
            encoded_sequence=values
        )


def test_sequence_encoding_rejects_shape_mismatch() -> None:
    history = _history()

    with pytest.raises(ValueError):
        _encoding(
            source_history=history,
            encoded_sequence=torch.zeros(
                (
                    N + 1,
                    T,
                    H,
                ),
                dtype=torch.float32,
            ),
        )

    with pytest.raises(ValueError):
        _encoding(
            source_history=history,
            encoded_sequence=torch.zeros(
                (
                    N,
                    T + 1,
                    H,
                ),
                dtype=torch.float32,
            ),
        )


def test_sequence_encoding_rejects_nonzero_padding() -> None:
    history = _history()
    values = _encoded_values(
        history
    )
    values[0, 3, 0] = 1.0

    with pytest.raises(ValueError):
        _encoding(
            source_history=history,
            encoded_sequence=values,
        )


def test_zero_history_sequence_row_must_be_zero() -> None:
    history = _history(
        mask=_zero_history_mask(),
        zero_history_policy=(
            HistoryZeroLengthPolicy.ALLOW_ZERO_HISTORY
        ),
    )
    values = _encoded_values(
        history
    )
    values[1, 0, 0] = 1.0

    with pytest.raises(ValueError):
        _encoding(
            source_history=history,
            encoded_sequence=values,
        )


def test_sequence_encoding_rejects_wrong_source_type() -> None:
    encoding = _encoding()

    with pytest.raises(TypeError):
        TemporalSequenceEncoding(
            encoded_sequence=(
                encoding
                .encoded_sequence
            ),
            source_history=object(),  # type: ignore[arg-type]
            encoder_kind="gru",
            computation_provenance=(
                encoding
                .computation_provenance
            ),
        )


def test_sequence_encoding_requires_source_lineage() -> None:
    history = _history()
    bad = _computation(
        source_fingerprints=(
            "not-the-history",
        ),
        architecture_name="encode_history",
        architecture_kind="gru",
        architecture_fingerprint="encoder-architecture-v1",
        source_history=history,
    )

    with pytest.raises(ValueError):
        _encoding(
            source_history=history,
            computation_provenance=bad,
        )


@pytest.mark.parametrize(
    (
        "axis_name",
        "bad_value",
    ),
    [
        (
            "node_axis_fingerprint",
            "wrong-node-axis",
        ),
        (
            "temporal_axis_fingerprint",
            "wrong-temporal-axis",
        ),
        (
            "feature_axis_fingerprint",
            "wrong-feature-axis",
        ),
    ],
)
def test_sequence_encoding_rejects_lineage_axis_mismatch(
    axis_name: str,
    bad_value: str,
) -> None:
    history = _history()
    architecture = _architecture(
        name="encode_history",
        kind="gru",
        fingerprint="encoder-architecture-v1",
    )
    kwargs = {
        "operation_name": "encode_history",
        "source_lineage_fingerprints": (
            history.lineage_fingerprint(),
        ),
        "architecture_fingerprint": (
            "encoder-architecture-v1"
        ),
        "node_axis_fingerprint": (
            history.node_axis.fingerprint()
        ),
        "temporal_axis_fingerprint": (
            history.temporal_alignment_fingerprint()
        ),
        "feature_axis_fingerprint": (
            history.feature_axis.fingerprint()
        ),
    }
    kwargs[axis_name] = bad_value
    lineage = MemoryExecutionLineage(
        **kwargs,
    )
    computation = MemoryComputationProvenance(
        architecture=architecture,
        lineage=lineage,
    )

    with pytest.raises(ValueError):
        _encoding(
            source_history=history,
            computation_provenance=computation,
        )


def test_sequence_encoding_is_frozen_and_replace_revalidates() -> None:
    encoding = _encoding()

    with pytest.raises(
        (FrozenInstanceError, AttributeError),
    ):
        encoding.encoding_name = "changed"  # type: ignore[misc]

    renamed = encoding.replace(
        encoding_name="renamed"
    )
    assert renamed.encoding_name == "renamed"
    assert renamed.source_history is encoding.source_history

    with pytest.raises(ValueError):
        encoding.replace(
            encoding_name=""
        )


def test_sequence_encoding_to_same_device_preserves_fingerprint() -> None:
    encoding = _encoding()
    moved = encoding.to(
        "cpu"
    )

    assert moved is not encoding
    assert moved.source_history is not encoding.source_history
    assert moved.fingerprint() == encoding.fingerprint()


# =============================================================================
# TemporalPoolingOutput
# =============================================================================


def test_pooling_preserves_exact_source_encoding() -> None:
    sequence = _encoding()
    pooling = _pooling(
        source_encoding=sequence
    )

    assert pooling.source_encoding is sequence
    assert pooling.source_sequence_encoding is sequence
    assert pooling.encoded_sequence is sequence.encoded_sequence
    assert pooling.timestep_mask is sequence.timestep_mask
    assert pooling.node_axis is sequence.node_axis
    assert pooling.temporal_coordinates is sequence.temporal_coordinates


def test_pooling_structural_properties() -> None:
    pooling = _pooling()

    assert pooling.node_count == N
    assert pooling.item_count == N
    assert pooling.pooled_dim == H
    assert pooling.output_dim == H
    assert pooling.num_heads == 1
    assert pooling.sequence_length == T
    assert pooling.pooled_shape == (
        N,
        H,
    )
    assert pooling.weight_shape == (
        N,
        1,
        T,
    )
    assert pooling.pooling_kind == TemporalPoolingKind.MASKED_MEAN
    assert pooling.head_reduction == (
        TemporalPoolingHeadReduction.SINGLE_HEAD
    )


def test_pooling_weights_are_normalized_over_valid_history() -> None:
    pooling = _pooling()
    mask = pooling.timestep_mask
    weights = pooling.pooling_weights

    assert torch.equal(
        weights[
            ~mask.unsqueeze(1).expand_as(
                weights
            )
        ],
        torch.zeros_like(
            weights[
                ~mask.unsqueeze(1).expand_as(
                    weights
                )
            ]
        ),
    )
    assert torch.allclose(
        weights.sum(dim=-1),
        torch.ones(
            (N, 1),
            dtype=weights.dtype,
        ),
    )


def test_multihead_weighted_pooling_contract() -> None:
    sequence = _encoding()
    output = _pooling(
        source_encoding=sequence,
        pooling_weights=_multihead_weights(
            sequence,
            heads=2,
        ),
        head_reduction=(
            TemporalPoolingHeadReduction.WEIGHTED_MEAN
        ),
        head_reduction_weights=torch.tensor(
            [0.25, 0.75],
            dtype=sequence.dtype,
        ),
    )

    assert output.num_heads == 2
    assert output.head_reduction_weights is not None
    assert torch.equal(
        output.head_reduction_weights,
        torch.tensor(
            [0.25, 0.75],
            dtype=sequence.dtype,
        ),
    )


def test_pooling_semantic_dictionary() -> None:
    output = _pooling()
    payload = output.semantic_dict()

    assert payload["pooling_kind"] == "masked_mean"
    assert payload["head_reduction"] == "single_head"
    assert payload["zero_history_policy"] == "error"
    assert payload["weight_semantics"] == (
        TEMPORAL_POOLING_WEIGHT_SEMANTICS
    )
    assert payload["padding_policy"] == (
        TEMPORAL_POOLING_PADDING_POLICY
    )
    assert payload["normalization_policy"] == (
        TEMPORAL_POOLING_NORMALIZATION_POLICY
    )


def test_pooling_fingerprints_are_deterministic() -> None:
    first = _pooling()
    second = _pooling()

    assert first.semantic_fingerprint() == second.semantic_fingerprint()
    assert first.value_fingerprint() == second.value_fingerprint()
    assert first.lineage_fingerprint() == second.lineage_fingerprint()


@pytest.mark.parametrize(
    "pooled_memory",
    [
        torch.zeros(
            (N, H, 1),
            dtype=torch.float32,
        ),
        torch.zeros(
            (N, H),
            dtype=torch.long,
        ),
        torch.zeros(
            (0, H),
            dtype=torch.float32,
        ),
        torch.tensor(
            [
                [float("nan"), 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        ),
    ],
)
def test_pooling_rejects_invalid_pooled_memory(
    pooled_memory: torch.Tensor,
) -> None:
    with pytest.raises(
        (TypeError, ValueError),
    ):
        _pooling(
            pooled_memory=pooled_memory
        )


@pytest.mark.parametrize(
    "weights",
    [
        torch.zeros(
            (N, T),
            dtype=torch.float32,
        ),
        torch.zeros(
            (N, 1, T),
            dtype=torch.long,
        ),
        torch.zeros(
            (0, 1, T),
            dtype=torch.float32,
        ),
        torch.full(
            (N, 1, T),
            -0.1,
            dtype=torch.float32,
        ),
    ],
)
def test_pooling_rejects_invalid_weights(
    weights: torch.Tensor,
) -> None:
    with pytest.raises(
        (TypeError, ValueError),
    ):
        _pooling(
            pooling_weights=weights
        )


def test_pooling_rejects_nonzero_padding_mass() -> None:
    sequence = _encoding()
    weights = _single_head_weights(
        sequence
    )
    weights[0, 0, 3] = 0.1
    weights[0, 0, 0] -= 0.1

    with pytest.raises(ValueError):
        _pooling(
            source_encoding=sequence,
            pooling_weights=weights,
        )


def test_pooling_rejects_nonunit_mass() -> None:
    sequence = _encoding()
    weights = _single_head_weights(
        sequence
    )
    weights[0, 0, :3] *= 0.5

    with pytest.raises(ValueError):
        _pooling(
            source_encoding=sequence,
            pooling_weights=weights,
        )


def test_single_head_reduction_requires_one_head() -> None:
    sequence = _encoding()

    with pytest.raises(ValueError):
        _pooling(
            source_encoding=sequence,
            pooling_weights=_multihead_weights(
                sequence
            ),
            head_reduction="single_head",
        )


def test_multihead_reduction_requires_multiple_heads() -> None:
    with pytest.raises(ValueError):
        _pooling(
            head_reduction="mean",
        )


def test_weighted_mean_requires_normalized_head_weights() -> None:
    sequence = _encoding()
    multi = _multihead_weights(
        sequence
    )

    with pytest.raises(ValueError):
        _pooling(
            source_encoding=sequence,
            pooling_weights=multi,
            head_reduction="weighted_mean",
            head_reduction_weights=None,
        )

    with pytest.raises(ValueError):
        _pooling(
            source_encoding=sequence,
            pooling_weights=multi,
            head_reduction="weighted_mean",
            head_reduction_weights=torch.tensor(
                [0.2, 0.2],
                dtype=sequence.dtype,
            ),
        )


def test_head_weights_forbidden_for_nonweighted_reduction() -> None:
    sequence = _encoding()

    with pytest.raises(ValueError):
        _pooling(
            source_encoding=sequence,
            pooling_weights=_multihead_weights(
                sequence
            ),
            head_reduction="mean",
            head_reduction_weights=torch.tensor(
                [0.5, 0.5],
                dtype=sequence.dtype,
            ),
        )


def test_pooling_zero_history_error_policy_rejects() -> None:
    history = _history(
        mask=_zero_history_mask(),
        zero_history_policy="allow_zero_history",
    )
    sequence = _encoding(
        source_history=history
    )

    with pytest.raises(ValueError):
        _pooling(
            source_encoding=sequence,
            zero_history_policy="error",
        )


def test_pooling_zero_policy_accepts_zero_row() -> None:
    history = _history(
        mask=_zero_history_mask(),
        zero_history_policy="allow_zero_history",
    )
    sequence = _encoding(
        source_history=history
    )
    output = _pooling(
        source_encoding=sequence,
        zero_history_policy="zero",
    )

    assert torch.equal(
        output.pooling_weights[1],
        torch.zeros_like(
            output.pooling_weights[1]
        ),
    )
    assert torch.equal(
        output.pooled_memory[1],
        torch.zeros_like(
            output.pooled_memory[1]
        ),
    )


def test_pooling_zero_policy_rejects_nonzero_fallback() -> None:
    history = _history(
        mask=_zero_history_mask(),
        zero_history_policy="allow_zero_history",
    )
    sequence = _encoding(
        source_history=history
    )
    pooled = _pooled_values(
        sequence
    )
    pooled[1] = 1.0

    with pytest.raises(ValueError):
        _pooling(
            source_encoding=sequence,
            pooled_memory=pooled,
            zero_history_policy="zero",
        )


def test_pooling_learned_fallback_accepts_nonzero_context() -> None:
    history = _history(
        mask=_zero_history_mask(),
        zero_history_policy="allow_zero_history",
    )
    sequence = _encoding(
        source_history=history
    )
    pooled = _pooled_values(
        sequence
    )
    pooled[1] = 1.0

    output = _pooling(
        source_encoding=sequence,
        pooled_memory=pooled,
        zero_history_policy="learned_fallback",
    )

    assert torch.equal(
        output.pooled_memory[1],
        torch.ones_like(
            output.pooled_memory[1]
        ),
    )


def test_pooling_requires_source_lineage() -> None:
    sequence = _encoding()
    bad = _computation(
        source_fingerprints=(
            "wrong-source",
        ),
        architecture_name="pool_history",
        architecture_kind="masked_mean",
        architecture_fingerprint="pooling-architecture-v1",
        source_history=sequence.source_history,
    )

    with pytest.raises(ValueError):
        _pooling(
            source_encoding=sequence,
            computation_provenance=bad,
        )


def test_pooling_is_frozen_and_replace_revalidates() -> None:
    pooling = _pooling()

    with pytest.raises(
        (FrozenInstanceError, AttributeError),
    ):
        pooling.pooling_name = "changed"  # type: ignore[misc]

    renamed = pooling.replace(
        pooling_name="renamed"
    )
    assert renamed.pooling_name == "renamed"
    assert renamed.source_encoding is pooling.source_encoding


def test_pooling_to_preserves_source_relationship_and_fingerprint() -> None:
    pooling = _pooling()
    moved = pooling.to(
        "cpu"
    )

    assert moved.source_encoding is not pooling.source_encoding
    assert moved.encoded_sequence is moved.source_encoding.encoded_sequence
    assert moved.fingerprint() == pooling.fingerprint()


# =============================================================================
# UrbanMemory
# =============================================================================


def test_sequence_only_urban_memory_preserves_sequence() -> None:
    sequence = _encoding()
    memory = _urban_memory(
        sequence=sequence
    )

    assert memory.sequence_encoding is sequence
    assert memory.encoded_sequence is sequence.encoded_sequence
    assert memory.temporal_pooling is None
    assert memory.generic_pooled_memory is None
    assert memory.temporal_pooling_weights is None
    assert not memory.has_temporal_pooling
    assert memory.assembly_policy == (
        UrbanMemoryAssemblyPolicy.SEQUENCE_ONLY
    )


def test_pooled_urban_memory_preserves_exact_sequence_and_pooling() -> None:
    sequence = _encoding()
    pooling = _pooling(
        source_encoding=sequence
    )
    memory = _urban_memory(
        sequence=sequence,
        pooling=pooling,
    )

    assert memory.sequence_encoding is sequence
    assert memory.temporal_pooling is pooling
    assert memory.temporal_pooling.source_encoding is sequence
    assert memory.encoded_sequence is sequence.encoded_sequence
    assert memory.generic_pooled_memory is pooling.pooled_memory
    assert memory.temporal_pooling_weights is pooling.pooling_weights
    assert memory.has_temporal_pooling
    assert memory.has_generic_pooled_memory


def test_urban_memory_structural_properties() -> None:
    memory = _urban_memory(
        with_pooling=True
    )

    assert memory.node_count == N
    assert memory.item_count == N
    assert memory.sequence_length == T
    assert memory.sequence_hidden_dim == H
    assert memory.encoded_shape == (
        N,
        T,
        H,
    )
    assert memory.pooled_dim == H
    assert memory.dtype == torch.float32


def test_urban_memory_sequence_only_rejects_pooling() -> None:
    sequence = _encoding()
    pooling = _pooling(
        source_encoding=sequence
    )
    computation = _urban_computation(
        sequence=sequence,
        pooling=pooling,
    )

    with pytest.raises(ValueError):
        UrbanMemory(
            sequence_encoding=sequence,
            assembly_policy="sequence_only",
            computation_provenance=computation,
            temporal_pooling=pooling,
        )


def test_urban_memory_pooled_policy_requires_pooling() -> None:
    sequence = _encoding()
    computation = _urban_computation(
        sequence=sequence,
        pooling=None,
    )

    with pytest.raises(ValueError):
        UrbanMemory(
            sequence_encoding=sequence,
            assembly_policy="sequence_with_generic_pooling",
            computation_provenance=computation,
        )


def test_urban_memory_rejects_equivalent_but_distinct_pooling_source() -> None:
    first_sequence = _encoding()
    second_sequence = _encoding()
    pooling = _pooling(
        source_encoding=second_sequence
    )
    computation = _urban_computation(
        sequence=first_sequence,
        pooling=pooling,
    )

    assert (
        first_sequence.fingerprint()
        == second_sequence.fingerprint()
    )
    assert first_sequence is not second_sequence

    with pytest.raises(ValueError):
        UrbanMemory(
            sequence_encoding=first_sequence,
            assembly_policy="sequence_with_generic_pooling",
            computation_provenance=computation,
            temporal_pooling=pooling,
        )


def test_urban_memory_requires_all_source_lineages() -> None:
    sequence = _encoding()
    pooling = _pooling(
        source_encoding=sequence
    )
    bad = _computation(
        source_fingerprints=(
            sequence.lineage_fingerprint(),
        ),
        architecture_name="assemble_urban_memory",
        architecture_kind="sequence_with_generic_pooling",
        architecture_fingerprint="urban-memory-architecture-v1",
        source_history=sequence.source_history,
    )

    with pytest.raises(ValueError):
        _urban_memory(
            sequence=sequence,
            pooling=pooling,
            computation_provenance=bad,
        )


def test_urban_memory_semantic_and_value_fingerprints() -> None:
    sequence_only = _urban_memory()
    pooled = _urban_memory(
        with_pooling=True
    )

    assert sequence_only.semantic_fingerprint() != (
        pooled.semantic_fingerprint()
    )
    assert sequence_only.value_fingerprint() != (
        pooled.value_fingerprint()
    )
    assert sequence_only.alignment_fingerprint() == (
        pooled.alignment_fingerprint()
    )


def test_urban_memory_is_frozen_and_replace_revalidates_identity() -> None:
    memory = _urban_memory(
        with_pooling=True
    )

    with pytest.raises(
        (FrozenInstanceError, AttributeError),
    ):
        memory.memory_name = "changed"  # type: ignore[misc]

    with pytest.raises(ValueError):
        memory.replace(
            sequence_encoding=_encoding()
        )


def test_urban_memory_to_preserves_exact_moved_identity() -> None:
    memory = _urban_memory(
        with_pooling=True
    )
    moved = memory.to(
        "cpu"
    )

    assert moved.sequence_encoding is (
        moved.temporal_pooling.source_encoding
    )
    assert moved.fingerprint() == memory.fingerprint()


# =============================================================================
# TemporalQueryRetrievalOutput
# =============================================================================


def test_retrieval_preserves_exact_source_sequence() -> None:
    urban = _urban_memory()
    retrieval, _, _ = _retrieval(
        urban_memory=urban
    )

    assert retrieval.source_sequence is urban.sequence_encoding
    assert retrieval.source_sequence_encoding is urban.sequence_encoding
    assert retrieval.encoded_sequence is urban.encoded_sequence
    assert retrieval.timestep_mask is urban.timestep_mask
    assert retrieval.node_axis is urban.node_axis


def test_retrieval_structural_properties() -> None:
    retrieval, _, _ = _retrieval()

    assert retrieval.node_count == N
    assert retrieval.item_count == N
    assert retrieval.retrieval_dim == H
    assert retrieval.output_dim == H
    assert retrieval.num_heads == 1
    assert retrieval.sequence_length == T
    assert retrieval.context_shape == (
        N,
        H,
    )
    assert retrieval.weight_shape == (
        N,
        1,
        T,
    )
    assert retrieval.retrieval_kind == (
        TemporalQueryRetrievalKind.MULTIHEAD_CROSS_ATTENTION
    )


def test_retrieval_fingerprints_are_deterministic() -> None:
    first, _, _ = _retrieval()
    second, _, _ = _retrieval()

    assert first.semantic_fingerprint() == second.semantic_fingerprint()
    assert first.value_fingerprint() == second.value_fingerprint()
    assert first.lineage_fingerprint() == second.lineage_fingerprint()


@pytest.mark.parametrize(
    "context",
    [
        torch.zeros(
            (N, H, 1),
            dtype=torch.float32,
        ),
        torch.zeros(
            (N, H),
            dtype=torch.long,
        ),
        torch.zeros(
            (0, H),
            dtype=torch.float32,
        ),
    ],
)
def test_retrieval_rejects_invalid_context(
    context: torch.Tensor,
) -> None:
    with pytest.raises(
        (TypeError, ValueError),
    ):
        _retrieval(
            retrieved_context=context
        )


def test_retrieval_rejects_nonzero_padding_and_nonunit_mass() -> None:
    urban = _urban_memory()
    weights = _single_head_weights(
        urban.sequence_encoding
    )
    weights[0, 0, 3] = 0.1
    weights[0, 0, 0] -= 0.1

    with pytest.raises(ValueError):
        _retrieval(
            urban_memory=urban,
            attention_weights=weights,
        )

    weights = _single_head_weights(
        urban.sequence_encoding
    )
    weights[0, 0, :3] *= 0.5

    with pytest.raises(ValueError):
        _retrieval(
            urban_memory=urban,
            attention_weights=weights,
        )


def test_retrieval_head_reduction_rules() -> None:
    urban = _urban_memory()
    multi = _multihead_weights(
        urban.sequence_encoding
    )

    with pytest.raises(ValueError):
        _retrieval(
            urban_memory=urban,
            attention_weights=multi,
            head_reduction="single_head",
        )

    with pytest.raises(ValueError):
        _retrieval(
            urban_memory=urban,
            head_reduction="mean",
        )

    output, _, _ = _retrieval(
        urban_memory=urban,
        attention_weights=multi,
        head_reduction="weighted_mean",
        head_reduction_weights=torch.tensor(
            [0.4, 0.6],
            dtype=urban.dtype,
        ),
    )
    assert output.num_heads == 2


def test_retrieval_zero_history_policies() -> None:
    history = _history(
        mask=_zero_history_mask(),
        zero_history_policy="allow_zero_history",
    )
    sequence = _encoding(
        source_history=history
    )
    urban = _urban_memory(
        sequence=sequence
    )

    with pytest.raises(ValueError):
        _retrieval(
            urban_memory=urban,
            zero_history_policy="error",
        )

    zero_output, _, _ = _retrieval(
        urban_memory=urban,
        zero_history_policy="zero",
    )
    assert torch.equal(
        zero_output.retrieved_context[1],
        torch.zeros_like(
            zero_output.retrieved_context[1]
        ),
    )

    fallback = _pooled_values(
        sequence
    )
    fallback[1] = 1.0
    learned_output, _, _ = _retrieval(
        urban_memory=urban,
        retrieved_context=fallback,
        zero_history_policy="learned_fallback",
    )
    assert torch.equal(
        learned_output.retrieved_context[1],
        torch.ones_like(
            learned_output.retrieved_context[1]
        ),
    )


def test_retrieval_requires_query_and_sequence_lineages() -> None:
    urban = _urban_memory()
    query = _graph_hazard_query()
    node_query = _aligned_node_query(
        urban,
        query,
    )
    alignment = hazard_query_alignment_fingerprint(
        source_urban_memory=urban,
        source_hazard_query=query,
        node_hazard_query=node_query,
    )
    bad = _computation(
        source_fingerprints=(
            urban
            .sequence_encoding
            .lineage_fingerprint(),
        ),
        architecture_name="retrieve_history",
        architecture_kind="cross_attention",
        architecture_fingerprint="retrieval-architecture-v1",
        source_history=urban.sequence_encoding.source_history,
    )

    with pytest.raises(ValueError):
        _retrieval(
            urban_memory=urban,
            hazard_query=query,
            node_query=node_query,
            query_alignment_fingerprint=alignment,
            computation_provenance=bad,
        )


def test_retrieval_is_frozen_and_to_preserves_fingerprint() -> None:
    retrieval, _, _ = _retrieval()

    with pytest.raises(
        (FrozenInstanceError, AttributeError),
    ):
        retrieval.retrieval_name = "changed"  # type: ignore[misc]

    moved = retrieval.to(
        "cpu"
    )
    assert moved.fingerprint() == retrieval.fingerprint()


def test_retrieval_to_accepts_matching_injected_source() -> None:
    urban = _urban_memory()
    retrieval, _, _ = _retrieval(
        urban_memory=urban
    )
    moved_source = (
        urban
        .sequence_encoding
        .to("cpu")
    )
    moved = retrieval.to(
        "cpu",
        source_sequence=moved_source,
    )

    assert moved.source_sequence is moved_source


def test_retrieval_to_rejects_different_source_lineage() -> None:
    retrieval, _, _ = _retrieval()
    changed_history = _history()
    changed_values = changed_history.history.clone()
    changed_values[0, 0, 0] += 1.0
    changed_history = changed_history.replace(
        history=changed_values
    )
    different = _encoding(
        source_history=changed_history
    )

    with pytest.raises(ValueError):
        retrieval.to(
            "cpu",
            source_sequence=different,
        )


# =============================================================================
# Hazard query alignment
# =============================================================================


def test_graph_query_alignment_is_explicit_broadcast() -> None:
    urban = _urban_memory()
    query = _graph_hazard_query()
    node_query = _aligned_node_query(
        urban,
        query,
    )

    assert torch.equal(
        node_query,
        query.query[
            urban.node_batch_index
        ],
    )
    fingerprint = hazard_query_alignment_fingerprint(
        source_urban_memory=urban,
        source_hazard_query=query,
        node_hazard_query=node_query,
    )
    assert fingerprint


def test_node_query_alignment_preserves_membership() -> None:
    urban = _urban_memory()
    query = _node_hazard_query(
        urban
    )

    fingerprint = hazard_query_alignment_fingerprint(
        source_urban_memory=urban,
        source_hazard_query=query,
        node_hazard_query=query.query,
    )
    assert fingerprint


def test_alignment_rejects_implicit_or_wrong_query_values() -> None:
    urban = _urban_memory()
    query = _graph_hazard_query()
    wrong = torch.zeros(
        (
            N,
            QUERY_DIM,
        ),
        dtype=urban.dtype,
    )

    with pytest.raises(ValueError):
        hazard_query_alignment_fingerprint(
            source_urban_memory=urban,
            source_hazard_query=query,
            node_hazard_query=wrong,
        )


def test_node_alignment_rejects_membership_mismatch() -> None:
    urban = _urban_memory()
    query = _node_hazard_query(
        urban
    )
    query.source_embedding.node_batch_index = torch.tensor(
        [0, 1, 1],
        dtype=torch.long,
    )

    with pytest.raises(ValueError):
        hazard_query_alignment_fingerprint(
            source_urban_memory=urban,
            source_hazard_query=query,
            node_hazard_query=query.query,
        )


def test_graph_alignment_rejects_wrong_graph_count() -> None:
    urban = _urban_memory()
    values = torch.tensor(
        [[1.0, 0.0]],
        dtype=urban.dtype,
    )
    query = FakeHazardQueryEncoding(
        query=values,
        source_embedding=FakeHazardEmbeddingLookup(
            embeddings=values
        ),
    )
    node_query = values[
        torch.zeros(
            N,
            dtype=torch.long,
        )
    ]

    with pytest.raises(ValueError):
        hazard_query_alignment_fingerprint(
            source_urban_memory=urban,
            source_hazard_query=query,
            node_hazard_query=node_query,
        )


# =============================================================================
# HazardQueriedMemory
# =============================================================================


def test_hazard_memory_preserves_all_exact_source_objects() -> None:
    urban = _urban_memory(
        with_pooling=True
    )
    retrieval, query, node_query = _retrieval(
        urban_memory=urban
    )
    output = _hazard_memory(
        urban_memory=urban,
        hazard_query=query,
        node_query=node_query,
        retrieval=retrieval,
    )

    assert output.source_urban_memory is urban
    assert output.source_hazard_query is query
    assert output.retrieval is retrieval
    assert output.retrieval.source_sequence is urban.sequence_encoding
    assert output.sequence_encoding is urban.sequence_encoding
    assert output.encoded_sequence is urban.encoded_sequence
    assert output.generic_pooled_memory is urban.generic_pooled_memory
    assert output.retrieved_context is retrieval.retrieved_context
    assert output.temporal_retrieval_weights is retrieval.attention_weights


def test_hazard_memory_keeps_retrieval_and_fusion_distinct() -> None:
    output = _hazard_memory()

    assert output.retrieved_context.shape == (
        N,
        H,
    )
    assert output.fused_memory.shape == (
        N,
        H + 1,
    )
    assert output.retrieved_context is not output.fused_memory
    assert output.retrieval_dim == H
    assert output.fused_dim == H + 1


def test_graph_scoped_hazard_memory_reports_alignment_scope() -> None:
    output = _hazard_memory()

    assert output.alignment_scope == (
        HazardQueryAlignmentScope.GRAPH_BROADCAST
    )


def test_node_scoped_hazard_memory_reports_alignment_scope() -> None:
    urban = _urban_memory()
    query = _node_hazard_query(
        urban
    )
    retrieval, _, node_query = _retrieval(
        urban_memory=urban,
        hazard_query=query,
    )
    output = _hazard_memory(
        urban_memory=urban,
        hazard_query=query,
        node_query=node_query,
        retrieval=retrieval,
    )

    assert output.alignment_scope == HazardQueryAlignmentScope.NODE


def test_retrieved_only_policy_requires_exact_equality() -> None:
    valid = _hazard_memory(
        fusion_policy="retrieved_only",
    )
    assert valid.fused_memory is valid.retrieved_context

    urban = _urban_memory()
    retrieval, query, node_query = _retrieval(
        urban_memory=urban
    )
    wrong = retrieval.retrieved_context.clone()
    wrong[0, 0] += 1.0

    with pytest.raises(ValueError):
        _hazard_memory(
            urban_memory=urban,
            hazard_query=query,
            node_query=node_query,
            retrieval=retrieval,
            fused_memory=wrong,
            fusion_policy="retrieved_only",
        )


def test_hazard_memory_rejects_different_retrieval_sequence_object() -> None:
    urban = _urban_memory()
    equivalent_urban = _urban_memory()
    retrieval, query, node_query = _retrieval(
        urban_memory=equivalent_urban
    )
    assert (
        equivalent_urban
        .sequence_encoding
        .fingerprint()
        == urban
        .sequence_encoding
        .fingerprint()
    )

    with pytest.raises(ValueError):
        _hazard_memory(
            urban_memory=urban,
            hazard_query=query,
            node_query=node_query,
            retrieval=retrieval,
        )


def test_hazard_memory_rejects_query_alignment_mismatch() -> None:
    urban = _urban_memory()
    retrieval, query, node_query = _retrieval(
        urban_memory=urban,
        query_alignment_fingerprint="wrong-alignment",
    )

    with pytest.raises(ValueError):
        _hazard_memory(
            urban_memory=urban,
            hazard_query=query,
            node_query=node_query,
            retrieval=retrieval,
        )


def test_hazard_memory_rejects_query_lineage_mismatch() -> None:
    urban = _urban_memory()
    retrieval, query, node_query = _retrieval(
        urban_memory=urban,
        query_lineage_fingerprint="wrong-query-lineage",
    )

    with pytest.raises(ValueError):
        _hazard_memory(
            urban_memory=urban,
            hazard_query=query,
            node_query=node_query,
            retrieval=retrieval,
        )


@pytest.mark.parametrize(
    "fused",
    [
        torch.zeros(
            (N, H, 1),
            dtype=torch.float32,
        ),
        torch.zeros(
            (N, H),
            dtype=torch.long,
        ),
        torch.zeros(
            (N + 1, H),
            dtype=torch.float32,
        ),
    ],
)
def test_hazard_memory_rejects_invalid_fused_memory(
    fused: torch.Tensor,
) -> None:
    with pytest.raises(
        (TypeError, ValueError),
    ):
        _hazard_memory(
            fused_memory=fused
        )


def test_fusion_components_are_frozen_and_preserved() -> None:
    component = torch.ones(
        (
            N,
            H,
        ),
        dtype=torch.float32,
    )
    original = {
        "projected_context": component,
    }
    output = _hazard_memory(
        fusion_components=original
    )

    original[
        "new"
    ] = component

    assert isinstance(
        output.fusion_components,
        MappingProxyType,
    )
    assert tuple(
        output.fusion_components
    ) == (
        "projected_context",
    )

    with pytest.raises(TypeError):
        output.fusion_components[
            "new"
        ] = component  # type: ignore[index]


@pytest.mark.parametrize(
    "components",
    [
        {
            "": torch.ones(
                (N, H)
            ),
        },
        {
            "wrong-rows": torch.ones(
                (N + 1, H)
            ),
        },
        {
            "wrong-rank": torch.ones(
                (N, H, 1)
            ),
        },
        {
            "wrong-dtype": torch.ones(
                (N, H),
                dtype=torch.long,
            ),
        },
    ],
)
def test_hazard_memory_rejects_invalid_fusion_components(
    components: dict[str, torch.Tensor],
) -> None:
    with pytest.raises(
        (TypeError, ValueError),
    ):
        _hazard_memory(
            fusion_components=components
        )


def test_hazard_memory_requires_all_source_lineages() -> None:
    urban = _urban_memory()
    retrieval, query, node_query = _retrieval(
        urban_memory=urban
    )
    bad = _computation(
        source_fingerprints=(
            urban.lineage_fingerprint(),
            query.lineage_fingerprint,
        ),
        architecture_name="fuse_hazard_memory",
        architecture_kind="concat_projection",
        architecture_fingerprint="hazard-fusion-architecture-v1",
        source_history=urban.sequence_encoding.source_history,
    )

    with pytest.raises(ValueError):
        _hazard_memory(
            urban_memory=urban,
            hazard_query=query,
            node_query=node_query,
            retrieval=retrieval,
            computation_provenance=bad,
        )


def test_hazard_memory_fingerprints_are_deterministic() -> None:
    first = _hazard_memory()
    second = _hazard_memory()

    assert first.semantic_fingerprint() == second.semantic_fingerprint()
    assert first.value_fingerprint() == second.value_fingerprint()
    assert first.lineage_fingerprint() == second.lineage_fingerprint()


def test_hazard_memory_is_frozen_and_replace_revalidates() -> None:
    output = _hazard_memory()

    with pytest.raises(
        (FrozenInstanceError, AttributeError),
    ):
        output.memory_name = "changed"  # type: ignore[misc]

    renamed = output.replace(
        memory_name="renamed"
    )
    assert renamed.memory_name == "renamed"


def test_hazard_memory_to_preserves_moved_exact_identity() -> None:
    output = _hazard_memory()
    moved = output.to(
        "cpu"
    )

    assert moved.retrieval.source_sequence is (
        moved
        .source_urban_memory
        .sequence_encoding
    )
    assert moved.source_hazard_query is output.source_hazard_query
    assert moved.fingerprint() == output.fingerprint()


# =============================================================================
# Conditional CUDA boundaries
# =============================================================================


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is not available.",
)
def test_sequence_pooling_and_urban_memory_move_to_cuda() -> None:
    memory = _urban_memory(
        with_pooling=True
    )
    moved = memory.to(
        "cuda"
    )

    assert moved.device.type == "cuda"
    assert moved.temporal_pooling is not None
    assert moved.temporal_pooling.device.type == "cuda"
    assert moved.sequence_encoding is (
        moved.temporal_pooling.source_encoding
    )
