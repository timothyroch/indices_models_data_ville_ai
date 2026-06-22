"""
Contract tests for the metadata-preserving fusion schemas.

Target repository path:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            tests/
                test_fusion_schemas.py

Implementation under test:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            fusion/
                schemas.py

This suite freezes only the schema layer. It does not test projection,
concatenation, mode dispatch, or trainable fusion mathematics.

The contracts covered here are:

- stable schema-version identity;
- strict node/item and packed-graph alignment;
- immutable metadata-preserving generic components;
- preservation of complete lag-memory and hazard-query results;
- exact row, graph-membership, dtype, and device agreement;
- deterministic semantic, value, and lineage fingerprints;
- device-preserving reconstruction;
- immutable fusion outputs with validated projected components;
- lazy normalization of the public fusion-mode enum.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Any

import pytest
import torch

from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.config import (
    HAZARD_EMBEDDING_INIT_NORMAL,
    HAZARD_EMBEDDING_MODE_LEARNED,
    HazardEmbeddingConfig,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.fusion.node_state_fusion import (
    NodeStateFusionMode,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.fusion.schemas import (
    NODE_ALIGNMENT_SCHEMA_VERSION,
    NODE_STATE_COMPONENT_SCHEMA_VERSION,
    NODE_STATE_FUSION_INPUT_SCHEMA_VERSION,
    NODE_STATE_FUSION_OUTPUT_SCHEMA_VERSION,
    NodeAlignment,
    NodeStateComponent,
    NodeStateFusionInputs,
    NodeStateFusionOutput,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.hazard.hazard_embeddings import (
    HazardEmbeddingLayer,
    NodeAlignedHazardEmbeddingLookup,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.hazard.hazard_query_encoder import (
    HazardQueryEncoder,
    HazardQueryEncoding,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.hazard.hazard_registry import (
    HazardKind,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.memory.lag_memory_encoder import (
    LagMemoryBatch,
    LagMemoryEncoder,
    LagMemoryEncoding,
)


STATIC_DIM = 5
MEMORY_DIM = 7
HAZARD_MEMORY_DIM = 6
HAZARD_EMBEDDING_DIM = 4
HAZARD_QUERY_DIM = 8
FUSED_DIM = 11


# =============================================================================
# Helpers
# =============================================================================


def _alignment(
    item_count: int,
    *,
    item_ids: tuple[str, ...] | None = None,
    node_batch_index: torch.Tensor | None = None,
    graph_count: int | None = None,
    source_fingerprint: str = "alignment-source",
    alignment_name: str = "node_state_alignment",
) -> NodeAlignment:
    return NodeAlignment(
        item_count=item_count,
        item_ids=(
            tuple(
                f"item-{index}"
                for index in range(item_count)
            )
            if item_ids is None
            else item_ids
        ),
        node_batch_index=node_batch_index,
        graph_count=graph_count,
        source_fingerprint=source_fingerprint,
        alignment_name=alignment_name,
    )


def _component(
    item_count: int,
    feature_dim: int,
    *,
    name: str,
    offset: float = 0.0,
    dtype: torch.dtype = torch.float32,
    alignment: NodeAlignment | None = None,
    source_fingerprint: str | None = None,
) -> NodeStateComponent:
    values = (
        torch.arange(
            item_count * feature_dim,
            dtype=dtype,
        )
        .reshape(item_count, feature_dim)
        / 10.0
        + offset
    )

    return NodeStateComponent(
        values=values,
        component_name=name,
        source_fingerprint=(
            f"{name}-source"
            if source_fingerprint is None
            else source_fingerprint
        ),
        alignment_fingerprint=(
            alignment.fingerprint()
            if alignment is not None
            else None
        ),
    )


def _memory_encoding(
    item_count: int,
    *,
    hidden_dim: int = MEMORY_DIM,
    offset: float = 0.0,
) -> LagMemoryEncoding:
    values = (
        torch.arange(
            item_count * 3,
            dtype=torch.float32,
        )
        .reshape(item_count, 3)
        / 10.0
        + offset
    )
    batch = LagMemoryBatch(
        lag_values=values,
        lag_feature_names=(
            "burden_lag_1",
            "burden_lag_3",
            "burden_lag_12",
        ),
        source_history_length=12,
        source_fingerprint="lag-source",
    )
    encoder = LagMemoryEncoder(
        lag_feature_names=batch.lag_feature_names,
        hidden_dim=hidden_dim,
        return_lag_states=True,
        return_lag_weights=True,
    )
    encoder.eval()
    return encoder(batch)


def _hazard_layer() -> HazardEmbeddingLayer:
    config = HazardEmbeddingConfig(
        embedding_dim=HAZARD_EMBEDDING_DIM,
        mode=HAZARD_EMBEDDING_MODE_LEARNED,
        initialization=HAZARD_EMBEDDING_INIT_NORMAL,
        initialization_seed=17,
        initialization_std=0.05,
    )
    layer = HazardEmbeddingLayer.from_config(config)
    layer.eval()
    return layer


def _item_hazard_query(
    item_count: int,
) -> HazardQueryEncoding:
    layer = _hazard_layer()
    hazards = tuple(
        (
            HazardKind.FLOOD
            if index % 2 == 0
            else HazardKind.HEAT
        )
        for index in range(item_count)
    )
    lookup = layer.lookup_names(hazards)

    encoder = HazardQueryEncoder(
        hazard_embedding_dim=HAZARD_EMBEDDING_DIM,
        output_dim=HAZARD_QUERY_DIM,
        dropout=0.0,
    )
    encoder.eval()
    return encoder(lookup)


def _node_aligned_hazard_query(
    node_batch_index: torch.Tensor,
) -> HazardQueryEncoding:
    graph_count = int(
        node_batch_index.max().item()
    ) + 1
    available = (
        HazardKind.FLOOD,
        HazardKind.HEAT,
        HazardKind.OUTAGE,
        HazardKind.ROAD_DISRUPTION,
    )
    hazards = tuple(
        available[index % len(available)]
        for index in range(graph_count)
    )

    layer = _hazard_layer()
    lookup = layer.lookup_graph_hazards_for_nodes(
        hazards,
        node_batch_index,
    )
    assert isinstance(
        lookup,
        NodeAlignedHazardEmbeddingLookup,
    )

    encoder = HazardQueryEncoder(
        hazard_embedding_dim=HAZARD_EMBEDDING_DIM,
        output_dim=HAZARD_QUERY_DIM,
        dropout=0.0,
    )
    encoder.eval()
    return encoder(lookup)


def _fusion_inputs(
    item_count: int = 3,
) -> NodeStateFusionInputs:
    alignment = _alignment(item_count)
    return NodeStateFusionInputs(
        alignment=alignment,
        static_state=_component(
            item_count,
            STATIC_DIM,
            name="static_state",
            alignment=alignment,
        ),
    )


def _fusion_output(
    *,
    item_count: int = 3,
    fused_dim: int = FUSED_DIM,
    mode: NodeStateFusionMode | str = (
        NodeStateFusionMode.CONCAT_PROJECTION
    ),
) -> NodeStateFusionOutput:
    inputs = _fusion_inputs(item_count)
    fused = torch.arange(
        item_count * fused_dim,
        dtype=torch.float32,
    ).reshape(item_count, fused_dim)
    projected = {
        "static_state": fused.clone(),
    }
    return NodeStateFusionOutput(
        fused_state=fused,
        source_inputs=inputs,
        projected_components=projected,
        fusion_mode=mode,
        encoder_architecture_fingerprint="fusion-architecture",
        lineage_fingerprint="fusion-lineage",
    )


# =============================================================================
# Published schema identity
# =============================================================================


def test_schema_versions_are_nonempty_strings() -> None:
    for value in (
        NODE_ALIGNMENT_SCHEMA_VERSION,
        NODE_STATE_COMPONENT_SCHEMA_VERSION,
        NODE_STATE_FUSION_INPUT_SCHEMA_VERSION,
        NODE_STATE_FUSION_OUTPUT_SCHEMA_VERSION,
    ):
        assert isinstance(value, str)
        assert value.strip()


# =============================================================================
# NodeAlignment: valid contracts and identity
# =============================================================================


def test_item_alignment_preserves_semantic_identity() -> None:
    alignment = _alignment(3)

    assert alignment.item_count == 3
    assert alignment.item_ids == (
        "item-0",
        "item-1",
        "item-2",
    )
    assert alignment.source_fingerprint == "alignment-source"
    assert alignment.alignment_name == "node_state_alignment"
    assert not alignment.graph_aligned
    assert alignment.graph_count is None
    assert alignment.device is None

    semantic = alignment.semantic_dict()
    assert semantic == {
        "schema_version": NODE_ALIGNMENT_SCHEMA_VERSION,
        "alignment_name": "node_state_alignment",
        "item_count": 3,
        "item_ids": [
            "item-0",
            "item-1",
            "item-2",
        ],
        "graph_count": None,
        "graph_aligned": False,
        "source_fingerprint": "alignment-source",
    }


def test_item_alignment_fingerprints_are_deterministic() -> None:
    first = _alignment(3)
    second = _alignment(3)

    assert first.semantic_fingerprint() == (
        second.semantic_fingerprint()
    )
    assert first.value_fingerprint() == (
        second.value_fingerprint()
    )
    assert first.fingerprint() == second.fingerprint()


def test_alignment_semantic_fingerprint_changes_with_metadata() -> None:
    baseline = _alignment(2)
    changed_ids = _alignment(
        2,
        item_ids=("a", "b"),
    )
    changed_source = _alignment(
        2,
        source_fingerprint="other-source",
    )
    changed_name = _alignment(
        2,
        alignment_name="other-alignment",
    )

    assert baseline.semantic_fingerprint() != (
        changed_ids.semantic_fingerprint()
    )
    assert baseline.semantic_fingerprint() != (
        changed_source.semantic_fingerprint()
    )
    assert baseline.semantic_fingerprint() != (
        changed_name.semantic_fingerprint()
    )


def test_graph_alignment_preserves_membership() -> None:
    membership = torch.tensor(
        [0, 0, 1, 2, 2],
        dtype=torch.long,
    )
    alignment = _alignment(
        5,
        node_batch_index=membership,
        graph_count=3,
    )

    assert alignment.graph_aligned
    assert alignment.graph_count == 3
    assert alignment.device == membership.device
    assert torch.equal(
        alignment.node_batch_index,
        membership,
    )


def test_graph_alignment_value_fingerprint_changes_with_membership() -> None:
    first = _alignment(
        4,
        node_batch_index=torch.tensor(
            [0, 0, 1, 1],
            dtype=torch.long,
        ),
        graph_count=2,
    )
    second = _alignment(
        4,
        node_batch_index=torch.tensor(
            [0, 1, 1, 1],
            dtype=torch.long,
        ),
        graph_count=2,
    )

    assert first.semantic_fingerprint() == (
        second.semantic_fingerprint()
    )
    assert first.value_fingerprint() != (
        second.value_fingerprint()
    )
    assert first.fingerprint() != second.fingerprint()


def test_alignment_to_cpu_preserves_fingerprint() -> None:
    membership = torch.tensor(
        [0, 0, 1],
        dtype=torch.long,
    )
    alignment = _alignment(
        3,
        node_batch_index=membership,
        graph_count=2,
    )
    moved = alignment.to("cpu")

    assert moved is not alignment
    assert moved.fingerprint() == alignment.fingerprint()
    assert torch.equal(
        moved.node_batch_index,
        alignment.node_batch_index,
    )


def test_zero_item_non_graph_alignment_is_valid() -> None:
    alignment = NodeAlignment(
        item_count=0,
    )

    assert alignment.item_count == 0
    assert alignment.item_ids == ()
    assert not alignment.graph_aligned


# =============================================================================
# NodeAlignment: invalid contracts
# =============================================================================


@pytest.mark.parametrize(
    "item_count",
    (
        -1,
        True,
        1.5,
    ),
)
def test_alignment_rejects_invalid_item_count(
    item_count: Any,
) -> None:
    with pytest.raises(ValueError, match="nonnegative integer"):
        NodeAlignment(
            item_count=item_count,
        )


def test_alignment_rejects_duplicate_item_ids() -> None:
    with pytest.raises(ValueError, match="duplicates"):
        NodeAlignment(
            item_count=2,
            item_ids=("same", "same"),
        )


def test_alignment_rejects_blank_item_id() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        NodeAlignment(
            item_count=2,
            item_ids=("valid", " "),
        )


def test_alignment_rejects_item_id_count_mismatch() -> None:
    with pytest.raises(ValueError, match="align"):
        NodeAlignment(
            item_count=2,
            item_ids=("only-one",),
        )


def test_alignment_rejects_graph_count_without_membership() -> None:
    with pytest.raises(ValueError, match="requires"):
        NodeAlignment(
            item_count=2,
            graph_count=1,
        )


def test_alignment_rejects_membership_without_graph_count() -> None:
    with pytest.raises(ValueError, match="graph_count"):
        NodeAlignment(
            item_count=2,
            node_batch_index=torch.tensor(
                [0, 0],
                dtype=torch.long,
            ),
        )


def test_alignment_rejects_non_tensor_membership() -> None:
    with pytest.raises(TypeError, match="tensor"):
        NodeAlignment(
            item_count=2,
            node_batch_index=[0, 0],  # type: ignore[arg-type]
            graph_count=1,
        )


def test_alignment_rejects_membership_rank() -> None:
    with pytest.raises(ValueError, match="shape"):
        NodeAlignment(
            item_count=2,
            node_batch_index=torch.zeros(
                2,
                1,
                dtype=torch.long,
            ),
            graph_count=1,
        )


def test_alignment_rejects_membership_dtype() -> None:
    with pytest.raises(ValueError, match="torch.long"):
        NodeAlignment(
            item_count=2,
            node_batch_index=torch.zeros(
                2,
                dtype=torch.int32,
            ),
            graph_count=1,
        )


def test_alignment_rejects_membership_length_mismatch() -> None:
    with pytest.raises(ValueError, match="align"):
        NodeAlignment(
            item_count=3,
            node_batch_index=torch.tensor(
                [0, 0],
                dtype=torch.long,
            ),
            graph_count=1,
        )


@pytest.mark.parametrize(
    "graph_count",
    (
        0,
        -1,
        True,
        1.5,
    ),
)
def test_alignment_rejects_invalid_graph_count(
    graph_count: Any,
) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        NodeAlignment(
            item_count=1,
            node_batch_index=torch.tensor(
                [0],
                dtype=torch.long,
            ),
            graph_count=graph_count,
        )


def test_alignment_rejects_graph_aligned_empty_batch() -> None:
    with pytest.raises(ValueError, match="zero items"):
        NodeAlignment(
            item_count=0,
            node_batch_index=torch.empty(
                0,
                dtype=torch.long,
            ),
            graph_count=1,
        )


def test_alignment_rejects_negative_graph_id() -> None:
    with pytest.raises(ValueError, match="negative"):
        NodeAlignment(
            item_count=2,
            node_batch_index=torch.tensor(
                [-1, 0],
                dtype=torch.long,
            ),
            graph_count=1,
        )


def test_alignment_rejects_graph_id_outside_count() -> None:
    with pytest.raises(ValueError, match="outside"):
        NodeAlignment(
            item_count=2,
            node_batch_index=torch.tensor(
                [0, 2],
                dtype=torch.long,
            ),
            graph_count=2,
        )


def test_alignment_rejects_noncontiguous_graph_ids() -> None:
    with pytest.raises(ValueError, match="contiguous"):
        NodeAlignment(
            item_count=3,
            node_batch_index=torch.tensor(
                [0, 2, 2],
                dtype=torch.long,
            ),
            graph_count=3,
        )


def test_alignment_requires_every_declared_graph_to_be_represented() -> None:
    with pytest.raises(ValueError, match="represent every"):
        NodeAlignment(
            item_count=3,
            node_batch_index=torch.tensor(
                [0, 0, 1],
                dtype=torch.long,
            ),
            graph_count=3,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("source_fingerprint", ""),
        ("alignment_name", " "),
        ("schema_version", ""),
    ),
)
def test_alignment_rejects_blank_string_metadata(
    field: str,
    value: str,
) -> None:
    kwargs = {
        "item_count": 1,
        field: value,
    }

    with pytest.raises(ValueError, match="non-empty"):
        NodeAlignment(**kwargs)


# =============================================================================
# NodeStateComponent: valid contracts and identity
# =============================================================================


def test_component_preserves_values_and_metadata() -> None:
    alignment = _alignment(3)
    component = _component(
        3,
        STATIC_DIM,
        name="static_state",
        alignment=alignment,
    )

    assert component.item_count == 3
    assert component.feature_dim == STATIC_DIM
    assert component.device == torch.device("cpu")
    assert component.component_name == "static_state"
    assert component.source_fingerprint == "static_state-source"
    assert component.alignment_fingerprint == (
        alignment.fingerprint()
    )
    assert component.schema_version == (
        NODE_STATE_COMPONENT_SCHEMA_VERSION
    )


def test_component_fingerprints_are_deterministic() -> None:
    first = _component(
        2,
        STATIC_DIM,
        name="static_state",
    )
    second = _component(
        2,
        STATIC_DIM,
        name="static_state",
    )

    assert first.value_fingerprint() == (
        second.value_fingerprint()
    )
    assert first.lineage_fingerprint() == (
        second.lineage_fingerprint()
    )


def test_component_value_fingerprint_changes_with_values() -> None:
    first = _component(
        2,
        STATIC_DIM,
        name="static_state",
        offset=0.0,
    )
    second = _component(
        2,
        STATIC_DIM,
        name="static_state",
        offset=1.0,
    )

    assert first.value_fingerprint() != (
        second.value_fingerprint()
    )
    assert first.lineage_fingerprint() != (
        second.lineage_fingerprint()
    )


def test_component_lineage_changes_with_provenance() -> None:
    first = _component(
        2,
        STATIC_DIM,
        name="static_state",
        source_fingerprint="source-a",
    )
    second = _component(
        2,
        STATIC_DIM,
        name="static_state",
        source_fingerprint="source-b",
    )

    assert first.value_fingerprint() == (
        second.value_fingerprint()
    )
    assert first.lineage_fingerprint() != (
        second.lineage_fingerprint()
    )


def test_component_to_cpu_preserves_lineage() -> None:
    component = _component(
        2,
        STATIC_DIM,
        name="static_state",
    )
    moved = component.to("cpu")

    assert moved is not component
    assert torch.equal(
        moved.values,
        component.values,
    )
    assert moved.lineage_fingerprint() == (
        component.lineage_fingerprint()
    )


def test_component_allows_empty_item_axis() -> None:
    component = NodeStateComponent(
        values=torch.empty(
            0,
            STATIC_DIM,
            dtype=torch.float32,
        ),
        component_name="static_state",
    )

    assert component.item_count == 0
    assert component.feature_dim == STATIC_DIM


# =============================================================================
# NodeStateComponent: invalid contracts
# =============================================================================


def test_component_rejects_non_tensor_values() -> None:
    with pytest.raises(TypeError, match="tensor"):
        NodeStateComponent(
            values=[[1.0]],  # type: ignore[arg-type]
            component_name="static_state",
        )


@pytest.mark.parametrize(
    "values",
    (
        torch.zeros(STATIC_DIM),
        torch.zeros(2, STATIC_DIM, 1),
    ),
)
def test_component_rejects_invalid_rank(
    values: torch.Tensor,
) -> None:
    with pytest.raises(ValueError, match="shape"):
        NodeStateComponent(
            values=values,
            component_name="static_state",
        )


def test_component_rejects_zero_feature_width() -> None:
    with pytest.raises(ValueError, match="at least one feature"):
        NodeStateComponent(
            values=torch.empty(
                2,
                0,
                dtype=torch.float32,
            ),
            component_name="static_state",
        )


def test_component_rejects_nonfloating_values() -> None:
    with pytest.raises(ValueError, match="floating-point"):
        NodeStateComponent(
            values=torch.zeros(
                2,
                STATIC_DIM,
                dtype=torch.long,
            ),
            component_name="static_state",
        )


@pytest.mark.parametrize(
    "bad_value",
    (
        float("nan"),
        float("inf"),
        float("-inf"),
    ),
)
def test_component_rejects_nonfinite_values(
    bad_value: float,
) -> None:
    values = torch.zeros(
        2,
        STATIC_DIM,
    )
    values[0, 0] = bad_value

    with pytest.raises(ValueError, match="finite"):
        NodeStateComponent(
            values=values,
            component_name="static_state",
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("component_name", ""),
        ("source_fingerprint", " "),
        ("alignment_fingerprint", ""),
        ("schema_version", " "),
    ),
)
def test_component_rejects_blank_metadata(
    field: str,
    value: str,
) -> None:
    kwargs = {
        "values": torch.zeros(
            1,
            STATIC_DIM,
        ),
        "component_name": "static_state",
        field: value,
    }

    with pytest.raises(ValueError, match="non-empty"):
        NodeStateComponent(**kwargs)


# =============================================================================
# NodeStateFusionInputs: valid contracts and preservation
# =============================================================================


def test_alignment_only_inputs_are_valid() -> None:
    inputs = NodeStateFusionInputs(
        alignment=_alignment(2),
    )

    assert inputs.item_count == 2
    assert inputs.device is None
    assert inputs.component_lineage_dict() == {
        "schema_version": (
            NODE_STATE_FUSION_INPUT_SCHEMA_VERSION
        ),
        "alignment_fingerprint": (
            inputs.alignment.fingerprint()
        ),
        "source_fingerprint": None,
    }


def test_inputs_preserve_static_component() -> None:
    alignment = _alignment(3)
    static = _component(
        3,
        STATIC_DIM,
        name="static_state",
        alignment=alignment,
    )
    inputs = NodeStateFusionInputs(
        alignment=alignment,
        static_state=static,
    )

    assert inputs.alignment is alignment
    assert inputs.static_state is static
    assert inputs.item_count == 3
    assert inputs.device == torch.device("cpu")
    assert inputs.component_lineage_dict()[
        "static_state"
    ] == static.lineage_fingerprint()


def test_inputs_preserve_complete_memory_encoding() -> None:
    memory = _memory_encoding(4)
    inputs = NodeStateFusionInputs(
        alignment=_alignment(4),
        memory_state=memory,
    )

    assert inputs.memory_state is memory
    assert inputs.component_lineage_dict()[
        "memory_state"
    ] == memory.lineage_fingerprint


def test_inputs_preserve_complete_item_hazard_query() -> None:
    hazard_query = _item_hazard_query(4)
    inputs = NodeStateFusionInputs(
        alignment=_alignment(4),
        hazard_context=hazard_query,
    )

    assert inputs.hazard_context is hazard_query
    assert inputs.component_lineage_dict()[
        "hazard_context"
    ] == hazard_query.lineage_fingerprint


def test_item_hazard_query_does_not_require_graph_membership() -> None:
    hazard_query = _item_hazard_query(3)

    inputs = NodeStateFusionInputs(
        alignment=_alignment(3),
        hazard_context=hazard_query,
    )

    assert inputs.alignment.node_batch_index is None


def test_inputs_preserve_all_typed_sources() -> None:
    item_count = 5
    alignment = _alignment(item_count)
    static = _component(
        item_count,
        STATIC_DIM,
        name="static_state",
        alignment=alignment,
    )
    memory = _memory_encoding(item_count)
    hazard_memory = _component(
        item_count,
        HAZARD_MEMORY_DIM,
        name="hazard_memory_state",
        alignment=alignment,
    )
    hazard_query = _item_hazard_query(item_count)
    node_types = torch.tensor(
        [0, 1, 2, 0, 1],
        dtype=torch.long,
    )

    inputs = NodeStateFusionInputs(
        alignment=alignment,
        static_state=static,
        memory_state=memory,
        hazard_memory_state=hazard_memory,
        hazard_context=hazard_query,
        node_type_ids=node_types,
        source_fingerprint="fusion-input-source",
    )

    assert inputs.static_state is static
    assert inputs.memory_state is memory
    assert inputs.hazard_memory_state is hazard_memory
    assert inputs.hazard_context is hazard_query
    assert inputs.node_type_ids is node_types
    assert inputs.source_fingerprint == "fusion-input-source"

    lineage = inputs.component_lineage_dict()
    assert tuple(lineage) == (
        "schema_version",
        "alignment_fingerprint",
        "source_fingerprint",
        "static_state",
        "memory_state",
        "hazard_memory_state",
        "hazard_context",
        "node_type_ids",
    )


def test_node_aligned_hazard_query_requires_matching_membership() -> None:
    membership = torch.tensor(
        [0, 0, 1, 1, 1],
        dtype=torch.long,
    )
    hazard_query = _node_aligned_hazard_query(
        membership
    )
    alignment = _alignment(
        5,
        node_batch_index=membership,
        graph_count=2,
    )

    inputs = NodeStateFusionInputs(
        alignment=alignment,
        hazard_context=hazard_query,
    )

    assert inputs.hazard_context is hazard_query
    assert torch.equal(
        inputs.alignment.node_batch_index,
        membership,
    )


def test_inputs_lineage_is_deterministic() -> None:
    first = _fusion_inputs(3)
    second = _fusion_inputs(3)

    assert first.component_lineage_dict() == (
        second.component_lineage_dict()
    )
    assert first.lineage_fingerprint() == (
        second.lineage_fingerprint()
    )


def test_inputs_lineage_changes_with_component_values() -> None:
    alignment = _alignment(2)
    first = NodeStateFusionInputs(
        alignment=alignment,
        static_state=_component(
            2,
            STATIC_DIM,
            name="static_state",
            offset=0.0,
        ),
    )
    second = NodeStateFusionInputs(
        alignment=alignment,
        static_state=_component(
            2,
            STATIC_DIM,
            name="static_state",
            offset=1.0,
        ),
    )

    assert first.lineage_fingerprint() != (
        second.lineage_fingerprint()
    )


def test_inputs_lineage_changes_with_node_types() -> None:
    alignment = _alignment(3)
    first = NodeStateFusionInputs(
        alignment=alignment,
        node_type_ids=torch.tensor(
            [0, 1, 2],
            dtype=torch.long,
        ),
    )
    second = NodeStateFusionInputs(
        alignment=alignment,
        node_type_ids=torch.tensor(
            [0, 2, 1],
            dtype=torch.long,
        ),
    )

    assert first.lineage_fingerprint() != (
        second.lineage_fingerprint()
    )


def test_inputs_to_cpu_preserves_lineage_and_typed_memory() -> None:
    alignment = _alignment(2)
    memory = _memory_encoding(2)
    inputs = NodeStateFusionInputs(
        alignment=alignment,
        static_state=_component(
            2,
            STATIC_DIM,
            name="static_state",
            alignment=alignment,
        ),
        memory_state=memory,
        node_type_ids=torch.tensor(
            [0, 1],
            dtype=torch.long,
        ),
    )

    moved = inputs.to("cpu")

    assert moved is not inputs
    assert moved.lineage_fingerprint() == (
        inputs.lineage_fingerprint()
    )
    assert moved.memory_state is not None
    assert moved.memory_state is not memory
    assert torch.equal(
        moved.memory_state.memory_state,
        memory.memory_state,
    )
    assert moved.memory_state.lineage_fingerprint == (
        memory.lineage_fingerprint
    )


def test_inputs_to_cpu_preserves_existing_cpu_hazard_query_object() -> None:
    query = _item_hazard_query(2)
    inputs = NodeStateFusionInputs(
        alignment=_alignment(2),
        hazard_context=query,
    )

    moved = inputs.to("cpu")

    assert moved.hazard_context is query
    assert moved.lineage_fingerprint() == (
        inputs.lineage_fingerprint()
    )


# =============================================================================
# NodeStateFusionInputs: invalid contracts
# =============================================================================


@pytest.mark.parametrize(
    ("field", "value", "expected_type"),
    (
        (
            "static_state",
            torch.zeros(2, STATIC_DIM),
            "NodeStateComponent",
        ),
        (
            "memory_state",
            torch.zeros(2, MEMORY_DIM),
            "LagMemoryEncoding",
        ),
        (
            "hazard_memory_state",
            torch.zeros(2, HAZARD_MEMORY_DIM),
            "NodeStateComponent",
        ),
        (
            "hazard_context",
            torch.zeros(2, HAZARD_QUERY_DIM),
            "HazardQueryEncoding",
        ),
    ),
)
def test_inputs_reject_wrong_component_types(
    field: str,
    value: Any,
    expected_type: str,
) -> None:
    kwargs = {
        "alignment": _alignment(2),
        field: value,
    }

    with pytest.raises(TypeError, match=expected_type):
        NodeStateFusionInputs(**kwargs)


def test_inputs_reject_non_alignment_object() -> None:
    with pytest.raises(TypeError, match="NodeAlignment"):
        NodeStateFusionInputs(
            alignment=object(),  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "field",
    (
        "static_state",
        "memory_state",
        "hazard_memory_state",
        "hazard_context",
    ),
)
def test_inputs_reject_component_row_mismatch(
    field: str,
) -> None:
    alignment = _alignment(3)

    values: dict[str, Any] = {
        "static_state": _component(
            2,
            STATIC_DIM,
            name="static_state",
        ),
        "memory_state": _memory_encoding(2),
        "hazard_memory_state": _component(
            2,
            HAZARD_MEMORY_DIM,
            name="hazard_memory_state",
        ),
        "hazard_context": _item_hazard_query(2),
    }

    kwargs = {
        "alignment": alignment,
        field: values[field],
    }

    with pytest.raises(ValueError, match="row counts"):
        NodeStateFusionInputs(**kwargs)


def test_inputs_reject_node_type_row_mismatch() -> None:
    with pytest.raises(ValueError, match="row counts"):
        NodeStateFusionInputs(
            alignment=_alignment(3),
            node_type_ids=torch.tensor(
                [0, 1],
                dtype=torch.long,
            ),
        )


def test_inputs_reject_component_alignment_fingerprint_mismatch() -> None:
    expected = _alignment(
        3,
        source_fingerprint="expected",
    )
    other = _alignment(
        3,
        source_fingerprint="other",
    )
    static = _component(
        3,
        STATIC_DIM,
        name="static_state",
        alignment=other,
    )

    with pytest.raises(
        ValueError,
        match="different node alignment",
    ):
        NodeStateFusionInputs(
            alignment=expected,
            static_state=static,
        )


def test_node_aligned_query_rejects_missing_alignment_membership() -> None:
    membership = torch.tensor(
        [0, 0, 1],
        dtype=torch.long,
    )
    query = _node_aligned_hazard_query(
        membership
    )

    with pytest.raises(
        ValueError,
        match="requires alignment.node_batch_index",
    ):
        NodeStateFusionInputs(
            alignment=_alignment(3),
            hazard_context=query,
        )


def test_node_aligned_query_rejects_different_membership() -> None:
    query_membership = torch.tensor(
        [0, 0, 1, 1],
        dtype=torch.long,
    )
    alignment_membership = torch.tensor(
        [0, 1, 1, 1],
        dtype=torch.long,
    )
    query = _node_aligned_hazard_query(
        query_membership
    )

    with pytest.raises(
        ValueError,
        match="differs from NodeAlignment",
    ):
        NodeStateFusionInputs(
            alignment=_alignment(
                4,
                node_batch_index=alignment_membership,
                graph_count=2,
            ),
            hazard_context=query,
        )


def test_inputs_reject_non_tensor_node_types() -> None:
    with pytest.raises(TypeError, match="tensor"):
        NodeStateFusionInputs(
            alignment=_alignment(2),
            node_type_ids=[0, 1],  # type: ignore[arg-type]
        )


def test_inputs_reject_node_type_rank() -> None:
    with pytest.raises(ValueError, match="shape"):
        NodeStateFusionInputs(
            alignment=_alignment(2),
            node_type_ids=torch.zeros(
                2,
                1,
                dtype=torch.long,
            ),
        )


def test_inputs_reject_node_type_dtype() -> None:
    with pytest.raises(ValueError, match="torch.long"):
        NodeStateFusionInputs(
            alignment=_alignment(2),
            node_type_ids=torch.zeros(
                2,
                dtype=torch.int32,
            ),
        )


def test_inputs_reject_blank_source_fingerprint() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        NodeStateFusionInputs(
            alignment=_alignment(1),
            source_fingerprint=" ",
        )


def test_inputs_reject_blank_schema_version() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        NodeStateFusionInputs(
            alignment=_alignment(1),
            schema_version="",
        )


# =============================================================================
# NodeStateFusionOutput: valid contracts and preservation
# =============================================================================


def test_output_preserves_source_and_projected_components() -> None:
    output = _fusion_output(
        item_count=3,
        fused_dim=FUSED_DIM,
    )

    assert output.item_count == 3
    assert output.output_dim == FUSED_DIM
    assert output.alignment is (
        output.source_inputs.alignment
    )
    assert output.fusion_mode == (
        NodeStateFusionMode.CONCAT_PROJECTION
    )
    assert tuple(
        output.projected_components
    ) == ("static_state",)
    assert isinstance(
        output.projected_components,
        MappingProxyType,
    )
    assert output.schema_version == (
        NODE_STATE_FUSION_OUTPUT_SCHEMA_VERSION
    )


def test_output_normalizes_string_mode_to_enum() -> None:
    output = _fusion_output(
        mode="concat_projection",
    )

    assert output.fusion_mode is (
        NodeStateFusionMode.CONCAT_PROJECTION
    )


def test_output_projected_mapping_is_read_only() -> None:
    output = _fusion_output()

    with pytest.raises(TypeError):
        output.projected_components[
            "other"
        ] = torch.zeros(3, FUSED_DIM)  # type: ignore[index]


def test_output_copies_projected_mapping_structure() -> None:
    inputs = _fusion_inputs(2)
    fused = torch.zeros(
        2,
        FUSED_DIM,
    )
    original = {
        "static_state": torch.ones(
            2,
            FUSED_DIM,
        ),
    }
    output = NodeStateFusionOutput(
        fused_state=fused,
        source_inputs=inputs,
        projected_components=original,
        fusion_mode=(
            NodeStateFusionMode.CONCAT_PROJECTION
        ),
        encoder_architecture_fingerprint="architecture",
        lineage_fingerprint="lineage",
    )

    original["other"] = torch.zeros(
        2,
        FUSED_DIM,
    )

    assert tuple(
        output.projected_components
    ) == ("static_state",)


def test_output_allows_empty_item_axis() -> None:
    alignment = NodeAlignment(
        item_count=0,
    )
    static = NodeStateComponent(
        values=torch.empty(
            0,
            STATIC_DIM,
        ),
        component_name="static_state",
    )
    inputs = NodeStateFusionInputs(
        alignment=alignment,
        static_state=static,
    )

    output = NodeStateFusionOutput(
        fused_state=torch.empty(
            0,
            FUSED_DIM,
        ),
        source_inputs=inputs,
        projected_components={
            "static_state": torch.empty(
                0,
                FUSED_DIM,
            ),
        },
        fusion_mode="concat_projection",
        encoder_architecture_fingerprint="architecture",
        lineage_fingerprint="lineage",
    )

    assert output.item_count == 0
    assert output.output_dim == FUSED_DIM


# =============================================================================
# NodeStateFusionOutput: invalid contracts
# =============================================================================


def test_output_rejects_non_tensor_fused_state() -> None:
    with pytest.raises(TypeError, match="tensor"):
        NodeStateFusionOutput(
            fused_state=[[0.0]],  # type: ignore[arg-type]
            source_inputs=_fusion_inputs(1),
            projected_components={
                "static_state": torch.zeros(
                    1,
                    FUSED_DIM,
                ),
            },
            fusion_mode="concat_projection",
            encoder_architecture_fingerprint="architecture",
            lineage_fingerprint="lineage",
        )


@pytest.mark.parametrize(
    "fused",
    (
        torch.zeros(FUSED_DIM),
        torch.zeros(2, FUSED_DIM, 1),
    ),
)
def test_output_rejects_invalid_fused_rank(
    fused: torch.Tensor,
) -> None:
    with pytest.raises(ValueError, match="shape"):
        NodeStateFusionOutput(
            fused_state=fused,
            source_inputs=_fusion_inputs(2),
            projected_components={
                "static_state": torch.zeros(
                    2,
                    FUSED_DIM,
                ),
            },
            fusion_mode="concat_projection",
            encoder_architecture_fingerprint="architecture",
            lineage_fingerprint="lineage",
        )


def test_output_rejects_nonfloating_fused_state() -> None:
    with pytest.raises(ValueError, match="floating-point"):
        NodeStateFusionOutput(
            fused_state=torch.zeros(
                2,
                FUSED_DIM,
                dtype=torch.long,
            ),
            source_inputs=_fusion_inputs(2),
            projected_components={
                "static_state": torch.zeros(
                    2,
                    FUSED_DIM,
                ),
            },
            fusion_mode="concat_projection",
            encoder_architecture_fingerprint="architecture",
            lineage_fingerprint="lineage",
        )


def test_output_rejects_fused_row_mismatch() -> None:
    with pytest.raises(ValueError, match="rows"):
        NodeStateFusionOutput(
            fused_state=torch.zeros(
                2,
                FUSED_DIM,
            ),
            source_inputs=_fusion_inputs(3),
            projected_components={
                "static_state": torch.zeros(
                    3,
                    FUSED_DIM,
                ),
            },
            fusion_mode="concat_projection",
            encoder_architecture_fingerprint="architecture",
            lineage_fingerprint="lineage",
        )


@pytest.mark.parametrize(
    "bad_value",
    (
        float("nan"),
        float("inf"),
        float("-inf"),
    ),
)
def test_output_rejects_nonfinite_fused_state(
    bad_value: float,
) -> None:
    fused = torch.zeros(
        2,
        FUSED_DIM,
    )
    fused[0, 0] = bad_value

    with pytest.raises(ValueError, match="finite"):
        NodeStateFusionOutput(
            fused_state=fused,
            source_inputs=_fusion_inputs(2),
            projected_components={
                "static_state": torch.zeros(
                    2,
                    FUSED_DIM,
                ),
            },
            fusion_mode="concat_projection",
            encoder_architecture_fingerprint="architecture",
            lineage_fingerprint="lineage",
        )


def test_output_rejects_non_mapping_projected_components() -> None:
    with pytest.raises(TypeError, match="mapping"):
        NodeStateFusionOutput(
            fused_state=torch.zeros(
                2,
                FUSED_DIM,
            ),
            source_inputs=_fusion_inputs(2),
            projected_components=[  # type: ignore[arg-type]
                torch.zeros(2, FUSED_DIM)
            ],
            fusion_mode="concat_projection",
            encoder_architecture_fingerprint="architecture",
            lineage_fingerprint="lineage",
        )


def test_output_rejects_empty_projected_components() -> None:
    with pytest.raises(ValueError, match="At least one"):
        NodeStateFusionOutput(
            fused_state=torch.zeros(
                2,
                FUSED_DIM,
            ),
            source_inputs=_fusion_inputs(2),
            projected_components={},
            fusion_mode="concat_projection",
            encoder_architecture_fingerprint="architecture",
            lineage_fingerprint="lineage",
        )


def test_output_rejects_blank_projected_component_name() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        NodeStateFusionOutput(
            fused_state=torch.zeros(
                2,
                FUSED_DIM,
            ),
            source_inputs=_fusion_inputs(2),
            projected_components={
                " ": torch.zeros(
                    2,
                    FUSED_DIM,
                ),
            },
            fusion_mode="concat_projection",
            encoder_architecture_fingerprint="architecture",
            lineage_fingerprint="lineage",
        )


def test_output_rejects_non_tensor_projected_component() -> None:
    with pytest.raises(TypeError, match="tensors"):
        NodeStateFusionOutput(
            fused_state=torch.zeros(
                2,
                FUSED_DIM,
            ),
            source_inputs=_fusion_inputs(2),
            projected_components={
                "static_state": [[0.0]],  # type: ignore[dict-item]
            },
            fusion_mode="concat_projection",
            encoder_architecture_fingerprint="architecture",
            lineage_fingerprint="lineage",
        )


@pytest.mark.parametrize(
    "projected",
    (
        torch.zeros(FUSED_DIM),
        torch.zeros(2, FUSED_DIM, 1),
    ),
)
def test_output_rejects_projected_component_rank(
    projected: torch.Tensor,
) -> None:
    with pytest.raises(ValueError, match="shape"):
        NodeStateFusionOutput(
            fused_state=torch.zeros(
                2,
                FUSED_DIM,
            ),
            source_inputs=_fusion_inputs(2),
            projected_components={
                "static_state": projected,
            },
            fusion_mode="concat_projection",
            encoder_architecture_fingerprint="architecture",
            lineage_fingerprint="lineage",
        )


def test_output_rejects_projected_row_mismatch() -> None:
    with pytest.raises(ValueError, match="rows"):
        NodeStateFusionOutput(
            fused_state=torch.zeros(
                2,
                FUSED_DIM,
            ),
            source_inputs=_fusion_inputs(2),
            projected_components={
                "static_state": torch.zeros(
                    3,
                    FUSED_DIM,
                ),
            },
            fusion_mode="concat_projection",
            encoder_architecture_fingerprint="architecture",
            lineage_fingerprint="lineage",
        )


def test_output_allows_projected_width_different_from_fused_width() -> None:
    """
    The schema intentionally validates rows, not algorithm-specific width.

    Width equality belongs to ConcatProjectionFusionOutput and to the
    orchestrator algorithm contract, not to this generic fusion schema.
    """

    output = NodeStateFusionOutput(
        fused_state=torch.zeros(
            2,
            FUSED_DIM,
        ),
        source_inputs=_fusion_inputs(2),
        projected_components={
            "experimental_component": torch.zeros(
                2,
                3,
            ),
        },
        fusion_mode="concat_projection",
        encoder_architecture_fingerprint="architecture",
        lineage_fingerprint="lineage",
    )

    assert output.projected_components[
        "experimental_component"
    ].shape == (2, 3)


def test_output_rejects_nonfloating_projected_component() -> None:
    with pytest.raises(ValueError, match="floating-point"):
        NodeStateFusionOutput(
            fused_state=torch.zeros(
                2,
                FUSED_DIM,
            ),
            source_inputs=_fusion_inputs(2),
            projected_components={
                "static_state": torch.zeros(
                    2,
                    FUSED_DIM,
                    dtype=torch.long,
                ),
            },
            fusion_mode="concat_projection",
            encoder_architecture_fingerprint="architecture",
            lineage_fingerprint="lineage",
        )


def test_output_rejects_nonfinite_projected_component() -> None:
    projected = torch.zeros(
        2,
        FUSED_DIM,
    )
    projected[0, 0] = float("nan")

    with pytest.raises(ValueError, match="finite"):
        NodeStateFusionOutput(
            fused_state=torch.zeros(
                2,
                FUSED_DIM,
            ),
            source_inputs=_fusion_inputs(2),
            projected_components={
                "static_state": projected,
            },
            fusion_mode="concat_projection",
            encoder_architecture_fingerprint="architecture",
            lineage_fingerprint="lineage",
        )


def test_output_rejects_unknown_fusion_mode() -> None:
    with pytest.raises(ValueError):
        NodeStateFusionOutput(
            fused_state=torch.zeros(
                2,
                FUSED_DIM,
            ),
            source_inputs=_fusion_inputs(2),
            projected_components={
                "static_state": torch.zeros(
                    2,
                    FUSED_DIM,
                ),
            },
            fusion_mode="not_a_real_mode",
            encoder_architecture_fingerprint="architecture",
            lineage_fingerprint="lineage",
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        (
            "encoder_architecture_fingerprint",
            "",
        ),
        (
            "lineage_fingerprint",
            " ",
        ),
        (
            "schema_version",
            "",
        ),
    ),
)
def test_output_rejects_blank_identity_fields(
    field: str,
    value: str,
) -> None:
    kwargs = {
        "fused_state": torch.zeros(
            2,
            FUSED_DIM,
        ),
        "source_inputs": _fusion_inputs(2),
        "projected_components": {
            "static_state": torch.zeros(
                2,
                FUSED_DIM,
            ),
        },
        "fusion_mode": "concat_projection",
        "encoder_architecture_fingerprint": "architecture",
        "lineage_fingerprint": "lineage",
        field: value,
    }

    with pytest.raises(ValueError, match="non-empty"):
        NodeStateFusionOutput(**kwargs)


# =============================================================================
# Optional device checks
# =============================================================================


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is unavailable.",
)
def test_inputs_reject_cross_component_device_mismatch() -> None:
    alignment = _alignment(2)
    cpu_static = _component(
        2,
        STATIC_DIM,
        name="static_state",
    )
    gpu_memory = _memory_encoding(2)
    gpu_memory = LagMemoryEncoding(
        memory_state=gpu_memory.memory_state.cuda(),
        source_batch=gpu_memory.source_batch.to("cuda"),
        lag_feature_states=(
            gpu_memory.lag_feature_states.cuda()
            if gpu_memory.lag_feature_states is not None
            else None
        ),
        lag_weights=(
            gpu_memory.lag_weights.cuda()
            if gpu_memory.lag_weights is not None
            else None
        ),
        encoder_architecture_fingerprint=(
            gpu_memory.encoder_architecture_fingerprint
        ),
        lineage_fingerprint=(
            gpu_memory.lineage_fingerprint
        ),
        schema_version=gpu_memory.schema_version,
    )

    with pytest.raises(ValueError, match="share one device"):
        NodeStateFusionInputs(
            alignment=alignment,
            static_state=cpu_static,
            memory_state=gpu_memory,
        )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is unavailable.",
)
def test_output_rejects_source_device_mismatch() -> None:
    inputs = _fusion_inputs(2)

    with pytest.raises(ValueError, match="share one device"):
        NodeStateFusionOutput(
            fused_state=torch.zeros(
                2,
                FUSED_DIM,
                device="cuda",
            ),
            source_inputs=inputs,
            projected_components={
                "static_state": torch.zeros(
                    2,
                    FUSED_DIM,
                    device="cuda",
                ),
            },
            fusion_mode="concat_projection",
            encoder_architecture_fingerprint="architecture",
            lineage_fingerprint="lineage",
        )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is unavailable.",
)
def test_output_rejects_projected_device_mismatch() -> None:
    inputs = _fusion_inputs(2)

    with pytest.raises(ValueError, match="share one device"):
        NodeStateFusionOutput(
            fused_state=torch.zeros(
                2,
                FUSED_DIM,
            ),
            source_inputs=inputs,
            projected_components={
                "static_state": torch.zeros(
                    2,
                    FUSED_DIM,
                    device="cuda",
                ),
            },
            fusion_mode="concat_projection",
            encoder_architecture_fingerprint="architecture",
            lineage_fingerprint="lineage",
        )
