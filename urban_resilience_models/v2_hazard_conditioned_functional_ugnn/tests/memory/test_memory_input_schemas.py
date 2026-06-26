"""
Consolidated contract tests for temporal-memory input schemas.

Target repository path:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            tests/
                memory/
                    test_memory_input_schemas.py

Implementations under test:
    memory/schemas/provenance.py
    memory/schemas/temporal_coordinates.py
    memory/schemas/history_inputs.py

This suite freezes the complete model-facing historical-input boundary:

- neutral node and feature-axis identities;
- source, architecture, parameter-snapshot, and execution provenance;
- absolute and relative temporal coordinates;
- event-sequence and regular-grid layouts;
- left, right, and unpadded sequence alignment;
- history values ``[N, T, D]``;
- timestep masks ``[N, T]``;
- feature-observation masks ``[N, T, D]``;
- explicit missing-value and zero-history policies;
- deterministic semantic, value, alignment, and lineage fingerprints;
- frozen metadata and validated reconstruction;
- device-preserving movement and optional history dtype conversion.

The suite does not test recurrent, Transformer, pooling, hazard retrieval, or
fusion behavior.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import MappingProxyType
from typing import Any

import pytest
import torch

from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.memory.schemas.history_inputs import (
    CANONICAL_HISTORY_MISSING_VALUE_POLICIES,
    CANONICAL_HISTORY_ZERO_LENGTH_POLICIES,
    FEATURE_OBSERVED_MASK_SEMANTICS,
    HISTORICAL_SEQUENCE_INPUTS_SCHEMA_VERSION,
    HISTORY_CANONICAL_PADDING_VALUE,
    HISTORY_INPUT_SCIENTIFIC_INTERPRETATION,
    HISTORY_VALUE_SEMANTICS,
    TIMESTEP_MASK_SEMANTICS,
    HistoricalSequenceInputs,
    HistoryInputs,
    HistoryMissingValuePolicy,
    HistoryZeroLengthPolicy,
    TemporalHistoryInputs,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.memory.schemas.provenance import (
    ArchitectureProvenance,
    ComputationProvenance,
    ExecutionLineage,
    FeatureAxisIdentity,
    MEMORY_ARCHITECTURE_PROVENANCE_SCHEMA_VERSION,
    MEMORY_COMPUTATION_PROVENANCE_SCHEMA_VERSION,
    MEMORY_EXECUTION_LINEAGE_SCHEMA_VERSION,
    MEMORY_PARAMETER_SNAPSHOT_POLICY,
    MEMORY_PARAMETER_SNAPSHOT_PROVENANCE_SCHEMA_VERSION,
    MEMORY_PROVENANCE_SCIENTIFIC_INTERPRETATION,
    MEMORY_SOURCE_PROVENANCE_SCHEMA_VERSION,
    MemoryArchitectureProvenance,
    MemoryComputationProvenance,
    MemoryExecutionLineage,
    MemoryParameterSnapshotProvenance,
    MemorySourceProvenance,
    NodeAxisIdentity,
    ParameterSnapshotProvenance,
    SourceDataProvenance,
    TEMPORAL_FEATURE_AXIS_SCHEMA_VERSION,
    TEMPORAL_NODE_AXIS_SCHEMA_VERSION,
    TemporalFeatureAxis,
    TemporalNodeAxis,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.memory.schemas.temporal_coordinates import (
    ABSOLUTE_TEMPORAL_COORDINATES_SCHEMA_VERSION,
    CANONICAL_ABSOLUTE_TEMPORAL_REFERENCE_KINDS,
    CANONICAL_RELATIVE_TEMPORAL_ANCHORS,
    CANONICAL_TEMPORAL_CHRONOLOGICAL_ORDERS,
    CANONICAL_TEMPORAL_COORDINATE_KINDS,
    CANONICAL_TEMPORAL_DUPLICATE_POLICIES,
    CANONICAL_TEMPORAL_LAYOUTS,
    CANONICAL_TEMPORAL_PADDING_DIRECTIONS,
    RELATIVE_TEMPORAL_COORDINATES_SCHEMA_VERSION,
    TEMPORAL_COORDINATE_CANONICAL_PADDING_VALUE,
    TEMPORAL_COORDINATE_SCIENTIFIC_INTERPRETATION,
    AbsoluteTemporalCoordinates,
    AbsoluteTemporalReferenceKind,
    RelativeTemporalAnchor,
    RelativeTemporalCoordinates,
    TemporalChronologicalOrder,
    TemporalCoordinateKind,
    TemporalCoordinates,
    TemporalDuplicatePolicy,
    TemporalLayout,
    TemporalPaddingDirection,
    temporal_coordinates_fingerprint,
    validate_temporal_coordinates,
)


N = 3
T = 4
D = 2


# =============================================================================
# Fixtures and factories
# =============================================================================


def _node_axis(
    *,
    node_ids: tuple[str, ...] = (
        "node-0",
        "node-1",
        "node-2",
    ),
    node_batch_index: torch.Tensor | None = None,
    graph_count: int = 2,
    graph_ids: tuple[str, ...] = (
        "graph-0",
        "graph-1",
    ),
    source_fingerprint: str | None = "node-source",
) -> TemporalNodeAxis:
    if node_batch_index is None:
        node_batch_index = torch.tensor(
            [0, 0, 1],
            dtype=torch.long,
        )

    return TemporalNodeAxis(
        node_ids=node_ids,
        node_batch_index=node_batch_index,
        graph_count=graph_count,
        graph_ids=graph_ids,
        source_fingerprint=source_fingerprint,
    )


def _feature_axis(
    *,
    feature_names: tuple[str, ...] = (
        "historical_burden",
        "reporting_intensity",
    ),
    source_fingerprint: str | None = "feature-source",
) -> TemporalFeatureAxis:
    return TemporalFeatureAxis(
        feature_names=feature_names,
        source_fingerprint=source_fingerprint,
    )


def _source_provenance(
    *,
    source_name: str = "montreal-monthly-panel",
    source_kind: str = "historical_node_panel",
    source_fingerprint: str = "history-source",
    dataset_version: str | None = "2021-2025-v1",
    preprocessing_fingerprint: str | None = "preprocess-v1",
    imputation_fingerprint: str | None = "imputation-v1",
    upstream_fingerprints: dict[str, str] | None = None,
) -> MemorySourceProvenance:
    return MemorySourceProvenance(
        source_name=source_name,
        source_kind=source_kind,
        source_fingerprint=source_fingerprint,
        dataset_version=dataset_version,
        dataset_snapshot_fingerprint="snapshot-v1",
        extraction_fingerprint="window-v1",
        preprocessing_fingerprint=preprocessing_fingerprint,
        imputation_fingerprint=imputation_fingerprint,
        upstream_fingerprints=(
            {
                "raw_panel": "raw-v1",
                "node_index": "nodes-v1",
            }
            if upstream_fingerprints is None
            else upstream_fingerprints
        ),
    )


def _right_mask() -> torch.Tensor:
    return torch.tensor(
        [
            [True, True, True, False],
            [True, True, False, False],
            [True, True, True, True],
        ],
        dtype=torch.bool,
    )


def _left_mask() -> torch.Tensor:
    return torch.tensor(
        [
            [False, True, True, True],
            [False, False, True, True],
            [True, True, True, True],
        ],
        dtype=torch.bool,
    )


def _full_mask() -> torch.Tensor:
    return torch.ones(
        (N, T),
        dtype=torch.bool,
    )


def _relative_values_for_right_mask(
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    return torch.tensor(
        [
            [-3.0, -2.0, -1.0, 0.0],
            [-7.0, -2.0, 0.0, 0.0],
            [-12.0, -6.0, -2.0, -0.5],
        ],
        dtype=dtype,
    )


def _relative_values_for_left_mask(
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    return torch.tensor(
        [
            [0.0, -3.0, -2.0, -1.0],
            [0.0, 0.0, -7.0, -2.0],
            [-12.0, -6.0, -2.0, -0.5],
        ],
        dtype=dtype,
    )


def _relative_coordinates(
    *,
    values: torch.Tensor | None = None,
    layout: TemporalLayout | str = TemporalLayout.EVENT_SEQUENCE,
    anchor: RelativeTemporalAnchor | str = (
        RelativeTemporalAnchor.PREDICTION_ORIGIN
    ),
    regular_step: float | None = None,
    duplicate_policy: TemporalDuplicatePolicy | str = (
        TemporalDuplicatePolicy.ALLOW_EQUAL
    ),
    require_nonpositive_history: bool = True,
) -> RelativeTemporalCoordinates:
    if values is None:
        values = _relative_values_for_right_mask()

    return RelativeTemporalCoordinates(
        values=values,
        unit="months",
        anchor=anchor,
        anchor_source_fingerprint="origin-v1",
        layout=layout,
        duplicate_policy=duplicate_policy,
        regular_step=regular_step,
        require_nonpositive_history=require_nonpositive_history,
    )


def _absolute_values_for_right_mask() -> torch.Tensor:
    return torch.tensor(
        [
            [100, 110, 120, 0],
            [90, 115, 0, 0],
            [80, 95, 110, 125],
        ],
        dtype=torch.long,
    )


def _absolute_values_for_left_mask() -> torch.Tensor:
    return torch.tensor(
        [
            [0, 100, 110, 120],
            [0, 0, 90, 115],
            [80, 95, 110, 125],
        ],
        dtype=torch.long,
    )


def _absolute_coordinates(
    *,
    values: torch.Tensor | None = None,
    layout: TemporalLayout | str = TemporalLayout.EVENT_SEQUENCE,
    regular_step: int | None = None,
    duplicate_policy: TemporalDuplicatePolicy | str = (
        TemporalDuplicatePolicy.ALLOW_EQUAL
    ),
    reference_kind: AbsoluteTemporalReferenceKind | str = (
        AbsoluteTemporalReferenceKind.PREDICTION_ORIGIN
    ),
    reference_time_values: torch.Tensor | None = None,
) -> AbsoluteTemporalCoordinates:
    if values is None:
        values = _absolute_values_for_right_mask()

    if (
        reference_time_values is None
        and reference_kind
        != AbsoluteTemporalReferenceKind.NONE
        and reference_kind != "none"
    ):
        reference_time_values = torch.tensor(
            [130, 130, 130],
            dtype=torch.long,
        )

    return AbsoluteTemporalCoordinates(
        values=values,
        unit="unix_days",
        calendar="proleptic_gregorian",
        timezone="UTC",
        reference_kind=reference_kind,
        reference_time_values=reference_time_values,
        layout=layout,
        duplicate_policy=duplicate_policy,
        regular_step=regular_step,
    )


def _history_for_mask(
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


def _feature_mask_for_timestep_mask(
    timestep_mask: torch.Tensor,
) -> torch.Tensor:
    return timestep_mask.unsqueeze(-1).expand(
        N,
        T,
        D,
    ).clone()


def _history_inputs(
    *,
    history: torch.Tensor | None = None,
    timestep_mask: torch.Tensor | None = None,
    node_axis: TemporalNodeAxis | None = None,
    feature_axis: TemporalFeatureAxis | None = None,
    temporal_coordinates: TemporalCoordinates | None = None,
    source_provenance: MemorySourceProvenance | None = None,
    feature_observed_mask: torch.Tensor | None = None,
    padding_direction: TemporalPaddingDirection | str = (
        TemporalPaddingDirection.RIGHT
    ),
    missing_value_policy: HistoryMissingValuePolicy | str = (
        HistoryMissingValuePolicy.UPSTREAM_IMPUTED
    ),
    zero_length_policy: HistoryZeroLengthPolicy | str = (
        HistoryZeroLengthPolicy.ERROR
    ),
) -> HistoricalSequenceInputs:
    if timestep_mask is None:
        timestep_mask = _right_mask()

    if history is None:
        history = _history_for_mask(
            timestep_mask
        )

    if node_axis is None:
        node_axis = _node_axis()

    if feature_axis is None:
        feature_axis = _feature_axis()

    if temporal_coordinates is None:
        temporal_coordinates = _relative_coordinates()

    if source_provenance is None:
        source_provenance = _source_provenance()

    return HistoricalSequenceInputs(
        history=history,
        timestep_mask=timestep_mask,
        node_axis=node_axis,
        feature_axis=feature_axis,
        temporal_coordinates=temporal_coordinates,
        source_provenance=source_provenance,
        feature_observed_mask=feature_observed_mask,
        padding_direction=padding_direction,
        missing_value_policy=missing_value_policy,
        zero_length_policy=zero_length_policy,
    )


def _architecture(
    *,
    architecture_fingerprint: str = "architecture-v1",
    configuration_fingerprint: str | None = "config-v1",
) -> MemoryArchitectureProvenance:
    return MemoryArchitectureProvenance(
        component_name="temporal_encoder",
        component_kind="gru",
        architecture_fingerprint=architecture_fingerprint,
        configuration_fingerprint=configuration_fingerprint,
        implementation_version="0.1",
        architecture_metadata={
            "hidden_dim": 16,
            "bidirectional": False,
            "dropout": 0.0,
            "nested": {
                "layers": [16, 16],
            },
        },
    )


def _parameter_snapshot(
    *,
    fingerprint: str = "parameters-v1",
) -> MemoryParameterSnapshotProvenance:
    return MemoryParameterSnapshotProvenance(
        parameter_snapshot_fingerprint=fingerprint,
        checkpoint_id="checkpoint-7",
        checkpoint_fingerprint="checkpoint-fingerprint-v1",
        training_step=100,
        parameter_count=1000,
        trainable_parameter_count=900,
    )


def _lineage(
    *,
    architecture_fingerprint: str | None = "architecture-v1",
    parameter_snapshot_fingerprint: str | None = None,
    configuration_fingerprint: str | None = "config-v1",
) -> MemoryExecutionLineage:
    return MemoryExecutionLineage(
        operation_name="encode_history",
        source_lineage_fingerprints=(
            "history-lineage-v1",
        ),
        architecture_fingerprint=architecture_fingerprint,
        parameter_snapshot_fingerprint=(
            parameter_snapshot_fingerprint
        ),
        configuration_fingerprint=configuration_fingerprint,
        node_axis_fingerprint="node-axis-v1",
        temporal_axis_fingerprint="time-axis-v1",
        feature_axis_fingerprint="feature-axis-v1",
        lineage_metadata={
            "training": False,
            "retention": ["sequence"],
        },
    )


# =============================================================================
# Published schema identity and aliases
# =============================================================================


def test_schema_versions_are_nonempty_strings() -> None:
    for value in (
        TEMPORAL_NODE_AXIS_SCHEMA_VERSION,
        TEMPORAL_FEATURE_AXIS_SCHEMA_VERSION,
        MEMORY_SOURCE_PROVENANCE_SCHEMA_VERSION,
        MEMORY_ARCHITECTURE_PROVENANCE_SCHEMA_VERSION,
        MEMORY_PARAMETER_SNAPSHOT_PROVENANCE_SCHEMA_VERSION,
        MEMORY_EXECUTION_LINEAGE_SCHEMA_VERSION,
        MEMORY_COMPUTATION_PROVENANCE_SCHEMA_VERSION,
        ABSOLUTE_TEMPORAL_COORDINATES_SCHEMA_VERSION,
        RELATIVE_TEMPORAL_COORDINATES_SCHEMA_VERSION,
        HISTORICAL_SEQUENCE_INPUTS_SCHEMA_VERSION,
    ):
        assert isinstance(value, str)
        assert value.strip()


def test_compact_type_aliases_preserve_exact_classes() -> None:
    assert NodeAxisIdentity is TemporalNodeAxis
    assert FeatureAxisIdentity is TemporalFeatureAxis
    assert SourceDataProvenance is MemorySourceProvenance
    assert ArchitectureProvenance is MemoryArchitectureProvenance
    assert ParameterSnapshotProvenance is (
        MemoryParameterSnapshotProvenance
    )
    assert ExecutionLineage is MemoryExecutionLineage
    assert ComputationProvenance is MemoryComputationProvenance
    assert TemporalHistoryInputs is HistoricalSequenceInputs
    assert HistoryInputs is HistoricalSequenceInputs


def test_scientific_interpretation_constants_are_explicit() -> None:
    for value in (
        MEMORY_PARAMETER_SNAPSHOT_POLICY,
        MEMORY_PROVENANCE_SCIENTIFIC_INTERPRETATION,
        TEMPORAL_COORDINATE_SCIENTIFIC_INTERPRETATION,
        HISTORY_INPUT_SCIENTIFIC_INTERPRETATION,
        HISTORY_VALUE_SEMANTICS,
        TIMESTEP_MASK_SEMANTICS,
        FEATURE_OBSERVED_MASK_SEMANTICS,
    ):
        assert isinstance(value, str)
        assert value.strip()


def test_canonical_padding_values_are_zero() -> None:
    assert TEMPORAL_COORDINATE_CANONICAL_PADDING_VALUE == 0
    assert HISTORY_CANONICAL_PADDING_VALUE == 0


def test_canonical_policy_vocabularies_match_enums() -> None:
    assert CANONICAL_HISTORY_MISSING_VALUE_POLICIES == tuple(
        value.value
        for value in HistoryMissingValuePolicy
    )
    assert CANONICAL_HISTORY_ZERO_LENGTH_POLICIES == tuple(
        value.value
        for value in HistoryZeroLengthPolicy
    )
    assert CANONICAL_TEMPORAL_COORDINATE_KINDS == tuple(
        value.value
        for value in TemporalCoordinateKind
    )
    assert CANONICAL_TEMPORAL_LAYOUTS == tuple(
        value.value
        for value in TemporalLayout
    )
    assert CANONICAL_TEMPORAL_CHRONOLOGICAL_ORDERS == tuple(
        value.value
        for value in TemporalChronologicalOrder
    )
    assert CANONICAL_TEMPORAL_PADDING_DIRECTIONS == tuple(
        value.value
        for value in TemporalPaddingDirection
    )
    assert CANONICAL_TEMPORAL_DUPLICATE_POLICIES == tuple(
        value.value
        for value in TemporalDuplicatePolicy
    )
    assert CANONICAL_ABSOLUTE_TEMPORAL_REFERENCE_KINDS == tuple(
        value.value
        for value in AbsoluteTemporalReferenceKind
    )
    assert CANONICAL_RELATIVE_TEMPORAL_ANCHORS == tuple(
        value.value
        for value in RelativeTemporalAnchor
    )


# =============================================================================
# TemporalNodeAxis
# =============================================================================


def test_node_axis_preserves_alignment_and_identity() -> None:
    axis = _node_axis()

    assert axis.node_count == 3
    assert axis.item_count == 3
    assert axis.graph_count == 2
    assert axis.graph_aligned
    assert axis.device == torch.device("cpu")
    assert axis.node_ids == (
        "node-0",
        "node-1",
        "node-2",
    )
    assert axis.graph_ids == (
        "graph-0",
        "graph-1",
    )

    assert axis.semantic_dict() == {
        "schema_version": TEMPORAL_NODE_AXIS_SCHEMA_VERSION,
        "axis_name": "temporal_node_axis",
        "node_count": 3,
        "node_ids": [
            "node-0",
            "node-1",
            "node-2",
        ],
        "graph_count": 2,
        "graph_ids": [
            "graph-0",
            "graph-1",
        ],
        "source_fingerprint": "node-source",
    }


def test_node_axis_fingerprints_are_deterministic() -> None:
    first = _node_axis()
    second = _node_axis()

    assert first.semantic_fingerprint() == second.semantic_fingerprint()
    assert first.value_fingerprint() == second.value_fingerprint()
    assert first.fingerprint() == second.fingerprint()
    assert first.alignment_fingerprint() == first.fingerprint()


def test_node_axis_fingerprint_changes_with_node_order() -> None:
    first = _node_axis()
    second = _node_axis(
        node_ids=(
            "node-1",
            "node-0",
            "node-2",
        )
    )

    assert first.semantic_fingerprint() != second.semantic_fingerprint()
    assert first.fingerprint() != second.fingerprint()


def test_node_axis_value_fingerprint_changes_with_graph_membership() -> None:
    first = _node_axis()
    second = _node_axis(
        node_batch_index=torch.tensor(
            [0, 1, 1],
            dtype=torch.long,
        )
    )

    assert first.semantic_fingerprint() == second.semantic_fingerprint()
    assert first.value_fingerprint() != second.value_fingerprint()
    assert first.fingerprint() != second.fingerprint()


@pytest.mark.parametrize(
    "node_ids",
    [
        (),
        ("node-0", "", "node-2"),
        ("node-0", "node-0", "node-2"),
    ],
)
def test_node_axis_rejects_invalid_node_ids(
    node_ids: tuple[str, ...],
) -> None:
    with pytest.raises(
        (TypeError, ValueError),
    ):
        _node_axis(
            node_ids=node_ids,
            node_batch_index=torch.zeros(
                len(node_ids),
                dtype=torch.long,
            ),
            graph_count=1,
            graph_ids=("graph-0",),
        )


@pytest.mark.parametrize(
    "node_batch_index",
    [
        torch.tensor(
            [[0, 0, 1]],
            dtype=torch.long,
        ),
        torch.tensor(
            [0.0, 0.0, 1.0],
            dtype=torch.float32,
        ),
        torch.tensor(
            [0, 1],
            dtype=torch.long,
        ),
    ],
)
def test_node_axis_rejects_invalid_membership_tensor(
    node_batch_index: torch.Tensor,
) -> None:
    with pytest.raises(
        (TypeError, ValueError),
    ):
        _node_axis(
            node_batch_index=node_batch_index
        )


@pytest.mark.parametrize(
    (
        "node_batch_index",
        "graph_count",
    ),
    [
        (
            torch.tensor(
                [-1, 0, 1],
                dtype=torch.long,
            ),
            2,
        ),
        (
            torch.tensor(
                [0, 0, 2],
                dtype=torch.long,
            ),
            2,
        ),
        (
            torch.tensor(
                [0, 0, 2],
                dtype=torch.long,
            ),
            3,
        ),
        (
            torch.tensor(
                [0, 0, 0],
                dtype=torch.long,
            ),
            2,
        ),
    ],
)
def test_node_axis_rejects_invalid_packed_graph_ids(
    node_batch_index: torch.Tensor,
    graph_count: int,
) -> None:
    with pytest.raises(ValueError):
        _node_axis(
            node_batch_index=node_batch_index,
            graph_count=graph_count,
            graph_ids=tuple(
                f"graph-{index}"
                for index in range(graph_count)
            ),
        )


@pytest.mark.parametrize(
    "graph_count",
    [
        0,
        -1,
        True,
        1.5,
    ],
)
def test_node_axis_rejects_invalid_graph_count(
    graph_count: Any,
) -> None:
    with pytest.raises(
        (TypeError, ValueError),
    ):
        _node_axis(
            node_batch_index=torch.tensor(
                [0, 0, 0],
                dtype=torch.long,
            ),
            graph_count=graph_count,
            graph_ids=(),
        )


def test_node_axis_rejects_invalid_graph_ids() -> None:
    with pytest.raises(ValueError):
        _node_axis(
            graph_ids=(
                "graph-0",
            )
        )

    with pytest.raises(ValueError):
        _node_axis(
            graph_ids=(
                "graph-0",
                "graph-0",
            )
        )


def test_node_axis_is_frozen() -> None:
    axis = _node_axis()

    with pytest.raises(
        (FrozenInstanceError, AttributeError),
    ):
        axis.graph_count = 3  # type: ignore[misc]


def test_node_axis_to_reconstructs_and_preserves_fingerprints() -> None:
    axis = _node_axis()
    moved = axis.to("cpu")

    assert moved is not axis
    assert moved.node_ids == axis.node_ids
    assert torch.equal(
        moved.node_batch_index,
        axis.node_batch_index,
    )
    assert moved.fingerprint() == axis.fingerprint()


# =============================================================================
# TemporalFeatureAxis
# =============================================================================


def test_feature_axis_preserves_ordered_identity() -> None:
    axis = _feature_axis()

    assert axis.feature_dim == 2
    assert axis.feature_names == (
        "historical_burden",
        "reporting_intensity",
    )
    assert axis.semantic_dict() == {
        "schema_version": TEMPORAL_FEATURE_AXIS_SCHEMA_VERSION,
        "axis_name": "temporal_feature_axis",
        "feature_dim": 2,
        "feature_names": [
            "historical_burden",
            "reporting_intensity",
        ],
        "source_fingerprint": "feature-source",
    }


def test_feature_axis_fingerprint_is_order_sensitive() -> None:
    first = _feature_axis()
    second = _feature_axis(
        feature_names=(
            "reporting_intensity",
            "historical_burden",
        )
    )

    assert first.fingerprint() != second.fingerprint()


@pytest.mark.parametrize(
    "feature_names",
    [
        (),
        ("feature-a", ""),
        ("feature-a", "feature-a"),
    ],
)
def test_feature_axis_rejects_invalid_names(
    feature_names: tuple[str, ...],
) -> None:
    with pytest.raises(
        (TypeError, ValueError),
    ):
        _feature_axis(
            feature_names=feature_names
        )


def test_feature_axis_is_frozen() -> None:
    axis = _feature_axis()

    with pytest.raises(
        (FrozenInstanceError, AttributeError),
    ):
        axis.axis_name = "changed"  # type: ignore[misc]


# =============================================================================
# Source and computation provenance
# =============================================================================


def test_source_provenance_preserves_complete_lineage() -> None:
    provenance = _source_provenance()
    payload = provenance.provenance_dict()

    assert payload["source_name"] == "montreal-monthly-panel"
    assert payload["source_kind"] == "historical_node_panel"
    assert payload["source_fingerprint"] == "history-source"
    assert payload["dataset_version"] == "2021-2025-v1"
    assert payload["dataset_snapshot_fingerprint"] == "snapshot-v1"
    assert payload["extraction_fingerprint"] == "window-v1"
    assert payload["preprocessing_fingerprint"] == "preprocess-v1"
    assert payload["imputation_fingerprint"] == "imputation-v1"
    assert payload["upstream_fingerprints"] == {
        "raw_panel": "raw-v1",
        "node_index": "nodes-v1",
    }
    assert provenance.lineage_fingerprint() == provenance.fingerprint()


def test_source_provenance_defensively_copies_and_freezes_mapping() -> None:
    original = {
        "raw_panel": "raw-v1",
    }
    provenance = _source_provenance(
        upstream_fingerprints=original
    )

    original["raw_panel"] = "changed"

    assert isinstance(
        provenance.upstream_fingerprints,
        MappingProxyType,
    )
    assert provenance.upstream_fingerprints["raw_panel"] == "raw-v1"

    with pytest.raises(TypeError):
        provenance.upstream_fingerprints["new"] = "value"  # type: ignore[index]


@pytest.mark.parametrize(
    (
        "field_name",
        "value",
    ),
    [
        ("source_name", ""),
        ("source_kind", ""),
        ("source_fingerprint", ""),
        ("dataset_version", ""),
        ("preprocessing_fingerprint", ""),
    ],
)
def test_source_provenance_rejects_empty_identity_fields(
    field_name: str,
    value: str,
) -> None:
    kwargs = {
        "source_name": "source",
        "source_kind": "panel",
        "source_fingerprint": "fingerprint",
    }
    kwargs[field_name] = value

    with pytest.raises(ValueError):
        MemorySourceProvenance(
            **kwargs,
        )


def test_source_provenance_rejects_invalid_upstream_mapping() -> None:
    with pytest.raises(ValueError):
        _source_provenance(
            upstream_fingerprints={
                "": "value",
            }
        )

    with pytest.raises(ValueError):
        _source_provenance(
            upstream_fingerprints={
                "artifact": "",
            }
        )


def test_architecture_provenance_freezes_nested_metadata() -> None:
    metadata = {
        "hidden_dim": 16,
        "layers": [16, 16],
        "nested": {
            "dropout": 0.1,
        },
    }
    provenance = MemoryArchitectureProvenance(
        component_name="encoder",
        component_kind="gru",
        architecture_fingerprint="arch-v1",
        architecture_metadata=metadata,
    )

    metadata["hidden_dim"] = 99
    metadata["layers"].append(32)

    assert isinstance(
        provenance.architecture_metadata,
        MappingProxyType,
    )
    assert provenance.architecture_metadata["hidden_dim"] == 16
    assert provenance.architecture_metadata["layers"] == (
        16,
        16,
    )
    assert isinstance(
        provenance.architecture_metadata["nested"],
        MappingProxyType,
    )

    with pytest.raises(TypeError):
        provenance.architecture_metadata["new"] = 1  # type: ignore[index]


@pytest.mark.parametrize(
    "invalid_metadata",
    [
        {
            "tensor": torch.tensor(
                [1.0]
            ),
        },
        {
            "bad": float("nan"),
        },
        {
            "bad": float("inf"),
        },
        {
            "": "value",
        },
    ],
)
def test_architecture_provenance_rejects_invalid_metadata(
    invalid_metadata: dict[str, Any],
) -> None:
    with pytest.raises(
        (TypeError, ValueError),
    ):
        MemoryArchitectureProvenance(
            component_name="encoder",
            component_kind="gru",
            architecture_fingerprint="arch-v1",
            architecture_metadata=invalid_metadata,
        )


def test_architecture_fingerprint_changes_with_metadata() -> None:
    first = _architecture()
    second = MemoryArchitectureProvenance(
        component_name="temporal_encoder",
        component_kind="gru",
        architecture_fingerprint="architecture-v1",
        configuration_fingerprint="config-v1",
        implementation_version="0.1",
        architecture_metadata={
            "hidden_dim": 32,
        },
    )

    assert first.fingerprint() != second.fingerprint()


def test_parameter_snapshot_preserves_optional_artifact_identity() -> None:
    snapshot = _parameter_snapshot()
    payload = snapshot.provenance_dict()

    assert payload["parameter_snapshot_fingerprint"] == "parameters-v1"
    assert payload["checkpoint_id"] == "checkpoint-7"
    assert payload["training_step"] == 100
    assert payload["parameter_count"] == 1000
    assert payload["trainable_parameter_count"] == 900
    assert payload["snapshot_policy"] == (
        MEMORY_PARAMETER_SNAPSHOT_POLICY
    )


@pytest.mark.parametrize(
    (
        "field_name",
        "value",
    ),
    [
        ("training_step", -1),
        ("parameter_count", -1),
        ("trainable_parameter_count", -1),
    ],
)
def test_parameter_snapshot_rejects_negative_counts(
    field_name: str,
    value: int,
) -> None:
    kwargs: dict[str, Any] = {
        "parameter_snapshot_fingerprint": "parameters-v1",
    }
    kwargs[field_name] = value

    with pytest.raises(ValueError):
        MemoryParameterSnapshotProvenance(
            **kwargs,
        )


def test_parameter_snapshot_rejects_trainable_count_above_total() -> None:
    with pytest.raises(ValueError):
        MemoryParameterSnapshotProvenance(
            parameter_snapshot_fingerprint="parameters-v1",
            parameter_count=10,
            trainable_parameter_count=11,
        )


def test_execution_lineage_preserves_axes_and_metadata() -> None:
    lineage = _lineage()
    payload = lineage.lineage_dict()

    assert payload["operation_name"] == "encode_history"
    assert payload["source_lineage_fingerprints"] == [
        "history-lineage-v1",
    ]
    assert payload["architecture_fingerprint"] == "architecture-v1"
    assert payload["node_axis_fingerprint"] == "node-axis-v1"
    assert payload["temporal_axis_fingerprint"] == "time-axis-v1"
    assert payload["feature_axis_fingerprint"] == "feature-axis-v1"
    assert payload["lineage_metadata"] == {
        "training": False,
        "retention": [
            "sequence",
        ],
    }
    assert lineage.fingerprint() == lineage.lineage_fingerprint()


def test_execution_lineage_freezes_nested_metadata() -> None:
    lineage = _lineage()

    assert isinstance(
        lineage.lineage_metadata,
        MappingProxyType,
    )
    assert lineage.lineage_metadata["retention"] == (
        "sequence",
    )

    with pytest.raises(TypeError):
        lineage.lineage_metadata["new"] = True  # type: ignore[index]


@pytest.mark.parametrize(
    "sources",
    [
        (),
        ("",),
        ("same", "same"),
    ],
)
def test_execution_lineage_rejects_invalid_sources(
    sources: tuple[str, ...],
) -> None:
    with pytest.raises(
        (TypeError, ValueError),
    ):
        MemoryExecutionLineage(
            operation_name="encode",
            source_lineage_fingerprints=sources,
        )


def test_computation_provenance_without_parameter_snapshot() -> None:
    architecture = _architecture()
    lineage = _lineage()
    computation = MemoryComputationProvenance(
        architecture=architecture,
        lineage=lineage,
    )

    assert computation.architecture is architecture
    assert computation.lineage is lineage
    assert computation.parameter_snapshot is None
    assert computation.architecture_fingerprint == "architecture-v1"
    assert computation.parameter_snapshot_fingerprint is None
    assert computation.lineage_fingerprint == (
        lineage.lineage_fingerprint()
    )
    assert computation.provenance_dict()[
        "scientific_interpretation"
    ] == MEMORY_PROVENANCE_SCIENTIFIC_INTERPRETATION


def test_computation_provenance_with_parameter_snapshot() -> None:
    architecture = _architecture()
    snapshot = _parameter_snapshot()
    lineage = _lineage(
        parameter_snapshot_fingerprint=(
            snapshot.parameter_snapshot_fingerprint
        )
    )
    computation = MemoryComputationProvenance(
        architecture=architecture,
        lineage=lineage,
        parameter_snapshot=snapshot,
    )

    assert computation.parameter_snapshot is snapshot
    assert computation.parameter_snapshot_fingerprint == "parameters-v1"


def test_computation_provenance_rejects_architecture_mismatch() -> None:
    with pytest.raises(ValueError):
        MemoryComputationProvenance(
            architecture=_architecture(
                architecture_fingerprint="architecture-a",
            ),
            lineage=_lineage(
                architecture_fingerprint="architecture-b",
            ),
        )


def test_computation_provenance_rejects_configuration_mismatch() -> None:
    with pytest.raises(ValueError):
        MemoryComputationProvenance(
            architecture=_architecture(
                configuration_fingerprint="config-a",
            ),
            lineage=_lineage(
                configuration_fingerprint="config-b",
            ),
        )


def test_computation_provenance_rejects_unresolved_snapshot_reference() -> None:
    with pytest.raises(ValueError):
        MemoryComputationProvenance(
            architecture=_architecture(),
            lineage=_lineage(
                parameter_snapshot_fingerprint="parameters-v1",
            ),
        )


def test_computation_provenance_rejects_snapshot_mismatch() -> None:
    with pytest.raises(ValueError):
        MemoryComputationProvenance(
            architecture=_architecture(),
            lineage=_lineage(
                parameter_snapshot_fingerprint="parameters-a",
            ),
            parameter_snapshot=_parameter_snapshot(
                fingerprint="parameters-b",
            ),
        )


def test_computation_provenance_fingerprint_is_deterministic() -> None:
    first = MemoryComputationProvenance(
        architecture=_architecture(),
        lineage=_lineage(),
    )
    second = MemoryComputationProvenance(
        architecture=_architecture(),
        lineage=_lineage(),
    )

    assert first.fingerprint() == second.fingerprint()


# =============================================================================
# Absolute temporal coordinates
# =============================================================================


def test_absolute_coordinates_preserve_semantics() -> None:
    coordinates = _absolute_coordinates()

    assert coordinates.coordinate_kind == TemporalCoordinateKind.ABSOLUTE
    assert coordinates.node_count == N
    assert coordinates.sequence_length == T
    assert coordinates.shape == (N, T)
    assert coordinates.device == torch.device("cpu")
    assert coordinates.dtype == torch.long
    assert coordinates.layout == TemporalLayout.EVENT_SEQUENCE
    assert coordinates.chronological_order == (
        TemporalChronologicalOrder.OLDEST_TO_NEWEST
    )
    assert coordinates.reference_kind == (
        AbsoluteTemporalReferenceKind.PREDICTION_ORIGIN
    )

    semantic = coordinates.semantic_dict()
    assert semantic["coordinate_kind"] == "absolute"
    assert semantic["unit"] == "unix_days"
    assert semantic["calendar"] == "proleptic_gregorian"
    assert semantic["timezone"] == "UTC"
    assert semantic["has_reference_time_values"] is True


def test_absolute_coordinates_validate_right_padding() -> None:
    coordinates = _absolute_coordinates()
    coordinates.validate_against_mask(
        _right_mask(),
        padding_direction=TemporalPaddingDirection.RIGHT,
    )
    validate_temporal_coordinates(
        coordinates,
        _right_mask(),
        padding_direction="right",
    )


def test_absolute_coordinates_validate_left_padding() -> None:
    coordinates = _absolute_coordinates(
        values=_absolute_values_for_left_mask(),
    )
    coordinates.validate_against_mask(
        _left_mask(),
        padding_direction=TemporalPaddingDirection.LEFT,
    )


def test_absolute_coordinates_validate_unpadded_history() -> None:
    coordinates = _absolute_coordinates(
        values=torch.tensor(
            [
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
            ],
            dtype=torch.long,
        ),
        reference_time_values=torch.tensor(
            [4, 8, 12],
            dtype=torch.long,
        ),
    )
    coordinates.validate_against_mask(
        _full_mask(),
        padding_direction=TemporalPaddingDirection.NONE,
    )


@pytest.mark.parametrize(
    "values",
    [
        torch.zeros(
            (N, T),
            dtype=torch.float32,
        ),
        torch.zeros(
            (N, T, 1),
            dtype=torch.long,
        ),
        torch.zeros(
            (0, T),
            dtype=torch.long,
        ),
        torch.zeros(
            (N, 0),
            dtype=torch.long,
        ),
    ],
)
def test_absolute_coordinates_reject_invalid_values(
    values: torch.Tensor,
) -> None:
    with pytest.raises(
        (TypeError, ValueError),
    ):
        _absolute_coordinates(
            values=values,
        )


def test_absolute_coordinates_require_reference_values_when_declared() -> None:
    with pytest.raises(ValueError):
        AbsoluteTemporalCoordinates(
            values=_absolute_values_for_right_mask(),
            unit="unix_days",
            reference_kind=(
                AbsoluteTemporalReferenceKind.PREDICTION_ORIGIN
            ),
            reference_time_values=None,
        )


def test_absolute_coordinates_forbid_reference_values_for_none_kind() -> None:
    with pytest.raises(ValueError):
        _absolute_coordinates(
            reference_kind=AbsoluteTemporalReferenceKind.NONE,
            reference_time_values=torch.tensor(
                [1, 1, 1],
                dtype=torch.long,
            ),
        )


@pytest.mark.parametrize(
    "reference_values",
    [
        torch.tensor(
            [1.0, 1.0, 1.0],
            dtype=torch.float32,
        ),
        torch.tensor(
            [[1, 1, 1]],
            dtype=torch.long,
        ),
        torch.tensor(
            [1, 1],
            dtype=torch.long,
        ),
    ],
)
def test_absolute_coordinates_reject_invalid_reference_vector(
    reference_values: torch.Tensor,
) -> None:
    with pytest.raises(
        (TypeError, ValueError),
    ):
        _absolute_coordinates(
            reference_time_values=reference_values,
        )


def test_absolute_event_sequence_forbids_regular_step() -> None:
    with pytest.raises(ValueError):
        _absolute_coordinates(
            regular_step=1,
        )


def test_absolute_regular_grid_requires_integer_step() -> None:
    with pytest.raises(ValueError):
        _absolute_coordinates(
            layout=TemporalLayout.REGULAR_GRID,
            regular_step=None,
        )

    with pytest.raises(TypeError):
        _absolute_coordinates(
            layout=TemporalLayout.REGULAR_GRID,
            regular_step=1.5,  # type: ignore[arg-type]
        )


def test_absolute_regular_grid_validates_spacing() -> None:
    values = torch.tensor(
        [
            [10, 20, 30, 0],
            [5, 15, 0, 0],
            [2, 12, 22, 32],
        ],
        dtype=torch.long,
    )
    coordinates = _absolute_coordinates(
        values=values,
        layout=TemporalLayout.REGULAR_GRID,
        regular_step=10,
        reference_time_values=torch.tensor(
            [40, 40, 40],
            dtype=torch.long,
        ),
    )

    coordinates.validate_against_mask(
        _right_mask(),
        padding_direction="right",
    )


def test_absolute_regular_grid_rejects_irregular_spacing() -> None:
    coordinates = _absolute_coordinates(
        values=_absolute_values_for_right_mask(),
        layout=TemporalLayout.REGULAR_GRID,
        regular_step=10,
    )

    with pytest.raises(ValueError):
        coordinates.validate_against_mask(
            _right_mask(),
            padding_direction="right",
        )


def test_absolute_coordinates_reject_nonzero_padding() -> None:
    values = _absolute_values_for_right_mask()
    values[0, 3] = 999
    coordinates = _absolute_coordinates(
        values=values,
    )

    with pytest.raises(ValueError):
        coordinates.validate_against_mask(
            _right_mask(),
            padding_direction="right",
        )


def test_absolute_coordinates_reject_internal_padding() -> None:
    mask = torch.tensor(
        [
            [True, False, True, False],
            [True, True, False, False],
            [True, True, True, True],
        ],
        dtype=torch.bool,
    )
    coordinates = _absolute_coordinates()

    with pytest.raises(ValueError):
        coordinates.validate_against_mask(
            mask,
            padding_direction="right",
        )


def test_absolute_coordinates_reject_wrong_padding_direction() -> None:
    coordinates = _absolute_coordinates()

    with pytest.raises(ValueError):
        coordinates.validate_against_mask(
            _right_mask(),
            padding_direction="left",
        )

    with pytest.raises(ValueError):
        coordinates.validate_against_mask(
            _right_mask(),
            padding_direction="none",
        )


def test_absolute_coordinates_reject_nonmonotonic_history() -> None:
    values = _absolute_values_for_right_mask()
    values[0, :3] = torch.tensor(
        [100, 90, 120],
        dtype=torch.long,
    )
    coordinates = _absolute_coordinates(
        values=values,
    )

    with pytest.raises(ValueError):
        coordinates.validate_against_mask(
            _right_mask(),
            padding_direction="right",
        )


def test_absolute_duplicate_policy_can_be_strict() -> None:
    values = _absolute_values_for_right_mask()
    values[0, :3] = torch.tensor(
        [100, 100, 120],
        dtype=torch.long,
    )
    coordinates = _absolute_coordinates(
        values=values,
        duplicate_policy=TemporalDuplicatePolicy.ERROR,
    )

    with pytest.raises(ValueError):
        coordinates.validate_against_mask(
            _right_mask(),
            padding_direction="right",
        )


def test_absolute_coordinates_reject_history_after_reference() -> None:
    coordinates = _absolute_coordinates(
        reference_time_values=torch.tensor(
            [115, 130, 130],
            dtype=torch.long,
        ),
    )

    with pytest.raises(ValueError):
        coordinates.validate_against_mask(
            _right_mask(),
            padding_direction="right",
        )


def test_absolute_coordinate_fingerprints_are_deterministic_and_sensitive() -> None:
    first = _absolute_coordinates()
    second = _absolute_coordinates()
    changed = _absolute_coordinates(
        values=torch.tensor(
            [
                [101, 110, 120, 0],
                [90, 115, 0, 0],
                [80, 95, 110, 125],
            ],
            dtype=torch.long,
        )
    )

    assert first.semantic_fingerprint() == second.semantic_fingerprint()
    assert first.value_fingerprint() == second.value_fingerprint()
    assert first.fingerprint() == second.fingerprint()
    assert temporal_coordinates_fingerprint(first) == first.fingerprint()
    assert first.value_fingerprint() != changed.value_fingerprint()
    assert first.fingerprint() != changed.fingerprint()


def test_absolute_coordinates_to_preserves_identity() -> None:
    coordinates = _absolute_coordinates()
    moved = coordinates.to("cpu")

    assert moved is not coordinates
    assert moved.fingerprint() == coordinates.fingerprint()
    assert torch.equal(
        moved.values,
        coordinates.values,
    )


# =============================================================================
# Relative temporal coordinates
# =============================================================================


def test_relative_coordinates_preserve_staleness() -> None:
    coordinates = _relative_coordinates()

    assert coordinates.coordinate_kind == TemporalCoordinateKind.RELATIVE
    assert coordinates.anchor == RelativeTemporalAnchor.PREDICTION_ORIGIN
    assert coordinates.node_count == N
    assert coordinates.sequence_length == T
    assert coordinates.shape == (N, T)
    assert coordinates.dtype == torch.float32
    assert coordinates.require_nonpositive_history

    # The latest valid observations intentionally have different offsets.
    mask = _right_mask()
    latest = tuple(
        float(
            coordinates.values[index][
                mask[index]
            ][-1].item()
        )
        for index in range(N)
    )
    assert latest == (
        -1.0,
        -2.0,
        -0.5,
    )


def test_relative_coordinates_validate_right_padding() -> None:
    coordinates = _relative_coordinates()
    coordinates.validate_against_mask(
        _right_mask(),
        padding_direction="right",
    )


def test_relative_coordinates_validate_left_padding() -> None:
    coordinates = _relative_coordinates(
        values=_relative_values_for_left_mask(),
    )
    coordinates.validate_against_mask(
        _left_mask(),
        padding_direction="left",
    )


@pytest.mark.parametrize(
    "values",
    [
        torch.zeros(
            (N, T),
            dtype=torch.long,
        ),
        torch.zeros(
            (N, T, 1),
            dtype=torch.float32,
        ),
        torch.tensor(
            [
                [0.0, float("nan"), 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        ),
        torch.tensor(
            [
                [0.0, float("inf"), 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        ),
    ],
)
def test_relative_coordinates_reject_invalid_values(
    values: torch.Tensor,
) -> None:
    with pytest.raises(
        (TypeError, ValueError),
    ):
        _relative_coordinates(
            values=values,
        )


def test_relative_coordinates_reject_positive_history_by_default() -> None:
    values = _relative_values_for_right_mask()
    values[0, 0] = 1.0

    with pytest.raises(ValueError):
        _relative_coordinates(
            values=values,
        )


def test_relative_coordinates_can_explicitly_allow_positive_offsets() -> None:
    values = torch.tensor(
        [
            [1.0, 2.0, 3.0, 0.0],
            [1.0, 2.0, 0.0, 0.0],
            [1.0, 2.0, 3.0, 4.0],
        ],
        dtype=torch.float32,
    )
    coordinates = _relative_coordinates(
        values=values,
        require_nonpositive_history=False,
    )

    coordinates.validate_against_mask(
        _right_mask(),
        padding_direction="right",
    )


def test_relative_latest_observation_anchor_requires_zero_latest_value() -> None:
    coordinates = _relative_coordinates(
        anchor=RelativeTemporalAnchor.LATEST_OBSERVATION,
    )

    with pytest.raises(ValueError):
        coordinates.validate_against_mask(
            _right_mask(),
            padding_direction="right",
        )


def test_relative_latest_observation_anchor_accepts_zero_latest_value() -> None:
    values = torch.tensor(
        [
            [-2.0, -1.0, 0.0, 0.0],
            [-5.0, 0.0, 0.0, 0.0],
            [-3.0, -2.0, -1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    coordinates = _relative_coordinates(
        values=values,
        anchor=RelativeTemporalAnchor.LATEST_OBSERVATION,
    )

    coordinates.validate_against_mask(
        _right_mask(),
        padding_direction="right",
    )


def test_relative_event_sequence_forbids_regular_step() -> None:
    with pytest.raises(ValueError):
        _relative_coordinates(
            regular_step=1.0,
        )


@pytest.mark.parametrize(
    "regular_step",
    [
        None,
        0.0,
        -1.0,
        float("nan"),
    ],
)
def test_relative_regular_grid_rejects_invalid_step(
    regular_step: float | None,
) -> None:
    with pytest.raises(
        (TypeError, ValueError),
    ):
        _relative_coordinates(
            layout=TemporalLayout.REGULAR_GRID,
            regular_step=regular_step,
        )


def test_relative_regular_grid_validates_spacing() -> None:
    values = torch.tensor(
        [
            [-3.0, -2.0, -1.0, 0.0],
            [-2.0, -1.0, 0.0, 0.0],
            [-4.0, -3.0, -2.0, -1.0],
        ],
        dtype=torch.float32,
    )
    coordinates = _relative_coordinates(
        values=values,
        layout=TemporalLayout.REGULAR_GRID,
        regular_step=1.0,
    )

    coordinates.validate_against_mask(
        _right_mask(),
        padding_direction="right",
    )


def test_relative_regular_grid_rejects_irregular_spacing() -> None:
    coordinates = _relative_coordinates(
        layout=TemporalLayout.REGULAR_GRID,
        regular_step=1.0,
    )

    with pytest.raises(ValueError):
        coordinates.validate_against_mask(
            _right_mask(),
            padding_direction="right",
        )


def test_relative_coordinates_reject_nonzero_padding() -> None:
    values = _relative_values_for_right_mask()
    values[1, 3] = -999.0
    coordinates = _relative_coordinates(
        values=values,
    )

    with pytest.raises(ValueError):
        coordinates.validate_against_mask(
            _right_mask(),
            padding_direction="right",
        )


def test_relative_coordinates_reject_nonmonotonic_history() -> None:
    values = _relative_values_for_right_mask()
    values[0, :3] = torch.tensor(
        [-3.0, -4.0, -1.0]
    )
    coordinates = _relative_coordinates(
        values=values,
    )

    with pytest.raises(ValueError):
        coordinates.validate_against_mask(
            _right_mask(),
            padding_direction="right",
        )


def test_relative_coordinate_fingerprints_change_with_anchor() -> None:
    first = _relative_coordinates()
    second = _relative_coordinates(
        anchor=RelativeTemporalAnchor.EXPLICIT_REFERENCE_TIME,
    )

    assert first.semantic_fingerprint() != second.semantic_fingerprint()
    assert first.fingerprint() != second.fingerprint()


def test_relative_coordinates_to_preserves_dtype_and_fingerprint() -> None:
    coordinates = _relative_coordinates(
        values=_relative_values_for_right_mask(
            dtype=torch.float64
        )
    )
    moved = coordinates.to("cpu")

    assert moved.dtype == torch.float64
    assert moved.fingerprint() == coordinates.fingerprint()


def test_temporal_coordinate_helpers_reject_unknown_objects() -> None:
    with pytest.raises(TypeError):
        validate_temporal_coordinates(
            object(),  # type: ignore[arg-type]
            _right_mask(),
            padding_direction="right",
        )

    with pytest.raises(TypeError):
        temporal_coordinates_fingerprint(
            object()  # type: ignore[arg-type]
        )


# =============================================================================
# HistoricalSequenceInputs: valid contracts
# =============================================================================


def test_history_inputs_preserve_complete_alignment() -> None:
    inputs = _history_inputs()

    assert inputs.node_count == N
    assert inputs.item_count == N
    assert inputs.sequence_length == T
    assert inputs.feature_dim == D
    assert inputs.history_shape == (
        N,
        T,
        D,
    )
    assert inputs.device == torch.device("cpu")
    assert inputs.dtype == torch.float32
    assert inputs.node_ids == (
        "node-0",
        "node-1",
        "node-2",
    )
    assert torch.equal(
        inputs.node_batch_index,
        torch.tensor(
            [0, 0, 1],
            dtype=torch.long,
        ),
    )
    assert inputs.graph_count == 2
    assert inputs.graph_ids == (
        "graph-0",
        "graph-1",
    )
    assert inputs.feature_names == (
        "historical_burden",
        "reporting_intensity",
    )
    assert inputs.source_fingerprint == "history-source"
    assert not inputs.has_feature_observation_mask
    assert not inputs.has_zero_history


def test_history_inputs_valid_lengths_and_boundary_indices() -> None:
    inputs = _history_inputs()

    assert torch.equal(
        inputs.valid_lengths,
        torch.tensor(
            [3, 2, 4],
            dtype=torch.long,
        ),
    )
    assert torch.equal(
        inputs.first_valid_indices,
        torch.tensor(
            [0, 0, 0],
            dtype=torch.long,
        ),
    )
    assert torch.equal(
        inputs.last_valid_indices,
        torch.tensor(
            [2, 1, 3],
            dtype=torch.long,
        ),
    )


def test_left_padded_history_indices() -> None:
    mask = _left_mask()
    inputs = _history_inputs(
        history=_history_for_mask(mask),
        timestep_mask=mask,
        temporal_coordinates=_relative_coordinates(
            values=_relative_values_for_left_mask(),
        ),
        padding_direction=TemporalPaddingDirection.LEFT,
    )

    assert torch.equal(
        inputs.valid_lengths,
        torch.tensor(
            [3, 2, 4],
            dtype=torch.long,
        ),
    )
    assert torch.equal(
        inputs.first_valid_indices,
        torch.tensor(
            [1, 2, 0],
            dtype=torch.long,
        ),
    )
    assert torch.equal(
        inputs.last_valid_indices,
        torch.tensor(
            [3, 3, 3],
            dtype=torch.long,
        ),
    )


def test_history_inputs_accept_absolute_coordinates() -> None:
    inputs = _history_inputs(
        temporal_coordinates=_absolute_coordinates(),
    )

    assert isinstance(
        inputs.temporal_coordinates,
        AbsoluteTemporalCoordinates,
    )


def test_history_inputs_accept_all_features_missing_at_real_timestep() -> None:
    feature_mask = _feature_mask_for_timestep_mask(
        _right_mask()
    )
    feature_mask[0, 1, :] = False

    inputs = _history_inputs(
        feature_observed_mask=feature_mask,
        missing_value_policy=(
            HistoryMissingValuePolicy.FINITE_PLACEHOLDER_WITH_MASK
        ),
    )

    assert inputs.timestep_mask[0, 1]
    assert not bool(
        inputs.feature_observed_mask[
            0,
            1,
        ].any().item()
    )


def test_complete_missingness_policy_accepts_absent_feature_mask() -> None:
    inputs = _history_inputs(
        missing_value_policy=HistoryMissingValuePolicy.COMPLETE,
    )

    assert inputs.missing_value_policy == (
        HistoryMissingValuePolicy.COMPLETE
    )
    assert inputs.feature_observed_mask is None


def test_complete_missingness_policy_accepts_all_observed_mask() -> None:
    feature_mask = _feature_mask_for_timestep_mask(
        _right_mask()
    )
    inputs = _history_inputs(
        feature_observed_mask=feature_mask,
        missing_value_policy=HistoryMissingValuePolicy.COMPLETE,
    )

    assert inputs.has_feature_observation_mask


def test_upstream_imputed_policy_can_preserve_observation_mask() -> None:
    feature_mask = _feature_mask_for_timestep_mask(
        _right_mask()
    )
    feature_mask[0, 1, 0] = False

    inputs = _history_inputs(
        feature_observed_mask=feature_mask,
        missing_value_policy=HistoryMissingValuePolicy.UPSTREAM_IMPUTED,
    )

    assert not inputs.feature_observed_mask[0, 1, 0]


def test_history_inputs_normalize_string_policies() -> None:
    inputs = _history_inputs(
        padding_direction="right",
        missing_value_policy="upstream_imputed",
        zero_length_policy="error",
    )

    assert inputs.padding_direction == TemporalPaddingDirection.RIGHT
    assert inputs.missing_value_policy == (
        HistoryMissingValuePolicy.UPSTREAM_IMPUTED
    )
    assert inputs.zero_length_policy == HistoryZeroLengthPolicy.ERROR


def test_history_inputs_semantic_dictionary_is_tensor_free() -> None:
    inputs = _history_inputs()
    semantic = inputs.semantic_dict()

    assert semantic["history_shape"] == [
        N,
        T,
        D,
    ]
    assert semantic["history_value_semantics"] == HISTORY_VALUE_SEMANTICS
    assert semantic["timestep_mask_semantics"] == TIMESTEP_MASK_SEMANTICS
    assert semantic["padding_direction"] == "right"
    assert semantic["missing_value_policy"] == "upstream_imputed"
    assert semantic["zero_length_policy"] == "error"
    assert semantic["canonical_padding_value"] == 0
    assert semantic["scientific_interpretation"] == (
        HISTORY_INPUT_SCIENTIFIC_INTERPRETATION
    )
    assert not any(
        isinstance(value, torch.Tensor)
        for value in semantic.values()
    )


def test_history_input_fingerprints_are_deterministic() -> None:
    first = _history_inputs()
    second = _history_inputs()

    assert first.timestep_mask_fingerprint() == (
        second.timestep_mask_fingerprint()
    )
    assert first.temporal_alignment_fingerprint() == (
        second.temporal_alignment_fingerprint()
    )
    assert first.alignment_fingerprint() == (
        second.alignment_fingerprint()
    )
    assert first.semantic_fingerprint() == second.semantic_fingerprint()
    assert first.value_fingerprint() == second.value_fingerprint()
    assert first.lineage_fingerprint() == second.lineage_fingerprint()
    assert first.fingerprint() == first.lineage_fingerprint()


def test_history_value_change_affects_value_and_lineage_not_alignment() -> None:
    first = _history_inputs()
    changed_history = first.history.clone()
    changed_history[0, 0, 0] += 1.0
    second = _history_inputs(
        history=changed_history,
    )

    assert first.alignment_fingerprint() == second.alignment_fingerprint()
    assert first.value_fingerprint() != second.value_fingerprint()
    assert first.lineage_fingerprint() != second.lineage_fingerprint()


def test_feature_observation_change_affects_value_and_lineage() -> None:
    first_mask = _feature_mask_for_timestep_mask(
        _right_mask()
    )
    second_mask = first_mask.clone()
    second_mask[0, 1, 0] = False

    first = _history_inputs(
        feature_observed_mask=first_mask,
    )
    second = _history_inputs(
        feature_observed_mask=second_mask,
    )

    assert first.feature_observation_fingerprint() != (
        second.feature_observation_fingerprint()
    )
    assert first.value_fingerprint() != second.value_fingerprint()
    assert first.lineage_fingerprint() != second.lineage_fingerprint()


def test_history_inputs_are_frozen() -> None:
    inputs = _history_inputs()

    with pytest.raises(
        (FrozenInstanceError, AttributeError),
    ):
        inputs.history_name = "changed"  # type: ignore[misc]


def test_history_inputs_replace_reconstructs_and_revalidates() -> None:
    inputs = _history_inputs()
    replacement = inputs.replace(
        history_name="alternate_history",
    )

    assert replacement is not inputs
    assert replacement.history_name == "alternate_history"
    assert replacement.history is inputs.history

    with pytest.raises(ValueError):
        inputs.replace(
            history_name="",
        )


def test_history_inputs_to_can_change_history_dtype() -> None:
    inputs = _history_inputs()
    moved = inputs.to(
        "cpu",
        dtype=torch.float64,
    )

    assert moved is not inputs
    assert moved.history.dtype == torch.float64
    assert moved.timestep_mask.dtype == torch.bool
    assert moved.node_batch_index.dtype == torch.long
    assert moved.temporal_coordinates.device == torch.device("cpu")
    assert moved.semantic_fingerprint() == inputs.semantic_fingerprint()
    assert moved.value_fingerprint() != inputs.value_fingerprint()


def test_history_inputs_to_rejects_nonfloating_dtype() -> None:
    inputs = _history_inputs()

    with pytest.raises(ValueError):
        inputs.to(
            "cpu",
            dtype=torch.long,
        )


# =============================================================================
# HistoricalSequenceInputs: invalid contracts
# =============================================================================


@pytest.mark.parametrize(
    "history",
    [
        torch.zeros(
            (N, T),
            dtype=torch.float32,
        ),
        torch.zeros(
            (N, T, D),
            dtype=torch.long,
        ),
        torch.zeros(
            (0, T, D),
            dtype=torch.float32,
        ),
        torch.zeros(
            (N, 0, D),
            dtype=torch.float32,
        ),
        torch.zeros(
            (N, T, 0),
            dtype=torch.float32,
        ),
    ],
)
def test_history_inputs_reject_invalid_history_tensor(
    history: torch.Tensor,
) -> None:
    with pytest.raises(
        (TypeError, ValueError),
    ):
        _history_inputs(
            history=history,
        )


@pytest.mark.parametrize(
    "nonfinite",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_history_inputs_reject_nonfinite_values(
    nonfinite: float,
) -> None:
    history = _history_for_mask(
        _right_mask()
    )
    history[0, 0, 0] = nonfinite

    with pytest.raises(ValueError):
        _history_inputs(
            history=history,
        )


@pytest.mark.parametrize(
    "mask",
    [
        torch.ones(
            (N, T),
            dtype=torch.long,
        ),
        torch.ones(
            (N, T, 1),
            dtype=torch.bool,
        ),
        torch.ones(
            (N, T - 1),
            dtype=torch.bool,
        ),
    ],
)
def test_history_inputs_reject_invalid_timestep_mask(
    mask: torch.Tensor,
) -> None:
    with pytest.raises(
        (TypeError, ValueError),
    ):
        _history_inputs(
            timestep_mask=mask,
            history=torch.zeros(
                (N, T, D),
                dtype=torch.float32,
            ),
        )


@pytest.mark.parametrize(
    "feature_mask",
    [
        torch.ones(
            (N, T, D),
            dtype=torch.long,
        ),
        torch.ones(
            (N, T),
            dtype=torch.bool,
        ),
        torch.ones(
            (N, T, D + 1),
            dtype=torch.bool,
        ),
    ],
)
def test_history_inputs_reject_invalid_feature_mask(
    feature_mask: torch.Tensor,
) -> None:
    with pytest.raises(
        (TypeError, ValueError),
    ):
        _history_inputs(
            feature_observed_mask=feature_mask,
        )


def test_history_inputs_reject_observed_features_at_padding() -> None:
    feature_mask = _feature_mask_for_timestep_mask(
        _right_mask()
    )
    feature_mask[0, 3, 0] = True

    with pytest.raises(ValueError):
        _history_inputs(
            feature_observed_mask=feature_mask,
        )


def test_history_inputs_reject_nonzero_history_padding() -> None:
    history = _history_for_mask(
        _right_mask()
    )
    history[0, 3, 0] = 123.0

    with pytest.raises(ValueError):
        _history_inputs(
            history=history,
        )


def test_complete_policy_rejects_missing_valid_feature() -> None:
    feature_mask = _feature_mask_for_timestep_mask(
        _right_mask()
    )
    feature_mask[0, 1, 0] = False

    with pytest.raises(ValueError):
        _history_inputs(
            feature_observed_mask=feature_mask,
            missing_value_policy=HistoryMissingValuePolicy.COMPLETE,
        )


def test_placeholder_policy_requires_feature_mask() -> None:
    with pytest.raises(ValueError):
        _history_inputs(
            missing_value_policy=(
                HistoryMissingValuePolicy.FINITE_PLACEHOLDER_WITH_MASK
            ),
        )


def test_history_inputs_reject_node_axis_count_mismatch() -> None:
    node_axis = TemporalNodeAxis(
        node_ids=(
            "node-0",
            "node-1",
        ),
        node_batch_index=torch.tensor(
            [0, 0],
            dtype=torch.long,
        ),
        graph_count=1,
        graph_ids=("graph-0",),
    )

    with pytest.raises(ValueError):
        _history_inputs(
            node_axis=node_axis,
        )


def test_history_inputs_reject_feature_axis_count_mismatch() -> None:
    with pytest.raises(ValueError):
        _history_inputs(
            feature_axis=_feature_axis(
                feature_names=(
                    "only-one",
                )
            ),
        )


def test_history_inputs_reject_temporal_shape_mismatch() -> None:
    coordinates = RelativeTemporalCoordinates(
        values=torch.tensor(
            [
                [-2.0, -1.0, 0.0],
                [-2.0, -1.0, 0.0],
                [-2.0, -1.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        unit="months",
    )

    with pytest.raises(ValueError):
        _history_inputs(
            temporal_coordinates=coordinates,
        )


def test_history_inputs_reject_internal_padding() -> None:
    mask = torch.tensor(
        [
            [True, False, True, False],
            [True, True, False, False],
            [True, True, True, True],
        ],
        dtype=torch.bool,
    )
    history = _history_for_mask(
        mask
    )

    with pytest.raises(ValueError):
        _history_inputs(
            history=history,
            timestep_mask=mask,
        )


def test_history_inputs_reject_padding_direction_mismatch() -> None:
    with pytest.raises(ValueError):
        _history_inputs(
            padding_direction=TemporalPaddingDirection.LEFT,
        )


def test_history_inputs_reject_padding_under_none_policy() -> None:
    with pytest.raises(ValueError):
        _history_inputs(
            padding_direction=TemporalPaddingDirection.NONE,
        )


def test_zero_history_is_rejected_by_default() -> None:
    mask = _right_mask()
    mask[1, :] = False
    history = _history_for_mask(
        mask
    )
    coordinates = _relative_coordinates(
        values=torch.tensor(
            [
                [-3.0, -2.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [-12.0, -6.0, -2.0, -0.5],
            ],
            dtype=torch.float32,
        )
    )

    with pytest.raises(ValueError):
        _history_inputs(
            history=history,
            timestep_mask=mask,
            temporal_coordinates=coordinates,
        )


def test_zero_history_can_be_preserved_explicitly() -> None:
    mask = _right_mask()
    mask[1, :] = False
    history = _history_for_mask(
        mask
    )
    coordinates = _relative_coordinates(
        values=torch.tensor(
            [
                [-3.0, -2.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [-12.0, -6.0, -2.0, -0.5],
            ],
            dtype=torch.float32,
        )
    )

    inputs = _history_inputs(
        history=history,
        timestep_mask=mask,
        temporal_coordinates=coordinates,
        zero_length_policy=(
            HistoryZeroLengthPolicy.ALLOW_ZERO_HISTORY
        ),
    )

    assert inputs.has_zero_history
    assert torch.equal(
        inputs.valid_lengths,
        torch.tensor(
            [3, 0, 4],
            dtype=torch.long,
        ),
    )
    assert torch.equal(
        inputs.first_valid_indices,
        torch.tensor(
            [0, -1, 0],
            dtype=torch.long,
        ),
    )
    assert torch.equal(
        inputs.last_valid_indices,
        torch.tensor(
            [2, -1, 3],
            dtype=torch.long,
        ),
    )


def test_zero_history_is_incompatible_with_no_padding() -> None:
    mask = _full_mask()
    mask[1, :] = False
    history = _history_for_mask(
        mask
    )
    coordinates = RelativeTemporalCoordinates(
        values=torch.tensor(
            [
                [-4.0, -3.0, -2.0, -1.0],
                [0.0, 0.0, 0.0, 0.0],
                [-4.0, -3.0, -2.0, -1.0],
            ],
            dtype=torch.float32,
        ),
        unit="months",
    )

    with pytest.raises(ValueError):
        _history_inputs(
            history=history,
            timestep_mask=mask,
            temporal_coordinates=coordinates,
            padding_direction=TemporalPaddingDirection.NONE,
            zero_length_policy=(
                HistoryZeroLengthPolicy.ALLOW_ZERO_HISTORY
            ),
        )


@pytest.mark.parametrize(
    (
        "field_name",
        "invalid_value",
    ),
    [
        ("node_axis", object()),
        ("feature_axis", object()),
        ("temporal_coordinates", object()),
        ("source_provenance", object()),
    ],
)
def test_history_inputs_reject_wrong_contract_types(
    field_name: str,
    invalid_value: object,
) -> None:
    kwargs = {
        "history": _history_for_mask(
            _right_mask()
        ),
        "timestep_mask": _right_mask(),
        "node_axis": _node_axis(),
        "feature_axis": _feature_axis(),
        "temporal_coordinates": _relative_coordinates(),
        "source_provenance": _source_provenance(),
    }
    kwargs[field_name] = invalid_value

    with pytest.raises(TypeError):
        HistoricalSequenceInputs(
            **kwargs,
        )


# =============================================================================
# Device behavior
# =============================================================================


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is not available.",
)
def test_complete_history_contract_moves_to_cuda() -> None:
    inputs = _history_inputs(
        feature_observed_mask=(
            _feature_mask_for_timestep_mask(
                _right_mask()
            )
        )
    )
    moved = inputs.to(
        "cuda",
        dtype=torch.float64,
    )

    assert moved.device.type == "cuda"
    assert moved.history.dtype == torch.float64
    assert moved.timestep_mask.device.type == "cuda"
    assert moved.feature_observed_mask is not None
    assert moved.feature_observed_mask.device.type == "cuda"
    assert moved.node_axis.device.type == "cuda"
    assert moved.temporal_coordinates.device.type == "cuda"

    returned = moved.to(
        "cpu",
        dtype=torch.float32,
    )
    assert returned.device.type == "cpu"
    assert returned.history.dtype == torch.float32
    assert returned.semantic_fingerprint() == inputs.semantic_fingerprint()
