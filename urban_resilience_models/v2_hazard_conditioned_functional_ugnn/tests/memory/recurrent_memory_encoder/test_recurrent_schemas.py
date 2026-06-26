"""
Tests for immutable Phase 6 recurrent runtime schemas.

Target repository path:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            tests/
                memory/
                    recurrent_memory_encoder/
                        test_recurrent_schemas.py

Module under test:
    memory/recurrent_memory_encoder/schemas.py

The suite focuses on schema-level invariants rather than recurrent numerical
execution. It validates:

- packed/reference permutation semantics;
- stable equal-length sorting;
- exact nonempty/zero-history node partitions;
- source-device, detached metadata ownership;
- canonical versus flat recurrent state layouts;
- GRU/LSTM hidden and cell-state distinctions;
- cross-object alignment with ``TemporalSequenceEncoding``;
- exact-zero recurrent outputs and states for zero-history nodes;
- private canonical right-padded batch invariants;
- deterministic fingerprints and frozen dataclass behavior.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import inspect

import pytest
import torch

import urban_resilience_models.v2_hazard_conditioned_functional_ugnn.memory.recurrent_memory_encoder.schemas as recurrent_schemas_module
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.memory.config import (
    RecurrentCellKind,
    RecurrentSequenceEncoderConfig,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.memory.recurrent_memory_encoder._provenance import (
    build_recurrent_sequence_computation_provenance,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.memory.recurrent_memory_encoder.schemas import (
    CANONICAL_RECURRENT_EXECUTION_PATHS,
    RECURRENT_CANONICAL_PADDING_DIRECTION,
    RECURRENT_DIRECTION_FEATURE_ORDER,
    RECURRENT_EXECUTION_METADATA_SCHEMA_VERSION,
    RECURRENT_EXECUTION_METADATA_SCIENTIFIC_INTERPRETATION,
    RECURRENT_FLAT_STATE_AXIS_ORDER,
    RECURRENT_SCHEMAS_VERSION,
    RECURRENT_SEQUENCE_ENCODER_RUN_SCHEMA_VERSION,
    RECURRENT_SEQUENCE_ENCODER_RUN_SCIENTIFIC_INTERPRETATION,
    RECURRENT_STATE_AXIS_ORDER,
    RECURRENT_STATE_LAYOUT_SCHEMA_VERSION,
    RECURRENT_STATE_LAYOUT_SCIENTIFIC_INTERPRETATION,
    RECURRENT_ZERO_HISTORY_STATE_POLICY,
    RecurrentEncoderRun,
    RecurrentExecutionMetadata,
    RecurrentExecutionPath,
    RecurrentRun,
    RecurrentSequenceEncoderRun,
    RecurrentStateAxisLayout,
    RecurrentStateLayout,
    _CanonicalRecurrentBatch,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.memory.schemas.history_inputs import (
    HistoricalSequenceInputs,
    HistoryZeroLengthPolicy,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.memory.schemas.provenance import (
    MemorySourceProvenance,
    TemporalFeatureAxis,
    TemporalNodeAxis,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.memory.schemas.sequence_encoding import (
    TemporalSequenceEncoderKind,
    TemporalSequenceEncoding,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.memory.schemas.temporal_coordinates import (
    RelativeTemporalCoordinates,
    TemporalPaddingDirection,
)


N = 6
T = 4
D = 2
H = 3
LAYERS = 2
DIRECTIONS = 2


# =============================================================================
# Shared factories
# =============================================================================


def _lengths() -> torch.Tensor:
    return torch.tensor(
        [2, 0, 4, 1, 3, 2],
        dtype=torch.long,
    )


def _mask_from_lengths(
    lengths: torch.Tensor,
    *,
    sequence_length: int = T,
    padding_direction: TemporalPaddingDirection = (
        TemporalPaddingDirection.RIGHT
    ),
) -> torch.Tensor:
    node_count = int(
        lengths.numel()
    )
    mask = torch.zeros(
        node_count,
        sequence_length,
        dtype=torch.bool,
        device=lengths.device,
    )

    for node_index, length_value in enumerate(
        lengths.tolist()
    ):
        length = int(
            length_value
        )

        if length == 0:
            continue

        if padding_direction == TemporalPaddingDirection.LEFT:
            mask[
                node_index,
                sequence_length - length :,
            ] = True
        else:
            mask[
                node_index,
                :length,
            ] = True

    return mask


def _history(
    *,
    padding_direction: TemporalPaddingDirection = (
        TemporalPaddingDirection.RIGHT
    ),
    lengths: torch.Tensor | None = None,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> HistoricalSequenceInputs:
    if lengths is None:
        lengths = _lengths()

    lengths = lengths.to(
        device=device
    )
    node_count = int(
        lengths.numel()
    )
    mask = _mask_from_lengths(
        lengths,
        padding_direction=padding_direction,
    )

    history = torch.zeros(
        node_count,
        T,
        D,
        dtype=dtype,
        device=device,
    )

    coordinates = torch.zeros(
        node_count,
        T,
        dtype=dtype,
        device=device,
    )

    next_value = 1.0

    for node_index in range(
        node_count
    ):
        valid_indices = torch.nonzero(
            mask[
                node_index
            ],
            as_tuple=False,
        ).flatten()
        length = int(
            valid_indices.numel()
        )

        for logical_index, temporal_index in enumerate(
            valid_indices.tolist()
        ):
            history[
                node_index,
                temporal_index,
            ] = torch.tensor(
                [
                    next_value,
                    next_value + 0.5,
                ],
                dtype=dtype,
                device=device,
            )
            next_value += 1.0
            coordinates[
                node_index,
                temporal_index,
            ] = float(
                logical_index - length
            )

    node_ids = tuple(
        f"node-{index}"
        for index in range(
            node_count
        )
    )

    return HistoricalSequenceInputs(
        history=history,
        timestep_mask=mask,
        node_axis=TemporalNodeAxis(
            node_ids=node_ids,
            node_batch_index=torch.zeros(
                node_count,
                dtype=torch.long,
                device=device,
            ),
            graph_count=1,
            graph_ids=("graph-0",),
            source_fingerprint="recurrent-schema-node-axis-v1",
        ),
        feature_axis=TemporalFeatureAxis(
            feature_names=(
                "feature-0",
                "feature-1",
            ),
            source_fingerprint="recurrent-schema-feature-axis-v1",
        ),
        temporal_coordinates=RelativeTemporalCoordinates(
            values=coordinates,
            unit="months",
        ),
        source_provenance=MemorySourceProvenance(
            source_name="recurrent-schema-panel",
            source_kind="historical-node-sequence",
            source_fingerprint="recurrent-schema-source-v1",
            preprocessing_fingerprint=(
                "recurrent-schema-preprocessing-v1"
            ),
        ),
        padding_direction=padding_direction,
        zero_length_policy=(
            HistoryZeroLengthPolicy.ALLOW_ZERO_HISTORY
            if bool(
                torch.any(
                    lengths == 0
                ).item()
            )
            else HistoryZeroLengthPolicy.ERROR
        ),
    )


def _packed_metadata(
    *,
    history: HistoricalSequenceInputs | None = None,
) -> RecurrentExecutionMetadata:
    if history is None:
        history = _history()

    return RecurrentExecutionMetadata(
        history_lengths=history.valid_lengths,
        nonempty_node_indices=torch.tensor(
            [0, 2, 3, 4, 5],
            dtype=torch.long,
            device=history.device,
        ),
        zero_history_node_indices=torch.tensor(
            [1],
            dtype=torch.long,
            device=history.device,
        ),
        sorted_to_nonempty=torch.tensor(
            [1, 3, 0, 4, 2],
            dtype=torch.long,
            device=history.device,
        ),
        nonempty_to_sorted=torch.tensor(
            [2, 0, 4, 1, 3],
            dtype=torch.long,
            device=history.device,
        ),
        execution_path=RecurrentExecutionPath.PACKED,
        sort_was_applied=True,
        original_padding_direction=(
            history.padding_direction
        ),
    )


def _reference_metadata(
    *,
    history: HistoricalSequenceInputs | None = None,
) -> RecurrentExecutionMetadata:
    if history is None:
        history = _history()

    return RecurrentExecutionMetadata(
        history_lengths=history.valid_lengths,
        nonempty_node_indices=torch.tensor(
            [0, 2, 3, 4, 5],
            dtype=torch.long,
            device=history.device,
        ),
        zero_history_node_indices=torch.tensor(
            [1],
            dtype=torch.long,
            device=history.device,
        ),
        sorted_to_nonempty=torch.arange(
            5,
            dtype=torch.long,
            device=history.device,
        ),
        nonempty_to_sorted=torch.arange(
            5,
            dtype=torch.long,
            device=history.device,
        ),
        execution_path=RecurrentExecutionPath.REFERENCE,
        sort_was_applied=False,
        original_padding_direction=(
            history.padding_direction
        ),
    )


def _all_zero_history(
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> HistoricalSequenceInputs:
    return _history(
        lengths=torch.zeros(
            3,
            dtype=torch.long,
        ),
        dtype=dtype,
        device=device,
    )


def _all_zero_metadata(
    history: HistoricalSequenceInputs,
    *,
    execution_path: (
        RecurrentExecutionPath
        | str
    ) = RecurrentExecutionPath.PACKED,
) -> RecurrentExecutionMetadata:
    return RecurrentExecutionMetadata(
        history_lengths=history.valid_lengths,
        nonempty_node_indices=torch.empty(
            0,
            dtype=torch.long,
            device=history.device,
        ),
        zero_history_node_indices=torch.arange(
            history.node_count,
            dtype=torch.long,
            device=history.device,
        ),
        sorted_to_nonempty=torch.empty(
            0,
            dtype=torch.long,
            device=history.device,
        ),
        nonempty_to_sorted=torch.empty(
            0,
            dtype=torch.long,
            device=history.device,
        ),
        execution_path=execution_path,
        sort_was_applied=False,
        original_padding_direction=(
            history.padding_direction
        ),
    )


def _config(
    *,
    cell_kind: RecurrentCellKind = RecurrentCellKind.GRU,
    pack_sequences: bool = True,
    bidirectional: bool = True,
    num_layers: int = LAYERS,
) -> RecurrentSequenceEncoderConfig:
    return RecurrentSequenceEncoderConfig(
        cell_kind=cell_kind,
        input_dim=D,
        hidden_dim=H,
        num_layers=num_layers,
        dropout=0.0,
        bidirectional=bidirectional,
        use_bias=True,
        pack_sequences=pack_sequences,
        enforce_sorted_lengths=False,
    )


def _encoded_values(
    history: HistoricalSequenceInputs,
    *,
    output_dim: int,
    requires_grad: bool = False,
) -> torch.Tensor:
    values = torch.zeros(
        history.node_count,
        history.sequence_length,
        output_dim,
        dtype=history.dtype,
        device=history.device,
    )

    for node_index in range(
        history.node_count
    ):
        valid_indices = torch.nonzero(
            history.timestep_mask[
                node_index
            ],
            as_tuple=False,
        ).flatten()

        if valid_indices.numel() > 0:
            values[
                node_index,
                valid_indices,
            ] = float(
                node_index + 1
            )

    if requires_grad:
        values.requires_grad_()

    return values


def _public_output(
    *,
    history: HistoricalSequenceInputs | None = None,
    encoder_kind: (
        TemporalSequenceEncoderKind
        | str
    ) = TemporalSequenceEncoderKind.GRU,
    pack_sequences: bool = True,
    bidirectional: bool = True,
    num_layers: int = LAYERS,
    encoded_values: torch.Tensor | None = None,
) -> TemporalSequenceEncoding:
    if history is None:
        history = _history()

    cell_kind = (
        RecurrentCellKind.GRU
        if TemporalSequenceEncoderKind(
            encoder_kind
        )
        == TemporalSequenceEncoderKind.GRU
        else RecurrentCellKind.LSTM
    )
    config = _config(
        cell_kind=cell_kind,
        pack_sequences=pack_sequences,
        bidirectional=bidirectional,
        num_layers=num_layers,
    )
    metadata = (
        _packed_metadata(
            history=history
        )
        if pack_sequences
        else _reference_metadata(
            history=history
        )
    )
    provenance = (
        build_recurrent_sequence_computation_provenance(
            source_history=history,
            config=config,
            execution_path=metadata.execution_path,
            sort_was_applied=metadata.sort_was_applied,
            module_training=False,
            nonempty_node_count=(
                metadata.nonempty_node_count
            ),
            zero_history_count=(
                metadata.zero_history_count
            ),
            adapter_executed=True,
            recurrent_kernel_executed=True,
            all_zero_history_short_circuit=False,
        )
    )

    output_dim = (
        H
        * (
            2
            if bidirectional
            else 1
        )
    )

    if encoded_values is None:
        encoded_values = _encoded_values(
            history,
            output_dim=output_dim,
        )

    return TemporalSequenceEncoding(
        encoded_sequence=encoded_values,
        source_history=history,
        encoder_kind=encoder_kind,
        computation_provenance=provenance,
    )


def _all_zero_public_output(
    *,
    history: HistoricalSequenceInputs,
    encoder_kind: (
        TemporalSequenceEncoderKind
        | str
    ) = TemporalSequenceEncoderKind.GRU,
    pack_sequences: bool = True,
) -> TemporalSequenceEncoding:
    cell_kind = (
        RecurrentCellKind.GRU
        if TemporalSequenceEncoderKind(
            encoder_kind
        )
        == TemporalSequenceEncoderKind.GRU
        else RecurrentCellKind.LSTM
    )
    config = _config(
        cell_kind=cell_kind,
        pack_sequences=pack_sequences,
        bidirectional=False,
        num_layers=1,
    )
    provenance = (
        build_recurrent_sequence_computation_provenance(
            source_history=history,
            config=config,
            execution_path=(
                RecurrentExecutionPath.PACKED
                if pack_sequences
                else RecurrentExecutionPath.REFERENCE
            ),
            sort_was_applied=False,
            module_training=False,
            nonempty_node_count=0,
            zero_history_count=history.node_count,
            adapter_executed=False,
            recurrent_kernel_executed=False,
            all_zero_history_short_circuit=True,
        )
    )

    return TemporalSequenceEncoding(
        encoded_sequence=torch.zeros(
            history.node_count,
            history.sequence_length,
            H,
            dtype=history.dtype,
            device=history.device,
        ),
        source_history=history,
        encoder_kind=encoder_kind,
        computation_provenance=provenance,
    )


def _state_layout(
    *,
    num_layers: int = LAYERS,
    num_directions: int = DIRECTIONS,
    hidden_dim: int = H,
) -> RecurrentStateLayout:
    return RecurrentStateLayout(
        num_layers=num_layers,
        num_directions=num_directions,
        hidden_dim=hidden_dim,
    )


def _state(
    *,
    layout: RecurrentStateLayout,
    history: HistoricalSequenceInputs,
    requires_grad: bool = False,
) -> torch.Tensor:
    state = torch.zeros(
        layout.num_layers,
        layout.num_directions,
        history.node_count,
        layout.hidden_dim,
        dtype=history.dtype,
        device=history.device,
    )

    for node_index in range(
        history.node_count
    ):
        if history.valid_lengths[
            node_index
        ] > 0:
            state[
                :,
                :,
                node_index,
                :,
            ] = float(
                node_index + 1
            )

    if requires_grad:
        state.requires_grad_()

    return state


def _gru_run(
    *,
    history: HistoricalSequenceInputs | None = None,
    packed: bool = True,
    requires_grad: bool = False,
) -> RecurrentSequenceEncoderRun:
    if history is None:
        history = _history()

    layout = _state_layout()
    output = _public_output(
        history=history,
        encoder_kind=TemporalSequenceEncoderKind.GRU,
        pack_sequences=packed,
        encoded_values=_encoded_values(
            history,
            output_dim=layout.output_dim,
            requires_grad=requires_grad,
        ),
    )
    metadata = (
        _packed_metadata(
            history=history
        )
        if packed
        else _reference_metadata(
            history=history
        )
    )

    return RecurrentSequenceEncoderRun(
        public_output=output,
        final_hidden_state=_state(
            layout=layout,
            history=history,
            requires_grad=requires_grad,
        ),
        final_cell_state=None,
        state_layout=layout,
        execution_metadata=metadata,
    )


def _lstm_run(
    *,
    history: HistoricalSequenceInputs | None = None,
    packed: bool = False,
    requires_grad: bool = False,
) -> RecurrentSequenceEncoderRun:
    if history is None:
        history = _history()

    layout = _state_layout()
    output = _public_output(
        history=history,
        encoder_kind=TemporalSequenceEncoderKind.LSTM,
        pack_sequences=packed,
        encoded_values=_encoded_values(
            history,
            output_dim=layout.output_dim,
            requires_grad=requires_grad,
        ),
    )
    metadata = (
        _packed_metadata(
            history=history
        )
        if packed
        else _reference_metadata(
            history=history
        )
    )
    hidden = _state(
        layout=layout,
        history=history,
        requires_grad=requires_grad,
    )
    cell = _state(
        layout=layout,
        history=history,
        requires_grad=requires_grad,
    )

    return RecurrentSequenceEncoderRun(
        public_output=output,
        final_hidden_state=hidden,
        final_cell_state=cell,
        state_layout=layout,
        execution_metadata=metadata,
    )


# =============================================================================
# Module identity, exports, and import boundaries
# =============================================================================


@pytest.mark.parametrize(
    "value",
    (
        RECURRENT_SCHEMAS_VERSION,
        RECURRENT_EXECUTION_METADATA_SCHEMA_VERSION,
        RECURRENT_STATE_LAYOUT_SCHEMA_VERSION,
        RECURRENT_SEQUENCE_ENCODER_RUN_SCHEMA_VERSION,
        RECURRENT_STATE_AXIS_ORDER,
        RECURRENT_FLAT_STATE_AXIS_ORDER,
        RECURRENT_DIRECTION_FEATURE_ORDER,
        RECURRENT_ZERO_HISTORY_STATE_POLICY,
        RECURRENT_EXECUTION_METADATA_SCIENTIFIC_INTERPRETATION,
        RECURRENT_STATE_LAYOUT_SCIENTIFIC_INTERPRETATION,
        RECURRENT_SEQUENCE_ENCODER_RUN_SCIENTIFIC_INTERPRETATION,
    ),
)
def test_schema_identity_strings_are_nonempty(
    value: str,
) -> None:
    assert isinstance(
        value,
        str,
    )
    assert value.strip()


def test_execution_path_vocabulary_is_exact() -> None:
    assert CANONICAL_RECURRENT_EXECUTION_PATHS == (
        "packed",
        "reference",
    )
    assert RecurrentExecutionPath.PACKED.value == "packed"
    assert RecurrentExecutionPath.REFERENCE.value == "reference"


def test_canonical_padding_direction_is_right() -> None:
    assert RECURRENT_CANONICAL_PADDING_DIRECTION == (
        TemporalPaddingDirection.RIGHT
    )


def test_schema_aliases_preserve_exact_classes() -> None:
    assert RecurrentEncoderRun is RecurrentSequenceEncoderRun
    assert RecurrentRun is RecurrentSequenceEncoderRun
    assert RecurrentStateAxisLayout is RecurrentStateLayout


def test_public_all_has_no_duplicates_and_resolves() -> None:
    exported = recurrent_schemas_module.__all__

    assert isinstance(
        exported,
        tuple,
    )
    assert len(
        exported
    ) == len(
        set(
            exported
        )
    )

    for name in exported:
        assert hasattr(
            recurrent_schemas_module,
            name,
        )


def test_private_canonical_batch_is_not_publicly_exported() -> None:
    assert "_CanonicalRecurrentBatch" not in (
        recurrent_schemas_module.__all__
    )


def test_schema_source_has_no_direct_trainable_encoder_dependencies() -> None:
    source = inspect.getsource(
        recurrent_schemas_module
    )

    forbidden_import_fragments = (
        "baseline_encoders",
        "gru_encoder",
        "lstm_encoder",
        "transformer",
        "functional_message_passing",
        "hazard_query_encoder",
    )

    for fragment in forbidden_import_fragments:
        assert fragment not in source


# =============================================================================
# RecurrentExecutionMetadata: valid constructions and properties
# =============================================================================


def test_packed_metadata_accepts_frozen_permutation_example() -> None:
    metadata = _packed_metadata()

    assert metadata.history_lengths.tolist() == [
        2,
        0,
        4,
        1,
        3,
        2,
    ]
    assert metadata.nonempty_node_indices.tolist() == [
        0,
        2,
        3,
        4,
        5,
    ]
    assert metadata.zero_history_node_indices.tolist() == [
        1
    ]
    assert metadata.sorted_to_nonempty.tolist() == [
        1,
        3,
        0,
        4,
        2,
    ]
    assert metadata.nonempty_to_sorted.tolist() == [
        2,
        0,
        4,
        1,
        3,
    ]
    assert metadata.sorted_history_lengths.tolist() == [
        4,
        3,
        2,
        2,
        1,
    ]
    assert metadata.sorted_node_indices.tolist() == [
        2,
        4,
        0,
        5,
        3,
    ]


def test_reference_metadata_accepts_unsorted_lengths() -> None:
    metadata = _reference_metadata()

    assert metadata.execution_path == (
        RecurrentExecutionPath.REFERENCE
    )
    assert metadata.identity_permutation
    assert not metadata.sort_was_applied
    assert metadata.sorted_history_lengths.tolist() == [
        2,
        4,
        1,
        3,
        2,
    ]


@pytest.mark.parametrize(
    (
        "execution_path",
        "expected",
    ),
    (
        (
            RecurrentExecutionPath.PACKED,
            RecurrentExecutionPath.PACKED,
        ),
        (
            "reference",
            RecurrentExecutionPath.REFERENCE,
        ),
    ),
)
def test_metadata_normalizes_execution_path(
    execution_path: RecurrentExecutionPath | str,
    expected: RecurrentExecutionPath,
) -> None:
    history = _history()
    metadata = RecurrentExecutionMetadata(
        history_lengths=history.valid_lengths,
        nonempty_node_indices=torch.tensor(
            [0, 2, 3, 4, 5],
            dtype=torch.long,
        ),
        zero_history_node_indices=torch.tensor(
            [1],
            dtype=torch.long,
        ),
        sorted_to_nonempty=(
            torch.tensor(
                [1, 3, 0, 4, 2],
                dtype=torch.long,
            )
            if expected == RecurrentExecutionPath.PACKED
            else torch.arange(
                5,
                dtype=torch.long,
            )
        ),
        nonempty_to_sorted=(
            torch.tensor(
                [2, 0, 4, 1, 3],
                dtype=torch.long,
            )
            if expected == RecurrentExecutionPath.PACKED
            else torch.arange(
                5,
                dtype=torch.long,
            )
        ),
        execution_path=execution_path,
        sort_was_applied=(
            expected == RecurrentExecutionPath.PACKED
        ),
        original_padding_direction="right",
    )

    assert metadata.execution_path == expected
    assert metadata.original_padding_direction == (
        TemporalPaddingDirection.RIGHT
    )


def test_metadata_structural_properties() -> None:
    metadata = _packed_metadata()

    assert metadata.source_node_count == N
    assert metadata.node_count == N
    assert metadata.nonempty_node_count == 5
    assert metadata.zero_history_count == 1
    assert metadata.has_zero_history
    assert not metadata.all_zero_history
    assert metadata.device.type == "cpu"
    assert metadata.canonical_padding_direction == (
        TemporalPaddingDirection.RIGHT
    )
    assert not metadata.identity_permutation


def test_all_zero_metadata_accepts_empty_permutations() -> None:
    history = _all_zero_history()
    metadata = _all_zero_metadata(
        history
    )

    assert metadata.source_node_count == 3
    assert metadata.nonempty_node_count == 0
    assert metadata.zero_history_count == 3
    assert metadata.has_zero_history
    assert metadata.all_zero_history
    assert metadata.nonempty_history_lengths.numel() == 0
    assert metadata.sorted_history_lengths.numel() == 0
    assert metadata.sorted_node_indices.numel() == 0
    assert metadata.identity_permutation


@pytest.mark.parametrize(
    "execution_path",
    (
        RecurrentExecutionPath.PACKED,
        RecurrentExecutionPath.REFERENCE,
    ),
)
def test_all_zero_metadata_supports_both_selected_paths(
    execution_path: RecurrentExecutionPath,
) -> None:
    metadata = _all_zero_metadata(
        _all_zero_history(),
        execution_path=execution_path,
    )

    assert metadata.execution_path == execution_path
    assert not metadata.sort_was_applied


def test_metadata_accepts_left_padding_as_original_layout() -> None:
    history = _history(
        padding_direction=TemporalPaddingDirection.LEFT
    )
    metadata = _packed_metadata(
        history=history
    )

    assert metadata.original_padding_direction == (
        TemporalPaddingDirection.LEFT
    )
    assert metadata.canonical_padding_direction == (
        TemporalPaddingDirection.RIGHT
    )


def test_metadata_owns_detached_tensor_clones() -> None:
    history_lengths = _lengths()
    nonempty = torch.tensor(
        [0, 2, 3, 4, 5],
        dtype=torch.long,
    )
    zero = torch.tensor(
        [1],
        dtype=torch.long,
    )
    sorted_to_nonempty = torch.tensor(
        [1, 3, 0, 4, 2],
        dtype=torch.long,
    )
    nonempty_to_sorted = torch.tensor(
        [2, 0, 4, 1, 3],
        dtype=torch.long,
    )

    metadata = RecurrentExecutionMetadata(
        history_lengths=history_lengths,
        nonempty_node_indices=nonempty,
        zero_history_node_indices=zero,
        sorted_to_nonempty=sorted_to_nonempty,
        nonempty_to_sorted=nonempty_to_sorted,
        execution_path="packed",
        sort_was_applied=True,
        original_padding_direction="right",
    )

    assert metadata.history_lengths is not history_lengths
    assert metadata.nonempty_node_indices is not nonempty
    assert metadata.zero_history_node_indices is not zero
    assert metadata.sorted_to_nonempty is not (
        sorted_to_nonempty
    )
    assert metadata.nonempty_to_sorted is not (
        nonempty_to_sorted
    )

    history_lengths[0] = 99
    nonempty[0] = 5
    zero[0] = 4
    sorted_to_nonempty[0] = 0
    nonempty_to_sorted[0] = 0

    assert metadata.history_lengths.tolist() == [
        2,
        0,
        4,
        1,
        3,
        2,
    ]
    assert metadata.nonempty_node_indices.tolist() == [
        0,
        2,
        3,
        4,
        5,
    ]
    assert not metadata.history_lengths.requires_grad


def test_metadata_fingerprint_is_deterministic() -> None:
    first = _packed_metadata()
    second = _packed_metadata()

    assert first.semantic_dict() == second.semantic_dict()
    assert first.value_fingerprint() == (
        second.value_fingerprint()
    )
    assert first.fingerprint() == second.fingerprint()


def test_metadata_fingerprint_changes_with_execution_path() -> None:
    packed = _packed_metadata()
    reference = _reference_metadata()

    assert packed.fingerprint() != reference.fingerprint()


def test_metadata_to_revalidates_and_preserves_values() -> None:
    metadata = _packed_metadata()
    moved = metadata.to(
        "cpu"
    )

    assert moved is not metadata
    assert moved.fingerprint() == metadata.fingerprint()
    assert moved.device.type == "cpu"


def test_metadata_replace_revalidates() -> None:
    metadata = _reference_metadata()
    replaced = metadata.replace(
        schema_version="0.1-test"
    )

    assert replaced.schema_version == "0.1-test"
    assert replaced.history_lengths.tolist() == (
        metadata.history_lengths.tolist()
    )


def test_metadata_is_frozen() -> None:
    metadata = _packed_metadata()

    with pytest.raises(
        (
            FrozenInstanceError,
            AttributeError,
        )
    ):
        metadata.execution_path = (  # type: ignore[misc]
            RecurrentExecutionPath.REFERENCE
        )


# =============================================================================
# RecurrentExecutionMetadata: invalid constructions
# =============================================================================


@pytest.mark.parametrize(
    "field_name",
    (
        "history_lengths",
        "nonempty_node_indices",
        "zero_history_node_indices",
        "sorted_to_nonempty",
        "nonempty_to_sorted",
    ),
)
def test_metadata_rejects_non_tensor_index_fields(
    field_name: str,
) -> None:
    kwargs = {
        "history_lengths": _lengths(),
        "nonempty_node_indices": torch.tensor(
            [0, 2, 3, 4, 5],
            dtype=torch.long,
        ),
        "zero_history_node_indices": torch.tensor(
            [1],
            dtype=torch.long,
        ),
        "sorted_to_nonempty": torch.tensor(
            [1, 3, 0, 4, 2],
            dtype=torch.long,
        ),
        "nonempty_to_sorted": torch.tensor(
            [2, 0, 4, 1, 3],
            dtype=torch.long,
        ),
        "execution_path": "packed",
        "sort_was_applied": True,
        "original_padding_direction": "right",
    }
    kwargs[
        field_name
    ] = object()

    with pytest.raises(TypeError):
        RecurrentExecutionMetadata(
            **kwargs
        )


@pytest.mark.parametrize(
    "field_name",
    (
        "history_lengths",
        "nonempty_node_indices",
        "zero_history_node_indices",
        "sorted_to_nonempty",
        "nonempty_to_sorted",
    ),
)
def test_metadata_rejects_non_vector_index_fields(
    field_name: str,
) -> None:
    kwargs = {
        "history_lengths": _lengths(),
        "nonempty_node_indices": torch.tensor(
            [0, 2, 3, 4, 5],
            dtype=torch.long,
        ),
        "zero_history_node_indices": torch.tensor(
            [1],
            dtype=torch.long,
        ),
        "sorted_to_nonempty": torch.tensor(
            [1, 3, 0, 4, 2],
            dtype=torch.long,
        ),
        "nonempty_to_sorted": torch.tensor(
            [2, 0, 4, 1, 3],
            dtype=torch.long,
        ),
        "execution_path": "packed",
        "sort_was_applied": True,
        "original_padding_direction": "right",
    }
    kwargs[
        field_name
    ] = torch.zeros(
        1,
        1,
        dtype=torch.long,
    )

    with pytest.raises(ValueError):
        RecurrentExecutionMetadata(
            **kwargs
        )


@pytest.mark.parametrize(
    "field_name",
    (
        "history_lengths",
        "nonempty_node_indices",
        "zero_history_node_indices",
        "sorted_to_nonempty",
        "nonempty_to_sorted",
    ),
)
def test_metadata_rejects_non_int64_index_fields(
    field_name: str,
) -> None:
    kwargs = {
        "history_lengths": _lengths(),
        "nonempty_node_indices": torch.tensor(
            [0, 2, 3, 4, 5],
            dtype=torch.long,
        ),
        "zero_history_node_indices": torch.tensor(
            [1],
            dtype=torch.long,
        ),
        "sorted_to_nonempty": torch.tensor(
            [1, 3, 0, 4, 2],
            dtype=torch.long,
        ),
        "nonempty_to_sorted": torch.tensor(
            [2, 0, 4, 1, 3],
            dtype=torch.long,
        ),
        "execution_path": "packed",
        "sort_was_applied": True,
        "original_padding_direction": "right",
    }
    kwargs[
        field_name
    ] = kwargs[
        field_name
    ].to(
        dtype=torch.int32
    )

    with pytest.raises(ValueError):
        RecurrentExecutionMetadata(
            **kwargs
        )


def test_metadata_rejects_empty_source_node_axis() -> None:
    with pytest.raises(ValueError):
        RecurrentExecutionMetadata(
            history_lengths=torch.empty(
                0,
                dtype=torch.long,
            ),
            nonempty_node_indices=torch.empty(
                0,
                dtype=torch.long,
            ),
            zero_history_node_indices=torch.empty(
                0,
                dtype=torch.long,
            ),
            sorted_to_nonempty=torch.empty(
                0,
                dtype=torch.long,
            ),
            nonempty_to_sorted=torch.empty(
                0,
                dtype=torch.long,
            ),
            execution_path="packed",
            sort_was_applied=False,
            original_padding_direction="right",
        )


def test_metadata_rejects_negative_history_length() -> None:
    lengths = _lengths()
    lengths[0] = -1

    with pytest.raises(ValueError):
        RecurrentExecutionMetadata(
            history_lengths=lengths,
            nonempty_node_indices=torch.tensor(
                [0, 2, 3, 4, 5],
                dtype=torch.long,
            ),
            zero_history_node_indices=torch.tensor(
                [1],
                dtype=torch.long,
            ),
            sorted_to_nonempty=torch.tensor(
                [1, 3, 0, 4, 2],
                dtype=torch.long,
            ),
            nonempty_to_sorted=torch.tensor(
                [2, 0, 4, 1, 3],
                dtype=torch.long,
            ),
            execution_path="packed",
            sort_was_applied=True,
            original_padding_direction="right",
        )


@pytest.mark.parametrize(
    (
        "nonempty",
        "zero",
    ),
    (
        (
            [0, 2, 3, 4],
            [1],
        ),
        (
            [0, 2, 3, 4, 5],
            [],
        ),
        (
            [0, 2, 3, 4, 5],
            [1, 5],
        ),
    ),
)
def test_metadata_rejects_incomplete_or_overlapping_partition(
    nonempty: list[int],
    zero: list[int],
) -> None:
    with pytest.raises(ValueError):
        RecurrentExecutionMetadata(
            history_lengths=_lengths(),
            nonempty_node_indices=torch.tensor(
                nonempty,
                dtype=torch.long,
            ),
            zero_history_node_indices=torch.tensor(
                zero,
                dtype=torch.long,
            ),
            sorted_to_nonempty=torch.arange(
                len(
                    nonempty
                ),
                dtype=torch.long,
            ),
            nonempty_to_sorted=torch.arange(
                len(
                    nonempty
                ),
                dtype=torch.long,
            ),
            execution_path="reference",
            sort_was_applied=False,
            original_padding_direction="right",
        )


@pytest.mark.parametrize(
    (
        "nonempty",
        "zero",
    ),
    (
        (
            [2, 0, 3, 4, 5],
            [1],
        ),
        (
            [0, 2, 3, 4, 5],
            [1, 0],
        ),
        (
            [0, 2, 2, 4, 5],
            [1],
        ),
    ),
)
def test_metadata_rejects_non_strict_partition_order(
    nonempty: list[int],
    zero: list[int],
) -> None:
    with pytest.raises(ValueError):
        RecurrentExecutionMetadata(
            history_lengths=_lengths(),
            nonempty_node_indices=torch.tensor(
                nonempty,
                dtype=torch.long,
            ),
            zero_history_node_indices=torch.tensor(
                zero,
                dtype=torch.long,
            ),
            sorted_to_nonempty=torch.arange(
                len(
                    nonempty
                ),
                dtype=torch.long,
            ),
            nonempty_to_sorted=torch.arange(
                len(
                    nonempty
                ),
                dtype=torch.long,
            ),
            execution_path="reference",
            sort_was_applied=False,
            original_padding_direction="right",
        )


def test_metadata_rejects_zero_length_inside_nonempty_partition() -> None:
    with pytest.raises(ValueError):
        RecurrentExecutionMetadata(
            history_lengths=_lengths(),
            nonempty_node_indices=torch.tensor(
                [0, 1, 2, 3, 4],
                dtype=torch.long,
            ),
            zero_history_node_indices=torch.tensor(
                [5],
                dtype=torch.long,
            ),
            sorted_to_nonempty=torch.arange(
                5,
                dtype=torch.long,
            ),
            nonempty_to_sorted=torch.arange(
                5,
                dtype=torch.long,
            ),
            execution_path="reference",
            sort_was_applied=False,
            original_padding_direction="right",
        )


def test_metadata_rejects_positive_length_inside_zero_partition() -> None:
    with pytest.raises(ValueError):
        RecurrentExecutionMetadata(
            history_lengths=_lengths(),
            nonempty_node_indices=torch.tensor(
                [0, 1, 2, 3, 5],
                dtype=torch.long,
            ),
            zero_history_node_indices=torch.tensor(
                [4],
                dtype=torch.long,
            ),
            sorted_to_nonempty=torch.arange(
                5,
                dtype=torch.long,
            ),
            nonempty_to_sorted=torch.arange(
                5,
                dtype=torch.long,
            ),
            execution_path="reference",
            sort_was_applied=False,
            original_padding_direction="right",
        )


@pytest.mark.parametrize(
    (
        "sorted_to_nonempty",
        "nonempty_to_sorted",
    ),
    (
        (
            [1, 3, 0, 4],
            [2, 0, 4, 1, 3],
        ),
        (
            [1, 3, 0, 4, 4],
            [2, 0, 4, 1, 3],
        ),
        (
            [1, 3, 0, 4, 2],
            [2, 0, 3, 1, 4],
        ),
    ),
)
def test_metadata_rejects_invalid_or_noninverse_permutations(
    sorted_to_nonempty: list[int],
    nonempty_to_sorted: list[int],
) -> None:
    with pytest.raises(ValueError):
        RecurrentExecutionMetadata(
            history_lengths=_lengths(),
            nonempty_node_indices=torch.tensor(
                [0, 2, 3, 4, 5],
                dtype=torch.long,
            ),
            zero_history_node_indices=torch.tensor(
                [1],
                dtype=torch.long,
            ),
            sorted_to_nonempty=torch.tensor(
                sorted_to_nonempty,
                dtype=torch.long,
            ),
            nonempty_to_sorted=torch.tensor(
                nonempty_to_sorted,
                dtype=torch.long,
            ),
            execution_path="packed",
            sort_was_applied=True,
            original_padding_direction="right",
        )


def test_metadata_rejects_packed_lengths_that_are_not_nonincreasing() -> None:
    with pytest.raises(ValueError):
        RecurrentExecutionMetadata(
            history_lengths=_lengths(),
            nonempty_node_indices=torch.tensor(
                [0, 2, 3, 4, 5],
                dtype=torch.long,
            ),
            zero_history_node_indices=torch.tensor(
                [1],
                dtype=torch.long,
            ),
            sorted_to_nonempty=torch.arange(
                5,
                dtype=torch.long,
            ),
            nonempty_to_sorted=torch.arange(
                5,
                dtype=torch.long,
            ),
            execution_path="packed",
            sort_was_applied=False,
            original_padding_direction="right",
        )


def test_metadata_rejects_unstable_equal_length_order() -> None:
    with pytest.raises(ValueError):
        RecurrentExecutionMetadata(
            history_lengths=_lengths(),
            nonempty_node_indices=torch.tensor(
                [0, 2, 3, 4, 5],
                dtype=torch.long,
            ),
            zero_history_node_indices=torch.tensor(
                [1],
                dtype=torch.long,
            ),
            sorted_to_nonempty=torch.tensor(
                [1, 3, 4, 0, 2],
                dtype=torch.long,
            ),
            nonempty_to_sorted=torch.tensor(
                [3, 0, 4, 1, 2],
                dtype=torch.long,
            ),
            execution_path="packed",
            sort_was_applied=True,
            original_padding_direction="right",
        )


def test_metadata_rejects_incorrect_sort_was_applied_flag() -> None:
    with pytest.raises(ValueError):
        RecurrentExecutionMetadata(
            history_lengths=_lengths(),
            nonempty_node_indices=torch.tensor(
                [0, 2, 3, 4, 5],
                dtype=torch.long,
            ),
            zero_history_node_indices=torch.tensor(
                [1],
                dtype=torch.long,
            ),
            sorted_to_nonempty=torch.tensor(
                [1, 3, 0, 4, 2],
                dtype=torch.long,
            ),
            nonempty_to_sorted=torch.tensor(
                [2, 0, 4, 1, 3],
                dtype=torch.long,
            ),
            execution_path="packed",
            sort_was_applied=False,
            original_padding_direction="right",
        )


def test_metadata_rejects_reference_nonidentity_permutation() -> None:
    with pytest.raises(ValueError):
        RecurrentExecutionMetadata(
            history_lengths=_lengths(),
            nonempty_node_indices=torch.tensor(
                [0, 2, 3, 4, 5],
                dtype=torch.long,
            ),
            zero_history_node_indices=torch.tensor(
                [1],
                dtype=torch.long,
            ),
            sorted_to_nonempty=torch.tensor(
                [1, 0, 2, 3, 4],
                dtype=torch.long,
            ),
            nonempty_to_sorted=torch.tensor(
                [1, 0, 2, 3, 4],
                dtype=torch.long,
            ),
            execution_path="reference",
            sort_was_applied=False,
            original_padding_direction="right",
        )


def test_metadata_rejects_reference_sort_flag() -> None:
    with pytest.raises(ValueError):
        RecurrentExecutionMetadata(
            history_lengths=_lengths(),
            nonempty_node_indices=torch.tensor(
                [0, 2, 3, 4, 5],
                dtype=torch.long,
            ),
            zero_history_node_indices=torch.tensor(
                [1],
                dtype=torch.long,
            ),
            sorted_to_nonempty=torch.arange(
                5,
                dtype=torch.long,
            ),
            nonempty_to_sorted=torch.arange(
                5,
                dtype=torch.long,
            ),
            execution_path="reference",
            sort_was_applied=True,
            original_padding_direction="right",
        )


@pytest.mark.parametrize(
    (
        "execution_path",
        "padding_direction",
    ),
    (
        (
            "unknown",
            "right",
        ),
        (
            "packed",
            "unknown",
        ),
    ),
)
def test_metadata_rejects_unknown_enum_values(
    execution_path: str,
    padding_direction: str,
) -> None:
    with pytest.raises(ValueError):
        RecurrentExecutionMetadata(
            history_lengths=_lengths(),
            nonempty_node_indices=torch.tensor(
                [0, 2, 3, 4, 5],
                dtype=torch.long,
            ),
            zero_history_node_indices=torch.tensor(
                [1],
                dtype=torch.long,
            ),
            sorted_to_nonempty=torch.tensor(
                [1, 3, 0, 4, 2],
                dtype=torch.long,
            ),
            nonempty_to_sorted=torch.tensor(
                [2, 0, 4, 1, 3],
                dtype=torch.long,
            ),
            execution_path=execution_path,
            sort_was_applied=True,
            original_padding_direction=padding_direction,
        )


# =============================================================================
# RecurrentStateLayout
# =============================================================================


@pytest.mark.parametrize(
    (
        "num_layers",
        "num_directions",
        "hidden_dim",
        "output_dim",
    ),
    (
        (
            1,
            1,
            3,
            3,
        ),
        (
            1,
            2,
            3,
            6,
        ),
        (
            3,
            1,
            5,
            5,
        ),
        (
            3,
            2,
            5,
            10,
        ),
    ),
)
def test_state_layout_properties(
    num_layers: int,
    num_directions: int,
    hidden_dim: int,
    output_dim: int,
) -> None:
    layout = RecurrentStateLayout(
        num_layers=num_layers,
        num_directions=num_directions,
        hidden_dim=hidden_dim,
    )

    assert layout.flat_layer_direction_size == (
        num_layers
        * num_directions
    )
    assert layout.output_dim == output_dim
    assert layout.is_bidirectional == (
        num_directions == 2
    )
    assert layout.state_axis_order == (
        "layer_direction_node_hidden"
    )
    assert layout.flat_axis_order == (
        "layer_major_direction_minor"
    )


def test_state_layout_direction_order_is_derived() -> None:
    assert RecurrentStateLayout(
        1,
        1,
        2,
    ).direction_order == (
        "forward",
    )
    assert RecurrentStateLayout(
        1,
        2,
        2,
    ).direction_order == (
        "forward",
        "backward",
    )


def test_state_layout_flat_index_is_layer_major_direction_minor() -> None:
    layout = RecurrentStateLayout(
        num_layers=3,
        num_directions=2,
        hidden_dim=4,
    )

    assert layout.flat_index(
        layer_index=0,
        direction_index=0,
    ) == 0
    assert layout.flat_index(
        layer_index=0,
        direction_index=1,
    ) == 1
    assert layout.flat_index(
        layer_index=1,
        direction_index=0,
    ) == 2
    assert layout.flat_index(
        layer_index=2,
        direction_index=1,
    ) == 5


@pytest.mark.parametrize(
    (
        "layer_index",
        "direction_index",
        "error_type",
    ),
    (
        (
            -1,
            0,
            ValueError,
        ),
        (
            3,
            0,
            IndexError,
        ),
        (
            0,
            -1,
            ValueError,
        ),
        (
            0,
            2,
            IndexError,
        ),
    ),
)
def test_state_layout_flat_index_rejects_invalid_coordinates(
    layer_index: int,
    direction_index: int,
    error_type: type[Exception],
) -> None:
    layout = RecurrentStateLayout(
        3,
        2,
        4,
    )

    with pytest.raises(error_type):
        layout.flat_index(
            layer_index=layer_index,
            direction_index=direction_index,
        )


@pytest.mark.parametrize(
    (
        "num_layers",
        "num_directions",
        "node_count",
        "hidden_dim",
    ),
    (
        (
            1,
            1,
            1,
            1,
        ),
        (
            1,
            2,
            4,
            3,
        ),
        (
            3,
            1,
            2,
            5,
        ),
        (
            2,
            2,
            6,
            4,
        ),
    ),
)
def test_state_layout_canonical_flat_round_trip(
    num_layers: int,
    num_directions: int,
    node_count: int,
    hidden_dim: int,
) -> None:
    layout = RecurrentStateLayout(
        num_layers,
        num_directions,
        hidden_dim,
    )
    canonical = torch.arange(
        num_layers
        * num_directions
        * node_count
        * hidden_dim,
        dtype=torch.float64,
    ).reshape(
        num_layers,
        num_directions,
        node_count,
        hidden_dim,
    )

    flat = layout.flatten_state(
        canonical
    )
    restored = layout.unflatten_state(
        flat
    )

    assert flat.shape == (
        num_layers
        * num_directions,
        node_count,
        hidden_dim,
    )
    assert torch.equal(
        restored,
        canonical,
    )
    assert flat.data_ptr() == canonical.data_ptr()
    assert restored.data_ptr() == canonical.data_ptr()


def test_state_layout_validation_accepts_empty_node_axis() -> None:
    layout = _state_layout()
    state = torch.empty(
        LAYERS,
        DIRECTIONS,
        0,
        H,
        dtype=torch.float32,
    )

    layout.validate_canonical_state(
        state,
        node_count=0,
    )
    flat = layout.flatten_state(
        state
    )

    assert flat.shape == (
        LAYERS
        * DIRECTIONS,
        0,
        H,
    )


@pytest.mark.parametrize(
    (
        "num_layers",
        "num_directions",
        "hidden_dim",
        "error_type",
    ),
    (
        (
            0,
            1,
            3,
            ValueError,
        ),
        (
            1,
            0,
            3,
            ValueError,
        ),
        (
            1,
            3,
            3,
            ValueError,
        ),
        (
            1,
            1,
            0,
            ValueError,
        ),
        (
            True,
            1,
            3,
            TypeError,
        ),
    ),
)
def test_state_layout_rejects_invalid_dimensions(
    num_layers: int,
    num_directions: int,
    hidden_dim: int,
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type):
        RecurrentStateLayout(
            num_layers=num_layers,
            num_directions=num_directions,
            hidden_dim=hidden_dim,
        )


@pytest.mark.parametrize(
    "state",
    (
        torch.zeros(
            2,
            2,
            3,
        ),
        torch.zeros(
            2,
            2,
            3,
            4,
            5,
        ),
        torch.zeros(
            2,
            2,
            3,
            4,
            dtype=torch.long,
        ),
        torch.full(
            (
                2,
                2,
                3,
                4,
            ),
            float(
                "nan"
            ),
        ),
    ),
)
def test_state_layout_rejects_invalid_canonical_state(
    state: torch.Tensor,
) -> None:
    layout = RecurrentStateLayout(
        2,
        2,
        4,
    )

    with pytest.raises(ValueError):
        layout.validate_canonical_state(
            state
        )


@pytest.mark.parametrize(
    "shape",
    (
        (
            1,
            2,
            3,
            4,
        ),
        (
            2,
            1,
            3,
            4,
        ),
        (
            2,
            2,
            3,
            5,
        ),
        (
            2,
            2,
            4,
            4,
        ),
    ),
)
def test_state_layout_rejects_canonical_dimension_mismatch(
    shape: tuple[int, int, int, int],
) -> None:
    layout = RecurrentStateLayout(
        2,
        2,
        4,
    )
    state = torch.zeros(
        *shape
    )

    with pytest.raises(ValueError):
        layout.validate_canonical_state(
            state,
            node_count=3,
        )


@pytest.mark.parametrize(
    "state",
    (
        torch.zeros(
            4,
            3,
        ),
        torch.zeros(
            4,
            3,
            4,
            5,
        ),
        torch.zeros(
            4,
            3,
            4,
            dtype=torch.long,
        ),
        torch.full(
            (
                4,
                3,
                4,
            ),
            float(
                "inf"
            ),
        ),
    ),
)
def test_state_layout_rejects_invalid_flat_state(
    state: torch.Tensor,
) -> None:
    layout = RecurrentStateLayout(
        2,
        2,
        4,
    )

    with pytest.raises(ValueError):
        layout.validate_flat_state(
            state
        )


@pytest.mark.parametrize(
    "shape",
    (
        (
            3,
            3,
            4,
        ),
        (
            4,
            3,
            5,
        ),
        (
            4,
            4,
            4,
        ),
    ),
)
def test_state_layout_rejects_flat_dimension_mismatch(
    shape: tuple[int, int, int],
) -> None:
    layout = RecurrentStateLayout(
        2,
        2,
        4,
    )
    state = torch.zeros(
        *shape
    )

    with pytest.raises(ValueError):
        layout.validate_flat_state(
            state,
            node_count=3,
        )


def test_state_layout_fingerprint_is_deterministic_and_sensitive() -> None:
    first = RecurrentStateLayout(
        2,
        2,
        3,
    )
    same = RecurrentStateLayout(
        2,
        2,
        3,
    )
    different = RecurrentStateLayout(
        2,
        1,
        3,
    )

    assert first.fingerprint() == same.fingerprint()
    assert first.fingerprint() != different.fingerprint()


def test_state_layout_replace_revalidates() -> None:
    layout = RecurrentStateLayout(
        2,
        2,
        3,
    )
    replaced = layout.replace(
        hidden_dim=5
    )

    assert replaced.hidden_dim == 5
    assert replaced.output_dim == 10

    with pytest.raises(ValueError):
        layout.replace(
            num_directions=3
        )


def test_state_layout_is_frozen() -> None:
    layout = _state_layout()

    with pytest.raises(
        (
            FrozenInstanceError,
            AttributeError,
        )
    ):
        layout.hidden_dim = 99  # type: ignore[misc]


# =============================================================================
# RecurrentSequenceEncoderRun: valid GRU/LSTM contracts
# =============================================================================


@pytest.mark.parametrize(
    (
        "factory",
        "expected_kind",
        "has_cell",
    ),
    (
        (
            _gru_run,
            TemporalSequenceEncoderKind.GRU,
            False,
        ),
        (
            _lstm_run,
            TemporalSequenceEncoderKind.LSTM,
            True,
        ),
    ),
)
def test_recurrent_run_valid_contracts(
    factory,
    expected_kind: TemporalSequenceEncoderKind,
    has_cell: bool,
) -> None:
    run = factory()

    assert run.encoder_kind == expected_kind
    assert run.node_count == N
    assert run.sequence_length == T
    assert run.hidden_dim == H
    assert run.output_dim == H * DIRECTIONS
    assert run.num_layers == LAYERS
    assert run.num_directions == DIRECTIONS
    assert run.is_bidirectional
    assert run.has_cell_state == has_cell
    assert run.device.type == "cpu"
    assert run.dtype == torch.float64
    assert run.source_history is (
        run.public_output.source_history
    )


def test_gru_run_has_no_cell_state() -> None:
    run = _gru_run()

    assert run.final_cell_state is None
    assert run.final_cell_state_flat is None


def test_lstm_run_preserves_hidden_and_cell_layouts() -> None:
    run = _lstm_run()

    assert run.final_hidden_state.shape == (
        LAYERS,
        DIRECTIONS,
        N,
        H,
    )
    assert run.final_cell_state is not None
    assert run.final_cell_state.shape == (
        LAYERS,
        DIRECTIONS,
        N,
        H,
    )
    assert run.final_hidden_state_flat.shape == (
        LAYERS
        * DIRECTIONS,
        N,
        H,
    )
    assert run.final_cell_state_flat is not None
    assert run.final_cell_state_flat.shape == (
        LAYERS
        * DIRECTIONS,
        N,
        H,
    )


@pytest.mark.parametrize(
    "factory",
    (
        _gru_run,
        _lstm_run,
    ),
)
def test_run_preserves_autograd_state_tensors(
    factory,
) -> None:
    run = factory(
        requires_grad=True
    )

    assert run.public_output.encoded_sequence.requires_grad
    assert run.final_hidden_state.requires_grad

    if run.final_cell_state is not None:
        assert run.final_cell_state.requires_grad

    loss = (
        run.public_output.encoded_sequence.sum()
        + run.final_hidden_state.sum()
    )

    if run.final_cell_state is not None:
        loss = (
            loss
            + run.final_cell_state.sum()
        )

    loss.backward()

    assert run.public_output.encoded_sequence.grad is not None
    assert run.final_hidden_state.grad is not None

    if run.final_cell_state is not None:
        assert run.final_cell_state.grad is not None


@pytest.mark.parametrize(
    "factory",
    (
        _gru_run,
        _lstm_run,
    ),
)
def test_run_zero_history_slices_are_exact_zero(
    factory,
) -> None:
    run = factory()
    zero_indices = (
        run
        .execution_metadata
        .zero_history_node_indices
    )

    assert torch.count_nonzero(
        run.public_output.encoded_sequence.index_select(
            0,
            zero_indices,
        )
    ).item() == 0
    assert torch.count_nonzero(
        run.final_hidden_state.index_select(
            2,
            zero_indices,
        )
    ).item() == 0

    if run.final_cell_state is not None:
        assert torch.count_nonzero(
            run.final_cell_state.index_select(
                2,
                zero_indices,
            )
        ).item() == 0


@pytest.mark.parametrize(
    (
        "encoder_kind",
        "packed",
    ),
    (
        (
            TemporalSequenceEncoderKind.GRU,
            True,
        ),
        (
            TemporalSequenceEncoderKind.GRU,
            False,
        ),
        (
            TemporalSequenceEncoderKind.LSTM,
            True,
        ),
        (
            TemporalSequenceEncoderKind.LSTM,
            False,
        ),
    ),
)
def test_all_zero_run_contract(
    encoder_kind: TemporalSequenceEncoderKind,
    packed: bool,
) -> None:
    history = _all_zero_history()
    output = _all_zero_public_output(
        history=history,
        encoder_kind=encoder_kind,
        pack_sequences=packed,
    )
    layout = RecurrentStateLayout(
        1,
        1,
        H,
    )
    hidden = torch.zeros(
        1,
        1,
        history.node_count,
        H,
        dtype=history.dtype,
    )
    cell = (
        torch.zeros_like(
            hidden
        )
        if encoder_kind
        == TemporalSequenceEncoderKind.LSTM
        else None
    )
    metadata = _all_zero_metadata(
        history,
        execution_path=(
            RecurrentExecutionPath.PACKED
            if packed
            else RecurrentExecutionPath.REFERENCE
        ),
    )

    run = RecurrentSequenceEncoderRun(
        public_output=output,
        final_hidden_state=hidden,
        final_cell_state=cell,
        state_layout=layout,
        execution_metadata=metadata,
    )

    assert run.execution_metadata.all_zero_history
    assert torch.count_nonzero(
        run.public_output.encoded_sequence
    ).item() == 0
    assert torch.count_nonzero(
        run.final_hidden_state
    ).item() == 0


@pytest.mark.parametrize(
    "factory",
    (
        _gru_run,
        _lstm_run,
    ),
)
def test_run_fingerprint_is_deterministic(
    factory,
) -> None:
    first = factory()
    second = factory()

    assert first.semantic_dict() == second.semantic_dict()
    assert first.value_fingerprint() == (
        second.value_fingerprint()
    )
    assert first.lineage_fingerprint() == (
        second.lineage_fingerprint()
    )
    assert first.fingerprint() == first.lineage_fingerprint()


def test_run_value_fingerprint_changes_with_state_values() -> None:
    first = _gru_run()
    second = _gru_run()

    with torch.no_grad():
        second.final_hidden_state[
            0,
            0,
            0,
            0,
        ] += 1.0

    assert first.value_fingerprint() != (
        second.value_fingerprint()
    )


def test_run_exposes_public_provenance_fingerprints() -> None:
    run = _gru_run()

    assert run.architecture_fingerprint == (
        run
        .public_output
        .architecture_fingerprint
    )
    assert run.computation_lineage_fingerprint == (
        run
        .public_output
        .computation_lineage_fingerprint
    )


def test_run_replace_revalidates() -> None:
    run = _gru_run()
    replaced = run.replace(
        run_name="gru-audit-run"
    )

    assert replaced.run_name == "gru-audit-run"
    assert replaced.public_output is run.public_output


def test_run_is_frozen() -> None:
    run = _gru_run()

    with pytest.raises(
        (
            FrozenInstanceError,
            AttributeError,
        )
    ):
        run.run_name = "changed"  # type: ignore[misc]


# =============================================================================
# RecurrentSequenceEncoderRun: invalid cross-object contracts
# =============================================================================


def test_run_rejects_wrong_public_output_type() -> None:
    with pytest.raises(TypeError):
        RecurrentSequenceEncoderRun(
            public_output=object(),  # type: ignore[arg-type]
            final_hidden_state=torch.zeros(
                1,
                1,
                1,
                H,
            ),
            final_cell_state=None,
            state_layout=RecurrentStateLayout(
                1,
                1,
                H,
            ),
            execution_metadata=_packed_metadata(),
        )


def test_run_rejects_wrong_layout_type() -> None:
    output = _public_output()

    with pytest.raises(TypeError):
        RecurrentSequenceEncoderRun(
            public_output=output,
            final_hidden_state=torch.zeros(
                LAYERS,
                DIRECTIONS,
                N,
                H,
                dtype=torch.float64,
            ),
            final_cell_state=None,
            state_layout=object(),  # type: ignore[arg-type]
            execution_metadata=_packed_metadata(),
        )


def test_run_rejects_wrong_metadata_type() -> None:
    output = _public_output()

    with pytest.raises(TypeError):
        RecurrentSequenceEncoderRun(
            public_output=output,
            final_hidden_state=torch.zeros(
                LAYERS,
                DIRECTIONS,
                N,
                H,
                dtype=torch.float64,
            ),
            final_cell_state=None,
            state_layout=_state_layout(),
            execution_metadata=object(),  # type: ignore[arg-type]
        )


def test_run_rejects_non_recurrent_public_encoder_kind() -> None:
    history = _history()
    config = _config()
    provenance = (
        build_recurrent_sequence_computation_provenance(
            source_history=history,
            config=config,
            execution_path="packed",
            sort_was_applied=True,
            module_training=False,
            nonempty_node_count=5,
            zero_history_count=1,
            adapter_executed=True,
            recurrent_kernel_executed=True,
            all_zero_history_short_circuit=False,
        )
    )
    output = TemporalSequenceEncoding(
        encoded_sequence=_encoded_values(
            history,
            output_dim=H * DIRECTIONS,
        ),
        source_history=history,
        encoder_kind=(
            TemporalSequenceEncoderKind.TRANSFORMER
        ),
        computation_provenance=provenance,
    )

    with pytest.raises(ValueError):
        RecurrentSequenceEncoderRun(
            public_output=output,
            final_hidden_state=_state(
                layout=_state_layout(),
                history=history,
            ),
            final_cell_state=None,
            state_layout=_state_layout(),
            execution_metadata=_packed_metadata(
                history=history
            ),
        )


def test_gru_run_rejects_cell_state() -> None:
    run = _gru_run()

    with pytest.raises(ValueError):
        run.replace(
            final_cell_state=torch.zeros_like(
                run.final_hidden_state
            )
        )


def test_lstm_run_requires_cell_state() -> None:
    run = _lstm_run()

    with pytest.raises(ValueError):
        run.replace(
            final_cell_state=None
        )


@pytest.mark.parametrize(
    "shape",
    (
        (
            1,
            DIRECTIONS,
            N,
            H,
        ),
        (
            LAYERS,
            1,
            N,
            H,
        ),
        (
            LAYERS,
            DIRECTIONS,
            N - 1,
            H,
        ),
        (
            LAYERS,
            DIRECTIONS,
            N,
            H + 1,
        ),
    ),
)
def test_run_rejects_hidden_state_shape_mismatch(
    shape: tuple[int, int, int, int],
) -> None:
    run = _gru_run()

    with pytest.raises(ValueError):
        run.replace(
            final_hidden_state=torch.zeros(
                *shape,
                dtype=run.dtype,
            )
        )


def test_run_rejects_output_width_layout_mismatch() -> None:
    history = _history()
    output = _public_output(
        history=history,
        bidirectional=False,
    )
    metadata = _packed_metadata(
        history=history
    )
    wrong_layout = RecurrentStateLayout(
        num_layers=LAYERS,
        num_directions=2,
        hidden_dim=H,
    )
    hidden = _state(
        layout=wrong_layout,
        history=history,
    )

    with pytest.raises(ValueError):
        RecurrentSequenceEncoderRun(
            public_output=output,
            final_hidden_state=hidden,
            final_cell_state=None,
            state_layout=wrong_layout,
            execution_metadata=metadata,
        )


def test_run_rejects_hidden_dtype_mismatch() -> None:
    run = _gru_run()

    with pytest.raises(ValueError):
        run.replace(
            final_hidden_state=(
                run
                .final_hidden_state
                .float()
            )
        )


def test_lstm_run_rejects_cell_shape_mismatch() -> None:
    run = _lstm_run()

    with pytest.raises(ValueError):
        run.replace(
            final_cell_state=torch.zeros(
                1,
                DIRECTIONS,
                N,
                H,
                dtype=run.dtype,
            )
        )


def test_lstm_run_rejects_cell_dtype_mismatch() -> None:
    run = _lstm_run()

    with pytest.raises(ValueError):
        run.replace(
            final_cell_state=(
                run
                .final_cell_state
                .float()
            )
        )


def test_run_rejects_metadata_node_count_mismatch() -> None:
    run = _gru_run()
    smaller_history = _history(
        lengths=torch.tensor(
            [2, 0, 4],
            dtype=torch.long,
        )
    )
    smaller_metadata = RecurrentExecutionMetadata(
        history_lengths=smaller_history.valid_lengths,
        nonempty_node_indices=torch.tensor(
            [0, 2],
            dtype=torch.long,
        ),
        zero_history_node_indices=torch.tensor(
            [1],
            dtype=torch.long,
        ),
        sorted_to_nonempty=torch.tensor(
            [1, 0],
            dtype=torch.long,
        ),
        nonempty_to_sorted=torch.tensor(
            [1, 0],
            dtype=torch.long,
        ),
        execution_path="packed",
        sort_was_applied=True,
        original_padding_direction="right",
    )

    with pytest.raises(ValueError):
        run.replace(
            execution_metadata=smaller_metadata
        )


def test_run_rejects_metadata_history_length_mismatch() -> None:
    run = _gru_run()
    lengths = run.execution_metadata.history_lengths.clone()
    lengths[0] = 1
    bad_metadata = RecurrentExecutionMetadata(
        history_lengths=lengths,
        nonempty_node_indices=torch.tensor(
            [0, 2, 3, 4, 5],
            dtype=torch.long,
        ),
        zero_history_node_indices=torch.tensor(
            [1],
            dtype=torch.long,
        ),
        sorted_to_nonempty=torch.tensor(
            [1, 3, 4, 0, 2],
            dtype=torch.long,
        ),
        nonempty_to_sorted=torch.tensor(
            [3, 0, 4, 1, 2],
            dtype=torch.long,
        ),
        execution_path="packed",
        sort_was_applied=True,
        original_padding_direction="right",
    )

    with pytest.raises(ValueError):
        run.replace(
            execution_metadata=bad_metadata
        )


def test_run_rejects_original_padding_direction_mismatch() -> None:
    run = _gru_run()
    metadata = run.execution_metadata.replace(
        original_padding_direction=(
            TemporalPaddingDirection.LEFT
        )
    )

    with pytest.raises(ValueError):
        run.replace(
            execution_metadata=metadata
        )


@pytest.mark.parametrize(
    "state_kind",
    (
        "hidden",
        "cell",
    ),
)
def test_run_rejects_nonfinite_states(
    state_kind: str,
) -> None:
    run = (
        _lstm_run()
        if state_kind == "cell"
        else _gru_run()
    )

    if state_kind == "hidden":
        corrupted = (
            run
            .final_hidden_state
            .detach()
            .clone()
        )
        corrupted[0, 0, 0, 0] = float(
            "nan"
        )

        with pytest.raises(ValueError):
            run.replace(
                final_hidden_state=corrupted
            )
    else:
        corrupted = (
            run
            .final_cell_state
            .detach()
            .clone()
        )
        corrupted[0, 0, 0, 0] = float(
            "inf"
        )

        with pytest.raises(ValueError):
            run.replace(
                final_cell_state=corrupted
            )


def test_run_rejects_nonzero_zero_history_hidden_state() -> None:
    run = _gru_run()
    hidden = (
        run
        .final_hidden_state
        .detach()
        .clone()
    )
    hidden[
        0,
        0,
        1,
        0,
    ] = 1.0

    with pytest.raises(ValueError):
        run.replace(
            final_hidden_state=hidden
        )


def test_lstm_run_rejects_nonzero_zero_history_cell_state() -> None:
    run = _lstm_run()
    cell = (
        run
        .final_cell_state
        .detach()
        .clone()
    )
    cell[
        0,
        0,
        1,
        0,
    ] = 1.0

    with pytest.raises(ValueError):
        run.replace(
            final_cell_state=cell
        )


def test_run_rejects_nonzero_zero_history_encoded_sequence() -> None:
    history = _history()
    output = _public_output(
        history=history,
    )

    # TemporalSequenceEncoding correctly rejects nonzero padding during its
    # own construction. Mutate the already-valid tensor afterward to verify
    # that the recurrent run independently rechecks zero-history rows.
    with torch.no_grad():
        output.encoded_sequence[
            1,
            0,
            0,
        ] = 1.0

    with pytest.raises(ValueError):
        RecurrentSequenceEncoderRun(
            public_output=output,
            final_hidden_state=_state(
                layout=_state_layout(),
                history=history,
            ),
            final_cell_state=None,
            state_layout=_state_layout(),
            execution_metadata=_packed_metadata(
                history=history
            ),
        )


# =============================================================================
# Private canonical recurrent batch
# =============================================================================


def _canonical_batch(
    *,
    requires_grad: bool = False,
) -> _CanonicalRecurrentBatch:
    values = torch.tensor(
        [
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [0.0, 0.0],
            ],
            [
                [5.0, 6.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ],
        ],
        dtype=torch.float64,
        requires_grad=requires_grad,
    )

    return _CanonicalRecurrentBatch(
        values=values,
        timestep_mask=torch.tensor(
            [
                [True, True, False],
                [True, False, False],
            ],
            dtype=torch.bool,
        ),
        lengths=torch.tensor(
            [2, 1],
            dtype=torch.long,
        ),
        nonempty_node_indices=torch.tensor(
            [0, 3],
            dtype=torch.long,
        ),
        source_node_count=4,
        original_padding_direction="left",
    )


def test_canonical_batch_valid_contract() -> None:
    batch = _canonical_batch()

    assert batch.nonempty_node_count == 2
    assert batch.sequence_length == 3
    assert batch.feature_dim == 2
    assert batch.device.type == "cpu"
    assert batch.dtype == torch.float64
    assert batch.original_padding_direction == (
        TemporalPaddingDirection.LEFT
    )
    assert batch.canonical_padding_direction == (
        TemporalPaddingDirection.RIGHT
    )


def test_canonical_batch_preserves_value_autograd() -> None:
    batch = _canonical_batch(
        requires_grad=True
    )

    assert batch.values.requires_grad
    batch.values.sum().backward()
    assert batch.values.grad is not None


def test_canonical_batch_clones_integer_metadata() -> None:
    lengths = torch.tensor(
        [2, 1],
        dtype=torch.long,
    )
    indices = torch.tensor(
        [0, 3],
        dtype=torch.long,
    )
    values = torch.tensor(
        [
            [
                [1.0],
                [2.0],
                [0.0],
            ],
            [
                [3.0],
                [0.0],
                [0.0],
            ],
        ]
    )
    batch = _CanonicalRecurrentBatch(
        values=values,
        timestep_mask=torch.tensor(
            [
                [True, True, False],
                [True, False, False],
            ]
        ),
        lengths=lengths,
        nonempty_node_indices=indices,
        source_node_count=4,
        original_padding_direction="right",
    )

    lengths[0] = 1
    indices[0] = 2

    assert batch.lengths.tolist() == [
        2,
        1,
    ]
    assert batch.nonempty_node_indices.tolist() == [
        0,
        3,
    ]


@pytest.mark.parametrize(
    "feature_dim",
    (
        1,
        3,
        8,
    ),
)
def test_canonical_batch_supports_arbitrary_positive_feature_width(
    feature_dim: int,
) -> None:
    values = torch.zeros(
        2,
        3,
        feature_dim,
    )
    values[0, :2] = 1.0
    values[1, :1] = 2.0

    batch = _CanonicalRecurrentBatch(
        values=values,
        timestep_mask=torch.tensor(
            [
                [True, True, False],
                [True, False, False],
            ]
        ),
        lengths=torch.tensor(
            [2, 1],
            dtype=torch.long,
        ),
        nonempty_node_indices=torch.tensor(
            [0, 2],
            dtype=torch.long,
        ),
        source_node_count=3,
        original_padding_direction="none",
        value_stage="input_adapter_output",
    )

    assert batch.feature_dim == feature_dim
    assert batch.value_stage == "input_adapter_output"


@pytest.mark.parametrize(
    "execution_shape",
    (
        (
            0,
            3,
            2,
        ),
        (
            0,
            1,
            1,
        ),
    ),
)
def test_canonical_batch_accepts_empty_nonempty_axis(
    execution_shape: tuple[int, int, int],
) -> None:
    _, sequence_length, feature_dim = execution_shape
    batch = _CanonicalRecurrentBatch(
        values=torch.empty(
            execution_shape
        ),
        timestep_mask=torch.empty(
            0,
            sequence_length,
            dtype=torch.bool,
        ),
        lengths=torch.empty(
            0,
            dtype=torch.long,
        ),
        nonempty_node_indices=torch.empty(
            0,
            dtype=torch.long,
        ),
        source_node_count=3,
        original_padding_direction="right",
    )

    assert batch.nonempty_node_count == 0
    assert batch.feature_dim == feature_dim


@pytest.mark.parametrize(
    "values",
    (
        object(),
        torch.zeros(
            2,
            3,
        ),
        torch.zeros(
            2,
            3,
            2,
            dtype=torch.long,
        ),
        torch.full(
            (
                2,
                3,
                2,
            ),
            float(
                "nan"
            ),
        ),
        torch.empty(
            2,
            0,
            2,
        ),
        torch.empty(
            2,
            3,
            0,
        ),
    ),
)
def test_canonical_batch_rejects_invalid_values(
    values: object,
) -> None:
    with pytest.raises(
        (
            TypeError,
            ValueError,
        )
    ):
        _CanonicalRecurrentBatch(
            values=values,  # type: ignore[arg-type]
            timestep_mask=torch.tensor(
                [
                    [True, True, False],
                    [True, False, False],
                ]
            ),
            lengths=torch.tensor(
                [2, 1],
                dtype=torch.long,
            ),
            nonempty_node_indices=torch.tensor(
                [0, 3],
                dtype=torch.long,
            ),
            source_node_count=4,
            original_padding_direction="right",
        )


@pytest.mark.parametrize(
    "mask",
    (
        object(),
        torch.zeros(
            2,
            3,
            dtype=torch.float32,
        ),
        torch.zeros(
            2,
            2,
            dtype=torch.bool,
        ),
        torch.zeros(
            2,
            3,
            1,
            dtype=torch.bool,
        ),
    ),
)
def test_canonical_batch_rejects_invalid_mask(
    mask: object,
) -> None:
    with pytest.raises(
        (
            TypeError,
            ValueError,
        )
    ):
        _CanonicalRecurrentBatch(
            values=torch.zeros(
                2,
                3,
                2,
            ),
            timestep_mask=mask,  # type: ignore[arg-type]
            lengths=torch.tensor(
                [2, 1],
                dtype=torch.long,
            ),
            nonempty_node_indices=torch.tensor(
                [0, 3],
                dtype=torch.long,
            ),
            source_node_count=4,
            original_padding_direction="right",
        )


@pytest.mark.parametrize(
    (
        "lengths",
        "indices",
    ),
    (
        (
            torch.tensor(
                [2],
                dtype=torch.long,
            ),
            torch.tensor(
                [0, 3],
                dtype=torch.long,
            ),
        ),
        (
            torch.tensor(
                [2, 1],
                dtype=torch.long,
            ),
            torch.tensor(
                [0],
                dtype=torch.long,
            ),
        ),
        (
            torch.tensor(
                [2, 0],
                dtype=torch.long,
            ),
            torch.tensor(
                [0, 3],
                dtype=torch.long,
            ),
        ),
        (
            torch.tensor(
                [2, 4],
                dtype=torch.long,
            ),
            torch.tensor(
                [0, 3],
                dtype=torch.long,
            ),
        ),
        (
            torch.tensor(
                [2, 1],
                dtype=torch.long,
            ),
            torch.tensor(
                [3, 0],
                dtype=torch.long,
            ),
        ),
        (
            torch.tensor(
                [2, 1],
                dtype=torch.long,
            ),
            torch.tensor(
                [0, 4],
                dtype=torch.long,
            ),
        ),
    ),
)
def test_canonical_batch_rejects_invalid_lengths_or_indices(
    lengths: torch.Tensor,
    indices: torch.Tensor,
) -> None:
    with pytest.raises(ValueError):
        _CanonicalRecurrentBatch(
            values=torch.tensor(
                [
                    [
                        [1.0],
                        [2.0],
                        [0.0],
                    ],
                    [
                        [3.0],
                        [0.0],
                        [0.0],
                    ],
                ]
            ),
            timestep_mask=torch.tensor(
                [
                    [True, True, False],
                    [True, False, False],
                ]
            ),
            lengths=lengths,
            nonempty_node_indices=indices,
            source_node_count=4,
            original_padding_direction="right",
        )


def test_canonical_batch_rejects_lengths_mask_mismatch() -> None:
    with pytest.raises(ValueError):
        _CanonicalRecurrentBatch(
            values=torch.tensor(
                [
                    [
                        [1.0],
                        [2.0],
                        [0.0],
                    ],
                    [
                        [3.0],
                        [0.0],
                        [0.0],
                    ],
                ]
            ),
            timestep_mask=torch.tensor(
                [
                    [True, True, False],
                    [True, True, False],
                ]
            ),
            lengths=torch.tensor(
                [2, 1],
                dtype=torch.long,
            ),
            nonempty_node_indices=torch.tensor(
                [0, 3],
                dtype=torch.long,
            ),
            source_node_count=4,
            original_padding_direction="right",
        )


@pytest.mark.parametrize(
    "mask",
    (
        torch.tensor(
            [
                [True, False, True],
                [True, False, False],
            ]
        ),
        torch.tensor(
            [
                [False, True, True],
                [True, False, False],
            ]
        ),
    ),
)
def test_canonical_batch_rejects_non_prefix_valid_mask(
    mask: torch.Tensor,
) -> None:
    with pytest.raises(ValueError):
        _CanonicalRecurrentBatch(
            values=torch.zeros(
                2,
                3,
                1,
            ),
            timestep_mask=mask,
            lengths=mask.sum(
                dim=1,
                dtype=torch.long,
            ),
            nonempty_node_indices=torch.tensor(
                [0, 3],
                dtype=torch.long,
            ),
            source_node_count=4,
            original_padding_direction="left",
        )


def test_canonical_batch_rejects_nonzero_padded_values() -> None:
    values = torch.tensor(
        [
            [
                [1.0],
                [2.0],
                [9.0],
            ],
            [
                [3.0],
                [0.0],
                [0.0],
            ],
        ]
    )

    with pytest.raises(ValueError):
        _CanonicalRecurrentBatch(
            values=values,
            timestep_mask=torch.tensor(
                [
                    [True, True, False],
                    [True, False, False],
                ]
            ),
            lengths=torch.tensor(
                [2, 1],
                dtype=torch.long,
            ),
            nonempty_node_indices=torch.tensor(
                [0, 3],
                dtype=torch.long,
            ),
            source_node_count=4,
            original_padding_direction="right",
        )


@pytest.mark.parametrize(
    "source_node_count",
    (
        0,
        -1,
        True,
    ),
)
def test_canonical_batch_rejects_invalid_source_node_count(
    source_node_count: int,
) -> None:
    with pytest.raises(
        (
            TypeError,
            ValueError,
        )
    ):
        _CanonicalRecurrentBatch(
            values=torch.empty(
                0,
                3,
                2,
            ),
            timestep_mask=torch.empty(
                0,
                3,
                dtype=torch.bool,
            ),
            lengths=torch.empty(
                0,
                dtype=torch.long,
            ),
            nonempty_node_indices=torch.empty(
                0,
                dtype=torch.long,
            ),
            source_node_count=source_node_count,
            original_padding_direction="right",
        )


@pytest.mark.parametrize(
    (
        "padding_direction",
        "value_stage",
    ),
    (
        (
            "unknown",
            "canonical_raw_history",
        ),
        (
            "right",
            "",
        ),
    ),
)
def test_canonical_batch_rejects_invalid_descriptive_fields(
    padding_direction: str,
    value_stage: str,
) -> None:
    with pytest.raises(ValueError):
        _CanonicalRecurrentBatch(
            values=torch.empty(
                0,
                3,
                2,
            ),
            timestep_mask=torch.empty(
                0,
                3,
                dtype=torch.bool,
            ),
            lengths=torch.empty(
                0,
                dtype=torch.long,
            ),
            nonempty_node_indices=torch.empty(
                0,
                dtype=torch.long,
            ),
            source_node_count=3,
            original_padding_direction=padding_direction,
            value_stage=value_stage,
        )


# =============================================================================
# Dtype and conditional device coverage
# =============================================================================


@pytest.mark.parametrize(
    "dtype",
    (
        torch.float32,
        torch.float64,
    ),
)
def test_run_supports_common_floating_dtypes(
    dtype: torch.dtype,
) -> None:
    history = _history(
        dtype=dtype
    )
    run = _gru_run(
        history=history
    )

    assert run.dtype == dtype
    assert run.final_hidden_state.dtype == dtype


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is not available.",
)
def test_recurrent_schemas_support_cuda() -> None:
    history = _history(
        device="cuda",
        dtype=torch.float32,
    )
    metadata = _packed_metadata(
        history=history
    )
    layout = _state_layout()
    output = _public_output(
        history=history,
        encoded_values=_encoded_values(
            history,
            output_dim=layout.output_dim,
        ),
    )
    hidden = _state(
        layout=layout,
        history=history,
    )
    run = RecurrentSequenceEncoderRun(
        public_output=output,
        final_hidden_state=hidden,
        final_cell_state=None,
        state_layout=layout,
        execution_metadata=metadata,
    )

    assert metadata.device.type == "cuda"
    assert run.device.type == "cuda"
    assert run.final_hidden_state.device.type == "cuda"

    moved = metadata.to(
        "cpu"
    )
    assert moved.device.type == "cpu"
    assert moved.history_lengths.tolist() == (
        metadata.history_lengths.cpu().tolist()
    )
