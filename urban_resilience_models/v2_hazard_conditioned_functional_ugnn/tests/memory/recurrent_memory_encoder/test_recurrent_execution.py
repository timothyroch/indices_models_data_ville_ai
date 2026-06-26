"""
Consolidated execution-layout tests for Phase 6 recurrent memory encoders.

Target repository path:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            tests/
                memory/
                    recurrent_memory_encoder/
                        test_recurrent_execution.py

Module under test:
    memory/recurrent_memory_encoder/sequence_packing.py

The suite validates the complete shared execution transformation without
testing the future GRU/LSTM encoder classes themselves:

- source left/right/no-padding canonicalization;
- stable metadata-defined execution ordering;
- explicit sorted packing and total-length unpacking;
- restoration of nonempty-node and source temporal order;
- exact-zero source padding and zero-history rows;
- GRU and LSTM state restoration;
- reference-path helpers;
- packed/reference numerical and gradient equivalence;
- all-zero-history behavior;
- float32, float64, and conditional CUDA execution.
"""

from __future__ import annotations

import copy
import inspect

import pytest
import torch
from torch import nn
from torch.nn.utils.rnn import (
    PackedSequence,
)

import urban_resilience_models.v2_hazard_conditioned_functional_ugnn.memory.recurrent_memory_encoder.sequence_packing as sequence_packing_module
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.memory.config import (
    RecurrentCellKind,
    RecurrentSequenceEncoderConfig,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.memory.recurrent_memory_encoder.history_lengths import (
    build_recurrent_execution_metadata,
    build_recurrent_execution_metadata_from_lengths,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.memory.recurrent_memory_encoder.initial_state import (
    build_recurrent_state_layout,
    build_zero_gru_initial_state,
    build_zero_lstm_initial_state,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.memory.recurrent_memory_encoder.schemas import (
    RecurrentExecutionMetadata,
    RecurrentExecutionPath,
    RecurrentStateLayout,
    _CanonicalRecurrentBatch,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.memory.recurrent_memory_encoder.sequence_packing import (
    RECURRENT_SEQUENCE_PACKING_CANONICALIZATION_VERSION,
    RECURRENT_SEQUENCE_PACKING_COMPONENT_KIND,
    RECURRENT_SEQUENCE_PACKING_COMPONENT_NAME,
    RECURRENT_SEQUENCE_PACKING_IMPLEMENTATION_VERSION,
    RECURRENT_SEQUENCE_PACKING_PACK_POLICY,
    RECURRENT_SEQUENCE_PACKING_PADDING_VALUE,
    RECURRENT_SEQUENCE_PACKING_RESTORE_POLICY,
    RECURRENT_SEQUENCE_PACKING_SCIENTIFIC_INTERPRETATION,
    RECURRENT_SEQUENCE_PACKING_SORT_POLICY,
    RECURRENT_SEQUENCE_PACKING_STATE_POLICY,
    canonicalize_history,
    canonicalize_recurrent_history,
    gather_canonical_node_sequence,
    gather_single_node_valid_sequence,
    order_batch_for_execution,
    order_canonical_recurrent_batch_for_execution,
    pack_canonical_recurrent_batch,
    pack_recurrent_batch,
    restore_and_scatter_lstm_states,
    restore_and_scatter_recurrent_state,
    restore_and_scatter_recurrent_states,
    restore_nonempty_flat_state_order,
    restore_nonempty_sequence_order,
    restore_recurrent_sequence_to_source,
    restore_sequence_to_source,
    restore_single_node_sequence,
    restore_source_temporal_layout,
    restore_state_to_source,
    scatter_nonempty_canonical_state_to_source,
    scatter_nonempty_sequence_to_source,
    sort_canonical_recurrent_batch,
    unpack_recurrent_output,
    unpack_recurrent_sequence,
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
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.memory.schemas.temporal_coordinates import (
    RelativeTemporalCoordinates,
    TemporalPaddingDirection,
)


T = 4
D = 2
H = 3
LENGTHS = (
    2,
    0,
    4,
    1,
    3,
    2,
)


# =============================================================================
# Shared factories and execution oracles
# =============================================================================


def _mask_from_lengths(
    lengths: torch.Tensor,
    *,
    sequence_length: int = T,
    padding_direction: TemporalPaddingDirection,
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
    lengths: tuple[int, ...] | list[int] = LENGTHS,
    padding_direction: TemporalPaddingDirection = (
        TemporalPaddingDirection.RIGHT
    ),
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
    feature_dim: int = D,
    requires_grad: bool = False,
) -> HistoricalSequenceInputs:
    length_tensor = torch.tensor(
        lengths,
        dtype=torch.long,
        device=device,
    )
    node_count = int(
        length_tensor.numel()
    )
    mask = _mask_from_lengths(
        length_tensor,
        padding_direction=padding_direction,
    )
    values = torch.zeros(
        node_count,
        T,
        feature_dim,
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
        positions = torch.nonzero(
            mask[
                node_index
            ],
            as_tuple=False,
        ).flatten()
        logical_length = int(
            positions.numel()
        )

        for logical_position, temporal_position in enumerate(
            positions.tolist()
        ):
            values[
                node_index,
                temporal_position,
            ] = torch.arange(
                next_value,
                next_value + feature_dim,
                dtype=dtype,
                device=device,
            )
            coordinates[
                node_index,
                temporal_position,
            ] = float(
                logical_position - logical_length
            )
            next_value += float(
                feature_dim
            )

    if requires_grad:
        values.requires_grad_()

    zero_policy = (
        HistoryZeroLengthPolicy.ALLOW_ZERO_HISTORY
        if any(
            length == 0
            for length in lengths
        )
        else HistoryZeroLengthPolicy.ERROR
    )

    return HistoricalSequenceInputs(
        history=values,
        timestep_mask=mask,
        node_axis=TemporalNodeAxis(
            node_ids=tuple(
                f"node-{index}"
                for index in range(
                    node_count
                )
            ),
            node_batch_index=torch.zeros(
                node_count,
                dtype=torch.long,
                device=device,
            ),
            graph_count=1,
            graph_ids=("graph-0",),
            source_fingerprint=(
                "recurrent-execution-node-axis-v1"
            ),
        ),
        feature_axis=TemporalFeatureAxis(
            feature_names=tuple(
                f"feature-{index}"
                for index in range(
                    feature_dim
                )
            ),
            source_fingerprint=(
                "recurrent-execution-feature-axis-v1"
            ),
        ),
        temporal_coordinates=RelativeTemporalCoordinates(
            values=coordinates,
            unit="months",
        ),
        source_provenance=MemorySourceProvenance(
            source_name="recurrent-execution-panel",
            source_kind="historical-node-sequence",
            source_fingerprint=(
                "recurrent-execution-source-v1"
            ),
            preprocessing_fingerprint=(
                "recurrent-execution-preprocessing-v1"
            ),
        ),
        padding_direction=padding_direction,
        zero_length_policy=zero_policy,
    )


def _no_padding_history(
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> HistoricalSequenceInputs:
    return _history(
        lengths=(
            T,
            T,
            T,
        ),
        padding_direction=TemporalPaddingDirection.NONE,
        dtype=dtype,
        device=device,
    )


def _config(
    *,
    cell_kind: RecurrentCellKind = RecurrentCellKind.GRU,
    pack_sequences: bool = True,
    input_dim: int = D,
    hidden_dim: int = H,
    num_layers: int = 2,
    bidirectional: bool = True,
    use_bias: bool = True,
    dropout: float = 0.0,
) -> RecurrentSequenceEncoderConfig:
    return RecurrentSequenceEncoderConfig(
        cell_kind=cell_kind,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        bidirectional=bidirectional,
        use_bias=use_bias,
        dropout=dropout,
        pack_sequences=pack_sequences,
        enforce_sorted_lengths=False,
    )


def _metadata(
    history: HistoricalSequenceInputs,
    *,
    pack_sequences: bool = True,
    cell_kind: RecurrentCellKind = RecurrentCellKind.GRU,
    hidden_dim: int = H,
    num_layers: int = 2,
    bidirectional: bool = True,
) -> RecurrentExecutionMetadata:
    return build_recurrent_execution_metadata(
        history,
        _config(
            cell_kind=cell_kind,
            pack_sequences=pack_sequences,
            input_dim=history.feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
        ),
    )


def _canonical(
    history: HistoricalSequenceInputs,
    *,
    pack_sequences: bool = True,
    value_stage: str = "canonical_test_values",
) -> tuple[
    RecurrentExecutionMetadata,
    _CanonicalRecurrentBatch,
]:
    metadata = _metadata(
        history,
        pack_sequences=pack_sequences,
    )
    batch = canonicalize_recurrent_history(
        history,
        metadata,
        value_stage=value_stage,
    )

    return (
        metadata,
        batch,
    )


def _canonical_output_values(
    batch: _CanonicalRecurrentBatch,
    *,
    output_dim: int = 3,
    requires_grad: bool = False,
) -> torch.Tensor:
    values = torch.zeros(
        batch.nonempty_node_count,
        batch.sequence_length,
        output_dim,
        dtype=batch.dtype,
        device=batch.device,
    )

    for nonempty_position, source_node_index in enumerate(
        batch.nonempty_node_indices.tolist()
    ):
        length = int(
            batch.lengths[
                nonempty_position
            ].item()
        )

        for temporal_position in range(
            length
        ):
            base = (
                1000.0
                * float(
                    source_node_index + 1
                )
                + 10.0
                * float(
                    temporal_position + 1
                )
            )
            values[
                nonempty_position,
                temporal_position,
            ] = torch.arange(
                base,
                base + output_dim,
                dtype=batch.dtype,
                device=batch.device,
            )

    if requires_grad:
        values.requires_grad_()

    return values


def _execution_output_values(
    batch: _CanonicalRecurrentBatch,
    metadata: RecurrentExecutionMetadata,
    *,
    output_dim: int = 3,
    requires_grad: bool = False,
) -> torch.Tensor:
    canonical = _canonical_output_values(
        batch,
        output_dim=output_dim,
    )
    execution = canonical.index_select(
        0,
        metadata.sorted_to_nonempty,
    )

    if requires_grad:
        execution = (
            execution
            .detach()
            .clone()
            .requires_grad_()
        )

    return execution


def _manual_full_source_output(
    canonical_nonempty: torch.Tensor,
    history: HistoricalSequenceInputs,
    metadata: RecurrentExecutionMetadata,
) -> torch.Tensor:
    full = canonical_nonempty.new_zeros(
        (
            history.node_count,
            history.sequence_length,
            int(
                canonical_nonempty.shape[-1]
            ),
        )
    )

    for nonempty_position, source_node_index in enumerate(
        metadata.nonempty_node_indices.tolist()
    ):
        valid_positions = torch.nonzero(
            history.timestep_mask[
                source_node_index
            ],
            as_tuple=False,
        ).flatten()
        length = int(
            valid_positions.numel()
        )

        if length > 0:
            full[
                source_node_index,
                valid_positions,
            ] = canonical_nonempty[
                nonempty_position,
                :length,
            ]

    return full


def _execution_flat_state(
    metadata: RecurrentExecutionMetadata,
    layout: RecurrentStateLayout,
    *,
    offset: float = 0.0,
    requires_grad: bool = False,
) -> torch.Tensor:
    state = torch.empty(
        layout.flat_layer_direction_size,
        metadata.nonempty_node_count,
        layout.hidden_dim,
        dtype=torch.float64,
        device=metadata.device,
    )

    for flat_index in range(
        layout.flat_layer_direction_size
    ):
        for execution_position, source_node_index in enumerate(
            metadata.sorted_node_indices.tolist()
        ):
            base = (
                offset
                + 10000.0
                * float(
                    flat_index + 1
                )
                + 100.0
                * float(
                    source_node_index + 1
                )
            )
            state[
                flat_index,
                execution_position,
            ] = torch.arange(
                base,
                base + layout.hidden_dim,
                dtype=state.dtype,
                device=state.device,
            )

    if requires_grad:
        state.requires_grad_()

    return state


def _expected_full_state(
    metadata: RecurrentExecutionMetadata,
    layout: RecurrentStateLayout,
    *,
    offset: float = 0.0,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    expected = torch.zeros(
        layout.num_layers,
        layout.num_directions,
        metadata.source_node_count,
        layout.hidden_dim,
        dtype=dtype,
        device=metadata.device,
    )

    for layer_index in range(
        layout.num_layers
    ):
        for direction_index in range(
            layout.num_directions
        ):
            flat_index = layout.flat_index(
                layer_index=layer_index,
                direction_index=direction_index,
            )

            for source_node_index in (
                metadata.nonempty_node_indices.tolist()
            ):
                base = (
                    offset
                    + 10000.0
                    * float(
                        flat_index + 1
                    )
                    + 100.0
                    * float(
                        source_node_index + 1
                    )
                )
                expected[
                    layer_index,
                    direction_index,
                    source_node_index,
                ] = torch.arange(
                    base,
                    base + layout.hidden_dim,
                    dtype=dtype,
                    device=metadata.device,
                )

    return expected


def _kernel(
    config: RecurrentSequenceEncoderConfig,
    *,
    dtype: torch.dtype,
    device: torch.device | str = "cpu",
) -> nn.GRU | nn.LSTM:
    kernel_type = (
        nn.GRU
        if config.cell_kind == RecurrentCellKind.GRU
        else nn.LSTM
    )

    return kernel_type(
        input_size=config.input_dim,
        hidden_size=config.hidden_dim,
        num_layers=config.num_layers,
        bias=config.use_bias,
        batch_first=True,
        dropout=config.dropout,
        bidirectional=config.bidirectional,
    ).to(
        device=device,
        dtype=dtype,
    )


def _run_packed_kernel(
    history: HistoricalSequenceInputs,
    kernel: nn.GRU | nn.LSTM,
    config: RecurrentSequenceEncoderConfig,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
]:
    metadata = build_recurrent_execution_metadata(
        history,
        config,
    )
    canonical = canonicalize_recurrent_history(
        history,
        metadata,
    )
    packed = pack_canonical_recurrent_batch(
        canonical,
        metadata,
    )

    if config.cell_kind == RecurrentCellKind.GRU:
        initial = build_zero_gru_initial_state(
            kernel,
            config,
            batch_size=metadata.nonempty_node_count,
        )
        packed_output, hidden = kernel(
            packed,
            initial,
        )
        cell = None
    else:
        initial = build_zero_lstm_initial_state(
            kernel,
            config,
            batch_size=metadata.nonempty_node_count,
        )
        packed_output, (
            hidden,
            cell,
        ) = kernel(
            packed,
            initial,
        )

    unpacked = unpack_recurrent_sequence(
        packed_output,
        metadata,
        total_length=history.sequence_length,
    )
    full_output = restore_recurrent_sequence_to_source(
        unpacked,
        history,
        metadata,
    )
    layout = build_recurrent_state_layout(
        config
    )
    full_hidden, full_cell = (
        restore_and_scatter_recurrent_states(
            hidden,
            cell,
            metadata,
            layout,
        )
    )

    return (
        full_output,
        full_hidden,
        full_cell,
    )


def _run_reference_kernel(
    history: HistoricalSequenceInputs,
    kernel: nn.GRU | nn.LSTM,
    config: RecurrentSequenceEncoderConfig,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
]:
    metadata = build_recurrent_execution_metadata(
        history,
        config,
    )
    canonical = canonicalize_recurrent_history(
        history,
        metadata,
    )

    restored_rows: list[torch.Tensor] = []
    hidden_rows: list[torch.Tensor] = []
    cell_rows: list[torch.Tensor] = []

    for nonempty_position, source_node_index in enumerate(
        metadata.nonempty_node_indices.tolist()
    ):
        node_sequence = gather_canonical_node_sequence(
            canonical,
            nonempty_position,
        )

        if config.cell_kind == RecurrentCellKind.GRU:
            initial = build_zero_gru_initial_state(
                kernel,
                config,
                batch_size=1,
            )
            output, hidden = kernel(
                node_sequence,
                initial,
            )
        else:
            initial = build_zero_lstm_initial_state(
                kernel,
                config,
                batch_size=1,
            )
            output, (
                hidden,
                cell,
            ) = kernel(
                node_sequence,
                initial,
            )
            cell_rows.append(
                cell
            )

        restored_rows.append(
            restore_single_node_sequence(
                output,
                history,
                source_node_index,
            )
        )
        hidden_rows.append(
            hidden
        )

    nonempty_output = torch.cat(
        restored_rows,
        dim=0,
    )
    full_output = scatter_nonempty_sequence_to_source(
        nonempty_output,
        metadata,
    )
    execution_hidden = torch.cat(
        hidden_rows,
        dim=1,
    )
    execution_cell = (
        torch.cat(
            cell_rows,
            dim=1,
        )
        if cell_rows
        else None
    )
    layout = build_recurrent_state_layout(
        config
    )
    full_hidden, full_cell = (
        restore_and_scatter_recurrent_states(
            execution_hidden,
            execution_cell,
            metadata,
            layout,
        )
    )

    return (
        full_output,
        full_hidden,
        full_cell,
    )


# =============================================================================
# Module identity, aliases, exports, and boundaries
# =============================================================================


@pytest.mark.parametrize(
    "value",
    (
        RECURRENT_SEQUENCE_PACKING_IMPLEMENTATION_VERSION,
        RECURRENT_SEQUENCE_PACKING_COMPONENT_NAME,
        RECURRENT_SEQUENCE_PACKING_COMPONENT_KIND,
        RECURRENT_SEQUENCE_PACKING_CANONICALIZATION_VERSION,
        RECURRENT_SEQUENCE_PACKING_SORT_POLICY,
        RECURRENT_SEQUENCE_PACKING_PACK_POLICY,
        RECURRENT_SEQUENCE_PACKING_RESTORE_POLICY,
        RECURRENT_SEQUENCE_PACKING_STATE_POLICY,
        RECURRENT_SEQUENCE_PACKING_SCIENTIFIC_INTERPRETATION,
    ),
)
def test_execution_policy_strings_are_nonempty(
    value: str,
) -> None:
    assert isinstance(
        value,
        str,
    )
    assert value.strip()


def test_padding_value_is_exact_zero() -> None:
    assert RECURRENT_SEQUENCE_PACKING_PADDING_VALUE == 0.0


def test_sequence_packing_aliases_are_exact() -> None:
    assert canonicalize_history is canonicalize_recurrent_history
    assert order_batch_for_execution is (
        order_canonical_recurrent_batch_for_execution
    )
    assert pack_recurrent_batch is pack_canonical_recurrent_batch
    assert unpack_recurrent_output is unpack_recurrent_sequence
    assert restore_sequence_to_source is (
        restore_recurrent_sequence_to_source
    )
    assert restore_state_to_source is (
        restore_and_scatter_recurrent_state
    )


def test_sequence_packing_all_is_unique_and_resolves() -> None:
    exported = sequence_packing_module.__all__

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
            sequence_packing_module,
            name,
        )


def test_execution_ordered_batch_remains_package_private() -> None:
    assert "_ExecutionOrderedRecurrentBatch" not in (
        sequence_packing_module.__all__
    )


def test_sequence_packing_has_no_encoder_or_provenance_dependencies() -> None:
    source = inspect.getsource(
        sequence_packing_module
    )

    forbidden = (
        "gru_encoder",
        "lstm_encoder",
        "recurrent_memory_encoder import",
        "_provenance import",
        "diagnostics import",
        "input_adapter import",
        "initial_state import",
    )

    for fragment in forbidden:
        assert fragment not in source


# =============================================================================
# Canonicalization
# =============================================================================


@pytest.mark.parametrize(
    "padding_direction",
    (
        TemporalPaddingDirection.RIGHT,
        TemporalPaddingDirection.LEFT,
    ),
)
def test_canonicalization_builds_right_padded_valid_prefixes(
    padding_direction: TemporalPaddingDirection,
) -> None:
    history = _history(
        padding_direction=padding_direction
    )
    metadata, batch = _canonical(
        history
    )

    assert batch.values.shape == (
        metadata.nonempty_node_count,
        T,
        D,
    )
    assert batch.lengths.tolist() == [
        2,
        4,
        1,
        3,
        2,
    ]
    assert batch.nonempty_node_indices.tolist() == [
        0,
        2,
        3,
        4,
        5,
    ]
    assert batch.timestep_mask.tolist() == [
        [True, True, False, False],
        [True, True, True, True],
        [True, False, False, False],
        [True, True, True, False],
        [True, True, False, False],
    ]
    assert batch.original_padding_direction == padding_direction
    assert batch.canonical_padding_direction == (
        TemporalPaddingDirection.RIGHT
    )
    assert torch.count_nonzero(
        batch.values[
            ~batch.timestep_mask
        ]
    ).item() == 0


def test_left_and_right_sources_canonicalize_identically() -> None:
    right = _history(
        padding_direction=TemporalPaddingDirection.RIGHT
    )
    left = _history(
        padding_direction=TemporalPaddingDirection.LEFT
    )
    right_metadata, right_batch = _canonical(
        right
    )
    left_metadata, left_batch = _canonical(
        left
    )

    assert torch.equal(
        right_batch.values,
        left_batch.values,
    )
    assert torch.equal(
        right_batch.timestep_mask,
        left_batch.timestep_mask,
    )
    assert torch.equal(
        right_metadata.history_lengths,
        left_metadata.history_lengths,
    )


def test_no_padding_source_is_preserved() -> None:
    history = _no_padding_history()
    metadata, batch = _canonical(
        history
    )

    assert metadata.zero_history_count == 0
    assert torch.equal(
        batch.values,
        history.history,
    )
    assert bool(
        batch.timestep_mask.all().item()
    )
    assert batch.original_padding_direction == (
        TemporalPaddingDirection.NONE
    )


def test_canonicalization_preserves_valid_values_in_chronological_order() -> None:
    history = _history(
        padding_direction=TemporalPaddingDirection.LEFT
    )
    _, batch = _canonical(
        history
    )

    for nonempty_position, source_node_index in enumerate(
        batch.nonempty_node_indices.tolist()
    ):
        source_valid = history.history[
            source_node_index
        ][
            history.timestep_mask[
                source_node_index
            ]
        ]
        length = int(
            batch.lengths[
                nonempty_position
            ].item()
        )

        assert torch.equal(
            batch.values[
                nonempty_position,
                :length,
            ],
            source_valid,
        )


def test_canonicalization_preserves_custom_value_stage() -> None:
    history = _history()
    metadata = _metadata(
        history
    )
    batch = canonicalize_recurrent_history(
        history,
        metadata,
        value_stage="post_external_feature_transform",
    )

    assert batch.value_stage == "post_external_feature_transform"


def test_all_zero_history_canonicalizes_to_empty_nonempty_axis() -> None:
    history = _history(
        lengths=(
            0,
            0,
            0,
        )
    )
    metadata = _metadata(
        history
    )
    batch = canonicalize_recurrent_history(
        history,
        metadata,
    )

    assert metadata.all_zero_history
    assert batch.values.shape == (
        0,
        T,
        D,
    )
    assert batch.timestep_mask.shape == (
        0,
        T,
    )
    assert batch.lengths.numel() == 0
    assert batch.nonempty_node_indices.numel() == 0


def test_canonicalization_source_padding_has_zero_gradient() -> None:
    history = _history(
        padding_direction=TemporalPaddingDirection.LEFT,
        requires_grad=True,
    )
    metadata = _metadata(
        history
    )
    batch = canonicalize_recurrent_history(
        history,
        metadata,
    )
    weights = torch.arange(
        1,
        batch.values.numel() + 1,
        dtype=batch.dtype,
        device=batch.device,
    ).reshape_as(
        batch.values
    )
    (
        batch.values
        * weights
    ).sum().backward()

    assert history.history.grad is not None
    assert torch.count_nonzero(
        history.history.grad[
            ~history.timestep_mask
        ]
    ).item() == 0
    assert torch.count_nonzero(
        history.history.grad[
            history.timestep_mask
        ]
    ).item() > 0


def test_canonicalization_rejects_wrong_source_type() -> None:
    metadata = _metadata(
        _history()
    )

    with pytest.raises(TypeError):
        canonicalize_recurrent_history(
            object(),  # type: ignore[arg-type]
            metadata,
        )


def test_canonicalization_rejects_wrong_metadata_type() -> None:
    with pytest.raises(TypeError):
        canonicalize_recurrent_history(
            _history(),
            object(),  # type: ignore[arg-type]
        )


def test_canonicalization_rejects_blank_value_stage() -> None:
    history = _history()
    metadata = _metadata(
        history
    )

    with pytest.raises(ValueError):
        canonicalize_recurrent_history(
            history,
            metadata,
            value_stage="",
        )


def test_canonicalization_rejects_metadata_length_mismatch() -> None:
    history = _history()
    wrong_metadata = build_recurrent_execution_metadata_from_lengths(
        torch.tensor(
            [1, 0, 4, 1, 3, 2],
            dtype=torch.long,
        ),
        source_sequence_length=T,
        zero_length_policy="allow_zero_history",
        original_padding_direction="right",
        execution_path="packed",
    )

    with pytest.raises(ValueError):
        canonicalize_recurrent_history(
            history,
            wrong_metadata,
        )


def test_canonicalization_rejects_metadata_padding_direction_mismatch() -> None:
    history = _history(
        padding_direction=TemporalPaddingDirection.RIGHT
    )
    metadata = _metadata(
        history
    ).replace(
        original_padding_direction=(
            TemporalPaddingDirection.LEFT
        )
    )

    with pytest.raises(ValueError):
        canonicalize_recurrent_history(
            history,
            metadata,
        )


# =============================================================================
# Single-node gathering helpers
# =============================================================================


@pytest.mark.parametrize(
    "padding_direction",
    (
        TemporalPaddingDirection.RIGHT,
        TemporalPaddingDirection.LEFT,
    ),
)
@pytest.mark.parametrize(
    (
        "node_index",
        "expected_length",
    ),
    (
        (
            0,
            2,
        ),
        (
            1,
            0,
        ),
        (
            2,
            4,
        ),
        (
            3,
            1,
        ),
    ),
)
def test_gather_single_node_valid_sequence(
    padding_direction: TemporalPaddingDirection,
    node_index: int,
    expected_length: int,
) -> None:
    history = _history(
        padding_direction=padding_direction
    )

    gathered = gather_single_node_valid_sequence(
        history,
        node_index,
    )

    assert gathered.shape == (
        1,
        expected_length,
        D,
    )
    assert torch.equal(
        gathered.squeeze(
            0
        ),
        history.history[
            node_index
        ][
            history.timestep_mask[
                node_index
            ]
        ],
    )


def test_single_node_gather_left_right_equivalence() -> None:
    left = _history(
        padding_direction=TemporalPaddingDirection.LEFT
    )
    right = _history(
        padding_direction=TemporalPaddingDirection.RIGHT
    )

    for node_index in range(
        left.node_count
    ):
        assert torch.equal(
            gather_single_node_valid_sequence(
                left,
                node_index,
            ),
            gather_single_node_valid_sequence(
                right,
                node_index,
            ),
        )


@pytest.mark.parametrize(
    (
        "node_index",
        "error_type",
    ),
    (
        (
            -1,
            ValueError,
        ),
        (
            6,
            IndexError,
        ),
        (
            True,
            TypeError,
        ),
    ),
)
def test_gather_single_node_rejects_invalid_index(
    node_index: int,
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type):
        gather_single_node_valid_sequence(
            _history(),
            node_index,
        )


def test_gather_single_node_rejects_wrong_source_type() -> None:
    with pytest.raises(TypeError):
        gather_single_node_valid_sequence(
            object(),  # type: ignore[arg-type]
            0,
        )


@pytest.mark.parametrize(
    (
        "nonempty_position",
        "expected_length",
    ),
    (
        (
            0,
            2,
        ),
        (
            1,
            4,
        ),
        (
            2,
            1,
        ),
        (
            4,
            2,
        ),
    ),
)
def test_gather_canonical_node_sequence(
    nonempty_position: int,
    expected_length: int,
) -> None:
    _, batch = _canonical(
        _history()
    )

    gathered = gather_canonical_node_sequence(
        batch,
        nonempty_position,
    )

    assert gathered.shape == (
        1,
        expected_length,
        D,
    )
    assert torch.equal(
        gathered,
        batch.values[
            nonempty_position : nonempty_position + 1,
            :expected_length,
        ],
    )


@pytest.mark.parametrize(
    (
        "position",
        "error_type",
    ),
    (
        (
            -1,
            ValueError,
        ),
        (
            5,
            IndexError,
        ),
        (
            True,
            TypeError,
        ),
    ),
)
def test_gather_canonical_node_rejects_invalid_position(
    position: int,
    error_type: type[Exception],
) -> None:
    _, batch = _canonical(
        _history()
    )

    with pytest.raises(error_type):
        gather_canonical_node_sequence(
            batch,
            position,
        )


# =============================================================================
# Execution ordering
# =============================================================================


def test_packed_order_matches_metadata() -> None:
    history = _history()
    metadata, batch = _canonical(
        history
    )

    ordered = sort_canonical_recurrent_batch(
        batch,
        metadata,
    )

    assert ordered.lengths.tolist() == [
        4,
        3,
        2,
        2,
        1,
    ]
    assert ordered.source_node_indices.tolist() == [
        2,
        4,
        0,
        5,
        3,
    ]
    assert torch.equal(
        ordered.values,
        batch.values.index_select(
            0,
            metadata.sorted_to_nonempty,
        ),
    )
    assert torch.equal(
        ordered.timestep_mask,
        batch.timestep_mask.index_select(
            0,
            metadata.sorted_to_nonempty,
        ),
    )


def test_reference_order_is_identity() -> None:
    history = _history()
    metadata, batch = _canonical(
        history,
        pack_sequences=False,
    )

    ordered = order_canonical_recurrent_batch_for_execution(
        batch,
        metadata,
    )

    assert metadata.execution_path == (
        RecurrentExecutionPath.REFERENCE
    )
    assert torch.equal(
        ordered.values,
        batch.values,
    )
    assert ordered.source_node_indices.tolist() == [
        0,
        2,
        3,
        4,
        5,
    ]


def test_equal_length_ties_preserve_original_nonempty_order() -> None:
    history = _history(
        lengths=(
            2,
            4,
            2,
            1,
            2,
        )
    )
    metadata, batch = _canonical(
        history
    )
    ordered = sort_canonical_recurrent_batch(
        batch,
        metadata,
    )

    assert ordered.lengths.tolist() == [
        4,
        2,
        2,
        2,
        1,
    ]
    assert ordered.source_node_indices.tolist() == [
        1,
        0,
        2,
        4,
        3,
    ]


def test_ordering_preserves_value_stage() -> None:
    history = _history()
    metadata, batch = _canonical(
        history,
        value_stage="adapted_features",
    )
    ordered = order_canonical_recurrent_batch_for_execution(
        batch,
        metadata,
    )

    assert ordered.value_stage == "adapted_features"


def test_sort_wrapper_rejects_reference_metadata() -> None:
    history = _history()
    metadata, batch = _canonical(
        history,
        pack_sequences=False,
    )

    with pytest.raises(ValueError):
        sort_canonical_recurrent_batch(
            batch,
            metadata,
        )


def test_ordering_rejects_batch_metadata_mismatch() -> None:
    history = _history()
    metadata, batch = _canonical(
        history
    )
    other_history = _history(
        lengths=(
            4,
            3,
            2,
            1,
            1,
            1,
        )
    )
    other_metadata = _metadata(
        other_history
    )

    with pytest.raises(ValueError):
        order_canonical_recurrent_batch_for_execution(
            batch,
            other_metadata,
        )

    assert metadata.source_node_count == (
        other_metadata.source_node_count
    )


# =============================================================================
# Packing and unpacking
# =============================================================================


@pytest.mark.parametrize(
    "dtype",
    (
        torch.float32,
        torch.float64,
    ),
)
def test_pack_returns_packed_sequence_with_expected_data(
    dtype: torch.dtype,
) -> None:
    history = _history(
        dtype=dtype
    )
    metadata, batch = _canonical(
        history
    )
    ordered = sort_canonical_recurrent_batch(
        batch,
        metadata,
    )

    packed = pack_canonical_recurrent_batch(
        batch,
        metadata,
    )

    assert isinstance(
        packed,
        PackedSequence,
    )
    assert packed.data.dtype == dtype
    assert packed.data.device == history.device
    assert packed.sorted_indices is None
    assert packed.unsorted_indices is None
    assert torch.isfinite(
        packed.data
    ).all()

    unpacked = unpack_recurrent_sequence(
        packed,
        metadata,
        total_length=T,
    )

    assert torch.equal(
        unpacked,
        ordered.values,
    )


def test_pack_uses_sorted_metadata_even_when_sort_is_identity() -> None:
    history = _history(
        lengths=(
            4,
            3,
            3,
            1,
        )
    )
    metadata, batch = _canonical(
        history
    )

    assert metadata.identity_permutation
    assert not metadata.sort_was_applied

    packed = pack_canonical_recurrent_batch(
        batch,
        metadata,
    )
    unpacked = unpack_recurrent_sequence(
        packed,
        metadata,
        total_length=T,
    )

    assert torch.equal(
        unpacked,
        batch.values,
    )


def test_pack_rejects_reference_metadata() -> None:
    history = _history()
    metadata, batch = _canonical(
        history,
        pack_sequences=False,
    )

    with pytest.raises(ValueError):
        pack_canonical_recurrent_batch(
            batch,
            metadata,
        )


def test_pack_rejects_all_zero_history() -> None:
    history = _history(
        lengths=(
            0,
            0,
            0,
        )
    )
    metadata, batch = _canonical(
        history
    )

    with pytest.raises(ValueError):
        pack_canonical_recurrent_batch(
            batch,
            metadata,
        )


def test_unpack_rejects_non_packed_input() -> None:
    metadata = _metadata(
        _history()
    )

    with pytest.raises(TypeError):
        unpack_recurrent_sequence(
            torch.zeros(1),  # type: ignore[arg-type]
            metadata,
            total_length=T,
        )


def test_unpack_rejects_reference_metadata() -> None:
    history = _history()
    reference_metadata, _ = _canonical(
        history,
        pack_sequences=False,
    )
    packed_metadata, batch = _canonical(
        history
    )
    packed = pack_canonical_recurrent_batch(
        batch,
        packed_metadata,
    )

    with pytest.raises(ValueError):
        unpack_recurrent_sequence(
            packed,
            reference_metadata,
            total_length=T,
        )


@pytest.mark.parametrize(
    "total_length",
    (
        0,
        -1,
        True,
    ),
)
def test_unpack_rejects_invalid_total_length(
    total_length: int,
) -> None:
    history = _history()
    metadata, batch = _canonical(
        history
    )
    packed = pack_canonical_recurrent_batch(
        batch,
        metadata,
    )

    with pytest.raises(
        (
            TypeError,
            ValueError,
        )
    ):
        unpack_recurrent_sequence(
            packed,
            metadata,
            total_length=total_length,
        )


def test_unpack_rejects_nonfinite_packed_data() -> None:
    history = _history()
    metadata, batch = _canonical(
        history
    )
    packed = pack_canonical_recurrent_batch(
        batch,
        metadata,
    )
    corrupted_data = packed.data.clone()
    corrupted_data[0, 0] = float(
        "nan"
    )
    corrupted = PackedSequence(
        corrupted_data,
        packed.batch_sizes,
        packed.sorted_indices,
        packed.unsorted_indices,
    )

    with pytest.raises(ValueError):
        unpack_recurrent_sequence(
            corrupted,
            metadata,
            total_length=T,
        )


# =============================================================================
# Sequence restoration
# =============================================================================


def test_restore_nonempty_sequence_order_uses_inverse_permutation() -> None:
    history = _history()
    metadata, batch = _canonical(
        history
    )
    canonical = _canonical_output_values(
        batch
    )
    execution = canonical.index_select(
        0,
        metadata.sorted_to_nonempty,
    )

    restored = restore_nonempty_sequence_order(
        execution,
        metadata,
    )

    assert torch.equal(
        restored,
        canonical,
    )


def test_reference_restore_nonempty_order_is_identity() -> None:
    history = _history()
    metadata, batch = _canonical(
        history,
        pack_sequences=False,
    )
    canonical = _canonical_output_values(
        batch
    )

    restored = restore_nonempty_sequence_order(
        canonical,
        metadata,
    )

    assert torch.equal(
        restored,
        canonical,
    )


@pytest.mark.parametrize(
    "padding_direction",
    (
        TemporalPaddingDirection.RIGHT,
        TemporalPaddingDirection.LEFT,
        TemporalPaddingDirection.NONE,
    ),
)
def test_restore_source_temporal_layout(
    padding_direction: TemporalPaddingDirection,
) -> None:
    history = (
        _no_padding_history()
        if padding_direction == TemporalPaddingDirection.NONE
        else _history(
            padding_direction=padding_direction
        )
    )
    metadata, batch = _canonical(
        history
    )
    canonical = _canonical_output_values(
        batch
    )

    restored = restore_source_temporal_layout(
        canonical,
        history,
        metadata,
    )
    expected = _manual_full_source_output(
        canonical,
        history,
        metadata,
    ).index_select(
        0,
        metadata.nonempty_node_indices,
    )

    assert torch.equal(
        restored,
        expected,
    )

    source_mask = history.timestep_mask.index_select(
        0,
        metadata.nonempty_node_indices,
    )
    assert torch.count_nonzero(
        restored[
            ~source_mask
        ]
    ).item() == 0


def test_scatter_nonempty_sequence_inserts_exact_zero_rows() -> None:
    history = _history()
    metadata, batch = _canonical(
        history
    )
    canonical = _canonical_output_values(
        batch
    )
    source_layout = restore_source_temporal_layout(
        canonical,
        history,
        metadata,
    )

    full = scatter_nonempty_sequence_to_source(
        source_layout,
        metadata,
    )

    assert full.shape == (
        history.node_count,
        T,
        3,
    )
    assert torch.equal(
        full.index_select(
            0,
            metadata.nonempty_node_indices,
        ),
        source_layout,
    )
    assert torch.count_nonzero(
        full.index_select(
            0,
            metadata.zero_history_node_indices,
        )
    ).item() == 0


@pytest.mark.parametrize(
    "padding_direction",
    (
        TemporalPaddingDirection.RIGHT,
        TemporalPaddingDirection.LEFT,
    ),
)
@pytest.mark.parametrize(
    "pack_sequences",
    (
        True,
        False,
    ),
)
def test_complete_sequence_restoration_matches_manual_oracle(
    padding_direction: TemporalPaddingDirection,
    pack_sequences: bool,
) -> None:
    history = _history(
        padding_direction=padding_direction
    )
    metadata, batch = _canonical(
        history,
        pack_sequences=pack_sequences,
    )
    canonical = _canonical_output_values(
        batch,
        output_dim=5,
    )
    execution = canonical.index_select(
        0,
        metadata.sorted_to_nonempty,
    )

    restored = restore_recurrent_sequence_to_source(
        execution,
        history,
        metadata,
    )
    expected = _manual_full_source_output(
        canonical,
        history,
        metadata,
    )

    assert torch.equal(
        restored,
        expected,
    )
    assert torch.count_nonzero(
        restored[
            ~history.timestep_mask
        ]
    ).item() == 0


def test_complete_sequence_restoration_preserves_gradients() -> None:
    history = _history(
        padding_direction=TemporalPaddingDirection.LEFT
    )
    metadata, batch = _canonical(
        history
    )
    execution = _execution_output_values(
        batch,
        metadata,
        output_dim=4,
        requires_grad=True,
    )

    full = restore_recurrent_sequence_to_source(
        execution,
        history,
        metadata,
    )
    weights = torch.arange(
        1,
        full.numel() + 1,
        dtype=full.dtype,
    ).reshape_as(
        full
    )
    (
        full
        * weights
    ).sum().backward()

    assert execution.grad is not None
    assert torch.count_nonzero(
        execution.grad
    ).item() > 0
    assert torch.isfinite(
        execution.grad
    ).all()


def test_all_zero_nonempty_scatter_produces_full_zero_output() -> None:
    history = _history(
        lengths=(
            0,
            0,
            0,
        )
    )
    metadata, _ = _canonical(
        history
    )
    empty = torch.empty(
        0,
        T,
        4,
        dtype=history.dtype,
    )

    full = scatter_nonempty_sequence_to_source(
        empty,
        metadata,
    )

    assert full.shape == (
        3,
        T,
        4,
    )
    assert torch.count_nonzero(
        full
    ).item() == 0


def test_restore_nonempty_sequence_rejects_nonfinite_values() -> None:
    metadata = _metadata(
        _history()
    )
    values = torch.zeros(
        metadata.nonempty_node_count,
        T,
        3,
        dtype=torch.float64,
    )
    values[0, 0, 0] = float(
        "nan"
    )

    with pytest.raises(ValueError):
        restore_nonempty_sequence_order(
            values,
            metadata,
        )


def test_restore_source_layout_rejects_nonzero_canonical_padding() -> None:
    history = _history()
    metadata, batch = _canonical(
        history
    )
    canonical = _canonical_output_values(
        batch
    )
    canonical[0, 3, 0] = 1.0

    with pytest.raises(ValueError):
        restore_source_temporal_layout(
            canonical,
            history,
            metadata,
        )


# =============================================================================
# Single-node temporal restoration
# =============================================================================


@pytest.mark.parametrize(
    "padding_direction",
    (
        TemporalPaddingDirection.RIGHT,
        TemporalPaddingDirection.LEFT,
    ),
)
@pytest.mark.parametrize(
    (
        "node_index",
        "length",
    ),
    (
        (
            0,
            2,
        ),
        (
            1,
            0,
        ),
        (
            2,
            4,
        ),
        (
            3,
            1,
        ),
    ),
)
def test_restore_single_node_sequence(
    padding_direction: TemporalPaddingDirection,
    node_index: int,
    length: int,
) -> None:
    history = _history(
        padding_direction=padding_direction
    )
    valid = torch.arange(
        length * 3,
        dtype=history.dtype,
    ).reshape(
        1,
        length,
        3,
    )

    restored = restore_single_node_sequence(
        valid,
        history,
        node_index,
    )

    assert restored.shape == (
        1,
        T,
        3,
    )
    mask = history.timestep_mask[
        node_index : node_index + 1
    ]
    assert torch.equal(
        restored[
            mask
        ],
        valid.reshape(
            length,
            3,
        ),
    )
    assert torch.count_nonzero(
        restored[
            ~mask
        ]
    ).item() == 0


def test_restore_single_node_preserves_gradient() -> None:
    history = _history(
        padding_direction=TemporalPaddingDirection.LEFT
    )
    valid = torch.arange(
        6,
        dtype=history.dtype,
    ).reshape(
        1,
        2,
        3,
    ).requires_grad_()

    restored = restore_single_node_sequence(
        valid,
        history,
        0,
    )
    restored.sum().backward()

    assert valid.grad is not None
    assert torch.equal(
        valid.grad,
        torch.ones_like(
            valid
        ),
    )


@pytest.mark.parametrize(
    (
        "valid_sequence",
        "node_index",
        "error_type",
    ),
    (
        (
            torch.zeros(2, 2, 3, dtype=torch.float64),
            0,
            ValueError,
        ),
        (
            torch.zeros(1, 3, 3, dtype=torch.float64),
            0,
            ValueError,
        ),
        (
            torch.zeros(1, 2, 0, dtype=torch.float64),
            0,
            ValueError,
        ),
        (
            torch.zeros(1, 2, 3, dtype=torch.long),
            0,
            ValueError,
        ),
        (
            torch.zeros(1, 2, 3, dtype=torch.float64),
            6,
            IndexError,
        ),
    ),
)
def test_restore_single_node_rejects_invalid_inputs(
    valid_sequence: torch.Tensor,
    node_index: int,
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type):
        restore_single_node_sequence(
            valid_sequence,
            _history(),
            node_index,
        )


# =============================================================================
# Recurrent-state restoration
# =============================================================================


@pytest.mark.parametrize(
    (
        "num_layers",
        "num_directions",
        "hidden_dim",
    ),
    (
        (
            1,
            1,
            2,
        ),
        (
            1,
            2,
            3,
        ),
        (
            3,
            1,
            4,
        ),
        (
            2,
            2,
            5,
        ),
    ),
)
def test_restore_flat_state_order(
    num_layers: int,
    num_directions: int,
    hidden_dim: int,
) -> None:
    metadata = _metadata(
        _history()
    )
    layout = RecurrentStateLayout(
        num_layers,
        num_directions,
        hidden_dim,
    )
    execution = _execution_flat_state(
        metadata,
        layout,
    )

    restored = restore_nonempty_flat_state_order(
        execution,
        metadata,
        layout,
    )

    for nonempty_position, source_node_index in enumerate(
        metadata.nonempty_node_indices.tolist()
    ):
        execution_position = int(
            metadata.nonempty_to_sorted[
                nonempty_position
            ].item()
        )

        assert torch.equal(
            restored[
                :,
                nonempty_position,
            ],
            execution[
                :,
                execution_position,
            ],
        )


def test_scatter_canonical_state_to_full_node_axis() -> None:
    metadata = _metadata(
        _history()
    )
    layout = RecurrentStateLayout(
        2,
        2,
        3,
    )
    execution = _execution_flat_state(
        metadata,
        layout,
    )
    nonempty_flat = restore_nonempty_flat_state_order(
        execution,
        metadata,
        layout,
    )
    nonempty_canonical = layout.unflatten_state(
        nonempty_flat
    )

    full = scatter_nonempty_canonical_state_to_source(
        nonempty_canonical,
        metadata,
        layout,
    )
    expected = _expected_full_state(
        metadata,
        layout,
    )

    assert torch.equal(
        full,
        expected,
    )
    assert torch.count_nonzero(
        full.index_select(
            2,
            metadata.zero_history_node_indices,
        )
    ).item() == 0


def test_complete_state_restoration_matches_oracle() -> None:
    metadata = _metadata(
        _history()
    )
    layout = RecurrentStateLayout(
        3,
        2,
        4,
    )
    execution = _execution_flat_state(
        metadata,
        layout,
    )

    full = restore_and_scatter_recurrent_state(
        execution,
        metadata,
        layout,
    )
    expected = _expected_full_state(
        metadata,
        layout,
    )

    assert torch.equal(
        full,
        expected,
    )
    assert full.shape == (
        3,
        2,
        metadata.source_node_count,
        4,
    )


def test_bidirectional_state_order_is_not_swapped() -> None:
    metadata = _metadata(
        _history()
    )
    layout = RecurrentStateLayout(
        1,
        2,
        2,
    )
    execution = _execution_flat_state(
        metadata,
        layout,
    )
    full = restore_and_scatter_recurrent_state(
        execution,
        metadata,
        layout,
    )
    expected = _expected_full_state(
        metadata,
        layout,
    )

    assert torch.equal(
        full[
            0,
            0,
        ],
        expected[
            0,
            0,
        ],
    )
    assert torch.equal(
        full[
            0,
            1,
        ],
        expected[
            0,
            1,
        ],
    )
    assert not torch.equal(
        full[
            0,
            0,
        ],
        full[
            0,
            1,
        ],
    )


def test_hidden_and_cell_states_restore_independently() -> None:
    metadata = _metadata(
        _history()
    )
    layout = RecurrentStateLayout(
        2,
        2,
        3,
    )
    hidden = _execution_flat_state(
        metadata,
        layout,
        offset=0.0,
    )
    cell = _execution_flat_state(
        metadata,
        layout,
        offset=1_000_000.0,
    )

    full_hidden, full_cell = (
        restore_and_scatter_lstm_states(
            hidden,
            cell,
            metadata,
            layout,
        )
    )

    assert torch.equal(
        full_hidden,
        _expected_full_state(
            metadata,
            layout,
            offset=0.0,
        ),
    )
    assert torch.equal(
        full_cell,
        _expected_full_state(
            metadata,
            layout,
            offset=1_000_000.0,
        ),
    )
    assert not torch.equal(
        full_hidden,
        full_cell,
    )


def test_optional_cell_state_wrapper_supports_gru() -> None:
    metadata = _metadata(
        _history()
    )
    layout = RecurrentStateLayout(
        2,
        1,
        3,
    )
    hidden = _execution_flat_state(
        metadata,
        layout,
    )

    full_hidden, full_cell = (
        restore_and_scatter_recurrent_states(
            hidden,
            None,
            metadata,
            layout,
        )
    )

    assert full_cell is None
    assert torch.equal(
        full_hidden,
        _expected_full_state(
            metadata,
            layout,
        ),
    )


def test_state_restoration_preserves_gradient() -> None:
    metadata = _metadata(
        _history()
    )
    layout = RecurrentStateLayout(
        2,
        2,
        3,
    )
    execution = _execution_flat_state(
        metadata,
        layout,
        requires_grad=True,
    )

    full = restore_and_scatter_recurrent_state(
        execution,
        metadata,
        layout,
    )
    weights = torch.arange(
        1,
        full.numel() + 1,
        dtype=full.dtype,
    ).reshape_as(
        full
    )
    (
        full
        * weights
    ).sum().backward()

    assert execution.grad is not None
    assert torch.count_nonzero(
        execution.grad
    ).item() > 0
    assert torch.isfinite(
        execution.grad
    ).all()


def test_all_zero_state_restoration_produces_full_zeros() -> None:
    history = _history(
        lengths=(
            0,
            0,
            0,
        )
    )
    metadata = _metadata(
        history
    )
    layout = RecurrentStateLayout(
        2,
        2,
        3,
    )
    empty = torch.empty(
        4,
        0,
        3,
        dtype=torch.float64,
    )

    full = restore_and_scatter_recurrent_state(
        empty,
        metadata,
        layout,
    )

    assert full.shape == (
        2,
        2,
        3,
        3,
    )
    assert torch.count_nonzero(
        full
    ).item() == 0


def test_lstm_wrapper_rejects_missing_cell_state() -> None:
    metadata = _metadata(
        _history()
    )
    layout = RecurrentStateLayout(
        1,
        1,
        3,
    )
    hidden = _execution_flat_state(
        metadata,
        layout,
    )

    with pytest.raises(TypeError):
        restore_and_scatter_lstm_states(
            hidden,
            None,  # type: ignore[arg-type]
            metadata,
            layout,
        )


def test_state_restoration_rejects_wrong_batch_width() -> None:
    metadata = _metadata(
        _history()
    )
    layout = RecurrentStateLayout(
        2,
        2,
        3,
    )
    wrong = torch.zeros(
        4,
        metadata.nonempty_node_count - 1,
        3,
        dtype=torch.float64,
    )

    with pytest.raises(ValueError):
        restore_and_scatter_recurrent_state(
            wrong,
            metadata,
            layout,
        )


def test_state_restoration_rejects_nonfinite_state() -> None:
    metadata = _metadata(
        _history()
    )
    layout = RecurrentStateLayout(
        2,
        2,
        3,
    )
    state = _execution_flat_state(
        metadata,
        layout,
    )
    state[0, 0, 0] = float(
        "nan"
    )

    with pytest.raises(ValueError):
        restore_and_scatter_recurrent_state(
            state,
            metadata,
            layout,
        )


# =============================================================================
# Actual packed/reference numerical equivalence
# =============================================================================


@pytest.mark.parametrize(
    "cell_kind",
    (
        RecurrentCellKind.GRU,
        RecurrentCellKind.LSTM,
    ),
)
@pytest.mark.parametrize(
    "bidirectional",
    (
        False,
        True,
    ),
)
@pytest.mark.parametrize(
    "padding_direction",
    (
        TemporalPaddingDirection.RIGHT,
        TemporalPaddingDirection.LEFT,
    ),
)
def test_packed_reference_kernel_equivalence_float64(
    cell_kind: RecurrentCellKind,
    bidirectional: bool,
    padding_direction: TemporalPaddingDirection,
) -> None:
    torch.manual_seed(
        1207
    )
    history = _history(
        padding_direction=padding_direction,
        dtype=torch.float64,
    )
    packed_config = _config(
        cell_kind=cell_kind,
        pack_sequences=True,
        hidden_dim=3,
        num_layers=2,
        bidirectional=bidirectional,
    )
    reference_config = _config(
        cell_kind=cell_kind,
        pack_sequences=False,
        hidden_dim=3,
        num_layers=2,
        bidirectional=bidirectional,
    )
    kernel = _kernel(
        packed_config,
        dtype=torch.float64,
    )
    kernel.eval()

    packed_output, packed_hidden, packed_cell = (
        _run_packed_kernel(
            history,
            kernel,
            packed_config,
        )
    )
    reference_output, reference_hidden, reference_cell = (
        _run_reference_kernel(
            history,
            kernel,
            reference_config,
        )
    )

    torch.testing.assert_close(
        packed_output,
        reference_output,
        rtol=1e-7,
        atol=1e-9,
    )
    torch.testing.assert_close(
        packed_hidden,
        reference_hidden,
        rtol=1e-7,
        atol=1e-9,
    )

    if cell_kind == RecurrentCellKind.LSTM:
        assert packed_cell is not None
        assert reference_cell is not None
        torch.testing.assert_close(
            packed_cell,
            reference_cell,
            rtol=1e-7,
            atol=1e-9,
        )
    else:
        assert packed_cell is None
        assert reference_cell is None

    assert torch.count_nonzero(
        packed_output[
            ~history.timestep_mask
        ]
    ).item() == 0


@pytest.mark.parametrize(
    "cell_kind",
    (
        RecurrentCellKind.GRU,
        RecurrentCellKind.LSTM,
    ),
)
def test_packed_reference_kernel_equivalence_float32(
    cell_kind: RecurrentCellKind,
) -> None:
    torch.manual_seed(
        912
    )
    history = _history(
        dtype=torch.float32,
    )
    packed_config = _config(
        cell_kind=cell_kind,
        pack_sequences=True,
        hidden_dim=4,
        num_layers=1,
        bidirectional=True,
    )
    reference_config = _config(
        cell_kind=cell_kind,
        pack_sequences=False,
        hidden_dim=4,
        num_layers=1,
        bidirectional=True,
    )
    kernel = _kernel(
        packed_config,
        dtype=torch.float32,
    )
    kernel.eval()

    packed = _run_packed_kernel(
        history,
        kernel,
        packed_config,
    )
    reference = _run_reference_kernel(
        history,
        kernel,
        reference_config,
    )

    torch.testing.assert_close(
        packed[0],
        reference[0],
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(
        packed[1],
        reference[1],
        rtol=1e-5,
        atol=1e-6,
    )

    if cell_kind == RecurrentCellKind.LSTM:
        assert packed[2] is not None
        assert reference[2] is not None
        torch.testing.assert_close(
            packed[2],
            reference[2],
            rtol=1e-5,
            atol=1e-6,
        )


def test_packed_reference_gru_gradient_equivalence_float64() -> None:
    torch.manual_seed(
        77
    )
    packed_history = _history(
        padding_direction=TemporalPaddingDirection.LEFT,
        dtype=torch.float64,
        requires_grad=True,
    )
    reference_history = _history(
        padding_direction=TemporalPaddingDirection.LEFT,
        dtype=torch.float64,
        requires_grad=True,
    )
    packed_config = _config(
        cell_kind=RecurrentCellKind.GRU,
        pack_sequences=True,
        hidden_dim=3,
        num_layers=2,
        bidirectional=True,
    )
    reference_config = _config(
        cell_kind=RecurrentCellKind.GRU,
        pack_sequences=False,
        hidden_dim=3,
        num_layers=2,
        bidirectional=True,
    )
    packed_kernel = _kernel(
        packed_config,
        dtype=torch.float64,
    )
    reference_kernel = copy.deepcopy(
        packed_kernel
    )
    packed_kernel.eval()
    reference_kernel.eval()

    packed_output, packed_hidden, _ = (
        _run_packed_kernel(
            packed_history,
            packed_kernel,
            packed_config,
        )
    )
    reference_output, reference_hidden, _ = (
        _run_reference_kernel(
            reference_history,
            reference_kernel,
            reference_config,
        )
    )

    output_weights = torch.arange(
        1,
        packed_output.numel() + 1,
        dtype=torch.float64,
    ).reshape_as(
        packed_output
    )
    hidden_weights = torch.arange(
        1,
        packed_hidden.numel() + 1,
        dtype=torch.float64,
    ).reshape_as(
        packed_hidden
    )

    packed_loss = (
        packed_output
        * output_weights
    ).sum() + (
        packed_hidden
        * hidden_weights
    ).sum()
    reference_loss = (
        reference_output
        * output_weights
    ).sum() + (
        reference_hidden
        * hidden_weights
    ).sum()

    packed_loss.backward()
    reference_loss.backward()

    assert packed_history.history.grad is not None
    assert reference_history.history.grad is not None
    torch.testing.assert_close(
        packed_history.history.grad,
        reference_history.history.grad,
        rtol=1e-7,
        atol=1e-9,
    )
    assert torch.count_nonzero(
        packed_history.history.grad[
            ~packed_history.timestep_mask
        ]
    ).item() == 0

    packed_parameters = dict(
        packed_kernel.named_parameters()
    )
    reference_parameters = dict(
        reference_kernel.named_parameters()
    )

    assert packed_parameters.keys() == (
        reference_parameters.keys()
    )

    for name in packed_parameters:
        packed_gradient = (
            packed_parameters[
                name
            ].grad
        )
        reference_gradient = (
            reference_parameters[
                name
            ].grad
        )

        assert packed_gradient is not None
        assert reference_gradient is not None
        torch.testing.assert_close(
            packed_gradient,
            reference_gradient,
            rtol=1e-7,
            atol=1e-9,
        )


# =============================================================================
# Conditional CUDA coverage
# =============================================================================


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is not available.",
)
@pytest.mark.parametrize(
    "cell_kind",
    (
        RecurrentCellKind.GRU,
        RecurrentCellKind.LSTM,
    ),
)
def test_sequence_execution_cuda(
    cell_kind: RecurrentCellKind,
) -> None:
    torch.manual_seed(
        808
    )
    history = _history(
        dtype=torch.float32,
        device="cuda",
        padding_direction=TemporalPaddingDirection.LEFT,
    )
    packed_config = _config(
        cell_kind=cell_kind,
        pack_sequences=True,
        hidden_dim=3,
        num_layers=1,
        bidirectional=True,
    )
    reference_config = _config(
        cell_kind=cell_kind,
        pack_sequences=False,
        hidden_dim=3,
        num_layers=1,
        bidirectional=True,
    )
    kernel = _kernel(
        packed_config,
        dtype=torch.float32,
        device="cuda",
    )
    kernel.eval()

    packed = _run_packed_kernel(
        history,
        kernel,
        packed_config,
    )
    reference = _run_reference_kernel(
        history,
        kernel,
        reference_config,
    )

    assert packed[0].device.type == "cuda"
    assert packed[1].device.type == "cuda"
    assert torch.isfinite(
        packed[0]
    ).all()
    assert torch.isfinite(
        packed[1]
    ).all()
    assert torch.count_nonzero(
        packed[0][
            ~history.timestep_mask
        ]
    ).item() == 0

    torch.testing.assert_close(
        packed[0],
        reference[0],
        rtol=1e-4,
        atol=1e-5,
    )
    torch.testing.assert_close(
        packed[1],
        reference[1],
        rtol=1e-4,
        atol=1e-5,
    )

    if cell_kind == RecurrentCellKind.LSTM:
        assert packed[2] is not None
        assert reference[2] is not None
        torch.testing.assert_close(
            packed[2],
            reference[2],
            rtol=1e-4,
            atol=1e-5,
        )
