"""
Consolidated tests for Phase 6 GRU and LSTM sequence encoders.

Target repository path:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            tests/
                memory/
                    recurrent_memory_encoder/
                        test_recurrent_encoders.py

Modules under test:
    memory/recurrent_memory_encoder/gru_encoder.py
    memory/recurrent_memory_encoder/lstm_encoder.py

This suite validates the complete recurrent encoder contracts rather than the
lower-level preprocessing and sequence-packing utilities, which are covered by
their own consolidated suites.

Coverage includes:

- GRU/LSTM construction, properties, aliases, and builders;
- public ``forward`` and ``encode_with_state`` contracts;
- packed/reference equivalence;
- left/right/no-padding behavior;
- projection, LayerNorm, bias, layers, and directions;
- exact-zero padding and zero-history states;
- all-zero-history short-circuit behavior;
- explicit parameter snapshots and stale/foreign snapshot rejection;
- architecture and execution provenance;
- source-history, temporal-coordinate, and observation-mask boundaries;
- state-dict round trips;
- node-permutation equivariance;
- float32/float64 execution and gradients;
- dropout behavior in training/evaluation;
- conditional CUDA execution.
"""

from __future__ import annotations

import copy
import inspect

import pytest
import torch
from torch import nn

import urban_resilience_models.v2_hazard_conditioned_functional_ugnn.memory.recurrent_memory_encoder.gru_encoder as gru_encoder_module
import urban_resilience_models.v2_hazard_conditioned_functional_ugnn.memory.recurrent_memory_encoder.lstm_encoder as lstm_encoder_module
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.memory.config import (
    RecurrentCellKind,
    RecurrentSequenceEncoderConfig,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.memory.recurrent_memory_encoder.gru_encoder import (
    GRU_SEQUENCE_ENCODER_ALL_ZERO_POLICY,
    GRU_SEQUENCE_ENCODER_COMPONENT_KIND,
    GRU_SEQUENCE_ENCODER_COMPONENT_NAME,
    GRU_SEQUENCE_ENCODER_ENCODING_NAME,
    GRU_SEQUENCE_ENCODER_FEATURE_OBSERVATION_MASK_CONSUMED,
    GRU_SEQUENCE_ENCODER_HAZARD_CONDITIONED,
    GRU_SEQUENCE_ENCODER_IMPLEMENTATION_VERSION,
    GRU_SEQUENCE_ENCODER_INITIAL_STATE_POLICY,
    GRU_SEQUENCE_ENCODER_OPERATION_NAME,
    GRU_SEQUENCE_ENCODER_OUTPUT_DIRECTION_ORDER,
    GRU_SEQUENCE_ENCODER_REFERENCE_POLICY,
    GRU_SEQUENCE_ENCODER_RUN_NAME,
    GRU_SEQUENCE_ENCODER_SCIENTIFIC_INTERPRETATION,
    GRU_SEQUENCE_ENCODER_TEMPORAL_COORDINATES_CONSUMED,
    GRUEncoder,
    GRUSequenceEncoder,
    build_gru_encoder,
    build_gru_parameter_snapshot,
    build_gru_sequence_encoder,
    snapshot_gru_parameters,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.memory.recurrent_memory_encoder.initial_state import (
    build_zero_gru_initial_state,
    build_zero_lstm_initial_state,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.memory.recurrent_memory_encoder.lstm_encoder import (
    LSTM_SEQUENCE_ENCODER_ALL_ZERO_POLICY,
    LSTM_SEQUENCE_ENCODER_COMPONENT_KIND,
    LSTM_SEQUENCE_ENCODER_COMPONENT_NAME,
    LSTM_SEQUENCE_ENCODER_ENCODING_NAME,
    LSTM_SEQUENCE_ENCODER_FEATURE_OBSERVATION_MASK_CONSUMED,
    LSTM_SEQUENCE_ENCODER_HAZARD_CONDITIONED,
    LSTM_SEQUENCE_ENCODER_IMPLEMENTATION_VERSION,
    LSTM_SEQUENCE_ENCODER_INITIAL_STATE_POLICY,
    LSTM_SEQUENCE_ENCODER_OPERATION_NAME,
    LSTM_SEQUENCE_ENCODER_OUTPUT_DIRECTION_ORDER,
    LSTM_SEQUENCE_ENCODER_PYTORCH_PROJECTION_SUPPORTED,
    LSTM_SEQUENCE_ENCODER_REFERENCE_POLICY,
    LSTM_SEQUENCE_ENCODER_RUN_NAME,
    LSTM_SEQUENCE_ENCODER_SCIENTIFIC_INTERPRETATION,
    LSTM_SEQUENCE_ENCODER_TEMPORAL_COORDINATES_CONSUMED,
    LSTMEncoder,
    LSTMSequenceEncoder,
    build_lstm_encoder,
    build_lstm_parameter_snapshot,
    build_lstm_sequence_encoder,
    snapshot_lstm_parameters,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.memory.recurrent_memory_encoder.schemas import (
    RecurrentExecutionPath,
    RecurrentSequenceEncoderRun,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.memory.recurrent_memory_encoder.sequence_packing import (
    canonicalize_recurrent_history,
    gather_canonical_node_sequence,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.memory.schemas.history_inputs import (
    HistoricalSequenceInputs,
    HistoryMissingValuePolicy,
    HistoryZeroLengthPolicy,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.memory.schemas.provenance import (
    MemoryParameterSnapshotProvenance,
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


T = 4
D = 3
H = 4
DEFAULT_LENGTHS = (
    2,
    0,
    4,
    1,
    3,
    2,
)


# =============================================================================
# Shared factories
# =============================================================================


def _mask_from_lengths(
    lengths: torch.Tensor,
    *,
    sequence_length: int = T,
    padding_direction: TemporalPaddingDirection,
) -> torch.Tensor:
    mask = torch.zeros(
        int(lengths.numel()),
        sequence_length,
        dtype=torch.bool,
        device=lengths.device,
    )

    for node_index, length_value in enumerate(lengths.tolist()):
        length = int(length_value)

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


def _coordinates_from_mask(
    mask: torch.Tensor,
    *,
    dtype: torch.dtype,
    shift: float = 0.0,
) -> torch.Tensor:
    coordinates = torch.zeros(
        *mask.shape,
        dtype=dtype,
        device=mask.device,
    )

    for node_index in range(int(mask.shape[0])):
        positions = torch.nonzero(
            mask[node_index],
            as_tuple=False,
        ).flatten()
        length = int(positions.numel())

        for logical_index, temporal_index in enumerate(
            positions.tolist()
        ):
            coordinates[
                node_index,
                temporal_index,
            ] = (
                float(logical_index - length)
                + shift
            )

    return coordinates


def _history(
    *,
    lengths: tuple[int, ...] | list[int] = DEFAULT_LENGTHS,
    padding_direction: TemporalPaddingDirection = (
        TemporalPaddingDirection.RIGHT
    ),
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
    feature_dim: int = D,
    requires_grad: bool = False,
    coordinate_shift: float = 0.0,
    feature_observed_mask: torch.Tensor | None = None,
    missing_value_policy: HistoryMissingValuePolicy = (
        HistoryMissingValuePolicy.UPSTREAM_IMPUTED
    ),
    source_suffix: str = "base",
) -> HistoricalSequenceInputs:
    length_tensor = torch.tensor(
        lengths,
        dtype=torch.long,
        device=device,
    )
    mask = _mask_from_lengths(
        length_tensor,
        padding_direction=padding_direction,
    )
    node_count = int(length_tensor.numel())

    values = torch.zeros(
        node_count,
        T,
        feature_dim,
        dtype=dtype,
        device=device,
    )
    next_value = 1.0

    for node_index in range(node_count):
        positions = torch.nonzero(
            mask[node_index],
            as_tuple=False,
        ).flatten()

        for temporal_position in positions.tolist():
            values[
                node_index,
                temporal_position,
            ] = torch.arange(
                next_value,
                next_value + feature_dim,
                dtype=dtype,
                device=device,
            )
            next_value += float(feature_dim)

    if requires_grad:
        values.requires_grad_()

    if feature_observed_mask is not None:
        feature_observed_mask = feature_observed_mask.to(
            device=device
        )

    return HistoricalSequenceInputs(
        history=values,
        timestep_mask=mask,
        node_axis=TemporalNodeAxis(
            node_ids=tuple(
                f"node-{index}"
                for index in range(node_count)
            ),
            node_batch_index=torch.zeros(
                node_count,
                dtype=torch.long,
                device=device,
            ),
            graph_count=1,
            graph_ids=("graph-0",),
            source_fingerprint=(
                f"recurrent-encoder-node-axis-{source_suffix}"
            ),
        ),
        feature_axis=TemporalFeatureAxis(
            feature_names=tuple(
                f"feature-{index}"
                for index in range(feature_dim)
            ),
            source_fingerprint=(
                f"recurrent-encoder-feature-axis-{source_suffix}"
            ),
        ),
        temporal_coordinates=RelativeTemporalCoordinates(
            values=_coordinates_from_mask(
                mask,
                dtype=dtype,
                shift=coordinate_shift,
            ),
            unit="months",
        ),
        source_provenance=MemorySourceProvenance(
            source_name="recurrent-encoder-panel",
            source_kind="historical-node-sequence",
            source_fingerprint=(
                f"recurrent-encoder-source-{source_suffix}"
            ),
            preprocessing_fingerprint=(
                f"recurrent-encoder-preprocessing-{source_suffix}"
            ),
        ),
        feature_observed_mask=feature_observed_mask,
        padding_direction=padding_direction,
        missing_value_policy=missing_value_policy,
        zero_length_policy=(
            HistoryZeroLengthPolicy.ALLOW_ZERO_HISTORY
            if any(length == 0 for length in lengths)
            else HistoryZeroLengthPolicy.ERROR
        ),
    )


def _all_observed_mask(
    history: HistoricalSequenceInputs,
) -> torch.Tensor:
    return history.timestep_mask.unsqueeze(-1).expand(
        history.node_count,
        history.sequence_length,
        history.feature_dim,
    ).clone()


def _all_missing_valid_step_mask(
    history: HistoricalSequenceInputs,
    *,
    node_index: int,
    temporal_index: int,
) -> torch.Tensor:
    observed = _all_observed_mask(history)
    assert bool(
        history.timestep_mask[
            node_index,
            temporal_index,
        ].item()
    )
    observed[
        node_index,
        temporal_index,
    ] = False
    return observed


def _config(
    *,
    cell_kind: RecurrentCellKind,
    pack_sequences: bool = True,
    input_dim: int = D,
    hidden_dim: int = H,
    num_layers: int = 2,
    dropout: float = 0.0,
    bidirectional: bool = True,
    use_bias: bool = True,
    input_projection_dim: int | None = None,
    layer_normalization: bool = False,
    enforce_sorted_lengths: bool = False,
) -> RecurrentSequenceEncoderConfig:
    return RecurrentSequenceEncoderConfig(
        cell_kind=cell_kind,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=bidirectional,
        use_bias=use_bias,
        input_projection_dim=input_projection_dim,
        layer_normalization=layer_normalization,
        pack_sequences=pack_sequences,
        enforce_sorted_lengths=enforce_sorted_lengths,
    )


def _encoder_class(
    cell_kind: RecurrentCellKind,
):
    return (
        GRUSequenceEncoder
        if cell_kind == RecurrentCellKind.GRU
        else LSTMSequenceEncoder
    )


def _build_encoder(
    config: RecurrentSequenceEncoderConfig,
):
    return _encoder_class(config.cell_kind)(
        config
    )


def _clone_encoder_with_config(
    encoder: GRUSequenceEncoder | LSTMSequenceEncoder,
    config: RecurrentSequenceEncoderConfig,
):
    clone = _build_encoder(config).to(
        device=next(encoder.parameters()).device,
        dtype=next(encoder.parameters()).dtype,
    )
    clone.load_state_dict(
        encoder.state_dict()
    )
    clone.train(
        encoder.training
    )
    return clone


def _expected_state_shape(
    config: RecurrentSequenceEncoderConfig,
    node_count: int,
) -> tuple[int, int, int, int]:
    return (
        config.num_layers,
        2 if config.bidirectional else 1,
        node_count,
        config.hidden_dim,
    )


def _assert_run_contract(
    run: RecurrentSequenceEncoderRun,
    history: HistoricalSequenceInputs,
    config: RecurrentSequenceEncoderConfig,
) -> None:
    assert isinstance(
        run,
        RecurrentSequenceEncoderRun,
    )
    assert isinstance(
        run.public_output,
        TemporalSequenceEncoding,
    )
    assert run.public_output.source_history is history
    assert run.node_count == history.node_count
    assert run.sequence_length == history.sequence_length
    assert run.hidden_dim == config.hidden_dim
    assert run.output_dim == config.output_dim
    assert run.num_layers == config.num_layers
    assert run.num_directions == (
        2 if config.bidirectional else 1
    )
    assert run.is_bidirectional is config.bidirectional
    assert run.device == history.device
    assert run.dtype == history.dtype
    assert run.public_output.encoded_shape == (
        history.node_count,
        history.sequence_length,
        config.output_dim,
    )
    assert run.final_hidden_state.shape == (
        _expected_state_shape(
            config,
            history.node_count,
        )
    )
    assert run.final_hidden_state_flat.shape == (
        config.num_layers
        * (
            2 if config.bidirectional else 1
        ),
        history.node_count,
        config.hidden_dim,
    )
    assert torch.isfinite(
        run.public_output.encoded_sequence
    ).all()
    assert torch.isfinite(
        run.final_hidden_state
    ).all()
    assert torch.count_nonzero(
        run.public_output.encoded_sequence[
            ~history.timestep_mask
        ]
    ).item() == 0

    zero_indices = torch.nonzero(
        history.valid_lengths == 0,
        as_tuple=False,
    ).flatten()

    if zero_indices.numel() > 0:
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

    if config.cell_kind == RecurrentCellKind.GRU:
        assert run.encoder_kind == TemporalSequenceEncoderKind.GRU
        assert run.final_cell_state is None
        assert run.final_cell_state_flat is None
        assert run.has_cell_state is False
    else:
        assert run.encoder_kind == TemporalSequenceEncoderKind.LSTM
        assert run.final_cell_state is not None
        assert run.final_cell_state.shape == (
            _expected_state_shape(
                config,
                history.node_count,
            )
        )
        assert run.final_cell_state_flat is not None
        assert run.final_cell_state_flat.shape == (
            config.num_layers
            * (
                2 if config.bidirectional else 1
            ),
            history.node_count,
            config.hidden_dim,
        )
        assert torch.isfinite(
            run.final_cell_state
        ).all()
        assert run.has_cell_state is True

        if zero_indices.numel() > 0:
            assert torch.count_nonzero(
                run.final_cell_state.index_select(
                    2,
                    zero_indices,
                )
            ).item() == 0


def _weighted_loss(
    run: RecurrentSequenceEncoderRun,
) -> torch.Tensor:
    sequence = run.public_output.encoded_sequence
    hidden = run.final_hidden_state

    sequence_weights = torch.arange(
        1,
        sequence.numel() + 1,
        dtype=sequence.dtype,
        device=sequence.device,
    ).reshape_as(sequence)
    hidden_weights = torch.arange(
        1,
        hidden.numel() + 1,
        dtype=hidden.dtype,
        device=hidden.device,
    ).reshape_as(hidden)

    loss = (
        sequence
        * sequence_weights
    ).sum() + (
        hidden
        * hidden_weights
    ).sum()

    if run.final_cell_state is not None:
        cell = run.final_cell_state
        cell_weights = torch.arange(
            1,
            cell.numel() + 1,
            dtype=cell.dtype,
            device=cell.device,
        ).reshape_as(cell)
        loss = loss + (
            cell
            * cell_weights
        ).sum()

    return loss


def _permuted_history(
    history: HistoricalSequenceInputs,
    permutation: torch.Tensor,
) -> HistoricalSequenceInputs:
    assert permutation.dtype == torch.long
    assert permutation.ndim == 1
    assert permutation.numel() == history.node_count

    feature_mask = (
        history.feature_observed_mask.index_select(
            0,
            permutation,
        )
        if history.feature_observed_mask is not None
        else None
    )

    coordinates = history.temporal_coordinates.values.index_select(
        0,
        permutation,
    )
    node_ids = tuple(
        history.node_ids[index]
        for index in permutation.tolist()
    )

    return HistoricalSequenceInputs(
        history=history.history.index_select(
            0,
            permutation,
        ),
        timestep_mask=history.timestep_mask.index_select(
            0,
            permutation,
        ),
        node_axis=TemporalNodeAxis(
            node_ids=node_ids,
            node_batch_index=history.node_batch_index.index_select(
                0,
                permutation,
            ),
            graph_count=history.graph_count,
            graph_ids=history.graph_ids,
            source_fingerprint=(
                "recurrent-encoder-permuted-node-axis"
            ),
        ),
        feature_axis=history.feature_axis,
        temporal_coordinates=RelativeTemporalCoordinates(
            values=coordinates,
            unit=history.temporal_coordinates.unit,
        ),
        source_provenance=history.source_provenance,
        feature_observed_mask=feature_mask,
        padding_direction=history.padding_direction,
        missing_value_policy=history.missing_value_policy,
        zero_length_policy=history.zero_length_policy,
        history_name=history.history_name,
    )


# =============================================================================
# Module identity, aliases, exports, and boundaries
# =============================================================================


@pytest.mark.parametrize(
    "value",
    (
        GRU_SEQUENCE_ENCODER_IMPLEMENTATION_VERSION,
        GRU_SEQUENCE_ENCODER_COMPONENT_NAME,
        GRU_SEQUENCE_ENCODER_COMPONENT_KIND,
        GRU_SEQUENCE_ENCODER_OPERATION_NAME,
        GRU_SEQUENCE_ENCODER_ENCODING_NAME,
        GRU_SEQUENCE_ENCODER_RUN_NAME,
        GRU_SEQUENCE_ENCODER_INITIAL_STATE_POLICY,
        GRU_SEQUENCE_ENCODER_REFERENCE_POLICY,
        GRU_SEQUENCE_ENCODER_ALL_ZERO_POLICY,
        GRU_SEQUENCE_ENCODER_SCIENTIFIC_INTERPRETATION,
        LSTM_SEQUENCE_ENCODER_IMPLEMENTATION_VERSION,
        LSTM_SEQUENCE_ENCODER_COMPONENT_NAME,
        LSTM_SEQUENCE_ENCODER_COMPONENT_KIND,
        LSTM_SEQUENCE_ENCODER_OPERATION_NAME,
        LSTM_SEQUENCE_ENCODER_ENCODING_NAME,
        LSTM_SEQUENCE_ENCODER_RUN_NAME,
        LSTM_SEQUENCE_ENCODER_INITIAL_STATE_POLICY,
        LSTM_SEQUENCE_ENCODER_REFERENCE_POLICY,
        LSTM_SEQUENCE_ENCODER_ALL_ZERO_POLICY,
        LSTM_SEQUENCE_ENCODER_SCIENTIFIC_INTERPRETATION,
    ),
)
def test_encoder_policy_strings_are_nonempty(
    value: str,
) -> None:
    assert isinstance(
        value,
        str,
    )
    assert value.strip()


def test_frozen_encoder_boolean_policies() -> None:
    assert GRU_SEQUENCE_ENCODER_FEATURE_OBSERVATION_MASK_CONSUMED is False
    assert GRU_SEQUENCE_ENCODER_TEMPORAL_COORDINATES_CONSUMED is False
    assert GRU_SEQUENCE_ENCODER_HAZARD_CONDITIONED is False
    assert LSTM_SEQUENCE_ENCODER_FEATURE_OBSERVATION_MASK_CONSUMED is False
    assert LSTM_SEQUENCE_ENCODER_TEMPORAL_COORDINATES_CONSUMED is False
    assert LSTM_SEQUENCE_ENCODER_HAZARD_CONDITIONED is False
    assert LSTM_SEQUENCE_ENCODER_PYTORCH_PROJECTION_SUPPORTED is False


def test_direction_order_constants() -> None:
    assert GRU_SEQUENCE_ENCODER_OUTPUT_DIRECTION_ORDER == (
        "forward",
        "backward",
    )
    assert LSTM_SEQUENCE_ENCODER_OUTPUT_DIRECTION_ORDER == (
        "forward",
        "backward",
    )


def test_gru_aliases_are_exact() -> None:
    assert GRUEncoder is GRUSequenceEncoder
    assert build_gru_encoder is build_gru_sequence_encoder
    assert snapshot_gru_parameters is build_gru_parameter_snapshot


def test_lstm_aliases_are_exact() -> None:
    assert LSTMEncoder is LSTMSequenceEncoder
    assert build_lstm_encoder is build_lstm_sequence_encoder
    assert snapshot_lstm_parameters is build_lstm_parameter_snapshot


@pytest.mark.parametrize(
    "module",
    (
        gru_encoder_module,
        lstm_encoder_module,
    ),
)
def test_encoder_module_all_is_unique_and_resolves(
    module,
) -> None:
    exported = module.__all__

    assert isinstance(
        exported,
        tuple,
    )
    assert len(exported) == len(set(exported))

    for name in exported:
        assert hasattr(
            module,
            name,
        )


def test_gru_module_does_not_import_lstm_encoder_or_dispatcher() -> None:
    source = inspect.getsource(
        gru_encoder_module
    )

    for forbidden in (
        "lstm_encoder",
        "recurrent_memory_encoder import",
        "diagnostics import",
        "baseline_encoders",
        "transformer",
    ):
        assert forbidden not in source


def test_lstm_module_does_not_import_gru_encoder_or_dispatcher() -> None:
    source = inspect.getsource(
        lstm_encoder_module
    )

    for forbidden in (
        "gru_encoder",
        "recurrent_memory_encoder import",
        "diagnostics import",
        "baseline_encoders",
        "transformer",
    ):
        assert forbidden not in source


# =============================================================================
# Construction and structural properties
# =============================================================================


@pytest.mark.parametrize(
    "cell_kind",
    (
        RecurrentCellKind.GRU,
        RecurrentCellKind.LSTM,
    ),
)
@pytest.mark.parametrize(
    (
        "projection_dim",
        "layer_normalization",
        "use_bias",
        "bidirectional",
        "num_layers",
    ),
    (
        (
            None,
            False,
            True,
            False,
            1,
        ),
        (
            None,
            True,
            True,
            True,
            2,
        ),
        (
            5,
            False,
            False,
            False,
            3,
        ),
        (
            5,
            True,
            True,
            True,
            2,
        ),
    ),
)
def test_encoder_construction_and_properties(
    cell_kind: RecurrentCellKind,
    projection_dim: int | None,
    layer_normalization: bool,
    use_bias: bool,
    bidirectional: bool,
    num_layers: int,
) -> None:
    config = _config(
        cell_kind=cell_kind,
        input_projection_dim=projection_dim,
        layer_normalization=layer_normalization,
        use_bias=use_bias,
        bidirectional=bidirectional,
        num_layers=num_layers,
    )




    encoder = _build_encoder(config)

    assert encoder.config is config
    assert encoder.cell_kind == cell_kind
    assert encoder.input_dim == D
    assert encoder.recurrent_input_dim == (
        projection_dim
        if projection_dim is not None
        else D
    )
    assert encoder.hidden_dim == H
    assert encoder.output_dim == (
        H
        * (
            2
            if bidirectional
            else 1
        )
    )
    assert encoder.num_layers == num_layers
    assert encoder.num_directions == (
        2
        if bidirectional
        else 1
    )
    assert encoder.is_bidirectional is bidirectional
    assert encoder.state_layout.num_layers == num_layers
    assert encoder.state_layout.num_directions == (
        2
        if bidirectional
        else 1
    )
    assert encoder.state_layout.hidden_dim == H
    assert encoder.parameter_count == sum(
        parameter.numel()
        for parameter in encoder.parameters()
    )
    assert encoder.trainable_parameter_count == sum(
        parameter.numel()
        for parameter in encoder.parameters()
        if parameter.requires_grad
    )

    if cell_kind == RecurrentCellKind.GRU:
        assert isinstance(
            encoder.kernel,
            nn.GRU,
        )
        assert encoder.kernel is encoder.gru
    else:
        assert isinstance(
            encoder.kernel,
            nn.LSTM,
        )
        assert encoder.kernel is encoder.lstm
        assert encoder.kernel.proj_size == 0

    assert encoder.kernel.batch_first is True
    assert encoder.kernel.bias is use_bias
    assert encoder.kernel.bidirectional is bidirectional
    assert encoder.kernel.num_layers == num_layers
    assert encoder.kernel.hidden_size == H
    assert encoder.kernel.input_size == encoder.recurrent_input_dim


@pytest.mark.parametrize(
    "cell_kind",
    (
        RecurrentCellKind.GRU,
        RecurrentCellKind.LSTM,
    ),
)
def test_encoder_builder_returns_correct_type(
    cell_kind: RecurrentCellKind,
) -> None:
    config = _config(
        cell_kind=cell_kind
    )

    if cell_kind == RecurrentCellKind.GRU:
        encoder = build_gru_sequence_encoder(
            config
        )
        assert isinstance(
            encoder,
            GRUSequenceEncoder,
        )
    else:
        encoder = build_lstm_sequence_encoder(
            config
        )
        assert isinstance(
            encoder,
            LSTMSequenceEncoder,
        )


def test_gru_rejects_lstm_config() -> None:
    with pytest.raises(ValueError):
        GRUSequenceEncoder(
            _config(
                cell_kind=RecurrentCellKind.LSTM
            )
        )


def test_lstm_rejects_gru_config() -> None:
    with pytest.raises(ValueError):
        LSTMSequenceEncoder(
            _config(
                cell_kind=RecurrentCellKind.GRU
            )
        )


@pytest.mark.parametrize(
    "encoder_class",
    (
        GRUSequenceEncoder,
        LSTMSequenceEncoder,
    ),
)
def test_encoder_rejects_wrong_config_type(
    encoder_class,
) -> None:
    with pytest.raises(TypeError):
        encoder_class(
            object()  # type: ignore[arg-type]
        )


# =============================================================================
# Public forward and run contracts
# =============================================================================


@pytest.mark.parametrize(
    "cell_kind",
    (
        RecurrentCellKind.GRU,
        RecurrentCellKind.LSTM,
    ),
)
@pytest.mark.parametrize(
    "pack_sequences",
    (
        True,
        False,
    ),
)
@pytest.mark.parametrize(
    "padding_direction",
    (
        TemporalPaddingDirection.RIGHT,
        TemporalPaddingDirection.LEFT,
    ),
)
@pytest.mark.parametrize(
    "bidirectional",
    (
        False,
        True,
    ),
)
def test_forward_and_encode_with_state_contracts(
    cell_kind: RecurrentCellKind,
    pack_sequences: bool,
    padding_direction: TemporalPaddingDirection,
    bidirectional: bool,
) -> None:
    torch.manual_seed(
        11
    )
    config = _config(
        cell_kind=cell_kind,
        pack_sequences=pack_sequences,
        bidirectional=bidirectional,
        num_layers=2,
    )
    encoder = _build_encoder(config).double()
    encoder.eval()
    history = _history(
        padding_direction=padding_direction
    )

    public_output = encoder(
        history
    )
    run = encoder.encode_with_state(
        history
    )

    assert isinstance(
        public_output,
        TemporalSequenceEncoding,
    )
    _assert_run_contract(
        run,
        history,
        config,
    )
    assert public_output.source_history is history
    assert public_output.encoder_kind == (
        config.schema_encoder_kind
    )
    assert torch.equal(
        public_output.encoded_sequence,
        run.public_output.encoded_sequence,
    )
    assert public_output.architecture_fingerprint == (
        run.architecture_fingerprint
    )
    assert run.execution_metadata.execution_path == (
        RecurrentExecutionPath.PACKED
        if pack_sequences
        else RecurrentExecutionPath.REFERENCE
    )
    assert run.execution_metadata.original_padding_direction == (
        padding_direction
    )


@pytest.mark.parametrize(
    "cell_kind",
    (
        RecurrentCellKind.GRU,
        RecurrentCellKind.LSTM,
    ),
)
def test_public_output_preserves_exact_source_contract(
    cell_kind: RecurrentCellKind,
) -> None:
    config = _config(
        cell_kind=cell_kind
    )
    encoder = _build_encoder(config).double()
    history = _history()

    output = encoder(
        history
    )

    assert output.source_history is history
    assert output.timestep_mask is history.timestep_mask
    assert output.node_axis is history.node_axis
    assert output.feature_axis is history.feature_axis
    assert output.temporal_coordinates is history.temporal_coordinates
    assert output.source_provenance is history.source_provenance
    assert output.node_ids == history.node_ids
    assert output.graph_ids == history.graph_ids
    assert output.valid_lengths.tolist() == list(
        DEFAULT_LENGTHS
    )


@pytest.mark.parametrize(
    "cell_kind",
    (
        RecurrentCellKind.GRU,
        RecurrentCellKind.LSTM,
    ),
)
def test_forward_does_not_create_parameter_snapshot_implicitly(
    cell_kind: RecurrentCellKind,
) -> None:
    encoder = _build_encoder(
        _config(
            cell_kind=cell_kind
        )
    ).double()
    output = encoder(
        _history()
    )

    assert output.computation_provenance.parameter_snapshot is None
    assert output.parameter_snapshot_fingerprint is None


@pytest.mark.parametrize(
    "cell_kind",
    (
        RecurrentCellKind.GRU,
        RecurrentCellKind.LSTM,
    ),
)
def test_wrong_source_type_is_rejected(
    cell_kind: RecurrentCellKind,
) -> None:
    encoder = _build_encoder(
        _config(
            cell_kind=cell_kind
        )
    )

    with pytest.raises(TypeError):
        encoder(
            object()  # type: ignore[arg-type]
        )


# =============================================================================
# Packed/reference equivalence
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
@pytest.mark.parametrize(
    (
        "projection_dim",
        "layer_normalization",
        "use_bias",
    ),
    (
        (
            None,
            False,
            True,
        ),
        (
            5,
            False,
            False,
        ),
        (
            5,
            True,
            True,
        ),
    ),
)
def test_packed_reference_equivalence_float64(
    cell_kind: RecurrentCellKind,
    bidirectional: bool,
    padding_direction: TemporalPaddingDirection,
    projection_dim: int | None,
    layer_normalization: bool,
    use_bias: bool,
) -> None:
    torch.manual_seed(
        1234
    )
    packed_config = _config(
        cell_kind=cell_kind,
        pack_sequences=True,
        bidirectional=bidirectional,
        input_projection_dim=projection_dim,
        layer_normalization=layer_normalization,
        use_bias=use_bias,
        num_layers=2,
    )
    reference_config = _config(
        cell_kind=cell_kind,
        pack_sequences=False,
        bidirectional=bidirectional,
        input_projection_dim=projection_dim,
        layer_normalization=layer_normalization,
        use_bias=use_bias,
        num_layers=2,
    )
    packed_encoder = _build_encoder(
        packed_config
    ).double()
    reference_encoder = _clone_encoder_with_config(
        packed_encoder,
        reference_config,
    )
    packed_encoder.eval()
    reference_encoder.eval()
    history = _history(
        padding_direction=padding_direction,
        dtype=torch.float64,
    )

    packed = packed_encoder.encode_with_state(
        history
    )
    reference = reference_encoder.encode_with_state(
        history
    )

    torch.testing.assert_close(
        packed.public_output.encoded_sequence,
        reference.public_output.encoded_sequence,
        rtol=1e-7,
        atol=1e-9,
    )
    torch.testing.assert_close(
        packed.final_hidden_state,
        reference.final_hidden_state,
        rtol=1e-7,
        atol=1e-9,
    )

    if cell_kind == RecurrentCellKind.LSTM:
        assert packed.final_cell_state is not None
        assert reference.final_cell_state is not None
        torch.testing.assert_close(
            packed.final_cell_state,
            reference.final_cell_state,
            rtol=1e-7,
            atol=1e-9,
        )

    assert packed.architecture_fingerprint == (
        reference.architecture_fingerprint
    )
    assert packed.computation_lineage_fingerprint != (
        reference.computation_lineage_fingerprint
    )


@pytest.mark.parametrize(
    "cell_kind",
    (
        RecurrentCellKind.GRU,
        RecurrentCellKind.LSTM,
    ),
)
def test_packed_reference_equivalence_float32(
    cell_kind: RecurrentCellKind,
) -> None:
    torch.manual_seed(
        4321
    )
    packed_config = _config(
        cell_kind=cell_kind,
        pack_sequences=True,
        input_projection_dim=5,
        layer_normalization=True,
        num_layers=1,
        bidirectional=True,
    )
    reference_config = _config(
        cell_kind=cell_kind,
        pack_sequences=False,
        input_projection_dim=5,
        layer_normalization=True,
        num_layers=1,
        bidirectional=True,
    )
    packed_encoder = _build_encoder(
        packed_config
    ).float()
    reference_encoder = _clone_encoder_with_config(
        packed_encoder,
        reference_config,
    )
    packed_encoder.eval()
    reference_encoder.eval()
    history = _history(
        dtype=torch.float32
    )

    packed = packed_encoder.encode_with_state(
        history
    )
    reference = reference_encoder.encode_with_state(
        history
    )

    torch.testing.assert_close(
        packed.public_output.encoded_sequence,
        reference.public_output.encoded_sequence,
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(
        packed.final_hidden_state,
        reference.final_hidden_state,
        rtol=1e-5,
        atol=1e-6,
    )

    if cell_kind == RecurrentCellKind.LSTM:
        assert packed.final_cell_state is not None
        assert reference.final_cell_state is not None
        torch.testing.assert_close(
            packed.final_cell_state,
            reference.final_cell_state,
            rtol=1e-5,
            atol=1e-6,
        )


# =============================================================================
# Left/right/no-padding equivalence
# =============================================================================


@pytest.mark.parametrize(
    "cell_kind",
    (
        RecurrentCellKind.GRU,
        RecurrentCellKind.LSTM,
    ),
)
@pytest.mark.parametrize(
    "pack_sequences",
    (
        True,
        False,
    ),
)
def test_left_right_padding_numerical_equivalence(
    cell_kind: RecurrentCellKind,
    pack_sequences: bool,
) -> None:
    torch.manual_seed(
        77
    )
    config = _config(
        cell_kind=cell_kind,
        pack_sequences=pack_sequences,
        input_projection_dim=5,
        layer_normalization=True,
    )
    encoder = _build_encoder(
        config
    ).double()
    encoder.eval()

    right = _history(
        padding_direction=TemporalPaddingDirection.RIGHT
    )
    left = _history(
        padding_direction=TemporalPaddingDirection.LEFT
    )

    right_run = encoder.encode_with_state(
        right
    )
    left_run = encoder.encode_with_state(
        left
    )

    for node_index, length in enumerate(
        DEFAULT_LENGTHS
    ):
        if length == 0:
            continue

        right_valid = right_run.public_output.encoded_sequence[
            node_index
        ][
            right.timestep_mask[
                node_index
            ]
        ]
        left_valid = left_run.public_output.encoded_sequence[
            node_index
        ][
            left.timestep_mask[
                node_index
            ]
        ]
        torch.testing.assert_close(
            right_valid,
            left_valid,
            rtol=1e-7,
            atol=1e-9,
        )

    torch.testing.assert_close(
        right_run.final_hidden_state,
        left_run.final_hidden_state,
        rtol=1e-7,
        atol=1e-9,
    )

    if cell_kind == RecurrentCellKind.LSTM:
        assert right_run.final_cell_state is not None
        assert left_run.final_cell_state is not None
        torch.testing.assert_close(
            right_run.final_cell_state,
            left_run.final_cell_state,
            rtol=1e-7,
            atol=1e-9,
        )


@pytest.mark.parametrize(
    "cell_kind",
    (
        RecurrentCellKind.GRU,
        RecurrentCellKind.LSTM,
    ),
)
def test_no_padding_history_executes(
    cell_kind: RecurrentCellKind,
) -> None:
    config = _config(
        cell_kind=cell_kind
    )
    encoder = _build_encoder(
        config
    ).double()
    history = _history(
        lengths=(
            T,
            T,
            T,
        ),
        padding_direction=TemporalPaddingDirection.NONE,
    )

    run = encoder.encode_with_state(
        history
    )

    _assert_run_contract(
        run,
        history,
        config,
    )
    assert run.execution_metadata.zero_history_count == 0
    assert bool(
        history.timestep_mask.all().item()
    )


# =============================================================================
# Gradients and strong packed/reference oracle
# =============================================================================


@pytest.mark.parametrize(
    "cell_kind",
    (
        RecurrentCellKind.GRU,
        RecurrentCellKind.LSTM,
    ),
)
def test_packed_reference_gradient_equivalence_float64(
    cell_kind: RecurrentCellKind,
) -> None:
    torch.manual_seed(
        909
    )
    packed_config = _config(
        cell_kind=cell_kind,
        pack_sequences=True,
        input_projection_dim=5,
        layer_normalization=True,
        bidirectional=True,
        num_layers=2,
        dropout=0.0,
    )
    reference_config = _config(
        cell_kind=cell_kind,
        pack_sequences=False,
        input_projection_dim=5,
        layer_normalization=True,
        bidirectional=True,
        num_layers=2,
        dropout=0.0,
    )
    packed_encoder = _build_encoder(
        packed_config
    ).double()
    reference_encoder = _clone_encoder_with_config(
        packed_encoder,
        reference_config,
    )
    packed_encoder.eval()
    reference_encoder.eval()

    packed_history = _history(
        padding_direction=TemporalPaddingDirection.LEFT,
        requires_grad=True,
    )
    reference_history = _history(
        padding_direction=TemporalPaddingDirection.LEFT,
        requires_grad=True,
    )

    packed_run = packed_encoder.encode_with_state(
        packed_history
    )
    reference_run = reference_encoder.encode_with_state(
        reference_history
    )

    _weighted_loss(
        packed_run
    ).backward()
    _weighted_loss(
        reference_run
    ).backward()

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
        packed_encoder.named_parameters()
    )
    reference_parameters = dict(
        reference_encoder.named_parameters()
    )

    assert packed_parameters.keys() == reference_parameters.keys()

    for name in packed_parameters:
        packed_gradient = packed_parameters[
            name
        ].grad
        reference_gradient = reference_parameters[
            name
        ].grad

        assert packed_gradient is not None
        assert reference_gradient is not None

        torch.testing.assert_close(
            packed_gradient,
            reference_gradient,
            rtol=1e-7,
            atol=1e-9,
        )


@pytest.mark.parametrize(
    "cell_kind",
    (
        RecurrentCellKind.GRU,
        RecurrentCellKind.LSTM,
    ),
)
@pytest.mark.parametrize(
    "pack_sequences",
    (
        True,
        False,
    ),
)
def test_valid_and_padding_gradient_behavior(
    cell_kind: RecurrentCellKind,
    pack_sequences: bool,
) -> None:
    config = _config(
        cell_kind=cell_kind,
        pack_sequences=pack_sequences,
        input_projection_dim=5,
        layer_normalization=True,
    )
    encoder = _build_encoder(
        config
    ).double()
    history = _history(
        requires_grad=True
    )

    run = encoder.encode_with_state(
        history
    )
    (
        run.public_output.encoded_sequence.square().sum()
        + run.final_hidden_state.square().sum()
        + (
            run.final_cell_state.square().sum()
            if run.final_cell_state is not None
            else 0.0
        )
    ).backward()

    assert history.history.grad is not None
    assert torch.count_nonzero(
        history.history.grad[
            ~history.timestep_mask
        ]
    ).item() == 0
    assert torch.isfinite(
        history.history.grad[
            history.timestep_mask
        ]
    ).all()
    assert any(
        parameter.grad is not None
        for parameter in encoder.parameters()
    )


# =============================================================================
# Zero-history and all-zero-history behavior
# =============================================================================


@pytest.mark.parametrize(
    "cell_kind",
    (
        RecurrentCellKind.GRU,
        RecurrentCellKind.LSTM,
    ),
)
@pytest.mark.parametrize(
    "pack_sequences",
    (
        True,
        False,
    ),
)
def test_mixed_zero_history_rows_remain_exact_zero(
    cell_kind: RecurrentCellKind,
    pack_sequences: bool,
) -> None:
    config = _config(
        cell_kind=cell_kind,
        pack_sequences=pack_sequences,
    )
    encoder = _build_encoder(
        config
    ).double()
    history = _history()

    run = encoder.encode_with_state(
        history
    )

    zero_indices = torch.tensor(
        [1],
        dtype=torch.long,
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
    "cell_kind",
    (
        RecurrentCellKind.GRU,
        RecurrentCellKind.LSTM,
    ),
)
@pytest.mark.parametrize(
    "pack_sequences",
    (
        True,
        False,
    ),
)
def test_all_zero_history_short_circuit_skips_adapter_and_kernel(
    cell_kind: RecurrentCellKind,
    pack_sequences: bool,
) -> None:
    config = _config(
        cell_kind=cell_kind,
        pack_sequences=pack_sequences,
    )
    encoder = _build_encoder(
        config
    ).double()
    history = _history(
        lengths=(
            0,
            0,
            0,
        )
    )
    adapter_calls: list[int] = []
    kernel_calls: list[int] = []

    adapter_hook = encoder.input_adapter.register_forward_hook(
        lambda *args: adapter_calls.append(1)
    )
    kernel_hook = encoder.kernel.register_forward_hook(
        lambda *args: kernel_calls.append(1)
    )

    run = encoder.encode_with_state(
        history
    )

    adapter_hook.remove()
    kernel_hook.remove()

    assert adapter_calls == []
    assert kernel_calls == []
    _assert_run_contract(
        run,
        history,
        config,
    )
    assert torch.count_nonzero(
        run.public_output.encoded_sequence
    ).item() == 0
    assert torch.count_nonzero(
        run.final_hidden_state
    ).item() == 0

    if run.final_cell_state is not None:
        assert torch.count_nonzero(
            run.final_cell_state
        ).item() == 0

    lineage = (
        run.public_output
        .computation_provenance
        .lineage
        .lineage_metadata
    )
    assert lineage["all_zero_history_short_circuit"] is True
    assert lineage["adapter_executed"] is False
    assert lineage["recurrent_kernel_executed"] is False
    assert lineage["execution_path"] == (
        "packed"
        if pack_sequences
        else "reference"
    )


@pytest.mark.parametrize(
    "cell_kind",
    (
        RecurrentCellKind.GRU,
        RecurrentCellKind.LSTM,
    ),
)
def test_all_zero_output_has_no_artificial_autograd_anchor(
    cell_kind: RecurrentCellKind,
) -> None:
    config = _config(
        cell_kind=cell_kind
    )
    encoder = _build_encoder(
        config
    ).double()
    history = _history(
        lengths=(
            0,
            0,
        ),
        requires_grad=True,
    )

    run = encoder.encode_with_state(
        history
    )

    assert run.public_output.encoded_sequence.requires_grad is False
    assert run.final_hidden_state.requires_grad is False

    if run.final_cell_state is not None:
        assert run.final_cell_state.requires_grad is False


@pytest.mark.parametrize(
    "cell_kind",
    (
        RecurrentCellKind.GRU,
        RecurrentCellKind.LSTM,
    ),
)
def test_all_zero_still_rejects_dtype_mismatch(
    cell_kind: RecurrentCellKind,
) -> None:
    encoder = _build_encoder(
        _config(
            cell_kind=cell_kind
        )
    ).float()
    history = _history(
        lengths=(
            0,
            0,
        ),
        dtype=torch.float64,
    )

    with pytest.raises(ValueError):
        encoder(
            history
        )


@pytest.mark.parametrize(
    "cell_kind",
    (
        RecurrentCellKind.GRU,
        RecurrentCellKind.LSTM,
    ),
)
def test_all_zero_still_rejects_feature_width_mismatch(
    cell_kind: RecurrentCellKind,
) -> None:
    encoder = _build_encoder(
        _config(
            cell_kind=cell_kind,
            input_dim=D,
        )
    ).double()
    history = _history(
        lengths=(
            0,
            0,
        ),
        feature_dim=D + 1,
    )

    with pytest.raises(ValueError):
        encoder(
            history
        )


# =============================================================================
# Explicit parameter snapshots
# =============================================================================


@pytest.mark.parametrize(
    "cell_kind",
    (
        RecurrentCellKind.GRU,
        RecurrentCellKind.LSTM,
    ),
)
def test_explicit_parameter_snapshot_is_linked(
    cell_kind: RecurrentCellKind,
) -> None:
    encoder = _build_encoder(
        _config(
            cell_kind=cell_kind,
            input_projection_dim=5,
            layer_normalization=True,
        )
    ).double()

    if cell_kind == RecurrentCellKind.GRU:
        snapshot = build_gru_parameter_snapshot(
            encoder,
            checkpoint_id="gru-checkpoint",
            checkpoint_fingerprint="gru-checkpoint-fingerprint",
            training_step=12,
        )
    else:
        snapshot = build_lstm_parameter_snapshot(
            encoder,
            checkpoint_id="lstm-checkpoint",
            checkpoint_fingerprint="lstm-checkpoint-fingerprint",
            training_step=13,
        )

    assert isinstance(
        snapshot,
        MemoryParameterSnapshotProvenance,
    )
    assert snapshot.parameter_count == encoder.parameter_count
    assert snapshot.trainable_parameter_count == (
        encoder.trainable_parameter_count
    )

    run = encoder.encode_with_state(
        _history(),
        parameter_snapshot=snapshot,
    )

    assert (
        run.public_output
        .computation_provenance
        .parameter_snapshot
        is snapshot
    )
    assert run.public_output.parameter_snapshot_fingerprint == (
        snapshot.parameter_snapshot_fingerprint
    )
    assert (
        run.public_output
        .computation_provenance
        .lineage
        .parameter_snapshot_fingerprint
        == snapshot.parameter_snapshot_fingerprint
    )


@pytest.mark.parametrize(
    "cell_kind",
    (
        RecurrentCellKind.GRU,
        RecurrentCellKind.LSTM,
    ),
)
def test_stale_snapshot_is_rejected(
    cell_kind: RecurrentCellKind,
) -> None:
    encoder = _build_encoder(
        _config(
            cell_kind=cell_kind
        )
    ).double()
    snapshot = encoder.build_parameter_snapshot()

    with torch.no_grad():
        next(
            encoder.parameters()
        ).add_(0.01)

    with pytest.raises(ValueError):
        encoder(
            _history(),
            parameter_snapshot=snapshot,
        )


@pytest.mark.parametrize(
    "cell_kind",
    (
        RecurrentCellKind.GRU,
        RecurrentCellKind.LSTM,
    ),
)
def test_stale_snapshot_is_rejected_on_all_zero_batch(
    cell_kind: RecurrentCellKind,
) -> None:
    encoder = _build_encoder(
        _config(
            cell_kind=cell_kind
        )
    ).double()
    snapshot = encoder.build_parameter_snapshot()

    with torch.no_grad():
        next(
            encoder.parameters()
        ).add_(0.01)

    with pytest.raises(ValueError):
        encoder(
            _history(
                lengths=(
                    0,
                    0,
                )
            ),
            parameter_snapshot=snapshot,
        )


def test_gru_snapshot_rejected_by_lstm() -> None:
    gru = GRUSequenceEncoder(
        _config(
            cell_kind=RecurrentCellKind.GRU
        )
    ).double()
    lstm = LSTMSequenceEncoder(
        _config(
            cell_kind=RecurrentCellKind.LSTM
        )
    ).double()
    snapshot = gru.build_parameter_snapshot()

    with pytest.raises(ValueError):
        lstm(
            _history(),
            parameter_snapshot=snapshot,
        )


def test_lstm_snapshot_rejected_by_gru() -> None:
    lstm = LSTMSequenceEncoder(
        _config(
            cell_kind=RecurrentCellKind.LSTM
        )
    ).double()
    gru = GRUSequenceEncoder(
        _config(
            cell_kind=RecurrentCellKind.GRU
        )
    ).double()
    snapshot = lstm.build_parameter_snapshot()

    with pytest.raises(ValueError):
        gru(
            _history(),
            parameter_snapshot=snapshot,
        )


def test_snapshot_builder_wrappers_reject_wrong_encoder_type() -> None:
    gru = GRUSequenceEncoder(
        _config(
            cell_kind=RecurrentCellKind.GRU
        )
    )
    lstm = LSTMSequenceEncoder(
        _config(
            cell_kind=RecurrentCellKind.LSTM
        )
    )

    with pytest.raises(TypeError):
        build_gru_parameter_snapshot(
            lstm  # type: ignore[arg-type]
        )

    with pytest.raises(TypeError):
        build_lstm_parameter_snapshot(
            gru  # type: ignore[arg-type]
        )


# =============================================================================
# Provenance and architecture identity
# =============================================================================


@pytest.mark.parametrize(
    "cell_kind",
    (
        RecurrentCellKind.GRU,
        RecurrentCellKind.LSTM,
    ),
)
def test_packed_reference_share_architecture_identity(
    cell_kind: RecurrentCellKind,
) -> None:
    packed_config = _config(
        cell_kind=cell_kind,
        pack_sequences=True,
    )
    reference_config = _config(
        cell_kind=cell_kind,
        pack_sequences=False,
    )
    packed = _build_encoder(
        packed_config
    )
    reference = _build_encoder(
        reference_config
    )

    packed_provenance = packed.architecture_provenance()
    reference_provenance = reference.architecture_provenance()

    assert (
        packed_provenance.architecture_fingerprint
        == reference_provenance.architecture_fingerprint
    )
    assert (
        packed_provenance.configuration_fingerprint
        == reference_provenance.configuration_fingerprint
    )
    assert (
        packed_provenance.architecture_metadata
        == reference_provenance.architecture_metadata
    )
    assert "pack_sequences" not in (
        packed_provenance.architecture_metadata
    )
    assert "enforce_sorted_lengths" not in (
        packed_provenance.architecture_metadata
    )


def test_gru_and_lstm_have_distinct_architecture_identity() -> None:
    gru = GRUSequenceEncoder(
        _config(
            cell_kind=RecurrentCellKind.GRU
        )
    )
    lstm = LSTMSequenceEncoder(
        _config(
            cell_kind=RecurrentCellKind.LSTM
        )
    )

    assert (
        gru.architecture_provenance().architecture_fingerprint
        != lstm.architecture_provenance().architecture_fingerprint
    )


@pytest.mark.parametrize(
    "cell_kind",
    (
        RecurrentCellKind.GRU,
        RecurrentCellKind.LSTM,
    ),
)
def test_architecture_metadata_uses_explicit_allowlist(
    cell_kind: RecurrentCellKind,
) -> None:
    config = _config(
        cell_kind=cell_kind,
        input_projection_dim=5,
        layer_normalization=True,
        bidirectional=True,
        use_bias=False,
        dropout=0.2,
        num_layers=2,
    )
    encoder = _build_encoder(
        config
    )
    metadata = (
        encoder
        .architecture_provenance()
        .architecture_metadata
    )

    assert metadata["cell_kind"] == cell_kind.value
    assert metadata["input_dim"] == D
    assert metadata["effective_recurrent_input_dim"] == 5
    assert metadata["hidden_dim"] == H
    assert metadata["output_dim"] == H * 2
    assert metadata["num_layers"] == 2
    assert metadata["num_directions"] == 2
    assert metadata["dropout"] == 0.2
    assert metadata["bidirectional"] is True
    assert metadata["use_bias"] is False
    assert metadata["input_projection_enabled"] is True
    assert metadata["input_projection_dim"] == 5
    assert metadata["input_projection_bias_enabled"] is False
    assert metadata["layer_normalization"] is True
    assert metadata["layer_norm_elementwise_affine"] is True
    assert metadata["feature_observation_mask_consumed"] is False
    assert metadata["temporal_coordinates_consumed"] is False
    assert metadata["hazard_conditioned"] is False


@pytest.mark.parametrize(
    "cell_kind",
    (
        RecurrentCellKind.GRU,
        RecurrentCellKind.LSTM,
    ),
)
@pytest.mark.parametrize(
    "pack_sequences",
    (
        True,
        False,
    ),
)
def test_execution_lineage_reports_actual_path(
    cell_kind: RecurrentCellKind,
    pack_sequences: bool,
) -> None:
    config = _config(
        cell_kind=cell_kind,
        pack_sequences=pack_sequences,
    )
    encoder = _build_encoder(
        config
    ).double()
    run = encoder.encode_with_state(
        _history()
    )
    lineage = (
        run.public_output
        .computation_provenance
        .lineage
        .lineage_metadata
    )

    assert lineage["execution_path"] == (
        "packed"
        if pack_sequences
        else "reference"
    )
    assert lineage["module_training"] is True
    assert lineage["dropout_active"] is False
    assert lineage["nonempty_node_count"] == 5
    assert lineage["zero_history_count"] == 1
    assert lineage["source_node_count"] == 6
    assert lineage["adapter_executed"] is True
    assert lineage["recurrent_kernel_executed"] is True
    assert lineage["all_zero_history_short_circuit"] is False


# =============================================================================
# Observation-mask and temporal-coordinate boundaries
# =============================================================================


@pytest.mark.parametrize(
    "cell_kind",
    (
        RecurrentCellKind.GRU,
        RecurrentCellKind.LSTM,
    ),
)
def test_feature_observed_mask_does_not_change_execution(
    cell_kind: RecurrentCellKind,
) -> None:
    base = _history(
        source_suffix="observed-base"
    )
    all_observed = _all_observed_mask(
        base
    )
    one_missing_step = _all_missing_valid_step_mask(
        base,
        node_index=0,
        temporal_index=0,
    )

    history_a = _history(
        feature_observed_mask=all_observed,
        missing_value_policy=(
            HistoryMissingValuePolicy.FINITE_PLACEHOLDER_WITH_MASK
        ),
        source_suffix="observed-a",
    )
    history_b = _history(
        feature_observed_mask=one_missing_step,
        missing_value_policy=(
            HistoryMissingValuePolicy.FINITE_PLACEHOLDER_WITH_MASK
        ),
        source_suffix="observed-b",
    )
    encoder = _build_encoder(
        _config(
            cell_kind=cell_kind
        )
    ).double()
    encoder.eval()

    run_a = encoder.encode_with_state(
        history_a
    )
    run_b = encoder.encode_with_state(
        history_b
    )

    torch.testing.assert_close(
        run_a.public_output.encoded_sequence,
        run_b.public_output.encoded_sequence,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        run_a.final_hidden_state,
        run_b.final_hidden_state,
        rtol=0.0,
        atol=0.0,
    )

    if cell_kind == RecurrentCellKind.LSTM:
        assert run_a.final_cell_state is not None
        assert run_b.final_cell_state is not None
        torch.testing.assert_close(
            run_a.final_cell_state,
            run_b.final_cell_state,
            rtol=0.0,
            atol=0.0,
        )

    # The valid all-feature-missing timestep is still executed.
    assert torch.count_nonzero(
        run_b.public_output.encoded_sequence[
            0,
            0,
        ]
    ).item() > 0


@pytest.mark.parametrize(
    "cell_kind",
    (
        RecurrentCellKind.GRU,
        RecurrentCellKind.LSTM,
    ),
)
def test_temporal_coordinate_values_do_not_change_execution(
    cell_kind: RecurrentCellKind,
) -> None:
    history_a = _history(
        coordinate_shift=0.0,
        source_suffix="time-a",
    )
    history_b = _history(
        coordinate_shift=-100.0,
        source_suffix="time-b",
    )
    encoder = _build_encoder(
        _config(
            cell_kind=cell_kind
        )
    ).double()
    encoder.eval()

    run_a = encoder.encode_with_state(
        history_a
    )
    run_b = encoder.encode_with_state(
        history_b
    )

    torch.testing.assert_close(
        run_a.public_output.encoded_sequence,
        run_b.public_output.encoded_sequence,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        run_a.final_hidden_state,
        run_b.final_hidden_state,
        rtol=0.0,
        atol=0.0,
    )

    if cell_kind == RecurrentCellKind.LSTM:
        assert run_a.final_cell_state is not None
        assert run_b.final_cell_state is not None
        torch.testing.assert_close(
            run_a.final_cell_state,
            run_b.final_cell_state,
            rtol=0.0,
            atol=0.0,
        )

    assert run_a.computation_lineage_fingerprint != (
        run_b.computation_lineage_fingerprint
    )


# =============================================================================
# Node-permutation equivariance
# =============================================================================


@pytest.mark.parametrize(
    "cell_kind",
    (
        RecurrentCellKind.GRU,
        RecurrentCellKind.LSTM,
    ),
)
@pytest.mark.parametrize(
    "pack_sequences",
    (
        True,
        False,
    ),
)
def test_node_permutation_equivariance(
    cell_kind: RecurrentCellKind,
    pack_sequences: bool,
) -> None:
    torch.manual_seed(
        181
    )
    config = _config(
        cell_kind=cell_kind,
        pack_sequences=pack_sequences,
        input_projection_dim=5,
        layer_normalization=True,
    )
    encoder = _build_encoder(
        config
    ).double()
    encoder.eval()
    history = _history()
    permutation = torch.tensor(
        [4, 1, 5, 0, 3, 2],
        dtype=torch.long,
    )
    inverse = torch.empty_like(
        permutation
    )
    inverse[
        permutation
    ] = torch.arange(
        permutation.numel(),
        dtype=torch.long,
    )
    permuted = _permuted_history(
        history,
        permutation,
    )

    original_run = encoder.encode_with_state(
        history
    )
    permuted_run = encoder.encode_with_state(
        permuted
    )

    restored_sequence = (
        permuted_run
        .public_output
        .encoded_sequence
        .index_select(
            0,
            inverse,
        )
    )
    restored_hidden = (
        permuted_run
        .final_hidden_state
        .index_select(
            2,
            inverse,
        )
    )

    torch.testing.assert_close(
        original_run.public_output.encoded_sequence,
        restored_sequence,
        rtol=1e-7,
        atol=1e-9,
    )
    torch.testing.assert_close(
        original_run.final_hidden_state,
        restored_hidden,
        rtol=1e-7,
        atol=1e-9,
    )

    if cell_kind == RecurrentCellKind.LSTM:
        assert original_run.final_cell_state is not None
        assert permuted_run.final_cell_state is not None
        restored_cell = (
            permuted_run
            .final_cell_state
            .index_select(
                2,
                inverse,
            )
        )
        torch.testing.assert_close(
            original_run.final_cell_state,
            restored_cell,
            rtol=1e-7,
            atol=1e-9,
        )


# =============================================================================
# State-dict round trip and deterministic evaluation
# =============================================================================


@pytest.mark.parametrize(
    "cell_kind",
    (
        RecurrentCellKind.GRU,
        RecurrentCellKind.LSTM,
    ),
)
def test_state_dict_round_trip(
    cell_kind: RecurrentCellKind,
) -> None:
    torch.manual_seed(
        222
    )
    config = _config(
        cell_kind=cell_kind,
        input_projection_dim=5,
        layer_normalization=True,
    )
    source_encoder = _build_encoder(
        config
    ).double()
    target_encoder = _build_encoder(
        config
    ).double()
    target_encoder.load_state_dict(
        copy.deepcopy(
            source_encoder.state_dict()
        )
    )
    source_encoder.eval()
    target_encoder.eval()
    history = _history()

    source_run = source_encoder.encode_with_state(
        history
    )
    target_run = target_encoder.encode_with_state(
        history
    )

    torch.testing.assert_close(
        source_run.public_output.encoded_sequence,
        target_run.public_output.encoded_sequence,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        source_run.final_hidden_state,
        target_run.final_hidden_state,
        rtol=0.0,
        atol=0.0,
    )

    if cell_kind == RecurrentCellKind.LSTM:
        assert source_run.final_cell_state is not None
        assert target_run.final_cell_state is not None
        torch.testing.assert_close(
            source_run.final_cell_state,
            target_run.final_cell_state,
            rtol=0.0,
            atol=0.0,
        )


@pytest.mark.parametrize(
    "cell_kind",
    (
        RecurrentCellKind.GRU,
        RecurrentCellKind.LSTM,
    ),
)
def test_eval_mode_is_deterministic_with_configured_dropout(
    cell_kind: RecurrentCellKind,
) -> None:
    config = _config(
        cell_kind=cell_kind,
        num_layers=3,
        dropout=0.4,
    )
    encoder = _build_encoder(
        config
    ).double()
    encoder.eval()
    history = _history()

    first = encoder.encode_with_state(
        history
    )
    second = encoder.encode_with_state(
        history
    )

    assert encoder.dropout_active is False
    torch.testing.assert_close(
        first.public_output.encoded_sequence,
        second.public_output.encoded_sequence,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        first.final_hidden_state,
        second.final_hidden_state,
        rtol=0.0,
        atol=0.0,
    )

    if cell_kind == RecurrentCellKind.LSTM:
        assert first.final_cell_state is not None
        assert second.final_cell_state is not None
        torch.testing.assert_close(
            first.final_cell_state,
            second.final_cell_state,
            rtol=0.0,
            atol=0.0,
        )


# =============================================================================
# Training dropout behavior
# =============================================================================


@pytest.mark.parametrize(
    "cell_kind",
    (
        RecurrentCellKind.GRU,
        RecurrentCellKind.LSTM,
    ),
)
@pytest.mark.parametrize(
    "pack_sequences",
    (
        True,
        False,
    ),
)
def test_training_with_dropout_preserves_contract_and_gradients(
    cell_kind: RecurrentCellKind,
    pack_sequences: bool,
) -> None:
    config = _config(
        cell_kind=cell_kind,
        pack_sequences=pack_sequences,
        num_layers=3,
        dropout=0.35,
        input_projection_dim=5,
        layer_normalization=True,
    )
    encoder = _build_encoder(
        config
    ).float()
    encoder.train()
    history = _history(
        dtype=torch.float32,
        requires_grad=True,
    )

    run = encoder.encode_with_state(
        history
    )

    assert encoder.dropout_active is True
    _assert_run_contract(
        run,
        history,
        config,
    )

    lineage = (
        run.public_output
        .computation_provenance
        .lineage
        .lineage_metadata
    )
    assert lineage["dropout_active"] is True
    assert lineage["module_training"] is True

    _weighted_loss(
        run
    ).backward()

    assert history.history.grad is not None
    assert torch.count_nonzero(
        history.history.grad[
            ~history.timestep_mask
        ]
    ).item() == 0
    assert all(
        parameter.grad is not None
        for parameter in encoder.parameters()
    )


# =============================================================================
# Direct returned-state oracle
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
def test_reference_run_states_match_direct_kernel_return(
    cell_kind: RecurrentCellKind,
    bidirectional: bool,
) -> None:
    torch.manual_seed(
        333
    )
    config = _config(
        cell_kind=cell_kind,
        pack_sequences=False,
        bidirectional=bidirectional,
        num_layers=2,
        input_projection_dim=5,
        layer_normalization=True,
    )
    encoder = _build_encoder(
        config
    ).double()
    encoder.eval()
    history = _history(
        lengths=(
            3,
            2,
        )
    )
    run = encoder.encode_with_state(
        history
    )

    metadata = run.execution_metadata
    canonical = canonicalize_recurrent_history(
        history,
        metadata,
    )
    adapted = encoder.input_adapter.adapt_canonical_batch(
        canonical
    )
    node_sequence = gather_canonical_node_sequence(
        adapted,
        0,
    )

    if cell_kind == RecurrentCellKind.GRU:
        initial = build_zero_gru_initial_state(
            encoder.kernel,
            config,
            batch_size=1,
        )
        _, direct_hidden = encoder.kernel(
            node_sequence,
            initial,
        )
        direct_cell = None
    else:
        initial = build_zero_lstm_initial_state(
            encoder.kernel,
            config,
            batch_size=1,
        )
        _, (
            direct_hidden,
            direct_cell,
        ) = encoder.kernel(
            node_sequence,
            initial,
        )

    direct_hidden = encoder.state_layout.unflatten_state(
        direct_hidden
    )

    torch.testing.assert_close(
        run.final_hidden_state[
            :,
            :,
            0:1,
            :,
        ],
        direct_hidden,
        rtol=1e-7,
        atol=1e-9,
    )

    if cell_kind == RecurrentCellKind.LSTM:
        assert direct_cell is not None
        assert run.final_cell_state is not None
        direct_cell = encoder.state_layout.unflatten_state(
            direct_cell
        )
        torch.testing.assert_close(
            run.final_cell_state[
                :,
                :,
                0:1,
                :,
            ],
            direct_cell,
            rtol=1e-7,
            atol=1e-9,
        )


# =============================================================================
# Defensive source/module validation
# =============================================================================


@pytest.mark.parametrize(
    "cell_kind",
    (
        RecurrentCellKind.GRU,
        RecurrentCellKind.LSTM,
    ),
)
def test_encoder_rejects_dtype_mismatch(
    cell_kind: RecurrentCellKind,
) -> None:
    encoder = _build_encoder(
        _config(
            cell_kind=cell_kind
        )
    ).float()
    history = _history(
        dtype=torch.float64
    )

    with pytest.raises(ValueError):
        encoder(
            history
        )


@pytest.mark.parametrize(
    "cell_kind",
    (
        RecurrentCellKind.GRU,
        RecurrentCellKind.LSTM,
    ),
)
def test_encoder_rejects_feature_width_mismatch(
    cell_kind: RecurrentCellKind,
) -> None:
    encoder = _build_encoder(
        _config(
            cell_kind=cell_kind,
            input_dim=D,
        )
    ).double()
    history = _history(
        feature_dim=D + 1
    )

    with pytest.raises(ValueError):
        encoder(
            history
        )


@pytest.mark.parametrize(
    "cell_kind",
    (
        RecurrentCellKind.GRU,
        RecurrentCellKind.LSTM,
    ),
)
def test_encoder_rechecks_mutated_padding(
    cell_kind: RecurrentCellKind,
) -> None:
    encoder = _build_encoder(
        _config(
            cell_kind=cell_kind
        )
    ).double()
    history = _history()

    with torch.no_grad():
        padded_position = torch.nonzero(
            ~history.timestep_mask,
            as_tuple=False,
        )[0]
        history.history[
            int(padded_position[0]),
            int(padded_position[1]),
            0,
        ] = 1.0

    with pytest.raises(ValueError):
        encoder(
            history
        )


@pytest.mark.parametrize(
    "cell_kind",
    (
        RecurrentCellKind.GRU,
        RecurrentCellKind.LSTM,
    ),
)
def test_encoder_rechecks_mutated_nonfinite_value(
    cell_kind: RecurrentCellKind,
) -> None:
    encoder = _build_encoder(
        _config(
            cell_kind=cell_kind
        )
    ).double()
    history = _history()

    with torch.no_grad():
        history.history[
            0,
            0,
            0,
        ] = float("nan")

    with pytest.raises(ValueError):
        encoder(
            history
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
@pytest.mark.parametrize(
    "pack_sequences",
    (
        True,
        False,
    ),
)
def test_recurrent_encoder_cuda(
    cell_kind: RecurrentCellKind,
    pack_sequences: bool,
) -> None:
    torch.manual_seed(
        444
    )
    config = _config(
        cell_kind=cell_kind,
        pack_sequences=pack_sequences,
        input_projection_dim=5,
        layer_normalization=True,
        num_layers=2,
        bidirectional=True,
    )
    encoder = _build_encoder(
        config
    ).cuda().float()
    encoder.train()
    history = _history(
        dtype=torch.float32,
        device="cuda",
        padding_direction=TemporalPaddingDirection.LEFT,
        requires_grad=True,
    )

    run = encoder.encode_with_state(
        history
    )

    _assert_run_contract(
        run,
        history,
        config,
    )
    assert run.device.type == "cuda"

    _weighted_loss(
        run
    ).backward()

    assert history.history.grad is not None
    assert torch.count_nonzero(
        history.history.grad[
            ~history.timestep_mask
        ]
    ).item() == 0
    assert torch.isfinite(
        history.history.grad
    ).all()
