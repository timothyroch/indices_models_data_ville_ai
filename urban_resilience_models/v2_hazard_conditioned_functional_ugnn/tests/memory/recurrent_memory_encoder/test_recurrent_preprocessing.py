"""
Consolidated preprocessing tests for Phase 6 recurrent memory encoders.

Target repository path:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            tests/
                memory/
                    recurrent_memory_encoder/
                        test_recurrent_preprocessing.py

Modules under test:
    memory/recurrent_memory_encoder/history_lengths.py
    memory/recurrent_memory_encoder/input_adapter.py
    memory/recurrent_memory_encoder/initial_state.py

This consolidated suite keeps deterministic preprocessing, feature adaptation,
and exact-zero state construction in one place. It deliberately excludes
sequence packing/restoration and complete GRU/LSTM execution, which belong to
``test_recurrent_execution.py`` and ``test_recurrent_encoders.py``.
"""

from __future__ import annotations

import inspect

import pytest
import torch
from torch import nn

import urban_resilience_models.v2_hazard_conditioned_functional_ugnn.memory.recurrent_memory_encoder.history_lengths as history_lengths_module
import urban_resilience_models.v2_hazard_conditioned_functional_ugnn.memory.recurrent_memory_encoder.initial_state as initial_state_module
import urban_resilience_models.v2_hazard_conditioned_functional_ugnn.memory.recurrent_memory_encoder.input_adapter as input_adapter_module
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.memory.config import (
    RecurrentCellKind,
    RecurrentSequenceEncoderConfig,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.memory.recurrent_memory_encoder.history_lengths import (
    RECURRENT_HISTORY_LENGTHS_COMPONENT_NAME,
    RECURRENT_HISTORY_LENGTHS_IMPLEMENTATION_VERSION,
    RECURRENT_HISTORY_LENGTHS_OPERATION_NAME,
    RECURRENT_HISTORY_LENGTHS_SCIENTIFIC_INTERPRETATION,
    RECURRENT_HISTORY_LENGTHS_SORT_POLICY,
    RECURRENT_HISTORY_LENGTHS_ZERO_POLICY_SOURCE,
    build_execution_metadata,
    build_recurrent_execution_metadata,
    build_recurrent_execution_metadata_from_lengths,
    build_recurrent_sort_permutations,
    derive_history_lengths,
    derive_recurrent_history_lengths,
    gather_nonempty_history_lengths,
    inverse_permutation,
    lengths_are_nonincreasing,
    partition_history_nodes,
    partition_recurrent_history_nodes,
    stable_descending_length_permutation,
    stable_length_sort,
    validate_recurrent_history_lengths,
    validate_source_zero_history_policy,
    validate_zero_history_policy,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.memory.recurrent_memory_encoder.initial_state import (
    RECURRENT_INITIAL_STATE_CALLER_SUPPLIED_SUPPORTED,
    RECURRENT_INITIAL_STATE_CANONICAL_AXIS_ORDER,
    RECURRENT_INITIAL_STATE_COMPONENT_KIND,
    RECURRENT_INITIAL_STATE_COMPONENT_NAME,
    RECURRENT_INITIAL_STATE_FLAT_AXIS_ORDER,
    RECURRENT_INITIAL_STATE_IMPLEMENTATION_VERSION,
    RECURRENT_INITIAL_STATE_LEARNED_SUPPORTED,
    RECURRENT_INITIAL_STATE_OPERATION_NAME,
    RECURRENT_INITIAL_STATE_POLICY,
    RECURRENT_INITIAL_STATE_SCIENTIFIC_INTERPRETATION,
    RECURRENT_INITIAL_STATE_STATEFUL_SUPPORTED,
    build_recurrent_state_layout,
    build_state_layout,
    build_zero_canonical_cell_state,
    build_zero_canonical_final_states,
    build_zero_canonical_hidden_state,
    build_zero_final_states,
    build_zero_flat_cell_state,
    build_zero_flat_hidden_state,
    build_zero_gru_initial_state,
    build_zero_initial_state,
    build_zero_lstm_initial_state,
    build_zero_recurrent_initial_state,
    flatten_recurrent_state,
    flatten_state,
    infer_recurrent_kernel_device_dtype,
    recurrent_canonical_state_shape,
    recurrent_flat_state_shape,
    unflatten_recurrent_state,
    unflatten_state,
    validate_exact_zero_recurrent_state,
    validate_flat_initial_state,
    validate_recurrent_kernel_configuration,
    validate_recurrent_kernel_runtime,
    validate_zero_recurrent_initial_state,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.memory.recurrent_memory_encoder.input_adapter import (
    RECURRENT_INPUT_ADAPTER_BIAS_POLICY,
    RECURRENT_INPUT_ADAPTER_COMPONENT_KIND,
    RECURRENT_INPUT_ADAPTER_COMPONENT_NAME,
    RECURRENT_INPUT_ADAPTER_FEATURE_OBSERVATION_POLICY,
    RECURRENT_INPUT_ADAPTER_HAZARD_POLICY,
    RECURRENT_INPUT_ADAPTER_IMPLEMENTATION_VERSION,
    RECURRENT_INPUT_ADAPTER_LAYER_NORM_ELEMENTWISE_AFFINE,
    RECURRENT_INPUT_ADAPTER_NORMALIZATION_POLICY,
    RECURRENT_INPUT_ADAPTER_OPERATION_NAME,
    RECURRENT_INPUT_ADAPTER_OUTPUT_STAGE,
    RECURRENT_INPUT_ADAPTER_PADDING_POLICY,
    RECURRENT_INPUT_ADAPTER_PROJECTION_POLICY,
    RECURRENT_INPUT_ADAPTER_SCIENTIFIC_INTERPRETATION,
    RECURRENT_INPUT_ADAPTER_TEMPORAL_COORDINATE_POLICY,
    RECURRENT_INPUT_ADAPTER_TEMPORAL_INTERACTION,
    InputAdapter,
    RecurrentInputAdapter,
    adapt_canonical_recurrent_batch,
    adapt_input_values,
    adapt_recurrent_input_values,
    build_input_adapter,
    build_recurrent_input_adapter,
    exact_zero_recurrent_padding,
    validate_recurrent_module_source_compatibility,
    validate_recurrent_module_tensor_compatibility,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.memory.recurrent_memory_encoder.schemas import (
    RecurrentExecutionPath,
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
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.memory.schemas.temporal_coordinates import (
    RelativeTemporalCoordinates,
    TemporalPaddingDirection,
)


T = 4
D = 2
H = 3


# =============================================================================
# Shared factories
# =============================================================================


def _mask_from_lengths(
    lengths: torch.Tensor,
    *,
    sequence_length: int = T,
    padding_direction: TemporalPaddingDirection = (
        TemporalPaddingDirection.RIGHT
    ),
) -> torch.Tensor:
    mask = torch.zeros(
        int(lengths.numel()),
        sequence_length,
        dtype=torch.bool,
        device=lengths.device,
    )

    for node_index, length_value in enumerate(
        lengths.tolist()
    ):
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


def _history(
    *,
    lengths: torch.Tensor | None = None,
    padding_direction: TemporalPaddingDirection = (
        TemporalPaddingDirection.RIGHT
    ),
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
    feature_dim: int = D,
) -> HistoricalSequenceInputs:
    if lengths is None:
        lengths = torch.tensor(
            [2, 0, 4, 1, 3, 2],
            dtype=torch.long,
        )

    lengths = lengths.to(
        device=device
    )
    node_count = int(lengths.numel())
    mask = _mask_from_lengths(
        lengths,
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

    for node_index in range(node_count):
        valid_positions = torch.nonzero(
            mask[node_index],
            as_tuple=False,
        ).flatten()
        logical_length = int(valid_positions.numel())

        for logical_index, temporal_index in enumerate(
            valid_positions.tolist()
        ):
            values[
                node_index,
                temporal_index,
            ] = torch.arange(
                next_value,
                next_value + feature_dim,
                dtype=dtype,
                device=device,
            )
            coordinates[
                node_index,
                temporal_index,
            ] = float(
                logical_index - logical_length
            )
            next_value += float(feature_dim)

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
                "recurrent-preprocessing-node-axis-v1"
            ),
        ),
        feature_axis=TemporalFeatureAxis(
            feature_names=tuple(
                f"feature-{index}"
                for index in range(feature_dim)
            ),
            source_fingerprint=(
                "recurrent-preprocessing-feature-axis-v1"
            ),
        ),
        temporal_coordinates=RelativeTemporalCoordinates(
            values=coordinates,
            unit="months",
        ),
        source_provenance=MemorySourceProvenance(
            source_name="recurrent-preprocessing-panel",
            source_kind="historical-node-sequence",
            source_fingerprint=(
                "recurrent-preprocessing-source-v1"
            ),
            preprocessing_fingerprint=(
                "recurrent-preprocessing-pipeline-v1"
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


def _config(
    *,
    cell_kind: RecurrentCellKind = RecurrentCellKind.GRU,
    input_dim: int = D,
    hidden_dim: int = H,
    num_layers: int = 2,
    dropout: float = 0.0,
    bidirectional: bool = True,
    use_bias: bool = True,
    input_projection_dim: int | None = None,
    layer_normalization: bool = False,
    pack_sequences: bool = True,
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


def _kernel(
    config: RecurrentSequenceEncoderConfig,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> nn.GRU | nn.LSTM:
    effective_input_dim = (
        config.input_projection_dim
        if config.input_projection_dim is not None
        else config.input_dim
    )
    kernel_type = (
        nn.GRU
        if config.cell_kind == RecurrentCellKind.GRU
        else nn.LSTM
    )
    kernel = kernel_type(
        input_size=effective_input_dim,
        hidden_size=config.hidden_dim,
        num_layers=config.num_layers,
        bias=config.use_bias,
        batch_first=True,
        dropout=config.dropout,
        bidirectional=config.bidirectional,
    )

    return kernel.to(
        device=device,
        dtype=dtype,
    )


def _adapter_config(
    *,
    projection_dim: int | None = 4,
    layer_normalization: bool = True,
    use_bias: bool = True,
) -> RecurrentSequenceEncoderConfig:
    return _config(
        input_projection_dim=projection_dim,
        layer_normalization=layer_normalization,
        use_bias=use_bias,
    )


def _canonical_batch(
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
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
        dtype=dtype,
        device=device,
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
            device=device,
        ),
        lengths=torch.tensor(
            [2, 1],
            dtype=torch.long,
            device=device,
        ),
        nonempty_node_indices=torch.tensor(
            [0, 3],
            dtype=torch.long,
            device=device,
        ),
        source_node_count=4,
        original_padding_direction=(
            TemporalPaddingDirection.LEFT
        ),
    )


class _CompleteRecurrentModule(nn.Module):
    def __init__(
        self,
        config: RecurrentSequenceEncoderConfig,
    ) -> None:
        super().__init__()
        self.adapter = RecurrentInputAdapter(
            config
        )
        self.kernel = _kernel(
            config,
            dtype=torch.float32,
        )


# =============================================================================
# Module identity, aliases, exports, and boundaries
# =============================================================================


@pytest.mark.parametrize(
    "value",
    (
        RECURRENT_HISTORY_LENGTHS_IMPLEMENTATION_VERSION,
        RECURRENT_HISTORY_LENGTHS_COMPONENT_NAME,
        RECURRENT_HISTORY_LENGTHS_OPERATION_NAME,
        RECURRENT_HISTORY_LENGTHS_SORT_POLICY,
        RECURRENT_HISTORY_LENGTHS_ZERO_POLICY_SOURCE,
        RECURRENT_HISTORY_LENGTHS_SCIENTIFIC_INTERPRETATION,
        RECURRENT_INPUT_ADAPTER_IMPLEMENTATION_VERSION,
        RECURRENT_INPUT_ADAPTER_COMPONENT_NAME,
        RECURRENT_INPUT_ADAPTER_COMPONENT_KIND,
        RECURRENT_INPUT_ADAPTER_OPERATION_NAME,
        RECURRENT_INPUT_ADAPTER_OUTPUT_STAGE,
        RECURRENT_INPUT_ADAPTER_PROJECTION_POLICY,
        RECURRENT_INPUT_ADAPTER_BIAS_POLICY,
        RECURRENT_INPUT_ADAPTER_NORMALIZATION_POLICY,
        RECURRENT_INPUT_ADAPTER_PADDING_POLICY,
        RECURRENT_INPUT_ADAPTER_FEATURE_OBSERVATION_POLICY,
        RECURRENT_INPUT_ADAPTER_TEMPORAL_COORDINATE_POLICY,
        RECURRENT_INPUT_ADAPTER_HAZARD_POLICY,
        RECURRENT_INPUT_ADAPTER_SCIENTIFIC_INTERPRETATION,
        RECURRENT_INITIAL_STATE_IMPLEMENTATION_VERSION,
        RECURRENT_INITIAL_STATE_COMPONENT_NAME,
        RECURRENT_INITIAL_STATE_COMPONENT_KIND,
        RECURRENT_INITIAL_STATE_OPERATION_NAME,
        RECURRENT_INITIAL_STATE_POLICY,
        RECURRENT_INITIAL_STATE_FLAT_AXIS_ORDER,
        RECURRENT_INITIAL_STATE_CANONICAL_AXIS_ORDER,
        RECURRENT_INITIAL_STATE_SCIENTIFIC_INTERPRETATION,
    ),
)
def test_preprocessing_policy_strings_are_nonempty(
    value: str,
) -> None:
    assert isinstance(value, str)
    assert value.strip()


def test_boolean_preprocessing_policies_are_frozen() -> None:
    assert RECURRENT_INPUT_ADAPTER_TEMPORAL_INTERACTION is False
    assert (
        RECURRENT_INPUT_ADAPTER_LAYER_NORM_ELEMENTWISE_AFFINE
        is True
    )
    assert (
        RECURRENT_INITIAL_STATE_CALLER_SUPPLIED_SUPPORTED
        is False
    )
    assert RECURRENT_INITIAL_STATE_LEARNED_SUPPORTED is False
    assert RECURRENT_INITIAL_STATE_STATEFUL_SUPPORTED is False


def test_history_length_aliases_are_exact() -> None:
    assert derive_history_lengths is derive_recurrent_history_lengths
    assert partition_history_nodes is partition_recurrent_history_nodes
    assert stable_length_sort is stable_descending_length_permutation
    assert build_execution_metadata is (
        build_recurrent_execution_metadata
    )


def test_input_adapter_aliases_are_exact() -> None:
    assert InputAdapter is RecurrentInputAdapter
    assert build_input_adapter is build_recurrent_input_adapter
    assert adapt_input_values is adapt_recurrent_input_values


def test_initial_state_aliases_are_exact() -> None:
    assert build_state_layout is build_recurrent_state_layout
    assert build_zero_initial_state is (
        build_zero_recurrent_initial_state
    )
    assert build_zero_final_states is (
        build_zero_canonical_final_states
    )
    assert flatten_state is flatten_recurrent_state
    assert unflatten_state is unflatten_recurrent_state


@pytest.mark.parametrize(
    "module",
    (
        history_lengths_module,
        input_adapter_module,
        initial_state_module,
    ),
)
def test_preprocessing_module_all_is_unique_and_resolves(
    module,
) -> None:
    exported = module.__all__

    assert isinstance(exported, tuple)
    assert len(exported) == len(set(exported))

    for name in exported:
        assert hasattr(module, name)


@pytest.mark.parametrize(
    "module",
    (
        history_lengths_module,
        input_adapter_module,
        initial_state_module,
    ),
)
def test_preprocessing_modules_do_not_import_complete_encoders(
    module,
) -> None:
    source = inspect.getsource(module)

    forbidden = (
        "gru_encoder",
        "lstm_encoder",
        "recurrent_memory_encoder import",
        "diagnostics import",
        "sequence_packing import",
        "baseline_encoders",
        "transformer_encoder",
    )

    for fragment in forbidden:
        assert fragment not in source


# =============================================================================
# History lengths: derivation and validation
# =============================================================================


@pytest.mark.parametrize(
    "padding_direction",
    (
        TemporalPaddingDirection.RIGHT,
        TemporalPaddingDirection.LEFT,
    ),
)
def test_derive_lengths_is_padding_direction_invariant(
    padding_direction: TemporalPaddingDirection,
) -> None:
    history = _history(
        padding_direction=padding_direction
    )

    lengths = derive_recurrent_history_lengths(history)

    assert lengths.tolist() == [2, 0, 4, 1, 3, 2]
    assert lengths.dtype == torch.long
    assert lengths.device == history.device
    assert not lengths.requires_grad


def test_derive_lengths_returns_unaliased_tensor() -> None:
    history = _history()
    lengths = derive_recurrent_history_lengths(history)

    lengths[0] = 99

    assert history.valid_lengths.tolist() == [2, 0, 4, 1, 3, 2]


def test_derive_lengths_rejects_wrong_source_type() -> None:
    with pytest.raises(TypeError):
        derive_recurrent_history_lengths(object())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "lengths",
    (
        torch.tensor([1, 2, 0], dtype=torch.long),
        torch.tensor([4], dtype=torch.long),
        torch.tensor([0, 0], dtype=torch.long),
    ),
)
def test_validate_lengths_accepts_legal_vectors(
    lengths: torch.Tensor,
) -> None:
    validated = validate_recurrent_history_lengths(
        lengths,
        source_node_count=int(lengths.numel()),
        source_sequence_length=T,
    )

    assert torch.equal(validated, lengths)
    assert validated is not lengths


@pytest.mark.parametrize(
    (
        "lengths",
        "error_type",
    ),
    (
        (
            object(),
            TypeError,
        ),
        (
            torch.zeros(2, 1, dtype=torch.long),
            ValueError,
        ),
        (
            torch.zeros(2, dtype=torch.int32),
            ValueError,
        ),
        (
            torch.tensor([1, -1], dtype=torch.long),
            ValueError,
        ),
        (
            torch.empty(0, dtype=torch.long),
            ValueError,
        ),
    ),
)
def test_validate_lengths_rejects_invalid_vectors(
    lengths: object,
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type):
        validate_recurrent_history_lengths(
            lengths,  # type: ignore[arg-type]
            source_sequence_length=T,
        )


def test_validate_lengths_rejects_source_count_mismatch() -> None:
    with pytest.raises(ValueError):
        validate_recurrent_history_lengths(
            torch.tensor([1, 2], dtype=torch.long),
            source_node_count=3,
            source_sequence_length=T,
        )


def test_validate_lengths_rejects_length_above_source_horizon() -> None:
    with pytest.raises(ValueError):
        validate_recurrent_history_lengths(
            torch.tensor([T + 1], dtype=torch.long),
            source_sequence_length=T,
        )


@pytest.mark.parametrize(
    "policy",
    (
        HistoryZeroLengthPolicy.ALLOW_ZERO_HISTORY,
        "allow_zero_history",
    ),
)
def test_zero_history_policy_allows_zero_when_configured(
    policy: HistoryZeroLengthPolicy | str,
) -> None:
    validate_zero_history_policy(
        torch.tensor([2, 0, 1], dtype=torch.long),
        zero_length_policy=policy,
    )


@pytest.mark.parametrize(
    "policy",
    (
        HistoryZeroLengthPolicy.ERROR,
        "error",
    ),
)
def test_zero_history_policy_rejects_zero_under_error(
    policy: HistoryZeroLengthPolicy | str,
) -> None:
    with pytest.raises(ValueError):
        validate_zero_history_policy(
            torch.tensor([2, 0, 1], dtype=torch.long),
            zero_length_policy=policy,
        )


def test_zero_history_error_policy_accepts_all_nonempty() -> None:
    validate_zero_history_policy(
        torch.tensor([2, 4, 1], dtype=torch.long),
        zero_length_policy=HistoryZeroLengthPolicy.ERROR,
    )


def test_zero_history_policy_rejects_unknown_value() -> None:
    with pytest.raises(ValueError):
        validate_zero_history_policy(
            torch.tensor([1], dtype=torch.long),
            zero_length_policy="unknown",
        )


def test_validate_source_zero_history_policy_returns_exact_lengths() -> None:
    history = _history()
    lengths = validate_source_zero_history_policy(history)

    assert lengths.tolist() == [2, 0, 4, 1, 3, 2]


def test_validate_source_zero_history_policy_accepts_matching_supplied_lengths() -> None:
    history = _history()
    lengths = validate_source_zero_history_policy(
        history,
        history_lengths=history.valid_lengths,
    )

    assert torch.equal(lengths, history.valid_lengths)


def test_validate_source_zero_history_policy_rejects_mismatched_supplied_lengths() -> None:
    history = _history()
    wrong = history.valid_lengths.clone()
    wrong[0] = 1

    with pytest.raises(ValueError):
        validate_source_zero_history_policy(
            history,
            history_lengths=wrong,
        )


def test_validate_source_policy_defensively_rejects_mutated_zero_under_error() -> None:
    history = _history(
        lengths=torch.tensor(
            [2, 1, 4],
            dtype=torch.long,
        )
    )

    with torch.no_grad():
        history.timestep_mask[1] = False
        history.history[1] = 0

    with pytest.raises(ValueError):
        validate_source_zero_history_policy(history)


# =============================================================================
# History lengths: partitioning and gathering
# =============================================================================


def test_partition_nodes_returns_original_order() -> None:
    nonempty, zero = partition_recurrent_history_nodes(
        torch.tensor(
            [2, 0, 4, 1, 0, 2],
            dtype=torch.long,
        )
    )

    assert nonempty.tolist() == [0, 2, 3, 5]
    assert zero.tolist() == [1, 4]


@pytest.mark.parametrize(
    (
        "lengths",
        "expected_nonempty",
        "expected_zero",
    ),
    (
        (
            [1, 2, 3],
            [0, 1, 2],
            [],
        ),
        (
            [0, 0, 0],
            [],
            [0, 1, 2],
        ),
        (
            [1],
            [0],
            [],
        ),
        (
            [0],
            [],
            [0],
        ),
    ),
)
def test_partition_node_edge_cases(
    lengths: list[int],
    expected_nonempty: list[int],
    expected_zero: list[int],
) -> None:
    nonempty, zero = partition_recurrent_history_nodes(
        torch.tensor(lengths, dtype=torch.long)
    )

    assert nonempty.tolist() == expected_nonempty
    assert zero.tolist() == expected_zero


def test_gather_nonempty_lengths() -> None:
    lengths = torch.tensor(
        [2, 0, 4, 1, 3, 2],
        dtype=torch.long,
    )
    indices = torch.tensor(
        [0, 2, 3, 4, 5],
        dtype=torch.long,
    )

    gathered = gather_nonempty_history_lengths(
        lengths,
        indices,
    )

    assert gathered.tolist() == [2, 4, 1, 3, 2]


def test_gather_nonempty_lengths_accepts_empty_index() -> None:
    result = gather_nonempty_history_lengths(
        torch.tensor([0, 0], dtype=torch.long),
        torch.empty(0, dtype=torch.long),
    )

    assert result.shape == (0,)
    assert result.dtype == torch.long


@pytest.mark.parametrize(
    "indices",
    (
        torch.tensor([2, 0], dtype=torch.long),
        torch.tensor([0, 0], dtype=torch.long),
        torch.tensor([-1], dtype=torch.long),
        torch.tensor([6], dtype=torch.long),
        torch.tensor([1], dtype=torch.long),
        torch.tensor([0], dtype=torch.int32),
        torch.zeros(1, 1, dtype=torch.long),
    ),
)
def test_gather_nonempty_lengths_rejects_invalid_indices(
    indices: torch.Tensor,
) -> None:
    with pytest.raises(ValueError):
        gather_nonempty_history_lengths(
            torch.tensor(
                [2, 0, 4, 1, 3, 2],
                dtype=torch.long,
            ),
            indices,
        )


def test_gather_nonempty_lengths_rejects_non_tensor_indices() -> None:
    with pytest.raises(TypeError):
        gather_nonempty_history_lengths(
            torch.tensor([1], dtype=torch.long),
            object(),  # type: ignore[arg-type]
        )


# =============================================================================
# History lengths: sorting and permutations
# =============================================================================


@pytest.mark.parametrize(
    (
        "lengths",
        "expected",
    ),
    (
        (
            [],
            True,
        ),
        (
            [1],
            True,
        ),
        (
            [4, 3, 3, 1],
            True,
        ),
        (
            [4, 4, 4],
            True,
        ),
        (
            [2, 4, 1],
            False,
        ),
        (
            [3, 2, 4],
            False,
        ),
    ),
)
def test_lengths_are_nonincreasing(
    lengths: list[int],
    expected: bool,
) -> None:
    result = lengths_are_nonincreasing(
        torch.tensor(lengths, dtype=torch.long)
    )

    assert result is expected


def test_stable_descending_sort_preserves_equal_length_order() -> None:
    lengths = torch.tensor(
        [2, 4, 1, 3, 2, 2],
        dtype=torch.long,
    )

    permutation = stable_descending_length_permutation(lengths)

    assert permutation.tolist() == [1, 3, 0, 4, 5, 2]
    assert lengths.index_select(0, permutation).tolist() == [
        4,
        3,
        2,
        2,
        2,
        1,
    ]


def test_stable_descending_sort_accepts_empty() -> None:
    permutation = stable_descending_length_permutation(
        torch.empty(0, dtype=torch.long)
    )

    assert permutation.shape == (0,)


def test_stable_descending_sort_rejects_nonpositive_lengths() -> None:
    with pytest.raises(ValueError):
        stable_descending_length_permutation(
            torch.tensor([2, 0, 1], dtype=torch.long)
        )


@pytest.mark.parametrize(
    "permutation",
    (
        [0],
        [1, 0],
        [2, 0, 3, 1],
        [3, 2, 1, 0],
    ),
)
def test_inverse_permutation_round_trip(
    permutation: list[int],
) -> None:
    forward = torch.tensor(
        permutation,
        dtype=torch.long,
    )
    inverse = inverse_permutation(forward)
    expected = torch.arange(
        len(permutation),
        dtype=torch.long,
    )

    assert torch.equal(
        forward.index_select(0, inverse),
        expected,
    )
    assert torch.equal(
        inverse.index_select(0, forward),
        expected,
    )


def test_inverse_permutation_accepts_empty() -> None:
    inverse = inverse_permutation(
        torch.empty(0, dtype=torch.long)
    )

    assert inverse.shape == (0,)


@pytest.mark.parametrize(
    "permutation",
    (
        torch.tensor([0, 0], dtype=torch.long),
        torch.tensor([0, 2], dtype=torch.long),
        torch.tensor([0, 1], dtype=torch.int32),
        torch.zeros(1, 1, dtype=torch.long),
    ),
)
def test_inverse_permutation_rejects_invalid_input(
    permutation: torch.Tensor,
) -> None:
    with pytest.raises(ValueError):
        inverse_permutation(permutation)


def test_inverse_permutation_rejects_non_tensor() -> None:
    with pytest.raises(TypeError):
        inverse_permutation(object())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "execution_path",
    (
        RecurrentExecutionPath.REFERENCE,
        "reference",
    ),
)
def test_reference_sort_plan_is_identity(
    execution_path: RecurrentExecutionPath | str,
) -> None:
    lengths = torch.tensor(
        [2, 4, 1, 3],
        dtype=torch.long,
    )

    sorted_to_nonempty, inverse, applied = (
        build_recurrent_sort_permutations(
            lengths,
            execution_path=execution_path,
            enforce_sorted_lengths=False,
        )
    )

    assert sorted_to_nonempty.tolist() == [0, 1, 2, 3]
    assert inverse.tolist() == [0, 1, 2, 3]
    assert applied is False


def test_reference_sort_plan_rejects_enforce_sorted_lengths() -> None:
    with pytest.raises(ValueError):
        build_recurrent_sort_permutations(
            torch.tensor([2, 4], dtype=torch.long),
            execution_path="reference",
            enforce_sorted_lengths=True,
        )


def test_packed_sort_plan_sorts_arbitrary_order() -> None:
    sorted_to_nonempty, inverse, applied = (
        build_recurrent_sort_permutations(
            torch.tensor(
                [2, 4, 1, 3, 2],
                dtype=torch.long,
            ),
            execution_path="packed",
            enforce_sorted_lengths=False,
        )
    )

    assert sorted_to_nonempty.tolist() == [1, 3, 0, 4, 2]
    assert inverse.tolist() == [2, 0, 4, 1, 3]
    assert applied is True


def test_packed_sort_plan_reports_false_when_already_sorted() -> None:
    sorted_to_nonempty, inverse, applied = (
        build_recurrent_sort_permutations(
            torch.tensor(
                [4, 3, 3, 1],
                dtype=torch.long,
            ),
            execution_path="packed",
            enforce_sorted_lengths=False,
        )
    )

    assert sorted_to_nonempty.tolist() == [0, 1, 2, 3]
    assert inverse.tolist() == [0, 1, 2, 3]
    assert applied is False


def test_strict_packed_sort_plan_accepts_already_sorted() -> None:
    result = build_recurrent_sort_permutations(
        torch.tensor(
            [4, 3, 3, 1],
            dtype=torch.long,
        ),
        execution_path="packed",
        enforce_sorted_lengths=True,
    )

    assert result[0].tolist() == [0, 1, 2, 3]
    assert result[1].tolist() == [0, 1, 2, 3]
    assert result[2] is False


def test_strict_packed_sort_plan_rejects_unsorted() -> None:
    with pytest.raises(ValueError):
        build_recurrent_sort_permutations(
            torch.tensor([3, 4, 1], dtype=torch.long),
            execution_path="packed",
            enforce_sorted_lengths=True,
        )


@pytest.mark.parametrize(
    "execution_path",
    (
        "packed",
        "reference",
    ),
)
def test_sort_plan_supports_empty_nonempty_axis(
    execution_path: str,
) -> None:
    sorted_to_nonempty, inverse, applied = (
        build_recurrent_sort_permutations(
            torch.empty(0, dtype=torch.long),
            execution_path=execution_path,
            enforce_sorted_lengths=False,
        )
    )

    assert sorted_to_nonempty.numel() == 0
    assert inverse.numel() == 0
    assert applied is False


# =============================================================================
# History lengths: metadata assembly
# =============================================================================


def test_build_metadata_from_lengths_packed() -> None:
    metadata = build_recurrent_execution_metadata_from_lengths(
        torch.tensor(
            [2, 0, 4, 1, 3, 2],
            dtype=torch.long,
        ),
        source_sequence_length=T,
        zero_length_policy="allow_zero_history",
        original_padding_direction="right",
        execution_path="packed",
        enforce_sorted_lengths=False,
    )

    assert metadata.execution_path == RecurrentExecutionPath.PACKED
    assert metadata.nonempty_node_indices.tolist() == [
        0,
        2,
        3,
        4,
        5,
    ]
    assert metadata.zero_history_node_indices.tolist() == [1]
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
    assert metadata.sort_was_applied


def test_build_metadata_from_lengths_reference() -> None:
    metadata = build_recurrent_execution_metadata_from_lengths(
        torch.tensor(
            [2, 0, 4, 1, 3, 2],
            dtype=torch.long,
        ),
        source_sequence_length=T,
        zero_length_policy="allow_zero_history",
        original_padding_direction="left",
        execution_path="reference",
    )

    assert metadata.execution_path == RecurrentExecutionPath.REFERENCE
    assert metadata.identity_permutation
    assert metadata.original_padding_direction == (
        TemporalPaddingDirection.LEFT
    )


def test_build_metadata_from_lengths_all_zero() -> None:
    metadata = build_recurrent_execution_metadata_from_lengths(
        torch.zeros(3, dtype=torch.long),
        source_sequence_length=T,
        zero_length_policy="allow_zero_history",
        original_padding_direction="right",
        execution_path="packed",
    )

    assert metadata.all_zero_history
    assert metadata.nonempty_node_indices.numel() == 0
    assert metadata.zero_history_node_indices.tolist() == [0, 1, 2]
    assert not metadata.sort_was_applied


def test_build_metadata_from_lengths_error_policy_rejects_zero() -> None:
    with pytest.raises(ValueError):
        build_recurrent_execution_metadata_from_lengths(
            torch.tensor([2, 0, 1], dtype=torch.long),
            source_sequence_length=T,
            zero_length_policy="error",
            original_padding_direction="right",
            execution_path="packed",
        )


@pytest.mark.parametrize(
    (
        "pack_sequences",
        "enforce_sorted_lengths",
        "expected_path",
    ),
    (
        (
            True,
            False,
            RecurrentExecutionPath.PACKED,
        ),
        (
            False,
            False,
            RecurrentExecutionPath.REFERENCE,
        ),
    ),
)
def test_build_metadata_from_source_and_config(
    pack_sequences: bool,
    enforce_sorted_lengths: bool,
    expected_path: RecurrentExecutionPath,
) -> None:
    history = _history()
    config = _config(
        pack_sequences=pack_sequences,
        enforce_sorted_lengths=enforce_sorted_lengths,
    )

    metadata = build_recurrent_execution_metadata(
        history,
        config,
    )

    assert metadata.execution_path == expected_path
    assert torch.equal(
        metadata.history_lengths,
        history.valid_lengths,
    )
    assert metadata.original_padding_direction == (
        history.padding_direction
    )


def test_build_metadata_strict_mode_ignores_interspersed_zero_nodes() -> None:
    history = _history(
        lengths=torch.tensor(
            [4, 0, 3, 0, 3, 1],
            dtype=torch.long,
        )
    )
    config = _config(
        enforce_sorted_lengths=True
    )

    metadata = build_recurrent_execution_metadata(
        history,
        config,
    )

    assert metadata.nonempty_history_lengths.tolist() == [
        4,
        3,
        3,
        1,
    ]
    assert metadata.identity_permutation


def test_build_metadata_strict_mode_rejects_unsorted_nonempty_nodes() -> None:
    history = _history(
        lengths=torch.tensor(
            [3, 0, 4],
            dtype=torch.long,
        )
    )
    config = _config(
        enforce_sorted_lengths=True
    )

    with pytest.raises(ValueError):
        build_recurrent_execution_metadata(
            history,
            config,
        )


def test_config_rejects_reference_strict_sort_combination() -> None:
    with pytest.raises(ValueError):
        _config(
            pack_sequences=False,
            enforce_sorted_lengths=True,
        )


# =============================================================================
# Input adapter: construction and architecture
# =============================================================================


@pytest.mark.parametrize(
    (
        "projection_dim",
        "normalization",
        "use_bias",
        "expected_output_dim",
        "projection_enabled",
        "normalization_enabled",
        "projection_bias_enabled",
    ),
    (
        (
            None,
            False,
            True,
            D,
            False,
            False,
            False,
        ),
        (
            None,
            True,
            True,
            D,
            False,
            True,
            False,
        ),
        (
            4,
            False,
            True,
            4,
            True,
            False,
            True,
        ),
        (
            4,
            False,
            False,
            4,
            True,
            False,
            False,
        ),
        (
            4,
            True,
            True,
            4,
            True,
            True,
            True,
        ),
    ),
)
def test_input_adapter_configuration_properties(
    projection_dim: int | None,
    normalization: bool,
    use_bias: bool,
    expected_output_dim: int,
    projection_enabled: bool,
    normalization_enabled: bool,
    projection_bias_enabled: bool,
) -> None:
    adapter = RecurrentInputAdapter(
        _adapter_config(
            projection_dim=projection_dim,
            layer_normalization=normalization,
            use_bias=use_bias,
        )
    )

    assert adapter.input_dim == D
    assert adapter.effective_input_dim == expected_output_dim
    assert adapter.output_dim == expected_output_dim
    assert adapter.projection_enabled is projection_enabled
    assert (
        adapter.layer_normalization_enabled
        is normalization_enabled
    )
    assert (
        adapter.projection_bias_enabled
        is projection_bias_enabled
    )
    assert adapter.has_temporal_interaction is False


def test_input_adapter_builder() -> None:
    config = _adapter_config()
    adapter = build_recurrent_input_adapter(config)

    assert isinstance(adapter, RecurrentInputAdapter)
    assert adapter.config is config


def test_identity_adapter_has_no_parameters() -> None:
    adapter = RecurrentInputAdapter(
        _adapter_config(
            projection_dim=None,
            layer_normalization=False,
        )
    )

    assert adapter.parameter_count == 0
    assert adapter.trainable_parameter_count == 0
    assert isinstance(adapter.projection, nn.Identity)
    assert isinstance(adapter.normalization, nn.Identity)


def test_projection_bias_follows_use_bias() -> None:
    with_bias = RecurrentInputAdapter(
        _adapter_config(
            projection_dim=4,
            layer_normalization=False,
            use_bias=True,
        )
    )
    without_bias = RecurrentInputAdapter(
        _adapter_config(
            projection_dim=4,
            layer_normalization=False,
            use_bias=False,
        )
    )

    assert isinstance(with_bias.projection, nn.Linear)
    assert with_bias.projection.bias is not None
    assert isinstance(without_bias.projection, nn.Linear)
    assert without_bias.projection.bias is None


def test_layer_norm_uses_affine_parameters() -> None:
    adapter = RecurrentInputAdapter(
        _adapter_config(
            projection_dim=None,
            layer_normalization=True,
        )
    )

    assert isinstance(adapter.normalization, nn.LayerNorm)
    assert adapter.normalization.elementwise_affine is True
    assert adapter.normalization.weight is not None
    assert adapter.normalization.bias is not None


def test_adapter_architecture_metadata_is_complete() -> None:
    adapter = RecurrentInputAdapter(
        _adapter_config()
    )
    metadata = adapter.architecture_metadata()

    assert metadata["component_name"] == (
        RECURRENT_INPUT_ADAPTER_COMPONENT_NAME
    )
    assert metadata["input_dim"] == D
    assert metadata["effective_input_dim"] == 4
    assert metadata["projection_enabled"] is True
    assert metadata["projection_bias_enabled"] is True
    assert metadata["layer_normalization_enabled"] is True
    assert metadata["temporal_interaction"] is False
    assert metadata["parameter_count"] == adapter.parameter_count


def test_adapter_rejects_wrong_config_type() -> None:
    with pytest.raises(TypeError):
        RecurrentInputAdapter(object())  # type: ignore[arg-type]


# =============================================================================
# Input adapter: forward semantics
# =============================================================================


@pytest.mark.parametrize(
    (
        "projection_dim",
        "normalization",
    ),
    (
        (
            None,
            False,
        ),
        (
            None,
            True,
        ),
        (
            4,
            False,
        ),
        (
            4,
            True,
        ),
    ),
)
def test_adapter_forward_shape_dtype_and_padding(
    projection_dim: int | None,
    normalization: bool,
) -> None:
    config = _adapter_config(
        projection_dim=projection_dim,
        layer_normalization=normalization,
    )
    adapter = RecurrentInputAdapter(config).double()
    batch = _canonical_batch()

    output = adapter(
        batch.values,
        batch.timestep_mask,
    )

    expected_dim = (
        projection_dim
        if projection_dim is not None
        else D
    )
    assert output.shape == (2, 3, expected_dim)
    assert output.dtype == torch.float64
    assert output.device.type == "cpu"
    assert torch.isfinite(output).all()
    assert torch.count_nonzero(
        output[~batch.timestep_mask]
    ).item() == 0


def test_identity_adapter_preserves_valid_values_exactly() -> None:
    adapter = RecurrentInputAdapter(
        _adapter_config(
            projection_dim=None,
            layer_normalization=False,
        )
    ).double()
    batch = _canonical_batch()

    output = adapter(
        batch.values,
        batch.timestep_mask,
    )

    assert torch.equal(output, batch.values)


def test_projection_bias_cannot_leak_into_padding() -> None:
    adapter = RecurrentInputAdapter(
        _adapter_config(
            projection_dim=4,
            layer_normalization=False,
            use_bias=True,
        )
    ).double()
    batch = _canonical_batch()

    with torch.no_grad():
        adapter.projection.weight.zero_()
        adapter.projection.bias.fill_(7.0)

    output = adapter(
        batch.values,
        batch.timestep_mask,
    )

    assert torch.all(
        output[batch.timestep_mask] == 7.0
    )
    assert torch.count_nonzero(
        output[~batch.timestep_mask]
    ).item() == 0


def test_layer_norm_affine_cannot_leak_into_padding() -> None:
    adapter = RecurrentInputAdapter(
        _adapter_config(
            projection_dim=None,
            layer_normalization=True,
        )
    ).double()
    batch = _canonical_batch()

    with torch.no_grad():
        adapter.normalization.weight.fill_(2.0)
        adapter.normalization.bias.fill_(5.0)

    output = adapter(
        batch.values,
        batch.timestep_mask,
    )

    assert torch.count_nonzero(
        output[~batch.timestep_mask]
    ).item() == 0


def test_adapter_checks_pre_mask_finiteness() -> None:
    adapter = RecurrentInputAdapter(
        _adapter_config(
            projection_dim=4,
            layer_normalization=False,
            use_bias=False,
        )
    ).double()
    batch = _canonical_batch()

    with torch.no_grad():
        adapter.projection.weight[0, 0] = float("nan")

    with pytest.raises(ValueError):
        adapter(
            batch.values,
            batch.timestep_mask,
        )


def test_adapter_rejects_nonfinite_raw_input_even_at_padding() -> None:
    adapter = RecurrentInputAdapter(
        _adapter_config(
            projection_dim=None,
            layer_normalization=False,
        )
    )
    batch = _canonical_batch(
        dtype=torch.float32
    )
    values = batch.values.clone()
    values[0, 2, 0] = float("nan")

    with pytest.raises(ValueError):
        adapter(values, batch.timestep_mask)


def test_exact_zero_padding_uses_where_semantics() -> None:
    values = torch.tensor(
        [
            [
                [1.0],
                [float("nan")],
                [float("inf")],
            ]
        ]
    )
    mask = torch.tensor(
        [
            [True, False, False]
        ]
    )

    output = exact_zero_recurrent_padding(
        values,
        mask,
    )

    assert output[0, 0, 0].item() == 1.0
    assert output[0, 1, 0].item() == 0.0
    assert output[0, 2, 0].item() == 0.0


@pytest.mark.parametrize(
    (
        "values",
        "mask",
        "error_type",
    ),
    (
        (
            object(),
            torch.ones(1, 1, dtype=torch.bool),
            TypeError,
        ),
        (
            torch.zeros(2, 2),
            torch.ones(2, 2, dtype=torch.bool),
            ValueError,
        ),
        (
            torch.zeros(2, 3, 2, dtype=torch.long),
            torch.ones(2, 3, dtype=torch.bool),
            ValueError,
        ),
        (
            torch.zeros(2, 3, 2),
            object(),
            TypeError,
        ),
        (
            torch.zeros(2, 3, 2),
            torch.ones(2, 3),
            ValueError,
        ),
        (
            torch.zeros(2, 3, 2),
            torch.ones(2, 2, dtype=torch.bool),
            ValueError,
        ),
    ),
)
def test_adapter_rejects_invalid_values_or_mask(
    values: object,
    mask: object,
    error_type: type[Exception],
) -> None:
    adapter = RecurrentInputAdapter(
        _adapter_config(
            projection_dim=None,
            layer_normalization=False,
        )
    )

    with pytest.raises(error_type):
        adapter(
            values,  # type: ignore[arg-type]
            mask,  # type: ignore[arg-type]
        )


def test_adapter_rejects_wrong_feature_width() -> None:
    adapter = RecurrentInputAdapter(
        _adapter_config()
    )

    with pytest.raises(ValueError):
        adapter(
            torch.zeros(2, 3, D + 1),
            torch.ones(2, 3, dtype=torch.bool),
        )


def test_adapter_supports_empty_nonempty_axis() -> None:
    adapter = RecurrentInputAdapter(
        _adapter_config()
    ).double()

    output = adapter(
        torch.empty(0, 3, D, dtype=torch.float64),
        torch.empty(0, 3, dtype=torch.bool),
    )

    assert output.shape == (0, 3, 4)


def test_adapter_functional_wrapper_matches_module() -> None:
    adapter = RecurrentInputAdapter(
        _adapter_config()
    ).double()
    batch = _canonical_batch()

    direct = adapter(
        batch.values,
        batch.timestep_mask,
    )
    wrapped = adapt_recurrent_input_values(
        adapter,
        batch.values,
        batch.timestep_mask,
    )

    assert torch.equal(direct, wrapped)


def test_adapter_functional_wrapper_rejects_wrong_adapter() -> None:
    with pytest.raises(TypeError):
        adapt_recurrent_input_values(
            object(),  # type: ignore[arg-type]
            torch.zeros(1, 1, D),
            torch.ones(1, 1, dtype=torch.bool),
        )


# =============================================================================
# Input adapter: gradients and canonical-batch adaptation
# =============================================================================


@pytest.mark.parametrize(
    (
        "projection_dim",
        "normalization",
    ),
    (
        (
            None,
            False,
        ),
        (
            None,
            True,
        ),
        (
            4,
            False,
        ),
        (
            4,
            True,
        ),
    ),
)
def test_adapter_padding_receives_zero_input_gradient(
    projection_dim: int | None,
    normalization: bool,
) -> None:
    adapter = RecurrentInputAdapter(
        _adapter_config(
            projection_dim=projection_dim,
            layer_normalization=normalization,
        )
    ).double()
    batch = _canonical_batch(
        requires_grad=True
    )

    output = adapter(
        batch.values,
        batch.timestep_mask,
    )
    weights = torch.arange(
        1,
        output.numel() + 1,
        dtype=output.dtype,
        device=output.device,
    ).reshape_as(output)
    loss = (output * weights).sum()
    loss.backward()

    assert batch.values.grad is not None
    assert torch.count_nonzero(
        batch.values.grad[
            ~batch.timestep_mask
        ]
    ).item() == 0
    assert torch.isfinite(
        batch.values.grad[
            batch.timestep_mask
        ]
    ).all()


def test_adapter_gradients_reach_projection_and_layer_norm() -> None:
    adapter = RecurrentInputAdapter(
        _adapter_config(
            projection_dim=4,
            layer_normalization=True,
        )
    ).double()
    batch = _canonical_batch(
        requires_grad=True
    )

    output = adapter(
        batch.values,
        batch.timestep_mask,
    )
    weights = torch.arange(
        1,
        output.numel() + 1,
        dtype=output.dtype,
    ).reshape_as(output)
    (output * weights).sum().backward()

    assert adapter.projection.weight.grad is not None
    assert adapter.projection.bias.grad is not None
    assert adapter.normalization.weight.grad is not None
    assert adapter.normalization.bias.grad is not None
    assert torch.isfinite(adapter.projection.weight.grad).all()
    assert torch.isfinite(adapter.normalization.weight.grad).all()


def test_adapt_canonical_batch_preserves_metadata() -> None:
    adapter = RecurrentInputAdapter(
        _adapter_config()
    ).double()
    batch = _canonical_batch()

    adapted = adapter.adapt_canonical_batch(batch)

    assert adapted.values.shape == (2, 3, 4)
    assert adapted.value_stage == (
        RECURRENT_INPUT_ADAPTER_OUTPUT_STAGE
    )
    assert torch.equal(
        adapted.timestep_mask,
        batch.timestep_mask,
    )
    assert adapted.lengths.tolist() == [2, 1]
    assert adapted.nonempty_node_indices.tolist() == [0, 3]
    assert adapted.source_node_count == 4
    assert adapted.original_padding_direction == (
        TemporalPaddingDirection.LEFT
    )
    assert torch.count_nonzero(
        adapted.values[~adapted.timestep_mask]
    ).item() == 0


def test_adapt_canonical_batch_preserves_autograd() -> None:
    adapter = RecurrentInputAdapter(
        _adapter_config()
    ).double()
    batch = _canonical_batch(
        requires_grad=True
    )

    adapted = adapt_canonical_recurrent_batch(
        adapter,
        batch,
    )
    adapted.values.sum().backward()

    assert batch.values.grad is not None


def test_adapt_canonical_batch_rejects_wrong_batch_type() -> None:
    adapter = RecurrentInputAdapter(
        _adapter_config()
    )

    with pytest.raises(TypeError):
        adapter.adapt_canonical_batch(object())  # type: ignore[arg-type]


def test_adapt_canonical_batch_rejects_feature_width_mismatch() -> None:
    adapter = RecurrentInputAdapter(
        _config(
            input_dim=3,
            input_projection_dim=None,
        )
    )
    batch = _canonical_batch(
        dtype=torch.float32
    )

    with pytest.raises(ValueError):
        adapter.adapt_canonical_batch(batch)


# =============================================================================
# Input adapter: module/source compatibility preflight
# =============================================================================


def test_module_tensor_compatibility_accepts_matching_adapter() -> None:
    adapter = RecurrentInputAdapter(
        _adapter_config()
    ).double()
    batch = _canonical_batch()

    validate_recurrent_module_tensor_compatibility(
        adapter,
        batch.values,
        expected_input_dim=D,
        require_floating_parameter=True,
    )


def test_module_tensor_compatibility_accepts_parameterless_identity_adapter() -> None:
    adapter = RecurrentInputAdapter(
        _adapter_config(
            projection_dim=None,
            layer_normalization=False,
        )
    )
    batch = _canonical_batch(
        dtype=torch.float32
    )

    validate_recurrent_module_tensor_compatibility(
        adapter,
        batch.values,
        expected_input_dim=D,
        require_floating_parameter=False,
    )


def test_module_tensor_compatibility_can_require_parameter() -> None:
    adapter = RecurrentInputAdapter(
        _adapter_config(
            projection_dim=None,
            layer_normalization=False,
        )
    )
    batch = _canonical_batch(
        dtype=torch.float32
    )

    with pytest.raises(RuntimeError):
        validate_recurrent_module_tensor_compatibility(
            adapter,
            batch.values,
            expected_input_dim=D,
            require_floating_parameter=True,
        )


def test_module_tensor_compatibility_rejects_dtype_mismatch() -> None:
    adapter = RecurrentInputAdapter(
        _adapter_config()
    ).float()
    batch = _canonical_batch(
        dtype=torch.float64
    )

    with pytest.raises(ValueError):
        validate_recurrent_module_tensor_compatibility(
            adapter,
            batch.values,
            expected_input_dim=D,
        )


def test_module_tensor_compatibility_rejects_wrong_feature_width() -> None:
    adapter = RecurrentInputAdapter(
        _adapter_config()
    )

    with pytest.raises(ValueError):
        validate_recurrent_module_tensor_compatibility(
            adapter,
            torch.zeros(2, 3, D + 1),
            expected_input_dim=D,
        )


def test_source_compatibility_accepts_complete_encoder_module() -> None:
    config = _adapter_config()
    complete = _CompleteRecurrentModule(config).double()
    history = _history()

    validate_recurrent_module_source_compatibility(
        complete,
        history,
        expected_input_dim=D,
    )


def test_source_compatibility_rejects_dtype_mismatch() -> None:
    config = _adapter_config()
    complete = _CompleteRecurrentModule(config).float()
    history = _history(
        dtype=torch.float64
    )

    with pytest.raises(ValueError):
        validate_recurrent_module_source_compatibility(
            complete,
            history,
            expected_input_dim=D,
        )


def test_source_compatibility_rejects_feature_width_mismatch() -> None:
    config = _adapter_config()
    complete = _CompleteRecurrentModule(config).double()
    history = _history(
        feature_dim=D + 1
    )

    with pytest.raises(ValueError):
        validate_recurrent_module_source_compatibility(
            complete,
            history,
            expected_input_dim=D,
        )


def test_source_compatibility_rechecks_mutated_padding() -> None:
    config = _adapter_config()
    complete = _CompleteRecurrentModule(config).double()
    history = _history()

    with torch.no_grad():
        padded_position = torch.nonzero(
            ~history.timestep_mask,
            as_tuple=False,
        )[0]
        node_index = int(padded_position[0])
        temporal_index = int(padded_position[1])
        history.history[
            node_index,
            temporal_index,
            0,
        ] = 1.0

    with pytest.raises(ValueError):
        validate_recurrent_module_source_compatibility(
            complete,
            history,
            expected_input_dim=D,
        )


# =============================================================================
# Initial state: layout and shape helpers
# =============================================================================


@pytest.mark.parametrize(
    (
        "cell_kind",
        "num_layers",
        "bidirectional",
        "hidden_dim",
        "expected_directions",
    ),
    (
        (
            RecurrentCellKind.GRU,
            1,
            False,
            3,
            1,
        ),
        (
            RecurrentCellKind.GRU,
            3,
            True,
            5,
            2,
        ),
        (
            RecurrentCellKind.LSTM,
            2,
            False,
            4,
            1,
        ),
        (
            RecurrentCellKind.LSTM,
            2,
            True,
            4,
            2,
        ),
    ),
)
def test_build_state_layout_from_config(
    cell_kind: RecurrentCellKind,
    num_layers: int,
    bidirectional: bool,
    hidden_dim: int,
    expected_directions: int,
) -> None:
    config = _config(
        cell_kind=cell_kind,
        num_layers=num_layers,
        bidirectional=bidirectional,
        hidden_dim=hidden_dim,
    )

    layout = build_recurrent_state_layout(config)

    assert isinstance(layout, RecurrentStateLayout)
    assert layout.num_layers == num_layers
    assert layout.num_directions == expected_directions
    assert layout.hidden_dim == hidden_dim


@pytest.mark.parametrize(
    (
        "num_layers",
        "bidirectional",
        "batch_size",
        "expected",
    ),
    (
        (
            1,
            False,
            0,
            (1, 0, H),
        ),
        (
            1,
            True,
            4,
            (2, 4, H),
        ),
        (
            3,
            False,
            2,
            (3, 2, H),
        ),
        (
            3,
            True,
            5,
            (6, 5, H),
        ),
    ),
)
def test_flat_state_shape(
    num_layers: int,
    bidirectional: bool,
    batch_size: int,
    expected: tuple[int, int, int],
) -> None:
    config = _config(
        num_layers=num_layers,
        bidirectional=bidirectional,
    )

    assert recurrent_flat_state_shape(
        config,
        batch_size=batch_size,
    ) == expected


@pytest.mark.parametrize(
    (
        "num_layers",
        "bidirectional",
        "node_count",
        "expected",
    ),
    (
        (
            1,
            False,
            0,
            (1, 1, 0, H),
        ),
        (
            1,
            True,
            4,
            (1, 2, 4, H),
        ),
        (
            3,
            False,
            2,
            (3, 1, 2, H),
        ),
        (
            3,
            True,
            5,
            (3, 2, 5, H),
        ),
    ),
)
def test_canonical_state_shape(
    num_layers: int,
    bidirectional: bool,
    node_count: int,
    expected: tuple[int, int, int, int],
) -> None:
    config = _config(
        num_layers=num_layers,
        bidirectional=bidirectional,
    )

    assert recurrent_canonical_state_shape(
        config,
        node_count=node_count,
    ) == expected


@pytest.mark.parametrize(
    "count",
    (
        -1,
        True,
    ),
)
def test_state_shape_helpers_reject_invalid_counts(
    count: int,
) -> None:
    config = _config()

    with pytest.raises(
        (
            TypeError,
            ValueError,
        )
    ):
        recurrent_flat_state_shape(
            config,
            batch_size=count,
        )

    with pytest.raises(
        (
            TypeError,
            ValueError,
        )
    ):
        recurrent_canonical_state_shape(
            config,
            node_count=count,
        )


# =============================================================================
# Initial state: kernel compatibility
# =============================================================================


@pytest.mark.parametrize(
    "cell_kind",
    (
        RecurrentCellKind.GRU,
        RecurrentCellKind.LSTM,
    ),
)
def test_matching_kernel_configuration_is_accepted(
    cell_kind: RecurrentCellKind,
) -> None:
    config = _config(
        cell_kind=cell_kind,
        input_projection_dim=4,
        num_layers=2,
        dropout=0.2,
        bidirectional=True,
        use_bias=False,
    )
    kernel = _kernel(config)

    validate_recurrent_kernel_configuration(
        kernel,
        config,
    )


def test_kernel_configuration_rejects_wrong_cell_type() -> None:
    gru_config = _config(
        cell_kind=RecurrentCellKind.GRU
    )
    lstm = _kernel(
        _config(
            cell_kind=RecurrentCellKind.LSTM
        )
    )

    with pytest.raises(TypeError):
        validate_recurrent_kernel_configuration(
            lstm,
            gru_config,
        )


@pytest.mark.parametrize(
    "mismatch",
    (
        "input_size",
        "hidden_size",
        "num_layers",
        "bias",
        "batch_first",
        "bidirectional",
        "dropout",
    ),
)
def test_kernel_configuration_rejects_architecture_mismatch(
    mismatch: str,
) -> None:
    config = _config(
        cell_kind=RecurrentCellKind.GRU,
        input_projection_dim=4,
        hidden_dim=3,
        num_layers=2,
        dropout=0.2,
        bidirectional=True,
        use_bias=True,
    )
    kwargs = {
        "input_size": 4,
        "hidden_size": 3,
        "num_layers": 2,
        "bias": True,
        "batch_first": True,
        "dropout": 0.2,
        "bidirectional": True,
    }

    if mismatch == "input_size":
        kwargs["input_size"] = 5
    elif mismatch == "hidden_size":
        kwargs["hidden_size"] = 4
    elif mismatch == "num_layers":
        kwargs["num_layers"] = 3
    elif mismatch == "bias":
        kwargs["bias"] = False
    elif mismatch == "batch_first":
        kwargs["batch_first"] = False
    elif mismatch == "bidirectional":
        kwargs["bidirectional"] = False
    elif mismatch == "dropout":
        kwargs["dropout"] = 0.1

    kernel = nn.GRU(**kwargs)

    with pytest.raises(ValueError):
        validate_recurrent_kernel_configuration(
            kernel,
            config,
        )


def test_kernel_configuration_rejects_lstm_projection() -> None:
    config = _config(
        cell_kind=RecurrentCellKind.LSTM,
        hidden_dim=5,
        bidirectional=False,
        num_layers=1,
    )
    kernel = nn.LSTM(
        input_size=D,
        hidden_size=5,
        proj_size=3,
        batch_first=True,
    )

    with pytest.raises(ValueError):
        validate_recurrent_kernel_configuration(
            kernel,
            config,
        )


@pytest.mark.parametrize(
    (
        "dtype",
        "expected_dtype",
    ),
    (
        (
            torch.float32,
            torch.float32,
        ),
        (
            torch.float64,
            torch.float64,
        ),
    ),
)
def test_infer_kernel_device_dtype(
    dtype: torch.dtype,
    expected_dtype: torch.dtype,
) -> None:
    config = _config()
    kernel = _kernel(
        config,
        dtype=dtype,
    )

    device, inferred_dtype = (
        infer_recurrent_kernel_device_dtype(
            kernel
        )
    )

    assert device.type == "cpu"
    assert inferred_dtype == expected_dtype


def test_infer_kernel_device_dtype_rejects_parameterless_module() -> None:
    with pytest.raises(RuntimeError):
        infer_recurrent_kernel_device_dtype(
            nn.Identity()
        )


# =============================================================================
# Initial state: exact-zero construction
# =============================================================================


@pytest.mark.parametrize(
    (
        "cell_kind",
        "dtype",
        "batch_size",
    ),
    (
        (
            RecurrentCellKind.GRU,
            torch.float32,
            0,
        ),
        (
            RecurrentCellKind.GRU,
            torch.float64,
            5,
        ),
        (
            RecurrentCellKind.LSTM,
            torch.float32,
            1,
        ),
        (
            RecurrentCellKind.LSTM,
            torch.float64,
            4,
        ),
    ),
)
def test_generic_zero_initial_state(
    cell_kind: RecurrentCellKind,
    dtype: torch.dtype,
    batch_size: int,
) -> None:
    config = _config(
        cell_kind=cell_kind,
        num_layers=2,
        bidirectional=True,
    )
    kernel = _kernel(
        config,
        dtype=dtype,
    )

    state = build_zero_recurrent_initial_state(
        kernel,
        config,
        batch_size=batch_size,
    )

    expected_shape = (
        4,
        batch_size,
        H,
    )

    if cell_kind == RecurrentCellKind.GRU:
        assert isinstance(state, torch.Tensor)
        assert state.shape == expected_shape
        assert state.dtype == dtype
        assert torch.count_nonzero(state).item() == 0
    else:
        assert isinstance(state, tuple)
        assert len(state) == 2

        for tensor in state:
            assert tensor.shape == expected_shape
            assert tensor.dtype == dtype
            assert torch.count_nonzero(tensor).item() == 0


def test_gru_specific_zero_initial_state() -> None:
    config = _config(
        cell_kind=RecurrentCellKind.GRU
    )
    kernel = _kernel(config)

    state = build_zero_gru_initial_state(
        kernel,
        config,
        batch_size=3,
    )

    assert state.shape == (4, 3, H)
    assert torch.count_nonzero(state).item() == 0


def test_lstm_specific_zero_initial_state() -> None:
    config = _config(
        cell_kind=RecurrentCellKind.LSTM
    )
    kernel = _kernel(config)

    hidden, cell = build_zero_lstm_initial_state(
        kernel,
        config,
        batch_size=3,
    )

    assert hidden.shape == (4, 3, H)
    assert cell.shape == (4, 3, H)
    assert torch.count_nonzero(hidden).item() == 0
    assert torch.count_nonzero(cell).item() == 0


def test_flat_hidden_state_builder_supports_both_cells() -> None:
    for cell_kind in (
        RecurrentCellKind.GRU,
        RecurrentCellKind.LSTM,
    ):
        config = _config(cell_kind=cell_kind)
        kernel = _kernel(config)
        hidden = build_zero_flat_hidden_state(
            kernel,
            config,
            batch_size=2,
        )

        assert hidden.shape == (4, 2, H)
        assert torch.count_nonzero(hidden).item() == 0


def test_flat_cell_builder_rejects_gru() -> None:
    config = _config(
        cell_kind=RecurrentCellKind.GRU
    )
    kernel = _kernel(config)

    with pytest.raises(ValueError):
        build_zero_flat_cell_state(
            kernel,
            config,
            batch_size=2,
        )


def test_cell_specific_builder_requires_lstm_kernel() -> None:
    config = _config(
        cell_kind=RecurrentCellKind.LSTM
    )
    gru = _kernel(
        _config(
            cell_kind=RecurrentCellKind.GRU
        )
    )

    with pytest.raises(TypeError):
        build_zero_flat_cell_state(
            gru,
            config,
            batch_size=2,
        )


def test_gru_specific_builder_rejects_lstm_config() -> None:
    config = _config(
        cell_kind=RecurrentCellKind.LSTM
    )
    kernel = _kernel(config)

    with pytest.raises(ValueError):
        build_zero_gru_initial_state(
            kernel,
            config,
            batch_size=2,
        )


def test_lstm_specific_builder_rejects_gru_config() -> None:
    config = _config(
        cell_kind=RecurrentCellKind.GRU
    )
    kernel = _kernel(config)

    with pytest.raises(ValueError):
        build_zero_lstm_initial_state(
            kernel,
            config,
            batch_size=2,
        )


@pytest.mark.parametrize(
    "cell_kind",
    (
        RecurrentCellKind.GRU,
        RecurrentCellKind.LSTM,
    ),
)
def test_zero_canonical_final_states(
    cell_kind: RecurrentCellKind,
) -> None:
    config = _config(
        cell_kind=cell_kind,
        num_layers=3,
        bidirectional=True,
        hidden_dim=4,
    )
    kernel = _kernel(config)

    hidden, cell = build_zero_canonical_final_states(
        kernel,
        config,
        node_count=5,
    )

    assert hidden.shape == (3, 2, 5, 4)
    assert torch.count_nonzero(hidden).item() == 0

    if cell_kind == RecurrentCellKind.GRU:
        assert cell is None
    else:
        assert cell is not None
        assert cell.shape == (3, 2, 5, 4)
        assert torch.count_nonzero(cell).item() == 0


def test_zero_canonical_hidden_state() -> None:
    config = _config()
    kernel = _kernel(config)

    hidden = build_zero_canonical_hidden_state(
        kernel,
        config,
        node_count=0,
    )

    assert hidden.shape == (2, 2, 0, H)


def test_zero_canonical_cell_state() -> None:
    config = _config(
        cell_kind=RecurrentCellKind.LSTM,
        bidirectional=False,
        num_layers=1,
    )
    kernel = _kernel(config)

    cell = build_zero_canonical_cell_state(
        kernel,
        config,
        node_count=3,
    )

    assert cell.shape == (1, 1, 3, H)
    assert torch.count_nonzero(cell).item() == 0


def test_initial_states_do_not_require_grad() -> None:
    config = _config(
        cell_kind=RecurrentCellKind.LSTM
    )
    kernel = _kernel(config)
    hidden, cell = build_zero_lstm_initial_state(
        kernel,
        config,
        batch_size=2,
    )

    assert not hidden.requires_grad
    assert not cell.requires_grad


# =============================================================================
# Initial state: conversion and validation
# =============================================================================


@pytest.mark.parametrize(
    (
        "num_layers",
        "bidirectional",
        "node_count",
        "hidden_dim",
    ),
    (
        (
            1,
            False,
            0,
            1,
        ),
        (
            1,
            True,
            4,
            3,
        ),
        (
            3,
            False,
            2,
            5,
        ),
        (
            2,
            True,
            6,
            4,
        ),
    ),
)
def test_flatten_unflatten_round_trip(
    num_layers: int,
    bidirectional: bool,
    node_count: int,
    hidden_dim: int,
) -> None:
    config = _config(
        num_layers=num_layers,
        bidirectional=bidirectional,
        hidden_dim=hidden_dim,
    )
    layout = build_recurrent_state_layout(config)
    directions = 2 if bidirectional else 1
    state = torch.arange(
        num_layers
        * directions
        * node_count
        * hidden_dim,
        dtype=torch.float64,
    ).reshape(
        num_layers,
        directions,
        node_count,
        hidden_dim,
    )

    flat = flatten_recurrent_state(
        state,
        layout=layout,
    )
    restored = unflatten_recurrent_state(
        flat,
        layout=layout,
    )

    assert torch.equal(restored, state)
    assert flat.data_ptr() == state.data_ptr()
    assert restored.data_ptr() == state.data_ptr()


def test_conversion_rejects_wrong_layout_type() -> None:
    with pytest.raises(TypeError):
        flatten_recurrent_state(
            torch.zeros(1, 1, 1, 1),
            layout=object(),  # type: ignore[arg-type]
        )

    with pytest.raises(TypeError):
        unflatten_recurrent_state(
            torch.zeros(1, 1, 1),
            layout=object(),  # type: ignore[arg-type]
        )


def test_validate_exact_zero_state_accepts_zero() -> None:
    layout = RecurrentStateLayout(
        2,
        2,
        H,
    )
    state = torch.zeros(
        2,
        2,
        3,
        H,
    )

    validate_exact_zero_recurrent_state(
        state,
        layout=layout,
        node_count=3,
    )


def test_validate_exact_zero_state_rejects_nonzero() -> None:
    layout = RecurrentStateLayout(
        2,
        2,
        H,
    )
    state = torch.zeros(
        2,
        2,
        3,
        H,
    )
    state[0, 0, 0, 0] = 1.0

    with pytest.raises(ValueError):
        validate_exact_zero_recurrent_state(
            state,
            layout=layout,
            node_count=3,
        )


@pytest.mark.parametrize(
    "cell_kind",
    (
        RecurrentCellKind.GRU,
        RecurrentCellKind.LSTM,
    ),
)
def test_validate_zero_initial_state_accepts_built_state(
    cell_kind: RecurrentCellKind,
) -> None:
    config = _config(
        cell_kind=cell_kind
    )
    kernel = _kernel(config)
    state = build_zero_recurrent_initial_state(
        kernel,
        config,
        batch_size=3,
    )

    validate_zero_recurrent_initial_state(
        state,
        kernel=kernel,
        config=config,
        batch_size=3,
    )


def test_validate_gru_initial_state_rejects_tuple() -> None:
    config = _config(
        cell_kind=RecurrentCellKind.GRU
    )
    kernel = _kernel(config)
    hidden = build_zero_flat_hidden_state(
        kernel,
        config,
        batch_size=2,
    )

    with pytest.raises(TypeError):
        validate_zero_recurrent_initial_state(
            (
                hidden,
                hidden,
            ),
            kernel=kernel,
            config=config,
            batch_size=2,
        )


def test_validate_lstm_initial_state_rejects_non_tuple() -> None:
    config = _config(
        cell_kind=RecurrentCellKind.LSTM
    )
    kernel = _kernel(config)
    hidden = build_zero_flat_hidden_state(
        kernel,
        config,
        batch_size=2,
    )

    with pytest.raises(TypeError):
        validate_zero_recurrent_initial_state(
            hidden,
            kernel=kernel,
            config=config,
            batch_size=2,
        )


@pytest.mark.parametrize(
    "cell_kind",
    (
        RecurrentCellKind.GRU,
        RecurrentCellKind.LSTM,
    ),
)
def test_validate_zero_initial_state_rejects_nonzero(
    cell_kind: RecurrentCellKind,
) -> None:
    config = _config(
        cell_kind=cell_kind
    )
    kernel = _kernel(config)
    state = build_zero_recurrent_initial_state(
        kernel,
        config,
        batch_size=2,
    )

    if isinstance(state, torch.Tensor):
        corrupted = state.clone()
        corrupted[0, 0, 0] = 1.0
    else:
        hidden, cell = state
        corrupted_cell = cell.clone()
        corrupted_cell[0, 0, 0] = 1.0
        corrupted = (
            hidden,
            corrupted_cell,
        )

    with pytest.raises(ValueError):
        validate_zero_recurrent_initial_state(
            corrupted,
            kernel=kernel,
            config=config,
            batch_size=2,
        )


def test_validate_flat_initial_state_rejects_dtype_mismatch() -> None:
    config = _config()
    kernel = _kernel(
        config,
        dtype=torch.float64,
    )
    state = torch.zeros(
        4,
        2,
        H,
        dtype=torch.float32,
    )

    with pytest.raises(ValueError):
        validate_flat_initial_state(
            state,
            kernel=kernel,
            config=config,
            batch_size=2,
        )


def test_validate_flat_initial_state_rejects_shape_mismatch() -> None:
    config = _config()
    kernel = _kernel(config)
    state = torch.zeros(
        3,
        2,
        H,
        dtype=torch.float64,
    )

    with pytest.raises(ValueError):
        validate_flat_initial_state(
            state,
            kernel=kernel,
            config=config,
            batch_size=2,
        )


# =============================================================================
# Conditional CUDA coverage
# =============================================================================


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is not available.",
)
def test_preprocessing_cuda_device_preservation() -> None:
    history = _history(
        dtype=torch.float32,
        device="cuda",
    )
    config = _adapter_config()
    metadata = build_recurrent_execution_metadata(
        history,
        config,
    )

    assert metadata.device.type == "cuda"

    adapter = RecurrentInputAdapter(config).cuda()
    batch = _canonical_batch(
        dtype=torch.float32,
        device="cuda",
        requires_grad=True,
    )
    adapted = adapter(
        batch.values,
        batch.timestep_mask,
    )

    assert adapted.device.type == "cuda"
    assert torch.count_nonzero(
        adapted[~batch.timestep_mask]
    ).item() == 0

    kernel = _kernel(
        config,
        dtype=torch.float32,
        device="cuda",
    )
    hidden = build_zero_flat_hidden_state(
        kernel,
        config,
        batch_size=2,
    )

    assert hidden.device.type == "cuda"
    assert hidden.dtype == torch.float32
