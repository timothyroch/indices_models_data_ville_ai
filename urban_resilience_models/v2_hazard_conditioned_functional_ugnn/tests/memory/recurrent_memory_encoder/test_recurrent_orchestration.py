"""
Final orchestration tests for Phase 6 recurrent temporal memory encoding.

Target repository path:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            tests/
                memory/
                    recurrent_memory_encoder/
                        test_recurrent_orchestration.py

Modules under test:
    memory/recurrent_memory_encoder/diagnostics.py
    memory/recurrent_memory_encoder/recurrent_memory_encoder.py
    memory/recurrent_memory_encoder/__init__.py

This suite validates the final Phase 6 integration boundary:

- direct recurrent configuration and top-level sequence configuration dispatch;
- GRU/LSTM construction through direct builders and the public facade;
- direct encoder, helper, facade, and package-level numerical parity;
- architecture identity and explicit parameter-snapshot parity;
- detached run diagnostics and packed/reference comparison diagnostics;
- disabled diagnostics behavior;
- registered child-module semantics, hooks, train/eval propagation, dtype/device
  movement, parameters, gradients, and state dictionaries;
- all-zero-history orchestration;
- public package re-exports and import boundaries;
- defensive rejection of unsupported branches and invalid objects.

Lower-level schema, preprocessing, sequence execution, and cell-specific encoder
behavior are tested in their dedicated Phase 6 suites.
"""

from __future__ import annotations

import copy
import importlib
import inspect
import json

import pytest
import torch
from torch import nn

import urban_resilience_models.v2_hazard_conditioned_functional_ugnn.memory.recurrent_memory_encoder as recurrent_package
import urban_resilience_models.v2_hazard_conditioned_functional_ugnn.memory.recurrent_memory_encoder.diagnostics as diagnostics_module
import urban_resilience_models.v2_hazard_conditioned_functional_ugnn.memory.recurrent_memory_encoder.recurrent_memory_encoder as orchestration_module
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.memory.config import (
    RecurrentCellKind,
    RecurrentSequenceEncoderConfig,
    TemporalSequenceEncoderConfig,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.memory.recurrent_memory_encoder import (
    IMPLEMENTED_RECURRENT_CELL_KINDS,
    IMPLEMENTED_RECURRENT_SEQUENCE_ENCODER_KINDS,
    RECURRENT_ENCODER_LIKE_TYPES,
    RECURRENT_MEMORY_ENCODER_ARCHITECTURE_IDENTITY_POLICY,
    RECURRENT_MEMORY_ENCODER_COMPONENT_KIND,
    RECURRENT_MEMORY_ENCODER_COMPONENT_NAME,
    RECURRENT_MEMORY_ENCODER_DIAGNOSTICS_POLICY,
    RECURRENT_MEMORY_ENCODER_IMPLEMENTATION_VERSION,
    RECURRENT_MEMORY_ENCODER_OPERATION_NAME,
    RECURRENT_MEMORY_ENCODER_PARAMETER_SNAPSHOT_POLICY,
    RECURRENT_MEMORY_ENCODER_SCIENTIFIC_INTERPRETATION,
    RECURRENT_SEQUENCE_ENCODER_TYPES,
    PackedReferenceComparisonDiagnostics,
    RecurrentDiagnosticReport,
    RecurrentDiagnostics,
    RecurrentEncoder,
    RecurrentEncoderDiagnostics,
    RecurrentEncoderLike,
    RecurrentMemoryEncoder,
    RecurrentRunDiagnostics,
    RecurrentSequenceEncoder,
    RecurrentSequenceEncoderModule,
    build_recurrent_diagnostic_report,
    build_recurrent_encoder,
    build_recurrent_memory_encoder,
    build_recurrent_sequence_encoder,
    build_sequence_encoder,
    compare_packed_and_reference_runs,
    compare_packed_reference_runs,
    compare_recurrent_execution_paths,
    diagnose_recurrent_encoder_run,
    diagnose_recurrent_run,
    encode_history,
    encode_history_with_state,
    encode_recurrent_history,
    encode_recurrent_history_with_state,
    encode_recurrent_memory,
    encode_recurrent_memory_with_state,
    extract_recurrent_config,
    extract_recurrent_sequence_config,
    is_recurrent_encoder_like,
    is_recurrent_sequence_encoder,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.memory.recurrent_memory_encoder.gru_encoder import (
    GRUSequenceEncoder,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.memory.recurrent_memory_encoder.lstm_encoder import (
    LSTMSequenceEncoder,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.memory.recurrent_memory_encoder.schemas import (
    RecurrentExecutionPath,
    RecurrentSequenceEncoderRun,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.memory.schemas.history_inputs import (
    HistoricalSequenceInputs,
    HistoryMissingValuePolicy,
    HistoryZeroLengthPolicy,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.memory.schemas.provenance import (
    MemoryArchitectureProvenance,
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
LENGTHS = (
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
    padding_direction: TemporalPaddingDirection,
) -> torch.Tensor:
    mask = torch.zeros(
        int(lengths.numel()),
        T,
        dtype=torch.bool,
        device=lengths.device,
    )

    for node_index, value in enumerate(
        lengths.tolist()
    ):
        length = int(value)

        if length == 0:
            continue

        if padding_direction == TemporalPaddingDirection.LEFT:
            mask[
                node_index,
                T - length :,
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
    with_feature_observed_mask: bool = True,
    source_suffix: str = "base",
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

    for node_index in range(node_count):
        positions = torch.nonzero(
            mask[node_index],
            as_tuple=False,
        ).flatten()
        logical_length = int(
            positions.numel()
        )

        for logical_index, temporal_index in enumerate(
            positions.tolist()
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
            next_value += float(
                feature_dim
            )

    if requires_grad:
        values.requires_grad_()

    feature_observed_mask = None

    if with_feature_observed_mask:
        feature_observed_mask = (
            mask
            .unsqueeze(
                -1
            )
            .expand(
                node_count,
                T,
                feature_dim,
            )
            .clone()
        )

        if node_count > 0 and bool(
            mask[0, 0].item()
        ):
            feature_observed_mask[
                0,
                0,
            ] = False

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
                f"orchestration-node-axis-{source_suffix}"
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
                f"orchestration-feature-axis-{source_suffix}"
            ),
        ),
        temporal_coordinates=RelativeTemporalCoordinates(
            values=coordinates,
            unit="months",
        ),
        source_provenance=MemorySourceProvenance(
            source_name="orchestration-panel",
            source_kind="historical-node-sequence",
            source_fingerprint=(
                f"orchestration-source-{source_suffix}"
            ),
            preprocessing_fingerprint=(
                f"orchestration-preprocessing-{source_suffix}"
            ),
        ),
        feature_observed_mask=(
            feature_observed_mask
        ),
        padding_direction=padding_direction,
        missing_value_policy=(
            HistoryMissingValuePolicy.FINITE_PLACEHOLDER_WITH_MASK
            if feature_observed_mask is not None
            else HistoryMissingValuePolicy.UPSTREAM_IMPUTED
        ),
        zero_length_policy=(
            HistoryZeroLengthPolicy.ALLOW_ZERO_HISTORY
            if any(
                length == 0
                for length in lengths
            )
            else HistoryZeroLengthPolicy.ERROR
        ),
    )


def _recurrent_config(
    *,
    cell_kind: RecurrentCellKind,
    pack_sequences: bool = True,
    input_dim: int = D,
    hidden_dim: int = H,
    num_layers: int = 2,
    dropout: float = 0.0,
    bidirectional: bool = True,
    use_bias: bool = True,
    input_projection_dim: int | None = 5,
    layer_normalization: bool = True,
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


def _top_level_config(
    recurrent: RecurrentSequenceEncoderConfig,
) -> TemporalSequenceEncoderConfig:
    return TemporalSequenceEncoderConfig.recurrent_encoder(
        recurrent
    )


def _encoder_type(
    cell_kind: RecurrentCellKind,
):
    return (
        GRUSequenceEncoder
        if cell_kind == RecurrentCellKind.GRU
        else LSTMSequenceEncoder
    )


def _build_direct(
    config: RecurrentSequenceEncoderConfig,
):
    return build_recurrent_sequence_encoder(
        config
    )


def _assert_run_shape_contract(
    run: RecurrentSequenceEncoderRun,
    history: HistoricalSequenceInputs,
    config: RecurrentSequenceEncoderConfig,
) -> None:
    assert run.source_history is history
    assert run.encoder_kind == config.schema_encoder_kind
    assert run.public_output.encoded_shape == (
        history.node_count,
        history.sequence_length,
        config.output_dim,
    )
    assert run.final_hidden_state.shape == (
        config.num_layers,
        2 if config.bidirectional else 1,
        history.node_count,
        config.hidden_dim,
    )
    assert torch.count_nonzero(
        run.public_output.encoded_sequence[
            ~history.timestep_mask
        ]
    ).item() == 0

    if config.cell_kind == RecurrentCellKind.GRU:
        assert run.final_cell_state is None
    else:
        assert run.final_cell_state is not None
        assert run.final_cell_state.shape == (
            config.num_layers,
            2 if config.bidirectional else 1,
            history.node_count,
            config.hidden_dim,
        )


def _weighted_loss(
    run: RecurrentSequenceEncoderRun,
) -> torch.Tensor:
    output = run.public_output.encoded_sequence
    hidden = run.final_hidden_state
    output_weight = torch.arange(
        1,
        output.numel() + 1,
        dtype=output.dtype,
        device=output.device,
    ).reshape_as(
        output
    )
    hidden_weight = torch.arange(
        1,
        hidden.numel() + 1,
        dtype=hidden.dtype,
        device=hidden.device,
    ).reshape_as(
        hidden
    )
    loss = (
        output
        * output_weight
    ).sum() + (
        hidden
        * hidden_weight
    ).sum()

    if run.final_cell_state is not None:
        cell = run.final_cell_state
        cell_weight = torch.arange(
            1,
            cell.numel() + 1,
            dtype=cell.dtype,
            device=cell.device,
        ).reshape_as(
            cell
        )
        loss = loss + (
            cell
            * cell_weight
        ).sum()

    return loss


# =============================================================================
# Identity, aliases, capabilities, and import boundaries
# =============================================================================


@pytest.mark.parametrize(
    "value",
    (
        RECURRENT_MEMORY_ENCODER_IMPLEMENTATION_VERSION,
        RECURRENT_MEMORY_ENCODER_COMPONENT_NAME,
        RECURRENT_MEMORY_ENCODER_COMPONENT_KIND,
        RECURRENT_MEMORY_ENCODER_OPERATION_NAME,
        RECURRENT_MEMORY_ENCODER_DIAGNOSTICS_POLICY,
        RECURRENT_MEMORY_ENCODER_PARAMETER_SNAPSHOT_POLICY,
        RECURRENT_MEMORY_ENCODER_ARCHITECTURE_IDENTITY_POLICY,
        RECURRENT_MEMORY_ENCODER_SCIENTIFIC_INTERPRETATION,
    ),
)
def test_orchestration_policy_strings_are_nonempty(
    value: str,
) -> None:
    assert isinstance(
        value,
        str,
    )
    assert value.strip()


def test_implemented_kind_vocabularies_are_exact() -> None:
    assert IMPLEMENTED_RECURRENT_CELL_KINDS == (
        "gru",
        "lstm",
    )
    assert IMPLEMENTED_RECURRENT_SEQUENCE_ENCODER_KINDS == (
        "gru",
        "lstm",
    )


def test_public_aliases_are_exact() -> None:
    assert RecurrentEncoder is RecurrentMemoryEncoder
    assert RecurrentSequenceEncoder is RecurrentMemoryEncoder
    assert extract_recurrent_config is extract_recurrent_sequence_config
    assert build_recurrent_encoder is build_recurrent_memory_encoder
    assert build_sequence_encoder is build_recurrent_sequence_encoder
    assert encode_history is encode_recurrent_memory
    assert encode_history_with_state is (
        encode_recurrent_memory_with_state
    )
    assert RecurrentEncoderDiagnostics is RecurrentDiagnostics
    assert RecurrentDiagnosticReport is RecurrentRunDiagnostics
    assert diagnose_recurrent_encoder_run is diagnose_recurrent_run
    assert build_recurrent_diagnostic_report is diagnose_recurrent_run
    assert compare_recurrent_execution_paths is (
        compare_packed_reference_runs
    )
    assert compare_packed_and_reference_runs is (
        compare_packed_reference_runs
    )


def test_type_collections_are_exact() -> None:
    assert RECURRENT_SEQUENCE_ENCODER_TYPES == (
        GRUSequenceEncoder,
        LSTMSequenceEncoder,
    )
    assert RECURRENT_ENCODER_LIKE_TYPES == (
        GRUSequenceEncoder,
        LSTMSequenceEncoder,
        RecurrentMemoryEncoder,
    )


def test_orchestration_module_all_is_unique_and_resolves() -> None:
    exported = orchestration_module.__all__

    assert isinstance(
        exported,
        tuple,
    )
    assert len(exported) == len(set(exported))

    for name in exported:
        assert hasattr(
            orchestration_module,
            name,
        )


def test_diagnostics_module_all_is_unique_and_resolves() -> None:
    exported = diagnostics_module.__all__

    assert isinstance(
        exported,
        tuple,
    )
    assert len(exported) == len(set(exported))

    for name in exported:
        assert hasattr(
            diagnostics_module,
            name,
        )


def test_package_all_is_unique_and_resolves() -> None:
    exported = recurrent_package.__all__

    assert isinstance(
        exported,
        tuple,
    )
    assert len(exported) == len(set(exported))

    for name in exported:
        assert hasattr(
            recurrent_package,
            name,
        )


def test_package_reexports_orchestration_and_diagnostics_symbols_by_identity() -> None:
    for module in (
        orchestration_module,
        diagnostics_module,
    ):
        for name in module.__all__:
            assert getattr(
                recurrent_package,
                name,
            ) is getattr(
                module,
                name,
            )


def test_private_provenance_helpers_are_not_package_exports() -> None:
    forbidden = (
        "build_recurrent_architecture_provenance",
        "build_recurrent_sequence_computation_provenance",
        "validate_parameter_snapshot_provenance",
        "fingerprint_module_parameter_state",
    )

    for name in forbidden:
        assert name not in recurrent_package.__all__
        assert not hasattr(
            recurrent_package,
            name,
        )


def test_private_canonical_batch_is_not_package_exported() -> None:
    assert "_CanonicalRecurrentBatch" not in recurrent_package.__all__
    assert not hasattr(
        recurrent_package,
        "_CanonicalRecurrentBatch",
    )


def test_star_import_contains_complete_public_api() -> None:
    namespace: dict[str, object] = {}

    exec(
        "from urban_resilience_models."
        "v2_hazard_conditioned_functional_ugnn."
        "memory.recurrent_memory_encoder import *",
        namespace,
    )

    for name in recurrent_package.__all__:
        assert name in namespace


def test_orchestration_source_does_not_duplicate_numerical_execution() -> None:
    source = inspect.getsource(
        orchestration_module
    )

    forbidden = (
        "pack_padded_sequence",
        "pad_packed_sequence",
        "canonicalize_recurrent_history",
        "build_zero_gru_initial_state",
        "build_zero_lstm_initial_state",
        "restore_recurrent_sequence_to_source",
        "nn.GRU(",
        "nn.LSTM(",
    )

    for fragment in forbidden:
        assert fragment not in source


def test_importing_package_constructs_no_parameters() -> None:
    before = {
        name
        for name, value in vars(
            recurrent_package
        ).items()
        if isinstance(
            value,
            nn.Parameter,
        )
    }

    reloaded = importlib.reload(
        recurrent_package
    )

    after = {
        name
        for name, value in vars(
            reloaded
        ).items()
        if isinstance(
            value,
            nn.Parameter,
        )
    }

    assert before == set()
    assert after == set()


# =============================================================================
# Configuration extraction and dispatch
# =============================================================================


@pytest.mark.parametrize(
    "cell_kind",
    (
        RecurrentCellKind.GRU,
        RecurrentCellKind.LSTM,
    ),
)
def test_extract_direct_recurrent_config_returns_same_object(
    cell_kind: RecurrentCellKind,
) -> None:
    config = _recurrent_config(
        cell_kind=cell_kind
    )

    assert extract_recurrent_sequence_config(
        config
    ) is config


@pytest.mark.parametrize(
    "cell_kind",
    (
        RecurrentCellKind.GRU,
        RecurrentCellKind.LSTM,
    ),
)
def test_extract_top_level_recurrent_config_returns_active_branch(
    cell_kind: RecurrentCellKind,
) -> None:
    recurrent = _recurrent_config(
        cell_kind=cell_kind
    )
    top_level = _top_level_config(
        recurrent
    )

    assert extract_recurrent_sequence_config(
        top_level
    ) is recurrent


@pytest.mark.parametrize(
    "value",
    (
        object(),
        None,
        "gru",
        7,
    ),
)
def test_extract_recurrent_config_rejects_wrong_type(
    value: object,
) -> None:
    with pytest.raises(TypeError):
        extract_recurrent_sequence_config(
            value  # type: ignore[arg-type]
        )


def test_extract_recurrent_config_rejects_baseline_top_level() -> None:
    baseline = TemporalSequenceEncoderConfig.identity(
        input_dim=D
    )

    with pytest.raises(NotImplementedError):
        extract_recurrent_sequence_config(
            baseline
        )


@pytest.mark.parametrize(
    "cell_kind",
    (
        RecurrentCellKind.GRU,
        RecurrentCellKind.LSTM,
    ),
)
@pytest.mark.parametrize(
    "use_top_level",
    (
        False,
        True,
    ),
)
def test_direct_builder_dispatches_correct_cell_encoder(
    cell_kind: RecurrentCellKind,
    use_top_level: bool,
) -> None:
    recurrent = _recurrent_config(
        cell_kind=cell_kind
    )
    config = (
        _top_level_config(
            recurrent
        )
        if use_top_level
        else recurrent
    )

    encoder = build_recurrent_sequence_encoder(
        config
    )

    assert isinstance(
        encoder,
        _encoder_type(
            cell_kind
        ),
    )
    assert encoder.config is recurrent
    assert is_recurrent_sequence_encoder(
        encoder
    )
    assert is_recurrent_encoder_like(
        encoder
    )


def test_direct_builder_rejects_baseline_top_level() -> None:
    with pytest.raises(NotImplementedError):
        build_recurrent_sequence_encoder(
            TemporalSequenceEncoderConfig.identity(
                input_dim=D
            )
        )


@pytest.mark.parametrize(
    "value",
    (
        nn.Identity(),
        object(),
        None,
        "encoder",
    ),
)
def test_recurrent_predicates_reject_unrelated_objects(
    value: object,
) -> None:
    assert not is_recurrent_sequence_encoder(
        value
    )
    assert not is_recurrent_encoder_like(
        value
    )


# =============================================================================
# Facade construction and registered-module semantics
# =============================================================================


@pytest.mark.parametrize(
    "cell_kind",
    (
        RecurrentCellKind.GRU,
        RecurrentCellKind.LSTM,
    ),
)
@pytest.mark.parametrize(
    "use_top_level",
    (
        False,
        True,
    ),
)
def test_facade_construction_and_properties(
    cell_kind: RecurrentCellKind,
    use_top_level: bool,
) -> None:
    recurrent = _recurrent_config(
        cell_kind=cell_kind,
        hidden_dim=5,
        num_layers=3,
        bidirectional=False,
        input_projection_dim=6,
    )
    requested = (
        _top_level_config(
            recurrent
        )
        if use_top_level
        else recurrent
    )
    diagnostics = RecurrentDiagnostics(
        hidden_boundary_abs_threshold=0.95
    )
    facade = RecurrentMemoryEncoder(
        requested,
        diagnostics=diagnostics,
    )

    assert facade.requested_config is requested
    assert facade.config is recurrent
    assert facade.diagnostics is diagnostics
    assert facade.diagnostics_enabled
    assert isinstance(
        facade.encoder,
        _encoder_type(
            cell_kind
        ),
    )
    assert facade.core_encoder is facade.encoder
    assert facade.cell_kind == cell_kind
    assert facade.encoder_kind == recurrent.schema_encoder_kind
    assert facade.input_dim == D
    assert facade.recurrent_input_dim == 6
    assert facade.hidden_dim == 5
    assert facade.output_dim == 5
    assert facade.num_layers == 3
    assert facade.num_directions == 1
    assert not facade.is_bidirectional
    assert facade.kernel is facade.encoder.kernel
    assert facade.input_adapter is facade.encoder.input_adapter
    assert facade.state_layout == facade.encoder.state_layout
    assert facade.parameter_count == facade.encoder.parameter_count
    assert facade.trainable_parameter_count == (
        facade.encoder.trainable_parameter_count
    )
    assert is_recurrent_encoder_like(
        facade
    )
    assert not is_recurrent_sequence_encoder(
        facade
    )


def test_facade_rejects_wrong_diagnostics_type() -> None:
    with pytest.raises(TypeError):
        RecurrentMemoryEncoder(
            _recurrent_config(
                cell_kind=RecurrentCellKind.GRU
            ),
            diagnostics=object(),  # type: ignore[arg-type]
        )


def test_facade_builder_returns_recurrent_memory_encoder() -> None:
    config = _recurrent_config(
        cell_kind=RecurrentCellKind.GRU
    )
    facade = build_recurrent_memory_encoder(
        config
    )

    assert isinstance(
        facade,
        RecurrentMemoryEncoder,
    )
    assert facade.config is config


@pytest.mark.parametrize(
    "cell_kind",
    (
        RecurrentCellKind.GRU,
        RecurrentCellKind.LSTM,
    ),
)
def test_facade_registers_exactly_one_cell_encoder_child(
    cell_kind: RecurrentCellKind,
) -> None:
    facade = RecurrentMemoryEncoder(
        _recurrent_config(
            cell_kind=cell_kind
        )
    )

    children = dict(
        facade.named_children()
    )

    assert children == {
        "encoder": facade.encoder,
    }


@pytest.mark.parametrize(
    "cell_kind",
    (
        RecurrentCellKind.GRU,
        RecurrentCellKind.LSTM,
    ),
)
def test_facade_parameter_names_are_prefixed_once(
    cell_kind: RecurrentCellKind,
) -> None:
    facade = RecurrentMemoryEncoder(
        _recurrent_config(
            cell_kind=cell_kind
        )
    )
    direct_names = {
        name
        for name, _ in facade.encoder.named_parameters()
    }
    facade_names = {
        name
        for name, _ in facade.named_parameters()
    }

    assert facade_names == {
        f"encoder.{name}"
        for name in direct_names
    }


@pytest.mark.parametrize(
    "cell_kind",
    (
        RecurrentCellKind.GRU,
        RecurrentCellKind.LSTM,
    ),
)
def test_train_eval_mode_propagates_to_selected_encoder(
    cell_kind: RecurrentCellKind,
) -> None:
    facade = RecurrentMemoryEncoder(
        _recurrent_config(
            cell_kind=cell_kind,
            num_layers=3,
            dropout=0.4,
        )
    )

    facade.eval()

    assert not facade.training
    assert not facade.encoder.training
    assert not facade.kernel.training
    assert not facade.dropout_active

    facade.train()

    assert facade.training
    assert facade.encoder.training
    assert facade.kernel.training
    assert facade.dropout_active


@pytest.mark.parametrize(
    "cell_kind",
    (
        RecurrentCellKind.GRU,
        RecurrentCellKind.LSTM,
    ),
)
def test_dtype_movement_propagates_to_selected_encoder(
    cell_kind: RecurrentCellKind,
) -> None:
    facade = RecurrentMemoryEncoder(
        _recurrent_config(
            cell_kind=cell_kind
        )
    ).double()

    assert all(
        parameter.dtype == torch.float64
        for parameter in facade.parameters()
    )

    facade.float()

    assert all(
        parameter.dtype == torch.float32
        for parameter in facade.parameters()
    )


# =============================================================================
# Direct helpers, facade, and package-level execution parity
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
def test_direct_helpers_preserve_direct_encoder_contract(
    cell_kind: RecurrentCellKind,
    pack_sequences: bool,
) -> None:
    torch.manual_seed(
        100
    )
    config = _recurrent_config(
        cell_kind=cell_kind,
        pack_sequences=pack_sequences,
    )
    encoder = _build_direct(
        config
    ).double()
    encoder.eval()
    history = _history()

    output = encode_recurrent_history(
        encoder,
        history,
    )
    run = encode_recurrent_history_with_state(
        encoder,
        history,
    )

    assert isinstance(
        output,
        TemporalSequenceEncoding,
    )
    assert isinstance(
        run,
        RecurrentSequenceEncoderRun,
    )
    assert output.source_history is history
    assert run.source_history is history
    assert torch.equal(
        output.encoded_sequence,
        run.public_output.encoded_sequence,
    )
    _assert_run_shape_contract(
        run,
        history,
        config,
    )


@pytest.mark.parametrize(
    "helper",
    (
        encode_recurrent_history,
        encode_recurrent_history_with_state,
    ),
)
def test_direct_helpers_reject_non_recurrent_module(
    helper,
) -> None:
    with pytest.raises(TypeError):
        helper(
            nn.Identity(),  # type: ignore[arg-type]
            _history(),
        )


@pytest.mark.parametrize(
    "helper",
    (
        encode_recurrent_history,
        encode_recurrent_history_with_state,
    ),
)
def test_direct_helpers_reject_wrong_source_type(
    helper,
) -> None:
    encoder = GRUSequenceEncoder(
        _recurrent_config(
            cell_kind=RecurrentCellKind.GRU
        )
    )

    with pytest.raises(TypeError):
        helper(
            encoder,
            object(),  # type: ignore[arg-type]
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
def test_direct_encoder_and_facade_numerical_parity(
    cell_kind: RecurrentCellKind,
    pack_sequences: bool,
) -> None:
    torch.manual_seed(
        101
    )
    config = _recurrent_config(
        cell_kind=cell_kind,
        pack_sequences=pack_sequences,
    )
    direct = _build_direct(
        config
    ).double()
    facade = RecurrentMemoryEncoder(
        config
    ).double()
    facade.encoder.load_state_dict(
        direct.state_dict()
    )
    direct.eval()
    facade.eval()
    history = _history(
        padding_direction=TemporalPaddingDirection.LEFT
    )

    direct_run = direct.encode_with_state(
        history
    )
    facade_run = facade.encode_with_state(
        history
    )
    facade_output = facade(
        history
    )

    torch.testing.assert_close(
        direct_run.public_output.encoded_sequence,
        facade_run.public_output.encoded_sequence,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        direct_run.final_hidden_state,
        facade_run.final_hidden_state,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        facade_output.encoded_sequence,
        facade_run.public_output.encoded_sequence,
        rtol=0.0,
        atol=0.0,
    )

    if cell_kind == RecurrentCellKind.LSTM:
        assert direct_run.final_cell_state is not None
        assert facade_run.final_cell_state is not None
        torch.testing.assert_close(
            direct_run.final_cell_state,
            facade_run.final_cell_state,
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
def test_generic_memory_helpers_accept_direct_and_facade(
    cell_kind: RecurrentCellKind,
) -> None:
    torch.manual_seed(
        102
    )
    config = _recurrent_config(
        cell_kind=cell_kind
    )
    direct = _build_direct(
        config
    ).double()
    facade = RecurrentMemoryEncoder(
        config
    ).double()
    facade.encoder.load_state_dict(
        direct.state_dict()
    )
    direct.eval()
    facade.eval()
    history = _history()

    direct_output = encode_recurrent_memory(
        direct,
        history,
    )
    facade_output = encode_recurrent_memory(
        facade,
        history,
    )
    direct_run = encode_recurrent_memory_with_state(
        direct,
        history,
    )
    facade_run = encode_recurrent_memory_with_state(
        facade,
        history,
    )

    torch.testing.assert_close(
        direct_output.encoded_sequence,
        facade_output.encoded_sequence,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        direct_run.final_hidden_state,
        facade_run.final_hidden_state,
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize(
    "helper",
    (
        encode_recurrent_memory,
        encode_recurrent_memory_with_state,
    ),
)
def test_generic_memory_helpers_reject_unrelated_module(
    helper,
) -> None:
    with pytest.raises(TypeError):
        helper(
            nn.Identity(),  # type: ignore[arg-type]
            _history(),
        )


# =============================================================================
# Hooks and single-execution behavior
# =============================================================================


@pytest.mark.parametrize(
    "cell_kind",
    (
        RecurrentCellKind.GRU,
        RecurrentCellKind.LSTM,
    ),
)
def test_facade_forward_preserves_outer_and_inner_hooks(
    cell_kind: RecurrentCellKind,
) -> None:
    facade = RecurrentMemoryEncoder(
        _recurrent_config(
            cell_kind=cell_kind
        )
    ).double()
    history = _history()
    outer_calls: list[int] = []
    inner_calls: list[int] = []
    kernel_calls: list[int] = []

    outer_hook = facade.register_forward_hook(
        lambda *args: outer_calls.append(1)
    )
    inner_hook = facade.encoder.register_forward_hook(
        lambda *args: inner_calls.append(1)
    )
    kernel_hook = facade.kernel.register_forward_hook(
        lambda *args: kernel_calls.append(1)
    )

    output = facade(
        history
    )

    outer_hook.remove()
    inner_hook.remove()
    kernel_hook.remove()

    assert isinstance(
        output,
        TemporalSequenceEncoding,
    )
    assert outer_calls == [1]
    assert inner_calls == [1]
    assert kernel_calls == [1]


@pytest.mark.parametrize(
    "cell_kind",
    (
        RecurrentCellKind.GRU,
        RecurrentCellKind.LSTM,
    ),
)
def test_encode_with_diagnostics_executes_kernel_once(
    cell_kind: RecurrentCellKind,
) -> None:
    facade = RecurrentMemoryEncoder(
        _recurrent_config(
            cell_kind=cell_kind
        )
    ).double()
    history = _history()
    kernel_calls: list[int] = []
    kernel_hook = facade.kernel.register_forward_hook(
        lambda *args: kernel_calls.append(1)
    )

    run, report = facade.encode_with_diagnostics(
        history
    )

    kernel_hook.remove()

    assert isinstance(
        run,
        RecurrentSequenceEncoderRun,
    )
    assert isinstance(
        report,
        RecurrentRunDiagnostics,
    )
    assert kernel_calls == [1]


# =============================================================================
# Architecture identity and explicit parameter snapshots
# =============================================================================


@pytest.mark.parametrize(
    "cell_kind",
    (
        RecurrentCellKind.GRU,
        RecurrentCellKind.LSTM,
    ),
)
def test_direct_and_facade_architecture_provenance_are_identical(
    cell_kind: RecurrentCellKind,
) -> None:
    config = _recurrent_config(
        cell_kind=cell_kind,
        pack_sequences=True,
    )
    direct = _build_direct(
        config
    )
    facade = RecurrentMemoryEncoder(
        config
    )

    direct_provenance = direct.architecture_provenance()
    facade_provenance = facade.architecture_provenance()

    assert isinstance(
        direct_provenance,
        MemoryArchitectureProvenance,
    )
    assert facade_provenance is not direct_provenance
    assert (
        facade_provenance.architecture_fingerprint
        == direct_provenance.architecture_fingerprint
    )
    assert (
        facade_provenance.architecture_metadata
        == direct_provenance.architecture_metadata
    )


@pytest.mark.parametrize(
    "cell_kind",
    (
        RecurrentCellKind.GRU,
        RecurrentCellKind.LSTM,
    ),
)
def test_packed_reference_facades_share_architecture_identity(
    cell_kind: RecurrentCellKind,
) -> None:
    packed = RecurrentMemoryEncoder(
        _recurrent_config(
            cell_kind=cell_kind,
            pack_sequences=True,
        )
    )
    reference = RecurrentMemoryEncoder(
        _recurrent_config(
            cell_kind=cell_kind,
            pack_sequences=False,
        )
    )

    assert (
        packed.architecture_provenance().architecture_fingerprint
        == reference.architecture_provenance().architecture_fingerprint
    )


@pytest.mark.parametrize(
    "cell_kind",
    (
        RecurrentCellKind.GRU,
        RecurrentCellKind.LSTM,
    ),
)
def test_direct_and_facade_snapshot_fingerprints_match(
    cell_kind: RecurrentCellKind,
) -> None:
    torch.manual_seed(
        103
    )
    config = _recurrent_config(
        cell_kind=cell_kind
    )
    direct = _build_direct(
        config
    ).double()
    facade = RecurrentMemoryEncoder(
        config
    ).double()
    facade.encoder.load_state_dict(
        direct.state_dict()
    )

    direct_snapshot = direct.build_parameter_snapshot(
        checkpoint_id="shared-checkpoint",
        checkpoint_fingerprint="shared-checkpoint-fingerprint",
        training_step=51,
    )
    facade_snapshot = facade.build_parameter_snapshot(
        checkpoint_id="shared-checkpoint",
        checkpoint_fingerprint="shared-checkpoint-fingerprint",
        training_step=51,
    )

    assert isinstance(
        facade_snapshot,
        MemoryParameterSnapshotProvenance,
    )
    assert (
        direct_snapshot.parameter_snapshot_fingerprint
        == facade_snapshot.parameter_snapshot_fingerprint
    )
    assert direct_snapshot.parameter_count == (
        facade_snapshot.parameter_count
    )


@pytest.mark.parametrize(
    "cell_kind",
    (
        RecurrentCellKind.GRU,
        RecurrentCellKind.LSTM,
    ),
)
def test_facade_snapshot_validates_execution(
    cell_kind: RecurrentCellKind,
) -> None:
    facade = RecurrentMemoryEncoder(
        _recurrent_config(
            cell_kind=cell_kind
        )
    ).double()
    snapshot = facade.build_parameter_snapshot()
    run = facade.encode_with_state(
        _history(),
        parameter_snapshot=snapshot,
    )

    assert (
        run.public_output.parameter_snapshot_fingerprint
        == snapshot.parameter_snapshot_fingerprint
    )


@pytest.mark.parametrize(
    "cell_kind",
    (
        RecurrentCellKind.GRU,
        RecurrentCellKind.LSTM,
    ),
)
def test_facade_rejects_stale_snapshot(
    cell_kind: RecurrentCellKind,
) -> None:
    facade = RecurrentMemoryEncoder(
        _recurrent_config(
            cell_kind=cell_kind
        )
    ).double()
    snapshot = facade.build_parameter_snapshot()

    with torch.no_grad():
        next(
            facade.parameters()
        ).add_(0.01)

    with pytest.raises(ValueError):
        facade(
            _history(),
            parameter_snapshot=snapshot,
        )


# =============================================================================
# Run diagnostics
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
def test_facade_encode_with_diagnostics_report(
    cell_kind: RecurrentCellKind,
    pack_sequences: bool,
) -> None:
    torch.manual_seed(
        104
    )
    config = _recurrent_config(
        cell_kind=cell_kind,
        pack_sequences=pack_sequences,
    )
    facade = RecurrentMemoryEncoder(
        config
    ).double()
    facade.eval()
    history = _history()

    run, report = facade.encode_with_diagnostics(
        history,
        component_name="phase6-recurrent-test",
    )

    assert isinstance(
        run,
        RecurrentSequenceEncoderRun,
    )
    assert isinstance(
        report,
        RecurrentRunDiagnostics,
    )
    assert report.component_name == "phase6-recurrent-test"
    assert report.encoder_kind == cell_kind.value
    assert report.execution_path == (
        "packed"
        if pack_sequences
        else "reference"
    )
    assert report.node_count == history.node_count
    assert report.sequence_length == history.sequence_length
    assert report.input_dim == history.feature_dim
    assert report.output_dim == config.output_dim
    assert report.hidden_dim == config.hidden_dim
    assert report.num_layers == config.num_layers
    assert report.num_directions == (
        2 if config.bidirectional else 1
    )
    assert report.parameter_count == facade.parameter_count
    assert report.trainable_parameter_count == (
        facade.trainable_parameter_count
    )
    assert report.zero_history_count == 1
    assert report.nonempty_history_count == 5
    assert report.valid_timestep_count == sum(
        LENGTHS
    )
    assert report.padded_timestep_count == (
        history.node_count
        * T
        - sum(
            LENGTHS
        )
    )
    assert report.feature_observed_mask_available
    assert report.all_features_missing_valid_timestep_count == 1
    assert report.exact_zero_padding
    assert report.exact_zero_history_output
    assert report.exact_zero_history_hidden_state
    assert report.is_numerically_clean
    assert report.execution_is_structurally_consistent
    assert report.architecture_fingerprint == (
        run.architecture_fingerprint
    )
    assert report.run_lineage_fingerprint == (
        run.lineage_fingerprint()
    )

    if cell_kind == RecurrentCellKind.GRU:
        assert report.has_cell_state is False
        assert report.nonfinite_cell_state_count is None
        assert report.exact_zero_history_cell_state is None
    else:
        assert report.has_cell_state is True
        assert report.nonfinite_cell_state_count == 0
        assert report.exact_zero_history_cell_state is True

    payload = report.to_dict()
    serialized = report.to_json()

    assert payload["encoder_kind"] == cell_kind.value
    assert json.loads(
        serialized
    ) == payload
    assert report.fingerprint() == report.fingerprint()


@pytest.mark.parametrize(
    "cell_kind",
    (
        RecurrentCellKind.GRU,
        RecurrentCellKind.LSTM,
    ),
)
def test_diagnostics_do_not_detach_original_run_graph(
    cell_kind: RecurrentCellKind,
) -> None:
    facade = RecurrentMemoryEncoder(
        _recurrent_config(
            cell_kind=cell_kind
        )
    ).double()
    history = _history(
        requires_grad=True
    )
    run = facade.encode_with_state(
        history
    )

    report = facade.diagnose_run(
        run
    )

    assert isinstance(
        report,
        RecurrentRunDiagnostics,
    )
    _weighted_loss(
        run
    ).backward()

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


@pytest.mark.parametrize(
    "cell_kind",
    (
        RecurrentCellKind.GRU,
        RecurrentCellKind.LSTM,
    ),
)
def test_disabled_diagnostics_returns_none_without_changing_execution(
    cell_kind: RecurrentCellKind,
) -> None:
    facade = RecurrentMemoryEncoder(
        _recurrent_config(
            cell_kind=cell_kind
        ),
        diagnostics=RecurrentDiagnostics(
            enabled=False
        ),
    ).double()
    history = _history()

    run, report = facade.encode_with_diagnostics(
        history
    )

    assert isinstance(
        run,
        RecurrentSequenceEncoderRun,
    )
    assert report is None
    assert not facade.diagnostics_enabled


def test_diagnose_run_rejects_wrong_type() -> None:
    facade = RecurrentMemoryEncoder(
        _recurrent_config(
            cell_kind=RecurrentCellKind.GRU
        )
    )

    with pytest.raises(TypeError):
        facade.diagnose_run(
            object()  # type: ignore[arg-type]
        )


def test_diagnose_run_rejects_architecture_mismatch() -> None:
    gru = RecurrentMemoryEncoder(
        _recurrent_config(
            cell_kind=RecurrentCellKind.GRU
        )
    ).double()
    lstm = RecurrentMemoryEncoder(
        _recurrent_config(
            cell_kind=RecurrentCellKind.LSTM
        )
    ).double()
    lstm_run = lstm.encode_with_state(
        _history()
    )

    with pytest.raises(ValueError):
        gru.diagnose_run(
            lstm_run
        )


@pytest.mark.parametrize(
    "cell_kind",
    (
        RecurrentCellKind.GRU,
        RecurrentCellKind.LSTM,
    ),
)
def test_all_zero_history_diagnostics(
    cell_kind: RecurrentCellKind,
) -> None:
    facade = RecurrentMemoryEncoder(
        _recurrent_config(
            cell_kind=cell_kind
        )
    ).double()
    history = _history(
        lengths=(
            0,
            0,
            0,
        ),
        with_feature_observed_mask=False,
    )

    run, report = facade.encode_with_diagnostics(
        history
    )

    assert report is not None
    assert report.all_zero_history_short_circuit
    assert report.zero_history_count == 3
    assert report.nonempty_history_count == 0
    assert report.valid_timestep_count == 0
    assert report.padded_timestep_count == 12
    assert not report.adapter_executed
    assert not report.recurrent_kernel_executed
    assert report.execution_is_structurally_consistent
    assert report.exact_zero_padding
    assert report.exact_zero_history_output
    assert report.exact_zero_history_hidden_state
    assert torch.count_nonzero(
        run.public_output.encoded_sequence
    ).item() == 0


# =============================================================================
# Packed/reference comparison diagnostics
# =============================================================================


@pytest.mark.parametrize(
    "cell_kind",
    (
        RecurrentCellKind.GRU,
        RecurrentCellKind.LSTM,
    ),
)
def test_facade_packed_reference_comparison_with_verified_parameters(
    cell_kind: RecurrentCellKind,
) -> None:
    torch.manual_seed(
        105
    )
    packed_config = _recurrent_config(
        cell_kind=cell_kind,
        pack_sequences=True,
    )
    reference_config = _recurrent_config(
        cell_kind=cell_kind,
        pack_sequences=False,
    )
    packed = RecurrentMemoryEncoder(
        packed_config
    ).double()
    reference = RecurrentMemoryEncoder(
        reference_config
    ).double()
    reference.encoder.load_state_dict(
        packed.encoder.state_dict()
    )
    packed.eval()
    reference.eval()
    history = _history()

    packed_snapshot = packed.build_parameter_snapshot()
    reference_snapshot = reference.build_parameter_snapshot()

    assert (
        packed_snapshot.parameter_snapshot_fingerprint
        == reference_snapshot.parameter_snapshot_fingerprint
    )

    packed_run = packed.encode_with_state(
        history,
        parameter_snapshot=packed_snapshot,
    )
    reference_run = reference.encode_with_state(
        history,
        parameter_snapshot=reference_snapshot,
    )

    comparison = packed.compare_execution_runs(
        packed_run,
        reference_run,
    )

    assert isinstance(
        comparison,
        PackedReferenceComparisonDiagnostics,
    )
    assert comparison.encoder_kind == cell_kind.value
    assert comparison.packed_execution_path == "packed"
    assert comparison.reference_execution_path == "reference"
    assert comparison.architecture_match
    assert comparison.source_object_identity
    assert comparison.source_alignment_match
    assert comparison.source_values_match
    assert comparison.source_masks_match
    assert comparison.source_lineage_match
    assert comparison.state_layout_match
    assert not comparison.parameter_snapshot_both_absent
    assert comparison.parameter_snapshot_match
    assert comparison.parameter_identity_verified
    assert not comparison.dropout_active_in_either
    assert comparison.equality_expectation_valid
    assert comparison.output_allclose
    assert comparison.hidden_allclose
    assert comparison.is_numerically_equivalent
    assert comparison.is_controlled_equivalence
    assert comparison.padded_output_max_abs_error == 0.0
    assert comparison.zero_history_output_max_abs_error == 0.0
    assert comparison.zero_history_hidden_max_abs_error == 0.0

    if cell_kind == RecurrentCellKind.GRU:
        assert not comparison.cell_comparison_available
        assert comparison.cell_allclose is None
    else:
        assert comparison.cell_comparison_available
        assert comparison.cell_allclose is True
        assert comparison.zero_history_cell_max_abs_error == 0.0


@pytest.mark.parametrize(
    "cell_kind",
    (
        RecurrentCellKind.GRU,
        RecurrentCellKind.LSTM,
    ),
)
def test_comparison_without_snapshots_is_numerical_but_not_controlled(
    cell_kind: RecurrentCellKind,
) -> None:
    packed = RecurrentMemoryEncoder(
        _recurrent_config(
            cell_kind=cell_kind,
            pack_sequences=True,
        )
    ).double()
    reference = RecurrentMemoryEncoder(
        _recurrent_config(
            cell_kind=cell_kind,
            pack_sequences=False,
        )
    ).double()
    reference.encoder.load_state_dict(
        packed.encoder.state_dict()
    )
    packed.eval()
    reference.eval()
    history = _history()

    comparison = compare_packed_reference_runs(
        packed.encode_with_state(
            history
        ),
        reference.encode_with_state(
            history
        ),
    )

    assert comparison.parameter_snapshot_both_absent
    assert comparison.parameter_snapshot_match
    assert not comparison.parameter_identity_verified
    assert comparison.is_numerically_equivalent
    assert not comparison.is_controlled_equivalence


@pytest.mark.parametrize(
    "cell_kind",
    (
        RecurrentCellKind.GRU,
        RecurrentCellKind.LSTM,
    ),
)
def test_comparison_marks_training_dropout_as_invalid_equality_expectation(
    cell_kind: RecurrentCellKind,
) -> None:
    packed = RecurrentMemoryEncoder(
        _recurrent_config(
            cell_kind=cell_kind,
            pack_sequences=True,
            num_layers=3,
            dropout=0.4,
        )
    ).double()
    reference = RecurrentMemoryEncoder(
        _recurrent_config(
            cell_kind=cell_kind,
            pack_sequences=False,
            num_layers=3,
            dropout=0.4,
        )
    ).double()
    reference.encoder.load_state_dict(
        packed.encoder.state_dict()
    )
    packed.train()
    reference.train()
    history = _history()

    packed_run = packed.encode_with_state(
        history
    )
    reference_run = reference.encode_with_state(
        history
    )
    comparison = compare_packed_reference_runs(
        packed_run,
        reference_run,
    )

    assert comparison.dropout_active_in_either
    assert not comparison.equality_expectation_valid
    assert not comparison.is_controlled_equivalence


def test_comparison_rejects_reversed_execution_paths() -> None:
    packed = RecurrentMemoryEncoder(
        _recurrent_config(
            cell_kind=RecurrentCellKind.GRU,
            pack_sequences=True,
        )
    ).double()
    reference = RecurrentMemoryEncoder(
        _recurrent_config(
            cell_kind=RecurrentCellKind.GRU,
            pack_sequences=False,
        )
    ).double()
    reference.encoder.load_state_dict(
        packed.encoder.state_dict()
    )
    history = _history()
    packed_run = packed.encode_with_state(
        history
    )
    reference_run = reference.encode_with_state(
        history
    )

    with pytest.raises(ValueError):
        compare_packed_reference_runs(
            reference_run,
            packed_run,
        )


def test_comparison_rejects_cross_cell_runs() -> None:
    gru = RecurrentMemoryEncoder(
        _recurrent_config(
            cell_kind=RecurrentCellKind.GRU,
            pack_sequences=True,
        )
    ).double()
    lstm = RecurrentMemoryEncoder(
        _recurrent_config(
            cell_kind=RecurrentCellKind.LSTM,
            pack_sequences=False,
        )
    ).double()
    history = _history()

    with pytest.raises(ValueError):
        compare_packed_reference_runs(
            gru.encode_with_state(
                history
            ),
            lstm.encode_with_state(
                history
            ),
        )


# =============================================================================
# State dictionaries, gradients, and optimizer visibility
# =============================================================================


@pytest.mark.parametrize(
    "cell_kind",
    (
        RecurrentCellKind.GRU,
        RecurrentCellKind.LSTM,
    ),
)
def test_facade_state_dict_round_trip(
    cell_kind: RecurrentCellKind,
) -> None:
    torch.manual_seed(
        106
    )
    config = _recurrent_config(
        cell_kind=cell_kind
    )
    source = RecurrentMemoryEncoder(
        config
    ).double()
    target = RecurrentMemoryEncoder(
        config
    ).double()
    target.load_state_dict(
        copy.deepcopy(
            source.state_dict()
        )
    )
    source.eval()
    target.eval()
    history = _history()

    source_run = source.encode_with_state(
        history
    )
    target_run = target.encode_with_state(
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
def test_facade_gradients_reach_source_and_all_parameters(
    cell_kind: RecurrentCellKind,
) -> None:
    facade = RecurrentMemoryEncoder(
        _recurrent_config(
            cell_kind=cell_kind
        )
    ).double()
    facade.train()
    history = _history(
        requires_grad=True
    )

    run = facade.encode_with_state(
        history
    )
    _weighted_loss(
        run
    ).backward()

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

    for name, parameter in facade.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(
            parameter.grad
        ).all(), name


@pytest.mark.parametrize(
    "cell_kind",
    (
        RecurrentCellKind.GRU,
        RecurrentCellKind.LSTM,
    ),
)
def test_optimizer_sees_all_facade_parameters(
    cell_kind: RecurrentCellKind,
) -> None:
    facade = RecurrentMemoryEncoder(
        _recurrent_config(
            cell_kind=cell_kind
        )
    )
    optimizer = torch.optim.Adam(
        facade.parameters(),
        lr=1e-3,
    )
    optimizer_parameters = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    facade_parameters = {
        id(parameter)
        for parameter in facade.parameters()
    }

    assert optimizer_parameters == facade_parameters


# =============================================================================
# Defensive orchestration behavior
# =============================================================================


@pytest.mark.parametrize(
    "cell_kind",
    (
        RecurrentCellKind.GRU,
        RecurrentCellKind.LSTM,
    ),
)
def test_facade_rejects_wrong_source_type(
    cell_kind: RecurrentCellKind,
) -> None:
    facade = RecurrentMemoryEncoder(
        _recurrent_config(
            cell_kind=cell_kind
        )
    )

    with pytest.raises(TypeError):
        facade(
            object()  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "cell_kind",
    (
        RecurrentCellKind.GRU,
        RecurrentCellKind.LSTM,
    ),
)
def test_facade_rejects_dtype_mismatch(
    cell_kind: RecurrentCellKind,
) -> None:
    facade = RecurrentMemoryEncoder(
        _recurrent_config(
            cell_kind=cell_kind
        )
    ).float()

    with pytest.raises(ValueError):
        facade(
            _history(
                dtype=torch.float64
            )
        )


@pytest.mark.parametrize(
    "cell_kind",
    (
        RecurrentCellKind.GRU,
        RecurrentCellKind.LSTM,
    ),
)
def test_facade_rejects_feature_width_mismatch(
    cell_kind: RecurrentCellKind,
) -> None:
    facade = RecurrentMemoryEncoder(
        _recurrent_config(
            cell_kind=cell_kind,
            input_dim=D,
        )
    ).double()

    with pytest.raises(ValueError):
        facade(
            _history(
                feature_dim=D + 1
            )
        )


def test_facade_rejects_baseline_top_level_config() -> None:
    with pytest.raises(NotImplementedError):
        RecurrentMemoryEncoder(
            TemporalSequenceEncoderConfig.identity(
                input_dim=D
            )
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
def test_recurrent_orchestration_cuda(
    cell_kind: RecurrentCellKind,
    pack_sequences: bool,
) -> None:
    facade = RecurrentMemoryEncoder(
        _recurrent_config(
            cell_kind=cell_kind,
            pack_sequences=pack_sequences,
        )
    ).cuda().float()
    facade.train()
    history = _history(
        dtype=torch.float32,
        device="cuda",
        padding_direction=TemporalPaddingDirection.LEFT,
        requires_grad=True,
    )

    run, report = facade.encode_with_diagnostics(
        history
    )

    assert report is not None
    assert run.device.type == "cuda"
    assert report.is_numerically_clean

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
