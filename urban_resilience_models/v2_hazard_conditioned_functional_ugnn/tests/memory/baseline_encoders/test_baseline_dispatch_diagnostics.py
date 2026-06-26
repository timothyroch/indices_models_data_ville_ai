"""
Consolidated tests for Phase 5 dispatch and detached diagnostics.

Target repository path:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            tests/
                memory/
                    baseline_encoders/
                        test_baseline_dispatch_diagnostics.py

Modules under test:
    memory/baseline_encoders/baseline_encoders.py
    memory/baseline_encoders/diagnostics.py

The suite verifies that:

- sequence encoding and temporal pooling remain separate dispatch stages;
- all implemented Phase 5 configurations construct the correct modules;
- deferred pooling modes fail loudly rather than silently falling back;
- thin execution helpers preserve exact Phase 4 object identities;
- optional parameter snapshots are forwarded without implicit generation;
- diagnostics remain detached, immutable, deterministic, and descriptive;
- clean outputs are reported as clean;
- finite post-construction corruption is detected without changing forward
  behavior;
- combined diagnostics preserve the exact sequence-to-pooling relationship.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import MethodType
from typing import Callable
import json

import pytest
import torch
from torch import nn

import urban_resilience_models.v2_hazard_conditioned_functional_ugnn.memory.baseline_encoders.baseline_encoders as dispatcher_module
import urban_resilience_models.v2_hazard_conditioned_functional_ugnn.memory.baseline_encoders.diagnostics as diagnostics_module
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.memory.baseline_encoders._provenance import (
    build_parameter_snapshot_provenance,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.memory.baseline_encoders.baseline_encoders import (
    BASELINE_ENCODER_DISPATCHER_IMPLEMENTATION_VERSION,
    BASELINE_ENCODER_DISPATCHER_SCIENTIFIC_INTERPRETATION,
    BASELINE_SEQUENCE_ENCODER_TYPES,
    BASELINE_TEMPORAL_POOLER_TYPES,
    IMPLEMENTED_BASELINE_SEQUENCE_ENCODER_KINDS,
    IMPLEMENTED_BASELINE_TEMPORAL_POOLING_KINDS,
    RECOGNIZED_UNIMPLEMENTED_BASELINE_TEMPORAL_POOLING_KINDS,
    BaselineSequenceEncoderModule,
    BaselineTemporalPoolerModule,
    build_baseline_sequence_encoder,
    build_baseline_temporal_pooler,
    build_sequence_encoder,
    build_temporal_pooler,
    encode_baseline_history,
    encode_history,
    is_baseline_sequence_encoder,
    is_baseline_temporal_pooler,
    pool_baseline_sequence,
    pool_sequence,
    run_baseline_pipeline,
    run_pipeline,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.memory.baseline_encoders.diagnostics import (
    BASELINE_DIAGNOSTICS_SCHEMA_VERSION,
    BASELINE_DIAGNOSTICS_SCIENTIFIC_INTERPRETATION,
    BaselineDiagnostics,
    BaselinePoolingDiagnostics,
    BaselineSequenceDiagnostics,
    collect_baseline_diagnostics,
    collect_pooling_baseline_diagnostics,
    collect_sequence_baseline_diagnostics,
    diagnose_baseline_pipeline,
    diagnose_pooling_baseline,
    diagnose_sequence_baseline,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.memory.baseline_encoders.identity_sequence_encoder import (
    IdentitySequenceEncoder,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.memory.baseline_encoders.last_valid_pooling import (
    LastValidTemporalPooler,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.memory.baseline_encoders.linear_projection_sequence_encoder import (
    LinearProjectionSequenceEncoder,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.memory.baseline_encoders.masked_mean_pooling import (
    MaskedMeanTemporalPooler,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.memory.baseline_encoders.pointwise_mlp_sequence_encoder import (
    PointwiseMLPSequenceEncoder,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.memory.config import (
    BaselineSequenceEncoderConfig,
    BaselineSequenceEncoderKind,
    MemoryActivation,
    RecurrentSequenceEncoderConfig,
    TemporalPoolingConfig,
    TemporalSequenceEncoderConfig,
    TransformerSequenceEncoderConfig,
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
    TemporalSequenceEncoding,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.memory.schemas.temporal_coordinates import (
    RelativeTemporalCoordinates,
    TemporalPaddingDirection,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.memory.schemas.temporal_pooling import (
    TemporalPoolingHeadReduction,
    TemporalPoolingKind,
    TemporalPoolingOutput,
    TemporalPoolingZeroHistoryPolicy,
)


N = 3
T = 4
D = 2
H = 5


# =============================================================================
# Factories
# =============================================================================


def _mask(
    *,
    include_zero_history: bool = False,
    padding_direction: TemporalPaddingDirection = (
        TemporalPaddingDirection.RIGHT
    ),
) -> torch.Tensor:
    right = torch.tensor(
        [
            [True, True, True, False],
            [True, True, False, False],
            [True, True, True, True],
        ],
        dtype=torch.bool,
    )

    if include_zero_history:
        right[1] = False

    if padding_direction == TemporalPaddingDirection.RIGHT:
        return right

    left = torch.zeros_like(
        right
    )

    for row, length in enumerate(
        right.sum(
            dim=-1
        ).tolist()
    ):
        if length > 0:
            left[
                row,
                T - length :,
            ] = True

    return left


def _history(
    *,
    include_zero_history: bool = False,
    padding_direction: TemporalPaddingDirection = (
        TemporalPaddingDirection.RIGHT
    ),
    latest_all_features_missing: bool = False,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
    requires_grad: bool = False,
) -> HistoricalSequenceInputs:
    mask = _mask(
        include_zero_history=include_zero_history,
        padding_direction=padding_direction,
    ).to(
        device=device
    )

    logical = (
        torch.arange(
            N * T * D,
            dtype=dtype,
            device=device,
        )
        .reshape(
            N,
            T,
            D,
        )
        / 10.0
        + 0.1
    )

    values = torch.zeros_like(
        logical
    )

    for row in range(
        N
    ):
        valid_indices = torch.nonzero(
            mask[row],
            as_tuple=False,
        ).flatten()
        length = int(
            valid_indices.numel()
        )

        if length > 0:
            values[
                row,
                valid_indices,
            ] = logical[
                row,
                :length,
            ]

    if requires_grad:
        values.requires_grad_()

    observed = (
        mask
        .unsqueeze(-1)
        .expand(
            -1,
            -1,
            D,
        )
        .clone()
    )

    if latest_all_features_missing:
        for row in range(
            N
        ):
            valid_indices = torch.nonzero(
                mask[row],
                as_tuple=False,
            ).flatten()

            if valid_indices.numel() > 0:
                observed[
                    row,
                    int(
                        valid_indices[-1].item()
                    ),
                ] = False

    coordinates = torch.zeros(
        (
            N,
            T,
        ),
        dtype=dtype,
        device=device,
    )

    for row in range(
        N
    ):
        valid_indices = torch.nonzero(
            mask[row],
            as_tuple=False,
        ).flatten()
        length = int(
            valid_indices.numel()
        )

        if length > 0:
            coordinates[
                row,
                valid_indices,
            ] = torch.arange(
                -length,
                0,
                dtype=dtype,
                device=device,
            )

    return HistoricalSequenceInputs(
        history=values,
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
                device=device,
            ),
            graph_count=2,
            graph_ids=(
                "graph-0",
                "graph-1",
            ),
            source_fingerprint="dispatch-node-axis-v1",
        ),
        feature_axis=TemporalFeatureAxis(
            feature_names=(
                "feature-0",
                "feature-1",
            ),
            source_fingerprint="dispatch-feature-axis-v1",
        ),
        temporal_coordinates=RelativeTemporalCoordinates(
            values=coordinates,
            unit="months",
        ),
        source_provenance=MemorySourceProvenance(
            source_name="dispatch-panel",
            source_kind="historical-node-sequence",
            source_fingerprint="dispatch-source-v1",
            preprocessing_fingerprint="dispatch-preprocessing-v1",
            imputation_fingerprint="dispatch-imputation-v1",
        ),
        feature_observed_mask=observed,
        padding_direction=padding_direction,
        zero_length_policy=(
            HistoryZeroLengthPolicy.ALLOW_ZERO_HISTORY
            if include_zero_history
            else HistoryZeroLengthPolicy.ERROR
        ),
    )


def _identity_config() -> BaselineSequenceEncoderConfig:
    return BaselineSequenceEncoderConfig(
        kind=BaselineSequenceEncoderKind.IDENTITY,
        input_dim=D,
        output_dim=D,
    )


def _linear_config(
    *,
    output_dim: int = H,
) -> BaselineSequenceEncoderConfig:
    return BaselineSequenceEncoderConfig(
        kind=BaselineSequenceEncoderKind.LINEAR_PROJECTION,
        input_dim=D,
        output_dim=output_dim,
        activation=MemoryActivation.GELU,
        dropout=0.0,
        layer_normalization=False,
        use_bias=True,
    )


def _mlp_config(
    *,
    output_dim: int = H,
) -> BaselineSequenceEncoderConfig:
    return BaselineSequenceEncoderConfig(
        kind=BaselineSequenceEncoderKind.TEMPORAL_MLP,
        input_dim=D,
        hidden_dim=4,
        output_dim=output_dim,
        num_hidden_layers=2,
        activation=MemoryActivation.GELU,
        dropout=0.0,
        layer_normalization=False,
        use_bias=True,
    )


def _mean_config(
    *,
    output_dim: int = D,
    zero_history_policy: (
        TemporalPoolingZeroHistoryPolicy
        | str
    ) = TemporalPoolingZeroHistoryPolicy.ERROR,
) -> TemporalPoolingConfig:
    return TemporalPoolingConfig(
        kind=TemporalPoolingKind.MASKED_MEAN,
        output_dim=output_dim,
        zero_history_policy=zero_history_policy,
    )


def _last_config(
    *,
    output_dim: int = D,
    zero_history_policy: (
        TemporalPoolingZeroHistoryPolicy
        | str
    ) = TemporalPoolingZeroHistoryPolicy.ERROR,
) -> TemporalPoolingConfig:
    return TemporalPoolingConfig(
        kind=TemporalPoolingKind.LAST_VALID,
        output_dim=output_dim,
        zero_history_policy=zero_history_policy,
    )


def _identity_encoder() -> IdentitySequenceEncoder:
    return IdentitySequenceEncoder(
        _identity_config()
    )


def _linear_encoder(
    *,
    output_dim: int = H,
) -> LinearProjectionSequenceEncoder:
    return LinearProjectionSequenceEncoder(
        _linear_config(
            output_dim=output_dim
        )
    )


def _mlp_encoder(
    *,
    output_dim: int = H,
) -> PointwiseMLPSequenceEncoder:
    return PointwiseMLPSequenceEncoder(
        _mlp_config(
            output_dim=output_dim
        )
    )


def _mean_pooler(
    *,
    output_dim: int = D,
    zero_history_policy: (
        TemporalPoolingZeroHistoryPolicy
        | str
    ) = TemporalPoolingZeroHistoryPolicy.ERROR,
) -> MaskedMeanTemporalPooler:
    return MaskedMeanTemporalPooler(
        _mean_config(
            output_dim=output_dim,
            zero_history_policy=zero_history_policy,
        )
    )


def _last_pooler(
    *,
    output_dim: int = D,
    zero_history_policy: (
        TemporalPoolingZeroHistoryPolicy
        | str
    ) = TemporalPoolingZeroHistoryPolicy.ERROR,
) -> LastValidTemporalPooler:
    return LastValidTemporalPooler(
        _last_config(
            output_dim=output_dim,
            zero_history_policy=zero_history_policy,
        )
    )


def _encode_identity(
    history: HistoricalSequenceInputs,
) -> TemporalSequenceEncoding:
    return _identity_encoder()(
        history
    )


def _build_pipeline(
    *,
    pooling_kind: TemporalPoolingKind = (
        TemporalPoolingKind.MASKED_MEAN
    ),
    latest_all_features_missing: bool = False,
    include_zero_history: bool = False,
    zero_history_policy: (
        TemporalPoolingZeroHistoryPolicy
        | str
    ) = TemporalPoolingZeroHistoryPolicy.ERROR,
) -> tuple[
    IdentitySequenceEncoder,
    TemporalSequenceEncoding,
    nn.Module,
    TemporalPoolingOutput,
]:
    history = _history(
        latest_all_features_missing=(
            latest_all_features_missing
        ),
        include_zero_history=include_zero_history,
    )
    encoder = _identity_encoder()
    encoding = encoder(
        history
    )

    if pooling_kind == TemporalPoolingKind.MASKED_MEAN:
        pooler = _mean_pooler(
            zero_history_policy=zero_history_policy
        )
    else:
        pooler = _last_pooler(
            zero_history_policy=zero_history_policy
        )

    pooling = pooler(
        encoding
    )

    return (
        encoder,
        encoding,
        pooler,
        pooling,
    )


# =============================================================================
# Module metadata and aliases
# =============================================================================


@pytest.mark.parametrize(
    "value",
    (
        BASELINE_ENCODER_DISPATCHER_IMPLEMENTATION_VERSION,
        BASELINE_ENCODER_DISPATCHER_SCIENTIFIC_INTERPRETATION,
        BASELINE_DIAGNOSTICS_SCHEMA_VERSION,
        BASELINE_DIAGNOSTICS_SCIENTIFIC_INTERPRETATION,
    ),
)
def test_module_identity_strings_are_nonempty(
    value: str,
) -> None:
    assert isinstance(
        value,
        str,
    )
    assert value.strip()


def test_dispatcher_capability_vocabularies_are_exact() -> None:
    assert IMPLEMENTED_BASELINE_SEQUENCE_ENCODER_KINDS == (
        "identity",
        "linear_projection",
        "temporal_mlp",
    )
    assert IMPLEMENTED_BASELINE_TEMPORAL_POOLING_KINDS == (
        "masked_mean",
        "last_valid",
    )
    assert (
        RECOGNIZED_UNIMPLEMENTED_BASELINE_TEMPORAL_POOLING_KINDS
        == (
            "masked_max",
            "learned_attention",
            "multihead_attention",
            "other",
        )
    )


def test_dispatcher_aliases_preserve_exact_functions() -> None:
    assert build_sequence_encoder is (
        build_baseline_sequence_encoder
    )
    assert build_temporal_pooler is (
        build_baseline_temporal_pooler
    )
    assert encode_history is encode_baseline_history
    assert pool_sequence is pool_baseline_sequence
    assert run_pipeline is run_baseline_pipeline


def test_diagnostic_aliases_preserve_exact_functions() -> None:
    assert collect_sequence_baseline_diagnostics is (
        diagnose_sequence_baseline
    )
    assert collect_pooling_baseline_diagnostics is (
        diagnose_pooling_baseline
    )
    assert collect_baseline_diagnostics is (
        diagnose_baseline_pipeline
    )


def test_dispatcher_module_all_has_no_duplicates_and_resolves() -> None:
    exported = dispatcher_module.__all__

    assert len(
        exported
    ) == len(
        set(
            exported
        )
    )

    for name in exported:
        assert hasattr(
            dispatcher_module,
            name,
        )


def test_diagnostics_module_all_has_no_duplicates_and_resolves() -> None:
    exported = diagnostics_module.__all__

    assert len(
        exported
    ) == len(
        set(
            exported
        )
    )

    for name in exported:
        assert hasattr(
            diagnostics_module,
            name,
        )


# =============================================================================
# Predicates and supported module type tuples
# =============================================================================


@pytest.mark.parametrize(
    "module",
    (
        pytest.param(
            _identity_encoder(),
            id="identity",
        ),
        pytest.param(
            _linear_encoder(),
            id="linear",
        ),
        pytest.param(
            _mlp_encoder(),
            id="mlp",
        ),
    ),
)
def test_sequence_predicate_accepts_implemented_modules(
    module: nn.Module,
) -> None:
    assert is_baseline_sequence_encoder(
        module
    )
    assert isinstance(
        module,
        BASELINE_SEQUENCE_ENCODER_TYPES,
    )


@pytest.mark.parametrize(
    "module",
    (
        pytest.param(
            _mean_pooler(),
            id="masked-mean",
        ),
        pytest.param(
            _last_pooler(),
            id="last-valid",
        ),
    ),
)
def test_pooler_predicate_accepts_implemented_modules(
    module: nn.Module,
) -> None:
    assert is_baseline_temporal_pooler(
        module
    )
    assert isinstance(
        module,
        BASELINE_TEMPORAL_POOLER_TYPES,
    )


def test_predicates_reject_opposite_stage_and_unrelated_objects() -> None:
    encoder = _identity_encoder()
    pooler = _mean_pooler()

    assert not is_baseline_sequence_encoder(
        pooler
    )
    assert not is_baseline_temporal_pooler(
        encoder
    )
    assert not is_baseline_sequence_encoder(
        nn.Linear(
            2,
            3,
        )
    )
    assert not is_baseline_temporal_pooler(
        object()
    )


def test_type_aliases_accept_concrete_modules_at_runtime() -> None:
    sequence: BaselineSequenceEncoderModule = (
        _identity_encoder()
    )
    pooler: BaselineTemporalPoolerModule = (
        _mean_pooler()
    )

    assert is_baseline_sequence_encoder(
        sequence
    )
    assert is_baseline_temporal_pooler(
        pooler
    )


# =============================================================================
# Sequence builders
# =============================================================================


@pytest.mark.parametrize(
    (
        "config",
        "expected_type",
    ),
    (
        (
            _identity_config(),
            IdentitySequenceEncoder,
        ),
        (
            _linear_config(),
            LinearProjectionSequenceEncoder,
        ),
        (
            _mlp_config(),
            PointwiseMLPSequenceEncoder,
        ),
    ),
)
def test_direct_sequence_builder_dispatch(
    config: BaselineSequenceEncoderConfig,
    expected_type: type[nn.Module],
) -> None:
    module = build_baseline_sequence_encoder(
        config
    )

    assert isinstance(
        module,
        expected_type,
    )
    assert module.config is config


def test_wrapped_identity_sequence_builder_dispatch() -> None:
    config = TemporalSequenceEncoderConfig.identity(
        input_dim=D
    )
    module = build_baseline_sequence_encoder(
        config
    )

    assert isinstance(
        module,
        IdentitySequenceEncoder,
    )
    assert module.input_dim == D
    assert module.output_dim == D


def test_wrapped_temporal_mlp_sequence_builder_dispatch() -> None:
    config = TemporalSequenceEncoderConfig.temporal_mlp(
        input_dim=D,
        hidden_dim=4,
        output_dim=H,
    )
    module = build_baseline_sequence_encoder(
        config
    )

    assert isinstance(
        module,
        PointwiseMLPSequenceEncoder,
    )
    assert module.output_dim == H


def test_sequence_builder_returns_fresh_modules() -> None:
    config = _linear_config()
    first = build_baseline_sequence_encoder(
        config
    )
    second = build_baseline_sequence_encoder(
        config
    )

    assert first is not second
    assert next(
        first.parameters()
    ) is not next(
        second.parameters()
    )


@pytest.mark.parametrize(
    "bad_config",
    (
        None,
        object(),
        _mean_config(),
    ),
)
def test_sequence_builder_rejects_wrong_config_type(
    bad_config: object,
) -> None:
    with pytest.raises(TypeError):
        build_baseline_sequence_encoder(
            bad_config  # type: ignore[arg-type]
        )


def test_sequence_builder_rejects_recurrent_dispatch_config() -> None:
    config = TemporalSequenceEncoderConfig.recurrent_encoder(
        RecurrentSequenceEncoderConfig(
            input_dim=D,
            hidden_dim=H,
        )
    )

    with pytest.raises(NotImplementedError):
        build_baseline_sequence_encoder(
            config
        )


def test_sequence_builder_rejects_transformer_dispatch_config() -> None:
    config = TemporalSequenceEncoderConfig.transformer_encoder(
        TransformerSequenceEncoderConfig(
            input_dim=D,
            model_dim=4,
            num_heads=2,
        )
    )

    with pytest.raises(NotImplementedError):
        build_baseline_sequence_encoder(
            config
        )


# =============================================================================
# Pooler builders
# =============================================================================


@pytest.mark.parametrize(
    (
        "config",
        "expected_type",
    ),
    (
        (
            _mean_config(),
            MaskedMeanTemporalPooler,
        ),
        (
            _last_config(),
            LastValidTemporalPooler,
        ),
    ),
)
def test_pooler_builder_dispatch(
    config: TemporalPoolingConfig,
    expected_type: type[nn.Module],
) -> None:
    module = build_baseline_temporal_pooler(
        config
    )

    assert isinstance(
        module,
        expected_type,
    )
    assert module.config is config


def test_pooler_builder_returns_fresh_modules() -> None:
    config = _mean_config()
    first = build_baseline_temporal_pooler(
        config
    )
    second = build_baseline_temporal_pooler(
        config
    )

    assert first is not second


@pytest.mark.parametrize(
    "bad_config",
    (
        None,
        object(),
        _identity_config(),
    ),
)
def test_pooler_builder_rejects_wrong_config_type(
    bad_config: object,
) -> None:
    with pytest.raises(TypeError):
        build_baseline_temporal_pooler(
            bad_config  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "config",
    (
        TemporalPoolingConfig(
            kind=TemporalPoolingKind.MASKED_MAX,
            output_dim=D,
        ),
        TemporalPoolingConfig(
            kind=TemporalPoolingKind.LEARNED_ATTENTION,
            output_dim=D,
            score_hidden_dim=4,
        ),
        TemporalPoolingConfig(
            kind=TemporalPoolingKind.MULTIHEAD_ATTENTION,
            output_dim=D,
            num_heads=2,
            head_reduction=TemporalPoolingHeadReduction.MEAN,
            score_hidden_dim=4,
        ),
        TemporalPoolingConfig(
            kind=TemporalPoolingKind.OTHER,
            output_dim=D,
        ),
    ),
)
def test_pooler_builder_rejects_recognized_deferred_modes(
    config: TemporalPoolingConfig,
) -> None:
    with pytest.raises(NotImplementedError):
        build_baseline_temporal_pooler(
            config
        )


# =============================================================================
# Execution helpers
# =============================================================================


@pytest.mark.parametrize(
    "encoder",
    (
        pytest.param(
            _identity_encoder(),
            id="identity",
        ),
        pytest.param(
            _linear_encoder(),
            id="linear",
        ),
        pytest.param(
            _mlp_encoder(),
            id="mlp",
        ),
    ),
)
def test_encode_helper_preserves_exact_history(
    encoder: nn.Module,
) -> None:
    history = _history()
    output = encode_baseline_history(
        encoder,  # type: ignore[arg-type]
        history,
    )

    assert isinstance(
        output,
        TemporalSequenceEncoding,
    )
    assert output.source_history is history


@pytest.mark.parametrize(
    "pooler",
    (
        pytest.param(
            _mean_pooler(),
            id="mean",
        ),
        pytest.param(
            _last_pooler(),
            id="last",
        ),
    ),
)
def test_pool_helper_preserves_exact_encoding(
    pooler: nn.Module,
) -> None:
    encoding = _encode_identity(
        _history()
    )
    output = pool_baseline_sequence(
        pooler,  # type: ignore[arg-type]
        encoding,
    )

    assert isinstance(
        output,
        TemporalPoolingOutput,
    )
    assert output.source_encoding is encoding


def test_encode_helper_rejects_pooler() -> None:
    with pytest.raises(TypeError):
        encode_baseline_history(
            _mean_pooler(),  # type: ignore[arg-type]
            _history(),
        )


def test_pool_helper_rejects_sequence_encoder() -> None:
    encoding = _encode_identity(
        _history()
    )

    with pytest.raises(TypeError):
        pool_baseline_sequence(
            _identity_encoder(),  # type: ignore[arg-type]
            encoding,
        )


def test_encode_helper_rejects_wrong_history_type() -> None:
    with pytest.raises(TypeError):
        encode_baseline_history(
            _identity_encoder(),
            object(),  # type: ignore[arg-type]
        )


def test_pool_helper_rejects_wrong_encoding_type() -> None:
    with pytest.raises(TypeError):
        pool_baseline_sequence(
            _mean_pooler(),
            object(),  # type: ignore[arg-type]
        )


def test_encode_helper_forwards_explicit_parameter_snapshot() -> None:
    history = _history()
    encoder = _linear_encoder()
    snapshot = build_parameter_snapshot_provenance(
        encoder,
        checkpoint_id="linear-v1",
    )

    output = encode_baseline_history(
        encoder,
        history,
        parameter_snapshot=snapshot,
    )

    assert (
        output
        .computation_provenance
        .parameter_snapshot
        is snapshot
    )


def test_pool_helper_forwards_explicit_parameter_snapshot() -> None:
    encoding = _encode_identity(
        _history()
    )
    pooler = _mean_pooler()
    snapshot = build_parameter_snapshot_provenance(
        pooler,
        checkpoint_id="mean-v1",
    )

    output = pool_baseline_sequence(
        pooler,
        encoding,
        parameter_snapshot=snapshot,
    )

    assert (
        output
        .computation_provenance
        .parameter_snapshot
        is snapshot
    )


def test_helpers_do_not_generate_parameter_snapshots() -> None:
    history = _history()
    encoder = _linear_encoder()
    encoding = encode_baseline_history(
        encoder,
        history,
    )
    pooler = _mean_pooler(
        output_dim=H
    )
    pooling = pool_baseline_sequence(
        pooler,
        encoding,
    )

    assert (
        encoding
        .computation_provenance
        .parameter_snapshot
        is None
    )
    assert (
        pooling
        .computation_provenance
        .parameter_snapshot
        is None
    )


def test_encode_helper_uses_module_call_hooks() -> None:
    history = _history()
    encoder = _identity_encoder()
    observed: list[
        bool
    ] = []

    handle = encoder.register_forward_hook(
        lambda module, inputs, output: observed.append(
            True
        )
    )

    try:
        encode_baseline_history(
            encoder,
            history,
        )
    finally:
        handle.remove()

    assert observed == [
        True
    ]


def test_pool_helper_uses_module_call_hooks() -> None:
    encoding = _encode_identity(
        _history()
    )
    pooler = _mean_pooler()
    observed: list[
        bool
    ] = []

    handle = pooler.register_forward_hook(
        lambda module, inputs, output: observed.append(
            True
        )
    )

    try:
        pool_baseline_sequence(
            pooler,
            encoding,
        )
    finally:
        handle.remove()

    assert observed == [
        True
    ]


def test_encode_helper_rejects_invalid_runtime_contract() -> None:
    encoder = _identity_encoder()

    def wrong_forward(
        self: IdentitySequenceEncoder,
        source_history: HistoricalSequenceInputs,
        *,
        parameter_snapshot: object = None,
    ) -> object:
        return object()

    encoder.forward = MethodType(  # type: ignore[method-assign]
        wrong_forward,
        encoder,
    )

    with pytest.raises(RuntimeError):
        encode_baseline_history(
            encoder,
            _history(),
        )


def test_pool_helper_rejects_invalid_runtime_contract() -> None:
    pooler = _mean_pooler()

    def wrong_forward(
        self: MaskedMeanTemporalPooler,
        source_encoding: TemporalSequenceEncoding,
        *,
        parameter_snapshot: object = None,
    ) -> object:
        return object()

    pooler.forward = MethodType(  # type: ignore[method-assign]
        wrong_forward,
        pooler,
    )

    with pytest.raises(RuntimeError):
        pool_baseline_sequence(
            pooler,
            _encode_identity(
                _history()
            ),
        )


def test_encode_helper_rejects_changed_source_identity() -> None:
    first_history = _history()
    second_history = _history(
        padding_direction=TemporalPaddingDirection.LEFT
    )
    encoder = _identity_encoder()
    alternate_output = _identity_encoder()(
        second_history
    )

    def wrong_forward(
        self: IdentitySequenceEncoder,
        source_history: HistoricalSequenceInputs,
        *,
        parameter_snapshot: object = None,
    ) -> TemporalSequenceEncoding:
        return alternate_output

    encoder.forward = MethodType(  # type: ignore[method-assign]
        wrong_forward,
        encoder,
    )

    with pytest.raises(RuntimeError):
        encode_baseline_history(
            encoder,
            first_history,
        )


def test_pool_helper_rejects_changed_source_identity() -> None:
    first_encoding = _encode_identity(
        _history()
    )
    second_encoding = _encode_identity(
        _history(
            padding_direction=TemporalPaddingDirection.LEFT
        )
    )
    pooler = _mean_pooler()
    alternate_output = _mean_pooler()(
        second_encoding
    )

    def wrong_forward(
        self: MaskedMeanTemporalPooler,
        source_encoding: TemporalSequenceEncoding,
        *,
        parameter_snapshot: object = None,
    ) -> TemporalPoolingOutput:
        return alternate_output

    pooler.forward = MethodType(  # type: ignore[method-assign]
        wrong_forward,
        pooler,
    )

    with pytest.raises(RuntimeError):
        pool_baseline_sequence(
            pooler,
            first_encoding,
        )


# =============================================================================
# Pipeline execution
# =============================================================================


def test_sequence_only_pipeline_returns_no_pooling() -> None:
    history = _history()
    encoder = _identity_encoder()

    sequence, pooling = run_baseline_pipeline(
        encoder,
        history,
    )

    assert sequence.source_history is history
    assert pooling is None


@pytest.mark.parametrize(
    "pooler",
    (
        pytest.param(
            _mean_pooler(),
            id="mean",
        ),
        pytest.param(
            _last_pooler(),
            id="last",
        ),
    ),
)
def test_pipeline_preserves_stage_object_identity(
    pooler: nn.Module,
) -> None:
    history = _history()
    encoder = _identity_encoder()

    sequence, pooling = run_baseline_pipeline(
        encoder,
        history,
        temporal_pooler=pooler,  # type: ignore[arg-type]
    )

    assert pooling is not None
    assert sequence.source_history is history
    assert pooling.source_encoding is sequence


def test_pipeline_forwards_both_parameter_snapshots() -> None:
    history = _history()
    encoder = _identity_encoder()
    pooler = _mean_pooler()
    sequence_snapshot = build_parameter_snapshot_provenance(
        encoder
    )
    pooling_snapshot = build_parameter_snapshot_provenance(
        pooler
    )

    sequence, pooling = run_baseline_pipeline(
        encoder,
        history,
        temporal_pooler=pooler,
        sequence_parameter_snapshot=(
            sequence_snapshot
        ),
        pooling_parameter_snapshot=(
            pooling_snapshot
        ),
    )

    assert pooling is not None
    assert (
        sequence
        .computation_provenance
        .parameter_snapshot
        is sequence_snapshot
    )
    assert (
        pooling
        .computation_provenance
        .parameter_snapshot
        is pooling_snapshot
    )


def test_pipeline_rejects_pooling_snapshot_without_pooler() -> None:
    pooler = _mean_pooler()
    snapshot = build_parameter_snapshot_provenance(
        pooler
    )

    with pytest.raises(ValueError):
        run_baseline_pipeline(
            _identity_encoder(),
            _history(),
            pooling_parameter_snapshot=snapshot,
        )


# =============================================================================
# Sequence diagnostics
# =============================================================================


@pytest.mark.parametrize(
    "encoder",
    (
        pytest.param(
            _identity_encoder(),
            id="identity",
        ),
        pytest.param(
            _linear_encoder(),
            id="linear",
        ),
        pytest.param(
            _mlp_encoder(),
            id="mlp",
        ),
    ),
)
def test_sequence_diagnostics_cover_all_encoder_families(
    encoder: nn.Module,
) -> None:
    history = _history()
    output = encoder(
        history
    )
    diagnostics = diagnose_sequence_baseline(
        encoder,
        output,
    )

    assert isinstance(
        diagnostics,
        BaselineSequenceDiagnostics,
    )
    assert diagnostics.component_name == (
        encoder.__class__.__name__
    )
    assert diagnostics.node_count == N
    assert diagnostics.sequence_length == T
    assert diagnostics.hidden_dim == output.hidden_dim
    assert diagnostics.parameter_count == sum(
        parameter.numel()
        for parameter in encoder.parameters()
    )
    assert diagnostics.trainable_parameter_count == sum(
        parameter.numel()
        for parameter in encoder.parameters()
        if parameter.requires_grad
    )
    assert diagnostics.nonfinite_output_count == 0
    assert diagnostics.max_abs_padded_output == 0.0
    assert diagnostics.exact_zero_padding
    assert diagnostics.is_numerically_clean


def test_sequence_diagnostics_valid_length_statistics() -> None:
    encoder = _identity_encoder()
    output = encoder(
        _history()
    )
    diagnostics = diagnose_sequence_baseline(
        encoder,
        output,
    )

    assert diagnostics.zero_history_count == 0
    assert diagnostics.nonempty_history_count == 3
    assert diagnostics.valid_timestep_count == 9
    assert diagnostics.padded_timestep_count == 3
    assert diagnostics.valid_length_min == 2
    assert diagnostics.valid_length_mean == pytest.approx(
        3.0
    )
    assert diagnostics.valid_length_max == 4


def test_sequence_diagnostics_zero_history_statistics() -> None:
    encoder = _identity_encoder()
    output = encoder(
        _history(
            include_zero_history=True
        )
    )
    diagnostics = diagnose_sequence_baseline(
        encoder,
        output,
    )

    assert diagnostics.zero_history_count == 1
    assert diagnostics.nonempty_history_count == 2
    assert diagnostics.valid_length_min == 0


def test_sequence_diagnostics_optional_pre_mask_tensor() -> None:
    encoder = _linear_encoder()
    history = _history()
    output = encoder(
        history
    )
    pre_mask = encoder.projection(
        history.history
    )

    diagnostics = diagnose_sequence_baseline(
        encoder,
        output,
        pre_mask_output=pre_mask,
    )

    assert diagnostics.pre_mask_output_supplied
    assert diagnostics.nonfinite_pre_mask_count == 0


def test_sequence_diagnostics_count_nonfinite_pre_mask_values() -> None:
    encoder = _identity_encoder()
    output = encoder(
        _history()
    )
    pre_mask = output.encoded_sequence.detach().clone()
    pre_mask[0, 0, 0] = float(
        "nan"
    )
    pre_mask[0, 0, 1] = float(
        "inf"
    )

    diagnostics = diagnose_sequence_baseline(
        encoder,
        output,
        pre_mask_output=pre_mask,
    )

    assert diagnostics.nonfinite_pre_mask_count == 2
    assert not diagnostics.is_numerically_clean


def test_sequence_diagnostics_reject_wrong_pre_mask_shape() -> None:
    encoder = _identity_encoder()
    output = encoder(
        _history()
    )

    with pytest.raises(ValueError):
        diagnose_sequence_baseline(
            encoder,
            output,
            pre_mask_output=torch.zeros(
                1
            ),
        )


def test_sequence_diagnostics_support_custom_component_name() -> None:
    encoder = _identity_encoder()
    output = encoder(
        _history()
    )
    diagnostics = diagnose_sequence_baseline(
        encoder,
        output,
        component_name="raw-history-control",
    )

    assert diagnostics.component_name == (
        "raw-history-control"
    )


@pytest.mark.parametrize(
    "bad_module,bad_output",
    (
        (
            object(),
            None,
        ),
        (
            nn.Identity(),
            object(),
        ),
    ),
)
def test_sequence_diagnostics_reject_wrong_types(
    bad_module: object,
    bad_output: object,
) -> None:
    if not isinstance(
        bad_module,
        nn.Module,
    ):
        with pytest.raises(TypeError):
            diagnose_sequence_baseline(
                bad_module,  # type: ignore[arg-type]
                _encode_identity(
                    _history()
                ),
            )
    else:
        with pytest.raises(TypeError):
            diagnose_sequence_baseline(
                bad_module,
                bad_output,  # type: ignore[arg-type]
            )


# =============================================================================
# Pooling diagnostics
# =============================================================================


@pytest.mark.parametrize(
    "pooling_kind",
    (
        TemporalPoolingKind.MASKED_MEAN,
        TemporalPoolingKind.LAST_VALID,
    ),
)
def test_pooling_diagnostics_cover_both_poolers(
    pooling_kind: TemporalPoolingKind,
) -> None:
    _, encoding, pooler, output = _build_pipeline(
        pooling_kind=pooling_kind
    )
    diagnostics = diagnose_pooling_baseline(
        pooler,
        output,
    )

    assert isinstance(
        diagnostics,
        BaselinePoolingDiagnostics,
    )
    assert diagnostics.pooling_kind == pooling_kind.value
    assert diagnostics.node_count == N
    assert diagnostics.sequence_length == T
    assert diagnostics.output_dim == D
    assert diagnostics.num_heads == 1
    assert diagnostics.parameter_count == 0
    assert diagnostics.trainable_parameter_count == 0
    assert diagnostics.nonfinite_weight_count == 0
    assert diagnostics.negative_weight_count == 0
    assert diagnostics.nonfinite_pooled_output_count == 0
    assert diagnostics.max_normalization_error == 0.0
    assert diagnostics.max_padded_pooling_mass == 0.0
    assert diagnostics.deterministic_weight_max_error == 0.0
    assert diagnostics.pooled_reconstruction_max_error == 0.0
    assert diagnostics.exact_source_encoding_identity
    assert diagnostics.source_lineage_fingerprint == (
        encoding.lineage_fingerprint()
    )
    assert diagnostics.is_numerically_clean


def test_mean_diagnostics_have_no_last_valid_selection_fields() -> None:
    _, _, pooler, output = _build_pipeline(
        pooling_kind=TemporalPoolingKind.MASKED_MEAN
    )
    diagnostics = diagnose_pooling_baseline(
        pooler,
        output,
    )

    assert (
        diagnostics
        .last_valid_selection_mismatch_count
        is None
    )
    assert not (
        diagnostics
        .selected_all_features_missing_status_available
    )
    assert (
        diagnostics
        .selected_all_features_missing_count
        is None
    )
    assert (
        diagnostics
        .selected_all_features_missing_fraction
        is None
    )


def test_last_valid_diagnostics_report_all_missing_selection() -> None:
    _, _, pooler, output = _build_pipeline(
        pooling_kind=TemporalPoolingKind.LAST_VALID,
        latest_all_features_missing=True,
    )
    diagnostics = diagnose_pooling_baseline(
        pooler,
        output,
    )

    assert (
        diagnostics
        .last_valid_selection_mismatch_count
        == 0
    )
    assert (
        diagnostics
        .selected_all_features_missing_status_available
    )
    assert (
        diagnostics
        .selected_all_features_missing_count
        == N
    )
    assert (
        diagnostics
        .selected_all_features_missing_fraction
        == 1.0
    )


@pytest.mark.parametrize(
    "pooling_kind",
    (
        TemporalPoolingKind.MASKED_MEAN,
        TemporalPoolingKind.LAST_VALID,
    ),
)
def test_pooling_diagnostics_zero_history_cleanliness(
    pooling_kind: TemporalPoolingKind,
) -> None:
    _, _, pooler, output = _build_pipeline(
        pooling_kind=pooling_kind,
        include_zero_history=True,
        zero_history_policy=(
            TemporalPoolingZeroHistoryPolicy.ZERO
        ),
    )
    diagnostics = diagnose_pooling_baseline(
        pooler,
        output,
    )

    assert diagnostics.zero_history_count == 1
    assert diagnostics.nonempty_history_count == 2
    assert diagnostics.max_zero_history_weight_mass == 0.0
    assert (
        diagnostics
        .max_abs_zero_history_pooled_memory
        == 0.0
    )
    assert diagnostics.is_numerically_clean


def test_pooling_diagnostics_detect_finite_weight_corruption() -> None:
    _, _, pooler, output = _build_pipeline(
        pooling_kind=TemporalPoolingKind.MASKED_MEAN
    )

    with torch.no_grad():
        output.pooling_weights[
            0,
            0,
            3,
        ] = 0.25

    diagnostics = diagnose_pooling_baseline(
        pooler,
        output,
    )

    assert diagnostics.max_padded_pooling_mass == pytest.approx(
        0.25
    )
    assert diagnostics.max_normalization_error == pytest.approx(
        0.25
    )
    assert diagnostics.deterministic_weight_max_error == pytest.approx(
        0.25
    )
    assert diagnostics.pooled_reconstruction_max_error == 0.0
    # The corrupted mass was placed on a padded encoded vector, which is
    # canonically zero. Weight-policy diagnostics detect the violation even
    # though the weighted reconstruction remains numerically unchanged.
    assert not diagnostics.is_numerically_clean


def test_pooling_diagnostics_detect_valid_weight_reconstruction_mismatch() -> None:
    _, _, pooler, output = _build_pipeline(
        pooling_kind=TemporalPoolingKind.MASKED_MEAN
    )

    with torch.no_grad():
        output.pooling_weights[
            0,
            0,
            0,
        ] += 0.25

    diagnostics = diagnose_pooling_baseline(
        pooler,
        output,
    )

    assert diagnostics.pooled_reconstruction_max_error is not None
    assert diagnostics.pooled_reconstruction_max_error > 0.0
    assert diagnostics.max_normalization_error == pytest.approx(
        0.25
    )
    assert diagnostics.deterministic_weight_max_error == pytest.approx(
        0.25
    )
    assert not diagnostics.is_numerically_clean


def test_pooling_diagnostics_detect_negative_weight() -> None:
    _, _, pooler, output = _build_pipeline(
        pooling_kind=TemporalPoolingKind.MASKED_MEAN
    )

    with torch.no_grad():
        output.pooling_weights[
            0,
            0,
            0,
        ] = -0.1

    diagnostics = diagnose_pooling_baseline(
        pooler,
        output,
    )

    assert diagnostics.negative_weight_count == 1
    assert not diagnostics.is_numerically_clean


def test_last_valid_diagnostics_detect_selection_mismatch() -> None:
    _, _, pooler, output = _build_pipeline(
        pooling_kind=TemporalPoolingKind.LAST_VALID
    )

    with torch.no_grad():
        output.pooling_weights[
            0
        ].zero_()
        output.pooling_weights[
            0,
            0,
            0,
        ] = 1.0

    diagnostics = diagnose_pooling_baseline(
        pooler,
        output,
    )

    assert (
        diagnostics
        .last_valid_selection_mismatch_count
        == 1
    )
    assert not diagnostics.is_numerically_clean


def test_pooling_diagnostics_support_custom_component_name() -> None:
    _, _, pooler, output = _build_pipeline()
    diagnostics = diagnose_pooling_baseline(
        pooler,
        output,
        component_name="generic-mean-control",
    )

    assert diagnostics.component_name == (
        "generic-mean-control"
    )


@pytest.mark.parametrize(
    "bad_module,bad_output",
    (
        (
            object(),
            None,
        ),
        (
            nn.Identity(),
            object(),
        ),
    ),
)
def test_pooling_diagnostics_reject_wrong_types(
    bad_module: object,
    bad_output: object,
) -> None:
    _, _, pooler, output = _build_pipeline()

    if not isinstance(
        bad_module,
        nn.Module,
    ):
        with pytest.raises(TypeError):
            diagnose_pooling_baseline(
                bad_module,  # type: ignore[arg-type]
                output,
            )
    else:
        with pytest.raises(TypeError):
            diagnose_pooling_baseline(
                bad_module,
                bad_output,  # type: ignore[arg-type]
            )


# =============================================================================
# Combined pipeline diagnostics
# =============================================================================


@pytest.mark.parametrize(
    "pooling_kind",
    (
        TemporalPoolingKind.MASKED_MEAN,
        TemporalPoolingKind.LAST_VALID,
    ),
)
def test_combined_diagnostics_preserve_stage_relationship(
    pooling_kind: TemporalPoolingKind,
) -> None:
    encoder, encoding, pooler, output = _build_pipeline(
        pooling_kind=pooling_kind
    )

    diagnostics = diagnose_baseline_pipeline(
        encoder,
        encoding,
        pooling_module=pooler,
        pooling_output=output,
    )

    assert isinstance(
        diagnostics,
        BaselineDiagnostics,
    )
    assert diagnostics.pooling_present
    assert diagnostics.pooling is not None
    assert diagnostics.pooling_source_is_sequence is True
    assert diagnostics.sequence.output_lineage_fingerprint == (
        encoding.lineage_fingerprint()
    )
    assert diagnostics.pooling.output_lineage_fingerprint == (
        output.lineage_fingerprint()
    )
    assert diagnostics.is_numerically_clean


def test_combined_diagnostics_sequence_only_mode() -> None:
    history = _history()
    encoder = _identity_encoder()
    encoding = encoder(
        history
    )

    diagnostics = diagnose_baseline_pipeline(
        encoder,
        encoding,
    )

    assert diagnostics.pooling is None
    assert not diagnostics.pooling_present
    assert diagnostics.pooling_source_is_sequence is None
    assert diagnostics.is_numerically_clean


def test_combined_diagnostics_reject_partial_pooling_arguments() -> None:
    encoder, encoding, pooler, output = _build_pipeline()

    with pytest.raises(ValueError):
        diagnose_baseline_pipeline(
            encoder,
            encoding,
            pooling_module=pooler,
        )

    with pytest.raises(ValueError):
        diagnose_baseline_pipeline(
            encoder,
            encoding,
            pooling_output=output,
        )


def test_combined_diagnostics_detect_wrong_pooling_source_identity() -> None:
    encoder = _identity_encoder()
    first_encoding = encoder(
        _history()
    )
    second_encoding = encoder(
        _history(
            padding_direction=TemporalPaddingDirection.LEFT
        )
    )
    pooler = _mean_pooler()
    second_pooling = pooler(
        second_encoding
    )

    diagnostics = diagnose_baseline_pipeline(
        encoder,
        first_encoding,
        pooling_module=pooler,
        pooling_output=second_pooling,
    )

    assert diagnostics.pooling_present
    assert diagnostics.pooling_source_is_sequence is False
    assert not diagnostics.is_numerically_clean


# =============================================================================
# Diagnostic serialization, immutability, and dataclass validation
# =============================================================================


def test_sequence_diagnostics_serialization_is_deterministic() -> None:
    encoder = _identity_encoder()
    output = encoder(
        _history()
    )
    diagnostics = diagnose_sequence_baseline(
        encoder,
        output,
    )

    first = diagnostics.to_json()
    second = diagnostics.to_json()

    assert first == second
    assert json.loads(
        first
    ) == diagnostics.to_dict()
    assert diagnostics.fingerprint() == (
        diagnostics.fingerprint()
    )


def test_pooling_diagnostics_serialization_is_deterministic() -> None:
    _, _, pooler, output = _build_pipeline()
    diagnostics = diagnose_pooling_baseline(
        pooler,
        output,
    )

    first = diagnostics.to_json()
    second = diagnostics.to_json()

    assert first == second
    assert json.loads(
        first
    ) == diagnostics.to_dict()
    assert diagnostics.fingerprint() == (
        diagnostics.fingerprint()
    )


def test_combined_diagnostics_serialization_is_deterministic() -> None:
    encoder, encoding, pooler, output = _build_pipeline()
    diagnostics = diagnose_baseline_pipeline(
        encoder,
        encoding,
        pooling_module=pooler,
        pooling_output=output,
    )

    first = diagnostics.to_json()
    second = diagnostics.to_json()

    assert first == second
    assert json.loads(
        first
    ) == diagnostics.to_dict()
    assert diagnostics.fingerprint() == (
        diagnostics.fingerprint()
    )


@pytest.mark.parametrize(
    "diagnostics",
    (
        pytest.param(
            diagnose_sequence_baseline(
                _identity_encoder(),
                _encode_identity(
                    _history()
                ),
            ),
            id="sequence",
        ),
        pytest.param(
            (
                lambda data: diagnose_pooling_baseline(
                    data[2],
                    data[3],
                )
            )(
                _build_pipeline()
            ),
            id="pooling",
        ),
        pytest.param(
            (
                lambda data: diagnose_baseline_pipeline(
                    data[0],
                    data[1],
                    pooling_module=data[2],
                    pooling_output=data[3],
                )
            )(
                _build_pipeline()
            ),
            id="combined",
        ),
    ),
)
def test_diagnostic_artifacts_are_frozen(
    diagnostics: object,
) -> None:
    with pytest.raises(
        (
            FrozenInstanceError,
            AttributeError,
        )
    ):
        diagnostics.schema_version = "changed"  # type: ignore[attr-defined]


def test_sequence_diagnostics_reject_inconsistent_counts() -> None:
    with pytest.raises(ValueError):
        BaselineSequenceDiagnostics(
            component_name="x",
            node_count=2,
            sequence_length=3,
            hidden_dim=4,
            parameter_count=0,
            trainable_parameter_count=0,
            module_training=False,
            zero_history_count=2,
            nonempty_history_count=1,
            valid_timestep_count=4,
            padded_timestep_count=2,
            valid_length_min=0,
            valid_length_mean=2.0,
            valid_length_max=3,
            pre_mask_output_supplied=False,
            nonfinite_pre_mask_count=None,
            nonfinite_output_count=0,
            max_abs_padded_output=0.0,
            mean_valid_vector_l2_norm=1.0,
            max_valid_vector_l2_norm=2.0,
            exact_zero_padding=True,
            output_requires_grad=False,
            architecture_fingerprint="a",
            source_lineage_fingerprint="b",
            output_lineage_fingerprint="c",
        )


def test_pooling_diagnostics_reject_fraction_above_one() -> None:
    with pytest.raises(ValueError):
        BaselinePoolingDiagnostics(
            component_name="x",
            pooling_kind="last_valid",
            node_count=1,
            sequence_length=2,
            output_dim=3,
            num_heads=1,
            parameter_count=0,
            trainable_parameter_count=0,
            module_training=False,
            zero_history_count=0,
            nonempty_history_count=1,
            nonfinite_weight_count=0,
            negative_weight_count=0,
            nonfinite_pooled_output_count=0,
            max_normalization_error=0.0,
            max_padded_pooling_mass=0.0,
            max_zero_history_weight_mass=0.0,
            max_abs_zero_history_pooled_memory=0.0,
            deterministic_weight_max_error=0.0,
            pooled_reconstruction_max_error=0.0,
            mean_pooled_vector_l2_norm=1.0,
            max_pooled_vector_l2_norm=1.0,
            last_valid_selection_mismatch_count=0,
            selected_all_features_missing_status_available=True,
            selected_all_features_missing_count=1,
            selected_all_features_missing_fraction=1.1,
            exact_source_encoding_identity=True,
            pooled_output_requires_grad=False,
            architecture_fingerprint="a",
            source_lineage_fingerprint="b",
            output_lineage_fingerprint="c",
        )


def test_combined_diagnostics_reject_presence_mismatch() -> None:
    sequence = diagnose_sequence_baseline(
        _identity_encoder(),
        _encode_identity(
            _history()
        ),
    )

    with pytest.raises(ValueError):
        BaselineDiagnostics(
            sequence=sequence,
            pooling=None,
            pooling_present=True,
            pooling_source_is_sequence=True,
        )


# =============================================================================
# Diagnostic detachment and state preservation
# =============================================================================


@pytest.mark.parametrize(
    "module_factory",
    (
        _linear_encoder,
        _mlp_encoder,
    ),
)
def test_sequence_diagnostics_do_not_change_module_state(
    module_factory: Callable[[], nn.Module],
) -> None:
    module = module_factory()
    module.train()
    history = _history(
        requires_grad=True
    )
    output = module(
        history
    )
    state_before = {
        key: value.detach().clone()
        for key, value in module.state_dict().items()
    }

    diagnose_sequence_baseline(
        module,
        output,
    )

    assert module.training

    for key, value in module.state_dict().items():
        assert torch.equal(
            value,
            state_before[
                key
            ],
        )


@pytest.mark.parametrize(
    "pooler_factory",
    (
        _mean_pooler,
        _last_pooler,
    ),
)
def test_pooling_diagnostics_do_not_change_tensors(
    pooler_factory: Callable[[], nn.Module],
) -> None:
    encoding = _encode_identity(
        _history()
    )
    pooler = pooler_factory()
    output = pooler(
        encoding
    )
    weights_before = (
        output
        .pooling_weights
        .detach()
        .clone()
    )
    pooled_before = (
        output
        .pooled_memory
        .detach()
        .clone()
    )

    diagnose_pooling_baseline(
        pooler,
        output,
    )

    assert torch.equal(
        output.pooling_weights,
        weights_before,
    )
    assert torch.equal(
        output.pooled_memory,
        pooled_before,
    )


def test_diagnostics_do_not_trigger_backward() -> None:
    history = _history(
        requires_grad=True
    )
    encoder = _linear_encoder(
        output_dim=D
    )
    encoding = encoder(
        history
    )
    pooler = _mean_pooler()
    pooling = pooler(
        encoding
    )

    diagnose_baseline_pipeline(
        encoder,
        encoding,
        pooling_module=pooler,
        pooling_output=pooling,
    )

    assert history.history.grad is None

    for parameter in encoder.parameters():
        assert parameter.grad is None


# =============================================================================
# Dtype and conditional device coverage
# =============================================================================


def test_diagnostics_support_float64() -> None:
    history = _history(
        dtype=torch.float64
    )
    encoder = _identity_encoder().double()
    encoding = encoder(
        history
    )
    pooler = _mean_pooler()
    pooling = pooler(
        encoding
    )

    diagnostics = diagnose_baseline_pipeline(
        encoder,
        encoding,
        pooling_module=pooler,
        pooling_output=pooling,
    )

    assert diagnostics.is_numerically_clean


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is not available.",
)
def test_dispatch_and_diagnostics_execute_on_cuda() -> None:
    history = _history(
        device="cuda"
    )
    encoder = build_baseline_sequence_encoder(
        _linear_config(
            output_dim=D
        )
    ).to(
        "cuda"
    )
    pooler = build_baseline_temporal_pooler(
        _mean_config()
    ).to(
        "cuda"
    )

    encoding, pooling = run_baseline_pipeline(
        encoder,
        history,
        temporal_pooler=pooler,
    )

    assert pooling is not None
    diagnostics = diagnose_baseline_pipeline(
        encoder,
        encoding,
        pooling_module=pooler,
        pooling_output=pooling,
    )

    assert encoding.device.type == "cuda"
    assert pooling.device.type == "cuda"
    assert diagnostics.is_numerically_clean
