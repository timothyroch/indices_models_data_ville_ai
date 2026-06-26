"""
Consolidated tests for deterministic Phase 5 temporal pooling baselines.

Target repository path:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            tests/
                memory/
                    baseline_encoders/
                        test_pooling_baselines.py

Implementations under test:
    memory/baseline_encoders/masked_mean_pooling.py
    memory/baseline_encoders/last_valid_pooling.py

The suite freezes the distinction between:

``MaskedMeanTemporalPooler``
    Uniform, order-insensitive reduction over valid temporal positions.

``LastValidTemporalPooler``
    Order-sensitive selection of the greatest valid temporal index.

Both implementations:

- consume the canonical ``TemporalSequenceEncoding`` contract;
- return the canonical ``TemporalPoolingOutput`` contract;
- preserve the exact source-encoding object;
- use deterministic weights ``[N, 1, T]``;
- remain parameter-free and hazard-independent;
- support zero-history policies ``error`` and ``zero``;
- reject learned fallbacks and hidden output projection.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import MappingProxyType
from typing import Callable

import pytest
import torch
from torch import nn

from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.memory.baseline_encoders._provenance import (
    build_parameter_snapshot_provenance,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.memory.baseline_encoders.identity_sequence_encoder import (
    IdentitySequenceEncoder,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.memory.baseline_encoders.last_valid_pooling import (
    LAST_VALID_POOLING_COMPONENT_KIND,
    LAST_VALID_POOLING_COMPONENT_NAME,
    LAST_VALID_POOLING_HAZARD_CONDITIONED,
    LAST_VALID_POOLING_IMPLEMENTATION_VERSION,
    LAST_VALID_POOLING_MISSINGNESS_POLICY,
    LAST_VALID_POOLING_OPERATION_NAME,
    LAST_VALID_POOLING_PADDING_POLICY,
    LAST_VALID_POOLING_PROJECTION_POLICY,
    LAST_VALID_POOLING_TEMPORAL_INTERACTION,
    LAST_VALID_POOLING_WEIGHT_POLICY,
    LAST_VALID_POOLING_ZERO_HISTORY_POLICIES,
    LastObservationTemporalPooler,
    LastValidPooling,
    LastValidTemporalPooler,
    LastValidTemporalPooling,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.memory.baseline_encoders.linear_projection_sequence_encoder import (
    LinearProjectionSequenceEncoder,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.memory.baseline_encoders.masked_mean_pooling import (
    MASKED_MEAN_POOLING_COMPONENT_KIND,
    MASKED_MEAN_POOLING_COMPONENT_NAME,
    MASKED_MEAN_POOLING_HAZARD_CONDITIONED,
    MASKED_MEAN_POOLING_IMPLEMENTATION_VERSION,
    MASKED_MEAN_POOLING_OPERATION_NAME,
    MASKED_MEAN_POOLING_PADDING_POLICY,
    MASKED_MEAN_POOLING_PROJECTION_POLICY,
    MASKED_MEAN_POOLING_TEMPORAL_INTERACTION,
    MASKED_MEAN_POOLING_WEIGHT_POLICY,
    MASKED_MEAN_POOLING_ZERO_HISTORY_POLICIES,
    MaskedMeanPooling,
    MaskedMeanTemporalPooler,
    MaskedMeanTemporalPooling,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.memory.baseline_encoders.pointwise_mlp_sequence_encoder import (
    PointwiseMLPSequenceEncoder,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.memory.config import (
    BaselineSequenceEncoderConfig,
    BaselineSequenceEncoderKind,
    MemoryActivation,
    TemporalPoolingConfig,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.memory.schemas.history_inputs import (
    HistoricalSequenceInputs,
    HistoryZeroLengthPolicy,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.memory.schemas.provenance import (
    MemoryParameterSnapshotProvenance,
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


def _right_mask(
    *,
    include_zero_history: bool = False,
    sequence_length: int = T,
) -> torch.Tensor:
    if sequence_length < 3:
        raise ValueError(
            "sequence_length must be at least three for these tests."
        )

    mask = torch.zeros(
        (
            N,
            sequence_length,
        ),
        dtype=torch.bool,
    )
    mask[0, : sequence_length - 1] = True
    mask[1, : max(sequence_length - 2, 1)] = True
    mask[2, :] = True

    if include_zero_history:
        mask[1] = False

    return mask


def _left_mask(
    *,
    include_zero_history: bool = False,
    sequence_length: int = T,
) -> torch.Tensor:
    right = _right_mask(
        include_zero_history=include_zero_history,
        sequence_length=sequence_length,
    )
    left = torch.zeros_like(
        right
    )

    for row, length in enumerate(
        right.sum(dim=-1).tolist()
    ):
        if length:
            left[
                row,
                sequence_length - length :,
            ] = True

    return left


def _relative_coordinates(
    mask: torch.Tensor,
    *,
    dtype: torch.dtype,
    device: torch.device | str,
    gap_scale: float = 1.0,
) -> torch.Tensor:
    coordinates = torch.zeros(
        mask.shape,
        dtype=dtype,
        device=device,
    )

    for row in range(
        int(mask.shape[0])
    ):
        valid_indices = torch.nonzero(
            mask[row],
            as_tuple=False,
        ).flatten()
        length = int(
            valid_indices.numel()
        )

        if length == 0:
            continue

        # Irregular but strictly increasing oldest-to-newest offsets.
        base = torch.arange(
            -length,
            0,
            dtype=dtype,
            device=device,
        )
        irregular = (
            base
            * float(
                gap_scale
            )
        )
        coordinates[
            row,
            valid_indices,
        ] = irregular

    return coordinates


def _logical_sequences(
    *,
    feature_dim: int = D,
    sequence_length: int = T,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> list[torch.Tensor]:
    lengths = (
        sequence_length - 1,
        max(sequence_length - 2, 1),
        sequence_length,
    )

    sequences: list[
        torch.Tensor
    ] = []

    cursor = 1.0

    for length in lengths:
        values = torch.arange(
            cursor,
            cursor
            + length * feature_dim,
            dtype=dtype,
            device=device,
        ).reshape(
            length,
            feature_dim,
        )
        sequences.append(
            values
        )
        cursor += length * feature_dim

    return sequences


def _pack_sequences(
    sequences: list[torch.Tensor],
    *,
    sequence_length: int,
    padding_direction: TemporalPaddingDirection,
    include_zero_history: bool,
    requires_grad: bool,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
]:
    device = sequences[0].device
    dtype = sequences[0].dtype
    feature_dim = int(
        sequences[0].shape[-1]
    )

    mask = (
        _right_mask(
            include_zero_history=include_zero_history,
            sequence_length=sequence_length,
        )
        if padding_direction
        == TemporalPaddingDirection.RIGHT
        else _left_mask(
            include_zero_history=include_zero_history,
            sequence_length=sequence_length,
        )
    ).to(
        device=device
    )

    values = torch.zeros(
        (
            N,
            sequence_length,
            feature_dim,
        ),
        dtype=dtype,
        device=device,
    )

    for row, sequence in enumerate(
        sequences
    ):
        if (
            include_zero_history
            and row == 1
        ):
            continue

        length = int(
            sequence.shape[0]
        )

        if padding_direction == TemporalPaddingDirection.RIGHT:
            values[
                row,
                :length,
            ] = sequence
        else:
            values[
                row,
                sequence_length - length :,
            ] = sequence

    if requires_grad:
        values.requires_grad_()

    return (
        values,
        mask,
    )


def _history(
    *,
    feature_dim: int = D,
    sequence_length: int = T,
    padding_direction: TemporalPaddingDirection | str = (
        TemporalPaddingDirection.RIGHT
    ),
    include_zero_history: bool = False,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
    requires_grad: bool = False,
    include_feature_observed_mask: bool = True,
    latest_all_features_missing: bool = False,
    gap_scale: float = 1.0,
    sequences: list[torch.Tensor] | None = None,
) -> HistoricalSequenceInputs:
    padding_direction = TemporalPaddingDirection(
        padding_direction
    )

    if padding_direction == TemporalPaddingDirection.NONE:
        raise ValueError(
            "This test factory uses padded variable-length histories."
        )

    if sequences is None:
        sequences = _logical_sequences(
            feature_dim=feature_dim,
            sequence_length=sequence_length,
            dtype=dtype,
            device=device,
        )

    values, mask = _pack_sequences(
        sequences,
        sequence_length=sequence_length,
        padding_direction=padding_direction,
        include_zero_history=include_zero_history,
        requires_grad=requires_grad,
    )

    observed = None

    if include_feature_observed_mask:
        observed = (
            mask
            .unsqueeze(-1)
            .expand(
                -1,
                -1,
                feature_dim,
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
            source_fingerprint="pooling-node-axis-v1",
        ),
        feature_axis=TemporalFeatureAxis(
            feature_names=tuple(
                f"feature-{index}"
                for index in range(
                    feature_dim
                )
            ),
            source_fingerprint="pooling-feature-axis-v1",
        ),
        temporal_coordinates=RelativeTemporalCoordinates(
            values=_relative_coordinates(
                mask,
                dtype=dtype,
                device=device,
                gap_scale=gap_scale,
            ),
            unit="months",
        ),
        source_provenance=MemorySourceProvenance(
            source_name="phase-five-pooling-panel",
            source_kind="historical-node-sequence",
            source_fingerprint="pooling-history-source-v1",
            preprocessing_fingerprint="pooling-preprocessing-v1",
            imputation_fingerprint="pooling-imputation-v1",
        ),
        feature_observed_mask=observed,
        padding_direction=padding_direction,
        zero_length_policy=(
            HistoryZeroLengthPolicy.ALLOW_ZERO_HISTORY
            if include_zero_history
            else HistoryZeroLengthPolicy.ERROR
        ),
    )


def _identity_encoding(
    history: HistoricalSequenceInputs,
) -> TemporalSequenceEncoding:
    return IdentitySequenceEncoder(
        BaselineSequenceEncoderConfig(
            kind=BaselineSequenceEncoderKind.IDENTITY,
            input_dim=history.feature_dim,
            output_dim=history.feature_dim,
        )
    )(
        history
    )


def _linear_encoding(
    history: HistoricalSequenceInputs,
    *,
    output_dim: int = H,
) -> TemporalSequenceEncoding:
    encoder = LinearProjectionSequenceEncoder(
        BaselineSequenceEncoderConfig(
            kind=BaselineSequenceEncoderKind.LINEAR_PROJECTION,
            input_dim=history.feature_dim,
            output_dim=output_dim,
            dropout=0.0,
            layer_normalization=False,
            use_bias=True,
        )
    )
    encoder.eval()
    return encoder(
        history
    )


def _mlp_encoding(
    history: HistoricalSequenceInputs,
    *,
    output_dim: int = H,
) -> TemporalSequenceEncoding:
    encoder = PointwiseMLPSequenceEncoder(
        BaselineSequenceEncoderConfig(
            kind=BaselineSequenceEncoderKind.TEMPORAL_MLP,
            input_dim=history.feature_dim,
            hidden_dim=4,
            output_dim=output_dim,
            num_hidden_layers=2,
            activation=MemoryActivation.GELU,
            dropout=0.0,
            layer_normalization=False,
            use_bias=True,
        )
    )
    encoder.eval()
    return encoder(
        history
    )


def _mean_config(
    *,
    output_dim: int = D,
    zero_history_policy: (
        TemporalPoolingZeroHistoryPolicy
        | str
    ) = TemporalPoolingZeroHistoryPolicy.ERROR,
    project_output: bool = False,
) -> TemporalPoolingConfig:
    return TemporalPoolingConfig(
        kind=TemporalPoolingKind.MASKED_MEAN,
        output_dim=output_dim,
        project_output=project_output,
        zero_history_policy=zero_history_policy,
    )


def _last_config(
    *,
    output_dim: int = D,
    zero_history_policy: (
        TemporalPoolingZeroHistoryPolicy
        | str
    ) = TemporalPoolingZeroHistoryPolicy.ERROR,
    project_output: bool = False,
) -> TemporalPoolingConfig:
    return TemporalPoolingConfig(
        kind=TemporalPoolingKind.LAST_VALID,
        output_dim=output_dim,
        project_output=project_output,
        zero_history_policy=zero_history_policy,
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


PoolerFactory = Callable[
    [],
    nn.Module,
]


@pytest.fixture(
    params=(
        "masked_mean",
        "last_valid",
    )
)
def pooler_and_encoding(
    request: pytest.FixtureRequest,
) -> tuple[
    nn.Module,
    TemporalSequenceEncoding,
]:
    encoding = _identity_encoding(
        _history(
            requires_grad=True
        )
    )

    if request.param == "masked_mean":
        return (
            _mean_pooler(),
            encoding,
        )

    return (
        _last_pooler(),
        encoding,
    )


def _permute_history(
    history: HistoricalSequenceInputs,
    permutation: torch.Tensor,
) -> HistoricalSequenceInputs:
    return history.replace(
        history=history.history.index_select(
            0,
            permutation,
        ),
        timestep_mask=history.timestep_mask.index_select(
            0,
            permutation,
        ),
        node_axis=TemporalNodeAxis(
            node_ids=tuple(
                history.node_ids[
                    int(index)
                ]
                for index in permutation.tolist()
            ),
            node_batch_index=(
                history
                .node_batch_index
                .index_select(
                    0,
                    permutation,
                )
            ),
            graph_count=history.graph_count,
            graph_ids=history.graph_ids,
            source_fingerprint=(
                history
                .node_axis
                .source_fingerprint
            ),
        ),
        temporal_coordinates=RelativeTemporalCoordinates(
            values=(
                history
                .temporal_coordinates
                .values
                .index_select(
                    0,
                    permutation,
                )
            ),
            unit=history.temporal_coordinates.unit,
            anchor=history.temporal_coordinates.anchor,
            anchor_source_fingerprint=(
                history
                .temporal_coordinates
                .anchor_source_fingerprint
            ),
            layout=history.temporal_coordinates.layout,
            chronological_order=(
                history
                .temporal_coordinates
                .chronological_order
            ),
            duplicate_policy=(
                history
                .temporal_coordinates
                .duplicate_policy
            ),
            regular_step=(
                history
                .temporal_coordinates
                .regular_step
            ),
        ),
        feature_observed_mask=(
            history
            .feature_observed_mask
            .index_select(
                0,
                permutation,
            )
            if history.feature_observed_mask is not None
            else None
        ),
    )


def _last_indices(
    mask: torch.Tensor,
) -> torch.Tensor:
    indices = torch.arange(
        mask.shape[1],
        dtype=torch.long,
        device=mask.device,
    ).unsqueeze(0).expand_as(
        mask
    )

    return torch.where(
        mask,
        indices,
        torch.full_like(
            indices,
            -1,
        ),
    ).max(
        dim=-1
    ).values


# =============================================================================
# Component identity and aliases
# =============================================================================


def test_pooling_aliases_preserve_exact_classes() -> None:
    assert MaskedMeanPooling is MaskedMeanTemporalPooler
    assert MaskedMeanTemporalPooling is MaskedMeanTemporalPooler
    assert LastValidPooling is LastValidTemporalPooler
    assert LastValidTemporalPooling is LastValidTemporalPooler
    assert LastObservationTemporalPooler is LastValidTemporalPooler


@pytest.mark.parametrize(
    "value",
    (
        MASKED_MEAN_POOLING_COMPONENT_NAME,
        MASKED_MEAN_POOLING_COMPONENT_KIND,
        MASKED_MEAN_POOLING_OPERATION_NAME,
        MASKED_MEAN_POOLING_IMPLEMENTATION_VERSION,
        MASKED_MEAN_POOLING_WEIGHT_POLICY,
        MASKED_MEAN_POOLING_PADDING_POLICY,
        MASKED_MEAN_POOLING_PROJECTION_POLICY,
        LAST_VALID_POOLING_COMPONENT_NAME,
        LAST_VALID_POOLING_COMPONENT_KIND,
        LAST_VALID_POOLING_OPERATION_NAME,
        LAST_VALID_POOLING_IMPLEMENTATION_VERSION,
        LAST_VALID_POOLING_WEIGHT_POLICY,
        LAST_VALID_POOLING_PADDING_POLICY,
        LAST_VALID_POOLING_PROJECTION_POLICY,
        LAST_VALID_POOLING_MISSINGNESS_POLICY,
    ),
)
def test_pooling_identity_strings_are_nonempty(
    value: str,
) -> None:
    assert isinstance(
        value,
        str,
    )
    assert value.strip()


def test_poolers_declare_no_temporal_interaction() -> None:
    assert MASKED_MEAN_POOLING_TEMPORAL_INTERACTION is False
    assert LAST_VALID_POOLING_TEMPORAL_INTERACTION is False


def test_poolers_declare_hazard_independence() -> None:
    assert MASKED_MEAN_POOLING_HAZARD_CONDITIONED is False
    assert LAST_VALID_POOLING_HAZARD_CONDITIONED is False


def test_supported_zero_history_vocabularies_are_exact() -> None:
    expected = (
        "error",
        "zero",
    )

    assert MASKED_MEAN_POOLING_ZERO_HISTORY_POLICIES == expected
    assert LAST_VALID_POOLING_ZERO_HISTORY_POLICIES == expected


# =============================================================================
# Construction and configuration boundaries
# =============================================================================


def test_masked_mean_constructor_accepts_valid_config() -> None:
    pooler = MaskedMeanTemporalPooler(
        _mean_config()
    )

    assert pooler.pooling_kind == TemporalPoolingKind.MASKED_MEAN
    assert pooler.output_dim == D
    assert pooler.num_heads == 1
    assert (
        pooler.head_reduction
        == TemporalPoolingHeadReduction.SINGLE_HEAD
    )


def test_last_valid_constructor_accepts_valid_config() -> None:
    pooler = LastValidTemporalPooler(
        _last_config()
    )

    assert pooler.pooling_kind == TemporalPoolingKind.LAST_VALID
    assert pooler.output_dim == D
    assert pooler.num_heads == 1
    assert (
        pooler.head_reduction
        == TemporalPoolingHeadReduction.SINGLE_HEAD
    )


def test_masked_mean_rejects_wrong_kind() -> None:
    with pytest.raises(ValueError):
        MaskedMeanTemporalPooler(
            _last_config()
        )


def test_last_valid_rejects_wrong_kind() -> None:
    with pytest.raises(ValueError):
        LastValidTemporalPooler(
            _mean_config()
        )


@pytest.mark.parametrize(
    "constructor,config",
    (
        (
            MaskedMeanTemporalPooler,
            _mean_config(
                project_output=True
            ),
        ),
        (
            LastValidTemporalPooler,
            _last_config(
                project_output=True
            ),
        ),
    ),
)
def test_phase_five_poolers_reject_output_projection(
    constructor: Callable[
        [TemporalPoolingConfig],
        nn.Module,
    ],
    config: TemporalPoolingConfig,
) -> None:
    with pytest.raises(NotImplementedError):
        constructor(
            config
        )


@pytest.mark.parametrize(
    "constructor,config",
    (
        (
            MaskedMeanTemporalPooler,
            _mean_config(
                zero_history_policy=(
                    TemporalPoolingZeroHistoryPolicy.LEARNED_FALLBACK
                )
            ),
        ),
        (
            LastValidTemporalPooler,
            _last_config(
                zero_history_policy=(
                    TemporalPoolingZeroHistoryPolicy.LEARNED_FALLBACK
                )
            ),
        ),
    ),
)
def test_deterministic_poolers_reject_learned_fallback(
    constructor: Callable[
        [TemporalPoolingConfig],
        nn.Module,
    ],
    config: TemporalPoolingConfig,
) -> None:
    with pytest.raises(NotImplementedError):
        constructor(
            config
        )


@pytest.mark.parametrize(
    "pooler",
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
def test_wrong_source_type_is_rejected(
    pooler: nn.Module,
) -> None:
    with pytest.raises(TypeError):
        pooler(
            object()  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "pooler_factory",
    (
        _mean_pooler,
        _last_pooler,
    ),
)
def test_output_width_mismatch_is_rejected(
    pooler_factory: Callable[..., nn.Module],
) -> None:
    encoding = _identity_encoding(
        _history()
    )
    pooler = pooler_factory(
        output_dim=D + 1
    )

    with pytest.raises(ValueError):
        pooler(
            encoding
        )


# =============================================================================
# Shared canonical output contract
# =============================================================================


def test_poolers_return_canonical_output(
    pooler_and_encoding: tuple[
        nn.Module,
        TemporalSequenceEncoding,
    ],
) -> None:
    pooler, encoding = pooler_and_encoding
    output = pooler(
        encoding
    )

    assert isinstance(
        output,
        TemporalPoolingOutput,
    )
    assert output.source_encoding is encoding
    assert output.source_sequence_encoding is encoding
    assert output.source_history is encoding.source_history
    assert output.pooled_shape == (
        N,
        D,
    )
    assert output.weight_shape == (
        N,
        1,
        T,
    )
    assert output.num_heads == 1
    assert output.output_dim == D


def test_poolers_preserve_exact_source_metadata(
    pooler_and_encoding: tuple[
        nn.Module,
        TemporalSequenceEncoding,
    ],
) -> None:
    pooler, encoding = pooler_and_encoding
    output = pooler(
        encoding
    )

    assert output.timestep_mask is encoding.timestep_mask
    assert output.node_axis is encoding.node_axis
    assert output.feature_axis is encoding.feature_axis
    assert output.temporal_coordinates is encoding.temporal_coordinates
    assert output.node_batch_index is encoding.node_batch_index
    assert output.node_ids == encoding.node_ids
    assert output.graph_ids == encoding.graph_ids


def test_pooling_weights_match_source_dtype_and_device(
    pooler_and_encoding: tuple[
        nn.Module,
        TemporalSequenceEncoding,
    ],
) -> None:
    pooler, encoding = pooler_and_encoding
    output = pooler(
        encoding
    )

    assert output.pooling_weights.dtype == encoding.dtype
    assert output.pooling_weights.device == encoding.device
    assert output.pooled_memory.dtype == encoding.dtype
    assert output.pooled_memory.device == encoding.device


def test_pooling_weights_are_finite_and_nonnegative(
    pooler_and_encoding: tuple[
        nn.Module,
        TemporalSequenceEncoding,
    ],
) -> None:
    pooler, encoding = pooler_and_encoding
    output = pooler(
        encoding
    )

    assert bool(
        torch.isfinite(
            output.pooling_weights
        ).all().item()
    )
    assert bool(
        (
            output.pooling_weights
            >= 0
        ).all().item()
    )


def test_pooling_weights_have_exact_zero_padding(
    pooler_and_encoding: tuple[
        nn.Module,
        TemporalSequenceEncoding,
    ],
) -> None:
    pooler, encoding = pooler_and_encoding
    output = pooler(
        encoding
    )
    padded = (
        ~encoding.timestep_mask
    ).unsqueeze(
        1
    ).expand_as(
        output.pooling_weights
    )

    assert torch.equal(
        output.pooling_weights[
            padded
        ],
        torch.zeros_like(
            output.pooling_weights[
                padded
            ]
        ),
    )


def test_nonempty_pooling_weights_have_unit_mass(
    pooler_and_encoding: tuple[
        nn.Module,
        TemporalSequenceEncoding,
    ],
) -> None:
    pooler, encoding = pooler_and_encoding
    output = pooler(
        encoding
    )

    sums = output.pooling_weights.sum(
        dim=-1
    )
    nonempty = encoding.valid_lengths > 0

    assert torch.allclose(
        sums[
            nonempty
        ],
        torch.ones_like(
            sums[
                nonempty
            ]
        ),
    )


def test_pooling_output_has_no_head_reduction_weights(
    pooler_and_encoding: tuple[
        nn.Module,
        TemporalSequenceEncoding,
    ],
) -> None:
    pooler, encoding = pooler_and_encoding
    output = pooler(
        encoding
    )

    assert output.head_reduction_weights is None
    assert (
        output.head_reduction
        == TemporalPoolingHeadReduction.SINGLE_HEAD
    )


def test_pooling_output_is_frozen(
    pooler_and_encoding: tuple[
        nn.Module,
        TemporalSequenceEncoding,
    ],
) -> None:
    pooler, encoding = pooler_and_encoding
    output = pooler(
        encoding
    )

    with pytest.raises(
        (
            FrozenInstanceError,
            AttributeError,
        )
    ):
        output.pooling_name = "changed"  # type: ignore[misc]


def test_architecture_metadata_is_immutable(
    pooler_and_encoding: tuple[
        nn.Module,
        TemporalSequenceEncoding,
    ],
) -> None:
    pooler, encoding = pooler_and_encoding
    output = pooler(
        encoding
    )
    metadata = (
        output
        .computation_provenance
        .architecture
        .architecture_metadata
    )

    assert isinstance(
        metadata,
        MappingProxyType,
    )

    with pytest.raises(TypeError):
        metadata[
            "changed"
        ] = True  # type: ignore[index]


def test_repeated_forward_is_fingerprint_deterministic(
    pooler_and_encoding: tuple[
        nn.Module,
        TemporalSequenceEncoding,
    ],
) -> None:
    pooler, encoding = pooler_and_encoding

    first = pooler(
        encoding
    )
    second = pooler(
        encoding
    )

    assert first.semantic_fingerprint() == second.semantic_fingerprint()
    assert first.value_fingerprint() == second.value_fingerprint()
    assert first.lineage_fingerprint() == second.lineage_fingerprint()


# =============================================================================
# Exact masked-mean semantics
# =============================================================================


@pytest.mark.parametrize(
    "padding_direction",
    (
        TemporalPaddingDirection.RIGHT,
        TemporalPaddingDirection.LEFT,
    ),
)
def test_masked_mean_exact_weights(
    padding_direction: TemporalPaddingDirection,
) -> None:
    history = _history(
        padding_direction=padding_direction
    )
    encoding = _identity_encoding(
        history
    )
    output = _mean_pooler()(
        encoding
    )

    lengths = history.valid_lengths.to(
        dtype=history.dtype
    )
    expected = (
        history.timestep_mask.to(
            dtype=history.dtype
        )
        / lengths.unsqueeze(-1)
    ).unsqueeze(1)

    assert torch.equal(
        output.pooling_weights,
        expected,
    )


@pytest.mark.parametrize(
    "padding_direction",
    (
        TemporalPaddingDirection.RIGHT,
        TemporalPaddingDirection.LEFT,
    ),
)
def test_masked_mean_exact_values(
    padding_direction: TemporalPaddingDirection,
) -> None:
    history = _history(
        padding_direction=padding_direction
    )
    encoding = _identity_encoding(
        history
    )
    output = _mean_pooler()(
        encoding
    )

    expected = []

    for row in range(
        N
    ):
        expected.append(
            history.history[
                row,
                history.timestep_mask[
                    row
                ],
            ].mean(
                dim=0
            )
        )

    assert torch.equal(
        output.pooled_memory,
        torch.stack(
            expected
        ),
    )


def test_masked_mean_is_order_insensitive() -> None:
    sequences = _logical_sequences()
    reversed_sequences = [
        sequence.flip(
            dims=(0,)
        )
        for sequence in sequences
    ]

    original = _identity_encoding(
        _history(
            sequences=sequences
        )
    )
    reversed_encoding = _identity_encoding(
        _history(
            sequences=reversed_sequences
        )
    )

    original_output = _mean_pooler()(
        original
    )
    reversed_output = _mean_pooler()(
        reversed_encoding
    )

    assert torch.equal(
        original_output.pooled_memory,
        reversed_output.pooled_memory,
    )


def test_masked_mean_gradient_is_uniform_over_valid_timesteps() -> None:
    history = _history(
        requires_grad=True
    )
    output = _mean_pooler()(
        _identity_encoding(
            history
        )
    )

    output.pooled_memory.sum().backward()

    assert history.history.grad is not None

    expected = (
        history
        .timestep_mask
        .to(
            dtype=history.dtype
        )
        / history
        .valid_lengths
        .to(
            dtype=history.dtype
        )
        .unsqueeze(-1)
    ).unsqueeze(-1).expand_as(
        history.history
    )

    assert torch.equal(
        history.history.grad,
        expected,
    )


def test_masked_mean_architecture_metadata_is_explicit() -> None:
    metadata = _mean_pooler().architecture_metadata()

    assert metadata["pooling_kind"] == "masked_mean"
    assert metadata["trainable_temporal_weights"] is False
    assert metadata["order_sensitive_pooling"] is False
    assert metadata["hazard_conditioned"] is False
    assert metadata["temporal_interaction"] is False


def test_masked_mean_extra_repr_is_informative() -> None:
    representation = _mean_pooler().extra_repr()

    assert "head_reduction=single_head" in representation
    assert "parameters=0" in representation
    assert "temporal_interaction=False" in representation


# =============================================================================
# Exact last-valid semantics
# =============================================================================


@pytest.mark.parametrize(
    "padding_direction",
    (
        TemporalPaddingDirection.RIGHT,
        TemporalPaddingDirection.LEFT,
    ),
)
def test_last_valid_exact_weights(
    padding_direction: TemporalPaddingDirection,
) -> None:
    history = _history(
        padding_direction=padding_direction
    )
    encoding = _identity_encoding(
        history
    )
    output = _last_pooler()(
        encoding
    )

    last_indices = _last_indices(
        history.timestep_mask
    )
    expected = torch.zeros(
        (
            N,
            1,
            T,
        ),
        dtype=history.dtype,
    )

    for row in range(
        N
    ):
        expected[
            row,
            0,
            int(
                last_indices[
                    row
                ].item()
            ),
        ] = 1.0

    assert torch.equal(
        output.pooling_weights,
        expected,
    )


@pytest.mark.parametrize(
    "padding_direction",
    (
        TemporalPaddingDirection.RIGHT,
        TemporalPaddingDirection.LEFT,
    ),
)
def test_last_valid_exact_values(
    padding_direction: TemporalPaddingDirection,
) -> None:
    history = _history(
        padding_direction=padding_direction
    )
    encoding = _identity_encoding(
        history
    )
    output = _last_pooler()(
        encoding
    )

    last_indices = _last_indices(
        history.timestep_mask
    )
    expected = torch.stack(
        [
            history.history[
                row,
                int(
                    last_indices[
                        row
                    ].item()
                ),
            ]
            for row in range(
                N
            )
        ]
    )

    assert torch.equal(
        output.pooled_memory,
        expected,
    )


def test_last_valid_is_order_sensitive() -> None:
    sequences = _logical_sequences()
    reversed_sequences = [
        sequence.flip(
            dims=(0,)
        )
        for sequence in sequences
    ]

    original = _last_pooler()(
        _identity_encoding(
            _history(
                sequences=sequences
            )
        )
    )
    reversed_output = _last_pooler()(
        _identity_encoding(
            _history(
                sequences=reversed_sequences
            )
        )
    )

    assert not torch.equal(
        original.pooled_memory,
        reversed_output.pooled_memory,
    )


def test_last_valid_gradient_reaches_only_selected_timesteps() -> None:
    history = _history(
        requires_grad=True
    )
    output = _last_pooler()(
        _identity_encoding(
            history
        )
    )

    output.pooled_memory.sum().backward()

    assert history.history.grad is not None

    expected = (
        output
        .pooling_weights
        .squeeze(1)
        .unsqueeze(-1)
        .expand_as(
            history.history
        )
    )

    assert torch.equal(
        history.history.grad,
        expected,
    )


def test_last_valid_selects_real_all_missing_latest_slot() -> None:
    history = _history(
        latest_all_features_missing=True
    )
    output = _last_pooler()(
        _identity_encoding(
            history
        )
    )

    last_indices = _last_indices(
        history.timestep_mask
    )

    for row in range(
        N
    ):
        index = int(
            last_indices[
                row
            ].item()
        )
        assert not bool(
            history
            .feature_observed_mask[
                row,
                index,
            ]
            .any()
            .item()
        )
        assert torch.equal(
            output.pooled_memory[
                row
            ],
            history.history[
                row,
                index,
            ],
        )


def test_last_valid_records_all_missing_selection_statistics() -> None:
    history = _history(
        latest_all_features_missing=True
    )
    output = _last_pooler()(
        _identity_encoding(
            history
        )
    )
    metadata = (
        output
        .computation_provenance
        .lineage
        .lineage_metadata
    )

    assert metadata[
        "selected_all_features_missing_count"
    ] == N
    assert metadata[
        "nonempty_node_count"
    ] == N
    assert metadata[
        "selected_all_features_missing_fraction"
    ] == 1.0


def test_last_valid_without_feature_mask_records_unavailable_as_zero() -> None:
    history = _history(
        include_feature_observed_mask=False
    )
    output = _last_pooler()(
        _identity_encoding(
            history
        )
    )
    metadata = (
        output
        .computation_provenance
        .lineage
        .lineage_metadata
    )

    assert metadata[
        "selected_all_features_missing_count"
    ] == 0
    assert metadata[
        "nonempty_node_count"
    ] == N
    assert metadata[
        "selected_all_features_missing_fraction"
    ] == 0.0


def test_last_valid_architecture_metadata_is_explicit() -> None:
    metadata = _last_pooler().architecture_metadata()

    assert metadata["pooling_kind"] == "last_valid"
    assert metadata["order_sensitive_pooling"] is True
    assert metadata["selection_uses_timestep_mask"] is True
    assert metadata[
        "feature_observed_mask_controls_selection"
    ] is False
    assert metadata["hazard_conditioned"] is False


def test_last_valid_extra_repr_is_informative() -> None:
    representation = _last_pooler().extra_repr()

    assert "order_sensitive=True" in representation
    assert "parameters=0" in representation
    assert "temporal_interaction=False" in representation


# =============================================================================
# Left/right padding equivalence
# =============================================================================


@pytest.mark.parametrize(
    "pooler_factory",
    (
        _mean_pooler,
        _last_pooler,
    ),
)
def test_left_and_right_padding_preserve_logical_pooled_values(
    pooler_factory: Callable[[], nn.Module],
) -> None:
    sequences = _logical_sequences()
    right = _identity_encoding(
        _history(
            sequences=sequences,
            padding_direction=TemporalPaddingDirection.RIGHT,
        )
    )
    left = _identity_encoding(
        _history(
            sequences=sequences,
            padding_direction=TemporalPaddingDirection.LEFT,
        )
    )

    right_output = pooler_factory()(
        right
    )
    left_output = pooler_factory()(
        left
    )

    assert torch.equal(
        right_output.pooled_memory,
        left_output.pooled_memory,
    )


def test_left_and_right_last_valid_weights_use_actual_storage_indices() -> None:
    sequences = _logical_sequences()
    right_history = _history(
        sequences=sequences,
        padding_direction=TemporalPaddingDirection.RIGHT,
    )
    left_history = _history(
        sequences=sequences,
        padding_direction=TemporalPaddingDirection.LEFT,
    )

    right = _last_pooler()(
        _identity_encoding(
            right_history
        )
    )
    left = _last_pooler()(
        _identity_encoding(
            left_history
        )
    )

    assert not torch.equal(
        right.pooling_weights,
        left.pooling_weights,
    )
    assert torch.equal(
        right.pooled_memory,
        left.pooled_memory,
    )


# =============================================================================
# Zero-history behavior
# =============================================================================


@pytest.mark.parametrize(
    "pooler_factory",
    (
        _mean_pooler,
        _last_pooler,
    ),
)
def test_error_policy_rejects_zero_history(
    pooler_factory: Callable[..., nn.Module],
) -> None:
    encoding = _identity_encoding(
        _history(
            include_zero_history=True
        )
    )
    pooler = pooler_factory(
        zero_history_policy=(
            TemporalPoolingZeroHistoryPolicy.ERROR
        )
    )

    with pytest.raises(ValueError):
        pooler(
            encoding
        )


@pytest.mark.parametrize(
    "pooler_factory",
    (
        _mean_pooler,
        _last_pooler,
    ),
)
def test_zero_policy_emits_exact_zero_row(
    pooler_factory: Callable[..., nn.Module],
) -> None:
    encoding = _identity_encoding(
        _history(
            include_zero_history=True
        )
    )
    output = pooler_factory(
        zero_history_policy=(
            TemporalPoolingZeroHistoryPolicy.ZERO
        )
    )(
        encoding
    )

    assert output.has_zero_history
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


@pytest.mark.parametrize(
    "pooler_factory",
    (
        _mean_pooler,
        _last_pooler,
    ),
)
def test_zero_policy_preserves_unit_mass_for_nonempty_rows(
    pooler_factory: Callable[..., nn.Module],
) -> None:
    encoding = _identity_encoding(
        _history(
            include_zero_history=True
        )
    )
    output = pooler_factory(
        zero_history_policy=(
            TemporalPoolingZeroHistoryPolicy.ZERO
        )
    )(
        encoding
    )

    sums = output.pooling_weights.sum(
        dim=-1
    ).squeeze(
        1
    )

    assert torch.equal(
        sums,
        torch.tensor(
            [1.0, 0.0, 1.0],
            dtype=encoding.dtype,
        ),
    )


@pytest.mark.parametrize(
    "pooler_factory",
    (
        _mean_pooler,
        _last_pooler,
    ),
)
def test_zero_history_rows_have_zero_input_gradient(
    pooler_factory: Callable[..., nn.Module],
) -> None:
    history = _history(
        include_zero_history=True,
        requires_grad=True,
    )
    output = pooler_factory(
        zero_history_policy=(
            TemporalPoolingZeroHistoryPolicy.ZERO
        )
    )(
        _identity_encoding(
            history
        )
    )

    output.pooled_memory.sum().backward()

    assert history.history.grad is not None
    assert torch.equal(
        history.history.grad[1],
        torch.zeros_like(
            history.history.grad[1]
        ),
    )


def test_last_valid_zero_history_audit_excludes_empty_nodes() -> None:
    history = _history(
        include_zero_history=True,
        latest_all_features_missing=True,
    )
    output = _last_pooler(
        zero_history_policy=(
            TemporalPoolingZeroHistoryPolicy.ZERO
        )
    )(
        _identity_encoding(
            history
        )
    )
    metadata = (
        output
        .computation_provenance
        .lineage
        .lineage_metadata
    )

    assert metadata[
        "nonempty_node_count"
    ] == 2
    assert metadata[
        "selected_all_features_missing_count"
    ] == 2
    assert metadata[
        "selected_all_features_missing_fraction"
    ] == 1.0


# =============================================================================
# Source mutation and numerical protection
# =============================================================================


@pytest.mark.parametrize(
    "pooler_factory",
    (
        _mean_pooler,
        _last_pooler,
    ),
)
@pytest.mark.parametrize(
    "nonfinite",
    (
        float("nan"),
        float("inf"),
        float("-inf"),
    ),
)
def test_mutated_nonfinite_encoding_is_rejected(
    pooler_factory: Callable[[], nn.Module],
    nonfinite: float,
) -> None:
    encoding = _identity_encoding(
        _history()
    )

    with torch.no_grad():
        encoding.encoded_sequence[0, 0, 0] = nonfinite

    with pytest.raises(ValueError):
        pooler_factory()(
            encoding
        )


@pytest.mark.parametrize(
    "pooler_factory",
    (
        _mean_pooler,
        _last_pooler,
    ),
)
def test_mutated_nonzero_encoded_padding_is_rejected(
    pooler_factory: Callable[[], nn.Module],
) -> None:
    encoding = _identity_encoding(
        _history()
    )
    padded = ~encoding.timestep_mask

    with torch.no_grad():
        encoding.encoded_sequence[
            padded
        ][0, 0] = 5.0

    # Advanced indexing above returns a copy, so mutate one concrete location.
    row, timestep = torch.nonzero(
        padded,
        as_tuple=False,
    )[0].tolist()

    with torch.no_grad():
        encoding.encoded_sequence[
            row,
            timestep,
            0,
        ] = 5.0

    with pytest.raises(ValueError):
        pooler_factory()(
            encoding
        )


# =============================================================================
# Provenance and parameter snapshots
# =============================================================================


def test_pooling_families_have_distinct_architecture_identity() -> None:
    encoding = _identity_encoding(
        _history()
    )
    mean = _mean_pooler()(
        encoding
    )
    last = _last_pooler()(
        encoding
    )

    assert (
        mean.architecture_fingerprint
        != last.architecture_fingerprint
    )


@pytest.mark.parametrize(
    "pooler_factory",
    (
        _mean_pooler,
        _last_pooler,
    ),
)
def test_source_lineage_is_exact(
    pooler_factory: Callable[[], nn.Module],
) -> None:
    encoding = _identity_encoding(
        _history()
    )
    output = pooler_factory()(
        encoding
    )

    assert (
        output
        .computation_provenance
        .lineage
        .source_lineage_fingerprints
        == (
            encoding.lineage_fingerprint(),
        )
    )


@pytest.mark.parametrize(
    "pooler_factory",
    (
        _mean_pooler,
        _last_pooler,
    ),
)
def test_axis_fingerprints_match_source(
    pooler_factory: Callable[[], nn.Module],
) -> None:
    encoding = _identity_encoding(
        _history()
    )
    output = pooler_factory()(
        encoding
    )
    lineage = (
        output
        .computation_provenance
        .lineage
    )

    assert lineage.node_axis_fingerprint == (
        encoding.node_axis.fingerprint()
    )
    assert lineage.temporal_axis_fingerprint == (
        encoding.temporal_alignment_fingerprint()
    )
    assert lineage.feature_axis_fingerprint == (
        encoding.feature_axis.fingerprint()
    )


@pytest.mark.parametrize(
    "pooler_factory",
    (
        _mean_pooler,
        _last_pooler,
    ),
)
def test_parameter_snapshot_is_not_generated_implicitly(
    pooler_factory: Callable[[], nn.Module],
) -> None:
    output = pooler_factory()(
        _identity_encoding(
            _history()
        )
    )

    assert (
        output
        .computation_provenance
        .parameter_snapshot
        is None
    )
    assert output.parameter_snapshot_fingerprint is None


@pytest.mark.parametrize(
    "pooler_factory",
    (
        _mean_pooler,
        _last_pooler,
    ),
)
def test_explicit_empty_parameter_snapshot_is_preserved(
    pooler_factory: Callable[[], nn.Module],
) -> None:
    pooler = pooler_factory()
    snapshot = build_parameter_snapshot_provenance(
        pooler,
        checkpoint_id="pooling-v1",
        training_step=7,
    )
    output = pooler(
        _identity_encoding(
            _history()
        ),
        parameter_snapshot=snapshot,
    )

    assert snapshot.parameter_count == 0
    assert snapshot.trainable_parameter_count == 0
    assert (
        output
        .computation_provenance
        .parameter_snapshot
        is snapshot
    )


@pytest.mark.parametrize(
    "pooler_factory",
    (
        _mean_pooler,
        _last_pooler,
    ),
)
def test_nonempty_parameter_snapshot_is_rejected(
    pooler_factory: Callable[[], nn.Module],
) -> None:
    linear = nn.Linear(
        2,
        3,
    )
    snapshot = build_parameter_snapshot_provenance(
        linear
    )

    with pytest.raises(ValueError):
        pooler_factory()(
            _identity_encoding(
                _history()
            ),
            parameter_snapshot=snapshot,
        )


@pytest.mark.parametrize(
    "pooler_factory",
    (
        _mean_pooler,
        _last_pooler,
    ),
)
def test_wrong_reported_zero_parameter_snapshot_count_is_rejected(
    pooler_factory: Callable[[], nn.Module],
) -> None:
    pooler = pooler_factory()
    correct = build_parameter_snapshot_provenance(
        pooler
    )
    wrong = MemoryParameterSnapshotProvenance(
        parameter_snapshot_fingerprint=(
            correct.parameter_snapshot_fingerprint
        ),
        parameter_count=1,
        trainable_parameter_count=1,
    )

    with pytest.raises(ValueError):
        pooler(
            _identity_encoding(
                _history()
            ),
            parameter_snapshot=wrong,
        )


@pytest.mark.parametrize(
    "pooler_factory",
    (
        _mean_pooler,
        _last_pooler,
    ),
)
def test_poolers_have_no_parameters_or_buffers(
    pooler_factory: Callable[[], nn.Module],
) -> None:
    pooler = pooler_factory()

    assert tuple(
        pooler.parameters()
    ) == ()
    assert tuple(
        pooler.buffers()
    ) == ()
    assert pooler.state_dict() == {}
    assert pooler.parameter_count == 0
    assert pooler.trainable_parameter_count == 0


# =============================================================================
# Generic sequence-encoder compatibility
# =============================================================================


@pytest.mark.parametrize(
    "encoding_factory,output_dim",
    (
        (
            _identity_encoding,
            D,
        ),
        (
            _linear_encoding,
            H,
        ),
        (
            _mlp_encoding,
            H,
        ),
    ),
)
@pytest.mark.parametrize(
    "pooler_class",
    (
        MaskedMeanTemporalPooler,
        LastValidTemporalPooler,
    ),
)
def test_poolers_accept_all_phase_five_sequence_encodings(
    encoding_factory: Callable[..., TemporalSequenceEncoding],
    output_dim: int,
    pooler_class: type[nn.Module],
) -> None:
    history = _history()
    encoding = encoding_factory(
        history
    )
    config = TemporalPoolingConfig(
        kind=(
            TemporalPoolingKind.MASKED_MEAN
            if pooler_class
            is MaskedMeanTemporalPooler
            else TemporalPoolingKind.LAST_VALID
        ),
        output_dim=output_dim,
    )
    output = pooler_class(
        config
    )(
        encoding
    )

    assert output.source_encoding is encoding
    assert output.pooled_shape == (
        N,
        output_dim,
    )


# =============================================================================
# Coordinates, missingness, and node equivariance
# =============================================================================


@pytest.mark.parametrize(
    "pooler_factory",
    (
        _mean_pooler,
        _last_pooler,
    ),
)
def test_temporal_coordinate_gaps_do_not_change_pooling_values(
    pooler_factory: Callable[[], nn.Module],
) -> None:
    regular = _identity_encoding(
        _history(
            gap_scale=1.0
        )
    )
    sparse = _identity_encoding(
        _history(
            gap_scale=10.0
        )
    )

    regular_output = pooler_factory()(
        regular
    )
    sparse_output = pooler_factory()(
        sparse
    )

    assert torch.equal(
        regular_output.pooling_weights,
        sparse_output.pooling_weights,
    )
    assert torch.equal(
        regular_output.pooled_memory,
        sparse_output.pooled_memory,
    )
    assert (
        regular_output.lineage_fingerprint()
        != sparse_output.lineage_fingerprint()
    )


def test_masked_mean_feature_observation_mask_does_not_change_weights() -> None:
    with_mask = _identity_encoding(
        _history(
            latest_all_features_missing=True
        )
    )
    without_mask = _identity_encoding(
        _history(
            include_feature_observed_mask=False
        )
    )

    first = _mean_pooler()(
        with_mask
    )
    second = _mean_pooler()(
        without_mask
    )

    assert torch.equal(
        first.pooling_weights,
        second.pooling_weights,
    )
    assert torch.equal(
        first.pooled_memory,
        second.pooled_memory,
    )


@pytest.mark.parametrize(
    "pooler_factory",
    (
        _mean_pooler,
        _last_pooler,
    ),
)
def test_node_permutation_equivariance(
    pooler_factory: Callable[[], nn.Module],
) -> None:
    history = _history()
    permutation = torch.tensor(
        [2, 0, 1],
        dtype=torch.long,
    )
    permuted = _permute_history(
        history,
        permutation,
    )

    original = pooler_factory()(
        _identity_encoding(
            history
        )
    )
    changed = pooler_factory()(
        _identity_encoding(
            permuted
        )
    )

    assert torch.equal(
        changed.pooled_memory,
        original
        .pooled_memory
        .index_select(
            0,
            permutation,
        ),
    )
    assert torch.equal(
        changed.pooling_weights,
        original
        .pooling_weights
        .index_select(
            0,
            permutation,
        ),
    )


# =============================================================================
# Dtype, sequence length, and device boundaries
# =============================================================================


@pytest.mark.parametrize(
    "pooler_factory",
    (
        _mean_pooler,
        _last_pooler,
    ),
)
def test_float64_execution_is_supported(
    pooler_factory: Callable[[], nn.Module],
) -> None:
    encoding = _identity_encoding(
        _history(
            dtype=torch.float64
        )
    )
    output = pooler_factory()(
        encoding
    )

    assert output.dtype == torch.float64
    assert output.pooling_weights.dtype == torch.float64


@pytest.mark.parametrize(
    "pooler_factory",
    (
        _mean_pooler,
        _last_pooler,
    ),
)
@pytest.mark.parametrize(
    "sequence_length",
    (
        3,
        6,
    ),
)
def test_variable_sequence_lengths_are_supported(
    pooler_factory: Callable[[], nn.Module],
    sequence_length: int,
) -> None:
    encoding = _identity_encoding(
        _history(
            sequence_length=sequence_length
        )
    )
    output = pooler_factory()(
        encoding
    )

    assert output.sequence_length == sequence_length
    assert output.weight_shape == (
        N,
        1,
        sequence_length,
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is not available.",
)
@pytest.mark.parametrize(
    "pooler_factory",
    (
        _mean_pooler,
        _last_pooler,
    ),
)
def test_poolers_execute_on_cuda(
    pooler_factory: Callable[[], nn.Module],
) -> None:
    encoding = _identity_encoding(
        _history(
            device="cuda"
        )
    )
    output = pooler_factory()(
        encoding
    )

    assert output.device.type == "cuda"
    assert output.source_encoding is encoding
