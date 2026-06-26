"""
Consolidated tests for Phase 5 sequence-preserving temporal baselines.

Target repository path:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            tests/
                memory/
                    baseline_encoders/
                        test_sequence_baselines.py

Implementations under test:
    memory/baseline_encoders/_provenance.py
    memory/baseline_encoders/identity_sequence_encoder.py
    memory/baseline_encoders/linear_projection_sequence_encoder.py
    memory/baseline_encoders/pointwise_mlp_sequence_encoder.py

The suite freezes the scientific distinction among:

``IdentitySequenceEncoder``
    Parameter-free raw sequence lower bound ``[N,T,D] -> [N,T,D]``.

``LinearProjectionSequenceEncoder``
    Width-matched affine feature transformation with no interaction across
    time ``[N,T,D] -> [N,T,H]``.

``PointwiseMLPSequenceEncoder``
    Nonlinear feature-capacity control with no interaction across time
    ``[N,T,D] -> [N,T,H]``.

All trainable encoders must validate complete pre-mask finiteness and restore
exact zero padding using the Phase 4 timestep mask. Missingness masks and
temporal coordinates remain preserved in the exact source-history object but
are not consumed as model features.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import MappingProxyType
from typing import Callable

import pytest
import torch
from torch import nn

from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.memory.baseline_encoders._provenance import (
    BASELINE_PROVENANCE_IMPLEMENTATION_VERSION,
    BASELINE_PROVENANCE_SCOPE,
    baseline_architecture_fingerprint,
    baseline_configuration_fingerprint,
    build_parameter_snapshot_provenance,
    count_module_parameters,
    module_parameter_snapshot_fingerprint,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.memory.baseline_encoders.identity_sequence_encoder import (
    IDENTITY_SEQUENCE_ENCODER_COMPONENT_KIND,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.memory.baseline_encoders.identity_sequence_encoder import (
    IDENTITY_SEQUENCE_ENCODER_COMPONENT_NAME,
    IDENTITY_SEQUENCE_ENCODER_FEATURE_OBSERVATION_POLICY,
    IDENTITY_SEQUENCE_ENCODER_IMPLEMENTATION_VERSION,
    IDENTITY_SEQUENCE_ENCODER_OPERATION_NAME,
    IDENTITY_SEQUENCE_ENCODER_PADDING_POLICY,
    IDENTITY_SEQUENCE_ENCODER_TEMPORAL_COORDINATE_POLICY,
    IDENTITY_SEQUENCE_ENCODER_TEMPORAL_INTERACTION,
    IdentitySequenceEncoder,
    IdentityTemporalSequenceEncoder,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.memory.baseline_encoders.linear_projection_sequence_encoder import (
    LINEAR_PROJECTION_SEQUENCE_ENCODER_ACTIVATION_POLICY,
    LINEAR_PROJECTION_SEQUENCE_ENCODER_COMPONENT_KIND,
    LINEAR_PROJECTION_SEQUENCE_ENCODER_COMPONENT_NAME,
    LINEAR_PROJECTION_SEQUENCE_ENCODER_FEATURE_OBSERVATION_POLICY,
    LINEAR_PROJECTION_SEQUENCE_ENCODER_IMPLEMENTATION_VERSION,
    LINEAR_PROJECTION_SEQUENCE_ENCODER_OPERATION_NAME,
    LINEAR_PROJECTION_SEQUENCE_ENCODER_PADDING_POLICY,
    LINEAR_PROJECTION_SEQUENCE_ENCODER_TEMPORAL_COORDINATE_POLICY,
    LINEAR_PROJECTION_SEQUENCE_ENCODER_TEMPORAL_INTERACTION,
    LinearProjectionSequenceEncoder,
    PerTimestepLinearSequenceEncoder,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.memory.baseline_encoders.pointwise_mlp_sequence_encoder import (
    POINTWISE_MLP_SEQUENCE_ENCODER_COMPONENT_KIND,
    POINTWISE_MLP_SEQUENCE_ENCODER_COMPONENT_NAME,
    POINTWISE_MLP_SEQUENCE_ENCODER_FEATURE_OBSERVATION_POLICY,
    POINTWISE_MLP_SEQUENCE_ENCODER_IMPLEMENTATION_VERSION,
    POINTWISE_MLP_SEQUENCE_ENCODER_LAYER_POLICY,
    POINTWISE_MLP_SEQUENCE_ENCODER_OPERATION_NAME,
    POINTWISE_MLP_SEQUENCE_ENCODER_PADDING_POLICY,
    POINTWISE_MLP_SEQUENCE_ENCODER_TEMPORAL_COORDINATE_POLICY,
    POINTWISE_MLP_SEQUENCE_ENCODER_TEMPORAL_INTERACTION,
    PerTimestepMLPSequenceEncoder,
    PointwiseMLPSequenceEncoder,
    TemporalMLPSequenceEncoder,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.memory.config import (
    BaselineSequenceEncoderConfig,
    BaselineSequenceEncoderKind,
    MemoryActivation,
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
    TemporalSequenceEncoderKind,
    TemporalSequenceEncoding,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.memory.schemas.temporal_coordinates import (
    RelativeTemporalCoordinates,
    TemporalPaddingDirection,
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
            "Test sequence_length must be at least three."
        )

    rows = [
        [True] * (sequence_length - 1) + [False],
        [True] * max(sequence_length - 2, 1)
        + [False] * min(2, sequence_length - 1),
        [True] * sequence_length,
    ]

    mask = torch.tensor(
        rows,
        dtype=torch.bool,
    )

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

    for row_index, length in enumerate(
        right.sum(dim=-1).tolist()
    ):
        if length > 0:
            left[
                row_index,
                sequence_length - length :,
            ] = True

    return left


def _coordinates_for_mask(
    mask: torch.Tensor,
    *,
    dtype: torch.dtype,
    device: torch.device | str = "cpu",
    extra_staleness: float = 0.0,
) -> torch.Tensor:
    values = torch.zeros(
        mask.shape,
        dtype=dtype,
        device=device,
    )

    for row_index in range(
        int(mask.shape[0])
    ):
        valid_indices = torch.nonzero(
            mask[row_index],
            as_tuple=False,
        ).flatten()
        length = int(
            valid_indices.numel()
        )

        if length == 0:
            continue

        offsets = torch.arange(
            -length,
            0,
            dtype=dtype,
            device=device,
        )
        offsets = offsets - float(
            extra_staleness
        )
        values[
            row_index,
            valid_indices,
        ] = offsets

    return values


def _history_values_for_mask(
    mask: torch.Tensor,
    *,
    feature_dim: int = D,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
    requires_grad: bool = False,
) -> torch.Tensor:
    values = (
        torch.arange(
            int(mask.shape[0])
            * int(mask.shape[1])
            * feature_dim,
            dtype=dtype,
            device=device,
        )
        .reshape(
            int(mask.shape[0]),
            int(mask.shape[1]),
            feature_dim,
        )
        / 10.0
        + 0.1
    )
    values = (
        values
        * mask
        .to(device=device)
        .unsqueeze(-1)
    )

    if requires_grad:
        values.requires_grad_()

    return values


def _feature_observed_mask(
    mask: torch.Tensor,
    *,
    feature_dim: int = D,
    all_missing_valid_position: tuple[int, int] | None = (0, 1),
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    observed = (
        mask
        .to(device=device)
        .unsqueeze(-1)
        .expand(
            -1,
            -1,
            feature_dim,
        )
        .clone()
    )

    if all_missing_valid_position is not None:
        row, timestep = all_missing_valid_position
        if bool(
            mask[row, timestep].item()
        ):
            observed[
                row,
                timestep,
            ] = False

    return observed


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
    all_missing_valid_position: tuple[int, int] | None = (0, 1),
    extra_staleness: float = 0.0,
    history_values: torch.Tensor | None = None,
) -> HistoricalSequenceInputs:
    padding_direction = TemporalPaddingDirection(
        padding_direction
    )

    if padding_direction == TemporalPaddingDirection.RIGHT:
        mask = _right_mask(
            include_zero_history=include_zero_history,
            sequence_length=sequence_length,
        )
    elif padding_direction == TemporalPaddingDirection.LEFT:
        mask = _left_mask(
            include_zero_history=include_zero_history,
            sequence_length=sequence_length,
        )
    else:
        mask = torch.ones(
            (
                N,
                sequence_length,
            ),
            dtype=torch.bool,
        )

    mask = mask.to(
        device=device
    )

    if history_values is None:
        values = _history_values_for_mask(
            mask,
            feature_dim=feature_dim,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
    else:
        values = history_values

    observed = (
        _feature_observed_mask(
            mask,
            feature_dim=feature_dim,
            all_missing_valid_position=(
                all_missing_valid_position
            ),
            device=device,
        )
        if include_feature_observed_mask
        else None
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
            source_fingerprint="node-axis-source-v1",
        ),
        feature_axis=TemporalFeatureAxis(
            feature_names=tuple(
                f"feature-{index}"
                for index in range(
                    feature_dim
                )
            ),
            source_fingerprint="feature-axis-source-v1",
        ),
        temporal_coordinates=RelativeTemporalCoordinates(
            values=_coordinates_for_mask(
                mask,
                dtype=dtype,
                device=device,
                extra_staleness=extra_staleness,
            ),
            unit="months",
        ),
        source_provenance=MemorySourceProvenance(
            source_name="phase-five-panel",
            source_kind="historical-node-sequence",
            source_fingerprint="history-source-v1",
            preprocessing_fingerprint="preprocessing-v1",
            imputation_fingerprint="imputation-v1",
        ),
        feature_observed_mask=observed,
        padding_direction=padding_direction,
        zero_length_policy=(
            HistoryZeroLengthPolicy.ALLOW_ZERO_HISTORY
            if include_zero_history
            else HistoryZeroLengthPolicy.ERROR
        ),
    )


def _identity_config(
    *,
    feature_dim: int = D,
    layer_normalization: bool = False,
    use_bias: bool = True,
) -> BaselineSequenceEncoderConfig:
    return BaselineSequenceEncoderConfig(
        kind=BaselineSequenceEncoderKind.IDENTITY,
        input_dim=feature_dim,
        output_dim=feature_dim,
        layer_normalization=layer_normalization,
        use_bias=use_bias,
    )


def _linear_config(
    *,
    input_dim: int = D,
    output_dim: int = H,
    dropout: float = 0.0,
    layer_normalization: bool = False,
    use_bias: bool = True,
    activation: MemoryActivation | str = MemoryActivation.GELU,
) -> BaselineSequenceEncoderConfig:
    return BaselineSequenceEncoderConfig(
        kind=BaselineSequenceEncoderKind.LINEAR_PROJECTION,
        input_dim=input_dim,
        output_dim=output_dim,
        activation=activation,
        dropout=dropout,
        layer_normalization=layer_normalization,
        use_bias=use_bias,
    )


def _mlp_config(
    *,
    input_dim: int = D,
    hidden_dim: int = 4,
    output_dim: int = H,
    num_hidden_layers: int = 2,
    activation: MemoryActivation | str = MemoryActivation.GELU,
    dropout: float = 0.0,
    layer_normalization: bool = False,
    use_bias: bool = True,
) -> BaselineSequenceEncoderConfig:
    return BaselineSequenceEncoderConfig(
        kind=BaselineSequenceEncoderKind.TEMPORAL_MLP,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_hidden_layers=num_hidden_layers,
        activation=activation,
        dropout=dropout,
        layer_normalization=layer_normalization,
        use_bias=use_bias,
    )


def _identity_encoder(
    *,
    feature_dim: int = D,
) -> IdentitySequenceEncoder:
    return IdentitySequenceEncoder(
        _identity_config(
            feature_dim=feature_dim
        )
    )


def _linear_encoder(
    **config_kwargs: object,
) -> LinearProjectionSequenceEncoder:
    return LinearProjectionSequenceEncoder(
        _linear_config(
            **config_kwargs,
        )
    )


def _mlp_encoder(
    **config_kwargs: object,
) -> PointwiseMLPSequenceEncoder:
    return PointwiseMLPSequenceEncoder(
        _mlp_config(
            **config_kwargs,
        )
    )


EncoderFactory = Callable[
    [],
    nn.Module,
]


@pytest.fixture(
    params=(
        "identity",
        "linear",
        "mlp",
    )
)
def encoder_and_history(
    request: pytest.FixtureRequest,
) -> tuple[
    nn.Module,
    HistoricalSequenceInputs,
]:
    history = _history(
        requires_grad=True
    )

    if request.param == "identity":
        return (
            _identity_encoder(),
            history,
        )

    if request.param == "linear":
        return (
            _linear_encoder(),
            history,
        )

    return (
        _mlp_encoder(),
        history,
    )


def _module_output_dim(
    module: nn.Module,
) -> int:
    return int(
        getattr(
            module,
            "output_dim",
        )
    )


def _permute_history(
    history: HistoricalSequenceInputs,
    permutation: torch.Tensor,
) -> HistoricalSequenceInputs:
    node_ids = tuple(
        history.node_ids[
            int(index)
        ]
        for index in permutation.tolist()
    )

    graph_ids = history.graph_ids

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
            node_ids=node_ids,
            node_batch_index=(
                history
                .node_batch_index
                .index_select(
                    0,
                    permutation,
                )
            ),
            graph_count=history.graph_count,
            graph_ids=graph_ids,
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


def _set_simple_linear_parameters(
    encoder: LinearProjectionSequenceEncoder,
) -> None:
    with torch.no_grad():
        values = torch.arange(
            encoder.output_dim
            * encoder.input_dim,
            dtype=encoder.projection.weight.dtype,
            device=encoder.projection.weight.device,
        ).reshape(
            encoder.output_dim,
            encoder.input_dim,
        )
        encoder.projection.weight.copy_(
            values / 10.0 + 0.1
        )

        if encoder.projection.bias is not None:
            encoder.projection.bias.copy_(
                torch.arange(
                    encoder.output_dim,
                    dtype=encoder.projection.weight.dtype,
                    device=encoder.projection.weight.device,
                )
                / 10.0
            )


def _set_simple_mlp_parameters(
    encoder: PointwiseMLPSequenceEncoder,
) -> None:
    with torch.no_grad():
        for layer_index, layer in enumerate(
            encoder.hidden_layers
        ):
            layer.weight.fill_(
                0.1
                * (
                    layer_index
                    + 1
                )
            )
            if layer.bias is not None:
                layer.bias.fill_(
                    0.05
                    * (
                        layer_index
                        + 1
                    )
                )

        encoder.output_projection.weight.fill_(
            0.2
        )
        if encoder.output_projection.bias is not None:
            encoder.output_projection.bias.fill_(
                0.1
            )


# =============================================================================
# Component identities, aliases, and private provenance
# =============================================================================


def test_sequence_encoder_aliases_preserve_exact_classes() -> None:
    assert IdentityTemporalSequenceEncoder is IdentitySequenceEncoder
    assert PerTimestepLinearSequenceEncoder is (
        LinearProjectionSequenceEncoder
    )
    assert PerTimestepMLPSequenceEncoder is (
        PointwiseMLPSequenceEncoder
    )
    assert TemporalMLPSequenceEncoder is (
        PointwiseMLPSequenceEncoder
    )


@pytest.mark.parametrize(
    "value",
    (
        IDENTITY_SEQUENCE_ENCODER_COMPONENT_NAME,
        IDENTITY_SEQUENCE_ENCODER_COMPONENT_KIND,
        IDENTITY_SEQUENCE_ENCODER_OPERATION_NAME,
        IDENTITY_SEQUENCE_ENCODER_IMPLEMENTATION_VERSION,
        IDENTITY_SEQUENCE_ENCODER_FEATURE_OBSERVATION_POLICY,
        IDENTITY_SEQUENCE_ENCODER_TEMPORAL_COORDINATE_POLICY,
        IDENTITY_SEQUENCE_ENCODER_PADDING_POLICY,
        LINEAR_PROJECTION_SEQUENCE_ENCODER_COMPONENT_NAME,
        LINEAR_PROJECTION_SEQUENCE_ENCODER_COMPONENT_KIND,
        LINEAR_PROJECTION_SEQUENCE_ENCODER_OPERATION_NAME,
        LINEAR_PROJECTION_SEQUENCE_ENCODER_IMPLEMENTATION_VERSION,
        LINEAR_PROJECTION_SEQUENCE_ENCODER_FEATURE_OBSERVATION_POLICY,
        LINEAR_PROJECTION_SEQUENCE_ENCODER_TEMPORAL_COORDINATE_POLICY,
        LINEAR_PROJECTION_SEQUENCE_ENCODER_PADDING_POLICY,
        LINEAR_PROJECTION_SEQUENCE_ENCODER_ACTIVATION_POLICY,
        POINTWISE_MLP_SEQUENCE_ENCODER_COMPONENT_NAME,
        POINTWISE_MLP_SEQUENCE_ENCODER_COMPONENT_KIND,
        POINTWISE_MLP_SEQUENCE_ENCODER_OPERATION_NAME,
        POINTWISE_MLP_SEQUENCE_ENCODER_IMPLEMENTATION_VERSION,
        POINTWISE_MLP_SEQUENCE_ENCODER_FEATURE_OBSERVATION_POLICY,
        POINTWISE_MLP_SEQUENCE_ENCODER_TEMPORAL_COORDINATE_POLICY,
        POINTWISE_MLP_SEQUENCE_ENCODER_PADDING_POLICY,
        POINTWISE_MLP_SEQUENCE_ENCODER_LAYER_POLICY,
        BASELINE_PROVENANCE_IMPLEMENTATION_VERSION,
        BASELINE_PROVENANCE_SCOPE,
    ),
)
def test_component_identity_strings_are_nonempty(
    value: str,
) -> None:
    assert isinstance(
        value,
        str,
    )
    assert value.strip()


def test_all_sequence_baselines_declare_no_temporal_interaction() -> None:
    assert IDENTITY_SEQUENCE_ENCODER_TEMPORAL_INTERACTION is False
    assert (
        LINEAR_PROJECTION_SEQUENCE_ENCODER_TEMPORAL_INTERACTION
        is False
    )
    assert POINTWISE_MLP_SEQUENCE_ENCODER_TEMPORAL_INTERACTION is False


def test_configuration_fingerprint_is_deterministic() -> None:
    first = _linear_config()
    second = _linear_config()

    assert baseline_configuration_fingerprint(
        first
    ) == baseline_configuration_fingerprint(
        second
    )
    assert baseline_configuration_fingerprint(
        first
    ) == first.config_hash()


def test_configuration_fingerprint_changes_with_semantics() -> None:
    first = _linear_config(
        use_bias=True
    )
    second = _linear_config(
        use_bias=False
    )

    assert baseline_configuration_fingerprint(
        first
    ) != baseline_configuration_fingerprint(
        second
    )


def test_architecture_fingerprint_is_order_independent() -> None:
    first = baseline_architecture_fingerprint(
        component_name="component",
        component_kind="kind",
        architecture_metadata={
            "input_dim": 2,
            "output_dim": 3,
        },
    )
    second = baseline_architecture_fingerprint(
        component_name="component",
        component_kind="kind",
        architecture_metadata={
            "output_dim": 3,
            "input_dim": 2,
        },
    )

    assert first == second


def test_architecture_fingerprint_rejects_nonfinite_metadata() -> None:
    with pytest.raises(ValueError):
        baseline_architecture_fingerprint(
            component_name="component",
            component_kind="kind",
            architecture_metadata={
                "bad": float("nan"),
            },
        )


def test_private_provenance_parameter_counting() -> None:
    module = nn.Sequential(
        nn.Linear(
            2,
            3,
        ),
        nn.Linear(
            3,
            1,
            bias=False,
        ),
    )
    total, trainable = count_module_parameters(
        module
    )

    assert total == 12
    assert trainable == 12

    module[0].weight.requires_grad_(
        False
    )
    total, trainable = count_module_parameters(
        module
    )

    assert total == 12
    assert trainable == 6


def test_parameter_snapshot_fingerprint_changes_after_mutation() -> None:
    module = nn.Linear(
        2,
        3,
    )
    before = module_parameter_snapshot_fingerprint(
        module
    )

    with torch.no_grad():
        module.weight.add_(
            1.0
        )

    after = module_parameter_snapshot_fingerprint(
        module
    )
    assert before != after


def test_parameter_snapshot_fingerprint_is_device_independent_for_cpu_copy() -> None:
    module = nn.Linear(
        2,
        3,
    )
    copied = nn.Linear(
        2,
        3,
    )
    copied.load_state_dict(
        module.state_dict()
    )

    assert module_parameter_snapshot_fingerprint(
        module
    ) == module_parameter_snapshot_fingerprint(
        copied
    )


# =============================================================================
# Configuration and construction boundaries
# =============================================================================


def test_identity_constructor_accepts_valid_config() -> None:
    encoder = IdentitySequenceEncoder(
        _identity_config()
    )

    assert encoder.input_dim == D
    assert encoder.output_dim == D


def test_identity_constructor_rejects_nonidentity_config() -> None:
    with pytest.raises(ValueError):
        IdentitySequenceEncoder(
            _linear_config(
                output_dim=D
            )
        )


def test_identity_constructor_rejects_layer_normalization() -> None:
    with pytest.raises(ValueError):
        IdentitySequenceEncoder(
            _identity_config(
                layer_normalization=True
            )
        )


def test_linear_constructor_accepts_valid_config() -> None:
    encoder = LinearProjectionSequenceEncoder(
        _linear_config()
    )

    assert encoder.input_dim == D
    assert encoder.output_dim == H


def test_linear_constructor_rejects_wrong_kind() -> None:
    with pytest.raises(ValueError):
        LinearProjectionSequenceEncoder(
            _identity_config()
        )


def test_mlp_constructor_accepts_valid_config() -> None:
    encoder = PointwiseMLPSequenceEncoder(
        _mlp_config()
    )

    assert encoder.input_dim == D
    assert encoder.hidden_dim == 4
    assert encoder.output_dim == H
    assert encoder.num_hidden_layers == 2


def test_mlp_constructor_rejects_wrong_kind() -> None:
    with pytest.raises(ValueError):
        PointwiseMLPSequenceEncoder(
            _linear_config()
        )


@pytest.mark.parametrize(
    (
        "activation",
        "expected_type",
    ),
    (
        (
            MemoryActivation.RELU,
            nn.ReLU,
        ),
        (
            MemoryActivation.GELU,
            nn.GELU,
        ),
        (
            MemoryActivation.SILU,
            nn.SiLU,
        ),
        (
            MemoryActivation.TANH,
            nn.Tanh,
        ),
    ),
)
def test_mlp_constructs_configured_activation(
    activation: MemoryActivation,
    expected_type: type[nn.Module],
) -> None:
    encoder = _mlp_encoder(
        activation=activation,
        num_hidden_layers=3,
    )

    assert len(
        encoder.hidden_activations
    ) == 3
    assert all(
        isinstance(
            module,
            expected_type,
        )
        for module in encoder.hidden_activations
    )


def test_linear_configured_activation_is_not_applied() -> None:
    encoder = _linear_encoder(
        activation=MemoryActivation.TANH
    )

    assert not any(
        isinstance(
            module,
            (
                nn.ReLU,
                nn.GELU,
                nn.SiLU,
                nn.Tanh,
            ),
        )
        for module in encoder.modules()
        if module is not encoder
    )
    assert (
        encoder
        .architecture_metadata()[
            "configured_activation_not_applied"
        ]
        == "tanh"
    )


# =============================================================================
# Shared output contracts
# =============================================================================


def test_all_encoders_return_canonical_sequence_contract(
    encoder_and_history: tuple[
        nn.Module,
        HistoricalSequenceInputs,
    ],
) -> None:
    encoder, history = encoder_and_history
    output = encoder(
        history
    )

    assert isinstance(
        output,
        TemporalSequenceEncoding,
    )
    assert output.source_history is history
    assert output.node_count == N
    assert output.sequence_length == T
    assert output.hidden_dim == _module_output_dim(
        encoder
    )


def test_all_encoders_preserve_exact_source_metadata(
    encoder_and_history: tuple[
        nn.Module,
        HistoricalSequenceInputs,
    ],
) -> None:
    encoder, history = encoder_and_history
    output = encoder(
        history
    )

    assert output.timestep_mask is history.timestep_mask
    assert output.node_axis is history.node_axis
    assert output.feature_axis is history.feature_axis
    assert output.temporal_coordinates is history.temporal_coordinates
    assert output.source_provenance is history.source_provenance
    assert output.node_batch_index is history.node_batch_index
    assert output.node_ids == history.node_ids
    assert output.graph_ids == history.graph_ids


def test_all_outputs_have_no_generic_terminal_state(
    encoder_and_history: tuple[
        nn.Module,
        HistoricalSequenceInputs,
    ],
) -> None:
    encoder, history = encoder_and_history
    output = encoder(
        history
    )

    assert not hasattr(
        output,
        "final_state",
    )
    assert not hasattr(
        output,
        "hidden_state",
    )
    assert not hasattr(
        output,
        "cell_state",
    )


def test_all_outputs_preserve_source_lineage(
    encoder_and_history: tuple[
        nn.Module,
        HistoricalSequenceInputs,
    ],
) -> None:
    encoder, history = encoder_and_history
    output = encoder(
        history
    )

    assert (
        output
        .computation_provenance
        .lineage
        .source_lineage_fingerprints
        == (
            history.lineage_fingerprint(),
        )
    )


def test_all_outputs_have_axis_aligned_lineage(
    encoder_and_history: tuple[
        nn.Module,
        HistoricalSequenceInputs,
    ],
) -> None:
    encoder, history = encoder_and_history
    output = encoder(
        history
    )
    lineage = (
        output
        .computation_provenance
        .lineage
    )

    assert lineage.node_axis_fingerprint == (
        history.node_axis.fingerprint()
    )
    assert lineage.temporal_axis_fingerprint == (
        history.temporal_alignment_fingerprint()
    )
    assert lineage.feature_axis_fingerprint == (
        history.feature_axis.fingerprint()
    )


def test_all_outputs_have_immutable_architecture_metadata(
    encoder_and_history: tuple[
        nn.Module,
        HistoricalSequenceInputs,
    ],
) -> None:
    encoder, history = encoder_and_history
    output = encoder(
        history
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
            "new"
        ] = True  # type: ignore[index]


def test_output_fingerprints_are_deterministic_for_repeated_eval_forward(
    encoder_and_history: tuple[
        nn.Module,
        HistoricalSequenceInputs,
    ],
) -> None:
    encoder, history = encoder_and_history
    encoder.eval()

    first = encoder(
        history
    )
    second = encoder(
        history
    )

    assert first.semantic_fingerprint() == second.semantic_fingerprint()
    assert first.value_fingerprint() == second.value_fingerprint()
    assert first.lineage_fingerprint() == second.lineage_fingerprint()


def test_output_is_frozen(
    encoder_and_history: tuple[
        nn.Module,
        HistoricalSequenceInputs,
    ],
) -> None:
    encoder, history = encoder_and_history
    output = encoder(
        history
    )

    with pytest.raises(
        (
            FrozenInstanceError,
            AttributeError,
        )
    ):
        output.encoding_name = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    "encoder_factory",
    (
        _identity_encoder,
        _linear_encoder,
        _mlp_encoder,
    ),
)
def test_wrong_source_type_is_rejected(
    encoder_factory: Callable[[], nn.Module],
) -> None:
    encoder = encoder_factory()

    with pytest.raises(TypeError):
        encoder(
            object()  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "encoder_factory",
    (
        _identity_encoder,
        _linear_encoder,
        _mlp_encoder,
    ),
)
def test_feature_dimension_mismatch_is_rejected(
    encoder_factory: Callable[[], nn.Module],
) -> None:
    encoder = encoder_factory()
    wrong = _history(
        feature_dim=3
    )

    with pytest.raises(ValueError):
        encoder(
            wrong
        )


# =============================================================================
# Identity baseline
# =============================================================================


def test_identity_returns_exact_source_tensor() -> None:
    history = _history(
        requires_grad=True
    )
    encoder = _identity_encoder()
    output = encoder(
        history
    )

    assert output.encoded_sequence is history.history
    assert output.encoder_kind == (
        TemporalSequenceEncoderKind.IDENTITY_SEQUENCE
    )
    assert output.encoding_name == (
        "identity_sequence_encoding"
    )


def test_identity_has_no_parameters_or_buffers() -> None:
    encoder = _identity_encoder()

    assert tuple(
        encoder.parameters()
    ) == ()
    assert tuple(
        encoder.buffers()
    ) == ()
    assert encoder.state_dict() == {}
    assert encoder.parameter_count == 0
    assert encoder.trainable_parameter_count == 0


def test_identity_autograd_alias_is_exact() -> None:
    history = _history(
        requires_grad=True
    )
    output = _identity_encoder()(
        history
    )

    output.encoded_sequence.sum().backward()

    assert history.history.grad is not None
    assert torch.equal(
        history.history.grad,
        torch.ones_like(
            history.history
        ),
    )


def test_identity_rejects_mutated_nonzero_padding() -> None:
    history = _history()
    padded = ~history.timestep_mask

    with torch.no_grad():
        history.history[
            padded
        ] = 7.0

    with pytest.raises(ValueError):
        _identity_encoder()(
            history
        )


@pytest.mark.parametrize(
    "nonfinite",
    (
        float("nan"),
        float("inf"),
        float("-inf"),
    ),
)
def test_identity_rejects_mutated_nonfinite_history(
    nonfinite: float,
) -> None:
    history = _history()

    with torch.no_grad():
        history.history[0, 0, 0] = nonfinite

    with pytest.raises(ValueError):
        _identity_encoder()(
            history
        )


def test_identity_architecture_metadata_is_explicit() -> None:
    encoder = _identity_encoder()
    metadata = encoder.architecture_metadata()

    assert metadata["trainable"] is False
    assert metadata["feature_projection"] is False
    assert metadata["feature_mixing"] is False
    assert metadata["temporal_interaction"] is False
    assert metadata["output_aliases_source_history_tensor"] is True


def test_identity_extra_repr_is_informative() -> None:
    representation = _identity_encoder().extra_repr()

    assert "input_dim=2" in representation
    assert "parameters=0" in representation
    assert "temporal_interaction=False" in representation


# =============================================================================
# Linear projection baseline
# =============================================================================


def test_linear_projection_matches_direct_module_application() -> None:
    history = _history()
    encoder = _linear_encoder()
    _set_simple_linear_parameters(
        encoder
    )

    output = encoder(
        history
    )
    expected = encoder.projection(
        history.history
    )
    expected = torch.where(
        history
        .timestep_mask
        .unsqueeze(-1),
        expected,
        torch.zeros_like(
            expected
        ),
    )

    assert torch.equal(
        output.encoded_sequence,
        expected,
    )


def test_linear_projection_bias_cannot_leak_into_padding() -> None:
    history = _history()
    encoder = _linear_encoder(
        use_bias=True
    )

    with torch.no_grad():
        encoder.projection.weight.zero_()
        assert encoder.projection.bias is not None
        encoder.projection.bias.fill_(
            5.0
        )

    output = encoder(
        history
    )
    padded = (
        ~history
        .timestep_mask
    ).unsqueeze(-1).expand_as(
        output.encoded_sequence
    )

    assert torch.equal(
        output.encoded_sequence[
            padded
        ],
        torch.zeros_like(
            output.encoded_sequence[
                padded
            ]
        ),
    )


def test_linear_projection_parameter_count_without_normalization() -> None:
    encoder = _linear_encoder(
        input_dim=2,
        output_dim=5,
        use_bias=True,
        layer_normalization=False,
    )

    assert encoder.parameter_count == 15
    assert encoder.trainable_parameter_count == 15


def test_linear_projection_parameter_count_with_normalization() -> None:
    encoder = _linear_encoder(
        input_dim=2,
        output_dim=5,
        use_bias=False,
        layer_normalization=True,
    )

    assert encoder.parameter_count == 20
    assert encoder.trainable_parameter_count == 20


def test_linear_projection_preserves_no_temporal_interaction() -> None:
    history = _history()
    encoder = _linear_encoder()
    encoder.eval()

    original = encoder(
        history
    )

    changed_values = history.history.clone()
    changed_values[0, 0] += 100.0
    changed = history.replace(
        history=changed_values
    )
    modified = encoder(
        changed
    )

    assert torch.equal(
        original.encoded_sequence[0, 1:],
        modified.encoded_sequence[0, 1:],
    )


def test_linear_projection_padding_input_content_cannot_affect_output() -> None:
    history = _history()
    encoder = _linear_encoder()
    encoder.eval()

    reference = encoder(
        history
    )
    padded = ~history.timestep_mask

    with torch.no_grad():
        history.history[
            padded
        ] = 123.0

    mutated = encoder(
        history
    )

    assert torch.equal(
        reference.encoded_sequence,
        mutated.encoded_sequence,
    )


def test_linear_projection_padded_outputs_induce_zero_input_gradient() -> None:
    history = _history(
        requires_grad=True
    )
    encoder = _linear_encoder()
    output = encoder(
        history
    )

    output.encoded_sequence.sum().backward()
    padded = ~history.timestep_mask

    assert history.history.grad is not None
    assert torch.equal(
        history.history.grad[
            padded
        ],
        torch.zeros_like(
            history.history.grad[
                padded
            ]
        ),
    )


def test_linear_projection_architecture_metadata_is_explicit() -> None:
    encoder = _linear_encoder(
        dropout=0.2,
        layer_normalization=True,
        use_bias=False,
        activation=MemoryActivation.TANH,
    )
    metadata = encoder.architecture_metadata()

    assert metadata["feature_projection"] is True
    assert metadata["feature_mixing"] is True
    assert metadata["pointwise_nonlinear_activation"] is False
    assert metadata["temporal_interaction"] is False
    assert metadata["dropout"] == pytest.approx(
        0.2
    )
    assert metadata["configured_activation_not_applied"] == "tanh"


def test_linear_projection_extra_repr_is_informative() -> None:
    representation = _linear_encoder(
        dropout=0.1,
        layer_normalization=True,
    ).extra_repr()

    assert "activation=none" in representation
    assert "layer_normalization=True" in representation
    assert "temporal_interaction=False" in representation


# =============================================================================
# Pointwise MLP baseline
# =============================================================================


def test_pointwise_mlp_matches_internal_pointwise_network() -> None:
    history = _history()
    encoder = _mlp_encoder(
        activation=MemoryActivation.RELU
    )
    _set_simple_mlp_parameters(
        encoder
    )

    output = encoder(
        history
    )
    expected = encoder._apply_pointwise_network(
        history.history
    )
    expected = torch.where(
        history
        .timestep_mask
        .unsqueeze(-1),
        expected,
        torch.zeros_like(
            expected
        ),
    )

    assert torch.equal(
        output.encoded_sequence,
        expected,
    )


def test_pointwise_mlp_has_expected_layer_shapes() -> None:
    encoder = _mlp_encoder(
        input_dim=2,
        hidden_dim=7,
        output_dim=3,
        num_hidden_layers=3,
    )

    assert len(
        encoder.hidden_layers
    ) == 3
    assert encoder.hidden_layers[0].in_features == 2
    assert encoder.hidden_layers[0].out_features == 7
    assert encoder.hidden_layers[1].in_features == 7
    assert encoder.hidden_layers[2].in_features == 7
    assert encoder.output_projection.in_features == 7
    assert encoder.output_projection.out_features == 3


def test_pointwise_mlp_parameter_count_without_normalization() -> None:
    encoder = _mlp_encoder(
        input_dim=2,
        hidden_dim=4,
        output_dim=3,
        num_hidden_layers=2,
        use_bias=True,
        layer_normalization=False,
    )

    assert encoder.parameter_count == 47
    assert encoder.trainable_parameter_count == 47


def test_pointwise_mlp_parameter_count_with_normalization() -> None:
    encoder = _mlp_encoder(
        input_dim=2,
        hidden_dim=4,
        output_dim=3,
        num_hidden_layers=2,
        use_bias=False,
        layer_normalization=True,
    )

    affine = (
        2 * 4
        + 4 * 4
        + 4 * 3
    )
    normalization = (
        2 * 4 * 2
        + 2 * 3
    )

    assert encoder.parameter_count == (
        affine
        + normalization
    )


def test_pointwise_mlp_final_projection_is_not_activated() -> None:
    history = _history()
    encoder = _mlp_encoder(
        hidden_dim=3,
        output_dim=1,
        num_hidden_layers=1,
        activation=MemoryActivation.RELU,
    )

    with torch.no_grad():
        encoder.hidden_layers[0].weight.fill_(
            1.0
        )
        assert encoder.hidden_layers[0].bias is not None
        encoder.hidden_layers[0].bias.zero_()
        encoder.output_projection.weight.fill_(
            -1.0
        )
        assert encoder.output_projection.bias is not None
        encoder.output_projection.bias.zero_()

    output = encoder(
        history
    )
    valid = history.timestep_mask

    assert bool(
        (
            output
            .encoded_sequence[
                valid
            ]
            < 0
        ).all().item()
    )


def test_pointwise_mlp_preserves_no_temporal_interaction() -> None:
    history = _history()
    encoder = _mlp_encoder()
    encoder.eval()

    original = encoder(
        history
    )

    changed_values = history.history.clone()
    changed_values[0, 0] += 100.0
    changed = history.replace(
        history=changed_values
    )
    modified = encoder(
        changed
    )

    assert torch.equal(
        original.encoded_sequence[0, 1:],
        modified.encoded_sequence[0, 1:],
    )


def test_pointwise_mlp_padding_input_content_cannot_affect_output() -> None:
    history = _history()
    encoder = _mlp_encoder()
    encoder.eval()

    reference = encoder(
        history
    )
    padded = ~history.timestep_mask

    with torch.no_grad():
        history.history[
            padded
        ] = 321.0

    mutated = encoder(
        history
    )

    assert torch.equal(
        reference.encoded_sequence,
        mutated.encoded_sequence,
    )


def test_pointwise_mlp_padded_outputs_induce_zero_input_gradient() -> None:
    history = _history(
        requires_grad=True
    )
    encoder = _mlp_encoder(
        activation=MemoryActivation.SILU
    )
    output = encoder(
        history
    )

    output.encoded_sequence.sum().backward()
    padded = ~history.timestep_mask

    assert history.history.grad is not None
    assert torch.equal(
        history.history.grad[
            padded
        ],
        torch.zeros_like(
            history.history.grad[
                padded
            ]
        ),
    )


def test_pointwise_mlp_architecture_metadata_is_explicit() -> None:
    encoder = _mlp_encoder(
        hidden_dim=8,
        num_hidden_layers=3,
        activation=MemoryActivation.SILU,
        dropout=0.2,
        layer_normalization=True,
    )
    metadata = encoder.architecture_metadata()

    assert metadata["hidden_dim"] == 8
    assert metadata["num_hidden_layers"] == 3
    assert metadata["activation"] == "silu"
    assert metadata["nonlinear_feature_transformation"] is True
    assert metadata["temporal_interaction"] is False
    assert metadata["flattened_temporal_window"] if "flattened_temporal_window" in metadata else True


def test_pointwise_mlp_extra_repr_is_informative() -> None:
    representation = _mlp_encoder(
        hidden_dim=8,
        num_hidden_layers=3,
        activation=MemoryActivation.SILU,
    ).extra_repr()

    assert "hidden_dim=8" in representation
    assert "num_hidden_layers=3" in representation
    assert "activation=silu" in representation
    assert "temporal_interaction=False" in representation


# =============================================================================
# Padding direction, zero history, missingness, and time metadata
# =============================================================================


@pytest.mark.parametrize(
    "encoder_factory",
    (
        _identity_encoder,
        _linear_encoder,
        _mlp_encoder,
    ),
)
@pytest.mark.parametrize(
    "padding_direction",
    (
        TemporalPaddingDirection.LEFT,
        TemporalPaddingDirection.RIGHT,
    ),
)
def test_encoders_support_left_and_right_padding(
    encoder_factory: Callable[[], nn.Module],
    padding_direction: TemporalPaddingDirection,
) -> None:
    history = _history(
        padding_direction=padding_direction
    )
    encoder = encoder_factory()
    encoder.eval()
    output = encoder(
        history
    )

    padded = (
        ~history
        .timestep_mask
    ).unsqueeze(-1).expand_as(
        output.encoded_sequence
    )

    assert torch.equal(
        output.encoded_sequence[
            padded
        ],
        torch.zeros_like(
            output.encoded_sequence[
                padded
            ]
        ),
    )


@pytest.mark.parametrize(
    "encoder_factory",
    (
        _identity_encoder,
        _linear_encoder,
        _mlp_encoder,
    ),
)
def test_left_and_right_padding_produce_same_valid_pointwise_values(
    encoder_factory: Callable[[], nn.Module],
) -> None:
    right = _history(
        padding_direction=TemporalPaddingDirection.RIGHT
    )
    left_mask = _left_mask()
    left_values = torch.zeros_like(
        right.history
    )

    for row, length in enumerate(
        right.valid_lengths.tolist()
    ):
        if length > 0:
            left_values[
                row,
                -length:,
            ] = right.history[
                row,
                :length,
            ]

    left = _history(
        padding_direction=TemporalPaddingDirection.LEFT,
        history_values=left_values,
    )

    encoder = encoder_factory()
    encoder.eval()
    right_output = encoder(
        right
    )
    left_output = encoder(
        left
    )

    for row, length in enumerate(
        right.valid_lengths.tolist()
    ):
        assert torch.equal(
            right_output.encoded_sequence[
                row,
                :length,
            ],
            left_output.encoded_sequence[
                row,
                -length:,
            ],
        )


@pytest.mark.parametrize(
    "encoder_factory",
    (
        _identity_encoder,
        _linear_encoder,
        _mlp_encoder,
    ),
)
def test_zero_history_rows_produce_exact_zero_sequences(
    encoder_factory: Callable[[], nn.Module],
) -> None:
    history = _history(
        include_zero_history=True
    )
    encoder = encoder_factory()
    encoder.eval()
    output = encoder(
        history
    )

    assert torch.equal(
        output.encoded_sequence[1],
        torch.zeros_like(
            output.encoded_sequence[1]
        ),
    )
    assert output.has_zero_history


@pytest.mark.parametrize(
    "encoder_factory",
    (
        _identity_encoder,
        _linear_encoder,
        _mlp_encoder,
    ),
)
def test_feature_observed_mask_is_preserved_but_not_consumed(
    encoder_factory: Callable[[], nn.Module],
) -> None:
    with_mask = _history(
        include_feature_observed_mask=True,
        all_missing_valid_position=(0, 1),
    )
    without_mask = _history(
        include_feature_observed_mask=False,
    )
    encoder = encoder_factory()
    encoder.eval()

    first = encoder(
        with_mask
    )
    second = encoder(
        without_mask
    )

    assert first.source_history.feature_observed_mask is (
        with_mask.feature_observed_mask
    )
    assert torch.equal(
        first.encoded_sequence,
        second.encoded_sequence,
    )
    assert first.lineage_fingerprint() != second.lineage_fingerprint()


@pytest.mark.parametrize(
    "encoder_factory",
    (
        _identity_encoder,
        _linear_encoder,
        _mlp_encoder,
    ),
)
def test_temporal_coordinate_values_are_preserved_but_not_consumed(
    encoder_factory: Callable[[], nn.Module],
) -> None:
    recent = _history(
        extra_staleness=0.0
    )
    stale = _history(
        extra_staleness=10.0
    )
    encoder = encoder_factory()
    encoder.eval()

    recent_output = encoder(
        recent
    )
    stale_output = encoder(
        stale
    )

    assert torch.equal(
        recent_output.encoded_sequence,
        stale_output.encoded_sequence,
    )
    assert (
        recent_output.temporal_alignment_fingerprint()
        != stale_output.temporal_alignment_fingerprint()
    )
    assert (
        recent_output.lineage_fingerprint()
        != stale_output.lineage_fingerprint()
    )


@pytest.mark.parametrize(
    "encoder_factory",
    (
        _identity_encoder,
        _linear_encoder,
        _mlp_encoder,
    ),
)
def test_real_all_feature_missing_timestep_remains_part_of_sequence(
    encoder_factory: Callable[[], nn.Module],
) -> None:
    history = _history(
        include_feature_observed_mask=True,
        all_missing_valid_position=(0, 1),
    )
    encoder = encoder_factory()
    encoder.eval()
    output = encoder(
        history
    )

    assert bool(
        history.timestep_mask[0, 1].item()
    )
    assert not bool(
        history
        .feature_observed_mask[
            0,
            1,
        ]
        .any()
        .item()
    )
    assert bool(
        torch.isfinite(
            output.encoded_sequence[
                0,
                1,
            ]
        ).all().item()
    )


# =============================================================================
# Numerical protection and dtype/device compatibility
# =============================================================================


@pytest.mark.parametrize(
    "encoder_factory",
    (
        _identity_encoder,
        _linear_encoder,
        _mlp_encoder,
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
def test_mutated_nonfinite_source_is_rejected(
    encoder_factory: Callable[[], nn.Module],
    nonfinite: float,
) -> None:
    history = _history()

    with torch.no_grad():
        history.history[0, 0, 0] = nonfinite

    with pytest.raises(ValueError):
        encoder_factory()(
            history
        )


def test_linear_rejects_nonfinite_pre_mask_parameter_output() -> None:
    history = _history()
    encoder = _linear_encoder()

    with torch.no_grad():
        encoder.projection.weight.fill_(
            float("inf")
        )

    with pytest.raises(ValueError):
        encoder(
            history
        )


def test_mlp_rejects_nonfinite_pre_mask_parameter_output() -> None:
    history = _history()
    encoder = _mlp_encoder()

    with torch.no_grad():
        encoder.hidden_layers[0].weight.fill_(
            float("inf")
        )

    with pytest.raises(ValueError):
        encoder(
            history
        )


@pytest.mark.parametrize(
    "encoder_factory",
    (
        _linear_encoder,
        _mlp_encoder,
    ),
)
def test_trainable_encoder_rejects_dtype_mismatch(
    encoder_factory: Callable[[], nn.Module],
) -> None:
    history = _history(
        dtype=torch.float64
    )
    encoder = encoder_factory()

    with pytest.raises(ValueError):
        encoder(
            history
        )


@pytest.mark.parametrize(
    "encoder_factory",
    (
        _identity_encoder,
        _linear_encoder,
        _mlp_encoder,
    ),
)
def test_float64_execution_is_supported(
    encoder_factory: Callable[[], nn.Module],
) -> None:
    history = _history(
        dtype=torch.float64
    )
    encoder = encoder_factory().double()
    encoder.eval()
    output = encoder(
        history
    )

    assert output.dtype == torch.float64
    assert output.device.type == "cpu"


@pytest.mark.parametrize(
    "encoder_factory",
    (
        _identity_encoder,
        _linear_encoder,
        _mlp_encoder,
    ),
)
def test_variable_sequence_lengths_are_supported(
    encoder_factory: Callable[[], nn.Module],
) -> None:
    encoder = encoder_factory()
    encoder.eval()

    short = _history(
        sequence_length=3
    )
    long = _history(
        sequence_length=6
    )

    assert encoder(
        short
    ).sequence_length == 3
    assert encoder(
        long
    ).sequence_length == 6


# =============================================================================
# Dropout and normalization behavior
# =============================================================================


def test_linear_optional_modules_match_configuration() -> None:
    plain = _linear_encoder()
    regularized = _linear_encoder(
        dropout=0.2,
        layer_normalization=True,
    )

    assert isinstance(
        plain.normalization,
        nn.Identity,
    )
    assert isinstance(
        plain.dropout,
        nn.Identity,
    )
    assert isinstance(
        regularized.normalization,
        nn.LayerNorm,
    )
    assert isinstance(
        regularized.dropout,
        nn.Dropout,
    )
    assert regularized.applies_layer_normalization
    assert regularized.applies_dropout


def test_mlp_optional_modules_match_configuration() -> None:
    plain = _mlp_encoder(
        num_hidden_layers=2
    )
    regularized = _mlp_encoder(
        num_hidden_layers=2,
        dropout=0.2,
        layer_normalization=True,
    )

    assert all(
        isinstance(
            module,
            nn.Identity,
        )
        for module in plain.hidden_normalizations
    )
    assert all(
        isinstance(
            module,
            nn.Identity,
        )
        for module in plain.hidden_dropouts
    )
    assert all(
        isinstance(
            module,
            nn.LayerNorm,
        )
        for module in regularized.hidden_normalizations
    )
    assert all(
        isinstance(
            module,
            nn.Dropout,
        )
        for module in regularized.hidden_dropouts
    )
    assert isinstance(
        regularized.output_normalization,
        nn.LayerNorm,
    )
    assert isinstance(
        regularized.output_dropout,
        nn.Dropout,
    )


@pytest.mark.parametrize(
    "encoder",
    (
        pytest.param(
            _linear_encoder(
                dropout=0.5
            ),
            id="linear",
        ),
        pytest.param(
            _mlp_encoder(
                dropout=0.5
            ),
            id="mlp",
        ),
    ),
)
def test_dropout_is_deterministic_in_evaluation_mode(
    encoder: nn.Module,
) -> None:
    history = _history()
    encoder.eval()

    first = encoder(
        history
    )
    second = encoder(
        history
    )

    assert torch.equal(
        first.encoded_sequence,
        second.encoded_sequence,
    )


@pytest.mark.parametrize(
    "encoder",
    (
        pytest.param(
            _linear_encoder(
                dropout=0.5
            ),
            id="linear",
        ),
        pytest.param(
            _mlp_encoder(
                dropout=0.5
            ),
            id="mlp",
        ),
    ),
)
def test_dropout_keeps_padding_exactly_zero_during_training(
    encoder: nn.Module,
) -> None:
    history = _history()
    encoder.train()
    output = encoder(
        history
    )
    padded = (
        ~history
        .timestep_mask
    ).unsqueeze(-1).expand_as(
        output.encoded_sequence
    )

    assert torch.equal(
        output.encoded_sequence[
            padded
        ],
        torch.zeros_like(
            output.encoded_sequence[
                padded
            ]
        ),
    )


# =============================================================================
# Parameter snapshots
# =============================================================================


@pytest.mark.parametrize(
    "encoder_factory",
    (
        _identity_encoder,
        _linear_encoder,
        _mlp_encoder,
    ),
)
def test_explicit_parameter_snapshot_is_preserved(
    encoder_factory: Callable[[], nn.Module],
) -> None:
    history = _history()
    encoder = encoder_factory()
    snapshot = build_parameter_snapshot_provenance(
        encoder,
        checkpoint_id="checkpoint-v1",
        training_step=10,
    )
    output = encoder(
        history,
        parameter_snapshot=snapshot,
    )

    assert (
        output
        .computation_provenance
        .parameter_snapshot
        is snapshot
    )
    assert output.parameter_snapshot_fingerprint == (
        snapshot.parameter_snapshot_fingerprint
    )


@pytest.mark.parametrize(
    "encoder_factory",
    (
        _identity_encoder,
        _linear_encoder,
        _mlp_encoder,
    ),
)
def test_parameter_snapshot_is_not_generated_implicitly(
    encoder_factory: Callable[[], nn.Module],
) -> None:
    output = encoder_factory()(
        _history()
    )

    assert (
        output
        .computation_provenance
        .parameter_snapshot
        is None
    )
    assert output.parameter_snapshot_fingerprint is None


@pytest.mark.parametrize(
    "encoder_factory",
    (
        _linear_encoder,
        _mlp_encoder,
    ),
)
def test_stale_parameter_snapshot_is_rejected(
    encoder_factory: Callable[[], nn.Module],
) -> None:
    history = _history()
    encoder = encoder_factory()
    snapshot = build_parameter_snapshot_provenance(
        encoder
    )

    first_parameter = next(
        encoder.parameters()
    )
    with torch.no_grad():
        first_parameter.add_(
            1.0
        )

    with pytest.raises(ValueError):
        encoder(
            history,
            parameter_snapshot=snapshot,
        )


def test_identity_rejects_nonempty_parameter_snapshot() -> None:
    history = _history()
    linear = _linear_encoder()
    snapshot = build_parameter_snapshot_provenance(
        linear
    )

    with pytest.raises(ValueError):
        _identity_encoder()(
            history,
            parameter_snapshot=snapshot,
        )


@pytest.mark.parametrize(
    "encoder_factory",
    (
        _linear_encoder,
        _mlp_encoder,
    ),
)
def test_snapshot_with_wrong_reported_count_is_rejected(
    encoder_factory: Callable[[], nn.Module],
) -> None:
    history = _history()
    encoder = encoder_factory()
    correct = build_parameter_snapshot_provenance(
        encoder
    )
    wrong = MemoryParameterSnapshotProvenance(
        parameter_snapshot_fingerprint=(
            correct.parameter_snapshot_fingerprint
        ),
        parameter_count=0,
        trainable_parameter_count=0,
    )

    with pytest.raises(ValueError):
        encoder(
            history,
            parameter_snapshot=wrong,
        )


# =============================================================================
# Equivariance, state restoration, and architecture identity
# =============================================================================


@pytest.mark.parametrize(
    "encoder_factory",
    (
        _identity_encoder,
        _linear_encoder,
        _mlp_encoder,
    ),
)
def test_node_permutation_equivariance(
    encoder_factory: Callable[[], nn.Module],
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

    encoder = encoder_factory()
    encoder.eval()
    original = encoder(
        history
    )
    changed = encoder(
        permuted
    )

    assert torch.equal(
        changed.encoded_sequence,
        original
        .encoded_sequence
        .index_select(
            0,
            permutation,
        ),
    )


@pytest.mark.parametrize(
    "encoder_factory",
    (
        _linear_encoder,
        _mlp_encoder,
    ),
)
def test_state_dict_round_trip_preserves_outputs(
    encoder_factory: Callable[[], nn.Module],
) -> None:
    history = _history()
    first = encoder_factory()
    second = encoder_factory()
    second.load_state_dict(
        first.state_dict()
    )
    first.eval()
    second.eval()

    first_output = first(
        history
    )
    second_output = second(
        history
    )

    assert torch.equal(
        first_output.encoded_sequence,
        second_output.encoded_sequence,
    )
    assert (
        first_output.architecture_fingerprint
        == second_output.architecture_fingerprint
    )
    assert (
        first_output.lineage_fingerprint()
        == second_output.lineage_fingerprint()
    )


@pytest.mark.parametrize(
    "encoder_factory",
    (
        _linear_encoder,
        _mlp_encoder,
    ),
)
def test_same_architecture_different_parameters_share_architecture_identity(
    encoder_factory: Callable[[], nn.Module],
) -> None:
    history = _history()
    torch.manual_seed(
        1
    )
    first = encoder_factory()
    torch.manual_seed(
        2
    )
    second = encoder_factory()
    first.eval()
    second.eval()

    first_output = first(
        history
    )
    second_output = second(
        history
    )

    assert (
        first_output.architecture_fingerprint
        == second_output.architecture_fingerprint
    )
    assert (
        first_output.value_fingerprint()
        != second_output.value_fingerprint()
    )


def test_encoder_families_have_distinct_architecture_identity() -> None:
    history = _history()
    identity = _identity_encoder()(
        history
    )
    linear = _linear_encoder(
        output_dim=D
    )(
        history
    )
    mlp = _mlp_encoder(
        output_dim=D
    )(
        history
    )

    fingerprints = {
        identity.architecture_fingerprint,
        linear.architecture_fingerprint,
        mlp.architecture_fingerprint,
    }

    assert len(
        fingerprints
    ) == 3


def test_module_configuration_is_frozen() -> None:
    encoder = _linear_encoder()

    with pytest.raises(
        (
            FrozenInstanceError,
            AttributeError,
        )
    ):
        encoder.config.output_dim = 99  # type: ignore[misc]


# =============================================================================
# Conditional CUDA boundary
# =============================================================================


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is not available.",
)
@pytest.mark.parametrize(
    "encoder_factory",
    (
        _identity_encoder,
        _linear_encoder,
        _mlp_encoder,
    ),
)
def test_encoders_execute_on_cuda(
    encoder_factory: Callable[[], nn.Module],
) -> None:
    history = _history(
        device="cuda"
    )
    encoder = encoder_factory().to(
        "cuda"
    )
    encoder.eval()
    output = encoder(
        history
    )

    assert output.device.type == "cuda"
    assert output.source_history is history
