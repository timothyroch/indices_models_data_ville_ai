"""
Shared pointwise MLP sequence encoder.

Target path:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            memory/
                baseline_encoders/
                    pointwise_mlp_sequence_encoder.py

This baseline applies the same multilayer perceptron independently at every
temporal position:

    history[n, t, :] -> shared MLP -> encoded[n, t, :]

The module never mixes values across the temporal axis. It is therefore the
nonlinear capacity control for determining whether gains from GRU, LSTM, or
Transformer encoders require temporal interaction rather than only richer
feature transformation.

Architecture
------------
``num_hidden_layers`` is the number of hidden affine layers. For each hidden
layer, the computation is:

    Linear
    optional LayerNorm
    configured activation
    optional Dropout

A final linear projection maps the hidden width to ``output_dim``. To keep the
shared baseline configuration semantics consistent with the linear-projection
baseline, optional output LayerNorm and output Dropout are then applied.

The complete pointwise network is shared across every node and timestep.
Applying PyTorch ``Linear``, ``LayerNorm``, activation, and dropout modules to
``[N, T, D]`` affects only the last dimension and does not aggregate across
``T``.

Missingness and temporal-coordinate policy
------------------------------------------
The encoder consumes only the finite model-ready ``history`` tensor.
``feature_observed_mask`` and temporal coordinates remain attached to the
exact source-history object for diagnostics and lineage, but are not supplied
as additional MLP inputs.

Padding and numerical policy
----------------------------
The complete pre-mask MLP output is checked for finiteness. Padded temporal
positions are then replaced with exact zeros using ``torch.where``.

The finiteness check occurs before masking so numerical failures at padded
positions cannot be hidden. Multiplication by a Boolean mask is not used
because ``0 * NaN`` remains ``NaN``.

Parameter snapshots
-------------------
Parameter snapshots are optional and explicit. They are never generated
automatically during ``forward``. When supplied, a snapshot must identify the
current parameters of this exact module.
"""

from __future__ import annotations

from typing import Final

import torch
from torch import nn

from ..config import (
    BaselineSequenceEncoderConfig,
    BaselineSequenceEncoderKind,
    MemoryActivation,
)
from ..schemas.history_inputs import (
    HistoricalSequenceInputs,
)
from ..schemas.provenance import (
    MemoryParameterSnapshotProvenance,
)
from ..schemas.sequence_encoding import (
    TemporalSequenceEncoderKind,
    TemporalSequenceEncoding,
)
from ._provenance import (
    BASELINE_PROVENANCE_IMPLEMENTATION_VERSION,
    baseline_configuration_fingerprint,
    build_sequence_encoder_computation_provenance,
    count_module_parameters,
    module_parameter_snapshot_fingerprint,
)


# =============================================================================
# Component identity
# =============================================================================


POINTWISE_MLP_SEQUENCE_ENCODER_COMPONENT_NAME: Final[str] = (
    "pointwise_mlp_sequence_encoder"
)

POINTWISE_MLP_SEQUENCE_ENCODER_COMPONENT_KIND: Final[str] = (
    "temporal_mlp"
)

POINTWISE_MLP_SEQUENCE_ENCODER_OPERATION_NAME: Final[str] = (
    "encode_pointwise_mlp_sequence"
)

POINTWISE_MLP_SEQUENCE_ENCODER_IMPLEMENTATION_VERSION: Final[str] = (
    BASELINE_PROVENANCE_IMPLEMENTATION_VERSION
)

POINTWISE_MLP_SEQUENCE_ENCODER_TEMPORAL_INTERACTION: Final[bool] = False

POINTWISE_MLP_SEQUENCE_ENCODER_FEATURE_OBSERVATION_POLICY: Final[str] = (
    "preserved_for_diagnostics_not_consumed"
)

POINTWISE_MLP_SEQUENCE_ENCODER_TEMPORAL_COORDINATE_POLICY: Final[str] = (
    "preserved_for_lineage_not_consumed"
)

POINTWISE_MLP_SEQUENCE_ENCODER_PADDING_POLICY: Final[str] = (
    "validate_finite_pre_mask_then_exact_zero_with_torch_where"
)

POINTWISE_MLP_SEQUENCE_ENCODER_LAYER_POLICY: Final[str] = (
    "hidden_linear_norm_activation_dropout_then_output_linear_norm_dropout"
)


# =============================================================================
# Activation construction
# =============================================================================


def _build_activation(
    activation: MemoryActivation,
) -> nn.Module:
    if activation == MemoryActivation.RELU:
        return nn.ReLU()

    if activation == MemoryActivation.GELU:
        return nn.GELU()

    if activation == MemoryActivation.SILU:
        return nn.SiLU()

    if activation == MemoryActivation.TANH:
        return nn.Tanh()

    raise ValueError(
        f"Unsupported pointwise MLP activation {activation!r}."
    )


# =============================================================================
# Validation helpers
# =============================================================================


def _validate_pointwise_mlp_config(
    config: BaselineSequenceEncoderConfig,
) -> None:
    if not isinstance(
        config,
        BaselineSequenceEncoderConfig,
    ):
        raise TypeError(
            "config must be a BaselineSequenceEncoderConfig."
        )

    if config.kind != BaselineSequenceEncoderKind.TEMPORAL_MLP:
        raise ValueError(
            "PointwiseMLPSequenceEncoder requires "
            "config.kind='temporal_mlp'."
        )

    if config.hidden_dim is None:
        raise ValueError(
            "PointwiseMLPSequenceEncoder requires hidden_dim."
        )

    if config.num_hidden_layers <= 0:
        raise ValueError(
            "PointwiseMLPSequenceEncoder requires at least one hidden "
            "layer."
        )


def _validate_source_history(
    source_history: HistoricalSequenceInputs,
    *,
    expected_feature_dim: int,
) -> None:
    if not isinstance(
        source_history,
        HistoricalSequenceInputs,
    ):
        raise TypeError(
            "source_history must be a HistoricalSequenceInputs."
        )

    if source_history.feature_dim != expected_feature_dim:
        raise ValueError(
            "source_history feature dimension does not match the "
            "pointwise MLP configuration: "
            f"expected {expected_feature_dim}, observed "
            f"{source_history.feature_dim}."
        )

    if not bool(
        torch.isfinite(
            source_history.history
        ).all().item()
    ):
        raise ValueError(
            "source_history.history must remain finite at execution "
            "time."
        )


def _first_parameter(
    module: nn.Module,
) -> nn.Parameter:
    try:
        return next(
            module.parameters()
        )
    except StopIteration as error:
        raise RuntimeError(
            "PointwiseMLPSequenceEncoder unexpectedly has no "
            "parameters."
        ) from error


def _validate_module_input_compatibility(
    module: nn.Module,
    source_history: HistoricalSequenceInputs,
) -> None:
    parameter = _first_parameter(
        module
    )

    if parameter.device != source_history.device:
        raise ValueError(
            "PointwiseMLPSequenceEncoder parameters and "
            "source_history must share one device. Move the module or "
            "history explicitly before encoding."
        )

    if parameter.dtype != source_history.dtype:
        raise ValueError(
            "PointwiseMLPSequenceEncoder parameters and "
            "source_history must use the same floating dtype. Move "
            "the module or history explicitly before encoding."
        )


def _validate_parameter_snapshot(
    module: nn.Module,
    parameter_snapshot: (
        MemoryParameterSnapshotProvenance
        | None
    ),
) -> None:
    if parameter_snapshot is None:
        return

    if not isinstance(
        parameter_snapshot,
        MemoryParameterSnapshotProvenance,
    ):
        raise TypeError(
            "parameter_snapshot must be a "
            "MemoryParameterSnapshotProvenance or None."
        )

    parameter_count, trainable_count = (
        count_module_parameters(
            module
        )
    )

    if (
        parameter_snapshot.parameter_count
        is not None
        and parameter_snapshot.parameter_count
        != parameter_count
    ):
        raise ValueError(
            "parameter_snapshot.parameter_count does not match this "
            "PointwiseMLPSequenceEncoder."
        )

    if (
        parameter_snapshot.trainable_parameter_count
        is not None
        and parameter_snapshot.trainable_parameter_count
        != trainable_count
    ):
        raise ValueError(
            "parameter_snapshot.trainable_parameter_count does not "
            "match this PointwiseMLPSequenceEncoder."
        )

    expected = module_parameter_snapshot_fingerprint(
        module
    )

    if (
        parameter_snapshot
        .parameter_snapshot_fingerprint
        != expected
    ):
        raise ValueError(
            "parameter_snapshot does not identify the current "
            "PointwiseMLPSequenceEncoder parameters."
        )


def _require_finite_pre_mask_output(
    value: torch.Tensor,
) -> None:
    finite = torch.isfinite(
        value
    )

    if bool(
        finite.all().item()
    ):
        return

    nonfinite_count = int(
        (
            ~finite
        )
        .sum()
        .item()
    )
    raise ValueError(
        "PointwiseMLPSequenceEncoder produced nonfinite pre-mask "
        f"values. Nonfinite count: {nonfinite_count}."
    )


def _exact_zero_padding(
    value: torch.Tensor,
    timestep_mask: torch.Tensor,
) -> torch.Tensor:
    return torch.where(
        timestep_mask.unsqueeze(
            -1
        ),
        value,
        torch.zeros_like(
            value
        ),
    )


# =============================================================================
# Pointwise MLP sequence encoder
# =============================================================================


class PointwiseMLPSequenceEncoder(nn.Module):
    """
    Shared nonlinear per-timestep encoder with no temporal interaction.

    Parameters
    ----------
    config:
        Baseline sequence configuration with ``kind='temporal_mlp'``.

    Notes
    -----
    All submodules act on the final feature dimension. The implementation does
    not flatten or aggregate the temporal axis.
    """

    config: BaselineSequenceEncoderConfig
    hidden_layers: nn.ModuleList
    hidden_normalizations: nn.ModuleList
    hidden_activations: nn.ModuleList
    hidden_dropouts: nn.ModuleList
    output_projection: nn.Linear
    output_normalization: nn.Module
    output_dropout: nn.Module

    def __init__(
        self,
        config: BaselineSequenceEncoderConfig,
    ) -> None:
        super().__init__()

        _validate_pointwise_mlp_config(
            config
        )
        assert config.hidden_dim is not None

        self.config = config

        hidden_layers: list[
            nn.Module
        ] = []
        hidden_normalizations: list[
            nn.Module
        ] = []
        hidden_activations: list[
            nn.Module
        ] = []
        hidden_dropouts: list[
            nn.Module
        ] = []

        current_dim = config.input_dim

        for _ in range(
            config.num_hidden_layers
        ):
            hidden_layers.append(
                nn.Linear(
                    in_features=current_dim,
                    out_features=config.hidden_dim,
                    bias=config.use_bias,
                )
            )
            hidden_normalizations.append(
                nn.LayerNorm(
                    config.hidden_dim
                )
                if config.layer_normalization
                else nn.Identity()
            )
            hidden_activations.append(
                _build_activation(
                    config.activation
                )
            )
            hidden_dropouts.append(
                nn.Dropout(
                    p=config.dropout
                )
                if config.dropout > 0.0
                else nn.Identity()
            )
            current_dim = config.hidden_dim

        self.hidden_layers = nn.ModuleList(
            hidden_layers
        )
        self.hidden_normalizations = nn.ModuleList(
            hidden_normalizations
        )
        self.hidden_activations = nn.ModuleList(
            hidden_activations
        )
        self.hidden_dropouts = nn.ModuleList(
            hidden_dropouts
        )

        self.output_projection = nn.Linear(
            in_features=current_dim,
            out_features=config.output_dim,
            bias=config.use_bias,
        )

        self.output_normalization = (
            nn.LayerNorm(
                config.output_dim
            )
            if config.layer_normalization
            else nn.Identity()
        )

        self.output_dropout = (
            nn.Dropout(
                p=config.dropout
            )
            if config.dropout > 0.0
            else nn.Identity()
        )

        self._configuration_fingerprint = (
            baseline_configuration_fingerprint(
                config
            )
        )

    # -------------------------------------------------------------------------
    # Structural properties
    # -------------------------------------------------------------------------

    @property
    def input_dim(
        self,
    ) -> int:
        return self.config.input_dim

    @property
    def hidden_dim(
        self,
    ) -> int:
        assert self.config.hidden_dim is not None
        return self.config.hidden_dim

    @property
    def output_dim(
        self,
    ) -> int:
        return self.config.output_dim

    @property
    def num_hidden_layers(
        self,
    ) -> int:
        return self.config.num_hidden_layers

    @property
    def encoder_kind(
        self,
    ) -> TemporalSequenceEncoderKind:
        return TemporalSequenceEncoderKind.TEMPORAL_MLP

    @property
    def configuration_fingerprint(
        self,
    ) -> str:
        return self._configuration_fingerprint

    @property
    def parameter_count(
        self,
    ) -> int:
        count, _ = count_module_parameters(
            self
        )
        return count

    @property
    def trainable_parameter_count(
        self,
    ) -> int:
        _, count = count_module_parameters(
            self
        )
        return count

    @property
    def has_temporal_interaction(
        self,
    ) -> bool:
        return (
            POINTWISE_MLP_SEQUENCE_ENCODER_TEMPORAL_INTERACTION
        )

    @property
    def applies_layer_normalization(
        self,
    ) -> bool:
        return self.config.layer_normalization

    @property
    def applies_dropout(
        self,
    ) -> bool:
        return self.config.dropout > 0.0

    # -------------------------------------------------------------------------
    # Pointwise computation
    # -------------------------------------------------------------------------

    def _apply_pointwise_network(
        self,
        history: torch.Tensor,
    ) -> torch.Tensor:
        value = history

        for (
            linear,
            normalization,
            activation,
            dropout,
        ) in zip(
            self.hidden_layers,
            self.hidden_normalizations,
            self.hidden_activations,
            self.hidden_dropouts,
            strict=True,
        ):
            value = linear(
                value
            )
            value = normalization(
                value
            )
            value = activation(
                value
            )
            value = dropout(
                value
            )

        value = self.output_projection(
            value
        )
        value = self.output_normalization(
            value
        )
        value = self.output_dropout(
            value
        )

        return value

    # -------------------------------------------------------------------------
    # Provenance semantics
    # -------------------------------------------------------------------------

    def architecture_metadata(
        self,
    ) -> dict[str, object]:
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "num_hidden_layers": (
                self.num_hidden_layers
            ),
            "activation": (
                self.config.activation.value
            ),
            "use_bias": self.config.use_bias,
            "layer_normalization": (
                self.config.layer_normalization
            ),
            "dropout": self.config.dropout,
            "layer_policy": (
                POINTWISE_MLP_SEQUENCE_ENCODER_LAYER_POLICY
            ),
            "feature_projection": True,
            "feature_mixing": True,
            "nonlinear_feature_transformation": True,
            "temporal_interaction": False,
            "temporal_pooling": False,
            "shared_parameters_across_timesteps": True,
            "fixed_sequence_length_required": False,
            "padding_policy": (
                POINTWISE_MLP_SEQUENCE_ENCODER_PADDING_POLICY
            ),
            "feature_observation_policy": (
                POINTWISE_MLP_SEQUENCE_ENCODER_FEATURE_OBSERVATION_POLICY
            ),
            "temporal_coordinate_policy": (
                POINTWISE_MLP_SEQUENCE_ENCODER_TEMPORAL_COORDINATE_POLICY
            ),
            "parameter_count": self.parameter_count,
            "trainable_parameter_count": (
                self.trainable_parameter_count
            ),
            "implementation_version": (
                POINTWISE_MLP_SEQUENCE_ENCODER_IMPLEMENTATION_VERSION
            ),
        }

    # -------------------------------------------------------------------------
    # Forward
    # -------------------------------------------------------------------------

    def forward(
        self,
        source_history: HistoricalSequenceInputs,
        *,
        parameter_snapshot: (
            MemoryParameterSnapshotProvenance
            | None
        ) = None,
    ) -> TemporalSequenceEncoding:
        """
        Encode each timestep independently and preserve ``[N, T, H]``.

        The complete pre-mask output is checked for finiteness before padded
        positions are replaced with exact zeros.
        """

        _validate_source_history(
            source_history,
            expected_feature_dim=self.input_dim,
        )
        _validate_module_input_compatibility(
            self,
            source_history,
        )
        _validate_parameter_snapshot(
            self,
            parameter_snapshot,
        )

        pre_mask_output = (
            self._apply_pointwise_network(
                source_history.history
            )
        )

        _require_finite_pre_mask_output(
            pre_mask_output
        )

        encoded_sequence = _exact_zero_padding(
            pre_mask_output,
            source_history.timestep_mask,
        )

        computation_provenance = (
            build_sequence_encoder_computation_provenance(
                source_history=source_history,
                operation_name=(
                    POINTWISE_MLP_SEQUENCE_ENCODER_OPERATION_NAME
                ),
                component_name=(
                    POINTWISE_MLP_SEQUENCE_ENCODER_COMPONENT_NAME
                ),
                component_kind=(
                    POINTWISE_MLP_SEQUENCE_ENCODER_COMPONENT_KIND
                ),
                architecture_metadata=(
                    self.architecture_metadata()
                ),
                configuration_fingerprint=(
                    self.configuration_fingerprint
                ),
                parameter_snapshot=(
                    parameter_snapshot
                ),
                lineage_metadata={
                    "feature_observed_mask_consumed": False,
                    "temporal_coordinates_consumed": False,
                    "temporal_order_preserved": True,
                    "temporal_interaction": False,
                    "pre_mask_finiteness_validated": True,
                    "exact_zero_padding_applied": True,
                    "flattened_temporal_window": False,
                },
                implementation_version=(
                    POINTWISE_MLP_SEQUENCE_ENCODER_IMPLEMENTATION_VERSION
                ),
            )
        )

        return TemporalSequenceEncoding(
            encoded_sequence=encoded_sequence,
            source_history=source_history,
            encoder_kind=self.encoder_kind,
            computation_provenance=(
                computation_provenance
            ),
            encoding_name=(
                "pointwise_mlp_sequence_encoding"
            ),
        )

    # -------------------------------------------------------------------------
    # Representation
    # -------------------------------------------------------------------------

    def extra_repr(
        self,
    ) -> str:
        return (
            f"input_dim={self.input_dim}, "
            f"hidden_dim={self.hidden_dim}, "
            f"output_dim={self.output_dim}, "
            f"num_hidden_layers={self.num_hidden_layers}, "
            f"activation={self.config.activation.value}, "
            f"bias={self.config.use_bias}, "
            f"layer_normalization={self.config.layer_normalization}, "
            f"dropout={self.config.dropout}, "
            "temporal_interaction=False"
        )


# =============================================================================
# Compact aliases
# =============================================================================


PerTimestepMLPSequenceEncoder = PointwiseMLPSequenceEncoder
TemporalMLPSequenceEncoder = PointwiseMLPSequenceEncoder


# =============================================================================
# Public API
# =============================================================================


__all__ = (
    "POINTWISE_MLP_SEQUENCE_ENCODER_COMPONENT_NAME",
    "POINTWISE_MLP_SEQUENCE_ENCODER_COMPONENT_KIND",
    "POINTWISE_MLP_SEQUENCE_ENCODER_OPERATION_NAME",
    "POINTWISE_MLP_SEQUENCE_ENCODER_IMPLEMENTATION_VERSION",
    "POINTWISE_MLP_SEQUENCE_ENCODER_TEMPORAL_INTERACTION",
    "POINTWISE_MLP_SEQUENCE_ENCODER_FEATURE_OBSERVATION_POLICY",
    "POINTWISE_MLP_SEQUENCE_ENCODER_TEMPORAL_COORDINATE_POLICY",
    "POINTWISE_MLP_SEQUENCE_ENCODER_PADDING_POLICY",
    "POINTWISE_MLP_SEQUENCE_ENCODER_LAYER_POLICY",
    "PointwiseMLPSequenceEncoder",
    "PerTimestepMLPSequenceEncoder",
    "TemporalMLPSequenceEncoder",
)
