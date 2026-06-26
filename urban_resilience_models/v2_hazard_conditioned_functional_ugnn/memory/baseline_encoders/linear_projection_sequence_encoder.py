"""
Shared per-timestep linear projection sequence encoder.

Target path:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            memory/
                baseline_encoders/
                    linear_projection_sequence_encoder.py

This baseline applies the same learned feature projection independently at
every temporal position:

    history[n, t, :] -> Linear(D, H) -> encoded[n, t, :]

There is no interaction across the temporal axis. Changing one timestep cannot
alter another timestep's representation, except through external stochastic
state such as a caller-requested dropout configuration. With the default
``dropout=0``, the component is the primary width-matched control against GRU,
LSTM, and Transformer sequence encoders.

Optional post-projection behavior
---------------------------------
The shared baseline configuration permits:

- optional layer normalization over the projected feature axis;
- optional dropout;
- configurable bias in the linear projection.

The generic ``activation`` field is deliberately not applied. Applying it
would make the component a nonlinear pointwise encoder and overlap with the
separate pointwise temporal MLP baseline.

The effective computation is:

    projected = Linear(history)
    projected = LayerNorm(projected)     # optional
    projected = Dropout(projected)       # optional
    encoded = exact-zero-mask(projected)

None of these operations mixes information across ``T``.

Missingness and temporal-coordinate policy
------------------------------------------
The encoder consumes only the model-ready ``history`` tensor. The optional
``feature_observed_mask`` and temporal-coordinate values remain attached to
the exact source-history object for diagnostics and provenance. They are not
concatenated to the representation.

Padding and numerical policy
----------------------------
The full pre-mask output is checked for finiteness. Padded positions are then
replaced with exact zeros using ``torch.where``. Multiplication is not used
because ``0 * NaN`` remains ``NaN``.

Parameter snapshots
-------------------
Parameter snapshots are optional and explicit. They are never generated
automatically during ``forward``. When supplied, the snapshot must identify
the current parameters of this exact module.
"""

from __future__ import annotations

from typing import Final

import torch
from torch import nn

from ..config import (
    BaselineSequenceEncoderConfig,
    BaselineSequenceEncoderKind,
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


LINEAR_PROJECTION_SEQUENCE_ENCODER_COMPONENT_NAME: Final[str] = (
    "linear_projection_sequence_encoder"
)

LINEAR_PROJECTION_SEQUENCE_ENCODER_COMPONENT_KIND: Final[str] = (
    "linear_projection"
)

LINEAR_PROJECTION_SEQUENCE_ENCODER_OPERATION_NAME: Final[str] = (
    "encode_linear_projection_sequence"
)

LINEAR_PROJECTION_SEQUENCE_ENCODER_IMPLEMENTATION_VERSION: Final[str] = (
    BASELINE_PROVENANCE_IMPLEMENTATION_VERSION
)

LINEAR_PROJECTION_SEQUENCE_ENCODER_TEMPORAL_INTERACTION: Final[bool] = False

LINEAR_PROJECTION_SEQUENCE_ENCODER_FEATURE_OBSERVATION_POLICY: Final[str] = (
    "preserved_for_diagnostics_not_consumed"
)

LINEAR_PROJECTION_SEQUENCE_ENCODER_TEMPORAL_COORDINATE_POLICY: Final[str] = (
    "preserved_for_lineage_not_consumed"
)

LINEAR_PROJECTION_SEQUENCE_ENCODER_PADDING_POLICY: Final[str] = (
    "validate_finite_pre_mask_then_exact_zero_with_torch_where"
)

LINEAR_PROJECTION_SEQUENCE_ENCODER_ACTIVATION_POLICY: Final[str] = (
    "configured_activation_recorded_but_not_applied"
)


# =============================================================================
# Validation helpers
# =============================================================================


def _validate_linear_projection_config(
    config: BaselineSequenceEncoderConfig,
) -> None:
    if not isinstance(
        config,
        BaselineSequenceEncoderConfig,
    ):
        raise TypeError(
            "config must be a BaselineSequenceEncoderConfig."
        )

    if (
        config.kind
        != BaselineSequenceEncoderKind.LINEAR_PROJECTION
    ):
        raise ValueError(
            "LinearProjectionSequenceEncoder requires "
            "config.kind='linear_projection'."
        )

    if config.hidden_dim is not None:
        raise ValueError(
            "LinearProjectionSequenceEncoder requires hidden_dim=None."
        )

    if config.num_hidden_layers != 1:
        raise ValueError(
            "LinearProjectionSequenceEncoder requires "
            "num_hidden_layers=1."
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
            "linear projection configuration: "
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
            "LinearProjectionSequenceEncoder unexpectedly has no "
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
            "LinearProjectionSequenceEncoder parameters and "
            "source_history must share one device. Move the module or "
            "history explicitly before encoding."
        )

    if parameter.dtype != source_history.dtype:
        raise ValueError(
            "LinearProjectionSequenceEncoder parameters and "
            "source_history must use the same floating dtype. Move the "
            "module or history explicitly before encoding."
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
            "LinearProjectionSequenceEncoder."
        )

    if (
        parameter_snapshot.trainable_parameter_count
        is not None
        and parameter_snapshot.trainable_parameter_count
        != trainable_count
    ):
        raise ValueError(
            "parameter_snapshot.trainable_parameter_count does not "
            "match this LinearProjectionSequenceEncoder."
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
            "LinearProjectionSequenceEncoder parameters."
        )


def _require_finite_pre_mask_output(
    value: torch.Tensor,
) -> None:
    if not bool(
        torch.isfinite(
            value
        ).all().item()
    ):
        nonfinite_count = int(
            (
                ~torch.isfinite(
                    value
                )
            )
            .sum()
            .item()
        )
        raise ValueError(
            "LinearProjectionSequenceEncoder produced nonfinite "
            "pre-mask values. Nonfinite count: "
            f"{nonfinite_count}."
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
# Linear projection sequence encoder
# =============================================================================


class LinearProjectionSequenceEncoder(nn.Module):
    """
    Shared per-timestep feature projection with no temporal interaction.

    Parameters
    ----------
    config:
        Baseline sequence configuration with
        ``kind='linear_projection'``.

    Notes
    -----
    The same projection, normalization, and dropout modules are reused at every
    ``(n, t)`` position. No reduction or communication occurs over ``T``.
    """

    config: BaselineSequenceEncoderConfig
    projection: nn.Linear
    normalization: nn.Module
    dropout: nn.Module

    def __init__(
        self,
        config: BaselineSequenceEncoderConfig,
    ) -> None:
        super().__init__()

        _validate_linear_projection_config(
            config
        )

        self.config = config

        self.projection = nn.Linear(
            in_features=config.input_dim,
            out_features=config.output_dim,
            bias=config.use_bias,
        )

        self.normalization = (
            nn.LayerNorm(
                config.output_dim
            )
            if config.layer_normalization
            else nn.Identity()
        )

        self.dropout = (
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
    def output_dim(
        self,
    ) -> int:
        return self.config.output_dim

    @property
    def encoder_kind(
        self,
    ) -> TemporalSequenceEncoderKind:
        # Phase 4 uses IDENTITY_SEQUENCE for sequence-preserving simple
        # baselines other than the explicitly nonlinear temporal MLP.
        return TemporalSequenceEncoderKind.IDENTITY_SEQUENCE

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
            LINEAR_PROJECTION_SEQUENCE_ENCODER_TEMPORAL_INTERACTION
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
    # Provenance semantics
    # -------------------------------------------------------------------------

    def architecture_metadata(
        self,
    ) -> dict[str, object]:
        return {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "use_bias": self.config.use_bias,
            "layer_normalization": (
                self.config.layer_normalization
            ),
            "dropout": self.config.dropout,
            "activation_policy": (
                LINEAR_PROJECTION_SEQUENCE_ENCODER_ACTIVATION_POLICY
            ),
            "configured_activation_not_applied": (
                self.config.activation.value
            ),
            "feature_projection": True,
            "feature_mixing": True,
            "pointwise_nonlinear_activation": False,
            "temporal_interaction": False,
            "temporal_pooling": False,
            "shared_parameters_across_timesteps": True,
            "padding_policy": (
                LINEAR_PROJECTION_SEQUENCE_ENCODER_PADDING_POLICY
            ),
            "feature_observation_policy": (
                LINEAR_PROJECTION_SEQUENCE_ENCODER_FEATURE_OBSERVATION_POLICY
            ),
            "temporal_coordinate_policy": (
                LINEAR_PROJECTION_SEQUENCE_ENCODER_TEMPORAL_COORDINATE_POLICY
            ),
            "parameter_count": self.parameter_count,
            "trainable_parameter_count": (
                self.trainable_parameter_count
            ),
            "implementation_version": (
                LINEAR_PROJECTION_SEQUENCE_ENCODER_IMPLEMENTATION_VERSION
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
        Project every timestep independently and preserve ``[N, T, H]``.

        The full pre-mask representation is checked for finiteness before
        padded positions are replaced by exact zeros.
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

        projected = self.projection(
            source_history.history
        )
        projected = self.normalization(
            projected
        )
        projected = self.dropout(
            projected
        )

        _require_finite_pre_mask_output(
            projected
        )

        encoded_sequence = _exact_zero_padding(
            projected,
            source_history.timestep_mask,
        )

        computation_provenance = (
            build_sequence_encoder_computation_provenance(
                source_history=source_history,
                operation_name=(
                    LINEAR_PROJECTION_SEQUENCE_ENCODER_OPERATION_NAME
                ),
                component_name=(
                    LINEAR_PROJECTION_SEQUENCE_ENCODER_COMPONENT_NAME
                ),
                component_kind=(
                    LINEAR_PROJECTION_SEQUENCE_ENCODER_COMPONENT_KIND
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
                },
                implementation_version=(
                    LINEAR_PROJECTION_SEQUENCE_ENCODER_IMPLEMENTATION_VERSION
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
                "linear_projection_sequence_encoding"
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
            f"output_dim={self.output_dim}, "
            f"bias={self.config.use_bias}, "
            f"layer_normalization={self.config.layer_normalization}, "
            f"dropout={self.config.dropout}, "
            "activation=none, "
            "temporal_interaction=False"
        )


# =============================================================================
# Compact alias
# =============================================================================


PerTimestepLinearSequenceEncoder = LinearProjectionSequenceEncoder


# =============================================================================
# Public API
# =============================================================================


__all__ = (
    "LINEAR_PROJECTION_SEQUENCE_ENCODER_COMPONENT_NAME",
    "LINEAR_PROJECTION_SEQUENCE_ENCODER_COMPONENT_KIND",
    "LINEAR_PROJECTION_SEQUENCE_ENCODER_OPERATION_NAME",
    "LINEAR_PROJECTION_SEQUENCE_ENCODER_IMPLEMENTATION_VERSION",
    "LINEAR_PROJECTION_SEQUENCE_ENCODER_TEMPORAL_INTERACTION",
    "LINEAR_PROJECTION_SEQUENCE_ENCODER_FEATURE_OBSERVATION_POLICY",
    "LINEAR_PROJECTION_SEQUENCE_ENCODER_TEMPORAL_COORDINATE_POLICY",
    "LINEAR_PROJECTION_SEQUENCE_ENCODER_PADDING_POLICY",
    "LINEAR_PROJECTION_SEQUENCE_ENCODER_ACTIVATION_POLICY",
    "LinearProjectionSequenceEncoder",
    "PerTimestepLinearSequenceEncoder",
)
