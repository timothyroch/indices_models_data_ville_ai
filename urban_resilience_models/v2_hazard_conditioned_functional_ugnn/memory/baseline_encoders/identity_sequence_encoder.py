"""
Parameter-free identity sequence encoder.

Target path:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            memory/
                baseline_encoders/
                    identity_sequence_encoder.py

The identity baseline is the most transparent sequence-preserving lower bound:

    [N, T, D] -> [N, T, D]

It performs:

- no feature projection;
- no feature mixing;
- no temporal interaction;
- no temporal-coordinate transformation;
- no missingness-mask concatenation;
- no pooling;
- no trainable operation.

The output ``TemporalSequenceEncoding`` retains:

- the exact ``HistoricalSequenceInputs`` object;
- the exact ``history`` tensor as ``encoded_sequence``;
- the complete node, feature, temporal, and source-data lineage;
- an explicit architecture identity showing that no temporal interaction
  occurred.

This component is not the primary width-matched control when ``D != H``.
The width-matched no-temporal-interaction control is the separate
``LinearProjectionSequenceEncoder``.

Input policy
------------
The encoder consumes only the model-ready ``history`` tensor. The optional
``feature_observed_mask`` and temporal-coordinate values remain attached to
the source history for diagnostics and downstream provenance, but they are not
concatenated into the representation.

Padding policy
--------------
The Phase 4 input schema requires finite model-facing values and exact zero
padding. Because the identity encoder returns the exact source tensor, it
revalidates those two properties at execution time. This protects against
in-place tensor mutation after frozen-schema construction.

Parameter snapshots
-------------------
The module has no parameters. An optional parameter snapshot may be attached
only when explicitly supplied by the caller. It must identify this exact
empty-parameter module and must report zero total and trainable parameters.
Snapshots are never generated automatically during ``forward``.
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
    module_parameter_snapshot_fingerprint,
)


# =============================================================================
# Component identity
# =============================================================================


IDENTITY_SEQUENCE_ENCODER_COMPONENT_NAME: Final[str] = (
    "identity_sequence_encoder"
)

IDENTITY_SEQUENCE_ENCODER_COMPONENT_KIND: Final[str] = "identity"

IDENTITY_SEQUENCE_ENCODER_OPERATION_NAME: Final[str] = (
    "encode_identity_sequence"
)

IDENTITY_SEQUENCE_ENCODER_IMPLEMENTATION_VERSION: Final[str] = (
    BASELINE_PROVENANCE_IMPLEMENTATION_VERSION
)

IDENTITY_SEQUENCE_ENCODER_TEMPORAL_INTERACTION: Final[bool] = False

IDENTITY_SEQUENCE_ENCODER_FEATURE_OBSERVATION_POLICY: Final[str] = (
    "preserved_for_diagnostics_not_consumed"
)

IDENTITY_SEQUENCE_ENCODER_TEMPORAL_COORDINATE_POLICY: Final[str] = (
    "preserved_for_lineage_not_consumed"
)

IDENTITY_SEQUENCE_ENCODER_PADDING_POLICY: Final[str] = (
    "reuse_exact_canonical_zero_padded_source_tensor"
)


# =============================================================================
# Validation
# =============================================================================


def _validate_identity_config(
    config: BaselineSequenceEncoderConfig,
) -> None:
    if not isinstance(
        config,
        BaselineSequenceEncoderConfig,
    ):
        raise TypeError(
            "config must be a BaselineSequenceEncoderConfig."
        )

    if config.kind != BaselineSequenceEncoderKind.IDENTITY:
        raise ValueError(
            "IdentitySequenceEncoder requires "
            "config.kind='identity'."
        )

    if config.input_dim != config.output_dim:
        raise ValueError(
            "IdentitySequenceEncoder requires input_dim == output_dim."
        )

    if config.hidden_dim is not None:
        raise ValueError(
            "IdentitySequenceEncoder requires hidden_dim=None."
        )

    if config.num_hidden_layers != 1:
        raise ValueError(
            "IdentitySequenceEncoder requires num_hidden_layers=1."
        )

    if config.dropout != 0.0:
        raise ValueError(
            "IdentitySequenceEncoder requires dropout=0."
        )

    if config.layer_normalization:
        raise ValueError(
            "IdentitySequenceEncoder cannot apply layer normalization "
            "because that would no longer be an identity mapping."
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
            "identity encoder configuration: "
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

    padding_mask = (
        ~source_history
        .timestep_mask
    ).unsqueeze(
        -1
    ).expand_as(
        source_history.history
    )

    padded_values = source_history.history[
        padding_mask
    ]

    if (
        padded_values.numel() > 0
        and not torch.equal(
            padded_values,
            torch.zeros_like(
                padded_values
            ),
        )
    ):
        raise ValueError(
            "source_history.history must retain exact zero padding at "
            "execution time."
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

    if (
        parameter_snapshot.parameter_count
        not in (
            None,
            0,
        )
    ):
        raise ValueError(
            "IdentitySequenceEncoder parameter snapshots must report "
            "parameter_count=0 when the count is supplied."
        )

    if (
        parameter_snapshot.trainable_parameter_count
        not in (
            None,
            0,
        )
    ):
        raise ValueError(
            "IdentitySequenceEncoder parameter snapshots must report "
            "trainable_parameter_count=0 when the count is supplied."
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
            "parameter_snapshot does not identify this exact "
            "IdentitySequenceEncoder parameter state."
        )


# =============================================================================
# Identity sequence encoder
# =============================================================================


class IdentitySequenceEncoder(nn.Module):
    """
    Return the exact model-ready historical tensor as a sequence encoding.

    Parameters
    ----------
    config:
        Validated baseline configuration with ``kind='identity'``.

    Notes
    -----
    ``activation`` and ``use_bias`` are generic fields on the shared baseline
    configuration but are not operational for a parameter-free identity map.
    They remain part of the immutable configuration fingerprint so experiment
    artifacts preserve the exact supplied configuration.
    """

    config: BaselineSequenceEncoderConfig

    def __init__(
        self,
        config: BaselineSequenceEncoderConfig,
    ) -> None:
        super().__init__()

        _validate_identity_config(
            config
        )

        self.config = config
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
        return 0

    @property
    def trainable_parameter_count(
        self,
    ) -> int:
        return 0

    @property
    def has_temporal_interaction(
        self,
    ) -> bool:
        return IDENTITY_SEQUENCE_ENCODER_TEMPORAL_INTERACTION

    # -------------------------------------------------------------------------
    # Provenance semantics
    # -------------------------------------------------------------------------

    def architecture_metadata(
        self,
    ) -> dict[str, object]:
        return {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "trainable": False,
            "parameter_count": 0,
            "trainable_parameter_count": 0,
            "feature_projection": False,
            "feature_mixing": False,
            "temporal_interaction": False,
            "temporal_pooling": False,
            "output_aliases_source_history_tensor": True,
            "padding_policy": (
                IDENTITY_SEQUENCE_ENCODER_PADDING_POLICY
            ),
            "feature_observation_policy": (
                IDENTITY_SEQUENCE_ENCODER_FEATURE_OBSERVATION_POLICY
            ),
            "temporal_coordinate_policy": (
                IDENTITY_SEQUENCE_ENCODER_TEMPORAL_COORDINATE_POLICY
            ),
            "configured_activation_not_applied": (
                self.config.activation.value
            ),
            "configured_use_bias_not_applied": (
                self.config.use_bias
            ),
            "implementation_version": (
                IDENTITY_SEQUENCE_ENCODER_IMPLEMENTATION_VERSION
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
        Encode one historical batch without copying or transforming values.

        The returned ``encoded_sequence`` is the exact
        ``source_history.history`` tensor object.
        """

        _validate_source_history(
            source_history,
            expected_feature_dim=self.input_dim,
        )
        _validate_parameter_snapshot(
            self,
            parameter_snapshot,
        )

        computation_provenance = (
            build_sequence_encoder_computation_provenance(
                source_history=source_history,
                operation_name=(
                    IDENTITY_SEQUENCE_ENCODER_OPERATION_NAME
                ),
                component_name=(
                    IDENTITY_SEQUENCE_ENCODER_COMPONENT_NAME
                ),
                component_kind=(
                    IDENTITY_SEQUENCE_ENCODER_COMPONENT_KIND
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
                    "source_tensor_identity_preserved": True,
                    "feature_observed_mask_consumed": False,
                    "temporal_coordinates_consumed": False,
                    "temporal_order_preserved": True,
                },
                implementation_version=(
                    IDENTITY_SEQUENCE_ENCODER_IMPLEMENTATION_VERSION
                ),
            )
        )

        return TemporalSequenceEncoding(
            encoded_sequence=(
                source_history.history
            ),
            source_history=source_history,
            encoder_kind=self.encoder_kind,
            computation_provenance=(
                computation_provenance
            ),
            encoding_name=(
                "identity_sequence_encoding"
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
            "parameters=0, "
            "temporal_interaction=False"
        )


# =============================================================================
# Compact alias
# =============================================================================


IdentityTemporalSequenceEncoder = IdentitySequenceEncoder


# =============================================================================
# Public API
# =============================================================================


__all__ = (
    "IDENTITY_SEQUENCE_ENCODER_COMPONENT_NAME",
    "IDENTITY_SEQUENCE_ENCODER_COMPONENT_KIND",
    "IDENTITY_SEQUENCE_ENCODER_OPERATION_NAME",
    "IDENTITY_SEQUENCE_ENCODER_IMPLEMENTATION_VERSION",
    "IDENTITY_SEQUENCE_ENCODER_TEMPORAL_INTERACTION",
    "IDENTITY_SEQUENCE_ENCODER_FEATURE_OBSERVATION_POLICY",
    "IDENTITY_SEQUENCE_ENCODER_TEMPORAL_COORDINATE_POLICY",
    "IDENTITY_SEQUENCE_ENCODER_PADDING_POLICY",
    "IdentitySequenceEncoder",
    "IdentityTemporalSequenceEncoder",
)
